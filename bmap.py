#from __future__ import annotations

"""Numpy Buffer Map
===================

A compact toolkit for laying out multiple NumPy arrays inside one or more
aligned backing buffers using a tree of nodes. The tree encodes *sharing
semantics* (shared vs. distinct regions), alignment policy, offsets, and
friendly labels. The pipeline is:

1) **Reduction**: merge adjacent containers that share the same rule.
2) **Size propagation & offset assignment**: compute `nbytes` per node and
   assign byte offsets (respecting alignment and container flags).
3) **Allocation**: allocate a single aligned top buffer and map `ArrayNode`s
   to views into it.

Core node types:
- **ContainerNode**: internal node that carries a rule: `BufferMap.SHARED`
  (children occupy the same region sized to the max child) or
  `BufferMap.DISTINCT` (children are concatenated, optionally without
  per-child alignment when `no_align=True`).
- **ArrayNode**: leaf that becomes a concrete NumPy array; supports optional
  *aligned leading dimension* (`align_ldim`) producing a padded backing shape
  (`bshape`) while exposing a trimmed logical `shape`.
- **ValueNode**: leaf that participates in traversal/flattening but does not
  consume buffer bytes.

Export helpers:
- **`bmap_todot`**: emit a Graphviz DOT representation using `anytree`.
- **`bmap_pyvis`** (defined later): visualize via PyVis/NetworkX/pydot.

Quickstart
----------
Build a tree with containers and arrays, then run `build_bmap` → `allocate_bmap`,
or call `ContainerNode.build_flatmap()` for the one-shot variant. Flattening
helpers (`flatten_items`, `flat_inits`) return leaves or initialized values in
preorder.
"""

from functools import cache

from typing import Any, List, Sequence, Tuple, Union, Optional
import itertools
import copy
import io
import numpy as np
from anytree import NodeMixin, PreOrderIter, RenderTree, DoubleStyle
from anytree.exporter import DotExporter
from functools import partial


# __all__ = [
#     "BufferMap",
#     "ar_spec",
#     "ArrayNode",
#     "ContainerNode",
#     "sb_node",
#     "db_node",
#     "reduce_bmap",
#     "gen_buffer_specs",
#     "check_bmap",
#     "build_bmap",
#     "allocate_bmap",
#     "bmap_get",
#     "flat_bmap_inits",
#     "bmap_todot",
#     "bmap_pyvis",
#     "save_bmap_tree",
# ]

# ---------------------------------------------------------------------------
# Basic constants
# ---------------------------------------------------------------------------


class BufferMap:
    SHARED: int = 0
    DISTINCT: int = 1


class BufferAlign:
    """Common buffer alignment sizes in bytes."""

    BYTE = 1  # 8‑bit
    WORD = 2  # 16‑bit
    DWORD = 4  # 32‑bit
    QWORD = 8  # 64‑bit
    CACHE_LINE = 64  # Typical L1/L2 cache line size
    SSE = 16  # 128‑bit vector registers (SSE)
    AVX = 32  # 256‑bit vector registers (AVX/AVX2)
    AVX512 = 64  # 512‑bit vector registers (AVX‑512)
    PAGE = 4096  # Typical OS page size


import importlib

if importlib.util.find_spec('numba'):
    from numba.extending import register_jitable
    #fastmath should make no difference for allocating a buffer.
    rgc=register_jitable(cache=True,fastmath=True,error_model='numpy') 
    from numba import typeof
else:
    def typeof(*args):
        return np.dtype(type(args[0]))
    rgc=lambda f:f


class NativeTypes:
    """Sizes of common primitive types across NumPy and Numba ecosystems.

    The attributes ending with ``b`` expose the **byte-size** of the type.
    These are useful when interoperating with BLAS/LAPACK backends that expect
    LP64 vs. ILP64 integer widths, or when constructing aligned buffers and
    computing strides. Values are resolved at import-time from the active
    platform (e.g., pointer width via ``np.intp``) and Numba's typing layer.
    """

    POINTER = np.dtype(np.intp)  # system pointer size
    POINTERb = POINTER.itemsize

    NP_INT = np.dtype(np.int_)
    NP_INTb = NP_INT.itemsize
    NP_FLOAT = np.dtype(np.float_)
    NP_FLOATb = NP_FLOAT.itemsize
    NP_BOOL = np.dtype(np.bool_)
    NP_BOOLb = NP_BOOL.itemsize
    NP_COMPLEX = np.dtype(np.complex_)
    NP_COMPLEXb = NP_COMPLEX.itemsize

    NB_INT = np.dtype(typeof(1, 2).name)
    NB_INTb = NB_INT.itemsize
    NB_FLOAT = np.dtype(typeof(1.0, 2).name)
    NB_FLOATb = NB_FLOAT.itemsize
    NB_BOOL = np.dtype(typeof(True, 2).name)
    NB_BOOLb = NB_BOOL.itemsize
    NB_COMPLEX = np.dtype(typeof(1j, 2).name)
    NB_COMPLEXb = NB_COMPLEX.itemsize

    LP64_I = np.dtype(np.int32)
    LP64_Ib = LP64_I.itemsize
    ILP64_I = np.dtype(np.int64)
    ILP64_Ib = ILP64_I.itemsize


class _IDGen:
    _ctr = itertools.count()

    @classmethod
    def new_id(cls) -> int:
        return next(cls._ctr)


@rgc
def aligned_buffer(n_bytes: int, align: int = BufferAlign.AVX512) -> np.ndarray:
    """Return an aligned ``uint8`` view of length ``n_bytes``.

    A slightly oversized buffer is allocated and then *manually aligned* by
    slicing so that ``result.ctypes.data % align == 0``. The extra capacity is
    not exposed by the returned view.

    Parameters
    ----------
    n_bytes : int
        Logical size of the returned view (in bytes).
    align : int
        Desired byte alignment (power-of-two is assumed).
    """
    raw = np.empty(n_bytes + align, dtype=np.uint8)
    offset = (-raw.ctypes.data) & (align - 1)
    return raw[offset : offset + n_bytes]


class BaseNode(NodeMixin):
    """Base class for all nodes participating in the buffer map.

    Common fields
    -------------
    id : int
        Stable, monotonically increasing identifier for DOT/back-refs.
    label : str | None
        Human-readable label; may be joined by parent during reduction.
    nbytes : int | None
        Size contribution of the subtree or array (computed in ``build_bmap``).
    ofs : int | None
        Start offset within the root buffer (assigned in ``gen_buffer_specs``).
    buffer : np.ndarray | None
        For containers: a slice of the root buffer covering the subtree.
        For arrays: not used directly; see ``ArrayNode.array``.
    """

    def __init__(
        self,
        name: Optional[Union[str, int, float]] = None,
        no_merge: bool = False,
    ) -> None:
        self.id: int = _IDGen.new_id()
        self.label: Optional[str] = str(name) if name is not None else None
        self.no_merge: bool = bool(no_merge)
        self.nbytes: Optional[int] = None
        self.ofs: Optional[int] = None
        self.buffer: Optional[np.ndarray] = None

    def _short(self) -> str:
        return self.label or str(self.id)

    @property
    def name(self):
        return self.label

    @name.setter
    def name(self, value):
        self.label = value

    @name.deleter
    def name(self):
        del self.label


class ValueNode(BaseNode):
    """Leaf node carrying a Python value that does **not** occupy buffer space.

    Useful for threading literal parameters through the tree so that
    ``flatten_items`` and friends preserve order alongside arrays.
    """

    def __init__(
        self,
        value,
        name: Optional[Union[str, int, float]] = None,
    ) -> None:
        super().__init__(name=name, no_merge=True)
        self.value = value

    @property
    def vtype(self):
        return type(self.value)

    def __repr__(self):
        return f"|Value {self._short()} {repr(self.value)[:50]} {self.vtype}|"


class ArrayNode(BaseNode):
    """Leaf node that materializes as a NumPy array.

    Notes
    -----
    - ``shape`` is the logical array shape; ``bshape`` is the *backing* shape
      when an aligned leading dimension is requested via ``align_ldim``.
    - When ``align_ldim`` is provided, the array's last (C-order) or first
      (F-order) dimension is padded to an alignment-friendly multiple in the
      backing buffer, while ``array`` exposes the trimmed logical shape.
    - ``init_op`` can be a scalar, an ndarray (copied/viewed), or a callable
      that receives the newly created array for in-place initialization.
    """

    def __init__(
        self,
        shape: Union[int, Sequence[int]] = (0,),
        dtype: np.dtype = np.float64,
        order: str = "C",
        init_op: Optional[Union[int, float, np.ndarray, callable]] = None,
        name: Optional[Union[str, int, float]] = None,
        align_ldim=None,
    ) -> None:
        super().__init__(name=name, no_merge=True)
        self.shape: Tuple[int, ...]
        self.bshape: Tuple[int, ...]
        self.barray: Optional[np.ndarray]
        self.array: Optional[np.ndarray] = None
        self.align_ldim = align_ldim
        self.dtype: np.dtype = np.dtype(dtype)
        self.order: str = order.upper()
        self.init_op = init_op
        s = (shape,) if isinstance(shape, int) else tuple(int(s) for s in shape)
        if isinstance(align_ldim, tuple):
            self.bshape = align_ldim
            self.shape = s
        elif align_ldim is not None:
            self.shape = s
            rsz = align_ldim // self.dtype.itemsize
            if self.order == "C":
                self.bshape = (*s[:-1], ((s[-1] + rsz - 1) // rsz) * rsz)
            else:
                self.bshape = (((s[0] + rsz - 1) // rsz) * rsz, *s[1:])
        else:
            self.bshape = self.shape = s
        self.nbytes: int = int(np.prod(self.shape)) * self.dtype.itemsize

    def __repr__(self):
        shp = ", ".join(map(str, self.shape))
        if self.shape != self.bshape:
            bshp = ", ".join(map(str, self.bshape))
            return f"|Array {self._short()} ({bshp}) ({shp}) {self.dtype}|"
        return f"|Array {self._short()} ({shp}) {self.dtype}|"

    def _rinit(self):
        """Apply ``init_op`` to ``self.array`` if provided.

        - If ``init_op`` is callable, it is invoked with the array for in-place
          initialization.
        - Otherwise, the array is filled/broadcast-assigned with the value.
        """
        if self.init_op is not None:
            if callable(self.init_op):
                self.init_op(self.array)
            else:
                self.array[...] = self.init_op

    def mk_array(self, buffer: Optional[np.ndarray] = None) -> np.ndarray:
        """Materialize ``self.array`` either standalone or as a view on
        ``buffer``.

        Cases
        -----
        - ``buffer is None`` and no array exists: allocate a fresh ndarray with
          the logical ``shape``/``dtype``/``order``.
        - ``buffer is not None``: create a *view* into the provided root buffer
          starting at ``self.ofs`` with total elements ``prod(bshape)`` and
          reshape to ``bshape``.
        - If ``align_ldim`` is set, store the backing view in ``self.barray`` and
          expose a trimmed logical view in ``self.array`` (slice along the aligned
          dimension).

        ``_rinit`` is invoked at the end to apply ``init_op``.
        """
        if y := (buffer is None and self.array is None):
            self.array = np.empty(self.shape, dtype=self.dtype, order=self.order)
        elif y := (buffer is not None):
            cnt = int(np.prod(self.bshape))
            view = np.frombuffer(
                buffer,
                dtype=self.dtype,
                count=cnt,
                offset=0 if self.ofs is None else self.ofs,
            )
            view = view.reshape(self.bshape, order=self.order)
            self.array = view
        if y and self.align_ldim is not None:
            self.barray = self.array
            self.array = (
                self.barray[..., : self.shape[-1]]
                if self.order == "C"
                else self.barray[: self.shape[0], ...]
            )
        self._rinit()
        return self.array

    @property
    def value(self):
        return self.array


ItemNodeT = ValueNode | ArrayNode


class ContainerNode(BaseNode):
    """Internal node controlling how children share backing memory.

    Rules
    -----
    - ``DISTINCT``: children are concatenated in order; each child's region is
      rounded up to the global alignment unless ``no_align=True``, in which case
      only the *container's* total is rounded once.
    - ``SHARED``: all children map to the same region whose size equals the
      maximum child size, rounded up to alignment.

    Flags
    -----
    no_merge : bool
        Prevents being merged with a like-ruled parent/child during reduction.
    no_align : bool
        Only for ``DISTINCT``: disables per-child rounding so contiguous regions
        can exactly match external memory layouts.
    """

    def __init__(
        self,
        *children: "BaseNode",
        rule: int = BufferMap.DISTINCT,
        name: Optional[Union[str, int, float]] = None,
        no_merge: bool = False,
        no_align: bool = False,
    ) -> None:
        super().__init__(name=name, no_merge=no_merge)
        self.rule: int = int(rule)
        self.no_align = no_align  # only effective for distinct rule nodes.
        self.add(*children)

    def add(self, *kids: "BaseNode") -> "ContainerNode":
        """Append children to this container and set their parent pointer."""
        for k in kids:
            if not isinstance(k, BaseNode):
                print(k)
                raise TypeError("Children must be BaseNode instances")
            k.parent = self
        return self

    def insert(self, index, *kids):
        """Insert one or more children at a specific position.

        Existing instances of the same nodes are first removed to avoid
        duplicates, then the desired order is established.
        Returns ``self`` for chaining.
        """
        for k in kids:
            if not isinstance(k, BaseNode):
                raise TypeError("Children must be BaseNode instances")
        current = list(self.children)
        for k in kids:
            if k in current:
                current.remove(k)
        new_order = current[:index] + list(kids) + current[index:]
        for k in kids:
            k.parent = self
        self.children = tuple(new_order)
        return self

    def order(self, *indices):
        """Reorder existing children using a permutation of positions [0..n-1].

        Raises ``ValueError`` if the indices do not form a valid permutation.
        Returns ``self`` for chaining.
        """
        children = list(self.children)
        n = len(children)
        if len(indices) != n:
            raise ValueError(f"Expected {n} indices, got {len(indices)}")
        if sorted(indices) != list(range(n)):
            raise ValueError(f"Indices must be a permutation of 0..{n - 1}")
        reordered = [children[i] for i in indices]
        self.children = tuple(reordered)
        return self

    def pop(self, *idxs):
        """Remove and return children by index and/or label.

        Indices can be integers (negative supported) or labels (string).
        The returned list preserves the *request* order, not the tree
        order.
        """
        # Copy current children to a mutable list
        children = list(self.children)
        # Use dict to preserve requested order and avoid duplicates
        positions = {}
        for idx in idxs:
            if isinstance(idx, str):
                # Match all children with this label
                for i, child in enumerate(children):
                    if child.label == idx and i not in positions:
                        positions[i] = None
            elif isinstance(idx, int):
                # Support negative indices
                pos = idx if idx >= 0 else len(children) + idx
                if 0 <= pos < len(children) and pos not in positions:
                    positions[pos] = None
            else:
                raise TypeError("Indices must be either int or str")
        # Collect nodes to return in requested order
        popped = [children[i] for i in positions.keys()]
        # Remove from highest index to lowest to avoid shifting
        for pos in sorted(positions.keys(), reverse=True):
            node = children.pop(pos)
            # Detach from this container
            node.parent = None
        # Update children tuple
        self.children = tuple(children)
        return popped

    def build_flatmap(
        self,
        align: int = BufferAlign.AVX512,
        alignb: int = BufferAlign.PAGE,
        name_join: bool = True,
        force_merge: bool = False,
        verbose: bool = False,
    ):
        """Build, allocate, and return the initialized leaf values in preorder.

        Convenience wrapper around ``build_bmap`` → ``allocate_bmap`` →
        ``flat_inits``. See those functions for details.
        """
        build_bmap(
            self,
            align=align,
            name_join=name_join,
            force_merge=force_merge,
            verbose=verbose,
        )
        allocate_bmap(self, align=align,alignb=alignb)
        return flat_inits(self)

    def __repr__(self):
        return f"|Node {self._short()} rule={'S' if self.rule == BufferMap.SHARED else 'D'} children={len(self.children)}|"

    def __str__(self):
        v = str(RenderTree(self, style=DoubleStyle))
        return v


### Helpers
use_other = object()


def _chkfalign(fshape, forder, shape, order, align):
    """Decide whether to reuse an Array spec's aligned-leading-dimension.

    Returns ``True`` when the requested ``shape``/``order`` match the base
    spec (or are ``None``/unspecified) **and** the ``align`` parameter was not
    explicitly provided (``use_other`` sentinel). Otherwise, the caller is
    signaling intent to change layout, so the previous alignment is dropped.
    """
    # change this later so that if it's reversed dimensions in forder != order we keep align
    return (
        align is use_other
        and (shape is None or fshape == shape)
        and (order is None or forder == order)
    )


# ValueNode
def v_spec(value, name=None):
    """Wrap a literal value in a :class:`ValueNode` that doesn't use buffer
    bytes.

    The value participates in preorder flattening alongside arrays and is
    returned by ``flat_inits`` in position order.
    """
    return ValueNode(value, name)


# ArrayNode
def ar_spec(
    shape=(0,), dtype=np.float64, order="C", init_op=None, name=None, align_ldim=None
) -> ArrayNode:
    """Define an :class:`ArrayNode` with optional aligned-leading-dimension.

    ``align_ldim`` pads the leading dimension (last in C order, first in F)
    to a multiple of ``align_ldim // dtype.itemsize`` elements; the backing
    shape is stored in ``bshape`` while the exposed logical shape remains
    ``shape``. NOTE this could make certain kernels and array access patterns much
    quicker, but it also makes it non-contiguous which can break various numpy,
    scipy, and numba operations.
    """
    return ArrayNode(shape, dtype, order, init_op, name=name, align_ldim=align_ldim)


# @cache
def f_arspec_i(
    shape=None,
    dtype=None,
    order=None,
    init_op=use_other,
    name=None,
    align_ldim=use_other,
) -> callable:
    """Return a partially-applied constructor for :func:`f_arspec`.

    Any field left as ``None``/``use_other`` will inherit from the base spec
    supplied at call-time.
    """
    sshape, ddtype, oorder, iinit_op, nname, aalign_ldim = (
        shape,
        dtype,
        order,
        init_op,
        name,
        align_ldim,
    )
    return partial(
        f_arspec,
        shape=sshape,
        dtype=ddtype,
        order=oorder,
        init_op=iinit_op,
        name=nname,
        align_ldim=aalign_ldim,
    )


def f_arspec(
    arspec: ArrayNode,
    shape=None,
    dtype=None,
    order=None,
    init_op=use_other,
    name=None,
    align_ldim=use_other,
) -> ArrayNode:
    """Clone an :class:`ArrayNode` spec, overriding selected fields.

    ``align_ldim`` is inherited only if ``shape`` and ``order`` are unchanged
    and ``align_ldim`` wasn't explicitly provided (see ``_chkfalign``).
    Passing ``init_op=None`` explicitly disables the original initializer.
    """
    rs = arspec
    if not isinstance(rs, ArrayNode):
        raise ValueError("`f_arspec` only takes ArrayNode for `arspec`.")
    spc_align = _chkfalign(rs.shape, rs.order, shape, order, align_ldim)
    align_ldim = None if align_ldim is use_other else align_ldim
    sshape, ddtype, oorder, iinit_op, aalign_ldim, nname = (
        rs.shape if shape is None else shape,
        rs.dtype if dtype is None else dtype,
        rs.order if order is None else order,
        rs.init_op if init_op is use_other else init_op,
        rs.bshape if spc_align else align_ldim,
        name,
    )
    return ar_spec(sshape, ddtype, oorder, iinit_op, nname, aalign_ldim)


def array_arspec(
    arr: np.ndarray,
    shape=None,
    dtype=None,
    order=None,
    init_op=use_other,
    name=None,
    align_ldim=use_other,
    _aldim_def=BufferAlign.AVX512,
) -> ArrayNode:
    """Create an Array spec from an existing ndarray, preserving layout.

    Detects C/F/"aligned" order via strides. If requested order or shape differ
    and ``align_ldim`` wasn't provided, alignment is dropped to avoid slicing
    mismatches. By default, the original array is used as ``init_op``.
    """
    torder = _gao(arr)
    nr = order == "A" or order is None
    if torder == "A":
        torder, bshape = _sao(arr)
        if not (nr or order == torder) and align_ldim is use_other:
            align_ldim = None
    elif align_ldim is use_other:
        align_ldim = None
    if nr:
        order = torder
    if not (shape is None or shape == arr.shape) and align_ldim is use_other:
        align_ldim = None
    if align_ldim is use_other:
        align_ldim = bshape
    return ar_spec(
        arr.shape if shape is None else shape,
        arr.dtype if dtype is None else dtype,
        order,
        arr if init_op is use_other else init_op,
        name,
        align_ldim,
    )


def ft_arspec(
    arspec: ArrayNode,
    shape=None,
    dtype=None,
    order=None,
    init_op=use_other,
    name=None,
    align_ldim=use_other,
) -> ArrayNode:
    """Return a **transposed** Array spec without changing memory layout.

    Swaps C↔F logical order, flips shapes/``bshape``, and carries over
    ``init_op`` (transposed if ndarray, or via a wrapper if callable).
    """
    rs = arspec
    rop = rs.init_op
    top = (
        rop.T
        if isinstance(rs.init_op, np.ndarray)
        else (lambda ar: rop(ar.T))
        if rop is not None
        else rop
    )
    tspec = ar_spec(
        rs.shape[::-1],
        rs.dtype,
        "C" if rs.order != "C" else "F",
        top,
        rs.name,
        rs.bshape[::-1],
    )
    return f_arspec(
        tspec,
        shape=shape,
        dtype=dtype,
        order=order,
        init_op=init_op,
        name=name,
        align_ldim=align_ldim,
    )


# ContainerNode


def sb_node(*args, name=None, no_merge=False) -> ContainerNode:
    """Create a shared-region :class:`ContainerNode` (children overlay
    memory)."""
    return ContainerNode(*args, rule=BufferMap.SHARED, name=name, no_merge=no_merge)


def db_node(*args, name=None, no_merge=False, no_align=False) -> ContainerNode:
    """Create a distinct-region :class:`ContainerNode` (children are
    concatenated).

    ``no_align=True`` disables per-child alignment so the concatenation can
    exactly match an external memory layout.
    """
    return ContainerNode(
        *args, rule=BufferMap.DISTINCT, name=name, no_merge=no_merge, no_align=no_align
    )


def oneshot_args(bmap: BaseNode):
    """Build → allocate → return initialized leaf values in preorder."""
    return flat_inits(allocate_bmap(build_bmap(bmap)))


def flat_inits(bmap: BaseNode) -> Tuple[np.ndarray, ...]:
    """Return initialized values for all leaf items in preorder."""
    return items_init(flatten_items(bmap))


def flatten_items(bmap: BaseNode) -> Tuple[ItemNodeT]:
    """Return all leaf ``ItemNodeT`` instances in preorder."""
    return (*(n for n in PreOrderIter(bmap) if isinstance(n, ItemNodeT)),)


def items_init(items: Tuple[ItemNodeT]):
    """Extract ``.value`` from each item leaf in order (arrays/values)."""
    return (*(n.value for n in items if isinstance(n, ItemNodeT)),)


# Build Buffer Offset Tree and Metadata.
def build_bmap(
    bmap: ContainerNode,
    *,
    align: int = BufferAlign.AVX512,
    name_join: bool = True,
    force_merge: bool = False,
    verbose: bool = False,
) -> ContainerNode:
    """Run reduction → size/offset propagation on a buffer-map tree.

    Parameters
    ----------
    align : int
        Global alignment used during size rounding and offset assignment.
    name_join : bool
        When merging like-ruled containers, prefix child labels with the parent
        label to preserve uniqueness.
    force_merge : bool
        Merge even if either side has ``no_merge=True``.
    verbose : bool
        If true, run ``check_bmap`` to report duplicate labels.
    """
    reduce_bmap(bmap, name_join=name_join, force_merge=force_merge)
    gen_buffer_specs(bmap, align=align)
    if verbose:
        check_bmap(bmap)
    return bmap


# Allocate and assign buffers for buffer map/tree, only for the arrays.
def allocate_bmap(
    bmap: ContainerNode, align: int = BufferAlign.AVX512,alignb: int = BufferAlign.PAGE
) -> ContainerNode:
    """
    Allocate a single aligned top buffer and map arrays to views.
    Requires ``build_bmap`` to have populated sizes/offsets.
    
    :param bmap: Buffer map.
    :param align: What the ArrayNodes are offset aligned to. But not the initial buffer.
    :param alignb: How the starting buffer offset is aligned.
    If the buffer is page aligned, and most/all arrays can fit within the page, the memory will be more efficient.
    :return: 
    """
    if bmap.nbytes is None:
        raise RuntimeError("Call build_bmap() before allocate_bmap().")
    top = aligned_buffer(bmap.nbytes, align=max(alignb,align)) #must be atleast array align large.
    bmap.buffer = top
    _alloc(bmap, top, align)
    return bmap


# other
def _gao(arr):
    """Get array memory order: 'C', 'F', or 'A' (aligned/other)."""
    if arr.flags["C_CONTIGUOUS"]:
        return "C"
    elif arr.flags["F_CONTIGUOUS"]:
        return "F"
    return "A"


def _sao(arr: np.ndarray):
    """Infer (order, backing-shape) from strides for aligned buffers.

    Returns
    -------
    order : Literal['C','F']
    bshape : tuple[int, ...]
        Backing shape with an expanded leading dimension computed from strides.
    """
    s = list(arr.shape)
    if arr.ndim < 2:
        return tuple(s)
    s0, sn = abs(arr.strides[0]), abs(arr.strides[-1])
    i, o, od = (-1, -2, "C") if s0 >= sn else (0, 1, "F")
    s[i] = abs(arr.strides[o]) // abs(arr.strides[i])
    return od, tuple(s)


# ---------------------------------------------------------------------------
# 1) Reduction (merge adjacent identical‑rule containers)
# ---------------------------------------------------------------------------
def _merge_child_into_parent(
    parent: ContainerNode, child: ContainerNode, *, name_join: bool
):
    """Move ``child``'s children into ``parent`` (same rule), optionally
    renaming.

    When ``name_join`` is true and both have labels, grand-children labels are
    prefixed with the parent's label to preserve context and reduce collisions.
    """
    if name_join and parent.label and child.label:
        for grand in child.children:
            grand.label = (
                f"{parent.label}_{grand.label or ''}" if grand.label else parent.label
            )
    for g in list(child.children):
        g.parent = parent
    child.parent = None


def reduce_bmap(
    bmap: ContainerNode, *, name_join: bool = True, force_merge: bool = False
) -> ContainerNode:
    """Merge adjacent containers with the same rule.

    If ``force_merge`` is False, a ``no_merge=True`` on either container blocks
    the merge; otherwise it is ignored.
    """
    for node in list(PreOrderIter(bmap)):
        if not isinstance(node, ContainerNode):
            continue
        for ch in list(node.children):
            if (
                isinstance(ch, ContainerNode)
                and node.rule == ch.rule
                and (force_merge or (not node.no_merge and not ch.no_merge))
            ):
                _merge_child_into_parent(node, ch, name_join=name_join)
    return bmap


# ---------------------------------------------------------------------------
# 2) Size propagation & offset assignment w/ alignment
# ---------------------------------------------------------------------------
def _compute_nbytes(node: BaseNode, align: int = BufferAlign.AVX512) -> int:
    """Compute subtree byte sizes with alignment rounding.

    - ``ValueNode`` contributes 0.
    - ``ArrayNode`` contributes its exact size (no rounding here).
    - ``ContainerNode`` combines child sizes per rule; ``no_align`` on
      DISTINCT containers disables per-child rounding.
    """
    if isinstance(node, ValueNode):
        return 0
    if isinstance(node, ArrayNode):
        return node.nbytes
    # compute child sizes
    sizes = [_compute_nbytes(ch, align) for ch in node.children]
    if node.rule == BufferMap.SHARED:
        raw = max(sizes) if sizes else 0
        # round up shared region
        node.nbytes = ((raw + align - 1) // align) * align
    else:
        # distinct: sum rounded child regions
        if node.no_align:  # if no_align then nbytes will be rounded to match whole container alignment, but not children alignment. Which means that arraynodes that must be aligned contiguously can be forced to, if they represent the memory of another array exactly.
            ln = sum(size for size in sizes)
            node.nbytes = ((ln + align - 1) // align) * align
        else:
            node.nbytes = sum(((size + align - 1) // align) * align for size in sizes)
    return node.nbytes


def _assign_offsets(
    node: BaseNode, base: int = 0, align: int = BufferAlign.AVX512
) -> None:
    """Assign offsets in a single pass according to container rules.

    SHARED writes the same ``base`` to all children; DISTINCT advances by each
    child's (rounded) size unless ``no_align=True``.
    """
    node.ofs = base
    if isinstance(node, ArrayNode):
        node.ofs = base
    elif isinstance(node, ContainerNode):
        if node.rule == BufferMap.SHARED:
            for ch in node.children:
                _assign_offsets(ch, base, align)
        else:
            cur = base
            for ch in node.children:
                _assign_offsets(ch, cur, align)
                size = ch.nbytes or 0
                if node.no_align:
                    cur += size
                else:
                    cur += ((size + align - 1) // align) * align


def gen_buffer_specs(
    bmap: ContainerNode, align: int = BufferAlign.AVX512
) -> ContainerNode:
    """Populate ``nbytes`` and ``ofs`` for the entire tree (no allocation)."""
    _compute_nbytes(bmap, align)
    _assign_offsets(bmap, 0, align)
    return bmap


# ---------------------------------------------------------------------------
# 3) Sanity check for duplicate labels
# ---------------------------------------------------------------------------
def check_bmap(bmap: ContainerNode) -> None:
    """Print duplicate labels (diagnostic)."""
    seen: dict[str, List[BaseNode]] = {}
    for n in PreOrderIter(bmap):
        if n.label:
            seen.setdefault(n.label, []).append(n)
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    if dup:
        print("Duplicate labels:")
        for k, v in dup.items():
            print(f"  {k} × {len(v)}")


# ---------------------------------------------------------------------------
# 5) Allocation
# ---------------------------------------------------------------------------
def _alloc(
    node: BaseNode, root_buf: np.ndarray, align: int = BufferAlign.AVX512
) -> None:
    """Attach container slices and materialize ``ArrayNode`` views
    recursively."""
    if isinstance(node, ValueNode):
        return
    if not isinstance(node, ArrayNode):
        if node.ofs is None or node.nbytes is None:
            raise RuntimeError("Node must have ofs/nbytes (run build_bmap first)")
        node.buffer = root_buf[node.ofs : node.ofs + node.nbytes]
        for ch in node.children:
            _alloc(ch, root_buf, align)
        return

    if node.ofs is None:
        raise RuntimeError("ArrayNode must have ofs assigned (run build_bmap first)")
    node.mk_array(root_buf)


# 6) Query helpers
def _walk_index(node: BaseNode, idx: Sequence[int]) -> BaseNode:
    """Follow a child-index path from ``node`` and return the target node."""
    cur = node
    for i in idx:
        cur = cur.children[i]  # type: ignore
    return cur


def bmap_get(bid: Union[Sequence[int], str, BaseNode], bmap: ContainerNode) -> BaseNode:
    """Resolve a node reference by path, label, or pass-through instance."""
    if isinstance(bid, BaseNode):
        return bid
    if isinstance(bid, (list, tuple)):
        return _walk_index(bmap, bid)
    if isinstance(bid, str):
        for n in PreOrderIter(bmap):
            if n.label == bid:
                return n
        raise KeyError(bid)
    raise TypeError("bid must be path/label/node")


def dtype_abbr(dtype: np.dtype) -> str:
    """Return a short ``kind_bits`` abbreviation (e.g., ``'f_64'``,
    ``'i_32'``)."""
    kind = dtype.kind.lower()  # e.g. 'i', 'u', 'f', 'c', 'b', 'M', 'm', …
    bits = dtype.itemsize * 8  # itemsize is in bytes → *8 for bits
    return f"{kind}_{bits}"


# 7) DOT export
def _dot_node_attr(node: BaseNode, *, with_offsets: bool) -> str:
    """Generate DOT attributes with name inside node and other info as
    external."""
    # Secondary info for external label (xlabel)
    info_parts: List[str] = []
    if node.label:
        info_parts.append(f"- {node.label} -")
    if isinstance(node, ArrayNode):
        if node.align_ldim is not None:
            info_parts.append(f"{node.shape}, {node.vshape}, {dtype_abbr(node.dtype)}")
        else:
            info_parts.append(f"{node.shape}, {dtype_abbr(node.dtype)}")
        col = 'color="lime"'
    elif isinstance(node, ValueNode):
        vt = repr(node.value)
        qrg = 18
        vt = vt[:qrg]
        if qrg <= len(vt):
            vt = vt[:10] + "..."
        info_parts.append(vt)
        col = 'color="pink"'
    else:
        if node.rule == BufferMap.DISTINCT:
            col = 'color="orange"'
        elif node.rule == BufferMap.SHARED:
            col = 'color="cyan"'
    if with_offsets and node.ofs is not None:
        info_parts.append(f"[{node.ofs}, {node.ofs + (node.nbytes or 0)})")
    if node.nbytes is not None:
        info_parts.append(f"{node.nbytes / 1024:.2f} KB") #maybe change to 
    # Build attribute list
    label = "\n".join(info_parts)
    parts: List[str] = [f'label="{label}"', 'shape="circle"', 'labelloc="c"', col]
    # parts.append('style="filled"')
    # parts.append('fillcolor="white"')
    return ", ".join(parts)


def bmap_todot(bmap: ContainerNode, *, with_offsets: bool = True) -> str:
    """Serialize the tree to Graphviz DOT via
    ``anytree.exporter.DotExporter``."""
    lines = []
    for line in DotExporter(
        bmap,
        nodenamefunc=lambda n: f"n_{n.id}",
        nodeattrfunc=lambda n: _dot_node_attr(n, with_offsets=with_offsets),
    ):
        lines.append(line)
    return "\n".join(lines)


# 8) PyVis visualiser
def _bmap_pyvis(src: Union[str, ContainerNode], *, with_offsets: bool = False):
    """Internal helper that returns a configured ``pyvis.Network`` graph.

    Accepts either a DOT string or a ``ContainerNode``; when given a node,
    converts via ``bmap_todot`` then builds a NetworkX graph using ``pydot``.
    """
    try:
        from pyvis.network import Network
        import networkx as nx
        import pydot
    except ImportError as e:
        raise ImportError("pyvis, networkx, pydot required for bmap_pyvis()") from e
    if isinstance(src, str):
        dot_data = src
    elif isinstance(src, ContainerNode):
        dot_data = bmap_todot(src, with_offsets=with_offsets)
    else:
        raise TypeError("src must be DOT str or ContainerNode")
    graphs = pydot.graph_from_dot_data(dot_data)
    if not graphs:
        raise ValueError("Failed to parse DOT data")
    pgraph = graphs[0]
    g_nx = nx.drawing.nx_pydot.from_pydot(pgraph)
    net = Network(directed=True)
    net.from_nx(g_nx)
    # Strip embedded quotes from labels and colors
    for node in net.nodes:
        # Clean label
        lbl = node.get("label")
        if isinstance(lbl, str) and lbl.startswith('"') and lbl.endswith('"'):
            node["label"] = lbl.strip('"')
        # Clean color
        col = node.get("color")
        if isinstance(col, str) and col.startswith('"') and col.endswith('"'):
            node["color"] = col.strip('"')
    net.set_options("""{
  "layout": {"hierarchical":{"enabled":true,"direction":"UD","sortMethod":"hubsize","blockShifting":true,"edgeMinimization":true}},
  "physics":{"enabled":true}
}""")
    return net

BMAP_ROPTS2="""{
      "layout": {"hierarchical":{"enabled":false,"direction":"UD","sortMethod":"hubsize","blockShifting":true,"edgeMinimization":true}},
      "physics":{"enabled":true, "wind": { "x": 0.0, "y": 0 },"repulsion":{"nodeDistance": 200,"springLength": 200},"centralGravity":0.1}
    }"""

BMAP_ROPTS="""{
      "layout": {"hierarchical":{"enabled":true,"direction":"UD","sortMethod":"hubsize","blockShifting":true,"edgeMinimization":true}},
      "physics":{"enabled":true, "wind": { "x": 0.02, "y": 0 },"hierarchicalRepulsion":{"nodeDistance": 55,"avoidOverlap": 1}}
    }"""

_bopt="""{
      "layout": {"hierarchical":{"enabled":true,"direction":"UD","sortMethod":"hubsize","blockShifting":true,"edgeMinimization":true}},
      "physics":{"enabled":true, "wind": { "x": 0.02, "y": 0 },"hierarchicalRepulsion":{"nodeDistance": 67,"avoidOverlap": 1}}
    }"""

# 8) PyVis visualiser
def bmap_pyvis(src: Union[str, ContainerNode], with_offsets: bool = False,height='1000px',width='100%',render_options=BMAP_ROPTS):
    """Build a PyVis interactive tree using a top-down layout and physics
    enabled."""
    try:
        from pyvis.network import Network
        import networkx as nx
        import pydot
    except ImportError as e:
        raise ImportError("pyvis, networkx, pydot required for bmap_pyvis()") from e

    # Obtain DOT source and parse
    dot_data = (
        src if isinstance(src, str) else bmap_todot(src, with_offsets=with_offsets)
    )
    graphs = pydot.graph_from_dot_data(dot_data)
    if not graphs:
        raise ValueError("Failed to parse DOT data")
    print(graphs)
    pgraph = graphs[0]
    # Convert to NetworkX
    g_nx = nx.drawing.nx_pydot.from_pydot(pgraph)
    # Insert a blank white dummy sibling under the root container to anchor hubsize
    # Determine root: if ContainerNode passed, use its name; else take first node

    nodes = list(g_nx.nodes)
    root_id = nodes[0] if nodes else None
    dummy = 0
    #Note: For some reason we need a dummy node otherwise the top level node will be out of place try the old _bmap_pyvis to see strange behavior
    if dummy not in g_nx:
        g_nx.add_node(
            dummy,
            label="",
            shape="circle",
            style="filled",
            fillcolor="white",
            color="white",
            fixed=True,
        )

    succ = list(g_nx.successors(root_id))
    # # Remove all existing root->* edges
    for v in succ:
        g_nx.remove_edge(root_id, v)
    # Re-add with dummy first, then self-loop, then the rest
    new_order = [dummy] + [v for v in succ if v not in (dummy, root_id)]
    for v in new_order:
        g_nx.add_edge(root_id, v)
    # # No other graph modifications

    # Build PyVis network
    net = Network(
        directed=True,
        height=height,
        width=width,
        bgcolor="#111111",
        font_color="black",
    )
    net.from_nx(g_nx)

    # Clean any embedded quotes
    for node in net.nodes:
        lbl = node.get("label")
        if isinstance(lbl, str) and lbl.startswith('"') and lbl.endswith('"'):
            node["label"] = lbl.strip('"')
        col = node.get("color")
        if isinstance(col, str) and col.startswith('"') and col.endswith('"'):
            node["color"] = col.strip('"')

    if render_options==BMAP_ROPTS and with_offsets: render_options=_bopt #bootleg? Yes
    #render opts
    net.set_options(render_options)

    return net


def save_bmap_tree(
    bmap: ContainerNode, path: str = "bmap_tree.png", *, with_offsets: bool = False
) -> str:
    """Render the tree to a static image using Graphviz and return ``path``."""
    DotExporter(
        bmap,
        nodenamefunc=lambda n: n.id,
        nodeattrfunc=lambda n: _dot_node_attr(n, with_offsets=with_offsets),
    ).to_picture(path)
    return path


def clone_bmap(bmap: ContainerNode):
    """Deep-clone an entire buffer-map tree (labels, sizes, offsets, etc.)."""
    return copy.deepcopy(bmap)
