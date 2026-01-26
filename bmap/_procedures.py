import copy
import keyword
from collections.abc import Collection
from functools import partial
from typing import Any, Callable, List, Sequence, Tuple, TypeGuard, Union

import numpy as np
import sympy as sym
from anytree import NodeMixin, PreOrderIter
from anytree.exporter import DotExporter
from pyvis.network import Network  # type: ignore[untyped-import]

from bmap._nodes import ArrayNode, BaseNode, ContainerNode, ENode, ItemNode, ValueNode
from bmap._util import (
    UO,
    BButil,
    BufferAlign,
    BufferMap,
    DTypeLike,
    InitOp,
    ShapeInput,
    ShapeMaybe,
    SizeMaybe,
    SizeParam,
    SizeSeq,
    SymbolKey,
    UseOther,
    _arrbxpr,
    _arrpxpr,
    _bxpr,
    _chkfalign,
    _gao,
    _pxpr,
    _sao,
    aligned_buffer,
    buffer_expr,
    c_orlen,
    cse_codereduction,
    dtype_abbr,
    eval_buff_expr,
    gen_add,
    gen_max,
    gen_str,
    ls_layers,
    roundup,
)


def build_buffer_allocator(
    buffer_map: ContainerNode,
    args: Collection[SymbolKey] = (),
    kwargs: dict[SymbolKey, SizeParam] | None = None,
    tempvar: str = "t",
    subname: bool = False,
    chkforbuffer: bool = True,
    balign: SizeParam = BufferAlign.PAGE,
    fullreduce: bool = True,
) -> str:
    """
    Builds the string definition of a function that dynamically allocates the
    arrays and values of a buffer map.

    :param buffer_map: Original buffer map.
    :param balign: The initial alignment of the buffer. It must be greater than
        or equal to the alignment you choose to build the buffer_map with.
    """
    # Normalize args to names so user may pass sympy.Symbol or str.
    args = [
        *(
            (k.name if isinstance(k, sym.Symbol) else k)
            for k in args
            if k is not None and k != "None" and isinstance(k, (sym.Symbol, str))
        ),
    ]
    # Drop invalid parameter names; caller may accidentally pass non-symbol junk (e.g. build_bmap return helpers).
    args = [a for a in args if isinstance(a, str) and a.isidentifier() and not keyword.iskeyword(a)]
    # Sort args so dtype params are declared first in generated signature.
    args.sort(
        key=lambda x: c_orlen(x, "type_"),
    )
    # Normalize kwargs keys to names; only symbol keys are supported in allocator signature.
    kwargs = (
        {
            (k.name if isinstance(k, sym.Symbol) else k): v
            for k, v in kwargs.items()
            if k is not None and k != "None" and isinstance(k, (sym.Symbol, str))
        }
        if kwargs is not None
        else {}
    )
    kwargs = {k: v for k, v in kwargs.items() if isinstance(k, str) and k.isidentifier() and not keyword.iskeyword(k)}
    # For now there is also no dead-parameter/dead-code removal, and no checks
    # for if all used input parameters are referenced in the header.
    # If caller wants buffer existence check, include check_alloc logic and seed ct for reduced expr walk.
    if chkforbuffer:
        hd = BButil.build_header(buffer_map, args, kwargs, balign=balign)
        assert buffer_map.nbytes is not None
        allq: list[SizeParam] = [buffer_map.nbytes]
        ct = [1]
    # Otherwise omit check_alloc and start ct at 0.
    else:
        hd = BButil.build_header(buffer_map, args, kwargs, check_alloc=None, balign=balign)
        allq = []
        ct = [0]

    # If fullreduce is enabled, CSE-reduce expressions then emit compact allocator code.
    if fullreduce:
        strl, allc, mdst, estr = _full_reduce(buffer_map, allq, chkforbuffer, tempvar, subname, ct)
    # Otherwise emit direct expressions without reduction.
    else:
        # no reduction is quicker but also much more ugly and less readable.
        strl, allc, mdst, estr = _no_reduce(buffer_map, allq, chkforbuffer, subname)

    fullf = "\n    ".join((hd, *strl, *allc, mdst, "", estr))
    return fullf


def _full_reduce(
    buffer_map: ContainerNode,
    allq: list[SizeParam],
    chkforbuffer: bool,
    tempvar: str,
    subname: bool,
    ct: list[int],
):
    # Collect all expression placeholders from the tree so CSE can minimize them.
    allq.extend(_cxprs(buffer_map))  # already 'least expressions' form.
    # If the root is DISTINCT, drop the final trailing offset expression (it is a no-op slice).
    # if buffer_map.rule == BufferMap.DISTINCT: allq.pop()
    lyrs, reduc = cse_codereduction(allq, tempvar)
    strl = ls_layers(lyrs)
    # If chkforbuffer is enabled, inject allocator buffer creation using the reduced root size.
    allc = (BButil.add_balloc(reduc[0]),) if chkforbuffer else ()
    mkls = _bb2(buffer_map, reduc, subname=subname, ct=ct)
    mdst = "\n".join(mkls[0])
    mdst = mdst.replace("\n", "\n    ")  # in case there are multi-line statements
    estr = f"return {', '.join(str(v) for v in mkls[1])}"
    return strl, allc, mdst, estr


def _no_reduce(
    buffer_map: ContainerNode,
    allq: list[SizeParam],
    chkforbuffer: bool,
    subname: bool,
):
    mkls = _bba(buffer_map, subname=subname)
    # For DISTINCT roots, drop the last trailing slice statement (it would be an empty advancement).
    if buffer_map.rule == BufferMap.DISTINCT:
        mkls[0].pop()
    strl = ()
    # If chkforbuffer is enabled, inject allocator buffer creation using the unreduced root size.
    allc = (BButil.add_balloc(allq[0]),) if chkforbuffer else ()
    mdst = "\n".join(mkls[0])
    mdst = mdst.replace("\n", "\n    ")  # in case there are multi-line statements
    estr = f"return {', '.join(str(v) for v in mkls[1])}"
    return strl, allc, mdst, estr


def _is_shape_tuple(
    value: tuple[int, ...] | tuple[str, tuple[int, ...]],
) -> TypeGuard[tuple[int, ...]]:
    return not (value and isinstance(value[0], str))


def _cxprs(bn: BaseNode, excs: list[sym.Expr] | None = None) -> list[sym.Expr]:
    wn = excs is None
    if wn:
        excs = []
    # If node is a DISTINCT container, collect either aligned_eqns or per-child nbytes.
    if isinstance(bn, ContainerNode) and bn.rule == BufferMap.DISTINCT:
        # If align=True, include aligned_eqns (container-specific per-child offsets).
        if bn.align:
            for c, eq in zip(bn.children, bn.aligned_eqns, strict=False):
                _cxprs(c, excs)
                _pxpr(excs, eq)
        # Otherwise, offsets are driven by child nbytes directly.
        else:
            for c in bn.children:
                _cxprs(c, excs)
                _pxpr(excs, c.nbytes)
    # Otherwise recurse through children and add leaf expressions as needed.
    else:
        for c in bn.children:
            _cxprs(c, excs)
        if isinstance(bn, ArrayNode):
            _arrpxpr(excs, bn.sym_def())
        elif isinstance(bn, ValueNode):
            _pxpr(excs, bn.sym_def())

    return excs


def _cbb2(
    bnode: ContainerNode,
    exls: SizeSeq,
    subname: bool,
    sn: str | None,
    ct: list[int],
    mkls: tuple[list[str], list[str | int | float]],
    is_root: bool,
) -> None:
    # If container is DISTINCT, emit per-child buffer slicing statements.
    if bnode.rule == BufferMap.DISTINCT:
        mkls[0].append("")
        bc = bnode.children
        bcl = len(bc) - 1
        # If align=True, use aligned_eqns; otherwise drive offsets from child nbytes.
        eqs = bnode.aligned_eqns if bnode.align else tuple(c.nbytes for c in bc)
        for i, (c, eq) in enumerate(zip(bc, eqs, strict=False)):
            _bb2(c, exls, subname, bnode.name, ct, mkls)
            # Root DISTINCT drops its final trailing offset expression; stop before emitting the final pop.
            if is_root and i == bcl:
                break
            # Root DISTINCT may also be missing the last reduced expr; guard ct from running off the reduced list.
            if is_root and ct[0] >= len(exls):
                break  # root distinct pops final expr; guard ct
            ot = _bxpr(exls, eq, ct)
            mkls[0].append(ContainerNode.s_def(ot))
    # Otherwise container is SHARED: recurse only (no per-child buffer slicing).
    else:
        mkls[0].append("")
        for c in bnode.children:
            _bb2(c, exls, subname, bnode.name, ct, mkls)


def _bb2(
    bnode: ENode,
    exls: SizeSeq,
    subname: bool = False,
    sn: str | None = None,
    ct: list[int] | None = None,
    mkls: tuple[list[str], list[str | int | float]] | None = None,
) -> tuple[list[str], list[str | int | float]]:
    is_root = mkls is None
    if mkls is None:
        mkls = ([], [])
    if ct is None:
        ct = [0]
    # If node is a container, delegate to shared distinct handler.
    if isinstance(bnode, ContainerNode):
        _cbb2(bnode, exls, subname, sn, ct, mkls, is_root)
    # If node is an array, substitute reduced expressions into (nbytes, bshape, shape) and emit its gen_call.
    elif isinstance(bnode, ArrayNode):
        bytexpr, bshapet, shapet = _arrbxpr(exls, bnode.sym_def(), ct)
        s0, s1 = bnode.gen_call(
            bytexpr=bytexpr,
            bshape=bshapet,
            shape=shapet,
            subn=sn if subname else None,
        )
        mkls[0].append(s0)
        mkls[1].append(s1)
    # Otherwise treat as a ValueNode-like leaf and let its gen_call decide assignment vs inline return.
    else:
        s0, s1 = bnode.gen_call(
            subn=sn if subname else None,
        )
        # If ValueNode has no name, gen_call returns no assignment line; skip appending None.
        if s0 is not None:
            mkls[0].append(s0)
        mkls[1].append(s1)  # type: ignore[bad-argument-type]

    return mkls


def _bba(
    bnode: BaseNode,
    mkls: tuple[list[str], list[str | int | float]] | None = None,
    subname: bool = False,
    sn: str | None = None,
) -> tuple[list[str], list[str | int | float]]:
    if mkls is None:
        mkls = ([], [])
    # If node is a container, emit buffer slicing statements in preorder.
    if isinstance(bnode, ContainerNode):
        mkls[0].append("")
        # DISTINCT containers advance buffer between children.
        if bnode.rule == BufferMap.DISTINCT:
            # If align=True, use aligned_eqns; otherwise use child nbytes for slicing.
            defs = bnode.aligned_eqns if bnode.align else tuple(c.nbytes for c in bnode.children)
            for c, eq in zip(bnode.children, defs, strict=False):
                _bba(c, mkls, subname, bnode.name)
                mkls[0].append(ContainerNode.s_def(eq))
        # SHARED containers recurse only (no slicing).
        else:
            for c in bnode.children:
                _bba(c, mkls, subname, bnode.name)
    # If node is an ArrayNode, emit its allocation/view statement.
    elif isinstance(bnode, ArrayNode):
        s0, s1 = bnode.gen_call(subn=sn if subname else None)
        mkls[0].append(s0)
        mkls[1].append(s1)
    # Otherwise treat as ValueNode and emit its assignment/inline return.
    else:
        assert isinstance(bnode, ValueNode)
        s0, s1 = bnode.gen_call(subn=sn if subname else None)
        # If ValueNode has no name, gen_call returns no assignment line; skip appending None.
        if s0 is not None:
            mkls[0].append(s0)
        mkls[1].append(s1)  # type: ignore[bad-argument-type]

    return mkls


def build_bmap(
    bmap: ContainerNode,
    *,
    align: SizeParam = BufferAlign.AVX512,
    name_join: bool = True,
    force_merge: bool = False,
    verbose: bool = False,
) -> set[sym.Symbol]:
    """Run reduction → size/offset propagation on a buffer-map tree.

    Parameters
    ----------
    align : int | sym.Expr
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
    symbols: set[sym.Basic] = set()
    _compute_nbytes(bmap, align=align, _symbols=symbols)
    if verbose:
        check_bmap(bmap)
    return symbols  # type: ignore[bad-return]


# ---------------------------------------------------------------------------
# 1) Reduction (merge adjacent identical‑rule containers)
# ---------------------------------------------------------------------------
def _merge_child_into_parent(parent: ContainerNode, child: ContainerNode, *, name_join: bool) -> None:
    """Move ``child``'s children into ``parent`` (same rule), optionally
    renaming.

    When ``name_join`` is true and both have labels, grand-children labels are
    prefixed with the parent's label to preserve context and reduce collisions.
    """
    if name_join and parent.label and child.label:
        for grand in child.children:
            grand.label = f"{parent.label}_{grand.label or ''}" if grand.label else parent.label
    for g in list(child.children):
        g.parent = parent
    child.parent = None


def reduce_bmap(bmap: ContainerNode, *, name_join: bool = True, force_merge: bool = False) -> ContainerNode:
    """Merge adjacent containers with the same rule.

    If ``force_merge`` is False, a ``no_merge=True`` on either container blocks
    the merge; otherwise it is ignored.
    """
    # Walk the tree in preorder and merge container children into parents when compatible.
    for node in list(PreOrderIter(bmap)):
        # Only ContainerNodes participate in merge rules.
        if not isinstance(node, ContainerNode):
            continue
        # Consider each child for possible merge into this container.
        for ch in list(node.children):
            # Merge only when rules/align match, and no_merge doesn't block unless force_merge is set.
            if (
                isinstance(ch, ContainerNode)
                and node.rule == ch.rule
                and node.align == ch.align
                and (force_merge or (not node.no_merge and not ch.no_merge))
            ):
                _merge_child_into_parent(node, ch, name_join=name_join)
    return bmap


# ---------------------------------------------------------------------------
# 2) Size propagation & offset assignment w/ alignment
# ---------------------------------------------------------------------------
def _compute_nbytes(
    node: ENode,
    align: SizeParam = BufferAlign.AVX512,
    simplify: bool = True,
    _symbols: set[sym.Basic] | None = None,
) -> SizeParam:
    """Compute subtree byte sizes with alignment rounding.

    - ``ValueNode`` contributes 0.
    - ``ArrayNode`` contributes its exact size (no rounding here).
    - ``ContainerNode`` combines child sizes per rule; ``align`` on
      DISTINCT containers disables per-child rounding.
    """
    # If caller didn't supply symbol accumulator, create one.
    if _symbols is None:
        _symbols = set()
    align_expr = buffer_expr(align)
    # If leaf (ValueNode/ArrayNode), record its symbols and return its nbytes directly.
    if isinstance(node, ItemNode):
        _symbols.update(s for s in node.free_symbols)
        return node.nbytes
    # compute child sizes
    assert isinstance(node, ContainerNode)
    # If SHARED, use max(child sizes) then round once.
    if node.rule == BufferMap.SHARED:
        sizes = (*(_n_(ch, align, simplify, _symbols) for ch in node.children),)
        raw = gen_max(sizes, simplify) if sizes else 0
        # round up shared region
        node.nbytes = roundup(raw, align_expr)
    # If DISTINCT + align=True, round each child (arrays only) and sum.
    elif node.align:  # we make sure every node is aligned.
        ng = node.aligned_eqns
        ng.clear()
        # we already know Container nodes are certainly aligned.
        # this is true because aligned=False distinct nodes and shared nodes are still aligned at the container level.
        # so aligned=True distinct nodes will still be aligned as well.
        # So we can skip their expensive sympy expression build and just use the existing nbytes expression.
        ng.extend(
            roundup(_n_(ch, align, simplify, _symbols), align_expr)
            if isinstance(ch, ArrayNode)
            else _n_(ch, align, simplify, _symbols)
            for ch in node.children
        )
        node.nbytes = gen_add(ng, simplify) if ng else 0
    # Otherwise DISTINCT + align=False: sum then round once at container level.
    else:
        # If not align, nbytes is rounded at the container level, not per-child.
        # This can be useful when arrays must be aligned contiguously to match
        # an external layout exactly.
        # However, if another container is a non-leading child, this may break
        # alignment for nodes it contains, so DISTINCT should default to align.
        sizes = (*(_n_(ch, align, simplify, _symbols) for ch in node.children),)
        raw = gen_add(sizes, simplify) if sizes else 0
        node.nbytes = roundup(raw, align_expr)
    assert node.nbytes is not None
    return node.nbytes


_n_ = _compute_nbytes


def arrays_map(
    buffer_map: ContainerNode,
    sym_dic: dict[SymbolKey, int] | None = None,
    balign: int = BufferAlign.PAGE,
    _buffer: np.ndarray | None = None,
    _base: int = 0,
) -> None:
    """Inits arrays based off of the buffer map. Arrays are placed within their array node.
    And container nodes receive the buffer subindex range they represent.
    :param balign: The initial alignment of the buffer. It must be greater than
        or equal to the alignment you choose to build the buffer_map with.
    """
    # make sure sym_dic is using symbols, so for user it's ok to specify them with string keys
    if sym_dic is None:
        sym_dic = {}
    sym_map: dict[sym.Symbol, int] = {}
    for k, v in sym_dic.items():
        if isinstance(k, sym.Symbol):
            sym_map[k] = v
        else:
            sym_key = buffer_expr(k)
            sym_map[sym_key] = v
    # make a buffer if none was provided
    if _buffer is None:
        _buffer = aligned_buffer(eval_buff_expr(buffer_map.nbytes, sym_dic), balign)
    _generate_nodearrays(buffer_map, _buffer, _base, sym_map)


def allocate_bmap(
    bmap: ContainerNode,
    sym_dic: dict[SymbolKey, int] | None = None,
    balign: int = BufferAlign.PAGE,
    buffer: np.ndarray | None = None,
) -> ContainerNode:
    """Allocate backing buffers and populate array nodes in-place."""
    arrays_map(bmap, sym_dic=sym_dic, balign=balign, _buffer=buffer)
    return bmap


def _generate_nodearrays(
    node: BaseNode,
    buffer: np.ndarray | None = None,
    base: int = 0,
    sym_dic: dict[sym.Symbol, int] | None = None,
) -> None:
    if sym_dic is None:
        sym_dic = {}
    node.ofs = base
    if isinstance(node, ArrayNode):
        node.mk_array(buffer, sym_dic)
    elif isinstance(node, ContainerNode):
        if node.rule == BufferMap.SHARED:
            for ch in node.children:
                _generate_nodearrays(ch, buffer, base, sym_dic)
        elif node.align:
            for ch, eqn in zip(node.children, node.aligned_eqns, strict=False):
                _generate_nodearrays(ch, buffer, base, sym_dic)
                base += eval_buff_expr(eqn, sym_dic)
        else:
            for ch in node.children:
                _generate_nodearrays(ch, buffer, base, sym_dic)
                base += eval_buff_expr(ch.nbytes, sym_dic)


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


def build_flatmap(
    self: ContainerNode,
    align: SizeParam = BufferAlign.AVX512,
    alignb: SizeParam = BufferAlign.PAGE,
    name_join: bool = True,
    force_merge: bool = False,
    verbose: bool = False,
):
    """Build and return initialized leaf values in preorder."""
    build_bmap(
        self,
        align=align,
        name_join=name_join,
        force_merge=force_merge,
        verbose=verbose,
    )
    # allocate_bmap(self, align=align,alignb=alignb)
    return flat_inits(self)


def oneshot_args(bmap: ContainerNode) -> Tuple[np.ndarray | SizeMaybe, ...]:
    """Build → allocate → return initialized leaf values in preorder."""
    build_bmap(bmap)
    return flat_inits(allocate_bmap(bmap))


def flat_inits(bmap: BaseNode) -> Tuple[np.ndarray | SizeMaybe, ...]:
    """Return initialized values for all leaf items in preorder."""
    return items_init(flatten_items(bmap))


def flatten_items(bmap: BaseNode) -> Tuple[ItemNode, ...]:
    """Return all leaf ``ItemNode`` instances in preorder."""
    return (*(n for n in PreOrderIter(bmap) if isinstance(n, ItemNode)),)


def items_init(items: Tuple[ItemNode, ...]) -> Tuple[np.ndarray | SizeMaybe, ...]:
    """Extract ``.value`` from each item leaf in order (arrays/values)."""
    return (*(n.value for n in items if isinstance(n, ItemNode)),)


# ValueNode
def v_spec(value: Any, name: str | None = None) -> ValueNode:
    """Wrap a literal value in a :class:`ValueNode` that doesn't use buffer
    bytes.

    The value participates in preorder flattening alongside arrays and is
    returned by ``flat_inits`` in position order.
    """
    return ValueNode(value, name)


# ArrayNode
def ar_spec(
    shape: ShapeInput = (0,),
    dtype: DTypeLike = np.float64,
    order: str = "C",
    init_op: InitOp | None = None,
    name: str | None = None,
    align_ldim: ShapeInput | None = None,
) -> ArrayNode:
    """
    Constructs an ArrayNode with the specified shape, data type, memory order,
    initialization operation, name, and alignment along dimensions. It serves
    as a specification for array creation, allowing for detailed control over
    the array's properties and initialization.

    :param shape: Defines the desired shape of the array. Can be a tuple/sequence of
        strings, ints, or sympy positive integer expressions representing the dimensions.
    :param dtype: Specifies the data type of the array elements, providing
        options to use NumPy data types, generic types, or symbolic
        expressions.
    :param order: Determines the memory layout order of the array, either
        'C' for row-major or 'F' for column-major.
    :param init_op: An operation to initialize the array. This can be a None, a
        NumPy array, or a callable that performs initialization operations.
    :param name: An optional name for the array, useful for identification
        and reference within larger computations or models. Array names are required in the code generator.
    :param align_ldim: Aligns the leading dimension to a specified shape,
        which can aid in memory optimization and performance tuning. The leading
        dimension corresponds to the memory layout and not the array order. So
        the last dimension of a 'C' ordered array is leading, the first dimension of
        an 'F' ordered array is leading.
    :return: An ArrayNode configured with the specified parameters.
    """

    return ArrayNode(shape, dtype, order, init_op, name=name, align_ldim=align_ldim)


# @cache
def f_arspec_i(
    shape: ShapeInput | None = None,
    dtype: DTypeLike | None = None,
    order: str | None = None,
    init_op: InitOp | UseOther = UO,
    name: str | None = None,
    align_ldim: ShapeMaybe | UseOther = UO,
) -> Callable[..., ArrayNode]:
    """
    Create a partial function for array specification with the given parameters.

    This is no different from wrapping ``f_arspec`` with different defaults, could be useful
    for complex dynamic patterns, for this reason refer to ``f_arspec`` directly for
    documentation.
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
    shape: ShapeInput | None = None,
    dtype: DTypeLike | None = None,
    order: str | None = None,
    init_op: InitOp | UseOther = UO,
    name: str | None = None,
    align_ldim: ShapeMaybe | UseOther = UO,
) -> ArrayNode:
    """Clone an :class:`ArrayNode` spec, overriding selected fields.

    ``align_ldim`` is inherited only if ``shape`` and ``order`` are unchanged and
    ``align_ldim`` wasn't explicitly provided (see ``_chkfalign``). Passing
    ``init_op=None`` explicitly disables the original initializer.

    :param arspec: The ArrayNode to be cloned.
    :param shape: Optional specification for the shape to override.
    :param dtype: Optional specification for the data type to override.
    :param order: Optional specification for the memory layout order to override.
    :param init_op: Specifies the initialization operation; passing None explicitly
        disables the initializer. UO make it use the initializer from ``arspec``, be it None, an array or callable.
    :param name: Optional new name for the cloned ArrayNode.
    :param align_ldim: Optional parameter to override alignment, otherwise it is
        checked and inherited.
    :return: The cloned ArrayNode with specified fields overridden based on
        provided parameters.
    :rtype: ArrayNode
    """

    rs = arspec
    if not isinstance(rs, ArrayNode):
        raise ValueError("`f_arspec` only takes ArrayNode for `arspec`.")
    # do we keep the leading dimension alignment
    spc_align = _chkfalign(rs.shape, rs.order, shape, order, align_ldim)
    if isinstance(align_ldim, UseOther):
        align_ldim_val = None
    # We will use the new alignment where None forces no alignment; otherwise a
    # sympy expression/dimension tuple/byte ceiling forces new alignment.
    # Leaving it as UO uses arspec's alignment only if possible.
    else:
        align_ldim_val = align_ldim
    # for non-optional args, simply get if inputs are None.
    sshape = rs.shape if shape is None else shape
    ddtype = rs.dtype if dtype is None else dtype
    oorder = rs.order if order is None else order
    # None has the same 'nothing callable' meaning as with align_ldim, UO uses other arspecs init_op.
    if isinstance(init_op, UseOther):
        iinit_op = rs.init_op
    else:
        iinit_op = init_op
    aalign_ldim = rs.bshape if spc_align else align_ldim_val
    # Name is the only parameter that doesn't fall back to arspec's name, as it wouldn't really make sense to use this
    # function if that were the case.
    nname = name
    return ar_spec(
        sshape,
        ddtype,
        oorder,
        iinit_op,
        nname,
        aalign_ldim,
    )


def array_arspec(
    arr: np.ndarray,
    shape: ShapeInput | None = None,
    dtype: DTypeLike | None = None,
    order: str | None = None,
    init_op: InitOp | UseOther = UO,
    name: str | None = None,
    align_ldim: ShapeMaybe | UseOther = UO,
) -> ArrayNode:
    """
    Generates an array specification node with specific properties derived from
    the provided parameters and the input array. It verifies and adjusts the
    order and alignment settings according to the attributes of the input
    array, shape, data type, and other optional operation details.

    :param arr: The input NumPy array from which properties will be extracted
        and used in generating the array specification node.
    :param shape: Optional shape parameter used to override or specify a
        particular shape for the array specification node. Defaults to None,
        which means it will use the shape of `arr`.
    :param dtype: Optional data type to define the desired data type for the
        array specification node, defaults to None, meaning it will use the
        data type of `arr`.
    :param order: Optional string to indicate the desired memory layout order
        ('C', 'F', 'A', or None) for the array. If None or 'A', it adapts
        based on the input array.
    :param init_op: Operation to initialize the array; defaults to
        `UseOther`, which implies using the original array or given
        initialization operation.
    :param name: Optional parameter to specify a name for the array
        specification (needed for the code generator).
    :param align_ldim: Indicates whether to align the leading dimensions,
        defaults to `UseOther`, which will adjust based on input array
        properties unless specified otherwise. None will make it contiguous.
    :returns: An array specification node that details the properties like
        shape, data type, order, initialization operation, name, and leading
        dimension alignment of the input or specified parameters.
    :rtype: ArrayNode
    """
    torder = _gao(arr)
    nr = order == "A" or order is None
    # If our original array is discontinuous along an axis.
    if torder == "A":
        # assume it's either the front dimension (if C), or back dimension (if F)
        # use the *axis ordered* strides to figure out the actual backing array format.
        torder, bshape = _sao(arr)
        # No support yet for multiple discontinuous dimensions.
        # If caller forces incompatible order, drop inferred alignment.
        if not (nr or order == torder) and align_ldim is UO:
            align_ldim = None
    # Otherwise it's not discontinuous so the leading dimension doesn't have
    # alignment from the array (but we can still request it).
    # Otherwise, base array is contiguous so it doesn't carry alignment info by default.
    # If caller didn't request alignment explicitly, default to None for contiguous arrays.
    elif align_ldim is UO:
        align_ldim = None
    # A is not a real alignment type we could also use None, but its a little more readable.
    # If we didn't specify an order that is F or C
    # If order is not explicitly C/F, inherit it from the input array.
    # If caller didn't specify explicit C/F order, resolve 'A'/None to the array's order.
    if nr:
        order = torder
    # if (have a specified shape) and not (shapes are equal), but align_ldim is
    # UO, we will just assume we shouldn't use original alignment.
    # If caller forces a different logical shape, don't inherit alignment unless explicitly requested.
    # If requested shape differs from arr.shape, drop inherited align_ldim unless explicitly set.
    if not (shape is None or shape == arr.shape) and align_ldim is UO:
        align_ldim = None
    # but if no specified shape (will use original) or shapes are equal, assume
    # we can use base dimensions of array for new base.
    # If alignment remains undecided, inherit base backing shape from discontinuous array path.
    # If still undecided, inherit the backing shape (only set in discontinuous path).
    if align_ldim is UO:
        align_ldim = bshape  # type: ignore[unbound-name]
    # we don't have to worry about bshape not existing, because we know if bshape
    # is NA then arr is C or F contiguous and align_ldim=None from prev set.
    return _aarspec(arr, shape, dtype, order, init_op, name, align_ldim)


def _aarspec(
    arr: np.ndarray,
    shape: ShapeInput | None,
    dtype: DTypeLike | None,
    order: Any,
    init_op: InitOp | UseOther,
    name: str | None,
    align_ldim: Any,
) -> ArrayNode:
    # At this point order must be concrete and align_ldim cannot be UO.
    assert isinstance(order, str)
    assert not isinstance(align_ldim, UseOther)
    # Default missing shape/dtype from the input array to preserve semantics.
    sshape = arr.shape if shape is None else shape
    ddtype = arr.dtype if dtype is None else dtype
    # If init_op is UO, use the original array as init source; otherwise use caller-provided initializer.
    iinit = arr if isinstance(init_op, UseOther) else init_op
    return ar_spec(sshape, ddtype, order, iinit, name, align_ldim)


def ft_arspec(
    arspec: ArrayNode,
    shape: ShapeInput | None = None,
    dtype: DTypeLike | None = None,
    order: str | None = None,
    init_op: InitOp | UseOther = UO,
    name: str | None = None,
    align_ldim: ShapeInput | UseOther = UO,
) -> ArrayNode:
    """Return a **transposed** Array spec without changing memory layout.

    Swaps C↔F logical order, flips shapes/``bshape``, and carries over
    ``init_op`` (transposed if ndarray, or via a wrapper if callable).
    """
    rs = arspec
    rop = rs.init_op
    if isinstance(rop, np.ndarray):
        top = rop.T
    elif callable(rop):
        top = lambda ar: rop(ar.T)  # noqa: E731
    else:
        top = rop

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


def db_node(*args, name=None, no_merge=False, align=True) -> ContainerNode:
    """Create a distinct-region :class:`ContainerNode` (children are
    concatenated).

    ``align=True`` enables per-child alignment so the concatenation can
    exactly match an external memory layout.
    """
    return ContainerNode(*args, rule=BufferMap.DISTINCT, name=name, no_merge=no_merge, align=align)


# 7) DOT export
def _dtype_label(dtype: np.dtype | sym.Expr | sym.Symbol) -> str:
    # If dtype is concrete, abbreviate it.
    if isinstance(dtype, np.dtype):
        return dtype_abbr(dtype)
    # If dtype is symbolic expression, stringify via gen_str.
    if isinstance(dtype, sym.Expr):
        return gen_str(dtype)
    # Otherwise fall back to str() (e.g. sym.Symbol).
    return str(dtype)


def _dot_node_attr(node: ENode, *, with_offsets: bool) -> str:
    """Generate DOT attributes with name inside node and other info as
    external."""
    # Secondary info for external label (xlabel)
    info_parts: List[str] = []
    col = 'color="white"'
    # If node has a label, include it in the displayed DOT label.
    if node.label:
        info_parts.append(f"- {node.label} -")
    # If node is an ArrayNode, include shape/dtype details.
    if isinstance(node, ArrayNode):
        dtype_label = _dtype_label(node.dtype)
        # If align_ldim is enabled, show both logical and backing shapes.
        if node.align_ldim is not None:
            info_parts.append(f"{node.shape}, {node.bshape}, {dtype_label}")
        # Otherwise, show only logical shape and dtype.
        else:
            info_parts.append(f"{node.shape}, {dtype_label}")
        col = 'color="lime"'
    # If node is a ValueNode, show a short repr of its value.
    elif isinstance(node, ValueNode):
        vt = repr(node.value)
        qrg = 18
        vt = vt[:qrg]
        if qrg <= len(vt):
            vt = vt[:10] + "..."
        info_parts.append(vt)
        col = 'color="pink"'
    # If node is a DISTINCT container, color it orange.
    elif node.rule == BufferMap.DISTINCT:
        col = 'color="orange"'  # bc it can only be a ContainerNode by this conditional.
    # Otherwise it is a SHARED container, color it cyan.
    else:
        col = 'color="cyan"'  # it is shared
    # If with_offsets is enabled and ofs is known, include [ofs, end) range.
    if with_offsets and node.ofs is not None:
        # If nbytes is symbolic, display the expression; otherwise compute concrete end.
        if isinstance(node.nbytes, sym.Expr):
            end = f"{node.ofs}+{gen_str(node.nbytes)}"
        else:
            end = node.ofs + node.nbytes
        info_parts.append(f"[{node.ofs}, {end})")
    # If node has an nbytes field, show size summary.
    if hasattr(node.nbytes, "nbytes"):
        # If symbolic, show raw bytes expression string.
        if isinstance(node.nbytes, sym.Expr):
            info_parts.append(f"{gen_str(node.nbytes)} B")
        # Otherwise, show KB summary for human readability.
        else:
            info_parts.append(f"{node.nbytes / 1024:.2f} KB")  # maybe change to
    # Build attribute list
    label = "\n".join(info_parts)
    parts: List[str] = [f'label="{label}"', 'shape="circle"', 'labelloc="c"', col]
    # parts.append('style="filled"')
    # parts.append('fillcolor="white"')
    return ", ".join(parts)


def bmap_todot(bmap: BaseNode, *, with_offsets: bool = True) -> str:
    """Serialize the tree to Graphviz DOT via
    ``anytree.exporter.DotExporter``."""
    lines: list[str] = []
    for line in DotExporter(
        bmap,
        nodenamefunc=lambda n: f"n_{n.id}",
        nodeattrfunc=lambda n: _dot_node_attr(n, with_offsets=with_offsets),
    ):
        lines.append(line)
    return "\n".join(lines)


# 8) PyVis visualiser
def _clean_field(node: dict[str, Any], key: str) -> None:
    val = node.get(key)
    # If the field is a quoted DOT string, strip embedded quotes for display.
    if isinstance(val, str) and val.startswith('"') and val.endswith('"'):
        node[key] = val.strip('"')


def _clean_nodes(nodes: list[dict[str, Any]]) -> None:
    # Normalize all node labels/colors from pydot/pyvis which may embed quotes.
    for node in nodes:
        _clean_field(node, "label")
        _clean_field(node, "color")


def _bmap_pyvis(src: Union[str, BaseNode], *, with_offsets: bool = False) -> Network:  # pragma: no cover
    """Internal helper that returns a configured ``pyvis.Network`` graph.

    Accepts either a DOT string or a ``ContainerNode``; when given a node,
    converts via ``bmap_todot`` then builds a NetworkX graph using ``pydot``.
    """
    # Optional deps: keep ImportError path for environments without pyvis.
    try:
        import networkx as nx  # type: ignore[missing-import]
        import pydot  # type: ignore[missing-import]
        from pyvis.network import Network  # type: ignore[untyped-import]
    except ImportError as e:
        raise ImportError("pyvis, networkx, pydot required for bmap_pyvis()") from e
    # If src is already DOT, use it as-is.
    if isinstance(src, str):
        dot_data = src
    # If src is a buffer-map node, serialize it to DOT first.
    elif isinstance(src, BaseNode):
        dot_data = bmap_todot(src, with_offsets=with_offsets)
    # Otherwise, refuse unsupported inputs.
    else:
        raise TypeError("src must be DOT str or ContainerNode")
    graphs = pydot.graph_from_dot_data(dot_data)
    # If pydot couldn't parse, fail fast with a clear error.
    if not graphs:
        raise ValueError("Failed to parse DOT data")
    pgraph = graphs[0]
    g_nx: Any = nx.drawing.nx_pydot.from_pydot(pgraph)
    net = Network(directed=True)
    net.from_nx(g_nx)
    # Strip embedded quotes from labels and colors
    _clean_nodes(net.nodes)
    net.set_options(
        """{
  "layout": {
    "hierarchical": {
      "enabled": true,
      "direction": "UD",
      "sortMethod": "hubsize",
      "blockShifting": true,
      "edgeMinimization": true
    }
  },
  "physics": {"enabled": true}
}"""
    )
    return net


BMAP_ROPTS2 = """{
      "layout": {
        "hierarchical": {
          "enabled": false,
          "direction": "UD",
          "sortMethod": "hubsize",
          "blockShifting": true,
          "edgeMinimization": true
        }
      },
      "physics": {
        "enabled": true,
        "wind": { "x": 0.0, "y": 0 },
        "repulsion": { "nodeDistance": 200, "springLength": 200 },
        "centralGravity": 0.1
      }
    }"""

BMAP_ROPTS = """{
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "UD",
          "sortMethod": "hubsize",
          "blockShifting": true,
          "edgeMinimization": true
        }
      },
      "physics": {
        "enabled": true,
        "wind": { "x": 0.02, "y": 0 },
        "hierarchicalRepulsion": { "nodeDistance": 55, "avoidOverlap": 1 }
      }
    }"""

_bopt = """{
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "UD",
          "sortMethod": "hubsize",
          "blockShifting": true,
          "edgeMinimization": true
        }
      },
      "physics": {
        "enabled": true,
        "wind": { "x": 0.02, "y": 0 },
        "hierarchicalRepulsion": { "nodeDistance": 67, "avoidOverlap": 1 }
      }
    }"""


# 8) PyVis visualiser
def bmap_pyvis(
    src: Union[str, ContainerNode],
    with_offsets: bool = False,
    height: str = "1000px",
    width: str = "100%",
    render_options: str = BMAP_ROPTS,
) -> Network:
    """Build a PyVis interactive tree using a top-down layout and physics
    enabled."""
    # Optional deps: keep ImportError path for environments without pyvis.
    try:
        import networkx as nx  # type: ignore[missing-import]
        import pydot  # type: ignore[missing-import]
        from pyvis.network import Network  # type: ignore[untyped-import]
    except ImportError as e:
        raise ImportError("pyvis, networkx, pydot required for bmap_pyvis()") from e

    # Obtain DOT source and parse
    dot_data = src if isinstance(src, str) else bmap_todot(src, with_offsets=with_offsets)
    graphs = pydot.graph_from_dot_data(dot_data)
    # DOT parse must return at least one graph.
    if not graphs:
        raise ValueError("Failed to parse DOT data")
    # print(graphs)
    pgraph = graphs[0]
    # Convert to NetworkX
    g_nx: Any = nx.drawing.nx_pydot.from_pydot(pgraph)
    # Insert a blank white dummy sibling under the root container to anchor hubsize
    # Determine root: if ContainerNode passed, use its name; else take first node

    nodes = list(g_nx.nodes)
    # Pick a root id when graph has nodes; otherwise root_id stays None.
    root_id = nodes[0] if nodes else None
    dummy = 0
    # Note: For some reason we need a dummy node otherwise the top level node
    # will be out of place; try the old _bmap_pyvis to see strange behavior.
    # Ensure a dummy anchor node exists to stabilize hierarchical layout.
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
    # Remove existing root edges so we can reinsert in a preferred order.
    for v in succ:
        g_nx.remove_edge(root_id, v)
    # Re-add with dummy first, then self-loop, then the rest
    new_order = [dummy] + [v for v in succ if v not in (dummy, root_id)]
    # Re-add root edges so dummy comes first, then remaining children.
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
    _clean_nodes(net.nodes)

    # If default options + offsets, swap to bootleg config to keep layout stable.
    if render_options == BMAP_ROPTS and with_offsets:
        render_options = _bopt  # bootleg? Yes
    # render opts
    net.set_options(render_options)

    return net


def save_bmap_tree(bmap: NodeMixin, path: str = "bmap_tree.png", *, with_offsets: bool = False) -> str:
    """Render the tree to a static image using Graphviz and return ``path``."""
    DotExporter(
        bmap,
        nodenamefunc=lambda n: n.id,
        nodeattrfunc=lambda n: _dot_node_attr(n, with_offsets=with_offsets),
    ).to_picture(path)
    return path


def clone_bmap(bmap: NodeMixin) -> NodeMixin:
    """Deep-clone an entire buffer-map tree (labels, sizes, offsets, etc.)."""
    return copy.deepcopy(bmap)
