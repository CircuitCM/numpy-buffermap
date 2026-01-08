# ruff: noqa: E501
# from collections.abc import Collection
# from functools import cache

import math as mt
from itertools import chain
from numbers import Number
from types import NoneType
from typing import Optional, Sequence, TypeAlias, Union

import numpy as np
import sympy as sym
from anytree import DoubleStyle, LightNodeMixin, NodeMixin, RenderTree

from bmap._util import (
    _ESET,
    BufferAlign,
    BufferMap,
    BuffExprMaybe,
    DTypeLike,
    InitOp,
    ShapeLike,
    SizeExpr,
    SizeSeq,
    _IDGen,
    buffer_expr,
    c_orlen,
    dt_buff_exprs,
    eval_buff_expr,
    eval_buff_exprs,
    gen_str,
    roundup,
)

SymDef: TypeAlias = BuffExprMaybe | Sequence[BuffExprMaybe] | tuple[SizeExpr | None, SizeSeq, SizeSeq]


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

    id: int
    label: Optional[str]
    no_merge: bool
    nbytes: SizeExpr | None
    ofs: Optional[int]
    buffer: np.ndarray | None

    def __init__(
        self,
        name: Optional[Union[str, int, float]] = None,
        no_merge: bool = False,
    ) -> None:
        self.id = _IDGen.new_id()
        self.label = str(name) if name is not None else None
        self.no_merge = bool(no_merge)
        self.nbytes = None
        self.ofs = None
        # self.buffer: Optional[np.ndarray] = None

    def _short(self) -> str:
        return self.label or str(self.id)

    @property
    def name(self) -> str | None:
        return self.label

    @name.setter
    def name(self, name) -> None:
        self.label = name

    @name.deleter
    def name(self) -> None:
        del self.label

    def gen_call(self) -> tuple[str | tuple[str, ...] | None, str | int | float | None]:
        """This was previously called gen_def, now it defines it's string definition, based on certain modified dependents.

        This will emit the defining string of the representation in the code defining generator.

        The defining idea of the writer, is that if we start at the top of the tree. Then walk down the
        tree depth first, adding gen_defs to a list of strings on first visitation to a node, we should be able correctly
        define all buffer sub-indices offsets and array definitions/dependencies accurately.
        """
        # subn : sub-name for use in extending the name of the parameter eg containername_varname
        istr = "#The assignment/initializer codeline."
        rstr = "#The return statement/reference within the return line."
        return istr, rstr

    def sym_def(self) -> SymDef:
        """Returns the sequence of node objects that are dependent on the node's procedural generation and simplification by cse in sympy."""
        raise NotImplementedError

    @property
    def free_symbols(self) -> set[sym.Basic]:
        """If sympy is installed this is an api to return all used symbols within the node."""
        return _ESET


class ValueNode(BaseNode):
    """Leaf node carrying a Python value that does **not** occupy buffer space.

    Useful for threading literal parameters through the tree so that
    ``flatten_items`` and friends preserve order alongside arrays.
    """

    def __init__(
        self,
        value: int | str | sym.Expr | None,
        name: Optional[Union[str, int, float]] = None,
    ) -> None:
        super().__init__(name=name, no_merge=True)
        self.value = buffer_expr(value, safe=False)

    value: BuffExprMaybe

    @property
    def vtype(self) -> type[sym.Expr] | type[NoneType] | type[sym.Symbol] | type[int]:
        return type(self.value)

    def __repr__(self) -> str:
        return f"|Value {self._short()} {repr(self.value)[:50]} {self.vtype}|"

    def gen_call(
        self,
        valexpr: BuffExprMaybe | None = None,
        subn: str | None = None,
    ) -> tuple[str | None, str | int | None]:
        if valexpr is None:
            valexpr = self.value
        vg = gen_str(valexpr) if isinstance(valexpr, sym.Expr) else valexpr
        if self.name:
            rstr = f"{subn}_{self.name}" if subn else self.name
            istr = f"{rstr} = {vg}"
        else:
            istr = None
            rstr = vg  # we don't assign and go straight to the return
        return istr, rstr

    def sym_def(self) -> BuffExprMaybe:
        return self.value

    @property
    def free_symbols(self) -> set[sym.Basic]:
        """If sympy is installed this is an api to return all used symbols within the node."""
        if isinstance(self.value, sym.Expr):
            return self.value.free_symbols
        return _ESET


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

    shape: Sequence[SizeExpr]
    dtype: np.dtype | sym.Expr
    bshape: Sequence[SizeExpr]
    array: Optional[np.ndarray]
    barray: Optional[np.ndarray]
    align_ldim: SizeExpr | Sequence[SizeExpr] | None
    itsize: SizeExpr
    order: str
    init_op: InitOp | None

    def __init__(
        self,
        shape: ShapeLike = (0,),
        dtype: DTypeLike = np.float64,  # type: ignore[bad-function-definition]
        order: str = "C",
        init_op: InitOp | None = None,
        name: Optional[Union[str, int, float]] = None,
        align_ldim: ShapeLike | None = None,
    ) -> None:
        super().__init__(name=name, no_merge=True)
        shape_seq = dt_buff_exprs(shape if isinstance(shape, Sequence) and not isinstance(shape, str) else (shape,))
        shape_tuple = shape_seq if isinstance(shape_seq, tuple) else (shape_seq,)
        self.shape = shape_tuple
        self.array = None
        self.barray = None
        self.align_ldim = dt_buff_exprs(align_ldim)
        if isinstance(dtype, type) and issubclass(dtype, (np.generic, Number)):
            dtype = np.dtype(dtype)
        if isinstance(dtype, np.generic):
            dtype = np.dtype(dtype)
        if isinstance(dtype, np.dtype):
            self.dtype = dtype  # np.dtype(dtype)
            self.itsize = dtype.itemsize
        else:
            assert isinstance(dtype, (sym.Expr, int, str))
            syty = buffer_expr(dtype)
            assert isinstance(syty, sym.Symbol)
            syty_sym = syty
            if c_orlen(syty_sym.name, "type_"):
                self.dtype = syty_sym
                itsize_expr = buffer_expr(syty_sym.name[5:])
                assert itsize_expr is not None
                self.itsize = itsize_expr
            else:
                print(
                    f"Note: {dtype} is specified without a `type_` field.\nTo work in the allocation code generator make sure the var name is `type_{dtype}`."
                )
                dtype_expr = buffer_expr(f"type_{syty_sym.name}")
                assert isinstance(dtype_expr, sym.Expr)
                self.dtype = dtype_expr
                self.itsize = syty_sym
        self.order = order.upper()
        self.init_op = init_op
        if isinstance(self.align_ldim, tuple):
            self.bshape = self.align_ldim
        elif self.align_ldim is not None:
            # otherwise we can't be certain about if align
            print(self.align_ldim, self.itsize)
            rsz = self.align_ldim // self.itsize  # type: ignore[unsupported-operation]
            if self.order == "C":
                self.bshape = (*shape_tuple[:-1], roundup(shape_tuple[-1], rsz))
            else:
                self.bshape = (roundup(shape_tuple[0], rsz), *shape_tuple[1:])
        else:
            self.bshape = shape_tuple
        # ... this must have been bshape intended? Check later
        # self.nbytes: int = int(np.prod(self.shape)) * self.dtype.itemsize
        # likely
        self.nbytes = mt.prod(self.bshape) * self.itsize  # type: ignore[no-matching-overload]

    def __repr__(self) -> str:
        shp = ", ".join(map(str, self.shape))
        if self.shape != self.bshape:
            bshp = ", ".join(map(str, self.bshape))
            return f"|Array {self._short()} ({bshp}) ({shp}) {self.dtype}|"
        return f"|Array {self._short()} ({shp}) {self.dtype}|"

    def _rinit(self) -> None:
        """Apply ``init_op`` to ``self.array`` if provided.

        - If ``init_op`` is callable, it is invoked with the array for in-place
          initialization.
        - Otherwise, the array is filled/broadcast-assigned with the value.
        """
        if self.array is None:
            return
        if self.init_op is not None:
            if callable(self.init_op):
                self.init_op(self.array)
            else:
                self.array[...] = self.init_op

    def mk_array(
        self,
        buffer: Optional[np.ndarray] = None,
        sym_dic: dict[sym.Symbol, int] | None = None,
    ) -> np.ndarray:
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
        y = False
        if y := (buffer is None and self.array is None):
            bshape = tuple(int(v) for v in eval_buff_exprs(self.bshape, sym_dic))
            dtype = self.dtype if isinstance(self.dtype, np.dtype) else eval_buff_expr(self.dtype, sym_dic)
            self.array = np.empty(bshape, dtype=dtype, order=self.order)  # type: ignore[no-matching-overload]
        elif y := (buffer is not None):
            bshape = tuple(int(v) for v in eval_buff_exprs(self.bshape, sym_dic))
            dtype = self.dtype if isinstance(self.dtype, np.dtype) else eval_buff_expr(self.dtype, sym_dic)
            cnt = int(mt.prod(bshape))
            view = np.frombuffer(  # type: ignore[no-matching-overload]
                buffer,
                dtype=dtype,
                count=cnt,
                offset=0 if self.ofs is None else self.ofs,
            )
            view = view.reshape(bshape, order=self.order)
            self.array = view
        if y and self.align_ldim is not None:
            shape = tuple(int(v) for v in eval_buff_exprs(self.shape, sym_dic))
            assert self.array is not None
            self.barray = self.array
            self.array = self.barray[..., : shape[-1]] if self.order == "C" else self.barray[: shape[0], ...]
        self._rinit()
        assert self.array is not None
        return self.array

    @property
    def value(self) -> np.ndarray | None:
        return self.array

    @staticmethod
    def array_genc(offstr: str, dtstr: str, dmstr: str, ord: str) -> str:
        return f"fb_(buffer[:{offstr}],{dtstr}).reshape({dmstr}){'' if ord == 'C' else '.T'}"

    npspec = "np."

    def gen_call(
        self,
        bytexpr: SizeExpr | None = None,
        bshape: SizeSeq | None = None,
        shape: SizeSeq | None = None,
        subn: str | None = None,
    ) -> tuple[str, str]:
        """Generate string definition of array from symbolics or value."""
        if not subn and not self.name:
            raise NameError("Arrays without name or sub name are not supported.")
        st = self._short()  # name first, if fails falls back to an integer counter (not related to container position)
        rstr = f"{subn}_{st}" if subn else st
        if bytexpr is None:
            bytexpr = self.nbytes
        assert bytexpr is not None
        ols = gen_str(bytexpr)
        ds = gen_str(self.dtype) if isinstance(self.dtype, sym.Expr) else self.npspec + self.dtype.name

        bshape_seq = self.bshape if bshape is None else bshape
        bshape_tuple = tuple(bshape_seq)
        dms = gen_str(tuple(bshape_tuple[:: (1 if self.order == "C" else -1)]))
        istr = self.array_genc(ols, ds, dms, self.order)
        istr = f"{rstr} = {istr}"
        if self.align_ldim is not None:
            shape_seq = self.shape if shape is None else shape
            shape_tuple = tuple(shape_seq)
            istr += (
                f"[{f'...,:{gen_str(shape_tuple[-1])}' if self.order == 'C' else f':{gen_str(shape_tuple[0])},...'}]"
            )
            # adding \t here seems a bit informal, given we don't know how gen_def will be utilized outside of this method.
            # so do it at the function formatting scope.
        return istr, rstr

    def sym_def(self) -> tuple[SizeExpr, SizeSeq, SizeSeq]:
        assert self.nbytes is not None
        return self.nbytes, self.bshape, self.shape

    @property
    def free_symbols(self) -> set[sym.Basic]:
        """If sympy is installed this is an api to return all used symbols within the node."""
        fs: set[sym.Basic] = set()
        # itsize=symbol_ref is a special cased derived from type_{symbol_ref}'s so it won't go in the function header
        # and gets handled by the allocation handler.
        align_seq = (
            self.align_ldim
            if isinstance(self.align_ldim, Sequence) and not isinstance(self.align_ldim, sym.Expr)
            else None
        )
        for v in self.shape if align_seq is None else chain(self.shape, align_seq):
            if isinstance(v, sym.Expr):
                if v.is_symbol:
                    fs.add(v)
                else:
                    fs.update(v.free_symbols)
        if isinstance(v := self.align_ldim, sym.Expr):
            if v.is_symbol:
                fs.add(v)
            else:
                fs.update(v.free_symbols)
        if isinstance(v := self.dtype, sym.Expr):
            if v.is_symbol:
                fs.add(v)
            else:
                fs.update(v.free_symbols)

        # it's probably a decent bit slower to just call self.nbytes.free_symbols. that also won't include the dtype but will include self.itsize which we dont want

        return fs


class ContainerNode(BaseNode):
    """Internal node controlling how children share backing memory.

    Rules
    -----
    - ``DISTINCT``: children are concatenated in order; each child's region is
      rounded up to the global alignment unless ``align=True``, in which case
      only the *container's* total is rounded once.
    - ``SHARED``: all children map to the same region whose size equals the
      maximum child size, rounded up to alignment.

    Flags
    -----
    no_merge : bool
        Prevents being merged with a like-ruled parent/child during reduction.
    align : bool
        Only for ``DISTINCT``: enables per-child rounding so contiguous regions
        can exactly match external memory layouts.
    """

    # SYM_EQN=MulSym
    rule: int
    align: bool
    aligned_eqns: list[SizeExpr]

    def __init__(
        self,
        *children: "BaseNode",
        rule: int = BufferMap.DISTINCT,
        name: Optional[Union[str, int, float]] = None,
        no_merge: bool = False,
        align: bool = True,
    ) -> None:
        super().__init__(name=name, no_merge=no_merge)
        self.rule = int(rule)
        self.align = align and rule == BufferMap.DISTINCT  # only effective for distinct rule nodes.
        self.aligned_eqns = []
        self.add(*children)

    def add(self, *kids: "BaseNode") -> "ContainerNode":
        """Append children to this container and set their parent pointer."""
        for k in kids:
            if not isinstance(k, (NodeMixin, LightNodeMixin)):
                print(k)
                raise TypeError("Children must be NodeMixin instances")
            k.parent = self
        return self

    def insert(self, index, *kids):
        """Insert one or more children at a specific position.

        Existing instances of the same nodes are first removed to avoid
        duplicates, then the desired order is established.
        Returns ``self`` for chaining.
        #"""
        # for k in kids: #assigning to self.children already does a nodemixin check
        #     if not isinstance(k, BaseNode):
        #         raise TypeError("Children must be BaseNode instances")
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
        children: list[BaseNode] = list(self.children)
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
        children: list[BaseNode] = list(self.children)
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
            node: BaseNode = children.pop(pos)
            # Detach from this container
            node.parent = None
        # Update children tuple
        self.children = tuple(children)
        return popped

    def build_flatmap(
        self,
        align: SizeExpr = BufferAlign.AVX512,
        alignb: SizeExpr = BufferAlign.PAGE,
        name_join: bool = True,
        force_merge: bool = False,
        verbose: bool = False,
    ):
        """Build, allocate, and return the initialized leaf values in preorder.

        Convenience wrapper around ``build_bmap`` → ``allocate_bmap`` →
        ``flat_inits``. See those functions for details.
        """
        pass  # will finish later
        from bmap._procedures import build_bmap, flat_inits

        build_bmap(
            self,
            align=align,
            name_join=name_join,
            force_merge=force_merge,
            verbose=verbose,
        )
        # allocate_bmap(self, align=align,alignb=alignb)
        return flat_inits(self)

    @staticmethod
    def s_def(syq: sym.Expr | int | object | None) -> str:
        ols = gen_str(syq)
        return f"buffer = buffer[{ols}:]"

    def gen_call(
        self,
        setexpr: SizeExpr | SizeSeq | None = None,
    ) -> tuple[str, None] | tuple[tuple[str, ...], None]:
        """See BaseNode docstring."""
        if self.align:
            if setexpr is None:
                setexpr_seq = self.aligned_eqns
            elif isinstance(setexpr, Sequence) and not isinstance(setexpr, sym.Expr):
                setexpr_seq = setexpr
            else:
                setexpr_seq = (setexpr,)
            return (*(self.s_def(s) for s in setexpr_seq),), None
        if setexpr is None:
            setexpr = self.nbytes
        return self.s_def(setexpr), None

    def sym_def(self) -> SizeExpr | SizeSeq | None:
        return self.aligned_eqns if self.align else self.nbytes

    def __repr__(self) -> str:
        return f"|Node {self._short()} rule={'S' if self.rule == BufferMap.SHARED else 'D'} children={len(self.children)} {f'nbytes={self.nbytes}' if hasattr(self, 'nbytes') else ''}|"

    def __str__(self) -> str:
        v = str(RenderTree(self, style=DoubleStyle))
        return v

    @property
    def free_symbols(self) -> set[sym.Basic]:
        """If sympy is installed this is an api to return all used symbols within the node."""
        # it should be necessary and a little slower to call free_symbols on the container, than directly on value or array nodes.
        # which is why the build buffer map function only calls array/value nodes, they will also contain all symbols used on the graph there.
        if self.aligned_eqns:
            return set().union(*(eq.free_symbols for eq in self.aligned_eqns if isinstance(eq, sym.Expr)))
        elif (
            self.nbytes is not None
        ):  # assume nbytes has been simplified, so recursion to a large tree of nodes could be slower than the nbytes expr check. (post simplify)
            if isinstance(self.nbytes, sym.Expr):
                return self.nbytes.free_symbols
            else:
                return _ESET
        elif len(cs := self.children) > 0:
            return set().union(*(c.free_symbols for c in cs))
        else:
            return _ESET


ItemNodeT: TypeAlias = ValueNode | ArrayNode
