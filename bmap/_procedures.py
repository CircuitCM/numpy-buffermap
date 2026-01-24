import copy
from collections.abc import Collection
from functools import partial
from typing import Any, Callable, List, Sequence, Tuple, TypeGuard, Union, cast

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
    InitParam,
    ShapeInput,
    ShapeMaybe,
    ShapeParam,
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
    Builds the string definition of a function that dynamically allocates the arrays and values of a buffer map.

    :param buffer_map: Original buffer map.
    :param balign: The initial alignment of the buffer. It must be greater than or equal to the alignment you choose to build the buffer_map with.
    """
    # make sure sym_dic is using symbols, so for user it's ok to specify them with string keys
    args = [*((k.name if isinstance(k, sym.Symbol) else k) for k in args),]
    args.sort(key=lambda x: c_orlen(x, "type_"),)
    #only symbols not expressions supported now, all expressions should be evaluated into needed equations.
    kwargs ={(k.name if isinstance(k, sym.Symbol) else k): v for k, v in kwargs.items()} if kwargs is not None else {}
    # for now there is also no dead-parameter/dead-code removal, and no checks for if all used input parameters are referenced in the header
    if chkforbuffer:
        hd = BButil.build_header(buffer_map, args, kwargs, balign=balign)
        assert buffer_map.nbytes is not None
        allq: list[SizeParam] = [buffer_map.nbytes]
        ct = [1]
    else:
        hd = BButil.build_header(buffer_map, args, kwargs, check_alloc=None, balign=balign)
        allq = []
        ct = [0]

    if fullreduce:
        allq.extend(_cxprs(buffer_map))  # already 'least expressions' form.
        if buffer_map.rule == BufferMap.DISTINCT: allq.pop()  # if initial buffer node is distinct we don't need the
        lyrs, reduc = cse_codereduction(allq, tempvar)
        strl = ls_layers(lyrs)
        allc = (BButil.add_balloc(reduc[0]),) if chkforbuffer else ()
        mkls = _bb2(buffer_map, reduc, subname=subname, ct=ct)
        mdst = "\n".join(mkls[0])
        mdst = mdst.replace("\n", "\n    ")  # in case there are multi-line statements
        estr = f"return {', '.join(str(v) for v in mkls[1])}"
    else:
        # no reduction is quicker but also much more ugly and less readable.
        mkls = _bba(buffer_map, subname=subname)
        if buffer_map.rule == BufferMap.DISTINCT: mkls[0].pop()
        strl = ()
        allc = (BButil.add_balloc(allq[0]),) if chkforbuffer else ()
        mdst = "\n".join(mkls[0])
        mdst = mdst.replace("\n", "\n    ")  # in case there are multi-line statements
        estr = f"return {', '.join(str(v) for v in mkls[1])}"

    fullf = "\n    ".join((hd, *strl, *allc, mdst, "", estr))
    return fullf


def _is_shape_tuple(value: tuple[int, ...] | tuple[str, tuple[int, ...]],) -> TypeGuard[tuple[int, ...]]: 
    return not (value and isinstance(value[0], str))


def _cxprs(bn: BaseNode, excs: list[sym.Expr] | None = None) -> list[sym.Expr]:
    wn = excs is None
    if wn: excs = []
    if isinstance(bn, ContainerNode) and bn.rule == BufferMap.DISTINCT:
        if bn.align:
            for c, eq in zip(bn.children, bn.aligned_eqns, strict=False):
                _cxprs(c, excs)
                _pxpr(excs, eq)
        else:
            for c in bn.children:
                _cxprs(c, excs)
                _pxpr(excs, c.nbytes)
    else:
        for c in bn.children: _cxprs(c, excs)
        if isinstance(bn, ArrayNode): _arrpxpr(excs, bn.sym_def())
        elif isinstance(bn, ValueNode): _pxpr(excs, bn.sym_def())

    return excs


def _bb2(
    bnode: ENode,
    exls: SizeSeq,
    subname: bool = False,
    sn: str | None = None,
    ct: list[int] | None = None,
    mkls: tuple[list[str], list[str | int | float]] | None = None,
) -> tuple[list[str], list[str | int | float]]:
    is_root = mkls is None
    if mkls is None: mkls = ([], [])
    if ct is None: ct = [0]
    if isinstance(bnode, ContainerNode):
        if bnode.rule == BufferMap.DISTINCT:
            mkls[0].append("")
            bc = bnode.children
            bcl = len(bc) - 1
            if bnode.align:
                eqs = bnode.aligned_eqns
                for i, (c, eq) in enumerate(zip(bc, eqs, strict=False)):
                    _bb2(c, exls, subname, bnode.name, ct, mkls)
                    # so we don't run into exls that is too far extended
                    if is_root and i == bcl: break
                    ot = _bxpr(exls, eq, ct)
                    mkls[0].append(ContainerNode.s_def(ot))
            else:
                for i, c in enumerate(bc):
                    _bb2(c, exls, subname, bnode.name, ct, mkls)
                    if is_root and i == bcl: break
                    ot = _bxpr(exls, c.nbytes, ct)
                    mkls[0].append(ContainerNode.s_def(ot))
        else:
            mkls[0].append("")
            for c in bnode.children: _bb2(c, exls, subname, bnode.name, ct, mkls)
    elif isinstance(bnode, ArrayNode):
            bytexpr, bshapet, shapet = _arrbxpr(exls, bnode.sym_def(), ct)
            s0,s1 = bnode.gen_call(
                bytexpr=bytexpr,
                bshape=bshapet,
                shape=shapet,
                subn=sn if subname else None,
            )
            mkls[0].append(s0); mkls[1].append(s1)
    else:
        #assume it's something like a ValueNode that will succeed, or we just let it fail.
        s0,s1 = bnode.gen_call(
            subn=sn if subname else None,
        )
        mkls[0].append(s0); mkls[1].append(s1) #type: ignore[bad-argument-type]

    return mkls


def _bba(
    bnode: BaseNode,
    mkls: tuple[list[str], list[str | int | float]] | None = None,
    subname: bool = False,
    sn: str | None = None,
) -> tuple[list[str], list[str | int | float]]:
    if mkls is None: mkls = ([], [])
    if isinstance(bnode, ContainerNode):
        mkls[0].append("")
        if bnode.rule == BufferMap.DISTINCT:
            if bnode.align:
                defs, _ = bnode.gen_call()
                for c, eq in zip(bnode.children, defs, strict=False):
                    _bba(c, mkls, subname, bnode.name)
                    mkls[0].append(eq)
            else:
                for c in bnode.children:
                    _bba(c, mkls, subname, bnode.name)
                    mkls[0].append(ContainerNode.s_def(c.nbytes))
        else: 
            for c in bnode.children: _bba(c, mkls, subname, bnode.name)
    elif isinstance(bnode, ArrayNode): 
        s0,s1 = bnode.gen_call(subn=sn if subname else None)
        mkls[0].append(s0);mkls[1].append(s1)
    else:
        assert isinstance(bnode, ValueNode)
        s0,s1 = bnode.gen_call(subn=sn if subname else None)
        mkls[0].append(s0);mkls[1].append(s1) #type: ignore[bad-argument-type]
    
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
    if verbose: check_bmap(bmap)
    return symbols #type: ignore[bad-return]


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
        for grand in child.children: grand.label = f"{parent.label}_{grand.label or ''}" if grand.label else parent.label
    for g in list(child.children): g.parent = parent
    child.parent = None


def reduce_bmap(bmap: ContainerNode, *, name_join: bool = True, force_merge: bool = False) -> ContainerNode:
    """Merge adjacent containers with the same rule.

    If ``force_merge`` is False, a ``no_merge=True`` on either container blocks
    the merge; otherwise it is ignored.
    """
    for node in list(PreOrderIter(bmap)):
        if not isinstance(node, ContainerNode): continue
        for ch in list(node.children): 
            if (
                isinstance(ch, ContainerNode)
                and node.rule == ch.rule
                and node.align == ch.align
                and (force_merge or (not node.no_merge and not ch.no_merge))
            ): _merge_child_into_parent(node, ch, name_join=name_join)
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
    if _symbols is None: _symbols = set()
    align_expr = buffer_expr(align)
    if isinstance(node, ItemNode):
        _symbols.update(s for s in node.free_symbols)
        return node.nbytes
    # compute child sizes
    assert isinstance(node, ContainerNode)
    if node.rule == BufferMap.SHARED:
        sizes = (*(_n_(ch, align, simplify, _symbols) for ch in node.children),)
        raw = gen_max(sizes, simplify) if sizes else 0
        # round up shared region
        node.nbytes = roundup(raw, align_expr)
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
    else:
        # if not align then nbytes will be rounded to match whole container alignment, but not children alignment. Which means that arraynodes that must be aligned contiguously can be forced to, if they represent the memory of another array exactly.
        # Which could be useful maybe when you have a shared node, one array, then a distinct sibling node where all
        # however if another container is a child and it's not at the beginning, this could mess up alignment for all nodes it contains.
        # so distinct nodes should have alignment by default unless there is a special case for all arrays.
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
    :param balign: The initial alignment of the buffer. It must be greater than or equal to the alignment you choose to build the buffer_map with.
    """
    # make sure sym_dic is using symbols, so for user it's ok to specify them with string keys
    if sym_dic is None: sym_dic = {}
    sym_map: dict[sym.Symbol, int] = {}
    for k, v in sym_dic.items():
        if isinstance(k, sym.Symbol): sym_map[k] = v
        else:
            sym_key = buffer_expr(k)
            sym_map[sym_key] = v
    # make a buffer if none was provided
    if _buffer is None: _buffer = aligned_buffer(eval_buff_expr(buffer_map.nbytes, sym_dic), balign)
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
    if sym_dic is None: sym_dic = {}
    node.ofs = base
    if isinstance(node, ArrayNode): node.mk_array(buffer, sym_dic)
    elif isinstance(node, ContainerNode):
        if node.rule == BufferMap.SHARED: 
            for ch in node.children: _generate_nodearrays(ch, buffer, base, sym_dic)
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
        if n.label: seen.setdefault(n.label, []).append(n)
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    if dup:
        print("Duplicate labels:")
        for k, v in dup.items():
            print(f"  {k} × {len(v)}")


# 6) Query helpers
def _walk_index(node: BaseNode, idx: Sequence[int]) -> BaseNode:
    """Follow a child-index path from ``node`` and return the target node."""
    cur = node
    for i in idx: cur = cur.children[i]  # type: ignore
    return cur


def bmap_get(bid: Union[Sequence[int], str, BaseNode], bmap: ContainerNode) -> BaseNode:
    """Resolve a node reference by path, label, or pass-through instance."""
    if isinstance(bid, BaseNode): return bid
    if isinstance(bid, (list, tuple)): return _walk_index(bmap, bid)
    if isinstance(bid, str):
        for n in PreOrderIter(bmap): 
            if n.label == bid: return n
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
    init_op: InitOp|UseOther = UO,
    name: str | None = None,
    align_ldim: ShapeMaybe|UseOther = UO,

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
    init_op: InitOp|UseOther = UO,
    name: str | None = None,
    align_ldim: ShapeMaybe|UseOther = UO,
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
    if not isinstance(rs, ArrayNode): raise ValueError("`f_arspec` only takes ArrayNode for `arspec`.")
    #do we keep the leading dimension alignment
    spc_align = _chkfalign(rs.shape, rs.order, shape, order, align_ldim)
    if isinstance(align_ldim, UseOther): align_ldim_val = None
    #we will use the new alignment where None overrides it to definitely no alignment, otherwise eg sympy expression,
    #dimension tuple, or byte ceiling will force new alignment, leaving it as UO uses arspecs alignment only if possible
    else: align_ldim_val = align_ldim
    #for non-optional args, simply get if inputs are None.
    sshape = rs.shape if shape is None else shape
    ddtype = rs.dtype if dtype is None else dtype
    oorder = rs.order if order is None else order
    #None has the same 'nothing callable' meaning as with align_ldim, UO uses other arspecs init_op.
    if isinstance(init_op, UseOther): iinit_op = rs.init_op
    else: iinit_op = init_op
    aalign_ldim = rs.bshape if spc_align else align_ldim_val
    #Name is the only parameter that doesn't fall back to arspec's name, as it wouldn't really make sense to use this 
    #function if that were the case.
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
    init_op: InitOp|UseOther = UO,
    name: str | None = None,
    align_ldim: ShapeMaybe|UseOther = UO,
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
    #If our original array is discontinuous along an axis.
    if torder == "A":
        #assume it's either the front dimension (if C), or back dimension (if F)
        #use the *axis ordered* strides to figure out the actual backing array format.
        torder, bshape = _sao(arr)
        #No support yet for multiple discontinuous dimensions.
        if not (nr or order == torder) and align_ldim is UO: align_ldim = None
    #Otherwise it's not discontinuous so the leading dimension doesn't have alignment from the array (but we can still request it)
    elif align_ldim is UO: align_ldim = None
    #A is not a real alignment type we could also use None, but its a little more readable.
    #If we didn't specify an order that is F or C
    if nr:order = torder
    #if (have a specified shape) and not (shapes are equal), but align_ldim is UO, we will just assume we shouldn't use original alignment.
    if not (shape is None or shape == arr.shape) and align_ldim is UO: align_ldim = None
    #but if no specified shape (will use original) or shapes are equal, assume we can use base dimensions of array for new base.
    if align_ldim is UO: align_ldim = bshape #type: ignore[unbound-name]
    #we don't have to worry about bshape not existing, because we know if bshape is NA then arr is C or F contiguous and align_ldim=None from prev set.
    #To make pyrefly stop complaining. We know all UO's at this point are not possible.
    #todo: remove casts, make asserts.
    order,init_op,align_ldim = cast(str,order),cast(InitOp,init_op),cast(ShapeMaybe,align_ldim)
    return ar_spec(
        arr.shape if shape is None else shape,
        arr.dtype if dtype is None else dtype,
        order,
        arr if init_op is UO else init_op,
        name,
        align_ldim, 
    )


def ft_arspec(
    arspec: ArrayNode,
    shape: ShapeInput | None = None,
    dtype: DTypeLike | None = None,
    order: str | None = None,
    init_op: InitOp|UseOther = UO,
    name: str | None = None,
    align_ldim: ShapeInput|UseOther = UO,
) -> ArrayNode:
    """Return a **transposed** Array spec without changing memory layout.

    Swaps C↔F logical order, flips shapes/``bshape``, and carries over
    ``init_op`` (transposed if ndarray, or via a wrapper if callable).
    """
    rs = arspec
    rop = rs.init_op
    if isinstance(rop, np.ndarray): top = rop.T
    elif callable(rop): top = lambda ar: rop(ar.T)  # noqa: E731
    else: top = rop
    
    tspec = ar_spec(rs.shape[::-1], rs.dtype, "C" if rs.order != "C" else "F", top, rs.name, rs.bshape[::-1],)
    return f_arspec(tspec, shape=shape, dtype=dtype, order=order, init_op=init_op, name=name, align_ldim=align_ldim,)


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
def _dot_node_attr(node: ENode, *, with_offsets: bool) -> str:
    """Generate DOT attributes with name inside node and other info as
    external."""
    # Secondary info for external label (xlabel)
    info_parts: List[str] = []
    col = 'color="white"'
    if node.label: info_parts.append(f"- {node.label} -")
    if isinstance(node, ArrayNode):
        dtype_label = (
            dtype_abbr(node.dtype)
            if isinstance(node.dtype, np.dtype)
            else gen_str(node.dtype)
            if isinstance(node.dtype, sym.Expr)
            else str(node.dtype)
        )
        if node.align_ldim is not None: info_parts.append(f"{node.shape}, {node.bshape}, {dtype_label}")
        else: info_parts.append(f"{node.shape}, {dtype_label}")
        col = 'color="lime"'
    elif isinstance(node, ValueNode):
        vt = repr(node.value)
        qrg = 18
        vt = vt[:qrg]
        if qrg <= len(vt): vt = vt[:10] + "..."
        info_parts.append(vt)
        col = 'color="pink"'
    elif node.rule == BufferMap.DISTINCT: col = 'color="orange"' #bc it can only be a ContainerNode by this conditional.
    else: col = 'color="cyan"' #it is shared
    if with_offsets and node.ofs is not None:
        if isinstance(node.nbytes, sym.Expr): end = f"{node.ofs}+{gen_str(node.nbytes)}"
        else: end = node.ofs + node.nbytes
        info_parts.append(f"[{node.ofs}, {end})")
    if hasattr(node.nbytes,'nbytes'):
        if isinstance(node.nbytes, sym.Expr): info_parts.append(f"{gen_str(node.nbytes)} B")
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
    ): lines.append(line)
    return "\n".join(lines)


# 8) PyVis visualiser
def _bmap_pyvis(src: Union[str, BaseNode], *, with_offsets: bool = False) -> Network:
    """Internal helper that returns a configured ``pyvis.Network`` graph.

    Accepts either a DOT string or a ``ContainerNode``; when given a node,
    converts via ``bmap_todot`` then builds a NetworkX graph using ``pydot``.
    """
    try:
        import networkx as nx  # type: ignore[missing-import]
        import pydot  # type: ignore[missing-import]
        from pyvis.network import Network  # type: ignore[untyped-import]
    except ImportError as e: raise ImportError("pyvis, networkx, pydot required for bmap_pyvis()") from e
    if isinstance(src, str): dot_data = src
    elif isinstance(src, BaseNode): dot_data = bmap_todot(src, with_offsets=with_offsets)
    else: raise TypeError("src must be DOT str or ContainerNode")
    graphs = pydot.graph_from_dot_data(dot_data)
    if not graphs: raise ValueError("Failed to parse DOT data")
    pgraph = graphs[0]
    g_nx: Any = nx.drawing.nx_pydot.from_pydot(pgraph)
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
        if isinstance(col, str) and col.startswith('"') and col.endswith('"'): node["color"] = col.strip('"')
    net.set_options("""{
  "layout": {"hierarchical":{"enabled":true,"direction":"UD","sortMethod":"hubsize","blockShifting":true,"edgeMinimization":true}},
  "physics":{"enabled":true}
}""")
    return net


BMAP_ROPTS2 = """{
      "layout": {"hierarchical":{"enabled":false,"direction":"UD","sortMethod":"hubsize","blockShifting":true,"edgeMinimization":true}},
      "physics":{"enabled":true, "wind": { "x": 0.0, "y": 0 },"repulsion":{"nodeDistance": 200,"springLength": 200},"centralGravity":0.1}
    }"""

BMAP_ROPTS = """{
      "layout": {"hierarchical":{"enabled":true,"direction":"UD","sortMethod":"hubsize","blockShifting":true,"edgeMinimization":true}},
      "physics":{"enabled":true, "wind": { "x": 0.02, "y": 0 },"hierarchicalRepulsion":{"nodeDistance": 55,"avoidOverlap": 1}}
    }"""

_bopt = """{
      "layout": {"hierarchical":{"enabled":true,"direction":"UD","sortMethod":"hubsize","blockShifting":true,"edgeMinimization":true}},
      "physics":{"enabled":true, "wind": { "x": 0.02, "y": 0 },"hierarchicalRepulsion":{"nodeDistance": 67,"avoidOverlap": 1}}
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
    try:
        import networkx as nx  # type: ignore[missing-import]
        import pydot  # type: ignore[missing-import]
        from pyvis.network import Network  # type: ignore[untyped-import]
    except ImportError as e:
        raise ImportError("pyvis, networkx, pydot required for bmap_pyvis()") from e

    # Obtain DOT source and parse
    dot_data = src if isinstance(src, str) else bmap_todot(src, with_offsets=with_offsets)
    graphs = pydot.graph_from_dot_data(dot_data)
    if not graphs: raise ValueError("Failed to parse DOT data")
    #print(graphs)
    pgraph = graphs[0]
    # Convert to NetworkX
    g_nx: Any = nx.drawing.nx_pydot.from_pydot(pgraph)
    # Insert a blank white dummy sibling under the root container to anchor hubsize
    # Determine root: if ContainerNode passed, use its name; else take first node

    nodes = list(g_nx.nodes)
    root_id = nodes[0] if nodes else None
    dummy = 0
    # Note: For some reason we need a dummy node otherwise the top level node will be out of place try the old _bmap_pyvis to see strange behavior
    if dummy not in g_nx: g_nx.add_node(
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
        if isinstance(lbl, str) and lbl.startswith('"') and lbl.endswith('"'): node["label"] = lbl.strip('"')
        col = node.get("color")
        if isinstance(col, str) and col.startswith('"') and col.endswith('"'): node["color"] = col.strip('"')

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
