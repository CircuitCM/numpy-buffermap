from pyvis.network import Network
import copy
from collections.abc import Collection
from functools import partial
from typing import Callable, List, Sequence, Tuple, Union

import numpy as np
from anytree import NodeMixin, PreOrderIter
from anytree.exporter import DotExporter

from bmap._nodes import ArrayNode, BaseNode, ContainerNode, ItemNodeT, ValueNode
from bmap._util import (
    UO,
    BButil,
    BufferAlign,
    BufferMap,
    SymExpr,
    SymSymbol,
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
    ls_layers,
    roundup,
)


def build_buffer_allocator(buffer_map: ContainerNode, args:Collection=(),kwargs:dict=None,tempvar: str='t',subname: bool=False,chkforbuffer: bool=True,balign: int=BufferAlign.PAGE,fullreduce: bool=True) -> str:
    """Builds the string definition of a function that dynamically allocates the arrays and values of a buffer map.
    :param sym_dic: All value
    :param balign: The initial alignment of the buffer. It must be greater than or equal to the alignment you choose to build the buffer_map with.
    """
    #make sure sym_dic is using symbols, so for user it's ok to specify them with string keys
    args=[*((k.name if isinstance(k,SymSymbol) else k) for k in args),]
    args.sort(key=lambda x:c_orlen(x,'type_'),)
    kwargs={(k.name if isinstance(k,SymSymbol) else k):v for k,v in kwargs.items()} if kwargs is not None else {} #only symbols not expressions supported now, all expressions should be evaluated into needed equations.
    #for now there is also no dead-parameter/dead-code removal, and no checks for if all used input parameters are referenced in the header
    if chkforbuffer:
        hd=BButil.build_header(buffer_map,args,kwargs,balign=balign)
        allq,ct=[buffer_map.nbytes],[1]
    else:
        hd=BButil.build_header(buffer_map,args,kwargs,check_alloc=None,balign=balign)
        allq,ct=[],[0]

    if fullreduce:
        allq.extend(_cxprs(buffer_map)) #already 'least expressions' form.
        if buffer_map.rule==BufferMap.DISTINCT: allq.pop() #if initial buffer node is distinct we don't need the 
        lyrs,reduc=cse_codereduction(allq, tempvar)
        strl=ls_layers(lyrs)
        allc=(BButil.add_balloc(reduc[0]),) if chkforbuffer else ()
        mkls= _bb2(buffer_map, reduc, subname=subname,ct=ct)
        mdst='\n'.join(mkls[0])
        mdst=mdst.replace('\n','\n    ') #in case there are multi-line statements
        estr=f'return {", ".join(mkls[1])}'
    else: 
        #no reduction is quicker but also much more ugly and less readable.
        mkls= _bba(buffer_map, subname=subname)
        if buffer_map.rule==BufferMap.DISTINCT: mkls.pop()
        strl=()
        allc=(BButil.add_balloc(allq[0]),) if chkforbuffer else ()
        mdst='\n'.join(mkls[0])
        mdst=mdst.replace('\n','\n    ') #in case there are multi-line statements
        estr=f'return {", ".join(mkls[1])}'
        
    fullf='\n    '.join((hd,*strl,*allc,mdst,'',estr))
    return fullf


def _cxprs(bn:BaseNode, excs=None):
    wn= excs is None
    if wn: excs=[]
    if isinstance(bn, ContainerNode) and bn.rule==BufferMap.DISTINCT:
        if bn.align:
            for c,eq in zip(bn.children, bn.sym_def()):
                _cxprs(c, excs)
                _pxpr(excs,eq)
        else:
            for c in bn.children:
                _cxprs(c, excs)
                _pxpr(excs,c.nbytes)
    else:
        for c in bn.children: 
            _cxprs(c, excs)
        if isinstance(bn, ArrayNode):_arrpxpr(excs, bn.sym_def())
        elif isinstance(bn, ValueNode): _pxpr(excs,bn.sym_def())
    
    if wn:
        return excs
    

def _bb2(bnode: BaseNode, exls, subname=False, sn: str | None=None, ct=None, mkls=None):
    wn= mkls is None
    if wn: mkls=([],[])
    if ct is None:ct=[0]
    ctt=isinstance(bnode, ContainerNode)
    if ctt and bnode.rule==BufferMap.DISTINCT:
        mkls[0].append('')
        bc=bnode.children
        bcl=len(bc)-1
        if bnode.align:
            for i,(c,eq) in enumerate(zip(bc, bnode.sym_def())):
                _bb2(c, exls, subname, bnode.name, ct, mkls)
                if i==bcl and wn: break #so we don't run into exls that is too far extended
                ot=_bxpr(exls,eq,ct)
                mkls[0].append(ContainerNode.s_def(ot))
        else:
            for i,c in enumerate(bc):
                _bb2(c, exls, subname, bnode.name, ct, mkls)
                if i==bcl and wn: break #so we don't run into exls that is too far extended
                ot=_bxpr(exls,c.nbytes,ct)
                mkls[0].append(ContainerNode.s_def(ot))
    elif ctt:
        mkls[0].append('')
        for c in bnode.children:
            _bb2(c, exls, subname, bnode.name, ct, mkls)
    else:
        if isinstance(bnode, ArrayNode):
            st=bnode.gen_call(*_arrbxpr(exls, bnode.sym_def(), ct), subn=sn if subname else None)
        elif isinstance(bnode, ValueNode): 
            st=bnode.gen_call(_bxpr(exls,bnode.sym_def(),ct),subn=sn if subname else None)
        
        if st[0] is not None: mkls[0].append(st[0])
        if st[1] is not None: mkls[1].append(st[1])
    
    if wn:
        return mkls
    else:
        return ct
    

def _bba(bnode: BaseNode, mkls=None, subname=False, sn: str | None=None):
    wn= mkls is None
    if wn: mkls=([],[])
    ct=isinstance(bnode, ContainerNode)
    if ct and bnode.rule==BufferMap.DISTINCT:
        mkls[0].append('')
        if bnode.align:
            for c,eq in zip(bnode.children, bnode.gen_call(subn=sn if subname else None)[0]):
                _bba(c, mkls, subname, bnode.name)
                mkls[0].append(eq)
        else:
            for c in bnode.children:
                _bba(c, mkls, subname, bnode.name)
                mkls[0].append(ContainerNode.s_def(c.nbytes))
    else:
        if ct:
            mkls[0].append('')
        for c in bnode.children:
            _bba(c, mkls, subname, bnode.name)
        if not ct:
            st=bnode.gen_call(subn=sn if subname else None)
            if st[0] is not None: 
                mkls[0].append(st[0])
            if st[1] is not None: mkls[1].append(st[1])
    if wn:
        return mkls

def build_bmap(
    bmap: ContainerNode,
    *,
    align: int = BufferAlign.AVX512,
    name_join: bool = True,
    force_merge: bool = False,
    verbose: bool = False,
) -> set:
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
    symbols=_compute_nbytes(bmap, align=align)
    if verbose:
        check_bmap(bmap)
    return symbols


# ---------------------------------------------------------------------------
# 1) Reduction (merge adjacent identical‑rule containers)
# ---------------------------------------------------------------------------
def _merge_child_into_parent(
    parent: ContainerNode, child: ContainerNode, *, name_join: bool
) -> None:
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
                and node.align == ch.align
                and (force_merge or (not node.no_merge and not ch.no_merge))
            ):
                _merge_child_into_parent(node, ch, name_join=name_join)
    return bmap


# ---------------------------------------------------------------------------
# 2) Size propagation & offset assignment w/ alignment
# ---------------------------------------------------------------------------
def _compute_nbytes(node: BaseNode, align: int = BufferAlign.AVX512,simplify=True,_symbols=None) -> int|SymExpr|set:
    """Compute subtree byte sizes with alignment rounding.

    - ``ValueNode`` contributes 0.
    - ``ArrayNode`` contributes its exact size (no rounding here).
    - ``ContainerNode`` combines child sizes per rule; ``align`` on
      DISTINCT containers disables per-child rounding.
    """
    wn=_symbols is None
    if wn: _symbols=set()
    align=buffer_expr(align)
    if isinstance(node, ValueNode):
        _symbols.update(node.free_symbols)
        return 0
    if isinstance(node, ArrayNode):
        _symbols.update(node.free_symbols)
        return node.nbytes
    # compute child sizes
    node:ContainerNode
    if node.rule == BufferMap.SHARED:
        sizes = (*(_n_(ch, align,simplify,_symbols) for ch in node.children),)
        raw = gen_max(sizes,simplify) if sizes else 0
        # round up shared region
        node.nbytes = roundup(raw,align)
    else:
        # distinct: sum rounded child regions
        if node.align: #we make sure every node is aligned.
            ng=node.aligned_eqns
            ng.clear()
            #we already know Container nodes are certainly aligned.
            #this is true because aligned=False distinct nodes and shared nodes are still aligned at the container level.
            #so aligned=True distinct nodes will still be aligned as well.
            #So we can skip their expensive sympy expression build and just use the existing nbytes expression.
            ng.extend(roundup(_n_(ch, align,simplify,_symbols),align) if isinstance(ch, ArrayNode) else _n_(ch, align,simplify,_symbols) for ch in node.children)
            node.nbytes=gen_add(ng,simplify) if ng else 0
        else:
            # if not align then nbytes will be rounded to match whole container alignment, but not children alignment. Which means that arraynodes that must be aligned contiguously can be forced to, if they represent the memory of another array exactly.
            #Which could be useful maybe when you have a shared node, one array, then a distinct sibling node where all 
            #however if another container is a child and it's not at the beginning, this could mess up alignment for all nodes it contains.
            #so distinct nodes should have alignment by default unless there is a special case for all arrays.
            sizes = (*(_n_(ch, align,simplify,_symbols) for ch in node.children),)
            raw=gen_add(sizes,simplify) if sizes else 0
            node.nbytes = roundup(raw,align)
    if wn: #we know it was the most outer node so return the set for user.
        return _symbols
    else:
        return node.nbytes
_n_=_compute_nbytes


def arrays_map(buffer_map: BaseNode, sym_dic:dict=None,balign: int=BufferAlign.PAGE,_buffer:np.ndarray=None, _base: int = 0) -> None:
    """Inits arrays based off of the buffer map. Arrays are placed within their array node.
    And container nodes receive the buffer subindex range they represent.
    :param balign: The initial alignment of the buffer. It must be greater than or equal to the alignment you choose to build the buffer_map with.
    """
    #make sure sym_dic is using symbols, so for user it's ok to specify them with string keys
    od=sym_dic
    sym_dic={buffer_expr(k):v for k,v in od.items()}
    #make a buffer if none was provided
    if _buffer is None:
        _buffer = aligned_buffer(eval_buff_expr(buffer_map.nbytes, sym_dic), balign)
    _generate_nodearrays(buffer_map,_buffer,_base,sym_dic)
    

def _generate_nodearrays(
    node: BaseNode,buffer:np.ndarray=None, base: int = 0, sym_dic:dict=None,) -> None:
    node.ofs = base
    if isinstance(node, ArrayNode):
        node.mk_array(buffer,sym_dic)
    elif isinstance(node, ContainerNode):
        if node.rule == BufferMap.SHARED:
            for ch in node.children:
                _generate_nodearrays(ch, buffer, base, sym_dic)
        elif node.align:
            for ch,eqn in zip(node.children,node.aligned_eqns):
                _generate_nodearrays(ch, buffer,base, sym_dic)
                base+=eval_buff_expr(eqn, sym_dic)
        else:
            for ch in node.children:
                _generate_nodearrays(ch, buffer,base, sym_dic)
                base+=eval_buff_expr(ch.nbytes, sym_dic)


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
    self,
    align: int = BufferAlign.AVX512,
    alignb: int = BufferAlign.PAGE,
    name_join: bool = True,
    force_merge: bool = False,
    verbose: bool = False,
):
    build_bmap(
        self,
        align=align,
        name_join=name_join,
        force_merge=force_merge,
        verbose=verbose,
    )
    #allocate_bmap(self, align=align,alignb=alignb)
    return flat_inits(self)

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


# ValueNode
def v_spec(value, name=None) -> ValueNode:
    """Wrap a literal value in a :class:`ValueNode` that doesn't use buffer
    bytes.

    The value participates in preorder flattening alongside arrays and is
    returned by ``flat_inits`` in position order.
    """
    return ValueNode(value, name)


# ArrayNode
def ar_spec(
    shape=(0,), dtype=np.float64, order:str="C", init_op:np.ndarray|Callable|None=None, name:str|None=None, align_ldim:str|None=None
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
    init_op: type[object]=UO,
    name=None,
    align_ldim: type[object]=UO,
) -> Callable:
    """Return a partially-applied constructor for :func:`f_arspec`.

    Any field left as ``None``/``UO`` will inherit from the base spec
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
    init_op=UO,
    name=None,
    align_ldim=UO,
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
    align_ldim = None if align_ldim is UO else align_ldim
    sshape, ddtype, oorder, iinit_op, aalign_ldim, nname = (
        rs.shape if shape is None else shape,
        rs.dtype if dtype is None else dtype,
        rs.order if order is None else order,
        rs.init_op if init_op is UO else init_op,
        rs.bshape if spc_align else align_ldim,
        name,
    )
    return ar_spec(sshape, ddtype, oorder, iinit_op, nname, aalign_ldim)


def array_arspec(
    arr: np.ndarray,
    shape=None,
    dtype=None,
    order=None,
    init_op: type[object]=UO,
    name=None,
    align_ldim: type[object]=UO,
    _aldim_def: int=BufferAlign.AVX512,
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
        if not (nr or order == torder) and align_ldim is UO:
            align_ldim = None
    elif align_ldim is UO:
        align_ldim = None
    if nr:
        order = torder
    if not (shape is None or shape == arr.shape) and align_ldim is UO:
        align_ldim = None
    if align_ldim is UO:
        align_ldim = bshape
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
    shape=None,
    dtype=None,
    order=None,
    init_op: type[object]=UO,
    name=None,
    align_ldim: type[object]=UO,
) -> ArrayNode:
    """Return a **transposed** Array spec without changing memory layout.

    Swaps C↔F logical order, flips shapes/``bshape``, and carries over
    ``init_op`` (transposed if ndarray, or via a wrapper if callable).
    """
    rs = arspec
    rop = rs.init_op
    top = (
        rop.T
        if isinstance(rop, np.ndarray)
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


def db_node(*args, name=None, no_merge=False, align=True) -> ContainerNode:
    """Create a distinct-region :class:`ContainerNode` (children are
    concatenated).

    ``align=True`` enables per-child alignment so the concatenation can
    exactly match an external memory layout.
    """
    return ContainerNode(
        *args, rule=BufferMap.DISTINCT, name=name, no_merge=no_merge, align=align
    )


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
            info_parts.append(f"{node.shape}, {node.bshape}, {dtype_abbr(node.dtype)}")
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


def bmap_todot(bmap: BaseNode, *, with_offsets: bool = True) -> str:
    """Serialize the tree to Graphviz DOT via
    ``anytree.exporter.DotExporter``."""
    lines :list[str] = []
    for line in DotExporter(
        bmap,
        nodenamefunc=lambda n: f"n_{n.id}",
        nodeattrfunc=lambda n: _dot_node_attr(n, with_offsets=with_offsets),
    ):
        lines.append(line)
    return "\n".join(lines)


# 8) PyVis visualiser
def _bmap_pyvis(src: Union[str, BaseNode], *, with_offsets: bool = False) -> Network:
    """Internal helper that returns a configured ``pyvis.Network`` graph.

    Accepts either a DOT string or a ``ContainerNode``; when given a node,
    converts via ``bmap_todot`` then builds a NetworkX graph using ``pydot``.
    """
    try:
        import networkx as nx
        import pydot
        from pyvis.network import Network
    except ImportError as e:
        raise ImportError("pyvis, networkx, pydot required for bmap_pyvis()") from e
    if isinstance(src, str):
        dot_data = src
    elif isinstance(src, BaseNode):
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
def bmap_pyvis(src: Union[str, ContainerNode], with_offsets: bool = False,height: str='1000px',width: str='100%',render_options: str=BMAP_ROPTS) -> Network:
    """Build a PyVis interactive tree using a top-down layout and physics
    enabled."""
    try:
        import networkx as nx
        import pydot
        from pyvis.network import Network
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
    bmap: NodeMixin, path: str = "bmap_tree.png", *, with_offsets: bool = False
) -> str:
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
