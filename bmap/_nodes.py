# from collections.abc import Collection
# from functools import cache

from typing import Sequence, Union, Optional

import numpy as np
from anytree import NodeMixin,LightNodeMixin, RenderTree, DoubleStyle
from itertools import chain
import math as mt
from numbers import Number
from bmap._util import _IDGen, _ESET, gen_str, buffer_expr, SymExpr, roundup, dt_buff_exprs, eval_buff_exprs, \
    BufferMap, eval_buff_expr, c_orlen, BufferAlign


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
        #self.buffer: Optional[np.ndarray] = None

    def _short(self) -> str:
        return self.label or str(self.id)

    @property
    def name(self):
        return self.label

    @name.setter
    def name(self, name):
        self.label = name

    @name.deleter
    def name(self):
        del self.label
    
    
    def gen_call(self,*args,**kwargs):
        """This was previously called gen_def, now it defines it's string definition, based on certain modified dependents.
        
        This will emit the defining string of the representation in the code defining generator.
        
        The defining idea of the writer, is that if we start at the top of the tree. Then walk down the
        tree depth first, adding gen_defs to a list of strings on first visitation to a node, we should be able correctly
        define all buffer sub-indices offsets and array definitions/dependencies accurately.
        """
        #subn : sub-name for use in extending the name of the parameter eg containername_varname
        istr='#The assignment/initializer codeline.'
        rstr='#The return statement/reference within the return line.'
        raise NotImplementedError
    
    def sym_def(self):
        """Returns the sequence of node objects that are dependent on the node's procedural generation and simplification by cse in sympy."""
        return None
    
    def free_symbols(self):
        """If sympy is installed this is an api to return all used symbols within the node."""
        return _ESET


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
        self.value = buffer_expr(value,safe=False)

    @property
    def vtype(self):
        return type(self.value)

    def __repr__(self):
        return f"|Value {self._short()} {repr(self.value)[:50]} {self.vtype}|"
    
    def gen_call(self,valexpr=None, subn=None):
        if valexpr is None: valexpr = self.value
        vg=gen_str(valexpr) if isinstance(valexpr,SymExpr) else valexpr
        if self.name:
            rstr=f'{subn}_{self.name}' if subn else self.name
            istr= f'{rstr} = {vg}'
        else: 
            istr=None
            rstr=vg #we don't assign and go straight to the return
        return istr,rstr
    
    def sym_def(self):
        return self.value
    
    @property
    def free_symbols(self):
        """If sympy is installed this is an api to return all used symbols within the node."""
        if isinstance(self.value,SymExpr):
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
    
    

    def __init__(
        self,
        shape: Union[int, Sequence[int]] = (0,),
        dtype: np.dtype|np.generic = np.float64,
        order: str = "C",
        init_op: Optional[Union[int, float, np.ndarray, callable]] = None,
        name: Optional[Union[str, int, float]] = None,
        align_ldim=None,
    ) -> None:
        super().__init__(name=name, no_merge=True)
        s = dt_buff_exprs((shape,) if isinstance(shape, int) else shape)
        self.shape: Sequence[int|SymExpr, ...] = s
        self.dtype: np.dtype|SymExpr
        self.bshape: Sequence[int|SymExpr, ...]
        self.array: Optional[np.ndarray] = None
        self.barray: Optional[np.ndarray]
        self.align_ldim: int|Sequence[int|SymExpr, ...] = dt_buff_exprs(align_ldim)
        if isinstance(dtype, type) and issubclass(dtype,(np.generic,Number)):dtype=np.dtype(dtype)
        if isinstance(dtype,np.dtype):
            self.dtype = dtype #np.dtype(dtype)
            self.itsize=dtype.itemsize
        else:
            syty=buffer_expr(dtype)
            if c_orlen(syty.name,'type_'): 
                self.dtype= syty
                self.itsize=buffer_expr(syty.name[5:])
            else:
                print(f'Note: {dtype} is specified without a `type_` field.\nTo work in the allocation code generator make sure the var name is `type_{dtype}`.')
                self.dtype=buffer_expr(f'type_{syty.name}')
                self.itsize = syty
        self.order: str = order.upper()
        self.init_op = init_op
        if isinstance(align_ldim, tuple):
            self.bshape = self.align_ldim
        elif align_ldim is not None:
            #otherwise we can't be certain about if align
            print(self.align_ldim,self.itsize)
            rsz = self.align_ldim // self.itsize
            if self.order == "C":
                self.bshape = (*s[:-1], roundup(s[-1],rsz))
            else:
                self.bshape = (roundup(s[0],rsz), *s[1:])
        else:
            self.bshape = s
        #... this must have been bshape intended? Check later
        #self.nbytes: int = int(np.prod(self.shape)) * self.dtype.itemsize
        #likely
        self.nbytes: int|SymExpr = mt.prod(self.bshape) * self.itsize

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

    def mk_array(self, buffer: Optional[np.ndarray] = None,sym_dic:dict=None) -> np.ndarray:
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
            bshape=eval_buff_exprs(self.bshape, sym_dic)
            dtype=eval_buff_expr(self.dtype, sym_dic)
            self.array = np.empty(bshape, dtype=dtype, order=self.order)
        elif y := (buffer is not None):
            bshape=eval_buff_exprs(self.bshape, sym_dic)
            dtype=eval_buff_expr(self.dtype, sym_dic)
            cnt = int(mt.prod(bshape))
            view = np.frombuffer(
                buffer,
                dtype=dtype,
                count=cnt,
                offset=0 if self.ofs is None else self.ofs,
            )
            view = view.reshape(bshape, order=self.order)
            self.array = view
        if y and self.align_ldim is not None:
            shape=eval_buff_exprs(self.shape, sym_dic)
            self.barray = self.array
            self.array = (
                self.barray[..., : shape[-1]]
                if self.order == "C"
                else self.barray[: shape[0], ...]
            )
        self._rinit()
        return self.array

    @property
    def value(self):
        return self.array
    
    @staticmethod
    def array_genc(offstr,dtstr,dmstr,ord):
        return f'fb_(buffer[:{offstr}],{dtstr}).reshape({dmstr}){"" if ord=="C" else ".T"}'
    
    npspec='np.'
            
    
    def gen_call(self,bytexpr=None,bshapet=None,shapet=None,subn=None):
        """Generate string definition of array from symbolics or value."""
        if not subn and not self.name: raise NameError('Arrays without name or sub name are not supported.')
        st=self._short() #name first, if fails falls back to an integer counter (not related to container position)
        rstr=f'{subn}_{st}' if subn else st
        if bytexpr is None: bytexpr=self.nbytes
        ols=gen_str(bytexpr)
        ds=gen_str(self.dtype) if isinstance(self.dtype,SymExpr) else self.npspec + self.dtype.name
        
        if bshapet is None: bshapet=self.bshape
        dms=gen_str(tuple(bshapet[::(1 if self.order=='C' else -1)]))
        istr=self.array_genc(ols,ds,dms,self.order)
        istr=f'{rstr} = {istr}'
        if self.align_ldim is not None:
            if shapet is None: shapet=self.shape
            istr+= f'[{f"...,:{gen_str(shapet[-1])}" if self.order=="C" else f":{gen_str(shapet[0])},..."}]' 
            #adding \t here seems a bit informal, given we don't know how gen_def will be utilized outside of this method.
            #so do it at the function formatting scope.
        return istr,rstr
    
    def sym_def(self):
        return self.nbytes, self.bshape, self.shape
    
    @property
    def free_symbols(self):
        """If sympy is installed this is an api to return all used symbols within the node."""
        fs=set()
        #itsize=symbol_ref is a special cased derived from type_{symbol_ref}'s so it won't go in the function header
        #and gets handled by the allocation handler.
        ld= not isinstance(self.align_ldim,Sequence)
        for v in (self.shape if ld else chain(self.shape,self.align_ldim)):
            if isinstance(v,SymExpr):
                if v.is_symbol:fs.add(v)
                else:fs.update(v.free_symbols)
        if isinstance(v:=self.align_ldim,SymExpr):
            if v.is_symbol:fs.add(v)
            else:fs.update(v.free_symbols)
        if isinstance(v:=self.dtype,SymExpr):
            if v.is_symbol:fs.add(v)
            else:fs.update(v.free_symbols)

        #it's probably a decent bit slower to just call self.nbytes.free_symbols. that also won't include the dtype but will include self.itsize which we dont want
        
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
    #SYM_EQN=MulSym

    def __init__(
        self,
        *children: "BaseNode",
        rule: int = BufferMap.DISTINCT,
        name: Optional[Union[str, int, float]] = None,
        no_merge: bool = False,
        align: bool = True,
    ) -> None:
        super().__init__(name=name, no_merge=no_merge)
        self.rule: int = int(rule)
        self.align = align and rule==BufferMap.DISTINCT  # only effective for distinct rule nodes.
        self.aligned_eqns=[]
        self.nbytes:int|SymExpr
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
        # """
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
        pass
        build_bmap(
            self,
            align=align,
            name_join=name_join,
            force_merge=force_merge,
            verbose=verbose,
        )
        #allocate_bmap(self, align=align,alignb=alignb)
        return flat_inits(self)
    
    @staticmethod
    def s_def(syq):
        ols=gen_str(syq)
        return f'buffer = buffer[{ols}:]'
    
    def gen_call(self,setexpr=None, subn=None):
        """See BaseNode docstring."""
        if self.align:
            if setexpr is None: setexpr = self.aligned_eqns
            return (*(self.s_def(s) for s in setexpr),),None
        else:
            if setexpr is None: setexpr = self.nbytes
            return self.s_def(setexpr),None
        
    def sym_def(self):
        return self.aligned_eqns if self.align else self.nbytes


    def __repr__(self):
        return f"|Node {self._short()} rule={'S' if self.rule == BufferMap.SHARED else 'D'} children={len(self.children)} {f'nbytes={self.nbytes}'if hasattr(self,'nbytes') else ''}|"

    def __str__(self):
        v = str(RenderTree(self, style=DoubleStyle))
        return v
    
    @property
    def free_symbols(self):
        """If sympy is installed this is an api to return all used symbols within the node."""
        #it should be necessary and a little slower to call free_symbols on the container, than directly on value or array nodes.
        #which is why the build buffer map function only calls array/value nodes, they will also contain all symbols used on the graph there.
        if self.aligned_eqns:
            return set().union(*(eq.free_symbols for eq in self.aligned_eqns if isinstance(eq,SymExpr)))
        elif self.nbytes is not None: #assume nbytes has been simplified, so recursion to a large tree of nodes could be slower than the nbytes expr check. (post simplify)
            if isinstance(self.nbytes,SymExpr):
                return self.nbytes.free_symbols
            else: return _ESET
        elif len(cs:=self.children)>0:
            return set().union(*(c.free_symbols for c in cs))
        else: return _ESET

ItemNodeT = ValueNode | ArrayNode
