import importlib
import itertools
from typing import Any, Sequence

import numpy as np

if importlib.util.find_spec('numba'):
    from numba.extending import register_jitable
    #fastmath should make no difference for allocating a buffer.
    rgc=register_jitable(cache=True,fastmath=True,error_model='numpy') 
    from numba import typeof
else:
    def typeof(*args):
        return np.dtype(type(args[0]))
    rgc=lambda f:f

### Helpers
UO =use_other= object()
_ESET=set()
HAS_SYM=False
_GB_EXPR={}

SymSymbol=UO
SymExpr=UO
CD_ = 'cd_'  # ceildiv
FD_ = 'fd_' # floordiv

def gen_max(sq,simplify=True):
    """No sympy version"""
    return max(*sq)
def gen_add(sq,simplify=True):
    return sum(*sq)

def roundup(var,rup):
    if rup is None or rup<1: return var
    return ((var + rup - 1) // rup) * rup

def gen_str(expr):
    return str(expr)

if importlib.util.find_spec('sympy'):
    HAS_SYM=True
    import sympy as sym
    SymSymbol=sym.Symbol
    SymExpr=sym.Expr
    CD_ = sym.Function('cd_')  # ceildiv
    FD_ = sym.Function('fd_')  # floordiv
    #note sympy is generally not great with integer simplifications but it will take time to figure out a custom method
    #so for now we use this.
    def gen_max(sq,simplify=True):
        """General maximum, symbols and values."""
        mv=sym.Max(*sq)
        if mv.is_number:
            return int(mv)
        elif simplify:
            return mv#.simplify() #calling simplify solely for the max function but not the expressions might not actually do anything.
        return mv
    
    def gen_add(sq,simplify=True):
        """General add, symbols and values."""
        #it's faster to unload into a single Add expression than it is to +=
        mv=sym.Add(*sq)
        if mv.is_number:
            return int(mv)
        elif simplify:
            return mv.factor()#fraction=False,deep=True)#.simplify()
        return mv
    
    def roundup(var,rup):
        if rup is None or (isinstance(rup,int) and rup<1): return var
        return sym.ceiling(var/rup)*rup
    
    exec('from sympy import *', _GB_EXPR) #might need less imports than this, see about that later.
    _GB_EXPR['Symbol']=lambda s: sym.Symbol(s, integer=True, positive=True)
    
    from sympy.printing.str import StrPrinter
    class GenPrinter(StrPrinter):
        
        def _print_Max(self, expr):
            return "max(%s)" % ", ".join(self._print(a) for a in expr.args)
    
        def _print_floor(self, expr):
            x = expr.args[0]
            num, den = x.as_numer_denom()
            if den != 1 and den.is_integer: #and x.is_rational_function()
                na,da=num.is_Atom,den.is_Atom
                num_s = self._print(num)
                den_s = self._print(den)
                match (na, da):
                    case (True, True):   return f"({num_s}//{den_s})"
                    case (False, False): return f"(({num_s})//({den_s}))"
                    case (False, True):  return f"({num_s}//({den_s}))"
                    case (True, False):  return f"(({num_s})//{den_s})"
    
            # if hasattr(super(),'_print_floor'):
            #     return super()._print_floor(expr)
            return super()._print_Function(expr)
            
        def _print_ceiling(self, expr):
            x = expr.args[0]
            num, den = x.as_numer_denom()
            # Turn ceiling(num/den) into ((num + den - 1)//den) when den is integer-like.
            if den != 1 and den.is_integer: #and x.is_rational_function()
                num_s = self._print(num)
                den_s = self._print(den)
                return f"cd_({num_s},{den_s})"
    
            # if hasattr(super(), '_print_ceiling'):
            #     return super()._print_ceiling(expr)
            return super()._print_Function(expr)
        
        def _print_fd_(self, expr):
            # expr is fd_(x, y)
            x, y = expr.args
        
            na, da = x.is_Atom, y.is_Atom
            xs = self._print(x)
            ys = self._print(y)
        
            # special-case denominator == 1
            if y == 1 or y.is_One:
                return xs if na else f"({xs})"
        
            match (na, da):
                case (True, True):   return f"({xs}//{ys})"
                case (False, False): return f"(({xs})//({ys}))"
                case (False, True):  return f"({xs}//({ys}))"
                case (True, False):  return f"(({xs})//{ys})"
    
    _gpr=GenPrinter()
    
    def gen_str(expr):
        return _gpr.doprint(expr)
    
    
    
    
    

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
    AVX512 = 64  #512‑bit vector registers (AVX‑512)
    PAGE = 4096  #OS page size
    
    #You shouldn't ever need these:
    HUGE_PAGE_2MB = 2 * 1024 * 1024
    HUGE_PAGE_1GB = 1024 * 1024 * 1024

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

_SYMCACHE={}
_CHKEQ=set(' \t\n\r\f\v+-/*^%()[]{}.')

def check_eqstr(st):
    """Check if equation string.
    
    Assumes input is either symbol or equation str.
    """
    for c in st:
        if c in _CHKEQ: return True
    return False

def c_orlen(tgt,mtch):
    if len(tgt)<len(mtch):return False
    return tgt[:len(mtch)]==mtch

def buffer_expr(sy,warn=True,safe=True):
    """Buffer symbol or expression generator with object cache for faster tree operations.
    
    If sy is a string without whitespace, or an existing symbol: Makes a sympy symbol with constraints that are valid for use in dynamic/abstract buffer maps.
    
    If sy is a string with whitespace it will first assume it's an equation and try to eval it  
    
    If sy is an int value: returns the int value, which is the original static handling.
    
    If sy is another expression (e.g. a multi-symbol equation): It returns that expression alone.
    
    This expr building function will only handle single symbols or expressions that evaluate to a single output integer, not objects or tuples.
    This is because it's much more performant to swap symbols out by their string name with a literal int value, than it is to actually call .subs and then int cast
    an object expression or symbol in sympy.

    """
    if isinstance(sy,int) or sy is None: return sy #This allows value negatives which might be intentional. and None is a special return type.
    if HAS_SYM:
        if isinstance(sy,str):
            syc=_SYMCACHE.get(sy,None)
            if syc is None:
                if check_eqstr(sy):
                    #A call to free_symbols is not expensive. but parse_expr and creating symbols individually can be, 
                    #so we will also cache the symbols from the expression bc they could appear in the propagation.
                    #note if this comes out anything other than an expression or doesn't have free_symbols it will error.
                    #Which means symbols need to be a valid string without whitespaces or any other notation.
                    exp:SymExpr=sym.parse_expr(sy,global_dict=_GB_EXPR)
                    print(exp)
                    if exp.is_symbol:
                        _SYMCACHE[sy]=exp
                        _SYMCACHE[exp.name]=exp
                        if warn: print(f'Warning: Buffer Symbol cached under "{sy}" and "{exp.name}", correct symbol formatting does not include extra whitespace or operators.')
                    else:
                        _SYMCACHE[sy]=exp
                        sybs=exp.free_symbols
                        for syb in sybs:
                            if syb.name not in _SYMCACHE: _SYMCACHE[syb.name]=syb
                    return exp
                else:
                    _SYMCACHE[sy]=syc= sym.Symbol(sy,integer=True,positive=True)
                    return syc
            else:
                return syc
                
        #A symbol is an expression, but not all expressions are symbols (e.g. equations that are made of symbols).
        #So we check if symbol first. might also wanna just return the symbol straight up like exp.
        #by calling is_symbol this should also help keep errors between sym Basic that will have a bool is_symbol field always and any other
        #objects that might not.
        if sy.is_symbol:
            sn=sy.name
            syc=_SYMCACHE.get(sn,None)
            if syc is None:
                #positive means > 0.
                _SYMCACHE[sn]=syc= sy if sy.is_integer and sy.is_positive else sym.Symbol(sn,integer=True,positive=True)
            return syc
        #Otherwise return the expression, we shouldn't need to cache pre-made expressions because that means the user is making the expression externally first.
        #Also getting the string from an expression is expensive and might not match the original, eg using // instead of floor( ./.).
        #without type checking the class it means the error pop up elsewhere if it implements an is_symbol field, just note that.
        return sy
    if safe:
        raise ImportError(f'Sympy not installed but a dynamic parameter: {sy} was requested.\nPlease install sympy.')
    else: return sy


def mk_buff_dict(dc):
    return {buffer_expr(k):v for k,v in dc.items()}

def bdict(**kwargs):
    return mk_buff_dict(kwargs)

def buffer_symbols(*tgt):
    """Make buffer symbols in the same way as sym.symbols."""
    if len(tgt)==1: tgt=tgt[0]
    if isinstance(tgt,str) and ' ' in tgt:
        return (*(buffer_expr(l) for l in tgt.split(' ') if not check_eqstr(l) and l != ''),)
    return dt_buff_exprs(tgt)
        

def dt_buff_exprs(tgt):
    """Ducktape symbols in target, either return as tuple or singular value if not a sequence."""
    if not isinstance(tgt,str) and isinstance(tgt,Sequence):
        return (*(buffer_expr(t) for t in tgt),)
    else:
        return buffer_expr(tgt)

def eval_buff_expr(evt:Any, evd:dict=None):
    """Evaluate buffer expression or symbol."""
    if type(evt) is int: return evt
    if not HAS_SYM or evd is None: return int(evt) #which should raise if it can't be turned into an int.
    if evt.is_symbol: return evd[evt] #assume user correctly adds only ints to evd values
    return int(evt.subs(evd)) #if this doesn't eval to int then it will also correctly raise.

def eval_buff_exprs(evts:Sequence, evd:dict=None):
    """Evaluate sequence of buffers expression or symbols."""
    if not HAS_SYM or evd is None: 
        for evt in evts:
            if type(evt) is not int: raise TypeError(f'Element {evt} is not an integer but integers are required for buffer map specification.')
        return evts
    return (*(e if type(e) is int else evd[e] if e.is_symbol else int(e.subs(evd)) for e in evts),)


def cf_plcsym(exprs, cd=CD_, fd=FD_):
    """
    Preprocess expressions:
      ceiling(num/den) -> cd(num, den)
      floor(num/den)   -> fd(num, den)
    where num/den is detected by sympy.fraction(arg).
    """
    ceil = sym.ceiling
    floor = sym.floor

    def repl(obj):
        if obj.func is floor or obj.func is ceil:
            num, den = sym.fraction(obj.args[0])
            if den != 1:
                return fd(num, den) if obj.func is floor else cd(num, den)
        return obj

    return [e.replace(lambda obj: obj.func is floor or obj.func is ceil,repl) for e in exprs]


def numb_syms(prefix='t',start=0):
    while True:
        name = '%s%s' % (prefix, start)
        s = buffer_expr(name,)
        yield s
        start += 1
        
def is_eqn(expr):
    return isinstance(expr,SymExpr) and not expr.is_Atom #or is_symbol but atom probably more correct.


def _pxpr(excs,exprs):
    if is_eqn(exprs): excs.append(exprs)
    elif isinstance(exprs,(tuple,list)):
        for v in exprs:
            if is_eqn(v): excs.append(v)
            
def _arrpxpr(excs, exprss):
    _pxpr(excs,exprss[0])
    _pxpr(excs,exprss[1])
    if not (exprss[1] is exprss[2]):
        _pxpr(excs,exprss[2])
    
    

def _bxpr(exls,exprs,ct):
    if is_eqn(exprs):
        ex=exls[ct[0]]
        ct[0]+=1
        return ex
    elif isinstance(exprs,(tuple,list)):
        dl=[]
        for v in exprs:
            if is_eqn(v): 
                dl.append(exls[ct[0]])
                ct[0]+=1
            else: dl.append(v)
        return dl
            
def _arrbxpr(exls, exprss, ct):
    ot=[]
    ot.append(_bxpr(exls,exprss[0],ct))
    ot.append(t1:=_bxpr(exls,exprss[1],ct))
    if not (exprss[1] is exprss[2]):
        ot.append(_bxpr(exls,exprss[2],ct))
    else:
        ot.append(t1)
    #print(ct,ot)
    return ot


class BButil:
    
    typeref1=lambda t:f'{t[5:]} = np.dtype({t}).itemsize'
    typeref2=lambda t:f'{t[5:]} = nbu.prim_info({t},3)' #for my numba util library
    
    @classmethod
    def build_header(cls,bmap,args,kwargs,check_alloc='buffer',balign=BufferAlign.PAGE, typebytes:callable=None,ceilref:str=None):
        el=not isinstance(balign,int)
        if isinstance(balign,SymSymbol): balign=balign.name
        name=bmap.name.lower().replace(' ','_')
        return f"""def {name}({', '.join(args)}{', '*(len(args)>0)}{', '.join(f'{k}={v}' for k,v in kwargs.items())}{', '*(len(kwargs)>0)}{check_alloc}=None{f', {balign}'*el}):
    {BButil.add_ceildiv(ceilref)}
    {BButil.add_typeparams(args, kwargs, typebytes)}"""

       
    @classmethod
    def add_typeparams(cls, args, kwargs, tbld=None):
        if tbld is None: tbld=cls.typeref1
        tb=[tbld(v) for v in args if c_orlen(v,'type_')]
        tb.extend(tbld(v) for v in kwargs.keys() if c_orlen(v,'type_'))
        if len(tb)>0:
            tb.append('')
            return '\n    '.join(tb)
        else:
            return ''
    
    ceilref1='cd_ = lambda x, dv: (x + dv - 1)//dv'
    
    @classmethod
    def add_ceildiv(cls,clrf=None):
        if clrf is None:return cls.ceilref1 
        return clrf
    
    @classmethod
    def add_balloc(cls,alloc_eqn,check_alloc='buffer',alignf='aligned_buffer',balign=BufferAlign.PAGE):
        return f'''if {check_alloc} is None:
        {check_alloc} = {alignf}({gen_str(alloc_eqn)}, {balign})'''


def ls_layers(layers):
    """
    Turn layers into python tuple assignment strings:
      w0, w2, ... = expr0, expr2, ...
    One line per layer (even if single element).
    """
    lines = []
    for layer in layers:
        lhs = ", ".join(v.name for v, _ in layer)
        rhs = ", ".join(gen_str(e) for _, e in layer)
        lines.append(f"{lhs} = {rhs}")
    lines.append('')
    return lines

def lyr_str_key(lyr):
    #later on change this to an expression "does it contain function" instead of expensive string call.
    s = lyr
    if "cd" in s:
        r = 3
    elif "fd" in s:
        r = 2
    elif "(" in s:
        r = 1
    else:
        r = 0
    return r

def post_process_cse(repls, reduced, symbols,  oneassign_del=True, clean_order=True):
    """After cse, we group temps into tuple assignment layers based on their own dependencies. 
    
    :param clean_order: Groups functional declarations like ceiling div and floor div together per layer. And renames variables to be ordered at the assignment block.
    :param oneassign_del: cse sometimes creates assignment variables that are only referenced once. This will delete those variables and place their value declarations directly in the reference. Can reduce temp variables without increasing the represented arithmetic operations.
    """
    temps = [lhs for lhs, _ in repls]
    pending = set(temps)

    rhs_map = {lhs: rhs for lhs, rhs in repls}
    deps = {
        lhs: {s for s in rhs.free_symbols if s in pending}
        for lhs, rhs in repls
    }

    available = set()
    layers = []

    while pending:
        ready = [t for t in temps if t in pending and deps[t] <= available]
        if not ready:
            cycle = [t for t in temps if t in pending]
            raise RuntimeError(f"Could not layer assignments (cycle/unresolved deps). Remaining: {cycle}")

        # ready.sort(key=layer_sort_key)
        layers.append([[r, rhs_map[r], lyr_str_key(str(rhs_map[r]))] for r in ready])
        

        pending.difference_update(ready)
        available.update(ready)
    
    if oneassign_del:
        red_syms=[e.free_symbols for e in reduced]
        # for r in reduced:
        #     print(r)
        for i in range(len(layers)-1,-1,-1):
            #we need to: delete layers with only 1, and replace its symbol with the layer ahead of it
            #or within reduced.
            #because we use red_syms and we use deps we need to update or delete them from there too..
            tpl=layers[i]
            dg=[]
            for n in range(len(tpl)):
                lyg=tpl[n]
                if lyg[2]==2:continue
                lysm=lyg[0]
                rmi=None
                ipos=[]
                
                #Is there only one in the reduced equations
                for m in range(len(red_syms)):
                    if lysm in red_syms[m]:
                        ipos.append(m)
                        if rmi is not None:break
                        rmi=n
                lpos=[]
                #Is there only one in the dependent layers.
                for r in range(i+1,len(layers)):
                    rly=layers[r]
                    for k in range(len(rly)):
                        if lysm in deps[rly[k][0]]:
                            lpos.append((r,k))
                            if rmi is not None:break
                            rmi=n
                tl=len(ipos)+len(lpos)
                if tl==1:
                    dg.append(rmi)
                    print('ipos',len(ipos))
                    if len(lpos)==0:
                        rep=ipos[0]
                        reduced[rep]=reduced[rep].xreplace({lysm:lyg[1]})
                        rs=red_syms[rep]
                        red_syms[rep]=(rs-{lysm})|deps[lysm]
                        deps.pop(lysm)
                    else:
                        rp,kp=lpos[0]
                        rly=layers[rp]
                        rly[kp][1]=rly[kp][1].xreplace({lysm:lyg[1]})
                        rk=rly[kp][0]
                        rs=deps[rk]
                        deps[rk]=(rs-{lysm})|deps[lysm]
                        deps.pop(lysm)
            for v in dg[::-1]:
                tpl.pop(v)
    
    
    repd={}
    for lyr in layers:
        lyr.sort(key=lambda x: x[-1])
        for sg in lyr:
            if clean_order:
                sm=next(symbols)
                repd[sg[0]]=sm
                sg[0]=sm
            sg.pop()
    if clean_order:
        for i in range(len(reduced)):
            reduced[i]=reduced[i].xreplace(repd)  
        for lyr in layers:
            for sg in lyr:
                sg[1]=sg[1].xreplace(repd)
    

    return layers,reduced

def cse_codereduction(exprs, prefix='t', optimizations="basic", symbols=numb_syms):
    exprs_wrapped = cf_plcsym(exprs)
    repls, reduced = sym.cse(exprs_wrapped, symbols=symbols(prefix), optimizations=optimizations)
    layered_assigns,reduced=post_process_cse(repls, reduced, symbols(prefix))
    return layered_assigns,reduced
    

def _chkfalign(fshape, forder, shape, order, align):
    """Decide whether to reuse an Array spec's aligned-leading-dimension.

    Returns ``True`` when the requested ``shape``/``order`` match the base
    spec (or are ``None``/unspecified) **and** the ``align`` parameter was not
    explicitly provided (``UO`` sentinel). Otherwise, the caller is
    signaling intent to change layout, so the previous alignment is dropped.
    """
    # change this later so that if it's reversed dimensions in forder != order we keep align
    return (
            align is UO
            and (shape is None or fshape == shape)
            and (order is None or forder == order)
    )

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


def dtype_abbr(dtype: np.dtype) -> str:
    """Return a short ``kind_bits`` abbreviation (e.g., ``'f_64'``,
    ``'i_32'``)."""
    kind = dtype.kind.lower()  # e.g. 'i', 'u', 'f', 'c', 'b', 'M', 'm', …
    bits = dtype.itemsize * 8  # itemsize is in bytes → *8 for bits
    return f"{kind}_{bits}"
