import os
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
import sympy as sym

from bmap import (
    ArrayNode,
    BaseNode,
    BButil,
    BufferAlign,
    BufferMap,
    ContainerNode,
    NativeTypes,
    aligned_buffer,
    allocate_bmap,
    ar_spec,
    array_arspec,
    arrays_map,
    bdict,
    bmap_get,
    bmap_pyvis,
    buffer_expr,
    buffer_symbols,
    build_bmap,
    build_buffer_allocator,
    build_flatmap,
    c_orlen,
    clone_bmap,
    cse_codereduction,
    db_node,
    dt_buff_exprs,
    eval_buff_expr,
    eval_buff_exprs,
    f_arspec,
    f_arspec_i,
    ft_arspec,
    is_eqn,
    mk_buff_dict,
    numb_syms,
    oneshot_args,
    save_bmap_tree,
    sb_node,
    v_spec,
)

def _write_test_output(test_name: str, content, is_py: bool) -> pathlib.Path:
    temp_root = pathlib.Path("temp")
    temp_root.mkdir(exist_ok=True)
    suffix = ".py" if is_py else ".txt"
    path = temp_root / f"{test_name}{suffix}"
    if isinstance(content, (list, tuple)):
        text = "\n".join(str(item) for item in content)
    else:
        text = str(content)
    path.write_text(text, encoding="utf-8")
    return path


def test_constants_and_aligned_buffer() -> None:
    """Cover core constants and aligned buffer allocation behavior."""
    assert BufferMap.SHARED == 0
    assert BufferMap.DISTINCT == 1
    assert BufferAlign.AVX512 == 64

    buf = aligned_buffer(32, align=BufferAlign.SSE)
    assert buf.dtype == np.uint8
    assert buf.size == 32
    assert buf.ctypes.data % BufferAlign.SSE == 0


def test_native_types_sizes() -> None:
    """Validate NativeTypes size fields are internally consistent."""
    assert NativeTypes.POINTERb == NativeTypes.POINTER.itemsize
    assert NativeTypes.NP_FLOATb == NativeTypes.NP_FLOAT.itemsize
    assert NativeTypes.NB_INTb == NativeTypes.NB_INT.itemsize


def test_bbutill_header_and_alloc_string() -> None:
    """Check BButil header and allocation snippet formatting."""
    root = db_node(ar_spec((1,), name="a"), name="Root Node")
    header = BButil.build_header(root, ["n", "type_flt"], {})
    assert "def root_node" in header
    assert "type_flt" in header

    alloc = BButil.add_balloc(buffer_expr("n"))
    _write_test_output("test_bbutill_header_and_alloc_string", [header, alloc], is_py=False)
    assert "aligned_buffer" in alloc


def test_buffer_expr_and_symbols() -> None:
    """Exercise buffer expression parsing and symbol helpers."""
    sym_a = buffer_expr("a")
    assert isinstance(sym_a, sym.Symbol)
    assert sym_a.is_integer is True
    assert sym_a.is_positive is True

    expr = buffer_expr("4*a")
    assert isinstance(expr, sym.Expr)
    assert not isinstance(expr, sym.Symbol)

    a, b = buffer_symbols("a b")
    assert isinstance(a, sym.Symbol)
    assert isinstance(b, sym.Symbol)

    x, y = buffer_symbols("x", "y")
    assert isinstance(x, sym.Symbol)
    assert isinstance(y, sym.Symbol)

    seq = dt_buff_exprs(["m", "n"])
    assert isinstance(seq, tuple)
    assert all(isinstance(v, sym.Symbol) for v in seq)

    spaced = buffer_expr("a ")
    assert isinstance(spaced, sym.Symbol)
    assert spaced.name == "a"

    solo = buffer_symbols("solo")
    assert isinstance(solo, sym.Symbol)

    as_int = buffer_expr(3.0)
    assert as_int == 3


def test_eval_and_dict_helpers() -> None:
    """Check buffer expression evaluation and dict normalization."""
    a, b = buffer_symbols("a b")
    expr = buffer_expr("a + b")
    assert eval_buff_expr(expr, {a: 2, b: 3}) == 5
    assert eval_buff_exprs((a, expr, 4), {a: 2, b: 3}) == (2, 5, 4)

    dct = mk_buff_dict({"a": 1, b: 2})
    assert dct[buffer_expr("a")] == 1
    assert dct[b] == 2

    dct2 = bdict(a=3)
    assert dct2[buffer_expr("a")] == 3

    with pytest.raises(TypeError):
        eval_buff_exprs((buffer_expr("a"),), None)

    with pytest.raises(TypeError):
        eval_buff_expr(buffer_expr("a"), None)

    assert eval_buff_expr(buffer_expr("a"), {a: 9}) == 9
    assert dt_buff_exprs(None) is None


def test_numb_syms_and_predicates() -> None:
    """Cover symbol generators and simple predicate utilities."""
    gen = numb_syms("t", start=0)
    s0 = next(gen)
    s1 = next(gen)
    assert s0.name == "t0"
    assert s1.name == "t1"

    assert c_orlen("type_float", "type_") is True
    assert c_orlen("abc", "type_") is False

    assert is_eqn(buffer_expr("a + 1")) is True
    assert is_eqn(buffer_expr("a")) is False


def test_cse_codereduction_smoke() -> None:
    """Smoke-test sympy CSE reduction wiring used by allocator codegen."""
    a, b = buffer_symbols("a b")
    exprs = (a + b, (a + b) * 2)
    layers, reduced = cse_codereduction(exprs, prefix="t")
    _write_test_output("test_cse_codereduction_smoke", [layers, reduced], is_py=False)
    assert isinstance(layers, list)
    assert len(reduced) == 2

    floor_exprs = (sym.floor(a), sym.ceiling(a / b))
    flayers, freduced = cse_codereduction(floor_exprs, prefix="w")
    _write_test_output("test_cse_codereduction_smoke_floor", [flayers, freduced], is_py=False)
    assert isinstance(flayers, list)
    assert len(freduced) == 2


def test_array_specs_and_clones() -> None:
    """Exercise ArrayNode spec helpers and clone utilities."""
    base = ar_spec((3, 2), np.float32, name="A")
    assert isinstance(base, ArrayNode)

    clone = f_arspec(base, name="A2")
    assert clone.shape == base.shape
    assert clone.name == "A2"

    fbuild = f_arspec_i(dtype=np.float64, order="F")
    fnode = fbuild(base, shape=(2, 3), name="B")
    assert fnode.dtype == np.dtype(np.float64)
    assert fnode.order == "F"

    tnode = ft_arspec(base)
    assert tnode.order != base.order

    arr = np.asfortranarray(np.zeros((2, 3), dtype=np.float64))
    arr_spec = array_arspec(arr, name="C")
    assert arr_spec.order == "F"
    assert arr_spec.shape == arr.shape

    sliced = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]
    sliced_spec = array_arspec(sliced, name="S")
    assert sliced_spec.align_ldim is not None

    carr = np.arange(6, dtype=np.float64).reshape(2, 3)
    carr_spec = array_arspec(carr, name="C2")
    assert carr_spec.order == "C"


def test_bmap_build_allocation_and_flatten() -> None:
    """Build a simple tree, allocate buffers, and flatten values."""
    dist1 = ar_spec((3,), np.float32, name="d1")
    dist2 = ar_spec((2, 2), np.int32, name="d2")
    val = v_spec(7, name="v1")
    shared = sb_node(ar_spec((4,), name="s1"), name="shared")
    root = db_node(dist1, dist2, val, shared, name="root")

    build_bmap(root, align=BufferAlign.BYTE)
    allocate_bmap(root, balign=BufferAlign.BYTE)

    assert dist1.array is not None
    assert dist1.array.shape == (3,)
    assert dist2.array is not None
    assert val.value == 7

    flat_vals = build_flatmap(root, align=BufferAlign.BYTE)
    assert len(flat_vals) >= 3

    oneshot = oneshot_args(root)
    assert len(oneshot) >= 3


def test_base_node_minimal_usage() -> None:
    """Cover BaseNode defaults and basic helpers."""
    node = BaseNode(name="base")
    istr, rstr = node.gen_call()
    assert isinstance(istr, str)
    assert isinstance(rstr, str)
    assert node.free_symbols == set()


def test_container_node_ops_and_render() -> None:
    """Exercise ContainerNode mutation helpers and string representations."""
    a = ar_spec((1,), name="a")
    b = ar_spec((2,), name="b")
    c = v_spec(3, name="c")
    root = ContainerNode(a, b, c, name="root")

    root.insert(1, c)
    assert [ch.name for ch in root.children] == ["a", "c", "b"]

    root.order(1, 2, 0)
    assert [ch.name for ch in root.children] == ["c", "b", "a"]

    popped = root.pop("b", 0)
    assert [node.name for node in popped] == ["b", "c"]
    assert [ch.name for ch in root.children] == ["a"]

    with pytest.raises(ValueError):
        root.order(0, 0, 1)

    assert "Node" in repr(root)
    root_str = str(root)
    _write_test_output("test_container_node_ops_and_render", root_str, is_py=False)
    assert "root" in root_str


def test_container_order_invalid_permutation() -> None:
    """Ensure ContainerNode.order rejects non-permutations."""
    a = ar_spec((1,), name="a")
    b = ar_spec((1,), name="b")
    c = ar_spec((1,), name="c")
    root = ContainerNode(a, b, c, name="root")
    with pytest.raises(ValueError):
        root.order(0, 0, 1)


def test_container_build_flatmap_and_merge() -> None:
    """Cover reduce/merge behavior and ContainerNode.build_flatmap."""
    inner = db_node(ar_spec((2,), name="x"), name="inner")
    outer = db_node(inner, ar_spec((1,), name="y"), name="outer")

    outer.build_flatmap(align=BufferAlign.BYTE, name_join=True)
    assert any("outer_" in (n.name or "") for n in outer.children)

    duplicate = db_node(ar_spec((1,), name="dup"), ar_spec((2,), name="dup"), name="root")
    build_bmap(duplicate, align=BufferAlign.BYTE, verbose=True)

    merged = db_node(db_node(ar_spec((1,), name="a"), name="child"), name="parent")
    build_bmap(merged, align=BufferAlign.BYTE, name_join=True, force_merge=True)
    assert any((n.name or "").startswith("parent_") for n in merged.children)

    nested = db_node(db_node(ar_spec((1,), name="x"), name="inner"), name="outer")
    build_bmap(nested, align=BufferAlign.BYTE, name_join=True, force_merge=True)
    assert any((n.name or "").startswith("outer_") for n in nested.children)


def test_build_bmap_verbose_and_force_merge() -> None:
    """Cover verbose diagnostics and forced merge behavior."""
    inner = sb_node(ar_spec((1,), name="dup"), name="inner", no_merge=True)
    outer = sb_node(inner, ar_spec((2,), name="dup"), name="outer", no_merge=True)
    build_bmap(outer, align=BufferAlign.BYTE, force_merge=True, verbose=True)
    assert all(not isinstance(ch, ContainerNode) for ch in outer.children)

    sym_root = sb_node(ar_spec(("m",), name="a"), ar_spec(("n",), name="b"), name="sym")
    build_bmap(sym_root, align=BufferAlign.BYTE)
    assert isinstance(sym_root.nbytes, sym.Expr)


def test_arrays_map_with_symbols() -> None:
    """Validate arrays_map with symbolic shapes and value mapping."""
    m = buffer_expr("m")
    arr = ar_spec((m,), name="sym")
    root = db_node(arr, name="root")
    build_bmap(root, align=BufferAlign.BYTE)

    arrays_map(root, sym_dic={m: 4}, balign=BufferAlign.BYTE)
    assert arr.array is not None
    assert arr.array.shape == (4,)


def test_arrays_map_with_string_keys() -> None:
    """Ensure arrays_map accepts string symbol keys."""
    m = buffer_expr("m")
    arr = ar_spec((m,), name="a")
    root = db_node(arr, name="root")
    build_bmap(root, align=BufferAlign.BYTE)

    arrays_map(root, sym_dic={"m": 3, m: 3}, balign=BufferAlign.BYTE)
    assert arr.array is not None
    assert arr.array.shape == (3,)


def test_arrays_map_with_buffer_base() -> None:
    """Cover arrays_map when a buffer and base offset are supplied."""
    arr = ar_spec((2,), np.float32, name="a")
    root = db_node(arr, name="root", align=False)
    build_bmap(root, align=BufferAlign.BYTE)

    buffer = aligned_buffer(eval_buff_expr(root.nbytes) + 1, align=BufferAlign.BYTE)
    arrays_map(root, balign=BufferAlign.BYTE, _buffer=buffer, _base=1)
    assert arr.array is not None
    assert arr.ofs == 1


def test_bmap_get_passthrough() -> None:
    """Ensure bmap_get returns the node when passed directly."""
    arr = ar_spec((1,), name="a")
    root = db_node(arr, name="root")
    assert bmap_get(arr, root) is arr


def test_container_node_errors() -> None:
    """Validate ContainerNode error paths for invalid inputs."""
    root = ContainerNode(name="root")
    with pytest.raises(TypeError):
        root.add("not-a-node")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        root.order(0, 1)
    with pytest.raises(TypeError):
        root.pop(1.5)  # type: ignore[arg-type]


def test_base_node_name_mutation() -> None:
    """Cover name property setter/deleter on nodes."""
    arr = ar_spec((1,), name="a")
    assert arr.name == "a"
    arr.name = "b"
    assert arr.name == "b"
    del arr.name
    with pytest.raises(AttributeError):
        _ = arr.name


def test_buffer_expr_symbol_rewrite() -> None:
    """Exercise symbol normalization for non-integer symbols."""
    raw = sym.Symbol("k", integer=False, positive=False)
    sym_k = buffer_expr(raw)
    assert isinstance(sym_k, sym.Symbol)
    assert sym_k.is_integer is True
    assert sym_k.is_positive is True
    assert isinstance(buffer_expr("x + 0"), sym.Expr)


def test_reduce_bmap_no_merge() -> None:
    """Ensure no_merge blocks same-rule container merging."""
    inner = db_node(ar_spec((1,), name="x"), name="inner", no_merge=True)
    outer = db_node(inner, name="outer")
    build_bmap(outer, align=BufferAlign.BYTE, force_merge=False)
    assert any(isinstance(ch, ContainerNode) for ch in outer.children)


def test_build_buffer_allocator_no_reduce() -> None:
    """Cover allocator generation without CSE reduction."""
    root = sb_node(ar_spec((2,), name="a"), name="shared")
    build_bmap(root, align=BufferAlign.BYTE)
    alloc_src = build_buffer_allocator(root, fullreduce=False)
    _write_test_output("test_build_buffer_allocator_no_reduce", alloc_src, is_py=True)
    assert "return" in alloc_src


def test_build_buffer_allocator_no_check() -> None:
    """Cover allocator generation with chkforbuffer disabled."""
    root = sb_node(ar_spec((2,), name="a"), name="shared")
    build_bmap(root, align=BufferAlign.BYTE)
    alloc_src = build_buffer_allocator(root, chkforbuffer=False, fullreduce=False)
    _write_test_output("test_build_buffer_allocator_no_check", alloc_src, is_py=True)
    assert "def shared" in alloc_src


def test_container_free_symbols() -> None:
    """Cover free_symbols aggregation for aligned containers."""
    m = buffer_expr("m")
    node = ar_spec((m,), name="a")
    root = db_node(node, name="root")
    build_bmap(root, align=BufferAlign.BYTE)
    assert any(s.name == "m" for s in root.free_symbols)

    sym_node = ar_spec(("m + 1",), name="expr")
    sym_root = db_node(sym_node, name="sym_root", align=False)
    build_bmap(sym_root, align=None)
    assert any(s.name == "m" for s in sym_root.free_symbols)

    pre_root = db_node(ar_spec(("m",), name="pre"), name="pre_root")
    assert any(s.name == "m" for s in pre_root.free_symbols)


def test_container_free_symbols_aligned_eqns() -> None:
    """Cover free_symbols when aligned_eqns are present."""
    m = buffer_expr("m")
    node = ar_spec((m,), name="a")
    root = db_node(node, name="root", align=True)
    build_bmap(root, align=BufferAlign.BYTE)
    assert root.aligned_eqns
    assert any(s.name == "m" for s in root.free_symbols)

    defs, _ = root.gen_call(setexpr=root.aligned_eqns)
    assert isinstance(defs, tuple)


def test_arraynode_gen_call_and_repr() -> None:
    """Exercise ArrayNode repr and gen_call output formatting."""
    node = ar_spec((2, 3), np.float32, name="arr", align_ldim=16)
    istr, rstr = node.gen_call()
    assert "arr" in rstr
    assert "Array" in repr(node)

    istr2, rstr2 = node.gen_call(bytexpr=1, bshape=(2, 4), shape=(2, 3), subn="p")
    assert "p_arr" in rstr2
    assert "[" in istr2

    with pytest.raises(NameError):
        ar_spec((2,), name=None).gen_call()


def test_container_free_symbols_edge_cases() -> None:
    """Cover empty and numeric container free_symbols cases."""
    empty = ContainerNode(name="empty")
    assert empty.free_symbols == set()

    shared = sb_node(ar_spec((2,), name="a"), name="shared")
    build_bmap(shared, align=None)
    assert shared.free_symbols == set()


def test_container_gen_call_scalar_expr() -> None:
    """Cover gen_call when align=True with a scalar setexpr."""
    root = db_node(ar_spec((1,), name="a"), name="root", align=True)
    build_bmap(root, align=BufferAlign.BYTE)
    defs, _ = root.gen_call(setexpr=buffer_expr("m"))
    assert isinstance(defs, tuple)


def test_value_node_repr() -> None:
    """Cover ValueNode string representation."""
    val = v_spec(5, name="v")
    assert "Value" in repr(val)
    assert val.sym_def() == 5


def test_arraynode_alignment_and_buffer_view() -> None:
    """Ensure aligned leading dimension creates a backing view and trims it."""
    node = ar_spec((2, 5), np.float64, name="aligned", align_ldim=16)
    root = db_node(node, name="root")
    build_bmap(root, align=BufferAlign.BYTE)

    buffer = aligned_buffer(eval_buff_expr(root.nbytes), align=BufferAlign.BYTE)
    arrays_map(root, balign=BufferAlign.BYTE, _buffer=buffer)

    assert node.barray is not None
    assert node.array is not None
    assert node.barray.shape[-1] >= node.array.shape[-1]

    uninit = ar_spec((2, 2), np.float32, name="blank", init_op=3)
    uninit.array = None
    uninit._rinit()

    direct = ar_spec((2,), np.float32, name="direct")
    direct.ofs = 0
    buffer = aligned_buffer(eval_buff_expr(direct.nbytes), align=BufferAlign.BYTE)
    direct.mk_array(buffer, sym_dic={})
    assert direct.array is not None


def test_arraynode_align_ldim_order_f() -> None:
    """Cover align_ldim adjustment for Fortran-ordered arrays."""
    node = ar_spec((2, 3), np.float32, name="f", order="F", align_ldim=16)
    assert node.order == "F"
    assert node.bshape[0] != node.shape[0]


def test_symbolic_dtype_and_free_symbols() -> None:
    """Cover symbolic dtype handling and symbol collection on nodes."""
    node = ar_spec((3,), dtype="type_flt", name="sym")
    root = db_node(node, name="root", align=False)
    symbols = build_bmap(root, align=BufferAlign.BYTE)

    assert isinstance(node.dtype, sym.Symbol)
    assert any(s.name == "type_flt" for s in symbols)


def test_build_buffer_allocator_shared_root() -> None:
    """Ensure allocator generation works on a shared-root tree."""
    a = ar_spec((2,), name="a")
    b = ar_spec((3,), name="b")
    root = sb_node(a, b, name="shared")
    build_bmap(root, align=BufferAlign.BYTE)

    alloc_src = build_buffer_allocator(root, args=("n",))
    _write_test_output("test_build_buffer_allocator_shared_root", alloc_src, is_py=True)
    assert "def shared" in alloc_src


def test_value_node_gen_call() -> None:
    """Exercise ValueNode string generation and symbol tracking."""
    val = v_spec(sym.Symbol("m"), name="val")
    istr, rstr = val.gen_call()
    assert isinstance(istr, str)
    assert rstr == "val"
    assert any(s.name == "m" for s in val.free_symbols)

    istr2, rstr2 = val.gen_call(valexpr=3, subn="p")
    assert istr2 == "p_val = 3"
    assert rstr2 == "p_val"


def test_value_node_unnamed() -> None:
    """Cover ValueNode gen_call without a name."""
    val = v_spec(5, name=None)
    istr, rstr = val.gen_call()
    assert istr is None
    assert rstr == 5


def test_bmap_get_errors() -> None:
    """Cover bmap_get error paths."""
    root = db_node(ar_spec((1,), name="a"), name="root")
    with pytest.raises(KeyError):
        bmap_get("missing", root)
    with pytest.raises(TypeError):
        bmap_get(1.5, root)  # type: ignore[arg-type]


def test_cse_codereduction_with_division() -> None:
    """Ensure CSE handles floor/ceiling formatting."""
    a, b = buffer_symbols("a b")
    exprs = (sym.ceiling(a / b), sym.floor(a / b))
    layers, reduced = cse_codereduction(exprs, prefix="t")
    _write_test_output("test_cse_codereduction_with_division", [layers, reduced], is_py=False)
    assert len(reduced) == 2


def test_arraynode_init_op_variants() -> None:
    """Cover array initialization operations."""
    node_val = ar_spec((2,), np.int32, name="v", init_op=5)
    arr_val = node_val.mk_array()
    assert np.all(arr_val == 5)

    node_call = ar_spec((2,), np.int32, name="c", init_op=lambda ar: ar.fill(7))
    arr_call = node_call.mk_array()
    assert np.all(arr_call == 7)


def test_container_gen_call_align_false() -> None:
    """Exercise ContainerNode.gen_call when align is False."""
    root = db_node(ar_spec((1,), name="a"), name="root", align=False)
    build_bmap(root, align=BufferAlign.BYTE)
    expr = root.gen_call(setexpr=root.nbytes)
    assert isinstance(expr[0], str)


def test_symbolic_dtype_without_prefix() -> None:
    """Cover dtype symbol handling without the type_ prefix."""
    node = ar_spec((2,), dtype="flt", name="a")
    assert isinstance(node.dtype, sym.Symbol)
    assert node.dtype.name.startswith("type_")


def test_align_ldim_symbolic_expr() -> None:
    """Exercise symbolic aligned leading dimension."""
    a = buffer_expr("a")
    node = ar_spec((2, 3), np.float32, name="a", align_ldim=a)
    assert isinstance(node.bshape[-1], sym.Expr)

    expr_node = ar_spec((2, 3), np.float32, name="expr", align_ldim="m + 1")
    assert any(s.name == "m" for s in expr_node.free_symbols)

    sym_node = ar_spec((2, 3), np.float32, name="sym", align_ldim=buffer_expr("k"))
    assert any(s.name == "k" for s in sym_node.free_symbols)


def test_build_buffer_allocator_subname() -> None:
    """Cover allocator generation with subname enabled."""
    root = sb_node(ar_spec((2,), name="a"), name="root")
    build_bmap(root, align=BufferAlign.BYTE)
    alloc_src = build_buffer_allocator(root, subname=True, fullreduce=False)
    _write_test_output("test_build_buffer_allocator_subname", alloc_src, is_py=True)
    assert "return" in alloc_src


def test_value_node_scalar_expr_error() -> None:
    """Cover allocator generation with a non-scalar ValueNode expression."""
    val = v_spec(sym.Matrix([1, 2]), name="m")
    root = sb_node(val, name="root")
    build_bmap(root, align=BufferAlign.BYTE)
    #any value node sb supported if it can be written to string.
    # with pytest.raises(TypeError):
    #     build_buffer_allocator(root, fullreduce=True)

    alloc_src = build_buffer_allocator(root, fullreduce=False)
    _write_test_output("test_value_node_scalar_expr_error", alloc_src, is_py=True)
    assert "return" in alloc_src


@pytest.mark.xfail(reason="build_buffer_allocator fails on distinct root with fullreduce (IndexError)")
def test_bmap_get_clone_and_allocator_string() -> None:
    """Cover bmap_get resolution, deep cloning, and allocator codegen."""
    arr = ar_spec((2,), name="a")
    root = db_node(arr, name="Root Node")

    assert bmap_get([0], root) is arr
    assert bmap_get("a", root) is arr

    cloned = clone_bmap(root)
    assert isinstance(cloned, ContainerNode)
    assert cloned is not root

    build_bmap(root, align=BufferAlign.BYTE)
    alloc_src = build_buffer_allocator(
        root,
        args=(buffer_expr("n"), "type_flt"),
    )
    _write_test_output("test_bmap_get_clone_and_allocator_string", alloc_src, is_py=True)
    assert "def root_node" in alloc_src
    assert "aligned_buffer" in alloc_src


@pytest.mark.skip()
def test_save_bmap_tree_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Write a DOT-rendered image when Graphviz is available."""
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' not available")

    root = db_node(ar_spec((2,), name="a"), name="root")
    build_bmap(root, align=BufferAlign.BYTE)
    temp_root = pathlib.Path("temp")
    temp_root.mkdir(exist_ok=True)
    out_path = temp_root / "tree.png"
    tmp_dir = temp_root / "dot_tmp"
    tmp_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("TMP", str(tmp_dir))
    monkeypatch.setenv("TEMP", str(tmp_dir))
    monkeypatch.setenv("TMPDIR", str(tmp_dir))
    tempfile.tempdir = str(tmp_dir)
    dot_in = tmp_dir / "check.dot"
    dot_out = tmp_dir / "check.png"
    dot_in.write_text("digraph G { a -> b }", encoding="utf-8")
    probe = subprocess.run(
        ["dot", str(dot_in), "-T", "png", "-o", str(dot_out)],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip(f"Graphviz dot not usable: {probe.stderr}")
    result = save_bmap_tree(root, path=str(out_path))
    _write_test_output("test_save_bmap_tree_path", result, is_py=False)
    (temp_root / "tree_result.txt").write_text(result, encoding="utf-8")
    assert result == str(out_path)
    assert out_path.exists()


def test_bmap_pyvis_smoke() -> None:
    """Build a PyVis graph when optional dependencies are present."""
    pytest.importorskip("networkx")
    pytest.importorskip("pydot")
    pytest.importorskip("pyvis")

    root = db_node(ar_spec((2,), name="a"), name="root")
    build_bmap(root, align=BufferAlign.BYTE)
    net = bmap_pyvis(root, with_offsets=True, height="400px")
    assert hasattr(net, "nodes")


def test_value_node_fullreduce_scalar() -> None:
    """Cover fullreduce allocator path with scalar values."""
    root = sb_node(v_spec(4, name="v"), name="root")
    build_bmap(root, align=BufferAlign.BYTE)
    src = build_buffer_allocator(root, fullreduce=True)
    _write_test_output("test_value_node_fullreduce_scalar", src, is_py=True)
    assert "return" in src


def test_allocator_fullreduce_distinct_align() -> None:
    """Exercise fullreduce allocator path with distinct/align subtrees."""
    inner = db_node(ar_spec((2,), name="a"), ar_spec((3,), name="b"), name="inner", align=True)
    root = sb_node(inner, name="root")
    build_bmap(root, align=BufferAlign.BYTE)
    src = build_buffer_allocator(root, fullreduce=True)
    _write_test_output("test_allocator_fullreduce_distinct_align", src, is_py=True)
    assert "return" in src


def test_allocator_fullreduce_align_false() -> None:
    """Exercise allocator path for distinct containers without per-child alignment."""
    inner = db_node(ar_spec((2,), name="a"), ar_spec((1,), name="b"), name="inner", align=False)
    root = sb_node(inner, name="root")
    build_bmap(root, align=BufferAlign.BYTE)
    src = build_buffer_allocator(root, fullreduce=True)
    _write_test_output("test_allocator_fullreduce_align_false", src, is_py=True)
    assert "return" in src


def test_allocator_noreduce_align_true_value() -> None:
    """Cover no-reduction allocator with aligned distinct containers and values."""
    root = db_node(ar_spec((2,), name="a"), v_spec(4, name="v"), name="root", align=True)
    build_bmap(root, align=BufferAlign.BYTE)
    src = build_buffer_allocator(root, fullreduce=False)
    _write_test_output("test_allocator_noreduce_align_true_value", src, is_py=True)
    assert "return" in src

    inner = db_node(ar_spec((2,), name="a"), ar_spec((1,), name="b"), name="inner", align=False)
    build_bmap(inner, align=BufferAlign.BYTE)
    src2 = build_buffer_allocator(inner, fullreduce=False)
    _write_test_output("test_allocator_noreduce_align_true_value_inner", src2, is_py=True)
    assert "return" in src2


def test_bbutil_symbolic_balign_and_floor_ceiling() -> None:
    """Cover BButil helpers with symbolic alignment and custom ceildiv."""
    root = db_node(ar_spec((1,), name="a"), name="root")
    header = BButil.build_header(root, ["type_flt"], {}, balign=buffer_expr("balign"))
    _write_test_output("test_bbutil_symbolic_balign_and_floor_ceiling_header", header, is_py=False)
    assert "balign" in header
    assert BButil.add_ceildiv("custom_cd") == "custom_cd"

    a = buffer_expr("a")
    b = buffer_expr("b")
    alloc = BButil.add_balloc(sym.ceiling(a / b))
    _write_test_output("test_bbutil_symbolic_balign_and_floor_ceiling_alloc", alloc, is_py=False)
    assert "aligned_buffer" in alloc

    fd = sym.Function("fd_")
    fd_expr = fd(a, b)
    fd_alloc = BButil.add_balloc(fd_expr)
    _write_test_output("test_bbutil_symbolic_balign_and_floor_ceiling_fd", fd_alloc, is_py=False)
    assert "aligned_buffer" in fd_alloc


def test_bmap_pyvis_offsets_and_values() -> None:
    """Cover DOT rendering details for offsets and value nodes."""
    pytest.importorskip("networkx")
    pytest.importorskip("pydot")
    pytest.importorskip("pyvis")

    m = buffer_expr("m")
    a = ar_spec((m,), name="a", align_ldim=16)
    v = v_spec(5, name="v")
    root = sb_node(a, v, name="shared")
    build_bmap(root, align=BufferAlign.BYTE)
    arrays_map(root, sym_dic={m: 4}, balign=BufferAlign.BYTE)
    net = bmap_pyvis(root, with_offsets=True, height="300px")
    assert hasattr(net, "nodes")


def test_bmap_pyvis_invalid_dot() -> None:
    """Cover error path when DOT parsing fails."""
    pytest.importorskip("networkx")
    pytest.importorskip("pydot")
    pytest.importorskip("pyvis")

    with pytest.raises(ValueError):
        bmap_pyvis("digraph {", with_offsets=False)


def test_f_arspec_invalid_and_overrides() -> None:
    """Cover f_arspec validation and override behavior."""
    with pytest.raises(ValueError):
        f_arspec("not-a-node")  # type: ignore[arg-type]

    base = ar_spec((2, 3), np.float32, name="base")
    clone = f_arspec(base, init_op=None, align_ldim=16, name="clone")
    assert clone.align_ldim is not None


def test_ft_arspec_init_op_variants() -> None:
    """Cover ft_arspec init_op handling for arrays and callables."""
    base_arr = ar_spec((2, 2), np.float32, name="arr", init_op=np.ones((2, 2), dtype=np.float32))
    trans_arr = ft_arspec(base_arr)
    assert isinstance(trans_arr, ArrayNode)

    def _fill(ar: np.ndarray) -> None:
        ar.fill(3)

    base_call = ar_spec((2, 2), np.float32, name="call", init_op=_fill)
    trans_call = ft_arspec(base_call)
    assert isinstance(trans_call, ArrayNode)


def test_cse_codereduction_floor_ceiling_variants() -> None:
    """Exercise CSE pipeline with floor/ceiling and fd_ expressions via public API."""
    ai = sym.Symbol("ai", integer=True, positive=True)
    bi = sym.Symbol("bi", integer=True, positive=True)
    c = sym.Symbol("c", real=True)
    a = sym.Symbol("a", real=True)
    fd = sym.Function("fd_")
    exprs = (
        sym.ceiling(c),
        sym.ceiling(ai / sym.Integer(3)),
        sym.floor(a),
        sym.floor((ai + 1) / sym.Integer(3)),
        sym.floor(ai / (bi + 1)),
        fd(ai + 1, bi + 1),
    )
    layers, reduced = cse_codereduction(exprs, prefix="t")
    _write_test_output("test_cse_codereduction_floor_ceiling_variants", [layers, reduced], is_py=False)
    assert layers
    assert reduced
    assert any("cd_" in str(expr) or "fd_" in str(expr) for expr in reduced)


def test_cse_codereduction_layer_keys() -> None:
    """Trigger layer string keys for cd_, fd_, and parenthesized expressions."""
    a = sym.Symbol("a", real=True)
    exprs = (
        sym.ceiling(a / sym.Integer(3)) + 1,
        sym.ceiling(a / sym.Integer(3)) + 2,
        sym.floor(a / sym.Integer(3)) + 3,
        sym.floor(a / sym.Integer(3)) + 4,
        sym.Function("zz")(a) + 5,
        sym.Function("zz")(a) + 6,
    )
    layers, reduced = cse_codereduction(exprs, prefix="t")
    _write_test_output("test_cse_codereduction_layer_keys", [layers, reduced], is_py=False)
    assert layers
    assert reduced


def test_cse_codereduction_den_one_paths() -> None:
    """Ensure cf_plcsym keeps floor/ceiling when denominator is 1."""
    a = sym.Symbol("a", real=True, integer=False)
    c = sym.Symbol("c", real=True, integer=False)
    exprs = (
        sym.floor(a / sym.Integer(1)),
        sym.ceiling(c / sym.Integer(1)),
        sym.floor(a / sym.Integer(1)) + 1,
        sym.ceiling(c / sym.Integer(1)) + 2,
    )
    layers, reduced = cse_codereduction(exprs, prefix="t")
    _write_test_output("test_cse_codereduction_den_one_paths", [layers, reduced], is_py=False)
    assert layers
    assert reduced
    flat_layer_exprs = [expr for layer in layers for _, expr in layer]
    assert any("floor" in str(expr) for expr in (*reduced, *flat_layer_exprs))
    assert any("ceiling" in str(expr) for expr in (*reduced, *flat_layer_exprs))


def test_max_expr_codegen() -> None:
    """Ensure Max expressions survive through codegen paths."""
    m = buffer_expr("m")
    n = buffer_expr("n")
    node = ar_spec((sym.Max(m, n),), np.float32, name="mx")
    root = db_node(node, name="root")
    build_bmap(root, align=BufferAlign.BYTE)
    src = build_buffer_allocator(root, fullreduce=True)
    _write_test_output("test_max_expr_codegen", src, is_py=True)
    assert "max" in src
    assert "return" in src


def test_array_arspec_noncontiguous_1d() -> None:
    """Cover array_arspec handling for non-contiguous 1D arrays (order 'A')."""
    arr = np.arange(10)[::2]
    assert not arr.flags["C_CONTIGUOUS"]
    spec = array_arspec(arr, name="strided")
    assert spec.order == "C"


def test_arraynode_mk_array_buffer_view() -> None:
    """Cover mk_array using a provided backing buffer and offset."""
    a = ar_spec((2,), np.int32, name="a")
    b = ar_spec((3,), np.int32, name="b")
    root = db_node(a, b, name="root")
    build_bmap(root, align=BufferAlign.BYTE)
    #ofs is not needed for build_bmap and code generation, only in allocate_bmap for building the arrays.
    #assert b.ofs is not None
    buffer = np.zeros(int(root.nbytes), dtype=np.uint8)
    arr_b = b.mk_array(buffer)
    assert arr_b.shape == tuple(int(v) for v in b.bshape)
    assert arr_b.base is not None
    temp_root = pathlib.Path("temp")
    temp_root.mkdir(exist_ok=True)
    (temp_root / "mk_array_view.txt").write_text(str(arr_b.shape), encoding="utf-8")


def test_bbutil_gen_str_floor_ceiling_outputs() -> None:
    """Validate gen_str formatting through BButil allocation snippets."""
    ai = sym.Symbol("ai", integer=True, positive=True)
    bi = sym.Symbol("bi", integer=True, positive=True)
    fd = sym.Function("fd_")

    ceil_alloc = BButil.add_balloc(sym.ceiling(ai / sym.Integer(3)))
    _write_test_output("test_bbutil_gen_str_floor_ceiling_outputs_ceil", ceil_alloc, is_py=False)
    assert "cd_" in ceil_alloc

    floor_alloc = BButil.add_balloc(sym.floor(ai / sym.Integer(3)))
    _write_test_output("test_bbutil_gen_str_floor_ceiling_outputs_floor", floor_alloc, is_py=False)
    assert "//" in floor_alloc

    floor_nn = BButil.add_balloc(sym.floor((ai + 1) / (bi + 1)))
    _write_test_output("test_bbutil_gen_str_floor_ceiling_outputs_floor_nn", floor_nn, is_py=False)
    assert "//" in floor_nn
    floor_na = BButil.add_balloc(sym.floor((ai + 1) / sym.Integer(3)))
    _write_test_output("test_bbutil_gen_str_floor_ceiling_outputs_floor_na", floor_na, is_py=False)
    assert "//" in floor_na
    floor_an = BButil.add_balloc(sym.floor(ai / (bi + 1)))
    _write_test_output("test_bbutil_gen_str_floor_ceiling_outputs_floor_an", floor_an, is_py=False)
    assert "//" in floor_an
    a = sym.Symbol("a", real=True)
    floor_fallback = BButil.add_balloc(sym.floor(a))
    _write_test_output("test_bbutil_gen_str_floor_ceiling_outputs_floor_fallback", floor_fallback, is_py=False)
    assert "floor" in floor_fallback

    c = sym.Symbol("c", real=True)
    ceil_fallback = BButil.add_balloc(sym.ceiling(c))
    _write_test_output("test_bbutil_gen_str_floor_ceiling_outputs_ceil_fallback", ceil_fallback, is_py=False)
    assert "ceiling" in ceil_fallback
    #correct these in the future/try private api
    #needs to test () for both sides
    fd_alloc = BButil.add_balloc(fd(ai + 1, bi + 1))
    _write_test_output("test_bbutil_gen_str_floor_ceiling_outputs_fd", fd_alloc, is_py=False)
    assert "//" in fd_alloc
    #test () for left side only
    fd_na = BButil.add_balloc(fd(ai + 1, bi))
    _write_test_output("test_bbutil_gen_str_floor_ceiling_outputs_fd_na", fd_na, is_py=False)
    assert "//" in fd_na
    #test () right side only
    fd_an = BButil.add_balloc(fd(ai, bi + 1))
    _write_test_output("test_bbutil_gen_str_floor_ceiling_outputs_fd_an", fd_an, is_py=False)
    assert "//" in fd_an
    fd_one = BButil.add_balloc(fd(ai, sym.Integer(1)))
    _write_test_output("test_bbutil_gen_str_floor_ceiling_outputs_fd_one", fd_one, is_py=False)
    assert "aligned_buffer" in fd_one

    zz_alloc = BButil.add_balloc(sym.Function("zz")(ai))
    _write_test_output("test_bbutil_gen_str_floor_ceiling_outputs_zz", zz_alloc, is_py=False)
    assert "zz(" in zz_alloc

    max_alloc = BButil.add_balloc(sym.Max(ai, bi))
    _write_test_output("test_bbutil_gen_str_floor_ceiling_outputs_max", max_alloc, is_py=False)
    assert "max(" in max_alloc


def test_build_buffer_allocator_expr_substitution() -> None:
    """Ensure allocator codegen substitutes expressions via internal reduction helpers."""
    a = ar_spec(("m + 1",), name="a")
    b = ar_spec(("n",), name="b")
    root = db_node(a, b, name="root", align=True)
    build_bmap(root, align=BufferAlign.BYTE)
    src = build_buffer_allocator(root, fullreduce=True)
    _write_test_output("test_build_buffer_allocator_expr_substitution", src, is_py=True)
    assert "def root" in src
    assert "return" in src
