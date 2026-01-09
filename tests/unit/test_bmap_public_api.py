import pathlib
import shutil
import subprocess

import numpy as np
import pytest
import sympy as sym

from bmap import (
    ArrayNode,
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
    assert isinstance(layers, list)
    assert len(reduced) == 2


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

    assert "Node" in repr(root)
    assert "root" in str(root)


def test_container_build_flatmap_and_merge() -> None:
    """Cover reduce/merge behavior and ContainerNode.build_flatmap."""
    inner = db_node(ar_spec((2,), name="x"), name="inner")
    outer = db_node(inner, ar_spec((1,), name="y"), name="outer")

    outer.build_flatmap(align=BufferAlign.BYTE, name_join=True)
    assert any("outer_" in (n.name or "") for n in outer.children)


def test_arrays_map_with_symbols() -> None:
    """Validate arrays_map with symbolic shapes and value mapping."""
    m = buffer_expr("m")
    arr = ar_spec((m,), name="sym")
    root = db_node(arr, name="root")
    build_bmap(root, align=BufferAlign.BYTE)

    arrays_map(root, sym_dic={m: 4}, balign=BufferAlign.BYTE)
    assert arr.array is not None
    assert arr.array.shape == (4,)


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
    assert "return" in alloc_src


def test_container_free_symbols() -> None:
    """Cover free_symbols aggregation for aligned containers."""
    m = buffer_expr("m")
    node = ar_spec((m,), name="a")
    root = db_node(node, name="root")
    build_bmap(root, align=BufferAlign.BYTE)
    assert any(s.name == "m" for s in root.free_symbols)


def test_arraynode_gen_call_and_repr() -> None:
    """Exercise ArrayNode repr and gen_call output formatting."""
    node = ar_spec((2, 3), np.float32, name="arr", align_ldim=16)
    istr, rstr = node.gen_call()
    assert "arr" in rstr
    assert "Array" in repr(node)


def test_value_node_repr() -> None:
    """Cover ValueNode string representation."""
    val = v_spec(5, name="v")
    assert "Value" in repr(val)


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
    assert "def shared" in alloc_src


def test_value_node_gen_call() -> None:
    """Exercise ValueNode string generation and symbol tracking."""
    val = v_spec(sym.Symbol("m"), name="val")
    istr, rstr = val.gen_call()
    assert isinstance(istr, str)
    assert rstr == "val"
    assert any(s.name == "m" for s in val.free_symbols)


@pytest.mark.regression
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
    alloc_src = build_buffer_allocator(root, args=(buffer_expr("n"), "type_flt"))
    assert "def root_node" in alloc_src
    assert "aligned_buffer" in alloc_src


def test_save_bmap_tree_path(tmp_path: pathlib.Path) -> None:
    """Write a DOT-rendered image when Graphviz is available."""
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' not available")

    root = db_node(ar_spec((2,), name="a"), name="root")
    build_bmap(root, align=BufferAlign.BYTE)
    out_path = tmp_path / "tree.png"
    try:
        result = save_bmap_tree(root, path=str(out_path))
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"Graphviz failed to render: {exc}")
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
