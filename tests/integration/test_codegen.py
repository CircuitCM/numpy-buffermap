import numpy as np
import sympy as sym

import bmap as npb
from bmap import buffer_symbols


def test_codegen_large_symbolic_mem_map() -> None:
    """Exercise allocator codegen for a large symbolic tree."""
    m, n, r, z, t1, t2, ail = buffer_symbols("a b u y type_ota type_naa ail")
    dist1 = npb.ar_spec((m,), t2, name="h1")
    dist2 = npb.ar_spec((m, n, 5), np.float32, name="h2")
    dist3 = npb.ar_spec((n, z), int, name="h3")

    s1 = npb.ar_spec((z, m, 3), t1, name="sysr")
    s2 = npb.ar_spec((r, z, 50), t1, name="uxa", align_ldim=ail)
    s3 = npb.ar_spec((m, r), t2, name="ail20", order="F", align_ldim=npb.BufferAlign.AVX)

    o1 = npb.ar_spec((z, m, "4*y"), t1, name="f95")
    o2 = npb.ar_spec((z, m, r), t1, name="fi9")
    o3 = npb.ar_spec((z, m, "5*u*y"), t1, name="re", order="F")

    mem_map = npb.db_node(dist1, dist2, dist3, name="Algo Mem")
    shared_mem = npb.sb_node(s1, s2, s3, name="Shared Mem")
    shared_mem2 = npb.sb_node(o1, o2, o3, name="Shared Mem 2")
    mem_map.add(shared_mem, shared_mem2)

    rgs = npb.build_bmap(mem_map, align=ail)
    src = npb.build_buffer_allocator(mem_map, rgs, chkforbuffer=True)
    assert "def algo_mem" in src
    assert "aligned_buffer" in src


def test_codegen_lars1_spec() -> None:
    """Cover allocator generation for the LARS1 spec example."""
    m, n, dt, il = buffer_symbols("sample_size sample_dims type_flt alignb")
    at = npb.ar_spec((m, m), dt, name="At")
    t1 = npb.ar_spec((sym.Max(m * m, n),), dt, name="T1")
    t2 = npb.ar_spec((m,), dt, name="T2")
    t3 = npb.ar_spec((m,), dt, name="T3")
    c = npb.ar_spec((n,), dt, name="C")
    i = npb.ar_spec((m,), np.int64, name="I")
    ib = npb.ar_spec((n,), np.bool_, name="Ib")
    spec = npb.db_node(at, t1, t2, t3, c, i, ib, name="lars1_memspec", no_merge=True)

    rgs = npb.build_bmap(spec, align=il)
    src = npb.build_buffer_allocator(spec, rgs, {il: npb.BufferAlign.AVX512}, chkforbuffer=True)
    assert "def lars1_memspec" in src
    assert "aligned_buffer" in src
