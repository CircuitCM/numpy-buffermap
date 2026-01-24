# Stage 4 Step 1 Refactor Plan

## Context / inputs
- Use the latest `coverage.txt` from the most recent `rpytest bmap` run as the source of truth for uncovered regions. - TEST
- Use the most recent `tests/report/radon/*/flagged.md` as the source of truth for which functions are over CC threshold / high Bug Est. - REFAC
- Focus review and edits on flagged functions in `bmap/_nodes.py`, `bmap/_procedures.py`, `bmap/_util.py`. - REFAC

## Coverage gaps and dead code notes (from coverage + pcov)
### Unreachable by design / optional deps
- Treat optional dependency branches as optional and keep them; don’t spend time trying to reach them in tests. - REFAC
- Treat visualization ImportError branches as optional and keep them; don’t delete as dead code. - REFAC

### Special exceptions / platform behavior
- If Graphviz temp-file locking is the only reason `save_bmap_tree` is unreliable on Windows, try resolving the test by using a temp path inside this repository (e.g., under `temp/`). In temp also write the outputs of the tests so I can see results. If this doesn’t reliably fix it, keep the test guarded/skipped and move on. - TEST

### Bug-related / should be addressed
- Fix the distinct-root + `fullreduce=True` allocator bug (IndexError class) in `build_buffer_allocator` / `_bb2` with minimum-viable edits and without changing the overall style of the code block. - REFAC

### Other uncovered regions
- `ArrayNode.mk_array`: cover the “view from buffer” path by passing an explicit flat byte buffer and making sure the node has an offset; this should be reachable via public API and should be tested. - TEST
- `ArrayNode.free_symbols`: keep current behavior and don’t treat the “dtype as expression” branch as dead code; users may mutate dtype after node creation. - REFAC
- `array_arspec`: it is expected to be covered; remove any remaining `cast()` usage by switching to `assert`/debug branches without breaking typing. - REFAC
- `_arrbxpr`: convert the “if then raise” type-guard pattern into asserts. - REFAC
- Mark `_bmap_pyvis` as `pragma: no cover`. - REFAC

## Refactor plan (ordered, per Stage 4 Step 1)
### 1) Bug fixes with minimum-viable edits
- `build_buffer_allocator` fix distinct-root + `fullreduce=True` allocator bug by adjusting boundary handling around “final pop” and `ct` indexing; preserve layout and semantics and keep changes narrowly scoped. - REFAC
- Validate `ValueNode` scalar handling in `_bb2` once (reuse locals, avoid repeated checks) and keep the same error semantics. - REFAC

### 2) Reduce CC to <= 10 with cosmetic/logically equivalent edits
**bmap/_nodes.py**
- `ArrayNode.__init__`: introduce `_normalize_dtype` only; keep bshape calculation inside `__init__`. - REFAC
- `ArrayNode.mk_array`: keep it smart and equivalent (avoid extra calcs when array already exists); optionally use a small `_mkbsh_ty` helper to centralize bshape/dtype calcs at the two call sites. - REFAC
- `ArrayNode.gen_call`: reduce CC by ~1 via cosmetic reshaping only; avoid helper extraction. - REFAC
- `ArrayNode.free_symbols`: add `_add_sym`. - REFAC
- `ContainerNode.pop`: move only the for-loop body into a private module function; avoid larger restructuring. - REFAC

**bmap/_procedures.py**
- `build_buffer_allocator`: split into `_full_reduce` and `_no_reduce`. - REFAC
- `_bb2`: create `_cbb2` for ContainerNode handling; build eqns as a tuple from child `nbytes` first and use one enumerate/zip loop body (not duplicated between conditionals). - REFAC
- `build_buffer_allocator` fix distinct-root + `fullreduce=True` allocator bug by adjusting boundary handling around “final pop” and `ct` indexing; preserve layout and semantics and keep changes narrowly scoped. - REFAC
- `_bba`: do not add another helper; build a defs tuple from child `nbytes` and use a single zip loop. - REFAC
- `_compute_nbytes`: do not refactor further right now; re-check CC after other edits land. - REFAC
- `array_arspec`: add a `_aarspec` helper directly below; move the None/UO checking currently done inline at the `ar_spec` call into `_aarspec`; replace casts with asserts per the existing TODO. - REFAC
- `_dot_node_attr`: add a small `_dtype_label` callable and use it to reduce conditional clutter. - REFAC
- `_bmap_pyvis` and `bmap_pyvis`: add two private helpers: `_clean_nodes` (captures the entire node loop) and `_clean_field` (strip-if-exists for a given key). - REFAC

**bmap/_util.py**
- `post_process_cse`: build `rhs_map` and `deps` in a single for loop; remove any list building for the “not ready” error path; add line comments before every line inside the `while pending` block explaining why the layering works. Move the whole `oneassign_del` block into `_oneassign_del`, and extract the two deep inner loops into `_reduct_chk` to drop CC. - REFAC
- `buffer_expr`: only extract `_buffer_expr_from_str`. - REFAC

### 3) Dead code elimination strategy
- Do not delete optional-dependency branches; keep and document as optional. - REFAC
- Add short comments (near branches) explaining why specific branches remain uncovered or are guarded/skipped in tests. - REFAC

### 4) Algorithmic improvements
- No algorithmic changes beyond what is required to fix the allocator bug; if fixing distinct-root + fullreduce requires re-evaluating `_bb2` traversal order, keep it minimal and report exactly what changed afterward. - REFAC

### 5) MI improvements (comments)
- For any touched function that remains flagged, add short line comments adjacent to branch points per repo style; for the complex ones, add more frequent line comments. - REFAC
