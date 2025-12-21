
'''The goal is to **minimize repeated index arithmetic** while preserving exact integer semantics and alignment behavior.

---

## Constraints to obey
- Keep everything in **integer land** (no floats). All original variables are already >= 1.
- Do **not** introduce new lambdas; only use the existing `cd` (ceil-div) helper when needed.
- New declared variables must be short: **≤3 characters**, lowercase letters/digits (e.g. `mtz`, `c1`, `v2`).
- Do not alter view semantics: same slices, same shapes, same alignment and padding policy.
- Prefer refactors that reduce:
  - total repeated operations,
  - bloatiness (nested `cd`, `max`, padding products),
  - and duplicated expressions across both allocation and slicing.

---

## Step 0 — Inventory and annotate
1. Identify and list every arithmetic-heavy expression appearing in:
   - the buffer allocator (especially inside `aligned_buffer(...)`),
   - each view size (`buffer[:...]`),
   - each reshape argument,
   - each pointer bump (`buffer = buffer[...]`),
   - `max(...)` blocks,
   - padding/tiling expressions like `cd(x,k)*k`.

2. Normalize the expressions syntactically so duplicates are obvious:
   - pull out constants (`3*`, `64*`, etc.),
   - rewrite powers (`m**2` → `m*m`) mentally for matching,
   - rewrite `cd(a,b)*b` as “pad-to-multiple”.

This step is for yourself to reason, I don't need to see it.

---

## Step 1 — Greedy factoring loop (strategic “turn-based”)
This is a recursive/iterative loop: **introduce one reused parameter at a time**, then reevaluate the whole code with that new parameter included.

### 1A. Candidate scoring heuristic
For each candidate sub-expression `E`, estimate a score based on:
- **Occurrences**: how many times `E` (or a close variant) appears across allocator + views + bumps.
- **Captured operations**: how many multiplications/additions/`cd` calls `E` eliminates at each occurrence.
- **Bloatiness reduction**: prioritize candidates that shorten complex constructs:
  - `cd(... , ...)`, `cd(... , ...)*...`,
  - repeated `max(...)` terms,
  - padded-tile formulas.
- **Downstream unlock**: candidates that enable further factorization (e.g. making other expressions become simple multiples like `r*E` or `3*m*E`).

Pick the **single best** candidate `E*` (highest net benefit) and declare it near the top (above buffer init).

### 1B. After introducing a variable, immediately rewrite everything
After adding `x = E*`:
- Replace all matching occurrences with `x`.
- If this makes other expressions become *simple multiples*, rewrite them too:
  - e.g. if `rmz = r*mtz`, then rewrite `m*r*t1*z` as `rmz`, not as `r*mtz` in some places and expanded elsewhere.
- Reduce expressions in both:
  - allocator block count terms,
  - view slice lengths and reshapes,
  - pointer bump sizes.

### 1C. Repeat until no obvious win remains
Repeat 1A–1B until:
- there are no repeated “big” expressions left,
- remaining repetition is trivial (single multiply) or would increase clutter,
- further factoring would require long variable names or obscure intent.

**Important:** At each iteration, consider that previous declarations are now part of the search space. A good next move is often to create a new variable that causes prior ones to collapse into small multiples (e.g. `a = r*b`).

---

## Step 1 (Special patterns to always target)
These patterns consistently yield the largest wins in generated buffer-layout code.

### P1. “Core products” (high-leverage multiplies)
Search for repeated products of loop dimensions and dtype sizes.
Procedure:
1. factor smallest shared cores first (e.g. `mt1 = m*t1`, `mn = m*n`, `nz = n*z`),
2. then compose higher-level cores (e.g. `mtz = mt1*z`, `rmz = r*mtz`),
3. then rewrite larger forms in terms of these (e.g. `3*m**2*r*t1*z` → `3*m*rmz`).

### P2. “Pad-to-multiple” tiles
Any repeated `cd(X, K)*K` should almost always be factored:
- declare `K` if it recurs (e.g. `v2 = 64//t2`),
- declare padded lengths (e.g. `pm = cd(m, v2)*v2`),
- then reuse in slices/reshapes/max terms.

### P3. “Max workspace” blocks
When there is `max(A, B, C)` used for overlapping scratch:
- factor each arm (`A`, `B`, `C`) as much as possible,
- then factor the max itself: `mx = max(A, B, C)`,
- ensure every use of that max in allocator and pointer bump uses `mx` consistently.

### P4. “Ceil-block counts” used in multiple places
If you see the same `cd(..., ...)` feeding both:
- `aligned_buffer(64*(... + cd(...)) ...)`
- and later `buffer = buffer[64*cd(...):]`,
then:
- declare the count once (e.g. `c1 = 64*cd(..., ...)`),
- reuse it everywhere.

---

## Step 2 — Algebraic finishing touches (safe identities)
After the greedy loop stabilizes, do a final pass for safe simplifications that:
- reduce nested `cd`,
- collapse constant multipliers,
- remove redundant padding computations.
- remove redundant variable declarations and minimize their total, without altering the arithmetic reductions of step 1.

### 2A. Power-of-two dtype/alignment identities
Dtype sizes are constrained to power-of-two, so are alignments that may be a constant e.g. 64 or a variable. Alignments will always be greater than or equal to dtype size.
- Look for expressions of the form:
  - `cd(64, (64//t1))*(64//t1)`
- Attempt to simplify exactly:
  - If `t1` divides 64 and is power-of-two, then this expression equals `64`.
- Replace both slice sizes and reshape dims accordingly (e.g. reshape to `(r, z, 64)` directly).
**Only apply if guaranteed by your domain assumptions.** If not guaranteed, keep the original expression.

### 2B. Re-express high-order polynomials via existing cores
Rewrite anything like:
- `m**2 * r * t1 * z` as `m * (m*r*t1*z)` or `m*rmz`,
- `3*m**2*r*t1*z` as `3*m*rmz`,
to reduce both bloat and risk of mismatch.

### 2C. Normalize repeated constants
If many expressions contain the same scalar factors (3, 4, 5, 20, 64):
- fold them into a single term where it improves readability and reuse,
- but don’t over-factor constants if it creates noise.

---

## Step 3 — Structural sanity checks (must pass)
Before finalizing:
1. **Allocation coverage:** Confirm the total bytes carved match what’s allocated, considering alignment padding blocks.
2. **Pointer bump correctness:** Every `buffer = buffer[...]` should bump by the correct aligned block (typically `64 * ceil_count`), and the count should match the earlier allocator term.
3. **View equivalence:** For each view:
   - slice length equals `dtype_size * num_elements`,
   - reshape product equals number of elements,
   - any cropping (`[:m,...]`, `[..., :64]`) matches original intent.
4. **Overlapping regions:** If scratch/output arrays intentionally overlap, preserve the same base pointer and only advance the buffer after the “max” region (or after the final overlapping view).

---

## Output requirements
- Provide the refactored function with:
  - declarations clustered above buffer init,
  - consistent naming and reuse,
  - no semantic changes.
  - *zero* declarations that are referenced only once in the search space.

---

## Summary of the “game plan”
1. Factor core products → 2. Factor padded tiles → 3. Factor max workspace → 4. Factor ceil counts → 5. Apply safe algebraic identities → 6. Verify structure.
Repeat factoring recursively until no meaningful duplication remains.
'''

"""
- Prefer clear short variable names that encode meaning via pattern:
  - `mn`, `nz`, `mt1`, `mtz`, `rmz`, `pm`, `mx`, `c1..cN`.
"""

#The result after gpt 5.2 and this prompt.
import numpy as np
aligned_buffer=lambda x,y: x
fb_=np.frombuffer


def algo_mem(a, b, ail, y, u, type_ota, type_naa, buffer=None):
    cd_ = lambda x, dv: (x + dv - 1)//dv
    ota = np.dtype(type_ota).itemsize
    naa = np.dtype(type_naa).itemsize
    
    if buffer is None:
        buffer = aligned_buffer(ail*(cd_(max(3*a*ota*y, naa*u*cd_(a,(32//naa))*(32//naa), ota*u*y*cd_(50,(ail//ota))*(ail//ota)),ail) + cd_(20*a*b,ail) + cd_(a*ota,ail) + cd_(4*b*y,ail) + cd_(5*a*ota*u*y**2,ail)), 4096)
    
    h1 = fb_(buffer[:a*ota],type_ota).reshape((a,))
    buffer = buffer[ail*cd_(a*ota,ail):]
    h2 = fb_(buffer[:20*a*b],np.float32).reshape((a, b, 5))
    buffer = buffer[ail*cd_(20*a*b,ail):]
    h3 = fb_(buffer[:4*b*y],np.int32).reshape((b, y))
    buffer = buffer[ail*cd_(4*b*y,ail):]
    
    sysr = fb_(buffer[:3*a*ota*y],type_ota).reshape((y, a, 3))
    uxa = fb_(buffer[:ota*u*y*cd_(50,(ail//ota))*(ail//ota)],type_ota).reshape((u, y, cd_(50,(ail//ota))*(ail//ota)))[...,:50]
    ail20 = fb_(buffer[:naa*u*cd_(a,(32//naa))*(32//naa)],type_naa).reshape((u, cd_(a,(32//naa))*(32//naa))).T[:a,...]
    buffer = buffer[ail*cd_(max(3*a*ota*y, naa*u*cd_(a,(32//naa))*(32//naa), ota*u*y*cd_(50,(ail//ota))*(ail//ota)),ail):]
    
    f95 = fb_(buffer[:4*a*ota*y**2],type_ota).reshape((y, a, 4*y))
    fi9 = fb_(buffer[:a*ota*u*y],type_ota).reshape((y, a, u))
    re = fb_(buffer[:5*a*ota*u*y**2],type_ota).reshape((5*u*y, a, y)).T
    
    return h1, h2, h3, sysr, uxa, ail20, f95, fi9, re

def algo_mem_refac(a, b, ail, y, u, type_ota, type_naa, buffer=None):
    #significant improvement, however the prompt still seems pretty inefficient/reqs a lot of thinking
    #can be improved likely by specifying a much more clear cut thinking strategy.
    cd_ = lambda x, dv: (x + dv - 1)//dv
    ota = np.dtype(type_ota).itemsize
    naa = np.dtype(type_naa).itemsize

    ao = a * ota
    ab = a * b
    by = b * y
    uy = u * y
    y2 = y * y #it got this wrong, not needed

    p50 = cd_(50, (ail // ota)) * (ail // ota) #it should have assigned variables for these.
    pa = cd_(a, (32 // naa)) * (32 // naa)

    sy = 3 * ao * y
    ux = ota * uy * p50
    an = naa * u * pa

    b1 = ail * cd_(ao, ail)
    b2 = ail * cd_(20 * ab, ail) #could have assigned these as variables as well.
    b3 = ail * cd_(4 * by, ail)

    bmx = ail * cd_(max(sy, an, ux), ail)

    r5 = 5 * ao * u * y2
    br = ail * cd_(r5, ail)

    if buffer is None:
        buffer = aligned_buffer(b1 + b2 + b3 + bmx + br, 4096)

    h1 = fb_(buffer[:ao], type_ota).reshape((a,))
    buffer = buffer[b1:]
    h2 = fb_(buffer[:20 * ab], np.float32).reshape((a, b, 5))
    buffer = buffer[b2:]
    h3 = fb_(buffer[:4 * by], np.int32).reshape((b, y))
    buffer = buffer[b3:]

    sysr = fb_(buffer[:sy], type_ota).reshape((y, a, 3))
    uxa = fb_(buffer[:ux], type_ota).reshape((u, y, p50))[..., :50]
    ail20 = fb_(buffer[:an], type_naa).reshape((u, pa)).T[:a, ...]
    buffer = buffer[bmx:]

    f95 = fb_(buffer[:4 * ao * y2], type_ota).reshape((y, a, 4 * y))
    fi9 = fb_(buffer[:ao * uy], type_ota).reshape((y, a, u))
    re = fb_(buffer[:r5], type_ota).reshape((5 * uy, a, y)).T

    return h1, h2, h3, sysr, uxa, ail20, f95, fi9, re
