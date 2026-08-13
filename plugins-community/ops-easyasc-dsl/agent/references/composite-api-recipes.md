# Composite API Recipes

A lookup "dictionary" of how composite / high-level feature blocks (套路 feature
blocks) decompose into smaller ISA primitives, mined from CANN's own implementation
headers and verified here. When you must build a composite block that `a2.py` /
`a5.py` does **not** expose, check here first: if the recipe is recorded, reuse the
decomposition idea instead of re-deriving it.

This is the **reverse** direction of `agent/references/decomposition-primitives.md`:

| | `decomposition-primitives.md` | this file |
|---|---|---|
| Direction | forward: PyTorch function → legal split primitives | reverse: a finished composite effect → the small instructions that build it |
| Question | is this split numerically legal? (tiers P/L/A/F) | how is this effect assembled from primitives? |
| Source | our own split policy | reverse-engineered from CANN impl source + our verification |

Index entries by **operation shape** (the abstract pattern), not by API name — future
needs arrive as "I must sample features at gathered offsets with weights", not as a
vendor API name. The concrete API is one instance of the pattern.

## When to enter / when to add

- **Enter** when implementing a composite block not exposed by the DSL facade and you
  want a known primitive decomposition to copy.
- **If not recorded**, mine it (method below), then **add an entry**. The dictionary is
  only worth maintaining if it grows; an unmaintained entry rots, so always record the
  source path and verification status so a stale recipe is detectable.

## How to mine a recipe

1. **Locate the impl header.** CANN ships per-arch implementations at
   `<CANN>/aarch64-linux/asc/impl/basic_api/dav_<arch>/kernel_operator_vec_<name>_impl.h`
   (`<CANN>` e.g. `/usr/local/Ascend/cann-9.0.0` on the A2 box). Arch tags:
   `dav_c220` = A2 (`__DAV_C220_*`), `dav_c310` = A5 / 950 (`__DAV_C310__`);
   `dav_c100` / `dav_m200` / `dav_m300` / `dav_m310` are other targets. The public
   signature lives in `<CANN>/aarch64-linux/asc/include/basic_api/..._intf.h`.
2. **Read which primitives it composes.** The impl `#include`s the
   `kernel_operator_vec_*_intf.h` it calls (binary `Add` / `Mul`, `Brcb`, `Gatherb`,
   `Adds`, …) on A2, or `Reg::*` register ops inside a `__simd_vf__` body on A5. Those
   calls *are* the underlying interface — not the `(...)` category tag in the doc title.
3. **Capture the A2-vs-A5 divergence.** The same effect usually uses *different*
   primitive sets per arch (A2 often stages through a scratch UB buffer; A5 often keeps
   the accumulator in registers via a VF loop). This is exactly what bites a port.
4. **Verify numerically.** Trace the doc's worked example through the impl (stride /
   mask / repeat semantics) and check both endpoints. A simulator pass does not replace
   the A2 device gate; use `agent/references/constraints/a2.md` and
   `agent/references/facts-simulator-opexec.md` for the known boundary, and record
   *how* the recipe was checked.
5. **Check DSL exposure.** Grep `a2.py` / `a5.py` for the primitives the recipe needs; if
   all are exposed, the block is replicable in-DSL.
6. **Record** with source path + verification status + confidence.

## Entry template

```
### <operation shape> — <one-line math form>
- Composite API: <vendor API name + dtype constraints>
- Decomposition:
  - A2 (c220): <primitive sequence>   [scratch: <size or none>]
  - A5 (c310): <primitive sequence>   [scratch: <size or none>]
- Key trick: <the insight that transfers>
- DSL exposure: a2.py <primitives> / a5.py <primitives> -> replicable? <yes/no/partial>
- Source: <CANN impl header path(s)>; real usage: <kernel/op that uses it>
- Verified: <source-read | sim | real-HW | by-hand-numeric> + <confidence>
```

## Recipes

### tanh GELU activation — `x * sigmoid(1.59576912 * (x + 0.044715*x^3))`

- **Composite API:** AscendC `Gelu<T, highPrecision, highPerformance>` (CANN
  9.0.0 adv activation API). The public header formula is
  `x / (1 + exp(-1.59576912 * (x + 0.044715*x^3)))`, i.e. the tanh/sigmoid
  approximation, **not** exact `erf`/`erfc` GELU. The API check/static asserts
  support `half` and `float`; it does not support `bfloat16`. `FasterGelu` and
  `FasterGeluV2` are separate APIs with different formulas.
- **Decomposition:**
  - **A2 (c220)** — vector arithmetic over chunks from a caller or stack
    temporary buffer. First form `y = 1.5957691216057308 * (x + 0.044715*x^3)`:
    `Mul(x,x)` -> `Mul(x,x^2)` -> `Muls(0.044715)` -> `Add(x)` ->
    `Muls(1.5957691216057308)`. Then compute the overflow-safe sigmoid identity:
    `exp(min(y, 0)) * (x / (exp(-abs(y)) + 1))`. With
    `highPerformance=true`, only this denominator division changes from `Div` to
    `Reciprocal` + `Mul`; the final multiply by `exp(min(y,0))` remains.
    With `highPrecision=true` on `half`, the chunk is cast to `float`, computed
    as float, then cast back to half. **Scratch:** `sharedTmpBuffer` is
    reinterpreted as `T` (or `float` for half high-precision) and split into two
    aligned stack buffers; half high-precision also reserves one conversion
    buffer. The overload without explicit scratch calls
    `PopStackBuffer<uint8_t, TPosition::LCM>` and then runs the same path.
  - **A5 / C310** — register-VF implementation. Each vector repeat loads into
    registers, optionally unpacks/casts half to float for `highPrecision`, runs
    the same formula using `Reg::Mul/Muls/Add/Mins/Exp/Abs/Adds/Div/Mul`, then
    stores (packing back to half when needed). The `sharedTmpBuffer` overload is
    only checked and ignored. `highPerformance` is explicitly ignored for C310
    `GeluImpl`; the final denominator operation stays `Reg::Div`.
- **Key trick:** the API name `Gelu` is easy to misread in benchmark contexts:
  this composite API is only the tanh approximation. Its numerical stability
  comes from rewriting `sigmoid(y)` as
  `exp(min(y,0)) / (exp(-abs(y)) + 1)` rather than evaluating `exp(-y)`
  directly. AscendC `Erfc` is a separate math API with a different rational
  approximation; CANN `Gelu` does not call it.
- **DSL exposure:** a2.py exposes the A2 primitives needed for the formula
  (`mul`, `muls`, `add`, `vmins`/scalar min, `abs`, `exp`, `adds`, `div`,
  `rec`, `cast`), so the A2 recipe is replicable in DSL. Current benchmark work
  in `agent/example/kernels/a2/vec_only/gelu_general.py` intentionally differs: tanh mode uses
  the algebraically equivalent direct denominator `x / (1 + exp(-y))` plus a
  negative-infinity guard, exact mode uses a hand-written erfc-style polynomial,
  and fp16/bf16 paths always compute formulas in fp32 UB before cast-back.
- **Source:** `<CANN>/aarch64-linux/asc/include/adv_api/activation/gelu.h`;
  A2 impl `<CANN>/aarch64-linux/asc/impl/adv_api/detail/activation/gelu/gelu_impl.h`;
  C310 impl `<CANN>/aarch64-linux/asc/impl/adv_api/detail/activation/gelu/gelu_c310_impl.h`;
  API check `<CANN>/aarch64-linux/asc/impl/adv_api/detail/api_check/kernel_check/activation/gelu/gelu_check_common.h`.
- **Verified:** source-read on the A2 910B3 CANN 9.0.0 box on 2026-07-02,
  including A2, C310, and GELU API checks. Earlier generated-C++ real-HW
  experiments on A2 (2026-07-01) showed the default AscendC `Gelu` path is only
  slightly faster geomean on supported fp16/fp32 tanh rows and less accurate
  for fp16; A2 `highPerformance=true` is faster for fp16 but fails fp32
  accuracy due to reciprocal precision; A2 half high-precision matches the
  fp32-compute error profile but is much slower. Confidence: high for A2 source
  and good for C310 source-read.

### complementary error function — `erfc(x) ≈ exp(-xa^2) * R(z)/S(z) * xb + (1 - xb)`

- **Composite API:** AscendC `Erfc<T, isReuseSource>` (CANN 9.0.0 adv math
  API). Supports `half` and `float`; no `bfloat16` path was found in the mined
  headers.
- **Decomposition:**
  - **A2 (c220/common)** — clip `x` to `[-10, 10]`, compute
    `xa = abs(x) + 2^-62`, `xb = x / xa`, `z = min(xa, 10)`, evaluate degree-8
    numerator `R(z)` and degree-5 denominator `S(z)` with vector arithmetic,
    then form `exp(-xa^2) * R(z) / S(z) * xb + (1 - xb)`. Half is cast to
    float, computed in float scratch, then cast back. **Scratch:** common impl
    splits `sharedTmpBuffer` into 5 float-sized chunks for float and 12
    half-sized chunks for half.
  - **A5 / C310** — register-VF implementation of the same rational formula,
    keeping intermediates in `Reg::RegTensor<float>`; half is unpacked/cast to
    float and packed back on store. The `sharedTmpBuffer` overload checks the
    UB tensor but ignores it.
- **Key trick:** the vendor exact-erfc path is a clipped rational approximation
  in float, not the Abramowitz-Stegun-style polynomial currently used by the A2
  GELU benchmark helper.
- **DSL exposure:** a2.py exposes the arithmetic pieces (`abs`, `adds`,
  `vmins`, `vmaxs`, `mul`, `muls`, `div`, `exp`, `cast`) so the A2 recipe is
  replicable in DSL, but it is scratch-heavy. a5.py has the register primitives
  needed for a closer C310-style VF port.
- **Source:** `<CANN>/aarch64-linux/asc/include/adv_api/math/erfc.h`;
  common/A2 impl `<CANN>/aarch64-linux/asc/impl/adv_api/detail/math/erfc/erfc_common_impl.h`;
  C310 impl `<CANN>/aarch64-linux/asc/impl/adv_api/detail/math/erfc/erfc_c310_impl.h`.
- **Verified:** source-read on the A2 910B3 CANN 9.0.0 box on 2026-07-02.
  Not yet reimplemented and benchmarked in DSL. Confidence: good for primitive
  shape, medium for performance implications.

### weighted gather-accumulate — `dst[blk] = Σ_iter w(iter) · src0[offset(iter) + blk]`

- **Composite API:** AscendC `BilinearInterpolation` (`half` only). The `(ISASI)` tag in
  the doc title is an *API category* ("architecture-specific basic API", no cross-version
  compatibility guarantee — the same tag is on `Fixpipe`, `NdDma`, …), **not** an
  instruction name. The real primitives are below.
- **Decomposition:**
  - **A2 (c220)** — emulated through a scratch UB buffer:
    `Gatherb` (gather src0 DataBlocks via `src0Offset`) → `Brcb` (broadcast each `src1`
    weight to a full block) → `Mul` (gathered blocks × broadcast weights; `repeatMode=false`
    ⇒ `src0BlkStride=0` reuses one weight across the 8 blocks, `src0RepStride=1` advances
    the weight per horizontal rep) → `Add` (accumulate the `hRepeat` horizontal results per
    vertical row) → `Adds(…, 0)` (strided copy-out to `dst`).
    **Scratch:** `src0.GetSize()*32 + src1.GetSize()*32` bytes (the `sharedTmpBuffer`).
  - **A5 (c310)** — register VF loop (`__simd_vf__`), no scratch:
    `for i in vRepeat { Duplicate(dstReg, 0); for j in hRepeat { LoadAlign(idx) →
    Reg::GatherB(src0) → LoadAlign<DIST_BRC_B16 | DIST_E2B_B16>(src1) → Reg::Mul →
    Reg::Add (in-register MAC) }; StoreAlign<DATA_BLOCK_COPY>(dst + i*vROffset) }`.
- **Key trick:** the same effect uses different primitive sets per arch. A2 stages the
  gather + broadcast results in `sharedTmpBuffer` then does vector MAC; A5 accumulates in a
  register across `hRepeat`. That is *why* A2/A3 require the temp buffer and A5/Atlas-350 do
  not. `repeatMode` selects the weight broadcast: `false` = one scalar per horizontal rep
  (A5 `DIST_BRC_B16`), `true` = 8 distinct weights per rep (A5 `DIST_E2B_B16`). A DataBlock
  is 32 B = 16 `half`; `src0Offset` is in bytes; `vROffset` advances the `dst` row group.
- **DSL exposure:** a2.py exposes `gather_block` (AscendC `Gatherb`, added for exactly this
  decomposition — see `doc/api/a2_vec_control.md`) + `brcb` + `mul` / `add`; a5.py exposes
  `ub_to_reg_gather` / `gather` + `ub_to_reg_brcb` + `mul` / `add` + `dup`. -> **replicable
  in-DSL on both targets.** The a2 path now maps 1:1 onto the intrinsic's `gatherb` + `brcb`
  + MAC via `gather_block`; the in-DSL reimplementation is
  `agent/example/kernels/a2/composite_api/bilinear_interpolation.py` — simulator bit-exact (`dst[0]=389` /
  `dst[255]=4096`) **and verified on real A2** (same values, `max_abs_diff=0`) both via the
  official EasyASC `--run` path (`OpExec(simulator=False)`, end-to-end). No
  temporary launcher is required as evidence. HW gotcha (sim != HW): c220 `vgatherb` needs
  `uint16` operands for 16-bit data, so the kernel reinterprets `dst`/`src` to `uint16` around
  `gather_block` (the simulator accepts `half`; the on-box compiler rejects it).
- **Source:** `<CANN>/aarch64-linux/asc/impl/basic_api/dav_c220|dav_c310/kernel_operator_vec_bilinearinterpolation_impl.h`
  (intf: `.../include/basic_api/kernel_operator_vec_bilinearinterpolation_intf.h`).
  Real usage: `<CANN>/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/multi_scale_deformable_attn_function/ms_deform_attn_high_perf.h`
  (deformable-attention feature sampling).
- **Verified:** **real-HW on A2 (910B3 / dav_c220), bit-exact, both modes.** A standalone
  AscendC kernel-direct-launch test ran on the NPU (both the 110 and 168 boxes):
  `repeatMode=false` (the doc example `src0=[1..512]`, `src1=[2..17]`, `offset=[0,32,…,992]`,
  `mask=128, hRepeat=2, vRepeat=2`) → `dst[0]=389`, `dst[255]=4096` (matching the doc's
  `[389…4096]`); `repeatMode=true` (per-block weights `src1[rep*8+b]`) → `dst[0]=1162`,
  `dst[255]=25600`. Both `max_abs_diff=0` over all 256 outputs vs a half-faithful reference
  (mul-then-add in `half`). A5 (c310) stays **source-read only** (no A5 box). Confidence: high.
  (Harness + build recipe for the relocated read-only CANN: see `machine_specs.md`.)

### fixpipe cast+scale on the L0C→UB store, to both vec subblocks — `ub_sb = cast_to(dst_dtype)(scale · l0c[sb_rows] + offset)`, sb∈{0,1} (A5/C310)

- **Composite API:** the a5/C310 fixpipe scalar requant on an L0C→UB copy —
  `Tensor.requant(scale, offset, hif8_hybrid)` rider on `ub <<= l0c.requant(...)`, or the
  explicit `l0c_to_ub(..., scale=, offset=)`. Mode is inferred from the (src,dst) dtype
  pair: fp32→{fp16/bf16 scaled-cast `QF322F16_PRE`, int8/uint8 `QF322B8_PRE`},
  int32→{int8 `REQ8`, fp16 dequant `DEQF16`}. `relu()` composes (relu then quant).
- **Decomposition:**
  - **Cast/scale is SINGLE-mode ONLY.** The dual `SPLITM`/`SPLITN` mode (which auto-splits
    the L0C tile's rows across the two vec subblocks) carries ONLY a *same-type plain copy*
    (fp32→fp32 or int32→int32). It supports **neither** a requant scale/offset/relu **nor
    even a deqScalar-free float downcast** (fp32→fp16/bf16). Any cast-or-scale therefore
    forces `dual_mode=SINGLE`. (Gate: `easyasc/stub_functions/cube.py` l0c_to_ub, the
    `_dual_cast_ok` / `_requant_used` checks.)
  - **A SINGLE store lands in ONE subblock** (default `sub_block_id=0`). To fill BOTH vec
    subblocks, issue **two** stores, `sub_block_id=0` then `=1`, each sourcing the matching
    L0C row-slice (`l0c[0:64]` / `l0c[64:128]` for a 128-row tile). Keep the **full-tile M
    stride**: `M_src = L0C.shape[0]` (the declared tile height), NOT the sliced `span[0]` —
    `l0c[64:128]` still strides over the full 128-row fractal. (Auto-inferred if you pass the
    slice and leave M/N/M_src=None.) Bridge with one CvMutex around the pair of stores.
  - **`<<=` cannot target subblock 1.** The operator's L0C→UB path
    (`easyasc/utils/Tensor.py:319-333`) reads only the requant/relu riders
    (`_deq_scale/_deq_offset/_is_relu/_deq_hif8_hybrid`) and never `sub_block_id` (always 0).
    The **`.subblk(0/1)` rider** targets one subblock straight on the `<<=` path — it forces
    SINGLE (like requant/relu) and composes: `ub <<= l0c[half].requant(s).subblk(1)` is
    exactly `l0c_to_ub(ub, l0c[half], dual_mode=SINGLE, sub_block_id=1, scale=s)`. (requant +
    sub_block_id are NOT mutually exclusive; SINGLE+sub_block_id+scale compose, chained-rider
    order-independent.) Plain `ub <<= l0c.requant(s)` (no subblk) always lands subblock 0.
- **Key trick:** **fold a downstream scalar multiply into the fixpipe `scale`** — it is free
  on the FIX pipe and removes a vec op. e.g. softmax's `1/sqrt(D)` becomes
  `requant(scale=1/sqrt(D))`, deleting the per-row `* SCALE` from the softmax VF
  (math-equivalent). The fp16/bf16 dst simultaneously **halves the UB score buffer** (e.g.
  a [64,256] score: 64KB fp32 → 32KB fp16) — UB headroom for PV/accum. Caveat: fp16 score
  loses precision vs fp32 (range of QK·scale ~O(1–10) fits fp16's 5-bit exp, 10-bit mantissa
  is ample) — validate numerically once PV/accum is added.
- **DSL exposure:** a5.py exposes `l0c_to_ub` + `DualMode` + `Tensor.requant()` +
  `Tensor.subblk()` (the subblock-target rider). → replicable.
  Templates: `agent/example/kernels/a5/matmul/matmul_quant_ub.py` (every dtype mode; single-subblock
  `ub <<= l0c.requant(...)` AND the split-and-move two-half row-slice pattern
  `matmul_ub_split_quant_byte`), and `agent/example/kernels/a5/simt/simt_matmul_transpose_contig_write.py`
  (canonical two-subblock publish, `sub_block_id=0` then `1`).
- **Source:** `easyasc/stub_functions/cube.py` `l0c_to_ub` (SINGLE/dual + dtype gate, sig
  default `dual_mode=SPLITM, sub_block_id=0`); `easyasc/utils/Tensor.py:319-333` (`<<=` rider
  path), `:533` (`requant()`), `:92` (`_l0c_store_has_requant_or_relu`). First real usage
  was an fp16-score + SCALE-fold attention kernel (since pruned).
- **Verified:** **sim** — `matmul_quant_ub.py` all 7 modes bit-exact (unchanged after the
  `subblk()` addition); the `subblk()` rider numerically verified (two-subblock fp16
  scaled-cast, both halves `max_abs_diff=0`, chained `requant().subblk()` ⇄ `subblk().requant()`
  order-independent); the fp16-score + SCALE-fold attention path's codegen + UB-halving
  (32KB fp16 score) sim-confirmed. Real-HW board timing (the −16.5us SCALE-fold lever, fp16
  numerics) **pending
  board recovery**. Confidence: high on mechanism; perf delta + fp16 accuracy unverified on HW.
