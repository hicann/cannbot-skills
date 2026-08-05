---
applies_to: soc=all
reason: SIMD/Pipe/TQue patterns are the universal AscendC programming model — applies everywhere. Some platform-bug entries (e.g. F-P4 PipeBarrier alignment) cite A5 observations but are validated as same-behavior on V220 — flag any per-pattern A5-only nuances inline.
---

# Domain: Platform Compatibility & SIMD Specifics
> Patterns for AscendC platform-specific issues: PipeBarrier alignment, cache hints, TQue bugs.
> Load when: Analyzer detects SIMD DataCopy, PipeBarrier, __ldg usage, or TQue configuration.

---

## Patterns

### F-P4: SIMD PipeBarrier and DataCopy alignment

**Severity**: High | **Mode**: SIMD

**Anti-pattern**: `PipeBarrier<PIPE_MTE2>()` — fine-grained barrier can cause data races.

**Correct pattern**: `PipeBarrier<PIPE_ALL>()` guarantees correctness. Optimal: use TQue depth=2 double buffering.

**DataCopy alignment requirement**: fp32: %8==0, fp16/bf16: %16==0. When unaligned, fall back to SIMT or use `DataCopyPad`.

---

### P-P18: `__ldg`/`__stg` L2 Cache Hint (updated 2026-04-02)

**Severity**: High | **Source**: A5 measurement + HKV expert code + DavidV100 manual | **Platform**: Ascend950PR

**API**: AscendC provides templated `__ldg`/`__stg` to control L2 cache and L1/dcache behaviour:
```cpp
#include <kernel_operator.h>  // LD_L2CacheType, ST_L2CacheType, L1CacheType

// Read: controls L2 allocation policy + L1/dcache caching
T val = __ldg<LD_L2CacheType::hint, L1CacheType::hint>(ptr);

// Write: controls L2 write-back policy + L1 caching
__stg<ST_L2CacheType::hint, L1CacheType::hint>(ptr, val);
```

**Available hint values** (source: HKV expert code + manual HA.FS007):

| Read (LD_L2CacheType) | Meaning |
|---------------------|------|
| `L2_CACHE_HINT_NORMAL` | Normal caching (default; equivalent to no-arg `__ldg`) |
| `L2_CACHE_HINT_NOTALLOC_CLEAN` | Do not occupy an L2 slot after read; prevents large-range scans from polluting the cache |

| Write (ST_L2CacheType) | Meaning |
|---------------------|------|
| `L2_CACHE_HINT_NORMAL_FV` | Normal write-back to L2 |

| L1/dcache (L1CacheType) | Meaning |
|-------------------------|------|
| `CACHEABLE` | Cache through L1/dcache |
| `NON_CACHEABLE` | Bypass L1/dcache |

**Hint selection by access pattern**:

```cpp
// 1. Data read repeatedly by multiple cores/tokens (expert rows, embedding table)
//    -> Keep in L2 + dcache: maximize hit rate
val = __ldg<L2_CACHE_HINT_NORMAL, L1CacheType::CACHEABLE>(expert_ptr);

// 2. Sequential scan, read once (edge index, weight array)
//    -> L2 no-alloc: prevent cache pollution, leaving room for hot data
val = __ldg<L2_CACHE_HINT_NOTALLOC_CLEAN, L1CacheType::CACHEABLE>(index_ptr);

// 3. Output write (one-shot write, no subsequent read)
//    -> L1 not cached: do not waste dcache space
__stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(out_ptr, val);

// 4. HKV in-bucket random lookup (small chunk scanned repeatedly)
//    -> L2 no-alloc + L1 cached: in-bucket data takes the dcache fast path
val = __ldg<L2_CACHE_HINT_NOTALLOC_CLEAN, L1CacheType::CACHEABLE>(bucket_ptr);
```

**History**: Earlier tests of no-template-arg `__ldg` (OL-18, 2026-03-26) showed no effect — because the default is `L2_CACHE_HINT_NORMAL` + default L1 policy, indistinguishable from a plain read on wide sequential scan. The hinted version can differentiate hot data (keep in L2) from cold data (do not allocate L2) — that is the correct usage.

**Experiment result (Batch 14-5)**: On SIMT persistent SG forward, tested `NOTALLOC_CLEAN` (index/weight) + `NORMAL_PERS` (expert). **No positive effect** — dim=64 24% slower (instruction overhead); others unchanged. dcache was already caching effectively. The value of L2 hints lies in cross-core sharing scenarios (e.g., HKV), not in SIMT persistent sequential traversal.

**A5 measured data** (56 blocks x 32 threads, stride-scan, aclrtEvent timing):

| Data size | Plain-read BW | `__ldg` BW | Difference |
|---------|----------|-----------|------|
| 4 MB | 49.5 GB/s | 49.7 GB/s | +0.3% |
| 16 MB | 54.3 GB/s | 54.4 GB/s | +0.2% |
| 64 MB | 43.5 GB/s | 43.6 GB/s | +0.2% |
| 256 MB | 22.8 GB/s | 22.8 GB/s | -0.1% |

**Decision rule**:
- Dataset >> L2 cache -> do not use `__ldg` (pooling, SG, large-scale reduction)
- Dataset <= L2 cache and repeatedly accessed -> use `__ldg` (hash-bucket scan, small matmul)
- Uncertain -> do not add (zero benefit, added code complexity)

---

### P-P27: bf16 scalar conversion — Cast(bf16→float) + GetValue

**Severity**: CRITICAL | **Source**: A5 measurement (2026-03-31) | **Platform**: Ascend950PR + CANN 9.0.0

**Key finding**: bisheng does not support `static_cast<float>(bfloat16_t)` scalar conversion. SIMD `Cast()` vector intrinsic works fine.

**Anti-pattern**:
```cpp
// ❌ Compile fails: "not support bf16 type cast"
bfloat16_t val = gmBuf.GetValue(i);
float fval = static_cast<float>(val);  // FAIL

// ❌ Lossy: bf16 exponent=8bit > half exponent=5bit → value range overflows to inf
Cast(halfBuf, bf16Buf, RoundMode::CAST_NONE, n);  // bf16→half is lossy!
```

**Correct pattern (P-P27)**:
```cpp
// ✅ bf16 scalar read: DataCopyPad → Cast(bf16→float) → GetValue(float)
DataCopyPad(bf16Buf, weightGm_[offset], copyParams, padNone);
PipeBarrier<PIPE_ALL>();
Cast(floatBuf, bf16Buf, RoundMode::CAST_NONE, count);  // bf16→float is lossless
PipeBarrier<PIPE_V>();
float w = floatBuf.GetValue(i);  // float scalar read works normally

// ✅ SIMT context (cannot use Cast): bit-manipulation workaround
float simt_to_float(bfloat16_t v) {
  uint16_t bits; __builtin_memcpy(&bits, &v, sizeof(bits));
  uint32_t f32 = (uint32_t)bits << 16;
  float r; __builtin_memcpy(&r, &f32, sizeof(r)); return r;
}
```

**Type conversion path table** (reg_convert.h):

| Source→Target | SIMD Cast() | Scalar static_cast | Note |
|---------|:-----------:|:---------------:|------|
| bf16→float | ✅ `asc_bfloat162float` | ❌ | **Use Cast then GetValue** |
| float→bf16 | ✅ `asc_float2bfloat16_rn` | ❌ | Cast then SetValue |
| bf16→half | ✅ `asc_bfloat162half_rn` | ❌ | **Lossy!** exponent overflow |
| half→bf16 | ✅ `asc_half2bfloat16_rn` | ❌ | Lossy (mantissa truncation) |
| half→float | ✅ `asc_half2float` | ✅ | Both work |
| float→half | ✅ `asc_float2half_rn` | ✅ | Both work |

**Decision rules**:
1. bf16 needs a scalar value → **Cast(bf16→float) first, then GetValue**; do NOT Cast(bf16→half)
2. bf16 scalar inside a SIMT kernel → **simt_to_float bit-manipulation** (SIMD Cast is unavailable)
3. half scalar conversion → `static_cast<float>(half)` works directly, no special handling needed

**Reference**: CANN `reg_convert.h`; minimal repro at `tests/repro/bf16_cast_repro.cpp`

---

### P-P30: fp16/bf16 Scalar Kernel Argument Passing

**Severity**: High | **Source**: E2E skill test (2026-04-01) | **Platform**: All AscendC

**Problem**: `extern "C" __global__ __aicore__` kernel entry cannot directly take `half`/`bfloat16_t` scalar parameters. The ABI does not support this and it causes value corruption or undefined behaviour.

**Anti-pattern**:
```cpp
extern "C" __global__ __aicore__ void init_kernel_fp16(
    GM_ADDR data, half num, int64_t size) {  // ❌ half cannot cross the extern "C" boundary
```

**Correct pattern** (uint16_t bit-pattern):
```cpp
extern "C" __global__ __aicore__ void init_kernel_fp16(
    GM_ADDR data, uint16_t num_bits, int64_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  half num;
  *reinterpret_cast<uint16_t*>(&num) = num_bits;  // rebuild from bit pattern
  // ... use num ...
}

// Same idea for bf16:
extern "C" __global__ __aicore__ void init_kernel_bf16(
    GM_ADDR data, uint16_t num_bits, int64_t size) {
  bfloat16_t num;
  *reinterpret_cast<uint16_t*>(&num) = num_bits;
  // ...
}
```

**Host-side call**:
```cpp
half h_val = ...;
uint16_t bits = *reinterpret_cast<uint16_t*>(&h_val);
aclrtlaunch_init_kernel_fp16(..., bits, size);
```

**Trigger condition**: Any kernel with a scalar parameter that is not float/int/int64_t (especially initial-value parameters of init/fill kernels).

---

### A-P34: `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_*_ONLY)` — arch35-only macro family

**Severity**: BLOCKING (silent precision FAIL on V220) | **Source**: A3 13_Cat investigation 2026-04-25 (AIV), 1_BatchMatmul 2026-05-05 (AIC) | **chip_scope**: a5-only (banned on a3/a2)

**Anti-pattern**: emit `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);` OR `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);` in kernel.cpp when `TARGET ∈ {a3, a2}` (V220 / arch22). Both macros emit arch35-specific binary metadata.

**Symptom on V220**:
- Build succeeds; `_<op>_ext.cpython-...-aarch64-linux-gnu.so` produced normally
- Symbols look right (`aclrtlaunch_<op>_<dtype>` present)
- At Python `import _ext` time: stderr prints `RegisterAscendBinary aiv ret 107000` (AIV_ONLY) or `RegisterAscendBinary aic ret 107000` (AIC_ONLY) (= `ACL_ERROR_RT_PARAM_INVALID`)
- Kernel never executes — output stays at `torch::empty()` initial values (zeros / FP_MAX uninit pattern)
- Verification: 99% mismatch with `mean_abs_diff ≈ |ref|`, fp16/bf16 cases see `max_abs_diff = 3.40e+38` (FP_MAX, classic uninit-buffer fingerprint)

**Diagnosis**: the macros emit arch35-specific binary metadata that V220 runtime rejects. CANN's own op catalogue confirms: `grep -l KERNEL_TASK_TYPE_DEFAULT /usr/local/Ascend/cann/opp/built-in/op_impl/.../ascendc/*/*.h` matches almost exclusively under `arch35/` paths. The constraint applies to the entire `KERNEL_TYPE_*_ONLY` family — both AIC and AIV variants.

**Fix**:
- **Cleanest** (preferred for a3/a2-only kernels): omit the macro entirely. The default V220 launch path handles scheduling without an explicit task-type tag.
- **Cross-portable** (single source for a5+a3+a2): wrap in `#if __NPU_ARCH__ >= 3510` ... `#endif` so arch35 keeps the macro and arch22 skips it.

**Evidence**:
- 4 AIV ops on Ascend910_9382 / 198.51.100.92 (13_Cat / 1_GELU / 10_LayerNorm / 14_Split). Each had 3× of the macro in A5 archive; mechanical removal (`sed '/KERNEL_TASK_TYPE_DEFAULT/d'`) → all build + Pass A.
- 1_BatchMatmul (2026-05-05): `KERNEL_TYPE_AIC_ONLY` caused `RegisterAscendBinary aic ret 107000` on V220/A3. Cube kernel with 3 entry points. Wrapped with `#if __NPU_ARCH__ >= 3510` → all 3 entry points register correctly, 49/50 PASS.
- fa_gqa_grad kw-1 (2026-06-19, port_a3_to_a5, Ascend910_V220/arch22): first **multi-launch split** instance — a backward FA-class op with BOTH `KERNEL_TYPE_AIC_ONLY` (cube GEMM .cpp) AND `KERNEL_TYPE_AIV_ONLY` (softmax/dS/reduce .cpp) in the same op. On V220 it emitted `RegisterAscendBinary aiv/aic ret 107000` **plus the companion `LaunchAscendKernel ret 507000`**; kernel never executed, output stayed at `torch::empty/zeros` (absmax=0). Guarding BOTH macros with `#if defined(__NPU_ARCH__) && (__NPU_ARCH__ >= 3510)` → both register, kernels execute on cube/vec cores (107000/507000 gone). Confirms the family rule holds for a **split AIC+AIV multi-launch** (prior anchors were single-type: 13_Cat single AIV, 1_BatchMatmul single AIC) and documents `507000 LaunchAscendKernel` as the launch-side companion to the `107000 RegisterAscendBinary` register-side symptom.

**Enforcement**:
- `aog-kernel-worker.md` Phase 0 (V3.4) rule 2b: hard-exit when emitting the macro on `PLATFORM_SIMT=false`
- `workflow_critic.py` G7-target: SIMT_PRIMITIVES list includes `KERNEL_TASK_TYPE_DEFAULT` (with C/C++ comment-strip so explanatory `// removed for V220` notes don't false-positive)

---

### A-P35: fp32 transcendental Pass B exemption — when reference dispatches to CANN aclnn-* polynomial

**Severity**: Medium (contract gap, not a kernel bug) | **Source**: A3 1_GELU investigation 2026-04-25 | **chip_scope**: all

**Symptom**: Pass A (atol/rtol=1e-2) passes, Pass B (`torch.equal` bit-exact) fails with `max_abs_diff < 1e-3` across most cases.

**Diagnosis**: the kernel uses **manual decomposition** (e.g. `Exp + Reciprocal` for sigmoid, hand-rolled polynomial for tanh) instead of the corresponding **advanced API** (`Sigmoid`, `Tanh`, `Erf`, ...). Manual decomposition diverges from CANN's internal `aclnn<Op>` polynomial. The advanced API and `aclnn<Op>` share the same internal polynomial → bit-exact match.

**Two regimes — always check which applies**:

| Path | Bit-exact vs CANN ref? | Example |
|------|------------------------|---------|
| **Advanced API** (`Sigmoid`, `Tanh`, `Erf`, `Silu`, `Swish`, `Softmax`, `Matmul`) | **YES** — designed to match CANN polynomial bit-exact | op#1 GELU `Erf()` 50/50 PASS Pass A; op#11 `Sigmoid()` 50/50 PASS for silu |
| **Manual decomposition** (hand-rolled `Exp + Reciprocal`, custom polynomial coefficients) | NO — drifts from CANN polynomial by 4-4200 ULP | Old op#1 GELU before switch to Erf; any kernel that grep-misses catalog and falls back to manual primitives |

**Therefore**:
1. **First**: check `src/skills/references/target/ascendc/API_CATALOG.md §9.1` for an advanced API matching your op. If catalog list is incomplete, **`ls cann-{ver}/aarch64-linux/asc/include/adv_api/`** to enumerate real available advanced API directories (catalog miss ≠ API doesn't exist; see EC-34 / OL-91).
2. If advanced API exists → use it; expect bit-exact Pass A and Pass B.
3. If genuinely no advanced API for the op → manual decomposition is the only path; **then** A-P35's contract softening applies (Pass B 1e-3 tolerance OR skip Pass B when reference dispatches `aclnn-*`).

CANN's `aclnn<Op>` source for arch22/V220 is not exposed in the public AscendC API surface. The arch35 (a5) version exists at `cann/opp/.../arch35/<op>_dag.h` but is forbidden to copy per `CLAUDE.md` (NPUKernelBench scope rules).

**Affected ops** (when manual-decomposition path is forced): gelu / sigmoid / tanh / silu / swish / softmax (any reduction-coupled transcendental) / log / exp / erf / erfc / log1p / expm1. **All of these have advanced API entries** in catalog §9.1 — the manual-decomposition fallback is rarely the right choice.

**Fix (contract softening, manual-decomposition regime ONLY)**:
- Pass B should relax to `torch.allclose(atol=1e-3, rtol=1e-3)` for transcendental ops
- OR auto-skip Pass B when the op's reference dispatches to `aclnn-*` (detect by the aog-kernel-worker's analysis classifying op_type ∈ transcendental_set)

**Anti-pattern (kernel-side)**:
1. Do NOT try to "fix" by replacing the public advanced API with a hand-rolled polynomial that matches `aclnnGelu`'s coefficients. That would (a) require copying CANN source (forbidden), (b) drift on next CANN version, (c) miss the point.
2. Do NOT fall back to manual decomposition when an advanced API exists in the headers but is missing from catalog. Run `ls adv_api/` first.

**Verified evidence**:
- op#1 GELU on Ascend910_9382 with manual `Erf()` (advanced API): 50/50 Pass A bit-exact at 1e-2 ✓ (Pass B 6/30 with 4.7e-4 ULP residuals = polynomial-difference between hand-rolled Erf at higher precision vs aclnnGelu — but Pass A IS bit-exact when adv API matches).
- op#11 DequantSwigluQuant on Ascend950PR with `Sigmoid(tmp, work, ...)` (advanced API): 50/50 Pass A PASS ✓ (silu = Sigmoid * Self bit-exact via advanced API path).
- Counter-example: any pre-2026-04-28 kernel that fell back to `Exp + Reciprocal + Mul` for sigmoid → ULP-divergence from CANN reference, FAILED Pass A.

---

### P-P51: Lift `nblk` to runtime AIV count for V220 portability

**Severity**: Medium (perf optimization) | **Source**: A3 1_GELU optimizer experiment 2026-04-25 | **chip_scope**: v220-common

**Anti-pattern**: pybind11.cpp hardcodes `uint32_t nblk = 56;` (A5 AIV count) when launching AscendC kernels.

**Why it matters on V220**: Ascend 910C has 80–96 AIVs; Ascend 910B has 40–48. A static `nblk = 56` either underutilizes (910C) or overcommits (910B b3/b4 with 40 AIVs). Compute-light ops (where launch parallelism dominates) lose 15–30% mean perf on big-shape cases.

**Fix recipe** (pybind11.cpp):
```c++
// Lift nblk to a runtime query of current device's AIV count.
// Option A — pure host-side: derive via aclrtGetCurrentNPUInfo or torch_npu's
//   GetCurrentDeviceProperties (specifics depend on CANN version).
// Option B — wrap a tiny "info" kernel that returns GetBlockNum() and use that.
// Option C — fallback static table per SOC variant from a header constant.

uint32_t nblk;
{
    // PoC: read once and cache. Fall back to A5's 56 if query fails.
    static int cached_nblk = 0;
    if (cached_nblk == 0) {
        // ... CANN host API or device-info query ...
        cached_nblk = query_aiv_count_for_active_device();
        if (cached_nblk <= 0) cached_nblk = 56;  // safe default
    }
    nblk = cached_nblk;
}
```

**Quick proxy** (one-line, low effort): bump `nblk = 56` to `nblk = 80` for a3-only builds. Verified: 1_GELU mean perf 0.48x → 0.60x (+16%), Pass A unchanged. Caveat: hurts a3 chips with fewer AIVs and over-spawns blocks for small-shape ops (very small launch overhead, amortizes worse).

**Trade-off vs always-use-max**: with `nblk = max_AIV`, small-shape ops pay launch overhead for blocks that do trivial work. The optimizer should consider per-shape adaptive nblk for compute-light ops where launch dominates.

**Verified-on**: 1_GELU on Ascend910_9382 — mean +16%, median slightly worse for small shapes (launch amortization). 14_Split predicted +30% (large-shape tail offenders dominate mean). To-do: apply across the 4 archived A3 ops and remeasure.

---

### A-P36: V220 `aclnnCumsum` chip-specific dim-dependent fp16 path (V351 has none)

**Severity**: High (precision blocker for fp16 non-innermost-dim cumsum on V220) | **Source**: A3 5_Cumsum 2026-04-26 (probe + msprof + V351 sibling-chip cross-check) | **chip_scope**: v220-common

**Empirical observation** — same probe run on both chips, same fp16 inputs, identical kernel-side approach (movedim-to-last + sequential fp32-acc + RINT):

| shape | dim | V351 (Ascend950PR_957b) `y_a` ≡ `y_b`? | V220 (Ascend910_9382) `y_a` ≡ `y_b`? |
|---|---|---|---|
| `[128,128]` | 0 | **bit-identical** | differ, max=0.125 |
| `[256,256]` | -2 | **bit-identical** | differ, max=0.188 |
| `[1,16,64,64]` | 1 | **bit-identical** | differ, max≈0.05 |
| `[8192,16384]` | -2 | (within fp16 reduction envelope, max=0.25) | **differ, max=1.125** |
| `[1024,2048]` | -2 | **bit-identical** | differ, max=1.125 |

Where `y_a = torch.cumsum(x, dim=dim)` (direct CANN call) and `y_b = torch.cumsum(x.movedim(dim,-1).contiguous(), dim=-1).movedim(-1, dim).contiguous()` (movedim-equivalent path).

**What this means**: V220's `aclnnCumsum` for `dim ≠ ndim-1` takes a **chip-specific optimization path** that V351's CANN does NOT have. V351 just runs the obvious "permute → innermost-cumsum → permute back" algorithm — so any kernel that does the same (e.g. our movedim approach) bit-matches V351's reference. V220 instead launches a SIMD kernel **on the original layout** with BlockDim=48 (msprof confirmed: single kernel `aclnnCumsum_CumsumAiCore_Cumsum`, AIV-only, no multi-stage tree).

**Sub-cases (V220 path)**:
- **scan_len ≤ 48 + numLines small (BlockDim=1)**: pure fp16 vectorized running buffer on original layout. **Reverse-engineered, bit-reproducible** (probe 03/05). Implement with `Add<half>` + sequential row iterations on original layout (NO movedim).
- **scan_len ≥ 64 + numLines large (BlockDim=48)**: V220 multi-AIV path, each AIV processes a column-slice. Algorithm not bit-reproducible from public AscendC primitives after 4 reverse-engineering rounds (probe 04 swept 36 chunk-K × strategy combos; closest reaches max=1.125 vs y_a but never bit-exact). msprof shows aiv_vec_fp16_ratio=0.002 + aiv_vec_fp32_ratio=0.002 (both low, neither pure-fp16 nor pure-fp32).

**Why same algorithm passes on V351 + fails on V220**: V351's CANN does NOT have this dim-dependent optimization, so it accepts kernels that take the obvious movedim path (our kernel + A5's archived `output/npukernelbench/src/kernels/5_Cumsum/` both 51/51 PASS on V351). On V220, the CANN reference itself uses the chip-specific path → kernels that take the movedim path differ from the reference (even though our kernel is **closer to fp64 ground truth** than the V220 reference is).

**Fix recipe (V220 only)**:
1. **For BlockDim=1 paths (small numLines, scan_len ≤ 48)**: implement `ProcessLineFp16VectorRunning` per probe 03 reverse-engineering. Skeleton: keep tensor on original layout, iterate scan dim sequentially, use `Add<half>(running, running, x_row, row_width)` SIMD. Closes case 29-class shapes.
2. **For BlockDim=48 paths (long scans)**: NO known public-AscendC fix. Document as `chip_scope: a3-only` PARTIAL with empirical evidence. Do not label this as a generic precision floor — the gap is V220 chip-specific, not generic CANN reference quirk. V351 with same algorithm passes 51/51.

**Diagnostic protocol** when encountering a similar fp16 non-innermost-dim mismatch on V220:
1. Run sibling-chip probe (V351 via `/a5_op`): if V351 sees `y_a ≡ y_b`, you're hitting A-P36.
2. Run `msprof` on the V220 reference call to confirm single-kernel SIMD launch + BlockDim=48 signature.
3. Document with chip-specific KB entry, not generic waiver.

**Verified-on**: A3 5_Cumsum, Ascend910_9382 (198.51.100.92, container npu-a3), 2026-04-26 — 5 fp16 Pass A failures (cases 5/8/29/38/40 in benchmark JSON) + 19 fp32 edge-dataset failures, all share the V220 chip-specific reduction-order signature.

**Cross-reference**: aog-self-critic C19 (sibling-project cross-check) and C20 (use msprof before declaring "blocked") were added to catalog 2026-04-26 to prevent the next agent from spending 5 probe rounds reverse-engineering blind without first running msprof on the reference.

**Forward question for V220 future ops**: any reduction op (`aclnn{Sum,Mean,Norm,Softmax,...}`) on dim ≠ -1 may exhibit similar V220-vs-V351 divergence. Check sibling chip + msprof BEFORE assuming the kernel is at fault.

---

## Known Bugs

### TQue depth=1 data race

**Severity**: High | **Source**: A5 SIMD measurement

`TQue<TPosition::VECIN, 1>` exhibits intermittent data races in certain kernels (random result deviation).
**Temporary fix**: Use depth=2 double buffering. Root cause pending confirmation from the CANN team.

See the discussion of TQue in U-P2 in [unverified/candidates.md](../unverified/candidates.md).

---

## P-P68: Single-AIC GEMM with constexpr static tiling + on-stack TCubeTiling

**Severity**: HIGH | **Source**: Level-3 cube playbook (op#1 → op#3, 4-for-4)

### Trigger
Level-3 cube op (matmul / batch_matmul / gemm / Linear / Conv) where the gemm fits a single AIC (M/N ≤ ~512, fp32/fp16/bf16, ND inputs+outputs). When operands need logical transpose, layer P-P69 on top.

### Pattern

```cpp
template <typename T>
__aicore__ inline constexpr MatmulApiStaticTiling make_static_cfg() {
    MatmulApiStaticTiling t{};                      // every shape field defaults to -1
    t.cfg = CFG_NORM;
    t.usedCoreNum = 1;
    t.baseM = 128; t.baseN = 128;
    t.baseK = std::is_same_v<T, float> ? 64 : 128;  // L0A budget per dtype
    t.depthA1 = 8; t.depthB1 = 8;
    t.stepM = 1; t.stepN = 1; t.stepKa = 1; t.stepKb = 1;
    t.dbL0A = 1; t.dbL0B = 1; t.dbL0C = 1;
    t.iterateOrder = 0;
    t.isBias = 0; t.transLength = 0;
    return t;
}

template <typename T>
__aicore__ void op_one_impl(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                            int32_t M_, int32_t N_, int32_t K_) {
    static constexpr auto MM_CFG = make_static_cfg<T>();
    using AT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;
    using BT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;
    using CT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;

    GlobalTensor<T> aT, bT, cT;
    aT.SetGlobalBuffer((__gm__ T*)a, M_ * K_);
    bT.SetGlobalBuffer((__gm__ T*)b, K_ * N_);
    cT.SetGlobalBuffer((__gm__ T*)c, M_ * N_);

    TPipe pipe;
    TBuf<TPosition::VECCALC> ubScratch;
    pipe.InitBuffer(ubScratch, 1024);              // static-check workaround

    TCubeTiling tiling{};                           // on-stack, runtime fields only
    tiling.M = M_; tiling.N = N_; tiling.Ka = K_; tiling.Kb = K_;
    tiling.singleCoreM = M_; tiling.singleCoreN = N_; tiling.singleCoreK = K_;

    MatmulImpl<AT, BT, CT, /*BIAS=*/CT, MM_CFG> mm;
    mm.Init(&tiling, &pipe);                        // non-__gm__ overload — no host H2D
    mm.SetTensorA(aT, /*isTransposeA=*/false);      // see P-P69 for trans variants
    mm.SetTensorB(bT, /*isTransposeB=*/false);
    mm.SetSingleShape(M_, N_, K_);
    mm.IterateAll<true>(cT, 0, false, false, false);
    mm.End();
}
```

### Performance unlock — why constexpr beats runtime tiling

`asc/impl/adv_api/detail/matmul/utils/matmul_utils.h::CopyTiling`:
```cpp
if constexpr (MM_CFG.<field> == -1) cubeTiling.<field> = gmCubeTiling-><field>;
```
**Each non-(-1) field eliminates one GM read in the hot path.** With ~25 shape-independent fields constexpr, only `M/N/Ka/Kb/singleCore*` remain runtime — and those fit on the stack, eliminating the ~5–10 µs `torch::empty(200B) + .copy_()` H2D too.

### Determinism (by-construction)
- blockDim ≤ batch_n with 1 AIC per output tile — no cross-core comm
- No `SetAtomicAdd` — fp32-accum mmad order fixed by constexpr tiling
- `IterateAll<sync=true>` + `End()` barrier per launch
- Satisfies `DET_POLICY=required` without algorithm contortion

### Performance trajectory (op#1 BatchMatmul)

| Variant | Ratio | asc median ms | kernel task_dur | scalar_ratio |
|---------|-------|---------------|-----------------|--------------|
| Opt0 (runtime tiling 64×64) | 0.515× | 0.033 | 4.34 µs | 0.92 |
| Opt1 (128×128 block, runtime tiling) | 0.543× | 0.035 | 3.23 µs | 0.64 |
| **Opt2 (constexpr + on-stack)** | **1.267×** | **0.015** | **2.67 µs** | **0.51** |

Validated on:
- 1_BatchMatmul (Opt2): 1.27× median, 51/51 + 14/14 PASS, fp32 bit-exact
- 4_MatmulTransA: 1.36× median, 50/50 + 16/16 PASS (P-P69 layered)
- 5_MatmulTransB: 1.29× median, 50/50 + 16/16 PASS (P-P69 layered)
- 3_MatmulBothTrans: 1.45× median, 50/50 + 16/16 PASS (P-P69 layered, both bools true)

### Build invariants (CRITICAL — see EC-39 / EC-40)
- `MM_CFG` MUST be `MatmulApiStaticTiling` (NOT `CFG_NORM` directly) → EC-39
- Host-side POD mirroring `TCubeTiling` is 50 int32 = 200 B → EC-40
- `make_static_cfg<T>()` must be `__aicore__ inline constexpr`
- Local `constexpr auto X = factory<T>()` inside templated function must be `static constexpr`

### Static-check workaround
`ascendc_static_check.py kernel_has_computation` requires ≥3 of {TQue/TBuf, DataCopy, VEC_op, GlobalTensor, LocalTensor}. Cube kernels using `MatmulImpl::IterateAll` legitimately have only `GlobalTensor`. Use a 1-KB unused `TBuf<VECCALC>` scratch as workaround until the marker set is extended.

### When NOT to apply
- batch > 1 with non-uniform shapes per batch — use IterateBatch (TODO)
- Very large M/N/P (> ~1024) — needs multi-AIC partitioning
- MX-FP8 / quantized matmul — has its own scale-tile path

---

## P-P69: Cube transposed-input via runtime `SetTensor*(_, isTrans=true)` — NOT template ISTRANS

**Severity**: CRITICAL | **Source**: op#4 Phase D iter 1 (precision FAIL with template ISTRANS=true), validated across {A, B, both} corners

### Trigger
Level-3 cube op with logical transpose: `torch.matmul(A.T, B)`, `torch.matmul(A, B.T)`, `torch.matmul(A.T, B.T)`, `nn.Linear` with weight transpose, Conv backward weights. Direct extension of P-P68.

### Pattern (the runtime bool drives transpose, NOT the template flag)

```cpp
// All MatmulType template flags stay default false:
using AT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;
using BT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;
using CT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;

// Runtime bools do the actual transpose:
mm.SetTensorA(aTensor, /*isTransposeA=*/true);   // for A.T @ B  or  A.T @ B.T
mm.SetTensorB(bTensor, /*isTransposeB=*/true);   // for A   @ B.T or  A.T @ B.T
mm.SetSingleShape(/*M=*/output_rows, /*N=*/output_cols, /*K=*/reduction);
```

### Tiling field map — shape-flat regardless of which side transposes

| Op | M (out rows) | N (out cols) | Ka, Kb (reduction) | A bool | B bool |
|----|--------------|--------------|--------------------|--------|--------|
| `A @ B` (no trans) | A.size(0) | B.size(1) | A.size(1)=B.size(0) | false | false |
| `A.T @ B` (TransA) | A.size(1) | B.size(1) | A.size(0)=B.size(0) | **true** | false |
| `A @ B.T` (TransB) | A.size(0) | B.size(0) | A.size(1)=B.size(1) | false | **true** |
| `A.T @ B.T` (BothTrans) | A.size(1) | B.size(0) | A.size(0)=B.size(1) | **true** | **true** |

### Mechanism trace (CANN 9.0.0)
- Static `MatmulType<..., ISTRANS>` flag is stored as `A_TYPE::isTrans` (`asc/impl/adv_api/detail/matmul/utils/matmul_type_def.h:42`) but referenced ONLY in MX-FP8 / scale path (`mx_matmul_utils.h:321`).
- Actual ND→ND transpose driver: runtime member `MatmulShapeInfoBase::isTransposeA_` set by `SetTransposeA(bool)` or `SetTensorA(gm, bool)` 2nd arg. `IsTransposeA()` reads the runtime member.
- A-side and B-side bools are independent and symmetric — no special handling for "both true".

### Anti-pattern (compiles, FAILS precision with garbage output)
```cpp
using AT = MatmulType<..., /*ISTRANS=*/true>;   // template flag — IGNORED in ND→ND
mm.SetTensorA(aTensor, /*isTransposeA=*/false); // runtime — drives transpose
// → cube computes A @ B (no transpose); output has algorithmic noise (max_abs_diff 5-120)
```

### Evidence
- 4_MatmulTransA Phase D iter 1: `ISTRANS=true` template + `SetTensorA(_, false)` → max_abs_diff 5–120, mean_abs_diff 5–18. Iter 2 with runtime-bool fix: 50/50 + 16/16 PASS, fp32 bit-exact, 1.36× median.
- 5_MatmulTransB: `SetTensorB(_, true)` only, 0+0 iters, 1.29× median, fp32 bit-exact.
- 3_MatmulBothTrans: BOTH bools true, 0+0 iters, 1.45× median, fp32 bit-exact. Confirms A/B symmetry.
- The {none, A, B, both} 4-corner lattice is now fully validated.

### Combine with P-P68
P-P69 only specifies the transpose mechanism. constexpr static tiling, on-stack TCubeTiling, AIC scheduling come from P-P68. Use them together for any single-AIC transposed GEMM.

---

## P-P74: Multi-AIC partition-dispatch via host-precomputed segment offsets

> ID coordination note: a3 PR #2 reserves P-P70-P-P73 for fused-quant patterns. P-P74 is the next free slot at the time of writing. If a3 PR #2 lands after this entry but with overlapping numbering, rebase by shifting whichever side is later.

### When to use
Any kernel whose output is the row-concatenation of N independent sub-computations where the per-segment row counts are known on the host. Direct multi-AIC extension of P-P68 (which covers the single-AIC case). Concrete instances:
- Grouped / segmented matmul (`torch._grouped_mm`, `F.grouped_mm`, MoE expert dispatch)
- Variable-length / jagged batched GEMM where each "batch" has a distinct row count
- Segmented attention / segmented sparse-gather / GroupedConv where each group's compute is independent and outputs occupy a contiguous output row range

Trigger: source has a Python-level loop over groups (`for g: out_g = compute(A_g, ...)`) followed by a row-axis concat, OR a single fused op (`grouped_mm`) whose semantics are equivalent.

### Pattern
- **Dispatch**: `blockDim = G` (number of segments). Each AIC owns exactly one segment's compute. Layer P-P68's per-AIC machinery (constexpr `MatmulApiStaticTiling`, on-stack `TCubeTiling`) inside.
- **Host pre-compute** (pybind11.cpp): build `cum_out[G+1]` cumulative output row offsets in fp32 int32 vector, push as int32 NPU tensor. **Sentinel**: set `cum_out[G] = total_rows` so the kernel reads `end_row = cum_out[g+1]` uniformly with no special-case for the last segment.
- **Per-AIC kernel decode** — uniform-vs-variable input slicing is a host flag, not a per-op detail:
  ```cpp
  const int32_t bid = GetBlockIdx();
  // For segments with variable input rows (e.g. 2D-A grouped_mm):
  //   input_row_off = offsets[bid]; M_g = offsets[bid+1] - input_row_off;
  // For segments with uniform input rows (e.g. 3D-A stacked grouped_mm):
  //   input_row_off = bid * m_uniform; M_g = m_uniform;
  // Output side always uses cum_out[bid] / cum_out[bid+1] regardless.
  ```
- **Determinism by-construction**: each output row owned by exactly one AIC; no `SetAtomicAdd`; per-AIC mmad order fixed by constexpr tiling; `IterateAll<sync=true>` + `End()` per AIC. `DET_POLICY=required` satisfied without atomicCAS / lock-bit games.
- **Reference availability fallback** (when target torch+CANN lacks `grouped_mm`): use OL-89 prose-spec extension to write a workspace-local `model.py` that loops `torch.matmul(A_g, B[g])` per segment and concatenates. Decomposition is mechanical (no rounding ambiguity, no contraction-order ambiguity), satisfies `verification_ascendc.py` Pass A, matches per-AIC kernel output bit-exact in fp32.

### Limitation (when to reach for the next pattern)
`blockDim = G` caps parallelism at the number of segments. For G ≤ AIV_count with large per-segment GEMMs (M·N·K > ~1M elements), a single AIC cannot match the reference's 56-AIC `aclnnMatmul`. Worst-case ratio drops while median stays > 1× because small-G cases dominate. The next architectural move is a 2D dispatch (`blockDim = G × per_segment_tile_count`) with K-split or L0C-stage merge — aog-kernel-optimizer territory, not kw correctness.

### Evidence
- 2_GroupedMatmul (op#2, 2026-04-28): 50/50 + 16/16 PASS bit-exact, det 50/50 identical, median 1.05×, gmean 0.83× (PASS by V3.3.6 median methodology). 5th level-3 op closing the matmul family. **0 build iters + 0 precision iters first try** — cube playbook (P-P68 + P-P69 + P-P74) fully amortized.
- See OL-93 for the op#2-specific evidence record and 2D-A vs 3D-A flag handling.

### Combine with P-P68 / P-P69
- P-P74 specifies the multi-AIC dispatch architecture (blockDim=G, host-side row partitioning, sentinel-extended offsets, bid-as-segment decode).
- P-P68 supplies each AIC's GEMM machinery (constexpr static tiling, on-stack TCubeTiling, non-`__gm__` `Init`).
- P-P69 supplies any per-segment transpose (runtime `SetTensor*` bool, never template ISTRANS).
- Use all three together for grouped/segmented cube ops with logical transpose.

---

## P-P75: Manual TBuf pipeline with explicit `SetFlag/WaitFlag<MTE2_V>` event sync (V220-confirmed; A5 likely-applicable)

**Severity**: HIGH | **Source**: op#27 a3 V220 cold-start (2026-04-28); DS V4 worker session (same date) | **Platform**: V220 (Atlas A3) confirmed; A5 (Ascend950PR) UNVERIFIED but likely-applicable

### Trigger
Pure-VEC pipeline kernel (DataCopy in → VEC compute → DataCopy out, repeated per row/tile) where the chosen UB-resident primitive is `TBuf<VECCALC>` rather than `TQue<VECIN/VECOUT>`. See OL-94 for the decision rule on which primitive to pick; this pattern is for the TBuf branch.

### Why this template exists
The natural CANN-style port using `TBuf + PipeBarrier<PIPE_ALL>()` triggers PB-21 (silent crash 507015) on V220 — `PipeBarrier<PIPE_ALL>()` does NOT carry MTE2→V completion guarantees on TBuf-resident pipelines. Explicit `SetFlag/WaitFlag` is the only safe pattern. Worker sessions across two model classes (the A5 backend on op#27 and DS V4 on a similar fused op) hit this trap; this template + PB-21 + OL-94 close the loop.

### Pattern (concrete piece — copy and adapt, not a full kernel)

```cpp
#include "kernel_operator.h"
using namespace AscendC;

class MyTBufKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, /* tiling args */) {
        gmX_.SetGlobalBuffer((__gm__ T*)x, /*size*/);
        gmY_.SetGlobalBuffer((__gm__ T*)y, /*size*/);
        // Allocate UB buffers (TBuf, NOT TQue):
        pipe_.InitBuffer(bufA_, /*per-row bytes*/);
        pipe_.InitBuffer(bufB_, /*per-row bytes*/);
        // Fetch event IDs ONCE per kernel (do NOT re-fetch per iter):
        evMte2V_ = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        evVMte3_ = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    }

    __aicore__ inline void Process() {
        for (int r = 0; r < numRows_; ++r) {
            LocalTensor<T> a = bufA_.Get<T>();
            LocalTensor<T> b = bufB_.Get<T>();

            // Stage 1: MTE2 — load row r
            DataCopyPad(a, gmX_[r * H_], copyParams_, padParams_);
            SetFlag<HardEvent::MTE2_V>(evMte2V_);     // mark MTE2 done
            WaitFlag<HardEvent::MTE2_V>(evMte2V_);    // V waits for MTE2

            // Stage 2: V — compute (Cast / Mul / Add / etc.)
            Cast(b, a, RoundMode::CAST_NONE, H_);
            // ... more VEC ops on a / b ...
            SetFlag<HardEvent::V_MTE3>(evVMte3_);     // mark V done
            WaitFlag<HardEvent::V_MTE3>(evVMte3_);    // MTE3 waits for V

            // Stage 3: MTE3 — store row r
            DataCopy(gmY_[r * H_], b, H_);
        }
    }

private:
    GlobalTensor<T> gmX_, gmY_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> bufA_, bufB_;
    uint16_t evMte2V_, evVMte3_;
    int numRows_, H_;
    DataCopyExtParams copyParams_;
    DataCopyPadExtParams<T> padParams_;
};
```

### Critical rules
1. **Fetch event IDs ONCE** outside the loop (`FetchEventID` allocates from a finite pool — re-fetching per iter exhausts it).
2. **Pair every `SetFlag` with a `WaitFlag`** at the next stage boundary. Missing pair → silent stall or race.
3. **Do NOT use `PipeBarrier<PIPE_ALL>()`** between MTE2 and V on TBuf — PB-21 (silent crash 507015 on V220).
4. If two TBufs participate in the same MTE2→V handoff (e.g. loading both `a` and `b` per row before compute), use ONE `SetFlag<MTE2_V>` after both `DataCopyPad` calls + ONE `WaitFlag<MTE2_V>`. The barrier is per-stage, not per-buffer.
5. For multi-stage VEC compute that includes scalar dependencies, intermediate `S_V` / `V_S` events may be needed (see PB-9 / PB-20 for the scalar-pipe-on-V220 nuances).

### Anti-pattern (compiles, runs, silent crash 507015 on V220)
```cpp
DataCopyPad(a, gmX_[r * H_], cp, padParams_);
PipeBarrier<PIPE_ALL>();    // ❌ does NOT guarantee MTE2_V on V220 TBuf
Cast(b, a, RoundMode::CAST_NONE, H_);  // V op fires before MTE2 complete → garbage / crash
```
Replace with the `SetFlag/WaitFlag<MTE2_V>` pair shown above.

### When NOT to apply (use TQue instead)
- Standard pointwise / cast / strided copy chains where dataflow fits a 3-stage pipeline cleanly — TQue's auto-rotation is cheaper to author and equivalent in perf.
- See OL-94 decision table for the full pick-list.

### Evidence
- **op#27 `27_MultiMaskAttentionAggregation` a3 V220** (2026-04-28): worker initial impl with `TBuf + PipeBarrier<PIPE_ALL>()` → silent crash 507015 across all cases, 5 iters wasted. Switched to this `SetFlag/WaitFlag<MTE2_V>` pattern → 50/50 PASS, det 100/100. Probe report at `output/npukernelbench-a3/src/kernels/27_MultiMaskAttentionAggregation/probe_report.md` (a3 PR #2 v2).
- **DS V4 worker session** (2026-04-28): weaker model defaulted to TQue on an op needing TBuf (multi-buffer aliasing across phases). Crashed at runtime; recovered by switching to TBuf + this pattern after 5 iters. Surfaced the gap that prompted P-P75 + OL-94 + PB-21 codification.

### Cross-reference
- **OL-94**: when to pick TQue vs TBuf (decision rule + table).
- **PB-21**: the specific silent-crash-507015 trap this pattern avoids.
- **PB-9**: V220 UB→UB DataCopy nuance (different sync issue).
- **P-P28** (TQue<4> auto pipeline): when TQue is the right pick — TQue auto-rotation replaces the manual sync pattern when dataflow fits.

---

## P-P76: Aligned-base scratch via `Duplicate(0) + scalar SetValue` for inline VEC reductions with unaligned index offsets

**Severity**: HIGH | **Source**: 6_ConvStandard1d (first conv-family op, 2026-04-29) | **Platform**: Ascend950PR (AIV path)

### Trigger

Per-row VEC reduction kernel where the inner accumulation is `Axpy(acc, src[runtime_offset], scalar, len)` (or equivalent FMA chain), and the runtime offset is data-dependent and frequently NOT 32B-aligned. Canonical case: 1D/2D/3D conv with kernel_size > 1 and padding > 0 — the per-`kernel_position` offset `op_start = max(0, ceil_div(-(kp*dilation - padding), stride))` is typically 1/2/3 for the boundary positions. Other instances: pooling with stride > 1, dilated patches, scatter-with-runtime-offset, any reduction where the source slice base is data-dependent.

### The trap

VEC instructions on AIV require **32B-aligned bases** for both src and dst LocalTensor offsets (8 fp32 / 16 fp16 elements). The "natural" formulation:

```cpp
// ❌ Crash: error code 340 ("UB address not aligned")
// op_start can be 1/2/3 (small unaligned offsets) for boundary kernel positions
const int op_start = max(0, ceil_div(-base, stride));
const int in_start = max(0, base);
Axpy(accLocal[op_start], inLocal[in_start], w, len);
```

compiles fine, runs cleanly when `op_start = in_start = 0`, and crashes hard when either becomes non-aligned mid-loop. The crash is at runtime (error 340), not at compile time, so it's only caught by actually running on NPU.

### The pattern

Build an **aligned-base tmp scratch** via VEC zero-fill + scalar fill of the valid range:

```cpp
// ✅ Aligned-base — Axpy operates on len_pad with zero-padded inactive elements
const int Lo_pad = round_up_to_8(Lo);   // 32B align
TBuf<TPosition::VECCALC> tmpBuf;        // sized Lo_pad fp32

LocalTensor<float> tmp = tmpBuf.Get<float>();
Duplicate(tmp, 0.0f, Lo_pad);            // VEC zero, aligned

const int op_start = max(0, ceil_div(-base, stride));
const int op_end   = min(Lo, ceil_div(L - base, stride));
for (int op = op_start; op < op_end; ++op) {
    tmp.SetValue(op, in.GetValue(op * stride + base));   // scalar fill, no align constraint
}
SetFlag<HardEvent::S_V>(evSV); WaitFlag<HardEvent::S_V>(evSV);
Axpy(acc, tmp, w, Lo_pad);              // all bases aligned, single FMA op over Lo_pad
SetFlag<HardEvent::V_S>(evVS); WaitFlag<HardEvent::V_S>(evVS);
```

The zero-padded inactive elements (positions outside `[op_start, op_end)`) contribute `0 * w = 0` to `acc`, so they don't change the result. Cost: `~Lo` scalar SetValues per iteration of the outer loop — negligible on AIV vs the alternative of crashing or computing a fully aligned manual unroll.

### When to use vs alternatives

| Alternative | When to prefer |
|---|---|
| **Aligned-base scratch (this pattern)** | Inner reduction loop with data-dependent offsets, len ≤ a few KB, AIV path |
| Manual unroll of unaligned head/tail | Compile-time-known shape with mostly-aligned offsets and a small unaligned region |
| im2col + cube path | Throughput-bound reduction where the data-staging cost amortizes; offsets become loop indices not src bases |
| `DataCopy` with non-32B-aligned base | NEVER — same alignment requirement as VEC ops |

### Critical companion: output-side EC-23 workaround

For the same op family, the output is also typically not 32B-aligned (e.g. conv output length can be any positive int). Combined pattern:

```cpp
// pybind11.cpp: over-allocate output so each row is aligned
const int Lo_pad = round_up_to_8(Lo);
auto out = torch::empty({B, Cout, Lo_pad}, opts_f32);
// ... launch kernel, kernel writes [B, Cout, Lo_pad] via aligned DataCopy ...
return out.narrow(2, 0, Lo).contiguous();  // discard junk tail
```

This trades ~7 fp32 (28 bytes) per row of GM for crash-free output. Same trade-off principle as the input-scratch pattern: pay a small bounded overhead to satisfy alignment, instead of paying a large bounded overhead (or a crash) to handle unaligned offsets in-place. See **EC-23** for the underlying `DataCopyPad UB→GM crash error 507035` this works around.

### Determinism (by-construction)

The scalar-fill is sequential per-AIV; `Duplicate` is deterministic; `Axpy` is single-round FMA bit-aligned with `fmaf`. Reduction order over the outer accumulation loop is fixed by the for-loop structure. No atomic / no cross-AIV merge.

### Evidence

- **6_ConvStandard1d (2026-04-29)** — direct-VEC 1D conv via per-(batch, out_ch) AIV core. Phase D iter 1 hit error 340 with offset-Axpy formulation; switched to this pattern + EC-23 output pre-pad → 50/50 Pass A bit-exact at harness tolerance (atol=rtol=0.01) + 16/16 Pass B + det 50/50 + perf median 0.654× (≥ 0.6 threshold). 0 build + 1 precision iter total.

### Other instances (predicted)

- 7_ConvStandard2d / 8_ConvStandard3d — same kernel-size + stride + padding boundary pattern, same alignment issue. Direct port.
- 9_ConvDepthwise2d — depthwise variant, same trap (groups=Cin → Cin_per_g=1 → tighter weight-stride alignment too; weight loaded via scalar GetValue per kernel position).
- 10_ConvTranspose2d — output position arithmetic is different but boundary patterns still produce unaligned offsets.
- Any 1D/2D/3D pooling with stride > 1 + non-trivial padding.
- Any "scatter with runtime offset" pattern where the destination index is data-dependent.

### Combine with other patterns

- **OL-93** (multi-instance partition-dispatch): per-output-row dispatch with `blockDim = B × Cout` is the uniform-partition variant.
- **EC-23** (DataCopyPad UB→GM crash 507035): output over-allocation + narrow-on-return is the symmetric companion for write side.
- **Axpy bit-FMA semantics** (catalog): the inner accumulation is single-round FMA bit-matching `fmaf`, so accuracy is `≤ 1 ULP per accumulation step` regardless of `Lo_pad` vs `Lo`.


---

## P-P90: V220→V351 (arch22→arch35) — surgical strip pattern for op_kernel port

> **W10** (2026-05-12, ROADMAP §1.5) — extracted from `ctc_loss_v3_a5_migration_plan.md` §7 (PR4778 docs) + diff `op_kernel/ctc_loss_v3.h` vs `op_kernel/arch35/ctc_loss_v3.h` (831 vs 830 lines — surgical change, not rewrite).
>
> **ID rename note (2026-05-12)**: this entry was originally P-P89 in commit `267667a` (W8-W11 batch). Renamed to P-P90 in commit (this commit) to resolve collision with pre-existing `PATTERN_INDEX.md:148` P-P89 ("GM workspace contract for fused ops") from commit `21882d4`. References in OL-131/OL-132 cross-refs + KB_INDEX A3→A5 section + W12 op_taxonomy + test_port_a3_integration_smoke.py + ascend950pr.md + output/a3_to_a5_port/ project docs updated in same commit.

### Trigger conditions

- Op-class tag: `a3_to_a5_port`
- Phase: B.1 of port_from_a3_ascendc kw_brief (writing `op_kernel/arch35/<op>.h`)
- Input: existing `op_kernel/<op>.h` (A3/V220 kernel) — the algorithm spec

### Pattern

The V220→V351 port is mostly a **strip** operation, not a rewrite. Take the A3 kernel as starting point; remove or adjust these specific items:

1. **Strip V220 reg-primitives include**:
   ```cpp
   // REMOVE this line (V220-only):
   #include "impl/dav_c220/kernel_operator_reg_others_impl.h"
   ```
   A5 (arch35) provides reg-based primitives via default `kernel_operator.h`; the V220 explicit include conflicts on V351.

2. **Strip BF16 conditional compile blocks**:
   ```cpp
   // REMOVE:
   #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
       // V220-specific BF16 codepath
   #else
       // generic codepath
   #endif
   ```
   On V351, reg-based VEC ops support BF16/FP16/FP32 unconditionally — the V220 conditional was masking V220 BF16 quirks that don't exist on V351. Keep the body of the `#else` branch (the generic codepath); delete the `#if`/`#endif` and the V220 branch.

3. **`ToFloat<>` audit** — A5 restricts `ToFloat<T>` to `T ∈ {bfloat16_t, fp8_e5m2_t, fp8_e4m3fn_t}`. For FP16 source values, insert `.template ReinterpretCast<bfloat16_t>()` first:
   ```cpp
   // V220 (works on A3 with FP16 directly):
   float v = ToFloat(logProbTensor.GetValue(0));     // T = half

   // V351 (must reinterpret to bfloat16 first):
   float v = ToFloat(logProbTensor.template ReinterpretCast<bfloat16_t>().GetValue(0));
   ```
   See **W11** for the full ToFloat<> A5 restriction reference.

4. **Tiling header includes — generally unchanged**:
   - `kernel_operator.h` — keep (A5 native)
   - `kernel_tiling/kernel_tiling.h` — keep (target-agnostic)
   - The arch35/ kernel typically needs no new includes beyond these two; everything else was V220-specific noise.

### Anti-pattern (DO NOT)

- Do NOT delete the algorithm body. The A3 algorithm is what we want; only the platform plumbing changes.
- Do NOT add a `#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 300` wrapper. The arch35/ file is V351-only by virtue of its directory; conditional compile inside is redundant + confusing.
- Do NOT include `impl/dav_v300/*` headers explicitly unless a specific primitive requires it (rare; usually `kernel_operator.h` covers everything via target macros).
- Do NOT use the V220 conditional as a "where to insert V351 code" landmark and just flip the macro — that pattern leaves dead `#elif __CCE_AICORE__ == 200` (V220 single-die) branches that will compile on V200 builds.

### Diff size sanity-check

For L1 / L2 ports (gather_elements_v2, ctc_loss_v3, rms_norm_quant, top_k_top_p_sample_v2, group_norm_silu_quant from PR4778), `wc -l op_kernel/<op>.h` vs `wc -l op_kernel/arch35/<op>.h` typically differs by < 10 lines (just strip operations). If your `arch35/<op>.h` is more than ~10% different in line count, you're likely rewriting instead of porting — re-check Phase A analysis.md.

### Two-stage authoring pattern (recommended over derivation-strip)

Added 2026-05-12 from `gather_elements_v2` kw-1 finding: PR4778's arch35 kernel files are NOT a V220 strip-and-edit — they are **freshly authored** from the algorithm spec. Audit:
```bash
grep -nE "dav_c220|__CCE_AICORE__ == 220|ToFloat<|__NPU_ARCH__" \
    gather_elements_v2/op_kernel/arch35/*.h gather_elements_v2/op_kernel/arch35/*.cpp
# → zero matches
```

This is the **better** authoring pattern when feasible:
- **Derivation-strip** (this entry's main body): the arch35 kernel is derived from the V220 kernel by applying the 4 strip operations. Lowers cognitive load (algorithm body is copied) but inherits any V220 plumbing quirks unless every strip rule is applied carefully. Use when the V220 algorithm is the authoritative spec and no clean arch35 reference exists.
- **Fresh authoring** (gather_elements_v2 model): the arch35 kernel is written from scratch using the algorithm spec + arch35 native primitives (reg-based MicroAPI per CAND-A3A5-5, `WelfordUpdate` per OL-135, etc.) — V220 kernel kept in place untouched for backward compat. Cleaner end state; zero V220-leftover risk; reviewer-friendly. **Prefer this when the algorithm spec is independently authoritative AND the author can sustain the cost of writing two parallel kernels.**

Decision rule: if a `git diff master..FETCH_HEAD -- <op>/op_kernel/arch35/` reveals the arch35 files are NEW (not derived), apply audit but do NOT apply strip rules — they're already not present. If the diff reveals the arch35 files share most of the V220 body, apply the 4 strip rules above.

### Evidence

- ctc_loss_v3 (PR4778): **Derivation-strip path** — A3 = 831 lines → A5 arch35 = 830 lines (1-line strip of `impl/dav_c220/` include + `__CCE_AICORE__ == 220` conditional removal; rest identical). User-verified via `git show FETCH_HEAD:loss/ctc_loss_v3/op_kernel/arch35/ctc_loss_v3.h` vs `loss/ctc_loss_v3/op_kernel/ctc_loss_v3.h` on 2026-05-12.
- gather_elements_v2 (PR4778): **Fresh-authoring path** — 4 arch35 `.h` files (common + scalar + transpose + last_dim) are independently authored, no derivation from the master V220 `.cpp`. Audit grep returned 0 matches for `dav_c220`/`__CCE_AICORE__ == 220`/`ToFloat<`/`__NPU_ARCH__`. V220 master kept in place unchanged. 8/8 T1 bit-exact PASS on A5 pipeline-wiring verify (2026-05-12 kw-1).

### Cross-ref

- **W8** `ops_nn_layout/ops_nn_a5_artifact_layout.md` — what the arch35/<op>.h fits into
- **W11** `hardware/target/ascend950pr.md §Reg-based intrinsics restrictions` — full ToFloat<> rule
- **W9** OL-131 (cross-op router) — orthogonal host-side change for v2/v3-shared-aclnn ops
