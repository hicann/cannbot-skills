# Platform Known Bugs & Workarounds

Verified bugs in CANN/bisheng/hardware. Check this before debugging unexpected behavior.

> **`applies_to_backend:` tag** (P132): default is `ascendc`. Many PB
> entries are actually hardware-level bugs that BOTH backends hit (same
> Ascend910/950 silicon, same CANN runtime). When a PB entry describes
> a HARDWARE behavior (e.g. fp16/bf16 atomic precision, Tanh bimodal
> floor, transcendental ULP at output peak), it should be tagged
> `<!-- applies_to_backend: all -->` immediately after the `##` header
> so independent prototype briefs load it too. When a PB describes a
> bisheng/aclnn-compile-level bug that affects only AscendC-emitted IR,
> leave default (`ascendc`). See `OPERATIONAL_KNOWLEDGE.md` header note
> for full schema reference. independent prototype half landed `4532b461`.

## Tagging convention (P0aah, 2026-05-06)

New + refreshed PB entries use a structured tag block so the KB can be
mechanically refreshed when CANN/bisheng version bumps:

```markdown
- **applies_to**: `soc=<SOC>; cann=<version> (<innerversion>); bisheng=<ver>+<build_stamp>`
- **last_verified**: <YYYY-MM-DD>
- **status**: CONFIRMED | UNVERIFIED | FIXED-IN <release>
```

Why:
- Mechanical refresh: `grep cann=9.0.0 PLATFORM_BUGS.md` finds every entry
  pinned to current CANN; on version bump, run probes to refresh each.
- Multi-arch cross-validation: same finding on A5+A3 either gets two
  `applies_to` lines in one entry (when measurements match) OR a paired
  per-arch entry (when arches diverge).
- Staleness flag: `last_verified` > 90 days + CANN bumped → automation
  should surface the entry for re-test. Not yet wired; manual for now.

Older entries (PB-1 … PB-23) use free-text `Affected:` lines. They will
be migrated opportunistically when each is next refreshed; not a blocking
backfill (the prose pinning is still readable).

## CANN Bugs

### PB-1: Typed Kernel Entry Crash (CANN 9.0.0)

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: Error 507035 on kernel launch with typed entry points (e.g., `_fp32` suffix)
- **Affected**: CANN 9.0.0 with bisheng 2026-03-21
- **Workaround**: Use legacy untyped entry points (single dispatcher .cpp, cast inside kernel)
- **Status**: OPEN (not fixed in CANN 9.0.T501)
- **Evidence**: OPERATIONAL_KNOWLEDGE.md OL-16
### PB-2: TQue<VECIN,2> Data Corruption

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: 99.5% elements corrupted when using TQue with depth 2
- **Affected**: Ascend950PR, CANN 9.0.0
- **Workaround**: Use TQue<VECIN,4> (depth 4 works correctly)
- **Status**: OPEN
- **Evidence**: hardware/target/ascend950pr.md, E13 test data
- **V351 实证（2026-08-25，57_ParallelPolarizedSelfAttention_evo iter D1，Ascend950DT + CANN 9.2.0）**: A3 上验证过的 `TQue<...,2>` + InitBuffer depth-2 在 A5 上输出全零——O5 评测 45/50 FAIL 且 `matched_count==small_count`（57 ledger 行 9），57 是 PB-2 在 V351 的活实例。修复：3 个 TQue（xQue_/xOutQue_/wQue_）depth 2→4 + InitBuffer depth 参数同步 2→4。
- **UB 预算联动（57 ledger 行 11，iter D3）**: depth 翻倍直接放大 UB 占用；57 升 depth-4 后 11 个冻结 shape 的 InitBuffer 总和越 A5 248KB 可用上限。**升 depth 后必须立即重算全部 InitBuffer 字节和**；越界就减容 / 转 TQue<VECIN,1> / 重排 buffer 顺序。
- **判别注意（57-D3）**: 全 case mismatch 且输出全零时先确认 small_count>0，否则 `matched_count==small_count` 是伪迹。
### PB-3: NPU Device 0 Post-Reboot Failure

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: Error 507033 on device 0 after server reboot (2026-04-01)
- **Affected**: A5 server 198.51.100.35, device 0 only
- **Workaround**: Use devices 1-4, 7
- **Status**: Hardware issue, may require RMA

## Bisheng Compiler Bugs
### PB-4: bf16 Scalar Cast Failure

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (20_FusedRopeWithQkNormAndKvCacheUpdate). Do not downgrade.
- **Symptom**: `static_cast<float>(bf16_var)` produces wrong values in scalar context
- **Affected**: bisheng 2026-03-21 (CANN 9.0.0)
- **Workaround**: Use bit-manipulation helpers (`bf16_scalar_to_float`, `simt_to_float`, `simt_from_float`) OR SIMD `Cast()` intrinsic
- **Status**: OPEN
- **Evidence**: `tests/repro/bf16_cast_repro.cpp` (7 test cases), P-P27 pattern
  - 7_MoeGatingTopKSoftmax Phase C iter 1 (2026-04-17): `static_cast<bfloat16_t>(float)` caused compile error; fixed via SIMD Cast through fp32 scratch buffer (same workaround)
  - 14_AdaptiveInstanceNormalization2DBackward kw-1 iter 1 (2026-05-03): `(bfloat16_t)gw_partial` C-style cast in scalar emit context — bisheng "not support bf16 type cast". Fixed via `EmitScalarFromFloat` helper using SIMD Cast tensor-based path (1-element local tensor → Cast → GetValue). Confirms scalar bf16 cast remains broken on bisheng 2026-03-21 / CANN 9.0.0.
  - op#28 MultimodalRopePositionComputationWithGridBasedIndexing (2026-04-22): bit-manip helpers `simt_to_float<bfloat16_t>` (shift-by-16) and `simt_from_float<bfloat16_t>` (explicit IEEE RNE) compiled and ran cleanly inside `__simt_vf__` functions; PB-4 workaround is the durable path for bf16 scalar conversion in pure-SIMT contexts.
- **Detail**: Scalar bf16→float cast emits wrong instruction sequence. SIMD Cast() with `RoundMode::CAST_NONE` works fine.
### PB-16: `DataCopy(UB, TBuf<TPosition::A1>)` in pure-AIV kernel — silent miscompile → runtime illegal instruction

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (22_HybridAttentionMaskPreparation). Do not downgrade.
- **Symptom**: Kernel compiles with NO warnings or errors. At runtime, first `DataCopy` touching an L1-backed `LocalTensor` (TPosition A1) from a pure-AIV kernel faults with `aivec error exception, core id is 0, error code = 259, subErrType: 0x4, "Illegal instruction, which is usually caused by unaligned UUB addresses"`. Runtime status 507035. Error persists regardless of explicit `SetFlag<HardEvent::MTE3_MTE1>` sync — the fault is at the opcode level, not a sync bug.
- **Affected**: CANN 9.0.0 (innerversion V100R001C10SPC001B218), bisheng 2026-03-21, Ascend950PR_9589. Pure-AIV kernels (no Cube/Mmad). Both `DataCopy(UB, L1)` and `DataCopy(L1, UB)` overloads.
- **Misleading error message**: The "unaligned UUB" wording is not the actual root cause. Probe used 4 KB buffer, 32 B-aligned LocalTensors, element count a multiple of block — no alignment issue. The message is the generic err 259 text.
- **Workaround**: Do not use `TBuf<TPosition::A1>` (or A2/B1/B2/C1/C2/CO1/CO2) in a pure-AIV kernel on this toolchain. To use the hardware UB↔L1 channel documented in the 351x arch page, either (a) wait for CANN to expose a dedicated AIV-scope intrinsic (e.g. `CopyUbufToL1` variant), or (b) run the kernel as a mixed AIC+AIV task so a Cube context exists. For UB-budget-overflow optimization, use alternative axes: smaller tiles, fp16 intermediate with fp32 compute, split kernel into 2 launches.
- **Status**: OPEN. Escalate to CANN team asking whether a dedicated pure-AIV UB↔L1 intrinsic is planned. Re-probe after CANN version upgrade.
- **Evidence**: `src/skills/references/hardware/probe_findings/2026-04-21_Q_l1_scratch_op11_kind2.md` — full probe report with build logs + runtime stderr from 2 iterations (PipeBarrier + explicit MTE3→MTE1 SetFlag/WaitFlag variants both fail identically).
- **Severity**: HIGH. The silent-compile behavior is particularly dangerous: any optimizer that reads the public AscendC API ref (TPosition includes A1 as valid; TBuf accepts any TPosition; TBufPool docs explicitly say L1 is a managed resource) would reasonably conclude this is a legitimate optimization path and burn a full aog-kernel-worker iteration budget before discovering the runtime fault.
- **2026-04-21 update — CANN source cross-check confirms constraint**:
  - Low-level intrinsic `DataCopyUB2L1Impl((__cbuf__ T*)dst, (__ubuf__ T*)src, DataCopyParams)` exists at `ops-nn/matmul/common/cmct/tile/copy_ub_to_l1.h` + catlass parallels. This IS the functional UB→L1 DMA primitive.
  - Every single `TPosition::A1` usage across `ops-transformer / ops-nn / opbase / catlass / graph-autofusion` (grep -rlI) is inside matmul/Cube kernels. **Zero hits in pure-AIV kernels.**
  - Interpretation: the generic `DataCopy(LocalTensor<A1>, LocalTensor<UB>)` template does NOT route to `DataCopyUB2L1Impl` in pure-AIV compile context — it resolves to a no-op or placeholder opcode. The correct path is either (a) use `DataCopyUB2L1Impl` directly with memory-space-tagged raw pointers (but this is an internal API, not in public AscendC ref), or (b) run the kernel as a mixed AIC+AIV task so the Cube lowering is active.
  - Therefore: **PB-16 is not a bisheng bug in the "miscompile" sense — it's an undocumented constraint ("TPosition::A1 is Cube-context only"). Bisheng should warn/error, that part IS a bug. The runtime behavior is expected.**
### PB-17: UB aliasing cross-row V→MTE2 hazard — silent data corruption

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: When a fused op's ProcessRow() aliases two UB buffers (P-P65 pattern), if the alias target is (a) VEC-written near the end of ProcessRow AND (b) MTE2-written near the start of the next ProcessRow, with no explicit V→MTE2 sync, MTE2 overlaps the still-in-flight VEC writes → data corruption, precision FAIL. The kernel itself compiles and a single case may PASS — only batched runs surface the bug.
- **Affected**: Ascend950PR / CANN 9.0.0 / bisheng 2026-03-21 (expected to apply to all CANN versions; not probe-confirmed).
- **Root cause**: The AIV VEC pipe and MTE2 pipe run in parallel; AscendC queue's EnQue/DeQue only syncs MTE2→VEC, not V→MTE2. When the aliased physical slot is accessed by both pipes on different rows, the hardware provides no automatic barrier.
- **Workaround**: Insert `SetFlag<HardEvent::V_MTE2>` at the end of ProcessRow(), and `WaitFlag<HardEvent::V_MTE2>` at the start of the next row. Cost ~100-200 ns/row, which typically comes close to cancelling the savings from aliasing. If the sync cost offsets the benefit, the alias isn't worth it.
- **Status**: OPEN (architectural constraint, not a CANN bug).
- **Evidence**:
  - op#11 aog-fused-optimizer pilot Iter1 C5 attempt (2026-04-21): aliasing `fp16Buf_ ← tmpBuf_` → precision FAIL (6430/262144 int8 mismatch, max_abs_diff=147), REVERT; static dataflow audit identified a V→MTE2 hazard.
  - `workspace/dequantswigluquant/fused_analysis.md` §Iter 1 preliminary + §Handoff
- **Detection heuristic (for aog-fused-optimizer agents)**:
  - For each alias candidate, check the alias target's last write in the current row (VEC) and its first write in the next row (MTE2).
  - If there is no sync event between those two writes, this alias is a PB-17 risk.
  - Either add a sync (evaluate the net benefit) or drop this alias.
- **Cross-reference**: PB-47 (the cross-ITERATION variant — same V→MTE2 hazard class, but a per-tile buffer reloaded each chunk-loop iteration rather than two aliased buffers within `ProcessRow`; signature "every tile wrong except the last").
### PB-5: -O2 Required for NPU (No -O0)

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: Kernel may produce wrong results or crash with -O0 on NPU
- **Affected**: All NPU builds
- **Workaround**: Always use -O2 for NPU builds (-O0 only for CPU debug mode)
- **Status**: By design (bisheng optimizations required for correct codegen)
### PB-7: CANN merge_mix_obj.sh Crash with Empty --build-type

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: `make` fails at 95% with `Error 1` in `merge_mix_obj.sh` — `shift 2` fails
- **Root cause**: `cmake` invokes `merge_mix_obj.sh --build-type` without a value when `CMAKE_BUILD_TYPE` is unset. The bash `shift 2` fails because only 1 arg remains.
- **Affected**: CANN 9.0.0, AIV-only kernels (AIC dir empty, merge step still runs)
- **Workaround**: Always set `-DCMAKE_BUILD_TYPE=Release` in cmake invocation
- **Status**: OPEN (CANN build system bug)
- **Evidence**: MXFP4 project (2026-04-07), `merge_mix_obj.sh` line `shift 2` on `--build-type`

## Build Integration Issues
### PB-8: aclrtlaunch Stub Requires extern "C" Declaration

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: Linker error `undefined reference to aclrtlaunch_xxx(...)` when calling kernel from test code
- **Root cause**: Auto-generated `host_stub.cpp` exports functions as C symbols (no name mangling). Test code declaring them as C++ gets mangled names → linker mismatch.
- **Workaround**: Always use `extern "C" { uint32_t aclrtlaunch_xxx(...); }` in test code
- **Status**: By design (not a bug, but easy to forget)
- **Evidence**: MXFP4 test (2026-04-07)

## Operational Issues
### PB-6: Zombie Process Accumulation

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: Training/benchmark hangs, resource exhaustion after multiple runs
- **Affected**: Docker containers on A5 server
- **Workaround**: **Always restart container before every experiment**
- **Evidence**: 2280 zombies found after E13h

## Archived

### PB-7 (duplicate, line 68): Shared NPU Contention
- **Archived**: 2026-04-09. Reason: duplicate ID with PB-7 (line 43, merge_mix_obj). Content moved to PB-10.

---
### PB-10: Shared NPU Contention

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: Benchmark results vary wildly between runs
- **Affected**: A5 server (shared infrastructure)
- **Workaround**: Run `npu-smi info` before benchmarking, check for other processes
- **Evidence**: OPERATIONAL_KNOWLEDGE.md OL-15
### PB-9: UB-to-UB DataCopy Silent Data Corruption

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass ops (17_EmbeddingWithInitialLayernormBackward, 20_FusedRopeWithQkNormAndKvCacheUpdate). Do not downgrade.
- **Symptom**: `DataCopy(localDst, localSrc, count)` between two LocalTensors (both in UB) silently produces garbage data. No compile error, no runtime error — just wrong values. Discovered when LayerNorm V2 passed for norm_size ≤ 4096 (single tile) but produced ~20% mismatch with mean_abs_diff ~1.14 for norm_size > 4096 (multi-tile). Removing the UB-to-UB DataCopy and operating directly on the dequeued tensor fixed it completely.
- **Affected**: Ascend950PR, CANN 9.0.0
- **Workaround**: Never copy between LocalTensors using DataCopy. Instead:
  - Operate directly on the source tensor (e.g., run BinaryFoldReduceSum on the dequeued xd tensor)
  - Use VEC ops as a "copy": `Adds(dst, src, 0.0f, count)` if you must copy
  - Or use `Duplicate` to zero a buffer, then `Add(dst, dst, src, count)`
- **Status**: OPEN
- **Evidence**: LayerNorm V2 debugging session 2026-04-09; kernel/layernorm_kernel.h Pass 1 fix
  - op#30 NMS a3 ds kw-1 (2026-05-07): Used `Adds(dst, src, 0.0f, count)` identity copy from TQue<VECIN,1> dequeued tensors to persistent UB compute buffers, avoiding UB→UB DataCopy entirely. 31/31 bit-exact vs Python CPU reference. Confirms Adds-identity as canonical V220 workaround for VECIN→VECCALC data movement.
### PB-11: DataCopy to TBuf<VECCALC> — silent corruption on multi-iteration loops

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (17_EmbeddingWithInitialLayernormBackward). Do not downgrade.
- **Symptom**: `DataCopy(TBuf<VECCALC>::Get<T>(), GM_tensor, count)` with manual `SetFlag<MTE2_V>/WaitFlag<MTE2_V>` sync produces correct data on the first loop iteration but stale/corrupt data on subsequent iterations. Related to PB-9 (both are DataCopy corruption) but distinct mechanism: PB-9 is UB→UB; PB-11 is GM→VECCALC with manual sync in a loop.
- **Affected**: Ascend950PR, CANN 9.0.0
- **Workaround**: Use `TQue<VECIN, 1>` instead of `TBuf<VECCALC>` for any buffer that receives DataCopy from GM in a loop. The TQue's `AllocTensor/EnQue/DeQue/FreeTensor` pattern provides reliable MTE2→VEC synchronization. Single-iteration usage of TBuf<VECCALC> with DataCopy appears safe.
- **Status**: OPEN
- **Evidence**: DynamicQuant (#29) smooth_scales — 3 cases with row_size > TILE_SIZE failed (2.7%-22.6% mismatch, max_abs_diff=252) due to stale smooth_scales data on 2nd+ tile. First mismatch always at exact TILE_SIZE boundary. Fixed by switching to TQue<VECIN,1>. 42/42 PASS after fix.

---

## How to Add New Bugs

Append to the appropriate section with:
```
### PB-N: Short Description
- **Symptom**: What you observe
- **Affected**: Platform/version
- **Workaround**: How to work around it
- **Status**: OPEN/FIXED(version)/BY_DESIGN
- **Evidence**: Link to test/doc
```
### PB-12: SOC_VERSION Ascend910B2 causes 507035 on Ascend950PR hardware

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with three CANN-pass ops (17_EmbeddingWithInitialLayernormBackward, 20_FusedRopeWithQkNormAndKvCacheUpdate, 22_HybridAttentionMaskPreparation). Do not downgrade.
- **Status**: CONFIRMED (2026-04-16)
- **Symptom**: ALL kernels crash with 507035 (vector core exception, error 259 = illegal instruction) at PC offset 0x80. Build succeeds but runtime crashes every time.
- **Root cause**: `build_ascendc.py` defaults to `-v Ascend910B2` if no SOC_VERSION specified. Ascend910B2 binary contains instructions not supported on Ascend950PR AIV cores.
- **Fix**: Always pass `-v Ascend950PR_9589` when building on A5 hardware.
- **Impact**: This explains ALL previous unexplained 507035 crashes. Workers MUST always specify SOC_VERSION.
- **Evidence**: 14_AdaptiveInstanceNormalization2DBackward (4th attempt): all kernel variants crashed identically until SOC_VERSION corrected.
### PB-13: `Adds<int32_t>` buffer-to-buffer (dst ≠ src) silently corrupts output at N=1088

```yaml
applies_to:
  paradigm: ascendc
```
- **Status**: OPEN (observed 2026-04-19, needs next CANN version re-verify)
- **Affected**: CANN 9.0.0 (`/usr/local/Ascend/cann-9.0.0`), SOC Ascend950PR_9589, bisheng compiler as shipped with CANN 9.0.0 on A5 container `npu_dev3`. NOT yet retested on CANN 9.0.T501 or later.
- **Symptom**: `Adds<int32_t>(dst, src, 0, N)` where `dst != src` (buffer-to-buffer "copy via zero-add" pattern) silently produces wrong output data. Kernel compiles and runs without crash; downstream verification detects the corruption.
  - Specific observation: at count N=1088 (TOPK_CAP used by 9_TopKTopP V3.3 kind-2 rewrite), after `Adds<int32_t>` only the first entry appeared correct and the rest were zeroed/garbled (manifest: 49/50 cases fail precision with pattern "only gmax retained per row, rest become -inf").
  - **In-place** `Adds<int32_t>(buf, buf, 0, N)` works fine in the same build.
  - `Adds<float>` (same code shape, different dtype) works fine at same N — suggests bisheng codegen bug is `int32_t`-specific for this pattern.
- **Root cause**: Unconfirmed. Likely bisheng codegen quirk for the non-in-place pattern on int32. Minimal repro not yet created.
- **Workaround**: For copy-back of int32 buffers, use scalar loop (`for (int i=0; i<N; i++) dst.SetValue(i, src.GetValue(i));`) or `Cast int32→fp32, Adds<float>, Cast fp32→int32` roundtrip (only safe if all int32 values fit exactly in fp32 range — 24-bit signed range, int32 values > 2^24 would round).
- **Detection**: If you use `Adds<int32_t>` as a copy-back and see downstream results with all but the first element wrong, suspect this bug. Confirm by comparing to scalar loop baseline.
- **Perf impact**: On 9_TopKTopP V3.3 kind-2 rewrite, the scalar-loop workaround for int32 copy-back is a key reason V3.3 hit 0.191x sum-ratio (vs R3b 0.222x). R3b was measured on different NPU state (possibly different bisheng patch level) where this bug may not have manifested — hence R3b achieved full VEC copy-back. Re-verify on next CANN version to confirm whether R3b's VEC Adds<int32_t> approach now works.
- **Re-validation checklist when CANN updates** (per user directive 2026-04-19):
  1. Write minimal repro: `Adds<int32_t>(dst_buf, src_buf, 0, 1088)` with dst != src, check output bit-exact vs scalar loop
  2. Test at N ∈ {256, 512, 1024, 1088, 2048, 4096} to see if N-dependent
  3. If fixed: remove scalar-loop workaround from V3.3 kernel (or new kernels) and re-benchmark
- **Evidence**: 9_TopKTopP V3.3 kind-2 rewrite, Phase D iter 2 (2026-04-19). Kernel: `output/npukernelbench/src/kernels/9_TopKTopP/kernel/topktopp_kernel.h` (after V3.3 archival — TBD) workaround uses hybrid `Adds<float>` for val + scalar loop for idx. verification.json records the failed VEC Adds<int32_t> attempt. V3.2 t2 optimizer Opt4b earlier saw a related failure with same symptom — cross-confirmed on two independent attempts in same session.
### PB-15: Parallel NPU launches on same docker container produce cross-kernel state pollution

```yaml
applies_to:
  paradigm: ascendc
```
- **Status**: CONFIRMED (2026-04-21)
- **Affected**: A5 container `npu_dev3` with multiple `_<op>_ext.so` pybind modules launching on NPU 0, 1, 2, 3 concurrently via separate Python processes via `docker exec ... &` bash-backgrounded.
- **Symptom**: Reverify outputs show spurious `max_abs_diff ≈ 3e+38` (near fp32 inf) for kernels that return bf16 tensors in normal value ranges. Sequential re-runs of same (kernel, input-seed) pair show 0 drift. **False positive magnitude drift only under concurrent NPU launches.**
- **Root cause (hypothesis)**: aclrt/CANN stream state is container-global, not per-NPU. Concurrent launches across NPU 0..3 via independent Python processes of the same container share some runtime state — possibly allocator, HCCL, or operator proto registration — and one kernel's output buffer gets transiently corrupted while another kernel is mid-launch.
- **Workaround**: **Serialize all NPU reverify/test invocations within a single container.** If parallel test speedup is required, use 1 docker container per NPU (not just distinct `--device=npu:X` on same container). A second independent container per NPU isolates allocator state.
- **Detection**: Any time reverify scaffolding shows `ko_max > 1e10` for kernels producing bf16/fp16 outputs in normal ranges, suspect crosstalk. Re-run sequentially to verify. The 2026-04-21 drift-triage session was initially misdiagnosed as "real kernel bug" until sequential re-run showed the crosstalk signature.
- **Evidence**: 2026-04-21 batch reverify across 13 L2 PASS ops. Parallel-4-NPU re-run of op#9/19/21/26 showed `kernel_drift` for all 4 with `ko_max=3e+38` for op#19 case 5. Standalone re-run of each showed aggregate: op#9 spec_ambiguous, op#19 torch_npu_drift, op#21 **all_bit_exact 10/10**, op#26 spec_ambiguous. All "drift" signals from parallel run disappeared.
- **Related**: `src/scripts/batch_oracle_reverify.sh` header note (added 2026-04-21).
### PB-14: Branchless merge requires `CHUNK >= TOPK_CAP` invariant

```yaml
applies_to:
  paradigm: ascendc
```
- **Status**: CONFIRMED (2026-04-19)
- **Affected**: Any kernel using the R3b-style "branchless merge with tail-padded sentinel" pattern. See `output/npukernelbench/src/kernels/9_TopKTopP/kernel/topktopp_kernel.h` Phase 1 merge.
- **Symptom**: If `CHUNK < TOPK_CAP`, the branchless merge reads past `sortValOut[CHUNK..TOPK_CAP)` which is OOB. Result: most cases FAIL precision with `max_abs_diff=3.4e38, mean_abs_diff=inf`.
- **Root cause**: Branchless merge is designed under the invariant that both inputs (existing top buffer and new chunk's sorted output) have at least TOPK_CAP usable slots (with sentinel padding in unused slots). If the chunk's `sortValOut` is only `CHUNK` elements, reading `CHUNK..TOPK_CAP` goes into whatever's after the chunk buffer in UB — typically garbage values that compare greater than real data.
- **Fix**: Enforce `CHUNK >= TOPK_CAP` via `static_assert(CHUNK >= TOPK_CAP, "branchless merge requires CHUNK >= TOPK_CAP")` in tiling.h. If CHUNK needs to be smaller for UB reasons, use conditional-merge variant instead (slower but safe).
- **Evidence**: 9_TopKTopP V3.3 kind-2 Phase D iter 1 (2026-04-19) — CHUNK=1024 < TOPK_CAP=1088 caused 49/50 FAIL. Fixed CHUNK→2048 + static_assert → 50/50 PASS.
- **Related**: P-P59 Layer 1 canonical sketch. Add this invariant to any canonical sketch that uses branchless merge.
### PB-18: CANN install path drift on npu_dev3 — `/usr/local/Ascend/cann-9.0.0` empty, real install at `/data/cann_b103/cann-9.0.0`

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (17_EmbeddingWithInitialLayernormBackward). Do not downgrade.
- **Status**: CONFIRMED (2026-04-22, op#28)
- **Affected**: A5 container `npu_dev3` on host `198.51.100.35` as of 2026-04-22. Likely recurs after any host-side CANN reinstall that moves the working install to a non-standard path.
- **Symptom (CMake)**:
  ```
  CMake Error at CMakeLists.txt:19 (message):
    ascendc_kernel_cmake does not exist, please check whether the cann package is installed.
  ```
  at `cmake -S _autogen_cmake -B build -DSOC_VERSION=... -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0`.
- **Symptom (runtime)**:
  ```
  ImportError: libhccl.so: cannot open shared object file: No such file or directory
  ...
  RuntimeError: Failed to load the backend extension: torch_npu
  ```
  when running `utils/verification_ascendc.py` / `utils/performance.py` without properly sourcing the correct `set_env.sh`.
- **Root cause**: `/usr/local/Ascend/cann-9.0.0/` and `/usr/local/Ascend/cann-9.0.T501/` are empty directories. `/usr/local/Ascend/cann → cann-9.0.T501` broken symlink. `/usr/local/Ascend/ascend-toolkit/latest → /usr/local/Ascend/cann` also dead. The real working install is at `/data/cann_b103/cann-9.0.0/` (contains `x86_64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake`, `compiler/`, `opp/`, `set_env.sh`, etc.). `/root/.bashrc` tries to `source /usr/local/Ascend/ascend-toolkit/set_env.sh` — that path is dead, default shell env is wrong.
- **Additional gotcha — libhccl path**: Even after `source /data/cann_b103/cann-9.0.0/set_env.sh`, `LD_LIBRARY_PATH` only includes `/data/cann_b103/cann-9.0.0/lib64` but `libhccl.so` lives at `/data/cann_b103/cann-9.0.0/x86_64-linux/lib64/libhccl.so`. Must manually `export LD_LIBRARY_PATH=/data/cann_b103/cann-9.0.0/x86_64-linux/lib64:$LD_LIBRARY_PATH`.
- **Workaround**:
  - **Build**: update `workspace/.ascendc_env` → `CANN_PATH=/data/cann_b103/cann-9.0.0`. `deploy_to_a5.sh` (2026-04-22 patched) picks up via `$ASCEND_HOME_PATH` / `$LD_LIBRARY_PATH` exports.
  - **Verification / perf runs**: prefix with `bash -lc 'source /data/cann_b103/cann-9.0.0/set_env.sh >/dev/null; export LD_LIBRARY_PATH=/data/cann_b103/cann-9.0.0/x86_64-linux/lib64:$LD_LIBRARY_PATH; ...'`.
- **Detection**: `find / -type d -name ascendc_kernel_cmake 2>/dev/null` in container — if only `/data/` hits, drift confirmed. Or `ls /usr/local/Ascend/cann-9.0.0/` empty → confirmed.
- **Long-term fix**: `/aog-preflight` should probe `$CANN_PATH/x86_64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake` before writing `.ascendc_env`; scan `/data/` + `/usr/local/Ascend/` for a working install when probe fails. Orchestrator's build step could add the same pre-check.
- **Status**: OPEN (environment drift, not a code bug). May auto-resolve on next container rebuild.
- **Evidence**: op#28 MultimodalRopePositionComputationWithGridBasedIndexing Phase C iter 1 (2026-04-22) — CMake fail on orchestrator's default `/usr/local/Ascend/cann-9.0.0` path. Worker traced via `find` to `/data/cann_b103/cann-9.0.0`. Updated `.ascendc_env` → iter 2 configure succeeded. Phase D: `torch_npu` import failed with `libhccl.so: cannot open` until explicit `LD_LIBRARY_PATH` fix applied (orchestrator independent perf re-run).
### PB-19: AscendC VEC `Sin()` / `Cos()` lack Payne-Hanek argument reduction — ±inf / huge-error on |x| ≥ 1e10

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (20_FusedRopeWithQkNormAndKvCacheUpdate). Do not downgrade.
- **Status**: CONFIRMED (2026-04-22, op#16)
- **Affected**: AscendC SIMD `Sin<T, false>(dst, src, tmp, count)` / `Cos<T, false>(...)` on Ascend950PR / CANN 9.0.0 / bisheng 2026-03-21. Both `false` and `true` (HIGH_PRECISION) modes exhibit the same limit.
- **Symptom**: For `|x| ≥ 1e10`, the primitive produces numerically huge (~1e21 at |x|=1e10) or ±inf values. Reference `torch.sin(x)` on NPU (dispatches through CANN aclnnSin) produces finite, correct values at the same inputs.
- **Root cause**: AscendC VEC trig primitives use a fixed-precision polynomial evaluation after a simple modulo-2π argument reduction that loses precision at large magnitudes. aclnnSin implements proper Payne-Hanek reduction (multi-limb multiply by 2/π with catastrophic-cancellation-resistant carry chain), which our VEC primitive does not.
- **Affected usage**: any kernel using AscendC VEC `Sin()`/`Cos()`/`Tan()` on inputs not pre-constrained to `[-π, π]` (or similar small range).
- **Workaround**:
  - **Domain-scoped ops** (RoPE, sinusoidal PE, etc.): theta values are naturally in `[-π, π]` — primitive works correctly, no action needed. Mark adversarial out-of-domain edge_dataset cases as PARTIAL honestly rather than waiving.
  - **Domain-unconstrained ops**: implement Payne-Hanek reduction in kernel (~50-100 LoC, +compute), or call torch.sin via pybind pre-kernel (not CANN-delegation since it's a different compute phase).
- **Detection**: edge_dataset cases with `dist_large_mag` distribution (|x|~1e29) will expose this. If kernel uses trig primitives AND operational domain is bounded, document as "out-of-domain limit" rather than attempting to fix.
- **Evidence**: op#16 Batched2DRopePositionEncodingBackward edge_dataset 29/31 PASS; failing cases 21/22 (`dist_large_mag_seed{0,1}`, |t|~1e29): kernel output ±inf, torch.sin output finite. 50/50 benchmark cases (realistic RoPE domain) all PASS — confirms primitive is correct in operational range.
- **Related**: EC-35 (AIV libm split) is orthogonal — that's about scalar `cosf`/`sinf` unavailability in SIMT; PB-19 is about SIMD `Cos()`/`Sin()` range limit. Both constrain how trig is implemented on A5.
### PB-20: `GlobalTensor<T>::SetValue(idx, val)` is silent no-op on Ascend950PR

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (20_FusedRopeWithQkNormAndKvCacheUpdate). Do not downgrade.
- **Status**: CONFIRMED (2026-04-22 op#5; 2026-04-24 cross-checked against CANN source)
- **Affected**: Ascend950PR / CANN 9.0.0 / bisheng 2026-03-21. SIMT-AIV kernel context. Only `GlobalTensor<T>::SetValue(uint64_t idx, T val)` (scalar GM write at indexed position) — bulk `DataCopy(GM, UB)` works fine.
- **Symptom**: kernel compiles, runs, returns success. Output GM tensor contains uninitialized values (`torch::empty` garbage). No diagnostic output. Precision FAIL on every case.
- **Confirmation against CANN source (2026-04-24, op#3)**: CANN's own `cann/ops-nn/optim/advance_step/op_kernel/advance_step.h` writes every output via `GlobalTensor<int64_t>::SetValue(idx, val)` — exactly the broken pattern. Crucially: that op's `op_host/advance_step_def.cpp` only registers `ascend910b` + `ascend910_93` AICore configs; **A5 is excluded specifically because the kernel pattern doesn't work there**. So PB-20 is not a worker quirk — it's a fundamental CANN-vs-A5 SIMT-AIV write-path mismatch that CANN itself works around by simply not shipping A5 binaries.
- **Workaround — context-dependent decision tree** (refined 2026-04-30 op#22 Nonzero):
  - **SIMT VF kernel** (`Simt::VF_CALL<f>(Simt::Dim3{N}, ...)` wrapped functions, `LAUNCH_BOUND(K)` annotated): use raw `__gm__ T*` pointer indirect writes via `reinterpret_cast<__gm__ T*>(GM_ADDR_arg)` then `gm[i] = val;`. Different opcode path (scalar pipe direct GM access). Reference templates: `output/npukernelbench/src/kernels/19_IndexPut/`, `output/npukernelbench/src/kernels/3_AdvanceStepFlashattn/`.
  - **Pure-AIV class kernel** (plain `extern "C" __global__ __aicore__ void f(...) { KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); class.Init(...).Process(); }` — the multi-core SIMD style used by MoeInitRouting / FusedAddRmsnorm): **DataCopy from UB LocalTensor to GM** via MTE3 pipe. Empirical: in this context, BOTH `GlobalTensor::SetValue` AND raw `__gm__ p[i]=v` silently fail (op#22 sentinel test, 2026-04-30, 8 hypotheses by kw-2 + minimal-sentinel reproduction by orchestrator). Pattern:
    ```cpp
    GlobalTensor<int32_t> g;  g.SetGlobalBuffer(...);
    LocalTensor<int32_t> ub = ub_buf.Get<int32_t>();
    ub.SetValue(0, my_val);              // UB scalar SetValue OK
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopy(g[off], ub, count);          // UB→GM via MTE3 — WORKS
    ```
  - Reference template for pure-AIV class kernel: `output/npukernelbench/src/kernels/5_MoeInitRouting/kernel/moeinitrouting_kernel.h` — every GM write goes through `DataCopy(...)` or `AtomicAdd(...)`, NEVER `GlobalTensor::SetValue`.
- **Implication for aog-kernel-worker / optimizer**: when porting from CANN reference kernels (especially scatter / in-place ops), grep the CANN source for `SetValue(` calls on `GlobalTensor` — every one of those needs to be rewritten per the kernel-context above. **Do not** assume "CANN does it this way, so we can too". When designing a new multi-core kernel, prefer the pure-AIV class pattern AND use DataCopy(UB→GM) for GM writes.
- **Implication for benchmark reference**: if a benchmark op's CANN kernel uses `SetValue` on GM, attempting to build that kernel from source against A5 will produce a binary that compiles but silently produces wrong output. This is "OL-68 Case B" — see OL-68 sub-case taxonomy.
- **Evidence**:
  - op#5 MoeInitRouting probe (2026-04-22) — sort kernel Pass 3 had this pattern → 0/50 PASS, all output uninitialized. After switching to UB TBuf staging + `DataCopy UB→GM` + `SetFlag/WaitFlag<HardEvent::S_MTE3>` sync (or alternatively raw `__gm__` pointer writes), → 53/53 PASS.
  - op#3 AdvanceStepFlashattn (2026-04-24) — used raw `__gm__ int64_t*` pattern from the start (per IndexPut precedent), 50/50 + 28/28 PASS.
  - op#22 Nonzero V2 (2026-04-30) — kw-2 attempted multi-core SIMD class kernel, K=0 silent-fail on all 50 cases despite 8 hypotheses (rename ws→ws_buf to dodge HAVE_WORKSPACE codegen, raw `__gm__` writes, GlobalTensor::SetValue, MTE2→S sync, fixed nblk=56, minimal sentinel `SetValue(0,42)`, bypass class entirely). Orchestrator reproduced K=0 with minimal sentinel kernel; switching to `DataCopy(UB→GM)` immediately worked. Confirms raw `__gm__` workaround does NOT generalize from SIMT VF to pure-AIV class context.
  - op#22 Nonzero V4 (2026-04-30, second probe-before-edit success) — kw-5 wrote `workspace/22_Nonzero/probes/gathermask_probe/` BEFORE attempting V4 SIMD-emit kernel, verified `GatherMask + ReinterpretCast<uint32_t>(CompareScalar bitmask)` API contract on a5 in ~3 minutes. V4 then built first try, no compile-fix iters, no precision-fix iters, 50/50 + 10/10 + det 50/50 PASS. Second consecutive op (after V2 sentinel) where probe-first prevented worker churn. Recommendation: when an API has thin documentation (single-line catalog entry, zero codebase precedent), run a 1-block probe before committing kernel architecture.
  - group_norm_silu_quant (2026-05-13, A5 fused GroupNorm+SiLU+Quant port from A3 aclnn) — iter 1→2 attempted `const_cast<__gm__ T*>(gmMean_.GetPhyAddr())` + `mean_ptr[idx] = mean_t` for per-(n,g) scalar mean/rstd outputs from pure-AIV class kernel. Compiled clean, ran clean, mean/rstd contained uninitialized garbage (max_abs_diff up to 2.09e+28 / FP_MAX sentinel on case 6 bf16). Exactly the second bullet of the decision tree above. Fix (iter 3→4): switched to `DataCopy(gmMean_[idx*16], ub_block, 16)` with 16x-padded GM workspace (each `(n,g)` element gets its own 32B-aligned slot), pybind extracts `mean_ws.select(2, 0).contiguous()` post-kernel → mean/rstd bit-exact (0.0 diff) on all 8 cases. **Sub-pattern for small per-work-unit scalar outputs**: see CAND-PB20-GMPAD candidate — wraps the `DataCopy(UB→GM)` workaround with the 16x-pad allocator so scalar-per-(n,g) outputs satisfy 32B alignment without inter-AIV races.

---
### PB-21: `PipeBarrier<PIPE_ALL>()` between TBuf-resident MTE2 load and V compute → silent crash 507015 (V220-confirmed)

```yaml
applies_to:
  paradigm: ascendc
```
- **Severity**: CRITICAL
- **Status**: OPEN (CANN 9.0.0)
- **Affected hardware**: Atlas A3 (V220 / Ascend910_93) confirmed. A5 / Ascend950PR/950DT (V351) — **CONFIRMED**（2026-08-25，55_OutlookAttention kw-5，Ascend950DT + CANN 9.2.0；症状形态不同，见下）。
- **Symptom**: kernel using a manual TBuf pipeline pattern (`TBuf<VECCALC> bufA_, bufB_;` + `DataCopyPad → PipeBarrier<PIPE_ALL>() → VEC compute`) crashes at runtime with `aclrtLaunchKernel` returning error code **507015**. No exception, no Python traceback — kernel silently terminates after launch and the host blocks at the next sync. aicpu / aiv logs show MTE2→V handoff aborted mid-pipeline.
- **Trigger pattern**: TBuf (NOT TQue) pipeline mixed with `PipeBarrier<PIPE_ALL>()` as the synchronization primitive between MTE2 load and V compute. The bug fires reliably when ALL of:
  - All loads/computes run on a single `TBuf<VECCALC>` (no `TQue<VECIN>` queue rotation)
  - Sync between MTE2 (DataCopy) and V (Cast/Mul/Add) is via `PipeBarrier<PIPE_ALL>()` rather than explicit `SetFlag<HardEvent::MTE2_V>(eventId) + WaitFlag<HardEvent::MTE2_V>(eventId)`
  - Loop body has ≥ 2 iterations of MTE2→V on the same TBuf
- **Fix**: replace `PipeBarrier<PIPE_ALL>()` with explicit `SetFlag<HardEvent::MTE2_V>(eventId) + WaitFlag<HardEvent::MTE2_V>(eventId)` (and analogous `V_MTE3` between V compute and DataCopy back to GM). The event ID is fetched via `uint16_t ev = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);` once per kernel and reused across loop iterations. Reference template: `patterns/domains/platform_compat.md` §"Manual TBuf pipeline with explicit event sync" (P-P70).
- **Why it happens (hypothesis, unconfirmed)**: V220 `PipeBarrier<PIPE_ALL>()` semantics on TBuf (vs TQue) appear to skip MTE2→V completion guarantees. TQue `<VECIN, depth=2>` has hardware-managed queue rotation that includes the implicit barrier; TBuf does not. CANN docs do not call this out explicitly; CANN's own kernels using TBuf consistently use explicit event sync.
- **Decision rule** (when to use TBuf+manual sync vs TQue auto-rotation): see OL-94.
- **Evidence**: op#27 `27_MultiMaskAttentionAggregation` a3 V220 cold-start (2026-04-28) — worker initial impl used `TBuf + PipeBarrier<PIPE_ALL>()` per natural CANN-style port → silent crash 507015 across all cases. Five compile/precision iters wasted before probe identified the sync primitive as the culprit. Switched to explicit `SetFlag<HardEvent::MTE2_V>/WaitFlag<HardEvent::MTE2_V>` → 50/50 PASS, det 100/100. Probe report: `output/npukernelbench-a3/src/kernels/27_MultiMaskAttentionAggregation/probe_report.md` (a3 PR #2 v2 archive).
- **V351 实证（2026-08-25，55_OutlookAttention iter kw-5，Ascend950DT + CANN 9.2.0）**: 55 的 8 个 kernel 全部是纯 `TBuf<VECCALC>` + `PipeBarrier<PIPE_ALL>()`（共 **57 个 PIPE_ALL 全错**、零 SetFlag/WaitFlag），本卡教科书形态（55 ledger 行 60-61）。V351 上症状**不是 V220 的 507015 崩溃，而是静默"有限垃圾"**（finite garbage）：V351 严格 pipe 分离（EC-81），PIPE_ALL 在 TBuf 上不排序 MTE2→V / V→MTE3 / V→MTE2，VEC 读 stale/partial UB，全部 case MERE 133-338、matched_ratio ~1e-5、无 NaN（ledger 行 55-58）。同代码在 A3(V220) 仅靠隐式跨 pipe 转发跑对（行 63-64）。修复（行 64-68）：逐 kernel 在 Init 取一次事件 id，在每个跨 pipe 边界插显式 SetFlag/WaitFlag（V_MTE2 在覆盖读过 buffer 的 GM→UB 之前、MTE2_V 在 GM→UB 之后 VEC 读之前、V_MTE3 在 UB→GM 写回之前）；**57 个 PIPE_ALL 全部保留**（V→V drain 仍需，EC-77），只加事件配对、不改算术。**非机械修复**：摆位须按各 kernel 实际读写顺序逐个分析（55-kw5 与 57-D4 各烧一整轮）。另注意：修完 kw-5 后 55 仍 all-cases-wrong（行 71）——事件配对修复是必要非充分。判别技巧（55-kw4）：host 侧输出预填 NaN——全 NaN=kernel 未执行、有限垃圾=执行了但算错。详见 `okf/runbooks/field-notes/precision/v351-pipe-all-tbuf-stale-001.md`。
- **Cross-reference**: F-P4 (PipeBarrier alignment) covers a different PipeBarrier failure mode (alignment); PB-21 is specifically the TBuf+PIPE_ALL combo. PB-9 (UB→UB DataCopy on V220) is another V220-only sync nuance. OL-94 has the broader "when to pick which sync mechanism" decision rule.

---
### PB-23: SIMD binary-scalar VEC ops (Divs/Muls/Adds/Subs/Sub/Add/Mul/Div) reject int32, int16, int8, bf16 — supported dtype list verified

```yaml
applies_to:
  paradigm: ascendc
```
- **Severity**: HIGH (silent compile-time rejection — bisheng `static_assert` halts the build with no fallback)
- **Status**: CONFIRMED 2026-04-30 op#22 Nonzero V4 kw-5 build iter 1
- **Affected**: Ascend950PR / CANN 9.0.0 b103 / bisheng 2026-03-21. Likely V220 too (not yet checked).
- **Symptom**: Building a kernel that uses `AscendC::Divs<int32_t>(dst, src, scalar, count)` (or any of `Muls / Adds / Subs / Sub / Add / Mul / Div` with int32) fails at compile with the bisheng error: `static_assert(SupportType<T, half, float, int64_t, uint64_t, complex32, complex64>())`. The supported dtype set for these binary-scalar VEC ops is exactly `{half, float, int64_t, uint64_t, complex32, complex64}` — int32 is NOT in the list, nor are int16, int8, or bf16 (bf16 must Cast→fp32 first per PB-4).
- **Workaround (canonical)**: for index arithmetic, use `int64_t` for the SIMD vector buffer dtype. The supported-list includes `int64_t` and is verified working on a5 for Adds/Muls/Sub. If a 32-bit integer path is genuinely needed (e.g. memory-pressure on tile sizes), the upstream operation must use a Cast to fp32 first, then back to int via floor/round Cast — but be aware of fp32's 24-bit mantissa precision limit (indices > 2^24 = 16M lose accuracy).
- **Workaround for index ops specifically**: in V4 GatherMask + N-D-decode flows (P-P80), promote the per-element packed positions from int32 (output of GatherMask) to int64 via Cast<int64,int32> immediately, then do all `Divs<int64>`, `Muls<int64>`, `Sub<int64>` on the int64 buffer. This is the verified pattern from op#22 V4 kw-5.
- **Why this matters for skill briefs**: when designing SIMD vector decode for nonzero / scatter / index ops, do NOT assume int32 is supported. The static_assert is silent until you instantiate the template with int32, then halts the build. **Workers grepping ASCENDC_API_CATALOG.md for "Divs" find no dtype list and assume general support — file the dtype list in the catalog explicitly.**
- **Evidence**: op#22 22_Nonzero V4 kw-5 (2026-04-30) — initial V4 design used int32 buffers for `pos_local` (GatherMask output) and chained `Divs<int32_t>` / `Muls<int32_t>` / `Sub<int32_t>` for N-D index decode. Build failed with the static_assert. Single-fix iter switched all 5 SIMD-decode buffers to int64 → build PASS first try, V4 50/50 PASS + det 50/50.

---
### PB-22: MTE2 DataCopy on V220 CANN 9.0.0 has 32-byte (8 fp32 element) transfer limit per destination TBuf [V220]

```yaml
applies_to:
  paradigm: ascendc
```
- **Severity**: CRITICAL (silent data corruption — elements beyond index 7 are zero without error)
- **Status**: CONFIRMED 2026-05-01 op#31 IOU pp-1 probe (13 iterations, V220 CANN 9.0.0)
- **Symptom**: `DataCopy(dst_TBuf, gm, count)` only writes the first 32 bytes (8 fp32 elements) into the destination TBuf. Elements beyond index 7 are always zero, regardless of count parameter (40, 64), chunking (single call vs serialized), or TBuf position (VECCALC vs VECIN).
- **Fix**: Use `TQue<QuePosition::VECIN, depth>` for input streaming instead of `TBuf + DataCopy`. TQue's EnQue/DeQue rotation uses a different MTE2 path. Or deinterleave in pybind host-side and pass column vectors directly.
- **Evidence**: op#31 IOU a3 V220 pp-1 (2026-05-01) — 13 probe iterations confirmed across multiple TBuf positions and chunking strategies. A5 working kernel uses TQue<VECIN> successfully.
  - op#30 NMS a3 ds kw-1 (2026-05-07): Used `TQue<VECIN,1>` for streaming box coordinate and score inputs (fp32), avoiding TBuf+DataCopy 32-byte limit. 31/31 bit-exact vs Python CPU reference. Confirms TQue<VECIN> as canonical V220 input path when element count per tile exceeds 8 fp32.
  - fatrelu_mul port_a3_to_a5 kw-1 (2026-05-17, **V351 sub-block confirmation**): case 7 has lastDim=2 → d=1 → DataCopyPad blockLen = 1×4 = 4 bytes (well below 32B alignment boundary). Both input (`DataCopyPadExtParams<float>{false,0,0,0}`) and output paths handled the sub-block transfer correctly — 0.0 max_abs_diff vs A3. Validates that **V351 (A5) DataCopyPad handles unaligned blockLen natively**; the 32-byte-limit failure mode in this PB-22 entry is V220-specific (the symptom does not transfer to V351 even on extremely small d=1 tiles).
- **Cross-ref**: OL-94 (TQue vs TBuf decision), OL-124 (TBuf→MTE3 coherence), PB-9 (UB→UB DataCopy), PB-11 (multi-iteration DataCopy)
### PB-24: V220 TBuf `GetValue` returns sequential-position values (not stored data) when interleaved with TQue `CopyTile` operations [V220]

```yaml
applies_to:
  paradigm: ascendc
```
- **Severity**: CRITICAL (silent data corruption — kernel produces wrong output with no error)
- **Status**: CONFIRMED 2026-05-02 op#18 Index pp-1 (12 bisection iterations, V220 CANN 9.0.0)
- **Symptom**: When scalar values are read from a `TBuf<VECCALC>` via `GetValue()` AND the same kernel also uses TQue `EnQue/DeQue` (CopyTile) in interleaved fashion, the TBuf GetValue returns the **sequential position index** (0, 1, 2, ...) instead of the actual stored value. All GM scalar read paths are affected: `GlobalTensor::GetValue`, raw `__gm__` pointer dereference, TBuf `DataCopy` with MTE2_V sync.
- **Fix**: Read ALL TBuf values into a local C++ stack array BEFORE any TQue CopyTile operations begin. This eliminates the interleaving that triggers the corruption.
  ```cpp
  // Read all indices before any TQue operations
  int32_t idx_buf[MAX_INDICES];
  for (int i = 0; i < n_indices; ++i)
      idx_buf[i] = idxBuf_.GetValue<int32_t>(i);
  // NOW start TQue CopyTile operations using idx_buf[]
  ```
- **Evidence**: op#18 Index DS kw-1 (2026-05-02): 9 iterations tested all GM read paths — all returned sequential positions. pp-1 (2026-05-02): 12 bisection iterations confirmed the interleaving root cause. kw-2 applied the "read-all-before-TQue" fix → 41/41 PASS (bit-exact, MERE=0, MARE=0), perf 10.78x vs CANN.
- **Other instances (predicted)**: any V220 kernel that mixes TBuf scalar reads with TQue pipeline operations, especially index/gather/scatter ops where indices are loaded via TBuf.
- **Cross-ref**: PB-22 (MTE2 DataCopy 32B limit), OL-124 (TBuf→MTE3 coherence), OL-123 (V220 API gaps).
### PB-26: AscendC `Tanh<fp32>` primitive — bimodal precision floor; catastrophic small-x identity loss

```yaml
applies_to:
  paradigm: ascendc
```
- **applies_to**: `soc=Ascend950PR_9579; cann=9.0.0 (V100R001C25B046); bisheng=15.0.5+2026-04-13`
- **applies_to**: `soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=15.0.5+2026-01-28`
- **last_verified**: 2026-05-07 (A3 cross-arch confirmation)
- **status**: CONFIRMED on A5 AND A3 — bimodal Tanh floor is **chip-family-wide**, not arch-specific. Bisheng Tanh polynomial doesn't preserve `tanh(x)≈x` near zero on either arch.
- **Severity**: HIGH for kernels that call `Tanh` on inputs near zero (residual norms, KV cache normalize). LOW for transcendental kernels that pre-amplify away from zero (GELU's `0.7978·(x + 0.0447·x³)` pre-amp masks the failure mode).
- **Symptom (measured, both A5 and A3)**: `AscendC::Tanh<float>` exhibits a **bimodal floor**, not the uniform ~1-ULP ceiling previously inferred from end-to-end op#1 GELU measurement.
  - **A5 `|x| ≥ 0.1`**: clean 2-ULP uniform floor across mid / transition / plateau. max_abs_err 1.37e-7. **Saturation band is bit-clean.**
  - **A5 `|x| < 0.1`**: catastrophic. Worst case 1599 ULP at x≈1.7e-4, blowing up to 2.7M ULP at x=1e-7. CPU `numpy.tanh(1e-7)` returns exactly `1e-7` to fp32; NPU returns `1.192e-7`.
  - **A3 `|x| ≥ 0.1`**: max 4 ULP (slightly looser than A5 but same class). Transition band (3..5): max 2 ULP, mean 0.29.
  - **A3 `|x| < 0.1`**: same failure-mode-class — up to 906 ULP in band [1e-4, 0.1].
  - **Joint conclusion**: bimodal floor is bisheng-Tanh-polynomial wide; saturation band is consistently bit-clean (≤4 ULP); near-zero band consistently fails (small-x identity loss). The original OL-103 "saturation is the worst case" framing was wrong on both arches.
- **Root cause hypothesis (unconfirmed)**: bisheng's `Tanh` polynomial implementation lacks a small-x bypass / Taylor-series fallback that established public math libraries (Cephes, fdlibm, libm) use. Abs-err is uniformly tiny (≤1e-7) but expected-output magnitude is ALSO near zero, so ULP measurement at near-0 outputs explodes.
- **Bisheng-version sensitivity (2026-05-07 cross-arch finding)**: A5 bisheng `2026-04-13` is **strictly worse** at small-x (2.7M ULP) than A3 bisheng `2026-01-28` (906 ULP), even though A5 is the newer chip. The newer bisheng build appears to have regressed the Tanh polynomial near zero. Worth re-running A5 probe on next bisheng release to see if this is monotonic. Bisheng build stamp is the load-bearing version field — driver / CANN minor are not.
- **A3-side build note (V220-specific, separate from precision claim)**: V220 (Ascend910_9382 arch) does NOT honor `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` — using it on V220 causes `RegisterAscendBinary 107000`. Probe must use single-block ctypes host launch instead. This is a host-side caller pattern, not a kernel-side precision issue.
- **Workaround**:
  - For GELU and Tanh-class kernels — use sigmoid-form rewrite: `0.5·x·(1+tanh(y)) = x / (1 + exp(-2y))` via `Mul + Axpy + Muls + Exp + Adds + Div`. Confirmed by CANN's own arch35 GELU source (`ops-nn/activation/gelu/op_kernel/arch35/gelu_dag.h` — does NOT use the `Tanh` primitive). See P-P88.
  - For other Tanh-using kernels (residual / KV-cache near-zero paths) — audit input domain. If inputs span `|x| < 0.1`, consider direct `Exp+Add+Div` Cephes-form `tanh(y) = 1 - 2/(exp(2y)+1)` or precondition to amplify away from zero.
- **Detection**: edge_dataset cases with `dist_small_mag` distribution (|x|~1e-30 to 1e-20) or `dist_denormal` will expose this when input crosses through `|x| < 0.1`. If kernel uses `Tanh(...)` AND operational domain includes near-zero, document explicitly.
- **Evidence**:
  - A5 isolated probe `workspace/_probes/tanh_sigmoid_precision_a5_cann9.0.0/PROBE_REPORT.md` (P0aae, agent a57355497e0a0e575, 2026-05-06). Sweep 14,349 fp32 points across `[-10, 10]`. Histogram: `0`→7240, `1`→4180, `2`→795, `4`→792, `8`→599, `16`→377, `≥32`→366. Worst 5 inputs all in `|x|<1e-4` band.
  - A3 isolated probe (DS-side, 2026-05-07) `workspace/_probes/tanh_sigmoid_precision_a3/PROBE_REPORT.md`. 66% bit-exact, 96% within 1 ULP overall. `|x| ∈ [1e-4, 0.1]` band peaks at 906 ULP. Build note: V220 (arch35-only) does not honor `KERNEL_TASK_TYPE_DEFAULT` — use single-block ctypes launch instead (host-side caller pattern, not relevant to the precision claim itself).
- **Cross-reference**:
  - OL-103 §Refined-statement (still mentions inferred Tanh ceiling — to be softened to point at PB-24 once DS A3 data lands; held per user direction "wait for both arches before shipping KB edits").
  - P-P88 (Cephes-form rewrite recommendation; CANN-source-confirmed for GELU; small-x failure mode is the real algorithmic reason, not the previously-imagined saturation cancellation).
  - P0aac (ar_brief Phase R-B step 5 — researcher mandated to consult public math-library literature before declaring "no vendor strategy known"). Closes the harness blind spot that caused op#1 1_GELU iterations to miss this finding.

---
### PB-27: AscendC `Sigmoid<fp32>` primitive — uniform 2-ULP floor (slight correction to OL-103 "1-ULP" claim)

```yaml
applies_to:
  paradigm: ascendc
```
- **applies_to**: `soc=Ascend950PR_9579; cann=9.0.0 (V100R001C25B046); bisheng=15.0.5+2026-04-13`
- **applies_to**: `soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=15.0.5+2026-01-28`
- **last_verified**: 2026-05-07 (A3 cross-arch confirmation)
- **status**: CONFIRMED on A5 AND A3 — uniform 2-ULP floor is **chip-family-wide** for Sigmoid. Both arches well-behaved.
- **Severity**: LOW (uniform 2-ULP across all bands — well-behaved, no failure mode on either arch).
- **Symptom (measured, both A5 and A3)**: `AscendC::Sigmoid<float>` floor is **2 ULP** uniformly across all bands (small / mid / transition / plateau). No degeneracy near zero (unlike Tanh in PB-24).
  - **A5**: max 2 ULP, mean 0.39 ULP, 99% of 14,349 points within 1 ULP, max abs err 1.04e-7. Histogram: 0→8941, 1→5210, 2→198, ≥4→0.
  - **A3**: max 2 ULP across ALL bands (including tiny + small), 62% bit-exact, **99.95% within 1 ULP** — even cleaner aggregate than A5. No points above 3 ULP.
- **Root cause hypothesis**: bisheng's `Sigmoid` polynomial is correctly-rounded-most-of-the-time, 2-ULP bound. Compared to `Tanh`, does NOT have a small-x failure mode — the `1/(1+exp(-x))` formulation preserves expected output magnitude near 0 (sigmoid(0) = 0.5), so the ULP measurement isn't pathological.
- **Workaround**: for sub-ULP fp32 sigmoid, implement via `Exp + Reciprocal + Add` (same primitives `Tanh` Cephes-form uses). For 2-ULP-tolerant uses (most ML inference paths), the primitive is fine on both arches.
- **Detection**: edge_dataset cases at the 2-ULP boundary may show 0/1/2 ULP scatter; distinguish primitive ceiling from kernel cancellation drift via this entry.
- **Evidence**:
  - A5: `workspace/_probes/tanh_sigmoid_precision_a5_cann9.0.0/PROBE_REPORT.md` (2026-05-06)
  - A3: `workspace/_probes/tanh_sigmoid_precision_a3/PROBE_REPORT.md` (2026-05-07, DS-side)
- **Cross-reference**: PB-24 (paired Tanh measurement, very different bimodal profile despite both being polynomial-evaluation primitives), OL-103 §Refined-statement (1-ULP floor → 2-ULP floor refinement, pending DS A3 + soften edit).

---
### PB-37: AscendC `Cast<float, uint8_t>` is unsupported — silent garbage, not an error
`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5+2026-04-13; op_class=all (any kernel that needs uint8/bool tensor in fp32 compute)`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_9382 (A3 V220 — pattern not yet probed there)`
- **Severity**: HIGH (silent miscompile — output values 10×–100× reference magnitude; no compile error, no runtime error, no NaN/Inf signal).
- **Symptom**: `Cast(fp32_dst, u8_src, RoundMode::CAST_NONE, count)` produces an instruction that does NOT do what the call site implies. The destination LocalTensor is left in an indeterminate state — observed values uncorrelated with input (large-magnitude garbage even when input is bool 0/1).
- **Root cause**: AscendC `Cast` only supports a specific table of (src_dtype, dst_dtype) pairs. Per official docs, supported uint8 SRC pairs are: `uint8 → half`, `uint8 → uint16_t`, `uint8 → uint32_t`. NOT in the table (and silently produce garbage):
  - `uint8 → float` ✗
  - `uint8 → bfloat16_t` ✗
  - `uint8 → int8 / int16 / int32` ✗
- **Workaround — canonical two-step lift through half**:
  ```cpp
  __aicore__ inline void CastU8ToFp32(
      const LocalTensor<float>& dst,
      const LocalTensor<uint8_t>& src,
      const LocalTensor<half>& tmp_half,   // VECCALC scratch, count-aligned to 16
      int32_t count)
  {
      Cast(tmp_half, src,        RoundMode::CAST_NONE, count);  // u8 → half (exact for 0/1 mask values)
      PipeBarrier<PIPE_V>();
      Cast(dst,      tmp_half,   RoundMode::CAST_NONE, count);  // half → fp32 (exact for 0/1)
      PipeBarrier<PIPE_V>();
  }
  ```
  Both legs ARE in the supported pair table. For boolean masks the two-cast chain is bit-exact (0/1 representable in half exactly).
- **Detection**: precision `max_abs_diff` shows large-magnitude divergence with no obvious per-element correlation; mismatch count not factorable into alignment-overshoot. Before assuming algorithm bug, **grep the kernel for `Cast(.*float.*uint8_t.*)` or `Cast(<float-LT>, <u8-LT>, ...)` patterns** — even one such call short-circuits to PB-26.
- **Prevention (Phase B mandatory check)**: any `Cast<DST, SRC>` in a kernel must be cross-checked against the AscendC Cast precision-conversion table before the build. OL-80 + OL-84 already mandate "check docs before Cast"; PB-26 is the named instance that catches the cost of skipping it.
- **Evidence**: op#25 MaskedSoftmaxWithAttentionDropoutBackward kw-2 (2026-04-24). Boolean mask `attention_mask` (uint8) and dropout mask (uint8) needed in fp32 compute path. Direct `Cast<float, uint8_t>(maskFp32, maskU8, CAST_NONE, count)` produced output values 10×–100× reference. ~1.5 hours of precision debugging — kw-1 had diagnosed three other root causes (alignment, broadcast, scalar fusion) but missed this. Two-step lift via `tmp_half` fixed the path; rest of the kernel was already correct.
- **Related**: OL-80 / OL-84 (always check API surface before assuming an op exists), EC-23 (DataCopy alignment — orthogonal), P-P52 (fp32 promotion — different problem; P-P52 assumes the cast itself is supported).

---

### PB-28: `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` macro is arch35-only — `RegisterAscendBinary 107000` on V220 [V220]

```yaml
applies_to:
  paradigm: ascendc
```

- **Severity**: HIGH (build succeeds but launch fails; kernel emits AIC `.o` without runnable binary; no degraded-mode workaround)
- **Status**: CONFIRMED 2026-05-07/2026-05-08 DS A3 cold-starts — multiple ops (4_Abs, 22_Nonzero, 5_Cumsum) hit this on every kernel that started from an A5 archive copy
- **applies_to**: soc=Ascend910_9382 (V220 single-die — A2/A3); does NOT apply on Ascend950PR (A5/V351/arch35 where the macro is the canonical entry-form)
- **Symptom**: Kernel source uses `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` as the entry-form macro (the canonical A5 pattern in many op#X archives). Build succeeds, but launch fails with `RegisterAscendBinary 107000` on V220. The macro expands to arch35-only entry-attribute metadata that V220's kernel registration loader rejects.
- **Fix**: On V220 use the bare `__global__ __aicore__ void <kernel>(args)` entry-form, no macro wrapper:
  ```cpp
  // V220 (A2/A3) — bare entry, no KERNEL_TASK_TYPE_DEFAULT
  extern "C" __global__ __aicore__ void my_kernel(GM_ADDR x, GM_ADDR y, ...) {
      // body
  }

  // A5 (Ascend950PR/V351/arch35) — KERNEL_TASK_TYPE_DEFAULT canonical
  extern "C" __global__ __aicore__ void my_kernel(GM_ADDR x, GM_ADDR y, ...) {
      KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
      // body
  }
  ```
- **Detection**: build PASS but launch-time `RegisterAscendBinary 107000` is the smoking gun. Pre-build grep: `grep -E "KERNEL_TASK_TYPE_DEFAULT\(KERNEL_TYPE_AIV_ONLY\)" workspace/<op>/kernel/*.{cpp,h}` — if any hit AND TARGET ∈ {a3, a2, a3-ds, a2-ds}, rewrite to bare form before build. (Note: grep MUST include the `KERNEL_TYPE_AIV_ONLY` argument — see scope-clarification below.)
- **Scope is `KERNEL_TYPE_AIV_ONLY` ONLY — do NOT generalize**: this entry covers the `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` macro specifically. Other `KERNEL_TYPE_*` variants are NOT subject to the same arch35 restriction:
  - `KERNEL_TYPE_MIX_AIC_1_2` is **valid on V220** (verified by CANN's own `flash_attention_score` arch22 source — see `patterns/unverified/candidates.md` CAND-FA1). Wrapping a mixed cube+vec entry in `#if __NPU_ARCH__ >= 3510` because "PB-28 says it's arch35-only" is the wrong inference — the historical kw-3 hard-hang behind that defensive guard was NOT a `MIX_AIC_1_2` register-binary failure; it was the `MatmulImpl<> + manual CrossCoreSet/Wait` deadlock now codified as [PB-34](#pb-34-matmulimpl-with-manual-crosscoresetflagwaitflag--mix_aic_1_2-deadlock-on-v220-v220-mixed-mode-sync).
  - `KERNEL_TYPE_AIC_ONLY` and `KERNEL_TYPE_MIX_AIC_1_1` are out of scope of PB-28 — no V220 register-binary evidence either way; if you encounter `107000` with those, file a separate PB entry.
  - **Anti-pattern caught 2026-05-21** (3_FusionAttention kw-1): kernel comment "PB-28: KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2) is arch35-only" — false attribution. PB-28 never claimed that.
- **Evidence**: Promoted to PB-28 2026-05-09 from DS-local `src/scripts/env_quirks_a3-ds.json` quirk #4 (DS-flagged repeat hit across 4_Abs / 22_Nonzero / 5_Cumsum cold-starts). Pattern was the cross-arch issue of porting an A5 kernel to A3 without rewriting the entry-form. **All confirmed instances used `KERNEL_TYPE_AIV_ONLY`** — no `MIX_AIC_*` instance has ever produced `RegisterAscendBinary 107000`.
- **Positive-side confirmation (arch35 canonical form works)**: recurrent_gated_delta_rule kw-1 (2026-06-18, A5 Ascend950PR_957b / arch35, CANN 9.1.T500): a genuinely vec-only recurrent-decode kernel using `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` built + launched + ran 30/30 clean on arch35 with no RegisterAscendBinary rejection — confirms the macro is the canonical entry-form on A5 (the 107000 rejection is V220-only, exactly as scoped above). The port_a3 brief §2b V220-reject note does NOT apply on arch35.
- **Cross-ref**: `hardware/target/ascend910c.md` § Kernel-launch (V220 A3 entry form), `hardware/target/ascend910b.md` § Kernel-launch (V220 A2 entry form, same family), `env_quirks_a3-ds.json` quirk #4 (DS env preflight catalog), `patterns/unverified/candidates.md` CAND-FA1 (`MIX_AIC_1_2` V220 source-derived evidence), PB-34 (the actual mixed-mode failure mode kw-3 hit, mis-blamed on PB-28).

---
### PB-29: `add_modules_sources` called twice for same op → silent target conflict in `generate_bin_scripts` [V351, build-system]

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 SKILL.md §339-352`

**Symptom**: At link/binary-generation time, `generate_bin_scripts` reports duplicate target name or duplicate symbol for the same op. Build log may also show repeated entries in `COMPILED_OPS` / `COMPILED_OP_DIRS`.

**Root cause**: `add_modules_sources` appends the op name to global cache variables (`COMPILED_OPS`, `COMPILED_OP_DIRS`) via `set(... CACHE FORCE)`. Two calls = two appends = duplicate listing. Downstream `generate_bin_scripts` then tries to generate the same kernel binary target twice.

**Anti-pattern** (BAD):
```cmake
# A3 baseline already registers:
add_modules_sources(... OPTYPE foo ACLNNTYPE aclnn_exclude DEPENDENCIES ...)

# Then A5 port ADDS a second call (WRONG):
add_modules_sources(... OPTYPE foo ACLNNTYPE aclnn_exclude
    COMPUTE_UNIT "ascend950" TILING_DIR "arch35")
```

**Fix** (consolidate to ONE call):
```cmake
add_modules_sources(... OPTYPE foo ACLNNTYPE aclnn_exclude
    COMPUTE_UNIT "ascend910b" "ascend910_93" "ascend950"
    TILING_DIR   "default"    "default"      "arch35"
    DEPENDENCIES ...)
```

**Detection signature**:
```bash
# In CMakeLists.txt of an op being ported, count add_modules_sources calls for that op
grep -c "add_modules_sources.*OPTYPE\s*${op_name}\b" op_host/CMakeLists.txt
# > 1 → BUG
```

**Evidence**:
- PR 103 codifies this as Trap #1 in CMakeLists.txt 经验教训
- Reasoning is `CACHE FORCE` semantics in CMake — verifiable in build/CMakeCache.txt

**Mitigation gate**: `aog-self-critic` post-worker pass should grep CMakeLists.txt for duplicate `add_modules_sources` registrations of the same op name; reject if found.

---

### PB-30: CMake `COMPUTE_UNIT` ≠ `TILING_DIR` list length → `find_value_by_key` FATAL_ERROR [V351, build-system]

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 SKILL.md §354-367`

**Symptom**: At CMake configure time, `find_value_by_key()` (internal CANN cmake helper) calls `message(FATAL_ERROR ...)` complaining about COMPUTE_UNIT and TILING_DIR list-length mismatch.

**Root cause**: CMake silently drops empty strings (`""`) from list arguments. When the author intends `TILING_DIR = ["", "", "arch35"]` to mean "A3/910b/910_93 use root, A5 uses arch35/", CMake collapses it to `["arch35"]` (1 element) while `COMPUTE_UNIT` stays at 3 elements → length mismatch.

**Anti-pattern** (BAD — empty string eaten):
```cmake
COMPUTE_UNIT "ascend910b" "ascend910_93" "ascend950"
TILING_DIR   ""           ""              "arch35"
# Actual after parse: COMPUTE_UNIT=3 items, TILING_DIR=1 item → FATAL_ERROR
```

**Fix** (use named subdirs even when not strictly needed):
```cmake
COMPUTE_UNIT "ascend910b" "ascend910_93" "ascend950"
TILING_DIR   "default"    "default"      "arch35"
# Then create op_host/default/ alongside op_host/arch35/
```

**Detection signature**:
```bash
# Count list items in each — must match
awk '/COMPUTE_UNIT/{n=0; for(i=2;i<=NF;i++) if($i!~/^$/) n++; print "cu",n}
     /TILING_DIR/{n=0; for(i=2;i<=NF;i++) if($i!~/^$/) n++; print "td",n}' \
     op_host/CMakeLists.txt
```

**Evidence**:
- PR 103 codifies as Trap #2; ties directly to CMake list-arg semantics
- Reproducible: `cmake -DLIST_VAR="" ""` produces 0-length list, not 2-length

**Mitigation gate**: `aog-self-critic` post-worker — when adding an `ascend950` element to COMPUTE_UNIT, REQUIRE that TILING_DIR has a corresponding non-empty entry; if mismatched, suggest the named-subdir fix.

---

### PB-31: Missing `config/ascend950/<op>_binary.json` → 950 build SILENTLY SKIPPED [V351, build-system]

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 SKILL.md §369-372`

**Symptom**: Build appears to succeed (no error / warning), but the resulting op package does NOT include the A5 (ascend950) kernel binary. At runtime, aclnn or torch_npu reports "operator not supported on this device" or similar generic missing-op error.

**Root cause**: The CANN build system gates per-SoC kernel compilation on the presence of `op_host/config/<compute_unit>/<op_name>_binary.json`. The check is "if file exists, compile for this SoC; else skip without warning". For port_a3_to_a5, if the dev added `OpAICoreConfig` for ascend950 in `_def.cpp` AND modified CMakeLists.txt AND created `arch35/` source — but FORGOT to copy `config/ascend950/<op>_binary.json` — the 950 path is silently skipped and the bug surfaces only at runtime on the target device.

**Anti-pattern**:
```bash
ls op_host/config/
# ascend910b/  ascend910_93/    ← ✘ no ascend950/
```

**Fix**:
```bash
mkdir -p op_host/config/ascend950
cp op_host/config/ascend910b/<op>_binary.json       op_host/config/ascend950/
cp op_host/config/ascend910b/<op>_simplified_key.ini op_host/config/ascend950/
# Adjust _binary.json contents if A5 supports different dtypes (FP8/HiFloat8/etc.)
```

**Detection signature**:
```bash
# After kw declares done, verify:
test -f op_host/config/ascend950/${op}_binary.json &&
test -f op_host/config/ascend950/${op}_simplified_key.ini ||
  echo "BUG: config/ascend950/ incomplete — 950 build will be silently skipped"
```

**Evidence**:
- PR 103 codifies as Trap #3; explicitly described as "缺失则 950 编译被静默跳过（无报错），极易遗漏"
- Direct hardware reproduction: when build/ pipeline doesn't even reach compile-arch35 step

**Mitigation gate**: `aog-self-critic` post-worker pass MUST verify both files exist in `config/ascend950/`; reject finalize if missing. Additionally, `aog-prior-art-verify` Phase 3 (build candidate) should fail-fast if these files don't exist.

**Other instances (predicted)**: any future A3→A5 port. The 14 no-upstream ops in our scan list (cohort 1+2) will all need this check at finalize time.

### PB-32: SIMT DCache shares UB → tiling MUST reserve 40KB or risk silent UB OOB [V351, simt-tiling]

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=scatter,gather,simt-l3; phase=tiling`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 l3-simt-optimization-guide.md §264-300`

**Symptom**: Runtime UB-out-of-bounds error OR silent data corruption when running a SIMT (L3) kernel on A5. Memory-based path on same op + same data works correctly. Tiling code computed UB allocation against the FULL UB capacity, not the SIMT-reserved subset.

**Root cause**: On A5, SIMT DCache **physically shares the UB SRAM** with the standard UB Buffer. Hardware reserves `SIMT_UB_SIZE_BYTE = 40960` (40KB) at the top of UB for SIMT thread state when ANY SIMT kernel runs on the core. If tiling code allocates UB assuming the full advertised capacity — physical 256KB, or even the framework-usable 248KB (`GetCoreMemSize` = `ub_size` 253952 = 262144 − 8KB framework reserve) — SIMT execution overwrites the last 40KB → UB OOB. **Effective UB for SIMT tiling ≈ 208KB** (253952 − `SIMT_UB_SIZE_BYTE` 40960 = 212992). Two-layer reservation: 256KB physical → 248KB usable (framework) → 208KB SIMT-effective. Canonical constants: `hardware/target/ascend950pr.md`.

The kicker: the error is **silent** when the corrupted region holds data that was already consumed by the time SIMT runs. Sporadic perf-test failures, NaN spikes, or wrong reduction results that don't reproduce deterministically all point here.

**Fix** (host tiling code MUST apply, not kernel side):

```cpp
const static int64_t SIMT_UB_SIZE_BYTE = 40960;

uint64_t ubSizePlatForm;
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
aicoreParams_.ubSize = ubSizePlatForm;

// On A5 SoC, reserve SIMT DCache:
if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
    aicoreParams_.ubSize = ubSizePlatForm - SIMT_UB_SIZE_BYTE;
}
```

**Or use a shared helper** (canonical, multi-op reusable):

```cpp
namespace Ops { namespace Common {
    constexpr int64_t SIMT_UB_SIZE_BYTE = 40960;

    inline uint64_t GetAvailableUbSize(platform_ascendc::PlatformAscendC& platform,
                                        bool isRegbase) {
        uint64_t ubSizePlatForm;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        if (isRegbase) {
            ubSizePlatForm -= SIMT_UB_SIZE_BYTE;
        }
        return ubSizePlatForm;
    }
}}
```

**Detection signature** (host tiling .cpp audit):

```bash
# In an op marked L3 (uses SIMT), check tiling reserves SIMT UB
grep -nE "SIMT_UB_SIZE_BYTE|IsRegbaseSocVersion\(\)" op_host/*.cpp op_host/*/<op>_tiling.cpp
# Should appear at least once in the tiling computation
```

**Anti-patterns**:
- Hardcoding `ubSize = 256 * 1024` for A5 — bypasses the reservation, silent OOB
- Reserving 40KB ONLY when classify_tier == "L3" — wrong; ANY SIMT kernel on the core triggers reservation, even if this op is L1 elsewhere. The check is `IsRegbaseSocVersion()`, not per-op SIMT usage.
- Reserving in kernel side rather than tiling side — host tiling decides UB allocation; kernel just consumes

**Evidence**:
- PR 103 l3-guide §264-300 codifies the rule + provides the canonical template
- Listed in PR 103 SKILL.md §38-42 as one of two L4-escalation triggers ("UB 预留不足（需 40KB SIMT DCache） → L4")

**Other instances (predicted)**: every L3-classified op (cohort 2 ACTIONABLE candidates: `flash_attention_score`, `moe_init_routing_v3`, possibly `repeat_interleave_v2`, `masked_select_v3`); also any op-set with mixed L1+L3 kernels — `IsRegbaseSocVersion()` test triggers the reservation for the entire op-set on that SoC.

**Mitigation gate**: `aog-self-critic` post-tiling-author pass MUST grep host tiling code for `SIMT_UB_SIZE_BYTE` / `IsRegbaseSocVersion()` when the op is classified L3; reject finalize if missing.

**Cross-reference**:
- OL-143 (L1/L2/L3 classifier — L3 path needs this reservation)
- OL-150 (SIMT programming model)
- OL-151 (SIMT helpers — `__local_mem__` allocations ALSO consume UB, must come from the post-reservation balance)

### PB-33: op_host/ archive incomplete — patches shipped instead of complete files [V351+all-modes, finalize-pipeline]

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=all; phase=finalize`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: User feedback 2026-05-14T02:55Z (Discord) — "op_host 代码似乎有一些缺失"`

**Symptom**: Downstream reviewers / CANN team / `cann/ops-nn` PR审核 cannot apply our archived ops because `op_host/` either:
1. **Missing entirely** — no `op_host/` directory in the archive (5 archived ops at scan time: rms_norm_quant, group_norm_silu_quant), OR
2. **Only has `<op>_def.cpp.patch`** + `config/` — patches without the master they were generated against (ctc_loss_v3, gather_elements_v2, apply_adam_w_quant)

PR4778 packaging REQUIRES a complete task-owned `op_host/` layout: `<op>_def.cpp` +
`<op>_tiling.cpp` + `<op>_tiling.h` + `CMakeLists.txt` +
`config/ascend950/{*_binary.json,_simplified_key.ini}` (plus optional `op_api/<op>.{cpp,h}` +
`<op>_infershape.cpp`). Patches are review aids only. Target layout may guide directory completeness,
but copied target bodies do not satisfy generation or validation.

**Root cause**: kw_brief Phase B.4 (pre-2026-05-14) only required `workspace/{op}/op_host/<op>_def.cpp.patch`. Worker complied — wrote only the patch. `_tiling.cpp/.h`, `CMakeLists.txt`, `op_api/` were never required to be produced. For ops without ANY worker-written op_host/ (rms_norm_quant), even the patch wasn't required because workspace was Path B cpp-binary style which skipped `op_host/` setup entirely.

**Fix (2026-05-14, commit pending)**:

1. **kw_brief Phase B.4 rewritten** to mandate complete files (not patches). Patches still produced as review-aid trail, but the SHIP artifact is the complete `<op>_def.cpp` + `_tiling.{cpp,h}` + `CMakeLists.txt` + optional `op_api/`.

2. **finalize_pipeline `_check_op_host_completeness` gate added** (`GateID.OP_HOST_COMPLETENESS`). Counts non-config / non-patch files in `workspace/op_host/`; < 3 → ROLLBACK to `await_worker`. Apply it when the selected arch22→arch35 packaging contract requires `op_host/`; standalone backward pybind builds use their explicit carve-out.

3. **`briefs/_common.py:fixed_layout_block()` added** + injected into ALL 6 agent briefs (kw / ko / fo / pp / ar / da). Every agent now sees the same PR4778 contract; no agent can produce or modify an op without knowing the required output layout.

**Detection signature** (post-finalize audit):

```bash
# Count non-patch, non-config files in archive/op_host/
n=$(find <archive>/op_host -type f \
    -not -path '*/config/*' -not -name '*.patch' | wc -l)
[ $n -lt 3 ] && echo "PB-33 violation: only $n files"
```

**Evidence**:
- 5 archived ops at scan time (2026-05-14) violated PR4778:
  - ctc_loss_v3 / gather_elements_v2 / apply_adam_w_quant: only `.patch` + `config/`
  - rms_norm_quant / group_norm_silu_quant: no `op_host/` at all
- User feedback (Discord 02:55Z 2026-05-14): "op_host 的代码似乎有一些缺失，有的算子给了 ctc_loss_v3_def.cpp.patch，有的算子产物没有 op_host"
- expand_into_jagged_permute 2026-05-17 (port_a3_to_a5, kw-1): first port_a3 op produced post-fix. Workspace ships 6 complete `op_host/` files (`<op>_def.cpp` + `<op>_tiling.{cpp,h}` + `CMakeLists.txt` + 2 `config/ascend950/` files) + 2 `op_kernel/` files (`arch35/<op>.h` + `<op>_apt.cpp`). No `.patch`-only artifact. Reviewer can diff against current upstream without needing the original master snapshot. Confirms kw_brief Phase B.4 rewrite + `_check_op_host_completeness` gate enforce the contract.
- fatrelu_mul 2026-05-17 (port_a3_to_a5, kw-1): mirrored `op_host/` + `op_kernel/arch35/` + `<op>_apt.cpp` files for PR4778 layout-contract completeness BUT NOT BUILT — actual ship + verify path runs the standalone pybind kernel at `workspace/fatrelu_mul/kernel/` via ACLRT_LAUNCH_KERNEL (per P140 pivot). The op_host/arch35 artifacts serve as upstream-ready review aids: a CANN-team reviewer can apply these to ops-nn directly with no algorithmic changes (arch35 `.h` would need V220 implementation body copied verbatim — only macro guards and includes changed in the L1 mechanical edit). Demonstrates the layout contract can be satisfied as review trail even when the active build path is pybind/ACLRT_LAUNCH_KERNEL rather than the full ops-nn pipeline.

**Mitigation gate**: `finalize_pipeline.check_finalize_eligibility` returns `GateID.OP_HOST_COMPLETENESS` on `op_host/` insufficient. Tests at `test_finalize_gate_contract.py::test_op_host_missing_dir` + `::test_op_host_only_patch_no_complete_files`.

**Other instances (predicted)**: every future op-gen run. Without this gate, regress to the shipping-incomplete pattern is silent (precision/perf gates don't catch missing artifacts).

**Historical backfill note (superseded)**: the old proposal copied target tiling files into five
incomplete archives. Current policy requires regenerating complete task-owned files from the selected
source contract and rerunning build/provenance/truth gates.

**Shared-common layout guidance (adaptive_avg_pool3d witness)**: when target prior art uses a shared
`<family>_common/op_kernel/arch35/` layout, use it to infer package dependencies, not to copy bodies.
A strict local-only layout would otherwise tempt one of:
- (a) copying shared kernel files into the op's local `op_kernel/arch35/`, OR
- (b) rewriting `<op>_apt.cpp`'s `#include "../<family>_common/arch35/..."` to `arch35/...`

Both break on-host build deployment because upstream's tree expects the shared-common path; siblings (`adaptive_max_pool3d` etc.) co-rely on that same dir. **Permitted layout for Mode B ports**: workspace MAY include a sibling `<family>_common/` directory mirroring upstream's structure, when ALL of:

1. The op's `op_kernel/arch35/` is empty (no `<op>_*.{h,cpp}` files at that path)
2. Upstream has a sibling `<family>_common/op_kernel/arch35/` containing `<op>_*` files (Mode B per OL-141)
3. The op's `<op>_apt.cpp` references the shared path via `#include "../<family>_common/arch35/..."`
4. Only the target op's arch35 dependencies are shipped from the sibling dir (other family-member ops MUST be excluded from the archive — one op per archive, even when they share kernel sources)

In this case PB-33's completeness intent is satisfied by task-owned `op_kernel/` + `op_host/` and the
declared shared dependency surface. The `_check_op_host_completeness` gate still counts `op_host/`;
a kernel-completeness gate must understand shared dependencies without accepting a target mirror.

Detection signature for Mode B archive:
```bash
# Mode B if op_kernel/arch35/ is empty AND a sibling *_common/op_kernel/arch35 has <op>_*
op_arch35=$(ls <archive>/op_kernel/arch35/*.{h,cpp} 2>/dev/null | wc -l)
sibling_arch35=$(ls <archive>/../*_common/op_kernel/arch35/<op>_*.{h,cpp} 2>/dev/null | wc -l)
[ "$op_arch35" -eq 0 ] && [ "$sibling_arch35" -gt 0 ] && echo "Mode B archive"
```

**Cross-reference**:
- PR4778 spec (CANN ops-nn) — the canonical layout being mirrored
- `kw_brief.py §Phase B.4` (rev 2026-05-14) — the producer-side rule
- `_common.py:fixed_layout_block()` — the shared contract block
- `finalize_pipeline._check_op_host_completeness` — the enforcement

---

### PB-34: `MatmulImpl<>` with manual `CrossCoreSetFlag`/`WaitFlag` + `MIX_AIC_1_2` deadlock on V220 [V220, mixed-mode-sync]

`applies_to: soc=Ascend910_9382 (V220 A2/A3 single-die); cann=9.0.0+; op_class=mixed_aic_aiv_with_high_level_matmul_library; macro=KERNEL_TYPE_MIX_AIC_1_2`
`verified_on: a5_ops:3_FusionAttention case_b27a259d (kw-3, 2026-05-07) — independently witnessed in cann_learn offline scan 2026-05-21 (run_id 5f1f559cb8fa)`
`verified_does_not_reproduce_on: Ascend950PR_9579 (V351 / A5) — probe_a5_v300_fa_sync 2026-05-23 — Pattern A runs clean: MatmulImpl<> + manual CrossCoreSetFlag<0x2>(FLAG_AIC_DONE=0) + MIX_AIC_1_2 + 16×16×16 fp16 mm.IterateAll + AIV Muls(*,2.0) all complete in 0.036ms steady state with bit-exact y output and non-zero matmul C output. V220 FFTS sync-slot conflict does NOT reproduce on V351 hardware. Cross-ref: workspace/probe_a5_v300_fa_sync/PROBE_REPORT.md`
`verified_does_not_reproduce_on (FULL-OP scale): Ascend950PR (V351 / A5, CANN 9.1.T500) — chunk_gated_delta_rule (GDN) light-port 2026-06-15 — the FULL multi-stage upstream V220 kernel (8 matmul::MatmulImpl<> instances ×3 cube stages + KERNEL_TYPE_MIX_AIC_1_2 + manual CrossCoreSetFlag<0x2,PIPE_FIX|MTE3>/WaitFlag handshakes + SyncAll + sequential UT-inverse) COMPILED FIRST-TRY on bisheng dav-c310 + RAN without hang + 122/122 T1 PASS. This confirms the no-reproduce verdict at PRODUCTION FULL-OP scale, not just the trivial micro-probe — definitively falsifying the "MatmulImpl<>+MIX+manual-CrossCore needs structural regbase rewrite on arch35" inference for the full-op case. Consequence: for a V220 cube-MIX fused op, the DEFAULT A5 route is a LIGHT PORT (keep MatmulImpl<> + the manual flag chain; adapt only the ACLRT_LAUNCH entry + host tiling), NOT a hand-rolled tile-Mmad rewrite.`

- **Severity**: HIGH (build + register-binary both succeed, kernel launches, then hard-hangs forever; vector core fault `LaunchAscendKernel 507035` or AICore timeout `507014` depending on which side starves first; no degraded-mode workaround — the only safe response is REWRITE to one of the two valid patterns below).
- **Status**: CONFIRMED 2026-05-07 (3_FusionAttention kw-3 first-witnessed); CODIFIED 2026-05-21 after cann_learn extracted CAND-FA1 with hard-do-not-apply clause naming this exact combination. This PB documents the negative-evidence side of CAND-FA1's pattern — CAND-FA1 says "do not apply when MatmulImpl<> is in use", this PB says "and here's specifically what goes wrong if you do".
- **Symptom**: Mixed cube+vec kernel dispatched via `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`. AIC half uses the high-level `MatmulImpl<>` / `MatmulClient<>` / `KFC` library template for matmul stages. AIV half pairs with the AIC stages via `CrossCoreSetFlag<0x2, PIPE_MTE3>(FLAG_X_DONE)` + `CrossCoreWaitFlag<0x2>(FLAG_X_DONE)`. Build succeeds, register-binary succeeds, kernel launches — then either AIV hangs at `CrossCoreWaitFlag(FLAG_MM_DONE)` waiting on a flag the cube side never publishes, OR AIC hangs at its `Iterate()`/`GetTensor()` body waiting on KFC-internal events that don't fire because the user-owned flag chain has captured the same hardware sync slots. Result on the host: `LaunchAscendKernel` returns `507035` (vector core abnormal exit) or `507014` (AICore timeout), depending on which engine starves first.
- **Root cause**: `MatmulImpl<>` runs its own internal cross-core synchronization through the KFC (Kernel Framework Client) protocol, consuming the same FFTS flag-ID hardware slots that user-owned `CrossCoreSetFlag<0x2>(0..7)` calls allocate from. The user-owned and library-owned slot lifetimes are incompatible: KFC expects exclusive ownership of the AIC↔AIV handshake space inside one `Iterate()` call, but the user-owned flag chain re-enters the same slots between AIV stages, corrupting KFC's internal flag-count state machine. The hang is not a software bug in either side individually — it is a hardware-level resource conflict on the FFTS sync slots.
- **Fix — pick exactly ONE of two valid V220 mixed-mode patterns; never mix them**:
  1. **Pattern A — tile-MMAD primitives + manual CrossCore** (CAND-FA1's recommendation). Replace `MatmulImpl<>` instantiations with raw tile-MMAD calls: `LoadData2D` / `LoadData3D` for L1→L0A/B fills, `Mad<>` for the multiply-accumulate, `FixpipeOut` for L0C→GM write-back. Keep the existing `CrossCoreSetFlag<0x2, PIPE_FIX>(...)` / `CrossCoreWaitFlag<0x2>(...)` chain — it works (verified by CANN's own `flash_attention_score` arch22 source).
  2. **Pattern B — `MatmulImpl<>` with KFC-implicit sync ONLY, NO manual CrossCore**. Keep the `MatmulImpl<>` instantiations. Remove ALL `CrossCoreSetFlag<0x2>` and `CrossCoreWaitFlag<0x2>` calls from BOTH the AIC and AIV sides of the kernel. Stage handoff must go through the matmul library's own queue/callback surface (`SetTensorA/SetTensorB/Iterate/GetTensor` on AIC, mirroring `GetTensorC` consumers on AIV). This pattern is only viable when the cross-stage handoff fits inside one `Iterate()` boundary — multi-stage FA pipelines often do NOT, which is why CAND-FA1 favors Pattern A.
- **Detection** (pre-build static guard):
  ```bash
  # Hard-stop: same kernel file mentions BOTH MatmulImpl AND CrossCoreSetFlag<0x2 — PB-34 collision
  for f in workspace/<op>/kernel/*.{h,cpp}; do
      grep -lq "MatmulImpl<\|MatmulClient<" "$f" \
        && grep -lq "CrossCoreSetFlag<0x2\|CrossCoreWaitFlag<0x2" "$f" \
        && echo "PB-34 violation candidate: $f"
  done
  ```
  Runtime smoking-gun: `LaunchAscendKernel` returns `507035` (vec) or `507014` (cube) on a kernel that built + registered cleanly. If the kernel ALSO previously ran fine on AIV_ONLY fallback, that confirms the issue is in the cube+vec sync surface, not the math.
- **Anti-pattern (DO NOT EMIT)**:
  ```cpp
  // BAD — Pattern A and Pattern B mixed → V220 deadlock
  MatmulImpl<AT, BT, CT, /*BIAS=*/CT, MM_CFG_STATIC> mm;  // library cube
  mm.Init(...); mm.SetTensorA(...); mm.SetTensorB(...); mm.Iterate(); mm.GetTensor(...);
  CrossCoreSetFlag<0x2, PIPE_FIX>(FLAG_MM_DONE);          // user-owned flag on top of KFC
  // ... on AIV side ...
  CrossCoreWaitFlag<0x2>(FLAG_MM_DONE);                   // hangs forever
  ```
- **Evidence**:
  - 3_FusionAttention kw-3 (2026-05-07) `case_b27a259d`: cube+vec MIX_AIC_1_2 + `MatmulImpl<>` + `CrossCoreWaitFlag(FLAG_MM1_DONE)` — AICore timeout 507014, AIVec stuck on `FLAG_MM1_DONE`. kw-3's defensive response: wrap the entire mixed entry in `#if __NPU_ARCH__ >= 3510` and route V220 traffic through an AIV-only fallback (achieved 0.04× CANN perf — wrong root-cause fix).
  - cann_learn offline scan 2026-05-21 (run_id `5f1f559cb8fa`): CAND-FA1 extracted from CANN `flash_attention_score` arch22 source, including hard-do-not-apply clause "kernel must NOT instantiate `MatmulImpl<>` / `MatmulClient` / KFC" — this PB-34 documents the SPECIFIC failure mode that clause exists to forbid.
  - 3_FusionAttention kw-1 (2026-05-20) `fusion_attention_fused_kernels.cpp`: emitted `MatmulImpl<>` (line 493/547 of `fusion_attention_kernel.h`) AND `CrossCoreSetFlag<0x2, PIPE_MTE3>(FLAG_CANON_DONE)` (line 103 of fused_kernels.cpp) — the exact PB-34 collision. Built clean but `LaunchAscendKernel 507035` at runtime on every test case (1/61 PASS).
  - **chunk_gated_delta_rule (GDN) light-port (2026-06-15, A5 / arch35 / CANN 9.1.T500) — FULL-OP no-reproduce witness**: the upstream V220 ChunkGatedDeltaRule kernel (8 `matmul::MatmulImpl<MatmulType<GM,ND,bf16,transpose>>` across 3 cube stages, `KERNEL_TYPE_MIX_AIC_1_2`, manual `CrossCoreSetFlag<0x2,PIPE_FIX|MTE3>`/`WaitFlag` handshakes, `SyncAll`, sequential UT-inverse) compiled FIRST-TRY on bisheng dav-c310 and ran without hang — `122/122 T1 PASS`, perf ~89–121µs. The exact PB-34 collision pattern (MatmulImpl<> + manual CrossCoreSetFlag<0x2> in MIX_AIC_1_2) is BENIGN on V351 at full-op scale. This is the positive negative-evidence: the deadlock is V220-specific FFTS slot behavior; on A5 the same code is the recommended light-port route.
  - **Compiler-generated Pattern-A witness + host FFTS mechanism (2026-07-14, a3 / Ascend910_9382 / CANN 9.0.0)**: an independent AscendC compiler emitted tile-MMAD plus manual `CrossCoreSetFlag<0x2,PIPE_FIX>(0)`/`CrossCoreWaitFlag(0)` and the host FFTS descriptor. Its one-cube/one-vector matmul and two-cube FA both compiled and ran on a real A3 device, matched the reference, and the FA was deterministic in process and across fresh processes. The working launch obtains `fftsAddr` through `rtGetC2cCtrlAddr`, passes it as the kernel's first argument, and initializes it with `AscendC::SetSyncBaseAddr`. The standalone harness does not yet expose the required host include path, so this remains a compile-path gap tracked by DEBT-210(d′), not a platform limitation or a worker-side one-line fix. This is a positive witness for Pattern A only; it does not exercise the `MatmulImpl`/KFC slot-collision deadlock documented here. Cross-ref OL-275 and `CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP`.
- **Cross-reference**: `patterns/unverified/candidates.md` CAND-FA1 (Pattern A recommendation + hard-do-not-apply clause this PB enforces); PB-28 (over-generalization of which was the historical pretext for the bogus `__NPU_ARCH__ >= 3510` defensive guard that masked PB-34 on V220 by routing around it); `ascend950pr.md` § Cross-core sync (`MAX_REVERSE_DEPTH = 16` slot-count; user-owned 0..7 vs reserved 8..10 BarrierFlag IDs).

---

### PB-35: `event_t(0)` for cube-internal pipe sync (`MTE1_M` / `M_FIX` / `MTE2_MTE1`) collides with AIC↔AIV CrossCoreSetFlag `FLAG_CANON_DONE` chain in `MIX_AIC_1_2` mode → silent hang [V220 + V351, mixed-mode-sync]

`applies_to: soc=Ascend910_9382,Ascend950PR_9579; cann=9.0.0+; op_class=mixed_aic_aiv_pattern_a_tile_mmad; macro=KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`
`scope_note: BOTH SoCs — V220 (Ascend910_9382, A2/A3) and V351 (Ascend950PR_9579, A5) — but ONLY for the Pattern-A case the op_class tag already names: USER-OWNED cube tile-MMAD (raw Mmad primitives) + HAND-ROLLED CrossCoreSetFlag/WaitFlag chain. The LIBRARY-matmul path (MatmulImpl<> / MatmulClient / KFC + the matmul API's own internal cross-core sync) is EXCLUDED on both SoCs — that path is PB-34's domain, and on V351 it is exactly where the "Pattern A runs clean" negative evidence applies (see unverified_on). Per-SoC evidence: V220 = verified_on (3_FusionAttention kw-4 cycle 3, 2026-05-21); V351/A5 = confirmed_on (kw-gb2 hermetic graybox, 2026-06-03). Scope history: applies_to read soc=Ascend910_9382 only until 2026-07-17, which contradicted this entry's own confirmed_on and would have SUPPRESSED it on A5 — the one SoC where it is CONFIRMED — once composers honor applies_to (DEBT-208).`
`verified_on: a5_ops:3_FusionAttention kw-4 cycle 3 (run buksn5pky 2026-05-21T09:00Z) — Pattern A tile-MMAD primitives + SetFlag/WaitFlag MTE1_M with event_t(0) → kernel enqueues, torch.npu.synchronize() hangs past 90s timeout, no fault thrown`
`unverified_on: Ascend950PR_9579 (V351 / A5) LIBRARY-MATMUL path only — probe_a5_v300_fa_sync 2026-05-23 Pattern C probe was malformed (cross-core HardEvent semantics instead of intra-AIC cube-internal pipe sync); produced a hang for unrelated reason. The run that "works cleanly" on the same V351/CANN combo was MIX_AIC_1_2 + MatmulImpl<> + cross-core flag chain — i.e. the LIBRARY-matmul path, which applies_to EXCLUDES (see scope_note). [SUPERSEDED 2026-07-17 for the hand-rolled case] the former reading of this line — "weakening but not definitively closing PB-35 for V351, follow-up probe needed (intra-AIC SetFlag<HardEvent::M_FIX>(event_t(4..7)) cube primitive-decomp on V351)" — is CLOSED by confirmed_on below: kw-gb2 (2026-06-03) ran that user-owned-cube case on V351 and it DEADLOCKED. V351 is CONFIRMED, not unverified, for the user-owned-cube + hand-rolled-flags case. Cross-ref: workspace/probe_a5_v300_fa_sync/PROBE_REPORT.md`
`confirmed_on: Ascend950PR_9579 (V351 / A5) — kw-gb2 hermetic graybox 2026-06-03 — CONFIRMED for the USER-OWNED-cube + HAND-ROLLED-cross-core-flags case. A cube-MIX FA built from canonical KB (user-owned Mmad tile primitives + hand-rolled CrossCoreSetFlag<0x2>/WaitFlag chain, NOT library matmul) DEADLOCKED at runtime: torch.npu.synchronize() hangs, no aicore exception. Scope clarification of the "weakens PB-35" note above: the prior negative-evidence (probe_a5_v300_fa_sync Pattern A runs clean) applies ONLY to the LIBRARY-matmul path (MatmulImpl<> + the matmul API's own internal cross-core sync). It does NOT cover the user-owned-Mmad + user-hand-rolled-flag case, which DOES deadlock on V351. Root cause of the hand-roll deadlock is now identified (see cross_core_sync.md §4 RUNNABLE): the hand-roll used SYNC MODE 2 (1:2 ratio) + a SHARED flag id for both AIV sub-blocks; the working wholeport uses MODE 4 (1:1, AIV0/AIV1 individually triggerable) + DISJOINT per-sub-block flag ids (id and id+16). The fix is public-API runnable.`

- **Severity**: HIGH (silent hang; no error code; only symptom is sync timeout — easy to misdiagnose as algorithm bug rather than sync collision)
- **Symptom**: Mixed cube+vec kernel using Pattern A (tile-MMAD primitives + manual `CrossCoreSetFlag<0x2>(FLAG_X)` chain at flag IDs 0..7). Cube tile body adds pipe-sync events via `SetFlag<HardEvent::MTE1_M>(event_t(0))` / `WaitFlag<HardEvent::MTE1_M>(event_t(0))` between LoadData and Mmad. Build clean. Kernel enqueue succeeds (`kernel(...)` returns). `torch.npu.synchronize()` then hangs past timeout. No `LaunchAscendKernel` error code, no aicore exception.
- **Root cause**: In `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` mode, low event IDs (0, 1) are shared between the AIC↔AIV cross-core `CrossCoreSetFlag<0x2>(flagId)` chain AND the cube-internal hardware-pipe `SetFlag<HardEvent::X>(event_t(N))` events. Using `event_t(0)` for cube-internal pipe sync collides with the cross-core flag ID 0 (typically `FLAG_CANON_DONE` in FA-class kernels): the cube's MTE1→M wait blocks on a counter that the AIV's CrossCoreSetFlag<0x2>(0) is also feeding, but with incompatible producer/consumer semantics. Result: deadlock with no observable error state.
- **Fix**: Use distinct event IDs ≥ 4 for cube-internal pipe sync. Reserve IDs 0..3 for cross-core flags (the canonical FA-class chain `FLAG_CANON_DONE=0` / `FLAG_MM1_DONE=1` / `FLAG_SOFTMAX_DONE=2` / `FLAG_MM2_DONE=3`). Practical scheme for Pattern A FA tile:
  ```cpp
  // Cross-core flags (reserved IDs 0..3 — used in fused_kernels.cpp top-level):
  constexpr int32_t FLAG_CANON_DONE   = 0;
  constexpr int32_t FLAG_MM1_DONE     = 1;
  constexpr int32_t FLAG_SOFTMAX_DONE = 2;
  constexpr int32_t FLAG_MM2_DONE     = 3;

  // Cube-internal pipe sync (use IDs ≥ 4):
  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event_t(4));
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(event_t(4));
  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(event_t(5));
  AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(event_t(5));
  AscendC::SetFlag<AscendC::HardEvent::M_FIX>(event_t(6));
  AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(event_t(6));
  ```
  Per `ascend950pr.md` § Cross-core sync: user-owned flag ID range is `0..7` (FFTS_MAX_FLAG=7); reserved barrier IDs at `8..10`. Cube-internal pipe events and cross-core flags share that range, so they MUST be allocated disjointly.
- **Detection** (pre-build static guard):
  ```bash
  # Hard-warn: a kernel file using BOTH CrossCoreSetFlag<0x2>(0..3) AND SetFlag<HardEvent::*>(event_t(0..3))
  for f in workspace/<op>/kernel/*.{h,cpp}; do
      cross_core_ids=$(grep -oE "CrossCore(Set|Wait)Flag<0x2[^>]*>\([0-9]+\)" "$f" | grep -oE "\([0-9]+\)" | tr -d "()" | sort -u)
      pipe_ids=$(grep -oE "(Set|Wait)Flag<.*HardEvent::[A-Z_]+>\(event_t\([0-9]+\)" "$f" | grep -oE "event_t\([0-9]+" | grep -oE "[0-9]+" | sort -u)
      overlap=$(comm -12 <(echo "$cross_core_ids") <(echo "$pipe_ids"))
      [ -n "$overlap" ] && echo "PB-35 violation $f: shared IDs $overlap"
  done
  ```
- **Anti-pattern (DO NOT EMIT)**:
  ```cpp
  // BAD — event_t(0) collides with FLAG_CANON_DONE (= 0) in cross-core chain
  // ... AIV side issues: CrossCoreSetFlag<0x2, PIPE_MTE3>(FLAG_CANON_DONE);  // ID 0
  // ... AIC side does:
  DataCopy(l1A, qGm, Nd2NzParams{1, S, D, 0, D, D/16, 16, 0});
  SetFlag<HardEvent::MTE2_MTE1>(event_t(0));  // ← COLLIDES with cross-core ID 0
  WaitFlag<HardEvent::MTE2_MTE1>(event_t(0)); // ← hangs forever
  LoadData(l0A, l1A, LoadData2DParams{...});
  ```
- **Evidence**:
  - 3_FusionAttention kw-4 cycle 3 (`buksn5pky` 2026-05-21T09:00Z): Pattern A tile-MMAD primitives with corrected `Nd2NzParams` shape. Phase 1 (no pipe sync) → fault `0x8000004000` (L0B read/write conflict, sync genuinely missing). Phase 2 (added pipe sync at `event_t(0)`) → silent hang at `torch.npu.synchronize()` past 90s. No fault thrown. Cost: $10.47 / 30min — Phase 2 hang was the directly-observed evidence for this PB.
  - 3_FusionAttention `fusion_attention_fused_kernels.cpp` (existing kw-1 baseline): cross-core flag IDs `FLAG_CANON_DONE=0`, `FLAG_MM1_DONE=1`, `FLAG_SOFTMAX_DONE=2`, `FLAG_MM2_DONE=3` are all in the 0..3 range that the cube-internal pipe sync at `event_t(0)` would collide with.
  - **3_FusionAttention kw-5 cycle (iter 4, 2026-05-21T~14:00Z) — "Use IDs ≥ 4" Fix EMPIRICALLY FALSIFIED**: tested 3 distinct sync schemes — (a) raw `event_t(2,3,4)` for mm1 + `event_t(5,6,7)` for mm2 (distinct IDs ≥ 2, dodging cross-core 0/1); (b) `GetTPipePtr()->FetchEventID(HardEvent::X)` runtime-allocated (canonical pattern). BOTH produce the same silent-hang signature as `event_t(0)` — kernel enqueues cleanly, `torch.npu.synchronize()` hangs past 45s with no aicore exception. The "low IDs collide with cross-core" hypothesis (and thus the Fix proposal of "use IDs ≥ 4") has been EXPERIMENTALLY DISPROVEN at the case_3 [4,64,512] BSH fp16 head=8 shape. The deadlock is at a deeper layer than event-ID allocation — root cause hypotheses now open for fo: (1) `CrossCoreSetFlag<0x2>` chain imposes barriers that collide with `HardEvent::M_FIX` regardless of event ID (cross-core semantics are uniform per HardEvent class); (2) `MIX_AIC_1_2` requires uniform Cross* sync across ALL pipe events on the cube side, not local SetFlag/WaitFlag mixed with CrossCoreSetFlag for cross-core; (3) FFTSCNT mailbox semantics may prevent any cube tile-MMAD with internal sync regardless of event ID scheme. **Implication for Fix section above**: the "Use distinct event IDs ≥ 4" Fix is a HYPOTHESIS that closes the visible `event_t(0)` collision but does NOT close the actual deadlock; the deeper sync-discipline question is fo-scope. Pattern A on V220 MIX_AIC_1_2 with user-owned cube-internal pipe sync remains UNSOLVED in canonical KB. Mitigation: stay on AIV-only VEC fallback for fp16 FA on V220 until canonical V220 cube workflow lands (likely via `aog-cann-learner` Mode 5 extraction of CANN ops_transformer arch22 `flash_attention_score` kernel structure).
  - **RESOLVED for V351 cross-core direction (2026-06-03, cann-learn Mode 5)**: the V351/A5 cube↔vec cross-core deadlock (the `MIX 1:1` AIC↔AIV handshake, distinct from the intra-AIC cube-internal pipe sync this PB is named for) is now closed as a RUNNABLE public-API pattern in `fa_class/cross_core_sync.md` §4. The deadlock-avoiding handshake is: SYNC MODE 4 (1:1, AIV0/AIV1 individually triggerable) + per-sub-block disjoint flag ids (`id` and `id+16`) + direction-pinned literal pipe + Set-after-own-write. Verdict: PUBLIC-API-runnable — the FA whole-port's working sync is customer-reproducible by hand (no privileged vendor class required). NOTE this resolves the **cross-core** edge; the separate **intra-AIC cube-internal** pipe-sync deadlock (the `event_t` collision this PB is primarily about) is still governed by the falsification above when user-owned tile-MMAD is used.
- **Cross-reference**: PB-34 (the other end of the cube+vec sync minefield — Matmul library vs user-owned flags); CAND-FA1 (Pattern A recommendation; pre-PB-35 anchor in CAND-FA1 used `event_t(0)` in pseudocode — needs amendment to `event_t(4..7)` after PR lands); `ascend950pr.md` § Cross-core sync (user-owned ID range `0..7`, reserved barrier IDs at `8..10`); 3_FusionAttention workspace knowledge_update.md Finding 14 + Finding 16; `patterns/unverified/candidates.md` CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP (the open-hypothesis follow-up candidate).

### PB-36: [ARCHIVED/DEPRECATED 2026-05-22] `DataCopy` GM->UB with non-zero `srcStride` for BSH->BNSD layout canonicalize on V220 produces wrong output [V220, layout-canonicalize, DataCopy-stride]

**ARCHIVED 2026-05-22 — SUPERSEDED by CAND-FA-CANON-FREE.** The original entry framed this as a V220 hardware/DataCopy bug + recommended a Python-side `torch.reshape().permute().contiguous()` workaround as the "Preferred pattern." Both framings were wrong:

1. **CANN's own `aclnnFlashAttentionScoreV2`** runs on the same V220 / CANN 9.0.0 hardware and produces correct output for BSH/SBH/BSND/BNSD without any such workaround — verified directly on A3 (exit=0, math correct, output 256.0). If this were a hardware bug, CANN's FA would fail too. It doesn't, so the bug was in our kernel's design choice (using an AIV canon stage at all), not in the underlying hardware.
2. **The recommended "Python-side reshape" workaround was a No-Delegation rule violation**: `torch.reshape().permute().contiguous()` on a `.npu()` tensor dispatches to torch_npu → CANN `aclnnPermute` / `aclnnContiguous`. The 0.45× CANN perf claim it produced was a misleading hybrid pipeline benchmark, not a kernel-vs-kernel comparison.
3. **The structural fix** (CAND-FA-CANON-FREE) eliminates the AIV canon stage entirely: mm1/mm2 read strided source GM directly via `MatmulImpl::SetOrgShape` 5-arg variant with `orgN/orgKa/orgKb = sS` (physical row stride), matching CANN's own implementation. Hits 0.603× CANN on the same shape, pure AscendC, no delegation.

**Audit trail (per safety rule 5: keep deprecated entries for history; do not delete)**: the original body is preserved below for context. Do NOT cite PB-36 as a current pattern; cite CAND-FA-CANON-FREE instead. Cross-ref: `docs/design/FA_CLASS_DESIGN_NOTES.md#fa-canon-removal-structural-rewrite` for the structural rewrite design + verification numbers.

---

**Original (now-archived) body:**

`applies_to: soc=Ascend910_9382 (V220 A2/A3); cann=9.0.0; op_class=fp16 layout canonicalize via DataCopy with non-zero srcStride for strided GM gather (BSH/SBH/BSND -> BNSD)`
`verified_on: a5_ops:3_FusionAttention kw-5 (2026-05-22, A3 npu-a3-test) — FaCanonicalizeQKVKernel::CopyOne with DataCopyParams{blockCount=S, blockLen=D*sizeof(T)/32, srcStride=(H-D)*sizeof(T)/32, dstStride=0} produces wrong output on BSH path (max_abs=0.055 vs CANN aclnn ground-truth) while BNSD path (srcStride=0) is correct (max_abs=3e-5). Falsification chain rules out infrastructure causes: (1) CANN's own aclnnFlashAttentionScoreV2 exit=0 on A3 (proves CANN runtime + driver OK); (2) torch_npu.npu_fusion_attention PASS on A3 (proves Python-aclnn bridge OK); (3) 1_BatchMatmul AscendC build PASS max_abs=1.5e-5 on A3 with default config + isTransposeB=true + 64x64x64 sub-tile (proves NPUKernelBench build pipeline + cube path infrastructure OK); (4) FA BNSD path with canon-skipped passes; (5) FA BSH path with canon-active fails. Bug is therefore confined to this kernel's strided-DataCopy path.`
`unverified_on: V351 (Ascend950PR / A5) — peer A5 agent reports same Python-side BSH<->BNSD workaround on case_3 [4,64,512] head=8 fp16 gives max_abs=1.54e-2 (above T1=1e-2); separate cross-arch precision divergence (500x worse than A3 with same workaround), root cause TBD. Open question: does the canon stride bug exist on V351 too, OR is canon correct on V351 but the V351 cube path itself has a different precision issue?`

- **Severity**: HIGH (silent wrong output; verification can miss without ground-truth comparison since output is plausibly-scaled noise; canon AIV stage fires on EVERY layout != BNSD, which is the common case for FA users — `BSH`, `SBH`, `BSND` all transit through this code path)
- **Symptom**: Kernel function with internal AIV layout canonicalize (BSH -> BNSD via `DataCopy` with row-gather `srcStride`). Build clean. Runs to completion, no fault. Output has structurally correct shape, but values diverge from ground truth at ~5e-2 level (full output scale, above T1=1e-2). Direct probe of canon output: head 0 produces all-rows-identical output (canon reads same source row repeatedly); head 1 has zero rows at scattered `s` positions.
- **Root cause hypothesis** (unconfirmed; KB carries the EMPIRICAL fact + workaround): MTE2 engine on V220 CANN 9.0.0 mishandles `srcStride` field semantic for certain GM->UB `(blockCount, blockLen, srcStride)` tuples — either ignored (-> same row read N times) or misinterpreted (-> skipped rows). Possibly related to PB-22 (V220 MTE2 DataCopy 32B transfer limit per destination TBuf) or PB-9 (V220 UB->UB DataCopy nuances). Pinning the exact failure condition (which `srcStride` values trip the bug at which `blockCount`/`blockLen` combos) is a follow-up probe job — not a blocker for the workaround.
- **Fix / workaround**: Move layout canonicalization to the Python side (BSH -> BNSD via `torch.reshape().permute(0,2,1,3).contiguous()`), pass already-BNSD tensors to the kernel, kernel skips canon. After kernel returns, transpose output back. Costs one extra alloc + copy on Python side. Measured at 0.45x CANN on S=64, D=64, N=2 BSH fp16 (181 us vs CANN 82 us) — 10x improvement over kw-1 VEC fallback baseline (0.046x CANN), still below 0.6x target but a real, mergeable improvement. PERMANENT in-kernel fix (future-kw scope) options: (a) replace strided `DataCopy` with row-by-row scalar loop, (b) try `DataCopyExtParams` (V220 extended variant) — needs probe, (c) move canon to AIC side with `Nd2NzParams` (different DMA engine path).
- **Anti-pattern (DO NOT EMIT on V220)**:
  ```cpp
  // BAD on V220: GM->UB DataCopy with non-zero srcStride from BSH layout
  DataCopyParams p;
  p.blockCount = S;                              // gather S rows
  p.blockLen   = (D * sizeof(T)) / 32;           // each row D fp16 elements
  p.srcStride  = ((H - D) * sizeof(T)) / 32;     // non-zero (=4 for H=128,D=64,fp16) -> wrong output
  p.dstStride  = 0;
  DataCopy(ubBuf, gmBSH[off], p);                // silently broken on V220
  ```
- **Preferred pattern** (Python-side layout transpose + kernel sees only BNSD):
  ```python
  # Python (model_new_ascendc.py forward path):
  if layout == "BSH":
      B, S, H = query.shape
      D = H // head_num
      N = head_num
      q_bnsd = query.reshape(B, S, N, D).permute(0, 2, 1, 3).contiguous()
      # k, v same; pass q_bnsd / k_bnsd / v_bnsd to kernel with kern_layout="BNSD"
      attn_out_bnsd = _ext.run_fusion_attention(q_bnsd, k_bnsd, v_bnsd, N, "BNSD", scale)
      attention_out = attn_out_bnsd.permute(0, 2, 1, 3).contiguous().reshape(B, S, H)
  # Mirror handling for SBH / BSND. Kernel always sees BNSD; canon AIV stage is no-op.
  ```
  ```cpp
  // GOOD: contiguous BNSD DataCopy in kernel (no strided gather needed)
  DataCopyParams p;
  p.blockCount = 1;
  p.blockLen   = (S * D * sizeof(T)) / 32;
  p.srcStride  = 0;
  p.dstStride  = 0;
  DataCopy(ubBuf, gmBNSD[off], p);
  ```
- **Evidence**:
  - 3_FusionAttention kw-5 cycle 6 (2026-05-22T03:50Z, A3 npu-a3-test): Direct probe (`fa_canon_dbg.py`) on BSH input revealed head0 all-rows-identical output, head1 zero rows at s in {8,16,24,32,40} — the canon stage was reading source row 0 for every destination row of head 0, and skipping source rows entirely for head 1. BNSD-direct path on same numerical input (no canon) showed max_abs=3e-5 (T1 PASS).
  - Falsification chain (lets us localize the bug to *this* kernel's `srcStride` use and not infrastructure): `/tmp/test_fa.cpp` = CANN's own `test_aclnn_flash_attention_score.cpp` compiled on A3, ran exit=0 with correct math (output value 256.0). `/tmp/bmm_test/` = 1_BatchMatmul AscendC archive copied to A3 + rebuilt, PASS max_abs=1.5e-5 with default config; PASS max_abs=0.0002 with isTransposeB=true; PASS max_abs=0.0001 with 64x64x64 sub-tile. Both proved CANN runtime + NPUKernelBench infra are NOT the problem.
  - Owner mid-session correction (2026-05-22): pushed back on initial "V220 MIX dispatch broken" / "V351->V220 cross-arch divergence" / "CANN version bug" framings — drove the CANN-aclnn + minimal-pybind11 probes that produced the falsification chain above. All three prior framings empirically falsified.
  - After workaround applied: FA cube path PASS_T1 with max_abs=3.05e-5, MARE=2e-4. Pass B 9/9 PASS preserved. Perf 0.45x CANN on B=1, S=64, N=2, D=64 BSH fp16 (181 us vs CANN 82 us) — 10x over kw-1 VEC fallback baseline (0.046x CANN), still below 0.6x target.
  - Peer cross-arch datapoint (A5 V351, 2026-05-22 from agent-main): same Python-side reshape workaround on case_3 [4,64,512] head=8 fp16 gives max_abs=1.54e-2 (above T1=1e-2) at ~1 ms. Cross-arch divergence at 500x max_abs — root cause separate from canon bug. Open hypothesis: V351 cube path itself has precision regression (possibly fp16 accumulator behavior, or unverified cube_eligible code path that canon was previously hiding).
- **Cross-reference**: PB-9 (V220 UB->UB DataCopy -> `Adds` identity workaround; same V220 MTE-engine family of nuances but different direction); PB-22 (V220 MTE2 DataCopy 32B transfer limit per destination TBuf); related in-tree code: `workspace/3_FusionAttention/kernel/fusion_attention_kernel.h::FaCanonicalizeQKVKernel::CopyOne` (line ~747); `workspace/3_FusionAttention/kernel/fusion_attention_kernel.h::FaUncanonicalizeKernel` (mirror operation — `DataCopy` with `dstStride` to write BNSD output back as BSH; likely same bug, currently also bypassed by Python output-reshape) — follow-up CAND needed to confirm.

---

### PB-38: Auto-generated host_stub.cpp `FreeAscendMemDevice` races with async kernel execution on V220 [V220]

`applies_to: soc=Ascend910_9382; cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend910_9382 (V220); cann=9.0.0`
`unverified_on: soc=Ascend950PR (A5 V351 — LaunchAscendKernel may be synchronous on V351)`

- **Severity**: HIGH (silent data corruption on multi-launch)
- **Status**: CONFIRMED 2026-05-21 12_Permute a3-ds kw-3
- **Symptom**: First kernel launch produces bit-exact output; subsequent launches show max_abs_diff 5-7 for fp32 tensors with random values.
- **Root cause**: Auto-generated `host_stub.cpp` calls `FreeAscendMemDevice(overflow_buf)` immediately after `LaunchAscendKernel`, but kernel executes asynchronously. On subsequent launches the same physical memory may be reused while previous kernel still references it.
- **Fix**: Call `torch.npu.synchronize()` after each kernel launch in pybind wrapper.
- **Cross-ref**: OL-66 (torch::zeros not stream-ordered — same class of stream-safety issue).

---

### PB-39: bisheng `--enable-simt` codegen — `select i1` on `dav-c310-vec` fires `Copy register different width` [V351/A5, --enable-simt-only]

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=2026-04-03+ through 2026-04-30; mode=--npu-arch=dav-3510 --enable-simt`
`also_affects: cann=9.1.0-beta.1 — partial fix only`
`verified_on: bisheng 2026-04-03 (CANN 9.0.0), 2026-04-28 (per user report), 2026-04-30 (CANN 9.1.0-beta.1)`

- **Severity**: HIGH for any code path emitting `--enable-simt` IR (generated pure-SIMT ports, hand-written pure-SIMT kernels, possibly some compiler-generated SIMT lowering paths). NOT relevant to ordinary AscendC operator development that uses SIMD intrinsics + `Select<T>()`.
- **Status**: OPEN. CANN 9.1.0-beta.1 ships a narrow codegen patch that handles the simple multi-`select`-with-shared-cond pattern, but kernels with richer i1 fan-out still trip.
- **Symptom**: `fatal error: error in backend: Copy one register into another with a different width` from `bisheng --npu-arch=dav-3510 --enable-simt`. Larger kernels (5K+ lines of generated SSA) segfault inside `HiTPE DAG->DAG Pattern Instruction Selection` instead — same defect reached one pass earlier.
- **Trigger**: LLVM `select i1 %cond, T %a, T %b` instructions where the i1 predicate is routed through a width-mismatched register MOV during SDAG→MachineInstr lowering. Multiple selects sharing one i1 condition (the SROA'd form of an aggregate ternary) trigger reliably for most result-type/fan-out combinations.
- **Secondary bug — surfaces only after the primary is patched**: `fatal error: error in backend: MaxThreads out of range!` — backend requires `!"simt-max-threads", i32 2048` annotation on every function in the module that holds (transitively) a 3-arg scoped atomic intrinsic.
- **Workaround**: 6-step IR-rewrite pipeline. Reproducible end-to-end from `gitcode.com/example/bisheng-crash-repro-cann-9` commit `0e89c0d` (private; share key on request). The pipeline:
  1. **Source**: change `&&` → `&` in `assert(...)` calls in any `array.h`-style bounds-check header (eliminates `select i1, i1, i1` from short-circuit AND).
  2. **Emit O3 LLVM bitcode** from the source: `bisheng --npu-arch=dav-3510 --enable-simt -DWP_ENABLE_ASCEND -O3 -emit-llvm -c <src>.asc`
  3. **Disassemble**: `bisheng -cc1 -triple hiipu64-hisilicon-cce -x ir -S -emit-llvm <bc>` → text IR.
  4. **IR rewrite**: replace each `select i1 %c, T %a, T %b` with a `br + phi` diamond, preserving SSA dominance (relabel every successor-phi predecessor reference from the original block label to the new tail-block label; entry-block's implicit numeric label is the param count). For `T==i1` widen through `zext i1→i32 / phi i32 / icmp ne` to avoid emitting `phi i1` (also broken).
  5. **IR metadata**: append `!annotation !<id>` (id → `!{!"simt-max-threads", i32 2048}`) on every function definition and matching `{ptr @<fn>, !"simt-max-threads", i32 2048}` entries in `!hivm.annotations`. Brute-force every-function form needed — annotating only obvious helpers leaves siblings unannotated.
  6. **Compile + link**: `bisheng -cc1 -triple hiipu64-hisilicon-cce -x ir -O0 -emit-obj` (-O0 mandatory — any -O1+ pass re-folds the diamonds back into `select i1`), then `ld.lld -shared -Bsymbolic` (-Bsymbolic needed because `g_sysSimtPrintFifoSpace` is weak and the backend emits PC-relative relocs for it).
- **Source-level workarounds that DON'T work**: per-component scalar select, `if/else` + out-param, `-fno-vectorize -fno-slp-vectorize`, `-DNDEBUG`, `__attribute__((noinline))/((optnone))/((flatten))`, packed struct, raw float pointer instead of struct, `volatile` tmp. All hit the same backend error because the optimizer re-fuses to the buggy pattern before codegen.
- **Evidence**:
  - Minimal repro: 30 lines, no Warp deps, parameterised on `SLEN` for struct length. CANN 9.0.0 N=1/3/5/6/7/8 FAIL at -O3; CANN 9.1.0 all N OK.
- **Recommended report to Huawei**: minimal repro `repro_min.asc` (30 lines, no proprietary deps) is sufficient to drive the bisheng team's diagnosis. The secondary "MaxThreads" diagnostic should also be improved to name the offending function.

---

### PB-40: `RegisterAscendBinary mix ret 107000` printed at teardown on `KERNEL_TYPE_MIX_AIC_1_2` V220 — non-fatal, independent of matmul-primitive [V220, mixed-mode-binary-register]

`applies_to: soc=Ascend910_9382 (V220 / A3); cann=9.0.0; kernel_type=KERNEL_TYPE_MIX_AIC_1_2`
`verified_on: npu-a3@198.51.100.92 (NPU 0 idle), bisheng/910C, CANN 9.0.0, 2026-05-27 independent prototype firsthand`
`unverified_on: V351/A5 (not tested)`

- **Severity (UNDER REVIEW 2026-05-27)**: initially documented LOW based on "kernel produces output before 107000". independent prototype blackbox D-sweep 2026-05-27 02:22Z refuted that — at D≥64 main shapes `cand_absmax=0.000` (output ZERO). Then 02:27Z independent prototype surfaced **target mismatch**: the FSM-emit kernel is A5/V351-targeted (per `PROGRESS.md` + UB-budget 248KB + `kernel.h:114` "A5 V351 cross-core" comment) and was being tested on A3/V220 — i.e. all V220-side observations are running an A5-incompatible binary on V220 hardware. A5-side re-verify (02:41Z) on `Ascend950PR_957b` (correct target) showed the SAME kernel **deadlocks** (kernel launch >60s, no return, PYEXIT=124). So on V220 the kernel fails-to-compute (this PB's 107000), on A5 it deadlocks (separate failure). **Open question**: is 107000 a real V220 platform bug for multi-entry `MIX_AIC_1_2`, OR is it V220's loader correctly rejecting an A5-format binary? Cannot distinguish without testing a V220-CORRECT multi-entry `MIX_AIC_1_2` kernel on V220. Severity stays UNDER REVIEW pending that disambiguating run.
- **Symptom**: stdout (NOT stderr) prints exactly one line `RegisterAscendBinary mix ret 107000` AFTER the kernel's `RAN_OK latency_ms=...` + output comparison. A bare `[ERROR]` line with no message follows. No `aiv ret 0` / `aic ret 0` companion success lines. Kernel still produces output (numerical correctness of that output is a separate concern).
- **Trigger**: A `.cpp` file contains TWO or more `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` entry-pairs where only ONE is invoked at runtime (e.g. `fusion_attention_custom_fp16_nopse` + `fusion_attention_custom_fp16_pse`, dispatched by `has_pse` in pybind; case0 uses `pse=None` → only `nopse` entry called → `pse` binary's deferred registration fails at teardown).
- **Cross-ref**: PB-28 (V220 + `KERNEL_TYPE_AIV_ONLY` + fatal-never-register). Scope distinct: PB-28 is `AIV_ONLY` + fatal-never-register; PB-40 is `MIX_AIC_1_2` + non-fatal deferred-register-at-teardown.
- **Independence from matmul-primitive**: confirmed across attempt-2 (Matmul-lib×3 + Mmad×5 emit) and attempt-3 (manual Mmad-only emit, post PR #191 `83f0bf0a`). Both produced the identical 107000 signature on the same V220 host. Therefore NOT coupled to cube primitive choice — it is a mix-binary registration property.
- **Detection**: stdout grep `RegisterAscendBinary mix ret 107000` on V220 builds containing multiple `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` entries. Pre-build heuristic: `grep -cE "KERNEL_TASK_TYPE_DEFAULT\(KERNEL_TYPE_MIX_AIC_1_2\)" workspace/<op>/kernel/*.cpp` returning > 1 with same-architecture entry pairs flags the risk.
- **Recommended action (proposed, NOT verified)**:
  1. Emit only ONE `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` per `.cpp`; collapse `_nopse` / `_pse` codepath divergence via runtime conditional inside a single entry, OR
  2. Split into multiple `.cpp` files (one entry-pair each) so the unused binary isn't co-loaded with the used one.
  Owner verification still required before promoting either to a hard rule.
- **Evidence**: independent prototype 2026-05-27 A3 firsthand on FSM-emit 3_FusionAttention attempt-3 (post PR #191 `83f0bf0a`). Disk artifact: `workspace/3_FusionAttention/kernel/fusion_attention_fp16.cpp` (two `MIX_AIC_1_2` entries). Timing trace: `RAN_OK latency_ms=79.572` → output emitted → `RegisterAscendBinary mix ret 107000` → bare `[ERROR]`. Discord context: 2026-05-27 01:32Z (timing trace) + 02:17Z (scope-2 independence verdict after matmul-primitive change).
- **Other instances (predicted)**: any V220 archive that registers >1 entry-pair of the same `MIX_AIC_*` type in a single .cpp where some entries are unused per dispatch — generic to mixed-mode + multi-entry-per-source-file emission patterns.

---

---

### PB-41: Mmad/HAVE_WORKSPACE kernel — host pybind MUST allocate sysWs + userWs (kernel gets `GetUserWorkspace(w)=w+sysWsSize`) [V220, workspace, matmul]

`applies_to: soc=Ascend910_9382 (V220 / A3); cann=9.0.0; bisheng=n/a; op_class=all (any Mmad/cube kernel)`
`verified_on: soc=Ascend910_9382; cann=9.0.0`
`unverified_on: soc=Ascend950PR (V351 / A5 — same auto_gen wrapper expected, untested)`

- **Severity**: HIGH — silent data corruption (no error, no hang), produces garbage / nan / degenerate output that masquerades as a precision/algorithm bug.
- **Mechanism**: when a kernel uses `Mmad` (cube), the auto_gen / `KERNEL_TASK_TYPE` wrapper enables `HAVE_WORKSPACE` and the device entry computes the kernel-visible user workspace as `GetUserWorkspace(rawWorkspace) == rawWorkspace + sysWsSize` — it reserves a **system-workspace prefix** (FFTS / matmul scratch) ahead of the user region. If the HOST (pybind) allocates only `userWs` bytes, the kernel's user-workspace window `[w+sysWsSize, w+sysWsSize+userWs)` runs OFF THE END of the buffer → overwrites unrelated GM / reads uninitialized GM → corrupt cube↔vec workspace exchange (gather/scores/dscores/dgk all garbage).
- **Detection**: cube kernel produces structurally-wrong output (huge-finite for bf16 / nan for fp16) on EVERY case, while the same kernel's non-Mmad paths look fine; the magnitude is "uninitialized memory" not "slightly off". Pre-check: kernel calls `Mmad` AND pybind allocates `at::zeros({userWs})` without a `sysWs` term.
- **Fix**: allocate `sysWs + userWs` on the host. Get `sysWs` from `PlatformAscendCManager::GetInstance(soc)->GetLibApiWorkSpaceSize()` (the precise reserve), or over-allocate a safe constant — **16 MiB is a safe V220 over-alloc** for FFTS+matmul. Over-allocation is harmless; under-allocation corrupts.
  ```cpp
  const uint64_t sysWs = 16ull * 1024 * 1024;   // V220 FFTS+matmul system reserve
  uint64_t totalWs = sysWs + userWs;            // NOT just userWs
  at::Tensor w = at::zeros({(int64_t)totalWs}, at::device(kPrivateUse1).dtype(at::kByte));
  ```
- **Evidence**: lightning_indexer_grad (generated AscendC kernel, A3, 2026-05-27) — cube↔vec LIG-backward kernel; allocating only userWs gave garbage/degenerate dq/dk/dweights; adding the 16 MiB sysWs prefix moved precision 3/38 → 12/38 (and removed all garbage/huge values), isolating the remaining failures to genuine compute bugs.
- **Other instances (predicted)**: any cube/matmul AscendC kernel with a user-managed GM workspace (FlashAttention, GroupedMatmul, fused norm+matmul, MoE finalize, any two-stage cube↔vec handoff). The bug is generic to `HAVE_WORKSPACE` + host-allocated workspace, not LIG-specific.
- **A5/V351 CONFIRMED + nuance (FlashAttention-A5, 2026-06-02 — upgrades the `unverified_on: Ascend950PR` line above for the custom-`<<<>>>`-launch case)**: on a hand-written launch (pybind + `<<<>>>`, no aclnn/GE framework) the framework's pre-op workspace registration is ABSENT, so the kernel-visible `GetUserWorkspace(workspace)` returns garbage and cube GM-staging writes OOB → `507015`. Two A5-specific facts beyond the V220 entry:
  - **`GetUserWorkspace(workspace)` IGNORES its argument** — it returns the GLOBAL `g_sysWorkspaceReserved + RESERVED_WORKSPACE` (`RESERVED_WORKSPACE = 16 MiB` on arch35; on dav-c310/`__NPU_ARCH__==3510` the base resolves via `GetSysWorkSpacePtr() = __get_kfc_workspace_addr()`). The launch MUST set that global, not merely pass a sized buffer.
  - **`SetSysWorkspace` is `[[deprecated]]` and CONDITIONAL** (`if (g_sysWorkspaceReserved == nullptr)`) → silent no-op if the global is already set or the call is optimized away → global stays nullptr → `GetUserWorkspace` returns `nullptr + 16MB = 0x1000000` garbage → OOB `507015`. The custom launch must call **`AscendC::SetSysWorkspaceForce(workspace)`** (unconditional) before `GetUserWorkspace`, with the workspace sized `data + 16MB`.
  - **Manifestation split**: the MULTI-CORE FFTS cross-core deadlock (≥2 MIX groups, no registered sys-scratch base → hang at `synchronize()`) is **DS-confirmed** (FA-A5 multi-core CASE14 9.9s RC=0 after the fix; independent --clean build, device.o recompile VALID). The single-core large-D GM-staging OOB recovery via `SetSysWorkspaceForce` is **provisional** (independent prototype FA-A5 large-D, pending DS corroborate — see candidates.md `CAND-FA-A5-KFC-WORKSPACE`).
  - **Diagnostic**: a D≤128 (UB-resident) path never dereferences the user-workspace pointer → it passes even when the global is unset; only D>192 (GM-staging) or multi-core (FFTS) exposes the missing registration. A subset passing while large-D / multi-core crashes is the smoking gun.

---

### PB-42: V220 cube/AIC GEMM must NOT Fixpipe directly to the fp16/bf16 OUTPUT tensor — route through an fp32 workspace + a vec cast [V220, fixpipe, matmul]

`applies_to: soc=Ascend910_9382 (V220 / A3); cann=9.0.0; bisheng=n/a; op_class=all (any cube GEMM writing a low-precision output)`
`verified_on: soc=Ascend910_9382; cann=9.0.0`
`unverified_on: soc=Ascend950PR (V351 / A5 — untested)`

- **Severity**: HIGH — wrong output values (not a crash); the affected output is the only one routed cube-direct, so it fails while sibling outputs pass, which misleads diagnosis toward the GEMM math.
- **Mechanism**: a cube/AIC `Fixpipe` from the fp32 L0C accumulator straight into a **fp16/bf16 OUTPUT GM tensor** produces wrong output on V220. Routing the same L0C through an **fp32 WORKSPACE** Fixpipe (fp32→fp32) is correct; a subsequent vec/AIV stage then `DataCopy`s the fp32 workspace into UB, `Cast`s to the low-precision dtype, and `DataCopy`s to the output. This mirrors cv-agent FlashAttention BMM2, which Fixpipes O into an fp32 `oSlot` workspace and lets the vec write the final output — never cube-direct to the output.
- **Diagnostic fingerprint**: among multiple cube outputs, the ONE written by a cube-direct fp32-L0C→fp16-output Fixpipe is wrong while outputs routed via fp32 workspace are correct.
- **Fix**: make EVERY cube Fixpipe target an fp32 workspace; write EVERY low-precision output from a vec/AIV cast stage (uniform structure). Add a `ws_<out>` fp32 region; cube does `Fixpipe(wsOutGm, cL0, fp)` (fp32→fp32); vec does `DataCopy(ub_f32, wsOutGm); Cast(ub_half, ub_f32, CAST_ROUND); DataCopy(outGm, ub_half)`.
- **Evidence**: lightning_indexer_grad (A3, 2026-05-27) — `dq` (the only output written by a cube-direct fp32→fp16 Fixpipe) was wrong while `dk`/`dweights` (fp32-workspace + vec-cast) were exactly correct; re-routing `dq` through an fp32 `ws_dq` + a vec `CastDq` stage moved precision 12/38 → 30/38.
- **Other instances (predicted)**: any V220 cube op whose final output is fp16/bf16 and is currently Fixpipe'd straight from L0C — attention scores/outputs, matmul-epilogue casts, fused GEMM+activation low-precision stores.

### PB-43: V220 manual-cube operand-load forms COMPILE + run clean on A5 but compute garbage — build-success is NOT A5-validation [V351/A5, port_a3, cube, silent-wrong-result]

`applies_to: soc=Ascend950PR (V351 / A5); cann=9.0.0; bisheng=n/a; op_class=all (any cube op that hand-builds L0A/L0B operand loads with V220 fractal addressing)`
`verified_on: soc=Ascend950PR_9579; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (where these forms are CORRECT — this bug is the A5-side non-equivalence, not a V220 bug)`

- **Severity**: HIGH — silent wrong result, NOT a crash or compile error. The kernel builds, the `.so` links, the kernel launches and returns success; the output is simply wrong. Nothing in the build/run signal warns you.
- **Symptom**: a port_a3 cube kernel whose operand load into L0A/L0B is written with a **V220 manual fractal-addressing form** — either (a) a **per-M-fractal `LoadData2DParams` (V1) `i`-loop** (`startIndex=i`, `dst=aL0[C0*i*kAligned]` advancing one 16-row fractal per iteration), or (b) a **3D im2col helper `LoadNzL1ToZzL0A`** with a manual `colC0Stride` — produces RUNTIME GARBAGE on A5. Observed on FlashAttention: attn `max_rel ~470000×`, `sm_max`/`sm_sum` all FAIL. The magnitude is roughly right (K-contraction accumulation IS happening) but the values are wrong (operand fragment rows/stride are arranged wrong in L0).
- **Mechanism**: the V220 cube fractal-addressing semantics (the per-fractal `startIndex` advance + the Zz destination placement) are **not equivalent on the A5 cube**. The same source that is correct on Ascend910/V220 reads operand fragments from the wrong rows/stride on A5 — a layout non-equivalence between arch22 and arch35 cube load paths, not a math/accumulation error.
- **Detection trap (the load-bearing lesson)**: **build-success ≠ A5-validation.** Because there is no compile error and no runtime error, a worker who only checks "did it build / did it run" will wrongly conclude success. The bug is ONLY caught by an actual numerical `pass_a` on A5 hardware. Worse, **controlled-input probes partially mask it**: identity/one-hot operand probes (e.g. K=identity, single-`d` one-hot) often pass or mostly-pass (the wrong-row arrangement happens to coincide for sparse inputs), while **dense random-signed inputs go full-garbage**. Do NOT trust a clean controlled-probe as validation — the arbiter is dense-input pass_a on hardware.
- **Fix**: see **OL-197** (the resolution half) — replace the manual per-fractal / 3D-helper form with the **arch35-native single 2D `LoadData2DParamsV2`** mStep-encoded load (`mStartPosition=0`, `mStep=ceil(M/16)`, `kStep=GetBlockNum<T>(K)`, `srcStride=dstStride=mStep`; no `colC0Stride`); **B-operand `ifTranspose=!isRightTranspose`** (the negate is load-bearing). Convert the `kRemain>0` / `D%BASE_K≠0` tail path too (it carries the same V1 i-loop and stays dormant for D=128 / mult-16 shapes).
- **Evidence**: flash_attention_score port_a3 on Ascend950PR_9579 (2026-05-29) — pre-fix attn 0/8 (garbage `max_rel ~470000×`), sm_max/sm_sum all FAIL; after the OL-197 2D-`LoadData2DParamsV2` rewrite at BOTH cube sites (MM1 QK^T + MM2 PV), attn 8/8 (`max_abs 2.4e-4`), sm_max 8/8, sm_sum 8/8 (origin/main `45fdc7c0`). Root cause was reached after 5+ empirically-refuted hypotheses arbitrated on A5 hardware (zero wrong fixes shipped). Cross-ref `CAND-V220-V351-FA-DIFF-1` (the V220-monolithic vs V351-per-engine structural-port companion).
- **Other instances (predicted)**: any port_a3 cube op that hand-writes a V220 manual `LoadData2D` operand load into L0A/L0B and is "ported" to A5 by only stripping the `__CCE_AICORE__==220` gate without converting the load form — non-FA GEMM-family ports, fused cube+vec ops, backward cube kernels. The general guard: a V220→V351 cube port that compiles is unvalidated until dense-input pass_a runs on A5.

### PB-44: `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` → `RegisterAscendBinary aiv 107000` register-FAIL on A5/950PR under the CANN 9.0.0 toolkit; CANN 9.1.T500 registers clean — A5-safety is toolkit-version-gated (refines PB-28) [V351/A5, cann-version, kernel-registration, refines-PB-28]

`applies_to: soc=Ascend950PR (V351 / A5); cann=9.0.0 (107000 register-FAIL) / 9.1.T500 (clean); macro=KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`
`verified_on: soc=Ascend950PR_957b; cann=9.0.0 (107000) AND cann=9.1.T500 (matched) — clean single-variable (CANN-only) A/B`

- **Severity**: HIGH — build/link/compile PASS (bisheng OK), but launch-time `RegisterAscendBinary aiv ret 107000` → program register failed → Status FAIL. Kernel never registers/runs; end-to-end blocked. No degraded-mode workaround under 9.0.0.
- **Status**: CONFIRMED 2026-06-16 by a clean single-variable A/B on npu_dev3 (.171); both arm logs read independently, same-kernel md5 independently verified.
- **Refines PB-28**: PB-28 states this macro is "arch35-only ... does NOT apply on Ascend950PR (where the macro is the canonical entry-form)." That A5-safety is **toolkit-version-gated**: under the **CANN 9.0.0** toolkit on 950PR the macro STILL fails registration with `107000`; under **CANN 9.1.T500** it registers cleanly (the canonical-on-A5 behaviour PB-28 describes). So "legal/canonical on arch35" is NOT the same as "the 9.0.0 toolkit's 950PR registration loader can register it." No conflict with PB-28 — its A5-clean claim holds for 9.1.T500.
- **Symptom**: an A5/950PR kernel using `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` builds clean but launch fails `RegisterAscendBinary aiv ret 107000` (program register failed) when built against the CANN 9.0.0 toolkit. A downstream `507034` vector-core-timeout may follow — it is a **consequence** of the failed registration, NOT an independent cause. Rebuilding the SAME kernel against CANN 9.1.T500 → registers, runs, all cases matched.
- **Evidence (clean single-variable A/B, main-verified 2026-06-16)**: same `4_Abs` AIV_ONLY kernel (`abs_kernels.cpp` md5 `1ff22805c8b6e52bff3f2c288cc47bfc`, md5 independently verified), fixed SOC `Ascend950PR_957b`, identical nsenter/env, ONLY the CANN toolkit flipped:
  - CANN 9.0.0 → `RegisterAscendBinary aiv ret 107000` → register failed → Status FAIL (+ downstream `507034`).
  - CANN 9.1.T500 → all cases matched, MERE=0 / MARE=0 (50/50).
  - Arm logs: `/tmp/ab2_v_cann-*.log` on npu_dev3 (.171).
  - Surfacing: primary repro by the independent reviewer (agent-open); the version-direction lead came from the back agent noting its FAG `flash_attention_grad` AIV_ONLY kernels built/ran clean under 9.1.T500 (which pointed at the toolkit version). Note: that FAG observation only evidences "AIV_ONLY registers under 9.1.T500"; it is NOT part of the causal A/B (which is the single-kernel CANN flip above).
- **Fix / workaround**: build AIV_ONLY-macro kernels on A5/950PR against **CANN 9.1.T500** (present in the npu_dev3 container). If pinned to the 9.0.0 toolkit, fall back to the bare `__global__ __aicore__` entry-form (per PB-28's V220 fix) and re-verify.
- **Detection**: build PASS + launch-time `RegisterAscendBinary aiv ret 107000` on A5 → check the CANN toolkit version; if 9.0.0, rebuild against 9.1.T500.
- **Cross-ref**: PB-28 (same macro + `107000` signature on V220; THIS entry refines its "A5-safe" claim to toolkit-version-gated), PB-40 (`RegisterAscendBinary mix ret 107000` for MIX multi-entry on V220 — different trigger, same error code). DISTINCT from the GDN regbase `507015` aicore-trap (MIX cube/vector CrossCore sync on 9.1.T500) — that is a different failure mode, not this; see **PB-45**.

### PB-45: `TPipe::Reset()` frees the GLOBAL `g_tpipeImpl` event pool + buffer cursor on arch35 — a multi-stage MIX kernel that calls `pipe_->Reset()` between stages CANNOT carry persistent cross-call sync state across the Reset boundary [V351/A5, mixed-mode-sync, TPipe-Reset, multi-stage]

`applies_to: soc=Ascend950PR (V351/A5); cann=9.1.T500; bisheng=n/a; arch=arch35; op_class=multi_stage_mix_aic_aiv; macro=KERNEL_TYPE_MIX_AIC_1_2`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A2/A3 — PB-35 is the V220 sibling; the global-pool-free root cause is read from arch35 kernel_tpipe_impl.h and not confirmed identical on V220)`

- **Severity**: HIGH — two distinct SILENT failure modes (HANG and `507015` aicore trap), neither surfaced by build (compiles clean) and both easily misdiagnosed as an algorithm/layout bug.
- **The platform fact** (read from `kernel_tpipe_impl.h` on-container): `g_tpipeImpl` is a **GLOBAL singleton** — ONE event pool + ONE buffer cursor shared by ALL `TPipe` objects in the kernel (so a second `TPipe` instance does NOT isolate events/memory — rejects the "dedicated second pipe survives Reset" idea). `TPipe::Reset()` → `ResetPool()` (a) iterates ALL events setting `evt->eventOccupy = 0` — **frees every `AllocEventID`/`FetchEventID` id** — and (b) resets the L0/L1 buffer cursor (frees all TBuf memory).
- **Symptom / why it bites a multi-stage MIX kernel**: a kernel structured `Stage1 → SyncAll → Stage2 → SyncAll → Stage3` with each stage `Init → Process → pipe_->Reset()` (looped per group) frees the entire global event pool at every stage boundary. Any sync mechanism that holds **persistent state across the Reset** desyncs:
  - **Failure mode A — HANG**: a persistent `Buffer<>` double-buffer credit model (`Buffer::Init` does `AllocEventID` + a priming `SetFlag<EventC2P>`) leaves a set-but-unconsumed credit at stage end; `Reset()` frees that id WITHOUT draining the pending hardware flag, then the next stage re-primes on a recycled id → flag-counter double-count / a later `FetchEventID` grabs the still-pending id → set/wait desync → silent hang (sync/runner timeout, no fault). The `Buffer<>` persistent-credit model is INCOMPATIBLE with per-stage Reset.
  - **Failure mode B — `507015` aicore trap**: a hardcoded literal `EVENT_ID0` reused for all four cube fences (MTE2_MTE1 / MTE1_M / M_FIX / FIX_M) aliases the surrounding code's own dynamic `FetchEventID` fences; after a Reset recycles ids the literal collides with a managed id → the M unit consumes a half-loaded L0 descriptor → `507015` aicore exception on the first matmul.
- **Why library `MatmulImpl` is immune**: its L0 and events are KFC-managed and self-consistent across `Reset()` (it leaves no dangling user-visible priming credits in the shared pool), and its `Init` is called ONCE before the stage loop — not per-stage. (This is why the GDN **light-port** — MatmulImpl + OL-220 build recipe — runs 122/122; the **regbase** hand-rolled cube hits this bug.) A user-owned hand-rolled cube must be Reset-safe BY CONSTRUCTION (see OL-223).
- **Fix / workaround**: drive all cube-internal L0 fences with Reset-safe dynamic back-to-back `FetchEventID` Set+Wait — no persistent `Buffer<>` credits, no hardcoded ids (full rule + anchor in **OL-223**). Re-`InitBuffer` raw L0 TBufs per-stage (cursor is reset by Reset → re-alloc each stage is correct, not wrong).
- **Evidence**: GDN `chunk_gated_delta_rule` regbase whitebox (A5/V351 arch35, CANN 9.1.T500, 2026-06-16, kernel md5 `8b0b90cb`). Approach A' (hardcoded `EVENT_ID0`) → `507015` trap on case_0 first matmul; Approach B (persistent `Buffer<>` credits) → silent hang (timeout). NOOP+NOINIT bisection localized the hang to per-stage `mm.Init` Buffer/AllocEventID setup (NOT `mm.Run`); reading `kernel_tpipe_impl.h` confirmed `ResetPool()` frees the global event pool. Reset-safe rewrite → 0 hang, 0 trap, all 122 cases run clean, ~118/122 通过 vs fp64 oracle (the MIX cube↔vector sync wall, 0/122 → ~118/122). **Honest caveat (count NOT bit-stable)**: the "0 hang / 0 trap / all-122-run-clean" claim is solid and reproducible — that IS the MIX-sync fix. The *precision* count (~118) is NOT reproducible run-to-run: after the Reset/event-pool fix the M-tail cross-core cube↔vector handshake is still non-deterministic below event-id granularity (the irreducible PB-35 wall for a hand-rolled cube), so residual + exact pass-count fluctuate. The deterministic answer is a library: the GDN **light-port** (MatmulImpl + OL-220) runs 122/122; the **catlass** composition is deterministic ×3 (see `docs/design/FA_CLASS_DESIGN_NOTES.md#gdn-catlass-composable-primitives-design`). This entry's lesson — the Reset/global-pool root cause + the Reset-safe fix — is reading-verified (`kernel_tpipe_impl.h`) and holds regardless of the count.
- **Cross-reference**: PB-35 (the **V220** sibling — `event_t(0)` cube-internal sync collides with the cross-core flag chain; its "use IDs ≥4" fix was falsified on V220 and its V351 note covers only the **cross-core** handshake. PB-45 is the **V351 intra-AIC** counterpart and adds the *Reset/global-pool* root cause PB-35 lacks). PB-44 (the AIV_ONLY 107000 — its cross-ref already points here as the distinct `507015` mode). PB-34 (MatmulImpl + manual CrossCore deadlock, V220-only — confirmed NOT reproduced on A5: GDN light-port runs 122/122). OL-223 (the Reset-safe fix), OL-220 (the sibling GDN light-port build recipe), OL-197 (A5-valid 2D fractal load), OL-206 (prefer managed cross-core abstraction), cross_core_sync.md §4 (the cross-core-direction V351 RUNNABLE handshake).

---
### PB-46: torch_npu `.contiguous()` / `.copy_()` of a transpose-to-INNERMOST permute HANGS (wedges) the device — a permute that keeps the last dim is fine
<!-- applies_to_backend: all -->
`applies_to: soc=Ascend950PR_957b; cann=9.1.T500; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A2/A3 — not retested; this is a torch_npu/CANN-runtime layout-materialization behavior, may differ by CANN version)`

- **Severity**: HIGH — silent device wedge (no error code, no fault). The hung op leaves the NPU unusable for subsequent launches (D-state, unkillable), so it also masquerades as a downstream "kernel hang" on the same card (see OL-189 for the rotate-to-fresh-NPU diagnostic).
- **The runtime fact**: a 4D **middle-swap** permute such as `(0,2,1,3)` that KEEPS the last (innermost) dim materializes fine via `.contiguous()` / `dst.copy_(view)`. But ANY permute that moves a dim INTO the innermost position — e.g. `(0,2,3,4,1)`, `(0,2,3,4,5,1)`, or even a collapsed 3D `(0,2,1)` — HANGS both `.contiguous()` and `dst.copy_(view)`. The trigger is "innermost-stride change requiring a true transpose-materialize", not rank.
- **Symptom**: the pybind/host call to `.contiguous()` or `.copy_()` never returns; the device wedges. Easily misdiagnosed as a kernel bug because the next kernel launch on the same (now-wedged) card also hangs.
- **Consequence for port_a3**: a V220 generic kernel whose input layout is head-major / query-innermost CANNOT be produced by a pybind transpose into that layout. **Mitigation — author the A5 kernel to read the STANDARD (un-transposed) framework layout directly.** For MSDA: query-major — each core owns whole `(batch, query)` rows; `loc[b,q,:]` and `attn[b,q,:]` are contiguous, `value[b,key,h,:]` is element-contiguous with key-stride `nh*ed`. Reading the standard layout also removed SetAtomicAdd → deterministic by construction.
- **Evidence**: MultiScaleDeformableAttnFunction port_a3_to_a5 kw-1 (2026-06-20, A5/Ascend950PR_957b/CANN 9.1.T500). iter-2 hung on `attn (0,2,3,4,1)` + `loc (0,2,3,4,5,1)` transposes; rewriting the kernel to read the standard mmcv query-major layout (no pybind transpose) → 33/34 inclusive vs fp64 CPU truth, determinism 34/34, std median 80.9µs.
- **Other instances (predicted)**: any port that tries to pre-transpose framework inputs into a kernel-private innermost layout via `.contiguous()`/`.copy_()` — head-major attention layouts, channel-last↔channel-first conversions, any `permute` that lands a previously-outer dim last. Prefer authoring the kernel to consume the framework's native layout.
- **Cross-reference**: OL-189 (wedged-NPU masquerades as kernel hang — rotate to a fresh physical NPU before declaring a kernel bug), OL-165 (no `.cpu()` round-trip / pybind-transpose ban — author the kernel for the native layout), P140 (standalone pybind + `ACLRT_LAUNCH_KERNEL` verify path used here).
### PB-47: chunk-loop UB write-after-read hazard — a per-tile buffer RELOADED via MTE2 at the top of each loop iteration and consumed by V in the SAME iteration needs a `V→MTE2` fence at the ITERATION BOUNDARY
<!-- applies_to_backend: all -->
`applies_to: soc=Ascend950PR_957b; cann=9.1.T500; bisheng=AIV; op_class=all (chunked/tiled kernels reloading a per-tile input via MTE2)`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A2/A3 — not retested; same AIV V/MTE2 parallel-pipe architecture, expected to apply, no cross-arch witness yet)`

- **Severity**: HIGH — deterministic, input-dependent silent data corruption. The kernel compiles and a single chunk may PASS; only multi-chunk runs corrupt.
- **Symptom (distinctive signature)**: **every iteration wrong EXCEPT the last** (the last iteration has no successor reload to overwrite its buffer). The corrupted value is the buffer contents from a LATER iteration that the next MTE2 reload has already written.
- **Root cause**: when a UB buffer is RELOADED via MTE2 at the TOP of each iteration of an outer loop (e.g. an L-chunk loop) AND consumed by a V-pipe op LATER in the same iteration, the iteration boundary must carry a `V→MTE2` fence. AscendC's queue EnQue/DeQue only orders MTE2→V, never V→MTE2; without an explicit boundary fence the NEXT iteration's MTE2 reload overwrites the buffer before the CURRENT iteration's V op has read it. This is the CROSS-ITERATION analogue of PB-17 (which is the cross-ROW alias variant within a single fused `ProcessRow`).
- **Distinct from PB-17**: PB-17 is the P-P65 alias case — two UB buffers aliased within `ProcessRow`, V-write near end of row vs MTE2-write near start of NEXT row. PB-47 is NOT an alias — it is the SAME buffer reloaded each loop iteration. Same underlying hazard class (the AIV V and MTE2 pipes run in parallel and AscendC auto-syncs only MTE2→V), but the trigger context (chunk-loop reload vs intra-row alias) and the diagnostic signature ("every chunk wrong except last") differ enough to catalogue separately.
- **Workaround**: insert a `V→MTE2` fence at the chunk-loop TOP for every non-first iteration — e.g. `if (l0 > 0) SyncVtoMTE2();` (`SetFlag`/`WaitFlag<HardEvent::V_MTE2>`). ~8 lines, cheap relative to the per-chunk compute.
- **Status**: OPEN (architectural — AIV pipe parallelism, not a CANN bug).
- **Evidence**: selective_scan_source_a5 bwd_simd ③ grad_z bug (2026-06-22, A5/Ascend950PR_957b/CANN 9.1.T500). PASS-A L-chunk loop loaded `Ct` (MTE2) and consumed it at the ysc `Mul(prodC, xall, Ct, CN)` (V pipe); the boundary only fenced MTE3→V, so every non-final chunk read the LAST chunk's `Ct` (at L=512, chunk0 read `C[l=256]`). Root-caused by UB-instrumentation (dump UB→GM-scratch, read back from host) — smoking gun: `Ct in UB == truth C[l=256], diff 0.0`. Fix = `if(l0>0) SyncVtoMTE2();` at the chunk-loop top → grad_z MERE 1.499 → 1.85e-7. Whitebox-derived (UB-instrument), not guessed.
- **Evidence (fence survives vectorization)**: selective_scan_full_grad bwd 2.69× scan-vectorization (PR#37, merged main `bda9cb3c`, 2026-06-22, same env). The 2.69× opt re-wrote PASS A's per-chunk scan into the `[l*N+n]` Hillis-Steele form (P-P106) and added a reverse-suffix HS on PASS B — re-exercising the SAME chunk-loop that carries this `V→MTE2` boundary fence. Precision held 30/30 truth-backed after the rewrite, confirming the fence is still required and correctly placed under the vectorized scan (a missing/misplaced fence would have re-surfaced the "every chunk wrong except last" signature). The per-chunk-boundary fence and the intra-chunk scan are independent — both needed.
- **Detection heuristic**: for any multi-chunk / multi-tile loop that reloads a per-tile input buffer via MTE2, check whether that buffer is consumed by a V-pipe op LATER in the same iteration with no `V→MTE2` event between the V consume and the next iteration's MTE2 reload. If the precision signature is "all tiles wrong except the last", suspect PB-47 first.
- **Cross-reference**: PB-17 (the cross-ROW alias sibling — same V→MTE2 hazard class), EC-13 (HardEvent sync list), P-P106 (the L-chunk + parallel-scan pattern this surfaced under).

### PB-48: SIMT GM-scratch written then read ACROSS grid-stride iterations is not coherent even with `asc_threadfence` — prefer all-UB recompute-from-boundary over GM staging for cross-iteration carries
<!-- applies_to_backend: all -->
`applies_to: soc=Ascend950PR_957b; cann=9.1.T500; bisheng=SIMT; op_class=scan/recurrence (any SIMT kernel staging a cross-iteration carry through GM scratch)`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A2/A3 — not retested; same SIMT GM-visibility model expected, no cross-arch witness yet)`

- **Severity**: HIGH — silent, input-dependent data corruption in a SIMT kernel that compiles and runs. The single-iteration path may PASS; only the cross-grid-stride-iteration carry is wrong.
- **Symptom**: a value STORED to a GM scratch buffer in one grid-stride iteration and LOADED back in a LATER iteration reads a STALE/uncommitted value — the store is not guaranteed visible to the later load even when an `asc_threadfence()` is placed between them. `asc_threadfence` provides memory ORDERING (visibility ordering, no blocking — see LANGUAGE_REFERENCE), NOT a cross-iteration store→load completion guarantee for the grid-stride re-entry of the same thread/block over GM scratch.
- **Root cause**: SIMT GM stores are not coherently observable by a subsequent GM load across grid-stride loop iterations within the same kernel launch; `asc_threadfence()` orders visibility but does not force the store to be globally committed-and-readable before the next iteration's load issues. The forward carry staged to GM scratch is read back corrupt.
- **Distinct from PB-47**: PB-47 is the AscendC SIMD/AIV case — a UB buffer reloaded via MTE2 and consumed by the V pipe in the same iteration, fixed by a `V→MTE2` UB fence. PB-48 is the SIMT case — a GM (not UB) scratch buffer written then read across grid-stride iterations, where the fix is NOT a fence at all (`asc_threadfence` does not fix it). Different memory space (GM vs UB), different programming model (SIMT vs SIMD/AIV), different remedy. Catalogue separately.
- **Workaround**: do NOT stage cross-iteration carries through GM scratch in a SIMT kernel. Either (a) keep the carry entirely in UB / registers (all-UB), or (b) RECOMPUTE the carry from the chunk boundary in each iteration rather than reading a previously-stored GM value. The selective_scan coop-bwd L-chunk fix went all-UB / recompute-from-boundary and the corruption cleared.
- **Status**: OPEN (architectural — SIMT GM cross-iteration coherency model, not a CANN bug).
- **Evidence**: selective_scan coop-bwd L-chunk (2026-06-23, A5/Ascend950PR_957b/CANN 9.1.T500, whitebox). The cooperative-scan backward staged the forward `x` to a GM scratch buffer to carry it across L-chunk grid-stride iterations; the cross-iteration GM load read stale data, and inserting `asc_threadfence()` between the store and the later load did NOT fix it. Fix = go all-UB / recompute-x-from-the-chunk-boundary (no GM staging of the carry) → corruption cleared. Whitebox-derived (not guessed).
- **Detection heuristic**: in any SIMT kernel, flag a GM scratch buffer that is `*ptr = v` in one grid-stride iteration and read `v = *ptr` in a LATER iteration of the SAME loop. If precision is wrong on the carried value and adding `asc_threadfence()` does NOT fix it, suspect PB-48 — remove the GM round-trip (all-UB or recompute) rather than chasing more fences.
- **Cross-reference**: PB-47 (the SIMD/AIV UB V→MTE2 sibling — different memory space + remedy), P-P106 (the L-chunk scan pattern this surfaced under), LANGUAGE_REFERENCE `asc_threadfence` (ordering, not completion).

### PB-49: chunk-loop CROSS-ROW UB write-after-read hazard — a SHARED per-tile buffer written by V in one grid-stride ROW and reloaded via MTE2 at the top of the NEXT row needs a `V→MTE2` fence at the ROW boundary (AscendC auto-syncs only MTE2→V)
<!-- applies_to_backend: all -->
`applies_to: soc=Ascend950PR_957b; cann=9.1.T500; bisheng=AIV; op_class=all (any SIMD/AIV kernel whose grid-stride row loop reuses shared UB working buffers)`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A2/A3 — same MTE2→V-only auto-sync model expected, no cross-arch witness)`

- **Severity**: HIGH — silent, timing-dependent wrong output in a kernel that compiles and runs. The single-row path PASSES; only the multi-row-per-block (rows = B·D > nblk) case corrupts.
- **Symptom**: a kernel processing multiple rows per block via a grid-stride row loop, reusing the SAME UB working buffers each row, produces wrong output for some rows — distinctively, **only the LAST row per block is correct** (it has no successor to clobber its in-flight writes). The corruption WIDENS with the per-row working-set size (state width N): masked at small N (the buffers drain within pipeline time), grows to ~2/3 of rows wrong at larger N.
- **Root cause**: the prior row's last V-pipe writes to the shared UB buffers are still in flight when the next row's first MTE2 reload (e.g. an `Af`/`B`/`C` `DataCopy`) issues. AscendC auto-inserts only the `MTE2→V` dependency (reload-then-compute), NEVER the reverse `V→MTE2` (compute-then-next-reload) across the loop boundary → the next row's reload races the prior row's compute. A `PipeBarrier<PIPE_V>` does NOT help (it orders V-vs-V, not V-vs-MTE2).
- **Fix (minimal)**: ONE `SetFlag/WaitFlag<HardEvent::V_MTE2>` at the row-loop top for `r>0` — NOT a heavy `PipeBarrier<PIPE_ALL>` (which also works but is a superset). Orders the prior row's V writes before this row's first MTE2 reload.
- **Distinct from PB-47/PB-48**: PB-47 = same-iteration UB reload→consume (intra-row). PB-48 = SIMT GM cross-iteration coherency (different memory space + no-fence remedy). PB-49 = SIMD/AIV UB shared-buffer reuse across ROW iterations, fixed by a row-boundary `V→MTE2` fence. Whole-row variant of PB-47.
- **Status**: architectural (AscendC auto-sync covers MTE2→V only).
- **Evidence**: selective_scan fwd-SIMD (2026-06-24, A5/Ascend950PR_957b/CANN 9.1.T500, PR #50). Pristine kernel: N=32 rows=114 → 76/114 wrong (only last-row-per-block correct); N=16 rows=114 → 13/114 wrong (reaches N=16 too, timing-fragile); one `V_MTE2` row-boundary fence → N∈{16,32,64} rows>56 all 0-wrong, customer N=16 B8/D192/L5000 0/1536, perf-neutral (+0.4%). NOTE: at L=5000 large-CH the customer shape did NOT manifest the hazard (latent — confirmed correct pre-fix); the leak only bit small-L (≲256) multi-row.
- **Detection heuristic**: any SIMD/AIV kernel with a grid-stride ROW loop reusing shared UB buffers, where multi-row-per-block output is wrong but single-row is correct AND only the last row per block survives → add a `V→MTE2` fence at the row-loop top.
- **Cross-reference**: PB-47 (intra-iteration sibling), PB-48 (SIMT GM sibling), EC-77 (the carry-fold RAW fence in the same op), OL-253 (N-adaptive chunk in the same op).


### PB-50: Ascend950PR int8 `Cast` silently drops tail elements when count not multiple of 32 (VEC 32B width)

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=quantization; dtype=int8`
`verified_on: soc=Ascend950PR_9579; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A3 — Cast int8 tail behavior on V220 unconfirmed; VEC width may differ)`

- **Severity**: HIGH — silent data loss: tail elements zeroed without compile error, runtime error, or NaN/Inf signal.
- **Symptom**: `Cast(dst_int8, src, RoundMode::CAST_TRUNC, count)` where `count` is not a multiple of 32 produces zero-valued int8 elements at tail positions `[floor(N/32)*32, N)`. The kernel compiles and runs cleanly — no 507035, no error code. Only precision verification reveals the zeroed tail. Observed at N=129, 130, 131, 144, 257 (all N where N % 32 ≠ 0).
- **Root cause**: The VEC `Cast` to int8 operates at 32B granularity (32 int8 elements per VEC operation). When `count` is not a multiple of 32, the hardware writes only `floor(count/32)*32` complete VEC blocks; the partial tail block is silently dropped (set to zero) rather than written with valid elements.
- **Workaround**: align the quantization count to 32 before `Cast`:
  ```cpp
  int32_t n_al_quant = ((N + 31) / 32) * 32;  // round up to 32B boundary
  // Size all quant-path buffers (i32Buf_, fp16Buf_, y1Buf_) for n_al_quant, not N
  Cast(y1Buf_, fp16Buf_, RoundMode::CAST_TRUNC, n_al_quant);
  ```
  Pass both `valid_count=N` and `aligned_count=n_al_quant` to the kernel so the output path knows where valid data ends. The pybind layer allocates int8 output with `AlignInt8(N_padded)` = N rounded up to 32, so the aligned tail elements land in the padded output region (discarded by pybind post-kernel narrow).
- **Detection**: int8 output shows zero values at tail positions for any N where `N % 32 ≠ 0`. If per-element `max_abs_diff` is non-zero ONLY at indices `[floor(N/32)*32, N)` and the diff magnitude equals the reference value (kernel=0, ref=non-zero), suspect this bug. Systematic sweep across N=128..257 will expose it.
- **Status**: OPEN (VEC hardware constraint; VEC int8 Cast width is 32B = 32 elements).
- **Evidence**: add_rms_norm_quant (2026-06-23, Ascend950PR_9579, CANN 9.0.0): aog-precision-probe iter 1 identified the tail-zero signature across 12 previously-failing N values. Align-to-32 fix resolved all 12.
- **Cross-reference**: PB-22 (similar 32B-alignment truncation for plain DataCopy on V351 — same hardware width class, different primitive), EC-23 (DataCopyPad UB->GM crash — adjacent alignment surface).

### PB-51: msprof / torch_npu.profiler ANALYZE stage fails `Operation not permitted` (EPERM) in the `.171` npu_dev3 container despite Privileged=true → use NPU `aclrtEvent` elapsed_time device-time as the fallback
`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all`

In the `.171` (203.0.113.171) `npu_dev3` container, both `torch_npu.profiler` and the `msprof` CLI collect raw profiling data but the **analyze/export stage fails with `Operation not permitted` (EPERM)** — so per-launch isolated `kernel_details.csv` device-time is UNAVAILABLE there, even though the container reports `Privileged=true` (the EPERM is from a profiling-syscall/securityfs restriction the privileged flag doesn't cover).

- **Workaround**: measure device-time via NPU events — `aclrtEvent` / `torch.npu.Event(enable_timing=True)` `start/end` + `elapsed_time` around the kernel launch (this is device-side timing, NOT host wall-clock). Validate it once against any host where msprof analyze DOES work (e.g. a committed msprof number) — observed agreement ~7%. The Event device-time is a valid same-host A/B basis (the EPERM is an export-tooling limit, not a timing-accuracy limit).
- **Distinct from the .141 policy**: `.141` FORBIDS msprof by team policy (different reason); `.171` ALLOWS msprof but its analyze stage EPERMs in npu_dev3. So for `.171` perf A/B, use NPU-Event device-time.
- **Status**: OPEN (container/securityfs profiling-syscall restriction).
- **Evidence**: selective_scan bwd perf (2026-06-30, .171 npu_dev3, CANN 9.0.0/9.1.0, PR #71): msprof analyze EPERM on both tools; NPU-Event device-time used + cross-validated ~7% vs the committed msprof figure.
- **Cross-reference**: [[feedback_report_perf_only_from_probe_device_time]] (the device-time-not-wall discipline — NPU-Event IS device-time, satisfies it), OL-245 (the perf A/B this surfaced under).

### PB-52: `basic_api` scalar-GM atomic RMW (`AtomicMax`/`AtomicAdd`/`AtomicMin`/`AtomicCas`/`AtomicExch`) is arch-gated OFF on `__NPU_ARCH__==2201` (a3/V220) — guarded to `5102||3101` only
`applies_to: soc=Ascend910_9382 (a3/V220); cann=9.0.0; bisheng=n/a; api=AtomicMax/AtomicAdd/AtomicMin/AtomicCas/AtomicExch (basic_api scalar __gm__ RMW form)`
`verified_on: soc=Ascend910_9382; cann=9.0.0; arch=__NPU_ARCH__==2201`
`unverified_on: whether a later CANN release adds 2201 support — re-grep the shipped header before relying on this being permanent`

The scalar global-memory atomic RMW intrinsics declared in `basic_api/kernel_operator_atomic_intf.h` (`AtomicMax<T>(__gm__ T*, T)` etc.) are wrapped `#if __NPU_ARCH__==5102||3101`, which **EXCLUDES** the a3/V220 target's actual `__NPU_ARCH__==2201` (confirmed by `basic_api/kernel_common.h`'s separate `==2201` branch). A kernel design assuming a device-side scalar atomic-max/add/etc. on a3/V220 will not compile.

- **Distinct from `SetAtomicAdd`**: `SetAtomicAdd<T>()` + `DataCopy(VECOUT→GM)` is a DIFFERENT mechanism (DMA-mode accumulate, widely used) and is NOT what this entry covers; the arch gate here is on the scalar `__gm__`-pointer RMW form only. (Max/Min DMA-mode variants unverified.)
- **Portable fallback**: when the problem reduces to a small-N sequential scan, launch with `blockDim=1` and use `GlobalTensor<T>::GetValue`/`SetValue` scalar GM accessors — these carry NO `__NPU_ARCH__` guard (unconditionally available) and single-core program order makes read-after-write race-free by construction (no atomics, no scratch-reset needed when Pass-1-writes and Pass-2-reads share the same launch + row set).
- **Grep-scope safety (folded-in near-miss)**: the "grep the shipped SDK header before assuming a primitive exists" discipline MUST be scoped to the public header tree (`$CANN_PATH/*-linux/asc/include/`), NEVER a raw `find $CANN_PATH -iname '*.h'` — on a real CANN install the latter also matches the FORBIDDEN op-impl source tree (`opp/built-in/op_impl/ai_core/tbe/impl/ops_*/ascendc/*`, off-limits for NPUKernelBench per CLAUDE.md "No CANN Source Code Copy"). Scope the grep to `asc/include/` up front. (NB: the worker note that surfaced this called the grep-before-assuming rule "OL-54" — that ID is actually "Reg-based SIMD"; the grep-first discipline lives in the spirit of OL-80/OL-84 "check the table/header before assuming", not OL-54.)
- **Status**: OPEN (arch-gated, not a bug per se — a real capability gap on 2201).
- **Evidence**: `12_KvRmsnormRopeCache` Part B (2026-07-16, a3 Ascend910_9382, CANN 9.0.0): pre-implementation grep found `AtomicMax` gated away from 2201 → avoided a wasted atomic-based dedup attempt (the exact failure mode where a prior fo/ko pair had assumed `Reg::Rsqrt` existed); used a `blockDim=1` sequential `GetValue`/`SetValue` last-writer-wins dedup instead → perf 0.5704×→0.9329×.
- **Cross-reference**: OL-80 / OL-84 (check the shipped header/table before assuming an API compiles); CLAUDE.md "No CANN Source Code Copy — NPUKernelBench ONLY" (the grep-scope guard).

---

### PB-53: standalone-pybind MIX (cube+vec) `507014` silent-hang is a DOUBLE-BOOTSTRAP — the `auto_gen_*_kernel` device wrapper ALREADY did the full KFC/workspace bootstrap; a kernel that RE-RUNS `SetSysWorkspaceForce`/`GetUserWorkspace` double-applies the 16 MB reserve → KFC msg-ring lands in user scratch → AIC↔AIV rendezvous in corrupted GM → exit-124 [V220, MIX, workspace-bootstrap, standalone-pybind]

`applies_to: soc=Ascend910_9392 (V220 / A3); cann=9.0.0; bisheng=n/a; op_class=standalone_pybind_mix_aic_1_2 (matmul::Matmul / KFC cube+vec launched via the framework aclrtlaunch_* stub); macro=KERNEL_TYPE_MIX_AIC_1_2`
`verified_on: soc=Ascend910_9392; cann=9.0.0`
`unverified_on: soc=Ascend950PR (V351 / A5 — untested; A5 resolves the base via a distinct dav-c310 `__get_kfc_workspace_addr` — do NOT carry this attribution to an A5 target)`
`status: CONFIRMED 2026-07-17 (device-proven; corrects the earlier "op-framework required" and "GE-supplies-FFTS" mis-attributions)`

- **Severity**: HIGH — silent hang, NO aicore exception; the host blocks at `torch.npu.synchronize()`, the launch reports aicore timeout `507014` / `INNER_EXIT=124`. Trivially misread as "the MIX launch is MISSING an FFTS/workspace descriptor" or "standalone KFC needs the CANN op-framework" — the opposite of the truth.
- **Mechanism (the DOUBLE-bootstrap)**: the framework's `auto_gen_<op>_kernel` **device wrapper** performs the ENTIRE KFC/workspace bootstrap BEFORE it calls the user kernel: `set_ffts_base_addr(...)` (host-side `rtGetC2cCtrlAddr` supplies the FFTS C2C control address) + `SetSysWorkspaceForce(raw)` + `matmul::clearWorkspace(...)` + then **reassigns** `workspace = GetUserWorkspace(raw)` (= `raw + 16 MiB`). So by the time the USER kernel entry runs, `workspace` is ALREADY the user region and the 16 MiB system/KFC reserve is ALREADY applied. If the kernel entry then RE-RUNS `SetSysWorkspaceForce(workspace)` / `GetUserWorkspace(workspace)` on that already-offset pointer, it **double-applies** the reserve (`raw + 16MB + 16MB`) → the KFC message ring that the AIC server and AIV client rendezvous on lands **inside the user scratch region** → the AIC↔AIV handshake reads/writes corrupted GM → AIV hangs at its first `Iterate`, the AIC server spins `while(isRun)` forever → `507014` / exit-124.
- **Fix**:
  1. The kernel must **NOT re-bootstrap**. The `workspace` argument is ALREADY the user region — use it directly. `GetSysWorkSpacePtr()` returns the RAW (pre-offset) base — that is exactly what `REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), ...)` wants. Do NOT call `SetSysWorkspaceForce` / `GetUserWorkspace` inside the kernel entry.
  2. Launch through the framework stub `aclrtlaunch_<kernel>(...)` (which routes via `auto_gen_<op>_kernel` = the wrapper that does the single correct bootstrap), NOT a raw `<<<>>>` launch — a raw `<<<>>>` BYPASSES the wrapper, so NO bootstrap runs at all and you get the genuinely-missing-descriptor variant of the hang. The two failure modes (nothing bootstraps vs bootstrapped twice) have the SAME `507014` symptom and OPPOSITE fixes.
- **Meta-lesson (why this was misattributed for so long)**: the 2026-07-12 bisection (see `CAND-KFC-standalone-bootstrap-teardown` VERIFICATION RESULT) applied the full KFC recipe at kernel level, still hit `507014`, and concluded "the msg-ring only bootstraps under the op-framework." That was the RIGHT observation (the wrapper does the bootstrap) with the WRONG inference (that a standalone kernel therefore *can't* be made to work). "**missing**" and "**done twice**" produce the identical symptom (a `507014` KFC-rendezvous hang) but demand OPPOSITE fixes — when a mechanism looks *missing*, first check whether something already does it. This also corrects the even-earlier "the FFTS descriptor is GE-op-build-provisioned at launch, so a hand-rolled launch cannot supply it" framing (KB_INDEX §By-Symptom, since reverted): a MIX launch does need FFTS installed, and the `auto_gen` wrapper's `set_ffts_base_addr`→`rtGetC2cCtrlAddr` DOES install it — the bug was never a MISSING descriptor, it was a DUPLICATED bootstrap.
- **Detection**: a standalone-pybind cube+vec (`matmul::Matmul` / KFC) `MIX_AIC_1_2` kernel builds + register-binaries clean, then `torch.npu.synchronize()` hangs, host reports aicore timeout `507014` / `INNER_EXIT=124`, no aicore exception. Static smell: the kernel entry BODY itself calls `SetSysWorkspaceForce(` or `GetUserWorkspace(` AND the op is launched via `aclrtlaunch_*` (the wrapper path) — that is the double.
- **Evidence**: standalone-pybind `matmul::Matmul` MIX cube+vec op, V220/A3 `Ascend910_9392`, CANN 9.0.0, 2026-07-17. Isolation carried over from the CAND-KFC bisection: FFTS `WORKSPACE_SYNC_ID` event channel WORKS (`ffts_probe` exit 0); AIC and AIV read the same `GetSysWorkSpacePtr` base; the residual hang isolates to the KFC msg-ring server/client machinery. Ruled out (tested, not assumed): SoC-label (built `_9382` vs chip `_9392` → rebuild `_9392` → same hang). Removing the kernel's redundant re-bootstrap so the wrapper's single bootstrap stands is what lets the AIC↔AIV rendezvous land in the correct system reserve.
- **Scope (falsifiable)**: V220 / CANN 9.0.0 / standalone-pybind `matmul::Matmul` (KFC) MIX path launched via `aclrtlaunch_*`. Bound to the `auto_gen` wrapper's 16 MiB-reserve layout; A5/arch35 resolves the KFC base differently (`__get_kfc_workspace_addr`) and is untested — do NOT carry the attribution across SoC.
- **Cross-reference**: **PB-34** — a DISTINCT sibling `507014` hang on the SAME V220 MIX surface but a DIFFERENT cause: PB-34 is `MatmulImpl<>` + **manual** `CrossCoreSetFlag<0x2>`/`WaitFlag` whose user-owned FFTS *sync-slot* IDs collide with KFC's internal flag pool; PB-53 is a DOUBLED *workspace/msg-ring* bootstrap corrupting the KFC GM rendezvous. A kernel can hit either (or both) — PB-34 is fixed by removing the manual flag chain (Pattern B) / hand-rolling tile-MMAD (Pattern A); PB-53 is fixed by NOT re-bootstrapping the workspace. PB-54 (the `507015` sibling in the same standalone-pybind matmul workspace-wiring family — fault vs hang). PB-41 (the workspace-prefix contract the wrapper IMPLEMENTS — PB-53 is what breaks when you re-apply that offset a second time). `CAND-KFC-standalone-bootstrap-teardown` (this PB RESOLVES its open VERIFICATION-RESULT question: the "op-framework required" verdict was the wrapper-does-the-bootstrap half seen from one side; the standalone fix is don't-do-it-twice). OL-275 (the managed-lifecycle KFC bootstrap the `auto_gen` wrapper embodies). KB_INDEX §By-Symptom DEBT-210 sync-base discussion is ORTHOGONAL (that concerns the FFTS C2C *sync base* / `SetSyncBaseAddr`; this concerns the *workspace/msg-ring* 16 MiB reserve).

### PB-54: passing `GetSysWorkSpacePtr()` to `REGIST_MATMUL_OBJ` (unset/invalid for a plain `ascendc_library` stub) → the matmul lib's fixpipe L0C→GM write targets an MPU-protected GM address → `507015` AIC fixpipe MPU-invalid (`subErrType:4`, `fixp_error0=0xb0b`) [V220, matmul, fixpipe, workspace-base, standalone-pybind]

`applies_to: soc=Ascend910_9392 (V220 / A3); cann=9.0.0; bisheng=n/a; op_class=standalone_pybind matmul::Matmul (REGIST_MATMUL_OBJ on an ascendc_library stub, cube L0C→GM fixpipe)`
`verified_on: soc=Ascend910_9392; cann=9.0.0`
`unverified_on: soc=Ascend950PR (V351 / A5 — untested)`
`status: 507015 ROOT-CAUSE CONFIRMED 2026-07-17 (single-variable minimal non-MIX matmul::Matmul repro + catlass 00_basic_matmul positive control on the same host/CANN); END-TO-END MIX/KFC FIX UNSOLVED as of 2026-07-17 — a fully-separate `aclrtMalloc` buffer clears 507015 but HANGS the full MIX op (see Fix → Remaining)`

- **Severity**: HIGH — the AIC fixpipe faults hard: `507015` / `The MPU address access is invalid` / `subErrType:4` / `fixp_error0=0xb0b`. Distinctive fingerprint vs the other `507015` flavors (PB-21 `PipeBarrier<PIPE_ALL>`-on-TBuf; PB-45 Reset/event-id alias; the `Nd2NzParams` cube-layout `507015`) — this one is specifically a fixpipe (`fixp_error0`) MPU-address fault on the cube's GM write-back.
- **Mechanism**: `REGIST_MATMUL_OBJ(&tPipe, <wsBase>, mm, tiling)` gives the matmul library the GM base it stages its L1/L0 scratch AND its L0C→GM fixpipe write-back into. Passing `GetSysWorkSpacePtr()` as `<wsBase>` is WRONG for a plain `ascendc_library` pybind stub: with no op-framework wrapper having bootstrapped the system workspace (contrast PB-53's `auto_gen` path), `GetSysWorkSpacePtr()` is **UNSET / not a valid matmul-output workspace** — the library's fixpipe then writes to an MPU-protected / invalid GM address → the FIX/M unit faults `507015`.
- **Fix (removes the `507015` fault on a NON-MIX matmul — PARTIAL on MIX, read Remaining)**: pass an **EXPLICITLY-allocated valid device workspace base** — size it from the tiling (`GetWorkspaceSize()`), `aclrtMalloc(GetWorkspaceSize(), ACL_MEM_MALLOC_HUGE_FIRST)`, and hand THAT buffer's device pointer to `REGIST_MATMUL_OBJ` / `Initialize(args, deviceWorkspace)`. It MUST be a buffer **DISTINCT from the output C tensor** (matmul workspace ≠ output). This is exactly what catlass `00_basic_matmul` does: `GetWorkspaceSize()` → `aclrtMalloc(...)` → `Initialize(args, deviceWorkspace)`. On a plain **non-MIX** `matmul::Matmul` op this is the COMPLETE fix (507015 gone, verified). On a **MIX/KFC** op it is NOT complete — see Remaining.
- **⚠ Remaining (MIX/KFC op — UNSOLVED as of 2026-07-17; do NOT naively apply the non-MIX fix)**: on a full MIX (cube+vec / KFC) op the `REGIST_MATMUL_OBJ` workspace base must be BOTH (a) **valid** — NOT `GetSysWorkSpacePtr()`'s unset base (→ else `507015`) — AND (b) **coupled to the framework system-workspace** where the KFC message ring lives. A **fully-separate** `aclrtMalloc` buffer satisfies (a) but breaks (b): the KFC ring (in the sys-workspace) and the matmul compute (in the separate buffer) end up in mismatched GM regions → the AIC↔AIV KFC handshake stalls → **HANG (RC=124)**, verified on the full MIX op 2026-07-17. So on a MIX op the separate-buffer "fix" merely **trades the `507015` fault for a `507014`-class hang** — it is NOT a complete fix. The complete fix — a **valid sub-region WITHIN the framework system-workspace** (so it is both valid AND coupled to the KFC ring) rather than a disjoint buffer — is LOCALIZED but UNSOLVED as of 2026-07-17. (This is the same coupling PB-53 exploits from the other direction: the `auto_gen` wrapper hands the kernel a workspace that IS the framework sys-workspace region — a hand-rolled separate buffer is not.)
- **Distinct from PB-41**: PB-41 is host UNDER-ALLOCATION of the `(sysWs + userWs)` workspace UNDER the `auto_gen` wrapper (kernel gets `GetUserWorkspace(w) = w + sysWsSize`, buffer too small → overrun). PB-54 is passing the WRONG BASE to `REGIST_MATMUL_OBJ` in a standalone stub with NO wrapper — an invalid/unset pointer, not a too-small one. Same family (matmul workspace wiring), different failure: over-run garbage vs MPU-invalid fixpipe fault.
- **Evidence (single-variable)**: minimal NON-MIX `matmul::Matmul` repro, V220/A3 `Ascend910_9392`, CANN 9.0.0, 2026-07-17 — `GetSysWorkSpacePtr()` base → `507015` (MPU-invalid, `subErrType:4`, `fixp_error0=0xb0b`); flip ONLY the workspace base to an explicit `aclrtMalloc(GetWorkspaceSize())` buffer → fault gone. **POSITIVE CONTROL**: catlass `00_basic_matmul` (low-level `BlockMmad` cube→GM) "Compare success." on the SAME host / CANN 9.0.0 → proves the fault is NOT environmental and NOT MIX-specific (a minimal non-MIX matmul reproduces it, and correctly-wired matmul passes). Also ruled out (tested, not assumed): SoC-label (built `_9382` vs chip `_9392` → rebuild `_9392` → same fault); tiling (hand-rolled vs official `MultiCoreMatmulTiling`); tile size; C-target; workspace-SIZING (not the variable — the wrapper's `clearWorkspace` already sizes it; the BASE is the variable). **MIX-op data point (2026-07-17, the Remaining wall)**: applying the confirmed fix (explicit `aclrtMalloc` buffer, distinct from output) to the FULL MIX op did NOT produce a working demo — it cleared `507015` but then **HANGS (RC=124)** because the separate buffer decouples the matmul compute from the KFC ring in the framework sys-workspace → AIC↔AIV handshake stalls. So the 507015→fault-gone single-variable finding is real (non-MIX), but the end-to-end MIX fix is unsolved.
- **Scope (falsifiable)**: V220 / CANN 9.0.0 / standalone-pybind `matmul::Matmul` (`REGIST_MATMUL_OBJ`) on an `ascendc_library` stub with a cube L0C→GM fixpipe. A5/arch35 untested.
- **Cross-reference**: PB-41 (workspace-prefix under the wrapper — sibling: under-alloc vs wrong-base). PB-53 (the `507014` DOUBLE-bootstrap hang in the MIX/KFC standalone-pybind path — hang vs fault, but the SAME "standalone-pybind matmul workspace-wiring" family). PB-21 / PB-45 (other `507015` flavors — distinct mechanisms, distinct fingerprints). `CAND-FA-A5-KFC-WORKSPACE` / `CAND-KFC-standalone-bootstrap-teardown` (A5 + KFC workspace-base siblings). catlass `00_basic_matmul` (the positive-control good-wiring reference: `GetWorkspaceSize` → `aclrtMalloc` → `Initialize(args, deviceWorkspace)`).


---

### PB-55: `MIX_AIC_1_2` AIC↔AIV cross-core handshake is DIRECTION-ASYMMETRIC — the reverse (AIV→AIC) flag is per-subblock-COUNTED, not broadcast → a single-setter reverse handshake DEADLOCKS [220x/Ascend910_9382, mixed-mode-sync]

`applies_to: soc=Ascend910_9382; cann=9.0.0; op_class=mixed_aic_aiv_pattern_a_tile_mmad; macro=KERNEL_TYPE_MIX_AIC_1_2`
`verified_on: soc=Ascend910_9382 (Atlas A3 / 220x single-die); cann=9.0.0; 2026-07-18 — DS hand-authored FA core (S=Q@Kᵀ cube#1 → softmax vector fp32-accum → O=P@V cube#2; both matmuls MatmulImpl::IterateAll<sync=true>) on a3 device. Cosine 0.999999 vs pure-torch attention golden; allclose(rtol=1e-2,atol=1e-3) 100% PASS; deterministic across 3 fresh processes (bit-identical). Distinct flag ids per handshake; never flag id 0 (PB-35).`
`unverified_on: soc=Ascend950PR_9579 (V351 / A5) — NOT tested; the asymmetry may differ on arch35, do NOT claim it there.`
`status: DEVICE-CONFIRMED + FIX DEVICE-VALIDATED 2026-07-18 (a3 / Ascend910_9382 / CANN 9.0.0). Bisection-pinned single-setter deadlock; the both-subblocks-signal fix passes precision + determinism.` **[cann corrected 9.1.0 → 9.0.0 on 2026-07-18: the container is CANN 9.0.0 (`version.info` Version=9.0.0; no 9.1.0 exists); device re-confirmed the MIX builds+runs+precision-passes on 9.0.0. The earlier `9.1.0` was an unverified (propagated) label — it made an op-gen worker wrongly decline the MIX on a false version-mismatch. `verified_on ⊆ applies_to` both now 9.0.0.]**

- **Severity**: HIGH (a PURE wait-deadlock — host-side timeout / process exit 124 at `torch.npu.synchronize()`, with **NO fault code in a fresh plog**; trivially misdiagnosed as an algorithm/precision bug or blamed on the 507014/507015/507035 FAULT family of PB-34 which it is NOT).
- **Symptom**: a multi-stage `KERNEL_TYPE_MIX_AIC_1_2` kernel (e.g. a 2-cube + softmax FlashAttention core) HANGS forever — host `torch.npu.synchronize()` times out / process exits 124 — with a CLEAN plog (no `LaunchAscendKernel` error code, no aicore exception). Occurs specifically when the REVERSE `AIV→AIC` handshake flag is raised from only ONE AIV subblock of the 1:2 pair.
- **Mechanism (bisection-pinned)**: in `MIX_AIC_1_2` (1 AIC : 2 AIV) the `CrossCore` flag handshake is DIRECTION-asymmetric:
  - **Forward `AIC→AIV`** (e.g. `FLAG_S`): **BROADCAST** — one AIC `CrossCoreSetFlag<MODE2, PIPE_FIX>(flag)` releases BOTH AIV subblocks' `CrossCoreWaitFlag`. (Verified by the bisect variant: cube#1 + forward handshake + softmax, reverse handshake removed → SYNC_OK.)
  - **Reverse `AIV→AIC`** (e.g. `FLAG_P`): **per-subblock-COUNTED, NOT broadcast** — the single AIC `CrossCoreWaitFlag(flag)` requires a `CrossCoreSetFlag` from EVERY AIV subblock of the 1:2 pair. If only subblock 0 sets it, the AIC blocks FOREVER. (Verified by the bisect variant: reverse handshake kept but single-setter, cube#2 removed → DEADLOCK, exit 124, no fault code.)
- **Fix (device-validated)**: BOTH AIV subblocks must `CrossCoreSetFlag(reverse_flag)`. The single logical WRITER of the shared buffer (e.g. subblock 0 writes P) still writes ALONE; both subblocks merely SIGNAL the reverse flag. **Mode-2 multi-setter suffices — §4 / mode-4 (1:1 per-subblock-disjoint-ids) is NOT required for this** particular reverse-direction count.
  ```cpp
  // AIV side — reverse AIV→AIC handshake (FLAG_P). BOTH subblocks must SIGNAL:
  if (subBlockIdx == 0) {
      // subblock 0 is the sole WRITER of the shared P buffer …
      CrossCoreSetFlag<MIX_SYNC_MODE2, PIPE_MTE3>(FLAG_P);   // signal
  } else {
      CrossCoreSetFlag<MIX_SYNC_MODE2, PIPE_MTE3>(FLAG_P);   // subblock 1 signals too (no write)
  }
  // AIC side: a SINGLE CrossCoreWaitFlag(FLAG_P) now unblocks (count = 2 reached).
  ```
- **Detection** (pre-build static guard):
  ```bash
  # Warn: a MIX_AIC_1_2 kernel with an AIV→AIC reverse CrossCoreWaitFlag on AIC but
  # only ONE CrossCoreSetFlag(<reverse_flag>) reachable on the AIV side → count never reached.
  for f in workspace/<op>/kernel/*.{h,cpp}; do
      grep -q "KERNEL_TYPE_MIX_AIC_1_2" "$f" || continue
      revflags=$(grep -oE "CrossCoreWaitFlag(<[^>]*>)?\(([A-Za-z0-9_]+)\)" "$f")
      # for each reverse flag the AIC waits on, count how many CrossCoreSetFlag(flag) the AIV emits;
      # a single-setter under MODE2 for a reverse flag is the PB-55 smell.
      echo "$f: inspect reverse-flag setter multiplicity"
  done
  ```
- **Companion minimal-MIX witness (same device, same date)**: a 1-cube + 1-vec MIX runs on a3 via `MatmulImpl` `sync=true` + a SINGLE canonical `CrossCore` handshake (the `add_lora` pattern) + the standard `aclrtlaunch` runtime — which ALREADY supplies FFTS via `rtGetC2cCtrlAddr` (`rtGetC2cCtrlAddr` → `main_kernel<<<blockDim, nullptr, stream>>>(..., fftsAddr)`). So the "host FFTS gap" is NOT a gap for the standard framework build; the multi-stage cross-sync half is what PB-55 resolves.
- **Cross-reference**: **PB-34** (`MatmulImpl<>` + manual `CrossCore` FFTS sync-slot COLLISION → 507014/507015/507035 FAULTS — a DIFFERENT mechanism; PB-55 is a PURE wait-deadlock from handshake mis-COUNTING, and it uses `IterateAll<sync=true>` which AVOIDS PB-34's async-KFC slot-collision path). **PB-35** (`event_t(0)` / flag-id-0 collision — PB-55 uses distinct flag ids per handshake and never id 0). `CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP` + **DEBT-222** (PB-55 is the DEVICE-MEASURED resolution of the multi-stage-cross-sync half of that gap). `fa_class/cross_core_sync.md` §4 (the mode-4 / disjoint-id cross-core recipe — related but NOT required for this reverse-direction count).

### PB-56: a3/220x multi-core (blockDim>1) MIX cube↔vec LOOPED bidirectional handshake deadlocks (507014)

```yaml
applies_to:
  paradigm: ascendc
  soc: Ascend910_9382
  cann: 9.0.0
```
- **Status**: RESOLVED (2026-07-20, device-measured — multi-core FA ACHIEVED, see Resolution). This PB documents that a **HAND-ROLLED** per-core cube↔vec cross-core LOOPED ring deadlocks (mode3/mode5) AND — the key point — that FA multi-core does NOT need it. NOT a hardware wall; NOT "perf N/A" (a3 vendor baseline = `npu_fusion_attention`, measured running ~102µs).
- **Affected**: only **hand-rolled** multi-core (blockDim>1) MIX 1:1 cube↔vec handshakes on a3/220x (`CrossCoreSetFlag`/GM-slot per-core). Single-core (blockDim=1) hand-rolled handshake is fine (works + verified 17/17). The library/KFC multi-core path is NOT affected — it does not hand-roll.
- **Symptom**: a **hand-rolled** per-core cube↔vec handshake in the **LOOPED bidirectional** form hangs: `torch.npu.synchronize()` blocks >90s, no aicore exception, prints kernel-launch line then no result. Device latches AICore=100% (recovers on process-kill / >120s idle). **1-shot (non-looped)** concurrent multi-group MIX runs clean.
- **Root cause (device-measured, by elimination)**: NOT distinct-vs-shared flag-id orphan-credit — BOTH distinct-id (mode3) AND the shipped c220 `pto::TSYNC_CVID` shared-flag+GM-slot recipe (mode5, **per-round** sync) deadlock. §4's arch35 mode-4 resolution does NOT port (`INTRA_MODE=4` collapses to mode-0 on c220 via `mode & 0x3` mask). Deeper than flag-id disambiguation. See `fa_class/cross_core_sync.md` §5 for the 4-mode device table. **The deeper lesson: do not hand-roll multi-core cube↔vec sync — use the library.**
- **Resolution (device-measured 2026-07-20 — multi-core FA ACHIEVED)**: extend the VERIFIED single-core base (library `MatmulImpl IterateAll<sync=true>` + per-pair PB-55 handshake) to blockDim>1, ONE head-slice per core, keeping ONLY the per-pair `MODE2`/`CV_CORE_SYNC` handshake — NO cross-core ring. **Runs DEADLOCK-FREE, precision 20/20** (5 shapes × cores{1,2,4,20}, cos 0.999998+, deterministic, bit-identical to 1-core), perf **0.186× vendor** (550µs vs `npu_fusion_attention` ~102µs @ 20 cores; recovered 14.3× over single-core 7866µs). **Key insight: FA multi-core is per-head-INDEPENDENT** — each core is a self-contained single-core FA, `MODE2` scopes the handshake to that core's OWN AIC↔AIV group (zero cross-core interaction), so the LOOPED cross-core ring (mode3/mode5 deadlock) is UNNECESSARY for head-parallel FA. Perf gap is STRUCTURAL (non-flash: materializes full [S,S] score+prob to GM per head + row-serial softmax vs vendor's fused online-softmax L1/UB-resident); next lever = flash/online-softmax rework (cf independent generated-kernel witness 0.51×), NOT a sync fix. NOTE: the *async*-KFC path (`matmul::Matmul`+KfcServer) remains a3-standalone-BLOCKED (CAND-KFC-standalone-bootstrap-teardown / PB-53 / PB-54) — the `IterateAll<sync=true>` route sidesteps it. Artifacts (in-repo, disk-verifiable): `fa_class/evidence/a3_multicore_fa_20260720/` — `RESULT.md` + `mc_verify.json` (20/20) + `mc_perf.json`. See `cross_core_sync.md` §5.
- **DETECTION methodology (stomp-probe)**: to distinguish a real multi-core deadlock from a kernel bug, build a minimal blockDim=2 handshake probe with 4 modes (1-shot distinct / 1-shot shared / looped distinct / looped shared+GM-slot). Run the CANDIDATE mode FIRST on a **verified-clean** device (a hang LATCHES the device — budget for it), and the known-deadlock mode LAST as control. Verify device-clean (`npu-smi` AICore≈0%, no proc, no zombie) before EACH run; a hang is a valid negative verdict; do NOT hammer a latched device (>120s idle to recover). Write the run-log incrementally to disk (the probe can die on API error — disk is the salvage).
- **large-D note (507015 refuted for the device-side head-loop)**: the 507015 fault class (single-core, D>128, N>1) is for hand-rolled **HOST** multi-head tiling; a **device-side head-loop** kernel (`MatmulImpl IterateAll<sync=true>` per head, internal K-tiling) does NOT hit it — D=256/384/512/768/1024/1280 all computed + matched golden (cos≥0.99998) at single-core on a3/220x.
- **Evidence**: DS device build+probes 2026-07-19/20 → `fa_class/evidence/a3_multicore_fa_20260720/GMSLOT_RUN_LOG.md` (hand-rolled mode 0/1/3/5 characterization) + the multi-core RESULT.md/mc_verify.json/mc_perf.json in that dir. Grounded in c220 `pto/npu/a2a3` TSyncCVID.hpp / TSync_Custom.hpp / TPush.hpp. Cross-ref: `fa_class/cross_core_sync.md` §5 (sibling to the §4 V351/A5 RUNNABLE handshake), PB-55 (single-core both-AIV-subblocks-must-Set).

### PB-57: a3/arch22 whole-device `SyncAll<false>()` MIX (cube+vector) requires `KERNEL_TYPE_MIX_AIC_1_1` (1:1) — the no-macro 1:2 default makes idle AIV cores skip the barrier loop → 507014 deadlock (narrows PB-28/A-P34)

```yaml
applies_to:
  paradigm: ascendc
  soc: Ascend910_9382
  cann: 9.1.0
  op_class: attention-fwd
  macro: KERNEL_TYPE_MIX_AIC_1_1
  note: multi_stage_mix_aic_aiv (cube+vector) with whole-device SyncAll<false>()
```
- **Status**: CONFIRMED (2026-07-20, device-verified on a3/Ascend910_9382 — root cause pinned + empirically flipped on a SINGLE variable = the task-type macro). GDR forward gen3, archive `output/fqa_gated_delta_rule/`.
- **Severity**: HIGH — builds + packs clean (bisheng OK, no 107000 at `RegisterAscendBinary`); hangs only at RUNTIME as aicore-timeout 507014, easily misdiagnosed as an algorithm/layout bug.
- **Symptom**: a MIX (cube+vector) kernel that uses whole-device `SyncAll<false>()` as its cross-core barrier and emits NO task-type macro hangs at `torch.npu.synchronize()` with aicore-timeout 507014; adding `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)` as the first entry statement → EXIT=0, runs, no hang.
- **Mechanism**: no task-type macro → default arch22 MIX dispatch = **1:2**, so AIV `GetBlockNum()` (8) ≠ AIC (4). A per-work-item loop `for (bh = GetBlockIdx(); bh < nHead; bh += GetBlockNum())` is entered a different number of times per core: AIV cores with `GetBlockIdx() >= nHead` never enter → issue **zero** `SyncAll` calls, while AIC + low-index AIV cores issue N each. Whole-device `SyncAll<false>()` waits for ALL cores → the cores that skipped never arrive → permanent cross-core deadlock (507014). Pinning **1:1** (`MIX_AIC_1_1`) gives AIC/AIV identical `GetBlockNum()` → symmetric barrier count → no deadlock.
- **Narrows PB-28 / A-P34**: those state "`KERNEL_TASK_TYPE_DEFAULT` is arch35-only, rejected on Ascend910_9382 → 107000; emit NO task-type macro on arch22." That is over-broad. It holds for `KERNEL_TYPE_AIV_ONLY`/`AIC_ONLY` (A-P34) and `MIX_AIC_1_2` (PB-28/PB-40), but is REFUTED for `MIX_AIC_1_1`: the customer cv-reference emits exactly this macro, builds `SOC_VERSION=Ascend910_9382`, and runs clean (device-measured). PB-28 already declares `MIX_AIC_1_1` out-of-scope of the 107000 ban — this entry promotes that exemption to a positive decision rule.
- **Decision rule**: (a) MIX + whole-device `SyncAll<false>()` + per-core work-item loop → emit `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)` (1:1 mandatory; the 1:2 default breaks SyncAll symmetry). (b) MIX + scoped `CrossCoreSetFlag`/`WaitFlag` MODE2 handshake → default 1:2 is DEADLOCK-free, NO macro (that path is built around the 1:2 asymmetry, e.g. P-P116) — **BUT deadlock-free does not guarantee determinism; see (b')**. (c) NEVER mix "no macro (1:2)" with "whole-device SyncAll" — that is the GDR gen3 507014 trap.
- **(b') Generated scoped-handshake caveat — 1:2 is deadlock-free but can be RUN-TO-RUN NON-DETERMINISTIC at high parallelism**: a generated fused MIX kernel with `MIX_AIC_1_2` plus many per-id `CrossCoreSetFlag<0x2>` handshakes flipped bit outputs across runs above roughly 12 concurrent MIX blocks, consistent with FFTS flag-pool aliasing. **Fix**: select the balanced `MIX_AIC_1_1` template. No whole-device `SyncAll` is needed; pairing balance alone removed the nondeterminism. Keep the per-id flags and WAR barriers unchanged. Scope is limited to the measured generated-kernel lane; behavior of the hand-written scoped-1:2 path (P-P116) remains unverified.
- **Evidence**: GDR forward gen3 (a3, Ascend910_9382, 2026-07-20). No-macro + 15 whole-device `SyncAll` → 507014 on case0 `[1,64,4,128]`. Add `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)` as first entry statement → EXIT=0, runs, no hang. cv-reference (same macro) runs clean on the same chip. Archive `output/fqa_gated_delta_rule/` (verification.json + SYNCFIX_RUN_LOG.md).
- **Evidence (b')**: generated GDR forward kernel (a3, Ascend910_9382, 2026-07-21). `MIX_AIC_1_2` plus 11 per-id `CrossCoreSetFlag<0x2>` handshakes was deadlock-free but nondeterministic at high parallelism (fp64 gate 15/16). Selecting `MIX_AIC_1_1` made all 16 cases deterministic over N=30 runs and preserved measured device performance.
- **Cross-ref**: PB-28 (the `MIX_AIC_1_1`-out-of-scope note this entry promotes to a rule), A-P34 (the `*_ONLY` macro ban, still valid), PB-40 (`MIX_AIC_1_2` 107000 at teardown on arch22), PB-55 (single-core both-AIV-subblocks-must-Set — the scoped-handshake sibling), `patterns/domains/fa_class_a3_mix_template.md` §(c) (P-P116, the scoped-CrossCore MODE2 alternative), `fa_class/cross_core_sync.md`, P-P117 (the chunked-GDR pattern that surfaced this).
