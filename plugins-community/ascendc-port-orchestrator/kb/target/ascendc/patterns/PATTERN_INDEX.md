# Pattern Library — Routing Index

> Always loaded by all skills. Use to identify which domain files to load.
> Domain files contain full pattern details; this index is for routing only.

## Domain Routing Table

| ID | Name | Domain | Trigger | Severity |
|----|------|--------|---------|----------|
| F-P1 | bf16 precision handling | precision | bf16 dtype in test | MEDIUM |
| F-P2 | Multi-dtype template | precision | multi-dtype kernel | LOW |
| F-P3 | SIMD bf16 MicroAPI Cast | precision | SIMD + bf16 | MEDIUM |
| F-P4 | SIMD PipeBarrier alignment | platform_compat | SIMD DataCopy | HIGH |
| F-P5 | Warp-aligned loop boundary | precision | cooperative group ops | HIGH |
| F-AP1 | dtype string match trap | precision | compare_data with dtype | CRITICAL |
| F-AP2 | __threadfence misuse | precision | __threadfence in code | MEDIUM |
| P-P1 | numBlocks dynamic | thread_utilization | any kernel launch | HIGH |
| P-P2 | WarpReduceAddSync | scatter_add | atomicAdd in reduction | HIGH |
| P-P3 | vec4 vectorization | memory_access | dim % 4 == 0 | MEDIUM |
| P-P4 | Dynamic block size | thread_utilization | variable work per block | MEDIUM |
| P-P5 | LAUNCH_BOUND + CHECK | kernel_launch | every kernel | MEDIUM |
| P-P6 | grid_y consistency | kernel_launch | multi-grid kernel | MEDIUM |
| P-P7 | pragma unroll scope | kernel_launch | inner loops | LOW |
| P-P8 | Host benchmark practice | kernel_launch | benchmark code | LOW |
| P-P9 | **SIMT vs SIMD decision framework** | platform_compat | algorithm classification | **HIGH** |
| P-P10 | Block oversubscription | scatter_add | atomicAdd contention | HIGH |
| P-P11 | Adaptive tile size | memory_access | multi-dim tiling | HIGH |
| P-P12 | int32 inner loop | memory_access | int64 in hot loop | MEDIUM |
| P-P13 | Cooperative traversal | cooperative | GROUP_SIZE parallel | HIGH |
| P-P16 | Cooperative value copy | cooperative | large vector copy | MEDIUM |
| P-P17 | Prefix-sum + block atomic | scatter_add | scatter-add aggregation | HIGH |
| P-P18 | __ldg/__stg L2 cache hint | platform_compat | SIMT GM read/write with cache control | HIGH |
| P-P19 | Kernel UT requirement | kernel_launch | every kernel | HIGH |
| P-P20 | Thread Utilization (BRE=dim) | thread_utilization | multi-dim decomposition | HIGH |
| P-P21 | Sorted-edge accumulation | scatter_add | atomicAdd in scatter loop | HIGH |
| P-P22 | Persistent kernel | thread_utilization | work_items >> 56 | MEDIUM |
| P-P23 | Contiguous-chunk vs grid-stride | memory_access | arr[i-1] neighbor access in loop | HIGH |
| P-P24 | Sort-to-reuse (GM read amplification) | memory_access | indirect GM read shared by N items | **CRITICAL** |
| P-P25 | SetAtomicAdd + DataCopyPad | memory_access | SIMD scatter-add to GM | **CRITICAL** |
| P-P26 | SetFlag/WaitFlag fine-grained sync | memory_access | SIMD pipeline overlap | HIGH |
| P-P27 | bf16 scalar via Cast(bf16→float) | platform_compat | bfloat16_t GetValue or static_cast | **CRITICAL** |
| P-P28 | **TQue<4> auto pipeline overlap** (replaces PipeBarrier Ping-Pong) | memory_access | SIMD kernel with DataCopy+VEC loop | **CRITICAL** |
| P-P29 | Batch preload cache (index/weight) | memory_access | GetValue in loop from GM (scalar bottleneck) | **CRITICAL** |
| P-P30 | fp16/bf16 scalar kernel arg (uint16_t bits) | platform_compat | half/bf16 scalar in extern "C" kernel params | **CRITICAL** |
| P-P32 | Sorted-edge dedup (atomicCAS-free) | scatter_add | atomicCAS first-occurrence on sortable data | HIGH |
| P-P33 | **SIMT→SIMD for memory-bound elementwise** | memory_access | msprof MTE2=0% AND throughput < 50% theoretical BW | **HIGH** |
| P-P34 | **SIMT per-element indirect gather** | memory_access | torch.gather, index_select, per-element indirect addressing | **HIGH** |
| P-P35 | bf16 direct assign in SIMT (no Cast needed) | platform_compat | SIMT kernel with bf16 copy (no arithmetic) | MEDIUM |
| P-P36 | TQueBind for pure data-movement ops | memory_access | pure data-movement ops (no VEC compute) — replaces Adds bridge | **HIGH** (unverified on A5) |
| P-P37 | DataCopyParams for strided/non-contiguous copy | memory_access | non-contiguous memory copy (columns, strided data) — replaces for loop | **HIGH** |
| P-P38 | Vector Counter mode for auto tail handling | memory_access | VEC op non-aligned tail handling — replaces manual mask management | **HIGH** (unverified) |
| P-P39 | Fan-in threshold for sort decision | scatter_add | scatter-add fan-in ratio determines whether to sort | HIGH |
| P-P40 | Double-buffer accumulation with flush | scatter_add | UB accumulation double-buffer swap after sort + fp16 precision protection | HIGH |
| P-P41 | 2D tiling for large embedding scatter | scatter_add | embDim too large to fit with sort buffer in UB | MEDIUM |
| P-P42 | Hardware Sort Pipeline (Concat+Sort+Extract) | sort | hardware bitonic sort network, N ≤ 4096 fp32 | **CRITICAL** |
| P-P43 | Sort algorithm selection decision tree | sort | select sort algorithm by N/dtype/batch | HIGH |
| P-P44 | Radix sort twiddle transform | sort | bit-twiddle transform for float/signed radix sort | MEDIUM |
| P-P45 | Single-pass UB-resident dynamic quantization | reduction_quant | absmax+scale+quantize all in UB, avoid 2-pass HBM | **CRITICAL** |
| P-P46 | Quantize cast chain (fp32→int8) | reduction_quant | AscendC multi-step cast: RINT→NONE(SetDeqScale)→TRUNC | HIGH |
| P-P47 | Half-interval tree reduction | reduction_quant | O(log₂ D) in-place fold reduction down to 64 elements → WholeReduce | HIGH |
| P-P49 | Per-uid 8-aligned GM scalar slot | memory_access | SIMD multi-core, each core writes 1 scalar to shared GM array (mean/rstd/sum, etc.) | **HIGH** |
| P-P50 | bf16 intermediate precision matching (backward ops) | precision | reference multiplies in bf16 then casts to fp32 for accumulation; kernel must mirror. **[CPU-truth audit 2026-04-29: ✅ VALIDATED — co-occurs with CANN-pass ops 17_EmbeddingWithInitialLayernormBackward, 20_FusedRopeWithQkNormAndKvCacheUpdate. Do not downgrade.]** | **CRITICAL** |
| P-P51 | Native-dtype constraint for scalar coefficient chain (fp16 normalization backward) | precision | reduction result → multi-step scalar arithmetic → broadcast; every step must stay in native dtype (fp16/bf16), cannot do all math in fp32 and cast only at the end. **[CPU-truth audit 2026-04-29: ✅ VALIDATED — co-occurs with CANN-pass op 20_FusedRopeWithQkNormAndKvCacheUpdate. Do not downgrade.]** | **CRITICAL** |
| P-P52 | CANN reduction precision contract — bf16/fp16 must use fp32 promotion | precision | all CANN reduce ops follow a fixed DAG: CopyIn→Cast<fp32,T>→ReduceOp<fp32>→Cast<T,fp32>→CopyOut; use CAST_RINT. **[CPU-truth audit 2026-04-29: ✅ VALIDATED — co-occurs with CANN-pass op 17_EmbeddingWithInitialLayernormBackward. Note: aligns with CPU's fp32 promotion convention for reductions, so this IS CPU-truth-aligned. Do not downgrade.]** | **CRITICAL** |
| P-P53 | Mean = Sum × (1/N), not Sum ÷ N | precision | Target CANN prior-art uses Muls(invSpatial), but final choice must match OL-97 selected-source behavior or CPU fp64 truth; target code is advisory. | **CRITICAL** |
| P-P54 | Reciprocal uses tensor Div(1,x), not scalar 1/x | precision | Select scalar or vector form from OL-97 selected-source behavior plus CPU fp64 edge probes. Target CANN bodies are advisory prior-art. | HIGH |
| P-P55 | Pow strategy depends on exponent: integer/half exponents use Mul/Div/Sqrt decomposition, general exponent uses Power API | precision | Decide among advisory target-derived candidates using selected-source capture and CPU fp64 edge/backward probes. | HIGH |
| P-P56 | All scalar arithmetic under bf16/fp16 must go through VEC path | precision | Scalar and VEC units may round differently; compare both candidates against selected-source behavior and CPU fp64/backward truth. | HIGH |
| P-P57 | **SIMD ReduceMax(calcIndex=true) for small-k vectorized topk** | sort | small k (≤ 16) topk over N ≥ 256, avoid O(N·k) scalar insertion loop | **CRITICAL** |
| P-P59 | **Masked reduction with strict-`<` threshold — tied-threshold buffer truncation** | precision | reference uses `v < threshold` STRICT mask + reduce → kept = count(v ≥ threshold) exceeds named k; fixed top-K/top-N buffer truncates the tied set; Layer 1 (buffer bump + global denom) / Layer 2 (three-layer per-column) / Layer 3 (full sort). Applies to: top_k_top_p, nucleus sampling, attention tail-drop, sparse gather with score filter, quantile-based masking | **CRITICAL** |
| P-P60 | **AscendC Sort ASC tie-break direction REVERSED from PyTorch stable ASC (ANTI-PATTERN)** | sort | `Sort<ASC>` places larger-idx BEFORE smaller-idx among ties, opposite of PyTorch stable ASC (smaller-idx-first). Affects any kernel that needs to match PyTorch stable-sort tie ordering. Fix: post-walk re-selects cutoff_orig_idx by picking the n_drop_tied-th smallest original idx in ascending order. Complementary to P-P59: correct buffer size ≠ correct tie ordering; both are required | **CRITICAL** |
| P-P61 | **Kernel runtime determinism — preserving patterns and anti-patterns** | determinism | Covers 3 groups: (a) 6 det-preserving positive patterns (hw Sort+canonical tie / single-core per-row / pure VEC no-atomic / queue depth=1 for observable outputs / explicit secondary sort / det reduction tree); (b) 5 det-breaking anti-patterns (concurrent atomicAdd / unordered multi-core merge / uninitialized scratch / data-dependent reduction order / missing pipe barrier); (c) orchestrator Phase O1.5 op-level determinism policy classification (required/best_effort/n/a, per reference semantics). Propagates as `DET_POLICY` in all agent briefs. Loaded by worker/probe/optimizer/researcher when policy != n/a. | **CRITICAL** |
| P-P62 | **Row-Scalar VEC Multiply — per-row scalar multiplication that avoids the Scalar pipe** | memory_access | **Precondition: the kernel must have a multi-row parallel axis (R ≥ 8 rows batched per iter) to amortize.** One `Brcb` fills a `[dealRowCount × 8]` UB tensor, then `Mul(dst, src0, src1Ub)` with `BinaryRepeatParams{src1BlkStride=0, src1RepStride=1}`. Standard CANN flash-attention RowMuls pattern. Measured (H=8 block-matched): Brcb 25.3× K_base; the "Muls flexible scalar position" variant does NOT bypass (0.97×, argument-order variant). **Counter-example**: op#11's single-row-per-iter kernel cannot be Kind-1 retrofitted (a single scalar Brcb becomes pure overhead); needs Kind-2 batch rewrite + UB compression to apply. | HIGH (applies to multi-row batch kernels) |
| P-P65 | **Fused-op cross-phase buffer-liveness aliasing** | memory_access | When a fused op has tight UB budget, alias dead buffers to save space. Method: list each UB buffer's live range (which phases read/write, down to which row), find slots that are "dead past phase N", reuse for scratch in later phases. **Key: must be dead across phase boundary; buffers still alive within the same phase cannot be aliased**. op#11 evidence: aog-kernel-optimizer Opt4 wrongly picked tmpBuf (still alive) → silent overwrite; aog-fused-optimizer C4 correctly picked otherBuf (dead after SwiGLU) → PASS on first try + 20 KB freed. Watch out for cross-row V↔MTE2 hazard (see PB-17). | HIGH |
| P-P66 | **Pybind H-axis alignment padding convention (kernel DataCopy datablock alignment)** | memory_access | Inside the kernel, `DataCopy(dst, src, count)` requires count aligned to the dtype datablock (fp32: 8, fp16/bf16: 16). If H may be non-aligned, do NOT add large DataCopyPad branches inside the kernel — instead, in the **pybind layer** use `torch::zeros((*, H_padded, *)).copy_(..., src)` to pre-pad input H to a multiple of `lcm(8,16)=16`. The kernel continues computing with H_padded; the output uses `out.narrow(dim, 0, H_orig)` to slice back to the original dim. The padded region is naturally zero and does not affect reduce/scatter semantics. **Applies to**: fused ops with multi-dtype inputs (mixed bf16+fp32) / scatter+reduce combinations / any multi-kernel pipeline that needs a uniform alignment assumption. First used in op#19 FusedResidualRmsNormBackward; inherited by op#17 EmbeddingWithInitialLayernormBackward (probe CONVENTION classification). **Extension — small per-channel affine params (group_norm_grad, 2026-06-03, port_a3_to_a5 V220)**: the same convention generalizes from the H-axis to tiny per-channel affine vectors. `weight[C]` with C=8 fp16 = 16 bytes is not 32B-aligned and V220 has no usable `DataCopyPad` UB→GM (EC-23); pad `weight` to `Caln=ceil(C/16)*16` via `torch::empty({Caln}).narrow(0,0,C).copy_(weight)`, and the multi-output grads `dweight`/`dbias` use a Caln-padded reduce buffer narrowed back in pybind. **Legitimacy distinction vs the OL-167 cheat**: this is NOT host-side data-path repair — the kernel reads only the valid channel indices via `GetValue` (padding lanes are never consumed) and the reduce writes correct values to the valid prefix, so the narrowed output is genuinely kernel-computed. EC-23 sanctions pybind padding on V220 (where DataCopyPad UB→GM crashes); on V351 the kernel MUST instead use DataCopyPad per P-P98 / OL-167. | **HIGH** |
| P-P67 | **PyTorch-UB-class scatter-overwrite — NPU reference non-deterministic, kernel cannot chase** | scatter_add | `index_put_(accumulate=False)` / `scatter_(reduce=None)` on duplicate indices is PyTorch-UB; NPU torch resolves duplicates via HW thread scheduling — non-deterministic across runs (op#19 pp-2: 14/17 cases jitter, max pairwise diff 4.39 fp16). No deterministic rule (last/first/wave-chunk W∈{8..512}/AIV-block A∈{1..40}/round-robin) reaches 90% match (best WF32 mean 61%, min 31%). Kernel MUST NOT use atomic / multi-core scatter to "match NPU randomness" (re-introduces kernel non-det); MUST NOT use case-specific predicates (OL-85 violation). Ship deterministic single-core CPU-semantic scan; gap is REQUIREMENT not kernel bug. Mitigation = verifier-side alt-ref / case-gen `allow_dup_indices=False`. **See OL-90 for full protocol**. | **CRITICAL** |
| P-P68 | **Single-AIC GEMM with constexpr static tiling + on-stack TCubeTiling (Opt2 baseline)** | platform_compat | Level-3 cube ops (matmul / batch_matmul / gemm / Linear / Conv) where the gemm fits one AIC. `MatmulImpl<AT,BT,CT,BIAS,MM_CFG>` + `MM_CFG = constexpr MatmulApiStaticTiling` with ~25 shape-independent fields lifted to compile-time + on-stack `TCubeTiling tiling{}` filled at kernel entry + non-`__gm__` `Init(const TCubeTiling*, TPipe*)`. Eliminates ~25 GM tiling-reads per call AND the host-side `torch::empty(200B) + .copy_()` H2D (~5–10 µs). Determinism by-construction (single-AIC, no atomicAdd, IterateAll<sync=true>+End()). Validated on op#1 BatchMatmul (Opt0 0.515× → Opt2 1.27×). Carries cleanly to TransA/TransB/BothTrans. See OL-91 / EC-39 / EC-40 for the build invariants. | **HIGH** |
| P-P69 | **Cube transposed-input via runtime `SetTensor*(_, isTrans=true)` bool — NOT template ISTRANS** | platform_compat | Level-3 cube op with one or both operands logically transposed (TransA / TransB / BothTrans, Conv backward weights, Linear with weight transpose). All `MatmulType<...>` template ISTRANS args stay default `false`; the actual transpose driver is the runtime member set by `SetTensorA(gm, /*isTransposeA=*/true)` (or B-side analogue). Tiling field map is shape-flat: M=output_rows, N=output_cols, Ka=Kb=reduction, regardless of which operand transposes. A-side and B-side bools are independent and symmetric — BothTrans = both bools true with no further changes. Validated across the {none, A, B, both} 4-corner lattice on op#1/#4/#5/#3 (each ≤1 precision iter, last two ops 0+0). Direct extension of P-P68. **Anti-pattern**: `MatmulType<...,ISTRANS=true>` + `SetTensorA(_, false)` compiles but produces garbage output (the static template flag is MX-FP8-only, never consulted in ND→ND path). See OL-92 for full mechanism trace. | **CRITICAL** |
| P-P70 | **Fused dequant→activation→quant pipeline algorithm shape (generic for L2 fused activation+quant ops)** | memory_access | Generic algorithm structure for fused L2 ops of class `dequant→activation→quant`. Members: op#10 SwigluQuant, op#11 DequantSwigluQuant, future GroupNormSiluQuant / RMSnormGeluQuant / etc. **Pipeline**: (1) Dequant: `Cast(x_fp, x_int32, NONE)` then `Mul(x_fp, x_fp, weight_scale)` then `Mul(x_fp, x_fp, activation_scale)` (broadcasted per-row); add bias if non-None. (2) Activation chunk + compute: see P-P71 for layout choice. (3) Smooth-quant: `Mul(out, out, quant_scale)` if quant_scale non-None. (4) Dynamic int8 quant tail: see P-P72. **UB layout for fused ops**: 4-6 buffers per row (x_int32 source, x_fp working, gate, linear, out_fp32, out_int8). For V220 (UB=192KB nominal, ~126KB effective per KC-2 candidate after runtime reservations) tile rows by D. **Per-row round-robin across AIV** — single-AIV-per-row, no inter-AIV reduction needed (quant amax is per-row). Determinism by-construction. Combine with P-P65 (cross-phase buffer aliasing) for tight UB budgets. | **HIGH** |
| P-P71 | **Activation-layout convention disambiguation via CANN op-parameter default fingerprinting** | precision | When a CANN fused op exposes scalar parameters with **non-trivial default values** (irrational constants, small-int saturation limits, named bias terms), those defaults are usually the FINGERPRINT of a published public formula — and CANN is faithfully implementing that formula INCLUDING its data-layout convention. **Workflow**: (1) grep the CANN op definition / `torch_npu.npu_<X>` signature for default values; (2) WebFetch known public OSS reference repos (`github.com/openai/gpt-oss`, `github.com/meta-llama/llama`, `github.com/mistralai/mistral-src`) for the default-value triple; (3) if a match is found, the layout (chunked vs interleaved-stride-2 vs others) matches that source's tensor packing. **Canonical instance** (current evidence): SwiGLU has two valid formulations — (A) **chunked halves** (PyTorch native): `gate, linear = chunk(x, 2, dim=-1)`, used by `swiglu_mode=0` of `aclnnDequantSwigluQuantV2` and most PyTorch references; (B) **interleaved stride-2** (gpt-oss / OpenAI): `gate = x[..., ::2]`, `linear = x[..., 1::2]`, used by `swiglu_mode=1` of `aclnnDequantSwigluQuantV2`. CANN convention NOT documented in torch_npu help — must determine empirically via the fingerprint workflow above. Op#11 evidence: probe pp-1 found `α=1.702 + L=7.0 + b=1.0` matched gpt-oss exactly (`gpt_oss/torch/model.py:358-365`) → interleaved stride-2 layout → bit-exact 100% match across 8 case combinations. **AscendC implementation choices for stride-2 extraction**: (a) `DataCopyPad` with `srcStride/dstStride/repeatStride` params, OR (b) full-row load + two `Mul` ops with `BinaryRepeatParams{src1BlkStride, src1RepStride}` patterning (preferred for V220 — avoids extra MTE2), OR (c) scalar-loop GetValue (slowest, only for tiny H). **Generalizes** to any future fused activation op exposing non-trivial scalar defaults: tanh-clamp variants, GeLU-tanh-approximation parameters, RoPE base-frequency, LayerNorm-eps fingerprinting, etc. | **HIGH** |
| P-P72 | **Dynamic int8 per-token quant tail — generic primitive chain** | reduction_quant | Standard tail of dequant→activation→quant ops: per-row dynamic int8 quant. Inputs: fp32 tensor `out [N, H]`. Outputs: int8 `q [N, H]` + fp32 `quant_scales [N]` (BEFORE clamp — this is what CANN returns). **Sequence**: (1) `Abs(abs_buf, out)` per row, (2) `ReduceMax<float>(amax_scalar, abs_buf, count=H)` per row → scalar, (3) emit `quant_scales[i] = amax/127.0` BEFORE clamping (this is the returned scale; clamp is internal-only div-guard), (4) `clamped_scale = max(amax/127.0, 1e-10)` via `Maxs(scale, scale, 1e-10)` to avoid div-by-zero on all-zero rows, (5) `Muls(out, out, 1/clamped_scale)` per row (literal-first per OL-82 if precision-sensitive), (6) `Mins(out, out, 127.0)` then `Maxs(out, out, -128.0)` for symmetric saturation, (7) `Cast(q_int32, out, RoundMode::CAST_RINT)` IEEE-RNE per OL-81, (8) `Cast(q_int8, q_int32, RoundMode::CAST_NONE)` per P-P46. **ERRATA (dav_3510, 2026-08-29, OL-293)**: step (8)'s int32→int8 pair is NOT in the dav_3510 Cast dispatch table — the device-side `ASCENDC_ASSERT` guard is an empty macro under `__NPU_DEVICE__`, so the Cast silently compiles to zero instructions and the int8 output is stale UB residue (bimodal repeat fingerprint: repeat 1 near-zero, repeat 2+ garbage). The direct fp32→int8 leg is equally absent. On dav_3510 the only path INTO int8 is half→int8: use int32→half(CAST_NONE)→int8(CAST_RINT) per OL-293 (exact for [-128,127] integers, no double-rounding). **Critical contract**: `ReduceMax` is hardware-deterministic per fixed input (by-construction det). The `quant_scales` returned MUST be the pre-clamp `amax/127.0` value — the clamped variant is kernel-internal guard only. Op#10/op#11 use this exact chain bit-exact vs CANN reference. **Common bug**: returning the clamped scale instead of pre-clamp causes 1-ULP residuals on near-zero rows. | **HIGH** |
| P-P73 | **Op-signature-as-public-formula recognition (META workflow for fused activation ops)** | precision | When CANN exposes a fused op via `torch_npu.npu_<X>` and its parameter list contains scalar attributes with **non-trivial default values** (i.e., not `0.0` / `1.0` / generic), those defaults are usually the FINGERPRINT of a published public formula. **Recognition checklist**: scan op signature for: `_alpha=<irrational>` (e.g. 1.702 = SiLU/Swish-1 approximation, 1.41 = √2/sqrt(π)), `clamp_limit=<small int>` (e.g. 7.0 = gpt-oss tanh saturation, 6.0 = ReLU6), `_beta=<small float>`, `_eps=<small>` (norm-class). **Action when match suspected**: (1) WebFetch `github.com/openai/gpt-oss` / `github.com/meta-llama/llama` / `github.com/mistralai/mistral-src` / model repos for the default-value triple; (2) read those repos' formula definitions; (3) probe a3 NPU reference vs that public formula on a small test case (4-row × 128-col is enough); (4) on bit-exact match, the kernel formula is now publicly traceable + KB-codifiable. **Why this matters**: avoids 50+ formula hypotheses (op#11 kw-1 worker burned ~2h before pp-1 recognized gpt-oss fingerprint and resolved in ~30 min). Pattern applies to ALL future fused activation ops where CANN op def has non-trivial defaults. | **HIGH** |
| P-P74 | **Multi-AIC partition-dispatch via host-precomputed segment offsets** | platform_compat | Any kernel whose output is the row-concat of N independent sub-computations with host-knowable per-segment row counts. `blockDim = G` (one AIC per segment); each AIC layers P-P68 inside (constexpr static tiling + on-stack TCubeTiling). Host pybind builds `cum_out[G+1]` int32 with sentinel `cum_out[G] = total_rows` so the kernel reads `end_row = cum_out[g+1]` uniformly. Per-AIC decode: `bid = GetBlockIdx(); input_row_off / M_g` from a host flag (variable-input segments → use offsets[bid..bid+1]; uniform-input segments → use bid·m_uniform). Output side always uses `cum_out`. Determinism by-construction (each output row owned by exactly one AIC, no atomicAdd, fixed mmad order). Concrete instances: grouped/segmented matmul, MoE expert dispatch, GroupedConv, segmented attention/sparse-gather. Reference-fallback: when target torch+CANN lacks `grouped_mm`, OL-89 prose-spec extension lets `model.py` loop `torch.matmul(A_g, B[g])` per segment. **Limitation**: `blockDim=G` caps parallelism; for G ≤ AIV_count with M·N·K > ~1M, the next move is 2D dispatch (aog-kernel-optimizer territory). Validated op#2 GroupedMatmul (50/50 + 16/16 PASS bit-exact, det 50/50, 1.05× median, 0+0 iters). See OL-93 for op#2 evidence record. | **HIGH** |
| P-P75 | **Manual TBuf pipeline with explicit `SetFlag/WaitFlag<MTE2_V>` event sync** | platform_compat | TBuf-based VEC pipeline (DataCopy in → V compute → DataCopy out, looped) on V220-class chips MUST use explicit `SetFlag<HardEvent::MTE2_V>(eventId) + WaitFlag<HardEvent::MTE2_V>(eventId)` (and `V_MTE3` analogue) between stages — `PipeBarrier<PIPE_ALL>()` does NOT guarantee MTE2→V completion on TBuf and triggers silent crash 507015 (PB-21). Event IDs fetched once via `GetTPipePtr()->FetchEventID(...)` outside the loop and reused per iter. Decision rule for TBuf vs TQue lives in OL-94. Use TBuf when dataflow needs multi-buffer aliasing across phases (P-P65), persistent buffers across inner loops, or other constraints TQue's fixed depth can't express; otherwise prefer TQue auto-rotation (P-P28). Validated op#27 a3 V220 (5-iter rescue from 507015 crash); DS V4 worker session surfaced as the gap that prompted codification — weaker models default to TQue without an explicit decision table to consult. **Anti-pattern**: TBuf + `PipeBarrier<PIPE_ALL>()` (silent 507015 crash). | **HIGH** |
| P-P76 | **Aligned-base scratch via `Duplicate(0) + scalar SetValue` for inline VEC reductions with unaligned index offsets** | platform_compat | When an inline VEC reduction (`Axpy(acc, src[off], scalar, len)` or analogous) has a runtime index offset that's not 32B-aligned (e.g. conv `op_start = max(0, ceil_div(-base, stride))` is 1/2/3 for typical kernel_size+padding combos), the natural offset-Axpy formulation triggers UB-alignment crash error 340. Workaround: build an **aligned-base tmp scratch** by `Duplicate(tmp, T(0), len_pad)` then a scalar-loop `tmp.SetValue(op, src.GetValue(op*stride + base))` for the valid range, then `Axpy(acc, tmp, scalar, len_pad)` with all-aligned bases. Cost: ~`len_pad` scalar SetValues per (ic, kp) iteration — negligible on AIV vs the alignment-crash alternative. Combine with EC-23 output over-allocation pattern (pybind allocates `[B, *, len_pad]` instead of `[B, *, len]`, narrows on return). Validated op#6 ConvStandard1d (2026-04-29) — direct-VEC 1D conv via per-(batch, out_ch) AIV core. Likely-applicable to any inline reduction where the per-iter src offset is data-dependent and can be unaligned: 1D/2D/3D conv, 1D/2D pooling with stride>1, dilated patches. | **HIGH** |
| P-P77 | **Per-iter output via `TQue<VECOUT, depth=2>` (the TQue side of the OL-94 decision)** | memory_access | The TQue counterpart to P-P75: when a per-iter loop emits via `Cast(out_ub, work_ub, ...) ; DataCopy(gm_out, out_ub, count)` and the output UB region does NOT need cross-phase buffer-liveness aliasing (P-P65) or persistent-buffer semantics, **prefer `TQue<QuePosition::VECOUT, depth=2>` over a bare `TBuf<VECCALC>`** for the output buffer. The depth=2 queue's `AllocTensor`/`EnQue`/`DeQue`/`FreeTensor` rotation provides automatic MTE3↔V sync via slot rotation: slot N+1's `AllocTensor` blocks until slot N's prior MTE3 retires, removing the need for explicit `SetFlag<HardEvent::MTE3_V>+WaitFlag<...>` flags between the per-iter `DataCopy` and the next iter's V write to the same UB region. **Anti-patterns observed when worker leaves a TBuf in this slot**: (a) `PipeBarrier<PIPE_ALL>()` at iter top — disrupts TPipe's queue scheduling, op#27 Phase D iter-5 regressed to 9/10 wrong-output runs; (b) extra `PipeBarrier<PIPE_V>` between unrelated V ops — drains the V pipe prematurely, op#27 6/10 wrong. Always rotate the output buffer through a queue first; only escalate to the manual-event-sync path (P-P75) if the TQue refactor cannot apply for structural reasons. **Cross-ref**: OL-94 (decision table TQue vs TBuf), P-P75 (TBuf manual-event side), PB-21 (PipeBarrier silent crash on V220), A-P61 (det anti-patterns; TBuf-output-race is the practitioner-side downstream of A-P61's atomicAdd-upstream). Validated op#27 27_MultiMaskAttentionAggregation a3 V220 2026-04-28. | **HIGH** |
| P-P78 | **Row-parallel data unfolding to expose AIV parallelism upstream of single-AIC cube call** | platform_compat | When a workload decomposes into "unfold input → cube GEMM → reshape output" and the cube call is single-AIC-per-tile (P-P68), the unfold stage is the parallelism floor: dispatching the unfold by `(b, g)` alone leaves 50+ AIVs idle when `B·G ≪ 56`. Pattern: dispatch `blockDim_unfold = B · G · K_total` where `K_total` is the per-`(b, g)` row count of the unfolded matrix (e.g. for conv `K_total = Cin_per_g · K_h · K_w`); decode `bid → (b, g, k_idx) → (ic, kh, kw, ...)`; each AIV builds exactly one row of `HW_pad` fp32 elements via `Duplicate(0) → fill-from-source → DataCopy` (no cross-AIV communication). Determinism by-construction (each row owned by exactly one AIV). 3-kernel pipeline shape: AIV unfold (row-parallel) + AIC cube (per-(b,g)) + AIV reshape/bias (row-parallel). Combine with EC-42 (split AIV vs AIC into separate .cpp). **Generalizes** to: 2D/3D conv via im2col, attention K/V re-assembly before flash-attn cube, group-conv unfolding, segmented-batch attention. **Anti-pattern**: dispatching the unfold by `(b, g)` only — leaves 90%+ of AIVs idle when `B·G ≪ AIV_count`. Validated op#7 ConvStandard2d ko-1 (2026-04-29): direct-VEC 0.087× → cube `(b,g)`-only unfold 0.155× → row-parallel unfold **0.705× median** (4.55× over per-(b,g), 8.10× over Opt0). Slow case `[1,32,64,64] k=7` 91.3 ms → 2.31 ms (39× speedup) when row-parallel exposes K_total=1568 to the dispatcher. | **HIGH** |
| P-P79 | **Load-reverse trick to flip ReduceMax tie-break direction (P-P57 + stable-sort tie compatibility)** | sort | When a kernel uses P-P57 SIMD ReduceMax(calcIndex=true) for top-k AND must match a reference that uses `sort(stable=True, descending=False) + mask(value < kth_value)` (stable-ascending sort + mask), the tie-break directions disagree: Ascend `ReduceMax<calcIndex>` returns the **lowest** index for ties; reference's stable-ASC sort + mask keeps the **largest-original-index** at the kth boundary. fp16/bf16 quantization creates many ties, so ours and reference pick a different SUBSET of survivors → MERE/MARE = inf at the boundary positions. **Fix**: in-place reverse the working buffer `xf[0..N_pad)` after the Cast/Adds load step (e.g., scalar-loop swap or vector-rotate idiom). Pad NEG_SENTINEL at the FRONT instead of tail. ReduceMax now returns reduced_idx; convert back to original via `orig_idx = N_pad - 1 - reduced_idx`. Mask via `xf.SetValue(reduced_idx, NEG_SENTINEL)` so the next iter picks the next-largest. The reversed-coordinate "lowest-reduced-idx" maps to "largest-original-idx" — matches reference's tie convention exactly. **Generalizes**: any P-P57 user that needs PyTorch stable-ASC compatibility. **Anti-patterns** that don't work: secondary-key sort by -orig_idx within ties (more expensive); flipping dtype sentinel (orthogonal); expanding tie-inclusion buffer K_MAX_TIE alone (only matters when k_input == K_MAX, refuted on op#9 TopKTopP a3 — tied 0/24 → 0/24 fp16 cases). Validated op#9 TopKTopP a3 V200 2026-04-30 (aog-precision-probe `topktopp-pp-1` 8-iter probe): 22/50 → 43/50 (+21 cases) with load-reverse alone, then +1 with ascending-cumsum top-p walk = 44/50. Compare to P-P60 which addresses the same tie-direction issue for the V220-only `Sort<>` API; P-P79 is the V200-portable counterpart for ReduceMax-based top-k. | **CRITICAL** |

## Domain Files

| Domain | File | When to Load | Pattern Count |
|--------|------|-------------|---------------|
| precision | `domains/precision.md` | Always (mandatory audits) | 7 |
| scatter_add | `domains/scatter_add.md` | atomicAdd detected in scatter pattern | 8 |
| sort | `domains/sort.md` | Sort, argsort, topk operations | 12 |
| reduction_quant | `domains/reduction_quant.md` | Dynamic quantization, fused norm+quant, reduction | 3 |
| thread_utilization | `domains/thread_utilization.md` | Multi-dim decomposition or launch config | 4 |
| memory_access | `domains/memory_access.md` | Memory optimization opportunity | 9 |
| kernel_launch | `domains/kernel_launch.md` | Every kernel (basic compliance) | 5 |
| cooperative | `domains/cooperative.md` | Cooperative group / shuffle ops | 2 |
| platform_compat | `domains/platform_compat.md` | SIMD or platform-specific features, bf16 | 5 |

## Operator-Specific Files

| File | When to Load |
|------|-------------|
| `ops_specific/hkv_patterns.md` | Hash-table (HKV) operations |
| `ops_specific/pooling_sg_patterns.md` | Pooling / Sparse-Gather (future) |

## Unverified

| File | Description |
|------|------------|
| `unverified/candidates.md` | Candidates awaiting validation on 2+ operators |

## Always-Load References

| File | Description |
|------|------------|
| `../ASCENDC_LANGUAGE_REFERENCE.md` | SIMD sync (TQue/TBuf/PipeBarrier), SIMT sync (ThreadBarrier/CrossCore/atomics), mixed mode, HardEvent table |
| `../SIMT_VS_SIMD_DECISION.md` | **P-P9 full decision framework**: decision tree, 4 case studies, precision constraints (OL-30) |
| `../ASCENDC_SIMD_DEVELOPMENT_REFERENCE.md` | SIMD integer/bit-operation APIs, 950PR vs A3 differences, int32 type restrictions |

## Loading Protocol

```
1. Analyzer loads this INDEX (always, ~60 lines)
2. ALWAYS load ASCENDC_LANGUAGE_REFERENCE.md (covers both SIMD and SIMT)
3. Classifies source kernel → identifies relevant domains
4. Loads ONLY matching domain files (typically 2-3 files, ~200 lines total)
5. Loads ops_specific if operator type matches
6. Generator/Optimizer receive exact pattern IDs from Analyzer output
```
| P-P80 | **Vectorized index-compression emit via GatherMask for predicate-driven kernels** | scatter_add | For kernels of the shape "emit positions where predicate is true" (e.g. nonzero, where, scatter-with-mask, sparse compress), the canonical multi-core SIMD V4 pattern is: (1) `CompareScalar` over fp32-promoted source → packed bitmask UB (one bit per element, LSB-first); (2) `ArithProgression<int32>(pos_local, 0, 1, tile_aligned)` to materialize local positions [0, tile); (3) `GatherMask<int32, uint32>(pos_compressed, pos_local, ReinterpretCast<uint32_t>(mask_packed), reduceMode=true, mask=tile_size, params={1,1,0,0}, rsvdCnt)` — compresses positions where bitmask is set into a dense prefix; `rsvdCnt` returns count by reference; (4) `Cast<int64,int32>(flat64, pos_compressed)` + `Adds<int64>(flat64, base_offset)` to convert to global flat index; (5) per N-D dim, `Divs<int64>` / `Muls<int64>` / `Sub<int64>` chunked at 32B-aligned CHUNK_ROWS=256 for UB budget; (6) scalar pack into row-major out_ub + DataCopy bulk emit to GM. **Determinism**: GatherMask emits in increasing input-position order → A-P61 fixed-order multi-core merge ALLOWED when each block owns disjoint input/output ranges. **Caveats**: (a) ALL SIMD binary-scalar ops MUST use int64 (not int32) per PB-23; (b) chunk-base addresses MUST be 32B-aligned via fixed CHUNK_ROWS stride, NOT partial-chunk size (causes runtime error 340 unaligned access); (c) dense-case GM write bandwidth dominates — for sparse (<25% true) reaches ~0.5-0.9× CANN, but for dense workloads a different design (dim-major direct stream, no UB staging) is needed. Validated op#22 22_Nonzero V4 kw-5 (2026-04-30): probe-first verified GatherMask API contract (`probes/gathermask_probe/RESULTS.md`), then V4 kernel landed 50/50 + 10/10 + det 50/50 PASS, median 0.0108×, p90 0.5319×, max 0.9487× (case 19). **Generalizes to**: any "compress positions matching a predicate" kernel — torch.where, torch.masked_select, sparse coalesce, scatter-with-mask. **Cross-ref**: PB-23 (SIMD int32 reject), PB-20 (GM write workarounds for SetValue), P-P57 (SIMD ReduceMax for top-k), determinism.md A-P61 (fixed-order multi-core merge ALLOWED). | **HIGH** |
| P-P81 | **Runtime-bounded loop cap for top-K-style merge / scan with constexpr K_MAX buffer** | sort | When a kernel uses a `constexpr K_MAX` to size a top-K-style buffer (e.g. `TOPK_CAP = max_benchmark_k + tie_margin`), and the per-row `k` is a runtime scalar that varies across the batch, the for-loop bound for the merge/scan that fills this buffer should be `Align8(k_runtime + tie_margin)`, NOT `K_MAX`. The buffer itself stays sized at `K_MAX` (initialized once per row to a sentinel value such as `-inf`), but the work-loop terminates at the per-row bound. Snippet: `int32_t loop_cap = Align8(k_runtime + tie_margin); if (loop_cap > K_MAX) loop_cap = K_MAX; for (int32_t k = 0; k < loop_cap; ++k) { /* merge/scan */ }`. Buffer init MUST cover the full K_MAX so positions `[loop_cap..K_MAX)` read as sentinel (preserves downstream cumsum/threshold-walk invariants). Downstream copy-back uses `loop_cap` as count. **Why it works**: saves `(K_MAX − loop_cap) × ops_per_iter` of scalar-pipe work per chunk per row; for top-K-then-top-P sampling where `k` is typically 64–128 and `K_MAX` is 1088, this is an 8–15× work reduction in the merge stage. **Determinism (P-P61-class)**: `loop_cap` is a pure function of the runtime `k_runtime` scalar, identical across repeat runs; per-row path width varies but each row's path is deterministic. P-P61 4-prong (single-AIV-per-row, no atomic, hardware Sort + scalar merge, queue-rotated output) is unaffected. **Anti-pattern boundary**: if `k_runtime ≥ K_MAX − tie_margin` for ALL rows (i.e. everyone uses `k = K_MAX − tie_margin` so `loop_cap` saturates at `K_MAX`), the optimization yields zero gain. Useful only when the `k`-distribution has variance. **Generalizes** to any constexpr-capped iterative reduction whose true work is per-row variable: top-K, top-P, beam search prefix, segment-sum cap, masked nucleus. Validated op#9 9_TopKTopP ko-1 (2026-05-02 Ascend950PR_9579): scalar_ratio 0.94→0.78 (fp32 small k=64), 0.91→0.72 (fp16 mid k=128); wall-clock −75% / −70% on small/mid; bf16 large k=1024 unchanged (loop_cap saturates at K_MAX). Pass A 16/16 bit-exact + canonical det 50/50 preserved. Median ratio 0.271×→0.397× (+47%). **Cross-ref**: P-P59 (TOPK_CAP sizing for ties — sets the K_MAX), P-P60 (Sort tie direction), P-P79 (load-reverse for ReduceMax-based top-k tie). | **HIGH** |
| P-P83 | **Column-major intermediate output + pybind transpose to eliminate per-element scalar pack in predicate-driven N-D coord emit** | scatter_add | When the V4 GatherMask emit pattern (P-P80) is bottlenecked on the per-element scalar pack from column-major coord UB → row-major output GM (per-element interleave at ~10–15 scalar cycles per int64 dimension), the **column-major intermediate output composite** eliminates the scalar pack entirely. **Kernel side**: allocate output GM as `[ndim, numel]` column-major instead of `[numel, ndim]` row-major; after SIMD coord decode produces `coord64[d * CHUNK_ROWS + r]` per dim, emit each dim's contiguous chunk via aligned `DataCopy(outputGm[d * numel + my_offset + emitted], coord64[d * CHUNK_ROWS], chunk_aligned)` (chunk_aligned = `(chunk/4)*4` for int64 32B alignment; tail <4 elements via uint32 split-AtomicAdd on pre-zeroed slots). No `out_ub` row-major staging buffer needed — saves UB and eliminates per-element scalar copy. **Pybind side**: allocate `torch::empty({ndim, numel}, kInt64, kNPU)`, kernel writes column-major, then `output_cm.slice(1, 0, K).transpose(0, 1).contiguous()` returns `[K, ndim]` row-major to caller. Pybind transpose is a single GM→GM strided copy of `K * ndim * 8` bytes by torch (~400 GB/s HBM → ~2.5 ms for 67M-row dense). **Determinism (A-P61)**: each block writes disjoint `[d * numel + my_offset, ... + my_count)` per dim — no race; within block, GatherMask emits in increasing pos order, SIMD decode is order-fixed, DataCopy is bulk MTE3 deterministic; pybind transpose deterministic per torch contract. **When to use**: predicate-driven emit kernels with N-D coord decode (N > 1) where the V4 P-P80 pattern is profile-confirmed bottlenecked on the scalar interleave. **When NOT useful**: ndim==1 (no interleave to avoid); kernel-natural layout already matches output contract (rare for index ops); stretch perf demand requires fused single-pass transpose-fused kernel (different architecture). Validated op#22 22_Nonzero V5 kw-2 (2026-05-03 Ascend950PR_9579): Pass A 50/50 + Pass B 10/10 bit-exact, Det 50/50, perf overall 0.3563× (median 0.2832×) vs V4 baseline 0.1603× (median 0.0368×) → **2.22× cumulative speedup**, 7.7× median improvement. Some sparse cases now beat CANN (max 1.79×). **Generalizes** to any N-D index-emit kernel sharing the column-major-coord-UB → row-major-GM scalar-pack bottleneck (where, masked_select, sparse coalesce, scatter-with-mask, sparse-COO build). **Cross-ref**: P-P80 (the V4 pattern this composite improves on — same kernel surface), PB-23 (SIMD int32 reject — int64 indices required throughout), determinism.md A-P61 (block-disjoint write ordering preserved). | **HIGH** |
| P-P85 | **`AscendC::TopK` adv_api primitive — use this instead of hand-rolled chunked-Sort + scalar 2-pointer merge for k-selection on arch 3510/5102/3003/3113** | sort | When implementing TopK / TopKTopP / quantile / threshold-mask kernels on Ascend950PR (arch 3510) and similar, the public `AscendC::TopK<T, isInitIndex, isHasfinish, isReuseSrc, topkMode, config>` from `adv_api/topk/topk.h` (host helpers `GetTopKMaxMinTmpSize` + `TopKTilingFunc` from `adv_api/sort/topk_tiling.h`) delegates to vec-pipe-bound primitives (likely the private `KernelVbsMergeSort` family in `opp/built-in/op_impl/.../arch35/merge_sort_simd.h`). **dtype matrix** — `TopKConfig::algo`: `RADIX_SELECT` supports ALL int + half + float + **bf16**; `MERGE_SORT` supports half + float ONLY (no bf16). For bf16 hot-path ops RADIX_SELECT is mandatory; for fp16/fp32 benchmark both. **Snippet**: `TopKConfig cfg{ TopKAlgo::RADIX_SELECT, TopKOrder::LARGEST, /*sorted=*/true }; TopKTilingFunc(platform, /*inner=*/N, /*outter=*/B_per_AIV, k_runtime, sizeof(T), false, TopKMode::TOPK_NORMAL, true, cfg, tiling);` host-side, then `AscendC::TopK<T, false, false, false, TopKMode::TOPK_NORMAL, cfg>(sortedVal, sortedIdx, srcVal, dummyIdx, finishLocal, tmpLocal, k_runtime, tiling, info, true);` kernel-side. **Anti-pattern (current op#9 9_TopKTopP Phase 1 pre-kw-4)**: chunked `AscendC::Sort<>` over CHUNK_LEN=2048 per chunk + scalar 2-pointer merge into TOPK_CAP buffer → `aiv_scalar_ratio=0.898` (scalar-pipe bound) → 1024 us/row for [N=65536, k=1024] bf16. **CANN's own `npu_top_k_top_p` internal Sort kernel** (measured via msprof on a single fused call): `aiv_vec_ratio=0.721`, 204 us/row for the same shape — **5× faster**, vec-pipe-bound, almost certainly using the adv_api or equivalent vec-merge primitive. **Determinism risk (medium)**: RADIX_SELECT may have its own tie-break ordering — characterize with a 5-rep det-check before adopting on DET_POLICY=required ops. **Generalizes** to any kernel currently doing chunked-Sort + scalar-merge for k-selection: rejection-sampling top-K, beam-search prefix, threshold-mask + scatter, sparse-select. **Cross-ref**: P-P81 (runtime-bounded loop cap — orthogonal optimization once the merge primitive is replaced); EC-33 (RADIX_SORT in `Sort<>` chunked context still defensive-MERGE per EC-33; AscendC::TopK::RADIX_SELECT is a DIFFERENT API path from `Sort<>::RADIX_SORT` — does NOT trip EC-33 per pp-2 measurement); P-P84 (analytical decomposition methodology — anti-pattern fixed by P-P85's "use the adv_api primitive" approach). Validated empirically op#9 9_TopKTopP pp-2 (2026-05-03 Ascend950PR_9579, CANN 9.0.0 b103) — measured 3.35× total gap is recoverable, NOT a structural ceiling. kw-4 spawn pending. | **CRITICAL** |
| P-P86 | **Empirical-measurement-required for fused-op gap analysis — analytical-only decomposition is reward-hacking on the diagnostic side** | platform_compat | When estimating "CANN's per-sub-op cost" for a fused op, analytical-only decomposition (estimate "CANN's Phase-X equivalent" by selecting a standalone CANN op that LOOKS LIKE Phase X, then compute analytical gap) is unreliable AND can be wildly wrong. **Op#9 9_TopKTopP fo-1 vs pp-2 evidence (2026-05-03 Ascend950PR_9579)**: fo-1 estimated `npu_top_k(k=1024)` standalone at ~7 us/row → claimed "146× gap on Phase 1" → declared structural ceiling. pp-2 directly msprof'd `torch_npu.npu_top_k_top_p` and found CANN's internal Sort kernel runs at **204 us/row** (full-row sort, not k=1024-only). Real gap = **5× on Phase 1, 3.35× overall** — **fo-1's claim was off by ~30×**. The analytical estimate was wrong because fused ops decompose internally into sub-ops with DIFFERENT shapes than the user-facing standalone equivalents (CANN's `npu_top_k_top_p` does full-row sort then apply-on-sorted; standalone `npu_top_k(k=K)` does only top-K — different cost model). **Methodology rule (mandatory for fused-op gap analysis)**: profile BOTH (1) `torch_npu.<fused_op>(...)` directly with msprof to observe internal kernel decomposition (kernel names, BlockDim, dur, pipe ratios per internal kernel), AND (2) each kernel name from (1) — that's the actual sub-op-level reference cost. If only fused-op msprof is available, per-row dur is still tighter than analytical estimates: divide by BlockDim × per-AIV row count. **Sustained-call EC-33 mitigation for the standalone-ref measurement step**: each measurement should run in a FRESH Python process (multi-process msprof), not back-to-back in one process — pp-2 measured this directly (M2 process aborted after 3 sub-ops succeeded clean within one process; M1+M1b in fresh processes were stable). **Block conditions (when this rule applies)**: any time aog-fused-optimizer / aog-kernel-optimizer / orchestrator is about to declare "structural ceiling" or "perf plateau" on a fused op. Analytical-only verdict is INSUFFICIENT — must include measured per-internal-kernel msprof of the fused reference op. **Generalizes**: any multi-stage fused op where the user-facing standalone equivalents may use different internal algorithms (Softmax+Mul, RmsNorm+Cast, RoPE+Cache, etc.). Validated op#9 fo-1 vs pp-2 cross-comparison 2026-05-03. **Cross-ref**: P-P84 (the analytical methodology — now downgraded as fallback-only when measurement is impossible); aog-self-critic C27 + new C28 (this rule is C28's empirical-evidence requirement). | **CRITICAL** |
| P-P99 | **Manual-Mmad B-operand contraction axis = the L1 tile's C0/inner dim; A and B must source the contraction from the SAME axis** | platform_compat | For hand-rolled tile-MMAD (`Mmad` + `LoadData`-based L1→L0 loads, NOT `MatmulImpl<>`): the 2D-params `LoadNzL1ToZnL0B` (`LoadData2DParams`, `ifTranspose=false`) makes the L0B **contraction (k) axis = the L1 tile's C0/inner (column) dim** — i.e. the `dValue` used in the GM→L1 `Nd2Nz` load. The 2D-params `LoadNzL1ToZzL0A` does the same for A. A and B must therefore source the contraction from the **same physical axis** — either BOTH from C0 (plain loads), or BOTH from the ROW dim (matched transposed loads). A **mixed** pairing (A contracts over its C0 while B contracts over its ROW, or vice-versa) mis-aligns the L0A/L0B fractals → wrong/zero output even though the logical `mp.k` value matches. **Decision rule**: identify each operand's contraction axis relative to its L1 layout. (a) contraction == C0 dim → plain `LoadNz...` (no transpose); (b) contraction == ROW dim → either a transposed load (`LoadDataWithTranspose` / `LoadData2DParams{ifTranspose=true}`, V220-verified for the K^T side) OR the 3D `LoadData3DParamsV2` form (`mExtension`=contraction, `channelSize`/`kExtension`=output-n) which contracts over the L1 ROW axis. Whatever form A uses, B must use the **matching** form so both source k consistently. **Anti-pattern**: pairing a non-transposed A (k from C0) with a transposed/3D B (k from ROW) — the empirical failure was correct-magnitude-but-wrong dq. **Evidence**: lightning_indexer_grad (A3 V220, 2026-05-27) — dgk = dscores^T @ Q (contract over N1): A=dscores^T (trans, k from ROW) + B=Q (trans, k from ROW) → correct (dk ~0.1%). dq = dscores @ gk (contract over topK): A=dscores (plain, k from C0) + B=gk (trans, k from ROW) → wrong; matching both to the 3D form (the cv-agent FA BMM2 P@V mechanism: A=P k-from-C0, B=V k-from-ROW, both 3D) was required. **Cross-ref**: P-P100 (multi-C0-block tiling — orthogonal axis-size concern), CAND-FA1 (`LoadData2DParams{ifTranspose=true}` for K^T verified form). `applies_to: soc=Ascend910_9382 (V220/A3); cann=9.0.0; unverified_on: V351/A5`. | **HIGH** |
| P-P100 | **Hand-rolled Mmad: output-n and contraction-k axes spanning >1 C0 block (>BASE_K=16) MUST be tiled in BASE_K chunks (+accumulate for k)** | platform_compat | For hand-rolled tile-MMAD, a single `LoadData`+`Mmad` whose **output-n** or **contraction-k** axis spans more than one C0 block (>BASE_K=16, e.g. D=64 = 4 blocks) mis-packs the fractals: a too-wide **n** load spreads the output over a 2× stride (observable as alternating-zero columns in the output); a too-wide **k** in one Mmad mis-contracts. **Rule**: tile both axes in BASE_K(=C0) chunks. For **n** (output): loop `ni` over `nTiles=nAlign/BASE_K`, B-load one C0 col-block per tile, `Mmad` with `mp.n=BASE_K`, `Fixpipe` to `out[.., ni*BASE_K]`. For **k** (contraction): loop `ki` over `kTiles=kAlign/BASE_K`, slice each operand's L1 col-block at offset `ki*stride*BASE_K`, `Mmad` with `mp.k=BASE_K` and `cmatrixInitVal=(ki==0)` (init on first tile, accumulate after via the 4-arg `Mmad(c,a,b,c,mp)`). Both loops reduce to a single iteration (the prior single-shot path) when the axis ≤16, so small-dim cases are unaffected. This mirrors cv-agent FlashAttention: BMM1 tiles the head-dim contraction, BMM2 tiles the output-n. **Evidence**: lightning_indexer_grad (A3 V220, 2026-05-27) — all D=16 cases passed with single-shot loads; D=64 cases (4 C0 blocks) failed on every output. Tiling C1's k=D and C3a/C3b's n=D in BASE_K chunks (k with accumulate) fixed the D=64 cases while leaving D=16 bit-identical. **Cross-ref**: P-P99 (contraction-axis SOURCE — orthogonal; this is about axis SIZE). `applies_to: soc=Ascend910_9382 (V220/A3); cann=9.0.0; unverified_on: V351/A5`. | **HIGH** |
| P-P101 | **De-scalarize flash online-softmax — replace `SoftmaxFlashV2`/scalar `RowMuls`/`RowDivs` with hand-rolled mem-based VEC online-softmax + precision-safety triad** | precision+memory_access | Attention/FA softmax that is scalar-bound (`aiv_scalar` dominant pipe: A5 FA 0.27–0.31 inside SoftmaxFlashV2, A3 FA 0.503). Standard `LocalTensor` vec ops (`WholeReduceMax`/subtract-rowmax-then-`Exp`/`WholeReduceSum`/`Muls`/`Div`) replace the scalar pole; **precision-triad** (masked→finite `minValue` not −inf / subtract-rowmax-BEFORE-Exp / vector-`Div`-with-sum>0) fixes the inf bug; **pitfall** stat-buffer aliasing (V-pipe `Brcb` aliasing the MTE3 sm-emit → sm_sum corruption → inf). **Perf SCOPED (kernel≠wall)**: kernel-msprof −24% but e2e-wall NEUTRAL (host-fold+pybind dominate, see OL-201) — MEASURE wall decomp, don't assume kernel-win reaches e2e. Arch-independent in principle (V220+V351). Full mechanism: `fa_class/cv_reference_concrete_params.md`. Verified A5/CANN-9.0.0 precision 5/5 (5a3a1cee,697ed8c8). `applies_to: soc=Ascend950PR; cann=9.0.0; op_class=attention-softmax; unverified_on: Ascend910_V220 (arch22 — arch-independent in principle; A3 precision/e2e not yet measured, independent prototype cross-pollination pending)`. | **HIGH** |
| P-P102 | **Cube-class A5 ports use native MIX (Mmad cube + vector epilogue + WorkspaceQueue cross-core sync), never pure-vec** | platform_compat | For a `port_a3_to_a5` op whose CANN reference is cube-required (matmul/attention/conv/rnn/gmm/ffn family — tagged `CUBE_MIX` by `_cmd_port_a3` Layer 1, enforced by finalize gate `_check_architecture_class` per OL-188/PR#316: pure-VEC = `ARCHITECTURAL_HACK`). **Scaffold**: file split `<op>_cube.h`(cube class `Cube` in name) + `<op>_vec.h`(`Vec`) + `<op>_kernel.h`/`.cpp` orchestrator; cube primitive = manual `AscendC::Mmad` (`cmatrixInitVal=true` single-K-tile; FA chose `Mmad` over `matmul::Matmul<>` after V220 ~500× numerical error + KFC standalone deadlock); vec epilogue = op reduction/activation (FA: `SoftmaxFlashV2`); cross-core sync = `WorkspaceQueue<T,RING_SLOTS=3>` ONE per producer↔consumer direction (FA:3), paired flag IDs, **raw `PIPE_FIX`/`PIPE_MTE3`/`PIPE_MTE2` literals** (CANN 9.0.0 forbids templated `pipe_t`; NOT inline `CrossCoreSetFlag` in loop → 507015); task type `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` (V220-native, PB-28 FALSIFIED — don't arch-guard out); regbase per-AIV sub-block tiling under V351 248KiB UB. **WHY** (adapt-not-copy): AIC/AIV are separate cores; sync is a counting-semaphore handshake pairing one AIC producer↔AIV consumer per direction with producer-PIPE↔consumer-PIPE matching — reproduce the contract (one queue/direction, PIPE pairing, `i%K`-staggered schedule per OL-200), not FA tile sizes. **Methodology**: understand(V220 src)→KB(this pattern)→research(A5 ref for understanding, anti-copy net = ARCH35_WRAP_CHEAT + copy-shape scanner)→generate→regenerate on reject (NEVER pure-vec fallback). Worked example reached via THIS KB, **NOT** another op's `output/` archive (own-dir-only guard `b96508ec`). Full body: `patterns/domains/cube_vector_fusion.md`. Cross-ref OL-188/OL-190/OL-200/EC-57/EC-58/PB-35. **Harness-linkage root cause (OL-235)**: the manual-Mmad default is not only the FA numerical/deadlock evidence — `build_ascendc.py`'s pybind link line is fixed to `kernels torch_npu m dl`, so the host `TCubeTiling` that a `matmul::Matmul<>` kernel needs (`MatmulApiTiling::GetTiling`) is never linkable at the launch layer ⇒ matmul-library cube path is structurally unbuildable through this harness for ALL CUBE_MIX ports (not just FA). `applies_to: soc=Ascend950PR/V351; cann=9.0.0; op_class=CUBE_MIX (matmul/attention/conv/rnn/gmm/ffn); verified_on=flash_attention_score A5 (decision_manifest 2026-05-29); deformable_conv2d port_a3 2026-06-20 (CUBE_MIX conv-family, empirical pybind-link probe → manual-Mmad + bilinear-deform vec MIX, structural_rewrite verdict)`. | **HIGH** |
| P-P103 | **FlashAttention-class advisory template knowledge — arch35 cube-MIX FA interfaces, recipe, tiling, and sync** | flash_attention | For FA-class arch22→arch35 generation. Target templates preserve block interfaces, phase order, host-tiling categories, and completeness hypotheses; they are advisory only. Emit task-owned code from the selected arch22 contract and current arch35 public APIs, then validate against source-NPU truth. A copied target body or target output cannot close generation. Full body: `patterns/domains/fa_class_template.md`. | **HIGH** |
| P-P104 | **GMM SwiGLU Quant A8W8-class template — reproducible A5 cube-MIX grouped-matmul-swiglu-quant framework (recipe phase-order + host/kernel tiling + stitching spec-map + 12-variant X-macro)** | grouped_matmul | For any GMM SwiGLU Quant A8W8 op on A5 (grouped-matmul-swiglu-quant family). The worked template: MIX_AIC_1_2 Cube↔Vector pipeline with CrossCoreSetFlag/WaitFlag(0x8/0x9) handshake + Backpressure depth=14 + BasicBlock global indexing + ProcessDSQ fused dequant-swiglu-quant body (manual SiLU⊙Gate 5-op + per-token WholeReduceMax quant). Spec-map: 3 dequantDtype × 2 wFormat × 2 transB = 12 kernel variants via X-macro, 3-in-lockstep. Host tiling: CalcBasicBlock(baseM=128/N=256/K=128) + CalcUBFactorDimX(N→{1,2,4}) + CalcWorkspaceSize(M*N*4+20MB) + SelectA8W8Launcher dispatch. K2 invariant: host CalcBasicBlock ↔ kernel template params 1:1. Meta-lessons: V1 split/unsplit dead weight, cross-core sync arch35-specific, SwiGLU formula canonical SiLU⊙Gate, quant uses 1/127 multiply not divide (CAND-PP103). Cross-ref: **P-P102** (cube-MIX scaffold), **P-P70** (fused dequant→activation→quant pipeline), **CAND-V351-AIV-WholeReduceMax-fp32-mask-cap** (fp32 mask=64 cap), **CAND-PP102** (two-kernel split broken on V351). Full body: `patterns/domains/gmm_swiglu_quant_a8w8_class_template.md`. `applies_to: soc=Ascend950PR/V351; cann=9.0.0; op_class=grouped-matmul-swiglu-quant/CUBE_MIX (A8W8 path)`. | **HIGH** |
| P-P116 | **FlashAttention-class MIX template (a3/arch22) — hand-authored cube+vector MIX_AIC_1_2 attention starting skeleton, device-proven** | flash_attention | For any a3 (Ascend910_9382, arch22) FA-class / attention-fwd op needing a hand-authored cube+vector MIX kernel — the **a3 counterpart** of P-P102/P-P103 (which are a5/arch35 ONLY). Device-proven skeleton (DS `famix`/`famix_mh`, Ascend910_9382): `S=Q@Kᵀ`(cube#1, AIC) → `P=softmax(S/√d)`(vector, AIV, fp32 row-wise) → `O=P@V`(cube#2, AIC). **Dispatch**: MIX_AIC_1_2 via standard AscendC AIC+AIV device objects + `aclrtlaunch` (FFTS descriptor auto-supplied via `rtGetC2cCtrlAddr`); **do NOT** emit `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` on arch22 (arch35-only macro → 107000; the a5-native macro P-P102/103 KEEP, a3 must OMIT). **Cube**: `MatmulImpl<> IterateAll<sync=true>` (NEVER async KfcServer `Iterate()/GetTensor()` → PB-34), runtime transpose-B (static ISTRANS false). **Handshake**: `CrossCoreSetFlag<MODE2,PIPE_FIX>(FLAG_S=4)` fwd (broadcast) / **BOTH AIV subblocks** `CrossCoreSetFlag<MODE2,PIPE_MTE3>(FLAG_P=5)` reverse — reverse is per-subblock-COUNTED (PB-55: single-setter DEADLOCKS); flag ids distinct, NEVER 0 (PB-35); MODE2 suffices (NOT arch35 §4 mode-4). **Scratch**: `S`,`P` `[seq,seq]` fp16 in GM; softmax one row at a time in UB; multi-head = device loop over `nheads=B*H`, one reused S/P pair (FLAG chain serializes). Genuine cube both matmuls (pure-VEC = OL-188 hack, forbidden). **HONEST SCOPE** device-verified: seq≤384, d=64, fp16, single-pass NON-flash softmax, multi-head, cos 0.999999, deterministic; NOT proven: flash online-softmax KV-tiling / causal mask / perf. Closes the a3 half of DEBT-222. Full body: `patterns/domains/fa_class_a3_mix_template.md`. Cross-ref PB-55/PB-34/PB-35/OL-188/OL-235/P-P101/P-P102/P-P103. `applies_to: soc=Ascend910_9382; cann=9.1.0; op_class=attention-fwd/CUBE_MIX (a3 hand-authored); verified_on=Ascend910_9382 famix/famix_mh (DS 2026-07-18); unverified_on: Ascend950PR`. | **HIGH** |
| P-P117 | **Chunked gated-delta-rule / gated-linear-attention forward — a3 MIX multi-chunk recurrence (per-head sequential chunk loop + persistent GM state `S[D,D]` + ON-DEVICE per-chunk decay-fold so every kernel matmul is a plain `A@B`)** | fa_class | For a chunked gated-delta-rule / gated-linear-attention forward on a3 (Ascend910_9382, arch22; arbitrary T, GQA, multi-batch). Per head, loop chunks `c=0..ceil(T/64)-1` **sequentially** with persistent state `S[128,128]` in GM (init 0). The kernel folds per-chunk operands **ON-DEVICE** (AIV Stage 0 — NO host torch compute; the no-delegation rule requires all compute in AscendC) from `gc=cumsum(g)` (in-chunk scalar prefix-sum), `eg=Exp(gc)`: `kb=k*beta*eg`, `knT=(k*exp(-gc))^T`, `qs=scale*q*eg`, `vb=v*beta`, `kdT=(k*exp(gc_last-gc))^T`, `sc=eg[C-1]=exp(gc_last)` — into scratch slots `S_KB/S_QS/S_VB/S_KNT/S_KDT`, so EVERY kernel matmul is a plain `A@B`. Fold uses `Broadcast<T,2,1>` (row-scale kb/qs/vb) + repeat-`Mul` broadcast (col-scale transposed knT/kdT, the sibling `recurrent_gated_delta_rule::MatVecMul` idiom); fp32 UB, `CAST_RINT` to fp16 GM. Host does layout marshaling only (GQA head-expand, zero-pad, chunk-reshape, cast, head-major permute incl. raw `kT`). **Do NOT fold on the host with torch cumsum/exp** — an earlier host-fold variant scored 4 delegation violations. Zero-pad `T→Nc*64` + per-chunk cumsum makes the padded tail carry `gc_last` automatically and padded rows/cols vanish under causal masking (no explicit last-chunk fill). Within-chunk: `Acc=kb@knT`; `L=-strict(Acc)`; `T=(I+strict(Acc))^-1` via Neumann power product `prod(I+L^{2^k})` k=0..5 (see CAND-GDR-1 for the solve-sign gotcha); `Am=incl(qs@knT)`; `U=T@vb`. Cross-chunk: `W=T@kb`; `WS=W@S`; `o_inter=qs@S`; `vn=U-WS`; `o=Am@vn+o_inter`; `S=S*sc+kdT@vn`. `Nc==1` (S=0) collapses to single-chunk. MIX: matmuls on AIC (`MatmulImpl`+`IterateAll<sync=true>`, arch22-safe NON-KFC cube per P-P68 — reuse one mm object needs `SetOrgShape` per call, CAND-GDR-3), elementwise on AIV (incl. the Stage-0 on-device decay fold), whole-device `SyncAll<false>()` sync (see **PB-57: needs `KERNEL_TYPE_MIX_AIC_1_1`**; intra-AIV UB reuse needs a `PipeBarrier<PIPE_ALL>` fence, CAND-GDR-2). GQA via host `repeat_interleave` head-expand (layout). **Device-verified 16/16 @ fp64 customer gate (rtol=0.02, ≥17× headroom, deterministic, NaN-free), 9–28× device-time vs torch_npu** (a3/Ascend910_9382, DS 2026-07-20). Full body: `patterns/domains/gated_delta_rule_a3_recurrence.md`. Cross-ref P-P68 (NON-KFC cube), P-P116 (a3 MIX sync template), PB-57 (1:1 macro), CAND-GDR-1/2/3/4. `applies_to: soc=Ascend910_9382 (a3/arch22); cann=9.1.0; op_class=gated-linear-attention/CUBE_MIX; verified_on=gated_delta_rule fwd a3 (DS 2026-07-20); unverified_on: Ascend950PR`. | **HIGH** |
| P-P118 | **Chunked gated-delta-rule / gated-linear-attention BACKWARD — a3 MIX 3-pass reverse-recurrence (parallel PASS A using the passed-in state h, one REVERSE-recurrence PASS B for dstate, per-chunk PASS C for the step5/6 grads; ON-DEVICE decay-fold + reverse-cumsum(dg) + GQA head-sum + matmul-with-ones reductions)** | fa_class | For the BACKWARD of a chunked gated-delta-rule / gated-linear-attention on a3 (Ascend910_9382, arch22; arbitrary T, GQA, multi-batch) — six gradients `dq,dk,dv,dg,db,dh0` from `(q,k,v,g,beta,A,h,do,dht,initial_state)`. **Two math simplifications**: (1) use the passed-in recurrent state `h` DIRECTLY (the reference recomputes it identically) → NO in-kernel forward sweep, `vn=u-w@h[c]`, PASS A is parallel per-chunk; (2) only step4 (`dstate`) is a cross-chunk recurrence, and it runs in REVERSE chunk order (PASS B); steps 1/3/5/6 are per-chunk. **3 passes** (per head, blockdim=nHead, MIX_AIC_1_1): PASS A (fwd c) `w=A@kbg; u=A@vb; vn=u-w@h[c]; AmT=mUpInc⊙(kn@qsT); dv=AmT@do; dsi=qs^T@do`; PASS B (REVERSE c) `dh[c]=dstate; dv[c]+=kd@dstate; dstate=dstate*sc+dsi-w^T@dv` (fp32 GM accum), `dh0=dstate`; PASS C (fwd c) step5+step6 per-chunk matmul assembly (verified vs model.py — signs/masks/transposes + the order `dg5` from `dq5` PRE-`ds@k`). **ON-DEVICE (all compute in AscendC, NO host torch)**: per-chunk operand fold via `Exp()` (`g` arrives already per-chunk cumsum'd so kernel does exp only); `dg`/`db` reductions as matmul-with-ones → `[C,16]` fp32 accumulators (host col0); `reverse-cumsum(dg)` in-chunk scalar prefix-sum on UB; GQA head-sum(dq,dk) a device compaction stage. **Do NOT host-fold with torch exp/cumsum/sum** — an earlier host-fold bwd variant scored delegation violations pinpointed in `op_host`. Transposes via runtime `SetTensorA/B(bool)` (P-P69). Partial chunk (T=200): no fill needed (that case has `dht=0`; causal mask + zero-pad vanish the tail). MIX substrate same as fwd: NON-KFC cube (P-P68, `SetOrgShape` per reused-mm call = CAND-GDR-3), intra-AIV UB fence (CAND-GDR-2), whole-device `SyncAll<false>()` (PB-57: needs `MIX_AIC_1_1`). **Device-verified 11/11 @ fp64 customer gate (rtol=0.02 per-gradient, ~14-18× headroom, deterministic, NaN-free, input-mutation-safe), 2.22–11.25× vs the PyTorch-NPU reference** (a3/Ascend910_9382, DS 2026-07-20). Backward-specific gotchas: CAND-GDR-BWD-1 (dsMask2 swapaxes = head relocation not [C,C] transpose), CAND-GDR-BWD-2 (`.clone()` dstate init or it aliases + mutates input dht), CAND-GDR-BWD-3 (thread `initial_state` when building the passed-in h in the test harness). Full body: `patterns/domains/gated_delta_rule_bwd_a3_recurrence.md`. Cross-ref P-P117 (forward), P-P116 (a3 MIX sync template), P-P68, P-P69, PB-57, CAND-GDR-2/3, CAND-GDR-BWD-1/2/3. `applies_to: soc=Ascend910_9382 (a3/arch22); cann=9.1.0; op_class=gated-linear-attention-backward/CUBE_MIX; verified_on=gated_delta_rule bwd a3 (DS 2026-07-20); unverified_on: Ascend950PR`. | **HIGH** |
| P-P119 | **Many-small-matmul a3 MIX: emit a HAND-ROLLED single-block cube primitive (`Nd2Nz→LoadData3D→Mmad→Fixpipe[F322F16]`) per matmul, NOT `MatmulImpl::IterateAll` — the library's per-call tiling scalar setup DOMINATES (aic_scalar-bound, not MAC-bound)** | fa_class | For an a3 (Ascend910_9382, arch22) MIX **chunked-recurrence / gated-linear-attention** op (P-P117) where EACH chunk performs MANY SMALL matmuls (`M,N,K ≤ 128`, one L0C tile, no inner K-loop; a gated_delta_rule chunk runs ~8+ such matmuls × Nc chunks × heads → hundreds-to-thousands of tiny matmuls per op). `MatmulImpl::IterateAll<sync=true>` (P-P68 NON-KFC cube) is correct for a FEW LARGE GEMMs, but for MANY SMALL matmuls its per-call tiling/address scalar setup is fixed-cost and DOMINATES the tiny `≤128³` MAC work → the cube becomes **aic_scalar-bound**. **msprof evidence** (a3/Ascend910_9382, GDR fwd, IterateAll version): `aic_scalar ≈32%` / `aic_mac ≈3.7%` / `cube_util ≈17%` (~9× more time in scalar tiling setup than actual MACs — the per-call-overhead signature). **Fix**: emit a hand-rolled single-block cube per matmul — `Nd2Nz` (GM/UB→L1, ND→NZ) → `LoadData/LoadData3D` (L1→L0A/L0B) → `Mmad` (single pass, ≤128³) → `Fixpipe[F322F16]` (L0C fp32→fp16). Block dims compile-time-known → NO tiling scalar setup per call. Reuse the exact `matmul_primitive` param rules from `fa_class/cv_reference_concrete_params.md` (Mmad 4-arg accumulate + per-context `Fixpipe srcStride`: `/C0` for L0C→workspace, ELEMENTS for L0C→GM; wrong unit → 507015 ECC read) — do NOT re-derive. **Measured** (device, DS 2026-07-20): this change ALONE closed a ~1.3× compute-bound deficit vs the cv-reference to PARITY (T4096 −35%, T1024 −30% device-time), precision unchanged (still 16/16 @ fp64). **Decision rule**: few-large → IterateAll (P-P68); many-small (≤128, hundreds+, chunked-recurrence) → hand-rolled primitive, confirm via msprof `aic_scalar ≫ aic_mac`. Precision-invariant across both (same Mmad math) — a PERF lever, never correctness. Full body: `patterns/domains/a3_mix_small_matmul_cube.md`. Cross-ref P-P117 (the many-small call-site), P-P116 (a3 MIX sync template), P-P68 (the IterateAll it replaces for many-small / keeps for few-large), `fa_class/cv_reference_concrete_params.md` (matmul_primitive params), CAND-GDR-3. `applies_to: soc=Ascend910_9382 (a3/arch22); cann=9.1.0; op_class=gated-linear-attention/CUBE_MIX/many-small-matmul; verified_on=gated_delta_rule fwd a3 (DS 2026-07-20); unverified_on: Ascend950PR`. | **HIGH** |
| P-P120 | **Neumann / (I+P) triangular-solve L0C-fold via accumulate-mode Mmad (`cmatrixInitVal`): compute `R@(I+P)=R+R@P` in ONE L0C pass with NO separate +I vector-add** | fa_class | For an a3 (Ascend910_9382, arch22) cube computing an iterative **(I+P)-style matmul chain** — canonically the Neumann power-product triangular solve `A=(I+x)^-1=Π_i(I+x^{2^i})` (each step `R_next=R@(I+P)`, `P=x^{2^i}`) used to invert `(I+strict-lower)` in chunked gated-delta-rule / gated-linear-attention. Naive lowering emits a cube `R@P` PLUS a vector `Add` of the identity/additive term PLUS the MIX barriers between them — ~9 AIV ops for a 5-step chain, pure overhead since the cube can carry the additive term. **Fold on the cube via `MmadParams.cmatrixInitVal`**: (1) first Mmad with `cmatrixInitVal=1` seeds the L0C accumulator with the identity/additive term (`R`, i.e. `R@I`) — NO vector op; (2) second Mmad with `cmatrixInitVal=0` accumulates `R@P` onto the SAME L0C → `R+R@P=R@(I+P)`; (3) hold ONE L0C accumulator across the product chain, Fixpipe only at the end. Same `cmatrixInitVal=(ki==0)` accumulate idiom as the FA K-tile loop (`fa_class/cv_reference_concrete_params.md::matmul_primitive`), repurposed to inject an ALGEBRAIC additive term instead of accumulating K-tiles. **Measured**: cuts AIV ops + barriers to ~6 vs ~9 for a 5-step Neumann chain; precision unchanged (additive term exact in the fp32 L0C accumulator — more faithful than an fp16 round-tripped vector add). Reference STRUCTURE: cv-ref `gated_delta_rule` NeumannSolve. Watch the solve SIGN (CAND-GDR-1) — the fold is sign-agnostic, so the `(I+strict)^-1` vs `(I-strict)^-1` sign bug is orthogonal and handled at the operand level. Full body: `patterns/domains/a3_mix_small_matmul_cube.md`. Cross-ref P-P117 (the `T=(I+strict)^-1` solve site), P-P119 (sibling cube-emit decision), `fa_class/cv_reference_concrete_params.md` (cmatrixInitVal params), CAND-GDR-1. `applies_to: soc=Ascend910_9382 (a3/arch22); cann=9.1.0; op_class=gated-linear-attention/CUBE_MIX/neumann-triangular-solve; verified_on=gated_delta_rule fwd a3 (DS 2026-07-20); cross-witness=gated_delta_rule CV-fusion (customer Kimi-K3, 910B2C/220x/CANN8.5.1, 8.85× geomean, PR#200, 2026-07-20); unverified_on: Ascend950PR`. | **HIGH** |
| P-P105 | **Boundary-clamped interpolation: a single OOB→0 gather reproduces zeros/border/reflection padding when the clamped coordinate's overflow neighbor always carries zero interpolation weight** | memory_access | For linear/bilinear/trilinear sampling ops where the coordinate is first mapped into `[0, size-1]` by a per-mode clamp (border = clamp, reflection = reflect, zeros = identity): a single fetch helper returning `0` for out-of-bounds indices reproduces ALL THREE padding modes without per-mode neighbor clamping. **Why**: after the clamp, the floor neighbor `x0` is always in-bounds; the `+1` neighbor only lands OOB (at exactly `size`) when the clamped coord sits on the integer upper boundary, where the interpolation weight `dx == 0` — so the OOB neighbor's contribution is `weight * 0 = 0` regardless. Full body below. | MEDIUM |
| P-P106 | **Serial-in-L linear recurrence on AscendC: L-chunk to bound UB + Hillis-Steele parallel associative scan over the affine pair within each chunk, with an L-major `[l*N+n]` layout so one L-shift is one wide Mul/Add across all N lanes** | scan | For a serial linear recurrence `x[l] = a[l]·x[l-1] + b[l]` (SSM/Mamba selective-scan, prefix-products, cumulative ops) where `L` exceeds the UB budget: (1) **L-CHUNK** to bound UB — chunk ≤ CH, carry scan-state across chunks via a prefix fold at position 0 and read the chunk's last element out as the next prefix; (2) within each chunk replace the O(L) serial per-`l` loop with the O(log₂ chunk) **Hillis-Steele** parallel scan over the affine pair `(a,b)` combined as `(a2,b2)⊗(a1,b1) = (a2·a1, a2·b1+b2)`. **Layout key**: store `[l*N+n]` L-major → an L-shift by `stride` is an element-shift of `stride*N` → ONE wide `Mul`/`Add` per stride-pass across ALL N lanes (no per-N-lane loop) = high vector-util, the win for small-N + large-L. **Inclusive Hillis-Steele needs NO pow2-pad** for a ragged last chunk (pow2-pad is a Brent-Kung concern — verify which scan variant you use). **Forward AND backward**: the same HS covers the forward (prefix, left-to-right) scan AND the backward/adjoint (suffix, reverse, right-to-left) scan in the gradient — direction is the only difference. **HS micro-opt**: drop the protective shift-copy of any operand that is read BEFORE the write that would clobber it (the `b`-update reads the old prefix before the `Add` writes it) → 5→4 vec ops/pass, precision-neutral. A/B (selective_scan L=5000): fwd fp32 1.71× / fp16,bf16 1.93× vs serial-within (both L-chunked); bwd 2.69× (reverse-suffix HS + micro-opt, PR#37); ratio grows with L. **A5 selection rule (sub-case)**: pick the scan with the SHALLOWEST dependent-barrier DEPTH, not the least total work — A5 inserts a PipeBarrier per dependent vec step so per-step LATENCY dominates a memory-bound scan; work-efficient O(L) serial (256-deep chain) measured 2–5× SLOWER than O(log L) Hillis-Steele (8-deep), and row-batching does NOT rescue a linear-depth chain. Full body below. | HIGH |
| P-P107 | **Tiled reduction+normalization via GM workspace (3-pass streaming) when the reduction span exceeds UB** | sort | Per-row softmax/L1/L2-normalization over a vector V longer than UB: 3 streaming passes through a per-row GM workspace carrying only two scalars (globalMax, globalSum) across tiles — Pass1 tiled ReduceMax running-max → Pass2 tiled Sub(max)+Exp+ReduceSum, materialize exp to GM → Pass3 tiled Div(globalSum) from GM. Replaces a fixed `MAX_V_UB` buffer that silently overflows/garbles or 507035-crashes at V>cap. UB↔GM DataCopyPad works on V351 (EC-23). Full body in `patterns/domains/sort.md`. `applies_to: soc=Ascend950PR; cann=9.0.0; op_class=normalization/sampling/long-vector-reduce`. | HIGH |
| P-P108 | **Iterative top-K / argmax selection from a GM-resident buffer — valid ONLY for bounded top-K (k≪V); ANTI-PATTERN for nucleus/top-P (O(V²))** | sort | ⚠️ ANTI-PATTERN for nucleus/top-P. Mechanism: when the candidate buffer lives in GM (V>UB), P-P57's UB-resident k×ReduceMax loop becomes per-iteration `FindGlobalMaxFromGM` (tile-scan DataCopyPad GM→UB + ReduceMax(calcIndex=true), running (val,globalIdx)) → record → `MaskGmValue` (load the one tile holding idx, SetValue -inf, DataCopyPad UB→GM, **fence MTE3_MTE2** so next load sees the mask); final selected-index gather from a 2nd V-buffer = scalar `qGm.GetValue(rowId*V+idx)` (O(k) reads). Cost O(k·V). **VALID only for bounded top-K (k≤1024, e.g. Branch A). For nucleus/top-P, k=topPNum≈p·V → O(V²)**: measured 134.3 ms @ V=32000 fp16 (Ascend950PR, NPU 0, 2026-06-26) vs ~6 ms for the Sort+cumsum approach (P-P109) → ~21× (38× @ V=8192), widening with V. The kw-5 iteration hid this via a peaked test distribution (topPNum=2) — a cheat, NOT a technique. **USE P-P109 (hardware Sort + cumsum, O(V log²V)) for nucleus/top-P.** DEBT-169 tracks the Sort re-impl. Full body in `patterns/domains/sort.md`. `applies_to: soc=Ascend950PR; cann=9.0.0; op_class=selection/topk/sampling`. | **HIGH** |
| P-P109 | **Hardware Sort + cumsum nucleus finding (O(V log²V)) for top-P / Branch-C sampling — the correct replacement for P-P108's O(V²) iterative selection** | sort | Nucleus (top-P) selection over a per-row V-length softmax buffer in GM (V≫UB): find the smallest prefix of the descending-sorted distribution whose cumsum first exceeds `p`. (1) SEGMENT SORT — chunk V into ~1024-elem segments, carry original index [0..V-1], `Sort<float,DESC>` per chunk → descending (value,index) runs in GM; (2) k-WAY MERGE + CUMSUM — `MrgSort4Que` (≤4 runs) or iterative `MrgSort`/`MrgSortMQue` (>4 runs, GM ping-pong) merging descending while `ReduceSum`-cumulating, `GatherMask` to de-interleave (value,index); stop at first position where cumsum>p → `topPNum`; (3) EMIT `sortValGm[0..topPNum]` (desc values) + `sortIdxGm[0..topPNum]` (orig indices) + count. Drop-in for P-P108's consumer contract (no downstream change). Deterministic (per-row, no atomicAdd); verify Sort ASC tie order if exact boundary indices matter (P-P60). **Evidence (2026-06-26, NPU 0, fp16)**: our O(V²) iterative = 134.3 ms vs sort+cumsum decomposed = 5.98 ms @ V=32000 → ~21× (38× @8192, 11× @4096), widening with V; kernel idx == decomposition idx (B=4 V=2048 sanity). 5.98 ms is an UPPER BOUND (≈10 CANN launches); a fused AscendC kernel (DEBT-169) is faster still. Full body in `patterns/domains/sort.md`. `applies_to: soc=Ascend950PR; cann=9.0.0; op_class=sampling/nucleus/topk`. | **HIGH** |
| P-P84 | **Sub-op gap decomposition via msprof PipeUtilization anchor + analytical scalar-op counting (fused-optimizer methodology when standalone refs are unreliable)** | platform_compat | For multi-stage fused ops where running per-sub-op standalone CANN refs (`npu_top_k`, `npu_softmax`, `npu_cumsum`, etc.) is unreliable due to EC-33-class instability, NPU lane contention, or shared-resource concerns, decompose per-row dur into per-sub-op contributions analytically. **Procedure**: (1) take ONE high-quality msprof PipeUtilization profile of the dominant case — record `aiv_scl_ratio`, `aiv_vec_ratio`, `aiv_mte2_ratio`, `aiv_mte3_ratio`, total dur, rows-per-AIV; (2) per-row dur = total_dur / rows_per_aiv; per-pipe budget = pipe_ratio × per_row_dur; (3) for each sub-op, count scalar ops (GetValue/SetValue/Set+Get pairs) by code inspection — calibrate per-op latency from observed dur (typically 3–6 ns/op on Ascend950PR scalar pipe); (4) per-sub-op scalar contribution = scalar_op_count × calibrated_latency; vector contribution via cycle accounting (e.g. ReduceMax over N ≈ N/64 + log₂(64)); (5) sub-op classification: BOTTLENECK if contribution > 50% of dominant pipe budget; UNCOMPARABLE if no standalone ref exists for the sub-op's full role (e.g. row-max into shared accumulator across phases); NECESSARY if structurally required (e.g. emit phase doing full-row scatter). **When applicable**: fused op where `npu_<X>` standalone ref exists but cannot be measured cleanly; per-AIV serial work dominates (single-AIV-per-row); bottleneck dominated by scalar pipe (analytical counting gives high-confidence cost). **Limitations**: ±2× per-cell precision (qualitatively decisive for finding dominant cell, not for ranking near-equal cells); doesn't help when standalone refs ARE reliable — direct measurement is better. **Anti-pattern complement**: When the localized bottleneck is scalar-pipe-bound (`aiv_scl_ratio > 0.6`), buffer-aliasing optimizations have ~zero ROI even if the aliasing is correct — recovered UB doesn't help unless it unlocks a tile/chunk-size increase that itself unlocks vector work. Op#9 fo-1 found 19.4 KB recoverable from idle multi-AIV buffers + 8.8 KB from sortedVal/Idx ↔ mergeTmp aliasing — ~28 KB total. CHUNK_LEN expansion 2048→4096 would consume 16 KB more (fits!), but archive iter 2 already tried this and reverted for **precision** (Sort intrinsic drift on >2048 elements), independent of UB. So even with the alias fix, the obvious downstream lever isn't safe. Validated op#9 9_TopKTopP fo-1 (2026-05-03 Ascend950PR_9579): localized 81% of dur to Phase 1 inner loop (32 chunks × 1088 merge_cap × 6 scalar ops × ~5ns ≈ 740 us/row of 1264 us/row total). Saved 4× standalone-CANN benchmark calls that would have hit EC-33 truncation. Verdict: structural ceiling 0.385× confirmed via 3rd independent diagnostic angle. **Cross-ref**: MSPROF_AGENT_GUIDE.md (PipeUtilization extraction), EC-33 (sustained-call instability rationale for analytical decomposition), OL-82 (scalar_ratio thresholds). | **HIGH** |
| P-P82 | **Deterministic cross-AIV tournament merge of P partial top-K buffers via composite key (value DESC, orig_idx ASC)** | sort | When partitioning a wide row across P AIVs (each AIV produces a per-partition top-K_partial buffer sorted DESC by value), the final K outputs need merging into a globally-sorted top-K. **Cross-partition ties** (positions in different partitions with the same value) must resolve by global `orig_idx` to match `torch.sort(stable=True)` semantics, regardless of which AIV finished first. A naive tournament using only value comparison is non-deterministic when ties exist. **Solution**: composite key `(value DESC, orig_idx ASC)` — strict total order on outputs because `orig_idx` is unique per row by construction. Snippet: `bool wins = (v > best_v) || (v == best_v && i < best_i);` inside the per-output-position scan over P partial heads. **Determinism guarantee**: composite key gives strict total order on `(value, orig_idx)` pairs; the merge output is bit-exact regardless of AIV scheduling order. Combined with P-P61 4-prong (single-AIV-per-partition Phase 1 + `SyncAll<true>()` Phase 1.5 + no atomicAdd), the entire kernel remains deterministic by construction. **Novel coverage** vs siblings: P-P61 covers single-AIV-per-row det; P-P79 covers load-reverse tie-direction within a single AIV; **P-P82 is the bridge** that allows multi-AIV reductions to remain deterministic AND match PyTorch stable-sort. **Generalizes**: any cross-AIV merge of K-sorted streams that must match PyTorch stable-sort tie semantics (multi-core top-K, histogram quantile reductions, segmented sort merges, beam-search aggregations). **Activation gate**: only fires when `B < TOTAL_AIV` so multi-AIV partition is profitable (see OL-124). For `B ≥ TOTAL_AIV` the partition path falls back to single-AIV-per-row P-P61. Validated op#9 9_TopKTopP kw-2 (2026-05-03 Ascend950PR_9579): Pass A 16/16 bit-exact + det 50/50 PRESERVED; tournament merge correctly implemented. Perf neutral on B≥56 harness (architectural fallback fires); architecturally forward-compatible for any future B<56 wide-N case. **Cross-ref**: P-P61 (single-AIV-per-row determinism — the per-partition prerequisite), P-P79 (load-reverse intra-AIV tie-break), P-P81 (runtime-bounded loop cap — orthogonal cap optimization), OL-124 (multi-AIV-per-row activation gate B<TOTAL_AIV). | **HIGH** |
| P-P88 | **Hand-rolled transcendental via Cephes-form range reduction with fp32-grade primitives — when an end-to-end transcendental kernel hits a precision ceiling AND vendor source confirms the algorithmic reformulation** | precision | **Three independent evidence streams (cross-validated 2026-05-07)** — (1) **PB-24/25 isolated-primitive measurement on A5 (Ascend950PR)**: `Tanh<fp32>` has a bimodal floor — clean 2 ULP for `|x| ≥ 0.1`, catastrophic small-x failure (1599 ULP at x≈1.7e-4, 2.7M ULP at x=1e-7) due to polynomial that doesn't preserve `tanh(x)≈x` near zero. `Sigmoid<fp32>` is a clean 2-ULP uniform floor. (2) **A3 (Ascend910 V220) cross-arch confirmation, DS-side 2026-05-07**: same bimodal Tanh floor (max 4 ULP for `|x| ≥ 0.1`, up to 906 ULP in `[1e-4, 0.1]`) — failure mode is **chip-family-wide**, not arch-specific. Sigmoid same uniform 2-ULP behavior. The primitive failure mode for transcendental kernels is small-x identity loss, not the previously-imagined saturation cancellation. (3) **Vendor-source evidence (P0aad)**: CANN's own arch35 GELU (`~/workspace/cann/ops-nn/activation/gelu/op_kernel/arch35/gelu_dag.h`) does **NOT** call the AscendC `Tanh` primitive at all. It implements GELU as `x / (1 + exp(-1.5957691·0.044715·(x/0.044715 + x³)))` using only `MicroAPI::Mul / Axpy / Muls / Exp / Adds / Div` — exactly the sigmoid-form reformulation of `0.5x(1+tanh(y))`. fp16/bf16 paths cast input → fp32 (`Vec::Cast<float, U, CAST_MODE_NONE>`), run the fp32 sigmoid-form DAG, then cast back with `CAST_MODE_RINT`. This is **independent vendor-source confirmation** that the right answer for transcendental gelu/swish-class ops is sigmoid-form via `Exp + Add + Div`, NOT the AscendC `Tanh` primitive. Whether `Tanh` ceilings at 1 ULP or CANN simply prefers the reformulation for perf is irrelevant — the algorithmic choice is empirically validated by reading the vendor kernel. **Honesty caveat retained**: probe-reported "Tanh 1-ULP ceiling" on op#1 (2026-05-04) was inferred from end-to-end kernel measurement, not from isolated `Tanh(x)` vs fp64 measurement; an isolated-primitive probe is still the correct evidence form for *generalizing* this pattern to other transcendentals (Erf, Atan, etc.). For GELU specifically, vendor-source evidence is sufficient to invoke P-P88. Established public math libraries (Cephes, fdlibm, libm, ARM Compute Library) document range-reduction algorithms that achieve sub-ULP precision using only fp32-grade primitives (`Exp`, `Reciprocal`, `Add`, `Mul`). **Canonical Cephes-form for Tanh** (large `\|y\|`): `tanh(y) = 1 - 2/(exp(2y) + 1)` eliminates the `(1+tanh(y))` cancellation in the saturation band that the public `Tanh` primitive can't avoid. **Canonical Cephes-form for Sigmoid**: `sigmoid(y) = 1/(1+exp(-y))` directly via `Exp + Reciprocal + Add` — does NOT use AscendC `Sigmoid` (fp16-grade). **Workflow**: (1) probe identifies precision-bottlenecked primitive via API substitution test (e.g. op#1 GELU iter 1 swapped Tanh→Sigmoid identity, regressed → primitive-ceiling diagnosed); (2) WebSearch the algorithm class (`tanh range reduction Cephes`, `IEEE 754 fp32 transcendental`); (3) verify the algorithm's primitives are fp32-grade per OL-103 §Refined-statement (Exp/Reciprocal/Add/Mul confirmed; Sigmoid/Tanh hit ceilings); (4) implement from primitives, NOT via the public transcendental API. **Determinism**: pure arithmetic + Exp, no transcendental primitive — by-construction deterministic per A-P61. **Op-class examples**: 1_GELU fp32-tanh saturation band (-4.5 < x < -3.0), Swish/SiLU activation, BatchNorm + sigmoid, stable softmax via `exp(x - max(x))`. **Anti-pattern boundary**: don't apply when the AscendC primitive IS already sub-ULP (e.g. Exp itself — Cephes-form would just re-derive Exp). **Sources**: Cephes (netlib.org/cephes), fdlibm, libm, ARM Compute Library activation kernels, Beebe "Accurate Hyperbolic Tangent Computation" (math.utah.edu/~beebe/software/ieee/tanh.pdf), arXiv 2008.02078, RLIBM (Rutgers VSS 2025). **Cross-ref**: OL-103 §Refined-statement (per-primitive precision ceiling table — `Tanh` fp32-grade-but-1-ULP, `Sigmoid` fp16-grade, `Exp/Reciprocal/Add/Mul` fp32-grade), OL-85 anti-overfit (this is NOT case-specific — it's a general remediation class), aog-researcher Phase R-B step 5 (P0aac 2026-05-06 — researcher brief now mandates public-numerical-algorithm literature search when probe verdict cites primitive ceiling). **Origin** (P0aac 2026-05-06): user 2026-05-06 caught the harness blind spot — kw / ko / probe / researcher all missed this strategy on 1_GELU because briefs framed the question as "what does CANN do?" not "what do public math libraries do?". This pattern entry + ar_brief Phase R-B step 5 close that gap. **Enforcement (P0abi 2026-05-08): MANDATORY-on-match for PB-24 (Tanh) only.** Scope narrowed post-DS portfolio scan (2026-05-08): PB-24 `Tanh<fp32>` has the BIMODAL small-x failure (1599 ULP at x≈1.7e-4) that produced the cold-start non-monotonicity regression — schedule-sensitive to tile size. PB-25 `Sigmoid<fp32>` is uniform 2-ULP, no bimodal cliff, no schedule-sensitivity → sigmoid-form rewrite is RECOMMENDED for vendor-source-alignment but NOT structurally enforced. When kernel emits `AscendC::Tanh` (any namespace/template variant) AND op-class is transcendental (gelu/silu/sigmoid/tanh/softmax/erf-bearing), the finalize gate (`finalize_pipeline._check_pp88_compliance` invoking `scan_pp88_compliance.scan_workspace`) REQUIRES a structured `p_p88:` YAML block in `knowledge_update.md` with `status: applied` (+ non-empty `diff_refs` to the rewrite) OR `status: exempt` (+ `evidence.isolated_primitive_measurements` proving the small-x failure regime doesn't reach this op). Citing P-P88 as diagnosis without applying remediation = rejected. Origin: 1_GELU regressed 50/50 (May-4 archive PASS_WITHIN_TOLERANCE) → 44/50 (May-8 cold-start PARTIAL) when kw cited P-P88 but kept the `Tanh()` call; different tile-size choice (4096 → 6144) routed Tanh's internal SIMD differently on small-value inputs, exposing PB-24's bimodal floor. The gate makes cite-vs-apply structural — kw can't ship a Tanh-using transcendental kernel without YAML evidence either way. | **HIGH** |
| P-P87 | **Vendor adv_api regbase primitive substitution requires per-call batch axis A>1 to amortize internal scalar-broadcast cost** | platform_compat | When considering substituting hand-rolled per-row code with a vendor `AscendC::<Primitive><U,T>` adv_api call (Normalize, LayerNorm, RowMuls, Softmax, Logit, etc.), the perf advantage materializes ONLY when the dispatch shape supplies multiple rows (A>1) per primitive call. Vendor c310 regbase impls (e.g. `normalize_c310_impl.h`) use `Reg::LoadAlign<DIST_BRC_B32>` to broadcast scalar inputs (mean, rstd, gamma, beta, scale, etc.) into vector registers across A rows in ONE call — the broadcast-load cost is fixed per call, so per-row cost scales as `(broadcast_cost / A) + per_row_compute`. At A=1 the broadcast happens once per row → no amortization → vendor primitive performs same as (or worse than) hand-rolled code. **Decision rule (mandatory pre-substitution check)**: (a) for **pure precision improvement** (e.g. CPU-truth alignment via vendor's bit-canonical regbase output), A=1 substitution is net-positive — the regbase impl produces bit-identical results and never regresses precision; (b) for **perf improvement**, FIRST verify the call-site dispatch can supply A>1. If outer loop is per-row (A=1, each `Normalize` call processes 1 row of K elements), the substitution is NOT a perf lever — needs upstream restructure (Kind-2 batch rewrite: load K rows into one buffer of shape [A, K], single batched call, scatter results back). **Snippet**: vendor signature is typically `<Primitive><U,T>(dst[A][K], src[A][K], scalars[A], ...)` — the leading A axis IS the amortization axis; never call with A=1 expecting perf gain. **Anti-pattern (op#10 LayerNorm kw-2-this-session 2026-05-05)**: substituted `AscendC::Normalize<{half\|float\|bfloat16_t}, ...>` at 3 single-pass sites within a `K_ROWS_PER_AIV` outer loop dispatching A=1 per call → Pass A 60/60 preserved + Pass B 10/16 → 16/16 BIT-EXACT (precision improved via regbase output) + Det 60/60 preserved + **Perf 0.19× = baseline (no improvement)**. Diagnosis: outer loop dispatches A=1 per Normalize, so `Reg::LoadAlign<DIST_BRC_B32>` of mean+rstd+gamma+beta runs once per row with no amortization. Vendor LayerNormV4 perf advantage requires batched A=K rewrite, not exposed by single-row substitution. **Generalizes** to: any vendor adv_api regbase primitive (Normalize, LayerNorm, Softmax, RowMuls/RowAdds, RmsNorm, GroupNorm, etc.) where the impl uses `Reg::LoadAlign<DIST_BRC_*>` for scalar broadcast — these all have the same A>1 amortization gate. **Cross-ref**: OL-54 (Reg-based SIMD overview + adv_api impl-header path caveat + docstring-vs-static_assert caveat); P-P62 (Row-Scalar VEC Multiply via Brcb — same amortization principle for hand-rolled multi-row scalar Mul; precondition R≥8 rows batched); OL-89 (vendor primitive substitution opportunities in the analyzer phase). | **HIGH** |
## P-P89: GM workspace contract for fused ops — public outputs stay separate; opaque scratch is one aligned byte workspace sliced by host offsets
`applies_to: any soc with __gm__ pointer arithmetic; cann=9.0.0+; op_class=fused_with_aux_output`
`derived-from: cann-source (FA-class workspace layout convention, 2026-05-09)`
`unverified_on: a5_ops`

`applies_to: any soc with __gm__ pointer arithmetic; cann=9.0.0+; op_class=fused_with_aux_output | multi_stage_fused`
`derived-from: cann-source (FA-class workspace layout convention, 2026-05-09)`
`a5_ops_anchor: 3_FusionAttention emits auxiliary public outputs (softmax_max/softmax_sum) as separate tensors; packed single-workspace scratch remains convention-level, not yet fully shipped in a5_ops`

**Trigger**: Fused op has (a) multiple cross-stage GM scratch tensors, such as matmul scratch, post-activation probabilities, partial accumulators, layout-conversion temporaries, or (b) auxiliary public outputs needed by later passes/backward restore, such as FA `lse`/softmax stats, fused-dropout `mask`, softmax+CE `logsumexp`.

**Recommendation**: Use a two-tier memory contract.

1. **Public outputs** are caller-visible tensors. Allocate and return them as normal torch/CANN outputs, pass each output pointer separately, and bind each to its own `AscendC::GlobalTensor<T>::SetGlobalBuffer(...)`. Public outputs must not be hidden inside opaque workspace, because tests, callers, autograd restore, and shape contracts need torch-visible tensors.

2. **Workspace scratch** is kernel-internal and opaque to the caller. Allocate one `params.workspace` byte buffer. Host tiling computes a packed layout and passes byte offsets in tilingdata. Every typed scratch offset must be alignment-padded:

```cpp
off_qk      = 0;
off_probs   = AlignUp(off_qk + qkBytes, 512);
off_partial = AlignUp(off_probs + probsBytes, 512);
totalBytes  = AlignUp(off_partial + partialBytes, 512);
```

Kernel entry slices the workspace via `__gm__ uint8_t*` pointer arithmetic:

```cpp
auto wsBase = reinterpret_cast<__gm__ uint8_t*>(params.workspace);

gQKScratch.SetGlobalBuffer(
    reinterpret_cast<__gm__ float*>(wsBase + tiling.offQk));
gProbsScratch.SetGlobalBuffer(
    reinterpret_cast<__gm__ half*>(wsBase + tiling.offProbs));
gPartialOut.SetGlobalBuffer(
    reinterpret_cast<__gm__ float*>(wsBase + tiling.offPartial));

// Public outputs stay separate:
gO.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(params.o));
gLse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.lse));
```

3. **Pybind contract**: pybind queries host tiling / `<op>_workspace_size(...)`, allocates `torch::empty({totalBytes}, kInt8, kNPU)`, and passes `workspace.data_ptr()` plus each public output’s `data_ptr()` to the launch. The workspace tensor must remain alive until the launched work using it has completed on the relevant stream. Do not assume “pybind return” equals kernel completion unless the launch path/allocator gives stream-ordered lifetime guarantees; otherwise synchronize or retain ownership through the stream work.

**Why it matters**:
- Reduces N scratch allocations to one device allocation.
- Keeps deterministic, shape-derived workspace sizing in host tiling.
- Preserves public-output visibility while avoiding scratch leakage into the Python API.
- Gives a uniform convention for multi-stage fused kernels and aux-output fused kernels.


## P-P91: Multi-variant kernel organization — sibling `<op>_<variant>.h` files dispatched by tiling-key from a thin `.cpp`

> **Source**: cann-learner CAND-A3A5-4 (promoted 2026-05-12 from PR4778 cross-op-evidence batch, 5 ops).

`applies_to: soc=all; cann=9.0.0; bisheng=15.0.5; op_class=all`
`verified_on: soc=Ascend950PR (PR4778 arch35 ports — 5 ops with variant-split confirmed); soc=Ascend910_V220 (the SAME 5 ops' V220 master state ALSO uses this convention — pattern pre-dates A5 port; PR4778 preserves it)`
`unverified_on: soc=Ascend910_V220 in a3/a2 from-scratch op-gen context (we observed it in PORTED ops; if a3 kw decides to write a SINGLE mega-template kernel that violates this convention, the precision/perf cost is unknown — no counter-evidence yet)`
`note: this pattern is a CODE-ORGANIZATION rule, not a hardware-tied rule. Applies wherever multiple algorithmic variants live in the same op_kernel/ directory regardless of target SoC. The V220 master state already follows it; A5 arch35 port preserves it. From-scratch a3/a2 op-gen via aog-kernel-worker should follow it too — but we have not validated the negative case empirically.`

**Principle**: when a kernel has multiple algorithmic variants (different dim layouts, dtype-divergent codepaths, scatter vs gather phases, etc.), keep variants in **sibling header files** under `op_kernel/` (or `op_kernel/arch35/` for A5 ports) and dispatch via TILING_KEY in a thin top-level `.cpp`. **Don't merge variants into one mega-template kernel** even when bodies share 80% of code.

A5's wider regbase MicroAPI surface tempts authors to write one mega-template kernel that switches via `if constexpr` on every axis. The master-state convention (preserved through A5 port) is more readable AND produces better object-file structure (per-variant `.o` files, smaller per-launch binary).

**Concrete anchor**:
```cpp
// op_kernel/<op>.cpp (top-level dispatcher — thin)
#include "<op>_scalar.h"
#include "<op>_transpose.h"
#include "<op>_last_dim.h"
#include "<op>_common.h"
extern "C" __global__ __aicore__ void op(GM_ADDR ..., GM_ADDR tiling) {
    GET_TILING_DATA(td, tiling);
    if (TILING_KEY_IS(0)) { OpScalar<...> op; op.Init(...); op.Process(); }
    else if (TILING_KEY_IS(1)) { OpTranspose<...> op; op.Init(...); op.Process(); }
    else if (TILING_KEY_IS(2)) { OpLastDim<...> op; op.Init(...); op.Process(); }
}
```

**Evidence** (cross-op, 5 ops):
- `gather_elements_v2`: 4 variant files (scalar, transpose, last_dim, common)
- `index_put_with_sort`: 3 phase files via inheritance (base, gather_data, scatter_data)
- `apply_adam_w_quant`: 2 dtype-split files (fp16, fp32) + shared base
- `top_k_top_p_sample_v2`: 3 files (main, comm, sort_cumsum) — see also OL-133
- `group_norm_silu_quant`: 2 files (base, b16)

**Sub-patterns**:
- **Dtype-split (sub-case)**: when fp16 + fp32 algorithms differ in buffer count / quantization LUT / accumulator dtype, splitting by dtype is cheaper to port than templating because per-dtype divergences span the whole Process() body. Evidence: `apply_adam_w_quant_fp16.h` + `apply_adam_w_quant_fp32.h`; `group_norm_silu_quant_b16.h` + `group_norm_silu_quant_base.h`. (Originally CAND-A3A5-6, folded into P-P91.)
- **Sort + cumsum split**: sort algorithm in dedicated header (`<op>_sort_cumsum.h`); main flow in `<op>.h`; shared types in `<op>_comm.h`. Evidence: `top_k_top_p_sample_v2`. (Originally CAND-A3A5-13, folded into P-P91.)

**Anti-pattern (DO NOT)**:
- Single mega-template kernel with `if constexpr` on every axis — harder to debug, larger per-launch binary, defeats per-variant TILING_KEY dispatch.
- Mega-template using `template<bool IsTranspose, bool IsLastDim, int Dtype>` — A5's regbase MicroAPI surface MAKES THIS POSSIBLE but the master-state convention says don't.

**Cross-ref**: OL-133 (`ASCENDC_TPL_ARGS_DECL` for compile-time axis enumeration — complementary to P-P91; use TPL_ARGS_DECL to declare WHICH variants exist, P-P91 to organize WHERE they live).


## P-P92: Multi-phase op via inheritance chain — phases share state via class-hierarchy, dispatch by TILING_KEY in same .so

> **Source**: cann-learner CAND-A3A5-7 (promoted 2026-05-12, Mode 5 batch 2). C36 lift applied — op-class generalized from "index_put_with_sort" → "multi-phase-scatter-gather" (covers scatter-with-pre-sort, gather-then-scatter, segmented-reduction with intermediate workspace, etc.).

`applies_to: soc=all; cann=9.0.0; bisheng=15.0.5; op_class=multi-phase-scatter-gather`
`verified_on: soc=Ascend950PR (index_put_with_sort/op_kernel/arch35/)`
`unverified_on: soc=Ascend910_V220 — pattern observed in 1 op's A5 port; V220 master uses different organization (may or may not benefit from this convention)`

**Principle**: when an op decomposes into 2-3 distinct algorithmic phases (e.g. sort-then-scatter, gather-then-process, segmented-reduce-then-finalize) that need to share intermediate state, organize the kernels as a **class inheritance chain in the SAME `.so`** dispatched by TILING_KEY values 0/1/2. State flows between phases via class member fields (the derived class sees the base class's accumulators directly), NOT via cross-kernel GM-workspace passing.

This is preferable to a multi-kernel approach (separate `.so` files for each phase, GM workspace handoff) because:
1. **No cross-kernel synchronization** — phases run in the same kernel-launch context with implicit ordering by TILING_KEY dispatch.
2. **Shared UB state** — intermediate buffers persist across phases without writeback-then-reload.
3. **Smaller binary footprint** — one `.so`, one launch.
4. **Simpler debugging** — single stack frame; inheritance chain is greppable.

**Concrete anchor** (from `index_put_with_sort/op_kernel/arch35/`):
```cpp
// base.h — common state + helpers
class IndexPutWithSortBase {
protected:
    GlobalTensor<int32_t> sortedIndicesGm;  // shared across phases
    LocalTensor<half> workBuffer;
    // ... init common state
};

// gather_data.h — inherits base, implements gather phase
class GatherDataOp : public ScatterDataInKernelOp {  // chains to scatter
    void Process() { /* gather, leaves results in base member */ }
};

// scatter_data.h — inherits base, implements scatter phase
class ScatterDataInKernelOp : public IndexPutWithSortBase {
    void Process() { /* scatter using sortedIndicesGm from base */ }
};

// In <op>.cpp dispatcher:
if (TILING_KEY_IS(0))      { IndexPutWithSortBase op; op.Init(...); op.Process(); }
else if (TILING_KEY_IS(1)) { GatherDataOp op; op.Init(...); op.Process(); }
else if (TILING_KEY_IS(2)) { ScatterDataInKernelOp op; op.Init(...); op.Process(); }
```

**Anti-pattern (DO NOT)**:
- Two separate `.so` files (one per phase) communicating via GM workspace — adds a kernel-launch boundary + double-write of intermediate state.
- Single mega-template kernel with `if constexpr` per phase — defeats the per-variant TILING_KEY dispatch.
- Using virtual functions in the inheritance chain — A5 kernel objects must be POD/concrete.

**Other instances (predicted)**: segmented-reduction ops (e.g. `MoeFinalizeRouting` could use this), sparse-to-dense ops with sorting prologue, any op with a "preprocess → main-compute" split where preprocess outputs structure the main-compute access pattern.

**Cross-ref**: P-P91 variant-split (P-P92 is the inheritance-based variant of variant-split — use P-P92 when phases share state, P-P91 when variants are independent dtype/layout dispatches); patterns/domains/scatter_add.md (scatter-add primitives that the scatter phase invokes).

## P-P93: Quant-op CPU reference MUST `.clamp(low, high).to(int_dtype)` to match NPU hardware clamp

`applies_to: soc=Ascend950PR,Ascend910_V220; cann=9.0.0; bisheng=all; op_class=quant,fused-quant; phase=precision-verify`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 references/precision-testing/OPS_PRECISION_STANDARDS.md "迁移场景特殊考虑 / 量化算子(Quant)精度测试设计规范"`

**Pattern**: When writing CPU reference / golden for a quant op (output dtype is INT8 / UINT8 / INT4-in-INT8), the reference **MUST** clamp to the output dtype's representable range BEFORE the integer cast. NPU hardware clamps automatically; without matching clamp in the reference, out-of-range cases produce false-FAIL on the cast result.

**Concrete pattern**:

```python
# ❌ WRONG — silent saturation difference NPU vs CPU
def cpu_reference_quant_int8(x_fp32, scale):
    pre = x_fp32 / scale
    ref = torch.round(pre).to(torch.int8)   # CPU: overflow → undefined; NPU: clamps to ±127
    return ref

# ✅ RIGHT — explicit clamp matches NPU clamp
def cpu_reference_quant_int8(x_fp32, scale):
    pre = x_fp32 / scale
    ref = torch.round(pre).clamp(-128, 127).to(torch.int8)
    return ref
```

**Clamp range table by output dtype**:

| Output dtype | clamp range | code |
|---|---|---|
| INT8 | [-128, 127] | `.clamp(-128, 127).to(torch.int8)` |
| UINT8 | [0, 255] | `.clamp(0, 255).to(torch.uint8)` |
| INT4 (stored as INT8) | [-8, 7] | `.clamp(-8, 7).to(torch.int8)` |
| FP8 E4M3FN | [-448, 448] (max representable, hardware clamp at ±448 saturating) | `.clamp(-448, 448).to(<fp8>)` |
| FP8 E5M2 | [-57344, 57344] | `.clamp(-57344, 57344).to(<fp8>)` |
| HiFloat8 | n/a (built-in range-encoding, no clamp needed) | `.to(<hifloat8>)` direct |

**Core insight** (PR 103 OPS_PRECISION_STANDARDS.md "核心认知"):
1. **quantScale value need NOT be restricted** — even when quantScale is tiny and most values overflow INT8 range, BOTH NPU and CPU clamp to ±127 → saturation region results agree.
2. **CPU reference MUST add clamp** — to align with NPU behavior; otherwise PyTorch's `.to(torch.int8)` on overflow is **undefined behavior**, not equivalent to saturation.
3. **Precision diff source**: ONLY in the non-overflow region, from FP16/BF16 vs FP32 intermediate compute differences. After `round`, max 1 ULP diff per element.

**Why this matters for us**:
- Cohort 1 archived ops: `rms_norm_quant` (8/8 PASS T1) + `group_norm_silu_quant` (Pass A 8/8 T1, Pass B 2/7 T1 + 5/7 T2). If those tests used unclamped reference, the 5/7 T2 cases on `group_norm_silu_quant` Pass B may be hiding clamp-mismatch instead of genuine fp16/bf16 precision difference. **Worth re-verifying.**
- Cohort 2 quant ops pending: `add_rms_norm_quant`, `flat_quant`, `grouped_matmul_swiglu_quant`, `fused_quant_mat_mul` — all need this rule in their test harness from day 1.
- Upcoming fp8 / mxfp8 / mxfp4 kernels: clamp values per OL-144 narrow-float ranges.

**Detection signature**:

```bash
# Search workspace for quant-test CPU reference code that may be missing clamp
grep -nE "\.to\(torch\.int8|\.to\(torch\.uint8|\.to\(torch\.int4" workspace/*/run_a3_reference.py workspace/*/edge_runner.py | \
  grep -v "clamp"
# Each hit = candidate for adding clamp before .to(...)
```

**Anti-patterns**:
- Limit `quantScale` to avoid overflow — masks the real test surface; NPU handles overflow correctly, test should too
- Use `torch.clamp` AFTER `.to(int)` — UB triggered before clamp, garbage values

**Evidence**:
- PR 103 PRECISION_STANDARDS.md codifies as MANDATORY for quant op test design
- Our `group_norm_silu_quant` Pass B 5/7 T2 result (instead of T1) — possible clamp-mismatch contribution; verification.json pre-dates this rule

**Other instances (predicted)**: every quant / dynamic-quant / quantize-with-scale op in cohort 2; every fp8/mxfp8/mxfp4 kernel where output is the narrow type.

**Cross-reference**:
- OL-144 (narrow-float range table — supplies clamp bounds for FP8/HiFloat8)
- OL-146 (CastTrait `SatMode::SAT` for quant — hardware-side clamp; this P-P is the reference-side equivalent)
- P-P94 (MERE/MARE thresholds — clamp affects which test cases hit the metric ceiling)

---

## P-P94: MERE/MARE aux precision standard (Mean/Max Relative Error per dtype Threshold) — ecosystem-blessed metric

`applies_to: soc=Ascend950PR,Ascend910_V220; cann=9.0.0; bisheng=all; op_class=all; phase=precision-verify`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 references/precision-testing/OPS_PRECISION_STANDARDS.md (生态算子开源精度标准)`

**Pattern**: When verifying an A5 kernel's precision against a CPU reference, compute and report **MERE** (Mean Relative Error) and **MARE** (Max Relative Error) as **aux metrics** alongside our existing T1/T2 tier classification. MERE/MARE is the official Ascend ecosystem standard for open-source-grade compute ops — having both metrics enables cross-comparison and lets us calibrate our T1/T2 tiers against the wider ecosystem.

**Formulas**:

```
MERE = avg( abs(actual - golden) / (abs(golden) + 1e-7) )
MARE = max( abs(actual - golden) / (abs(golden) + 1e-7) )
```

The `+1e-7` avoids div-by-zero when `golden` has zero / near-zero elements.

**PASS criterion**: `MERE < Threshold AND MARE < 10 × Threshold`.

**Threshold table by dtype**:

| dtype | Threshold | MERE limit | MARE limit (10×) |
|---|---|---|---|
| float16 | 2⁻¹⁰ ≈ 9.77e-4 | 9.77e-4 | 9.77e-3 |
| bfloat16 | 2⁻⁷ ≈ 7.81e-3 | 7.81e-3 | 7.81e-2 |
| float32 | 2⁻¹³ ≈ 1.22e-4 | 1.22e-4 | 1.22e-3 |
| HiFloat32 | 2⁻¹¹ ≈ 4.88e-4 | 4.88e-4 | 4.88e-3 |
| FP8 E4M3 | 2⁻³ = 0.125 | 0.125 | 1.25 |
| FP8 E5M2 | 2⁻² = 0.25 | 0.25 | 2.5 |
| INT8 | 0 (exact) | exact | exact |

**Relationship to our existing T1/T2 tiers** (from `ASCEND_OP_PRECISION_STANDARD_v2.1.md`):

| Tier | Our definition | MERE/MARE rough analog |
|---|---|---|
| T1 (bit-exact) | `max_abs_err == 0.0` | MARE strictly = 0 |
| T2 (compute-grade tolerance) | `atol=1e-3, rtol=1e-3` | Roughly MERE < 1e-3 for fp32 (≈ Threshold of `1.22e-4` × ~10) |

T1/T2 is **stricter** than MERE/MARE for most dtypes (we measure absolute error against an atol+rtol envelope; they measure relative error against a div-by-near-zero-protected denominator). MERE/MARE is **more permissive on small-value cases** (the +1e-7 epsilon prevents division blow-up).

**Why add this as AUX (not replacement)**:
- T1/T2 keeps producing the binary verdict (PASS / FAIL) for finalize decisions — keep it.
- MERE/MARE numerical values get exported alongside, enabling:
  - **Cross-comparison with ecosystem (Ascend Modelzoo, ops-nn upstream)** — they report MERE/MARE
  - **Calibration**: if T1/T2 marks PARTIAL but MERE/MARE shows clean PASS → indicates our tolerance is over-strict
  - **Detection of clamp-mismatch (per P-P93)** — high MARE concentrated on out-of-range cases is a clamp signature

**Implementation** (proposed `verification.json` schema addition):

```json
{
  "precision": {
    "status": "PASS",                  // existing T1/T2 verdict
    "tier": "T1",
    "max_abs_err": 0.0,
    "aux_metrics": {                   // NEW — MERE/MARE block
      "standard": "ecosystem_MERE_MARE_v1",
      "per_case": [
        {
          "case_id": 1,
          "dtype": "fp32",
          "MERE": 1.4e-5,
          "MARE": 8.2e-5,
          "threshold": 1.22e-4,
          "MERE_pass": true,
          "MARE_pass": true
        }
      ],
      "summary": {
        "n_pass": 8,
        "n_total": 8,
        "median_MERE": 1.4e-5,
        "p99_MARE": 8.2e-5
      }
    }
  }
}
```

**Helper code** (Python, to land in `aog-a3-author` Path A template):

```python
import torch

def compute_mere_mare(actual: torch.Tensor, golden: torch.Tensor) -> tuple[float, float]:
    """MERE/MARE per Ascend ecosystem precision standard."""
    rel = (actual.float() - golden.float()).abs() / (golden.float().abs() + 1e-7)
    return float(rel.mean().item()), float(rel.max().item())

DTYPE_THRESHOLD = {
    "torch.float32": 2 ** -13,
    "torch.float16": 2 ** -10,
    "torch.bfloat16": 2 ** -7,
    # narrow floats:
    "torch.float8_e4m3fn": 2 ** -3,
    "torch.float8_e5m2": 2 ** -2,
}

def passes_mere_mare(actual, golden, dtype_str: str) -> dict:
    threshold = DTYPE_THRESHOLD.get(dtype_str, None)
    if threshold is None:
        return {"verdict": "SKIP_NO_THRESHOLD"}
    mere, mare = compute_mere_mare(actual, golden)
    return {
        "MERE": mere, "MARE": mare,
        "threshold": threshold,
        "MERE_pass": mere < threshold,
        "MARE_pass": mare < 10 * threshold,
        "verdict": "PASS" if (mere < threshold and mare < 10 * threshold) else "FAIL",
    }
```

**Evidence**:
- PR 103 OPS_PRECISION_STANDARDS.md codifies as the official ecosystem standard for compute-op precision
- We have ASCEND_OP_PRECISION_STANDARD_v2.1.md (vendor v2.1 with MARE/MERE/RMSE) — MERE/MARE here is a subset of v2.1, formulas match

**Other instances (predicted)**: every op verification — adding aux fields is no-cost. Especially valuable for narrow-float ops where T1 (bit-exact) is unrealistic but MARE < `10 × 2⁻³` is meaningful.

**Cross-reference**:
- P-P93 (clamp rule — clamp mismatches show as high MARE concentration)
- `ASCEND_OP_PRECISION_STANDARD_v2.1.md` (vendor v2.1 — superset of MERE/MARE)
- Workflow Batch 4: `verification.json` schema gets `aux_metrics` field with MERE/MARE per-case

## P-P95: `LocalMemBar<MemType::UB>` replaces `SetFlag<MTE2_V>`+`WaitFlag` in A5 L2 MicroAPI [V351, microapi-sync]

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=l2-microapi; phase=kernel-author`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 l2-register-based-guide.md §250`

**Pattern**: In A5 L2 (Register-based MicroAPI) kernels, replace the A3 explicit pipe-event sync pair `SetFlag<HardEvent::MTE2_V>(EVENT_ID) + WaitFlag<HardEvent::MTE2_V>(EVENT_ID)` with the simpler `LocalMemBar<MemType::UB>()` memory barrier. The new barrier is type-targeted (UB), no event-id management, less error-prone.

**Concrete**:

```cpp
// ❌ A3 style — explicit pipe-event sync, error-prone event-id juggling
DataCopy(srcLocal, srcGm, count);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
Mul(dstLocal, src1Local, src2Local, count);

// ✅ A5 L2 style — type-targeted barrier
DataCopy<LoadDist::DIST_UNPACK_B16>(reg0, src_addr);
LocalMemBar<MemType::UB>();         // wait for UB-affecting prior ops
MicroAPI::Mul(reg1, reg0, reg0, maskReg);
```

**Where**: L2 inner loops between data-load and compute, between compute and store. Inside `__VEC_SCOPE__` blocks, the compiler tracks dependencies per register so most intra-scope sync is elided; explicit `LocalMemBar` is needed at scope boundaries.

**`MemType` options**:
- `MemType::UB` — barrier against all UB-affecting prior operations
- (additional types in CANN headers — `L1`, `L0A`/`L0B`/`L0C` for cube-side L2 ports)

**When NOT to use**:
- L1 mechanical ports (Memory-based) — keep `SetFlag`/`WaitFlag` for compatibility with existing A3 sync chains
- L3 SIMT kernels — they live outside the UB-pipe model, sync is implicit per-thread

**Detection signature**:

```bash
# In arch35/ kernels (L2), find lingering A3-style sync pairs
grep -nE "SetFlag<HardEvent::|WaitFlag<HardEvent::" arch35/*.h
# Each pair is a substitution candidate.
```

**Why simpler is correct**:
- A3 `SetFlag`/`WaitFlag` requires manually picking EVENT_ID (0-7), threading it through compute, freeing it later. Per-instance bugs accumulate.
- A5 `LocalMemBar<MemType::UB>` has no event-id parameter — compiler tracks dependencies. Less code, fewer bugs.

**Evidence**: PR 103 l2-guide §250 row "内存屏障" entry.

**Other instances (predicted)**: every L2-classified op in cohort 2.

**Cross-reference**:
- OL-152 (Memory↔Register API mapping — this is one row)
- P-P96 (`__VEC_SCOPE__` inner loop — uses `LocalMemBar` at boundaries)

---

## P-P96: `__VEC_SCOPE__ { RegTensor / MaskReg / MicroAPI::* }` — canonical A5 L2 inner-loop shape

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=l2-microapi; phase=kernel-author`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 l2-register-based-guide.md §114-155`

**Pattern**: A5 L2 (Register-based) compute MUST be wrapped in `__VEC_SCOPE__ { ... }`. Inside the scope, declare `RegTensor<T>` registers + `MaskReg` masks, use `MicroAPI::Op(...)` calls. The scope marker tells the compiler "lower to register-resident codegen", enabling the perf win that motivates L2.

**Canonical shape**:

```cpp
__aicore__ inline void ComputeL2(__ubuf__ T_KV* x_ub, __ubuf__ T_KV* gamma_ub,
                                  __local_mem__ float* dst_ub,
                                  uint16_t loopTimes, uint32_t stride,
                                  uint16_t count, float reciprocal, float epsilon)
{
    __VEC_SCOPE__
    {
        // 1. Declare all RegTensors at scope head (up to ~8 typical without spill)
        RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;

        // 2. Build masks once (per-mask helpers)
        MaskReg pMask = UpdateMask<float>(count);                  // partial — first `count` lanes
        MaskReg pFull = CreateMask<float, MaskPattern::ALL>();      // full — all lanes active

        // 3. Inner loop — entire compute in register space
        for (uint16_t i = 0; i < loopTimes; ++i) {
            LoadTensorForDtypeT<T_KV>(x_ub, reg0, pMask, i * stride);
            LoadTensorForDtypeT<T_KV>(gamma_ub, reg1, pMask, 0);
            MicroAPI::Mul(reg2, reg0, reg0, pMask);          // x * x
            ReduceSum(reg2, reg2, pMask);                      // ΣΣ
            MicroAPI::Muls(reg3, reg2, reciprocal, pFull);     // / N
            MicroAPI::Adds(reg4, reg3, epsilon, pFull);        // + eps
            MicroAPI::Sqrt(reg5, reg4, pFull);
            MicroAPI::Div(reg6, reg0, reg5, pMask);
            MicroAPI::Mul(reg7, reg1, reg6, pMask);
            StoreTensorForDtypeTOut<float>(dst_ub, reg7, pMask, i * stride);
        }
    }   // ← __VEC_SCOPE__ end. Registers freed.
}
```

**Five mandatory ingredients**:

1. `__VEC_SCOPE__` — outer brace block enables register codegen
2. `RegTensor<T>` — single vector register typed for `T`
3. `MaskReg` — predicate register controlling per-lane execution
4. `MicroAPI::Op(...)` prefix — namespace-qualified vector ops
5. **`LoadTensorForDtypeT` + `StoreTensorForDtypeTOut`** helpers — handle automatic Cast (half / bf16 → fp32 on load; fp32 → narrow on store)

**Discipline rules**:

- **Declare all RegTensors at scope head** — compiler does register allocation per scope; mid-scope declarations create lifetime bugs
- **Reuse registers across the loop body** — `reg2, reg3, ..., reg7` get reused per iteration (the compiler allocates them to the same physical registers across iterations). DON'T declare `reg8, reg9, ..., regN` for each loop iter
- **Keep scope tight** — only the hot inner loop in `__VEC_SCOPE__`. UB allocation, tiling decode, GM→UB copy stay outside the scope
- **No `SetFlag`/`WaitFlag` inside scope** — use `LocalMemBar<MemType::UB>()` at boundaries (see P-P95)
- **No `LocalTensor` declarations inside scope** — only `RegTensor` lives here; `LocalTensor` (UB tensor) is for outside-scope UB tracking

**Anti-patterns**:

```cpp
// ❌ RegTensor declared mid-loop — confuses register allocator
__VEC_SCOPE__ {
    for (uint16_t i = 0; i < loopTimes; ++i) {
        RegTensor<float> reg;     // ← wrong place
        ...
    }
}

// ❌ A3 LocalTensor inside __VEC_SCOPE__
__VEC_SCOPE__ {
    LocalTensor<float> tmp;       // ← wrong type
    ...
}

// ❌ Whole-kernel __VEC_SCOPE__ wrapping unrelated UB management
__VEC_SCOPE__ {
    DataCopy(srcLocal, srcGm, count);   // ← outside-scope work in scope
    ...vector compute...
    DataCopy(dstGm, dstLocal, count);   // ← outside-scope work in scope
}
```

**Detection signature**:

```bash
# In L2 arch35/ kernels, verify __VEC_SCOPE__ is present in compute paths
grep -nc "__VEC_SCOPE__" arch35/*.h
# Compute path counted >= 1 per kernel function with MicroAPI:: calls

# Conversely, find MicroAPI:: calls NOT wrapped in __VEC_SCOPE__:
awk '/__VEC_SCOPE__/{in_scope=1} /^}/{in_scope=0}
     /MicroAPI::/&&!in_scope{print FILENAME":"NR": MicroAPI outside __VEC_SCOPE__: "$0}' arch35/*.h
```

**Evidence**: PR 103 l2-guide §114-155 codifies as L2 canonical pattern (RMSNorm + Mul example).

**Other instances (predicted)**: every L2 norm / activation / quant kernel in cohort 2.

**Cross-reference**:
- OL-152 (Memory↔Register API map — this pattern uses every right-column entry)
- OL-146 (CastTrait — used in inline Cast calls within the scope)
- OL-148 (SPR overflow toggle — often paired with the scope for bounded-output ops)
- P-P95 (`LocalMemBar` — boundary sync)

---

## P-P97: Mask-free additive piecewise decomposition — replace MicroAPI Select-based activation forms with `Maxs/Mins` branch gates so each domain reduces to 0 outside its active range, then sum

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=elementwise-activation, port_a3_to_a5`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A3 family — pattern is dtype-arithmetic and should transfer, but no cross-arch witness yet)`

**Principle**: Activation functions defined as `out = mask ? f_pos(x) : f_neg(x)` (one branch per sign of `x`, or per side of a threshold `c`) can be rewritten without any mask register as `out = f_pos(Maxs(x, c)) + f_neg(Mins(x, c))` IF AND ONLY IF each branch evaluates to `0` outside its active range — i.e., `f_pos(c) = 0` AND `f_neg(c) = 0`. The additive form uses only universal SIMD primitives (`Maxs`, `Mins`, `Add`, `Adds`, `Muls`, plus whatever transcendental each branch needs), avoiding the MicroAPI Select / `MaskReg` machinery. Bit-equivalence to the masked form follows from `f_pos(Maxs(x,c)) ≡ 0` whenever `x ≤ c` and `f_neg(Mins(x,c)) ≡ 0` whenever `x ≥ c`, so the sum collapses to exactly one branch at any input.

**When this applies**:
- Activation has a single threshold `c` (typically 0) separating two analytic branches
- One branch reduces to 0 at the threshold (and saturation: `f_pos(x ≤ c) = 0` or `f_neg(x ≥ c) = 0` after the clamp)
- Both branches are expressible in SIMD primitives present in `API_CATALOG.md` (so neither requires the Select / Mask form)

**Canonical anchor (ELU, op#elu kw-1 2026-05-17)**:

```cpp
// MicroAPI Select form (what upstream arch35/elu.h uses inside ElementwiseSch<EluDag>):
//   out = (x >= 0) ? scale * x
//                  : scale * alpha * (exp(input_scale * x) - 1)
//
// Mask-free additive equivalent (bare AscendC SIMD):
LocalTensor<float> pos = ...;   // clamps to x for x >= 0, else 0
LocalTensor<float> neg = ...;   // clamps to x for x <= 0, else 0
LocalTensor<float> exp_in = ...;

Maxs(pos, x, 0.0f, count);                              // pos = max(x, 0)
Mins(neg, x, 0.0f, count);                              // neg = min(x, 0)
Muls(exp_in, neg, input_scale, count);                  // input_scale * min(x, 0)
Exp(exp_in, exp_in, count);                             // exp(...)
Adds(exp_in, exp_in, -1.0f, count);                     // ... - 1
Muls(exp_in, exp_in, alpha, count);                     // alpha * (...)
Add(out, pos, exp_in, count);                           // pos + neg-branch
Muls(out, out, scale, count);                           // outer scale
```

**Bit-equivalence proof sketch**: `pos = Maxs(x, 0)` is `x` for `x ≥ 0` and `0` otherwise. `Mins(x, 0)` is `x` for `x ≤ 0` and `0` otherwise. The neg-branch chain has `exp(0) - 1 = 0`, so for `x ≥ 0` the neg-branch contribution is exactly `0` and `out = scale * pos = scale * x`. For `x < 0` the pos contribution is `0` and `out = scale * alpha * (exp(input_scale * x) - 1)`. `Add(x, 0) = x` is exact in fp32 (no precision loss). Confirmed bit-equivalent for all 8 elu cases on Ascend950PR (3 bit-exact + 5 within T2 ULP tolerance vs CPU truth) with the same Iron-law §5 literal-first ordering as the masked reference.

**Why use this rewrite**:
- The verify-artifact path (OL-164) requires bare AscendC primitives bound via `extern "C" __global__ __aicore__` — MicroAPI Select needs the L2 `__VEC_SCOPE__` + `MaskReg` machinery (P-P96), which is heavier to wire and ties the verify kernel to a specific bisheng codegen path
- The additive form composes with `LAUNCH_BOUND` / TQue depth-4 (OL-63) without extra register-pressure analysis
- The five SIMD primitives used (`Maxs`, `Mins`, `Muls`, `Adds`, `Exp`, `Add`) all live in `API_CATALOG.md` — no missing-primitive risk, no version pinning

**Anti-pattern (don't apply when)**:
- Branches don't reduce to 0 at the threshold — e.g. `f_pos(x) = a*x + b1`, `f_neg(x) = c*x + b2` where `b1 ≠ 0 ≠ b2`: the additive form double-counts the offsets. Either pre-subtract the offsets (so the rewritten branches DO reduce to 0) or keep the masked form.
- The activation requires more than two branches (e.g., a 3-piece function like ReLU6 with clamp ceiling). Recursive application is possible but the `Mins/Maxs` chain grows; consider a `Maxs(Mins(x, hi), lo)` clamp + single-branch instead.
- One branch is transcendental and the other diverges at the threshold (e.g., `1/x` for `x > 0`, `0` for `x ≤ 0`): the `Mins(x, ε)` clamp shifts the divergence to a fixed point but the precision behavior near the seam needs separate verification.

**Other instances (predicted)**:
- ReLU: `out = Maxs(x, 0)` — already this form, trivially
- LeakyReLU: `out = Maxs(x, 0) + alpha * Mins(x, 0)` — direct application
- Softplus: branchless via `log(1 + exp(-|x|)) + Maxs(x, 0)` (different rewrite — Maxs handles the linear tail, log1pexp handles the curved tail)
- SELU: `out = scale * (Maxs(x, 0) + alpha * (exp(Mins(x, 0)) - 1))` — same pattern as ELU with `input_scale=1`
- HardSwish below threshold: piecewise-clamp branches that reduce to 0 outside active range
- Any future elewise activation whose upstream `op_kernel/arch35/<op>.h` uses `ElementwiseSch<<Op>Dag>` and whose `<Op>Dag::Compute` uses a `Select`/`Mask`-based branch — applying this rewrite gives the verify-artifact a SIMD-only implementation

**Cross-reference**:
- OL-164 — the dual-output rule that motivates this rewrite (verify-artifact requires bare SIMD primitives, not MicroAPI Select)
- OL-63 — TQue depth=4 for elementwise (composed alongside this pattern in the verify kernel template)
- OL-81 — CAST_RINT for narrow-dtype cast-back at chain end (composed for half/bf16 paths)
- OL-82 / Iron law §5 — literal-first VEC ordering (preserved across the rewrite; no fusion, no strength reduction)
- P-P96 — the L2 `__VEC_SCOPE__` MicroAPI form (the form being avoided in the verify-artifact)
- `output/npukernelbench/src/kernels/1_GELU/` — companion verify-artifact template that this pattern slots into

## P-P98: Non-aligned tail write to GM via `DataCopyPad` + `DataCopyExtParams` byte-level `blockLen` — replaces host-side pad+narrow cheat [V351+V220, data-movement, anti-cheat]

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5; op_class=elementwise+quant+strided-write`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5 (2026-05-18 task #22 hardware probe — workspace/probe_datacopypad_v300_tail/PROBE_REPORT.md: blockLen ∈ {31,33,47,63} all wrote exactly N bytes, no overflow, no runtime error)`

**EC-23 scope note**: EC-23 documents V220 `DataCopyPad` UB→GM crash; this is **V220-only**. On V351, the primitive works cleanly per the hardware probe above. P-P98 applies on V351; foreach-class ops with non-aligned tails MUST use DataCopyPad in kernel rather than host-side pad+narrow workaround in pybind.

**Trigger**: kernel writes a row / tail / variable-length output whose byte length is NOT a multiple of 32 (e.g., int8 outputs with row width `H` bytes, fp16 outputs with column width not 16-multiple, variable-length scatter tails). Naive `DataCopy(gm, ub, count)` silently truncates to nearest 32B; pre-allocating padded output + post-trim in pybind is the cheat we want to retire (OL-167).

**Technique**: use `DataCopyPad(gm, ub, DataCopyExtParams)` for the non-aligned write. `DataCopyExtParams.blockLen` is BYTES (not 32B-blocks), supports `1..2097151`, no alignment constraint on the write size. UB source still needs 32B-aligned start; framework reads the aligned-up source block, writes exactly `blockLen` bytes to GM (downstream bytes untouched).

```cpp
// 3-line anchor — UB→GM non-aligned write:
DataCopyExtParams cp{
    /*blockCount=*/ static_cast<uint16_t>(numRows),
    /*blockLen=*/   static_cast<uint32_t>(rowBytes),        // bytes, can be ANY 1..2097151
    /*srcStride=*/  static_cast<uint32_t>(ubStrideBytes),   // GM-side bytes, UB-side 32B-blocks
    /*dstStride=*/  static_cast<uint32_t>(gmStrideBytes),
    /*rsv=*/        0,
};
DataCopyPad(dstGm, srcUb, cp);   // GM receives exactly blockCount × blockLen bytes
```

**Symmetric form for GM→UB non-aligned reads (when relevant)**: requires `DataCopyPadExtParams<T>` to specify padding (right-pad to 32B with `padValue` or random):

```cpp
DataCopyExtParams cp{1, /*byteLen=*/47, 0, 0, 0};
DataCopyPadExtParams<half> pad{/*isPad=*/true, /*leftPad=*/0, /*rightPad=*/8, half(0)};   // 47B + 8×2 = 63B ≥ 32B aligned
DataCopyPad(dstUb, srcGm, cp, pad);
```

**Anti-pattern (don't apply when)**:
- `count * sizeof(T) % 32 == 0`: just use plain `DataCopy(gm, ub, count)` — DataCopyPad path has slightly heavier setup, no benefit when alignment is naturally met.
- UB→UB transfer: `DataCopyPad` on UB→UB routes through GM internally (VECIN→GM→TSCM with ND→NZ conversion) — much slower than direct `DataCopy(ub, ub, count)`. Use plain `DataCopy` for UB↔UB.
- Aligned wide stride with 32B-block granularity: plain `DataCopyParams` with 32B-block `blockLen` is more compact when the math naturally lands on block boundaries (e.g., matrix row-write with row width = 32B multiple).
- "Tail is just a few extra bytes, I'll pad in pybind": NO — that's the cheat this pattern exists to replace (OL-167).

**Other instances (predicted)**:
- Quantization output writers (int8 / int4 outputs with arbitrary H).
- Variable-length scatter/gather tails (NNZ-driven write extent).
- Sparse op outputs where NNZ is data-dependent and not 32-aligned.
- Cross-row strided writes where row width is dtype-dependent (fp16 with odd column count).
- Any kernel currently relying on `align_up64(rowBytes, 32)` over-allocation in pybind followed by `narrow + contiguous` — should migrate to DataCopyPad in kernel.

**Evidence**:
- Direct API reference: `~/workspace/a5/tasks/datacopy_api.md` §2.3 (silent truncation behavior) + §2.4 (DataCopyPad usage) + §3.3 (UB→GM examples) + §6.1 (alignment trap).
- Migration target archives (currently using the cheat — to be reworked when their op-gen is re-run with this pattern in worker brief): `11_DequantSwigluQuant_v3.2_cold` (pybind11.cpp:213-259), elu (a3_to_a5_port pybind11.cpp:52,69,100), clipped_swiglu (a3_to_a5_port pybind11.cpp:108).
- P149 finalize gate logs (`finalize_pipeline.py:GateID.PYBIND_HOST_BUSINESS_LOGIC`): catches `narrow(..., 0, ...).contiguous()` post-kernel in pybind; this pattern is the kernel-side fix that prevents P149 retraction.

**Cross-reference**:
- OL-167 — the principle (DataCopy silent truncation + anti-cheat policy)
- EC-23 — DataCopyPad UB→GM V220 crash mitigation (orthogonal: this is V220 build-side gotcha)
- `vendor/ascendc-kernelgen-data/npu_benchmark/level2/8_QuantScatter.py` — typical op shape where this pattern applies (int8 output, non-32B row widths)

## P-P101: De-scalarize flash online-softmax — hand-rolled mem-based VEC online-softmax + precision-safety triad (replaces `SoftmaxFlashV2` scalar pole)

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5; op_class=attention-softmax (FA forward/backward, masked-LM, any flash online-softmax)`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend950PR; cann=9.0.0 b226; bisheng=15.0.5 (FA-A5 path-B 2026-05-31, in-scope 5/5 precision, commits 5a3a1cee v2 / 697ed8c8 v5)`
`unverified_on: soc=Ascend910_V220 (arch22 — the technique is arch-independent in principle: standard LocalTensor vec ops run on V220 too; A3 precision/e2e NOT yet measured — independent prototype A3 FA cross-pollination pending, feed WITH the kernel≠wall caveat)`

**Trigger**: an attention/FA softmax stage that is scalar-bound — the `AscendC::SoftmaxFlashV2` library call (its internal max/sum reduce + `[rows,8]` stat-packing / `DeinterleaveStat8`) OR per-row `GetValue`/`RowMuls`/`RowDivs`. msprof signature: `aiv_scalar_ratio` is the dominant pipe (A5 FA: 0.27–0.31 inside SoftmaxFlashV2; A3 FA independent prototype: 0.503), the cube starved behind the vec long-pole (Amdahl). Cross-ref OL-200 (the cube/vec pipeline overlap is the complementary lever; this shortens the vec stage so the overlap pays off).

**Technique**: replace the scalar softmax with a hand-written online flash softmax using standard AscendC `LocalTensor` vector ops (NOT MicroAPI register-compute `__simd_vf__`/`RegTensor` — that path is a separate, in-FA-context runtime-UNVERIFIED concern, see CAND-FA-MICROAPI-REG-507015). Per KV-tile, m-loop over rows:
1. **row-max**: `WholeReduceMax` (replaces the SoftmaxFlashV2 internal scalar).
2. **exp(x − rowmax)**: subtract rowmax (broadcast) THEN `Exp` — never bare `Exp(x)` then divide.
3. **row-sum**: `WholeReduceSum`.
4. **online combine**: `newMax=max(runMax,tileMax)`; `corrPrev=exp(runMax-newMax)`; `corrCur=exp(tileMax-newMax)`; `newSum=runSum*corrPrev + tileSum*corrCur` (standard `Max`/`Muls`/`Add`).
5. **O rescale**: `O = O*corrPrev + Otile*corrCur` via `Muls` with the per-row corr **broadcast** across D cols.
6. **final normalize**: `O /= runSum` via vector `Div` (or `Reciprocal`+`Muls`) with runSum **broadcast** — NOT scalar per-row `RowDivs`/`GetValue`.

**Precision-safety triad (mandatory — THE inf-bug fix)**:
1. **masked positions → `minValue` (large negative FINITE, NOT −inf)** → `exp≈0` AND the exp-sum is never 0 (even a fully-masked row sums to `row_len·exp(0)`). #1 inf防线.
2. **subtract rowmax BEFORE exp** (`ExpSub` fused, or `Sub` then `Exp`) — no overflow. Never bare `Exp` then divide.
3. **normalize via vector `Div`/`Reciprocal`+`Muls`** (sum>0 guaranteed by #1) — NOT scalar `RowDivs`.

**Pitfall (cost an iter — stat-buffer aliasing)**: a V-pipe `Brcb(softmaxSumUb_)` for the broadcast-normalize that ALIASES the MTE3 `DataCopy(smSumGm_, softmaxSumUb_)` sm-emit on the SAME buffer → sm_sum corruption + softmax_out inf. Fix: a DISTINCT spare buffer for the Brcb (copy the stat first), OR a proper `SetFlag`/`WaitFlag` barrier between the MTE3 sm-emit and the V-pipe Brcb. (Same class as the tileMax/tileSum aliasing fix.)

**Perf — SCOPED (kernel≠wall; do NOT write a bare "−24%")**: removing the SoftmaxFlashV2 scalar pole gave **kernel-msprof task-duration −24%** (sum 349→264us) + `aiv_scalar` materially reduced — a **kernel-time** result. **e2e WALL was NEUTRAL** in the benchmark (independent same-card A/B, author≠measurer): host BNSD-fold + pybind/wrapper overhead dominate the wall, kernel is only ~20–40% of it, so the kernel win does NOT transmit to the benchmark e2e wall. Whether it reaches a given op's customer e2e depends on that op's bottleneck profile — MEASURE the wall decomposition (host / launch / kernel), do NOT assume. See OL-201 (the pybind-wrapper-wall-not-vendor-fair measurement caveat).

**Anti-pattern**: (a) bare `Exp(x)` then divide (ITER-9 inf path — missing the max-subtract); (b) −inf masking (exp-sum can hit 0 → inf on normalize); (c) scalar `RowDivs`/per-row `GetValue` normalize (re-introduces the scalar pole this pattern removes); (d) Brcb aliasing the sm-emit buffer (the pitfall above).

**Other instances (predicted)**: any scalar-softmax-bound attention op — FA forward/backward, GQA/MQA, masked-LM softmax, flash-decoding; more broadly any per-row reduce-then-normalize stage (the de-scalarize-via-WholeReduce + broadcast-normalize shape) where msprof shows `aiv_scalar` dominant. Detailed concrete params (tile shapes, exact reduce/broadcast op signatures, stat-buffer layout) live in `fa_class/cv_reference_concrete_params.md`.

**Cross-ref**: OL-200 (cube/vec pipeline overlap — the complementary perf lever; softmax de-scalarize shortens the vec stage so the overlap is not Amdahl-capped); OL-201 (kernel≠wall measurement caveat that scopes the −24%); OL-54 (the MicroAPI register path — explicitly NOT used here); CAND-FA-MICROAPI-REG-507015 (the register-reduction route that crashed; this mem-based path is the route-around); P-P62 (Row-Scalar VEC Multiply via Brcb — the broadcast-multiply primitive used in steps 5–6).

## P-P105: Boundary-clamped interpolation — a single OOB→0 gather reproduces zeros/border/reflection padding because the clamped coordinate's overflow neighbor always carries zero interpolation weight

`applies_to: soc=Ascend950PR_957b; cann=9.1.T500; bisheng=n/a; op_class=sampling-interpolation (grid_sample, interpolate, affine-warp)`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A3 family — this is dtype-agnostic coordinate arithmetic and should transfer, but no cross-arch witness yet)`

**Principle**: A linear-interpolation sampler (bilinear / trilinear) supporting multiple padding modes does NOT need per-mode neighbor-index clamping if (1) the coordinate is mapped into the valid range `[0, size-1]` by a per-mode clamp BEFORE neighbor selection (border → `clamp`, reflection → `reflect`, zeros → identity), and (2) the per-pixel fetch helper returns `0` for any out-of-bounds index. After the clamp, the floor neighbor `x0 = floor(coord)` is always in-bounds; the overflow neighbor `x0 + 1` only lands OOB (at exactly `size`) when `coord` sits on the integer upper boundary `size - 1`, where the fractional weight `dx = coord - x0 == 0`. Since that neighbor's contribution is `dx * fetch(x0+1) = 0` regardless of what `fetch` returns, a unified `fetch` that yields `0` for OOB is correct for all three modes. The zeros mode then "falls out" for free: for zeros, the coord is NOT clamped, so any OOB neighbor genuinely contributes `0` — exactly what the unified fetch produces.

**Concrete anchor (grid_sample, kw-1 2026-06-20)**:

```cpp
// per-mode coordinate clamp into [0, size-1] (zeros = identity, border = clamp, reflection = reflect)
float cx = gs_clip(coord_x, W, paddingMode, alignCorners);
int64_t x0 = gs_floor_i(cx);
float dx = cx - (float)x0;                 // dx == 0 exactly when cx is on the integer boundary
// unified fetch returns 0 for OOB (mirrors upstream arch35 GetInputPointValue):
float v00 = gs_fetch(img, base_nc, y0, x0,     H, W);   // x0 always in-bounds after clip
float v01 = gs_fetch(img, base_nc, y0, x0 + 1, H, W);   // OOB only when dx==0 → weight kills it
float top = v00 * (1.0f - dx) + v01 * dx;               // border/reflection/zeros all correct
```

**Why it's safe (one fetch, no per-mode branch)**: the only way `x0 + 1 == size` (OOB) after `gs_clip` is `cx == size - 1` (integer boundary), which forces `dx == 0`; `weight * fetch == 0` whatever `fetch` returns. This matches upstream arch35 `GetInputPointValue` semantics line-for-line — no per-mode neighbor clamp is authored.

**When this applies**:
- Linear/bilinear/trilinear sampling where the coordinate is clamped/reflected into range before neighbor pick
- Padding modes are {zeros, border, reflection} (the standard `F.grid_sample` / `align_corners` family)
- The interpolation weight on a neighbor goes to 0 exactly at the boundary where that neighbor would overflow

**Anti-pattern (don't apply when)**:
- Nearest-neighbor sampling (no fractional weight to zero out the OOB neighbor) — needs explicit clamp
- A padding mode whose OOB contribution is nonzero (e.g. a constant-fill ≠ 0, or wrap/circular padding) — the "weight kills it" argument fails
- Cubic interpolation (4-tap): the far taps can be OOB with nonzero weight — this 2-tap argument does not extend without per-tap analysis

**Evidence**: grid_sample port_a3_to_a5 kw-1 (2026-06-20, A5/Ascend950PR_957b/CANN 9.1.T500): one `gs_fetch` (OOB→0) + one `gs_clip` covers zeros/border/reflection × align_corners {false,true}; precision 29/29 T1 PASS (fp16 23/23, fp32 3/3, bf16 3/3), 29/29 deterministic. Verified against `F.grid_sample` fp32 CPU truth cross-checked line-by-line vs arch35 `GetInputPointValue`.

**Other instances (predicted)**: `interpolate`/`upsample` bilinear with boundary handling, affine-grid warp samplers, any 2-tap-per-axis sampler with a pre-clamp + zero-fill OOB fetch.

**Cross-reference**: OL-150 (SIMT programming model — the per-thread gather this sampler runs as), EC-74 (`__simt_callee__` for the VF-called fetch/clip helpers), A.2.6 dual-input faithful-reference rule (model.py reproduces the arch35 unnormalize/clip sequence).

## P-P106: Serial-in-L linear recurrence on AscendC — L-chunk to bound UB + Hillis-Steele parallel associative scan over the affine pair, with an L-major `[l*N+n]` layout so one L-shift is one wide Mul/Add across all N lanes

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=AIV; op_class=scan/recurrence (SSM/Mamba selective-scan, cumsum, cumprod, prefix-scan, any associative linear recurrence)`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A3 family — the algorithm is arch-independent; the perf ratios and the 248KiB-UB chunk-size math are A5-specific, re-tune for A3 UB)`

**Principle**: A serial-in-`L` linear recurrence `x[l] = a[l]·x[l-1] + b[l]` (the SSM/Mamba selective-scan core; also prefix-products, cumulative ops) lowers to AscendC in two layers:

1. **L-CHUNK to bound UB**: split `L` into chunks of length ≤ `CH`. Carry the scan-state across chunk boundaries by folding the incoming prefix into position 0 of the chunk, and read the chunk's LAST element out as the next chunk's prefix. This makes arbitrarily large `L` (customer `L=5000`) fit a fixed UB budget.

2. **Hillis-Steele parallel scan WITHIN each chunk**: replace the O(L) serial per-`l` loop with the O(log₂ chunk) inclusive Hillis-Steele scan over the AFFINE PAIR `(a, b)`, using the associative combine `(a2,b2) ⊗ (a1,b1) = (a2·a1, a2·b1 + b2)`. log₂(CH) stride-passes instead of CH serial steps.

**Layout key (this is the actual perf lever)**: store the working buffers `[l*N + n]` **L-major** (L outer, N inner). Then an L-shift by `stride` is just an element-shift of `stride*N` — so each Hillis-Steele stride-pass is ONE wide `Mul` / `Add` across ALL `N` lanes at once, with NO per-N-lane scalar loop. For the customer regime (small `N=16`, large `L`) this is what converts a scalar-bound kernel into a vector-bound one (msprof `aiv_vec_ratio` 0.948, previously scalar-bound).

**Scan-variant caveat**: an INCLUSIVE Hillis-Steele scan needs NO power-of-two padding for a ragged last chunk — it processes any chunk length directly. The pow2-pad requirement is a **Brent-Kung** (work-efficient up-sweep/down-sweep) concern, not a Hillis-Steele one. Verify which variant you implemented before adding padding logic you don't need.

**Forward AND backward scans (direction is the only difference)**: the SAME associative Hillis-Steele machinery covers both the forward recurrence and the BACKWARD/adjoint recurrence that arises in the gradient (an SSM/selective-scan backward has a reverse recurrence `dx[l] = a[l+1]·dx[l+1] + s[l]`, gradient flows high→low in `L`). The forward scan is a PREFIX (inclusive) HS that combines the affine pair left-to-right (each stride-pass reads the element at `l-stride`); the backward adjoint scan is a SUFFIX (reverse) HS over the SAME affine combine `(a2,b2)⊗(a1,b1)=(a2·a1, a2·b1+b2)` but each stride-pass reads the element at `l+stride` and folds right-to-left. The `[l*N+n]` L-major layout still applies unchanged — a suffix shift by `stride` is the element at `+stride*N`, again ONE wide `Mul`/`Add` per pass across all N lanes. So a serial O(cl) reverse carry chain (per-`l` `GetValue` + V↔S round-trips) becomes O(log₂ cl) vector passes, exactly mirroring the forward win. This extends P-P106 from forward-only to forward+backward associative scans.

**HS micro-opt — drop the redundant shift-copy when the read precedes the write (5→4 vec ops/pass)**: the textbook Hillis-Steele combine defensively shift-copies BOTH operands `a` and `b` into clean scratch before the strided update, to avoid read-after-write aliasing — 5 vec ops per stride-pass (copy `a_shifted`, copy `b_shifted`, `prod = a·b_shifted`, `b += prod`, `a *= a_shifted`). But when the pass reads the to-be-shifted operand BEFORE the op that would clobber it, that one shift-copy is redundant. In the affine combine the `b`-update `b[l] += a[l]·b_neighbor[l]` reads the OLD `b` prefix that is NOT yet written this pass (the `Add` writing `b[l+off]` comes AFTER, and the product lands in a DISTINCT buffer so there is no aliasing) → read `b_neighbor` DIRECTLY and DROP its shift-copy: **5 ops/pass → 4**. The `a` shift-copy is still needed (`a *= a_shifted` clobbers `a` in place). Same reasoning applies to BOTH the prefix and the suffix scan. Precision-neutral by construction (byte-identical output — it removes a copy, not a compute). General rule: in ANY strided in-place associative combine, audit each operand for "is it read before the write that overwrites it?" — if yes the protective copy is dead code.

**Chunk-size vs layout tension**: the `[l*N+n]` L-major layout means UB usage scales with `CH*N`. selective_scan used `CH=256` (UB ~222KB of the 248KB A5 budget); `CHUNK=2048` is INFEASIBLE at this layout (would exceed UB). Pick `CH` from the UB budget and `N`, not arbitrarily.

**Scan algorithm-selection on A5 (decision rule) — minimize DEPENDENT-BARRIER DEPTH, not total work**: on A5 (Ascend950PR) the AIV vector pipe inserts a `PipeBarrier` per dependent vector step, so PER-STEP dependency LATENCY dominates a memory-bound scan — NOT the total vector op-count. Consequence: a work-EFFICIENT O(L) SERIAL scan (a 256-deep cross-step dependency chain for `CH=256`) is the WRONG choice on A5 even though it does the least work — it measured **2–5× SLOWER** than the work-INefficient O(log₂ L) Hillis-Steele (an 8-deep dependency chain at `CH=256`) on the same op, because HS's log-depth wins on barrier latency despite doing MORE total vector work. The standard ILP rescue (row-batching independent rows to fill the pipe) does NOT save a linear-DEPTH barrier chain — a batched-serial scan was still ~5× slower — AND the 248KiB UB caps how many rows you can batch before you hit the required chunk length (the same UB-vs-`CH` tension above). **Net selection rule for a serial-in-L recurrence on A5**: pick the scan variant with the SHALLOWEST dependent-barrier chain (Hillis-Steele log-depth) within the LocalTensor structure, even at the cost of more total vector work; do NOT pick the work-efficient serial scan, and do NOT expect row-batching to rescue a linear-depth chain. **Remaining work**: The only unexhausted lever is a LARGE rewrite — a work-efficient AND log-depth Brent-Kung scan via reg-stride `BinaryRepeatParams`, or a reg-base/SIMT rewrite escaping the per-op LocalTensor overhead — flagged as a research direction, NOT a quick win. **Generalizes to**: any memory-bound, latency-(not throughput-)bound serial recurrence on a per-step-`PipeBarrier` architecture — algorithm selection is governed by dependency-chain DEPTH, the same latency-vs-throughput discriminator OL-231 used to settle the bwd carry-chain floor.

**Evidence**: selective_scan_source_a5 fwd_simd ① (L-chunk, PR#23) + ② SIMT cooperative-scan (PR#24) + ④ parallel-scan (PR#28), all 2026-06-22 on A5/Ascend950PR_957b/CANN 9.1.T500. Rigorous same-NPU device-time A/B (serial-within vs parallel-within, BOTH L-chunked): L=5000 **fp32 1.71× / fp16,bf16 1.93×** faster; ratio grows with L (1.47×@L=256 → 1.93×@L=5000). msprof `aiv_vec_ratio` 0.948 (was scalar-bound). bf16 PASS; fp32/fp16 at the near-zero dtype floor (graded with the input-cast floor, OL-243); carry-exact (zero boundary spike); determinism N=5. Expert design (Tencent customer) confirmed the approach. Whitebox-derived (expert-design + on-device verify).
- **Backward/adjoint scan + HS micro-opt** — selective_scan_full_grad (bwd_simd, PR#37, merged main `bda9cb3c`, 2026-06-22, A5/Ascend950PR_957b/CANN 9.1.T500). The reverse-suffix HS (KO-bwd3) replaced the serial O(cl) reverse adjoint carry chain on PASS B; the 5→4 shift-copy drop (KO-bwd5) applied to both PASS A's forward HS and PASS B's reverse HS; fwd_simd got the same drop (KO-fwd4). Same-NPU back-to-back msprof device-time A/B, median 30-rep, tolerance NOT loosened: bwd **fp32 L=5000 22883→8514µs = 2.69×** (bf16 22887→8835µs); scalar-ratio 32.5%→10.5%. fwd_simd bf16 L=5000 2678→2490µs = 1.08×. **Precision-neutral**: bwd 30/30 truth-backed (fp32/fp16/bf16, graded with input-cast floor OL-243); fwd byte-identical to baseline at the dtype floor; determinism 5/5 (<0.2%). **Dtype-invariance (perf characteristic)**: the scan runs the affine recurrence in fp32 internally regardless of I/O dtype — so the bwd 2.69× / fwd 1.08× and the precision-neutrality hold across fp32/fp16/bf16 (the only dtype-dependent cost is the I/O cast, not the scan). Whitebox-derived (msprof ablation: reverse serial carry ~35% of bwd device-time; HS scan ~45% of fwd bf16).

- **Barrier-depth scan selection + utilization-ceiling (A5 memory-bound investigation)** — selective_scan fwd memory-bound study (2026-06-23, A5/Ascend950PR_957b/CANN 9.1.T500, whitebox). msprof on the HS @ `CH=256` fwd at L=5000: **vec 95.8% / mte2 9.1%** = vec-bound on a memory-bound algorithm; element-work-at-peak ≈167µs ≈ the 116µs memory floor → memory-bound IN PRINCIPLE. Four restructure approaches tried + all REVERTED (no improvement or regression): (1) work-efficient O(L) SERIAL scan = 5169µs / ~2× SLOWER (latency-bound 256-deep barrier chain — CONFIRMS the decision rule above); (2) shrink-CH = worse; (3) remove-barriers = no change (auto-inserted PipeBarriers re-added by the compiler on the dependent chain); (4) row-batch for ILP = ~5× slower (linear-depth barrier chain not rescued by batching; UB caps batch-vs-`CH` anyway). Conclusion: HS @ `CH=256` is A5-optimal within the LocalTensor structure. Aligns with OL-231. `(REVERT: 4 restructure hypotheses invalidated — recorded as anti-pattern: do NOT re-try serial O(L) / shrink-CH / remove-barriers / row-batch on this scan class)`.

**Other instances (predicted)**: `cumsum` / `cumprod`, prefix-scan, gated linear attention / linear-RNN recurrences, any associative-scan op where `L` exceeds the UB limit. The `[l*N+n]`-shift-as-wide-vec-op trick generalizes to any per-lane independent scan with a small lane count. The **barrier-depth selection rule + utilization-ceiling caveat** generalize to ANY memory-bound, latency-bound serial recurrence on a per-step-`PipeBarrier` architecture (A5/A3 AIV) where the state width is too narrow to fill the pipe via ILP.

**Cross-reference**: OL-242 (the SIMT-scalar transcendental intrinsics used in the gate computation of the same op), OL-243 (input-cast dtype floor used to grade this op's fp32/fp16 precision), PB-47 (the chunk-loop V→MTE2 fence hazard that bit the bwd path of this same kernel), PB-48 (SIMT GM-scratch cross-grid-stride coherency hazard the bwd L-chunk staging hit — sibling SIMT failure mode), OL-231 (latency-vs-throughput discriminator that settles whether a serial-chain floor is genuine), P-P88/OL-103 (vector-Exp/Ln precision floor for the gate transcendentals).


## P-P114: Multi-core outer_blocks partitioning template (elementwise/fused)

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=elementwise,fused-elementwise`
`verified_on: soc=Ascend950PR_957b; cann=9.0.0`
`status: canonical`

Template for partitioning independent outer iterations across multiple AI cores
via `GetBlockIdx()`. Each core processes a disjoint range of outer_blocks.

**When to use**: Any elementwise or fused-elementwise op where:
  1. Each outer_block is independent (no cross-block data dependency)
  2. outer_blocks >= 2
  3. The op has no atomic/scatter operations

**Template — kernel.h Init()**:

```cpp
__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t N, uint64_t half,
                             uint64_t stride, uint32_t num_cores) {
    uint64_t raw_block = half * stride;
    uint64_t outer_blocks = (stride > 0 && half > 0) ? (N / (2 * raw_block)) : 1;
    num_cores_ = (num_cores > 0) ? num_cores : 1;
    uint64_t core_idx = GetBlockIdx();
    uint64_t base = outer_blocks / num_cores_;
    uint64_t rem  = outer_blocks % num_cores_;
    start_ob_ = core_idx * base + (core_idx < rem ? core_idx : rem);
    count_ob_ = base + (core_idx < rem ? 1 : 0);
    block_size_ = raw_block;
    tileLen_ = MAX_TILE;
}
```

**Template — kernel.h Process()**:

```cpp
__aicore__ inline void Process() {
    for (uint64_t k = 0; k < count_ob_; ++k) {
        uint64_t ob = start_ob_ + k;
        uint64_t xb = ob * 2 * block_size_;
        for (uint64_t o = 0; o < block_size_; o += tileLen_) {
            uint64_t c = (o + tileLen_ > block_size_) ? (block_size_ - o) : tileLen_;
            // ... per-tile compute ...
        }
    }
}
```

**Template — pybind11.cpp launch**:

```cpp
static uint32_t compute_nblk(uint64_t outer_blocks) {
    if (outer_blocks <= 1) return 1;
    return static_cast<uint32_t>(std::min<uint64_t>(outer_blocks, 32));
}
```

**Cross-ref**: OL-254 (decision rule), EC-78 (NBLK=1 diagnostic fallback), P-P115 (combine with zero-copy strided access for split-input ops).

## P-P115: Zero-copy strided split-input kernel template (chunk-then-compute)

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=fused-elementwise,split-input`
`verified_on: soc=Ascend950PR_957b; cann=9.0.0`
`status: canonical`

Template for ops that split input x into (A, B) along a dimension, then compute
`op(A) ⊙ B`. Instead of pybind-side `narrow().contiguous()` (which copies),
pass x directly and let the kernel compute the split internally via offset math.

**Pattern detection** (Phase A analysis):
  Reference contains: `a, b = chunk/split/narrow(x, 2, dim)` followed by `op(a) * b`
  -> Apply this template.

**Pybind side** — NO narrow, NO contiguous A/B copies:

```cpp
torch::Tensor xc = x.contiguous();  // ensure contiguous (no-op if already)
int64_t half = xc.size(dim) / 2;
int64_t stride_dim = xc.stride(dim);
// Launch kernel with (xc, N, half, stride_dim) — no separate A/B tensors
```

**Kernel side** — internal offset computation:

```cpp
uint64_t raw_block = h * s;  // elements per half per outer prefix
uint64_t outer_blocks = (s > 0 && h > 0) ? (N / (2 * raw_block)) : 0;
// In tile loop:
uint64_t xb = ob * 2 * block_size;
DataCopy(a_tile, xGm_[xb + offset], c);     // A = x[ob*2*bs + off]
DataCopy(b_tile, xGm_[xb + block_size + offset], c); // B = x[ob*2*bs+bs + off]
```

**Savings**: Eliminates 2 x (N/2) NPU memory copies. Example [4096,8192] fp32: saves 134MB.

**Anti-pattern**:
```cpp
// DON'T: Copies A and B separately
auto a = x.narrow(dim, 0, half).contiguous();  // NPU copy
auto b = x.narrow(dim, half, half).contiguous(); // another NPU copy
```

**Cross-ref**: OL-255 (decision rule + when NOT to apply), OL-254 (multi-core — combine both), P-P114 (multi-core outer_blocks template — use the same partition structure).

## P-P121: Eliminate UB `Broadcast` in a small-N VF outer-product build via in-register `Gather` — value-identical (precision-neutral), wins on large-row shapes where the broadcast cost bites

`applies_to: soc=Ascend950PR (a5, arch 351x); cann=9.1.x; op_class=SIMD/VF vec builds with a small broadcast dim (N); kernel_type=ascendc __simd_vf__`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500 (selective_scan_fwd_simd R2, 2026-07-24, .171 card2)`

**Pattern**: an outer-product-style VF build that materializes `[L,N]` from `[L]`- and `[N]`-shaped operands typically UB-`Broadcast`s each operand up to `[L,N]` before the elementwise chain (3 `Broadcast<float,2,axis>` + their `PipeBarrier<PIPE_V>` per chunk). For **small N** (N=16 = 4 rows per 64-lane fp32 VL) you can DELETE those UB Broadcasts and generate the broadcast **logically in-register via `Gather`**: build the lane→N index once with `Arange`→`ShiftRights`(÷N)→`ShiftLefts`→`Sub` (`nIndex`, `rowInTile`), `Gather` the tile-invariant `[N]` operand once per call (`Gather(rAf, Af, nIndex)`), and `Gather` the `[L]`-varying operands per tile by `rowIndex`. The Gather reproduces the exact broadcast layout → **value-identical** (fp16/bf16 bit-identical, fp32 unchanged at its floor) = precision-neutral. Fold R1's real `Exp` into the same VF.

**Scope / no-regression**: gate on `if (N==16)` on the bf16/fp16 fast path only; keep the membase `Broadcast`+build path for general-N and the fp32 `softTrans` path (they are UNTOUCHED — no regression).

**Perf (why it wins only on large rows)**: broadcast-elimination bites where the per-row work is large. selective_scan_fwd_simd customer **L=5000 N=16 bf16**: baseline `2467.9µs` → R2 `2214.2µs` = **−10.3% (1.115×)** device-time, precision bit-identical (same-session back-to-back npu.Event A/B, median+min agree 0.1%). Small stock shapes (L≤768) are launch-bound/noisy — the reliable signal is the large customer row (≈80k elem/row vs ≤12k).

**Method note**: this op is custom-`ACLRT`-launch → `torch_npu.profiler` exports empty (DEBT-149) and in-container msprof analyzer is EPERM — measure device-time with `torch.npu.Event`, same session, back-to-back baseline-vs-opt on the SAME card (per the "same-condition A/B" rule). Reconcile the rig's built kernel to the current production md5 FIRST (OL-283) or the baseline is wrong.

**Cross-ref**: P-P106 (the L-chunk + Hillis-Steele scan build this optimizes), OL-245 (regbase amortization boundary — the Gather-build is the amortized-WIN case: one wide gather-fed chain, not high-freq tiny VF), OL-231 (A5 small-N issue-bound ceiling — this shaves a real per-chunk term under it), DEBT-149 (profiler-empty on custom-ACLRT → npu.Event), OL-283 (reconcile rig kernel to production before A/B). backend=ascendc.
