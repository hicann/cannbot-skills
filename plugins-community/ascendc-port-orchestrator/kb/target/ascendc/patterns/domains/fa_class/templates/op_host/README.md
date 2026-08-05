# FA-class op_host prior-art asset (config-agnostic host-tiling reference)

These files are provenance-preserving target-side snapshots. A worker may inspect their field
relationships, lifecycle, public-API usage, and failure hypotheses, but the asset is not an
include-ready delivery and must not be copied verbatim into generated output. The task-owned host
tiling must be emitted from the selected arch22 contract and current arch35 public APIs, then built
and validated through the independent truth boundary.

The archive retains the historical whole-port host tiling so the generator can recover invariants
that are easy to omit (workspace, block distribution, sparse modes, and dispatch fields). It is
advisory evidence for re-derivation, test design, and differential review, not the generated body.

Provenance snapshot from the historical wholeport host
`output/a3_to_a5_port/src/kernels/flash_attention_score/op_kernel/pybind11.cpp`
(`run_flash_attention_score` DoTiling). Its historical arithmetic and comments are retained to preserve
provenance, but delivery code must re-derive the contract rather than mirror this body. The snapshot is
self-contained (only `<cstdint>`/`<algorithm>`/`<limits>`/`<cmath>`); RED LINE: CPU host C++ only,
NO `#include "arch35/"`, NO aclnn/aclop.

## What the ASSET DOCUMENTS (config-agnostic Calc-* prior art)

All in `namespace wp_fa_host`. Each takes shape/dtype params -> computes tiling VALUES (no `at::Tensor`,
no TilingData fill, no dispatch):

| Function | Signature (returns) | Ports arch35 |
|---|---|---|
| `CeilDiv` / `AlignUp` | `(int64_t a, int64_t b) -> int64_t` | op_host CeilDivision / AlignUp |
| `ComputeSparseTiling` | `(sparseMode, s1, s2, preTok, nextTok) -> SparseTiling` | PretokenAndNexttokenAdjustment + GetSparseInfo |
| `CalcEffSparseMode` | `(sparseMode, hasAttenMask) -> int64_t` (effective sparse mode; NO_MASK when no mask) | GetSparseInfo `if (!hasAttenMask) return dense` (pybind11.cpp ~L305-337/L382) |
| `CalcDBasicBlock` | `(D) -> int64_t` | CalcDBasicBlock = AlignUp(D,64) |
| `CalcS1S2BasicBlock` | `(B, n2Size, gSize, S1, D, dBasicBlock, inputDtypeBytes) -> int64_t` (s1BasicBlock) | CalcS1S2BasicBlock L79-115 + CalcTotalSize core-fill L35-49 |
| `SetMultiCoreParamsRegbase` | `(B, n2Size, gSize, S1, s1BasicBlock, coreNum) -> MultiCoreParams` | set_s1OuterSize + CalcTotalSize + SetMultiCoreParamsRegbase regbase.cpp:942-949 |
| `CalcThresholdForS2Size` | `(B, n2Size, gSize, S1, D, attenMaskSize, dataTypeSize, actualUsedCoreNum, l2CacheSize) -> int64_t` | CalcThresholdForS2Size regbase.cpp:1010-1077 |
| `IsUseSplitCoreMode` | `(mode, S1, S2, thresholdS2Size) -> bool` | IsUseSplitCoreMode regbase.cpp:1079-1096 |
| `SetSplitCoreModeParam` | `(effSparseMode, S1, S2, preTokEff, nextTokEff, s1Basic, s1OuterSize, threshold, hasAttenMask) -> SplitCoreResult` | SetSplitCoreModeParam regbase.cpp:1098-1151 |
| `CalcUseDn` | `(hasAtten, hasPse, hasSink, dBasicBlock, useAligned64Kernel) -> bool` (useDn predicate; s1!=64 term dropped — vacuous-true 128-const + subsumed by !useAligned64Kernel, DS re-audit) | `useDnWs` boolean (pybind11.cpp ~L790-800) / kernel IsDn wp_common_regbase.h:163 |
| `CalcWorkspaceSize` | `(D, s1BaseSizeWs, actualUsedCoreNum, useDn) -> int64_t` (totalWsBytes) | GetWorkspaceSize basic.cpp:180 + PostTiling reserve |

Result structs (POD outputs of the Calc-* functions, NOT the kernel TilingData):
`SparseTiling`, `MultiCoreParams`, `SplitCoreResult`. Enums: `SparseMode`, `SparseEnum`,
`AttenMaskCompressMode`.

## What the worker RE-DERIVES (task-owned DoTiling — the advisory checklist)

The kw authors the per-config orchestration ITSELF (it is the gap#2 answer, deliberately ABSENT from
the asset). Recipe for the host side:

1. **config-extract**: read `B, N, S1, D, S2, N_kv, gSize` from the `q/k/v` `at::Tensor` shapes;
   `isFp16/isBf16/isFp32` from dtype; optional-input presence (`maskProvided`, `hasPse`, `hasSink`,
   `hasDropOut`) from the `c10::optional` args; derive `inputDtypeBytes` (fp16/bf16=2, fp32=4).
2. **call the Calc-* functions** (this asset): `ComputeSparseTiling` -> `st`; resolve the raw mask
   predicates `hasAtten` (userMask / compress per the reference-match rule, = config-extract) then
   `effSparseMode = CalcEffSparseMode(sparse_mode, hasAtten)` (the asset derives the dense-force);
   `dBasicBlock = CalcDBasicBlock(D)`; `s1BasicBlock = CalcS1S2BasicBlock(...)`;
   `useAligned64Kernel = (s1BasicBlock==64) && (dBasicBlock<=256)`;
   `mc = SetMultiCoreParamsRegbase(...)`; compute `attenMaskSize` from the effective mask path;
   `threshold = CalcThresholdForS2Size(...)`;
   `scm = SetSplitCoreModeParam(effSparseMode, ..., mc.s1OuterSize, threshold, hasAtten)`;
   `useDn = CalcUseDn(hasAtten, hasPse, hasSink, dBasicBlock, useAligned64Kernel)`
   (the asset derives the predicate; the kw only supplies the extracted presence flags);
   `totalWsBytes = CalcWorkspaceSize(D, s1BaseSizeWs=128, mc.actualUsedCoreNum, useDn)`.
3. **fill the TilingData POD**: `FlashAttentionScoreSimplifiedTilingData td; memset(&td,0,...)`; copy the
   Calc results + config into `td.inputParamsRegbase` (`ip.*`) and `td.multiCoreParamsRegbase` (`mc.*`),
   incl the path-specific fields (attenMaskShapeType, dropout seed/offset/keepProbUint8, pse dims).
4. **dispatch + launch**: write the launcher-selection table itself (the `(dtype × D-bucket × s1Basic ×
   feature) -> wp_fa_do_*` symbol map, e.g. `SelectLauncher`/`SelectPseLauncher`), then alloc the mask /
   workspace tensors, `SetSysWorkspaceForce`, set `blockDim = mc.actualUsedCoreNum`, and call the
   selected `wp_fa_do_*` launcher. Author the `PYBIND11_MODULE`.

## gap#2 integrity boundary (DS-pinned, LOCKED)

**This asset = config-AGNOSTIC Calc-* LOGIC ONLY. ZERO per-config orchestration.** ABSENT by design
(the kw reproduces these): config-extract, dispatch table (`SelectLauncher`/`SelectPseLauncher`), the
`FlashAttentionScoreSimplifiedTilingData` POD fill, the launch glue, the pybind module. If the
orchestration leaked here, the agent could copy it -> gap#2 trivialized.

**Integrity proof** (grep-verified, code-only — comment mentions excluded):
`grep -nvE '^\s*(\*|//|/\*)' wp_fa_host_tiling.h | grep -E
'run_flash_attention_score|PYBIND11_MODULE|SelectLauncher|SelectPseLauncher|FlashAttentionScoreSimplifiedTilingData|wp_fa_do_|at::Tensor|launchFn|inputParamsRegbase|multiCoreParamsRegbase'`
= **0 hits**. (The `mc.*` writes in the asset are to the LOCAL `MultiCoreParams` return struct of the
agnostic `SetMultiCoreParamsRegbase`, NOT the kernel `td.multiCoreParamsRegbase`.)
