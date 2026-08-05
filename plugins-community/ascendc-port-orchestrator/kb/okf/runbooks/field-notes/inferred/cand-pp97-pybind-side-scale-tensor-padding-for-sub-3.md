---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Pybind-side scale-tensor padding for sub-32B-per-row inputs in per-block quant kernels"
description: "applies_to: soc=Ascend950PR; cann=9.0.0+; bisheng=all; op_class=mx-quant,mxfp8,mxfp4,per-block-quant,any kernel reading a sub-32B-per-row scale/index/metadata tensor verified_on: soc=Ascend950PR_957c;"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0+; bisheng=all; op_class=mx-quant,mxfp8,mxfp4,per-block-quant,any kernel reading a sub-32B-per-row scale/index/metadata t"
confidence: inferred
status: stub
original_id: CAND-PP97
timestamp_inferred: true
tags: [candidate, inferred, count, datacopy, n_blocks_active, n_blocks_stride, datacopypad, cand-pp97]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0+; bisheng=all; op_class=mx-quant,mxfp8,mxfp4,per-block-quant,any kernel reading a sub-32B-per-row scale/index/metadata tensor`
`verified_on: soc=Ascend950PR_957c; cann=9.1.0.B010 (MxFp8LayerNorm — single op evidence on D=768 case)`
`unverified_on: soc=Ascend910_V220; op_class=other-quant-formats (int4-grouped, GPTQ-style, AWQ, mxfp4)`

**Principle**: AscendC's `DataCopy(dst, src, count)` requires `count * sizeof(T)` to be a multiple of 32 bytes — non-aligned `count` silently rounds DOWN (see OL-167). For input-side tensors whose per-row payload is naturally sub-32B (e.g. per-block scale bytes in MX-format kernels: `n_blocks = D / 32` uint8 bytes where small `D` makes `n_blocks < 32`), the kernel cannot use plain DataCopy to load one row's scales. The fix is **host-side pybind padding** that zero-extends the scale tensor along the last dim to a multiple of 32, while the kernel iterates only over the original (unpadded) per-row scale count.

This is **distinct from OL-167's anti-pattern** (host-side OUTPUT padding + `narrow+contiguous` to hide kernel non-aligned writes — that's data-path cheating). Here the padding is on the INPUT side, the kernel READS only the original-count active scales, and the zero-padded tail is provably unused. Input metadata layout IS a pybind responsibility because:
1. The kernel cannot reshape inputs (it only consumes GM pointers).
2. The per-block-scale layout convention (`scales.shape[-1] = D / block_size`) is part of the public op contract — the kernel can't unilaterally change it.
3. The `DataCopy` 32B-alignment is a hardware constraint on the LOAD primitive, NOT a contract on the data layout.

**Concrete anchor** (pybind sketch, per-block-scale variant):

```cpp
// inputs: x_scales is uint8 shape [..., n_blocks] where n_blocks = D / BLOCK_SIZE
int64_t n_blocks = D / BLOCK_SIZE;
int64_t n_blocks_pad = ((n_blocks + 31) / 32) * 32;        // align_up to 32 bytes

torch::Tensor scales_eff = x_scales;
if (n_blocks_pad != n_blocks) {
    auto pad_shape = x_scales.sizes().vec();
    pad_shape.back() = n_blocks_pad;
    scales_eff = torch::zeros(pad_shape, x_scales.options());    // zero-init padding (NOT empty)
    scales_eff.narrow(-1, 0, n_blocks).copy_(x_scales);
}
// kernel reads scales_eff with stride n_blocks_pad per row, iterates only n_blocks per row
launch_kernel(..., scales_eff.data_ptr<uint8_t>(), /*n_blocks_active=*/n_blocks,
                                                    /*n_blocks_stride=*/n_blocks_pad, ...);
```

**Correctness invariant**: the kernel's per-row loop bound is `n_blocks_active` (the ORIGINAL count); the kernel's DataCopy uses `n_blocks_stride` (the padded count) ONLY for the DataCopy granularity. The padded tail bytes are READ into UB but never INDEXED by the compute loop, so their zero-init content is provably ignored.

**Why this is NOT covered by OL-162 or OL-167**:
- OL-162 covers asymmetric-shape kernels where one input's dim is hard-coded as buffer extent for a differently-shaped tensor (kernel-internal OOR). Different fault class.
- OL-167 forbids host-side OUTPUT padding + `narrow+contiguous` (kernel hides non-aligned writes). Different fault direction (input vs output) and different load-bearing reason (kernel CAN handle non-aligned input via DataCopyPad, but DataCopyPad for tiny scale tensors costs more than the host-side zero-extend; for per-block scales, the host pad is the natural fix).

**Trigger classifier**:
- `inputs.<scale_or_metadata_tensor>.shape[-1] * sizeof(dtype) < 32B` per row → pybind padding applies.
- `inputs.<main_data_tensor>.shape[-1] * sizeof(dtype) < 32B` per row → main-data tile so small the kernel architecture is wrong; revisit tiling.

**Anti-patterns**:
- ❌ `DataCopy(scalesLocal, gmXScales, n_blocks)` with `n_blocks < 32` → silently transfers ZERO bytes per OL-167. Symptom: scale = uninitialized → dequant produces garbage → precision verification fails with wildly out-of-range MERE.
- ❌ Using `DataCopyPad` for tiny per-block-scale loads — works but adds kernel complexity (extra params, smaller throughput); the host zero-extend is strictly simpler for INPUT scales.
- ❌ Forcing the kernel to compute the padding at GM-load time via element-wise scalar `GetValue` — wastes scalar pipe, much slower than a single DataCopy from a pre-padded host tensor.
- ❌ Assuming all valid op shapes will have `n_blocks ≥ 32` naturally — sweep your `D` cases; small-D variants (D=768 in MXFP8 LayerNorm, D=256 in some attention configs) will hit this.

**Evidence**:
- MxFp8LayerNorm kw-1 (2026-05-21 Ascend950PR_957c, CANN 9.1.0.B010): 8 benchmark cases sweep D ∈ {768, 1024, 2048, 4096} → only D=768 (case 3, n_blocks=24 < 32) needs padding. D ∈ {1024, 2048, 4096} → n_blocks ∈ {32, 64, 128} naturally aligned. Pybind detects via `n_blocks < 32` and zero-pads `x_scales` to `[..., 32]`. Kernel reads `n_blocks_active=24` for the compute loop. Pass-A 8/8 + Pass-B 11/11 PASS (case 3 specifically validates the padded path).

**Promotion gate**: 2+ op evidence required. Next candidate: any future MX-format / per-block-quant op with sub-32B per-row metadata (mxfp4 grouped matmul, per-block int4-quant with E4M3 scales, AWQ/GPTQ-style scale tensors). Then promote to `patterns/domains/quant.md` or `patterns/domains/data_movement.md` as a P-Pxx alongside P-P98 (DataCopyPad).

**Cross-ref**:
- OL-167 (DataCopy `count` silent truncation — explains why the bare DataCopy doesn't work; this candidate is the input-side mitigation, NOT the cheat OL-167 forbids)
- OL-162 (pybind padding wrapper for asymmetric-shape OOR — different fault class but same broad architectural shape: host pads to make kernel-internal extent invariants hold)
- P-P98 (DataCopyPad — the alternative for output-side non-aligned writes; mentioned here only to contrast: not applicable to small input scales)
- PB-26 / OL-144 (MX-format / fp8_e8m0_t scale convention — where the sub-32B-per-row constraint comes from)

## CAND-DEPLOY-STAGE-LOCAL_TASK: Standalone-mode kernel-worker spawns must stage workspace/<op>/ → LOCAL_TASK before invoking deploy_to_npu.sh

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all; mode=standalone_worker_spawn (NOT orchestrator-driven)`
`verified_on: soc=Ascend950PR_957c; cann=9.1.0.B010 (MxFp8LayerNorm kw-1 — single op evidence)`
`unverified_on: soc=Ascend910_V220 (A3 deploy path uses a different `deploy_to_npu.sh` stagedir layout — pattern may transfer but not yet replayed); orchestrator-driven mode (stage step is handled by orchestrator's Phase O3 hand-off, this candidate doesn't apply)`

**Principle**: `src/scripts/deploy_to_npu.sh` canonicalizes the kernel source through `LOCAL_TASK=$HOME/workspace/AscendOpGenAgent/current_task` when the env var is unset. In orchestrator-driven `/ascendc-op-gen` runs, the orchestrator's Phase O3 hand-off stages `workspace/<op>/{kernel,op_host,op_kernel,...}` into this LOCAL_TASK path before invoking the deploy script. When a worker is spawned **standalone** (no orchestrator wrapping — e.g. via direct `Agent(subagent_type=aog-kernel-worker, ...)` call from another agent / from `/aog-orchestrator-recover`, or via the `Skill` invocation entry-point), there is no Phase O3 stager, and the worker MUST stage its own files OR `deploy_to_npu.sh` re-deploys whatever was last in LOCAL_TASK from the previous op.

**Concrete anchor** (worker-side guard, drop in front of every deploy):

```bash
LOCAL_TASK=${LOCAL_TASK:-$HOME/workspace/AscendOpGenAgent/current_task}
WORKSPACE_OP="${WORKSPACE_ROOT:-/mnt/d/projects/a5_ops/workspace}/${OP_NAME}"

# Standalone-mode stage: mirror workspace/<op>/ → LOCAL_TASK
if [ -z "${ORCHESTRATOR_STAGED:-}" ]; then
    rm -rf "$LOCAL_TASK"
    cp -r "$WORKSPACE_OP/." "$LOCAL_TASK/"
fi

bash src/scripts/deploy_to_npu.sh --build
```

The orchestrator sets `ORCHESTRATOR_STAGED=1` after Phase O3 — so the guard is a no-op in orchestrator-driven mode but covers the standalone-spawn case.

**Symptom when missing** (caught first-build, kw-1): build appears to succeed but the resulting `.so` exports the previous op's symbols. Verification then fails with `ModuleNotFoundError: No module named '<current_op>'` (the build deployed `<previous_op>`'s files because LOCAL_TASK still held them). The error is far from the root cause — looks like a Python import bug, actually is a deploy-stage bug.

**Detection signature** (post-build sanity check):

```bash
# After deploy, verify LOCAL_TASK contains the CURRENT op's kernel files
test -f "$LOCAL_TASK/op_host/<op>_def.cpp" || echo "BUG: LOCAL_TASK contains wrong op's files"
test -f "$LOCAL_TASK/kernel/<op>_kernel.h" || echo "BUG: LOCAL_TASK contains wrong op's files"
```

**Anti-patterns**:
- ❌ Trusting `deploy_to_npu.sh` to read from `workspace/<op>/` directly — it does NOT (LOCAL_TASK is the canonical source).
- ❌ Overriding `LOCAL_TASK` env to point at `workspace/<op>/` without mirroring — `deploy_to_npu.sh` writes intermediates into LOCAL_TASK, corrupting the workspace working copy.
- ❌ Skipping the `rm -rf` before the `cp -r` — stale files from a previous op (op_host JSONs, partial CMake artifacts) survive and the build picks them up alongside the current op's files.

**Why this surfaced**: MxFp8LayerNorm kw-1 was spawned in this session right after 10_LayerNorm finalized. LOCAL_TASK still held 10_LayerNorm's files. First build went green (rebuilt the 10_LayerNorm .so), then verification failed with `ModuleNotFoundError: No module named 'MxFp8LayerNorm'` because the .so didn't export the MxFp8LayerNorm pybind module. Re-staged + rebuild → clean.

**Promotion gate**: 2+ op evidence in standalone-spawn mode (next candidate: any future direct `Agent(subagent_type=aog-kernel-worker, ...)` spawn that exhibits the same first-build wrong-op symptom). Then either:
1. Promote to OL alongside OL-160 (canonical entry-points) covering deploy-stage canonicality.
2. Land a `src/scripts/stage_and_deploy.sh workspace/<op>` wrapper that bakes the guard in, making the candidate self-obsoleting.

**Cross-ref**:
- OL-160 (canonical entry-point file names — the safety net assumes the .so it loads exports the canonical op; this candidate explains how the .so can end up exporting the WRONG op)
- `src/scripts/deploy_to_npu.sh` (the script being staged for)
- `ascendc-op-gen` Phase O3 (orchestrator-driven stager — sets `ORCHESTRATOR_STAGED=1` so the guard noops)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP97，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
