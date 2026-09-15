---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Cube-class A5 ports use native MIX (Mmad cube + vector epilogue + WorkspaceQueue cross-core sync), never pure-vec"
description: "For a port_a3_to_a5 op whose CANN reference is cube-required (matmul/attention/conv/rnn/gmm/ffn family — tagged CUBE_MIX by _cmd_port_a3 Layer 1, enforced by finalize gate _check_architecture_class pe"
severity: high
confidence: single_run
original_id: P-P102
timestamp_inferred: true
tags: [platform_compat, optimization, port_a3_to_a5, cube_mix, _cmd_port_a3, _check_architecture_class, architectural_hack, p-p102, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

For a `port_a3_to_a5` op whose CANN reference is cube-required (matmul/attention/conv/rnn/gmm/ffn family — tagged `CUBE_MIX` by `_cmd_port_a3` Layer 1, enforced by finalize gate `_check_architecture_class` per OL-188/PR#316: pure-VEC = `ARCHITECTURAL_HACK`). **Scaffold**: file split `<op>_cube.h`(cube class `Cube` in name) + `<op>_vec.h`(`Vec`) + `<op>_kernel.h`/`.cpp` orchestrator; cube primitive = manual `AscendC::Mmad` (`cmatrixInitVal=true` single-K-tile; FA chose `Mmad` over `matmul::Matmul<>` after V220 ~500× numerical error + KFC standalone deadlock); vec epilogue = op reduction/activation (FA: `SoftmaxFlashV2`); cross-core sync = `WorkspaceQueue<T,RING_SLOTS=3>` ONE per producer↔consumer direction (FA:3), paired flag IDs, **raw `PIPE_FIX`/`PIPE_MTE3`/`PIPE_MTE2` literals** (CANN 9.0.0 forbids templated `pipe_t`; NOT inline `CrossCoreSetFlag` in loop → 507015); task type `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` (V220-native, PB-28 FALSIFIED — don't arch-guard out); regbase per-AIV sub-block tiling under V351 248KiB UB. **WHY** (adapt-not-copy): AIC/AIV are separate cores; sync is a counting-semaphore handshake pairing one AIC producer↔AIV consumer per direction with producer-PIPE↔consumer-PIPE matching — reproduce the contract (one queue/direction, PIPE pairing, `i%K`-staggered schedule per OL-200), not FA tile sizes. **Methodology**: understand(V220 src)→KB(this pattern)→research(A5 ref for understanding, anti-copy net = ARCH35_WRAP_CHEAT + copy-shape scanner)→generate→regenerate on reject (NEVER pure-vec fallback). Worked example reached via THIS KB, **NOT** another op's `output/` archive (own-dir-only guard `b96508ec`). Full body: `patterns/domains/cube_vector_fusion.md`. Cross-ref OL-188/OL-190/OL-200/EC-57/EC-58/PB-35. **Harness-linkage root cause (OL-235)**: the manual-Mmad default is not only the FA numerical/deadlock evidence — `build_ascendc.py`'s pybind link line is fixed to `kernels torch_npu m dl`, so the host `TCubeTiling` that a `matmul::Matmul<>` kernel needs (`MatmulApiTiling::GetTiling`) is never linkable at the launch layer ⇒ matmul-library cube path is structurally unbuildable through this harness for ALL CUBE_MIX ports (not just FA). `applies_to: soc=Ascend950PR/V351; cann=9.0.0; op_class=CUBE_MIX (matmul/attention/conv/rnn/gmm/ffn); verified_on=flash_attention_score A5 (decision_manifest 2026-05-29); deformable_conv2d port_a3 2026-06-20 (CUBE_MIX conv-family, empirical pybind-link probe → manual-Mmad + bilinear-deform vec MIX, structural_rewrite verdict)`.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P102，convert_patterns_to_okf.py）。confidence 未升格。 -->
