# ST 测试覆盖率报告

当前 ST 框架覆盖 **101 个 Skill + 8 个 Team**，共 **543 个评测用例**（仅统计已启用用例，截止 2026-08-20）。

## 1. 五维看护说明

| 维度 | 测试目标 | 判定标志（evals.json） |
| --- | --- | --- |
| **正向看护** | 在多个类似 skill/team 同时存在时，AI 能正确选择目标 skill | `config.distractor_skills` 非空 |
| **负向看护** | 在边界/无关场景下，AI 不会被误触发 | `expectations` 中存在 `not_contains` |
| **正确性看护** | 黑盒场景验证：AI 回复语义覆盖关键要点 | `expected_output` 定义预期要点 |
| **调用流程看护** | 验证关键工具被调用、关键文件被生成 | `expectations` 中存在 `file_exists`/`file_list`/`file_contains`/`skill_activated` |
| **资源消耗看护** | Token 消耗监控，防止资源浪费 | `config.max_tokens` 已配置 |

> 仅统计**已启用**（`config.disabled` 非 true）的用例。仅在已禁用用例中配置的维度视同无覆盖。

## 2. Skill 覆盖率

按域分组统计：

### ops/（69 Skills，共 353 个用例）

| Skill | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| aiss-tiling-solver | 5 |  |  | √ |  | √ |
| ascendc-api-best-practices | 10 | √ |  | √ |  | √ |
| ascendc-blaze-best-practice | 39 |  |  | √ |  | √ |
| ascendc-blaze-migration | 13 |  |  | √ |  | √ |
| ascendc-code-review | 7 | √ |  | √ |  | √ |
| ascendc-crash-debug | 7 | √ |  | √ |  | √ |
| ascendc-direct-invoke-template | 7 | √ |  | √ |  | √ |
| ascendc-direct-invoke-to-registry-invoke | 10 | √ |  | √ |  | √ |
| ascendc-docs-gen | 11 |  |  | √ |  | √ |
| ascendc-docs-search | 9 | √ |  | √ |  | √ |
| ascendc-env-check | 11 | √ |  | √ |  | √ |
| ascendc-mc2-best-practice | 9 | √ |  | √ | √ | √ |
| ascendc-perf-optimize | 8 |  |  | √ |  | √ |
| ascendc-performance-best-practices | 7 | √ |  | √ |  | √ |
| ascendc-precision-debug | 9 | √ |  | √ |  | √ |
| ascendc-regbase-best-practice | 7 |  |  | √ |  | √ |
| ascendc-registry-invoke-template | 14 | √ |  | √ |  | √ |
| ascendc-registry-invoke-to-direct-invoke | 3 |  |  | √ |  | √ |
| ascendc-runtime-debug | 7 |  |  | √ |  | √ |
| ascendc-simt-best-practices | 2 |  |  | √ |  | √ |
| ascendc-simt-tiling-design | 2 |  |  | √ |  | √ |
| ascendc-st-design | 2 |  |  | √ |  | √ |
| ascendc-tiling-design | 2 |  |  | √ |  | √ |
| ascendc-ut-develop | 10 | √ | √ | √ | √ | √ |
| ascendc-whitebox-design | 10 | √ | √ | √ |  | √ |
| cann-env-setup | 7 | √ |  | √ | √ | √ |
| catlass-op-design | 2 |  |  | √ |  | √ |
| catlass-op-develop | 2 |  |  | √ |  | √ |
| catlass-op-perf-tune | 2 |  |  | √ |  | √ |
| npu-arch | 10 | √ |  | √ | √ | √ |
| ops-precision-standard | 1 |  |  | √ |  | √ |
| ops-profiling | 4 |  |  | √ |  | √ |
| ops-simulator | 2 |  |  | √ |  | √ |
| ops-spec-gen | 2 |  |  | √ |  | √ |
| pypto-api-explore | 2 |  |  | √ |  | √ |
| pypto-docs-search | 3 |  |  | √ |  | √ |
| pypto-general-debug | 3 |  |  | √ |  | √ |
| pypto-golden-generate | 2 |  |  | √ |  | √ |
| pypto-intent-understand | 2 |  |  | √ |  | √ |
| pypto-memory-template | 5 |  |  | √ |  | √ |
| pypto-op-construct | 5 |  |  | √ |  | √ |
| pypto-op-design | 5 | √ | √ | √ | √ | √ |
| pypto-op-develop | 2 |  |  | √ |  | √ |
| pypto-op-knowledge | 5 |  |  | √ |  | √ |
| pypto-op-perf-tune | 2 |  |  | √ |  | √ |
| pypto-op-plan | 4 |  |  | √ |  | √ |
| pypto-op-review | 4 |  |  | √ |  | √ |
| pypto-op-verify | 5 |  |  | √ |  | √ |
| pypto-orchestration-manual | 5 |  |  | √ |  | √ |
| pypto-precision-compare | 2 |  |  | √ |  | √ |
| pypto-precision-debug | 2 |  |  | √ |  | √ |
| tilelang-api-best-practices | 2 |  |  | √ |  | √ |
| tilelang-env-check | 2 |  |  | √ |  | √ |
| tilelang-op-design | 2 |  |  | √ |  | √ |
| tilelang-op-develop | 2 |  |  | √ |  | √ |
| tilelang-op-test-design | 2 |  |  | √ |  | √ |
| tilelang-perf-optimization | 2 |  |  | √ |  | √ |
| tilelang-programming-model-guide | 2 |  |  | √ |  | √ |
| tilelang-review | 2 |  |  | √ |  | √ |
| tilelang-submodule-pull | 2 |  |  | √ |  | √ |
| torch-ascendc-op-extension | 2 |  |  | √ |  | √ |
| torch-ops-profiler | 2 |  |  | √ |  | √ |
| triton-latency-optimizer | 2 |  |  | √ |  | √ |
| triton-op-coding | 2 |  |  | √ |  | √ |
| triton-op-designer | 2 |  |  | √ |  | √ |
| triton-op-verifier | 2 |  |  | √ |  | √ |
| triton-precision-debug | 5 | √ |  | √ |  | √ |
| triton-simulator-optimizer | 5 |  |  | √ |  | √ |
| triton-task-extractor | 2 |  |  | √ |  | √ |

### graph/（8 Skills，共 31 个用例）

| Skill | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| ge-fusion-pass-skill | 5 | √ |  | √ | √ | √ |
| torch-custom-ops-guide | 5 | √ |  | √ | √ | √ |
| torch-npugraph-ex-compile-error-diagnosis | 3 | √ |  | √ | √ | √ |
| torch-npugraph-ex-dfx-triage | 4 | √ |  | √ | √ | √ |
| torch-npugraph-ex-knowledge | 4 | √ |  | √ | √ | √ |
| torch-npugraph-ex-performance-diagnosis | 4 | √ |  | √ | √ | √ |
| torch-npugraph-ex-runtime-error-diagnosis | 4 | √ |  | √ | √ | √ |
| torch-npugraph-ex-template | 2 |  |  | √ | √ | √ |

### model/（18 Skills，共 99 个用例）

| Skill | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| model-infer-fusion | 6 | √ |  | √ | √ | √ |
| model-infer-graph-mode | 6 | √ |  | √ | √ | √ |
| model-infer-harmony | 6 | √ |  | √ | √ | √ |
| model-infer-kvcache | 6 | √ |  | √ | √ | √ |
| model-infer-migrator | 6 | √ |  | √ | √ | √ |
| model-infer-multi-stream | 6 | √ |  | √ | √ | √ |
| model-infer-parallel-analysis | 6 | √ |  | √ | √ | √ |
| model-infer-parallel-impl | 6 | √ |  | √ | √ | √ |
| model-infer-perf-breakdown | 2 |  |  | √ | √ | √ |
| model-infer-precision-debug | 6 | √ |  | √ | √ | √ |
| model-infer-prefetch | 6 | √ |  | √ | √ | √ |
| model-infer-profiling | 2 |  |  | √ | √ | √ |
| model-infer-quantization | 6 | √ |  | √ | √ | √ |
| model-infer-runtime-debug | 6 | √ |  | √ | √ | √ |
| model-infer-superkernel | 6 | √ |  | √ | √ | √ |
| model-train-accuracy-debug | 6 | √ |  | √ | √ | √ |
| model-train-log-visualization | 5 | √ |  | √ | √ | √ |
| model-train-oom-analysis | 6 | √ |  | √ | √ | √ |

### infra/（5 Skills，共 24 个用例）

| Skill | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| cannbot-skill-reviewer | 5 |  |  | √ |  | √ |
| gitcode-issue-gen | 7 | √ |  | √ |  | √ |
| gitcode-issue-handler | 1 | √ |  | √ | √ | √ |
| gitcode-pr-handler | 4 |  |  | √ |  | √ |
| gitcode-toolkit | 7 | √ |  | √ | √ | √ |

### runtime/（1 Skills，共 5 个用例）

| Skill | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| runtime_migration | 5 |  |  | √ |  | √ |

## 3. Team 覆盖率

### Team（8 Teams，共 31 个用例）

| Team | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| model-infer-optimize | 7 | √ |  | √ |  | √ |
| ops-code-reviewer | 3 | √ |  | √ |  | √ |
| ops-direct-invoke | 2 |  |  | √ |  | √ |
| ops-direct-invoke-flash | 4 | √ |  | √ |  | √ |
| ops-registry-invoke | 8 | √ |  | √ | √ | √ |
| pypto-op-orchestrator | 3 | √ |  | √ |  | √ |
| tilelang-op-orchestrator | 3 | √ |  | √ |  | √ |
| torch-compile | 1 |  |  | √ |  | √ |
