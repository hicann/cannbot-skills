# ST 测试覆盖率报告

当前 ST 框架覆盖 **102 个 Skill + 18 个 Team**，共 **865 个评测用例**（仅统计已启用用例，截止 2026-08-29）。

> **统计口径**：统计代码仓内所有符合结构要求的 Skill（`skill_dirs` 下含 `SKILL.md`）与 Team（`team_dirs` 下含 `AGENTS.md` + `.claude-plugin/plugin.json`），与是否携带 `evals/evals.json` 无关——无 evals 的实体用例数计 0。

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

### ops/（70 Skills，共 555 个用例）

| 名称 | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| aiss-tiling-solver | 5 |   |   | √ |   | √ |
| ascendc-api-best-practices | 10 | √ |   | √ |   | √ |
| ascendc-blaze-best-practice | 39 |   |   | √ |   | √ |
| ascendc-blaze-migration | 10 |   |   | √ | √ | √ |
| ascendc-code-review | 7 | √ |   | √ |   | √ |
| ascendc-crash-debug | 7 | √ |   | √ |   | √ |
| ascendc-direct-invoke-template | 7 | √ |   | √ |   | √ |
| ascendc-direct-invoke-to-registry-invoke | 8 |   |   | √ |   | √ |
| ascendc-docs-gen | 11 |   |   | √ |   | √ |
| ascendc-docs-search | 9 | √ |   | √ |   | √ |
| ascendc-env-check | 11 | √ |   | √ |   | √ |
| ascendc-mc2-best-practice | 19 | √ |   | √ | √ | √ |
| ascendc-perf-optimize | 8 |   |   | √ |   | √ |
| ascendc-performance-best-practices | 7 | √ |   | √ |   | √ |
| ascendc-precision-debug | 9 | √ |   | √ |   | √ |
| ascendc-regbase-best-practice | 7 |   |   | √ |   | √ |
| ascendc-registry-invoke-template | 14 | √ |   | √ |   | √ |
| ascendc-registry-invoke-to-direct-invoke | 3 |   |   | √ |   | √ |
| ascendc-runtime-debug | 7 |   |   | √ |   | √ |
| ascendc-simt-best-practices | 2 |   |   | √ |   | √ |
| ascendc-simt-tiling-design | 2 |   |   | √ |   | √ |
| ascendc-st-design | 2 |   |   | √ |   | √ |
| ascendc-sync-audit | 5 | √ |   | √ |   | √ |
| ascendc-tiling-design | 2 |   |   | √ |   | √ |
| ascendc-ut-develop | 10 | √ | √ | √ | √ | √ |
| ascendc-whitebox-design | 10 | √ | √ | √ |   | √ |
| cann-env-setup | 7 | √ |   | √ | √ | √ |
| catlass-op-design | 2 |   |   | √ |   | √ |
| catlass-op-develop | 2 |   |   | √ |   | √ |
| catlass-op-perf-tune | 2 |   |   | √ |   | √ |
| npu-arch | 19 | √ | √ | √ | √ | √ |
| ops-precision-standard | 6 |   |   | √ |   | √ |
| ops-profiling | 9 |   |   | √ |   | √ |
| ops-simulator | 7 |   |   | √ |   | √ |
| ops-spec-gen | 7 |   |   | √ |   | √ |
| pypto-api-explore | 7 |   |   | √ |   | √ |
| pypto-docs-search | 8 |   |   | √ |   | √ |
| pypto-general-debug | 8 |   |   | √ |   | √ |
| pypto-golden-generate | 7 |   |   | √ |   | √ |
| pypto-intent-understand | 7 |   |   | √ |   | √ |
| pypto-memory-template | 10 |   |   | √ |   | √ |
| pypto-op-construct | 10 |   |   | √ |   | √ |
| pypto-op-design | 10 | √ | √ | √ | √ | √ |
| pypto-op-develop | 7 |   |   | √ |   | √ |
| pypto-op-knowledge | 8 |   |   | √ |   | √ |
| pypto-op-perf-tune | 7 |   |   | √ |   | √ |
| pypto-op-plan | 8 |   |   | √ |   | √ |
| pypto-op-review | 8 |   |   | √ |   | √ |
| pypto-op-verify | 10 |   |   | √ |   | √ |
| pypto-orchestration-manual | 8 |   |   | √ |   | √ |
| pypto-precision-compare | 7 |   |   | √ |   | √ |
| pypto-precision-debug | 7 |   |   | √ |   | √ |
| tilelang-api-best-practices | 7 |   |   | √ |   | √ |
| tilelang-env-check | 7 |   |   | √ |   | √ |
| tilelang-op-design | 7 |   |   | √ |   | √ |
| tilelang-op-develop | 7 |   |   | √ |   | √ |
| tilelang-op-test-design | 7 |   |   | √ |   | √ |
| tilelang-perf-optimization | 7 |   |   | √ |   | √ |
| tilelang-programming-model-guide | 7 |   |   | √ |   | √ |
| tilelang-review | 7 |   |   | √ |   | √ |
| tilelang-submodule-pull | 6 |   |   | √ |   | √ |
| torch-ascendc-op-extension | 7 |   |   | √ |   | √ |
| torch-ops-profiler | 2 |   |   | √ |   | √ |
| triton-latency-optimizer | 7 |   |   | √ |   | √ |
| triton-op-coding | 7 |   |   | √ |   | √ |
| triton-op-designer | 7 |   |   | √ |   | √ |
| triton-op-verifier | 7 |   |   | √ |   | √ |
| triton-precision-debug | 10 | √ |   | √ |   | √ |
| triton-simulator-optimizer | 10 |   |   | √ |   | √ |
| triton-task-extractor | 7 |   |   | √ |   | √ |

### graph/（8 Skills，共 66 个用例）

| 名称 | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| ge-fusion-pass-skill | 5 | √ |   | √ | √ | √ |
| torch-custom-ops-guide | 10 | √ |   | √ | √ | √ |
| torch-npugraph-ex-compile-error-diagnosis | 8 | √ |   | √ | √ | √ |
| torch-npugraph-ex-dfx-triage | 9 | √ |   | √ | √ | √ |
| torch-npugraph-ex-knowledge | 9 | √ |   | √ | √ | √ |
| torch-npugraph-ex-performance-diagnosis | 9 | √ |   | √ | √ | √ |
| torch-npugraph-ex-runtime-error-diagnosis | 9 | √ |   | √ | √ | √ |
| torch-npugraph-ex-template | 7 |   |   | √ | √ | √ |

### model/（18 Skills，共 169 个用例）

| 名称 | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| model-infer-fusion | 11 | √ |   | √ | √ | √ |
| model-infer-graph-mode | 11 | √ |   | √ | √ | √ |
| model-infer-harmony | 6 | √ |   | √ | √ | √ |
| model-infer-kvcache | 11 | √ |   | √ | √ | √ |
| model-infer-migrator | 11 | √ |   | √ | √ | √ |
| model-infer-multi-stream | 11 | √ |   | √ | √ | √ |
| model-infer-parallel-analysis | 11 | √ |   | √ | √ | √ |
| model-infer-parallel-impl | 11 | √ |   | √ | √ | √ |
| model-infer-perf-breakdown | 7 |   |   | √ | √ | √ |
| model-infer-precision-debug | 11 | √ |   | √ | √ | √ |
| model-infer-prefetch | 11 | √ |   | √ | √ | √ |
| model-infer-profiling | 7 |   |   | √ | √ | √ |
| model-infer-quantization | 11 | √ |   | √ | √ | √ |
| model-infer-runtime-debug | 11 | √ |   | √ | √ | √ |
| model-infer-superkernel | 11 | √ |   | √ | √ | √ |
| model-train-accuracy-debug | 6 | √ |   | √ | √ | √ |
| model-train-log-visualization | 5 | √ |   | √ | √ | √ |
| model-train-oom-analysis | 6 | √ |   | √ | √ | √ |

### infra/（5 Skills，共 34 个用例）

| 名称 | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| cannbot-skill-reviewer | 5 |   |   | √ |   | √ |
| gitcode-issue-gen | 7 | √ |   | √ |   | √ |
| gitcode-issue-handler | 1 | √ |   | √ | √ | √ |
| gitcode-pr-handler | 9 |   |   | √ |   | √ |
| gitcode-toolkit | 12 | √ |   | √ | √ | √ |

### runtime/（1 Skills，共 0 个用例）

| 名称 | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| runtime_migration | 0 |   |   |   |   |   |

## 3. Team 覆盖率

### plugins-official/（10 Teams，共 41 个用例）

| 名称 | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| catlass-op-generator | 0 |   |   |   |   |   |
| model-infer-optimize | 7 | √ |   | √ |   | √ |
| ops-code-reviewer | 3 | √ |   | √ |   | √ |
| ops-direct-invoke | 2 |   |   | √ |   | √ |
| ops-direct-invoke-flash | 9 | √ |   | √ |   | √ |
| ops-registry-invoke | 13 | √ |   | √ | √ | √ |
| pypto-op-orchestrator | 3 | √ |   | √ |   | √ |
| tilelang-op-orchestrator | 3 | √ |   | √ |   | √ |
| torch-compile | 1 |   |   | √ |   | √ |
| triton-op-generator | 0 |   |   |   |   |   |

### plugins-community/（8 Teams，共 0 个用例）

| 名称 | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| ascendc-port-orchestrator | 0 |   |   |   |   |   |
| autoresearch | 0 |   |   |   |   |   |
| cannbot-knowledge | 0 |   |   |   |   |   |
| ops-perf-evolution | 0 |   |   |   |   |   |
| ops-perf-optimize | 0 |   |   |   |   |   |
| shmem-ops-generator | 0 |   |   |   |   |   |
| tilelang2ascendc-ops-generator | 0 |   |   |   |   |   |
| triton-optimizer | 0 |   |   |   |   |   |
