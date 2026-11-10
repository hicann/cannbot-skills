# ST 测试覆盖率报告

历史统计基准（截至 2026-10-23）：**112 个 Skill + 17 个 Team**，共 **913 个评测用例**（仅统计已启用用例）。以下表格保留该次统计结果，不代表当前仓库全量。

> **2026-10-23 迁出增量**：本次删除 `ops-direct-invoke` 旧副本，Team 减少 1；该副本没有 `evals/evals.json`，启用用例变化为 0，公共 Skill 数不变。本次未重新生成全量报告，不能将历史基准直接视为最新总数。

> **2026-10-29 迁出增量**：删除 `ops-perf-optimize` 旧副本，Team 再减少 1；该副本没有 `evals/evals.json`，启用用例变化为 0，公共 Skill 数不变。下表保留历史条目并标注迁出，未重新生成全量报告。

> **2026-10-29 迁出增量（Flash）**：删除 `ops-direct-invoke-flash` 旧副本（含 `evals/evals.json`，9 个启用用例），Team 再减少 1；本仓统计不再包含该 9 个用例，下表保留历史条目并标注迁出，未重新生成全量报告。

> **2026-10-30 模型推理迁出增量**：删除 `model-infer-optimize` 旧副本，Team 减少 1，7 个启用用例由 cannbot 应用仓保留；本仓不再发现该 Team，原子 Skill 数不变。下表保留历史统计，不表示应用仓动态评测已执行。

> **2026-11-04 SHMEM 迁出增量**：删除 `shmem-ops-generator` 旧插件副本，Team 减少 1；该副本没有 `evals/evals.json`，启用用例变化为 0，`community/ops/` 下的 10 个 SHMEM Skill 保留。下表保留历史条目并标注迁出，未重新生成全量报告。

> **2026-11-06 aiss-tiling-solver 归位增量**：该 Skill（含 5 个用例）由 `ops/` 迁至 `community/ops/`，用例随技能整体迁移、总数不变；下表保留历史条目并标注新位置，未重新生成全量报告。

> **2026-11-07 PyPTO 迁出增量**：删除 `pypto-op-orchestrator` 旧插件副本（含 3 个启用用例），Team 减少 1；本仓统计不再包含该 3 个用例。下表保留历史条目并标注迁出，未重新生成全量报告。

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

### ops/（71 Skills，共 526 个用例）

| 名称 | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| aiss-tiling-solver（2026-11-06 已迁至 community/ops/） | 5 |   |   | √ |   | √ |
| ascendc-api-best-practices | 10 | √ |   | √ |   | √ |
| ascendc-api-docs-gen | 4 |   |   | √ |   | √ |
| ascendc-blaze-best-practice | 15 |   |   | √ |   | √ |
| ascendc-blaze-migration | 10 |   |   | √ | √ | √ |
| ascendc-code-review | 8 | √ |   | √ |   | √ |
| ascendc-crash-debug | 7 | √ |   | √ |   | √ |
| ascendc-direct-invoke-template | 7 | √ |   | √ |   | √ |
| ascendc-direct-invoke-to-registry-invoke | 8 |   |   | √ |   | √ |
| ascendc-docs-gen | 11 |   |   | √ |   | √ |
| ascendc-docs-search | 9 | √ |   | √ |   | √ |
| ascendc-env-check | 11 | √ |   | √ |   | √ |
| ascendc-mc2-best-practice | 20 | √ |   | √ | √ | √ |
| ascendc-perf-optimize | 9 |   |   | √ |   | √ |
| ascendc-performance-best-practices | 8 | √ |   | √ |   | √ |
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
| pypto-golden-generate | 6 |   |   | √ |   | √ |
| pypto-intent-understand | 7 |   |   | √ |   | √ |
| pypto-memory-template | 10 |   |   | √ |   | √ |
| pypto-op-design | 10 | √ |   | √ |   | √ |
| pypto-op-develop | 6 |   |   | √ |   | √ |
| pypto-op-knowledge | 8 |   |   | √ |   | √ |
| pypto-op-perf-panko-manual | 0 |   |   |   |   |   |
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
| torch-ascendc-op-extension | 6 |   |   | √ |   | √ |
| torch-ops-profiler | 2 |   |   | √ |   | √ |
| triton-latency-optimizer | 7 |   |   | √ |   | √ |
| triton-op-coding | 7 |   |   | √ |   | √ |
| triton-op-designer | 7 |   |   | √ |   | √ |
| triton-op-verifier | 7 |   |   | √ |   | √ |
| triton-precision-debug | 10 | √ |   | √ |   | √ |
| triton-simulator-optimizer | 10 |   |   | √ |   | √ |
| triton-task-extractor | 7 |   |   | √ |   | √ |

### community/ops/（1 Skill，共 6 个用例）

| 名称 | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| pypto-pro-kernel | 6 |   |   | √ |   | √ |

### graph/（12 Skills，共 96 个用例）

| 名称 | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| ge-coredump-diagnose | 9 |   |   | √ |   | √ |
| ge-dynamic-shape-diagnose | 7 |   |   | √ |   | √ |
| ge-fusion-pass-skill | 5 | √ |   | √ | √ | √ |
| ge-memory-analysis | 9 |   |   | √ |   | √ |
| ge-stream-log-analysis | 5 | √ |   | √ | √ | √ |
| torch-custom-ops-guide | 10 | √ |   | √ | √ | √ |
| torch-npugraph-ex-compile-error-diagnosis | 8 | √ |   | √ | √ | √ |
| torch-npugraph-ex-dfx-triage | 9 | √ |   | √ | √ | √ |
| torch-npugraph-ex-knowledge | 9 | √ |   | √ | √ | √ |
| torch-npugraph-ex-performance-diagnosis | 9 | √ |   | √ | √ | √ |
| torch-npugraph-ex-runtime-error-diagnosis | 9 | √ |   | √ | √ | √ |
| torch-npugraph-ex-template | 7 |   |   | √ | √ | √ |

### model/（21 Skills，共 215 个用例）

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
| model-recommend-analysis | 4 |   |   | √ | √ | √ |
| model-train-log-visualization | 5 | √ |   | √ | √ | √ |
| model-train-oom-analysis | 6 | √ |   | √ | √ | √ |
| model-train-precision-determinism | 14 | √ | √ | √ | √ | √ |
| model-train-precision-nonfinite | 16 | √ | √ | √ | √ | √ |
| model-train-precision-numerical-mismatch | 18 | √ | √ | √ | √ | √ |

### infra/（5 Skills，共 34 个用例）

| 名称 | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| cannbot-skill-reviewer | 5 |   |   | √ |   | √ |
| gitcode-issue-gen | 7 | √ |   | √ |   | √ |
| gitcode-issue-handler | 1 | √ |   | √ | √ | √ |
| gitcode-pr-handler | 9 |   |   | √ |   | √ |
| gitcode-toolkit | 12 | √ |   | √ | √ | √ |

### runtime/（2 Skills，共 0 个用例）

| 名称 | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| runtime-llt-generator | 0 |   |   |   |   |   |
| runtime_migration | 0 |   |   |   |   |   |

## 3. Team 覆盖率

### plugins-official/（8 Teams，共 33 个用例）

| 名称 | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| catlass-op-generator | 0 |   |   |   |   |   |
| [model-infer-optimize（已迁出）](https://gitcode.com/cann/cannbot/tree/master/plugins-official/model-infer-optimize) | 7 | √ |   | √ |   | √ |
| [ops-direct-invoke（2026-10-23 已迁出）](https://gitcode.com/cann/cannbot/tree/master/plugins-official/ops-direct-invoke) | 0 |   |   |   |   |   |
| [ops-direct-invoke-flash（2026-10-29 已迁出）](https://gitcode.com/cann/cannbot/tree/master/plugins-official/ops-direct-invoke-flash) | 9 | √ |   | √ |   | √ |
| [pypto-op-orchestrator（2026-11-07 已迁出）](https://gitcode.com/cann/cannbot/tree/master/plugins-official/pypto-op-orchestrator) | 3 | √ |   | √ |   | √ |
| ops-registry-invoke | 13 | √ |   | √ | √ | √ |
| tilelang-op-orchestrator | 3 | √ |   | √ |   | √ |
| torch-compile（已迁至 cannbot，见应用仓安装测试） | 1 |   |   | √ |   | √ |
| triton-op-generator | 0 |   |   |   |   |   |

### plugins-community/（8 Teams，共 0 个用例）

| 名称 | 用例数 | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| autoresearch | 0 |   |   |   |   |   |
| cannbot-knowledge | 0 |   |   |   |   |   |
| catlass-dsl-generator | 0 |   |   |   |   |   |
| ops-perf-evolution | 0 |   |   |   |   |   |
| [ops-perf-optimize（2026-10-29 已迁出）](https://gitcode.com/cann/cannbot/tree/master/plugins-official/ops-perf-optimize) | 0 |   |   |   |   |   |
| [shmem-ops-generator（2026-11-04 已迁出）](https://gitcode.com/cann/cannbot/tree/master/plugins-community/shmem-ops-generator) | 0 |   |   |   |   |   |
| tilelang2ascendc-ops-generator | 0 |   |   |   |   |   |
