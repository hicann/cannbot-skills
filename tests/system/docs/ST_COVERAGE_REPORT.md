# ST 测试覆盖率报告

当前 ST 框架覆盖 **80 个 Skill + 1 个 Team**，共 **406 个评测用例**（截止 2026-06-22）。

## 1. 五维看护说明

| 维度 | 测试目标 | 判定标志 |
|------|---------|---------|
| **正向看护** | 在多个类似 skill/team 同时存在时，AI 能正确选择目标 skill | `## Config` 中配置 `Distractor skills` + Expectations 中有 `[skill_activated]` |
| **负向看护** | 在边界/无关场景下，AI 不会被误触发 | Expectations 中有 `[not_contains]` |
| **正确性看护** | 黑盒场景验证：AI 回复语义覆盖关键要点 | `## Expected Output` 定义了预期要点 |
| **调用流程看护** | 验证关键工具被调用、关键文件被生成 | Expectations 中有 `[file_exists]`、`[file_list]`、`[file_contains]` 或 `[skill_activated]` |
| **资源消耗看护** | Token 消耗监控，防止资源浪费 | `## Config` 中配置 `Max Tokens` |

> 仅统计**已启用**的用例。仅在已禁用用例中配置的维度视同无覆盖。

## 2. Skill 覆盖率

按域分组统计：

### ops/（57 Skills，共 296 个用例）

| Skill | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
|-------|:-------:|:-------:|:--------:|:----------:|:----------:|
| aiss-tiling-solver |  |  | √ |  | √ |
| ascendc-api-best-practices | √ |  | √ | √ | √ |
| ascendc-blaze-best-practice |  |  | √ |  | √ |
| ascendc-code-review |  |  | √ |  | √ |
| ascendc-crash-debug |  |  | √ |  | √ |
| ascendc-direct-invoke-template |  |  | √ |  | √ |
| ascendc-direct-invoke-to-registry-invoke | √ | √ | √ | √ | √ |
| ascendc-docs-gen | √ |  | √ | √ | √ |
| ascendc-docs-search | √ | √ | √ | √ | √ |
| ascendc-env-check | √ |  | √ | √ | √ |
| ascendc-perf-optimize |  |  | √ |  | √ |
| ascendc-performance-best-practices |  |  | √ |  | √ |
| ascendc-precision-debug |  |  | √ |  | √ |
| ascendc-regbase-best-practice |  |  | √ |  | √ |
| ascendc-registry-invoke-template | √ | √ | √ | √ | √ |
| ascendc-registry-invoke-to-direct-invoke |  |  | √ |  | √ |
| ascendc-runtime-debug |  |  | √ |  | √ |
| ascendc-simt-best-practices |  |  | √ |  | √ |
| ascendc-simt-tiling-design |  |  | √ |  | √ |
| ascendc-st-design |  |  | √ |  | √ |
| ascendc-task-focus |  |  | √ | √ | √ |
| ascendc-tiling-design |  |  | √ |  | √ |
| ascendc-ut-develop | √ | √ | √ | √ | √ |
| ascendc-whitebox-design |  | √ | √ |  | √ |
| cann-env-setup | √ |  | √ | √ | √ |
| catlass-op-design |  |  | √ |  | √ |
| catlass-op-develop |  |  | √ |  | √ |
| catlass-op-perf-tune |  |  | √ |  | √ |
| npu-arch | √ |  | √ | √ | √ |
| ops-precision-standard |  |  | √ |  | √ |
| ops-profiling |  |  | √ |  | √ |
| ops-simulator |  |  | √ |  | √ |
| ops-spec-gen |  |  | √ |  | √ |
| pypto-api-explore |  |  | √ |  | √ |
| pypto-golden-generate |  |  | √ |  | √ |
| pypto-intent-understand |  |  | √ |  | √ |
| pypto-op-design | √ | √ | √ | √ | √ |
| pypto-op-develop |  |  | √ |  | √ |
| pypto-op-perf-tune |  |  | √ |  | √ |
| pypto-precision-compare |  |  | √ |  | √ |
| pypto-precision-debug |  |  | √ |  | √ |
| tilelang-api-best-practices |  |  | √ |  | √ |
| tilelang-env-check |  |  | √ |  | √ |
| tilelang-op-design |  |  | √ |  | √ |
| tilelang-op-develop |  |  | √ |  | √ |
| tilelang-op-test-design |  |  | √ |  | √ |
| tilelang-perf-optimization |  |  | √ |  | √ |
| tilelang-programming-model-guide |  |  | √ |  | √ |
| tilelang-review |  |  | √ |  | √ |
| tilelang-submodule-pull |  |  | √ |  | √ |
| torch-ascendc-op-extension |  |  | √ |  | √ |
| torch-ops-profiler |  |  | √ |  | √ |
| triton-latency-optimizer |  |  | √ |  | √ |
| triton-op-coding |  |  | √ |  | √ |
| triton-op-designer |  |  | √ |  | √ |
| triton-op-verifier |  |  | √ |  | √ |
| triton-task-extractor |  |  | √ |  | √ |

### graph/（7 Skills，共 40 个用例）

| Skill | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
|-------|:-------:|:-------:|:--------:|:----------:|:----------:|
| torch-custom-ops-guide | √ |  | √ | √ | √ |
| torch-npugraph-ex-compile-error-diagnosis | √ |  | √ | √ | √ |
| torch-npugraph-ex-dfx-triage | √ |  | √ | √ | √ |
| torch-npugraph-ex-knowledge | √ |  | √ | √ | √ |
| torch-npugraph-ex-performance-diagnosis | √ |  | √ | √ | √ |
| torch-npugraph-ex-runtime-error-diagnosis | √ |  | √ | √ | √ |
| torch-npugraph-ex-template |  |  | √ | √ | √ |

### model/（12 Skills，共 36 个用例）

| Skill | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
|-------|:-------:|:-------:|:--------:|:----------:|:----------:|
| model-infer-fusion | √ |  | √ | √ | √ |
| model-infer-graph-mode | √ |  | √ | √ | √ |
| model-infer-kvcache | √ |  | √ | √ | √ |
| model-infer-migrator | √ |  | √ | √ | √ |
| model-infer-multi-stream | √ |  | √ | √ | √ |
| model-infer-optimize |  |  | √ |  | √ |
| model-infer-parallel-analysis | √ |  | √ | √ | √ |
| model-infer-parallel-impl | √ |  | √ | √ | √ |
| model-infer-precision-debug | √ |  | √ | √ | √ |
| model-infer-prefetch | √ |  | √ | √ | √ |
| model-infer-runtime-debug | √ |  | √ | √ | √ |
| model-infer-superkernel | √ |  | √ | √ | √ |

### infra/（4 Skills，共 30 个用例）

| Skill | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
|-------|:-------:|:-------:|:--------:|:----------:|:----------:|
| gitcode-issue-gen |  |  | √ |  | √ |
| gitcode-issue-handler | √ | √ | √ | √ | √ |
| gitcode-pr-handler |  |  | √ |  | √ |
| gitcode-toolkit | √ |  | √ | √ | √ |

### ops-direct-invoke Team

| Team | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
|------|:-------:|:-------:|:--------:|:----------:|:----------:|
| ops-direct-invoke |  |  | √ | √ | √ |

## 3. 平台覆盖

所有用例已配置 `Ascend Platform`，支持 `--ascend-platform` 参数在对应平台上执行。

## 4. 更新指南

新增 skill 或 team 的 ST 用例后，同步更新本文档：

1. 在 §2/§3 的表格中追加新行（或更新已有行的维度标记）
2. 更新 §1 开头的统计数据（用例数、日期、平台信息）
