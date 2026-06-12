# ST 测试覆盖率报告

当前 ST 框架覆盖 **11 个 Skill + 1 个 Team**，共 **92 个评测用例**（89 启用 / 3 禁用，截止 2026-06-12）。

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

| Skill | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
|-------|:-------:|:-------:|:--------:|:----------:|:----------:|
| ascendc-direct-invoke-to-registry-invoke | √ | √ | √ | √ | √ |
| ascendc-docs-gen | √ | √ | √ | √ | √ |
| ascendc-env-check | √ | | √ | √ | √ |
| ascendc-registry-invoke-template | √ | √ | √ | | √ |
| ascendc-task-focus | | | √ | √ | √ |
| ascendc-ut-develop | √ | √ | √ | | √ |
| ascendc-whitebox-design | √ | | √ | √ | √ |
| cann-env-setup | √ | | √ | | √ |
| gitcode-issue-gen | | | √ | | √ |
| npu-arch | √ | | √ | | √ |
| pypto-op-design | √ | √ | √ | √ | √ |

## 3. Team 覆盖率

| Team | 正向看护 | 负向看护 | 正确性看护 | 调用流程看护 | 资源消耗看护 |
|------|:-------:|:-------:|:--------:|:----------:|:----------:|
| ops-direct-invoke | | | | | √ |

## 4. 平台覆盖

所有 92 个用例已配置 `Ascend Platform: A2`，支持 `--ascend-platform A2` 在 A2 服务器上执行。

## 5. 更新指南

新增 skill 或 team 的 ST 用例后，同步更新本文档：

1. 在 §2/§3 的表格中追加新行（或更新已有行的维度标记）
2. 更新 §1 开头的统计数据（用例数、日期、平台信息）
