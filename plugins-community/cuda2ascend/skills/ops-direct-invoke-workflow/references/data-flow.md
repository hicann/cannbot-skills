# 数据流

> 各阶段的文件输入/输出与 `.cannbot` 产物清单。编号与 [SKILL.md 统一流程表](../SKILL.md#统一流程表) 及 [task-prompts.md](task-prompts.md) 对齐。

## `.cannbot` 目录布局

所有流程中间产物落 `.cannbot`；最终交付物（算子代码 / test / doc）落各自专业目录，不进 `.cannbot`。`.cannbot` 所有角色均可写，任意文件类型；各交付件的具体路径与命名在任务下发时约定。

```
.cannbot/
├── state.json                    # 工作流状态（见 state-schema.md）
├── LOG.md                        # 开发日志（模板 LOG-开发日志）
├── issues/                       # 问题记录（模板 Issue-问题记录）
├── requirement.md                # 需求文档（1.1）
├── spec.yaml                     # Spec，机器可校验（1.2；规范见 ops-spec-gen）
├── design/
│   ├── test-plan.md              # 测试方案（2.1）
│   └── dev-plan.md               # 开发方案（2.2）
├── reports/                     # 验收报告（QA 产出）
│   ├── cp3-functional.md         # 功能验收报告（CP3）
│   ├── cp4-performance.md        # 性能验收报告（CP4）
│   └── cp5-review.md             # 代码检视报告（CP5）
├── questionnaires/              # 用户确认问卷 json（QA 生成，PM 发送）
│   └── cp0-env.json             # 环境确认问卷（CP0；CP1/CP1' 等需用户确认时同此）
└── summary/
    ├── dev-report.md             # 开发报告（7.1）
    └── experience.md             # 经验总结（7.2）
```

> 文件名为默认约定，子仓可通过 override `workflow-doc-templates` 调整模板但保持交付件语义不变。

## 阶段数据流

| 编号 | 输入 | 产出 | 落盘位置 | 下游消费方 |
|------|------|------|----------|------------|
| 0 | — | 环境信息文档 | `.cannbot/`（developer 写入） | CP0 |
| CP0 | 环境信息文档 | 问卷 json + 确认结论 | `.cannbot/questionnaires/` + state.json | 1.1 |
| 1.1 | 对话上下文、设计约束 | requirement.md | `.cannbot/` | CP1 → 1.2 |
| CP1 | requirement.md | 确认/修改意见（需用户确认时附问卷 json） | state.json（问卷 json 落 questionnaires/） | 1.2 或回退 1.1 |
| 1.2 | requirement.md | spec.yaml | `.cannbot/` | CP1' → 2.1/2.2 |
| CP1' | spec.yaml | 确认/修改意见 | state.json | 2.1/2.2 或回退 1.2 |
| 2.1 | spec.yaml | design/test-plan.md | `.cannbot/design/` | CP2.1 → 3.2 |
| 2.2 | spec.yaml | design/dev-plan.md | `.cannbot/design/` | CP2.2 → 3.1 |
| CP2.1 / CP2.2 / CP2' | 对应方案 | 确认/修改意见 | state.json | 阶段 3 或回退 |
| 3.1 | dev-plan.md、修改要求 | 算子代码 | 代码目录 | CP3 |
| 3.2 | test-plan.md | golden + 用例表 + 性能框架 | test 目录 | CP3 |
| 3.3 | 算子代码、测试代码 | 白盒用例 | test 目录 | CP3 |
| CP3 | 算子代码 + 测试代码 | reports/cp3-functional.md | `.cannbot/reports/` | 4.1 或回退 3.1 |
| 4.1 | 算子代码（CP3 通过后） | 性能数据 | test 目录 / 采集输出 | CP4 |
| CP4 | 算子代码 + 性能数据 | reports/cp4-performance.md | `.cannbot/reports/` | CP5 或回退 3.1 |
| CP5 | 全部变更文件 | reports/cp5-review.md | `.cannbot/reports/` | 阶段 6 或回退 3.1 |
| 6.1 | 算子代码 + 设计文档 | 算子文档 | doc 目录 | 6.2 |
| 6.2 | 全部代码 + 文档 | PR | 远端 | 6.3 |
| 6.3 | PR | CI 报告 | CI 系统 | 6.4/6.5/CP6 |
| 6.4 / 6.5 | CI 报告 / 检视意见 | 修复后代码 | 代码目录 | CP6 |
| CP6 | CI 报告 + PR 状态 | 上库确认 | state.json | 阶段 7 |
| 7.1 / 7.2 | 全部交付物 | dev-report.md / experience.md | `.cannbot/summary/`（developer-doc 写入） | — |

## 真值源与字段所有权

- **spec.yaml** 是 L0 数学契约的唯一真值源：2.1/2.2 的 dtype/shape/容差以 spec 为准，冲突时停止并报告，不从需求正文重新解释后覆盖。
- **需求文档** 承载需求背景、接口自然语言说明、性能目标来源。
- **测试配置**（test_matrix）由 `ascendc-st-design` 管理，不在 spec 内。
