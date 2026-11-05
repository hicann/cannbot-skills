# 知识库升级入口（维护者）

本文件只负责启动，不承载流程细节。适用于 **catlass-dsl-generator 插件维护者** 在
`ascend-catlass-dsl` 发布新版本（如 v2.3）后更新知识库与版本基线。

- 不适用于算子设计、实现、调试、benchmark 或优化——那些走 `catlass-dsl-workflow`。
- 本 Skill 与算子工作流**互斥**：执行升级时不要加载 `catlass-dsl-workflow`。
- `init.sh` 不安装本 Skill（避免进入客户工程），维护者在插件仓内直接使用。
- 本版本不提供 `upgrade_state.py`、`gen_api_matrix.py`、`check_knowledge.py` 等脚本；
  矩阵生成与校验脚本由维护者在升级工作区自行编写并记录命令，产物与判据见下。

## 第一步

完整加载 `skills/catlass-dsl-knowledge-upgrade/SKILL.md`，建立升级工作区
`ITERATIONS.md`，然后输出升级卡：

```text
📋 升级卡
版本：<from> → <to> ｜ catlass：<root>@<ref|worktree>
阶段：<K0-K5>
基线：功能/语义=<tag 或已安装 wheel> ｜ 参考路径=<master>
```

## 五条底线

- **双基线**：能力与语义以发布 tag / 已安装 wheel 为准；文件路径以 master 为准；冲突以
  wheel 为准。
- **矩阵由脚本生成**：API 与样例清单只能来自可复现的生成脚本，禁止手写或凭记忆。
- **不确定进 review**：无法判定的条目必须列出，不允许猜。
- **先预览后写**：批量改知识库前先出改动清单。
- **同 PR**：知识库、manifest、测试必须一起提交。

## 阶段产物

| 阶段 | 必需产物 |
|---|---|
| K0 预检 | `ITERATIONS.md` 中的版本、checkout、wheel 记录 |
| K1 矩阵 | `api_matrix_from.json`、`api_matrix_to.json`、`api_diff.json`、`api_diff.md` |
| K2 分类 | `classification.md`，不确定项进入 review |
| K3 迁移 | 知识库改动 + `knowledge_changes.md` + manifest |
| K4 验证 | `check-report.md` + 插件 pytest 结果 |
| K5 交付 | `upgrade-report.md` + `pr-checklist.md` |

- 逐阶段判据：`skills/catlass-dsl-knowledge-upgrade/references/gates.md`
- 完整发布顺序与双基线细节：`skills/catlass-dsl-knowledge-upgrade/references/release-runbook.md`
