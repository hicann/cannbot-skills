---
name: catlass-dsl-knowledge-upgrade
description: 维护 catlass-dsl-generator 知识库与版本基线。当用户提到 catlass 新版本发布、v2.3、ascend-catlass-dsl 升级、知识库更新、API 矩阵、样例路径失效或知识库漂移时使用；本版本不使用状态机脚本，按文本清单推进。
---

# CATLASS DSL 知识库升级（维护者，清单引导）

本 Skill 只服务**插件维护者**，与算子工作流互斥：**禁止加载 `catlass-dsl-workflow`**，
也不得用它替代 design/develop/bench/optimize。任务对象是 `knowledge/` 与升级产物中的版本
manifest，不是用户算子。仓库不预置任何版本的 manifest：基线 manifest 由 K1 生成的 API
矩阵产出，未完成迁移前不代表知识库已支持该版本。

**唯一目标**：DSL 发布新版本时，用可复现的证据把知识库、manifest 与测试对齐到新版本，
并留下可复核的升级报告。

> **自动执行**：遇到问题只要有可行方案就逐步尝试，一个方案失败就换下一个，不停下来找
> 用户确认。等待长跑任务时直接等待并轮询，不结束回合。

> **超时纪律**：编译、运行与普通 shell 命令拆成独立命令块；编译单独成块、不套 60s，但必须
> 带明确上限（默认 `timeout 600s`）；单例运行一条命令只跑一个 case/shape，加
> `timeout -k 5 60s`；批量按 2~5 片拆分，分片命令用分片级超时（用例数 × 60s + 余量），
> 片内逐例 60s 硬超时，禁止整批套一次性 60s 或裸跑。
> 超时或卡住立即终止。
> 远端/容器调试时，`timeout` 必须写在最终执行环境的命令内，外层只负责连接。

## 1. 启动（每个新任务第一条回复前）

1. 确定目标版本 `X.Y.Z` 与基线版本。
2. 确认 catlass checkout 路径（可以是客户 master，也可以是任意 tag 的 worktree）。
3. 在升级工作区建立 `ITERATIONS.md`，输出升级卡：

```text
📋 升级卡
版本：<from> → <to> ｜ catlass：<root>@<ref|worktree>
阶段：<K0-K5>
基线：功能/语义=<tag 或已安装 wheel> ｜ 参考路径=<master>
下一步：<本阶段第一个动作>
```

## 2. 硬规则（不可豁免）

- **R0 双基线**：功能、API、语义以**已发布 tag 或已安装 wheel** 为准；文件路径、目录结构
  以**客户 master** 为准。两者冲突时**以 wheel 为准**，并在报告中记录冲突条目。
- **R1 矩阵必须由脚本生成**：API 与样例清单只能由可复现的生成脚本产出，禁止凭记忆、
  凭旧文档或手写清单。脚本可以写在升级工作区（不属于插件），但必须记录命令与版本。
- **R2 不确定就进 review**：脚本无法判定的条目（签名变化是否影响语义、样例是否等价）
  必须写入 classification 的 `review` 列表，禁止猜测后直接改知识库。
- **R3 先预览后写**：批量修改知识库前先输出改动清单与 diff，确认后才落盘；禁止一次性
  重写多个章节而不留对照。
- **R4 版本写法统一**：pip 版本写 `2.3.0`，tag 写 `v2.3.0`；禁止 `2.3`、`v2.3` 这类
  混写。
- **R5 同 PR 交付**：知识库改动、manifest、测试更新必须在同一个 PR；manifest 未更新视为
  升级未完成。
- **R6 不碰算子工作流产物**：不修改 `workspace/`、`.catlass-dsl/develop-runs/`、
  `optimize-runs/`，不改 DESIGN、reference、workload。

## 3. 阶段清单

| 阶段 | 动作 | 完成判据 |
|---|---|---|
| K0 预检 | 确认版本、checkout、当前分支、wheel 版本 | 版本合法、`core_api.py` 存在，记录在 `ITERATIONS.md` |
| K1 矩阵 | 生成 from/to 矩阵与 diff | 三个产物存在：`api_matrix_from.json`、`api_matrix_to.json`、`api_diff.json` |
| K2 分类 | 把 diff 分成 API/样例/语义/文档四类，逐条给动作 | `classification.md` 存在，不确定项全部进 review |
| K3 迁移 | 按分类改知识库、更新 manifest | `knowledge_changes.md` 列出每个文件与理由 |
| K4 验证 | 人工检查知识库 + 插件 pytest | `check-report.md` 状态为 passed，pytest 全绿 |
| K5 交付 | 升级报告 + PR checklist | `upgrade-report.md`、`pr-checklist.md` 存在 |

阶段顺序不可跳过；每阶段完成后在 `ITERATIONS.md` 追加一行（阶段、命令、结果、产物）。
脚本报错先处理错误，不得手工伪造产物。

## 4. 各阶段做法

### K0 预检

- 确认目标版本与基线版本，记录 wheel 版本、tag、checkout 路径和当前分支；
- 确认目标 ref 包含 DSL 包（存在 `core_api.py`）。

### K1 矩阵

- 从基线 ref 生成 `api_matrix_from.json`，从目标 ref 生成 `api_matrix_to.json`；
- 两者都必须记录 ref、版本、commit 和生成命令；
- 生成 diff 并同时产出 `api_diff.md` 供人工阅读；
- 生成目标版本 manifest 草案（`api_digest`、`commit`、`allowed_commits`）。

脚本自行编写，放在升级工作区，不提交进插件；命令与 commit 必须可复现。

### K2 分类

把 diff 分成四类并逐条给出动作：

- `api_added` / `api_removed` / `api_signature_changed`：是否影响知识库用法；
- 样例变化：路径是否失效、是否有等价替代；
- 语义变化：是否需要更新约束或数值口径；
- 文档变化：是否需要改描述。

无法判定的条目写入 `review`，不得猜测。

### K3 迁移

- 按分类修改 `knowledge/`，每个文件在 `knowledge_changes.md` 写明改了什么和依据；
- 更新 `dsl_version` 与 manifest；
- 新样例路径必须来自实际解析结果，不写“看起来像”的路径。

### K4 验证

人工检查（本版本无脚本）：

- 每个 concept 的 frontmatter 完整，`sources` 可追溯；
- `index.md` 链接形式正确，无失效路径；
- 知识库版本与 manifest 一致；
- 运行插件 pytest，记录命令和结果。

### K5 交付

- `upgrade-report.md`：版本对、矩阵摘要、知识库改动清单、验证结果、未覆盖项与 review；
- `pr-checklist.md`：按 [`templates/PR_CHECKLIST.md`](templates/PR_CHECKLIST.md) 填写；
- 知识库、manifest、测试在同一 PR 提交。

## 5. 失败处理

- 找不到 `core_api.py`：先确认根目录是仓根而不是子目录，再确认该 ref 确实包含 DSL 包。
- 样例名解析不到：列出实际存在的样例 id 后再更新知识库，不要改写成猜测路径。
- 知识库或 manifest 版本不一致：判断哪一侧过期，两者都修到一致，不允许单边放宽。
- 环境没有 wheel：记录 `unavailable`，不得写成 `aligned`。

## 6. 输出

每次任务结束必须给出：版本对（from→to）、矩阵摘要（API/样例增减）、知识库改动文件清单、
验证命令与结果、未覆盖项与 review 清单、下一步建议。禁止只给结论不给证据。

详细发布顺序见 `references/release-runbook.md`，判据说明见 `references/gates.md`。
