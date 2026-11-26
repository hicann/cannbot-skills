# PR 检视模式（pr-review）

对 GitCode PR 变更的 `.md` 文档做规范化质检。一份引擎、两类入口：本地三模式（门禁 / 全量 / 自定义）检任意文件清单；本模式自动定位 PR 变更、只报 PR 引入的问题。检视引擎完全复用本地模式，本文件只定义 PR 专属的阶段链。

## 模式判定与执行方式

- **判定**：输入含 PR 编号 / PR 链接且要求检视或检查 PR 文档 → 本模式；否则走本地三模式（见 SKILL.md「模式选择」）。
- **本地检查模式确定**：默认全量（9 项全覆盖）；用户在提示词中显式指定了门禁 / 全量 / 自定义时按指定执行，不再询问。
- **执行方式**：阶段 1~5 全部由 agent 自动执行，无需人工逐步操作；用户仅在三处介入：① 触发时提供 PR 编号 / 链接与（可选）本地模式；② 阶段 1 目标分支候选两可时确认；③ 阶段 5 提交评论前确认与（首次）Token 配置。
- 仅支持 GitCode 平台 PR；非 GitCode → 告知并终止。

## 阶段 1：确认目标分支并切换到 PR 分支

**必须先确认 MR 的目标分支（base），再同步该分支。目标分支不一定是 master，也可能是 9.2.0 等发布分支；基准选错（如用了落后的本地 master）会把已合入其他分支的提交误算进 PR 范围。** 确认方法按优先级：

1. 查 MR 信息：`GET https://api.gitcode.com/api/v5/repos/<owner>/<repo>/pulls/<pr_number>` 返回的 `base.ref` 字段（同时记录 `base.sha`、`head.sha`、变更文件数，供阶段 2 基线核验）
2. 本地启发式（无法查 API 时）：先 fetch MR head，再对候选分支（`git ls-remote --heads upstream` 列出的 master 与各发布分支）逐一执行 `git merge-base pr_<pr_number> remotes/upstream/<分支>`——正确目标分支的 merge-base 紧贴该分支 tip，且 `git log remotes/upstream/<分支>..pr_<pr_number> --oneline` 只包含 PR 自身提交；两个候选都满足时须询问用户，不得默认 master

merge-requests ref 只存在于上游仓，本地 origin 若是 fork 则取不到，统一通过 upstream 远端获取：

```bash
git remote add upstream https://gitcode.com/<owner>/<repo>.git 2>/dev/null
git fetch upstream <目标分支>
git fetch upstream refs/merge-requests/<pr_number>/head:pr_<pr_number>
git checkout pr_<pr_number>
```

**工作区保护**：目标仓检出在其他分支或有未提交变更时，改用隔离 worktree 替代 checkout（`git worktree add --detach <临时目录>/<仓名> <head_sha>`，仓名目录名保持与真实仓名一致，后续所有命令以该目录为项目根），检视结束后 `git worktree remove` 清理，不污染用户工作区。

## 阶段 2：获取变更清单

```bash
# 三个点(...) + 基于 upstream/<目标分支>；本地 master 可能落后，禁止用本地分支做基准
git diff remotes/upstream/<目标分支>...pr_<pr_number> --name-only -- '*.md'
git diff remotes/upstream/<目标分支>...pr_<pr_number> -U0 -- '*.md' > <项目根>/output/temp/pr_changes.diff
```

无 `.md` 变更则报告"PR 无文档变更"并结束。

**基线核验（API 元数据可用时必做）**：本地 diff 文件总数应与阶段 1 记录的变更文件数一致；不一致说明本地基准漂移（目标分支在拉取后有新合入），改用 API 权威口径（`GET .../pulls/<pr_number>/files`）作为变更清单，禁止据本地 diff 开检。

**范围 sanity check（必做）**：变更范围须与 PR 主题相称。若小型修改类 PR 却出现大量提交/文件（如文档澄清却改了几百个文件），通常是目标分支判断错误（本地分支落后、误用本地分支或 fork origin 做基准），须回到阶段 1 复核后再继续，不得直接开检。

## 阶段 3：本地检查 + 红线过滤（只检视 PR 引入的问题）

1. Read 所选本地模式的 workflow 文件并执行，仅两处适配：检查目标 = 阶段 2 的变更 `.md` 清单；目录展开跳过。其余步骤（环境前置、并发编排、完整性校验、聚合、失败隔离）原样执行，产物路径规则不变。
2. **红线过滤（确定性脚本，必做）**：

```bash
python3 <skill_base>/scripts/pr_review_filter.py filter \
  --diff "<项目根>/output/temp/pr_changes.diff" \
  --result "<产物目录>/docs-common-review-result.json" \
  --output "<项目根>/output/temp/pr-filtered-issues.json" \
  --comments "<项目根>/output/temp/pr-comments.json"
```

   过滤规则：问题行号落在该文件变更行 ±3 行（`--context` 可调）→ 保留；行号在变更范围之外 → 存量问题剔除；无行号的文件级问题在文件属 PR 变更时保留。保留问题中行号**严格落在变更 hunk 内**的渲染为行内评论（`"inline": true`），其余仅入报告。
   `python3` 不可用或脚本失败 → 明示"未做红线过滤"，报告退化为本地检查全量结果，禁止编造过滤统计。

**只检视 PR 变更引入的问题**：目标分支上已存在的问题不重复报告（过滤已保证）；与全仓既有惯例一致的表述不算问题。

## 阶段 4：输出检视报告

按 11 字段问题契约输出（file_name / line_number / rule_group / rule_item / rule_name / rule_level / description / suggestion 等），格式：

```markdown
## 文档检视报告: PR #<N> - [PR标题]

### 概览
| 检查项 | 严重 | 一般 | 提示 |

### 问题列表
**[1]** [严重] [通用规则] [punctuation-check] [成对标点符号不完整] docs/zh/api/xxx.md:55
- 问题：...
- 建议：...
```

- 问题条目首行的四个中括号分别为 级别(`rule_level`)、问题大类(`rule_group`)、规则项名称(`rule_item`)、规则名称(`rule_name`)，与问题契约字段名一一对应；问题大类或规则名称为空时省略对应中括号，便于直接看出具体规则

- 概览表下方附合计：保留 N 条（严重 x / 一般 y / 提示 z），剔除存量 M 条；存量只呈现计数，不逐条列出
- summary.failed_check 非空时，先输出检查项失败提示块（见 SKILL.md「检查项失败提示」节）再输出常规汇报

## 阶段 5：评论提交（可选，需用户确认）

生成报告后询问用户需提交哪些意见（提交全部 / 提交部分指定编号 / 不提交）。**严禁未经用户确认自动提交。** 存在疑似重复或语义待确认的意见时单列展示（附已有评论与判重理由），默认不提交。

**Token 配置（提交评论的前提）**：Token 存于本地配置文件，由脚本读取，全程不回显、不打印、不暴露给上下文：

- 默认路径：`~/.config/opencode/skills/docs-common-review/config/gitcode-config.json`（或经 `--config` 参数指定）
- 文件格式：`{"token": "<GitCode 私人令牌>", "baseUrl": "https://api.gitcode.com/api/v5"}`（`baseUrl` 可省略，缺省即此值）
- 亦可 `export GITCODE_TOKEN=<令牌>` 代替配置文件
- **配置缺失时停止提交**，指引用户自行创建（令牌在 GitCode「设置 → 私人令牌」生成，需具备 PR 评论写权限），待用户完成后再继续

**评论正文格式**（filter 脚本已按此渲染）：

- 首行：`[级别] [问题大类] [规则项名称] [规则名称]`——四个中括号分别取 `rule_level` / `rule_group` / `rule_item` / `rule_name`，与问题契约字段名一一对应，行内评论锚定在对应行上，正文**不重复文件与行号**（问题大类或规则名称为空时省略对应中括号；如 `[严重] [通用规则] [punctuation-check] [中英文符号混用]`）。**例外**：降级为普通评论（无行内锚定）或文件级问题时，首行保留 `文件(:行号)`，位置信息自含
- 正文仅保留 **问题** 与 **建议** 两段：首行已含级别、检查项、文件与行号，正文不重复这些信息。评论须自明——「问题」写清现象与违反的要求，「建议」给出明确、可直接落地的修改

**行内评论字段**：GitCode 接受 GitHub 风格的三个扁平字段——`path`（文件路径，必填）、`position`（**PR head 侧绝对行号**，直接取报告中的 `文件:行号`，无需 diff hunk 偏移反算）、`side`（默认 `"RIGHT"`）。POST 只接受扁平字段，勿写嵌套 `position` 对象（会被 400 拒绝）。

**提交方法**（未安装 gitcode-toolkit 时；已安装则优先委托其 `references/pr-inline-comment-workflow.md` 与配套脚本，token 遵循其 token-config 约定，且须同样执行本节去重与逐条复查要求）：

```bash
python3 <skill_base>/scripts/pr_review_filter.py submit \
  --comments "<项目根>/output/temp/pr-comments.json" \
  --repo <owner>/<repo> --pr <pr_number> \
  [--expect-sha <阶段1记录的head_sha>] [--config <token配置.json>]
```

**流程：先去重、smoke test 再逐条复查**（脚本已内置，语义与 ascendc-code-review 对齐）：

0. **提交前去重（必做，硬门禁）**：
   - 先 GET PR 核对 head SHA：`--expect-sha` 传入阶段 1 记录值，不一致 → 中止提交；未提供则取当前值作基准，提交过程中逐条复核
   - 分页拉取 PR 全部已有评论；**拉取失败或 PR 信息无法获取 → 中止提交，不假定没有重复**（禁止跳过去重直接提交）
   - **确定性去重**（脚本）：`file`+`line` 命中已有行内评论、正文指纹（去空白）相同、或同文件落点距离 ≤5 行（`--proximity` 可调，0 关闭）且已有评论正文含本条规则标识（检查项/规则名，同一风险的确定性近似）→ SKIP 并计数
   - **语义去重**（agent）：对不含规则标识的已有评论（如人工评论），agent 须比对正文与对应代码——同文件、落点 ≤5 行且描述同一风险/根因 → 从评论清单剔除并记录判重理由；措辞、规则名不同仍算重复，仅问题类别相同但涉及不同操作或不同风险不算
   - filter 阶段已做清单内部去重（11 字段全等的问题只保留首条）
1. 先 POST 1 条。响应中数字型 `id` 即 `note_id`（后续 GET/DELETE 均用它作路径参数）
2. `GET /repos/<owner>/<repo>/pulls/comments/<note_id>` 验证返回的 `position.new_line` 与目标行号一致——这是 GitCode 文档不完整情况下唯一可信的对错验证
3. 错位则立即 `DELETE` 撤回（返回 204 即成功）并**中止批量**，修正后重跑；不要先批量提交再发现全错位
4. 验证通过后**逐条串行**提交剩余意见；**每条发送前重新核对 head SHA 并刷新已有评论重新判重**——版本已变化 → 停止剩余提交（退出码 5）；刷新失败 → 中止（退出码 2）；判重命中 → SKIP 该条继续；请求超时或回查失败先查询确认，不盲目重试 POST
5. **防重发**：发布记录默认落在评论清单同目录 `<pr>_inline_comment_results.json`（`--record` 可指定），成功项实时写入；重跑先回查记录，已成功项直接 SKIP；最后输出提交结果清单（成功 / 降级 / 失败 / 重复跳过计数，含逐条复查阶段跳过数）
6. 兜底：某文件行内评论 POST 返回 400（无法锚定 position）时，脚本自动降级为不带 `path`/`position` 的普通评论（文件与行号已在正文首行）

## 收尾

与所选本地模式一致（SKILL.md「通用收尾」）：询问是否导出 csv/md → 按选择重跑 aggregate → 清理中间产物（必做，额外包含 `temp/pr_changes.diff`、`temp/pr-filtered-issues.json`、`temp/pr-comments.json` 与发布记录 `<pr>_inline_comment_results.json`）→ 使用隔离 worktree 时一并 `git worktree remove`。

## 约束

- 仅支持 GitCode PR；基准分支必须先确定再切换，候选两可时必须询问用户，禁止默认 master
- 基线核验与范围 sanity check 必做；基准可疑时回退阶段 1 复核，不得直接开检
- 报告只报 PR 引入的问题（以红线过滤结果为准），存量问题仅呈现计数
- 评论提交必须经用户确认，默认不提交；token 只经配置文件 / 环境变量传递，禁止进入对话上下文、报告或命令明文
- 本 skill 自包含：评论提交不依赖其他检视 skill，gitcode-toolkit 仅为可选加速
- 本地检查步骤不得偏离所选本地模式 workflow；失败隔离、计时、零编造等执行原则照常适用
