---
name: gitcode-toolkit
description: GitCode 协作通用基础参考（内部参考，不直接触发）。提供 GitCode API、Token 配置、URL 解析、日志规范、变更展示，Git 克隆/分支/diff/log/remote 通用操作，以及 PR 创建工作流（API/模板/head 格式等）等共享文档。供 gitcode-pr-handler、gitcode-issue-gen、gitcode-issue-handler 等 GitCode 协作类 skill 通过相对路径引用本目录下的 references/ 与本文档章节使用，本 skill 自身不响应用户触发。
disable-model-invocation: true
license: MIT
---

# GitCode Toolkit

GitCode 协作类 skill 的共享基础文档集合。

> **定位**：内部参考，不直接触发。其他 GitCode skill（`gitcode-pr-handler`、`gitcode-issue-gen`、`gitcode-issue-handler` 等）按需引用本目录下的 `references/*.md` 与本文档「PR 创建工作流」章节，无需在各自 skill 内重复实现。

---

## 速查表

### 环境预检（Step 0，所有 GitCode skill 必经）

按顺序检查：token / git / curl / `/tmp` / 输出目录。任一失败立即 AskUserQuestion 询问（一次只问一个）。详见 [references/env-check.md](references/env-check.md)。

### Token 配置

按优先级获取：1) 用户请求中直接提供 → 2) 环境变量 `GITCODE_TOKEN` → 3) 询问用户。详见 [references/token-config.md](references/token-config.md)。

### URL 格式

| 类型 | 格式 |
|------|------|
| PR | `/pull/{n}`, `/pulls/{n}`, `/merge_requests/{n}` |
| Issue | `/issues/{n}` |

详见 [references/url-parsing.md](references/url-parsing.md)。

### Git 操作核心命令

```bash
# 克隆（depth=500）
git clone --depth=500 https://gitcode.com/{owner}/{repo}.git /tmp/{prefix}_{owner}_{repo}_{timestamp}

# 检出 PR 分支
git fetch origin +refs/merge-requests/{pr_number}/head:pr_{pr_number}
git checkout pr_{pr_number}

# Merge-base
git fetch origin {base_ref}:base_branch
MERGE_BASE=$(git merge-base base_branch pr_{pr_number})

# Diff（merge-base 模式：code-review、gitcode-pr-handler、gitcode-issue-gen）
git diff --numstat $MERGE_BASE pr_{pr_number}
git diff --name-status $MERGE_BASE pr_{pr_number}

# Diff（triple-dot 模式：pr-to-design-doc、PR 创建流程）
git diff --numstat "origin/${BASE_BRANCH}...HEAD"

# Log / Show
git log -1 --pretty=format:"%s"
git log --oneline "origin/${BASE_BRANCH}..HEAD"
git show HEAD:path/to/file

# Remote / Push
git remote -v
git ls-remote --heads origin ${branch}
git push -u origin ${branch}
```

更详细的操作手册见 references/clone-and-checkout.md、references/diff-and-changes.md、references/log-and-show.md、references/remote-and-branch.md、references/pitfalls.md。

---

## PR 创建工作流

从 fork 仓库向上游 `cann` 组织仓库创建 Pull Request 的标准步骤，作为 `gitcode-issue-handler` 等 skill 的 PR 创建子流程被引用。

```
1. 获取信息   → 分支名、commit历史、目标仓库
2. 获取模板   → 从目标仓库获取 PR 模板
3. 分析填充   → 分析 commit 内容，自动填充模板
4. 用户确认   → 展示填充后的模板，等待用户确认/修改
5. 推送分支   → 确保分支已推送到 origin
6. 创建 PR    → 调用 GitCode API 创建 PR
7. 记录日志   → 保存操作日志
```

### Step 1: 获取信息

**必需参数**

| 参数 | 说明 | 获取方式 |
|------|------|----------|
| 分支名 | 源分支名称 | 从当前 git 分支获取或用户指定 |
| commit 历史 | 用于分析生成 PR 内容 | git log 获取 |
| 变更文件列表 | 用于推断模板字段 | git diff 获取 |

**默认配置**

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| 上游仓库 | `cann` | 目标组织名称 |
| 目标分支 | `master` | 上游仓库的目标分支 |
| 用户仓库 | 当前 git 配置 | 从 git remote 获取 |

**1.1 检测 Remote 配置**

```bash
git remote -v
```

自动识别逻辑：
- 上游仓库：URL 中包含 `cann/` 的 remote
- Fork 仓库：其他 remote（非 cann 组织）

**1.2 如果无法自动识别**：用 AskUserQuestion 让用户选择哪个是 fork 仓库。

**1.3 获取当前信息**

```bash
current_branch=$(git branch --show-current)
username=$(git remote get-url ${fork_remote} | sed -E 's|.*[:/]([^/]+)/[^/]+\.git|\1|')
repo=$(git remote get-url ${fork_remote} | sed -E 's|.*[:/][^/]+/([^/]+)\.git|\1|')

git log master..HEAD --pretty=format:"%s" --no-merges
git diff master...HEAD --name-only
git log master..HEAD --pretty=format:"%s%n%b" --no-merges
```

### Step 2: 获取 PR 模板

模板文件按优先级：
- `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`
- `.gitcode/PULL_REQUEST_TEMPLATE.md`
- `PULL_REQUEST_TEMPLATE.md`

```bash
git show origin/master:.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md
git show origin/master:.gitcode/PULL_REQUEST_TEMPLATE.md
```

**默认模板**（仓库无模板时使用）：

```markdown
## 描述
<!--详细描述改动-->

## 关联的Issue
<!--Issue链接或问题单单号-->

## 测试
<!--测试验证内容-->

## 文档更新
<!--文档更新说明-->

## 类型标签
- [ ] Bug修复
- [ ] 新特性
- [ ] 性能优化
- [ ] 文档更新
- [ ] 其他
```

### Step 3: 分析并填充模板

**信息来源映射**

| 模板字段 | 自动获取方式 | 备选方案 |
|----------|--------------|----------|
| **描述** | 从 commit messages 汇总生成 | 用户输入 |
| **关联的Issue** | 从 commit message 提取 `#数字` 或 `fix #数字` | 用户输入 |
| **测试** | 检测 tests/ 目录变更，提示用户填写 | 用户输入 |
| **文档更新** | 检测 docs/、README.md 等文件变更 | 用户输入 |
| **类型标签** | 从 PR 标题前缀推断 | 用户选择 |

**类型标签推断规则**

| 标题前缀 | 类型标签 |
|----------|----------|
| `fix:` | Bug修复 |
| `feat:` | 新特性 |
| `perf:` | 性能优化 |
| `docs:` | 文档更新 |
| `refactor:` / `test:` / `chore:` | 其他 |

**分析脚本要点**

```bash
commits=$(git log master..HEAD --pretty=format:"%s" --no-merges)
issues=$(git log master..HEAD --pretty=format:"%s %b" --no-merges | grep -oE '#[0-9]+' | sort -u)
test_files=$(git diff master...HEAD --name-only | grep -E '(tests?/|_test\.|_spec\.)')
doc_files=$(git diff master...HEAD --name-only | grep -E '(docs?/|README|\.md$)')
first_commit=$(git log master..HEAD --pretty=format:"%s" --no-merges | head -1)
```

### Step 4: 用户确认

用 AskUserQuestion 展示填充后的模板预览，选项：
1. **确认创建** - 使用当前模板内容创建 PR
2. **修改模板** - 用户手动编辑
3. **取消操作** - 终止流程

确认时展示：PR 标题、源分支 → 目标分支、填充后的模板内容。

### Step 5: 推送分支

```bash
git push -u origin ${branch_name}
git ls-remote --heads origin ${branch_name}
```

### Step 6: 创建 PR

**API**

```
POST https://api.gitcode.com/api/v5/repos/{upstream_owner}/{upstream_repo}/pulls
```

| 参数 | 类型 | 说明 |
|------|------|------|
| access_token | string | GitCode API Token |
| title | string | PR 标题 |
| body | string | PR 描述内容（填充后的模板） |
| head | string | 源分支，格式: `{username}:{branch}` |
| base | string | 目标分支，通常为 `master` |

```bash
curl -X POST "https://api.gitcode.com/api/v5/repos/${upstream_owner}/${upstream_repo}/pulls" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "access_token=${token}" \
  -d "title=${pr_title}" \
  -d "body=${pr_body}" \
  -d "head=${username}:${branch_name}" \
  -d "base=master" \
  --connect-timeout 30
```

**head 参数格式**：从 fork 仓库向上游创建 PR 时，`head` 必须是 `{fork用户名}:{分支名}`，例如 `your-username:fix/xxx`。当 fork 改过名时建议用更稳的 `{fork_owner}/{fork_repo}:{branch}` 格式。

**成功响应 (HTTP 201)**：

```json
{
  "id": 8395063,
  "iid": 1564,
  "title": "fix: 修复异构安装路径问题",
  "state": "opened",
  "web_url": "https://gitcode.com/cann/ops-math/merge_requests/1564",
  "source_branch": "fix/heterogeneous-install-path",
  "target_branch": "master"
}
```

> 错误码处理详见 [references/gitcode-api.md](references/gitcode-api.md)。

### Step 7: 记录日志

日志文件命名：`logs/pr-create_{YYYYMMDD}_{HHMMSS}.log`。日志格式详见 [references/logging-conventions.md](references/logging-conventions.md)。

### 常见问题

**Q1: PR 创建失败，提示 "head not found"**：分支未推送到 origin，先 `git push -u origin ${branch_name}`。

**Q2: PR 创建失败，提示 "Another open merge request already exists"**：该分支已有未合并 PR，从 API 返回里取已有 PR 链接。

**Q3: 模板获取失败**：仓库无模板时退回默认模板（见 Step 2）。

**Q4: 查看已有 PR**：

```bash
curl "https://api.gitcode.com/api/v5/repos/${upstream_owner}/${upstream_repo}/pulls?state=opened&source_branch=${branch_name}&access_token=${token}"
```

---

## 参考文档索引

| 文档 | 说明 | 适用 skill |
|------|------|-----------|
| [references/env-check.md](references/env-check.md) | Step 0 环境预检（token / git / 临时目录 / 输出目录） | 所有 GitCode skill |
| [references/gitcode-api.md](references/gitcode-api.md) | PR/Issue/仓库 API + 错误处理 + 命令速查 | code-review, gitcode-pr-handler, gitcode-issue-gen, pr-compile, gitcode-issue-handler |
| [references/url-parsing.md](references/url-parsing.md) | URL 格式识别与解析（PR/Issue） | code-review, gitcode-pr-handler, gitcode-issue-gen, pr-to-design-doc, pr-compile, gitcode-issue-handler |
| [references/token-config.md](references/token-config.md) | Token 获取优先级 | 所有 GitCode skill |
| [references/logging-conventions.md](references/logging-conventions.md) | 日志命名与记录规范 | 所有 GitCode skill |
| [references/change-table-display.md](references/change-table-display.md) | 变更文件列表展示格式 | code-review, gitcode-pr-handler, gitcode-issue-gen, pr-to-design-doc |
| [references/clone-and-checkout.md](references/clone-and-checkout.md) | 克隆、浅克隆、PR 分支检出、base 分支确定、merge-base | code-review, gitcode-pr-handler, gitcode-issue-gen, pr-to-design-doc |
| [references/diff-and-changes.md](references/diff-and-changes.md) | diff 变更统计（merge-base 模式 + triple-dot 模式） | code-review, gitcode-pr-handler, gitcode-issue-gen, pr-to-design-doc, PR 创建流程 |
| [references/log-and-show.md](references/log-and-show.md) | git log 元信息提取、git show 文件读取 | code-review, gitcode-pr-handler, gitcode-issue-gen, pr-to-design-doc, PR 创建流程 |
| [references/remote-and-branch.md](references/remote-and-branch.md) | remote 管理、分支查询、push、ls-remote | PR 创建流程 |
| [references/pitfalls.md](references/pitfalls.md) | Git 操作易错点对照表 | 所有使用 git 的 skill |
