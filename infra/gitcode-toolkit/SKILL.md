---
name: gitcode-toolkit
description: GitCode 协作通用基础参考（内部参考，不直接触发）。提供 GitCode API、Token 配置、URL 解析、日志规范、变更展示，Git 克隆/分支/diff/log/remote 通用操作，以及 PR 创建工作流和 Issue 创建工作流（API/模板/head 格式等）等共享文档。供 gitcode-pr-handler、gitcode-issue-gen、gitcode-issue-handler 等 GitCode 协作类 skill 通过相对路径引用本目录下的 references/ 与本文档章节使用，本 skill 自身不响应用户触发。
disable-model-invocation: true
license: CANN-2.0
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
5. 校验身份   → 校验 git user.name / user.email 已配置
6. 推送分支   → 确保分支已推送到 origin
7. 创建 PR    → 调用 GitCode API 创建 PR
8. 记录日志   → 保存操作日志
```

> **前置校验**：进入 Step 5 推送之前，**必须**确认 git 提交身份（`user.name` / `user.email`）已配置——这是 PR 提交流程的硬性前置条件。缺失时立刻停下来问用户，不要带着空身份往下走。详见下方 Step 5 与 [references/env-check.md](references/env-check.md) 的「Git 提交用户信息」。

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

### Step 5: 校验 git 提交身份

推送前对 git 提交身份做两层校验：**5.1 是否已配置**（硬性，缺失即阻断）和 **5.2 email 是否与 GitCode 账号绑定一致**（建议，不一致仅告警）。`git push` 本身不会因身份缺失或 email 不匹配而失败，但 commit author 是公开字段：缺失/配错人，或 email 没对上账号绑定邮箱，会导致 commit 挂错身份、或在 GitCode 上显示为「未关联用户」，事后难补救。

#### 5.1 校验是否已配置（硬性，缺失即阻断）

```bash
NAME=$(git config user.name 2>/dev/null)
EMAIL=$(git config user.email 2>/dev/null)
if [ -z "$NAME" ] || [ -z "$EMAIL" ]; then
  echo "MISSING: git author identity (user.name / user.email)"
fi
```

- 两项都已配置 → 展示读到的 `Name <email>` 让用户一眼确认是不是本次想用的身份，确认无误后进入 5.2。
- 任一缺失 → **立即停下来用 AskUserQuestion 询问**，不要继续 push：

```
问题: 未检测到 git 提交身份（user.name / user.email），无法安全提交 PR，请提供：
选项:
  - 用我下面提供的 name 和 email（在下一条消息中给出）
  - 已在别处配置好，让我重新读取一次
  - 取消本次操作
```

拿到用户提供的值后**只在当前工作目录写 local 配置**，禁止改全局 `~/.gitconfig`（理由同 env-check.md：用户全局身份可能服务于多个项目，skill 不应擅自覆盖）：

```bash
git -C "$WORK_DIR" config user.name  "$NAME"
git -C "$WORK_DIR" config user.email "$EMAIL"
```

> 完整规则（global 继承、反模式、禁止用 `--author=` / `-c user.name=` inline 绕过）详见 [references/env-check.md](references/env-check.md) 的「5. Git 提交用户信息」。

#### 5.2 校验 email 是否与 GitCode 账号绑定一致（建议，不一致仅告警）

git 里配的 `user.email` 只是本地字符串，只有当它**等于 token 对应 GitCode 账号的某个已绑定邮箱**时，commit 才会关联到该用户主页；否则 PR 仍能提，但 commit 显示为「未关联用户」。用已有的 token 调账号绑定邮箱接口做一次比对：

```bash
# 拉取当前 token 账号的全部绑定邮箱（小写化，去除 state 未确认项可按需保留）
BOUND=$(curl -s "https://api.gitcode.com/api/v5/emails?access_token=${token}" \
  --connect-timeout 20 --max-time 40 \
  | python3 -c "import sys,json; 
try:
  print('\n'.join(e['email'].lower() for e in json.load(sys.stdin)))
except Exception:
  pass")

if [ -z "$BOUND" ]; then
  echo "SKIP: 无法获取账号绑定邮箱（token 缺 user/email scope 或接口不可用），跳过一致性校验"
elif printf '%s\n' "$BOUND" | grep -qxF "$(echo "$EMAIL" | tr 'A-Z' 'a-z')"; then
  echo "OK: git user.email 与 GitCode 账号绑定邮箱一致"
else
  echo "WARN: git user.email ($EMAIL) 不在账号绑定邮箱中，commit 将不会关联到 GitCode 主页"
fi
```

> 备用端点：若只需账号主邮箱，可用 `GET /api/v5/user` 取单个 `email` 字段比对（详见 [references/gitcode-api.md](references/gitcode-api.md#7-用户账号-api)）。

处理策略（**这是建议性校验，绝不硬阻断**）：

- `OK` → 一行通过提示，进入 Step 6。
- `SKIP`（接口不可用 / token 无 scope）→ 打印一行「已跳过 email 绑定校验」，**直接继续**，不打扰用户。
- `WARN`（不一致）→ 用 AskUserQuestion 提示风险，让用户决定，**默认不阻断**：

```
问题: git user.email 与 GitCode 账号绑定邮箱不一致，commit 将显示为「未关联用户」。如何处理？
选项:
  - 继续提交（接受 commit 不关联到我的主页）
  - 我改用账号绑定的邮箱（在下一条消息给出，仅写工作目录 local 配置）
  - 取消本次操作
```

### Step 6: 推送分支

```bash
git push -u origin ${branch_name}
git ls-remote --heads origin ${branch_name}
```

### Step 7: 创建 PR

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

### Step 8: 记录日志

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

## Issue 创建工作流

创建 GitCode Issue 的标准步骤，供其他需要创建issue的工作流/skill 引用。

```
1. 获取信息   → 目标仓库、问题数据
2. 获取模板   → 从目标仓库查询 Issue 模板
3. 选择模板   → （业务层）问题类型→模板映射
4. 填充内容   → （业务层）根据问题内容生成 Issue body
5. 用户确认   → 展示 Issue 内容，等待确认
6. 创建 Issue → 调用 GitCode API 创建 Issue
7. 记录日志   → 保存操作日志
```

> **Step 3、4 为业务层**，由调用方 skill 根据业务场景实现（如扫描报告→模板选择、问题列表→Issue内容）。infra 仅提供 Step 1、2、5、6、7 的通用能力。

### Step 1: 获取信息

**必需参数**

| 参数 | 说明 | 获取方式 |
|------|------|----------|
| 目标仓库 | Issue 创建目标仓库 | 用户指定或从问题数据推断 |
| 问题数据 | 问题列表、扫描报告 | 用户提供 |

**仓库路径格式**

| 格式 | 示例 |
|------|------|
| URL | `https://gitcode.com/cann/ops-math` |
| 项目路径 | `cann/ops-math` |
| 仓库名 | `ops-math`（默认组织为 `cann`） |

**解析命令**

```bash
# 从 URL 提取项目路径
project_path=$(echo "${url}" | sed -E 's|.*gitcode\.com/([^/]+/[^/]+).*|\1|')

# 从仓库名构建项目路径（默认 cann 组织）
project_path="cann/${repo_name}"
```

### Step 2: 获取 Issue 模板

**模板查询优先级**

| 优先级 | 路径 | 说明 |
|:---:|------|------|
| 1 | `.gitcode/ISSUE_TEMPLATE/*.zh-CN.yml` | GitCode 中文模板（优先） |
| 2 | `.gitcode/ISSUE_TEMPLATE/*.yml` | GitCode YAML 表单模板 |
| 3 | `.gitcode/ISSUE_TEMPLATE/*.md` | GitCode Markdown 模板 |
| 4 | `.github/ISSUE_TEMPLATE/*.yml` | GitHub YAML 表单模板（兼容） |
| 5 | `.github/ISSUE_TEMPLATE/*.md` | GitHub Markdown 模板（兼容） |
| 6 | **预设模板** | 仓库无模板时使用 |

**API 查询命令**

```bash
owner="cann"
repo="ops-math"
token="${GITCODE_TOKEN}"

# 查询模板目录列表（.gitcode）
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/.gitcode/ISSUE_TEMPLATE?access_token=${token}"

# 查询模板目录列表（.github）
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/.github/ISSUE_TEMPLATE?access_token=${token}"

# 获取单个模板内容（返回 Base64 编码）
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/.gitcode/ISSUE_TEMPLATE/bug_report.yml?access_token=${token}"

# 解码模板内容
content=$(curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/.gitcode/ISSUE_TEMPLATE/bug_report.yml?access_token=${token}" | jq -r '.content')
echo "${content}" | base64 -d
```

**模板类型解析**

**YAML 表单模板**（`.yml` 文件）：

```yaml
name: Bug Report
description: 报告一个缺陷
title: "[Bug]: "
labels: ["bug-report"]
body:
  - type: textarea
    id: description
    attributes:
      label: 问题描述
```

解析方式：`yaml.safe_load(content)`，提取 `name`、`title`、`labels`、`body`。

**Markdown 模板**（`.md` 文件）：

```markdown
---
name: Bug Report
about: 报告一个缺陷
title: '[Bug]: '
labels: bug-report
---

**问题描述**
{问题描述内容}
```

解析方式：提取 `---` 之间的 YAML front-matter，获取 `name`、`title`、`labels`。

**预设模板列表**（仓库无模板时 fallback）

| 模板类型 | 标签 | 适用场景 |
|---------|------|---------|
| Bug-Report | `bug-report` | 缺陷反馈 |
| Documentation | `documentation` | 文档问题 |
| Requirement | `requirement` | 需求建议 |
| Question | `question` | 咨询讨论 |
| Blank | 无 | 通用问题 |

**Bug-Report 预设模板**

```markdown
Thanks for sending an issue! Please fill in the following template to help quickly solve your problem.

### Describe the current behavior / 问题描述

{问题描述}

### Environment / 环境信息

**软件环境**:
- CANN 版本: {版本}
- 操作系统: {OS}

**硬件环境**:
- NPU 型号: {芯片型号}

### Steps to reproduce the issue / 重现步骤

{重现步骤}

### Describe the expected behavior / 预期结果

{预期结果}

### Related log / screenshot / 日志 / 截图

{日志/截图}
```

**Documentation 预设模板**

```markdown
Thanks for sending an issue! Please fill in the following template to help quickly solve your problem.

### Document Link（文档链接）

{文档链接}

### Issues Section（问题文档片段）

{问题片段}

### Existing Issues（存在的问题）

{问题描述}

### Suggested Fix（修复建议）

{修复建议}
```

**Requirement 预设模板**

```markdown
Thanks for sending an requirement! Please fill in the following template to help quickly solve your problem.

### Background（背景信息）

{背景}

### Benefit / Necessity（价值/作用）

{价值说明}

### Design（设计方案）

{设计方案}
```

### Step 3: 选择模板（业务层）

> ⚠️ **此步骤为业务层**，由调用方 skill 实现。infra 不提供具体实现。

**业务层职责**：
- 根据问题类型选择模板（如 UT缺失→Bug-Report、README缺失→Documentation）
- 确定模板标签、标题前缀

**示例映射**（由调用方 skill 定义）：

| 问题类型 | 模板类型 | 标签 |
|---------|---------|------|
| UT缺失 | Bug-Report | bug-report |
| README缺失 | Documentation | documentation |
| 功能需求 | Requirement | requirement |

### Step 4: 填充内容（业务层）

> ⚠️ **此步骤为业务层**，由调用方 skill 实现。infra 不提供具体实现。

**业务层职责**：
- 根据问题数据生成 Issue 标题
- 根据模板字段填充 Issue body 内容
- 处理合并场景（多个问题合并为一个 Issue）

**标题格式建议**

| 模板类型 | 单问题标题 | 合并标题 |
|---------|-----------|---------|
| Bug-Report | `[Bug-Report]: {repo} {op_name} {简述}` | `[Bug-Report]: {repo} {简述}（{n}个算子）` |
| Documentation | `[Documentation]: {repo} {op_name} {简述}` | `[Documentation]: {repo} {简述}（{n}个算子）` |

### Step 5: 用户确认

用 AskUserQuestion 展示 Issue 内容预览，选项：
1. **确认创建** - 使用当前内容创建 Issue
2. **修改内容** - 用户手动编辑
3. **取消操作** - 终止流程

确认时展示：
- Issue 标题
- 目标仓库
- Issue body 内容预览
- 标签

### Step 6: 创建 Issue

**API**

```
POST https://api.gitcode.com/api/v5/repos/{owner}/{repo}/issues
```

| 参数 | 必填 | 类型 | 说明 |
|------|:----:|------|------|
| `title` | Y | string | Issue 标题（最大 255 字符） |
| `body` | Y | string | Issue 描述（支持 Markdown） |
| `labels` | | **string** | 标签名称，**单个字符串**（如 `"bug-report"`），不支持数组 |
| `assignees` | | **string** | 指派用户名，**单个字符串**，不支持数组 |

> **重要**：GitCode API 的 `labels` 和 `assignees` 必须使用字符串格式，不支持 JSON 数组。多个标签可用逗号分隔（如 `"bug,enhancement"`）。

**curl 命令**

```bash
# 获取 Token（从 git credential store）
token=$(git credential fill <<< "protocol=https\nhost=gitcode.com\n\n" | grep "password=" | cut -d= -f2)

# 或使用环境变量
token="${GITCODE_TOKEN}"

# 创建 Issue
curl -X POST "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/issues?access_token=${token}" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "title": "${issue_title}",
    "body": "${issue_body}",
    "labels": "${labels}"
  }' \
  --connect-timeout 30
```

**使用 PRIVATE-TOKEN 方式**（更安全）

```bash
curl -X POST "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/issues" \
  -H "PRIVATE-TOKEN: ${token}" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "${issue_title}",
    "body": "${issue_body}",
    "labels": "${labels}"
  }'
```

**成功响应 (HTTP 201)**

```json
{
  "id": 123456,
  "iid": 42,
  "title": "[Bug-Report]: ops-math add CMake OPTYPE error",
  "state": "opened",
  "web_url": "https://gitcode.com/cann/ops-math/issues/42",
  "labels": [{"name": "bug-report"}]
}
```

**权限要求**

| 权限等级 | access_level | 能否创建 Issue |
|---------|:------------:|:-------------:|
| Guest | 10 | Web 可，API 不行 |
| Reporter | 20 | ✅ 可 |
| Developer | 30 | ✅ 可 |

**失败处理**

| 状态码 | 说明 | 处理方式 |
|--------|------|----------|
| 401 | Token 无效 | 提示用户提供新 token |
| 403 | 无权限 | 降级为手动提交（提供提交链接） |
| 404 | 项目不存在 | 确认仓库路径是否正确 |
| 422 | 参数验证失败 | 检查参数格式（labels 是否为字符串） |

**手动提交 fallback**

```bash
# 生成提交链接
submit_url="https://gitcode.com/${owner}/${repo}/issues/new"

# 如果有标签
submit_url="https://gitcode.com/${owner}/${repo}/issues/new?labels=${labels}"

echo "Issue 文件已生成，请手动提交："
echo "提交链接：${submit_url}"
echo "Issue 标题：${issue_title}"
echo "Issue 内容见文件：${issue_file}"
```

### Step 7: 记录日志

日志文件命名：`logs/issue-create_{YYYYMMDD}_{HHMMSS}.log`。日志格式详见 [references/logging-conventions.md](references/logging-conventions.md)。

### 常见问题

**Q1: 模板查询返回空**：仓库无 Issue 模板目录，使用预设模板（见 Step 2）。

**Q2: 创建失败，提示 403 Forbidden**：用户权限不足（Guest 级），提供手动提交链接。

**Q3: labels 参数报错**：必须使用字符串格式 `"bug-report"`，不能用数组 `["bug-report"]`。

**Q4: Issue body 过长**：GitCode 支持长内容，但如果超过限制，分多次 PATCH 更新。

**Q5: 查看已有 Issue**

```bash
curl "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/issues?state=opened&access_token=${token}"
```

---

## 参考文档索引

| 文档 | 说明 | 适用 skill |
|------|------|-----------|
| [references/env-check.md](references/env-check.md) | Step 0 环境预检（token / git / 临时目录 / 输出目录） | 所有 GitCode skill |
| [references/gitcode-api.md](references/gitcode-api.md) | PR/Issue/仓库 API + 错误处理 + 命令速查 | code-review, gitcode-pr-handler, gitcode-issue-gen, pr-compile, gitcode-issue-handler, tool-reports-to-issue |
| [references/url-parsing.md](references/url-parsing.md) | URL 格式识别与解析（PR/Issue） | code-review, gitcode-pr-handler, gitcode-issue-gen, pr-to-design-doc, pr-compile, gitcode-issue-handler, tool-reports-to-issue |
| [references/token-config.md](references/token-config.md) | Token 获取优先级 | 所有 GitCode skill |
| [references/logging-conventions.md](references/logging-conventions.md) | 日志命名与记录规范 | 所有 GitCode skill |
| [references/change-table-display.md](references/change-table-display.md) | 变更文件列表展示格式 | code-review, gitcode-pr-handler, gitcode-issue-gen, pr-to-design-doc |
| [references/clone-and-checkout.md](references/clone-and-checkout.md) | 克隆、浅克隆、PR 分支检出、base 分支确定、merge-base | code-review, gitcode-pr-handler, gitcode-issue-gen, pr-to-design-doc |
| [references/diff-and-changes.md](references/diff-and-changes.md) | diff 变更统计（merge-base 模式 + triple-dot 模式） | code-review, gitcode-pr-handler, gitcode-issue-gen, pr-to-design-doc, PR 创建流程 |
| [references/log-and-show.md](references/log-and-show.md) | git log 元信息提取、git show 文件读取 | code-review, gitcode-pr-handler, gitcode-issue-gen, pr-to-design-doc, PR 创建流程 |
| [references/remote-and-branch.md](references/remote-and-branch.md) | remote 管理、分支查询、push、ls-remote | PR 创建流程 |
| [references/pitfalls.md](references/pitfalls.md) | Git 操作易错点对照表 | 所有使用 git 的 skill |
| **本文档章节** | | |
| PR 创建工作流（Step 1-8） | 分支→模板→填充→确认→校验身份→推送→API创建→日志 | gitcode-issue-handler, fixer-broken-link |
| Issue 创建工作流（Step 1-7） | 仓库→模板→选择→填充→确认→API创建→日志 | tool-reports-to-issue, gitcode-issue-gen |
