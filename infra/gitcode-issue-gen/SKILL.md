---
name: gitcode-issue-gen
description: 根据 GitCode PR 的代码变更生成关联 Issue（按变更类型自动选用 feature-request / bug-report / documentation 等仓库已有的 Issue 模板），通过 GitCode REST API 创建 Issue 后回写 PR 描述完成双向关联，并可选地把新 Issue 自助 assign 给当前 token 的 GitCode 用户。当用户提供 PR 链接、要求"创建 Issue / 生成 Issue / 关联 Issue / 给 PR 建 Issue"时触发此 skill。
license: CANN-2.0
---

# GitCode Issue 生成与关联

根据 GitCode PR 的代码变更，按变更类型自动选定 Issue 模板，生成对应的 Issue 并完成 PR ↔ Issue 双向关联。

---

## 核心原则

### 内容必须来源于代码

- 所有技术信息必须从 PR 代码变更中直接获取，禁止基于经验推断或假设
- 使能方式、支持框架、数据类型、硬件平台等信息必须能在代码里找到依据
- 代码中没有明确说明的特性，**不要**写进 Issue body

### 双向关联

创建 Issue 后必须更新 PR 描述添加 Issue 链接，Issue body 中也必须包含 PR 链接，二者相互引用。

### 禁止创建测试内容

严禁创建测试 Issue 或调试评论到目标仓库。API 调用问题应通过本地构造 JSON 排查，**绝不**通过向目标仓库发请求试错。

### 交互节奏：环境预检 + 终局确认 + 可选 Assign，最多三次卡点

整个流程仅在以下三类时刻打断用户：

1. **Step 0 环境预检** — token / git / curl / /tmp 等缺失时 AskUserQuestion 询问
2. **Step 7 提交确认** — Issue 模板、Issue body、PR 关联写法生成完毕后，**一次** AskUserQuestion 询问是否提交
3. **Step 8 自助 Assign（可选）** — 仅在 Issue 创建成功时弹**一次**，询问是否将新 Issue assign 给当前 token 用户

中间过程的所有"判断"——模板选择、Issue body 草稿——**不要**用 AskUserQuestion 打断，按下文「自动决策」执行并在文本里说明"我做了什么、为什么"。

AskUserQuestion 一次只问一个。

### 自动决策

| 决策点 | 做法 |
|--------|------|
| Issue 模板选择 | 按下表"Issue 模板按变更类型推荐"自动选定，文本说明所选项与原因 |
| Issue body 草稿 | 直接生成；用户在 Step 7 看到完整 body 再统一确认 |

**Issue 模板按变更类型推荐：**

| 变更特征 | 推荐模板 |
|----------|----------|
| RFC 提案 / 架构设计 / 重大变更 | request-for-comments |
| 新增功能 / 新增算子 / 新增接口 | feature-request / requirement |
| Bug 修复 / 异常处理 / 回退 | bug-report |
| 文档变更 | documentation |
| 重构 / 性能优化 | feature-request（无 refactor 模板时退而求其次） |

仓库实际可用的模板与推荐不同时，按"实际存在 ∩ 推荐项"取交集；推荐项不存在则按 `requirement` → `feature-request` → `bug-report` 顺序降级。

---

## 工作流程

### Step 0：环境预检（必经）

> 详见 [gitcode-toolkit/references/env-check.md](../gitcode-toolkit/references/env-check.md)

token / git / curl / python3 / /tmp 任一缺失 → AskUserQuestion 询问（一次只问一个）；全部通过则输出预检报告进入 Step 1。

### Step 1-6：解析与上下文获取

1. **解析 PR 链接** → 提取 owner / repo / pr_number（详见 [gitcode-toolkit/references/url-parsing.md](../gitcode-toolkit/references/url-parsing.md)）
2. **获取 PR 详情** → 调用 API 取 title / body / head.ref / base.ref（详见 [gitcode-toolkit/references/gitcode-api.md](../gitcode-toolkit/references/gitcode-api.md)）；如果 PR body 中已能识别到 `#issue_number`，向用户说明"PR 已关联 Issue #N，是否仍创建新 Issue"（AskUserQuestion 二选一）
3. **克隆代码** → `/tmp/gitcode-issue-gen_${owner}_${repo}_${ts}`（详见 [gitcode-toolkit/references/clone-and-checkout.md](../gitcode-toolkit/references/clone-and-checkout.md)）
4. **展示变更列表** → `git diff --numstat` + `--name-status` 表格展示，**直接进入下一步**（不询问）
5. **查找 Issue 模板** → 分别检查 `.gitcode/ISSUE_TEMPLATE/` 与 `.github/ISSUE_TEMPLATE/`（必须独立检查，禁止 `&&` 链式）；收集所有可用模板后按上文「Issue 模板按变更类型推荐」自动选定，文本说明"已识别可用模板：[…]，自动选用 X，原因：…"
6. **生成 Issue body** → 使用模板填充，结尾固定追加 `关联 PR：https://gitcode.com/${owner}/${repo}/pull/${pr_number}`（模板格式见 [references/issue-templates.md](references/issue-templates.md)）

### Step 7：最终提交确认（流程中**唯一**的提交确认）

完成 Issue 生成后统一展示：

```
即将执行：
  PR #${pr_number}
    将在 PR body 末尾追加："关联的Issue：#${新 Issue 号待生成}"
  Issue（新建）
    模板：${xxx.yml}（自动选择，原因：…）
    标题：…
    body：…
```

调用**唯一一次** AskUserQuestion：

```
问题: 以上内容是否提交到 GitCode？
选项:
  - 全部按此提交 (Recommended)
  - 取消，保留本地草稿
```

确认后顺序执行：
1. **POST** 创建 Issue（API 详见 [gitcode-toolkit/references/gitcode-api.md](../gitcode-toolkit/references/gitcode-api.md)）→ 取回 `new_issue_number`
2. **PATCH** PR body 追加关联链接（不重写其他段落，仅在 PR body 末尾或"关联的Issue"段插入 `#${new_issue_number}`）
3. **GET** 回查 PR / Issue 验证写入成功（**PATCH 200 不代表写入成功**）

### Step 8：Issue 自助 Assign（可选）

**触发前提**：本次 Issue 创建成功（API 返回 201 且 GET 回查到 Issue）。失败则跳过整个 Step 8。

**执行**：

1. 用 token 调 `GET /api/v5/user` 取当前认证用户 login：

```bash
curl -sS "https://api.gitcode.com/api/v5/user?access_token=${GITCODE_TOKEN}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('login',''))"
```

> 用 `/user` 而非要求用户配置环境变量——token 已唯一标识认证主体，再问 username 冗余且容易拼错（GitCode login ≠ 昵称、≠ git config user.name）。

2. 若 `/user` 返回非空 login，发起**唯一一次** AskUserQuestion：

```
问题: Issue #${new_issue_number} 已创建。是否将其 assign 给 @${login}？
        将通过在 Issue 下评论 "/assign @${login}" 实现。
选项:
  - 是，assign 给我 (Recommended)
  - 否，保持未指派
```

3. 若 `/user` 调用失败（401/403/网络）：跳过询问，输出一行说明「无法通过 token 解析当前用户（HTTP X），跳过 Issue 自助 assign」。**不要**追问用户手填 username——根因是 token 权限/网络，问用户也救不回来。

4. 用户选"是"则 POST Issue 评论：

```bash
curl -X POST -H "Content-Type: application/json" \
  -d "{\"body\": \"/assign @${login}\"}" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/issues/${new_issue_number}/comments?access_token=${GITCODE_TOKEN}"
```

> 中文/特殊字符场景推荐 Python `urllib` 构造请求，避免 shell 转义出错。
> 评论 body **必须严格**为 `/assign @${login}`，**不**附加文字/emoji/换行（附加内容会破坏平台指令解析）。

**反模式**：

- ❌ Issue 创建失败 / GET 回查失败也询问 assign
- ❌ 凭 git config user.name 或 PR 作者推断当前用户
- ❌ 把 Step 8 询问和 Step 7 提交确认合并成一条 AskUserQuestion
- ❌ `/user` 失败时降级追问 username

---

## API / Token / 错误处理

GitCode API、Token、HTTP 状态码错误处理统一详见：

- [gitcode-toolkit/references/gitcode-api.md](../gitcode-toolkit/references/gitcode-api.md)（注意 `labels`/`assignees` 必须字符串而非数组）
- [gitcode-toolkit/references/env-check.md](../gitcode-toolkit/references/env-check.md) / [token-config.md](../gitcode-toolkit/references/token-config.md)

本 skill 的本地约束：

- 调试 API 参数时，先在本地构造 JSON 验证格式，**禁止向目标仓库发送测试请求**

---

## 参考文档

- [references/issue-templates.md](references/issue-templates.md) — Feature Request / Bug Report Issue 模板
- [gitcode-toolkit/references/gitcode-api.md](../gitcode-toolkit/references/gitcode-api.md) — GitCode PR / Issue API 详细文档
