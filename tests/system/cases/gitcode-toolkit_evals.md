---
skill_name: gitcode-toolkit
eval_mode: text
---

# Case 1: GitCode API Token 获取方式

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

我需要在自动化脚本中调用 GitCode API，请问如何获取 access_token？有哪些获取方式？

## Expected Output

回复应说明 GitCode API Token 的获取方式和优先级：用户直接提供 → 环境变量 GITCODE_TOKEN → 询问用户。应提及 PRIVATE-TOKEN header 方式和 access_token 参数方式，以及 token 的权限 scope（至少 reporter 以上才能创建 Issue）。

## Expectations
- [contains] GITCODE_TOKEN
- [contains] access_token
- [contains] PRIVATE-TOKEN


---

# Case 2: 从 fork 仓库创建 PR 的完整流程

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 160000
- Ascend Platform: A2

## Prompt

我们有一个 fork 仓库 hello_simida/cannbot-skills，现在想往上游 cann/cannbot-skills 的 master 分支提 PR。请问完整的操作流程是什么？涉及哪些 git 命令和 API 调用？

## Expected Output

回复应说明从 fork 仓库创建 PR 到上游的完整流程。关键步骤包括：获取分支信息和 commit 历史、推送分支到远程、调用 GitCode API 创建 PR（POST /api/v5/repos/{owner}/{repo}/pulls，head 参数格式为 {username}:{branch}）。应包含 git push、git remote 等核心命令及 PR 模板的结构。

## Expectations
- [contains] git push
- [contains] POST
- [contains] head


---

# Case 3: GitCode PR 分支检出与 diff

## Config
- Max Tokens: 120000
- Max Tokens (deepseek-v4-flash): 140000
- Max Tokens (glm-5): 130000
- Ascend Platform: A2

## Prompt

我需要审查一个 PR（https://gitcode.com/cann/ops-math/merge_requests/1564），如何拉取这个 PR 的代码到本地？如何查看 PR 相比 master 的变更？

## Expected Output

回复应说明 PR 分支的检出方法：通过 git fetch 拉取 PR 分支后再切换。应说明如何对比 PR 与目标分支的变更差异，包括查看文件变更列表和具体 diff 内容。应包含 git fetch、git diff 等命令的使用方式。

## Expectations
- [contains] git fetch
- [contains] git diff


---

# Case 4: Git URL 格式识别

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

GitCode 上 PR 和 Issue 的 URL 格式是什么样的？如何从 URL 中提取项目路径（owner/repo）？

## Expected Output

回复应说明 GitCode 的 URL 格式：
- PR 格式：/pull/{n}、/pulls/{n}、/merge_requests/{n}
- Issue 格式：/issues/{n}
应说明从 URL 提取项目路径的方法：sed 正则提取 gitcode.com/{owner}/{repo} 部分，以及克隆 URL（https://gitcode.com/{owner}/{repo}.git）和 API URL 的格式。

## Expectations
- [contains] /issues/
- [contains] gitcode.com


---

# Case 5: Issue 创建流程与模板

## Config
- Max Tokens: 120000
- Max Tokens (deepseek-v4-flash): 140000
- Max Tokens (glm-5): 130000
- Ascend Platform: A2

## Prompt

我需要通过 API 在 cann/ops-math 仓库创建一个 Bug Report Issue，请问完整的操作流程是什么？API 参数需要注意什么？

## Expected Output

回复应说明通过 GitCode API 或 MCP 工具创建 Issue 的方法和流程，包括 API 参数（title、body、labels）和注意事项。应提及创建 Issue 所需的权限等级和 labels 参数的特殊要求（字符串格式而非 JSON 数组）。

## Expectations
- [contains] labels


---

# Case 6: 环境预检流程

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

GitCode 协作类 skill 在执行前需要做哪些环境检查？检查顺序是什么？

## Expected Output

回复应说明 GitCode 协作类 skill 在执行前需要做的环境检查，包括但不限于：Token 配置检查、Git 环境检查、临时目录可用性等。应说明检查失败时的处理方式。

## Expectations
- [contains] token


---

# Case 7: 正向看护-多 GitCode skill 环境下正确触发

## Config
- Max Tokens: 120000
- Max Tokens (deepseek-v4-flash): 140000
- Max Tokens (glm-5): 130000
- Distractor skills: gitcode-issue-gen;gitcode-pr-handler;gitcode-issue-handler;cannbot-skill-reviewer
- Ascend Platform: A2

## Prompt

我想了解 GitCode 上 PR 的完整创建流程：从 fork 仓库推送到上游仓库的全部步骤，包括 git 操作和 API 调用。请参考 gitcode-toolkit 内部参考 skill 来回答。

## Expected Output

回复应基于 gitcode-toolkit 的 PR 创建工作流给出完整指导，涵盖 8 个步骤：获取信息、获取 PR 模板、分析填充、用户确认、校验身份、推送分支、API 创建、记录日志。即使在多个 GitCode skill 同时可用的环境下，也应正确激活 gitcode-toolkit skill 来提供参考信息。

## Expectations
- [contains] git push
- [contains] POST
- [skill_activated] gitcode-toolkit
