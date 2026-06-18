---
skill_name: gitcode-pr-handler
eval_mode: text
---

# Case 1: PR 标题与描述重生成流程

## Config
- Max Tokens: 180000
- Max Tokens (deepseek-v4-flash): 210000
- Max Tokens (glm-5): 195000
- Ascend Platform: A2

## Prompt

有一个 PR（https://gitcode.com/cann/ops-math/merge_requests/1564），需要重新生成 PR 标题和描述。gitcode-pr-handler 的完整工作流程是什么？

## Expected Output

回复应说明完整流程：Step 0 环境预检（token/git/curl/tmp）→ Step 1-6 解析 PR 链接、获取 PR 详情、克隆代码、展示变更列表、查找 PR 模板、生成新标题与新描述 → Step 7 最终提交确认。应说明中间步骤不打断用户（AskUserQuestion 仅用在环境预检和最终提交两次），以及 PATCH 后必须 GET 回查验证。

## Expectations
- [contains] PR 模板
- [contains] PATCH


---

# Case 2: PR 标题格式（约定式提交）

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 165000
- Ascend Platform: A2

## Prompt

gitcode-pr-handler 生成 PR 标题时使用什么格式？有哪些类型前缀？如果一次 PR 包含多种变更怎么处理？

## Expected Output

回复应说明使用约定式提交格式：类型前缀包括 feat/fix/docs/refactor/test/perf 等。一次 PR 含多类变更时取占比最大的类型，混合改动可加 scope（如 feat(login): ...）。

## Expectations
- [contains] feat
- [contains] fix


---

# Case 3: PR 模板选择与填充原则

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 165000
- Ascend Platform: A2

## Prompt

gitcode-pr-handler 如何选择 PR 模板？模板填充时有哪些关键原则？

## Expected Output

回复应说明模板优先级：.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md → .gitcode/PULL_REQUEST_TEMPLATE.md → .github/PULL_REQUEST_TEMPLATE.md。应说明内容必须来源于代码变更进行分析，不能凭经验编造。应说明沿用模板的原章节标题、仅替换占位、不增加/删减章节的原则。

## Expectations
- [contains] PULL_REQUEST_TEMPLATE
- [contains] 模板


---

# Case 4: 内容生成与自动决策

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 165000
- Ascend Platform: A2

## Prompt

gitcode-pr-handler 在内容生成时，哪些决策是自动做的不需要问用户？哪些地方需要打断用户确认？

## Expected Output

回复应说明交互节奏仅两次卡点：Step 0 环境预检和 Step 7 最终提交确认。中间过程（模板选择、标题草稿、描述草稿）均为自动决策，在文本中说明"做了什么、为什么"。最终提交时统一展示旧内容和新内容的对比，用一次 AskUserQuestion 确认。

## Expectations
- [contains] 自动
- [contains] 确认
