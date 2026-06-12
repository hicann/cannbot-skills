---
skill_name: gitcode-issue-gen
eval_mode: text
---
# Case 1: 文档变更的模板选择

## Config
- Max Tokens: 120000
- Max Tokens (deepseek-v4-flash): 140000
- Max Tokens (glm-5): 130000
- Ascend Platform: A2

## Prompt

我创建了一个PR，同步更新了README.md、CHANGELOG、ST测试框架及PR模板等文档
请问用 gitcode-issue-gen 来为这个PR创建Issue，应该选用什么 Issue 模板？Issue body 应该包含哪些部分？

## Expected Output

回复应说明为 documentation 类型的变更创建 Issue，Issue body 应包含背景信息、价值作用、设计方案等章节，并在末尾关联 PR 链接。提及 Step 0 环境预检和 Step 7 提交确认的工作流步骤。

## Expectations
- [contains] gitcode-issue-gen
- [contains] documentation


---

# Case 2: Bug Fix 变更的模板选择

## Config
- Ascend Platform: A2

## Prompt

我修复了一个 bug,问题是算子在高版本驱动下计算结果错误，原因是某个 API 参数类型不匹配。请告诉我 gitcode-issue-gen 对这个 PR 应该用什么模板？

## Expected Output

回复应说明本次变更为 bug fix 类型，应选用 bug-report 模板。

## Expectations

---

# Case 3: 文档变更的模板选择

## Config
- Ascend Platform: A2

## Prompt

这是一个文档 PR：https://gitcode.com/cann/cannbot-skills/pull/132
主要更新了 ascendc-precision-debug 的 README，补充了常见精度问题的排查步骤。用 gitcode-issue-gen 处理这类 PR，推荐用哪个模板？

## Expected Output

回复应说明本次变更为文档类型，应选用 documentation 模板。Issue body 应描述变更内容和目的。

---

# Case 4: 正向看护-多 Infra Skill 环境下正确选择目标 Skill

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Distractor skills: gitcode-issue-handler;gitcode-pr-handler;gitcode-toolkit
- Ascend Platform: A2

## Prompt

我想为一个新增算子的 PR 创建关联 Issue，应该使用哪个 skill？流程是怎样的？

## Expected Output

回复应说明使用 gitcode-issue-gen skill 来处理这个需求，并概述核心工作流程步骤。

## Expectations

---

# Case 5: 负向看护-不相关的查询不应触发 Issue 工作流

## Config
- Ascend Platform: A2

## Prompt

请问 GitCode 上怎么创建一个新的仓库？初始化和推送代码的步骤是什么？

## Expected Output

回复应说明 GitCode 上创建仓库和推送代码的通用 Git 操作步骤（git init, git remote add, git push 等），不应涉及 Issue 创建或 PR 关联的工作流。

# Case 6: 工作流知识验证

## Config
- Ascend Platform: A2

## Prompt

我想用 gitcode-issue-gen 给 PR 自动创建 Issue，能介绍一下整个工作流程和涉及的 API 吗？

## Expected Output

回复应至少覆盖 gitcode-issue-gen 的核心工作流阶段：Step 0 环境预检、PR 解析与代码克隆、变更列表展示、Issue 模板查找与选择、Issue body 生成、提交确认、Issue 创建与 PR 关联、可选 Assign。应提及 GitCode API 端点如 repos/issues 和双向关联机制。

## Expectations
- [contains] gitcode-issue-gen
- [contains] 环境预检
- [contains] 双向关联
- [contains] api/v5


---

# Case 7: 边界场景-信息不足时主动追问

## Config
- Ascend Platform: A2

## Prompt

帮我创建一个 Issue，我有个 PR 是关于修复异步执行时 context 丢失的问题。

## Expected Output

回复应识别到用户提供了 PR 信息但缺少关键细节（PR 链接或目标仓库等），主动追问缺失信息。不应在缺少必要信息的情况下直接生成 Issue body 或自动执行操作。询问 PR 链接、目标仓库、Issue 类型等任一关键信息均可。

## Expectations
- [contains] 创建

