# 经验总结插件 Task 调用契约

> 各步的角色、prompt。编号与 [SKILL.md 内部步骤表](../SKILL.md#内部步骤表) 一一对应。
>
> **调用原则**：PM 每阶段首次调度子 Agent 时，必须严格按照本文档指定的角色和 prompt **原样调用**，仅允许替换 prompt 中的 `<算子名>` 项，**严禁干涉实现细节**。
>
> **并行说明**：plugin-experience-summary-1/2/3 互无依赖，PM 一次性并行派发三个子任务；每个子任务只读过程产物、只写自己的 tmp 产出，不感知其它并行任务。

## plugin-experience-summary-1 开发回顾与踩坑总结

- **角色**：developer-doc

```md
- 【权限】你只可写 `.cannbot/<算子名>/tmp/`，其它写入操作会被 hooks 拦截。
- 【输入】本次开发全量过程产物：`.cannbot/<算子名>/`（state.json、LOG.md、Issue-问题记录、各阶段交付件、questionnaires/、tmp/）、`.cannbot/环境信息.md`。
- 【输出】踩坑回顾段落，写入 `.cannbot/<算子名>/tmp/经验总结-1-踩坑回顾.md`。
- 【skills】立即加载 `workflow-doc-templates`。
- 【验收标准】回顾覆盖本次开发的关键环节（需求/方案/开发/测试/验收/上库）；踩坑点逐条列出「现象 → 根因 → 建议」，且每条标注是否已有本仓 skill 明确说明——**只收录本仓 skill 未说明确的点**，已说明的不得重复收录。
```

## plugin-experience-summary-2 hooks 异常拦截分析

- **角色**：developer-doc

```md
- 【权限】你只可写 `.cannbot/<算子名>/tmp/`，其它写入操作会被 hooks 拦截。
- 【输入】本次会话的拦截记录（从 state.json、LOG.md、Issue-问题记录、tmp/ 中寻找 permission-guard / 静默问卷拦截相关记录）；hook 源文件：`hooks/opencode/permission-guard.js` 与 `hooks/claude/permission-guard.js`（只读，不修改）。
- 【输出】hooks 分析与修复建议段落，写入 `.cannbot/<算子名>/tmp/经验总结-2-hooks分析.md`。
- 【参考资料】读取 `workflow-agent-permissions` 的 SKILL.md **原文**作为权限范围与守卫边界的基准（该 skill 不经 skill 加载工具触发，直接读文件）。
- 【验收标准】逐条记录本次会话中的实际拦截/漏拦/误拦事件（含事件上下文、角色、工具、结果）；对照两侧 hook 源给出建议修复点（如规则缺失、误伤、语义不一致、两侧语义不同步），每条标注严重度与修复位置；无拦截记录时明确写出「本次会话无 hooks 拦截记录」，并基于两侧 hook 源给静态建议。**误拦事件须追到规则层根因**（是路径分类规则不匹配，还是角色权限范围本身不含该目录）——绕开守卫落盘属绕行手段，不作为修复结论。
```

## plugin-experience-summary-3 耗时统计与重试定位

- **角色**：developer-doc

```md
- 【权限】你只可写 `.cannbot/<算子名>/tmp/`，其它写入操作会被 hooks 拦截。
- 【输入】`.cannbot/<算子名>/state.json`（各阶段/CP 完成时间戳）、`LOG.md`、回退与重试记录（error-handling 相关、rounds 字段）、CI 等待记录（pending_ci）。
- 【输出】耗时统计段落，写入 `.cannbot/<算子名>/tmp/经验总结-3-耗时统计.md`。
- 【skills】立即加载 `workflow-doc-templates`。
- 【验收标准】给出本次工作流运行总时长（首步启动至收尾，注明统计口径）；按阶段/CP 分解耗时占比；明确列出反复重试耗时的地方（回退轮次、CI 等待、性能迭代等）并给出耗时原因与改进建议；时间戳缺失时如实标注并说明统计口径。
```

## plugin-experience-summary-4 汇总与 issue 征询

- **角色**：PM（你自己）

```md
- 读取 `.cannbot/<算子名>/tmp/经验总结-{1,2,3}-*.md` 三段，整合为完整经验总结：① 踩坑回顾（仅本仓 skill 未说明确的点）② hooks 异常拦截分析与建议修复点 ③ 耗时统计与反复重试定位 ④ 汇总与后续建议；落盘 `.cannbot/<算子名>/插件经验总结.md`。
- **环境检查**：读取 `.cannbot/环境信息.md` 的「环境补充记录」节——若该节非空（本次开发中环境项曾被探索补充），说明初始环境检查（步骤 0）未覆盖到这些项，在总结中**建议提 Issue 补充环境检查内容**（列出被补充的环境项与缺失原因）；该建议与其它建议一并纳入下方 issue 征询。
- 三段任一缺失或不全：回退对应步骤（plugin-experience-summary-1/2/3）补齐后重试，往返不超过 2 轮。
- 落盘后征询用户：是否需要拟一条 issue 反馈到 cannbot-skills 仓（gitcode.com/cann/cannbot-skills）？——交互模式用 question 工具直接发送用户；静默模式并入任务完成总结列出选项。
- 用户同意：加载 `gitcode-issue-gen`，按模板拟 issue 草稿（标题、分类、现象、建议修复点、复现/证据），**先交用户确认后再提交**；用户不同意或暂缓：在插件经验总结.md 中记录征询结论。
```
