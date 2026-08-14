# 逐条检视 prompt 模板（大型 PR 检视）

workflow 按全局波次规划逐波派发子 Agent。每组使用以下 prompt 模板。
与标准 `pr-review.clause-review.md` 的核心差异：增加文件组标注和文件列表约束，限制子 Agent 上下文范围。

## prompt 模板

```
【已由上游完成】
- 文件组：{group_name}
- 代码侧别：{Kernel侧/Tiling侧/混合}
- 条款过滤：已按侧别过滤
- 代码概要：{code_summary_path}
- API 预研报告：{api_prestudy_path}（仅 Kernel 侧，若存在）

检视 PR diff：{diff_file_path}
检视文件列表（仅读取以下文件）：{group_file_list}
完整源码路径：{repo_path}

检视条款：{条例ID-1} {条例标题} (references/{file}.md:{line})、{条例ID-2} {条例标题} (references/{file}.md:{line})

【执行要求】
- 第一步加载 ascendc-code-review skill，然后 Read skill 目录下的 `core/methodology.md` 掌握假设检验方法、置信度标准、红线问题和 PR 交叉验证规则
- 若提供了代码概要，Read 获取本文件组的函数清单和 API 调用索引
- API 约束信息：若已提供 API 预研报告，以其为主要来源。若预研报告未覆盖当前条款涉及的 API，使用 `/ascendc-docs-search` 补充查阅
- 对每条分配的条例：若检视条款中已附带行号（references/{file}.md:{line}），从该行号起 Read 到下一个 `^####` 标题为止；否则 Grep `^{条例ID}` 定位起始行号 + 下一标题定位结束行号，Read offset={start} limit={end-start}。**只读该条例章节，禁止 Read 整个规则文档**。若条例包含专属检视方法，必须严格按该指引执行（如 cpp-style 声明全文 Read 且不走假设检验，则覆盖本条逐条 Grep 与下方的假设检验要求）
- 若检视条款来自 ascendc-api / ascendc-perf / simt-api-analysis / mc2-specific 且预研报告未覆盖，使用 `/ascendc-docs-search` 查阅对应 API 的最新官方文档
- 若分配的条款包含 RB-\*（RegBase 路线专项），需额外加载 `ascendc-regbase-best-practice` skill 获取 API 白名单和参考实现文档
- **严格约束**：只读取「检视文件列表」中的文件，不越界读取其他文件组的文件
- 先 Read diff 中本组文件的变更部分，再按需 Read 完整源码追溯变量来源
- 大 PR 模式下深度分析（变量溯源、TilingData 值域）需自行按需 grep，summary 不提供
- 严格按假设检验驱动流程执行（H0/H1、证据收集、自信值计算）。**例外**：若条例的专属检视方法已声明不走假设检验（如 cpp-style），按专属方法执行，不收集证据分值
- 所有条款检视完成后直接输出逐条结果，禁止生成报告文件
- 每条结果标注文件组：`[{group_name}] {条例ID} PASS/FAIL`

【⚠️ 逃逸信号检测】
一旦发现自己即将输出以下内容，立即停止并重新从第一条条款开始：
- "批量处理多个任务"/"合并处理" → 每条必须独立经过完整假设检验流程
- "直接生成检视报告"/"总结所有结果" → 必须完成所有分配条款后才能输出
- "提高效率"/"节省时间"/"简化流程" → 效率不是跳过步骤的理由
触发时输出 `⚠️ 检测到逃逸信号，重置到第一条条款` → 立即重新执行
```

## 输出格式

PASS 条例仅列出 ID 和文件组名，无需展开分析：
```
[{group_name}] {条例ID} PASS
```

FAIL/SUSPICIOUS 展开完整分析：
```
[{group_name}] {条例ID} FAIL 置信度:HIGH
- 问题描述：{描述}
- 代码片段（{完整文件路径} 行 N-M）：
  ```cpp
  {至少 10 行代码，含上下文}
  ```
- 假设检验证据：正向证据 + 负向证据 + 自信值 = {累计}% ≥ 70% → 判定违规
- 修复建议：{建议}
```

`[STYLE]` 标记的代码风格条例（专项检视子 agent，不走假设检验，无证据分值；style_global 组的 group_name 为 `style_global`）：
```
[style_global] [STYLE] {条例编号+名称} FAIL 置信度:MED/LOW
- 问题描述：{描述}
- 代码片段（{完整文件路径} 行 N-M）：
  ```cpp
  {至少 10 行代码，含上下文}
  ```
- 修复建议：{建议}
```
风格条例 PASS 同样仅列 `[style_global] [STYLE] {条例编号+名称} PASS`。

**文件路径硬约束**：代码片段必须标注完整文件路径（相对于 repo_path），格式为 `{完整文件路径} 行 N-M`。禁止只写短文件名或仅标注 group_name。

禁止为 PASS 条例输出置信度或分析过程。禁止生成报告文件。

## 与标准 clause-review 的差异速查

| 维度 | 标准 | 大型 PR |
|------|------|---------|
| 文件组标注 | 无 | 每条结果标注 `[{group_name}]` |
| 文件范围 | 全部变更文件 | 显式文件列表，禁止越界 |
| 深度分析 | summary 已提供 | 子 Agent 自行按需 grep |
