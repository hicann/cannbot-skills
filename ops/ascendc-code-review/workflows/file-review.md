# 文件检视场景

## 触发
检视代码、审核代码、检查规范、代码审查、帮我检视 xxx

---

## 编排

### 任务清单

启动时创建 4 个固定任务（全部 pending）：

| 任务 | 阶段 | 内容 |
|------|------|------|
| 任务0 | 代码概要 + API 预研 + 设计文档探测 + 检视计划设计 | 并行派发 code-summarize + api-prestudy（仅 Kernel 侧）+ docs-detect → 检视计划设计 |
| 任务1 | 逐条检视 | 按波次派发通用检视子 agent |
| 任务2 | 行号校对 | steps/common.line-verify.md |
| 任务3 | 撰写报告 | steps/common.report-write.md |

### 输入解析

从用户输入提取代码文件（支持：单文件路径、多文件路径、目录路径 find 枚举），统一为 `file_input`（可以是单个路径，也可以是多个路径）。

### 阶段0：代码概要 + API 预研 + 设计文档探测 + 检视计划设计

1. 将任务0 标记为 in_progress
2. 从 file_input 提取算子名，确认文件存在
3. **在单个消息中并行派发子 Agent**（代码概要、设计文档探测总是派发，API 预研仅当 file_input 含 `op_kernel/` 或判定为 Kernel/混合侧时派发）：

- **代码概要子 Agent**：Read+执行 `steps/file-review.code-summarize.md`，传入 file_input + 概要输出路径 `./operators/{operator_name}/code_summary.md`
- **API 预研子 Agent**（仅当 file_input 含 `op_kernel/` 或判定为 Kernel/混合侧）：Read+执行 `steps/common.api-prestudy.md`，传入 file_input（仅 Kernel 侧）+ 输出路径 `./operators/{operator_name}/api_prestudy.md`
- **设计文档探测子 Agent**：Read+执行 `steps/common.docs-detect.md`，传入 file_input + 用户已指明的文档路径（明确给出则传，否则为空）→ 返回 docs_input（路径/目录或空）

4. 等待三个子 Agent 全部返回，收集：代码概要→侧别+概要路径；API 预研→预研报告路径（若已派发）；设计文档探测→docs_input（路径/目录或空）
5. **侧别回填**：若 API 预研子 Agent 未派发（纯 Tiling 侧），跳过 API 预研路径
6. Read + 执行 `steps/common.plan-design.md`，派发子 Agent 产出检视计划（通用分组 + 专项清单 + 跳过清单 + 仅核对清单）。传入 file_input + 概要路径 + API 预研路径（若存在）+ docs_input + scope_hint
7. 将任务0 标记为 done

### 阶段1：逐条检视 + 设计一致性检查

1. 将任务1 标记为 in_progress
2. Read `steps/file-review.clause-review.md` 获取 prompt 模板
3. 按阶段0 检视计划的通用分组，逐波派发：
   - 每波在单个消息中并行调用 ≤6 个 `Agent` 工具（上限见 `core/review-load-balance.md`）
   - `subagent_type` 使用 `"general"`
   - 每组用 prompt 模板填入：侧别 + 条例ID和标题 + file_input + 代码概要路径 + API 预研路径（若存在）
   - 波次内并行，波次间串行
4. **🆕design-check 与波次1 同消息并行**：派发**波次1 的那一条消息时**，若检视计划专项清单含 design-check，在同一消息里额外加入 1 个专项检视子 agent（design-check，`subagent_type: "general"`，不进 plan-design 通用分组，独立输出）。填入 docs_input + file_input + 代码概要路径 + API 预研路径（若存在）。子 Agent 内部读设计文档 + 建立设计映射 + 复用概要/API预研做 S1-S7 + D8 对照。**禁止把 design-check 排到所有波次之后单独派发**——它必须与波次1 的通用检视子 agent 在同一条消息里发出，以实现真正并行
5. 波次2 及之后：仅派发通用检视子 agent（design-check 已在波次1 并行发出，无需重复）
6. 每波完成后输出进度，所有波次完成后汇总（含 design-check 的 S1-S7 + D8 结果）
7. 将任务1 标记为 done

### 阶段2：行号校对

1. 将任务2 标记为 in_progress
2. 传入阶段1 的 FAIL/SUSPICIOUS 列表（含设计一致性 ❌ 项），Read + 执行 `steps/common.line-verify.md`
3. 将任务2 标记为 done

### 阶段3：撰写报告

1. 将任务3 标记为 in_progress
2. 传入阶段1+2 的结果，Read + 执行 `steps/common.report-write.md`
3. 报告输出路径 `./operators/{operator_name}/{source_file}_review_summary.md`
4. 将任务3 标记为 done

---

## 上下文传递链

```
                 ┌─ code-summarize → 侧别 + 概要路径 + 跨文件关系
阶段0（并行） ───┤─ api-prestudy → API 预研报告路径（仅 Kernel 侧）
                 └─ docs-detect → docs_input（设计文档路径/目录或空）
                         ↓ 三个子 Agent 全部返回后
                  plan-design → 检视计划（通用分组 + 专项清单 + 跳过/仅核对清单）
                         ↓
阶段1 → 逐条结果 (PASS/FAIL/SUSPICIOUS) +（docs_input 非空时）design-check 的 S1-S7 + D8 结果
         ↓
阶段2 → 校对后的 FAIL/SUSPICIOUS（含设计一致性 ❌ 项）
         ↓
阶段3 → 报告文件（docs_input 非空时含设计一致性章节）
```

## 约束

- 严格按阶段顺序执行，禁止跳步
- 阶段0 的子 Agent 必须在单个消息中并行派发（代码概要 + 设计文档探测总是，API 预研仅 Kernel 侧）；plan-design 在三个子 Agent 全部返回后单独派发
- design-check 发射与否由 plan-design 专项清单决定,workflow 阶段1 只按专项清单执行派发（含 design-check 则与波次1 同消息并行,不进通用分组）；专项清单不含 design-check 时报告退化为纯条例检视
- 禁止提前 Read 未执行阶段的 step 文件
