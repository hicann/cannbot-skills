# PR 检视场景

## 触发
检视 PR、审核 PR、帮我检视这个 PR

---

## 编排

### 任务清单

启动时创建 4 个固定任务（全部 pending）：

| 任务 | 阶段 | 内容 |
|------|------|------|
| 任务0 | 获取 diff + 代码概要 + API 预研 + 设计文档探测 + 检视计划设计 | code-fetch → 并行派发 code-summarize + api-prestudy（仅 Kernel 侧）+ docs-detect → 检视计划设计 |
| 任务1 | 逐条检视 | 按波次派发通用检视子 agent |
| 任务2 | 行号校对 | steps/pr-review.line-verify.md |
| 任务3 | 撰写报告 | steps/common.report-write.md |

### 阶段0：获取 diff + 代码概要 + API 预研 + 设计文档探测 + 检视计划设计

1. 将任务0 标记为 in_progress
2. 提取 PR 链接，判断托管平台
3. Read + 执行 `steps/pr-review.code-fetch.md` 的派发指令，派发子 Agent 获取 diff 和完整源码
4. 等待 code-fetch 子 Agent 返回（产出 diff_path + repo_path）
5. **快速检测 diff 规模**：
   - 统计 diff 的变更文件数（提取 diff 中所有变更文件路径列表）和 diff 总变更行数（新增+删除行数，排除注释空行）
   - 若**文件数 >20 且总变更行数 ≥3000**：输出「检测到大型 PR（{N} 个文件 / {M} 行变更），自动切换大型 PR 检视流程」→ 将全部现有任务标记为 deleted → 转至执行 `workflows/pr-large-review.md`（diff_path + repo_path 已就绪，从该 workflow 的阶段0 Step 5 file-split 开始，该 workflow 会创建新的任务清单；并传入 docs_input 供其 design-check）→ 本 workflow 终止
   - 否则（文件数 ≤20 或总变更行数 <3000）：继续执行下方标准流程
6. **在单个消息中并行派发子 Agent**（代码概要、设计文档探测总是派发，API 预研仅当 diff 含 `op_kernel/` 或判定为 Kernel/混合侧时派发）：

- **代码概要子 Agent**：Read+执行 `steps/pr-review.code-summarize.md`，传入 diff 路径 + repo_path + 概要输出路径 `./operators/pr-{pr_number}/code_summary.md`
- **API 预研子 Agent**（仅当 diff 含 `op_kernel/` 或判定为 Kernel/混合侧）：Read+执行 `steps/common.api-prestudy.md`，传入 Kernel 侧文件列表（从 repo_path 筛选）+ 输出路径 `./operators/pr-{pr_number}/api_prestudy.md`
- **设计文档探测子 Agent**：Read+执行 `steps/common.docs-detect.md`，传入 repo_path + 用户已指明的文档路径（明确给出则传，否则为空）→ 返回 docs_input（路径/目录或空）

7. 等待三个子 Agent 全部返回，收集：代码概要→侧别+概要路径；API 预研→预研报告路径（若已派发）；设计文档探测→docs_input（路径/目录或空）
8. Read + 执行 `steps/common.plan-design.md`，派发子 Agent 产出检视计划（通用分组 + 专项清单 + 跳过清单 + 仅核对清单）。传入 diff 路径 + repo_path + 概要路径 + API 预研路径（若存在）+ docs_input + scope_hint
9. 将任务0 标记为 done

### 阶段1：逐条检视 + 设计一致性检查

1. 将任务1 标记为 in_progress
2. Read `steps/pr-review.clause-review.md` 获取 prompt 模板
3. 按阶段0 检视计划的通用分组，逐波派发：
   - 每波在单个消息中并行调用 ≤6 个 `Agent` 工具（上限见 `core/review-load-balance.md`），`subagent_type` 使用 `"general"`
   - 每组用 prompt 模板填入：侧别 + 条例ID + diff路径 + repo_path + 概要路径 + API 预研路径（若存在）
   - **代码范围**：使用 plan-design 输出中每组的侧别标签（仅Kernel / 仅Tiling / 全部），填入 prompt 的「检视代码范围」字段
   - 波次内并行，波次间串行
4. **🆕design-check 与波次1 同消息并行**：派发**波次1 的那一条消息时**，若检视计划专项清单含 design-check，在同一消息里额外加入 1 个专项检视子 agent（design-check，`subagent_type: "general"`，不进 plan-design 通用分组，独立输出）。填入 docs_input + diff路径 + repo_path + 概要路径 + API 预研路径（若存在）。**禁止把 design-check 排到所有波次之后单独派发**——它必须与波次1 的通用检视子 agent 在同一条消息里发出，以实现真正并行
5. 波次2 及之后：仅派发通用检视子 agent（design-check 已在波次1 并行发出，无需重复）
6. 每波完成后输出进度，所有波次完成后汇总（含 design-check 的 S1-S7 + D8 结果）
7. 将任务1 标记为 done

### 阶段2：行号校对

1. 将任务2 标记为 in_progress
2. **拆分输入路由（correctness 关键）**：clause-review 的 FAIL/SUSPICIOUS → Read+执行 `steps/pr-review.line-verify.md`（带 diff 范围红线）；design-check 的 S1-S7 + D8 ❌ 项 → Read+执行 `steps/common.line-verify.md`（无 diff 红线，因设计偏差常指向未变更代码）
3. 将任务2 标记为 done

### 阶段3：撰写报告

1. 将任务3 标记为 in_progress
2. Read + 执行 `steps/common.report-write.md`
3. 报告输出路径 `./operators/pr-{pr_number}/{pr_number}_review_summary.md`
4. 将任务3 标记为 done

---

## 与文件检视的关键差异

| 差异点 | 说明 |
|--------|------|
| 阶段0 多一步 code-fetch | 先获取 diff + clone 源码，再并行派发 |
| 阶段1 传 diff + 完整源码 | 每组额外传入 diff 路径、完整源码路径、代码范围 |
| 阶段2 PR 独有 | 越界校验 + 实际行号定位 |
| 报告路径 | `./operators/pr-{pr_number}/` |

## 约束

- 严格按阶段顺序执行，禁止跳步
- 阶段0 的子 Agent 必须在单个消息中并行派发（代码概要 + 设计文档探测总是，API 预研仅 Kernel 侧）；plan-design 在三个子 Agent 全部返回后单独派发
- design-check 发射与否由 plan-design 专项清单决定,workflow 阶段1 只按专项清单执行派发（含 design-check 则与波次1 同消息并行,不进通用分组）
- 阶段2 必须拆分行号校对路由：clause 走 pr-review.line-verify（diff 红线），S1-S7 + D8 ❌ 走 common.line-verify（无红线）
- PR 检视模式下 code-fetch 失败则终止流程
- 禁止提前 Read 未执行阶段的 step 文件
