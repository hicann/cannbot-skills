# PR 检视场景

## 触发
检视 PR、审核 PR、帮我检视这个 PR

---

## 编排

### 任务清单

启动时创建 4 个固定任务（全部 pending）：

| 任务 | 阶段 | 内容 |
|------|------|------|
| 任务0 | 获取 diff + 代码概要 + 条例分组 + API 预研 + 设计文档探测 | code-fetch → 并行派发 code-summarize + clause-routing + api-prestudy（仅 Kernel 侧）+ docs-detect |
| 任务1 | 逐条检视 | 按波次派发检视子 Agent |
| 任务2 | 行号校对 | steps/pr-review.line-verify.md |
| 任务3 | 撰写报告 | steps/common.report-write.md |

### 阶段0：获取 diff + 代码概要 + 条例分组 + 设计文档探测

1. 将任务0 标记为 in_progress
2. 提取 PR 链接，判断托管平台
3. Read + 执行 `steps/pr-review.code-fetch.md` 的派发指令，派发子 Agent 获取 diff 和完整源码
4. 等待 code-fetch 子 Agent 返回（产出 diff_path + repo_path）
5. **快速检测 diff 规模**：
   - Read diff 文件的前 200 行，提取变更文件路径列表，统计总数
   - 若文件数 >10：输出「检测到大型 PR（{N} 个文件），自动切换大型 PR 检视流程」→ 将全部现有任务标记为 deleted → 转至执行 `workflows/pr-large-review.md`（diff_path + repo_path 已就绪，从该 workflow 的阶段0 Step 5 file-split 开始，该 workflow 会创建新的任务清单；并传入 docs_input 供其 design-check）→ 本 workflow 终止
   - 若文件数 ≤10：继续执行下方标准流程
6. **在单个消息中并行派发子 Agent**（A、B、D 总是派发，C 仅当侧别包含 Kernel 时派发）：

- **A 代码概要**：Read+执行 `steps/pr-review.code-summarize.md`，传入 diff 路径 + repo_path + 概要输出路径 `./operators/pr-{pr_number}/code_summary.md`
- **B 条例分组**：Read+执行 `steps/common.clause-routing.md`，传入 代码文件路径 + diff 路径 + 用户意图范围（指定则传类别名，否则空表全量）
- **C API 文档预研**（仅当 diff 含 `op_kernel/` 或判定为 Kernel/混合侧）：Read+执行 `steps/common.api-prestudy.md`，传入 Kernel 侧文件列表（从 repo_path 筛选）+ 输出路径 `./operators/pr-{pr_number}/api_prestudy.md`
- **D 设计文档探测**：Read+执行 `steps/common.docs-detect.md`，传入 repo_path + 用户已指明的文档路径（明确给出则传，否则为空）→ 返回 docs_input（路径/目录或空）

7. 等待所有子 Agent 返回，收集：A→侧别+概要路径；B→分组规划表；C→API 预研报告路径（若已派发）；D→docs_input（路径/目录或空）
8. 将任务0 标记为 done

### 阶段1：逐条检视 + 设计一致性检查

1. 将任务1 标记为 in_progress
2. Read `steps/pr-review.clause-review.md` 获取 prompt 模板
3. 按阶段0 的分组规划表，逐波派发：
   - 每波在单个消息中并行调用 ≤10 个 `Agent` 工具，`subagent_type` 使用 `"general"`
   - 每组用 prompt 模板填入：侧别 + 条例ID + diff路径 + repo_path + 概要路径 + API 预研路径（若存在）
   - **代码范围**：使用 routing 输出中每组的侧别标签（仅Kernel / 仅Tiling / 全部），填入 prompt 的「检视代码范围」字段
   - 波次内并行，波次间串行
4. **🆕design-check 与波次1 同消息并行**：派发**波次1 的那一条消息时**，若 docs_input 非空，在同一消息里额外加入 1 个 `common.design-check` 子 Agent（`subagent_type: "general"`，不进 clause-routing 分组规划，独立输出）。填入 docs_input + diff路径 + repo_path + 概要路径 + API 预研路径（若存在）。**禁止把 design-check 排到所有波次之后单独派发**——它必须与波次1 的 clause 子 Agent 在同一条消息里发出，以实现真正并行
5. 波次2 及之后：仅派发 clause 子 Agent（design-check 已在波次1 并行发出，无需重复）
6. 每波完成后输出进度，所有波次完成后汇总（含 design-check 的 S1-S7 结果）
7. 将任务1 标记为 done

### 阶段2：行号校对

1. 将任务2 标记为 in_progress
2. **拆分输入路由（correctness 关键）**：clause-review 的 FAIL/SUSPICIOUS → Read+执行 `steps/pr-review.line-verify.md`（带 diff 范围红线）；design-check 的 S1-S7 ❌ 项 → Read+执行 `steps/common.line-verify.md`（无 diff 红线，因设计偏差常指向未变更代码）
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
- 阶段0 的子 Agent 必须在单个消息中并行派发（A + B + D 总是，C 仅 Kernel 侧）
- design-check 与 clause-review 波次1 并行派发，但属独立轨道，不进 clause-routing 分组规划
- 阶段2 必须拆分行号校对路由：clause 走 pr-review.line-verify（diff 红线），S1-S7 ❌ 走 common.line-verify（无红线）
- docs_input 为空时不派发 design-check
- PR 检视模式下 code-fetch 失败则终止流程
- 禁止提前 Read 未执行阶段的 step 文件
