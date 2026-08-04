# Audit v4 审计流程

> agent 中心 · 三维度审计（功能完成 / 开发效率 / 开发质量）。独立于 v1-v3，原流程零改动。

## 总览

v4 以**每个 agent**（主 agent + 每个子 agent，含嵌套）为评审单位，从三个维度评判：

| 维度 | 由谁评 | 判据 |
|---|---|---|
| 功能完成情况（completion） | LLM | input 意图 vs actions/output/artifacts；**核心产出未交付→fail** |
| 开发质量（quality） | LLM | 派发标准 vs actions + artifacts + error/retry |
| 开发效率（efficiency） | **服务端确定性**（不靠 LLM） | envelope（latency/tokens/turn/error/retry）阈值映射 |

两步（用户原话）：
1. 确定性提取每个 agent 的输入/输出/信封指标。
2. 用每个 agent 的 I/O 做完成度+质量的审计；效率由信封确定性计算。

## 数据流

```
前端 WorkflowAnalyseTab（选 v4）
  → POST /api/ai/audit-session-py  body={taskId, framework, provider, mode:"v4"}
  → route.ts:
      exportSessionToMarkdown()（仍生成 MD，v4 不用，留作上下文）
      buildAgentIO(taskId, prisma)            ← Step1 确定性提取（TS + Prisma，全保真）
      读 prompts/audit-v4-agent.md / audit-v4-agg.md（Python 自载，热更新）
      POST smart-agent /compress-and-analyze  body 含 agentIo
  → smart-agent server.py（mode=="v4" 分支，跳过压缩）:
      写 {basename}-agentio.json
      run_v4_pipeline(agent_io, ...):
        ┌ Step2a 并行 per-agent LLM 调用（ThreadPoolExecutor, max 6）
        │   每个 agent 一次 call_llm(json_object)，输入=该 agent 的 input/output/actions/artifacts/envelope
        │   → {completion, quality, efficiency:{note?}}
        └ Step2b 一次聚合 LLM 调用
            所有 agent ratings 摘要 + taskQuery → sessionSummary / crossIssues / optimizationPriorities
      validate_v4_schema + inject_v4_merge:
        用 agent_io 覆盖结构字段（envelope/actions/artifacts/name/turns/parentId）
        确定性填 efficiency.rating（_efficiency_rating 阈值）
        LLM 漏 evidence 时 _synthesize_evidence 合成依据兜底
      → NDJSON: extract-start → claude(round, "N/M: id") → claude-done → done → result
  → 前端 audit-job.ts:
      逐行读 NDJSON → progress / result
      结果 stamp _auditMeta{generatedAt, elapsedSec}
      persist(localStorage) + 渲染 <WorkflowAgentAudit>
```

## Step 1：确定性 agent-IO 提取（`src/lib/export/agent-io-export.ts`）

`buildAgentIO(taskId, prisma)` 走 Prisma 构建 agent 树：

- **agent 划分**：一个 agent = 一次 session 级执行。主 agent id=`"main"`；子 agent id=`subagentSessionId`。判据来自 turn 的 `isSubagent`+`subagentSessionId`（opencode 原始 session 层级，cannbot-insight 只读不分配）。
- **父子关系**：靠 `InteractionBridge`（dispatchTurnId → subagentSessionId）。父 agent = dispatch turn 的属主；递归建嵌套树（支持 subagent-of-subagent）。
- **每节点字段**：
  - `inputSummary`：主=首条 user turn；子=bridge.dispatchContent + 派发 Task 的 argsJson 全文。
  - `outputSummary`：主=末条 assistant turn；子=bridge.responseContent。
  - `artifacts`：该 agent 自己 Write/Edit 的文件路径。
  - `actions`：按 turn 顺序的工具调用时间线（`turn/tool/arg截断/state/durationMs/result摘要`，每 agent 上限 80）。
  - `turns`：该 agent 拥有的 turnIndex 列表。
  - `envelope`：latencySec/tokensKt/turnCount/toolCallCount/errorCount/retryCount/reasoningTokensKt。
- 输出扁平 `agents[]`（带 parentId，main 在前），可由 parentId 重建树。

## Step 2：并行 LLM 审计（`smart-agent/trajectory_analyzer.py`）

- **并行 per-agent**（`run_v4_pipeline`）：`ThreadPoolExecutor(max_workers=6)`，每 agent 一个 `call_llm`（json_object）。每 agent 输入小、不截断、覆盖 100%；互不依赖天然并行。失败回退 `n-a`（`call_llm` 自带 3 次重试）。
- **聚合**：所有 agent ratings 摘要 + taskQuery → 一次 LLM → sessionSummary / crossIssues / optimizationPriorities。
- **`inject_v4_merge`**：把 agent_io 的 envelope/actions/artifacts/name/turns/parentId 覆盖到 LLM 产出；确定性填 `efficiency.rating`（保留 LLM 的 note/diagnosis/suggestion）；LLM 漏 evidence 时 `_synthesize_evidence` 用 envelope/actions 合成"（合成依据）"兜底，保证每项 rating 都有依据可追溯。
- prompt 在 `prompts/audit-v4-agent.md`（per-agent）与 `prompts/audit-v4-agg.md`（聚合），Python 每次调用从磁盘读（**热更新，编辑即生效，无需重启**）。

## 评判规则要点（prompt 内）

- **所有 rating 都必填 evidence**（依据），引用 `#turn tool` 或 envelope/artifacts，且与 rating 自洽（pass→有达成证据；fail→有未达成证据）。
- rating 与结论一致：结论"未交付/中途终止"→ 必须 fail，不得 weak。
- 主 agent（编排者）看整体交付：哪怕部分子任务已勾选，整体交付物未完成→fail。
- 不只看 outputSummary 自评"已完成"——子 agent responseContent 常含自夸，必须对照 actions 实际动作。
- diagnosis 带根因前缀：`[skill-defect]/[execution-deviation]/[infra-issue]/[workflow-design]`。
- efficiency.rating 留空串（服务端填），LLM 只可选写 note。

## 效率确定性评级（`_efficiency_rating`）

按 envelope 阈值映射（启发式，可调）：
- `fail`：errorCount≥5 或 retryCount≥5 或 turnCount≥80
- `weak`：errorCount≥2 或 retryCount≥2 或 turnCount≥30
- `pass`：其余

## 渲染（`src/components/observe/WorkflowAgentAudit.tsx`）

- 顶部：sessionSummary + 审计时间（`_auditMeta`）+ 三维度总计数。
- agent 树（按 parentId 重建，可折叠嵌套）：每 agent 卡片 = 三维度 rating 色块（pass 绿/weak 黄/fail 红/n-a 灰）+ **依据**（常驻）+ envelope 指标 + **turns 芯片**（可点击跳 turns tab）+ **动作时间线**（可展开）。
- 子 agent 在父卡片内带竖向连接线 + `↳ <父> 派发 N 个子 agent` 标签。
- 底部：crossIssues（跨 agent 问题）+ optimizationPriorities（优化优先级）。

## 关键文件

| 文件 | 作用 |
|---|---|
| `src/lib/export/agent-io-export.ts` | Step1 确定性 agent-IO 提取（buildAgentIO） |
| `src/lib/audit-job.ts` | v4 分发（mode:"v4"）+ _auditMeta stamp + 类型放宽（Analysis\|V4Analysis） |
| `src/app/api/ai/audit-session-py/route.ts` | v4 时调 buildAgentIO 传 agentIo |
| `smart-agent/server.py` | /compress-and-analyze 的 mode=="v4" 分支 |
| `smart-agent/trajectory_analyzer.py` | run_v4_pipeline（并行）+ validate_v4_schema + inject_v4_merge + _synthesize_evidence + _efficiency_rating |
| `prompts/audit-v4-agent.md` | per-agent 审计 prompt（热更新） |
| `prompts/audit-v4-agg.md` | 聚合 prompt（热更新） |
| `src/components/observe/WorkflowAgentAudit.tsx` | v4 渲染 + V4Analysis 类型 + isV4Analysis 守卫 |

## 已知局限

1. **子 agent session 被重用**：opencode 偶尔往同一子 session 发多次派发但只记一条 InteractionBridge。TurnTimeline 按 subagentSessionId 分组，会把后续回合的 turn 也挂到首次派发的 root turn 块下（turn 的 subagent 归属本身仍正确，只是显示分组粗）。
2. **turnIndex 是分配的**：cannbot-insight 在 ingest 时按 `time_created` 全局排序赋序（`turn-split.ts:198 const turnIndex = i`），非 opencode 原始字段。主+子 turn 按时间交错编号。
3. **效率阈值是启发式**：`_efficiency_rating` 的 error/retry/turn 阈值为经验值，可在 trajectory_analyzer.py 调。
4. **glm 并发**：max_workers=6 硬编码；撞限流靠 `call_llm` 3 次重试兜，可改环境变量化。

## 验证

- `npm run test`：含 `tests/agent-io-export.test.ts`（Step1 真实库）+ `smart-agent/tests/test_v4_pipeline.py`（schema/merge/efficiency 纯逻辑）。
- 端到端：`./start.sh`（起 web + smart-agent）→ Audit tab 选 v4 → 生成 → 进度流 `extract-start → claude(N/M) → claude-done → result` → agent 树 + 三维度 + 依据 + 动作 + turns。
