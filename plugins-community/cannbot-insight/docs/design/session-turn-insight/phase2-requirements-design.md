# Session 对话级洞察 — Phase2 需求设计

版本：v0.2  
最后更新：2026-06-12

> 文档类型：Phase2 需求设计 ｜ 关联项目：CANNBot-Insight  
> 复杂度：**Medium-High**  
> 关联需求分析：[phase1-requirements-analysis.md](phase1-requirements-analysis.md)

---

## 导读

本文是 Phase1 需求分析的后续，聚焦**数据模型设计、API 契约、UI 交互、代码结构**，工程师可直接进入对应章节。

---

## §1 整体架构

### 1.1 篇目标架构概览

```
本地文件导入 (唯一数据入口，暂不实现实时上报)
  → CLI: cannbot-insight import --source opencode-db --path <file>
    → opencode-db adapter: 打开 sessions.db → 查询所有 session 列表
    → 展示 session 选择界面（创建时间、第一个提示词、turn 数、模型等）
    → 用户选择 → 读取选中 session 的 messages → normalize → turn-split
  → CLI: cannbot-insight import --source claude-jsonl --path <file>
    → claude-jsonl adapter: 读取 JSONL → ClaudeParser 解析 → normalize → turn-split
  → API: POST /api/ingest/import-file (multipart form)
    → 同 CLI 逻辑，opencode-db 需先返回 session 列表供前端选择

  → 同一入库 pipeline (turn-split → bridge-builder → execution-split → write-all)

UI reads:
  → /api/observe/data (Session 列表)
  → /api/observe/session?taskId=xxx (Session 详情 + Turn 列表)
  → /api/observe/session/turns?taskId=xxx (Turn 明细)
  → /api/observe/session/turns?taskId=xxx&subagent=true (Subagent Turn 明细)
  → /api/observe/session/bridges?taskId=xxx (交互链路)
```

### 1.2 项目定位

CANNBot-Insight 是**独立项目**，从零开始实现，无外部代码依赖：
- 以 Turn 为一等公民建表，每轮交互独立可查
- Subagent 自动拆出独立 Execution，写入路径实际创建
- 主↔子交互显式建桥记录（InteractionBridge）
- LLM 输入上下文和输出内容独立存储，支持上下文治理

---

## §2 数据模型设计

### 2.1 Prisma Schema（SQLite）

```prisma
datasource db {
  provider = "sqlite"
  url      = env("DATABASE_URL")
}

// ─── Session ────────────────────────────────────────────────
model Session {
  id           String   @id @default(cuid())
  taskId       String   @unique
  label        String?
  query        String?
  framework    String?
  model        String?
  startTime    DateTime @default(now())
  endTime      DateTime?

  // 顶层聚合指标（由 turn 数据派生）
  totalTokens         Int      @default(0)
  totalInputTokens    Int      @default(0)
  totalOutputTokens   Int      @default(0)
  totalReasoningTokens Int    @default(0)
  totalCacheReadTokens Int      @default(0)
  totalCacheWriteTokens Int     @default(0)
  totalCost           Float    @default(0)
  totalLatencyMs      Int      @default(0)      // first→last interaction
  totalToolCallCount  Int      @default(0)
  totalLlmCallCount   Int      @default(0)
  totalSkillLoadCount Int      @default(0)
  totalSubagentCount  Int      @default(0)
  rootExecutionId     String?                   // 主 agent Execution.id

  user        String?
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt

  turns          Turn[]
  executions     Execution[]
  interactionBridges InteractionBridge[]
  skills         SessionSkill[]
}

// ─── Turn（每次对话轮次的独立记录）──────────────────
model Turn {
  id             String   @id @default(cuid())
  sessionId      String
  session        Session  @relation(fields: [sessionId], references: [id], onDelete: Cascade)

  turnIndex      Int                       // 0-based, 在 session 内的顺序号
  role           String                     // "user" | "assistant" | "system" | "tool_result"
  content        String?                    // 文本内容摘要（完整内容存 contentJson）
  contentJson    String?                    // 完整内容 JSON（parts/structured blocks：thinking/text/tool_calls）
  contentSummary String?                    // 折叠态摘要（前 200 字截断，用于默认不展开时展示）
  inputMessagesJson String?                // 发给 LLM 的完整 prompt 消息序列 JSON
                                            // [{role, content, tokenCount?, name?}] — system/user/assistant/tool_result
                                            // 可能很长(10KB~100KB)，用于上下文治理
  inputMessagesCount Int    @default(0)     // inputMessagesJson 中消息条数（折叠态摘要用）
  inputMessagesTokens Int    @default(0)    // 输入消息总 token 数（折叠态摘要 + contextWindowPct 计算用）
  contextWindowPct  Float?                  // inputMessagesTokens / model context window 百分比
  agentName      String?                    // 标识哪个 agent（root/subagent）
  subagentName   String?                    // subagent 显示名
  subagentSessionId String?                 // subagent 专属 session ID

  // ── Token 消耗 ──
  totalTokens        Int      @default(0)
  inputTokens        Int      @default(0)
  outputTokens       Int      @default(0)
  reasoningTokens    Int      @default(0)
  cacheReadTokens    Int      @default(0)
  cacheWriteTokens   Int      @default(0)

  // ── 时间 ──
  createdAt       DateTime?                  // 交互创建时间
  completedAt     DateTime?                  // 交互完成时间
  latencyMs       Int      @default(0)       // created→completed 毫秒数
  ttftMs          Int?                        // time-to-first-token（流式首 token 延迟）

  // ── 模型 ──
  model           String?
  modelId         String?
  providerId      String?
  temperature     Float?
  maxTokens       Int?
  finishReason    String?

  // ── 元数据 ──
  isSubagent      Boolean  @default(false)
  parentExecutionId String?                  // 所属 agent Execution.id

  createdAt       DateTime @default(now())

  toolCalls      ToolCall[]
  skillEvents    SkillEvent[]

  @@index([sessionId, turnIndex])
  @@index([sessionId, isSubagent])
  @@index([subagentSessionId])
  @@index([agentName])
}

// ─── ToolCall（每个工具调用的独立记录）──────────────────
model ToolCall {
  id             String   @id @default(cuid())
  turnId         String
  turn           Turn     @relation(fields: [turnId], references: [id], onDelete: Cascade)

  toolCallId     String                     // LLM 给的 tool_call id
  toolName       String                     // function.name
  argsJson       String?                    // function.arguments JSON
  resultJson     String?                    // tool output/result JSON
  state          String   @default("ok")    // "ok" | "error" | "failed"
  errorType      String?                    // "timeout" | "permission" | "format" | "server_error" | null
  errorMessage   String?                    // 错误信息摘要

  // ── 时间 ──
  startedAt      DateTime?
  completedAt    DateTime?
  durationMs     Int      @default(0)       // started→completed 毫秒数

  // ── 关联 ──
  dispatchBridgeId String?                  // 如果是 task() 调用, 关联到 InteractionBridge
  isSkillRelated  Boolean  @default(false)  // 是否关联 skill 加载/调用

  createdAt       DateTime @default(now())

  @@index([turnId])
  @@index([toolName])
  @@index([toolName, state])
}

// ─── SkillEvent（每个 skill 加载/调用事件）─────────────────
model SkillEvent {
  id             String   @id @default(cuid())
  turnId         String
  turn           Turn     @relation(fields: [turnId], references: [id], onDelete: Cascade)

  skillName      String                     // skill 名称
  skillVersion   Int?                        // 加载时定格的版本号
  eventType      String                     // "load" | "invoke" | "unload"
  success        Boolean  @default(true)    // 是否成功
  errorMessage   String?                    // 失败时的错误信息
  argsJson       String?                    // invoke 时的参数 JSON

  // ── 时间 ──
  startedAt      DateTime?
  completedAt    DateTime?
  durationMs     Int      @default(0)       // 加载/调用耗时毫秒

  createdAt       DateTime @default(now())

  @@index([turnId])
  @@index([skillName])
  @@index([skillName, eventType])
}

// ─── Execution（按 agent 维度聚合）──────────────────
model Execution {
  id              String   @id @default(cuid())
  sessionId       String
  session         Session  @relation(fields: [sessionId], references: [id], onDelete: Cascade)

  // ── Agent 身份 ──
  agentName       String?                    // agent 显示名
  agentSessionId  String?                    // opencode session ID
  isSubagent      Boolean  @default(false)
  subagentType    String?                    // "kuafu" | "general" | null
  subagentName    String?                    // 完整显示名

  // ── 父子关系 ──
  parentExecutionId String?                  // null=root; 否则指父 Execution.id
  rootExecutionId   String?                  // 树的根 Execution.id
  depth            Int      @default(0)      // 层级深度：root=0, sub=1, sub-sub=2...

  // ── 聚合指标 ──
  tokens              Int      @default(0)
  inputTokens         Int      @default(0)
  outputTokens        Int      @default(0)
  reasoningTokens     Int      @default(0)
  cacheReadInputTokens Int     @default(0)
  cacheCreationInputTokens Int  @default(0)
  maxSingleCallTokens Int      @default(0)
  cost                Float    @default(0)    // 写入时定格，不再动态重算
  latencyMs           Int      @default(0)   // first→last 毫秒
  toolCallCount       Int      @default(0)
  toolCallErrorCount  Int      @default(0)
  llmCallCount        Int      @default(0)
  skillLoadCount      Int      @default(0)
  skillInvokeCount    Int      @default(0)

  finalResult     String?
  model           String?

  createdAt       DateTime @default(now())

  // ── 关联 ──
  executionSkills ExecutionSkill[]

  @@index([sessionId])
  @@index([parentExecutionId])
  @@index([rootExecutionId])
  @@index([isSubagent])
  @@index([agentSessionId])
}

// ─── ExecutionSkill（Execution↔skill 绑定）──────────────────
model ExecutionSkill {
  id           String    @id @default(cuid())
  executionId  String
  execution    Execution @relation(fields: [executionId], references: [id], onDelete: Cascade)
  skillName    String
  skillVersion Int?
  isPrimary    Boolean   @default(false)
  user         String?
  createdAt    DateTime  @default(now())

  @@index([skillName, skillVersion])
  @@index([executionId])
}

// ─── SessionSkill（Session↔skill 绑定，全 session 籇度）────────────
model SessionSkill {
  id           String    @id @default(cuid())
  sessionId    String
  session      Session  @relation(fields: [sessionId], references: [id], onDelete: Cascade)
  skillName    String
  skillVersion Int?
  invocationCount Int   @default(0)  // 调用次数
  user         String?
  createdAt    DateTime  @default(now())

  @@unique([sessionId, skillName])
  @@index([skillName])
}

// ─── InteractionBridge（主agent↔subagent 交互链路）──────────
model InteractionBridge {
  id             String   @id @default(cuid())
  sessionId      String
  session        Session  @relation(fields: [sessionId], references: [id], onDelete: Cascade)

  // ── dispatch 侧（父 agent） ──
  dispatchExecutionId  String                  // 父 Execution.id
  dispatchTurnId       String?                 // 父哪个 turn 发出 task()
  dispatchToolCallId   String?                 // 具体哪个 tool_call（task() 调用）
  dispatchContent      String?                 // 摘要：发给 subagent 的指令/参数
  dispatchTimestamp    DateTime?               // 发出时间

  // ── response 侧（子 agent） ──
  responseExecutionId  String?                 // 子 Execution.id
  responseTurnId       String?                 // 子哪个 turn 返回结果
  responseContent      String?                 // 摘要：子 agent 返回的内容
  responseTimestamp    DateTime?               // 返回时间

  // ── 链路元数据 ──
  subagentSessionId    String?                 // 子 agent 的 session ID
  subagentType         String?                 // subagent 类型
  subagentName         String?                 // subagent 显示名
  status               String   @default("dispatched") // "dispatched" | "running" | "completed" | "failed" | "timeout"

  // ── 子 agent 转发指标 ──
  subagentTokens       Int      @default(0)
  subagentLatencyMs    Int      @default(0)  // 从 dispatch 到 response 的总耗时

  createdAt       DateTime @default(now())
  updatedAt       DateTime @updatedAt

  @@index([sessionId])
  @@index([dispatchExecutionId])
  @@index([responseExecutionId])
  @@index([subagentSessionId])
  @@index([status])
}
```

### 2.2 数据流详解

#### 上传拆解流程

```
POST /api/ingest/upload {task_id, interactions[]}
  │
  ├─ normalizeInteractions() → RawInteraction[]
  │
  ├─ 挆 session 是否已存在
  │   ├─ 新 session: 创建 Session 行
  │   └─ 已存在: mergeSessionInteractionsMonotonic() 合并
  │
  ├─ turn-split: 按 role 拆解每个 RawInteraction → Turn[]
  │   ├─ 每个 Turn: 计算 token/时间/模型字段
  │   ├─ 每个 Turn.tool_calls: 拆解 → ToolCall[]
  │   ├─ 每个 Turn 中 skill 相关: 拆解 → SkillEvent[]
  │   ├─ 每个 assistant Turn: 重构 inputMessages → inputMessagesJson
  │   │   ├─ 从 preceding interactions 重建 request messages (system + history + tool results)
  │   │   ├─ 计算 inputMessagesCount / inputMessagesTokens / contextWindowPct
  │   │   ├─ 生成 contentSummary (前 200 字截断)
  │
  ├─ bridge-builder: 扫描 task() tool_calls + 对应 subagent turns
  │   ├─ 每个 task() 调用 → InteractionBridge (dispatch 侧)
  │   ├─ 匹配 subagent_session_id → 填充 response 侧
  │   ├─ 计算 subagent 转发指标（token/耗时）
  │
  ├─ execution-split: 按 agent 维度拆分 Execution 行
  │   ├─ root agent → Execution(isSubagent=false)
  │   ├─ 每个 subagent → Execution(isSubagent=true, parentExecutionId=root.id)
  │   ├─ 每层聚合: sum(turn token/时间/工具数/skill数)
  │
  ├─ write-all:
  │   ├─ Upsert Session (聚合指标)
  │   ├─ Upsert Turn[] (per-turn)
  │   ├─ Upsert ToolCall[] (per-tool-call)
  │   ├─ Upsert SkillEvent[] (per-skill-event)
  │   ├─ Upsert InteractionBridge[] (交互链路)
  │   ├─ Upsert Execution[] (per-agent 聚合)
  │   ├─ Upsert ExecutionSkill[] (per-execution skill 绑定)
  │   ├─ Upsert SessionSkill[] (per-session skill 绑定)
```

#### 关键设计决策

1. **Turn 行独立存储**而非 JSON blob：每个 turn 有独立 DB 行，可索引、可查询、可聚合、无需解析 JSON。
2. **ToolCall 行独立存储**：工具调用有自己的行，可按 name 查询、按 error 分类统计。
3. **SkillEvent 行独立存储**：skill 加载/调用是独立事件，区分 load/invoke/unload 三种类型。
4. **InteractionBridge 显式链路**：主→子交互不再是隐含推断，而是显式记录 dispatch→response 全链路。
5. **Execution 行按 agent 拆分**：每个 subagent 有独立 Execution 行，聚合指标可直接查询。
6. **cost 写入时定格**：不再依赖读取时动态计算，避免 pricing 表变更导致历史 cost 漂移。
7. **Session 聚合指标派生**：Session 的总量指标由 turn 数据 SUM 派生，保持一致性。
8. **LLM 输入上下文独立存储**：assistant turn 的 inputMessagesJson 存完整 prompt 消息序列（system + history + tool results），用于上下文治理。单独字段而非嵌入 contentJson，便于独立查询和折叠展示。
9. **长内容默认折叠**：inputMessagesJson 和 contentJson 可能极长，Turn 行同时存 inputMessagesCount/inputMessagesTokens/contentSummary 等摘要字段，UI 默认展示摘要、点击展开全文。

---

## §3 API 契约设计

### 3.1 写入端（本地文件导入，唯一数据入口）

| 路径 | 方法 | 描述 | 入站 | 出站 |
|-|-|-|-|-|
| `/api/ingest/import-file` | POST | 本地文件导入（multipart form） | `{source: "opencode-db" | "claude-jsonl", file: binary, sessionId?: string}` | `{sessionId[], importedCount, errors[]}` |
| `/api/ingest/import-file/sessions` | POST | Opencode DB session 列表查询（第一步） | `{source: "opencode-db", file: binary}` | `{sessions: [{id, createdAt, firstQuery, turnCount, model}]}` |

导入流程:
1. **opencode-db**：上传文件 → 读取 DB → 返回 session 列表 → 用户选择 → 读取选中 session 数据 → adapter 解析 → turn-split → bridge-builder → execution-split → write-all
2. **claude-jsonl**：上传文件 → 直接解析入库 → turn-split → bridge-builder → execution-split → write-all

### 3.2 读取端

| 路径 | 方法 | 描述 | 核心查询参数 |
|-|-|-|-|
| `/api/observe/data` | GET | Session 列表（分页） | `?page=1&pageSize=20&isSubagent=false&user=xxx` |
| `/api/observe/session` | GET | Session 详情（含聚合指标） | `?taskId=xxx` |
| `/api/observe/session/turns` | GET | Session 的 Turn 列表 | `?taskId=xxx&isSubagent=false&role=assistant` |
| `/api/observe/session/turns/:turnId` | GET | 单个 Turn 详情（含 contentJson + inputMessagesJson） | — |
| `/api/observe/session/bridges` | GET | Session 的 InteractionBridge 列表 | `?taskId=xxx` |
| `/api/observe/executions` | GET | Session 的 Execution 列表（含 subagent） | `?taskId=xxx` |
| `/api/observe/executions/:executionId` | GET | 单个 Execution 详情 | — |
| `/api/observe/stats` | GET | 快速聚合统计 | `?taskId=xxx` (返回 token/耗时/cost 汇总) |

#### 核心 API 响应结构

**Session 列表** (`/api/observe/data`)
```json
{
  "items": [
    {
      "sessionId": "...",
      "taskId": "...",
      "query": "...",
      "startTime": "...",
      "endTime": "...",
      "totalTokens": 15000,
      "totalCost": 0.45,
      "totalLatencyMs": 120000,
      "totalToolCallCount": 8,
      "totalSkillLoadCount": 2,
      "totalSubagentCount": 1,
      "model": "...",
      "user": "..."
    }
  ],
  "total": 100,
  "page": 1
}
```

**Session 详情** (`/api/observe/session`)
```json
{
  "sessionId": "...",
  "taskId": "...",
  "query": "...",
  "startTime": "...",
  "endTime": "...",
  "totalTokens": 15000,
  "totalCost": 0.45,
  "agents": [
    {
      "executionId": "...",
      "agentName": "root",
      "isSubagent": false,
      "tokens": 12000,
      "cost": 0.36,
      "toolCallCount": 6,
      "skillLoadCount": 2,
      "subagentCount": 1
    },
    {
      "executionId": "...",
      "agentName": "Kuafu",
      "isSubagent": true,
      "parentExecutionId": "...",
      "tokens": 3000,
      "cost": 0.09,
      "toolCallCount": 2,
      "skillLoadCount": 0
    }
  ],
  "skills": [
    {"skillName": "agent-debug-diagnosis", "version": 3, "invocationCount": 1}
  ]
}
```

**Turn 列表** (`/api/observe/session/turns`)
```json
{
  "items": [
    {
      "turnId": "...",
      "turnIndex": 0,
      "role": "user",
      "contentSummary": "用户查询...",
      "agentName": "root",
      "isSubagent": false,
      "totalTokens": 0,
      "inputMessagesCount": 0,
      "inputMessagesTokens": 0,
      "latencyMs": 0,
      "createdAt": "...",
      "toolCalls": [],
      "skillEvents": []
    },
    {
      "turnId": "...",
      "turnIndex": 1,
      "role": "assistant",
      "contentSummary": "助手回复摘要前200字...",
      "inputMessagesCount": 5,
      "inputMessagesTokens": 3000,
      "contextWindowPct": 15.0,
      "agentName": "root",
      "agentName": "root",
      "isSubagent": false,
      "totalTokens": 500,
      "inputTokens": 300,
      "outputTokens": 200,
      "latencyMs": 3500,
      "createdAt": "...",
      "completedAt": "...",
      "model": "gpt-4o",
      "toolCalls": [
        {"toolCallId": "...", "toolName": "bash", "state": "ok", "durationMs": 1200}
      ],
      "skillEvents": []
    }
  ]
}
```

**InteractionBridge 列表** (`/api/observe/session/bridges`)
```json
{
  "items": [
    {
      "bridgeId": "...",
      "dispatchExecutionId": "...",
      "dispatchContent": "分析 vmcore 文件...",
      "dispatchTimestamp": "...",
      "responseExecutionId": "...",
      "responseContent": "诊断结果：...",
      "responseTimestamp": "...",
      "subagentName": "Kuafu",
      "status": "completed",
      "subagentTokens": 3000,
      "subagentLatencyMs": 45000
    }
  ]
}
```

---

## §4 UI 交互设计

### 4.1 页面结构

```
/observe                     → Session 列表页
/observe/:taskId             → Session 详情页
  ├── Overview Tab           → 聚合指标 + Agent 概览 + Skill 概览
  ├── Turns Tab              → Turn 列表（支持 filter: agent/role/tool/skill）
  ├── Timeline Tab           → 甘特图时间线（turn → tool → skill → subagent dispatch）
  ├── Subagents Tab          → Subagent 详情（每个 subagent 卡片 + 交互链路）
  ├── Skills Tab             → Skill 调用明细
  └── Interactions Tab       → 交互链路图（主↔子通信可视化）
```

### 4.2 Session 详情页 — Overview Tab

- 顶部：聚合指标卡片（总 token、总耗时、总 cost、工具调用数、skill 数、subagent 数）
- 中部：Agent 概览（root + 每个 subagent 的小卡片，含各自的 token/耗时/cost）
- 底部：Skill 概览（每个 skill 的调用次数 + 版本）

### 4.3 Session 详情页 — Turns Tab

- 左侧：Turn 时间线列表
  - 每个 turn 显示：角色 badge + agent badge + token 数 + 耗时 + 工具调用摘要
  - 过滤器：agent（root/subagent name）、role、有/无 tool call、有/无 skill event
- 右侧：Turn 详情面板（选中 turn 后展示）
  - Content 区：完整内容（markdown 渲染 + 折叠）
  - **上下文区（LLM Input，默认折叠）**：
    - 折叠态：显示"输入 N 条消息，共 X tokens（占 context window Y%）"摘要行 + 展开按钮
    - 展开态：逐条显示发给 LLM 的消息列表，每条消息显示：
      - role badge（system=紫色 / user=蓝色 / assistant=灰色 / tool_result=绿色）
      - token 数（如有）
      - 消息内容（默认折叠超过 200 字的消息，点击可独立展开）
    - 用途：上下文治理——理解 LLM 收到了什么上下文、上下文是否过长、是否包含冗余/误导信息
  - **输出区（LLM Output，默认折叠）**：
    - 折叠态：显示 contentSummary（前 200 字）+ "输出 X tokens" + 展开按钮
    - 展开态：完整内容渲染
      - thinking/reasoning blocks 独立折叠区
      - text 部分 markdown 渲染
      - tool_calls 以 badge 形式嵌入
    - 用途：理解 LLM 产出了什么、输出质量如何
  - Token 区：input/output/reasoning/cache 条形图
  - Tools 区：工具调用列表（name + duration + state + 展开看 args/result）
  - Skills 区：skill 事件列表（name + type + duration + success）
  - Timing 区：created/completed/latency/ttft
  - Model 区：model/modelId/provider/temperature/finishReason

### 4.4 Session 详情页 — Timeline Tab

- 甘特图时间线：
  - 行：按 agent 分组（root / subagent）
  - 列：每个 turn 的时间段 + 内嵌 tool/skill/subagent dispatch 的子段
  - 颜色区分：LLM 思考（蓝）、工具调用（绿）、skill 加载（黄）、subagent dispatch（橙）
  - 悬浮：显示 token 数 + 耗时

### 4.5 Session 详情页 — Subagents Tab

- 每个 subagent 一个独立卡片：
  - 聚合指标（token/耗时/cost/工具数/skill数）
  - Turn 概览（该 subagent 的所有 turn 摘要）
  - 交互链路（该 subagent 与主 agent 的 InteractionBridge）
- 点击可展开该 subagent 的 Turn 详情（与 Turns Tab 共用详情面板组件）

### 4.6 Session 详情页 — Interactions Tab

- 交互链路图：
  - 左侧：主 agent 的 task() dispatch 事件列表
  - 中间：连接线（dispatch → response）
  - 右侧：subagent 的 response 事件列表
  - 每条连接线标注：status + subagent 耗时 + token 数
  - 点击连接线：弹出详细 dispatch content + response content

### 4.7 Session 列表页

- 表格列：taskId、query、startTime、endTime、totalTokens、totalCost、totalLatencyMs、model、user、subagentCount
- 快速筛选：按 user、model、有无 subagent、有无 skill

---

## §5 代码结构设计

### 5.1 目录结构

```
cannbot-insight/
├── bin/
│   └─ cli.js                           # CLI 入口：cannbot-insight import --source <type> --path <file>
├── prisma/
│   └── schema.prisma                    # §2 定义的数据模型
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   ├── ingest/
│   │   │   │   ├── import-file/route.ts # 本地文件导入入口（multipart）
│   │   │   │   └── import-file/sessions/route.ts # opencode DB session 列表查询（选择前展示）
│   │   │   └── observe/
│   │   │       ├── data/route.ts        # Session 列表
│   │   │       ├── session/route.ts     # Session 详情
│   │   │       ├── session/turns/route.ts  # Turn 列表
│   │   │       ├── session/bridges/route.ts # Bridge 列表
│   │   │       ├── executions/route.ts  # Execution 列表
│   │   │       └── stats/route.ts       # 聚合统计
│   │   ├── observe/
│   │   │   ├── page.tsx                 # Session 列表页
│   │   │   └── [taskId]/page.tsx        # Session 详情页
│   │   └── layout.tsx
│   │   └─ globals.css
│   ├── lib/
│   │   ├── ingest/
│   │   │   ├── normalize.ts             # 交互归一化
│   │   │   ├── turn-split.ts            # turn 拆解器
│   │   │   ├── bridge-builder.ts        # InteractionBridge 构建器
│   │   │   ├── execution-split.ts       # Execution 拆分器
│   │   │   ├── merge.ts                # 增量上传合并
│   │   │   └── adapters/               # 数据源适配器（本地文件解析）
│   │   │       ├── opencode-db.ts       # opencode sessions.db SQLite 读取器
│   │   │       ├── claude-jsonl.ts      # Claude Code session JSONL 解析器
│   │   │       └── index.ts             # 适配器注册表（按 source type 路由）
│   │   ├── storage/
│   │   │   ├── data-service.ts          # 数据读写服务
│   │   │   └─ cost-calculator.ts        # cost 计算（写入时定格）
│   │   └── shared/
│   │       ├── types.ts                 # 共享类型定义
│   │       └── constants.ts             # 常量/配置
│   ├── components/
│   │   ├── observe/
│   │   │   ├── SessionList.tsx          # Session 列表
│   │   │   ├── SessionOverview.tsx      # 详情页 Overview Tab
│   │   │   ├── TurnTimeline.tsx         # Turns Tab 时间线
│   │   │   ├── TurnDetail.tsx           # Turn 详情面板
│   │   │   ├── LlmContextView.tsx       # LLM 输入上下文展示（默认折叠，逐条消息可独立展开）
│   │   │   ├── LlmOutputView.tsx        # LLM 输出内容展示（默认折叠，thinking/text 分区）
│   │   │   ├── TimelineGantt.tsx        # 甘特图时间线
│   │   │   ├── SubagentCards.tsx        # Subagent Tab
│   │   │   ├── InteractionGraph.tsx     # 交互链路图
│   │   │   ├── SkillDetail.tsx          # Skills Tab
│   │   │   └─ MetricCards.tsx           # 指标卡片
│   │   └─ ui/                           # shadcn/ui 基础组件
│   └─ instrumentation.ts               # Next.js 启动钩子
├── docs/
│   └── design/                          # 设计文档
├── package.json
└── tsconfig.json
```

### 5.2 核心模块职责

| 模块 | 职责 |
|-|-|
| `normalize.ts` | 将原始上传的 flat messages 归一化为标准 RawInteraction 格式；识别框架类型 |
| `turn-split.ts` | 将 RawInteraction[] 拆解为 Turn[] + ToolCall[] + SkillEvent[]；计算每项的 token/时间字段；识别 skill 相关工具调用 |
| `bridge-builder.ts` | 扫描 task() tool_calls 和 subagent turns，构建 InteractionBridge[]；匹配 dispatch→response；计算 subagent 转发指标 |
| `execution-split.ts` | 按 agent 维度拆分 Execution 行（root + 每个 subagent）；计算每层聚合指标；建立父子关系 |
| `data-service.ts` | 协调写入：创建 Session → 创建 Turn/ToolCall/SkillEvent → 创建 InteractionBridge → 创建 Execution/ExecutionSkill → 更新 Session 聚合指标 |
| `cost-calculator.ts` | 模型 pricing 表 + cache 定价；写入时计算 cost 并定格存储 |
| `opencode-db.ts` | 读取 opencode sessions.db SQLite 文件：用 better-sqlite3 打开 DB → **第一步：查询所有 session 列表**（返回 id/创建时间/第一个用户提示词/turn 数/模型名供用户选择）→ **第二步：读取用户选中的 session 的 messages/tools 数据** → 转换为 RawInteraction[] 格式 → 交给 normalize + turn-split pipeline |
| `claude-jsonl.ts` | 读取 Claude Code session JSONL 文件：逐行解析 JSON → 提取 message/tool_use/tool_result → 转换为 RawInteraction[] 格式 → 交给 normalize + turn-split pipeline |
| `adapters/index.ts` | 适配器注册表：按 source type（"opencode-db"/"claude-jsonl"）路由到对应解析器；opencode-db adapter 提供两个接口：`listSessions(filePath)` → session 列表 + `readSession(filePath, sessionId)` → RawInteraction[]；claude-jsonl adapter 提供 `readFile(filePath)` → RawInteraction[] |

### 5.3 turn-split.ts 核心逻辑

```
输入: RawInteraction[]
输出: {turns: TurnInput[], toolCalls: ToolCallInput[], skillEvents: SkillEventInput[]}

对每个 RawInteraction:
   1. 计算 turnIndex（递增）
   2. 提取 role, content, contentJson
   3. 提取 agentName/subagentName/subagentSessionId/isSubagent
   4. 提取 usage → totalTokens/inputTokens/outputTokens/reasoningTokens/cacheReadTokens/cacheWriteTokens
   5. 提取 timeInfo → createdAt/completedAt/latencyMs
   6. 提取 model 元数据 → model/modelId/providerId/temperature/maxTokens/finishReason
   7. 遍历 tool_calls:
      a. 创建 ToolCallInput: toolCallId/toolName/argsJson/resultJson/state/startedAt/completedAt/durationMs
      b. 识别 skill 相关:
         - toolName === "skill" → 创建 SkillEventInput(eventType="invoke", skillName=args.name)
         - toolName === "load_skill" → 创建 SkillEventInput(eventType="load", skillName=args.skill_name)
         - toolName === "task" → 标记 isSkillRelated=false, 留给 bridge-builder
      c. 推算 errorType: 如果 state=error, 从 errorMessage 分类(timeout/permission/format/server_error)
   8. 识别 skill 事件中的 version: 从 args 或从 Skill.activeVersion 快照
   9. 对于 role=assistant 的 Turn:
      a. 重构 inputMessages: 从 preceding interactions (system prompt + 对话历史 + tool results)
         → 生成 inputMessagesJson [{role, content, tokenCount?}]
      b. 计算 inputMessagesCount (消息条数)
      c. 计算 inputMessagesTokens (≈ inputTokens from usage)
      d. 计算 contextWindowPct = inputMessagesTokens / model context window size
      e. 生成 contentSummary: 从 content/contentJson 截断前 200 字
```

### 5.4 bridge-builder.ts 核心逻辑

```
输入: RawInteraction[], ToolCallInput[]
输出: InteractionBridgeInput[]

1. 收集所有 task() tool_calls → potential dispatches
   - 每个 task() call → {dispatchToolCallId, dispatchContent=args.summary, dispatchTimestamp=startedAt}
2. 收集所有 subagent turns → potential responses
   - 按 subagent_session_id 分组
   - 每个 subagent 的最后一个 turn → {responseContent=content, responseTimestamp=completedAt}
3. 匹配: task() args.subagent_session_id === turn.subagentSessionId
   - 如果 args 无 session_id → 按时间顺序最近原则匹配
4. 构建 InteractionBridgeInput:
   - dispatch 侧: 从 task() tool_call
   - response 侧: 从匹配的 subagent turns
   - subagent 聚合: sum(subagent turns tokens), max(response)-min(dispatch) → latency
5. 状态: 
   - 有 response → "completed"
   - subagent turns 有 error → "failed"
   - 无 response 且超时 → "timeout"
   - 无 response 且未超时 → "dispatched"
```

---

## §6 设计决策总结

| 维度 | 传统做法 | CANNBot-Insight 设计 | 改进点 |
|-|-|-|-|
| Turn 数据存储 | Session.interactions JSON blob（需全量解析） | Turn 独立 DB 行（可索引可查询） | 无需全量解析 JSON；支持 per-turn 过滤和聚合 |
| Per-turn token | 存在 JSON blob 中，Execution 只有聚合 | Turn 行含 input/output/reasoning/cache | 可直接查"第3轮用了多少token" |
| Per-turn timing | timeInfo 在 JSON blob 中，Execution 只有 latency | Turn 行含 createdAt/completedAt/latencyMs/ttftMs | 可直接查每轮耗时；新增 TTFT |
| 工具调用存储 | tool_calls 嵌在 interaction JSON 中 | ToolCall 独立 DB 行 | 可按 toolName 查询；可按 errorType 分类统计 |
| 工具调用错误 | 只有 state=error/failed | 增加 errorType 分类 | 可统计"多少次timeout vs permission error" |
| Skill 加载事件 | 无独立记录，靠 tool_calls 中隐含推断 | SkillEvent 独立 DB 行，区分 load/invoke/unload | 可追踪 skill 加载成功/失败/耗时 |
| Skill 调用次数 | ExecutionSkill 只记绑定，不记次数 | SessionSkill 记 invocationCount | 可统计"skill X 被调了 5 次" |
| Subagent Execution | schema 有字段但未实际写入 | 执行 execution-split 硬创建子行 | 每个 subagent 有独立聚合指标 |
| 主↔子交互 | 隐含在 task() tool_call JSON 中 | InteractionBridge 显式链路 | 可直观查看 dispatch→response 全链路 |
| Cost 存储 | 读取时动态计算（会漂移） | 写入时定格存储 | 历史 cost 不随 pricing 变更漂移 |
| TTFT | 无 | Turn.ttftMs | 可分析流式首 token 延迟 |
| 工具调用结果大小 | 无 | ToolCall.resultJson 可计算大小 | 可分析"bash 输出了 50KB" |
| LLM 输入上下文存储 | 不在 DB 独立存储 | Turn.inputMessagesJson + inputMessagesCount/Tokens | 上下文治理：可直接查"第5轮 LLM 收到了什么" |
| LLM 输出折叠展示 | 全量渲染 | contentSummary 折叠态 + 点击展开 | 长内容不自动展开，减少视觉噪音 |
| 上下文长度指标 | 动态计算 | Turn.contextWindowPct 写入定格 | 可判断"上下文是否已占满 context window" |
