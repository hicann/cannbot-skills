# Workflow (AI) 设计方案

## 1. 核心理念

与正则解析方案不同，本方案**让 AI 理解 session 数据**，由 AI 输出结构化的工作流分析结果，前端根据 AI 的输出渲染可视化。

### 对比

| 维度 | 正则解析方案 | AI 分析方案 |
|------|-------------|-------------|
| 阶段识别 | 正则匹配"阶段一"等关键词 | AI 从对话上下文推理 |
| 适用范围 | 仅适用于有明确标记的工作流 | 适用于任何 agent session |
| 灵活性 | 需要为每种工作流写正则 | 一套 prompt 适配所有 |
| 成本 | 零 | 每次分析消耗 tokens |
| 速度 | 毫秒级 | 秒级（取决于数据量） |
| 分析深度 | 仅提取标签 | 可理解语义、因果关系、瓶颈 |
| 持久化 | 写入 DB | 写入 DB（AI 分析结果缓存） |

### 最佳实践：两者结合

```
导入 session
    │
    ├──▶ 正则解析（快速，零成本）→ 提取明确的阶段/checkpoint 标记
    │
    └──▶ AI 分析（深度，可选）→ 补充语义理解、因果关系、瓶颈识别
              │
              ▼
         合并两者结果 → 渲染 Workflow (AI) 页面
```

---

## 2. 整体流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户操作流程                               │
└─────────────────────────────────────────────────────────────────┘

Step 1: 配置 API Key
  ┌──────────────────────────────────────────────┐
  │ Settings 页面（新）                           │
  │  Provider: [Alibaba Cloud ▾]                 │
  │  Base URL: [https://token-plan...       ]     │
  │  API Key:  [sk-xxxxx••••••••            ]     │
  │  Model:    [glm-5                       ]     │
  │                                              │
  │  [保存]  [测试连接]                            │
  └──────────────────────────────────────────────┘

Step 2: 导入 Session（已有功能）
  → 导入 opencode db → 写入 Prisma DB

Step 3: 进入 Session 详情页 → 点击 [Workflow (AI)] Tab
  │
  ├── 情况 A: 已有 AI 分析缓存 → 直接渲染
  │
  └── 情况 B: 未分析 → 显示 "点击 AI 分析" 按钮
        │
        ▼
Step 4: 用户点击 [🤖 AI 分析]
  │
  ├── 前端 → POST /api/ai/analyze-workflow
  │          携带 { taskId, options }
  │
  ├── 后端：
  │   1. 从 DB 读取 session 的 turns + bridges + toolCalls
  │   2. 构建 prompt（session 摘要 + turns 内容 + bridges）
  │   3. 调用 LLM API
  │   4. 解析 LLM 返回的 JSON
  │   5. 写入 DB（WorkflowAIResult 表）
  │   6. 返回结构化数据给前端
  │
  └── 前端：渲染 Workflow (AI) 可视化

Step 5: 用户交互
  → 点击节点展开详情
  → 切换视图（树形 / 流程图 / 时间线）
  → 重新分析（如修改了 prompt 模板）
```

---

## 3. 数据模型

### 3.1 新增 Prisma Models

```prisma
// ─── AI Provider 配置 ──────────────────────────────────────
model AIProvider {
  id        String   @id @default(cuid())
  name      String                      // "Alibaba Cloud" / "OpenAI" / "自定义"
  provider  String                      // "alibaba-cn" / "openai" / "custom"
  baseUrl   String                      // API base URL
  apiKey    String                      // 加密存储
  model     String                      // "glm-5" / "gpt-4o" / etc.
  isActive  Boolean  @default(true)     // 当前激活的配置

  // 分析参数
  maxTokens     Int     @default(8192)
  temperature   Float   @default(0.3)
  promptTemplate String?                // 可自定义 prompt 模板

  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt

  results   WorkflowAIResult[]

  @@unique([name, provider])
}

// ─── AI 分析结果 ──────────────────────────────────────────
model WorkflowAIResult {
  id            String   @id @default(cuid())
  sessionId     String
  session       Session  @relation(fields: [sessionId], references: [id], onDelete: Cascade)
  providerId    String
  provider      AIProvider @relation(fields: [providerId], references: [id])

  // 分析状态
  status        String   @default("pending")  // pending / analyzing / completed / failed
  errorMessage  String?

  // 原始数据
  promptTokens    Int    @default(0)
  responseTokens  Int    @default(0)
  analysisCost    Float  @default(0)
  analysisDurationMs Int @default(0)

  // AI 分析结果（JSON）
  analysisJson    String?                    // 完整的 AI 输出 JSON

  // 摘要字段（从 analysisJson 提取的快捷字段）
  phaseCount      Int    @default(0)
  stepCount       Int    @default(0)
  bottleneckCount Int    @default(0)

  // AI 生成的自然语言总结
  summary         String?

  // 分析版本（支持多次重新分析）
  version         Int    @default(1)
  isLatest        Boolean @default(true)

  createdAt     DateTime @default(now())
  updatedAt     DateTime @updatedAt

  @@index([sessionId])
  @@index([sessionId, isLatest])
}

// Session model 扩展
model Session {
  // ... 现有字段 ...
  workflowAIResults WorkflowAIResult[]
}
```

### 3.2 AI 输出的 JSON Schema

AI 返回的结构化数据定义：

```typescript
interface WorkflowAIAnalysis {
  // 工作流概览
  overview: {
    workflowType: string;          // "算子开发" / "代码重构" / "调试修复" / "通用对话"
    description: string;           // AI 生成的工作流描述
    totalDuration: string;         // "10小时10分钟"
    activeTime: string;            // "2小时30分钟"
    waitTime: string;              // "7小时40分钟"
    efficiency: number;            // 0-100, 有效工作时间占比
  };

  // 阶段列表
  phases: AIPhase[];

  // 瓶颈分析
  bottlenecks: AIBottleneck[];

  // 关键洞察
  insights: string[];

  // 改进建议
  suggestions: string[];
}

interface AIPhase {
  phaseIndex: number;
  name: string;                    // "需求与设计"
  fullLabel: string;               // "阶段一：需求与设计阶段"
  description: string;             // AI 对这个阶段的描述
  startTime: string;               // ISO timestamp
  endTime: string;                 // ISO timestamp

  // 聚合指标
  tokens: number;
  cost: number;
  toolCallCount: number;

  // 子步骤
  steps: AIStep[];

  // 检查点（如果有）
  checkpoints: AICheckpoint[];

  // 迭代组（如果有）
  iterations?: AIIteration[];
}

interface AIStep {
  stepIndex: number;
  name: string;                    // "开发准备"
  label: string;                   // "1.1 开发准备"
  description: string;             // AI 对这一步的描述

  // 关联信息
  turnIndex: number;               // 对应的 turn 序号（用于前端跳转）
  bridgeId: string | null;         // 关联的 bridge

  // Subagent 信息
  subagentType: string | null;     // "general" / "ascendc-kernel-developer"
  subagentName: string | null;

  // 指标
  tokens: number;
  cost: number;
  durationMs: number;
  status: "completed" | "failed" | "running" | "skipped";

  // 并行信息
  parallelGroupId: string | null;
  parallelWith: string[];          // 并行执行的 step names

  // AI 评价
  efficiency: "high" | "medium" | "low";
  note: string | null;             // AI 对这步的备注
}

interface AICheckpoint {
  name: string;                    // "CP1 需求分析确认"
  type: "block" | "info" | "review";
  requestedAt: string;
  approvedAt: string | null;
  waitTimeMs: number;
  summaryContent: string | null;   // agent 展示的摘要
  userResponse: string | null;     // 用户回应
  turnIndex: number;
}

interface AIIteration {
  iterationIndex: number;
  name: string;                    // "迭代一：骨架搭建"
  description: string;
  steps: AIStep[];                 // 此迭代的 steps
  totalTokens: number;
  totalCost: number;
  durationMs: number;
  outcome: string;                 // AI 对迭代结果的总结
}

interface AIBottleneck {
  type: "wait" | "retry" | "excessive_tokens" | "error_loop" | "redundant";
  severity: "high" | "medium" | "low";
  description: string;
  location: string;                // "阶段一 / CP1" 或 "迭代二 / UT开发"
  timeImpact: string;              // "浪费 3.5 小时"
  tokenImpact: string;             // "多消耗 2M tokens"
  suggestion: string;              // AI 的改进建议
}
```

---

## 4. Prompt 设计

### 4.1 System Prompt

```
你是一个 AI Agent 工作流分析专家。你的任务是分析一个 Coding Agent 的执行过程，
理解它的工作流程，并输出结构化的分析结果。

你将收到以下数据：
1. Session 基本信息（总 tokens、cost、duration、tool calls 等）
2. Turn 列表（每条对话的 role、content 摘要、tokens、tools、时间戳）
3. Subagent dispatch 记录（父 agent 给子 agent 的任务描述和返回结果）

请分析并输出 JSON 格式的结果，包含：
- 工作流阶段划分
- 每个阶段的步骤分解
- 检查点（用户确认点）
- 迭代轮次
- 并行任务识别
- 瓶颈分析
- 改进建议

输出要求：
1. 阶段划分基于对话内容的语义理解，不仅依赖关键词
2. 步骤应关联到具体的 turn index 和 bridge
3. 并行任务通过时间重叠或显式标记识别
4. 瓶颈分析要具体，指出时间/token浪费的位置和原因
5. 严格输出 JSON，不要包含 markdown 代码块标记
```

### 4.2 User Prompt 模板

```typescript
function buildAnalysisPrompt(session: SessionData, turns: TurnData[], bridges: BridgeData[]): string {
  // 构建紧凑的 session 摘要（控制 token 消耗）
  const sessionSummary = buildSessionSummary(session);
  const turnDigests = buildTurnDigests(turns);      // 精简版 turn 数据
  const bridgeDigests = buildBridgeDigests(bridges); // 精简版 bridge 数据

  return `
## Session 基本信息
${sessionSummary}

## Turn 列表 (${turns.length} 条)
每条格式: [#序号] [role] [agent] [time] [tokens_in/tokens_out] [tools] [content前200字]
${turnDigests}

## Subagent Dispatch 记录 (${bridges.length} 条)
每条格式: [dispatch#] [时间] [subagent_type] [任务描述] [状态] [tokens] [duration] [返回摘要前100字]
${bridgeDigests}

请分析此 session 的工作流，输出 JSON 结果。
`;
}
```

### 4.3 Turn 数据精简策略

一个 session 可能有 600+ 条 turn，直接全部发给 AI 会消耗大量 tokens。精简策略：

```typescript
function buildTurnDigests(turns: TurnData[]): string {
  return turns.map((t, i) => {
    // 1. 压缩 content：取前 200 字符
    const content = (t.contentSummary || '').substring(0, 200);

    // 2. 压缩 tools：只显示 tool name 列表
    const tools = t.toolCalls.map(tc => tc.toolName).join(',');

    // 3. 时间格式化为相对时间（如 +5min, +3.5h）
    const relTime = formatRelativeTime(t.createdAt_ts, turns[0].createdAt_ts);

    return `#${i} [${t.role}] ${t.agentName || '-'} ${relTime} ` +
           `${t.inputTokens}/${t.outputTokens}tok ` +
           `tools:[${tools}] "${content}"`;
  }).join('\n');
}
```

**Token 估算**：
- 600 条 turns × ~100 字/条 ≈ 60K 字 ≈ ~20K tokens
- 27 条 bridges × ~150 字/条 ≈ 4K 字 ≈ ~1.3K tokens
- 总计输入约 22K tokens + system prompt ≈ **25K tokens**
- 输出 JSON ≈ 5-8K tokens
- 每次分析成本：约 $0.05-0.10（glm-5 定价）

### 4.4 大数据量处理

如果 session 特别大（>1000 turns），采用分批策略：

```
方案 A：分段分析 + 合并
  Turn[0-200]   → AI 分析 → 局部结果 A
  Turn[200-400] → AI 分析 → 局部结果 B
  Turn[400-600] → AI 分析 → 局部结果 C
  合并 A+B+C    → AI 合并  → 最终结果

方案 B：预处理 + 单次分析
  正则预提取阶段标记 → 构建阶段框架
  每个阶段取关键 turns → 精简后发 AI
  AI 补充语义分析 → 最终结果
```

---

## 5. API 设计

### 5.1 Settings API

```
GET  /api/settings/ai-provider        # 获取当前配置
POST /api/settings/ai-provider        # 创建/更新配置
POST /api/settings/ai-provider/test   # 测试连接
```

### 5.2 Workflow AI API

```
POST /api/ai/analyze-workflow
Body: { "taskId": "ses_xxx", "forceReanalyze": false }

Response (streaming):
  { "status": "analyzing", "progress": "正在读取 session 数据..." }
  { "status": "analyzing", "progress": "正在调用 AI 分析..." }
  { "status": "analyzing", "progress": "正在解析 AI 输出..." }
  { "status": "completed", "result": { ...WorkflowAIAnalysis } }

GET  /api/ai/workflow-result?taskId=ses_xxx
Response: { "status": "completed", "result": { ... }, "analyzedAt": "..." }
  或: { "status": "not_analyzed" }

DELETE /api/ai/workflow-result?taskId=ses_xxx
  → 删除分析结果，可重新分析
```

### 5.3 Streaming 实现

分析过程可能耗时 10-30 秒，使用 SSE (Server-Sent Events) 实时推送进度：

```typescript
// route.ts
export async function POST(req: NextRequest) {
  const stream = new ReadableStream({
    async start(controller) {
      const send = (data: any) => {
        controller.enqueue(`data: ${JSON.stringify(data)}\n\n`);
      };

      send({ status: 'reading', progress: '读取 session 数据...' });
      const sessionData = await loadSessionData(taskId);

      send({ status: 'building', progress: '构建分析 prompt...' });
      const prompt = buildAnalysisPrompt(sessionData);

      send({ status: 'analyzing', progress: 'AI 正在分析（预计 15-30 秒）...' });
      const aiResponse = await callLLM(prompt);

      send({ status: 'parsing', progress: '解析 AI 输出...' });
      const result = parseAIResponse(aiResponse);

      send({ status: 'saving', progress: '保存分析结果...' });
      await saveToDatabase(taskId, result);

      send({ status: 'completed', result });
      controller.close();
    }
  });

  return new Response(stream, {
    headers: { 'Content-Type': 'text/event-stream' }
  });
}
```

---

## 6. AI Provider 集成

### 6.1 多 Provider 支持

```typescript
interface AIProviderAdapter {
  chat(messages: ChatMessage[], options: ChatOptions): Promise<ChatResponse>;
}

// 阿里云（DashScope / Token Plan）
class AlibabaProvider implements AIProviderAdapter {
  async chat(messages, options) {
    const res = await fetch(`${this.baseUrl}/chat/completions`, {
      headers: { 'Authorization': `Bearer ${this.apiKey}` },
      body: JSON.stringify({ model: this.model, messages, ...options })
    });
    return res.json();
  }
}

// OpenAI 兼容
class OpenAIProvider implements AIProviderAdapter { /* ... */ }

// 自定义
class CustomProvider implements AIProviderAdapter { /* ... */ }
```

### 6.2 Settings UI

```
┌─ Settings ──────────────────────────────────────────────────┐
│                                                              │
│  AI Provider 配置                                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Provider:  [Alibaba Cloud Token Plan     ▾]            │  │
│  │ Base URL:  [https://token-plan.cn-beijing...    ]      │  │
│  │ API Key:   [sk-05b27••••••••                    ]      │  │
│  │ Model:     [glm-5                           ]          │  │
│  │                                                        │  │
│  │ 高级设置                                               │  │
│  │ Max Tokens:     [8192]                                 │  │
│  │ Temperature:    [0.3]                                  │  │
│  │ Prompt 模板:    [默认 ▾]                               │  │
│  │                                                        │  │
│  │ [测试连接]    [保存]                                    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  预设模板                                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ ○ 默认模板（阶段+步骤+瓶颈）                            │  │
│  │ ○ 简洁模板（仅阶段概览）                                │  │
│  │ ○ 详细模板（含改进建议和效率分析）                       │  │
│  │ ○ 自定义模板...                                        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. UI 设计

### 7.1 页面入口

在 Session 详情页新增 Tab：**Workflow (AI)**

```
[Overview] [Turns] [Timeline] [Subagents] [Skills] [Interactions] [Workflow] [Workflow (AI)] ⭐
```

### 7.2 三种视图模式

#### 视图 A：树形视图（默认）

类似正则解析方案的树形展示，但数据来自 AI 分析：

```
┌─ AI 分析概览 ─────────────────────────────────────────────────────────────┐
│ 📊 工作流类型: 算子开发 (AscendC DataFormatDimMap)                         │
│ ⏱ 总耗时: 10h10m │ 有效: 2h30m (25%) │ 等待: 7h40m (75%)                │
│ 💰 总成本: $6.56 │ 577 LLM calls │ 922 tool calls │ 26 subagents        │
│                                                                            │
│ 💡 AI 洞察:                                                                │
│   • 75% 时间花在等待用户确认（CP1/CP2/CP3），可考虑批量确认                  │
│   • 迭代三 token 消耗最大(9.1M)，因为上下文累积到 70K+                      │
│   • 方案设计阶段 ascendc-kernel-architect 效率最高（$0.74/18min）           │
│                                                                            │
│ ⚠️ 瓶颈:                                                                   │
│   • 🔴 CP1 等待 3h23m — 建议提前准备确认摘要                               │
│   • 🟡 迭代三上下文窗口 55% — 建议分 session 执行                          │
│   • 🟡 14:09 ascendc-ops-tester 调用失败 — 浪费一次 dispatch              │
└────────────────────────────────────────────────────────────────────────────┘

📋 阶段一：需求与设计 ──── 10:28 → 14:34 ──── Active 1h0m / Wait 3h6m
  ├── ✅ 1.1 开发准备 (general)                    3.9min  368K  $0.08
  │      AI: "初始化项目结构，检查 CANN 环境，准备开发模板"
  ├── ✅ 1.2 需求分析 (ascendc-kernel-architect)   3.7min  386K  $0.10
  │      AI: "分析 DataFormatDimMap 公式，确定 int32/int64 支持策略"
  ├── ⛔ CP1 需求分析确认                           ⏳ 3h23m
  │      AI: "Agent 展示了需求摘要表格，等待用户确认"
  │      [展开摘要] [查看 turn #12]
  ├── ✅ 1.3 方案设计 (ascendc-kernel-architect)   18.3min 1.0M  $0.74  ┐
  │      AI: "设计条件选择法避免整数除法，确定 Adds+Comp API 映射"    │ 并行
  ├── ✅ 1.4 测试设计 (general)                    23.5min 417K  $0.08  ┘
  │      AI: "设计 ST/UT 测试矩阵，覆盖 int32/int64 多 shape"
  └── ⛔ CP2 设计阶段确认                           ⏳ 47m
```

#### 视图 B：流程图视图

用 Mermaid 或自定义 SVG 渲染 DAG：

```
                    ┌──────────┐
                    │  用户输入  │
                    │ 开发需求   │
                    └────┬─────┘
                         │
              ┌──────────▼──────────┐
              │ 阶段一：需求与设计    │
              │ 10:28 - 14:34       │
              └──────────┬──────────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
     ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
     │ 1.1 开发   │ │ 1.2 需求 │ │ ...       │
     │ 准备       │ │ 分析     │ │           │
     │ general    │ │architect│ │           │
     │ 3.9min     │ │ 3.7min  │ │           │
     └─────┬─────┘ └────┬────┘ └─────┬─────┘
           │             │            │
           └─────────────┼────────────┘
                         │
              ┌──────────▼──────────┐
              │ ⛔ CP1 用户确认       │
              │ 等待 3h23m          │
              └──────────┬──────────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
     ┌─────▼─────┐ ┌────▼────┐
     │ 1.3 方案   │ │ 1.4 测试 │     ← 并行
     │ 设计       │ │ 设计     │
     └─────┬─────┘ └────┬────┘
           └─────────────┘
                         │
              ┌──────────▼──────────┐
              │ 阶段二：迭代开发      │
              │ 3 轮迭代             │
              └─────────────────────┘
```

#### 视图 C：时间线视图

在现有 Timeline Gantt 基础上叠加 AI 分析的阶段标注：

```
0min    100min    200min    300min    400min    500min    600min
|─────────|─────────|─────────|─────────|─────────|─────────|
|═════ 阶段一 ═══════════════|==CP1==|═════ 阶段二 ══════════|==CP2==|═══ 阶段三 ═══|==CP3==|
|██ 需求与设计                |██████|██ 迭代开发              |██████|██ 验收       |██████|
                             ↑                                  ↑                    ↑
                         等待 3h23m                          等待 47m            等待 3h33m

build     |████████████████████████████████████████████████████████████████████████|
general   |  ████          ████    ██   ██   ██  ██   ██   ██   ██   ██   ██      |
architect |     ████  ██████                                                          |
developer |          ████████████████████████████████████████████████                  |
```

### 7.3 节点交互

#### 点击 Step 节点

展开内联面板：

```
┌─ Step 详情: 1.3 方案设计 ──────────────────────────────────────────┐
│                                                                     │
│ AI 分析:                                                            │
│ "Architect subagent 分析了 DataFormatDimMap 的数学公式，确定了       │
│  条件选择法来避免整数除法（Ascend950 不支持 int div），              │
│  设计了 Adds + Comp 的 API 映射方案。这是整个开发的核心设计。"       │
│                                                                     │
│ 指标: 18.3min | 1,036.8K tokens | $0.74 | 39 tools | 1 skill       │
│                                                                     │
│ Bridge 交互:                                                        │
│ ┌─ Dispatch (父→子) ─────────────────────────────────────────────┐ │
│ │ "请设计 DataFormatDimMap 算子的实现方案。要求：                  │ │
│ │  1. 支持 int32/int64 数据类型                                   │ │
│ │  2. Ascend950 (arch35) 平台约束                                 │ │
│ │  3. 避免整数除法指令..."                                        │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│ ┌─ Response (子→父) ─────────────────────────────────────────────┐ │
│ │ "方案设计完成。核心策略：条件选择法，使用 Comp 指令实现           │ │
│ │  mod 运算，Adds 指令实现索引映射。详见 DESIGN.md..."             │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ [查看 Turns →]  [查看 Subagent Session →]                          │
└─────────────────────────────────────────────────────────────────────┘
```

#### 点击 Checkpoint 节点

```
┌─ Checkpoint: ⛔ CP1 需求分析确认 ─────────────────────────────────┐
│                                                                    │
│ ⏱ 等待时间: 3 小时 23 分钟                                        │
│    请求时间: 10:36  │  批准时间: 14:09                              │
│                                                                    │
│ Agent 展示的需求摘要:                                              │
│ ┌──────────────────────────────────────────────────────────────┐   │
│ │ | 项目 | 内容 |                                              │   │
│ │ | 算子名称 | DataFormatDimMap |                              │   │
│ │ | 功能 | 维度索引映射，支持 NHWC→NCHW 等格式转换 |           │   │
│ │ | 数学公式 | x_mod = (x + N) % N, y = dst_idx[x_mod] |      │   │
│ │ | 数据类型 | int32 / int64 |                                  │   │
│ └──────────────────────────────────────────────────────────────┘   │
│                                                                    │
│ 用户回应: "批准，继续"                                             │
│                                                                    │
│ AI 评价: "这是一个阻塞性检查点。3h23m 的等待占整个 session         │
│  时长的 33%。如果用户能快速确认，可节省大量时间。"                  │
│                                                                    │
│ [查看 Turn #12 →]                                                  │
└────────────────────────────────────────────────────────────────────┘
```

#### 点击 Bottleneck 节点

```
┌─ 🔴 瓶颈: CP1 等待过长 ───────────────────────────────────────────┐
│                                                                    │
│ 类型: 用户等待 (wait)                                              │
│ 严重度: 🔴 高                                                      │
│ 位置: 阶段一 / CP1 需求分析确认                                     │
│                                                                    │
│ 影响:                                                              │
│   • 时间: 浪费 3 小时 23 分钟（占 session 总时长 33%）              │
│   • Token: 无额外消耗（等待期间不调用 LLM）                         │
│                                                                    │
│ AI 建议:                                                           │
│ "考虑以下优化：                                                     │
│  1. 在 prompt 中预填确认选项，减少用户决策时间                       │
│  2. 将 CP1 和 CP2 合并为一次性确认                                  │
│  3. 设置自动确认模式（如果信任 agent 的分析）"                       │
└────────────────────────────────────────────────────────────────────┘
```

---

## 8. 新增文件清单

```
src/
├── app/
│   ├── settings/
│   │   └── page.tsx                     ← Settings 页面（AI Provider 配置）
│   └── api/
│       ├── settings/
│       │   └── ai-provider/
│       │       ├── route.ts             ← GET/POST AI Provider 配置
│       │       └── test/
│       │           └── route.ts         ← POST 测试连接
│       └── ai/
│           ├── analyze-workflow/
│           │   └── route.ts             ← POST 触发 AI 分析（SSE）
│           └── workflow-result/
│               └── route.ts             ← GET/DELETE 分析结果
├── lib/
│   ├── ai/
│   │   ├── providers/
│   │   │   ├── index.ts                 ← Provider 工厂
│   │   │   ├── alibaba.ts              ← 阿里云 Provider
│   │   │   └── openai.ts               ← OpenAI Provider
│   │   ├── prompts/
│   │   │   ├── workflow-analysis.ts     ← 默认分析 prompt
│   │   │   ├── workflow-simple.ts       ← 简洁 prompt
│   │   │   └── workflow-detailed.ts     ← 详细 prompt
│   │   ├── analyzer.ts                 ← 核心分析逻辑
│   │   └── parser.ts                   ← AI 输出 JSON 解析
│   └── ingest/
│       └── turn-digest.ts              ← Turn 数据精简（给 AI 的输入）
├── components/
│   ├── settings/
│   │   └── AIProviderSettings.tsx       ← Settings UI 组件
│   └── observe/
│       ├── WorkflowAIView.tsx           ← 主组件
│       ├── WorkflowAISummary.tsx        ← 概览条
│       ├── WorkflowAITree.tsx           ← 树形视图
│       ├── WorkflowAIFlowchart.tsx      ← 流程图视图
│       ├── WorkflowAITimeline.tsx       ← 时间线视图
│       ├── WorkflowAINodeGroup.tsx      ← 节点组件组（Phase/Step/CP/Bottleneck）
│       └── WorkflowAIInsights.tsx       ← 洞察 & 建议面板
```

---

## 9. 实现计划

### Phase 1：基础设施

| 步骤 | 说明 | 预估 |
|------|------|------|
| 1.1 | Prisma schema: AIProvider + WorkflowAIResult | 0.5h |
| 1.2 | AI Provider 适配器（alibaba + openai） | 2h |
| 1.3 | Settings 页面 UI + API | 2h |
| 1.4 | 测试连接功能 | 0.5h |

### Phase 2：分析引擎

| 步骤 | 说明 | 预估 |
|------|------|------|
| 2.1 | turn-digest.ts: 精简 turn 数据 | 1h |
| 2.2 | prompt 模板设计（3 个模板） | 2h |
| 2.3 | analyzer.ts: 调用 AI + 流式进度 | 3h |
| 2.4 | parser.ts: JSON 解析 + 校验 | 2h |
| 2.5 | API route (SSE streaming) | 2h |
| 2.6 | 大数据量分批处理 | 2h |

### Phase 3：UI 渲染

| 步骤 | 说明 | 预估 |
|------|------|------|
| 3.1 | WorkflowAIView 主组件 + Tab 集成 | 1h |
| 3.2 | 树形视图渲染 | 4h |
| 3.3 | 节点展开详情面板 | 3h |
| 3.4 | 流程图视图（Mermaid/SVG） | 4h |
| 3.5 | 时间线视图（叠加阶段标注） | 3h |
| 3.6 | 洞察 & 建议面板 | 1h |

### Phase 4：优化

| 步骤 | 说明 | 预估 |
|------|------|------|
| 4.1 | 分析结果缓存 + 重新分析 | 1h |
| 4.2 | 分析与正则结果合并 | 2h |
| 4.3 | prompt 模板自定义 | 1h |
| 4.4 | 分析结果导出（JSON/PDF） | 1h |

**总计: ~36h**

---

## 10. 与正则方案的协作

两种方案不互斥，可以组合使用：

```
导入 Session
    │
    ├──▶ 正则解析（自动，零成本，毫秒级）
    │    → 提取明确的 phase/checkpoint/iteration 标记
    │    → 写入 WorkflowPhase/Step/Checkpoint 表
    │
    └──▶ AI 分析（用户触发，付费，秒级）
         → 深度语义理解
         → 瓶颈识别 + 改进建议
         → 写入 WorkflowAIResult 表
              │
              ▼
         Workflow (AI) Tab
         → 优先展示 AI 分析结果
         → AI 未覆盖的字段 fallback 到正则结果
         → 用户可切换 "正则视图" / "AI 视图"
```

**Workflow Tab**: 正则解析结果（结构化，精确）
**Workflow (AI) Tab**: AI 分析结果（语义丰富，含洞察和建议）
