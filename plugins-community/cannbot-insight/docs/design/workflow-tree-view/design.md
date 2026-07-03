# Workflow Tree View 设计方案

## 1. 需求概述

在 Session 详情页新增 **Workflow** Tab，以树形结构展示 Agent 的工作流执行过程：

```
阶段一：需求与设计 (10:28 - 14:34, ~4h)
  ├── 1.1 开发准备        (general subagent)          ✅ 3.9min
  ├── 1.2 需求分析        (ascendc-kernel-architect)  ✅ 3.7min
  ├── ⛔ CP1 用户确认                                 ⏳ 等待3h23min
  ├── 1.3 方案设计        (ascendc-kernel-architect)  ✅ 18.3min  ┐ 并行
  ├── 1.4 测试设计        (general)                   ✅ 23.5min  ┘
  └── ⛔ CP2 用户确认                                 ⏳ 等待47min

阶段二：迭代开发 (15:02 - 16:33, ~1.5h)
  ├── 迭代一：骨架搭建
  │   ├── A1-Main + A1-P + B (并行启动)              ✅ 14.6min
  │   ├── A2 UT开发                                   ✅ 8.8min
  │   ├── 联调验证                                     ✅ 9.0min
  │   └── 测试验收                                     ✅ 1.5min
  ├── 迭代二：策略整合
  │   └── ...
  └── 迭代三：规格完整
      └── ...

阶段三：验收 (16:34 - 20:38, ~4h)
  ├── PyTorch ST 测试开发                              ✅ 7.4min
  ├── 3.1 最终精度验收                                 ✅ 3.4min
  ├── ⚪ CP3 用户确认                                  ⏳ 等待3h33min
  └── 真实NPU验证                                      ✅ 19.2min
```

每个节点可点击展开，查看：
- 对应的 subagent 详情（tokens/cost/tools/skills）
- 对应的 turn 列表
- 对应的 bridge 交互内容（dispatch prompt + response）
- Checkpoint 等待时间明细

---

## 2. 数据来源分析

### 2.1 核心发现：阶段信息来自 text content

通过查询 opencode 数据库，发现 build agent 在 text 类型的 part 中**显式标注**了工作流阶段：

```
10:28:11  "开始执行 **阶段一：需求与设计阶段**，首先进入 **1.1 开发准备**。"
10:36:37  "⛔ CP1 需求分析确认"
14:09:08  "用户已批准，现在**并行执行 1.3 方案设计和 1.4 测试设计**"
14:34:10  "⛔ CP2 设计阶段确认"
15:02:47  "用户批准进入开发阶段。开始**阶段二：迭代一骨架搭建**"
15:39:54  "迭代一验收通过 ✅。进入**迭代二：策略整合**"
16:33:58  "迭代三验收通过 ✅。进入**阶段三：验收阶段**"
16:45:05  "⚪ CP3 精度验收确认"
```

### 2.2 已有的结构化数据

现有数据库已有：
- `Turn` 表：每条 turn 有 `role`, `content`, `agentName`, `subagentSessionId`, `createdAt_ts`, `tokens`
- `InteractionBridge` 表：每次 subagent dispatch 有 `dispatchContent`, `dispatchTimestamp`, `subagentName`, `subagentTokens`, `subagentLatencyMs`, `status`
- `Execution` 表：每个 agent execution 有聚合指标
- `ToolCall` 表：`toolName = 'task'` 就是 subagent 派发

### 2.3 需要新建的数据

需要从 Turn 的 text content 中**解析提取**：
- 阶段（Phase）边界
- 步骤（Step）标签
- Checkpoint（CP）事件
- 并行任务组
- 迭代（Iteration）分组

---

## 3. 架构设计

### 3.1 数据流

```
opencode.db (原始数据)
    │
    ▼
opencode-db.ts (adapter: readSession)
    │
    ▼
normalize.ts (标准化 RawInteraction[])
    │
    ▼
turn-split.ts (拆分为 Turn[] + ToolCall[] + SkillEvent[])
    │
    ├──▶ bridge-builder.ts (构建 InteractionBridge[])  ← 已有
    ├──▶ execution-split.ts (构建 Execution[])          ← 已有
    └──▶ phase-split.ts (构建 WorkflowTree)              ← 新增 ⭐
            │
            ▼
        WorkflowPhase[] + WorkflowStep[] + Checkpoint[]
            │
            ▼
        Prisma DB (WorkflowPhase / WorkflowStep / Checkpoint 表)
            │
            ▼
        /api/observe/session/workflow API
            │
            ▼
        WorkflowTreeView 组件 (新增 Tab)
```

### 3.2 新增文件清单

```
src/
├── lib/
│   └── ingest/
│       └── phase-split.ts          ← 核心：从 Turn[] 解析工作流树
├── app/
│   └── api/
│       └── observe/
│           └── session/
│               └── workflow/
│                   └── route.ts    ← API：返回工作流树数据
├── components/
│   └── observe/
│       ├── WorkflowTreeView.tsx    ← 主组件：树形展示
│       ├── PhaseNode.tsx           ← 阶段节点
│       ├── StepNode.tsx            ← 步骤节点（关联 subagent）
│       ├── CheckpointNode.tsx      ← Checkpoint 节点（等待时间）
│       ├── ParallelGroup.tsx       ← 并行任务组
│       └── IterationGroup.tsx      ← 迭代分组
└── app/
    └── session/
        └── [taskId]/
            └── page.tsx            ← 修改：新增 Workflow Tab
prisma/
└── schema.prisma                   ← 修改：新增 3 个 model
```

---

## 4. 数据模型设计

### 4.1 Prisma Schema 新增

```prisma
// ─── WorkflowPhase ──────────────────────────────────────────
// 工作流阶段：如"阶段一：需求与设计"
model WorkflowPhase {
  id              String   @id @default(cuid())
  sessionId       String
  session         Session  @relation(fields: [sessionId], references: [id], onDelete: Cascade)

  phaseIndex      Int                          // 阶段序号 (1, 2, 3)
  phaseName       String                       // "需求与设计" / "迭代开发" / "验收"
  fullLabel       String                       // "阶段一：需求与设计阶段"

  startTime       DateTime?
  endTime         DateTime?
  durationMs      Int       @default(0)

  activeTimeMs    Int       @default(0)        // 实际执行时间（排除 checkpoint 等待）
  waitTimeMs      Int       @default(0)        // checkpoint 等待总时间

  totalTokens     Int       @default(0)
  totalCost       Float     @default(0)
  toolCallCount   Int       @default(0)
  subagentCount   Int       @default(0)

  // 由哪个 turn 触发的（包含"开始执行 **阶段一..."的那条 turn）
  triggerTurnId   String?

  createdAt       DateTime  @default(now())

  steps           WorkflowStep[]
  checkpoints     WorkflowCheckpoint[]

  @@index([sessionId])
  @@index([sessionId, phaseIndex])
}

// ─── WorkflowStep ───────────────────────────────────────────
// 工作流步骤：如 "1.1 开发准备", "迭代一：骨架搭建"
model WorkflowStep {
  id              String   @id @default(cuid())
  sessionId       String
  session         Session  @relation(fields: [sessionId], references: [id], onDelete: Cascade)
  phaseId         String
  phase           WorkflowPhase @relation(fields: [phaseId], references: [id], onDelete: Cascade)

  stepIndex       Int                          // 步骤在阶段内的序号
  stepName        String                       // "开发准备" / "迭代一：骨架搭建"
  stepLabel       String                       // "1.1 开发准备"

  // 迭代信息（如果有）
  iterationIndex  Int?                         // 所属迭代轮次 (1, 2, 3), null 表示不在迭代中
  iterationName   String?                      // "迭代一：骨架搭建"

  startTime       DateTime?
  endTime         DateTime?
  durationMs      Int       @default(0)

  totalTokens     Int       @default(0)
  totalCost       Float     @default(0)
  toolCallCount   Int       @default(0)

  // 关联的 bridge（如果是 subagent 任务）
  bridgeId        String?                      // 关联 InteractionBridge
  subagentSessionId String?
  subagentType    String?                      // "general" / "ascendc-kernel-developer"
  subagentName    String?                      // "ascendc-kernel-architect"
  status          String    @default("unknown") // completed / failed / running

  // 并行组
  parallelGroupId String?                      // 同一并行组的 step 共享此 ID

  // 由哪个 turn 触发的
  triggerTurnId   String?

  createdAt       DateTime  @default(now())

  @@index([phaseId])
  @@index([sessionId])
  @@index([parallelGroupId])
  @@index([bridgeId])
}

// ─── WorkflowCheckpoint ─────────────────────────────────────
// 工作流检查点：如 "⛔ CP1 用户确认"
model WorkflowCheckpoint {
  id              String   @id @default(cuid())
  sessionId       String
  session         Session  @relation(fields: [sessionId], references: [id], onDelete: Cascade)
  phaseId         String
  phase           WorkflowPhase @relation(fields: [phaseId], references: [id], onDelete: Cascade)

  checkpointIndex Int                          // 在阶段内的序号
  checkpointType  String                       // "block" (⛔) / "info" (⚪)
  checkpointLabel String                       // "CP1 需求分析确认"

  // 时间
  requestedAt     DateTime?                    // agent 提出 checkpoint 的时间
  approvedAt      DateTime?                    // 用户批准的时间
  waitTimeMs      Int       @default(0)        // 等待时长

  // 确认内容
  summaryContent  String?                      // agent 展示的摘要（CP1/CP2/CP3 的表格内容）
  userResponse    String?                      // 用户的回应（通常是 "批准" / "继续"）

  // 由哪个 turn 触发
  triggerTurnId   String?
  responseTurnId  String?

  createdAt       DateTime  @default(now())

  @@index([phaseId])
  @@index([sessionId])
}
```

### 4.2 Session model 扩展

```prisma
model Session {
  // ... 现有字段 ...

  workflowPhases    WorkflowPhase[]
  workflowSteps     WorkflowStep[]
  workflowCheckpoints WorkflowCheckpoint[]
}
```

---

## 5. 核心算法：phase-split.ts

### 5.1 解析策略

从 root agent 的 assistant turns 的 text content 中提取工作流结构。

#### 正则模式

```typescript
// 阶段开始
const PHASE_START_RE = /(?:开始执行\s*\**)?阶段([一二三四五六七八九十\d]+)[：:]\s*(.+?)(?:阶段)?\**/;
// 例: "开始执行 **阶段一：需求与设计阶段**"
// 例: "用户批准进入开发阶段。开始**阶段二：迭代一骨架搭建**"

// Checkpoint
const CHECKPOINT_RE = /[⛔⚪🔴🟡]\s*(CP\d+)\s+(.+?)(?:确认|通过)?$/;
// 例: "⛔ CP1 需求分析确认"
// 例: "⚪ CP3 精度验收确认"

// 迭代开始
const ITERATION_RE = /迭代([一二三四五六七八九十\d]+)[：:]\s*(.+)/;
// 例: "迭代一验收通过 ✅。进入**迭代二：策略整合**"
// 例: "开始**迭代三：规格完整**"

// 步骤完成标记
const STEP_COMPLETE_RE = /(.+?)完成\s*✅/;
// 例: "1.1 开发准备已完成并提交"
// 例: "联调验证通过 ✅"

// 并行启动
const PARALLEL_RE = /并行(?:执行|启动)\s*(.+?)(?:：|$)/;
// 例: "并行执行 1.3 方案设计和 1.4 测试设计"
// 例: "并行启动 A1-Main + A1-P + B"

// 用户批准
const USER_APPROVED_RE = /用户(?:已|已)?批准/;
// 例: "用户已批准"
// 例: "用户批准进入开发阶段"
```

#### 辅助匹配：bridge dispatch content

每个 `InteractionBridge` 的 `dispatchContent` 包含 subagent 任务描述：
```
"开发准备"
"需求分析"
"方案设计"
"迭代一主线开发"
"模板穿刺 P1-P5"
```

这些可以和 step name 做**模糊匹配**，从而关联 step → bridge → subagent。

### 5.2 核心函数签名

```typescript
export interface WorkflowTree {
  phases: WorkflowPhaseRow[];
  steps: WorkflowStepRow[];
  checkpoints: WorkflowCheckpointRow[];
}

export interface WorkflowPhaseRow {
  id: string;
  sessionId: string;
  phaseIndex: number;
  phaseName: string;
  fullLabel: string;
  startTime: string | null;
  endTime: string | null;
  durationMs: number;
  activeTimeMs: number;
  waitTimeMs: number;
  totalTokens: number;
  totalCost: number;
  toolCallCount: number;
  subagentCount: number;
  triggerTurnId: string | null;
}

export interface WorkflowStepRow {
  id: string;
  sessionId: string;
  phaseId: string;
  stepIndex: number;
  stepName: string;
  stepLabel: string;
  iterationIndex: number | null;
  iterationName: string | null;
  startTime: string | null;
  endTime: string | null;
  durationMs: number;
  totalTokens: number;
  totalCost: number;
  toolCallCount: number;
  bridgeId: string | null;
  subagentSessionId: string | null;
  subagentType: string | null;
  subagentName: string | null;
  status: string;
  parallelGroupId: string | null;
  triggerTurnId: string | null;
}

export interface WorkflowCheckpointRow {
  id: string;
  sessionId: string;
  phaseId: string;
  checkpointIndex: number;
  checkpointType: string;      // "block" | "info"
  checkpointLabel: string;
  requestedAt: string | null;
  approvedAt: string | null;
  waitTimeMs: number;
  summaryContent: string | null;
  userResponse: string | null;
  triggerTurnId: string | null;
  responseTurnId: string | null;
}

export function splitWorkflow(
  turns: TurnRow[],
  bridges: InteractionBridgeRow[],
  sessionId: string,
): WorkflowTree;
```

### 5.3 算法流程

```
1. 过滤 root agent 的 assistant turns（isSubagent = false）
2. 按时间排序

3. 遍历 turns，逐条匹配模式：

   IF 匹配 PHASE_START_RE:
     → 创建新 Phase
     → 重置 stepIndex, iterationIndex

   IF 匹配 CHECKPOINT_RE:
     → 创建 Checkpoint
     → 记录 requestedAt = 当前 turn 时间
     → 继续扫描后续 turns，找到 user role turn 或
       下一个 assistant turn 包含 USER_APPROVED_RE
       → 记录 approvedAt
       → 计算 waitTimeMs = approvedAt - requestedAt
     → 累加到当前 Phase 的 waitTimeMs

   IF 匹配 ITERATION_RE:
     → 记录当前 iterationIndex/iterationName
     → 后续 steps 归属此迭代

   IF 匹配 PARALLEL_RE:
     → 创建 parallelGroupId
     → 后续的 subagent dispatches（同一 turn 内多个 task toolCall）
       共享此 parallelGroupId

   IF 当前 turn 有 toolName='task' 的 toolCall:
     → 从 bridge 中找到对应的 bridge（通过 dispatchTurnId 匹配）
     → 创建 Step，关联 bridge
     → 从 bridge 获取 subagentSessionId, subagentTokens, duration 等

   IF 匹配 STEP_COMPLETE_RE（且没有对应 bridge）:
     → 创建 Step（纯 root agent 步骤，如 git 操作等）

4. 后处理：
   → 计算每个 Phase 的聚合指标（tokens/cost/tools = 所属 steps 之和）
   → 计算 Phase duration = endTime - startTime
   → 计算 Phase activeTime = duration - waitTime
   → 标记并行组内的 steps
```

---

## 6. API 设计

### 6.1 GET /api/observe/session/workflow?taskId=xxx

```json
{
  "phases": [
    {
      "phaseIndex": 1,
      "phaseName": "需求与设计",
      "fullLabel": "阶段一：需求与设计阶段",
      "startTime": "2026-04-22T10:28:11.000Z",
      "endTime": "2026-04-22T14:34:10.000Z",
      "durationMs": 14759000,
      "activeTimeMs": 3600000,
      "waitTimeMs": 11159000,
      "totalTokens": 5000000,
      "totalCost": 1.23,
      "toolCallCount": 45,
      "subagentCount": 4,
      "steps": [
        {
          "stepIndex": 1,
          "stepName": "开发准备",
          "stepLabel": "1.1 开发准备",
          "subagentType": "general",
          "status": "completed",
          "durationMs": 234000,
          "totalTokens": 368500,
          "bridgeId": "cxxx",
          "parallelGroupId": null
        },
        {
          "stepIndex": 2,
          "stepName": "需求分析",
          "stepLabel": "1.2 需求分析",
          "subagentType": "ascendc-kernel-architect",
          "status": "completed",
          "durationMs": 222000,
          "totalTokens": 386400
        }
      ],
      "checkpoints": [
        {
          "checkpointIndex": 1,
          "checkpointType": "block",
          "checkpointLabel": "CP1 需求分析确认",
          "requestedAt": "2026-04-22T10:36:37.000Z",
          "approvedAt": "2026-04-22T14:09:08.000Z",
          "waitTimeMs": 12751000,
          "summaryContent": "| 项目 | 内容 |\n| 算子名称 | DataFormatDimMap |..."
        }
      ]
    }
  ],
  "summary": {
    "totalPhases": 3,
    "totalSteps": 27,
    "totalCheckpoints": 3,
    "totalActiveTimeMs": 5400000,
    "totalWaitTimeMs": 23400000,
    "activeTimePct": 18.8,
    "iterations": 3
  }
}
```

---

## 7. UI 组件设计

### 7.1 页面结构

在 Session 详情页现有的 6 个 Tab 后新增第 7 个 Tab：**Workflow**

```
[Overview] [Turns] [Timeline] [Subagents] [Skills] [Interactions] [Workflow] ⭐
```

### 7.2 顶部汇总条

```
┌─────────────────────────────────────────────────────────────────┐
│ 3 Phases │ 27 Steps │ 3 Checkpoints │ Active: 1.5h │ Wait: 6.5h│
│                                                    Active 18.8% │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 树形视图

```typescript
interface WorkflowTreeViewProps {
  taskId: string;
}

// 树形结构渲染
function WorkflowTreeView({ taskId }: WorkflowTreeViewProps) {
  return (
    <div className="space-y-4">
      <WorkflowSummaryBar />

      {phases.map(phase => (
        <PhaseNode key={phase.phaseIndex} phase={phase}>
          {/* 子节点按时间排序混合渲染 steps + checkpoints */}
          {phase.children.map(child => {
            if (child.type === 'checkpoint') return <CheckpointNode />;
            if (child.type === 'parallel-group') return <ParallelGroup />;
            if (child.iterationIndex) return <IterationGroup />;
            return <StepNode />;
          })}
        </PhaseNode>
      ))}
    </div>
  );
}
```

### 7.4 节点设计

#### PhaseNode（阶段节点）

```
┌──────────────────────────────────────────────────────────────────┐
│ 📋 阶段一：需求与设计阶段                                         │
│    10:28 ──── 14:34  │  Duration: 4h 6min  │  Active: 1h 0min   │
│    Tokens: 5.0M  │  Cost: $1.23  │  Wait: 3h 6min (76%)         │
│                         [展开/折叠]                               │
└──────────────────────────────────────────────────────────────────┘
```

- 折叠时只显示汇总
- 展开时显示所属 steps 和 checkpoints

#### StepNode（步骤节点）

```
  ├── ✅ 1.1 开发准备                    general           3.9min  368.5k tokens  $0.08
  │      [点击展开查看 subagent turns / bridge 交互内容]
```

- 左侧：状态图标 + 步骤标签
- 中间：subagent type badge
- 右侧：duration + tokens + cost
- 点击展开：显示关联的 bridge dispatch/response content

#### CheckpointNode（检查点节点）

```
  ├── ⛔ CP1 需求分析确认                 ⏳ 3h 23min 等待
  │      Agent 摘要: | 算子名称 | DataFormatDimMap | ...
  │      用户回应: "批准，继续"
```

- ⛔ = 阻塞性 checkpoint（必须用户确认才能继续）
- ⚪ = 信息性 checkpoint（展示结果，不阻塞）
- 显示等待时间和 agent 展示的摘要

#### ParallelGroup（并行任务组）

```
  ├── ┌─ 并行 ──────────────────────────────────────────────────┐
  │   │  1.3 方案设计    ascendc-kernel-architect  18.3min  1.0M │
  │   │  1.4 测试设计    general                   23.5min  417k  │
  │   └──────────────────────────────────────────────────────────┘
```

- 用虚线框包围并行的 steps
- 并行组内显示总 duration = max(各 step duration)

#### IterationGroup（迭代组）

```
  ├── 🔄 迭代一：骨架搭建 ──────────────────────────────────────┐
  │   ├── A1-Main 主线开发    ascendc-kernel-developer  14.2min  3.2M
  │   ├── A1-P 模板穿刺       ascendc-kernel-developer   6.8min  991k
  │   ├── B ST测试开发        general                     28.6s   89.7k
  │   ├── A2 UT开发           ascendc-kernel-developer   8.8min  2.1M
  │   ├── 联调验证            ascendc-kernel-developer   9.0min  2.1M
  │   └── 测试验收            general                     1.2min  181k
  │   Total: 14.2min (并行) │ 9.1M tokens │ $0.91
  └───────────────────────────────────────────────────────────────┘
```

### 7.5 交互设计

1. **点击 Step**：展开内联面板，显示：
   - bridge dispatch content（父 agent 发给 subagent 的 prompt）
   - bridge response content（subagent 返回的结果）
   - "查看 turns →" 链接跳转到 Turns tab 并过滤到对应 interaction

2. **点击 Checkpoint**：展开显示：
   - agent 展示的摘要内容（markdown 渲染）
   - 等待时间可视化（进度条）
   - 用户的回应

3. **时间轴叠加**：
   - 在 PhaseNode 上显示一个简化版的 Gantt 条
   - 用颜色区分：active（蓝色）vs wait（灰色）vs parallel（紫色）

4. **全局时间线**：
   - 页面顶部一个全 session 的时间线，标注 phase 边界和 checkpoint 位置
   ```
   |════Phase1════|==CP1==|════Phase2════|==CP2==|════Phase3════|==CP3==|
   10:28         14:34   15:02          16:33   20:19         20:38
   ```

---

## 8. 数据集成

### 8.1 修改 data-service.ts

在现有的 `ingestSession` 流程中，在 `buildBridges` 和 `splitExecutions` 之后，新增 `splitWorkflow` 步骤：

```typescript
// data-service.ts 中的 ingestSession 函数
export async function ingestSession(...) {
  // ... 现有流程 ...
  const turns = splitIntoTurns(interactions, sessionId);
  const bridges = buildBridges(interactions, toolCalls, turns, sessionId, rootExecId);
  const executions = splitExecutions(turns, toolCalls, skillEvents, sessionId);

  // ⭐ 新增：解析工作流树
  const workflow = splitWorkflow(turns, bridges, sessionId);

  // ... 写入 DB ...
  // 新增写入 workflowPhases, workflowSteps, workflowCheckpoints
}
```

### 8.2 修改 merge.ts

在 `dedupSession` 中，清理旧数据时也要清理 workflow 相关表：

```typescript
// 清理时级联删除（Prisma onDelete: Cascade 已处理）
await prisma.workflowPhase.deleteMany({ where: { sessionId } });
await prisma.workflowStep.deleteMany({ where: { sessionId } });
await prisma.workflowCheckpoint.deleteMany({ where: { sessionId } });
```

---

## 9. 实现计划

### Phase 1：数据层（核心）

| 步骤 | 文件 | 说明 |
|------|------|------|
| 1.1 | `prisma/schema.prisma` | 新增 3 个 model |
| 1.2 | `prisma/migrations/` | 生成 migration |
| 1.3 | `src/lib/ingest/phase-split.ts` | 核心解析算法 |
| 1.4 | `src/lib/ingest/phase-split.test.ts` | 单元测试 |
| 1.5 | `src/lib/ingest/data-service.ts` | 集成 phase-split |
| 1.6 | `src/lib/ingest/merge.ts` | dedup 时清理 workflow 数据 |

### Phase 2：API 层

| 步骤 | 文件 | 说明 |
|------|------|------|
| 2.1 | `src/app/api/observe/session/workflow/route.ts` | GET API |

### Phase 3：UI 层

| 步骤 | 文件 | 说明 |
|------|------|------|
| 3.1 | `src/components/observe/WorkflowTreeView.tsx` | 主组件 |
| 3.2 | `src/components/observe/PhaseNode.tsx` | 阶段节点 |
| 3.3 | `src/components/observe/StepNode.tsx` | 步骤节点 |
| 3.4 | `src/components/observe/CheckpointNode.tsx` | Checkpoint 节点 |
| 3.5 | `src/components/observe/ParallelGroup.tsx` | 并行组 |
| 3.6 | `src/components/observe/IterationGroup.tsx` | 迭代组 |
| 3.7 | `src/app/session/[taskId]/page.tsx` | 新增 Workflow Tab |

### Phase 4：增强

| 步骤 | 文件 | 说明 |
|------|------|------|
| 4.1 | - | 全局时间线条（phase 边界 + checkpoint 位置）|
| 4.2 | - | Step 点击展开 bridge 内容 |
| 4.3 | - | Checkpoint 摘要 markdown 渲染 |
| 4.4 | - | Active vs Wait 时间可视化 |

---

## 10. 测试计划

### 10.1 phase-split.ts 单元测试

```typescript
describe('splitWorkflow', () => {
  it('should parse phase boundaries from text content', () => {
    // 输入包含 "开始执行 **阶段一：需求与设计阶段**" 的 turn
    // 期望创建 phaseIndex=1 的 WorkflowPhase
  });

  it('should parse checkpoints with wait time', () => {
    // 输入包含 "⛔ CP1 需求分析确认" 的 turn
    // + 后续 user turn（3小时后）
    // 期望 waitTimeMs ≈ 3h
  });

  it('should match steps to bridges', () => {
    // step "开发准备" 匹配 bridge dispatchContent "开发准备"
    // 关联 subagentSessionId 和 tokens
  });

  it('should detect parallel groups', () => {
    // 同一 turn 内有多个 task toolCall
    // 或 text 包含 "并行启动"
    // 期望 parallelGroupId 相同
  });

  it('should group iterations', () => {
    // "迭代一" / "迭代二" / "迭代三" 的 steps
    // 期望 iterationIndex 正确
  });

  it('should handle sessions without workflow markers', () => {
    // 非工作流 session（没有"阶段一"等文本）
    // 期望返回空 phases[]
  });
});
```

### 10.2 端到端测试

用 `opencode_db_andong.db` 中的 `ses_24cfbf2a2ffenf6HPFpHx0Fh2n` 作为测试数据：
- 期望 3 phases, 27 steps, 3 checkpoints
- 验证时间计算正确
- 验证 bridge 关联正确
