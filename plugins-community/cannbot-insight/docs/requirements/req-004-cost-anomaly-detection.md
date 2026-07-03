# REQ-004: Cost Anomaly Detection

## 概述

自动检测 session 中 token 异常飙升的 turn，标记并在 UI 中高亮显示，关联 tool call 和 context 变化帮助用户分析原因。

## 背景

在长 session 中，某些 turn 的 token 用量会突然飙升（如上下文爆炸、无效重试、大文件读取），用户需要快速定位这些异常 turn 并理解原因。

## 功能要求

### 1. 异常检测算法

在 `src/lib/ingest/` 下新建 `anomaly-detector.ts`：

#### 检测策略

对每个 session 的 assistant turns，使用以下策略检测异常：

**策略 A — 统计偏离（>2σ）**

```typescript
function detectAnomalies(turns: TurnRow[]): AnomalyResult[] {
  const assistantTurns = turns.filter(t => t.role === 'assistant' && t.totalTokens > 0);
  if (assistantTurns.length < 3) return []; // 样本太少，跳过

  const mean = assistantTurns.reduce((s, t) => s + t.totalTokens, 0) / assistantTurns.length;
  const stddev = Math.sqrt(
    assistantTurns.reduce((s, t) => s + Math.pow(t.totalTokens - mean, 2), 0) / assistantTurns.length
  );

  const threshold = mean + 2 * stddev;

  return assistantTurns
    .filter(t => t.totalTokens > threshold)
    .map(t => ({
      turnId: t.id,
      turnIndex: t.turnIndex,
      totalTokens: t.totalTokens,
      mean,
      stddev,
      zScore: (t.totalTokens - mean) / stddev,
      severity: classifySeverity(t.totalTokens, mean, stddev),
      possibleCauses: analyzePossibleCauses(t, turns),
    }));
}
```

**策略 B — 绝对阈值**

单个 turn 的 input tokens > context window 的 70% 时标记。

**策略 C — 环比突增**

当前 turn 的 totalTokens > 上一 turn 的 3 倍时标记。

#### 严重级别

| 级别 | 条件 | UI 颜色 |
|---|---|---|
| warning | zScore > 2 且 < 3 | 黄色 |
| critical | zScore ≥ 3 | 红色 |

#### 可能原因分析

```typescript
function analyzePossibleCauses(turn: TurnRow, allTurns: TurnRow[]): string[] {
  const causes: string[] = [];

  // 原因1: 上下文过大
  if (turn.contextWindowPct && turn.contextWindowPct > 70) {
    causes.push('Context window usage high');
  }

  // 原因2: 大 tool call 结果
  if (turn.inputMessagesTokens > 0) {
    const prevTurn = allTurns.find(t => t.turnIndex === turn.turnIndex - 1);
    if (prevTurn && turn.inputMessagesTokens > prevTurn.inputMessagesTokens * 2) {
      causes.push('Large input message increase');
    }
  }

  // 原因3: cache miss
  if (turn.cacheReadTokens === 0 && turn.inputTokens > 10000) {
    causes.push('No cache hit (cold start or cache eviction)');
  }

  // 原因4: 大量 reasoning
  if (turn.reasoningTokens > turn.outputTokens * 5) {
    causes.push('Heavy reasoning (thinking tokens >> output tokens)');
  }

  return causes;
}
```

### 2. 数据模型

在 `Turn` 模型中添加异常标记字段（不修改 Prisma schema，通过运行时计算）：

```typescript
// src/lib/ingest/anomaly-detector.ts
export interface AnomalyResult {
  turnId: string;
  turnIndex: number;
  totalTokens: number;
  mean: number;
  stddev: number;
  zScore: number;
  severity: 'warning' | 'critical';
  possibleCauses: string[];
}
```

### 3. API 变更

新增 API 路由：

```
GET /api/observe/session/{sessionId}/anomalies
```

返回：

```json
{
  "anomalies": [
    {
      "turnId": "...",
      "turnIndex": 5,
      "totalTokens": 85000,
      "mean": 12000,
      "stddev": 8000,
      "zScore": 9.125,
      "severity": "critical",
      "possibleCauses": ["Context window usage high", "No cache hit"],
      "turn": { ...turnData },
      "toolCalls": [ ...toolCallData ]
    }
  ],
  "stats": {
    "mean": 12000,
    "stddev": 8000,
    "threshold": 28000
  }
}
```

### 4. UI 变更

#### 4.1 TurnTimeline 异常标记

在 `TurnTimeline.tsx` 中，异常 turn 的时间线条目添加：
- ⚠️ 图标（warning 黄色 / critical 红色）
- hover 显示 z-score 和可能原因

#### 4.2 Overview 异常统计

在 Session 详情 Overview tab 添加 Anomaly Summary 卡片：
- 异常 turn 数量
- 最高 z-score
- 异常 turn 的总 token 占比

#### 4.3 Anomaly 详情面板

点击异常 turn 时，在 `TurnDetail` 面板顶部显示 Anomaly 信息卡：
- token 分布（当前 vs 平均）
- z-score 可视化（进度条）
- 可能原因列表
- 关联的 tool calls

### 5. 文件变更

| 文件 | 变更 |
|---|---|
| `src/lib/ingest/anomaly-detector.ts` | **新建** — 异常检测算法 |
| `src/app/api/observe/session/[sessionId]/anomalies/route.ts` | **新建** — Anomaly API |
| `src/components/observe/AnomalyBadge.tsx` | **新建** — 异常标记组件 |
| `src/components/observe/AnomalySummary.tsx` | **新建** — Overview 异常统计卡 |
| `src/components/observe/AnomalyDetail.tsx` | **新建** — 异常详情面板 |
| `src/components/observe/TurnTimeline.tsx` | **修改** — 添加异常标记 |
| `src/components/observe/TurnDetail.tsx` | **修改** — 添加异常信息卡 |
| `src/app/session/[taskId]/page.tsx` | **修改** — 传递 anomaly 数据 |

## 测试要求

在 `tests/` 下创建 `anomaly-detector.test.ts`，覆盖：

- 正常数据不产生异常
- 单个异常 turn（zScore > 2）
- 多个异常 turn
- 样本不足（< 3 个 turn）跳过检测
- 绝对阈值检测（contextWindowPct > 70%）
- 环比突增检测
- 严重级别分类正确
- 可能原因分析准确

## 关键文件参考

- `src/lib/ingest/turn-split.ts` — `TurnRow` 接口定义
- `src/lib/ingest/cost-calculator.ts` — 费用计算参考
- `src/components/observe/TurnTimeline.tsx` — Turn 时间线组件
- `src/components/observe/TurnDetail.tsx` — Turn 详情组件
- `src/components/observe/TokenBarChart.tsx` — SVG 图表参考
- `src/app/api/observe/session/turns/route.ts` — Turns API 参考
- `src/app/session/[taskId]/page.tsx` — Session 详情页面
