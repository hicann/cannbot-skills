# Performance Tab 设计文档

## 1. 概述

Performance tab 是 CANNBot-Insight session 详情页的一个独立 tab。提供**耗时性能分析**和**Token 成本分析**两大功能，所有数据直接读 DB，不依赖 LLM 审计输出。

## 2. 架构

### 2.1 组件层级

```
page.tsx (session detail page)
├── Tab: "performance" (GaugeIcon 玫红色)
│   └── renderPerformance()
│       └── <PerfPanorama>  — 性能全景（可折叠）
│
├── Tab: "trace" (TraceView)   — 与 performance 共享 lazy-mount 策略
└── Tab: 其他 (overview/turns/...) — 常规 mount/unmount
```

### 2.2 懒挂载策略

| Tab | 挂载时机 | 切走时行为 |
|-----|----------|-----------|
| trace | 页面加载即挂载 | `hidden` CSS 隐藏，不卸载 |
| performance | 首次点击时挂载（`performanceMounted` 状态） | `hidden` 隐藏，不卸载 |
| 其他 | `activeTab` 匹配时挂载 | 卸载（丢失状态） |

### 2.3 回调

| 组件 | 回调 | 参数 | 目标 |
|------|------|------|------|
| PerfPanorama | `onJumpToTurn` | `turnIndex: number` | 查找非子代理 turn → `navigateToTab("turns", turnId)` |

### 2.4 折叠/展开

PerfPanorama header 可点击切换折叠状态，三角 `▶` 旋转 90° 表示展开。CSS `hidden` 隐藏内容，组件保持挂载不 re-fetch。

## 3. PerfPanorama — 性能全景

### 3.1 数据源

| API | 用途 |
|-----|------|
| `GET /api/observe/session/turns?taskId=...` | per-turn 性能数据 |
| `GET /api/observe/session/bridges?taskId=...` | 子代理 dispatch 配对（用于计算净 root 耗时） |

### 3.2 净 root 耗时计算

```
waitByTurn[turnId] = max(bridge.subagentLatencyMs)
                     ↑ 同一 turn 的并行 dispatch 取 max

netRootMs[turnId] = max(0, turn.latencyMs - waitByTurn[turnId])
                    ↑ 扣除子代理等待时间
```

root turn 的 `latencyMs` 包含等待子代理返回的壁钟时间，扣除后得到主 Agent 实际工作时间。

### 3.3 总览卡片

7 张卡片，响应式 grid（2/3/6 列）：

| 卡片 | 数值 | 颜色 |
|------|------|------|
| 主 Agent 耗时 | `rootMs`（净，已扣除子代理等待） | 红 `#dc2626` |
| 子代理耗时 | `subMs`（子代理 turns latencyMs 求和） | 橙 `#ea580c` |
| 总 Tokens | 所有 turns `totalTokens` 求和 | 蓝 `#2563eb` |
| Cache 命中率 | `cacheRead / (cacheRead + input) * 100%` | 绿 `#16a34a` |
| 主 Agent turns | root turn 计数 | — |
| 子代理 sessions | 唯一 `subagentSessionId` 计数 | — |
| 工具调用 | 所有 `toolCalls.length` 求和 | — |

### 3.4 面板详情

#### ① perf top — 子代理 Top 排序表

仿 Linux `top` 命令的可排序表格。

**表头**（居中对齐，可点击切换排序键，活跃列显示 `↓`）：

| 列 | 宽度 | 排序键 | 说明 |
|----|------|--------|------|
| SUBAGENT | 1fr | — | 三角 `▶` + 名称，下方显示占比条 |
| (占比) | 52px | — | 百分比文字，垂直居中 |
| TURNS | 56px | `turns` | turn 数 |
| TIME | 80px | `ms` | 总耗时 |
| TOKENS | 80px | `tokens` | 总 token |
| AVG/T | 72px | — | 平均 token/turn |

**占比条**：SUBAGENT 列内，名称下方显示水平条形图。占比 = 当前 subagent 排序值 / 所有 subagent 排序值之和。条色跟随排序键：TIME 红、TOKENS 蓝、TURNS 灰。

**下钻**：点击 subagent 行展开 turn 列表（默认 top 10，底部"展开全部"链接切换全部/收起）。下钻 turn 行显示：§N + 耗时 + tokens + tools 数 + model。

**列头点击**：切换排序键（降序），箭头 `↓` 始终占位（非活跃列 `opacity-0`），确保文字位置不偏移。

#### ② Token 成本分析

替换原 Token 构成环图，聚焦三个可行动维度：

**Cache 节省量**（卡片摘要）：
- Cache 节省：`cacheReadTokens` 总量 + 百分比 → 命中缓存，无需重发
- Cache 未命中：`inputTokens` 总量 + 百分比 → 冷启动/compaction 后重复输入，有优化空间
- 颜色：节省用绿 `#16a34a`，未命中用红 `#dc2626`

**每轮 Cache 命中趋势线**（SVG 折线图）：
- X 轴：turn index（root turns）
- Y 轴：cache 命中率 = `cacheReadTokens / (cacheReadTokens + inputTokens) * 100`
- compaction 事件标注（命中率骤降点）
- 异常低命中轮次标红
- 面积渐变填充

**perf top 下钻 cache 维度**：
- 下钻 turn 行增加 cache 命中率列
- 低命中 turn（<50%）标红，高亮优化目标

#### ③ Top 工具调用

- 数据：所有 turns 的 `toolCalls` 按 `durationMs` 降序 Top 10
- 水平条形图（红色 `#dc2626`）
- 标签：`toolName §turnIndex`
- 点击跳转到 turns tab

## 4. 颜色系统

| 用途 | 颜色 |
|------|------|
| Root / Input | `#2563eb` 蓝 |
| Subagent / Output / Cache 命中 | `#16a34a` 绿 |
| Reasoning | `#8b5cf6` 紫 |
| Cache Read | `#f59e0b` 橙 |
| Danger / 未命中 / 高耗时 | `#dc2626` 红 |
| 轴线 / 网格 | `#94a3b8` / `#e2e8f0` |

## 5. 文件清单

| 文件 | 职责 |
|------|------|
| `src/components/observe/PerfPanorama.tsx` | 性能全景（perf top + Token 成本分析 + Top 工具调用） |
| `src/app/session/[taskId]/page.tsx` | Tab 注册、lazy-mount、回调接线 |

## 6. 已知限制

1. **并行 dispatch 等待时间**：用 `max(subagentLatencyMs)` 估算，假设同一 dispatch turn 的多个 dispatch 是并行的
2. **耗时细分精度**：当 `toolCall.durationMs` 全为 0 时（常见于 claude-code 导入的 session），按 turn 级别分类，非精确 per-tool 分配
3. **Cache 命中率精度**：`cacheReadTokens` 和 `inputTokens` 来自 DB 记录，依赖 opencode 上报精度
