# REQ-002: Session Compare

## 概述

在首页支持选取两个 session 进行并排对比，展示 token、费用、耗时、工具调用、subagent 等指标的差异。

## 背景

同一类任务跑多次时，用户需要对比不同 session 的效率和成本差异，以评估优化效果（如 prompt 改进、模型切换等）。

## 功能要求

### 1. Session 选择

在首页 `SessionList` 组件中添加多选模式：

- 添加 checkbox 列，允许勾选 2 个 session
- 勾选 2 个后显示 "Compare Selected" 按钮
- 点击后跳转到 `/compare?sessions={id1},{id2}` 页面

### 2. Compare 页面

新建 `src/app/compare/page.tsx`，包含以下对比区域：

#### 2.1 概览对比卡片

左右并排展示两个 session 的核心指标，差异用颜色标记：

| 指标 | 说明 |
|---|---|
| Model | 模型名称 |
| Total Tokens | 总 token，较低者绿色 |
| Total Cost | 总费用，较低者绿色 |
| Latency | 总耗时，较短者绿色 |
| LLM Call Count | LLM 调用次数 |
| Tool Call Count | 工具调用次数 |
| Subagent Count | 子 agent 数量 |
| Skill Count | Skill 加载数 |

#### 2.2 Token 对比柱状图

并排柱状图对比 token 五项拆解：
- input / output / reasoning / cacheRead / cacheWrite

使用纯 SVG 绘制（项目不使用图表库），参考 `TokenBarChart.tsx` 的实现风格。

#### 2.3 Turn 时间线对比

左右各一个 TurnTimeline（可复用现有组件），展示两个 session 的 turn 时间线。

#### 2.4 Tool Call 汇总对比

表格对比两个 session 中各工具的使用次数和成功率：

| Tool Name | Session A Count | Session B Count | Diff |
|---|---|---|---|
| read_file | 15 | 8 | -7 |
| terminal | 10 | 12 | +2 |

### 3. API 支持

不需要新的 API 路由，直接使用现有的 `/api/observe/session` 和 `/api/observe/session/turns` 获取两个 session 的数据。

## 实现要求

### 文件变更

| 文件 | 变更 |
|---|---|
| `src/components/SessionList.tsx` | 添加 checkbox 多选和 "Compare Selected" 按钮 |
| `src/app/compare/page.tsx` | **新建** — 对比页面（Server Component） |
| `src/components/compare/CompareOverviewCards.tsx` | **新建** — 概览对比卡片 |
| `src/components/compare/CompareTokenChart.tsx` | **新建** — Token 对比柱状图（SVG） |
| `src/components/compare/CompareToolTable.tsx` | **新建** — 工具调用对比表 |

### 设计约束

- 图表使用纯 SVG，不使用 recharts 等图表库
- UI 组件使用 shadcn/ui（项目已有）
- 保持与现有页面一致的深色主题风格
- 响应式布局：桌面左右分栏，移动端上下堆叠

## 测试要求

- `tests/components/compare.test.ts` — 对比组件的单元测试
- 覆盖：指标计算、差异颜色判断逻辑、空数据处理

## 关键文件参考

- `src/app/page.tsx` — 首页，了解 SessionList 用法
- `src/components/SessionList.tsx` — Session 列表组件
- `src/components/observe/TokenBarChart.tsx` — SVG 柱状图参考
- `src/app/api/observe/session/route.ts` — Session 数据 API
- `src/app/api/observe/stats/route.ts` — 统计 API
