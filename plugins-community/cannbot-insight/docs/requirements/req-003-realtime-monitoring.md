# REQ-003: Real-time Session Monitoring

## 概述

实现 WebSocket 实时监听 opencode sessions.db 变化，支持在 session 运行过程中实时查看 turn 数据。

## 背景

当前只能导入已完成的 session，无法在 agent 运行过程中实时观察。对于长时间的 coding session，用户希望边跑边看，及时发现问题。

## 功能要求

### 1. 后端 WebSocket 服务

在 Next.js 中添加 WebSocket 支持（使用 `ws` 库 + Next.js custom server 或 API route + WebSocket upgrade）：

#### 方案选择

推荐使用 Next.js 的 `instrumentation.ts` 或自定义 server 方案：

```typescript
// src/lib/ws/server.ts
import { WebSocketServer } from 'ws';

export function setupWebSocket(server: HTTPServer) {
  const wss = new WebSocketServer({ server, path: '/ws/monitor' });
  wss.on('connection', (ws) => {
    // 处理连接
  });
}
```

#### WebSocket 协议

客户端连接时发送订阅消息：

```json
// 客户端 → 服务端：订阅某个 DB 文件
{"action": "subscribe", "dbPath": "/path/to/sessions.db"}

// 服务端 → 客户端：推送新增 turn
{"type": "turn", "data": {...TurnRow...}}

// 服务端 → 客户端：推送 session 更新
{"type": "session_update", "data": {...SessionRow...}}
```

### 2. DB 文件监听

使用轮询方式监听 opencode DB 变化（SQLite 不支持 inotify）：

```typescript
// src/lib/ws/db-watcher.ts
export function watchDb(dbPath: string, callback: (changes: Change[]) => void, intervalMs = 2000) {
  // 每 2 秒检查一次
  // 记录上次读取的最大 time_created
  // 查询新增的 message 记录
  // 转换为 RawInteraction 并通过 callback 返回
}
```

轮询逻辑：
1. 记录已读取的最新 message `time_created`
2. 每 2 秒查询 `WHERE time_created > lastTime ORDER BY time_created`
3. 将新消息转换为 `RawInteraction`
4. 调用 `splitIntoTurns` 生成 turn 行
5. 增量推送到 WebSocket 客户端

### 3. 前端实时页面

新建 `src/app/monitor/page.tsx`：

#### 3.1 连接面板

- 输入 opencode DB 文件路径
- "Connect" 按钮建立 WebSocket 连接
- 连接状态指示器（Connecting / Connected / Disconnected）

#### 3.2 实时 Turn 流

- 新 turn 到达时自动追加到 timeline
- 最新 turn 自动滚动到可视区域
- 正在进行的 turn 显示加载动画（如果 assistant 消息还未完成）

#### 3.3 实时指标

顶部指标卡实时更新：
- Turn 计数（user/assistant）
- 累计 Token 数
- 累计费用
- 已用时间

### 4. API 变更

不需要新的 REST API，数据通过 WebSocket 推送。但需要一个初始加载的 API：

```
GET /api/monitor/initial?dbPath=...&sessionId=...
```

返回当前已有的 turn 数据，WebSocket 连接后只推送增量。

## 实现要求

### 文件变更

| 文件 | 变更 |
|---|---|
| `src/lib/ws/server.ts` | **新建** — WebSocket 服务 |
| `src/lib/ws/db-watcher.ts` | **新建** — DB 轮询监听 |
| `src/lib/ws/types.ts` | **新建** — WS 消息类型定义 |
| `src/app/monitor/page.tsx` | **新建** — 实时监控页面 |
| `src/components/monitor/ConnectionPanel.tsx` | **新建** — 连接面板 |
| `src/components/monitor/LiveTurnStream.tsx` | **新建** — 实时 turn 流 |
| `src/components/monitor/LiveMetrics.tsx` | **新建** — 实时指标卡 |
| `src/hooks/useWebSocket.ts` | **新建** — WebSocket React hook |
| `next.config.ts` | 可能需要 custom server 配置 |
| `package.json` | 添加 `ws` 依赖 |
| `server.ts` | **新建** — 自定义 server 入口 |

### 设计约束

- WebSocket 断线后自动重连（指数退避，最多 5 次）
- 轮询间隔默认 2 秒，可通过 UI 调节（1s / 2s / 5s）
- 大量 turn 时做好虚拟滚动（复用 TurnTimeline 的渲染逻辑）
- 不修改现有 opencode DB 文件，只读方式打开

## 测试要求

- `tests/ws/db-watcher.test.ts` — DB 轮询逻辑测试
- `tests/ws/server.test.ts` — WebSocket 服务测试
- `tests/hooks/useWebSocket.test.ts` — Hook 测试

## 关键文件参考

- `src/lib/ingest/adapters/opencode-db.ts` — 复用 `readSession` 和 `listSessions`
- `src/lib/ingest/turn-split.ts` — 复用 `splitIntoTurns`
- `src/lib/ingest/data-service.ts` — 了解导入流程
- `src/components/observe/TurnTimeline.tsx` — 复用 turn 展示组件
- `src/components/observe/TurnDetail.tsx` — 复用 turn 详情组件
- `src/app/page.tsx` — 参考页面结构
