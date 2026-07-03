# REQ-001: Claude Code JSONL Adapter

## 概述

实现 Claude Code session JSONL 文件的解析适配器，使用户可以导入 Claude Code 的会话数据到 CANNBot-Insight。

## 背景

当前系统只支持 opencode 的 SQLite sessions.db 格式。Claude Code 将 session 数据保存为 JSONL 文件（每行一个 JSON 对象），需要开发一个 adapter 来解析并转换为系统内部的 `RawInteraction[]` 格式。

## Claude Code JSONL 文件格式

Claude Code 的 session 文件位于 `~/.claude/projects/*/sessions/` 目录下，每个 session 是一个 `.jsonl` 文件，每行一个 JSON 对象，包含以下类型：

### 消息类型

1. **user message**: `{"type":"user","message":{"role":"user","content":"..."}}`
2. **assistant message**: `{"type":"assistant","message":{"role":"assistant","content":[...],"model":"...","usage":{...}}}`
3. **result**: `{"type":"result","subtype":"success","result":"...","cost_usd":0.01,"duration_ms":1234}`

### assistant message.content 数组元素

- `{"type":"text","text":"..."}`
- `{"type":"tool_use","id":"...","name":"...","input":{...}}`
- `{"type":"tool_result","tool_use_id":"...","content":"..."}`

### usage 字段

```json
{
  "input_tokens": 100,
  "output_tokens": 200,
  "cache_read_input_tokens": 50,
  "cache_creation_input_tokens": 30
}
```

## 实现要求

### 1. 新建 adapter 文件

创建 `src/lib/ingest/adapters/claude-jsonl.ts`，导出以下函数：

```typescript
export function listSessions(dirPath: string): SessionListItem[]
export function readSession(filePath: string, sessionId: string): RawInteraction[]
```

- `listSessions`: 扫描目录下的 `.jsonl` 文件，返回 session 列表
  - `dirPath` 可以是包含 JSONL 文件的目录
  - 从文件内容中提取 session 元数据（首条用户消息、时间、模型名等）
- `readSession`: 读取单个 JSONL 文件，解析为 `RawInteraction[]`
  - `filePath` 是 JSONL 文件路径
  - `sessionId` 可以从文件名推导

### 2. 注册 adapter

修改 `src/lib/ingest/adapters/index.ts`，将 `case 'claude-jsonl'` 从 `return null` 改为返回实际的 adapter 实例。

### 3. 转换逻辑

将 Claude Code JSONL 数据转换为 `RawInteraction` 接口：

| JSONL 字段 | RawInteraction 字段 | 说明 |
|---|---|---|
| `type` | `role` | "user"/"assistant"/"result" |
| `message.content` | `content` | 提取文本内容 |
| 行号 | `timestamp` | 如果没有时间戳，用文件修改时间 + 行号偏移 |
| `message.model` | `model` | 模型名称 |
| `message.usage` | `usage` | token 用量映射 |
| tool_use / tool_result | `tool_calls` | 转换为 `ToolCallInfo[]` |

### 4. token 映射

```typescript
usage: {
  total: input_tokens + output_tokens,
  input: input_tokens,
  output: output_tokens,
  reasoning: 0, // Claude Code 暂无 reasoning token
  cacheRead: cache_read_input_tokens ?? 0,
  cacheWrite: cache_creation_input_tokens ?? 0,
  cost: 0, // 后续由 cost-calculator 计算
}
```

### 5. tool call 处理

Claude Code 的 tool_use 和 tool_result 是分开的内容块，需要配对：
- 在 assistant message 中找到 `type: "tool_use"` 块
- 在后续 user message 中找到对应的 `type: "tool_result"` 块（通过 `tool_use_id` 匹配）
- 合并为 `ToolCallInfo` 对象

### 6. Session ID 推导

- 文件名格式通常是 `{session-id}.jsonl`
- 如果目录下有多个文件，每个文件就是一个 session

### 7. 导入入口适配

修改 `LocalFileImport.tsx` 和 `import-file/sessions/route.ts`，支持选择目录或 JSONL 文件，自动识别 sourceType 为 `claude-jsonl`。

## 测试要求

在 `tests/adapters/` 下创建 `claude-jsonl.test.ts`，覆盖：
- 解析空文件
- 解析包含 user/assistant/result 的完整 session
- tool_use / tool_result 配对
- 多 session 文件扫描
- 错误处理（格式错误、文件不存在）

## 关键文件参考

- `src/lib/ingest/adapters/opencode-db.ts` — 参考现有 adapter 实现
- `src/lib/ingest/adapters/index.ts` — adapter 注册
- `src/lib/shared/types.ts` — `SessionListItem`, `RawInteraction`, `ToolCallInfo`, `TokenUsage` 接口
- `src/lib/ingest/data-service.ts` — `importSession` 函数，了解 adapter 如何被消费
- `tests/adapters/opencode-db.test.ts` — 参考现有测试
