# Codex Proxy 适配 — 设计文档

## 目标

在现有 cpx proxy 框架上新增第三个 agent 框架适配：OpenAI Codex CLI。
产出的 jsonl 走 x_cannbay schema（cc-wire-round/cc-wire-input/cc-session-meta），
insight 原生导入即可消费，与 claude-code / opencode 两路形状对齐。

## Codex 特征（源码验证）

| 维度 | 值 |
|---|---|
| 协议 | OpenAI **Responses API**（`POST /v1/responses`，SSE 流） |
| wire_api | 仅 `"responses"`（`"chat"` 已移除，硬错误） |
| 配置文件 | `~/.codex/config.toml`（TOML） |
| proxy hook | `openai_base_url = "http://127.0.0.1:<port>"` 或自定义 `model_providers` 的 `base_url` |
| API key | `env_key` 指向环境变量名（如 `OPENAI_API_KEY`） |
| 请求体 | `{input: [...items], tools: [...], model: "...", stream: true}` |
| 响应 SSE | `response.created` → `response.output_item.added` → `response.output_text.delta` → `response.output_text.done` → `response.function_call_arguments.delta` → `response.function_call_arguments.done` → `response.output_item.done` → `response.completed` |
| input item 类型 | `message`（role + content[{type:"input_text",text}]）、`function_call`（name+arguments+call_id）、`function_call_output`（call_id+output） |
| output item 类型 | `message`（content[{type:"output_text",text}]）、`function_call`（name+arguments+call_id）、`reasoning`（summary/content） |
| 子代理 | `[agents]` config roles + multi-thread（与 claude 的 Task tool / opencode 的 task tool 机制不同） |
| skills | `.codex/skills/<name>/SKILL.md`（同 claude/opencode 约定） |
| session 存储 | SQLite state DB + `~/.codex/history.jsonl`（与 claude jsonl 不同，不走 claude 原生导入） |

## 与现有框架的差异（需要新代码的部分）

| 差异 | 影响 | 方案 |
|---|---|---|
| **第三个协议**（Responses API ≠ Anthropic ≠ Chat Completions） | `Protocol` type、reassembler、emitter 都需新增 | 新增 `Protocol = 'responses'`、`ResponsesApiReassembler`、`codex-emitter.ts` |
| **item-based 请求体**（`input` 不是 `messages`） | emitter 的 delta-emit 逻辑不同（input items ≠ messages array） | codex-emitter 自行解析 input items，映射到 user/assistant 行 |
| **function_call 独立 item**（不是 message 内 block） | tool_use 提取逻辑不同 | reassembler 把 function_call item → AnthropicContentBlock {type:"tool_use"} |
| **function_call_output 独立 item** | tool_result 提取逻辑不同 | emitter 把 function_call_output item → user 行的 tool_result block |
| **TOML 配置** | cpx-cli 的 profile 解析不同（TOML ≠ JSON） | cpx-cli 读 `~/.codex/config.toml`，提取 `openai_base_url` |
| **无 x-session-id / cc_is_subagent** | 子代理路由信号不同 | codex 的子代理路由信号待验证（header？request body？）—— 初版按单会话处理，子代理后续迭代 |

## 架构

```
codex CLI
  │ POST /v1/responses (SSE)
  ▼
┌─ cpx proxy ────────────────────────────────────────────┐
│  server.ts          统一 HTTP 拦截/转发                  │
│  session-resolver   按 /v1/responses 路径 → protocol:'responses' │
│  stream-reassembler ┌ AnthropicReassembler             │
│                     ├ OpenAIReassembler                │
│                     └ ResponsesApiReassembler (新增)   │
│  codex-emitter.ts   (新增) responses → claude-format   │
│  normalize/         (既有) 布局归一                       │
│  writer.ts          (既有) 落盘 + x_cannbay             │
└────────────────────────────────────────────────────────┘
  ↓ norm/<sid>.jsonl (cc-wire-round, protocol:'responses')
insight 原生导入 → DB → UI
```

## 模块设计

### 1. Protocol type + session-resolver

```typescript
// types.ts
export type Protocol = 'anthropic' | 'openai' | 'responses';

// session-resolver.ts
export function protocolFromPath(urlPath: string): Protocol | null {
  if (urlPath.includes('/v1/responses')) return 'responses';     // ← 新增
  if (urlPath.includes('/v1/messages')) return 'anthropic';
  if (urlPath.includes('/v1/chat/completions')) return 'openai';
  return null;
}
```

### 2. ResponsesApiReassembler（stream-reassembler.ts）

Responses API SSE 事件流 → AnthropicContentBlock[] + usage。

```
事件处理：
  response.created              → 记 model、usage.input_tokens
  response.output_item.added    → 按 item.type 建块（message→text, function_call→tool_use）
  response.output_text.delta    → 累积 text
  response.function_call_arguments.delta → 累积 tool_use input JSON
  response.output_item.done     → 定型块（补 id/name/input）
  response.completed            → 取 usage.output_tokens、stop_reason

输出：{ model, stop_reason, content: AnthropicContentBlock[], usage }
```

### 3. codex-emitter.ts

独立的第三个 emitter（与 claude-emitter / opencode-emitter 同级）。

**输入**：`ProxyRecord`（protocol='responses'），request.body 是 Responses API 请求体。

**请求体结构**：
```json
{
  "input": [
    {"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]},
    {"type":"function_call","name":"shell","arguments":"{...}","call_id":"call_xxx"},
    {"type":"function_call_output","call_id":"call_xxx","output":"result text"}
  ],
  "tools": [{"type":"function","name":"shell","description":"...","parameters":{...}}],
  "model": "gpt-5",
  "instructions": "You are codex...",
  "stream": true
}
```

**emit 逻辑**：

| input item | claude jsonl 行 |
|---|---|
| `message` role=user | user 行（content = content blocks） |
| `message` role=assistant | 跳过（是 prior response，由 assistant 行覆盖） |
| `message` role=system/developer | 跳过（system 在 assistant 行扩展字段） |
| `function_call` | 跳过（是 prior assistant 的 tool_use，由 assistant 行覆盖） |
| `function_call_output` | user 行 with tool_result block（tool_use_id = call_id） |

**assistant 行**（response）：
```json
{
  "type": "assistant",
  "message": {"role":"assistant","id":"<resp_id>","content":[reassembled blocks],"model":"...","usage":{...}},
  "timestamp": "...",
  "system": body.instructions,          // codex 的 system prompt 在 instructions 字段
  "tools": body.tools,                  // verbatim
  "source": "codex-proxy",
  "x_cannbay": {"schema":"cc-wire-round","version":1,"data":{
    "protocol": "responses",
    "roundIndex": ...,
    "system": body.instructions,
    "tools": body.tools,
    "requestParams": {"model": body.model},
    "latencyMs": rec.latencyMs,
    "ttftMs": rec.ttftMs,
    "stopReason": rec.response.stop_reason,
    "status": rec.response.status
  }}
}
```

**delta-emit**：Responses API 的 `input` 数组是 append-only（每轮把上一轮的 input 全发回 + 新增），与 opencode 的 `messages` 相同模式。用 prevInputCount cursor 做 delta。

**session-meta**：
```json
{
  "x_cannbay": {"schema":"cc-session-meta","version":1,"data":{
    "producer": "cpx",
    "framework": "codex",
    "protocol": "responses",
    "sid": "..."
  }}
}
```

### 4. cpx-cli codex profile

```
cpx codex [args...]              # 启动 codex，注入 openai_base_url 指向 proxy
```

- 读 `~/.codex/config.toml`（如果存在）取 API key 环境变量名
- 设置 `OPENAI_BASE_URL` env 或 `openai_base_url` config → proxy port
- 启动 codex 子进程
- 退出后 normalize → 导入（与 claude/opencode 同款流程）

### 5. proxySourceMarker 扩展

```typescript
export function proxySourceMarker(protocol: Protocol): string {
  if (protocol === 'openai') return 'opencode-proxy';
  if (protocol === 'responses') return 'codex-proxy';   // ← 新增
  return 'claude-proxy';
}
```

### 6. server.ts dispatchEmit 扩展

```typescript
function dispatchEmit(rec: ProxyRecord): void {
  if (rec.protocol === 'anthropic') claudeEmit(rec);
  else if (rec.protocol === 'responses') codexEmit(rec);  // ← 新增
  else opencodeEmit(rec);
}
```

### 7. server.ts resolveUpstreamUrl 扩展

Responses API 请求转发到 OpenAI 上游（`https://api.openai.com/v1`）或自定义 provider base_url。

## 适配性设计

| 变化场景 | 是否需改代码 | 原因 |
|---|---|---|
| Responses API 加新事件类型 | 是 | reassembler 注册表加分支 |
| codex 配置改 config 格式 | 可能 | cpx-cli 的 config 读取要跟 |
| codex 加子代理（multi-thread 路由信号） | 是 | 初版单会话，子代理信号待验证后加 |
| Responses API 加新 input/output item 类型 | 是 | emitter/reassembler 映射表加分支 |

## 技术选型

| 组件 | 选型 | 理由 |
|---|---|---|
| TOML 解析 | 手写轻量提取（不引入 toml 依赖） | 只需提取 `openai_base_url` / `env_key` 几个字段 |
| Responses API SSE 解析 | 同 AnthropicReassembler 模式（buffer + line + event dispatch） | 事件驱动，与现有 reassembler 同构 |
| output → AnthropicContentBlock 映射 | 手写 switch on item.type | 类型有限，不需通用映射框架 |

## 不做的事

- 不做 codex 子代理路由（初版）—— codex 的 multi-thread 路由信号待实测
- 不做 codex session DB 解析 —— 走 proxy 捕获即可（capture ≠ interpret）
- 不做 Responses API WebSocket transport —— 初版只处理 HTTP SSE
- 不做 codex skills 解析 —— skills 在 `.codex/skills/`，与 claude/opencode 同结构，由 insight adapter 消费
