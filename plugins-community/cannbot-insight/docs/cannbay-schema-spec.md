# CANNBay 捕获/导出格式标准（x_cannbay v1）

适用范围：cpx 捕获文件、insight DB 导出件、CANNBay 仓内 `sessions/<sid>/` 布局的全部 jsonl/meta 数据。

## 0. 什么场景要看本文档

判定标准：数据离开当前进程、可能被另一个版本的代码或另一台机器读取的时刻。

| 场景 | 时刻 | 义务 |
|---|---|---|
| **生产** | cpx 捕获落盘（每 wire 轮）；cpx normalize；CANNBay 上传（直拷 / DB 导出）；任何写 `sessions/<sid>/` 布局的新工具 | 写方义务：信封不碰、扩展进口袋、带声明 |
| **消费** | cpx 退出自动导入 / 手动导入 / CANNBay 导入；turns API 读 Full Context 与 wire enrichment；任何解析这些文件的新代码 | 读方义务：按 `(schema, version)` 分发，不认识整块跳过，禁止字段嗅探 |
| **演进** | 加扩展字段、改字段语义、加新 payload schema | 见 §1 修改规则 |

不需要遵循的边界：无 `x_cannbay` 的老文件（legacy fallback，永久容忍）；纯 native claude jsonl；v1 CANNBay 的 `.db`；进程内数据结构（Prisma DB、RawInteraction 等）。

## 1. 修改规则（什么情况可以改、怎么改）

| 想做的事 | 允许？ | 怎么做 |
|---|---|---|
| data 里**加可选字段** | ✅ | 不升 version；读方必须容忍未知可选字段；同步更新本文档字段表与形状校验测试 |
| **改语义 / 改类型 / 删字段** | ✅ 但必升 version | `version` +1；读方注册表同时支持新旧版本；老 version 永久可读（不回收） |
| **加新 payload schema** | ✅ | 补齐本文档字段表、类型定义、形状校验、混排测试四项后合入 |
| **改信封层字段的含义**（`type` / `timestamp` / `message{role,content,id,model,usage}`） | ❌ 永久禁止 | 信封是 claude 原生语义，冻结是本标准的存在前提 |
| **复用信封字段装别的数据**（如 `duration_ms` 装 wire latency） | ❌（存量例外见 §6 双轨说明） | 一律进 `x_cannbay.data` |
| **改物理布局**（`sessions/<sid>/` 文件树） | ⚠️ 谨慎 | 布局是 adapter 发现约定（子代理目录、meta 命名），改动需同步 adapter 发现逻辑 |

加字段不升版本、改语义不升版本均视为破坏契约。每次 payload 改动须同步更新形状校验测试。

## 2. 物理布局

```
sessions/<sid>/
  <sid>.jsonl              # 主会话行流（信封 + x_cannbay）
  <sid>.meta.json          # 主会话元数据（cc-session-meta）
  subagents/
    <subId>.jsonl          # 子代理行流（同上行格式）
    <subId>.meta.json      # 子代理关联元数据（cc-subagent-meta）
```

cpx 本地目录（`~/.cannbot-insight/proxy/`）与 norm/ 镜像同一约定（顶层捕获文件带 `cpx-` 前缀，meta 与主文件同 stem 并列）。

## 3. 行格式与信封规则

```jsonl
{"type":"user","timestamp":"...","message":{"role":"user","content":[...]}}
{"type":"assistant","timestamp":"...","message":{"role":"assistant","id":"...","content":[...],"model":"...","usage":{...}},
 "x_cannbay":{"schema":"cc-wire-round","version":1,"data":{...}}}
```

字段类型记法：`int?` / `string?` = **可选字段**——没有该数据时 JSON 里该键不出现（缺失 ≠ 0，读方遇缺失跳过、不按 0 处理）；无 `?` = 必填。

四条规则：

1. **信封冻结**：`type` / `timestamp` / `message{role,content,id,model,usage}` 是 claude 原生语义，永不重解释。`message.usage` 为 anthropic 原生 usage 对象（`input_tokens` / `output_tokens` / `cache_read_input_tokens` / `cache_creation_input_tokens`），属信封层
2. **单一归属**：每个信息单元只有一个家。扩展一律进 `x_cannbay: {schema, version, data}`，信封不复用
3. **声明式分发**：读方按 `(schema, version)` 路由；不认识的声明整块跳过（行本体照常按信封处理）；禁止依据 data 内容或字段存在性判定身份
4. **双向兼容**：无 `x_cannbay` 的行 = native（现有路径逐字节不变）；新行被老代码读 = 忽略未知顶层字段

## 4. Payload Schema 定义

### 4.1 `cc-wire-round` v1（assistant 行，生产者 = cpx）

| 字段 | 类型 | 含义 | 导入归宿 |
|---|---|---|---|
| `roundIndex` | int | 该上下文的 wire 轮次号，从 0 连续 | wire 轮次对齐权威键 |
| `protocol` | `'anthropic'\|'openai'` | 线路协议 | session framework / 徽标 |
| `system` | verbatim | 该轮请求 `body.system` 原文 | Full Context（API 层扩展读） |
| `tools` | verbatim[] | 该轮请求 `body.tools` 原文 | Full Context |
| `requestParams` | `{temperature?, maxTokens?, model?}` | 请求侧参数（model 可与响应侧不同） | Turn.temperature/maxTokens（post-patch） |
| `latencyMs` | int | wire 往返耗时 | Turn.latencyMs |
| `ttftMs` | int? | 首 token 耗时（流式） | Turn.ttftMs（post-patch） |
| `stopReason` | string? | 响应 stop_reason | Turn.finishReason |
| `status` | int | wire 响应状态码 | 错误响应可观测 |
| `ccVersion` | string? | 计费头提取的 claude-code 版本 | Session.version |

### 4.2 `cc-wire-input` v1（user/system 行，生产者 = cpx）

| 字段 | 类型 | 含义 |
|---|---|---|
| `roundIndex` | int | 该增量消息所属 wire 轮 |
| `kind` | `'user'\|'tool-result'\|'injection'\|'system'\|'command-message'\|'dedup-placeholder'` | 结构化分类 |
| `dedup` | `{originalChars:int, fingerprint:string}` | 仅 `dedup-placeholder`：原文长度 + md5 指纹（占位文本保留在 content，UI 兼容） |

### 4.3 `cc-db-turn` v1（assistant 行，生产者 = insight-export）

| 字段 | 类型 | 含义 | 导入归宿 |
|---|---|---|---|
| `latencyMs` | int | DB 记录的轮耗时 | Turn.latencyMs |
| `ttftMs` | int? | DB 记录的 TTFT | Turn.ttftMs（post-patch） |
| `stopReason` | string? | finishReason | Turn.finishReason |
| `reasoningTokens` | int? | 推理 token（anthropic usage 无此字段） | Turn.reasoningTokens（`TokenUsage.reasoning` 既有通道） |
| `modelId` / `providerId` | string? | 模型/供应商标识 | Turn 同名列（post-patch） |
| `temperature` / `maxTokens` | number? | 请求参数 | Turn 同名列（post-patch） |
| `toolCalls` | `[{toolUseId, state?, errorType?, errorMessage?, durationMs?, startedAt?}]` | 工具调用明细 | ToolCall 明细列（post-patch） |
| `turnKind` | `'normal'\|'system'` | system 轮还原标记 | system turn 分类 |

### 4.4 `cc-session-meta` v1（`<sid>.meta.json`，主会话）

```json
{ "x_cannbay": { "schema": "cc-session-meta", "version": 1,
  "data": { "producer": "cpx|insight-export", "framework": "...", "protocol": "anthropic|openai", "ccVersion": "...", "sid": "..." } } }
```

文件级声明：生产者、原会话 framework、协议、版本。导入时恢复 Session.framework/version。

### 4.5 `cc-subagent-meta` v1（`subagents/<subId>.meta.json`）

```json
{ "toolUseId": "...", "name": "...", "agentType": "...",
  "x_cannbay": { "schema": "cc-subagent-meta", "version": 1,
  "data": { "toolUseId": "...", "name": "...", "agentType": "...", "subagentSessionId": "<subId>" } } }
```

顶层旧字段与 `x_cannbay.data` 同值双写（老代码读顶层，新代码读 data）。

## 5. 信息承载映射

每个信息单元恰好属于一类：

| 信息单元 | 落点 |
|---|---|
| 逐轮消息增量（user / tool_result / 注入原文）、响应 blocks、model、usage 四段 | 信封 verbatim |
| reasoning / latency / ttft / stopReason / system / tools / protocol / ccVersion / roundIndex / dedup / toolCall 明细 / 请求参数 / wire status | `x_cannbay` 直载（§4 各表） |
| subagent 关联（toolUseId / name / type） | `cc-subagent-meta` |
| session framework / version / producer | `cc-session-meta` |
| Turn.inputMessagesJson / contextWindowPct、Execution 聚合、SkillEvent、InteractionBridge、Session totals / label / query | **重推导**——导入管线从行内容重建，语义等价但不保证逐字节相同，schema 不直载 |

## 6. 读方分发规则（含双轨优先级）

```
行/文件解析：
  有 x_cannbay？
    ├─ 是 → 按 (schema, version) 查注册表
    │        ├─ 认识 → 按 payload schema 消费 data
    │        └─ 不认识 → 整块跳过（行本体照常按信封处理）
    └─ 否 → legacy 路径（现有 sniffing，永久保留）
meta.json：有 x_cannbay → 用声明式 data；无 → 读顶层旧字段
优先级：x_cannbay > legacy 字段 > 推断
```

**双轨说明（存量例外）**：过渡期 legacy 顶层字段（`source` / `duration_ms` / `stopReason` / `system` / `tools` / `ttftMs` / `version` / `deduped` / `originalChars`）与 `x_cannbay` 并存写、同值。其中 `duration_ms` 是历史遗留的**信封字段复用**（claude 原生语义为会话时长，本格式用它装 wire latency）——它是本标准唯一的规则 2 例外，收敛期结束后停止写入，读方优先 `x_cannbay.latencyMs`。

**收敛里程碑**：双生产者（claude-emitter + opencode-emitter）已对齐双轨写（P0-1 落地）。**下一个大版本**（v2 schema 上线时）停止写入 legacy 顶层字段（保留 `source` 作来源徽标例外），读方届时可只读 `x_cannbay`。在 v2 之前，双轨同值是硬性约束（由 `tests/cannbay-schema.test.ts` ②双写一致锁 + ⑥opencode-producer 形状锁守卫）。
