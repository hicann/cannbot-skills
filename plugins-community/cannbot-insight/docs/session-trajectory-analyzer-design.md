# Session Trajectory Analyzer — 设计文档

## 目标

输入轨迹 MD 文件，输出分析 JSON 文件。不依赖 Claude Code，自建轻量 agent。

## 方案选型

### 核心思路：正则预处理 + 有界 LLM 对话

轨迹 MD 动辄 10 万行，LLM context 放不下。但其中大部分是子代理 turn 正文，分析只需要：

- **结构骨架**（turn 列表 + skill 调用序列）— 正则提取
- **SKILL.md 全文**（workflow 定义）— 正则定位截取
- **Stats 表**（token/turn 汇总）— 正则解析
- **门控结果 + Error 段落** — 正则定位 + 定点截取
- **质量评估 + JSON 生成** — LLM 推理

正则把 10 万行压到 ~5-15K tokens 的摘要，交给 LLM。但纯正则单轮不够——分析过程中可能需要看某个子代理的 turn 正文、crash 前后更多上下文、DESIGN.md 某节等，正则写不全。

**解决方案：固定正则 pipeline + LLM 可发定点读请求，有界循环。**

```
Round 1:  正则提取骨架+摘要 → 喂 LLM
          LLM 输出两种可能：
            A) 完整 JSON → 校验 → 写文件 → 结束
            B) {"read": {"lines": [9000, 9500]}} 或 {"read": {"section": "§84.1"}} → 继续

Round 2:  代码截取请求段落 → 追加喂 LLM → LLM 再判
          （最多 3 轮，防死循环）
```

LLM 只能发一种"工具"——读指定行范围或 section。代码截取后喂回。不开放任意 tool，不需要 agent 框架。

### 为什么不用多步 agent / tool-use

| 方案 | 否决理由 |
|------|----------|
| 全文塞 LLM | 10 万行超 context |
| 多轮 tool-use（LLM 自由选 grep/read/bash） | 需要完整 agent 框架；LLM 可能跑偏死循环 |
| 纯正则单轮（无读请求） | 扩展不了——以后要读子代理 turn 正文/DESIGN 某节/crash 更多上下文，正则写不全 |
| 纯正则/规则（无 LLM） | 质量评估需要语义理解，规则写不死 |

本方案是"正则单轮"和"多轮 agent"的折中：正则打底，LLM 有界补充读。

### 为什么不把 prompt 解析成结构化配置

session-trajectory-analyse.md 会变化（维度增减、字段改名、规则调整）。如果写代码解析这份 MD 提取规则，解析器本身又需要维护——换了个遍。

**直接把 prompt 文件全文作为 LLM system prompt**。LLM 按 prompt 指令执行，prompt 改了 LLM 自动跟。agent 代码只管「正则提取 → 拼 prompt → 调 LLM（有界循环）→ 校验 JSON → 写文件」。

## 架构

```
┌──────────────────────────────────────────────────────────┐
│ 前端 WorkflowAIView.tsx（复用现有配置框）                  │
│   apiKey / baseUrl / model → localStorage                 │
└──────────────────────────┬───────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│ POST /api/ai/analyze-trajectory  (Next.js API Route)     │
│   body: { trajectoryPath, provider: { apiKey,baseUrl,model } }│
└──────────────────────────┬───────────────────────────────┘
                           ▼
轨迹 MD ──→ [TrajectoryParser] ──→ 结构化摘要
                                    │
                                    ├─ skeleton:    [{turn, skill, type, status}] + occurrences
                                    ├─ skillMd:      首次 invoke 的 SKILL.md 原文
                                    ├─ stats:        {tokens, turns, subagents, duration, cost}
                                    ├─ gates:        [{turn, result, snippet}]
                                    └─ errors:      [{turn, lineRange, snippet}]
                                    │
                                    + session-trajectory-analyse.md (全文)
                                    │
                                    ▼
                    ┌──→ [LLM Caller] ←──┐
                    │      (有界循环)      │
                    │                    │
                    │    LLM 输出 JSON?   │
                    │    ├ Yes → 结束     │
                    │    └ No  → 读请求   │
                    │         ↓           │
                    │  [SectionReader]    │
                    │  截取行段 → 喂回 ───┘
                    │
                    ▼
              JSON 输出
                    │
                    ▼
          [SchemaValidator] → [FileWriter] → logs/xxx-analysis.json
                    │
                    ▼
          前端读取 → WorkflowFlowChart 渲染
```

## 模块设计

### 1. TrajectoryParser

纯正则，无 LLM。轨迹 MD 的格式由 CANNBot-Insight 导出决定，稳定不变。

#### 1.1 SkeletonExtractor

提取 turn 列表 + skill 调用序列。

```typescript
const PATTERNS = {
  turn:       /^## §(\d+) /,                      // 主 turn 标题
  skill_call: /^\*Skill: (\S+) \((\w+)\) (✅|❌)/, // skill 调用
  tool_skill: /^\*\*Tool: skill\*\*/,              // invoke 输出块起点
  tool_task:  /^\*\*Tool: task\*\*/,               // dispatch 输出块起点
} as const
```

输出：
```json
[
  {"turn": 2, "skill": "ops-registry-invoke-glacier", "type": "invoke", "status": "ok", "line": 27},
  {"turn": 4, "skill": "developer", "type": "dispatch", "status": "ok", "line": 190},
  {"turn": 5, "skill": "verifier", "type": "dispatch", "status": "ok", "line": 452}
]
```

#### 1.2 SkillContentExtractor

从首次 `*Skill: xxx (invoke)` 后的 `**Output:**` 块中截取 `<skill_content name="...">...</skill_content>`。

输出：SKILL.md 全文字符串 + skillName。

#### 1.3 StatsParser

读末尾 50 行，正则解析 `## Stats` 表格。

```typescript
const PATTERNS = {
  duration:   /\*\*Duration:\*\*\s*(\S+)/,
  tokens:     /\*\*Tokens:\*\*\s*(\S+)/,
  // Stats 表行
  stats_row:  /\|\s*(\w+)\s*\|\s*([\d.]+[MK]?)\s*\|\s*([\d.]+[MK]?)\s*\|\s*(\d+)\s*\|/,
} as const
```

#### 1.4 GateResultExtractor

定位门控结果，排除测试用例噪音。

```typescript
const PATTERNS = {
  pass:      /^(PASS)\s*$/,           // 独立行的 PASS（门控判定）
  fail:      /^(FAILED|FAIL)\s*$/,   // 独立行的 FAIL
  skill_err: /^\*Skill:.*❌\*/,      // skill 调用失败
  tool_err:  /^\*Error:/,            // 工具执行错误
} as const
```

**噪音过滤**：`[FAIL]` / `FAIL:` 带冒号的（测试用例输出）不作为门控结果，只取独立行 PASS/FAIL。进一步检查是否在 `<details>` 块内（子代理折叠块内的 PASS/FAIL 是测试结果不是门控）。

#### 1.5 ErrorContextExtractor

对每个 Error/❌ 点，截取前后各 15 行作为上下文，供 LLM 诊断根因。

### 1.6 SectionReader

LLM 在有界循环中发读请求时，由 SectionReader 截取段落。两种请求：

- `{"lines": [from, to]}` — 直接截取行范围
- `{"section": "§N.M"}` — 正则定位 `### **§N.M**` 到下一个同级标题，截取该段

输出截取文本，追加到 LLM 对话历史。

### 2. LLM Caller（有界循环）

#### 循环逻辑

```typescript
const MAX_ROUNDS = 3

for (let round = 0; round < MAX_ROUNDS; round++) {
  const resp = await llm.call({ system: promptMd, messages })
  const parsed = tryParseJson(resp)

  if (parsed) return parsed  // 完整 JSON，结束

  const readReq = tryParseReadRequest(resp)
  if (readReq) {
    const snippet = sectionReader.read(trajectoryText, readReq)
    messages.push({ role: "assistant", content: resp })
    messages.push({ role: "user", content: `已读取：\n${snippet}` })
    continue
  }

  // 既不是 JSON 也不是读请求，带错误重试
  messages.push({ role: "user", content: '输出格式错误，请输出完整 JSON 或 {"read": {"lines": [from, to]}}' })
}

throw new AnalysisError(`LLM ${MAX_ROUNDS} 轮未产出有效 JSON`)
```

#### LLM 输出格式

每轮 LLM 输出二选一：

| LLM 输出 | 含义 | 代码动作 |
|----------|------|----------|
| `{"sessionSummary": "...", ...}` 完整 JSON | 分析完成 | 校验 → 写文件 |
| `{"read": {"lines": [9000, 9500]}}` | 要读行段 | 截取 → 追加喂回 |
| `{"read": {"section": "§84.1"}}` | 要读子代理 turns | 定位 → 截取 → 追加喂回 |

#### 初始 Prompt 构造

```
System: <session-trajectory-analyse.md 全文>

User:
## 轨迹骨架
<json: skeleton + occurrences>

## Workflow SKILL.md 原文
<skillMd>

## Stats
<json: stats>

## 门控结果
<json: gates>

## 异常段落
<json: errors with context snippets>

请按 prompt 指令输出分析 JSON。
若需更多上下文，输出 {"read": {"lines": [from, to]}} 或 {"read": {"section": "§N.M"}}。
最多 3 轮读请求。
```

#### 输出约束

- `response_format: json_object`（OpenAI 兼容）
- 代码层 `json.loads` 判定是否完整 JSON
- 非 JSON 时尝试解析 `read` 请求

### 3. SchemaValidator

不做严格 JSON Schema（schema 随 prompt 变化）。只校验：

- 顶层 7 个 key 齐全：`sessionSummary` / `workflowMeta` / `sessionMeta` / `flow` / `skillQuality` / `workflowLevelIssues` / `optimizationPriorities`
- `sessionMeta` 的三个数组字段存在且为数组（`cpsExecuted` / `cpsMissing` / `phasesNotReached`），空也填 `[]`
- `flow` 和 `skillQuality` 是数组

这三个校验是组件渲染的硬依赖，不会随 prompt 变化。

### 4. FileWriter

写单行紧凑 JSON 到 `logs/{轨迹文件名}-analysis.json`，终端输出路径。

## 适配性设计

| 变化场景 | 是否需改代码 | 原因 |
|----------|-------------|------|
| prompt 增减分析维度（如加 G6） | 否 | LLM 按 prompt 全文执行 |
| prompt 改字段名（如 gates→cps） | 否 | 同上 |
| prompt 改 JSON schema 结构 | 否 | 同上；SchemaValidator 只校验组件硬依赖 |
| CANNBot-Insight 导出格式变（§N 改为 turn-N） | 是 | 正则 patterns 写在配置常量里，改配置即可 |
| 新增轨迹来源（非 opencode） | 是 | 新增 Parser 子类 |

正则 patterns 集中在一个 `TRAJECTORY_PATTERNS` 配置常量中，不散落在代码里。

## 技术选型

集成进 cannbot-insight Next.js 应用，复用现有 AI 配置基础设施。

| 组件 | 选型 | 理由 |
|------|------|------|
| 语言 | TypeScript | 与 cannbot-insight 同栈，前端配置直接传后端 |
| 运行环境 | Next.js API Route (Node.js runtime) | 无需独立进程，前后端一体 |
| LLM API | OpenAI 兼容 `/chat/completions` | 复用现有 `analyzer.ts` 调用逻辑 |
| 正则 | JS `RegExp` 标准库 | 轨迹 MD 格式规整，不需高级特性 |
| JSON 校验 | `JSON.parse` + 手写检查 | 不引入 ajv/jsonschema 依赖 |
| 配置 | 前端 localStorage（复用 WorkflowAIView） | 与 cannbot-insight 现有 AI 配置方式一致 |

## LLM 配置

复用现有 `WorkflowAIView.tsx` 的 AI Provider 配置（apiKey / baseUrl / model），不另建配置入口。

```
前端 WorkflowAIView.tsx
  ├─ apiKey   (localStorage)   ← sk-xxx
  ├─ baseUrl  (localStorage)   ← https://dashscope.aliyuncs.com/v1
  └─ model    (localStorage)   ← qwen-max / gpt-4o / ...
        │
        ▼
  POST /api/ai/analyze-trajectory
  body: { trajectoryPath, provider: { apiKey, baseUrl, model } }
        │
        ▼
  后端 route.ts → TrajectoryParser → LLMCaller(provider) → FileWriter
```

用户在前端填一次配置（同现有 Workflow 分析 tab 的配置框），trajectory 分析器直接复用。环境变量方式（`ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL`）仅作后端 fallback——前端未配置时从 `process.env` 读取。

## 接口

### 前端 → 后端 API

```
POST /api/ai/analyze-trajectory
Content-Type: application/json

{
  "trajectoryPath": "logs/log1.md",          // 轨迹 MD 文件路径（相对项目根）
  "provider": {                               // 可选，缺省从 env 读
    "apiKey": "sk-xxx",
    "baseUrl": "https://dashscope.aliyuncs.com/v1",
    "model": "qwen-max"
  }
}

Response 200: { "outputPath": "logs/log1-analysis.json" }
Response 400: { "error": "trajectory file not found" }
Response 502: { "error": "LLM call failed: ..." }
```

### 后端内部函数

```typescript
async function analyzeTrajectory(opts: {
  trajectoryPath: string
  promptPath?: string                  // 默认 prompts/session-trajectory-analyse.md
  outputDir?: string                   // 默认 logs/
  provider: { apiKey: string; baseUrl: string; model: string }
}): Promise<{ outputPath: string }>
```

前端拿到 `outputPath` 后，读取该 JSON 文件内容粘进 Workflow 分析 tab 渲染（同现有手动流程）。

## 不做的事

- 不做 LLM 自由 tool-use（LLM 不能自己选 grep/read/bash）— 只开放定点读请求，有界 3 轮
- 不做 agent 框架（planning/reflection）— 固定 pipeline
- 不解析 prompt MD 提取结构化规则 — 全文喂 LLM
- 不做增量分析 — 每次全量
- 不做并行 LLM 调用 — 串行有界循环
