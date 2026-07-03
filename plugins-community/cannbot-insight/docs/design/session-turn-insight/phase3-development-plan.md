# Session 对话级洞察 — Phase3 开发计划（TDD 方式）

版本：v0.4  
最后更新：2026-06-12

> 开发策略：**先实现 opencode DB 全链路（端到端可跑通），Claude Code JSONL 适配后续迭代**  
> 开发方式：**开发者测试驱动** — 每个模块先写测试（用真实/合成数据），再实现，测试通过才算完成  
> 状态标记：⬜ 未开始 ｜ 🟡 进行中 ｜ ✅ 已完成

---

## §1 开发原则

### 1.1 TDD 流程

每个功能模块的开发遵循：

```
准备测试数据 → 写测试 → 实现 → 测试通过 → 下一个模块
```

- **测试数据**：用真实的 opencode sessions.db 作为测试输入；无真实数据时用合成 JSON 构造
- **先写测试**：每个模块先写单元测试或集成测试，定义"输入→期望输出"
- **实现到测试通过**：实现模块逻辑，跑测试，通过后才算完成该模块
- **持续可运行**：每完成一个模块，整个系统必须处于可运行状态

### 1.2 测试数据准备

| 数据类型 | 来源 | 用途 |
|-|-|-|
| opencode sessions.db | 从实际 opencode 使用中导出一个含 3+ session 的 DB 文件 | adapter 测试 + 全链路集成测试 |
| opencode 合成 JSON | 手工合成一个含 root + 1 subagent + skill load 的完整 RawInteraction 数组 | turn-split / bridge-builder / execution-split 测试 |

测试数据文件放在 `tests/data/` 目录下。

---

## §2 开发步骤

### Step 1：项目骨架 + 测试框架（1 天）

| 编号 | 状态 | 任务 | 验收 |
|-|-|-|-|
| S1-01 | ✅ | 初始化 Next.js 项目：`npx create-next-app` + App Router + TypeScript | `npm run dev` 可启动 |
| S1-02 | ✅ | 安装依赖：Prisma + better-sqlite3 + shadcn/ui + tailwindcss + vitest | `npm install` 无报错 |
| S1-03 | ✅ | Prisma schema 萬地（Phase2 §2.1 全部 8 个 model）+ `npx prisma migrate` | DB 文件生成，`npx prisma studio` 可打开 |
| S1-04 | ✅ | 配置 vitest：`vitest.config.ts` + `tests/setup.ts` | `npm run test` 可运行（空测试通过） |
| S1-05 | ✅ | 基础 UI shell：layout.tsx + globals.css + shadcn/ui 初始化 | `npm run dev` 页面可访问 |
| S1-06 | ✅ | 准备测试数据：收集真实 opencode DB 放入 `tests/data/`；合成含 subagent + skill 的 JSON | 测试数据文件存在且格式正确 |

### Step 2：数据源适配器 — opencode-db（1 天）

**TDD 流程**：先写 `listSessions()` 和 `readSession()` 的测试 → 实现 → 测试通过

| 编号 | 状态 | 任务 | 验收 |
|-|-|-|-|
| S2-01 | ✅ | 写测试 `tests/adapters/opencode-db.test.ts`：`listSessions()` 返回 session 列表（含 id/创建时间/第一个提示词/turn 数/模型名）；`readSession(sessionId)` 返回 RawInteraction[]；空 DB / 无 session / 格式不兼容的错误处理 | 测试写好（先 failing） |
| S2-02 | ✅ | 实现 `adapters/opencode-db.ts`：用 better-sqlite3 打开外部 DB → 查询 sessions/messages/tools 表 → `listSessions()` 和 `readSession()` | S2-01 测试全部通过 |
| S2-03 | ✅ | 手工验证：CLI 或脚本调用 `listSessions()` 对真实 DB，输出 session 列表，确认时间/提示词/turn 数可读 | 真实 DB 的 session 列表输出正确 |

### Step 3：适配器注册表 + normalize（0.5 天）

> 注：claude-jsonl adapter 暂不实现，本轮只跑通 opencode DB 全链路

| 编号 | 状态 | 任务 | 验收 |
|-|-|-|-|
| S3-01 | ✅ | 写测试 `tests/adapters/index.test.ts`：source type "opencode-db" 路由到 opencode-db adapter | 测试通过 |
| S3-02 | ✅ | 实现 `adapters/index.ts`（只注册 opencode-db，claude-jsonl 留 stub） | S3-01 通过 |
| S3-03 | ✅ | 写测试 `tests/normalize.test.ts`：opencode 格式 RawInteraction 归一化 | 测试通过 |
| S3-04 | ✅ | 实现 `normalize.ts`（只处理 opencode 格式） | S3-03 通过 |

### Step 4：turn-split — 核心拆解（1 天）

**TDD 流程**：用合成 JSON 数据先写测试 → 实现 → 测试通过 → 用真实 DB 数据再验证

| 编号 | 状态 | 任务 | 验收 |
|-|-|-|-|
| S4-01 | ✅ | 写测试 `tests/turn-split.test.ts`：RawInteraction[] → Turn[] + ToolCall[] + SkillEvent[]；token 五项分项赋值；时间字段赋值；skill/invoke/load 识别；assistant turn 的 inputMessages 重构；contentSummary 截断（≤200字）；contextWindowPct 计算；role/agent_identity 正确 | 测试写好（先 failing） |
| S4-02 | ✅ | 实现 `turn-split.ts` | S4-01 测试全部通过 |
| S4-03 | ✅ | 用真实 opencode DB 数据验证：`readSession()` → `turn-split()`，确认真实数据拆解结果合理 | 真实数据拆解输出可检视 |

### Step 5：bridge-builder + execution-split（1 天）

**TDD 流程**：用含 subagent 的合成数据先写测试 → 实现 → 用真实数据验证

| 编号 | 状态 | 任务 | 验收 |
|-|-|-|-|
| S5-01 | ✅ | 写测试 `tests/bridge-builder.test.ts` (18 tests) | 测试写好 |
| S5-02 | ✅ | 实现 `bridge-builder.ts` | S5-01 通过 |
| S5-03 | ✅ | 写测试 `tests/execution-split.test.ts` (29 tests) | 测试写好 |
| S5-04 | ✅ | 实现 `execution-split.ts` | S5-03 通过 |
| S5-05 | ✅ | 用真实 opencode DB（含 subagent session）验证完整拆解链路 | 真实数据全链路输出可检视 |

### Step 6：cost-calculator + merge + data-service（0.5 天）

| 编号 | 状态 | 任务 | 验收 |
|-|-|-|-|
| S6-01 | ✅ | 写测试 `tests/cost-calculator.test.ts` (12 tests) + 实现 | 测试通过 |
| S6-02 | ✅ | 写测试 `tests/merge.test.ts` (15 tests)（dedup + 增量合并）+ 实现 | 测试通过 |
| S6-03 | ✅ | 实现 `data-service.ts`（协调写入流程）— 此模块用集成测试验证 | 下一步集成测试验证 |

### Step 7：导入 API + CLI — 全链路集成测试（1 天）

**关键里程碑**：此时整个入库 pipeline 应可端到端跑通

| 编号 | 状态 | 任务 | 验收 |
|-|-|-|-|
| S7-01 | ✅ | 写集成测试 `tests/integration/import-opencode.test.ts` (3 tests): listSessions + importSession + DB行验证 | 测试通过 |
| S7-02 | ✅ | 实现 `/api/ingest/import-file/route.ts` + `/api/ingest/import-file/sessions/route.ts` + `src/lib/db.ts` | S7-01 通过 |
| S7-03 | ✅ | 实现 CLI `scripts/import-opencode.ts`：opencode-db 交互式选择 session | 手工 CLI 验证通过 |
| S7-04 | ✅ | dedup 测试：同一 session 重复导入 → 第二次 imported=false, 不产生重复行 | 测试通过 |

### Step 8：读取 API + Session 列表页（1 天）

| 编号 | 状态 | 任务 | 验收 |
|-|-|-|-|
| S8-01 | ✅ | 实现 `/api/observe/data`（Session 列表分页）+ `/api/observe/session`（详情聚合）+ `/api/observe/stats` | API 返回正确数据 |
| S8-02 | ✅ | 实现 `SessionList.tsx` + `MetricCards.tsx` | UI 列表页可查看已导入的 session |

### Step 9：Turn 详情 + 上下文治理 UI（1 天）

| 编号 | 状态 | 任务 | 验收 |
|-|-|-|-|
| S9-01 | ✅ | 实现 `/api/observe/session/turns` + `/api/observe/session/turns/:turnId`（含 inputMessagesJson + contentJson） | API 返回 turn 数据 |
| S9-02 | ✅ | 实现 `TurnTimeline.tsx` + `TurnDetail.tsx` | UI 时间线 + 详情面板可交互 |
| S9-03 | ✅ | 实现 `LlmContextView.tsx`：默认折叠"输入 N 条消息，共 X tokens（占 Y%）"；展开后逐条消息独立可折叠 | 上下文区 UI 可用 |
| S9-04 | ✅ | 实现 `LlmOutputView.tsx`：默认折叠摘要 + token 数；展开后 thinking/text 分区 | 输出区 UI 可用 |
| S9-05 | ✅ | 实现 Token 条形图 + 工具列表 + skill 事件列表 | 详情面板完整 |

### Step 10：Timeline + Subagent + 交互链路（1 天）

| 编号 | 状态 | 任务 | 验收 |
|-|-|-|-|
| S10-01 | ✅ | 实现 `/api/observe/executions` + `/api/observe/session/bridges` | API 返回正确 |
| S10-02 | ✅ | 实现 `TimelineGantt.tsx`（甘特图） | 时间线渲染正确 |
| S10-03 | ✅ | 实现 `SubagentCards.tsx` | Subagent 卡片展示正确 |
| S10-04 | ✅ | 实现 `InteractionGraph.tsx`（主↔子交互链路图） | 链路图渲染正确 |
| S10-05 | ✅ | 实现 `SkillDetail.tsx` | Skill 明细展示正确 |

### Step 11：UI 导入入口 + Session 详情页整合 + 完善（1 天）

| 编号 | 状态 | 任务 | 验收 |
|-|-|-|-|
| S11-01 | ✅ | 实现 `LocalFileImport.tsx`：按钮 + 文件路径输入 + session 选择 + 导入进度 | UI 导入可用 |
| S11-02 | ✅ | Session 详情页 Tab 切换整合（Overview/Turns/Timeline/Subagents/Skills/Interactions） | 详情页完整可交互 |
| S11-03 | ✅ | 过滤/搜索：按 agent/role/tool/skill 过滤 Turn 列表 | 过滤功能可用 |
| S11-04 | ✅ | 边界处理：空 session / 无 subagent / 极长 content / 极长 inputMessages | 不崩溃 |

---

## §3 后续迭代（本轮不实现）

| 编号 | 任务 | 说明 |
|-|-|-|
| F-01 | claude-jsonl adapter | `adapters/claude-jsonl.ts` — Claude Code session JSONL 解析 |
| F-02 | normalize claude 格式 | `normalize.ts` 增加 Claude Code 格式归一化 |
| F-03 | Claude 导入集成测试 | `tests/integration/import-claude.test.ts` |
| F-04 | Claude UI 导入入口 | LocalFileImport 增加 Claude JSONL 上传选项 |

---

## §4 技术选型

| 维度 | 选择 | 原因 |
|-|-|-|
| 前端框架 | Next.js (App Router) | 生态成熟 |
| 数据库 | SQLite (Prisma) | 开发阶段轻量 |
| DB 文件读取 | better-sqlite3 | 读取 opencode sessions.db 需直接打开外部 DB |
| UI 组件 | shadcn/ui + Tailwind CSS | 组件丰富 |
| 测试框架 | vitest | 快速、ESM 支持、与 Next.js 兼容好 |
| 图表 | recharts + dagre | 条形图/折线图 + 甘特图布局 |
| JSON 渲染 | react18-json-view | ToolCall args/result 展示 |
| Markdown | react-markdown + remark-gfm | Turn content 渲染 |

---

## §5 验收标准

| 验收点 | 标准 |
|-|-|
| opencode DB 导入 | `listSessions()` 返回 session 列表含时间/提示词/turn数/模型；选择 session 后入库，DB 行正确 |
| dedup | 同一 session 重复导入不产生重复行 |
| Turn 拆解 | 每个 RawInteraction 正确拆出 Turn + ToolCall + SkillEvent；assistant turn 的 inputMessagesJson/contentSummary/contextWindowPct 正确 |
| Subagent 拆分 | 含 subagent 的 session 正确创建 subagent Execution + Turn 行；InteractionBridge 正确 |
| Session 列表 | 分页、过滤、排序正常 |
| Turn 详情 | 上下文区默认折叠→展开→逐条消息可独立折叠；输出区默认折叠→展开→thinking 可折叠 |
| 交互链路 | 主↔子 dispatch→response 连线正确 |
| 全链路 | 从 opencode DB 文件导入 → UI 查看 session → 下钻 turn → 查看 subagent → 查看交互链路，端到端跑通 |

---

## §6 测试数据目录结构

```
cannbot-insight/
├── tests/
│   ├── data/
│   │   ├── opencode-sessions.db        # 真实 opencode DB（含 3+ session）
│   │   ├── synthetic-opencode.json     # 合成 RawInteraction[]（含 subagent + skill）
│   │   └── expected/                   # 期望输出（用于对比验证）
│   │       ├── turns.json
│   │       ├── bridges.json
│   │       └── executions.json
│   ├── adapters/
│   │   ├── opencode-db.test.ts
│   │   └── index.test.ts
│   ├── normalize.test.ts
│   ├── turn-split.test.ts
│   ├── bridge-builder.test.ts
│   ├── execution-split.test.ts
│   ├── cost-calculator.test.ts
│   ├── merge.test.ts
│   ├── integration/
│   │   └── import-opencode.test.ts
│   └── setup.ts
