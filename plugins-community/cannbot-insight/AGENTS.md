# AGENTS.md — cannbot-insight

## Project

CANNBot-Insight: Session-level observability tool for LLM coding agents (opencode).
Next.js 16 App Router + Prisma + SQLite. Chinese/English mixed UI.
UI version tracked in `src/lib/version.ts` (v0.31) — `package.json` version (0.02) is not the display version.

## CLI Frontend (New - 2026-06)

Adding a CLI frontend alongside the existing web UI. Backend remains unchanged; CLI is a pure API client.

### Tech Stack
- **Ink 7.0.6** + React 19.2.7 (ESM required: `"type": "module"` in package.json)
- **Commander.js** for command parsing
- **string-width** + **cli-truncate** for CJK character width handling
- Run with **tsx** (not ts-node) due to yoga-layout top-level await

### Key Decisions (PoC Verified)
- **No third-party Ink components** — ink-table/ink-select/ink-spinner all incompatible with Ink v7
- **Self-implement all components**: DataTable, Spinner (10 lines), TextInput (30 lines)
- **Dual render strategy**: ink-testing-library for tests (has lastFrame), Ink native render for production
- **stdin handling**: `resume() + setRawMode(true)` before Ink render, restore on exit
- **Pagination over virtual scroll**: pageSize=20, n/p keys (Ink re-renders full tree anyway)
- **CLI types use `Api` prefix** (ApiSessionListItem) to distinguish from shared types

### Architecture
```
src/cli/
├── index.ts              # Commander entry point
├── client.ts             # API client (15 endpoints)
├── types.ts              # Api-prefixed response types
├── errors.ts             # Error hierarchy
├── config.ts             # Config management
├── commands/             # CLI commands (sessions, session, turn, search, compare, stats, import, delete, config)
├── hooks/                # useApi, useKeyboard, useNavigation, useTable
├── utils/                # format, colors, table (with padEndVisual/truncateVisual)
└── tui/
    ├── App.tsx           # Root component (stdin switching, useMemo client)
    ├── screens/          # SessionList, SessionDetail, TurnDetail, CompareView, ImportPanel
    ├── tabs/             # OverviewTab, TurnsTab, WorkflowTab, SubagentsTab, SkillsTab, BridgesTab, ContextTab
    └── components/       # StatusBar, KeyBar, DataTable, MetricCards, AsciiBar, TreeView, TabBar, Spinner, TextInput
```

### Dev Commands (CLI)
| Command | Notes |
|---------|-------|
| `npx tsx src/cli/index.ts tui` | Launch TUI mode (requires real TTY) |
| `npx tsx src/cli/index.ts sessions` | List sessions (command mode) |
| `npx tsx src/cli/index.ts stats` | Global statistics |
| `npm run test:cli` | CLI tests (uses ink-testing-library) |

### CLI Constraints
- **Ink v7 native render() has no lastFrame/frames/output** — only ink-testing-library provides these
- **ESM strict**: Cannot use require(), must use import
- **CJ width**: Chinese characters = 2 columns, use string-width for all text measurement
- **No PTY in tests**: TUI E2E tests only run manually, CI runs component tests only
- **Version**: Import from `@/lib/version`, never hardcode

## Dev Commands

| Command | Notes |
|---------|-------|
| `./start.sh` | Auto port (21025+), kills existing dev server via `.next/dev/lock` PID, runs `prisma migrate dev` + `next dev` |
| `./start.sh -u` | Same but also runs `npm install` first |
| `npm run dev` | Runs on default port 3000 — prefer `./start.sh` instead |
| `npm run build` | Standard `next build` |
| `npm run test` | `vitest run` — 187 tests (web) |
| `npm run test:cli` | CLI frontend tests |
| `npm run lint` | `eslint` (flat config, `eslint.config.mjs`) — run after edits to verify |

## Architecture

### Data Flow

```
opencode sessions.db (external) → better-sqlite3 read → opencode-db adapter → normalize → turn-split → bridge-builder → execution-split → merge → data-service → Prisma write (SQLite)
```

### Page Structure

- **Home** (`page.tsx`): Server component, direct Prisma queries (no API layer)
- **Session detail** (`session/[taskId]/page.tsx`): `"use client"` component, fetches from `/api/observe/*` routes via `useEffect`
- **9 tabs**: Overview → Turns → Workflow ✦ → Trace 🔍 → Subagents → Skills → Interactions → AI Workflow (beta) → Context 📊

### Key Directories

- `src/app/api/ingest/` — Import/delete routes
- `src/app/api/observe/` — Read routes (session, turns, turns/search, bridges, workflow, stats, data)
- `src/app/api/ai/` — AI analysis routes (analyze-workflow, test-provider)
- `src/lib/ingest/` — Ingest pipeline (opencode-db adapter, turn-split, bridge-builder, execution-split, phase-split)
- `src/lib/ingest/turn-split.ts` — `MODEL_CONTEXT_WINDOWS` mapping (10 models), `DEFAULT_CONTEXT_WINDOW = 128000`. Frontend ContextTracker.tsx duplicates this mapping — keep them in sync.
- `src/lib/ai/analyzer.ts` — AI phase divider (LLM call → WorkflowTree structure)
- `src/components/observe/` — 16 tab view components (TurnTimeline, WorkflowTreeView, SubagentCards, TraceView, ContextTracker, TokenTrendChart, TokenBarChart, TimelineGantt, SkillEventList, LlmContextView, LlmOutputView, ToolCallList, etc.)
- `src/components/ui/dialog.tsx` — Built on `@base-ui/react` (not radix-ui). Use this for custom confirm dialogs instead of browser `confirm()`.

## Constraints & Gotchas

- **Zero Prisma schema changes** for feature work — computed data (workflow, context charts) built at API/render time, not persisted
- **AI provider config stored in localStorage only** — lost on browser change, never sent to server
- **AI only supports OpenAI-compatible `/chat/completions`** — `/apps/anthropic` path rejected with red warning
- **`createdAt_ts` is nullable DateTime** — need `.toISOString()` fallback to `createdAt`
- **Hydration error** — inner expand buttons in TurnTimeline must be `<span role="button">` not `<button>` to avoid nesting
- **Next.js dev lock**: `.next/dev/lock` contains JSON with PID/port — used by start.sh to kill existing server
- **`response_format: { type: "json_object" }`** required for AI analysis — no fallback for non-supporting models
- **AI input optimization**: Only sends root assistant/user/system turns, 30K chars budget with auto-truncation, summaries 80 chars max
- **Version bumps**: +0.01 per feature commit, update `src/lib/version.ts` only
- **Turn model has no `cost` field** — set `cost: 0` in API mapping
- **Context tracker groups by `subagentSessionId`** (27 independent sessions) not `agentName` (only 4), because same agentName has multiple independent executions

## 测试要求

1. **集成优先** — 从页面或 API 层面验证，不写简单函数级单元测试
2. **数据驱动** — 通过 JSON 或 DB 原始数据驱动测试，数据放在 `tests/data/`
3. **Pipeline 覆盖** — 覆盖完整数据流管道（如 JSONL → adapter → turn-split → aggregates）
4. **功能修复必须写 IT 测试** — 每个 bug 修复或功能改进必须配套集成测试，验证修复后的完整数据流表现
5. **Fixture 管理** — 为每种场景创建独立 fixture 文件

## 提交流程

1. **同步远端** — `git pull`
2. **解决冲突** — 确保代码不丢失
3. **运行测试** — `npm run test` 全部通过
4. **更新文档** — README.md / README-zh.md（如有功能变更）
5. **更新版本号** — `src/lib/version.ts` 仅新增特性或严重 bug 时 +0.01
6. **提交推送** — `git commit -m "[type] v0.xx: 描述"` → `[feat]` `[fix]` `[test]` `[docs]` `[chore]`

## Code Conventions

- **Comments**: Don't write them unless asked
- **Doc files**: Don't create `.md` / README unless explicitly requested
- **README sync**: When features are added, removed, or changed, update both `README.md` and `README-zh.md` feature lists and tab descriptions to match reality
- **Badge variants**: default, secondary, destructive, outline, blue, green, orange, purple, gray, red, yellow (from badge.tsx cva)
- **Path alias**: Always `@/lib/...` / `@/components/...`, never relative `../../`
- **Git**: Hosted on gitcode.com — never use `gh` CLI
- **UI primitives**: All from `@base-ui/react` (not radix-ui) via shadcn v4 + Tailwind v4
- **Charts**: Pure SVG, no chart libraries (recharts etc.) — matches TraceView DAG style
- **Search history**: sessionStorage only, never persisted
