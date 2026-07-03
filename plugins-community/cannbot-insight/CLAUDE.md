# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CANNBot-Insight: Session-level observability tool for LLM coding agents (opencode).
Next.js 16 App Router + Prisma 6 + SQLite. Chinese/English mixed UI.
UI version tracked in `src/lib/version.ts` — `package.json` version is NOT the display version.

## Commands

| Command | Purpose |
|---------|---------|
| `./start.sh` | Auto port (21025+), kills existing server via `.next/dev/lock` PID, runs `prisma migrate dev` + `next dev` |
| `./start.sh -u` | Same but also `npm install` first |
| `./start.sh -c tui` | Backend + CLI TUI mode (backend stops when CLI exits) |
| `npm run test` | `vitest run` — all tests |
| `npm run test:watch` | `vitest` — watch mode |
| `npm run test:cli` | `vitest run tests/cli/` — CLI tests only |
| `npx vitest run tests/cli/unit/commands/import.test.ts` | Run a single test file |
| `npm run lint` | eslint (flat config, `eslint.config.mjs`) |
| `npm run build` | `next build` |

## Architecture

### Data Flow

```
opencode sessions.db (external)
  → better-sqlite3 read
  → opencode-db adapter (src/lib/ingest/adapters/)
  → normalize → turn-split → bridge-builder → execution-split
  → data-service (orchestrator)
  → Prisma write (8 models, SQLite)
```

Adapter registry in `src/lib/ingest/adapters/index.ts` — currently `opencode-db` and `claude-jsonl`.

### Frontend Modes (shared backend)

All three modes are pure API clients of the same 15 `/api/observe/*` endpoints:

| Mode | Entry | Rendering |
|------|-------|-----------|
| **Web UI** | `src/app/page.tsx` (server component, direct Prisma) | Next.js + shadcn/ui v4 + Tailwind v4 + @base-ui/react |
| **TUI** | `src/cli/tui/App.tsx` | Ink 7 + React 19 (ESM strict) |
| **CLI** | `src/cli/index.ts` (Commander) | chalk + string-width + cli-truncate |

Session detail page (`session/[taskId]/`) is `"use client"` with 9 tabs fetching from `/api/observe/*`.

### Key Code Paths

- **Ingest write**: `src/lib/ingest/data-service.ts` → `importSession()` — currently uses per-row `create()` (P0 optimization: change to `createMany` + `$transaction`, see `docs/import-batch-write-optimization.md`)
- **Ingest read**: `src/lib/ingest/adapters/opencode-db.ts` — N+1 query pattern (per-session/per-message sub-queries to `part` table)
- **Context windows**: `src/lib/context-window-config.ts` — configurable via `context-windows.json`, hardcoded defaults as fallback. Web `ContextTracker.tsx` has its own model mapping — keep in sync.
- **AI analysis**: `src/lib/ai/analyzer.ts` — only OpenAI-compatible `/chat/completions`, config in localStorage only
- **CLI client**: `src/cli/client.ts` — InsightClient wrapping 15 API endpoints
- **CLI types**: `src/cli/types.ts` — `Api`-prefixed interfaces to distinguish from `src/lib/shared/types.ts`

## Constraints & Gotchas

- **ESM strict** (`"type": "module"` in package.json) — no `require()`, use `import`
- **Ink v7 has no lastFrame/frames/output** on native `render()` — use `ink-testing-library` for tests only
- **No third-party Ink components** — ink-table/ink-select/ink-spinner incompatible with Ink v7; self-implement all components
- **CJK width**: Chinese chars = 2 columns; use `string-width` for all text measurement in CLI
- **Zero Prisma schema changes** for feature work — computed data built at API/render time, not persisted
- **Turn model has no `cost` field** — set `cost: 0` in API mapping, strip from `TurnRow` before Prisma write
- **`createdAt_ts` is nullable DateTime** — fallback to `createdAt` with `.toISOString()`
- **Hydration error**: inner expand buttons must be `<span role="button">` not `<button>` to avoid nesting
- **`response_format: { type: "json_object" }`** required for AI analysis — no fallback
- **AI input optimization**: Only root assistant/user/system turns, 30K chars budget, summaries 80 chars max
- **Git hosted on gitcode.com** — never use `gh` CLI
- **Path alias**: Always `@/lib/...` / `@/components/...`, never relative `../../`

## Testing

- **Integration over unit**: 写测试用例时，不要简单的函数级别用例，需要从页面或 API 层面进行验证
- **Data-driven**: 通过 JSON 或 DB 原始数据驱动测试，数据准备在 `tests/data/` 目录下
- **Pipeline coverage**: 测试应覆盖完整数据流（如 JSONL → adapter → turn-split → SkillEvent → aggregates）
- **功能修复必须写 IT 测试**: 每个 bug 修复或功能改进必须配套集成测试，验证修复后的完整数据流表现

## Commit Workflow

每次提交代码前必须完成以下步骤（顺序执行）：

1. **同步远端**: `git pull` — 远端可能有更新
2. **解决冲突**: 如有冲突，手动解决后继续
3. **运行测试**: `npm run test` — 全部通过才能继续
4. **更新文档**: README.md / README-zh.md（如有功能变更）
5. **更新版本号**: `src/lib/version.ts` — 仅在新增特性或修复严重 bug 时 +0.01，小修小补不升版本
6. **提交推送**: `git commit -m "[type] v0.xx: 描述"` → `[feat]` `[fix]` `[test]` `[docs]` `[chore]`

## Code Conventions

- **Comments**: Don't write them unless asked
- **Doc files**: Don't create `.md` / README unless explicitly requested
- **README sync**: When features change, update both `README.md` and `README-zh.md` feature lists and tab descriptions
- **Version bumps**: +0.01 only when adding features or fixing serious bugs, update `src/lib/version.ts` only
- **Badge variants**: default, secondary, destructive, outline, blue, green, orange, purple, gray, red, yellow
- **UI primitives**: All from `@base-ui/react` (not radix-ui) via shadcn v4 + Tailwind v4
- **Charts**: Pure SVG, no chart libraries — match TraceView DAG style
- **Dialogs**: Use `src/components/ui/dialog.tsx` (built on @base-ui/react), never browser `confirm()`
- **Search history**: sessionStorage only, never persisted
