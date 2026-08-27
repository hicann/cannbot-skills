# CANNBot-Insight

Session-level observability tool for LLM coding agents. Helps analyze long-context usage patterns, detect model hallucination issues, and govern context window growth across agent sessions.

**[中文文档](README-zh.md)**

## Features

Import SQLite (opencode.db or CANNBot-Insight archive, auto-detected) or JSONL (Claude Code / cpx proxy capture) logs, then analyze Agent sessions turn-by-turn:

- **Tokens & Cost** — 5-item token breakdown per turn with bar chart relative to model context window; cost estimation from token usage and model pricing
- **Context Growth** — Context growth chart per subagent session; animated replay of context window evolution with subagent spawn/death markers; `/compact` markers with context-drop annotations across multiple compactions
- **Context Governance** — View LLM input context composition: visible messages + stable "System (hidden)" overhead; input window correctly truncated at each `/compact` boundary
- **Subagent Tracking** — Identify subagent sessions and dispatch→response bridges; subagent turns surface inline in the Turns timeline with badges, and counts appear in the Overview cards
- **Turn Quick-Jump** — In the Turns timeline, type a turn number (e.g. `#459`) and press Enter to jump straight to that turn and scroll it into view; subagent turns auto-expand their ancestor dispatch lanes
- **Wire Rounds (proxy captures)** — Rebuild each wire round directly from the cpx capture file: full input (accumulated messages with new-of-this-round highlighted) plus output (raw response), to cross-check the DB-reconstructed LLM Input; tab hidden by default, shown with `./start.sh -a`
- **Key Redaction (proxy captures)** — API keys never touch disk: sensitive fields and vendor key shapes (`sk-ant-` / `AIza` / `gsk_` / `LTAI` / `Bearer` / env echoes / URL query params) are masked (`abcd…wxyz`) at the single write chokepoint before capture files are written; the cpx launch log is masked with the same rules
- **Round-Pair Turns (proxy captures)** — Proxy sessions split Turns into per-round input/output pairs (badged in the timeline): input turns list this round's new verbatim wire messages one by one; output turns store the accumulated request at import and serve it verbatim, preserving exact wire order (including tool_result vs system-injection interleaving) with no reconstruction
- **Skill Events** — Track skill load/invoke/use events per turn
- **Skill Content (SKILL.md 全文)** — Per-skill button in Skill Summary and Skills per Agent reconstructs the skill's full SKILL.md content from the native Skill tool injection or Read-of-SKILL.md results; supports download
- **Concept Tracing** — Search keywords across turns, view propagation chain and DAG graph; refine results by source type, thinking, or tool; clicking a result's "View Turn" jumps to the Turns tab and auto-expands + highlights the keyword at its match location (content / tool args / result / error)
- **File Read Analysis** — Detect duplicate and unnecessary file reads
- **File Content Restore (循迹复卷)** — Per-file button reconstructs the full file content from collected read/write tool calls by timestamp (last-write-wins per line); uncovered lines marked `--line N not found --`; supports download
- **Gather & Rebuild Directory (重编汇册)** — One-click rebuild all read/written files in the session's working directory, preserving the path tree; download as a zip (pure-JS STORE, no deps)
- **Session Compare** — Compare two sessions on tokens, cost, latency, tool calls, and subagents
- **Audit** — Paste or import the workflow-analysis JSON (produced by the Audit prompt + Claude Code) to render the actual flow chart with per-node problems, the G/S 8-dimension skill-quality board, and optimization priorities. `§N` turn refs in evidence are clickable to jump to that turn; parallel branches (same `parallel` id) render in one row; the 3 slowest nodes are highlighted in red; supports importing JSON from a file and exporting the rendered analysis back to a file. Generation state survives tab switches (cross-tab resume). **v4**: agent-centric 3-dimension audit (completion / efficiency / quality) — deterministically extracts each agent's input/output/envelope first, then LLM judges completion & quality while efficiency is computed from the envelope; renders an agent tree. The agent tree hides fully-passing subagents by default (toggle to reveal them); cross-agent issues and optimization priorities each cite evidence (`agent:<name> #turn`) with clickable turn jumps, and the audit header shows the audit's total token consumption. The Audit tab is split into two sub-tabs: **Workflow audit** (the v1-v4 analysis above) and **Skill audit** — run skill-eval's `audit` against the current session to reconcile real execution vs each invoked skill's SKILL.md (no re-run), vs each dispatched agent's `.md` (scanned from local `AGENTS_SCAN_ROOT`, multi-plugin disambiguated by session coverage), and vs the main agent's workflow-level SKILL.md (`--kind root` — audits the top-level main agent's orchestration; the workflow-skill body is recovered from the session's first user turn, i.e. the injected system prompt, since the main agent typically only dispatches subagents and never invokes a skill itself; surfaced only when that first user turn is non-trivial, ≥500 chars), with per-target summary + a native report view (no iframe): five-state verdict badges, filterable findings (by verdict / category / method / instruction-or-evidence text), multi-transcript by-instruction tables, collapsible per-instruction grouping with FAIL-first sort, a verdict/method legend, and a "open raw skill-eval HTML" escape hatch. A real-time progress bar with percentage streams during the run (parsed from skill-eval's on_progress output); results persist across tab switches; requires the `skill-eval` + `claude` CLIs; `AGENTS_SCAN_ROOT` env (defaults to auto-detected skills-dev repo root) for agent audit.

## Option 0: tgz install (Recommended)

No source clone, no manual env setup, no npm registry config. Install from a distributed `.tgz` package file.

**Install** (after obtaining `cannbot-insight-1.83.0.tgz` from the maintainer / GitCode Release):

```bash
npm install -g ./cannbot-insight-1.83.0.tgz
cannbot-insight            # first run: auto-build + migrate (~30s-2min), then start + open browser
```

Runtime deps (next/prisma/better-sqlite3/...) auto-fetch from the public npmjs registry during install; only the package body is private in the `.tgz`. First install compiles `better-sqlite3` native (~1-2 min, needs `python3`/`make`/`g++` or a prebuild hit). Requires Node.js >= 20.

All writable state lives in `~/.cannbot-insight/` (SQLite DB + `.next/` build cache); the installed package dir stays read-only. Port 21025 by default (auto-finds a free port if occupied).

| Flag | Effect |
|------|--------|
| `-a`, `--advanced` | Show advanced tabs (wireRounds/replay) |
| `-k`, `--kill` | Kill any process on port 21025, reuse it |
| `-f`, `--fresh` | Clear `.next` build cache and rebuild |

CLI subcommands pass straight through (same as Option 2, but no `cd`/`npx tsx` prefix):

```bash
cannbot-insight upload --file ~/.local/share/opencode/opencode.db --list
cannbot-insight sessions
cannbot-insight --help
```

**Requires Node.js >= 20.** Python 3 is optional (auto-detected): if present, the smart-agent backend (breather/v2 analysis) starts on port 21026; if absent, it gracefully degrades and the rest works normally.

## Option 1: Web UI

**Requires Node.js >= 20.x** (v18.19.x fails to install better-sqlite3 / Prisma 6). If you have nvm, `start.sh` auto-switches to Node 20 LTS.

Log file locations:
- opencode: `~/.local/share/opencode/opencode.db`
- Claude Code: `~/.claude/projects/<hash>/sessions/<id>.jsonl`, or point to a directory to scan all .jsonl files

Linux (for Windows systems, use start.bat instead):

```bash
./start.sh          # Auto install + migrate + start Web UI on port 21025
./start.sh -u       # Update dependencies + migrate + start Web UI
./start.sh -f       # Fresh build (clear .next cache, rebuild from scratch)
```

Open `http://localhost:21025`. After importing a log file, click a session to explore 9 analysis tabs.

Web UI also supports: exporting sessions to standalone SQLite or hierarchical Markdown; uploading sessions to CANNBay v2 (an atomgit dataset repo) with a description dialog. Uploads use the proxy-capture claude-jsonl folder format (main session + subagents); opencode sessions are exported to jsonl from the DB automatically. Every upload passes mandatory data governance (secret redaction + residue circuit-breaker — no keys leave the public repo). CANNBay listing/download uses a partial-clone (`--filter=blob:none`) persistent mirror: listing reads metadata only (instant), importing one session fetches just its blobs on demand — fast even with thousands of sessions. The legacy .db-snapshot upload/parse (gitcode CANNBay) remains available at the API level for reading historical data, hidden from the UI. All upload/capture/export jsonl now also carries a declarative `x_cannbay` extension namespace (envelope frozen, `{schema, version, data}` pockets — spec: docs/cannbay-schema-spec.md) with 6 export losses fixed (reasoningTokens/ttftMs/ToolCall details/framework/model params/wire status); a column-exhaustive round-trip IT proves DB → export → re-import loses nothing.

## Option 2: CLI Upload + Web Analysis

Designed for SSH remote servers, Web IDEs, and other environments without a browser. CLI imports and uploads in one step, then analyze in Web UI.

Log file locations:
- opencode: `~/.local/share/opencode/opencode.db`
- Claude Code: `~/.claude/projects/<hash>/sessions/<id>.jsonl`, or point to a directory to scan all .jsonl files

### One-time Setup (first run in a new environment)

```bash
cd cannbot-insight
npm install
echo 'DATABASE_URL="file:./dev.db"' > .env
npx prisma generate
npx prisma migrate deploy
```

Takes ~1 minute. No need to repeat unless dependencies change. The upload command auto-starts the backend — no need to run `start.sh` manually.

### Upload a Session

**opencode:**

```bash
# List available sessions
npx tsx src/cli/index.ts upload --file ~/.local/share/opencode/opencode.db --list

# Upload a specific session (non-interactive, script-friendly)
npx tsx src/cli/index.ts upload \
  --file ~/.local/share/opencode/opencode.db \
  --session-id <session-id> \
  --description "description" \
  --yes --json

# Interactive session picker + description prompt
npx tsx src/cli/index.ts upload --file ~/.local/share/opencode/opencode.db
```

**Claude Code:**

```bash
# List available sessions (point to projects dir, auto-scans .jsonl recursively)
npx tsx src/cli/index.ts upload --file ~/.claude/projects/ --list

# Upload a specific session (session-id = .jsonl filename without extension)
npx tsx src/cli/index.ts upload \
  --file ~/.claude/projects/ \
  --session-id <uuid> \
  --description "description" \
  --yes --json

# Or point directly to a single .jsonl file
npx tsx src/cli/index.ts upload --file ~/.claude/projects/<hash>/<uuid>.jsonl --list
```

| Option | Description |
|--------|-------------|
| `--file <path>` | Source path: opencode `.db` file / Claude Code `.jsonl` file or directory; source type auto-detected |
| `--session-id <id>` | Upload a specific session (skip interactive picker) |
| `--description <text>` | Upload description (skip interactive prompt) |
| `--yes` | Skip confirmation prompt |
| `--json` | JSON output (non-interactive, script-friendly) |
| `--list` | List available sessions without uploading |
| `--framework <name>` | Framework type (auto-inferred: opencode-db → opencode, claude-jsonl → claude-code) |

After upload, view analysis in Web UI: click the **CANNBay** button during import to select and import session folders directly from the repository (blobs fetched on demand) — no manual clone needed.

## Option 3: Zero-dependency Session Export

Extract a single session (with subagents) from opencode.db or Claude Code JSONL into a standalone file. **Zero dependencies** — requires only Node.js >= 22 (built-in `node:sqlite`); no `npm install`, no Prisma, no backend. Auto-detects framework from `--file`: `.db` → opencode, `.jsonl`/directory → Claude Code.

```bash
# List available sessions (auto-detect source)
node export-db.mjs --list
node export-db.mjs -f ~/.claude/projects/ --list

# Interactive session picker
node export-db.mjs
node export-db.mjs -f ~/.claude/projects/

# Export a specific session
node export-db.mjs -s ses_xxx                          # opencode
node export-db.mjs -f ~/.claude/projects/ -s <uuid>     # Claude Code

# Export to a custom path
node export-db.mjs -s ses_xxx -o ~/my-session.db
node export-db.mjs -f ~/.claude/projects/ -s <uuid> -o ~/my-session.jsonl

# JSON output (script-friendly)
node export-db.mjs -s ses_xxx --json
```

When `-o` is omitted, output defaults to the script directory:
- opencode → `dbfile/session_<id>.db`
- Claude Code → `jsonlfile/<id>.jsonl`

The exported file can be re-imported into Insight or uploaded to CANNBay via Option 2's `upload --file` command:

```bash
npx tsx src/cli/index.ts upload --file /abs/path/to/session_xxx.db  --session-id ses_xxx --yes --json
npx tsx src/cli/index.ts upload --file /abs/path/to/<uuid>.jsonl    --session-id <uuid>  --yes --json
```

| Option | Description |
|--------|-------------|
| `-f, --file <path>` | Source path (auto-detected: opencode.db or ~/.claude/projects/) |
| `-s, --session-id <id>` | Export a specific session (skip interactive picker) |
| `-o, --output <path>` | Output file path (default: `dbfile/` or `jsonlfile/`) |
| `-l, --list` | List available sessions without exporting |
| `-j, --json` | JSON output |
| `-h, --help` | Show help |
