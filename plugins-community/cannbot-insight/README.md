# CANNBot-Insight

Session-level observability tool for LLM coding agents. Helps analyze long-context usage patterns, detect model hallucination issues, and govern context window growth across agent sessions.

**[中文文档](README-zh.md)**

## Features

Import opencode sessions.db or Claude Code JSONL logs, then analyze Agent sessions turn-by-turn:

- **Tokens & Cost** — 5-item token breakdown per turn with bar chart relative to model context window; cost estimation from token usage and model pricing
- **Context Growth** — Context growth chart per subagent session; animated replay of context window evolution with subagent spawn/death markers; `/compact` markers with context-drop annotations across multiple compactions
- **Context Governance** — View LLM input context composition: visible messages + stable "System (hidden)" overhead; input window correctly truncated at each `/compact` boundary
- **Subagent Tracking** — Identify subagent sessions and dispatch→response bridges; subagent turns surface inline in the Turns timeline with badges, and counts appear in the Overview cards
- **Skill Events** — Track skill load/invoke/use events per turn
- **Skill Content (SKILL.md 全文)** — Per-skill button in Skill Summary and Skills per Agent reconstructs the skill's full SKILL.md content from the native Skill tool injection or Read-of-SKILL.md results; supports download
- **Concept Tracing** — Search keywords across turns, view propagation chain and DAG graph; refine results by source type, thinking, or tool
- **File Read Analysis** — Detect duplicate and unnecessary file reads
- **File Content Restore (循迹复卷)** — Per-file button reconstructs the full file content from collected read/write tool calls by timestamp (last-write-wins per line); uncovered lines marked `--line N not found --`; supports download
- **Gather & Rebuild Directory (重编汇册)** — One-click rebuild all read/written files in the session's working directory, preserving the path tree; download as a zip (pure-JS STORE, no deps)
- **Session Compare** — Compare two sessions on tokens, cost, latency, tool calls, and subagents
- **Audit** — Paste or import the workflow-analysis JSON (produced by the Audit prompt + Claude Code) to render the actual flow chart with per-node problems, the G/S 8-dimension skill-quality board, and optimization priorities. `§N` turn refs in evidence are clickable to jump to that turn; parallel branches (same `parallel` id) render in one row; the 3 slowest nodes are highlighted in red; supports importing JSON from a file and exporting the rendered analysis back to a file. Generation state survives tab switches (cross-tab resume). **v4**: agent-centric 3-dimension audit (completion / efficiency / quality) — deterministically extracts each agent's input/output/envelope first, then LLM judges completion & quality while efficiency is computed from the envelope; renders an agent tree. The Audit tab is split into two sub-tabs: **Workflow audit** (the v1-v4 analysis above) and **Skill audit** — run skill-eval's `audit` against the current session to reconcile real execution vs each invoked skill's SKILL.md (no re-run), vs each dispatched agent's `.md` (scanned from local `AGENTS_SCAN_ROOT`, multi-plugin disambiguated by session coverage), and vs the main agent's workflow-level SKILL.md (`--kind root` — audits the top-level main agent's orchestration; the workflow-skill body is recovered from the session's first user turn, i.e. the injected system prompt, since the main agent typically only dispatches subagents and never invokes a skill itself; surfaced only when that first user turn is non-trivial, ≥500 chars), with per-target summary + a native report view (no iframe): five-state verdict badges, filterable findings (by verdict / category / method / instruction-or-evidence text), multi-transcript by-instruction tables, collapsible per-instruction grouping with FAIL-first sort, a verdict/method legend, and a "open raw skill-eval HTML" escape hatch. A real-time progress bar with percentage streams during the run (parsed from skill-eval's on_progress output); results persist across tab switches; requires the `skill-eval` + `claude` CLIs; `AGENTS_SCAN_ROOT` env (defaults to auto-detected skills-dev repo root) for agent audit.

## Option 1: Web UI

**Requires Node.js >= 20.x** (v18.19.x fails to install better-sqlite3 / Prisma 6). If you have nvm, `start.sh` auto-switches to Node 20 LTS.

Log file locations:
- opencode: `~/.local/share/opencode/sessions.db`
- Claude Code: `~/.claude/projects/<hash>/sessions/<id>.jsonl`, or point to a directory to scan all .jsonl files

Linux (for Windows systems, use start.bat instead):

```bash
./start.sh          # Auto install + migrate + start Web UI on port 21025
./start.sh -u       # Update dependencies + migrate + start Web UI
./start.sh -f       # Fresh build (clear .next cache, rebuild from scratch)
```

Open `http://localhost:21025`. After importing a log file, click a session to explore 9 analysis tabs.

Web UI also supports: exporting sessions to standalone SQLite or hierarchical Markdown; uploading sessions to CANNBay with a description dialog. CANNBay master is capped at the 20 newest sessions (older ones auto-rotate to a date-bucketed `archive-YYYY-MM` branch); uploads use a persistent local mirror so only the delta is fetched each time.

## Option 2: CLI Upload + Web Analysis

Designed for SSH remote servers, Web IDEs, and other environments without a browser. CLI imports and uploads in one step, then analyze in Web UI.

Log file locations:
- opencode: `~/.local/share/opencode/sessions.db`
- Claude Code: `~/.claude/projects/<hash>/sessions/<id>.jsonl`, or point to a directory to scan all .jsonl files

```bash
# Upload from source file (source type auto-detected from file extension)
npx tsx src/cli/index.ts upload --file ./sessions.db           # Interactive picker if multiple sessions
npx tsx src/cli/index.ts upload --file ./logs/                 # Claude JSONL (directory)
```

Upload triggers an interactive description prompt. Backend auto-starts if not running and stops after upload completes.

After upload, view analysis in Web UI: click the **CANNBay** button during import to select and import DB files directly from the repository — no manual download needed.
