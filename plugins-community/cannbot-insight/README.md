# CANNBot-Insight

Session-level observability tool for LLM coding agents. Helps analyze long-context usage patterns, detect model hallucination issues, and govern context window growth across agent sessions.

**[中文文档](README-zh.md)**

## Features

Import opencode sessions.db or Claude Code JSONL logs, then analyze Agent sessions turn-by-turn:

- **Tokens & Cost** — 5-item token breakdown per turn with bar chart relative to model context window; cost estimation from token usage and model pricing
- **Context Growth** — Context growth chart per subagent session; animated replay of context window evolution with subagent spawn/death markers; `/compact` markers with context-drop annotations across multiple compactions
- **Context Governance** — View LLM input context composition: visible messages + stable "System (hidden)" overhead; input window correctly truncated at each `/compact` boundary
- **Subagent Tracking** — Identify subagent sessions, build dispatch→response bridges; SVG diagram showing main↔subagent connections
- **Skill Events** — Track skill load/invoke/use events per turn
- **Workflow** — Divide a session into phases by the workflow skill's own progress markers (阶段一/二/三/四 in non-thinking assistant output; future-tense mentions like "后续可继续执行阶段二" are ignored); skills attach to whichever phase their dispatch root turn falls in, so a session that only completed 阶段一 shows as a single 阶段一 — not every session runs the full workflow. Falls back to skill-family gap heuristics when no markers exist
- **Concept Tracing** — Search keywords across turns, view propagation chain and DAG graph
- **File Read Analysis** — Detect duplicate and unnecessary file reads
- **Session Compare** — Compare two sessions on tokens, cost, latency, tool calls, and subagents

## Option 1: Web UI

**Requires Node.js >= 20.x** (v18.19.x fails to install better-sqlite3 / Prisma 6). If you have nvm, `start.sh` auto-switches to Node 20 LTS.

Log file locations:
- opencode: `~/.local/share/opencode/sessions.db`
- Claude Code: `~/.claude/projects/<hash>/sessions/<id>.jsonl`, or point to a directory to scan all .jsonl files

```bash
./start.sh          # Auto install + migrate + start Web UI on port 21025
./start.sh -u       # Update dependencies + migrate + start Web UI
./start.sh -f       # Fresh build (clear .next cache, rebuild from scratch)
```

Open `http://localhost:21025`. After importing a log file, click a session to explore 9 analysis tabs.

Web UI also supports: exporting sessions to standalone SQLite or hierarchical Markdown; uploading sessions to CANNBay with a description dialog.

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
