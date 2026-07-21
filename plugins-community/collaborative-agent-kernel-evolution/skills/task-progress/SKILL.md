---
name: task-progress
description: >
  Track and manage task progress via a PROGRESS.md. MUST use when:
  (1) Starting a new task or operator workflow,
  (2) Entering or completing any step/stage/substage,
  (3) After context compaction or reset — read PROGRESS.md first to recover state.
  Also call when the user asks about current progress or status.
allowed-tools: Read, Write, Edit, Bash, TaskCreate, TaskGet, TaskUpdate, TaskList
---

# Task Progress Tracking

Maintain a `PROGRESS.md` file in the task output directory to track progress across stages.
This file survives context compaction and serves as both external observability and self-recovery.

## File Location

`PROGRESS.md` lives in the task's output directory (e.g., `output/<OpName>/PROGRESS.md`).
If no output directory exists yet, create it at the project root as `PROGRESS.md`.

**Convention**: All bash commands below use `$P` as shorthand for the PROGRESS.md path.
On Task Start, set this variable:
```bash
P=output/<OpName>/PROGRESS.md
```

## File Structure

PROGRESS.md has two sections separated by a ruler (`---`):

### Section 1: TODO List (lines 1–40, always read)

A hierarchical checklist — the "dashboard". Always kept current.

```markdown
# <OpName> Progress

**Status**: 🔄 In Progress | ✅ Complete | ❌ Failed
**Started**: 2026-03-16 08:30 UTC
**Last updated**: 2026-03-16 09:15 UTC
**Current stage**: 2. Kernel development

## TODO
- [x] 0. Environment detection
- [x] 1. Op description generation
- [ ] 2. Kernel development
- [ ] 3. Evaluation
- [ ] 4. Export and cleanup
```

### Section 2: Detailed Log (line 41+, append-only)

Timestamped entries for each action. **Only read when explicitly needed.**

```markdown
---

## Detailed Log

### [08:30] 0. Environment detection
CANN detected locally. Device 2.
```

## Claude Code Task Management

Create/manage Claude Tasks as the TODO list. Later tasks should be blocked by previous tasks.

---

## Rules

> **CRITICAL**: Every rule below contains **exact bash commands** that you MUST execute
> via the Bash tool. Do NOT skip them. Do NOT just print the confirmation text without
> running the commands. The confirmation line is ONLY valid if the bash commands ran first.

### Before Any Stage Work — MUST execute first

If you are starting a new session, resuming after a pause, or are unsure what stage you are in:

```bash
head -40 "$P"
```

Then state: `"Current stage from PROGRESS.md: [X]"` — this proves the file was read.
Resume from the first incomplete stage; **do NOT restart completed stages**.

### On Task Start

1. Create `PROGRESS.md` using the Write tool with the full TODO list
2. Set the path variable for later use:
   ```bash
   P=output/<OpName>/PROGRESS.md
   ```
3. Begin work

### On Stage Enter — MUST execute via Bash before any stage work

Run these **exact bash commands** (substitute `{STAGE}` with e.g. `5. Ascend call generation`):

```bash
# 1. Update "Current stage" line in-place
sed -i "s/^\*\*Current stage\*\*:.*/**Current stage**: {STAGE}/" "$P"

# 2. Update "Last updated" timestamp
sed -i "s/^\*\*Last updated\*\*:.*/**Last updated**: $(date -u '+%Y-%m-%d %H:%M') UTC/" "$P"

# 3. Append log entry (no read needed — just append)
echo -e "\n### [$(date -u '+%H:%M')] {STAGE}\nStarting..." >> "$P"

# 4. Print dashboard
head -25 "$P"
```

Then output:
```
PROGRESS: stage [X] entered, log entry appended
```

### On Stage Complete — MUST execute via Bash before moving to next stage

Run these **exact bash commands** (substitute `{N}` with stage number, `{DETAIL}` with outcome):

```bash
# 1. Mark stage checkbox: "- [ ] {N}." → "- [x] {N}."
sed -i "s/^- \[ \] ${N}\./- [x] ${N}./" "$P"

# 2. Update "Last updated" timestamp
sed -i "s/^\*\*Last updated\*\*:.*/**Last updated**: $(date -u '+%Y-%m-%d %H:%M') UTC/" "$P"

# 3. Append completion detail to log
echo "{DETAIL}" >> "$P"

# 4. Print dashboard
head -25 "$P"
```

Then output:
```
PROGRESS: stage [X] marked [x], log entry appended
```

### On Task Complete

```bash
sed -i "s/^\*\*Status\*\*:.*/**Status**: ✅ Complete/" "$P"
sed -i "s/^\*\*Last updated\*\*:.*/**Last updated**: $(date -u '+%Y-%m-%d %H:%M') UTC/" "$P"
```

### On Task Failure

```bash
sed -i "s/^\*\*Status\*\*:.*/**Status**: ❌ Failed/" "$P"
sed -i "s/^- \[ \] ${N}\./- [❌] ${N}./" "$P"
echo -e "\n### [$(date -u '+%H:%M')] ${N}. FAILED\n${REASON}" >> "$P"
```

### After Context Compaction (critical!)

If you are unsure what stage you are in, or feel like context was reset:

```bash
head -40 "$P"
```

Read the current stage from the TODO list, resume from the first `- [ ]` item.
**Do NOT restart completed stages.**

---

## Why Bash Commands Instead of Edit/Write

Previous versions used natural-language instructions ("MUST update", "MUST append") that
the LLM would acknowledge but skip the actual file operation. By providing exact `sed` and
`echo >>` commands:

1. **Append is atomic**: `echo >> file` needs no Read, no string matching
2. **Checkbox update is targeted**: `sed -i 's/- \[ \] N\./- [x] N./'` modifies one line
3. **Cannot be faked**: Either the Bash tool runs or it doesn't — no ambiguity
4. **No Read dependency**: All operations work without reading the file first

## Reading Optimization

- **Default read**: `head -40 "$P"` (TODO only, cheap)
- **Full read**: Only when debugging, reviewing history, or user requests it
- **Append**: Always `echo >>`, never rewrite the log section
- Continue to the next step in agent workflow
