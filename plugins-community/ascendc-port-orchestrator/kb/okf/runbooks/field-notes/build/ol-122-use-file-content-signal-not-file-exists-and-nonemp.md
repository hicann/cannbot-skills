---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Use file-CONTENT signal, not file-EXISTS-AND-NONEMPTY signal, for build-status wait loops"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "paradigm: ascendc"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-122
timestamp_inferred: true
tags: [ascendc, ol-122]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Category**: process / infrastructure / build pipeline
- **Loaded by**: aog-kernel-worker (Phase C build-and-wait), aog-kernel-optimizer (per-iter rebuild), any agent polling for build outcome via filesystem markers

### Principle

When using a filesystem marker to communicate build outcome (e.g., `.last_build.class` containing "pass" / "compile" / "infra"), the wait loop must check **file content**, not file emptiness. Specifically: do NOT use `[ -s file ]` (non-empty check) as the "build done" sentinel — that pattern hangs forever when the producer clears the file to empty on success.

The fix is symmetric on producer + consumer:
- Producer (build script) writes a non-empty sentinel like "pass" on success
- Consumer (agent wait loop) checks `[ -f file ]` (exists) and reads content

### Concrete anchor

```bash
# WRONG: hangs forever when build succeeds + file is cleared empty
until [ -f $WS/.last_build.class ] && [ -s $WS/.last_build.class ]; do
  sleep 5
done
cat $WS/.last_build.class   # only runs if file was non-empty

# WRONG: producer clears on success → consumer's `-s` check never satisfied
: > $WS/.last_build.class   # producer side, on PASS

# RIGHT: producer writes content sentinel
echo "pass" > $WS/.last_build.class   # producer side, on PASS

# RIGHT: consumer waits for FILE to exist, reads content
until [ -f $WS/.last_build.class ]; do sleep 5; done
result=$(cat $WS/.last_build.class)
case "$result" in
  pass)    echo "build PASS";;
  compile) echo "build FAIL: compile error";;
  infra)   echo "build FAIL: infra error";;
  *)       echo "build status unknown: $result";;
esac

# BEST: don't use wait-loops at all when the build command is synchronous
# `bash deploy_to_npu_lane.sh --lane N --build` is synchronous — it returns when
# build completes. Just chain `&& cat $WS/.last_build.class`:
bash src/scripts/deploy_to_npu_lane.sh --lane $L --build && cat $WS/.last_build.class
```

### Diagnostic signature

- Agent hangs for many minutes despite build completing successfully
- `cat .last_build.class` shows empty file (0 bytes)
- Agent's last bash command is a `until [ -s ... ]` wait-loop
- Process tree shows the wait bash burning CPU on `sleep 5` cycles

### Evidence

- 2026-05-02 (this session): op 10 ko-1 worker spawned a `until [ -f X ] && [ -s X ]; do sleep 5; done` wait-loop. `deploy_to_npu.sh` cleared CLASS_MARKER to empty on success. Loop hung 20+ minutes before TaskStop. V3.7.7 fix: producer now writes `"pass"` (5 bytes) to CLASS_MARKER on success; consumer wait-loops can drop the `-s` check or just read content.

### Anti-patterns (DO NOT)

- ❌ `[ -s file ]` as "result available" check when producer might write empty-on-success
- ❌ Empty-file as a meaningful semantic signal — empty == "not done" OR "done with nothing to report" is ambiguous; pick exactly one
- ❌ Wait-loops at all when the producing command is synchronous

### Mandatory pattern (V3.7.7+)

For deploy_to_npu.sh + agent wait-loops:
- Producer: `echo "pass" > $CLASS_MARKER` on success (NOT `: > $CLASS_MARKER`)
- Consumer: chain `&& cat $WS/.last_build.class` directly OR use `[ -f ]` exists check
- NEVER use `[ -s ]` as the build-done signal

### Related

- aog-self-critic C13 (verify runtime state — "build done" must be verified by content, not heuristic)
- OL-121 (sibling: SIGTERM propagation — same family of "local view of remote/async state can lie")

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-122（category=process / infrastructure / build pipeline，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
