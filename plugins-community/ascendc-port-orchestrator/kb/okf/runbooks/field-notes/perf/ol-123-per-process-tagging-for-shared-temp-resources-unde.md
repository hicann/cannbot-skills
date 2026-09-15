---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Per-process tagging for shared-temp resources under parallel-batch workloads"
description: "paradigm: ascendc"
phenomenon: perf_regression
signal:
  - "paradigm: ascendc"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-123
timestamp_inferred: true
tags: [ascendc, ol-123]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Category**: process / infrastructure / parallelism / orchestration
- **Loaded by**: any script invoked by N concurrent agents on the same host (deploy scripts, build scripts, profiling scripts), aog-kernel-worker / aog-kernel-optimizer briefs that include shell command templates

### Principle

When N concurrent workers run on the same host (e.g., parallel-batch dispatch with multiple lanes), any shared-name temp file (e.g., `/tmp/<fixed_name>.tar.gz`) is a race-condition surface. One worker's `mv` / `tar czf` will clobber another's in-flight upload. The race window is between "create temp file" and "consume/upload temp file" — usually milliseconds, but probabilistic-collision rises with concurrency.

**The fix is uniform**: tag the temp resource with `${LANE:-0}_$$` (lane id + PID) so concurrent workers don't collide. Cleanup the tagged file on success.

### Diagnostic signature

- Multiple concurrent deploys → some succeed, some fail with silent exit-2 + empty stderr_marker
- File-content corruption: extracted tarball has wrong content (another worker's payload)
- Worker reports "I sent file X but server received Y" — content mismatch between local and remote
- No deterministic repro on single-worker runs, only fires under N>2 concurrency

### Concrete anchor

```bash
# WRONG: shared name → race-prone under concurrency
tar czf /tmp/current_task_deploy.tar.gz -C "$LOCAL_TASK" .
scp /tmp/current_task_deploy.tar.gz user@host:/path/

# RIGHT: per-process tag
TAR_TAG="${LANE:-0}_$$"
TAR_LOCAL="/tmp/current_task_deploy_${TAR_TAG}.tar.gz"
TAR_NAME="current_task_deploy_${TAR_TAG}.tar.gz"  # remote path also tagged
tar czf "$TAR_LOCAL" -C "$LOCAL_TASK" .
scp "$TAR_LOCAL" "user@host:/path/$TAR_NAME"
ssh user@host "docker exec ... tar xzf /path/$TAR_NAME && rm -f /path/$TAR_NAME"
rm -f "$TAR_LOCAL"  # cleanup on success
```

### Detection workflow

1. If you see worker reporting "deploy script exit 2 with empty stderr" under parallel batch — suspect race
2. Check the deploy script for any hardcoded `/tmp/<name>` paths — those are race surfaces
3. Run `lsof | grep /tmp/<name>` while batch is running — if N processes hold it, that's the bug
4. Check timestamps: temp file's mtime jumps backward (clobbered) during the deploy window

### Evidence

- 2026-05-02 (this session): 12_KvRmsnormRopeCache kw-1 worker spent 3 deploy attempts hitting silent exit-2 / empty markers despite analysis + kernel + Phase O2.5 all complete. Root cause: `/tmp/current_task_deploy.tar.gz` shared across 4+ concurrent workers (lane 1, 2, 4, ds, plus this op's lane 0). Worker rolled own isolated deploy with PID-tagged tar to escape. V3.7.8 deploy_to_npu.sh fix tags tar with `${LANE}_$$`.
- Side-effect: worker bypassed `BUILD_ARCHIVE_ENABLED=1` because it used isolated deploy → op 12's build trajectory NOT captured in `/data/build_archive/`. Lost training data for GEPA.

### Other instances (predicted)

- ANY orchestrator script that does `tar czf /tmp/<fixed_name>` or `scp /tmp/<fixed_name> ...`
- Build artifact collection scripts using `/tmp/build_output/` shared dir
- `docker cp src dst` where src is a shared local file
- Probe scripts that write `/tmp/probe_output.json` from N concurrent probe agents
- Any cron job + manual-invoke that overlap on a shared marker file

### Anti-patterns (DO NOT)

- ❌ `/tmp/<fixed_name>` for any temp file used by parallel workers
- ❌ "It works in single-worker tests" — test under N=4+ concurrency before declaring stable
- ❌ Per-LANE tag without per-PID — multiple respawns of same lane can still race within retries

### Mandatory pattern (V3.7.8+)

For any shared-temp resource in parallel-batch scripts:
- Tag with `${LANE:-0}_$$` (lane id + PID)
- Clean up the tagged file on success in deploy script (not relying on /tmp cron cleanup)
- Pass tagged name to remote side too — `scp src user@host:/path/$TAR_NAME` not `scp src user@host:/path/` (which uses src's basename, defeating the tag)

### Related

- aog-self-critic C13 (verify runtime state — "deploy succeeded" must be verified by remote ls + content check, not just exit code)
- OL-121 (cross-host process lifetime — same family: local-view-of-remote-state can lie under concurrency)
- OL-122 (file-content vs file-existence semantics — same family: file-based signaling needs unique-by-construction names)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-123（category=process / infrastructure / parallelism / orchestration，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
