---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "SIGTERM does NOT propagate through `ssh + docker exec` chain — wrap remote workloads with `timeout` to bound them"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "paradigm: ascendc"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-121
timestamp_inferred: true
tags: [timeout, ascendc, ol-121]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Category**: process / infrastructure / orchestration
- **Loaded by**: aog-kernel-worker (Phase D perf measurement), aog-kernel-optimizer (every iter perf re-measure), aog-precision-probe (when running on-chip probes), any workflow that spawns remote python3/binaries through `ssh user@host "docker exec container <cmd>"`

### Principle

When the orchestrator (or any local agent) kills the local `ssh` / `sshpass+ssh` process via SIGTERM, the **remote-side** process (the python3 / binary running inside the docker container on the target host) does NOT receive the signal. Signal flow stops at the local ssh process boundary; the remote container keeps the orphaned process alive until its own completion (or container restart).

This causes **silent zombies** that occupy NPU/HBM resources for hours. The orchestrator believes "I killed that perf measurement" while the target-side process is still running and blocking new agents from using the same NPU.

### Diagnostic signature

- Local ssh process killed N minutes ago
- New agent attempts to use the same NPU + reports "NPU busy" or perf numbers contaminated by the zombie's load
- `docker exec <container> ps aux | grep python3` on the remote host shows long-running processes (hours-old) that have no corresponding local ssh
- `npu-smi info` shows process pid for an NPU that no longer maps to any active orchestrator workload

### Concrete anchor

```bash
# WRONG: orphan-prone
sshpass -p "$PWD" ssh root@$HOST "docker exec $CONTAINER bash -c 'python3 perf.py ...'"

# RIGHT: timeout bound, won't outlive its budget
sshpass -p "$PWD" ssh root@$HOST "docker exec $CONTAINER bash -c '
  timeout 540 python3 perf.py ...
'"

# ALSO ACCEPTABLE: explicit cleanup-on-trap (more code, but signal propagation works)
sshpass -p "$PWD" ssh -t root@$HOST "docker exec $CONTAINER bash -c '
  trap "kill 0" SIGTERM SIGINT
  python3 perf.py ...
'"  # -t forces TTY; kill 0 = kill all in process group
```

### Detection workflow (pre-spawn pattern)

Before spawning any remote-side compute, check for orphans on the target NPU:

```bash
sshpass -p "$PWD" ssh root@$HOST "docker exec $CONTAINER ps aux 2>&1 | grep -E '(python3|kernel)' | grep -v grep"
# Look for processes >1h old that don't correspond to active orchestrator state
```

### Evidence

- 2026-05-02 (this session): op 10 ko-1 spawned `python3 utils/performance.py` via ssh+docker exec for `perf_opt0_c.json`. ko-1 hung on a separate wait-loop bug (OL-122 below) and was TaskStop'd at 11:54. Local sshpass+ssh killed at 12:14. **In-container python3 PID 1584422 kept running another 7+ hours** until orchestrator manually `docker exec ... kill` killed it. The zombie occupied NPU 0 the entire time, would have contaminated subsequent ko-2 perf measurements if not caught.

### Other instances (predicted)

- Any benchmark / perf agent that spawns remote workloads
- Long-running probe scripts on remote hosts
- Multi-NPU experiments where NPUs are shared resources
- Cross-host orchestration (arch35 ↔ arch22)
- CI/CD pipelines that ssh into containers

### Anti-patterns (DO NOT)

- ❌ Spawn `python3 perf.py` via ssh+docker exec without `timeout N` wrapper
- ❌ Assume `kill <local-ssh-pid>` ends the workload — only ends LOCAL view of it
- ❌ Build agent wait-loops that "wait for the ssh to return" when the underlying compute might have orphaned

### Mandatory pattern (now codified in worker / optimizer briefs)

All remote python3 / binary invocations in worker / optimizer / probe briefs MUST include:
```
timeout <budget> python3 ...
```

`<budget>` should be set generously (e.g., 540s for perf measurements, 1200s for builds) but bounded — even if your local ssh dies, the remote process self-terminates within the budget. Already-active pattern in op 22's brief; needs to be standard.

### Related

- aog-self-critic C13 (claim runtime state without verification — "killed the process" claim must be verified by `docker exec ps aux` on the remote)
- CLAUDE.md "Shared NPU — performance test prep" — A5 is shared infra; orphans hurt everyone

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-121（category=process / infrastructure / orchestration，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
