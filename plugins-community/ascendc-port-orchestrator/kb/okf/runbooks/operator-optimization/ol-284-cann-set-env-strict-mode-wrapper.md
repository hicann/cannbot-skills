---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Source CANN set_env.sh outside shell strict mode, then validate the environment"
description: "CANN environment scripts can exit a set -euo pipefail caller on benign branches; relax strict mode only around source, restore it, and validate ASCEND_HOME_PATH."
paradigm: ascendc
confidence: single_run
original_id: OL-284
timestamp_inferred: false
tags: [ascendc, environment, set-env, shell, strict-mode, ol-284]
created_at: 2026-07-29T00:00:00Z
updated_at: 2026-07-29T00:00:00Z
---

`applies_to: soc=all; cann=all; op_class=build/deploy/verification harness; backend=ascendc`

## Principle

CANN environment scripts may read an unset `LD_LIBRARY_PATH`/`PYTHONPATH`, run lookup pipelines whose misses are benign, or return non-zero from optional branches. A caller running `set -euo pipefail` can therefore exit while sourcing an otherwise usable environment. Relax strict mode only around `source`, restore it immediately, and validate a required postcondition instead of trusting the source command's return code.

```bash
set +eu
set +o pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source "${CANN_ROOT}/set_env.sh" || true
set -eu
set -o pipefail

test -n "${ASCEND_HOME_PATH:-}" || {
    echo "CANN environment initialization failed" >&2
    exit 1
}
```

Use this relaxation only for the vendor environment script; keep the caller's build/deploy logic strict. A fresh `bash -c` or container exec must source the environment explicitly. Cross-ref EC-51 and OL-180.

**Evidence / provenance**: derived from historical card TR-OL-20. On 2026-05-17, a hardened install script exited silently at `source set_env.sh`; three revisions isolated the strict-mode cascade. A separate 22_Nonzero non-interactive verifier failed every case at `aclrtSetPrecisionMode` with error 500001 until the environment was sourced. The wrapper is measured; the broader shell rule is the reusable generalization.
