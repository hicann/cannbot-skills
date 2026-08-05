---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Container `CA_cann_9_b2_kevin` is compile-only — stub CANN libraries, `aclInit` returns 100039"
description: "applies_to: soc=Ascend910_9382; cann=9.0.0-beta.2; bisheng=n/a; op_class=all"
phenomenon: build_failure
signal:
  - "Kernel builds successfully and .so is produced, but aclInit returns error code 100039 (\"stub library cannot be used for execution\"). All subsequent ACL calls fa"
confidence: single_run
original_id: EC-45
timestamp_inferred: true
tags: [100039, ca_cann_9_b2_kevin, aclinit, ascendc, ec-45]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_9382; cann=9.0.0-beta.2; bisheng=n/a; op_class=all`
- **Severity**: HIGH (runtime blocker — kernel builds but cannot execute)
- **Status**: CONFIRMED 2026-05-07 op#30 NMS ds kw-1
- **Symptom**: Kernel builds successfully and .so is produced, but `aclInit` returns error code 100039 ("stub library cannot be used for execution"). All subsequent ACL calls fail.
- **Root cause**: container image was built with stub/development CANN libraries, not full runtime. NPU passthrough is absent or the installed CANN lacks libascendcl.so with runtime backend.
- **Detection rule**: after build, run a minimal `aclInit` + `aclrtSetDevice` smoke test before proceeding to precision/perf verification. If `aclInit` returns non-zero, the container cannot run kernels — switch to a runtime-capable container or accept build-only verification.
- **Workaround**: runtime verification must use a container with full CANN runtime + NPU device passthrough (e.g., `--device=/dev/davinciX`).
- **Generalizes to**: any container where CANN was installed from a dev/stub package rather than the full runtime package. Check `aclInit` return code as the first step of any runtime verification.
- **Evidence**: op#30 NMS ds kw-1 (2026-05-07) — build succeeded (9.2MB .so), `aclInit` returned 100039 on CA_cann_9_b2_kevin. Runtime verification was performed on a different container/setup.

<!-- 迁移自 porter kb/target/ascendc/（EC-45，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
