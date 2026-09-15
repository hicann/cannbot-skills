---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "STOP and search existing skills before inventing workarounds"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "when hitting infrastructure problems (network, auth, deployment)"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-13
timestamp_inferred: true
tags: [ascendc, ol-13]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: process
- **Loaded by**: Builder, Optimizer, QA
- **Trigger**: when hitting infrastructure problems (network, auth, deployment)
- **Lesson**: The A5 container has proxy access via `a5_exec.py --proxy` (sources `/home/npu_user/setup_proxy.sh`). The agent spent 30+ minutes on scp/tar/docker cp workarounds instead of checking the `/a5_op` skill documentation it was already using. The proxy flag enables git clone, pip install, and all internet access. Always check existing skill docs FIRST.
- **Evidence**: Session 2026-03-29, SG deployment failures

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-13（category=process，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
