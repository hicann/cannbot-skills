---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`deploy_to_npu.sh` TARGET case statement rejects `a3-ds` and other alias targets"
description: "applies_to: soc=all; cann=all; bisheng=all; op_class=all"
phenomenon: build_failure
signal:
  - "deploy_to_npu.sh's case statement only accepts a5|a3|a2 for TARGET. When .ascendc_env sets TARGET=a3-ds (DS-backend isolation), the script exits with no matchin"
confidence: single_run
original_id: EC-44
timestamp_inferred: true
tags: [ascendc, ec-44]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=all; cann=all; bisheng=all; op_class=all`
`verified_on: soc=Ascend910_9382; cann=9.0.0-beta.2`
- **Severity**: LOW (deploy-time blocker, workaround exists)
- **Status**: OPEN
- **Symptom**: `deploy_to_npu.sh`'s case statement only accepts `a5|a3|a2` for TARGET. When `.ascendc_env` sets `TARGET=a3-ds` (DS-backend isolation), the script exits with no matching case and deploy is blocked.
- **Workaround**: manual `tar+scp+docker exec` deploy, or temporarily change TARGET in .ascendc_env.
- **Fix**: add case aliases in the script (`a3-ds|a3_kimi|a3`) all mapping to the a3 code path.
- **Detection rule**: if `deploy_to_npu.sh` exits silently without deploying, check that TARGET matches one of the script's expected values.
- **Evidence**: op#30 NMS a3 ds kw-1 (2026-05-07) — manual tar+scp+extract deploy required because script rejected TARGET=a3-ds.

<!-- 迁移自 porter kb/target/ascendc/（EC-44，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
