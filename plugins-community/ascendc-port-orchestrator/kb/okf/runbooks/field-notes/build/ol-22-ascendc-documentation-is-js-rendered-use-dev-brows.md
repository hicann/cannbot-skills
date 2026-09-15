---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "AscendC documentation is JS-rendered — use dev-browser plugin to access"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "need to look up AscendC API, SIMD/SIMT programming guide, type support"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-22
timestamp_inferred: true
tags: [ascendc, ol-22]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: environment
- **Loaded by**: all skills
- **Trigger**: need to look up AscendC API, SIMD/SIMT programming guide, type support
- **Lesson**: AscendC official docs at hiascend.com are JS SPA — direct URL fetch returns empty or wrong page. Use dev-browser plugin with Playwright. Key documentation URLs:
  - API reference (CANN 9.0 beta2): `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlasascendc_api_07_10293.html`
  - Type conversion API (reg_convert.h): see CANN source `/usr/local/Ascend/cann-9.0.0/x86_64-linux/asc/include/c_api/reg_compute/reg_convert.h`
  - CANN source code: `~/workspace/cann` (local copy)
- **When to check docs**: Before assuming a type cast or API doesn't exist, ALWAYS check reg_convert.h and the CANN source for alternative APIs.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-22（category=environment，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
