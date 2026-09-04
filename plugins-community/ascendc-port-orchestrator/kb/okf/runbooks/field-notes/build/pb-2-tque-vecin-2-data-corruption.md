---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "TQue<VECIN,2> Data Corruption"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "99.5% elements corrupted when using TQue with depth 2"
confidence: single_run
original_id: PB-2
timestamp_inferred: true
tags: [ascendc, pb-2]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-08-26T12:52:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: 99.5% elements corrupted when using TQue with depth 2
- **Affected**: Ascend950PR, CANN 9.0.0
- **Workaround**: Use TQue<VECIN,4> (depth 4 works correctly)
- **Status**: OPEN
- **Evidence**: hardware/target/ascend950pr.md, E13 test data

<!-- 迁移自 porter kb/target/ascendc/（PB-2，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

## V351 实战补充（2026-08-26，57_ParallelPolarizedSelfAttention_evo D1 沉淀）

- 57 是 PB-2 在 V351 的活实例：A3 上验证过的 `TQue<...,2>` + InitBuffer depth-2 在 A5 输出全零——O5 45/50 FAIL 且 `matched_count==small_count`（57 ledger 行 9）。
- 修复：3 个 TQue（xQue_/xOutQue_/wQue_）depth 2→4 + InitBuffer depth 参数同步 2→4（ledger 行 9）。
- **联动代价**：depth 翻倍直接放大 UB 占用；57 升 depth-4 后 11 个冻结 shape 的 InitBuffer 总和越 A5 248KB 可用上限（ledger 行 11）。**升 depth 后必须立即重算全部 InitBuffer 字节和**；越界就减容 / 转 TQue<VECIN,1> / 重排 buffer 顺序。
- 判别注意：全 case mismatch 且输出全零时先确认 small_count>0，否则 `matched_count==small_count` 是伪迹（57-D3）。
