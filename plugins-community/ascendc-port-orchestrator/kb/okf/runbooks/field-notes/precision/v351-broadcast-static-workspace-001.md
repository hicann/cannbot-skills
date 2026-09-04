---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "static Broadcast<T,N,axis,false>(...,workspace) sharing a workspace with RA-form ReduceSum undersizes it on V351 — switch to dynamic GetBroadcastTilingInfo API"
description: "57-D7: a static AscendC::Broadcast overload that shares the reduction workspace with an RA-form ReduceSum computes that workspace to the ReduceSum size, which is too small for Broadcast → systematic output mismatch that survives correct math. Workspace oversizing to feed Broadcast then triggers AICore timeout (507014/EE9999). Fix = dynamic GetBroadcastTilingInfo + Broadcast(&tiling), removing the shared-workspace dependency entirely."
phenomenon: precision_issue
signal:
  - "systematic mismatch persists after kernel math is proven correct by CPU simulation"
  - "static Broadcast overload taking a workspace argument that is shared with ReduceSum/reduction calls"
  - "workspace size changes cause AICore timeout (507014/EE9999) — the size itself is timeout-sensitive"
confidence: single_run
original_id: v351-broadcast-static-workspace-001
classified_by: llm-assisted
timestamp_inferred: false
tags: [ascendc, precision, broadcast, workspace, reducesum, v351]
created_at: 2026-08-26T12:42:00Z
updated_at: 2026-08-26T12:42:00Z
---
## 现象 / 触发

57_ParallelPolarizedSelfAttention_evo（iter D7）实测：

- kw-8 修复后仍 4/50 PASS，系统性 mismatch 不消；CPU 仿真确认意图 fp32 数学正确，残余缺陷定位到 `BroadTile` 里的静态 `AscendC::Broadcast<float,2,axis,false>(dst,src,dstShape,srcShape,redWs_)`——与 RA 形式 ReduceSum 共享 `redWs_` workspace，尺寸按 ReduceSum 需求算，对 Broadcast 欠尺寸。
- 连带事故：曾把 `redWs_` 从 8KB 加到 48KB 试图喂饱 Broadcast，立刻 AICore timeout（507014/EE9999，iter D6）；回退 24KB 后恢复。

## 修复

弃静态重载，改用 DAV_3510 动态 API：

```cpp
AscendC::GetBroadcastTilingInfo<float>(2, dstShape, srcShape, false, tiling);
AscendC::Broadcast<float>(dst, src, dstShape, srcShape, &tiling);
```

彻底移除共享 workspace 依赖；`redWs_` 缩回 8192B（只供 RA ReduceSum），最大 UB 预算回到 ~198KB。

## 动作规则

1. 迁移源码里静态 Broadcast 重载（dst,src,dstShape,srcShape,workspace 五参形态）且 workspace 与归约类调用共享 → 机械替换为动态 `GetBroadcastTilingInfo` API（trap_scan 规则 13）。
2. workspace 尺寸不要乱加：A5 上 workspace 过大可致 AICore timeout（57-D6）；先消除共享依赖，再按各 API 各自需求精确给尺寸。

## 证据

- 57_ParallelPolarizedSelfAttention_evo failures_ledger.md 行 15（D7）与行 14（D6，48KB workspace 超时副作用）。
