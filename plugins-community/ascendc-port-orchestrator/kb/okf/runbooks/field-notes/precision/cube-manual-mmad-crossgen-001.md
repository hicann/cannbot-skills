---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V220 hand-written bare-mmad cube paths silently miscalculate on A5 (L0A transpose misplacement at blockM>=2) — route-guard them off and keep MatmulImpl (PB-43 cross-gen instance)"
description: "Cross-gen (A3→A5) instance of PB-43: 910B-era hand-written cube pipelines (*_manual.h, 16x16-fractal LoadData(ifTranspose=true) + bare Mmad hardcoding DAV_2201 L0A/L0B/L0C fractal layout) compute silently wrong on A5 when a tile spans multiple M fractals (blockM>=2, M>16) — max_abs_diff 32-148, output std ~13, no error. Same-op MatmulImpl (reserved==0) is bit-exact (diff=0.0) for fp32/fp16/bf16."
phenomenon: precision_issue
signal:
  - "diff magnitude >> rounding error (multiples of output std) and strongly correlated with blockM/tile size — tile misplacement, not accumulation precision"
  - "hand-written *_manual.h cube path comments claim reliability only on DAV_2201 / 910B"
  - "large-shape cases route to manual mmad paths despite the intuition that they use the library path"
confidence: single_run
original_id: cube-manual-mmad-crossgen-001
classified_by: llm-assisted
timestamp_inferred: false
tags: [ascendc, precision, cube, mmad, matmul, conv, crossgen, pb-43]
created_at: 2026-08-26T12:44:00Z
updated_at: 2026-08-26T12:44:00Z
---
## 现象 / 触发

MatmulTransA（Ascend950DT + CANN 9.2.0 实测）：

- A3 时代手写裸 cube 流水线（`*_manual.h`：逐 16x16 分形 `LoadData(ifTranspose=true)` + 裸 `Mmad`，硬编码 DAV_2201 的 L0A/L0B/L0C 分形布局）在 A5 上**当 tile 跨多个 M 分形（blockM≥2，M 维>16）时 L0A 转置错位、静默算错**（max_abs_diff 32~148，输出 std ~13）；blockM==1 时布局退化一致、碰巧正确。
- 同一算子里的高层库 `MatmulImpl`（reserved==0，MDL scheduler）在 A5 上对 fp32/fp16/bf16 **全部 bit 级正确（diff=0.0）**。

## 动作规则（A3→A5 cube 类移植第一步）

1. **先做路由探针**：给 host tiling 加调试输出，确认每种 dtype/shape 实际走哪条路径（reserved/分支），不要凭"大 shape 走库路径"的直觉下结论——MatmulTransA 的 fp16/bf16 大 shape 实测全走手工 mmad。
2. **禁用手工裸 cube 路径**（路由守卫 `kDisableManualCubePaths=true` 或删路由分支），全部落 MatmulImpl/高层库；tiny shape 若依赖手工路径破 ~13us 固定开销，需为 arch35 重写，正确性优先先禁用。
3. 修复前先看源码自证：手工路径头文件常有 "only ... reliable on DAV_2201" 类注释，即高危标记。

## 方法普适性验证（2026-08-22，deepseek-v4-flash × 2 agent 并行）

- **3_MatmulBothTrans_evo**：无手写裸 mmad，运行级零回归；阻塞在编译期——纯 AIC `*_cube.cpp` 在 CANN 9.2.0 arch35 报 set_padding 错误，最小修复 = `if ASCEND_IS_AIC` 设备侧守卫（~6 行/3 文件）。修复后 55/55 PASS diff=0.0。
- **8_ConvStandard2d_evo**：35/59 FAIL 全为 AIC 裸 cube 变体（裸 Mmad + 手工 L0 分形 load），NaN 输出；AIV Vector 路径全对。conv **无 MatmulImpl 可退** → `kDisableCubePaths` 全走 AIV → 59/59 PASS（1.2e-7~5.5e-6）。代价：大 K 慢 30-60×。
- **教训汇总**：① 路由探针是通用第一步（3/3 算子都靠它纠正前提）；② 手工裸 cube 路径跨代必炸（matmul/conv 两例实证）；③ 有库路径退库路径（零代价），无库路径退 AIV 兜底（正确但大性能回退）；④ 纯 AIC 编译在 arch35 需要 `ASCEND_IS_AIC` 守卫；⑤ 共享卡可能被外部会话占满——运行中也要盯 npu-smi。

## 证据

- `docs/mmta_fp16_rootcause_report.md`、`docs/conv2d_a5_rootcause_report.md`、`docs/mbt_a5_rootcause_report.md`。
- 关联：PB-43（cube 手工 fractal load 静默算错，build-success≠validation）——本条即其跨代移植实例。
