---
schema_version: okf.v1
kind: guide
type: optimization_runbook
source_family: curated
title: "A3→A5 source_migration 排查总纲：路由探针 → 原位修复阶梯 → CPU 仿真分算法/执行 → NaN 哨兵 → 楔死清单 → near-miss bisect"
description: "Cross-gen (source_migration) diagnostic master recipe distilled from the 45/55/57 campaign ledgers: probe the actual code path first, fix in place along the trap ladder, split algorithm vs execution with CPU simulation against the FROZEN reference, use NaN sentinels to separate never-executed from executed-wrong, run the wedge checklist for 507035-class device faults, and bisect near-misses instead of tuning tolerance first."
confidence: single_run
original_id: a5-source-migration-recipe-001
timestamp_inferred: false
tags: [ascendc, crossgen, diagnostic, recipe, precision, npubench]
created_at: 2026-08-26T12:46:00Z
updated_at: 2026-08-26T12:46:00Z
---
# 排查顺序（按信号具体度从高到低）

1. **路由探针先行**：给 host tiling 加调试输出，确认每种 dtype/shape 实际走哪条代码路径——MatmulTransA 的 fp16/bf16 大 shape 实测全走手工 mmad 没走 MatmulImpl；3/3 算子（matmul/conv/mbt）都靠路由探针纠正前提。
2. **原位修复阶梯**（默认，高性能路径不动）：按 trap_scan 清单逐条修（TQue depth、PIPE 事件配对、Exp config、Broadcast workspace、UB 预算、入口宏、Release/-O2、裸 cube 守卫）。降级（AIV/纯 VEC 兜底）是例外，需要客观死亡证据（最小原语复现/跨代死亡归档豁免）；45 iter1 整体降级属降级失控案例。
3. **首次大面积错（≥80% case FAIL）强制一次 CPU 仿真**：参照系 = 冻结 NPUKernelBench reference（不是 kernel 意图自洽）。仿真过 → 执行缺陷 → trap 清单/bisect；仿真不过 → 算法错 → 修算法前不许再上 NPU。55-kw6 / 57-D4 实证。
4. **NaN 哨兵分"执行了但错" vs "根本没执行"**：host 侧预填 NaN，全 NaN=kernel 未执行（查入口宏/注册/launch），有限垃圾=执行了但算错（查同步/数据通路）。55-kw4 固化。
5. **楔死清单**（507035/507014/EE9999 且 build 正常）：DataCopyPad UB→GM（EC-23）、2D ReduceSum 步长、SetFlag/WaitFlag 配对、V_MTE2/V_MTE3 HardEvent；build 为 Debug/-O0 先判 PB-5（45 iter4）。
6. **near-miss 走逐 kernel/stage bisect，不先改精度**：57-D3 根因是 UB 越界不是精度；多 kernel 流水线用 env-gated 中间输出（OA_DBG_KERNELS=1..7）定位第一个坏 stage。

## 反模式（均有 ledger 实证）

- 打地鼠（每颗雷留到运行期引爆，烧一轮 worker + 一次全量评测）；
- 凭直觉下根因结论不跑路由探针；无死亡证据就整体降级；
- near-miss 先调容差/精度（57-D3 教训）；把 Debug/-O0 的 507035 误判成 infra 空转（45 iter4）。

## 证据

- 55_OutlookAttention failures_ledger.md 行 42-54（NaN 哨兵）、行 71（CPU 仿真）；
- 57_…_evo failures_ledger.md 行 11-12（UB 越界 near-miss、CPU 仿真）；
- 45_CrossformerAttention failures_ledger.md 行 11-13（降级、fp32 未初始化读、PB-5）。
