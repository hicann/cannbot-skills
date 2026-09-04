---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V351 DataCopyPad 尾 tile 按 32B 上取整多读，读到的是下一行真实数据而非零区，污染真实输出列"
description: "V351 DataCopyPad 尾 tile 按 32B 上取整多读，读到的是下一行真实数据而非零区，污染真实输出列. Applies: soc=Ascend950DT/950PR(V351/arch35); cann=9.2.0; mode=port_a3_to_a5; op_class=vector/attention 尾 tile 裁剪 + DataCopyPad 载入（im2col/sliding-window 类）. Provenance: 42_CoTAttention kw-23 翻案（2026-08-27，dsh 反演 + kimi dump 列剖面 + 主线 kimi 会话修复实测验证）"
phenomenon: precision_issue
confidence: multi_run
original_id: user-v351-datacopypad-32b-roundup-oob-001
classified_by: llm-assisted
timestamp_inferred: false
tags: [ascendc, v351, arch35, precision, datacopypad, tail-tile, oob-read]
created_at: "2026-08-28T03:00:00Z"
updated_at: "2026-08-28T03:00:00Z"
---

# DataCopyPad 尾 tile 32B 上取整：多读的是"下一行的数据"（行跨步寻址），host 侧扩缓冲是无效修复；正解 = 读后在 UB 里重清零取整尾

## 事实（42_CoTAttention，Ascend950DT + CANN 9.2.0 实测，全部卡 3 复现）
- 形态：im2col 尾 tile 裁剪后 `L = mS − S` 非 32B 对齐（case1：L=409 元素=818B）→ `DataCopyPad` 按 32B 上取整读 416 元素。**关键：源是行跨步寻址 `xGM_[srcRow*mS_ + S]`，多读的 7 个元素落在缓冲内下一行的开头（真实数据），不是缓冲区尾巴的零**。→ ub 列 409..415 被邻行数据污染 → 写出到真实输出列 m=6041..6047（ki=7 是唯一 `dw==0` 的 tap，跳过边界清零，其他 tap 的越界列会被 `dw != 0` 块兜住）。
- **无效修复（亲测零效果）**：host 侧 `mSReadLocal = RoundUpMultiple(mS + deltaMax + 16, 64)`——① `RoundUp64(6080+84)` 和 `RoundUp64(6080+100)` 都是 6208，缓冲根本没变大；② 即使变大，多读落点也是下一行数据而非零区。dsh 的 §8.2 方案已被实测证伪（work3 构建 xColTHi 污染 13248 元素与未修版逐字节相同）。
- **有效修复（卡 3 实测验证）**：kernel 内读后重清零取整尾——
  ```cpp
  int32_t lRnd = (L + 15) & ~15;
  for (int32_t c = L; c < lRnd && c < act; ++c)
      ub.SetValue(r * COT_IM2COL_MBLK + c, static_cast<T>(0));
  ```
  （tile 在进入读循环前已整体 Duplicate 置零，只需补回被取整多读覆盖的 `[L, roundUp16(L))` 区间。）修复后 case1 xColTHi 真实列全清（badcols real=0，残留 21 列全在 pad 区且下游 GEMM 逐列隔离不外溢）。
- 判别签名：**matched_ratio 几乎不动、MERE/max_error_cap 爆表**（本案 7/6059 列垃圾独撑 MERE 0.57）→ 优先查尾 tile 边界，不要归为"精度地板"。
- 影响面分析方法：逐位重建比对后按列坐标聚类；逐 case 解析计算哪些 case 越界落真实列（42 案 11 个 fp32 case 仅 1/27/43 三个污染真实列）。

## 动作规则
1. 迁移源码有"尾 tile 裁剪 + DataCopyPad 读" → 直接加读后重清零（上面的模板），不必先烧评测确认。
2. 不要试图用"扩大 host 缓冲"修复行跨步寻址的上取整多读——多读的是下一行，扩缓冲无效。
3. fp16/bf16 同路径同样受影响（同一 im2col），修复后应一并复查。

## 证据
- 修复验证实验：cot-dsh-diag/work（kw-23+修复）阻塞模式 case1：xColTHi badcols real=0/pad=21（修复前 real=7）；最终 out 真实列 <1e-3 全清。
- docs/cot-fp32-gemm-dsh-diag-20260827.md §3.4（机制反演，结论对但修复错）；docs/cot-fp32-gemm-kimi-diag-20260827.md §1 缺陷 2（dump 列剖面独立确认）。

## 关联 KB
- `v351-fp32-hilo-3mma-mte1-m-fence-001`（同算子的 fence 假设——已被证伪，见该条目）。
- DataCopyPad 32B 对齐语义是 arch35 通用行为，任何"尾 tile 裁剪 + DataCopyPad"迁移源码都应过此检查。
