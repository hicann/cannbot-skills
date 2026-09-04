---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V351 '读钳制 L=mS-S 触发 + PIPE_ALL 屏障 + 标量 SetValue 尾清零块' 执行即楔死设备（507035/errcode 264）；修复 = 加宽源缓冲使钳制永不触发"
description: "V351 '读钳制 L=mS-S 触发 + PIPE_ALL 屏障 + 标量 SetValue 尾清零块' 执行即楔死设备（507035/errcode 264）；修复 = 加宽源缓冲使钳制永不触发. Applies: soc=Ascend950DT/950PR(V351/arch35); cann=9.2.0; mode=port_a3_to_a5; op_class=vector/attention im2col/sliding-window 类 DataCopyPad 行跨步读 + 尾清零. Provenance: 42_CoTAttention kw-28 前置诊断（2026-08-27 主线 kimi 会话，卡 3 逐 case 隔离 + 逐 launch 同步 + kernel 体二分 5 轮全实测）"
phenomenon: build_issue
confidence: multi_run
original_id: user-v351-datacopypad-clamp-lrnd-zeroing-trap-001
classified_by: llm-assisted
timestamp_inferred: false
tags: [ascendc, v351, arch35, build, 507035, wedge, datacopypad, im2col]
created_at: "2026-08-28T03:00:00Z"
updated_at: "2026-08-28T03:00:00Z"
---

# 读钳制触发的 lRnd 尾清零块（PipeBarrier<PIPE_ALL> + 标量 SetValue 循环）在 dav_3510 上执行即楔死设备；正解 = 加宽源缓冲消除钳制条件

## 事实（42_CoTAttention，Ascend950DT + CANN 9.2.0，卡 3 全实测）
- 现象：7/50 case 确定性 507035（device error type 3，aivec errcode 264 "scalar access GM invalid"，其余核 "timeout or trap"），跨 fp16/bf16/fp32；形状触发（hw 大且尾 tile 落在正向 tap 区的形状）。
- 定位链（可复用三板斧）：① 逐 case 子进程隔离探针锁定崩溃 case 集；② host 每个 launch 后插 `aclrtSynchronizeStream` 返回值检查（**host 侧 launch 后不查 sync 返回值时，故障归因会漂移到下一个 aclnn 算子**——本案 Slice/aclnn 背锅多时）锁定首个 kernel im2col；③ kernel 体内 `#ifdef` 二分（跳读/跳写/置空/跳清零块）钉死到唯一代码块。
- 真凶块（上午尾 tile 修复引入）：
  ```cpp
  int32_t lRnd = (L + 15) & ~15;
  if (lRnd > L) { PipeBarrier<PIPE_ALL>();
      for (int32_t c = L; c < lRnd && c < act; ++c) ub.SetValue(..., 0); }
  ```
  只跳这一块 → 全 PASS；只保留这块（跳另一清零块）→ 崩。**该块只在读钳制 `L = mS_ - S` 触发（L 非 16 对齐）时执行**——不触发则一切正常，这就是形状选择性。
- 钳制为什么会被触发：源缓冲只给了 `RoundUp64(mS + deltaMax)`，但正向 tap 读窗上界 = `m + shift + deltaMax + act ≤ mS + 2*deltaMax`（shift 最大 +deltaMax，零前缀补偿再加一个 deltaMax）——**缓冲窄了一个 deltaMax**。
- 修复（已实测）：源缓冲加宽到 `RoundUp64(mS + 2*deltaMax)` → 钳制 provably 永不触发 → 清零块死代码化；正向 tap 读到 at::zeros 真零，语义与钳制+清零完全等价。7 崩溃 case + 5 回归 case 12/12 PASS。

## 动作规则
1. im2col/滑窗类 kernel 用"零前缀 + deltaMax 偏移"读窗模式时，源缓冲宽度必须是 `mS + 2*deltaMax`（前缀 deltaMax + 正向 tap 余量 deltaMax），只加前缀一半必在尾 tile 正向 tap 触发读钳制。
2. 钳制后的非对齐 L 清零块（PIPE_ALL + 标量 SetValue）在本平台是楔死炸弹——优先用"加宽缓冲让钳制死代码化"消除触发条件，不要试图保留该路径。
3. 507035 异步归因陷阱仍在：host launch 序列里不查 sync 返回值时，报错会落在后面的 aclnn 算子上；诊断先插逐 launch sync 检查再谈归因。
4. 与 `v351-datacopypad-32b-roundup-oob-001` 的关系：那条的"扩缓冲无效"指的是 +16 元素级微调（被 RoundUp64 吞掉且多读落点是下一行）；本条是结构性加宽一个 deltaMax 以消除钳制本身——两者不矛盾，钳制消失后那条的读后清零也不再需要执行。

## 证据
- a5-data/cot-dsh-diag/isolate42/：isolate42_result.json（7 崩溃 case）、case7_synccheck.log（首败=im2col sync）、case7_bisect_{skipread,skipboth,nozeroing,noA}.log（二分链）、widen_verify.{log,json}（修复 12/12）。
- docs/cot-fp32-gemm-kimi-diag-20260827.md §9。
- open question：清零块 trap 264 的微机制（PIPE_ALL+标量 UB 写组合为何 trap）未完全定性，修复路线是消除触发条件。
