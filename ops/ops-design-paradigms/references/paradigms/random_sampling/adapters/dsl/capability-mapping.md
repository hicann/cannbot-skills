<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 范式 CANNBotDSL 能力映射

> 通用层 [capability-contract.md](../../generic/capability-contract.md) 谓词在 CANNBotDSL（下文简称 DSL，0.8.0，Ascend 950PR/950DT / NPU ARCH 3510 口径）下的真值。其他 SoC 须重新核实，禁止外推。
>
> 依据：cannbot-dsl 仓 `docs/zh/guide/`（compute / hardware-model / synchronization）、`docs/zh/api/compute/vector/（reg_* 类目）`。**DSL 仓无随机算子样例**——标注"类目成立、无先例"的项须在首例接入时实测收敛。

## 1 真值表

| 谓词 | 真值（DSL 0.8.0 / 3510） | 依据 |
|------|---------------------------|------|
| `CAP_PRNG_ALG` | **Philox4x32-10（语义锁定，载体自研）**——无公共底座；轮函数经 reg 算子（SIMD）或 `cb.simt` 标量数学（SIMT）自实现。常数池见 §2 | 通用层 prng-model §1；DSL API 面核对（无 RNG 类目） |
| `CAP_PRNG_VECTORIZED` | ✅ 类目成立、**无先例**——SIMD reg 算子齐备：`vmull`（int32/uint32 widening 乘，返回 (high,low) 对）、`vbitwise-and/or/xor`、`vshl/vshr`、`vadd/vaddc`、`vreinterpret`、`vcast`。验证义务：首例与流基准对拍 | reg_compute API（reg_arith/reg_logic/reg_convert） |
| `CAP_THREAD_LAUNCH` | ✅ 纯 SIMT：`kernel[dim3(block), dim3(thread)]`，thread ∈ [1,1024]、block ≤ 65535；混合：`vf(mode="simt", thread=N)` | compute.md「Host 启动规则」；scatter_elements MAX_CORE_NUM 口径 |
| `CAP_VEC_IN_SIMT` / `CAP_SIMT_IN_VEC` | `CAP_SIMT_IN_VEC` ✅——同一 `@kernel` 内 SIMD VF 与 SIMT VF **串接**（UB Buffer 交接）；`CAP_VEC_IN_SIMT` ⚠️ 限线程内短向量——VF 不可嵌套（一个 VF 只选一种模式），SIMT 线程内可用 `make_short_vector`/`v*` 短向量运算（非跨线程 SIMD） | compute.md 混合编程 / 接口分类「短向量」 |
| `CAP_INDEX_BIT_WIDTH` | 64bit 可全程承载（`cb.simt.cast(..., dtypes.int64)` 先例）；32bit 为快路径优化档（host 判 `numel ≤ INT32_MAX` 选路径）——块协议按 32bit 快路径缺失触发 | scatter_elements `has_flat_axis_coordinates` 先例 |
| `CAP_SPLIT_BLOCK_MAX` | **待验证**——无 enableSplitBlocks 底座；DSL 形态 = host 多段 launch + offset 累进（块数上限自定，首例登记） | — |
| `CAP_STATE_WRITE_BACK` | ✅ 类目成立、无先例——SIMT VF 内 GM 直写 + `cb.simt.threadfence*`（同步与 Cache 类目）；写者选唯一线程（`block_idx==0 and thread_idx==0`）。验证义务：可见性与时序首例实测 | compute.md 接口分类「同步与 Cache」 |
| `CAP_CORE_SYNC_ALL` | ✅ `global_sync_all()`（模式2→0→0→2 四次同步，禁内层循环；0.8 新增 `flag_ids` 参数——默认占 flag id 0/1/2，与手写 block 同步协议共存时须避开或显式指定）；替代惯用法：**回写拆独立顺序 launch**（stream 排序天然有序） | synchronization.md |
| `CAP_FAST_DIV` | ✅ host 侧可算（构图期纯 Python 预计算 magic/shift，同 AscendC `GetUintDivMagicAndShift` 算法）；kernel 侧可表达（SIMD `vmul/vshr`、SIMT 整数乘移位）。**无库先例**，首例登记实现 | jit-arguments（host 常量经 Constexpr/params 传递） |
| `CAP_LOCAL_MEM_HYBRID_SIZE` | SIMD 口径 248 KB（`get_mem_size("ub")`=253952，勿再减顶部预留）；开 SIMT 后按 **216 KB** 口径（SIMT Data Cache ~32 KB 不在 `get_mem_size` 返回内，自己留）；混用按最紧侧 | hardware-model.md danger 块 |

## 2 派生事实（通用层符号的 DSL 真值）

| 通用层符号 | 真值 | 出处 / 说明 |
|-----------|------|------------|
| k（步产出） | 4（Philox4x32 语义，语言无关） | prng-model §1 |
| R（每输出预留） | 256（预留律上界口径沿用通用层；DSL 侧无模板实测值，首例按 256 设计并复核） | splitting §5 |
| Philox 轮数 | **10**（流基准）；DSL 自研无模板默认值陷阱（AscendC 的 `<7>` 默认轮陷阱不存在，但轮数仍须显式常量登记） | prng-model §1.1 |
| Philox 常数 | W32_A=0x9E3779B9、W32_B=0xBB67AE85、M4X32_A=0xD2511F53、M4X32_B=0xCD9E8D57 | prng-model §1.1（语言无关） |
| u 构造口径 | 双口径均合法（位重解释 [0,1) / 乘法+半步 (0,1)）；DSL 落地：`vreinterpret`+位拼接（SIMD）或 `cb.simt.bitcast`+标量位运算（SIMT）——**与流基准同路径**（R7 硬规则） | distribution-transform §4 |
| seed=0 语义 | 按接口族（仅 C 族 TF 时间熵）；DSL host 构图期无法取时间熵进流——C 族时间熵分支在**公开入口**（构图前）取值传入 | state-management §2 |
| VL（SIMD 矢量长） | 256 B/寄存器（fp32 64 lane / fp16 128 lane）；32 个数据寄存器，超限溢出 UB | vector-registers.md |
| 块协议线程几何 | 纯 SIMT 惯例：512 线程/核起步（寄存器档位权衡，见 pitfalls §2.3） | compute.md「两个代价」 |

## 3 待验证项

| 项 | 验证义务 |
|----|---------|
| SIMD Philox 与流基准逐位一致（vmull 高低位序、掩码行为） | 首例：单核小批量与 torch/TF 对标序列逐位对拍 |
| u 构造两条口径的 DSL 实现值域 | 首例：按所选口径断言 [0,1) 或 (0,1) 边界（含 0 / 不含 0 用例） |
| offset 回写 threadfence 时序（跨核读一致性） | 首例：连续两次调用 offset 递进实测 |
| 快除 magic/shift 的 kernel 侧复算正确性 | 首例：与 `/` 结果全值域对拍 |
| 块协议多段 launch 的流区间隔离 | 首例：块边界处序列连续性验证 |
