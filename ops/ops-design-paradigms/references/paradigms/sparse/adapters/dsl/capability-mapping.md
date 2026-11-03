<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式 CANNBotDSL 能力映射

> 通用层 [capability-contract.md](../../generic/capability-contract.md) 谓词在 CANNBotDSL（下文简称 DSL，0.7.0，Ascend 950PR/950DT / NPU ARCH 3510 口径）下的真值。其他 SoC 须重新核实，禁止外推。
>
> 依据：cannbot-dsl 仓 `docs/zh/guide/`（hardware-model / compute / synchronization）与 `docs/zh/api/compute/atomic/`、`samples/scatter_elements/`。SIMT 需 CANN ≥ 9.2.0。

## 1 真值表

| 谓词 | 真值（DSL 0.7.0 / 3510） | 依据 |
|------|---------------------------|------|
| `CAP_SIMT_GM_DIRECT` | ✅ `vf(mode="simt")` 内 Tensor 下标读写直访 GM 离散地址（`output[flat] = updates[flat]`）；每个线程独立计算地址、可走不同控制流 | scatter_elements 全部 SIMT kernel；compute.md「SIMT 模式」 |
| `CAP_SIMT_ATOMIC_ADD_DTYPE_SET` | 标量 `cb.simt.atomic_add(ptr, value)`：`{int32, uint32, float32, int64, uint64}`——**无 fp16/bf16/double**。批量累积另有 `mem_copy(..., atomic_add=True)`：`{int8, int16, int32, bf16, fp16, fp32}`，仅 UB/L0C → GM | atomic API 文档（产品支持：仅 950PR/DT）；compute.md「原子操作」 |
| `CAP_SIMT_ATOMIC_ADD_RETURN_RELIABLE_SET` | **待验证**——接口声明返回计算前原值（`-> ScalarValue`），返回值语义已文档化但 Sparse 场景实测口径未登记。验证义务：首例以单测复核返回旧值语义；此前只用写语义 | atomic_add API 文档 |
| `CAP_SIMT_ATOMIC_CAS_WIDTH` | `cb.simt.atomic_cas` 存在；支持的位宽集合**待验证**（API 分类可证类目存在，dtype/位宽表未核对）。double CAS 模拟路线就位前须复核 | compute.md 接口分类表；atomic API 文档 |
| `CAP_SYNC_ALL` | ✅ `global_sync_all()`（等价 模式2→0→0→2 四次同步，代价高，禁入内层循环）；细粒度：`vec_sync_block_arrive/wait`（模式 0 全 AIV 间）、`cube/vec_sync_intra_*`（模式 4） | synchronization.md「核间同步」 |
| `CAP_SYNC_SCHEDULE_MODE` | ✅（语义变化）DSL 无 `SetScheduleMode` 概念：`global_sync_all()` 无需 host 开关即生效；**替代惯用法是同 stream 顺序 launch 两次 kernel（stream 排序天然有序，免栅栏）** | synchronization.md；scatter_elements 两段 launch 实测 |
| `CAP_SIMT_THREAD_MAX` | 1024（纯 SIMT：`kernel[dim3(block), dim3(thread)]`，`thread.x*y*z ∈ [1,1024]`）；block 上限 65535（scatter_elements MAX_CORE_NUM 口径）；混合 SIMD+SIMT：`vf(mode="simt", thread=N)` 由 VF 参数定，host 不传 thread 维 | compute.md「Host 启动规则」 |
| `CAP_STACK_PER_THREAD` | **待验证**——已知约束：SIMT 寄存器文件总容量固定，线程数翻倍、每线程寄存器减半，溢出到 GM 栈性能劣化；每线程栈字节数未登记。验证义务：首个行级状态类算子实测 | compute.md「两个代价」 |
| `CAP_VF_PARAM_MAX` | 无硬上限文档化——kernel 形参自由（Tensor + `cb.Constexpr[int]`）；元数据数组走 **params Tensor（GM）**而非入参（scatter_elements ABI 先例）。约束：`Constexpr` 个数放大编译变体数，非接口上限 | scatter_elements；jit-arguments.md（编译缓存） |
| `CAP_UB_HYBRID_SIZE` | SIMD 口径 248 KB（`get_mem_size("ub")`=253952，**不要再减顶部预留**）；开 SIMT 后按 **216 KB** 口径（SIMT Data Cache 划走 ~32 KB，**不反映在 `get_mem_size` 里，要自己留**） | hardware-model.md 两个 danger 块 |

## 2 派生事实

- **launch 形态二选一且编译器校验**：纯 SIMT `kernel[dim3(block), dim3(thread)]`（device 侧禁写死 `vf(mode="simt", thread=N)`）；混合 `kernel[block_count]`（thread 在 VF 参数）。混用即编译失败
- **原子 dtype 双轨**：离散散射点加用 `cb.simt.atomic_add`（int64 原生支持，AscendC 需专用接口）；UB 内整块累积用 `mem_copy(..., atomic_add=True)`（fp16/bf16 走此轨，标量原子不支持）
- **DMA 初始化惯用法**：大块连续零初始化/拷贝经 `Channel(MemLoc.UB, ...)` + `mem_copy`（MTE2/MTE3），尾部不足整 tile 段用 SIMT 标量循环补（scatter_elements ScatterDmaCopyKernel 先例：tile 4096B、`min(DMA_COPY_MIN_BYTES)` 分档）
- **DCache 陷阱**：Scalar 单元经 DCache 访 GM，多核各持独立缓存——核 A Scalar 写、核 B 读同地址须显式 dcci 类操作；SIMT VF 的 GM 写由框架保证一致性
- **分核调度依赖张量内容时**：host（构图期）不可读，走 `@aicpu_kernel`（运行期 AI CPU 读 GM 出计划，stream 排序先于 AI Core kernel）——稀疏候选块筛选/变长前缀和的DSL 正解

## 3 待验证项

| 项 | 验证义务 |
|----|---------|
| `atomic_add` 返回旧值的可靠性（`CAP_SIMT_ATOMIC_ADD_RETURN_RELIABLE_SET`） | 首个依赖返回值的链式逻辑前，单测复核 |
| `atomic_cas` 位宽与 double CAS 模拟可行性 | double 累积路径设计前，API 文档 + 单测复核 |
| 每线程栈/寄存器分档量化值 | 行级状态类算子首例实测（线程数 × 寄存器档位） |
| `mem_copy(atomic_add=True)` 与标量原子混用时的可见性顺序 | 复合累积路径首例实测 |
