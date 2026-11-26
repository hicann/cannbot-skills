<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式 CANNBotDSL 能力映射

> 通用层 [capability-contract.md](../../generic/capability-contract.md) 谓词在 CANNBotDSL（下文简称 DSL，0.8.0，Ascend 950PR/950DT / NPU ARCH 3510 口径）下的真值。其他 SoC 须重新核实，禁止外推。
>
> 依据：cannbot-dsl 仓 `docs/zh/api/simt/math/`（floor / fma / fmin / fmax / isfinite）、`docs/zh/guide/`（hardware-model / compute / synchronization）与 `samples/scatter_elements/`。SIMT 需 CANN ≥ 9.2.0（cannbot-dsl 仓配套 9.2.1）。

## 1 真值表

| 谓词 | 真值（DSL 0.8.0 / 3510） | 依据 |
|------|---------------------------|------|
| `CAP_SIMT_GM_DIRECT` | ✅ `vf(mode="simt")` 内 Tensor 下标读写直访 GM 离散地址；每个线程独立计算地址、可走不同控制流 | scatter_elements 全部 SIMT kernel；compute.md「SIMT 模式」 |
| `CAP_FP64_AICORE` | ❌ AICore 无 fp64 运算：SIMT 数学 API（floor / fma / 算术）输入域仅文档化 fp16 / bf16 / fp32——`dtypes.float64` 只存在于 TensorSpec 枚举，不代表 compute 可用。fp64 golden 须走补偿累加模拟（sampling-ops §4） | simt math API 文档；与 AscendC 侧同芯片事实 |
| `CAP_FMA_CONTROL` | **待验证**——DSL 无 `volatile`；`w0*f0 + w1*f1 + w2*f2` 表达式是否被收缩为 FMA 未实测。候选手段：ScalarValue 中间绑定 / 显式分步表达式。验证义务：首例以恒定 ULP 偏差用例（issue #35 类）实测判定，未判定前禁写单表达式 | AscendC 侧 ccec 收缩先例（issue #35）；DSL 代码生成路径未登记 |
| `CAP_FMA_ERROR_TERM` | ✅ `simt.fma(x, y, z)` 单次舍入语义——`simt.fma(h, q, -th)` 即 TwoProdFMA 精确误差项 | fma.md「功能说明」（中间乘积不单独舍入） |
| `CAP_FP_FLOOR` | ✅ `simt.floor(x)` 向负无穷取整、返回浮点值（fp16/bf16/fp32；NaN/±inf 语义已文档化）；配套 `simt.fmin` / `simt.fmax` / `simt.isfinite` | floor.md / fmin.md / fmax.md / isfinite.md |
| `CAP_SIMT_ATOMIC_ADD_DTYPE_SET` | 标量 `cb.simt.atomic_add(ptr, value)`：`{int32, uint32, float32, int64, uint64}`——无 fp16/bf16/double；批量 `mem_copy(..., atomic_add=True)` 另轨 | Sparse 范式同平台登记（[../../../sparse/adapters/dsl/capability-mapping.md](../../../sparse/adapters/dsl/capability-mapping.md)）；本范式 grad 散射沿用 |
| `CAP_SYNC_ALL` | ✅ `global_sync_all()`（等价 模式2→0→0→2 四次同步，代价高，禁入内层循环；0.8 新增 `flag_ids` 参数——默认占 flag id 0/1/2，与手写 block 同步协议共存时须避开或显式指定） | synchronization.md「核间同步」 |
| `CAP_SYNC_SCHEDULE_MODE` | ✅（语义变化）DSL 无 `SetScheduleMode` 概念：替代惯用法是同 stream 顺序 launch 两次 kernel（stream 排序天然有序，免栅栏）——grad 零初始化 + 散射的标准形态 | synchronization.md；scatter_elements 两段 launch 实测 |
| `CAP_SIMT_THREAD_MAX` | 1024（纯 SIMT：`kernel[dim3(block), dim3(thread)]`，`thread.x*y*z ∈ [1,1024]`）；block 上限 65535；混合 SIMD+SIMT：`vf(mode="simt", thread=N)` 由 VF 参数定 | compute.md「Host 启动规则」 |
| `CAP_STACK_PER_THREAD` | **待验证**——已知 SIMT 寄存器文件总容量固定，线程数翻倍、每线程寄存器减半；补偿累加路线的 df32 高低位字开销未量化。验证义务：首个启用补偿路线的算子实测 | compute.md「两个代价」 |
| `CAP_VF_PARAM_MAX` | 无硬上限文档化——kernel 形参自由（Tensor + `cb.Constexpr[int]`）；元数据数组走 params Tensor（GM）；`Constexpr` 个数放大编译变体数 | scatter_elements ABI 先例；jit-arguments.md |
| `CAP_UB_HYBRID_SIZE` | SIMD 口径 248 KB（`get_mem_size("ub")`=253952，不要再减顶部预留）；开 SIMT 后按 **216 KB** 口径（SIMT Data Cache 划走 ~32 KB，不反映在 `get_mem_size` 里，要自己留） | hardware-model.md 两个 danger 块 |

## 2 派生事实

- **launch 形态二选一且编译器校验**：纯 SIMT `kernel[dim3(block), dim3(thread)]`（device 侧禁写死 `vf(mode="simt", thread=N)`）；混合 `kernel[block_count]`（thread 在 VF 参数）。混用即编译失败
- **坐标分解的整除/取模**：DSL 无 AscendC 的 `UintDiv` magic/shift 函数族——`//` 与 `%` 直接表达（生成路径的除法开销未登记，性能敏感场景待实测；本范式正向坐标分解每元素 O(rank) 次）
- **对齐模式分派**：host 侧 Python 路径选择（half_pixel / align_corners 各自独立编译实例）或 `cb.Constexpr[bool]` 编译期分支——对齐 AscendC TilingKey 场景维度的 DSL 对等物
- **原子 dtype 双轨**（grad 散射）：离散点加 `cb.simt.atomic_add`（int64 原生支持）；fp16/bf16 输出域升 fp32 原子或批量 `mem_copy(atomic_add=True)` 轨
- **值依赖框坐标 / crop_size**：host 构图期不可读 GM——AI CPU metadata 路线或入口预解析（kernel-entrance §5）

## 3 待验证项

| 项 | 验证义务 |
|----|---------|
| `CAP_FMA_CONTROL`（表达式收缩行为） | 首例以 issue #35 类恒定 ULP 偏差用例实测；未判定前加权和一律分步中间绑定 |
| `simt.fma(h, q, -th)` 误差项提取的逐位精度 | 首个启用补偿路线的算子以 fp64 golden 对拍复核 |
| 整除/取模生成路径的性能档位 | 坐标分解热点算子（totalOut 大）实测；必要时手工 magic/shift（host 预计算入 params Tensor） |
| 每线程寄存器/栈分档量化值 | 补偿路线（df32 高低位字）首例实测 |
