<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 范式 CANNBotDSL 能力映射

> 通用层 [capability-contract.md](../../generic/capability-contract.md) 谓词在 CANNBotDSL（下文简称 DSL，0.8.0，Ascend 950PR/950DT 口径）下的真值。其他 SoC 须重新核实，禁止外推。
>
> 依据：cannbot-dsl 仓 docs/zh/api/compute/scalar/ 与 docs/zh/api/execution/device/（system 类目）、docs/zh/guide/compute.md。

## 1 真值表

| 谓词 | 真值（DSL 0.8.0 / 3510） | 依据 |
|------|---------------------------|------|
| `CAP_FP_STATUS_READ` | ❌ **未暴露**——DSL kernel API 面无 FP 异常状态读取 intrinsic（scalar_compute 仅位运算/转换/bypass 访存）。get 的空实现（恒写规范零值）在 DSL 是**当前唯一形态** | docs/zh/api/compute/scalar/ 类目核对 |
| `CAP_FP_STATUS_CLEAR_API` | ❌ **未暴露**——无标志复位接口 | 同上 |
| `CAP_VECTOR_SIDE_EFFECT_CLEAR` | **待验证**——DSL 可触发向量单元活动（SIMD VF 计算），副作用通道有效性无接口保证；逐平台实测（clear→get==0 闭环），同 AscendC 负知识 | signal-model §3 义务同款 |
| `CAP_SCALAR_SCOPE_INTRINSIC` | **N/A 化**——无状态 intrinsic 可登记，不存在作用域问题；若未来暴露状态接口，登记其 VF 作用域约束 | — |
| `CAP_DEVICE_CLOCK` | ✅ `get_system_cycle()`（kernel API system 类目，性能分析用周期读取）；**SIMT VF 内可用性待验证** | hardware-model.md 速查表 |
| `CAP_ASSERT_SIGNAL_LAYOUT` | 同接口契约（布局语言中立：`[1]`/`[8]` × 接口 dtype，通过=0/失败=1）；失败载体为**状态标志输出**（`@kernel` SIMT VF 写，宿主读取后中止）——DSL kernel 无宿主日志/错误中止对应物 | signal-model §6（设计约定） |
| `CAP_PREBUILT_BINARY` | ✅ AOT / Native 打包——`@export("name")` 声明 + `cannbotdsl.aot.build(collectors=..., target=..., output=...)`；`Dim` 动态维度支持编译期泛化（对应 AscendC binary.json 的 shape[-2] 单实例语义） | aot-packaging.md |
| `CAP_STATUS_LAYOUT` | `[8]` × fp32（元素数 8、float32）——接口契约，语言中立，随 ACLNN 契约传递 | FloatStatus 三算子（AscendC 侧同值） |

## 2 派生事实

| 项 | 真值 | 出处 |
|----|------|------|
| 退化形态 | DSL **无 tiling 概念**——退化 kernel 在 DSL = host 构图期直接启动常量几何（`dim3(1)` / 最小 block），无需 AscendC 的空 TilingData/常量 TilingFunc 四步 | 本适配层 status-kernel §5 |
| 线程数 | 最小有效档：1（sleep 自旋）/ 8 元素用 ≤32；纯 SIMT thread ≤ 1024 | compute.md 启动规则 |
| 输出缓存 | SIMT VF 写 GM 后框架保证一致性；Scalar 侧 Tensor 下标写经 DCache，跨核读须显式 Cache 操作 | hardware-model.md |
| assert 语义载体 | 失败路径 = 状态标志输出（SIMT VF 写，宿主消费闭环）——与 AscendC 版同构，见 [assert-case-kernel.md](assert-case-kernel.md) §1 | signal-model §6 |

## 3 待验证项

| 项 | 验证义务 |
|----|---------|
| `get_system_cycle()` 在 SIMT VF 内的可用性与调用位置约束 | sleep 型算子设计前，API 文档 + 编译实测 |
| 向量活动副作用通道在 3510 的有效性（clear→get==0 闭环） | 逐平台实测，不信任 AscendC 侧结论平移（语言不同、指令序列不同） |
| assert 标志取值（通过/失败码）与消费方约定的接口契约登记 | assert 型算子首例接入时，随目标接口契约核对并回写本表 |
