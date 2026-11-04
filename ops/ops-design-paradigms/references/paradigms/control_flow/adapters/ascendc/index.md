<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 范式 AscendC 语言适配层

> **层级**：AscendC 语言适配层（adapters/ascendc/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。
>
> 本适配层覆盖**状态驱动控制信号算子**（B 类）的设备计算核（AICore）实现路线：SIMT 为主，SIMD/混合按信号类型选型（断言失败路径信号化，无宿主上报依赖）。图级编排算子（If/While/Switch/Merge 等，无 kernel）不属本范式，见 [../../generic/scene.md](../../generic/scene.md) §5。

## 1 定位

本目录承载 ControlFlow 范式中强耦合 AscendC 的知识：信号算子的设备核（AICore）形态选型（SIMT/SIMD/混合）、副作用工程、降级兼容契约与陷阱。

```
通用信号模型（generic/）× AscendC 实现细节（本目录）→ 信号算子设计与代码产出
```

## 2 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| 第一决策：形态选型 | [../../generic/kernel-placement.md](../../generic/kernel-placement.md) | 决策树（输出契约 / 副作用触达性）+ 设备核能力边界（负知识） |
| 设备核路线 | [status-kernel.md](status-kernel.md) | alloc 写零+DCCI、get 空实现+intrinsic 作用域+预编译二进制、clear 副作用工程、sleep 时钟自旋 |
| 断言/条件求值路线 | [assert-case-kernel.md](assert-case-kernel.md) | assert 失败路径信号化（标志 + 消费闭环）、case_condition 离散求值、与状态类形态对齐 |
| 能力真值 | [capability-mapping.md](capability-mapping.md) | 谓词真值（含降级态登记）、派生事实、待验证项 |
| 陷阱与定位 | [pitfalls.md](pitfalls.md) | intrinsic 作用域、副作用闭环、幂等、降级误修、二进制漂移等错题集 |

## 3 形态选型表（机械 if-then，CF-R1 依据）

与 [../../generic/kernel-placement.md](../../generic/kernel-placement.md) §2 决策树逐行对齐（通用层判据 → 本表形态落点）。**全部信号类型落设备计算核（AICore）**：

| 信号操作（通用层判据） | 形态 | 禁止项 |
|------|------|--------|
| 标量条件的断言判定（失败上报/中止语义） | **SIMT 单线程**写通过/失败标志（assert 类），消费方读标志闭环 | 禁止依赖设备核宿主日志/错误中止通道（仅调试 printf，生产禁用）——失败路径必须信号化 |
| 小规模输入条件求值 | **SIMT 单线程**写离散编号（case_condition 类） | 分支链禁不互斥/不兜底 |
| 硬件状态的设备侧读/清（低时延） | **SIMT 幂等写 / 混合**（FloatStatus 类） | 状态类 intrinsic 禁入线程函数（作用域铁律，status-kernel §2） |
| 借道计算单元活动的副作用清标志 | **混合形态**（向量触发 + SIMT 写零） | 副作用通道无接口保证，逐平台实测（I4） |
| 设备时钟（时序控制） | **SIMT 单线程自旋**（sleep 类） | 单执行体承担；验证只做下界断言 |
| 图级编排（If/While/Switch/Merge 等 `REG_OP+.GRAPH` 无 kernel） | **不属本范式**（scene §5 边界） | 禁入本范式设计流程 |

选型按信号类型机械匹配（无裁量空间）；偏离本表形态即范式违规。

## 4 基准算子表

| 算子 | 路线 | 信号类型 | 关键特征 |
|------|------|---------|---------|
| npu_alloc_float_status | 设备核（SIMD） | 状态分配 | `[8]` 写零 + DCCI |
| npu_get_float_status | 设备核（SIMT）+ 二进制 | 状态读取 | 950 空实现、intrinsic 作用域、binary.json |
| npu_clear_float_status | 设备核（混合） | 状态清除 | 副作用工程（Duplicate×6 触发向量单元）+ SIMT 写零 |
| sleep | 设备核（SIMT） | 时序 | `clock()` 自旋、单线程 |
| assert | 设备核（SIMT） | 断言 | 失败路径信号化（标志 + 消费闭环）、条件真短路 |
| case_condition | 设备核（SIMT） | 条件求值 | i/j/k→case 编号、真值表验证 |
| update_tensor_desc | 设备核 | 描述符更新 | 运行时 shape/format 控制（与 DynamicShape 交界） |

FloatStatus 家族源仓为开源算子仓 [cann/ops-nn](https://gitcode.com/cann/ops-nn) 的 `control/` 目录（get 已整包内嵌，见下方内嵌样例包）；assert / case_condition / sleep / update_tensor_desc 为**语义基准**（源仓实现非 regbase AICore 形态，无编码参照价值，仅作信号语义锚点）。

**内嵌样例包**：npu_get_float_status 已整包镜像至 [example/npu_get_float_status_package/](example/npu_get_float_status_package/)（op_graph/op_host/op_kernel/tests 完整结构，含 `binary.json` 预编译注册形态；provenance 见 [example/README.md](example/README.md)）——编码前对照源码组织方式的首选入口。

## 5 公共知识依赖

| 主题 | 位置 |
|------|------|
| 值依赖处理 | [../../../common/value-depend-process.md](../../../common/value-depend-process.md) |
| 同步与一致性 | [../../../common/sync-and-consistency.md](../../../common/sync-and-consistency.md) |

## 6 消费规则

1. 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载其他语言适配层
2. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题
3. 能力谓词以 [capability-mapping.md](capability-mapping.md) 为真值；**降级态与"逐平台实测"项必须原样传递到设计产出**（含验证义务）
4. 编码前对照 §3 基准算子核实形态；参考实现中的副作用通道参数须按目标平台重新论证
5. 本范式算子**无 tiling 设计空间**（退化 kernel，见 [../../generic/kernel-placement.md](../../generic/kernel-placement.md) §3）——不要套用常规 tiling 方法论
