<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# SlidingWindow 范式能力谓词契约

> **层级**：通用知识层（generic/），语言中立。
>
> 通用层算法引用以下能力参数；真值由命中语言的适配层 capability-mapping 填写。标注"待验证"的谓词须在产出中声明假设并给出验证义务。

## 1 谓词定义

| 谓词 | 类型 | 含义 | 影响的通用层结论 |
|------|------|------|----------------|
| `CAP_SIMT_GM_DIRECT` | bool | 执行单元可持 `GM` 指针直接读写离散地址 | 窗口元素离散读、梯度直写路线成立（scene §4、splitting §2） |
| `CAP_FP_NEG_INF` | bool | 可获得 fp32 负无穷哨兵（位模式或常量） | max 聚合初始化（window-math §3） |
| `CAP_FP_MAX_NAN` | bool | 标量比较 + isnan 组合可实现 NaN 优先传播的 max 更新 | max 族 NaN 语义（window-math §3） |
| `CAP_INT_DIV_COST` | enum | 整除/取模实现的代价档位（直除代价 / 快除可行） | 坐标分解路线选择（splitting §2） |
| `CAP_SIMT_ATOMIC_ADD_DTYPE_SET` | set | 原子加支持的 dtype 集合 | 分支 C 散射累积的浮点/整型分支（grad-routing §5） |
| `CAP_SYNC_ALL` | bool | 提供跨全部参与核的栅栏原语 | 分支 C 零初始化→散射两阶段骨架（grad-routing §5） |
| `CAP_SYNC_SCHEDULE_MODE` | bool | host 侧可切换同步调度模式 | 栅栏生效的前提（grad-routing §5） |
| `CAP_SIMT_THREAD_MAX` | int | 单核执行单元数上限 | 线程分档上界（splitting §4） |
| `CAP_STACK_PER_THREAD` | int | 每执行单元栈字节数 | 窗口循环局部变量体积上限（splitting §4） |
| `CAP_VF_PARAM_MAX` | int | 执行段入口参数个数上限 | 窗口参数标量入参的预算（splitting §5） |
| `CAP_UB_HYBRID_SIZE` | int | 混合编程下片上共享内存可用字节 | 快除参数表体积上限（splitting §5） |

## 2 使用规则

1. 通用层文档中出现的具体取值（如"正向 256 / 反向 1024 分档"）只作分档示例；设计产出必须引用适配层真值
2. 适配层未登记真值的谓词按"待验证"处理：产出中标注假设 + 验证义务（验证方法：官方 API 文档、参考实现、实测）
3. 谓词以平台为单位登记：不同 SoC/架构（如 ascend950 与 910B）真值可能不同，禁止跨平台外推
4. `CAP_INT_DIV_COST` 按实测档位登记（直除可接受 / 须快除），登记时附参照算子与 shape 规模
