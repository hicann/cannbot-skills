<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式能力谓词契约

> **层级**：通用知识层（generic/），语言中立。
>
> 通用层算法引用以下能力参数；真值由命中语言的适配层 capability-mapping 填写。标注"待验证"的谓词须在产出中声明假设并给出验证义务。

## 1 谓词定义

| 谓词 | 类型 | 含义 | 影响的通用层结论 |
|------|------|------|----------------|
| `CAP_SIMT_GM_DIRECT` | bool | 执行单元可持 `GM` 指针直接读写离散地址 | 数据面直访 GM 路线成立（scene §4、access-model §1） |
| `CAP_SIMT_ATOMIC_ADD_DTYPE_SET` | set | 原子加支持的 dtype 集合 | 冲突决策树的浮点/整型分支（access-model §3） |
| `CAP_SIMT_ATOMIC_ADD_RETURN_RELIABLE_SET` | set | 原子加**返回值**可靠的 dtype 集合（⊆ 上一行） | 是否可依赖返回值做链式逻辑（atomic-ops） |
| `CAP_SIMT_ATOMIC_CAS_WIDTH` | set | 原子 CAS 支持的位宽 | double 等无原生原子 dtype 的模拟路线（access-model §3） |
| `CAP_SYNC_ALL` | bool | 提供跨全部参与核的栅栏原语 | 零初始化→散射两阶段骨架（sync-model §1） |
| `CAP_SYNC_SCHEDULE_MODE` | bool | host 侧可切换同步调度模式 | 栅栏生效的前提（sync-model §3） |
| `CAP_SIMT_THREAD_MAX` | int | 单核执行单元数上限 | 线程数分档上界（splitting §4） |
| `CAP_STACK_PER_THREAD` | int | 每执行单元栈字节数 | 行级粒度局部变量的体积上限（splitting §4） |
| `CAP_VF_PARAM_MAX` | int | 执行段入口参数个数上限 | 元数据传参方式（入参 vs 共享内存，access-model §1） |
| `CAP_UB_HYBRID_SIZE` | int | 混合编程下片上共享内存可用字节 | stride 表等元数据的体积上限（access-model §1） |

## 2 使用规则

1. 通用层文档中出现的具体取值（如"平台上限 2048 线程"）只作分档示例；设计产出必须引用适配层真值
2. 适配层未登记真值的谓词按"待验证"处理：产出中标注假设 + 验证义务（验证方法：官方 API 文档、参考实现、实测）
3. 谓词以平台为单位登记：不同 SoC/架构（如 ascend950 与 910B）真值可能不同，禁止跨平台外推
