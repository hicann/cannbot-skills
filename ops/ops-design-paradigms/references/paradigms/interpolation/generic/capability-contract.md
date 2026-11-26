<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式能力谓词契约

> **层级**：通用知识层（generic/），语言中立。
>
> 通用层算法引用以下能力参数；真值由命中语言的适配层 capability-mapping 填写。标注"待验证"的谓词须在产出中声明假设并给出验证义务。

## 1 谓词定义

| 谓词 | 类型 | 含义 | 影响的通用层结论 |
|------|------|------|----------------|
| `CAP_SIMT_GM_DIRECT` | bool | 执行单元可持 `GM` 指针直接读写离散地址 | 邻居 gather 直访 GM 路线成立（scene §4、weight-model §2） |
| `CAP_FP64_AICORE` | bool | 计算单元原生支持 fp64 运算 | 累加精度路线：false 时须以补偿累加模拟 golden 的 fp64 语义（weight-model §3） |
| `CAP_FMA_CONTROL` | bool | 可阻断编译器将相邻乘加融合为 FMA（如经中间变量强制先舍入） | 坐标映射浮点序列与 golden 同序的可行性（coordinate-mapping §5） |
| `CAP_FMA_ERROR_TERM` | bool | 提供 FMA 的精确误差项提取原语（`fmaf(a,b,-a*b)` 形态） | 补偿累加 TwoProdFMA 分解的前提（weight-model §3） |
| `CAP_FP_FLOOR` | bool | 提供向负无穷取整的浮点原语 | 邻居分解 `i0 = floor(src)` 的语义基点（coordinate-mapping §3） |
| `CAP_SIMT_ATOMIC_ADD_DTYPE_SET` | set | 原子加支持的 dtype 集合 | grad 散射累积的浮点/整型分支（splitting §5） |
| `CAP_SYNC_ALL` | bool | 提供跨全部参与核的栅栏原语 | grad 零初始化→散射两阶段骨架（splitting §5） |
| `CAP_SYNC_SCHEDULE_MODE` | bool | host 侧可切换同步调度模式 | 栅栏生效的前提（splitting §5） |
| `CAP_SIMT_THREAD_MAX` | int | 单核执行单元数上限 | 线程数分档上界（splitting §4） |
| `CAP_STACK_PER_THREAD` | int | 每执行单元栈字节数 | 坐标映射局部变量的体积上限（splitting §4） |
| `CAP_VF_PARAM_MAX` | int | 执行段入口参数个数上限 | scale/shape 元数据传参方式（入参 vs 共享内存，splitting §3） |
| `CAP_UB_HYBRID_SIZE` | int | 混合编程下片上共享内存可用字节 | 快除参数表等元数据的体积上限（splitting §3） |

## 2 使用规则

1. 通用层文档中出现的具体取值（如"512 线程分档"）只作分档示例；设计产出必须引用适配层真值
2. 适配层未登记真值的谓词按"待验证"处理：产出中标注假设 + 验证义务（验证方法：官方 API 文档、参考实现、实测）
3. 谓词以平台为单位登记：不同 SoC/架构（如 ascend950 与 910B）真值可能不同，禁止跨平台外推
4. `CAP_FP64_AICORE` 为 false 不阻断设计——补偿累加路线（weight-model §3）为此准备；但产出必须声明"golden 为 fp64 语义、kernel 为模拟"的精度口径差异
