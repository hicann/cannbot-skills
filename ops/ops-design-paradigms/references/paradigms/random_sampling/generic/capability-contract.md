<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 范式能力谓词契约

> **层级**：通用知识层（generic/），语言中立。
>
> 通用层算法引用以下能力参数；真值由命中语言的适配层 capability-mapping 填写。标注"待验证"的谓词须在产出中声明假设并给出验证义务。

## 1 谓词定义

| 谓词 | 类型 | 含义 | 影响的通用层结论 |
|------|------|------|----------------|
| `CAP_PRNG_ALG` | enum | 流代数 step 的具体算法（Philox4x32-10 等）与步产出 k | prng-model §1/§2、distribution-transform §1 |
| `CAP_PRNG_VECTORIZED` | bool | 流推进可向量化批量执行 | 向量路线成立（splitting §1 vs §2 选型） |
| `CAP_THREAD_LAUNCH` | bool/int | 可启动标量线程组（线程组规模上限） | 离散/连续布局路线（splitting §2） |
| `CAP_VEC_IN_SIMT` / `CAP_SIMT_IN_VEC` | bool | 向量核内嵌标量线程调用 / 反之 | 混合路线（变换段内嵌，simd-path） |
| `CAP_INDEX_BIT_WIDTH` | int | 线性索引/字节偏移的安全位宽 | 块协议触发条件（splitting §3） |
| `CAP_SPLIT_BLOCK_MAX` | int | 块协议的块数上限 M | 递归二分熔断（splitting §3） |
| `CAP_STATE_WRITE_BACK` | bool | 提供带缓存清理的唯一写回原语 | offset 递进协议（state-management §3） |
| `CAP_CORE_SYNC_ALL` | bool | 全部执行单元栅栏 | 回写时序约束（state-management §3） |
| `CAP_FAST_DIV` | bool | 魔数+移位快除 | 布局公式的性能前提（splitting §2） |
| `CAP_LOCAL_MEM_HYBRID_SIZE` | int | 混合编程下片上可用容量（扣减后） | 缓冲预算（splitting §1） |

## 2 使用规则

1. 通用层文档中的具体取值（k=4、R=256、M=8、L_min=256 等）只作分档示例；设计产出必须引用适配层真值
2. 适配层未登记真值的谓词按"待验证"处理：产出中标注假设 + 验证义务（验证方法：官方 API 文档、random_common 参考实现、实测）
3. 谓词以平台为单位登记，不同 SoC/架构禁止外推
4. `CAP_PRNG_ALG` 换型（如 Philox→ThreeFry）时，prng-model 的常数（轮数、k）与预留 R 下界须重新论证
