<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 范式 AscendC 能力映射

> 通用层 [capability-contract.md](../../generic/capability-contract.md) 谓词在 AscendC（ascend950 / arch35，random_common 实测口径）下的真值。其他 SoC 须重新核实，禁止外推。

## 1 真值表

| 谓词 | 真值（ascend950 / arch35） | 依据 |
|------|---------------------------|------|
| `CAP_PRNG_ALG` | Philox4x32-10（轮数 10、步产出 k=4）；SIMD 侧 `PhiloxRandom<10>`、SIMT 侧 `PhiloxRandomSimt` | random_common 双实现 |
| `CAP_PRNG_VECTORIZED` | ✅ 向量核批量流推进（寄存器级，含 uint32→float/half/bf16 尾数拼接） | U32Conversion 三实现 |
| `CAP_THREAD_LAUNCH` | ✅ `VF_CALL`/`asc_vf_call`，线程组上限 2048；随机算子惯例 512（DEFAULT_SIMT_THREAD_NUM），混合内嵌档 1024 | sim_thread_exponential / random_standard_normal_v2 |
| `CAP_SIMT_IN_VEC` | ✅ 向量核内 `asc_vf_call<...>(dim3{n}, __ubuf__ 指针, count)`（混合形态） | random_standard_normal_v2 |
| `CAP_INDEX_BIT_WIDTH` | 32bit（`numel ≤ 2^31−1` 且最大字节偏移 ≤ 2^31−1，双条件） | TensorSliceState::Is32bitIndexable |
| `CAP_SPLIT_BLOCK_MAX` | 8（MAX_SPLIT_BLOCKS，超限报错） | CalcSplitBlocks |
| `CAP_STATE_WRITE_BACK` | ✅ `GlobalTensor::SetValue` + `DataCacheCleanAndInvalid<..., CACHELINE_ATOMIC/OUT>` | random_uniform_v2 / random_standard_normal_v2 |
| `CAP_CORE_SYNC_ALL` | ✅ `SyncAll()`（tiling 需 `SetScheduleMode(1)`，经 `isNeedSyncAll` 配置） | random_tiling_arch35 WriteBack |
| `CAP_FAST_DIV` | ✅ `Simt::UintDiv` + host 侧 `GetUintDivMagicAndShift` | ThreadMappingAndSkip |
| `CAP_LOCAL_MEM_HYBRID_SIZE` | 混合场景片上可用 = UB − DcacheSize（`SetLocalMemorySize` 下发）；非混合全量 | DoUbTiling / GetPlatformInfo |

## 2 派生事实（分档示例的真值来源）

| 通用层符号 | 真值 | 出处 |
|-----------|------|------|
| a（单元对齐粒度） | 4（coreAlignSize 默认） | OpTilingConfig |
| L_min（每单元下限） | 256（MIN_CORE_PRO） | random_tiling_arch35.cpp |
| U（展开因子） | 4（unrollFactor 默认） | OpTilingConfig |
| R（每输出预留） | 256（K_Reserved_Per_Output / RESERVED_SAMPLES_PER_OUTPUT）；rrelu_with_noise 型调用契约推进量 12/调用 | random_uniform_v2 / truncated_normal_v2 / rrelu_with_noise |
| M（块上限） | 8 | MAX_SPLIT_BLOCKS |
| 块协议线程组 | 256/组，每核 8 组，核上限 78（AIC_CLUSTER_COUNT） | CalcExecutionPoliciesForBlocks |
| **Philox 轮数** | 流基准 10 轮；**`PhiloxRandom` 模板默认 7 轮（陷阱）**——必须显式 `<10>` | philox.h / rrelu_with_noise 实录 |
| **count 形参位宽** | `PhiloxRandom` count 为 uint16 → 单次批量产出 < 65535（超限静默截断） | rrelu_with_noise 实录 |
| **流字段布局** | 接口族三型（原生 seed/offset → offset 入 counter[2:3] subsequence；TF 直给 → counter_data 低/高 64 位；seed 属性族） | prng-model §1.1 |
| 常量池 | `RAND_2POW32_INV = 2.3283064e-10`、`+RAND_2POW32_INV_HALF`（**乘法+半步口径，值域 (0,1)**——SIMT 模板族实存；与位重解释 [0,1) 同流不同值，按 R7 路径一致性选用） | random_kernel_base.h |
| Philox 常数 | W32_A=0x9E3779B9、W32_B=0xBB67AE85、M4X32_A=0xD2511F53、M4X32_B=0xCD9E8D57 | 同上 |

## 3 待验证项

| 项 | 验证义务 |
|----|---------|
| A2/A3 等 SoC 的 `CAP_THREAD_LAUNCH`/`CAP_SIMT_IN_VEC`（sim_thread_exponential README 声称支持 A3/A2，混合形态未逐平台实测） | 迁移平台时以官方文档+编译实测复核 |
| FPGA 线程降档（128）与其他平台的档位关系 | 仅 arch35 FPGA 分支先例，外推前实测 |
| `SetCtrlSpr<60,60>` 半精度旁路在其他架构的行为 | 仅 fp16/bf16 转换路径依赖，迁移时核对 |
