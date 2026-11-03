<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling torch 对标 SIMT 路线

> **层级**：AscendC 语言适配层。对标 PyTorch 接口族（seed/offset 张量、self 填充语义）的标量线程实现路线；也是变长消耗（拒绝式）算子的标准路线。
>
> 基准算子：sim_thread_exponential（最小样例，kernel 73 行）、stateless_truncated_normal_v2（变长消耗）、stateless_randperm、stateless_sample_multinomial、drop_out_v3。
> **适用域硬边界（R20）**：本路线适用于 torch 对标（B/D 族：seed/offset 张量、self 填充）接口族；跨族引用即范式违规——按其他族的接口套用本路线形态、或以本路线参照算子为其他族实现背书，均违反路线选型机械匹配（patterns 场景判定·第二步）。


## 1 路线特征

| 维度 | 形态 |
|------|------|
| 执行体 | `__simt_vf__` 线程组（`VF_CALL` / `asc_vf_call` 启动），线程常量 512（`DEFAULT_SIMT_THREAD_NUM`） |
| 随机数生成 | `PhiloxRandomSimt`：每线程本地 key/counter 数组，10 轮 Philox4x32；uint2 乘法优化（比 uint32 拆写编译产物更优） |
| 数据通路 | GM 直写（`__gm__ volatile T*`），无 TQue 流水；变换即写 |
| 线程映射 | 模板内置（连续/离散双布局 + 魔数快除），算子无感 |

## 2 双模板选型

| 模板 | 布局 | 参数 | 适用 |
|------|------|------|------|
| `PhiloxSimtKernelContinuous<T, Transform, ThreadNum, Vec>` | 连续 | Vec=4/8/16 | 每输出消耗多随机数（c>1）、整组消费后写连续输出段 |
| `PhiloxSimtKernelDiscontinuous<T, Transform, ThreadNum, UNROLL>` | 离散 | UNROLL=4 | c=1、输出散布写；展开隐藏流推进延迟 |

模板内部职责（算子侧禁止越权，见 [common-template.md](common-template.md) §3）：

- `PhiloxAlgParsInit(key, counter, seed, offset)`：offset→counter 起点换算（`⌈offset/4⌉`）
- `ThreadMappingAndSkip<Vec, MODE>(i, counterTmp, magic, shift, totalThreads)`：输出索引 → 流位置（连续/离散两公式，prng-model §3）+ Skip
- 每轮 `PhiloxRandomSimt` 后把 results[4] 分发给 TransformFunc 的 U 个 iStep

## 3 魔数快除

`i / totalThreads` 用 `Simt::UintDiv(i, magic, shift)` 替代（host 侧 `GetUintDivMagicAndShift` 预计算）；线程映射每元素一次除法，快除是本路线的性能前提。

## 4 变长消耗（拒绝式）实现（stateless_truncated_normal_v2 实例）

```cpp
// 预留律落地：RESERVED_SAMPLES_PER_OUTPUT = 256
// 每线程的流位置 = counter₀ + groupIndex × R（组内第 (i mod B) 路）
// 变换内：
//   u → Box-Muller → z；|z| < 2.0 接受并写出
//   否则本配额内取下一组 u 重试（重试上限 = R/c，溢出即重样 bug）
```

- key/counter 的 TF V2 接口映射：`key_data[0] → key[0..1]`、`counter_data[0] → counter[0..1]`（低 64 位）、`counter_data[1] → counter[2..3]`（高 64 位）——**接口位拆分须逐位核对**，错位不报错但序列完全错误
- FPGA 平台线程数降档（128）的编译分支先例：`#ifdef __DAV_FPGA__`

## 5 块协议数据流

`enableSplitBlocks=true` 时 tiling 产出 `splitBlocks[≤8]`（每块 numel/gmOffset/grid/totalThreads/kernelOffset），kernel 侧：

```cpp
Process(self, tilingData):
    if (GetBlockIdx() >= usedCoreNum) return;
    MyLauncher launcher(seed, params, self);
    ProcessWithSplitBlocks(tilingData, launcher);   // 逐块：VF_CALL（块流起点 = kernelOffset_b）
```

- 块参数由 host 的 `CalcExecutionPoliciesForBlocks` 推导（grid 按 256 线程组对齐、上限 78 核×8 组；offset 累进见 prng-model §5）
- **kernel 不做块间同步**——块互不重叠（GM 偏移 + 流区间双重隔离）

## 6 离散采样与排列（特殊族）

| 算子 | 形态 | 要点 |
|------|------|------|
| stateless_sample_multinomial | SIMT + 前缀和查找 | 概率归一化 → 逆 CDF；无放回时分布修改 |
| stateless_randperm | 生成 + 排序两段（random.h 284 行 + sort.h 151 行） | 随机键生成（本范式）→ 键排序（SortSelect 手段） |
| drop_out_v3 | SIMT + keepProb 字段 | 掩码生成族：`u < p` 直接写掩码/填充值 |

这些算子超出双模板覆盖范围的部分（排序、前缀和）以**手段**身份借用相邻范式知识，随机流管理仍按本范式。
