<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 断言/条件求值信号算子的设备核实现（SIMT）

> **层级**：AscendC 语言适配层。语言中立模型见 [../../generic/index.md](../../generic/index.md)；形态选型见 [../../generic/kernel-placement.md](../../generic/kernel-placement.md) §2。
>
> 覆盖信号类型：**断言判定**（assert 型）、**条件求值**（case_condition 型）。实现形态为**设备计算核（AICore）SIMT**——读微量输入 → 离散输出/状态标志，与 sleep 同款"单线程退化 kernel"形态（kernel-placement §3）。
>
> **负知识（设备核边界）**：设备核无宿主级日志/字符串/错误中止通道（仅调试 printf，生产禁用）。断言失败路径必须**信号化**——设备核写失败标志输出，上报与中止由消费方（宿主/下游节点）读取标志后执行，摘要打印在宿主侧完成。禁止在设备核内依赖任何宿主上报能力。

## 1 assert 型：断言判定 → 状态标志

语义形态：**控制面信号生产者**——条件真写通过标志（常量），条件假写失败标志（常量）。输出契约：固定 `[1]`（或对齐 `[8]` 布局）状态张量，幂等常量写（I3）。

```cpp
template <typename T>
__aicore__ inline void Process(GM_ADDR cond, GM_ADDR output)
{
    __gm__ bool* condGm = (__gm__ bool*)cond;
    __gm__ T* outGm = (__gm__ T*)output;
    asc_vf_call<AssertSignalSimt<T>>(dim3(1), condGm, outGm);   // 单线程退化 launch
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void AssertSignalSimt(
    const __gm__ bool* condGm, __gm__ T* outGm)
{
    if (condGm[0]) {
        outGm[0] = (T)0;      // ① 条件真：通过标志（常量写，短路零开销）
    } else {
        outGm[0] = (T)1;      // ② 条件假：失败标志，消费方读取后中止/上报
    }
}
```

要点：

- **条件真短路**：成功路径无打印、无额外分支副作用（与原语义"条件真零开销"对齐）
- **失败路径 = 标志 + 消费闭环（I4）**：验证断言双用例——条件假 `out == 失败码`、条件真 `out == 0`；消费方行为（读标志后中止调用链）在设计文档中向调用方声明
- **摘要打印移宿主侧**：失败详情（输入数据摘要、`summarize` 截断约束）由宿主读取输入张量生成，设备核不承担
- **幂等合法（I3）**：blockDim=coreNum 时全核同值写，核数只影响冗余度
- **值依赖输入**：条件标量经 `TilingInputsDataDependency` / infershape `InputsDataDependency` 登记（[../../../common/value-depend-process.md](../../../common/value-depend-process.md)）
- **退化 tiling 契约**：与 [status-kernel.md](status-kernel.md) §6 同款（空 TilingData + 四步 TilingFunc + 单 key 集）

## 2 case_condition 型：条件求值 → 离散编号

```cpp
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void CaseConditionSimt(
    const __gm__ T* inGm, __gm__ int32_t* outGm)
{
    const T i = inGm[0], j = inGm[1], k = inGm[2];
    if (i >= k && j < k)       { outGm[0] = 0; }
    else if (i == k && j == k) { outGm[0] = 1; }
    // ...                        // 离散编号枚举（互斥分支链）
    else                       { outGm[0] = CASE_DEFAULT; }  // else 兜底
}
```

要点：

- **分支链互斥完备**（else 兜底）：任何输入落进唯一编号——真值表驱动验证（[../../generic/verification.md](../../generic/verification.md) §2）
- **单线程（LAUNCH_BOUND(1)）**：三元素读 + 分支 + 单点写，无并行收益（同 sleep 自旋形态）
- **设备核路线保证运行期真值**：构图期不可读的输入值由 kernel 运行期读取；tiling/infershape 若需该值须走值依赖登记

## 3 与状态类设备核的形态对齐

| 要素 | assert 型 | case_condition 型 | FloatStatus get |
|------|-----------|--------------------|-----------------|
| 输入 | 标量条件（值依赖） | `[3]` 标量（值依赖） | 无 |
| 输出 | `[1]` 通过/失败标志 | `[1]` int32 编号 | `[8]` 快照/规范零值 |
| 线程 | 1（LAUNCH_BOUND(1)） | 1 | 32（8 元素最小档） |
| 写语义 | 幂等常量（I3） | 幂等（输入确定 → 编号确定） | 幂等常量 |
| tiling | 退化四步（status-kernel §6） | 同 | 同 |
| 失败上报 | 标志 + 宿主消费闭环（无设备核直报通道） | 无失败语义 | 无（写规范值，不失败） |
