---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "C/V 融合计算总参考:Init 与 Process"
description: "C/V 融合计算总参考：Init 与 Process 本文档适用于 C/V 融合算子，或设备侧存在多个 Scope、多个计算阶段、AIC/AIV 协同的 AscendC 实现。 它不是用来替代 vector 或 cube 详细参考，而是给出融合场景下的组合阅读顺序与协同组织方式。 概览与判断规则见 .claude/references/dsl-to-ascendc/references/dsl2A"
confidence: single_run
original_id: doc/target/ascendc/fa_class/cv_lowering.md
timestamp_inferred: true
tags: [fa-class, cv-fusion, aic-aiv, lowering, reference, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## C/V 融合计算总参考：Init 与 Process

本文档适用于 C/V 融合算子，或设备侧存在多个 `Scope`、多个计算阶段、AIC/AIV 协同的 AscendC 实现。
它不是用来替代 `vector` 或 `cube` 详细参考，而是给出融合场景下的组合阅读顺序与协同组织方式。
概览与判断规则见 `.claude/references/dsl-to-ascendc/references/dsl2Ascendc.md`。

---

## 第三章：Kernel 入口（C/V 融合总览）

### 1. 阅读顺序

C/V 融合算子先看本文，再分别阅读以下文档的第四章
- `.claude/references/dsl-to-ascendc/references/dsl2Ascendc_compute_cube.md`的`## 第四章`
- `.claude/references/dsl-to-ascendc/references/dsl2Ascendc_compute_vector.md`的`## 第四章`

### 2. kernel 入口形态

C/V 融合算子的入口通常同时接收输入、输出、workspace 和 tiling：

```cpp
extern "C" __global__ __aicore__ void kernel_custom(GM_ADDR ...inputs..., GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    AscendC::TPipe pipe;
    KernelClass kernel;
    kernel.Init(..., workspace, tiling, &pipe);
    kernel.Process();
}
```

### 3. `vec_num` 与 block 组成

| DSL `vec_num` | KERNEL_TYPE | 每个 block 组成 | GetSubBlockNum() (AIC) | GetSubBlockNum() (AIV) |
|:---|:---|:---|:---|:---|
| 1 | `KERNEL_TYPE_MIX_AIC_1_1` | 1 AIC + 1 AIV | 1 | 1 |
| 2 | `KERNEL_TYPE_MIX_AIC_1_2` | 1 AIC + 2 AIV | 1 | 2 |

> **⚠️ Attention 类 CV 融合算子必须使用 `KERNEL_TYPE_MIX_AIC_1_2`**
>
> Attention（FlashAttention / PageAttention 等）类 CV 融合算子必须使用 `KERNEL_TYPE_MIX_AIC_1_2`（`vec_num = 2`）。
> `KERNEL_TYPE_MIX_AIC_1_1` 仅启动 1 个 Vector 核，Vector 侧并行度严重不足，Attention 场景下性能差距巨大，**禁止使用**。

### 4. Host 侧入口函数（`_do` 函数）

核函数定义之后，必须提供 Host 侧入口函数供 pybind11 调用。该函数位于 `#ifndef ASCENDC_CPU_DEBUG` 保护块内。

#### 命名规则

| 场景 | 核函数名 | Host 入口函数名 |
|:---|:---|:---|
| 单一 dtype | `<op_name>_custom` | `<op_name>_custom_do` |
| 多 dtype 分发 | `<op_name>_custom_<dtype>` | `<op_name>_custom_<dtype>_do` |

其中 `<op_name>` 来自 `proto.yaml` 中的 `op_name`，`<dtype>` 使用小写短名（如 `fp16`、`bf16`）。


```cpp
// Host 侧入口函数
#ifndef ASCENDC_CPU_DEBUG
extern "C" void flash_attention_custom_fp16_do(
    uint32_t blockDim, void *stream,
    uint8_t *q, uint8_t *k, uint8_t *v,
    uint8_t *output, uint8_t *workspace, uint8_t *tiling)
{
    flash_attention_custom_fp16<<<blockDim, nullptr, stream>>>(q, k, v, output, workspace, tiling);
}
#endif
```

#### 要点

- `_do` 函数的参数列表与核函数一一对应，但 GM_ADDR 变为 `uint8_t *`，并在头部增加 `uint32_t blockDim` 和 `void *stream`。
- 调用语法为 `kernel_name<<<blockDim, nullptr, stream>>>(...)`。
- `#ifndef ASCENDC_CPU_DEBUG` 保护块确保 CPU 调试模式下不会编译该函数。
- 如果支持多 dtype，每种 dtype 对应一对独立的核函数和 `_do` 函数，放在独立的 `.cpp` 文件中（如 `kernel_fp16.cpp`、`kernel_bf16.cpp`），pybind11 中根据输入 dtype 分发调用。

---

## 第四章：主 Kernel 类（C/V 融合）

参考 `matmul_leakyrelu/kernel/matmul_leakyrelu.h`，C/V 融合主 `Kernel` 类建议按 `Init()` 和 `Process()` 两个大阶段组织。

### 1. `Init()`：接收 tiling 字段并初始化 GM / workspace / 子模块

`Init()` 主要负责：

- 读取并保存 tiling 字段
- 绑定输入 / 输出 GM tensor
- 初始化调度器与 workspace
- 分别初始化 Cube 子模块和 Vector 子模块

#### A. tiling 字段、GM 绑定与调度

常见模式：

- `CopyTiling(&tiling_, tilingGM)` —— **必须**用 byte-copy helper（见下硬约束），不可 struct copy-assign
- `SetGlobalBuffer(...)` 绑定 A/B/C 等 GM tensor
- 根据 `GetBlockIdx()`、`GetSubBlockNum()` 派生 `coreIdx`
- 初始化调度器，如 `sched_.Init(...)`

> **⚠️ CANN 9.0.0 tiling-load 硬约束（A3 build-FAIL round-2 教训 2026-05-27）**：
> tiling blob 从 GM 读到 local struct **必须用 `CopyTiling` byte-copy helper**（int32 逐字段拷贝）。
> **绝不可** `t_ = *reinterpret_cast<__gm__ T*>(tilingGm);`（`__gm__` 指针解引用 + struct copy-assign）——
> CCE 编不过：`error: no viable overloaded '='`（`__gm__` address-space struct 不能 copy-assign 到 Local Memory）。
> 照抄 cv-agent verified `kernel_common.h` 的 helper（A3 编过+跑过）：
> ```cpp
> template <typename T>
> __aicore__ inline void CopyTiling(T *tiling, GM_ADDR tilingGM) {
>     int32_t *ptr = reinterpret_cast<int32_t *>(tiling);
>     auto tiling32 = reinterpret_cast<__gm__ int32_t *>(tilingGM);
>     for (size_t i = 0; i < sizeof(T) / sizeof(int32_t); ++i, ++ptr) {
>         *ptr = *(tiling32 + i);
>     }
> }
> ```
> 在 `Init()`：`CopyTiling(&tiling_, tilingGm);`（helper 放进 `kernel_common.h`）。

#### B. workspace 与跨核协同

如果 C/V 之间通过 workspace 传递中间结果，通常在 `Init()` 中完成：

- workspace 基址和每个 core 的偏移计算
- ring buffer / `WorkspaceQueue` 初始化
- C/V 协同所需 flag 或队列的初始化

若存在跨核同步或 producer / consumer 关系，继续结合 `.claude/references/dsl-to-ascendc/references/dsl2Ascendc_cross_core_sync.md`；如果需要判断 PipeBarrier、SetFlag/WaitFlag 或 CrossCore 同步边界，再读取 `.claude/references/dsl-to-ascendc/references/AscendCSyncPipelinePattern.md`。

#### C. 子模块初始化

融合场景下，通常同时存在：

- Cube 子模块：如 `matmul.h`
- Vector 子模块：如 `leakyrelu.h`、`scale.h`

推荐在 `Init()` 中按分支初始化：

- `ASCEND_IS_AIC` 分支初始化 Cube 子模块
- `ASCEND_IS_AIV` 分支初始化 Vector 子模块

### 2. `Process()`：组织调度、AIC/AIV 分支与阶段调用

`Process()` 负责把工作负载循环、AIC/AIV 分支和模块调用串起来。

#### A. 工作负载循环

常见骨架：

```cpp
__aicore__ inline void KernelClass::Process()
{
    int mIdx, nIdx;
    while (sched_.HasNext()) {
        sched_.Next(mIdx, nIdx);

        if ASCEND_IS_AIC {
            // Cube 侧
        }

        if ASCEND_IS_AIV {
            // Vector 侧
        }
    }
}
```

#### B. AIC 分支

AIC 分支通常负责：

- 从 GM 取当前 tile 的输入
- 获取 workspace 生产者槽位
- 调用 Cube 子模块，如 `mm_.ComputeBlock(...)`
- 释放生产者槽位或发送完成信号

#### C. AIV 分支

AIV 分支通常负责：

- 获取 workspace 消费者槽位
- 根据 `GetSubBlockIdx()` 计算当前子块偏移
- 从 workspace 中取本子块负责的数据
- 调用 Vector 子模块完成后处理并写回 GM
- 释放消费者槽位或发送完成信号

#### D. 何时拆单独子模块

当满足以下任一条件时，建议拆出单独计算子模块文件：

- 设备侧设计中有多个职责清晰的计算阶段
- 同时存在 Cube 计算阶段和 Vector 后处理阶段
- 需要在主 `Kernel` 类中复用某段计算逻辑

建议让一个主要计算阶段对应 AscendC 中一个子模块。

<!-- 迁移自 porter kb/target/ascendc/fa_class/cv_lowering.md(整档忠实搬运,convert_docs_to_okf.py)。跨 op 参考/方法论知识,非机械家族。 -->
