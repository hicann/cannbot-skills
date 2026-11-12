---
type: CATLASS DSL Debugging Guide
title: 原子累加残值分布诊断
description: 把 atomic add 临时降级为普通写入，用残值元素的模粒度指纹定位未初始化搬运、写者所有权或同步缺口。
tags: [catlass-dsl, debug, atomic-add, workspace, residue, granularity, synchronization, numeric]
status: stable
generated: {by: process:catlass-dsl-knowledge-rebuild, at: '2026-09-23T00:00:00-07:00'}
verified:
  - {by: human:maintainer, at: '2026-09-23T00:00:00-07:00'}
sources:
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: CATLASS copy frontend and CopyL0C2DstParams atomic_mode
  - id: params
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/params.py
    title: CATLASS AtomicMode values and validation
  - id: mixed
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul_atomic_add.py
    title: CATLASS atomic accumulation example with a non-atomic first write
operator_families: [matmul, sparse-attention]
arch: [c310]
---

# 接口与概念

`atomic_mode` 只有 `NONE` 和 `ADD` 两个合法取值：`NONE` 是普通写入（覆盖目标），`ADD` 是读取
并累加到目标。[^params] 二者是同一个 `tla.copy` 的不同参数，因此可以在不改动任何数据流、
layout 或同步的前提下把 `ADD` 临时换成 `NONE`。[^api]

这个可切换性本身就是最强的诊断手段：普通写入会把“谁最后写了这个位置”暴露成残值，而原子
累加会把它掩盖成“总和看起来差不多”。

# 用法

先确认症状阶段，再决定是否值得做这项诊断：

- 累加结果整体偏小/偏大、且误差随 tile 数增长：先怀疑重复计入或漏算，不是本诊断。
- 少量离散元素错、且位置稳定：适用本诊断。
- 位置偶发漂移：先做同步/所有权审计，再回来做本诊断。

诊断步骤：

1. 把疑似路径的 `atomic_mode=AtomicMode.ADD` 换成默认（`NONE`），其余一律不动。
2. 用**同一个**输入重复运行，记录错误元素的数量、位置集合与其模粒度分布。
3. 按粒度指纹分流，见下表。
4. 定位后改回 `ADD`，用原用例复验，确认修复与降级诊断无关。

# 代码模式

```text
残值模粒度                       -> 首个怀疑对象
整 tile 连续一段                 -> 该 tile 的基址/stride，或某个 producer 完全没写
固定长度块（如 32 / 64 个元素）  -> 该粒度的搬运 chunk、mask lane 或 UB->GM 分块
每个 64 lane 的头部前 N 个        -> 向量寄存器 part / RegSlot 只填了部分 lane
按 512 元素周期重复              -> 一次搬运的整块，块内前段未初始化
无规律                           -> 同步/竞争，而不是搬运粒度
```

“某块从未被写过”和“某块被写成了旧值”要分开：把目标 workspace 预先填一个不可能出现的哨兵
（例如 `-3.0e38` 或一个非法 bit pattern），就能区分二者。

# 约束

- 降级只改 `atomic_mode`，不同时改 layout、同步或循环结构，否则残值指纹不可归因。
- 降级会让结果本身不再正确；它只用于定位，不能作为最终实现。
- 诊断用的哨兵与临时打印不得保留在最终 kernel 中。
- 必须先确认参考侧的舍入/精度路径没有改变观测量（见下），再开始归因。

# 失败表现

- 残值数量与“未初始化元素数”完全相等：问题在写者，不在累加。
- 降级后错误消失：真凶是累加语义（重复计入/漏算），不是未初始化内存。
- 残值位置随运行变化：同步或所有权问题，粒度指纹不适用。
- 观测值集中在某个特定数值（例如窄 dtype 的上界）：先检查诊断路径本身是否做了会改变数值的
  转换——把 FP32 中间量读回 host 再用 Torch 转成 FP16/BF16 时，宿主与设备侧的舍入/饱和规则
  不同，可能把一个普通数值伪装成“饱和值”，从而把归因引向错误方向。

# 验证方法

修复后用原用例复验，并额外跑一次“同一输入重复 N 次”的一致性检查与一次相邻边界（尾块、
空行、单 tile）检查。诊断结论只有在降级版与修复版给出可解释的对照时才算成立。

[^api]: 固定提交 `core_api.py` 中 `tla.copy` 与 `CopyL0C2DstParams` 的 `atomic_mode` 参数。
[^params]: 固定提交 `params.py` 中 `AtomicMode` 的取值与合法性校验。
[^mixed]: 固定提交 atomic 累加示例中“首个 tile 普通写、后续 tile 原子加”的写法。
