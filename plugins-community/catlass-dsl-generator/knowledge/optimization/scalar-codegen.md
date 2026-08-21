---
type: CATLASS DSL Optimization Guide
title: CANN Performance：Scalar 代码生成减负
description: 从 ScalarBound 专题提取寄存器 spill、I-cache、分支和动态别名的诊断与可证伪代码形态候选。
tags: [catlass-dsl, optimization, scalar, codegen, register-spill, icache, branching]
status: draft
generated: {by: process:cann-samples-performance-extract, at: '2026-08-12T00:00:00Z'}
verified:
  - {by: process:cann-samples-source-audit, at: '2026-08-12T00:00:00Z'}
sources:
  - id: scalar-story
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/2_Performance/scalar_story/README.md
    title: Ascend 950 ScalarBound diagnosis and optimization story
operator_families: [matmul, convolution, attention, mixed, elementwise]
arch: [c310]
---

# 接口与概念

固定专题把 ScalarBound 区分为 scalar 无法及时发射而使 Cube/Vector/MTE 出现 bubble，并要求进一步
用指令日志、PC-cycle 和分支统计区分 spill、I-cache 与预测失败。其候选包括：I-cache 预取；静态
创建本地 tensor；把热成员缓存为局部值；缩短 live range；消除多级指针；拆开动态索引数组；把
主/尾循环分离；用显式循环替代大型状态机；缩小热结构体。[^scalar-story]

对 CATLASS Python DSL，这些不是机械翻译 C++ 类布局，而是 IR 首查：只有 lowering 后出现对应
scalar load/store、spill、动态地址或 I-cache 症状时，才测试等价的 Python 控制流/常量化重写。

# 用法

先确认 scalar 时长高于有效 producer pipe 且其它 pipe 有 bubble；统计 scalar load/store，定位 PC
停顿和分支。spill 优先减少动态对象/索引与长 live scalar；I-cache 优先精简/拆热路径；预测失败
优先把 prologue、无分支 hot loop、epilogue 分开。每次只改变一种代码形态。

# 代码模式

```python
# 候选：把首尾条件移出热循环，并把只读动态值缓存为局部 scalar。
k_total = runtime_k_tiles
if k_total > 0:
    run_first_tile(init=True)
for k in range(1, max(k_total - 1, 1)):
    run_middle_tile_no_tail_branch(k)
if k_total > 1:
    run_last_tile(final=True)

# 候选：避免热循环中的多级动态索引。
shape_k = problem_shape_k
event0, event1 = make_two_named_events()
event = event0 if slot == 0 else event1
```

当循环很短时，三段式会增大代码体积，可能反向触发 I-cache；动态数组拆标量也只在 IR 显示常量
传播/别名受阻时实验。[^scalar-story]

# 约束

- 适用：`c310` 且 profile/IR 已证实 ScalarBound；不能用 scalar ratio 单独下结论。
- 保持：循环迭代集合、首尾事件、tail、地址、同步和异常路径完全等价。
- 代价：代码复制增加 I-cache；局部缓存增加寄存器；预取本身有指令成本；过度内联会膨胀 binary。
- 可证伪预期：scalar load/store、spill、PC 长停顿或 mispredict 下降，同时其它 pipe bubble 和总延迟下降。
- 若根因是下游 issue queue 反压，应优化对应 pipe，不应继续改 scalar 代码。

# 失败表现

- scalar 指令下降但 I-cache miss 上升：撤销展开/三段复制，缩小热函数。
- 仅 tail shape 错：首尾拆分遗漏 0/1 次循环，恢复统一循环并重写边界证明。
- 寄存器 spill 增加：局部缓存/live range 过多，缩短作用域或分阶段。
- scalar 下降而总延迟不变：原先不是临界路径，停止该轴。

# 验证方法

正确性覆盖循环次数 0、1、2、长循环、周期事件和全部 tail；比较 lowering IR 与 scalar 指令日志中的
load/store、spill、branch、代码大小和 PC-cycle。用同配置 profile 验证 Cube/Vector/MTE bubble 是否
同步缩短，并重复 benchmark；只保留高于噪声且 fresh 复测通过的候选。

[^scalar-story]: 固定提交中的 ScalarBound 分类、诊断流程、I-cache/局部值/别名/循环/结构体候选及适用条件。
