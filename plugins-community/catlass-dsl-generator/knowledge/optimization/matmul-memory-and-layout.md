---
type: CATLASS DSL Optimization Guide
title: CANN 样例：MatMul 片上复用与布局候选
description: 从 CANN Features 样例提取全载、L1 Bank 隔离、Scale 聚合缓存、SWAT 遍历和预转换权重候选。
tags: [catlass-dsl, optimization, matmul, l1, cache, layout, mte1, mte2]
status: draft
generated: {by: process:cann-samples-feature-extract, at: '2026-08-12T00:00:00Z'}
verified:
  - {by: process:cann-samples-source-audit, at: '2026-08-12T00:00:00Z'}
sources:
  - id: full-load
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/memory_optimization/full_load/main.asc
    title: Full-load implementation
  - id: l1-bank
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/memory_optimization/l1_bank_conflict/main.asc
    title: L1 bank-isolated ping-pong implementation
  - id: scale-cache
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/memory_optimization/scale_cache/main.asc
    title: Quantized MatMul scale-cache implementation
  - id: swat
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/memory_optimization/slide_window_adaptive_template/main.asc
    title: Sliding-window adaptive tile ordering
  - id: weight-nz
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/instruction_optimization/weightnz/main.asc
    title: Prepacked FRACTAL_NZ weight implementation
operator_families: [matmul, quantized-matmul]
arch: [c310]
---

# 接口与概念

本页把五种固定源码机制转写成 CATLASS DSL 的待验证候选，而不继承样例 README 的性能结论。
它们分别减少重复 GM→L1 搬运、避免 L1 ping/pong 地址冲突、合并小粒度 Scale 搬运、
改变多核 tile 访问顺序以争取 L2 复用，以及把固定权重预先转换为计算亲和布局。

`full_load` 把可容纳的 A 块常驻 L1，并在多个 N tile 间复用；实现保留 B 的 ping/pong，
且用 shape、核数和 L1 容量检查拒绝不支持的输入。[^full-load]
`l1_bank_conflict` 以 `TOTAL_L1_SIZE / 2` 为边界，把 `[Ping A, Ping B]` 与
`[Pong A, Pong B]` 放到不同半区。[^l1-bank]

`scale_cache` 以 `scaleKL1Ratio` 控制每隔多少个 K tile 才重载 Scale，并让一次加载覆盖后续
多个 K tile；`SWAT` 把线性 tile 映射为窗口内 M 优先、窗口间 N 蛇形的坐标；`weightnz`
让固定 B 权重以对齐后的 NZ 布局进入 kernel。[^scale-cache][^swat][^weight-nz]

# 用法

只在 profile 支持相应瓶颈时选择一个轴：

- MTE2 重复读同一小矩阵：测试 L1 全载；
- MTE1 等待随 ping/pong 地址变化：测试 Bank 边界隔离；
- Scale 单次字节数过小且重复加载：测试聚合缓存；
- 早期轮次 L2 miss、后续命中，且多核 tile 顺序可变：测试 SWAT；
- 推理权重固定且 kernel 内存在 ND→NZ 随路转换：测试离线预转换。

不要同时更改 tile shape、buffer 数和布局；先得到可归因的单轴结果。

# 代码模式

## 常驻复用与聚合小搬运

```python
# 伪代码：具体 layout、capacity 和 copy API 以当前 CATLASS DSL 为准。
resident_a = make_l1_tensor((m_tile, k))
copy(resident_a, gm_a_block)                 # 每个可复用域仅一次
for n_tile in range(n_tiles_in_reuse_domain):
    for k_tile in range(k_tiles):
        copy(l0_a, resident_a[k_tile])
        copy(l0_b, gm_b[n_tile, k_tile])
        mmad(...)

if k_tile % scale_group_tiles == 0:
    copy(scale_l1, gm_scale[k_tile:k_tile + scale_group_tiles])
scale_for_tile = scale_l1[k_tile % scale_group_tiles]
```

全载的所有权域必须与复用域相同；Scale 聚合的最后一组必须按剩余 K 截断。
源码分别显式实现了“首批加载后复用”和 `iter0 % scaleKL1Ratio == 0`。
[^full-load][^scale-cache]

## Bank 隔离与 SWAT 坐标

```python
half_l1 = total_l1_bytes // 2
stage_base = stage_id * half_l1
a_l1 = l1_at(stage_base)
b_l1 = l1_at(stage_base + aligned_a_bytes)

window = min(window_len, m_tiles)
row = linear_tile // (n_tiles * window)
m_tile = row * window + linear_tile % window
n_tile = (linear_tile // window) % n_tiles
if row % 2:
    n_tile = n_tiles - 1 - n_tile
```

这只是地址与调度候选；CATLASS 编译后的真实 L1 地址、Bank 映射和 block 调度必须从 IR/profile
复核。尾窗口需单独计算，不能把主窗口公式直接用于不足 `window` 的末行。[^l1-bank][^swat]

## 固定权重预转换

调用侧一次性把固定权重 pad 到计算基本块并转换为 NZ；kernel 的 GM tensor layout、地址公式、
实际分配字节数和 golden 都同步切换。源码对 FP16/BF16 与 FP32 使用不同的基本块宽度，
因此 dtype 是布局合同的一部分。[^weight-nz]

# 约束

- 适用范围：Ascend 950 / `c310` 候选；MatMul 或量化 MatMul；具体 dtype/layout/shape 由实验记录。
- 保持条件：数学结果、累加精度、tail/mask、GM ABI、tile 所有权和同步顺序不变；预转换方案除外，
  其调用侧 ABI 必须成套更新并与原始 ND 权重逐元素等价。
- 容量代价：全载和 Scale 缓存占用额外 L1，Bank 隔离把每 stage 的可用上限收紧到半区；padding
  增大权重存储；SWAT 增加坐标计算和尾窗口分支。
- 负载前提：全载需要足够复用次数；Scale 缓存需要多个消费者 tile；SWAT 需要可观察的 L2
  热身/复用；预转换只适合可摊销转换成本的固定权重。
- 源码中的具体 512 KiB/256 KiB 容量与 shape 阈值是该固定样例的实现条件，不能直接当作所有
  CATLASS kernel 的通用常量。[^full-load][^l1-bank]

# 失败表现

- 编译或运行时本地内存越界、并行 block 数下降：回退容量扩大，缩小驻留域或缓存组。
- 偶发错、只在 ping/pong 轮换时错：地址重叠或 release/ready 同步缺失，恢复原地址布局。
- 尾 K、尾窗口或 padding shape 错：恢复原 tail 路径并逐轴检查坐标。
- MTE2/MTE1 指标不降或 kernel 延迟落在噪声内：瓶颈或复用假设被证伪，回退候选。
- 预转换端到端更慢：转换未被复用摊销，保留 ND 路径。

# 验证方法

1. 先用原始布局跑完整正确性，覆盖单 tile、多 tile、奇偶轮换、尾 M/N/K 和容量边界。
2. 每次只启用一个候选；检查 lowering 后的 L1 地址、copy 次数、layout 和同步。
3. 同配置采集 MTE1/MTE2、Cube 等待、L2/流量指标以及 kernel latency；重复 benchmark，
   设定高于噪声的接受阈值。
4. 对最终 fresh best 再跑完整正确性和 profile；否则结论保持 `experimental`，不得写入 learned。

[^full-load]: 固定提交中的 A 全载地址、一次加载/多轮复用和准入检查实现。
[^l1-bank]: 固定提交中按 L1 半区隔离 ping/pong 的地址计算。
[^scale-cache]: 固定提交中按 K 比例聚合并复用 Scale 的实现。
[^swat]: 固定提交中主窗口、尾窗口及蛇形 N 坐标重映射。
[^weight-nz]: 固定提交中预转换 NZ 权重的布局、对齐和 GM 视图。
