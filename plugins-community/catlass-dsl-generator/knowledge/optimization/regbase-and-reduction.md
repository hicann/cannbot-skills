---
type: CATLASS DSL Optimization Guide
title: CANN Performance：RegBase、SIMD 与归约链优化
description: 从 GELU、Softmax、RmsNormQuant、KV-RMSNorm-RoPE 和 SIMD VF 专题提取寄存器融合、循环组织、归约并行与大 tile 候选。
tags: [catlass-dsl, optimization, vector, regbase, simd, reduction, softmax, rmsnorm]
status: draft
generated: {by: process:cann-samples-performance-extract, at: '2026-08-12T00:00:00Z'}
verified:
  - {by: process:cann-samples-source-audit, at: '2026-08-12T00:00:00Z'}
sources:
  - id: gelu
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/2_Performance/gelu_eltwise_regbase_story/README.md
    title: GELU plus elementwise RegBase story
  - id: softmax
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/2_Performance/softmax_regbase_story/README.md
    title: Softmax RegBase optimization story
  - id: rmsnorm
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/2_Performance/rms_norm_quant_story/README.md
    title: RmsNormQuant optimization story
  - id: kv-rope
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/2_Performance/kv_rms_norm_rope_cache_story/README.md
    title: KV RMSNorm RoPE cache MemBase-to-RegBase story
  - id: simd-vf
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/2_Performance/simd_vf_story/README.md
    title: SIMD VF broadcast, elementwise and reduction patterns
operator_families: [elementwise, softmax, normalization, reduction, rope]
arch: [c310]
---

# 接口与概念

共性主线是固定外层 GM/UB 数据流，只把热 compute body 从“每步读写 UB”改为“一次 load、寄存器串联、
一次 store”；随后才测试 VF 循环拆分/合并、常量外提、多行并行和 tile 扩大。[^gelu][^kv-rope]
RmsNormQuant 明确展示瓶颈迁移顺序：共享 gamma 预加载、多核、VF 融合减少中间 UB 写、等 Vector 与
MTE2 接近后再开双缓冲、扩大 UB tile，最后单独实验二分累加。[^rmsnorm]

Softmax 对 reduce 使用两条独立链或多行累加器隐藏依赖延迟，搬运成为新瓶颈后再扩大 tile；合并 VF
只有减少真实调用/屏障才可能兑现，单纯重排 barrier 可能时间守恒。[^softmax]
SIMD 专题把 broadcast 外层循环收入 VF 以减少 VF/PB 次数；无循环携带依赖的 elemwise 通常无需过度
展开；reduce_max 用多累加器，允许改变求和顺序的 reduce_sum 可测试二分树。[^simd-vf]

# 用法

先冻结 tiling/copy/输出合同，只替换 compute body。若 UB 往返或 VF 调用碎片化明显，测试融合；若
reduce 指令间有长依赖链，测试 2/4 路累加或二分；若 Vector 降下后 MTE2 成最长轴，再测试大 tile
或 double buffer。常量外提与 loop split/unroll 分开测，不能一次叠加。

# 代码模式

```python
with tla.vec.func(mode="simd"):
    # 多路 partial 独立，最后再合并。
    acc0, acc1 = init0(), init1()
    for chunk in range(0, chunks, 2):
        acc0 = reduce_update(acc0, load(chunk))
        if chunk + 1 < chunks:
            acc1 = reduce_update(acc1, load(chunk + 1))
    total = combine(acc0, acc1)
    out = normalize_and_fuse(total)  # 中间量留寄存器
    store(out, mask=tail_mask)
```

Broadcast 优先选择能同时保持主数据连续、复用广播寄存器且让最内层足够长的循环顺序；没有统一的
`n→m` 或 `m→n` 最优规则。KV-RMSNorm-RoPE 的迁移边界是保留 cache offset、double-buffer copy
和 golden，只替换 RMSNorm/RoPE VF。[^simd-vf][^kv-rope]

# 约束

- 适用：`c310` Vector/RegBase；dtype、axis、行长、tile 与 tail mask 是实验条件。
- 保持：公式、BF16↔FP32 转换点、epsilon、量化舍入/饱和、cache layout、mask 与写回顺序不变。
- 浮点 sum 二分会改变结合顺序，只能在既定误差合同内接受；max 多累加器仍需处理 masked lane identity。
- 代价：融合/展开增加寄存器和代码体积；大 tile 占 UB；按需 cast gamma 减 UB 但增加重复计算；
  non-aligned store 必须与 post 收尾成对。
- 可证伪预期：UB load/store、VF/PB、依赖空泡或 MTE 次数下降，并改善总延迟而非只迁移 idle。

# 失败表现

- spill、software loop 或编译失败：减少融合范围、unroll 和并行累加器。
- tail/邻接内存错：恢复原 mask/对齐 store，检查 MERGING/ZEROING 语义。
- 计算侧下降但总延迟不变：MTE/idle 成新瓶颈，停止继续融合并回到 profile 路由。
- 大 tile 更慢：UB occupancy、tail 浪费或流水深度抵消收益，回退原 tile。
- sum 超容差：恢复原归约顺序或提高 accumulator 精度。

# 验证方法

覆盖 axis 两向、lane/tile 边界、奇数 chunk、单/多行、BF16 特殊值和 cache 四输出；复用同一 golden。
检查 IR/trace 的 VF count、PUSH_PB、RVECLD/RVECST、spill、hardware loop 与 barrier。正确后同配置比较
Vector/MTE2/MTE3/scalar、UB 容量和总延迟；任何负优化立即回退，fresh best 再跑完整矩阵。

[^gelu]: 固定提交中的 MemBase→VF 融合、循环拆分、展开和常量外提递进版本。
[^softmax]: 固定提交中的 binary fold、多行并行、prefetch 反例、大 tile与合并 VF 实验。
[^rmsnorm]: 固定提交中的 gamma 预加载、多核、VF、DB、UB 利用率和二分累加步骤。
[^kv-rope]: 固定提交中的固定外层、RMSNorm/RoPE 寄存器链、按需 gamma cast 与 cache 合同。
[^simd-vf]: 固定提交中的 broadcast 循环收拢、elemwise/reduce 判别、多累加器和二分范式。
