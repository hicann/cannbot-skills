---
type: CATLASS DSL Optimization Guide
title: CANN 样例：Vector、寄存器与 CUBE–Vector 数据路径候选
description: 从 CANN Features 样例提取 VF 融合、寄存器搬运选型、SIMD 约束、SIMT Gather、HiF8 与 L0C→UB 直通候选。
tags: [catlass-dsl, optimization, vector, register, simd, simt, mixed, cv, hif8]
status: draft
generated: {by: process:cann-samples-feature-extract, at: '2026-08-12T00:00:00Z'}
verified:
  - {by: process:cann-samples-source-audit, at: '2026-08-12T00:00:00Z'}
sources:
  - id: cv-path
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/hardware_features/cv_datapath/src/2_mix_scenario2.asc
    title: Mixed CUBE-Vector L0C-to-UB path
  - id: vector-function
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/hardware_features/vector_function/gelu_with_vf.asc
    title: Register-resident fused GeLU vector function
  - id: simd-constraints
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/hardware_features/simd_vf_constraints/README.md
    title: SIMD vector-function constraints and examples
  - id: register-movement
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/memory_optimization/reg_data_movement/Reg%E6%95%B0%E6%8D%AE%E6%90%AC%E8%BF%90%E5%9C%BA%E6%99%AF%E9%80%89%E5%9E%8B%E6%8C%87%E5%8D%97.md
    title: Register data-movement selection guide
  - id: simt-gather
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/hardware_features/simt/main.asc
    title: SIMT Gather implementation
  - id: hif8
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/hardware_features/hif8/quantize_custom.asc
    title: HiFloat8 quantization implementation
operator_families: [elementwise, gather, mixed, quantization]
arch: [c310]
---

# 接口与概念

可迁移的优化轴有六个：把多段 elementwise 表达式放进一次寄存器驻留的 Vector Function；按
连续、变位宽、广播/标量、块跳跃/Gather、交织/掩码/非对齐场景选择寄存器搬运；保持可识别的
hardware loop；对不规则逐元素索引测试 SIMT；对存储/带宽受限链路测试 HiF8；在 mixed
CUBE→Vector 中测试 L0C→UB 直通，消除 GM 中转。[^vector-function][^register-movement]

固定 SIMD 指南要求把可外提 scalar 和地址计算移出 VF、缩短寄存器 live range、避免循环内
整数除模破坏 hardware loop，并只在真实跨 pipe/跨轮地址依赖上插入 LocalMemBar。
[^simd-constraints] SIMT 样例把 Gather 映射为线程级索引；mixed 样例使用跨核就绪通知和双 AIV
分工消费从 L0C 直达 UB 的结果。[^simt-gather][^cv-path]

# 用法

- 多个 vector op 之间反复寄存器↔UB：先做单 VF 融合；
- tail、变位宽、广播、Gather 或非对齐访问：按数据分布选 load/store dist，再独立选地址推进和 mask；
- SIMD 因分支/地址不规则而低效：比较 SIMT，但必须计入线程映射与单 AIV block 限制；
- CUBE 输出立刻被 Vector 消费：比较 GM 中转与 L0C→UB；
- 带宽/容量主导且允许量化误差：比较 HiF8，正确性合同需改为量化误差合同。

# 代码模式

```python
# CATLASS DSL 示意：一次 load，多步寄存器计算，一次 store。
with tla.vec.func(mode="simd"):
    x = ub_in.load(mask=tail_mask)
    y = gelu_fused(x)
    ub_out.store(y, mask=tail_mask)

# Mixed 直通：producer/consumer 对相同 UB 分区和跨核事件达成一致。
with tla.cube():
    tla.mmad(l0_c, l0_a, l0_b, init_c=True)
    tla.copy(shared_ub_for_subblock, l0_c)
    tla.cross_core_set_flag(ready, tla.arch.FIX)
with tla.vector():
    tla.cross_core_wait_flag(ready, tla.arch.VECTOR)
    with tla.vec.func(mode="simd"):
        ...
    tla.copy(gm_out, shared_ub_for_subblock)
```

寄存器搬出 tail 必须使用有效 lane mask；非对齐 store 是成对协议，不能漏掉收尾步骤；Gather
索引的单位、范围和 dtype 必须与源 tensor 一致。[^register-movement]

# 约束

- 适用范围：`c310` elementwise/gather/mixed/quantization；具体 CATLASS lowering 是否支持对应
  dist、SIMT 或 HiF8，必须先由编译/IR 确认。
- 保持条件：融合不改变公式和舍入点；mask 不脏写 tail；L0C→UB 的 AIC/AIV 分区一致；事件覆盖
  producer 的最后写和 consumer 的首次读；量化路径遵守明确 scale、饱和、NaN/Inf 和误差合同。
- 资源代价：融合增加寄存器压力和代码体积；参数过多增加 Parameter Buffer/标量压力；SIMT 有
  线程与 block 上限；直通占用共享 UB 并增加跨核同步；HiF8 引入量化开销和精度损失。
- LocalMemBar 会限制展开/并行发射；先消除别名或跨轮依赖，再考虑精确方向的 barrier。
- 预期 metric：UB/GM 往返、AIV MTE2、register spill、software-loop 退化、scalar/address 指令、
  跨核等待和总延迟；任一单项不能独立证明收益。

# 失败表现

- 编译失败或 hardware loop 退化：简化循环头、外提 scalar/地址除模、减少参数与 live register。
- tail 后邻接数据被改写：store mask、地址单位或非对齐收尾错误，恢复对齐路径。
- mixed 偶发错/hang：SubBlock 映射或 cross-core set/wait 不闭合，回退 GM 中转。
- 直通正确但更慢：UB 压力、同步或 Vector 尾部未被 CUBE 覆盖。
- SIMT 正确但更慢：访问仍规则、线程开销超过 Gather 收益或并行 block 受限。
- HiF8 超容差或特殊值不一致：回退原 dtype，不能以性能替代正确性。

# 验证方法

正确性覆盖 lane-1/lane/lane+1 tail、非对齐地址、越界索引防护、NaN/Inf/饱和值、双 AIV 分区
边界和多轮重复。检查 IR 的 hardware/software loop、spill、barrier 方向、GM round trip 和实际
dtype。随后在同配置下比较 UB/GM 字节、AIV MTE2/MTE3、Vector/scalar、跨核等待和总 latency；
只有完整正确性与高于噪声的 fresh benchmark 才能进入 learned。

[^cv-path]: 固定提交中 mixed kernel 的 L0C→UB、双 AIV 分区和跨核同步实现。
[^vector-function]: 固定提交中 GeLU 的寄存器驻留融合实现。
[^simd-constraints]: 固定提交文档及其配套源码定义的 SIMD 参数、hardware loop、barrier 和资源约束。
[^register-movement]: 固定提交中的寄存器搬运模式、mask、寻址及错误案例。
[^simt-gather]: 固定提交中的 SIMT 线程级 Gather 映射。
[^hif8]: 固定提交中的 HiFloat8 量化与转换实现。
