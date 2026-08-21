---
type: CATLASS DSL Operator Example
title: Basic MMAD 分块与双缓冲
description: GM-L1-L0 分层、layout、dtype、init_c、unit_flag 与 K 循环双缓冲模式。
tags: [catlass-dsl, operator, matmul, double-buffer]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
sources:
  - id: kernel
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_mmad/basic_matmul.py
    title: Basic MMAD kernels
  - id: guide
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_mmad/README.md
    title: Basic MMAD guide
operator_families: [matmul]
arch: [c310]
---

# 接口与概念

## 算子算法

计算 `C = A @ B`。K 维 partial product 在 L0C 中累加；`init_c` 仅首个 K tile
为真，末次计算用 `unit_flag=0b11` 完成 MMAD/FIX 协议。A/B 为 f16，L0C 使用
f32 accumulator。[^kernel][^guide]

# 用法

## 分核策略与基本块切分

默认问题为 `M=256, N=512, K=1024`。L1 tile 是 `256x256x128`，L0 tile 是
`256x256x32`。输出二维 tile 线性化后由 grid-stride 覆盖：[^kernel]

```python
L1_M = 256
L1_N = 256
m = gm_a.origin_shape[0]
n = gm_b.origin_shape[1]
grid_m = (m + L1_M - 1) // L1_M
grid_n = (n + L1_N - 1) // L1_N
for task in tla.range(tla.arch.block_idx(), grid_m * grid_n, tla.arch.block_num()):
    block_m = task // grid_n
    block_n = task % grid_n
```

每个输出 tile 先遍历 128 宽的 L1 K tile，再遍历 32 宽的 L0 K tile。
上述大写 tile 名表示 kernel 函数内的 Python 局部常量，不是模块级配置。

# 代码模式

## 数据路径与存储层级

```text
GM A/B -> L1 A(zN)/B(nZ) -> L0A/L0B -> MMAD -> L0C(fp32) -> FIX -> GM C
```

L1 A/B 和 L0A/L0B 均双缓冲，L0C 单缓冲；容量分别为 131072、131072、
32768、32768 和 262144 bytes。[^kernel]

上游样例把 tile 和输入 dtype 保存在模块级，并由 host 修改 dtype；这只能作为算法、
layout 和同步的来源证据，不能原样复制为生成规范。生成实现只声明一个 kernel，在函数
内定义固定 tile，从输入 tensor 的 `dtype` 和 metadata 读取 dtype 与实际 M/N/K。

```python
l1_a = tla.make_tensor_like(l1a_ptr0 if l1_buf == 0 else l1a_ptr1, a_src)
l1_b = tla.make_tensor_like(l1b_ptr0 if l1_buf == 0 else l1b_ptr1, b_src)
tla.mmad(l0_c, l0_a, l0_b, init_c=first_k, unit_flag=unit_flag)
tla.copy(gm_c, l0_c, tla.params.CopyL0C2DstParams(unit_flag=0b11))
```

## 流水排布、同步关系与数值精度

双 buffer 交替承载下一块搬运和当前块计算。MTE2→MTE1、MTE1→CUBE 使用
data-ready flag，反方向使用 buffer-available flag；CUBE/FIX 由 unit flag 或显式
flag 协调 L0C。f16 A/B 进入 MMAD，partial sum 始终保留为 f32。[^kernel][^guide]

# 约束

- A/B packed layout 必须匹配 L0A/L0B 的硬件解释。
- `init_c` 只在第一个 K tile 为真。
- buffer 索引、set/wait 和 unit flag 必须同步轮换。
- kernel 不得读取模块级 tile/dtype，也不得由 host 改写全局 dtype；fp16、bf16 等编译期
  变体必须复用同一个自包含 `@tla.kernel`，并由输入 tensor 类型生成不同编译产物。

# 失败表现

- `init_c` 每轮为真：只保留最后一段 K。
- packed layout 错误：输出块转置或错位。
- buffer release 错误：读旧数据或流水挂起。
- dtype 或 tile 随调用顺序变化：复制了上游样例的模块级特化写法。

# 验证方法

使用非对称 M/N/K、跨多个 L1/L0 K tile 的 case，并检查 TLAIR 中 copy、MMAD、
unit flag 和同步顺序；同时枚举 kernel 自由名字，除 `tla` 和 Python 内建符号外必须为空。

[^kernel]: 固定提交 MMAD kernel 的算法、切分、layout、buffer 与同步实现。
[^guide]: 固定提交 MMAD README 的数据流说明。
