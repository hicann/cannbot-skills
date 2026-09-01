---
type: CATLASS DSL Operator Example
title: Basic MMAD 分块、数据类型路由与双缓冲
description: GM-L1-L0 分层、动态 tiling、FP/BF16/FP8/INT8/HF32 路由及 K 循环双缓冲模式。
tags: [catlass-dsl, operator, matmul, double-buffer]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
  - {by: process:catlass-dsl-source-audit, at: '2026-08-29T00:00:00Z'}
sources:
  - id: kernel
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/examples/end_to_end/basic_mmad/basic_matmul.py
    title: Basic MMAD kernels
  - id: guide
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/examples/end_to_end/basic_mmad/README.md
    title: Basic MMAD guide
operator_families: [matmul]
arch: [c310]
---

# 接口与概念

## 算子算法

计算 `C = A @ B`。K 维 partial product 在 L0C 中累加；`init_c` 仅首个 K tile
为真，末次计算用 `unit_flag=0b11` 完成 MMAD/FIX 协议。当前实现支持同型
f16/bf16/f32 输入、任意 f8e4m3fn/f8e5m2 输入配对，均以 f32 L0C 累加；另有
i8×i8→i32 路由。f32 输入可用 `HF32Mode` 选择 HF32 舍入，其他路由禁用 HF32。
[^kernel][^guide]

# 用法

## 分核策略与基本块切分

默认问题为 `M=256, N=512, K=1024`。默认 `TilingParams` 的 L1 tile 是
`256x256x128`，L0 tile 是 `256x256x32`；tile 字段以 dataclass/`Constexpr` 进入
kernel，不再依赖模块级常量。输出二维 tile 线性化后由 grid-stride 覆盖：[^kernel]

```python
L1_M = _tiling.l1_tm
L1_N = _tiling.l1_tn
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

默认 f16 路径下，L1 A/B 和 L0A/L0B 均双缓冲，L0C 单缓冲；单个逻辑 buffer
容量分别为 131072、131072、16384、16384 和 262144 bytes。实际 bytes 必须按
本次 dtype 与 `_tiling` 重算；FP8/INT8 输入 buffer 更小，但 i8 路由的 L0C 为 i32。
[^kernel]

当前样例使用单一 kernel，从 `gm_a.ptr.dtype` / `gm_b.ptr.dtype`、tensor metadata、
`TilingParams` 和 `Constexpr` 参数取得 dtype、M/N/K、tile、HF32 与整数累加选择，不需要
Host 改写模块全局状态。FP8 因 PyTorch DLPack 不能导出对应 dtype，Host 绑定通过
`create_tla_tensor(..., element_type=...)` 显式覆盖元素类型。[^kernel]

```python
l1_a = tla.make_tensor_like(l1a_ptr0 if l1_buf == 0 else l1a_ptr1, a_src)
l1_b = tla.make_tensor_like(l1b_ptr0 if l1_buf == 0 else l1b_ptr1, b_src)
tla.mmad(l0_c, l0_a, l0_b, init_c=first_k, unit_flag=unit_flag)
tla.copy(gm_c, l0_c, tla.params.CopyL0C2DstParams(unit_flag=0b11))
```

## 流水排布、同步关系与数值精度

双 buffer 交替承载下一块搬运和当前块计算。MTE2→MTE1、MTE1→CUBE 使用
data-ready flag，反方向使用 buffer-available flag；CUBE/FIX 由 unit flag 或显式
flag 协调 L0C。浮点/FP8 路径保留 f32 partial sum，整数路由保留 i32 partial sum；
HF32 reference 会先按选择的 HF32 mode 舍入 f32 输入。[^kernel][^guide]

# 约束

- A/B packed layout 必须匹配 L0A/L0B 的硬件解释。
- `init_c` 只在第一个 K tile 为真。
- buffer 索引、set/wait 和 unit flag 必须同步轮换。
- f16、bf16、f32 只接受同型输入；FP8 两种格式允许任意配对且 C 为 f32；整数路由严格为
  i8×i8→i32。HF32 只用于 f32×f32。
- kernel 不得读取模块级 tile/dtype，也不得由 host 改写全局 dtype；变体复用同一个
  自包含 `@tla.kernel`，由 tensor 类型和显式 `Constexpr` 生成不同编译产物。

# 失败表现

- `init_c` 每轮为真：只保留最后一段 K。
- packed layout 错误：输出块转置或错位。
- buffer release 错误：读旧数据或流水挂起。
- dtype 或 tile 随调用顺序变化：复制了上游样例的模块级特化写法。
- MMAD dtype contract 错误：L0C 仍固定 f32，却选择了 i8×i8，或 FP8/i8 输出类型不匹配。
- f32 结果与普通 FP32 oracle 系统偏差：启用了 HF32，却没有对 oracle 做相同舍入。

# 验证方法

使用非对称 M/N/K、跨多个 L1/L0 K tile 的 case，并按路由覆盖 f16、bf16、f32、两种
FP8 的四种配对及 i8×i8→i32。检查 TLAIR 中 copy、MMAD dtype、`hf32_mode`、unit flag
和同步顺序；整数使用精确相等，FP8/HF32 使用与输入量化/舍入一致的独立 oracle。

[^kernel]: 固定提交 MMAD kernel 的算法、切分、layout、buffer 与同步实现。
[^guide]: 固定提交 MMAD README 的数据流说明。
