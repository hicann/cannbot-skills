---
type: "operator"
title: "Linear Attention 矩阵逆模式"
description: "单位下三角矩阵块化求逆及 Vector/Cube Stage。"
tags: ["catlass-cpp", "linear-attention", "pr1069"]
status: "stable"
generated: {"by": "process:catlass-cpp-pr1069-extraction", "at": "2026-09-18T00:00:00Z"}
verified: [{"by": "process:architecture-separation-source-check", "at": "2026-09-20T00:00:00Z"}]
sources: [{"id": "pr1069-matrix-inverse-patterns", "resource": "git:cann/cannbot-skills@4950cbd7c45e0d44cb8ec52598edf53cdc7f16ca:ops/catlass-linear-attention-workflow/references/matrix-inverse-patterns.md", "title": "PR1069 fixed source for Linear Attention 矩阵逆模式", "kind": "repository"}, {"id": "catlass-arch-separation", "resource": "git:cann/catlass@0d78a73c192fc04741a4fd57c94080209424775c", "title": "CATLASS fixed source for AtlasA2 and Ascend950 architecture separation", "kind": "repository"}]
operator_families: ["linear-attention"]
architectures: ["atlas-a2-a3", "ascend950"]
---
# 接口与概念

本 concept 是 PR1069 固定来源的结构化入口；规范性技术正文完整保留在“PR1069 原始正文”。

## 算子算法

算法定义、公式、计算顺序和边界语义以原始正文及当前任务冻结的 operator/golden contract 为准。

## 分核策略与基本块切分

分核、任务组、基本块和尾块规则以原始正文为准，并需针对当前 shape、SoC 和资源重新推导。

## 数据路径与存储层级

GM、workspace、L1、UB、L0 的地址、生命周期和复用要求以原始正文为设计输入。

## 流水排布、同步关系与数值精度

Stage、slot、CrossCore/HardEvent、累加和转换规则以原始正文为准，实际结论必须通过验证。

# 用法

先通过 compact query 定位本 concept，再完整读取；不得把知识内容当作当前工程已验收的证据。

# 代码模式

原始正文中的伪代码和命令保持原义。具体 API 必须核对目标版本 CATLASS/CANN 源码。

# 约束

不得删减或放宽原始规则、阈值、失败条件、状态恢复或验收要求。

# 失败表现

若设计、代码或测试与原始约束不一致，按五阶段 workflow 的 issue_type/resume_from 矩阵恢复。

# 验证方法

对照 frontmatter `sources` 中的固定 PR SHA 和原文件逐段校验；阶段通过仍需实际构建、精度和性能证据。

# PR1069 原始正文

# 单位下三角矩阵求逆的块化实现经验

本参考用于 chunk 线性 Attention 中的单位下三角矩阵求逆。目标是把固定大小的矩阵求逆改写成可在 Ascend 上由 Vector 和 Cube 协同完成的块算法，保留 FP32 累加和明确的中间精度边界。

## 架构适用性

矩阵分块公式、FP32 累加、Stage 依赖和精度检查点是 A2/A3 与 A5 共享的算法知识。实现路径按
`target_architecture` 分支：A2/A3 使用 `2201/Arch::AtlasA2`、MemBase 和 GM 中转；
A5 使用 `3510/Arch::Ascend950`，可在组件支持时选择 RegBase/VF、Fixpipe 直达 AIV UB 或
AIV UB 直达 AIC L1，也可使用 GM 基线。[^catlass-arch-separation]

## 1. 先利用矩阵结构

对严格下三角矩阵 `L`，要求 `A = (I + L)^(-1)`。当 chunk 为 64 时，按 32×32 分成 2×2 块：

```text
I + L = [ I + L00    0  ]
        [   L10    I + L11 ]
```

分别求两个对角叶子：

```text
XR = (I + L00)^(-1)
XL = (I + L11)^(-1)

LeafRight = [ XR  0 ]
             [  0  0 ]

LeafLeft  = [ 0  0 ]
             [ 0 XL ]
```

再用两次块矩阵乘法恢复完整结果：

```text
Y = I + LeafLeft @ (-L)
A = LeafLeft + Y @ LeafRight
```

展开后得到：

```text
A = [ XR              0 ]
    [ XL @ (-L10) @ XR  XL ]
```

这里的 `-L` 只包含严格下三角有效区；上三角和 padding 区在进入高风险计算前置零。

## 2. 对角叶子使用 Vector 递推

每个 32×32 叶子都是单位下三角矩阵。逐行递推：

```text
X[0,0] = 1
for i = 1 .. 31:
    X[i,:] = I[i,:] - sum(j=0 .. i-1, L[i,j] * X[j,:])
```

实现时将两个 32×32 叶子沿列拼成 `[32,64]`：

```text
src0 = [L00 | L11]
src1 = [I32 | I32]
dst  = [X00 | X11]
```

每次 Vector 处理一行，沿 K 维做乘法和归约，并用 scatter 将结果写回列偏移 `{0,32}`。`dst` 可以与 `src1` 原位复用，但每一行写完后必须建立 VEC store→load 的顺序，再让后续行读取已完成的前序行。

关键点：

- 只计算严格下三角，单位对角由 `I` 提供；
- 两个叶子共享一次向量归约，减少重复循环；
- 累加保持 FP32，最终按输出接口需要再转 BF16；
- scatter 索引和 `[32,64]` 打包布局必须与后续 NZ 搬运保持一致。

## 3. Cube 只承担块乘法

对角叶子逆完成后，Cube 执行两层 MBH：

1. `Y = I + LeafLeft @ (-L)`：先以 `I` 初始化 L0C，再累加第二个 MMAD；
2. `A = LeafLeft + Y @ LeafRight`：先计算 `I @ LeafLeft` 作为初值，再累加 `Y @ LeafRight`。

每个累加链都要明确 `cmatrixInitVal` 的首次初始化和后续累加状态。两次 MMAD 之间使用对应的 `M_FIX`/`FIX_M` 事件保护 L0C，Fixpipe 完成后再由 MTE2 回装后续 Cube 需要的 L1 数据。

对 64×64、FP32 的矩阵，L1 使用目标架构支持的 Cube NZ 布局；`I`、`-L`、`LeafLeft`、
`LeafRight` 和 `Y` 的 ND/NZ 转换在数据搬运阶段完成，避免在热路径使用逐元素转置。
Ascend950 的 FP32 C0 为 8；A2/A3 必须从 `Arch::AtlasA2` 的 layout、组件和目标版本接口重新确认
C0、搬运 stride、`LoadData2D` transpose 参数及 `FixpipeParamsV220`，不能复用 A5 常量。

## 4. 推荐的 kernel Stage

```text
Stage A：Vector
  生成严格下三角 -L
  打包两个 32×32 对角叶子
  Vector 递推求出 X00、X11

Stage B：AIC/Cube
  Y = I + LeafLeft @ (-L)
  Fixpipe / workspace / L1 回装

Stage C：AIC/Cube
  A = LeafLeft + Y @ LeafRight
  Fixpipe 写出最终 A
```

Stage 划分依据是数据依赖、矩阵布局和中间结果生命周期。AIV 负责叶子递推，AIC 负责块乘法；同一个执行单元承载多个逻辑阶段时，仍保留各阶段的精度检查点和交接边界。

## 5. 任务、尾块和流水

- 每个 `(sequence, chunk, value-head)` 是一个独立矩阵任务；`k' @ k'^T` 只需由每个 query-key head 的 owner 计算一次。A5 可在组件支持时通过 Fixpipe 向 AIV UB 共享；A2/A3 使用 `L0C -> GM -> UB`，两种路径都必须按 value-head owner 和有效区闭合 ready/free。
- 尾块统一在 UB 中补齐到 64，使用零值或单位矩阵的中性值完成主路径；最终只写回有效行。
- L1 中 `-L`、`Y` 和 `k'` 复用前，必须等待上一阶段最后消费者完成。下一 pack 可以在当前 pack 的后续输出阶段运行，但不能覆盖仍被 Cube 读取的数据。
- ready/free 通知按 task/slot 配对；AIV→AIC 的通知使用产生数据的 MTE3/Fixpipe pipe，AIC→AIV 的通知使用消费侧等待 pipe。AIV1 或 non-owner 仍按任务协议参与必要通知。

## 6. 精度和验证检查点

至少分别检查：

1. `KKT = k' @ k'^T` 的 FP32 结果；
2. `-L` 的严格下三角、上三角零区和 gate clip 后的有限性；
3. 两个对角叶子 `X00/X11` 的单位对角和下三角；
4. `Y` 的块乘结果；
5. 最终 `A` 的下三角、单位对角、上三角零区和尾块。

输入 gate 差值参与指数前先执行设计规定的 clip；clip、mask、BF16 cast 和 FP32 累加顺序必须与 CPU 标杆一致。矩阵求逆出现异常时，先区分叶子递推错误、块布局错误、MMAD 累加初始化错误和同步/复用错误。

## 7. 对当前 GDN2 intra 的适配提示

当前算子的 64×64 `A32` 可按相同 2×2 块结构设计候选 Cube 路径：保留现有 `T` 的数学语义，把 `A32` 的对角叶子递推和跨块乘法拆开；`safe_gate` 的 BF16 舍入仍在生成 `T` 或 `-L` 时按原标杆执行，不能在矩阵乘阶段重新解释。采用该路径前，需补齐 Cube NZ 地址、L0/L1 容量、Fixpipe 和 Stage 交接的具体证明，并用 CPU 标杆逐检查点回归。

[^catlass-arch-separation]: git:cann/catlass@0d78a73c192fc04741a4fd57c94080209424775c; paths: README.md, include/catlass/arch/arch.hpp, include/catlass/gemm/tile/copy_l0c_to_gm.hpp, include/catlass/gemm/tile/copy_l0c_to_ub.hpp, include/catlass/epilogue/tile/copy_ub_to_l1_tla.hpp

[^pr1069-matrix-inverse-patterns]: git:cann/cannbot-skills@4950cbd7c45e0d44cb8ec52598edf53cdc7f16ca:ops/catlass-linear-attention-workflow/references/matrix-inverse-patterns.md
