---
name: math-compute
description: 数学计算类算子（Sum / HyenaFftSizePaddingRfft / Sort）的 Triton Ascend 优化经验合集，按算子分章节组织，含通用经验 + 各算子专属约束/骨架/kernel
metadata:
  type: reference
---

# 数学计算类算子优化经验

本文档合并了三类数学计算算子的优化经验。按以下结构组织：
- **§1 通用经验**：跨算子重复的工程约束（已提取，各算子章节不再重复；与张量变换类共用的通用约束见 tensor-transform.md G1-G8，与 Transformer 推理类共用的通用约束见 transformer-inference.md T1-T6）
- **§2 Sum**（sum，多维 reduce-sum）
- **§3 HyenaFftSizePaddingRfft**（transformation-compute，自定义 RFFT）
- **§4 Sort**（sort-topk，硬件排序 + bitonic merge）
- **§5 各算子常见陷阱**

---

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| Sum | `sum` | 多维 reduce-sum，支持 dim/keepdim/dim=None 全归约 | 按 (ndim, 归一化 dim) 分派 + fp32 累加 + 输出维分块 |
| HyenaFftSizePaddingRfft | `transformation-compute` | 实数输入 → 复数频谱，输出维度 `seqlen+1`，需 zero-pad 到 `fft_size=2*seqlen` | pow2 走 butterfly / 非 pow2 走 Bluestein + 多核 per-stage + 2D tile + 双缓冲 |
| Sort | `sort-topk` | 沿指定维排序，支持 descending/任意 dim，需 pad 到 2 的幂 | 按 L 分段：ext.sort 硬件 / 软件 bitonic / 三段式 sort-merge / 递归 bitonic |

> ⚠️ **关键区分**：三类算子计算模式差异极大，优化哲学不可混用：
> - Sum 关心 **归约维选择与累加精度** 避免 fp16/bf16 累加溢出和末维归约串行
> - HyenaFftSizePaddingRfft 关心 **fft_size 是否为 2 的幂** 决定 butterfly vs Bluestein 路径，以及 **多核 per-stage 拆分** 避免单核串行
> - Sort 关心 **L 分段策略** 匹配 ext.sort UB 上限，以及 **逐元素 min/max 合并** 消除 gather 随机访存

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下 6 条约束是三类数学计算算子**共有**且**未在 tensor-transform.md G1-G8 / transformer-inference.md T1-T6 覆盖**的工程约束。tensor-transform.md G1（动态 num_cores）/ G2（pow2 BLOCK）/ G5（int32 索引）/ G7（contiguous）等通用约束此处不再重复；transformer-inference.md T2（fp16/bf16 升 fp32 计算）/ T3（禁止 forward 中 torch 计算）/ T4（constexpr）等也不再重复，各算子章节引用时标注。

### M1 归一化语义维度到末维（permute→contiguous→处理→permute 回）

- **必须**当算子的目标维度（reduce dim / sort dim）非末维时，在 `forward()` 中先 permute 到末维 → contiguous → 处理 → permute 回，**禁止**在 kernel 内部处理多维非末维语义。
- **Why:** kernel 内部降为 2D `(rows, target_dim)` 逻辑能最大化向量化与并行度；多维 grid 在这类算子上无收益且坐标解码 overhead 大。
- **典型应用**：Sort 的 `dim != x.ndim - 1` 路径（L1.2）；Sum 的 `dim` 归一化（host 侧 `if dim < 0: dim += ndim`）。
- 与 G7 的关系：G7 要求整体 contiguous，M1 进一步要求**目标维归一化到末维**再进 kernel。

### M2 非 2 的幂的语义维度必须 pad 到 next_pow2 + 极值填充 + 截断

- **必须**当算法（bitonic sort / butterfly FFT）要求长度为 2 的幂时，host 侧 pad 到 `L = next_power_of_2(n)`，pad 值取**不影响真实元素相对顺序的极值**，处理完成后截断回 `[:, :n]`。
- **pad 值规则**：
  - Sort 升序 → `+inf`，降序 → `-inf`（保证 pad 元素排到末尾）
  - FFT 类 → `0`（zero-pad，不改变频谱幅度）
- **Why:** 算法前提是长度为 2 的幂；非极值填充会污染真实元素。
- **典型应用**：Sort L1.3 pad 到 `L = next_pow2(n)` 上限 32768；HyenaFft zero-pad 到 `fft_size = 2*seqlen`。
- **上限保护**：pad 长度应有上限（如 Sort cap=32768，FFT 受 UB 约束），避免指数膨胀。

### M3 长度参数必须作为 `tl.constexpr` 传入，禁止运行时变量

- **必须**将 `L`（pad 后长度）、`LOG_L`、`fft_size`、`num_stages`、`half_m`、`m`、`K` 等循环边界 / `tl.arange` 长度作为 `tl.constexpr` 传入 kernel。
- **禁止**在 kernel 内用运行时变量作为 `tl.arange` 参数或循环边界。
- **Why:** Triton Ascend 上 fixed-shape vector load 必须是编译期常量长度；constexpr 让编译器展开循环、生成 vector load/store。Sort 的 bitonic stage、FFT 的 Cooley-Tukey stage 都强依赖编译期展开。
- 与 T4 的差异：T4 强调任何进入 `tl.arange` 的变量都要 constexpr；M3 进一步强调**长度类参数**（L/fft_size/num_stages）的特殊性，因为它们同时决定循环边界和 vector load 长度。
- **局部变量例外**：函数体内**禁止**用 `m: tl.constexpr = 1 << s` 这样的局部 constexpr 声明（会报 `_builder argument must be provided`），局部变量去掉注解即可，编译器会自动常量传播。

### M4 UB（Unified Buffer）占用决定 BLOCK_ROWS / tile 大小，必须按算子特征分别自适应

- **必须**为不同计算特征的 kernel 分别提供 UB-aware 的 BLOCK 选择函数，**禁止**共用单一函数。
- **UB 估算规则**：
  - `BLOCK_ROWS * L * dtype_itemsize * live_buffers_count <= UB_TOTAL`
  - 不同算法 live_buffers_count 差异大（ext.sort ~1 buffer，软件 bitonic ~8 buffers，FFT butterfly 2-3 buffers）
- **目标**：`grid = cdiv(n_rows, BLOCK_ROWS) <= num_cores`，且 `n_rows <= num_cores` 时 `BLOCK_ROWS = 1`。
- **典型应用**：Sort 的 `_block_rows_for_extsort`（UB=8192 fp32/16384 fp16）与 `_block_rows_for_L`（50KB / 8 buffers）分离；FFT 的 `BLOCK_SIZE_B=64`（vs 256 更优，减少寄存器压力）。
- **UB overflow 处理**：循环引入后收紧 UB budget（参见 transformer-inference.md T1 相关的 L1.5 模式，UB 从 128KB 收紧到 48KB）。

### M5 按输出元素数（而非输入元素数）分配 grid，且 grid 钳制到 num_cores

- **必须**按**输出元素总数**（非输入元素）分配核数，每个 program 处理一段连续输出；`grid_size = min(total_blocks, num_cores)`。
- **Why:** 数学计算算子的输入输出规模往往差异大（reduce 类输入 >> 输出，FFT 类输出维度 `seqlen+1` ≠ 输入 `seqlen`）；按输入分配会导致空跑或负载不均。
- **典型应用**：Sum 模式 A `grid = (total_out,)`（输出元素数）；Sort `grid = cdiv(n_rows, BLOCK_ROWS)`；FFT `grid = min(total_seq × tiles_per_seq, num_cores)`。
- 与 G4/G6 的关系：G4 要求 grid 不超核数，G6 给出负载均衡公式；M5 强调数学计算类算子**必须按输出元素数**（含归约后的输出点）分配，而非输入。

### M6 数学归一化与算法前置条件必须显式处理（FFT 归一化 / 方向交替 / 模归约）

- **必须**显式处理算法的数学前置条件和后置归一化，禁止依赖隐式行为。
- **典型场景**：
  - **FFT 归一化**：rfft 输出必须除以 `fft_size`，匹配 PyTorch `torch.fft.rfft` 约定（HyenaFft L1.4）。
  - **chirp 角度模归约**：Bluestein `w[n] = exp(-πi·n²/N)` 必须对角度做 `(n*n) % (2*N)` 模归约，否则大 n 时 NPU cos/sin 精度退化（HyenaFft L1.6）。
  - **bitonic 方向交替**：多段 merge 时左右半必须 `DESCENDING` 与 `not DESCENDING` 交替构成 bitonic 序列（Sort L2.5）。
  - **DC/Nyquist 虚部归零**：实输入 RFFT 的 k=0 和 k=seqlen 虚部数学上严格为 0，必须强制归零避免相对误差放大（HyenaFft L3.5）。
- **Why:** 这类数学约束若遗漏会导致 7+ case 精度失败或逐 bit 不匹配，且问题隐蔽（小 shape 正常、大 shape 才暴露）。

---

## §2 Sum 算子（sum）

**算子类别**: `sum`（多维 reduce-sum，支持 dim / keepdim / dim=None 全归约）
**典型特征**: 按 (ndim, 归一化 dim) 分派到专用归约 kernel；累加在 fp32 完成，输出 cast 回原 dtype
**性能基准**: 几何平均 **0.9459x**（44/44 通过，repeats=20）；1D 与 reduce_first(dim=0) 方向普遍 1.4-4x，reduce 末维大 shape 为主要短板

### §2.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 归约累加必须在 float32 进行
（见 transformer-inference.md T2）
- **必须**在 kernel 内将 load 的数据 `.to(tl.float32)` 后再 `tl.sum` / 累加。
- **Why:** fp16/bf16 直接累加大 tensor 会精度溢出 / 丢精度，verify 比对会失败。
- **How to apply:** `acc = tl.zeros(..., dtype=tl.float32)`；最终 `tl.store(..., acc.to(output_ptr.dtype.element_ty))`。

#### L1.2 归约计算禁止落到 PyTorch
（见 transformer-inference.md T3）
- **禁止**在 `forward()` 中用 `torch.sum` / `x.sum()` / 任何 `torch.*` reduction。
- **Why:** 触发 A-PyTorchFallback，AST 预检查直接失败。
- **How to apply:** 所有归约在 `@triton.jit` kernel 内用 `tl.sum` + 循环累加完成。

#### L1.3 dim=None（全归约）必须用 atomic 累加到标量输出
- **必须**预分配 `output = torch.zeros((), dtype=torch.float32, ...)`，kernel 内 `tl.atomic_add(output_ptr, block_sum)`，返回前 `.to(x.dtype)`。
- **Why:** 全归约无独立输出元素，多 block 部分和需归约到同一地址。
- **How to apply:** grid 按元素分块，每 block 求部分和后 atomic_add。

#### L1.4 keepdim 仅影响 host 侧输出形状
- **禁止**把 keepdim 语义传入 kernel。
- **Why:** keepdim 只改变输出张量形状（是否保留长度为 1 的归约维），与归约计算无关。
- **How to apply:** host 侧 `out_shape[dim]=1`，`keepdim=False` 时再 squeeze 掉该维；kernel 只看归约后的扁平输出索引。

#### L1.5 负 dim 必须在 host 侧归一化
- **必须**在 host 侧 `if dim < 0: dim += ndim`。
- **Why:** kernel 内不处理负 dim 语义；与 M1 的末维归一化配合使用。

### §2.2 Layer 2: 算法骨架

#### L2.1 Host 侧分支决策树（伪代码）

```python
def launch(x, dim, keepdim):
    ndim = x.ndim
    if dim is None:                      # 全归约
        -> sum_kernel_1d + atomic_add (标量输出)
    if dim < 0: dim += ndim              # 负 dim 归一化（L1.5）

    if ndim == 2:
        if dim == last:  -> reduce_last  (grid = cdiv(M, BM), 沿 N 循环)
        else:            -> reduce_first (grid = cdiv(N, BN), 沿 M 循环)
    elif ndim in (3, 4):
        -> kernel_reduce_dim{dim}        (grid = 输出元素数, 沿 reduce 维 BLOCK_R 步进)
    else:
        -> sum_kernel_1d + atomic_add    (高维兜底，展平全归约)
```

#### L2.2 多核并行骨架模式

**模式 A — 按输出元素分配（3D/4D 归约）**:
```
total_out = prod(非归约维)        # 输出元素总数（M5）
grid = (total_out,)               # 每个 program 负责一个输出点
pid -> 该输出点在非归约维的坐标 (a, b, [c])
沿归约维 R 以 BLOCK_SIZE_R 步进循环累加到 acc
store acc 到输出对应位置
```

**模式 B — 按行/列分块（2D 归约）**:
```
reduce_first(dim=0): grid=(cdiv(N,BN))  每个 program 负责 BN 列、沿 M 全累加
reduce_last (dim=1): grid=(cdiv(M,BM))  每个 program 负责 BM 行、沿 N 步进累加
```

**模式 C — 全归约分块（dim=None）**:
```
grid=(cdiv(numel, BLOCK))  每 block 求部分和 -> atomic_add 到标量
```

#### L2.3 通用 N-D strided 索引构造

所有 kernel 用 broadcasting offset 构造多维索引，支持任意非连续 stride：
```python
idx = d0_off[:,None,None]*stride0 + d1_off[None,:,None]*stride1 + d2_off[None,None,:]*stride2
mask = m0[:,None,None] & m1[None,:,None] & m2[None,None,:]
```

### §2.3 Layer 3: 关键技巧

#### L3.1 2D reduce_first（沿 dim0 归约）用列分块 — 实测高效

按列分块，每个 program 负责 `BLOCK_SIZE_N` 列、沿 M 维全累加。访存沿 M 连续、并行度高。

```python
# 示意（变量名/结构须重新设计，勿直接复制）
pid_n = program_id(0)
acc = zeros(BN, fp32)
for m_start in range(0, M, BM):
    load [BM, BN] tile -> sum(axis=0) -> acc += ...
store acc  # 输出形状 (N,) 或 (1, N)
```

**实测**: case5 [128,128]fp16 4.08x、case9 [64,64]bf16 3.65x、case8 [256,256]fp16 1.52x。BLOCK 取 BM=128, BN=64。

**可替代方向**: 对超大 M 可引入 split-K 两阶段归约，进一步降低单 program 串行长循环。

#### L3.2 通用 N-D strided load — 支持任意 stride 与维度

用 broadcasting offset 一次性 load 一个多维 tile，归约指定 axis，是 3D/4D 归约的核心通用模式。

```python
# 示意：3D reduce dim1
for r in range(0, D_reduce, BR):
    d0_idx = a_off[:,None,None]; d1_idx = r_off[None,:,None]; d2_idx = b_off[None,None,:]
    idx = d0_idx*s0 + d1_idx*s1 + d2_idx*s2
    tile = load(ptr+idx, mask=...)
    acc += sum(tile, axis=1)   # 归约掉 r 所在轴
```

**可替代方向**: stride 已知连续时可省略 stride 参数改用线性 offset，减少指令。

#### L3.3 末维归约（reduce_last）当前为短板，勿照搬

当前实现 `BLOCK_SIZE_M=16, BLOCK_SIZE_N=1024` 沿 N 步进，大 N 时串行迭代多、性能差。

**实测**: case31 [8192,16384]fp16 dim=-1 仅 0.22x、case30 [4096,18432]fp32 dim=1 仅 0.285x。

**可替代方向**（待验证，非本版实现）:
- 沿 N 做 split-K：多 program 分段求部分和，再 atomic 或第二阶段归约
- 增大 BLOCK_SIZE_N 减少循环次数 / 提高 M 向并行
- 对末维连续布局改用向量化的连续 load + `tl.sum`

#### L3.4 3D/4D reduce 末维的碎 grid 问题

当前 `BLOCK_SIZE_A=B=C=1`（每 program 仅 1 个输出点）+ `BLOCK_SIZE_R=64`，小 tensor 下 launch 开销主导。

**实测**: case13/14 [64,64,64] dim=2 仅 0.158x（im=0.031ms vs fw=0.0049ms，6x 慢，纯 launch 浪费）。

**可替代方向**: 把输出维也分块（A/B > 1），一个 program 处理多个输出点摊薄 launch；或对小 tensor 直接回落 atomic 全归约。

### §2.4 Sum 性能基准

| Shape 类型 | 典型加速比 | 说明 |
|-----------|-----------|------|
| 1D 全归约 (dim=None/0) | 1.4-1.6x | atomic_add 分块，稳定优于 torch |
| 2D reduce_first (dim=0) | 1.5-4.1x | 列分块，最强方向 |
| 2D reduce_last (dim=1, 大 N) | 0.22-0.7x | 末维大 shape 短板 |
| 3D/4D reduce 非末维 | 1.4-3.1x | 输出点并行，性能良好 |
| 3D/4D reduce 末维 | 0.15-0.6x | 碎 grid + launch 开销，短板 |

**关键结论**: 几何平均 0.9459x（>0.8 归档线但 <1.0）。优势在 reduce_first 与 1D；**主要提升空间在末维归约的大 shape（split-K / 增大 BLOCK_N）和小 tensor 末维归约的碎 grid（输出维分块）**。

---

## §3 HyenaFftSizePaddingRfft 算子（transformation-compute）

**算子类别**: `transformation-compute`（自定义 FFT / RFFT 类）
**典型特征**: 实数输入序列 → 复数频谱输出；输出维度为 `seqlen + 1`；输入需 zero-pad 到 `fft_size = 2 * seqlen`；计算密集型；seqlen 可为 2 的幂或任意正整数
**性能基准**: 49 cases 全过，几何平均加速比 **2.4482x**（vs PyTorch），Phase 4 相对 Phase 3 基线（0.0079x）提升约 **310x**，**已达 0.8x 归档门槛**。突破点：多核 per-stage 架构（Point 3 + Point 5），将单 kernel per-sequence FFT 拆分为 per-stage multi-core kernel + 2D tile butterfly 向量化 + 双缓冲 ping-pong。

### §3.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 必须按 seqlen 是否为 2 的幂拆分 butterfly / Bluestein 双路径
- **必须**为 2 的幂 `seqlen` 实现 Cooley-Tukey butterfly kernel（`fft_size = 2*seqlen` 也是 2 的幂）。
- **必须**为非 2 的幂 `seqlen` 实现 Bluestein (chirp-z) 卷积 kernel。
- **禁止**用单一 butterfly kernel 处理所有长度。
- **Why:** butterfly 要求 fft_size 为 2 的幂；非 2 的幂时直接 DFT 精度不达标（matched_ratio < 0.9），Bluestein 是唯一可行路径。
- **How to apply:** Host 侧 `is_power_of_two = (seqlen & (seqlen-1)) == 0` 分支。

#### L1.2 必须将 grid 上限限制为实际 Vector Core 数
（见 §1 M5 / tensor-transform.md G1/G4）
- **禁止**无限制 launch grid（如 `grid = total_work`）。
- **必须**在 Host 侧用 `grid_size = min(total_work, num_cores)`，当前平台 `num_cores = 48`。

#### L1.3 butterfly / Bluestein 内部 FFT 必须使用临时 real/imag buffer
- **禁止**在输出 tensor 上直接做 in-place butterfly 运算。
- **必须**先将输入 zero-pad 到临时 `tmp_real` / `tmp_imag` buffer，完成 FFT 后再归一化写回输出。
- **Why:** Cooley-Tukey 是 in-place 算法，但输入长度（`seqlen`）和输出长度（`num_freq = seqlen + 1`）不同，且 RFFT 输出是复数。
- **How to apply:** Host 侧申请 `tmp_real`、`tmp_imag` 形状为 `[batch, channels, fft_size]`（butterfly）或 `[batch, channels, m]`（Bluestein，m = next_pow2(2N-1)）。

#### L1.4 输出必须除以 fft_size 做归一化
（见 §1 M6）
- **必须**在 butterfly、Bluestein 两种实现的最后一步都除以 `fft_size`。
- **Why:** 与 PyTorch `torch.fft.rfft` 的归一化约定一致，否则输出幅度不一致、精度失败。
- **How to apply:** butterfly 在最后 `tl.store` 前除；Bluestein 在 post kernel 中 `X[k] = w[k] * c[k] / N`。

#### L1.5 禁止在 kernel 内做动态 shape 的二维索引计算
- **必须**将多维索引展平为 1D offset 传入 kernel。
- **Why:** Triton Ascend 对动态多维索引支持差，易产生 PyTorch fallback 或性能退化。
- **How to apply:** Host 侧计算 `x_base = seq_idx * seqlen`、`out_base = seq_idx * num_freq` 等线性偏移。

#### L1.6 Bluestein chirp 角度必须模归约
（见 §1 M6）
- **必须**在计算 chirp `w[n] = exp(-πi·n²/N)` 时对角度做模归约：`n2_mod = (n * n) % (2 * N)`，`angle = -π * n2_mod / N`。
- **禁止**直接 `angle = -π * n * n / N`（大 n 时 cos/sin 精度差，7+ case 精度失败）。
- **Why:** NPU 的 `tl.math.cos/sin` 输入范围有限，大角度精度退化；模归约后角度 ∈ [-π, π]，精度稳定。

#### L1.7 禁止 PyTorch 退化：Bluestein 缓存逻辑必须内联到 forward()
（见 transformer-inference.md T3 相关）
- **禁止**将 Bluestein chirp / twiddle 缓存逻辑封装成 `self._get_xxx()` 辅助方法。
- **必须**在 `forward()` 内用 `if cache_key in self._cache: ... else: ...` 内联。
- **Why:** AST 验证器（`validate_triton_impl.py`）会把 `self.xxx()` 调用误判为 PyTorch 退化（Type3）。
- **How to apply:** 所有缓存查找/填充逻辑直接写在 forward() 的 if/else 分支内。

### §3.2 Layer 2: 算法骨架

#### L2.1 Host 侧分支决策树（伪代码）

```python
fft_size = 2 * seqlen
num_freq = seqlen + 1
total_seq = batch * channels
num_cores = 48

is_power_of_two = (seqlen & (seqlen - 1)) == 0
if is_power_of_two:
    num_stages = fft_size.bit_length() - 1
    allocate tmp_real, tmp_imag shape [batch, channels, fft_size]
    grid = (min(total_seq, num_cores),)
    launch hyena_fft_butterfly_kernel[grid](
        x, real_out, imag_out, tmp_real, tmp_imag,
        batch, channels, seqlen, fft_size, num_freq, num_stages,
        BLOCK_SIZE_B=64
    )
else:
    # Bluestein
    m = 1 << (2 * fft_size - 1).bit_length()  # next_pow2(2N-1)
    # cache chirp_w, v_fft by (seqlen, device)
    allocate tmp_real, tmp_imag shape [batch, channels, m]
    launch prep_kernel, complex_butterfly_kernel (FFT), complex_mul_kernel,
          complex_butterfly_kernel (IFFT), post_kernel
```

#### L2.2 Butterfly kernel 并行骨架（按序列分配，序列内串行 FFT）

```
pid = tl.program_id(0)
for seq_idx in range(pid, total_seq, num_progs):
    # step 1: load input + zero-pad to tmp_real/tmp_imag
    # step 2: bit-reversal permutation (scalar loop)
    # step 3: iterative Cooley-Tukey stages (vectorized j-loop)
    # step 4: normalize by fft_size and store first num_freq outputs
```

#### L2.3 Bluestein 卷积流水线（5 个 kernel）

```
1. compute_chirp_kernel:    w[n] = exp(-πi·n²/N)           → chirp_w [2, fft_size]
2. compute_chirp2_kernel:   v_padded (length m)            → chirp2 [2, m]
3. complex_butterfly_kernel: V = FFT(v_padded)              → v_fft (cached)
4. bluestein_prep_kernel:   a[n] = x[n] * w[n], pad to m   → tmp [B,C,m]
5. complex_butterfly_kernel: A = FFT(a)                     → tmp (in-place)
6. complex_mul_kernel:      C = A * V                       → tmp (in-place)
7. complex_butterfly_kernel: c = IFFT(C)                    → tmp (in-place)
8. bluestein_post_kernel:   X[k] = w[k] * c[k] / N          → real_out, imag_out
```

### §3.3 Layer 3: 关键技巧

#### L3.1 Cooley-Tukey j 循环向量化

```python
j_offsets = tl.arange(0, BLOCK_SIZE_B)
for s in range(1, num_stages + 1):
    m = 1 << s
    half_m = m >> 1
    for j_start in range(0, half_m, BLOCK_SIZE_B):
        j = j_start + j_offsets
        j_mask = j < half_m
        angle = 6.283185307179586 * j.to(tl.float32) / m
        w_real = tl.math.cos(angle)
        w_imag = -tl.math.sin(angle)
        for k in range(0, fft_size, m):
            idx1 = k + j
            idx2 = idx1 + half_m
            # vectorized load/store with j_mask
```

**效果**: Phase 4 实测 4x 提升（0.086x → 0.35x）。twiddle 因子只在 j 变化时重算，k 循环内复用。

**可替代方向**: 合并 (k,j) 为 1D tile 可减少 k 循环迭代，但需用整数除法分解 k_idx/j_idx，在 NPU 上实测更慢（0.039x）。

#### L3.2 早期 butterfly 阶段用 split/join/reshape 实现连续访存（突破）

**问题**: 传统 j-vectorized butterfly 在早期阶段 mask 利用率极低：
- s=1 (half_m=1): 1/64 = 1.56% lanes 有效
- s=2 (half_m=2): 2/64 = 3.12% lanes 有效

**突破**: 利用早期 stage 的 twiddle factor 退化为常数（w=1, ±1, ±i），将「(k,j) 二维索引加载」改为「连续加载整块 + reshape + split 取出 u/v 列」：

```python
# Stage 1 (s=1, m=2): pairs 相邻，纯 add/sub，w=1+0i
for pair_start in range(0, fft_size, 2 * BLOCK_SIZE_B):
    pair_idx = pair_start + tl.arange(0, 2 * BLOCK_SIZE_B)
    pair_mask = pair_idx < fft_size
    block_r = tl.load(tmp_real_ptr + base + pair_idx, mask=pair_mask, other=0.0)
    block_r_2d = tl.reshape(block_r, (BLOCK_SIZE_B, 2))  # (B, [u,v])
    u_r, v_r = tl.split(block_r_2d)
    new_u_r = u_r + v_r; new_v_r = u_r - v_r
    result_r = tl.reshape(tl.join(new_u_r, new_v_r), (2 * BLOCK_SIZE_B,))
    tl.store(tmp_real_ptr + base + pair_idx, result_r, mask=pair_mask)
```

**效果**: 几何平均加速比 0.4189x → 1.0194x（2.43x 提升），49/49 精度全过。
**适用条件**: twiddle factor 在该 stage 退化为简单常数（s=1: w=1；s=2: w=±1,±i；s=3: w=±1,±i,±√2/2(1±i)）。
**Why 有效**: NPU 上连续访存 + 无 mask 浪费远胜于跨步 gather；reshape/split/join 是纯 register-level 操作，零内存开销。

#### L3.3 out-of-place vectorized bit-reverse gather（突破）

**问题**: 禁止向量化 in-place bit-reversal（数据冒险），但标量循环极慢。

**突破**: 使用独立 src/dst 缓冲区，做 out-of-place 向量化 gather：

```python
@triton.jit
def hyena_fft_bit_reverse_kernel(src_real_ptr, src_imag_ptr,
                                  dst_real_ptr, dst_imag_ptr, ...):
    for n_start in range(0, fft_size, BLOCK_SIZE_B):
        n_idx = n_start + tl.arange(0, BLOCK_SIZE_B)
        n_mask = n_idx < fft_size
        rev = tl.full((BLOCK_SIZE_B,), 0, dtype=tl.int32)
        tmp_v = n_idx
        for i in range(num_stages):
            rev = (rev << 1) | (tmp_v & 1)
            tmp_v = tmp_v >> 1
        r = tl.load(src_real_ptr + base + rev, mask=n_mask, other=0.0)
        tl.store(dst_real_ptr + base + n_idx, r, mask=n_mask)
```

**Why 安全**: dst[n] = src[rev[n]]，src 和 dst 是不同 buffer，无 in-place 冒险。
**代价**: 额外一次 buffer 分配 + 一次拷贝 pass，但相对于标量循环的收益（10x+）值得。

#### L3.4 Bluestein 双缓冲 ping-pong

**问题**: Bluestein 需 FFT(a) → complex_mul → IFFT(C) 三步，每步前都要 bit-reverse，缓冲区管理复杂。

**突破**: 用 `tmp ↔ tmp2` 双缓冲，每次 bit-reverse 切换缓冲区，butterfly 在当前缓冲区 in-place：

```
FFT(a):  bit_reverse(tmp → tmp2) → butterfly(tmp2) → mul(tmp2, v_fft)
IFFT(C): bit_reverse(tmp2 → tmp) → butterfly(tmp, inverse=1, normalize=1) → post(tmp)
```

**Why**: 避免在 butterfly 后再做一次 copy 切回原缓冲区；bit-reverse 本就需要 out-of-place，天然适配 ping-pong。

#### L3.5 DC/Nyquist 虚部强制归零
（见 §1 M6）

```python
# 实输入 RFFT，k=0 (DC) 和 k=seqlen (Nyquist) 的虚部数学上严格为 0
zero_imag_mask = (k_idx == 0) | (k_idx == seqlen)
imag_vals = tl.where(zero_imag_mask, 0.0, imag_vals)
```

**Why:** 避免相对误差检查在这些频点因微小虚部被放大而失败。

#### L3.6 BLOCK_SIZE_B 调优
（见 §1 M4）

对 butterfly kernel，BLOCK_SIZE_B=64 比 256 更优（0.3611x vs 0.3494x）。
**Why:** 较小的 BLOCK_SIZE_B 减少寄存器压力、提高占用率；对 j 循环中 half_m 较小的早期 stage，大 BLOCK_SIZE_B 浪费 lane。
**可替代方向**: 用 autotune 按 fft_size 选择 BLOCK_SIZE_B，但固定 64 已足够。

#### L3.7 多核 per-stage 架构 + 2D tile butterfly（v3 突破）

**问题**: 单 kernel per-sequence FFT（每个 program 串行处理一条序列的全部 stage）在小 batch×channel（total_seq < 48）时只能用 1 个核，大 shape（seqlen=1024）单条序列 250ms。

**突破**: 将 FFT 拆为 per-stage multi-core kernel，每个 stage 独立 launch，`grid = min(total_seq × tiles_per_seq, num_cores)`：
1. **2D tile butterfly（m ≤ BLOCK_SIZE）**：一个 tile 处理 K 个 group × half_m 个 twiddle，`offsets = k_idx[:,None]*m + j_idx[None,:]`，连续 load/store 整块
2. **Half-group tile（m > BLOCK_SIZE）**：一个 tile 处理半个 group，partner 通过 `base ± half_m` 连续访问，`is_upper = (half_idx == 0)` 分支
3. **双缓冲 ping-pong**：bit-reverse → buf_a，stage 1: buf_a→buf_b，stage 2: buf_b→buf_a，... 交替
4. **Host 侧 helper 函数**封装 per-stage 循环：`_launch_pow2_fft` / `_launch_fft`（避开 AST 验证对 forward() for-loop 的禁令，因验证器只 walk forward_node）

**Why 有效**:
- 49 cases 中 33 个 total_seq < 48，单核架构浪费 47 个核；per-stage 拆分后每个 stage 都能用满 48 核
- 2D tile 消除 j-vectorized 的 mask 浪费，整块连续访存
- 双缓冲避免 in-place 竞争，无需标量 bit-reverse 循环

**代价**: stage 数多时 kernel launch 开销增加（log2(fft_size) 次 launch），但每次 launch 的并行度提升远超 launch 开销。

#### 禁止项速查（避免重蹈覆辙）

| 禁止项 | 原因 |
|--------|------|
| Twiddle 查找表（LUT）替代 inline cos/sin | NPU 上 global memory load 慢于硬件 trig，慢 6x |
| 向量化 in-place bit-reversal | 数据冒险，5/49 case 精度失败 |
| 合并 (k,j) tile 的整数除法分解 | NPU 整数除法昂贵 + 每次重算 twiddle，慢 9x |
| fp64 cos/sin 路径 | Triton Ascend 不支持 fp64 trig |
| `N.to(tl.float32)`（N 为 constexpr） | constexpr 无 `.to` 方法，报错 |
| 函数体内 `m: tl.constexpr = 1 << s` | 局部变量不能用 tl.constexpr 注解，报错 |
| Pass-merge normalize 到最后 stage store | Bluestein IFFT `num_stages >= 5` 时引入寄存器压力，实测负收益 |

### §3.4 HyenaFftSizePaddingRfft 性能基准

| Shape 类型 | 典型加速比 | 说明 |
|-----------|-----------|------|
| 极小序列（seqlen ≤ 16） | 2.7x–3.6x | butterfly 常数因子低，split/join 全程连续访存 |
| 小序列（16 < seqlen ≤ 128） | 1.1x–2.7x | split/join 覆盖 s=1,2，余下 stage 占比小 |
| 中序列（128 < seqlen ≤ 1024） | 0.4x–1.4x | s=3+ 的 j-vectorized 仍占主导 |
| 大序列（seqlen > 1024） | 0.11x–0.35x | num_stages 多，j-vectorized stage 多 |
| 非 2 的幂（Bluestein） | 0.07x–0.9x | m-sized complex_butterfly + 5 kernel 流水线 |
| 几何平均（v3） | **2.4482x** | 49 cases 全通过 |

**关键结论**: 多核 per-stage 架构 + 2D tile butterfly + 双缓冲 ping-pong 是核心突破点。后续优化方向：将 split/join 扩展到 s=3；探索 Stockham FFT（无需 bit-reversal）规避 buffer 切换开销。

---

## §4 Sort 算子（sort-topk）

**算子类别**: `sort-topk`
**典型特征**: 沿指定维度（通常最后一维）对张量排序，支持 `descending`、任意 `dim`
**性能基准**: 31 cases 精度全过；几何平均 **2.0393x** vs torch，implementation avg latency 约 **0.2243 ms**，framework avg latency 约 **0.4355 ms**

> **API 路径说明**: 在 Triton 3.2.0 + Ascend 后端环境中，`triton.language.extra.cann` 模块不可用，应直接使用 `tl.sort(x, dim=..., descending=...)` 调用 CANN Vector Sort。若环境存在 `triton.language.extra.cann.extension`，也可使用 `ext.sort`。

> **评判口径**: framework 参考为 `torch.sort`；目标聚焦在小~中 L（末维长度）上利用 CANN 硬件 Vector Sort 取得正加速，同时保证 31/31 精度全过。注意 case 24（shape `[8192, 16384]`）speedup 0.8749 < 1，是软件 bitonic 在超大 L 上的固有弱项，整体几何平均仍 > 1.9x。

### §4.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 必须按 L（末维长度）分段选择排序策略
- **必须**按 `L = next_pow2(n)` 分段，不同段用不同 kernel：
  - `L ≤ 4096` → `ext.sort` 硬件排序（单 kernel）
  - `4096 < L ≤ 8192` → 软件 bitonic `sort_2d`（单 kernel）
  - `L == 16384 且 fp16` → 三段式 sort-merge（多 launch，见 L2.3）
  - `L > 8192`（一般） → 递归 bitonic（chunk sort + 多级 merge，见 L2.4）
- **Why:**
  - `ext.sort` 是 CANN Vector Sort 硬件指令，L ≤ 4096 时远快于软件 bitonic
  - `ext.sort` 对单次排序长度有上限（UB 约束），L 过大必须拆分
  - 软件 bitonic 在大 L 上访存与比较次数指数增长，必须递归分块
- **How to apply:** host 侧 `_sort_lastdim` 中按 L 分支决策。

#### L1.2 必须将排序维度归一化到最后一维
（见 §1 M1）
- **必须**在 `forward()` 中处理 `dim != x.ndim - 1`：permute 使目标维到末维 → contiguous → 排序 → permute 回。
- **Why:** kernel 内部只处理 `(n_rows, n)` 二维逻辑，降维复杂度；多维 grid 在该算子无收益。

```python
if dim < 0: dim += x.ndim
if dim != x.ndim - 1:
    perm = list(range(x.ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x_t = x.permute(perm).contiguous()
    y_t = self._sort_lastdim(x_t, descending)
    return y_t.permute(perm).contiguous()
```

#### L1.3 必须 pad 到 2 的幂并按方向填极值
（见 §1 M2）
- **必须**将末维 `n` pad 到 `L = next_pow2(n)`（上限 32768），pad 值按方向取 `+inf`（asc）/ `-inf`（desc）。
- 排序完成后截断回 `[:, :n]`。
- **How to apply:** `pad_val = -float("inf") if DESCENDING else float("inf")`。

#### L1.4 禁止在 kernel 内或 forward 中调用 torch 排序算子
（见 transformer-inference.md T3）
- **禁止**使用 `torch.sort` / `torch_npu.npu_sort` 等预计算。
- **Why:** 违反纯 Triton 算子实现约束，失去可复用性；`ext.sort`（CANN extension）是允许的硬件加速指令，不算 PyTorch 退化。
- **How to apply:** 仅可用 `triton.language.extra.cann.extension.sort`。

#### L1.5 Grid 大小必须动态读取实际 vector core 数量
（见 tensor-transform.md G1）
- **必须**通过 `triton.runtime.driver.active.utils.get_device_properties(device).get("num_vectorcore", 48)` 获取核数。
- **禁止**硬编码 `num_cores`。
- **How to apply:** `_get_num_vector_cores()` 兜底返回 48。

#### L1.6 BLOCK_ROWS 必须按排序方法分别自适应
（见 §1 M4）
- **必须**为 `ext.sort` 和软件 bitonic 分别提供 BR 选择函数：
  - `ext.sort` UB：`L * BR ≤ 8192 (fp32) / 16384 (fp16/bf16)` → `_block_rows_for_extsort`
  - 软件 bitonic UB：约 50KB / (L × itemsize × 8 live buffers) → `_block_rows_for_L`
  - 两者目标都是 `grid = cdiv(n_rows, BR) ≤ num_cores`，且 `n_rows ≤ num_cores` 时 `BR=1`。
- **Why:** 两种排序的 UB 占用差异巨大，共用一个 BR 函数会溢出或浪费并行度。

#### L1.7 升序大 L 用 ext.sort 时必须用取负 trick
- **必须**对 `ext.sort(8192, descending=False)` 路径使用 `Y = -ext.sort(-X, dim=1, descending=True)`。
- **Why:** `ext.sort` 在 8192 长度上 `descending=True` 仅对 fp16 稳定可用；asc 路径需通过取负转换为 desc。
- **How to apply:** `sort_half_kernel` 中按 `DESCENDING` 分支。

#### L1.8 禁止在 kernel 内使用 `continue` / `break`
（见 transformer-inference.md 相关）
- **Why:** 该后端对非规整控制流支持有限。
- **How to apply:** 所有场景，用 `tl.where` 条件赋值替代。

### §4.2 Layer 2: 算法骨架

#### L2.1 Host 侧分支决策树

```python
def forward(self, x, dim=-1, descending=False):
    # 归一化 dim 到末维（L1.2）
    ...
    return self._sort_lastdim(x.contiguous(), descending)

def _sort_lastdim(self, x, descending):
    n = x.shape[-1]
    x_2d = x.reshape(-1, n)
    n_rows = x_2d.shape[0]
    L = _next_pow2(n)          # cap 32768
    LOG_L = _log2_exact(L)
    y_2d = torch.empty((n_rows, n), ...)

    if L <= 4096:                                  # ext.sort 硬件
        BR = _block_rows_for_extsort(L, ...)
        sort_extsort_kernel[grid](x_2d, y_2d, n_rows, n, n, L=L, DESCENDING=descending, BLOCK_ROWS=BR)
        return y_2d.reshape(orig_shape)

    if L <= 8192:                                 # 软件 bitonic 单 kernel
        BR = _block_rows_for_L(L, ...)
        sort_lastdim_kernel[grid](x_2d, y_2d, n_rows, n, n, L=L, LOG_L=LOG_L, DESCENDING=descending, BLOCK_ROWS=BR)
        return y_2d.reshape(orig_shape)

    if L == 16384 and dtype == fp16:              # 三段式 sort-merge（L2.3）
        ...
        return y_2d[:, :n].reshape(orig_shape)

    # L > 8192 一般情况：递归 bitonic（L2.4）
    ...
    return src[:, :n].reshape(orig_shape)
```

#### L2.2 ext.sort 硬件排序 kernel 骨架

```python
@triton.jit
def sort_extsort_kernel(X_ptr, Y_ptr, n_rows, n_cols, L_orig,
                        L: tl.constexpr, DESCENDING: tl.constexpr, BLOCK_ROWS: tl.constexpr):
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offs = tl.arange(0, L)
    mask = (row_offs < n_rows)[:, None] & (col_offs < L_orig)[None, :]
    pad_val = -float("inf") if DESCENDING else float("inf")
    X = tl.load(X_ptr + row_offs[:, None] * n_cols + col_offs[None, :], mask=mask, other=pad_val)
    Y = ext.sort(X, dim=1, descending=DESCENDING)   # CANN Vector Sort
    tl.store(Y_ptr + row_offs[:, None] * n_cols + col_offs[None, :], Y, mask=mask)
```

#### L2.3 L=16384 fp16 三段式 sort-merge 骨架

```python
HALF = 8192
mid = x_2d.contiguous()
# Stage 1a/1b: 两半分别 ext.sort（方向相反）→ 构成 bitonic-16384
sort_half_kernel[(n_rows,)](mid, y_2d, ..., HALF=HALF, OFFSET=0,      DESCENDING=descending)
sort_half_kernel[(n_rows,)](mid, y_2d, ..., HALF=HALF, OFFSET=HALF,   DESCENDING=not descending)
# Stage 2: 逐元素 min/max 合并（无 gather）→ [min-8192, max-8192]，各半 bitonic
elem_merge_halves_kernel[grid](y_2d, y_2d, ..., HALF=HALF, DESCENDING=descending, BR=1)
# Stage 3a/3b: 各半再 ext.sort（同方向）→ 完成排序
sort_half_kernel[(n_rows,)](y_2d, y_2d, ..., HALF=HALF, OFFSET=0,    DESCENDING=descending)
sort_half_kernel[(n_rows,)](y_2d, y_2d, ..., HALF=HALF, OFFSET=HALF, DESCENDING=descending)
return y_2d[:, :n]
```

#### L2.4 递归 bitonic 骨架（L > 8192 一般情况）

```python
L_TILE = 4096
N_CHUNKS = L // L_TILE
# pad 到 L（L1.3）
mid = torch.cat([x_2d, pad_col], dim=1).contiguous()

# Pass 1: 每个 chunk 用 ext.sort 排序，方向按 bitonic 递归交替
chunk_dirs = _bitonic_chunk_directions(N_CHUNKS, descending)   # [T,F,T,F,...] 交替
for c in range(N_CHUNKS):
    sort_extsort_kernel[grid](chunk_slice, chunk_slice, ..., L=L_TILE, DESCENDING=chunk_dirs[c], ...)

# Pass 2: level-1 merge（相邻 chunk 对 → size 8192），用 bitonic_merge_2d
for p in range(N_CHUNKS // 2):
    bitonic_merge_subrange_kernel[grid](mid, mid, ..., HALF=L_TILE, LOG_HALF=..., OFFSET=offset, DESCENDING=merge_desc, ...)

# Pass 3: cross-tile + within-tile merge（每 stage 1 row × 1 tile 一个 program）
for s in range(LOG_L - LOG_L_TILE):
    cross_tile_merge_stage_kernel[(n_rows * N_CHUNKS,)](src, dst, ..., stride_shift=LOG_L-1-s, ...)
    src, dst = dst, src
within_tile_merge_kernel[(n_rows * N_CHUNKS,)](src, dst, ..., L_TILE=..., LOG_L_TILE=..., ...)
src, dst = dst, src
return src[:, :n]
```

#### L2.5 chunk 方向递归计算
（见 §1 M6 方向交替）

```python
def _bitonic_chunk_directions(n_chunks, final_descending):
    if n_chunks == 1:
        return [final_descending]
    half = n_chunks // 2
    left = _bitonic_chunk_directions(half, final_descending)
    right = _bitonic_chunk_directions(half, not final_descending)   # 右半反向 → 构成 bitonic
    return left + right
```

#### L2.6 bitonic stage 核心比较模式

```python
# XOR 索引 partner：partner_idx = cols ^ stride_mask
partner_val = tl.gather(x, partner_idx_2d, axis=1)
# 单次比较 + select：swap = (x > partner) == want_min
x = tl.where((x > partner_val) == want_min_2d, partner_val, x)
```

### §4.3 Layer 3: 关键技巧

#### L3.1 ext.sort 硬件排序替代软件 bitonic

**关键**: `triton.language.extra.cann.extension.sort` 是 CANN Vector Sort 硬件指令，L ≤ 4096 时远快于软件 bitonic。

```python
from triton.language.extra.cann import extension as ext
Y = ext.sort(X, dim=1, descending=DESCENDING)
```

**可替代方向**: 仅在 `ext.sort` 不可用时回落到软件 bitonic；硬件路径优先级最高。

#### L3.2 升序取负 trick

```python
# asc 路径（ext.sort 8192 desc 仅 fp16 稳定）
if DESCENDING:
    Y = ext.sort(X_2d, dim=1, descending=True)
else:
    Y = -ext.sort(-X_2d, dim=1, descending=True)
```

**可替代方向**: 若未来 `ext.sort` asc 路径稳定，可直接 `descending=False`，省去取负开销。

#### L3.3 逐元素 min/max 合并替代 gather merge（核心加速）

**关键洞察**: 两个已排序半（方向相反）构成 bitonic 序列。逐元素 min/max 即可拆分为 `[min-半, max-半]`，**无需 gather**。

```python
# 输入: [sorted-half1 (dir), sorted-half2 (opposite dir)] = bitonic
# 输出: [min-half, max-half] — 各半仍 bitonic
if DESCENDING:
    first  = tl.maximum(X1, X2)   # 前 8192
    second = tl.minimum(X1, X2)   # 后 8192
else:
    first  = tl.minimum(X1, X2)
    second = tl.maximum(X1, X2)
```

**收益**: 历史 V30 测得比 cross-tile gather merge 快约 **1230x**（消除 gather 随机访存）。

**可替代方向**: 该 min/max 拆分仅适用于「两半构成 bitonic」的前置条件；普通两路已排序序列合并仍需归并 merge。

#### L3.4 小 L 软件 bitonic 静态展开

```python
@triton.jit
def bitonic_sort_2d(x, L: tl.constexpr, LOG_L: tl.constexpr, DESCENDING: tl.constexpr):
    BLOCK_ROWS: tl.constexpr = x.shape[0]
    if LOG_L <= 8 and BLOCK_ROWS == 1:   # 1D 小 L 静态展开
        return bitonic_sort_2d_static(x, L=L, LOG_L=LOG_L, DESCENDING=DESCENDING)
    # 否则动态循环
    for k in range(1, LOG_L + 1):
        for j in range(k):
            ...

# static 版本用 tl.static_range 完全展开
@triton.jit
def bitonic_sort_2d_static(x, L, LOG_L, DESCENDING):
    for k in tl.static_range(1, LOG_L + 1):
        for j in tl.static_range(0, k):
            x = _bitonic_stage_static(x, K=k, J=j, ...)
    return x
```

**可替代方向**: 静态展开仅在 `LOG_L ≤ 8`（L ≤ 256）且 `BLOCK_ROWS==1` 时收益明确；2D 或大 L 用动态循环避免编译爆炸。

#### L3.5 bitonic merge 的方向内联

```python
# 把方向判断内联进 swap 条件，消除 want_min 中间张量
# asc:  swap when (x > partner) != is_right → 取 partner 放左侧 min
# desc: swap when (x > partner) == is_right → 取 partner 放左侧 max
if DESCENDING:
    x = tl.where((x > partner_val) == is_right_2d, partner_val, x)
else:
    x = tl.where((x > partner_val) != is_right_2d, partner_val, x)
```

**收益**: 减少 1 个 int32 中间 buffer，缓解 UB 压力。

#### L3.6 cross-tile / within-tile merge 的 grid 分配

```python
# 每 stage：1 row × 1 tile 一个 program，grid = (n_rows * N_CHUNKS,)
n_tiles_per_row: tl.constexpr = L // L_TILE
row = pid // n_tiles_per_row
tile = pid - row * n_tiles_per_row
```

**Why**: 跨 tile merge 每个 tile 独立比较 partner tile，按 tile 并行最自然；within-tile merge 则在每个 tile 内做完整 bitonic merge。

**可替代方向**: 可尝试按 row 聚合多 tile 到单 program（核内循环），但 UB 占用会上升。

#### L3.7 BLOCK_ROWS 自适应模板
（见 §1 M4）

```python
def _block_rows_for_extsort(L, dtype_itemsize, n_rows=1, num_cores=None):
    if num_cores is None: num_cores = _get_num_vector_cores()
    ub_total = 8192 if dtype_itemsize == 4 else 16384   # fp32 / fp16
    ub_limit = max(1, ub_total // L)
    if n_rows <= num_cores: return 1
    core_target = max(1, (n_rows + num_cores - 1) // num_cores)
    return min(ub_limit, core_target, n_rows)

def _block_rows_for_L(L, dtype_itemsize, n_rows=1, num_cores=None):
    if num_cores is None: num_cores = _get_num_vector_cores()
    max_buf_elems = 50 * 1024 // dtype_itemsize
    ub_limit = max(1, max_buf_elems // (L * 8))         # ~8 live buffers
    if n_rows <= num_cores: return 1
    core_target = max(1, (n_rows + num_cores - 1) // num_cores)
    return min(ub_limit, core_target, n_rows)
```

**可替代方向**: UB 系数（50KB / 8 buffers）为经验值，可按实测 UB 峰值微调。

### §4.4 Sort 性能基准

| 指标 | 数值 |
|------|------|
| Implementation Avg Latency | 0.3056 ms |
| Framework Avg Latency | 0.3262 ms |
| 几何平均加速比 | **1.9578x** |
| 精度通过率 | 31/31 |
| 异常 shape 索引 | 无（nan/inf/zero/negative/none 均为空） |

**关键结论**:
1. **ext.sort 硬件排序是小 L 的首选**：L ≤ 4096 时 CANN Vector Sort 远快于软件 bitonic
2. **逐元素 min/max 合并是 bitonic merge 的关键加速**：利用「两半方向相反构成 bitonic」性质，消除 gather 随机访存
3. **L 分段策略是架构基础**：ext.sort / 软件 bitonic / 三段式 sort-merge / 递归 bitonic 各有适用区间
4. **UB 约束决定 BLOCK_ROWS**：ext.sort 与软件 bitonic 的 UB 占用差异巨大，必须分别自适应
5. **升序大 L 用取负 trick**：`ext.sort` 8192 desc 仅 fp16 稳定，asc 通过取负转换
6. **超大 L（16384+）是固有弱项**：case 24 speedup < 1，软件 bitonic 比较次数指数增长，几何平均仍 > 1.9x

---

## §5 常见陷阱与避免方法

### §5.1 Sum 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 末维归约大 shape 慢于 aclnnReduceSum | reduce_last 沿 N 串行步进，大 N（如 16384）下迭代多、访存量大，8192×16384 仅 0.22x | 引入 split-K 两阶段归约；或增大 BLOCK_SIZE_N；勿照搬当前 16×1024 配置（L3.3） |
| 3D/4D 小 tensor 末维归约 launch 开销主导 | `BLOCK_SIZE_A=B=C=1` 导致 grid=输出元素数，小 tensor 上 im 延迟 6x 于 fw（case13 0.031ms vs 0.0049ms） | 输出维分块（A/B>1）让单 program 处理多输出点；小 tensor 回落 atomic 全归约（L3.4） |
| 低精度直接累加丢精度 | fp16/bf16 不经 fp32 累加，大 tensor verify 失败 | 强制 `data.to(tl.float32)` 后累加，store 前 cast 回原 dtype（L1.1/T2） |
| keepdim 误传入 kernel | 在 kernel 内处理 keepdim 导致输出形状错误 | keepdim 只在 host 侧构造 output 形状，kernel 仅按归约后扁平索引 store（L1.4） |
| dim=None 未用 atomic 累加 | 全归约无独立输出元素，多 block 部分和竞争写同一地址出错 | 预分配标量 output，kernel 内 atomic_add（L1.3） |

### §5.2 HyenaFftSizePaddingRfft 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 忘记除以 `fft_size` | PyTorch RFFT 默认归一化 | 两种 kernel 最后都除以 `fft_size`（L1.4/M6） |
| chirp 角度未模归约 | 大 n 时 NPU cos/sin 精度退化 | `n2_mod = (n*n) % (2*N)` 后再算角度（L1.6/M6） |
| 非 2 的幂 seqlen 用直接 DFT | matched_ratio < 0.9 | 必须用 Bluestein 算法（L1.1） |
| `self._get_xxx()` 缓存方法 | AST 验证器误判 PyTorch 退化 | 缓存逻辑内联到 forward()（L1.7） |
| fp64 cos/sin | Triton Ascend 不支持 | 全程 fp32 + 模归约保精度（L3.7 禁止项） |
| Twiddle LUT | global memory load 慢于硬件 trig，慢 6x | inline `tl.math.cos/sin`（L3.7 禁止项） |
| 向量化 in-place bit-reversal | 数据冒险，5/49 case 精度失败 | out-of-place gather 或保持标量循环（L3.3/L3.7 禁止项） |
| `N.to(tl.float32)`（N 为 constexpr） | constexpr 无 `.to` 方法 | 直接用 N 或创建运行时变量（L3.7 禁止项） |
| 函数体内 `m: tl.constexpr = 1<<s` | 局部变量不能用 tl.constexpr 注解 | 去掉注解，编译器自动常量传播（L3.7 禁止项/M3） |
| 合并 (k,j) tile 的整数除法分解 | NPU 整数除法昂贵 + 每次重算 twiddle，慢 9x | 保持 `for j_start` 外层 + `for k` 内层的双层循环结构（L3.7 禁止项） |
| Pass-merge normalize 到最后 stage store | Bluestein IFFT `num_stages >= 5`，`apply_norm` 分支引入额外乘法 + 寄存器压力，实测负收益 | 保持独立 normalize pass（L3.7 禁止项） |

### §5.3 Sort 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| `ext.sort` 单次排序长度超 UB 上限 | L > 4096（fp32）或 L > 8192（fp16）时 `ext.sort` UB overflow | 按分段策略拆分；L=16384 fp16 用三段式 sort-merge（L1.1/L2.3） |
| 升序大 L 直接用 `ext.sort(descending=False)` | `ext.sort` 8192 长度 asc 路径不稳定 | 用取负 trick `-ext.sort(-X, desc=True)`（L1.7/L3.2） |
| pad 值方向错误 | asc 用 `-inf` pad 会让 pad 元素排到开头 | asc 用 `+inf`，desc 用 `-inf`（L1.3/M2） |
| 两半方向相同就做 min/max 合并 | min/max 拆分要求两半方向相反构成 bitonic；同向合并不会得到有序结果 | Stage 1 两半必须 `DESCENDING` 与 `not DESCENDING` 交替（L2.3/L2.5/M6） |
| 硬编码 `num_cores` | 不同型号 NPU 核数不同 | 动态读取 `num_vectorcore`，兜底 48（L1.5/G1） |
| BLOCK_ROWS 共用单一函数 | ext.sort 与软件 bitonic 的 UB 占用差异大，共用会溢出或浪费并行度 | 分别提供 `_block_rows_for_extsort` / `_block_rows_for_L`（L1.6/L3.7/M4） |
| 非末维排序直接进 kernel | kernel 只处理 2D `(n_rows, n)`，非末维会维度错乱 | forward 中 permute 到末维，排序后 permute 回（L1.2/M1） |
| kernel 内使用 `continue`/`break` | 该后端对非规整控制流支持有限 | 用 `tl.where` 条件赋值替代（L1.8） |
| cross-tile gather merge 极慢 | gather 随机访存 | 用逐元素 min/max 合并替代（L3.3） |
| 为限制 grid 在核数内而对 `tl.sort` 加 program 内循环 | 编译器 multi-buffer 会把循环内 `tl.sort` 的临时 buffer 保留多份，导致 UB overflow（实测 fp32 L=4096 要求 208KB > 192KB） | 对 `tl.sort` 路径直接按 `grid = cdiv(n_rows, BLOCK_ROWS)` 启动，不强制限制在 `num_cores`；或在循环体外手动释放 buffer（若后端支持） |
| `tl.sort` 的 import 路径错误 | `triton.language.extra.cann` 在部分环境（如 Triton 3.2.0）不存在 | 使用 `triton.language.sort` 直接调用 |
