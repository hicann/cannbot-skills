---
name: transformer-inference
description: Transformer 推理类算子（RotaryMul / MoeComputeExpertTokens / MoeGatingTopKSoftmax / AttentionSoftmaxWithSoftcappingAndDropout / LightningIndexer / FFN）的 Triton Ascend 优化经验合集，按算子分章节组织，含通用经验 + 各算子专属约束/骨架/kernel
metadata:
  type: reference
---

# Transformer 推理类算子优化经验

本文档合并了六类 Transformer 推理算子的优化经验。按以下结构组织：
- **§1 通用经验**：跨算子重复的工程约束（已提取，各算子章节不再重复；与张量变换类共用的通用约束见 tensor-transform.md G1-G8）
- **§2 RotaryMul**（rotarymul，RoPE 旋转位置编码）
- **§3 MoeComputeExpertTokens**（indexing-gather / counting，MoE 专家 token 计数 + 前缀和）
- **§4 MoeGatingTopKSoftmax**（sort-topk，门控 softmax + 迭代 top-k）
- **§5 AttentionSoftmaxWithSoftcappingAndDropout**（reduce，softcapping + 行级 softmax 融合）
- **§6 LightningIndexer**（topk-select，稀疏注意力打分 + 降序 top-K 选位）
- **§7 FFN**（dual-gemm-activation，双 GEMM + 激活融合，含 GLU 变体与 MoE 分组）
- **§8 各算子常见陷阱**

---

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| RotaryMul | `rotarymul` | 4D `[B,H,S,D]` 张量 half/interleave 模式旋转乘法，fp16/bf16/fp32 | per-position 向量化 + uniform grid splitting + 2D tiling + broadcast stride |
| MoeComputeExpertTokens | `indexing-gather / counting` | expert 维度 token 计数（histogram）+ 小数组前缀和 | 两阶段分离：expert-parallel 无竞争计数 + 单 block 串行 prefix sum |
| MoeGatingTopKSoftmax | `sort-topk` | softmax over last dim + 迭代 top-k 选择，per-row 独立 | 纯寄存器 top-k 路径（无 GM temp buffer）+ grid 钳制到 num_cores |
| AttentionSoftmaxWithSoftcappingAndDropout | `reduce` | Gemma3 风格 softcapping `tanh(x/30)*30` + 行级 softmax，多 dtype 混合 | 4-kernel 分离强制中间舍入 + 分核优化（grid 钳制 + 循环处理多块） |
| LightningIndexer | `topk-select` | QK^T 打分 + relu + 权重头归约 + 行内降序 top-K 选位（输出索引/值），大 KV 稀疏注意力的索引器 | 位级一致分数链 + 大 shape 双路径拆分（打分/头归约分核）+ `.sort` 稳定排序选位 |
| FFN | `dual-gemm-activation` | y = act(x@W1+b1)@W2+b2，两段 GEMM 串行 + 中间激活（7 种激活 / 3 种 GLU 变体），可选 MoE 专家分组 | 双 kernel 拆分 + GLU 双累加器 epilogue 融合 + GM 物化保舍入链 + 固定核数 Swizzle2D |

> ⚠️ **关键区分**：六类算子计算模式差异极大，优化哲学不可混用：
> - RotaryMul 关心 **per-position 向量化** 避免 flat-1D 标量退化
> - MoeComputeExpertTokens 关心 **无竞争 expert-parallel 计数** 避免 atomic contention
> - MoeGatingTopKSoftmax 关心 **寄存器内 top-k 路径** 避免 GM 往返导致 `tl.argmax` 不可靠
> - AttentionSoftmaxWithSoftcappingAndDropout 关心 **多 kernel 分离强制 dtype 舍入** 匹配 PyTorch 中间物化行为
> - LightningIndexer 关心 **位级一致分数链**（索引输出逐位比对，任何近似选位都会翻边失败）与 **大 shape 双路径拆分** 绕过融合核微 dot 的同步税
> - FFN 关心 **双 kernel 拆分**（GEMM2 全量依赖 GEMM1 输出，单 kernel 需栅格同步不可行）与 **epilogue GM 物化**（dot 累加器 cast 被编译器消除，舍入链丢失）

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下 8 条约束是六类 Transformer 推理算子**共有**且**未在 tensor-transform.md G1-G8 覆盖**的工程约束。tensor-transform.md 中已提取的 G1（动态 num_cores）/ G2（pow2 BLOCK）/ G3（多策略分派）/ G4（grid 不超核数）/ G5（int32 索引）/ G6（负载均衡）/ G7（contiguous）/ G8（坐标 float32 比较）此处不再重复，各算子章节引用时标注。

### T1 grid_size 必须与 num_cores 严格匹配（grid = num_cores 或 grid ≤ num_cores）
- **必须**令 `grid_size = min(batch_size/total_blocks, VEC_CORE_NUM)`，且 `num_cores` 入参 = `grid_size`。
- **禁止**设置 `grid_size > VEC_CORE_NUM` 或 `num_cores ≠ grid_size`。
- **Why:** 当 `num_cores ≠ grid_size` 时，任务划分公式 `rows_per_core = cdiv(batch, num_cores)` 与实际 grid 不一致，导致部分 core 处理范围错误（MoeGatingTopKSoftmax 实测引发结果错误）。
- **典型应用**：MoeGatingTopKSoftmax（grid = min(batch_size, VEC_CORE_NUM)，num_cores = grid_size）、AttentionSoftmax（grid_size = min(natural_blocks, num_cores)）。
- 与 G4 的差异：G4 仅要求 grid 不超过核数，T1 进一步要求当 kernel 内依赖 `num_cores` 入参计算循环 stride 时，二者必须**严格相等**。

### T2 fp16/bf16 输入必须在 kernel 内升精度到 fp32 计算
- **必须**对 fp16/bf16 输入在 kernel 内 `.to(tl.float32)` 计算，再转回原精度存储。
- **Why:** 直接以 fp16/bf16 做乘加减/归约会导致精度误差超出 verify 阈值（relative error > 1e-3），且 PyTorch 参考实现普遍在算子内部隐式提升到 fp32。
- **典型应用**：RotaryMul（fp16/bf16 → fp32 旋转乘法 → 转回）、AttentionSoftmax（所有 div/tanh/mul/softmax 子算子均 fp32 计算）、MoeGatingTopKSoftmax（softmax 在 fp32 下做 max/exp/sum/div）。

### T3 禁止在 forward() 中使用 PyTorch 计算（Type-3 退化）
- **必须** `ModelNew.forward()` 中只负责 shape 计算、分派、host 预计算；所有数值计算必须在 `@triton.jit` kernel 内完成。
- **Why:** `validate_triton_impl.py` Type-3 检查会 flag 任何 forward 中的 torch 计算为退化；同时也是 Triton-Ascend 算子的基本要求。
- **典型应用**：MoeGatingTopKSoftmax 禁用 `torch.softmax` / `torch.topk`；MoeComputeExpertTokens 禁用 `torch.bincount` / `torch.cumsum`。

### T4 编译期常量化：循环边界、tl.arange 长度、repeat 次数等必须为 tl.constexpr
- **必须**将 `num_expert`、`k`、`MAX_HALF_D`、`BLOCK_SIZE`、`TILE_S`、`r` 等作为 `tl.constexpr` 传入 kernel。
- **禁止**在 kernel 内使用 `tl.arange(0, num_experts)` 其中 `num_experts` 为普通入参（非 constexpr）。
- **Why:** Triton Ascend 要求 `tl.arange` 参数必须是编译时常量，否则报 `ValueError: arange's arguments must be of type tl.constexpr`；同时 constexpr 让编译器展开循环、生成 vector load/store。
- 与 G2 的差异：G2 强调 BLOCK 长度取 pow2，T4 强调**任何**进入 `tl.arange` / 循环 `range()` 的变量都必须 constexpr，包括非 pow2 的语义维度（如 `num_expert=64`、`k=8`）。

### T5 编译器融合会消除中间 dtype 舍入，必要时拆分独立 kernel 强制 GM round-trip
- **必须**当参考实现（PyTorch C++）在每个子算子处物化中间结果为输入 dtype 时，Triton kernel 若被编译器融合为单一 fp32 计算会跳过舍入，导致逐 bit 不匹配。
- **解决**：将表达式拆分为多个独立 kernel，每个 kernel 通过 GM store+load 往返强制中间 dtype 舍入。
- **典型应用**：AttentionSoftmaxWithSoftcappingAndDropout 的 `tanh(x/30)*30` 在 bf16/fp16 下必须拆为 div/tanh/mul 三独立 kernel。
- **fp32 输入例外**：fp32 下融合后仍 fp32，无舍入损失，不必拆分。

### T6 multibuffer / unit_flag 等编译选项需实测验证，禁止默认开启
- **禁止**盲目添加 `multibuffer=True, unit_flag=True` 期望提升内存密集型算子性能。
- **Why:** 在含迭代标量循环（如 MoeGatingTopKSoftmax 的 top-k）的算子上，multibuffer 的流水线优化收益被迭代开销掩盖，实测可能劣化（MoeGatingTopKSoftmax v3: 0.8527x vs 基线 0.8836x，v10: 0.5720x vs 基线 0.8194x）。
- **正确做法**：在 Phase 4 中实测验证后再决定是否启用。

### T7 dot 累加器派生值的舍入必须用 GM 物化（store-reload），cast round-trip 会被编译器消除
- **必须**当 `tl.dot` 的 fp32 累加器需先舍入到低精度 dtype 再参与后续逐元素运算（bias/激活链）时，用「store 到 GM → reload」物化舍入结果，再执行后续运算链。
- **禁止**依赖 `acc.to(f16).to(f32)` 这类 round-trip cast 保留舍入——编译器判定其为 no-op **直接消除**；bitcast、`tl.where` 强制依赖、`tl.debug_barrier` 同样会被消除，勿再尝试。
- **Why:** 逐 op「fp32 计算 → 落回输入 dtype」的舍入链是与参考实现对齐的关键；cast 被消除后后续运算实际作用在未舍入的 fp32 上，近零消输出误差可放大 2-3 个数量级。
- **UB 联动**：store-reload 工作区若按全 tile 宽度物化可能超 192KB UB，需按子切片（如列宽 64）分批 reload，K 循环 tile 宽度与后续链宽度解耦。
- 与 T5 的差异：T5 讲**跨 kernel** 拆分强制 GM round-trip；T7 讲**同一 kernel 内** dot 累加器派生值的 cast 消除问题。

### T8 大偏移寻址必须 int64，host 侧小张量下传用 pinned 环形缓冲
- **必须**多维大张量（E×K×N 类）按块基址偏移寻址时，偏移量先 `.to(tl.int64)` 再乘加，防 int32 溢出地址回绕。
- **必须**每次前向需下传 host 计算的小张量（前缀和/索引表等）时，用 pinned memory 环形缓冲 + `copy_(..., non_blocking=True)` 异步上传；**禁止**每调用同步 H2D（实测每次 ~150us 开销）。
- **可选扩展路径**：分支路径用 `tl.constexpr` 布尔标志隔离，非该路径输入传 dummy 指针，编译产物互不影响。

---

## §2 RotaryMul 算子（rotarymul）

**算子类别**: `rotarymul`（RoPE 旋转位置编码乘法）
**典型特征**: 4D 张量 `[B, H, S, D]`，支持 `half` / `interleave` 两种模式，支持 fp16/bf16/fp32，r1/r2 支持 broadcast
**性能基准**: 几何平均加速比 **1.02x** vs torch（50 cases 全通过）

### §2.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 禁止 flat-1D 索引分解
- **禁止**在 kernel 内将一维 flat index 通过 `%` 和 `//` 运算反向分解为 `(b, h, s, d)` 多维坐标。
- **Why:** Ascend 编译器会将这些 per-element 的地址计算标量化，生成大量 `scf.for` step-1 循环和 `memref<1xf16>` 标量 load/store，导致 HIVM intrinsics 利用率极低（实测仅 19.8%）。
- **How to apply:** 每个 program 直接处理一个完整的 `(b, h, s)` 位置，使用 `tl.arange` 做 contiguous vector load/store。

#### L1.2 必须显式处理 broadcast stride
- **必须**在 Host 侧计算 r1/r2 的 broadcast stride：若某维 shape 为 1 且 input 对应维 >1，则该维 stride 设为 0。
- **Why:** RotaryMul 的 r1/r2 常见 broadcast 模式（如 `[B,1,S,D]` 对 `[B,H,S,D]`），若直接传 `t.stride()` 会导致 kernel 内地址计算错误。
- **How to apply:**
  ```python
  def _bc_stride(t, input_shape):
      return tuple(0 if t.shape[i] == 1 and input_shape[i] > 1 else t.stride(i) for i in range(4))
  ```

#### L1.3 禁止在 kernel 内做 dtype 分支判断
- **禁止**在 `@triton.jit` kernel 内通过 `if dtype == torch.float16` 这类运行时分支判断数据类型。
- **Why:** Triton kernel 内无法直接访问 PyTorch dtype 对象；应通过 `tl.constexpr` 布尔标志（如 `IS_FP16`, `IS_BF16`）在编译期确定分支。
- **How to apply:** Host 侧计算 `IS_FP16 = (dtype == torch.float16)` 等标志，作为 `tl.constexpr` 传入 kernel。

#### L1.4 fp16/bf16 必须在 kernel 内升精度到 fp32 计算
（见 §1 T2）

#### L1.5 禁止 Adaptive TILE_S（编译期动态 tile 大小）
- **禁止**在 Host 侧根据 S/D 大小选择不同的 `TILE_S`（如 `TILE_S = 32 if S >= 512 and D <= 64 else 16`）。
- **Why:** `TILE_S` 作为 `tl.constexpr`，若在不同 shape 间变化会导致 kernel 被重新编译；更大的问题是过大的 2D tile（如 32x32）在 Ascend 上会被编译器标量化，造成灾难性性能退化（实测 65x slowdown）。
- **How to apply:** 固定 `TILE_S = 16`，通过 uniform grid splitting 解决负载均衡问题。

#### L1.6 必须添加 `tl.assume` 编译器提示
- **必须**对 stride 和 shape 添加 `tl.assume` 提示，尤其是 `stride_d == 1`、`half_d >= 16`、`TILE_S > 0` 等。
- **Why:** 帮助 Ascend 编译器生成 vector load/store 而非标量循环。
- **How to apply:** 在 kernel 入口放置 `tl.assume(stride_in_d == 1)` 等。

### §2.2 Layer 2: 算法骨架

#### L2.1 Host 侧分支决策树（伪代码）

```python
# 1. 数据准备
input_c = input if input.is_contiguous() else input.contiguous()    # G7
B, H, S, D = input_c.shape
output = torch.empty_like(input_c)

# 2. Broadcast stride 处理（L1.2）
r1_stride = _bc_stride(r1, input_c.shape)
r2_stride = _bc_stride(r2, input_c.shape)

# 3. 编译期常量（T4）
IS_HALF = (rotary_mode == 'half')
IS_FP16 = (dtype == torch.float16)
IS_BF16 = (dtype == torch.bfloat16)
MAX_HALF_D = D // 2          # 作为 tl.constexpr 传入
TILE_S = 16                  # 固定值（L1.5）
assert S % TILE_S == 0

# 4. Grid 计算（G1 动态 num_cores + G4 grid 不超核数）
num_s_tiles = S // TILE_S
num_blocks = B * H * num_s_tiles
def _grid(meta): return (min(num_blocks, VEC_CORE_NUM),)

# 5. Kernel 启动
kernel[_grid](..., num_cores=VEC_CORE_NUM, TILE_S=TILE_S, MAX_HALF_D=MAX_HALF_D, ...)
```

#### L2.2 Kernel 内多核并行骨架（Uniform Grid Splitting）

**核心思想**：将 `num_blocks = B * H * num_s_tiles` 均匀分配到 `num_cores` 个 vector core 上，避免 naive `ceil(num_blocks / num_cores)` 导致的 idle core。

```python
pid = tl.program_id(0)
num_blocks = B * H * num_s_tiles

blocks_per_core = num_blocks // num_cores
remainder = num_blocks % num_cores
block_start = blocks_per_core * pid + tl.minimum(pid, remainder)
block_end = block_start + blocks_per_core + tl.where(pid < remainder, 1, 0)

for block_idx in range(block_start, block_end):
    # 将 block_idx 解码为 (b, h, s_tile)
    tmp = block_idx // num_s_tiles
    s_tile = block_idx - tmp * num_s_tiles
    tmp2 = tmp // H
    h = tmp - tmp2 * H
    b = tmp2
    s_start = s_tile * TILE_S
    # ... load/compute/store ...
```

#### L2.3 2D Tiling 向量加载模式

**模式**：每个 block 处理 `TILE_S` 个连续 S 位置 × `MAX_HALF_D` 个连续 D 位置。

```python
s_offs = tl.arange(0, TILE_S)[:, None]      # [TILE_S, 1]
d_offs = tl.arange(0, MAX_HALF_D)[None, :]  # [1, MAX_HALF_D]

in_base = b * stride_b + h * stride_h + s_start * stride_s
idx1 = in_base + s_offs * stride_s + d_offs * stride_d
idx2 = in_base + s_offs * stride_s + (d_offs + half_d) * stride_d

inp1 = tl.load(input_ptr + idx1)  # 2D vector load
inp2 = tl.load(input_ptr + idx2)
```

#### L2.4 half vs interleave 模式处理

- **half 模式**：将 D 维从中间切分，`[..., :half_d]` 和 `[..., half_d:]` 分别与 r1/r2 的对应半区做旋转乘法。
- **interleave 模式**：将 D 维按奇偶分离，`[..., 0::2]` 和 `[..., 1::2]` 分别做旋转乘法。
- **统一公式**：
  - half: `out1 = x1 * r1_1 - x2 * r2_1`, `out2 = x2 * r1_2 + x1 * r2_2`
  - interleave: `out_even = x1 * r1_e - x2 * r2_e`, `out_odd = x2 * r1_o + x1 * r2_o`

### §2.3 Layer 3: 关键技巧

#### L3.1 从 flat-1D 到 per-position vectorization

**问题**：初始实现使用 `BLOCK_SIZE=1024` 的 flat-1D 循环，每个 thread 处理一个 flat index，通过 `%` 和 `//` 分解坐标，导致标量退化。

**解决**：改为每个 program 处理一个 `(b, h, s)` 位置，D 维用 `tl.arange(0, MAX_HALF_D)` 向量化。关键变化：
- 去掉 `BLOCK_SIZE`，改为 `MAX_HALF_D = D // 2` 作为 `tl.constexpr`
- 去掉 `mask = offsets < total_elements`
- 坐标分解从 per-element 变为 per-position（仅分解 `b, h, s`，`d` 由 `tl.arange` 覆盖）

**可替代方向**: 若 D 不固定，可用 `tl.arange(0, BLOCK_D)` 配合 `for d_tile in range(0, half_d, BLOCK_D)` 做 D 维循环分块。

#### L3.2 Uniform Grid Splitting 消除 idle core（G6 负载均衡的具体实现）

```python
blocks_per_core = num_blocks // num_cores
remainder = num_blocks % num_cores
block_start = blocks_per_core * pid + tl.minimum(pid, remainder)
block_end = block_start + blocks_per_core + tl.where(pid < remainder, 1, 0)
```

前 `remainder` 个 core 各多处理 1 个 block，实现完全均匀分配。

**可替代方向**: 若 block 粒度极不均匀（如不同 block 工作量差异大），可考虑 dynamic work stealing 或按工作量加权分配，但 RotaryMul 中每个 block 工作量相同，uniform splitting 最优。

#### L3.3 2D Tiling (TILE_S=16) 摊平同步开销

**问题**：per-position kernel（每个 program 只处理 1 个 S 位置）在大 S shape 下产生过多 pipeline sync（`hivm.hir.set_flag`, `wait_flag`, `pipe_barrier`），大 S 性能差。

**解决**：每个 program 一次处理 `TILE_S=16` 个连续 S 位置，用 2D `tl.arange` 做 `[TILE_S, MAX_HALF_D]` 的向量化 load/store。

**关键参数选择**：
- `TILE_S = 16` 是经验最优值（非 IR 分析得出）
- `TILE_S = 8` 导致 2D tile 太小（256 elements），vectorization 效果差
- `TILE_S = 32` 导致编译器标量化，性能退化 65x

**可替代方向**: 对于 S 较小（如 S < 16）的 shape，可回退到 TILE_S=1 的 per-position 模式，但 RotaryMul 的 S 通常为 128/256/512/1024+，固定 16 即可。

#### L3.4 避免 Host 侧 dtype 转换和 expand

**问题**：早期实现在 Host 侧将输入 `.to(torch.float32)` 并用 `expand_as()` 处理 broadcast，引入额外内存拷贝和峰值内存占用（16.9MB → 7.44MB）。

**解决**：
- 不在 Host 侧做 dtype 转换，仅在 kernel 内对 fp16/bf16 升精度（T2）
- 不在 Host 侧 `expand` broadcast 张量，而是通过自定义 `_bc_stride` 在 kernel 内用 stride=0 处理 broadcast（L1.2）

### §2.4 RotaryMul 性能基准

| Shape 类型 | 典型加速比 | 说明 |
|-----------|-----------|------|
| 小 shape [1,1,128,64] | 2.4x | 小 S 高并行度，轻松超越 torch |
| 中 shape [1,8,512,64] | 0.67x | torch aclnn 优化充分，Triton 有 gap |
| 大 shape [1,8,32768,64] | 0.10x | 内存带宽瓶颈，Triton 仍落后 |
| 全量 50 cases | 1.02x | 几何平均刚好达标 |

**关键结论**：
1. RotaryMul 在小 shape 上 Triton 有明显优势（2-3x），但在大 shape / 高并行度场景下，torch aclnn 的 `aclnnRotaryPositionEmbedding` 高度优化，Triton 难以超越。
2. 2D Tiling + Uniform Grid Splitting 是达到目标加速比（0.8x）的关键；去掉任一项都会使几何平均低于目标。
3. 标量退化（flat-1D index 分解）是 Ascend 上最常见的性能陷阱，必须从算法设计阶段避免。

---

## §3 MoeComputeExpertTokens 算子（indexing-gather / counting）

**算子类别**: `indexing-gather / counting`
**典型特征**: expert 维度 token 计数（histogram）+ 小数组前缀和（prefix sum / cumsum）
**性能基准**: 几何平均加速比 **1.7764x** vs torch（50/50 cases 全通过）

### §3.0 算子描述

输入：
- `sorted_expert_for_source_row`: int32 1D tensor，每个 token 对应的 expert ID（已排序）
- `num_expert`: int32 scalar，专家总数

输出：
- int32 1D tensor，长度 `num_expert`，`output[i]` = 专家 `0..i` 的累计 token 数

本质：先对每个 expert 做 token 计数（histogram），再做前缀和（prefix sum / cumsum）。

### §3.1 Layer 1: 设计约束（硬性边界）

#### L1.1 禁止在计数阶段使用 `tl.atomic_add`
- **原因**：所有 block 竞争写入同一 `counts` 地址，Ascend NPU 上 atomic 操作开销极大，导致性能严重退化（实测劣化至 0.3x 以下，归档 1.93x → 当前 0.30x）。
- **正确做法**：每个 expert 分配一个独立的 block，该 block 独享一个输出地址，通过串行循环累加完成计数，无需原子操作。
- **⚠️ 反模式警示**：`grid = (cdiv(N, BLOCK_SIZE),)` + block 内 `for expert in range(num_expert)` + `tl.atomic_add(output+expert, count)` 是错误的并行方向，必然触发本条禁忌，性能劣化 6 倍以上。

#### L1.2 禁止在 prefix sum 阶段使用 `tl.cumsum`
- **原因**：`tl.cumsum` 在 Ascend 后端可能退化为低效实现或引发 PyTorch fallback；对于 `num_expert <= 64` 的小数组，单 block 串行 scan 更可靠且足够快。
- **正确做法**：单 block 串行 `for` 循环读取-累加-存储。

#### L1.3 禁止在计数 kernel 中做跨 expert 的循环
- **禁止**：`for e in range(num_expert)` 在每个 block 内遍历所有 expert。
- **原因**：这会导致每个 block 对所有 expert 都做判断，计算冗余；且当 `num_expert` 较大时（如 64），循环展开后指令膨胀。
- **正确做法**：grid 维度 1 = `num_expert`，每个 block 只负责一个 expert 的计数。

#### L1.4 禁止在 `forward()` 中引入 `torch_npu` 依赖或动态 grid 计算
- **原因**：引入 `torch_npu` 增加环境依赖，且动态 grid 对编译器优化不友好；计数阶段 grid 固定为 `num_expert` 即可。
- **正确做法**：grid 直接传 `(num_expert,)` 和 `(1,)`，无需运行时计算。
- 与 G1 的差异：G1 要求动态读取 num_cores 用于一般算子；MoeComputeExpertTokens 的 grid 由语义维度（num_expert）决定，与核数无关，固定即可。

#### L1.5 grid 维度必须 = num_expert（expert-parallel），禁止 token-parallel
- **必须** `grid = (num_expert,)`，每 block 处理 1 个 expert，block 内向量化遍历所有 token。
- **禁止** `grid = (cdiv(N, BLOCK_SIZE),)`（token-parallel），这会强制 block 内跨 expert 循环 + atomic_add，违反 L1.1/L1.3，性能劣化至 0.3x。
- **正确骨架**：
  ```python
  # grid = (num_expert,), block 内向量化遍历 token
  expert_id = tl.program_id(0)
  count = 0
  for chunk_start in range(0, N, BLOCK_SIZE):
      tokens = tl.load(token_ptrs + chunk_start + tl.arange(0, BLOCK_SIZE), mask=...)
      is_match = (tokens == expert_id).to(tl.int32)
      count += tl.sum(is_match)  # 向量化规约，无 atomic
  tl.store(counts_ptr + expert_id, count)
  ```
- **Why:** expert-parallel 让每 block 独享输出地址（无竞争），向量化 tl.load + tl.sum 远快于逐元素循环。

### §3.2 Layer 2: 算法骨架（两阶段分离架构）

```
Phase 1: 计数（Count）
  Grid: (num_expert,)
  每个 block 对应一个 expert_id
  block 内：for chunk over tokens (BLOCK_SIZE=1024)
    load tokens -> compare == expert_id -> sum -> accumulate to local count
  最后 store count to counts[expert_id]

Phase 2: 前缀和（Prefix Sum）
  Grid: (1,)
  单 block 串行 scan
  for e in range(num_expert):
    count = load counts[e]
    prefix += count
    store prefix to output[e]
```

**核心设计决策**：

| 决策 | 选择 | 理由 |
|------|------|------|
| 计数并行维度 | expert 维度（grid = num_expert） | 无竞争、无 atomic、自然负载均衡 |
| 前缀和并行度 | 单 block 串行 | num_expert 很小，串行足够快且避免复杂同步 |
| BLOCK_SIZE | 1024 | 经验值，在 Ascend 上向量 load/store 效率较高 |
| 中间 buffer | `torch.empty` 分配 counts | 避免在 kernel 内做复杂内存管理 |

### §3.3 Layer 3: 关键技巧

#### L3.1 无竞争计数（Expert-Parallel Counting）

```python
@triton.jit
def moe_count_kernel(
    sorted_expert_ptr, counts_ptr, num_tokens,
    num_expert: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    expert_id = tl.program_id(0)
    if expert_id >= num_expert:
        return

    count = 0
    num_chunks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * BLOCK_SIZE
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_tokens

        tokens = tl.load(sorted_expert_ptr + offsets, mask=mask, other=0)
        is_match = (tokens == expert_id) & mask
        chunk_count = tl.sum(is_match.to(tl.int32))
        count += chunk_count

    tl.store(counts_ptr + expert_id, count)
```

**要点**：
- `grid = (num_expert,)`，每个 block 只写一个输出地址。
- `tl.sum(is_match.to(tl.int32))` 将 mask 向量规约为标量，无需 atomic。
- `other=0` 避免越界读取影响比较结果。

#### L3.2 小数组串行前缀和

```python
@triton.jit
def prefix_sum_kernel(counts_ptr, output_ptr, num_expert: tl.constexpr):
    if tl.program_id(0) > 0:
        return

    prefix = 0
    for eid in range(num_expert):
        count = tl.load(counts_ptr + eid)
        prefix += count
        tl.store(output_ptr + eid, prefix)
```

**要点**：
- 仅允许 `program_id(0) == 0` 执行，其余 block 直接返回。
- `num_expert` 为 `tl.constexpr`（T4），编译器可展开循环。
- 对于 `num_expert <= 64`，latency 极低（实测约 2.8ms）。

#### L3.3 避免 `torch.zeros` 引入额外算子

```python
# 不良：counts = torch.zeros(...)
# 会引入 aclnnInplaceZero / aten::zero_ 等 PyTorch 算子

# 良好：counts = torch.empty(...)
# 计数 kernel 会覆盖全部 num_expert 个元素，无需预清零
```

**要点**：`torch.empty` 不初始化，但计数 kernel 保证每个位置都被写入，安全且避免额外算子。

### §3.4 MoeComputeExpertTokens 性能基准

- **框架延迟**：0.0126 ms（平均）
- **实现延迟**：0.0108 ms（平均）
- **几何平均加速比**：**1.7764x**
- **Shape 范围**：tokens = 50~10000，experts = 4/8/16/32/64
- **最差 case**：experts=64, tokens=1000，仍达 0.66x（未劣化）
- **最佳 case**：experts=4, tokens=500，达 6.68x

> **基准差异说明**：上述 1.7764x 基准对应 PyTorch 参考为纯 Python 实现（`bincount + cumsum`）或较低优化度的 torch 路径。若任务文件的 `Model.forward` 直接调用 `torch_npu.npu_moe_compute_expert_tokens`（CANN 原生高度优化算子），Triton 实现受限于两阶段 kernel 启动开销与 O(num_expert × num_tokens) 计数复杂度，几何平均加速比可能显著下降（实测约 0.25~0.30x）。此时小 shape（tokens ≤ 500、experts ≤ 8）仍可接近或超过 0.8x，大 tokens / 大 experts 是主要瓶颈。

---

## §4 MoeGatingTopKSoftmax 算子（sort-topk）

**算子类别**: `sort-topk`
**典型特征**: softmax over last dim + iterative top-k selection, per-row independent processing
**性能基准**: 几何平均加速比 **0.8194x** vs torch（50/50 cases pass，最新验证版本 v10_20260624）
**历史最佳版本**: v8_20260624，**0.8967x**（50/50 cases pass）

### §4.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 禁止在 kernel 内使用 PyTorch 计算
（见 §1 T3）
- **必须** 将所有计算（softmax + top-k）放在 `@triton.jit` kernel 内完成。
- **禁止** 在 `forward()` 中使用 `torch.softmax`、`torch.topk` 等 PyTorch 算子。

#### L1.2 禁止 bool/int1 GM 缓冲区
- **禁止** 使用 `torch.bool` 类型作为 GM 临时缓冲区并通过 `tl.load`/`tl.store` 访问。
- **Why:** Triton 将 `tl.int1` GM 存取视为 `int8`，导致 `tl.where` 条件类型不匹配，在 f32 大 shape 上产生精度错误（max_rel_err ~1.0）。
- **How to apply:** `finished` 标志应直接以 `torch.bool` 传入 kernel，`tl.load` 后直接作为 `tl.where` 条件使用，无需中间 GM 缓冲区。

#### L1.3 禁止 GM temp buffer 存储中间 softmax 结果
- **禁止** 将 `softmax_val` 写入 GM temp buffer 再读回用于 top-k。
- **Why:** GM store/load 引入额外内存带宽，且 `tl.where` 修改后的向量经 GM 往返后 `tl.argmax` 不可靠。
- **How to apply:** 保持 `softmax_val` 在寄存器中，top-k 循环内通过 `curr = tl.where(offsets == best_idx, -inf, curr)` 直接更新寄存器向量。

#### L1.4 Grid 大小必须匹配 num_cores（见 §1 T1）

#### L1.5 禁止在 kernel 参数中使用运行时变量作为 tl.arange 参数（见 §1 T4）
- **必须** 将 `num_experts`、`k` 等作为 `tl.constexpr` 传入 kernel。

#### L1.6 禁止在 kernel 内使用 Python if/else 判断指针是否为 None
- **禁止** 在 `@triton.jit` kernel 内使用 `if finished_ptr is not None:`
- **Why:** Triton 不支持 Python 的 `is not None` 指针判断，会编译失败或行为不可预期。
- **How to apply:** 在 host 侧 `forward()` 中处理：若 `finished is None`，创建一个全 `False` 的 dummy tensor 传入 kernel，kernel 内直接 `tl.load(finished_ptr + row)` 无需判断。

#### L1.7 禁止在 kernel 内使用 Python break/continue
- **禁止** 在 `@triton.jit` kernel 内使用 `break` 或 `continue`。
- **Why:** Triton 不支持 Python 循环控制流，会报 `unsupported AST node type: Break`。
- **How to apply:** 使用 `tl.where` 条件赋值替代。例如：`if row >= end_row: break` → 改为 `row_valid = row < end_row` + `tl.where(row_valid, ...)` 或确保循环范围正确。

#### L1.8 NUM_EXPERTS 较小时必须单次整行 load，禁止 chunked multi-pass
- **必须** 当 `NUM_EXPERTS` 是编译期常量且 `NUM_EXPERTS <= BLOCK_E`（通常 ≤1024，pow2 时编译期 `tl.arange` 安全）时，用**单次整行 load**：`offsets = tl.arange(0, NUM_EXPERTS)` 一次 load 整行 gating 权重到寄存器，单 pass 完成 max/sum/topk。
- **禁止** 对小 `NUM_EXPERTS` 使用 chunked multi-pass（PASS1 求 max + PASS2 求 sum + PASS3 topk 分三次循环 load 同一 chunk），这会把内存带宽放大 3 倍，大 shape 性能跌到 0.1-0.4x。
- **Why:** `tl.arange(0, NUM_EXPERTS)` 当 NUM_EXPERTS 是 `tl.constexpr` 时是编译期固定长度向量，单次 load 到寄存器后所有 max/sum/argmax 运算都在寄存器内完成，零重复访存。chunked 路径仅当 `NUM_EXPERTS > UB 容量`（实测 >4096）才需要。
- **L1.3 适用范围澄清**：L1.3 警告"GM temp buffer 会使 argmax 不可靠"指的是**写回 GM 再读**的场景；寄存器内的 `tl.argmax` 在单 chunk 内是可靠的，可直接使用。
- **How to apply:** 见 L3 中"单次整行 load + 寄存器 topk"骨架；仅当 num_experts 编译期未知或 >4096 时才走 chunked 路径（且应合并 max+sum 为 2-pass）。

### §4.2 Layer 2: 算法骨架

#### L2.1 Host 侧分支决策树

```python
forward(x, finished, k):
  1. x_flat = x.view(-1, num_experts).contiguous()        # G7
  2. finished_flat = finished.view(-1) if finished else zeros(batch_size, bool)  # L1.6
  3. grid_size = min(batch_size, VEC_CORE_NUM)            # T1
  4. kernel[grid_size](..., num_cores=grid_size, k=k)      # num_cores == grid_size（L1.4）
```

#### L2.2 Kernel 内按行分配骨架

```python
pid = tl.program_id(0)
rows_per_core = cdiv(batch_size, num_cores)  # num_cores == grid_size（T1）
start_row = pid * rows_per_core
end_row = min(start_row + rows_per_core, batch_size)

for row in range(start_row, end_row):
    # 1. Load row + softmax（fp32，见 T2）
    # 2. Top-k in registers（k iterations of max+argmax+where mask）
    # 3. Store top-k values/indices
    # 4. Store row_idx
```

#### L2.3 纯寄存器 top-k 骨架

```python
vals_vec = tl.full((k,), 0.0, tl.float32)
idxs_vec = tl.full((k,), 0, tl.int32)
curr = softmax_val  # register vector

for i in range(k):
    best_val = tl.max(curr, axis=0)
    best_idx = tl.argmax(curr, axis=0)
    final_idx = tl.where(finished_flag, num_experts, best_idx)

    mask = tl.arange(0, k) == i
    vals_vec = tl.where(mask, best_val, vals_vec)
    idxs_vec = tl.where(mask, final_idx.to(tl.int32), idxs_vec)

    curr = tl.where(offsets == best_idx, -inf, curr)
```

### §4.3 Layer 3: 关键技巧

#### L3.1 纯寄存器路径（已验证有效）

```python
# 完全避免 GM temp buffer
# softmax_val 保持在寄存器中
curr = softmax_val
for i in range(k):
    best_val = tl.max(curr, axis=0)
    best_idx = tl.argmax(curr, axis=0)
    # ... accumulate ...
    curr = tl.where(offsets == best_idx, -float('inf'), curr)
```

**可替代方向**: 若 k 极大且寄存器压力过高，可考虑分块处理，但需验证 `tl.argmax` 在 GM 往返后的可靠性。

#### L3.2 k=1 快速路径（已验证有效）

```python
if k == 1:
    best_val = tl.max(softmax_val, axis=0)
    best_idx = tl.argmax(softmax_val, axis=0)
    final_idx = tl.where(finished_flag, num_experts, best_idx)
    tl.store(topk_values_ptr + row * k, best_val)
    tl.store(topk_indices_ptr + row * k, final_idx.to(tl.int32))
    tl.store(row_idx_ptr + row * k, row.to(tl.int32))
```

**可替代方向**: 也可统一走循环路径，但 k=1 快速路径减少标量循环开销。

#### L3.3 finished 标志直接 bool 传入（已验证有效）

```python
# Host side
finished_flat = finished.view(-1) if finished is not None else torch.zeros(
    batch_size, dtype=torch.bool, device=x.device)

# Kernel side
finished_flag = tl.load(finished_ptr + row)  # directly bool
final_idx = tl.where(finished_flag, num_experts, best_idx)
```

**可替代方向**: 若编译器对 bool GM 支持不稳定，可尝试 int8，但 int32 会引入额外 cast 开销。

#### L3.4 向量存储 top-k 结果（已验证有效）

```python
# Accumulate into register vectors during loop
vals_vec = tl.full((k,), 0.0, tl.float32)
idxs_vec = tl.full((k,), 0, tl.int32)
# ... loop ...
# Single vector store per row
tl.store(topk_values_ptr + row * k + store_offsets, vals_vec)
tl.store(topk_indices_ptr + row * k + store_offsets, idxs_vec)
```

**可替代方向**: 循环内逐元素 store 也可工作，但向量 store 减少 GM 写指令数。

#### L3.5 小 shape profiler 测量异常处理（已验证有效）

**问题**: framework 延迟 < 0.1ms 的 case，profiler 测量误差导致 speedup 异常高（2.5x~1251x）

**解决方案**:
1. 在 benchmark 后过滤异常 case：排除 framework 延迟 < 0.1ms 或 speedup > 2.5x 的 case
2. 重新计算几何平均加速比
3. 报告时注明过滤的 case 数量和原因

```python
MIN_FRAMEWORK_MS = 0.1   # 最小 framework 延迟阈值
MAX_SPEEDUP = 2.5        # 最大合理加速比

valid_cases = [
    r for r in per_shape_results
    if r['framework']['avg_latency_ms'] >= MIN_FRAMEWORK_MS
    and r['speedup_vs_torch'] is not None
    and r['speedup_vs_torch'] <= MAX_SPEEDUP
]
```

**可替代方向**: 增加 repeats 次数（50→100）可能减少测量误差，但无法完全消除系统噪声。

#### L3.6 大 NUM_EXPERTS 自适应 chunk 策略（2026-07-02 验证）

**问题**: 当 NUM_EXPERTS > 1536 时，单次整行 load 在 Ascend 上产生错误结果（输出大部分为 0）；而固定小 chunk（如 1024）会把大 E shape 拆成过多 chunk，内存遍历次数高，性能跌至 0.04x 以下。

**解决方案**: 采用自适应 CHUNK_E + online softmax + all-candidates 寄存器合并：
1. **单 chunk 安全阈值**: NUM_EXPERTS <= 1536 走整行寄存器路径（与 L1.8 一致，但实测安全上限为 1536 而非 1024）。
2. **大 E chunk 选择**: 
   - 1536 < NUM_EXPERTS <= 2048：CHUNK_E = 2048（pow2，单 chunk）。
   - NUM_EXPERTS > 2048：CHUNK_E = 4096（在 UB 容量内，最大化每 chunk 利用率，减少 chunk 数）。
3. **online softmax**: 第一遍遍历同时更新 global_max 与 global_sum，避免 3-pass。
4. **all-candidates 寄存器缓冲区**: 每 chunk 选出 top-K 后存入大小为 `NUM_CHUNKS * K` 的寄存器向量，最后统一选全局 top-K，避免每 chunk 与全局 top-K 反复合并。
5. **mask 加载**: 对 chunk 使用 `tl.load(..., mask=offsets < NUM_EXPERTS, other=-inf)`，避免越界并保证大 E 向量稳定。

**性能基准**: 该策略将 50 case 几何平均从 Phase 3 基线 0.46x 提升至 0.66x（仍低于 0.8x 目标，主要瓶颈为超大 batch + 大 E 的标量 row 循环）。

**可替代方向**: 对超大 batch 场景可进一步尝试 ROW_TILE=2/4 的行向量化，但需处理 2D top-K 候选缓冲区，复杂度较高。

### §4.4 MoeGatingTopKSoftmax 性能基准（几何平均）

| Shape 类型 | 典型加速比 | 说明 |
|-----------|-----------|------|
| 小 shape (batch<=32) | 1.0x - 2.5x | 调度开销占比高，profiler测量不可靠，需过滤后评估 |
| 中 shape (32<batch<=512) | 0.4x - 0.8x | 主要优化区间 |
| 大 shape (batch>512) | 0.15x - 0.5x | 受限于内存带宽和 top-k 迭代开销 |
| 3D shape | 0.12x - 0.17x | 大内存 footprint，性能最差 |

**关键结论**（2026-06-24 更新）：
1. 纯寄存器路径（无 GM temp buffer）是正确性和性能的关键
2. `tl.argmax` 在寄存器向量上可靠，但在 GM 往返后不可靠
3. `grid_size = num_cores` 且 `num_cores` 参数匹配 grid 是正确性的硬性要求（T1）
4. bool finished 直接传入可避免 host-side cast 开销和 int8 警告
5. `multibuffer=True, unit_flag=True` 编译选项在本算子上未带来性能提升（v3: 0.8527x vs 基线 0.8836x，v9: 0.7353x vs 基线 0.7700x，v10: 0.5720x vs 基线 0.8194x），不建议默认开启（T6）
6. 该算子在 Ascend 上整体慢于 PyTorch（0.67x~0.89x），主要瓶颈是大 shape 的 top-k 迭代和内存带宽
7. 小 shape case（framework 延迟 <0.1ms）的 profiler 测量不可靠，应设置最小延迟阈值或排除这些 case
8. 环境变量 `LLVM_ROOT` 必须正确设置，否则 Triton 编译失败（symbol lookup error）
9. CANN 9.1.0 与 Triton 存在兼容性常量名差异（`RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE` → `RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE`）

---

## §5 AttentionSoftmaxWithSoftcappingAndDropout 算子（reduce）

**算子类别**: `reduce`（行级归约 + 前置 elementwise 变换 softcapping）
**典型特征**: Gemma3 风格 softcapping `tanh(x/30)*30` + 行级 softmax(dim=-1)，多 dtype (fp32/fp16/bf16) 混合输入
**性能基准**: **1.1388x**（geomean, 50/50），4-kernel 分离 + 分核优化（grid 钳制到 num_cores + 循环处理多块）

### §5.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 bf16/fp16 输入下 softcapping 必须拆分为独立 kernel（见 §1 T5）
- **必须** 将 `tanh(x/scale)*scale` 拆分为 div / tanh / mul 三个独立 kernel，每个 kernel 通过 GM store+load 往返强制中间 dtype 舍入。
- **禁止** 在 bf16/fp16 输入下用单 kernel 完成 `tanh(x/30)*30` 这类 softcapping 表达式。
- **Why:** Triton Ascend 编译器会将 `tanh(x/scale)*scale` 融合为单一 fp32 计算，跳过中间 bf16/fp16 舍入。PyTorch 参考实现在每个算子处物化中间结果为输入 dtype，融合后逐 bit 不匹配。
- **How to apply:** 当算子含 softcapping 且输入 dtype 为 bf16/fp16 时，必须拆分为独立 kernel。fp32 输入下不触发（融合后仍 fp32，无舍入损失）。

#### L1.2 softcapping 各子算子须 fp32 计算 + cast 回输入 dtype
- **必须** 在 div/tanh/mul kernel 内部将输入 load 后 cast 到 fp32，在 fp32 下做 div/tanh/mul，再 cast 回输入 dtype 后 store GM。
- **Why:** PyTorch 参考实现 `attn_weights / 30.0`、`torch.tanh(...)`、`clamped * 30.0` 中 Python float 30.0 会触发 PyTorch 内部提升到 fp32 计算，再 cast 回输入 dtype。若 kernel 内用输入 dtype 直接计算，bf16/fp16 下除法/乘法精度不足。
- **How to apply:** `sc_fp32 = tl.full((), SOFTCAP, dtype=tl.float32); scaled = (x_fp32 / sc_fp32).to(in_dtype)`

#### L1.3 softmax kernel 必须 fp32 内部计算（见 §1 T2）
- **必须** 在 softmax kernel 内部将输入 load 后 cast 到 fp32，再做 max/exp/sum/div。
- **Why:** 参考实现 `F.softmax(x, dim=-1, dtype=torch.float32)` 显式指定 fp32 计算。
- **How to apply:** load → cast fp32 → where(mask, x, -inf) → max → exp → where(mask, exp, 0) → sum → div → cast output dtype。

#### L1.4 mask 元素必须排除出归约
- **必须** 在 max 前用 `tl.where(mask, x, -inf)`，在 sum 前用 `tl.where(mask, exp_val, 0.0)`。
- **Why:** padding 元素若参与 max 会污染结果；若参与 sum 会让 sum 偏大。
- **How to apply:** load 时 `other=-inf`，max 前 where，exp 后 where 为 0 再 sum。

#### L1.5 引入循环后必须收紧 UB budget
- **必须** 当 kernel 内引入 `for block_id in range(pid, n_blocks, num_pids)` 循环时，UB budget 从 128KB 收紧到 48KB。
- **Why:** 循环展开后编译器需多份副本空间，大 K (BLOCK_N=1024) + ROW_TILE=16 时 2D tensor 占 64KB，循环内若 multibuffer 会溢出 192KB UB，触发 BiShengIR `ub over` 编译错误。
- **How to apply:** `_pick_row_tile` 中 `ub_budget = 48 * 1024`，确保 `ROW_TILE * BLOCK_N * 4 <= 48KB`。

### §5.2 Layer 2: 算法骨架

#### L2.1 Host 侧分支决策树

```
输入 x [B, H, Q, K], dtype ∈ {fp32, fp16, bf16}
    ↓
view → x2d [num_rows=B*H*Q, K]
    ↓
判断算子是否含 softcapping:
  ├─ 是 → 4-kernel 分离架构（div / tanh / mul / softmax）   # L1.1
  └─ 否 → 1-kernel softmax
    ↓
BLOCK_N = next_pow2(K), 上限 4096                            # G2
    ↓
ROW_TILE 自适应选择:
  num_rows >= 512 → ROW_TILE = 16
  num_rows >= 128 → ROW_TILE = 8
  num_rows >= 32  → ROW_TILE = 4
  num_rows >= 8   → ROW_TILE = 2
  otherwise       → ROW_TILE = 1
  + UB guard: ROW_TILE * BLOCK_N * 4 <= 48KB（循环场景，L1.5）
    ↓
num_cores = torch_npu.npu.npu_config.get_device_limit(0).get('vector_core_num', 40)  # G1
natural_blocks = ceil(num_rows / ROW_TILE)
grid_size = min(natural_blocks, num_cores)                   # T1，分核优化关键
grid = (grid_size,)
    ↓
逐 kernel 启动（每个 kernel 内部循环处理多块）
```

#### L2.2 多核并行骨架模式（分核优化版）

**模式 - grid 钳制 + 循环处理多块**:
```python
@triton.jit
def kernel(x_ptr, y_ptr, num_rows, K, stride_row, num_pids,
           BLOCK_N: tl.constexpr, ROW_TILE: tl.constexpr):
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(num_rows, ROW_TILE)
    # grid 钳制到 num_pids 后，每个 program 处理多个 block（连续划分，stride=num_pids）
    for block_id in range(pid, n_blocks, num_pids):
        row_start = block_id * ROW_TILE
        row_offs = row_start + tl.arange(0, ROW_TILE)[:, None]
        col_offs = tl.arange(0, BLOCK_N)[None, :]
        mask = (row_offs < num_rows) & (col_offs < K)
        # ... load → compute → store
```

**关键点**:
- `num_pids` 作为 kernel 入参传入（= grid_size），用于循环 stride（T1）
- `for block_id in range(pid, n_blocks, num_pids)` 是连续划分（符合 checklist「禁止交织」规范）
- grid 不再远超核数，避免 NPU 串行调度开销

### §5.3 Layer 3: 关键技巧

#### L3.1 分核优化的 grid 钳制 + 循环模式

```python
# Host 侧
num_cores = torch_npu.npu.npu_config.get_device_limit(0).get('vector_core_num', 40)
natural_blocks = (num_rows + ROW_TILE - 1) // ROW_TILE
grid_size = natural_blocks if natural_blocks < num_cores else num_cores
grid = (grid_size,)

# Kernel 内
@triton.jit
def kernel(..., num_pids, ROW_TILE: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(num_rows, ROW_TILE)
    for block_id in range(pid, n_blocks, num_pids):
        row_start = block_id * ROW_TILE
        # ... 2D block 处理
```

**收益**: 相比"每个 program 处理 1 块、grid=natural_blocks"的朴素模式，grid 钳制到大 K (1024) + 大 num_rows 场景加速比从 1.06x 提升到 1.14x（+7.6%）。

**可替代方向**: 可用 `@triton.autotune` 让编译器自动搜索最优 ROW_TILE，但多 shape 场景下 autotune 会触发多次重新调优，对小 shape 可能引入额外开销。

#### L3.2 4-kernel 分离的 div/tanh/mul 模板

```python
@triton.jit
def _div_kernel(x_ptr, s_ptr, num_rows, K, stride_row, num_pids,
                SOFTCAP: tl.constexpr, BLOCK_N: tl.constexpr, ROW_TILE: tl.constexpr):
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(num_rows, ROW_TILE)
    for block_id in range(pid, n_blocks, num_pids):
        row_offs = block_id * ROW_TILE + tl.arange(0, ROW_TILE)[:, None]
        col_offs = tl.arange(0, BLOCK_N)[None, :]
        mask = (row_offs < num_rows) & (col_offs < K)
        in_dtype = x_ptr.dtype.element_ty
        x = tl.load(x_ptr + row_offs * stride_row + col_offs, mask=mask, other=0.0)
        # fp32 计算 + cast 回输入 dtype（对齐 PyTorch 参考实现，T2/L1.2）
        x_fp32 = x.to(tl.float32)
        sc_fp32 = tl.full((), SOFTCAP, dtype=tl.float32)
        scaled = (x_fp32 / sc_fp32).to(in_dtype)
        tl.store(s_ptr + row_offs * stride_row + col_offs, scaled, mask=mask)
```

**关键点**:
- `sc_fp32 = tl.full((), SOFTCAP, dtype=tl.float32)` — 常量用 fp32，保证 div/mul 在 fp32 下计算
- `.to(in_dtype)` — 显式 cast 回输入 dtype，但真正强制舍入的是 store 到 GM 再由下个 kernel load（T5）
- tanh/mul kernel 结构相同，只是中间算子不同

#### L3.3 softmax kernel 的 fp32 + mask 排除归约

```python
@triton.jit
def _softmax_kernel(x_ptr, y_ptr, num_rows, K, stride_row, num_pids,
                    BLOCK_N: tl.constexpr, ROW_TILE: tl.constexpr):
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(num_rows, ROW_TILE)
    for block_id in range(pid, n_blocks, num_pids):
        row_offs = block_id * ROW_TILE + tl.arange(0, ROW_TILE)[:, None]
        col_offs = tl.arange(0, BLOCK_N)[None, :]
        mask = (row_offs < num_rows) & (col_offs < K)
        x = tl.load(x_ptr + row_offs * stride_row + col_offs, mask=mask, other=-float("inf"))
        x_fp32 = x.to(tl.float32)
        x_fp32 = tl.where(mask, x_fp32, -float("inf"))
        row_max = tl.max(x_fp32, axis=1)[:, None]   # ← [:, None] 关键
        shifted = x_fp32 - row_max
        exp_val = tl.exp(shifted)
        exp_val = tl.where(mask, exp_val, 0.0)
        row_sum = tl.sum(exp_val, axis=1)[:, None]
        out_fp32 = exp_val * (1.0 / row_sum)
        tl.store(y_ptr + row_offs * stride_row + col_offs,
                 out_fp32.to(y_ptr.dtype.element_ty), mask=mask)
```

**注意**: `tl.max(x, axis=1)` 返回 1D tensor，必须加 `[:, None]` 才能 broadcast 回 2D。

### §5.4 AttentionSoftmaxWithSoftcappingAndDropout 性能基准（几何平均）

| Shape 类型 | 典型加速比 | 说明 |
|-----------|-----------|------|
| 大 K (1024) + fp16 | 1.58~1.60 | 最佳区间，分核优化充分发挥多核并行 |
| 中等 shape + bf16/fp16 | 1.1~1.6 | 良好区间，2D block 充分利用 vector core |
| 小 shape + fp32 | 0.73~0.93 | 较弱区间，4-kernel GM round-trip 开销在小 shape 下显著 |
| 极小 shape (K=1) | 1.33 | case 47 [1,1,1,1]，单元素 softmax 退化 |

**关键结论**: 4-kernel 分离是精度必需（bf16/fp16 编译器融合问题），分核优化是性能关键（grid 钳制 + 循环处理多块，大 K 场景 +7.6%）。小 shape + fp32 场景因 4-kernel GM round-trip 开销仍有优化空间。

## §6 LightningIndexer 算子（topk-select）

### 算子背景与语义

LightningIndexer 是 Transformer 大 KV cache 推理中**稀疏注意力的索引器**：对每个 query token，
它先计算该 token 与所有 key token 的注意力分数（QK^T → ReLU → 按头权重归约），再选出分数最高的
top-K 个 key 位置，供后续稀疏注意力只在这些位置做真实计算。选位的成本通常不低于打分成本
（`topk-select` 类算子的共同特征），因此本算子的优化重点在于**打分链与选位结构的整体设计**，
而不是单纯把 GEMM 做快。

本节记录的路线（下称**路线 B**）语义为：输出**按分数降序排列的 top-K 索引**（同分按下标升序，
由稳定排序保证），`sparse_count` 之外的掩码位置填 `-1`/`-inf`。它与 `index-computation.md §5.5`
的路线 A（npu_sort_v2 sort-compaction，升序索引集合语义）**并存不互斥**，选择依据是：

- 路线 A：环境有 `npu_sort_v2` 且 validator 放行 `torch_npu.npu_*`；golden 只需"前 K 个索引的集合"（顺序无关）。
- 路线 B（本节）：`npu_sort_v2` 已弃用/损坏或被 validator 拦截；golden 需要**完整降序 top-K**（顺序敏感），
  此时选位只能依赖放行的 `.sort()` 稳定排序原语，并且分数必须与参考实现**位级一致**才能通过索引的逐位比对。

**算子类别**: `topk-select`
**典型特征**: fp16 QK^T（fp32 累加、fp16 量化输出）→ relu → 按头权重求和（fp32）→ causal mask（-inf）→ 稳定降序 top-K；输出 `int32` 索引 + 可选 `fp32` 分数
**性能基准**: 任务 8-shape（verify 8/8）**5.0528x** vs torch；16-shape（work 0.1M~268M，mode3）vs ops-transformer AscendC 单融合 kernel 核级几何平均 **0.404x**、墙钟 0.325x
**历史最佳版本**: opt_iter_16（初版 2.56x → 16 轮迭代）

### §6.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 分数链必须与参考实现位级一致（本路线唯一可靠的正确性路径）

- **必须** 让实现计算出的 score 与 torch 参考**逐位相同**，而不是"在误差范围内近似"。
  具体做法：① QK^T 用 `tl.dot`（fp16 输入、fp32 累加）后立即 `.to(tl.float16)`，
  与参考 `torch.bmm` 的 fp16 输出位级一致（实测 65536 元素 0 个不匹配——两者共用同一 cube 硬件的 K 维归约序）；
  ② 头归约用 cube dot（见 L1.2）；③ 选位用与参考同源的稳定排序（见 L1.4）。
- **禁止** 任何"近似选位"：把分数转 fp16/bf16 键再排序、用 topk 的平局语义替代稳定排序、
  降低累加精度——这些都会让两个分数接近的位置交换次序，导致输出索引与参考相差不止 1，
  verify 的量化类判定（`|diff| <= 1`）立即失败。
- **Why**：输出是整数索引，验证是**逐位**的。分数只要差 1 ulp，位于 top-K 边界附近的近等分就可能翻边；
  大 K（如 2048）时边界密度高，任何"几乎一致"的方案都会以失败告终。位级一致则 `|diff| = 0`，天然通过。

#### L1.2 头归约必须用 fp32 的块对角 cube dot，禁止向量 FMA 或 fp16 键

- **必须** 用 `m1 [R, R*N]` 块对角矩阵与逐头分数 y 做 `tl.dot`（K = R*N），
  其中 `m1[r, c] = w_all[c]`（当 `c // N == r`）否则为 0。结构零在 cube 的顺序累加中是精确 no-op
  （`x + 0 = x`），因此该 dot 与参考 `torch.bmm([M,1,N],[M,N,S2])`（K=N=8）位级一致——
  实测 K=8/16/32/64/128 五种配置全部 0 不匹配。
- **禁止** ① 用向量 FMA 逐头累加（顺序与 cube 的 K=8 归约不同，实测差 ~1.5e-8，足以翻边）；
  ② 用 fp16 键做 dot（fp16×fp16 乘积 22 bit 精确、无舍入，而参考是 fp32 乘积舍入，两者不一致）。
- **Why**：参考的 K=8 归约顺序是硬件固定的，只有同一硬件的同型 cube dot 能复刻；
  这是"位级一致"约束下不可讨价还价的一点。

#### L1.3 编译器缺陷规避（triton-ascend 3.2.1 / 910B3 实测三处，务必绕开）

- **必须** 避免 `tl.reshape` 作用于 3D tile 或偏移张量：实测产生 NaN 或编译崩溃
  （`LLVM ERROR: unexpected op in rewrite` / `@malloc` 无法编译）。
  R 行合并时改为**连续 flat 行加载**：行 r 的 head h 恰好位于 flat 行 `r*N+h`，
  直接 `fr = pid*(R*N) + tl.arange(0, R*N)` 即可，无需 reshape。
- **必须** 掩码保持 int 比较：fp32 派生的 bool 掩码用于 `tl.load`/`tl.store` 会得到错误结果
  （本算子 3 处独立复现：finalize 的 km、score 的 store mask、valid 掩码）。
  数值比较（分数 vs 阈值）才转 fp32；索引/边界比较保持 int。
- **必须** 实测 BLOCK 阶梯而非凭 UB 估算：本算子 `BLOCK_J=512` 编译失败（需 334KB），
  但 hsum_kernel 的 `y16→f32` 转换编译器会**分段搬运、边转换边送入 dot**（不需要在 UB 中一次性放下完整
  fp32 副本），手工估算会高估占用、可能错杀可行配置。

#### L1.4 选位原语与排序语义（环境约束下的唯一选择）

- **必须** 用 tensor 方法 `x.sort(dim=-1, descending=True)` 完成选位。
- **Why（三层原因叠加，缺一不可）**：
  ① validator 的白名单禁止 `torch.topk` / `torch.cumsum` / `torch.cat` 等 torch 计算调用，
  也禁止 `torch_npu.npu_*` 顶层调用（属"融合算子外包"拦截规则，`npu_sort_v2` 正是 `npu_` 前缀，会被直接标红）；
  ② 本机 `torch_npu.npu_sort_v2` 本身已弃用且运行时损坏——调用会触发 AOE/TBE 初始化，因环境缺 `scipy`
  报 `SetPrecisionMode ... 500001`，且当前版本只返回值、不返回排序索引，而路线 A 的 rank 恢复恰恰依赖该索引
  （注意：npu_sort_v2 是 torch_npu/CANN 接口，与 triton-ascend 无关）；
  ③ `.sort()` tensor 方法不在禁止名单、由设备侧 aclnnSort 执行、实测**稳定**（同分按下标升序，含 -inf 平局），
  与参考 `torch.sort(stable=True, descending=True)` 语义一致。
  三层叠加后，`.sort()` 是唯一放行且满足正确性要求的选位原语。
- 排序输入喂 2D `[rows, S2P]`（`S2P = max(S2, sparse_count)`）；`S2 < sparse_count` 的补齐列由
  `j >= ak` 条件注入 `-inf`，不要依赖 k 载入的零填充参与排序。

#### L1.5 大 shape 必须双路径分派（拆分 vs 融合）

- **必须** 按 `rows * S2P >= 4_000_000` 分派：大 shape 走拆分路径（qk_kernel 与 hsum_kernel 分核、
  y16 中间缓冲），小 shape 保留单融合核。
- **Why**：融合核里每个 tile 要做 2 个 dot。其中头归约 dot 的 M=8、只有 65K MAC（qk dot 的 1/32），
  但每个 dot 都要付出固定的一次"指令发射 + cube↔vector 握手等待"开销——dot 很小、开销却与 dot 大小无关，
  所以 tile 数大时这一税被放大（消融实测：去掉头归约 dot 省 63% 的 score 核时间）；拆分后头归约 dot 可 16 行合并、
  tile 放大，大 shape 全流水 -9~15%。但拆分要额外一次 kernel 调度 + 一份 `[rows*N, S2P]` 中间缓冲，
  小 shape 反而回退（任务 8-shape 4.39 vs 5.16），所以必须分派而不是一刀切。
- R 行合并**必须**满足 `S1 % R == 0`（R 行必须同 batch，k 按 batch 共享），否则取错 key 数据。

#### L1.6 结构参数边界（UB / grid）

- qk_kernel 的 R 上限是 8：R=16 时 dot 的 fixpipe 输出 `[128, 256] fp32` 必须**一次性整块搬进 UB**
  （没有分段搬运的余地），连同 q/k 载入共需 320KB > 192KB。hsum_kernel 可以 R=16（其 y16→f32 转换
  由编译器分段搬运，216KB 可行）。
- mix 算子 grid ≤ cube 核数（910B3 = 20）。

### §6.2 Layer 2: 算法骨架

```python
# host 侧 forward 骨架
rows, S2P = B * S1, max(S2, sparse_count)
aq_dev = 实际长度 or torch.full((B,), S1, int32)     # None 时用常量张量, 禁止传 None 进 kernel
ak_dev = 实际长度 or torch.full((B,), S2, int32)

score_buf = torch.empty((rows, S2P), torch.float32)
if rows * S2P >= 4_000_000:                          # L1.5 大 shape 拆分路径
    y_buf = torch.empty((rows * N, S2P), torch.float16)   # y 值即 fp16, GM 往返无损
    r_a = 8 if S1 % 8 == 0 else ...                  # L1.5 同批合并门控
    qk_kernel[(cdiv(rows, r_a),)](q, k, y_buf, ..., BLOCK_J=256, R=r_a, multibuffer=True)
    r_b = 16 if S1 % 16 == 0 else ...
    hsum_kernel[(cdiv(rows, r_b),)](y_buf, w, score_buf, aq_dev, ak_dev, ..., BLOCK_J=256, R=r_b)
else:                                                # 小 shape 融合路径
    score_kernel[(cdiv(rows, 8),)](q, k, w, score_buf, aq_dev, ak_dev, ..., BLOCK_J=128, R=8)
sv, si = score_buf.sort(dim=-1, descending=True)     # L1.4 唯一放行选位原语
finalize_kernel[(rows,)](sv, si, out_idx, out_val, S2P, K, next_pow2(K))   # -inf -> -1/-inf, int64 -> int32
```

- **qk_kernel**：flat 行加载 q `[R*N, D]` → `tl.dot(q, tl.trans(k_tile))` → `.to(fp16)` → relu（fp16 域，与参考位级等价）→ 存 y16。
- **hsum_kernel**：块对角 m1（循环不变量外提，见 L3.1）→ `tl.dot(m1, y16.to(fp32))` → causal mask（`s1>=aq | j>=ak | mode3: j >= ak-aq+s1+1`）→ 存 scores fp32。
- **消融归因法（决定是否拆分的标准动作）**：对 score 核做"完整 / 无头归约 dot / 无 cast-relu / 无 qk dot"四变体实测
  边际成本。本算子实测访存地板仅 6%（k 载入+scores 写回+mask），dot+逐元素流水占 94%——据此把资源投向流水结构而非访存。

### §6.3 Layer 3: 关键技巧

#### L3.1 连续 flat 行加载 + 块对角头归约（位级一致的实现载体）

```python
# q/w 按连续 flat 行加载: 行 r 的 head h 位于 flat 行 r*N+h (避开 tl.reshape 缺陷)
fr = pid * (R * N) + tl.arange(0, R * N)
q_tile = tl.load(q_ptr + fr[:, None] * D + dd[None, :],
                 mask=valid_flat[:, None], other=0.0)          # [R*N, D] f16
w_all = tl.load(w_ptr + fr, mask=valid_flat, other=0.0).to(tl.float32)

# 块对角 m1 是循环不变量, 外提 tile 循环
cc = tl.arange(0, R * N)
m1 = tl.where((cc // N)[None, :] == tl.arange(0, R)[:, None], w_all[None, :], 0.0)
acc2 = tl.dot(m1, y)                                           # [R, BJ] fp32, K=R*N 结构零精确 no-op
```

**可替代方向**：若 golden 不要求顺序敏感输出，可改用 index-computation.md §5.5 的 rank 编码路线，
省去 fp32 头归约的精度约束。

#### L3.2 causal mask 阈值外提（削减 tile 内向量工作）

```python
cond_row = s1f[:, None] >= aqf      # 行掩码与 j0 无关, 外提
thr3 = (akf - aqf) + s1f + 1.0      # mode3 阈值同样外提
# tile 循环内仅剩 j 维比较
cond = cond_row | (jf[None, :] >= akf) | (jf[None, :] >= thr3[:, None])   # mode3 时含第三项
```

**可替代方向**：每 tile 重算正确但略慢；若去掉外提需确认编译器 LICM 是否自动完成（本平台实测不自动）。

#### L3.3 大 shape 拆分后的 tile 配置（探查驱动，勿凭估算）

| kernel | 配置 | 单 tile 成本（B2,2048,16384 实测） | 说明 |
|--------|------|------------------------------------|------|
| qk_kernel | R=8 / BLOCK_J=256 | 67ns（2.1M MAC，63 TFLOPS） | R=16 需 320KB UB，锁死 |
| hsum_kernel | R=16 / BLOCK_J=256 | 320ns（524K MAC fp32） | BJ 128→256 收益 -30%；multibuffer 开关中性 |

**可替代方向**：hsum 的 320ns 由 fp32 dot 税 + K=128 结构零构成，是位级一致约束的固有成本；
解除该约束（golden 允许近似）可换 fp16 键大幅提速，但本路线不可。

### §6.4 性能演进

| 版本 | 任务 8-shape | 16-shape vs op (核级) | 关键变更 |
|------|------------|----------------------|---------|
| iter_0（单行融合核 + .sort） | 2.56x | — | 基线 |
| opt_1/3（#21 M 维合并 R=2→4） | 3.74 / 4.75x | — | 头归约块对角 + flat 行加载 |
| opt_11（R=8 + grid≤核数） | 5.16x | 0.339x | checklist 规则 5 |
| opt_14（拆分 + 双路径分派） | 5.12x | 0.362x | 大 shape 全流水 -9~11% |
| **opt_16（hsum BLOCK_J 128→256）** | **5.05x** | **0.404x** | 变体探查 -30% |

## §7 FFN 算子（dual-gemm-activation）

### 算子背景与语义

FFN 是 Transformer 每层的前馈网络：`y = act(x@W1 + b1) @ W2 + b2`，两段 GEMM 串行、中间夹逐元素激活。
GLU 变体（geglu/swiglu/reglu）中 W1 的输出列宽翻倍（N1 = 2*KH），左半过激活后与右半逐元素相乘。
MoE 扩展中 W1/W2/b1/b2 带专家维 `[E, ...]`，`expert_tokens[E]` 给出每专家的 token 数，x 行按专家连续分段。

本算子的优化重点不是单个 GEMM 的极限 tiling，而是**两段 GEMM 的拆分结构、激活/门控的 epilogue 融合位置、
以及与参考实现逐 op 舍入语义的对齐**。

**算子类别**: `dual-gemm-activation`
**典型特征**: x[M,K1] fp16/bf16；7 种激活（gelu/fastgelu/relu/silu/geglu/swiglu/reglu）；b1/b2 可选；MoE 扩展（expert_tokens 分组，空专家需自然跳过）
**性能基准**: dense 10-case 几何平均 **1.1615x** vs torch（benchmark 10/10，verify 8/10）；MoE 5 场景 vs CANN 内置 FFN 融合算子（inner_precise=1，msprof kernel Duration）几何平均 **0.83x**（0.72~0.95x），精度 6/6 全过（fp64 参考，含空专家/极不均衡/GLU+MoE——后者内置融合算子不支持）。第二轮性能优化（寄存器单 pass epilogue + 双 kernel tile 解耦 + MoE tile 双路径 + GM_EPI 大权重路径）后：dense **1.29x** / MoE **1.00x** / 全 15 case **1.18x** vs 内置
**历史最佳版本**: opt_iter_8（初版 MoE 0.83x → 8 轮迭代 1.00x 追平内置，全部 15 行不劣于初版）

### §7.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 双 kernel 结构强制（GEMM1+act → H → GEMM2）
- **必须**将两段 GEMM 拆为两个 kernel 串行 launch：Kernel1 产出中间量 H 落 GM，Kernel2 读 H 产出 Y。
- **禁止**单 kernel 融合两段 GEMM（GEMM2 沿 K 轴完整依赖 GEMM1 全量输出，需要跨块栅格同步，Triton-Ascend 无高效原语）。
- **Why:** 单 kernel 需 grid 级同步，退化为串行或 atomic 轮询，实测不可行；双 kernel 各自独立最优 tiling。

#### L1.2 GLU 变体必须双累加器 epilogue 融合
- **必须**对 geglu/swiglu/reglu 在 GEMM1 kernel 内用双累加器：K 循环同时累计左半（列 j）与右半（列 j+KH）两个 [BM,BN] fp32 累加器，共享同一 A tile，epilogue 做 `h = act(left) * right`。
- **禁止**先算完整 g[M,N1] 落盘再启动第二个逐元素 kernel 做 split+act+mul（多两趟 GM 访存 + 一次额外 launch）。
- **Why:** 双累加器方案 A tile 复用两次 dot（GM 中 A 流量减半），且消除 g 的落盘/回读。
- 与 L1.1 的关系：L1.1 约束两段 GEMM **之间**必须拆；L1.2 约束 GEMM1 与激活**之间**必须融。

#### L1.3 固定核数启动 + Swizzle2D + 自适应分组方向
- **必须** `grid=(NUM_CORES,)`，kernel 内 `for block_idx in range(pid, NUM_BLOCKS, NUM_CORES)` 每核循环处理多块（继承 G1/G4/T1）。
- **必须**块序 Swizzle2D 重排（GROUP_SIZE=4 起步，autotune [1,2,3,4,5,8]）；按 M/N 比例自适应方向：M≥N 行优先 `tl.swizzle2d`，M<N 列优先手动分组。
- **Why:** 提升权重矩阵的 L2 局部性；权重是两段 GEMM 的最大重复访存源。

#### L1.4 激活与控制参数编译期分派
- **必须** ACT_TYPE / IS_GLU / HAS_B1 / HAS_B2 / HAS_EXPERT 全部 `tl.constexpr`，host 侧将激活字符串映射为枚举整数后传入。
- **禁止**激活字符串或运行时分支进入 kernel（device 侧无字符串比较；运行时分支会拖累所有路径的编译产物）。

#### L1.5 GLU 双累加器受 L0C 容量约束
- **必须** GLU 路径满足 2 × BM × BN × 4B ≤ L0C（910B3/DAV_2201: 128KB，即 BM×BN ≤ 16384），建议 (128,128) 起步 autotune。
- 非 GLU 路径单累加器，可沿用 (BM,BK,BN)=(128,256,256)（fp16/bf16）。
- **Why:** tl.dot 累加器映射 L0C，双累加器超容 spill 到下一级存储，性能骤降。

#### L1.6 维度约束 host 侧前置校验
- **必须**在 wrapper 中校验：K1 == N2；GLU 时 N1 == 2*K2（KH = N1//2），非 GLU 时 N1 == K2（KH = N1）。不满足直接抛错，不进 kernel。
- **Why:** 语义前提；错误维度进 kernel 产生越界读或静默错值。

#### L1.7 中间量 H 精度与分配
- **必须** H 由 host 侧 `torch.empty` 分配于 NPU，dtype 与输入一致；Kernel1 存储前 fp32→输入 dtype 舍入，Kernel2 以输入 dtype 读入（fp32 累加）。
- **Why:** 遵循 T2/T5；H 落盘精度高于输入 dtype 无收益且浪费带宽。

#### L1.8 Epilogue 舍入链必须 GM 物化（T7 的 GEMM epilogue 形态）
- **必须**在 GEMM K 循环结束后，将 `round_dtype(acc)` 先 `tl.store` 到 GM（输出缓冲或 scratch），再 `tl.load` 回来做 bias/激活链；GEMM2 的 bias 同理（store → reload → +b2 → store）。
- **禁止**依赖 `acc.to(DT).to(tl.float32)` round-trip cast 保留舍入——编译器将 dot 累加器派生值的有损 cast 判定为 no-op 直接消除（bitcast / `tl.where` 强制依赖 / `tl.debug_barrier` 拦截同样被消除，勿再尝试）。
- 与 T5 的关系：T5 是逐元素链拆**独立 kernel**；FFN 双 kernel 结构下**无需再拆**，kernel 内 store-reload 到自身输出缓冲即可（T7 的共性约束，此处为 GEMM epilogue 具体形态）。
- **性能变体（verify 口径不要求 bit-exact 时）**：epilogue 可改为寄存器单 pass（K 循环后直接在 fp32 累加器上做 bias+激活，单次舍入单次 store），移除 store-reload 三趟 GM 遍历（dense 几何平均 1.06→1.21x vs 内置）；此时 UB 活跃量按 ~4/5 单元（非 GLU/GLU）× BM×BN×4B 预算，(128,128) 溢出、非 GLU (64,128) / GLU (128,64) 通过；权重流量主导的大 K 路径（K1≥4096）则应保留 GM 物化形态以换取 BM=128（W1 只读一遍）。
- **Why:** 逐 op 舍入链（fp32 opmath → 每步 round 回输入 dtype）是内置算子的语义；cast 被消除后激活作用在未舍入 fp32 上。实测 GM 物化后与参考 100% bit-exact。

#### L1.9 Epilogue 激活链工作区受 UB 容量约束（BN_EPI 子切片）
- **必须**保证 epilogue 逐 op 舍入链的活跃工作区（reload tile + 各中间量）≤ UB（910B3: 192KB）。erf/tanh 全链 + store-reload 在 [128,128] tile 下即超限。
- **How to apply:** epilogue 按 BN_EPI（实测 64）对 BN 子切片循环 `for ns in range(0, BN, BN_EPI)`，工作区降为 BM×BN_EPI；K 循环 tile 宽度（L0C 约束）与 epilogue 宽度（UB 约束）解耦。

#### L1.10 MoE 专家分组用前缀和 + device 侧扫描定位（T8 的 MoE 形态）
- **必须** host 侧构建 `tokens_prefix[E+1]` 与 `blk_prefix[E+1]`（每专家 M-block 数前缀），kernel 内按全局 block_idx 扫描定位专家 e 与局部 block；权重/偏置按 `e.to(tl.int64) * K * N` 基址偏移寻址（int64 防大 E×K×N 溢出，T8）。
- **必须**自适应 BM：最小化 `sum(ceil(tokens_e/BM))`（每 block 重读该专家全部权重，MoE 小 token 场景权重流量主导），平局取大 BM。
- 前缀张量上传用 pinned 环形缓冲 + `non_blocking=True`（T8，避免每次调用同步 H2D ~150us）。
- **Why:** x 行按专家连续分段时无需 gather/scatter，block 映射即可覆盖；空专家前缀不变自然跳过。

#### L1.11 两个 kernel 的 tile 与前缀解耦（gemm2 无激活链，BM 可更大）
- **必须** gemm2 使用独立于 gemm1 的 tile：gemm1 的激活 epilogue 把 UB 活跃量推到 ~4/5 单元（非 GLU/GLU，单元 = BM×BN×4B），BM 受限；gemm2 只有 acc+可选 bias 约 2 单元，dense 下 M%128==0 时 BM2=128 让 W2 只读一遍（MTE2 流量减半）。
- **禁止** M 非 128 整数倍时放大 BM2（padding + L0A 满载反而劣化，M=320 实测回落）；禁止 MoE 下 gemm2 沿用 gemm1 的 BM——gemm1 用小 BM 压 padding 会导致 W2 按小 BM 多重读。
- **How to apply:** MoE 下 host 侧为 gemm2 单独构建 blk2 前缀（BM 只影响 M-block 前缀，BN/BK 不影响），pinned 环形缓冲扩展携带两套前缀。

### §7.2 Layer 2: 算法骨架（Agent 可参考架构）

```python
# host 分派骨架
ACT_MAP = {"gelu":(0,0), "fastgelu":(1,0), "relu":(2,0), "silu":(3,0),
           "geglu":(4,1), "swiglu":(5,1), "reglu":(6,1)}
act_type, is_glu = ACT_MAP[activation]
KH = N1 // 2 if is_glu else N1
# 校验 K1==N2; is_glu ? N1==2*K2 : N1==K2（L1.6）
# 常数按 dtype 分派: fp16 预舍入到 f16 再用; bf16 保持 fp32（L3.2）
H = torch.empty((M, KH), dtype=x.dtype, device=x.device)
# BM 自适应: M>64→128, M>32→64, else 32（MoE 按专家 token 分布取 argmin 总 block 数）
# BN/BK 分派: MoE 大 KH 非 GLU → (256,128); 其余 → (128,256)（L3.3）
ffn_gemm1_act[(NUM_CORES,)](x, W1, b1, H, G_scratch, M, K1, KH,
                            tokens_t, blk_t, E, BM=BM, BN=BN, BK=BK,
                            BN_EPI=64, GROUP_SIZE=4,
                            ACT_TYPE=act_type, IS_GLU=is_glu, HAS_B1=has_b1,
                            HAS_EXPERT=has_expert, C_SQRT2=..., C_A=..., C_B=...)
ffn_gemm2[(NUM_CORES,)](H, W2, b2, Y, M, KH, N2,
                        tokens_t, blk_t, E, BM=BM, BN=BN, BK=BK,
                        GROUP_SIZE=4, HAS_B2=has_b2, HAS_EXPERT=has_expert)
```

MoE host 侧前缀构建：

```python
# expert_tokens: [E]，x 行按专家连续分段
et = expert_tokens.tolist()               # 校验 sum(et)==M, 非负, weight1/2 为 3-D [E,K,N]
BM = argmin_{cand in (32,64,128)} sum(ceil(t_e/cand))   # 平局取大 BM（L1.10）
tok_prefix = [0]; blk_prefix = [0]
for t in et:
    tok_prefix.append(tok_prefix[-1] + t)
    blk_prefix.append(blk_prefix[-1] + (t + BM - 1) // BM)
# pinned 环形缓冲 non_blocking 上传 (E+1) int32 两个张量（L3.4）
```

Kernel 内 MoE block 定位（两个 kernel 同构）：

```python
if HAS_EXPERT:
    HB = tl.cdiv(KH, BN)                  # 每专家的 N-block 数
    NUM_BLOCKS = tl.load(blk_m_ptr + E) * HB
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        # 标量前缀扫描定位专家（实测比向量化 tl.sum 快 ~25%，可与 MTE1 流水重叠）
        e = 0
        for i in range(1, E + 1):
            e += tl.where(block_idx >= tl.load(blk_m_ptr + i) * HB, 1, 0)
        local = block_idx - tl.load(blk_m_ptr + e) * HB
        bi0, bj0 = local // HB, local % HB
        bi, bj = tl.swizzle2d(bi0, bj0, mb_e, HB, GROUP_SIZE)
        w_base = e.to(tl.int64) * K1 * N1_cols   # int64 基址偏移（T8）
        _tile(...)                                # 与非 MoE 共用 tile 函数
```

### §7.3 Layer 3: 关键技巧（可参考但不可直接复制）

#### L3.1 GM 物化 epilogue（精度对齐核心）

```python
# K 循环结束后（acc 为 fp32 累加器）:
DT = h_ptr.dtype.element_ty
# 步骤1: round 后的值强制落 GM（真实 store 前的舍入不会被编译器消除）
tl.store(h_ptr + rm[:, None] * KH + rn[None, :], acc_l.to(DT), mask=msk)
if IS_GLU:  # 右半写 scratch，与左半同法物化
    tl.store(g_scratch_ptr + ..., acc_r.to(DT), mask=msk)
# 步骤2: BN_EPI 子切片 reload + bias + 激活链（逐 op: fp32 opmath → round 回 DT）
for ns in range(0, BN, BN_EPI):
    g = tl.load(h_ptr + ..., mask=msk_e, other=0.0)
    if HAS_B1:
        bl = tl.load(b1_ptr + b_base + rn_e, mask=mask_e, other=0.0)
        g = (g.to(tl.float32) + bl.to(tl.float32)).to(DT)   # bias 先加并落 dtype
    h = _act_chain(g, ACT_TYPE, C_SQRT2, C_A, C_B)          # 逐 op 舍入激活链
    tl.store(h_ptr + ..., h, mask=msk_e)
```

**可替代方向**: 若未来编译器修复 cast 消除，可去掉 scratch 改回寄存器 round-trip；store-reload 的 GM 代价已实测被 BN_EPI 子切片摊薄。

#### L3.2 逐 op 舍入激活链 + 常数 dtype 分派（微基准对齐内置算子语义）

```python
@triton.jit
def _act_chain(g, ACT_TYPE: tl.constexpr, C_SQRT2: tl.constexpr, ...):
    DT = g.dtype
    x = g.to(tl.float32)
    if ACT_TYPE == 0:  # gelu(erf): 每步 fp32 算完立即 round 回 DT
        t = (x / C_SQRT2).to(DT)
        e = tl.erf(t.to(tl.float32)).to(DT)
        s = (1.0 + e.to(tl.float32)).to(DT)
        p = (x * 0.5).to(DT)
        y = (p.to(tl.float32) * s.to(tl.float32)).to(DT)
    ...
```

常数 host 侧分派：fp16 用 `float(torch.tensor(c, dtype=torch.float16))` **预舍入**（内置算子在 fp16 下先把常数舍入到 f16 再参与运算）；bf16 保持 fp32 原值。

**可替代方向**: silu 的 `1/(1+exp(-x))` 与 vendor sigmoid 实现等价（实测 bit-exact）；若对齐目标更换需重跑微基准。

#### L3.3 MoE BN=256 dispatch（大 KH 权重流带宽）

```python
if has_expert and not is_glu and KH >= 2048:
    BN, BK = 256, 128     # 权重流带宽显著提升: E16K4096 930→678us, E32K2048 837→590us
else:
    BN, BK = 128, 256     # 非 MoE 路径保持（bitwise 等价要求）
```

编译边界实测：BN=256 在 MoE 大 KH 非 GLU 通过；KH=1024 与 GLU 路径编译失败，需保持 (128,256)。

**可替代方向**: E=64 场景剩余瓶颈为专家扫描（每 block O(E) 标量链），可改 device 侧二分查找（`tl.load` 前缀数组 + log2(E) 步）。

#### L3.4 pinned 环形缓冲上传前缀张量

```python
ring = [(torch.empty(E+1, dtype=torch.int32, pin_memory=True),   # x2 (tok, blk)
         torch.empty(E+1, dtype=torch.int32, device=x.device))   # x2
        for _ in range(64)]            # 每专家数 E 一组，64 槽环形
pin_tok.copy_(torch.tensor(tok_prefix, dtype=torch.int32))
tokens_t.copy_(pin_tok, non_blocking=True)   # 避免每次调用同步 H2D (~150us)
```

### §7.4 性能基准

> 以下两表为**初版（GM 物化 epilogue）**数据；第二轮优化（寄存器单 pass epilogue +
> 双 kernel tile 解耦 + MoE tile 双路径 + GM_EPI）后的演进见本节末性能演进表。

**dense 10-case**（几何平均 1.1615x vs torch，benchmark 10/10，verify 8/10）：

| case | 配置 | verify | speedup |
|---|---|---|---|
| 1 | gelu fp16 M=128 K=1024 N1=1024 | pass | 1.1156 |
| 2 | gelu fp16 M=320 K=1024 N1=1536 b1+b2 | pass | 1.0253 |
| 3 | fastgelu fp16 M=256 K=2048 N1=2048 | pass | 1.3695 |
| 4 | relu fp16 M=64 K=512 N1=512 b1+b2 | pass | 1.0619 |
| 5 | silu fp16 M=128 K=4096 N1=4096 b1+b2 | fail | — |
| 6 | geglu fp16 M=128 K=1024 N1=2048 | pass | 1.3410 |
| 7 | swiglu fp16 M=256 K=2048 N1=4096 b1+b2 | fail | — |
| 8 | reglu fp16 M=64 K=512 N1=1024 | pass | 1.0000 |
| 9 | gelu bf16 M=128 K=1024 N1=1024 | pass | 1.0411 |
| 10 | silu fp16 M=8 K=1024 N1=1024 b1+b2 | pass | 1.0256 |

**MoE 扩展**（vs CANN 内置 FFN 融合算子 inner_precise=1，msprof kernel Duration）：

| 场景 | 本实现 us | 内置算子 us | 加速比 |
|---|---|---|---|
| silu E=8 M=256 K=2048/1024 b1+b2 | 90.8 | 69.9 | 0.77x |
| gelu E=16 M=1024 K=4096/2048 | 676.7 | 612.9 | 0.91x |
| relu E=32 M=512 K=2048/2048 b1 | 589.2 | 559.3 | 0.95x |
| silu E=64 M=1024 K=2048/1024 b2 | 851.9 | 613.5 | 0.72x |
| gelu E=4 M=128 K=1024/512 b1+b2 | 34.1 | 28.1 | 0.82x |

**性能演进**（vs CANN 内置 FFN 融合算子，msprof kernel Duration 几何平均）：

| 版本 | dense 10-case | MoE 5 场景 | 关键变更 |
|------|------------|----------------------|---------|
| iter_5（初版，GM 物化 epilogue） | 1.06x | 0.84x | 基线（bit-exact 对齐逐 op 链） |
| opt_iter_0（寄存器单 pass epilogue） | 1.21x | 0.81x | 移除 store-reload 三趟 GM 遍历 |
| opt_iter_3（MoE tile 双路径） | 1.21x | 0.94x | M/E≤16 或 K1≥4096 → (32,256,128) |
| ir_2（gemm2 tile 解耦 BM2=128） | 1.27x | 0.94x | 无激活链 kernel 的 UB 余量换大 BM |
| ir_3（MoE gemm2 独立前缀） | 1.27x | 0.96x | W2 重读减半（case12 0.66→0.78x） |
| **opt_iter_8（GM_EPI 大权重路径）** | **1.29x** | **1.00x** | 权重流量主导时回 GM 物化换 BM=128；全部 15 行 ≥ 初版 |

---

## §8 常见陷阱与避免方法

### §8.1 RotaryMul 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| Flat-1D 索引分解导致标量退化 | kernel 内用 `d = offsets % half_d; s = (offsets // half_d) % S` 等分解坐标 | 改用 per-position 处理 + `tl.arange` 向量化 D 维（L1.1/L3.1）；验证方法：检查编译后 IR 是否含大量 `scf.for` step-1 循环和 `memref.load %ptr[%c0]` 标量模式 |
| Adaptive TILE_S 导致编译器标量化 | 根据 S/D 动态选择 TILE_S（如 16/32/64），大 tile 被编译器拆成标量循环 | 固定 `TILE_S = 16`，用 grid splitting 解决负载均衡（L1.5/L3.3） |
| 忽略 broadcast stride | r1/r2 的 shape 为 `[B,1,S,D]` 时直接传 `t.stride()`，导致 kernel 内地址跳变错误 | Host 侧显式计算 broadcast stride（broadcast 维 stride=0，L1.2） |
| fp16/bf16 精度不足 | kernel 内直接以 fp16 做乘加减，relative error 超标 | kernel 内升 fp32 计算，存回前转回原精度（T2/L1.4） |
| Naive grid splitting 导致 idle core | `grid = (num_cores,)` + `for block in range(pid, num_blocks, num_cores)` 在 `num_blocks < num_cores` 时大量 core 空闲 | Uniform grid splitting（L2.2/L3.2）确保每个 core 处理连续且均匀的 block 范围 |

### §8.2 MoeComputeExpertTokens 陷阱

| 陷阱 | 表现 | 避免方法 |
|------|------|---------|
| atomic_add 竞争 | 某些 shape 延迟飙升到 0.04x | 改为 expert-parallel 无竞争计数（L1.1/L3.1） |
| torch.zeros 引入额外算子 | benchmark 中出现 `aclnnInplaceZero` | 使用 `torch.empty` + kernel 全覆盖写入（L3.3） |
| tl.cumsum fallback | 精度或性能异常 | 小数组直接用串行 for 循环（L1.2/L3.2） |
| 动态 grid 计算 | 增加 host 侧开销、编译器优化受限 | grid 固定为 `(num_expert,)` 和 `(1,)`（L1.4） |
| 跨 expert 循环计数 | 每个 block 做 64 次比较，指令膨胀 | grid 映射到 expert，每个 block 只比较一次（L1.3） |

### §8.3 MoeGatingTopKSoftmax 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| bool GM 缓冲区导致精度失败 | 使用 `torch.empty(..., dtype=torch.bool)` 作为 mask temp buffer，`tl.store`/`tl.load` 后 `tl.where` 条件类型变为 int8，导致大 f32 shape 精度错误 | 完全避免 bool GM 缓冲区，直接在寄存器中用 `tl.where` 更新 `curr` 向量（L1.2/L3.1） |
| num_cores 与 grid_size 不匹配 | `grid = (320,)` 但 `num_cores=40`，导致任务划分与实际 core 数不一致 | 始终令 `grid_size = min(batch_size, VEC_CORE_NUM)` 且 `num_cores = grid_size`（T1/L1.4） |
| tl.argmax 在 GM 往返后不可靠 | 将 `curr` store 到 GM temp buffer 再 load 回来，`tl.argmax` 返回错误索引 | 保持 `curr` 始终在寄存器中更新，不经过 GM（L1.3/L3.1） |
| host-side int32 cast 开销 | `finished.view(-1).to(torch.int32)` 引入额外 host 端计算 | 直接传入 `torch.bool`，kernel 内 `tl.load` 后直接使用（L3.3） |
| multibuffer/unit_flag 编译选项无效优化 | 盲目添加 `multibuffer=True, unit_flag=True` 期望提升内存密集型算子性能，本算子添加后性能从 0.8836x 下降至 0.8527x | 对 sort-topk 类算子（含迭代标量循环），multibuffer 收益被 top-k 迭代开销掩盖，不建议默认开启；应在 Phase 4 中实测验证后再决定（T6） |
| kernel 内使用 Python if/else 判断 None 指针 | `if finished_ptr is not None:` 在 Triton kernel 内编译失败或行为不可预期 | host 侧传入 dummy tensor（全 False），kernel 内直接 load 不使用条件判断（L1.6） |
| kernel 内使用 Python break | `break` 在 Triton kernel 中报 `unsupported AST node type: Break` | 使用 `tl.where` 条件赋值，或确保循环范围正确无需 break（L1.7） |
| 小 shape profiler 测量异常导致加速比失真 | framework 延迟 < 0.1ms 时，profiler 测量误差导致 speedup 异常高（2.5x~1251x） | benchmark 后过滤 framework 延迟 < 0.1ms 的 case，或设置 speedup 上限阈值（如 2.5x），重新计算几何平均加速比（L3.5） |
| 整行 load 向量长度超过 Ascend 稳定上限 | `tl.arange(0, NUM_EXPERTS)` 在 NUM_EXPERTS > 1536 时可能编译/运行正常但输出全 0（尤其非 pow2 如 1792） | 单 chunk 路径上限设为 1536；更大 E 使用带 mask 的 chunked load，CHUNK_E 按 2048/4096 自适应选择（L3.6） |
| LLVM_ROOT 环境变量未设置导致编译失败 | `clang++: symbol lookup error: undefined symbol: _ZN4llvm24createAutotuningDumpPassEv` | 设置 `LLVM_ROOT` 指向包含完整 libLLVM-17.so 的路径 |
| CANN 9.1.0 与 Triton 常量名不兼容 | `RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE` 在 CANN 9.1.0 中已重命名 | 修改 Triton 的 `npu_utils.cpp` 中的常量名为 `RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE`（一次性修复） |

### §8.4 AttentionSoftmaxWithSoftcappingAndDropout 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 编译器融合消除中间舍入 | 单 kernel `tanh(x/30)*30` 在 bf16 下 MERE 高达 22% | 拆为 div/tanh/mul 三个独立 kernel，GM round-trip 强制舍入（T5/L1.1） |
| tl.max axis=1 broadcast 错误 | `row_max = tl.max(t_fp32, axis=1); shifted = t_fp32 - row_max` 报 `Cannot make_shape_compatible` | 必须加 `[:, None]`：`row_max = tl.max(t_fp32, axis=1)[:, None]`（L3.3） |
| softcapping 子算子 dtype 不匹配 | `sc = 30.0`（Python float）或 `sc = tl.full((), 30.0, dtype=in_dtype)` 会导致 bf16/fp16 下除法精度不足 | `sc_fp32 = tl.full((), SOFTCAP, dtype=tl.float32)`，在 fp32 下做 div/tanh/mul，再 cast 回输入 dtype（T2/L1.2） |
| 循环引入后 UB overflow | 分核优化引入 `for block_id in range(pid, n_blocks, num_pids)` 循环后，大 K (BLOCK_N=1024) + ROW_TILE=16 触发 BiShengIR `ub over` 编译错误 | 收紧 UB budget 从 128KB 到 48KB，确保 `ROW_TILE * BLOCK_N * 4 <= 48KB`（L1.5） |
| grid 远超核数导致串行调度 | 朴素模式 `grid = ceil(num_rows/ROW_TILE)`，num_rows=32768 时 grid=2048，远超 48 核，NPU 串行执行 | `grid_size = min(natural_blocks, num_cores)`，每个 program 循环处理多块（T1/L3.1） |
| mask 元素污染归约 | padding 元素（load 时 other=0）参与 max/sum 导致结果错误 | max 前 `tl.where(mask, x, -inf)`，sum 前 `tl.where(mask, exp_val, 0.0)`（L1.4） |
### §8.5 LightningIndexer 陷阱

| 陷阱 | 现象 | 避免方法 |
|------|------|---------|
| 分数链非位级一致 | 索引 |diff|>1 大量失败（大 K 时边界翻边） | L1.1/L1.2：fp16 量化 + 块对角 fp32 头归约 + 同源稳定排序 |
| 近似选位（fp16 键/topk 平局） | verify 量化类失败 | L1.4：只用 `.sort()` 稳定排序 |
| R 行合并跨 batch | k 取错 batch、分数错乱 | L1.5：门控 `S1 % R == 0` |
| 拆分路径只在小 shape 验证 | verify 8/8 全走融合路径，拆分路径未被验证 | 拆分路径单独做大 shape 位级复验（实测 0/8.4M 差异） |
| 同名 kernel 混淆 profiling 归因 | qk/hsum 都显示 "kernel" | 归因用唯一 kernel 名，勿凭名字判断耗时 |
| torch 参考大 shape OOM | `expand` 是零拷贝视图，`.reshape()` 触发真实拷贝，把 `[B,S1,D,S2]` 整体复制成 68GB 内存 | 基线缺陷不可改（freeze 锚定），报告标注 |

### §8.6 FFN 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 单 kernel 融合两段 GEMM | GEMM2 需全量 H，跨块同步不可用 | 双 kernel 拆分（L1.1） |
| GLU 先落盘 g 再逐元素 kernel | 多两趟 GM 访存 + 额外 launch | 双累加器 epilogue 融合（L1.2） |
| GLU 双累加器超 L0C | 2×BM×BN×4B > 128KB spill | BM×BN ≤ 16384（L1.5） |
| 右半列偏移写成 bj*BN+KH 与 K2 混淆 | KH=N1//2 与 W2 的 K2 同值但语义不同源 | 统一用 KH 变量 |
| 激活字符串传进 kernel | device 侧无字符串分支 | constexpr 枚举分派（L1.4） |
| fastgelu 用错公式 | tanh 近似系数固定 0.7978845608028654/0.044715 | 对照公式表，与 gelu(erf) 严格区分 ACT_TYPE |
| epilogue 用 round-trip cast 保留舍入 | dot 累加器派生值的有损 cast 被编译器消除，舍入链丢失 | GM 物化 store-reload（T7/L1.8/L3.1） |
| epilogue 全宽 [BM,BN] 做 erf/tanh 链 | 激活链 + store-reload 工作区超 192KB UB | BN_EPI=64 子切片（L1.9） |
| fp16 常数直接用 fp32 字面量 | 内置算子 fp16 路径先把常数舍入到 f16 再运算 | host 侧 `torch.tensor(c, dtype=f16)` 预舍入（L3.2） |
| MoE 权重偏移用 int32 索引 | E×K×N 超 int32 上限地址回绕 | `e.to(tl.int64) * K * N` 基址（T8/L1.10） |
| MoE 每次调用同步 H2D 上传前缀 | 每次前向 ~150us 同步开销 | pinned 环形缓冲 + non_blocking（T8/L3.4） |
| gemm2 沿用 gemm1 的小 BM | gemm2 无激活链，UB 余量闲置；W2 按多 M-block 重读，MTE2 饱和 | 双 kernel tile 解耦，dense M%128==0 时 BM2=128，MoE 独立 BM2+blk2 前缀（L1.11） |
| 寄存器 epilogue 配 (128,128) tile | 激活链活跃量 4/5 单元 × 64KB 超 192KB UB（实测需 256KB） | 非 GLU (64,128) / GLU (128,64)，按单元 = BM×BN×4B 预算活跃量 |
| K 循环 mask 比较转 fp32 防标量降级 | 新增活跃缓冲打破 multibuffer 预算，实测 UB overflow 228KB | 不动热路径 mask，或与 tile 降档配套编译实测 |
| MoE 大权重的 gemm1 强用寄存器 epilogue | UB 限 BM≤64，W1 读多遍，MTE2 饱和 | K1≥4096 回 GM 物化 epilogue + BM=128（§7.4 GM_EPI 路径） |

