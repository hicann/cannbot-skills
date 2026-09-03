---
name: transformer-inference
description: Transformer 推理类算子（RotaryMul / MoeComputeExpertTokens / MoeGatingTopKSoftmax / AttentionSoftmaxWithSoftcappingAndDropout）的 Triton Ascend 优化经验合集，按算子分章节组织，含通用经验 + 各算子专属约束/骨架/kernel
metadata:
  type: reference
---

# Transformer 推理类算子优化经验

本文档合并了四类 Transformer 推理算子的优化经验。按以下结构组织：
- **§1 通用经验**：跨算子重复的工程约束（已提取，各算子章节不再重复；与张量变换类共用的通用约束见 tensor-transform.md G1-G8）
- **§2 RotaryMul**（rotarymul，RoPE 旋转位置编码）
- **§3 MoeComputeExpertTokens**（indexing-gather / counting，MoE 专家 token 计数 + 前缀和）
- **§4 MoeGatingTopKSoftmax**（sort-topk，门控 softmax + 迭代 top-k）
- **§5 AttentionSoftmaxWithSoftcappingAndDropout**（reduce，softcapping + 行级 softmax 融合）
- **§6 各算子常见陷阱**

---

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| RotaryMul | `rotarymul` | 4D `[B,H,S,D]` 张量 half/interleave 模式旋转乘法，fp16/bf16/fp32 | per-position 向量化 + uniform grid splitting + 2D tiling + broadcast stride |
| MoeComputeExpertTokens | `indexing-gather / counting` | expert 维度 token 计数（histogram）+ 小数组前缀和 | 两阶段分离：expert-parallel 无竞争计数 + 单 block 串行 prefix sum |
| MoeGatingTopKSoftmax | `sort-topk` | softmax over last dim + 迭代 top-k 选择，per-row 独立 | 纯寄存器 top-k 路径（无 GM temp buffer）+ grid 钳制到 num_cores |
| AttentionSoftmaxWithSoftcappingAndDropout | `reduce` | Gemma3 风格 softcapping `tanh(x/30)*30` + 行级 softmax，多 dtype 混合 | 4-kernel 分离强制中间舍入 + 分核优化（grid 钳制 + 循环处理多块） |

> ⚠️ **关键区分**：四类算子计算模式差异极大，优化哲学不可混用：
> - RotaryMul 关心 **per-position 向量化** 避免 flat-1D 标量退化
> - MoeComputeExpertTokens 关心 **无竞争 expert-parallel 计数** 避免 atomic contention
> - MoeGatingTopKSoftmax 关心 **寄存器内 top-k 路径** 避免 GM 往返导致 `tl.argmax` 不可靠
> - AttentionSoftmaxWithSoftcappingAndDropout 关心 **多 kernel 分离强制 dtype 舍入** 匹配 PyTorch 中间物化行为

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下 6 条约束是四类 Transformer 推理算子**共有**且**未在 tensor-transform.md G1-G8 覆盖**的工程约束。tensor-transform.md 中已提取的 G1（动态 num_cores）/ G2（pow2 BLOCK）/ G3（多策略分派）/ G4（grid 不超核数）/ G5（int32 索引）/ G6（负载均衡）/ G7（contiguous）/ G8（坐标 float32 比较）此处不再重复，各算子章节引用时标注。

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

## §6 常见陷阱与避免方法

### §6.1 RotaryMul 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| Flat-1D 索引分解导致标量退化 | kernel 内用 `d = offsets % half_d; s = (offsets // half_d) % S` 等分解坐标 | 改用 per-position 处理 + `tl.arange` 向量化 D 维（L1.1/L3.1）；验证方法：检查编译后 IR 是否含大量 `scf.for` step-1 循环和 `memref.load %ptr[%c0]` 标量模式 |
| Adaptive TILE_S 导致编译器标量化 | 根据 S/D 动态选择 TILE_S（如 16/32/64），大 tile 被编译器拆成标量循环 | 固定 `TILE_S = 16`，用 grid splitting 解决负载均衡（L1.5/L3.3） |
| 忽略 broadcast stride | r1/r2 的 shape 为 `[B,1,S,D]` 时直接传 `t.stride()`，导致 kernel 内地址跳变错误 | Host 侧显式计算 broadcast stride（broadcast 维 stride=0，L1.2） |
| fp16/bf16 精度不足 | kernel 内直接以 fp16 做乘加减，relative error 超标 | kernel 内升 fp32 计算，存回前转回原精度（T2/L1.4） |
| Naive grid splitting 导致 idle core | `grid = (num_cores,)` + `for block in range(pid, num_blocks, num_cores)` 在 `num_blocks < num_cores` 时大量 core 空闲 | Uniform grid splitting（L2.2/L3.2）确保每个 core 处理连续且均匀的 block 范围 |

### §6.2 MoeComputeExpertTokens 陷阱

| 陷阱 | 表现 | 避免方法 |
|------|------|---------|
| atomic_add 竞争 | 某些 shape 延迟飙升到 0.04x | 改为 expert-parallel 无竞争计数（L1.1/L3.1） |
| torch.zeros 引入额外算子 | benchmark 中出现 `aclnnInplaceZero` | 使用 `torch.empty` + kernel 全覆盖写入（L3.3） |
| tl.cumsum fallback | 精度或性能异常 | 小数组直接用串行 for 循环（L1.2/L3.2） |
| 动态 grid 计算 | 增加 host 侧开销、编译器优化受限 | grid 固定为 `(num_expert,)` 和 `(1,)`（L1.4） |
| 跨 expert 循环计数 | 每个 block 做 64 次比较，指令膨胀 | grid 映射到 expert，每个 block 只比较一次（L1.3） |

### §6.3 MoeGatingTopKSoftmax 陷阱

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

### §6.4 AttentionSoftmaxWithSoftcappingAndDropout 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 编译器融合消除中间舍入 | 单 kernel `tanh(x/30)*30` 在 bf16 下 MERE 高达 22% | 拆为 div/tanh/mul 三个独立 kernel，GM round-trip 强制舍入（T5/L1.1） |
| tl.max axis=1 broadcast 错误 | `row_max = tl.max(t_fp32, axis=1); shifted = t_fp32 - row_max` 报 `Cannot make_shape_compatible` | 必须加 `[:, None]`：`row_max = tl.max(t_fp32, axis=1)[:, None]`（L3.3） |
| softcapping 子算子 dtype 不匹配 | `sc = 30.0`（Python float）或 `sc = tl.full((), 30.0, dtype=in_dtype)` 会导致 bf16/fp16 下除法精度不足 | `sc_fp32 = tl.full((), SOFTCAP, dtype=tl.float32)`，在 fp32 下做 div/tanh/mul，再 cast 回输入 dtype（T2/L1.2） |
| 循环引入后 UB overflow | 分核优化引入 `for block_id in range(pid, n_blocks, num_pids)` 循环后，大 K (BLOCK_N=1024) + ROW_TILE=16 触发 BiShengIR `ub over` 编译错误 | 收紧 UB budget 从 128KB 到 48KB，确保 `ROW_TILE * BLOCK_N * 4 <= 48KB`（L1.5） |
| grid 远超核数导致串行调度 | 朴素模式 `grid = ceil(num_rows/ROW_TILE)`，num_rows=32768 时 grid=2048，远超 48 核，NPU 串行执行 | `grid_size = min(natural_blocks, num_cores)`，每个 program 循环处理多块（T1/L3.1） |
| mask 元素污染归约 | padding 元素（load 时 other=0）参与 max/sum 导致结果错误 | max 前 `tl.where(mask, x, -inf)`，sum 前 `tl.where(mask, exp_val, 0.0)`（L1.4） |

