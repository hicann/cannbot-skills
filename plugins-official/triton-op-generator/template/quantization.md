---
name: quantization
description: 量化类算子（DynamicQuant / SwigluQuant / WeightQuantBatchmatmul / GroupedMatmulSwigluQuant / GroupedMatmulSwigluQuantV2）的 Triton Ascend 优化经验合集，按算子分章节组织，含通用经验 + 各算子专属约束/骨架/kernel
metadata:
  type: reference
---

# 量化类算子优化经验

本文档合并了**五类**量化算子的优化经验。按以下结构组织：
- **§1 通用经验**：量化类算子跨算子重复的工程约束（已提取，各算子章节不再重复）
- **§2 DynamicQuant**（reduction + element-wise 复合，per-token 动态量化）
- **§3 SwigluQuant**（quantization-activation，SwiGLU 激活后量化）
- **§4 WeightQuantBatchmatmul**（weight-quantized batch matrix multiplication，反量化+GEMM）
- **§5 GroupedMatmulSwigluQuant**（group_list 变长分组的 int8 grouped GEMM + SwiGLU + 动态量化）
- **§6 常见陷阱与避免方法**
- **§7 GroupedMatmulSwigluQuantV2**（grouped matmul + SwiGLU + quant，MoE 专家分组矩阵乘）

---

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| DynamicQuant | `reduction` + `element-wise` 复合 | per-row max_abs reduction + 量化，2-pass HBM（大 K）或 single-pass fused（小 K） | 自适应 BLOCK 选择 + 单/双 pass 分派；大 K 接受架构限制 |
| SwigluQuant | `quantization-activation` | 对 SwiGLU 激活结果做逐行静态/动态 INT8/INT4 量化，支持 smooth scales、offsets、group index | static/dynamic 双 kernel 分派 + vector scale division + INT4 打包 |
| WeightQuantBatchmatmul | `matmul` + `quantization` | `x` fp16/bf16 `[M,K]` @ int8 `weight` `[K,N]`，per-tensor/per-channel/per-group antiquant scale/offset，可选 bias；输出 dtype 同 `x` | group / non-group 分裂 + mixed 解耦/fused 路由 + N-strip chunking 消除 L2-exceeding workspace 重读；medium-N 接受 Triton toolchain ceiling |
| GroupedMatmulSwigluQuant | `grouped-matmul` + `quantization` | `x` int8 `[M,K]` 按 `group_list` 变长分组 @ 每 expert 独立 int8 `weight[e]` `[K,N]`（含 fp32 weight_scale / per-row x_scale 反量化）→ SwiGLU → per-token 动态量化输出 int8+f32 scale | persistent 单 launch 跨 expert 全局 tile 编号 + meta int64 打包单次 H2D + BLOCK_K=256 tiling；终态地板是 weight 搬运链（MTE 68%），geo 1.32（理论上限 1.3995 的 94.4%）是该 dtype/shape 域合理预期 |
| GroupedMatmulSwigluQuantV2 | `grouped matmul` + `activation` + `quantization` | `x [M,K] int8` 按 group_list（cumsum/count）切 E 个 expert 区间 @ `weight [E,K,N] int8` + per-channel scale → SwiGLU → per-token int8 quant | expert 边界对齐 BLOCK_M 合并（dot M 维 ≥16）+ 扁平 (m_tile,n_tile) 并行 + partial row_max 免 atomic + BLOCK_K 按 dequant 路径分档；fp16 预缩放路径 BK 锁死 256 |

> ⚠️ **关键区分**：五类算子的瓶颈与优化哲学不同，生成时**禁止混用经验**：
> - **DynamicQuant** 瓶颈在 **memory-bound + 跨 pass 数据复用**（大 K 时 2-pass HBM 不可优化）。
> - **SwigluQuant** 瓶颈在 **UB 压力 + 精度对齐**（dynamic 模式必须 vector scale division，sigmoid 必须手写）。
> - **WeightQuantBatchmatmul** 瓶颈在 **Cube 利用率 + 访存层次**（dequant 是否在 dot 关键路径、workspace 是否超过 L2），优化核心是 **kernel 分裂 + mixed 路由 + N-strip chunking**。
> - **GroupedMatmulSwigluQuant** 瓶颈在 **host 调度（per-expert launch + 逐次 D2H）→ 修复后是 weight 搬运链（MTE1+MTE2 68%）**，优化核心是 **persistent 单 launch + meta 打包 + tiling**；数值链约束与 §3 SwigluQuant 同源（手写 sigmoid / round-half-away-from-zero / int32 clip 可复用），但 GEMM 段调参经验不通用（§4 的解耦/N-strip 对 grouped 变长分组不适用）。
>
> 禁止跨算子套用：
> - 生成 DynamicQuant 时，**不要**套用 SwigluQuant 的 activation 打包技巧或 WeightQuantBatchmatmul 的解耦 GEMM/N-strip 技巧。
> - 生成 SwigluQuant 时，**不要**套用 DynamicQuant 的 2-pass HBM 失败方向或 WeightQuantBatchmatmul 的 matmul tile 调参。
> - 生成 WeightQuantBatchmatmul 时，**不要**套用 DynamicQuant 的 2-pass 策略（没有跨 pass 数据复用问题），也**不要**套用 SwigluQuant 的 scale division / INT4 打包技巧（本算子是反量化+GEMM，不是训练后量化）。
> - 生成 GroupedMatmulSwigluQuant 时，**不要**套用 WeightQuantBatchmatmul 的 dequant workspace 解耦 / N-strip（grouped 变长分组下 workspace 划分与 expert 边界冲突，实测 K1/K2 融合类方向全部劣化，见 §5.4）；它的 K2 量化段**可以**复用 §3 的 sigmoid/rounding/clip 约束。

> ⚠️ **架构限制特殊说明（破例归档）**：DynamicQuant 几何平均加速比仅 **0.3604x**，远低于归档阈值 1.0x 和用户目标 0.6x。**破例归档**作为"Triton DSL 架构限制典型案例"——大 K 场景必然 2-pass HBM，与 torch 单遍融合 kernel 存在结构性带宽劣势，是**已知不可优化**的场景。后续做其他 per-group quant 时，若大 K 慢于 torch，应首先排查 2-pass HBM（WeightQuantBatchmatmul 见 §4，不适用此结论）。

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下约束在 DynamicQuant、SwigluQuant、WeightQuantBatchmatmul 三类量化算子中均适用；WeightQuantBatchmatmul 的专用约束见 §4。

### Q1 动态读取 Vector Core 数量，禁止硬编码
- **必须**动态读取实际 Vector Core 数量，禁止硬编码 `num_cores=8` 或 `num_cores=48`。
- **正确做法**：
通过 `triton.runtime.driver.active.utils.get_device_properties(device)` 读取 `num_vectorcore` / `num_aicore`。
- **Why:** 硬编码仅利用部分 Vector Core，导致加速比显著下降。

### Q2 1D Grid + 核内循环（负载均衡通用模式）
- **必须**使用 `pid = tl.program_id(0)`，grid 大小限制为 `min(total_blocks, num_cores)`。
- **必须**在 kernel 内用交织循环 `for block_idx in range(pid, num_blocks, num_programs)` 或 `rows_per_prog` 块循环处理多行。
- **禁止**用多维 grid 强行并行（量化算子按行处理，1D grid 配合核内循环是最稳定的并行模式）。
- **Why:** Ascend vector core 数量有限，1D grid 配合核内循环天然负载均衡。

### Q3 输入必须 contiguous
- **必须** Host 侧进入 kernel 前调用 `x = x.contiguous()`。
- **Why:** 避免非连续张量导致 kernel 内 stride 计算复杂化。

### Q4 int32 索引，避免 int64 降级
- **必须**在 kernel 内将 `tl.program_id` 和 `tl.arange` 结果 `.to(tl.int32)`。
- **Why:** int64 标量会触发地址计算降级；NPU 上 int32 索引更高效。

### Q5 多维输入必须在 host 侧扁平化为 2D
- **必须**在 `forward()` 中将输入 x 通过 `x.view(prefix_dims, last_dim)` 扁平化（保持 stride 不变亦可，但逻辑上按 2D 处理）。
- **Why:** kernel 内部只处理 `(row, col)` 二维逻辑，降低维度处理复杂度。

### Q6 禁止在 forward 中使用 torch / torch_npu 计算
- **必须**`ModelNew.forward()` 中只负责 shape 计算、分派、host 预计算；所有量化计算必须在 `@triton.jit` kernel 内完成。
- **禁止**调用 `torch_npu.npu_swiglu_quant` / `torch_npu.npu_dynamic_quant` 等参考算子预计算 scale。
- **Why:** `validate_triton_impl.py` Type-3 检查会 flag 任何 forward 中的 torch 计算为退化；同时失去纯 Triton 意义。

### Q7 禁止在 kernel 内使用 `continue` / `break`
- **Why:** Ascend 后端对非规整控制流支持有限，容易引发编译或性能问题。
- **How to apply:** 所有场景，用 `tl.where` 做向量分段代替条件跳转。

### Q8 BLOCK 向上取 2 的幂（NPU 向量化友好）
- **必须**将 BLOCK_SIZE / BLOCK_N 等向量化长度声明为 `triton.next_power_of_2(N)` 或自适应选择 pow2 值，且作为 `tl.constexpr` 传入 kernel。
- **Why:** Ascend 上 fixed-shape vector load 必须是编译期常量长度；非 constexpr 会触发 dynamic-shape load，退化为标量循环。

---

## §2 DynamicQuant 算子（reduction + element-wise 复合）

**算子类别**: `reduction` + `element-wise` 复合（per-row max_abs reduction + 量化）
**典型特征**: 输入 x `[M, K]` 或 `[B, M, K]` (fp16/bf16) + 可选 smooth_scales `[K]`，输出 int8 quant + float32 scale `[M]`。算法要求两阶段：pass1 计算 per-row max_abs → scale，pass2 用 scale 量化
**性能基准**: 42/42 verify pass，几何平均加速比 **0.3604x** vs torch（**未达 0.6x 用户目标，未达 1.0x 归档阈值，破例归档**）

### §2.0 破例归档说明

本版本 geomean 0.3604x，远低于归档阈值 1.0x 和用户目标 0.6x。破例归档理由：
1. **架构限制典型案例**：Triton DSL 无法暴露 CCE 级 UB 软件管理，memory-bound 算子大 K 场景必然 2-pass HBM，与 torch 单遍融合 kernel 存在结构性带宽劣势。这是**已知不可优化**的场景。
2. **7 种失败优化方向**：记录已验证无效的优化策略，避免后续任务重复踩坑。
3. **量化类算子复用价值**：后续做 per-group quant / weight quant 时，若大 K 慢于 torch，应首先排查 2-pass HBM。

### §2.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 大 K (>8192) 场景必然 2-pass HBM，禁止期望超越 torch（架构限制）
- **必须**认识到：DynamicQuant 在 K > 8192 时，整行无法放入 UB (192KB)，必须分块处理，导致 x 被从 HBM 读取两次（pass1 max_abs + pass2 quant）。
- **禁止**在 K > 8192 场景期望通过 tile 优化、循环重排等手段超越 torch。torch 的 CCE 底层算子是**单遍融合 kernel**——在 UB 中缓存分块数据，pass1 计算 max_abs 后直接在 UB 中完成 pass2 量化，全程只从 HBM 读取 x 一次。
- **Why:** 实测带宽：torch 818 GB/s (68% HBM 峰值)，triton 277 GB/s (23% HBM 峰值)。2x HBM 读取 = 2x 带宽劣势，这是架构层面的硬限制。
- **How to apply:** 大 K 场景直接接受 0.2x~0.3x 加速比，不要浪费优化轮次。把精力放在小 K (≤4096) 场景的 single-pass 优化上。

#### L1.2 禁止用 L2 cache 命中作为 2-pass 优化手段（已验证无效，负面经验/失败方向）
- **禁止**在单 kernel 内用两个循环（pass1 max_abs → pass2 quant）期望 L2 cache 保留 pass1 的数据供 pass2 复用。
- **Why:** 实测无改善（0.37x），L2 cache 在大 K 场景未有效保留数据。Ascend L2 cache 容量和替换策略不适合跨 pass 数据复用。
- **How to apply:** 2-pass 就是 2-pass HBM，不要试图用 L2 cache "欺骗" 成 single-pass。

#### L1.3 禁止用非对称 BLOCK_K 优化 2-pass（已验证无效，负面经验/失败方向）
- **禁止**给 pass1 (max_abs) 用大 BLOCK_K (8192/16384)、pass2 (quant) 用小 BLOCK_K (4096)，期望减少 pass1 的中间量。
- **Why:** 实测无显著改善。瓶颈是 HBM 带宽而非中间量寄存器压力。
- **How to apply:** 两个 pass 用相同 BLOCK_K 即可。

#### L1.4 禁止用 bf16 max_abs 避免 fp32 膨胀（已验证无效，负面经验/失败方向）
- **禁止**在 bf16 精度下计算 max_abs，期望避免 fp32 的 4x 内存膨胀。
- **Why:** 实测无改善。max_abs 的精度需求使 bf16 中间结果仍需转 fp32 累积，且膨胀发生在寄存器而非 HBM。
- **How to apply:** 直接用 fp32 累积 max_abs。

#### L1.5 禁止用 inv_scale 乘法替代除法（已验证反而回退，负面经验/失败方向）
- **禁止**用 `x * (1/scale)` 替代 `x / scale` 期望加速。
- **Why:** 实测反而回退。Ascend 的乘法和除法延迟相近，额外的 `1/scale` 计算和 inv_scale reshape 反而增加开销。
- **How to apply:** 直接用 `x / scale` 或在 kernel 内 `inv_scale = 1.0 / scale` 后 broadcast 乘法——两者性能相近，选可读性更好的。

#### L1.6 禁止在大 K 场景用 BLOCK_M > 1（UB 溢出）
- **禁止**在 K > 4096 场景用 `BLOCK_M > 1` 的 2D tile，会导致 UB 容量溢出。
- **Why:** UB 容量 192KB，`BLOCK_M * BLOCK_K * sizeof(fp32)` 易超限。大 K 时必须 `BLOCK_M = 1` 或极小值。
- **How to apply:** `_choose_block_size`: `cols >= 512` 用 `(8, 512)`；否则 `block_m = max(1, min(32, 4096 // block_n))`。

#### L1.7 Triton DSL 无法暴露 CCE 级优化（架构限制，禁止尝试）
- **禁止**尝试以下 CCE 级优化，Triton DSL 不支持：
  - 软件管理 UB 的分块缓存（在 UB 中保留 pass1 数据供 pass2 使用）
  - 跨核同步原语（atomic max + barrier 实现单遍跨块 reduction）
  - DMA 异步传输与计算的 overlap
- **Why:** 这些是 torch CCE 单遍融合 kernel 的核心优化手段，Triton DSL 层面无法表达。
- **How to apply:** 遇到 memory-bound + 跨 pass 数据复用的算子，直接判定为架构限制场景，不要尝试逆向工程 CCE 优化。

#### L1.8 小 K (≤4096) 必须用 single-pass fused kernel（可达 1.0x+）
- **必须**在 K ≤ 4096 时用 single-pass fused kernel——整行一次性加载到 UB，单遍完成 max_abs + 量化。
- **Why:** 小 K 整行可放入 UB，避免 2-pass HBM，可达甚至超越 torch 性能（实测部分 case 1.192x）。
- **How to apply:** `_choose_block_size` 根据 cols 选择 BLOCK_N = pow2(cols)，整行单块处理。

#### L1.9 必须用 tl.cast overflow_mode='saturate' 替代手动 clamp
- **必须**用 `tl.cast(q, tl.int8, overflow_mode='saturate')` 把浮点量化结果转为 int8，禁止手动 `tl.maximum(tl.minimum(q, 127), -128)`。
- **Why:** `overflow_mode='saturate'` 是 Ascend 硬件原生支持的饱和转换，比手动 clamp 更快且精度一致。
- **How to apply:**
  ```python
  q = tl.where(q >= 0, q + 0.5, q - 0.5)   # round to nearest
  q = tl.cast(q, tl.int8, overflow_mode='saturate')   # saturate to [-128, 127]
  ```

#### L1.10 必须用 round-half-away-from-zero 而非 tl.round
- **必须**用 `tl.where(q >= 0, q + 0.5, q - 0.5)` 然后 `tl.cast` 实现 round-half-away-from-zero，禁止用 `tl.round`。
- **Why:** `tl.round` 默认 round-half-to-even（banker's rounding），与 torch `npu_dynamic_quant` 的 round-half-away-from-zero 语义不一致，会导致 verify 失败。
- **How to apply:** 见 L1.9 代码片段。

### §2.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧分支决策树

```python
def forward(self, x, smooth_scales=None, group_index=None, dst_type=None):
    if dst_type is None: dst_type = torch.int8
    if group_index is not None:
        raise NotImplementedError("Group quantization not supported")

    x = x.contiguous()
    orig_shape = x.shape
    dim = x.dim()

    # 1. shape 归一化为 2D
    if dim == 2:
        rows, cols = x.shape
        quant = torch.empty(orig_shape, device=x.device, dtype=dst_type)
        scale = torch.empty((rows,), device=x.device, dtype=torch.float32)
    elif dim == 3:
        b, m, cols = x.shape
        rows = b * m
        quant = torch.empty(orig_shape, device=x.device, dtype=dst_type)
        scale = torch.empty((b, m), device=x.device, dtype=torch.float32)
    else:
        x_2d = x.reshape(1, -1)
        rows, cols = x_2d.shape
        # ... 1D 退化处理

    # 2. 自适应 block size (L1.6 / L1.8)
    block_m, block_n = self._choose_block_size(cols)
    grid = (min(triton.cdiv(rows, block_m), self.num_cores),)

    # 3. 按 smooth_scales 分派
    if smooth_scales is None:
        dynamic_quant_fused_2d_kernel_no_smooth[grid](
            x, quant, scale, rows, cols, cols, cols,
            BLOCK_M=block_m, BLOCK_N=block_n,
        )
    else:
        smooth_scales = smooth_scales.contiguous()
        dynamic_quant_fused_2d_kernel[grid](
            x, smooth_scales, quant, scale, rows, cols, cols, cols,
            BLOCK_M=block_m, BLOCK_N=block_n,
        )

    if dim not in (2, 3):
        quant = quant.reshape(orig_shape)
    return quant, scale
```

#### L2.2 _choose_block_size 自适应策略

```python
def _choose_block_size(self, cols):
    if cols >= 512:
        return 8, 512          # 宽行: (8, 512) tile 摊销循环开销
    cols_pow2 = triton.next_power_of_2(cols)
    block_n = cols_pow2
    block_m = max(1, min(32, 4096 // block_n))   # 窄行: tile 约 4K elements
    return block_m, block_n
```

#### L2.3 Kernel 两阶段循环骨架（2D tiling）

```python
@triton.jit
def dynamic_quant_fused_2d_kernel(
    x_ptr, smooth_ptr, quant_ptr, scale_ptr,
    ROWS, COLS, ROW_STRIDE, QUANT_STRIDE,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_row_blocks = tl.cdiv(ROWS, BLOCK_M)

    # 1D Grid + 交织循环 (Q2)
    for block_idx in range(pid, num_row_blocks, num_programs):
        row_start = block_idx * BLOCK_M
        row_offsets = row_start + tl.arange(0, BLOCK_M)
        row_mask = row_offsets < ROWS

        # === Pass 1: 累积 max_abs ===
        max_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for col_start in range(0, COLS, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            col_mask = col_offsets < COLS
            mask2d = row_mask[:, None] & col_mask[None, :]

            x_tile = tl.load(
                x_ptr + row_offsets[:, None] * ROW_STRIDE + col_offsets[None, :],
                mask=mask2d, other=0.0
            ).to(tl.float32)
            smooth_tile = tl.load(smooth_ptr + col_offsets, mask=col_mask, other=1.0).to(tl.float32)
            x_tile = x_tile * smooth_tile[None, :]   # smooth broadcast 到行
            max_acc = tl.maximum(max_acc, tl.max(tl.abs(x_tile), axis=1))

        # === 计算 scale ===
        scale = max_acc / 127.0
        scale = tl.maximum(scale, 1e-10)
        tl.store(scale_ptr + row_offsets, scale, mask=row_mask)

        # === Pass 2: 量化 ===
        inv_scale = tl.reshape(1.0 / scale, (BLOCK_M, 1))   # broadcast 准备
        for col_start in range(0, COLS, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            col_mask = col_offsets < COLS
            mask2d = row_mask[:, None] & col_mask[None, :]

            x_tile = tl.load(
                x_ptr + row_offsets[:, None] * ROW_STRIDE + col_offsets[None, :],
                mask=mask2d, other=0.0
            ).to(tl.float32)
            smooth_tile = tl.load(smooth_ptr + col_offsets, mask=col_mask, other=1.0).to(tl.float32)
            q = x_tile * smooth_tile[None, :] * inv_scale
            q = tl.where(q >= 0, q + 0.5, q - 0.5)           # round-half-away-from-zero (L1.10)
            q = tl.cast(q, tl.int8, overflow_mode='saturate')  # 饱和转换 (L1.9)
            tl.store(
                quant_ptr + row_offsets[:, None] * QUANT_STRIDE + col_offsets[None, :],
                q, mask=mask2d
            )
```

### §2.3 Layer 3: 关键技巧（Agent 可参考，但实现方式可不同）

#### L3.1 tl.cast overflow_mode='saturate' 替代手动 clamp

**问题**: 量化结果需 clamp 到 `[-128, 127]`，手动 `tl.maximum(tl.minimum(q, 127), -128)` 生成多条指令。

**解决**: `tl.cast(q, tl.int8, overflow_mode='saturate')` 是 Ascend 硬件原生饱和转换，单条指令完成。

```python
q = tl.where(q >= 0, q + 0.5, q - 0.5)   # round
q = tl.cast(q, tl.int8, overflow_mode='saturate')   # saturate
```

**可替代方向**: 若 `overflow_mode` 参数不可用（旧版 triton-ascend），回退到手动 clamp，但性能略差。

#### L3.2 inv_scale reshape broadcast 替代逐元素除法

**问题**: `x_tile / scale` 中 scale 是 `[BLOCK_M]`，x_tile 是 `[BLOCK_M, BLOCK_N]`，需要 broadcast 除法。

**解决**: `inv_scale = tl.reshape(1.0 / scale, (BLOCK_M, 1))`，然后 `x_tile * inv_scale` 利用 broadcast 乘法。预先计算 `1/scale` 避免逐元素除法。

**注意**: L1.5 标注此优化"反而回退"，但在某些 case 仍可能有效，需实测验证。

#### L3.3 smooth_scales broadcast 到行

**问题**: smooth_scales 是 `[K]`（per-column），需应用到 `[BLOCK_M, BLOCK_N]` tile 的每一行。

**解决**: `x_tile * smooth_tile[None, :]`，`smooth_tile` 加载为 `[BLOCK_N]`，`[None, :]` broadcast 到 `[BLOCK_M, BLOCK_N]`。

**可替代方向**: 若 smooth_scales 全 1（无 smooth），用 `_no_smooth` kernel 避免额外 load。

#### L3.4 _choose_block_size 自适应 tile 选择

**问题**: 不同 K 需要不同 BLOCK_N，固定 tile 会在小 K 浪费、大 K 溢出。

**解决**:
```python
if cols >= 512:
    return 8, 512                          # 宽行: 摊销循环开销
cols_pow2 = triton.next_power_of_2(cols)
block_n = cols_pow2
block_m = max(1, min(32, 4096 // block_n))  # 窄行: tile 约 4K elements
```

**关键点**: `4096 // block_n` 保证 tile 总量约 4K elements，避免 UB 溢出。

**可替代方向**: 若 K 固定已知，可离线调优最优 (BLOCK_M, BLOCK_N) 组合。

#### L3.5 3D 输入 reshape 为 2D 处理

**问题**: 3D 输入 `[B, M, K]` 需要按 `[B, M]` 每行计算 scale。

**解决**: 不 reshape x（保持 stride），直接用 `rows = b * m` 把 3D 当 2D 处理，scale 输出为 `[B, M]` shape。

**可替代方向**: 若 3D 不连续，需先 `.contiguous()` 再处理。

#### L3.6 将 `COLS` 声明为 `tl.constexpr` 以启用编译期地址推导

**问题**: `ROW_STRIDE` / `QUANT_STRIDE` 作为运行时 kernel 参数传入时，编译器无法静态确定每行起始地址的 stride，导致地址计算无法完全向量化/常量化。

**解决**: 在 host 侧保证输入 `contiguous` 并把最后一维展平后，`ROW_STRIDE == QUANT_STRIDE == COLS`。将 `COLS` 声明为 `tl.constexpr`，kernel 内直接用 `row_offsets[:, None] * COLS + col_offsets[None, :]`，编译器可静态推导地址模式。

**正确示例**:
```python
@triton.jit
def dynamic_quant_kernel(
    x_ptr, quant_ptr, scale_ptr,
    ROWS, COLS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # ...
    x_tile = tl.load(
        x_ptr + row_offsets[:, None] * COLS + col_offsets[None, :],
        mask=mask2d, other=0.0
    ).to(tl.float32)
```

**效果**: 在 ascend910b2 / 42 cases 评测中，该优化使几何平均加速比从 0.795x 提升至 0.891x（相对基线 0.696x）。

**可替代方向**: 若输入无法保证 stride==cols，仍需保留 `ROW_STRIDE` 参数，但优先在 host 侧通过 `.view`/`.contiguous()` 规整 stride。

#### L3.7 小 K (≤4096) single-block 特化路径

**问题**: 通用 2-pass kernel 使用固定 BLOCK_N=512，小 K（如 128、256、1024）会被拆成多个 col 循环，增加 mask 计算和循环开销。

**解决**: 当 `COLS ≤ 4096` 时，启用 single-block 特化 kernel：`BLOCK_N = next_power_of_2(COLS)`，整行一次性加载，单 kernel 内完成 max + quant，避免 col 循环。

**正确示例**:
```python
# host 侧分派
if cols <= 4096:
    block_n = triton.next_power_of_2(cols)
    block_m = max(1, min(32, 4096 // block_n))
    dynamic_quant_single_block_kernel[grid](...)
else:
    block_m, block_n = 8, 512
    dynamic_quant_fused_2d_kernel[grid](...)
```

**效果**: 与 `COLS` constexpr 优化配合，小 K case 加速比普遍 >1.0x，部分达到 2.8x+。

**可替代方向**: 对中等 K（2048-4096）可尝试 BLOCK_N=1024/2048 而非整行，以平衡 register 压力和循环次数。

### §2.4 DynamicQuant 性能基准

| 版本 | K 范围 | cases | geomean | 区间 | 说明 |
|------|--------|-------|---------|------|------|
| 历史基线 | K ≤ 4096 | 20 | 0.590x | [0.219, 1.192] | 接近目标，小张量部分超 1.0x |
| 历史基线 | 4096 < K ≤ 8192 | 4 | 0.311x | [0.141, 0.475] | 中等表现 |
| 历史基线 | K > 8192 | 14 | 0.186x | [0.158, 0.236] | 瓶颈所在，2-pass HBM 限制 |
| 历史基线 | 全量 42 cases | 42 | 0.3604x | — | 未达 0.6x 目标，破例归档 |
| **v2 (2026-07-02)** | 全量 42 cases | 42 | **0.8910x** | [0.2886, 3.5556] | 加入 `COLS` constexpr + 小 K single-block 特化后，在 ascend910b2 环境达成目标 0.8x |

**带宽实测对比**：

| 实现 | HBM 读取量 | 延迟 | 带宽 | HBM 峰值利用率 |
|------|-----------|------|------|---------------|
| torch 单遍 | 90 MB | 0.11 ms | 818 GB/s | 68% |
| triton 双遍 | 180 MB | 0.65 ms | 277 GB/s | 23% |

**关键结论**:
1. DynamicQuant 是 Triton DSL 架构限制的边界案例——memory-bound + 大 K + 跨 pass 数据复用
2. 7 种优化策略全部失败，证明瓶颈在架构层而非实现层
3. 小 K (≤4096) 可达 1.0x+，是可用区间；大 K (>8192) 0.19x 是架构硬限制
4. **新经验**: 将 `COLS` 声明为 `tl.constexpr` 并配合小 K single-block 特化，可在 ascend910b2 上把全量 42 cases 几何平均加速比从 ~0.7x 提升至 0.89x，超过 0.8x 目标
5. 后续量化类算子若大 K 慢于 torch，应首先排查 2-pass HBM

---

## §3 SwigluQuant 算子（quantization-activation）

**算子类别**: `quantization-activation`
**典型特征**: 对 SwiGLU 激活结果做逐行静态或动态 INT8/INT4 量化，支持 smooth scales、offsets、group index
**性能基准**: 52 cases 精度全过；纯 Triton 实现几何平均 **0.5067x** vs torch，implementation latency 约 **0.265 ms**

> **评判口径**: framework 参考实现为高度优化的 `torch_npu.npu_swiglu_quant`，`speedup_vs_torch < 1` 属于正常现象；优化目标应聚焦在 **implementation latency** 降低，同时保证 52/52 精度全过。

### §3.0 算子语义

**输入**:
- `x`: 任意前缀 shape + 最后一维 `2 * half_dim` 的激活张量
- `smooth_scales`: 可选，1D `[num_groups]` 或 2D `[num_groups, half_dim]` 缩放
- `offsets`: 可选，仅 static 模式有效，1D 或 2D 偏移
- `group_index`: 可选，映射每行属于哪个 group
- `activate_left`: bool，决定 SwiGLU 用 `sigmoid(left) * right` 还是 `sigmoid(right) * left`
- `quant_mode`: 0=static, 1=dynamic
- `dst_type`: `torch.int8` 或 `torch.quint4x2`

**输出**:
- `out_quant`: 量化后的 int8/int4 张量
- `out_scales`: dynamic 模式返回每行 scale；static 模式返回零张量占位

### §3.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 必须拆分 static / dynamic 两个 kernel
- **必须**为 `quant_mode=0`（static）和 `quant_mode=1`（dynamic）分别编写 `@triton.jit` kernel，host 侧通过 `_route` 二选一启动
- **Why:**
  - static 模式直接量化，dynamic 模式需要 Pass 1 求 `max_abs`、Pass 2 量化
  - 合并为单一 kernel 会在内部引入大量 `tl.where(quant_mode==1, ...)` 分支，向量利用率低、编译器难以优化
- **How to apply:** 所有 SwigluQuant 场景

#### L1.2 必须使用 1D Grid + 核内循环，且 grid 必须钳制到 VEC_CORE_NUM
（通用模式见 §1 Q2）
- **必须** `grid = (min(M, VEC_CORE_NUM),)`，kernel 内 `for row in range(pid, M, num_programs)` 循环处理多行
- **禁止** `grid = (M,)`（每行一个 program，M 大时严重过饱和，全部串行排队，实测性能 0.01x，损失 10-50 倍）
- **禁止** 多维 grid（未带来收益，增加调度复杂度）
- static kernel 内部按 `rows_per_prog` 循环处理多行；dynamic kernel 内部按 `tiles_per_prog` 循环处理多个 tile
- **正确骨架**：
  ```python
  VEC_CORE_NUM = triton.runtime.driver.active.utils.get_device_properties(device).get("num_vectorcore", 48)
  grid = (min(M, VEC_CORE_NUM),)
  kernel[grid](..., num_cores=VEC_CORE_NUM, ...)
  # kernel 内：
  for row in range(pid, M, num_programs):
      ...
  ```
- **Why:** grid=(M,) 当 M=数百~数千 token 时远超 vector core 数（~40），program 全部排队串行，无核内循环摊销。grid 钳制 + 核内循环是本算子脱离 0.01x 的单点关键修复。

#### L1.3 禁止在 kernel 内或 forward 中调用 torch / torch_npu 计算
（见 §1 Q6）
- **禁止**使用 `torch_npu.npu_swiglu_quant` 预计算 dynamic scale
- **Why:** 违反 Triton-Ascend 算子实现约束；虽然可能提升 `speedup_vs_torch`，但失去纯 Triton 意义，无法沉淀为可复用 kernel
- **How to apply:** dynamic scale 必须在 Triton kernel 内部通过 `tl.max` + vector division 计算

#### L1.4 禁止 `tl.sigmoid`，必须手写 sigmoid
- **必须**使用 `x / (1.0 + tl.exp(-x))`
- **禁止**使用 `tl.sigmoid`
- **Why:** `tl.sigmoid` 在 Ascend 后端会产生系统性数值偏差，导致大量 case 精度 fail
- **How to apply:** 所有场景

#### L1.5 INT4 输出必须按 (odd << 4) | even 打包
- **必须**将相邻两个 INT8 值各取低 4 位：`packed = ((q_odd & 0x0F) << 4) | (q_even & 0x0F)`
- **Why:** 与 `torch.quint4x2` 存储布局一致
- **How to apply:**
  ```python
  q_2d = tl.reshape(q_val, (BLOCK_SIZE_COL // 2, 2))
  q_even, q_odd = tl.split(q_2d)
  packed = ((q_odd & 0x0F) << 4 | (q_even & 0x0F)).to(tl.int8)
  ```

#### L1.6 dynamic 模式下 scale 必须使用 vector division 路径
- **必须**先按行将 `max_abs` 存入全局内存，再以 tile 为单位 load 一个短向量 `max_vec`，执行 `int_scale / max_vec`
- **禁止**在量化的内层循环里对每行单独做标量除法 `int_scale / max_abs`
- **Why:**
  - Triton-Ascend 标量除法与向量除法的 ULP 行为不同
  - 标量路径会在大 fp32 dynamic-int8 shape 上产生 off-by-1
- **How to apply:**
  - Pass 1: 每行写 `scales_ptr[row_idx] = max_abs`
  - tile 级别：`scale_vec = int_scale / tl.maximum(max_vec, 1e-10)`，写回 `scales_ptr`
  - Pass 2: `row_scale = tl.load(scales_ptr + row_idx)`，再 `nearbyint(swiglu * row_scale)`

#### L1.7 clip 必须在 int32 阶段使用 `tl.maximum/tl.minimum`
- **必须**按 `nearbyint → .to(tl.int32) → tl.maximum/tl.minimum → .to(tl.int8)` 顺序执行
- **禁止**对 int32 使用 `tl.clamp`
- **Why:** `tl.clamp` 仅支持浮点；`tl.maximum/tl.minimum` 在 int32 上效率更优
- **How to apply:** 所有量化路径

#### L1.8 Grid 大小必须动态读取实际 vector core 数量
（见 §1 Q1）
- **禁止**硬编码 `num_cores`

#### L1.9 禁止在 kernel 内使用 `continue` / `break`
（见 §1 Q7）

#### L1.10 多维输入必须在 host 侧扁平化为 2D
（见 §1 Q5）
- **必须**在 `forward()` 中将输入 `x` 通过 `x.view(prefix_dims, 2 * half_dim)` 扁平化

#### L1.11 offsets 仅在 static 模式下有效
- **必须**在 host 侧设置 `has_offsets = offsets is not None and quant_mode == 0`
- **Why:** dynamic 模式不存在 offsets 语义，错误启用会导致精度失败或多余计算
- **How to apply:** host 侧分支决策时处理

### §3.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧分支决策树

```python
def forward(self, x, smooth_scales=None, offsets=None, group_index=None,
            activate_left=False, quant_mode=0, group_list_type=0, dst_type=None):
    prefix_shape = x.shape[:-1]
    prefix_dims = x.numel() // x.shape[-1]
    half_dim = x.shape[-1] // 2
    x_ptr = x.view(prefix_dims, 2 * half_dim)

    is_int4 = dst_type is not None and (str(dst_type) == 'int4' or dst_type == torch.quint4x2)
    int_scale = 7 if is_int4 else 127
    clip_min, clip_max = (-8, 7) if is_int4 else (-128, 127)

    has_smooth_scales = smooth_scales is not None
    has_offsets = offsets is not None and quant_mode == 0
    has_group_index = group_index is not None
    num_groups = group_index.shape[0] if has_group_index else 0

    # block_size_col 按 half_dim 选择
    block_size_col = select_block_size_col(is_int4, half_dim)

    if quant_mode == 1:
        out_scales = torch.empty(prefix_dims, dtype=torch.float32, device=x.device)
        grid = (min((prefix_dims + BLOCK_ROWS - 1) // BLOCK_ROWS, VEC_CORE_NUM),)
        swiglu_quant_dynamic_kernel[grid](...)
    else:
        out_scales = torch.zeros(prefix_shape, dtype=torch.float32, device=x.device)
        grid = (min(prefix_dims, VEC_CORE_NUM),)
        swiglu_quant_static_kernel[grid](...)

    reshape and return
```

#### L2.2 Static kernel 骨架

```python
pid = tl.program_id(0)
num_progs = tl.num_programs(0)
rows_per_prog = (prefix_dims + num_progs - 1) // num_progs
start_row = pid * rows_per_prog
end_row = tl.minimum(start_row + rows_per_prog, prefix_dims)

for row_idx in range(start_row, end_row):
    gid = 0
    in_bounds = True
    if has_group_index:
        gid = row_idx // group_size
        gid = tl.minimum(gid, num_groups - 1)
        in_bounds = row_idx < num_groups * group_size

    scale_scalar = tl.load(smooth_scales_ptr + gid) if has_smooth_scales else 0.0
    offset_scalar = tl.load(offsets_ptr + gid) if has_offsets else 0.0

    for col_start in range(0, half_dim, BLOCK_SIZE_COL):
        num_cols = tl.minimum(BLOCK_SIZE_COL, half_dim - col_start)
        col_arange = tl.arange(0, BLOCK_SIZE_COL)
        mask = col_arange < num_cols

        left_base = row_idx * x_stride_row + col_start
        right_base = left_base + half_dim
        left = tl.load(x_ptr + left_base + col_arange, mask=mask, other=0.0).to(tl.float32)
        right = tl.load(x_ptr + right_base + col_arange, mask=mask, other=0.0).to(tl.float32)

        sig_left = left / (1.0 + tl.exp(-left))
        sig_right = right / (1.0 + tl.exp(-right))
        swiglu = tl.where(activate_left, sig_left * right, sig_right * left)

        if has_smooth_scales:
            if smooth_scales_is_2d:
                scale_block = tl.load(smooth_scales_ptr + gid * half_dim + col_start + col_arange,
                                      mask=mask, other=1.0).to(tl.float32)
                swiglu = tl.where(in_bounds, swiglu * scale_block, swiglu)
            else:
                swiglu = tl.where(in_bounds, swiglu * scale_scalar, swiglu)

        if has_offsets:
            if offsets_is_2d:
                offset_block = tl.load(offsets_ptr + gid * half_dim + col_start + col_arange,
                                       mask=mask, other=0.0).to(tl.float32)
                swiglu = tl.where(in_bounds, swiglu + offset_block, swiglu)
            else:
                swiglu = tl.where(in_bounds, swiglu + offset_scalar, swiglu)

        q_val = tl.extra.ascend.libdevice.nearbyint(swiglu)
        q_val = q_val.to(tl.int32)
        q_val = tl.maximum(q_val, clip_min_val)
        q_val = tl.minimum(q_val, clip_max_val)

        if is_int4:
            # reshape + split + pack
            ...
        else:
            q_val = q_val.to(tl.int8)
            tl.store(out_ptr + row_idx * half_dim + col_start + col_arange,
                     q_val, mask=mask)
```

#### L2.3 Dynamic kernel 骨架

```python
pid = tl.program_id(0)
num_progs = tl.num_programs(0)
num_tiles = (prefix_dims + BLOCK_ROWS - 1) // BLOCK_ROWS
tiles_per_prog = (num_tiles + num_progs - 1) // num_progs
start_tile = pid * tiles_per_prog
end_tile = tl.minimum(start_tile + tiles_per_prog, num_tiles)

for tile_idx in range(start_tile, end_tile):
    start_row = tile_idx * BLOCK_ROWS
    end_row = tl.minimum(start_row + BLOCK_ROWS, prefix_dims)

    # Pass 1: per-row max_abs
    for row_idx in range(start_row, end_row):
        gid, in_bounds = compute_group(row_idx, ...)
        scale_scalar = load_scalar_smooth(gid) if has_smooth_scales else 0.0
        max_abs = 0.0
        for col_start in range(0, half_dim, BLOCK_SIZE_COL):
            swiglu = compute_row_swiglu(row_idx, col_start, activate_left,
                                        has_smooth_scales, smooth_scales_is_2d, ...)
            max_abs = tl.maximum(max_abs, tl.max(tl.abs(swiglu), axis=0))
        tl.store(scales_ptr + row_idx, max_abs)

    # Vector scale computation for this tile
    row_arange = tl.arange(0, BLOCK_ROWS)
    row_mask = row_arange < (end_row - start_row)
    max_vec = tl.load(scales_ptr + start_row + row_arange, mask=row_mask, other=1.0).to(tl.float32)
    scale_vec = int_scale_val / tl.maximum(max_vec, 1e-10)
    tl.store(scales_ptr + start_row + row_arange, scale_vec, mask=row_mask)

    # Pass 2: quantize
    for row_idx in range(start_row, end_row):
        row_scale = tl.load(scales_ptr + row_idx)
        for col_start in range(0, half_dim, BLOCK_SIZE_COL):
            swiglu = compute_row_swiglu(row_idx, col_start, activate_left,
                                        has_smooth_scales, smooth_scales_is_2d, ...)
            q_val = tl.extra.ascend.libdevice.nearbyint(swiglu * row_scale)
            q_val = clip(q_val, clip_min_val, clip_max_val)
            store_int8_or_pack_int4(q_val, ...)
```

#### L2.4 Block Size 选择策略

static/dynamic 共用 `BLOCK_SIZE_COL`，按 `half_dim` 选择：

```python
if is_int4:
    if half_dim >= 2048:   block_size_col = 4096
    elif half_dim >= 1024: block_size_col = 2048
    elif half_dim >= 512:  block_size_col = 1024
    elif half_dim >= 256:  block_size_col = 512
    else:                   block_size_col = 256
else:
    if half_dim > 2048:    block_size_col = 4096
    elif half_dim > 1024:  block_size_col = 2048
    else:                   block_size_col = 1024
```

dynamic kernel 可固定 `BLOCK_ROWS=4` 并配合 autotune 尝试 `{2, 4, 6}`。

### §3.3 Layer 3: 关键技巧（Agent 可参考但不可复制）

#### L3.1 手写 sigmoid 保证精度

**错误**:
```python
sig_left = tl.sigmoid(left)
```

**正确**:
```python
sig_left = left / (1.0 + tl.exp(-left))
```

**可替代方向**: 可尝试 `tl.extra.ascend.libdevice` 中其他 exp 近似，但需逐 case 验证精度。

#### L3.2 标量 smooth/offset 的冗余边界合并

对标量** smooth/offset 路径，可将边界判断合并到 `tl.load`：

```python
# 原实现
scale_scalar = tl.load(smooth_scales_ptr + gid)
swiglu = tl.where(in_bounds, swiglu * scale_scalar, swiglu)

# 优化后
scale_scalar = tl.load(smooth_scales_ptr + gid, mask=in_bounds, other=1.0)
swiglu = swiglu * scale_scalar
```

**收益**: 实现延时几何平均下降约 2.6%。

**注意**: 对 **2D smooth/offset** 做同样合并会触发 UB overflow，应保持 `tl.where(in_bounds, swiglu * scale_block, swiglu)`。

#### L3.3 dynamic 模式 vector scale division

**错误（标量路径，ULP 漂移）**:
```python
max_abs = tl.max(tl.abs(swiglu), axis=0)
row_scale = int_scale / tl.maximum(max_abs, 1e-10)
q = nearbyint(swiglu * row_scale)
```

**正确（向量路径，精度对齐 torch）**:
```python
# Pass 1: 写 max_abs
max_abs = tl.max(tl.abs(swiglu), axis=0)
tl.store(scales_ptr + row_idx, max_abs)

# tile 级别重新 load 向量
max_vec = tl.load(scales_ptr + start_row + row_arange, mask=row_mask, other=1.0)
scale_vec = int_scale / tl.maximum(max_vec, 1e-10)
tl.store(scales_ptr + start_row + row_arange, scale_vec, mask=row_mask)

# Pass 2: 读 row_scale
row_scale = tl.load(scales_ptr + row_idx)
q = nearbyint(swiglu * row_scale)
```

**可替代方向**: 可尝试 `BLOCK_ROWS={2,4,6}` autotune，但历史测试显示收益有限。

#### L3.4 INT4 打包模板

```python
q_val = tl.extra.ascend.libdevice.nearbyint(swiglu).to(tl.int32)
q_val = tl.maximum(q_val, clip_min_val)
q_val = tl.minimum(q_val, clip_max_val)

q_2d = tl.reshape(q_val, (BLOCK_SIZE_COL // 2, 2))
q_even, q_odd = tl.split(q_2d)

q_even_4 = q_even & 0x0F
q_odd_4 = q_odd & 0x0F
packed = ((q_odd_4 << 4) | q_even_4).to(tl.int8)

pair_arange = tl.arange(0, BLOCK_SIZE_COL // 2)
pair_mask = pair_arange < (num_cols // 2)
out_base = row_idx * (half_dim // 2) + (col_start // 2)
tl.store(out_ptr + out_base + pair_arange, packed, mask=pair_mask)
```

**可替代方向**: 可尝试 interleave 而非 split，但当前实现已验证正确。

#### L3.5 编译选项

- `multibuffer=True` 在该算子上有稳定收益，保持开启
- `unit_flag=True` 历史上测得性能下降，应避免

#### L3.6 用 `tl.load(..., mask=..., other=...)` 减少 `tl.where`

```python
# 推荐
scale_block = tl.load(ptr + off, mask=mask, other=1.0)

# 不推荐
scale_block = tl.load(ptr + off, mask=mask)
scale_block = tl.where(mask, scale_block, 1.0)
```

### §3.4 SwigluQuant 优化尝试记录与性能基准

#### 有效优化点

| 优化点 | 效果 | 关键代码位置 |
|--------|------|-------------|
| 标量 smooth/offset 边界合并到 `tl.load` | 延时下降约 2.6% | static/dynamic kernel 标量分支 |
| vector scale division | 大 fp32 dynamic shape 精度对齐 | dynamic kernel tile 级别 scale 计算 |
| `multibuffer=True` | 稳定收益 | `_route` 中 kernel 启动参数 |
| 手写 sigmoid | 避免系统性精度偏差 | sigmoid 计算处 |

#### 尝试后无效或劣化的优化点（负面经验/失败方向）

| 优化点 | 结果 | 原因 |
|--------|------|------|
| recompute 核 autotune `BLOCK_ROWS={2,4,6}` | 无提升 | `BLOCK_ROWS=4` 已接近最优 |
| 把 int32 `max/min` clip 改为 fp32 clip | 失败 | 增大 fp32 中间结果 live range，case 27 UB overflow |
| 把 recompute `BLOCK_ROWS` 从 4 降到 2 | 劣化 | 0.2434 ms > 0.2382 ms |
| 静态 `block_size_col` 限制到 2048 | 异常 | benchmark framework 延时异常，结果不可信 |

#### 参考性能数据

| 指标 | 数值 |
|------|------|
| Implementation Avg Latency | 0.2654 ms |
| Framework Avg Latency | 0.1239 ms |
| 几何平均加速比 | 0.5067x |
| 相对 Phase 3 基线 | 0.8990x |

**关键结论**:
1. **static/dynamic 分 kernel 是架构基础**：合并会导致控制流复杂、向量利用率低
2. **手写 sigmoid 是精度前提**：`tl.sigmoid` 在该后端有系统性偏差
3. **vector scale division 是 dynamic 模式精度关键**：标量除法会在大 fp32 shape 上 off-by-1
4. **INT4 打包必须严格按 (odd << 4) \| even 布局**：与 `torch.quint4x2` 一致
5. **UB 压力是主要瓶颈**：动态核 `BLOCK_ROWS` 受限；避免长期保留 fp32 中间结果
6. **speedup_vs_torch < 1 不代表失败**：framework 是优化后的 NPU 原生算子，重点看 implementation latency

---

## §4 WeightQuantBatchmatmul 算子（weight-quantized batch matrix multiplication）

**算子类别**: `matmul` + `quantization`
**典型特征**: `x` fp16/bf16 `[M, K]`, `weight` int8 `[K, N]`, `antiquant_scale/offset` 支持 per-tensor `[1]` / per-channel `[1, N]` / per-group `[G, N]`；可选 `bias`；输出 dtype 与 `x` 相同
**性能基准**: 48/48 verify pass，几何平均加速比 **0.9133x** vs torch（本轮 repeats=50 实测，run-to-run 方差带 0.91–0.95；超 target 0.8x 达 14%；refactor_iter_20 = opt_iter_19 + N-strip chunking for L2-exceeding workspaces）

> **评判口径**: framework 参考实现为 `torch_npu.npu_weight_quant_batchmatmul`，是 NPU 高度优化的原生算子；Triton DSL 在部分 shape 上存在结构性劣势。优化目标应聚焦在 **implementation latency** 降低与 **48/48 精度全过**。
>
> ⚠️ **方差带提示（benchmark 噪声）**: 本算子含多个 ~17-20us 的极小 M case（c1/5/8/11/14/21/24/27/30/33/38/40/42），对 profiler 抖动敏感，单次 geomean 浮动 ±0.03 属正常。代码不变（sha256 一致）时多次实测会落在 **0.91–0.95** 区间。**判 target_reached 必须看是否稳定 ≥ 0.8，不要纠结单次 0.94 vs 0.91 的差异**。

### §4.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 必须按 group / non-group 分裂为两个专用 kernel
- **必须**为 `antiquant_group_size > 0`（group 模式）和 `antiquant_group_size == 0`（non-group 模式）分别编写 `@triton.jit` kernel，host 侧通过简单 `if is_group` 分支启动对应 kernel。
- **Why:** group 模式需要按 group id 加载 scale/offset，non-group 模式 scale/offset 在 K 维度恒定，合并会导致大量 `tl.where(IS_GROUP, ...)` 分支，降低向量利用率并增加 UB 压力。
- **注意**: 当前项目的 `validate_triton_impl.py` 要求 `kernel[grid](...)` 必须直接出现在 `ModelNew.forward()` 内部，不能封装到 `self._route()` 等辅助方法中。

#### L1.2 group 模式必须保留原始 `[G, N]` scale/offset，禁止 host 侧物化为 `[K, N]`
- **必须**将 group scale/offset 以原始 `[G, N]` 布局传入 kernel，由 kernel 按 `group_id = k_start // group_size` 加载单条 `[1, N]` 行。
- **禁止**在 host 侧用 `view(G,1,N).expand(G,group_size,N).contiguous()` 物化为 `[K, N]`。
- **Why:** host 物化会引入大量 CPU 时间与额外 HBM 写入；原始 `[G, N]` 每 K-block 只读一条 `[1, N]`，总读取量仅为 `K/group_size * N`，远低于物化后的 `K * N`。

#### L1.3 group kernel 中每个 K-block 必须落在单一 group 内
- **必须**选择 `BLOCK_K` 满足 `BLOCK_K <= group_size` 且 `group_size % BLOCK_K == 0`。
- **Why:** 保证当前 K-block 内所有 K 位置共享同一个 group 的 scale/offset，从而只需加载一条 `[1, N]` scale/offset 行并广播到 `[BLOCK_K, N]`。
- **推荐配置**:
  - `group_size % 128 == 0` 且 dtype 为 fp16: `BLOCK_K = 128`
  - 其他 group 场景（含 bf16 group）: `BLOCK_K = 64`

#### L1.4 dequant 域必须按 `x.dtype` 分流
- **fp16 输入**: 在 fp16 低精度域完成 `(weight + offset) * scale`，最后 cast 到 fp32 累加。
- **bf16 输入**: 在 fp32 域完成 `(weight + offset) * scale`，再 cast 到 bf16 后回 fp32（模拟 NPU cube L0C 行为）。
- **Why:** bf16 加法尾数损失会导致 verify fail；fp16 路径若在 fp32 域 dequant 也会与 NPU 参考实现产生微小差异。

#### L1.5 必须使用 1D Grid，grid 大小钳制到实际 Vector Core 数量
- **必须** `grid = (min(total_tiles, num_cores),)`，其中 `num_cores = triton.runtime.driver.active.utils.get_device_properties(device).get("num_vectorcore", 48)`。
- **禁止**多维 grid。
- **Why:** 1D grid 配合核内 `for tile_idx in range(pid, total_tiles, num_programs)` 是 Ascend vector core 上最稳定的负载均衡模式。

#### L1.6 `ModelNew.forward()` 中禁止任何 torch 计算，kernel 启动必须直接可见
- **必须**所有核心计算在 `@triton.jit` kernel 内完成；`forward()` 仅做 shape 处理、contiguous、buffer 分配、条件分派。
- **Why:** `validate_triton_impl.py` 会 flag `forward()` 中的 torch 计算或间接 kernel 调用。

#### L1.7 必须使用 int32 索引
- **必须**将 `tl.program_id`、`tl.arange` 结果 `.to(tl.int32)`，避免 int64 地址计算降级。

#### L1.8 bf16 group 路径必须保守选择 block size
- **必须**对 bf16 group 使用 `BLOCK_N = 128, BLOCK_K = 64`；更大的 `BLOCK_N` 或 `BLOCK_K` 会触发 `MLIRCompilationError` / `hivm.hir.vcast` 不支持。

#### L1.9 non-group 大 N fp16 路径可放大 block size
- 当 `x.dtype == fp16` 且 `N > 1024` 时，可安全使用 `BLOCK_N = 256, BLOCK_K = 128`。
- 当 `x.dtype == bf16` 或 group 模式时，`BLOCK_N = 256` 容易触发编译错误，应保持 `BLOCK_N = 128`。

#### L1.10 non-group 大 M 路径必须解耦为「dequant workspace + 纯 GEMM」（msprof Roofline 验证）
- **必须**对 `is_group == False` 且 `M >= 256` 的 case 走**两 kernel 解耦**路径：
  1. `dequant_weight_kernel`：int8 weight 一次性 dequant 到 low_dtype workspace `[K, N]`（消除 fused kernel 中 dequant 被 M/BLOCK_M 个 tile 重复执行的冗余）；
  2. `matmul_kernel`：纯 GEMM `x[M,K] @ w_fp[K,N]`（+bias），K 循环无 dequant 依赖，Cube 可软件流水转为 compute-bound。
- **禁止**对小 M（`M < 256`）的 non-group 路径使用解耦：无 dequant 冗余可消，workspace 分配/读写开销主导，实测劣化。
- **Why（msprof 证据）**: fused kernel 经 `msprof op --aic-metrics=Roofline` 判定为 `latency bound: pipeline caused`，Cube ratio 仅 **7.6%（bf16）/ 15.8%（fp16 大 N）**，MTE2 21–24% 未饱和、compute <20%。根因是 int8→low_dtype dequant 位于每个 `tl.dot` 关键路径且跨 M-tile 重复执行，Cube↔Vector 同步等待饿死 Cube。解耦后 non-group 大 M case 实测 geomean 0.5807→0.6240。
- **UB 约束**: dequant kernel 用 `BLOCK_K=64, BLOCK_N=128`（fp32 中间缓冲 + group 全 tile scale/offset 控制 <192KB）。纯 GEMM BLOCK 选择见 **L1.12**（修正早期错误：BM128×BN256×BK128 恰好容纳 192KB 双 buffer，是最优配置；**仅 BM256×BN256 溢出**）。
- **精度保障**: workspace dequant 与 fused 逐元素数学完全一致（bf16 走 fp32 中间域），48/48 verify 全过。

#### L1.11 group 路径可在「结构信号」命中时解耦，但 dequant 必须用标量 g（msprof 验证，修正 L1.10 旧禁令）
- ⚠️ **本条修正 L1.10 早期的「禁止对 group 解耦」禁令**——该禁令是基于首版 group 解耦的**逐元素 gather dequant** 实测劣化得出的，但根因是 gather 而非 group 本身。修正后 group 解耦在结构信号命中时显著收益。
- **结构信号（命中任一即对 group 解耦）**:
  1. `group_size <= 64`：fused kernel 被迫 `BLOCK_K <= group_size`（16/32/64），Cube 在微 dot 上严重饿死；
  2. `x.dtype == bfloat16`：fused 把 fp32 dequant 留在 dot 关键路径（Vector/MTE2 主导，msprof fixpipe 仅 0.5%——bf16 真实瓶颈是关键路径 dequant 而非 fixpipe）；
  3. `M >= 512`：M-tile dequant 冗余高 + `[K,N]` workspace 摊销好。
- **保留 fused group** 的场景：`fp16 + group_size >= 128 + M < 512`（fused 本就用 `BLOCK_K=128`，解耦 workspace 开销略亏）。
- **关键实现约束——dequant 必须用标量 g + 单行连续 scale 加载**:
  - host 选 `DQ_BLOCK_K` 使 `DQ_BLOCK_K | group_size`，整个 tile 落在单一 group；
  - kernel 内 `g_scalar = k_start // group_size`（标量），加载单条连续 `[1,N]` scale/offset 行并广播；
  - **禁止** `g = offs_k // group_size`（`[BLOCK_K]` 向量）做 2D gather——离散访存使 dequant 慢 ~4x，首版因此对全部 group case 劣化（0 胜 15 负）。
- **msprof 证据**: fused group case Cube 常 <8%，标量 g 解耦后 geomean 0.6240→0.6770（+8.5%），11 个 group 解耦 case 全部净收益、0 回退。

#### L1.12 解耦 GEMM 必须用 BLOCK_N=256（大 N bandwidth-bound，msprof Roofline 验证）
- **必须**对解耦路径的纯 GEMM（`matmul_kernel`）使用 `BLOCK_N = 256, BLOCK_K = 128`（BLOCK_M 自适应 32/64/128），适用于 fp16 与 bf16（matmul 只加载 low_dtype，UB 占用与 dtype 无关）。
- **Why（msprof 证据）**: 大 N 解耦 `matmul_kernel` 为 bandwidth bound：`Cube 20% / MTE2 54% / aic_main_mem_read_bw ~1.6 TB/s 聚合`（HBM 打满）。根因是 dequant workspace fp16 `[K,N]`（64–128MB）远大于 ~96MB L2，被每个 M-tile 从 HBM 重读。放大 BLOCK_N 128→256 减半 N-tile 数，纯 GEMM geomean 0.6770→0.7946（+17.4%）。
- **UB 边界**: `BM=128 × BN=256 × BK=128` 双 buffer = 192KB（恰好容纳）。**仅 `BM=256 × BN=256` 溢出**。
- **⚠️ 否决方向（避免重复尝试）**:
  1. 改 tile 遍历序为 n-slow/m-fast 企图 L2 复用 workspace——实测更慢；L2 跨核共享不足以摊销大 footprint。
  2. 大 N 改回 fused 靠 int8 权重命中 L2——多数更差；大 M 时 dequant 在 dot 关键路径主导，fused 同样不命中 L2。
- **适用判据**: `MTE2 > 40% 且 Cube < 30% 且 read_bw≈HBM 峰值` → bandwidth bound → 放大 BLOCK_N。

#### L1.13 大 N 解耦路径必须做 N-strip chunking（L2-exceeding workspace 的算法级解法，refactor_iter_20）
- **必须**对解耦路径（L1.10/L1.11 命中）且 `K * N * sizeof(low_dtype) > L2_THRESHOLD`（取 **60MB**，~96MB L2 留 1/3 余量）的 case 走 **N-strip 切分**：把 N 切成多个 strip，每个 strip 的 dequant workspace `[K, chunk_n]` 能完整驻留 L2，再对该 strip 跑全部 M-tile 的 GEMM——此时 workspace 从 L2 命中而非 HBM 重读。
- **Why（修正 L1.12 末尾"workspace 重读是固有代价"的判断——该判断在 N-strip 下不再成立）**: L1.12 的 BLOCK_N=256 把单次 GEMM 访存减半，但整个 `[K,N]` workspace（大 N 时 64–128MB）仍远大于 ~96MB L2，被每个 M-tile 从 HBM 重读 8×。**正确破法是 N-strip**：每次只物化一个 strip 的 workspace（32MB ≤ L2），让它跨 M-tile 全程命中 L2，把 HBM 重读从 "M-tile 数 × 全 workspace" 降到 "num_strips × strip workspace"。
- **实现要点（deliverable 实测 1.48–1.75× on c4/37/45/46/47/48）**：
  - 触发条件：`K * N * sizeof(low_dtype) > 60MB`（约 ~96MB L2 的 2/3）。
  - 计算单 strip 大小：`chunk_n = floor(48MB / (K * dtype_size))`，再向下对齐到 `BLOCK_N_GEMM=256` 的整数倍，保证至少一个完整 N-tile。
  - Host 侧按 `chunk_idx` 循环 N 维：对每段 `weight[:, n_start:n_end]` 启动 `dequant_weight_kernel` 写 `[K, chunk_n]` workspace；再启动 `matmul_kernel` 让**所有 M-tile 顺序消费该 strip**。workspace 全程驻留 L2，跨 M-tile 命中复用，避免整 `[K,N]` buffer 被反复从 HBM 重读。
  - GEMM tile 仍用 `BM128/BN256/BK128`（L1.12）。
  - 输出按 N 维 slice 写入 `out[:, n_start:n_end]`，不同 strip 输出 disjoint，**禁止用 atomic_add**（相对 K-split 的核心优势）。
  - strip 内 dequant 仍遵守 L1.11：group 模式用标量 `g = k_start // group_size` + `DQ_BLOCK_K | group_size` + 单行连续 scale/offset 加载，`DQ_BLOCK_N=128`。
- **触发 case 实测（repeats=50）**: c4 M=1024 K=4096 N=8192 → 1.481×；c37/47 M=1024 K=4096 N=16384 → 1.58–1.59×；c45 → 1.752×；c46/48 M=2048 K=8192 N=4096 → 1.56–1.60×。相对 opt_iter_19（这些 case 仅 ~0.54×）实现质变，geomean **0.7946 → 0.9133**（+15%）。
- **适用判据**: 解耦路径 + `K*N*2 > 60MB` → 必走 N-strip；`K*N*2 ≤ 60MB` 走常规单 workspace 解耦（L1.12）即可（strip 切分对能装进 L2 的 workspace 反而增加 host 循环开销）。

### §4.2 Layer 2: 算法骨架（纯描述，无完整源码）

#### L2.1 Host 侧分派

`forward()` 仅负责：
1. 输入 `contiguous()` + reshape 为 2D `[M, K]` / `[K, N]`；
2. 按 `antiquant_group_size>0` 判定 `is_group`；
3. 按 dtype / shape 选择 `(BLOCK_M, BLOCK_N, BLOCK_K)`；
4. 启动对应 kernel。

分派策略表（遵守 L1.8–L1.13）：

| 场景 | kernel | BLOCK_N | BLOCK_K | BLOCK_M |
|------|--------|---------|---------|---------|
| non-group, N≤1024 或 bf16 | fused | 128 | 128 | 32/64/128 按 M |
| non-group, N>1024, fp16 | fused | 256 | 128 | 32/64/128 按 M |
| group, bf16 | group | 128 | 64 | 32/64/128 按 M |
| group, gs%128==0, fp16 | group | 128/256 | 128 | 32/64/128 按 M |
| 其他 group | group | 128 | 64 | 32/64/128 按 M |

解耦判定见 L1.10–L1.13：命中解耦时先启动 `dequant_weight_kernel` 写 workspace，再启动 `matmul_kernel` 做纯 GEMM；命中 N-strip 时 host 侧循环 N 维 strip。

#### L2.2 Group kernel 骨架

1. **1D grid + 核内 stride 循环**：`pid = program_id(0)`，`for tile_idx in range(pid, total_tiles, num_programs)`。
2. 每个 tile 取 `m_tile = tile_idx // num_n_tiles`，`n_tile = tile_idx % num_n_tiles`。
3. K-loop 内用**标量 group id**：`g = k_start // group_size`；按 `g` 加载单条 `[1, N]` scale/offset，广播到 `[BLOCK_K, BLOCK_N]`。
4. 每个 K-block 必须满足 `BLOCK_K | group_size`（L1.3），保证不跨 group。
5. dequant 域按 `IS_BFLOAT16` 分流（L1.4）：fp16 在 fp16 域，bf16 在 fp32 域。
6. 累加用 fp32 accumulator，`tl.dot(x_tile, w_tile, acc)`；最后加 bias 并 cast 回 `x.dtype` store。

#### L2.3 Non-group kernel 骨架

1. 同样 1D grid + stride 循环。
2. scale/offset 在 K 维恒定，**提到 K 循环外预加载 + cast**（L3.4）：
   - 预计算 `scale_f32` / `scale_low`（fp16 路径）和 `offset_f32` / `offset_low`（若存在 offset）。
3. K-loop 内只做：load int8 weight → cast → `*` scale `(+ offset)` → cast → `tl.dot`。
4. 最后加 bias、cast、store。

#### L2.4 N-strip chunking 骨架（解耦路径）

当 `K*N*sizeof(low_dtype) > 60MB` 时：
1. 计算 `max_chunk_n = 48MB // (K * dtype_size)`，向下对齐到 `BLOCK_N_GEMM=256`。
2. Host 侧 `for chunk_idx in range(cdiv(N, n_chunk))`：
   - 取 `weight[:, n_start:n_end]`；
   - 启动 `dequant_weight_kernel` 写 `w_fp_slice[K, chunk_n]`；
   - 启动 `matmul_kernel` 计算 `x @ w_fp_slice → out[:, n_start:n_end]`。
3. 所有 M-tile 顺序处理同一个 strip，workspace 在 L2 中命中复用；输出 N 维 disjoint，**无需 atomic**。

### §4.3 Layer 3: 关键技巧（Agent 可参考但不可复制代码结构）

#### L3.1 group scale 单 row 广播替代 host 物化

**问题**: group 模式若将 `[G, N]` expand 为 `[K, N]`，会占用大量 host 时间与 HBM 带宽。

**解决**: kernel 内按 `g = k_start // group_size` 只加载一条 `[1, N]`，通过 `scale_row[None, :]` 广播到 `[BLOCK_K, N]`。

**可替代方向**: 若未来后端支持更高效的 gather，可尝试 vector gather；但在当前 Ascend 后端，单 row 广播更稳定。

#### L3.2 按 `group_size` 选择 `BLOCK_K`

**问题**: group_size 为 64 与 128 时，最优 `BLOCK_K` 不同。

**解决**:
- `group_size % 128 == 0` 且 dtype 为 fp16: `BLOCK_K = 128`，减少 K 循环迭代数。
- 其他 group 场景: `BLOCK_K = 64`，避免跨 group 并保证编译通过。

**可替代方向**: 对固定 group_size 可离线搜索最优 `BLOCK_K`，但需保证不跨 group。

#### L3.3 按 dtype / shape 分 block 策略

| 场景 | 推荐 BLOCK_N | 推荐 BLOCK_K | 原因 |
|------|-------------|-------------|------|
| fp16 non-group, N > 1024 | 256 | 128 | 大 tile 提升 Cube 利用率 |
| fp16 non-group, N <= 1024 | 128 | 128 | 平衡 |
| bf16 non-group | 128 | 128 | BLOCK_N=256 会触发 MLIRCompilationError |
| fp16 group, gs%128==0 | 128/256 | 128 | 单 group 单 block |
| bf16 group | 128 | 64 | 保守，避免编译错误 |

**可替代方向**: 对小 M (<64) 使用 `BLOCK_M=32`，中等 M (64-127) 使用 `BLOCK_M=64`，大 M 使用 `BLOCK_M=128`；`BLOCK_M=256` 易触发 UB overflow，不建议。

#### L3.4 循环不变量 cast 外提

**问题**: non-group 模式下 `scale_tile` / `offset_tile` 在 K 循环内重复 cast。

**解决**: 在 K 循环外预计算 `scale_f32` / `scale_low` / `offset_f32` / `offset_low`，K 循环内直接使用。

**可替代方向**: 现代编译器可能自动外提，但显式外提更可控，实测有小幅收益。

### §4.4 性能基准与优化尝试记录

| 版本 | cases | geomean | 相对 Phase 3 基线 | 说明 |
|------|-------|---------|------------------|------|
| Phase 3 基线 | 48/48 | 0.1026x | 1.0x | 通用 kernel，host 物化 group scale |
| opt_iter_9 (kernel 分裂 + group 单 row 广播) | 48/48 | 0.4558x | 4.44x | 首次算法级优化，group 路径去掉 host 物化 |
| opt_iter_10 (fp16 大 N BLOCK_K=128) | 48/48 | 0.4979x | 4.85x | non-group 大 N 同时放大 N/K tile |
| opt_iter_11 (group_size=128 BLOCK_K=128) | 48/48 | 0.5290x | 5.16x | group_size=128 的 fp16 group 减少循环 |
| opt_iter_12 (bf16 non-group BLOCK_K=128) | 48/48 | 0.5582x | 5.44x | bf16 非 group 放大 K tile |
| opt_iter_14 (fp16 group N>=1024 BLOCK_N=256) | 48/48 | 0.5705x | 5.56x | fp16 group N=1024 时放大 N tile |
| **opt_iter_15b (cast 外提)** | **48/48** | **0.5807x** | **5.66x** | fused 最终版本 |
| opt_iter_16 (纯解耦 dequant+GEMM) | 48/48 | 0.3640x | 3.55x | 全 case 解耦：non-group 大 M 赢，group/小 M 大输 → 总体劣化（group 输因逐元素 gather dequant，见 opt_iter_18 修正） |
| opt_iter_17 (mixed: 解耦 non-group M>=256 + fused 其余) | 48/48 | 0.6240x | 6.08x | msprof Roofline 驱动 non-group 大 M 解耦 |
| opt_iter_18 v1 (group 解耦-逐元素 gather) | 48/48 | 0.4275x | 4.17x | group 解耦首版：dequant 用 `g[:,None]` 2D gather，15 个 group case 全劣化（0 胜 15 负） |
| **opt_iter_18 (group 解耦-标量 g + 结构路由)** | **48/48** | **0.6770x** | **6.60x** | **msprof 驱动 group 路径解耦：标量 g 连续 scale 加载消除 gather；gs<=64/bf16/M>=512 解耦，其余 fused** |
| **opt_iter_19 (解耦 GEMM BLOCK_N=256)** | **48/48** | **0.7946x** | **7.74x** | **msprof 驱动：大 N 解耦 GEMM bandwidth-bound（MTE2 54%, Cube 20%, HBM 打满），workspace(64-128MB)>L2 被 M-tile 重读；放大 BLOCK_N 128→256，纯 GEMM +36-38%** |
| **refactor_iter_20 (N-strip chunking, 最终 deliverable)** | **48/48** | **0.9133x**（方差带 0.91-0.95） | **8.90x** | **算法级消除 L2-exceeding workspace 重读：`K*N*2>60MB` 时按 N 切 L2-resident strip（48MB 上限，对齐 BN256），每 strip dequant 一次 → 全部 M-tile L2 命中复用；输出 N 维 disjoint 无需 atomic。大 N case 0.54→1.48-1.75×（c4/37/45/46/47/48）** |

**有效优化点**:
1. group / non-group kernel 分裂 + group 单 row 广播（最大收益，4.4x）
2. fp16 大 N 使用 BLOCK_N=256 / BLOCK_K=128
3. 按 group_size 选择 BLOCK_K=128（gs=128 时）
4. bf16 non-group 使用 BLOCK_K=128
5. 循环不变量 cast 外提（小幅收益）
6. **non-group 大 M 解耦为 dequant workspace + 纯 GEMM**（msprof 驱动，0.5807→0.6240）：消除 dequant 跨 M-tile 重复执行，Cube 从 latency-bound(7.6-15.8%) 转 compute-bound
7. **group 路径解耦（标量 g + 结构路由，0.6240→0.6770）**：gs<=64/bf16/M>=512 命中时解耦；dequant 用标量 g（`DQ_BLOCK_K | group_size`）+ 单行连续 scale 加载，禁用逐元素 2D gather；fp16/gs>=128/M<512 保留 fused
8. **解耦 GEMM 放大 BLOCK_N=256（大 N bandwidth-bound，0.6770→0.7946）**：msprof 显示大 N 解耦 GEMM 是 bandwidth bound（MTE2 54% / Cube 20% / HBM 打满），workspace(64-128MB)>L2 被 M-tile 重读；放大 BLOCK_N 128→256，纯 GEMM +36-38%（case37/46/4）
9. **N-strip chunking（L2-exceeding workspace 算法级解法，0.7946→0.9133）**：L1.12 的 BLOCK_N=256 只减半单次访存，整 workspace 仍被 M-tile 重读。N-strip 把 N 切成 L2-resident 的 strip（48MB 上限），每 strip dequant 一次供全部 M-tile L2 复用，输出 N 维 disjoint 无需 atomic。大 N case 0.54→1.48-1.75×（详见 L1.13）

**已验证无效/失败的优化方向**:

| 优化方向 | 结果 | 原因 |
|----------|------|------|
| group scale host 物化为 `[K, N]` | 劣化 | host CPU 时间与额外 HBM 写入 |
| group 模式 in-kernel per-K gather（未广播） | 劣化 | gather 开销大，0.0735x |
| bf16 non-group / bf16 group **fused** BLOCK_N=256 | 编译失败 | `hivm.hir.vcast` unsupported / UB overflow（仅限 **fused** kernel 的 fp32 dequant 中间体；解耦 GEMM 的 BN=256 不受此限，见 L1.12） |
| bf16 group BLOCK_K=128 | 编译失败 | `hivm.hir.vcast` unsupported |
| BLOCK_M=256 × BLOCK_N=256 | UB overflow | `cc overflow`（仅 BM256×BN256 组合溢出；BM128×BN256×BK128 恰好容纳 192KB） |
| 解耦 GEMM tile 遍历序改 n-slow/m-fast（L2 复用 workspace N-strip） | 更慢 | case37 GEMM 0.94 vs 0.85ms；L2 跨核共享不足以摊销 128MB footprint |
| 大 N 改回 fused（靠 int8 权重 64MB 命中 L2） | 多数更差 | 仅 case45 略胜；大 M 时 dequant 在 dot 关键路径主导，fused 重读 int8 8× 同样不命中 L2 |
| **scale-hoisted fused / fold-out（post-deliverable 重构）** | 35/48 挂 | 把 per-channel scale 移出 K-loop 直读 int8 权重省 workspace 往返。no-offset 精度超阈（fold-out 比 torch_npu 更精）；offset 触发 `hivm.hir.vcast root alloc` UB overflow。精度与编译双重阻塞 |
| **BLOCK_M=64 并行填充（post-deliverable 重构）** | 0.9437→0.7903(-0.15) | 推翻并行假设：decoupled matmul 是 per-tile Cube 效率受限而非 aggregate-MLP 受限，[128,128]@[128,256] 是高效 Cube dot shape，BM=64 崩 Cube 利用率 |

**关键结论**:
1. **Kernel 分裂是架构基础**：group 与 non-group 的 scale 访问模式差异巨大，必须分 kernel。
2. **group scale 单 row 广播是核心优化**：避免 host 物化和 gather，是收益最大的单点改进。
3. **block size 必须按 dtype/shape 分派**：bf16 和 group 模式对大块更敏感，fp16 non-group 可激进放大。
4. **BLOCK_M=256 仅在与 BLOCK_N=256 组合时溢出**：`BM=256×BN=128` 单独可行；但解耦 GEMM 实测 `BN=256×BM=128` 更优（N-tile 减半胜于 M-tile 减半），故 BLOCK_M 仍保持 128。
5. **dequant 域必须按 dtype 分流**：fp16 在 fp16 域，bf16 在 fp32 域，否则精度 fail。
6. **msprof Roofline 是判定 latency-bound 的权威手段**：fused kernel 经 profiling 判定 `latency bound: pipeline caused`（Cube 7.6-15.8%），直接驱动「non-group 大 M 解耦 + 纯 GEMM」重构，geomean 0.5807→0.6240。后续 quant 类 matmul 若 Cube<20% 且带宽未饱和，应优先考虑解耦 dequant。
7. **解耦须按 case 分流（mixed strategy），且 group 解耦成败取决于 dequant 访存模式**：早期「group 解耦必劣化」的结论是**错误**的——根因是首版用了逐元素 2D gather dequant（离散访存慢 ~4x）。改用标量 g + 连续行加载后，group 解耦在结构信号（gs<=64 / bf16 / M>=512）命中时显著收益（opt_iter_18：11 个 group case 全净收益，0 回退）。仍须保留 fused 的场景：fp16+gs>=128+M<512（fused 本就 BLOCK_K=128，workspace 开销略亏）。
8. **bf16 真实瓶颈是关键路径 dequant 而非 fixpipe**：msprof 显示 fused bf16 的 fixpipe 仅 0.5%，bf16 慢是因为 fp32 dequant 留在 dot 关键路径（Vector/MTE2 主导）。解耦即消除——最差 bf16 group case 由 group 解耦一并修复（c28 0.283→0.519, c29 0.279→0.607, c36 0.518→0.910）。
9. **目标 0.8x 已达成（refactor_iter_20 实测 0.9133x，方差带 0.91-0.95，超目标 14%）**：N-strip chunking（L1.13）是破局点——把大 N 解耦 case 从 opt_iter_19 的 ~0.54 拉到 1.48-1.75×，彻底解决 L1.12 遗留的 "workspace > L2 被 M-tile 重读" 问题。剩余拖低 geomean 的 laggard 转移到 **medium-N 解耦 case（c2/12/25/28/41/44，0.62-0.79）**，这些 case 见结论 11。
10. **bandwidth-bound 与 compute-bound 的 msprof 区分（L1.12）**：解耦后 GEMM 的瓶颈从 fused 的「Cube 饿死（latency-bound, Cube<16%）」转为「HBM 打满（bandwidth-bound, MTE2 54% / Cube 20% / read_bw 接近峰值）」。前者靠解耦/放大 tile 提升 Cube，后者放大 BLOCK_N 减半访存次数。**判据**：msprof `MTE2>40% 且 Cube<30% 且 read_bw≈HBM 峰值` → bandwidth bound → 放大 BLOCK_N；若 workspace 体积>L2，重读无法靠遍历序（n-slow/m-fast）消除（已实测更慢）。
11. **medium-N laggard 已到 Triton toolchain 天花板**：refactor_iter_20 达 0.9133x 后，剩余 laggard 集中在 M∈[200,512] 的 medium-N 解耦路径（0.62–0.79）。msprof 显示该路径 `latency bound: memory caused` 且 MTE2 带宽余量 <80% peak，说明有 headroom 但 Triton 抓不到。两条算法重构均证伪：scale-hoisted fused 被精度（no-offset）和编译（offset UB overflow）双重阻塞；BLOCK_M=64 并行填充 -0.15 崩 Cube 效率。medium-N 剩余 gap 到 1.0 是 **torch_npu 硬件 dequant-on-load 单 kernel 的结构性优势**——Triton 无法软件流水 dequant-across-dot（vmul/vcast root-alloc rejection），也无法访问硬件 dequant 路径。**后续 medium-N 量化 matmul 若解耦路径 <0.8x，不要调参/fold-out/BM缩放，直接标注 toolchain ceiling**。

---

## §5 GroupedMatmulSwigluQuant 算子（group_list 变长分组 int8 GEMM + SwiGLU + 动态量化）

**算子类别**: `grouped-matmul` + `quantization`
**典型特征**: `x` int8 `[M,K]`，`weight` 为长度 E 的 tensor list（每 expert 独立 int8 `[K,N]` + fp32 `weight_scale[e]` `[N]`），fp32 per-row `x_scale` `[M]`，`group_list` `[E]` 变长行分组（前缀和定界，允许空 expert）；计算链 = int8 grouped GEMM（int32 acc）→ 两级反量化（×ws → fp16 → ×xs → fp16）→ SwiGLU → per-token 动态量化（int8 + f32 scale 输出）
**性能基准**: 60/60 verify pass，几何平均加速比 **1.3202x** vs torch（相对 Phase 3 基线 0.7978x 提升 1.6548x；impl avg 0.1010ms；E=2~8 / M=100~4096 / K=1024~8192 / N=256~4096；终态为 opt_iter_11）

> **硬件 / 工具链绑定**: Ascend **910B2C**（24 cube / 48 vector cores）/ CANN 8.5.1。L0C 128KB 档位、UB 192KB、MTE pipe 画像均与该 SKU 绑定，跨芯片外推前必须重验。
>
> **与 §3 / §4 的关系**: 本算子是 §4（int8 weight GEMM + 反量化）的 grouped 变长分组形态 + §3（SwiGLU + 动态量化）的尾部融合。**K2 量化段的数值链约束与 §3 完全同源**（手写 sigmoid / round-half-away-from-zero / int32 阶段 clip / vector scale），可复用；**K1 GEMM 段的调参与解耦经验不通用**——§4 的 dequant workspace 解耦 / N-strip chunking 在 grouped 变长分组下不适用（见 §5.4 失败方向表）。

### §5.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 ★★★ 禁止 per-expert 循环 launch，必须 persistent 单 launch 跨 expert 全局编号 tile
- **禁止** `for e in range(E): kernel[grid](...)` —— 每次迭代隐含同步与调度开销，speedup 随 E 单调恶化（初始实现 E=2 geo 1.104 → E=8 geo 0.472）。
- **必须** tile 跨 expert 全局连续编号，`grid = (min(cube_cores, total_tiles),)`，kernel 内 `for tile_idx in range(pid, num_tiles, NUM_CORES)`。
- **Why:** E 次 launch 的调度/同步开销在多 case 几何平均下是最大单一损失源；persistent 化实测 0.7978→1.0965（+37.4%，本算子头号杠杆）。
- **How to apply:** 见 L3.1。

#### L1.2 ★★★ 禁止逐 expert `.item()`，group_list 只做一次 D2H
- **必须** host 侧一次 `gl_cpu = group_list.cpu().tolist()`，后续全部用 CPU list。
- **禁止** kernel 启动前逐个 `group_list[e].item()`（E 次设备→主机同步）。
- **Why:** 每次 `.item()` 是一次完整 D2H 同步，与 L1.1 叠加构成 host 调度瓶颈。

#### L1.3 ★★ 跨 kernel 元数据必须单次 H2D 打包为 int64 tensor
- **必须** 把全部 per-expert 元数据（指针 delta / tile 边界 / 行边界）打包为单个 `torch.tensor(meta, dtype=torch.int64)` 一次上传，kernel 内按 constexpr 偏移切分。
- **推荐布局**: `[w_delta(E) | wsd_delta(E) | tile_lo(E+1) | row_lo(E) | row_hi(E)]`，空 expert 剔除后 `E_eff` 作为 constexpr 传入。
- ⚠️ `tile_lo` 比 E **多一个末位哨兵**（= total_tiles，供前缀扫描比较），因此其后 `row_lo` / `row_hi` 段的首地址偏移是 `3*E_eff+1` / `4*E_eff+1`——这个 **+1** 推导易错，偏移错一位全盘访存错位（精度灾难）。
- **Why:** 逐项 `torch.tensor(...).to(device)` 是 E+ 次小 H2D；meta 常驻 L2，kernel 内标量 load 开销可忽略。

#### L1.4 指针差必须按元素单位换算
- **必须** host 侧 `(ptr_a - ptr_b) // element_size()` 后再入 meta（kernel 内指针算术以**元素**为单位）。
- **Why:** int8 weight 与 fp32 weight_scale 的 element_size 不同，字节差直接传会错位访存（精度灾难，不是性能问题）。

#### L1.5 BM×BN 不得跨 L0C 32768 元素档
- **禁止** `BM×BN = 32768`（如 128×256）——L0C 档位硬失败（编译/运行错误），实证 BM≤64（BN=256 时）。
- **安全档**: 64×256 = 128×128 = 16384 元素（int32 acc 64KB）。
- **Why:** L0C 128KB 物理上限按 acc 元素数分档，16384×4B=64KB 安全，32768×4B=128KB 恰好顶满被拒。

#### L1.6 双缓冲 tile 必须核算 UB 192KB
- **必须** `2*(BM*BK + BK*BN) ≤ 192KB`（int8 输入按 1B/elt）。BLOCK_K=256、BM=64、BN=256 时 2*(16KB+64KB)=160KB 可行；BK=512 溢出。
- **Why:** 编译器自动双缓冲 A/B tile，超 UB 直接编译失败或静默拆小 tile。

#### L1.7 数值链冻结（bit 级对齐）
- **必须**保持反量化链 op 顺序：`acc(i32).to(f32) × ws → fp16 → f32 × xs → fp16`（两次 fp16 舍入点）；sigmoid 手写 `g/(1.0+tl.exp(-g))` 在 f32（同 §3 L1.4）；sw → fp16 → f32；round-half-away-from-zero（`tl.where(q>=0, q+0.5, q-0.5)`，同 §3 L1.10 思路）；clamp 在 int32 阶段 `tl.maximum/tl.minimum`（同 §3 L1.7）。
- **禁止** libdevice sigmoid 替换（硬件 sigmoid 指令舍入行为不同 → fp16 舍入点漂移）、禁止融合/移动 fp16 舍入点。
- **Why:** 60/60 全过的精度是 bit 级对齐的结果；K1 epilogue 与 K2 之间通过 GM fp16 中转固化了舍入点位置（de-mix 实验证明移动舍入点位置即使"等价"也会改变实测行为，见 §5.4）。

#### L1.8 expert 定位段保持 constexpr 标量前缀扫描，禁止向量化
- **必须** `for i in range(E_C - 1): e += (tile_idx >= bound).to(tl.int32)`（E_C constexpr → 编译期展开，int32 比较）。
- **禁止** bounds 向量 load + `tl.sum` 聚合的向量化改写（实测 0.9059 显著劣化）。
- **Why:** E≤8 次标量 load 全部 L2 命中，向量化引入 gather + reduce 反而更慢。

### §5.2 Layer 2: 算法骨架

#### L2.1 双 kernel 拆分（K1 Cube / K2 AIV）

```
K1 gmm_dequant_persistent_kernel（Cube, persistent 单 launch）:
  grid = min(cube_cores, total_tiles)，tile 跨 expert 全局连续编号
  for tile_idx in range(pid, num_tiles, NUM_CORES):
      e = constexpr 前缀扫描定位 expert（L1.8）
      w_ptr = w_base + w_delta[e]（元素单位）
      local_tile → (bm, bn) → m_start = row_lo[e] + bm*BLOCK_M
      k 循环: int8 tl.dot(a, b, acc_i32)（BLOCK_K by K 整除性）
      epilogue: 两级 dequant + fp16 舍入（L1.7）→ store y(fp16)

K2 swiglu_dquant_kernel（AIV, 行循环）:
  grid = min(M, 4*vec_cores)，行交织 for row_idx in range(pid, M, num_programs)
  load gate/up(fp16) → sigmoid 链（f32）→ sw → fp16 → f32
  → 行 max_abs → scale = 127/max(mabs, 1e-10)
  → round-half-away-from-zero + clamp[-128,127] → store int8 + f32 scale
```

**语义要点（文中不展开代码，但写错任一条 verify 必挂）**：

1. **K1 行界 mask 用 `row_hi[e]` 而非全局 M** —— grouped 下 m_e 非 BLOCK_M 倍数时，用 M 会越过当前 expert 读到下一 expert 的行；`x` / `xs` 的 load 与 `y` 的 store 三处 mask 同源。
2. **K1 dot 必须原生 int8 操作数 + int32 acc** —— 禁止先 `.to(f16/f32)` 再 dot（改变累加语义且损失吞吐）；仅 weight 非 int8 的备用分支（`W_IS_I32`）才转 f32。
3. **K2 gate/up 拆分方向是语义** —— gate 取前半 `[:, :HALF_N]`、up 取后半 `[:, HALF_N:]`；搞反则 sig×up 颠倒必挂。两段 load 各自按半宽 mask。
   **sw 公式 = `sigmoid(gate) × up`，不含 gate 因子** —— 标杆 `npu_swiglu(dim=-1)` 的实证口径是 σ(前半)×后半，**不是**通识 SwiGLU 的 silu 变体 `gate·σ(gate)·up`；多乘 gate 会 60/60 全挂（sw 分布变 → per-row scale 全变，|diff|≤1 容差下 81% 元素超限、max_abs_diff=127 满量程）。2026-08-22 从零生成首版实测踩中。
4. **K2 BLOCK_N 按 `next_power_of_2(HALF_N)` 取**，不沿用 K1 的 128/256 —— HALF_N 可达 2048，行内单 tile 覆盖全半宽。
5. **out_s 存的是 `max_abs / 127`（scale 的倒数）**，不是 scale 本身 —— 与标杆反量化约定一致，见 L3.3。
6. **dequant 广播方向** —— `ws [N]` 按 N 列广播、`xs [M]` 按 M 行广播，方向写反是维度错误。

#### L2.2 Host 侧 BLOCK 门控分派表

| 条件 | BLOCK_M | BLOCK_N | BLOCK_K | 说明 |
|------|---------|---------|---------|------|
| N % 256 == 0（默认大 N） | 64 / 32 / 16 by avg_m≥48/32 | **256** | 256 / 128 / 64 by K%256/%128 | 主路径，UB 160KB |
| N % 256 != 0 | 64 / 32 / 16 by avg_m≥48/32 | 128 | 同上 | 次路径 |

- `avg_m = M // E`（行均摊）；BLOCK_M 按 avg_m 分档避免小 expert padding 浪费；BM=64 需 `E*ceil(avg_m/64)*num_tiles_n ≥ cube_cores`（24 core 满负载守卫，小 N shape 退回 BM=32 防并行度损失）。
- ⛔ 已证伪：`avg_m≥128 时 BM=128/BN=128` 方形 tile（见 §5.4 失败表）。

#### L2.3 元数据构建（host，每 forward 一次）

1. `gl_cpu = group_list.cpu().tolist()`（唯一 D2H，L1.2）；
2. 遍历 E 个 expert：空 expert（m_e==0）剔除，非空的累积 `w_delta`（元素单位，L1.4）/ `wsd_delta` / `row_lo` / `row_hi` / `tile_lo`（`tiles_e = ceil(m_e/BM) * num_tiles_n`）；
3. `meta = w_deltas + wsd_deltas + tile_lo + row_lo_l + row_hi_l` → 单 tensor H2D（L1.3）。

### §5.3 Layer 3: 关键技巧（Agent 可参考但不可复制代码结构）

#### L3.1 ★★★ persistent tile 循环 + constexpr 前缀扫描 expert 定位

```python
# E_C: tl.constexpr（剔空后的 expert 数）→ 编译期完全展开，int32 比较
for tile_idx in range(pid, num_tiles, NUM_CORES):
    e = 0
    for i in range(E_C - 1):
        bound = tl.load(tile_lo_ptr + i + 1).to(tl.int32)
        e += (tile_idx >= bound).to(tl.int32)
    w_ptr = w_base_ptr + tl.load(w_delta_ptr + e)   # 元素单位 delta
    lo_e = tl.load(tile_lo_ptr + e).to(tl.int32)
    local_tile = tile_idx - lo_e
    bm = local_tile // num_tiles_n
    m_start = tl.load(row_lo_ptr + e).to(tl.int32) + bm * BLOCK_M
```

**可替代方向**: 按行分组（row_hi 定界）也可定位，但 tile 粒度才能配合 BLOCK_M 分档；向量化前缀扫描已证伪（L1.8）。

#### L3.2 meta 打包 + 单次 H2D / 单次 D2H

```python
gl_cpu = group_list.cpu().tolist()        # 唯一一次同步（替代 E 次 .item()）
meta_t = torch.tensor(
    w_deltas + wsd_deltas + tile_lo + row_lo_l + row_hi_l,
    dtype=torch.int64, device=device)     # 唯一一次小 H2D
```

**注意**: M==0 边界（全部 expert 空）时 meta_t 仍要构建（K2 行循环零迭代，但 kernel 签名需要指针）。

#### L3.3 K2 标量 scale store 的 broadcast trick

**问题**: 每行输出一个 f32 scale，标量 `tl.store` 会降级。

**解决**:
```python
r1 = tl.arange(0, 1)
tl.store(out_s_ptr + row_idx + r1, (max_abs / 127.0) + r1 * 0.0)
```

**可替代方向**: 把 scale 打包进 int8 输出尾部的融合布局——会改变输出语义，禁止。

### §5.4 性能基准与优化尝试记录

| 版本 | cases | geomean | 相对上版 | 说明 |
|------|-------|---------|---------|------|
| Phase 3 基线（per-expert launch + 逐次 .item()） | 60/60 | 0.7978x | — | 双 kernel，E 次同步调度 |
| **opt_iter_0（#12 persistent + meta 打包）** | 60/60 | **1.0965x** | +37.4% | 头号杠杆（L1.1-L1.3） |
| opt_iter_1（#1 constexpr mask 折叠） | 60/60 | 1.0647x | ↓ | 无增量，回退 |
| **opt_iter_2（#2 BLOCK_K=256）** | 60/60 | **1.1516x** | +5.0% | 当时 best（L1.6 UB 核算） |
| opt_iter_3（#5 expert 定位向量化） | 60/60 | 0.9059x | ↓↓ | 回退（L1.8 证伪依据） |
| opt_iter_4（#13 autotune） | 60/60 | 1.0840x | ↓ | 回退 |
| opt_iter_5（#29 K1/K2 CV 融合） | 60/60 | 0.5596x | ↓↓↓ | 回退 |
| opt_iter_6（#31 de-mix：int32 acc 直存，dequant 挪 K2） | 60/60 | 1.0495x | ↓ | bit 级等价但劣化，见失败表 |
| opt_iter_7_ir_2（#31 IR 轮 2：K2 分析） | — | — | — | 无新建议，IR 耗尽 |
| opt_iter_8（#2 变体 BM=128/BN=128 方形 tile） | 60/60 | 0.9301x | ↓ | 追加轮证伪，见失败表 |
| **opt_iter_9（#11 Load 重排序：weight load 先于 x load 发射）** | 60/60 | **1.2235x** | ↑ | 掩盖 L1→L0 2D transpose 慢搬运链延迟，搬运瓶颈 case 38 0.529→0.605 |
| opt_iter_10（框架外 weight-stationary N_M_SUB=2 双 kernel） | 60/60 | 0.8604x | ↓ | 修复 constexpr 假分支竞态后正确，但 B 搬运无实际节省 + int8 fractal 利用下降，见失败表 |
| **opt_iter_11（#2 下探 BLOCK_M=64 到 avg_m≥48 带）** | 60/60 | **1.3202x** | ↑ | **终态 best**：tile 数≥cube_cores 守卫下减半中型 shape weight 搬运量（case 48/46/45/58/39 -9~36%），0 回归超 10% |
| opt_iter_12（框架外 tile 顺序 m_group 主序 → weight L2 复用） | 60/60 | 0.9770x | ↓ | 破坏 x 复用 + weight 超 L2 不命中 + MTE2 ND2NZ 粒度下降，见失败表 |
| opt_iter_13（KKB exp_perf_012：存活 mask i32→fp32 比较转换） | 60/60 | 不可用 | ↓↓↓ | K1 fp32 行 mask aicore 挂死；K2 fp32 half mask 非 2 幂 case 14× 变慢，见失败表 |

**有效优化点**:
1. **persistent 单 launch + meta int64 打包单次 H2D + group_list 单次 D2H**（0.7978→1.0965，+37.4%）
2. **BLOCK_K=256 tiling**（K%256==0 时 k 循环迭代数减半，→1.1516，+5.0%）
3. K2 标量 scale store broadcast trick（L3.3，随主版本带入）
4. **#11 Load 重排序（weight load 提前发射）**（→1.2235，+6.2%）：weight 搬运链是唯一主导 pipe（MTE1+MTE2=68.1%），把慢搬运 load 提到 x load 前发射掩盖延迟，数值 bit 级等价
5. **#2 下探 BLOCK_M=64（avg_m≥48 带 + tile 数≥cube_cores 守卫）**（→1.3202，+7.9%）：减半中型 shape 的 weight 搬运量；守卫防小 N shape 并行度损失

**已验证无效/失败的优化方向**:

| 优化方向 | 结果 | 原因 |
|----------|------|------|
| constexpr mask 折叠（#1） | 1.0647x ↓ | mask 已被编译器处理，手动折叠无增量 |
| expert 定位向量化（bounds gather + tl.sum） | 0.9059x ↓↓ | E≤8 标量 load L2 命中；向量化引入 gather+reduce 更慢 |
| autotune（#13） | 1.0840x ↓ | 编译/搜索开销吃掉收益，60 case 多 shape 场景尤甚 |
| K1/K2 CV 融合单 kernel（#29） | 0.5596x ↓↓↓ | Cube persistent tile 循环与 AIV 行循环粒度不匹配，互相拖累 |
| K1 de-mix（int32 acc 直存去 lock-step，dequant 链挪 K2，#31） | 1.0495x ↓ | bit 级等价仍劣化：IR 上 mix lock-step（每 tile 两次跨核握手 + 64KB workspace）真实代价仅 FLOWCTRL 2.3%（simulator 证实），编译器 mix 流水已有效重叠 AIC/AIV；而 K2 读 int32(4B/elt) 比读 fp16(2B/elt) 带宽翻倍 + 新增 dequant 计算，大 case 劣化 5~17%。**IR 结构性瓶颈必须配分段流量账核算（GM 总量账 ≠ 分段带宽账）** |
| BM=128/BN=128 方形 tile（avg_m≥128 门控） | 0.9301x ↓ | 纸面装载/计算比 -20% + weight 流量减半，实测 58/60 case 劣化：int8 fractal 装载下 BN 256→128 使 B 的 N 向装载趟数翻倍（碎片/启动开销升），ND2NZ 大块搬运更高效，avg_m≥128 不保证每 expert m_e≥128（padding 浪费）。**纸面矩形 tile 推演必须经 fractal 对齐与 padding 校正** |
| BM=128/BN=256（增大 tile） | 编译硬失败 | L0C 32768 元素档溢出（L1.5） |
| BLOCK_K=512 | 编译硬失败 | UB 192KB 溢出（L1.6） |
| weight-stationary（M 方向 weight 复用，N_M_SUB=2 双 kernel） | 0.8604x ↓↓ | 修复 constexpr 假分支竞态后正确，但 M 复用权重搬运无实际节省 + 每 program 变稀疏的 int8 fractal 装载效率下降，B 搬运仍是瓶颈 |
| tile 遍历 m_group 主序（相邻 program 共享 weight 走 L2） | 0.9770x ↓ | 破坏原 n_block 主序的 x 复用（x 流量与 weight 同量级）；weight 470MB 远超 L2，grid-stride 使 program 跨 bn 处理不连续列 → MTE2 ND2NZ 大块转换效率下降（同 opt_iter_8 BN 粒度教训） |
| **i32→fp32 存活 mask 转换（KKB exp_perf_012，Cube row_hi mask）** | aicore 挂死 | triton_ascend 后端对 fp32 派生 load/store mask 生成损坏 MTE 任务描述符：K1 Cube kernel 永不完成（507014 watchdog），进程死后 aicore 仍 99% 空转卡死整块 device |
| **i32→fp32 存活 mask 转换（KKB exp_perf_012，AIV half-width mask）** | 14× 变慢 | K2 非 2 幂 HALF_N case（13/60）speedup 塌到 0.04-0.07（K2 单次 forward ~4s）；i32 比较虽"标量降级"却是后端处理良好的路径。结论：fp32 mask 转换在本算子双向失效，勿在 mask 上使用 exp_perf_012 |

**关键结论**:
1. **host 调度是 grouped 变长分组的第一瓶颈**：per-expert launch + 逐次 D2H 使 speedup 随 E 单调恶化（E=8 时 0.472x）；persistent + meta 打包一步修复（+37.4%）。
2. **修复后瓶颈立即转移到 weight 搬运链**：simulator 实测（真实配置 BN256）K1 Cube 核 **MTE1 44.2%（L1→L0 2D transpose）+ MTE2 23.9%（GM→L1 ND2NZ）= 68.1%**，MMAD 仅 6.4%（三组 shape 稳定 5.5~6.4%）。非计算 bound，但 #7/#21/#10 核对不适用（单 pass GEMM / 无重复 gather / 无循环不变量），增大 tile 与 bf16 化不可行 → 属 int8 grouped GEMM 结构性搬运代价。**终态 1.3202x（opt_iter_11）已达理论上限 1.3995 的 94.4%**；向量化/autotune/融合/de-mix/方形 tile/weight-stationary/tile 序/fp32 mask 全部证伪。**后续同类算子不要按"计算/调度"方向继续硬攻**。
3. **IR 结构瓶颈必须配分段流量账**：见失败表 de-mix 行。simulator 小 shape 会高估 SCALAR / 低估搬运（K64 时 SCALAR 60.2%，K1024 真实配置下 21%）——诊断必须用接近线上的 shape + BLOCK 配置扫描。
4. **§4 的解耦/N-strip 经验对本算子不适用**：dequant 已在 K1 epilogue 单次执行（无跨 M-tile 重复 dequant 问题）；workspace/N-strip 与 expert 变长分组边界冲突，K1/K2 融合类方向全部实测劣化。
5. **benchmark 环境污染判定法（共享 NPU 服务器必读）**：framework（sha256 冻结的 PyTorch 参考）延迟漂移 >50% 即为争用铁证（framework 代码不变，波动只能来自环境）；impl 绝对延迟 per-case 对比不受 framework 噪声影响，可作交叉验证。被污染的 run 必须重测，不可用修正值充当官方结果。
6. **数值链 bit 级对齐是 60/60 的前提**：两级 fp16 舍入点 + 手写 sigmoid + round-half-away-from-zero + int32 clip（与 §3 同源）；任何"等价改写"（libdevice sigmoid、移动舍入点）都会破坏对齐。
7. **fp32 mask 转换（exp_perf_012）在本算子双向失效**：Cube load/store mask 转 fp32 → aicore 挂死（损坏 MTE 任务描述符，kernel 永不完成，507014 watchdog，进程死后 aicore 仍 99% 空转卡死整块 device）；AIV mask 转 fp32 → 非 2 幂 case 14× 变慢（speedup 0.04-0.07）。i32 比较虽"标量降级"却是后端处理良好的路径。**该 KKB 技术仅适用于纯向量算术路径，不适用于 mask 生成**。
8. **挂死 kernel 的隔离方法论**：verify 60/60 全灭且报 `aicore timeout 507014` 时，先隔离 impl 单 kernel 复现（impl forward → framework forward → sync 的 verify 忠实驱动）；挂死是 Cube 还是 AIV 用单点变体二分（`iso/k1only.py` / `iso/k2only.py`）。挂死会污染 device 整块卡死（进程死但 aicore 空转），**必须换空闲 device（`ASCEND_RT_VISIBLE_DEVICES=<id>`，verify.py 固定 `torch.device("npu")` 取可见首卡）重测**，否则后续所有 benchmark 全灭误判为代码 bug。

---

## §6 常见陷阱与避免方法

### §6.1 DynamicQuant 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 期望优化大 K 场景超越 torch | 在 K > 8192 场景反复尝试 tile 优化、循环重排，期望超越 torch | 这是架构限制 (L1.1)，2-pass HBM 无法避免；直接接受 0.2x~0.3x，把精力放在小 K 优化 |
| 用 L2 cache 期望跨 pass 数据复用 | 单 kernel 两循环期望 L2 cache 保留 pass1 数据 | 已验证无效 (L1.2)，L2 cache 不保留跨 pass 数据 |
| 用非对称 BLOCK_K 优化 2-pass | pass1 用大 BLOCK_K、pass2 用小 BLOCK_K | 已验证无效 (L1.3)，瓶颈是 HBM 带宽而非中间量寄存器压力 |
| 用 bf16 max_abs 避免 fp32 膨胀 | 在 bf16 精度下计算 max_abs | 已验证无效 (L1.4)，max_abs 仍需转 fp32 累积 |
| 用 inv_scale 乘法替代除法 | `x * (1/scale)` 替代 `x / scale` | 已验证反而回退 (L1.5)，乘除延迟相近 |
| 用 `tl.round` 导致 verify 失败 | `tl.round` 默认 round-half-to-even，与 torch 的 round-half-away-from-zero 不一致 | 用 `tl.where(q >= 0, q + 0.5, q - 0.5)` + `tl.cast` (L1.10) |
| 手动 clamp 到 [-128, 127] | `tl.maximum(tl.minimum(q, 127), -128)` 生成多条指令 | `tl.cast(overflow_mode='saturate')` 硬件原生饱和转换 (L1.9) |
| 大 K 场景 BLOCK_M > 1 导致 UB 溢出 | 2D tile `BLOCK_M * BLOCK_K * sizeof(fp32)` 超 192KB UB | `_choose_block_size` 限制 `block_m = max(1, min(32, 4096 // block_n))` (L1.6 / L3.4) |
| 误把 DynamicQuant 当纯 reduction 优化 | 按 reduction 优化思路尝试向量化、tiling | DynamicQuant 瓶颈是跨 pass 数据复用 (2-pass HBM)；reduction 优化对大 K 无效 |
| 试图逆向工程 CCE 单遍融合 kernel | 试图用 Triton 复现 torch CCE 的 UB 缓存 + 跨核同步 | Triton DSL 不支持这些 CCE 级原语 (L1.7)，直接判定为架构限制 |
| profiler 失败误判为代码错误 | 4 个 case 报 `RuntimeError: 无法从 profiler 提取有效时延数据` | 这是 profiling 基础设施问题，verify 阶段 42/42 全部通过 |
| 忽略 smooth_scales 分支 | 只写一个 kernel，用 `if smooth_ptr is None` 分支，导致 kernel 内有条件分支 | 分两个 kernel (`_fused_2d_kernel` 和 `_no_smooth`)，host 侧分派 (L2.1) |
| 3D 输入处理不当 | 3D 输入 `[B, M, K]` 直接按 2D 处理，scale shape 错误 | `rows = b * m`，scale 输出为 `[B, M]` shape (L3.5) |

### §6.2 SwigluQuant 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 使用 `tl.sigmoid` 导致精度系统性 fail | `tl.sigmoid` 在 Ascend 后端产生系统性数值偏差 | 始终使用 `x / (1.0 + tl.exp(-x))` (L1.4) |
| dynamic 模式标量除法 ULP 漂移 | `int_scale / max_abs` 在标量路径下与 torch 的 `aclnnDiv` 偏差 1-2 ULP，经 `nearbyint` 后 off-by-1 | 先存 `max_abs`，tile 级别以向量 load 再做除法 (L1.6 / L3.3) |
| 对 int32 使用 `tl.clamp` | 报 `Only floating point clamp is supported` | 用 `tl.maximum/tl.minimum` 在 int32 阶段做 clip (L1.7) |
| INT4 打包顺序错误 | `quint4x2` 期望低位为 even、高位为 odd，顺序错误会导致量化结果整体错位 | 严格按 `((q_odd & 0x0F) << 4) \| (q_even & 0x0F)` 打包 (L1.5 / L3.4) |
| 2D smooth/offset 的边界合并导致 UB overflow | 将 `mask & in_bounds` 合并到 `tl.load` 可减少一次 `tl.where`，但会增大 fp32 张量 live range，在大 shape 上 UB overflow | 2D 路径保持 `tl.where(in_bounds, swiglu * scale_block, swiglu)`；仅标量路径可合并 (L3.2) |
| 硬编码 `num_cores` | 不同 NPU 核数不同 | 动态读取 `triton.runtime.driver.active.utils.get_device_properties(device).get("num_vectorcore", 48)` (Q1) |
| offsets 在 dynamic 模式下误启用 | dynamic 模式没有 offsets 语义，启用会导致参考实现不一致 | host 侧强制 `has_offsets = offsets is not None and quant_mode == 0` (L1.11) |
| 把 int32 clip 改为 fp32 clip | 增大 fp32 中间结果 live range，触发 UB overflow (case 27) | 保持 int32 阶段 `tl.maximum/tl.minimum` clip，避免 fp32 中间结果长期保留 |
| 把 dynamic `BLOCK_ROWS` 从 4 降到 2 | 劣化，0.2434 ms > 0.2382 ms | `BLOCK_ROWS=4` 已接近最优，autotune `{2,4,6}` 收益有限 |

### §6.3 GroupedMatmulSwigluQuant 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| per-expert 循环 launch / 逐 expert `.item()` | E 次隐式同步与调度，speedup 随 E 单调恶化（E=8 时 0.472x） | persistent 单 launch + group_list 单次 `.cpu().tolist()` + meta 单次 H2D (L1.1-L1.3) |
| 指针差按字节传 meta | kernel 内指针算术以元素为单位，int8 与 fp32 element_size 不同 | host 侧 `(ptr_a - ptr_b) // element_size()` 换算 (L1.4) |
| meta 段偏移漏掉 tile_lo 的 +1 | tile_lo 比 E 多一个末位哨兵，`row_lo`/`row_hi` 段首偏移是 `3E+1`/`4E+1` 而非 `3E`/`4E` | 按 L1.3 布局字符串逐段推偏移，错一位全盘访存错位 |
| K1 行界 mask 误用全局 M | grouped 下尾部 tile 会越过当前 expert 读到下一 expert 的行 | mask 用 `row_hi[e]`，x/xs load 与 y store 三处同源 (§5.2 语义要点 1) |
| K2 gate/up 搞反（前半当 up） | 拆分方向是算子语义，sig×up 颠倒后数值全错 | gate=前半、up=后半，各自按半宽 mask (§5.2 语义要点 3) |
| sw 写成通识 SwiGLU `gate·σ(gate)·up`（silu 变体） | 标杆 `npu_swiglu` 实证口径是 σ(gate)×up，多乘 gate 使 sw 分布与 per-row scale 全变 | 公式冻结为 `sigmoid(gate) × up`（无 gate 因子），60 case 无一幸免、max_abs_diff=127 (§5.2 语义要点 3) |
| K2 BLOCK_N 沿用 K1 的 128/256 | HALF_N 可达 2048，固定小块导致行内多趟或 mask 越界 | `next_power_of_2(HALF_N)` 单 tile 覆盖全半宽 (§5.2 语义要点 4) |
| BM×BN 顶到 32768 档（如 128×256） | L0C 128KB 按元素数分档，32768×4B 恰好顶满被拒 | 保持 16384 档内（64×256 或 128×128 acc 均为 64KB）(L1.5) |
| BK=512 双缓冲溢出 | 2*(BM*BK+BK*BN) 超 192KB UB | BLOCK_K≤256 并核算 UB 预算 (L1.6) |
| expert 定位段向量化 | E≤8 次标量 load 全 L2 命中，向量化引入 gather+reduce 反而慢（0.9059x） | 保持 constexpr 标量前缀扫描 (L1.8) |
| K1/K2 融合或 de-mix 改写 | Cube tile 循环与 AIV 行循环粒度不匹配；mix lock-step 真实代价仅 2.3%，去 mix 省的同步抵不过中间缓冲 dtype 翻倍的带宽 | 保持双 kernel + fp16 GM 中转；IR 结构改动前先算分段流量账 (§5.4) |
| BM=128/BN=128 方形 tile（纸面装载比更优） | int8 fractal 装载下 BN 减半使 B 装载趟数翻倍；avg_m 门控挡不住 m_e 不均的 padding 浪费（0.9301x） | 保持 BN=256 宽 tile + BM≤64 分档 (L2.2) |
| libdevice sigmoid / 移动 fp16 舍入点 | 硬件 sigmoid 舍入行为不同；舍入点位置变化破坏 bit 级对齐 | 数值链冻结 (L1.7，同 §3) |
| 把共享服务器 benchmark 的病态 case 当代码 bug | 环境争用使 1-5 个 case 病态（framework 也同步漂移） | framework（冻结参考）漂移 >50% 即判污染，重测；impl 绝对延迟交叉验证 (§5.4 结论 5) |
| 存活 mask 转 fp32（i32→fp32 比较） | Cube mask aicore 挂死 / AIV mask 14× 变慢 | mask 保持 i32（后端良好路径）；exp_perf_012 仅限纯向量算术 (§5.4 结论 7) |
| device 挂死后继续在同一 device 上 benchmark | 挂死 kernel 污染整块 device（aicore 99% 空转），后续全灭误判代码 bug | 换空闲 device（`ASCEND_RT_VISIBLE_DEVICES=<id>`）重测 (§5.4 结论 8) |

---

## §7 GroupedMatmulSwigluQuantV2 算子（grouped matmul + SwiGLU + quant）

**算子类别**: `grouped matmul` + `activation` + `quantization`
**典型特征**: `x [M,K] int8` 按 `group_list`（cumsum=type 0 / count=type 1）切分为 E 个 expert 行区间，各 expert 配 `weight [E,K,N] int8` + per-channel weight_scale；matmul 后 SwiGLU（SiLU gate）再 per-token int8 量化。`dequant_mode=0/1` 框架实际均为 per-channel weight scaling，但 **mode=1 必须走 fp16 预缩放 dot 路径**才能对齐 NPU 数值（int8 dot + fp32 后缩放会产生 off-by-N 量化误差，见 general_debug/exp_debug_002_20260825）；`quant_mode=0/1` 框架行为一致均为 per-token scale。
**性能基准**: 50/50 verify 通过；几何平均 **0.7738x** vs torch_npu（19 case ≥1.0x，最高 1.68x）；优化过程 implementation 平均延迟 **0.4129 → 0.0459 ms（约 9x）**
**详细版**: `.claude/skills/triton-latency-optimizer/references/operators/grouped-matmul-tile-merge.md`（含完整骨架/坑表/优化点映射）

### §7.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 行 tile 必须按 expert 边界对齐，禁止跨界
per-expert 行区间 `[start_e, end_e)` 内切 BLOCK_M tile，`mask_m = offs_m < end_e`；kernel 内标量 E 循环（found-flag + `tl.where(hit,...)`)定位 expert_id/row_base/end_sel/m_local。越界行 x_scale 用 `other=1.0`，workspace 与 partial max **必须同样 mask**，否则以错误 expert 的垃圾值污染下一 expert 的行。

#### L1.2 禁止 host 读 group_list（D2H 禁令）
tile 规划只用 M/K/N/E 元数据；grid 用上界 `(cdiv(M,BM)+E)*num_n_tiles`，多余 program 由空 stride 循环跳过。

#### L1.3 dot kernel grid ≤ num_aicore（910b1=20），纯 vector kernel ≤ num_vectorcore（40）
含 `tl.dot` 的 mix kernel 用 aicore 数；核数查询按 device index 缓存到 `ModelNew._core_cache`。

#### L1.4 dot 的 M 维必须 ≥16（核心，latency-bound 修复）
BLOCK_M ∈ {16,32,64} 自适应（取最大 BM 使 `(M//BM)*num_n_tiles >= num_aicore`，链式三元表达式实现，**禁 while**）；连续 `[BLOCK_M, BLOCK_N]` 单 tile。BLOCK_M=1 逐行是 0.079x 的根因；BM 上限 64（UB：acc 2×64×128×4=64KB 主导，峰值 ~180KB，BM=128 必溢出）。

#### L1.5 禁 atomic_max，row_max 用 per-n_tile partial + Pass2 归约
partial 布局 `[num_n_tiles, M]`；归约是顺序无关 max，数值 bit-exact。

#### L1.6 fp16 预缩放路径（dequant_mode=1）BLOCK_K 锁死 256
该路径需物化 w fp16 tile（BK×BN×2B），BK=512 时仅 w_fp16=128KB 必溢出 UB；int8 dot 路径（mode=0）BK 可到 512（dot 数减半，+12% geomean）。host 按 `dequant_mode`（python 标量参数）分档。

#### L1.7 多 case verify 三件套
verify_dir 内 `{op}_torch.py` + `{op}_triton_<impl>.py` + `{op}.json` 同名共存。

### §7.2 Layer 2: 算法骨架（要点）

1. **Pass1**：扁平 tile 空间 `(m_tile, n_tile)`，`m_tile = flat // NUM_N_TILES`、`n_tile = flat - m_tile * NUM_N_TILES`（禁 `%`）；stride 循环 `range(pid, total_flat, NUM_CORES)`；每 program 处理同一 expert 内连续 BLOCK_M 行 × 单 n_tile，K 循环左右半各一次 dot，SwiGLU 后写 fp32 workspace + partial max。
2. **Pass2**：行块循环，先标量循环归约 partial max，再 `inv_scale` 向量量化（round-half-away-from-zero + int32 clip + int8）。
3. **手写 sigmoid**：`gate = acc_left / (1.0 + tl.exp(-acc_left))`，禁 `tl.sigmoid`。

### §7.3 Layer 3: 关键技巧

- **BLOCK_K 分档**：`512 if (dequant_mode == 0 and K % 512 == 0) else 256`
- **count 模式边界累计**：`end_e = boundary + load(offsets+e)`，与 cumsum 模式统一在同一 E 循环（`group_list_type` 作 constexpr 分支）
- **冒烟测试先行**：改 tiling 后先用 5 个代表 case（最大 K mode-0/mode-1、count 模式、最小 M、最大 E）验 int8 maxdiff ≤ 1，再进全量；粗略 python 计时被 host 开销淹没，不可用于判优
- **收益递减判据**：BM 轴 +599%→+19%→+4.4%，BK 轴 +12%；轴收益 <5% 或被 UB 硬锁时停止

### §7.4 性能历程与陷阱

| 版本 | 改动 | geomean vs torch_npu |
|------|------|---------------------|
| 初始 | BLOCK_M=1 行循环 | 0.0793x |
| opt1 | expert 对齐 BM=16 + 扁平并行 + partial max + grid 修正 | 0.5539x |
| opt2/opt3 | BM 32→64 | 0.6600x / 0.6889x |
| opt4 | mode-0 BK 256→512 | **0.7738x** |

| 陷阱 | 修复 |
|------|------|
| BLOCK_M=1 行循环（0.079x） | 行维并入 dot M 维（L1.4） |
| tile 跨 expert 边界 | per-expert 对齐 + end_sel mask（L1.1） |
| atomic_max | partial + 二阶段归约（L1.5） |
| fp16 预缩放路径硬上 BK=512 | UB 锁死 BK=256（L1.6），差距属结构性 |
| forward 内 while 自适应 | AST 校验 Type-3 拦截，改链式三元 |
| kernel 内 `%` | `flat - m_tile * NUM_N_TILES` |
| mix kernel grid 用 vectorcore | dot kernel 用 num_aicore（L1.3） |

**关键结论**:
1. grouped matmul 的首要瓶颈是"行循环锁死 dot M 维=1 + weight 整份重复加载"，expert 边界对齐的 BLOCK_M 合并是 9x 级修复。
2. mode=1（fp16 预缩放）与 mode=0 的 BLOCK_K 上限不同，本质是按数值路径分裂（优化点 18 思想）。
3. 剩余差距（偶数 case 全部 mode=1 大 K）是 fp16 物化副本 vs torch_npu 专用融合 kernel 的结构性差距，不要再调参。
