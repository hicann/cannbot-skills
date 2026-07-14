---
name: quantization
description: 量化类算子（DynamicQuant / SwigluQuant）的 Triton Ascend 优化经验合集，按算子分章节组织，含通用经验 + 各算子专属约束/骨架/kernel
metadata:
  type: reference
---

# 量化类算子优化经验

本文档合并了两类量化算子的优化经验。按以下结构组织：
- **§1 通用经验**：量化类算子跨算子重复的工程约束（已提取，各算子章节不再重复）
- **§2 DynamicQuant**（reduction + element-wise 复合，per-token 动态量化）
- **§3 SwigluQuant**（quantization-activation，SwiGLU 激活后量化）
- **§4 各算子常见陷阱**

---

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| DynamicQuant | `reduction` + `element-wise` 复合 | per-row max_abs reduction + 量化，2-pass HBM（大 K）或 single-pass fused（小 K） | 自适应 BLOCK 选择 + 单/双 pass 分派；大 K 接受架构限制 |
| SwigluQuant | `quantization-activation` | 对 SwiGLU 激活结果做逐行静态/动态 INT8/INT4 量化，支持 smooth scales、offsets、group index | static/dynamic 双 kernel 分派 + vector scale division + INT4 打包 |

> ⚠️ **关键区分**：DynamicQuant 的瓶颈在 **memory-bound + 跨 pass 数据复用**（大 K 时 2-pass HBM 不可优化），SwigluQuant 的瓶颈在 **UB 压力 + 精度对齐**（dynamic 模式必须 vector scale division，sigmoid 必须手写）。两类优化哲学不同，生成时**禁止混用经验**：
> - 生成 DynamicQuant 时，**不要**套用 SwigluQuant 的 activation 打包技巧
> - 生成 SwigluQuant 时，**不要**套用 DynamicQuant 的 2-pass HBM 失败方向

> ⚠️ **架构限制特殊说明（破例归档）**：DynamicQuant 几何平均加速比仅 **0.3604x**，远低于归档阈值 1.0x 和用户目标 0.6x。**破例归档**作为"Triton DSL 架构限制典型案例"——大 K 场景必然 2-pass HBM，与 torch 单遍融合 kernel 存在结构性带宽劣势，是**已知不可优化**的场景。后续做 per-group quant / weight quant 时，若大 K 慢于 torch，应首先排查 2-pass HBM。

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下约束在两个量化算子中均适用，各算子章节不再重复。

### Q1 动态读取 Vector/Cube Core 数量，禁止硬编码
- **必须**动态读取实际核数，禁止硬编码 `num_cores=8` 或 `num_cores=48`。
- **正确做法**（一次拿 vector + cube，权威值，无需设备 init）：
  ```python
  import torch_npu
  import triton.runtime.driver as driver

  device = torch_npu.npu.current_device()
  properties = driver.active.utils.get_device_properties(device)
  vectorcore_num = properties["num_vectorcore"]   # elementwise / reduction 用它做 grid 钳制
  aicore_num = properties["num_aicore"]           # cube / matmul 用它
  ```
- **已弃用**：旧式 `npu_config` 取值方式（仅 vector、硬编码 40 不准、设备未 init 会抛 `RuntimeError`）。
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
  import torch_npu
  import triton.runtime.driver as driver

  VEC_CORE_NUM = driver.active.utils.get_device_properties(torch_npu.npu.current_device())["num_vectorcore"]
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

## §4 常见陷阱与避免方法

### §4.1 DynamicQuant 陷阱

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

### §4.2 SwigluQuant 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 使用 `tl.sigmoid` 导致精度系统性 fail | `tl.sigmoid` 在 Ascend 后端产生系统性数值偏差 | 始终使用 `x / (1.0 + tl.exp(-x))` (L1.4) |
| dynamic 模式标量除法 ULP 漂移 | `int_scale / max_abs` 在标量路径下与 torch 的 `aclnnDiv` 偏差 1-2 ULP，经 `nearbyint` 后 off-by-1 | 先存 `max_abs`，tile 级别以向量 load 再做除法 (L1.6 / L3.3) |
| 对 int32 使用 `tl.clamp` | 报 `Only floating point clamp is supported` | 用 `tl.maximum/tl.minimum` 在 int32 阶段做 clip (L1.7) |
| INT4 打包顺序错误 | `quint4x2` 期望低位为 even、高位为 odd，顺序错误会导致量化结果整体错位 | 严格按 `((q_odd & 0x0F) << 4) \| (q_even & 0x0F)` 打包 (L1.5 / L3.4) |
| 2D smooth/offset 的边界合并导致 UB overflow | 将 `mask & in_bounds` 合并到 `tl.load` 可减少一次 `tl.where`，但会增大 fp32 张量 live range，在大 shape 上 UB overflow | 2D 路径保持 `tl.where(in_bounds, swiglu * scale_block, swiglu)`；仅标量路径可合并 (L3.2) |
| 硬编码 `num_cores` | 不同 NPU 核数不同 | 动态读取 `driver.active.utils.get_device_properties(device)["num_vectorcore"]` (Q1) |
| offsets 在 dynamic 模式下误启用 | dynamic 模式没有 offsets 语义，启用会导致参考实现不一致 | host 侧强制 `has_offsets = offsets is not None and quant_mode == 0` (L1.11) |
| 把 int32 clip 改为 fp32 clip | 增大 fp32 中间结果 live range，触发 UB overflow (case 27) | 保持 int32 阶段 `tl.maximum/tl.minimum` clip，避免 fp32 中间结果长期保留 |
| 把 dynamic `BLOCK_ROWS` 从 4 降到 2 | 劣化，0.2434 ms > 0.2382 ms | `BLOCK_ROWS=4` 已接近最优，autotune `{2,4,6}` 收益有限 |
