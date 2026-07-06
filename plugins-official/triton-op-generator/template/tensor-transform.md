---
name: tensor-transform
description: 张量变换类算子（Interpolate / Pad / Repeat）的 Triton Ascend 优化经验合集，按算子分章节组织，含通用经验 + 各算子专属约束/骨架/kernel
metadata:
  type: reference
---

# 张量变换类算子优化经验

本文档合并了三类张量变换算子的优化经验。按以下结构组织：
- **§1 通用经验**：跨算子重复的工程约束（已提取，各算子章节不再重复）
- **§2 Interpolate**（transformation-compute，计算密集）
- **§3 Pad**（transformation-memory，搬运密集）
- **§4 Repeat**（transformation-memory，搬运密集）
- **§5 各算子常见陷阱**

---

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| Interpolate | `transformation-compute` | 数据重排伴随权重计算（nearest/bilinear/bicubic/area） | 多策略分派 + 精度匹配 + 采样特化 |
| Pad | `transformation-memory` | 数据搬运，计算极简（仅边界坐标映射） | fill+copy 拆分 + 维度压缩 + 边界映射 |
| Repeat | `transformation-memory` | 数据搬运，计算极简（仅坐标映射） | outer×inner 连续块 + constexpr 展开 |

> ⚠️ **关键区分**：Interpolate 属 **compute 类**（关心采样算法和权重精度），Pad/Repeat 属 **memory 类**（关心连续访存和坐标解码开销）。两类优化哲学相反，生成时**禁止混用经验**：
> - 生成 Interpolate 时，**不要**套用 Pad/Repeat 的搬运类技巧
> - 生成 Pad/Repeat 时，**不要**套用 Interpolate 的采样/权重类技巧

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下 8 条约束在三个算子中均适用，各算子章节不再重复。

### G1 动态读取 Vector Core 数量，禁止硬编码
- **必须**动态读取实际 Vector Core 数量，禁止硬编码 `num_cores=8` 或 `num_cores=48`。
- **正确做法**：
  ```python
  VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get('vector_core_num', 40)
  ```
  或通过 `triton.runtime.driver.active.utils.get_device_properties(device)` 读取 `num_vectorcore` / `num_aicore`。
- **Why:** 硬编码仅利用 20% Vector Core，导致加速比从 ~1.3x 跌至 0.67x（慢于 PyTorch）。

### G2 BLOCK 向上取 2 的幂（NPU 向量化友好）
- **必须**将 BLOCK / MAX_IN_W 等向量化长度声明为 `triton.next_power_of_2(N)`，且作为 `tl.constexpr` 传入 kernel。
- **Why:** Ascend 上 fixed-shape vector load 必须是编译期常量长度；若 `tl.arange(0, N)` 中 N 非 constexpr 会触发 dynamic-shape load，退化为标量循环。
- **典型 BLOCK 选择**：
  ```
  width <= 64    -> 64
  width <= 128   -> 128
  width <= 256   -> 256
  width <= 512   -> 512
  width <= 1024  -> 1024
  width <= 2048  -> 2048
  width <= 4096  -> 4096
  else           -> 8192
  ```

### G3 禁止单 kernel 统所有路径，必须多策略分派
- **必须**按 `(维度, 模式, 采样比例)` 等特征在 host 侧分派到特化 kernel。
- **禁止**用单一通用 kernel 统一处理所有路径（坐标解码 overhead 极大，且丢失特化机会）。
- **通用 kernel 仅作兜底**：仅当不存在特化匹配时使用。

### G4 Grid 总数不超过核数
- **必须**`grid = (min(total_blocks, num_cores),)`，禁止直接 `grid = (total_blocks,)`。
- **Why:** 输出规模大时 total_blocks 可能远超核数（如 Interpolate 上采样 N*C*H_out），超 grid 上限；且多余 program 会空跑。
- **2D grid 例外**：当 `total_blocks <= num_cores` 时可用 2D grid 充分利用多核。

### G5 int32 索引，避免 int64 降级
- **必须**在 kernel 内将 `tl.program_id` 和 `tl.arange` 结果 `.to(tl.int32)`。
- **Why:** int64 标量会触发地址计算降级；NPU 上 int32 索引更高效。
  ```python
  pid = tl.program_id(0).to(tl.int32)
  offs = (block_start + tl.arange(0, BLOCK)).to(tl.int32)
  ```

### G6 多核负载均衡分配公式
- **必须**按输出元素总数（非输入元素）分配核数，每个 program 处理一段连续输出。
- **负载均衡公式**（确保每个 core 处理的 block 数差不超过 1）：
  ```python
  blocks_per_core = total_blocks // num_cores
  remainder = total_blocks - blocks_per_core * num_cores
  if pid < remainder:
      my_blocks = blocks_per_core + 1
      start_block = pid * (blocks_per_core + 1)
  else:
      my_blocks = blocks_per_core
      start_block = remainder * (blocks_per_core + 1) + (pid - remainder) * blocks_per_core
  ```

### G7 输入必须 contiguous
- **必须** Host 侧进入 kernel 前调用 `x = x.contiguous()`。
- **Why:** 避免非连续张量导致 kernel 内 stride 计算复杂化。

### G8 坐标比较转 float32
- **禁止**直接对整数坐标使用 `tl.where(coord < 0, ...)`。
- **必须**先 `.to(tl.float32)` 再比较。
- **Why:** Triton Ascend 整数比较可能降级；同时 `tl.cast` 对负数是向零截断而非 floor，坐标计算需用 float32 比较修正。

---

## §2 Interpolate 算子（transformation-compute）

**算子类别**: `transformation-compute`（数据重排伴随权重计算；nearest/bilinear/bicubic/area 重采样）
**典型特征**: 4D NCHW 张量，支持 `nearest` / `bilinear` / `bicubic` / `area` 四种 mode，`align_corners` True/False，`size`/`scale_factor` 指定输出尺寸，fp32/fp16/bf16
**性能基准**: 73/73 pass，几何平均加速比 **1.3834x** vs torch（超 0.8x 项目目标、0.6x 用户目标）

### §2.0 首次生成必读：为什么必须把主要框架写对

Interpolate 是一个**计算特征高度异构**的算子：四种 mode × align_corners × 上/下采样 × 整数倍/分数倍比例，最优数据访问模式完全不同。**首次生成如果把框架写偏（例如用单一 kernel 统所有路径、漏掉 0.5x 下采样特化、用 gather 做整数倍上采样），后续迭代很难通过局部修bug把性能救回来**——历史上出现过迭代 58 轮仍无法精度/性能双达标的情况。

本章按 Layer 1→3 组织：
- **L1 是硬性约束**，首次生成必须全部满足；
- **L2 是 host 分派骨架 + 辅助函数**，必须一次写对；
- **L3 是关键 kernel 实现**，贴出优化的重点代码和那些不容易首次生成就生成出来的代码。

### §2.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 必须按 (mode, align_corners, 采样比例) 多策略分派
- **必须**在 host 侧 `_route` 方法中按 `(mode, align_corners, H_out vs H_in, W_out vs W_in)` 分派到特化 kernel。
- **禁止**用单一 kernel 统一处理所有 mode 和采样比例。
- **Why:** 不同场景的最优数据访问模式完全不同（2x 上采样用 broadcast，0.5x 下采样用 strided/avgpool，通用用 gather），强行统一会丢失所有特化机会。
- **How to apply:** 见 L2.1 分派决策树。

#### L1.2 行预载必须用 pow2 MAX_IN 作为 constexpr
- **必须**用 `MAX_IN_W = next_power_of_2(W_in)`（或受 cap 限制）把输入宽度 pad 到 2 的幂，作为 constexpr 传入。
- **Why:** Ascend 上 fixed-shape vector load 必须是编译期常量长度；若 `tl.arange(0, W_in)` 中 W_in 非 constexpr 会触发 dynamic-shape load，退化为标量循环。
- **How to apply:** 见 L2.2 `_select_max_in_w` 与 L3 kernel 中的 `tl.arange(0, MAX_IN_W)`。

#### L1.3 2x 上采样必须用 reshape broadcast 复制像素
- **必须**在 `H_out == 2*H_in 且 W_out == 2*W_in` 时用 `nearest_2x_upsample_kernel`，通过 `vals.reshape([BLOCK_IN,1]) * ones([1,2])` 把 `[a,b,c]` 复制成 `[a,a,b,b,c,c]`。
- **禁止**用 gather 实现整数倍上采样。
- **Why:** 整数倍上采样的每个输入像素映射到固定的输出块，reshape broadcast 是 contiguous 计算，远快于 gather 的随机访问。
- **How to apply:** 见 L3.1 `nearest_2x_upsample_kernel`。

#### L1.4 0.5x 下采样必须用 strided contiguous load
- **必须**在 `H_in == 2*H_out 且 W_in == 2*W_out` 时用 strided load（步长 2），禁止用 gather。
- **Why:** 0.5x 下采样的输出像素 (h,w) 映射到输入 (2h, 2w)，是规则的步长访问，contiguous strided load 比 gather 快。
- **How to apply:** 见 L3.2 `nearest_half_downsample_kernel`。

#### L1.5 bilinear 0.5x 下采样 (ac=False) 必须退化为 2x2 avg pool
- **必须**在 `mode='bilinear' and not align_corners and H_in==2*H_out and W_in==2*W_out` 时用 `bilinear_half_pool_kernel`，把 bilinear 退化为 2x2 均值池化。
- **Why:** ac=False + 0.5x 下采样时，bilinear 的 4 个采样点正好是 2x2 输入块，权重全是 0.25，等价于均值池化。用 `reshape([BLOCK_W,2]) + tl.sum` 实现远快于通用 bilinear 的 gather。
- **⚠️ 关键细节**: kernel 内部必须**一次 contiguous load `2*BLOCK_W` 元素**，再 `reshape([BLOCK_W,2]) + tl.sum`。禁止 `x0=w_idx*2; x1=x0+1` 做 4 次离散 strided load——strided load 无法 coalesce，实测性能差 100 倍，单点决定 geomean 能否过 0.6x。
- **How to apply:** 见 L3.3 `bilinear_half_pool_kernel`。

#### L1.6 上采样必须用 2D 垂直分块复用输入行 + host 预计算 x 坐标
- **必须**在 `H_out > H_in`（上采样）时用 `bilinear_2d_tiled_precomputed_x_kernel`，BLOCK_H=2、MAX_KH=3，一次性 load 3 行输入供 2 行输出复用。
- **必须** host 侧用 `_precompute_bilinear_x_coords(W_in, W_out, align_corners, scale_w)` 预计算 x0/x1/xlambda 三张量表，传入 kernel。**禁止**在 kernel 内每个 w tile 重算 x 坐标（cast/floor/where 链）。
- **Why:** 上采样时多个输出行共享同一批输入行，2D 垂直分块复用输入行减少 HBM 读取；x 坐标对所有行相同，host 预计算一次比 kernel 内每个 tile 重算快得多。历史上 kernel 内重算 x 坐标的版本（vec_kernel）比 precomputed_x 版本慢，导致部分上采样 shape 极慢（<0.1x），几何平均从 0.5x 跌到 0.29x。
- **How to apply:** 见 L3.5 `bilinear_2d_tiled_precomputed_x_kernel` 与 L2.2 `_precompute_bilinear_x_coords`。

#### L1.7 align_corners=True 必须在 host 预计算坐标 + 标量累积
- **必须**ac=True 路径在 host 用 `numpy.float32` 预算坐标和权重，传入 kernel 作为指针。
- **必须**bicubic ac=True 用**标量逐项累加**（16 项 `val = val + w*pixel`）匹配 PyTorch C++ 标量计算顺序。
- **Why:** PyTorch C++ 的 ac=True 坐标用 float32 计算 (`scale = (H_in-1)/(H_out-1)`)，Triton kernel 内若重新计算会有精度差异导致 verify 失败。bicubic 的 16 项累加若向量化会改变累加顺序，与 PyTorch 标量顺序不一致。
- **How to apply:** 见 L2.2 `_precompute_bicubic_ac_true_all` 与 L3.6 `bicubic_ac_true_vec_kernel`。

#### L1.8 三次权重必须用 tl.where 向量化分段，禁止 if
- **必须**用 `tl.where(at < 1.0, w1, tl.where(at < 2.0, w2, 0.0))` 计算三次权重，禁止 `if` 分支。
- **Why:** Triton kernel 内 `if` 对 vector 数据无效，必须用 `tl.where` 做向量分段。
- **How to apply:** 见 L3.5 `bicubic_vec_kernel` 中的权重计算。

#### L1.9 grid_size 必须动态限制为核数
（见 §1 G4）

#### L1.10 NPU scale 必须用 vdiv 匹配 CANN 精度（仅低精度 ac=True）
- **必须**在 `align_corners=True 且 dtype in (fp16, bf16)` 时用 `_npu_scale`（NPU vdiv）计算 scale，其他场景用 `_cpu_scale`（CPU float32）避免 host-device sync。
- **Why:** 低精度 + ac=True 路径对 scale 精度敏感，CANN 的 vdiv 与 CPU float32 除法结果有微小差异，会导致 verify 不过。其他场景 CPU 计算足够且无同步开销。
- **How to apply:** 见 L2.2 `_npu_scale`、`_cpu_scale`、`_need_npu_scale`。

#### L1.11 ac=False 负坐标 floor 必须修正
- **必须**在 ac=False 计算 `src_y = scale * (i + 0.5) - 0.5` 后，用 `tl.where(src_y < y0_f, y0 - 1, y0)` 修正 floor。
- **Why:** `tl.cast(src_y, tl.int32)` 向零截断而非 floor，src_y 为负时会导致 t 偏移错误，verify 失败（case 13/44 等 upsampling 首行/首列）。
- **How to apply:** 见 L3.5 `bicubic_vec_kernel` 与 L3.7 说明。

#### L1.12 所有 kernel 调用加 multibuffer=True
- **必须**所有 kernel 调用附加 `multibuffer=True`。
- **Why:** 插值 kernel 为内存密集型，大量 global load/store；`multibuffer=True` 是 Ascend 内存密集型 kernel 的默认必备选项。
- **How to apply:** 见 L2.1 所有 kernel 调用末尾的 `multibuffer=True`。

#### L1.13 禁止在 forward 中使用 torch 计算
- **必须**`ModelNew.forward()` 中只负责 shape 计算、分派、host 预计算；所有像素级计算必须在 `@triton.jit` kernel 内完成。
- **Why:** `validate_triton_impl.py` Type-3 检查会 flag 任何 forward 中的 torch 计算为退化。

#### L1.14 bilinear 下采样/general 路径也必须用 host 预计算 x 坐标
- **必须** bilinear 的 **所有** 路径（上采样 + 下采样 + general）都用 `_precompute_bilinear_x_coords(W_in, W_out, align_corners, scale_w)` 预计算 x0/x1/xlambda，传入 kernel。
- **必须** 下采样/general 路径用 `bilinear_general_precomputed_x_kernel`（见 L3.4b），**禁止** 用 `bilinear_general_vec_kernel`（kernel 内重算 x 坐标）。
- **Why:** x 坐标对所有输出行相同，host 预计算一次比 kernel 内每个 w tile 重算（cast/floor/where/clamp 链）快得多。历史上 general 路径用 vec_kernel 会导致 bilinear ac=True 下采样 shape 极慢（<0.3x），Phase 4 改用 precomputed_x 后 11 个 shape 救回，geomean 从 0.53x 升到 0.73x。
- **How to apply:** 见 L3.4b `bilinear_general_precomputed_x_kernel` 与 L2.2 `_precompute_bilinear_x_coords`。

### §2.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧多策略分派 `_route`

```python
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_cores = _get_num_cores()
        self._coord_cache = {}

    def _route(self, x, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias):
        N, C, H_in, W_in = x.shape

        if size is not None:
            H_out, W_out = int(size[0]), int(size[1])
        elif scale_factor is not None:
            H_out = int(H_in * scale_factor)
            W_out = int(W_in * scale_factor)
        else:
            raise ValueError("either size or scale_factor should be defined")

        output = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
        total_rows = N * C * H_out
        grid_cap = int(os.environ.get('TRITON_INTERP_GRID_CAP', str(self.num_cores)))
        grid = (min(total_rows, grid_cap),)   # L1.9
        BLOCK_W = self._select_block_w(W_out)  # pow2, cap 512
        MAX_IN_W = self._select_max_in_w(W_in) # pow2, cap 4096 (L1.2)

        need_npu_scale = _need_npu_scale(x.dtype, align_corners)

        if mode == 'nearest':
            use_size = 1 if size is not None else 0
            if H_out == 2 * H_in and W_out == 2 * W_in:
                nearest_2x_upsample_kernel[grid](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    BLOCK_W=BLOCK_W, multibuffer=True)               # L1.3
            elif H_in == 2 * H_out and W_in == 2 * W_out:
                nearest_half_downsample_kernel[grid](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    BLOCK_W=BLOCK_W, multibuffer=True)               # L1.4
            else:
                scale_h, scale_w = _get_scales(
                    [(H_in, H_out), (W_in, W_out)], use_npu=need_npu_scale)
                nearest_vec_kernel[grid](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    use_size=use_size, scale_h=scale_h, scale_w=scale_w,
                    BLOCK_W=BLOCK_W, MAX_IN_W=MAX_IN_W, multibuffer=True)

        elif mode == 'bilinear' and align_corners and H_out > 1 and W_out > 1:
            ac = 1
            scale_h, scale_w = _get_scales(
                [(H_in - 1, H_out - 1), (W_in - 1, W_out - 1)],
                use_npu=need_npu_scale)
            if H_out > H_in:
                BLOCK_H = 2
                MAX_KH = 3
                num_h_tiles = (H_out + BLOCK_H - 1) // BLOCK_H
                total_tiles = N * C * num_h_tiles
                grid_2d = (min(total_tiles, grid_cap),)
                bilinear_2d_tiled_vec_kernel[grid_2d](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    align_corners=ac, scale_h=scale_h, scale_w=scale_w,
                    BLOCK_W=BLOCK_W, MAX_IN_W=MAX_IN_W,
                    BLOCK_H=BLOCK_H, MAX_KH=MAX_KH, multibuffer=True) # L1.6
            else:
                bilinear_general_vec_kernel[grid](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    align_corners=ac, scale_h=scale_h, scale_w=scale_w,
                    BLOCK_W=BLOCK_W, MAX_IN_W=MAX_IN_W, multibuffer=True)

        elif mode == 'bilinear':
            ac = 1 if align_corners else 0
            if align_corners:
                pairs = [(H_in - 1 if H_out > 1 else 0, H_out - 1 if H_out > 1 else 1),
                         (W_in - 1 if W_out > 1 else 0, W_out - 1 if W_out > 1 else 1)]
            else:
                pairs = [(H_in, H_out), (W_in, W_out)]
            scale_h, scale_w = _get_scales(pairs, use_npu=need_npu_scale)
            if align_corners and H_out == 1:
                scale_h = 0.0
            if align_corners and W_out == 1:
                scale_w = 0.0
            if not align_corners and H_in == 2 * H_out and W_in == 2 * W_out:
                bilinear_half_pool_kernel[grid](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    BLOCK_W=BLOCK_W, multibuffer=True)               # L1.5
            elif H_out > H_in:
                BLOCK_H = 2
                MAX_KH = 3
                num_h_tiles = (H_out + BLOCK_H - 1) // BLOCK_H
                total_tiles = N * C * num_h_tiles
                grid_2d = (min(total_tiles, grid_cap),)
                bilinear_2d_tiled_vec_kernel[grid_2d](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    align_corners=ac, scale_h=scale_h, scale_w=scale_w,
                    BLOCK_W=BLOCK_W, MAX_IN_W=MAX_IN_W,
                    BLOCK_H=BLOCK_H, MAX_KH=MAX_KH, multibuffer=True) # L1.6
            else:
                bilinear_general_vec_kernel[grid](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    align_corners=ac, scale_h=scale_h, scale_w=scale_w,
                    BLOCK_W=BLOCK_W, MAX_IN_W=MAX_IN_W, multibuffer=True)

        elif mode == 'bicubic':
            ac = 1 if align_corners else 0
            if align_corners:
                y_all, x_all = self._precompute_bicubic_ac_true_all(
                    H_in, W_in, H_out, W_out)                         # L1.7
                (y_coords, y_idx_m1, y_idx_0, y_idx_p1, y_idx_p2,
                 y_w_m1, y_w_0, y_w_p1, y_w_p2) = y_all
                (x_coords, x_idx_m1, x_idx_0, x_idx_p1, x_idx_p2,
                 x_w_m1, x_w_0, x_w_p1, x_w_p2) = x_all
                device = x.device
                bicubic_ac_true_vec_kernel[grid](
                    x, output,
                    y_idx_m1.to(device), y_idx_0.to(device), y_idx_p1.to(device), y_idx_p2.to(device),
                    y_w_m1.to(device), y_w_0.to(device), y_w_p1.to(device), y_w_p2.to(device),
                    x_idx_m1.to(device), x_idx_0.to(device), x_idx_p1.to(device), x_idx_p2.to(device),
                    x_w_m1.to(device), x_w_0.to(device), x_w_p1.to(device), x_w_p2.to(device),
                    N, C, H_in, W_in, H_out, W_out,
                    BLOCK_W=BLOCK_W, MAX_IN_W=MAX_IN_W, multibuffer=True)
            else:
                scale_h, scale_w = _get_scales(
                    [(H_in, H_out), (W_in, W_out)], use_npu=need_npu_scale)
                bicubic_vec_kernel[grid](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    align_corners=ac, scale_h=scale_h, scale_w=scale_w,
                    BLOCK_W=BLOCK_W, MAX_IN_W=MAX_IN_W, multibuffer=True)

        elif mode == 'area':
            max_kw = (W_in + W_out - 1) // W_out
            max_kw = min(max_kw, 64)
            if max_kw < 1:
                max_kw = 1
            area_vec_kernel[grid](
                x, output, N, C, H_in, W_in, H_out, W_out,
                BLOCK_W=BLOCK_W, MAX_KW=max_kw, MAX_IN_W=MAX_IN_W,
                multibuffer=True)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return output

    def forward(self, x, size=None, scale_factor=None,
                mode='nearest', align_corners=None,
                recompute_scale_factor=None, antialias=False):
        return self._route(x, size, scale_factor, mode, align_corners,
                           recompute_scale_factor, antialias)
```

#### L2.2 辅助函数

```python
def _get_num_cores():
    try:
        device = torch.npu.current_device()
        props = triton.runtime.driver.active.utils.get_device_properties(device)
        num_cores = props.get("num_vectorcore", -1)
        if num_cores <= 0:
            num_cores = props.get("num_aicore", 48)
        return max(num_cores, 1)
    except Exception:
        return 48


def _select_block_w(W_out, cap=512):
    if W_out <= cap:
        return triton.next_power_of_2(W_out)
    return cap


def _select_max_in_w(W_in, cap=4096):
    if W_in <= cap:
        return triton.next_power_of_2(W_in)
    return cap


def _cpu_scale(num, den):
    if den == 0:
        return 0.0
    return float(np.float32(num) / np.float32(den))


_SCALE_CACHE = {}


def _npu_scale(num, den):
    if den == 0:
        return 0.0
    key = (num, den)
    if key not in _SCALE_CACHE:
        a = torch.tensor([num], dtype=torch.float32, device='npu')
        b = torch.tensor([den], dtype=torch.float32, device='npu')
        _SCALE_CACHE[key] = (a / b).cpu().item()
    return _SCALE_CACHE[key]


def _get_scales(pairs, use_npu=False):
    if not use_npu:
        return [_cpu_scale(num, den) for num, den in pairs]

    nums, dens, indices = [], [], []
    for i, (num, den) in enumerate(pairs):
        key = (num, den)
        if key in _SCALE_CACHE:
            continue
        if den == 0:
            _SCALE_CACHE[key] = 0.0
            continue
        nums.append(num)
        dens.append(den)
        indices.append(i)

    if nums:
        a = torch.tensor(nums, dtype=torch.float32, device='npu')
        b = torch.tensor(dens, dtype=torch.float32, device='npu')
        vals = (a / b).cpu().tolist()
        for idx, val in zip(indices, vals):
            _SCALE_CACHE[pairs[idx]] = val

    return [_SCALE_CACHE.get((num, den), _cpu_scale(num, den)) for num, den in pairs]


def _need_npu_scale(dtype, align_corners):
    return align_corners and dtype in (torch.float16, torch.bfloat16)


def _precompute_bicubic_ac_true_all(self, H_in, W_in, H_out, W_out):
    key = ('bicubic_all', H_in, W_in, H_out, W_out)
    if key not in self._coord_cache:
        scale_y = np.float32(H_in - 1) / np.float32(max(H_out - 1, 1))
        scale_x = np.float32(W_in - 1) / np.float32(max(W_out - 1, 1))
        A = -0.75

        def _cubic(t):
            t = abs(float(t))
            if t <= 1.0:
                return float(((A + 2.0) * t - (A + 3.0)) * t * t + 1.0)
            else:
                return float(((A * t - 5.0 * A) * t + 8.0 * A) * t - 4.0 * A)

        def _compute_dim(L_in, L_out, scale):
            coords, idx_m1, idx_0, idx_p1, idx_p2 = [], [], [], [], []
            w_m1, w_0, w_p1, w_p2 = [], [], [], []
            for i in range(L_out):
                s = np.float32(i) * scale
                coords.append(float(s))
                y0 = int(np.float32(np.floor(s)))
                if s < 0:
                    y0 -= 1
                wy = float(s - np.float32(y0))
                idx_m1.append(max(0, min(y0 - 1, L_in - 1)))
                idx_0.append(max(0, min(y0, L_in - 1)))
                idx_p1.append(max(0, min(y0 + 1, L_in - 1)))
                idx_p2.append(max(0, min(y0 + 2, L_in - 1)))
                w_m1.append(_cubic(-1.0 - wy))
                w_0.append(_cubic(0.0 - wy))
                w_p1.append(_cubic(1.0 - wy))
                w_p2.append(_cubic(2.0 - wy))
            return (
                torch.tensor(coords, dtype=torch.float32),
                torch.tensor(idx_m1, dtype=torch.int32),
                torch.tensor(idx_0, dtype=torch.int32),
                torch.tensor(idx_p1, dtype=torch.int32),
                torch.tensor(idx_p2, dtype=torch.int32),
                torch.tensor(w_m1, dtype=torch.float32),
                torch.tensor(w_0, dtype=torch.float32),
                torch.tensor(w_p1, dtype=torch.float32),
                torch.tensor(w_p2, dtype=torch.float32),
            )

        y_all = _compute_dim(H_in, H_out, scale_y)
        x_all = _compute_dim(W_in, W_out, scale_x)
        self._coord_cache[key] = (y_all, x_all)
    return self._coord_cache[key]
```

### §2.3 Layer 3: 关键 kernel 实现（优化重点与易错代码）

#### L3.1 nearest 2x 上采样 `nearest_2x_upsample_kernel`

```python
@triton.jit
def nearest_2x_upsample_kernel(
    input_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)
    BLOCK_IN: tl.constexpr = BLOCK_W // 2

    for row in range(pid, total_rows, num_programs):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        src_h = h // 2
        input_base = (n * C + c) * H_in * W_in
        output_row_off = ((n * C + c) * H_out + h) * W_out

        for w_start in range(0, W_out, BLOCK_W):
            in_w_start = w_start // 2
            in_offs = tl.arange(0, BLOCK_IN)
            in_idx = in_w_start + in_offs
            in_mask = in_idx < W_in

            vals = tl.load(input_ptr + input_base + src_h * W_in + in_idx,
                           mask=in_mask, other=0.0).to(tl.float32)

            # [a, b, c] -> [a, a, b, b, c, c]
            vals = tl.reshape(vals, [BLOCK_IN, 1])
            ones = tl.full([1, 2], 1.0, tl.float32)
            vals_dup = vals * ones
            out_vals = tl.reshape(vals_dup, [BLOCK_W])

            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out
            tl.store(output_ptr + output_row_off + w_idx, out_vals, mask=mask)
```

#### L3.2 nearest 0.5x 下采样 `nearest_half_downsample_kernel`

```python
@triton.jit
def nearest_half_downsample_kernel(
    input_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)

    for row in range(pid, total_rows, num_programs):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        src_h = 2 * h
        input_base = (n * C + c) * H_in * W_in
        output_row_off = ((n * C + c) * H_out + h) * W_out

        for w_start in range(0, W_out, BLOCK_W):
            in_w_start = 2 * w_start
            in_offs = tl.arange(0, BLOCK_W)
            in_idx = in_w_start + 2 * in_offs   # 步长 2，但仍为 contiguous vector
            in_mask = in_idx < W_in

            vals = tl.load(input_ptr + input_base + src_h * W_in + in_idx,
                           mask=in_mask, other=0.0).to(tl.float32)

            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out
            tl.store(output_ptr + output_row_off + w_idx, vals, mask=mask)
```

#### L3.3 bilinear 0.5x 下采样退化为 avg pool `bilinear_half_pool_kernel`

```python
@triton.jit
def bilinear_half_pool_kernel(
    input_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)
    rows_per_prog = (total_rows + num_programs - 1) // num_programs
    row_start = pid * rows_per_prog
    row_end = row_start + rows_per_prog
    if row_end > total_rows:
        row_end = total_rows

    QUARTER = 0.25

    for row in range(row_start, row_end):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        y0 = 2 * h
        y1 = y0 + 1
        if y1 >= H_in:
            y1 = H_in - 1

        input_base = (n * C + c) * H_in * W_in
        output_row_off = ((n * C + c) * H_out + h) * W_out

        for w_start in range(0, W_out, BLOCK_W):
            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out

            # 一次 contiguous load 2*BLOCK_W 元素，禁止拆成 4 次 strided load
            in_start = 2 * w_start
            in_offs = tl.arange(0, 2 * BLOCK_W)
            in_idx = in_start + in_offs
            in_mask = in_idx < W_in

            row0 = tl.load(input_ptr + input_base + y0 * W_in + in_idx,
                           mask=in_mask, other=0.0).to(tl.float32)
            row1 = tl.load(input_ptr + input_base + y1 * W_in + in_idx,
                           mask=in_mask, other=0.0).to(tl.float32)

            row0_pairs = tl.reshape(row0, [BLOCK_W, 2])
            row1_pairs = tl.reshape(row1, [BLOCK_W, 2])
            row0_sum = tl.sum(row0_pairs, axis=1)
            row1_sum = tl.sum(row1_pairs, axis=1)

            val = (row0_sum + row1_sum) * QUARTER
            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

#### L3.4 通用 bilinear `bilinear_general_vec_kernel`

```python
@triton.jit
def bilinear_general_vec_kernel(
    input_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    align_corners: tl.constexpr,
    scale_h, scale_w,
    BLOCK_W: tl.constexpr,
    MAX_IN_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)

    r_offs = tl.arange(0, MAX_IN_W)
    r_mask = r_offs < W_in

    for row in range(pid, total_rows, num_programs):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        h_f = tl.cast(h, tl.float32)
        sh = tl.cast(scale_h, tl.float32)
        if align_corners == 1:
            src_y = sh * h_f
        else:
            half = tl.full((), 0.5, tl.float32)
            src_y = sh * (h_f + half) - half

        y0 = tl.cast(src_y, tl.int32)
        y0_f = tl.cast(y0, tl.float32)
        if align_corners != 1:
            y0 = tl.where(src_y < y0_f, y0 - 1, y0)    # L1.11 负坐标 floor 修正
            y0_f = tl.cast(y0, tl.float32)
        y1 = y0 + 1
        y0 = tl.where(y0 < 0, 0, y0)
        y1 = tl.where(y1 >= H_in, H_in - 1, y1)

        ylambda = src_y - y0_f
        ylambda = tl.where(ylambda < 0.0, 0.0, ylambda)
        ylambda = tl.where(ylambda > 1.0, 1.0, ylambda)
        y0lambda = 1.0 - ylambda

        input_base = (n * C + c) * H_in * W_in
        row0 = tl.load(input_ptr + input_base + y0 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        row1 = tl.load(input_ptr + input_base + y1 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        output_row_off = ((n * C + c) * H_out + h) * W_out

        for w_start in range(0, W_out, BLOCK_W):
            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out

            w_f = tl.cast(w_idx, tl.float32)
            sw = tl.cast(scale_w, tl.float32)
            if align_corners == 1:
                src_x = sw * w_f
            else:
                half = tl.full((), 0.5, tl.float32)
                src_x = sw * (w_f + half) - half

            x0 = tl.cast(src_x, tl.int32)
            x0_f = tl.cast(x0, tl.float32)
            if align_corners != 1:
                x0 = tl.where(src_x < x0_f, x0 - 1, x0)
                x0_f = tl.cast(x0, tl.float32)
            x1 = x0 + 1
            x0 = tl.where(x0 < 0, 0, x0)
            x1 = tl.where(x1 >= W_in, W_in - 1, x1)

            xlambda = src_x - x0_f
            xlambda = tl.where(xlambda < 0.0, 0.0, xlambda)
            xlambda = tl.where(xlambda > 1.0, 1.0, xlambda)
            x0lambda = 1.0 - xlambda

            i00 = tl.gather(row0, x0, 0)
            i01 = tl.gather(row0, x1, 0)
            i10 = tl.gather(row1, x0, 0)
            i11 = tl.gather(row1, x1, 0)

            top = x0lambda * i00 + xlambda * i01
            bot = x0lambda * i10 + xlambda * i11
            val = y0lambda * top + ylambda * bot

            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

#### L3.4b bilinear 下采样/general 预计算 x 坐标版 `bilinear_general_precomputed_x_kernel`（L1.14 强制）

L3.4 的 `bilinear_general_vec_kernel` 在 kernel 内重算 x 坐标，性能差。**必须**用本节版本：x0/x1/xlambda 由 host 预计算传入。

```python
@triton.jit
def bilinear_general_precomputed_x_kernel(
    input_ptr, output_ptr,
    x0_ptr, x1_ptr, xlambda_ptr,
    N, C, H_in, W_in, H_out, W_out,
    align_corners: tl.constexpr,
    scale_h, scale_w,
    BLOCK_W: tl.constexpr,
    MAX_IN_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)
    r_offs = tl.arange(0, MAX_IN_W)
    r_mask = r_offs < W_in

    for row in range(pid, total_rows, num_programs):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        # y 坐标仍在 kernel 内算（每行不同）
        h_f = tl.cast(h, tl.float32)
        sh = tl.cast(scale_h, tl.float32)
        if align_corners == 1:
            src_y = sh * h_f
        else:
            half = tl.full((), 0.5, tl.float32)
            src_y = sh * (h_f + half) - half
        y0 = tl.cast(src_y, tl.int32)
        y0_f = tl.cast(y0, tl.float32)
        if align_corners != 1:
            y0 = tl.where(src_y < y0_f, y0 - 1, y0)
            y0_f = tl.cast(y0, tl.float32)
        y1 = y0 + 1
        y0 = tl.where(y0 < 0, 0, y0)
        y1 = tl.where(y1 >= H_in, H_in - 1, y1)
        ylambda = src_y - y0_f
        ylambda = tl.where(ylambda < 0.0, 0.0, ylambda)
        ylambda = tl.where(ylambda > 1.0, 1.0, ylambda)
        y0lambda = 1.0 - ylambda

        input_base = (n * C + c) * H_in * W_in
        row0 = tl.load(input_ptr + input_base + y0 * W_in + r_offs, mask=r_mask, other=0.0).to(tl.float32)
        row1 = tl.load(input_ptr + input_base + y1 * W_in + r_offs, mask=r_mask, other=0.0).to(tl.float32)
        output_row_off = ((n * C + c) * H_out + h) * W_out

        for w_start in range(0, W_out, BLOCK_W):
            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out
            # x 坐标从 host 预计算表 load（L1.14）
            x0 = tl.load(x0_ptr + w_idx, mask=mask, other=0).to(tl.int32)
            x1 = tl.load(x1_ptr + w_idx, mask=mask, other=0).to(tl.int32)
            xlambda = tl.load(xlambda_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)
            x0lambda = 1.0 - xlambda

            i00 = tl.gather(row0, x0, 0)
            i01 = tl.gather(row0, x1, 0)
            i10 = tl.gather(row1, x0, 0)
            i11 = tl.gather(row1, x1, 0)
            top = x0lambda * i00 + xlambda * i01
            bot = x0lambda * i10 + xlambda * i11
            val = y0lambda * top + ylambda * bot
            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

> host 分派：bilinear 的 general/下采样分支（非 2x 上采样、非 0.5x 下采样）调用本 kernel，先 `x0,x1,xl = _precompute_bilinear_x_coords(W_in, W_out, align_corners, scale_w)` 再传入。

#### L3.5 上采样 2D 垂直分块 + host 预计算 x 坐标 `bilinear_2d_tiled_precomputed_x_kernel`

**关键**：x 坐标（x0/x1/xlambda）对所有输出行相同，**必须 host 预计算**为三张量表传入 kernel，禁止 kernel 内每个 w tile 重算。kernel 内重算 x 坐标的旧版本（vec_kernel）会导致上采样 shape 极慢。

```python
@triton.jit
def bilinear_2d_tiled_precomputed_x_kernel(
    input_ptr, output_ptr,
    x0_ptr, x1_ptr, xlambda_ptr,
    N, C, H_in, W_in, H_out, W_out,
    align_corners: tl.constexpr,
    scale_h, scale_w,
    BLOCK_W: tl.constexpr,
    MAX_IN_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MAX_KH: tl.constexpr,
):
    pid = tl.program_id(0)
    num_h_tiles = (H_out + BLOCK_H - 1) // BLOCK_H
    total_tiles = N * C * num_h_tiles
    num_programs = tl.num_programs(0)

    r_offs = tl.arange(0, MAX_IN_W)
    kh_offs = tl.arange(0, MAX_KH)

    for tile in range(pid, total_tiles, num_programs):
        n = tile // (C * num_h_tiles)
        tmp = tile - n * C * num_h_tiles
        c = tmp // num_h_tiles
        h_tile = tmp - c * num_h_tiles
        h_start = h_tile * BLOCK_H

        input_base = (n * C + c) * H_in * W_in
        output_base = ((n * C + c) * H_out + h_start) * W_out

        h_f = tl.cast(h_start, tl.float32)
        sh = tl.cast(scale_h, tl.float32)
        if align_corners == 1:
            src_y = sh * h_f
        else:
            half = tl.full((), 0.5, tl.float32)
            src_y = sh * (h_f + half) - half
        y_min = tl.cast(src_y, tl.int32)
        y_min_f = tl.cast(y_min, tl.float32)
        if align_corners != 1:
            y_min = tl.where(src_y < y_min_f, y_min - 1, y_min)

        # 一次性 load MAX_KH 行输入到 flat UB buffer
        y_idx = y_min + kh_offs
        y_idx = tl.where(y_idx < 0, 0, y_idx)
        y_idx = tl.where(y_idx >= H_in, H_in - 1, y_idx)
        rr = y_idx[:, None]
        cc = r_offs[None, :]
        block2d = tl.load(input_ptr + input_base + rr * W_in + cc,
                          mask=(cc < W_in), other=0.0).to(tl.float32)
        flat_buf = tl.reshape(block2d, [MAX_KH * MAX_IN_W])

        for dh in range(BLOCK_H):
            h = h_start + dh
            h_valid = h < H_out

            h_f = tl.cast(h, tl.float32)
            if align_corners == 1:
                src_y = sh * h_f
            else:
                src_y = sh * (h_f + half) - half
            y0 = tl.cast(src_y, tl.int32)
            y0_f = tl.cast(y0, tl.float32)
            if align_corners != 1:
                y0 = tl.where(src_y < y0_f, y0 - 1, y0)
            y1 = y0 + 1
            y0 = tl.where(y0 < 0, 0, y0)
            y1 = tl.where(y1 >= H_in, H_in - 1, y1)
            ylambda = src_y - tl.cast(y0, tl.float32)
            ylambda = tl.where(ylambda < 0.0, 0.0, ylambda)
            ylambda = tl.where(ylambda > 1.0, 1.0, ylambda)
            y0lambda = 1.0 - ylambda

            ry0 = y0 - y_min
            ry1 = y1 - y_min

            for w_start in range(0, W_out, BLOCK_W):
                w_offs = tl.arange(0, BLOCK_W)
                w_idx = w_start + w_offs
                mask = (w_idx < W_out) & h_valid

                # x 坐标从 host 预计算的张量 load，禁止 kernel 内重算
                x0 = tl.load(x0_ptr + w_idx, mask=mask, other=0).to(tl.int32)
                x1 = tl.load(x1_ptr + w_idx, mask=mask, other=0).to(tl.int32)
                xlambda = tl.load(xlambda_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)
                x0lambda = 1.0 - xlambda

                fx0 = ry0 * MAX_IN_W + x0
                fx1 = ry0 * MAX_IN_W + x1
                fx2 = ry1 * MAX_IN_W + x0
                fx3 = ry1 * MAX_IN_W + x1

                i00 = tl.gather(flat_buf, fx0, 0)
                i01 = tl.gather(flat_buf, fx1, 0)
                i10 = tl.gather(flat_buf, fx2, 0)
                i11 = tl.gather(flat_buf, fx3, 0)

                top = x0lambda * i00 + xlambda * i01
                bot = x0lambda * i10 + xlambda * i11
                val = y0lambda * top + ylambda * bot

                tl.store(output_ptr + output_base + dh * W_out + w_idx, val, mask=mask)
```

**Opt10：x 表 load 复用（2D tiled kernel）**：上面 kernel 中 `x0/x1/xlambda` 的 `tl.load` 在 `for dh` 内、`for w_start` 内，每个 dh 重复 load 同一份 x 表。若 `BLOCK_H=2`，x 表被 load 2 次。优化：把 x 表 load 提到 `for dh` 循环外、在 `for w_start` 同级预先 load 整个 W_out 的 x0/x1/xlambda 到 UB buffer（一次性 `tl.load(x0_ptr + tl.arange(0, W_out_POW2))`），dh 循环内按 w_start 切片复用。对 BLOCK_H 较大或 W_out 较大的上采样 shape 提升明显（归档 Phase4 实测救回多个 bilinear ac=True 上采样 shape）。

**host 预计算函数**（必须配套使用）：
```python
def _precompute_bilinear_x_coords(W_in, W_out, align_corners, scale_w):
    import math
    x0 = torch.empty(W_out, dtype=torch.int32)
    x1 = torch.empty(W_out, dtype=torch.int32)
    xlambda = torch.empty(W_out, dtype=torch.float32)
    for i in range(W_out):
        if align_corners:
            src_x = scale_w * i
        else:
            src_x = scale_w * (i + 0.5) - 0.5
        x0_i = int(math.floor(src_x))
        x0_i = max(0, x0_i)
        x0_i = x0_i if x0_i < W_in - 1 else W_in - 1
        x1_i = x0_i + 1
        x1_i = x1_i if x1_i < W_in else W_in - 1
        xl = float(src_x - x0_i)
        xl = 0.0 if xl < 0.0 else xl
        xl = 1.0 if xl > 1.0 else xl
        xlambda[i] = xl
        x0[i] = x0_i
        x1[i] = x1_i
    return x0, x1, xlambda
```

#### L3.6 bicubic ac=False `bicubic_vec_kernel`

```python
@triton.jit
def bicubic_vec_kernel(
    input_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    align_corners: tl.constexpr,
    scale_h, scale_w,
    BLOCK_W: tl.constexpr,
    MAX_IN_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)
    A = -0.75

    r_offs = tl.arange(0, MAX_IN_W)
    r_mask = r_offs < W_in

    for row in range(pid, total_rows, num_programs):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        h_f = tl.cast(h, tl.float32)
        sh = tl.cast(scale_h, tl.float32)
        if align_corners == 1:
            src_y = sh * h_f
        else:
            half = tl.full((), 0.5, tl.float32)
            src_y = sh * (h_f + half) - half

        in_y = tl.cast(src_y, tl.int32)
        in_y_f = tl.cast(in_y, tl.float32)
        in_y = tl.where(src_y < in_y_f, in_y - 1, in_y)    # L1.11 负数 floor 修正
        in_y_f = tl.cast(in_y, tl.float32)                 # 修正后重新取 float
        t_y = src_y - in_y_f

        yy0 = in_y - 1
        yy1 = in_y
        yy2 = in_y + 1
        yy3 = in_y + 2
        yy0 = tl.where(yy0 < 0, 0, yy0)
        yy0 = tl.where(yy0 >= H_in, H_in - 1, yy0)
        yy1 = tl.where(yy1 < 0, 0, yy1)
        yy1 = tl.where(yy1 >= H_in, H_in - 1, yy1)
        yy2 = tl.where(yy2 < 0, 0, yy2)
        yy2 = tl.where(yy2 >= H_in, H_in - 1, yy2)
        yy3 = tl.where(yy3 < 0, 0, yy3)
        yy3 = tl.where(yy3 >= H_in, H_in - 1, yy3)

        # 垂直方向 4 个三次权重，用 tl.where 分段（L1.8）
        tt0 = t_y + 1.0
        tt0_abs = tl.where(tt0 < 0.0, -tt0, tt0)
        wy0 = tl.where(tt0_abs < 1.0,
                       ((A + 2.0) * tt0_abs - (A + 3.0)) * tt0_abs * tt0_abs + 1.0,
                       tl.where(tt0_abs < 2.0,
                                ((A * tt0_abs - 5.0 * A) * tt0_abs + 8.0 * A) * tt0_abs - 4.0 * A,
                                0.0))
        tt1 = t_y
        tt1_abs = tl.where(tt1 < 0.0, -tt1, tt1)
        wy1 = tl.where(tt1_abs < 1.0,
                       ((A + 2.0) * tt1_abs - (A + 3.0)) * tt1_abs * tt1_abs + 1.0,
                       tl.where(tt1_abs < 2.0,
                                ((A * tt1_abs - 5.0 * A) * tt1_abs + 8.0 * A) * tt1_abs - 4.0 * A,
                                0.0))
        tt2 = 1.0 - t_y
        tt2_abs = tl.where(tt2 < 0.0, -tt2, tt2)
        wy2 = tl.where(tt2_abs < 1.0,
                       ((A + 2.0) * tt2_abs - (A + 3.0)) * tt2_abs * tt2_abs + 1.0,
                       tl.where(tt2_abs < 2.0,
                                ((A * tt2_abs - 5.0 * A) * tt2_abs + 8.0 * A) * tt2_abs - 4.0 * A,
                                0.0))
        tt3 = 2.0 - t_y
        tt3_abs = tl.where(tt3 < 0.0, -tt3, tt3)
        wy3 = tl.where(tt3_abs < 1.0,
                       ((A + 2.0) * tt3_abs - (A + 3.0)) * tt3_abs * tt3_abs + 1.0,
                       tl.where(tt3_abs < 2.0,
                                ((A * tt3_abs - 5.0 * A) * tt3_abs + 8.0 * A) * tt3_abs - 4.0 * A,
                                0.0))
        w_sum_y = wy0 + wy1 + wy2 + wy3

        input_base = (n * C + c) * H_in * W_in
        row0 = tl.load(input_ptr + input_base + yy0 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        row1 = tl.load(input_ptr + input_base + yy1 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        row2 = tl.load(input_ptr + input_base + yy2 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        row3 = tl.load(input_ptr + input_base + yy3 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        output_row_off = ((n * C + c) * H_out + h) * W_out

        for w_start in range(0, W_out, BLOCK_W):
            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out

            w_f = tl.cast(w_idx, tl.float32)
            sw = tl.cast(scale_w, tl.float32)
            if align_corners == 1:
                src_x = sw * w_f
            else:
                half = tl.full((), 0.5, tl.float32)
                src_x = sw * (w_f + half) - half

            in_x = tl.cast(src_x, tl.int32)
            in_x_f = tl.cast(in_x, tl.float32)
            in_x = tl.where(src_x < in_x_f, in_x - 1, in_x)
            in_x_f = tl.cast(in_x, tl.float32)
            t_x = src_x - in_x_f

            xx0 = in_x - 1
            xx1 = in_x
            xx2 = in_x + 1
            xx3 = in_x + 2
            xx0 = tl.where(xx0 < 0, 0, xx0)
            xx0 = tl.where(xx0 >= W_in, W_in - 1, xx0)
            xx1 = tl.where(xx1 < 0, 0, xx1)
            xx1 = tl.where(xx1 >= W_in, W_in - 1, xx1)
            xx2 = tl.where(xx2 < 0, 0, xx2)
            xx2 = tl.where(xx2 >= W_in, W_in - 1, xx2)
            xx3 = tl.where(xx3 < 0, 0, xx3)
            xx3 = tl.where(xx3 >= W_in, W_in - 1, xx3)

            tx0 = t_x + 1.0
            tx0_abs = tl.where(tx0 < 0.0, -tx0, tx0)
            wx0 = tl.where(tx0_abs < 1.0,
                           ((A + 2.0) * tx0_abs - (A + 3.0)) * tx0_abs * tx0_abs + 1.0,
                           tl.where(tx0_abs < 2.0,
                                    ((A * tx0_abs - 5.0 * A) * tx0_abs + 8.0 * A) * tx0_abs - 4.0 * A,
                                    0.0))
            tx1 = t_x
            tx1_abs = tl.where(tx1 < 0.0, -tx1, tx1)
            wx1 = tl.where(tx1_abs < 1.0,
                           ((A + 2.0) * tx1_abs - (A + 3.0)) * tx1_abs * tx1_abs + 1.0,
                           tl.where(tx1_abs < 2.0,
                                    ((A * tx1_abs - 5.0 * A) * tx1_abs + 8.0 * A) * tx1_abs - 4.0 * A,
                                    0.0))
            tx2 = 1.0 - t_x
            tx2_abs = tl.where(tx2 < 0.0, -tx2, tx2)
            wx2 = tl.where(tx2_abs < 1.0,
                           ((A + 2.0) * tx2_abs - (A + 3.0)) * tx2_abs * tx2_abs + 1.0,
                           tl.where(tx2_abs < 2.0,
                                    ((A * tx2_abs - 5.0 * A) * tx2_abs + 8.0 * A) * tx2_abs - 4.0 * A,
                                    0.0))
            tx3 = 2.0 - t_x
            tx3_abs = tl.where(tx3 < 0.0, -tx3, tx3)
            wx3 = tl.where(tx3_abs < 1.0,
                           ((A + 2.0) * tx3_abs - (A + 3.0)) * tx3_abs * tx3_abs + 1.0,
                           tl.where(tx3_abs < 2.0,
                                    ((A * tx3_abs - 5.0 * A) * tx3_abs + 8.0 * A) * tx3_abs - 4.0 * A,
                                    0.0))
            w_sum_x = wx0 + wx1 + wx2 + wx3

            p00 = tl.gather(row0, xx0, 0)
            p01 = tl.gather(row0, xx1, 0)
            p02 = tl.gather(row0, xx2, 0)
            p03 = tl.gather(row0, xx3, 0)
            r0 = wx0 * p00 + wx1 * p01 + wx2 * p02 + wx3 * p03
            p10 = tl.gather(row1, xx0, 0)
            p11 = tl.gather(row1, xx1, 0)
            p12 = tl.gather(row1, xx2, 0)
            p13 = tl.gather(row1, xx3, 0)
            r1 = wx0 * p10 + wx1 * p11 + wx2 * p12 + wx3 * p13
            p20 = tl.gather(row2, xx0, 0)
            p21 = tl.gather(row2, xx1, 0)
            p22 = tl.gather(row2, xx2, 0)
            p23 = tl.gather(row2, xx3, 0)
            r2 = wx0 * p20 + wx1 * p21 + wx2 * p22 + wx3 * p23
            p30 = tl.gather(row3, xx0, 0)
            p31 = tl.gather(row3, xx1, 0)
            p32 = tl.gather(row3, xx2, 0)
            p33 = tl.gather(row3, xx3, 0)
            r3 = wx0 * p30 + wx1 * p31 + wx2 * p32 + wx3 * p33

            val = wy0 * r0 + wy1 * r1 + wy2 * r2 + wy3 * r3
            w_sum_total = w_sum_x * w_sum_y
            val = tl.where(w_sum_total != 0.0, val / w_sum_total, val)

            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

#### L3.7 bicubic ac=True `bicubic_ac_true_vec_kernel`

```python
@triton.jit
def bicubic_ac_true_vec_kernel(
    input_ptr, output_ptr,
    y_idx_m1_ptr, y_idx_0_ptr, y_idx_p1_ptr, y_idx_p2_ptr,
    y_w_m1_ptr, y_w_0_ptr, y_w_p1_ptr, y_w_p2_ptr,
    x_idx_m1_ptr, x_idx_0_ptr, x_idx_p1_ptr, x_idx_p2_ptr,
    x_w_m1_ptr, x_w_0_ptr, x_w_p1_ptr, x_w_p2_ptr,
    N, C, H_in, W_in, H_out, W_out,
    BLOCK_W: tl.constexpr,
    MAX_IN_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)
    ZERO = tl.full((), 0.0, tl.float32)

    r_offs = tl.arange(0, MAX_IN_W)
    r_mask = r_offs < W_in

    for row in range(pid, total_rows, num_programs):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        yy0 = tl.load(y_idx_m1_ptr + h).to(tl.int32)
        yy1 = tl.load(y_idx_0_ptr + h).to(tl.int32)
        yy2 = tl.load(y_idx_p1_ptr + h).to(tl.int32)
        yy3 = tl.load(y_idx_p2_ptr + h).to(tl.int32)
        wy0 = tl.load(y_w_m1_ptr + h).to(tl.float32)
        wy1 = tl.load(y_w_0_ptr + h).to(tl.float32)
        wy2 = tl.load(y_w_p1_ptr + h).to(tl.float32)
        wy3 = tl.load(y_w_p2_ptr + h).to(tl.float32)

        input_base = (n * C + c) * H_in * W_in
        row0 = tl.load(input_ptr + input_base + yy0 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        row1 = tl.load(input_ptr + input_base + yy1 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        row2 = tl.load(input_ptr + input_base + yy2 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        row3 = tl.load(input_ptr + input_base + yy3 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        output_row_off = ((n * C + c) * H_out + h) * W_out

        for w_start in range(0, W_out, BLOCK_W):
            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out

            xx0 = tl.load(x_idx_m1_ptr + w_idx, mask=mask, other=0).to(tl.int32)
            xx1 = tl.load(x_idx_0_ptr + w_idx, mask=mask, other=0).to(tl.int32)
            xx2 = tl.load(x_idx_p1_ptr + w_idx, mask=mask, other=0).to(tl.int32)
            xx3 = tl.load(x_idx_p2_ptr + w_idx, mask=mask, other=0).to(tl.int32)
            wx0 = tl.load(x_w_m1_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)
            wx1 = tl.load(x_w_0_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)
            wx2 = tl.load(x_w_p1_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)
            wx3 = tl.load(x_w_p2_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)

            p00 = tl.gather(row0, xx0, 0)
            p01 = tl.gather(row0, xx1, 0)
            p02 = tl.gather(row0, xx2, 0)
            p03 = tl.gather(row0, xx3, 0)
            p10 = tl.gather(row1, xx0, 0)
            p11 = tl.gather(row1, xx1, 0)
            p12 = tl.gather(row1, xx2, 0)
            p13 = tl.gather(row1, xx3, 0)
            p20 = tl.gather(row2, xx0, 0)
            p21 = tl.gather(row2, xx1, 0)
            p22 = tl.gather(row2, xx2, 0)
            p23 = tl.gather(row2, xx3, 0)
            p30 = tl.gather(row3, xx0, 0)
            p31 = tl.gather(row3, xx1, 0)
            p32 = tl.gather(row3, xx2, 0)
            p33 = tl.gather(row3, xx3, 0)

            # 16 项标量逐项累加，匹配 PyTorch C++ 顺序（L1.7）
            val = ZERO
            w_sum_total = ZERO
            val = val + wy0 * wx0 * p00
            w_sum_total = w_sum_total + wy0 * wx0
            val = val + wy0 * wx1 * p01
            w_sum_total = w_sum_total + wy0 * wx1
            val = val + wy0 * wx2 * p02
            w_sum_total = w_sum_total + wy0 * wx2
            val = val + wy0 * wx3 * p03
            w_sum_total = w_sum_total + wy0 * wx3
            val = val + wy1 * wx0 * p10
            w_sum_total = w_sum_total + wy1 * wx0
            val = val + wy1 * wx1 * p11
            w_sum_total = w_sum_total + wy1 * wx1
            val = val + wy1 * wx2 * p12
            w_sum_total = w_sum_total + wy1 * wx2
            val = val + wy1 * wx3 * p13
            w_sum_total = w_sum_total + wy1 * wx3
            val = val + wy2 * wx0 * p20
            w_sum_total = w_sum_total + wy2 * wx0
            val = val + wy2 * wx1 * p21
            w_sum_total = w_sum_total + wy2 * wx1
            val = val + wy2 * wx2 * p22
            w_sum_total = w_sum_total + wy2 * wx2
            val = val + wy2 * wx3 * p23
            w_sum_total = w_sum_total + wy2 * wx3
            val = val + wy3 * wx0 * p30
            w_sum_total = w_sum_total + wy3 * wx0
            val = val + wy3 * wx1 * p31
            w_sum_total = w_sum_total + wy3 * wx1
            val = val + wy3 * wx2 * p32
            w_sum_total = w_sum_total + wy3 * wx2
            val = val + wy3 * wx3 * p33
            w_sum_total = w_sum_total + wy3 * wx3

            val = tl.where(w_sum_total != 0.0, val / w_sum_total, val)
            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

#### L3.8 nearest 通用路径 `nearest_vec_kernel`

```python
@triton.jit
def nearest_vec_kernel(
    input_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    use_size: tl.constexpr,
    scale_h, scale_w,
    BLOCK_W: tl.constexpr,
    MAX_IN_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)
    rows_per_prog = (total_rows + num_programs - 1) // num_programs
    row_start = pid * rows_per_prog
    row_end = row_start + rows_per_prog
    if row_end > total_rows:
        row_end = total_rows

    r_offs = tl.arange(0, MAX_IN_W)
    r_mask = r_offs < W_in

    last_n = -1
    last_c = -1
    last_src_yi = -1
    in_row = tl.zeros((MAX_IN_W,), dtype=tl.float32)

    for row in range(row_start, row_end):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        if use_size == 1:
            src_yi = (h * H_in) // H_out
        else:
            src_y = tl.cast(scale_h, tl.float32) * tl.cast(h, tl.float32)
            src_yi = tl.cast(src_y, tl.int32)
        if src_yi < 0:
            src_yi = 0
        if src_yi >= H_in:
            src_yi = H_in - 1

        input_row_off = (n * C + c) * H_in * W_in + src_yi * W_in
        output_row_off = ((n * C + c) * H_out + h) * W_out

        # 跨行行复用：相邻输出行可能映射到同一输入行
        if (n != last_n) | (c != last_c) | (src_yi != last_src_yi):
            in_row = tl.load(input_ptr + input_row_off + r_offs,
                             mask=r_mask, other=0.0).to(tl.float32)
            last_n = n
            last_c = c
            last_src_yi = src_yi

        for w_start in range(0, W_out, BLOCK_W):
            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out

            if use_size == 1:
                src_xi = (w_idx * W_in) // W_out
            else:
                w_f = tl.cast(w_idx, tl.float32)
                src_x = tl.cast(scale_w, tl.float32) * w_f
                src_xi = tl.cast(src_x, tl.int32)
            src_xi = tl.where(src_xi < 0, 0, src_xi)
            src_xi = tl.where(src_xi >= W_in, W_in - 1, src_xi)

            val = tl.gather(in_row, src_xi, 0)
            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

#### L3.9 area 通用路径 `area_vec_kernel`

```python
@triton.jit
def area_vec_kernel(
    input_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    BLOCK_W: tl.constexpr,
    MAX_KW: tl.constexpr,
    MAX_IN_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)

    r_offs = tl.arange(0, MAX_IN_W)
    r_mask = r_offs < W_in

    for row in range(pid, total_rows, num_programs):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        istartH = (h * H_in) // H_out
        iendH = ((h + 1) * H_in + H_out - 1) // H_out
        kH = iendH - istartH

        input_base = (n * C + c) * H_in * W_in
        output_row_off = ((n * C + c) * H_out + h) * W_out

        for w_start in range(0, W_out, BLOCK_W):
            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out

            istartW = (w_idx * W_in) // W_out
            iendW = ((w_idx + 1) * W_in + W_out - 1) // W_out

            sum_val = tl.full([BLOCK_W], 0.0, tl.float32)
            for ih in range(istartH, iendH):
                row_off = input_base + ih * W_in
                in_row = tl.load(input_ptr + row_off + r_offs,
                                 mask=r_mask, other=0.0).to(tl.float32)
                for iw_offset in range(MAX_KW):
                    iw = istartW + iw_offset
                    valid = (iw < iendW) & mask
                    pixel = tl.gather(in_row, iw, 0)
                    sum_val = tl.where(valid, sum_val + pixel, sum_val)

            kW = iendW - istartW
            kH_f = tl.cast(kH, tl.float32)
            kW_f = tl.cast(kW, tl.float32)
            area = kH_f * kW_f
            val = tl.where(area > 0.0, sum_val / area, sum_val)

            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

### §2.4 Interpolate 性能基准

| Mode | cases | 加速比区间 | 备注 |
|------|-------|-----------|------|
| area (downsample) | 5 | 18x ~ 83x | 极快，部分超 profiler 分辨率 |
| bilinear (downsample) | ~15 | 1.5x ~ 6x | 表现优异 |
| bilinear (upsample, ac=F) | ~10 | 0.28x ~ 1.8x | 中等表现 |
| bilinear (upsample, ac=T) | ~8 | 0.18x ~ 0.6x | 较弱，gather 开销 |
| bicubic (ac=F) | 4 | 0.16x ~ 3.4x | 高方差 |
| bicubic (ac=T) | 3 | 0.07x ~ 1.9x | 高方差 |
| nearest | 8 | 0.29x ~ 6.4x | 中等表现 |
| 全量 73 cases | 73 | 1.3834x | 达标 0.8x 项目目标 |

**关键结论**:
1. 下采样路径是性能优势区（1.5x~83x），输出少、输入行复用充分
2. 上采样路径较弱（0.18x~1.8x），输出多、每行独立计算
3. 多策略分派是核心：不同 mode 和采样比例的最优数据访问模式完全不同
4. 边界 floor 修正对 bicubic ac=False 精度至关重要

---

## §3 Pad 算子（transformation-memory）

**算子类别**: `transformation-memory`
**典型特征**: 数据搬运为主，计算极简（仅边界坐标映射），输出 shape != 输入 shape
**性能基准**: 51 cases 全过，几何平均加速比 **1.68x**（大矩阵可达 18x+）

### §3.1 Layer 1: 设计约束

#### L1.1 必须做多 kernel 分支
（特化分派原则见 §1 G3）
- 通用 4D kernel 的逐元素坐标解码 overhead 极大，仅作为兜底方案
- **必须**为高频场景（2D/3D constant）写特化 kernel

#### L1.2 constant 模式必须拆分为 fill + copy
- **禁止**在 kernel 内逐元素判断 `if in_bounds else fill_value`
- **必须**先 `output.fill_(value)`，再用 copy kernel 搬运有效数据

#### L1.3 Host 侧必须做维度压缩
- **必须**在调用 kernel 前 squeeze 前导 size-1 维度
- 压缩后需同步调整 pad_list 的维度对应关系

#### L1.4 坐标比较必须用 float32
（见 §1 G8）

#### L1.5 禁止硬编码 num_cores
（见 §1 G1）

### §3.2 Layer 2: 算法骨架

#### L2.1 Host 侧分支决策树（伪代码）
```
ndim = squeeze(x) 后的维度
mode = constant/reflect/replicate/circular
if ndim == 2:
    if mode == constant:
        output.fill_(value)
        launch copy_kernel_2d
    else:
        launch pad_kernel_2d  # 逐行边界映射
elif ndim == 3:
    if mode == constant:
        output.fill_(value)
        launch copy_kernel_3d
    else:
        if D0 * D1_out > THRESHOLD:  # THRESHOLD ~ 3000
            launch pad_kernel_3d_nonconstant_v2  # 1D grid
        else:
            launch pad_kernel_3d_nonconstant_2d   # 2D grid
else:
    pad_to_4d()
    launch pad_kernel_4d  # 通用兜底
```

#### L2.2 多核并行骨架模式

**模式 A - 按元素分配（适合通用/1D 场景）**:
```
elements_per_core = cdiv(total_elements, num_cores)
core_start = pid * elements_per_core
core_end = min(core_start + elements_per_core, total_elements)
for block_idx in range(cdiv(core_end - core_start, BLOCK_SIZE)):
    # 处理一个 block
```

**模式 B - 按行分配（适合 2D/3D 场景）**:
```
rows_per_core = cdiv(total_rows, num_cores)
row_start = pid * rows_per_core
row_end = min(row_start + rows_per_core, total_rows)
for row_idx in range(row_end - row_start):
    # 处理一行，内部按 block 遍历列
```

### §3.3 Layer 3: 关键技巧

#### L3.1 坐标压缩公式（constant 模式）
```python
# 将多维坐标压缩为 1D 偏移，避免 kernel 内多维索引计算
coords = tl.arange(0, BLOCK_SIZE)
# 逐维度解码（仅对需要处理的维度）
d0 = coords // stride_d0
d1 = (coords % stride_d0) // stride_d1
# ...
```

#### L3.2 边界映射模板（reflect/replicate/circular）
```python
# reflect 模式：先转 float 比较，再映射
in_coord = out_coord - pad_left
in_coord_f = in_coord.to(tl.float32)
# 处理负边界
in_coord = tl.where(in_coord_f < 0, -in_coord, in_coord)
# 处理超界
in_coord = tl.where(in_coord_f >= in_dim, 2 * (in_dim - 1) - in_coord, in_coord)
```

#### L3.3 2D constant 特化 kernel 结构
```python
# Host 侧：先 fill，再启动 copy kernel
output = torch.empty(...)
output.fill_(fill_value)
# copy kernel：仅搬运有效数据，无边界判断
copy_kernel[grid](input, output, ...)
```

#### L3.4 Block Size Scaling（关键，决定能否过目标加速比）
- Pad 为纯内存搬运算子，**BLOCK cap 从默认 512 提升到 16384 可显著降低 kernel launch 次数、提高访存效率**。
- 实测 51 cases：cap=512 时 geomean ~1.08x；cap=1024 → 1.14x；cap=2048 → 1.20x；cap=4096/8192 → 1.20x；**cap=16384 → 1.25x**；cap=32768 时大 W 形状出现 MLIR 编译错误。
- **推荐策略**：在 verify 不失败的范围内选择最大 cap；典型有效区间为 **2048~16384**，需以实际 case 集合验证上限。
- 注意：cap 作为 `_select_block` 的可调参数，只需在 forward() 中修改变量值即可，无需改动 kernel 内部逻辑。

### §3.4 Pad 性能基准
- 几何平均 **1.68x**，51/51 cases 全过（大矩阵可达 18x+）

---

## §4 Repeat 算子（transformation-memory）

**算子类别**: `transformation-memory`
**典型特征**: 数据搬运为主，计算极简（仅坐标映射），输出 shape != 输入 shape

### §4.0 关键区分（必须先看懂）

Repeat 类算子在 PyTorch 里有两个不同 API，优化策略完全不同：

| API | 语义 | 典型调用 |
|-----|------|---------|
| `torch.repeat` / `Tensor.repeat` | 按**维度**重复整个张量 | `x.repeat(2, 1, 3)` |
| `torch.repeat_interleave` | 沿**单一维度**，把每个元素重复 N 次 | `x.repeat_interleave(2, dim=-1)` |

**性能基准**：
- `torch.repeat`: 几何平均加速比 **0.88x**，49/49 cases 全过（`repeat_v2_20260526`）
- `repeat_interleave`: 几何平均加速比 **1.50x**，75 cases 全过

> ⚠️ **禁止混淆**: Layer 2 以下方案按算子类型选择。`torch.repeat` 不要套用 `repeat_interleave` 的逐元素展开思路。

### §4.1 Layer 1: 设计约束

#### L1.1 禁止单 kernel 展平多维 repeat
（特化分派原则见 §1 G3）
- **禁止**用单一 1D kernel 处理多维 `torch.repeat`（如 `Tensor.repeat([2,3,4])`）
- 单 kernel 展平会导致坐标解码 overhead 极大，且无法利用多维并行
- **必须**按维度逐层串行处理，每层一个 kernel
- **注意**：`repeat_interleave(input, repeats, dim)` 只沿单一维度重复，不在此限制范围内

#### L1.2 必须使用 constexpr 循环展开
- **必须**将 repeat 次数声明为 `tl.constexpr`

#### L1.3 禁止在 kernel 内对向量做整数除法/取余
- **禁止**在 Triton kernel 内对 `tl.arange` 产生的向量做 `//` 或 `%` 运算
- Ascend NPU 上 int32 向量除法退化为**标量循环**，BLOCK=1024 时每个 block 执行 1024 次标量除法
- 正确方案：用标量索引做除法（如 `block_idx // num_inner_blocks`），或对连续内存块直接拷贝

#### L1.4 多核分配必须按输出元素
（见 §1 G6）
- **必须**按输出元素总数分配核数，而非按输入元素
- 每个 program 处理一段连续的输出元素

#### L1.5 输入必须 contiguous
（见 §1 G7）

### §4.2 Layer 2: 算法骨架

#### L2.0 子类型判定

根据任务描述判断：
- 若语义是 `Tensor.repeat(repeats)` → 走 **L2.1 torch.repeat 推荐架构**（outer×inner 连续块）
- 若语义是 `repeat_interleave(input, repeats, dim)` → 走 **L2.4 repeat_interleave 架构**

#### L2.1 torch.repeat 推荐架构：outer×inner 连续块路径（优先）

这是经过验证的**最优架构**（`repeat_v2_20260526`，0.88x）。核心思想：把 repeat 问题转换为"按 outer_idx 加载一段连续 inner_size 数据，然后连续写 r 次"。

**Host 侧流程**：

```python
def forward(self, x, repeats):
    x = x.contiguous()
    shape = list(x.shape)
    ndim = len(shape)

    # 将 repeats 扩展到与 ndim 相同长度（前面补 1）
    repeats = [1] * (ndim - len(repeats)) + list(repeats)
    out = x

    # 从最低维到最高维逐维度处理
    for dim_idx in range(ndim - 1, -1, -1):
        r = repeats[dim_idx]
        if r <= 1:
            continue

        outer_size = math.prod(shape[:dim_idx])
        inner_size = out.numel() // outer_size

        # 选择 BLOCK：按 inner_size 向上取 2 的幂，目标使 num_inner_blocks 尽量小
        BLOCK = _get_block_size(inner_size)
        num_inner_blocks = (inner_size + BLOCK - 1) // BLOCK
        total_blocks = outer_size * num_inner_blocks

        # 构造输出 tensor
        out_shape = list(out.shape)
        out_shape[dim_idx] *= r
        output = torch.empty(out_shape, dtype=out.dtype, device=out.device)

        # Grid 分发
        if total_blocks <= VEC_CORE_NUM:
            grid = (outer_size, num_inner_blocks)   # 2D grid
        else:
            grid_cores = total_blocks if total_blocks < VEC_CORE_NUM else VEC_CORE_NUM
            grid = (grid_cores,)                     # 1D grid，按 core 负载均衡

        launch_kernel(out, output, outer_size, inner_size, num_inner_blocks,
                      r=r, BLOCK=BLOCK, num_cores=grid_cores)

        out = output
        shape[dim_idx] *= r

    return out
```

**Kernel 侧（两种 grid 模式）**

**模式 A：Small Grid（2D grid）**

```python
@triton.jit
def repeat_small_kernel(x_ptr, out_ptr, inner_size, r: tl.constexpr, BLOCK: tl.constexpr):
    outer_idx = tl.program_id(0).to(tl.int32)
    local_block = tl.program_id(1).to(tl.int32)

    block_start = local_block * BLOCK
    offs = (block_start + tl.arange(0, BLOCK)).to(tl.int32)
    mask = offs < inner_size

    in_offset = outer_idx * inner_size
    val = tl.load(x_ptr + in_offset + offs, mask=mask)

    # r 为 constexpr，编译期展开
    for repeat_idx in range(r):
        out_offset = outer_idx * inner_size * r + repeat_idx * inner_size
        tl.store(out_ptr + out_offset + offs, val, mask=mask)
```

**模式 B：Large Grid（1D grid + 标量循环分配）**

```python
@triton.jit
def repeat_large_kernel(x_ptr, out_ptr, outer_size, inner_size, num_inner_blocks,
                        r: tl.constexpr, BLOCK: tl.constexpr, num_cores: tl.constexpr):
    pid = tl.program_id(0).to(tl.int32)
    total_blocks = outer_size * num_inner_blocks

    blocks_per_core = total_blocks // num_cores
    remainder = total_blocks - blocks_per_core * num_cores

    if pid < remainder:
        my_blocks = blocks_per_core + 1
        start_block = pid * (blocks_per_core + 1)
    else:
        my_blocks = blocks_per_core
        start_block = remainder * (blocks_per_core + 1) + (pid - remainder) * blocks_per_core

    for block_idx in range(start_block, start_block + my_blocks):
        outer_idx = block_idx // num_inner_blocks          # 标量除法
        local_block = block_idx - outer_idx * num_inner_blocks

        block_start = local_block * BLOCK
        offs = (block_start + tl.arange(0, BLOCK)).to(tl.int32)
        mask = offs < inner_size

        in_offset = outer_idx * inner_size
        val = tl.load(x_ptr + in_offset + offs, mask=mask)

        for repeat_idx in range(r):
            out_offset = outer_idx * inner_size * r + repeat_idx * inner_size
            tl.store(out_ptr + out_offset + offs, val, mask=mask)
```

**为什么这个架构快？**

1. **输入输出都连续**
   - 输入：每个 program 加载 `inner_size` 中一段连续数据
   - 输出：`outer_idx * inner_size * r + repeat_idx * inner_size + offs`
   - r 个副本在输出内存中**紧挨着**，无跨步写

2. **Grid 并行度高**
   - `total_blocks = outer_size * num_inner_blocks`
   - 2D grid 或按 core 数分发，充分利用多核

3. **无向量除法**
   - 所有除法/取余都是标量（`block_idx // num_inner_blocks`）
   - `block_idx` 是 program 内标量，不触发向量降级

4. **从低维到高维处理**
   - 每次只扩展当前最低维
   - 保持 `inner_size` 是一段逻辑上连续的数据

#### L2.2 torch.repeat 备选架构：lastdim + slice 路径

当实现环境受限，或外层循环已固定为高维到低维时，可作为备选。但**不是首选**，性能通常不如 L2.1。

**Lastdim 路径（零除法）**

```python
val = tl.load(in_ptr + in_offs, mask=in_mask, other=0.0)
val_2d = tl.broadcast_to(tl.reshape(val, (BLOCK_IN, 1)), (BLOCK_IN, repeats))
out_val = tl.reshape(val_2d, (BLOCK_IN * repeats,))
tl.store(out_ptr + out_offs, out_val, mask=out_mask)
```

**Slice 路径（标量除法）**

```python
for s in range(SLICES_PER_BLOCK):
    slice_idx = block_start_slice + s
    if slice_idx < num_out_slices:
        in_slice_idx = slice_idx // repeats          # 标量除法
        out_base = slice_idx * post_size
        in_base = in_slice_idx * post_size
        for p_start in range(0, post_size, BLOCK_POST):
            p_off = p_start + tl.arange(0, BLOCK_POST)
            val = tl.load(in_ptr + in_base + p_off, ...)
            tl.store(out_ptr + out_base + p_off, val, ...)
```

> ⚠️ **注意**：slice 路径的缺点是同一个输入 slice 的多个副本在输出内存中**不连续**（间隔 `DIM_SIZE * POST_SIZE`），会导致跨步写。只有在 `post_size` 很小或 `repeats=1` 时才能接受。

#### L2.3 repeat_interleave 架构

`repeat_interleave(input, repeats, dim)` 只沿单一维度重复，可直接用一个 kernel 处理。

```python
# 输出坐标 out_idx 映射到输入坐标 in_idx = out_idx // repeats
# 或对于可变 repeats，先前缀和再二分查找
```

### §4.3 Layer 3: 关键技巧

#### L3.1 BLOCK 选择公式

```python
def _get_block_size(inner_size: int) -> int:
    """按 inner_size 向上取 2 的幂，目标使 num_inner_blocks 尽量小。"""
    if inner_size <= 64:   return 64
    if inner_size <= 128:  return 128
    if inner_size <= 256:  return 256
    if inner_size <= 512:  return 512
    if inner_size <= 1024: return 1024
    if inner_size <= 2048: return 2048
    if inner_size <= 4096: return 4096
    return 8192
```

原则（pow2 BLOCK 见 §1 G2）：
- BLOCK 不超过 inner_size 太多（避免 mask 浪费）
- num_inner_blocks 尽量小，减少 kernel 内循环次数

#### L3.2 Grid 选择阈值

```python
VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)

if total_blocks <= VEC_CORE_NUM:
    grid = (outer_size, num_inner_blocks)   # 2D grid，每个 block 处理一个 outer × inner_block
else:
    grid_cores = min(total_blocks, VEC_CORE_NUM)
    grid = (grid_cores,)                     # 1D grid，按 core 负载均衡
```

#### L3.3 int32 索引
（见 §1 G5）

#### L3.4 负载均衡公式
（见 §1 G6）

### §4.4 Repeat 性能基准
- `torch.repeat`: **0.88x** 几何平均，49/49 cases 全过
- `repeat_interleave`: **1.50x** 几何平均，75 cases 全过

---

## §5 常见陷阱与避免方法

### §5.1 Interpolate 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 用单 kernel 统一处理多 mode | 所有 mode 走同一套逻辑，丢失特化机会 | 多策略分派 (L1.1)，按 (mode, ac, 采样比例) 选择特化 kernel |
| W_in 非 2 的幂导致 dynamic-shape load | `tl.arange(0, W_in)` 当 W_in 非 constexpr 退化为标量循环 | `MAX_IN_W = next_power_of_2(W_in)` (L1.2) |
| 整数倍上/下采样用 gather | 2x 上采样/0.5x 下采样用 gather 实现是随机访问，远慢于 contiguous | 2x 用 reshape broadcast (L1.3/L3.1)，0.5x 用 strided load (L1.4/L3.2) |
| bilinear 0.5x 下采样走通用路径或 strided load | ac=False + 0.5x 下采样时 bilinear 退化为 2x2 avg pool，走通用 gather 路径浪费；kernel 内用 strided load 也无法 coalesce | 特化分派 (L1.5) + 一次 contiguous load `2*BLOCK_W` + reshape + sum (L3.3) |
| 上采样逐行独立 load 输入 | 上采样时多个输出行共享输入行，逐行 load 重复访存 | 2D 垂直分块 BLOCK_H=2、MAX_KH=3 (L1.6/L3.5) |
| ac=True kernel 内重新计算坐标 | PyTorch C++ 的 ac=True 坐标用 float32 计算，kernel 内重算有精度差异，verify 失败 | host 用 numpy.float32 预算坐标和权重 (L1.7/L2.2) |
| bicubic 16 项向量化累加 | 向量化累加改变顺序，与 PyTorch 标量顺序不一致，verify 失败 | 标量逐项累加 `val = val + w*pixel` (L1.7/L3.7) |
| 三次权重用 if 分段 | `if at < 1.0: ...` 在 vector 数据上不生效 | `tl.where` 嵌套 (L1.8/L3.6) |
| grid_size 超核数 | 上采样 total_rows 极大，直接 `grid=(total_rows,)` 超上限或空跑 | `grid = (min(total_rows, num_cores),)` (L1.9) |
| 低精度 ac=True 用 CPU 计算 scale | CANN vdiv 与 CPU float32 除法有微小差异，低精度 ac=True verify 不过 | `_npu_scale` (NPU vdiv) + `_SCALE_CACHE` 缓存 (L1.10)，仅低精度 ac=True 需要 |
| ac=False 负坐标 floor 错误 | `src_y = scale*(i+0.5)-0.5` 可能为负，`tl.cast` 向零截断导致 t 偏移错误 | `tl.where(src_y < y0_f, y0 - 1, y0)` 修正，并重新计算 y0_f 再算 t (L1.11/L3.6) |
| 循环携带变量类型不一致 | kernel 内同一变量在循环外为标量、循环内被赋值为 vector，Triton 报 `Loop-carried variable ... initial type ... but re-assigned to <[N], fp32>` | 循环内临时变量用不同名字，或确保首次赋值就是 vector。典型出现在 bicubic x 权重计算（用 `at_x/at2_x/...` 而非复用 `at/at2/...`） |
| 嵌套函数定义 | `@triton.jit` kernel 内定义嵌套函数会报 `nested function definition is not supported` | 把工具函数内联到 kernel 中（如 `_cubic_weight` 直接展开） |
| forward 中调用 bare `min()` 或 tensor 方法 | `validate_triton_impl.py` Type-3 检查会把 `min(a, b)` 或 `self.xxx()` 视为 PyTorch fallback | `min(a, b)` 改为 `a if a < b else b`；`_precompute_*` 改为模块级函数而非类方法 |
| profiler 失败误判为代码错误 | 部分大 shape 报 `RuntimeError: 无法从 profiler 提取有效时延数据` | 这是 CANN profiler 分辨率/环境限制，与代码正确性无关；verify 阶段 73/73 全部通过即可 |

### §5.2 Pad 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 单 kernel 处理所有 mode | 通用 kernel 坐标解码 overhead 大 | 按 mode 分 kernel (L1.1) |
| kernel 内逐元素判断边界 | 每个元素都需分支，性能差 | constant 模式拆 fill+copy (L1.2) |
| 硬编码 num_cores | 不同 NPU 核数不同 | 动态读取 (L1.5/G1) |
| 整数坐标直接比较 | Triton Ascend 整数比较可能降级 | 转 float32 再比较 (L1.4/G8) |
| 未 squeeze 前导维度 | 前导 size-1 维度浪费 grid | Host 侧先 squeeze (L1.3) |

### §5.3 Repeat 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 向量除法 `offsets // repeats` | int32 向量除法退化为标量循环 | 用 outer×inner 连续块路径，除法只对 `block_idx` 标量做 (L1.3) |
| 输出跨步写 | slice 路径中同一份数据写到相距很远的输出位置 | 用 L2.1 outer×inner 路径，保证 r 个副本连续 |
| Grid 并行度过低 | SLICES_PER_BLOCK=1 导致 block 数太少 | 用 2D grid 或按 core 数分发 (L3.2) |
| 从高维到低维处理 | 中间张量变大，post_size 过大 | 用 L2.1 从低维到高维处理 |
| 混淆 repeat_interleave 和 torch.repeat | 两者语义不同，优化策略不同 | 先判定算子类型，再选 L2.1 或 L2.3 (§4.0) |
| `for r in range(repeats): store` 输出不连续 | constexpr 展开但地址跨步 | 确保输出地址连续，或用 reshape 代替 |

