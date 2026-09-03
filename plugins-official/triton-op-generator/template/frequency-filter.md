---
name: frequency-filter
description: 频域滤波类算子（rfft2 → 逐元素复数变换 → irfft2，如 GlobalFilter）的纯 Triton DFT 优化经验。业务约束：禁用 torch.fft。
metadata:
  type: reference
---

# Frequency-Filter 类算子优化经验

适用于 **FFT 变换 → 逐元素复数变换 → 逆 FFT** 形态的算子。典型代表：GlobalFilter。

**业务约束**：`torch.fft.rfft2` / `torch.fft.irfft2` / 所有 `torch.fft.*` 接口**禁止在 forward 或 helper 中出现**。整个 2D-DFT/IDFT 必须纯 Triton 实现。

经验基于 op_49_GlobalFilter 在 ascend910B1 上的完整探索（50 cases 混合 dtype，最终 geomean 2.3715x）。

> Layer 1 为硬性约束，后续同类算子生成时由 `triton-op-designer` 作为 negative_prompt 遵守；Layer 2-3 为骨架与技巧参考，禁止直接复制代码结构。

---

## §1 Layer 1：设计约束（必须遵守）

### F1 权重 RNG 的 key dtype 必须恒为 float32

若参考实现先对输入 `x.view(...).float()` 再生成 hash key，则缓存权重的 key 中 dtype 必须用 **`torch.float32`**，不能用输入原始 dtype。否则 fp16/bf16 与 fp32 的 seed 不同，权重完全不同，导致仅 fp32 通过。

### F2 完全禁用 torch.fft

- `forward()` 及所有模块级 helper 中**禁止调用** `torch.fft.rfft2` / `torch.fft.irfft2` / `torch.fft.fft` / `torch.fft.ifft` 等任何 `torch.fft.*` 接口。
- 2D-DFT 与 2D-IDFT 必须拆分为可分离的 1D-DFT/IDFT，并在 `@triton.jit` kernel 内用 `tl.dot` 实现。
- 频域逐元素变换（如滤波器复数乘）也必须封装为 `@triton.jit` kernel，complex64 按 (real, imag) 两个 float32 处理，禁止在 forward 内用 torch 算子完成。

### F3 纯 Triton 可分离 2D-DFT 骨架

必须采用以下流水线：

- w 轴前向 DFT
- h 轴前向 DFT + 频域滤波器复数乘
- h 轴逆 DFT
- w 轴 Hermitian 逆 DFT

所有矩阵乘法在 `@triton.jit` 内用 `tl.dot(..., input_precision="ieee")` 完成；E-table、滤波权重按 shape + device 缓存。

### F4 禁止 host 侧三角函数生成 E-table

E-table 的 sin/cos 必须在 `@triton.jit` kernel 内通过 `tl.cos` / `tl.sin` 完成。禁止 host 侧使用 `numpy.cos` / `numpy.sin` / `torch.cos` / `torch.sin` / `tensor.cos()` / `tensor.sin()` 生成 kernel 输入表。host 侧仅允许分配空 buffer 并缓存已由 Triton kernel 填充好的结果。

---

## §2 Layer 2：算法骨架

流程：可分离 2D-DFT 矩阵乘法流水线。

- 4 个 1D-grid compute kernel 顺序启动：w 轴 DFT → h 轴 DFT + 滤波乘 → h 轴 IDFT → w 轴 Hermitian IDFT。
- grid 按 `(B, 轴长度, C//BC)` 展开，`BC=32` 为经验 tile。
- 复数采用 planar 布局（实部/虚部分开连续缓冲），避免 complex64 stride-2 写。
- E-table 用 1–3 个小型 init kernel 填充并缓存；4 个主 kernel 读取缓存表执行 `tl.dot`。
- 输入零填充到 2 的幂（`PH/PW/PRW = next_power_of_2`），kernel 内通过 mask 忽略填充区。

---

## §3 Layer 3：关键技巧

### F-T1 planar 复数布局

纯 Triton DFT 中，实部/虚部分开放在两个连续缓冲区。相比交错 complex64，planar 布局使 `tl.dot` 输出写入 stride-1，且对 fp32 dot 更友好。

### F-T2 缓存 permuted 权重

参考权重通常为 `(h, rw, C, 2)` 交错布局。在 h 轴 DFT + 滤波乘的 kernel 中，将其 permute 为 planar `(rw, h, C)` 的实/虚两个 buffer 更利于 load。`W.permute(...).contiguous()` 每次调用都会拷贝 M 元素，应按 `(dim,h,rw,device,float32)` 缓存；key 中的 dtype 必须 float32（见 F1）。

### F-T3 Triton init kernel 填充并缓存 E-table

将 E-table 的 sin/cos 计算放在 1–3 个小型 `@triton.jit` init kernel 中，结果写入全局内存 buffer 并缓存；4 个主 compute kernel 直接 load 缓存表。这样避免每个 program 重复计算小尺寸 E-table，性能优于把 E-table 内联到 4 个主 kernel，同时满足 F4 的 host 侧无三角函数约束。

### F-T4 Hermitian IDFT 权重

w 轴逆变换需恢复实输出，IDFT 矩阵前乘 Hermitian 权重：DC 分量为 1.0，内部频率为 2.0；w 为偶数时 Nyquist 分量也为 1.0。

---

## §4 性能基准（op_49_GlobalFilter, ascend910B1, 50 cases 混合 dtype）

| 指标 | Phase 3 baseline | 纯 Triton DFT（Triton init kernel + 缓存 E-table） |
|------|------------------|---------------------------------------------------|
| 几何平均加速比 vs torch | 0.4934 | **2.3715** |
| speedup_vs_baseline | 1.0 | **4.8064** |
| 生成实现平均延迟 | — | 0.1555 ms |
| PyTorch 标杆平均延迟 | — | 0.3689 ms |
| 大 case (C=512,h=29,w=23) | <1x | ~9.5x |
| B=4 小 M case | ~0.58x | ~2.0x |
| 精度 (passed/total) | 50/50 | 50/50 |

- 纯 Triton DFT 通过 `tl.dot` 将 DFT 矩阵乘法 offload 到 NPU，大 shape 可获得数倍以上加速。
- 极小 shape 受 4 个 compute kernel + 3 个 init kernel 的 launch 固定开销影响，相对大 shape 收益下降。
- 用 Triton init kernel 填充并缓存 E-table 可同时满足 host 侧无三角函数约束与高性能（本例 2.3715x）。

---

## §5 常见陷阱

| 陷阱 | 症状 | 修复 |
|------|------|------|
| 权重 key 用输入 dtype | 仅 fp32 通过 | key dtype 恒 float32（F1） |
| forward 内调用 torch.fft.* | AST/业务违规，必须失败 | 改为纯 Triton 可分离 DFT（F2, F3） |
| forward 内 torch 计算 / while / self.method | AST 退化失败 | 模块级 helper + `@triton.jit` kernel（F2） |
| forward 内 torch.randn/manual_seed | validator 禁止 | 移入模块级函数并缓存 |
| BLOCK 过大 | MLIR PlanMemory Failed | BC=32， contracting dim 取 1024 左右 |
| 输入未零填充到 2 的幂 | 非 2 幂 contracting dim 编译/数值错 | `PH/PW/PRW = next_power_of_2` 并 mask |
| E-table 用 host 侧 numpy/torch 三角函数 | 违反 F4 | 用 Triton init kernel 填充并缓存（F-T3） |
| E-table 内联到 4 个主 kernel | geomean 从 2.37x 降至 1.66x | 改用 init kernel + 缓存（F-T3） |
| 复数 `tl.dot` 输出写入交错 complex64 | store 非连续 | 改用 planar 双缓冲区（F-T1） |
| grid 解码用 `r - r//T*T` | c0 tile 偏移错乱 | 必须 `t = pid - r * T` |
| benchmark 出现 >5x speedup | framework 计时 fluke | rerun 复核，禁直接采信 |
