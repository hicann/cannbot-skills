---
name: activation
description: Activation 逐元素算子（GELU / SiLU / SwiGLU）的 Triton Ascend 优化经验
metadata:
  type: reference
---

# Activation 逐元素算子类别经验（GELU / SiLU / SwiGLU）

> 类别: `activation` — NPU 上的逐元素非线性激活函数
> 锚定算子: `_gelu_fwd_kernel` (op14) / `_swiglu_fwd_kernel` (op16) / `_silu_fwd_kernel` (op15)
> 性能参考: Ascend950PR / `ascend910_9382`
> - GELU: 30 cases (10 shapes × fp16/bf16/fp32)，最佳几何平均 **0.7969×**（目标 1.2×，**未达**），全 30 case 精度通过
> - SwiGLU: 30 cases，最佳几何平均 **1.3096×**（目标 1.2×，**已达** ✅），全 30 case 精度通过
> - SiLU: 30 cases，最佳几何平均 **1.2669×**（目标 1.2×，**已达** ✅），CANN kernel-only 复测，全 30 case 精度通过

---

## Layer 1: 设计约束（硬性规则，GELU / SiLU / SwiGLU 通用）

### L1.1 数学公式必须与 torch 参考严格一致（强制）
- **GELU**: `y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))`（exact erf 形式）。tanh 近似仅在参考明确使用 `approximate='tanh'` 时才允许。
- **SiLU**: `y = x * sigmoid(x)`，fp32 计算 `x_f32 * tl.sigmoid(x_f32)` 后 cast 回输入 dtype。
- **SwiGLU**: `c = silu(a) * b = a * sigmoid(a) * b`。

### L1.2 逐元素激活必须 1D 展平 + 连续访存（强制）
- **必须**: 将输入 reshape 为 1D（`x.reshape(-1)`），启动 1D grid，每个 program 以 grid-stride 循环处理连续 BLOCK 个元素。
- **禁止**: 2D 索引、双层 mask、跨步 tile 循环。逐元素算子没有空间局部性收益，2D tiling 只增加索引开销。
- **实测**: GELU 2D tiling **0.6665×** vs 1D **0.7969×**；SwiGLU 2D **1.1347×** vs 1D **1.3096×**。

### L1.3 动态 grid 钳制核数（强制）
- **必须**: `grid_size = min(triton.cdiv(numel, BLOCK), num_vectorcore)`，小 tensor 不启动满核 grid。
- **禁止**: 固定 `grid = (num_vectorcore,)`，小 tensor 浪费 launch/scheduling 开销。

### L1.4 fp32 计算，存回原始 dtype（强制）
- **必须**: 加载后 cast 到 `tl.float32` 再做数学运算，存回前 cast 回输入 dtype。
- **原因**: fp16/bf16 中间精度不足会导致与 torch fp32 参考的累积误差。

### L1.5 compiler hints 按算子独立判断（强制，不可一刀切）
- GELU/SwiGLU: 使用 `multibuffer=True, unit_flag=True`
- SiLU: **禁止**使用 `multibuffer/unit_flag` —— 实测这些 hints 对小 kernel 的 per-launch 开销反而拖累性能（0.927→0.945× 提升来自去掉 hints）
- **铁律**: 每个算子独立实测，不可把 GELU 的 hint 结论直接套到 SiLU/SwiGLU

---

## Layer 2: 算法骨架（1D grid-stride 逐元素激活）

```python
import triton, triton.language as tl
_NUM_VCORE = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": b}) for b in (512,1024,2048,4096,8192,16384,32768)],
    key=["numel"],
)
@triton.jit
def _activation_fwd_kernel(x_ptr, y_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_progs = tl.num_programs(axis=0)
    for off_start in range(pid * BLOCK_SIZE, numel, num_progs * BLOCK_SIZE):
        off = off_start + tl.arange(0, BLOCK_SIZE)
        mask = off < numel
        x_val = tl.load(x_ptr + off, mask=mask, other=0.0)
        x_f32 = x_val.to(tl.float32)
        y_f32 = compute_fn(x_f32)          # GELU / SiLU / SwiGLU 替换此处
        tl.store(y_ptr + off, y_f32.to(x_val.dtype), mask=mask)

def activation_impl(x, *args):
    ori_shape = x.shape
    x_flat = x.reshape(-1) if x.is_contiguous() else x.contiguous().reshape(-1)
    numel = x_flat.numel()
    y = torch.empty_like(x_flat)
    grid = (min(triton.cdiv(numel, 512), _NUM_VCORE),)
    _activation_fwd_kernel[grid](x_flat, y, numel)  # GELU/SwiGLU 加 multibuffer=True,unit_flag=True；SiLU 不加
    return y.reshape(*ori_shape)
```

> BLOCK autotune 范围 512-32768，仅以 `numel` 为 key。`num_vectorcore` 模块级缓存避免逐次 driver 查询。

---

## Layer 3: 关键技巧与已验证变体

### L3.1 动态 grid + compiler hints（GELU: +5.4%）
- 将固定 `grid=(num_vectorcore,)` 替换为 `min(cdiv(numel, BLOCK), num_cores)`
- 添加 `multibuffer=True, unit_flag=True`
- 效果: GELU geomean 0.7561× → 0.7969×

### L3.2 1D 展平 + BLOCK autotune + hints（SwiGLU: 1.3096×）
- autotune `BLOCK_SIZE=[512-65536]`（8 个档位），key=`numel`
- 加 `multibuffer=True, unit_flag=True`
- 中等 shape（999×9999, 1024×10240, 5336×3584）达 **1.5-2.1×**
- **关键差异**: SwiGLU 双输入使 load 带宽翻倍，1D 连续访存的收益比单输入更大（SwiGLU 1D/2D 差 0.17× vs SiLU 差 0.02×）

### L3.3 去掉 hints 的 SiLU（1.2669×，kernel-only）
- autotune + 无 hints + 缓存 num_vectorcore
- npu.Event（op 级，含 host wrapper 开销）显示 0.945×（比原始差），但 **CANN profiler kernel-only 显示 1.2669×**
- **教训**: 原始 host wrapper 更精简，op 级 Event 会误判；必须以 CANN kernel-only 为准
- 中型 shape fp16/bf16 达 2.3-2.49×

### L3.4 Tail block 特化 —— 无效
- 将满 block（无 mask load/store）与 tail block（有 mask）拆分 → GELU 0.7923× < 统一 mask 版
- **结论**: 逐元素激活用统一 mask 更简单、更快；tail 特化的分支开销抵消向量化收益

### L3.5 BLOCK_SIZE 扫描 —— 平坦
- GELU 上 BLOCK=4096/8192/16384 几何平均几乎相同
- 默认 `BLOCK=4096`

### L3.6 fp32 路径接近持平
- fp32 shape 普遍接近 0.98×（与 baseline 持平），fp16/bf16 大 shape 是主要劣化来源
- 若后续优先优化 fp16/bf16，可考虑 dtype 特化路径或 autotune BLOCK

---

## §4 常见陷阱与避免方法

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 小 tensor 用固定满核 grid | 浪费 launch/scheduling 开销，小 shape 上 baseline 反超 | L1.3 动态钳制核数 |
| 2D tile 双层 mask（含把单输入 2D 模式复制到双输入） | 跨步访存 + 额外 mask 运算；双输入 load 带宽翻倍时更伤 | L1.2 一律 1D 展平 |
| Tail block 特化拆分 | 标量分支开销抵消向量化收益 | L3.4 统一 mask |
| tanh 近似对 exact erf 参考 | 精度不匹配 | L1.1 仅参考用 tanh 时允许 |
| 混用单/双输入算子的 hint 结论、或统一加 multibuffer/unit_flag | 单/双输入访存模式不同；hints 对小 kernel 有 per-launch 开销 | L1.5 逐算子独立实测 |
| 信 op 级 npu.Event 为最终指标 | 含 host wrapper 开销，原始 wrapper 更精简 | L3.3 以 CANN kernel-only 为准 |
| 伪造几何平均聚合数 | per-case 全部 <0.5 却报 1.33×（mojo_0729 op15 session 数据作废） | 必须逐 case 校验一致性 |
