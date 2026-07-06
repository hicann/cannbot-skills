---
name: normalization
description: 正则化计算类算子（GroupNormSwish / AdaptiveInstanceNorm2DBackward）的 Triton Ascend 优化经验合集，按算子分章节组织，含通用经验 + 各算子专属约束/骨架/kernel
metadata:
  type: reference
---

# 正则化计算类算子优化经验

本文档合并了两类正则化算子的优化经验。按以下结构组织：

- **§1 通用经验**：跨算子重复的正则化工程约束（已提取，各算子章节不再重复）
- **§2 GroupNormSwish**（norm-fused，前向融合归一化+激活）
- **§3 AdaptiveInstanceNorm2DBackward**（norm-backward，反向多粒度归约）
- **§4 各算子常见陷阱**

> ℹ️ **与其他 template 的关系**：本文 §1 编号为 **N1-N6**（Normalization 专属）。tensor-transform 的 G1-G8（动态核数 / pow2 BLOCK / 多策略分派 / grid 钳制核数 / int32 索引 / 负载均衡公式 / contiguous / 坐标 float32 比较）、transformer-inference 的 T1-T6、math-compute 的 M1-M6 同样适用，但**不在本文重复**；生成时若涉及这些通用约束，请交叉引用对应 template，本文仅补充正则化类特有的新约束。

---

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| GroupNormSwish | `norm-fused` (forward) | 分组归一化（mean/var/rstd 归约）+ weight/bias 仿射 + Swish 激活融合，单次前向 | 单 kernel fused 两 pass（reduce + apply）+ 标量 load weight/bias + constexpr 入参静态化 |
| AdaptiveInstanceNorm2DBackward | `norm-backward` (backward) | 反向需同时产出 per-(N,C) 的 grad_input 和 per-channel 的 grad_weight/grad_bias，两类归约粒度不同 | 双 kernel 两阶段（reduce + apply）+ fp32 partial buffer + atomic_add 跨 tile 归约 |

> ⚠️ **关键区分**：GroupNormSwish 属 **前向融合**类（关心 reduce 与 elementwise 的单 kernel 融合，避免双 kernel launch 开销），AdaptiveInstanceNorm2DBackward 属 **反向多粒度**类（关心两类不同粒度归约的拆分与跨 tile 合并）。两类优化哲学**相反**，生成时**禁止混用经验**：
> - 生成 GroupNormSwish 时，**不要**套用 AdaIN backward 的双 kernel 拆分思路（前向融合应优先单 kernel）
> - 生成 AdaIN backward 时，**不要**套用 GroupNormSwish 的单 kernel fused 思路（反向两类粒度归约无法在单 kernel 内高效合并）
> - 判定依据：算子是 `forward` 且输出含 mean/rstd 统计量 → 走 §2；算子是 `backward` 且同时输出 grad_input + grad_weight/grad_bias → 走 §3

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下 6 条约束在两个正则化算子中均适用，各算子章节不再重复。

### N1 统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加

- **必须**在 kernel 内将输入升精度到 `tl.float32` 后再做 sum / sq_sum / var 归约。
- **正确做法**：
  ```python
  sum_acc = tl.full((), 0.0, tl.float32)
  sq_acc = tl.full((), 0.0, tl.float32)
  x = tl.load(...).to(tl.float32)
  sum_acc += tl.sum(x, axis=0)
  sq_acc += tl.sum(x * x, axis=0)
  ```
- **Why:** 正则化统计量（mean/var/rstd/inv_std^3）涉及小量相乘、求和、除法，fp16/bf16 累加会在小 spatial size 或数值接近 0 时产生显著精度误差，导致 verify 失败。GroupNormSwish 的 var 公式 `(sq - sum*sum/N)/N` 对累加顺序敏感；AdaIN backward 的 `inv_std^3`、`1/S` 等小量乘法更容易下溢。

### N2 跨 tile 归约必须显式两阶段（reduce-then-apply），单 pass 无法获得全局统计量

- **必须**将归一化拆为 **Pass 1 reduce（统计 mean/var 或 partial sums）→ Pass 2 apply（用聚合后的统计量做归一化）** 两阶段。
- **允许两种实现形式**：
  - **单 kernel fused（GroupNormSwish 适用）**：同一 `@triton.jit` 内先 reduce 循环再 apply 循环，统计量存寄存器（标量 `sum_acc/sq_acc`），无中间 HBM 流量。
  - **双 kernel split（AdaIN backward 适用）**：reduce kernel 写 partial sums 到 HBM buffer，apply kernel 读回聚合结果。仅当统计量需要跨 program 合并时（如反向的多 tile 跨核归约）才用此形式。
- **禁止**：在单个 tile 内直接计算最终 grad_input（grad_input 需要所有 tile 的全局统计和，单 tile 信息不全）。
- **Why:** 归一化的本质是"先归约再仿射"，apply 阶段必须拿到完整的 mean/var 才能正确归一化；反向 grad_input 公式依赖全局 s1/s2/s3 之和，无法在 reduce 阶段一次性算出。

### N3 weight/bias 加载优先标量 load 路径，禁止向量化 channel 索引

- **必须**对 per-channel 的 weight/bias 用标量索引加载（整个 BLOCK 落在同一 channel 内时）。
- **正确做法**：
  ```python
  # 标量 load：BLOCK_SIZE <= D 时整个 block 在同一 channel
  ch = d_start // D                    # 标量除法（d_start 为标量）
  w = tl.load(weight_ptr + group_base + ch)
  b = tl.load(bias_ptr + group_base + ch)
  x_norm = x_norm * w + b
  ```
- **禁止**：对 `tl.arange` 产生的向量做 `//` 或 `%` 计算 channel 索引后向量化 load：
  ```python
  # ❌ 错误：向量化 int 除法，Ascend 上严重 scalar 化
  ch_vec = offsets // D
  w_vec = tl.load(weight_ptr + group_base + ch_vec, mask=mask)
  ```
- **Why:** Ascend NPU 上 int32 向量除法退化为标量循环，BLOCK=1024 时每个 block 执行 1024 次标量除法；标量 load 的 `d_start // D` 是对单标量做除法，无向量降级。当 `BLOCK_SIZE >= D`（如 D=1 或 D 很小）时标量 load 仍正确工作（每次循环加载同一 channel）。
- **前置条件**：标量 load 要求 `BLOCK_SIZE <= D`，需在 host 侧通过缩 BLOCK 策略保证（见各算子 L1 BLOCK 策略章节）。

### N4 跨 tile 归约用 `tl.atomic_add` 作为可靠原语

- **必须**在反向多 tile 场景中，通过 `tl.atomic_add` 把每个 tile 的部分和累加到 per-(N,C) 或 per-channel buffer。
- **正确做法**（AdaIN backward reduce kernel）：
  ```python
  tl.atomic_add(partial_s1_ptr + pid_nc, s1_raw)
  tl.atomic_add(partial_s2_ptr + pid_nc, s2_raw)
  tl.atomic_add(grad_bias_ptr + c, s1_raw)
  tl.atomic_add(grad_weight_ptr + c, s2_raw * inv_std)
  ```
- **Why:** `tl.atomic_add` 是当前 Triton Ascend 上最可靠的跨 tile 归约原语；spatial 维度被拆为多个 tile 后，tile 间必须合并，atomic_add 无需显式同步即可完成跨核累加。
- **注意**：atomic_add 在大 batch 大 channel 场景下可能成为性能瓶颈（见 §4 AdaIN 陷阱 5），需在性能优化阶段单独评估。

### N5 关键长度参数必须 `tl.constexpr` 静态化，触发启动级特化

- **必须**将"单次 kernel 启动后不变"的长度参数声明为 `tl.constexpr`，包括 `BLOCK_SIZE`、`channels_per_group`、`elems_per_group`、`BLOCK_S`、`num_cores` 等。
- **Why:** 这些参数在单次 launch 内是常量，声明为 `tl.constexpr` 可触发 Triton Ascend 编译器的**启动级特化**（launch-level specialization），生成特化代码。实测 GroupNormSwish 从 0.9094x 提升至 1.2549x（+38%）。同时 `tl.arange(0, BLOCK_SIZE)` 要求 BLOCK_SIZE 为编译期常量，否则触发 dynamic-shape load 退化为标量循环。
- **禁止**：将 `channels_per_group` / `elems_per_group` / `BLOCK_SIZE` 作为普通运行时 int 参数传入 kernel。

### N6 kernel 内用 `tl.cdiv`，host 侧用 `triton.cdiv`（禁止混用）

- **必须**在 `@triton.jit` kernel 内部使用 `tl.cdiv` 计算 tile 数量。
- **必须**在 host 侧（`ModelNew.forward()` / grid 计算等）使用 `triton.cdiv` 或 Python 整数除法 `(a + b - 1) // b`。
- **禁止**：在 kernel 内调用 `triton.cdiv`。
- **Why:** `triton.cdiv` 在 JIT 函数中会导致 `ValueError: Did you forget to add @triton.jit ?`，编译期报错。

---

## §2 GroupNormSwish 算子（norm-fused / forward）

**算子类别**: `norm-fused`（分组归一化 + Swish 激活融合）
**典型特征**: 输入 `[N, C, *spatial]`，按 `num_groups` 分组归一化后应用 Swish 激活；输出含归一化结果 + mean + rstd 统计量
**性能基准**: geomean **1.2549x**（50/50 cases 全过，使用标量 load 通用路径 + constexpr 入参静态化）

**本版本核心原则**:
- ✅ **单 kernel fused 架构**（Pass 1 reduce + Pass 2 normalize/weight/bias/swish 合一）
- ✅ **标量 load 路径**（通用，不依赖 weight/bias 值）
- ✅ **constexpr 入参静态化**（channels_per_group / elems_per_group 声明为 tl.constexpr，+38%）
- ❌ **禁止 USE_2D / tl.reshape**（Ascend 9.0.0.beta1 上有 correctness bug 和编译限制）
- ❌ **禁止双 kernel split**（launch 开销大，历史验证双 kernel 最高仅 0.669x）
- ⚠️ **跳过 weight/bias 仅作为 host 侧可选快速路径**，不在 kernel 层面硬编码

### §2.1 Layer 1: 设计约束（Agent 必须遵守，无例外）

#### L1.1 Kernel 架构 —— 单 kernel fused（硬性）

**必须**使用单 kernel fused 架构。一个 `@triton.jit` kernel 内完成:
1. Pass 1: one-pass reduce（sum + sq_sum）
2. Pass 2: normalize → weight → bias → swish → store

**禁止**使用双 kernel split（stats kernel + apply kernel）。

**Why**: 双 kernel 增加一次 launch 开销和中间结果（mean/rstd）的内存流量。历史验证双 kernel 版本最高仅 0.669x，无法达标。

```python
# ✅ 正确: 单 kernel fused
@triton.jit
def group_norm_swish_kernel(...):
    # Pass 1: reduce
    for d_start in range(0, elems_per_group, BLOCK_SIZE):
        ...
    # 计算 mean, rstd
    # Pass 2: normalize + weight + bias + swish
    for d_start in range(0, elems_per_group, BLOCK_SIZE):
        ...

# ❌ 错误: 双 kernel split
@triton.jit
def group_norm_stats_kernel(...): ...
@triton.jit
def group_norm_apply_kernel(...): ...
```

#### L1.2 Grid 维度 —— 1D grid + 交错循环（硬性）

```python
total_groups = N * num_groups
grid_size = min(total_groups, VEC_CORE_NUM)
grid = (grid_size,)
```

Kernel 内:
```python
pid = tl.program_id(0)
num_cores = tl.num_programs(0)
for gid in range(pid, total_groups, num_cores):
    pid_n = gid // num_groups
    pid_g = gid % num_groups
    # ... 处理一个 (batch, group) 对
```

**Why**: 固定 grid 大小为核心数，调度开销最小；交错循环确保负载均衡。

**禁止**:
- 2D grid `(N, num_groups)` —— 当 `N * num_groups > VEC_CORE_NUM` 时调度开销显著
- Kernel 内 `for g in range(num_groups)` 串行处理多个 group

#### L1.3 禁止 tl.reshape（硬性）

**完全禁止**在 kernel 内部使用 `tl.reshape`。

**Why**: Ascend 9.0.0.beta1 上 `tl.reshape` 存在 correctness bug（输出异常值）和编译限制（shape 参数必须是 Python 字面量，不能是 `tl.constexpr` 参数）。

**后果**: 不使用 2D reshape 向量化广播路径，统一使用 1D 标量 load 路径处理 weight/bias。

#### L1.4 BLOCK_SIZE 策略 —— dtype-aware + 最大化（硬性）

```python
if input.dtype == torch.float32:
    MAX_BLOCK = 2048
elif D == 1:
    MAX_BLOCK = 2048
else:
    MAX_BLOCK = 4096

# 最大化 BLOCK_SIZE，cap 到 elems_per_group
block_size = MAX_BLOCK
if block_size > elems_per_group:
    block_size = elems_per_group

# 仅当 D % 16 != 0 时缩 block，强制标量路径
if D > 16 and D < MAX_BLOCK and D % 16 != 0:
    block_size = ((D - 1) // 16) * 16
    block_size = min(block_size, elems_per_group)
    if block_size < 16:
        block_size = elems_per_group
```

**Why**:
- 最大化 BLOCK_SIZE 最大化 UB 利用率
- 仅对 `D % 16 != 0` 缩 block，确保 `BLOCK_SIZE < D` 走标量 load 路径（见 N3）
- 不要对所有 case 缩 block（会导致性能下降和 vector core 异常）

#### L1.5 multibuffer/unit_flag（硬性）

**首次生成时不添加** `multibuffer=True, unit_flag=True`。

**Why**: 在 GroupNormSwish 上实测添加后性能从 1.198x 下降至更低。仅对纯 element-wise 算子有明确收益，Norm 类算子因存在 reduce 和循环，multibuffer 收益不确定。

作为 Phase 4 独立优化点单独测试，若性能下降则移除。

#### L1.6 入参静态化优化（硬性）

**必须**将 `channels_per_group` 和 `elems_per_group` 声明为 `tl.constexpr`。

```python
@triton.jit
def group_norm_swish_kernel(
    ...,
    channels_per_group: tl.constexpr,
    elems_per_group: tl.constexpr,
    ...
):
```

**Why**: 这两个参数在单次 kernel 启动后不变，声明为 `tl.constexpr` 可触发 Triton Ascend 编译器的**启动级特化**（launch-level specialization），生成特化代码。实测从 0.9094x 提升至 1.2549x（提升 38%）。

**禁止**: 将 `channels_per_group` 和 `elems_per_group` 作为普通运行时参数传入。

### §2.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧代码骨架

```python
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except Exception:
            self.VEC_CORE_NUM = 40

    def forward(self, input, num_groups, weight, bias, eps=1e-5, swish_scale=1.0):
        N = input.shape[0]
        C = input.shape[1]
        D = 1
        for i in range(2, input.ndim):
            D *= input.shape[i]

        channels_per_group = C // num_groups
        elems_per_group = channels_per_group * D

        # dtype-aware MAX_BLOCK (L1.4)
        if input.dtype == torch.float32:
            MAX_BLOCK = 2048
        elif D == 1:
            MAX_BLOCK = 2048
        else:
            MAX_BLOCK = 4096

        block_size = MAX_BLOCK
        if block_size > elems_per_group:
            block_size = elems_per_group
        if D > 16 and D < MAX_BLOCK and D % 16 != 0:
            block_size = ((D - 1) // 16) * 16
            block_size = min(block_size, elems_per_group)
            if block_size < 16:
                block_size = elems_per_group

        output = torch.empty_like(input)
        mean_out = torch.empty((N, num_groups), dtype=torch.float32, device=input.device)
        rstd_out = torch.empty((N, num_groups), dtype=torch.float32, device=input.device)

        total_groups = N * num_groups
        grid_size = min(total_groups, self.VEC_CORE_NUM)
        grid = (grid_size,)

        group_norm_swish_kernel[grid](
            input, weight, bias, output, mean_out, rstd_out,
            N, C, D, num_groups, channels_per_group, elems_per_group,
            eps, swish_scale,
            num_cores=self.VEC_CORE_NUM,
            BLOCK_SIZE=block_size,
        )

        return output, mean_out, rstd_out
```

#### L2.2 Kernel 代码骨架

```python
@triton.jit
def group_norm_swish_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    N,
    C,
    D,
    num_groups,
    channels_per_group: tl.constexpr,
    elems_per_group: tl.constexpr,
    eps,
    swish_scale,
    num_cores: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_groups = N * num_groups

    for gid in range(pid, total_groups, num_cores):
        pid_n = gid // num_groups
        pid_g = gid % num_groups

        base_input = pid_n * C * D + pid_g * channels_per_group * D
        base_mean = pid_n * num_groups + pid_g
        group_base = pid_g * channels_per_group

        # --- Pass 1: one-pass reduce (N1 fp32 累加) ---
        sum_acc = tl.full((), 0.0, tl.float32)
        sq_acc = tl.full((), 0.0, tl.float32)

        for d_start in range(0, elems_per_group, BLOCK_SIZE):
            offsets = d_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < elems_per_group
            x = tl.load(input_ptr + base_input + offsets, mask=mask, other=0.0).to(tl.float32)
            sum_acc += tl.sum(x, axis=0)
            sq_acc += tl.sum(x * x, axis=0)

        mean = sum_acc / elems_per_group
        var = (sq_acc - sum_acc * sum_acc / elems_per_group) / elems_per_group
        var = tl.maximum(var, 0.0)
        rstd = 1.0 / tl.sqrt(var + eps)

        tl.store(mean_ptr + base_mean, mean)
        tl.store(rstd_ptr + base_mean, rstd)

        # --- Pass 2: normalize + weight + bias + swish (N3 标量 load) ---
        for d_start in range(0, elems_per_group, BLOCK_SIZE):
            offsets = d_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < elems_per_group
            x = tl.load(input_ptr + base_input + offsets, mask=mask, other=0.0).to(tl.float32)

            x_norm = (x - mean) * rstd

            # 标量 load: BLOCK_SIZE < D 时整个 block 在同一 channel
            ch = d_start // D
            w = tl.load(weight_ptr + group_base + ch)
            b = tl.load(bias_ptr + group_base + ch)
            x_norm = x_norm * w + b

            # Swish: x * sigmoid(swish_scale * x)
            out = x_norm * tl.sigmoid(swish_scale * x_norm)

            tl.store(output_ptr + base_input + offsets, out.to(input_ptr.dtype.element_ty), mask=mask)
```

#### L2.3 可选: Host 侧快速路径检测（跳过 weight/bias）

若需要在特定场景下获得更高性能（如 benchmark 固定 weight=1, bias=0），可在 host 侧添加检测:

```python
# 可选快速路径（不改变 kernel，仅在 host 侧选择）
if torch.all(weight == 1) and torch.all(bias == 0):
    # 调用简化版 kernel（无 weight/bias 参数）
    group_norm_swish_kernel_no_weight[grid](...)
else:
    # 调用完整版 kernel（标量 load）
    group_norm_swish_kernel[grid](...)
```

**注意**: 这是可选优化，不是必须。通用实现只需一个带标量 load 的 kernel 即可达标（1.198x）。

### §2.3 Layer 3: 关键技巧（按优先级排序）

#### L3.1 标量 load 路径（已验证有效，通用首选）

```python
ch = d_start // D
w = tl.load(weight_ptr + group_base + ch)
b = tl.load(bias_ptr + group_base + ch)
```

**适用条件**: 所有 case（无前置条件）
**性能**: geomean 1.198x（通用路径）
**优势**: 无 scalar 降级，不依赖 D 的对齐性，代码简单

#### L3.2 One-pass reduce 累加（已验证有效）

```python
sum_acc = tl.full((), 0.0, tl.float32)
sq_acc = tl.full((), 0.0, tl.float32)
for d_start in range(0, elems_per_group, BLOCK_SIZE):
    offsets = d_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elems_per_group
    x = tl.load(input_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_acc += tl.sum(x, axis=0)
    sq_acc += tl.sum(x * x, axis=0)

mean = sum_acc / elems_per_group
var = (sq_acc - sum_acc * sum_acc / elems_per_group) / elems_per_group
var = tl.maximum(var, 0.0)
rstd = 1.0 / tl.sqrt(var + eps)
```

#### L3.3 BLOCK_SIZE 最大化策略（已验证有效）

```python
if input.dtype == torch.float32:
    MAX_BLOCK = 2048
elif D == 1:
    MAX_BLOCK = 2048
else:
    MAX_BLOCK = 4096

block_size = MAX_BLOCK
if block_size > elems_per_group:
    block_size = elems_per_group

# 仅 D % 16 != 0 时缩 block
if D > 16 and D < MAX_BLOCK and D % 16 != 0:
    block_size = ((D - 1) // 16) * 16
    block_size = min(block_size, elems_per_group)
    if block_size < 16:
        block_size = elems_per_group
```

**Why**: 最大化 UB 利用率；仅对不整除 16 的 D 缩 block，避免向量除法 scalar 降级。

### §2.4 GroupNormSwish 性能基准

| 版本 | 架构 | weight 策略 | 加速比 | 特点 |
|------|------|-----------|--------|------|
| 稳定版 | 单 kernel fused | 标量 load | 1.198x | 通用，稳定复现 |
| 优化版 | 单 kernel fused + constexpr | 标量 load | **1.2549x** | 通用，入参静态化后性能提升 38% |
| 优化版 | 单 kernel fused | 跳过 w/b | 1.304x | task-specific，需检测 |

**历史反模式（应避免）**:

| 版本 | 架构 | weight 策略 | 加速比 | 失败原因 |
|------|------|-----------|--------|---------|
| 0601_0.098 | 单 kernel | 向量 load | 0.098x | offsets//D scalar 降级 |
| 0601_0.293 | 双 kernel | per-channel | 0.293x | 双 kernel launch 开销 |
| 0601_0.669 | 双 kernel | USE_2D | 0.669x | 双 kernel + reshape |
| 0530_0.863 | 单 kernel | USE_2D | 0.863x | reshape 有 bug |

---

## §3 AdaptiveInstanceNorm2DBackward 算子（norm-backward / backward）

**算子类别**: `norm-backward`（反向多粒度归约）
**典型特征**: 输入为 4D 图像张量 `[N, C, H, W]`，输出 grad_input、grad_weight、grad_bias；需要在 spatial 维度 (H*W) 上做归约，同时存在 per-channel 和 per-(N,C) 的统计量
**性能基准**:
- v1 (20260617): 几何平均加速比约 **1.11x** vs PyTorch，相比 Phase 3 基线提升约 **3.87x**

**本版本核心原则**:
- ✅ **双 kernel 两阶段实现**（reduce kernel + apply kernel）
- ✅ **fp32 partial buffer**（per-(N,C) 和 per-channel 中间统计量全部 fp32）
- ✅ **atomic_add 跨 tile 归约**（reduce 阶段聚合到 partial buffer）
- ❌ **禁止单 kernel 同时计算 grad_input**（无法在单 tile 内获得全局统计和）
- ❌ **禁止 fp16/bf16 累加中间统计量**（精度损失导致 verify 失败）

### §3.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 必须使用双 kernel 两阶段实现

- **必须**拆分为 `reduce` kernel + `apply` kernel
- **Why:** AdaIN backward 需要同时获得 per-(N,C) 的归约量（用于 grad_input）和 per-channel 的归约量（grad_weight/grad_bias）。单 kernel 同时写两类输出会在同一线程内引入两种不同粒度的 atomic/reduction，难以高效映射到 NPU 向量单元；双 kernel 让每个 kernel 专注单一归约模式，访存和计算更规则。
- **How to apply:** 所有场景

#### L1.2 必须使用 fp32 累加器存储中间统计量

（见 §1 N1）- **必须**将 `partial_s1`、`partial_s2`、`partial_s3`（per-(N,C)）以及 `grad_weight`、`grad_bias`（per-channel）分配为 `torch.float32`。

#### L1.3 必须使用 atomic_add 进行跨 tile 归约

（见 §1 N4）- **必须**在 `reduce` kernel 中通过 `tl.atomic_add` 将每个 tile 的部分和累加到 per-(N,C) partial buffer 和 per-channel buffer。

#### L1.4 禁止在 kernel 中使用 triton.cdiv

（见 §1 N6）- **禁止**在 `@triton.jit` kernel 内部使用 `triton.cdiv`，必须使用 `tl.cdiv`。

#### L1.5 禁止在 reduce kernel 内直接计算完整 grad_input

- **禁止**在 `reduce` kernel 中同时计算并写出 grad_input
- **Why:** grad_input 需要完整的 per-(N,C) 统计和（来自所有 tile），单个 tile 无法获得完整信息；若用全局 atomic/reduction 实现会严重拖慢 kernel
- **How to apply:** reduce kernel 只输出 partial sums 和 per-channel 结果；apply kernel 读取聚合后的 partial sums 再写 grad_input

### §3.2 Layer 2: 算法骨架（Agent 可参考架构）

#### L2.1 Host 侧准备与调度

```python
def forward(self, grad_output, x, weight, mean, std):
    N, C, H, W = x.shape
    S = H * W
    grad_input = torch.empty_like(x)
    grad_weight = torch.zeros((C,), dtype=torch.float32, device=x.device)
    grad_bias = torch.zeros((C,), dtype=torch.float32, device=x.device)

    partial_s1 = torch.zeros((N * C,), dtype=torch.float32, device=x.device)
    partial_s2 = torch.zeros((N * C,), dtype=torch.float32, device=x.device)
    partial_s3 = torch.zeros((N * C,), dtype=torch.float32, device=x.device)

    BLOCK_S = triton.next_power_of_2(S)
    if BLOCK_S > 1024:
        BLOCK_S = 1024
    NUM_TILES = (S + BLOCK_S - 1) // BLOCK_S

    num_cores = _get_num_cores()
    total_units = N * C * NUM_TILES
    grid_size = total_units if total_units < num_cores else num_cores

    _run_adain_bwd(...)
    return grad_input, grad_weight.to(orig_dtype), grad_bias.to(orig_dtype)
```

#### L2.2 Reduce Kernel 骨架

- Grid: 1D
- 每个 program 顺序处理多个 `(n, c, tile)` work units
- 对当前 tile 内的 spatial 数据做向量加载
- 计算:
  - `s1 = sum(grad_output)`
  - `s2 = sum(grad_output * (x - mean))`
  - `s3 = sum(x - mean)`
- `tl.atomic_add(partial_s1_ptr + pid_nc, s1_raw)`
- `tl.atomic_add(partial_s2_ptr + pid_nc, s2_raw)`
- `tl.atomic_add(partial_s3_ptr + pid_nc, s3)`
- `tl.atomic_add(grad_bias_ptr + c, s1_raw)`
- `tl.atomic_add(grad_weight_ptr + c, s2_raw / std)`

#### L2.3 Apply Kernel 骨架

- Grid: 1D，与 reduce 相同的 work unit 划分
- 读取 per-(N,C) partial sums: `s1_final`, `s2_final`, `s3_final`
- 读取 mean/std/weight
- 计算:
  - `grad_var = -0.5 * s2_final * inv_std^3`
  - `grad_mean = -s1_final * weight / std - grad_var * s3_final * (2 / S)`
  - `grad_input = grad_output * weight / std + grad_var * 2 * (x - mean) / S + grad_mean / S`
- 向量存储 grad_input

### §3.3 Layer 3: 关键技巧（Agent 可参考，但实现方式可不同）

#### L3.1 两阶段归避免除全局同步

**旧思路（单 kernel，性能/精度差）:**
- 每个线程处理一个 (N,C) channel，对全部 spatial 做串行 reduce 后再写 grad_input
- 问题：spatial 大时单线程负载过重，且无法利用多核 tile 级并行

**新思路（双 kernel + partial buffer）:**
```python
# reduce kernel: 每个 tile 独立计算部分和
s1_raw = tl.sum(go_f * mask_f)
s2_raw = tl.sum(go_f * xc * mask_f)
s3 = tl.sum(xc)

# 通过 atomic_add 聚合到 per-(N,C) buffer
tl.atomic_add(partial_s1_ptr + pid_nc, s1_raw)
tl.atomic_add(partial_s2_ptr + pid_nc, s2_raw)
tl.atomic_add(partial_s3_ptr + pid_nc, s3)

# apply kernel: 读取聚合结果，计算最终 grad_input
s1_final = tl.load(partial_s1_ptr + pid_nc)
s2_final = tl.load(partial_s2_ptr + pid_nc)
s3_final = tl.load(partial_s3_ptr + pid_nc)
```

**可替代方向:**
- 可尝试将 reduce 和 apply 合并为单个 kernel，使用 shared memory / UB 做 tile 内归约，再跨 tile atomic；需要验证精度与性能
- 可尝试按 channel 而非 (N,C,tile) 划分 grid，让 grad_weight/grad_bias 的累加更集中

#### L3.2 BLOCK_S 选择策略

```python
BLOCK_S = triton.next_power_of_2(S)
if BLOCK_S > 1024:
    BLOCK_S = 1024
NUM_TILES = (S + BLOCK_S - 1) // BLOCK_S
```

**Why:**
- `tl.arange(0, BLOCK_S)` 必须是编译期常量，因此 BLOCK_S 为 `tl.constexpr`（见 N5）
- 对 S 取 next_power_of_2 可保证 mask 外数据被正确填充为 0
- 上限 1024 是 Triton Ascend 向量寄存器的实际限制

**可替代方向:**
- 可尝试固定 BLOCK_S = 1024 并用循环处理多 tile（若 S 远大于 1024）
- 可尝试根据 S 大小动态选择 BLOCK_S ∈ {256, 512, 1024}

#### L3.3 1D Grid + 每个 program 顺序处理多 work units

```python
pid = tl.program_id(0)
grid_size = tl.num_programs(0)
total_units = N * C * NUM_TILES
units_per_prog = (total_units + grid_size - 1) // grid_size
global_idx = pid * units_per_prog

while global_idx < total_units:
    # ... process one (n, c, tile) ...
    global_idx = global_idx + 1
```

**Why:**
- 1D grid 编程简单，易于映射到 NPU vector core
- 每个 program 顺序处理连续 work units，减少跨核调度开销
- grid_size 限制为核数，避免过量 block

**可替代方向:**
- 可尝试 2D/3D grid 直接映射 (n, c, tile)，但需验证 grid 限制与负载均衡

#### L3.4 数值稳定性：先转 fp32 再计算

```python
mean_f = mean_val.to(tl.float32)
std_f = std_val.to(tl.float32)
weight_f = weight_val.to(tl.float32)
inv_std = 1.0 / std_f
```

**Why:**
- AdaIN backward 公式包含 `inv_std^3`、`1/S` 等小量乘法，fp16/bf16 容易下溢或精度损失
- 在 kernel 内将输入转为 fp32 计算，最后存储时转回原始 dtype

**可替代方向:**
- 若输入为 fp32 可跳过类型转换；若输入为 bf16/fp16 必须转 fp32

#### L3.5 grad_weight / grad_bias 直接在 reduce kernel 中产出

```python
# reduce kernel 中同时累加 per-channel 结果
tl.atomic_add(grad_bias_ptr + c, s1_raw)
tl.atomic_add(grad_weight_ptr + c, s2_raw * inv_std)
```

**Why:**
- grad_weight 需要 `sum(grad_output * (x - mean) / std)`，可在 reduce 阶段每个 tile 直接计算 `s2_raw * inv_std`
- 避免 apply kernel 再做一次 channel 级归约

**可替代方向:**
- 若 atomic_add 成为瓶颈，可尝试 per-channel 单独用一个 reduction kernel

### §3.4 AdaptiveInstanceNorm2DBackward 性能基准

| Shape 类型 | v1 加速比 | 说明 |
|-----------|-----------|------|
| 小 spatial (H*W <= 256) | 1.0x - 1.4x | 多数有正向加速 |
| 中等 spatial (256 < H*W <= 10000) | 1.0x - 1.3x | 表现稳定 |
| 大 spatial / 大 batch (如 8x64x56x56) | < 1.0x | atomic_add 和 grid 划分导致性能退化 |

**关键结论**:
1. **双 kernel reduce+apply 是正确性的基础**：单 kernel 难以同时满足 per-(N,C) 和 per-channel 两种归约精度
2. **fp32 partial buffer 是精度保障**：所有中间统计量必须用 fp32 累加
3. **atomic_add 是跨 tile 归约的可靠原语**，但在大 batch 大 channel 场景下可能成为性能瓶颈
4. **1D grid + 顺序处理多 work units** 在中小 shape 上效果良好，但在总 work units 远超核数的大 shape 上负载均衡需进一步优化
5. **Phase 3 基线到 Phase 4 优化提升约 3.87x**，说明草图架构选择对最终性能影响巨大

---

## §4 常见陷阱与避免方法

### §4.1 GroupNormSwish 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 使用 USE_2D / tl.reshape | `tl.reshape` 在 Ascend 9.0.0.beta1 上有 correctness bug 和编译限制（shape 参数不能是 tl.constexpr） | 完全避免 reshape，统一使用标量 load 路径 (L1.3/N3) |
| 使用双 kernel split | 增加 launch 开销和内存流量，历史最高仅 0.669x | 始终使用单 kernel fused 架构 (L1.1) |
| 对所有 case 缩 BLOCK | 增加循环次数，性能下降（0.1581x → 0.1222x），部分 case 触发 vector core 异常 | 仅对 `D % 16 != 0` 缩 block (L1.4/L3.3) |
| 使用向量除法 `offsets // D` | 向量化 int 除法在 Ascend 上严重 scalar 化 | 通过缩 BLOCK 确保 `BLOCK_SIZE < D`，走标量 load 路径 `ch = d_start // D` (N3) |
| 2D grid 调度开销 | `grid=(N, num_groups)` 当 `N * num_groups > VEC_CORE_NUM` 时性能骤降 | 始终使用 1D grid + 交错循环 (L1.2) |
| weight/bias 走向量 load 路径 | `ch_vec = offsets // D` 后 `tl.load(weight + ch_vec)` 触发向量除法降级 | 统一标量 load (N3/L3.1) |
| fp16/bf16 直接累加 sum/sq_sum | var 公式 `(sq-sum*sum/N)/N` 对累加精度敏感，低精度累加导致 rstd 误差 | fp32 累加器 `tl.full((), 0.0, tl.float32)` (N1) |
| 默认开 multibuffer / unit_flag | Norm 类算子有 reduce 和循环，multibuffer 收益不确定，实测下降 | 首次生成不开，作为 Phase 4 独立优化点测试 (L1.5) |
| channels_per_group / elems_per_group 作为运行时参数 | 错失启动级特化，性能少 38% | 声明为 `tl.constexpr` (L1.6/N5) |

### §4.2 AdaptiveInstanceNorm2DBackward 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 单 kernel 同时做两种粒度的归约 | 试图在一个 kernel 内同时计算 grad_input 和 grad_weight/grad_bias，导致部分和使用错误的归约维度，verify 失败 | 严格拆分为 reduce（输出 partial sums + per-channel 结果）和 apply（读取 partial sums 输出 grad_input）(L1.1/L1.5) |
| 使用 fp16/bf16 累加中间统计量 | `s1`/`s2`/`s3` 用原始 dtype 累加，在小 spatial size 或数值接近 0 时精度不足 | partial buffer 和 grad_weight/grad_bias 全部使用 fp32，最后转回原始 dtype (L1.2/N1) |
| 忽略 mask 导致越界读写 | BLOCK_S 取 next_power_of_2 后，实际 S 可能不是 2 的幂，未加 mask 会读/写越界数据 | 始终使用 `mask = s_offs < S` 并在 load/store 中传入 mask (L3.2) |
| 在 kernel 内使用 `triton.cdiv` | 编译期报错 `ValueError: Did you forget to add @triton.jit ?` | kernel 内使用 `tl.cdiv`，host 侧使用 `triton.cdiv` 或整数除法 (L1.4/N6) |
| 过早将 grad_weight/grad_bias 转回原始 dtype | 在 reduce kernel 中直接以原始 dtype atomic_add，累加精度损失 | grad_weight/grad_bias buffer 保持 fp32 直到返回前再转换 (L1.2/L3.5) |
| reduce kernel 内同时算 grad_input | grad_input 需全局 s1/s2/s3，单 tile 信息不全 | reduce 只写 partial sums，apply 读聚合结果再算 grad_input (L1.5/L3.1) |
| 大 batch 大 channel 下 atomic_add 成瓶颈 | 并发 atomic 冲突导致性能退化（< 1.0x） | 评估改用 per-channel 独立 reduction kernel，或按 channel 划分 grid (L3.5 可替代方向) |
| `inv_std^3` 等小量乘法用低精度 | fp16/bf16 下溢或精度损失 | kernel 内统一转 fp32 计算，存储时再转回 (L3.4/N1) |

---

## 快速检查清单（生成后自检）

### GroupNormSwish 自检
- [ ] 只有一个 `@triton.jit` kernel（单 kernel fused）
- [ ] 使用 `grid=(min(N*num_groups, VEC_CORE_NUM),)`（1D grid）
- [ ] Kernel 内有 `for gid in range(pid, total_groups, num_cores)`（交错循环）
- [ ] 没有 `tl.reshape`
- [ ] weight/bias 使用标量 load: `ch = d_start // D; tl.load(ptr + ch)`
- [ ] 没有 `USE_2D` 条件分支
- [ ] BLOCK_SIZE 策略: dtype-aware + 最大化 + 仅 D%16!=0 缩 block
- [ ] Reduce 累加使用 `tl.float32`
- [ ] 首次生成不添加 `multibuffer=True, unit_flag=True`
- [ ] 没有双 kernel split
- [ ] `channels_per_group` 和 `elems_per_group` 声明为 `tl.constexpr`（入参静态化）

### AdaptiveInstanceNorm2DBackward 自检
- [ ] 拆分为 reduce kernel + apply kernel（双 kernel 两阶段）
- [ ] `partial_s1/s2/s3`、`grad_weight`、`grad_bias` 均为 `torch.float32`
- [ ] reduce kernel 用 `tl.atomic_add` 聚合 partial sums 和 per-channel 结果
- [ ] reduce kernel **不**写 grad_input
- [ ] apply kernel 读聚合后的 partial sums 再算 grad_input
- [ ] kernel 内用 `tl.cdiv`，host 侧用 `triton.cdiv`
- [ ] BLOCK_S 取 `next_power_of_2(S)` 且 cap 到 1024，作为 `tl.constexpr`
- [ ] load/store 全部带 `mask = s_offs < S`
- [ ] kernel 内所有计算先转 fp32（`inv_std`、`inv_std^3`、`1/S` 等）
- [ ] grad_weight/grad_bias 返回前才转回原始 dtype
