# Triton-Ascend 精度对齐调试完整案例 —— DynamicQuant

## 问题背景

DynamicQuant 算子在 Ascend 910B1 上前期经历 22 轮迭代全部精度失败，典型特征：
- 误差分布广泛，不是局部 shape 问题
- 误差量级极小（1-2 ULP），但在 int8 量化场景中经 round + clamp 放大后足以让 MARE 越界
- 逐个排查 `tl.max`、`tl.abs`、`nearbyint`、`clamp`、`变量除法` 均声称"无偏差"
- 真正的凶手隐藏在最不起眼的常量除法 `scale = max_abs / 127.0` 中

## 核心根因

**Triton-Ascend 编译器对 scalar 浮点除法和 vector 浮点除法走不同代码生成路径**：

| 路径 | 触发条件 | 精度行为 |
|------|---------|---------|
| Scalar | 标量变量直接运算；或 BLOCK_SIZE=1 被编译器优化回标量 | 与 torch `aclnnDiv` 存在 1-2 ULP 偏差 |
| Vector | `tl.load(vector_offsets) / 127.0`，且 BLOCK_SIZE >= 2 | 与 torch `aclnnDiv` 完全一致 |

**关键陷阱**：即使代码形式上使用了 `tl.arange(0, BLOCK_SIZE)`，若 `BLOCK_SIZE=1`，编译器仍可能将其优化为标量操作，从而落入 Scalar 路径。

## 排查全流程：五阶隔离法

### Stage 0：前置检查（排除低级错误）

在怀疑编译器之前，先确认以下基础项：

```
□ 输出 shape 与参考完全一致
□ NaN 位置完全一致（mask 按位相等）
□ Inf 位置和符号完全一致
□ dtype 转换逻辑正确（如 bfloat16 -> float32 后再计算）
□ 未使用未初始化的 memory（tl.load 的 other=0.0 是否合适）
```

### Stage 1：端到端差异定位

先用最小 reproduction 找到差异最大的 case，缩小排查范围。

```python
# 在失败 case 上对比 Triton 输出与 torch 输出
triton_out, triton_scale = model_new(x, smooth)
torch_out, torch_scale = model_ref(x, smooth)

diff = (triton_out.float() - torch_out.float()).abs()
print(f"max_diff={diff.max().item():.2e}")
print(f"diff>0 count={(diff > 0).sum().item()}")

# 如果 diff 存在，打印前几处差异的原始值和量化值
indices = torch.where(diff > 0)
for i in range(min(10, len(indices[0]))):
    idx = tuple(t[i].item() for t in indices)
    print(f"idx={idx}: triton={triton_out[idx].item()}, torch={torch_out[idx].item()}")
```

**判定**：若 diff 存在且原始输入值差异为 0（即差异纯由量化引入），则进入 Stage 2。

### Stage 2：逐操作隔离微基准测试

将算子拆分为独立原子操作，每个操作写一个最小 Triton kernel，与 torch 一对一比对。

**检查清单（DynamicQuant 为例）**：

| # | 操作 | Triton 代码 | Torch 参考 | 预期结果 |
|---|------|------------|-----------|---------|
| 1 | `abs` | `tl.abs(x)` | `x.abs()` | 0 diff |
| 2 | `max` | `tl.max(abs_x, axis=0)` | `x.abs().max(dim=1)[0]` | 0 diff |
| 3 | 常量除法 | `max_val / 127.0` | `max_val / 127.0` | **需重点检查** |
| 4 | clamp scale | `tl.maximum(scale, 1e-10)` | `scale.clamp(min=1e-10)` | 0 diff |
| 5 | 变量除法 | `x / scale` | `x / scale` | 0 diff |
| 6 | round | `nearbyint(x)` | `torch.round(x)` | 0 diff |
| 7 | clamp quant | `max(min(val,127),-128)` | `val.clamp(-128,127)` | 0 diff |
| 8 | cast to int8 | `.to(tl.int8)` | `.to(torch.int8)` | 0 diff |

**微基准 kernel 模板（以 scale 除法为例）**：

```python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def scale_kernel(max_ptr, scale_ptr, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    max_vals = tl.load(max_ptr + offsets, mask=mask, other=0.0)
    scales = max_vals / 127.0
    scales = tl.maximum(scales, 1e-10)
    tl.store(scale_ptr + offsets, scales, mask=mask)

# 测试数据
torch.manual_seed(0)
x = torch.randn((512, 512), dtype=torch.bfloat16, device="npu")
max_abs = x.float().abs().max(dim=1)[0]

# Torch 参考
scale_torch = max_abs / 127.0
scale_torch = scale_torch.clamp(min=1e-10)

# Triton 测试
scale_triton = torch.empty_like(max_abs)
grid = ((M + BLOCK_SIZE - 1) // BLOCK_SIZE,)
scale_kernel[grid](max_abs, scale_triton, M, BLOCK_SIZE=BLOCK_SIZE)
torch.npu.synchronize()

# 比对
diff = (scale_torch.cpu() - scale_triton.cpu()).abs()
print(f"BLOCK_SIZE={BLOCK_SIZE}: max_diff={diff.max().item():.2e}, count={(diff > 0).sum().item()}")
```

### Stage 3：Scalar vs Vector 路径判定

当 Stage 2 发现某个操作存在差异时，用以下矩阵测试确定是否为 scalar/vector 编译器行为差异：

| 测试 | BLOCK_SIZE | Grid | 预期 |
|------|-----------|------|------|
| A | 1 | (M,) | **可能失败**（scalar 优化） |
| B | 2 | (M/2,) | 应通过（vector） |
| C | 4 | (M/4,) | 应通过（vector） |
| D | 2048 | (1,) | 应通过（vector loop） |
| E | scalar loop in kernel | (1,) | **可能失败**（显式 scalar） |

**测试 E 的代码形式**：
```python
@triton.jit
def scalar_loop_kernel(max_ptr, scale_ptr, M):
    for i in range(M):
        val = tl.load(max_ptr + i)
        scale = val / 127.0          # 显式标量除法
        tl.store(scale_ptr + i, scale)
```

**判定规则**：
- 若 A/E 失败但 B/C/D 通过 → **确诊 scalar/vector 编译器差异**
- 若全部失败 → 差异来源不是 scalar/vector，需继续排查（如 `127.0` 字面量精度、div 指令选型等）

### Stage 4：验证修复方案

确诊后，将算子中对应操作改为 vector 路径，然后做**两级验证**：

**Level 1 — 单操作验证**：仅修改目标操作，其余仍用 torch，确认该操作单独通过。

**Level 2 — 全链路验证**：将修改后的操作放回完整 pipeline，跑全量 case 验证。

## DynamicQuant 修复详情

### 修复前（失败版本）

问题代码片段：
```python
# 错误：在 kernel 内用 scalar 方式计算 scale
@triton.jit
def quant_kernel(x_ptr, smooth_ptr, quant_ptr, scale_ptr, ...):
    pid_m = tl.program_id(0)
    # ... 计算 max_abs via loop ...
    scale = max_abs / 127.0          # scalar 除法，精度偏差！
    scale = tl.maximum(scale, 1e-10)
    # ... quant loop ...
```

### 修复后（通过版本）

将 scale 计算拆分为独立 kernel，强制 vector 路径：

```python
@triton.jit
def find_max_kernel(x_ptr, smooth_ptr, max_ptr, M, N, ...):
    # 每行求 max_abs，输出到 max_ptr[pid_m]
    ...

@triton.jit
def compute_scale_kernel(max_ptr, scale_ptr, M, BLOCK_SIZE: tl.constexpr):
    for start in range(0, M, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        max_vals = tl.load(max_ptr + offsets, mask=mask, other=0.0)
        scales = max_vals / 127.0        # vector 除法，强制编译器走向量路径
        scales = tl.maximum(scales, 1e-10)
        tl.store(scale_ptr + offsets, scales, mask=mask)

@triton.jit
def quant_kernel(x_ptr, smooth_ptr, quant_ptr, scale_ptr, ...):
    # 从 scale_ptr 加载已算好的 scale，此处只做变量除法 x/scale（无偏差）
    ...
```

调用链：
```python
find_max_kernel[grid](x, smooth, max_temp, ...)      # 求 max
compute_scale_kernel[(1,)](max_temp, scale, M, BLOCK_SIZE=2048)  # vector 算 scale
quant_kernel[grid](x, smooth, quant, scale, ...)      # 量化
```

**BLOCK_SIZE 选择**：
- `BLOCK_SIZE=2048` 是安全值，覆盖绝大多数场景下的 M
- 若 M 较小（如 < 2），可用 `BLOCK_SIZE=2` + 相应 grid 数
- **底线**：`BLOCK_SIZE` 绝不能为 1

### AST 结构适配

`validate_triton_impl.py` 要求 `forward()` 只能直接启动一次 Triton kernel。多 kernel 场景下，使用 `_route()` 包装器：

```python
def _route(self, x, smooth_scales, dst_type):
    if x.dim() == 2:
        # ... launch find_max + compute_scale + quant kernels ...
    elif x.dim() == 3:
        # ... launch 3d variants ...
    else:
        # ... 1d fallback ...

def forward(self, x, smooth_scales=None, group_index=None, dst_type=None):
    if dst_type is None:
        dst_type = torch.int8
    if group_index is not None:
        raise NotImplementedError("Group quantization not supported yet")
    if smooth_scales is None:
        # ... default smooth_scales ...
    return self._route(x, smooth_scales, dst_type)
```

AST 校验器明确允许 `self._route()` 作为合法的 kernel dispatch 包装器。

## 扩展检查清单

以下是在其他算子中排查 ULP 级差异时的通用检查项：

### 运算类

- [ ] **常量除法**：`a / 127.0`、`a * 0.5` 等是否走了 scalar 路径？尝试改为 vector load 后再运算
- [ ] **常量乘法**：与除法同理，`a * inv_127` 和 `a / 127.0` 可能生成不同指令序列，需分别测试
- [ ] **整数除法 vs 浮点除法**：确认分子分母均为浮点类型，无隐式整数除法
- [ ] **减法顺序**：`a - b` 与 `b - a` 在浮点中不等价，确认与参考实现顺序一致
- [ ] **累加顺序**：reduce/scan 操作的累加顺序是否与参考一致？Triton 的 `tl.sum` 和 torch 的 `sum` 可能有不同并行规约顺序

### 类型转换类

- [ ] **中间精度**：是否在运算前统一升精度到 float32？低精度中间结果会引入差异
- [ ] **隐式类型提升**：`tl.int8` 和 `tl.float32` 混合运算时的隐式转换行为是否与 torch 一致
- [ ] **rounding 模式**：`nearbyint`、`rint`、`round` 在边界值（如 x.5）的行为是否一致

### 内存类

- [ ] **mask 处理**：`tl.load(..., mask=mask, other=0.0)` 中的 `other` 值是否会影响后续运算？（如求 max 时 other=0.0 是正确的，但求 sum 时不正确）
- [ ] **越界读**：mask 未正确覆盖时是否读取了脏数据？
- [ ] **内存对齐**：stride 是否为预期值？可用 `tensor.stride()` 打印确认

### 编译器行为类

- [ ] **BLOCK_SIZE=1 陷阱**：任何使用 `tl.arange(0, BLOCK_SIZE)` 的地方，若 BLOCK_SIZE=1，检查编译器日志确认未被优化为标量
- [ ] **循环展开**：显式 `for` 循环中的标量操作 vs 向量化操作，对比结果
- [ ] **grid=1 行为**：grid 数为 1 时是否触发了不同的调度或指令选择？

## 工具与命令速查

### 运行 AST 退化预检查
```bash
python3 /path/to/kernel-verifier/scripts/validate_triton_impl.py \
    generated_code.py --json
```

### 运行精度验证
```bash
python3 /path/to/kernel-verifier/scripts/verify.py \
    --op_name <op_name> \
    --verify_dir <verify_dir> \
    --triton_impl_name triton_ascend_impl \
    --timeout 900
```

### 运行性能基准
```bash
python3 /path/to/kernel-verifier/scripts/benchmark.py \
    --op_name <op_name> \
    --verify_dir <verify_dir> \
    --triton_impl_name triton_ascend_impl \
    --warmup 5 --repeats 50 \
    --output perf_result.json
```

## 案例数据

| 指标 | 修复前 | 修复后 |
|------|-------|-------|
| 通过 case | 18 / 42 | **42 / 42** |
| 失败原因 | scale 计算 1-2 ULP 偏差 | 无 |
| 关键改动 | — | `compute_scale_kernel` 使用 BLOCK_SIZE=2048 vector load |
| AST 检查 | 通过 | 通过 |
| 合规性 | 纯 Triton | 纯 Triton |

## 总结

1. **微基准隔离是定位 ULP 差异的唯一可靠手段**，不要依赖端到端的猜测
2. **标量/向量浮点操作在 Ascend 后端可能有不同精度**，这是容易被忽视的编译器行为差异
3. **BLOCK_SIZE=1 在 Triton 中不等于向量操作**，编译器可能优化回标量；保守选择 BLOCK_SIZE >= 2
4. **常量除法是量化类算子的高风险点**，`scale = max / constant` 这类操作应优先放入独立 kernel 并强制 vector 路径
5. **多 kernel 场景用 `_route()` 包装**，既满足 AST 校验，又保持代码结构清晰
