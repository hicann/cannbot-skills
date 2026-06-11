# 精度验证脚本编写规则（gen_data / golden / verify）

> **导航**：本文件规定 catlass 算子三件套测试脚本（`gen_data.py` / `golden.py` / `verify_result.py`）的编写规则，**强制对齐** `ops-precision-standard` skill。精度判定标准本身见该 skill；本文件只补充 catlass 算子（GEMM + epilogue，含融合激活/反量化）落地时**最容易踩错**的点。
>
> 这些规则源自实战教训：曾出现「内核数值完全正确（MERE≈6e-6）却被 verify 判 FAIL」「golden 写入被注释导致 verify 拿旧 golden 报错」「verify 在实网 shape 上从不运行」「量化 SwiGLU 结构性错误未被发现」等问题。

---

## 0. 黄金法则

1. **verify 的判定标准必须等于 `ops-precision-standard` 选出的官方标准**，不得自创更严或更松的门限。
2. **golden 必须真的被生成**——任何分支都不能把 golden 写入注释掉 / 跳过。
3. **verify 必须能在交付要求的全部 shape 上运行**（基础 shape + 实网 shape），不能只覆盖基础 shape。
4. **golden 必须镜像内核的数值路径**（累加精度、激活公式与常量、反量化顺序、输出 dtype 与形状），否则比的是两套不同算法。

---

## 1. 选标准：先走 `ops-precision-standard` 决策树

| 算子输出 | 标准文件 | 通过判据 |
|---------|---------|---------|
| 纯浮点计算（fp16/bf16/fp32），无商用要求 | `float_compute_community.md` | MERE < Threshold 且 MARE < 10·Threshold（FLOAT16 Threshold=2⁻¹⁰≈9.766e-4） |
| 整型↔浮点（量化 dequant 输出 fp16/bf16），无商用要求 | `quantization_community.md` | 同上的 MERE/MARE Threshold（按输出 dtype 取） |
| 含小值域（golden 存在 \|x\|<SV 阈值的元素） | `special_cases.md` §1 | 小值域子集单独按 **ErrorCount 比值 ≤ 2**（对三方标杆），**与正常域 MERE/MARE 并行**，不是替换 |

**实践等价形式（推荐，且与本仓已通过的 `catlass_quant_matmul_gelu` 一致）**：fp16/bf16 输出可用 aclnn 风格的
`atol=1e-3, rtol=1e-3, error_ratio ≤ 1e-3`（允许 ≤0.1% 元素越界）。它对正常域等价于 MERE/MARE 门限，对小值域用绝对容差 `atol` 自然兜底，避免相对误差在近零处不稳定。

```python
abs_err = np.abs(out - gold)
tol = ATOL + RTOL * np.abs(gold)          # ATOL=1e-3, RTOL=1e-3
error_ratio = (abs_err > tol).mean()
is_pass = error_ratio <= 1e-3             # ERROR_RATIO_THRESHOLD
```

---

## 2. 禁止：自创零容忍小值域门限（最常见的误判来源）

过零的激活（**SwiGLU**、tanh-GELU 等）输出会大量穿过 0。fp16 在零交叉处存在抵消噪声：golden≈0 的元素，内核与 golden 的**绝对误差可达 1e-5~1e-4**，但这只是 fp16 量化噪声，不是内核 bug。

- ❌ **错误做法**：要求「`|golden|<2⁻¹¹` 且 `|actual-golden|>2⁻¹⁶(≈1.5e-5)` 的元素个数为 0」。这是零容忍绝对门限，比官方标准严得多——官方小值域判据是**与三方标杆的 ErrorCount 比值 ≤ 2**，不是「必须为 0」。
- ❌ **错误做法**：把 `MARE = max(相对误差)` over **全体元素**作为硬门限。近零 golden 会让单点相对误差爆到 1e+1~1e+2，必然 FAIL，但毫无意义。
- ✅ **正确做法**：
  - 用 §1 的 `atol/rtol + error_ratio` 形式（小值域被 `atol` 兜底）；**或**
  - 正常域（`|golden| ≥ SV阈值`）算 MERE/MARE 门限，小值域按 `special_cases.md` 的比值判据（无三方标杆时退化为绝对容差 `2⁻¹⁶` 的 **ErrorCount 比例**报告，而非零容忍）。

**自检**：若内核 MERE 很小（如 1e-5 量级）却 verify FAIL，几乎一定是 verify 的小值域/相对误差门限写错了，先核对标准，再怀疑内核。

---

## 3. golden 必须镜像内核数值路径

| 内核行为 | golden 必须 |
|---------|------------|
| Cube fp32 累加（CType=float） | `a.astype(fp32) @ b.astype(fp32)`，**fp32 累加** |
| epilogue 在 fp32 计算后再 `Cast` 到输出 dtype | golden 在 fp32 算完激活，最后 `.astype(np.float16)` |
| 激活用 catlass Tile 的具体公式 | 镜像同一公式与常量（如 tanh-GELU 的 `-1.595769121`、`0.044715`；silu=`x/(1+exp(-x))`） |
| 量化：`Cint32 * scale[n] * perTokenScale[m]` | per-channel 先乘 scale，再乘 per-token，**fp32** |
| SwiGLU 输出 `[M, N/2]`，`silu(C[:,:H])*C[:,H:]` | golden 形状 `[M, N/2]`，左右半正确切分 |

镜像激活公式的目的：避免「numpy 用库 exp / 内核用硬件 Exp」的系统性偏差被误当成精度问题。如确需对比硬件激活，golden 用与 Tile 相同的算子序列（Muls/Exp/Adds/Div）实现。

---

## 4. int8 GEMM 的 golden：用 fp32 BLAS，别用 numpy 整数矩阵乘

numpy 的 `int32 @ int32` 走标量路径，大 shape（K 数千、N 上万）会**慢到分钟级**，拖垮整个验证流程。

- ✅ 把 int8 输入 `astype(np.float32)` 后用 BLAS `@`。当 `|Cint| < 2²⁴`（fp32 尾数上限）时结果**精确无误差**。
  - 估算：`|a|,|b| ≤ s`，则 `|Cint| ≤ K·s²`。如 `s=8, K=6144` → `≤ 393216 < 2²⁴=16777216`，安全。
- ✅ 在**小 shape** 上加一道整数参考校验（`(a.astype(int64)@b.astype(int64))` 对比 fp32 路径）确认 BLAS 路径的精确性，再用于大 shape。

```python
cint = (a.astype(np.float32) @ b.astype(np.float32))   # 精确（|Cint|<2^24）
cdq  = cint * scale[None, :] * per_token[:, None]       # fp32 反量化
```

---

## 5. gen_data 与数据幅度

- gen_data **必须**计算并写出 golden（`golden.bin`）——不要把写入注释掉。
- 输入幅度要让 GEMM 结果落在激活的有用区间且输出不溢出 fp16（`|D|<65504`）。例如 SwiGLU 会放大幅度（`silu(x1)*x2`），需用较小的 scale / 输入范围把反量化值压到 O(1)。
- 形状参数化：脚本要接受任意 `M N K`，输出文件大小随之变化，便于跑实网 shape。

---

## 6. verify 脚本模板（fp16 输出，社区标准实践形式）

```python
import sys, numpy as np
ATOL, RTOL, ERROR_RATIO_THRESHOLD = 1e-3, 1e-3, 1e-3  # 社区 fp16 标准的实践等价形式
# 依据: ops-precision-standard/reference/float_compute_community.md + special_cases.md
m, n = int(sys.argv[1]), int(sys.argv[2]); data = sys.argv[3]
out  = np.fromfile(f"{data}/out.bin",    dtype=np.float16).astype(np.float32).reshape(m, n)
gold = np.fromfile(f"{data}/golden.bin", dtype=np.float16).astype(np.float32).reshape(m, n)
abs_err = np.abs(out - gold)
tol = ATOL + RTOL * np.abs(gold)
error_ratio = float((abs_err > tol).mean())
print(f"[verify] error_ratio={error_ratio:.3e} (thr={ERROR_RATIO_THRESHOLD:.0e}, atol={ATOL}, rtol={RTOL})")
sys.exit(0 if error_ratio <= ERROR_RATIO_THRESHOLD else 1)
```

> 对量化输出（int↔fp16），同样适用；若用户明确要求商用标准，改走 `float_compute.md` / `quantization.md` 的门限。

---

## 7. 验证范围

- 必须在**所有交付 shape** 上跑 gen_data → 运行 → verify：基础 shape（如 512³，覆盖每个合法 dtype/转置/swizzle 分支）**与**实网 shape（大 M/N/K）。
- 仅基础 shape 通过 **不代表** 实网通过：本仓的 `catlass_quant_matmul_swiglu` 正是「512³ 过、N>512 全错」的结构性 bug（见 [mmad-epilogue 选型](../../catlass-op-design/references/mmad-epilogue-selection.md) §4 双操作数 epilogue）。verify 必须覆盖到能暴露这类问题的 shape。
