# A5 API 差异迁移适配指南

> 本文件汇总 A5（arch35/C310）相对 A2/A3 在**精度、兼容性、性能**三个维度的 API 差异，并给出迁移适配方案。
> 来源：`docs/api差异文档.md`（A5 vs A2/A3 指令代际差异对算子的影响分析）。
>
> **使用时机**：
> - 阶段 1 评估定级时，MUST 加载本文件做 API 差异风险扫描
> - 阶段 2 代码改造时，按本文件的适配清单逐项处理
> - 阶段 4 精度验证时，按 Subnormal 测试用例设计补充

---

## 0. 总体结论

| 维度 | 影响面 | 关键根因 | 适配策略 |
|------|--------|---------|---------|
| **精度** | 大（基础算数 API 普遍受影响） | A5 裁剪 subnormal 能力，subnormal 结果直接变 0 | 优先 eps 规避；其次 API 模板参数 `PRECISION_1ULP_FTZ_FALSE` 软件模拟（性能影响大，按需开 high_precision 分支） |
| **兼容性** | 小（仅 Mmad 不支持 int4） | Mmad 不支持 s4 类型 | 量化 matmul 需先 cast 成 int8（影响性能） |
| **性能** | 大（三类指令退化） | vnchwconv/VSLDB 吞吐下降；BilinearInterpolation 实现路径变化 | 按算子类别和 shape 风险分级，针对性优化 |

---

## 1. 精度差异：Subnormal 裁剪

### 1.1 背景

A5 芯片裁剪了 subnormal（次正规数）能力。当计算结果为 subnormal 时会直接变成 0，只能通过软件实现模拟。当前 AscendC API 提供了 `ExpAlgo::PRECISION_1ULP_FTZ_FALSE` 模板参数开启该功能。

> SubNormal 浮点数：指数位全为 0、尾数不为 0，用于表示比最小正常数更小的值，避免"下溢为 0"。

### 1.2 受影响 API 清单

以下基础算数 API 在 subnormal 场景均会有精度丢失：**Exp / Ln / Sqrt / Rsqrt / Div / Reciprocal**。

各 API 的 Config 结构体、algo 模板参数选项与具体用法属于 API 知识，由 API 最佳实践承载：
- Exp/Ln/Sqrt/Rsqrt/Div/Reciprocal 的 algo 参数与 Subnormal 处理：`cannbot-skills/ops/ascendc-api-best-practices/references/api-cross-gen-migration.md`
- asc-devkit 官方 API 文档路径：见 `references/devkit-path-map.md`
- 精度阈值与比对标准：`cannbot-skills/ops/ops-precision-standard/SKILL.md`

algo 参数语义速览（详见上述链接）：`INTRINSIC`（默认，单指令，最快，Subnormal 被近似为 0）；`PRECISION_1ULP_FTZ_FALSE`（软件模拟，支持 Subnormal 计算，**性能影响大**）。

### 1.3 适配策略（按优先级）

**建议优先使用 eps 规避、其次使用 A5 API 模板参数（对性能影响很大，建议有需要的算子开 high_precision 分支）。**

#### 策略 A：eps 规避（首选）

在算子计算路径中引入一个小的 eps 常数，避免输入落入 subnormal 区间：

```cpp
// 220x 版本（默认支持 subnormal，无需 eps）：
AscendC::Ln(dstLocal, srcLocal, count);

// 351x 版本 — eps 规避（首选，性能无损）：
constexpr float EPS = 1.0e-38f;  // 大于 FP32 最小正常数
LocalTensor<float> safeSrc = ...;
Adds(safeSrc, srcLocal, EPS, count);  // src += eps，避免落入 subnormal
AscendC::Ln(dstLocal, safeSrc, count);  // 默认 INTRINSIC，高性能
```

**适用场景**：输入数据范围已知，且 eps 引入的误差在精度阈值内。

#### 策略 B：API 模板参数（次选，按需分支）

当 eps 规避不可行（如输入范围不可控、精度要求极高）时，使用 `PRECISION_1ULP_FTZ_FALSE` 软件模拟。**因性能影响大，建议为算子开 high_precision 分支，仅在需要时启用**：

```cpp
// 351x 版本 — 高性能模式（subnormal → 0，默认）：
constexpr AscendC::LnConfig LN_CONFIG_FAST = { AscendC::LnAlgo::INTRINSIC };
AscendC::Ln<T, LN_CONFIG_FAST>(dstLocal, srcLocal, count);

// 351x 版本 — 高精度模式（支持 subnormal，软件模拟）：
constexpr AscendC::LnConfig LN_CONFIG_PRECISE = { AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE };
AscendC::Ln<T, LN_CONFIG_PRECISE>(dstLocal, srcLocal, count);
```

**分支选择建议**：

```cpp
// 算子实现中按 tiling 参数或属性切换精度模式
if (highPrecisionMode) {
    AscendC::Ln<T, LN_CONFIG_PRECISE>(dstLocal, srcLocal, count);
} else {
    AscendC::Ln<T, LN_CONFIG_FAST>(dstLocal, srcLocal, count);
}
```

#### 策略 C：Register-based 模式下的 Subnormal 处理

在 MicroAPI（Register-based）模式下，通过 `XxxSpecificMode` 结构体配置：

```cpp
__VEC_SCOPE__
{
    constexpr MicroAPI::LnSpecificMode LN_SUBNORMAL_MODE = {
        MicroAPI::MaskMergeMode::ZEROING,
        AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE
    };
    constexpr MicroAPI::ExpSpecificMode EXP_SUBNORMAL_MODE = {
        MicroAPI::MaskMergeMode::ZEROING,
        AscendC::ExpAlgo::PRECISION_1ULP_FTZ_FALSE
    };
    constexpr MicroAPI::SqrtSpecificMode SQRT_SUBNORMAL_MODE = {
        MicroAPI::MaskMergeMode::ZEROING,
        false,
        AscendC::SqrtAlgo::PRECISION_1ULP_FTZ_FALSE
    };
    constexpr MicroAPI::DivSpecificMode DIV_SUBNORMAL_MODE = {
        MicroAPI::MaskMergeMode::ZEROING,
        false,
        AscendC::DivAlgo::PRECISION_1ULP_FTZ_FALSE
    };

    RegTensor<float> regDst, regSrc;
    MaskReg pMask = CreateMask<float, MaskPattern::ALL>();
    MicroAPI::Ln<float, &LN_SUBNORMAL_MODE>(regDst, regSrc, pMask);
    MicroAPI::Exp<float, &EXP_SUBNORMAL_MODE>(regDst, regSrc, pMask);
    MicroAPI::Sqrt<float, &SQRT_SUBNORMAL_MODE>(regDst, regSrc, pMask);
    MicroAPI::Div<float, &DIV_SUBNORMAL_MODE>(regDst, regSrc, pMask);
}
```

### 1.4 Subnormal 风险扫描清单

在阶段 1 评估时，对算子源码做以下扫描：

| 扫描项 | 检测方法 | 命中后的处理 |
|--------|---------|-------------|
| 是否调用 Exp/Ln/Sqrt/Rsqrt/Div/Reciprocal | grep 源码中的 API 名 | 标记为 Subnormal 风险算子 |
| **★ 是否存在结构性 subnormal 排除（策略 0，最先检查）** | 追受影响操作数（通常是分母）的生成链，找**结构性下界证明**：① max 归一锚点（softmax 在线算法行内减 max → `exp(0)=1` → 分母 Σexp ≥ 1）；② invalid line 哨兵修复（检测哨兵值如 `-FLT_MAX` 后将分母强制写为 1.0，见 FA 系 `SoftmaxSumUpdate`）；③ 值域单调性/数学下界推导 | **给出证明链 → 无需任何 API 级适配，保持 INTRINSIC**；MUST 在分析摘要记录证明（含代码行号），禁止无证明直接跳过 |
| 输入数据是否可能包含 subnormal | **★ 基于证据分析**（dtype/输入范围/中间计算/常量/计算链），禁止仅凭业务经验判断 | 必须选择策略 A/B/C 之一 |
| 是否在性能关键路径 | 判断算子是否为高频算子（RMSNorm/RoPE/Softmax） | 性能关键路径优先用 eps 规避 |
| 是否已有 eps 处理 | 检查源码中 `+ epsilon` / `+ eps` / 分母保护常量 | **★ 仅有 eps 不能免处理**——MUST 按 §1.6 检查 eps 在所有支持 dtype 下是否可表示 |

**结构性排除（策略 0）的适用边界**：
- ✅ 适用：softmax 系（行 max 归一使分母 ≥ 1）、有 invalid line 哨兵修复的 attention 系、分母有数学下界的算子（FA/Softmax/LogSoftmax 及仓内 20+ FA 变体共享此模式）
- ❌ 不适用：裸 Div/Reciprocal（如 1/x、RMSNorm 的 1/sqrt(var)、无 max 归约的除法）——分母可真正落入 subnormal，必须走证据链 + 策略 A/B/C
- 实测案例（FA-score，arch22/arch35 双代一致）：softmax 在线算法 `p[i]=exp(score[i]−rowmax)` 使 `p[argmax]=1`、`denom≥1`；全 -inf 行由 `SoftmaxInvalidLineCheck` 检测哨兵后 `SoftmaxSumUpdate` 强制 `sum=1.0`——Div 分母被结构性排除在 subnormal 之外，保持 INTRINSIC 是正确设计（盲目加 `PRECISION_1ULP_FTZ_FALSE` 会损失 2 倍以上 Div 性能而无精度收益）

### 1.5 Subnormal 决策流程

```
算子是否使用 Exp/Ln/Sqrt/Div/Reciprocal/Rsqrt？
├─ 否 → 无需处理 Subnormal
└─ 是 → 【策略 0】是否存在结构性 subnormal 排除？（★ MUST 基于证据链：max 归一锚点 / 哨兵修复 / 数学下界推导，见 §1.4）
    ├─ 是 → 保持默认 INTRINSIC 模式（高性能），在分析摘要记录证明链
    └─ 否 → 输入数据是否可能包含 Subnormal 浮点数？（★ MUST 基于证据，见 §1.4）
        ├─ 否 → 使用默认 INTRINSIC 模式（高性能，Subnormal→0）
        └─ 是 → 算子是否在性能关键路径？
            ├─ 是 → 优先 eps 规避（策略 A）；不可行则开 high_precision 分支（策略 B/C）
            └─ 否 → 直接使用 PRECISION_1ULP_FTZ_FALSE（策略 B/C）
```

### 1.6 ★ eps / 常量 dtype 可表示性检查（所有算子 MUST 执行）

**核心原则**：源码中存在 `+eps` / `+epsilon` / 分母保护常量 **不能直接证明 eps protection 有效**。MUST 检查 eps 值在所有支持 dtype 下的实际表示。

#### 1.6.1 适用范围

以下类型的常量 MUST 进行 dtype 可表示性检查：

| 常量类型 | 源码中的常见形式 | 举例 |
|---------|----------------|------|
| eps / epsilon | `+ epsilon` / `+ eps` / `+ 1e-8` | `denom = sqrt(x) + eps` |
| 分母保护常量 | 除法分母中加的小常数 | `result = a / (b + 1e-8)` |
| 数值稳定常量 | 防止溢出/下溢的小常数 | `x = x + 1e-38` |
| 其他标量常量 | 硬编码的浮点常量 | `scale = 0.0001` |

#### 1.6.2 检查方法

对每个识别到的常量，针对算子支持的**每个 dtype** 执行以下检查：

| 序号 | 检查项 | 检查方法 | 命中后的处理 |
|------|--------|---------|-------------|
| 1 | 是否可表示 | 将常量值转为目标 dtype，检查是否为 0 | 转为 0 → 标记为"不可表示" |
| 2 | 是否为 normal | 检查 `abs(value) >= finfo(dtype).tiny` | 低于 tiny → 继续检查 3 |
| 3 | 是否为 subnormal | 检查 `0 < abs(value) < finfo(dtype).tiny` | subnormal → 在 A5 INTRINSIC 模式下变为 0 |
| 4 | 是否低于 minimum subnormal | 检查 `abs(value) < finfo(dtype).tiny / (2^(mantissa_bits-1))` | 低于 → 在任何平台都下溢为 0 |
| 5 | 转换后是否变为 0 | `np.dtype(value) == 0` | 为 0 → 检查是否导致除零/NaN |
| 6 | 是否可能导致 division by zero | 常量在分母位置且转换后为 0 | 标记为"除零风险" |
| 7 | 是否可能导致 NaN/Inf | 常量在关键计算路径且转换后为 0 | 标记为"NaN 风险" |

#### 1.6.3 dtype 数值范围参考

| dtype | minimum normal (tiny) | minimum subnormal | 典型不可表示值 |
|-------|----------------------|-------------------|---------------|
| FP32 | ~1.18e-38 (2^-126) | ~1.40e-45 (2^-149) | < 1.40e-45 |
| FP16 | ~6.10e-5 (2^-14) | ~5.96e-8 (2^-24) | 1e-8 (< subnormal min) |
| BF16 | ~1.18e-38 (2^-126) | ~9.18e-41 (2^-133) | < 9.18e-41 |

**典型问题案例**：`eps = 1e-8` 在 FP16 中：
- FP16 minimum normal = 6.10e-5，FP16 minimum subnormal = 5.96e-8
- 1e-8 < 5.96e-8 → **低于 minimum subnormal，在任何平台都下溢为 0**
- 如果 eps 在分母位置（如 `a / (sqrt(b) + eps)`），eps=0 导致除零 → NaN

#### 1.6.4 输出要求

在 `analysis_<op_name>.md` 的 §7.1 中 MUST 包含以下表格：

```markdown
| 常量名 | 值 | 位置 | FP32 可表示? | FP16 可表示? | BF16 可表示? | 风险 |
|--------|-----|------|------------|------------|------------|------|
| eps | 1e-8 | 分母保护 | 是(normal) | 否(下溢为0) | 是(normal) | FP16下除零→NaN |
```

---

### 1.7 subnormal 引发的精度验证度量问题（阶段 4 查阅）

A5 subnormal 裁剪除造成"输入 subnormal → 输出 0"的直接精度丢失外，还会在**精度验证阶段**表现为相对误差（MARE 分析指标）爆炸——这是**度量放大**而非计算错误，与 §1.1-1.5 的适配问题性质不同，判别与处置见 `stages/stage_4_precision.md` Step 4.6.3（标杆精度对齐 → denormal 域分桶 → 原版对照三步判别）。两类现象速查：

| 现象 | 本质 | 判别 |
|------|------|------|
| 输入/中间值 subnormal → 输出清零 | 真实精度丢失（A5 裁剪） | 输入分桶：subnormal 元素对应输出为 0 而 CPU 参考非 0 → 走 §1.3 适配策略 |
| matched_ratio 未达标、超标元素占比极低且集中在 \|golden\|<最小正常数域 | 相对误差度量在 denormal 域无判别力（1 ulp 绝对差被放大为 8%+ 相对差） | Step 4.6.3 第 2 步分桶统计；abs 误差 ≤ 数 ulp 即归类 dtype 表示粒度 |

## 2. 兼容性差异：Mmad 不支持 int4

### 2.1 影响范围

整体兼容性影响不大，**只有 Mmad 不支持 int4 类型会影响量化 matmul 计算**。

可能影响的算子类别：
- QuantBatchMatmul 等量化 mm 算子
- 使用 int4b_t 作为 Cube 计算输入的算子

### 2.2 适配方案

当算子使用 int4 量化输入做 Mmad 时，需**先 cast 成 int8**：

```cpp
// 220x 版本（Mmad 支持 int4）：
// 直接使用 int4b_t 作为 Mmad 输入

// 351x 版本（Mmad 不支持 int4，需先 cast）：
// int4 → int8 cast 后再送入 Mmad
Cast(int8Tensor, int4Tensor, RoundMode::CAST_RINT, count);
// ... Mmad 使用 int8 输入 ...
```

**性能影响**：额外的 cast 操作会带来性能劣化，需在性能验证阶段量化评估。

### 2.3 兼容性扫描清单

| 扫描项 | 检测方法 | 命中后的处理 |
|--------|---------|-------------|
| 是否使用 int4b_t 作为 Mmad 输入 | grep `int4b_t` + `Mmad` / `Matmul` | 需在 cast 成 int8 后再计算 |
| 是否使用 4:2 稀疏 | grep `Sparse` / `4:2` | A5 不支持 4:2 稀疏，需移除或替换实现 |

### 2.4 架构差异全景（兼容性相关）

| 维度 | 220x（A2/A3） | 351x（A5） | 影响 |
|------|--------------|-----------|------|
| int4b_t Cube 计算 | ✓ 支持 | ✗ 不支持 | 量化 matmul 需 cast |
| 4:2 稀疏 | ✓ 支持 | ✗ 不支持 | 需移除或替换 |
| L1→GM 通路 | ✓ 支持 | ✗ 删除 | 需调整数据通路 |
| GM→L0A/L0B 通路 | ✓ 支持 | ✗ 删除 | 需调整数据通路 |
| UB→L1 通路 | ✗ 不支持 | ✓ 新增 | 可利用新通路 |
| L0C→UB 通路 | ✗ 不支持 | ✓ 新增 | 可利用新通路 |
| SetLoadDataBoundary | ✓ 支持 | ✗ 删除 | 需移除调用 |
| L0A 分形格式 | ZZ | NZ | 需调整分形 |

---

## 3. 性能差异：指令代际退化

A5 相对 A2/A3 存在三类由底层指令能力、吞吐或软件实现方式变化引起的性能风险。

### 3.1 总览

| 类型 | A2/A3 主要能力 | A5 主要路径 | 根因摘要 | 影响范围 |
|------|--------------|-----------|---------|---------|
| `TransDataTo5HD` | 高吞吐 UB 转置（`vnchwconv`） | 仍以 `vnchwconv` 完成 UB 内转置 | `vnchwconv` 受 UB 写口带宽限制，有效转置吞吐下降 | 约 225 个调用点、71 个文件 |
| 高阶 `ReduceSum` | AR 小 R 可通过 DataBlock strided load 批量处理 | 特定分支使用 `VSLDB` | `VSLDB` 吞吐约为 A2/A3 的一半 | 集中在 Reduce、Softmax/LogSoftmaxGrad、Loss 等 |
| `BilinearInterpolation` | A2 原生 `VBI`；A3 批量 Vector repeat | Reg API 双层循环 | 失去原生融合，串行依赖和细粒度指令数增加 | 当前 built-in 无直接调用 |

**优先级**：

| 优先级 | 问题 | 原因 |
|--------|------|------|
| P0 | `TransDataTo5HD` | 现有调用面最广，大量算子在 tile 主循环或全量布局转换路径 |
| P0 | `ReduceSum` AR 小 R | 已确认 A5 `VSLDB` 吞吐约减半 |
| P1 | `BilinearInterpolation` | 单 API 潜在退化大，但当前 built-in 无直接调用 |

---

### 3.2 TransData / TransDataTo5HD（P0）

#### 3.2.1 实现关系

`TransDataTo5HD` 在算子实现中使用非常广泛（约 225 个调用点，71 个文件）。它常被当作通用 UB tile 转置原语使用，底层核心能力为 `vnchwconv`/scatter-vnchwconv 类指令。

#### 3.2.2 代际变化根因

**A5 上 `vnchwconv` 的性能受 UB 写口带宽限制，导致 `TransDataTo5HD` 的有效转置吞吐相对 A2/A3 下降。**

性能影响取决于：
- 是否每个输入/输出元素都经过转置
- 是否需要正反两次转置
- 是否位于 tile 主循环
- tile 是否很小（固定成本占比高）
- `vnchwconv` 的 UB 写回能否与其他流水有效重叠
- 算子本身的计算密度

#### 3.2.3 算子类别影响排序

| 优先级 | 算子类别 | 代表算子 | 影响 | 原因 |
|--------|---------|---------|------|------|
| P0 | 纯转置/格式转换 | `TransposeV2`、ND↔NZ、ND↔ZN | 极高 | 几乎所有数据都经过转置，缺少重计算掩盖 |
| P0 | Padding/Unfold 类重排 | `PadV4Grad`、`PadV3GradReplicate`、`ReflectionPad3DGrad`、`UnfoldGrad` | 极高 | Copy→Transpose→累加→Transpose→Copy，转置占比高 |
| P0/P1 | 图像采样 | `GridSample2D/3D`（nearest、小通道路径） | 高 | 转置处于通道/tile 主路径，常伴随同步 |
| P1 | Gather/Scatter | `GatherElementsV2`、`GatherV3`、`ScatterElementsV2` | 高 | 数学计算量低，转置覆盖整个 tile |
| P1 | small-R Softmax/LogSoftmax | `SoftmaxV2`、`SoftmaxGrad`、`LogSoftmaxV2/Grad` | 高 | 小 R 下算术工作量小，正反转置和同步固定成本占比大 |
| P1/P2 | 3D Pooling | `MaxPool3DWithArgmaxV2`、`MaxPool3DGradWithArgmax` | 中高 | NCDHW/NC1DHWC0 双向转换 |
| P2 | Attention 内部重排 | PFA NZ Softmax、Swin Attention/QKV | 局部高、端到端中等 | Softmax/QKV 敏感，但端到端有 MatMul 可摊薄 |
| P2/P3 | MatMul/Conv | `MatMulV3`、量化 MatMul、`Conv3D` | 通常中低 | 大 shape 下 Cube 计算占主导 |

#### 3.2.4 高风险维度

| 风险维度 | 高风险形态 | 说明 |
|---------|----------|------|
| 被交换轴大 | H/W、R/inner axis 大 | 全量转置数据量增大 |
| 外层 A 大 | batch、row、query、token 多 | 相同小 tile 转置被重复调用 |
| 通道小或尾块多 | C 小、非 16/32 对齐 | 单次转置有效数据少，固定成本占比高 |
| 正反转换 | ND→NZ→ND、NCDHW→5HD→NCDHW | 同一数据经历两次转置 |
| 输出/输入非对齐 | W 小、尾块比例高 | 需要额外 padding、mask |

#### 3.2.5 适配与优化方向

- 围绕 UB 写口瓶颈**减少转置总写入量和正反转换次数**
- 尽量**融合转置与后续计算**，通过合理 tile 和流水安排降低同步与写口竞争
- 纯转置类算子考虑替代实现路径（如直接用 DataCopy + 重组）

#### 3.2.6 回归建议

- `TransposeV2`：大张量和大量小 tile 两类 shape
- `PadV4Grad/UnfoldGrad`：小 W、非对齐 W、边界占比大的 shape
- `GridSample`：nearest/bilinear/bicubic；C=8/16/24/32/64；大输出空间
- `GatherElementsV2`：连续索引与随机索引分别测试
- small-R Softmax：R=8/16/24/32/64
- ND↔NZ MatMul：小矩阵、窄矩阵、多 group、尾块比例高的 shape

---

### 3.3 高阶 ReduceSum 与 VSLDB（P0）

#### 3.3.1 AR 模式

`Pattern::Reduce::AR` 将输入逻辑上看作 `[A, R] → [A]`，其中 A 为保留轴，R 为连续内层归约轴。

当 R 较小时，逐行加载会造成向量寄存器利用率不足。arch35 的实现会同时处理多个 A 行，按 DataBlock 从多行中抽取数据，再执行 DataBlock 级 ReduceSum，内部产生跨行、跨 DataBlock 的规则非连续访问，使用 `VSLDB`。

#### 3.3.2 代际变化根因

**已知 A5 上 `VSLDB` 性能约为 A2/A3 的一半。** 高阶 `ReduceSum` 只有下面这条窄分支会使用 `VSLDB`：

```
Pattern = AR
dtype = float / int32 / uint32
R 按 32B 对齐
R < 半个 VL
R != 一个 DataBlock
```

按 256B VL、b32 推导，典型命中 shape 为 `[A,16] → [A]` 和 `[A,24] → [A]`。

| R | innerFoldNum | strideA | VSLDB 典型配置 |
|---:|---:|---:|---|
| 16 | 2 | 16 | 首次 `(2,16)`，后续 `(2,1)` |
| 24 | 3 | 24 | 首次 `(3,24)`，后续 `(3,1)` |

#### 3.3.3 性能影响上限（Amdahl 模型）

若 A2/A3 上 `VSLDB` 占原算子时间比例为 `f`，A5 上该部分耗时变为两倍：

```
T_A5 / T_A2 ≈ 1 + f
吞吐比 ≈ 1 / (1 + f)
```

| 原 VSLDB 时间占比 | A5 预计耗时增加 | 吞吐下降 |
|---:|---:|---:|
| 10% | 10% | 约 9% |
| 20% | 20% | 约 17% |
| 30% | 30% | 约 23% |
| 50% | 50% | 约 33% |
| 80% | 80% | 约 44% |
| 100% | 100% | 50% |

#### 3.3.4 算子类别影响排序

| 优先级 | 算子类别 | 代表算子 | 触发场景 | 影响 |
|--------|---------|---------|---------|------|
| P0 | 纯求和归约 | `ReduceSum`、部分 `ReduceMean` | AR、b32、R=16/24、A 大 | 极高 |
| P0 | 小轴梯度归约 | `LogSoftmaxGrad` AR recompute、部分 `SoftmaxGrad` | `ubFactor=16/24` | 高 |
| P1 | 简单逐元素+求和 | `SquareSumV1`、`L2Loss`、`Dot`、`MSELoss`、`LpLoss` | 中间结果按 AR 小 R 归约 | 中高 |
| P1 | 小轴 Softmax | `SoftmaxV2` AR recompute | `ubFactor=16/24` | 中高 |
| P1 | 小类别交叉熵 | `CrossEntropyLoss` full-load | C 或 `cOnceNum=16/24` | 中高 |
| P2 | 梯度聚合 | `BiasAddGrad`、`PReluGradReduce` | tiling 后局部 AR R=16/24 | 中等 |
| P2 | 复杂 Loss/复合归约 | `KLDivV2`、`SmoothL1LossV2`、`BinaryCrossEntropy` | Sum 子阶段命中 AR 小 R | 中低 |
| P2 | 多阶段统计归约 | `ReduceLogSumExp`、`ReduceVar/Std` | Sum 阶段命中 | 中低 |
| P3 | 大隐藏维归约 | `RMSNormGrad` | 常见 `colsAlign2VL >= 128` | 很低 |

#### 3.3.5 高风险维度

| 上层维度 | 高风险值 | 说明 |
|---------|---------|------|
| 最后归约轴 R | 16、24 | 直接命中 b32 AR 小 R VSLDB 路径 |
| 外层 A | 大 | VSLDB 批量跨行归约循环次数增加 |
| Softmax axis | 16、24 或 tiling 后 ubFactor=16/24 | Softmax/LogSoftmaxGrad 风险集中 |
| class 数 C | 16、24 | CrossEntropy full-load 小类别场景 |
| tile 内部归约宽度 | 16、24 | 即使全局轴很大，特殊 tiling 也可能命中 |

#### 3.3.6 明确不受影响的场景

- `Pattern::Reduce::RA`
- `int64_t/uint64_t` B64 ReduceSum
- b32 的 R=8、32、40、48、56、64 或 R>64
- 非 32B 对齐的 unaligned 路径
- `BlockReduceSum`、`WholeReduceSum`、`RepeatReduceSum`、`PairReduceSum`、`MicroAPI::ReduceSum`

#### 3.3.7 适配与优化方向

- 为 R=16/24 比较 `VSLDB` 与普通 load/手工重排替代实现
- 按 A 大小选择策略（A 大时批量处理更有效）
- 考虑 tiling 调整使 R 不落入 16/24 窄分支

#### 3.3.8 回归建议

- `ReduceSum<float>`：`[A,16]`、`[A,24]`，A 从几十到数万
- 对照组：R=8、32、64、128，确认退化仅集中在 VSLDB 分支
- `LogSoftmaxGrad`：最后轴 16/24，并确认选择 AR recompute tiling
- `SoftmaxV2`：最后轴 16/24
- `CrossEntropyLoss`：C=16/24，大 batch
- `SquareSum/L2Loss/Dot/MSELoss`：局部归约轴 16/24

---

### 3.4 BilinearInterpolation（P1）

#### 3.4.1 三代实现差异

| 架构 | 实现方式 | 核心路径 | 特点 |
|------|---------|---------|------|
| A2/dav_m200 | 原生融合实现 | `VADDS(offset地址化) + VBI` | 一条 VBI 完成 Gather、权重、乘加、累加和写回 |
| A3/dav_c220 | 批量 Vector API 模拟 | `GatherB + Brcb + Mul + Add + Adds` | 需要临时 UB，但能跨 `hRepeat*vRepeat` 批量处理 |
| A5/dav_c310 | Reg API 双层循环 | `LoadOffset + GatherB + LoadWeight + Mul + Add + VSSTB` | 无大临时 UB，但指令拆分细、循环和依赖链明显 |

A5 每个调用的核心结构：

```
for i in vRepeat:
    Duplicate(dstReg, 0)
    for j in hRepeat:
        Load offset
        GatherB
        Load/Broadcast weight
        Mul
        Add to dstReg
    VSSTB output
```

指令组数量近似：`vRepeat × (2 + 5 × hRepeat)`。当 `hRepeat=4` 时，每个 `vRepeat` 大约包含 22 组寄存器操作。

#### 3.4.2 代际变化根因

1. **A2 原生 VBI 融合能力丢失**：A5 需显式展开 Gather、权重加载、Mul、Add 和 Store
2. **A3 批量 repeat 变成 A5 双层细粒度循环**：A3 可覆盖多个 repeat，A5 对每个 `(vRepeat,hRepeat)` 分别执行
3. **LoadIndex→GatherB 依赖**：每次 GatherB 必须等待 index register 就绪，缺少 index/Gather 双缓冲
4. **Mul→Add 串行累加链**：同一个 `dstReg` 被连续读写，依赖长度随 `hRepeat` 增加
5. **每个 vRepeat 独立 VSSTB 写回**：非连续 `dstBlkStride` 时风险更高
6. **mask 和小 tile 利用率**：通道/headDim 小或尾块多时，每条指令有效 lane 少

#### 3.4.3 API 参数到上层维度映射

| API 参数 | 底层含义 | 常见上层维度 |
|---------|---------|------------|
| `hRepeat` | 每个输出组累计的采样组数 | 邻点数、sampling ratio、point/level 展开维 |
| `vRepeat` | 一次调用生成的输出组数 | 输出 H×W、ROI bin、query、sampling point tile |
| `mask` | 每组有效向量元素 | C、headDim、embedDim 及尾块 |
| `dstBlkStride` | 同组 DataBlock 写回间隔 | 通道布局、NCHW/ND、head/query 交错布局 |
| `vROffset` | 相邻输出组距离 | 相邻像素、ROI bin、query/head 的跨度 |
| `src0Offset` | 每个 lane 的输入地址 | 输入 H/W、batch、head、level、采样坐标 |
| `repeatMode` | 权重广播/展开模式 | 权重是否逐点广播或已按 block 展开 |

#### 3.4.4 高风险维度

| 风险项 | 高风险上层维度 | 原因 |
|--------|-------------|------|
| `vRepeat` 大 | Hout×Wout 大、ROI 多、query/head 多 | A5 每个输出组单独循环和 VSSTB |
| `hRepeat` 大 | samplingRatio 大、numPoints/numLevels 多 | Gather/Mul/Add 次数增加，累加依赖链加长 |
| `dstBlkStride != 1` | NCHW 空间向量化、head/query 交错、带 padding 输出 | 依赖非连续 VSSTB 写回 |
| mask 小 | C/headDim 小、C 非 16 对齐、尾块比例大 | 有效 lane 少，固定指令开销占比高 |
| offset 随机 | GridSample、ROI、deformable offset、多尺度 level | GatherB 地址局部性差 |
| `vROffset` 大 | 相邻输出跨 head/query/bin | 写回局部性下降 |
| `repeatMode=false` 且 hRepeat 大 | 每个采样组使用独立标量权重 | 内层每次都需要 BRC load |

#### 3.4.5 算子类别影响排序（理论风险）

当前 built-in 实现中没有发现直接调用基础 `AscendC::BilinearInterpolation`。以下排序表示开发者算子使用该 API 或未来迁移时的理论风险：

| 优先级 | 算子类别 | 高风险 shape | 理论影响 |
|--------|---------|------------|---------|
| P0 | GridSample/SpatialTransformer | Hout×Wout 大、C 小、随机 grid、NCHW 非连续输出 | 极高 |
| P0 | ROIAlign/CropAndResize | ROI 多、pooledH/W 大、samplingRatio 大、C 小 | 极高 |
| P0 | Deformable Sampling/Attention | query/head/level/point 多、offset 随机、headDim 小 | 极高 |
| P1 | ResizeBilinear/UpsampleBilinear | Hout×Wout 大、上采样倍数大 | 高 |
| P1 | 多尺度特征融合 | level 多、输出点多、跨 level 访问 | 高 |
| P2 | 小图像规则 Resize | H/W 小、offset 规则、输出连续 | 中低 |
| P2 | 大且连续通道采样 | C/headDim 大且 16 对齐、满 mask、stride=1 | 中低 |
| P3 | 单点/少量采样 | vRepeat、hRepeat 和调用次数均小 | 低 |

#### 3.4.6 适配与优化方向

- 增加跨 vRepeat 的软件流水/批处理
- 预取 index（index/Gather 双缓冲）
- 采用多累加器或树形累加（减少 Mul→Add 串行依赖链）
- `dstBlkStride=1` 时考虑普通连续 Store 替代 VSSTB

#### 3.4.7 回归建议

- `hRepeat`：1、2、4、8
- `vRepeat`：1、2、4、8、16、32
- `dstBlkStride`：1、2、4、8
- mask：满 VL、半 VL、单 DataBlock、小通道尾块
- offset：连续、固定间隔、随机、大跨度跨 level
- `repeatMode`：true/false
- 独立测试 `GatherB`、`VSSTB`，并用连续 Load/Store 替换做消融实验
- 对 `hRepeat=4` 比较线性累加与树形累加，判断 RAW 依赖链占比

---

## 4. 三类问题的统一算子级优先级

| 总优先级 | 问题/算子类别 | 主要触发条件 | 风险判断 |
|---------|------------|------------|---------|
| P0 | `TransDataTo5HD`：纯转置、Pad/Unfold | 全量或双向转置、主循环、小 tile | 现有调用多，端到端影响最直接 |
| P0 | `ReduceSum`：纯 Reduce | AR、b32、R=16/24、A 大 | 已确认 VSLDB 吞吐约减半，最容易复现 |
| P0 | `TransDataTo5HD`：GridSample/Gather/Scatter | 大输出、低计算密度、同步多 | 转置与同步占比高 |
| P0/P1 | `ReduceSum`：LogSoftmaxGrad AR recompute | ubFactor=16/24 | Reduce 是后续计算的串行依赖 |
| P1 | `TransDataTo5HD`：small-R Softmax/Pooling | 小 R、双向布局转换 | 局部影响高 |
| P1 | `ReduceSum`：SquareSum/L2Loss/Dot/Softmax/CrossEntropy | 局部 R 或 C=16/24 | 被逐元素、Exp/Div 等部分摊薄 |
| P1（潜在） | `BilinearInterpolation`：GridSample/ROIAlign/Deformable | 输出点多、采样点多、随机 offset、小 C、非连续写回 | API 本身风险高，但当前 built-in 无直接调用 |
| P2 | `TransDataTo5HD`：大 MatMul/Conv | 小矩阵、多 group、尾块多时才敏感 | 正常大 shape 由 Cube 计算摊薄 |
| P2 | `ReduceSum`：复杂 Loss/统计归约 | Sum 子阶段 R=16/24 | 只影响多阶段计算中的一段 |
| P2（潜在） | `BilinearInterpolation`：规则 Resize | Hout×Wout 大 | offset 局部性较好，风险低于随机采样 |
| P3 | RA/B64/专用 Reduce；连续高利用率插值 | 不命中特定指令分支 | 无直接或很低影响 |

---

## 5. API 差异适配检查清单

在阶段 2 代码改造完成后，MUST 逐项核对：

### 5.1 精度（Subnormal）

- [ ] 算子是否使用 Exp/Ln/Sqrt/Rsqrt/Div/Reciprocal？已扫描
- [ ] 若使用，是否评估了输入 subnormal 风险？
- [ ] 已选择适配策略（eps 规避 / API 模板参数 / high_precision 分支）
- [ ] high_precision 分支仅在需要时启用，非默认路径

### 5.2 兼容性

- [ ] 算子是否使用 int4b_t 作为 Mmad 输入？已处理 cast 成 int8
- [ ] 算子是否使用 4:2 稀疏？A5 不支持，已移除或替换
- [ ] 是否检查了 L1→GM / GM→L0A/L0B 通路删除的影响
- [ ] 是否检查了 SetLoadDataBoundary 删除的影响
- [ ] L0A 分形格式 ZZ→NZ 是否已调整

### 5.3 性能

- [ ] 算子是否调用 `TransDataTo5HD`？评估了 UB 写口瓶颈影响
- [ ] 算子是否调用高阶 `ReduceSum`（AR 模式）？评估了 VSLDB 吞吐下降
- [ ] 算子是否调用 `BilinearInterpolation`？评估了 Reg API 双层循环影响
- [ ] 是否按高风险 shape 设计了性能回归用例
- [ ] 是否考虑了减少转置总写入量、融合转置与后续计算等优化

---

## 6. 最终建议

1. **先做指令 microbenchmark**：对 `vnchwconv/TransDataTo5HD` 重点测量 UB 写口带宽上限以及不同 tile/并发方式下的有效吞吐；另外分别测试 `VSLDB`、`GatherB` 和 `VSSTB`，区分指令吞吐问题与 API 软件展开问题。
2. **回归必须按触发 shape 设计**：平均随机 shape 容易掩盖 `ReduceSum R=16/24`、插值小 mask/非连续 stride 等窄路径。
3. **优先看端到端占比，不只看单指令倍率**：用 Amdahl 模型估算单指令退化对算子的上限影响。
4. **区分现有算子与开发者潜在影响**：`TransDataTo5HD` 已广泛使用；`ReduceSum` 有明确但较窄的现有影响；`BilinearInterpolation` 当前 built-in 未直接调用。
5. **A5 优化方向**：
   - 转置类：围绕 UB 写口瓶颈减少转置总写入量和正反转换次数，尽量融合转置与后续计算，并通过合理 tile 和流水安排降低同步与写口竞争。
   - ReduceSum：为 R=16/24 比较 VSLDB 与普通 load/手工重排替代实现，并按 A 大小选择策略。
   - BilinearInterpolation：增加跨 vRepeat 的软件流水/批处理，预取 index，采用多累加器或树形累加；`dstBlkStride=1` 时考虑普通连续 Store 替代 VSSTB。
