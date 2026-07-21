# A6: High-Precision rsqrt via Newton-Raphson (高精度rsqrt)
## Overview
Layer Norm的归一化需要计算1/sqrt(var + eps)。专家实现使用Newton-Raphson迭代提高rsqrt精度：初始近似y0 = rsqrt(x)（硬件指令，精度约8bit），第一次迭代y1 = y0 * (1.5 - 0.5 * x * y0^2)，第二次迭代y2 = y1 + 0.5 * y1 * (1 - x * y1^2)。相比单次sqrt+div，迭代方法避免了除法指令的高延迟，同时获得更高的精度（接近FP32全精度）。

## When to Use
- Normalization ops needing rsqrt
- 更高的精度和更好的性能，特殊值处理确保数值稳定性
- rsqrt相对误差从1%降低到0.001%，满足训练精度要求
- rsqrt精度从1%提升到0.001%，特殊值处理更安全

## Trade-off
- 代码更复杂，需要理解牛顿迭代原理
- 增加约6条Vector指令，适合训练场景
- 增加约8-10条Vector指令

**Source operators**: inplace_add_rms_norm, layer_norm_v3, layer_norm_v4

---

## Variant A: 牛顿迭代优化RMS计算
Source: inplace_add_rms_norm, layer_norm_v4

专家实现在RMS计算中使用了牛顿迭代法来优化倒数平方根的计算。传统的sqrt + div操作需要两次函数调用，而牛顿迭代法通过一次sqrt和几次乘加操作实现了更高的精度和性能。具体实现中，先通过Sqrt指令计算近似值，然后通过两次牛顿迭代来精化结果。此外，实现还包含了对特殊值（0和无穷大）的处理，通过CompareScalar和Select指令确保数值稳定性。

**Expert implementation:**
```cpp
// 专家实现 - 牛顿迭代
static constexpr float SCALAR1 = -0.5;
static constexpr float SCALAR2 = 1.5;

Div(r, one, var, pregLoop);   // 初始近似
Sqrt(y, r, pregLoop);
Muls(t, var, SCALAR1, pregLoop);
Mul(t, t, y, pregLoop);
Mula(t1, t, y, pregLoop);
Mul(rstd, y, t1, pregLoop);

// 特殊值处理
CompareScalar(cmpRegZero, var, POS_INF, pregLoop);
Select(rstd, scalarZero, rstd, cmpRegZero);
CompareScalar(cmpRegInf, var, float(0.0), pregLoop);
Select(rstd, scalarInf, rstd, cmpRegInf);
```

**vs. baseline:**
```cpp
// baseline - 传统方法
AscendC::Adds(rmsLocal, meanSqLocal, this->eps, 1);
AscendC::Sqrt(rmsLocal, rmsLocal, 1);
float rmsVal = rmsLocal.GetValue(0);
AscendC::Reciprocal(rmsLocal, rmsLocal, 1);
float invRms = rmsLocal.GetValue(0);
```

Benefit: 更高的精度和更好的性能，特殊值处理确保数值稳定性; 更高的数值精度和指令吞吐，正确处理特殊边界值
Trade-off: 代码更复杂，需要理解牛顿迭代原理; 增加指令数量，需要更多寄存器资源

---

## Variant B: 高精度rsqrt的Newton-Raphson迭代
Source: layer_norm_v3

Layer Norm的归一化需要计算1/sqrt(var + eps)。专家实现使用Newton-Raphson迭代提高rsqrt精度：初始近似y0 = rsqrt(x)（硬件指令，精度约8bit），第一次迭代y1 = y0 * (1.5 - 0.5 * x * y0^2)，第二次迭代y2 = y1 + 0.5 * y1 * (1 - x * y1^2)。相比单次sqrt+div，迭代方法避免了除法指令的高延迟，同时获得更高的精度（接近FP32全精度）。

**Expert implementation:**
```cpp
// Newton-Raphson迭代
Adds(var, var, epsilonLocal, pregLoop);
Div(r, one, var, pregLoop);
Sqrt(y, r, pregLoop);
Muls(t, var, float(-0.5), pregLoop);
Mul(t, t, y, pregLoop);
Mula(t1, t, y, pregLoop);
Mul(rstd, y, t1, pregLoop);
// ... second iteration
Mula(rstd, s, scalar1, pregLoop);
```

**vs. baseline:**
```cpp
// 简单实现
float stdVal = sqrt(varVal + this->eps);
AscendC::Muls(yLocal, yLocal, 1.0f / stdVal, this->tileLength);
```

Benefit: rsqrt相对误差从1%降低到0.001%，满足训练精度要求
Trade-off: 增加约6条Vector指令，适合训练场景

---

## Variant C: 高精度rstd计算
Source: layer_norm_v3

专家实现使用两次Newton-Raphson迭代，将rsqrt的相对误差从硬件指令的约1%降低到约0.001%，满足深度学习训练对梯度精度的要求。同时，通过Maxs(var, var, SCALAR0)和CompareScalar处理特殊值（Inf、NaN），确保数值安全。

**Expert implementation:**
```cpp
// 高精度Newton-Raphson迭代
Maxs(var, var, SCALAR0, pregMerge);  // avoid subnormal
CompareScalar(cmpRegZero, var, POS_INF, pregMerge);
Select(rstd, scalarZero, rstd, cmpRegZero);  // if var==inf, rstd=0
CompareScalar(cmpRegInf, var, zero, pregMerge);
Select(rstd, scalarInf, rstd, cmpRegInf);    // if var==0, rstd=inf

// Newton-Raphson iteration
Mul(rstd, y, t1, pregLoop);
Mula(rstd, s, scalar1, pregLoop);
```

**vs. baseline:**
```cpp
// 简单实现，精度较低
float stdVal = sqrt(varVal + this->eps);
AscendC::Muls(yLocal, yLocal, 1.0f / stdVal, this->tileLength);
```

Benefit: rsqrt精度从1%提升到0.001%，特殊值处理更安全
Trade-off: 增加约8-10条Vector指令
