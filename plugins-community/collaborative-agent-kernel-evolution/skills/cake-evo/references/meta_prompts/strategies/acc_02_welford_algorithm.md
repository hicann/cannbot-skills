# A2: Welford Numerically Stable Mean/Var (Welford数值稳定算法)
## Overview
在训练模式下，BatchNorm需要计算当前batch的均值和方差用于前向传播，同时更新running_mean和running_var。专家实现正确处理了样本方差（除以N-1）与总体方差（除以N）的区别：1）Batch统计量：使用batchVarScale系数将总体方差转换为样本方差（无偏估计），当patternR0 * patternR1 > 1时，batchVarScale = N / (N-1)；2）Running统计量：running_mean和running_var使用指数移动平均（EMA）更新，不涉及N-1校正；3）特殊值处理：当patternR0 * patternR1 == 1时（单样本场景），避免除以0，直接设置batchVarScale = 1.0。

## When to Use
- LayerNorm, BatchNorm, RMSNorm
- 解决Adam发散问题；提升训练稳定性；支持Adam/AMSGrad切换
- 训练收敛性更好；与PyTorch/TensorFlow标准实现对齐
- 大维度reduce场景下精度提升1-2个数量级，避免catastrophic cancellation

## Trade-off
- 额外的Max操作；需要维护max_grad_norm状态
- 需要额外的乘法操作；需要特殊处理单样本场景
- 算法复杂度增加，需要维护额外的累加器状态

**Source operators**: apply_adam_w_v2, batch_norm_v3, layer_norm_v3, layer_norm_v4, scaled_masked_softmax_grad_v2

---

## Variant A: AMSGrad变体的数值稳定性
Source: apply_adam_w_v2

专家实现完整支持AMSGrad变体，该变体通过维护`max_exp_avg_sq`解决了Adam算法在某些情况下可能发散的问题。在代码中，当`amsgrad_`为true时，算法会计算`max(max_grad_norm, exp_avg_sq)`作为分母，确保学习率不会突然增大。

**Expert implementation:**
```cpp
if (amsgrad_) {
    PipeBarrier<PIPE_V>();
    Max(dataOutLocal[maxGradOutOffset_], dataLocal[maxGradNormOffset_], 
        dataOutLocal[expAvgSqOffset_], dataCount);
    PipeBarrier<PIPE_V>();
    Sqrt(dataLocal[varOffset_], dataOutLocal[maxGradOutOffset_], dataCount);
    PipeBarrier<PIPE_V>();
    Muls(dataLocal[expAvgOffset_], dataLocal[varOffset_], biasCorrection2Sqrt_, dataCount);
} else {
    Sqrt(dataLocal[varOffset_], dataOutLocal[expAvgSqOffset_], dataCount);
    // ...
}
```

Benefit: 解决Adam发散问题；提升训练稳定性；支持Adam/AMSGrad切换
Trade-off: 额外的Max操作；需要维护max_grad_norm状态

---

## Variant B: Batch Variance无偏估计
Source: batch_norm_v3

在训练模式下，BatchNorm需要计算当前batch的均值和方差用于前向传播，同时更新running_mean和running_var。专家实现正确处理了样本方差（除以N-1）与总体方差（除以N）的区别：1）Batch统计量：使用batchVarScale系数将总体方差转换为样本方差（无偏估计），当patternR0 * patternR1 > 1时，batchVarScale = N / (N-1)；2）Running统计量：running_mean和running_var使用指数移动平均（EMA）更新，不涉及N-1校正；3）特殊值处理：当patternR0 * patternR1 == 1时（单样本场景），避免除以0，直接设置batchVarScale = 1.0。

**Expert implementation:**
```cpp
// Host端计算Batch Variance校正系数
float batchVarScale = (commonParams.patternR0 * commonParams.patternR1 == 1) ?
                          1.0 :
                          static_cast<float>(
                              static_cast<double>(commonParams.patternR0 * commonParams.patternR1) /
                              static_cast<double>(commonParams.patternR0 * commonParams.patternR1 - 1));
td_.set_batchVarScale(batchVarScale);

// Kernel端应用校正
Muls(momentumVarTensor, saveVarTensor, this->batchVarScale * this->momentum, aProcNum);
```

**vs. baseline:**
```cpp
# baseline有偏估计
var_val = channel_sq_diff_sum / elements_per_channel
```

Benefit: 训练收敛性更好；与PyTorch/TensorFlow标准实现对齐
Trade-off: 需要额外的乘法操作；需要特殊处理单样本场景

---

## Variant C: Welford算法的数值稳定性
Source: layer_norm_v3

Welford算法通过增量式计算避免了传统两-pass方法的数值不稳定问题。其核心公式为Mk = Mk-1 + (xk - Mk-1) / k和Sk = Sk-1 + (xk - Mk-1) * (xk - Mk)，其中Mk是当前mean，Sk是M2累加器（用于计算variance）。这种增量方式始终保持数值在合理范围，避免了(x - mean)^2中可能出现的大数相减。

**Expert implementation:**
```cpp
// Welford增量式计算
WelfordUpdateParam para;
para.rnLength = 1;
para.abLength = td_->tileLength;
para.nRec = 1.0f / static_cast<float>(welfordCount);
WelfordUpdate<T, float, false>(mean_, variance_, mean_, variance_, xTensor, shared_, para);
```

**vs. baseline:**
```cpp
// 传统Two-Pass，存在大数相减
float meanVal = rowSum / this->tileLength;
AscendC::Adds(sharedLocal, xLocal, -meanVal, this->tileLength);
AscendC::Mul(sharedLocal, sharedLocal, sharedLocal, this->tileLength);
```

Benefit: 大维度reduce场景下精度提升1-2个数量级，避免catastrophic cancellation
Trade-off: 算法复杂度增加，需要维护额外的累加器状态

---

## Variant D: Welford算法数值稳定性
Source: layer_norm_v4

Welford策略使用Welford在线算法计算均值和方差，这是数值分析中公认的稳定算法。通过增量更新避免大数据量时的精度损失：M_k = M_{k-1} + (x_k - M_{k-1}) / k，S_k = S_{k-1} + (x_k - M_{k-1}) * (x_k - M_k)。相比直接计算E[x^2] - (E[x])^2，在数据方差较小时避免灾难性抵消。baseline使用两趟计算，数据分布较广时引入较大数值误差。

**Expert implementation:**
```cpp
// Welford算法通过高阶API内部实现
AscendC::LayerNorm<U, T, true, hasGammaBetaConfig>(
    yInUb, batchMeanOutUb, batchRstdOutUb, xInUb, gammaInUb, betaInUb, 
    this->epsilon, binaryAddTensor, para, this->layerNormTiling);
```

**vs. baseline:**
```cpp
// 传统两趟计算
float rowSum = sharedLocal.GetValue(0);
float rowMean = rowSum / this->cols;
AscendC::Sub(tempLocal, inputLocal, meanLocal, this->cols);
AscendC::Mul(varLocal, tempLocal, tempLocal, this->cols);
AscendC::ReduceSum(sharedLocal, varLocal, sharedLocal, this->cols);
```

Benefit: 显著提升数值稳定性，特别是数据方差较小或数据量较大时
Trade-off: 计算复杂度略高，需要增量更新而非批量计算

---

## Variant E: Softmax梯度计算数值稳定性
Source: scaled_masked_softmax_grad_v2

专家实现使用SoftmaxGrad高阶API，内部实现了数值稳定的softmax梯度计算。Softmax梯度的数学公式为∂L/∂x = softmax(x) * (∂L/∂y - sum(∂L/∂y * softmax(x)))，直接实现面临指数爆炸和精度丢失问题。SoftmaxGrad API内部使用log-sum-exp技巧避免指数溢出，高精度累加使用Kahan求和等算法提高累加精度，规范化处理确保概率和为1的数值稳定性。

**Expert implementation:**
```cpp
SoftMaxShapeInfo srcShape = {
    static_cast<uint32_t>(this->lineNum), 
    static_cast<uint32_t>(this->paddedHeadDim_),
    static_cast<uint32_t>(this->lineNum), 
    static_cast<uint32_t>(this->paddedHeadDim_)};
SoftmaxGrad<float, false, false>(
    xGradLocal, yGradLocal, yLocal, softmaxGradTmpBuf, softmaxTiling_, false, srcShape);
```

**vs. baseline:**
```cpp
AscendC::Muls(tmpLocal, softmaxOutputLocal, -1.0f, this->tileSize);
AscendC::Adds(gradSoftmaxLocal, tmpLocal, 1.0f, this->tileSize);
AscendC::Mul(gradSoftmaxLocal, gradSoftmaxLocal, softmaxOutputLocal, this->tileSize);
```

Benefit: 数值稳定性保证，避免溢出和精度丢失，符合数学定义
Trade-off: 使用高阶API可能引入额外开销，需要额外buffer
