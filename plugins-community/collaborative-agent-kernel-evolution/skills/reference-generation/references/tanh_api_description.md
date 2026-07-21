# Tanh 算子 API 描述

## 1. 背景与动机

双曲正切(Tanh)函数是深度学习中常用的激活函数，特别是在循环神经网络(RNN/LSTM)中。作为平滑可微的非线性函数，Tanh将输入值映射到(-1, 1)范围，是零中心的激活函数，有助于梯度流动和训练收敛。

**主要应用场景**:
- LSTM/GRU的门控单元和细胞状态激活
- 需要零中心输出的特征归一化
- 传统前馈网络的激活层

## 2. 算子定义

### 数学公式

$$
\text{tanh}(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1}
$$

### 关键性质

- **值域**: $\text{tanh}(x) \in (-1, 1)$
- **零中心**: $\text{tanh}(0) = 0$
- **奇函数**: $\text{tanh}(-x) = -\text{tanh}(x)$
- **导数**: $\frac{d}{dx}\text{tanh}(x) = 1 - \text{tanh}^2(x)$

## 3. 接口规范

### 函数签名

```python
torch.tanh(input, *, out=None) → Tensor
```

### 参数说明

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| input | Tensor | 是 | 输入tensor，任意shape |
| out | Tensor | 否 | 与输入shape严格一致 |

### 数据类型

- **float32** (fp32): 标准精度，本benchmark使用此类型
- 注：Ascend NPU vector core内部统一转换为fp32进行运算

## 4. 计算流程

### 算法描述

```
对于输入tensor X中的每个元素 x:
    1. 计算 exp_2x = exp(2 * x)
    2. 计算 y = (exp_2x - 1) / (exp_2x + 1)
    3. 返回 y
```

**复杂度**: O(n) 时间，O(n) 空间，完全可并行

## 5. 数值特性

### 输入值范围与输出

| 输入范围 | 输出特性 | 说明 |
|---------|---------|------|
| \|x\| < 0.5 | 接近线性 | tanh(x) ≈ x |
| 0.5 < \|x\| < 3 | 非线性 | 平滑过渡 |
| \|x\| > 3 | 饱和 | tanh(x) ≈ ±1 |

### 数值精度 (float32)

- 精度: ~7位有效数字
- 数值范围: ±3.4×10³⁸
- 对于|x| < 88，计算数值稳定

### 梯度特性

- 梯度有界: $0 < \frac{d}{dx}\text{tanh}(x) \leq 1$
- 在|x| > 3时梯度很小(< 0.01)，可能导致梯度消失
- 相比sigmoid，零中心的特性有助于梯度传播

## 7. 约束与限制

### 输入约束
- 必须是PyTorch Tensor
- 数据类型: float32
- 支持任意shape

### 特殊值处理
- `tanh(0) = 0`
- `tanh(±inf) = ±1`
- `tanh(nan) = nan`

## 8. Golden定义

```python
import torch

def tanh_golden(self, x: torch.Tensor) -> torch.Tensor:
   """
   Apply tanh activation

   Args:
      x: input tensor [..., C]

   Returns:
      output tensor [..., C] with values in (-1, 1)
   """
   return torch.tanh(x)
```

## 9. 测试用例

### 用例总览

| 编号 | 名称 | Shape | 分布类型 | dtype | 说明 |
|------|------|-------|---------|-------|------|
| 1 | small_1d | (512,) | normal | float32 | 基础1D形状 |
| 2 | small_2d | (32, 256) | normal | float32 | 基础2D形状 |
| 3 | medium_3d | (4, 128, 768) | normal | float32 | 典型LLM: batch×seq×hidden |
| 4 | large_3d | (8, 512, 4096) | normal | float32 | 大规模LLM形状 |
| 5 | multihead_4d | (4, 12, 128, 64) | normal | float32 | 多头注意力: batch×heads×seq×head_dim |
| 6 | uniform_distribution | (4, 128, 768) | uniform_sym | float32 | 对称均匀分布 [-1, 1] |
| 7 | moderate_range | (4, 128, 768) | moderate | float32 | 非线性区间 [-3, 3] |
| 8 | near_zero | (4, 128, 768) | near_zero | float32 | 近零区间 [-0.01, 0.01]，近似线性区 |
| 9 | large_magnitude | (4, 128, 768) | large_values | float32 | 大值区间 [-10, 10]，测试饱和行为 |
| 10 | non_aligned | (3, 127, 1023) | normal | float32 | 非2幂次维度 |

### 分布类型说明

| 分布类型 | 取值范围 | 测试目的 |
|---------|---------|---------|
| normal | 标准正态 N(0,1) | 常规输入，覆盖线性与非线性区 |
| uniform_sym | 均匀 [-1, 1] | 非正态输入的泛化性 |
| moderate | 均匀 [-3, 3] | 主要覆盖非线性过渡区 |
| near_zero | 均匀 [-0.01, 0.01] | 近线性区 tanh(x) ≈ x |
| large_values | 均匀 [-10, 10] | 饱和区 tanh(x) ≈ ±1 |

## 10. 参考文献

**官方文档**:
- PyTorch torch.tanh: https://pytorch.org/docs/stable/generated/torch.tanh.html
- PyTorch nn.Tanh: https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html
