# GroupedMatmul (npu_grouped_matmul) 算子 API 描述

## 1. 背景与动机

分组矩阵乘法(Grouped Matrix Multiplication)是MoE(Mixture of Experts)架构的核心计算原语。与逐个执行独立矩阵乘法不同，分组矩阵乘法将多个矩阵乘法批量合并执行，减少kernel启动开销，优化内存访问模式。

**主要应用场景**:
- MoE模型中多个专家的前馈网络并行计算
- 异构batch处理(不同长度序列对应不同矩阵尺寸)
- 多头注意力的分组投影

## 2. 算子定义

### 数学公式

$$
y_i = x_i \mathbin{@} \text{weight}_i + \text{bias}_i
$$

其中 $i$ 表示第 $i$ 个分组(专家)，$@$ 表示矩阵乘法。

注：该算子还支持INT8/INT4量化模式，本benchmark仅关注bfloat16场景。

### 关键性质

- **分组独立**: 每个分组独立计算矩阵乘法
- **灵活分组**: 支持均匀和非均匀的分组大小
- **可选偏置**: bias参数可选

## 3. 接口规范

### 函数签名

```python
torch_npu.npu_grouped_matmul(
    x, weight, *,
    bias=None, group_list=None, group_type=None,) -> List[Tensor]
```

### 参数说明

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| x | Tensor | 是 | 输入token [total_M, K] |
| weight | Tensor | 是 | 专家权重 [num_groups, K, N] |
| bias | Tensor | 否 | 可选偏置 [num_groups, N] |
| group_list | Tensor | 否 | 每组token数（count模式）[num_groups]，用于沿dim-0切分输入 |
| group_type | int | 否 | 分组轴: -1=无分组, 0=M轴分组 |

### 数据类型

- **bfloat16** (bf16): 标准精度，本benchmark使用此类型
- 注：Ascend NPU cube core支持float16/bfloat16/float32，内部可能进行类型转换

## 4. 计算流程

### 算法描述

```
对于每个分组 i (i = 0, 1, ..., num_groups-1):
    1. 根据 group_list 获取第 i 组的输入 x_i
    2. 计算矩阵乘法: y_i = x_i @ weight_i
    3. 若有 bias，加上偏置: y_i = y_i + bias_i
最后拼接所有分组输出: y = concat(y_0, y_1, ..., y_{n-1})
```

**复杂度**: O(sum(m_i * K * N))，其中m_i为每组token数

## 5. 约束与限制

### 输入约束
- x和weight列表最大长度: 128
- 内维度(K)必须 < 65536
- group_list为count模式，每个元素表示该组的token数，总和须等于M

### 特殊值处理
- bias为None时不加偏置
- group_list为None时按多tensor输入模式处理

## 6. Golden定义

```python
import torch

def grouped_matmul_golden(x: torch.Tensor, weight: torch.Tensor,
                           group_list: torch.Tensor, bias=None) -> torch.Tensor:
    """
    Grouped matrix multiplication

    Args:
        x: input tokens [total_M, K]
        weight: expert weights [num_groups, K, N]
        group_list: per-group token counts [num_groups], int64
        bias: optional per-group bias [num_groups, N]

    Returns:
        output: [total_M, N]
    """
    num_groups = weight.shape[0]
    results = []
    offset = 0
    for i in range(num_groups):
        count = group_list[i].item()
        xi = x[offset:offset + count]
        yi = torch.matmul(xi, weight[i])
        if bias is not None:
            yi = yi + bias[i]
        results.append(yi)
        offset += count
    return torch.cat(results, dim=0)
```

## 7. 测试用例

### 用例总览

| 编号 | 名称 | M | K | N | num_groups | token分配 | bias | 说明 |
|------|------|---|---|---|-----------|---------|------|------|
| 1 | small_uniform | 32 | 128 | 64 | 2 | [16, 16] | ✓ | 小规模均匀 |
| 2 | small_no_bias | 32 | 128 | 64 | 2 | [16, 16] | ✗ | 无偏置 |
| 3 | small_many_groups | 64 | 64 | 32 | 8 | [8, 8, 8, 8, 8, 8, 8, 8] | ✓ | 多组小规模 |
| 4 | medium_uniform | 256 | 512 | 768 | 4 | [64, 64, 64, 64] | ✓ | 中等均匀 |
| 5 | medium_uneven | 256 | 768 | 512 | 4 | [128, 64, 32, 32] | ✓ | 中等非均匀 |
| 6 | llm_deepseek_v3 | 512 | 7168 | 2048 | 8 | [64, 64, 64, 64, 64, 64, 64, 64] | ✓ | DeepSeek-V3 FFN |
| 7 | llm_mixtral | 512 | 4096 | 14336 | 8 | [128, 64, 64, 64, 64, 64, 32, 32] | ✓ | Mixtral-8x7B FFN |
| 8 | llm_large_batch | 1024 | 4096 | 8192 | 4 | [256, 256, 256, 256] | ✓ | 大batch LLM推理 |
| 9 | nonuniform_small | 80 | 256 | 128 | 4 | [8, 24, 16, 32] | ✓ | 非均匀组（小） |
| 10 | nonuniform_large | 512 | 4096 | 4096 | 4 | [200, 56, 128, 128] | ✗ | 非均匀组（大）无偏置 |

## 8. 参考文献

**官方文档**:
- torch_npu.npu_grouped_matmul: Ascend NPU算子文档

**学术参考**:
- Hwang, C., et al. (2023). "Tutel: Adaptive Mixture-of-Experts at Scale". MLSys 2023.
- Gale, T., et al. (2023). "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts". MLSys 2023.
