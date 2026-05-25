# Kernel 路由：Quant Matmul 类算子

> **占位文件**。Quant Matmul 类算子的完整设计指南将在后续版本中补充。

## 场景定义

Quant Matmul 处理量化后的矩阵乘法（低精度权重 + 高精度或低精度激活）：
- W8A8: 8-bit 权重 × 8-bit 激活 → 反量化 → fp16 输出
- W8A16: 8-bit 权重 × 16-bit 激活 → 反量化 → fp16 输出
- W4A8: 4-bit 权重 × 8-bit 激活 → 反量化 → fp16 输出
- W4A4: 4-bit 权重 × 4-bit 激活 → 反量化 → fp16 输出

## 关键特征

1. **AIC/AIV 协同**: AIC 做 MMAD，AIV 做反量化 + Epilogue
2. **Kernel**: 必须用 `QuantMatmulMultiStageWorkspace`
3. **DispatchPolicy**: `MmadAtlasA2PreloadAsyncWithCallback`
4. **BlockEpilogue**: AIV 侧专用的反量化 Epilogue（如 `BlockEpiloguePerTokenDequant`）
5. **Workspace**: 必须提供 AIC→AIV 中间结果缓冲（`workspaceStages` 通常为 2）

## 典型 catlass example

| Example | 说明 |
|---------|------|
| `12_quant_matmul` | W8A8 量化矩阵乘 |
| `30_w8a16_matmul` | W8A16 量化矩阵乘 |
| `32_w4a8_matmul` | W4A8 量化矩阵乘 |
| `38_w4a4_matmul_per_token_per_channel_dequant` | W4A4 双量化 |
| `11_grouped_matmul_slice_k_per_token_dequant` | 分组 + Per-token 反量化 |

## 与 matmul 的关键差异

1. **Kernel 类型**: `QuantMatmulMultiStageWorkspace`（不是 BasicMatmul）
2. **Params 差异**: 多了 `ptrScale` / `ptrPerTokenScale`
3. **DispatchPolicy**: 必须含 Callback（AIC↔AIV 同步）
4. **精度阈值**: 量化算子精度容忍度不同（参考 `ops-precision-standard`）

> 完整设计指南待后续补充。当前可参考 catlass 官方文档：
> - `catlass/docs/zh/1_Practice/10_matmul_optimization.md` 中的量化部分
> - 对应 example 的 README.md
