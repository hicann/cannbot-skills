# Kernel 路由：Grouped Matmul 类算子

> **占位文件**。Grouped Matmul 类算子的完整设计指南将在后续版本中补充。

## 场景定义

Grouped Matmul 处理多组独立的矩阵乘法：
- `C_i = A_i * B_i`（i = 1..G）
- 各组 shape 不同，各自独立
- 常见于 MoE（Mixture of Experts）场景

## 典型 catlass example

| Example | 说明 |
|---------|------|
| `02_grouped_matmul_slice_m` | M 方向分组的 Grouped Matmul |
| `05_grouped_matmul_slice_k` | K 方向分组的 Grouped Matmul |
| `08_grouped_matmul` | 通用 Grouped Matmul |
| `07_grouped_matmul_slice_m_per_token_dequant_moe` | MoE 场景 + 反量化 |

## 与 matmul 的关键差异

1. **DispatchPolicy**: 通常使用 Async 系列（`PreloadAsync` / `PreloadAsyncWithCallback`）
2. **NBuffer**: 需要配置 L1/L0 的多 stage 参数
3. **Tiling**: 需要对每组独立计算 offset 和 shape
4. **Kernel**: 有对应的 grouped kernel 类型

> 完整设计指南待后续补充。当前可参考 catlass 官方文档：
> - `catlass/docs/zh/2_Design/01_kernel_design/04_matmul_summary.md` 中的 grouped matmul 部分
> - 对应 example 的 README.md
