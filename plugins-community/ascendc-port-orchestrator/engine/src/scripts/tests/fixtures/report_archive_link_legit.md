# X op arch22→arch35 报告（fixture: LEGIT — must pass report_archive_link_lint）

本项目归档 X 的前向与反向实现。

## 一、核心数据

### 前向（生产 kernel）—— 代表用例

| 用例 | dtype | device-time (µs) | 比例 vs arch22 |
|------|------|------|------|
| C0 | fp32 | 37.0 | **0.66×** |
| C1 | fp16 | 33.7 | **0.76×** |

归档: [x_fwd_simd](src/kernels/x_fwd_simd/) · [verification.json](src/kernels/x_fwd_simd/verification.json)

> 关键读法：前向生产 kernel，精度与 SIMT 完全一致、性能更快。

### 反向 —— 8 个梯度

| 反向实现 | 精度 | device-time（vs arch22） | 架构 |
|------|------|------|------|
| SIMD（生产） | 8 个梯度全部正确 | ~0.19× | 串行逐行 L 扫描 |
| coop-SIMT（证据） | 与 SIMD 一致 | ~0.11× | O(log L) 协作扫描 |

归档: [x_bwd_simd](src/kernels/x_bwd_simd/) · [verification.json](src/kernels/x_bwd_simd/verification.json) · [x_bwd_coopsimt](src/kernels/x_bwd_coopsimt/)

> 关键读法：两版数值完全一致，快慢是架构差异。

## 二、实现细节

这是一个小的 key=value 说明表（不是 core-results 表，应被跳过，无需 archive link）：

| 参数 | 值 |
|------|------|
| tile | 1024 |
