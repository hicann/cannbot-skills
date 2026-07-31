# X op arch22→arch35 报告（fixture: VIOLATING — must be caught by report_archive_link_lint）

本项目归档 X 的前向实现。

## 一、核心数据

### 前向（生产 kernel）—— 代表用例

| 用例 | dtype | device-time (µs) | 比例 vs arch22 |
|------|------|------|------|
| C0 | fp32 | 37.0 | **0.66×** |
| C1 | fp16 | 33.7 | **0.76×** |

> 关键读法：这个核心结果表没有任何到 archive 的链接（早期报告有、这份漏了），应被 lint 抓到。

## 二、实现细节
归档路径写在了实现区的散文里，没有挂在表格下面：见 src/kernels/x_fwd_simd 目录。
