# X op arch22→arch35 报告（fixture: VIOLATING — must be caught by report_jargon_lint）

本项目归档 X 的前向实现。

## 一、核心数据

### SIMD（生产 kernel）—— 代表用例

| 用例 | dtype | 绝对误差 vs fp64 | device-time (µs) | 比例 vs arch22 |
|------|------|------|------|------|
| C0 | fp32 | 2.1e-5 | 37.0 | **0.66×** |
| C1 | fp16 | 量化内 | 33.7 | **0.76×** |

归档: [x_fwd_simd](src/kernels/x_fwd_simd/)

这一版的精度结论是 cannbot 套壳的产物，按商用②标准判为 PARTIAL_PERSIST；competitor_mare 见 OL-230、DEBT-20，pass_a 通过 pass_b 失败，详见 commit a11d97ac 与任务 #459、P0cc。这段把内部 harness 术语写进了核心数据区，客户会觉得我们在骗人。

## 二、实现细节
（实现叙述。）
