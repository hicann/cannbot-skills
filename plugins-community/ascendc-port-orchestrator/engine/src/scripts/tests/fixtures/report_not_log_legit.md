# X op arch22→arch35 报告（fixture: LEGIT — must pass report_not_log_lint）

本项目归档 X 的前向与反向实现，给出精度与性能结论。前向 SIMD-KO2 为生产（原「40×-NO-GO」已翻案，详见 §三）。

## 一、核心数据

### SIMD-KO2（生产 kernel）—— 代表用例

| 用例 | dtype | 绝对误差 vs fp64 | device-time (µs) | 比例 vs arch22 |
|------|------|------|------|------|
| C0 | fp32 | 2.1e-5 | 37.0 | **0.66×** |
| C1 | fp32 | 3.6e-5 | 205.0 | **0.17×** |
| C2 | fp16 | 量化内 | 33.7 | **0.76×** |

> 生产前向 kernel：精度与对照实现完全一致、性能快 1.67–2.18×。比例 vs arch22 同口径。「曾被判 40×-NO-GO」翻案 + 优化手法见 §三。

## 二、SIMT 实现
（实现细节叙述允许在这里。）

## 三、SIMD 实现 + 翻案经过（历史/调查，非核心数据）
之前「SIMD 40×-NO-GO」是 regbase 原型未优化的误判；复查发现三套实现，natural-SIMD 优化后反超。优化手法：批量 Sum + Broadcast build。历史叙述属于这里。
