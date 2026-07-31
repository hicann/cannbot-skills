# X op arch22→arch35 报告（fixture: VIOLATING — must be caught by report_not_log_lint）

本项目归档 X 的前向与反向。

## 一、核心数据

### SIMT（生产 kernel）
| 用例 | dtype | device-time | 比例 vs arch22 |
|------|------|------|------|
| C0 | fp32 | 73.5 | 0.33× |

### SIMD —— ✅ NO-GO 已翻案（2026-06-19 whitebox 复查；见下表与 §三）

之前的「SIMD 40×-NO-GO」结论只对 regbase 原型成立，不是 SIMD 架构的判决。复查发现盘上有三套不同的 "SIMD" 实现：(A) regbase 原型是那个 40×/MERE45/NaN 的坏原型；(C) natural-SIMD 原本就精度完好、只是未优化。把 (C) 用 backward 同一招优化后（KO-1），比生产 SIMT 还快、精度与 SIMT 完全一致。这一整段把「怎么调出来的」过程叙述写进了核心数据区 = 典型的把报告写成日志（应在 §三）。

| 实现 | C1 device-time | 结论 |
|------|------|------|
| (A) regbase 原型 | ~16397µs | 死路 |
| (C) natural-SIMD KO-1 | 305.9µs | 比 SIMT 快 |

KO-1 同机同 session A/B：vec_ratio 0.044→0.975（向量管线从全空到打满）；这条优化手法的过程也属于日志。

## 二、实现细节
（实现叙述。）
