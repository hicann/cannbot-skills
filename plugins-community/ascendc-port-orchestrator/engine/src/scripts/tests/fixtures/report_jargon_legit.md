# X op arch22→arch35 报告（fixture: LEGIT — must pass report_jargon_lint）

本项目归档 X 的前向实现，给出精度与性能结论。生产 kernel 为 SIMD-KO2（详见 §三）。

## 一、核心数据

### SIMD（生产 kernel，md5 `a11d97ac`）—— 代表用例

| 用例 | dtype | 绝对误差 vs fp64 | device-time (µs) | 比例 vs arch22 |
|------|------|------|------|------|
| C0 | fp32 | 2.1e-5 | 37.0 | **0.66×** |
| C1 | fp16 | 量化内 | 33.7 | **0.76×** |

归档: [x_fwd_simd](src/kernels/x_fwd_simd/) · [verification.json](src/kernels/x_fwd_simd/verification.json)

精度只用原始量表述：绝对误差 vs fp64 真值，相对误差由流水线 live 判据给出，可复核于 `verification.json`。

> 关键读法：性能 vec-bound，向量指令吞吐打满；fp32 近零相消是固有底、非 kernel 缺陷。判据细节见 `verification_ascendc.py`，参考阈值 `compare_cv` 仅作参考。

## 二、SIMD 实现 + 历史（非核心数据）
这里允许出现内部术语：cannbot 自动生成的版本 hang；曾被判 NO-GO；OL-230 优化手法、DEBT-20、P0cc dual-count schema、commit 102ea0f0 等，全部住在实现/历史区，不在核心数据区。
