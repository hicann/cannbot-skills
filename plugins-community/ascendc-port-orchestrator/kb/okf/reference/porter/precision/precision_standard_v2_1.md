---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: curated
title: "AscendC 算子精度标准（CPU 真值口径）"
description: "跨代际移植与反向生成的精度标准（CPU 真值口径）：真值优先 CPU FP64 完整复现公式与边界语义，禁用外部加速器结果当真值；要求覆盖小值域/零值/极值/非对齐/空维/动态 shape，逐输出记录误差指标，不得删失败样本或放宽阈值，正反向级联须验全部梯度。"
tags: [precision]
resource: https://gitcode.com/cann/cannbot-skills/blob/master/plugins-community/ascendc-port-orchestrator/kb/okf/reference/porter/precision/precision_standard_v2_1.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
# AscendC 算子精度标准（CPU 真值口径）

## 1. 适用范围

本标准用于跨代际算子移植（当前 arch22→arch35）与正向→反向算子生成。精度结论以高精度 CPU 实现为真值，不使用外部加速器结果作为真值或安全网。

## 2. 真值构造

1. 优先使用 CPU FP64 实现，完整复现公式、广播、归约轴、属性和边界语义。
2. 无法直接使用 FP64 时，使用经过审计的 CPU PyTorch 规格实现，并记录内部 dtype 与 cast 时机。
3. 量化算子在量化前保留浮点中间真值，同时校验最终整数输出。
4. 正向→反向生成必须由同一份前向规格推导梯度，显式记录 `wrt` 顺序、上游梯度和不可导输入。

## 3. 验证要求

- 覆盖常规、小值域、零值、极值、非对齐、空维和动态 shape。
- 对每个输出记录 shape、dtype、有限值、最大绝对误差、最大相对误差及失败元素数。
- arch22 与 arch35 的 CANN 输出可作为迁移差异诊断证据，但不能覆盖 CPU 真值结论。
- 优化前后复用同一输入与同一阈值；不得删除失败样本、放宽阈值或用单次结果替代完整用例集。
- 正反向级联测试必须验证前向输出和所有梯度输出，不能只验证主梯度。

## 4. 结果分级

- `PASS`：所有必测用例在既定 dtype 阈值内通过。
- `PARTIAL`：仅部分用例通过；必须列出失败 shape、dtype、输出和误差。
- `FAIL`：存在规格、有限值、结构或精度门槛失败。
- `DEGENERATE`：CPU 真值本身非有限或用例不具判别力；不得计入通过率。

任何无法由 CPU 真值支撑的结论都必须保持未验证状态。
