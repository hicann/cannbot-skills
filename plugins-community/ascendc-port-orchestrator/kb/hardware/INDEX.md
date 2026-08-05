# Hardware Knowledge Index

> 面向跨代际算子移植（当前 arch22→arch35）与正向→反向生成。技能始终先读取此索引，再按需加载目标卡片。

## 目标硬件

| 架构 | 文档 | 主要用途 |
|---|---|---|
| arch22 | [target/ascend910c.md](target/ascend910c.md) | 源代实现、资源与兼容性事实 |
| arch35 | [target/ascend950pr.md](target/ascend950pr.md) | 目标代实现、RegBase/MicroAPI 与资源事实 |

## 探针结论

| 文档 | 主题 |
|---|---|
| [probe_findings/2026-04-21_Q_instruction_cycles.md](probe_findings/2026-04-21_Q_instruction_cycles.md) | 指令周期 |
| [probe_findings/2026-04-21_Q_l1_scratch_op11_kind2.md](probe_findings/2026-04-21_Q_l1_scratch_op11_kind2.md) | L1 scratch |
| [probe_findings/2026-04-21_Q_mte2_parallel.md](probe_findings/2026-04-21_Q_mte2_parallel.md) | MTE2 并行性 |
| [probe_findings/2026-04-21_Q_scalar_broadcast.md](probe_findings/2026-04-21_Q_scalar_broadcast.md) | 标量广播 |
| [probe_findings/2026-04-21_Q_ub_bank_count.md](probe_findings/2026-04-21_Q_ub_bank_count.md) | UB bank |
| [probe_findings/2026-05-18_adhoc_datacopypad_v300_tail.md](probe_findings/2026-05-18_adhoc_datacopypad_v300_tail.md) | DataCopyPad 尾块行为 |

## 加载规则

1. arch22→arch35：同时加载源代与目标代卡片，用实测/官方信息决定兼容性改造。
2. 正向→反向：加载目标代卡片，并以 CPU 真值和前向规格约束梯度实现。
3. 卡片与现场探针冲突时保留证据，标记适用 SoC/CANN 版本，不将单次现象扩写成通用事实。
