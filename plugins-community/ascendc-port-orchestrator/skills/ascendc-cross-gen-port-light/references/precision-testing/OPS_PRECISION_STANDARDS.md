# 生态算子精度标准（迁移场景接入）

> **判定标准真源**：`cannbot-skills/ops/ops-precision-standard/SKILL.md`（混合容差体系；本文件不复制阈值表，以真源为准）。

## 通过判定（混合容差 + 双门限）

**单标杆比对**：与更高精度的实现的单一精度标杆（CPU 或者昇腾小算子拼接）直接比较。本 skill 的标杆为阶段 1 从 A2 源码逆向合成的 CPU 参考实现。

**逐元素通过条件**：`|actual - golden| ≤ atol + rtol × |golden|`

**整体通过条件**（双门限，同时满足）：
1. `matched_ratio = 通过元素数 / 总元素数 ≥ required_matched_ratio`（浮点 0.99）
2. `max_abs_error ≤ max_abs_error_limit`（绝对误差硬帽，"A or B" 取较宽者）

各 dtype 的 rtol / atol / required_matched_ratio / max_abs_error_limit 阈值表见真源 `ops-precision-standard/references/float_compute.md` §3；整型按 `integer_compute.md` 精确匹配。

**特殊规则**（详见真源）：
- 整型输出：0 误差精确匹配；
- INF/NAN 位置做**结构比对**，不参与 matched_ratio / max_abs_error 计算。

## 使用要点（迁移场景）

- 测试脚本模板（`test_op_precision_aclnn_template.py.template` / `run_precision_report_aclnn_template.py.template`）已内置各 dtype 阈值并按混合容差 + 双门限判定，阈值来源同上；
- MERE（平均相对误差）/ MARE（最大相对误差）仅作为**分析指标**输出（用于误差定位，如 subnormal 引发的相对误差放大），不作为通过判定依据。
