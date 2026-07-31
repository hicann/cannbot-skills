# cannbot ops-precision-standard — VERBATIM COPY

> **NOTE (2026-06-30):** this is the cannbot **DESCRIPTION-library**, NOT the grader. The actual
> 生态 op-gen grader is `../cannbench_grader/compare.py` (verbatim cann-bench@007855b), which
> `precision_cannbot_adapter.py` calls BY DEFAULT. These scripts now back only the OPTIONAL 商用
> ratio / quantization / integer / non_compute routes. Do NOT edit the verbatim scripts here.


Source: gitcode.com/cann/cannbot-skills `ops/ops-precision-standard`
Commit: 7d13b6a (2026-06-12); verified byte-identical to chenshushu2020/cannbot-skills `br_asc_dev` for the entire ops-precision-standard tree (scripts + reference) as of 2026-06-18.
Copied verbatim per owner directive 2026-06-18 ("完全照抄 cannbot"). DO NOT edit these files — they are the upstream standard. a5_ops integration/adapter lives one level up (precision_cannbot_adapter.py).

Two criteria (select by scenario, cannbot 原样):
- 商用 双标杆Ratio: scripts/mare_mere_rmse_ratio.py（CPU-fp64 golden + 独立标杆；ratio L0/L1/L2）
- 生态 单标杆Threshold: scripts/mare_mere_threshold.py (absolute vs CPU-fp64 golden)
- special_cases: scripts/small_value_check.py + inf_nan_check.py
