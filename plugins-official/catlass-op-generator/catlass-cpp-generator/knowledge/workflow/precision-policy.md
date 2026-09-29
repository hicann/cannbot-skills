---
type: "workflow"
title: "精度策略"
description: "比较顺序、指标、有效分区、Stage 和报告字段。"
tags: ["catlass-cpp", "workflow", "reference", "test", "precision", "pr1069"]
status: "stable"
generated: {"by": "process:catlass-cpp-pr1069-extraction", "at": "2026-09-18T00:00:00Z"}
verified: [{"by": "process:content-equivalence-check", "at": "2026-09-18T00:00:00Z"}]
sources: [{"id": "pr1069-precision-policy", "resource": "git:cann/cannbot-skills@4950cbd7c45e0d44cb8ec52598edf53cdc7f16ca:ops/catlass-linear-attention-workflow/references/precision-policy.md", "title": "PR1069 fixed source for Linear Attention 精度策略", "kind": "repository"}]
consumers: ["catlass-cpp-reference", "catlass-cpp-test"]
---
# 接口与概念

本 concept 是 PR1069 固定来源的结构化入口；规范性技术正文完整保留在“PR1069 原始正文”。

## 算子算法

算法定义、公式、计算顺序和边界语义以原始正文及当前任务冻结的 operator/golden contract 为准。

## 分核策略与基本块切分

分核、任务组、基本块和尾块规则以原始正文为准，并需针对当前 shape、SoC 和资源重新推导。

## 数据路径与存储层级

GM、workspace、L1、UB、L0 的地址、生命周期和复用要求以原始正文为设计输入。

## 流水排布、同步关系与数值精度

Stage、slot、CrossCore/HardEvent、累加和转换规则以原始正文为准，实际结论必须通过验证。

# 用法

先通过 compact query 定位本 concept，再完整读取；不得把知识内容当作当前工程已验收的证据。

# 代码模式

原始正文中的伪代码和命令保持原义。具体 API 必须核对目标版本 CATLASS/CANN 源码。

# 约束

不得删减或放宽原始规则、阈值、失败条件、状态恢复或验收要求。

# 失败表现

若设计、代码或测试与原始约束不一致，按五阶段 workflow 的 issue_type/resume_from 矩阵恢复。

# 验证方法

对照 frontmatter `sources` 中的固定 PR SHA 和原文件逐段校验；阶段通过仍需实际构建、精度和性能证据。

# PR1069 原始正文

# CATLASS C++ 统一精度检验规则

线性 Attention 工作流使用这一套精度检验规则。唯一标杆源码用纯 PyTorch 编写，默认在 CPU 上
运行，生成正式精度比较使用的标杆输出（golden）。同一源码也可在 GPU 上运行，辅助核对结果。
待开发算子在 NPU 或其它执行后端生成待比较输出（actual）；各测试入口使用相同的比较逻辑。

## 比较顺序

1. 按用例的输出约定核对全部输出名称、形状（shape）和 actual 的原始数据类型（dtype）；用有效区域掩码标记参与比较的元素，并检查设计要求的关键分区是否齐全。
2. 在有效区域检查 actual 和 golden 均为有限值，即不含 NaN 或正负无穷。
3. 对每个元素计算混合容差：

   ```text
   abs(actual - golden) <= atol + rtol * abs(golden)
   ```

4. 统计满足上述公式的有效元素比例（`matched_ratio`），全局至少为 `0.999`。
5. 对每个注意力头（head）、数据块（chunk）、尾块（tail）和边界（boundary）等关键分区分别计算该比例；
   tail、boundary 和标记为 critical 的重点分区默认要求 `1.0`，其它已声明分区要求 `0.999`。
6. 全局和每个关键分区的最大绝对误差（`max_abs`）均不得超过对应 dtype 的 `max_abs_limit`。

上述条件全部满足才判定通过。不能使用“任一指标通过”作为 PASS 条件。

## 指标职责

`matched_ratio` 和 `max_abs` 直接决定是否通过。报告同时记录平均相对误差 `MERE`、最大相对
误差 `MARE`、按浮点间距估算的最大误差 `max_ulp`，以及超出混合容差的元素数 `error_count`，
用于定位整体偏差或单点异常。`MARE` 在 golden 接近 0 时可能被放大，不能单独决定整个输出是否通过。
ULP 指相邻浮点数之间的间距；当前脚本按 float32 间距估算 `max_ulp`，默认只用于诊断。
结合输出值域校准 `max_ulp_limit` 后，可在策略文件中启用严格 ULP 检查；启用后 ULP 和
`max_abs` 都必须通过。

默认 dtype 参数来自统一混合容差标准，保存在 `{skill_dir}/templates/precision-policy.json`。
`skill_dir` 是本 Skill 的真实目录；`operator_dir` 是当前算子目录。专用流程初始化时，将模板复制为
`{operator_dir}/reference/precision-policy.json`。其中 `max_abs_limit` 是起始值，必须在 02 阶段结合
该算子的输出值域和代表性用例校准，并将最终误差上限写入这份工程策略。

`docs/workflow.json` 的 `precision_policy_path` 相对 `operator_dir` 解析，初始值为
`reference/precision-policy.json`。继续已有工程时，先定位其实际使用的策略，保留已校准内容，
将工程内策略的位置记录为相对算子目录的路径。测试程序读取这份 JSON 作为 `policy` 参数；
通过命令行比较时，Agent 将脚本、用例数据和策略文件的位置展开为绝对路径，显式传入 `--policy`。

## 用例测试入口

Stage 和整算子的精度结论统一由 `scripts/compare_precision.py` 的 `compare_case()` 生成。
测试程序从 `docs/api.md` 的输出约定、`docs/design.md` 的测试计划和当前用例参数生成以下内容：

- `expected_outputs`：当前用例的全部输出名称，以及各输出的 `shape`、`dtype`、`required_regions`。
  这些预期独立于实际输出和已传入的掩码；可选输出按本例参数确定。
- `actual` / `golden`：以输出名称为键的 NumPy 数组或原生 PyTorch Tensor。golden 保留标杆精度，可使用 float64。
- `valid_masks`：每个输出的布尔 NumPy 掩码，与该输出同 shape；全区域有效时提供全 True 掩码。
- `regions`：每个输出的具名布尔 NumPy 掩码，与该输出同 shape。`required_regions` 列出本例应检查的
  head、chunk、tail、boundary 等分区；本例无适用分区时显式填写 `required_regions: []` 和 `regions[输出名]: {}`。

actual 保留设备输出的原始 dtype；BF16 使用原生 `torch.bfloat16` Tensor。
误差计算至少使用 float32；任一侧为 float64 时使用 float64，保留高精度标杆与实际输出的差异。
直调 host 输出裸二进制时，测试程序先按 host 输出约定核对字节数，再无损解码为对应 shape 和 dtype。
裸二进制的类型由 host 输出约定确定；比较入口核对解码后的原始 dtype，再转换为数值计算使用的类型。

例如测试计划规定长度 1025、分块长度 256，最后一个元素属于尾块。测试程序已取得 `actual`、
`golden` 输出字典，并从工程策略加载 `policy` 后，调用：

```python
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(skill_dir) / "scripts"))
from compare_precision import compare_case

report = compare_case(
    actual, golden, case="tail_1025", policy=policy,
    expected_outputs={
        "out": {"shape": [1025], "dtype": "float16", "required_regions": ["tail"]}
    },
    valid_masks={"out": np.ones(1025, dtype=bool)},
    regions={"out": {"tail": np.arange(1025) >= 1024}},
)
print(report)
raise SystemExit(2 if report["status"] == "CONFIG_ERROR" else (0 if report["pass"] else 1))
```

需要通过命令行比较时，当前测试程序在运行中生成用例 JSON：字段与上述 API 一致，
`actual`、`golden`、`valid_masks` 和 `regions` 中的数组换成数据文件路径；策略由 `--policy` 提供。
输出数据使用 `.npy`，或保存原始单个 Tensor 的 `.pt`；`.pt` 使用安全加载方式，掩码保存为布尔 `.npy`。
JSON 内的相对路径相对该 JSON 文件解析。这份 JSON 属于本轮测试数据；直接调用 API 时数据在内存中传递。

```bash
python "{skill_dir}/scripts/compare_precision.py" --case "{case_path}" \
  --policy "{operator_dir}/{precision_policy_path}" --report "{report_path}"
```

公共比较脚本保留在 Skill 目录，算子工程保存自己的精度策略和测试入口。

## 分区和计算阶段（Stage）

每个 Stage 的 `expected_outputs` 声明该 Stage 的全部检查输出和必需分区，整算子声明其全部接口输出。
分区在有效区域内比较；尾块和边界的名称使用 `tail`、`boundary` 前缀，其他重点分区使用 `critical`
前缀，使其使用已有的 `1.0` 通过比例。

缺少预期、有效区域掩码或必需分区，以及有效区域为空、必需分区在有效区域内为空时，
用例返回 `status: CONFIG_ERROR`、`pass: false`。先修复测试信息，再重新运行比较。
actual 的输出名称、shape 或原始 dtype 不符合约定时返回 `FAIL`；数值比较继续使用上述指标和阈值。
全部输出检查通过后，用例才返回 `PASS`。`compare()` 保留为单个输出的数值定位工具，
整用例验收采用 `compare_case()` 的结果。

失败报告记录输出名称、失败分区和 `failure_reason`，并回写 `docs/validation.md`；
修复后从最早受影响的 Stage 重新执行。

## 报告字段

用例报告记录 `case`、`status`、`pass` 和 `failure_reason`，并按输出名称汇总各输出结果。
完成数值比较的输出和分区同时记录 `shape`、`dtype`、`region`、`atol`、`rtol`、
`matched_ratio`、`error_count`、`max_abs`、`max_ulp`、`mere` 和 `mare`。
正式验收使用并归档该报告中的统一精度结论。

[^pr1069-precision-policy]: git:cann/cannbot-skills@4950cbd7c45e0d44cb8ec52598edf53cdc7f16ca:ops/catlass-linear-attention-workflow/precision-policy.md
