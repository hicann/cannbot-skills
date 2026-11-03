# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "catlass-cpp-design/scripts"))

from validate_design import (
    ARCHITECTURE_STAGE_RULES,
    REQUIRED_SECTIONS,
    main,
    validate,
    workflow_alignment_errors,
)


TEMPLATE = ROOT / "catlass-cpp-design/templates/design.md.template"
STAGE_BODY = """```text
score[32,32] = q[32,16] @ k[32,16]^T
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
|---|---|---|---|---|
| `score` | GM（用户 workspace） | `score_slot+work*4 KiB` | 使能 | shape=`[32,32]` FP32；物理2份、实际1份；S0 FIX写入，S0 AIV最后消费；`score_free` 后复用 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
|---|---|---|---|---|
| `q/k` | L1 | `[0,1) / [1,2) KiB` | 使能 | 各shape=`[32,16]` FP16；各1份；API输入搬入，MMAD末次读取后复用 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
|---|---|---|---|---|
| `qTile` | L0A | `L0A[0,1) KiB` | 不涉及 | shape=`[32,16]` FP16；1份；MTE1写入，MMAD末次读取后复用 |
| `kTile` | L0B | `L0B[0,1) KiB` | 不涉及 | shape=`[16,32]` FP16；1份；MTE1写入，MMAD末次读取后复用 |
| `scoreAcc` | L0C | `L0C[0,4) KiB` | 不涉及 | shape=`[32,32]` FP32；1份；MMAD写入，FIX末次读取后复用 |
"""

ATLAS_A2_A3_RULE_ROWS = """| R01 | 已应用 | 第4章 S0 以 Cube/Vector 混合计算类型实现 |
| R02 | 已应用 | 第3章和第4章将完整依赖链划为 S0 |
| R03 | 已应用 | 第2章按 logical core 连续推进多个 work |
| R04 | 已应用 | 第4章 score 由 AIC 生产并由 AIV 消费 |
| R05 | 不适用 | 当前方案没有同一执行单元跨 Stage 复用的中间结果 |
| R06 | 已应用 | 第4章和第5章为 score 配置双槽轮转 |
| R07 | 已应用 | 第4章列出 L1、L0A、L0B、L0C 的容量区间 |
| R08 | 已应用 | 第4章对 GM score relay 明确启用 L2 Cache |
| R09 | 已应用 | 第1章固定 FP32 累加和跨 Stage 交接类型 |"""

ASCEND950_RULE_ROWS = """| R01 | 已应用 | 第4章 S0 以 Cube/Vector 混合计算类型实现 |
| R02 | 已应用 | 第3章和第4章将完整依赖链划为 S0 |
| R03 | 不适用 | 当前 Vector 计算没有可合并的 VF 子流程 |
| R04 | 已应用 | 第2章按 logical core 连续推进多个 work |
| R05 | 已应用 | 第4章 score 由 AIC 生产并由 AIV 消费 |
| R06 | 不适用 | 当前方案没有同一执行单元跨 Stage 驻留的中间结果 |
| R07 | 已应用 | 第4章和第5章为 score 配置双槽轮转 |
| R08 | 已应用 | 第4章列出 L1、L0A、L0B、L0C 的容量区间 |
| R09 | 已应用 | 第4章对 GM score relay 明确启用 L2 Cache |
| R10 | 已应用 | 第1章固定 FP32 累加和跨 Stage 交接类型 |"""


def valid_design() -> str:
    return f"""# Score Atlas A2系列产品设计方案

## 1. 目标

主模型 case：B=1,H=8,S=32,D=16。
性能目标：BasicInfo Task Duration < 16 us，预热10次后取50次中位数。

### 1.1 数据类型

| 范围 | 数据 | 数据类型 |
|---|---|---|
| API 输入 | q/k | FP16 |
| API 输出 | score | FP32 |
| Vector 计算 | mask | FP32 |
| Cube 输入与累加 | q/k；scoreAcc | FP16；FP32 |
| 跨 Stage 交接 | score | FP32 |

### 1.2 理论性能上限分析

| 约束来源 | 工作量/数据量 | 峰值或带宽依据 | 理论最低耗时 | 重叠/依赖关系 | 证据 |
|---|---|---|---:|---|---|
| AIC/Cube | q/k MMAD | AIC 峰值 | 4 us | 与 AIV 重叠 | A2 计算峰值 |
| AIV/Vector | mask vector | AIV 峰值 | 3 us | 与 AIC 重叠 | A2 Vector 峰值 |
| GM/MTE | score relay | GM 带宽 | 2 us | 与计算重叠 | A2 GM 带宽 |
| 必要依赖链 | score 生产到消费 | ready/free | 5 us | 不可重叠部分 | CrossCore 依赖 |

理论最低耗时：`5 us`。
目标耗时：`16 us`。
理论可达性：`理论可达`。

## 2. 变量与任务映射

| 符号 | 含义 |
|---|---|
| `S` | 序列长度 |
| `CG` | 每个 logical core 同时推进的 head 数，本方案为1 |

```text
work = logicalCore + iteration*blockDim
head = work mod H
```

每个 core 连续处理分配到的 work，尾任务由 `head < H` 保护。

## 3. Stage 总览

| Stage | 执行单元 | 主要动作 | 并行与流水关系 | 结束条件 |
|---|---|---|---|---|
| S0 | AIC+AIV | 计算并消费 score | 多 work 并行，流水推进 | `score_free` |

### 3.1 芯片规则应用检查

| 规则 | 结论 | 设计证据/不适用原因 |
|---|---|---|
{ATLAS_A2_A3_RULE_ROWS}

## 4. Stage 详细设计

### S0 · Cube/Vector：计算并消费 score

{STAGE_BODY}

## 5. 用户额外 workspace

只计算算子执行期间额外占用的 GM，不包含 API 输入、最终输出、片上空间和固定系统 workspace。

```text
scoreBytes    = blockDim * 2 * 4 KiB
workspace_size = Align512(scoreBytes)
```

| 区域 | 数量与单份大小 | 主模型 case 大小 |
|---|---|---:|
| score relay | `blockDim*2` 个槽，每槽 `4 KiB` | `0.065536 MB` |
| 用户额外 workspace | 上述区域按 `512 B` 对齐 | **`0.065536 MB`** |

容量只由并发 logical core 数和槽深度决定，不随序列长度线性增长；`score_free` 后轮转复用。

## 6. 跨 Stage 同步契约

模式0x2覆盖一个AIC及两个AIV；空任务仍参与同代事件。

| 数据边 | Producer | Consumer | ready | free/复用条件 | 本地 pipe 完成 | 回绕/退出处理 |
|---|---|---|---|---|---|---|
| score relay | AIC/FIX | AIV/MTE2 | `score_ready` | `score_free` 后复用 | FIX 完成、MTE2 完成 | 空任务配对，退出 drain |

执行顺序：

```text
AIC: wait score_free -> MMAD/FIX -> set score_ready
AIV: wait score_ready -> consume score -> set score_free
```
"""


def a5_design() -> str:
    return (
        valid_design()
        .replace("Atlas A2系列产品", "Ascend 950PR&950DT系列产品")
        .replace(ATLAS_A2_A3_RULE_ROWS, ASCEND950_RULE_ROWS)
    )


class DesignValidatorTests(unittest.TestCase):
    def assert_fails_with(self, text: str, message: str) -> None:
        errors = validate(text)
        self.assertTrue(any(message in error for error in errors), errors)

    def test_complete_six_chapter_design_passes(self) -> None:
        self.assertEqual([], validate(valid_design()))

    def test_template_and_validator_require_six_chapters(self) -> None:
        titles = re.findall(
            r"^## \d+\. (.+)$", TEMPLATE.read_text(encoding="utf-8"), re.MULTILINE
        )
        self.assertEqual(list(REQUIRED_SECTIONS), titles)
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn("### 3.1 芯片规则应用检查", template)
        self.assertIn("{stage_rule_decision}", template)

    def test_extra_or_missing_top_level_chapter_fails(self) -> None:
        self.assert_fails_with(
            valid_design() + "\n## 7. 附录\n内容\n", "exactly the six"
        )
        self.assert_fails_with(
            valid_design().replace("## 1. 目标", "目标", 1), "expected one"
        )

    def test_official_product_title_and_dtype_table_are_required(self) -> None:
        self.assert_fails_with(
            valid_design().replace(
                "# Score Atlas A2系列产品设计方案", "# Score 设计方案"
            ),
            "official product series",
        )
        self.assert_fails_with(
            valid_design().replace("### 1.1 数据类型", "### 1.1 类型"), "数据类型"
        )

    def test_theoretical_bound_accepts_initial_estimate_above_target(self) -> None:
        text = valid_design().replace(
            "理论最低耗时：`5 us`。\n目标耗时：`16 us`。",
            "理论最低耗时：`5 us`。\n目标耗时：`16 us`。\n工程初版估算：`40 us`。",
        )
        self.assertEqual([], validate(text))

    def test_theoretical_bound_rejects_target_below_lower_bound(self) -> None:
        self.assert_fails_with(
            valid_design().replace("理论最低耗时：`5 us`。", "理论最低耗时：`20 us`。"),
            "below the theoretical minimum",
        )

    def test_stage_parallel_relation_is_required(self) -> None:
        text = valid_design().replace("多 work 并行，流水推进", "逐 work 串行执行")
        self.assert_fails_with(text, "parallelism or overlap")

    def test_final_design_requires_rule_check_but_not_workflow_metadata(
        self,
    ) -> None:
        text = valid_design()
        for term in (
            "workflow_id",
            "design_rule_version",
            "target_architecture",
            "TilingKey",
        ):
            self.assertNotIn(term, text)
        for rule in ARCHITECTURE_STAGE_RULES["atlas_a2_a3"]:
            self.assertIn(f"| {rule} |", text)
        self.assertNotIn("| R10 |", text)
        self.assertIn("| R10 |", a5_design())
        self.assertEqual([], validate(text))
        self.assertEqual([], validate(a5_design()))

    def test_rule_application_check_rejects_missing_duplicate_or_extra_rule(
        self,
    ) -> None:
        text = valid_design()
        start = text.index("### 3.1 芯片规则应用检查")
        end = text.index("## 4. Stage 详细设计")
        self.assert_fails_with(
            text[:start] + text[end:], "missing chip rule application table"
        )
        r03 = "| R03 | 已应用 | 第2章按 logical core 连续推进多个 work |"
        self.assert_fails_with(
            valid_design().replace(f"{r03}\n", ""),
            "missing chip rule application decision: R03",
        )
        self.assert_fails_with(
            valid_design().replace(r03, f"{r03}\n{r03}"),
            "duplicate chip rule application decision: R03",
        )
        self.assert_fails_with(
            valid_design().replace(
                "| R09 | 已应用 | 第1章固定 FP32 累加和跨 Stage 交接类型 |",
                "| R09 | 已应用 | 第1章固定 FP32 累加和跨 Stage 交接类型 |\n"
                "| R10 | 不适用 | A2/A3 规则文件没有 R10 |",
            ),
            "unexpected chip rule application decision: R10",
        )

    def test_rule_application_check_rejects_invalid_decision_or_empty_evidence(
        self,
    ) -> None:
        self.assert_fails_with(
            valid_design().replace("| R03 | 已应用 |", "| R03 | 部分应用 |"),
            "invalid chip rule application decision: R03",
        )
        self.assert_fails_with(
            valid_design().replace(
                "| R03 | 已应用 | 第2章按 logical core 连续推进多个 work |",
                "| R03 | 已应用 | |",
            ),
            "missing evidence or inapplicable reason: R03",
        )

    def test_ascend950_rule_check_requires_r10(self) -> None:
        self.assert_fails_with(
            a5_design().replace(
                "| R10 | 已应用 | 第1章固定 FP32 累加和跨 Stage 交接类型 |", ""
            ),
            "missing chip rule application decision: R10",
        )

    def test_design_title_and_workflow_architecture_must_match(self) -> None:
        marker = {
            "algorithm_family": "linear_attention",
            "target_architecture": "ascend950",
            "stage": "design",
            "issue_type": None,
        }
        errors = workflow_alignment_errors(valid_design(), marker)
        self.assertTrue(
            any("title product series does not match" in error for error in errors),
            errors,
        )
        self.assertEqual([], workflow_alignment_errors(a5_design(), marker))

    def test_design_requires_design_or_later_workflow_state(self) -> None:
        marker = {
            "algorithm_family": "linear_attention",
            "target_architecture": "atlas_a2_a3",
            "stage": "interface",
            "issue_type": None,
        }
        errors = workflow_alignment_errors(valid_design(), marker)
        self.assertTrue(any("stage=design" in error for error in errors), errors)

    def test_cli_runs_the_full_workflow_validator(self) -> None:
        workflow = {
            "algorithm_family": "linear_attention",
            "target_architecture": "atlas_a2_a3",
            "stage": "design",
            "issue_type": "precision_debug",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design_path = root / "design.md"
            workflow_path = root / "workflow.json"
            design_path.write_text(valid_design(), encoding="utf-8")
            workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "validate_design.py",
                        "--design",
                        str(design_path),
                        "--workflow",
                        str(workflow_path),
                    ],
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(1, main())

    def test_variables_require_symbols_and_mapping(self) -> None:
        no_symbols = (
            valid_design()
            .replace("| `S` | 序列长度 |", "|  |  |")
            .replace(
                "| `CG` | 每个 logical core 同时推进的 head 数，本方案为1 |", "|  |  |"
            )
        )
        self.assert_fails_with(no_symbols, "global symbols")
        self.assert_fails_with(
            valid_design().replace(
                "```text\nwork = logicalCore + iteration*blockDim\nhead = work mod H\n```",
                "",
            ),
            "task mapping",
        )

    def test_each_stage_requires_formula_and_three_space_tables(self) -> None:
        self.assert_fails_with(
            valid_design().replace(
                "```text\nscore[32,32] = q[32,16] @ k[32,16]^T\n```", ""
            ),
            "missing formula block",
        )
        for title in (
            "本 Stage 产生/写出",
            "其它 Stage 提供、当前 Stage 读取或需保护",
            "本 Stage 临时/固定",
        ):
            with self.subTest(title=title):
                self.assert_fails_with(
                    valid_design().replace(f"**{title}**", f"**{title}缺失**"),
                    "three bold space subsections",
                )

    def test_stage_space_table_uses_fixed_columns_and_storage_order(self) -> None:
        self.assert_fails_with(
            valid_design().replace(
                "| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |",
                "| 对象 | 地址 | L2Cache | 功能 |",
                1,
            ),
            "missing Stage space table",
        )
        out_of_order = (
            valid_design()
            .replace(
                "| `qTile` | L0A | `L0A[0,1) KiB`",
                "| `qTile` | L0C | `L0C[0,1) KiB`",
            )
            .replace(
                "| `scoreAcc` | L0C | `L0C[0,4) KiB`",
                "| `scoreAcc` | L0A | `L0A[0,4) KiB`",
            )
        )
        self.assert_fails_with(out_of_order, "storage order")
        self.assert_fails_with(
            valid_design().replace("| `qTile` | L0A |", "| `qTile` | DDR |"),
            "unsupported storage location",
        )
        self.assert_fails_with(
            valid_design().replace("| `qTile` | L0A |", "| `qTile` | L0A/L0B |"),
            "one storage location per row",
        )

    def test_stage_space_table_rejects_unknown_l2_cache_value(self) -> None:
        self.assert_fails_with(
            valid_design().replace("| 使能 |", "| bypass |", 1),
            "L2Cache must be one of",
        )

    def test_stage_space_table_allocation_address_exposes_size(self) -> None:
        self.assert_fails_with(
            valid_design().replace(
                "`score_slot+work*4 KiB`",
                "`s=workOrdinal mod 2`",
            ),
            "allocation address must expose size",
        )

    def test_stage_space_table_rejects_vague_component_bank(self) -> None:
        self.assert_fails_with(
            valid_design().replace(
                "`L0A[0,1) KiB`",
                "组件 bank；双 bank 共 `2 KiB`",
                1,
            ),
            "vague component-managed buffer wording",
        )

    def test_stage_l0_space_row_requires_buffering_mode(self) -> None:
        self.assert_fails_with(
            valid_design().replace(
                "shape=`[32,16]` FP16；1份；MTE1写入",
                "shape=`[32,16]` FP16；MTE1写入",
            ),
            "must state single buffering, ping/pong, or an explicit slot count",
        )

    def test_stage_l0_address_uses_storage_qualified_ranges(self) -> None:
        for invalid_address in (
            "相对基址 0；`[0,1) KiB`",
            "`[0,1) KiB`",
            "`L0B[0,1) KiB`",
        ):
            with self.subTest(address=invalid_address):
                self.assert_fails_with(
                    valid_design().replace("`L0A[0,1) KiB`", invalid_address, 1),
                    "must qualify every interval with its storage space (L0A)",
                )

    def test_design_supports_each_target_architecture(self) -> None:
        designs = {
            "atlas_a2_a3": valid_design(),
            "ascend950": a5_design(),
        }
        for target_architecture, design in designs.items():
            with self.subTest(target_architecture=target_architecture):
                marker = {
                    "algorithm_family": "linear_attention",
                    "target_architecture": target_architecture,
                    "stage": "design",
                    "issue_type": None,
                }
                self.assertEqual([], validate(design))
                self.assertEqual([], workflow_alignment_errors(design, marker))

    def test_design_checks_title_architecture(self) -> None:
        marker = {
            "algorithm_family": "flash_attention",
            "target_architecture": "ascend950",
            "stage": "design",
            "issue_type": None,
        }
        self.assertTrue(
            any(
                "title product series does not match" in error
                for error in workflow_alignment_errors(valid_design(), marker)
            )
        )

    def test_stage_api_io_requires_explicit_offset_and_size(self) -> None:
        vague_api_address = valid_design().replace(
            "| `score` | GM（用户 workspace） | `score_slot+work*4 KiB` |",
            "| `score` | GM（API 输入） | API tensor+work offset；`4 KiB` |",
        )
        self.assert_fails_with(
            vague_api_address,
            "must replace placeholder addressing with tensor base, offset, and transfer size",
        )
        missing_offset = valid_design().replace(
            "| `score` | GM（用户 workspace） | `score_slot+work*4 KiB` |",
            "| `score` | GM（API 输入） | `score_base`；`4 KiB` |",
        )
        self.assert_fails_with(
            missing_offset,
            "must expose tensor base and offset",
        )

    def test_stages_are_continuous_and_space_subsections_are_exact(self) -> None:
        self.assert_fails_with(
            valid_design().replace("### S0 · Cube/Vector", "### S1 · Cube/Vector"),
            "numbered continuously from S0",
        )
        self.assert_fails_with(
            valid_design().replace(
                "**本 Stage 临时/固定**",
                "**额外汇总**\n\n说明。\n\n**本 Stage 临时/固定**",
            ),
            "exactly the three bold space subsections in order",
        )

    def test_stage_summary_must_cover_every_detailed_stage(self) -> None:
        text = valid_design().replace(
            "## 5. 用户额外 workspace",
            f"### S1 · Vector：写出结果\n\n{STAGE_BODY}\n\n## 5. 用户额外 workspace",
        )
        self.assert_fails_with(text, "Stage summary missing row for S1")

    def test_workspace_zero_and_nonzero_forms(self) -> None:
        self.assertEqual([], validate(valid_design()))
        start = valid_design().index("## 5. 用户额外 workspace")
        end = valid_design().index("## 6. 跨 Stage 同步契约")
        zero = (
            "## 5. 用户额外 workspace\n\n"
            "只计算算子执行期间额外占用的 GM，不包含 API 输入、最终输出、片上空间和固定系统 "
            "workspace；本算子 workspace=0。\n\n"
        )
        zero_design = valid_design()[:start] + zero + valid_design()[end:]
        self.assertEqual([], validate(zero_design))
        self.assertEqual(
            [], validate(valid_design().replace("workspace_size =", "workspace ="))
        )
        self.assert_fails_with(
            valid_design().replace("workspace_size =", "workspaceTotal ="),
            "define workspace or workspace_size",
        )
        self.assert_fails_with(
            valid_design().replace("0.065536 MB", "按运行时决定"),
            "concrete model-case value",
        )
        self.assert_fails_with(
            valid_design().replace(
                "workspace_size = Align512(scoreBytes)",
                "workspace_size = Align512(system_workspace + scoreBytes)",
            ),
            "only contain user extra workspace regions",
        )
        contradictory_zero = valid_design().replace(
            "只计算算子执行期间额外占用的 GM，不包含 API 输入、最终输出、片上空间和固定系统 workspace。",
            "workspace_size = 0\n只计算算子执行期间额外占用的 GM，不包含 API 输入、最终输出、片上空间和固定系统 workspace。",
            1,
        )
        self.assert_fails_with(
            contradictory_zero,
            "cannot be combined with workspace regions",
        )

    def test_workspace_scope_is_explicit(self) -> None:
        self.assert_fails_with(
            valid_design().replace(
                "不包含 API 输入、最终输出", "不包含外部输入、最终输出"
            ),
            "API\\s*输入",
        )

    def test_sync_contract_requires_table_and_execution_order(self) -> None:
        self.assert_fails_with(
            valid_design().replace(
                "| 数据边 | Producer | Consumer | ready | free/复用条件 | 本地 pipe 完成 | 回绕/退出处理 |",
                "| 资源 | 说明 |",
            ),
            "synchronization contract",
        )
        self.assert_fails_with(
            valid_design().replace(
                "```text\n"
                "AIC: wait score_free -> MMAD/FIX -> set score_ready\n"
                "AIV: wait score_ready -> consume score -> set score_free\n"
                "```",
                "",
            ),
            "execution-order pseudocode",
        )

    def test_hidden_formula_is_not_evidence(self) -> None:
        formula = "```text\nscore[32,32] = q[32,16] @ k[32,16]^T\n```"
        self.assert_fails_with(
            valid_design().replace(formula, f"<!--\n{formula}\n-->"), "formula block"
        )

    def test_tilde_fenced_blocks_pass(self) -> None:
        self.assertEqual([], validate(valid_design().replace("```", "~~~")))

    def test_every_template_placeholder_is_rejected(self) -> None:
        placeholders = set(
            re.findall(
                r"\{[A-Za-z_][A-Za-z0-9_]*\}", TEMPLATE.read_text(encoding="utf-8")
            )
        )
        self.assertTrue(placeholders)
        for placeholder in placeholders:
            with self.subTest(placeholder=placeholder):
                self.assert_fails_with(
                    valid_design() + f"\n{placeholder}\n", "unresolved placeholder"
                )

    def test_normal_cpp_and_math_braces_are_allowed(self) -> None:
        text = (
            valid_design()
            + "\n```cpp\nint values[] = {0, 1};\nKernel{stage_params};\n```\n集合 {i | i < n}。"
        )
        self.assertEqual([], validate(text))

    def test_local_pipe_barrier_is_allowed_in_design(self) -> None:
        text = valid_design().replace("AIC: wait", "AIC: PipeBarrier<PIPE_V>() -> wait")
        self.assertEqual([], validate(text))


if __name__ == "__main__":
    unittest.main()
