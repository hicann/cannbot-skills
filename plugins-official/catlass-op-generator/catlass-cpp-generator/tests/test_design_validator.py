# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/catlass-cpp-design/scripts"))

from validate_design import REQUIRED_SECTIONS, validate, workflow_alignment_errors


TEMPLATE = ROOT / "skills/catlass-cpp-design/templates/design.md.template"
STAGE_BODY = """公式：score[32,32] = q[32,16] @ k[32,16]^T。
输入 shape=[32,16]，dtype=FP16，layout=ND；输出 dtype=FP32。
生命周期：q/k 在 MMAD 读完后释放，score 保留到消费者读完。
task/head 映射：每个 head 分配给一个 AIC 及其两个 AIV。
执行顺序：搬入 q/k，执行矩阵乘，写出 score，通知消费方。
tail 使用有效行 mask，varlen partial 和空任务按同步协议闭合 token。
同步依赖：Fixpipe 写出后发送 score_ready flag，复用前等待 score_free。
地址布局如下，并通过首次写入、最后消费和释放条件闭合生命周期。

| 存储 | 绝对半开区间 | tensor/slot/bank | shape/dtype/layout | 单份/总大小与对齐 | 首次写入 | 最后消费 | 释放/复用条件 |
|---|---|---|---|---|---|---|---|
| L1 | [0,2048) | q/k slot0 | [32,16]/FP16/ND | 2 KiB/2 KiB/32 B | MTE2 搬入 | MTE1 末次读取 | MMAD 装载后释放 |
| UB | [0,4096) | score bank0 | [32,32]/FP32/ND | 4 KiB/4 KiB/32 B | Fixpipe 写入 | AIV 末次读取 | score_free 后复用 |
"""
SYNC_CODE = """```text
Initialize():
  allocate event flag Mutex resources and initialize score_free for every slot
for work in work_items:
  if work is tail: use valid rows
  if work is empty: participate in paired notification without data access
  Producer(slot):
    wait score_free[slot]
    write score[slot]
    set score_ready[slot] flag
  Consumer(slot):
    wait score_ready[slot] flag
    read score[slot]
    set score_free[slot]
DrainAndRelease():
  wait final score_free state and drain all slots
  release event flag Mutex resources
```
"""
STAGE_ROW = "| Stage 0 | AIC owner | score=q@k.T | 无前驱，无并行Stage | q/k GM | score GM | 每head一个work，共8个 | L1与UB |"


def valid_design() -> str:
    return f"""# Score 设计
`workflow_id`: `catlass-linear-attention-v1`
`design_rule_version`: `V1`
`target_architecture`: `atlas_a2_a3`
开发期实测证据统一写入 docs/validation.md。
## 1. 目标与数学语义
### 1.1 目标、融合范围与目标场景
按 docs/api.md 的接口融合 score 计算，以纯 PyTorch CPU 标杆验收。
- 目标 SoC：Ascend910B。
- CATLASS 架构标识：target_architecture=atlas_a2_a3，CATLASS_ARCH=2201，ArchTag=Arch::AtlasA2。
- Vector 执行模型：MemBase，使用 TPipe/TQue/TBuf 分段流水。
- Cube→Vector 数据路径：L0C -> Fixpipe -> GM -> UB。
- Vector→Cube 数据路径：UB -> MTE3 -> GM workspace -> MTE2 -> L1。
| 模型 case | 关键 shape/属性 | 同口径基线 | 目标值 | 测量配置与统计口径 | 可达性分析 |
|---|---|---|---|---|---|
| case0 | B=1,H=8,S=32,D=16 | 20 us | 16 us | 预热10次后取50次中位数 | 搬运与计算可重叠 |
### 1.2 API 契约引用与内部映射
公开接口和 shape 以 docs/api.md 为唯一契约；q/k 映射到 kernel 参数，score 写正式 GM 输出。
### 1.3 完整数学语义与数值边界
score[32,32] = q[32,16] @ k[32,16]^T，与 CPU 标杆公式一致。
| 中间量/高风险运算 | CPU 标杆公式 | 实现运算序列 | 计算 dtype | 输入/中间值范围 | NaN/Inf 风险 | mask/cast/保护时机 |
|---|---|---|---|---|---|---|
| score | q @ k.T | MTE2→MMAD→Fixpipe | FP32 | 有限 FP16 输入与 FP32 累加 | 输入有限时无非有限值 | tail 在 MMAD 前 mask，写出时 cast |
## 2. Stage 总览与完整详设
### 2.1 全局符号、任务域与模型 case 代入
| 符号 | 定义 | 单位/取值域 | 来源或约束 |
|---|---|---|---|
| S | 序列长度 | token/正整数 | docs/api.md |
| CG | 每个 work 的 head 数 | head/1 | 容量与负载推导 |
| 模型 case | Nbase | CG | Nwork | blockDim | wave | 每核任务范围 | 尾 work/尾 head |
|---|---|---|---|---|---|---|---|
| case0 | 8 | 1 | 8 | 8 | 1 | 每核1个work | 无尾work，head完整 |
work_id 通过 grid-stride 映射到 batch/chunk/head，AIC/AIV owner 地址不重叠。
可达 TilingKey=0，适用于 FP16 完整块和 tail，空任务闭合通知。
### 2.2 依赖图与 Stage 结果
q/k -> Stage 0 -> score，依赖、生命周期、执行单元和 AIC/AIV 映射均在表中定义。
| Stage | 执行单元/owner | 功能与公式 | 前驱/可并行 Stage | 输入来源 | 输出落点 | 任务映射/任务数 | 主要空间 |
|---|---|---|---|---|---|---|---|
{STAGE_ROW}
| 数据 | shape/dtype/layout | 生产 Stage/owner | 消费 Stage/最后消费者 | 存放位置与份数依据 | 释放/复用条件 |
|---|---|---|---|---|---|
| score | [32,32]/FP32/ND | Stage 0/AIC | AIV/最后校验消费者 | UB slot，每活跃head一份 | score_free 后复用 |
### 2.3 Stage 0：计算 score
{STAGE_BODY}
### 2.4 数据搬运汇总
| tensor | GM 读/写次数 | 每次/每 work 字节数 | GM→L1/UB 路径 | L1/UB/L0 驻留与复用 | 批量搬运/stride | L2 Cache 策略 | 最后消费者 |
|---|---|---|---|---|---|---|---|
| q/k | 各读1次 | 每work各1 KiB | GM→L1→L0 | L1驻留到MMAD装载结束 | 连续二维搬运/无stride | 一次性输入关闭L2 | Stage 0 MTE1 |
### 2.5 Stage 间同步方案
| 数据边 | 生产 Stage/owner/pipe | 消费 Stage/owner/pipe | 存放位置/slot | ready 条件 | free 条件 | 空任务参与方式 |
|---|---|---|---|---|---|---|
| score | Stage 0/AIC/PIPE_FIX | Stage 0/AIV/PIPE_V | UB/score slot | Fixpipe写完set ready | AIV读完set free | 只通知不访问数据 |
| 同步资源 | 所属核/范围 | 类型与方向 | 保护的数据和 slot | 数量/初始状态 | set/wait 或 lock/unlock 顺序 | 复用/释放条件 | 硬件上限依据 |
|---|---|---|---|---|---|---|---|
| score_ready/free | AIC-AIV跨核 | CrossCore双向 | score slot | 2组/free初始可用 | producer set ready，consumer wait ready并set free | free后复用，最终排空后释放 | 不超过平台flag上限 |
{SYNC_CODE}
### 2.6 Kernel 组件与调用
按目标平台选择 BlockMmad；Host 通过直调入口启动。
### 2.7 Workspace 总量
本算子不使用外部中间区，明确设置 workspace_size=0；框架保留空间不属于算子中间量。
"""


class DesignValidatorTests(unittest.TestCase):
    def assert_fails_with(self, text: str, message: str) -> None:
        errors = validate(text)
        self.assertTrue(any(message in error for error in errors), errors)

    def test_complete_two_chapter_design_passes(self) -> None:
        self.assertEqual([], validate(valid_design()))

    def test_template_and_validator_require_exactly_two_chapters(self) -> None:
        titles = re.findall(r"^## \d+\. (.+)$", TEMPLATE.read_text(encoding="utf-8"), re.MULTILINE)
        self.assertEqual(list(REQUIRED_SECTIONS), titles)
        self.assertEqual(["目标与数学语义", "Stage 总览与完整详设"], titles)

    def test_extra_or_missing_top_level_chapter_fails(self) -> None:
        self.assert_fails_with(valid_design() + "\n## 3. 风险\n内容\n", "exactly the two")
        self.assert_fails_with(
            valid_design().replace("## 1. 目标与数学语义", "目标与数学语义"),
            "expected one required section",
        )

    def test_heading_in_code_does_not_count(self) -> None:
        text = valid_design().replace(
            "## 1. 目标与数学语义", "```text\n## 1. 目标与数学语义\n```"
        )
        self.assert_fails_with(text, "expected one required section")

    def test_metadata_and_evidence_boundary_are_required(self) -> None:
        self.assert_fails_with(
            valid_design().replace("`workflow_id`: `catlass-linear-attention-v1`", ""), "workflow_id"
        )
        self.assert_fails_with(
            valid_design().replace("`design_rule_version`: `V1`", "`design_rule_version`: `V2`"),
            "design_rule_version",
        )
        self.assert_fails_with(
            valid_design().replace("docs/validation.md", "validation record"), "docs/validation.md"
        )

        self.assert_fails_with(
            valid_design().replace("`target_architecture`: `atlas_a2_a3`", ""),
            "target_architecture",
        )

    def test_architecture_specific_paths_are_validated(self) -> None:
        self.assert_fails_with(
            valid_design().replace(
                "L0C -> Fixpipe -> GM -> UB", "L0C -> AIV UB"
            ),
            "A2/A3 Cube→Vector",
        )
        self.assert_fails_with(
            valid_design().replace("Vector 执行模型：MemBase", "Vector 执行模型：RegBase/VF"),
            "A2/A3 Vector execution model",
        )
        a5 = (
            valid_design()
            .replace("`target_architecture`: `atlas_a2_a3`", "`target_architecture`: `ascend950`")
            .replace(
                "target_architecture=atlas_a2_a3，CATLASS_ARCH=2201，ArchTag=Arch::AtlasA2",
                "target_architecture=ascend950，CATLASS_ARCH=3510，ArchTag=Arch::Ascend950",
            )
            .replace(
                "Vector 执行模型：MemBase，使用 TPipe/TQue/TBuf 分段流水",
                "Vector 执行模型：RegBase/VF，完整公式使用一次 VF",
            )
            .replace("L0C -> Fixpipe -> GM -> UB", "L0C -> Fixpipe -> AIV UB")
            .replace(
                "UB -> MTE3 -> GM workspace -> MTE2 -> L1",
                "AIV UB -> AIC L1",
            )
        )
        self.assertEqual([], validate(a5))
        no_vector = (
            valid_design()
            .replace(
                "Vector 执行模型：MemBase，使用 TPipe/TQue/TBuf 分段流水",
                "Vector 执行模型：不适用：本算子没有 Vector Stage",
            )
            .replace(
                "Cube→Vector 数据路径：L0C -> Fixpipe -> GM -> UB",
                "Cube→Vector 数据路径：不适用：不存在 Cube 到 Vector 数据边",
            )
            .replace(
                "Vector→Cube 数据路径：UB -> MTE3 -> GM workspace -> MTE2 -> L1",
                "Vector→Cube 数据路径：不适用：不存在 Vector 到 Cube 数据边",
            )
        )
        self.assertEqual([], validate(no_vector))

    def test_design_and_workflow_architecture_must_match(self) -> None:
        marker = {
            "workflow_id": "catlass-linear-attention-v1",
            "target_architecture": "ascend950",
        }
        errors = workflow_alignment_errors(valid_design(), marker)
        self.assertTrue(any("does not match" in error for error in errors), errors)

    def test_model_case_and_tiling_require_quantified_evidence(self) -> None:
        row = "| case0 | 8 | 1 | 8 | 8 | 1 | 每核1个work | 无尾work，head完整 |"
        self.assert_fails_with(
            valid_design().replace(row, "|  |  |  |  |  |  |  |  |"), "model case scheduling"
        )
        self.assert_fails_with(valid_design().replace("TilingKey=0", "dispatch key=0"), "TilingKey")

    def test_each_stage_requires_a_structured_address_map(self) -> None:
        text = re.sub(
            r"(?s)\n\| 存储 \| 绝对半开区间 .*?\n\| UB \| \[0,4096\).*?\n",
            "\n地址将在实现阶段决定。\n",
            valid_design(),
            count=1,
        )
        self.assert_fails_with(text, "missing structured address map")

    def test_data_movement_and_sync_tables_require_complete_rows(self) -> None:
        movement = "| q/k | 各读1次 | 每work各1 KiB | GM→L1→L0 | L1驻留到MMAD装载结束 | 连续二维搬运/无stride | 一次性输入关闭L2 | Stage 0 MTE1 |"
        sync_resource = "| score_ready/free | AIC-AIV跨核 | CrossCore双向 | score slot | 2组/free初始可用 | producer set ready，consumer wait ready并set free | free后复用，最终排空后释放 | 不超过平台flag上限 |"
        for row, label, count in (
            (movement, "data movement summary", 8),
            (sync_resource, "synchronization resources", 8),
        ):
            with self.subTest(label=label):
                self.assert_fails_with(valid_design().replace(row, "|" + "  |" * count), label)

    def test_sync_pseudocode_requires_full_lifecycle(self) -> None:
        for phrase, expected in (
            ("if work is tail: use valid rows", "tail|partial|尾"),
            ("if work is empty: participate in paired notification without data access", "empty|空任务|无效任务"),
            ("release event flag Mutex resources", r"release\s+"),
        ):
            with self.subTest(phrase=phrase):
                self.assert_fails_with(valid_design().replace(phrase, "compute current work"), expected)

    def test_workspace_zero_is_valid_and_unquantified_workspace_fails(self) -> None:
        self.assertEqual([], validate(valid_design()))
        self.assert_fails_with(
            valid_design().replace("workspace_size=0", "workspace is decided later"), "workspace total"
        )

    def test_nonzero_workspace_requires_table_and_total_formula(self) -> None:
        zero = "本算子不使用外部中间区，明确设置 workspace_size=0；框架保留空间不属于算子中间量。"
        nonzero = """| Workspace 区域 | offset/索引公式 | record/bank 数 | 单份大小与对齐 | 总量贡献 | 生产者与最后消费者 | 释放条件 |
|---|---|---|---|---|---|---|
| score | block_idx*8192 | blockDim 个 record | 8 KiB/512 B | blockDim*8192 | AIV生产/AIC最后消费 | free 后释放 |
workspace_size = AlignUp(blockDim*8192, 512)。"""
        self.assertEqual([], validate(valid_design().replace(zero, nonzero)))
        self.assert_fails_with(
            valid_design().replace(zero, nonzero.replace("workspace_size =", "total =")),
            "define workspace_size",
        )

    def test_workspace_must_be_final_subsection(self) -> None:
        text = valid_design().replace(
            "### 2.7 Workspace 总量", "### 2.7 Workspace 总量\nworkspace_size=0\n### 2.8 附加说明"
        )
        self.assert_fails_with(text, "final Stage detail subsection")

    def test_multiple_stages_and_summary_rows_pass(self) -> None:
        stage_one = "| Stage 1 | AIV owner | 校验score | Stage 0 | score UB | score GM | 每head一个work，共8个 | UB |"
        text = valid_design().replace(STAGE_ROW, f"{STAGE_ROW}\n{stage_one}").replace(
            "### 2.4 数据搬运汇总", f"### 2.4 Stage 1：写出 score\n{STAGE_BODY}\n### 2.5 数据搬运汇总"
        )
        self.assertEqual([], validate(text))

    def test_stage_summary_must_cover_every_detailed_stage(self) -> None:
        text = valid_design().replace(
            "### 2.4 数据搬运汇总", f"### 2.4 Stage 1：写出 score\n{STAGE_BODY}\n### 2.5 数据搬运汇总"
        )
        self.assert_fails_with(text, "Stage result summary missing row for Stage 1")

    def test_stage_content_cannot_come_from_a_sibling_section(self) -> None:
        self.assert_fails_with(
            valid_design().replace(STAGE_BODY, f"### 2.25 地址附注\n{STAGE_BODY}"), "Stage section"
        )

    def test_hidden_content_is_not_evidence(self) -> None:
        self.assert_fails_with(
            valid_design().replace(STAGE_BODY, f"<!--\n{STAGE_BODY}\n-->"), "Stage section"
        )

    def test_sync_code_cannot_come_from_workspace(self) -> None:
        text = valid_design().replace(SYNC_CODE, "").replace(
            "### 2.7 Workspace 总量", f"### 2.7 Workspace 总量\n{SYNC_CODE}"
        )
        self.assert_fails_with(text, "pseudocode")

    def test_sync_terms_in_prose_cannot_replace_code(self) -> None:
        text = valid_design().replace(SYNC_CODE, "wait ready free set event\n```text\ncompute()\n```\n")
        self.assert_fails_with(text, "pseudocode missing")

    def test_empty_and_unclosed_sync_blocks_fail(self) -> None:
        for replacement in ("```text\n```\n", SYNC_CODE.removesuffix("```\n")):
            with self.subTest(replacement=replacement):
                self.assert_fails_with(valid_design().replace(SYNC_CODE, replacement), "pseudocode")

    def test_tilde_fenced_pseudocode_passes(self) -> None:
        self.assertEqual([], validate(valid_design().replace("```", "~~~")))

    def test_every_template_placeholder_is_rejected(self) -> None:
        placeholders = set(re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", TEMPLATE.read_text(encoding="utf-8")))
        self.assertTrue(placeholders)
        placeholders.add("{stage_12_name}")
        for placeholder in placeholders:
            with self.subTest(placeholder=placeholder):
                self.assert_fails_with(valid_design() + f"\n{placeholder}\n", "unresolved placeholder")

    def test_normal_cpp_and_math_braces_are_allowed(self) -> None:
        text = valid_design() + "\n```cpp\nint values[] = {0, 1};\nKernel{stage_params};\n```\n集合 {i | i < n}。"
        self.assertEqual([], validate(text))

    def test_pipe_barrier_in_design_sync_is_rejected(self) -> None:
        text = valid_design().replace("Producer(slot):", "PipeBarrier<PIPE_V>()\nProducer(slot):")
        self.assert_fails_with(text, "paired HardEvent/CrossCore")

    def test_kernel_pipe_barrier_is_rejected(self) -> None:
        from tempfile import TemporaryDirectory
        from validate_design import validate_kernel

        with TemporaryDirectory() as directory:
            kernel = Path(directory) / "kernel.cpp"
            kernel.write_text("void Kernel() { PipeBarrier<PIPE_V>(); }", encoding="utf-8")
            self.assertEqual(
                ["kernel synchronization must use paired HardEvent/CrossCore notifications"],
                validate_kernel(kernel),
            )


if __name__ == "__main__":
    unittest.main()
