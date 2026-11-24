# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def catlass_rglob(pattern):
    return sorted(
        p
        for p in ROOT.rglob(pattern)
        if p.relative_to(ROOT).parts[0].startswith("catlass-python-")
    )
SKILLS = (
    "catlass-python-workflow",
    "catlass-python-design",
    "catlass-python-develop",
    "catlass-python-bench",
    "catlass-python-optimize",
    "catlass-python-knowledge",
    "catlass-python-knowledge-upgrade",
)
REMOVED_SCRIPTS = (
    "develop_state.py",
    "optimize_state.py",
    "bench.py",
    "verification.py",
    "validate_reference.py",
    "record_knowledge.py",
    "upgrade_state.py",
    "gen_api_matrix.py",
    "diff_api_matrix.py",
    "check_knowledge.py",
    "drift_report.py",
    "resolve_examples.py",
)


def skill_text(name):
    return (ROOT / name / "SKILL.md").read_text(encoding="utf-8")


def test_seven_skills_and_entry_docs_exist():
    for name in SKILLS:
        path = ROOT / name / "SKILL.md"
        assert path.is_file(), name
        assert path.read_text(encoding="utf-8").strip(), name


def test_only_gate_script_remains_under_skills():
    scripts = sorted(
        path.relative_to(ROOT).as_posix() for path in catlass_rglob("scripts")
    )
    assert scripts == ["catlass-python-workflow/scripts"]
    python_files = sorted(
        path.relative_to(ROOT).as_posix() for path in catlass_rglob("*.py")
    )
    # 只有门禁脚本会自动执行；templates 下的骨架脚本是待复制模板，不参与自动执行。
    assert python_files == [
        "catlass-python-develop/templates/skeleton_gemm.py",
        "catlass-python-develop/templates/skeleton_profile.py",
        "catlass-python-workflow/scripts/gate.py",
        "catlass-python-workflow/tests/test_catlass_python_skills.py",
    ]


@pytest.mark.parametrize("name", SKILLS)
def test_skill_keeps_auto_execution_and_timeout_guidance(name):
    text = skill_text(name)
    assert "自动执行" in text
    assert "超时纪律" in text


def test_docs_do_not_reference_removed_script_paths_or_state_file():
    paths = sorted(catlass_rglob("*.md"))
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for script in REMOVED_SCRIPTS:
            assert "scripts/{}".format(script) not in text, (relative, script)
        assert "state.json" not in text, relative


def test_entry_and_workflow_drop_machine_gates():
    for relative in ("catlass-python-workflow/SKILL.md",):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "G0" not in text, relative
        assert "G2 状态机" not in text, relative
        assert "next_action" not in text, relative


def test_workflow_requires_real_execution_evidence():
    text = skill_text("catlass-python-workflow")
    assert "真实执行记录" in text
    assert "blocked" in text
    assert "ITERATIONS.md" in text


def test_bench_skill_requires_self_written_measurement_script():
    text = skill_text("catlass-python-bench")
    assert "本版本不提供 `bench.py`" in text
    assert "performance.candidate.mean_ms" in text
    assert "torch.npu.Event" in text


def test_optimize_skill_uses_text_ledger():
    text = skill_text("catlass-python-optimize")
    assert "Performance Rounds" in text
    assert "2~3 轮" in text
    assert "上限 3 轮" in text
    assert "至少三轮" not in text
    assert "early_stop: skeleton_ratio >= 0.80" in text
    assert "MFU ≥ 90%" not in text
    assert "required_improvement" in text
    assert "Stall 与停止条件" in text
    assert "知识沉淀" in text
    assert "Gotchas" in text


def test_workflow_shares_early_stop_exception():
    text = (ROOT / "catlass-python-workflow" / "SKILL.md").read_text(encoding="utf-8")
    assert "skeleton_ratio" in text


def test_optimize_skill_requires_structural_delta():
    text = skill_text("catlass-python-optimize")
    assert "结构差异对照" in text
    assert "原始 golden" in text
    assert "单变量消融" in text
    assert "not measured" in text
    assert "骨架 GEMM 的比值" in text

    template = (ROOT / "catlass-python-optimize/templates/ITERATIONS.md").read_text(
        encoding="utf-8"
    )
    assert "## Structural Delta" in template
    assert "Golden / 主参考样例" in template
    assert "不得作为优化轴依据" in template


def test_design_and_develop_cover_single_kernel_merge():
    design = skill_text("catlass-python-design")
    assert "structural_verdict" in design
    assert "merge_required" in design
    develop = skill_text("catlass-python-develop")
    assert "merge_required" in develop
    assert "multi_kernel_to_single_kernel" in develop

    text = kb_card("optimizations/multi_kernel_to_single_kernel.md")
    assert "cross_flag" in text
    assert "空核必须到达每个栅栏" in text
    assert "SyncAll<false>()" in text


def test_develop_skill_prescreens_tiles_before_measuring():
    text = skill_text("catlass-python-develop")
    assert "解析预筛" in text
    assert "实跑不超过 3 组" in text
    assert "至少测 3 组" not in text
    assert "--dry-run" in text
    assert "有效 MTE2 带宽" in text

    template = (
        ROOT / "catlass-python-develop/templates/gemm_pipelined.md"
    ).read_text(encoding="utf-8")
    assert "预测比值" in template
    assert "至少扫 3 组" not in template


def test_skeleton_template_uses_explicit_mutex():
    template = (
        ROOT / "catlass-python-develop/templates/gemm_pipelined.md"
    ).read_text(encoding="utf-8")
    assert "tla.mutex" in template
    assert 'auto_sync="v0"' not in template
    assert "matmul_row_exp_scatter_add" in template

    text = skill_text("catlass-python-develop")
    assert "只允许用 `tla.mutex`" in text
    assert "自检第 8 项" in text
    assert "核内同步沿用骨架的 `tla.mutex`" in text
    assert "cross_flag" in text
    assert "块映射是否按整除规则算出" in text
    assert "时把 `sn = nn` 一并纳入候选" in text
    assert "唯一豁免：主参考样例本身不使用 mutex" in text

    optimize = skill_text("catlass-python-optimize")
    assert "不得更换同步原语" in optimize


def test_templates_for_text_workflow_exist():
    expected = (
        "catlass-python-design/templates/DESIGN.md",
        "catlass-python-develop/templates/ITERATIONS.md",
        "catlass-python-develop/templates/skeleton_gemm.py",
        "catlass-python-develop/templates/skeleton_profile.py",
        "catlass-python-develop/templates/skeleton_spec.json",
        "catlass-python-optimize/templates/ITERATIONS.md",
        "catlass-python-bench/templates/workload.jsonl",
        "catlass-python-bench/templates/definition.json",
        "catlass-python-bench/templates/solution.json",
        "catlass-python-knowledge/templates/learned-entry.md",
        "catlass-python-knowledge-upgrade/templates/UPGRADE_REPORT.md",
        "catlass-python-knowledge-upgrade/templates/PR_CHECKLIST.md",
    )
    for relative in expected:
        assert (ROOT / relative).is_file(), relative


def test_skeleton_runner_is_shipped_as_template():
    runner = (
        ROOT / "catlass-python-develop/templates/skeleton_profile.py"
    ).read_text(encoding="utf-8")
    assert "--dry-run" in runner
    assert "--ledger-row" in runner
    assert "skeleton_spec.json" in runner
    assert "skeleton_calibration.json" in runner
    assert "def _pick_round" in runner
    assert "Step Id" in runner
    assert "fallback" in runner
    assert "--dummy" in runner
    # 驱动器必须自带 profile 匹配、四段时间模型、流水强度淘汰与 DoD 判定。
    assert "--kernel-hint" in runner
    assert "hint_hit" in runner
    assert "pred_mte1_ms" in runner
    assert "流水强度不足" in runner
    assert "bandwidth-bound" in runner
    assert "failed (serial pipeline)" in runner
    # A/B 转置视图由 spec 的 layout_a / layout_b 驱动，kernel 不加分支。
    assert "layout_b" in runner
    assert "ColumnMajor" in runner

    kernel = (ROOT / "catlass-python-develop/templates/skeleton_gemm.py").read_text(
        encoding="utf-8"
    )
    assert "tla.mutex" in kernel
    assert "auto_sync=" not in kernel
    assert "cross_flag(" not in kernel
    assert "pipe_barrier(" not in kernel

    text = skill_text("catlass-python-develop")
    assert "脚本骨架随模板发布" in text
    assert "skeleton_profile.py" in text
    assert "skeleton_spec.json" in text


def test_no_gate_only_templates_remain():
    removed = (
        "catlass-python-develop/templates/state.json",
        "catlass-python-develop/templates/test-report.json",
        "catlass-python-develop/templates/focused-test-result.json",
        "catlass-python-develop/references/verification-reuse.md",
        "catlass-python-optimize/templates/proposal.json",
        "catlass-python-optimize/templates/round-result.json",
        "catlass-python-optimize/templates/final-result.json",
    )
    for relative in removed:
        assert not (ROOT / relative).exists(), relative


def test_stage_sequence_is_consistent_between_workflow_and_develop():
    sequence = "skeleton → implement（自然多 kernel）→ focused test → full test #1"
    assert sequence in skill_text("catlass-python-workflow")
    assert sequence in skill_text("catlass-python-develop")


def test_tolerance_standard_has_single_source_of_truth():
    bench = skill_text("catlass-python-bench")
    assert "abs(candidate-reference) <= max_atol + max_rtol * abs(reference)" in bench
    assert "required_matched_ratio" in bench
    assert "tolerance 的字段语义" in skill_text("catlass-python-design")
    assert "tolerance 规则" in skill_text("catlass-python-develop")


def test_full_test_is_mandatory_sharded():
    for name in ("catlass-python-workflow", "catlass-python-develop"):
        assert "必须分 2~5 片" in skill_text(name)


def test_bench_restores_device_retry_rule():
    assert "换卡重试一次" in skill_text("catlass-python-bench")


def _kb_root():
    env = os.environ.get("CANNBOT_KNOWLEDGE_ROOT")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(ROOT.parents[2] / "cannbot-knowledge" / "knowledge")
    candidates.append(Path.home() / ".cannbot" / "cannbot-knowledge" / "knowledge")
    for candidate in candidates:
        base = candidate if candidate.name == "knowledge" else candidate / "knowledge"
        if (base / "ops" / "catlass-python" / "index.md").is_file():
            return base
    return None


def kb_card(relative):
    root = _kb_root()
    if root is None:
        pytest.skip("cannbot-knowledge not found; set CANNBOT_KNOWLEDGE_ROOT")
    if not relative.startswith("ops/"):
        relative = "ops/catlass-dsl/" + relative
    path = root / relative
    assert path.is_file(), path
    return path.read_text(encoding="utf-8")


def test_knowledge_paths_referenced_by_skills_exist():
    pattern = re.compile(r"`(knowledge/ops/catlass-dsl/[A-Za-z0-9_./-]+)`")
    seen = 0
    for name in SKILLS:
        text = skill_text(name)
        for target in pattern.findall(text):
            root = _kb_root()
            if root is None:
                pytest.skip("cannbot-knowledge not found; set CANNBOT_KNOWLEDGE_ROOT")
            path = root / target[len("knowledge/"):]
            assert path.exists() or path.with_suffix(".md").exists(), (name, target)
            seen += 1
    assert seen > 0


GATE = ROOT / "catlass-python-workflow" / "scripts" / "gate.py"
SOLUTION_SOURCE = "def run(x):\n    return x\n"


def run_gate(stage, workspace):
    return subprocess.run(
        [sys.executable, str(GATE), stage, "--workspace", str(workspace)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_design(
    workspace,
    dev_count=30,
    mode="remote-connected",
    perf="requested",
    workload_count=1,
    workload_origin="user",
):
    reference = "import torch\n\n\ndef run(x):\n    return x + 1\n"
    write(workspace / "reference.py", reference)
    definition = {
        "name": "op",
        "axes": {},
        "inputs": {"x": {"shape": [4], "dtype": "float32"}},
        "outputs": {"out": {"shape": [4], "dtype": "float32"}},
        "reference": reference,
    }
    write(workspace / "definition.json", json.dumps(definition))
    workload = [
        {
            "uuid": "case-{:04d}".format(index),
            "inputs": {"x": [1, 2, 3, 4]},
            "origin": workload_origin,
        }
        for index in range(workload_count)
    ]
    write(
        workspace / "workload.jsonl",
        "\n".join(json.dumps(row) for row in workload) + "\n",
    )
    dev = []
    for index in range(dev_count):
        if index < workload_count:
            uuid = "case-{:04d}".format(index)
            row = {
                "uuid": uuid,
                "inputs": {"x": [1, 2, 3, 4]},
                "origin": workload_origin,
            }
            if workload_origin == "user":
                row["source"] = uuid
            dev.append(row)
        else:
            dev.append(
                {
                    "uuid": "extra-{:04d}".format(index - workload_count),
                    "inputs": {"x": [1, 2, 3, 4]},
                    "origin": "agent_supplemented",
                }
            )
    write(
        workspace / "workload_dev.jsonl",
        "\n".join(json.dumps(row) for row in dev) + "\n",
    )
    design = "# Design\n\nDSL 版本：2.2.0\n\n"
    if mode:
        design += "执行模式：{}\n\n".format(mode)
    if perf:
        design += "性能要求：{}\n\n".format(perf)
    design += "merge_required: false\n\n"
    design += "benchmark: python tools/bench_local.py\n\n"
    for name in ("reference.py", "definition.json", "workload.jsonl"):
        design += "{}: {}\n".format(name, digest(workspace / name))
    write(workspace / "DESIGN.md", design)


def build_workspace(workspace, mode="remote-connected", perf="requested"):
    build_design(workspace, mode=mode, perf=perf)
    write(workspace / "solution.py", SOLUTION_SOURCE)
    solution = {
        "name": "op",
        "definition": "op",
        "author": "tester",
        "description": "gate fixture",
        "spec": {
            "languages": ["python"],
            "target_hardware": ["LOCAL"],
            "entry_point": "solution.py::run",
            "destination_passing_style": False,
        },
        "sources": [{"path": "solution.py", "content": SOLUTION_SOURCE}],
    }
    write(workspace / "solution.json", json.dumps(solution))
    for name in (
        "skeleton.log",
        "focused.log",
        "fulltest1.json",
        "fulltest2.json",
        "bench.json",
        "review.md",
        "round-1.json",
        "round-2.json",
        "round-3.json",
    ):
        write(workspace / "evidence" / name, "ok\n")
    kernel = digest(workspace / "solution.py")
    rows = [
        "| 1 | skeleton | {0} | compile | passed | evidence/skeleton.log |".format(
            kernel
        ),
        "| 2 | focused test | {0} | run | passed | evidence/focused.log |".format(
            kernel
        ),
        "| 3 | full test #1 | {0} | run | passed | evidence/fulltest1.json |".format(
            kernel
        ),
    ]
    write(
        workspace / "ITERATIONS.md",
        "# Iteration Log\n\n## Milestones\n\n"
        "| Iter | Stage | Kernel SHA256 | Command | Result | Evidence |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n",
    )
    return kernel


def add_milestone(workspace, row):
    path = workspace / "ITERATIONS.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if line.strip().lower().startswith("## milestones")
    )
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].startswith("## ")
        ),
        len(lines),
    )
    insert_at = end
    while insert_at > start + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines.insert(insert_at, row)
    write(path, "\n".join(lines) + "\n")


def add_rounds(workspace, rounds=3, mode="explore", decision="adopted"):
    text = "## Optimization Policy\n- mode: {}\n\n".format(mode)
    text += (
        "## Performance Rounds\n\n"
        "| Round | Axis | Bottleneck | Before (ms) | After (ms) | Decision | Evidence |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    for index in range(1, rounds + 1):
        text += "| {0} | axis | mte2 | 1.0 | 0.9 | {1} | evidence/round-{0}.json |\n".format(
            index, decision
        )
    path = workspace / "ITERATIONS.md"
    write(path, path.read_text(encoding="utf-8") + "\n" + text)


def test_gate_design_requires_deliverables(tmp_path):
    result = run_gate("design", tmp_path)
    assert result.returncode == 1
    assert "设计交付物缺失" in result.stdout

    build_design(tmp_path)
    result = run_gate("design", tmp_path)
    assert result.returncode == 0, result.stdout


def test_gate_design_rejects_dev_subset_out_of_range(tmp_path):
    build_design(tmp_path, dev_count=29, workload_count=30)
    result = run_gate("design", tmp_path)
    assert result.returncode == 1
    assert "30~40" in result.stdout

    build_design(tmp_path, dev_count=41, workload_count=30)
    result = run_gate("design", tmp_path)
    assert result.returncode == 1
    assert "30~40" in result.stdout


def test_gate_design_allows_small_user_set_without_padding(tmp_path):
    build_design(tmp_path, dev_count=20, workload_count=20)
    result = run_gate("design", tmp_path)
    assert result.returncode == 0, result.stdout

    build_design(tmp_path, dev_count=23, workload_count=20)
    result = run_gate("design", tmp_path)
    assert result.returncode == 0, result.stdout


def test_gate_design_requires_all_user_cases_when_small(tmp_path):
    build_design(tmp_path, dev_count=19, workload_count=20)
    result = run_gate("design", tmp_path)
    assert result.returncode == 1
    assert "全部选入" in result.stdout


def test_gate_design_requires_full_coverage_up_to_40(tmp_path):
    build_design(tmp_path, dev_count=35, workload_count=35)
    result = run_gate("design", tmp_path)
    assert result.returncode == 0, result.stdout

    rows = (tmp_path / "workload_dev.jsonl").read_text(encoding="utf-8").splitlines()
    rows[-1] = json.dumps(
        {
            "uuid": "extra-0000",
            "inputs": {"x": [1, 2, 3, 4]},
            "origin": "agent_supplemented",
        }
    )
    write(tmp_path / "workload_dev.jsonl", "\n".join(rows) + "\n")
    result = run_gate("design", tmp_path)
    assert result.returncode == 1
    assert "全覆盖" in result.stdout


def test_gate_design_requires_30_user_cases_above_40(tmp_path):
    build_design(tmp_path, dev_count=30, workload_count=50)
    result = run_gate("design", tmp_path)
    assert result.returncode == 0, result.stdout

    rows = (tmp_path / "workload_dev.jsonl").read_text(encoding="utf-8").splitlines()
    rows[-1] = json.dumps(
        {
            "uuid": "extra-0000",
            "inputs": {"x": [1, 2, 3, 4]},
            "origin": "agent_supplemented",
        }
    )
    write(tmp_path / "workload_dev.jsonl", "\n".join(rows) + "\n")
    result = run_gate("design", tmp_path)
    assert result.returncode == 1
    assert "至少包含 30 条用户用例" in result.stdout


def test_gate_design_allows_small_agent_generated_dev_subset(tmp_path):
    build_design(
        tmp_path,
        dev_count=6,
        workload_count=12,
        workload_origin="agent_supplemented",
    )
    result = run_gate("design", tmp_path)
    assert result.returncode == 0, result.stdout


def test_gate_develop_requires_focused_test_record(tmp_path):
    build_workspace(tmp_path)
    path = tmp_path / "ITERATIONS.md"
    kept = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if "focused test" not in line
    ]
    write(path, "\n".join(kept) + "\n")

    result = run_gate("develop", tmp_path)
    assert result.returncode == 1
    assert "focused test" in result.stdout


def test_gate_optimize_requires_two_rounds_in_explore_mode(tmp_path):
    build_workspace(tmp_path)
    add_rounds(tmp_path, rounds=1)

    result = run_gate("optimize", tmp_path)
    assert result.returncode == 1
    assert "至少 2 轮" in result.stdout


def test_gate_optimize_allows_zero_rounds_with_skeleton_early_stop(tmp_path):
    build_workspace(tmp_path)
    path = tmp_path / "ITERATIONS.md"
    text = path.read_text(encoding="utf-8")
    text += (
        "\n## Optimization Policy\n"
        "- mode: explore\n"
        "- early_stop: skeleton_ratio >= 0.80\n"
    )
    write(path, text)

    result = run_gate("optimize", tmp_path)
    assert result.returncode == 0, result.stdout


def build_mmad_design(workspace, stages, listed):
    design = (workspace / "DESIGN.md").read_text(encoding="utf-8")
    design += "\n是否含 MMAD：是\n\n- 全部 matmul 阶段清单（逐行）：{}\n".format(listed)
    write(workspace / "DESIGN.md", design)
    (workspace / "tools").mkdir(exist_ok=True)
    write(
        workspace / "tools" / "skeleton_spec.json",
        json.dumps(
            {
                "dtype": "bf16",
                "stages": [
                    {"name": name, "m": 1024, "n": 1024, "k": 1024, "layout_a": "row"}
                    for name in stages
                ],
            }
        ),
    )


def test_gate_develop_rejects_matmul_without_skeleton_spec(tmp_path):
    build_workspace(tmp_path)
    build_mmad_design(
        tmp_path,
        stages=["logits"],
        listed=(
            "logits / k1 / 60% / 1024,1024,1024,bf16 / case-0000；"
            "dh / k2 / 40% / 1024,1024,1024,bf16 / case-0000"
        ),
    )

    result = run_gate("develop", tmp_path)
    assert result.returncode == 1
    assert "没有对应的骨架 spec" in result.stdout


def test_gate_develop_rejects_spec_stage_missing_from_design(tmp_path):
    build_workspace(tmp_path)
    build_mmad_design(
        tmp_path,
        stages=["logits", "dw"],
        listed="logits / k1 / 100% / 1024,1024,1024,bf16 / case-0000",
    )

    result = run_gate("develop", tmp_path)
    assert result.returncode == 1
    assert "未在 DESIGN 登记" in result.stdout


def test_gate_develop_accepts_matching_matmul_lists(tmp_path):
    build_workspace(tmp_path)
    build_mmad_design(
        tmp_path,
        stages=["logits", "dh"],
        # 模板的真实写法：整行包在一个反引号里，多项用 `；` 分隔
        listed=(
            "`logits / k1 / 60% / 1024,1024,1024,bf16 / case-0000；"
            "dh / k2 / 40% / 1024,1024,1024,bf16 / case-0000`"
        ),
    )

    result = run_gate("develop", tmp_path)
    assert result.returncode == 0, result.stdout


def test_gate_develop_rejects_partially_registered_matmul_spec(tmp_path):
    """整行反引号的清单必须逐项解析，否则 spec 少登记阶段会被漏掉。"""
    build_workspace(tmp_path)
    build_mmad_design(
        tmp_path,
        stages=["logits"],
        listed=(
            "`logits / k1 / 60% / 1024,1024,1024,bf16 / case-0000；"
            "dh / k2 / 20% / 1024,1024,1024,bf16 / case-0000；"
            "dw / k3 / 20% / 1024,1024,1024,bf16 / case-0000`"
        ),
    )

    result = run_gate("develop", tmp_path)
    assert result.returncode == 1
    assert "没有对应的骨架 spec" in result.stdout


def test_gate_develop_accepts_overview_when_detail_list_shrinks(tmp_path):
    """概览行与明细清单合并解析：明细缩水但概览齐全时不误判。"""
    build_workspace(tmp_path)
    build_mmad_design(
        tmp_path,
        stages=["logits", "dh", "dw"],
        listed="`logits / k1 / 100% / 1024,1024,1024,bf16 / case-0000`",
    )
    design = (tmp_path / "DESIGN.md").read_text(encoding="utf-8")
    design += (
        "\n- 包含的 matmul 阶段：`logits`、`dh`、`dw`；纯 vector 算子不适用。\n"
        "\n其他阶段 dh 与 dw 的流水与 logits 同构。\n"
    )
    write(tmp_path / "DESIGN.md", design)

    result = run_gate("develop", tmp_path)
    assert result.returncode == 0, result.stdout


def test_gate_optimize_rejects_non_skeleton_early_stop(tmp_path):
    build_workspace(tmp_path)
    path = tmp_path / "ITERATIONS.md"
    text = path.read_text(encoding="utf-8")
    text += "\n## Optimization Policy\n- mode: explore\n- early_stop: mfu >= 0.90\n"
    write(path, text)

    result = run_gate("optimize", tmp_path)
    assert result.returncode == 1
    assert "skeleton_ratio" in result.stdout


def early_stop_workspace(tmp_path, threshold="0.80"):
    build_workspace(tmp_path)
    path = tmp_path / "ITERATIONS.md"
    text = path.read_text(encoding="utf-8")
    text += (
        "\n## Optimization Policy\n"
        "- mode: explore\n"
        "- early_stop: skeleton_ratio >= {}\n".format(threshold)
    )
    write(path, text)


def test_gate_optimize_rejects_lowered_early_stop_threshold(tmp_path):
    """门槛不得自行降低，否则 0.5 也能收口。"""
    early_stop_workspace(tmp_path, threshold="0.50")

    result = run_gate("optimize", tmp_path)
    assert result.returncode == 1
    assert "不能低于 0.80" in result.stdout


def rewrite_kernel(workspace, source="def run(x):\n    return x + 1\n"):
    write(workspace / "solution.py", source)
    solution = json.loads((workspace / "solution.json").read_text(encoding="utf-8"))
    solution["sources"] = [{"path": "solution.py", "content": source}]
    solution.pop("kernel_sha256", None)
    write(workspace / "solution.json", json.dumps(solution))
    return digest(workspace / "solution.py")


def test_gate_optimize_allows_adopted_round_kernel_change(tmp_path):
    build_workspace(tmp_path)
    add_rounds(tmp_path, rounds=3)
    rewrite_kernel(tmp_path)

    result = run_gate("optimize", tmp_path)
    assert result.returncode == 0, result.stdout


def test_gate_fulltest_accepts_second_run_after_kernel_change(tmp_path):
    build_workspace(tmp_path)
    kernel = rewrite_kernel(tmp_path)
    add_milestone(
        tmp_path,
        "| 4 | full test #2 | {} | run | passed | evidence/fulltest2.json |".format(
            kernel
        ),
    )

    result = run_gate("fulltest", tmp_path)
    assert result.returncode == 0, result.stdout

    result = run_gate("develop", tmp_path)
    assert result.returncode == 0, result.stdout


def test_gate_fulltest_accepts_third_run_after_kernel_change(tmp_path):
    build_workspace(tmp_path)
    kernel = rewrite_kernel(tmp_path)
    write(tmp_path / "evidence" / "fulltest3.json", "ok\n")
    add_milestone(
        tmp_path,
        "| 4 | full test #3 | {} | run | passed | evidence/fulltest3.json |".format(
            kernel
        ),
    )

    result = run_gate("fulltest", tmp_path)
    assert result.returncode == 0, result.stdout


def test_gate_merge_requires_fusion_record_when_declared(tmp_path):
    base = build_workspace(tmp_path)
    design = tmp_path / "DESIGN.md"
    write(design, design.read_text(encoding="utf-8") + "\nmerge_required: true\n")

    result = run_gate("merge", tmp_path)
    assert result.returncode == 1
    assert "merge passed" in result.stdout

    fused = rewrite_kernel(tmp_path)
    write(tmp_path / "evidence" / "merge.json", "ok\n")
    write(tmp_path / "evidence" / "focused-fused.log", "ok\n")
    add_milestone(
        tmp_path,
        "| 4 | merge | {} | run baseline={} | passed | evidence/merge.json |".format(
            fused, base
        ),
    )
    add_milestone(
        tmp_path,
        "| 5 | focused test | {} | run | passed | evidence/focused-fused.log |".format(
            fused
        ),
    )
    add_milestone(
        tmp_path,
        "| 6 | full test #2 | {} | run | passed | evidence/fulltest2.json |".format(
            fused
        ),
    )

    result = run_gate("merge", tmp_path)
    assert result.returncode == 0, result.stdout


def test_gate_merge_requires_baseline_fulltest1(tmp_path):
    base = build_workspace(tmp_path)
    design = tmp_path / "DESIGN.md"
    write(design, design.read_text(encoding="utf-8") + "\nmerge_required: true\n")
    fused = rewrite_kernel(tmp_path)
    path = tmp_path / "ITERATIONS.md"
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if "full test #1" not in line
    ]
    write(path, "\n".join(lines) + "\n")
    write(tmp_path / "evidence" / "merge.json", "ok\n")
    write(tmp_path / "evidence" / "focused-fused.log", "ok\n")
    add_milestone(
        tmp_path,
        "| 4 | merge | {} | run baseline={} | passed | evidence/merge.json |".format(
            fused, base
        ),
    )
    add_milestone(
        tmp_path,
        "| 5 | focused test | {} | run | passed | evidence/focused-fused.log |".format(
            fused
        ),
    )
    add_milestone(
        tmp_path,
        "| 6 | full test #2 | {} | run | passed | evidence/fulltest2.json |".format(
            fused
        ),
    )

    result = run_gate("merge", tmp_path)
    assert result.returncode == 1
    assert "full test #1" in result.stdout


def test_gate_merge_rejects_fulltest1_only_on_fused_kernel(tmp_path):
    build_workspace(tmp_path)
    design = tmp_path / "DESIGN.md"
    write(design, design.read_text(encoding="utf-8") + "\nmerge_required: true\n")
    fused = rewrite_kernel(tmp_path)
    path = tmp_path / "ITERATIONS.md"
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if "full test #1" not in line
    ]
    write(path, "\n".join(lines) + "\n")
    add_milestone(
        tmp_path,
        "| 7 | full test #1 | {} | run | passed | evidence/fulltest1.json |".format(
            fused
        ),
    )
    write(tmp_path / "evidence" / "merge.json", "ok\n")
    write(tmp_path / "evidence" / "focused-fused.log", "ok\n")
    add_milestone(
        tmp_path,
        "| 4 | merge | {} | run | passed | evidence/merge.json |".format(fused),
    )
    add_milestone(
        tmp_path,
        "| 5 | focused test | {} | run | passed | evidence/focused-fused.log |".format(
            fused
        ),
    )
    add_milestone(
        tmp_path,
        "| 6 | full test #2 | {} | run | passed | evidence/fulltest2.json |".format(
            fused
        ),
    )

    result = run_gate("merge", tmp_path)
    assert result.returncode == 1
    assert "融合前" in result.stdout


def test_gate_merge_accepts_fulltest3_after_kernel_change(tmp_path):
    base = build_workspace(tmp_path)
    design = tmp_path / "DESIGN.md"
    write(design, design.read_text(encoding="utf-8") + "\nmerge_required: true\n")
    fused = rewrite_kernel(tmp_path)
    write(tmp_path / "evidence" / "merge.json", "ok\n")
    write(tmp_path / "evidence" / "focused-fused.log", "ok\n")
    add_milestone(
        tmp_path,
        "| 4 | merge | {} | run baseline={} | passed | evidence/merge.json |".format(
            fused, base
        ),
    )
    add_milestone(
        tmp_path,
        "| 5 | focused test | {} | run | passed | evidence/focused-fused.log |".format(
            fused
        ),
    )
    add_milestone(
        tmp_path,
        "| 6 | full test #2 | {} | run | passed | evidence/fulltest2.json |".format(
            fused
        ),
    )

    tuned = rewrite_kernel(tmp_path, source="def run(x):\n    return x + 2\n")
    write(tmp_path / "evidence" / "fulltest3.json", "ok\n")
    add_milestone(
        tmp_path,
        "| 7 | full test #3 | {} | run | passed | evidence/fulltest3.json |".format(
            tuned
        ),
    )

    result = run_gate("merge", tmp_path)
    assert result.returncode == 0, result.stdout
    assert "full test #3" in result.stdout


def test_gate_merge_accepts_not_applicable_when_not_declared(tmp_path):
    kernel = build_workspace(tmp_path)

    result = run_gate("merge", tmp_path)
    assert result.returncode == 1
    assert "not_applicable" in result.stdout

    add_milestone(
        tmp_path,
        "| 4 | merge | {} | not required | passed | - |".format(kernel),
    )
    result = run_gate("merge", tmp_path)
    assert result.returncode == 0, result.stdout


def test_gate_benchmark_exempts_fulltest_when_kernel_unchanged(tmp_path):
    kernel = build_workspace(tmp_path)
    add_rounds(tmp_path, rounds=3, decision="rejected")
    add_milestone(
        tmp_path,
        "| 4 | benchmark | {} | run | passed | evidence/bench.json |".format(kernel),
    )

    result = run_gate("benchmark", tmp_path)
    assert result.returncode == 0, result.stdout


def test_gate_benchmark_requires_fulltest_after_kernel_change(tmp_path):
    build_workspace(tmp_path)
    add_rounds(tmp_path, rounds=3)
    kernel = rewrite_kernel(tmp_path)

    result = run_gate("benchmark", tmp_path)
    assert result.returncode == 1
    assert "full test #2" in result.stdout

    add_milestone(
        tmp_path,
        "| 4 | full test #2 | {} | run | passed | evidence/fulltest2.json |".format(
            kernel
        ),
    )
    add_milestone(
        tmp_path,
        "| 5 | benchmark | {} | run | passed | evidence/bench.json |".format(kernel),
    )
    result = run_gate("benchmark", tmp_path)
    assert result.returncode == 0, result.stdout


def test_gate_finish_requires_merge_when_declared(tmp_path):
    kernel = build_workspace(tmp_path, perf="none")
    design = tmp_path / "DESIGN.md"
    write(
        design,
        design.read_text(encoding="utf-8").replace(
            "merge_required: false", "merge_required: true"
        ),
    )
    add_rounds(tmp_path, rounds=3, decision="rejected")
    add_milestone(
        tmp_path,
        "| 4 | full test #2 | {} | run | passed | evidence/fulltest2.json |".format(
            kernel
        ),
    )
    add_milestone(
        tmp_path,
        "| 5 | final review | {} | review | passed | evidence/review.md |".format(
            kernel
        ),
    )
    path = tmp_path / "ITERATIONS.md"
    write(path, path.read_text(encoding="utf-8") + "\n## Final Verdict\n\nconverged\n")

    result = run_gate("finish", tmp_path)
    assert result.returncode == 1
    assert "merge passed" in result.stdout

    write(tmp_path / "evidence" / "merge.json", "ok\n")
    add_milestone(
        tmp_path,
        "| 6 | merge | {} | run | passed | evidence/merge.json |".format(kernel),
    )
    result = run_gate("finish", tmp_path)
    assert result.returncode == 0, result.stdout


def test_gate_finish_reuses_fulltest1_when_kernel_unchanged(tmp_path):
    kernel = build_workspace(tmp_path, perf="none")
    add_rounds(tmp_path, rounds=3, decision="rejected")
    add_milestone(
        tmp_path,
        "| 4 | final review | {} | review | passed | evidence/review.md |".format(
            kernel
        ),
    )
    path = tmp_path / "ITERATIONS.md"
    write(path, path.read_text(encoding="utf-8") + "\n## Final Verdict\n\nconverged\n")

    result = run_gate("finish", tmp_path)
    assert result.returncode == 0, result.stdout


def set_milestones(workspace, rows):
    write(
        workspace / "ITERATIONS.md",
        "# Iteration Log\n\n## Milestones\n\n"
        "| Iter | Stage | Kernel SHA256 | Command | Result | Evidence |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n",
    )


def test_gate_design_requires_execution_mode(tmp_path):
    build_design(tmp_path, mode=None)
    result = run_gate("design", tmp_path)
    assert result.returncode == 1
    assert "执行模式" in result.stdout

    build_design(tmp_path, mode="unknown")
    result = run_gate("design", tmp_path)
    assert result.returncode == 1
    assert "执行模式非法" in result.stdout


def test_gate_remote_detached_stops_after_code_generation(tmp_path):
    kernel = build_workspace(tmp_path, mode="remote-detached")
    set_milestones(
        tmp_path,
        [
            "| 1 | skeleton | {0} | compile | passed | evidence/skeleton.log |".format(
                kernel
            ),
            "| 2 | focused test | {0} | run | deferred | run_focused.sh |".format(
                kernel
            ),
        ],
    )

    result = run_gate("develop", tmp_path)
    assert result.returncode == 0, result.stdout
    assert "代码生成完成" in result.stdout

    for stage in ("fulltest", "optimize", "benchmark", "finish"):
        result = run_gate(stage, tmp_path)
        assert result.returncode == 1, (stage, result.stdout)


def test_gate_remote_detached_requires_skeleton_before_focused(tmp_path):
    kernel = build_workspace(tmp_path, mode="remote-detached")
    set_milestones(
        tmp_path,
        [
            "| 1 | skeleton | {0} | python3 tools/skeleton_gemm.py | blocked | - |".format(
                kernel
            ),
            "| 2 | focused test | {0} | run | passed | evidence/focused.log |".format(
                kernel
            ),
        ],
    )

    result = run_gate("develop", tmp_path)
    assert result.returncode == 1
    assert "skeleton 未 passed" in result.stdout

    set_milestones(
        tmp_path,
        [
            "| 1 | skeleton | {0} | python3 tools/skeleton_gemm.py | passed | evidence/skeleton.log |".format(
                kernel
            ),
            "| 2 | focused test | {0} | run | passed | evidence/focused.log |".format(
                kernel
            ),
        ],
    )

    result = run_gate("develop", tmp_path)
    assert result.returncode == 0, result.stdout


def test_design_registers_reference_sample_runnability():
    design = skill_text("catlass-python-design")
    assert "参考样例可运行性" in design
    assert "not_run" in design
    template = (ROOT / "catlass-python-design/templates/DESIGN.md").read_text(
        encoding="utf-8"
    )
    assert "参考样例可运行性" in template
    develop = skill_text("catlass-python-develop")
    assert "参考样例可运行性" in develop


def test_precision_upgrade_allows_implementation_order_changes():
    for name in ("catlass-python-design", "catlass-python-develop"):
        text = skill_text(name)
        assert "精度不降即可自由实现" in text, name
        assert "不再要求逐项复现标杆" in text, name
    knowledge = kb_card("runbooks/runtime/multi_kernel_precision_localization.md")
    assert "精度不降即可自由实现" in knowledge
    assert "online / 两遍式" in knowledge
    assert "误差界不劣于标杆" in knowledge


def test_gate_remote_detached_rejects_deferred_full_test(tmp_path):
    kernel = build_workspace(tmp_path, mode="remote-detached")
    set_milestones(
        tmp_path,
        [
            "| 1 | skeleton | {0} | compile | passed | evidence/skeleton.log |".format(
                kernel
            ),
            "| 2 | focused test | {0} | run | deferred | run_focused.sh |".format(
                kernel
            ),
            "| 3 | full test #1 | {0} | run | deferred | run_full_test_1.sh |".format(
                kernel
            ),
            "| 4 | optimize | {0} | run | deferred | run_optimize.sh |".format(kernel),
            "| 5 | full test #2 | {0} | run | deferred | run_full_test_2.sh |".format(
                kernel
            ),
            "| 6 | benchmark | {0} | run | deferred | run_bench.sh |".format(kernel),
        ],
    )

    for stage in ("fulltest", "optimize", "benchmark", "finish"):
        result = run_gate(stage, tmp_path)
        assert result.returncode == 1, (stage, result.stdout)
    assert "必须先连接远端" in run_gate("fulltest", tmp_path).stdout


def test_gate_design_requires_perf_demand(tmp_path):
    build_design(tmp_path, perf=None)
    result = run_gate("design", tmp_path)
    assert result.returncode == 1
    assert "性能要求" in result.stdout

    build_design(tmp_path, perf="unknown")
    result = run_gate("design", tmp_path)
    assert result.returncode == 1
    assert "性能要求非法" in result.stdout


def test_gate_accepts_bold_declarations_from_template(tmp_path):
    build_design(tmp_path, mode="remote-detached", perf="none")
    design = (tmp_path / "DESIGN.md").read_text(encoding="utf-8")
    design = design.replace(
        "执行模式：remote-detached", "- **执行模式：** `remote-detached`"
    )
    design = design.replace("性能要求：none", "- **性能要求：** `none`")
    write(tmp_path / "DESIGN.md", design)

    result = run_gate("design", tmp_path)
    assert result.returncode == 0, result.stdout
    assert "remote-detached" in result.stdout
    assert "性能要求 none" in result.stdout


def test_gate_design_requires_merge_required_declaration(tmp_path):
    build_design(tmp_path)
    path = tmp_path / "DESIGN.md"
    write(
        path,
        path.read_text(encoding="utf-8").replace("merge_required: false\n\n", ""),
    )

    result = run_gate("design", tmp_path)
    assert result.returncode == 1
    assert "merge_required" in result.stdout


def test_gate_benchmark_skipped_when_perf_not_requested(tmp_path):
    kernel = build_workspace(tmp_path, perf="none")
    add_rounds(tmp_path, rounds=3)
    add_milestone(
        tmp_path,
        "| 4 | full test #2 | {} | run | passed | evidence/fulltest2.json |".format(
            kernel
        ),
    )

    result = run_gate("benchmark", tmp_path)
    assert result.returncode == 0, result.stdout
    assert "跳过 benchmark" in result.stdout

    result = run_gate("finish", tmp_path)
    assert result.returncode == 1
    assert "final review" in result.stdout

    add_milestone(
        tmp_path,
        "| 5 | final review | {} | review | passed | evidence/review.md |".format(
            kernel
        ),
    )
    path = tmp_path / "ITERATIONS.md"
    write(path, path.read_text(encoding="utf-8") + "\n## Final Verdict\n\nconverged\n")
    result = run_gate("finish", tmp_path)
    assert result.returncode == 0, result.stdout


def test_gate_benchmark_required_when_perf_requested(tmp_path):
    kernel = build_workspace(tmp_path, perf="requested")
    add_rounds(tmp_path, rounds=3)
    add_milestone(
        tmp_path,
        "| 4 | full test #2 | {} | run | passed | evidence/fulltest2.json |".format(
            kernel
        ),
    )

    result = run_gate("benchmark", tmp_path)
    assert result.returncode == 1
    assert "benchmark passed" in result.stdout


def test_gate_deferred_requires_rerun_command(tmp_path):
    kernel = build_workspace(tmp_path, mode="remote-detached")
    set_milestones(
        tmp_path,
        [
            "| 1 | skeleton | {0} | compile | passed | evidence/skeleton.log |".format(
                kernel
            ),
            "| 2 | focused test | {0} | - | deferred | - |".format(kernel),
        ],
    )

    result = run_gate("develop", tmp_path)
    assert result.returncode == 1
    assert "精确重跑命令" in result.stdout


@pytest.mark.parametrize("mode", ("remote-connected", "local"))
def test_gate_non_deferred_modes_reject_deferred(tmp_path, mode):
    kernel = build_workspace(tmp_path, mode=mode)
    set_milestones(
        tmp_path,
        [
            "| 1 | skeleton | {0} | compile | passed | evidence/skeleton.log |".format(
                kernel
            ),
            "| 2 | focused test | {0} | run_focused.sh | deferred | - |".format(kernel),
        ],
    )

    result = run_gate("develop", tmp_path)
    assert result.returncode == 1
    assert "remote-detached" in result.stdout


def test_gate_finish_requires_final_review_and_verdict(tmp_path):
    kernel = build_workspace(tmp_path)
    add_rounds(tmp_path, rounds=3)
    add_milestone(
        tmp_path,
        "| 4 | full test #2 | {} | run | passed | evidence/fulltest2.json |".format(
            kernel
        ),
    )
    add_milestone(
        tmp_path,
        "| 5 | benchmark | {} | run | passed | evidence/bench.json |".format(kernel),
    )

    result = run_gate("finish", tmp_path)
    assert result.returncode == 1
    assert "final review" in result.stdout

    add_milestone(
        tmp_path,
        "| 6 | final review | {} | review | passed | evidence/review.md |".format(
            kernel
        ),
    )
    path = tmp_path / "ITERATIONS.md"
    write(path, path.read_text(encoding="utf-8") + "\n## Final Verdict\n\nconverged\n")
    result = run_gate("finish", tmp_path)
    assert result.returncode == 0, result.stdout
