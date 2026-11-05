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
import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SKILLS = (
    "catlass-dsl-workflow",
    "catlass-dsl-design",
    "catlass-dsl-develop",
    "catlass-dsl-bench",
    "catlass-dsl-optimize",
    "catlass-dsl-knowledge",
    "catlass-dsl-knowledge-upgrade",
)
DOCS = ("AGENTS.md", "quickstart.md", "UPGRADE.md")
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
    return (ROOT / "skills" / name / "SKILL.md").read_text(encoding="utf-8")


def test_seven_skills_and_entry_docs_exist():
    for name in SKILLS:
        path = ROOT / "skills" / name / "SKILL.md"
        assert path.is_file(), name
        assert path.read_text(encoding="utf-8").strip(), name
    for name in DOCS:
        path = ROOT / name
        assert path.is_file(), name
        assert path.read_text(encoding="utf-8").strip(), name


def test_only_gate_script_remains_under_skills():
    scripts = sorted(
        path.relative_to(ROOT).as_posix() for path in (ROOT / "skills").rglob("scripts")
    )
    assert scripts == ["skills/catlass-dsl-workflow/scripts"]
    python_files = sorted(
        path.relative_to(ROOT).as_posix() for path in (ROOT / "skills").rglob("*.py")
    )
    assert python_files == ["skills/catlass-dsl-workflow/scripts/gate.py"]


@pytest.mark.parametrize("name", SKILLS)
def test_skill_keeps_auto_execution_and_timeout_guidance(name):
    text = skill_text(name)
    assert "自动执行" in text
    assert "超时纪律" in text


def test_docs_do_not_reference_removed_script_paths_or_state_file():
    paths = sorted((ROOT / "skills").rglob("*.md"))
    paths += [ROOT / name for name in DOCS]
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for script in REMOVED_SCRIPTS:
            assert "scripts/{}".format(script) not in text, (relative, script)
        assert "state.json" not in text, relative


def test_entry_and_workflow_drop_machine_gates():
    for relative in ("AGENTS.md", "skills/catlass-dsl-workflow/SKILL.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "G0" not in text, relative
        assert "G2 状态机" not in text, relative
        assert "next_action" not in text, relative


def test_workflow_requires_real_execution_evidence():
    text = skill_text("catlass-dsl-workflow")
    assert "真实执行记录" in text
    assert "blocked" in text
    assert "ITERATIONS.md" in text


def test_bench_skill_requires_self_written_measurement_script():
    text = skill_text("catlass-dsl-bench")
    assert "本版本不提供 `bench.py`" in text
    assert "performance.candidate.mean_ms" in text
    assert "torch.npu.Event" in text


def test_optimize_skill_uses_text_ledger():
    text = skill_text("catlass-dsl-optimize")
    assert "Performance Rounds" in text
    assert "至少三轮" in text or "至少 3 轮" in text
    assert "required_improvement" in text
    assert "Stall 与停止条件" in text
    assert "知识沉淀" in text
    assert "Gotchas" in text


def test_knowledge_bundle_is_intact():
    root = ROOT / "knowledge"
    assert (root / "index.md").is_file()
    assert (root / "query-vocabulary.yaml").is_file()
    for category in ("dsl", "operator", "debug", "profiler", "optimization", "learned"):
        assert (root / category / "index.md").is_file(), category


def test_templates_for_text_workflow_exist():
    expected = (
        "skills/catlass-dsl-design/templates/DESIGN.md",
        "skills/catlass-dsl-develop/templates/ITERATIONS.md",
        "skills/catlass-dsl-optimize/templates/ITERATIONS.md",
        "skills/catlass-dsl-bench/templates/workload.jsonl",
        "skills/catlass-dsl-bench/templates/definition.json",
        "skills/catlass-dsl-bench/templates/solution.json",
        "skills/catlass-dsl-knowledge/templates/learned-entry.md",
        "skills/catlass-dsl-knowledge-upgrade/templates/UPGRADE_REPORT.md",
        "skills/catlass-dsl-knowledge-upgrade/templates/PR_CHECKLIST.md",
    )
    for relative in expected:
        assert (ROOT / relative).is_file(), relative


def test_no_gate_only_templates_remain():
    removed = (
        "skills/catlass-dsl-develop/templates/state.json",
        "skills/catlass-dsl-develop/templates/test-report.json",
        "skills/catlass-dsl-develop/templates/focused-test-result.json",
        "skills/catlass-dsl-develop/references/verification-reuse.md",
        "skills/catlass-dsl-optimize/templates/proposal.json",
        "skills/catlass-dsl-optimize/templates/round-result.json",
        "skills/catlass-dsl-optimize/templates/final-result.json",
    )
    for relative in removed:
        assert not (ROOT / relative).exists(), relative


def test_quickstart_states_gate_guided_semantics():
    text = (ROOT / "quickstart.md").read_text(encoding="utf-8")
    assert "门禁引导工作流" in text
    assert "scripts/gate.py" in text
    assert re.search(r"不包含.*develop_state", text)


def test_stage_sequence_is_consistent_between_workflow_and_develop():
    sequence = "skeleton → implement → focused test → full test #1 → optimize"
    assert sequence in skill_text("catlass-dsl-workflow")
    assert sequence in skill_text("catlass-dsl-develop")


def test_tolerance_standard_has_single_source_of_truth():
    bench = skill_text("catlass-dsl-bench")
    assert "abs(candidate-reference) <= max_atol + max_rtol * abs(reference)" in bench
    assert "required_matched_ratio" in bench
    assert "tolerance 的字段语义" in skill_text("catlass-dsl-design")
    assert "tolerance 规则" in skill_text("catlass-dsl-develop")


def test_full_test_is_mandatory_sharded():
    for name in ("catlass-dsl-workflow", "catlass-dsl-develop"):
        assert "必须分 2~5 片" in skill_text(name)


def test_bench_restores_device_retry_rule():
    assert "换卡重试一次" in skill_text("catlass-dsl-bench")


def test_knowledge_paths_referenced_by_skills_exist():
    pattern = re.compile(r"`(knowledge/[A-Za-z0-9_./-]+)`")
    for name in SKILLS:
        text = skill_text(name)
        for target in pattern.findall(text):
            base = ROOT / target
            assert base.exists() or base.with_suffix(".md").exists(), (name, target)


GATE = ROOT / "skills" / "catlass-dsl-workflow" / "scripts" / "gate.py"
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
    workspace, dev_count=30, mode="remote-connected", perf="requested", workload_count=1
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
        {"uuid": "case-{:04d}".format(index), "inputs": {"x": [1, 2, 3, 4]}}
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
            dev.append(
                {
                    "uuid": uuid,
                    "inputs": {"x": [1, 2, 3, 4]},
                    "origin": "user",
                    "source": uuid,
                }
            )
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


def test_gate_optimize_requires_three_rounds_in_explore_mode(tmp_path):
    build_workspace(tmp_path)
    add_rounds(tmp_path, rounds=2)

    result = run_gate("optimize", tmp_path)
    assert result.returncode == 1
    assert "至少 3 轮" in result.stdout


def rewrite_kernel(workspace):
    source = "def run(x):\n    return x + 1\n"
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
