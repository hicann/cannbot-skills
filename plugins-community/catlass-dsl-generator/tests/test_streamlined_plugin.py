# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.2.0"


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_expected_public_skills_exist():
    public = sorted(
        path.parent.name for path in (ROOT / "skills").glob("*/SKILL.md")
    )
    assert public == [
        "catlass-dsl-bench",
        "catlass-dsl-design",
        "catlass-dsl-develop",
        "catlass-dsl-knowledge",
        "catlass-dsl-optimize",
    ]


def test_platform_manifests_share_streamlined_version():
    codex = load(".codex-plugin/plugin.json")
    records = [
        load(".claude-plugin/plugin.json"),
        load(".cursor-plugin/plugin.json"),
        load("package.json"),
        load(".claude-plugin/marketplace.json")["plugins"][0],
    ]
    assert {record["name"] for record in [codex, *records]} == {"catlass-dsl-generator"}
    assert {record["version"] for record in records} == {VERSION}
    assert codex["version"].startswith(VERSION + "+codex.")
    assert codex["skills"] == "./skills/"
    assert load(".cursor-plugin/plugin.json")["skills"] == "./skills/"


def test_codex_marketplace_installs_plugin_from_repository_root():
    marketplace = load(".agents/plugins/marketplace.json")
    assert marketplace["name"] == "catlass-dsl-generator-dev"
    assert marketplace["plugins"] == [
        {
            "name": "catlass-dsl-generator",
            "source": {"source": "url", "url": "./"},
            "policy": {
                "installation": "AVAILABLE",
                "authentication": "ON_INSTALL",
            },
            "category": "Developer Tools",
        }
    ]


def test_develop_uses_state_as_its_only_machine_authority():
    develop = ROOT / "skills/catlass-dsl-develop"
    design = ROOT / "skills/catlass-dsl-design"
    develop_source = (ROOT / "skills/catlass-dsl-develop/scripts/develop_state.py").read_text(encoding="utf-8")
    develop_skill = (ROOT / "skills/catlass-dsl-develop/SKILL.md").read_text(encoding="utf-8")
    for text in (develop_source, develop_skill):
        assert "worktree" not in text.lower()
        assert "candidate_worktree_parent" not in text
    readable = (design / "templates/DESIGN.md").read_text(encoding="utf-8")
    assert readable.startswith("# CATLASS DSL 算子设计\n")
    assert not (design / "templates/CONTRACT.md").exists()
    assert (develop / "templates/state.json").is_file()
    assert not (develop / "templates/development-state.json").exists()
    assert "--contract" not in develop_source
    assert "--contract" not in develop_skill
    for git_identity in ("current_head", "git_common_dir", "verify_repository", "git rev-parse"):
        assert git_identity not in develop_source
    state_template = (develop / "templates/state.json").read_text(encoding="utf-8")
    assert "git_common_dir" not in state_template
    assert "repository_root" in state_template
    for required in (
        "📋 Resolved Plan",
        "Record immediately",
        "ITERATIONS.md",
        "traces/iter-NNN-<stage>",
        "kernel.py",
        "Workload Speedups",
        "kernel_details.csv",
        "step_trace_time.csv",
        "catlass.dsl.workflow.v3",
    ):
        assert required in develop_skill
    for forbidden in ("state/state.json", "stages/NNN", "artifacts/commands"):
        assert forbidden not in develop_skill
    assert (develop / "scripts/develop_state.py").is_file()
    assert (develop / "templates/task-breakdown-result.json").is_file()
    assert not (design / "templates/design.md").exists()
    assert not (develop / "templates/design.md").exists()
    assert not (ROOT / "skills/catlass-dsl").exists()


def test_design_skill_requires_python_dsl_instead_of_cpp():
    design = ROOT / "skills/catlass-dsl-design"
    skill = (design / "SKILL.md").read_text(encoding="utf-8")
    template = (design / "templates/DESIGN.md").read_text(encoding="utf-8")
    for expected in (
        "CATLASS DSL 是以 Python 语法编写的算子 DSL，不是 C++ 算子库",
        "@tla.kernel",
        "tla.Tensor",
        "不得设计 C++ class/template",
        "python/tla_dsl",
    ):
        assert expected in skill
    assert "CATLASS Python DSL（`python/tla_dsl`）" in template
    assert "不设计 C++/CUDA C++ CATLASS" in template
    assert "<safe/repository-relative/operator.py>" in template


def test_design_skill_requires_executable_torch_reference_cases():
    design = ROOT / "skills/catlass-dsl-design"
    skill = (design / "SKILL.md").read_text(encoding="utf-8")
    assert (design / "scripts/validate_reference.py").is_file()
    for expected in (
        "reference.py",
        "def run(*inputs)",
        "definition.json",
        "workload.jsonl",
        "--definition <definition.json>",
        "--workload <workload.jsonl>",
        "--state <develop-run>/state.json",
        "SHA-256",
    ):
        assert expected in skill
    assert "全部非 performance" in skill
    assert "required cases" in skill
    assert "validate_reference.py" in skill
    assert "semantics.computation" in skill
    assert "state.json.config.reference_validation" in skill
    assert "reference-cases.json" in skill and "不生成" in skill


def test_benchmark_implementation_is_owned_by_bench_skill():
    bench = ROOT / "skills/catlass-dsl-bench"
    assert (bench / "scripts/bench.py").is_file()
    assert not (bench / "scripts/catlass_dsl_bench.py").exists()
    assert (bench / "templates/solution.json").is_file()
    assert (bench / "templates/workload.jsonl").is_file()
    assert (bench / "templates/definition.json").is_file()
    assert not (bench / "templates/bench-config.json").exists()
    assert not (bench / "scripts/benchmark").exists()
    assert not (ROOT / "skills/catlass-dsl/scripts/catlass_dsl_bench.py").exists()
    assert not (ROOT / "skills/catlass-dsl/scripts/benchmark").exists()


def test_benchmark_skill_documents_suite_interfaces():
    skill = (ROOT / "skills/catlass-dsl-bench/SKILL.md").read_text(encoding="utf-8")
    assert "Definition 接口" in skill
    assert "Workload 接口" in skill
    assert "Solution 接口" in skill
    assert "--solution <solution.json>" in skill
    assert "--workload <workload.jsonl>" in skill
    assert "--definition <definition.json>" in skill
    assert "tuple" in skill and "list" in skill
    assert "shape" in skill and "dtype" in skill
    for expected in (
        "Anti-hack 接口",
        "single",
        "kernel_details.csv",
        "category=hack",
        "status=not_applicable",
    ):
        assert expected in skill


def test_agent_workflows_require_one_runtime_catlass_kernel():
    design = (ROOT / "skills/catlass-dsl-design/SKILL.md").read_text(encoding="utf-8")
    develop = (ROOT / "skills/catlass-dsl-develop/SKILL.md").read_text(encoding="utf-8")
    optimize = (ROOT / "skills/catlass-dsl-optimize/SKILL.md").read_text(encoding="utf-8")
    assert "只 launch 一个" in design
    assert "anti_hack.status=passed" in develop
    assert "multi-launch" in optimize
    assert "GM 临时张量" in design + develop


def test_agent_workflows_require_self_contained_kernels():
    skill_names = (
        "catlass-dsl-design",
        "catlass-dsl-develop",
        "catlass-dsl-bench",
        "catlass-dsl-optimize",
    )
    skills = {
        name: (ROOT / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
        for name in skill_names
    }
    for text in skills.values():
        assert "自包含" in text
        assert "模块级" in text
        assert "closure" in text
        assert "global" in text
    assert "形式参数" in skills["catlass-dsl-design"]
    assert "自由名字" in skills["catlass-dsl-develop"]
    assert "结构审查失败" in skills["catlass-dsl-bench"]
    assert "不得把结构" in skills["catlass-dsl-optimize"]
    assert "违规源码作为 baseline" in skills["catlass-dsl-optimize"]
    for text in skills.values():
        assert "只能声明一个 `@tla.kernel`" in text or "只声明一个 `@tla.kernel`" in text
        assert "不得为编译期变体声明独立" in text


def test_optimize_owns_controller_and_machine_input_templates():
    optimize = ROOT / "skills/catlass-dsl-optimize"
    assert (optimize / "scripts/optimize_state.py").is_file()
    assert (optimize / "templates/proposal.json").is_file()
    assert (optimize / "templates/round-result.json").is_file()
    assert (optimize / "templates/final-result.json").is_file()


def test_optimize_skill_documents_agent_first_iteration_protocol():
    skill = (ROOT / "skills/catlass-dsl-optimize/SKILL.md").read_text(
        encoding="utf-8"
    )
    for required in (
        "📋 Resolved Plan",
        "profile → hypothesis → modify → correctness → benchmark → record",
        "Record immediately",
        "Query knowledge",
        "每轮都调用 `catlass-dsl-knowledge query`",
        "knowledge_sources",
        "--type optimization",
        "Knowledge admission after finalization",
        "至少 5% latency 改善",
        "kernel_sha256",
        "knowledge-candidates.json",
        "Stall handling",
        "When to stop",
        "恢复真实 best kernel",
        "不得创建 Git branch、commit 或 worktree",
        ".catlass-dsl/optimize-runs/<run-id>/",
        "traces/",
        "profile/case-NNNN/{kernel_details.csv,step_trace_time.csv}",
    ):
        assert required in skill
    for forbidden in (
        "ncu profiling",
        "bench/kernelbench",
        "bench-wrapper.sh",
        "HINTS.md",
        "ncu",
        "git checkout -b",
        "git commit -m",
        "CONTRACT.md",
        "--contract",
    ):
        assert forbidden not in skill


def test_readme_documents_user_workflows_and_public_skills():
    text = (ROOT / "quickstart.md").read_text(encoding="utf-8")
    for skill in (
        "catlass-dsl-design",
        "catlass-dsl-develop",
        "catlass-dsl-bench",
        "catlass-dsl-optimize",
        "catlass-dsl-knowledge",
    ):
        assert skill in text
    assert "快速入门" in text
    assert "开发新算子" in text
    assert "优化已有算子" in text
    assert "Benchmark" in text
    assert "Optimize" in text
    assert "Knowledge" in text
    assert "最终交付始终包含当前实现的性能现状" in text
    assert "初始实现未达标时，才会进入 optimize" in text
