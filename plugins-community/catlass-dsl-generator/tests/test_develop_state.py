import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/catlass-dsl-develop/scripts/develop_state.py"
DEVELOP_RUN_ID = "test-20260728-000000"


def load_module():
    spec = importlib.util.spec_from_file_location("develop_state", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def workspace_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    kernel = project / "src/operator.py"
    kernel.parent.mkdir()
    kernel.write_text("TILE = 8\n", encoding="utf-8")
    return project.resolve(), kernel


def contract_text(project, design_sha256, *, risk="standard", performance=False):
    benchmark_row = '\n| benchmark | benchmark | ["python3","bench.py"] |'
    performance_case = (
        "\n| performance-main | performance | shape=8x8,dtype=float16 | "
        "approved target | benchmark |"
        if performance
        else ""
    )
    performance_values = (
        (
            "yes",
            "performance.candidate.mean_ms",
            "lower",
            "10",
            "benchmark",
            "10",
            "3",
            "0.03",
            "yes",
        )
        if performance
        else (
            "no",
            "performance.candidate.mean_ms",
            "not_applicable",
            "not_applicable",
            "benchmark",
            "not_applicable",
            "not_applicable",
            "not_applicable",
            "yes",
        )
    )
    focus = "memory and synchronization" if risk == "high" else "not_applicable"
    reasons = "DSL memory semantics" if risk == "high" else "none"
    return """# CATLASS DSL Workflow Contract

## Approval

| field | value |
| --- | --- |
| revision | 1 |
| approval_status | approved |
| approved_by | tester |
| approved_at | 2026-07-28T00:00:00Z |
| contract_digest | <computed> |

## Repository Identity

| field | value |
| --- | --- |
| repository_root | {project} |

## Operator Specification

| field | value |
| --- | --- |
| operator_name | test_operator |
| operator_family | elementwise |
| purpose | test operator behavior |

## Tensor Interface

| name | direction | shape | dtype | layout | semantics |
| --- | --- | --- | --- | --- | --- |
| x | input | [8,8] | float16 | row_major | input values |
| output | output | [8,8] | float16 | row_major | computed values |

## Semantic Requirements

| field | value |
| --- | --- |
| computation | output equals x |
| boundary_behavior | fixed test shape |
| numerical_behavior | float16 allclose |

## Preliminary Design

| field | value |
| --- | --- |
| algorithm | direct elementwise implementation |
| dataflow | load compute store |
| tiling_and_layout | row major tile |
| memory_and_sync | local buffer and ordered completion |

## Scope And Allowed Paths

| path | purpose |
| --- | --- |
| src/operator.py | implementation |

## Approved Commands

| command_id | phase | argv |
| --- | --- | --- |
| full-test | full_test | ["python3","-m","pytest"] |{benchmark_row}

## Required Cases

| case_id | category | inputs | oracle | command_id |
| --- | --- | --- | --- | --- |
| correctness-main | correctness | shape=8x8,dtype=float16 | allclose | full-test |{performance_case}

## Performance Target

| field | value |
| --- | --- |
| required | {required} |
| metric_path | {metric_path} |
| direction | {direction} |
| threshold | {threshold} |
| benchmark_command_id | {benchmark_command} |
| max_iterations | {max_iterations} |
| stall_threshold | {stall_threshold} |
| min_improvement_fraction | {min_improvement_fraction} |
| profiling_required | {profiling_required} |

## Risk Classification

| field | value |
| --- | --- |
| risk_level | {risk} |
| risk_reasons | {reasons} |
| targeted_review_focus | {focus} |

## Evidence And Delivery

| field | value |
| --- | --- |
| design_document | DESIGN.md |
| design_sha256 | {design_sha256} |
| evidence_directory | .catlass-dsl/develop-runs/{develop_run_id} |
| knowledge_mode | batch_at_finish |
| delivery | working_tree |
""".format(
        project=project, benchmark_row=benchmark_row,
        design_sha256=design_sha256,
        performance_case=performance_case,
        required=performance_values[0], metric_path=performance_values[1],
        direction=performance_values[2], threshold=performance_values[3],
        benchmark_command=performance_values[4],
        max_iterations=performance_values[5],
        stall_threshold=performance_values[6],
        min_improvement_fraction=performance_values[7],
        profiling_required=performance_values[8],
        risk=risk, develop_run_id=DEVELOP_RUN_ID,
        reasons=reasons, focus=focus,
    )


def approved_contract(tmp_path, *, risk="standard", performance=False):
    module = load_module()
    project, kernel = workspace_project(tmp_path)
    design = tmp_path / "DESIGN.md"
    design.write_text("# Test operator design\n", encoding="utf-8")
    design_sha256 = module.hashlib.sha256(design.read_bytes()).hexdigest()
    path = tmp_path / "CONTRACT.md"
    draft = contract_text(
        project, design_sha256, risk=risk, performance=performance
    )
    digest = module.contract_digest(draft.encode())
    path.write_text(draft.replace("<computed>", digest), encoding="utf-8")
    return module, path, project, module.hashlib.sha256(kernel.read_bytes()).hexdigest()


def result(stage, kernel_sha256, **values):
    return {
        "stage": stage,
        "status": "passed",
        "kernel_sha256": kernel_sha256,
        "evidence": ["evidence/{}.json".format(stage)],
        **values,
    }


def breakdown(head):
    return result(
        "task_breakdown",
        head,
        tasks=[
            {
                "task_id": "implement-operator",
                "depends_on": [],
                "allowed_paths": ["src/operator.py"],
                "required_cases": ["correctness-main"],
                "done_when": "required correctness case passes",
            }
        ],
    )


def benchmark_result(head, *, target_met=None):
    profile_artifact = "artifacts/benchmark/profiling"
    values = {
        "correctness": "passed",
        "environment": {"device": "npu:0"},
        "anti_hack": {
            "status": "passed",
            "policy": "single-fused-catlass-kernel-v1",
            "declared_kernel_names": ["fused_kernel"],
            "observed_kernel_names": ["fused_kernel"],
            "profiled_iterations": 4,
            "observed_launches": 4,
            "launches_per_iteration": 1.0,
            "reason": None,
        },
        "performance": {
            "status": "passed",
            "candidate": {"mean_ms": 1.25},
        },
        "workloads": [
            {
                "uuid": "case-a",
                "status": "passed",
                "anti_hack": {
                    "status": "passed",
                    "policy": "single-fused-catlass-kernel-v1",
                    "declared_kernel_names": ["fused_kernel"],
                    "observed_kernel_names": ["fused_kernel"],
                    "profiled_iterations": 2,
                    "observed_launches": 2,
                    "launches_per_iteration": 1.0,
                    "reason": None,
                },
                "performance": {
                    "status": "passed",
                    "candidate": {"mean_ms": 1.0},
                    "reference": {"mean_ms": 2.5},
                    "speedup": 2.5,
                },
            },
            {
                "uuid": "case-b",
                "status": "passed",
                "anti_hack": {
                    "status": "passed",
                    "policy": "single-fused-catlass-kernel-v1",
                    "declared_kernel_names": ["fused_kernel"],
                    "observed_kernel_names": ["fused_kernel"],
                    "profiled_iterations": 2,
                    "observed_launches": 2,
                    "launches_per_iteration": 1.0,
                    "reason": None,
                },
                "performance": {
                    "status": "passed",
                    "candidate": {"mean_ms": 1.5},
                    "reference": {"mean_ms": 3.0},
                    "speedup": 2.0,
                },
            },
        ],
        "profiling": {
            "status": "passed",
            "candidate": {"artifact": profile_artifact},
        },
    }
    if target_met is not None:
        values["performance_target_met"] = target_met
    benchmark = result("benchmark", head, **values)
    benchmark["evidence"].append(profile_artifact)
    return benchmark


def test_contract_parses_and_verifies_workspace_without_git(tmp_path):
    module, path, project, _kernel_sha256 = approved_contract(tmp_path)
    contract = module.parse_contract(path)
    digest = module.validate_contract(contract, require_approved=True)
    assert digest == contract["approval"]["contract_digest"]
    assert module.verify_workspace(contract) == project
    assert contract["commands"][0]["parsed_argv"] == ["python3", "-m", "pytest"]
    assert contract["specification"]["operator_name"] == "test_operator"
    assert {row["direction"] for row in contract["tensor_interface"]} == {
        "input", "output"
    }


def test_contract_rejects_design_document_drift(tmp_path):
    module, path, _project, _head = approved_contract(tmp_path)
    (tmp_path / "DESIGN.md").write_text(
        "# Changed after approval\n", encoding="utf-8"
    )
    with pytest.raises(module.ContractError, match="design_sha256"):
        module.validate_contract(
            module.parse_contract(path), require_approved=True
        )


def test_design_contract_rejects_empty_specification_and_invalid_interface(tmp_path):
    module, path, _project, _head = approved_contract(tmp_path)
    contract = module.parse_contract(path)
    contract["specification"]["purpose"] = ""
    contract["tensor_interface"][0]["direction"] = "inout"
    with pytest.raises(
        module.ContractError,
        match="Operator Specification.*purpose|direction",
    ):
        module.validate_contract(contract, require_approved=True)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda text: text.replace("## Required Cases\n", ""), "章节"),
        (lambda text: text.replace("src/operator.py", "../operator.py"), "安全仓库相对路径"),
        (lambda text: text.replace("| full-test |\n\n## Performance", "| missing |\n\n## Performance"), "未知 command_id"),
    ],
)
def test_contract_rejects_missing_section_illegal_path_and_unknown_command(
    tmp_path, mutation, expected
):
    module, path, _project, _head = approved_contract(tmp_path)
    path.write_text(mutation(path.read_text(encoding="utf-8")), encoding="utf-8")
    with pytest.raises(module.ContractError, match=expected):
        contract = module.parse_contract(path)
        module.validate_contract(contract, require_approved=True)


@pytest.mark.parametrize(
    ("run_id", "expected"),
    [
        ("missing-time", "YYYYMMDD-HHMMSS"),
        ("invalid-20261340-256199", "UTC"),
    ],
)
def test_contract_requires_valid_second_precision_run_id(
    tmp_path, run_id, expected
):
    module, path, _project, _head = approved_contract(tmp_path)
    contract = module.parse_contract(path)
    contract["delivery"]["evidence_directory"] = (
        ".catlass-dsl/develop-runs/{}".format(run_id)
    )
    with pytest.raises(module.ContractError, match=expected):
        module.validate_contract(contract, require_approved=True)


def test_contract_change_invalidates_digest_and_state(tmp_path):
    module, path, _project, _head = approved_contract(tmp_path)
    contract = module.parse_contract(path)
    state = module.start_state(contract)
    path.write_text(
        path.read_text(encoding="utf-8").replace("implementation", "implementation update"),
        encoding="utf-8",
    )
    changed = module.parse_contract(path)
    with pytest.raises(module.ContractError, match="config_digest"):
        module.validate_contract(changed, require_approved=True)
    with pytest.raises(module.ContractError):
        module.advance_state(changed, state, {})


def test_standard_flow_has_one_review_and_requires_benchmark(tmp_path):
    module, path, _project, head = approved_contract(tmp_path)
    contract = module.parse_contract(path)
    state = module.start_state(contract)
    assert state["next_action"] == "task_breakdown"
    state = module.advance_state(contract, state, breakdown(head))
    for stage in ("implement", "final_review", "full_test"):
        state = module.advance_state(contract, state, result(stage, head))
    state = module.advance_state(contract, state, benchmark_result(head))
    state = module.advance_state(contract, state, result("finish", head))
    assert state["status"] == "passed"
    assert state["next_action"] == "complete"
    assert [event["stage"] for event in state["events"]] == [
        "task_breakdown", "implement", "final_review", "full_test", "benchmark",
        "finish",
    ]
    assert state["benchmark"]["benchmark"]["profiling"]["status"] == "passed"
    assert set(state["review"]) == {"final_review"}


def test_high_risk_adds_targeted_review_and_performance_adds_benchmark(tmp_path):
    module, path, _project, head = approved_contract(
        tmp_path, risk="high", performance=True
    )
    contract = module.parse_contract(path)
    state = module.start_state(contract)
    state = module.advance_state(contract, state, breakdown(head))
    stages = (
        ("implement", {}),
        ("targeted_review", {}),
        ("final_review", {}),
        ("full_test", {}),
        ("benchmark", benchmark_result(head, target_met=True)),
        ("finish", {}),
    )
    for stage, extra in stages:
        stage_result = extra if stage == "benchmark" else result(stage, head, **extra)
        state = module.advance_state(contract, state, stage_result)
    assert state["next_action"] == "complete"
    assert set(state["review"]) == {"targeted_review", "final_review"}
    assert "benchmark" in state["benchmark"]


def test_failed_performance_routes_to_optimize_then_rebenchmark(tmp_path):
    module, path, _project, head = approved_contract(tmp_path, performance=True)
    contract = module.parse_contract(path)
    state = module.start_state(contract)
    state = module.advance_state(contract, state, breakdown(head))
    for stage in ("implement", "final_review", "full_test"):
        state = module.advance_state(contract, state, result(stage, head))
    state = module.advance_state(
        contract, state,
        benchmark_result(head, target_met=False),
    )
    assert state["next_action"] == "optimize"
    state = module.advance_state(contract, state, result("optimize", head))
    assert state["next_action"] == "benchmark"
    state = module.advance_state(
        contract, state,
        benchmark_result(head, target_met=True),
    )
    assert state["next_action"] == "final_review"
    state = module.advance_state(contract, state, result("final_review", head))
    assert state["next_action"] == "finish"


def test_public_capabilities_can_start_at_their_own_stage(tmp_path):
    module, path, _project, _head = approved_contract(tmp_path, performance=True)
    contract = module.parse_contract(path)
    assert module.start_state_at(contract, "task_breakdown")["next_action"] == "task_breakdown"
    assert module.start_state_at(contract, "implement")["next_action"] == "implement"
    assert module.start_state_at(contract, "debug")["next_action"] == "debug"
    assert module.start_state_at(contract, "final_review")["next_action"] == "final_review"
    assert module.start_state_at(contract, "benchmark")["next_action"] == "benchmark"
    assert module.start_state_at(contract, "optimize")["next_action"] == "optimize"

    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()
    module, path, _project, _head = approved_contract(baseline_root)
    contract = module.parse_contract(path)
    assert module.start_state_at(contract, "benchmark")["next_action"] == "benchmark"


def test_develop_records_local_knowledge_queries(tmp_path):
    module, path, _project, head = approved_contract(tmp_path)
    contract = module.parse_contract(path)
    state = module.start_state_at(contract, "debug")
    query = {
        "trigger": "compile",
        "failure_signature": "compile|required-main|LayoutError|f16-8x8",
        "filters": {
            "tags": ["build"],
            "operator_family": "elementwise",
            "text": "LayoutError",
        },
        "matches": [".catlass-dsl/knowledge/debug/environment-and-build.md"],
        "evidence": ".catlass-dsl/develop-runs/{}/stages/001-debug/evidence/query-compile.json".format(DEVELOP_RUN_ID),
        "kernel_sha256": head,
        "retrieval": {
            "normalized_text": "layouterror",
            "normalized_operator_family": "elementwise",
            "match_mode": "all_terms",
        },
        "match_details": [{
            "path": ".catlass-dsl/knowledge/debug/environment-and-build.md",
            "score": 18,
            "matched_fields": ["title", "body"],
            "matched_terms": ["layouterror"],
        }],
    }
    state = module.advance_state(
        contract, state, result("debug", head, knowledge_queries=[query])
    )
    assert state["knowledge"]["queries"] == [query]
    assert state["events"][0]["knowledge_queries"] == [query]


def test_develop_rejects_stale_or_unsafe_knowledge_query(tmp_path):
    module, path, _project, head = approved_contract(tmp_path)
    contract = module.parse_contract(path)
    state = module.start_state_at(contract, "debug")
    query = {
        "trigger": "runtime",
        "failure_signature": "runtime|required-main|device-error",
        "filters": {"text": "device error"},
        "matches": [".catlass-dsl/knowledge/../outside.md"],
        "evidence": ".catlass-dsl/develop-runs/{}/stages/001-debug/evidence/query-runtime.json".format(DEVELOP_RUN_ID),
        "kernel_sha256": head,
    }
    with pytest.raises(module.ContractError, match="matches"):
        module.advance_state(
            contract, state, result("debug", head, knowledge_queries=[query])
        )
    query["matches"] = []
    query["kernel_sha256"] = "0" * 64
    with pytest.raises(module.ContractError, match="kernel_sha256"):
        module.advance_state(
            contract, state, result("debug", head, knowledge_queries=[query])
        )


def test_develop_accepts_legacy_empty_knowledge_state_and_rejects_duplicate_query(
    tmp_path,
):
    module, path, _project, head = approved_contract(tmp_path)
    contract = module.parse_contract(path)
    state = module.start_state_at(contract, "debug")
    state["knowledge"] = {}
    query = {
        "trigger": "correctness",
        "failure_signature": "test|required-main|mismatch|f16-8x8",
        "filters": {"tags": ["correctness"], "text": "mismatch"},
        "matches": [],
        "evidence": ".catlass-dsl/develop-runs/{}/stages/001-debug/evidence/query-correctness.json".format(DEVELOP_RUN_ID),
        "kernel_sha256": head,
    }
    state = module.advance_state(
        contract, state, result("debug", head, knowledge_queries=[query])
    )
    assert state["knowledge"]["queries"] == [query]
    state["next_action"] = "debug"
    with pytest.raises(module.ContractError, match="不得重复查询"):
        module.advance_state(
            contract, state, result("debug", head, knowledge_queries=[query])
        )


def test_only_optimize_entry_requires_an_approved_target(tmp_path):
    module, path, _project, _head = approved_contract(tmp_path)
    contract = module.parse_contract(path)
    with pytest.raises(module.ContractError, match="性能目标"):
        module.start_state_at(contract, "optimize")
    assert module.start_state_at(contract, "benchmark")["next_action"] == "benchmark"


def test_benchmark_requires_performance_and_candidate_profile(tmp_path):
    module, path, _project, head = approved_contract(tmp_path)
    contract = module.parse_contract(path)
    state = module.start_state_at(contract, "benchmark")
    missing_profile = benchmark_result(head)
    missing_profile.pop("profiling")
    with pytest.raises(module.ContractError, match="profiling.status=passed"):
        module.advance_state(contract, state, missing_profile)

    missing_performance = benchmark_result(head)
    missing_performance.pop("performance")
    with pytest.raises(module.ContractError, match="performance.status=passed"):
        module.advance_state(contract, state, missing_performance)

    missing_artifact = benchmark_result(head)
    missing_artifact["profiling"]["candidate"]["artifact"] = ""
    with pytest.raises(module.ContractError, match="candidate artifact"):
        module.advance_state(contract, state, missing_artifact)

    missing_workloads = benchmark_result(head)
    missing_workloads.pop("workloads")
    with pytest.raises(module.ContractError, match="workload"):
        module.advance_state(contract, state, missing_workloads)

    missing_anti_hack = benchmark_result(head)
    missing_anti_hack.pop("anti_hack")
    with pytest.raises(module.ContractError, match="anti_hack.status=passed"):
        module.advance_state(contract, state, missing_anti_hack)


def test_develop_cpu_benchmark_remains_compatible(tmp_path):
    module, path, _project, head = approved_contract(tmp_path)
    contract = module.parse_contract(path)
    state = module.start_state_at(contract, "benchmark")
    benchmark = benchmark_result(head)
    benchmark["environment"] = {"device": "cpu"}
    benchmark.pop("anti_hack")
    for workload in benchmark["workloads"]:
        workload.pop("anti_hack")
    advanced = module.advance_state(contract, state, benchmark)
    assert advanced["events"][-1]["status"] == "passed"


def test_final_iterations_report_contains_each_workload_speedup(tmp_path):
    module, path, _project, head = approved_contract(tmp_path)
    contract = module.parse_contract(path)
    benchmark = benchmark_result(head)
    benchmark.update({"sequence": 1, "next_action": "finish"})
    finish = result("finish", head, knowledge={})
    finish.update({"sequence": 2, "next_action": "complete"})
    state = module.start_state(contract)
    state.update({
        "events": [benchmark, finish],
        "next_action": "complete",
        "status": "passed",
    })
    report = module._iterations_markdown(contract, state)
    assert "## Workload Speedups" in report
    assert "| case-a | 1.000000 | 2.500000 | 2.5000x |" in report
    assert "| case-b | 1.500000 | 3.000000 | 2.0000x |" in report


@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("max_iterations", "0", "max_iterations"),
        ("min_improvement_fraction", "1", "min_improvement_fraction"),
        ("direction", "sideways", "direction"),
    ],
)
def test_performance_policy_rejects_invalid_values(tmp_path, field, value, expected):
    module, path, _project, _head = approved_contract(tmp_path, performance=True)
    contract = module.parse_contract(path)
    contract["performance"][field] = value
    with pytest.raises(module.ContractError, match=expected):
        module.validate_contract(contract, require_approved=True)


def test_public_controller_output_shape_is_stable(tmp_path):
    module, path, _project, _head = approved_contract(tmp_path)
    state = module.start_state(module.parse_contract(path))
    assert all(field in state for field in module.OUTPUT_FIELDS)


@pytest.mark.parametrize("schema", ["catlass.dsl.workflow.v1", "catlass.dsl.workflow.v2"])
def test_develop_rejects_old_schema(tmp_path, schema):
    module, path, _project, head = approved_contract(tmp_path)
    contract = module.parse_contract(path)
    state = module.start_state(contract)
    state["schema"] = schema
    with pytest.raises(module.ContractError, match="state schema 非法"):
        module.advance_state(contract, state, breakdown(head))


def test_cli_creates_compact_trace_and_single_iterations_log(tmp_path, capsys):
    module, contract_path, project, head = approved_contract(tmp_path)
    run_root = project / ".catlass-dsl/develop-runs" / DEVELOP_RUN_ID
    state_path = run_root / "state.json"
    state_path.parent.mkdir(parents=True)
    kernel = project / "src/operator.py"
    kernel.write_text("TILE = 1\n")
    head = module.hashlib.sha256(kernel.read_bytes()).hexdigest()
    state_path.write_text(json.dumps(module.start_state(module.parse_contract(contract_path))))
    assert module.main(["start", "--state", str(state_path)]) == 0
    capsys.readouterr()
    assert {path.name for path in run_root.iterdir()} == {
        "ITERATIONS.md",
        "state.json",
        "traces",
    }

    stage_dir = run_root / "traces/iter-001-task_breakdown"
    evidence = stage_dir / "evidence/task-breakdown.md"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("# Task Breakdown\n", encoding="utf-8")
    submission = breakdown(head)
    submission["evidence"] = [
        str(evidence.relative_to(project))
    ]
    submission_path = stage_dir / "submission.json"
    submission_path.write_text(json.dumps(submission), encoding="utf-8")

    assert module.main(
        [
            "advance",
            "--state",
            str(state_path),
            "--result",
            str(submission_path),
        ]
    ) == 0
    capsys.readouterr()
    assert (stage_dir / "result.json").is_file()
    assert {path.name for path in stage_dir.iterdir()} == {"kernel.py", "result.json"}
    assert (stage_dir / "kernel.py").read_text() == kernel.read_text()
    trace_result = json.loads((stage_dir / "result.json").read_text())
    assert trace_result["kernel_sha256"] == module.hashlib.sha256(
        (stage_dir / "kernel.py").read_bytes()
    ).hexdigest()
    assert not submission_path.exists()
    run_index = (run_root / "ITERATIONS.md").read_text(encoding="utf-8")
    assert "| Iter | Title | Score | Passed | Notes |" in run_index
    assert "task_breakdown" in run_index
    stored = json.loads(state_path.read_text(encoding="utf-8"))
    assert stored["events"][0]["sequence"] == 1


def test_failed_trace_keeps_only_bounded_failure(tmp_path):
    module, path, project, head = approved_contract(tmp_path)
    contract = module.parse_contract(path)
    state = module.start_state(contract)
    run_root = project / ".catlass-dsl/develop-runs" / DEVELOP_RUN_ID
    trace = run_root / "traces/iter-001-task_breakdown"
    trace.mkdir(parents=True)
    evidence = trace / "raw.log"
    evidence.write_text("raw failure")
    submission = trace / "submission.json"
    payload = result("task_breakdown", head, failure_summary="x" * 20000)
    payload["status"] = "failed"
    payload["evidence"] = [str(evidence.relative_to(project))]
    submission.write_text(json.dumps(payload))
    next_state = module.advance_state(contract, state, payload)
    module._record_stage(contract, {"events": [], "next_action": "task_breakdown"}, submission, payload, next_state)
    assert {item.name for item in trace.iterdir()} == {
        "kernel.py", "result.json", "failure.txt"
    }
    trace_result = json.loads((trace / "result.json").read_text())
    assert trace_result["kernel_sha256"] == module.hashlib.sha256(
        (trace / "kernel.py").read_bytes()
    ).hexdigest()
    assert (trace / "failure.txt").stat().st_size <= 16385


def test_record_rejects_kernel_changed_after_validation(tmp_path):
    module, path, project, head = approved_contract(tmp_path)
    contract = module.parse_contract(path)
    state = module.start_state(contract)
    trace = (
        project / ".catlass-dsl/develop-runs" / DEVELOP_RUN_ID
        / "traces/iter-001-task_breakdown"
    )
    trace.mkdir(parents=True)
    submission = trace / "submission.json"
    payload = breakdown(head)
    evidence = trace / "task-breakdown.json"
    evidence.write_text("{}")
    payload["evidence"] = [str(evidence.relative_to(project))]
    submission.write_text(json.dumps(payload))
    next_state = module.advance_state(contract, state, payload)
    (project / "src/operator.py").write_text("TILE = 99\n")
    with pytest.raises(module.ContractError, match="关闭前 kernel 已变化"):
        module._record_stage(
            contract,
            {"events": [], "next_action": "task_breakdown"},
            submission,
            payload,
            next_state,
        )


def test_finish_snapshots_kernel_and_whitelists_profile(tmp_path):
    module, path, project, head = approved_contract(tmp_path)
    contract = module.parse_contract(path)
    state = module.start_state(contract)
    kernel = project / "src/operator.py"
    kernel.write_text("TILE = 8\n")
    run_root = project / ".catlass-dsl/develop-runs" / DEVELOP_RUN_ID
    trace = run_root / "traces/iter-001-finish"
    trace.mkdir(parents=True)
    evidence = trace / "finish.json"
    evidence.write_text("{}")
    profiler = trace / "raw-profiler/worker"
    profiler.mkdir(parents=True)
    (profiler / "kernel_details.csv").write_text("Name\nk\n")
    (profiler / "step_trace_time.csv").write_text("Computing\n1\n")
    (profiler / "trace.json").write_text("{}")
    iteration = profiler.parent / "anti_hack/iteration-0000/kernel_details.csv"
    iteration.parent.mkdir(parents=True)
    iteration.write_text("Name,Type,OP State\nk,k,N/A\n")
    (profiler.parent / "anti_hack_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "policy": "single-fused-catlass-kernel-v1",
        "profiled_iterations": 1,
        "iterations": [{
            "iteration": 0,
            "kernel_details": "anti_hack/iteration-0000/kernel_details.csv",
            "sha256": module.hashlib.sha256(iteration.read_bytes()).hexdigest(),
            "observed_kernel_names": ["k"],
            "observed_launches": 1,
        }],
    }))
    payload = result("finish", head, knowledge={})
    payload["evidence"] = [str(evidence.relative_to(project))]
    payload["profiling"] = {"status": "passed", "candidate": {"artifact": str(profiler.parent)}}
    state["next_action"] = "finish"
    state["tests"] = {"status": "passed"}
    state["review"] = {"final_review": {"status": "passed"}}
    state["benchmark"] = {
        "benchmark": {
            "status": "passed",
            "environment": {"device": "npu:0"},
            "anti_hack": benchmark_result(head)["anti_hack"],
        }
    }
    submission = trace / "submission.json"
    submission.write_text(json.dumps(payload))
    next_state = module.advance_state(contract, state, payload)
    module._record_stage(contract, {"events": [], "next_action": "finish"}, submission, payload, next_state)
    final = run_root / "final"
    assert (final / "kernel.py").read_text() == kernel.read_text()
    final_result = json.loads((final / "result.json").read_text())
    assert final_result["kernel_sha256"] == module.hashlib.sha256(kernel.read_bytes()).hexdigest()
    assert next_state["events"][-1]["kernel_sha256"] == final_result["kernel_sha256"]
    assert {item.name for item in final.iterdir()} == {"kernel.py", "result.json", "profile"}
    assert {item.name for item in (final / "profile/case-0000").iterdir()} == {
        "kernel_details.csv",
        "step_trace_time.csv",
    }
    assert not (final / "profile/case-0000/anti_hack_manifest.json").exists()
    assert not (final / "profile/case-0000/anti_hack").exists()
    assert not list(run_root.glob("**/trace.json"))


def test_task_breakdown_rejects_unapproved_paths_and_forward_dependencies(tmp_path):
    module, path, _project, head = approved_contract(tmp_path)
    contract = module.parse_contract(path)
    state = module.start_state(contract)
    invalid = breakdown(head)
    invalid["tasks"][0]["allowed_paths"] = ["src/unapproved.py"]
    with pytest.raises(module.ContractError, match="批准路径"):
        module.advance_state(contract, state, invalid)

    state = module.start_state(contract)
    invalid = breakdown(head)
    invalid["tasks"][0]["depends_on"] = ["future-task"]
    with pytest.raises(module.ContractError, match="前序 task"):
        module.advance_state(contract, state, invalid)
