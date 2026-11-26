#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Check the declared artifacts for the single-launch delivery path."""

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path

DELIVERY_PATH = "multi-kernel baseline -> fused epilogue -> single launch"
PERF_TOLERANCE = 0.05  # 性能测量容差：有效下限 = floor × (1 - 容差)


def _decorator_is_triton_jit(node: ast.AST) -> bool:
    """与 validate_triton_impl.py 保持一致：@triton.jit、@jit 及带参数调用形式。"""
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == "triton"
            and node.attr == "jit"
        )
    if isinstance(node, ast.Name):
        return node.id == "jit"
    if isinstance(node, ast.Call):
        return _decorator_is_triton_jit(node.func)
    return False


LAUNCH_CALL_RE = re.compile(r"\b[A-Za-z_]\w*\[\([^\]]*\)\]")


def count_launch_call_sites(source: str) -> int:
    """宿主侧 kernel 启动点个数（`name[(grid,)](args)`）。helper @triton.jit 被内联，不计入。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return len(LAUNCH_CALL_RE.findall(source))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
    )


def read_artifact(op_dir: Path, name: str, *, process_evidence: bool) -> dict:
    path = (op_dir / name).resolve()
    if not path.is_relative_to(op_dir) or path.suffix != ".py":
        raise ValueError(
            f"expected a Python artifact inside the operator directory: {name}"
        )
    relative = path.relative_to(op_dir)
    if process_evidence != (relative.parts[0] == "dev"):
        raise ValueError(f"unexpected artifact location: {name}")

    source = path.read_bytes()
    tree = ast.parse(source, filename=str(relative))
    jit_count = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(_decorator_is_triton_jit(item) for item in node.decorator_list)
    )
    return {
        "path": relative.as_posix(),
        "jit_count": jit_count,
        "sha256": hashlib.sha256(source).hexdigest(),
    }


def validate_contract(op_dir: Path) -> None:
    precheck = json.loads((op_dir / "precheck.json").read_text(encoding="utf-8"))
    if not isinstance(precheck, dict):
        raise TypeError("precheck.json must contain an object")
    contract = precheck.get("delivery_contract", precheck)
    if not isinstance(contract, dict):
        raise TypeError("delivery_contract must contain an object")

    required_values = {
        "single_launch_required": True,
        "max_kernel_launches": 1,
        "delivery_path": DELIVERY_PATH,
    }
    for key, expected in required_values.items():
        value = contract.get(key)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"invalid delivery contract field: {key}")

    for key in ("golden_device", "golden_source"):
        value = contract.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing delivery contract field: {key}")

    config_path = op_dir.parent.parent / "triton_op_generator.config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("project config must contain an object")
        for key in (*required_values, "golden_device", "golden_source"):
            if key not in config:
                continue
            if (
                type(config[key]) is not type(contract[key])
                or config[key] != contract[key]
            ):
                raise ValueError(
                    f"delivery contract disagrees with project config: {key}"
                )

    anchor_path = op_dir / "output" / ".baseline_anchor.json"
    if anchor_path.is_file():
        anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
        if not isinstance(anchor, dict):
            raise ValueError("frozen baseline must contain an object")
        source_mode = anchor.get("source_mode")
        if source_mode in ("user", "auto") and source_mode != contract["golden_source"]:
            raise ValueError("golden_source disagrees with frozen baseline")

    for key in (*required_values, "golden_device", "golden_source"):
        if key in precheck and (
            type(precheck[key]) is not type(contract[key])
            or precheck[key] != contract[key]
        ):
            raise ValueError(f"conflicting delivery contract field: {key}")


def _iter_sort_key(path: Path):
    """iter_10 必须排在 iter_9 之后（字典序会把 iter_10 排前面）。"""
    suffix = path.name.removeprefix("iter_")
    return (0, int(suffix)) if suffix.isdigit() else (1, 0)


def check_iteration_evidence(op_dir, baselines):
    """Phase 3 的 iter_N 必须承载多 kernel 基线，且该轮验证全过并对应同一文件。"""
    iter_dirs = sorted((op_dir / "output").glob("iter_*"), key=_iter_sort_key)
    codes = [
        path
        for path in (directory / "generated_code.py" for directory in iter_dirs)
        if path.is_file()
    ]
    if not codes:
        raise ValueError("iteration phase requires output/iter_*/generated_code.py")
    code = codes[-1]
    artifact = read_artifact(
        op_dir, str(code.relative_to(op_dir)), process_evidence=False
    )
    if artifact["jit_count"] < 2:
        raise ValueError(
            "iteration artifact must be a multi-kernel baseline (>=2 @triton.jit)"
        )
    result = {
        "path": str(code.relative_to(op_dir)),
        "sha256": artifact["sha256"],
        "jit_count": artifact["jit_count"],
        "verify_validated": False,
    }
    # 该门在 Step 0.5（验证之前）执行，报告可能尚未产出：存在才校验，不存在留给后续阶段。
    report = code.parent / "verify" / "verify_result.json"
    if report.is_file():
        data = json.loads(report.read_text(encoding="utf-8"))
        total = data.get("total_cases")
        passed = data.get("passed_cases")
        if type(total) is not int or total <= 0 or passed != total:
            raise ValueError(f"iteration verification incomplete: {passed}/{total}")
        if data.get("implementation_sha256") != artifact["sha256"]:
            raise ValueError(
                "iteration verification was produced from a different file"
            )
        result["verify_validated"] = True
    return result


def check_baseline_verify(op_dir, report_name, baselines):
    if not report_name:
        raise ValueError("missing --baseline-verify result")
    path = (op_dir / report_name).resolve()
    if not path.is_relative_to(op_dir):
        raise ValueError("baseline verification must belong to this operator run")
    report = json.loads(path.read_text(encoding="utf-8"))
    total = report.get("total_cases")
    passed = report.get("passed_cases")
    if (
        type(total) is not int
        or total <= 0
        or type(passed) is not int
        or passed != total
        or report.get("failures") != []
    ):
        raise ValueError(f"baseline verification incomplete: {passed}/{total}")
    sources = report.get("local_source_sha256", {})
    if not isinstance(sources, dict):
        raise TypeError("invalid verification source hashes")
    executed = report.get("implementation_sha256")
    if not isinstance(executed, str):
        raise TypeError("verification report has no executed entry hash")
    if all(item["sha256"] != executed for item in baselines):
        raise ValueError("declared baseline was not the executed verification entry")
    hashes = set(sources.values())
    hashes.add(executed)
    if any(item["sha256"] not in hashes for item in baselines):
        raise ValueError("verification does not cover the declared baseline source")
    return {
        "path": str(path.relative_to(op_dir)),
        "passed_cases": passed,
        "total_cases": total,
        "executed_baseline_sha256": executed,
    }


ATTEMPT_MARKERS = ("稠密", "写者=消费者", "核边界")
ATTEMPT_HEADER = ("形态", "工作区规模", "写读形态", "正确性", "可复核")


def check_fallback_evidence(op_dir: Path, name: str | None) -> dict:
    """--fallback 需要 dev/feasibility_attempts.md：三项结构尝试 + 五列表头。"""
    path = op_dir / (name or "dev/feasibility_attempts.md")
    if not path.is_file():
        raise ValueError(
            "fallback requires dev/feasibility_attempts.md (dense handoff / writer=consumer / kernel-boundary split)"
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    missing = [m for m in ATTEMPT_MARKERS if m not in text]
    if missing:
        raise ValueError(f"feasibility attempts missing: {', '.join(missing)}")
    header_missing = [h for h in ATTEMPT_HEADER if h not in text]
    if header_missing:
        raise ValueError(
            f"feasibility attempts header missing: {', '.join(header_missing)}"
        )
    rows = [
        line
        for line in text.splitlines()
        if line.lstrip().startswith("|") and any(ch.isdigit() for ch in line)
    ]
    if len(rows) < 3:
        raise ValueError(
            "fallback requires >=3 measured structural attempts (table rows with numbers)"
        )
    for marker in ATTEMPT_MARKERS:
        marker_rows = [line for line in text.splitlines() if marker in line]
        if not marker_rows:
            raise ValueError(f"feasibility attempts missing: {marker}")
        implemented = False
        for line in marker_rows:
            for ref in re.findall(r"dev/[\w.-]+\.py", line):
                if (op_dir / ref).is_file():
                    implemented = True
        if not implemented:
            raise ValueError(
                f"feasibility attempt '{marker}' has no implemented artifact: each structural form "
                "must reference an existing dev/*.py file (analysis only is not accepted)"
            )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path.relative_to(op_dir)),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "measured_rows": len(rows),
    }


def find_fused_stage_verify(op_dir: Path) -> dict:
    """融合阶段验证证据：接受 verify_result*.json，但要求必测用例全过。"""
    verify_dir = op_dir / "output" / "opt_iter_0" / "verify"
    candidates = (
        sorted(verify_dir.glob("verify_result*.json")) if verify_dir.is_dir() else []
    )
    if not candidates:
        raise ValueError(
            "fallback requires the fused stage to be verified first "
            "(output/opt_iter_0/verify/verify_result*.json)"
        )
    failures = []
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            failures.append(f"{path.name}: unreadable")
            continue
        total = data.get("total_cases")
        passed = data.get("passed_cases")
        if type(total) is int and total > 0 and passed == total:
            return {
                "path": str(path.relative_to(op_dir)),
                "passed": passed,
                "total": total,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        failures.append(f"{path.name}: {passed}/{total}")
    raise ValueError(
        "fused stage verification did not pass ({}); fix the fused stage before single launch".format(
            ", ".join(failures)
        )
    )


def single_launch_audit(op_dir: Path) -> dict:
    """单 launch 证据审计：不管性能门是否通过都记录，保证证据不丢。"""
    audit = {}
    candidate = op_dir / "dev" / "single_launch.py"
    audit["candidate_exists"] = candidate.is_file()
    if candidate.is_file():
        try:
            artifact = read_artifact(
                op_dir, "dev/single_launch.py", process_evidence=True
            )
            audit["jit_count"] = artifact["jit_count"]
            audit["sha256"] = artifact["sha256"]
        except (OSError, TypeError, ValueError, SyntaxError) as exc:
            audit["candidate_error"] = str(exc)
        source = candidate.read_text(encoding="utf-8", errors="replace")
        audit["launch_call_sites"] = count_launch_call_sites(source)
    log = op_dir / "dev" / "single_launch_attempts.md"
    audit["attempts_exists"] = log.is_file()
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace")
        audit["attempt_rows"] = len(
            [
                line
                for line in text.splitlines()
                if line.lstrip().startswith("|") and any(ch.isdigit() for ch in line)
            ]
        )
        audit["mentions_verify_evidence"] = bool(
            re.search(r"verify_result[\w.-]*\.json", text)
        )
    try:
        fused = find_fused_stage_verify(op_dir)
        audit["fused_stage_verify"] = fused
        audit["order_ok"] = (
            candidate.is_file()
            and candidate.stat().st_mtime >= (op_dir / fused["path"]).stat().st_mtime
        )
    except ValueError as exc:
        audit["fused_stage_verify_error"] = str(exc)
    return audit


def check_single_launch_attempts(
    op_dir: Path, candidate: Path, fused_verify_path: str
) -> dict:
    """--fallback 需证明单 launch 是"分阶段尝试过、且证据未删除"，不规定实现方式。"""
    fused_verify = op_dir / fused_verify_path
    order_bad = candidate.stat().st_mtime < fused_verify.stat().st_mtime
    log = op_dir / "dev" / "single_launch_attempts.md"
    if not log.is_file():
        raise ValueError(
            "fallback requires dev/single_launch_attempts.md recording each staged attempt"
        )
    text = log.read_text(encoding="utf-8", errors="replace")
    rows = [
        line
        for line in text.splitlines()
        if line.lstrip().startswith("|") and any(ch.isdigit() for ch in line)
    ]
    if len(rows) < 2:
        raise ValueError(
            "dev/single_launch_attempts.md needs >=2 measured attempt rows "
            "(a minimalist sub-case first, then the full case)"
        )
    referenced = set(
        re.findall(
            r"(?:^|[\s`(\"'])((?:dev|output)/[\w./-]+\.(?:json|log|md|py))", text
        )
    )
    missing = [p for p in sorted(referenced) if not (op_dir / p).exists()]
    if missing:
        raise ValueError(
            "single-launch attempt evidence referenced but missing (deleted?): {}".format(
                ", ".join(missing)
            )
        )
    verify_refs = []
    for p in referenced:
        if re.search(r"verify_result[\w.-]*\.json$", p):
            verify_refs.append(p)
    if not verify_refs:
        raise ValueError(
            "single-launch attempts must include at least one verify_result*.json evidence "
            "(run verify.py on the candidate, even if it fails)"
        )
    if "未越序" in text and order_bad:
        raise ValueError(
            "attempt log claims no order violation, but the candidate predates the fused-stage "
            "verification (unsupported compliance claim)"
        )
    if order_bad:
        raise ValueError(
            "single-launch candidate was created before the fused stage was verified "
            "(order violation: attempt it after the fused stage passes)"
        )
    return {
        "attempts": str(log.relative_to(op_dir)),
        "measured_rows": len(rows),
        "evidence_refs": len(referenced),
        "verify_evidence": sorted(verify_refs),
        "fused_stage_verify": fused_verify_path,
    }


def check_single_launch_variants(op_dir: Path, attempts_path: Path) -> dict:
    """回退前必须有 >=3 个结构不同（内容 sha 不同）的单 launch 变体，且都在尝试记录中留痕。"""
    seen = {}
    for path in sorted((op_dir / "dev").glob("single_launch*.py")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        seen.setdefault(digest, []).append(path.name)
    distinct = sorted(seen.values(), key=lambda names: names[0])
    if len(distinct) < 3:
        raise ValueError(
            "fallback requires >=3 structurally different single-launch variants "
            f"(found {len(distinct)} distinct contents among dev/single_launch*.py)"
        )
    text = attempts_path.read_text(encoding="utf-8", errors="replace")
    missing = [names[0] for names in distinct if not any(n in text for n in names)]
    if missing:
        raise ValueError(
            "single-launch variants not recorded in dev/single_launch_attempts.md: {}".format(
                ", ".join(missing)
            )
        )
    return {"variants": [names[0] for names in distinct], "count": len(distinct)}


def check_single_launch_verify_rounds(op_dir: Path) -> dict:
    """回退前至少两轮单 launch 全量验证结果（不同版本各一轮）。"""
    paths = set()
    for directory in sorted((op_dir / "output").glob("single_launch*/verify")):
        paths.update(directory.glob("verify_result*.json"))
    if len(paths) < 2:
        raise ValueError(
            "fallback requires >=2 rounds of single-launch verification "
            f"(found {len(paths)}); run verify.py on at least two variants"
        )
    rounds = []
    for path in sorted(paths):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rounds.append(
                {"path": str(path.relative_to(op_dir)), "result": "unreadable"}
            )
            continue
        rounds.append(
            {
                "path": str(path.relative_to(op_dir)),
                "passed": data.get("passed_cases"),
                "total": data.get("total_cases"),
            }
        )
    return {"rounds": rounds}


def check_probe_claim(attempts_path: Path) -> dict:
    """最小同构探针通过时必须解释"为何全量不同构"，否则不得回退。"""
    text = attempts_path.read_text(encoding="utf-8", errors="replace")
    probe_lines = [line for line in text.splitlines() if "探针" in line]
    if not probe_lines:
        raise ValueError(
            "dev/single_launch_attempts.md must record the minimal isomorphic probe result"
        )
    joined = " ".join(probe_lines)
    if "3/3" not in joined:
        return {"probe_passed": False}
    explained = [
        line
        for line in text.splitlines()
        if ("不同构" in line or "差异" in line) and any(ch.isdigit() for ch in line)
    ]
    if not explained:
        raise ValueError(
            "minimal probe passed but the full kernel is non-isomorphic to it; "
            "record the concrete difference (program/tile/consumer counts or shapes) "
            "with numbers before falling back"
        )
    return {"probe_passed": True, "explained_rows": len(explained)}


def check_failure_clustering(attempts_path: Path) -> dict:
    """回退前必须给出失败 case 的 shape 聚类与针对性变体。"""
    text = attempts_path.read_text(encoding="utf-8", errors="replace")
    if "失败 case" not in text:
        raise ValueError(
            "dev/single_launch_attempts.md must list the failing cases with their shapes"
        )
    if "聚类" not in text and "定向" not in text:
        raise ValueError(
            "group the failing cases by shape feature and record at least one targeted "
            "variant for that group before falling back"
        )
    return {"ok": True}


def check_next_actions(op_dir: Path) -> dict:
    """回退只是检查点：必须给出可执行的下一步，供后续继续尝试。"""
    path = op_dir / "dev" / "next_actions.md"
    if not path.is_file():
        raise ValueError(
            "fallback is a checkpoint, not delivery: dev/next_actions.md with concrete "
            "next attempts is required"
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = [
        line
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#") and len(line.strip()) > 8
    ]
    if not rows:
        raise ValueError("dev/next_actions.md needs at least one concrete next action")
    return {
        "path": "dev/next_actions.md",
        "actions": len(rows),
        "bytes": path.stat().st_size,
    }


def check_single_launch_candidate(op_dir: Path) -> dict:
    """--fallback 前必须真的做过单 launch 候选：1 个 JIT + 宿主 1 处 launch 调用。"""
    candidate = op_dir / "dev" / "single_launch.py"
    if not candidate.is_file():
        raise ValueError(
            "fallback requires the attempted single-launch candidate dev/single_launch.py"
        )
    artifact = read_artifact(op_dir, "dev/single_launch.py", process_evidence=True)
    if artifact["jit_count"] < 1:
        raise ValueError(
            "dev/single_launch.py must define at least one @triton.jit kernel"
        )
    source = candidate.read_text(encoding="utf-8", errors="replace")
    launch_calls = count_launch_call_sites(source)
    if launch_calls != 1:
        raise ValueError(
            f"dev/single_launch.py must issue exactly one kernel launch call site "
            f"(got {launch_calls}); helper @triton.jit functions may be inlined but "
            f"must not be launched"
        )
    return {
        "path": "dev/single_launch.py",
        "sha256": artifact["sha256"],
        "jit_count": artifact["jit_count"],
        "launch_call_sites": launch_calls,
    }


PHASE2_STAGE_MARKERS = ("multi", "fused", "single")


def check_phase2_artifacts(op_dir: Path, required_path: bool) -> dict:
    """Phase 3 入口：precheck.json 必须存在；sketch.txt 可选，缺失时以 precheck.delivery_path 作为等价的三阶段路径证据。"""
    precheck = op_dir / "precheck.json"
    sketch = op_dir / "sketch.txt"
    if not precheck.is_file():
        raise ValueError("phase 3 requires precheck.json from Phase 2 Step 1")
    body = ""
    if sketch.is_file():
        body = sketch.read_text(encoding="utf-8", errors="replace")
        if not body.strip():
            raise ValueError("sketch.txt is empty")
    # 仓内没有统一产出 sketch.txt 的组件：允许以 precheck.json 的 delivery_path 作为等价证据。
    precheck_data = json.loads(precheck.read_text(encoding="utf-8"))
    declared_path = (
        (precheck_data.get("delivery_contract") or {}).get("delivery_path")
        or precheck_data.get("delivery_path")
        or ""
    )
    recorded = f"{body}\n{declared_path}".lower()
    if required_path:
        missing = [m for m in PHASE2_STAGE_MARKERS if m not in recorded]
        if missing:
            raise ValueError(
                "phase 3 requires sketch.txt or precheck.delivery_path recording the "
                "three-stage delivery path (missing: {})".format(", ".join(missing))
            )
    return {
        "precheck": str(precheck.relative_to(op_dir)),
        "sketch": str(sketch.relative_to(op_dir)) if sketch.is_file() else None,
        "sketch_bytes": len(body.encode("utf-8")),
        "delivery_path_declared": bool(declared_path),
    }


def _latest_iter_perf(op_dir: Path):
    """取最新一轮带 perf 结果的迭代目录（iter_9 < iter_10，且跳过无 perf 的目录）。"""
    for directory in sorted(
        (op_dir / "output").glob("iter_*"), key=_iter_sort_key, reverse=True
    ):
        perf = directory / "perf_result.json"
        if perf.is_file():
            return directory, perf
    return None, None


def check_multikernel_perf_floor(op_dir: Path, phase: str, floor: float) -> dict:
    """Phase 4 入口：最新一轮多 kernel 基线必须先过 per-case 下限，才允许进入融合。"""
    iter_dir, perf = _latest_iter_perf(op_dir)
    if perf is None:
        raise ValueError(
            "fusion/final requires output/iter_*/perf_result.json (multi-kernel performance gate)"
        )
    d = json.loads(perf.read_text(encoding="utf-8"))
    rows = d.get("per_shape_results") or []
    if not rows:
        raise ValueError(f"{iter_dir.name}/perf_result.json has no per-case results")
    vals = []
    unusable = []
    for row in rows:
        value = row.get("speedup_vs_torch")
        status = row.get("status")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or (status not in (None, "pass"))
        ):
            unusable.append(row.get("case_idx"))
            continue
        vals.append((row.get("case_idx"), float(value)))
    if unusable:
        raise ValueError(
            "per-case speedup missing or unusable (measurement issue, not a performance result): "
            f"cases {unusable}"
        )
    if not vals:
        raise ValueError(
            f"{iter_dir.name}/perf_result.json has no usable speedup values"
        )
    worst_case, worst = min(vals, key=lambda x: x[1])
    speedups = [value for _, value in vals]
    geomean = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
    if phase == "fusion" and floor < 1.0:
        raise ValueError(
            f"fusion requires the full per-case floor 1.0 (got --perf-floor {floor:.2f}); "
            "relaxing the floor may only be used when recording the final fallback, not to enter fusion"
        )
    effective_floor = round(floor * (1.0 - PERF_TOLERANCE), 6)
    if worst < effective_floor:
        raise ValueError(
            f"multi-kernel version below per-case floor: case {worst_case} = {worst:.4f} "
            f"< {effective_floor:.4f} (floor {floor:.2f} with {PERF_TOLERANCE:.0%} tolerance) "
            "(optimize the multi-kernel baseline first; do not enter fusion)"
        )
    if phase == "fusion" and geomean < effective_floor:
        raise ValueError(
            f"multi-kernel geomean {geomean:.4f} < 1.0 before fusion "
            "(optimize the multi-kernel baseline so that both the average and every case reach 1.0)"
        )
    return {
        "iter": iter_dir.name,
        "perf_result": str(perf.relative_to(op_dir)),
        "geomean_speedup": geomean,
        "min_case": worst_case,
        "min_speedup": worst,
        "floor": floor,
        "tolerance": PERF_TOLERANCE,
        "effective_floor": effective_floor,
        "floor_relaxed": floor < 1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-dir", required=True, type=Path)
    parser.add_argument(
        "--single-launch-required", choices=("true", "false"), default="true"
    )
    parser.add_argument(
        "--phase",
        choices=("precheck", "iteration", "fusion", "final"),
        default="iteration",
    )
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--baseline-verify")
    parser.add_argument("--fused-epilogue")
    parser.add_argument("--final-code")
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Record an explicit multi-kernel fallback",
    )
    parser.add_argument(
        "--feasibility-attempts",
        help="Path to dev/feasibility_attempts.md (required with --fallback)",
    )
    parser.add_argument(
        "--perf-floor",
        type=float,
        default=1.0,
        help="Per-case speedup floor that the multi-kernel version must meet before fusion",
    )
    args = parser.parse_args()
    if not (math.isfinite(args.perf_floor) and args.perf_floor > 0):
        print("check-launch-path: --perf-floor must be a positive finite number")
        return 1

    op_dir = args.op_dir.resolve()
    required = args.single_launch_required == "true"
    record = {
        "single_launch_required": required,
        "stage": args.phase,
        "op_dir": str(op_dir),
    }
    code = 0
    perf_error = None
    try:
        if not required:
            record["verdict"] = "not_required"
        else:
            validate_contract(op_dir)
            record["verdict"] = "pass_precheck"

        if required and args.phase != "precheck":
            baselines = [
                read_artifact(op_dir, name, process_evidence=True)
                for name in args.baseline
            ]
            record["baselines"] = baselines
            baseline_count = sum(
                artifact["jit_count"]
                for artifact in {item["path"]: item for item in baselines}.values()
            )
            if baseline_count < 2:
                raise ValueError("no declared multi-kernel baseline found")
            record["phase2_artifacts"] = check_phase2_artifacts(op_dir, required)
            if args.phase == "iteration":
                record["iteration_evidence"] = check_iteration_evidence(
                    op_dir, baselines
                )
            record["verdict"] = "pass_iteration"

        if required and args.phase in ("fusion", "final"):
            if args.phase == "final" and not args.fallback and args.perf_floor < 1.0:
                raise ValueError(
                    "relaxed --perf-floor may only be used when recording the final "
                    "fallback; final delivery requires floor >= 1.0"
                )
            if args.phase == "fusion" and args.fused_epilogue:
                fused = read_artifact(
                    op_dir, args.fused_epilogue, process_evidence=True
                )
                if not fused["jit_count"]:
                    raise ValueError("fused epilogue artifact has no @triton.jit")
                if any(fused["sha256"] == item["sha256"] for item in baselines):
                    raise ValueError(
                        "fused epilogue must be a distinct kernel artifact"
                    )
                record["fused_epilogue"] = fused
            try:
                record["multikernel_perf_gate"] = check_multikernel_perf_floor(
                    op_dir, args.phase, args.perf_floor
                )
            except ValueError as exc:
                perf_error = str(exc)
                record["multikernel_perf_gate_error"] = perf_error
            if perf_error:
                if args.phase == "fusion":
                    raise ValueError(perf_error)
            else:
                record["baseline_verify"] = check_baseline_verify(
                    op_dir, args.baseline_verify, baselines
                )
                record["verdict"] = "pass_baseline_verify"

        if required and args.phase == "final":
            if not args.final_code or not args.fused_epilogue:
                raise ValueError(
                    "final phase requires --final-code and --fused-epilogue"
                )
            fused = read_artifact(op_dir, args.fused_epilogue, process_evidence=True)
            record["fused_epilogue"] = fused
            if not fused["jit_count"] or any(
                fused["sha256"] == item["sha256"] for item in baselines
            ):
                raise ValueError("fused epilogue must be a distinct kernel artifact")
            final = read_artifact(op_dir, args.final_code, process_evidence=False)
            final_source = (op_dir / args.final_code).read_text(
                encoding="utf-8", errors="replace"
            )
            launch_sites = count_launch_call_sites(final_source)
            record["final"] = {**final, "launch_call_sites": launch_sites}
            record["single_launch_audit"] = single_launch_audit(op_dir)
            if not args.fallback and final["sha256"] == fused["sha256"]:
                raise ValueError(
                    "the final kernel is a copy of the fused epilogue stage"
                )
            if perf_error:
                raise ValueError(perf_error)
            if args.fallback and final["jit_count"] >= 2:
                if "fallback" not in Path(args.final_code).name.lower():
                    raise ValueError(
                        "multi-kernel fallback is not a delivery: name the artifact "
                        "*_fallback.py and report delivery_status=not_delivered"
                    )
                fused_verify = find_fused_stage_verify(op_dir)
                record["fused_stage_verify"] = fused_verify
                record["single_launch_candidate"] = check_single_launch_candidate(
                    op_dir
                )
                attempts_path = op_dir / "dev" / "single_launch_attempts.md"
                record["single_launch_attempts"] = check_single_launch_attempts(
                    op_dir, op_dir / "dev" / "single_launch.py", fused_verify["path"]
                )
                record["single_launch_variants"] = check_single_launch_variants(
                    op_dir, attempts_path
                )
                record["single_launch_verify_rounds"] = (
                    check_single_launch_verify_rounds(op_dir)
                )
                record["probe_claim"] = check_probe_claim(attempts_path)
                record["failure_clustering"] = check_failure_clustering(attempts_path)
                record["fallback_evidence"] = check_fallback_evidence(
                    op_dir, args.feasibility_attempts
                )
                record["next_actions"] = check_next_actions(op_dir)
                record["delivery_status"] = "not_delivered"
                record["target_reached"] = False
                record["verdict"] = "contract_conflict"
                code = 3
            elif args.fallback or launch_sites != 1:
                raise ValueError(
                    "final artifact must contain exactly one kernel launch call site "
                    f"(got {launch_sites}); @triton.jit helper functions are allowed but "
                    "must not be launched from the host"
                )
            else:
                record["verdict"] = "pass_final_shape"
    except (OSError, TypeError, ValueError, SyntaxError) as exc:
        record.update(verdict="delivery_path_violation", reason=str(exc))
        code = 1

    evidence = op_dir / "output" / "launch_path.json"
    try:
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"check-launch-path: cannot write evidence: {exc}")
        return 1

    print(f"check-launch-path: {record['verdict']}")
    if "reason" in record:
        print(record["reason"])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
