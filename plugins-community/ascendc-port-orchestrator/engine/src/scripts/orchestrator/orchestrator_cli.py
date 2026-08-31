# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Command-line entry for the scoped AscendC port orchestrator.

Customer work starts through exactly two modes:

* ``--port-a3-ops`` for the current arch22 to arch35 migration;
* ``--backward`` for backward-operator generation.

Status, resume, planning, and optimization are lifecycle operations over an
already-scoped workspace.  They cannot create a general operator-generation
run; ``run_single_op`` independently enforces the persisted workspace mode.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from logging_config import get_logger

log = get_logger(__name__)

_ORCH_MARKER = "run_single_op"


def _orch():
    """Resolve the live orchestrator module without an import-time cycle."""
    for name in ("orchestrator", "orchestrator.orchestrator", "__main__"):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, _ORCH_MARKER):
            return module
    return sys.modules.get("orchestrator") or sys.modules["__main__"]


def _orch_attr(name: str):
    """Resolve a live orchestrator implementation detail for CLI dispatch."""
    return getattr(_orch(), name)


def _normalize_precision_standard(value: str | None) -> int:
    """Publish the explicit precision choice for the arch migration grader."""
    if value is None:
        return 0
    aliases = {"生态": "ecosystem", "商用": "commercial"}
    normalized = aliases.get(value.strip(), value.strip())
    if normalized not in ("ecosystem", "commercial"):
        print(
            "ERROR: --precision-standard must be ecosystem|commercial "
            f"(aliases 生态|商用); got {value!r}"
        )
        return 2
    os.environ["AOG_PRECISION_STANDARD_CLI"] = normalized
    log.info(
        "[orch] --precision-standard=%s -> arch migration grader CLI",
        normalized,
    )
    return 0


def _conflicts(args, *, excluded: tuple[str, ...]) -> list[str]:
    """Return user-facing names for populated mutually-exclusive arguments."""
    labels = {
        "op": "positional <op>",
        "resume": "--resume",
        "all": "--all",
        "dry_run": "--dry-run",
        "plan": "--plan",
        "status": "--status",
        "optimize": "--optimize",
        "cold_start": "--cold-start",
        "lane": "--lane",
        "extra_lane": "--extra-lane",
        "workspace": "--workspace",
        "bump_cap": "--bump-cap",
        "precision_standard": "--precision-standard",
        "timing": "--timing",
        "port_a3": "--port-a3-ops",
        "backward": "--backward",
        "reference_source": "--reference-source",
        "npubench_task": "--npubench-task",
        "npubench_root": "--npubench-root",
        "source_kind": "--source-kind",
        "source_arch": "--source-arch",
        "candidate_kind": "--candidate-kind",
    }
    conflicts = []
    for name in excluded:
        value = getattr(args, name, None)
        if value is not None and (name == "lane" or bool(value)):
            conflicts.append(labels[name])
    return conflicts


def _validated_lane(value: int | None) -> tuple[int | None, int]:
    """Resolve the default lane and fail closed outside the detected range."""
    lane = 0 if value is None else value
    max_lane = _orch_attr("_detect_max_lane")()
    if lane < 0 or lane > max_lane:
        print(f"ERROR: --lane must be in range 0..{max_lane}; got {lane}")
        return None, 2
    return lane, 0


def _workspace_identity_matches(value: object, workspace: Path) -> bool:
    """Match a durable workspace identity without following an input alias."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        actual = Path(workspace).resolve(strict=False)
        declared = Path(value).expanduser()
        if declared.is_absolute():
            return declared.resolve(strict=False) == actual
        # Older durable state may persist the scoped directory name rather than
        # an absolute path.  Accept that narrow form and a path relative to the
        # workspace parent; never accept an arbitrary basename from another tree.
        return value == actual.name or (actual.parent / declared).resolve(strict=False) == actual
    except (OSError, RuntimeError, ValueError):
        return False


def _persisted_identity_error(op: str, workspace: Path) -> str | None:
    """Return a user-facing error when durable scope identity is contradictory."""
    workspace = Path(workspace)
    if workspace.is_symlink():
        return "workspace must be a real non-symlink directory"
    state_path = workspace / ".opgen_state.json"
    if not state_path.exists():
        return None
    if state_path.is_symlink() or not state_path.is_file():
        return "durable state must be a regular non-symlink file"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return f"durable state is unreadable: {type(exc).__name__}: {exc}"
    if not isinstance(state, Mapping):
        return "durable state must be a JSON object"

    records: list[Mapping] = [state]
    nested = state.get("identity")
    if nested is not None:
        if not isinstance(nested, Mapping):
            return "durable state identity must be an object"
        records.append(nested)
    for record in records:
        if "op" in record:
            declared_op = record.get("op")
            if not isinstance(declared_op, str) or not declared_op.strip():
                return "durable state op identity is malformed"
            if declared_op != op:
                return (
                    "positional op does not match durable state identity: "
                    f"state={declared_op!r}, requested={op!r}"
                )
        if "workspace" in record:
            declared_workspace = record.get("workspace")
            if not _workspace_identity_matches(declared_workspace, workspace):
                return (
                    "workspace does not match durable state identity: "
                    f"state={declared_workspace!r}, resolved={str(workspace)!r}"
                )
    return None


def main() -> int:
    _orch_attr("_refuse_if_detached")()
    parser = argparse.ArgumentParser(
        prog="orchestrator",
        allow_abbrev=False,
        description=(
            "AscendC arch22->arch35 migration and backward-operator generation"
        ),
    )
    parser.add_argument(
        "op",
        nargs="?",
        default=None,
        help="existing scoped workspace name (for --resume/--plan/--optimize only)",
    )
    parser.add_argument(
        "--port-a3-ops",
        dest="port_a3",
        type=Path,
        metavar="SOURCE_DIR",
        help=(
            "migrate an arch22 ops-nn operator from the CANN ops repository "
            "(canonical ops-format route)"
        ),
    )
    parser.add_argument(
        "--source-kind",
        choices=(
            "port-aclnn-tilelang2ascendc",
            "port_aclnn_tilelang2ascendc",
        ),
        default=None,
        help=(
            "source layout; omit for the legacy ACLNN ops-nn route; "
            "port-aclnn-tilelang2ascendc selects the TileLang2AscendC custom-op project"
        ),
    )
    parser.add_argument(
        "--source-arch",
        choices=("arch22", "arch35"),
        default=None,
        help=(
            "explicit architecture for TileLang2AscendC project sources (arch35)"
        ),
    )
    parser.add_argument(
        "--candidate-kind",
        choices=("tilelang2ascendc_custom_op",),
        default=None,
        help="required target ABI/project contract for explicit source kinds",
    )
    parser.add_argument(
        "--backward",
        type=Path,
        metavar="FORWARD_SPEC",
        help="generate a backward operator from a differentiable forward spec",
    )
    reference = parser.add_argument_group("reference provider")
    reference.add_argument(
        "--reference-source",
        choices=("npubench", "a3_live", "cannbench"),
        default=None,
        help=(
            "reference provider: npubench (NPUKernelBench task+sidecar), "
            "a3_live (explicit live A3 capture), or reserved cannbench"
        ),
    )
    reference.add_argument(
        "--npubench-task",
        type=Path,
        metavar="TASK_PY",
        help="original NPUKernelBench task .py (requires --reference-source npubench)",
    )
    reference.add_argument(
        "--npubench-root",
        type=Path,
        metavar="TASK_ROOT",
        help="optional source-root closure for --npubench-task",
    )

    lifecycle = parser.add_argument_group("lifecycle")
    lifecycle.add_argument("--resume", action="store_true", help="resume scoped state")
    lifecycle.add_argument(
        "--all", action="store_true", help="with --resume, scan all scoped workspaces"
    )
    lifecycle.add_argument(
        "--dry-run", action="store_true", help="with --resume, diagnose only"
    )
    lifecycle.add_argument("--plan", action="store_true", help="show the plan only")
    lifecycle.add_argument("--status", action="store_true", help="show workspace states")
    lifecycle.add_argument(
        "--optimize",
        action="store_true",
        help="re-enter optimization for an already verified scoped workspace",
    )
    lifecycle.add_argument(
        "--cold-start",
        action="store_true",
        help="restart a --port-a3-ops or --backward run after archiving prior state",
    )

    parser.add_argument("--lane", type=int, default=None, help="NPU lane id (default: 0)")
    parser.add_argument(
        "--extra-lane",
        action="append",
        type=int,
        default=None,
        metavar="N",
        help="additional assigned NPU lane for npubench precision/performance parallelism",
    )
    parser.add_argument(
        "--workspace", type=Path, help="explicit existing workspace for lifecycle use"
    )
    parser.add_argument(
        "--bump-cap",
        action="append",
        default=None,
        metavar="COUNTER:DELTA",
        help="audited iteration-cap increase for this scoped run",
    )
    parser.add_argument(
        "--precision-standard",
        default=None,
        metavar="STD",
        help="arch migration precision standard: ecosystem or commercial",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="generate TIMING_REPORT.md from the scoped run event logs",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("ORCH_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="logger threshold",
    )

    args = parser.parse_args()

    from logging_config import setup_run_logger

    setup_run_logger(workspace=None, level=args.log_level)

    reference_flags = (
        args.reference_source,
        args.npubench_task,
        args.npubench_root,
        args.source_kind,
        args.source_arch,
        args.candidate_kind,
    )
    reference_flags_present = any(flag is not None for flag in reference_flags)
    if reference_flags_present and args.port_a3 is None:
        print(
            "ERROR: --reference-source, --npubench-task, and --npubench-root "
            "require --port-a3-ops"
        )
        return 2

    if args.status:
        conflicts = _conflicts(
            args,
            excluded=(
                "op", "resume", "all", "dry_run", "plan", "optimize",
                "cold_start", "lane", "extra_lane", "workspace", "bump_cap",
                "precision_standard", "timing", "port_a3", "backward",
                "reference_source", "npubench_task", "npubench_root",
                "source_kind", "source_arch", "candidate_kind",
            ),
        )
        if conflicts:
            print(f"ERROR: --status cannot be combined with: {', '.join(conflicts)}")
            return 2
        return _orch_attr("_cmd_status")()

    if args.port_a3 is not None:
        conflicts = _conflicts(
            args,
            excluded=(
                "op", "resume", "all", "dry_run", "optimize", "workspace",
                "backward",
            ),
        )
        if conflicts:
            print(f"ERROR: --port-a3-ops cannot be combined with: {', '.join(conflicts)}")
            return 2
        lane, lane_rc = _validated_lane(args.lane)
        if lane_rc:
            return lane_rc
        extra_lanes: list[int] = []
        for requested in args.extra_lane or []:
            extra_lane, extra_lane_rc = _validated_lane(requested)
            if extra_lane_rc:
                return extra_lane_rc
            if extra_lane == lane:
                print("ERROR: --extra-lane must differ from --lane")
                return 2
            if extra_lane in extra_lanes:
                print(f"ERROR: duplicate --extra-lane {extra_lane}")
                return 2
            extra_lanes.append(extra_lane)
        precision_rc = _normalize_precision_standard(args.precision_standard)
        if precision_rc:
            return precision_rc
        try:
            cap_bumps = _orch_attr("_parse_bump_caps")(args.bump_cap or [])
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 2
        return _orch_attr("_cmd_port_a3")(
            port_a3_dir=args.port_a3,
            lane=lane,
            plan_only=args.plan,
            cold_start=args.cold_start,
            cap_bumps=cap_bumps,
            timing=args.timing,
            reference_source=args.reference_source,
            npubench_task=args.npubench_task,
            npubench_root=args.npubench_root,
            extra_lanes=extra_lanes,
            source_kind=args.source_kind,
            source_arch=args.source_arch,
            candidate_kind=args.candidate_kind,
        )

    if args.backward is not None:
        conflicts = _conflicts(
            args,
            excluded=(
                "op", "resume", "all", "dry_run", "optimize", "workspace",
                "port_a3", "bump_cap", "precision_standard",
            ),
        )
        if conflicts:
            print(f"ERROR: --backward cannot be combined with: {', '.join(conflicts)}")
            return 2
        lane, lane_rc = _validated_lane(args.lane)
        if lane_rc:
            return lane_rc
        return _orch_attr("_cmd_backward")(
            forward_spec=args.backward,
            lane=lane,
            plan_only=args.plan,
            cold_start=args.cold_start,
            timing=args.timing,
        )

    if args.cold_start:
        print("ERROR: --cold-start requires --port-a3-ops or --backward so mode input is explicit")
        return 2

    if args.resume:
        conflicts = _conflicts(
            args,
            excluded=(
                "optimize", "plan", "workspace", "bump_cap",
                "precision_standard", "timing",
            ),
        )
        if args.all and args.op:
            conflicts.append("positional <op> with --all")
        if conflicts:
            print(f"ERROR: --resume cannot be combined with: {', '.join(conflicts)}")
            return 2
        lane, lane_rc = _validated_lane(args.lane)
        if lane_rc:
            return lane_rc
        return _orch_attr("_cmd_resume")(
            op=args.op,
            all_mode=args.all,
            lane=lane,
            dry_run=args.dry_run,
        )

    if args.all or args.dry_run:
        print("ERROR: --all and --dry-run require --resume")
        return 2

    if not args.op:
        parser.print_help()
        return 2

    try:
        workspace = (
            Path(args.workspace)
            if args.workspace is not None
            else Path(_orch_attr("_resolve_workspace")(args.op, backend="ascendc"))
        )
    except Exception as exc:
        print(
            "ERROR: could not resolve positional workspace for "
            f"{args.op!r}: {type(exc).__name__}: {exc}"
        )
        return 2

    identity_error = _persisted_identity_error(args.op, workspace)
    if identity_error:
        print(f"ERROR: positional workspace rejected: {identity_error}")
        return 2

    if not (args.plan or args.optimize):
        try:
            scoped_mode = _orch_attr("_read_scoped_opgen_mode")(workspace)
        except Exception as exc:
            print(
                "ERROR: could not inspect positional workspace state: "
                f"{type(exc).__name__}: {exc}"
            )
            return 2
        if scoped_mode is None:
            print(
                "ERROR: new work must start with --port-a3-ops or --backward; "
                "a positional workspace is lifecycle-only"
            )
            return 2
        # A positional re-invocation of an existing workspace with a persisted
        # scoped mode continues the state machine from its current state;
        # resume.py's mid-flight recovery relies on this entry point.

    conflicts = _conflicts(
        args,
        excluded=("precision_standard",),
    )
    if args.plan and args.optimize:
        conflicts.append("--plan with --optimize")
    if conflicts:
        print(
            "ERROR: positional lifecycle operation cannot be combined with: "
            f"{', '.join(conflicts)}"
        )
        return 2

    lane, lane_rc = _validated_lane(args.lane)
    if lane_rc:
        return lane_rc

    cap_bumps = {}
    if args.bump_cap:
        try:
            cap_bumps = _orch_attr("_parse_bump_caps")(args.bump_cap)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 2

    if args.optimize:
        try:
            ok, message = _orch_attr("_optimize_reentry_workspace")(workspace)
        except Exception as exc:
            print(
                "ERROR (--optimize workspace rejected): "
                f"{type(exc).__name__}: {exc}"
            )
            return 2
        log.info("[--optimize] %s", message)
        if not ok:
            print(f"ERROR (--optimize rejected): {message}")
            return 7

    return _orch().run_single_op(
        args.op,
        workspace=workspace,
        lane=lane,
        plan_only=args.plan,
        cap_bumps=cap_bumps,
        timing=args.timing,
        backend="ascendc",
    )
