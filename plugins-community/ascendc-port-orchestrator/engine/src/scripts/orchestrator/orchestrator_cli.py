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

* ``--port-a3`` for the current arch22 to arch35 migration;
* ``--backward`` for backward-operator generation.

Status, resume, planning, and optimization are lifecycle operations over an
already-scoped workspace.  They cannot create a general operator-generation
run; ``run_single_op`` independently enforces the persisted workspace mode.
"""
from __future__ import annotations

import argparse
import os
import sys
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
        "workspace": "--workspace",
        "bump_cap": "--bump-cap",
        "precision_standard": "--precision-standard",
        "timing": "--timing",
        "port_a3": "--port-a3",
        "backward": "--backward",
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
        "--port-a3",
        type=Path,
        metavar="OPS_NN_OP_DIR",
        help="migrate an ops-nn operator from arch22 to arch35",
    )
    parser.add_argument(
        "--backward",
        type=Path,
        metavar="FORWARD_SPEC",
        help="generate a backward operator from a differentiable forward spec",
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
        help="restart a --port-a3 or --backward run after archiving prior state",
    )

    parser.add_argument("--lane", type=int, default=None, help="NPU lane id (default: 0)")
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

    if args.status:
        conflicts = _conflicts(
            args,
            excluded=(
                "op", "resume", "all", "dry_run", "plan", "optimize",
                "cold_start", "lane", "workspace", "bump_cap",
                "precision_standard", "timing", "port_a3", "backward",
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
            print(f"ERROR: --port-a3 cannot be combined with: {', '.join(conflicts)}")
            return 2
        lane, lane_rc = _validated_lane(args.lane)
        if lane_rc:
            return lane_rc
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
        print("ERROR: --cold-start requires --port-a3 or --backward so mode input is explicit")
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

    if not (args.plan or args.optimize):
        print(
            "ERROR: new work must start with --port-a3 or --backward; "
            "a positional workspace is lifecycle-only"
        )
        return 2

    conflicts = _conflicts(
        args,
        excluded=("bump_cap", "precision_standard"),
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

    workspace = args.workspace or _orch_attr("_resolve_workspace")(args.op, backend="ascendc")
    if args.optimize:
        ok, message = _orch_attr("_optimize_reentry_workspace")(workspace)
        log.info("[--optimize] %s", message)
        if not ok:
            print(f"ERROR (--optimize rejected): {message}")
            return 7

    return _orch().run_single_op(
        args.op,
        workspace=workspace,
        lane=lane,
        plan_only=args.plan,
        timing=args.timing,
        backend="ascendc",
    )
