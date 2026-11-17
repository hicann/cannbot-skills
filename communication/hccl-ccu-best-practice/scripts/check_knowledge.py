#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Validate the byte-level source scope of reusable CCU knowledge cards."""

import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# These read-only tools must not create caches beside linked Skill sources.
sys.dont_write_bytecode = True

DEFAULT_BASELINE = (
    Path(__file__).resolve().parents[1] / "references/knowledge-baseline.json"
)


def explain_source(root, source, reviewed_head, diff_lines=60):
    """Bounded navigation hints. A diff never changes a card's trust status."""
    from inspect_sources import inside

    if root is None or not inside(root, source["path"]).is_file():
        return {
            "status": "SOURCE_UNAVAILABLE",
            "next": "locate this source or its replacement only",
        }
    if not isinstance(reviewed_head, str) or not re.fullmatch(
        r"[0-9a-f]{40}|[0-9a-f]{64}", reviewed_head
    ):
        return {
            "status": "BASELINE_UNAVAILABLE",
            "next": "no verified baseline commit; inspect listed symbols directly",
        }
    try:
        result = subprocess.run(
            [
                "git",
                "--no-pager",
                "-C",
                str(root),
                "show",
                f"{reviewed_head}:{source['path']}",
            ],
            capture_output=True,
            timeout=5,
            check=False,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return {
            "status": "BASELINE_UNAVAILABLE",
            "next": "baseline blob is unavailable; inspect listed symbols directly",
        }
    if (
        result.returncode
        or hashlib.sha256(result.stdout).hexdigest() != source["sha256"]
    ):
        return {
            "status": "BASELINE_UNAVAILABLE",
            "next": "baseline blob missing or bytes differ from reviewed hash; do not infer a diff",
        }
    try:
        before = result.stdout.decode("utf-8").splitlines()
        after = inside(root, source["path"]).read_text(encoding="utf-8").splitlines()
    except UnicodeError:
        return {
            "status": "SOURCE_NOT_TEXT",
            "next": "inspect source format and hashes directly",
        }
    changes = [
        (kind, a, b, c, d)
        for kind, a, b, c, d in difflib.SequenceMatcher(
            None, before, after, autojunk=False
        ).get_opcodes()
        if kind != "equal"
    ]
    diff = list(
        difflib.unified_diff(
            before,
            after,
            fromfile="reviewed/" + source["path"],
            tofile="current/" + source["path"],
            n=2,
        )
    )
    anchors = {
        symbol: {
            "reviewed": [i for i, line in enumerate(before, 1) if symbol in line][:8],
            "current": [i for i, line in enumerate(after, 1) if symbol in line][:8],
        }
        for symbol in source.get("symbols", [])
    }
    nearby = [
        symbol
        for symbol, locations in anchors.items()
        if any(
            max(1, c - 10) <= line <= d + 10
            for _, _, _, c, d in changes
            for line in locations["current"]
        )
    ]
    return {
        "status": "DIFF_AVAILABLE",
        "reviewed_head": reviewed_head,
        "changed_ranges": [
            {
                "kind": kind,
                "reviewed_start": a + 1,
                "reviewed_count": b - a,
                "current_start": c + 1,
                "current_count": d - c,
            }
            for kind, a, b, c, d in changes[:12]
        ],
        "symbol_locations": anchors,
        "nearby_symbols": nearby,
        "diff": diff[:diff_lines],
        "truncated": len(diff) > diff_lines or len(changes) > 12,
        "scope": "text anchors and changed ranges only; review affected contracts and dependencies independently",
    }


def check_cards(
    roots, op, baseline_path=DEFAULT_BASELINE, only=None, explain=False, diff_lines=60
):
    from inspect_sources import inside

    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    if baseline.get("schema_version") != 1 or not isinstance(
        baseline.get("cards"), dict
    ):
        raise ValueError("invalid knowledge baseline")
    if op is not None:
        if not isinstance(op, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", op):
            raise ValueError(
                "--op must be a source operator directory name such as all_reduce"
            )
        if (
            roots.get("hccl") is not None
            and not inside(roots["hccl"], f"src/ops/{op}").is_dir()
        ):
            raise ValueError(
                f"operator directory does not exist: src/ops/{op}; use the HCCL directory name, e.g. all_reduce"
            )
    cards = []
    if type(diff_lines) is not int or not 1 <= diff_lines <= 200:
        raise ValueError("diff_lines must be between 1 and 200")
    for name, card in baseline["cards"].items():
        if only is not None and name not in only:
            continue
        if only is None and card.get("ops") and op not in card["ops"]:
            continue
        entries = []
        for source in card["files"]:
            root = roots.get(source["repo"])
            path = inside(root, source["path"]) if root else None
            digest = (
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path and path.is_file()
                else None
            )
            expected = source["sha256"]
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"{name}: invalid baseline digest")
            status = (
                "UNVERIFIED"
                if digest is None
                else "MATCH"
                if digest == expected
                else "CHANGED"
            )
            entry = {
                "repo": source["repo"],
                "path": source["path"],
                "sha256": digest,
                "expected_sha256": expected,
                "status": status,
                "symbols": source.get("symbols", []),
            }
            if explain and status != "MATCH":
                heads = card.get("reviewed_heads", baseline.get("reviewed_heads", {}))
                entry["review_hint"] = explain_source(
                    root, source, heads.get(source["repo"]), diff_lines
                )
            entries.append(entry)
        if not entries:
            raise ValueError(f"{name}: empty source scope cannot prove MATCH")
        statuses = {entry["status"] for entry in entries}
        status = (
            "UNVERIFIED"
            if "UNVERIFIED" in statuses
            else "CHANGED"
            if "CHANGED" in statuses
            else "MATCH"
        )
        cards.append(
            {
                "card_id": name,
                "status": status,
                "scope": card["scope"],
                "reference": card["reference"],
                "optional": card.get("optional", False),
                "files": entries,
            }
        )
    if only is not None and {card["card_id"] for card in cards} != set(only):
        raise ValueError("unknown knowledge card")
    available = [
        name
        for name, card in baseline["cards"].items()
        if op is not None and op in card.get("ops", [])
    ]
    checked = [card["card_id"] for card in cards if card["card_id"] in available]
    coverage = (
        "GENERIC_ONLY"
        if op is None
        else "CHECKED"
        if checked
        else "NOT_SELECTED"
        if available
        else "NO_CARD"
    )
    return {
        "baseline_sha256": hashlib.sha256(Path(baseline_path).read_bytes()).hexdigest(),
        "scope": "listed source bytes only; verify branch applicability and new dependencies separately",
        "operator": {
            "name": op,
            "directory_verified": op is not None and roots.get("hccl") is not None,
            "status": coverage,
            "available_cards": available,
            "checked_cards": checked,
        },
        "cards": cards,
    }


def main(argv=None):
    from inspect_sources import repository

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hccl-root", required=True, type=Path)
    parser.add_argument("--hcomm-root", required=True, type=Path)
    parser.add_argument(
        "--op",
        required=True,
        help="existing HCCL source directory name, e.g. all_reduce; not the HVM/DSL alias",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="emit reusable cards and changed paths only",
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--cards",
        nargs="+",
        metavar="CARD_ID",
        help="check only explicitly used cards; does not prove other contracts",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="add bounded source diffs and symbol locations to drift entries",
    )
    parser.add_argument(
        "--diff-lines",
        type=int,
        default=60,
        help="maximum diff output lines per file (1..200)",
    )
    args = parser.parse_args(argv)
    try:
        report = check_cards(
            {"hccl": repository(args.hccl_root), "hcomm": repository(args.hcomm_root)},
            args.op,
            args.baseline,
            only=set(args.cards) if args.cards else None,
            explain=args.explain,
            diff_lines=args.diff_lines,
        )
        if args.summary:
            for card in report["cards"]:
                files = card["files"]
                card["files"] = [entry for entry in files if entry["status"] != "MATCH"]
        print(json.dumps(report, ensure_ascii=False, indent=2))
        covered = report["operator"]["status"] == "CHECKED" or args.cards is not None
        return (
            0
            if covered and all(card["status"] == "MATCH" for card in report["cards"])
            else 1
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"knowledge check failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
