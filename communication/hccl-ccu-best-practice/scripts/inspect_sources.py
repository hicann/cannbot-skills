#!/usr/bin/env python3
# Copyright (c) 2026 CANNBot contributors
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Read-only CCU source discovery and baseline comparison for explicit repositories."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# These read-only tools must not create caches beside linked Skill sources.
sys.dont_write_bytecode = True

DEFAULT_MAP = Path(__file__).resolve().parents[1] / "references" / "source-map.json"


def git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"},
        timeout=30,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"git {args[0]} failed in {root}: {result.stderr.strip()}")
    return result.stdout.strip()


def repository(value):
    root = Path(value).resolve(strict=True)
    if (
        not root.is_dir()
        or Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root
    ):
        raise ValueError(f"not a Git worktree root: {root}")
    return root


def inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"expected repository-relative path: {relative!r}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes repository: {relative}") from exc
    return path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_entry(roots, entry):
    if not isinstance(entry, dict) or entry.get("repo") not in roots:
        raise ValueError("source map entry needs repo=hccl|hcomm")
    if type(entry.get("required")) is not bool:
        raise ValueError("source map entry needs boolean required")
    expected = entry.get("sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("source map entry needs a SHA-256 baseline")
    symbols = entry.get("symbols", [])
    if not isinstance(symbols, list) or any(
        not isinstance(s, str) or not s for s in symbols
    ):
        raise ValueError("source map symbols must be nonempty strings")
    path = inside(roots[entry["repo"]], entry.get("path"))
    result = {
        "repo": entry["repo"],
        "path": entry["path"],
        "required": entry["required"],
        "expected_sha256": expected,
    }
    if not path.is_file():
        return {**result, "status": "missing", "sha256": None, "symbols": {}}
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    # Textual anchors aid navigation; presence does not prove an interface's signature.
    text = content.decode("utf-8")
    anchors = {
        symbol: [i for i, line in enumerate(text.splitlines(), 1) if symbol in line]
        for symbol in symbols
    }
    return {
        **result,
        "status": "unchanged" if digest == expected else "changed",
        "sha256": digest,
        "symbols": anchors,
    }


def inventory(root, op):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", op):
        raise ValueError("--op must be a directory name such as all_gather")
    op_root = inside(root, f"src/ops/{op}")
    if not op_root.is_dir():
        raise ValueError(f"operator directory does not exist: {op_root}")
    current = op_root / "algorithm" / "template" / "ccu"
    legacy = op_root / "template" / "ccu"
    if current.is_dir():
        layout = "algorithm/template/ccu"
        template_root = current
        executor = op_root / "algorithm" / "executor"
    elif legacy.is_dir():
        layout = "legacy template/ccu; re-check API contracts before implementation"
        template_root = legacy
        executor = op_root / "executor"
    else:
        raise ValueError(
            f"no CCU template directory for {op}; inspect the operator before adding an engine"
        )
    files = []
    for directory, role in (
        (template_root, "template/kernel"),
        (executor, "executor"),
        (op_root / "selector", "selector"),
    ):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and (
                path.suffix in {".h", ".hpp", ".cc", ".cpp"}
                or path.name == "CMakeLists.txt"
            ):
                rel = path.relative_to(root).as_posix()
                actual = inside(root, rel)
                files.append({"path": rel, "role": role, "sha256": sha256(actual)})
    return {"op": op, "layout": layout, "files": files}


def inspect(hccl, hcomm, op, source_map, baseline=None):
    roots = {"hccl": repository(hccl), "hcomm": repository(hcomm)}
    if (
        not isinstance(source_map, dict)
        or type(source_map.get("schema_version")) is not int
        or source_map.get("schema_version") != 1
        or not isinstance(source_map.get("files"), list)
        or not source_map["files"]
    ):
        raise ValueError(
            "invalid source-map schema (expected version 1 and nonempty files)"
        )
    entries = [source_entry(roots, entry) for entry in source_map["files"]]
    found = inventory(roots["hccl"], op)
    revisions = {}
    for name, root in roots.items():
        # Scope status to source files: unrelated LFS assets must not invoke filters.
        scope = sorted({entry["path"] for entry in entries if entry["repo"] == name})
        if name == "hccl":
            scope.append(f"src/ops/{op}")
        revisions[name] = {
            "root": str(root),
            "head": git(root, "rev-parse", "HEAD"),
            "branch": git(root, "branch", "--show-current") or None,
            "status_scope": scope,
            "scoped_status": git(
                root,
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
                "--",
                *scope,
            ),
        }
    from check_knowledge import check_cards

    return {
        "schema_version": 1,
        "scope": "source inventory only; no device, build or semantic verification",
        "repositories": revisions,
        "sources": entries,
        "inventory": found,
        "knowledge": check_cards(roots, op, baseline)
        if baseline is not None
        else check_cards(roots, op),
    }


def report_destination(output, repositories, op):
    """Allow external evidence and the plugin's exact task-report path only."""
    destination = Path(output).resolve()
    for name, repo in repositories.items():
        root = Path(repo["root"])
        try:
            relative = destination.relative_to(root)
        except ValueError:
            continue
        parts = relative.parts
        task_report = (
            name == "hccl"
            and len(parts) == 6
            and parts[:4] == ("src", "ops", op, "docs")
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", parts[4])
            and re.fullmatch(r"source-report(?:-r[1-9][0-9]*)?\.json", parts[5])
        )
        if not task_report:
            raise ValueError(
                "--output inside source repositories is limited to "
                f"HCCL/src/ops/{op}/docs/<task-id>/source-report[-r<positive revision>].json"
            )
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hccl-root", required=True, type=Path)
    parser.add_argument("--hcomm-root", required=True, type=Path)
    parser.add_argument("--op", required=True)
    parser.add_argument("--source-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="explicit reviewed knowledge baseline, shared with contract checkers",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print concise drift/knowledge results; --output still saves full evidence",
    )
    parser.add_argument(
        "--check-baseline", action="store_true", help="exit 1 for any baseline drift"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "write outside source repositories or to a task "
            "source-report[-r<positive revision>].json; previous revisions are "
            "retained and never overwritten; default stdout"
        ),
    )
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.source_map.read_text(encoding="utf-8"))
        report = inspect(args.hccl_root, args.hcomm_root, args.op, data, args.baseline)
        output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            destination = report_destination(
                args.output, report["repositories"], args.op
            )
            with destination.open("x", encoding="utf-8") as stream:
                stream.write(output)
            print(f"source report: {destination}")
        elif not args.summary:
            print(output, end="")
        if args.summary:
            summary = {
                "repositories": report["repositories"],
                "changed_sources": [
                    s for s in report["sources"] if s["status"] != "unchanged"
                ],
                "inventory_file_count": len(report["inventory"]["files"]),
                "knowledge": {
                    "baseline_sha256": report["knowledge"]["baseline_sha256"],
                    "operator": report["knowledge"]["operator"],
                    "cards": [
                        {
                            k: card[k]
                            for k in ("card_id", "status", "scope", "reference")
                        }
                        for card in report["knowledge"]["cards"]
                    ],
                },
            }
            for repo in summary["repositories"].values():
                repo.pop("status_scope", None)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        missing = any(
            s["required"] and s["status"] == "missing" for s in report["sources"]
        )
        drift = any(s["status"] != "unchanged" for s in report["sources"])
        return 1 if missing or (args.check_baseline and drift) else 0
    except (OSError, UnicodeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"source inspection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
