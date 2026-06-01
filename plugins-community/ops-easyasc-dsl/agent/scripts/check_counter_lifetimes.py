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
"""Check counter/buffer lifetime smells in kernel files."""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCAN_ROOT = ROOT / "agent" / "example" / "kernels"
GENERIC_COUNTER_NAMES = {"cnt", "counter"}
BUFFER_BUILDERS = {"DBuff", "TBuff"}


def _call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _position_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


class KernelCounterVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.buffers: Dict[str, Dict[str, Any]] = {}
        self.counter_decls: Dict[str, Dict[str, Any]] = {}
        self.counter_uses: Dict[str, List[Dict[str, Any]]] = {}
        self.counter_increments: Dict[str, List[Dict[str, Any]]] = {}
        self.loop_stack: List[int] = []
        self.if_stack: List[int] = []

    def visit_For(self, node: ast.For) -> None:
        self.loop_stack.append(node.lineno)
        self.generic_visit(node)
        self.loop_stack.pop()

    def visit_If(self, node: ast.If) -> None:
        self.if_stack.append(node.lineno)
        self.generic_visit(node)
        self.if_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call):
            target = node.targets[0].id
            call_name = _call_name(node.value.func)
            if call_name in BUFFER_BUILDERS:
                position = None
                if len(node.value.args) >= 3:
                    position = _position_name(node.value.args[2])
                self.buffers[target] = {
                    "name": target,
                    "kind": call_name,
                    "position": position,
                    "line": node.lineno,
                }
            if call_name == "Var":
                self.counter_decls[target] = {
                    "name": target,
                    "line": node.lineno,
                }
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name):
            buffer_name = node.value.id
            if buffer_name in self.buffers:
                counter_name = self._extract_counter_name(node.slice)
                if counter_name is not None:
                    self.counter_uses.setdefault(counter_name, []).append(
                        {
                            "buffer": buffer_name,
                            "line": node.lineno,
                            "loop_depth": len(self.loop_stack),
                            "loop_signature": tuple(self.loop_stack),
                            "condition_signature": tuple(self.if_stack),
                            "position": self.buffers[buffer_name].get("position"),
                            "kind": self.buffers[buffer_name].get("kind"),
                        }
                    )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name) and isinstance(node.op, ast.Add):
            counter_name = node.target.id
            self.counter_increments.setdefault(counter_name, []).append(
                {
                    "line": node.lineno,
                    "loop_depth": len(self.loop_stack),
                    "loop_signature": tuple(self.loop_stack),
                    "condition_signature": tuple(self.if_stack),
                }
            )
        self.generic_visit(node)

    @staticmethod
    def _extract_counter_name(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Index):  # py38 compatibility
            return KernelCounterVisitor._extract_counter_name(node.value)
        return None


def _iter_python_files(paths: Sequence[str]) -> List[Path]:
    if not paths:
        paths = [str(DEFAULT_SCAN_ROOT)]
    collected: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_dir():
            collected.extend(sorted(item for item in path.rglob("*.py") if item.is_file()))
        elif path.is_file():
            collected.append(path)
        else:
            raise FileNotFoundError("Path not found: %s" % raw_path)
    unique = []
    seen = set()
    for path in collected:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _warning(code: str, counter: str, message: str, lines: Iterable[int]) -> Dict[str, Any]:
    return {
        "code": code,
        "counter": counter,
        "message": message,
        "lines": sorted(set(lines)),
    }


def analyze_file(path: Path) -> Dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = KernelCounterVisitor()
    visitor.visit(tree)

    counters = set(visitor.counter_decls) | set(visitor.counter_uses) | set(visitor.counter_increments)
    summaries = []
    warnings: List[Dict[str, Any]] = []

    for counter_name in sorted(counters):
        uses = visitor.counter_uses.get(counter_name, [])
        increments = visitor.counter_increments.get(counter_name, [])
        decl = visitor.counter_decls.get(counter_name)
        buffer_names = sorted({item["buffer"] for item in uses})
        positions = sorted({item["position"] for item in uses if item.get("position")})
        loop_depths = sorted({item["loop_depth"] for item in uses + increments})
        condition_signatures = sorted({item["condition_signature"] for item in uses if item["condition_signature"]})
        increment_loop_signatures = sorted({item["loop_signature"] for item in increments})
        usage_by_position_depth = sorted(
            {(item["position"], item["loop_depth"]) for item in uses if item.get("position")}
        )
        root_condition_lines = sorted(
            {
                signature[0]
                for signature in [
                    item["condition_signature"]
                    for item in uses + increments
                    if item["condition_signature"]
                ]
                if signature
            }
        )

        summary = {
            "name": counter_name,
            "decl_line": decl["line"] if decl else None,
            "buffers": buffer_names,
            "positions": positions,
            "use_lines": sorted({item["line"] for item in uses}),
            "increment_lines": sorted({item["line"] for item in increments}),
            "loop_depths": loop_depths,
            "condition_branch_count": len(condition_signatures),
            "buffer_count": len(buffer_names),
        }
        summaries.append(summary)

        if counter_name in GENERIC_COUNTER_NAMES:
            warnings.append(
                _warning(
                    "generic-counter-name",
                    counter_name,
                    "Use a stage-owned counter name such as l1_cnt, l0c_cnt, tile_cnt, or stage1_cnt"
                    " instead of a generic name.",
                    [decl["line"]] if decl else summary["use_lines"],
                )
            )

        if uses and not increments:
            warnings.append(
                _warning(
                    "counter-never-incremented",
                    counter_name,
                    "Counter indexes buffers but no '+=' increment site was found."
                    " Verify that slot lineage is intentional.",
                    summary["use_lines"],
                )
            )

        if len(summary["increment_lines"]) > 1:
            warnings.append(
                _warning(
                    "multiple-increment-sites",
                    counter_name,
                    "Counter is incremented at multiple source locations."
                    " This often means different lifetimes or loop-owned rhythms were mixed together.",
                    summary["increment_lines"],
                )
            )

        if len(increment_loop_signatures) > 1:
            warnings.append(
                _warning(
                    "multiple-loop-owned-increments",
                    counter_name,
                    "Counter is incremented under different loop signatures."
                    " Different loop-owned lifetimes should usually use different counters.",
                    summary["increment_lines"],
                )
            )

        if (
            "L1" in positions
            and any(pos in positions for pos in ["L0C", "UB"])
            and len({depth for _, depth in usage_by_position_depth}) > 1
        ):
            warnings.append(
                _warning(
                    "mixed-positions-across-depths",
                    counter_name,
                    "Counter mixes L1 streaming ownership with outer L0C/UB ownership across different loop depths."
                    " Review whether separate stage lifetimes were collapsed into one counter.",
                    summary["use_lines"],
                )
            )

        if len(root_condition_lines) > 1:
            warnings.append(
                _warning(
                    "conditional-stage-sharing",
                    counter_name,
                    "Counter is used across different root conditional branches."
                    " Review delayed producer/consumer ownership and confirm separate counters are not needed.",
                    summary["use_lines"] + summary["increment_lines"],
                )
            )

    try:
        display_path = str(path.relative_to(ROOT))
    except ValueError:
        display_path = str(path)

    return {
        "path": display_path,
        "counter_count": len(summaries),
        "counters": summaries,
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def print_text(results: List[Dict[str, Any]], show_summary: bool) -> None:
    total_warnings = sum(item["warning_count"] for item in results)
    for item in results:
        path = item["path"]
        warnings = item["warnings"]
        if not warnings and not show_summary:
            continue
        print(path)
        if warnings:
            for warning in warnings:
                print("  - [%s] %s" % (warning["code"], warning["message"]))
                print("    counter: %s | lines: %s" % (warning["counter"], ", ".join(str(x) for x in warning["lines"])))
        else:
            print("  - no warnings")
        if show_summary:
            for counter in item["counters"]:
                print(
                    "  * {name}: buffers={buffers} positions={positions} increments={increments}".format(
                        name=counter["name"],
                        buffers=counter["buffers"],
                        positions=counter["positions"],
                        increments=counter["increment_lines"],
                    )
                )
        print()
    if total_warnings == 0 and not show_summary:
        print("No warnings found.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check counter/buffer lifetime smells in kernel files.")
    parser.add_argument("paths", nargs="*", help="Kernel file or directory paths. Defaults to ./kernels.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument(
        "--show-summary", action="store_true",
        help="Print per-counter summaries even when no warnings exist.",
    )
    parser.add_argument("--fail-on-warning", action="store_true", help="Return exit code 1 if any warning is found.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        paths = _iter_python_files(args.paths)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    results = [analyze_file(path) for path in paths]
    total_warnings = sum(item["warning_count"] for item in results)

    if args.json:
        payload = {
            "path_count": len(results),
            "warning_count": total_warnings,
            "results": results,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_text(results, show_summary=args.show_summary)

    if args.fail_on_warning and total_warnings > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
