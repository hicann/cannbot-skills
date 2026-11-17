#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Check declared Variable arithmetic against source-backed generator tables.

Input is sync-contract.json with a compute_ops member. This does not parse C++
expressions, establish deployment identity, or prove the chosen lowering.
"""

import argparse
import importlib
import hashlib
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

KNOWLEDGE = importlib.import_module("check_knowledge")
DEFAULT_BASELINE = KNOWLEDGE.DEFAULT_BASELINE
check_cards = KNOWLEDGE.check_cards

ARITHMETIC_REPS = {
    "ASSIGN",
    "ADD",
    "MUL",
    "SUB",
    "AND",
    "OR",
    "XOR",
    "NOT",
    "SHL",
    "SHR",
}
GENERATOR_DIR = "src/base_comm/resources/ccu/ccu_representation/reps/translator"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value, where):
    require(
        isinstance(value, str) and bool(value.strip()),
        f"{where}: expected nonempty string",
    )


def read_capabilities(root, baseline=DEFAULT_BASELINE):
    root = Path(root).resolve()
    knowledge = check_cards({"hcomm": root}, None, baseline, only={"generator-arith"})
    card = knowledge["cards"][0]
    changed = [entry["path"] for entry in card["files"] if entry["status"] != "MATCH"]
    require(
        card["status"] == "MATCH",
        "UNVERIFIED generator-arith; review changed/missing sources: "
        + ", ".join(changed),
    )
    tables = {}
    for version in ("V1", "V2"):
        path = root / GENERATOR_DIR / f"ccu_ins_generator_{version.lower()}.h"
        source = path.read_text(encoding="utf-8")
        match = re.search(r"\brepTypeInstrCount\s*=\s*\{(.*?)\};", source, re.S)
        require(match is not None, f"cannot locate {version} repTypeInstrCount")
        reps = set(re.findall(r"\{\s*CcuRepType::([A-Z_0-9]+)\s*,", match.group(1)))
        require({"ASSIGN", "ADD"} <= reps, f"incomplete {version} arithmetic table")
        tables[version] = reps & ARITHMETIC_REPS
    return tables, knowledge["baseline_sha256"]


def validate_evidence(evidence, target, contract_dir):
    require(
        isinstance(evidence, dict),
        "target_evidence must be an object with path and sha256",
    )
    text(evidence.get("path"), "target_evidence.path")
    digest = evidence.get("sha256")
    require(
        isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
        "target_evidence.sha256 must be 64 lowercase hex digits",
    )
    path = Path(evidence["path"])
    if not path.is_absolute():
        path = Path(contract_dir) / path
    content = path.read_bytes()
    require(
        hashlib.sha256(content).hexdigest() == digest,
        "target generator evidence hash mismatch",
    )
    observed = set(
        re.findall(r"\bInit CcuInsGeneratorV([12])\b", content.decode("utf-8"))
    )
    require(
        len(observed) == 1,
        "target evidence must identify one unambiguous generator initialization",
    )
    if target != "UNKNOWN":
        require(observed == {target[1:]}, f"target {target} conflicts with evidence")


def validate_contract(document, tables, contract_dir):
    require(
        isinstance(document, dict)
        and type(document.get("schema_version")) is int
        and document["schema_version"] == 1,
        "expected schema_version=1 object",
    )
    compute = document.get("compute_ops")
    require(
        isinstance(compute, dict),
        "compute_ops must declare the target and runtime arithmetic mapping",
    )
    target = compute.get("target_generator")
    require(
        target in {"UNKNOWN", "V1", "V2"}, "target_generator must be UNKNOWN, V1 or V2"
    )
    evidence = compute.get("target_evidence")
    if target == "V2" or evidence is not None:
        validate_evidence(evidence, target, contract_dir)
    allowed = tables["V1"] & tables["V2"] if target == "UNKNOWN" else tables[target]
    operations = compute.get("operations")
    require(
        isinstance(operations, list),
        "operations must list each runtime arithmetic expression/lowering",
    )
    if not operations:
        text(compute.get("no_runtime_ops_reason"), "no_runtime_ops_reason")
    for index, operation in enumerate(operations):
        require(isinstance(operation, dict), f"operations[{index}] must be an object")
        text(operation.get("expression"), f"operations[{index}].expression")
        text(operation.get("source_evidence"), f"operations[{index}].source_evidence")
        rep = operation.get("rep")
        require(
            isinstance(rep, str) and rep in ARITHMETIC_REPS,
            f"operations[{index}]: unreviewed arithmetic rep {rep!r}",
        )
        require(
            rep in allowed,
            f"{rep} is not supported for {target}; available arithmetic reps={sorted(allowed)}",
        )
    return {
        "status": "consistent",
        "target_generator": target,
        "operations_checked": len(operations),
        "available_arithmetic_reps": sorted(allowed),
        "scope": (
            "declared mapping and evidence contents only; review C++ lowering "
            "and current deployment identity independently"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--hcomm-root", required=True, type=Path)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, type=Path)
    args = parser.parse_args(argv)
    try:
        tables, digest = read_capabilities(args.hcomm_root, args.baseline)
        result = validate_contract(
            json.loads(args.contract.read_text(encoding="utf-8")),
            tables,
            args.contract.resolve().parent,
        )
        result["source_baseline_sha256"] = digest
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
