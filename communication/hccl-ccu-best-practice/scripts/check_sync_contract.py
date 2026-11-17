#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Check declared CCU channel resources against reviewed HCOMM implementation.

This checks per-receiver-table slot bounds and simultaneous signal collisions.
It does not parse algorithm C++, prove liveness, or validate reuse proofs.
"""

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

# These read-only tools must not create caches beside linked Skill sources.
sys.dont_write_bytecode = True

KNOWLEDGE = importlib.import_module("check_knowledge")
DEFAULT_BASELINE = KNOWLEDGE.DEFAULT_BASELINE
check_cards = KNOWLEDGE.check_cards


class SyncError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise SyncError(message)


def integer(value, where, minimum=0):
    require(
        type(value) is int and value >= minimum,
        f"{where}: expected integer >= {minimum}",
    )
    return value


def nonempty(value, where):
    require(
        isinstance(value, str) and bool(value.strip()),
        f"{where}: expected nonempty evidence/string",
    )


def read_limits(root, baseline=DEFAULT_BASELINE):
    root = Path(root).resolve()
    knowledge = check_cards({"hcomm": root}, None, baseline, only={"sync-limits"})
    changed = [
        entry["path"]
        for entry in knowledge["cards"][0]["files"]
        if entry["status"] != "MATCH"
    ]
    require(
        knowledge["cards"][0]["status"] == "MATCH",
        "UNVERIFIED sync-limits: review changed/missing files before updating baseline: "
        + ", ".join(changed),
    )
    header = root / "src/base_comm/resources/ccu/ccu_transport/ccu_transport_.h"
    text = header.read_text(encoding="utf-8")
    limits = {"mask_bits": 16}
    for symbol, name in (("INIT_CKE_NUM", "notify_slots"), ("INIT_XN_NUM", "xn_slots")):
        matches = re.findall(r"\b" + symbol + r"\s*=\s*(\d+)\s*;", text)
        require(len(matches) == 1, f"cannot extract {symbol}")
        limits[name] = int(matches[0])
    return limits, knowledge


def validate_contract(document, limits):
    require(
        isinstance(document, dict)
        and type(document.get("schema_version")) is int
        and document["schema_version"] == 1,
        "expected schema_version=1 object",
    )
    pools = document.get("pools")
    require(
        isinstance(pools, list) and bool(pools),
        "pools must enumerate each distinct receiver table/scenario",
    )
    pool_names = set()
    max_mask = (1 << limits["mask_bits"]) - 1
    for pool in pools:
        require(isinstance(pool, dict), "pool must be an object")
        name = pool.get("name")
        nonempty(name, "pool.name")
        require(name not in pool_names, f"duplicate pool {name}")
        pool_names.add(name)
        nonempty(pool.get("source_evidence"), f"{name}.source_evidence")
        xns = pool.get("xn_indices")
        require(isinstance(xns, list), f"{name}.xn_indices must be an array")
        for index in xns:
            require(
                integer(index, f"{name}.XN") < limits["xn_slots"],
                f"{name}: XN index {index} outside {limits['xn_slots']} slots",
            )
        require(len(set(xns)) == len(xns), f"{name}: duplicate XN indices")
        epochs = pool.get("epochs")
        require(
            isinstance(epochs, list) and bool(epochs), f"{name}.epochs must be nonempty"
        )
        historical = {}
        epoch_names = set()
        for epoch in epochs:
            require(isinstance(epoch, dict), f"{name}: epoch must be an object")
            epoch_name = epoch.get("name")
            nonempty(epoch_name, f"{name}.epoch.name")
            require(epoch_name not in epoch_names, f"{name}: duplicate epoch")
            epoch_names.add(epoch_name)
            signals = epoch.get("signals")
            require(isinstance(signals, list), f"{name}: signals must be an array")
            active = {}
            signal_names = set()
            for signal in signals:
                require(isinstance(signal, dict), f"{name}: signal must be an object")
                signal_name = signal.get("name")
                nonempty(signal_name, f"{name}.signal.name")
                require(
                    signal_name not in signal_names,
                    f"{name}: duplicate signal {signal_name}",
                )
                signal_names.add(signal_name)
                slot = integer(signal.get("slot"), f"{name}.{signal_name}.slot")
                require(
                    slot < limits["notify_slots"],
                    f"{name}: Notify slot {slot} outside actual {limits['notify_slots']} slots",
                )
                mask = integer(signal.get("mask"), f"{name}.{signal_name}.mask", 1)
                require(
                    mask <= max_mask, f"{name}: mask exceeds {limits['mask_bits']} bits"
                )
                require(
                    not (active.get(slot, 0) & mask),
                    f"{name}/{epoch_name}: overlapping live Notify bits at slot {slot}",
                )
                active[slot] = active.get(slot, 0) | mask
            if any(historical.get(slot, 0) & mask for slot, mask in active.items()):
                nonempty(
                    epoch.get("reuse_proof"),
                    f"{name}/{epoch_name}.reuse_proof (all consumers complete before new producers)",
                )
            for slot, mask in active.items():
                historical[slot] = historical.get(slot, 0) | mask
    return {
        "status": "consistent",
        "pools_checked": len(pools),
        "limits": limits,
        "scope": "declared bounds/collisions only; C++, liveness and reuse proofs require independent review",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--hcomm-root", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args(argv)
    try:
        limits, evidence = read_limits(args.hcomm_root, args.baseline)
        result = validate_contract(
            json.loads(args.contract.read_text(encoding="utf-8")), limits
        )
        result["source_evidence"] = {
            "baseline_sha256": evidence["baseline_sha256"],
            "card_id": "sync-limits",
            "status": "MATCH",
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
