#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
VISIBLE_ESCAPE_RE = re.compile(r"␛\[[0-?]*[ -/]*[@-~]")

# Numeric extractor supporting comma separators, scientific notation, and
# non-finite values. Keep all inner groups non-capturing because callers read
# the numeric token from capture group 1 (and group 2 for memory percentage).
FINITE_NUM_RE = (
    r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
)
NONFINITE_NUM_RE = r"(?i:nan|inf(?:inity)?)"
NUM_RE = rf"([-+]?(?:{FINITE_NUM_RE}|{NONFINITE_NUM_RE}))"
NUM_END_RE = r"(?![\w.,])"

STEP_RE = re.compile(rf"\bstep:\s*{NUM_RE}{NUM_END_RE}")
# Callers mask AUX_LOSS_RE matches before searching this pattern, because
# a plain \bloss: also matches the "loss" in prefixed aux losses.
LOSS_RE = re.compile(rf"\bloss:\s*{NUM_RE}{NUM_END_RE}")
GRAD_NORM_RE = re.compile(rf"\bgrad_norm:\s*{NUM_RE}{NUM_END_RE}")
MEMORY_RE = re.compile(rf"\bmemory:\s*{NUM_RE}\s*GiB(?:\(\s*{NUM_RE}%\))?(?![\w.])")
TPS_RE = re.compile(rf"\btps:\s*{NUM_RE}{NUM_END_RE}")
TFLOPS_RE = re.compile(rf"\btflops:\s*{NUM_RE}{NUM_END_RE}")
MFU_RE = re.compile(rf"\bmfu:\s*{NUM_RE}%(?![\w.])")
ELAPSED_RE = re.compile(rf"\belapsed_time_per_step:\s*{NUM_RE}s(?![\w.])")
# torchtitan-npu prints aux losses on their own line.
AUX_STEP_RE = re.compile(rf"\baux_loss\s+step:\s*{NUM_RE}{NUM_END_RE}")
INDEXER_LOSS_RE = re.compile(rf"\bindexer[\s_]+loss(?:/mean)?:\s*{NUM_RE}{NUM_END_RE}")
# A loss field with a word prefix is an aux loss ("indexer loss:", "moe loss:",
# "load balancing loss:", ...), never the main loss. Require each prefix token
# to start with a letter so a preceding numeric value such as "1.0 loss:" is
# not swallowed.
AUX_LOSS_RE = re.compile(
    rf"\b[A-Za-z][A-Za-z0-9_]*(?:[\s_-]+[A-Za-z][A-Za-z0-9_]*)*"
    rf"[\s_-]+loss(?:/mean)?:\s*{NUM_RE}{NUM_END_RE}"
)


@dataclass(frozen=True)
class AlignedRecords:
    steps: list[int]
    records_a: list[dict]
    records_b: list[dict]
    missing_in_a: list[int]
    missing_in_b: list[int]


def _clean_line(line: str) -> str:
    line = ANSI_ESCAPE_RE.sub("", line)
    line = VISIBLE_ESCAPE_RE.sub("", line)
    return line


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def _search_float(pattern: re.Pattern[str], line: str) -> float | None:
    match = pattern.search(line)
    if not match:
        return None
    return _to_float(match.group(1))


def _is_valid_step(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value.is_integer()


def _search_memory(line: str) -> tuple[float | None, float | None]:
    match = MEMORY_RE.search(line)
    if not match:
        return None, None
    gib = _to_float(match.group(1))
    pct = _to_float(match.group(2)) if match.group(2) else None
    return gib, pct


def read_training_metrics(log_path: str | Path) -> tuple[list[dict], list[str]]:
    path = Path(log_path)
    if not path.exists():
        raise OSError(f"log file does not exist: {path}")

    warnings: list[str] = []
    records_by_step: dict[int, dict] = {}
    malformed_lines = 0
    duplicate_steps = 0
    pending_indexer_loss: float | None = None
    pending_elapsed: float | None = None
    indexer_by_step: dict[int, float] = {}
    last_record: dict | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = _clean_line(raw_line)

            indexer_match = INDEXER_LOSS_RE.search(line)
            line_indexer_loss = _to_float(indexer_match.group(1)) if indexer_match else None
            line = AUX_LOSS_RE.sub(
                lambda match: " " * (match.end() - match.start()), line
            )
            line_elapsed = _search_float(ELAPSED_RE, line)

            if "step:" not in line or "loss:" not in line:
                # Aux-loss and elapsed-time lines are printed separately from the
                # metrics line. The aux line carries its own step; the elapsed
                # line does not, so it belongs to the most recent metrics record.
                if line_indexer_loss is not None:
                    aux_step = _search_float(AUX_STEP_RE, line)
                    own_step = _search_float(STEP_RE, line)
                    if _is_valid_step(aux_step):
                        indexer_by_step[int(aux_step)] = line_indexer_loss
                    elif _is_valid_step(own_step):
                        # The aux-only line names its own step, so it must not
                        # be attached to the previous record.
                        indexer_by_step[int(own_step)] = line_indexer_loss
                    elif last_record is not None:
                        last_record.setdefault("indexer_loss", line_indexer_loss)
                    else:
                        pending_indexer_loss = line_indexer_loss
                if line_elapsed is not None:
                    if last_record is not None:
                        last_record.setdefault("elapsed_time_per_step", line_elapsed)
                    else:
                        pending_elapsed = line_elapsed
                continue

            step_val = _search_float(STEP_RE, line)
            loss_val = _search_float(LOSS_RE, line)
            if not _is_valid_step(step_val) or loss_val is None:
                malformed_lines += 1
                pending_indexer_loss = None
                pending_elapsed = None
                continue

            step = int(step_val)
            record: dict[str, float | int] = {
                "step": step,
                "loss": loss_val,
            }

            grad_norm = _search_float(GRAD_NORM_RE, line)
            if grad_norm is not None:
                record["grad_norm"] = grad_norm

            memory_gib, memory_pct = _search_memory(line)
            if memory_gib is not None:
                record["memory_gib"] = memory_gib
            if memory_pct is not None:
                record["memory_pct"] = memory_pct

            tps = _search_float(TPS_RE, line)
            if tps is not None:
                record["tps"] = tps

            tflops = _search_float(TFLOPS_RE, line)
            if tflops is not None:
                record["tflops"] = tflops

            mfu_pct = _search_float(MFU_RE, line)
            if mfu_pct is not None:
                record["mfu_pct"] = mfu_pct

            # elapsed_time_per_step may sit on this line or on the separate line
            # that follows it; the latter was captured into pending_elapsed.
            if line_elapsed is not None:
                record["elapsed_time_per_step"] = line_elapsed
            elif pending_elapsed is not None:
                record["elapsed_time_per_step"] = pending_elapsed
            pending_elapsed = None

            indexer_loss = line_indexer_loss
            if indexer_loss is None:
                indexer_loss = pending_indexer_loss
            if indexer_loss is not None:
                record["indexer_loss"] = indexer_loss
            pending_indexer_loss = None

            if step in records_by_step:
                duplicate_steps += 1
                warnings.append(f"duplicate step {step} at line {line_no}, keep latest record")

            records_by_step[step] = record
            last_record = record

    # Aux-loss lines that carried an explicit step are attributed after the fact,
    # so their position relative to the metrics line does not matter.
    for step, record in records_by_step.items():
        if "indexer_loss" not in record and step in indexer_by_step:
            record["indexer_loss"] = indexer_by_step[step]

    records = [records_by_step[s] for s in sorted(records_by_step)]
    if malformed_lines:
        warnings.append(f"ignored {malformed_lines} malformed metric lines")
    if duplicate_steps:
        warnings.append(f"found {duplicate_steps} duplicate steps")

    return records, warnings


def extract_metric_series(records: list[dict], metric_key: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    for record in records:
        value = record.get(metric_key)
        if value is None:
            continue
        steps.append(int(record["step"]))
        values.append(float(value))
    return steps, values


def align_by_common_steps(records_a: list[dict], records_b: list[dict]) -> AlignedRecords:
    map_a = {int(record["step"]): record for record in records_a}
    map_b = {int(record["step"]): record for record in records_b}

    steps_a = set(map_a)
    steps_b = set(map_b)
    common = sorted(steps_a & steps_b)
    missing_in_a = sorted(steps_b - steps_a)
    missing_in_b = sorted(steps_a - steps_b)

    return AlignedRecords(
        steps=common,
        records_a=[map_a[s] for s in common],
        records_b=[map_b[s] for s in common],
        missing_in_a=missing_in_a,
        missing_in_b=missing_in_b,
    )


def compute_errors(
    values_a: list[float], values_b: list[float], baseline: str = "a"
) -> tuple[list[float], list[float]]:
    """Return absolute errors and signed relative errors against the baseline.

    A relative error is NaN when it is not well-defined: a zero baseline with a
    non-zero difference, or a non-finite input. Callers must skip such points
    instead of treating them as a perfect match.
    """
    if len(values_a) != len(values_b):
        raise ValueError("values_a and values_b must have the same length")
    if baseline not in {"a", "b"}:
        raise ValueError("baseline must be 'a' or 'b'")

    absolute_errors: list[float] = []
    rel_errors: list[float] = []

    for a_val, b_val in zip(values_a, values_b, strict=True):
        if baseline == "a":
            diff = b_val - a_val
            denom = abs(a_val)
        else:
            diff = a_val - b_val
            denom = abs(b_val)
        absolute_errors.append(abs(diff))
        if not math.isfinite(diff) or not math.isfinite(denom):
            rel_errors.append(float("nan"))
        elif denom == 0.0:
            rel_errors.append(0.0 if diff == 0.0 else float("nan"))
        else:
            rel_errors.append(diff / denom)

    return absolute_errors, rel_errors


def summarize_records(records: list[dict]) -> dict[str, float | int]:
    if not records:
        return {
            "num_steps": 0,
            "first_step": 0,
            "last_step": 0,
            "first_loss": 0.0,
            "last_loss": 0.0,
            "first_grad_norm": 0.0,
            "last_grad_norm": 0.0,
        }

    first = records[0]
    last = records[-1]
    return {
        "num_steps": len(records),
        "first_step": int(first["step"]),
        "last_step": int(last["step"]),
        "first_loss": float(first.get("loss", 0.0)),
        "last_loss": float(last.get("loss", 0.0)),
        "first_grad_norm": float(first.get("grad_norm", 0.0)),
        "last_grad_norm": float(last.get("grad_norm", 0.0)),
    }
