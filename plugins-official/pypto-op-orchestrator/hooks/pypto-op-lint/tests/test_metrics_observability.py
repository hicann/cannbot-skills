# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import importlib.util
import json
import os
from pathlib import Path

from .helpers import build_stateless_op_dir, load_lint_module, run_rule, run_stop


def test_rule_check_emits_metric_event(tmp_path: Path):
    mod = load_lint_module()
    mod.LOGS_DIR = str(tmp_path / "logs")
    mod.LOGS_EVENTS_FILE = str(tmp_path / "logs" / "lint_events.jsonl")

    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL30")
    assert finding.status == "PASS"

    events_path = Path(mod.LOGS_EVENTS_FILE)
    assert events_path.exists()
    lines = [json.loads(x) for x in events_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert any(e.get("event_type") == "rule_check" and e.get("rule_id") == "OL30" for e in lines)


def test_stop_block_output_contains_blocking_rules_and_fix_hints(tmp_path: Path, capfd):
    mod = load_lint_module()
    mod.LOGS_DIR = str(tmp_path / "logs")
    mod.LOGS_EVENTS_FILE = str(tmp_path / "logs" / "lint_events.jsonl")

    op_dir = build_stateless_op_dir(tmp_path, "demo")
    (op_dir / "SPEC.md").write_text("# SPEC only\n", encoding="utf-8")
    os.environ["PYPTO_OP_LINT_STRICT"] = "1"
    code = run_stop(mod, str(op_dir))
    assert code == 2

    captured = capfd.readouterr().out
    assert "blocking_rules" in captured
    assert "fix_hints" in captured


def test_metrics_report_script_generates_summary(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "report_metrics.py"
    events = tmp_path / "lint_events.jsonl"
    events.write_text(
        "\n".join([
            json.dumps({"event_type": "rule_check", "rule_id": "OL30", "status": "PASS", "duration_ms": 1.2}),
            json.dumps({"event_type": "rule_check", "rule_id": "OL34", "status": "FAIL", "duration_ms": 2.5}),
            json.dumps({"event_type": "gate_decision", "blocked": True}),
        ]) + "\n",
        encoding="utf-8",
    )

    spec = importlib.util.spec_from_file_location("report_metrics", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.BASE = tmp_path
    mod.EVENTS = events
    mod.OUT_JSON = tmp_path / "summary_latest.json"
    mod.OUT_MD = tmp_path / "summary_latest.md"
    summary = mod.build_summary(mod.load_events())
    mod.write_outputs(summary)

    assert mod.OUT_JSON.exists()
    data = json.loads(mod.OUT_JSON.read_text(encoding="utf-8"))
    assert data["total_rule_checks"] == 2
    assert data["total_gate_blocks"] == 1


def _load_report_mod():
    script = Path(__file__).resolve().parents[1] / "report_metrics.py"
    spec = importlib.util.spec_from_file_location("report_metrics", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_uses_summary_events_when_available(tmp_path: Path):
    mod = _load_report_mod()
    events = [
        {"event_type": "rule_check", "rule_id": "OL01", "status": "PASS", "duration_ms": 1.0},
        {"event_type": "rule_check", "rule_id": "OL02", "status": "FAIL", "duration_ms": 2.0},
        {"event_type": "run_check_summary", "invocation_id": "abc123",
         "total": 2, "pass": 1, "fail": 1, "warn": 0, "skip": 0,
         "s0_fails": ["OL01"], "s1_fails": [], "total_duration_ms": 3.0,
         "op_dir": "custom/op1", "stage": 5, "mode": "cli"},
    ]
    summary = mod.build_summary(events)
    assert summary["data_source"] == "run_check_summary"
    assert summary["total_rule_checks"] == 2
    assert summary["s0_fails"] == ["OL01"]
    assert summary["total_duration_ms"] == 3.0


def test_report_effort_stats_aggregation(tmp_path: Path):
    mod = _load_report_mod()
    events = [
        {"event_type": "run_check_summary", "invocation_id": "a1",
         "total": 3, "pass": 1, "fail": 1, "warn": 1, "skip": 0,
         "s0_fails": [], "s1_fails": [], "total_duration_ms": 1.0,
         "effort_stats": {"E1": {"fails": ["OL01"], "warns": ["OL23"]},
                          "E2": {"fails": [], "warns": []}}},
        {"event_type": "run_check_summary", "invocation_id": "a2",
         "total": 3, "pass": 1, "fail": 1, "warn": 1, "skip": 0,
         "s0_fails": [], "s1_fails": [], "total_duration_ms": 2.0,
         "effort_stats": {"E1": {"fails": ["OL07"], "warns": []},
                          "E2": {"fails": ["OL09"], "warns": []}}},
    ]
    summary = mod.build_summary(events)
    es = summary["effort_stats"]
    assert es["E1"]["fails"] == ["OL01", "OL07"]
    assert es["E1"]["warns"] == ["OL23"]
    assert es["E2"]["fails"] == ["OL09"]


def test_report_invocation_breakdown(tmp_path: Path):
    mod = _load_report_mod()
    events = [
        {"event_type": "run_check_summary", "invocation_id": "inv1",
         "total": 5, "pass": 3, "fail": 2, "warn": 0, "skip": 0,
         "s0_fails": ["OL01"], "s1_fails": [], "total_duration_ms": 10.0,
         "op_dir": "custom/op1", "stage": 5, "mode": "cli"},
        {"event_type": "run_check_summary", "invocation_id": "inv2",
         "total": 5, "pass": 5, "fail": 0, "warn": 0, "skip": 0,
         "s0_fails": [], "s1_fails": [], "total_duration_ms": 8.0,
         "op_dir": "custom/op1", "stage": 5, "mode": "post-edit"},
    ]
    summary = mod.build_summary(events)
    invs = summary["invocations"]
    assert len(invs) == 2
    assert invs[0]["invocation_id"] == "inv1"
    assert invs[0]["fail"] == 2
    assert invs[1]["invocation_id"] == "inv2"
    assert invs[1]["fail"] == 0


def test_report_malformed_jsonl_skipped(tmp_path: Path):
    mod = _load_report_mod()
    events_file = tmp_path / "lint_events.jsonl"
    events_file.write_text(
        '{"event_type": "rule_check", "rule_id": "OL01", "status": "PASS"}\n'
        'THIS IS NOT JSON\n'
        '{"event_type": "rule_check", "rule_id": "OL02", "status": "FAIL"}\n',
        encoding="utf-8",
    )
    events = mod.load_events(path=events_file)
    assert len(events) == 2


def test_report_op_dir_filter(tmp_path: Path):
    mod = _load_report_mod()
    events_file = tmp_path / "lint_events.jsonl"
    pass_event = {
        "event_type": "rule_check", "op_dir": "custom/op1",
        "rule_id": "OL01", "status": "PASS",
    }
    fail_event = {
        "event_type": "rule_check", "op_dir": "custom/op2",
        "rule_id": "OL01", "status": "FAIL",
    }
    events_file.write_text(
        json.dumps(pass_event) + "\n" + json.dumps(fail_event) + "\n",
        encoding="utf-8",
    )
    events = mod.load_events(path=events_file, op_dir_filter="custom/op1")
    assert len(events) == 1
    assert events[0]["op_dir"] == "custom/op1"


def test_report_since_filter(tmp_path: Path):
    mod = _load_report_mod()
    events_file = tmp_path / "lint_events.jsonl"
    events_file.write_text(
        json.dumps({"event_type": "rule_check", "ts": 100, "rule_id": "OL01", "status": "PASS"}) + "\n"
        + json.dumps({"event_type": "rule_check", "ts": 200, "rule_id": "OL01", "status": "FAIL"}) + "\n"
        + json.dumps({"event_type": "rule_check", "ts": 300, "rule_id": "OL01", "status": "PASS"}) + "\n",
        encoding="utf-8",
    )
    events = mod.load_events(path=events_file, since_ts=200)
    assert len(events) == 2


def test_report_per_op_breakdown(tmp_path: Path):
    mod = _load_report_mod()
    events = [
        {"event_type": "run_check_summary", "invocation_id": "a1",
         "total": 3, "pass": 2, "fail": 1, "warn": 0, "skip": 0,
         "s0_fails": [], "s1_fails": [], "total_duration_ms": 1.0,
         "op_dir": "custom/op1", "stage": 5, "mode": "cli"},
        {"event_type": "run_check_summary", "invocation_id": "a2",
         "total": 3, "pass": 3, "fail": 0, "warn": 0, "skip": 0,
         "s0_fails": [], "s1_fails": [], "total_duration_ms": 1.0,
         "op_dir": "custom/op2", "stage": 5, "mode": "cli"},
    ]
    summary = mod.build_summary(events)
    per_op = summary["per_op"]
    assert "custom/op1" in per_op
    assert per_op["custom/op1"]["fail"] == 1
    assert per_op["custom/op2"]["fail"] == 0


def test_report_md_output_has_tables(tmp_path: Path):
    mod = _load_report_mod()
    events = [
        {"event_type": "run_check_summary", "invocation_id": "inv1",
         "total": 3, "pass": 1, "fail": 1, "warn": 1, "skip": 0,
         "s0_fails": ["OL01"], "s1_fails": [], "total_duration_ms": 5.0,
         "effort_stats": {"E1": {"fails": ["OL01"], "warns": ["OL23"]}},
         "op_dir": "custom/op1", "stage": 5, "mode": "cli"},
    ]
    summary = mod.build_summary(events)
    out_md = tmp_path / "summary.md"
    mod.write_outputs(summary, out_md=out_md, out_json=tmp_path / "summary.json")
    md_text = out_md.read_text(encoding="utf-8")
    assert "| 指标 | 值 |" in md_text
    assert "## 修复成本分布" in md_text
    assert "## 每次调用明细" in md_text
    assert "OL01" in md_text
