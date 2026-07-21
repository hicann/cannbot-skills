# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Interpreter affinity for commands emitted by parse_args.py."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "autoresearch_parse_args_interpreter_test",
    ROOT / "scripts" / "engine" / "parse_args.py",
)
PARSE_ARGS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARSE_ARGS)


def _dispatch(monkeypatch, *tokens: str) -> dict:
    payload = {}

    def capture(record: dict) -> int:
        payload.update(record)
        return 0

    monkeypatch.setattr(PARSE_ARGS, "_emit", capture)
    monkeypatch.setattr(
        PARSE_ARGS.sys,
        "argv",
        ["parse_args.py", *tokens],
    )
    PARSE_ARGS.main()
    return payload


@pytest.mark.level0
def test_scaffold_argv_reuses_current_interpreter(monkeypatch):
    payload = _dispatch(
        monkeypatch,
        "--ref",
        "reference.py",
        "--kernel",
        "kernel.py",
        "--op-name",
        "add",
        "--devices",
        "0",
    )
    argv = payload.get("argv")
    assert argv is not None
    assert argv[:2] == [PARSE_ARGS.sys.executable, "scripts/scaffold.py"]


@pytest.mark.level0
def test_resume_argv_reuses_current_interpreter(monkeypatch):
    payload = _dispatch(monkeypatch, "--resume", "task-dir")
    assert payload.get("argv") == [
        PARSE_ARGS.sys.executable,
        "scripts/resume.py",
        "task-dir",
    ]
