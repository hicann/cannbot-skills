# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Centralized logger setup (cv-agent style TIMESTAMP | LEVEL | message).

Owner via independent review 2026-05-27 00:12Z: adopt cv-agent's logging pattern.
Pins the format + handler shape so future drift gets caught at CI.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from logging_config import (  # noqa: E402
    ROOT_LOGGER_NAME,
    setup_run_logger,
    get_logger,
)


def _teardown_handlers():
    """Reset root logger between tests so handlers don't accumulate."""
    lg = logging.getLogger(ROOT_LOGGER_NAME)
    for h in list(lg.handlers):
        lg.removeHandler(h)
        try:
            h.close()
        except Exception:
            lg.debug("Ignoring handler-close failure during test cleanup", exc_info=True)


@pytest.fixture(autouse=True)
def _isolate():
    yield
    _teardown_handlers()


def test_setup_attaches_stream_handler_by_default():
    log = setup_run_logger(workspace=None)
    assert log.name == ROOT_LOGGER_NAME
    assert any(isinstance(h, logging.StreamHandler) for h in log.handlers)
    assert log.level == logging.INFO


def test_setup_attaches_file_handler_when_workspace_given(tmp_path: Path):
    log = setup_run_logger(workspace=tmp_path, level="DEBUG")
    fh = next((h for h in log.handlers if isinstance(h, logging.FileHandler)), None)
    assert fh is not None, "FileHandler missing when workspace provided"
    assert Path(fh.baseFilename) == tmp_path / ".opgen.log"
    assert log.level == logging.DEBUG


def test_setup_is_idempotent(tmp_path: Path):
    """Re-setup must clear prior handlers (idempotent re-entry from
    main() → run_single_op once workspace resolves).
    """
    log = setup_run_logger(workspace=None)
    n_first = len(log.handlers)
    setup_run_logger(workspace=tmp_path)
    n_second = len(log.handlers)
    # Re-setup should result in (FileHandler + StreamHandler) = 2, not 4.
    assert n_second == 2, (
        f"handlers accumulated across re-setup: first={n_first} second={n_second}"
    )


def test_log_format_matches_cv_agent_style(tmp_path: Path, capsys):
    """Verify on-the-wire format is `TIMESTAMP | LEVEL | message`."""
    setup_run_logger(workspace=tmp_path)
    log = get_logger("test_module")
    log.info("hello world")
    log.warning("careful now")

    log_file = tmp_path / ".opgen.log"
    assert log_file.is_file()
    content = log_file.read_text()
    # Pattern: 2026-05-27 00:12:47 | INFO | hello world
    info_re = re.compile(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| INFO \| hello world$",
        re.M,
    )
    warn_re = re.compile(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| WARNING \| careful now$",
        re.M,
    )
    assert info_re.search(content), f"INFO line format wrong; got:\n{content}"
    assert warn_re.search(content), f"WARNING line format wrong; got:\n{content}"


def test_get_logger_short_name():
    """get_logger strips long path prefixes for readable hierarchy."""
    log = get_logger("src.scripts.orchestrator.phase_o17_classify")
    assert log.name == "a5_orchestrator.phase_o17_classify"
    log2 = get_logger("orchestrator.briefs.kw_brief")
    assert log2.name == "a5_orchestrator.briefs.kw_brief"
    log3 = get_logger("__main__")
    assert log3.name == "a5_orchestrator.main"


def test_no_propagation_to_python_root():
    """Logger MUST NOT propagate to Python root — avoids duplicate output
    if pytest or another framework has already configured root logging.
    """
    log = setup_run_logger(workspace=None)
    assert log.propagate is False


def test_level_string_accepted():
    """Level can be passed as 'INFO' / 'DEBUG' string or stdlib int."""
    log = setup_run_logger(workspace=None, level="DEBUG")
    assert log.level == logging.DEBUG
    log = setup_run_logger(workspace=None, level="WARNING")
    assert log.level == logging.WARNING
    log = setup_run_logger(workspace=None, level=logging.ERROR)
    assert log.level == logging.ERROR
