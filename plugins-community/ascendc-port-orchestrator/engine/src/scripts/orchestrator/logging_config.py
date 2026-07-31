# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Centralized run-lifecycle logger for the orchestrator.

Adopts cv-agent's `TIMESTAMP | LEVEL | message` pattern (per
`cv_agent_workdir/run_task_with_agent.py:setup_logger`). Two handlers per
run:

  - FileHandler → `<workspace>/.opgen.log` (per-run, persistent)
  - StreamHandler → stdout (terminal)

Format: `"%(asctime)s | %(levelname)s | %(message)s"`
datefmt: `"%Y-%m-%d %H:%M:%S"`

This is **orthogonal to `orchestrator_events.jsonl`** (machine-readable
structured events). Both coexist:
  - JSONL = machine consumption (downstream analysis, dashboards)
  - logger = human run-lifecycle (timing, level, eyeball debugging)

cv-agent has the same split — don't merge them.

Why centralized (owner direction 2026-05-27 00:12Z): every module
calls `get_logger(__name__)`; `setup_run_logger(workspace)` is invoked
ONCE at orch entry so all child loggers inherit the same handlers.
Per-module setup duplicates handlers and produces interleaved log lines.

Usage:

    # In orch entry point (orchestrator.py main):
    from orchestrator.logging_config import setup_run_logger
    log = setup_run_logger(workspace, level="INFO")
    log.info("orchestrator: start op=%s mode=%s", op, mode)

    # In any other module:
    from orchestrator.logging_config import get_logger
    log = get_logger(__name__)
    log.info("phase O1.7: tags=%s", tags)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

# Canonical logger name; all submodules attach as children
# (e.g. "a5_orchestrator.phase_o17_classify"). Keeps cv-agent's structure.
ROOT_LOGGER_NAME = "a5_orchestrator"

# cv-agent format — verified field-for-field against
# cv_agent_workdir/run_task_with_agent.py:53-72.
_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_run_logger(
    workspace: Optional[Path] = None,
    *,
    level: Union[int, str] = logging.INFO,
    log_filename: str = ".opgen.log",
    add_stream_handler: bool = True,
) -> logging.Logger:
    """Configure the root orchestrator logger for one orch run.

    Idempotent: calling twice with the same workspace clears prior handlers
    (handlers don't accumulate across re-entry from the same process). The
    FIRST caller wins on level — re-setup with a different level only
    affects new handlers' threshold, not the logger's own (CRITICAL caveat
    for nested orch invocations; we explicitly set the logger level too).

    Args:
      workspace: directory to write `.opgen.log` into. If None, FileHandler
        is skipped (stdout-only mode — useful for unit tests + CLI
        sub-commands that don't have a workspace).
      level: logging level (int or string "DEBUG"/"INFO"/...).
      log_filename: filename within workspace. Default `.opgen.log` is
        gitignored via standard `.opgen_*` workspace hygiene.
      add_stream_handler: emit to stdout alongside the file. Default True
        — matches cv-agent + preserves current terminal-output UX.

    Returns:
      Configured root logger. Child loggers (`get_logger(__name__)`) inherit
      from it automatically.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    # Don't propagate to Python's root logger — avoids duplicate output if
    # the host process (e.g. pytest) has already configured logging.
    logger.propagate = False

    # Clear prior handlers — idempotent re-setup.
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    if workspace is not None:
        ws = Path(workspace)
        ws.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(ws / log_filename, mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if add_stream_handler:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the orchestrator root.

    Standard pattern: `log = get_logger(__name__)` at module top. Child
    propagates upward to the root configured by `setup_run_logger`; if
    setup hasn't run yet (e.g. unit test import time), output silently
    discards until the root has a handler — matches stdlib semantics.

    Accepts dotted `__name__` directly; we attach as child of
    ROOT_LOGGER_NAME so the hierarchy is `a5_orchestrator.phase_o17_classify`
    etc.
    """
    # Strip common path prefixes so the displayed name stays short.
    short = name
    for prefix in ("src.scripts.orchestrator.", "scripts.orchestrator.",
                   "orchestrator."):
        if short.startswith(prefix):
            short = short[len(prefix):]
            break
    if short == "__main__":
        short = "main"
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{short}")
