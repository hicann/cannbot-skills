# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression tests for the legacy-converter -> okf.v1 write boundary."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
OKF_DIR = SCRIPTS_DIR / "okf"
sys.path.insert(0, str(OKF_DIR))

from migrate_cards_to_okf_v1 import (  # noqa: E402
    OKFCardWriteError,
    write_okf_v1_card,
)


LEGACY_CARD = """---
type: build_card
title: "Remote generated card"
description: "A deterministic converter fixture"
confidence: single_run
original_id: EC-1
tags: [ascendc, ec-1]
timestamp: '2026-07-10T00:00:00+08:00'
timestamp_inferred: true
---
## Fix

Keep the local truth.
"""


def _frontmatter(raw: str) -> str:
    return raw.split("---", 2)[1]


def test_every_card_converter_uses_the_v1_write_boundary() -> None:
    converters = sorted(OKF_DIR.glob("convert_*_to_okf.py"))
    converters.append(OKF_DIR / "migrate_ol_dispositions.py")
    assert {path.name for path in converters} == {
        "convert_cand_to_okf.py",
        "convert_docs_to_okf.py",
        "convert_family_to_okf.py",
        "convert_hardware_to_okf.py",
        "convert_ol_to_okf.py",
        "convert_patterns_to_okf.py",
        "migrate_ol_dispositions.py",
    }
    for converter in converters:
        source = converter.read_text(encoding="utf-8")
        assert "from migrate_cards_to_okf_v1 import write_okf_v1_card" in source
        assert source.count("write_okf_v1_card(") >= 1


def test_writer_canonicalizes_new_converter_card_to_okf_v1(tmp_path: Path) -> None:
    target = tmp_path / "card.md"

    assert write_okf_v1_card(target, LEGACY_CARD) == "created"

    raw = target.read_text(encoding="utf-8")
    fm = _frontmatter(raw)
    assert "schema_version: okf.v1" in fm
    assert "kind: implementation_trap" in fm
    assert "type: implementation_trap" in fm
    assert "source_family: curated" in fm
    assert "created_at: 2026-07-09T16:00:00Z" in fm
    assert "updated_at: 2026-07-09T16:00:00Z" in fm
    assert re.search(r"(?m)^timestamp:", fm) is None
    assert "Keep the local truth." in raw


def test_writer_refuses_to_replace_different_existing_okf_v1(tmp_path: Path) -> None:
    target = tmp_path / "card.md"
    write_okf_v1_card(target, LEGACY_CARD)
    local_v1 = target.read_text(encoding="utf-8") + "\nLOCAL-OKF-V1-EDIT\n"
    target.write_text(local_v1, encoding="utf-8")

    with pytest.raises(OKFCardWriteError, match="existing okf.v1"):
        write_okf_v1_card(target, LEGACY_CARD)

    assert target.read_text(encoding="utf-8") == local_v1


def test_writer_fails_closed_for_unconvertible_generated_card(tmp_path: Path) -> None:
    target = tmp_path / "card.md"
    unsupported = LEGACY_CARD.replace("type: build_card", "type: unknown_remote_schema")

    with pytest.raises(OKFCardWriteError, match="cannot produce okf.v1"):
        write_okf_v1_card(target, unsupported)

    assert not target.exists()


def test_family_converter_emits_v1_and_preserves_conflicting_local_card(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ERROR_CORRECTIONS.md"
    source.write_text(
        "# Error corrections\n\n"
        "### EC-1: Remote title\n\n"
        "- **Error**: compile failed\n"
        "- **Fix**: use the supported intrinsic\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "kb" / "okf" / "runbooks" / "field-notes" / "build"
    converter = OKF_DIR / "convert_family_to_okf.py"
    command = [
        sys.executable,
        str(converter),
        "--family",
        "EC",
        str(source),
        str(out_dir),
    ]

    first = subprocess.run(command, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    card = out_dir / "ec-1-remote-title.md"
    raw = card.read_text(encoding="utf-8")
    assert "schema_version: okf.v1" in _frontmatter(raw)
    assert re.search(r"(?m)^timestamp:", _frontmatter(raw)) is None

    local_v1 = raw + "\nLOCAL-OKF-V1-EDIT\n"
    card.write_text(local_v1, encoding="utf-8")
    second = subprocess.run(command, text=True, capture_output=True, check=False)

    assert second.returncode != 0
    assert "refusing to overwrite existing okf.v1" in second.stderr
    assert card.read_text(encoding="utf-8") == local_v1
