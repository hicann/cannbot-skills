#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Resolve example-routes.yaml — accept spec.yaml, extract paradigms and language,
map to language-adapter reference example packages."""

from __future__ import annotations

import logging
import sys
import argparse
from pathlib import Path

import yaml

DEFAULT_LANGUAGE = "ascendc"
SUPPORTED_LANGUAGES = {"ascendc", "dsl"}


LOGGER = logging.getLogger(__name__)


def load_example_routes(skill_dir: Path) -> dict:
    routes_file = skill_dir / "references" / "paradigms" / "example-routes.yaml"
    if not routes_file.exists():
        return {}
    with open(routes_file) as f:
        data = yaml.safe_load(f) or {}
    return data.get("routes", {})


def parse_spec(spec_text: str) -> dict:
    """读取完整 YAML，并校验本模块消费的字段。"""
    data = yaml.safe_load(spec_text)
    if not isinstance(data, dict) or not isinstance(data.get("op"), dict):
        raise ValueError("spec.op must be a mapping")
    op = data["op"]
    category = op.get("category", "")
    if not isinstance(category, str):
        raise ValueError("op.category must be a string")
    paradigms = op.get("paradigms", [])
    if not isinstance(paradigms, list) or any(
        not isinstance(item, str) or not item.strip() for item in paradigms
    ):
        raise ValueError("op.paradigms must be a list of non-empty strings")
    language = op.get("language", "")
    if not isinstance(language, str):
        raise ValueError("op.language must be a string")
    if language and language not in SUPPORTED_LANGUAGES:
        raise ValueError("op.language must be ascendc or dsl")
    return op


def extract_paradigm_info(spec_text: str) -> tuple[str, list[str]]:
    op = parse_spec(spec_text)
    return op.get("category", ""), op.get("paradigms", [])


def extract_language(spec_text: str) -> str:
    return parse_spec(spec_text).get("language", "")


def _example_paths_for_language(entry: object, language: str) -> list[str]:
    """Select example paths for one language from a paradigm route entry.

    New format: {language: [paths]}; legacy flat format: [paths] (ascendc only).
    No cross-language fallback: an unregistered language resolves to no examples.
    """
    if isinstance(entry, dict):
        return list(entry.get(language, []))
    if isinstance(entry, list):
        return list(entry) if language == DEFAULT_LANGUAGE else []
    return []


def resolve_examples(
    routes: dict,
    paradigms: list[str],
    skill_dir: Path,
    language: str = "",
) -> dict[str, list[str]]:
    para_routes = routes.get("paradigms", {})
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError("an explicit supported language (ascendc or dsl) is required")
    lang = language
    result: dict[str, list[str]] = {}
    seen: set[str] = set()

    for p in paradigms:
        entry = para_routes.get(p)
        if entry is None:
            continue
        for ref_path in _example_paths_for_language(entry, lang):
            full = skill_dir / ref_path
            if full.exists():
                if str(full) not in seen:
                    result.setdefault(p, []).append(str(full))
                    seen.add(str(full))
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(
        description="Resolve paradigms to example packages from example-routes.yaml"
    )
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="调用方提供的算子规格 YAML 路径",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--language",
        choices=sorted(SUPPORTED_LANGUAGES),
        help="explicit language; overrides op.language in the spec",
    )
    args = parser.parse_args()

    skill_dir = Path(__file__).resolve().parents[2]
    try:
        op = parse_spec(args.spec.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    paradigms = op.get("paradigms", [])
    language = args.language or op.get("language", "")
    if language not in SUPPORTED_LANGUAGES:
        parser.error("provide --language ascendc|dsl or a supported op.language")

    routes = load_example_routes(skill_dir)
    resolved = resolve_examples(routes, paradigms, skill_dir, language)

    if args.json:
        import json

        print(
            json.dumps(
                {"language": language, "examples": resolved},
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        LOGGER.info(f"language: {language}")
        if not resolved:
            LOGGER.info(
                "  (no example matched; language adapters are mutually exclusive, no cross-language fallback)"
            )
        for para, paths in resolved.items():
            LOGGER.info(f"[{para}]")
            for p in paths:
                LOGGER.info(f"  {p}")


if __name__ == "__main__":
    main()
