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

"""Shared input loading and operator artifact paths for generator scripts."""

from dataclasses import dataclass
from pathlib import Path

import yaml


def load_yaml_mapping(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


@dataclass(frozen=True)
class ArtifactPaths:
    spec: Path
    test: Path
    design: Path | None = None
    units: Path | None = None


def add_io_arguments(parser, *, design=False, units=False):
    parser.add_argument(
        "--spec", type=Path, required=True, help="operator specification YAML"
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="directory for this operator's test assets",
    )
    if design:
        parser.add_argument(
            "--design", type=Path, required=True, help="operator design Markdown"
        )
    if units:
        parser.add_argument(
            "--units-dir", type=Path, required=True, help="directory containing U*.yaml"
        )


def artifact_paths(args) -> ArtifactPaths:
    return ArtifactPaths(
        spec=args.spec.resolve(),
        test=args.test_dir.resolve(),
        design=getattr(args, "design", None).resolve()
        if getattr(args, "design", None)
        else None,
        units=getattr(args, "units_dir", None).resolve()
        if getattr(args, "units_dir", None)
        else None,
    )


def spec_path(paths: ArtifactPaths) -> Path:
    return paths.spec


def design_path(paths: ArtifactPaths) -> Path:
    if paths.design is None:
        raise ValueError("--design is required")
    return paths.design


def units_dir(paths: ArtifactPaths) -> Path:
    if paths.units is None:
        raise ValueError("--units-dir is required")
    return paths.units


def test_dir(paths: ArtifactPaths, op: str) -> Path:
    return paths.test


def golden_path(paths: ArtifactPaths, op: str) -> Path:
    return test_dir(paths, op) / f"{op}_golden.py"
