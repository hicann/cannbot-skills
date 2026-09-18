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
"""Shared configuration loading and initialization for the Issue Handler."""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from runtime_paths import (
        CLASSIFY_CONFIG,
        CONFIG_DIR,
        LEGACY_CLASSIFY_CONFIG,
        LEGACY_OPERATOR_OWNERS_CONFIG,
        OPERATOR_OWNERS_CONFIG,
    )
except ModuleNotFoundError:  # direct import through importlib.util in tests
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runtime_paths import (  # type: ignore[no-redef]
        CLASSIFY_CONFIG,
        CONFIG_DIR,
        LEGACY_CLASSIFY_CONFIG,
        LEGACY_OPERATOR_OWNERS_CONFIG,
        OPERATOR_OWNERS_CONFIG,
    )

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_TEMPLATE_NAMES = {"classify_config", "operator_owners"}
_RESPONSIBILITY_LEVELS = {"handle", "list-only", "ignore"}
_AUTOMATION_KEYS = ("auto-response", "auto-assign")


class ConfigError(ValueError):
    """Raised when a handler YAML document is malformed."""


def _require_yaml():
    if yaml is None:
        raise ConfigError("PyYAML is required; install it with: pip install pyyaml")
    return yaml


def _read_yaml(path: Path) -> dict[str, Any]:
    y = _require_yaml()
    try:
        raw = y.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, y.YAMLError) as exc:
        raise ConfigError(f"cannot read YAML config {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"YAML config {path} must contain a mapping")
    return raw


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if (
            isinstance(result.get(key), dict)
            and isinstance(value, dict)
            and value
        ):
            result[key] = _merge(result[key], value)
        else:
            # Empty mappings/lists are intentional overrides and must be kept.
            result[key] = value
    return result


def _validate_responsibility(document: dict[str, Any], source: Path) -> None:
    responsibility = document.get("responsibility")
    if responsibility is None:
        return
    if not isinstance(responsibility, dict):
        raise ConfigError(f"responsibility in {source} must be a mapping")
    unknown = set(responsibility) - _RESPONSIBILITY_LEVELS
    if unknown:
        raise ConfigError(
            "responsibility contains invalid level key(s): "
            + ", ".join(map(str, sorted(unknown)))
        )
    for level, conditions in responsibility.items():
        invalid_list = not isinstance(conditions, list)
        invalid_item = not invalid_list and any(
            not isinstance(condition, str) for condition in conditions
        )
        if invalid_list or invalid_item:
            raise ConfigError(
                f"responsibility.{level} must be a list of strings"
            )


def _validate_automation(
    document: dict[str, Any], source: Path, require_automation: bool,
) -> None:
    fallback = document.get("auto-assign-fallback-user")
    fallback_is_string = isinstance(fallback, str)
    fallback_has_valid_shape = fallback_is_string and bool(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", fallback)
    )
    fallback_invalid = fallback is not None and (
        not fallback_is_string
        or (fallback != "" and (not fallback_has_valid_shape or fallback.casefold() == "direct"))
    )
    if fallback_invalid:
        raise ConfigError("auto-assign-fallback-user must be empty or a GitCode login (not a URL or nickname)")
    for key in _AUTOMATION_KEYS:
        if key not in document:
            if require_automation:
                raise ConfigError(
                    f"{key} in {source} must be a boolean (true or false)"
                )
            continue
        if not isinstance(document.get(key), bool):
            raise ConfigError(
                f"{key} in {source} must be a boolean (true or false)"
            )
    automation_complete = "auto-assign" in document and "auto-response" in document
    assignment_without_response = (
        automation_complete
        and document["auto-assign"]
        and not document["auto-response"]
    )
    if assignment_without_response:
        raise ConfigError(
            "启用 auto-assign 时必须同时启用 auto-response："
            "auto-assign: true 需要 auto-response: true"
        )


def _validate(
    document: dict[str, Any], source: Path, *, validate_automation: bool = False,
    require_automation: bool = False,
) -> None:
    _validate_responsibility(document, source)
    if validate_automation:
        _validate_automation(document, source, require_automation)


def load_template(name: str = "classify_config") -> dict[str, Any]:
    """Load and validate a packaged ``assets/<name>.yaml.template``."""
    if name not in _TEMPLATE_NAMES:
        raise ConfigError(f"unknown handler config template: {name}")
    path = _ASSETS / f"{name}.yaml.template"
    if not path.exists():
        raise ConfigError(f"handler config template does not exist: {path}")
    document = _read_yaml(path)
    _validate(document, path, validate_automation=name == "classify_config",
              require_automation=name == "classify_config")
    return document


def load_handler_config(
    path: str | Path | None = None, *, allow_legacy: bool = True
) -> dict[str, Any]:
    """Return template defaults recursively overridden by repository config."""
    return load_named_config("classify_config", path, allow_legacy=allow_legacy)


def load_named_config(
    name: str, path: str | Path | None = None, *, allow_legacy: bool = True
) -> dict[str, Any]:
    """Load one named template with a canonical or legacy repository override."""
    defaults = load_template(name)
    if name == "operator_owners":
        canonical, legacy = OPERATOR_OWNERS_CONFIG, LEGACY_OPERATOR_OWNERS_CONFIG
    elif name == "classify_config":
        canonical, legacy = CLASSIFY_CONFIG, LEGACY_CLASSIFY_CONFIG
    else:
        raise ConfigError(f"unknown handler config template: {name}")
    if path is None:
        config_path = canonical if canonical.exists() else (
            legacy if allow_legacy and legacy.exists() else canonical
        )
        explicit = False
    else:
        config_path, explicit = Path(path), True
    if explicit and not config_path.exists():
        raise ConfigError(f"config file does not exist: {config_path}")
    if config_path.exists():
        user = _read_yaml(config_path)
        _validate(user, config_path, validate_automation=name == "classify_config")
        merged = _merge(defaults, user)
        _validate(merged, config_path, validate_automation=name == "classify_config",
                  require_automation=name == "classify_config")
        return merged
    return defaults


def get_automation_policy(config: dict[str, Any]) -> dict[str, bool]:
    """Return normalized automation flags from ``load_handler_config``.

    The YAML/API keys retain their documented hyphenated names.  Callers use
    this small normalized view so automation policy is read consistently:
    ``{"auto_response": bool, "auto_assign": bool}``.
    """
    if not isinstance(config, dict):
        raise ConfigError("handler config must be a mapping")
    try:
        auto_response = config["auto-response"]
        auto_assign = config["auto-assign"]
    except KeyError as exc:
        raise ConfigError(f"missing automation config key: {exc.args[0]}") from exc
    if not isinstance(auto_response, bool) or not isinstance(auto_assign, bool):
        raise ConfigError("automation config values must be booleans")
    if auto_assign and not auto_response:
        raise ConfigError(
            "启用 auto-assign 时必须同时启用 auto-response："
            "auto-assign: true 需要 auto-response: true"
        )
    return {"auto_response": auto_response, "auto_assign": auto_assign}


def _write_yaml(path: Path, document: dict[str, Any]) -> None:
    y = _require_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        y.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def initialize_config(repository_root: str | Path) -> dict[str, str]:
    """Instantiate both packaged templates under a repository's canonical config."""
    root = Path(repository_root)
    written: dict[str, str] = {}
    for name, canonical, legacy in (
        ("classify_config", CLASSIFY_CONFIG, LEGACY_CLASSIFY_CONFIG),
        ("operator_owners", OPERATOR_OWNERS_CONFIG, LEGACY_OPERATOR_OWNERS_CONFIG),
    ):
        destination = root / canonical
        # A dangling symlink is still an existing user choice; never replace it.
        if destination.exists() or destination.is_symlink():
            continue
        template = load_template(name)
        legacy_path = root / legacy
        if legacy_path.exists():
            legacy_doc = _read_yaml(legacy_path)
            _validate(legacy_doc, legacy_path)
            document = _merge(template, legacy_doc)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as stream:
                stream.write((_ASSETS / f"{name}.yaml.template").read_text(encoding="utf-8"))
            written[name] = str(destination)
            continue
        _write_yaml(destination, document)
        written[name] = str(destination)
    return written
