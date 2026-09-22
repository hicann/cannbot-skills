# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Small, shared capability checks for the A5 target.

Ascend910 is useful for lightweight smoke checks, but it is not the A5/arch35
acceptance target used by the direct-launch NPUKernelBench validation route.
Keep this policy separate from the legacy ACLNN port path: callers opt in by
using the direct-launch provider and passing the resolved A5 environment.
"""
from __future__ import annotations

import re
from typing import Mapping

# Target → env-key prefix map: single source in briefs/target_env_map.py.
# briefs/__init__.py is empty and the map module is stdlib-only, so this does
# NOT pull the brief-templating stack — same pattern as agent_dispatch.py,
# which imports briefs._common at module level.
from briefs.target_env_map import TARGET_ENV_PREFIXES


LIMITED_A5_SOC_MARKER = "A5_SOC_UNSUPPORTED_FOR_VALIDATION"

# The A5 gate must not infer support from a prefix such as ``Ascend950``.
# Keep the accepted product families explicit and treat every other value as
# unsupported.  The 910 expression intentionally accepts the product suffixes
# used by CANN (A/B/C, B2C/B3/B4, V220, numeric SKU forms, and separators).
_SOC_TOKEN_RE = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$")
_ASCEND_910_RE = re.compile(
    r"^(?:ascend)?910(?:(?:[abc][a-z0-9]*)|(?:[_-][a-z0-9]+))*$"
)
_ASCEND_310P_RE = re.compile(r"^(?:ascend)?310p[a-z0-9]*$")
_NPU_SMI_DEVICE_RE = re.compile(
    r"^\|\s*(\d+)\s+\|?\s*([^|:]*[A-Za-z][^|:]*)\|"
)
_SUPPORTED_A5_SOCS = frozenset(
    {
        "950",
        "ascend950",
        "950pr",
        "ascend950pr",
        "950pr_9579",
        "ascend950pr_9579",
        "950pr_9589",
        "ascend950pr_9589",
        "950pr_957b",
        "ascend950pr_957b",
        # Keep the documented name-boundary regression case valid: the 9107x
        # suffix belongs to the Ascend950PR product, not Ascend910.
        "950pr_9107x",
        "ascend950pr_9107x",
        "950dt",
        "ascend950dt",
        # 950DT boards carry a numeric SKU in npu-smi and CANN platform_config
        # (e.g. Ascend950DT_9582.ini); accept the verified full SKU name at
        # the same granularity as the 950PR entries above.
        "950dt_9582",
        "ascend950dt_9582",
        "950dt_superpod384",
        "ascend950dt_superpod384",
    }
)


def _normalized_soc(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized or _SOC_TOKEN_RE.fullmatch(normalized) is None:
        return None
    return normalized


def a5_soc_version(env: Mapping[str, object]) -> str:
    """Resolve the explicit A5 SoC value without a permissive default.

    If a target-specific key is present but empty or malformed, do not fall
    back to a generic value that could describe another target.  The direct
    launch target must provide ``A5_SOC_VERSION`` explicitly; an absent key is
    therefore also treated as unset.
    """
    value = env.get("A5_SOC_VERSION")
    if isinstance(value, str):
        return value.strip()
    return ""


def is_known_910_soc(value: object) -> bool:
    """Return whether *value* names a recognized Ascend910 product family."""
    normalized = _normalized_soc(value)
    return normalized is not None and _ASCEND_910_RE.fullmatch(normalized) is not None


def is_supported_a5_soc(value: object) -> bool:
    """Return whether *value* is an explicitly recognized A5 validation SoC."""
    normalized = _normalized_soc(value)
    return normalized in _SUPPORTED_A5_SOCS if normalized is not None else False


def soc_product_family(value: object) -> str | None:
    """Return the hardware product family for a recognized SoC spelling."""
    if is_known_910_soc(value):
        return "ascend910"
    if is_supported_a5_soc(value):
        return "ascend950"
    # 310p: covers the configured spelling (Ascend310P3) AND the npu-smi model
    # token (plain "310P3" — parse_npu_smi_soc returns the first whitespace
    # token of the Name column, which on the 6x310P3 box is "310P3").
    if is_supported_310p_soc(value):
        return "ascend310p"
    return None


def parse_npu_smi_soc(output: str, device_id: int) -> str:
    """Extract one device's model from the two supported ``npu-smi`` layouts.

    The parser is deliberately narrow: it only accepts the device summary row
    for the requested numeric device and ignores process rows and bus-id rows.
    An empty result is an unusable hardware identity and must be handled by
    the caller as a closed gate.
    """
    for line in str(output).splitlines():
        match = _NPU_SMI_DEVICE_RE.match(line)
        if match is None or int(match.group(1)) != int(device_id):
            continue
        name = match.group(2).strip().split()
        if name:
            return name[0]
    return ""


def is_known_a5_soc(value: object) -> bool:
    """Return whether *value* is known to the direct-launch capability policy."""
    return is_known_910_soc(value) or is_supported_a5_soc(value)


# ---------------------------------------------------------------------------
# Target-parametric capability surface (310p destination).
#
# The functions above stay A5-shaped (byte-identical) because ~10 call sites
# and their tests depend on them. The surface below lets the SAME gates serve
# the 310p port destination: soc reads go through the target's env prefix
# (ASCEND310P_SOC_VERSION, never a digit-led `310P_*` key — DEBT-336 class),
# and "limited" (compile-only, no acceptance validation) is decided per
# target. 310P is a FULL on-device validation destination per the Phase-1
# decision (D1=A): Ascend310P3 is accepted for precision/determinism/perf
# acceptance, NOT limited the way Ascend910 is on the a5 gate.
# ---------------------------------------------------------------------------


def _target_id(value: object) -> str:
    """Normalize a target spelling to the canonical lowercase id ("310p").

    Accepts either the target id ("310p") or the env-key prefix
    ("ASCEND310P" — what `_Target.name` carries after `_normalise_target`)
    so call sites can pass whichever they hold. Unknown spellings pass
    through lowercased; the caller's known/limited gates reject them.
    """
    lowered = (value or "").strip().lower() if isinstance(value, str) else ""
    if lowered in TARGET_ENV_PREFIXES:
        return lowered
    for _name, _prefix in TARGET_ENV_PREFIXES.items():
        if _prefix.lower() == lowered:
            return _name
    return lowered


# Canonical arch-generation token per DESTINATION target (a5_ops KB naming:
# AICORE 220→"arch22", 3510→"arch35", 200/DAV_2002/dav-m200→"arch20").
# Destination-derived only — a3/a2 are port SOURCES and never appear as a
# written state's target_arch.  ``arch_for_target`` keeps unknown/legacy
# spellings byte-identical with the historical "arch35" metadata, so only
# 310p destinations change what gets persisted.
ARCH_BY_TARGET = {
    "a5": "arch35",
    "310p": "arch20",
}


def arch_for_target(target: object, default: str = "arch35") -> str:
    """Return the canonical arch token for a destination target spelling.

    Accepts id or env prefix (same spellings as ``_target_id``); a trailing
    "-ds" lane suffix is stripped the same way briefs/_common archive_project
    normalizes it.
    """
    tid = _target_id(target)
    if tid.endswith("-ds"):
        tid = tid[:-3]
    return ARCH_BY_TARGET.get(tid, default)


def is_destination_target_name(value: object) -> bool:
    """True when ``value`` names a DESTINATION target as ``_Target.name`` spells it.

    Receipt/evidence identity gates compare endpoint names in the env-prefix
    spelling (``"A5"``, ``"ASCEND310P"``).  a3/a2 are port SOURCES and can
    never be the endpoint that produced evaluation evidence, so the accepted
    set is exactly the ``ARCH_BY_TARGET`` destination keys — not one
    hard-coded chip (interleave_rope: a PASS 310P target receipt
    was rejected as "identity is invalid" because the gate only knew "A5").
    """
    return _target_id(value) in ARCH_BY_TARGET


def is_supported_310p_soc(value: object) -> bool:
    """Return whether *value* names the Ascend310P family (310P / 310P3 / ...).

    Fail-closed like the A5 gate: no prefix-inference from ``Ascend`` — only
    the 310P product family is accepted, everything else (including empty and
    malformed) is not a 310P validation SoC.
    """
    normalized = _normalized_soc(value)
    return normalized is not None and _ASCEND_310P_RE.fullmatch(normalized) is not None


# ``npu-smi info -m`` mapping-table row (multi-chip npu-smi generations, e.g.
# 26.2.rc1 on the 3x2 310P3 box): three integer id columns (NPU/card id,
# chip id, Chip Logic ID) then the chip name.  The Chip Logic ID IS the
# ASCEND_RT_VISIBLE_DEVICES index — the device id the direct-launch SoC probe
# holds — so on cards that carry multiple chips this table is the only
# npu-smi surface that still maps a logic device id to its SoC model
# (``-t board -i <logic id>`` is rejected there because -i means CARD id,
# and the per-card board listing carries no per-chip model at all).
_NPU_SMI_MAPPING_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\S.*?)\s*$"
)


def parse_npu_smi_mapping_soc(output: str, device_id: int) -> str:
    """Extract one logic device's chip model from an ``npu-smi info -m`` table.

    Rows are whitespace-separated: NPU ID, Chip ID, Chip Logic ID, Chip Name
    (e.g. ``0  1  1  Ascend 310P3``; management ``Mcu`` rows carry ``-`` as
    their logic id and never match).  The chip name's internal whitespace is
    collapsed ("Ascend 310P3" → "Ascend310P3") so the family regexes accept
    the same spellings as the board-listing parser ("Ascend310P3").  An empty
    result is an unusable identity and must be handled fail-closed by the
    caller, exactly like :func:`parse_npu_smi_soc`.
    """
    for line in str(output).splitlines():
        match = _NPU_SMI_MAPPING_RE.match(line)
        if match is None or int(match.group(3)) != int(device_id):
            continue
        name = match.group(4).split()
        if name:
            return "".join(name)
    return ""


def soc_version_for_target(env: Mapping[str, object], target: str) -> str:
    """Resolve the explicit SoC value for the ACTIVE target, no permissive default.

    a5 keeps the legacy ``a5_soc_version`` semantics; 310p reads
    ``ASCEND310P_SOC_VERSION``. Missing/empty stays empty — the caller's
    limited/known gates fail closed on it (never inherit another chip's SoC).
    """
    prefix = TARGET_ENV_PREFIXES.get(_target_id(target))
    if prefix is None:
        # Unknown target: return the raw generic value so existing callers'
        # error paths (is_known_* checks) reject it loudly downstream.
        value = env.get("SOC_VERSION")
        return value.strip() if isinstance(value, str) else ""
    value = env.get(f"{prefix}_SOC_VERSION")
    if value is None and prefix == "A5":
        value = env.get("SOC_VERSION")
    return value.strip() if isinstance(value, str) else ""


def is_limited_soc(value: object, target: str = "a5") -> bool:
    """Target-parametric "compile-only" capability stop.

    a5: the legacy allowlist semantics (only explicitly recognized Ascend950
    SoCs open validation; 910/unknown stay limited).
    310p: only the Ascend310P family opens validation; anything else is limited.
    """
    if _target_id(target) == "310p":
        return not is_supported_310p_soc(value)
    return is_limited_a5_soc(value)


def is_known_soc_for_target(value: object, target: str) -> bool:
    """Target-parametric "recognized at all" check (limited + known-or-limited).

    310p: the Ascend310P family. a5: the legacy 910/950 known set. Unknown
    targets fall back to the a5 set — callers treat them as limited anyway.
    """
    if _target_id(target) == "310p":
        return is_supported_310p_soc(value)
    return is_known_a5_soc(value)


def limited_target_validation_error(soc: str, target: str = "a5") -> str:
    """Target-parametric terminal error for a limited SoC at the O5 gate."""
    display = soc if isinstance(soc, str) and soc else "<unset>"
    if _target_id(target) == "310p":
        return (
            "310P_SOC_UNSUPPORTED_FOR_VALIDATION: direct-launch 310p validation "
            "requires an explicitly recognized Ascend310P-family SoC (e.g. "
            "Ascend310P3); configured SoC %s is missing, unknown, or malformed" % display
        )
    return limited_a5_validation_error(soc)


def limited_target_warning(soc: str, target: str = "a5") -> str:
    """Target-parametric preflight warning for a limited SoC."""
    if _target_id(target) == "310p":
        display = soc if isinstance(soc, str) and soc else "<unset>"
        return (
            "WARNING: 310p target SoC %s is missing, unknown, or malformed. "
            "Direct-launch 310p validation will stop closed; configure a "
            "recognized Ascend310P-family SoC (e.g. ASCEND310P_SOC_VERSION="
            "Ascend310P3) for acceptance validation." % display
        )
    return limited_a5_warning(soc)


def is_limited_a5_soc(value: object) -> bool:
    """Return whether *value* must not open A5 validation.

    Ascend910 is intentionally limited to build-smoke/code-generation, while
    empty, unknown, and malformed values fail closed.  Keeping this predicate
    true for every non-approved value is important because the existing O5
    caller uses it as the capability stop before acquiring validation lanes.
    """
    return not is_supported_a5_soc(value)


def limited_a5_warning(soc: str) -> str:
    display = soc if isinstance(soc, str) and soc else "<unset>"
    if not is_known_910_soc(soc):
        return (
            "WARNING: A5_SOC_VERSION=%s is missing, unknown, or malformed. "
            "Direct-launch A5 validation will stop closed; configure a recognized "
            "Ascend950/Ascend950PR SoC for acceptance validation."
        ) % display
    return (
        "WARNING: A5_SOC_VERSION=%s is an Ascend910 target. It is allowed for "
        "lightweight preflight/code-generation smoke checks only; direct-launch "
        "A5 validation is unsupported and will stop before the validation script. "
        "Use an Ascend950PR/Ascend950-class A5 target for acceptance validation."
    ) % display


def limited_a5_validation_error(soc: str) -> str:
    display = soc if isinstance(soc, str) and soc else "<unset>"
    if not is_known_910_soc(soc):
        return (
            "%s: direct-launch A5 validation requires an explicitly recognized "
            "Ascend950/Ascend950PR SoC; configured SoC %s is missing, unknown, "
            "or malformed"
        ) % (LIMITED_A5_SOC_MARKER, display)
    return (
        "%s: direct-launch A5 validation requires Ascend950PR/Ascend950; "
        "Ascend910 (%s) may run preflight and code generation but cannot run "
        "the final A5 validation script"
    ) % (LIMITED_A5_SOC_MARKER, display)
