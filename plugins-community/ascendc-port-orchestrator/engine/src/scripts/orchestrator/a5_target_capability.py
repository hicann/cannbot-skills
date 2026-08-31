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


LIMITED_A5_SOC_MARKER = "A5_SOC_UNSUPPORTED_FOR_VALIDATION"

# The A5 gate must not infer support from a prefix such as ``Ascend950``.
# Keep the accepted product families explicit and treat every other value as
# unsupported.  The 910 expression intentionally accepts the product suffixes
# used by CANN (A/B/C, B2C/B3/B4, V220, numeric SKU forms, and separators).
_SOC_TOKEN_RE = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$")
_ASCEND_910_RE = re.compile(
    r"^(?:ascend)?910(?:(?:[abc][a-z0-9]*)|(?:[_-][a-z0-9]+))*$"
)
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
