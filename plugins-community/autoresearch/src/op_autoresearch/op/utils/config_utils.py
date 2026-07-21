# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Single source of truth — SKU enumeration + per-family DSL whitelists.
#
# A SKU is a concrete arch string KernelVerifier accepts (e.g. ``ascend910b3``).
# Within the Ascend backend, SKUs cluster into FAMILIES that
# share a DSL capability list:
#   - ascend has two families: ``910`` (910B + 910_93xx + 950 — full DSL stack)
#     and ``310`` (310p3 — AscendC and CATLASS only).
#
# Ascend uses **explicit SKU tuples** — SKUs are discrete real
#     products (no ``ascend910b99``); enumeration gives crisp error msgs.
# Adding a new ascend SKU: one line in the matching ``_*_SKUS`` tuple.
# Adding a new DSL across an entire family: one entry in ``_DSL_TABLE``.
# ---------------------------------------------------------------------------

# Ascend 910-class families (full DSL stack) — explicit SKUs, real products
_ASCEND_910B_SKUS = (
    "ascend910b1", "ascend910b2", "ascend910b2c", "ascend910b3", "ascend910b4",
)
_ASCEND_910_93_SKUS = (
    "ascend910_9362", "ascend910_9372", "ascend910_9381",
    "ascend910_9382", "ascend910_9391", "ascend910_9392",
)
_ASCEND_950_SKUS = (
    "ascend950dt_95a",
    "ascend950pr_950z", "ascend950pr_9572", "ascend950pr_9574", "ascend950pr_9575",
    "ascend950pr_9576", "ascend950pr_9577", "ascend950pr_9578", "ascend950pr_9579",
    "ascend950pr_957b", "ascend950pr_957d", "ascend950pr_9581", "ascend950pr_9582",
    "ascend950pr_9584", "ascend950pr_9587", "ascend950pr_9588", "ascend950pr_9589",
    "ascend950pr_958a", "ascend950pr_958b", "ascend950pr_9591", "ascend950pr_9592",
    "ascend950pr_9599",
)
_ASCEND_910_FAMILY = _ASCEND_910B_SKUS + _ASCEND_910_93_SKUS + _ASCEND_950_SKUS

# Ascend 310 family (AscendC and CATLASS)
_ASCEND_310_SKUS = ("ascend310p3",)


# Family → DSL whitelist, keyed by (framework, backend, family-tag).
# family-tag is internal (``910`` / ``310``) — callers use the
# arch string directly and we resolve the family via ``_family_of``.
# Derived from adapters.factory.DSL_REGISTRY (the single source of truth);
# adding a new DSL is one entry in the registry, not two.
def _add_dsl_names(table: dict, support, names) -> None:
    for framework_backend_family in support:
        dsls = table.setdefault(framework_backend_family, [])
        for name in names:
            if name not in dsls:
                dsls.append(name)


def _build_dsl_table():
    from op_autoresearch.op.verifier.adapters.factory import DSL_REGISTRY
    table: dict = {}
    for name, entry in DSL_REGISTRY.items():
        _add_dsl_names(table, entry.support, (name,))
        _add_dsl_names(table, entry.support, entry.aliases)
    return {fbf: tuple(dsls) for fbf, dsls in table.items()}


_DSL_TABLE = _build_dsl_table()

# Canonical DSL set (single source — referenced by normalize_dsl / check_dsl
# instead of duplicating the literal list).
_ALL_DSLS = frozenset(
    dsl for dsls in _DSL_TABLE.values() for dsl in dsls
)


def _family_of(backend: str, arch: str) -> Optional[str]:
    """Return the family tag for (backend, arch), or None if the arch
    isn't recognized under the Ascend backend.
    """
    if backend != "ascend":
        return None
    if arch in _ASCEND_310_SKUS:
        return "310"
    if arch in _ASCEND_910_FAMILY:
        return "910"
    return None


def arch_hint(backend: str) -> str:
    """User-facing hint string describing what arch values ``backend``
    accepts. Used by error messages in this module + downstream CLI
    validators. Ascend is enumerated because its SKUs are discrete.
    """
    if backend == "ascend":
        return "/".join(_ASCEND_310_SKUS + _ASCEND_910_FAMILY)
    return ""


def check_backend_arch(backend: str, arch: str):
    """
    验证后端与架构的匹配关系
    Args:
        backend: 计算后端名称（ascend）
        arch: 硬件架构名称
    """
    if backend != "ascend":
        raise ValueError("backend must be ascend")
    if _family_of(backend, arch) is None:
        raise ValueError(
            f"{backend} backend does not recognize arch={arch} "
            f"(accepted: {arch_hint(backend)})"
        )


def is_supported_arch(backend: str, arch: str) -> bool:
    """Whether ``arch`` belongs to the canonical backend family table."""
    return backend == "ascend" and _family_of(backend, arch) is not None


def normalize_dsl(dsl: str, backend: str = None) -> str:
    """
    规范化 DSL 类型，将通用的 triton 转换为 triton_ascend。

    Args:
        dsl: 实现类型
        backend: 硬件后端名称（ascend）

    Returns:
        规范化后的DSL类型

    Raises:
        ValueError: 如果dsl为"triton"但backend未提供或无效
    """
    dsl = dsl.lower()

    # 如果已经是规范化的类型，直接返回
    if dsl in _ALL_DSLS:
        return dsl

    # 如果是通用的triton，需要根据backend转换
    if dsl == "triton":
        if backend is None:
            raise ValueError(
                "dsl='triton' requires backend='ascend'. "
                "Use 'triton_ascend' explicitly, or provide backend for conversion. "
            )
        backend = backend.lower()
        if backend == "ascend":
            return "triton_ascend"
        raise ValueError(f"dsl='triton' requires backend='ascend', got {backend!r}")

    # 其他情况直接返回
    return dsl


def check_dsl(dsl: str):
    """
    验证实现类型
    Args:
        dsl: triton_ascend、triton-russia、ascendc 或 ascendc_catlass
    """
    if dsl not in _ALL_DSLS:
        raise ValueError(
            f"dsl must be one of {sorted(_ALL_DSLS)}. "
            "Use 'triton_ascend' instead of the generic 'triton' name."
        )


def check_task_type(task_type: str):
    """
    验证任务类型
    Args:
        task_type: 任务类型(precision_only/profile)
    """
    if task_type not in ["precision_only", "profile"]:
        raise ValueError("task_type must be precision_only or profile")


def supported_dsls(framework: str, backend: str, arch: str) -> Optional[tuple]:
    """Return the DSL whitelist for ``(framework, backend, arch)``, or
    None if the combination is not supported. Single canonical lookup —
    every other validator in this module routes through this.
    """
    family = _family_of(backend, arch)
    if family is None:
        return None
    return _DSL_TABLE.get((framework, backend, family))


# Backward-compat: VALID_CONFIGS is the derived (framework, backend) →
# {arch: dsl_list} view. Ascend gets per-arch keys because its SKUs are
# explicitly enumerated.
# Use ``supported_dsls(framework, backend, arch)`` for membership checks;
# iterate ``VALID_CONFIGS[fw][be]`` only when you need the ascend SKU
# enumeration.
def _build_valid_configs() -> Dict[str, Dict[str, Dict[str, list]]]:
    table: Dict[str, Dict[str, Dict[str, list]]] = {}
    enumerated_skus = {
        ("ascend", "910"): _ASCEND_910_FAMILY,
        ("ascend", "310"): _ASCEND_310_SKUS,
    }
    for (fw, be, fam), dsls in _DSL_TABLE.items():
        be_table = table.setdefault(fw, {}).setdefault(be, {})
        for sku in enumerated_skus.get((be, fam), ()):
            be_table[sku] = list(dsls)
    return table


VALID_CONFIGS = _build_valid_configs()


def check_task_config(framework: str, backend: str, arch: str, dsl: str):
    """
    统一验证配置参数之间的依赖关系
    Args:
        framework: 框架类型
        backend: 硬件后端名称
        arch: 硬件架构名称
        dsl: 实现类型（通用 triton 会转换为 triton_ascend）
    """
    normalized_dsl = normalize_dsl(dsl, backend)

    if framework not in VALID_CONFIGS:
        raise ValueError(f"Unsupported framework: {framework}")
    if backend not in VALID_CONFIGS[framework]:
        raise ValueError(f"Framework {framework} does not support backend: {backend}")

    dsls = supported_dsls(framework, backend, arch)
    if dsls is None:
        # Distinguish "unknown arch under this backend" from "arch known but
        # this framework doesn't support that family" — the second case can
        # happen e.g. for mindspore + ascend910b3 (family 910 has no row
        # under mindspore at the moment? actually it does. example only).
        if _family_of(backend, arch) is None:
            raise ValueError(f"Backend {backend} does not support arch: {arch}")
        raise ValueError(
            f"Framework {framework} does not support arch {arch} on {backend}"
        )

    if normalized_dsl not in dsls:
        raise ValueError(f"Arch {arch} does not support dsl: {normalized_dsl}")

    return normalized_dsl
