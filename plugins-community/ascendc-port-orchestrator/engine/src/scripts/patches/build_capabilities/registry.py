# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Capability registry — name -> factory((BuildContext) -> CMakeContribution).

Named capabilities (e.g. `kfc_bootstrap`, `il_lowering`) register here and are
resolved from a kernel's declared `capabilities` array. A declared name that does
not resolve FAILS LOUDLY (design §2A) — get() raises, never silently skips.

Baseline accretions (DEBT-20/20.1, DEBT-110) are NOT registered here: they always
apply and live in build_capabilities.baseline.
"""
from __future__ import annotations

from typing import Callable

from .context import BuildContext
from .contribution import CMakeContribution

Factory = Callable[[BuildContext], CMakeContribution]


class Registry:
    def __init__(self) -> None:
        self._factories: dict[str, Factory] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._factories

    def register(self, name: str, factory: Factory) -> None:
        if name in self._factories:
            raise ValueError(f"build capability '{name}' already registered")
        self._factories[name] = factory

    def get(self, name: str) -> Factory:
        try:
            return self._factories[name]
        except KeyError:
            raise KeyError(
                f"unknown build capability '{name}'; "
                f"registered: {sorted(self._factories)}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._factories)


# Process-wide registry. Named capability modules register on import.
registry = Registry()
