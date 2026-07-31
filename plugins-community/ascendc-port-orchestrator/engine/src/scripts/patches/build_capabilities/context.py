# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""BuildContext — the read-only inputs a capability factory receives.

Kept deliberately small: a capability contributes build INPUTS derived from these
facts. It MUST NOT reach past them to weaken any finalize/anti-cheat gate, the
arch35-wrap gate, the entry-point gate, or the precision standard (the
canonical-boundary invariant, BUILD_CAPABILITY_EXTENSION_DESIGN.md §2B).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class BuildContext:
    kernel_dir: Path
    sources: tuple[Path, ...] = ()
    ascend_path: Optional[Path] = None
    soc: Optional[str] = None
    # Directory of the overlay build scripts (patches/), used e.g. to locate the
    # committed cann_stubs/ header tree. Mirrors build_ascendc.py's SCRIPT_DIR.
    script_dir: Optional[Path] = None
