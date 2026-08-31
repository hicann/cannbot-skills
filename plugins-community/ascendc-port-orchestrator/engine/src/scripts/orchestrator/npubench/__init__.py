# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""NPUKernelBench reference/evaluation subsystem of the orchestrator.

Import contract
---------------
External callers import these modules package-qualified, e.g.
``from npubench.npubench_runner import preflight_workspace`` — the top-level
``npubench`` package resolves via the existing mechanism that already puts the
parent ``orchestrator/`` directory on ``sys.path`` (``__main__.py`` bootstrap,
``tests/conftest.py`` and friends), so no new path wiring is required.

The six runner modules (``npubench_runner``/``npubench_core``/
``npubench_fixture``/``npubench_precision``/``npubench_profile``/
``npubench_inputs`` — see ``RUNNER_MODULE_FILENAMES``) are copied
byte-identically into a FLAT staged directory and executed there as a plain
script, and are also loaded by absolute path into the quick-profiler shim;
neither context has a package layout.  Their mutual imports therefore stay
flat (``from npubench_core import ...``).  The append below puts this
package's own directory on ``sys.path`` so those flat sibling imports also
resolve when the modules are imported package-qualified inside the checkout.
It is an append (never a prepend) so the flat names cannot shadow anything.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.append(_here)
del _here
