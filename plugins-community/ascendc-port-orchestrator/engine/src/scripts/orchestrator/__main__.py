# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import sys
import os
# Bootstrap: put this package dir (orchestrator/) on sys.path so the runtime's
# bare `from plugins import ...` (plugins lives at orchestrator/plugins/) resolves
# regardless of how we're launched. Redundant when PYTHONPATH is set (the `orch`
# wrapper does), but load-bearing when the package is launched from a context
# that doesn't (e.g. the cannbot plugin dispatch) — without it the whole plugin
# layer silently falls back (port_a3_to_a5 truth source -> cpu_pytorch instead of
# a3_cann). Upstreamed from the cannbot bundle so the mirror stays verbatim.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from orchestrator.orchestrator import main
sys.exit(main())
