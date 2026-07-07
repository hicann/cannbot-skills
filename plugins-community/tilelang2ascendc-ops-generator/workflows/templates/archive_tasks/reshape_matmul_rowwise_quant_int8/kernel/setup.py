#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# Template for: archive_tasks/reshape_matmul_rowwise_quant_int8/kernel

import sys
from pathlib import Path

_COMMON_DIR = Path(__file__).resolve().parent.parent.parent  # archive_tasks/
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))
from _kernel_setup_common import kernel_setup  

kernel_setup("reshape_matmul_quant", kernel_dir=__file__,
             ext_name="reshape_matmul_quant_ext",
             description="Reshape MatMul Quant AscendC kernel")
