# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import torch

DEFAULT_TOLERANCES = {
    torch.bfloat16: {
        "atol": 1e-2, "rtol": 1e-2, "ulp_tol": 2,
        "sv_th": 2**-8, "sv_err": 2**-16,
        "max_re_ratio_limit": 10.0,
        "mean_re_ratio_limit": 2.0,
        "rmse_ratio_limit": 2.0,
        "svec_ratio_limit": 2.0,
    },
    torch.float16: {
        "atol": 1e-3, "rtol": 1e-3, "ulp_tol": 2,
        "sv_th": 2**-11, "sv_err": 2**-16,
        "max_re_ratio_limit": 10.0,
        "mean_re_ratio_limit": 2.0,
        "rmse_ratio_limit": 2.0,
        "svec_ratio_limit": 2.0,
    },
    torch.float32: {
        "atol": 1e-5, "rtol": 1e-5, "ulp_tol": 2,
        "sv_th": 2**-14, "sv_err": 2**-30,
        "max_re_ratio_limit": 10.0,
        "mean_re_ratio_limit": 2.0,
        "rmse_ratio_limit": 2.0,
        "svec_ratio_limit": 2.0,
    },
    torch.int8: {"atol": 0, "rtol": 0.0},
    torch.int16: {"atol": 0, "rtol": 0.0},
    torch.int32: {"atol": 0, "rtol": 0.0},
    torch.int64: {"atol": 0, "rtol": 0.0},
    torch.uint8: {"atol": 0, "rtol": 0.0},
    torch.uint16: {"atol": 0, "rtol": 0.0},
    torch.uint32: {"atol": 0, "rtol": 0.0},
    torch.uint64: {"atol": 0, "rtol": 0.0},
    torch.bool: {"atol": 0, "rtol": 0.0},
}

DEFAULT_ULP_CONFIG = {
    "method": "bitwise",
    "include_subnormal": True,
}
