# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden reference for MaxPoolWithArgmax (NCHW layout).

语义与 kernel 逐点对齐（范式 verification.md §1：tie-break 契约一致才可比对）：
  - 窗口枚举按 dilation 格点（负起点 while 推进，非 clamp 到 0）
  - max 初始化 -Inf（空窗输出 -Inf，不转 0）
  - NaN 优先传播：NaN 恒成为 winner（> 比较前先判 isnan）
  - 平局首遇保留（严格大于才替换）
  - argmax 口径：批次内线性索引 h*W+w（不含 n/c 偏移）
"""

import numpy as np

F32_NEG_INF = np.float32(-np.inf)


def max_pool_with_argmax_golden(x, ksize, strides, pads, dilations):
    """NCHW 前向参考：返回 (y, argmax)。

    x: (N, C, H, W) ndarray；ksize/strides/pads/dilations: (kh, kw) 形式。
    """
    kh, kw = ksize
    sh, sw = strides
    ph, pw = pads
    dh, dw = dilations
    hspan = (kh - 1) * dh + 1
    wspan = (kw - 1) * dw + 1
    n, c, h, w = x.shape
    out_h = (h + 2 * ph - hspan) // sh + 1
    out_w = (w + 2 * pw - wspan) // sw + 1

    y = np.empty((n, c, out_h, out_w), dtype=x.dtype)
    argmax = np.empty((n, c, out_h, out_w), dtype=np.int64)
    xf = x.astype(np.float32)

    for idx_n in range(n):
        for idx_c in range(c):
            for ph_out in range(out_h):
                for pw_out in range(out_w):
                    hstart = ph_out * sh - ph
                    wstart = pw_out * sw - pw
                    hend = min(hstart + hspan, h)
                    wend = min(wstart + wspan, w)
                    while hstart < 0:
                        hstart += dh
                    while wstart < 0:
                        wstart += dw
                    maxval = F32_NEG_INF
                    maxidx = hstart * w + wstart
                    for hh in range(hstart, hend, dh):
                        for ww in range(wstart, wend, dw):
                            val = xf[idx_n, idx_c, hh, ww]
                            if (val > maxval) or np.isnan(val):
                                maxidx = hh * w + ww
                                maxval = val
                    y[idx_n, idx_c, ph_out, pw_out] = x.dtype.type(maxval)
                    argmax[idx_n, idx_c, ph_out, pw_out] = maxidx
    return y, argmax
