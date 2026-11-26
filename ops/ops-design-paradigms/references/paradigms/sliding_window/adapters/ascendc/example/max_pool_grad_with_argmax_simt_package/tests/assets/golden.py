# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden reference for MaxPoolGradWithArgmax (NCHW layout, argmax routing).

语义与 kernel 逐点对齐（范式 grad-routing.md §4 / verification.md §4）：
  - mask 由同口径前向参考复算（平局首遇、NaN 优先）
  - 每个梯度元素恰好路由一次（winner 唯一性），散布到 mask 指向的输入位置
  - 梯度守恒：sum(y) == sum(grad)
"""

import numpy as np


def _forward_mask(x, ksize, strides, pads, dilations):
    """复算 argmax mask（口径：批次内线性索引 h*W+w）。"""
    kh, kw = ksize
    sh, sw = strides
    ph, pw = pads
    dh, dw = dilations
    hspan = (kh - 1) * dh + 1
    wspan = (kw - 1) * dw + 1
    _, _, h, w = x.shape
    out_h = (h + 2 * ph - hspan) // sh + 1
    out_w = (w + 2 * pw - wspan) // sw + 1
    xf = x.astype(np.float32)
    mask = np.empty((x.shape[0], x.shape[1], out_h, out_w), dtype=np.int64)
    for idx_n in range(x.shape[0]):
        for idx_c in range(x.shape[1]):
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
                    maxval = np.float32(-np.inf)
                    maxidx = hstart * w + wstart
                    for hh in range(hstart, hend, dh):
                        for ww in range(wstart, wend, dw):
                            val = xf[idx_n, idx_c, hh, ww]
                            if (val > maxval) or np.isnan(val):
                                maxidx = hh * w + ww
                                maxval = val
                    mask[idx_n, idx_c, ph_out, pw_out] = maxidx
    return mask, out_h, out_w


def max_pool_grad_with_argmax_golden(x, grad, ksize, strides, pads, dilations):
    """NCHW 反向参考（argmax 路由散布形态）：返回 y。

    散布与 kernel 的反窗口枚举 + mask 检查等价（每个 grad 元素恰好路由一次）。
    """
    mask, out_h, out_w = _forward_mask(x, ksize, strides, pads, dilations)
    n, c, _, w = x.shape
    y = np.zeros((n, c, x.shape[2], w), dtype=grad.dtype)
    for idx_n in range(n):
        for idx_c in range(c):
            idx_n * c * out_h * out_w + idx_c * out_h * out_w
            base_in = idx_n * c * x.shape[2] * w + idx_c * x.shape[2] * w
            for ph_out in range(out_h):
                for pw_out in range(out_w):
                    src = int(mask[idx_n, idx_c, ph_out, pw_out])
                    y.flat[base_in + src] += grad[idx_n, idx_c, ph_out, pw_out]
    return y
