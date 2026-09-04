# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""UT for ascendc_static_check.check_kernel_has_computation helper-header carve-out.

Regression coverage for the 2026-08-24 extension: the carve-out must scan the
whole kernel tree, not just the flagged file's own directory. Multi-dir kernel
projects (op_kernel/ + utils/) legitimately keep compute-free tiling/ABI headers
in a separate directory; FusionAttention's utils/fa_tiling_data.h carries
``__aicore__`` only inside ``#ifdef`` host/device guards and was false-flagged
while the computation lives in ``../op_kernel/fusion_attention.cpp``.
"""

import ascendc_static_check as chk


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# A tiling-data header: __aicore__ appears only in preprocessor guards, no
# computation markers at all.
TILING_HEADER = """\
#ifndef FA_TILING_DATA_H
#define FA_TILING_DATA_H
#ifdef __aicore__
struct FaTilingData {
    int64_t numHeads;
    int64_t seqLen;
};
#else
struct FaTilingDataHost {
    int64_t numHeads;
    int64_t seqLen;
};
#endif
#endif
"""

# A real SIMD kernel: __global__ __aicore__ entry + TQue + DataCopy + VEC op.
KERNEL_CPP = """\
#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void fa_kernel(GM_ADDR q, GM_ADDR out) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQ;
    pipe.InitBuffer(inQ, 1, 256);
    GlobalTensor<float> gq;
    gq.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(q));
    LocalTensor<float> lq = inQ.AllocTensor<float>();
    DataCopy(lq, gq, 64);
    Adds(lq, lq, 1.0f, 64);
    inQ.EnQue(lq);
}
"""


def test_tiling_header_in_utils_dir_not_flagged_when_computation_lives_in_op_kernel(tmp_path):
    """A compute-free header under kernel/utils/ must pass.

    Multi-dir project: kernel/op_kernel/ carries the computation
    (FusionAttention false-positive regression).
    """
    kernel = tmp_path / "kernel"
    header = kernel / "utils" / "fa_tiling_data.h"
    _write(header, TILING_HEADER)
    _write(kernel / "op_kernel" / "fusion_attention.cpp", KERNEL_CPP)

    lines = TILING_HEADER.splitlines(keepends=True)
    assert chk.check_kernel_has_computation(str(header), lines) == []


def test_compute_free_header_still_flagged_when_whole_tree_lacks_computation(tmp_path):
    """Stub detection is preserved for a whole tree without markers.

    No file anywhere in the kernel tree carries computation markers, so the
    header must still be flagged.
    """
    kernel = tmp_path / "kernel"
    header = kernel / "utils" / "fa_tiling_data.h"
    _write(header, TILING_HEADER)
    _write(kernel / "op_kernel" / "stub.cpp", "#ifdef __aicore__\n#endif\n")

    violations = chk.check_kernel_has_computation(str(header), TILING_HEADER.splitlines(keepends=True))
    assert len(violations) == 1
    assert "computation markers" in violations[0]["detail"]


def test_header_without_kernel_ancestor_falls_back_to_same_dir_scan(tmp_path):
    """No 'kernel' directory up the path falls back to the same-dir scan.

    The legacy sibling scan still rescues helper headers.
    """
    src = tmp_path / "src"
    header = src / "fa_tiling_data.h"
    _write(header, TILING_HEADER)
    _write(src / "fusion_attention.cpp", KERNEL_CPP)

    assert chk.check_kernel_has_computation(str(header), TILING_HEADER.splitlines(keepends=True)) == []
