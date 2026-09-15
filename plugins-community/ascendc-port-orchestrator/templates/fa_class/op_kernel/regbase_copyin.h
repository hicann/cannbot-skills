/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file regbase_copyin.h
 * \brief task#34 close-port of the CANN regbase GM->L1 nd2nz staging helper.
 *
 * Close-port of CopyToL1Nd2Nz from
 *   ~/workspace/cann/ops-transformer/attention/common/op_kernel/CopyInL1.h:426-448
 *
 * WHY this exists (root cause of the prior 0/8): the regbase L1->L0 loads
 * (regbase_matmul.h LoadDataToL0A/B<MK,KN>, the 3D-im2col consumer) are a
 * MATCHED PAIR with this GM->L1 producer. The membase LoadNdGmToNzL1 produces a
 * subtly different NZ fractal layout (different dstNzC0Stride / srcDValue
 * semantics) → the 3D-im2col reads it wrong → S = Q.K^T systematically too small.
 * The fix is to use BOTH halves of the regbase matched pair.
 *
 * Key contract (vs membase LoadNdGmToNzL1):
 *   - srcDValue = GM ORG row width (full row stride in elements), NOT the tile dim.
 *   - dstNzC0Stride = AlignUp(nValue, 16)  (NOT raw nValue).
 *
 * Uses the PUBLIC AscendC::DataCopy(L1Tensor, GmTensor, Nd2NzParams) API only —
 * NO #include "arch35/...". Lives in op_kernel/ top-level (same allowance as the
 * other regbase_*.h library headers).
 */
#ifndef REGBASE_COPYIN_H
#define REGBASE_COPYIN_H

#include "kernel_operator.h"

// GM(ND) -> L1(NZ) nd2nz. rows = matrix height, cols = matrix width,
// srcOrgWidth = the GM matrix's full row stride in elements.
template<typename T>
__aicore__ inline void RegbaseCopyToL1Nd2Nz(const AscendC::LocalTensor<T> &l1Tensor,
                                            const AscendC::GlobalTensor<T> &gmTensor,
                                            uint32_t rows, uint32_t cols, uint32_t srcOrgWidth)
{
    AscendC::Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = rows;                              // single ND matrix actual row count
    p.dValue = cols;                             // single ND matrix actual col count (vD)
    p.srcNdMatrixStride = 0;
    p.srcDValue = srcOrgWidth;                   // GM org row width (FULL row stride)
    p.dstNzC0Stride = (rows + 15) >> 4 << 4;     // AlignUp(rows, 16)
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 0;
    AscendC::DataCopy(l1Tensor, gmTensor, p);
}

#endif // REGBASE_COPYIN_H
