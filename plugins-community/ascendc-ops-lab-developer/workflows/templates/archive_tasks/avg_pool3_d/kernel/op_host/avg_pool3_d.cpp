/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "torch_kernel_helper.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"

#include "avg_pool3_d_tiling.h"

#include "aclrtlaunch_avg_pool3_d_generic.h"
#include "aclrtlaunch_avg_pool3_d_reduce_d.h"
#include "aclrtlaunch_avg_pool3_d_split_c.h"
#include "aclrtlaunch_avg_pool3_d_split_w.h"
#include "aclrtlaunch_avg_pool3_d_multi_w.h"

namespace ascend_kernel {

namespace {

constexpr int32_t SPLIT_MODE_C = 1;
constexpr int32_t SPLIT_MODE_W = 2;
constexpr int32_t SPLIT_MODE_MULTI_W = 3;

constexpr int32_t IMPL_GENERIC = 0;
constexpr int32_t IMPL_REDUCE_D = 1;
constexpr int32_t IMPL_SPLIT_C = 2;
constexpr int32_t IMPL_SPLIT_W = 3;
constexpr int32_t IMPL_MULTI_W = 4;

inline int32_t CeilDivI32(int32_t a, int32_t b)
{
    if (b == 0) { return 0; }
    return (a + b - 1) / b;
}

int64_t CeilDiv(int64_t value, int64_t factor)
{
    if (factor == 0) {
        return 0;
    }
    if (value % factor == 0) {
        return value / factor;
    }
    return value / factor + 1;
}

c10::SmallVector<int64_t, 3> avg_pool3d_output_size(const at::Tensor &self, c10::IntArrayRef kernel_size,
                                                    c10::IntArrayRef stride, c10::IntArrayRef padding,
                                                    bool ceil_mode)
{
    int self_d = self.size(-3);
    int self_h = self.size(-2);
    int self_w = self.size(-1);

    int64_t kernel_d = ceil_mode ? (CeilDiv(self_d + 2 * padding[0] - kernel_size[0], stride[0]) + 1)
                                 : ((self_d + 2 * padding[0] - kernel_size[0]) / stride[0] + 1);
    int64_t kernel_h = ceil_mode ? (CeilDiv(self_h + 2 * padding[1] - kernel_size[1], stride[1]) + 1)
                                 : ((self_h + 2 * padding[1] - kernel_size[1]) / stride[1] + 1);
    int64_t kernel_w = ceil_mode ? (CeilDiv(self_w + 2 * padding[2] - kernel_size[2], stride[2]) + 1)
                                 : ((self_w + 2 * padding[2] - kernel_size[2]) / stride[2] + 1);

    if (ceil_mode) {
        if ((kernel_d - 1) * stride[0] >= self_d + padding[0]) {
            --kernel_d;
        }
        if ((kernel_h - 1) * stride[1] >= self_h + padding[1]) {
            --kernel_h;
        }
        if ((kernel_w - 1) * stride[2] >= self_w + padding[2]) {
            --kernel_w;
        }
    }

    c10::SmallVector<int64_t, 3> output_size;
    if (self.dim() == 4) {
        output_size = {self.size(0), kernel_d, kernel_h, kernel_w};
    } else {
        output_size = {self.size(0), self.size(1), kernel_d, kernel_h, kernel_w};
    }
    return output_size;
}

int32_t ChooseBlockM(int32_t mOut)
{
    for (int32_t candidate : {64, 32, 16, 8, 4, 2}) {
        if (candidate <= mOut && mOut % candidate == 0) {
            return candidate;
        }
    }
    TORCH_CHECK(false, "Unsupported output spatial size: M_out=", mOut);
}

int32_t ChooseBlockC(int32_t c)
{
    for (int32_t candidate : {256, 128, 64, 32}) {
        if (candidate <= c && c % candidate == 0) {
            return candidate;
        }
    }
    return 0;
}

int32_t ResolveImplMode(int32_t splitMode, int32_t kH, int32_t kW, int32_t sH, int32_t sW,
                        int32_t pH, int32_t pW, int32_t c)
{
    if (kH == 1 && kW == 1 && sH == 1 && sW == 1 && pH == 0 && pW == 0) {
        return IMPL_REDUCE_D;
    }
    if (splitMode == SPLIT_MODE_C) {
        int32_t blockC = ChooseBlockC(c);
        if (blockC > 0) {
            return IMPL_SPLIT_C;
        }
        TORCH_CHECK(false, "split_c scenario requires divisible channel tiles, got C=", c);
    }
    if (splitMode == SPLIT_MODE_W) {
        return IMPL_SPLIT_W;
    }
    if (splitMode == SPLIT_MODE_MULTI_W) {
        return IMPL_MULTI_W;
    }
    return IMPL_GENERIC;
}

struct Pool3dParams {
    int64_t k_d, k_h, k_w;
    int64_t s_d, s_h, s_w;
    int64_t p_d, p_h, p_w;
};

static Pool3dParams NormalizePool3dParams(at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                          at::IntArrayRef padding)
{
    Pool3dParams p;
    p.k_d = kernel_size[0];
    p.k_h = kernel_size.size() == 1 ? p.k_d : kernel_size[1];
    p.k_w = kernel_size.size() == 1 ? p.k_d : kernel_size[2];

    p.s_d = stride.empty() ? p.k_d : stride[0];
    p.s_h = stride.empty() ? p.k_h : (stride.size() == 1 ? p.s_d : stride[1]);
    p.s_w = stride.empty() ? p.k_w : (stride.size() == 1 ? p.s_d : stride[2]);

    p.p_d = padding[0];
    p.p_h = padding.size() == 1 ? p.p_d : padding[1];
    p.p_w = padding.size() == 1 ? p.p_d : padding[2];

    return p;
}

} // namespace

static inline void PopulateAvgPool3dTiling(
    AvgPool3DKernelTiling *tiling,
    int32_t n32, int32_t c32, int32_t d32, int32_t h32, int32_t w32,
    int32_t od32, int32_t oh32, int32_t ow32,
    int32_t kd32, int32_t kh32, int32_t kw32,
    int32_t sd32, int32_t sh32, int32_t sw32,
    int32_t pd32, int32_t ph32, int32_t pw32,
    int32_t countIncludePad, int32_t divisorOverride,
    int32_t splitMode, int32_t blockC, int32_t splitWTileKw, int32_t multiWWindow,
    int32_t blockM, int32_t subBlockM, int32_t mNum,
    int32_t outSpatial, int32_t inSpatial,
    int32_t usedCoreNum, int32_t tasksPerCore,
    int32_t vectorLen, int32_t implMode)
{
    tiling->N = n32;           tiling->C = c32;
    tiling->D = d32;           tiling->H = h32;           tiling->W = w32;
    tiling->OD = od32;         tiling->OH = oh32;         tiling->OW = ow32;
    tiling->kD = kd32;         tiling->kH = kh32;         tiling->kW = kw32;
    tiling->sD = sd32;         tiling->sH = sh32;         tiling->sW = sw32;
    tiling->pD = pd32;         tiling->pH = ph32;         tiling->pW = pw32;
    tiling->countIncludePad = countIncludePad;
    tiling->divisorOverride = divisorOverride;
    tiling->splitMode = splitMode;
    tiling->blockC = blockC;
    tiling->splitWTileKw = splitWTileKw;
    tiling->multiWWindow = multiWWindow;
    tiling->blockM = blockM;   tiling->subBlockM = subBlockM;
    tiling->mNum = mNum;       tiling->outSpatial = outSpatial;
    tiling->inSpatial = inSpatial;
    tiling->hw = h32 * w32;    tiling->cNum = 1;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->vectorLen = vectorLen;
    tiling->reserved0 = implMode;
    tiling->reserved1 = 0;
}

static inline void LaunchAvgPool3dKernel(
    int32_t implMode, uint32_t blockDim,
    const at::Tensor &x_flat, at::Tensor &y_flat, const at::Tensor &tilingNpu)
{
    switch (implMode) {
        case IMPL_REDUCE_D:
            EXEC_KERNEL_CMD(avg_pool3_d_reduce_d, blockDim, x_flat, y_flat, tilingNpu); break;
        case IMPL_SPLIT_C:
            EXEC_KERNEL_CMD(avg_pool3_d_split_c, blockDim, x_flat, y_flat, tilingNpu); break;
        case IMPL_SPLIT_W:
            EXEC_KERNEL_CMD(avg_pool3_d_split_w, blockDim, x_flat, y_flat, tilingNpu); break;
        case IMPL_MULTI_W:
            EXEC_KERNEL_CMD(avg_pool3_d_multi_w, blockDim, x_flat, y_flat, tilingNpu); break;
        case IMPL_GENERIC:
        default:
            EXEC_KERNEL_CMD(avg_pool3_d_generic, blockDim, x_flat, y_flat, tilingNpu); break;
    }
}

struct AvgPool3dI32Params {
    int32_t n, c, d, h, w, od, oh, ow;
    int32_t kd, kh, kw, sd, sh, sw, pd, ph, pw;
    int32_t cntPad, divOverride;
};

static inline AvgPool3dI32Params MakeAvgPool3dI32Params(
    int64_t n, int64_t c, int64_t d, int64_t h, int64_t w,
    int64_t o_d, int64_t o_h, int64_t o_w,
    int64_t k_d, int64_t k_h, int64_t k_w,
    int64_t s_d, int64_t s_h, int64_t s_w,
    int64_t p_d, int64_t p_h, int64_t p_w,
    bool count_include_pad, c10::optional<int64_t> divisor_override)
{
    AvgPool3dI32Params p;
    p.n = static_cast<int32_t>(n); p.c = static_cast<int32_t>(c);
    p.d = static_cast<int32_t>(d); p.h = static_cast<int32_t>(h);
    p.w = static_cast<int32_t>(w);
    p.od = static_cast<int32_t>(o_d); p.oh = static_cast<int32_t>(o_h);
    p.ow = static_cast<int32_t>(o_w);
    p.kd = static_cast<int32_t>(k_d); p.kh = static_cast<int32_t>(k_h);
    p.kw = static_cast<int32_t>(k_w);
    p.sd = static_cast<int32_t>(s_d); p.sh = static_cast<int32_t>(s_h);
    p.sw = static_cast<int32_t>(s_w);
    p.pd = static_cast<int32_t>(p_d); p.ph = static_cast<int32_t>(p_h);
    p.pw = static_cast<int32_t>(p_w);
    p.cntPad = count_include_pad ? 1 : 0;
    p.divOverride = divisor_override.has_value()
        ? static_cast<int32_t>(divisor_override.value()) : 0;
    return p;
}

static inline at::Tensor PrepareAndLaunchAvgPool3d(
    const at::Tensor &self, bool is_5d,
    int64_t n, int64_t c, int64_t d, int64_t h, int64_t w,
    int64_t o_d, int64_t o_h, int64_t o_w,
    int64_t k_d, int64_t k_h, int64_t k_w,
    int64_t s_d, int64_t s_h, int64_t s_w,
    int64_t p_d, int64_t p_h, int64_t p_w,
    bool count_include_pad, c10::optional<int64_t> divisor_override)
{
    auto ap = MakeAvgPool3dI32Params(
        n, c, d, h, w, o_d, o_h, o_w, k_d, k_h, k_w,
        s_d, s_h, s_w, p_d, p_h, p_w, count_include_pad, divisor_override);

    at::Tensor x_input = is_5d ? self.permute({0, 2, 3, 4, 1}).contiguous()
                               : self.permute({0, 2, 3, 1}).contiguous();
    at::Tensor x_flat = x_input.reshape({n * d * h * w, c});
    TORCH_CHECK(x_flat.is_contiguous(), "avg_pool3d: x_flat must be contiguous");
    TORCH_CHECK(x_flat.scalar_type() == at::kFloat, "avg_pool3d: only float32 is supported");

    int32_t inSpatial = ap.d * ap.h * ap.w;
    int32_t outSpatial = ap.od * ap.oh * ap.ow;
    int32_t mOut = ap.n * outSpatial;
    int32_t blockM = ChooseBlockM(mOut);
    if (blockM == 0) { blockM = 1; }
    int32_t subBlockM = blockM / DEFAULT_VEC_NUM;
    int32_t mNum = mOut / blockM;
    int32_t usedCoreNum = std::min(DEFAULT_NUM_PHYSICAL_CORES, mNum);
    int32_t tasksPerCore = CeilDivI32(mNum, usedCoreNum);
    int32_t implMode = ResolveImplMode(0, ap.kh, ap.kw, ap.sh, ap.sw, ap.ph, ap.pw, ap.c);

    at::Tensor y_flat = at::empty({mOut, ap.c}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    at::Tensor tilingCpu = at::empty({static_cast<long>(sizeof(AvgPool3DKernelTiling))},
                                      at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<AvgPool3DKernelTiling *>(tilingCpu.data_ptr());

    PopulateAvgPool3dTiling(tiling,
        ap.n, ap.c, ap.d, ap.h, ap.w, ap.od, ap.oh, ap.ow,
        ap.kd, ap.kh, ap.kw, ap.sd, ap.sh, ap.sw, ap.pd, ap.ph, ap.pw,
        ap.cntPad, ap.divOverride,
        0, 0, 0, 1, blockM, subBlockM, mNum, outSpatial, inSpatial,
        usedCoreNum, tasksPerCore, std::min<int32_t>(ap.c, 256), implMode);

    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    tiling->launchBlocks = static_cast<int32_t>(blockDim);
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    LaunchAvgPool3dKernel(implMode, blockDim, x_flat, y_flat, tilingNpu);
    return y_flat;
}

at::Tensor avg_pool3d(const at::Tensor &self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
                      at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
                      c10::optional<int64_t> divisor_override)
{
    TORCH_CHECK(self.dim() == 4 || self.dim() == 5, "avg_pool3d: input must be 4D or 5D");
    TORCH_CHECK(!kernel_size.empty(), "avg_pool3d: kernel_size must not be empty");

    auto params = NormalizePool3dParams(kernel_size, stride, padding);
    int64_t k_d = params.k_d, k_h = params.k_h, k_w = params.k_w;
    int64_t s_d = params.s_d, s_h = params.s_h, s_w = params.s_w;
    int64_t p_d = params.p_d, p_h = params.p_h, p_w = params.p_w;

    TORCH_CHECK(s_d != 0 && s_h != 0 && s_w != 0, "avg_pool3d: stride should not contain zero");

    c10::SmallVector<int64_t, 3> kernel_sizes = {k_d, k_h, k_w};
    c10::SmallVector<int64_t, 3> stride_sizes = {s_d, s_h, s_w};
    c10::SmallVector<int64_t, 3> padding_sizes = {p_d, p_h, p_w};

    auto output_size = avg_pool3d_output_size(self, at::IntArrayRef(kernel_sizes),
                                               at::IntArrayRef(stride_sizes),
                                               at::IntArrayRef(padding_sizes), ceil_mode);
    at::Tensor result = at_npu::native::empty_with_format(
        output_size, self.options(), at_npu::native::get_npu_format(self));

    int64_t n = self.size(0);
    bool is_5d = (self.dim() == 5);
    int64_t c = is_5d ? self.size(1) : 1;
    int64_t d = self.size(-3), h = self.size(-2), w = self.size(-1);
    int64_t dim_offset = is_5d ? 1 : 0;
    int64_t o_d = output_size[1 + dim_offset];
    int64_t o_h = output_size[2 + dim_offset];
    int64_t o_w = output_size[3 + dim_offset];

    at::Tensor y_flat = PrepareAndLaunchAvgPool3d(
        self, is_5d, n, c, d, h, w, o_d, o_h, o_w,
        k_d, k_h, k_w, s_d, s_h, s_w, p_d, p_h, p_w,
        count_include_pad, divisor_override);

    at::Tensor y_ndhwc = y_flat.reshape({n, o_d, o_h, o_w, c});
    at::Tensor y_ncdhw = y_ndhwc.permute({0, 4, 1, 2, 3}).contiguous();
    if (self.dim() == 4) {
        y_ncdhw = y_ncdhw.squeeze(1).contiguous();
    }
    if (!y_ncdhw.is_same(result)) {
        result.copy_(y_ncdhw);
    }
    return result;
}

} // namespace ascend_kernel
