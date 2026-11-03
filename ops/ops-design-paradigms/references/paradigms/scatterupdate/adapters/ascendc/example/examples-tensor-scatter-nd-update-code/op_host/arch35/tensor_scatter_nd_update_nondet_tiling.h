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
 * \file tensor_scatter_nd_update_nondet_tiling.h
 * \brief 非确定（直接写覆盖）模板声明（v2.0，MDE §11.5.3）
 *
 * TensorScatterNdUpdateNondetTiling：单 VF 直接写覆盖快速路径（无排序/段表/用户 ws/SyncAll），
 *   priority=0 优先；IsCapable = GetDeterministic()==0（严格 ==0，唯一门控，无 dtype
 *   白名单——覆盖语义只需对齐单条 store 原子，全部 10 种 data dtype 对齐写均原子安全，
 *   MDE §11.6.4 论证），不满足时 registry 自动降级到 DetTiling（priority=10 兜底）。
 */

#ifndef TENSOR_SCATTER_ND_UPDATE_NONDET_TILING_H_
#define TENSOR_SCATTER_ND_UPDATE_NONDET_TILING_H_

#include "tensor_scatter_nd_update_base_tiling.h"

namespace optiling {

class TensorScatterNdUpdateNondetTiling : public TensorScatterNdUpdateBaseTiling {
public:
    explicit TensorScatterNdUpdateNondetTiling(gert::TilingContext* context)
        : TensorScatterNdUpdateBaseTiling(context)
    {}
    ~TensorScatterNdUpdateNondetTiling() override {}

private:
    bool IsCapable() override;                   // GetDeterministic()==0（唯一门控，无 dtype 白名单）
    ge::graphStatus DoOpTiling() override;       // 填充 NondetTilingData + usedCoreNum_
    uint64_t GetTilingKey() const override;      // GET_TPL_TILING_KEY(U32, I32, NONDET)
    ge::graphStatus GetWorkspaceSize() override; // 仅系统 ws（用户 ws = 0）
    ge::graphStatus PostTiling() override;       // SetBlockDim + SetLocalMemorySize（无 ScheduleMode）
    void DumpTilingInfo() override;              // NondetTilingData 字段打印

    int64_t usedCoreNum_ = 1;
};

} // namespace optiling
#endif
