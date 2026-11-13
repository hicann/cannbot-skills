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
 * \file conv3ddx_tiling_arch35.h
 * \brief Conv3ddxCompileInfo 编译信息头（入图 TilingParse 场景依赖）
 *
 * 依据 design/HostTiling.md「Host 侧注册」§8：平台资源事实（AIC 核数 / L1 / L0A / L0B /
 * L0C 容量）经 CompileInfo 供 TilingFunc 消费；缺失或任一字段非正时回退运行时
 * platformInfo 查询（GetCoreNumAic / GetCoreMemSize），并与本文件登记的 arch35 参考值
 * 交叉核对（不一致以查询值为准并记录）。字段布局与 Tiling UT 的 Conv3ddxCompileInfoUT
 * 手工转写一致（int64 五字段）。
 */

#ifndef CONV3DDX_TILING_ARCH35_H
#define CONV3DDX_TILING_ARCH35_H

#include "exe_graph/runtime/tiling_parse_context.h"

namespace optiling {

/* 平台资源事实；权威值以运行时查询为准，参考值交叉核对（HostTiling.md §8） */
struct Conv3ddxCompileInfo {
    int64_t coreNum;  /* AIC 核数；arch35 参考值 36（CONV3DDX_MAX_CORES） */
    int64_t l1Bytes;  /* L1 容量；参考值 512 * 1024（CONV3DDX_L1_BYTES） */
    int64_t l0aBytes; /* L0A；参考值 64 * 1024（CONV3DDX_L0A_BYTES） */
    int64_t l0bBytes; /* L0B；参考值 64 * 1024（CONV3DDX_L0B_BYTES） */
    int64_t l0cBytes; /* L0C；参考值 256 * 1024（CONV3DDX_L0C_BYTES） */
};

/* 入图 TilingParse 回调：仅核对编译信息指针存在性（资源校验在 TilingFunc 侧执行） */
ge::graphStatus TilingParseForConv3ddx(gert::TilingParseContext* context);

} // namespace optiling

#endif /* CONV3DDX_TILING_ARCH35_H */
