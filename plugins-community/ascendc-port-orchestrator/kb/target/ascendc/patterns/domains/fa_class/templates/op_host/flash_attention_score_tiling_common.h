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
 * \file flash_attention_score_tiling_common.h
 * \brief GE op-build CompileInfo POD for the FA-class op_host.
 *
 * KB-AUTHORED (NOT a CANN-source copy). Per owner rule 2026-06-13: CANN *library
 * install* headers are usable AscendC API; CANN *source-only* headers are not.
 * The original `<op>_tiling_common.h` lives only in CANN source (not the install),
 * so this struct is RE-AUTHORED here from first principles: it is a trivial POD of
 * platform parameters, every field populated by INSTALLED `platform_ascendc` API
 * getters (GetCoreNumAiv / GetCoreNumAic / GetSocVersion / GetCurNpuArch /
 * GetCoreMemSize). The field TYPES (platform_ascendc::SocVersion, NpuArch) come
 * from the installed `tiling/platform/platform_ascendc.h`. No CANN-source content.
 *
 * Op-name parameterization: the `flash_attention_score` token (filename) is
 * substituted by assemble_ge_ophost for other FA-class ops; the CamelCase
 * `FlashAttentionScoreCompileInfo` struct name matches the op_host tiling.cpp
 * template (both CamelCase, kept consistent).
 */
#pragma once

#include <cstdint>
#include "tiling/platform/platform_ascendc.h"  // installed: platform_ascendc::SocVersion, NpuArch

namespace optiling {

// Plain platform-info POD passed via the GE TilingParseContext CompileInfo slot.
// The op author defines this struct freely; the framework only stores/returns the
// blob. Fields are exactly those the tiling.cpp reads (aicNum / l2CacheSize / ...)
// and writes from the installed ascendcPlatform getters.
struct FlashAttentionScoreCompileInfo {
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    uint64_t l2CacheSize = 0;
    platform_ascendc::SocVersion socVersion{};  // value-init; tiling.cpp overwrites via GetSocVersion()
    NpuArch npuArch{};                           // value-init; tiling.cpp overwrites via GetCurNpuArch()
};

} // namespace optiling
