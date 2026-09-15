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
 * \file ge_host_shim.h  (KB asset — GE op_host Tier-1 syntax-check shell boundary)
 * \brief Provides the CANN op-build-kit INTERNAL glue symbols that a GE op_host
 *        (def / infershape / tiling.cpp) references but that are ABSENT from the public
 *        CANN include tree (`<CANN>/x86_64-linux/include`).
 *
 * ============================================================================================
 * WHY THIS EXISTS (the GE-shell boundary, documented for the next assembler)
 * --------------------------------------------------------------------------------------------
 *   The graybox (`flash_attention_score-gb-1` / refined `-gb-2`, 2026-06-11) proved that an
 *   A5 GE op_host can be ASSEMBLED from the A3 input + KB with ZERO A5 CANN source. To
 *   Tier-1 `-fsyntax-only` type-check the result STANDALONE against the public headers, a
 *   handful of symbols are missing because they live in CANN-internal / op-build-kit headers:
 *
 *     | symbol                          | real (internal) home                                  |
 *     |---------------------------------|-------------------------------------------------------|
 *     | OP_LOGI / OP_LOGW / OP_LOGE /   | ops-transformer common/include/log/log.h              |
 *     |   OP_LOGD                       |                                                       |
 *     | OP_CHECK_NULL_WITH_CONTEXT /    | ops-transformer common/include/log/log.h (+ error_   |
 *     |   OP_CHECK_IF                   |   util) — null / condition guard macros               |
 *     | GET_TPL_TILING_KEY              | <op>/op_kernel/arch{22,35}/*_template_tiling_key.h    |
 *     | ASCENDC_EXTERN_C                | kernel_operator / ascendc runtime headers (= extern"C")|
 *     | platform_ascendc::PlatformAscendC| tiling/platform/platform_ascendc.h (op-build kit)     |
 *     | NpuArch / DAV_3510 / DAV_2201   | platform/soc_spec.h (real enum present, mirrored here |
 *     |                                 |   so the shell is self-contained for syntax-check)    |
 *
 *   ALL of these are GE-SHELL glue (logging, null-guards, the tiling-key macro, the platform
 *   wrapper) — NONE carries algorithm. The algorithm-bearing tiling VALUES come from
 *   `wp_fa_host_tiling.h` (`wfh::Calc*`), included separately.
 *
 *   AT TIER-1 (`-fsyntax-only`, standalone): include THIS shim — it lets g++ type-check the
 *   GE shell against the public GE / gert / register / platform headers without the op-build kit.
 *
 *   AT TIER-2 (real op build inside the CANN op-build kit / Ascend custom-op project): the kit's
 *   own headers provide the real symbols. DO NOT ship this shim into a Tier-2 build — either
 *   guard your `#include "ge_host_shim.h"` behind a `#ifdef GE_HOST_TIER1_SYNTAX_ONLY`, or drop
 *   the include and let the kit headers resolve the symbols. This shim is a SYNTAX-CHECK
 *   SCAFFOLD, not a production dependency.
 * ============================================================================================
 */
#ifndef GE_HOST_SHIM_H
#define GE_HOST_SHIM_H

#include <cstdint>
#include <cstddef>
#include <string>

/* ---- OP_LOG* (ops-transformer common/include/log/log.h is internal) ---- */
#define OP_LOGI(ctx, ...)  do { (void)(ctx); } while (0)
#define OP_LOGW(ctx, ...)  do { (void)(ctx); } while (0)
#define OP_LOGE(ctx, ...)  do { (void)(ctx); } while (0)
#define OP_LOGD(ctx, ...)  do { (void)(ctx); } while (0)

/* ---- OP_CHECK_* null/condition guards (internal) ---- */
#define OP_CHECK_NULL_WITH_CONTEXT(ctx, ptr)                       \
    do { if ((ptr) == nullptr) { return ge::GRAPH_FAILED; } } while (0)
#define OP_CHECK_IF(cond, logExpr, retExpr)                        \
    do { if (cond) { logExpr; retExpr; } } while (0)

/* ---- template-tiling-key macro (arch22/arch35 template_tiling_key.h are internal) ---- */
#define GET_TPL_TILING_KEY(...) (0ULL)

/* ---- ASCENDC_EXTERN_C (from kernel_operator/ascendc headers, not in this public tree) ---- */
#ifndef ASCENDC_EXTERN_C
#define ASCENDC_EXTERN_C extern "C"
#endif

/* ---- platform_ascendc tiling wrapper (tiling/platform/platform_ascendc.h ships in the
 *      op-build kit, not this public include tree). Minimal shim of the API surface the
 *      tiling.cpp + the wfh:: caller need (core counts, mem sizes, L2 cache size, arch query).
 *      The hardcoded values (28 cores, 192 MiB L2) are the A5/Ascend950PR platform constants;
 *      at Tier-2 the real PlatformAscendC reads them from the platform info. ---- */
namespace platform_ascendc {
enum class CoreMemType { L0_A, L0_B, L0_C, L1, L2, UB };
class PlatformAscendC {
public:
    PlatformAscendC() = default;
    template <typename T> explicit PlatformAscendC(T*) {}
    uint32_t GetCoreNumAiv() const { return 28U; }
    uint32_t GetCoreNumAic() const { return 28U; }
    uint32_t CalcTschBlockDim(uint32_t a, uint32_t, uint32_t) const { return a; }
    int GetSocVersion() const { return 0; }
    int GetCurNpuArch() const { return 0; }
    void GetCoreMemSize(CoreMemType, uint64_t &out) const { out = 0; }
    uint64_t GetCacheSize(int) const { return (uint64_t)192 * 1024 * 1024; }
};
}  // namespace platform_ascendc

/* NpuArch + DAV_3510 are referenced unqualified in the A3 tiling.cpp via `using namespace`;
 * the public soc_spec.h provides the real enum, but to keep the shell self-contained for
 * syntax-check we mirror the values the arch-dispatch compares against (DAV_2201 = arch22 /
 * A3; DAV_3510 = arch35 / A5). */
enum NpuArchShim { DAV_2201 = 2201, DAV_3510 = 3510 };

#endif  // GE_HOST_SHIM_H
