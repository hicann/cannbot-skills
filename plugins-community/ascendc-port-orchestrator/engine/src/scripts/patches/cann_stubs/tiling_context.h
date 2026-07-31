// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// DEBT-110 Scope B: CANN host-side stub for port_a3_to_a5 build.
// Minimal gert::TilingContext + fe::PlatFormInfos forward decls.

#ifndef A5OPS_CANN_STUB_TILING_CONTEXT_H_
#define A5OPS_CANN_STUB_TILING_CONTEXT_H_

#include <cstdint>

namespace af {
enum class graphStatus {
    SUCCESS = 0,
    FAILED = 1
};
}  // namespace af

namespace fe {
class PlatFormInfos {};
}  // namespace fe

namespace gert {
class TilingContext {
 public:
    af::graphStatus SetTilingKey(const uint64_t tiling_key) {
        tiling_key_ = tiling_key;
        return af::graphStatus::SUCCESS;
    }
    uint64_t GetTilingKey() const { return tiling_key_; }
    af::graphStatus SetBlockDim(const uint32_t block_dim) {
        block_dim_ = block_dim;
        return af::graphStatus::SUCCESS;
    }
    uint32_t GetBlockDim() const { return block_dim_; }
    fe::PlatFormInfos* GetPlatformInfo() const { return nullptr; }

 private:
    uint32_t block_dim_{1u};
    uint64_t tiling_key_{0u};
};
}  // namespace gert

#endif  // A5OPS_CANN_STUB_TILING_CONTEXT_H_
