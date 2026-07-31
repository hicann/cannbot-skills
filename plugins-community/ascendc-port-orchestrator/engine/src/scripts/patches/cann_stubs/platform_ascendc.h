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
// Minimal platform_ascendc::SocVersion enum + PlatformAscendC shim.
// Only the parts referenced by op_host/<op>_tiling.h files are kept;
// runtime initialization (PlatformAscendCManager) is declaration-only.

#ifndef A5OPS_CANN_STUB_PLATFORM_ASCENDC_H_
#define A5OPS_CANN_STUB_PLATFORM_ASCENDC_H_

#include <cstdint>

namespace fe {
class PlatFormInfos;
}

namespace platform_ascendc {

enum class CoreMemType {
    L0_A = 0,
    L0_B = 1,
    L0_C = 2,
    L1 = 3,
    L2 = 4,
    UB = 5,
    HBM = 6,
    RESERVED
};

// Mirrors CANN's enum; extra A3/A5-era variants added to satisfy newer
// op_host references (port_a3 cohort).
enum class SocVersion {
    ASCEND910 = 0,    // Ascend910A, Ascend910B
    ASCEND910B,       // Ascend910B1~4, Ascend910B2C
    ASCEND910_93,     // Ascend910_93 (A3)
    ASCEND310P,       // Ascend310P1, Ascend310P3
    ASCEND310B,       // Ascend310B1~4
    ASCEND950,        // A5 family
    ASCEND950PR,      // A5 PR variant
    RESERVED_VERSION = 99999
};

class PlatformAscendC {
 public:
    PlatformAscendC() = delete;
    ~PlatformAscendC() = default;
    explicit PlatformAscendC(fe::PlatFormInfos* platformInfo) : platformInfo_(platformInfo) {}

    uint32_t GetCoreNum(void) const { return 48u; }
    uint32_t GetCoreNumAic(void) const { return 24u; }
    uint32_t GetCoreNumAiv(void) const { return 48u; }
    uint32_t GetCoreNumVector(void) const { return 40u; }
    uint32_t GetLibApiWorkSpaceSize(void) const { return 0u; }

    void GetCoreMemSize(const CoreMemType& memType, uint64_t& size) const {
        switch (memType) {
            case CoreMemType::L0_A: size = 64 * 1024; break;
            case CoreMemType::L0_B: size = 64 * 1024; break;
            case CoreMemType::L0_C: size = 128 * 1024; break;
            case CoreMemType::L1:   size = 512 * 1024; break;
            case CoreMemType::L2:   size = 192ull * 1024 * 1024; break;
            case CoreMemType::UB:   size = 192 * 1024; break;
            case CoreMemType::HBM:  size = 10ull * 1024 * 1024; break;
            default: size = 0; break;
        }
    }

    void GetCoreMemBw(const CoreMemType& /*memType*/, uint64_t& bwSize) const {
        bwSize = 64u;
    }

    SocVersion GetSocVersion(void) const { return SocVersion::ASCEND910B; }

 private:
    fe::PlatFormInfos* platformInfo_{nullptr};
};

class PlatformAscendCManager {
 public:
    static PlatformAscendC* GetInstance() { return nullptr; }
    static PlatformAscendC* GetInstance(const char* /*customSocVersion*/) { return nullptr; }
};

}  // namespace platform_ascendc

#endif  // A5OPS_CANN_STUB_PLATFORM_ASCENDC_H_
