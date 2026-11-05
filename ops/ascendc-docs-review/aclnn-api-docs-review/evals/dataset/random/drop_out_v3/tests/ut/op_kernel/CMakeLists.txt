# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(KERNEL_STAGING_DIR ${CMAKE_CURRENT_BINARY_DIR}/kernel_dep_staging)
    file(MAKE_DIRECTORY ${KERNEL_STAGING_DIR}/drop_out_v3/arch35)
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${PROJECT_SOURCE_DIR}/random/random_common/op_kernel
        ${KERNEL_STAGING_DIR}/random_common)

    set(drop_out_v3_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch35/drop_out_v3_tiling_arch35.cpp
        ${PROJECT_SOURCE_DIR}/random/random_common/op_host/arch35/random_tiling_arch35.cpp)
    AddOpTestCase(drop_out_v3 "ascend950"
        "-DDTYPE_X=float -DDTYPE_P=float -DTestUtDefaultTilingStruct=RandomUnifiedSimtTilingDataStruct -I${KERNEL_STAGING_DIR}/drop_out_v3/arch35"
        "${drop_out_v3_tiling_files}")
endif()
