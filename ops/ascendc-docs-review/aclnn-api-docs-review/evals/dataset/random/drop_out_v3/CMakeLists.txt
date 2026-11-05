# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

# 设置算子定义时支持的芯片类型
set(SUPPORT_COMPUTE_UNIT "ascend310p" "ascend910_93" "ascend910b" "ascend950")
# 设置每种芯片类型对应的tiling文件目录，即采用op_host目录下哪个文件夹下的tiling文件编译
set(SUPPORT_TILING_DIR "arch32" "arch32" "arch32" "arch35")
add_all_modules_sources(OPTYPE drop_out_v3 ACLNNTYPE aclnn_exclude COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR} DEPENDENCIES random_common stateless_drop_out_gen_mask drop_out_do_mask DISABLE_IN_OPP TRUE)
