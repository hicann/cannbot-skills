# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/x86_64-linux/lib64:$ASCEND_HOME_PATH/x86_64-linux/devlib:$ASCEND_HOME_PATH/x86_64-linux/devlib/linux/x86_64:${LD_LIBRARY_PATH}"
export PYTHONPATH="$script_path/py_patch:$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH}"
export CPLUS_INCLUDE_PATH="/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:/usr/lib/gcc/x86_64-linux-gnu/11/include:${CPLUS_INCLUDE_PATH}"
export C_INCLUDE_PATH="/usr/include:/usr/lib/gcc/x86_64-linux-gnu/11/include:${C_INCLUDE_PATH}"