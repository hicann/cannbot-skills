# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Patch build_ascendc.py cmake search block to support CANN 9.0 layouts."""
import re
import sys

p = '/home/npu_user/workspace/AscendOpGenAgent/utils/build_ascendc.py'
src = open(p).read()

# Match the entire if/elseif/else cmake-search block — anchored on the first
# "if(EXISTS ..." through the next "else()"
pattern = re.compile(
    r'if\(EXISTS \$\{\{ASCEND_CANN_PACKAGE_PATH\}\}/tools/tikcpp/ascendc_kernel_cmake\)'
    r'.*?'
    r'else\(\)',
    re.DOTALL
)

new_block = '''if(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/aarch64-linux/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/aarch64-linux/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/x86_64-linux/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/x86_64-linux/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/ascendc_devkit/tikcpp/samples/cmake)
else()'''

m = pattern.search(src)
if not m:
    print("ERROR: cmake search block not found", file=sys.stderr)
    sys.exit(1)

src = pattern.sub(new_block, src)
open(p, 'w').write(src)
print("patched OK")
