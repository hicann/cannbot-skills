#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""hccl_test dtype 拼写预检；不启动测试程序，不证明算子支持该类型。"""
import argparse
from pathlib import Path
import re
import sys

# CANN 9.2.0 tools/hccl_test/common/src/hccl_test_common.cc:test_typenames，2026-09-12核对。
KNOWN_DTYPES = frozenset(
    'int8 int16 int32 fp16 fp32 int64 uint64 uint8 uint16 uint32 '
    'fp64 bfp16 int128 hif8 fp8e4m3 fp8e5m2 fp8e8m0'.split())


def validate_dtypes(values, test_bin_dir=None):
    allowed = KNOWN_DTYPES
    if test_bin_dir:
        source = Path(test_bin_dir).resolve().parent / 'common/src/hccl_test_common.cc'
        if source.is_file():
            text = source.read_text()
            text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
            match = re.search(r'\btest_typenames\s*\[[^]]*\]\s*=\s*\{([^}]+)\}', text)
            if not match:
                raise ValueError('测试程序 dtype 表格式变化，请核对 ' + str(source))
            allowed = frozenset(re.findall(r'"([a-z0-9]+)"', match[1]))
    bad = [v for v in values if v not in allowed]
    if bad:
        hint = '；本工具链使用 bfp16，不自动将 bf16 改名' if 'bf16' in bad else ''
        raise ValueError('非法 dtype: ' + ','.join(bad) + hint + '；可用拼写: ' + ','.join(sorted(allowed)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dtypes', required=True)
    p.add_argument('--test-bin-dir')
    a = p.parse_args()
    try:
        validate_dtypes(a.dtypes.split(','), a.test_bin_dir)
    except (OSError, ValueError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
