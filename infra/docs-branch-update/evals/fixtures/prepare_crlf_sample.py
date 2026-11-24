#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""生成 CRLF 行尾的评测样例文件, 用于 docs-branch-update 的 evals 用例。

仓库卫生检查禁止 *.md 携带 CRLF 入库, 因此 CRLF 样例在评测时生成本地文件而不入库
(与 knowledge-ingest 的 prepare_ingest_fixture.py 同一模式)。生成内容以 LF 定义,
统一转换为 \r\n 后按字节写出, 保证输出文件全部行尾为 CRLF、不含裸 LF。

用法:
    python3 prepare_crlf_sample.py <输出文件路径>
"""
import os
import sys

CONTENT = """# docs-branch-update CRLF 评测样例

本样例由 prepare_crlf_sample.py 生成, 全部为 CRLF 行尾, 验证字节级处理保持行尾不变。

## 产品内容示例

<!-- npu="x90" id1 -->
CRLF 单类型块, 整块删除且行尾保持 CRLF。
<!-- end id1 -->

<!-- npu="950,x90" id2 -->
CRLF 混合类型标签块, 移除 x90 后保留本段内容。
<!-- end id2 -->
"""


def main():
    if len(sys.argv) != 2:
        sys.stderr.write('用法: python3 prepare_crlf_sample.py <输出文件路径>\n')
        return 1
    path = sys.argv[1]
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    data = CONTENT.encode('utf-8').replace(b'\n', b'\r\n')
    with open(path, 'wb') as handle:
        handle.write(data)
    written = open(path, 'rb').read()
    crlf = written.count(b'\r\n')
    bare_lf = written.count(b'\n') - crlf
    if written != data or bare_lf:
        sys.stderr.write('[ERROR] 生成文件校验失败: CRLF {0} | 裸 LF {1}\n'.format(crlf, bare_lf))
        return 1
    print('[INFO] 已生成 CRLF 样例: {0} (CRLF {1} 行, 无裸 LF)'.format(path, crlf))
    return 0


if __name__ == '__main__':
    sys.exit(main())
