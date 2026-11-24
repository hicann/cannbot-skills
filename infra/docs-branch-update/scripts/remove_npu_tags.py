#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""删除指定 NPU 标签及其内容: 删除不必要的产品内容, 并输出修改清单供确认。

规则:
1. 标签类型列表仅为目标类型(单一类型)的块整块删除;
   id 反向引用保证嵌套时配对到正确的 end 标签(内层块的 end id 不会被误当作外层结束);
   行首引用前缀(>)及缩进随块整行删除; \r?\n? 兼容 LF/CRLF 行尾。
2. 混合类型标签(如 <!-- npu="950,A3,x90" idX -->): 从列表中移除目标类型, 保留块内容;
   移除后列表为空时(含多类型逐个移除后为空)整块删除, 由各类型顺序处理自然覆盖;
   混合块内部嵌套的目标类型子块由规则 1 删除, 外层块按规则 2 处理后保留。
   混合改写统计按最终数据对账, 仅计最终存活的改写(已整删块不计, 避免中间态虚增);
   被改写多次的块沿改写链合并, 明细报告原始列表 -> 最终列表。
3. 删除接缝处若产生冗余 <br> 或冗余空行, 同步清理(局部生效):
   - 接缝处 3 个及以上连续 <br>(含中间空白)视为删除残留, 整体删除;
   - 接缝处 2 个及以上 <br> 紧跟表格单元格分隔符 | 时, 收敛为单个空格;
   - 接缝处 3 个及以上连续空行(含仅空白字符的行)收敛为 1 个空行, 行尾风格保持不变。
   仅处理与删除接缝相邻(含跨越)的序列, 不处理文件其他位置的既有 <br>/空行,
   避免误删与标签无关的正常内容。

直接修改文件并输出修改清单 (修改后可用 git diff 复查、git checkout 回退)。
全部处理采用字节级读写与替换, 保持文件原有编码(UTF-8/GBK)与行尾(LF/CRLF)不变。

待删除标签缺省范围在线获取: 从 cann/docs 仓 specification/document_writing_specs.md 的
"分支文档维护要求"章节, 取"产品标签使用要求"列表项 (第一个无序列表项) 中不允许存在的
标签范围 (不允许存在`...` 标签内容) 作为默认删除范围; 仅在未显式指定 --npu 且需要删除
产品内容时查询一次, 获取失败时退出并提示显式指定 --npu。

用法:
    python3 remove_npu_tags.py [-d <目录|文件>] [--npu 类型...] [--output <清单文件>]
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request

NPU_START_TAG_RE = re.compile(rb'<!-- npu="([^"]+)" id(\d+) -->')

# 默认删除标签范围的在线获取来源:
# cann/docs 仓文档写作规范 "分支文档维护要求" 章节, "产品标签使用要求"列表项中
# 不允许存在的标签范围 = '不允许存在`...` 标签内容' 中的标签列表
SPEC_PATH = 'specification/document_writing_specs.md'
DISALLOWED_TAG_RE = re.compile(r'不允许存在`([^`]+)`')


def fetch_disallowed_npu_types():
    """从 cann/docs 仓 specification/document_writing_specs.md 的"分支文档维护要求"章节,
    取"产品标签使用要求"列表项 (该章节第一个无序列表项) 中不允许存在的标签范围,
    即 '不允许存在`...` 标签内容' 中的标签列表。
    例如 -> ['IPV350', '5162A', 'x90', '9020', '9030', '9050', 'kirin1', 'kirin2', 'kirin3']。
    获取或解析失败时返回 None。
    """
    url = ('https://gitcode.com/api/v5/repos/cann/docs/contents/'
           + urllib.parse.quote(SPEC_PATH) + '?ref=master')
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(request, timeout=30)
        payload = json.loads(response.read().decode('utf-8'))
        text = base64.b64decode(payload['content']).decode('utf-8')
    except Exception:
        return None
    in_section = False
    section_level = 0
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('#'):
            level = len(stripped) - len(stripped.lstrip('#'))
            if in_section and level <= section_level:
                break  # 已离开目标章节
            if '分支文档维护要求' in stripped:
                in_section = True
                section_level = level
            continue
        if in_section and stripped.startswith('- '):
            # 第一个无序列表项: 产品标签使用要求
            match = DISALLOWED_TAG_RE.search(stripped)
            if match:
                return [item.strip() for item in match.group(1).split(',') if item.strip()]
            return None
    return None

# 仅在删除接缝处生效的冗余 <br> / 冗余空行清理规则
BR_RUN_RE = re.compile(rb'(?:<br\s*/?>[ \t]*){3,}')
BR_TABLE_RE = re.compile(rb'(?:<br\s*/?>[ \t]*){2,}\|')
BLANK_RUN_RE = re.compile(rb'(?:\r?\n[ \t]*){2,}\r?\n')
BR_WINDOW = 100
BLANK_WINDOW = 200


def clean_residual_br_at(data, junction, stats):
    """在删除接缝处局部清理冗余 <br> 序列, 仅处理与接缝相邻(含跨越)的序列。

    返回 (清理后的数据, 接缝的新位置)。
    """
    while True:
        lo = max(0, junction - BR_WINDOW)
        hi = min(len(data), junction + BR_WINDOW)
        if hi <= lo:
            return data, junction
        window = data[lo:hi]
        offset = junction - lo
        hit = None
        for m in BR_RUN_RE.finditer(window):
            if m.start() <= offset <= m.end():
                hit = (m, b'')
                break
        if hit is None:
            for m in BR_TABLE_RE.finditer(window):
                if m.start() <= offset <= m.end():
                    hit = (m, b' |')
                    break
        if hit is None:
            return data, junction
        match, replacement = hit
        data = data[:lo + match.start()] + replacement + data[lo + match.end():]
        junction = lo + match.start() + len(replacement)
        stats['seam_br'] += 1


def collapse_blank_at(data, junction, stats):
    """在删除接缝处局部收敛冗余空行, 仅处理与接缝相邻(含跨越)的空行序列。

    3 个及以上连续空行(含仅空白字符的行)收敛为 1 个空行, 行尾风格保持不变。
    返回 (清理后的数据, 接缝的新位置)。
    """
    while True:
        lo = max(0, junction - BLANK_WINDOW)
        hi = min(len(data), junction + BLANK_WINDOW)
        if hi <= lo:
            return data, junction
        window = data[lo:hi]
        offset = junction - lo
        hit = None
        for m in BLANK_RUN_RE.finditer(window):
            if m.start() <= offset <= m.end():
                hit = m
                break
        if hit is None:
            return data, junction
        eol = b'\r\n' if b'\r\n' in hit.group(0) else b'\n'
        replacement = eol * 2
        data = data[:lo + hit.start()] + replacement + data[lo + hit.end():]
        junction = lo + hit.start() + len(replacement)
        stats['seam_blank'] += 1


def remove_single_blocks(data, npu_type, stats):
    """删除类型列表仅为 npu_type 的标签块, 并在各删除接缝处局部清理冗余 <br> 与空行。

    id 反向引用保证嵌套时配对到正确的 end 标签; 行首引用前缀(>)及缩进随块整行删除;
    \\r?\\n? 兼容 LF/CRLF 行尾。返回 (新数据, 删除块数)。
    """
    pattern = re.compile(
        (rb'(?m)(?:^[ \t]*(?:>[ \t]*)*)?'
         rb'<!-- npu="' + re.escape(npu_type.encode('utf-8')) + rb'" id(\d+) -->'
         rb'.*?<!-- end id\1 -->\r?\n?'),
        re.DOTALL)
    matches = list(pattern.finditer(data))
    if not matches:
        return data, 0
    pieces = []
    junctions = []
    running = 0
    pos = 0
    for m in matches:
        piece = data[pos:m.start()]
        pieces.append(piece)
        running += len(piece)
        junctions.append(running)
        pos = m.end()
    pieces.append(data[pos:])
    result = b''.join(pieces)
    # 自左向右在各接缝处局部清理冗余 <br> 与空行, 并平移后续接缝位置。
    # 注意: adjust 必须累计清理导致的文本收缩量 (清理前后长度差), 而非接缝点位移。
    # 接缝落在匹配内部时 (被删块前后均有空行/br), 匹配尾部超出接缝的部分会使
    # 接缝点位移小于实际收缩量, 按位移累计将导致后续接缝向右漂移 (每次漂移量 =
    # 匹配尾部超出接缝的长度), 造成真实接缝处漏清理、或错位窗口误清理既有格式。
    adjust = 0
    for junction in junctions:
        junction += adjust
        length_before = len(result)
        result, new_junction = clean_residual_br_at(result, junction, stats)
        result, new_junction = collapse_blank_at(result, new_junction, stats)
        adjust -= length_before - len(result)
    return result, len(matches)


def rewrite_mixed_tags(data, npu_type, stats):
    """混合类型标签: 从类型列表中移除 npu_type, 保留块内容。

    改写逐条记入 mixed_records, 每条为 (原列表, 新列表, 原起始标签, 新起始标签),
    不在此处计数; 移除后列表为空的场景由各类型顺序处理时的整块删除覆盖,
    最终统计由 process_file 按最终数据对账, 仅保留最终存活的改写。
    """
    def rewrite(match):
        attr = match.group(1).decode('utf-8', errors='replace')
        items = [item.strip() for item in attr.split(',')]
        if npu_type not in items or len(items) <= 1:
            return match.group(0)
        items.remove(npu_type)
        new_attr = ','.join(items)
        new_tag = match.group(0).replace(match.group(1), new_attr.encode('utf-8'))
        stats['mixed_records'].append((attr, new_attr, match.group(0), new_tag))
        return new_tag
    return NPU_START_TAG_RE.sub(rewrite, data)


def collect_md_files(target):
    """收集目标目录(递归)下的全部 .md 文件; 目标为单个文件时校验后缀。"""
    if os.path.isfile(target):
        if os.path.splitext(target)[1].lower() != '.md':
            sys.stderr.write('[ERROR] 目标文件不是 .md 文件: {0}\n'.format(target))
            sys.exit(1)
        return [target]
    files = []
    for root, _, names in os.walk(target):
        for name in names:
            if os.path.splitext(name)[1].lower() == '.md':
                files.append(os.path.join(root, name))
    files.sort()
    return files


def display_relpath(path):
    """相对当前目录的展示路径; Windows 跨盘符等无法取相对路径时回退绝对路径。"""
    try:
        return os.path.relpath(path)
    except ValueError:
        return os.path.abspath(path)


def process_file(path, npu_types):
    """处理单个文件的 NPU 标签删除, 返回 (新数据, 是否有变化, 文件级统计)。不写回。"""
    with open(path, 'rb') as handle:
        original = handle.read()
    data = original
    stats = {
        'npu_deleted': {},    # 类型 -> 删除块数
        'npu_mixed': {},      # (原列表, 新列表) -> 数量, 对账后仅含最终存活块
        'mixed_records': [],  # 逐条改写记录 (原列表, 新列表, 原起始标签, 新起始标签)
        'seam_br': 0,
        'seam_blank': 0,
    }
    for npu_type in npu_types:
        data, deleted = remove_single_blocks(data, npu_type, stats)
        if deleted:
            stats['npu_deleted'][npu_type] = stats['npu_deleted'].get(npu_type, 0) + deleted
        data = rewrite_mixed_tags(data, npu_type, stats)
    # 混合改写对账: 逐类型顺序处理会产生中间态 (如 x90,9030 移除 x90 后暂为 npu="9030",
    # 记一条 x90,9030 -> 9030 的改写, 随即被 9030 类型整块删除)。统计以最终数据为准:
    # 仅保留新起始标签仍存在于数据中的改写, 已整删块不计入混合改写; 被改写多次的块
    # 沿改写链回溯合并, 明细报告原始列表 -> 最终列表, 不出现中间态列表。
    survivors = [list(record) for record in stats['mixed_records'] if record[3] in data]
    dropped = [record for record in stats['mixed_records'] if record[3] not in data]
    merged = True
    while merged:
        merged = False
        for record in dropped:
            for survivor in survivors:
                if survivor[2] == record[3]:
                    survivor[0] = record[0]
                    survivor[2] = record[2]
                    dropped.remove(record)
                    merged = True
                    break
            if merged:
                break
    for old_attr, new_attr, _, _ in survivors:
        key = (old_attr, new_attr)
        stats['npu_mixed'][key] = stats['npu_mixed'].get(key, 0) + 1
    del stats['mixed_records']
    return data, data != original, stats


def render_manifest(target, npu_types, npu_source, file_stats):
    """渲染修改清单文本。"""
    lines = []
    lines.append('=' * 24 + ' NPU 标签删除修改清单 ' + '=' * 24)
    lines.append('目标: {0} | NPU 类型: {1} (来源: {2})'.format(
        target, ', '.join(npu_types), npu_source))
    lines.append('')

    changed = \
    [
        fs
        for fs in file_stats
        if fs[1]['npu_deleted'] or fs[1]['npu_mixed'] or fs[1]['seam_br'] or fs[1]['seam_blank']
    ]
    lines.append('---- 修改文件 ({0}) ----'.format(len(changed)))
    for path, stats in changed:
        lines.append(path)
        if stats['npu_deleted']:
            lines.append('    NPU 删除: ' + ', '.join(
                '{0} {1}'.format(k, v)
                for k, v in sorted(stats['npu_deleted'].items())))
        if stats['npu_mixed']:
            lines.append('    NPU 混合改写: {0} 处'.format(sum(stats['npu_mixed'].values())))
        if stats['seam_br'] or stats['seam_blank']:
            lines.append('    接缝清理: <br> {0} 处, 空行 {1} 处'
                         .format(stats['seam_br'], stats['seam_blank']))

    attr_counts = {}
    for _, stats in file_stats:
        for key, count in stats['npu_mixed'].items():
            attr_counts[key] = attr_counts.get(key, 0) + count
    if attr_counts:
        lines.append('')
        lines.append('---- NPU 标签列表改写明细 (去重) ----')
        for (before, after), count in sorted(attr_counts.items()):
            lines.append('npu="{0}" -> npu="{1}" ({2} 处)'.format(before, after, count))

    lines.append('')
    lines.append('---- 汇总 ----')
    total_deleted = sum(sum(s['npu_deleted'].values()) for _, s in file_stats)
    total_mixed = sum(sum(s['npu_mixed'].values()) for _, s in file_stats)
    total_br = sum(s['seam_br'] for _, s in file_stats)
    total_blank = sum(s['seam_blank'] for _, s in file_stats)
    lines.append('修改文件: {0} | NPU 删除: {1} 块 | 混合改写: {2} 处 | 接缝清理: <br> {3} 处, 空行 {4} 处'
                 .format(len(changed), total_deleted, total_mixed, total_br, total_blank))

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='删除指定 NPU 标签及其内容 (不必要的产品内容), 直接修改并输出修改清单')
    parser.add_argument('-d', '--dir', metavar='DIR', default=None,
                        help='目标目录或单个 .md 文件 (默认: <仓库根>/docs/zh)')
    parser.add_argument('--npu', metavar='TYPE', nargs='+', default=None,
                        help='待删除的 NPU 标签类型, 可多个 (默认: 从 cann/docs 仓文档写作规范 '
                             '"分支文档维护要求-产品标签使用要求" 在线获取不允许存在的标签范围, '
                             '仅查询一次)')
    parser.add_argument('--output', metavar='FILE', default=None,
                        help='将修改清单写入指定文件')
    args = parser.parse_args()

    repo_root = None
    try:
        out = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'],
                                      stderr=subprocess.DEVNULL)
        repo_root = out.decode('utf-8', errors='replace').strip()
    except Exception:
        repo_root = None

    target = args.dir
    if not target:
        if repo_root:
            target = os.path.join(repo_root, 'docs', 'zh')
        else:
            target = os.path.join('docs', 'zh')
    if not os.path.exists(target):
        sys.stderr.write('[ERROR] 目标不存在: {0}\n'.format(target))
        sys.exit(1)

    files = collect_md_files(target)

    # 待删除标签缺省范围: 仅在未显式指定 --npu 时在线查询一次
    npu_types = args.npu
    npu_source = '显式指定'
    if not npu_types:
        npu_types = fetch_disallowed_npu_types()
        npu_source = '文档写作规范"分支文档维护要求-产品标签使用要求"在线获取'
        if not npu_types:
            sys.stderr.write('[ERROR] 无法从文档写作规范获取默认删除的芯片标签范围 '
                             '(网络或规范文件不可用), 请通过 --npu 显式指定\n')
            sys.exit(1)

    file_stats = []

    for path in files:
        new_data, changed, stats = process_file(path, npu_types)
        file_stats.append((display_relpath(path), stats))
        if changed:
            with open(path, 'wb') as handle:
                handle.write(new_data)

    manifest = render_manifest(target, npu_types, npu_source, file_stats)
    print(manifest)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            handle.write('# NPU 标签删除修改清单\n\n```\n')
            handle.write(manifest)
            handle.write('\n```\n')
        print('')
        print('[INFO] 修改清单已写入: {0}'.format(args.output))

    return 0


if __name__ == '__main__':
    sys.exit(main())
