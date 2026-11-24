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
"""文档参考链接版本号更新: 按当前分支映射更新 gitcode.com 与 www.hiascend.com 链接中的版本号。

直接修改文件并输出修改清单 (修改后可用 git diff 复查、git checkout 回退)。
全部处理采用字节级读写与替换, 保持文件原有编码(UTF-8/GBK)与行尾(LF/CRLF)不变。

用法:
    python3 update_links.py [-d <目录|文件>] [-b <版本分支>] [--output <清单文件>]
"""

import argparse
import base64
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# 1. gitcode.com/cann 组织下仓库的 master -> 当前版本分支 (cann/cann-samples、cann/pypto-gym 仓库除外)
#    master 后加 (?![-\w.]) 边界: 仅匹配完整分支段, master-backup/masterDocs 等其他分支不受影响
CANN_LINK_PATTERN = re.compile(rb'(gitcode\.com/cann/(?!cann-samples/)(?!pypto-gym/)'
                              rb'[\w.-]+/(?:blob|tree)/)master(?![\w.-])')
# 2. gitcode.com/cann/pypto-gym 仓库的 master -> 当前分支映射的上一个正式版本
#    (该仓无 beta 分支, beta 分支按分组规则映射到上一个 9.x.0, 如 9.3.0-beta.1 -> 9.2.0)
PYTO_GYM_PATTERN = re.compile(rb'(gitcode\.com/cann/pypto-gym/(?:blob|tree)/)master(?![\w.-])')
# 3. gitcode.com/Ascend 组织下仓库的 master -> 当前 CANN 分支映射的 Ascend 分支
#    (Ascend/docs、Ascend/pytorch 仓库除外)
ASCEND_LINK_PATTERN = re.compile(rb'(gitcode\.com/Ascend/(?!docs/)(?!pytorch/)[\w.-]+/(?:blob|tree)/)master(?![\w.-])')
# 4. gitcode.com/Ascend/pytorch 仓库的 master -> CommonTerms 手册名称汇总表第三列第一个
#    pytorch 链接的分支号 (懒加载, 仅当目标中存在该类链接时才在线查询)
PYTORCH_CODE_PATTERN = re.compile(rb'(gitcode\.com/Ascend/pytorch/(?:blob|tree)/)master(?![\w.-])')
# 5. www.hiascend.com CANN 社区版: latest -> 分支名删除 '.' 与 '-'
HIAS_CANN_PATTERN = re.compile(rb'(www\.hiascend\.com/document/detail/zh/CANNCommunityEdition/)latest(/)')
# 6. www.hiascend.com Pytorch: latest -> Ascend 分支删除 '.'
HIAS_PYTORCH_PATTERN = re.compile(rb'(www\.hiascend\.com/document/detail/zh/Pytorch/)latest(/)')
# 7. www.hiascend.com TensorFlowCommunity: latest -> 分组后的 9.x.0 删除 '.'
HIAS_TF_PATTERN = re.compile(rb'(www\.hiascend\.com/document/detail/zh/TensorFlowCommunity/)latest(/)')

CANN_VERSION_RE = re.compile(r'^(\d+)\.(\d+)\.(\d+)(-beta\.\d+)?$')
PYTORCH_TERMS_PATH = 'FrameworkPTAdapter/CommonTerms/{0} TorchNPU手册名称汇总.md'
PYTORCH_LINK_RE = re.compile(r'gitcode\.com/Ascend/pytorch/(?:blob|tree)/([^/\s)]+)')

LINK_RULES = ('cann', 'cann/pypto-gym', 'Ascend', 'pytorch仓',
              'hiascendCANN', 'hiascendPytorch', 'hiascendTF')

# Ascend/pytorch 映射分支懒加载缓存: 整个运行过程只查询一次
PYTORCH_FETCH_STATE = {'fetched': False, 'repl': None}


def run_git(args):
    """执行 git 命令, 返回去除首尾空白的输出; 失败时返回 None。"""
    try:
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL)
        return out.decode('utf-8', errors='replace').strip()
    except Exception:
        return None


def detect_branch(explicit):
    """解析版本分支: 优先使用显式指定, 否则自动获取当前 git 分支。"""
    branch = explicit
    if not branch:
        branch = run_git(['git', 'branch', '--show-current'])
        if not branch:
            branch = run_git(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    return branch


def branch_to_doc_version(branch):
    """hiascend CANN 社区版文档版本号: 分支名删除 '.' 与 '-'。例如 9.3.0-beta.1 -> 930beta1。"""
    return branch.replace('.', '').replace('-', '')


def cann_to_ascend_branch(branch):
    """CANN 分支映射为对应的 Ascend 分支:
    - 正式版 9.X.0        -> 26.X.0
    - 测试版 9.X.0-beta.N -> 26.(X-1).0
    即 9.x.0 与 9.(x+1).0-beta.* 对应同一 Ascend 分支 (例如 9.2.0 与
    9.3.0-beta.* 均对应 26.2.0)。分支不符合版本号规则时返回 None。
    """
    match = CANN_VERSION_RE.match(branch)
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3))
    is_beta = match.group(4) is not None
    ascend_minor = minor - 1 if is_beta else minor
    if ascend_minor < 0:
        return None
    return '{0}.{1}.{2}'.format(major + 17, ascend_minor, patch)


def branch_to_tf_version(branch):
    """hiascend TensorFlowCommunity 文档版本号: 当前分支 9.x.0 与 9.(x+1).0-beta.* 同组
    对应 TF 版本 9.x.0, 再删除 '.'。例如 9.3.0-beta.1 -> 9.2.0 -> 920。
    """
    match = CANN_VERSION_RE.match(branch)
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2))
    is_beta = match.group(4) is not None
    tf_minor = minor - 1 if is_beta else minor
    if tf_minor < 0:
        return None
    return '{0}.{1}.0'.format(major, tf_minor).replace('.', '')


def branch_to_prev_release(branch):
    """当前分支映射为上一个正式版本(保留版本号格式): beta 9.X.0-beta.N -> 9.(X-1).0,
    正式版 9.X.0 -> 9.X.0。例如 9.3.0-beta.1 -> 9.2.0, 9.2.0 -> 9.2.0。
    用于无 beta 分支的仓(如 cann/pypto-gym)的链接映射。分支不符合版本号规则时返回 None。
    """
    match = CANN_VERSION_RE.match(branch)
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2))
    is_beta = match.group(4) is not None
    ref_minor = minor - 1 if is_beta else minor
    if ref_minor < 0:
        return None
    return '{0}.{1}.0'.format(major, ref_minor)


def fetch_pytorch_branch(ascend_branch):
    """从 Ascend/docs 仓 FrameworkPTAdapter/CommonTerms 下 '<Ascend分支> TorchNPU手册名称汇总.md'
    的表格第三列 (gitcode链接) 中, 取第一个 gitcode.com/Ascend/pytorch/ 链接的分支号。
    例如 Ascend 分支 26.2.0 -> v2.14.0-26.2.0。获取或解析失败时返回 None。
    """
    if not ascend_branch:
        return None
    url = ('https://gitcode.com/api/v5/repos/Ascend/docs/contents/'
           + urllib.parse.quote(PYTORCH_TERMS_PATH.format(ascend_branch)) + '?ref=master')
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(request, timeout=30)
        payload = json.loads(response.read().decode('utf-8'))
        text = base64.b64decode(payload['content']).decode('utf-8')
    except Exception:
        return None
    for line in text.splitlines():
        if not line.startswith('|'):
            continue
        cells = line.split('|')
        if len(cells) <= 3:
            continue
        match = PYTORCH_LINK_RE.search(cells[3])
        if match:
            return match.group(1)
    return None


def get_pytorch_repl(ascend_branch, warnings):
    """懒加载并缓存 Ascend/pytorch 映射分支的替换模板。"""
    if not PYTORCH_FETCH_STATE['fetched']:
        PYTORCH_FETCH_STATE['fetched'] = True
        pytorch_branch = fetch_pytorch_branch(ascend_branch)
        if pytorch_branch:
            PYTORCH_FETCH_STATE['repl'] = b'\\g<1>' + pytorch_branch.encode('utf-8')
        else:
            warnings.append('无法从 Ascend/docs 仓 CommonTerms 获取 Ascend/pytorch 映射分支, '
                            '相关链接保持 master')
    return PYTORCH_FETCH_STATE['repl']


def build_link_rules(branch):
    """按分支构建链接替换规则表 [(规则名, 已编译模式, 替换模板 or None)]。

    替换模板使用 \\g<1>/\\g<2> 组引用; 为 None 表示该规则依赖的版本映射失败或需懒加载。
    """
    ascend_branch = cann_to_ascend_branch(branch)
    tf_version = branch_to_tf_version(branch)
    prev_release = branch_to_prev_release(branch)
    rules = [
        ('cann', CANN_LINK_PATTERN, b'\\g<1>' + branch.encode('utf-8')),
        ('cann/pypto-gym', PYTO_GYM_PATTERN,
         b'\\g<1>' + prev_release.encode('utf-8') if prev_release else None),
        ('Ascend', ASCEND_LINK_PATTERN,
         b'\\g<1>' + ascend_branch.encode('utf-8') if ascend_branch else None),
        ('pytorch仓', PYTORCH_CODE_PATTERN, None),  # 懒加载
        ('hiascendCANN', HIAS_CANN_PATTERN,
         b'\\g<1>' + branch_to_doc_version(branch).encode('utf-8') + b'\\g<2>'),
        ('hiascendPytorch', HIAS_PYTORCH_PATTERN,
         b'\\g<1>' + ascend_branch.replace('.', '').encode('utf-8') + b'\\g<2>'
         if ascend_branch else None),
        ('hiascendTF', HIAS_TF_PATTERN,
         b'\\g<1>' + tf_version.encode('utf-8') + b'\\g<2>' if tf_version else None),
    ]
    return rules, ascend_branch


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


def process_file(path, branch, link_rules, ascend_branch, warnings):
    """处理单个文件的链接更新, 返回 (新数据, 是否有变化, 文件级统计)。不写回。"""
    with open(path, 'rb') as handle:
        original = handle.read()
    data = original
    stats = {'links': {}, 'link_pairs': {}, 'checked_urls': set()}

    for name, pattern, repl in link_rules:
        if not pattern.search(data):
            continue
        if name == 'pytorch仓' and repl is None:
            # 懒加载: 仅当目标中实际存在该类链接时才在线查询 CommonTerms (运行级缓存)
            repl = get_pytorch_repl(ascend_branch, warnings)
            if repl is None:
                continue
        elif repl is None:
            warnings.append('规则 {0} 无法映射 (分支 {1} 不符合版本号规则), 相关链接保持原样'
                            .format(name, branch))
            continue
        count = 0
        for m in pattern.finditer(data):
            after = m.expand(repl)
            key = (name, m.group(0), after)
            stats['link_pairs'][key] = stats['link_pairs'].get(key, 0) + 1
            stats['checked_urls'].add(extract_full_url(data, m.start(), m.end(), after))
            count += 1
        if count:
            stats['links'][name] = stats['links'].get(name, 0) + count
            data = pattern.sub(repl, data)

    return data, data != original, stats


# =========================================================================================================
# 修改后链接有效性校验
# =========================================================================================================
# URL 向前识别的 scheme 前缀
URL_SCHEMES = (b'https://', b'http://')
# URL 终止字符: 空白 ) " ' < > | ` ] 以及非 ASCII 字节
URL_STOP_BYTES = b' \t\r\n)"\'<>|`]'


def extract_full_url(data, match_start, match_end, replacement):
    """提取匹配所在的完整 URL, 返回替换后的完整 URL (bytes)。

    自匹配处向后扫描至链接终止字符, 向前识别 https:// 或 http:// 前缀;
    保留普通路径、查询串 (?...)、片段 (#...) 及 Markdown/HTML 包装的完整形态。
    """
    end = match_end
    while end < len(data):
        byte = data[end]
        if byte in URL_STOP_BYTES or byte > 126 or byte < 33:
            break
        end += 1
    start = match_start
    for scheme in URL_SCHEMES:
        pos = match_start - len(scheme)
        if pos >= 0 and data[pos:match_start] == scheme:
            start = pos
            break
    return data[start:match_start] + replacement + data[match_end:end]


GITCODE_URL_RE = re.compile(
    r'https?://gitcode\.com/([\w.-]+)/([\w.-]+)/(?:blob|tree)/([^/?#\s]+)(?:[/?#]\S*)?')

# ---- WAF 突发限流防护: 全局限速 + 429/403 退避重试 ----
# 相邻请求起始的最小间隔 (秒), 全局生效, 约束 API 请求总速率不超过 ~10 req/s
REQUEST_MIN_INTERVAL = 0.1
# 429 最多重试次数, 退避时长为 1s/2s/4s 再加随机抖动
RATE_LIMIT_RETRIES = 3
# Retry-After 响应头的等待上限 (秒), 防止异常大的响应值阻塞整个校验
RATE_LIMIT_MAX_SLEEP = 30.0

_throttle_lock = threading.Lock()
_throttle_next_start = [0.0]


def throttle_request():
    """全局限速: 锁内预约本请求的起始时刻, 保证相邻请求起始间隔不小于
    REQUEST_MIN_INTERVAL; 等待在锁外 sleep, 不阻塞其他线程预约。
    """
    with _throttle_lock:
        now = time.monotonic()
        start = max(now, _throttle_next_start[0] + REQUEST_MIN_INTERVAL)
        _throttle_next_start[0] = start
    if start > now:
        time.sleep(start - now)


def retry_after_seconds(error):
    """解析 429 响应的 Retry-After 头 (仅支持秒数值): 无头或无法解析返回 None,
    交由指数退避兜底; 解析结果按 RATE_LIMIT_MAX_SLEEP 封顶。
    """
    if not error.headers:
        return None
    try:
        return max(0.0, min(float(error.headers.get('Retry-After', '')),
                            RATE_LIMIT_MAX_SLEEP))
    except ValueError:
        return None


def check_link(url):
    """校验单个修改后链接的有效性, 返回 (状态, 说明)。

    gitcode.com 链接通过 GitCode API 检查分支与路径是否存在 (200 -> 有效,
    404 -> 分支或路径不存在): 有路径时用 contents API, 无路径 (仓库根) 时
    contents API 不校验 ref, 改用 branches API 校验分支存在性;
    www.hiascend.com 为 SPA 站点, HTTP 状态不反映内容存在性, 跳过校验。
    无 scheme 前缀的 gitcode.com / www.hiascend.com 链接 (替换与 URL 提取均
    不强制 scheme), 补 https:// 前缀后按上述规则正常校验/跳过。

    每次 API 请求 (含重试) 均先经全局限速; 429 优先按 Retry-After 头等待,
    否则指数退避 (1s/2s/4s + 抖动) 重试, 耗尽报 '限流'; 403 (WAF 拦截页
    常见) 延迟重试一次, 仍失败按 '不可达' 处理。
    """
    text = url.decode('utf-8', errors='replace')
    if text.startswith('gitcode.com/') or text.startswith('www.hiascend.com/'):
        text = 'https://' + text
    if text.startswith('https://www.hiascend.com') or text.startswith('http://www.hiascend.com'):
        return ('跳过', 'hiascend 为 SPA 站点, HTTP 状态不反映内容存在性')
    match = GITCODE_URL_RE.match(text)
    if not match:
        return ('格式异常', '链接格式无法解析')
    org, repo, branch = match.group(1), match.group(2), match.group(3)
    rest = text[match.end(3):]
    for sep in ('#', '?'):
        rest = rest.split(sep, 1)[0]
    path = rest.lstrip('/')
    if path:
        api = ('https://gitcode.com/api/v5/repos/{0}/{1}/contents/{2}?ref={3}'.format(
            org, repo, urllib.parse.quote(path, safe='/%'), branch))
    else:
        api = 'https://gitcode.com/api/v5/repos/{0}/{1}/branches/{2}'.format(org, repo, branch)

    rate_limit_retries = 0
    forbidden_retried = False
    while True:
        try:
            request = urllib.request.Request(api, headers={'User-Agent': 'Mozilla/5.0'})
            throttle_request()
            response = urllib.request.urlopen(request, timeout=10)
            response.read(256)
            response.close()
            return ('OK', '')
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return ('404', '分支或路径不存在')
            if error.code == 429:
                if rate_limit_retries >= RATE_LIMIT_RETRIES:
                    return ('限流', '重试 {0} 次后仍 HTTP 429'.format(rate_limit_retries))
                delay = retry_after_seconds(error)
                if delay is None:
                    delay = (1 << rate_limit_retries) + random.uniform(0, 0.5)
                time.sleep(delay)
                rate_limit_retries += 1
                continue
            if error.code == 403 and not forbidden_retried:
                forbidden_retried = True
                time.sleep(1.0 + random.uniform(0, 0.5))
                continue
            return ('不可达', 'HTTP {0}'.format(error.code))
        except Exception as error:
            return ('不可达', str(error)[:60])


def check_all_links(urls):
    """并发校验全部唯一链接 (全局限速 + 限流退避重试), 返回 {url: (状态, 说明)},
    按链接排序保证输出稳定。
    """
    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {url: executor.submit(check_link, url) for url in urls}
        for url in sorted(urls):
            results[url] = future_map[url].result()
    return results


def render_link_check(results):
    """渲染链接有效性校验段落, 问题链接在前。"""
    if not results:
        return []
    order = {'404': 0, '格式异常': 1, '限流': 2, '不可达': 3, '跳过': 4, 'OK': 5}
    lines = ['', '---- 链接有效性校验 ({0} 个唯一链接) ----'.format(len(results))]
    sorted_results = sorted(results.items(), key=lambda item: (order.get(item[1][0], 9), item[0]))
    for url, (status, detail) in sorted_results:
        suffix = ' ({0})'.format(detail) if detail else ''
        lines.append('[{0}] {1}{2}'.format(status, url.decode('utf-8', errors='replace'), suffix))
    counts = {}
    for _, (status, _) in results.items():
        counts[status] = counts.get(status, 0) + 1
    lines.append('汇总: ' + ', '.join('{0} {1}'.format(k, v) for k, v in sorted(counts.items())))
    if '404' in counts:
        lines.append('[WARN] 存在无效链接 (404), 请人工核查分支或路径是否正确')
    if '限流' in counts:
        lines.append('[WARN] {0} 个链接因接口限流未能完成校验, 建议稍后重跑复查'
                     .format(counts['限流']))
    return lines


def render_manifest(target, branch, ascend_branch, file_stats, warnings):
    """渲染修改清单文本。"""
    lines = []
    lines.append('=' * 24 + ' 链接版本更新修改清单 ' + '=' * 24)
    lines.append('分支: {0}'.format(branch))
    extras = []
    if ascend_branch:
        extras.append('Ascend 分支: {0}'.format(ascend_branch))
    doc_version = branch_to_doc_version(branch)
    if doc_version:
        extras.append('文档版本号: {0}'.format(doc_version))
    tf_version = branch_to_tf_version(branch)
    if tf_version:
        extras.append('Pytorch 文档: {0} | TF 文档: {1}'
                      .format(ascend_branch.replace('.', '') if ascend_branch else '-',
                              tf_version))
    if extras:
        lines.append('  ' + ' | '.join(extras))
    lines.append('目标: {0}'.format(target))
    lines.append('')

    changed = [fs for fs in file_stats if fs[1]['links']]
    lines.append('---- 修改文件 ({0}) ----'.format(len(changed)))
    for path, stats in changed:
        lines.append('{0}    链接更新: {1}'.format(
            path, ', '.join(
                '{0} {1}'.format(k, v)
                for k, v in sorted(stats['links'].items()))))

    pair_counts = {}
    for _, stats in file_stats:
        for key, count in stats['link_pairs'].items():
            pair_counts[key] = pair_counts.get(key, 0) + count
    if pair_counts:
        lines.append('')
        lines.append('---- 链接替换明细 (去重) ----')
        for (name, before, after), count in sorted(pair_counts.items()):
            lines.append('[{0}] {1} -> {2} ({3} 处)'.format(
                name, before.decode('utf-8', errors='replace'),
                after.decode('utf-8', errors='replace'), count))

    lines.append('')
    lines.append('---- 汇总 ----')
    total_links = sum(sum(s['links'].values()) for _, s in file_stats)
    if total_links:
        detail = ', '.join('{0} {1}'.format(
            name, sum(s['links'].get(name, 0) for _, s in file_stats))
            for name in LINK_RULES
            if any(s['links'].get(name) for _, s in file_stats))
        lines.append('修改文件: {0} | 链接替换: {1} 处 ({2})'
                     .format(len(changed), total_links, detail))

    if warnings:
        lines.append('')
        lines.append('---- 告警 ----')
        for warning in warnings:
            lines.append('[WARN] ' + warning)

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='文档参考链接版本号更新: 按当前分支映射更新 gitcode/hiascend 链接中的版本号, '
                    '直接修改并输出修改清单')
    parser.add_argument('-d', '--dir', metavar='DIR', default=None,
                        help='目标目录或单个 .md 文件 (默认: <仓库根>/docs/zh)')
    parser.add_argument('-b', '--branch', metavar='BRANCH', default=None,
                        help='目标版本分支 (默认: 当前 git 分支)')
    parser.add_argument('--output', metavar='FILE', default=None,
                        help='将修改清单写入指定文件')
    args = parser.parse_args()

    repo_root = run_git(['git', 'rev-parse', '--show-toplevel'])

    target = args.dir
    if not target:
        if repo_root:
            target = os.path.join(repo_root, 'docs', 'zh')
        else:
            target = os.path.join('docs', 'zh')
    if not os.path.exists(target):
        sys.stderr.write('[ERROR] 目标不存在: {0}\n'.format(target))
        sys.exit(1)

    branch = detect_branch(args.branch)
    if not branch:
        sys.stderr.write('[ERROR] 无法自动获取当前分支, 请通过 -b 指定版本分支\n')
        sys.exit(1)
    if branch in ('master', 'main', 'HEAD'):
        sys.stderr.write("[ERROR] 当前分支为 '{0}', 链接版本更新无意义, "
                         '请切换到版本分支或通过 -b 指定\n'.format(branch))
        sys.exit(1)

    link_rules, ascend_branch = build_link_rules(branch)
    files = collect_md_files(target)
    warnings = []
    file_stats = []

    for path in files:
        new_data, changed, stats = process_file(path, branch, link_rules,
                                                 ascend_branch, warnings)
        file_stats.append((display_relpath(path), stats))
        if changed:
            with open(path, 'wb') as handle:
                handle.write(new_data)

    manifest = render_manifest(target, branch, ascend_branch, file_stats, warnings)

    # 修改完成后, 对本次修改产生的所有唯一链接做在线有效性校验
    checked_urls = set()
    for _, stats in file_stats:
        checked_urls.update(stats['checked_urls'])
    if checked_urls:
        print('[INFO] 正在校验 {0} 个修改后链接的有效性...'.format(len(checked_urls)),
              file=sys.stderr)
        results = check_all_links(checked_urls)
        manifest += '\n'.join(render_link_check(results))
    print(manifest)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            handle.write('# 链接版本更新修改清单\n\n```\n')
            handle.write(manifest)
            handle.write('\n```\n')
        print('')
        print('[INFO] 修改清单已写入: {0}'.format(args.output))

    return 0


if __name__ == '__main__':
    sys.exit(main())
