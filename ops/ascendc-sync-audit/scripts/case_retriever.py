#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""
历史 case 检索器 — 从 sync_cases.db 检索与目标代码相似的历史同步修复 PR。

自动将脚本候选与历史 case 配对，输出修复模式证据，替代 LLM 推理否决。

用法:
  python3 case_retriever.py <目标文件> [--format text|json]
  python3 case_retriever.py <目标文件> --candidates '<候选 JSON>'
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from sync_logging import init_logging

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sync_cases.db')
JSONL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sync_cases.jsonl')

_LOGGER = logging.getLogger('case_retriever')
_STDERR_LOGGER = logging.getLogger('case_retriever.stderr')
init_logging(_LOGGER, _STDERR_LOGGER)


def _ensure_db() -> None:
    """db 为生成物（不入版本控制）：缺失且 jsonl 存在时自动重建（复用 build_db.py）。

    sync_cases.jsonl 是 source of truth（随仓库提交），db 由它构建——合入上游后
    首次运行自动完成，无需手动步骤。
    """
    if os.path.exists(DB_PATH):
        return
    if not os.path.exists(JSONL_PATH):
        return  # jsonl 也不在（异常环境），由调用方降级处理
    try:
        import importlib.util
        build_script = os.path.join(os.path.dirname(DB_PATH), 'build_db.py')
        spec = importlib.util.spec_from_file_location('build_db', build_script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.build()
        _STDERR_LOGGER.warning("[case_retriever] sync_cases.db 缺失，已从 sync_cases.jsonl 自动重建")
    except Exception as e:
        _STDERR_LOGGER.warning("[case_retriever] sync_cases.db 自动重建失败: %s；"
                                     "可手动执行 python3 data/build_db.py", e)


_ensure_db()

SYNC_APIS = ['SetFlag', 'WaitFlag', 'PipeBarrier', 'SyncFunc', 'SyncAll',
             'CrossCoreSetFlag', 'CrossCoreWaitFlag', 'SetAtomicAdd',
             'EnQue', 'DeQue', 'DataCopy', 'DataCopyPad']

VAR_PATTERN = re.compile(r'\b(\w*(?:LoopIdx|BufId|BufIdx|LoopCnt|PingPong)\w*)\b')

# 修复模式描述
FIX_PATTERNS = {
    'buffer_index_mismatch': {
        'description': '同步 flag 索引变量与 buffer 索引变量不同源',
        'fix_hint': '将 buffer 索引统一为 flag 索引变量（如 biasBufId_ = scaleL1BufId）',
        'sync_codes': ['SYNC-14'],
    },
    'atomic_ordering': {
        'description': 'SetAtomicAdd 后缺 PipeBarrier<PIPE_FIX>，atomic 乱序覆盖累加结果',
        'fix_hint': '在 SetAtomicAdd 后、Fixpipe/DataCopy 写 GM 前加 PipeBarrier<PIPE_FIX>()',
        'sync_codes': ['SYNC-13'],
    },
    'pipe_barrier_missing': {
        'description': '异步搬出后缺 PipeBarrier，读到半成品数据',
        'fix_hint': '在异步搬出后、SetFlag 释放 buffer 前加 PipeBarrier<PIPE_ALL>()',
        'sync_codes': ['SYNC-02'],
    },
    'flag_pair': {
        'description': 'SetFlag/WaitFlag 配对缺失或不对称',
        'fix_hint': '补齐缺失的 SetFlag 或 WaitFlag，确保所有路径都触发 SetFlag',
        'sync_codes': ['SYNC-04', 'SYNC-08'],
    },
    'cross_core': {
        'description': 'CrossCoreSetFlag/WaitFlag 配对不对称或 flagId 冲突',
        'fix_hint': '确保每个 WaitFlag 有对应 SetFlag，所有路径都触发 SetFlag',
        'sync_codes': ['SYNC-03', 'SYNC-04'],
    },
    'missing_sync': {
        'description': '跨流水数据依赖缺同步（DataCopy 后直接计算）',
        'fix_hint': '在 DataCopy 后、Compute 前加 EnQue/DeQue 或 SetFlag/WaitFlag',
        'sync_codes': ['SYNC-02'],
    },
    'sync_fix': {
        'description': '同步逻辑修复（方向、顺序、粒度等）',
        'fix_hint': '参考历史 PR 的 diff 修复方式',
        'sync_codes': ['SYNC-01', 'SYNC-02', 'SYNC-05', 'SYNC-06'],
    },
}


def extract_features(filepath: str) -> Dict:
    """从目标文件提取特征"""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    content_lower = content.lower()
    
    apis = [api for api in SYNC_APIS if api.lower() in content_lower]
    vars_found = list(set(VAR_PATTERN.findall(content)))[:20]

    keywords = _extract_keywords(content_lower)
    return {
        'apis': apis,
        'vars': vars_found,
        'keywords': keywords,
    }


KEYWORD_RULES = (
    ('atomic', ('atomic', 'setatomicadd')),
    ('pipebarrier', ('pipebarrier',)),
    ('setflag', ('setflag',)),
    ('waitflag', ('waitflag',)),
    ('crosscore', ('crosscore',)),
    ('mte3', ('mte3',)),
    ('mte2', ('mte2',)),
    ('bias', ('bias',)),
    ('scale', ('scale',)),
    ('l0', ('l0c', 'l0a', 'l0b')),
    ('pingpong', ('pingpong', 'double buffer')),
    ('fixpipe', ('fixpipe',)),
    ('mmad', ('mmad',)),
)


def _extract_keywords(content_lower: str) -> List[str]:
    keywords = [kw for kw, terms in KEYWORD_RULES if any(term in content_lower for term in terms)]
    if 'buffer' in content_lower and ('id' in content_lower or 'index' in content_lower):
        keywords.append('buffer_index')
    return keywords


def retrieve_cases(features: Dict, limit: int = 5) -> List[Dict]:
    """检索相似历史 case"""
    if not os.path.exists(DB_PATH):
        return []
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('SELECT * FROM sync_cases')
    all_cases = [dict(row) for row in c.fetchall()]
    conn.close()
    
    scored = []
    for case in all_cases:
        score = 0
        case_keywords = set(case.get('keywords', '').split(',')) if case.get('keywords') else set()
        case_apis = set(case.get('sync_apis', '').split(',')) if case.get('sync_apis') else set()
        case_vars = set(case.get('index_vars', '').split(',')) if case.get('index_vars') else set()
        
        feat_keywords = set(features['keywords'])
        feat_apis = set(features['apis'])
        feat_vars = set(features['vars'])
        
        common_kw = case_keywords & feat_keywords
        score += len(common_kw) * 3
        common_apis = case_apis & feat_apis
        score += len(common_apis) * 2
        common_vars = case_vars & feat_vars
        score += len(common_vars) * 5
        
        if case.get('fix_type') == 'buffer_index_mismatch' and 'buffer_index' in feat_keywords:
            score += 10
        if case.get('fix_type') == 'atomic_ordering' and 'atomic' in feat_keywords:
            score += 10
        if case.get('fix_type') == 'cross_core' and 'crosscore' in feat_keywords:
            score += 10
        
        if score > 0:
            scored.append({
                'case_id': case['id'],
                'repo': case['repo'],
                'pr_id': case['pr_id'],
                'pr_title': case['pr_title'],
                'file_path': case['file_path'],
                'fix_type': case['fix_type'],
                'fix_summary': case['fix_summary'],
                'keywords': case['keywords'],
                'sync_apis': case['sync_apis'],
                'index_vars': case['index_vars'],
                'score': score,
                'common_keywords': list(common_kw),
                'common_apis': list(common_apis),
                'common_vars': list(common_vars),
                'diff_patch': case['diff_patch'][:500],
            })
    
    scored.sort(key=lambda x: -x['score'])
    return scored[:limit]


def _extract_diff_pattern(diff_patch: str) -> Dict:
    """从 diff_patch 提取代码模式：删除了什么、添加了什么、涉及哪些变量。

    返回：
    - removed_lines: 删除的代码行（去前缀 -）
    - added_lines: 添加的代码行（去前缀 +）
    - removed_vars: 删除行中的索引变量
    - added_vars: 添加行中的索引变量
    - removed_apis: 删除行中的同步 API
    - added_apis: 添加行中的同步 API
    - pattern_summary: 模式摘要（如 'varA→varB' 或 'added PipeBarrier'）
    """
    removed_lines = []
    added_lines = []
    for line in diff_patch.split('\n'):
        if line.startswith('-') and not line.startswith('---'):
            removed_lines.append(line[1:].strip())
        elif line.startswith('+') and not line.startswith('+++'):
            added_lines.append(line[1:].strip())

    removed_text = ' '.join(removed_lines)
    added_text = ' '.join(added_lines)

    removed_vars = set(VAR_PATTERN.findall(removed_text))
    added_vars = set(VAR_PATTERN.findall(added_text))

    sync_api_re = re.compile(r'\b(SetFlag|WaitFlag|PipeBarrier|SyncFunc|SyncAll|SetAtomicAdd|EnQue|DeQue)\b')
    removed_apis = set(sync_api_re.findall(removed_text))
    added_apis = set(sync_api_re.findall(added_text))

    # 模式摘要
    pattern_summary = ''
    if added_apis and not removed_apis:
        pattern_summary = f'added {"+".join(added_apis)}'
    elif removed_apis and added_apis:
        pattern_summary = f'replaced {",".join(removed_apis)}→{",".join(added_apis)}'
    elif removed_vars and added_vars:
        # 变量替换模式：varA → varB
        old_only = removed_vars - added_vars
        new_only = added_vars - removed_vars
        if old_only and new_only:
            pattern_summary = f'var {"→".join(sorted(old_only))}→{",".join(sorted(new_only))}'
    elif added_apis:
        pattern_summary = f'added {"+".join(added_apis)}'

    return {
        'removed_lines': removed_lines[:5],
        'added_lines': added_lines[:5],
        'removed_vars': list(removed_vars)[:5],
        'added_vars': list(added_vars)[:5],
        'removed_apis': list(removed_apis),
        'added_apis': list(added_apis),
        'pattern_summary': pattern_summary,
    }


def _extract_var_fields(candidate: Dict, message: str, detail: str) -> Tuple[str, str]:
    """从候选字段与 message/detail 中提取 flag 与 buffer 索引变量。"""
    flag_var = candidate.get('flag_var', '')
    buffer_var = candidate.get('buffer_var', '')
    var_matches = re.findall(r'索引变量\((\w+)\)', message + detail)
    if not flag_var and var_matches:
        flag_var = var_matches[0]
    if not buffer_var and len(var_matches) > 1:
        buffer_var = var_matches[1]
    return flag_var, buffer_var


def _classify_fix_type(code: str, combined_text: str) -> str:
    """按候选 code/message 归类候选的修复类型。"""
    if code == 'SYNC-14':
        return 'buffer_index_mismatch'
    if code == 'SYNC-13':
        return 'atomic_ordering'
    if code == 'SYNC-02' and 'pipebarrier' in combined_text:
        return 'pipe_barrier_missing'
    if code == 'SYNC-02':
        return 'missing_sync'
    if code in ('SYNC-04', 'SYNC-08'):
        return 'flag_pair'
    if code == 'SYNC-03':
        return 'cross_core'
    if code in ('SYNC-01', 'SYNC-05', 'SYNC-06'):
        return 'sync_fix'
    return 'unknown'


def _candidate_pattern(code: str, flag_var: str, buffer_var: str, combined_text: str) -> Tuple[str, str]:
    """候选的代码模式特征：(变量替换描述, API 添加描述)。"""
    if code == 'SYNC-14' and flag_var and buffer_var:
        return f'{buffer_var}→{flag_var}', ''
    if code == 'SYNC-13':
        return '', 'PipeBarrier'
    if code == 'SYNC-02' and 'pipebarrier' in combined_text:
        return '', 'PipeBarrier'
    return '', ''


SPECIAL_KEYWORD_BONUS = (
    ('bias', 'bias'),
    ('scale', 'scale'),
    ('atomic', 'atomic'),
    ('crosscore', 'crosscore'),
    ('l0c', 'l0'),
    ('fixpipe', 'fixpipe'),
)


def _keyword_bonus(combined_text: str, case_keywords: set) -> int:
    bonus = 0
    for text, keyword in SPECIAL_KEYWORD_BONUS:
        if text in combined_text and keyword in case_keywords:
            bonus += 5
    return bonus


@dataclass
class CandidateContext:
    """候选的匹配上下文（计分/找最佳/构造结果各阶段共用）。"""

    code: str
    line: int
    message: str
    flag_var: str
    buffer_var: str
    candidate_fix_type: str
    candidate_replaced_vars: str
    candidate_added_api: str
    feat_keywords: set
    combined_text: str


def _base_type_score(case: Dict, ctx: CandidateContext) -> int:
    """修复类型(20) + SYNC 条例(15) + 索引变量(10×2) 计分。"""
    score = 0
    case_fix_type = case.get('fix_type', 'unknown')
    if case_fix_type == ctx.candidate_fix_type and ctx.candidate_fix_type != 'unknown':
        score += 20
    pattern = FIX_PATTERNS.get(case_fix_type, {})
    if ctx.code in pattern.get('sync_codes', []):
        score += 15
    case_vars = set(case.get('index_vars', '').split(',')) if case.get('index_vars') else set()
    if ctx.flag_var and ctx.flag_var in case_vars:
        score += 10
    if ctx.buffer_var and ctx.buffer_var in case_vars:
        score += 10
    return score


def _diff_pattern_score(case: Dict, ctx: CandidateContext, diff_pattern: Dict) -> int:
    """diff 代码模式计分：变量替换(+25/+15/+15) 与 API 添加(+25/+15)。"""
    score = 0
    case_pattern = diff_pattern.get('pattern_summary', '')
    if ctx.candidate_replaced_vars and case_pattern:
        if '→' in case_pattern and '→' in ctx.candidate_replaced_vars:
            score += 25
        if ctx.flag_var in diff_pattern.get('added_vars', []):
            score += 15
        if ctx.buffer_var in diff_pattern.get('removed_vars', []):
            score += 15
    if ctx.candidate_added_api and ctx.candidate_added_api in case_pattern:
        score += 25
    if ctx.candidate_added_api and ctx.candidate_added_api in diff_pattern.get('added_apis', []):
        score += 15
    return score


def _keyword_score(case: Dict, ctx: CandidateContext) -> int:
    """关键词计分：共同关键词(+2 each) + 特定关键词加分(+5)。"""
    score = 0
    case_keywords = set(case.get('keywords', '').split(',')) if case.get('keywords') else set()
    common = case_keywords & ctx.feat_keywords
    score += len(common) * 2
    score += _keyword_bonus(ctx.combined_text, case_keywords)
    return score


def _score_case(case: Dict, ctx: CandidateContext, diff_pattern: Dict) -> int:
    """单个历史 case 与候选的匹配得分。"""
    return (_base_type_score(case, ctx) + _diff_pattern_score(case, ctx, diff_pattern)
            + _keyword_score(case, ctx))


def _find_best_case(cases: List[Dict], ctx: CandidateContext) -> Tuple[Optional[Dict], int]:
    """在历史 case 中找最佳匹配。"""
    best_case = None
    best_score = 0
    for case in cases:
        diff_pattern = _extract_diff_pattern(case.get('diff_patch', ''))
        score = _score_case(case, ctx, diff_pattern)
        if score > best_score:
            best_score = score
            best_case = case
    return best_case, best_score


def _build_matching_result(ctx: CandidateContext, best_case: Optional[Dict],
                           best_score: int) -> Dict:
    """构建配对结果。"""
    result = {
        'candidate_code': ctx.code,
        'candidate_line': ctx.line,
        'candidate_message': ctx.message[:100],
        'flag_var': ctx.flag_var,
        'buffer_var': ctx.buffer_var,
        'candidate_fix_type': ctx.candidate_fix_type,
        'matched_case': None,
        'fix_pattern': None,
        'fix_hint': None,
        'evidence': None,
        'match_score': 0,
        'diff_pattern': '',
    }
    if best_case and best_score > 0:
        pattern = FIX_PATTERNS.get(best_case['fix_type'], {})
        diff_pattern = _extract_diff_pattern(best_case.get('diff_patch', ''))
        result['matched_case'] = {
            'pr_id': best_case['pr_id'],
            'repo': best_case['repo'],
            'pr_title': best_case['pr_title'],
            'file_path': best_case['file_path'],
            'fix_type': best_case['fix_type'],
        }
        result['fix_pattern'] = pattern.get('description', '')
        result['fix_hint'] = pattern.get('fix_hint', '')
        result['evidence'] = best_case['diff_patch'][:300]
        result['match_score'] = best_score
        result['diff_pattern'] = diff_pattern.get('pattern_summary', '')
    return result


def match_candidate_to_cases(candidate: Dict, features: Dict, cases: List[Dict]) -> Optional[Dict]:
    """将单个脚本候选与历史 case 自动配对。

    匹配逻辑：修复类型(20) + SYNC条例(15) + 索引变量(10) + diff代码模式(25) + 关键词(2) + 特定关键词加分(5)
    """
    code = candidate.get('code', '')
    line = candidate.get('line', 0)
    message = candidate.get('message', '')
    detail = candidate.get('detail', '')
    combined_text = (message + ' ' + detail).lower()
    flag_var, buffer_var = _extract_var_fields(candidate, message, detail)
    candidate_fix_type = _classify_fix_type(code, combined_text)
    candidate_replaced_vars, candidate_added_api = _candidate_pattern(code, flag_var, buffer_var,
                                                                      combined_text)
    ctx = CandidateContext(
        code=code,
        line=line,
        message=message,
        flag_var=flag_var,
        buffer_var=buffer_var,
        candidate_fix_type=candidate_fix_type,
        candidate_replaced_vars=candidate_replaced_vars,
        candidate_added_api=candidate_added_api,
        feat_keywords=set(features['keywords']),
        combined_text=combined_text,
    )
    best_case, best_score = _find_best_case(cases, ctx)
    return _build_matching_result(ctx, best_case, best_score)


def _print_features(features: Dict) -> None:
    _LOGGER.info('目标文件特征:')
    _LOGGER.info('  同步 API: %s', features['apis'])
    _LOGGER.info('  索引变量: %s', features['vars'])
    _LOGGER.info('  关键词: %s', features['keywords'])


def _print_cases(cases: List[Dict]) -> None:
    _LOGGER.info('')
    _LOGGER.info('相似历史 case（%d 个）:', len(cases))
    if not cases:
        _LOGGER.info('  未找到相似 case')
        return
    for i, case in enumerate(cases, 1):
        _LOGGER.info('')
        _LOGGER.info('  [%d] score=%d %s PR#%s', i, case['score'], case['repo'], case['pr_id'])
        _LOGGER.info('      title: %s', case['pr_title'][:60])
        _LOGGER.info('      fix_type: %s', case['fix_type'])
        _LOGGER.info('      file: %s', case['file_path'][:60])
        _LOGGER.info('      common_keywords: %s', case['common_keywords'])
        _LOGGER.info('      common_apis: %s', case['common_apis'])
        _LOGGER.info('      common_vars: %s', case['common_vars'])
        _LOGGER.info('      diff摘要: %s...', case['diff_patch'][:100])


def _print_type_based_hint(m: Dict) -> None:
    pattern = FIX_PATTERNS.get(m['candidate_fix_type'], {})
    if pattern:
        _LOGGER.info('    → 修复模式(基于类型): %s', pattern.get('description', ''))
        _LOGGER.info('    → 修复提示(基于类型): %s', pattern.get('fix_hint', ''))


def _print_matching(m: Dict) -> None:
    _LOGGER.info('')
    _LOGGER.info('  [%s] L%s: %s', m['candidate_code'], m['candidate_line'], m['candidate_message'])
    mc = m['matched_case']
    if not mc:
        _LOGGER.info('    → 无匹配历史 case（score=0）')
        _print_type_based_hint(m)
        return
    _LOGGER.info('    → 匹配 %s PR#%s (score=%s)', mc['repo'], mc['pr_id'], m['match_score'])
    _LOGGER.info('    → PR title: %s', mc['pr_title'][:50])
    _LOGGER.info('    → 修复类型: %s', mc['fix_type'])
    _LOGGER.info('    → 修复模式: %s', m['fix_pattern'])
    _LOGGER.info('    → 修复提示: %s', m['fix_hint'])
    _LOGGER.info('    → diff证据: %s...', m['evidence'][:80])


def _print_candidate_matches(candidates: List[Dict], features: Dict, cases: List[Dict]) -> None:
    _LOGGER.info('')
    _LOGGER.info('%s', '=' * 60)
    _LOGGER.info('候选自动配对结果:')
    for cand in candidates:
        m = match_candidate_to_cases(cand, features, cases)
        if m:
            _print_matching(m)


def _print_json_matchings(features: Dict, cases: List[Dict], candidates: List[Dict]) -> None:
    matchings = []
    for cand in candidates:
        m = match_candidate_to_cases(cand, features, cases)
        if m:
            matchings.append(m)
    _LOGGER.info(json.dumps({
        'features': features,
        'cases': cases,
        'matchings': matchings,
    }, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description='历史同步 case 检索器 + 候选自动配对')
    p.add_argument('file', help='目标文件')
    p.add_argument('--format', choices=['text', 'json'], default='text')
    p.add_argument('--limit', type=int, default=5)
    p.add_argument('--candidates', help='脚本候选 JSON（自动配对模式）')
    args = p.parse_args()

    features = extract_features(args.file)
    cases = retrieve_cases(features, args.limit)
    candidates = json.loads(args.candidates) if args.candidates else []

    if args.format == 'json':
        if candidates:
            _print_json_matchings(features, cases, candidates)
        else:
            _LOGGER.info(json.dumps({'features': features, 'cases': cases}, ensure_ascii=False, indent=2))
        return
    _print_features(features)
    _print_cases(cases)
    if candidates:
        _print_candidate_matches(candidates, features, cases)


if __name__ == '__main__':
    main()
