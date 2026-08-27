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
Ascend C data-flow graph extractor.

Uses a regex frontend to build a domain graph over:
  - buffer regions: buffer_[index expression]
  - index provenance: local alias -> root loop variable
  - buffer accesses: function-local read/write/address timeline
  - operations: producer address assignments, DataCopy, VF_CALL, SetFlag/WaitFlag
  - sync coverage: whether a SetFlag/WaitFlag covers the same buffer region
  - findings: graph queries over producer/consumer/sync buffer lifecycle consistency
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field as dc_field, replace
from typing import Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import sync_audit  # noqa: E402

from sync_logging import init_logging

_LOGGER = logging.getLogger('ascendc_flow_analyzer')
_STDERR_LOGGER = logging.getLogger('ascendc_flow_analyzer.stderr')
init_logging(_LOGGER, _STDERR_LOGGER)


OUTPUT_FIELD_RE = re.compile(r'(out|output|dst|highbit|f16|result)', re.IGNORECASE)
OUTPUT_BUFFER_RE = re.compile(r'(Out|Output|HighBit|Result|Dst)', re.IGNORECASE)
INPUT_FIELD_RE = re.compile(r'(in|input|src|lowbit|scale|biasIn)', re.IGNORECASE)
FIELD_ASSIGN_RE = re.compile(r'\b(\w+)\s*(?:\.|->)\s*(\w+)\s*=')
VAR_DECL_ASSIGN_RE = re.compile(
    r'\b(?:const\s+)?(?:uint\d+_t|int\d+_t|uint|int|auto)\s+([a-zA-Z_]\w*)\s*=\s*([^;]+);')
VAR_ASSIGN_RE = re.compile(r'^\s*([a-zA-Z_]\w*)\s*=\s*([^;]+);')
BUFFER_INDEX_RE = re.compile(r'(\w+_\w*|\w+Buf\w*|\w+Buffer\w*|\w+Ub\w*|\w+Local\w*)\s*\[\s*([^]]+)\]')
VAR_NAME_RE = re.compile(r'\b([a-z][a-zA-Z0-9_]+)\b')
DATA_COPY_RE = re.compile(r'\bDataCopy(?:Pad)?(?:Ext)?\s*(?:<[^>]*>)?\s*\(')
VF_CALL_RE = re.compile(r'\b(?:AscendC::)?VF_CALL\s*<[^>]+>\s*\(')
SKIP_VARS = {
    'true', 'false', 'nullptr', 'static_cast', 'reinterpret_cast', 'const_cast',
    'CeilAlign', 'CeilDivide', 'likely', 'unlikely', 'sizeof', 'min', 'max',
}
CONTROL_RE = re.compile(r'^\s*(?:if|for|while|switch|do|else|catch)\b')
DECL_BLOCK_RE = re.compile(r'^\s*(?:class|struct|enum|namespace|union)\b')
FUNCTION_NAME_RE = re.compile(r'([~\w:]+)\s*$')
KNOWN_CALL_NAMES = {
    'DataCopy', 'DataCopyPad', 'DataCopyPad2D', 'DataCopyExt', 'SetFlag',
    'WaitFlag', 'PipeBarrier', 'SyncAll', 'SyncFunc', 'CrossCoreSetFlag',
    'CrossCoreWaitFlag', 'VF_CALL', 'FetchEventID', 'GetTPipePtr',
}
RETURN_HINT_RE = re.compile(
    r'\b(?:void|bool|char|short|int|long|float|double|auto|uint|uint\d+_t|int\d+_t|size_t|'
    r'__aicore__|inline|static|constexpr|virtual|explicit)\b')


@dataclass
class IndexExpr:
    raw: str = ''
    root: str = ''
    index_var: str = ''
    slot: str = ''
    offset: int = 0
    modulo: str = ''
    stride: str = ''
    canonical: str = ''
    unknown: str = ''


@dataclass
class BufferRegion:
    name: str
    index_expr: str
    index_var: str
    index_root: str
    role: str
    index_summary: IndexExpr = dc_field(default_factory=IndexExpr)


@dataclass
class Operation:
    id: str
    kind: str
    file: str
    line: int
    func: str
    pipe: str = ''
    object: str = ''
    field: str = ''
    buffers: List[BufferRegion] = dc_field(default_factory=list)
    raw: str = ''


@dataclass
class BufferAccess:
    id: str
    file: str
    line: int
    func: str
    access: str
    pipe: str
    statement_kind: str
    buffer: BufferRegion
    symbol_id: str
    region_key: str
    raw: str


@dataclass
class BufferDependency:
    id: str
    file: str
    func: str
    producer_access_id: str
    consumer_access_id: str
    buffer_name: str
    producer_region: str
    consumer_region: str
    status: str
    distance: int
    message: str


@dataclass
class SyncEdge:
    id: str
    kind: str
    event_type: str
    flag_id: str
    index_expr: str
    index_var: str
    index_root: str
    file: str
    line: int
    func: str
    raw: str
    index_summary: IndexExpr = dc_field(default_factory=IndexExpr)


@dataclass
class SyncCoverage:
    id: str
    status: str
    file: str
    line: int
    func: str
    event_type: str
    sync_edge_id: str
    access_id: str
    distance: int
    expected_root: str
    actual_root: str
    expected_region: str
    actual_region: str
    message: str


@dataclass
class Finding:
    code: str
    severity: str
    file: str
    line: int
    message: str
    evidence: Dict[str, object]


@dataclass
class FlowGraph:
    files_scanned: List[str]
    frontends: Dict[str, str] = dc_field(default_factory=dict)
    operations: List[Operation] = dc_field(default_factory=list)
    buffer_accesses: List[BufferAccess] = dc_field(default_factory=list)
    buffer_dependencies: List[BufferDependency] = dc_field(default_factory=list)
    sync_edges: List[SyncEdge] = dc_field(default_factory=list)
    sync_coverages: List[SyncCoverage] = dc_field(default_factory=list)
    aliases: Dict[str, Dict[str, str]] = dc_field(default_factory=dict)
    findings: List[Finding] = dc_field(default_factory=list)


@dataclass
class StatementContext:
    """单条语句的扫描上下文：文件位置 + 语句 + 别名表 + 访问类型。"""

    path: str
    line: int
    func: str
    stmt: str
    aliases: Dict[str, str]
    statement_kind: str = ''
    object_name: str = ''
    field_name: str = ''


def collect_files(paths: Iterable[str]) -> List[str]:
    return sync_audit.collect_files(list(paths))


def regex_statements(path: str) -> List[Tuple[int, str, str]]:
    return merge_statements(scan_source(path))


def extract_statements(path: str, frontend: str = 'regex') -> Tuple[List[Tuple[int, str, str]], str]:
    return regex_statements(path), 'regex'


def strip_source_lines(path: str) -> List[Tuple[int, str]]:
    """Strip comments/strings with the same lightweight policy as sync_audit."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            raw_lines = handle.readlines()
    except OSError as exc:
        _STDERR_LOGGER.warning('warning: cannot read %s: %s', path, exc)
        return []

    stripped_lines: List[Tuple[int, str]] = []
    in_block_comment = False
    for line_no, raw in enumerate(raw_lines, start=1):
        line = raw.rstrip('\n')
        if in_block_comment:
            end = line.find('*/')
            if end == -1:
                stripped_lines.append((line_no, ''))
                continue
            line = line[end + 2:]
            in_block_comment = False

        code = sync_audit.strip_code(line)
        block_start = code.find('/*')
        while block_start != -1:
            block_end = code.find('*/', block_start + 2)
            if block_end == -1:
                code = code[:block_start]
                in_block_comment = True
                break
            code = code[:block_start] + code[block_end + 2:]
            block_start = code.find('/*')
        stripped_lines.append((line_no, code))
    return stripped_lines


def _is_decl_or_control_line(text: str) -> bool:
    """判断是否包含声明/控制语义，用于剔除函数名候选。"""
    return ';' in text or '=' in text or CONTROL_RE.match(text) or DECL_BLOCK_RE.match(text)


def candidate_function_name(header: str) -> str:
    """Return a function name for real definitions; reject common multiline calls."""
    text = ' '.join(part.strip() for part in header.splitlines()).strip()
    text = re.sub(r'\b[A-Z][A-Z0-9_]*_TEMPLATE_PARAM\b', ' ', text)
    text = text.strip()
    if not text or _is_decl_or_control_line(text):
        return ''
    open_pos = text.find('(')
    if open_pos < 0:
        return ''
    prefix = text[:open_pos].strip()
    match = FUNCTION_NAME_RE.search(prefix)
    if not match:
        return ''
    full_name = match.group(1)
    short_name = full_name.split('::')[-1].lstrip('~')
    if short_name in KNOWN_CALL_NAMES:
        return ''

    has_qualified_name = '::' in full_name
    has_return_hint = bool(RETURN_HINT_RE.search(prefix))
    prefix_without_name = prefix[:match.start()].strip()
    if not (has_qualified_name or has_return_hint or prefix_without_name):
        return ''
    return full_name


def _close_brace(scope_stack: List[str], depth: int) -> int:
    """处理 }：弹出函数/块作用域，返回更新后的深度。"""
    if len(scope_stack) > 1:
        scope_stack.pop()
    return max(0, depth - 1)


def scan_source(path: str) -> List[Tuple[int, int, str, str]]:
    """Return line records with conservative function scope assignment.

    The older sync_audit scanner intentionally accepts broad function-like
    patterns. That is useful for candidate discovery, but a data-flow graph
    must not let multiline calls such as DataCopy(...) become function scopes.
    """
    lines: List[Tuple[int, int, str, str]] = []
    scope_stack: List[str] = ['<global>']
    depth = 0
    pending_header: List[str] = []

    for line_no, code in strip_source_lines(path):
        current_func = scope_stack[-1]
        lines.append((line_no, depth, code, current_func))

        stripped = code.strip()
        if stripped:
            if pending_header:
                pending_header.append(stripped)
            elif '(' in stripped and not stripped.endswith(';') and not CONTROL_RE.match(stripped):
                pending_header = [stripped]

        for ch in code:
            if ch == '}':
                depth = _close_brace(scope_stack, depth)
                continue
            if ch == '{':
                func_name = candidate_function_name('\n'.join(pending_header)) if pending_header else ''
                scope_stack.append(func_name or scope_stack[-1])
                pending_header = []
                depth += 1

        if pending_header and (';' in code or code.endswith('}')):
            pending_header = []

    return lines


def _merge_continued_statement(lines: List[Tuple[int, int, str, str]], i: int,
                               text: str) -> Tuple[str, int]:
    """从第 i 行之后继续拼接多行语句，返回（合并文本, 下一行索引）。"""
    j = i + 1
    while j < len(lines):
        text += ' ' + lines[j][2].strip()
        if (';' in text
                and text.count('(') <= text.count(')')
                and text.count('[') <= text.count(']')):
            break
        if text.endswith('{') or text.endswith('}'):
            break
        j += 1
    return text, j + 1


def merge_statements(lines: List[Tuple[int, int, str, str]]) -> List[Tuple[int, str, str]]:
    """Merge multiline C++ statements enough for assignments and calls."""
    merged: List[Tuple[int, str, str]] = []
    i = 0
    while i < len(lines):
        ln, _depth, code, sig = lines[i]
        text = code.strip()
        if not text:
            i += 1
            continue

        should_merge = (
            ('=' in text or '(' in text or '[' in text)
            and not text.endswith('{')
            and not text.endswith('}')
            and ';' not in text
        )
        if should_merge:
            text, i = _merge_continued_statement(lines, i, text)
            merged.append((ln, sig, text))
        else:
            merged.append((ln, sig, text))
            i += 1
    return merged


def first_index_var(expr: str) -> str:
    for var in VAR_NAME_RE.findall(expr):
        if var not in SKIP_VARS and len(var) > 1:
            return var
    return ''


def resolve_alias(var: str, aliases: Dict[str, str]) -> str:
    seen = set()
    cur = var
    while cur in aliases and cur not in seen:
        seen.add(cur)
        cur = aliases[cur]
    return cur


def normalize_index_expr(expr: str) -> str:
    return re.sub(r'\s+', '', expr)


def _wraps_outer_parens(text: str) -> bool:
    depth = 0
    for idx, ch in enumerate(text):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0 and idx != len(text) - 1:
                return False
    return True


def strip_outer_parens(expr: str) -> str:
    text = expr.strip()
    while text.startswith('(') and text.endswith(')'):
        if not _wraps_outer_parens(text):
            break
        text = text[1:-1].strip()
    return text


def _delimiter_frame(text: str) -> Iterable[Tuple[int, str, int, int, int]]:
    """逐字符产出 (idx, ch, paren, bracket, angle) 括号深度帧。"""
    paren = bracket = angle = 0
    for idx, ch in enumerate(text):
        if ch == '(':
            paren += 1
        elif ch == ')':
            paren -= 1
        elif ch == '[':
            bracket += 1
        elif ch == ']':
            bracket -= 1
        elif ch == '<':
            angle += 1
        elif ch == '>':
            angle = max(0, angle - 1)
        yield idx, ch, paren, bracket, angle


def _at_top_level(paren: int, bracket: int, angle: int) -> bool:
    return paren == 0 and bracket == 0 and angle == 0


def split_top_level_operator(expr: str, operator: str) -> List[str]:
    parts: List[str] = []
    start = 0
    for _idx, ch, paren, bracket, angle in _delimiter_frame(expr):
        if ch == operator and _at_top_level(paren, bracket, angle):
            parts.append(expr[start:_idx].strip())
            start = _idx + 1
    parts.append(expr[start:].strip())
    return parts


def stride_like(expr: str) -> bool:
    text = strip_outer_parens(expr)
    if not text:
        return False
    if re.fullmatch(r'\d+', text):
        return True
    if re.search(r'(size|width|stride|bytes|len)', text, re.IGNORECASE):
        return True
    return bool(re.fullmatch(r'(?:[A-Z][A-Z0-9_]*::)?[A-Z][A-Z0-9_:]*', text))


def slot_index_expr(expr: str) -> str:
    text = strip_outer_parens(expr)
    factors = split_top_level_operator(text, '*')
    if len(factors) > 1:
        non_stride = [factor for factor in factors if not stride_like(factor)]
        if len(non_stride) == 1:
            return normalize_index_expr(strip_outer_parens(non_stride[0]))
    return normalize_index_expr(text)


def split_stride(expr: str) -> Tuple[str, str]:
    text = strip_outer_parens(expr)
    factors = split_top_level_operator(text, '*')
    if len(factors) > 1:
        non_stride = [factor for factor in factors if not stride_like(factor)]
        stride = [factor for factor in factors if stride_like(factor)]
        if len(non_stride) == 1 and stride:
            return strip_outer_parens(non_stride[0]), normalize_index_expr('*'.join(stride))
    return text, ''


def mask_to_modulo(mask: str) -> str:
    text = strip_outer_parens(mask)
    if re.fullmatch(r'\d+', text):
        value = int(text)
        modulo = value + 1
        if modulo > 0 and modulo & (modulo - 1) == 0:
            return str(modulo)
        return ''
    match = re.fullmatch(r'(.+)-1', normalize_index_expr(text))
    if match:
        return match.group(1)
    return ''


def split_modulo(expr: str) -> Tuple[str, str]:
    text = strip_outer_parens(expr)
    percent_parts = split_top_level_operator(text, '%')
    if len(percent_parts) == 2:
        return strip_outer_parens(percent_parts[0]), normalize_index_expr(percent_parts[1])

    mask_parts = split_top_level_operator(text, '&')
    if len(mask_parts) == 2:
        modulo = mask_to_modulo(mask_parts[1])
        return strip_outer_parens(mask_parts[0]), modulo or f'mask:{normalize_index_expr(mask_parts[1])}'
    return text, ''


def parse_base_offset(expr: str, variables: List[str]) -> Tuple[int, str]:
    text = normalize_index_expr(strip_outer_parens(expr))
    for var in variables:
        if not var:
            continue
        escaped = re.escape(var)
        if text == var:
            return 0, ''
        match = re.fullmatch(rf'{escaped}([+-]\d+)', text)
        if match:
            return int(match.group(1)), ''
        match = re.fullmatch(rf'([+-]?\d+)\+{escaped}', text)
        if match:
            return int(match.group(1) or 0), ''
    return 0, text


def summarize_index_expr(expr: str, index_var: str, root: str) -> IndexExpr:
    slot_expr, stride = split_stride(expr)
    base_expr, modulo = split_modulo(slot_expr)
    offset, unknown = parse_base_offset(base_expr, [index_var, root])
    canonical_root = root or index_var
    slot = slot_index_expr(expr)
    if canonical_root and not unknown:
        canonical = f'{canonical_root}:offset={offset}:mod={modulo or "-"}'
    elif canonical_root:
        canonical = f'{canonical_root}:expr={slot}'
    else:
        canonical = f'<constant>:expr={slot}'
    return IndexExpr(
        raw=expr,
        root=canonical_root,
        index_var=index_var,
        slot=slot,
        offset=offset,
        modulo=modulo,
        stride=stride,
        canonical=canonical,
        unknown=unknown,
    )


def symbol_id(func: str, var: str) -> str:
    if not var:
        return ''
    return f'{func}:{var}'


def region_key(buffer: BufferRegion) -> str:
    summary = buffer.index_summary or summarize_index_expr(buffer.index_expr, buffer.index_var, buffer.index_root)
    return f'{buffer.name}:{summary.canonical}'


def sync_region_key(buffer_name: str, edge: 'SyncEdge') -> str:
    summary = edge.index_summary or summarize_index_expr(edge.index_expr, edge.index_var, edge.index_root)
    return f'{buffer_name}:{summary.canonical}'


def extract_flag_index(flag_id: str) -> Tuple[str, str]:
    bracket = re.search(r'\[([^]]+)\]', flag_id)
    if bracket:
        expr = bracket.group(1)
        return expr, first_index_var(expr)
    return '', ''


def split_top_level_args(arg_text: str) -> List[str]:
    args: List[str] = []
    start = 0
    for _idx, ch, paren, bracket, angle in _delimiter_frame(arg_text):
        if ch == ',' and _at_top_level(paren, bracket, angle):
            args.append(arg_text[start:_idx].strip())
            start = _idx + 1
    tail = arg_text[start:].strip()
    if tail:
        args.append(tail)
    return args


def extract_call_args(stmt: str, open_paren_pos: int) -> List[str]:
    return split_top_level_args(sync_audit.extract_balanced_arg(stmt, open_paren_pos))


def classify_buffer_role(stmt: str, buffer_name: str, field_name: str = '') -> str:
    if field_name:
        if OUTPUT_FIELD_RE.search(field_name):
            return 'producer_output'
        if INPUT_FIELD_RE.search(field_name):
            return 'producer_input'
    if OUTPUT_BUFFER_RE.search(buffer_name):
        return 'output'
    return 'unknown'


def extract_buffers(stmt: str, aliases: Dict[str, str], field_name: str = '') -> List[BufferRegion]:
    buffers: List[BufferRegion] = []
    for match in BUFFER_INDEX_RE.finditer(stmt):
        name = match.group(1)
        expr = match.group(2).strip()
        idx_var = first_index_var(expr)
        root = resolve_alias(idx_var, aliases) if idx_var else ''
        buffers.append(BufferRegion(
            name,
            expr,
            idx_var,
            root,
            classify_buffer_role(stmt, name, field_name),
            summarize_index_expr(expr, idx_var, root),
        ))
    return buffers


def build_alias_snapshots(statements: List[Tuple[int, str, str]]) -> Tuple[
        Dict[Tuple[str, int], Dict[str, str]], Dict[str, Dict[str, str]]]:
    aliases_by_func: Dict[str, Dict[str, str]] = defaultdict(dict)
    snapshots: Dict[Tuple[str, int], Dict[str, str]] = {}
    for line, func, stmt in sorted(statements, key=lambda item: (item[1], item[0])):
        aliases = aliases_by_func[func]
        for match in VAR_DECL_ASSIGN_RE.finditer(stmt):
            dst = match.group(1)
            root = first_index_var(match.group(2))
            if root and root != dst:
                aliases[dst] = resolve_alias(root, aliases)
        assign = VAR_ASSIGN_RE.match(stmt)
        if assign:
            dst = assign.group(1)
            root = first_index_var(assign.group(2))
            if root and root != dst:
                aliases[dst] = resolve_alias(root, aliases)
        snapshots[(func, line)] = dict(aliases)
    return snapshots, dict(aliases_by_func)


def extract_sync_edges(path: str, statements: List[Tuple[int, str, str]],
                       aliases_by_func: Dict[str, Dict[str, str]],
                       aliases_at: Dict[Tuple[str, int], Dict[str, str]],
                       start_index: int) -> Tuple[List[SyncEdge], int]:
    edges: List[SyncEdge] = []
    sync_counter = start_index
    for line, func, stmt in statements:
        aliases = aliases_at.get((func, line), aliases_by_func.get(func, {}))
        for match in sync_audit.SET_WAIT_RE.finditer(stmt):
            kind = 'set' if match.group(1) == 'SetFlag' else 'wait'
            event_type = sync_audit.normalize_event(match.group(2))
            open_pos = stmt.index('(', match.start())
            flag_id = sync_audit.normalize_flag_id(sync_audit.extract_balanced_arg(stmt, open_pos))
            index_expr, index_var = extract_flag_index(flag_id)
            root = resolve_alias(index_var, aliases) if index_var else ''
            sync_counter += 1
            edges.append(SyncEdge(
                id=f's{sync_counter}',
                kind=kind,
                event_type=event_type,
                flag_id=flag_id,
                index_expr=index_expr,
                index_var=index_var,
                index_root=root,
                file=path,
                line=line,
                func=func,
                raw=stmt,
                index_summary=summarize_index_expr(index_expr, index_var, root),
            ))
    return edges, sync_counter


def infer_datacopy_pipe(stmt: str) -> str:
    open_pos = stmt.find('(')
    if open_pos < 0:
        return ''
    args = extract_call_args(stmt, open_pos)
    if not args:
        return ''
    dst = args[0]
    if re.search(r'(Global_|global_|Gm|GM|__gm__)', dst):
        return 'MTE3'
    if re.search(r'(L1|l1)', dst):
        return 'MTE3'
    return 'MTE2'


def make_buffer_access(access_id: str, ctx: StatementContext, access_kind: str, pipe: str,
                       buffer: BufferRegion) -> BufferAccess:
    return BufferAccess(
        id=access_id,
        file=ctx.path,
        line=ctx.line,
        func=ctx.func,
        access=access_kind,
        pipe=pipe,
        statement_kind=ctx.statement_kind,
        buffer=buffer,
        symbol_id=symbol_id(ctx.func, buffer.index_root or buffer.index_var),
        region_key=region_key(buffer),
        raw=ctx.stmt,
    )


def extract_datacopy_buffer_accesses(ctx: StatementContext,
                                     start_index: int) -> Tuple[List[BufferAccess], int]:
    accesses: List[BufferAccess] = []
    access_counter = start_index
    match = DATA_COPY_RE.search(ctx.stmt)
    if not match:
        return accesses, access_counter

    open_pos = ctx.stmt.index('(', match.start())
    args = extract_call_args(ctx.stmt, open_pos)
    pipe = infer_datacopy_pipe(ctx.stmt)
    dc_ctx = replace(ctx, statement_kind='datacopy')
    for idx, arg in enumerate(args):
        access_kind = 'write' if idx == 0 else 'read'
        for buffer in extract_buffers(arg, ctx.aliases):
            access_counter += 1
            accesses.append(make_buffer_access(f'a{access_counter}', dc_ctx, access_kind, pipe, buffer))
    return accesses, access_counter


def extract_statement_buffer_accesses(ctx: StatementContext,
                                      start_index: int) -> Tuple[List[BufferAccess], int]:
    accesses: List[BufferAccess] = []
    access_counter = start_index

    field_match = FIELD_ASSIGN_RE.search(ctx.stmt)
    if field_match:
        _obj, field_name = field_match.group(1), field_match.group(2)
        pipe = 'producer_addr' if OUTPUT_FIELD_RE.search(field_name) else ''
        access_kind = 'address' if OUTPUT_FIELD_RE.search(field_name) else 'unknown'
        field_ctx = replace(ctx, statement_kind='field_assign')
        for buffer in extract_buffers(ctx.stmt, ctx.aliases, field_name):
            access_counter += 1
            accesses.append(make_buffer_access(
                f'a{access_counter}', field_ctx, access_kind, pipe, buffer))

    datacopy_accesses, access_counter = extract_datacopy_buffer_accesses(ctx, access_counter)
    accesses.extend(datacopy_accesses)

    if VF_CALL_RE.search(ctx.stmt):
        vf_ctx = replace(ctx, statement_kind='vf_call')
        for buffer in extract_buffers(ctx.stmt, ctx.aliases):
            access_counter += 1
            accesses.append(make_buffer_access(
                f'a{access_counter}', vf_ctx, 'compute', 'V', buffer))

    return accesses, access_counter


def event_pipes(event_type: str) -> Tuple[str, str]:
    if '_' not in event_type:
        return '', ''
    producer, consumer = event_type.split('_', 1)
    return producer, consumer


def access_matches_pipe(access: BufferAccess, pipe: str) -> bool:
    if not pipe:
        return True
    if access.pipe == pipe:
        return True
    return access.pipe == 'producer_addr'


def lifecycle_relevant(access: BufferAccess) -> bool:
    return output_like(access.buffer) or access.access == 'address'


def dependency_status(producer: BufferAccess, consumer: BufferAccess) -> Tuple[str, str]:
    if producer.buffer.index_root != consumer.buffer.index_root:
        return 'wrong_index_root', 'producer and consumer use different index roots'
    if producer.region_key != consumer.region_key:
        return 'same_root_different_region', 'producer and consumer use same root but different buffer regions'
    return 'same_region', 'producer and consumer use the same buffer region'


def producer_like(access: BufferAccess) -> bool:
    return access.access in ('write', 'address', 'compute')


def consumer_like(access: BufferAccess) -> bool:
    return access.access in ('read', 'address', 'compute')


def _dep_from_producers(access: BufferAccess, producers: List[BufferAccess],
                        dep_counter: int) -> Tuple[Optional[BufferDependency], int]:
    """为单个 consumer access 在已有 producers 中寻找最近依赖，找不到返回 (None, counter)。"""
    for producer in reversed(producers):
        if access.line < producer.line or access.line - producer.line > 80:
            continue
        status, message = dependency_status(producer, access)
        dep_counter += 1
        return BufferDependency(
            id=f'd{dep_counter}',
            file=access.file,
            func=access.func,
            producer_access_id=producer.id,
            consumer_access_id=access.id,
            buffer_name=access.buffer.name,
            producer_region=producer.region_key,
            consumer_region=access.region_key,
            status=status,
            distance=access.line - producer.line,
            message=message,
        ), dep_counter
    return None, dep_counter


def _collect_buffer_dependencies(ordered, dependencies: List[BufferDependency],
                                 dep_counter: int) -> int:
    """按行序为每个 consumer access 寻找最近 producer，生成依赖边。"""
    producers: List[BufferAccess] = []
    for access in ordered:
        if consumer_like(access):
            dep, dep_counter = _dep_from_producers(access, producers, dep_counter)
            if dep:
                dependencies.append(dep)
        if producer_like(access):
            producers.append(access)
    return dep_counter


def build_buffer_dependencies(graph: FlowGraph) -> List[BufferDependency]:
    dependencies: List[BufferDependency] = []
    dep_counter = 0
    by_buffer: Dict[Tuple[str, str, str, str], List[BufferAccess]] = defaultdict(list)
    for access in graph.buffer_accesses:
        if access.buffer.index_root and lifecycle_relevant(access):
            by_buffer[(access.file, access.func, access.buffer.name, access.buffer.index_root)].append(access)

    for accesses in by_buffer.values():
        ordered = sorted(accesses, key=lambda item: (item.line, item.id))
        dep_counter = _collect_buffer_dependencies(ordered, dependencies, dep_counter)
    return dependencies


def dependencies_by_access(graph: FlowGraph) -> Dict[str, List[BufferDependency]]:
    by_access: Dict[str, List[BufferDependency]] = defaultdict(list)
    for dependency in graph.buffer_dependencies:
        by_access[dependency.producer_access_id].append(dependency)
        by_access[dependency.consumer_access_id].append(dependency)
    return by_access


def _too_far_from_edge(edge: SyncEdge, access: BufferAccess) -> bool:
    if edge.kind == 'set':
        return access.line > edge.line or edge.line - access.line > 25
    return access.line < edge.line or access.line - edge.line > 25


def _coverage_message(status: str) -> str:
    if status == 'covered':
        return 'sync covers same buffer region'
    if status == 'wrong_index_root':
        return 'sync index root differs from lifecycle buffer access root'
    return 'sync uses same index root but a different buffer region expression'


def _coverage_for_edge(edge: SyncEdge, accesses_by_func: Dict[Tuple[str, str], List[BufferAccess]],
                       coverage_counter: int) -> Tuple[List[SyncCoverage], int]:
    """为单个 sync edge 生成其覆盖范围内的 buffer access 覆盖记录。"""
    coverages: List[SyncCoverage] = []
    if not edge.index_root:
        return coverages, coverage_counter
    producer_pipe, consumer_pipe = event_pipes(edge.event_type)
    pipe = producer_pipe if edge.kind == 'set' else consumer_pipe
    candidates = accesses_by_func.get((edge.file, edge.func), [])
    expected_region_by_buffer: Dict[str, str] = {}
    for access in candidates:
        if _too_far_from_edge(edge, access):
            continue
        if not access_matches_pipe(access, pipe):
            continue
        if access.buffer.name not in expected_region_by_buffer:
            expected_region_by_buffer[access.buffer.name] = sync_region_key(access.buffer.name, edge)
        expected_region = expected_region_by_buffer[access.buffer.name]
        actual_region = access.region_key
        if access.buffer.index_root != edge.index_root:
            status = 'wrong_index_root'
        elif actual_region != expected_region:
            status = 'same_root_different_region'
        else:
            status = 'covered'
        coverage_counter += 1
        coverages.append(SyncCoverage(
            id=f'c{coverage_counter}',
            status=status,
            file=edge.file,
            line=edge.line,
            func=edge.func,
            event_type=edge.event_type,
            sync_edge_id=edge.id,
            access_id=access.id,
            distance=abs(edge.line - access.line),
            expected_root=edge.index_root,
            actual_root=access.buffer.index_root,
            expected_region=expected_region,
            actual_region=actual_region,
            message=_coverage_message(status),
        ))
    return coverages, coverage_counter


def build_sync_coverages(graph: FlowGraph) -> List[SyncCoverage]:
    coverages: List[SyncCoverage] = []
    coverage_counter = 0
    accesses_by_func: Dict[Tuple[str, str], List[BufferAccess]] = defaultdict(list)
    for access in graph.buffer_accesses:
        if access.buffer.index_root and lifecycle_relevant(access):
            accesses_by_func[(access.file, access.func)].append(access)

    for edge in graph.sync_edges:
        edge_coverages, coverage_counter = _coverage_for_edge(edge, accesses_by_func, coverage_counter)
        coverages.extend(edge_coverages)
    return coverages


def _new_operation(op_counter: int, kind: str, ctx: StatementContext, pipe: str,
                   buffers: List[BufferRegion]) -> Tuple[Operation, int]:
    op_counter += 1
    return Operation(
        id=f'o{op_counter}',
        kind=kind,
        file=ctx.path,
        line=ctx.line,
        func=ctx.func,
        pipe=pipe,
        object=ctx.object_name,
        field=ctx.field_name,
        buffers=buffers,
        raw=ctx.stmt,
    ), op_counter


def _field_assign_operation(ctx: StatementContext,
                            op_counter: int) -> Tuple[Optional[Operation], int]:
    field_match = FIELD_ASSIGN_RE.search(ctx.stmt)
    if not field_match:
        return None, op_counter
    obj, field_name = field_match.group(1), field_match.group(2)
    buffers = extract_buffers(ctx.stmt, ctx.aliases, field_name)
    if not buffers:
        return None, op_counter
    ctx = replace(ctx, object_name=obj, field_name=field_name)
    return _new_operation(op_counter, 'field_assign', ctx, '', buffers)




def _datacopy_operation(ctx: StatementContext, op_counter: int) -> Tuple[Optional[Operation], int]:
    if not DATA_COPY_RE.search(ctx.stmt):
        return None, op_counter
    return _new_operation(op_counter, 'datacopy', ctx, infer_datacopy_pipe(ctx.stmt),
                          extract_buffers(ctx.stmt, ctx.aliases))


def _vf_call_operation(ctx: StatementContext, op_counter: int) -> Tuple[Optional[Operation], int]:
    if not VF_CALL_RE.search(ctx.stmt):
        return None, op_counter
    return _new_operation(op_counter, 'vf_call', ctx, 'V', extract_buffers(ctx.stmt, ctx.aliases))


STATEMENT_BUILDERS = (_field_assign_operation, _datacopy_operation, _vf_call_operation)


def _collect_statement_ops(ctx: StatementContext, op_counter: int,
                           access_counter: int):
    """单条语句的 buffer access 与 operation 收集。"""
    ops = []
    buffer_accesses, access_counter = extract_statement_buffer_accesses(ctx, access_counter)
    for builder in STATEMENT_BUILDERS:
        op, op_counter = builder(ctx, op_counter)
        if op:
            ops.append(op)
    return ops, buffer_accesses, op_counter, access_counter


def build_graph(files: List[str]) -> FlowGraph:
    graph = FlowGraph(files_scanned=files)
    op_counter = 0
    sync_counter = 0
    access_counter = 0

    for path in files:
        statements, frontend_used = extract_statements(path)
        graph.frontends[path] = frontend_used
        aliases_at, aliases_by_func = build_alias_snapshots(statements)
        graph.aliases[path] = aliases_by_func

        sync_edges, sync_counter = extract_sync_edges(path, statements, aliases_by_func, aliases_at, sync_counter)
        graph.sync_edges.extend(sync_edges)

        for line, func, stmt in statements:
            aliases = aliases_at.get((func, line), aliases_by_func.get(func, {}))
            ctx = StatementContext(path, line, func, stmt, aliases)
            ops, buffer_accesses, op_counter, access_counter = _collect_statement_ops(
                ctx, op_counter, access_counter)
            graph.buffer_accesses.extend(buffer_accesses)
            graph.operations.extend(ops)

    graph.buffer_dependencies.extend(build_buffer_dependencies(graph))
    graph.sync_coverages.extend(build_sync_coverages(graph))
    graph.findings.extend(check_sync14_output_bundle(graph))
    graph.findings.extend(check_sync14_sync_to_nearby_buffers(graph))
    dedupe_findings(graph.findings)
    return graph


def output_like(buffer: BufferRegion) -> bool:
    return buffer.role in ('producer_output', 'output') or bool(OUTPUT_BUFFER_RE.search(buffer.name))


def _collect_index_roots(ops: List[Operation]) -> set:
    roots = set()
    for op in ops:
        for buf in op.buffers:
            if output_like(buf) and buf.index_root:
                roots.add(buf.index_root)
    return roots


def _find_reference_region(ops: List[Operation], sync_root: str) -> Tuple[Optional[Operation],
                                                                          Optional[BufferRegion]]:
    for op in ops:
        for buf in op.buffers:
            if output_like(buf) and buf.index_root == sync_root:
                return op, buf
    return None, None


def _bundle_root_mismatch_findings(ops: List[Operation], obj: str, sync_root: str,
                                   ref_op: Optional[Operation],
                                   ref_buf: Optional[BufferRegion]) -> List[Finding]:
    findings: List[Finding] = []
    for op in ops:
        for buf in op.buffers:
            if not output_like(buf) or not buf.index_root or buf.index_root == sync_root:
                continue
            findings.append(Finding(
                code='SYNC-14',
                severity='红线',
                file=op.file,
                line=op.line,
                message=f'同一输出参数对象 {obj} 的 producer output buffer 索引不同源',
                evidence={
                    'kind': 'output_bundle_index_mismatch',
                    'object': obj,
                    'bad_operation': asdict(op),
                    'bad_buffer': asdict(buf),
                    'reference_operation': asdict(ref_op) if ref_op else None,
                    'reference_buffer': asdict(ref_buf) if ref_buf else None,
                    'expected_root': sync_root,
                    'actual_root': buf.index_root,
                    'fix_hint': '统一 producer/consumer/sync 的 output buffer id，通常改为 ubComputeLoopIdx_ 派生索引。',
                },
            ))
    return findings


def check_sync14_output_bundle(graph: FlowGraph) -> List[Finding]:
    findings: List[Finding] = []
    by_object: Dict[Tuple[str, str, str], List[Operation]] = defaultdict(list)
    for op in graph.operations:
        if op.kind != 'field_assign' or not op.object:
            continue
        if not OUTPUT_FIELD_RE.search(op.field):
            continue
        if not any(output_like(buf) for buf in op.buffers):
            continue
        by_object[(op.file, op.func, op.object)].append(op)

    for (_file, _func, obj), ops in by_object.items():
        roots = _collect_index_roots(ops)
        if len(roots) < 2:
            continue
        compute_roots = [root for root in roots if 'ComputeLoopIdx' in root or root == 'ubComputeLoopIdx_']
        if not compute_roots:
            continue
        sync_root = compute_roots[0]
        ref_op, ref_buf = _find_reference_region(ops, sync_root)
        findings.extend(_bundle_root_mismatch_findings(ops, obj, sync_root, ref_op, ref_buf))
    return findings


def check_sync14_sync_to_nearby_buffers(graph: FlowGraph) -> List[Finding]:
    findings: List[Finding] = []
    accesses = {access.id: access for access in graph.buffer_accesses}
    edges = {edge.id: edge for edge in graph.sync_edges}
    deps_by_access = dependencies_by_access(graph)
    for coverage in graph.sync_coverages:
        if coverage.status not in ('wrong_index_root', 'same_root_different_region'):
            continue
        if coverage.event_type not in ('MTE3_V', 'V_MTE2', 'MTE2_V', 'V_MTE3'):
            continue
        access = accesses.get(coverage.access_id)
        edge = edges.get(coverage.sync_edge_id)
        if access is None or edge is None:
            continue
        root_mismatch = coverage.status == 'wrong_index_root'
        related_dependencies = [
            asdict(dep)
            for dep in deps_by_access.get(access.id, [])
            if dep.status != 'same_region'
        ]
        findings.append(Finding(
            code='SYNC-14',
            severity='红线' if root_mismatch else '高',
            file=access.file,
            line=access.line,
            message=(
                '同步 event 索引与 buffer 生命周期访问索引不同源'
                if root_mismatch
                else '同步 event 与 buffer 生命周期访问使用同一索引源但不同 region 表达式'
            ),
            evidence={
                'kind': (
                    'sync_lifecycle_index_root_mismatch'
                    if root_mismatch
                    else 'sync_lifecycle_region_mismatch'
                ),
                'sync_coverage': asdict(coverage),
                'sync_edge': asdict(edge),
                'buffer_access': asdict(access),
                'buffer_dependencies': related_dependencies,
                'buffer': asdict(access.buffer),
                'expected_root': coverage.expected_root,
                'actual_root': coverage.actual_root,
                'expected_region': coverage.expected_region,
                'actual_region': coverage.actual_region,
            },
        ))
    return findings


def dedupe_findings(findings: List[Finding]) -> None:
    seen = set()
    unique: List[Finding] = []
    for finding in findings:
        key = (finding.code, finding.file, finding.line, finding.message,
               json.dumps(finding.evidence, sort_keys=True, ensure_ascii=False)[:400])
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    findings[:] = unique


def print_text(graph: FlowGraph) -> None:
    _LOGGER.info('files: %d', len(graph.files_scanned))
    frontends = sorted(set(graph.frontends.values()))
    _LOGGER.info('frontends: %s', ', '.join(frontends) if frontends else 'unknown')
    _LOGGER.info('operations: %d', len(graph.operations))
    _LOGGER.info('buffer_accesses: %d', len(graph.buffer_accesses))
    _LOGGER.info('buffer_dependencies: %d', len(graph.buffer_dependencies))
    _LOGGER.info('sync_edges: %d', len(graph.sync_edges))
    _LOGGER.info('sync_coverages: %d', len(graph.sync_coverages))
    _LOGGER.info('findings: %d', len(graph.findings))
    for finding in graph.findings:
        _LOGGER.info('[%s] %s %s:%s %s', finding.severity, finding.code, finding.file, finding.line,
                     finding.message)
        kind = finding.evidence.get('kind', '')
        _LOGGER.info('  evidence: %s', kind)
        if 'expected_root' in finding.evidence:
            _LOGGER.info('  expected_root: %s', finding.evidence['expected_root'])
        if 'actual_root' in finding.evidence:
            _LOGGER.info('  actual_root: %s', finding.evidence['actual_root'])
        if 'expected_region' in finding.evidence:
            _LOGGER.info('  expected_region: %s', finding.evidence['expected_region'])
        if 'actual_region' in finding.evidence:
            _LOGGER.info('  actual_region: %s', finding.evidence['actual_region'])


def main() -> int:
    parser = argparse.ArgumentParser(description='Extract an Ascend C data-flow evidence graph.')
    parser.add_argument('targets', nargs='+', help='Source file or directory')
    parser.add_argument('--format', choices=('json', 'text'), default='json')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print JSON')
    args = parser.parse_args()

    files = collect_files(args.targets)
    try:
        graph = build_graph(files)
    except RuntimeError as exc:
        _STDERR_LOGGER.error('error: %s', exc)
        return 2
    if args.format == 'text':
        print_text(graph)
    else:
        _LOGGER.info(json.dumps(asdict(graph), ensure_ascii=False, indent=2 if args.pretty else None))
    return 1 if graph.findings else 0


if __name__ == '__main__':
    raise SystemExit(main())
