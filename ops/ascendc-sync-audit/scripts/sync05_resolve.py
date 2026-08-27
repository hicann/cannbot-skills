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
"""SYNC-05 CrossCoreWaitFlag PIPE 方向匹配解析（被 sync_audit.check_crosscore_pipe_direction 使用）。

维护函数/类/变量/using 别名的跨文件解析表：函数表按 (文件, 函数名) 与全局函数名
两级登记；被调对象经变量类型链（using 别名 / 模板形参 Foo_ / 类名前缀唯一匹配）
解析到具体类；operator 等高碰撞名不做全局函数名回退。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

FUNC_DEF_RE = re.compile(r'__aicore__\s+inline\s+\w+\s+(\w+)\s*\(')
CLASS_DEF_RE = re.compile(r'class\s+(\w+)')
VAR_DECL_RE = re.compile(r'\b(\w+)\s+(\w+)\s*[;=]')
USING_ALIAS_RE = re.compile(r'\busing\s+(\w+)\s*=\s*(?:typename\s+)?(?:\w+::)*(\w+)\s*[<;]')
VAR_STOP_TYPES = ('int', 'uint', 'float', 'bool', 'void', 'auto',
                  'const', 'static', 'if', 'for', 'while', 'return',
                  'constexpr', 'using', 'template', 'typename')

# API → PIPE 归属映射（用于跨函数 PIPE 方向匹配检查）
API_TO_PIPE = {
    # PIPE_V
    'Relu': 'PIPE_V', 'LeakyRelu': 'PIPE_V', 'Exp': 'PIPE_V', 'Sqrt': 'PIPE_V',
    'Log': 'PIPE_V', 'Abs': 'PIPE_V', 'Cast': 'PIPE_V', 'Duplicate': 'PIPE_V',
    'Adds': 'PIPE_V', 'Muls': 'PIPE_V', 'Add': 'PIPE_V', 'Sub': 'PIPE_V',
    'Mul': 'PIPE_V', 'Div': 'PIPE_V', 'Compare': 'PIPE_V', 'Select': 'PIPE_V',
    'ReduceMax': 'PIPE_V', 'ReduceMin': 'PIPE_V', 'ReduceSum': 'PIPE_V',
    'Brcb': 'PIPE_V', 'Gather': 'PIPE_V', 'Transpose': 'PIPE_V',
    'Axpy': 'PIPE_V', 'FusedMulAdd': 'PIPE_V',
    # PIPE_MTE3（DataCopy 方向取决于参数：dst 为 GM 则 MTE3，dst 为 UB 则 MTE2，见 check_flow）
    'DataCopy': 'PIPE_MTE3', 'DataCopyPad': 'PIPE_MTE3',
    # PIPE_M
    'Mmad': 'PIPE_M',
    # PIPE_FIX
    'Fixpipe': 'PIPE_FIX', 'FixPipe': 'PIPE_FIX',
}


@dataclass
class FirstOpInfo:
    """函数/类体内首个 API 操作：PIPE 归属 + 条件分支上下文 + 分支外回退操作。

    conditional=True 表示首个 API 位于 if/if constexpr 分支内（如 enableRelu 下的
    Relu），此时 fallback_pipe/fallback_api 记录分支外无条件路径的下一个 API（如
    DataCopyPad），供 SYNC-05 给出「按同条件分支选择 wait PIPE」的修法。
    """

    pipe: str = ''
    api: str = ''
    conditional: bool = False
    fallback_pipe: str = ''
    fallback_api: str = ''


@dataclass
class FuncScanState:
    """跨函数首个 PIPE 收集的扫描状态（函数名/类名/深度/条件块栈/已收集信息）。"""

    in_func: Optional[str] = None
    in_class: Optional[str] = None
    brace_depth: int = 0
    body_started: bool = False
    cond_depths: List[int] = field(default_factory=list)
    info: Optional[FirstOpInfo] = None

    def reset_func(self) -> None:
        self.in_func = None
        self.brace_depth = 0
        self.body_started = False
        self.cond_depths = []
        self.info = None


@dataclass
class ResolveCtx:
    """SYNC-05 被调解析上下文（函数/变量表本文件优先→全局，类表与 using 别名全局）。"""

    func_ops_file: Dict[Tuple[str, str], FirstOpInfo] = field(default_factory=dict)
    func_ops: Dict[str, FirstOpInfo] = field(default_factory=dict)
    class_ops: Dict[str, FirstOpInfo] = field(default_factory=dict)
    var_type_file: Dict[Tuple[str, str], str] = field(default_factory=dict)
    var_type: Dict[str, str] = field(default_factory=dict)
    using_alias: Dict[str, str] = field(default_factory=dict)

    def func_op(self, path: str, name: str) -> Optional[FirstOpInfo]:
        return self.func_ops_file.get((path, name)) or self.func_ops.get(name)

    def var_of(self, path: str, name: str) -> str:
        return self.var_type_file.get((path, name), '') or self.var_type.get(name, '')


def _match_line_api(code: str) -> Optional[Tuple[str, str]]:
    """当前行首个命中 API_TO_PIPE 的 API，返回 (api, pipe)。"""
    if 'Te::' in code or 'Reg::' in code:
        return None
    for api_name, pipe in API_TO_PIPE.items():
        if re.search(r'\b' + api_name + r'\s*(?:<[^>]*>)?\s*\(', code):
            return api_name, pipe
    return None


def _apply_line_braces(code: str, state: FuncScanState) -> None:
    """按当前行更新函数体大括号深度与 if 条件块栈（每行只统计一次）。"""
    opens, closes = code.count('{'), code.count('}')
    if opens and not state.body_started:
        state.body_started = True
    state.brace_depth += opens - closes
    while state.cond_depths and state.brace_depth < state.cond_depths[-1]:
        state.cond_depths.pop()
    if opens and re.search(r'\bif\b', code):
        state.cond_depths.append(state.brace_depth)


def _enter_func_def(code: str, state: FuncScanState) -> None:
    """不在函数体内时，尝试从当前行识别类定义与函数定义并初始化扫描状态。"""
    cm = CLASS_DEF_RE.search(code)
    if cm:
        state.in_class = cm.group(1)
    m = FUNC_DEF_RE.search(code)
    if not m:
        return
    if ';' in code and '{' not in code:
        return  # 纯声明（无函数体）不进入
    state.in_func = m.group(1)
    state.brace_depth = 0
    state.body_started = False
    state.cond_depths = []
    state.info = None
    _apply_line_braces(code, state)


def _absorb_line_api(code: str, state: FuncScanState) -> None:
    """函数体内收集首个 API；若其在条件分支内，继续收集分支外回退 API。"""
    hit = _match_line_api(code)
    if not hit:
        return
    api, pipe = hit
    conditional = bool(state.cond_depths)
    if state.info is None:
        state.info = FirstOpInfo(pipe=pipe, api=api, conditional=conditional)
    elif state.info.conditional and not state.info.fallback_pipe and not conditional:
        state.info.fallback_pipe = pipe
        state.info.fallback_api = api


def _finish_func(state: FuncScanState, path: str, ctx: ResolveCtx) -> None:
    """函数体结束：登记函数/类的首个 API 信息（首个有 API 的函数优先，不覆盖）。"""
    if state.info and state.in_func:
        ctx.func_ops_file.setdefault((path, state.in_func), state.info)
        ctx.func_ops.setdefault(state.in_func, state.info)
        if state.in_class:
            ctx.class_ops.setdefault(state.in_class, state.info)
    state.reset_func()


def _finish_single_line_func(state: FuncScanState, path: str, ctx: ResolveCtx) -> None:
    """函数签名与函数体同行的单行函数：进入即结束，直接登记。"""
    if state.in_func and state.body_started and state.brace_depth <= 0:
        _finish_func(state, path, ctx)


def _collect_func_first_pipe(lines_by_file, ctx: ResolveCtx) -> None:
    """收集每个函数/类体内首个 API 操作归属的 PIPE（跨函数 PIPE 方向匹配用）。

    修复 v8 回归：旧实现在函数签名行与函数体首行重复累计大括号深度，且深度≤0 即
    退出，导致「签名与 { 不同行」的函数（本仓主流风格）全部被丢弃、收集表恒空。
    函数表按 (文件, 函数名) 与全局函数名两级登记，降低跨文件同名碰撞误归属。
    """
    for path, lns in lines_by_file:
        state = FuncScanState()
        for (_ln, _depth, code, _sig) in lns:
            if not state.in_func:
                _enter_func_def(code, state)
                _finish_single_line_func(state, path, ctx)
                continue
            if not state.body_started and ';' in code and '{' not in code:
                state.reset_func()  # 多行签名后接分号：实为声明
                continue
            _apply_line_braces(code, state)
            if state.body_started and state.brace_depth > 0:
                _absorb_line_api(code, state)
            if state.body_started and state.brace_depth <= 0:
                _finish_func(state, path, ctx)


def _collect_decl_types(path: str, code: str, ctx: ResolveCtx) -> None:
    """从单行代码收集变量声明类型（过滤控制/修饰关键字）。"""
    for dm in VAR_DECL_RE.finditer(code):
        tname, vname = dm.group(1), dm.group(2)
        if tname not in VAR_STOP_TYPES:
            ctx.var_type_file.setdefault((path, vname), tname)
            ctx.var_type.setdefault(vname, tname)


def _collect_types_and_aliases(lines_by_file, ctx: ResolveCtx) -> None:
    """收集变量声明类型与 using 别名（using Alias = [ns::]Target<...>;）。"""
    for path, lns in lines_by_file:
        for (_ln, _depth, code, _sig) in lns:
            _collect_decl_types(path, code, ctx)
            m = USING_ALIAS_RE.search(code)
            if m and m.group(1) != m.group(2):
                ctx.using_alias.setdefault(m.group(1), m.group(2))


def _scope_pipe_params(events: List[SyncEvent]) -> Dict[Tuple[str, str], Set[str]]:
    """同 side 同作用域内已使用的 CrossCoreWaitFlag PIPE 参数集合。"""
    scope_side_pipes: Dict[Tuple[str, str], Set[str]] = {}
    for e in events:
        if e.kind == 'cwait' and e.pipe_param:
            key = (e.func, e.side)
            scope_side_pipes.setdefault(key, set()).add(e.pipe_param)
    return scope_side_pipes


def _resolve_class_first_op(type_name: str, using_alias: Dict[str, str],
                            class_ops: Dict[str, FirstOpInfo],
                            seen: Optional[Set[str]] = None
                            ) -> Tuple[Optional[FirstOpInfo], str]:
    """类型名 → 首个 API 信息。解析链：精确类名 → using 别名展开 → 模板形参
    约定（Foo_ → Foo）→ 前缀唯一兜底（如别名 BlockEpilogue → 类 BlockEpilogueFixpipe，
    要求所有前缀命中类的 PIPE 一致才采用）。返回 (信息, 解析路径描述)。"""
    if not type_name:
        return None, ''
    seen = seen if seen is not None else set()
    if type_name in seen:
        return None, ''
    seen.add(type_name)
    if type_name in class_ops:
        return class_ops[type_name], type_name
    if type_name in using_alias:
        info, via = _resolve_class_first_op(using_alias[type_name], using_alias, class_ops, seen)
        if info:
            return info, via
    stripped = type_name.rstrip('_')
    if stripped != type_name:
        info, via = _resolve_class_first_op(stripped, using_alias, class_ops, seen)
        if info:
            return info, via
    if len(type_name) >= 6:  # 短类型名前缀误配风险大，仅长名启用
        matches = [(c, i) for c, i in class_ops.items() if c.startswith(type_name)]
        if matches and len({i.pipe for _c, i in matches}) == 1:
            return matches[0][1], f'{matches[0][0]}(前缀匹配)'
    return None, ''


METHOD_CALL_RE = re.compile(r'\b(\w+)\.(?:template\s+)?(\w+)\s*[<(]')


def _parse_callee(next_code: str) -> Tuple[str, str]:
    """解析调用形态：obj.Method(...) / obj.template operator()<...>(...) 返回
    (对象名, 方法名)；普通调用 Foo(...) 返回 ('', 'Foo')。"""
    m = METHOD_CALL_RE.search(next_code)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r'\b(\w+)\s*\(', next_code)
    if m:
        return '', m.group(1)
    return '', ''


def _var_chain_first_op(obj: str, path: str, ctx: ResolveCtx
                        ) -> Tuple[Optional[FirstOpInfo], str]:
    """对象变量 → 声明类型 → 类首个 API 的解析链。"""
    tname = ctx.var_of(path, obj)
    if not tname:
        return None, ''
    info, via = _resolve_class_first_op(tname, ctx.using_alias, ctx.class_ops)
    if info:
        return info, f'{obj}:{tname}→{via}'
    return None, ''


def _callee_first_op(obj: str, method: str, path: str, ctx: ResolveCtx
                     ) -> Tuple[Optional[FirstOpInfo], str]:
    """被调函数（或其对象变量类型）的首个 API 信息与解析路径。

    对象方法调用优先走变量类型链；operator 等高碰撞名不做全局函数名回退，
    避免把别的类的 operator() 首操作误归属到当前调用点。
    """
    if obj:
        info, via = _var_chain_first_op(obj, path, ctx)
        if info:
            return info, via
        if method == 'operator':
            return None, ''
        info = ctx.func_op(path, method)
        return (info, f'{method}()') if info else (None, '')
    info = ctx.func_op(path, method)
    if info:
        return info, f'{method}()'
    info, via = _var_chain_first_op(method, path, ctx)
    if info:
        return info, via
    # 旧兜底保留：扫描集内仅一个类时直接采用
    if len(ctx.class_ops) == 1:
        cname, cinfo = next(iter(ctx.class_ops.items()))
        return cinfo, f'{cname}(唯一类兜底)'
    return None, ''


def _mismatch_finding(e: SyncEvent, callee: str, call_line: int, info: FirstOpInfo,
                      via: str) -> Finding:
    """生成 SYNC-05 finding；首操作在条件分支内且分支外与 wait 匹配时，给出
    「按同条件分支选择 wait PIPE」修法（如 enableRelu → PIPE_V / 否则 PIPE_MTE3）。"""
    from sync_audit import Finding

    wait_pipe = e.pipe_param
    if info.conditional and info.fallback_pipe and info.fallback_pipe == wait_pipe:
        return Finding(
            'SYNC-05', '高', e.file, e.line,
            f'CrossCoreWaitFlag PIPE 与条件分支内首个操作不匹配: WaitFlag<{wait_pipe}> 后调用 '
            f'{callee}()，条件分支内首个操作是 {info.api}({info.pipe})',
            f'WaitFlag<{wait_pipe}>@{e.line} 后调用 {callee}()@{call_line}（类型解析: {via}）；'
            f'其内部 if/if constexpr 成立路径首操作 {info.api} 归属 {info.pipe}，'
            f'不成立路径首操作 {info.fallback_api} 归属 {info.fallback_pipe}。'
            f'wait 只拦 {wait_pipe}，条件成立时 {info.pipe} 流水可提前读到未就绪数据。'
            f'建议按同条件选择 wait PIPE：if constexpr (条件) '
            f'CrossCoreWaitFlag<..., {info.pipe}>(flag); else '
            f'CrossCoreWaitFlag<..., {info.fallback_pipe}>(flag);')
    cond_note = '（位于 if/if constexpr 分支内）' if info.conditional else ''
    return Finding(
        'SYNC-05', '高', e.file, e.line,
        f'CrossCoreWaitFlag PIPE 方向与后续操作不匹配: WaitFlag<{wait_pipe}> 后调用 {callee}()，'
        f'其内部首个操作是 {info.pipe}{cond_note}',
        f'WaitFlag<{wait_pipe}>@{e.line} 后调用 {callee}()@{call_line}（类型解析: {via}）；'
        f'{callee} 内部首个操作 {info.api} 归属 {info.pipe}，与 {wait_pipe} 不匹配。'
        f'建议：若 {callee} 内部首个操作是 V（如 Relu），WaitFlag 应用 PIPE_V；'
        f'若是 MTE3（如 DataCopyPad），应用 PIPE_MTE3')


def _wait_pipe_mismatch_findings(e: SyncEvent, file_linecode, ctx: ResolveCtx) -> List[Finding]:
    """单个 CrossCoreWaitFlag 与其后首个调用的 PIPE 方向匹配检查。"""
    findings: List[Finding] = []
    lc = file_linecode.get(e.file, {})
    for delta in range(1, 6):
        next_code = lc.get(e.line + delta, '')
        if not next_code.strip():
            continue
        obj, method = _parse_callee(next_code)
        if not method:
            continue
        if method in ('if', 'for', 'while', 'return', 'CrossCoreSetFlag',
                      'CrossCoreWaitFlag', 'SetFlag', 'WaitFlag', 'PipeBarrier',
                      'enable2UB', 'enableUbDB'):
            continue
        info, via = _callee_first_op(obj, method, e.file, ctx)
        if info and info.pipe and info.pipe != e.pipe_param:
            callee = f'{obj}.{method}' if obj else method
            findings.append(_mismatch_finding(e, callee, e.line + delta, info, via))
        break
    return findings


def check_crosscore_pipe_direction(events: List[SyncEvent], lines_by_file) -> List[Finding]:
    """SYNC-05: CrossCoreWaitFlag 的 PIPE 参数与后续首个操作的 PIPE 是否匹配。
    场景：AIV 侧 CrossCoreWaitFlag<..., PIPE_MTE3> 后调用 epilogueOp()，
    若 epilogueOp 内部首个操作是 Relu(PIPE_V)，则 PIPE 方向不匹配 → 同步无效；
    若 Relu 位于 if constexpr (enableRelu) 分支内，则建议按同条件分支选择
    PIPE_V/PIPE_MTE3 两种 wait。被调对象经变量类型链（using 别名/模板形参
    Foo_/类名前缀唯一匹配）解析到具体类；函数/变量表本文件优先，operator 等
    高碰撞名不做全局回退。需要关联文件纳入扫描才能追踪到被调函数内部实现。
    """
    findings: List[Finding] = []
    file_linecode: Dict[str, Dict[int, str]] = {}
    if lines_by_file:
        for path, lns in lines_by_file:
            file_linecode[path] = {ln: code for (ln, _depth, code, _sig) in lns}

    ctx = ResolveCtx()
    _collect_func_first_pipe(lines_by_file, ctx)
    _collect_types_and_aliases(lines_by_file, ctx)
    scope_side_pipes = _scope_pipe_params(events)

    for e in events:
        if e.kind != 'cwait' or not e.pipe_param:
            continue
        # 若同 side 同作用域有多个不同 PIPE 参数，说明已做条件分支，跳过
        key = (e.func, e.side)
        if len(scope_side_pipes.get(key, set())) > 1:
            continue
        findings.extend(_wait_pipe_mismatch_findings(e, file_linecode, ctx))
    return findings