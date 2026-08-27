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
Ascend C 信号同步静态分析工具

扫描 Kernel 源码中的同步 API（SetFlag/WaitFlag、CrossCoreSetFlag/WaitFlag、SyncFunc、
PipeBarrier、SyncAll、ScalarBar、EnQue/DeQue、DataCopy/DataCopyPad、Vector 计算 API），
按 ascendc-sync-audit skill 的 SYNC-01~12 条例产出候选问题清单。

⚠️ 能力边界（重要）：
  - 本工具基于行级正则 + 函数级括号追踪，无完整 C++ 语义分析。
  - 输出是「候选问题」，必须人工结合数据流按 sync-checklist.md 确认，存在误报/漏报。
  - 数据流方向判定（SYNC-05）：HardEvent 字面量校验 + CrossCoreWaitFlag 的 PIPE 与
    被调函数首个 API 所在 PIPE 匹配（含 if constexpr 条件分支双 PIPE 修法建议）；
    被调经变量类型链解析，关联文件未纳入扫描时该项漏检，其余方向判定仍依赖人工。
  - CrossCore 配对需 cube+vec 两个文件都在扫描范围内；单文件扫描时跨核配对仅作提示。
  - 不覆盖：Tiling 侧、HCCL 集合通信内部时序、纯 EnQue/DeQue 内存配对。

用法:
  python3 sync_audit.py <file-or-dir>                 # 全量扫描（默认）
  python3 sync_audit.py <file-or-dir> --check pair    # 仅 Set/Wait 配对
  python3 sync_audit.py <file-or-dir> --check flow    # 仅数据流缺同步启发式
  python3 sync_audit.py <file-or-dir> --list-only     # 仅列出同步点，不做检查
  python3 sync_audit.py <file-or-dir> --format json   # JSON 输出
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple

from sync_logging import init_logging
from sync05_resolve import check_crosscore_pipe_direction

_LOGGER = logging.getLogger('sync_audit')
_STDERR_LOGGER = logging.getLogger('sync_audit.stderr')
init_logging(_LOGGER, _STDERR_LOGGER)

# ─── 同步 API 识别正则 ─────────────────────────────────────────────

SET_WAIT_RE = re.compile(r'\b(SetFlag|WaitFlag)\s*<([^>]*)>\s*\(')
CROSSCORE_RE = re.compile(r'\b(CrossCoreSetFlag|CrossCoreWaitFlag)\s*(?:<[^>]*>)?\s*\(')
CROSSCORE_PIPE_RE = re.compile(r'\b(CrossCoreSetFlag|CrossCoreWaitFlag)\s*<[^,]*,\s*(PIPE_\w+)\s*>')
SYNCFUNC_RE = re.compile(r'\bSyncFunc\s*<([^>]*)>\s*\(')
PIPEBAR_RE = re.compile(r'\bPipeBarrier\s*<([^>]*)>\s*\(')
SYNCALL_RE = re.compile(r'\bSyncAll\s*(?:<[^>]*>)?\s*\(')
SCALARBAR_RE = re.compile(r'\b(SetScalarBar|WaitScalarBar)\s*(?:<[^>]*>)?\s*\(')
# 高层同步封装（cce sysc 同步函数）：MTE2ToSSync()/SToVSync()/VToMTE3Sync() 等，
# 内部隐含 Set/Wait 语义，是合法的同步点，缺其识别会导致"搬入后未同步即计算"误报
HISYNC_RE = re.compile(r'\b((?:AIC|AIV|S|V|MTE2|MTE3|VM|SM)To(?:AIC|AIV|S|V|MTE2|MTE3|VM|SM))Sync\s*\(')
ENQUE_RE = re.compile(r'\bEnQue\s*\(')
DEQUE_RE = re.compile(r'\bDeQue\s*(?:<[^>]*>)?\s*\(')
ALLOCTENSOR_RE = re.compile(r'\bAllocTensor\s*(?:<[^>]*>)?\s*\(')
DATACOPY_RE = re.compile(r'(?<!Reg::)(?<!Te::)(?<!MicroAPI::)\bDataCopy(?:Pad)?(?:Ext)?\s*(?:<[^>]*>)?\s*\(')
# MicroAPI 寄存器级操作（不跨流水，不需要同步）
MICROAPI_RE = re.compile(r'\bMicroAPI::\w+\s*(?:<[^>]*>)?\s*\(|\bDataCopy(?:UnAlign|UnAlignPre)(?:<[^>]*>)?\s*\(')
# 自定义搬出/搬入函数命名模式（CopyOut*/CopyIn*/StoreResult* 等）
COPYOUT_RE = re.compile(r'\bCopyOut\w*\s*\(')
COPYIN_RE = re.compile(r'\bCopyIn\w*\s*\(')
RETURN_RE = re.compile(r'\b(?:return|break|continue|goto)\b')
# Atomic 操作检测（SYNC-13）
ATOMIC_ADD_RE = re.compile(r'\bSetAtomicAdd\s*(?:<[^>]*>)?\s*\(')
ATOMIC_DISABLE_RE = re.compile(r'\bDisableDmaAtomic\s*\(')
FIXPIPE_RE = re.compile(r'\bFixpipe\s*\(')
DATACOPY_CO12_RE = re.compile(r'\bDataCopy\s*\([^)]*cGlobal|CopyL0C2GM')

# Reg::LocalMemBar — DSL vmem_bar("vst_vld") 的 Ascend C 原生形态
# 生产代码 51 处（ops-nn/ops-tensor/ops-transformer），V 内 store→load 屏障
REG_LOCALMEMBAR_RE = re.compile(r'\bReg::LocalMemBar\s*<[^>]*>')
# Reg:: Store/Load — DSL vstore/vload 的 Ascend C 原生形态（V 内数据流）
REG_STORE_RE = re.compile(r'\bReg::(?:StoreAlign|StoreUnAlign|StoreUnAlignPost)\s*(?:<[^>]*>)?\s*\(')
REG_LOAD_RE = re.compile(
    r'\bReg::(?:LoadAlign|LoadUnAlign|LoadUnAlignPre|DataCopyUnAlign|'
    r'DataCopyUnAlignPre|DataCopyUnAlignPost)\s*(?:<[^>]*>)?\s*\(')
# Reg::DataCopy — 方向由参数顺序判定（mem,reg)=store / (reg,mem)=load
REG_DATACOPY_RE = re.compile(r'\bReg::DataCopy\s*(?:<[^>]*>)?\s*\(')

# Cube 侧 Te::Copy 搬运（L1↔L0、L0C↔GM 等，通过 MakeCopy 类型识别方向）
TE_COPY_RE = re.compile(r'\bTe::Copy\s*\(')
TE_COPY_L12L0_RE = re.compile(r'CopyL12L0[AB]')
TE_COPY_L0C2GM_RE = re.compile(r'CopyL0C2GM')
MMAD_RE = re.compile(r'\bTe::Mmad\s*\(')

# Vector 计算 API（用于数据流缺同步启发式）
COMPUTE_OPS = re.compile(
    r'\b(Adds?|Muls?|Sub|Mul|Div|Exp|Sqrt|Log|Cast|ReduceMax|ReduceMin|ReduceSum|'
    r'Compare|Max|Min|Abs|Duplicate|And|Or|Not|Mmad|Fixpipe|FixPipe|Brcb|Gather|TransDataTo5HD|'
    r'Relu|LeakyRelu|Rsqrt|Reciprocal|Axpy|FusedMulAdd|MulAddRelu|AddRelu|SubRelu|'
    r'AddReluCast|SubReluCast|MulCast|AddDeqRelu|CastDequant|WholeReduceMax|WholeReduceMin|WholeReduceSum|'
    r'BlockReduceMax|BlockReduceMin|BlockReduceSum|PairReduceSum|RepeatReduceSum|Transpose|'
    r'ShiftLeft|ShiftRight|Maxs|Mins|Select|GatherMask|CreateVecIndex|SetDeqScale|'
    r'SetMaskCount|SetMaskNorm|SetVectorMask|ResetMask)\s*(?:<[^>]*>)?\s*\(')

# Scalar 操作（PIPE_S）：GetValue/SetValue 是 V→S / S→V 跨流水依赖的关键检测点
SCALAR_OPS = re.compile(r'\b(GetValue|SetValue)\s*\(')

# ─── 函数级扫描正则（函数内复用的局部正则，模块级常量）──────────────

SIG_RE = re.compile(
    r'^\s*(?:template\s*<[^>]*>\s*)?'
    r'(?:__aicore__\s+|inline\s+|static\s+|constexpr\s+|virtual\s+|explicit\s+)*'
    r'[\w:<>,\s\*&]+?\s+\*?\s*[\w:]+\s*\(')
CTRL_RE = re.compile(r'^\s*(?:if|for|while|switch|do|else|catch|class|struct|enum|namespace|union)\b')
SIG_NAME_RE = re.compile(
    r'((?:__aicore__|inline|static|constexpr|virtual|explicit|[\w:<>,\s\*&])+\s+\*?\s*[\w:]+)\s*\(')
SIDE_AIC_RE = re.compile(r'\bASCEND_IS_AIC\b')
SIDE_AIV_RE = re.compile(r'\bASCEND_IS_AIV\b')
SYNC_CALL_RE = re.compile(r'\b(?:CrossCoreSetFlag|CrossCoreWaitFlag|SetFlag|WaitFlag)\b')
LOOP_RE = re.compile(r'\b(?:for|while)\s*\(')
CONSTANT_FLAG_RE = re.compile(r'^\d+$')
CONSTANT_RE = re.compile(r'^\d+$')

TENSOR_DECL_RE = re.compile(r'(?:LocalTensor|auto)\s*<?\w*?>?\s+(\w+)\s*=')
DC_WRITE_RE = re.compile(
    r'(?<!Reg::)(?<!Te::)(?<!MicroAPI::)\bDataCopy(?:Pad)?(?:Ext)?(?:<[^>]*>)?\s*\(\s*(\w+)')
COMPUTE_WRITE_RE = re.compile(
    r'\b(Adds?|Muls?|Sub|Mul|Div|Exp|Sqrt|Log|Cast|ReduceMax|ReduceMin|ReduceSum|'
    r'Compare|Abs|Duplicate|Relu|LeakyRelu|Rsqrt|Reciprocal|Axpy|FusedMulAdd|'
    r'Gather|Brcb|Transpose)\s*(?:<[^>]*>)?\s*\(\s*(\w+)')
GETVALUE_RE = re.compile(r'(\w+)\.GetValue\s*\(')
SYNC_RE = re.compile(r'\b(EnQue|DeQue|SetFlag|WaitFlag|PipeBarrier|SyncFunc)\b')
MICROAPI_LINE_RE = re.compile(r'MicroAPI::|__local_mem__|RegTensor')
REGTENSOR_DECL_RE = re.compile(r'RegTensor\s*<[^>]*>\s+([\w\s,]+?)(?:=|;|$)')
MICRO_REG_RE = re.compile(r'(?:UnalignReg|MaskReg)\s+([\w\s,]+?)(?:=|;|$)')
BUFFER_INDEX_RE = re.compile(r'(\w+_\w*|\w+Buf\w*|\w+Buffer\w*|\w+Ub\w*|\w+Local\w*)\s*\[\s*([^]]+)\]')
VAR_NAME_RE = re.compile(r'\b([a-z][a-zA-Z0-9_]+)\b')
SET_WAIT_CALL_RE = re.compile(r'\b(SetFlag|WaitFlag)\s*<[^>]*>\s*\(')
DATA_CALL_RE = re.compile(r'\b(DataCopy|DataCopyPad|Adds?|Muls?|Cast|Duplicate|Relu|Abs|'
                          r'ReduceMax|ReduceMin|ReduceSum)\b')
FIELD_ASSIGN_RE = re.compile(r'\b(\w+)\s*(?:\.|->)\s*(\w+)\s*=')
VAR_DECL_ASSIGN_RE = re.compile(
    r'\b(?:const\s+)?(?:uint\d+_t|int\d+_t|uint|int|auto)\s+([a-zA-Z_]\w*)\s*=\s*([^;]+);')
SKIP_VARS = {'true', 'false', 'nullptr', 'static_cast', 'reinterpret_cast', 'const_cast'}
OUTPUT_FIELD_RE = re.compile(r'(out|output|dst|highbit|f16|result)', re.IGNORECASE)
OUTPUT_BUFFER_RE = re.compile(r'(Out|Output|HighBit|Result|Dst)', re.IGNORECASE)

SYNC_ANY_RE = re.compile(r'\b(SetFlag|WaitFlag|CrossCoreSetFlag|CrossCoreWaitFlag|PipeBarrier|'
                         r'SyncFunc|SyncAll|EnQue|DeQue|ScalarBar)\b')
MAX_OP_DISTANCE = 20
VAR_STOP = {'constexpr', 'template', 'typename', 'static', 'return', 'if', 'for', 'while',
            'true', 'false', 'float', 'int', 'uint', 'int32', 'uint32', 'int64', 'uint64',
            'int16', 'uint16', 'double', 'void', 'auto', 'bool', 'char', 'size', 'int8',
            'uint8', 'GetTPipePtr', 'nullptr', 'this', 'new', 'byte', 'half'}

REVERSE_MAP = {
    'MTE2_V': 'V_MTE2',
    'V_MTE3': 'MTE3_V',
    'V_S': 'S_V',
    'MTE2_S': 'S_MTE2',
}

SYNC_OP_NAMES = {'SetFlag', 'WaitFlag', 'CrossCoreSetFlag', 'CrossCoreWaitFlag',
                 'SyncFunc', 'PipeBarrier', 'SyncAll', 'SetScalarBar', 'WaitScalarBar',
                 'EnQue', 'DeQue'}

# check_loop_underflow 用：无符号下溢风险检测
UNDERFLOW_RE = re.compile(r'\(\s*(\w+)\s*-\s*(\d+)\s*\)\s*%')
UINT_TYPE_RE = re.compile(r'\b(?:uint\d+_t|size_t)\b')
UINT_VAR_DECL_RE = re.compile(r'\b(?:uint\d+_t|size_t|int\d+_t)\s+(\w+)')

# 已知合法 HardEvent 方向（来源：CANN 官方文档 + ops-nn/ops-tensor/ops-blas 真实代码全集 31 种）
# 用于 SYNC-05 方向合法性校验：脚本无法判定方向是否匹配数据流，但能识别未知方向（如 MTE2_V 写成 MET2_V）
KNOWN_HARDEVENTS = {
    'MTE2_V', 'V_MTE3', 'V_M', 'M_V', 'V_V', 'MTE1_M', 'V_S', 'S_V',
    'S_MTE2', 'S_MTE3', 'MTE2_S', 'MTE3_S', 'MTE2_MTE3', 'MTE3_MTE2',
    'MTE2_MTE1', 'MTE1_MTE2', 'MTE1_MTE3', 'MTE3_MTE1', 'MTE2_M', 'M_MTE2',
    'M_MTE1', 'M_FIX', 'FIX_M', 'FIX_V', 'FIX_MTE1', 'FIX_MTE2', 'MTE2_FIX',
    'V_MTE2', 'MTE3_V', 'FIX_MTE3', 'M_S', 'FIX_S',
}

PIPE_PREFIX_MAP = {'V': 'V', 'MTE2': 'MTE2', 'MTE3': 'MTE3', 'M': 'M', 'S': 'S', 'MTE1': 'MTE1'}


# ─── 数据结构 ──────────────────────────────────────────────────────

@dataclass
class SyncEvent:
    kind: str            # set|wait|cset|cwait|syncfunc|pipebarrier|syncall|scalarbar
    event_type: str      # 归一化方向，如 V_MTE3 / PIPE_V / flag 变量
    flag_id: str         # EVENT_ID 字面量或 flag 变量
    line: int
    file: str
    func: str            # 所在函数/作用域签名
    raw: str
    side: str = ''       # AIC|AIV|''  所在侧别块（if ASCEND_IS_AIC/AIV）
    pipe_param: str = ''  # CrossCore 的 PIPE 参数，如 PIPE_MTE3/PIPE_V/PIPE_FIX


@dataclass
class Finding:
    code: str            # SYNC-01 .. SYNC-12
    severity: str        # 红线|高|性能|信息
    file: str
    line: int
    message: str
    detail: str = ''


@dataclass
class ScanResult:
    findings: List[Finding] = field(default_factory=list)
    events: List[SyncEvent] = field(default_factory=list)
    files_scanned: List[str] = field(default_factory=list)


@dataclass
class ScanContext:
    """逐行扫描上下文：文件、作用域签名、侧别、事件收集表。"""

    file: str
    func: str
    side: str
    events: List[SyncEvent]


@dataclass
class LineMaps:
    """行级索引：行号→代码/深度映射、循环体行集合、循环体基准深度。"""

    file_linecode: Dict[str, Dict[int, str]]
    file_loop_lines: Dict[str, Set[int]]
    file_depth: Dict[str, Dict[int, int]]
    file_loop_body_depth: Dict[str, Dict[int, int]]


@dataclass
class ScopeState:
    """scan_file 的花括号作用域状态：函数/块作用域栈、侧别栈、括号深度。"""

    scope_stack: List[str]
    side_stack: List[str]
    depth: int


@dataclass
class GroupStats:
    """单个 Set/Wait 组的统计切片（供配对检查使用）。"""

    ev: str
    flag: str
    sets: List[SyncEvent]
    waits: List[SyncEvent]
    pre_sets: List[SyncEvent]
    in_sets: List[SyncEvent]
    pre_waits: List[SyncEvent]
    in_waits: List[SyncEvent]
    in_sets_uncond: List[SyncEvent]
    in_waits_uncond: List[SyncEvent]
    ref: SyncEvent
    alias_skip_count: bool


# ─── 预处理：去注释/字符串 ─────────────────────────────────────────

def _skip_string_literal(line: str, i: int, quote: str, out: List[str]) -> int:
    """跳过字符串/字符字面量（双引号或单引号），追加占位并返回新的下标。"""
    out.append('""' if quote == '"' else "''")
    i += 1
    n = len(line)
    while i < n and line[i] != quote:
        if line[i] == '\\' and i + 1 < n:
            i += 2
            continue
        i += 1
    return i + 1


def strip_code(line: str) -> str:
    """去除行注释与字符串字面量（粗略，逐行）。块注释由调用方跨行维护状态。"""
    out = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == '/' and i + 1 < n and line[i + 1] == '/':
            break
        if c in ('"', "'"):
            i = _skip_string_literal(line, i, c, out)
            continue
        out.append(c)
        i += 1
    return ''.join(out)


def normalize_event(angle_content: str) -> str:
    """归一化 <...> 内容为方向键。
    AscendC::HardEvent::V_MTE3 -> V_MTE3
    HardEvent::V_MTE3 -> V_MTE3
    PIPE_V, PIPE_MTE3 -> V_MTE3（双 PIPE 同步，取后半）
    PIPE_V -> PIPE_V（单 PIPE 屏障，保留前缀区分）
    """
    s = angle_content.strip()
    s = s.replace('AscendC::', '').replace('HardEvent::', '').replace(' ', '')
    if ',' in s:
        parts = [p.replace('PIPE_', '') for p in s.split(',')]
        # SetFlag<PIPE_V, PIPE_MTE3> 表示 V→MTE3 方向，取末段作为方向键
        return parts[-1] if len(parts) == 2 else '_'.join(parts)
    if s.startswith('PIPE_'):
        return s  # 单 PIPE 屏障保留 PIPE_ 前缀
    return s


# ─── 函数作用域追踪 ────────────────────────────────────────────────

@dataclass
class Scope:
    sig: str
    start_line: int
    block_id: int


def _strip_block_comment(code: str, in_block_comment: bool) -> Tuple[str, bool]:
    """剥离行内块注释边界，返回（处理后的代码, 是否进入跨行块注释）。"""
    while True:
        bc_start = code.find('/*')
        if bc_start == -1:
            return code, in_block_comment
        bc_end = code.find('*/', bc_start + 2)
        if bc_end == -1:
            return code[:bc_start], True
        code = code[:bc_start] + code[bc_end + 2:]


def _merge_sync_call_lines(code: str, raw_lines: List[str], idx: int) -> str:
    """跨行合并：含同步 API 且括号未平衡时，拼接后续行直到括号平衡。"""
    if not SYNC_CALL_RE.search(code) or code.count('(') <= code.count(')'):
        return code
    merged = code
    j = idx
    while merged.count('(') > merged.count(')') and j < len(raw_lines):
        j += 1
        nl = strip_code(raw_lines[j - 1].rstrip('\n'))
        merged += ' ' + nl
    return merged


def _looks_like_signature(code: str, stripped: str, is_expr_statement: bool) -> bool:
    if is_expr_statement:
        return False
    if not SIG_RE.search(code):
        return False
    if CTRL_RE.search(code):
        return False
    return not stripped.endswith(';')


def _looks_like_macro_method(code: str, stripped: str, is_expr_statement: bool) -> bool:
    if is_expr_statement:
        return False
    if not re.search(r'::\w+\s*\(', code):
        return False
    if CTRL_RE.search(code):
        return False
    return not stripped.endswith(';')


def _pending_signature(code: str, stripped: str,
                       pending_sig: Optional[str]) -> Optional[str]:
    """识别函数签名（含跨行 pending），返回新的 pending_sig。

    pending 持锁修复（迭代六）：签名行以 ; 结尾且括号平衡（声明而非定义）时不挂起，
    避免把后续函数定义误归到前一个挂起签名下。
    注意：不排除以逗号结尾的行（跨行函数签名参数以逗号结尾，如 ...::Method(A,
    b)；排除仅限赋值（=）与行尾二元运算符（迭代六：逗号误排导致签名漏识别）
    """
    if pending_sig is not None:
        return pending_sig
    is_expr_statement = ('=' in stripped
                         or re.search(r'[\*+\-/%.&|<>=]\s*$', stripped))
    if _looks_like_signature(code, stripped, is_expr_statement):
        m = SIG_NAME_RE.search(code)
        if not m:
            return None
        sig = m.group(1).strip()
        if stripped.endswith(';') and code.count('(') == code.count(')'):
            return None
        return sig
    if _looks_like_macro_method(code, stripped, is_expr_statement):
        mm = re.search(r'(\w+(?:<[^>]*>)?)::(\w+)\s*\(', code)
        if mm:
            return mm.group(2)  # 方法名
    return None


def _apply_brace(ch: str, state: ScopeState, pending_sig: Optional[str],
                pending_side: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """处理单个花括号字符对作用域/侧别栈与深度的更新，返回新的 (pending_sig, pending_side)。

    '{'：挂起签名入栈为函数作用域（侧别重置），否则普通块继承当前作用域；
    AIC/AIV 侧别块标记 pending_side 在进入块时消费。
    """
    if ch == '}':
        if len(state.scope_stack) > 1:
            state.scope_stack.pop()
            state.side_stack.pop()
        state.depth = max(0, state.depth - 1)
        return pending_sig, pending_side
    if ch != '{':
        return pending_sig, pending_side
    if pending_sig:
        state.scope_stack.append(pending_sig)   # 函数作用域
        state.side_stack.append('')              # 函数体重置侧别
        state.depth += 1
        return None, pending_side
    state.scope_stack.append(state.scope_stack[-1])  # 非函数块继承当前作用域
    if pending_side:
        state.side_stack.append(pending_side)   # AIC/AIV 块
        state.depth += 1
        return None, None
    state.side_stack.append(state.side_stack[-1])  # 普通块继承
    state.depth += 1
    return None, pending_side


def scan_file(path: str) -> Tuple[List[SyncEvent], List[Tuple[int, int, str, str]]]:
    """扫描单文件，返回 (同步事件列表, (行号, 括号深度, 代码, 作用域签名) 列表)。"""
    events: List[SyncEvent] = []
    lines_info: List[Tuple[int, int, str, str]] = []
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            raw_lines = f.readlines()
    except OSError as e:
        _STDERR_LOGGER.warning("警告: 无法读取 %s: %s", path, e)
        return events, lines_info

    state = ScopeState(['<global>'], [''], 0)
    current_sig = '<global>'
    current_side = ''
    in_block_comment = False
    pending_sig: Optional[str] = None
    pending_side: Optional[str] = None  # 待入栈的侧别（if ASCEND_IS_AIC/AIV 后的块）

    for idx, raw in enumerate(raw_lines, start=1):
        line = raw.rstrip('\n')
        if in_block_comment:
            end = line.find('*/')
            if end == -1:
                lines_info.append((idx, state.depth, '', current_sig))
                continue
            line = line[end + 2:]
            in_block_comment = False

        code = strip_code(line)
        code, in_block_comment = _strip_block_comment(code, in_block_comment)
        code = _merge_sync_call_lines(code, raw_lines, idx)

        lines_info.append((idx, state.depth, code, current_sig))

        stripped = code.strip()
        pending_sig = _pending_signature(code, stripped, pending_sig)

        # 侧别块识别：if ASCEND_IS_AIC/AIV 后的 { 块标记侧别
        if SIDE_AIC_RE.search(code):
            pending_side = 'AIC'
        elif SIDE_AIV_RE.search(code):
            pending_side = 'AIV'

        for ch in code:
            pending_sig, pending_side = _apply_brace(ch, state, pending_sig, pending_side)
        current_sig = state.scope_stack[-1]
        current_side = state.side_stack[-1]

        _extract_events(code, idx, ScanContext(path, current_sig, current_side, events))

    return events, lines_info


def extract_balanced_arg(code: str, open_paren_pos: int) -> str:
    """从 code[open_paren_pos]=='(' 开始，提取括号匹配的完整内容。
    处理嵌套括号（static_cast<uint16_t>(l1BufId) 等）。
    返回括号内的原始字符串（未归一化）。
    """
    depth = 0
    start = open_paren_pos + 1
    for i in range(open_paren_pos, len(code)):
        if code[i] == '(':
            depth += 1
        elif code[i] == ')':
            depth -= 1
            if depth == 0:
                return code[start:i].strip()
    return code[start:].strip()  # 未闭合，返回剩余


def normalize_flag_id(raw: str) -> str:
    """归一化 flag_id：只剥类型转换，保留偏移表达式。
    static_cast<uint16_t>(l1BufId)        -> l1BufId       (剥 cast)
    static_cast<uint16_t>(l1BufId + 1)    -> l1BufId + 1   (剥 cast，保留偏移)
    (uint16_t)l1BufId                      -> l1BufId       (剥 C-style cast)
    AIV_SYNC_AIC_FLAG + FLAG_ID_MAX        -> AIV_SYNC_AIC_FLAG + FLAG_ID_MAX  (保留偏移！)
    AIV_SYNC_AIC_FLAG + 1                  -> AIV_SYNC_AIC_FLAG + 1            (保留偏移！)
    l1BufId                                -> l1BufId
    l1BufId & 1 / l1BufId % 2              -> l1BufId      (剥 ping-pong 索引掩码/取模！
                                                    位与/取模是同一 flag 的 ping-pong 槽位，
                                                    与无掩码写法同组配对；+N 偏移则是不同事件)
    配对按完整表达式，ping/pong 偏移区分配对，避免漏报。
    """
    s = raw.strip()
    # static_cast<T>(X) -> X
    m = re.match(r'static_cast\s*<[^>]*>\s*\((.*)\)\s*$', s)
    if m:
        s = m.group(1).strip()
    # C-style cast (uint16_t)X -> X
    m = re.match(r'\(\s*(?:uint\d+_t|int\d+_t|uint16_t)\s*\)\s*(.*)$', s)
    if m:
        s = m.group(1).strip()
    # ping-pong 索引掩码/取模（2026-08-21 迭代七）：X & N / X % N -> X
    # （同变量同 flag 的不同槽位写法，如 l0aPingPongFlag_ 与 l0aPingPongFlag_ & 1 应同组）
    m = re.match(r'^(.*?)\s*(?:&\s*\d+\s*|\%\s*\d+\s*)$', s)
    if m:
        return m.group(1).strip()
    return s


def _extract_events(code: str, line: int, ctx: ScanContext):
    for m in SET_WAIT_RE.finditer(code):
        kind = 'set' if m.group(1) == 'SetFlag' else 'wait'
        ev = normalize_event(m.group(2))
        paren_pos = code.index('(', m.start())
        flag = normalize_flag_id(extract_balanced_arg(code, paren_pos))
        ctx.events.append(SyncEvent(kind, ev, flag, line, ctx.file, ctx.func, m.group(0), ctx.side))
    for m in CROSSCORE_RE.finditer(code):
        kind = 'cset' if m.group(1) == 'CrossCoreSetFlag' else 'cwait'
        paren_pos = code.index('(', m.start())
        flag = normalize_flag_id(extract_balanced_arg(code, paren_pos))
        # 提取 PIPE 参数（如 PIPE_MTE3/PIPE_V/PIPE_FIX）
        pipe_param = ''
        pipe_m = CROSSCORE_PIPE_RE.search(code[m.start():m.end() + 5])
        if pipe_m:
            pipe_param = pipe_m.group(2)
        ctx.events.append(SyncEvent(kind, 'crosscore', flag, line, ctx.file, ctx.func, m.group(0),
                                    ctx.side, pipe_param))
    for m in SYNCFUNC_RE.finditer(code):
        ctx.events.append(SyncEvent('syncfunc', normalize_event(m.group(1)), '', line, ctx.file,
                                    ctx.func, m.group(0)))
    for m in PIPEBAR_RE.finditer(code):
        ev = 'PIPE_' + normalize_event(m.group(1)).replace('PIPE_', '')
        ctx.events.append(SyncEvent('pipebarrier', ev, '', line, ctx.file, ctx.func, m.group(0)))
    for m in SYNCALL_RE.finditer(code):
        ctx.events.append(SyncEvent('syncall', 'syncall', '', line, ctx.file, ctx.func, m.group(0)))
    for m in SCALARBAR_RE.finditer(code):
        kind = 'scalarbar_set' if m.group(1) == 'SetScalarBar' else 'scalarbar_wait'
        ctx.events.append(SyncEvent(kind, 'scalarbar', '', line, ctx.file, ctx.func, m.group(0)))
    for m in REG_LOCALMEMBAR_RE.finditer(code):
        ctx.events.append(SyncEvent('reg_membar', 'V_V', '', line, ctx.file, ctx.func, m.group(0)))


# ─── 检查实现 ──────────────────────────────────────────────────────

def _build_line_maps(lines_by_file) -> LineMaps:
    """构建行级索引：行号→代码/深度、循环体行集合、循环体基准深度。"""
    file_linecode = {}
    file_loop_lines = {}
    file_depth = {}
    file_loop_body_depth = {}
    if not lines_by_file:
        return LineMaps(file_linecode, file_loop_lines, file_depth, file_loop_body_depth)
    for path, lns in lines_by_file:
        file_linecode[path] = {ln: code for (ln, _depth, code, _sig) in lns}
        file_depth[path] = {ln: depth for (ln, depth, _code, _sig) in lns}
        loop_lines = set()
        loop_body = {}
        for i, (ln, _depth, code, _sig) in enumerate(lns):
            if not LOOP_RE.search(code):
                continue
            body_lines, body_depth = _scan_loop_body(lns, i)
            loop_lines |= body_lines
            loop_body[ln] = body_depth
        file_loop_lines[path] = loop_lines
        file_loop_body_depth[path] = loop_body
    return LineMaps(file_linecode, file_loop_lines, file_depth, file_loop_body_depth)


def _scan_loop_body(lns, i: int):
    """从循环头行 i 之后收集循环体行号集合与首行深度。"""
    loop_depth = lns[i][1]
    loop_lines = set()
    body_depth = None
    for j in range(i + 1, len(lns)):
        ln2, d2, _c2, _s2 = lns[j]
        if d2 <= loop_depth:
            break
        loop_lines.add(ln2)
        if body_depth is None:
            body_depth = d2
    return loop_lines, body_depth or loop_depth + 1


def _is_in_loop(e: SyncEvent, file_loop_lines) -> bool:
    return e.line in file_loop_lines.get(e.file, set())


def _is_loop_top_level(e: SyncEvent, maps: LineMaps) -> bool:
    """循环内语句是否在循环体直接层（无条件执行，非 if/else 嵌套块）。"""
    loop_headers = []
    for lh, _bd in maps.file_loop_body_depth.get(e.file, {}).items():
        if e.line in maps.file_loop_lines.get(e.file, set()) and e.line > lh:
            loop_headers.append(lh)
    if not loop_headers:
        # 无循环头引用时退回：所在行深度 == 该文件最小循环体深度
        bodies = [bd for bd in maps.file_loop_body_depth.get(e.file, {}).values()]
        return bool(bodies) and maps.file_depth.get(e.file, {}).get(e.line) <= min(bodies)
    body_min = min(maps.file_loop_body_depth.get(e.file, {}).get(lh, 999) for lh in loop_headers)
    return maps.file_depth.get(e.file, {}).get(e.line, 999) <= body_min


def _group_key(e: SyncEvent, direction_flags: Dict[str, Set[str]]) -> Tuple[str, str]:
    """Set/Wait 分组键：常量与变量 flag 混合时归入 __mixed__ 组。"""
    flags = direction_flags.get(e.event_type, set())
    has_const = any(CONSTANT_FLAG_RE.match(f) for f in flags)
    has_var = any(not CONSTANT_FLAG_RE.match(f) for f in flags)
    if has_const and has_var:
        return (e.event_type, '__mixed__')
    return (e.event_type, e.flag_id)


def _build_event_groups(events: List[SyncEvent]):
    """按 (event_type, flag_id) 跨函数分组的 Set/Wait 事件。"""
    groups: Dict[Tuple[str, str], List[SyncEvent]] = {}
    direction_flags: Dict[str, Set[str]] = {}
    for e in events:
        if e.kind in ('set', 'wait'):
            direction_flags.setdefault(e.event_type, set()).add(e.flag_id)
    for e in events:
        if e.kind in ('set', 'wait'):
            groups.setdefault(_group_key(e, direction_flags), []).append(e)
    return groups


def _classify_pure_groups(subgroups):
    """把子组分为「纯 Set 组」与「纯 Wait 组」，其余混合组忽略。"""
    pure_sets = {}
    pure_waits = {}
    for flag, evs in subgroups:
        ns = sum(1 for e in evs if e.kind == 'set')
        nw = len(evs) - ns
        if ns and not nw:
            pure_sets[flag] = (ns, evs)
        elif nw and not ns:
            pure_waits[flag] = (nw, evs)
    return pure_sets, pure_waits


def _is_balanced_alias(pure_sets, pure_waits) -> bool:
    """纯 Set 组与纯 Wait 组计数相等、同文件集合且均为变量别名 → 可合并。

    常量 flag（EVENT_ID0=0 / EVENT_ID1=1 具名宏或数字）是不同事件（sync04b 场景），
    只有变量/成员表达式别名（小写变量如 e12 vs eventId12）才可合并。
    """
    fsets = sum(n for n, _ in pure_sets.values())
    fwaits = sum(n for n, _ in pure_waits.values())
    if fsets != fwaits:
        return False
    all_var_flags = all(
        bool(re.search(r'[a-z]', f.strip()))
        for f in list(pure_sets) + list(pure_waits))
    if not all_var_flags:
        return False
    files_s = {e.file for _, evs in pure_sets.values() for e in evs}
    files_w = {e.file for _, evs in pure_waits.values() for e in evs}
    return files_s == files_w


def _compute_pure_balance(groups: Dict[Tuple[str, str], List[SyncEvent]]) -> Set[str]:
    """别名预聚合（迭代七）：同方向下"纯 Set 组 + 纯 Wait 组"且计数相等 → 同一事件别名。

    约束：两组事件必须同文件集合，避免跨文件 Cube/Vec 两侧真实拆分被误合并。
    """
    ev_pure_balance: Set[str] = set()
    ev_groups: Dict[str, List[Tuple[str, List[SyncEvent]]]] = defaultdict(list)
    for (ev, flag), evs in groups.items():
        ev_groups[ev].append((flag, evs))
    for ev, subgroups in ev_groups.items():
        pure_sets, pure_waits = _classify_pure_groups(subgroups)
        if pure_sets and pure_waits and _is_balanced_alias(pure_sets, pure_waits):
            ev_pure_balance.add(ev)
    return ev_pure_balance


def _legal_loop_extra_set(total_diff: int, in_sets: List[SyncEvent], pre_waits: List[SyncEvent],
                          pre_sets: List[SyncEvent]) -> bool:
    """循环内多 Set、循环外少 Wait 是否属于合法模式（每个 tile 末尾 Set，仅最后收尾 Wait）。"""
    if total_diff <= 0:
        return False
    if len(in_sets) < total_diff:
        return False
    if len(pre_waits) > 1:
        return False
    return len(pre_sets) <= 1


def _conditional_wait_imbalance(in_sets: List[SyncEvent], in_waits: List[SyncEvent],
                                in_sets_uncond: List[SyncEvent],
                                in_waits_uncond: List[SyncEvent]) -> bool:
    """循环内 Wait 位于条件块内（部分迭代缺失）且无条件 Set 多于无条件 Wait。"""
    if not in_sets or not in_waits:
        return False
    if len(in_waits_uncond) >= len(in_sets_uncond):
        return False
    return len(in_waits) > len(in_waits_uncond)


def _build_group_stats(ev: str, flag: str, evs: List[SyncEvent], ev_pure_balance: Set[str],
                       maps: LineMaps) -> GroupStats:
    ref = evs[0]
    sets = [e for e in evs if e.kind == 'set']
    waits = [e for e in evs if e.kind == 'wait']
    pre_sets = [e for e in sets if not _is_in_loop(e, maps.file_loop_lines)]
    in_sets = [e for e in sets if _is_in_loop(e, maps.file_loop_lines)]
    pre_waits = [e for e in waits if not _is_in_loop(e, maps.file_loop_lines)]
    in_waits = [e for e in waits if _is_in_loop(e, maps.file_loop_lines)]
    return GroupStats(
        ev=ev,
        flag=flag,
        sets=sets,
        waits=waits,
        pre_sets=pre_sets,
        in_sets=in_sets,
        pre_waits=pre_waits,
        in_waits=in_waits,
        in_sets_uncond=[s for s in in_sets if _is_loop_top_level(s, maps)],
        in_waits_uncond=[w for w in in_waits if _is_loop_top_level(w, maps)],
        ref=ref,
        alias_skip_count=ev in ev_pure_balance,
    )


def _check_total_balance(stats: GroupStats) -> List[Finding]:
    """Set/Wait 个数平衡检查（SYNC-04 / 部分 SYNC-02 候选）。"""
    findings: List[Finding] = []
    ev, flag = stats.ev, stats.flag
    sets, waits = stats.sets, stats.waits
    in_sets, in_waits = stats.in_sets, stats.in_waits
    pre_sets, pre_waits = stats.pre_sets, stats.pre_waits
    in_sets_uncond, in_waits_uncond = stats.in_sets_uncond, stats.in_waits_uncond
    ref = stats.ref
    total_diff = len(sets) - len(waits)
    if total_diff != 0 and not stats.alias_skip_count:
        if _legal_loop_extra_set(total_diff, in_sets, pre_waits, pre_sets):
            # 降级为存疑：循环内多 Set、循环外少 Wait 可能是 buffer 复用缺循环顶部 Wait
            s0 = in_sets[0] if in_sets else ref
            findings.append(Finding(
                'SYNC-02', '高', s0.file, s0.line,
                f'循环内 Set 多于 Wait（疑似 buffer 复用缺循环顶部 Wait）: 方向={ev} flag={flag}',
                f'循环内 Set={len(in_sets)} Wait={len(in_waits)}，多出 {total_diff} 个 Set 无 Wait 消费；'
                f'若涉及 L0C/buffer 复用（如 FIX_M/MTE1_M），循环顶部应补 Wait 等上一轮搬完'))
        else:
            set_locs = [f"{os.path.basename(s.file)}:{s.line}" for s in sets]
            wait_locs = [f"{os.path.basename(w.file)}:{w.line}" for w in waits]
            findings.append(Finding(
                'SYNC-04', '红线', ref.file, ref.line,
                f'Set/Wait 个数不一致: 方向={ev} flag={flag} Set={len(sets)} Wait={len(waits)}',
                f'Set: {set_locs}; Wait: {wait_locs}'))
    # total 相等但循环外有 Wait 无 Set，且循环内有 Set → 缺初始 Set（defect01 模式）
    elif pre_waits and not pre_sets and in_sets:
        w0 = pre_waits[0]
        findings.append(Finding(
            'SYNC-04', '红线', w0.file, w0.line,
            f'循环外 Wait 缺初始 Set（死等风险）: 方向={ev} flag={flag}',
            f'循环外 Wait@{w0.line} 无对应初始 Set；循环内 Set={len(in_sets)} Wait={len(in_waits)}，'
            f'收尾 Wait 无初始 Set 供首轮消费 → 首轮死等'))
    elif _conditional_wait_imbalance(in_sets, in_waits, in_sets_uncond, in_waits_uncond):
        s0 = in_sets[0]
        findings.append(Finding(
            'SYNC-02', '高', s0.file, s0.line,
            f'循环内 Set 多于（无条件）Wait（Wait 位于条件块内，部分迭代缺失）: 方向={ev} flag={flag}',
            f'循环内无条件 Set={len(in_sets_uncond)}、无条件 Wait={len(in_waits_uncond)}（'
            f'另有 {len(in_waits) - len(in_waits_uncond)} 个 Wait 在 if 条件块内）；'
            f'若某轮 Set 后无 Wait 消费，buffer 会被提前覆盖 → 数据错乱'))
    # 无 Set 的 Wait（别名平衡场景跳过计数类）
    if not sets and waits and not stats.alias_skip_count:
        w0 = waits[0]
        findings.append(Finding(
            'SYNC-04', '红线', w0.file, w0.line,
            f'Wait 无对应 Set（死等风险）: 方向={ev} flag={flag}',
            f'Wait@{w0.line} in {w0.func}，未见同组 SetFlag'))
    return findings


def _check_sync01_order(ev: str, flag: str, evs: List[SyncEvent], maps: LineMaps) -> List[Finding]:
    """SYNC-01: 仅同文件同函数内 Wait 先于 Set 才报（跨函数/跨文件顺序无意义）。"""
    findings: List[Finding] = []
    func_pairs: Dict[Tuple[str, str], List[SyncEvent]] = {}
    for e in evs:
        func_pairs.setdefault((e.file, e.func), []).append(e)
    for (fpath, func), fevs in func_pairs.items():
        fsets = [e for e in fevs if e.kind == 'set']
        fwaits = [e for e in fevs if e.kind == 'wait']
        if not fsets or not fwaits:
            continue
        min_set = min(s.line for s in fsets)
        min_wait = min(w.line for w in fwaits)
        if min_wait >= min_set:
            continue
        w0 = next(w for w in fwaits if w.line == min_wait)
        # 循环容忍：若 Wait→Set 之间跨 for/while，属双 buffer 跨迭代合法序，降级为存疑
        lc = maps.file_linecode.get(fpath, {})
        crosses_loop = any(LOOP_RE.search(lc.get(ln, ''))
                           for ln in range(min_wait + 1, min_set))
        if crosses_loop:
            findings.append(Finding(
                'SYNC-01', '高', w0.file, w0.line,
                f'Wait 先于 Set（跨循环，可能为双 buffer 合法序）: 方向={ev} flag={flag}',
                f'Wait@{min_wait} 早于 Set@{min_set}，同函数 {func}，'
                f'中间跨循环边界 → 常为合法跨迭代序；若非循环场景则为死等'))
        else:
            # ping-pong 降级：同函数同 flag 既有 Wait 又有 Set，
            # 说明该函数既消费(上一轮)又生产(本轮)该 flag → 跨调用迭代合法序
            # 典型：operator()/Iterate()/Run() 被外层循环调用，每次 Wait 上一轮 Set 本轮
            findings.append(Finding(
                'SYNC-01', '高', w0.file, w0.line,
                f'Wait 先于 Set（同函数消费+生产，可能为 ping-pong 合法序）: 方向={ev} flag={flag}',
                f'Wait@{min_wait} 早于 Set@{min_set}，同函数 {func} 既 Wait 又 Set；'
                f'若该函数被循环调用则为合法跨迭代序；若单次调用则为死等'))
    return findings


def _same_side_crosscore(set_sides: Set[str], wait_sides: Set[str]) -> bool:
    if not set_sides or not wait_sides:
        return False
    if set_sides != wait_sides:
        return False
    return len(set_sides) == 1


def _multi_file_crosscore_findings(flag: str, csets: List[SyncEvent], cwaits: List[SyncEvent],
                                   ref: SyncEvent) -> List[Finding]:
    findings: List[Finding] = []
    if len(csets) != len(cwaits):
        findings.append(Finding(
            'SYNC-04', '红线', ref.file, ref.line,
            f'CrossCore Set/Wait 个数不一致: flag={flag} Set={len(csets)} Wait={len(cwaits)}',
            f'Set: {[f"{os.path.basename(s.file)}:{s.line}" for s in csets]}; '
            f'Wait: {[f"{os.path.basename(w.file)}:{w.line}" for w in cwaits]}'))
    if not csets and cwaits:
        w0 = cwaits[0]
        findings.append(Finding(
            'SYNC-04', '红线', w0.file, w0.line,
            f'CrossCoreWaitFlag 无对应 Set（死等风险）: flag={flag}',
            '确认 cube 侧是否漏 SetFlag，或单核场景残留'))
    return findings


def _single_file_crosscore_findings(flag: str, csets: List[SyncEvent], cwaits: List[SyncEvent],
                                    ref: SyncEvent) -> List[Finding]:
    """单文件/部分文件：MIX kernel 单文件内 AIC 块和 AIV 块的 CrossCore 应配对。"""
    findings: List[Finding] = []
    if csets and cwaits:
        # 有 Set 有 Wait：检查 side 配对合理性（Set/Wait 全在同 side → 疑似配对错误）
        set_sides = set(s.side for s in csets)
        wait_sides = set(w.side for w in cwaits)
        if _same_side_crosscore(set_sides, wait_sides):
            side_val = next(iter(set_sides))
            findings.append(Finding(
                'SYNC-03', '高', ref.file, ref.line,
                f'CrossCore Set/Wait 同侧（{side_val}），疑似配对错误: flag={flag}',
                f'Set/Wait 都在 {side_val} 块内；CrossCore 应跨核（AIC↔AIV），请确认 Set/Wait 是否放错侧别块'))
    elif csets and not cwaits:
        findings.append(Finding(
            'SYNC-03', '信息', ref.file, ref.line,
            f'单文件内仅见 CrossCoreSetFlag 未见 Wait: flag={flag}',
            'CrossCore 配对需 cube+vec 两文件均在扫描范围；若为单核/合并核，应删除该 SetFlag'))
    elif cwaits and not csets:
        # 有 Wait 无 Set：若 side 已识别为 AIC，可能 Set 应在 AIV 块（同文件）但缺失
        w0 = cwaits[0]
        if w0.side:
            findings.append(Finding(
                'SYNC-04', '红线', w0.file, w0.line,
                f'CrossCoreWaitFlag({w0.side}) 无同文件对应 Set: flag={flag}',
                f'Wait 在 {w0.side} 块，Set 应在对侧块（{"AIV" if w0.side == "AIC" else "AIC"}）但缺失；'
                f'或 Set 在未扫描的另一核文件'))
        else:
            findings.append(Finding(
                'SYNC-03', '红线', ref.file, ref.line,
                f'单文件内仅见 CrossCoreWaitFlag 未见 Set（死等风险）: flag={flag}',
                '单核场景残留 WaitFlag → 永久阻塞；或 cube 文件未纳入扫描'))
    return findings


def _check_cross_core(events: List[SyncEvent], multi_file: bool) -> List[Finding]:
    """CrossCore 配对检查（SYNC-03/04/07）。"""
    findings: List[Finding] = []
    cc_groups: Dict[str, List[SyncEvent]] = {}
    for e in events:
        if e.kind in ('cset', 'cwait'):
            cc_groups.setdefault(e.flag_id, []).append(e)

    for flag, evs in cc_groups.items():
        csets = [e for e in evs if e.kind == 'cset']
        cwaits = [e for e in evs if e.kind == 'cwait']
        ref = evs[0]
        # SYNC-07: flagId 计数 > 15
        if len(csets) > 15:
            findings.append(Finding(
                'SYNC-07', '红线', ref.file, ref.line,
                f'flagId 复用超过 15 次: flag={flag} SetFlag 次数={len(csets)}',
                '同一 flagId 计数器最多设置 15 次，超过行为未定义'))
        if multi_file:
            findings.extend(_multi_file_crosscore_findings(flag, csets, cwaits, ref))
        else:
            findings.extend(_single_file_crosscore_findings(flag, csets, cwaits, ref))
    return findings


def check_pair(events: List[SyncEvent], multi_file: bool,
               lines_by_file: List[Tuple[str, List]] = None) -> List[Finding]:
    """SYNC-01 / SYNC-04 / SYNC-07（部分）：Set/Wait 配对、顺序、flagId 计数。"""
    findings: List[Finding] = []
    maps = _build_line_maps(lines_by_file)
    groups = _build_event_groups(events)
    ev_pure_balance = _compute_pure_balance(groups)

    for (ev, flag), evs in groups.items():
        # 别名平衡场景：同方向纯 Set 组与纯 Wait 组计数相等（如 e12/eventId12 简写）——
        # 单向组的"个数不一致"是别名误分组，跳过计数类检查（其余检查照常）
        stats = _build_group_stats(ev, flag, evs, ev_pure_balance, maps)
        findings.extend(_check_total_balance(stats))
        findings.extend(_check_sync01_order(ev, flag, evs, maps))

    findings.extend(_check_cross_core(events, multi_file))
    return findings
FUNC_EXIT_RE = re.compile(r'\b(?:return|break|goto)\b')
# 核身份/侧别守卫：这类 return 是"部分核/单侧执行"的框架标准模式（如
# if ASCEND_IS_AIC_SHOULD_RETURN / if (GetSubBlockIdx() > 0) / if (blockIdx_ >= usedCoreNum_)），
# 守卫的核不执行 SetFlag 是合法设计（对端有对称处理），不构成 SYNC-08 风险
CORE_GUARD_RE = re.compile(
    r'\b(ASCEND_IS_AIC|ASCEND_IS_AIV|GetSubBlockIdx|GetBlockIdx|GetBlockNum|GetCurBlockIdx|'
    r'blockIdx_|usedCoreNum_|needCoreNum_|coreNum|IsAIC|IsAIV)\b')


def check_early_return(events: List[SyncEvent], lines_info: List) -> List[Finding]:
    """SYNC-08: SetFlag 前存在函数退出路径（return/break/goto）可能绕过 Set。

    判定范围（2026-08-21 迭代三收敛误报）：
    - 只统计核内 SetFlag（排除 CrossCoreSetFlag：单侧发信号是合法设计）；
    - 只统计函数退出类控制流（return/break/goto），排除 continue：
      continue 仅跳过本迭代、信号配对由后续轮次维持，静态不可判（曾致 80% 误报）；
    - return 前的守卫若为核身份/侧别判定（CORE_GUARD_RE），视为框架标准"部分核执行"
      模式，豁免；
    - 保留兜底逻辑：退出行与目标 SetFlag 之间有其他 SetFlag 时，该退出已被保护，不报。
    """
    findings: List[Finding] = []
    # 按作用域组织退出行
    scope_exits: Dict[str, List[int]] = {}
    for (ln, _, code, sig) in lines_info:
        if FUNC_EXIT_RE.search(code):
            scope_exits.setdefault(sig, []).append(ln)

    # 按作用域组织所有 SetFlag 行（用于兜底判断）
    scope_setflags: Dict[str, List[int]] = {}
    for e in events:
        if e.kind == 'set':
            scope_setflags.setdefault(e.func, []).append(e.line)

    # 每行的前 3 行文本（用于识别退出行的守卫条件）
    line_to_code: Dict[int, str] = {ln: code for (ln, depth, code, sig) in lines_info}

    seen = set()
    for e in events:
        if e.kind != 'set':
            continue
        exits = scope_exits.get(e.func, [])
        all_sets = sorted(scope_setflags.get(e.func, []))
        earlier = [r for r in exits if r < e.line]
        if not earlier:
            continue
        # 排除被兜底的退出：退出行和目标 SetFlag 之间若有其他 SetFlag，则该退出已被保护
        risky = []
        for r in earlier:
            # 核身份/侧别守卫豁免
            guard_ctx = line_to_code.get(r - 1, '') + ' ' + line_to_code.get(r - 2, '') + ' ' + line_to_code.get(r, '')
            if CORE_GUARD_RE.search(guard_ctx):
                continue
            intervening = [s for s in all_sets if r < s < e.line]
            if not intervening:
                # 退出后、目标 SetFlag 前无其他 SetFlag → 退出真的跳过目标 SetFlag
                risky.append(r)
        if risky:
            key = (e.file, e.line)
            if key in seen:
                continue
            seen.add(key)
            findings.append(Finding(
                'SYNC-08', '红线', e.file, e.line,
                f'SetFlag 前存在提前退出（可能跳过 Set → 死锁）: {e.raw.split("(")[0]}',
                f'同函数 {e.func} 内未被兜底的 return/break: {risky}；确认所有路径都触发 SetFlag'))
    return findings


def check_matmul_mix(events: List[SyncEvent], all_lines: List[Tuple[int, int, str, str]],
                     files: List[str]) -> List[Finding]:
    """SYNC-07: CrossCore 与 Matmul 高阶 API 混用。"""
    findings: List[Finding] = []
    has_cc = any(e.kind in ('cset', 'cwait') for e in events)
    if not has_cc:
        return findings
    matmul_re = re.compile(r'\b(Matmul|MatmulSimple|MatmulImpl)\b')
    for (path, lns) in files:
        for ln, _, code, _ in lns:
            if matmul_re.search(code) and any(
                    e.kind in ('cset', 'cwait') and e.file == path for e in events):
                findings.append(Finding(
                    'SYNC-07', '红线', path, ln,
                    'CrossCoreSetFlag/WaitFlag 与 Matmul 高阶 API 混用（flagId 冲突风险）',
                    '高阶 API 内部已用 CrossCoreSetFlag，混用有 flagId 冲突；应删除手写 CrossCore'))
                break
    return findings


def check_pipebarrier(events: List[SyncEvent]) -> List[Finding]:
    """SYNC-09: PipeBarrier<PIPE_ALL> 过粗 / 连续 >3。"""
    findings: List[Finding] = []
    by_func: Dict[str, List[SyncEvent]] = {}
    for e in events:
        if e.kind == 'pipebarrier':
            by_func.setdefault(e.func, []).append(e)
            if e.event_type == 'PIPE_ALL':
                findings.append(Finding(
                    'SYNC-09', '性能', e.file, e.line,
                    'PipeBarrier<PIPE_ALL> 粒度过粗',
                    '若仅需 V 排序可用 PipeBarrier<PIPE_V>；确认无跨 PIPE 依赖后再收窄'))
    for func, evs in by_func.items():
        if len(evs) > 3:
            findings.append(Finding(
                'SYNC-09', '性能', evs[0].file, evs[0].line,
                f'同一函数连续 PipeBarrier 过多: {len(evs)} 个 in {func}',
                f'行: {[e.line for e in evs]}；分析是否可合并'))
    # 按文件统计 PipeBarrier<PIPE_V> 总数（跨函数但同一文件的过多也提示）
    by_file: Dict[str, List[SyncEvent]] = {}
    for e in events:
        if e.kind == 'pipebarrier' and e.event_type == 'PIPE_V':
            by_file.setdefault(e.file, []).append(e)
    for _, evs in by_file.items():
        if len(evs) > 5:
            findings.append(Finding(
                'SYNC-09', '性能', evs[0].file, evs[0].line,
                f'同一文件 PipeBarrier<PIPE_V> 过多: {len(evs)} 个',
                f'行: {[e.line for e in evs]}；分析是否可合并'))
    return findings


def _var_unsigned_types(lns) -> Dict[str, bool]:
    """收集文件内无符号/有符号变量声明类型表。"""
    var_types: Dict[str, bool] = {}
    for _ln, _depth, code, _sig in lns:
        for vm in UINT_VAR_DECL_RE.finditer(code):
            var_types[vm.group(1)] = bool(UINT_TYPE_RE.search(vm.group(0)))
    return var_types


def _underflow_line_findings(path, ln, code, var_types) -> List[Finding]:
    findings: List[Finding] = []
    for um in UNDERFLOW_RE.finditer(code):
        var = um.group(1)
        if var_types.get(var, False):
            findings.append(Finding(
                'SYNC-10', '高', path, ln,
                f'无符号下溢风险: ({var} - {um.group(2)}) % N，{var} 为无符号类型',
                f'当 {var}=0 时下溢为最大值；首次迭代应单独处理'))
    return findings


def check_loop_underflow(lines_by_file) -> List[Finding]:
    """SYNC-10: (loop - k) % N 形式无符号下溢风险。"""
    findings: List[Finding] = []
    for path, lns in lines_by_file:
        var_types = _var_unsigned_types(lns)
        for ln, _depth, code, _sig in lns:
            findings.extend(_underflow_line_findings(path, ln, code, var_types))
    return findings


def _add_reg_names(match, regtensor_vars: Set[str]) -> None:
    """把逗号分隔的寄存器变量声明名加入集合（仅字母开头名）。"""
    for raw_name in match.group(1).split(','):
        name = raw_name.strip()
        if name and name[0].isalpha():
            regtensor_vars.add(name)


def _collect_regtensor_vars(lns) -> Set[str]:
    """扫描 RegTensor/UnalignReg/MaskReg 声明的寄存器变量（MicroAPI，不跨流水）。"""
    regtensor_vars: Set[str] = set()
    for (_ln, _depth, code, _sig) in lns:
        for m in REGTENSOR_DECL_RE.finditer(code):
            _add_reg_names(m, regtensor_vars)
        for m in MICRO_REG_RE.finditer(code):
            _add_reg_names(m, regtensor_vars)
    return regtensor_vars


def _collect_datacopy_defs(line: int, code: str, regtensor_vars: Set[str],
                           var_defs: Dict[str, List[Tuple[int, str]]],
                           var_uses: Dict[str, List[Tuple[int, str]]]) -> None:
    m = DC_WRITE_RE.search(code)
    if not m:
        return
    dst = m.group(1)
    if dst in regtensor_vars:
        return
    if re.search(r'(Gm|GM|Global|output|yGm)', dst):
        var_uses.setdefault(dst, []).append((line, 'MTE3_write'))
    else:
        var_defs.setdefault(dst, []).append((line, 'MTE2_write'))


def _collect_compute_uses(line: int, code: str, regtensor_vars: Set[str],
                          var_uses: Dict[str, List[Tuple[int, str]]]) -> None:
    for m in COMPUTE_WRITE_RE.finditer(code):
        dst = m.group(2) if m.lastindex >= 2 else m.group(1)
        if dst in regtensor_vars:
            continue
        var_uses.setdefault(dst, []).append((line, 'V_write'))
    m2 = re.search(
        r'\b(Adds?|Muls?|Cast|ReduceMax|ReduceMin|ReduceSum|'
        r'Compare|Abs|Duplicate|Relu)\s*(?:<[^>]*>)?\s*\(\s*\w+\s*,\s*(\w+)', code)
    if m2:
        src = m2.group(2)
        if src not in regtensor_vars:
            var_uses.setdefault(src, []).append((line, 'V_read'))


def _collect_line_defs_uses(line: int, code: str, regtensor_vars: Set[str],
                            var_defs: Dict[str, List[Tuple[int, str]]],
                            var_uses: Dict[str, List[Tuple[int, str]]]) -> None:
    """单行的 def-use 收集（TENSOR_DECL / DataCopy / Compute / GetValue）。"""
    if not code.strip() or code.strip().startswith('//'):
        return
    is_microapi = bool(MICROAPI_LINE_RE.search(code))
    for m in TENSOR_DECL_RE.finditer(code):
        var_name = m.group(1)
        if var_name not in ('true', 'false', 'nullptr', 'this'):
            var_defs.setdefault(var_name, []).append((line, 'decl'))
    if not is_microapi:
        _collect_datacopy_defs(line, code, regtensor_vars, var_defs, var_uses)
    if not is_microapi:
        _collect_compute_uses(line, code, regtensor_vars, var_uses)
    m3 = GETVALUE_RE.search(code)
    if m3 and not is_microapi:
        var_name = m3.group(1)
        if var_name not in regtensor_vars:
            var_uses.setdefault(var_name, []).append((line, 'S_read'))


def _has_sync_between(lns, d_ln: int, u_ln: int) -> bool:
    for (check_ln, _depth, code, _sig) in lns:
        if d_ln < check_ln < u_ln and SYNC_RE.search(code):
            return True
    return False


def _check_var_sync(path: str, var: str, defs: List[Tuple[int, str]],
                    uses: List[Tuple[int, str]], lns) -> List[Finding]:
    """单个变量的 def-use 链跨流水缺同步检查。"""
    findings: List[Finding] = []
    if not uses:
        return findings
    for d_ln, d_pipe in defs:
        if d_pipe == 'decl':
            continue
        for u_ln, u_pipe in uses:
            if u_ln <= d_ln:
                continue
            if d_pipe == u_pipe:
                continue
            if _has_sync_between(lns, d_ln, u_ln):
                continue
            if u_ln - d_ln > 15:
                continue
            pipe_map = {'MTE2_write': 'MTE2', 'MTE3_write': 'MTE3',
                        'V_write': 'V', 'V_read': 'V', 'S_read': 'S'}
            src_pipe = pipe_map.get(d_pipe, d_pipe)
            dst_pipe = pipe_map.get(u_pipe, u_pipe)
            findings.append(Finding(
                'SYNC-02', '红线', path, u_ln,
                f'变量 {var} 跨流水缺同步: {src_pipe}({d_pipe})@L{d_ln} → {dst_pipe}({u_pipe})@L{u_ln}',
                f'{var} 在 L{d_ln} 被 {src_pipe} 写入，在 L{u_ln} 被 {dst_pipe} 读取，'
                f'中间无 EnQue/DeQue/SetFlag/WaitFlag/PipeBarrier 同步'))
    return findings


def _dedupe_findings(findings: List[Finding]) -> List[Finding]:
    seen = set()
    unique = []
    for f in findings:
        key = (f.file, f.line, f.message[:50])
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def check_missing_sync(lines_by_file) -> List[Finding]:
    """SYNC-02: 数据流追踪——检测 DataCopy(写)→Compute(读) 跨流水缺同步。
    简化版变量追踪：识别 LocalTensor 变量名，追踪其 def-use 链。
    排除 MicroAPI 寄存器操作（同流水，不需要跨流水同步）。
    """
    findings: List[Finding] = []
    for path, lns in lines_by_file:
        regtensor_vars = _collect_regtensor_vars(lns)
        var_defs: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        var_uses: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

        for (ln, _depth, code, _sig) in lns:
            _collect_line_defs_uses(ln, code, regtensor_vars, var_defs, var_uses)

        for var, defs in var_defs.items():
            findings.extend(_check_var_sync(path, var, defs, var_uses.get(var, []), lns))

    return _dedupe_findings(findings)


def _merge_multiline(lns, trigger, open_char: str, close_char: str):
    """按触发器与括号平衡跨行合并语句（用于 SetFlag/DataCopy/buffer 索引跨行场景）。"""
    merged = []
    i = 0
    while i < len(lns):
        ln, depth, code, sig = lns[i]
        if not trigger(code):
            merged.append(lns[i])
            i += 1
            continue
        if code.count(open_char) <= code.count(close_char):
            merged.append(lns[i])
            i += 1
            continue
        text = code
        j = i + 1
        while j < len(lns) and text.count(open_char) > text.count(close_char):
            text += ' ' + lns[j][2]
            j += 1
        merged.append((ln, depth, text, sig))
        i = j
    return merged


def _buffer_line(code: str) -> bool:
    return 'buffer' in code.lower() or 'Buf' in code


def _first_var_name(expr: str) -> str:
    for v in VAR_NAME_RE.findall(expr):
        if v not in ('true', 'false', 'nullptr') and len(v) > 1:
            return v
    return ''


def _full_bracket_expr(code: str, buf_name: str, m_start: int) -> str:
    """提取嵌套 buffer[ 的完整索引表达式（用括号匹配避免被内层 [ 截断）。"""
    buf_pos = code.find(buf_name, m_start)
    bracket_pos = code.find('[', buf_pos)
    if bracket_pos < 0:
        return ''
    depth_b = 0
    for i in range(bracket_pos, len(code)):
        if code[i] == '[':
            depth_b += 1
        elif code[i] == ']':
            depth_b -= 1
            if depth_b == 0:
                return code[bracket_pos + 1:i]
    return ''


def _expand_bracket_expr(code: str, buf_name: str, m_start: int, idx_expr: str) -> str:
    """嵌套 buffer[ 的索引表达式若被内层 [ 截断，用括号匹配补全。"""
    if '[' not in idx_expr or ']' in idx_expr:
        return idx_expr
    full = _full_bracket_expr(code, buf_name, m_start)
    return full if full else idx_expr




def _innermost_index(expr: str) -> str:
    inner = expr
    while True:
        inner_m = re.search(r'\[([^]]+)\]', inner)
        if not inner_m:
            return inner
        inner = inner_m.group(1)


def _extract_flag_indices(all_lines) -> List[Tuple[int, str, str]]:
    """收集所有 SetFlag/WaitFlag 的索引变量：(行号, 方向, 索引变量)。"""
    flag_indices: List[Tuple[int, str, str]] = []
    for ln, code in all_lines:
        if not SET_WAIT_CALL_RE.search(code):
            continue
        paren_pos = code.find('(', code.find('Flag'))
        if paren_pos == -1:
            continue
        end = _matching_paren(code, paren_pos)
        flag_arg = code[paren_pos + 1:end].strip()
        index_var = _flag_index_var(flag_arg)
        if index_var:
            dir_m = re.search(r'HardEvent::(\w+)', code)
            direction = dir_m.group(1) if dir_m else '?'
            flag_indices.append((ln, direction, index_var))
    return flag_indices


def _matching_paren(code: str, open_paren_pos: int) -> int:
    depth_p = 0
    for i in range(open_paren_pos, len(code)):
        if code[i] == '(':
            depth_p += 1
        elif code[i] == ')':
            depth_p -= 1
            if depth_p == 0:
                return i
    return len(code)


def _flag_index_var(flag_arg: str) -> str:
    """从 flag 参数提取索引变量：
    1. eventId[idx & (N-1)] — 数组索引形式
    2. FLAG_BASE + (var) 或 FLAG_BASE + var — 加法偏移形式
    """
    bracket_m = re.search(r'\[([^]]+)\]', flag_arg)
    if bracket_m:
        idx_expr = bracket_m.group(1)
        if not CONSTANT_RE.match(idx_expr.strip()):
            index_var = _first_var_name(idx_expr)
            if index_var:
                return index_var
    plus_m = re.search(r'\+\s*\(\s*(\w+)\s*\)', flag_arg)
    if plus_m:
        return plus_m.group(1)
    plus_m2 = re.search(r'\+\s*([a-z]\w*)', flag_arg)
    if plus_m2:
        return plus_m2.group(1)
    return ''


def _extract_buffer_indices(all_lines) -> List[Tuple[int, str, str]]:
    """收集所有数据操作的 buffer 索引变量：(行号, buffer 名, 索引变量)。"""
    buffer_indices: List[Tuple[int, str, str]] = []
    for ln, code in all_lines:
        for m in BUFFER_INDEX_RE.finditer(code):
            buf_name = m.group(1)
            idx_expr = m.group(2)
            if CONSTANT_RE.match(idx_expr.strip()):
                continue
            idx_expr = _expand_bracket_expr(code, buf_name, m.start(), idx_expr)
            index_var = _first_var_name(_innermost_index(idx_expr))
            if not index_var:
                index_var = _first_var_name(idx_expr)
            if index_var:
                buffer_indices.append((ln, buf_name, index_var))
    return buffer_indices


def _compare_flag_buffer_indices(path: str, flag_indices: List[Tuple[int, str, str]],
                                 buffer_indices: List[Tuple[int, str, str]]) -> List[Finding]:
    """对比：每个 flag 的索引变量 vs 附近（±15行）buffer 的索引变量。"""
    findings: List[Finding] = []
    seen_pairs = set()
    for f_ln, f_dir, f_var in flag_indices:
        for b_ln, b_name, b_var in buffer_indices:
            if abs(b_ln - f_ln) > 15:
                continue
            if f_var == b_var:
                continue
            key = (path, f_ln, b_ln, f_var, b_var)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            findings.append(Finding(
                'SYNC-14', '红线', path, b_ln,
                f'同步索引变量({f_var})与 buffer 索引变量({b_var})不一致',
                f'SetFlag/WaitFlag<{f_dir}>@L{f_ln} 用索引变量 {f_var}，'
                f'但 L{b_ln} 的 {b_name}[...] 用索引变量 {b_var}。'
                f'两个循环索引不同可能导致同步保护的 buffer 与实际操作的 buffer 对不上。'
                f'请确认 {f_var} 和 {b_var} 是否等价（不同循环计数器通常不等价）'))
    return findings


@dataclass
class WindowContext:
    """同步窗口索引检查上下文（findings 收集 + 去重）。"""

    path: str
    seen_pairs: set


def _emit_window_findings(flag_entry, var1: str, locs1, ctx: WindowContext) -> List[Finding]:
    """单个窗口内索引根变量的 SYNC-14 候选（去重后输出）。"""
    f_ln, f_dir, f_var = flag_entry
    findings: List[Finding] = []
    for ln1, _ in locs1:
        key = (ctx.path, f_ln, ln1, f'window:{f_var}', f'window:{var1}')
        if key in ctx.seen_pairs:
            continue
        ctx.seen_pairs.add(key)
        findings.append(Finding(
            'SYNC-14', '红线', ctx.path, ln1,
            f'同步窗口内 buffer 索引根变量不一致: {var1} vs flag({f_var})',
            f'SetFlag/WaitFlag<{f_dir}>@L{f_ln} 的 ±15 行窗口内存在多个 buffer 访问，'
            f'索引根变量不一致。'
            f'即使部分 buffer 无独立 flag（隐式复用邻居同步），'
            f'其索引变量也应与所复用同步的 flag 索引变量同源。'
            f'请确认窗口内各索引变量是否等价'))
    return findings


def _window_index_mismatch(path: str, flag_indices: List[Tuple[int, str, str]],
                           buffer_indices: List[Tuple[int, str, str]]) -> List[Finding]:
    """SYNC-14 扩展：同步时间窗口内，不同 buffer 访问的索引根变量不一致。"""
    findings: List[Finding] = []
    ctx = WindowContext(path, set())
    for f_ln, f_dir, f_var in flag_indices:
        window_buffers = [(b_ln, b_name, b_var) for b_ln, b_name, b_var in buffer_indices
                          if abs(b_ln - f_ln) <= 15]
        if len(window_buffers) < 2:
            continue
        root_vars_in_window = defaultdict(list)
        for b_ln, b_name, b_var in window_buffers:
            root_vars_in_window[b_var].append((b_ln, b_name))
        if len(root_vars_in_window) < 2:
            continue
        for var1, locs1 in root_vars_in_window.items():
            if var1 != f_var:
                findings.extend(_emit_window_findings((f_ln, f_dir, f_var), var1, locs1, ctx))
    return findings


def check_sync_buffer_index(lines_by_file) -> List[Finding]:
    """SYNC-14: 同步信号索引变量与 buffer 索引变量一致性。
    在同一函数内，SetFlag/WaitFlag 的 flag_id 索引变量必须与其保护的数据操作
    （DataCopy/Compute 等）的 buffer 索引变量一致。
    纯字符串级变量名对比，不做语义推理。
    """
    findings: List[Finding] = []
    for path, lns in lines_by_file:
        merged = _merge_multiline(lns, SYNC_CALL_RE.search, '(', ')')
        merged = _merge_multiline(merged, DATA_CALL_RE.search, '(', ')')
        merged = _merge_multiline(merged, _buffer_line, '[', ']')
        all_lines = [(ln, code) for (ln, _depth, code, _sig) in merged
                     if code.strip() and not code.strip().startswith('//')]
        flag_indices = _extract_flag_indices(all_lines)
        buffer_indices = _extract_buffer_indices(all_lines)
        findings.extend(_compare_flag_buffer_indices(path, flag_indices, buffer_indices))
        findings.extend(_window_index_mismatch(path, flag_indices, buffer_indices))
    return findings


def _merge_assign_lines(lns):
    """跨行合并含 field 赋值但无分号或括号不平衡的行。"""
    merged = []
    i = 0
    while i < len(lns):
        ln, depth, code, sig = lns[i]
        text = code
        needs_semicolon = FIELD_ASSIGN_RE.search(text) and ';' not in text
        needs_balance = text.count('(') > text.count(')') or text.count('[') > text.count(']')
        if not (needs_semicolon or needs_balance):
            merged.append(lns[i])
            i += 1
            continue
        j = i + 1
        while j < len(lns):
            if ';' in text and text.count('(') <= text.count(')') and text.count('[') <= text.count(']'):
                break
            text += ' ' + lns[j][2]
            j += 1
        merged.append((ln, depth, text, sig))
        i = j
    return merged


def _first_index_var(expr: str) -> str:
    for v in VAR_NAME_RE.findall(expr):
        if v not in SKIP_VARS and len(v) > 1:
            return v
    return ''


def _resolve_alias(var: str, aliases: Dict[str, str]) -> str:
    seen = set()
    cur = var
    while cur in aliases and cur not in seen:
        seen.add(cur)
        cur = aliases[cur]
    return cur


def _is_output_access(field_name: str, buf_name: str) -> bool:
    return bool(OUTPUT_FIELD_RE.search(field_name) or OUTPUT_BUFFER_RE.search(buf_name))


def _collect_aliases(func_lines) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for _ln, code in func_lines:
        for m in VAR_DECL_ASSIGN_RE.finditer(code):
            dst = m.group(1)
            src_var = _first_index_var(m.group(2))
            if src_var and src_var != dst:
                aliases[dst] = _resolve_alias(src_var, aliases)
    return aliases


def _collect_output_accesses(func_lines, aliases: Dict[str, str]):
    """按输出参数对象收集 producer 输出 buffer 写址记录。"""
    output_accesses: Dict[str, List[Tuple[int, str, str, str, str, str]]] = defaultdict(list)
    for ln, code in func_lines:
        field_m = FIELD_ASSIGN_RE.search(code)
        if not field_m:
            continue
        obj, field_name = field_m.group(1), field_m.group(2)
        for buf_m in BUFFER_INDEX_RE.finditer(code):
            buf_name = buf_m.group(1)
            if not _is_output_access(field_name, buf_name):
                continue
            idx_var = _first_index_var(buf_m.group(2))
            if not idx_var:
                continue
            root_var = _resolve_alias(idx_var, aliases)
            output_accesses[obj].append((ln, field_name, buf_name, idx_var, root_var, code.strip()))
    return output_accesses


def _output_bundle_findings(path: str, obj: str, accesses) -> List[Finding]:
    """同一输出参数对象内 producer output buffer 索引根不同源 → SYNC-14 候选。"""
    findings: List[Finding] = []
    if len(accesses) < 2:
        return findings
    roots = {a[4] for a in accesses}
    if len(roots) < 2:
        return findings
    compute_roots = [r for r in roots if 'ComputeLoopIdx' in r or r == 'ubComputeLoopIdx_']
    mte2_roots = [r for r in roots if 'Mte2' in r or 'mte2' in r]
    if not compute_roots or not mte2_roots:
        return findings
    compute_root = compute_roots[0]
    seen = set()
    for ln, field_name, buf_name, idx_var, root_var, raw in accesses:
        if root_var == compute_root:
            continue
        key = (path, ln, obj, field_name, buf_name, root_var, compute_root)
        if key in seen:
            continue
        seen.add(key)
        ref = next((a for a in accesses if a[4] == compute_root), None)
        ref_detail = ''
        if ref:
            ref_detail = f'；同一参数对象 {obj}.{ref[1]}@L{ref[0]} 使用输出索引变量 {compute_root}'
        findings.append(Finding(
            'SYNC-14', '红线', path, ln,
            f'输出 buffer 写址索引变量({root_var})与输出同步索引变量({compute_root})不一致',
            f'{obj}.{field_name}@L{ln} 写入 {buf_name}[...]，索引 {idx_var} 解析为 {root_var}'
            f'{ref_detail}。同一 VF/计算参数对象的输出地址必须使用同一 output buffer id；'
            f'否则后续 SetFlag/WaitFlag 按 {compute_root} 保护的 buffer 与实际写入 buffer 不一致。'
            f'原始代码: {raw}'))
    return findings


def check_output_buffer_producer_index(lines_by_file) -> List[Finding]:
    """SYNC-14: producer 侧输出 buffer 写址索引一致性。

    典型问题：
      params.weightHighBitPhyAddr = ubHighBitTotalBuffer_[ubComputeLoopIdx_ & ...].GetPhyAddr();
      params.biasOutUbAddr = ubBiasOutTotalBuffer_[ubMte2BufferIdx * ...].GetPhyAddr();

    后续 MTE3_V SetFlag 用 ubComputeLoopIdx_ 保护输出 buffer 复用，但 biasOut 由
    ubMte2LoopIdx_ 派生索引写入，导致 producer/consumer/sync 三者不一致。
    """
    findings: List[Finding] = []
    for path, lns in lines_by_file:
        merged_lns = _merge_assign_lines(lns)
        by_func: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for ln, _depth, code, sig in merged_lns:
            if code.strip() and not code.strip().startswith('//'):
                by_func[sig].append((ln, code))
        for _sig, func_lines in by_func.items():
            aliases = _collect_aliases(func_lines)
            output_accesses = _collect_output_accesses(func_lines, aliases)
            for obj, accesses in output_accesses.items():
                findings.extend(_output_bundle_findings(path, obj, accesses))
    return findings


def _collect_def_start_lines(lns) -> Set[int]:
    """函数定义行集合（sig 首次出现的前 2 行 = 函数签名区）：定义行不是调用。"""
    def_start_lines: Set[int] = set()
    prev_sig = None
    for ln, _depth, _code, sig in lns:
        if sig != '<global>' and sig != prev_sig:
            for d in range(1, 3):
                def_start_lines.add(ln - d)
        prev_sig = sig
    return def_start_lines


def _collect_def_sync_flags(lns) -> Dict[str, bool]:
    """函数体是否含同步 API（用于自定义 CopyIn/CopyOut 调用处的跨函数同步判定）。"""
    def_sync_flags: Dict[str, bool] = {}
    for _ln, _depth, code, sig in lns:
        if sig == '<global>':
            continue
        if sig not in def_sync_flags:
            def_sync_flags[sig] = bool(SYNC_ANY_RE.search(code))
        elif not def_sync_flags[sig] and SYNC_ANY_RE.search(code):
            def_sync_flags[sig] = True
    return def_sync_flags


def _def_has_sync(name: str, def_sync_flags: Dict[str, bool]) -> bool:
    """查询函数体内是否含同步 API。成员函数签名（Class<T>::Method）经 scan_file
    解析后 sig 为完整长串，需按方法短名后缀匹配（如 '...::CopyOut'）。
    """
    if def_sync_flags.get(name):
        return True
    return any(sig.endswith('::' + name) and flag for sig, flag in def_sync_flags.items())


def _datacopy_ops(code: str) -> List[Tuple[str, str]]:
    if not DATACOPY_RE.search(code) or MICROAPI_RE.search(code):
        return []
    dc_match = re.search(
        r'DataCopy(?:Pad)?(?:Ext)?(?:<[^>]*>)?\s*\(\s*'
        r'(\w+)(?:\[[^\]]*\])?\s*,\s*(\w+)(?:\[[^\]]*\])?', code)
    if not dc_match:
        # 首个参数为 cast/复杂表达式（如 DataCopy((__local_mem__*)cah, aReg, pMask)）
        # 无法可靠判定方向时按同流水处理，避免误当成 GM→UB load
        return [('ub_copy', '')]
    dst_name, src_name = dc_match.group(1), dc_match.group(2)
    if re.search(r'Reg$', dst_name):
        return [('reg_copy', '')]   # 寄存器级搬运是同步语义
    if re.search(r'(Gm|GM|gm|Global|output)', dst_name):
        return [('store', '')]      # UB→GM (MTE3)
    if re.search(r'(Gm|GM|gm|Global|input)', src_name):
        return [('load', '')]       # GM→UB (MTE2)
    return [('ub_copy', '')]        # UB→UB 同流水，不需跨流水同步


def _copyout_ops(code: str, def_sync_flags: Dict[str, bool]) -> List[Tuple[str, str]]:
    if not COPYOUT_RE.search(code):
        return []
    fn_m = re.search(r'\b(CopyOut\w*)\s*\(', code)
    ops = []
    if fn_m and _def_has_sync(fn_m.group(1), def_sync_flags):
        ops.append(('sync', 'other'))
    ops.append(('store', ''))
    return ops


def _copyin_ops(code: str, def_sync_flags: Dict[str, bool]) -> List[Tuple[str, str]]:
    if not COPYIN_RE.search(code):
        return []
    fn_m = re.search(r'\b(CopyIn\w*)\s*\(', code)
    ops = []
    if fn_m and _def_has_sync(fn_m.group(1), def_sync_flags):
        ops.append(('sync', 'other'))
    ops.append(('load', ''))
    return ops


def _te_copy_ops(code: str) -> List[Tuple[str, str]]:
    if not TE_COPY_RE.search(code):
        return []
    if TE_COPY_L12L0_RE.search(code):
        return [('te_copy_l12l0', '')]
    if TE_COPY_L0C2GM_RE.search(code):
        return [('te_copy_l0c2gm', '')]
    return [('te_copy', '')]


def _reg_ops(code: str) -> List[Tuple[str, str]]:
    """Reg:: Store/Load/DataCopy — V 内数据流（DSL vstore/vload 等价物）。"""
    ops = []
    if REG_STORE_RE.search(code):
        ops.append(('reg_store', ''))
    if REG_LOAD_RE.search(code):
        ops.append(('reg_load', ''))
    if REG_DATACOPY_RE.search(code):
        dc_reg = re.search(r'Reg::DataCopy\s*(?:<[^>]*>)?\s*\(\s*(\w+)', code)
        if dc_reg:
            first_arg = dc_reg.group(1)
            if re.search(r'(Addr|addr|tmp|Exp|exp|buf)', first_arg):
                ops.append(('reg_store', ''))   # Reg::DataCopy(mem, reg) = store
            else:
                ops.append(('reg_load', ''))    # Reg::DataCopy(reg, mem) = load
    return ops


def _sync_ops(code: str) -> List[Tuple[str, str]]:
    """同步类操作识别（sync 先于 store/load append：同行的"搬移+同步"需先被缺同步判定看到）。"""
    ops = []
    for m in PIPEBAR_RE.finditer(code):
        ops.append(('sync', 'PIPE_' + normalize_event(m.group(1)).replace('PIPE_', '')))
    for m in SET_WAIT_RE.finditer(code):
        ops.append(('sync', normalize_event(m.group(2))))
    if ENQUE_RE.search(code):
        ops.append(('sync', 'enque'))
    if DEQUE_RE.search(code):
        ops.append(('sync', 'deque'))
    for m in HISYNC_RE.finditer(code):
        ops.append(('sync', m.group(1).replace('To', '_')))
    if SYNCFUNC_RE.search(code) or CROSSCORE_RE.search(code) or SCALARBAR_RE.search(code):
        ops.append(('sync', 'other'))
    if REG_LOCALMEMBAR_RE.search(code):
        ops.append(('sync', 'V_V'))
    return ops


def _classify_line_ops(code: str, def_sync_flags: Dict[str, bool]) -> List[Tuple[str, str]]:
    """单行的操作序列分类（(op, name) 列表）。"""
    ops: List[Tuple[str, str]] = []
    if ALLOCTENSOR_RE.search(code):
        ops.append(('alloc', ''))
    ops.extend(_datacopy_ops(code))
    ops.extend(_copyout_ops(code, def_sync_flags))
    ops.extend(_copyin_ops(code, def_sync_flags))
    ops.extend(_te_copy_ops(code))
    if MMAD_RE.search(code):
        ops.append(('mmad', ''))
    if COMPUTE_OPS.search(code) and not MICROAPI_RE.search(code):
        m = COMPUTE_OPS.search(code)
        ops.append(('compute', m.group(1)))
    if SCALAR_OPS.search(code):
        ops.append(('scalar', ''))
    ops.extend(_sync_ops(code))
    ops.extend(_reg_ops(code))
    return ops


def _code_var_rel(ln_code: Dict[int, str], ln_i: int, ln_j: int) -> bool:
    """变量关联检查（迭代八）：两行存在公共标识符（tensor 变量）才判有数据依赖。"""
    ids_i = set(re.findall(r'\b[a-zA-Z_]\w*\b', ln_code.get(ln_i, ''))) - VAR_STOP
    ids_j = set(re.findall(r'\b[a-zA-Z_]\w*\b', ln_code.get(ln_j, ''))) - VAR_STOP
    return bool(ids_i & ids_j)


def _check_load_compute(seq, i: int, path: str, ln_code, findings: List[Finding]) -> None:
    """SYNC-02 模式A: load -> compute 之间无 sync。"""
    if seq[i][1] != 'load':
        return
    ln_i = seq[i][0]
    for j in range(i + 1, min(i + 6, len(seq))):
        ln_j, op_j, name_j = seq[j]
        if op_j == 'sync':
            break
        if (op_j == 'compute' and ln_j - ln_i <= MAX_OP_DISTANCE
                and _code_var_rel(ln_code, ln_i, ln_j)):
            findings.append(Finding(
                'SYNC-02', '红线', path, ln_j,
                f'疑似 MTE2 搬入后未同步即计算: {name_j}@{ln_j} (搬入@{ln_i})',
                '候选：DataCopy 后、Vector 计算前需 EnQue/DeQue 或 SetFlag/WaitFlag<MTE2_V>'))
            break


def _check_compute_store(seq, i: int, path: str, ln_code, findings: List[Finding]) -> None:
    """SYNC-02 模式B: compute -> store 之间无 sync（计算后搬出 GM）。"""
    if seq[i][1] != 'compute':
        return
    ln_i = seq[i][0]
    for j in range(i + 1, min(i + 4, len(seq))):
        ln_j, op_j, name_j = seq[j]
        if op_j == 'sync':
            break
        if (op_j == 'store' and ln_j - ln_i <= MAX_OP_DISTANCE
                and _code_var_rel(ln_code, ln_i, ln_j)):
            findings.append(Finding(
                'SYNC-02', '红线', path, ln_j,
                f'疑似计算后未同步即搬出 GM: 搬出@{ln_j} (计算@{ln_i})',
                '候选：Vector 计算后、DataCopy 出 GM 前需 V→MTE3 同步'))
            break


def _check_compute_scalar(seq, i: int, path: str, ln_code, findings: List[Finding]) -> None:
    """SYNC-02 模式C: compute(V) -> scalar(S) 之间无 sync（如 ReduceSum→GetValue）。"""
    if seq[i][1] != 'compute':
        return
    ln_i = seq[i][0]
    for j in range(i + 1, min(i + 4, len(seq))):
        ln_j, op_j, name_j = seq[j]
        if op_j == 'sync':
            break
        if (op_j == 'scalar' and ln_j - ln_i <= MAX_OP_DISTANCE
                and _code_var_rel(ln_code, ln_i, ln_j)):
            findings.append(Finding(
                'SYNC-02', '红线', path, ln_j,
                f'疑似 V 计算后未同步即 Scalar 读: Scalar@{ln_j} (计算@{ln_i})',
                '候选：Vector 计算（如 ReduceSum）后、GetValue 前需 V→S 同步'
                '（PipeBarrier<PIPE_V> 或 SetFlag/WaitFlag<V_S>）'))
            break


def _check_pipe_v_store(seq, i: int, path: str, _ln_code, findings: List[Finding]) -> None:
    """SYNC-06: PipeBarrier<PIPE_V> 后紧跟 store（跨 PIPE 误用，UB→GM 才跨 PIPE）。"""
    ln_i, op_i, name_i = seq[i]
    if not (op_i == 'sync' and name_i == 'PIPE_V'):
        return
    for j in range(i + 1, min(i + 3, len(seq))):
        ln_j, op_j, _name_j = seq[j]
        if op_j == 'store':
            findings.append(Finding(
                'SYNC-06', '红线', path, ln_j,
                f'PipeBarrier<PIPE_V> 后紧跟 DataCopy 出 GM（跨 PIPE 误用候选）@{ln_j}',
                '候选：V→MTE3 跨 PIPE 不能用 PipeBarrier<PIPE_V>，需 SetFlag/WaitFlag<V_MTE3>'))
            break


def _check_l1l0_mmad(seq, i: int, path: str, _ln_code, findings: List[Finding]) -> None:
    """SYNC-02 Cube 侧: L1→L0 Copy 后、Mmad 前需 MTE1→M 同步。"""
    if seq[i][1] != 'te_copy_l12l0':
        return
    ln_i = seq[i][0]
    for j in range(i + 1, min(i + 5, len(seq))):
        ln_j, op_j, _name_j = seq[j]
        if op_j == 'sync':
            break
        if op_j == 'mmad':
            findings.append(Finding(
                'SYNC-02', '红线', path, ln_j,
                f'疑似 L1→L0 搬运后未同步即 MMAD: MMAD@{ln_j} (L1→L0@{ln_i})',
                '候选：L1→L0 Copy 后、MMAD 前需 SetFlag/WaitFlag<MTE1_M>'))
            break


def _check_mmad_l0c2gm(seq, i: int, path: str, _ln_code, findings: List[Finding]) -> None:
    """SYNC-02 Cube 侧: Mmad 后、L0C→GM Copy 前需 M→FIX 同步。"""
    if seq[i][1] != 'mmad':
        return
    ln_i = seq[i][0]
    for j in range(i + 1, min(i + 5, len(seq))):
        ln_j, op_j, _name_j = seq[j]
        if op_j == 'sync':
            break
        if op_j == 'te_copy_l0c2gm':
            findings.append(Finding(
                'SYNC-02', '红线', path, ln_j,
                f'疑似 MMAD 后未同步即 L0C→GM 搬出: L0C→GM@{ln_j} (MMAD@{ln_i})',
                '候选：MMAD 后、L0C→GM Copy 前需 SetFlag/WaitFlag<M_FIX>'))
            break


def _v_mte2_after(seq, i: int) -> bool:
    for j in range(i + 1, min(i + 4, len(seq))):
        _ln_j, op_j, name_j = seq[j]
        if op_j == 'sync' and name_j == 'V_MTE2':
            return True
        if op_j == 'sync' and name_j != 'V_MTE2':
            break
    return False


def _has_barrier_or_copyout_before(seq, i: int) -> bool:
    """前方有搬出操作但无 PipeBarrier → False（报）。循环前初始化无搬出 → True（不报）。"""
    if i <= 0:
        return True
    has_barrier_before = False
    has_copyout_before = False
    for k in range(max(0, i - 5), i):
        _pk_ln, pk_op, pk_name = seq[k]
        if pk_op == 'sync' and pk_name == 'PIPE_ALL':
            has_barrier_before = True
        if pk_op in ('store', 'load') or pk_name in ('store', 'load'):
            has_copyout_before = True
    return has_barrier_before or not has_copyout_before


def _check_mte3v_mte2(seq, i: int, path: str, _ln_code, findings: List[Finding]) -> None:
    """SYNC-02: SetFlag<MTE3_V> + SetFlag<V_MTE2> 释放 buffer 复用前需有 PipeBarrier。"""
    ln_i, op_i, name_i = seq[i]
    if not (op_i == 'sync' and name_i == 'MTE3_V'):
        return
    if not _v_mte2_after(seq, i):
        return
    if _has_barrier_or_copyout_before(seq, i):
        return
    # 只在有搬出操作但无 PipeBarrier 时报（循环前初始化无搬出，不报）
    # 降为高级别存疑：ping-pong 的 SetFlag<MTE3_V>+<V_MTE2> 可能是合法的
    # 跨流水 flag 链同步，需用户结合数据流判断是否真缺屏障
    findings.append(Finding(
        'SYNC-02', '高', path, ln_i,
        'SetFlag<MTE3_V>+SetFlag<V_MTE2> 释放 buffer 复用前无 PipeBarrier<PIPE_ALL>',
        f'SetFlag<MTE3_V>@{ln_i} 后紧跟 SetFlag<V_MTE2>，前方有搬出操作但无 PipeBarrier。'
        f'若前方是异步搬出（如 CopyOutShareInput），需 PipeBarrier 确保搬出完成；'
        f'若 SetFlag/WaitFlag 链已覆盖跨流水同步则可能无需额外 PipeBarrier。需结合数据流判断。'))


def _compute_after(seq, j: int) -> bool:
    for k in range(j + 1, min(j + 4, len(seq))):
        _ln_k, op_k, name_k = seq[k]
        if op_k in ('compute', 'scalar', 'load', 'store', 'mmad', 'te_copy_l12l0',
                    'te_copy_l0c2gm', 'te_copy'):
            return True
        if op_k == 'sync' and name_k == 'enque':
            break
    return False


def _check_enque_deque(seq, i: int, path: str, _ln_code, findings: List[Finding]) -> None:
    """SYNC-11: EnQue 后立即 DeQue 且 DeQue 后无计算（TQue 当 TBuf 用）。"""
    ln_i, op_i, name_i = seq[i]
    if not (op_i == 'sync' and name_i == 'enque'):
        return
    for j in range(i + 1, min(i + 3, len(seq))):
        ln_j, op_j, name_j = seq[j]
        if op_j == 'sync' and name_j == 'deque':
            if not _compute_after(seq, j):
                findings.append(Finding(
                    'SYNC-11', '性能', path, ln_j,
                    'EnQue→DeQue 后无计算，TQue 当 TBuf 用',
                    f'EnQue@{ln_i} → DeQue@{ln_j}，DeQue 后无计算操作；'
                    f'EnQue/DeQue 的隐式同步被浪费'))
            break
        if op_j not in ('sync',):
            break  # 中间有非 sync 操作则不报


def _check_enque_sync(seq, i: int, path: str, _ln_code, findings: List[Finding]) -> None:
    """SYNC-11: EnQue 后紧跟 SetFlag/WaitFlag 或 PipeBarrier（EnQue 已覆盖的同步重复加）。"""
    ln_i, op_i, name_i = seq[i]
    if not (op_i == 'sync' and name_i == 'enque'):
        return
    for j in range(i + 1, min(i + 3, len(seq))):
        ln_j, op_j, name_j = seq[j]
        if op_j == 'sync' and name_j not in ('deque', 'enque'):
            findings.append(Finding(
                'SYNC-11', '性能', path, ln_j,
                f'EnQue 后紧跟 {name_j}（EnQue 已隐式同步，冗余）',
                f'EnQue@{ln_i} 后紧跟 {name_j}@{ln_j}；'
                f'EnQue/DeQue 已提供隐式同步，手动 Flag/Barrier 冗余'))
            break
        if op_j not in ('sync',):
            break


def _check_reg_store_load(seq, i: int, path: str, _ln_code, findings: List[Finding]) -> None:
    """SYNC-02: Reg:: store → load 无 Reg::LocalMemBar（V 内 store→load 缺屏障）。"""
    if seq[i][1] != 'reg_store':
        return
    ln_i = seq[i][0]
    for j in range(i + 1, min(i + 5, len(seq))):
        ln_j, op_j, _name_j = seq[j]
        if op_j == 'sync':
            break
        if op_j == 'reg_load':
            findings.append(Finding(
                'SYNC-02', '红线', path, ln_j,
                f'疑似 V 内 store→load 缺 Reg::LocalMemBar: load@{ln_j} (store@{ln_i})',
                f'Reg::StoreAlign/StoreUnAlign/DataCopy(mem,reg)@{ln_i} 写入后，'
                f'Reg::LoadAlign/LoadUnAlign/DataCopy(reg,mem)@{ln_j} 读取前缺 '
                f'Reg::LocalMemBar<VEC_STORE, VEC_LOAD>。'
                f'参考 cannbot-dsl vmem_bar("vst_vld") 范式。'))
            break


SEQ_CHECKS = (
    _check_load_compute,
    _check_compute_store,
    _check_compute_scalar,
    _check_pipe_v_store,
    _check_l1l0_mmad,
    _check_mmad_l0c2gm,
    _check_mte3v_mte2,
    _check_enque_deque,
    _check_enque_sync,
    _check_reg_store_load,
)


def _run_seq_checks(path, seq, ln_code, findings: List[Finding]) -> None:
    """对单个作用域操作序列依次运行全部 SEQ_CHECKS 检查模式。"""
    for i, _item in enumerate(seq):
        for check in SEQ_CHECKS:
            check(seq, i, path, ln_code, findings)


def check_flow(events: List[SyncEvent], lines_by_file) -> List[Finding]:
    """SYNC-02 / SYNC-06: 数据流缺同步启发式（高误报，仅候选）。"""
    findings: List[Finding] = []
    for path, lns in lines_by_file:
        # 按作用域组织操作序列
        def_start_lines = _collect_def_start_lines(lns)
        def_sync_flags = _collect_def_sync_flags(lns)
        ln_code = {ln: code for ln, _depth, code, _sig in lns}
        scope_ops: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
        for ln, _depth, code, sig in lns:
            # 函数定义行排除（迭代六）：sig 首次出现的前 2 行是函数签名区
            # （如 ...::CopyIndices( 定义曾被 COPYIN_RE 误当搬入调用）
            if ln in def_start_lines:
                scope_ops[sig].append((ln, 'def', ''))
                continue
            for op, name in _classify_line_ops(code, def_sync_flags):
                scope_ops[sig].append((ln, op, name))
        for _sig, seq in scope_ops.items():
            _run_seq_checks(path, seq, ln_code, findings)
    return findings


def _collect_prev_pipe_signals(lc, line: int) -> Set[str]:
    """回溯 SetFlag 前 5 行的操作所属 PIPE（跨代码块边界即停）。"""
    prev_ops_pipes: Set[str] = set()
    for delta in range(1, 6):
        prev_code = lc.get(line - delta, '')
        if not prev_code:
            continue  # 空行/空白行跳过，继续回溯
        stripped = prev_code.strip()
        if re.match(r'^[\{\}]\s*(?:else\s*)?$', stripped) or stripped.startswith('} else'):
            break  # 跨代码块边界，停止回溯（避免抓上一分支块的操作）
        if COMPUTE_OPS.search(prev_code):
            continue  # 纯 V 计算不作方向判据
        if SCALAR_OPS.search(prev_code) or 'GetValue' in prev_code or 'SetValue' in prev_code:
            prev_ops_pipes.add('S')
        if DATACOPY_RE.search(prev_code):
            dc_dst = re.search(r'DataCopy(?:Pad)?(?:Ext)?(?:<[^>]*>)?\s*\(\s*(\w+)', prev_code)
            if dc_dst and re.search(r'(Gm|GM|gm|Global|output)', dc_dst.group(1)):
                prev_ops_pipes.add('MTE3')  # UB→GM (store)
            else:
                prev_ops_pipes.add('MTE2')  # GM→UB (load) 或 UB→UB
        if MMAD_RE.search(prev_code):
            prev_ops_pipes.add('M')
        if FIXPIPE_RE.search(prev_code):
            prev_ops_pipes.add('FIX')
    return prev_ops_pipes


def check_sync_direction_reversed(events: List[SyncEvent], lines_by_file) -> List[Finding]:
    """SYNC-05: HardEvent 方向写反（如 MTE2→V 数据流却用 V_MTE2 方向）。
    检测模式：SetFlag/WaitFlag 的方向与实际数据流方向相反。
    通过检查 SetFlag 前的操作所属 PIPE vs HardEvent 方向判断。
    """
    findings: List[Finding] = []
    file_linecode: Dict[str, Dict[int, str]] = {}
    if lines_by_file:
        for path, lns in lines_by_file:
            file_linecode[path] = {ln: code for (ln, _depth, code, _sig) in lns}

    # 对每个 SetFlag/WaitFlag，检查其方向是否与数据流匹配
    for e in events:
        if e.kind not in ('set', 'wait') or e.event_type not in REVERSE_MAP:
            continue
        # e.event_type 是可能写反的方向（如 V_MTE2），正确方向是 REVERSE_MAP 中的对应值
        correct_direction = REVERSE_MAP[e.event_type]
        parts = e.event_type.split('_')
        if len(parts) != 2:
            continue
        src_pipe, dst_pipe = parts[0], parts[1]
        lc = file_linecode.get(e.file, {})
        # 如果前面操作属于 dst_pipe（说明数据流是 dst→src，但 flag 写成了 src→dst）
        if dst_pipe in _collect_prev_pipe_signals(lc, e.line):
            findings.append(Finding(
                'SYNC-05', '高', e.file, e.line,
                f'HardEvent 方向可能写反: {e.event_type}（应为 {correct_direction}）',
                f'{e.kind.capitalize()}Flag<{e.event_type}>@{e.line}，'
                f'前面操作属于 {dst_pipe}，数据流方向应为 {dst_pipe}→{src_pipe}（即 {correct_direction}），'
                f'但 flag 方向是 {src_pipe}→{dst_pipe}（即 {e.event_type}）'))
    return findings


def check_hardevent_validity(events: List[SyncEvent]) -> List[Finding]:
    """SYNC-05（脚本侧）: HardEvent 方向合法性校验。
    脚本无法判定方向是否匹配数据流，但能识别未知方向（疑似拼写错误/笔误）。
    真正的「方向与数据流是否匹配」仍需人工按 sync-mechanisms.md §3 判定。
    """
    findings: List[Finding] = []
    seen = set()
    for e in events:
        if e.kind not in ('set', 'wait', 'syncfunc'):
            continue
        ev = e.event_type
        if ev.startswith('PIPE_') or ev in KNOWN_HARDEVENTS:
            continue
        key = (e.file, e.line, ev)
        if key in seen:
            continue
        seen.add(key)
        findings.append(Finding(
            'SYNC-05', '高', e.file, e.line,
            f'未知 HardEvent 方向（疑似笔误）: {ev}',
            f'不在已知合法方向表（{len(KNOWN_HARDEVENTS)} 种）中；请核对是否拼写错误，'
            f'或为脚本未收录的新方向（用 ascendc-docs-search 核实后可补充 KNOWN_HARDEVENTS）'))
    return findings


def check_syncall(events: List[SyncEvent], lines_by_file=None) -> List[Finding]:
    """SYNC-12: SyncAll 用法检查。报高级别的条件：
    SyncAll 嵌套在条件/循环块内（部分核可能不到达 → 死等）。
    函数顶层直排视为合法用法，降为信息级提示。

    已知静态判定边界（不可静态区分，交 LLM/人工）：
    - SyncAll 之前存在提前 return：若 return 条件核间一致（如 tiling 全局广播）则无害，
      若核间不一致（如 sparse 部分核）则死等——脚本不据此判级（曾试点后因误报面过宽回退）。
    - SyncAll 前的部分核退出（如 CalcMask 后 blockIdx>=usedCoreNum 的核 return）是标准用法。
    """
    findings: List[Finding] = []
    if not lines_by_file:
        return findings
    # 函数体基准深度：该函数所有行的最小 depth（函数体内第一行即最小）
    func_base_depth: Dict[Tuple[str, str], int] = {}
    line_depth: Dict[Tuple[int, int], int] = {}
    for path, lns in lines_by_file:
        for ln, depth, _code, sig in lns:
            if sig == '<global>':
                line_depth[(path, ln)] = depth
                continue
            key = (path, sig)
            if key not in func_base_depth or depth < func_base_depth[key]:
                func_base_depth[key] = depth
            line_depth[(path, ln)] = depth
    for e in events:
        if e.kind != 'syncall':
            continue
        base = func_base_depth.get((e.file, e.func))
        d = line_depth.get((e.file, e.line))
        nested = base is not None and d is not None and d > base
        if nested:
            findings.append(Finding(
                'SYNC-12', '高', e.file, e.line,
                'SyncAll 位于条件/循环块内：部分核可能不到达 → 死等',
                f'SyncAll@{e.line} 所在行深度 {d} 大于函数 {e.func} 体基准深度 {base}，'
                f'分支不对称会导致部分核死等；应移至所有核都到达的位置'))
        else:
            findings.append(Finding(
                'SYNC-12', '信息', e.file, e.line,
                'SyncAll 全核同步：确认所有核都到达此点（顶层直排，视为合法用法）',
                '顶层直排的 SyncAll 为正常全核同步；注意分支不对称与单核场景'))
    return findings


def _has_barrier_before(lns, i: int) -> bool:
    """SetAtomicAdd 前 5 行是否有 PipeBarrier（正确写法：barrier 在 atomic 前）。"""
    for k in range(max(0, i - 5), i):
        if PIPEBAR_RE.search(lns[k][2]):
            return True
    return False


def _atomic_sink_finding(lns, i: int, path: str, atomic_line: int) -> Optional[Finding]:
    """向后找 Fixpipe/DataCopy：中间无 PipeBarrier → SYNC-13 候选。"""
    has_barrier_after = False
    for j in range(i + 1, min(i + 30, len(lns))):
        jc = lns[j][2]
        if PIPEBAR_RE.search(jc):
            has_barrier_after = True
            break
        if ATOMIC_DISABLE_RE.search(jc):
            break
        if FIXPIPE_RE.search(jc) or DATACOPY_CO12_RE.search(jc):
            if has_barrier_after:
                break
            return Finding(
                'SYNC-13', '红线', path, lns[j][0],
                'SetAtomicAdd 后、Fixpipe/DataCopy 写 GM 前缺 PipeBarrier<PIPE_FIX>',
                f'SetAtomicAdd@{atomic_line} 后紧跟 Fixpipe/DataCopy@{lns[j][0]}，'
                f'atomic 累加是乱序行为，同流水内也需 PipeBarrier<PIPE_FIX> 保证语义顺序；'
                f'否则非累加数据可能覆盖此前累加结果 → 精度错误')
    return None


def check_atomic_ordering(lines_by_file) -> List[Finding]:
    """SYNC-13: SetAtomicAdd 后、Fixpipe/DataCopy 写 GM 前缺 PipeBarrier<PIPE_FIX>。
    atomic 累加是乱序行为，同流水内也需要 barrier 保证语义顺序。
    """
    findings: List[Finding] = []
    for path, lns in lines_by_file:
        for i, (ln, _depth, code, _sig) in enumerate(lns):
            if not ATOMIC_ADD_RE.search(code):
                continue
            if _has_barrier_before(lns, i):
                continue
            finding = _atomic_sink_finding(lns, i, path, ln)
            if finding:
                findings.append(finding)
    return findings


def _collect_dir_files(dirpath: str) -> List[str]:
    files = []
    for root, _dirs, names in os.walk(dirpath):
        for nm in names:
            if nm.endswith(('.cpp', '.cc', '.c', '.h', '.hpp')):
                files.append(os.path.join(root, nm))
    return files


def collect_files(targets: List[str]) -> List[str]:
    files = []
    for target in targets:
        if os.path.isdir(target):
            files.extend(_collect_dir_files(target))
        elif os.path.isfile(target):
            files.append(target)
    # 去重保序
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    uniq.sort()
    return uniq


def run(targets: List[str], check: str, fmt: str, list_only: bool) -> ScanResult:
    files = collect_files(targets)
    result = ScanResult(files_scanned=files)
    if not files:
        result.findings.append(Finding('INFO', '信息', ','.join(targets), 0,
                                       '未找到 .cpp/.h 文件', ''))
        return result

    all_events: List[SyncEvent] = []
    lines_by_file: List[Tuple[str, List]] = []
    for path in files:
        evs, lns = scan_file(path)
        all_events.extend(evs)
        lines_by_file.append((path, lns))

    result.events = all_events
    multi_file = len(files) > 1

    if list_only:
        return result

    if check in ('pair', 'all'):
        result.findings += check_pair(all_events, multi_file, lines_by_file)
        result.findings += check_early_return(all_events,
                                              [li for _, lns in lines_by_file for li in lns])
    if check in ('flow', 'all'):
        result.findings += check_flow(all_events, lines_by_file)
    if check == 'all':
        result.findings += check_matmul_mix(all_events, lines_by_file, lines_by_file)
        result.findings += check_pipebarrier(all_events)
        result.findings += check_loop_underflow(lines_by_file)
        result.findings += check_hardevent_validity(all_events)
        # 注：曾有 check_set_wait_self_wait（紧邻自等检测），因真实代码中 SetFlag 后紧跟
        # WaitFlag 是 ISASI 标准用法、信息级噪声过大（kv_quant 11文件 27个）已移除。
        result.findings += check_syncall(all_events, lines_by_file)
        result.findings += check_crosscore_pipe_direction(all_events, lines_by_file)
        result.findings += check_atomic_ordering(lines_by_file)
        result.findings += check_sync_direction_reversed(all_events, lines_by_file)
        result.findings += check_missing_sync(lines_by_file)
        result.findings += check_sync_buffer_index(lines_by_file)
        result.findings += check_output_buffer_producer_index(lines_by_file)

    result.findings.sort(key=lambda f: ('红线' not in f.severity,
                                        '高' not in f.severity, f.line))

    # 集成 case_retriever：为红线和高级别候选自动配对历史 case 证据
    try:
        result = _augment_with_case_evidence(result, files)
    except Exception as e:  # case_retriever 失败不影响主流程，但不静默（SKILL 规则：失败必须报告）
        _STDERR_LOGGER.warning("[warn] case_retriever 历史 case 配对失败，候选缺少历史证据: %s", e)

    return result


def _augment_with_case_evidence(result: ScanResult, files: List[str]) -> ScanResult:
    """为每个红线/高级别候选自动配对历史 case 证据，写入 detail 字段。"""
    import importlib.util

    retriever_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'case_retriever.py')
    if not os.path.exists(retriever_path):
        return result

    # 动态导入 case_retriever
    spec = importlib.util.spec_from_file_location('case_retriever', retriever_path)
    if spec is None or spec.loader is None:
        return result
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not files:
        return result

    # 按候选所属文件提取特征并检索（缓存避免重复），关联文件的候选配对各自文件的证据
    feat_cache: Dict[str, Tuple] = {}

    def _features_for(path: str):
        if path not in feat_cache:
            try:
                features = mod.extract_features(path)
                cases = mod.retrieve_cases(features, limit=5)
            except Exception as exc:
                _STDERR_LOGGER.warning("候选特征提取失败 %s: %s", path, exc)
                features, cases = None, []
            feat_cache[path] = (features, cases)
        return feat_cache[path]

    # 为每个候选配对
    for finding in result.findings:
        if finding.severity not in ('红线', '高'):
            continue

        features, cases = _features_for(finding.file)
        if features is None or not cases:
            continue

        candidate = {
            'code': finding.code,
            'line': finding.line,
            'message': finding.message,
            'detail': finding.detail,
        }
        try:
            match = mod.match_candidate_to_cases(candidate, features, cases)
        except Exception as exc:
            _STDERR_LOGGER.warning("候选 case 配对异常: %s", exc)
            continue

        if match and match.get('matched_case'):
            mc = match['matched_case']
            evidence = f' | 历史 case: {mc["repo"]} PR#{mc["pr_id"]} ({mc["fix_type"]})'
            if match.get('fix_hint'):
                evidence += f' | 修复提示: {match["fix_hint"]}'
            if match.get('evidence'):
                evidence += f' | diff证据: {match["evidence"][:100]}...'
            finding.detail = (finding.detail or '') + evidence

    return result


# ─── 输出 ──────────────────────────────────────────────────────────

SEV_ORDER = {'红线': 0, '高': 1, '性能': 2, '信息': 3}


def print_text(result: ScanResult, list_only: bool):
    _LOGGER.info("扫描文件: %d", len(result.files_scanned))
    for f in result.files_scanned:
        _LOGGER.info("  - %s", f)
    _LOGGER.info("同步点总数: %d\n", len(result.events))

    if list_only:
        _LOGGER.info("=== 同步点清单 ===")
        for e in result.events:
            _LOGGER.info("  [%s] %s:%d  %s/%s  %s", e.kind.ljust(14),
                         os.path.basename(e.file), e.line, e.event_type, e.flag_id, e.raw[:60])
        return

    if not result.findings:
        _LOGGER.info("✅ 未发现同步候选问题（注意：脚本为启发式，仍建议人工按 sync-checklist.md 复核数据流）")
        return

    for sev in ['红线', '高', '性能', '信息']:
        items = [f for f in result.findings if f.severity == sev]
        if not items:
            continue
        _LOGGER.info("=== %s (%d) ===", sev, len(items))
        for f in items:
            loc = f"{os.path.basename(f.file)}:{f.line}"
            _LOGGER.info("  [%s] %s", f.code, loc)
            _LOGGER.info("    %s", f.message)
            if f.detail:
                _LOGGER.info("    → %s", f.detail)
        _LOGGER.info("")


def print_json(result: ScanResult):
    out = {
        'files_scanned': result.files_scanned,
        'event_count': len(result.events),
        'findings': [asdict(f) for f in result.findings],
        'events': [asdict(e) for e in result.events],
    }
    _LOGGER.info(json.dumps(out, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(
        description='Ascend C 信号同步静态分析工具（候选问题生成器）',
        epilog='注意：输出为候选问题，需人工按 sync-checklist.md 确认。')
    p.add_argument('targets', nargs='+', help='待扫描的文件或目录（可多个，CrossCore 配对需 cube+vec 同传）')
    p.add_argument('--check', choices=['pair', 'flow', 'all'], default='all',
                   help='pair=Set/Wait 配对; flow=数据流缺同步; all=全部（默认）')
    p.add_argument('--format', choices=['text', 'json'], default='text', help='输出格式')
    p.add_argument('--list-only', action='store_true', help='仅列出同步点，不做检查')
    args = p.parse_args()

    for t in args.targets:
        if not os.path.exists(t):
            _STDERR_LOGGER.error("错误: 路径不存在: %s", t)
            sys.exit(2)

    result = run(args.targets, args.check, args.format, args.list_only)
    if args.format == 'json':
        print_json(result)
    else:
        print_text(result, args.list_only)

    has_redline = any(f.severity == '红线' for f in result.findings)
    sys.exit(1 if has_redline and not args.list_only else 0)
if __name__ == '__main__':
    main()
