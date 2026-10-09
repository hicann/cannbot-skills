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

"""HCCL AICPU algorithm template 评审器（独立工具，不依赖其它 skill）。

三层，默认全跑：
    机械层 W01–W07   两处 CMake 接线、必须实现的方法、props/AlgoType、命名/guard/版权、构造签名、注册宏
    语义层 C/M/N/R/H/X  scratch 倍数、repeat 与 stride、同步配对、notify 自洽、软归约兜底、
                     channel 取用，以及 mesh / nhr / ring / hd 四族各自的算法不变式
    契约层 S00–S03   --spec 时先校验 spec，再比对代码与 dataflow spec

顺序不能反：没接进 CMake 的文件，语义评得再细也没有意义（已用别的工具查过接线可加 --semantic-only）。

用法：
    python3 scripts/review_template.py --repo <HCCL 仓根目录> src/ops/all_reduce/algorithm/template/aicpu/ins_temp_xxx.cc
    python3 scripts/review_template.py --repo <HCCL 仓根目录> --all
    python3 scripts/review_template.py --repo <HCCL 仓> <file.cc> --spec my-spec.md   # 再比对 dataflow spec
    python3 scripts/review_template.py --rules                                       # 打印规则目录
    python3 scripts/review_template.py --repo <HCCL 仓> --base HEAD                   # 未提交/已暂存改动
    python3 scripts/review_template.py --repo <HCCL 仓> --base HEAD^                  # 最近一次已提交改动

严重度：BLOCKER（必然错，合入前必须改） / MAJOR（很可能错，要作者答复）
        / MINOR（建议改） / INFO（提示，供人判断）
退出码：有 BLOCKER 时 1；加 --strict 则 MAJOR 也算 1。
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

BLOCKER, MAJOR, MINOR, INFO = "BLOCKER", "MAJOR", "MINOR", "INFO"
ORDER = {BLOCKER: 0, MAJOR: 1, MINOR: 2, INFO: 3}
MARK = {BLOCKER: "✗ BLOCKER", MAJOR: "▲ MAJOR  ", MINOR: "! MINOR  ", INFO: "· INFO   "}

# 规则目录：ID → (一句话说明, 参考文档)
RULES = [
    (
        "W01",
        "没登记进两处 CMake（局部 CMakeLists + scatter_aicpu_kernel.cmake）",
        "references/02 §0",
    ),
    ("W02", "缺少 InsAlgTemplateBase 要求的方法实现", "references/02 §0"),
    ("W03", "props / AlgoType 写错（含已删除的 isNhr）", "references/02 §0"),
    ("W04", "命名、include guard、版权头、namespace 不合规", "references/02 §0"),
    ("W05", "构造函数签名与 executor 的调用约定对不上", "references/02 §0"),
    ("W06", "没有任何 REGISTER_EXEC_V2 引用这个 template", "references/02 §0"),
    ("W07", "DPU template 缺 REGISTER_TEMPLATE_V2 字符串注册", "references/02 §0"),
    ("C01", "CalcScratchMultiple 里出现 repeatNum，倍数被双重放大", "references/02 §1"),
    (
        "C02",
        "CalcScratchMultiple 缺实现，或变体子类继承了父类的倍数",
        "references/02 §1",
    ),
    (
        "C03",
        "KernelRun 没有 repeat 循环，被分层 executor 复用时只搬第 0 块",
        "references/02 §2",
    ),
    ("C04", "scratch 地址跨度没区分 inBuffType == HCCL_BUFFER", "references/02 §2"),
    ("C05", "缺少 count == 0 早退", "references/02 §5"),
    ("C06", "缺少单 rank（rankSize == 1）早退", "references/02 §5"),
    ("C07", "PreSyncInterThreads / PostSyncInterThreads 不成对", "references/02 §3"),
    ("C08", "CalcRes/GetRes 里 thread 与 notify 的三个字段没填全", "references/02 §4"),
    (
        "C09",
        "做归约却看不到 64-bit / PROD 的兜底（template 内或选路层二选一）",
        "references/02 §6",
    ),
    (
        "C10",
        "软归约三步顺序不是 BatchModeEnd → BatchModeStart → ThreadJoin",
        "references/02 §6",
    ),
    ("C11", "channels.at() 之前没有 channels.count() 校验", "references/02 §7"),
    (
        "C12",
        "返回 HcclResult 的原语调用没有被 CHK_RET / CHK_PRT_RET 包住",
        "references/02 §8",
    ),
    ("C13", "Describe() 没带 templateRankSize_", "references/02 §9"),
    ("C14", "同时用了写模式与读模式原语但没有 PCIe 判据开关", "references/02 §10"),
    (
        "C15",
        "channels 的 key 疑似用了算法内序号而非通信域 userRank",
        "references/02 §7",
    ),
    ("C16", "scratch→out 本地拷贝所在函数缺输出落位分流", "references/02 §2"),
    ("M01", "Mesh one-shot 的 scratch 倍数应为 templateRankSize_", "references/03 §1"),
    ("M02", "Mesh two-shot 的 scratch 倍数应为 1 或 2", "references/03 §2"),
    ("M03", "Mesh 从流循环没有错峰 (myRank_ + i) % N", "references/03 §3"),
    (
        "M04",
        "Mesh 的 thread 数应由 templateRankSize_ / channelsPerRank_ 推导",
        "references/03 §4",
    ),
    ("M05", "Mesh one-shot 归约类缺收尾 LocalReduce", "references/03 §1"),
    ("N01", "NHR 的 scratch 倍数应为 1（图模式可为 0）", "references/04 §1"),
    ("N02", "NHR 缺 PreCopy(in→scratch) / PostCopy(scratch→out)", "references/04 §2"),
    ("N03", "NHR 的 step 数写死，没有从 rankSize 推导", "references/04 §3"),
    ("N04", "用了 channelsPerRank_ 却没覆写 GetRes + GetThreadNum", "references/04 §4"),
    (
        "N05",
        "人工确认：NHR 每 step 取 channel 前 from/to 两侧都要校验",
        "references/04 §5",
    ),
    ("R01", "Ring 却对全体 rank 建/取 channel，应只用左右邻居", "references/05 §2"),
    ("R02", "Ring 的 step 循环上界写死 / 与 rankSize 无关", "references/05 §3"),
    (
        "R03",
        "Ring 每步的 chunk 索引看不到 (idx - step + N) % N 的推导",
        "references/05 §3",
    ),
    (
        "R04",
        "Ring 相邻 step 之间没有同步，存在读到未写完数据的风险",
        "references/05 §4",
    ),
    ("R05", "轮转分片 Ring 的 scratch 倍数需按算子和模式确认", "references/05 §5"),
    (
        "R06",
        "Ring 照搬了 Mesh 的 thread 模型（slaveThreadNum = N-1）",
        "references/05 §5",
    ),
    ("R07", "Ring 的 RS 与 AG 两阶段没用同一张分片表", "references/05 §3"),
    (
        "R08",
        "人工确认 Ring 形态、全局覆盖与收发依赖（不适用轮转判据时）",
        "references/05 §7",
    ),
    ("R09", "人工确认链式 relay 收下即留的输出落位与回搬依据", "references/05 §8"),
    ("H01", "HD 没有处理非 2 的幂 rankSize（part1 / blockSize）", "references/06 §2"),
    ("H02", "HD 的步数不是从 log2(blockSize) 推导", "references/06 §3"),
    ("H03", "HD 每步对端看不到 XOR / 距离折半的推导", "references/06 §3"),
    ("H04", "HD 每步的数据量没有随 step 折半 / 翻倍", "references/06 §3"),
    (
        "H05",
        "HD 被淘汰的 part1 rank 没有排除在主循环外或末尾没回传",
        "references/06 §2",
    ),
    ("H06", "HD 的 scratch 倍数应为 1", "references/06 §5"),
    (
        "X01",
        "该拓扑在 alg_parse.h/.cc 的选路登记不完整",
        "references/05 §1、references/06 §1",
    ),
    ("S00", "dataflow spec 格式不完整或仍有 TBD，无法可靠比对", "references/07 §1"),
    ("S01", "代码与 dataflow spec 的 scratch 倍数对不上", "references/07"),
    ("S02", "代码与 spec 的 RPT 声明对不上", "references/07"),
    ("S03", "代码与 spec 的同步点数量对不上", "references/07"),
]

# 返回 HcclResult、必须被 CHK_RET / CHK_PRT_RET 包住的原语
# ring / HD 里这几条纯粹是把算法性质映射到本仓 API 约定（scratch 倍数、线程模型）：
# 既没有公开文献背书，仓内也没有实现验证过它们，所以只报 INFO 交人确认，
# 不拿它们阻塞或要求作者答复。有文献背书的判据（步数、片索引、XOR 对端等）不在此列。
UNVERIFIED_MAPPING = {"R05", "R06", "H06"}

STATUS_PRIMS = [
    "SendRecvBatchWriteReduce",
    "SendRecvBatchWrite",
    "SendRecvReadReduce",
    "SendRecvBatchRead",
    "LocalCopy",
    "LocalReduce",
    "PreSyncInterThreads",
    "PostSyncInterThreads",
    "HcommThreadJoin",
    "HcommBatchModeStart",
    "HcommBatchModeEnd",
]
WRITE_PRIMS = ["SendRecvBatchWrite", "SendRecvBatchWriteReduce"]
READ_PRIMS = ["SendRecvBatchRead", "SendRecvReadReduce"]


class Finding(object):
    def __init__(self, rule, sev, msg, line=None, hint=None, evidence=None):
        self.rule, self.sev, self.msg = rule, sev, msg
        self.line, self.hint, self.evidence = line, hint, evidence


class Review(object):
    def __init__(self, title, topo, variant):
        self.title, self.topo, self.variant = title, topo, variant
        self.findings = []

    def add(self, rule, sev, msg, line=None, hint=None, evidence=None):
        self.findings.append(Finding(rule, sev, msg, line, hint, evidence))

    def count(self, sev):
        return sum(1 for f in self.findings if f.sev == sev)

    def dump(self):
        print("=" * 78)
        print("%s   [拓扑 %s / 变体 %s]" % (self.title, self.topo, self.variant or "-"))
        print("=" * 78)
        if self.topo in ("ring", "hd"):
            print(
                "  ℹ R/H 规则含尚未获 CANN 官方实现验证的算法映射；X01/W/C 属仓内可核验规则。"
            )
        if not self.findings:
            print("  语义检查未发现问题（仍需人工过 references 里的清单）\n")
            return
        for f in sorted(self.findings, key=lambda x: (ORDER[x.sev], x.rule)):
            loc = ":%d" % f.line if f.line else ""
            print("%s %s  %s%s" % (MARK[f.sev], f.rule, f.msg, loc))
            if f.evidence:
                print("             > %s" % f.evidence.strip()[:110])
            if f.hint:
                for ln in f.hint.splitlines():
                    print("             %s" % ln)
        print()

    def dump_attributed(self, fresh, comparable, show_baseline=False):
        """--base 模式输出；默认仅展开本次引入，show_baseline 时再展开存量。

        归属靠「与 base 快照比对 finding 身份」得出，不看行号——纯删除、
        改 .h、改 CMake 这些改动本来就没有「新增行」可锚。
        """
        print("=" * 78)
        print("%s   [拓扑 %s / 变体 %s]" % (self.title, self.topo, self.variant or "-"))
        print("=" * 78)
        if self.topo in ("ring", "hd"):
            print(
                "  ℹ R/H 规则含尚未获 CANN 官方实现验证的算法映射；X01/W/C 属仓内可核验规则。"
            )
        if not self.findings:
            print("  语义检查未发现问题（仍需人工过 references 里的清单）\n")
            return
        if not comparable:
            print(
                "  ! 取不到 base 版本，无法比对——以下全部按「本次引入」计（宁可多报）"
            )
        ids = {id(f) for f in fresh}
        old = [f for f in self.findings if id(f) not in ids] if show_baseline else []
        for title, items, detail in (
            ("★ 本次改动引入（要作者答复）", fresh, True),
            ("· 存量欠账（base 上本来就有，供参考，别算到作者头上）", old, False),
        ):
            if not items:
                continue
            print("-- %s --" % title)
            for f in sorted(items, key=lambda x: (ORDER[x.sev], x.rule)):
                loc = ":%d" % f.line if f.line else ""
                print("%s %s  %s%s" % (MARK[f.sev], f.rule, f.msg, loc))
                if not detail:
                    continue
                if f.evidence:
                    print("             > %s" % f.evidence.strip()[:110])
                if f.hint:
                    for ln in f.hint.splitlines():
                        print("             %s" % ln)
        print()


# --------------------------------------------------------------------------- 基础工具
def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def strip_comments(text):
    """去掉注释，避免注释里的字样把检查骗过去；保留换行以维持行号。"""

    def repl(m):
        s = m.group(0)
        return "".join(c if c == "\n" else " " for c in s)

    text = re.sub(r"/\*.*?\*/", repl, text, flags=re.S)
    text = re.sub(r"//[^\n]*", repl, text)
    return text


def lineno(text, pos):
    return text.count("\n", 0, pos) + 1


def line_at(text, pos):
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return text[start:end if end != -1 else len(text)].strip()


def has_scratch_to_output_copy(code):
    """仅识别可直接追到 DataSlice 构造的 scratch -> output LocalCopy。"""

    def slice_args(before, name):
        decls = list(
            re.finditer(
                r"\bDataSlice\s+%s\s*(?:=\s*DataSlice\s*)?\(([^;]*)\)\s*;"
                % re.escape(name),
                before,
                re.S,
            )
        )
        return decls[-1].group(1) if decls else ""

    for call in re.finditer(r"\bLocalCopy\s*\([^,;]+,\s*(\w+)\s*,\s*(\w+)\s*\)", code):
        src, dst = (slice_args(code[: call.start()], name) for name in call.groups())
        if re.search(r"\bhcclBuff\.addr\b", src) and re.search(r"\boutputPtr\b", dst):
            return True
    return False


def output_copy_without_guard(code, cls):
    """只认可发生拷贝的同一方法中读取 outBuffType。"""
    bodies = []
    for method in re.finditer(r"\b%s::\w+\s*\(" % re.escape(cls), code):
        opening = code.find("{", method.end())
        if opening < 0 or ";" in code[method.end():opening]:
            continue
        depth = 1
        end = opening + 1
        while end < len(code) and depth:
            depth += (code[end] == "{") - (code[end] == "}")
            end += 1
        bodies.append(code[opening:end])
    for body in bodies or [code]:
        if (
            has_scratch_to_output_copy(body)
            and re.search(r"\bhcclBuffBaseOff\b", body)
            and re.search(r"\boutBuffBaseOff\b", body)
            and not re.search(r"\.outBuffType\b", body)
        ):
            return True
    return False


def has_repeat_loop(text):
    """只把循环条件里的 repeatNum 视为 repeat 循环，日志或普通表达式不算。"""
    return bool(
        re.search(r"\bfor\s*\([^;]*;[^;]*\brepeatNum_?\b[^;]*;", text, re.S)
        or re.search(r"\bwhile\s*\([^)]*\brepeatNum_?\b", text, re.S)
    )


def func_body(text, cls, method):
    """取 `Cls::method(...) { ... }` 的函数体（含起止行号）；找不到返回 (None, 0)。"""
    for m in re.finditer(r"\b%s::%s\s*\(" % (re.escape(cls), re.escape(method)), text):
        brace = text.find("{", m.end())
        if brace == -1:
            continue
        depth, i = 0, brace
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace:i + 1], lineno(text, brace)
            i += 1
    return None, 0


def inline_body(hh, method):
    """取头文件里 `method(...) override { ... }` 的内联实现。"""
    m = re.search(r"\b%s\s*\([^;{]*\)[^;{]*\{" % re.escape(method), hh)
    if not m:
        return None, 0
    brace = hh.rfind("{", 0, m.end())
    depth, i = 0, brace
    while i < len(hh):
        if hh[i] == "{":
            depth += 1
        elif hh[i] == "}":
            depth -= 1
            if depth == 0:
                return hh[brace:i + 1], lineno(hh, brace)
        i += 1
    return None, 0


def body_of(cc, hh, cls, method):
    b, ln = func_body(cc, cls, method)
    if b is not None:
        return b, ln, "cc"
    b, ln = inline_body(hh, method)
    if b is not None:
        return b, ln, "h"
    return None, 0, None


# --------------------------------------------------------------------------- 拓扑识别
def split_words(cls, filebase):
    """把类名+文件名切成下划线分隔的大写词串，用于需要词边界的短词匹配。

    紧凑形式（抹掉所有分隔符）对 MESH / NHR / RING 这种长词够用，但对 `HD` 这种
    两字母缩写不行：抹掉下划线后 HD 永远被前后字母夹住，词边界判定必然落空
    （InsTempAllReduceHd → INSTEMPALLREDUCEHD）。这里在驼峰边界补下划线并保留
    原有下划线，让 `_HD_` / `_HD$` 这类边界能真正匹配上。
    """
    camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cls)
    return re.sub(r"[^A-Z0-9]+", "_", (camel + "_" + filebase).upper())


def detect_topo(cls, filebase, cc, hh):
    """识别算法族。类名/文件名优先，其次看代码特征。返回 mesh / nhr / ring / hd / unknown。"""
    tokens = re.sub(r"[^A-Z0-9]", "", (cls + filebase).upper())
    words = split_words(cls, filebase)
    text = cc + hh
    if "NHR" in tokens or "GetNHRStep" in text or "NHRStepInfo" in text:
        return "nhr"
    if "MESH" in tokens:
        return "mesh"
    if "RING" in tokens:
        return "ring"
    # HALVINGDOUBLING / RECURSIVEHD 走紧凑形式（跨分隔符也能认）；
    # 裸 HD 必须走词边界形式，否则 InsTempAllReduceHd 这类正常命名一律漏认。
    # 末尾的变量名兜底只是补充：真实现未必叫 blockSize_ / part1Size_，不能只靠它。
    if (
        re.search(r"HALVINGDOUBLING|RECURSIVEHD", tokens)
        or re.search(r"(?<![A-Z0-9])HD(?![A-Z0-9])", words)
        or ("blockSize_" in text and "part1Size_" in text)
    ):
        return "hd"
    # 代码特征兜底：只认 mesh。ring / hd 仓内无存量实现，误判代价高，
    # 认不出时宁可报 unknown 让人用 --topo 指定。
    if re.search(r"CalcChannelRequestMesh|Mesh1D", text):
        return "mesh"
    return "unknown"


OP_TOKENS = [
    ("ALLGATHERV", "all_gather_v"),
    ("ALLGATHER", "all_gather"),
    ("REDUCESCATTERV", "reduce_scatter_v"),
    ("REDUCESCATTER", "reduce_scatter"),
    ("ALLREDUCE", "all_reduce"),
    ("ALLTOALLV", "all_to_all_v"),
    ("ALLTOALL", "all_to_all_v"),
    ("BROADCAST", "broadcast"),
    ("BARRIER", "barrier"),
    ("SCATTER", "scatter"),
    ("GATHER", "reduce"),
    ("REDUCE", "reduce"),
]


def infer_op(cls, filebase):
    """从类名 / 文件名推算子。只在路径不在 src/ops/<op>/ 下时用。"""
    u = re.sub(r"[^A-Z0-9]", "", (cls + filebase).upper())
    for token, op in OP_TOKENS:
        if token in u:
            return op
    return ""


def detect_variant(cls, filebase, cc):
    tokens = re.sub(r"[^a-z0-9]", "", (cls + filebase).lower())
    for key, name in (
        ("oneshot", "one-shot"),
        ("twoshot", "two-shot"),
        ("meshchunk", "mesh-chunk"),
        ("chunk", "chunk"),
        ("omnipipe", "omnipipe"),
        ("dpu", "dpu"),
        ("intra", "intra"),
        ("inter", "inter"),
        ("detour", "z-detour"),
        ("pcie", "pcie"),
    ):
        if key in tokens:
            return name
    return ""


def ring_shape(f):
    """按算子语义选择判据；Broadcast 仍须人工确认是否为链式全量转发。"""
    op = f.op_by_name or f.op
    if op in ("all_reduce", "all_gather", "reduce_scatter", "reduce_scatter_v"):
        return "rotating"
    if op == "broadcast":
        return "broadcast"
    return "unknown"


_CLASS_INDEX = {}


# --------------------------------------------------------------------------- 仓布局
# 不同 HCCL 分支下 template / executor 的位置不一样。必须在启动时识别出来：
# 认不出就直接失败，**不允许**降级成"扫到 0 个模板"或"静默跳过接线检查"。
class Layout(object):
    def __init__(self, name, tmid, emid):
        self.name = name  # 布局名，出现在报错与报告里
        self.tmid = tmid  # template 目录相对 src/ops/<op> 的中缀
        self.emid = emid  # executor 目录相对 src/ops/<op> 的中缀

    def template_glob(self, repo, ext="cc"):
        return os.path.join(repo, "src/ops/*", self.tmid, "*." + ext)

    def executor_glob(self, repo):
        return os.path.join(repo, "src/ops/*", self.emid, "*.cc")

    def op_of(self, cc_path):
        """从模板路径里取算子名；不在本布局的 template 目录下则返回 None。"""
        m = re.search(
            r"src/ops/([^/]+)/%s/" % re.escape(self.tmid), cc_path.replace(os.sep, "/")
        )
        return m.group(1) if m else None

    def cmake_local(self, repo, op):
        return os.path.join(repo, "src/ops", op, self.tmid, "CMakeLists.txt")

    def kernel_needle(self, op, filebase):
        return "ops/%s/%s/%s.cc" % (op, self.tmid, filebase)


# 只支持当前 HCCL 的目录结构。更早的分支（template/aicpu 平铺、没有 alg_parse.h 的 V1）
# 不在支持范围内：那里 AlgoType 这个真源都不存在，X01 / W03 无从判断。
LAYOUTS = [
    Layout("current", "algorithm/template/aicpu", "algorithm/executor"),
]


def detect_layout(repo):
    """按模板文件数量挑一个布局；一个都匹配不到返回 None（调用方必须失败退出）。"""
    best, best_n = None, 0
    for lay in LAYOUTS:
        n = len(
            [p for p in glob.glob(lay.template_glob(repo)) if "/op_common/" not in p]
        )
        if n > best_n:
            best, best_n = lay, n
    return best


def class_index(repo, layout):
    """class 名 → 头文件路径。用来沿继承链找父类实现（变体子类靠父类提供大半实现）。"""
    if repo in _CLASS_INDEX:
        return _CLASS_INDEX[repo]
    idx = {}
    for h in glob.glob(layout.template_glob(repo, "h")):
        for m in re.finditer(r"class\s+(\w+)\s*:\s*public\s+(\w+)", read(h)):
            idx.setdefault(m.group(1), h)
    _CLASS_INDEX[repo] = idx
    return idx


def family_text(repo, layout, parent, depth=3):
    """把父类（及其祖先）的 .h/.cc 文本拼起来，供「全文找不到 X」这类检查使用。"""
    idx, cc_all, hh_all = class_index(repo, layout), [], []
    while parent and parent != "InsAlgTemplateBase" and depth > 0:
        h = idx.get(parent)
        if not h:
            break
        c = h[:-2] + ".cc"
        hh_all.append(strip_comments(read(h)))
        if os.path.exists(c):
            cc_all.append(strip_comments(read(c)))
        m = re.search(r"class\s+%s\s*:\s*public\s+(\w+)" % re.escape(parent), read(h))
        parent = m.group(1) if m else None
        depth -= 1
    return "\n".join(cc_all), "\n".join(hh_all)


def load_algo_registrations(repo):
    """分别解析 alg_parse.cc 的三个登记点，避免把“缺一处”误报成“缺三处”。"""
    path = os.path.join(repo, "src/common/alg_parse.cc")
    if not os.path.exists(path):
        return set(), set(), set()
    text = read(path)

    def block(pattern):
        m = re.search(pattern, text, re.S)
        return m.group(1) if m else ""

    algo_block = block(r"ALGO_TYPES\s*=\s*\{(.*?)\};")
    type_to_name_block = block(
        r"GetAlgoTypeToNameMap\s*\([^)]*\)\s*\{.*?\bmap\s*=\s*\{(.*?)\};"
    )
    name_to_type_block = block(
        r"GetAlgoNameToTypeMap\s*\([^)]*\)\s*\{.*?\bmap\s*=\s*\{(.*?)\};"
    )
    algo_names = set(re.findall(r'\{\s*"([^"]+)"\s*,', algo_block))
    type_to_name = set(
        re.findall(r'\{\s*AlgoType::(\w+)\s*,\s*"[^"]+"', type_to_name_block)
    )
    name_to_type = set(
        re.findall(r'\{\s*"[^"]+"\s*,\s*AlgoType::(\w+)', name_to_type_block)
    )
    return algo_names, type_to_name, name_to_type


def load_algo_types(repo):
    path = os.path.join(repo, "src/common/alg_parse.h")
    if not os.path.exists(path):
        return set()
    m = re.search(r"enum\s+class\s+AlgoType\s*:[^{]*\{(.*?)\}", read(path), re.S)
    if not m:
        return set()
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return {v.strip() for v in body.split(",") if v.strip()}


# --------------------------------------------------------------------------- 事实抽取
class Facts(object):
    """从源码里抽出评审要用的事实。所有正则都跑在**去注释**后的文本上。"""

    def __init__(
        self, repo, cc_path, cls, parent, cc, hh, cc_raw="", hh_raw="", layout=None
    ):
        self.repo, self.path, self.cls, self.parent = repo, cc_path, cls, parent
        self.layout = layout
        self.cc_raw, self.hh_raw = cc_raw or cc, hh_raw or hh
        self.is_variant = parent != "InsAlgTemplateBase"
        self.cc, self.hh = cc, hh
        self.all = cc + "\n" + hh
        self.filebase = os.path.basename(cc_path)[:-3]
        op_in_tree = layout.op_of(cc_path) if layout else None
        # 路径不在仓里时（评审仓外的补丁文件、自测夹具）退回按类名推算子
        self.op = op_in_tree or infer_op(cls, self.filebase)
        self.in_repo_tree = op_in_tree is not None
        # 目录不等于语义：src/ops/all_reduce/ 下也放着 AllGather / ReduceScatter 的分层构件
        # （ins_temp_all_gather_nhr_dpu_inter.cc 就是 2 级 AllReduce 的 inter 段）。
        # 判 scratch 倍数要按**类名**认语义，且 AICPU 软归约变体天然要 N 份暂存。
        self.op_by_name = infer_op(cls, self.filebase)
        _tok = re.sub(r"[^A-Z0-9]", "", (cls + self.filebase).upper())
        self.is_allreduce_semantics = (
            self.op_by_name == "all_reduce"
            or (not self.op_by_name and self.op == "all_reduce")
        ) and "AICPUREDUCE" not in _tok
        self.topo = detect_topo(cls, self.filebase, cc, hh)
        self.variant = detect_variant(cls, self.filebase, cc)

        self.scratch_body, scratch_brace_line, scratch_where = body_of(
            cc, hh, cls, "CalcScratchMultiple"
        )
        self.scratch_rets = re.findall(r"return\s+([^;]+);", self.scratch_body or "")
        self.scratch_expr = " | ".join(
            dict.fromkeys(
                resolve_expr(self.scratch_body or "", r) for r in self.scratch_rets
            )
        )
        # 锚在真正决定倍数的那条语句上，而不是函数左花括号：
        # 花括号那行几乎不会被改，--base 归属会把「改了倍数」误判成存量欠账。
        self.scratch_line = scratch_brace_line
        if self.scratch_body and scratch_where == "cc":
            base_off = cc.find(self.scratch_body)
            if base_off != -1:
                names = [
                    n
                    for n in re.findall(
                        r"return\s+([A-Za-z_]\w*)\s*;", self.scratch_body
                    )
                ]
                pat = (
                    (r"\b(?:%s)\b\s*=" % "|".join(map(re.escape, names)))
                    if names
                    else r"return\s+[^;]+;"
                )
                hit = re.search(pat, self.scratch_body) or re.search(
                    r"return\s+[^;]+;", self.scratch_body
                )
                if hit:
                    self.scratch_line = lineno(cc, base_off + hit.start())

        self.kernel_body, self.kernel_line, _ = body_of(cc, hh, cls, "KernelRun")
        calc, _, _ = body_of(cc, hh, cls, "CalcRes")
        getres, _, _ = body_of(cc, hh, cls, "GetRes")
        self.res_body = (calc or "") + "\n" + (getres or "")
        self.desc_body, _, _ = body_of(cc, hh, cls, "Describe")

        fam_cc, fam_hh = (
            family_text(repo, layout, parent) if self.is_variant else ("", "")
        )
        self.fam_res = ""
        if fam_cc:
            for meth in ("CalcRes", "GetRes"):
                for mm in re.finditer(r"\w+::%s\s*\(" % meth, fam_cc):
                    b, _ = func_body(
                        fam_cc,
                        fam_cc[fam_cc.rfind(" ", 0, mm.start()) + 1:mm.start()].strip(
                            ":"
                        )
                        or "X",
                        meth,
                    )
                    if b:
                        self.fam_res += b
                    break
        # scan_* 用于「全文找不到 X」这类检查：变体子类要连父类一起看，否则全是误报
        self.scan_cc = cc + "\n" + fam_cc
        self.scan_all = self.all + "\n" + fam_cc + "\n" + fam_hh
        self.is_reduce_op = self.op in (
            "all_reduce",
            "reduce",
            "reduce_scatter",
            "reduce_scatter_v",
        )
        self.does_reduce = bool(
            re.search(r"\bLocalReduce\s*\(|Reduce\s*\(.*reduceOp", cc)
        ) or any(p in cc for p in ("SendRecvBatchWriteReduce", "SendRecvReadReduce"))

    def has(self, pattern, where=None):
        return re.search(pattern, where if where is not None else self.all) is not None

    def count_calls(self, name, where=None):
        text = where if where is not None else self.cc
        return len(re.findall(r"\b%s\s*\(" % re.escape(name), text))

    def find(self, pattern, where=None):
        text = where if where is not None else self.cc
        return [(lineno(text, m.start()), m) for m in re.finditer(pattern, text)]


def resolve_expr(body, expr):
    """return 的是局部变量时，回溯它在函数体里的赋值（仓内普遍写成 `u64 multiple = 2; return multiple;`）。"""
    expr = (expr or "").strip()
    for _ in range(3):
        if not re.match(r"^[A-Za-z_]\w*$", expr):
            break
        vals = [
            v.strip()
            for v in re.findall(r"\b%s\s*=\s*([^;]+);" % re.escape(expr), body)
        ]
        vals = [v for v in vals if v and v != expr]
        if not vals:
            break
        uniq = list(dict.fromkeys(vals))
        expr = " | ".join(uniq)
        if len(uniq) != 1:
            break
    return expr


def scratch_matches(expr, wanted):
    """scratch 倍数表达式是否落在期望集合内。wanted 是若干片段，命中任一即可。"""
    if not expr:
        return False
    for w in wanted:
        if w == "1" or w == "2" or w == "0":
            if re.search(r"(^|[^\w.])%s\s*($|[^\w.])" % w, expr):
                return True
        elif w in expr:
            return True
    return False


# --------------------------------------------------------------------------- 机械层（W01–W07）
# 本 skill 要能独立使用，所以机械层自带一份。它必须排在语义层前面：
# 一个没接进 CMake、或缺纯虚实现的文件，语义评得再细也没有意义。
REQUIRED_METHODS = [
    ("Describe", "纯虚，不实现编译不过"),
    ("GetNotifyIdxMainToSub", "纯虚，不实现编译不过"),
    ("GetNotifyIdxSubToMain", "纯虚，不实现编译不过"),
    ("CalcRes", "基类默认实现直接 HCCL_ERROR 返回失败"),
    ("KernelRun", "基类默认实现直接 HCCL_ERROR 返回失败"),
    ("CalcScratchMultiple", "基类默认实现直接 HCCL_ERROR 返回失败"),
]


def infer_algo_type(cls):
    """按类名推断该写哪个 AlgoType；推不出返回 None。"""
    u = re.sub(r"[^A-Z0-9]", "", cls.upper())
    if "NHR" in u:
        return "NHR_AICPU_REDUCE" if "AICPUREDUCE" in u else "NHR"
    if "MESHCHUNK" in u:
        return "MESH_CHUNK_TWOSHOT" if "TWOSHOT" in u else "MESH_CHUNK"
    if "ONESHOT" in u:
        return "MESH_ONESHOT"
    if "TWOSHOT" in u:
        return "MESH_TWOSHOT"
    if "MESH" in u:
        return "MESH"
    return None


def check_mechanical(f, rv, algo_types):
    cc_raw, hh_raw, cls = f.cc_raw, f.hh_raw, f.cls

    # ---- W01 两处 CMake 登记 ----
    if f.op and f.in_repo_tree:
        local = f.layout.cmake_local(f.repo, f.op)
        kernel = os.path.join(f.repo, "src/scatter_aicpu_kernel.cmake")
        if os.path.exists(local):
            if ("/%s.cc" % f.filebase) not in read(local):
                rv.add(
                    "W01",
                    BLOCKER,
                    "未登记到 %s" % os.path.relpath(local, f.repo),
                    hint="host 侧 libhccl.so 根本不会编它。加到 set(src_list ...) 里：\n"
                    "    ${CMAKE_CURRENT_SOURCE_DIR}/%s.cc" % f.filebase,
                )
        else:
            rv.add("W01", MAJOR, "找不到 %s" % local)
        needle = f.layout.kernel_needle(f.op, f.filebase)
        if os.path.exists(kernel):
            if needle not in read(kernel):
                rv.add(
                    "W01",
                    BLOCKER,
                    "未登记到 src/scatter_aicpu_kernel.cmake",
                    hint="★ 本仓最高频的接线 bug：只加了 host 侧 → 编得过，\n"
                    "但 device 侧 libscatter_aicpu_kernel.so 里没有这个符号，运行时才崩。\n"
                    "加到 add_library(scatter_aicpu_kernel SHARED ...) 块里：\n"
                    "    ${CMAKE_CURRENT_SOURCE_DIR}/%s" % needle,
                )
        else:
            rv.add("W01", MAJOR, "找不到 %s" % kernel)

    # ---- W02 必须实现的方法 ----
    for method, why in REQUIRED_METHODS:
        in_cc = (
            re.search(r"\b%s::%s\s*\(" % (re.escape(cls), method), cc_raw) is not None
        )
        inline = (
            re.search(r"\b%s\s*\([^;]*\)[^;]*\{" % method, hh_raw, re.S) is not None
        )
        if in_cc or inline:
            continue
        if f.is_variant:
            continue  # 变体子类由父类提供，见 family_text()
        rv.add("W02", BLOCKER, "缺少 %s() 的实现" % method, hint=why)

    # ---- W03 props / AlgoType ----
    want = infer_algo_type(cls)
    if "isNhr" in hh_raw:
        rv.add(
            "W03",
            BLOCKER,
            "用了已删除的 TemplateProp::isNhr",
            hint="当前 TemplateProp 只有 algoType 一个成员，指向不存在的成员会编译失败。\n"
            "改成：static constexpr TemplateProp props = {.algoType = AlgoType::%s};"
            % (want or "<枚举值>"),
        )
    else:
        decl = re.search(r"TemplateProp\s+props\s*=\s*\{([^}]*)\}", hh_raw)
        if decl:
            m = re.search(r"AlgoType::(\w+)", decl.group(1))
            if not m:
                rv.add(
                    "W03", BLOCKER, "props 没有按 {.algoType = AlgoType::XXX} 的形式写"
                )
            elif algo_types and m.group(1) not in algo_types:
                rv.add(
                    "W03",
                    BLOCKER,
                    "AlgoType::%s 不在 src/common/alg_parse.h 的枚举里" % m.group(1),
                    hint="可选值：%s" % "、".join(sorted(algo_types)),
                )
            elif want and (m.group(1).startswith("NHR") != want.startswith("NHR")):
                rv.add(
                    "W03",
                    MINOR,
                    "类名像 %s，props 却声明 AlgoType::%s" % (want, m.group(1)),
                    hint="拓扑家族（MESH_* / NHR_*）对不上通常是复制粘贴漏改。",
                )

    # ---- W04 命名 / guard / 版权 / namespace ----
    if not f.filebase.startswith(("ins_temp_", "aicpu_temp_")):
        rv.add("W04", MINOR, "文件名未以 ins_temp_ 开头: %s" % f.filebase)
    if not cls.startswith(("InsTemp", "AicpuTemp")):
        rv.add("W04", MINOR, "类名未以 InsTemp 开头: %s" % cls)
    guards = re.findall(r"#ifndef\s+(\w+)", hh_raw)
    if not guards:
        rv.add("W04", BLOCKER, ".h 缺少 include guard")
    elif guards[0] != f.filebase.upper() + "_H":
        rv.add(
            "W04",
            MINOR,
            "include guard 为 %s，建议 %s_H" % (guards[0], f.filebase.upper()),
        )
    for name, text in (("h", hh_raw), ("cc", cc_raw)):
        if "CANN Open Software License Agreement" not in text[:1500]:
            rv.add("W04", MAJOR, ".%s 缺少 CANN Open Software License 版权头" % name)
        if "namespace ops_hccl" not in text:
            rv.add("W04", MAJOR, ".%s 缺少 namespace ops_hccl" % name)

    # ---- W05 构造函数 ----
    if not re.search(r"const\s+OpParam&\s*\w+,\s*const\s+u32\s+\w+", hh_raw, re.S):
        rv.add(
            "W05",
            MAJOR,
            "带参构造签名与 executor 的调用约定不符",
            hint="executor 硬编码 make_shared<T>(param, rankId, subCommRanks)，\n"
            "签名必须是 (const OpParam&, const u32, const std::vector<std::vector<u32>>&)。",
        )
    if not re.search(r"%s\(\)\s*=\s*default" % re.escape(cls), hh_raw):
        rv.add(
            "W05",
            MINOR,
            "没有默认构造 `%s() = default;`" % cls,
            hint="走 FastLaunch 的 executor 用 make_unique<T>() 无参构造；\n"
            "DPU / Intra / Inter / OmniPipe 这类不走 FastLaunch 的可以没有。",
        )

    # ---- W06 / W07 注册 ----
    is_dpu = "dpu" in f.filebase.lower() or "DPU" in cls.upper()
    has_reg = "REGISTER_TEMPLATE_V2" in cc_raw
    if is_dpu and not has_reg:
        rv.add(
            "W07",
            MAJOR,
            "DPU template 缺少 REGISTER_TEMPLATE_V2 注册",
            hint="dpu/kernel_launch.cc 按字符串反查，需在 .cc 末尾加：\n"
            'REGISTER_TEMPLATE_V2("%s", %s);' % (cls, cls),
        )
    elif not is_dpu and has_reg:
        rv.add("W07", MINOR, "非 DPU template 写了 REGISTER_TEMPLATE_V2，通常没必要")
    if "REGISTER_TEMPLATE(" in cc_raw:
        rv.add(
            "W07",
            BLOCKER,
            "使用了 V1 的 REGISTER_TEMPLATE（枚举 key）",
            hint="V2 用 REGISTER_TEMPLATE_V2，且 AICPU 侧一般不需要注册。",
        )
    if f.op and f.in_repo_tree and not is_dpu:
        hits = [
            os.path.relpath(pp, f.repo)
            for pp in glob.glob(f.layout.executor_glob(f.repo))
            if re.search(
                r"REGISTER_EXEC_V2[^;]*\b%s\b" % re.escape(cls), read(pp), re.S
            )
        ]
        if not hits:
            rv.add(
                "W06",
                MINOR,
                "没有任何 REGISTER_EXEC_V2 引用这个类",
                hint="template 要被 executor 当模板参数注册后才会真正跑起来；\n"
                "新写的 template 如果 executor 还没接，这条先记着。",
            )


# --------------------------------------------------------------------------- 通用检查
def check_common(f, rv):
    cc = f.cc

    # ---- C01 / C02 scratch 倍数 ----
    if f.scratch_body is None and f.is_variant:
        rv.add(
            "C02",
            MINOR,
            "CalcScratchMultiple 继承自 %s，没有在本类覆写" % f.parent,
            hint="变体如果改了写进 hcclBuff 的最大字节数（换了收发模式、加了一段暂存），\n"
            "父类的倍数就不再成立。确认没改再放过。",
        )
    elif f.scratch_body is None:
        rv.add(
            "C02",
            MAJOR,
            "没有 CalcScratchMultiple 的实现，executor 无法算每轮数据量",
            hint="基类默认实现直接返回失败。不占 scratch 就显式 return 0。",
        )
    else:
        if re.search(r"\brepeatNum_?\b", f.scratch_body):
            # MAJOR 而非 BLOCKER：后果是确定性的性能退化（loop 翻倍、带宽减半），
            # 结果仍然算得对，不符合 BLOCKER"必然错或必然不生效"的定义。
            rv.add(
                "C01",
                MAJOR,
                "CalcScratchMultiple 里乘了 repeatNum，倍数被双重放大",
                line=f.scratch_line,
                evidence=f.scratch_expr,
                hint="倍数按单次 repeat 报；放大是 executor 的活\n"
                "（max(另一层, 本层 * rankSizeLevelX)）。自己再乘一遍 → loop 次数翻倍、带宽白掉一半，\n"
                "编译和 ST 都不报错，只能靠 review 抓。",
            )
        # 期望值按「拓扑 + 变体」查表，再按算子分档：
        #   AllReduce（in/out 等大）有唯一正解 → 严格集合 + MAJOR；
        #   AllGather / ReduceScatter / Scatter（in/out 不等大）N 倍往往是对的 → 放宽 + MINOR 人工确认。
        expect = {
            ("mesh", "one-shot"): (
                ["templateRankSize_"],
                ["templateRankSize_"],
                "M01",
                "Mesh one-shot 每个 rank 在我 scratch 独占一段，共 N 段",
            ),
            ("mesh", "two-shot"): (
                ["1", "2"],
                ["1", "2", "templateRankSize_"],
                "M02",
                "Mesh two-shot 理论 1 份够，非均衡切分留余量取 2",
            ),
            ("nhr", None): (
                ["1", "0"],
                ["1", "0", "templateRankSize_"],
                "N01",
                "NHR 原地跑在 scratch 上只需 1 份",
            ),
            ("ring", None): (
                ["1", "0"],
                ["1", "0", "templateRankSize_"],
                "R05",
                "AllReduce ring 在 scratch 上原地累加只需 1 份；"
                "AllGather / Scatter 类 ring 的中转副本要占满 N*S",
            ),
            ("hd", None): (
                ["1", "0"],
                ["1", "0", "templateRankSize_"],
                "H06",
                "AllReduce HD 原地折半累加只需 1 份",
            ),
        }
        key = (f.topo, f.variant) if (f.topo, f.variant) in expect else (f.topo, None)
        if (
            key in expect
            and f.scratch_expr
            and not (f.topo == "ring" and ring_shape(f) != "rotating")
        ):
            strict, lax, rule, why = expect[key]
            wanted, sev = (strict, MAJOR) if f.is_allreduce_semantics else (lax, MINOR)
            if rule in UNVERIFIED_MAPPING:
                sev = INFO
            if not scratch_matches(f.scratch_expr, wanted):
                rv.add(
                    rule,
                    sev,
                    "scratch 倍数 `%s` 与 %s%s 的预期不符（应是 %s）"
                    % (
                        f.scratch_expr,
                        f.topo,
                        "/" + f.variant if f.variant else "",
                        " 或 ".join(wanted),
                    ),
                    line=f.scratch_line,
                    hint="%s。\n报小 → hcclBuff 越界；报大 → 白切 loop。确有理由请在 spec 第 4 章写明。"
                    % why,
                )

    # ---- C03 / C04 repeat 与 stride ----
    if not has_repeat_loop(cc):
        rv.add(
            "C03",
            MINOR,
            "KernelRun 里没有以 repeatNum 为上界的循环",
            line=f.kernel_line,
            hint="RPT==1 时循环零代价退化；漏写则这个 template 一旦被 2/3 级 executor 当构件复用，\n"
            "只会处理第 0 块数据。确定只服务单层且不打算复用，请在 spec 第 5 章写明理由。",
        )
    elif "hcclBuff" in cc and not re.search(r"\binBuffType\b", cc):
        rv.add(
            "C04",
            MINOR,
            "用了 repeat 循环并落 scratch 地址，但没有区分 inBuffType == HCCL_BUFFER",
            hint="scratch 的 repeat 跨度默认自己算 S*N；只有输入本来就在 CCL buffer 里\n"
            "（intra 接在 inter 之后）时才改用入参的 inputRepeatStride / inputSliceStride。\n"
            "蓝本：ins_temp_all_gather_mesh_1D.cc:152-170。",
        )

    # ---- C16 输出落位；脚本只能提示缺少分流，不能证明跳过拷贝是安全的 ----
    if not re.search(r"Dpu|OmniPipe", f.cls, re.I) and output_copy_without_guard(
        f.scan_cc, f.cls
    ):
        rv.add(
            "C16",
            MINOR,
            "scratch/输出间有 LocalCopy，但没有读取 buffInfo.outBuffType",
            hint="逐段核对绑定 executor 的输出落位与 base offset；HCCL_BUFFER 不能无条件跳过，\n"
            "须证明数据已在目标槽位，否则仍需搬移。分流和理由需人工核对。",
        )

    # ---- C05 / C06 边界早退 ----
    if not re.search(r"\b(count|sliceSize|tailSize)\w*\s*==\s*0\b", f.scan_cc):
        rv.add(
            "C05",
            MINOR,
            "找不到 count == 0 的早退",
            line=f.kernel_line,
            hint="空数据仍会走完全部 send/recv 编排。存量里大多数也没写（靠 executor 保证 count>0），\n"
            "所以只报 MINOR；新写的 template 请补上（判据见 references/02 §5）。",
        )
    if not re.search(
        r"(templateRankSize_|\w*[Rr]ankSize\w*)\s*(==\s*1|>\s*1)|\.size\(\)\s*(==|>)\s*1",
        f.scan_cc,
    ):
        rv.add(
            "C06",
            MINOR,
            "找不到单 rank（rankSize == 1）早退",
            hint="单 rank 时只该做 LocalCopy；继续往下走会去取不存在的 channel。",
        )

    # ---- C07 同步配对 ----
    pre = f.count_calls("PreSyncInterThreads")
    post = f.count_calls("PostSyncInterThreads")
    if pre != post:
        one_side_missing = (pre == 0) != (post == 0)
        rv.add(
            "C07",
            MAJOR if one_side_missing else MINOR,
            "PreSyncInterThreads(%d 次) 与 PostSyncInterThreads(%d 次) 不成对"
            % (pre, post),
            hint="主→从 与 从→主 必须成对；少一半 → 主流可能在从流写完前就去归约，读到半成品。\n"
            "存量里也有故意不对称的用法（如 post-copy 线程间的单向握手），\n"
            "两边都非 0 时按 MINOR 报，请人工确认每一处的配对关系。",
        )
    elif pre > 0:
        if not f.has(r"GetNotifyIdxMainToSub") or not f.has(r"GetNotifyIdxSubToMain"):
            rv.add(
                "C07",
                MINOR,
                "调了 Pre/PostSyncInterThreads 但没取对应的 notifyIdx",
                hint="PreSync 前要 GetNotifyIdxMainToSub，PostSync 前要 GetNotifyIdxSubToMain。",
            )

    # ---- C08 资源三字段 ----
    res_scope = f.res_body + "\n" + (f.fam_res or "")
    fields = {
        name: bool(
            re.search(
                r"\b%s\s*(=|\.assign|\.push_back|\.resize|\.emplace)" % name, res_scope
            )
        )
        for name in ("slaveThreadNum", "notifyNumPerThread", "notifyNumOnMainThread")
    }
    single_thread = re.search(r"slaveThreadNum\s*=\s*0\s*;", res_scope) is not None
    if any(fields.values()) and not all(fields.values()) and not single_thread:
        miss = [k for k, v in fields.items() if not v]
        rv.add(
            "C08",
            MAJOR,
            "CalcRes/GetRes 里缺少 %s" % "、".join(miss),
            hint="三个字段要一起填：slaveThreadNum = 线程数-1；notifyNumPerThread 每从流 1 个；\n"
            "notifyNumOnMainThread = 从流数。少填 → notify 索引越界或死等。",
        )

    # ---- C09 / C10 软归约兜底 ----
    if f.does_reduce:
        if not re.search(
            r"HCCL_DATA_TYPE_INT64|needAicpuReduce|HCCL_REDUCE_PROD", f.scan_all
        ):
            rv.add(
                "C09",
                MINOR,
                "做归约却看不到 64-bit / PROD 的兜底判据，确认由谁兜底",
                hint="硬件归约不支持 INT64 / UINT64 / FP64 / PROD。仓内两种兜底方式：\n"
                "  a) template 内部：needAicpuReduce_ = 64-bit 或 PROD，跑完搬运再软归约（见 mesh one-shot）；\n"
                "  b) 选路层换到专门的 *_aicpu_reduce_* template。\n"
                "两条都没有 = INT64 / PROD 直接算错。请作者答复走哪条。",
            )
        pos = {
            n: cc.find(n)
            for n in ("HcommBatchModeEnd", "HcommBatchModeStart", "HcommThreadJoin")
        }
        if all(v != -1 for v in pos.values()):
            if not (
                pos["HcommBatchModeEnd"]
                < pos["HcommBatchModeStart"]
                < pos["HcommThreadJoin"]
            ):
                rv.add(
                    "C10",
                    MAJOR,
                    "软归约三步顺序不对（当前 %s）"
                    % " → ".join(sorted(pos, key=lambda k: pos[k])),
                    line=lineno(cc, min(pos.values())),
                    hint="必须是 BatchModeEnd → BatchModeStart → ThreadJoin，\n"
                    "顺序错会在从流还没落盘时就去读 scratch。",
                )

    # ---- C11 / C15 channel 取用 ----
    ats = [
        (ln, m.group(1), m.group(2).strip())
        for ln, m in f.find(
            r"\b(\w*[Cc]hannels?\w*)\.at\(([^()]*(?:\([^()]*\)[^()]*)*)\)"
        )
    ]
    counts = [
        m.group(2).strip()
        for _, m in f.find(
            r"\b(\w*[Cc]hannels?\w*)\.count\(([^()]*(?:\([^()]*\)[^()]*)*)\)"
        )
    ]
    if ats and not counts:
        rv.add(
            "C11",
            MINOR,
            "全文有 %d 处 channels.at() 却没有任何 channels.count() 校验" % len(ats),
            line=ats[0][0],
            hint="远端 rank 不在 channels 里时 .at() 抛异常，AICPU 上是直接崩：\n"
            'CHK_PRT_RET(channels.count(p) == 0, HCCL_ERROR("..."), HCCL_E_INTERNAL);',
        )
    elif ats:
        unchecked = [
            (ln, a)
            for ln, _, a in ats
            if a and not any(a in c for c in counts)
        ]
        for ln, a in unchecked[:3]:
            rv.add(
                "C11",
                MINOR,
                "channels.at(%s) 的 key 没有出现在任何 count() 校验里" % a,
                line=ln,
            )
    for ln, var, arg in ats[:6]:
        if (
            re.match(r"^[A-Za-z_]\w*$", arg)
            and not re.search(
                r"\b%s\s*=[^;]*(subCommRanks_|rankList_)" % re.escape(arg), cc
            )
            and arg
            not in ("peer", "peerRank", "remoteRank", "dstRank", "srcRank", "userRank")
        ):
            rv.add(
                "C15",
                MINOR,
                "channels.at(%s) 的 key 看不出是从 subCommRanks_/rankList_ 取的" % arg,
                line=ln,
                hint="channels 的 key 是通信域 userRank，不是算法内序号 0..N-1。混用是本仓高频 bug。",
            )
            break

    # ---- C12 CHK_RET 包裹 ----
    unwrapped = []
    for prim in STATUS_PRIMS:
        for ln, m in f.find(r"(?<![\w:])%s\s*\(" % prim):
            head = cc.rfind(";", 0, m.start())
            head = max(head, cc.rfind("{", 0, m.start()), cc.rfind("}", 0, m.start()))
            stmt = cc[head + 1:m.start()]
            if not re.search(r"CHK_RET|CHK_PRT_RET|=|return|\bif\b", stmt):
                unwrapped.append((ln, prim))
    for ln, prim in unwrapped[:5]:
        rv.add(
            "C12",
            MINOR,
            "%s() 的返回值没被 CHK_RET / CHK_PRT_RET 接住" % prim,
            line=ln,
            hint="失败会被静默吞掉，现象是结果错但没有任何报错。",
        )

    # ---- C13 Describe ----
    if f.desc_body is not None and "templateRankSize_" not in f.desc_body:
        rv.add(
            "C13",
            MINOR,
            "Describe() 没带 templateRankSize_",
            hint="它出现在日志和 DFX 里，不带 rankSize 排障时分不清是哪一层的实例。",
        )

    # ---- C14 读写模式混用 ----
    has_w = any(p in cc for p in WRITE_PRIMS)
    has_r = any(p in cc for p in READ_PRIMS)
    if has_w and has_r and not re.search(r"IsPcieProtocol|isDmaRead", cc):
        rv.add(
            "C14",
            MINOR,
            "同时用了写模式与读模式原语，却看不到 PCIe 判据",
            hint="读/写模式二选一：写模式底层只用 txSlicesList_，读模式只用 rxSlicesList_。\n"
            "两条路径都要有的话，用 isDmaRead_ = IsPcieProtocol(channels) 做分支。",
        )


# --------------------------------------------------------------------------- Mesh
def check_mesh(f, rv):
    cc = f.cc
    slave_loop = re.search(
        r"for\s*\([^)]*(queIdx|threadIdx|slaveIdx|i)\s*=\s*1\s*;", cc
    )
    if slave_loop and not re.search(r"\(\s*(myRank_|myRankIdx_)\s*\+[^)]*\)\s*%", cc):
        rv.add(
            "M03",
            MINOR,
            "从流循环里没看到 (myRank_ + i) % N 的错峰",
            line=lineno(cc, slave_loop.start()),
            hint="Mesh 下所有 rank 同时打同一个目标会把链路打爆；用错峰把对端顺序转一圈。",
        )
    m = re.search(r"slaveThreadNum\s*=\s*([^;]+);", f.res_body)
    if m and not re.search(r"templateRankSize_|channelsPerRank_|threadNum", m.group(1)):
        rv.add(
            "M04",
            MINOR,
            "slaveThreadNum = %s 不是从 templateRankSize_ / channelsPerRank_ 推导的"
            % m.group(1).strip(),
            hint="写死线程数在别的 rankSize 规格上就错了。",
        )
    if (
        f.variant == "one-shot"
        and f.is_reduce_op
        and not re.search(r"\bLocalReduce\s*\(", f.scan_cc)
    ):
        rv.add(
            "M05",
            MAJOR,
            "one-shot 归约类算子没有收尾 LocalReduce",
            hint="one-shot 是「全量交换 + 本地归约」；只交换不归约，输出里只有自己那份。",
        )
    if re.search(r"stepInfoList|GetNHRStep", cc):
        rv.add(
            "M04",
            MINOR,
            "Mesh 却出现了多 step 的编排结构",
            hint="Mesh 是一步到位；出现 step 列表通常是从 NHR 蓝本抄来没删干净。",
        )


# --------------------------------------------------------------------------- NHR
def check_nhr(f, rv):
    cc = f.cc
    if f.count_calls("LocalCopy", f.scan_cc) < 2 and not re.search(
        r"PreCopy|PostCopy", f.scan_cc
    ):
        rv.add(
            "N02",
            MINOR,
            "看不到 PreCopy(in→scratch) / PostCopy(scratch→out) 这一对",
            hint="NHR 原地跑在 hcclBuff 上：进场要把 input 拷进 scratch，退场要把结果拷回 output。",
        )
    lit = re.search(r"for\s*\([^;]*step[^;]*;\s*\w+\s*<\s*(\d+)\s*;", cc)
    if lit:
        rv.add(
            "N03",
            MAJOR,
            "step 数写死成 %s" % lit.group(1),
            line=lineno(cc, lit.start()),
            hint="NHR 的 step 数是 GetNHRStepNum(rankSize) 的函数，写死只在一种规格上对。",
        )
    elif not re.search(r"GetNHRStep|stepInfoList|stepNum|StepInfo", cc):
        rv.add(
            "N03",
            MINOR,
            "看不到 step 列表的构造（GetNHRStepNum / stepInfoList）",
            hint="NHR 的每一步「和谁换哪几片」应该由一张 step 表描述，别散在主循环里。",
        )
    if re.search(r"\bchannelsPerRank_\b", f.all):
        if not re.search(r"\bGetThreadNum\s*\(", f.scan_all) or not re.search(
            r"\bGetRes\s*\(", f.scan_all
        ):
            rv.add(
                "N04",
                MAJOR,
                "用了 channelsPerRank_（多 jetty）却没同时覆写 GetRes + GetThreadNum",
                hint="线程数由 channel 数推导时，框架要靠 GetRes 申请资源、靠 GetThreadNum 知道线程数；\n"
                "少一个就会申请到 rankSize 个线程再去跑 channelsPerRank_ 个 step。",
            )
    if (
        re.search(r"stepInfoList|StepInfo", cc)
        and len(re.findall(r"\.count\(", cc)) < 2
    ):
        rv.add(
            "N05",
            INFO,
            "人工确认：每 step 取 channel 前 from/to 两侧都要校验",
            hint="NHR 每步收发对端不同，两个都要 channels.count() 过一遍。",
        )


def algo_gap(f, rv, algo_types, family, name_keys):
    """X01：逐项核对选路侧四个登记点。缺任意一项，这个算法都选不上。"""
    algo_names, type_to_name, name_to_type = load_algo_registrations(f.repo)
    missing = []

    def matches(value):
        return family.lower() in value.lower() or any(
            k.lower() in value.lower() for k in name_keys
        )

    if algo_types and not any(matches(a) for a in algo_types):
        missing.append("src/common/alg_parse.h 的 `enum class AlgoType`")
    if not any(matches(name) for name in algo_names):
        missing.append(
            "src/common/alg_parse.cc 的 `ALGO_TYPES`（小写算法名 → 驼峰名）——\n"
            "     HcclAlgoParser 查不到就直接 `invalid algoType` / HCCL_E_PARA"
        )
    if not any(matches(name) for name in type_to_name):
        missing.append(
            "src/common/alg_parse.cc 的 `GetAlgoTypeToNameMap()`——\n"
            "     缺 `GetAlgoTypeToNameMap()` 中的枚举 → 名称映射"
        )
    if not any(matches(name) for name in name_to_type):
        missing.append(
            "src/common/alg_parse.cc 的 `GetAlgoNameToTypeMap()`——\n"
            "     缺名称 → 枚举映射"
        )
    if missing:
        rv.add(
            "X01",
            BLOCKER,
            "%s 族在选路侧登记不完整（缺 %d 处）" % (family, len(missing)),
            hint="要让这个 template 真的被选上，只补下面实际缺失的登记点：\n  - "
            + "\n  - ".join(missing),
        )


# --------------------------------------------------------------------------- Ring（按算子语义分派适用规则）
def check_ring(f, rv, algo_types):
    cc = f.cc
    algo_gap(f, rv, algo_types, "RING", ("ring",))
    full_loop = re.search(
        r"for\s*\([^)]*<\s*templateRankSize_\s*;[^{]*\{[^}]{0,400}?\bchannels?\w*\.at\(",
        cc,
        re.S,
    )
    if full_loop:
        rv.add(
            "R01",
            MAJOR,
            "对全体 rank 遍历取 channel",
            line=lineno(cc, full_loop.start()),
            hint="Ring 每个 rank 只和左右邻居通信：next = (idx + 1) % N，prev = (idx - 1 + N) % N。\n"
            "对全体建链是 Mesh 的形态，会白申请 N-2 条用不到的链路。",
        )
    _rs_pattern = r"(?:templateRankSize_|\w*[Rr]ankSize\w*)"
    has_next = re.search(r"\+\s*1\s*\)\s*%\s*" + _rs_pattern, cc)
    # prev 有两种等价写法：(idx - 1 + N) % N 与 (idx + N - 1) % N
    has_prev = re.search(
        r"(?:-\s*1\s*\+\s*" + _rs_pattern + r"|\+\s*" + _rs_pattern + r"\s*-\s*1)\s*\)\s*%", cc
    )
    # 读/写原语可在互斥模式分支里出现，不能据此断言实现是双向环。
    if not has_next and not has_prev:
        rv.add(
            "R01",
            MAJOR,
            "看不到任何相邻对端 (idx±1) % N 的推导",
            hint="核对 next/prev 是否由辅助函数或等价表达式推导；缺少语法特征不是错误证明。",
        )
    if (f.op_by_name or f.op) == "scatter" and re.search(r"\bLocalCopy\s*\(", cc):
        rv.add(
            "R09",
            INFO,
            "人工确认：链式 relay 收下即留的拷贝是否按输出落位处理",
            hint="先查各绑定 executor 的 outBuffType、outputPtr、outBuffBaseOff 与 hcclBuffBaseOff。\n"
            "HCCL_BUFFER 段只有在上一跳已直写目标槽位时才能跳过 LocalCopy；\n"
            "错位时仍需搬移，分流处要写明在位/错位依据。见 references/05 §8。",
        )
    shape = ring_shape(f)
    if shape != "rotating":
        rv.add(
            "R08",
            INFO,
            "人工确认：%s Ring 的形态与端点/收发依赖"
            % ("Broadcast" if shape == "broadcast" else "未识别算子语义的"),
            hint="未应用轮转分片的 R02–R07；这不代表这些性质已通过。\n"
            "链式广播按每次 repeat/分块核对：root 不接收、末端不转发、中间节点接收完成后转发；\n"
            "全局覆盖全部 rank、共 N-1 跳，不要求每个 rank 执行 N-1 次循环。\n"
            "scratch、线程、同步和尾块按实际模式核对，见 references/05 §7。",
        )
        return
    bounds = [
        (lineno(cc, m.start()), m.group(2).strip())
        for m in re.finditer(
            r"for\s*\([^;]*\b(\w*[Ss]tep\w*)\b[^;]*;\s*\1\s*<\s*([^;)]+)", cc
        )
    ]
    if not bounds:
        rv.add(
            "R02",
            MAJOR,
            "找不到 step 循环，看不出总步数",
            hint="单向 ring（AllGather / ReduceScatter / Scatter）：N-1 步一段；\n"
            "AllReduce ring：RS + AG 两段，共 2(N-1) 步，两段循环都要有。",
        )
    for ln, b in bounds:
        resolved = resolve_expr(
            cc, b
        )  # 上界常写成 const u32 nSteps = templateRankSize_ - 1;
        if re.match(r"^\d+$", resolved):
            rv.add(
                "R02",
                MAJOR,
                "step 循环上界写死成 %s" % resolved,
                line=ln,
                hint="步数必须是 rankSize 的函数（每段 N-1 步），写死只在一种规格上对。",
            )
        elif not re.search(r"[Rr]ankSize", resolved):
            rv.add(
                "R02",
                MAJOR,
                "step 循环上界 `%s`（解析为 `%s`）与 rankSize 无关" % (b, resolved),
                line=ln,
                hint="单向 ring（AllGather / ReduceScatter / Scatter）是 N-1 步；\n"
                "AllReduce ring 是 RS + AG 两段，共 2(N-1) 步。",
            )
    if not re.search(
        r"-\s*\w*[Ss]tep\w*[^;]*%\s*(templateRankSize_|\w*[Rr]ankSize)", cc
    ):
        rv.add(
            "R03",
            MAJOR,
            "看不到每步 chunk 索引 (idx - step + N) % N 的推导",
            hint="RS 阶段第 k 步发 (idx-k+N)%N 号片、收 (idx-k-1+N)%N 号片；\n"
            "AG 阶段整体后移一位。索引算错可能结果错误但不崩，需按调用工作流进行功能验证。",
        )
    step_loop = re.search(r"for\s*\([^)]*step", cc)
    if (
        step_loop
        and f.count_calls("PreSyncInterThreads") + f.count_calls("PostSyncInterThreads")
        == 0
        and not re.search(r"Notify|notify", cc)
    ):
        rv.add(
            "R04",
            MAJOR,
            "step 之间没有任何同步",
            line=lineno(cc, step_loop.start()),
            hint="Ring 是串行流水：第 k 步收到的片是第 k+1 步要发的内容。\n"
            "缺同步就会把上一步还没落盘的数据发出去。",
        )
    m = re.search(r"slaveThreadNum\s*=\s*([^;]+);", f.res_body)
    if m and "templateRankSize_" in m.group(1):
        rv.add(
            "R06",
            INFO,
            "人工确认：slaveThreadNum = %s，像是照搬了 Mesh 的线程模型"
            % m.group(1).strip(),
            hint="Ring 每个 rank 同时只有一收一发，线程数通常与 rankSize 无关（1 主 + 1 从）。\n"
            "但「本仓的 ring 该用几个线程」没有文献判据、仓内也无已验证实现，\n"
            "所以只提示、不阻塞——请人工确认这里的线程模型是有意为之。",
        )
    rv.add(
        "R07",
        INFO,
        "人工确认：各阶段共用同一张分片表（尾片归属一致）",
        hint="C 不能被 N 整除时，尾片的长度与偏移必须只算一次：\n"
        "单向 ring 里 LocalCopy / 每个 step / 收尾拷回要用同一张表；\n"
        "AllReduce ring 里 RS 与 AG 两段也要共用——各算一次是经典错法，AG 会把尾片错位一格。",
    )


# --------------------------------------------------------------------------- HD（仓内无存量实现）
def check_hd(f, rv, algo_types):
    cc = f.cc
    algo_gap(f, rv, algo_types, "HD", ("hd", "halvingdoubling"))
    if not re.search(
        r"part1|blockSize|IsPowerOfTwo|PowerOfTwo|1\s*<<|&\s*\(\s*\w+\s*-\s*1\s*\)", cc
    ):
        # MAJOR 而非 BLOCKER：判据本身有 MPICH / Rabenseifner 背书，但这里的检测靠
        # part1 / blockSize 这类字面命名，换个命名就会误报（实测可复现），
        # 够不上 BLOCKER 要求的"证据确凿"。
        rv.add(
            "H01",
            MAJOR,
            "看不到非 2 的幂 rankSize 的处理",
            hint="标准做法：先把 rankSize 降到不大于它的最大 2 的幂 blockSize；\n"
            "多出来的 part1Size = 2*(N - blockSize) 个 rank 里，偶数 rank 先把数据发给 rank+1，\n"
            "主循环只在 blockSize 个 rank 上跑，末尾再回传。缺这一段 → 非 2 幂卡数直接算错。",
        )
    elif not re.search(r"part1|blockSize", cc):
        rv.add(
            "H05",
            MAJOR,
            "有 2 的幂判断但看不到 part1 / blockSize 的淘汰与回传",
            hint="被淘汰的 rank 必须排除在主循环外，并在末尾把结果回传给它。",
        )
    if not re.search(r"log2|<<|>>", cc):
        rv.add(
            "H02",
            MAJOR,
            "步数不是从 log2(blockSize) 推导",
            hint="halving 阶段 log2(blockSize) 步，doubling 阶段同样步数（逆序）。",
        )
    if not re.search(r"\^|xor|1\s*<<\s*\w+", cc):
        rv.add(
            "H03",
            MAJOR,
            "看不到每步对端的推导（XOR / 距离折半）",
            hint="doubling 阶段第 k 步对端 = idx ^ (1 << k)；halving 阶段是它的逆序。",
        )
    if not re.search(r"/\s*2\b|>>\s*1\b|\*\s*2\b|<<\s*1\b", cc):
        rv.add(
            "H04",
            MINOR,
            "每步的数据量看不到折半 / 翻倍",
            hint="halving 阶段每步交换的数据量减半，doubling 阶段每步翻倍；\n"
            "整段用同一个 size 说明写成了 recursive doubling（allgather 语义），通信量翻 log N 倍。",
        )


# --------------------------------------------------------------------------- 与 dataflow spec 对照
def table_value(text, key):
    m = re.search(r"^\|[`*\s]*%s[`*\s]*\|([^|]*)\|" % re.escape(key), text, re.M)
    return m.group(1).strip() if m else None


def markdown_headings(text):
    """Yield ATX headings outside fenced blocks, retaining original body offsets.

    Standalone copy also in the template skill's check_spec.py; keep behavior
    aligned via fence/section regression tests, without a cross-skill import.
    """
    fence = None
    offset = 0
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*?)[\r\n]*$", line)
        if fence:
            if (
                marker
                and marker.group(1)[0] == fence[0]
                and len(marker.group(1)) >= fence[1]
                and not marker.group(2).strip()
            ):
                fence = None
        elif marker and not (marker.group(1)[0] == "`" and "`" in marker.group(2)):
            fence = (marker.group(1)[0], len(marker.group(1)))
        else:
            heading = re.match(r"^ {0,3}(#{1,6})(?:[ \t]+|$)(.*?)[\r\n]*$", line)
            if heading:
                yield (
                    offset,
                    offset + len(line),
                    len(heading.group(1)),
                    heading.group(2),
                )
        offset += len(line)


def markdown_section(text, number, title_pattern):
    """取章节正文；围栏内的 # 不作为标题，围栏正文仍参与同步计数。"""
    headings = list(markdown_headings(text))
    for index, (_, start, level, title) in enumerate(headings):
        if not re.search(
            r"^%d(?:\.|、)?\s+.*%s" % (number, title_pattern), title, re.I
        ):
            continue
        end = next(
            (pos for pos, _, depth, _ in headings[index + 1:] if depth <= level),
            len(text),
        )
        return text[start:end]
    return None


def spec_sync_counts(section):
    """Count action statements in fenced dataflow, never mentions in prose.

    This is a notation check, not a control-flow proof. Ambiguous action syntax
    and missing/unclosed fences must remain unverified rather than count as zero.
    """
    fence, blocks, body = None, [], []
    for line in section.splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*?)$", line)
        if fence:
            if (
                marker
                and marker.group(1)[0] == fence[0]
                and len(marker.group(1)) >= fence[1]
                and not marker.group(2).strip()
            ):
                blocks.append("\n".join(body))
                fence, body = None, []
            else:
                body.append(line)
        elif marker and not (marker.group(1)[0] == "`" and "`" in marker.group(2)):
            fence = (marker.group(1)[0], len(marker.group(1)))
    if fence or not blocks or not any(block.strip() for block in blocks):
        raise ValueError("第 6 章须有非空且闭合的数据流代码围栏")
    counts = {"main": 0, "sub": 0}
    for block in blocks:
        block = re.sub(r"/\*.*?\*/|<!--.*?-->", "", block, flags=re.S)
        for line in block.splitlines():
            action = re.split(r"#|//", line, maxsplit=1)[0].strip()
            match = re.fullmatch(r"sync\s+(main|sub)\s*->\s*(main|sub)\s*;?", action)
            if match and match.group(1) != match.group(2):
                counts[match.group(1)] += 1
            elif re.search(r"\bsync\s+(?:main|sub)\b", action):
                raise ValueError("同步动作须独占一行，条件写在外层：" + action)
    return counts["main"], counts["sub"]


def check_spec(f, rv, spec_path):
    spec = read(spec_path)
    sec4 = markdown_section(spec, 4, r"Buffer\s*布局")
    sec5 = markdown_section(spec, 5, "切片")
    sec6 = markdown_section(spec, 6, "数据流")
    missing = []
    for name, body in (
        ("第 4 章 Buffer 布局", sec4),
        ("第 5 章 切片", sec5),
        ("第 6 章 数据流", sec6),
    ):
        if body is None:
            missing.append(name)
    mult = table_value(sec4, "scratch 倍数") if sec4 is not None else None
    rpt = table_value(sec5, "RPT") if sec5 is not None else None
    if not mult:
        missing.append("第 4 章的 `scratch 倍数` 表格行")
    if not rpt:
        missing.append("第 5 章的 `RPT` 表格行")
    tbds = [
        line.strip()
        for line in spec.splitlines()
        if re.search(r"\bTBD\s*:", line, re.I)
    ]
    sync_counts = None
    if sec6 is not None:
        try:
            sync_counts = spec_sync_counts(sec6)
        except ValueError as exc:
            missing.append(str(exc))
    if missing or tbds:
        detail = []
        if missing:
            detail.append("缺少：" + "、".join(missing))
        if tbds:
            detail.append("仍有 TBD：" + "；".join(tbds[:3]))
        rv.add(
            "S00",
            MAJOR,
            "dataflow spec 无法可靠解析",
            hint="；".join(detail)
            + "。\n先补全 spec，再判断 S01–S03；本次不把静默跳过当作通过。",
        )
        return
    if mult and f.scratch_expr:
        norm = re.sub(r"[`*\s]", "", mult)
        code = re.sub(r"[`*\s]", "", f.scratch_expr)
        alias = {"N": "templateRankSize_"}
        norm_alias = alias.get(norm.split("（")[0], norm.split("（")[0])
        if norm_alias not in code and code not in norm:
            rv.add(
                "S01",
                MAJOR,
                "spec 第 4 章写 scratch 倍数 = %s，代码返回 %s"
                % (mult, f.scratch_expr),
                line=f.scratch_line,
            )
    if rpt:
        declared_multi = not re.match(r"^[`*\s]*1[`*\s]*$", rpt)
        has_loop = has_repeat_loop(f.cc)
        if declared_multi and not has_loop:
            rv.add(
                "S02",
                BLOCKER,
                "spec 声明 RPT = %s（非 1），代码里却没有 repeat 循环" % rpt,
            )
        elif not declared_multi and not has_loop:
            rv.add(
                "S02",
                INFO,
                "spec 与代码一致：RPT = 1 且没有 repeat 循环（不可被分层复用）",
            )
    spec_pre, spec_post = sync_counts
    code_pre = f.count_calls("PreSyncInterThreads")
    code_post = f.count_calls("PostSyncInterThreads")
    if (spec_pre, spec_post) != (code_pre, code_post):
        rv.add(
            "S03",
            MAJOR,
            "spec 同步点 main→sub / sub→main 为 %d / %d，代码 PreSync / PostSync 为 %d / %d"
            % (spec_pre, spec_post, code_pre, code_post),
            hint="静态出现次数不一致，请核对分支、循环及 spec 是否过期；数量相同也不能证明作用域一致。",
        )


# --------------------------------------------------------------------------- 驱动
def review_one(
    repo, cc_path, spec_path, algo_types, layout, force_topo=None, semantic_only=False
):
    base = cc_path[:-3] if cc_path.endswith(".cc") else cc_path
    cc_path, h_path = base + ".cc", base + ".h"
    rel = os.path.relpath(cc_path, repo)
    if not os.path.exists(cc_path) or not os.path.exists(h_path):
        rv = Review(rel, "-", "")
        rv.add("C02", BLOCKER, "找不到配对的 .cc / .h")
        return rv
    cc_raw, hh_raw = read(cc_path), read(h_path)
    m = re.search(r"class\s+(\w+)\s*:\s*public\s+(\w+)", hh_raw)
    if not m:
        rv = Review(rel, "-", "")
        rv.add("C02", BLOCKER, ".h 里找不到 `class X : public Y` 定义，无法评审")
        return rv
    cls = m.group(1)
    cc, hh = strip_comments(cc_raw), strip_comments(hh_raw)
    f = Facts(repo, cc_path, cls, m.group(2), cc, hh, cc_raw, hh_raw, layout)
    if force_topo:
        f.topo = force_topo
    rv = Review("%s  (%s)" % (rel, cls), f.topo, f.variant)

    if not semantic_only:
        if not f.in_repo_tree:
            # 仓内文件却不在受支持布局的 template 目录下 → 布局漂了，接线检查静默跳过是危险的。
            # 仓外文件（补丁、夹具）本就没有接线可查，如实说明即可。
            inside = os.path.abspath(cc_path).startswith(os.path.abspath(repo) + os.sep)
            rv.add(
                "W01",
                MAJOR if inside else INFO,
                "%s，两处 CMake 接线与 REGISTER_EXEC_V2（W01 / W06）无法检查"
                % (
                    "该文件在仓内，却不在受支持布局的 template 目录下"
                    if inside
                    else "该文件不在 --repo 指向的仓里"
                ),
                hint="受支持的路径形如 src/ops/<算子>/%s/<文件>.cc。\n%s"
                % (
                    layout.tmid,
                    "仓的目录结构可能变了——确认 --repo 指向的分支是否受支持。"
                    if inside
                    else "评审仓外文件（补丁、夹具）时属正常。",
                ),
            )
        check_mechanical(f, rv, algo_types)
    check_common(f, rv)
    if f.topo == "mesh":
        check_mesh(f, rv)
    elif f.topo == "nhr":
        check_nhr(f, rv)
    elif f.topo == "ring":
        check_ring(f, rv, algo_types)
    elif f.topo == "hd":
        check_hd(f, rv, algo_types)
    else:
        rv.add(
            "C02",
            INFO,
            "认不出算法族（mesh / nhr / ring / hd），只跑了通用检查",
            hint="用 --topo 指定，或让类名带上拓扑词。",
        )
    if spec_path:
        check_spec(f, rv, spec_path)
    return rv


def git_out(repo, *argv):
    """跑一条 git 命令，失败就抛 CalledProcessError（调用方负责 fail closed）。"""
    return subprocess.check_output(
        ("git", "-C", repo) + argv, stderr=subprocess.STDOUT
    ).decode("utf-8", "replace")


def git_bytes(repo, *argv):
    return subprocess.check_output(("git", "-C", repo) + argv, stderr=subprocess.PIPE)


def require_git_base(repo, base):
    try:
        git_out(repo, "rev-parse", "--git-dir")
    except Exception:
        sys.exit("--base 需要 %s 是一个 git 仓库，但它不是。" % repo)
    try:
        git_out(repo, "rev-parse", "--verify", "%s^{commit}" % base)
    except subprocess.CalledProcessError:
        sys.exit("--base 给的 `%s` 在 %s 里解析不出 commit。" % (base, repo))


def materialize_base(repo, base, layout, workdir):
    """把 base 版本里「评审要用到的那部分」解到临时目录。

    只读操作：走 `git archive`，不建 worktree、不碰 .git、不动工作树。
    解出来的子集要足够跑完整的机械层——两处 CMake、executor（W06）、
    alg_parse（X01 / W03）、以及全部 template（class_index 要认变体父类）。
    """
    # 用工作树里枚举出的**具体目录**当 pathspec：git 的 pathspec 不把
    # `src/ops/*/algorithm/template/aicpu` 当目录通配（ls-tree 直接匹配不到），
    # 而 git archive 只要有一条 pathspec 落空就整体失败（exit 128）。
    specs = ["src/common", "src/scatter_aicpu_kernel.cmake"]
    for mid in (layout.tmid, layout.emid):
        for d in glob.glob(os.path.join(repo, "src/ops/*", mid)):
            specs.append(os.path.relpath(d, repo).replace(os.sep, "/"))
    live = []
    for spec in specs:
        try:
            if git_out(repo, "ls-tree", "-r", "--name-only", base, "--", spec).strip():
                live.append(spec)
        except subprocess.CalledProcessError:
            pass
    if not live:
        return None
    try:
        blob = git_bytes(repo, "archive", "--format=tar", base, "--", *live)
    except subprocess.CalledProcessError:
        return None
    import io as _io
    import tarfile

    try:
        with tarfile.open(fileobj=_io.BytesIO(blob)) as tf:
            try:
                tf.extractall(workdir, filter="data")
            except TypeError:
                tf.extractall(workdir)  # nosec B202
    except Exception:
        return None
    return workdir


def changed_paths(repo, base):
    """相对 base 变动过的所有文件（含未跟踪），返回 {相对路径: 状态}。"""
    out = {}
    for line in git_out(repo, "diff", "--name-status", "-M", base, "--").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        st = parts[0][:1]
        if st in ("R", "C") and len(parts) >= 3:
            # 改名/复制有两端：只记新名会漏掉「旧路径的接线还留着」这类悬空引用。
            out.setdefault(parts[1], "D")  # 旧路径按删除处理
            out[parts[2]] = "A"  # 新路径在 base 上不存在，按新增处理
        else:
            out[parts[-1]] = st
    for line in git_out(
        repo, "ls-files", "--others", "--exclude-standard"
    ).splitlines():
        if line.strip():
            out[line.strip()] = "A"
    return out


def affected_templates(repo, layout, changes):
    """把「改了哪些文件」映射成「要重新评审哪些 template」。

    只看 .cc/.h 会漏掉三类真实改动：改了局部 CMakeLists、改了全局
    scatter_aicpu_kernel.cmake（W01）、改了 executor 的注册（W06）。
    删除同样要管：删掉配套 .h、删掉 CMake、删掉 executor 注册都是真实故障，
    以前一律 `continue` 跳过，等于把这类改动整个漏掉。

    返回 (要评审的 template 列表, 被整份删掉的 template 相对路径列表)。
    """
    all_tmpl = sorted(
        p for p in glob.glob(layout.template_glob(repo)) if "/op_common/" not in p
    )
    tmid, exdir = "/" + layout.tmid + "/", "/" + layout.emid + "/"
    hit, take_all, gone = set(), False, []
    for rel, st in changes.items():
        slashed = "/" + rel
        if (
            st == "D"
            and tmid in slashed
            and rel.endswith((".cc", ".h"))
            and "/op_common/" not in rel
        ):
            cc_rel = rel[:-2] + ".cc" if rel.endswith(".h") else rel
            cc_abs = os.path.join(repo, cc_rel)
            # 两半都要看：只删 .cc 留下 .h 同样是配对缺失，
            # 只看 .cc 存不存在会把它误判成「整份删掉」而只报 INFO。
            if os.path.exists(cc_abs) or os.path.exists(cc_abs[:-3] + ".h"):
                hit.add(os.path.abspath(cc_abs))  # 还剩一半 → 交给 C02 报配对缺失
            elif cc_rel not in gone:
                gone.append(cc_rel)  # 两半都没了 → 查有没有悬空的接线/注册
            continue
        if (
            slashed.startswith("/src/common/")
            or rel == "src/scatter_aicpu_kernel.cmake"
        ):
            take_all = True  # 真源/全局接线变了，影响面是全仓
            continue
        if tmid in slashed:
            if rel.endswith((".cc", ".h")) and "/op_common/" not in rel:
                cc = os.path.join(repo, rel[:-2] + ".cc" if rel.endswith(".h") else rel)
                if os.path.exists(cc):
                    hit.add(os.path.abspath(cc))
            else:  # 该算子目录下的 CMakeLists 等
                op_dir = os.path.join(repo, slashed[1:].split(tmid[1:])[0])
                hit.update(
                    os.path.abspath(t)
                    for t in all_tmpl
                    if os.path.abspath(t).startswith(os.path.abspath(op_dir))
                )
        elif exdir in slashed:  # executor 改了 → 该算子的 template 都要重查 W06
            op_dir = os.path.join(repo, slashed[1:].split(exdir[1:])[0])
            hit.update(
                os.path.abspath(t)
                for t in all_tmpl
                if os.path.abspath(t).startswith(os.path.abspath(op_dir))
            )
    if take_all:
        # all_tmpl 只 glob 得到「.cc 还在」的模板，半删的（.cc 没了、.h 还在）不在里面，
        # 直接返回 all_tmpl 会把 hit 里那些连带丢掉。
        hit.update(os.path.abspath(t) for t in all_tmpl)
    return sorted(hit), gone


def dangling_refs(repo, layout, cc_rel, base_root=None):
    """template 已经删了，但两处 CMake / executor 里还留着它 → 悬空引用。

    类名必须从 **base 快照的 .h** 里解析：按文件名拼类名会拼错
    （ins_temp_all_reduce_mesh_1D_one_shot → AllReduceMesh1dOneShot，
    丢了 InsTemp 前缀、把 1D 变成 1d、NHR 变成 Nhr），拼错就永远匹配不上注册宏。
    """
    filebase = os.path.basename(cc_rel)[:-3]
    op = layout.op_of(cc_rel)
    out = []
    kernel = os.path.join(repo, "src/scatter_aicpu_kernel.cmake")
    if os.path.exists(kernel) and layout.kernel_needle(op, filebase) in read(kernel):
        out.append("src/scatter_aicpu_kernel.cmake")
    if op:
        local = layout.cmake_local(repo, op)
        if os.path.exists(local) and ("/%s.cc" % filebase) in read(local):
            out.append(os.path.relpath(local, repo))

    cls = None
    if base_root:
        bh = os.path.join(base_root, cc_rel[:-3] + ".h")
        if os.path.exists(bh):
            m = re.search(r"class\s+(\w+)\s*:\s*public\s+\w+", read(bh))
            if m:
                cls = m.group(1)
    for ex in glob.glob(layout.executor_glob(repo)):
        txt = read(ex)
        why = []
        if re.search(r'#\s*include\s*"[^"]*%s\.h"' % re.escape(filebase), txt):
            why.append("include")
        if cls and re.search(
            r"REGISTER_EXEC_V2\w*[^;]*\b%s\b" % re.escape(cls), txt, re.S
        ):
            why.append("REGISTER_EXEC_V2(%s)" % cls)
        if why:
            out.append("%s（%s）" % (os.path.relpath(ex, repo), "、".join(why)))
    if cls is None and base_root:
        out.append("（注：base 快照里取不到类名，只查了 include 与两处 CMake）")
    return out


def identity(rel, f):
    """finding 的稳定身份。**不含行号**——行号只用来展示证据。

    带上严重度，这样"同一条规则从 MINOR 升到 BLOCKER"也算本次引入。
    """
    return (rel, f.rule, f.sev, f.msg)


def diff_findings(base_rv, cur_rv, rel):
    """当前结果里，base 上不存在（或条数变多）的那些 finding。"""
    from collections import Counter

    base_cnt = Counter(identity(rel, f) for f in (base_rv.findings if base_rv else []))
    new = []
    seen = Counter()
    for f in cur_rv.findings:
        k = identity(rel, f)
        seen[k] += 1
        if seen[k] > base_cnt.get(k, 0):
            new.append(f)
    return new


def resolve_repo(arg):
    """HCCL 仓路径只能来自使用者：--repo 或环境变量 HCCL_REPO。脚本里不带默认值，拿不到就报错。"""
    repo = arg or os.environ.get("HCCL_REPO")
    if not repo:
        sys.exit(
            "未指定 HCCL 仓路径。\n"
            "  用 --repo <HCCL 仓根目录> 传入，或先 export HCCL_REPO=<HCCL 仓根目录>。\n"
            "  ★ 不知道路径就去问用户，不要猜也不要扫盘。"
        )
    repo = os.path.abspath(os.path.expanduser(repo))
    if not os.path.isdir(os.path.join(repo, "src/ops")):
        sys.exit("不像是 HCCL 仓（缺少 src/ops 目录）: %s" % repo)
    return repo


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", help="HCCL 仓根目录；缺省读环境变量 HCCL_REPO")
    ap.add_argument("--all", action="store_true", help="扫描全仓所有 AICPU template")
    ap.add_argument("--spec", help="dataflow spec markdown，给了就一并做代码↔spec 对照")
    ap.add_argument(
        "--topo",
        choices=["mesh", "nhr", "ring", "hd"],
        help="为单个 template 强制指定算法族；不是 --all 的筛选条件",
    )
    ap.add_argument("--strict", action="store_true", help="MAJOR 也算失败（退出码 1）")
    ap.add_argument(
        "--quiet", action="store_true", help="--all 时只打印有 BLOCKER/MAJOR 的文件"
    )
    ap.add_argument(
        "--semantic-only",
        action="store_true",
        help="跳过机械层 W01–W07（已经用别的工具查过接线时用）",
    )
    ap.add_argument(
        "--base",
        metavar="REF",
        help="与该 commit 的快照比对 finding 集合，只把新出现的算作本次引入"
        "（如 --base HEAD / origin/master）；不给路径时自动取受影响的 template",
    )
    ap.add_argument(
        "--show-baseline",
        action="store_true",
        help="--base 时展开每个目标的存量 finding；默认只显示本次引入并汇总存量",
    )
    ap.add_argument("--rules", action="store_true", help="打印规则目录后退出")
    ap.add_argument(
        "paths", nargs="*", help="template 的 .cc 或 .h（相对仓根或绝对路径）"
    )
    args = ap.parse_args()

    if args.show_baseline and not args.base:
        ap.error("--show-baseline 只能和 --base 一起使用")

    if args.rules:
        print("规则目录（详见 references/）：\n")
        for rid, desc, ref in RULES:
            print("  %-4s %-58s %s" % (rid, desc, ref))
        return 0

    repo = resolve_repo(args.repo)
    layout = detect_layout(repo)
    if layout is None:
        sys.exit(
            '认不出这个仓的 AICPU template 布局，拒绝继续（不降级成"扫到 0 个"）。\n'
            "  仓: %s\n"
            "  已支持的布局：\n%s\n"
            "  本 skill 只支持当前 HCCL 结构；目录结构再变时在 LAYOUTS 里补一条。"
            % (
                repo,
                "\n".join(
                    "    - %-10s src/ops/<算子>/%s/" % (lay.name, lay.tmid)
                    for lay in LAYOUTS
                ),
            )
        )
    algo_types = load_algo_types(repo)
    if not algo_types:
        sys.exit(
            "解析不到 src/common/alg_parse.h 的 `enum class AlgoType`，拒绝继续。\n"
            "  仓: %s\n"
            "  X01（拓扑是否登记进选路）与 W03（props 的 AlgoType 是否存在）都以它为真源，\n"
            '  解析不到就无法判断，报"没问题"是错的。\n'
            "  没有这个文件通常说明该仓不是当前 HCCL 结构（本 skill 不支持更早的分支）。"
            % repo
        )
    targets = []
    if args.all:
        targets = [
            p
            for p in sorted(glob.glob(layout.template_glob(repo)))
            if "/op_common/" not in p
        ]
        if not targets:
            sys.exit(
                "--all 在 %s 布局下没扫到任何 template（src/ops/*/%s/*.cc）。\n"
                "  仓: %s" % (layout.name, layout.tmid, repo)
            )
    for p in args.paths:
        path = p if os.path.isabs(p) else os.path.join(repo, p)
        targets.append(path[:-2] + ".cc" if path.endswith(".h") else path)

    changes = None
    if args.base:
        require_git_base(repo, args.base)
        changes = changed_paths(repo, args.base)
        auto_targets, deleted = affected_templates(repo, layout, changes)
        if not targets:
            targets = auto_targets
            if not targets and not deleted:
                print(
                    "相对 %s 没有任何影响 AICPU template 的改动"
                    "（已一并考虑 .h、两处 CMake、executor、src/common 与删除）。"
                    % args.base
                )
                return 0
    if not targets and not (changes is not None and deleted):
        ap.error("请给出至少一个 template 路径，或使用 --all / --base")
    if args.spec and len(targets) != 1:
        ap.error("--spec 只能配单个 template（一个 spec 对应一个 template）")
    if args.topo and (args.all or len(targets) != 1):
        ap.error(
            "--topo 只能用于单个 template；它是归族覆盖开关，不是 --all 的筛选条件"
        )
    if args.spec and not os.path.isfile(args.spec):
        sys.exit("找不到 spec 文件: %s" % args.spec)

    tot = {BLOCKER: 0, MAJOR: 0, MINOR: 0, INFO: 0}
    new_tot = {BLOCKER: 0, MAJOR: 0, MINOR: 0, INFO: 0}
    per_rule = {}

    tmpdir = base_root = base_layout = base_algo = None
    if changes is not None:
        tmpdir = tempfile.mkdtemp(prefix="tmplreview-base-")
        base_root = materialize_base(repo, args.base, layout, tmpdir)
        if base_root:
            base_layout = detect_layout(base_root)
            base_algo = load_algo_types(base_root) if base_layout else None

    base_spec = None
    if changes is not None and args.spec:
        sp = os.path.abspath(args.spec)
        if sp.startswith(os.path.abspath(repo) + os.sep):
            srel = os.path.relpath(sp, repo).replace(os.sep, "/")
            try:
                blob = git_bytes(repo, "show", "%s:%s" % (args.base, srel))
                base_spec = os.path.join(tmpdir, "__base_spec.md")
                with open(base_spec, "wb") as fh:
                    fh.write(blob)
            except subprocess.CalledProcessError:
                base_spec = None  # base 上还没有这份 spec → 全部 S 记为本次引入
        # spec 在仓外时取不到 base 版本，同样按「全部 S 算本次引入」处理（宁可多报）

    def base_review_of(rel):
        """在 base 快照上评同一个文件。取不到就返回 (None, False) —— 无法比对时
        一律按「全部算本次引入」处理，宁可多报也不放过（门禁要 fail closed）。"""
        if not (base_root and base_layout and base_algo):
            return None, False
        bp = os.path.join(base_root, rel)
        if not (os.path.exists(bp) and os.path.exists(bp[:-3] + ".h")):
            return None, True  # base 上没有这个 template → 整份新增，比对结果就是全新
        try:
            return review_one(
                base_root,
                bp,
                base_spec,
                base_algo,
                base_layout,
                args.topo,
                args.semantic_only,
            ), True
        except Exception:
            return None, False

    per_rule = {}
    for path in targets:
        rv = review_one(
            repo, path, args.spec, algo_types, layout, args.topo, args.semantic_only
        )
        for sev in tot:
            tot[sev] += rv.count(sev)
        for f in rv.findings:
            per_rule.setdefault(f.rule, []).append(rv.title.split()[0])
        if changes is not None:
            rel = os.path.relpath(path, repo)
            base_rv, comparable = base_review_of(rel)
            fresh = diff_findings(base_rv, rv, rel) if comparable else list(rv.findings)
            for f in fresh:
                new_tot[f.sev] += 1
            if fresh or args.show_baseline or not comparable:
                rv.dump_attributed(fresh, comparable, args.show_baseline)
        elif not args.all or not args.quiet or rv.count(BLOCKER) or rv.count(MAJOR):
            rv.dump()

    if args.all:
        print("=" * 78)
        print("按规则汇总（%d 个文件）" % len(targets))
        print("=" * 78)
        for rid, files in sorted(per_rule.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            desc = next((d for r, d, _ in RULES if r == rid), "")
            print("  %-4s %3d 处  %s" % (rid, len(files), desc))
        print()
    if changes is not None:
        for cc_rel in deleted:
            refs = dangling_refs(repo, layout, cc_rel, base_root)
            rv = Review("%s  (已删除)" % cc_rel, "-", "")
            if refs:
                rv.add(
                    "W01",
                    BLOCKER,
                    "template 已删除，但仍被 %d 处引用" % len(refs),
                    hint="删干净了才算删：还留着引用的地方——\n  - "
                    + "\n  - ".join(refs)
                    + "\n这些地方会去编 / 注册一个不存在的文件。",
                )
                new_tot[BLOCKER] += 1
                tot[BLOCKER] += 1
            else:
                rv.add(
                    "W01",
                    INFO,
                    "template 已删除，两处 CMake 与 executor 里没有残留引用",
                    hint="删除本身是否符合预期，请人工确认。",
                )
                new_tot[INFO] += 1
                tot[INFO] += 1
            rv.dump_attributed(list(rv.findings), True, args.show_baseline)

    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if changes is not None:
        print(
            "本次改动引入：BLOCKER %d、MAJOR %d、MINOR %d、INFO %d"
            % (new_tot[BLOCKER], new_tot[MAJOR], new_tot[MINOR], new_tot[INFO])
        )
        print(
            "全文（含存量欠账）：BLOCKER %d、MAJOR %d、MINOR %d、INFO %d"
            % (tot[BLOCKER], tot[MAJOR], tot[MINOR], tot[INFO])
        )
        print(
            "退出码只看「本次引入」；存量欠账默认只汇总，需要明细时加 --show-baseline。"
        )
        if new_tot[BLOCKER] or (args.strict and new_tot[MAJOR]):
            return 1
        return 0
    print(
        "BLOCKER %d、MAJOR %d、MINOR %d、INFO %d"
        % (tot[BLOCKER], tot[MAJOR], tot[MINOR], tot[INFO])
    )
    if tot[BLOCKER] or (args.strict and tot[MAJOR]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
