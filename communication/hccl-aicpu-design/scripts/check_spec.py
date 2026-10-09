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

"""检查一份 HCCL AICPU template 数据流规格（dataflow spec）是否可以开始翻译成代码。

用法：
    python3 scripts/check_spec.py path/to/spec.md

记法定义见本 Skill 的 references/07-dataflow-spec.md。存在 ERROR 时退出码为 1。
"""

import argparse
import os
import re
import sys

from check_layout import validate as validate_layout

ERROR, WARN, OK = "ERROR", "WARN", "OK"

SECTIONS = [
    (1, "元信息"),
    (2, "适用条件"),
    (3, "资源"),
    (4, "Buffer 布局"),
    (5, "切片"),
    (6, "数据流"),
    (7, "边界条件"),
    (8, "Cost model"),
    (9, "验证"),
    (10, "待确认"),
]

WRITE_PRIMS = [
    "SendRecvBatchWriteReduce",
    "SendRecvBatchWrite",
    "SendRecvWriteReduce",
    "SendRecvWrite",
    "SendBatchWriteReduce",
    "SendBatchWrite",
    "SendWriteReduce",
    "SendWrite",
]
READ_PRIMS = [
    "SendRecvBatchReadReduce",
    "SendRecvBatchRead",
    "SendRecvReadReduce",
    "SendRecvRead",
    "RecvBatchReadReduce",
    "RecvBatchRead",
    "RecvReadReduce",
    "RecvRead",
]
LOCAL_PRIMS = ["LocalCopySlices", "LocalReduce", "LocalCopy"]
ALL_PRIMS = WRITE_PRIMS + READ_PRIMS + LOCAL_PRIMS

# ---------------------------------------------------------------------------
# 规则登记表：ID -> (等级, 适用范围, 一句话规则)
# 跨能力维护登记见 ../hccl-aicpu-best-practice/references/07-rules.md；
# 本脚本可独立检查设计规格，不在运行时加载实现 Skill。
#
# 与 check_template.py 的差别：spec 是**新写**的文档，不存在存量欠债，
# 所以这里没有 --strict-new；每个检查点的 ERROR/WARN 由调用处按具体情形决定
# （同一条规则的「反方向」提示通常降一级，如 R-FLOW-009 的「没用 SCR 却填了倍数」）。
# ---------------------------------------------------------------------------
RULES = {
    "R-SPEC-001": (
        "BLOCKER",
        "模式C",
        "10 个章节齐全，序号与标题关键字与 dataflow-spec-template.md 一致",
    ),
    "R-SPEC-002": ("BLOCKER", "模式C", "开始写代码前 TBD: 必须清零"),
    "R-SPEC-003": ("BLOCKER", "模式C", "模板占位符 <...> 必须全部替换"),
    "R-SPEC-004": ("BLOCKER", "模式C", "第 6 章必须有代码块且能解析出至少一条边"),
    "R-SPEC-005": (
        "BLOCKER",
        "模式C",
        "每条边必须有 -> / into / reduce-> 分隔源和目标",
    ),
    "R-SPEC-006": ("BLOCKER", "模式C", "第 1 章元信息必填字段齐全"),
    "R-SPEC-007": ("BLOCKER", "模式C", "第 7 章边界条件每项都要写出行为"),
    "R-FLOW-001": ("BLOCKER", "全量", "写模式的目标必须带 @p，源不许带 @p"),
    "R-FLOW-002": ("BLOCKER", "全量", "读模式的源必须带 @p，目标不许带 @p"),
    "R-FLOW-003": (
        "BLOCKER",
        "全量",
        "跨 rank 边只写一个方向；本地原语操作数不许带 @p",
    ),
    "R-FLOW-004": ("BLOCKER", "全量", "一条边的源长度与目标长度必须相等"),
    "R-FLOW-005": ("BLOCKER", "全量", "sync main -> sub 与 sync sub -> main 必须成对"),
    "R-FLOW-006": ("BLOCKER", "全量", "用到从流 T[i] 就必须有主从同步"),
    "R-FLOW-007": (
        "BLOCKER",
        "全量",
        "CalcScratchMultiple 与实际写进 hcclBuff 的最大字节数一致",
    ),
    "R-FLOW-008": (
        "BLOCKER",
        "全量",
        "scratch 倍数按单次 repeat 报，表达式里不许出现 RPT",
    ),
    "R-FLOW-009": ("MUST", "全量", "数据流用到了 SCR，scratch 倍数就不能填 0"),
    "R-FLOW-010": ("MUST", "全量", "RPT≠1 或分层构件时，数据流必须有 repeat for"),
    "R-FLOW-011": ("MUST", "全量", "声明了 RPT 就要给全 IRS/ORS/ISS/OSS"),
    "R-FLOW-012": ("SHOULD", "新增", "默认写上 repeat 循环；省略要在第 5 章写明理由"),
    "R-FLOW-018": (
        "MUST",
        "全量",
        "thread 总数与从流使用自洽；notify 数与 GetNotifyIdx* 长度自洽",
    ),
    "R-FLOW-021": (
        "BLOCKER",
        "模式C",
        "五参数赋值来源、rank 映射与独立期望偏移量化样例必须齐全且通过",
    ),
    "R-DEC-001": (
        "BLOCKER",
        "模式C",
        "spec 必须声明启用方式（标准配置显式选中 / 自动选路 / 私有开关）",
    ),
    "R-DEC-002": (
        "BLOCKER",
        "全量",
        "默认选路不变或按批准范围改变，须声明依据、范围及选路验收矩阵",
    ),
    "R-DEC-004": ("BLOCKER", "模式C", "记录既有阈值依据，不新增性能验收或豁免基线门禁"),
    "R-DEC-006": ("MUST", "模式C", "spec 必须声明逻辑 peer 与申请链路；不一致要写理由"),
    "R-DEC-007": ("MUST", "模式C", "spec 声明目标 runtime 模式，验收时验证支持范围"),
    "R-DEC-008": (
        "MUST",
        "模式C",
        "spec 必须声明 Cost model 策略；自动选路必须实现，显式路径的候选依赖需人工核对",
    ),
    "R-DEC-010": (
        "MUST",
        "模式C",
        "spec 必须给全支持矩阵（selector / 平台 / inplace）",
    ),
    "R-DEC-011": ("MUST", "模式C", "多 channel 切分公式必须在 spec 里固定"),
}


def print_rules(rid=None):
    """打印规则卡；本脚本的 RULES 可独立使用。"""
    if rid:
        key = rid.upper()
        if key not in RULES:
            print("check_spec.py 不检查这条规则：%s" % rid)
            print("本脚本涵盖：%s" % "、".join(sorted(RULES)))
            print(
                "跨能力维护登记见 ../hccl-aicpu-best-practice/references/07-rules.md。"
            )
            return 1
        level, scope, text = RULES[key]
        print("%s  [%s]  适用: %s" % (key, level, scope))
        print("  规则: %s" % text)
        print(
            "  设计说明: references/07-dataflow-spec.md、references/08-algorithm-design.md"
        )
        print("  跨能力维护登记: ../hccl-aicpu-best-practice/references/07-rules.md")
        return 0
    print(
        "check_spec.py 能自动检查的规则（按稳定 ID 查询；跨能力规则登记由仓库维护）："
    )
    print()
    for key in sorted(RULES):
        level, scope, text = RULES[key]
        print("  %-13s %-8s %-6s %s" % (key, level, scope, text))
    print()
    return 0


OPERAND = re.compile(r"\b(IN|OUT|SCR)(@\w+)?\[\s*([^,\]]*?)\s*,\s*([^\]]*?)\s*\]")
ARROW = re.compile(r"\s(?:reduce->|->|into)\s")


class Report(object):
    def __init__(self, title):
        self.title = title
        self.items = []

    def add(self, level, msg, hint=None, rid=None):
        self.items.append((level, msg, hint, rid))

    @property
    def errors(self):
        return sum(1 for lv, _, _, _ in self.items if lv == ERROR)

    def dump(self):
        print("=" * 78)
        print(self.title)
        print("=" * 78)
        for level, msg, hint, rid in self.items:
            print(
                "%s  %s%s"
                % (
                    {ERROR: "✗ ERROR", WARN: "! WARN ", OK: "✓ OK   "}[level],
                    ("[%s] " % rid) if rid else "",
                    msg,
                )
            )
            if hint:
                for line in hint.splitlines():
                    print("           %s" % line)
        print()


def strip_html_comments(text):
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def cut_to_form(text):
    """模板文件自带填写说明，说明部分不参与校验：从「模板正文」标记之后开始解析。"""
    m = None
    for pat in (r"^#+\s*↓+\s*以下是模板正文.*$", r"^#+\s*数据流规格[:：]"):
        m = re.search(pat, text, re.M)
        if m:
            break
    return text[m.start():] if m else text


def code_blocks(text):
    return re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.S)


def markdown_headings(text):
    """Yield ATX headings outside fenced blocks, retaining original body offsets.

    Standalone copy also in the review skill's review_template.py; keep behavior
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


def section_body(text, num, title):
    headings = list(markdown_headings(text))
    for index, (_, start, level, name) in enumerate(headings):
        if level != 2 or not re.match(r"%d\.\s*%s" % (num, re.escape(title)), name):
            continue
        end = next(
            (pos for pos, _, depth, _ in headings[index + 1:] if depth <= level),
            len(text),
        )
        return text[start:end]
    return None


def table_value(body, key):
    """从 markdown 表里取 `| key | value |` 的 value。"""
    if not body:
        return None
    for line in body.splitlines():
        if not line.strip().startswith("|") or key not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and key in cells[0]:
            val = cells[1].strip()
            first = re.search(r"`([^`]+)`", val)  # 优先取第一个反引号里的内容
            return first.group(1).strip() if first else val
    return None


def check(path):
    with open(path, encoding="utf-8") as source:
        raw = source.read()
    text = cut_to_form(strip_html_comments(raw))
    rep = Report(os.path.relpath(path))

    # ---------- 章节完整性 ----------
    bodies = {}
    missing = []
    for num, title in SECTIONS:
        body = section_body(text, num, title)
        bodies[num] = body
        if body is None:
            missing.append("%d. %s" % (num, title))
    if missing:
        rep.add(
            ERROR,
            "缺少章节：%s" % "、".join(missing),
            "章节标题必须是 `## <序号>. <标题>`，照 hccl-aicpu-design/dataflow-spec-template.md 的写法。",
            rid="R-SPEC-001",
        )
        return rep
    rep.add(OK, "10 个章节齐全", rid="R-SPEC-001")

    # ---------- TBD ----------
    tbds = re.findall(r"TBD\s*[:：]\s*([^\n]*)", text)
    if tbds:
        rep.add(
            ERROR,
            "残留 %d 处 TBD，未确认前不要开始写代码" % len(tbds),
            "\n".join("- %s" % t.strip().rstrip("`>") for t in tbds[:6]),
            rid="R-SPEC-002",
        )
    else:
        rep.add(OK, "没有残留 TBD", rid="R-SPEC-002")

    # ---------- 未替换的占位符 ----------
    outside = re.sub(r"```.*?```", "", text, flags=re.S)
    holes = [
        h
        for h in re.findall(r"<([^<>\n]{1,60})>", outside)
        if not h.startswith("/")
    ]
    if holes:
        rep.add(
            ERROR,
            "有 %d 处占位符没填：%s"
            % (len(holes), "、".join("<%s>" % h for h in holes[:5])),
            rid="R-SPEC-003",
        )
    else:
        rep.add(OK, "占位符已全部替换", rid="R-SPEC-003")

    # ---------- 数据流代码块 ----------
    flow_blocks = code_blocks(bodies[6] or "")
    if not flow_blocks:
        rep.add(ERROR, "第 6 章「数据流」里没有 ``` 代码块", rid="R-SPEC-004")
        return rep
    flow = "\n".join(flow_blocks)
    rep.add(
        OK,
        "数据流代码块 %d 个，共 %d 行"
        % (len(flow_blocks), len(flow.strip().splitlines())),
        rid="R-SPEC-004",
    )

    # ---------- 逐行解析数据流 ----------
    uses_scr, uses_sub_thread, edges = False, False, 0
    edge_errors = 0
    for lineno, line in enumerate(flow.splitlines(), 1):
        code = line.split("#", 1)[0].strip()
        if not code:
            continue
        if re.search(r"\bT\[", code):
            uses_sub_thread = True
        if re.search(r"\bSCR\b", code):
            uses_scr = True

        prim = next((p for p in ALL_PRIMS if re.search(r"\b%s\b" % p, code)), None)
        if prim is None:
            continue
        edges += 1
        rhs = code.split(prim, 1)[1]
        parts = ARROW.split(rhs)
        if len(parts) != 2:
            rep.add(
                ERROR,
                "数据流第 %d 行：`%s` 缺少 `->` / `into` / `reduce->` 分隔的源和目标"
                % (lineno, code.strip()),
                rid="R-SPEC-005",
            )
            edge_errors += 1
            continue
        src_txt, dst_txt = parts[0].strip(), parts[1].strip()
        src, dst = OPERAND.search(src_txt), OPERAND.search(dst_txt)
        if not src or not dst:
            rep.add(
                ERROR,
                "数据流第 %d 行：源和目标都必须是 IN/OUT/SCR[off,len] 操作数" % lineno,
                rid="R-SPEC-005",
            )
            edge_errors += 1
            continue

        if prim in WRITE_PRIMS:
            if not (dst and dst.group(2)):
                rep.add(
                    ERROR,
                    "数据流第 %d 行：写模式 `%s` 的目标必须是远端 `XXX@p[...]`"
                    % (lineno, prim),
                    "当前目标：%s\n写模式底层只用 txSlices，dst 必须落在对端 remoteCclMem 上。"
                    % dst_txt,
                    rid="R-FLOW-001",
                )
                edge_errors += 1
            if src and src.group(2):
                rep.add(
                    ERROR,
                    "数据流第 %d 行：写模式 `%s` 的源不能带 `@p`（源是本地）"
                    % (lineno, prim),
                    rid="R-FLOW-001",
                )
                edge_errors += 1
        elif prim in READ_PRIMS:
            if not (src and src.group(2)):
                rep.add(
                    ERROR,
                    "数据流第 %d 行：读模式 `%s` 的源必须是远端 `XXX@p[...]`"
                    % (lineno, prim),
                    "当前源：%s\n读模式底层只用 rxSlices，src 必须落在对端上。"
                    % src_txt,
                    rid="R-FLOW-002",
                )
                edge_errors += 1
            if dst and dst.group(2):
                rep.add(
                    ERROR,
                    "数据流第 %d 行：读模式 `%s` 的目标不能带 `@p`（目标是本地）"
                    % (lineno, prim),
                    rid="R-FLOW-002",
                )
                edge_errors += 1
        else:
            if (src and src.group(2)) or (dst and dst.group(2)):
                rep.add(
                    ERROR,
                    "数据流第 %d 行：本地原语 `%s` 的操作数不能带 `@p`"
                    % (lineno, prim),
                    rid="R-FLOW-003",
                )
                edge_errors += 1

        if src and dst:
            sl, dl = re.sub(r"\s", "", src.group(4)), re.sub(r"\s", "", dst.group(4))
            if sl != dl:
                rep.add(
                    ERROR,
                    "数据流第 %d 行：源长度 `%s` 与目标长度 `%s` 不等"
                    % (lineno, sl, dl),
                    "传输长度取源的 size_，两侧写成不同长度说明 spec 有歧义。",
                    rid="R-FLOW-004",
                )
                edge_errors += 1

    if edges == 0:
        rep.add(
            ERROR,
            "数据流里没有识别出任何搬运原语",
            "原语名必须与 wrapper 函数名一致，见 references/07 §2。",
            rid="R-SPEC-004",
        )
    elif edge_errors:
        rep.add(
            WARN,
            "解析出 %d 条数据流边，其中 %d 条有问题（见上）" % (edges, edge_errors),
        )
    else:
        rep.add(OK, "解析出 %d 条数据流边，方向与长度自洽" % edges, rid="R-FLOW-004")

    # ---------- 同步配对 ----------
    flow_code = "\n".join(line.split("#", 1)[0] for line in flow.splitlines())
    pre = len(re.findall(r"sync\s+main\s*->\s*sub", flow_code))
    post = len(re.findall(r"sync\s+sub\s*->\s*main", flow_code))
    if pre != post:
        rep.add(
            ERROR,
            "`sync main -> sub`(%d) 与 `sync sub -> main`(%d) 不配对" % (pre, post),
            rid="R-FLOW-005",
        )
    elif pre:
        rep.add(OK, "主从同步 %d 对，配对正确" % pre, rid="R-FLOW-005")
    elif uses_sub_thread:
        rep.add(
            ERROR,
            "用到了从流 `T[i]` 却没有任何 `sync main -> sub` / `sync sub -> main`",
            "从流的任务必须被主从同步夹住，否则主流会读到没写完的数据。",
            rid="R-FLOW-006",
        )

    # ---------- scratch 倍数一致性 ----------
    mult = table_value(bodies[4], "scratch 倍数")
    if mult is None:
        rep.add(
            ERROR,
            "第 4 章「Buffer 布局」里找不到 `scratch 倍数` 表格行",
            rid="R-FLOW-007",
        )
    else:
        mult_clean = mult.strip("*` ")
        if uses_scr and mult_clean in ("0", "`0`"):
            rep.add(
                ERROR,
                "数据流用到了 SCR，但 scratch 倍数填的是 0",
                "倍数为 0 表示不占 hcclBuff，executor 会据此放开单轮数据量，必然越界。",
                rid="R-FLOW-009",
            )
        else:
            rep.add(
                OK,
                "scratch 倍数 = %s，与数据流对 SCR 的使用一致" % mult_clean,
                rid="R-FLOW-007",
            )
        if not uses_scr and mult_clean not in ("0",):
            rep.add(
                WARN,
                "数据流没用到 SCR，但 scratch 倍数填了 %s" % mult_clean,
                rid="R-FLOW-009",
            )

    # ---------- repeat / stride 一致性 ----------
    #  RPT/IRS/ORS 由 executor 填、template 只读；漏写 repeat 循环的 template 一旦被
    #  分层 executor 复用就只处理第 0 块。定义见 references/07 §1.5。
    flow_repeat = bool(re.search(r"^\s*repeat\s+for\s+\w+\s+in\s+\S+", flow_code, re.M))
    layer = (table_value(bodies[1], "所属层级") or "").strip()
    kind = (table_value(bodies[1], "spec 类型") or "").strip()
    reverse = bool(re.search(r"逆向|legacy|存量", kind))
    layered = "单层" not in layer and bool(re.search(r"intra|inter|分层", layer, re.I))
    rpt = table_value(bodies[5], "RPT")
    rpt_clean = rpt.strip("*` ") if rpt else None
    declared = rpt_clean is not None and rpt_clean != ""
    is_one = rpt_clean in ("1",)

    if not declared:
        rep.add(
            WARN,
            "第 5 章「切片」没有声明 `RPT`（repeat 次数）",
            "按 executor 实际赋值填写，单层没有固定组合；定义见 references/09-template-data-params.md。",
            rid="R-FLOW-011",
        )
    else:
        strides = [
            k
            for k in ("IRS", "ORS", "ISS", "OSS")
            if not table_value(bodies[5], k)
        ]
        if strides:
            rep.add(
                ERROR,
                "第 5 章声明了 `RPT` 但缺少 %s 的取值"
                % "、".join("`%s`" % s for s in strides),
                "五个符号要一起声明：RPT / IRS / ORS / ISS / OSS。",
                rid="R-FLOW-011",
            )
        elif not is_one and not flow_repeat:
            rep.add(
                ERROR,
                "第 5 章声明 `RPT` = %s（非 1），但第 6 章数据流里没有 `repeat for`"
                % rpt_clean,
                "被重复执行的 template，每一块的地址都要带 rpt*IRS / rpt*ORS，\n"
                "只写一块等于只处理第 0 块数据。",
                rid="R-FLOW-010",
            )
        elif layered and not flow_repeat:
            rep.add(
                ERROR,
                "第 1 章「所属层级」是分层构件（%s），但数据流里没有 `repeat for`"
                % layer,
                "核对该阶段 executor 的 repeat 覆盖契约，\n"
                "不写 repeat 循环的 template 被复用时只处理第 0 块。见 references/07 §1.5。",
                rid="R-FLOW-010",
            )
        elif is_one and not flow_repeat and reverse:
            rep.add(
                OK,
                "逆向存量实现的 spec：RPT = 1 且源码无 repeat 循环，如实反映即可",
                rid="R-FLOW-012",
            )
        elif is_one and not flow_repeat:
            rep.add(
                WARN,
                "新建 spec 的数据流没有 `repeat for`（RPT = 1）",
                "默认就写上：RPT=1、IRS=ORS=0 时循环自动退化，零代价；不写则本 template\n"
                "以后不能被 2/3 级 executor 当 intra/inter 构件复用。\n"
                "确定只用于单层、也不打算被复用的，在第 5 章写明理由即可忽略本条；\n"
                "逆向存量实现的 spec 请在第 1 章把「spec 类型」填成「逆向存量」。",
                rid="R-FLOW-012",
            )
        else:
            rep.add(
                OK, "repeat 声明与数据流自洽（RPT = %s）" % rpt_clean, rid="R-FLOW-010"
            )

    if flow_repeat and not declared:
        rep.add(
            ERROR,
            "数据流用到了 repeat / RPT，但第 5 章「切片」没有声明这组符号",
            "补上 RPT / IRS / ORS / ISS / OSS 五行，见 references/07 §1.5。",
            rid="R-FLOW-011",
        )

    # scratch 倍数里不许出现 RPT（放大是 executor 的活，自己乘 = 双重放大）
    mult_for_rpt = table_value(bodies[4], "scratch 倍数")
    if mult_for_rpt and re.search(r"\b(RPT|repeatNum)\b", mult_for_rpt):
        rep.add(
            ERROR,
            "scratch 倍数表达式里出现了 `RPT`：%s" % mult_for_rpt.strip("*` "),
            "倍数按单次 repeat 报，executor 会自己算 max(另一层, 本层 * RPT)。\n"
            "自己先乘一遍 = 双重放大，loop 次数翻倍且编译/ST 都不报错。见 references/03 §2.1。",
            rid="R-FLOW-008",
        )

    # ---------- thread 数一致性 ----------
    tnum = table_value(bodies[3], "thread 总数")
    if tnum is None:
        rep.add(ERROR, "第 3 章「资源」里找不到 `thread 总数` 表格行", rid="R-FLOW-018")
    elif uses_sub_thread and tnum.strip("` ") == "1":
        rep.add(
            ERROR, "数据流用到了从流 `T[i]`，但 thread 总数填的是 1", rid="R-FLOW-018"
        )
    else:
        rep.add(OK, "thread 总数 = %s" % tnum, rid="R-FLOW-018")

    # ---------- 边界条件填写完整性 ----------
    boxes = re.findall(r"^\s*-\s*\[([ xX])\]\s*(.*)$", bodies[7] or "", re.M)
    if not boxes:
        rep.add(ERROR, "第 7 章「边界条件」里没有勾选项", rid="R-SPEC-007")
    else:
        blank = [t for _, t in boxes if not re.search(r"[:：]\s*\S", t)]
        if blank:
            rep.add(
                ERROR,
                "边界条件有 %d 项没写行为：%s"
                % (len(blank), "；".join(b[:28] for b in blank[:4])),
                "不适用也要写明理由，不要留空。",
                rid="R-SPEC-007",
            )
        else:
            rep.add(OK, "边界条件 %d 项，全部写了行为" % len(boxes), rid="R-SPEC-007")

    # ---------- 元信息必填 ----------
    for key in ("算子", "类名", "文件名", "拓扑", "算法名"):
        if not table_value(bodies[1], key):
            rep.add(ERROR, "第 1 章「元信息」缺少 `%s`" % key, rid="R-SPEC-006")

    # ---------- 架构决策必须落定（R-DEC-*）----------
    #  这些字段不填死，同一句提示词两次生成会得到「启用方式 / 默认选路 / runtime 模式 /
    #  链路资源 / cost model」五套互相冲突的实现——每一套都能过其余检查。
    required_decisions = [
        (
            1,
            "启用方式",
            "R-DEC-001",
            "三选一：标准配置显式选中 / 自动选路 / 私有开关。\n"
            "没有性能数据和产品需求时，只允许「标准配置显式选中」（HCCL_ALGO），\n"
            "不改默认路由，也不新造私有环境变量。",
        ),
        (
            1,
            "默认选路影响",
            "R-DEC-002",
            "填「不变」或「按批准范围改变」；后者须填写选路变更依据、允许变更范围和选路验收矩阵。",
        ),
        (
            2,
            "模式",
            "R-DEC-007",
            "OPBASE / OFFLOAD / 都支持。列明目标与待验证项，验收时验证支持范围——\n"
            "声明了 OFFLOAD 却仍走 scratch 中转，是「注册声称支持、执行路径没实现」的典型。",
        ),
        (
            2,
            "selector",
            "R-DEC-010",
            "老格式 / 新 DSL / 两者。只覆盖一条链路会出现「算法注册了但选不中」。",
        ),
        (
            2,
            "平台",
            "R-DEC-010",
            "不同平台解析 HCCL_ALGO 的路径不同，必须写明目标平台。",
        ),
        (2, "inplace", "R-DEC-010", "支持 / 拒绝 / 回退到 X，三选一。"),
        (
            3,
            "逻辑 peer",
            "R-DEC-006",
            "算法每轮实际访问的 peer 集合。「用 ring 算法」不等于「要申请全 Mesh 链路」。",
        ),
        (
            3,
            "申请链路",
            "R-DEC-006",
            "初始化阶段申请的链路集合，与逻辑 peer 不一致要写理由。",
        ),
        (
            3,
            "多 channel 切分",
            "R-DEC-011",
            "第 c 个 channel 的 offset/len 公式、取哪一侧的 channel 集合、非整除与空切片策略。\n"
            "NCH=1 时填「不切分」。",
        ),
        (
            8,
            "策略",
            "R-DEC-008",
            "必须实现 / 不需要并注明链路依据。自动选路或经成本候选过滤的显式选路均需实现；链路需人工核对。",
        ),
    ]
    missing = []
    for num, key, rid, hint in required_decisions:
        if not table_value(bodies[num], key):
            rep.add(ERROR, "第 %d 章缺少决策字段 `%s`" % (num, key), hint, rid=rid)
            missing.append(key)
    if not missing:
        rep.add(OK, "%d 项架构决策字段齐全" % len(required_decisions), rid="R-DEC-001")

    # Explicit decision values avoid treating the negation 不改变 as 改变.
    routing = table_value(bodies[1], "默认选路影响") or ""
    evidence = table_value(bodies[8], "自动阈值证据") or ""
    if routing == "按批准范围改变":
        for key in ("选路变更依据", "允许变更范围", "选路验收矩阵"):
            value = table_value(bodies[1], key)
            if not value or value.strip() in ("无", "None", "-", "不适用"):
                rep.add(
                    ERROR,
                    "改变默认选路时必须填写 `%s`；不能以 benchmark 替代上游需求或批准依据"
                    % key,
                    rid="R-DEC-002",
                )
        rep.add(
            WARN,
            "选路变更字段不证明授权有效；人工核对原始需求、变更范围和预先固定的矩阵",
            rid="R-DEC-002",
        )
    elif routing not in ("不变", "不改变", "保持不变", "保持现有算法不变"):
        rep.add(
            ERROR,
            "默认选路影响只能填「不变」或「按批准范围改变」，补充说明放在反引号外",
            rid="R-DEC-002",
        )
    if not evidence:
        rep.add(
            ERROR, "第 8 章需填写自动阈值证据；无阈值时明确填无阈值", rid="R-DEC-004"
        )

    # 自动选路 → cost model 必须实现
    activation = table_value(bodies[1], "启用方式") or ""
    if routing == "按批准范围改变" and "显式" in activation:
        rep.add(
            ERROR,
            "仅显式启用与改变默认选路冲突；先核对本任务的启用方式",
            rid="R-DEC-002",
        )
    policy = table_value(bodies[8], "策略") or ""
    if "自动" in activation and "不需要" in policy:
        rep.add(
            ERROR,
            "启用方式是「自动选路」，Cost model 策略却填「不需要」",
            "新 selector 会跳过没有 CalcCostCoeff 的算法，结果是注册成功但永远选不中。\n"
            "要么把策略改成「必须实现」，要么把启用方式改成「标准配置显式选中」。",
            rid="R-DEC-008",
        )

    # 申请链路 ≠ 逻辑 peer → 要有理由
    peers = table_value(bodies[3], "逻辑 peer") or ""
    links = table_value(bodies[3], "申请链路") or ""
    reason = table_value(bodies[3], "过度申请理由") or ""
    if (
        peers
        and links
        and peers != links
        and (not reason or reason.strip() in ("无", "None", "-"))
    ):
        rep.add(
            WARN,
            "申请链路(%s)与逻辑 peer(%s)不一致，但「过度申请理由」填的是「%s」"
            % (links[:24], peers[:24], reason or "空"),
            "单环这类算法默认只申请前驱/后继；要申请全 Mesh 就写明是复用既有框架还是性能原因。",
            rid="R-DEC-006",
        )

    for key in ("参数赋值来源", "算法 rank 映射"):
        if not table_value(bodies[5], key):
            rep.add(ERROR, "第 5 章缺少 `%s`" % key, rid="R-FLOW-021")
    for error in validate_layout(bodies[5] or ""):
        rep.add(ERROR, "地址量化检查：" + error, rid="R-FLOW-021")

    return rep


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--rule",
        nargs="?",
        const="",
        metavar="ID",
        help="打印一条规则（如 --rule R-FLOW-001）；不带参数则列出全部",
    )
    ap.add_argument("spec", nargs="*", help="dataflow spec markdown 路径")
    args = ap.parse_args()

    if args.rule is not None:
        return print_rules(args.rule or None)
    if not args.spec:
        ap.error("请给出至少一个 spec 路径，或用 --rule 查规则")

    total = 0
    for path in args.spec:
        if not os.path.exists(path):
            print("找不到文件: %s" % path)
            total += 1
            continue
        rep = check(path)
        rep.dump()
        total += rep.errors
    if total:
        print("共 %d 个 ERROR，先修掉再翻译成代码。" % total)
        return 1
    print("spec 检查通过，可以开始翻译成代码。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
