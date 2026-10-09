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

"""检查一个 HCCL AICPU algorithm template 的规范性与两处 CMake 接线。

用法：
    python3 scripts/check_template.py --repo <HCCL 仓根目录> \
            src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_foo.cc
    python3 scripts/check_template.py --repo <HCCL 仓根目录> --all          # 扫全仓所有 aicpu template
    python3 scripts/check_template.py --repo <HCCL 仓根目录> --strict-new <file>  # 新文件门禁
    python3 scripts/check_template.py --rule R-CMAKE-002                    # 查一条规则

传 .cc 或 .h 均可，会自动配对。存在 ERROR 时退出码为 1。

每条检查都带一个规则 ID，定义在 references/07-rules.md，与下面的 RULES 字典一一对应。
**改 RULES 必须同步改那份文档**（等级 / 适用范围 / 规则文字）。
"""

import argparse
import glob
import os
import re
import sys

from cmake_utils import has_source

ERROR, WARN, OK = "ERROR", "WARN", "OK"

BLOCKER, MUST, SHOULD = "BLOCKER", "MUST", "SHOULD"
ALL, NEW, NEWMOD = "全量", "新增", "新增+修改"

# ---------------------------------------------------------------------------
# 规则登记表：ID -> (等级, 适用范围, 一句话规则)
# 权威定义在 references/07-rules.md §1..§10，两边必须一致。
#
# 等级与适用范围如何决定脚本行为（与 07-rules.md §0 的映射表相同）：
#   适用=全量        BLOCKER / MUST -> ERROR ；SHOULD -> WARN
#   适用=新增/新增+修改  默认 WARN ；--strict-new 下 BLOCKER / MUST -> ERROR，SHOULD 仍 WARN
#
# 「适用=新增」是存量欠债的出口：存量文件违反它只报 WARN，新文件用 --strict-new 卡死。
# ---------------------------------------------------------------------------
RULES = {
    "R-PATH-003": (
        MUST,
        NEW,
        "template 落 src/ops/<op>/[algorithm/]template/aicpu/，.h 与 .cc 成对同目录",
    ),
    "R-NAME-001": (SHOULD, NEW, "文件名 ins_temp_<snake>.{h,cc}"),
    "R-NAME-002": (SHOULD, NEW, "类名 InsTemp + 算子 + 拓扑 + 变体"),
    "R-NAME-003": (SHOULD, NEW, "include guard = <文件名 UPPER_SNAKE>_H"),
    "R-NAME-004": (BLOCKER, ALL, ".h 必须有 include guard 且有配对 #endif"),
    "R-NAME-005": (BLOCKER, ALL, ".h 与 .cc 都在 namespace ops_hccl 内"),
    "R-NAME-006": (BLOCKER, ALL, "文件头必须带 CANN Open Software License 版权块"),
    "R-IFACE-001": (
        BLOCKER,
        ALL,
        "必须继承 InsAlgTemplateBase，或继承已有 InsTemp* 做变体",
    ),
    "R-IFACE-002": (
        BLOCKER,
        ALL,
        "带参构造签名 (const OpParam&, const u32, const std::vector<std::vector<u32>>&)",
    ),
    "R-IFACE-003": (
        BLOCKER,
        NEW,
        "保留默认构造 XXX() = default;（走 FastLaunch 的必须）",
    ),
    "R-IFACE-004": (
        BLOCKER,
        ALL,
        "实现三个纯虚：Describe / GetNotifyIdxMainToSub / GetNotifyIdxSubToMain",
    ),
    "R-IFACE-005": (
        BLOCKER,
        ALL,
        "实现三个实质必须：CalcRes / KernelRun / CalcScratchMultiple",
    ),
    "R-IFACE-006": (
        BLOCKER,
        ALL,
        "props 只能写 {.algoType = AlgoType::X}，禁用已删除的 isNhr",
    ),
    "R-IFACE-007": (MUST, ALL, "AlgoType::X 必须在 src/common/alg_parse.h 的枚举里"),
    "R-IFACE-008": (SHOULD, NEW, "实现了 CalcCostCoeff 就声明 props"),
    "R-IFACE-009": (SHOULD, ALL, "props.algoType 的拓扑家族要与类名一致"),
    "R-IFACE-010": (BLOCKER, ALL, "名字带 Dpu 的 template 必须写 REGISTER_TEMPLATE_V2"),
    "R-IFACE-011": (SHOULD, NEW, "非 DPU template 不要写 REGISTER_TEMPLATE_V2"),
    "R-IFACE-012": (BLOCKER, ALL, "禁用 V1 的 REGISTER_TEMPLATE（枚举 key）"),
    "R-IFACE-013": (SHOULD, NEW, "必须有某个 REGISTER_EXEC_V2 把它当模板参数引用"),
    "R-CMAKE-001": (BLOCKER, ALL, "登记进同目录 CMakeLists.txt 的 set(src_list ...)"),
    "R-CMAKE-002": (BLOCKER, ALL, "登记进 src/scatter_aicpu_kernel.cmake"),
    "R-STYLE-001": (MUST, ALL, "单行不超过 120 列"),
    "R-STYLE-002": (MUST, ALL, "4 空格缩进，禁用 tab"),
    "R-STYLE-004": (
        SHOULD,
        NEW,
        "类的私有/保护成员变量用小驼峰 + 尾下划线；POD struct 字段不加",
    ),
    "R-STYLE-005": (MUST, NEW, "常量与宏用 UPPER_SNAKE_CASE"),
    "R-STYLE-006": (SHOULD, NEW, "单个函数不超过 50 行"),
    "R-STYLE-010": (SHOULD, NEW, "日志首参带 [类名][方法名] 前缀"),
    "R-SAFE-001": (BLOCKER, NEWMOD, "channels.at(r) 之前必须有 channels.count(r) 校验"),
    "R-SAFE-002": (BLOCKER, ALL, "禁用 memcpy/strcpy/sprintf/strcat，用 _s 安全版本"),
    "R-IFACE-014": (
        MUST,
        NEWMOD,
        "executor 直接消费 GetRes 时须实现，线程/notify 一致性需人工核对",
    ),
    "R-IFACE-015": (MUST, ALL, "CalcCostCoeff 必须是 static，不是 virtual"),
    "R-STYLE-003": (MUST, NEWMOD, "类名与函数名 PascalCase"),
    "R-STYLE-007": (SHOULD, NEW, "参数超过 6 个就封装成结构体"),
    "R-STYLE-011": (MUST, NEWMOD, "Describe() 一行自述里要带 templateRankSize_"),
    "R-SAFE-003": (
        BLOCKER,
        ALL,
        "返回 HcclResult 的调用都要 CHK_RET / CHK_PRT_RET 包住（return / 赋值除外）",
    ),
    "R-SAFE-004": (BLOCKER, NEWMOD, "局部标量变量声明时就初始化"),
    "R-SAFE-007": (
        BLOCKER,
        NEWMOD,
        "以 templateRankSize_ / threadNum_ 作除数前要有非零守卫",
    ),
    "R-SAFE-010": (MUST, ALL, "禁止对指针变量 sizeof 取数组大小"),
    "R-FLOW-013": (
        BLOCKER,
        ALL,
        "repeatNum 与四个 stride 由 executor 填，template 只读不改",
    ),
    "R-FLOW-022": (
        MUST,
        NEW,
        "scratch 到输出的 LocalCopy 须核对 outBuffType 与 executor 分段落位",
    ),
    "R-DEC-003": (BLOCKER, ALL, "禁止裸 getenv 私有开关，复用仓内既有配置入口"),
    "R-DEC-015": (MUST, ALL, "显式启用目标禁止无条件非空候选（仅 --explicit-only）"),
    "R-DEC-009": (MUST, NEWMOD, "CalcCostCoeff 禁止未经测量的硬编码时间常数"),
}

# 已知返回 HcclResult 的搬运/同步原语（R-SAFE-003）
HCCL_PRIMS = (
    "SendRecvBatchWriteReduce",
    "SendRecvBatchWrite",
    "SendRecvBatchReadReduce",
    "SendRecvBatchRead",
    "SendRecvWriteReduce",
    "SendRecvReadReduce",
    "LocalCopySlices",
    "LocalReduce",
    "LocalCopy",
    "PreSyncInterThreads",
    "PostSyncInterThreads",
)

# executor 填进来、template 只读的字段（R-FLOW-013）
READONLY_FIELDS = (
    "repeatNum",
    "inputRepeatStride",
    "outputRepeatStride",
    "inputSliceStride",
    "outputSliceStride",
)

# 作除数时需要非零守卫的变量（R-SAFE-007）；字面量和查表常量不在此列
RISKY_DIVISORS = ("templateRankSize_", "threadNum_")

SCALARS = r"u8|u16|u32|u64|s32|s64|bool|float|double|size_t"

# 这些类型名开头的成员声明参与 R-STYLE-004 检查（保守清单，避免误报）
MEMBER_TYPES = (
    r"u8|u16|u32|u64|s32|s64|bool|float|double|size_t|std::string|"
    r"HcclDataType|HcclReduceOp|HcclComm|BufferType|std::vector<[^;=]*>"
)

UNSAFE_FUNCS = ("memcpy", "strcpy", "sprintf", "strcat", "strncpy", "vsprintf")

# (方法名, 是否纯虚/必须, 规则 ID)
REQUIRED = [
    ("Describe", True, "R-IFACE-004"),
    ("GetNotifyIdxMainToSub", True, "R-IFACE-004"),
    ("GetNotifyIdxSubToMain", True, "R-IFACE-004"),
    ("CalcRes", False, "R-IFACE-005"),
    ("KernelRun", False, "R-IFACE-005"),
    ("CalcScratchMultiple", False, "R-IFACE-005"),
]


def effective_level(rid, want, strict_new):
    """把「规则等级 + 适用范围 + 是否 --strict-new」映射成本次报告的实际等级。"""
    if want == OK or rid not in RULES:
        return want
    level, scope, _ = RULES[rid]
    if scope == ALL:
        return ERROR if level in (BLOCKER, MUST) else WARN
    # 适用=新增 / 新增+修改：默认只 WARN，把存量欠债和新代码分开
    if strict_new and level in (BLOCKER, MUST):
        return ERROR
    return WARN


class Report(object):
    def __init__(self, title, strict_new=False):
        self.title = title
        self.strict_new = strict_new
        self.items = []
        self.tags = []

    def add(self, want, msg, hint=None, tag=None, rid=None):
        level = effective_level(rid, want, self.strict_new)
        self.items.append((level, rid, msg, hint))
        self.tags.append("%s %s" % (rid, tag or msg) if rid else (tag or msg))

    @property
    def errors(self):
        return sum(1 for lv, _, _, _ in self.items if lv == ERROR)

    @property
    def warns(self):
        return sum(1 for lv, _, _, _ in self.items if lv == WARN)

    @property
    def warn_tags(self):
        return [tag for (lv, _, _, _), tag in zip(self.items, self.tags) if lv == WARN]

    def dump(self):
        print("=" * 78)
        print(self.title)
        print("=" * 78)
        for level, rid, msg, hint in self.items:
            mark = {ERROR: "✗ ERROR", WARN: "! WARN ", OK: "✓ OK   "}[level]
            print("%s  %s%s" % (mark, ("[%s] " % rid) if rid else "", msg))
            if hint:
                for line in hint.splitlines():
                    print("           %s" % line)
        print()


def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 编码风格与安全检查（镜像自 $HCCL 的 .clang-format / AGENTS.md §5 /
# .agents/skills/hccl-review/references/coding-and-security.md）
# ---------------------------------------------------------------------------
def strip_braced_blocks(text):
    """剥掉类体里所有花括号块，只留下声明行（R-STYLE-004 用）。

    一并解决两类误报：
      * 内联函数体里的**局部变量**（如 Describe() 里的 `std::string info`）不是成员；
      * 嵌套 POD `struct` 的字段按仓内约定本就不带尾下划线，属于豁免。
    成员声明本身不含花括号（`std::vector<u32> v{};` 这种被剥成 `... v;`，仍能匹配到名字）。
    """
    out, i, depth, start = [], 0, 0, 0
    for j, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                out.append(text[i:j])
                start = j
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                i = j + 1
    out.append(text[i:] if depth == 0 else text[i:start])
    return "".join(out)


def class_body(text, cls):
    """取出 `class <cls> ... { ... }` 的花括号内容；取不到返回空串。"""
    m = re.search(r"\bclass\s+%s\b[^;{]*\{" % re.escape(cls), text)
    if not m:
        return ""
    depth, j = 0, m.end() - 1
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[m.end():j]
        j += 1
    return ""


def long_functions(cc, cls, limit=50):
    """找出 .cc 里超过 limit 行的成员函数实现（R-STYLE-006）。"""
    hits = []
    for m in re.finditer(
        r"\b%s::(\w+)\s*\([^;]*?\)[^;{]*\{" % re.escape(cls), cc, re.S
    ):
        depth, j = 0, m.end() - 1
        while j < len(cc):
            if cc[j] == "{":
                depth += 1
            elif cc[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        nlines = cc.count("\n", m.start(), j)
        if nlines > limit:
            hits.append((m.group(1), nlines))
    return hits


def strip_comments_and_strings(text):
    """去掉注释与字符串字面量，**保留换行**，这样行号仍然准确。

    不这么做的话，块注释会把行号带偏，报出来的位置根本对不上源码。
    """
    text = re.sub(
        r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", "", m.group(0)), text, flags=re.S
    )
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r'"(?:[^"\\]|\\.)*"', '""', text)


def lineno(text, pos):
    return text.count("\n", 0, pos) + 1


def has_scratch_to_output_copy(code):
    """保守识别 DataSlice(hcclBuff.addr) -> DataSlice(outputPtr) 的 LocalCopy。"""

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
    """只认可执行 scratch->output 拷贝的同一方法里读取 outBuffType。"""
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


def check_semantics(rep, cls, cc, hh):
    """在代码本体上做的语义/安全检查，都跑在去注释去字符串的文本上。"""
    code = strip_comments_and_strings(cc)
    head = strip_comments_and_strings(hh)

    # ---- R-IFACE-015：CalcCostCoeff 必须 static ----
    if "CalcCostCoeff" in head:
        if re.search(r"static\s[\w:<>,\s*&]*\bCalcCostCoeff\s*\(", head):
            rep.add(OK, "CalcCostCoeff 声明为 static", rid="R-IFACE-015")
        else:
            rep.add(
                ERROR,
                "CalcCostCoeff 没有声明为 static",
                "executor 用 InsAlgTemplate::CalcCostCoeff(...) 静态分发调用它；\n"
                "写成 virtual / 普通成员函数不会被调到，cost model 就拿不到系数。",
                tag="CalcCostCoeff 不是 static",
                rid="R-IFACE-015",
            )

    # ---- R-FLOW-013：repeatNum 与四个 stride 只读 ----
    writes = []
    for m in re.finditer(
        r"\.\s*(%s)\s*(?:\+|-|\*|/)?=(?!=)" % "|".join(READONLY_FIELDS), code
    ):
        writes.append("%s (第 %d 行)" % (m.group(1), lineno(code, m.start())))
    if writes:
        rep.add(
            ERROR,
            "给 executor 填的只读字段赋了值：%s" % "、".join(writes[:4]),
            "repeatNum / inputRepeatStride / outputRepeatStride / inputSliceStride / outputSliceStride\n"
            "由 executor 每轮 loop 填入，template 只读。改它会让上层的地址推导与实际搬运脱节。",
            tag="给只读的 repeatNum/stride 赋值",
            rid="R-FLOW-013",
        )

    # DPU 与 OmniPipe 使用各自的 step 布局；仅凭两个 base offset 无法判断回搬语义。
    ordinary_aicpu = not re.search(r"Dpu|OmniPipe", cls, re.I)
    if ordinary_aicpu and output_copy_without_guard(code, cls):
        rep.add(
            WARN,
            "scratch/输出间有 LocalCopy，却未读取 outBuffType",
            "先逐段核对绑定 executor 的 outBuffType、指针、base offset 与 repeat stride。\n"
            "HCCL_BUFFER 段只有在数据已在目标槽位时才能跳过回搬；错位仍须搬移并写明理由。\n"
            "本检查只提示遗漏，不能证明分流分支正确。",
            tag="scratch 回搬缺少 outBuffType 分流",
            rid="R-FLOW-022",
        )

    # ---- R-SAFE-003：原语调用要被 CHK_RET 包住 ----
    #  豁免两种正确写法：`return Prim(...)`（直接把错误码传给调用方）、`ret = Prim(...)`（随后自己判）
    bare = []
    for prim in HCCL_PRIMS:
        for m in re.finditer(r"(?<![\w:.>])%s\s*\(" % prim, code):
            start = max(code.rfind(c, 0, m.start()) for c in ";{}")
            span = code[start + 1:m.start()]
            if "CHK_RET" in span or "CHK_PRT_RET" in span:
                continue
            if re.search(r"(?:return|=)\s*$", span):  # return Prim(...) / x = Prim(...)
                continue
            bare.append("%s (第 %d 行)" % (prim, lineno(code, m.start())))
    if bare:
        rep.add(
            ERROR,
            "原语调用没被 CHK_RET/CHK_PRT_RET 包住：%s" % "、".join(bare[:4]),
            "这些原语返回 HcclResult，吞掉失败会让错误在下一层以更难懂的形式爆出来。\n"
            "（`return Prim(...)` 和 `ret = Prim(...)` 是正确写法，本检查已豁免。）",
            tag="原语调用没被 CHK_RET 包住",
            rid="R-SAFE-003",
        )

    # ---- R-SAFE-004：局部标量声明即初始化 ----
    uninit = []
    for m in re.finditer(
        r"^[ \t]+(?:const\s+)?(?:%s)\s+(\w+)\s*;" % SCALARS, code, re.M
    ):
        uninit.append("%s (第 %d 行)" % (m.group(1), lineno(code, m.start())))
    if uninit:
        rep.add(
            WARN,
            "局部标量声明时未初始化：%s" % "、".join(uninit[:4]),
            "即使紧跟着当出参传出去，也按仓内习惯写 `u32 x = 0;`——\n"
            "出参函数提前返回时，未初始化的值会被后面的代码读到。",
            tag="局部标量未初始化",
            rid="R-SAFE-004",
        )

    # ---- R-SAFE-007：风险除数的非零守卫 ----
    unguarded = []
    for v in RISKY_DIVISORS:
        if not re.search(r"[/%%]\s*%s\b" % re.escape(v), code):
            continue
        guard = re.search(
            r"%s\s*(?:==|!=|<=?|>=?)\s*[012]\b|[012]\s*(?:<|<=|>|>=)\s*%s\b"
            % (re.escape(v), re.escape(v)),
            code,
        )
        if not guard:
            unguarded.append(v)
    if unguarded:
        rep.add(
            WARN,
            "拿 %s 作除数但全文没有非零守卫" % "、".join(unguarded),
            "切片计算里除零会直接崩。加一个早退：\n"
            "    if (templateRankSize_ <= 1) { ... return HCCL_SUCCESS; }\n"
            "满足 R-FLOW-020 的早退分支通常也就满足了这一条。",
            tag="风险除数缺非零守卫",
            rid="R-SAFE-007",
        )

    # ---- R-DEC-003：禁止绕开仓内配置体系的裸 getenv ----
    envs = []
    for m in re.finditer(r"\b(?:std::)?getenv\s*\(\s*([^)]*)\)", code + "\n" + head):
        envs.append(m.group(1).strip()[:40] or "?")
    if envs:
        rep.add(
            ERROR,
            "直接用 getenv 读环境变量：%s" % "、".join(envs[:3]),
            "新算法一律复用仓内既有配置入口（HCCL_ALGO 等），不要新造私有开关——\n"
            "私有开关是「最容易触发」的路子，但它绕开配置解析、没有文档和默认值，\n"
            "也让同一个 skill 两次生成出互不兼容的启用方式。\n"
            "确实需要新环境变量：补配置定义、解析、默认值、文档与测试后再用。",
            tag="裸 getenv 私有开关",
            rid="R-DEC-003",
        )

    # ---- R-DEC-009：CalcCostCoeff 里的硬编码时间常数 ----
    m = re.search(r"CalcCostCoeff\s*\([^)]*\)\s*\{", code)
    if m:
        depth, j = 0, m.end() - 1
        while j < len(code):
            if code[j] == "{":
                depth += 1
            elif code[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = code[m.end():j]
        lits = []
        for lit in re.finditer(
            r"(?<![\w.])(\d+\.\d*(?:[eE]-?\d+)?f?|\d+[eE]-?\d+f?)", body
        ):
            v = lit.group(1)
            if re.match(r"^(?:0\.0*|1\.0*)f?$", v):  # 0.0f / 1.0f 初始化，正常
                continue
            lits.append(v)
        # 检查被引用的命名常量，不能靠把 1e9 移到函数外绕过检查。
        for decl in re.finditer(
            r"\b(?:float|double)\s+(\w+)\s*=\s*(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[fF]?)\s*;",
            code + "\n" + head,
        ):
            name, value = decl.groups()
            if re.search(r"\b" + re.escape(name) + r"\b", body) and float(
                value.rstrip("fF")
            ) not in (0.0, 1.0):
                lits.append(name + "=" + value)
        if lits:
            rep.add(
                WARN,
                "CalcCostCoeff 里有硬编码浮点常数：%s"
                % "、".join(sorted(set(lits))[:5]),
                "时间/延迟系数要走 CostModelManager 的既有接口\n"
                "（CalcLatencyParams / CalcLaunchParams / CalcMeshParam / CalcNHRParams），\n"
                "不要自己填 0.000005f 这类没有测量依据的常数——换台机器就不成立。\n"
                "命名常量仍须核对真实来源；巨额惩罚不能代替显式配置守卫。",
                tag="CalcCostCoeff 硬编码常数",
                rid="R-DEC-009",
            )

    # ---- R-SAFE-010：sizeof 用在变量上 ----
    sizeofs = []
    for m in re.finditer(r"\bsizeof\s*\(\s*([A-Za-z_]\w*)\s*\)", code + "\n" + head):
        arg = m.group(1)
        # 只有能找到指针声明才报错；合法的 sizeof(scalar/array) 不能仅凭小写名字判错。
        if re.search(
            r"\b[\w:<>]+\s*\*\s*(?:const\s+)?%s\b" % re.escape(arg), code + "\n" + head
        ):
            sizeofs.append(arg)
    if sizeofs:
        rep.add(
            ERROR,
            "对变量取 sizeof：%s" % "、".join(sorted(set(sizeofs))[:4]),
            "对指针变量 sizeof 拿到的是指针宽度不是数据长度；容器对象 sizeof 也不等于数据大小。\n"
            "要长度就用显式的 count * DATATYPE_SIZE_TABLE[dataType_]。",
            tag="对变量取 sizeof",
            rid="R-SAFE-010",
        )


def check_style_and_safety(rep, cls, cc_path, h_path, cc, hh):
    # ---- R-STYLE-001 / 002：行宽与缩进（.clang-format 兜底，这里做门禁）----
    over, tabbed = [], []
    for name, text in (("h", hh), ("cc", cc)):
        for i, line in enumerate(text.splitlines(), 1):
            if len(line) > 120:
                over.append("%s:%d (%d 列)" % (name, i, len(line)))
            if line.startswith("\t") or re.match(r"^ *\t", line):
                tabbed.append("%s:%d" % (name, i))
    if over:
        rep.add(
            ERROR,
            "有 %d 行超过 120 列：%s" % (len(over), "、".join(over[:4])),
            "以 $HCCL/.clang-format 为准；pre-commit 会强行折回去，先自己折好免得 diff 变脏。",
            tag="超过 120 列",
            rid="R-STYLE-001",
        )
    else:
        rep.add(OK, "行宽 ≤ 120 列", rid="R-STYLE-001")
    if tabbed:
        rep.add(
            ERROR,
            "有 %d 行用了 tab 缩进：%s" % (len(tabbed), "、".join(tabbed[:4])),
            "统一 4 空格。",
            tag="tab 缩进",
            rid="R-STYLE-002",
        )

    # ---- R-STYLE-004：类成员尾下划线（跳过 POD struct 字段）----
    body = strip_braced_blocks(class_body(hh, cls))
    bad_members = []
    for m in re.finditer(
        r"^\s+(?:const\s+|static\s+|mutable\s+)*(?:%s)\s+(\w+)\s*(?:=[^;]*)?;"
        % MEMBER_TYPES,
        body,
        re.M,
    ):
        if not m.group(1).endswith("_"):
            bad_members.append(m.group(1))
    if bad_members:
        rep.add(
            WARN,
            "类成员未用尾下划线：%s" % "、".join(bad_members[:6]),
            "仓内约定：类的私有/保护成员小驼峰 + 尾下划线（myRank_）；\n"
            "POD struct 的字段不加下划线，本检查已跳过 struct。",
            tag="类成员缺尾下划线",
            rid="R-STYLE-004",
        )

    # ---- R-STYLE-005：宏与文件级常量 UPPER_SNAKE ----
    bad_const = []
    for m in re.finditer(r"^\s*#define\s+([A-Za-z_]\w*)", hh + "\n" + cc, re.M):
        name = m.group(1)
        if name != name.upper():
            bad_const.append("#define %s" % name)
    for m in re.finditer(
        r"^(?:static\s+)?constexpr\s+\w[\w:<>]*\s+([A-Za-z_]\w*)\s*=", cc, re.M
    ):
        if m.group(1) != m.group(1).upper():
            bad_const.append("constexpr %s" % m.group(1))
    if bad_const:
        rep.add(
            WARN,
            "常量/宏未用 UPPER_SNAKE_CASE：%s" % "、".join(bad_const[:4]),
            "$HCCL/AGENTS.md §5：常量与宏均 UPPER_SNAKE_CASE。",
            tag="常量/宏未用 UPPER_SNAKE",
            rid="R-STYLE-005",
        )

    # ---- R-STYLE-006：函数长度 ----
    longs = long_functions(cc, cls)
    if longs:
        rep.add(
            WARN,
            "函数超过 50 行：%s"
            % "、".join("%s(%d 行)" % (n, ln) for n, ln in longs[:4]),
            "KernelRun 常常超标，拆成 RunIntra / RunInter / DoLocalReduce 更好评审。",
            tag="函数超过 50 行",
            rid="R-STYLE-006",
        )

    # ---- R-STYLE-010：日志前缀 ----
    no_prefix = [
        m.group(1)[:32]
        for m in re.finditer(
            r'HCCL_(?:ERROR|WARNING|INFO|DEBUG|RUN_INFO)\s*\(\s*"([^"[][^"]*)', cc
        )
    ]
    if no_prefix:
        rep.add(
            WARN,
            '有 %d 条日志没带 [类名][方法名] 前缀，如 "%s"'
            % (len(no_prefix), no_prefix[0]),
            '仓内绝大多数日志形如 HCCL_INFO("[%s][KernelRun] ...")；\n'
            "不带前缀的日志在多 template 并发时无法定位。" % cls,
            tag="日志缺 [类][方法] 前缀",
            rid="R-STYLE-010",
        )

    # ---- R-STYLE-003：函数名 PascalCase（排除 std:: 这类限定名）----
    code_ns = strip_comments_and_strings(cc)
    bad_fn = []
    for m in re.finditer(
        r"\b([A-Z]\w+)::(\w+)\s*\([^;{]*?\)\s*(?:const\s*)?\{", code_ns, re.S
    ):
        if not re.match(r"^[A-Z][A-Za-z0-9]*$", m.group(2)):
            bad_fn.append(m.group(2))
    if bad_fn:
        rep.add(
            WARN,
            "函数名不是 PascalCase：%s" % "、".join(sorted(set(bad_fn))[:5]),
            "$HCCL/AGENTS.md §5：类/函数 PascalCase。",
            tag="函数名不是 PascalCase",
            rid="R-STYLE-003",
        )

    # ---- R-STYLE-007：参数个数 ----
    fat = []
    for m in re.finditer(
        r"\b%s::(\w+)\s*\(([^;{]*?)\)\s*(?:const\s*)?\{" % re.escape(cls), code_ns, re.S
    ):
        n = len([a for a in m.group(2).split(",") if a.strip()])
        if n > 6:
            fat.append("%s(%d 个)" % (m.group(1), n))
    if fat:
        rep.add(
            WARN,
            "参数超过 6 个：%s" % "、".join(fat[:4]),
            "封装成结构体（仓内 SendRecvInfo / DataSlice 就是这么做的）。",
            tag="函数参数超过 6 个",
            rid="R-STYLE-007",
        )

    # ---- R-STYLE-011：Describe() 要带 templateRankSize_ ----
    m = re.search(
        r"Describe\s*\([^)]*\)[^{]*\{(.*?)\n\s*\}",
        strip_comments_and_strings(cc) + "\n" + strip_comments_and_strings(hh),
        re.S,
    )
    if m and "templateRankSize_" not in m.group(1):
        rep.add(
            WARN,
            "Describe() 的自述里没有 templateRankSize_",
            "Describe() 是运行期唯一的自我标识，带上 rank 数才能区分同一算法的不同实例。",
            tag="Describe() 缺 templateRankSize_",
            rid="R-STYLE-011",
        )

    # ---- R-SAFE-001：channels 裸 at() ----
    if re.search(r"channels\.at\(", cc) and "channels.count(" not in cc:
        rep.add(
            WARN,
            "出现 channels.at() 但全文没有 channels.count() 校验",
            "远端 rank 不在 channels 里时 .at() 会抛异常；先 CHK_PRT_RET(channels.count(r) == 0, ...)。",
            tag="channels.at() 缺 count() 校验",
            rid="R-SAFE-001",
        )

    # ---- R-SAFE-002：禁用函数 ----
    unsafe = []
    for name, text in (("h", hh), ("cc", cc)):
        for fn in UNSAFE_FUNCS:
            if re.search(r"(?<![\w_])%s\s*\(" % fn, text):
                unsafe.append("%s(.%s)" % (fn, name))
    if unsafe:
        rep.add(
            ERROR,
            "使用了禁用函数：%s" % "、".join(sorted(set(unsafe))),
            "改用 _s 安全版本（memcpy_s / strcpy_s / sprintf_s）并检查返回值。\n"
            "CANN 编码规范红线，codecheck 阶段会拦。",
            tag="使用了禁用的不安全函数",
            rid="R-SAFE-002",
        )


def check_explicit_only(rep, cls, cc, hh):
    """Conservative local anti-pattern check, never a proof of configuration semantics."""
    code = strip_comments_and_strings(cc) + "\n" + strip_comments_and_strings(hh)
    pattern = r"\b" + re.escape(cls) + r"::CalcCostCoeff\s*\([^)]*\)\s*\{"
    match = re.search(pattern, code)
    if not match:
        # Inline definitions are scoped to the requested class, not other headers' classes.
        code = class_body(strip_comments_and_strings(hh), cls)
        match = re.search(r"\bCalcCostCoeff\s*\([^)]*\)\s*\{", code)
    body = ""
    if match:
        depth, end = 1, match.end()
        while end < len(code) and depth:
            depth += (code[end] == "{") - (code[end] == "}")
            end += 1
        if depth == 0:
            body = code[match.end():end - 1]
    returns = list(re.finditer(r"\breturn\s+([^;]+);", body))
    definite = False
    if len(returns) == 1 and not body[returns[0].end():].strip():
        result = returns[0].group(1).strip()
        prefix = body[: returns[0].start()]
        # The returned initializer is visibly nonempty, at function-body scope.
        level = prefix.count("{") - prefix.count("}")
        if level == 0 and (not prefix.strip() or prefix.rstrip().endswith((";", "}"))):
            definite = bool(re.fullmatch(r"\{\s*\{[^{}]+\}\s*\}", result))
            if re.fullmatch(r"[A-Za-z_]\w*", result):
                # Match an unconditional final insertion immediately before return.
                insertion = re.search(
                    r"\b"
                    + re.escape(result)
                    + r"\.push_back\s*\(\s*\{[^{}]+\}\s*\)\s*;\s*$",
                    prefix,
                )
                if insertion:
                    before = prefix[: insertion.start()]
                    definite = (
                        (before.count("{") == before.count("}"))
                        and (not before.strip() or before.rstrip().endswith((";", "}")))
                        and bool(
                            re.search(
                                r"std::vector\s*<\s*CostModelParam\s*>\s+"
                                + re.escape(result)
                                + r"\s*;",
                                before,
                            )
                        )
                    )
        # Preprocessor branches / return macros are outside this local check's model.
        if "#" in body or re.search(r"\b[A-Z][A-Z0-9_]+\s*\(", body):
            definite = False
    if definite:
        rep.add(
            ERROR,
            "显式启用目标的 CalcCostCoeff 存在无条件非空返回",
            "接口推导系数也会进入默认候选竞争。先沿现有配置解析链建立显式匹配守卫；"
            "空配置、其他算法配置和禁用路径应返回空候选。",
            rid="R-DEC-015",
        )
    else:
        # Deliberately no MUST rule mapping: uncertainty is a review, not a proven defect.
        rep.add(
            WARN,
            "R-DEC-015 人工复核待完成：未证明显式候选守卫生效",
            "这里只识别局部无条件非空返回。存在 if、空返回或函数委派均不证明语义正确；"
            "逐项核对配置来源、目标匹配、空配置、其他算法与禁用路径，并保留审查证据。",
            tag="R-DEC-015 显式守卫需人工复核",
        )


def check_one(repo, cc_path, strict_new=False, explicit_only=False):
    base, ext = os.path.splitext(cc_path)
    h_path = base + ".h"
    filebase = os.path.basename(base)
    rel = os.path.relpath(cc_path, repo)
    rep = Report(rel, strict_new=strict_new)

    if not os.path.exists(cc_path):
        rep.add(ERROR, ".cc 不存在: %s" % cc_path)
        return rep
    if not os.path.exists(h_path):
        rep.add(ERROR, ".h 不存在: %s" % h_path, ".h 与 .cc 必须成对同目录。")
        return rep

    cc, hh = read(cc_path), read(h_path)

    # 从路径推算子名
    m = re.search(
        r"src/ops/([^/]+)/(?:[^/]+/)?template/aicpu/", rel.replace(os.sep, "/")
    )
    if not m:
        rep.add(
            WARN,
            "路径不在 src/ops/{op}/[algorithm/]template/aicpu/ 下，跳过 CMake 检查",
            rid="R-PATH-003",
        )
        op = None
    else:
        op = m.group(1)

    # ---------- 命名 ----------
    if filebase.startswith("ins_temp_"):
        rep.add(OK, "文件名前缀 ins_temp_", rid="R-NAME-001")
    else:
        rep.add(
            WARN,
            "文件名未以 ins_temp_ 开头: %s" % filebase,
            "仓内 reduce/ 下有少量历史遗留不带前缀的文件，新文件请遵循 ins_temp_*",
            tag="文件名未以 ins_temp_ 开头",
            rid="R-NAME-001",
        )

    cls_m = re.search(r"class\s+(\w+)\s*:\s*public\s+(\w+)", hh)
    if not cls_m:
        rep.add(ERROR, ".h 里找不到 `class X : public Y` 定义", rid="R-IFACE-001")
        return rep
    cls, parent = cls_m.group(1), cls_m.group(2)
    rep.add(OK, "类 %s : public %s" % (cls, parent))

    if not cls.startswith("InsTemp"):
        rep.add(
            WARN,
            "类名未以 InsTemp 开头: %s" % cls,
            tag="类名未以 InsTemp 开头",
            rid="R-NAME-002",
        )

    derived_from_base = parent == "InsAlgTemplateBase"
    if not derived_from_base and not parent.startswith(("InsTemp", "Reduce")):
        rep.add(
            ERROR,
            "父类 %s 不是 InsAlgTemplateBase 也不是已有 template 类" % parent,
            "AICPU template 必须继承 InsAlgTemplateBase（V2），或继承一个已有的 InsTemp* 做变体。\n"
            "若继承的是 AlgTemplateBase，那是 V1 遗留 API，新代码不要用。",
            rid="R-IFACE-001",
        )
    elif not derived_from_base:
        rep.add(
            OK,
            "变体模式：继承已有 template %s，只覆写差异方法" % parent,
            rid="R-IFACE-001",
        )

    # ---------- 头文件卫士 ----------
    guards = re.findall(r"#ifndef\s+(\w+)", hh)
    if not guards:
        rep.add(ERROR, ".h 缺少 include guard", rid="R-NAME-004")
    else:
        expect = filebase.upper() + "_H"
        if guards[0] != expect:
            rep.add(
                WARN,
                "include guard 为 %s，建议 %s" % (guards[0], expect),
                tag="include guard 与文件名不一致",
                rid="R-NAME-003",
            )
        else:
            rep.add(OK, "include guard %s" % guards[0], rid="R-NAME-003")
    if "#endif" not in hh:
        rep.add(ERROR, ".h 缺少 #endif", rid="R-NAME-004")

    # ---------- 版权头 ----------
    for name, text in (("h", hh), ("cc", cc)):
        if "CANN Open Software License Agreement" not in text[:1200]:
            rep.add(
                ERROR,
                ".%s 缺少 CANN Open Software License 版权头" % name,
                "pre-commit 的 OAT 合规检查会拦下来。",
                rid="R-NAME-006",
            )

    # ---------- namespace ----------
    for name, text in (("h", hh), ("cc", cc)):
        if "namespace ops_hccl" not in text:
            rep.add(ERROR, ".%s 缺少 namespace ops_hccl" % name, rid="R-NAME-005")

    # ---------- 构造函数 ----------
    if re.search(r"%s\(\)\s*=\s*default" % re.escape(cls), hh):
        rep.add(OK, "保留了默认构造 %s() = default" % cls, rid="R-IFACE-003")
    else:
        rep.add(
            WARN,
            "没有默认构造 `%s() = default;`" % cls,
            "executor 的 FastLaunch 路径用 make_unique<InsAlgTemplate>() 无参构造；\n"
            "走 FastLaunch 的 executor（如各 Sole executor）绑定的 template 必须有它，否则编译失败。\n"
            "仓内 DPU / Intra / Inter / OmniPipe 这类不走 FastLaunch 的 template 确实没有，可忽略。",
            tag="没有默认构造 = default",
            rid="R-IFACE-003",
        )

    if re.search(r"const\s+OpParam&\s*\w+,\s*const\s+u32\s+\w+", hh, re.S):
        rep.add(
            OK,
            "带参构造签名匹配 (const OpParam&, const u32, const std::vector<std::vector<u32>>&)",
            rid="R-IFACE-002",
        )
    else:
        rep.add(
            ERROR,
            "带参构造签名不匹配",
            "executor 硬编码调用 make_shared<T>(param, rankId, subCommRanks)，签名必须一致。",
            rid="R-IFACE-002",
        )

    # ---------- 必须实现的方法 ----------
    for method, is_pure, rid in REQUIRED:
        in_h = re.search(r"\b%s\s*\(" % method, hh) is not None
        in_cc = re.search(r"\b%s::%s\s*\(" % (re.escape(cls), method), cc) is not None
        inline_in_h = (
            in_h
            and re.search(r"\b%s\s*\([^;]*\)[^;]*\{" % method, hh, re.S) is not None
        )
        if in_cc or inline_in_h:
            rep.add(OK, "实现了 %s()" % method, rid=rid)
        elif not derived_from_base:
            rep.add(OK, "%s() 继承自 %s" % (method, parent), rid=rid)
        elif is_pure:
            rep.add(
                ERROR,
                "缺少纯虚方法 %s() 的实现" % method,
                "InsAlgTemplateBase 里它是 = 0，不实现编译不过。",
                rid=rid,
            )
        else:
            rep.add(
                ERROR,
                "缺少 %s() 的实现" % method,
                (
                    "基类默认返回 0；本 Skill 要求显式声明 scratch 预算，不使用时返回 0。"
                    if method == "CalcScratchMultiple"
                    else "基类默认实现直接 HCCL_ERROR 返回失败，运行期必然报错。"
                ),
                rid=rid,
            )

    # ---------- TemplateProp / AlgoType ----------
    #  当前 TemplateProp 只有 algoType 一个成员（op_common 的 common_alg_template_base.h）。
    #  历史上的 isNhr 已被删除，仍这么写会编译失败。枚举值从 src/common/alg_parse.h 实时解析。
    known = load_algo_types(repo)
    want = infer_algo_type(cls, known)
    declared = re.search(r"TemplateProp\s+props\s*=\s*\{([^}]*)\}", hh)
    if "isNhr" in hh:
        rep.add(
            ERROR,
            "用了已删除的 `TemplateProp::isNhr`",
            "当前 TemplateProp 只有 algoType 成员，designated initializer 指向不存在的成员会编译失败。\n"
            "改成：static constexpr TemplateProp props = {.algoType = AlgoType::%s};"
            % (want or "<枚举值>"),
            rid="R-IFACE-006",
        )
    elif declared:
        m = re.search(r"AlgoType::(\w+)", declared.group(1))
        if not m:
            rep.add(
                ERROR,
                "props 没有按 `{.algoType = AlgoType::XXX}` 的形式写",
                "当前 TemplateProp 只有 algoType 一个成员。",
                rid="R-IFACE-006",
            )
        elif known and m.group(1) not in known:
            rep.add(
                ERROR,
                "AlgoType::%s 不在 src/common/alg_parse.h 的枚举里" % m.group(1),
                "可选值：%s" % "、".join(sorted(known)),
                rid="R-IFACE-007",
            )
        elif want and m.group(1) != want:
            rep.add(
                WARN,
                "类名推断出 AlgoType::%s，props 却声明 AlgoType::%s"
                % (want, m.group(1)),
                "算法身份是一个原子对象（R-DEC-005）：类名 / 文件名 / props.algoType /\n"
                "DSL token / DFX 解析必须同名同型。对不上通常是复制粘贴漏改——\n"
                "「Ring 类却填 AlgoType::MESH」这类错误就是这么来的。\n"
                "确实要用不同家族的，把理由写进 spec 第 1 章。",
                tag="props.algoType 与类名不符",
                rid="R-IFACE-009",
            )
        else:
            rep.add(
                OK, "props = {.algoType = AlgoType::%s}" % m.group(1), rid="R-IFACE-006"
            )
    elif want and "CalcCostCoeff" in hh + cc:
        rep.add(
            WARN,
            "实现了 CalcCostCoeff 却没有声明 `static constexpr TemplateProp props`",
            "建议按类名补上：static constexpr TemplateProp props = {.algoType = AlgoType::%s};\n"
            "（props 目前仓内还没有消费者，不写不会报错；但参与选路的 template 普遍都声明了。）"
            % want,
            tag="缺少 TemplateProp props（已实现 CalcCostCoeff）",
            rid="R-IFACE-008",
        )

    # ---------- DPU 字符串注册 ----------
    is_dpu = "dpu" in filebase.lower() or "Dpu" in cls or "DPU" in cls
    has_reg = "REGISTER_TEMPLATE_V2" in cc
    if is_dpu and not has_reg:
        rep.add(
            ERROR,
            "DPU template 缺少 REGISTER_TEMPLATE_V2 注册",
            "dpu/kernel_launch.cc 按字符串反查，需在 .cc 末尾加：\n"
            'REGISTER_TEMPLATE_V2("%s", %s);' % (cls, cls),
            rid="R-IFACE-010",
        )
    elif is_dpu and has_reg:
        rep.add(OK, "DPU template 已做 REGISTER_TEMPLATE_V2 注册", rid="R-IFACE-010")
    elif not is_dpu and has_reg:
        rep.add(
            WARN,
            "非 DPU template 写了 REGISTER_TEMPLATE_V2，通常没必要",
            "AICPU template 由 executor 通过 C++ 模板参数静态绑定，不走字符串注册表。",
            tag="非 DPU template 写了 REGISTER_TEMPLATE_V2",
            rid="R-IFACE-011",
        )

    if "REGISTER_TEMPLATE(" in cc:
        rep.add(
            ERROR,
            "使用了 V1 的 REGISTER_TEMPLATE（枚举 key）",
            "V2 template 请用 REGISTER_TEMPLATE_V2（且 AICPU 侧一般不需要注册）。",
            rid="R-IFACE-012",
        )

    # ---------- 编码风格与安全 ----------
    check_style_and_safety(rep, cls, cc_path, h_path, cc, hh)
    check_semantics(rep, cls, cc, hh)
    if explicit_only:
        check_explicit_only(rep, cls, cc, hh)

    # ---------- 两处 CMake 接线 ----------
    if op:
        local_cmake = os.path.join(os.path.dirname(cc_path), "CMakeLists.txt")
        kernel_cmake = os.path.join(repo, "src/scatter_aicpu_kernel.cmake")

        if os.path.exists(local_cmake):
            if has_source(
                read(local_cmake),
                "${CMAKE_CURRENT_SOURCE_DIR}/%s.cc" % filebase,
                "host",
            ):
                rep.add(
                    OK,
                    "已登记到 %s（host libhccl.so）"
                    % os.path.relpath(local_cmake, repo),
                    rid="R-CMAKE-001",
                )
            else:
                rep.add(
                    ERROR,
                    "未登记到 %s" % os.path.relpath(local_cmake, repo),
                    "加到 set(src_list ...) 里：\n"
                    "    ${CMAKE_CURRENT_SOURCE_DIR}/%s.cc" % filebase,
                    rid="R-CMAKE-001",
                )
        else:
            rep.add(ERROR, "找不到 %s" % local_cmake, rid="R-CMAKE-001")

        if os.path.exists(kernel_cmake):
            needle = src_rel(repo, cc_path)
            if has_source(
                read(kernel_cmake), "${CMAKE_CURRENT_SOURCE_DIR}/" + needle, "device"
            ):
                rep.add(
                    OK,
                    "已登记到 src/scatter_aicpu_kernel.cmake（device libscatter_aicpu_kernel.so）",
                    rid="R-CMAKE-002",
                )
            else:
                rep.add(
                    ERROR,
                    "未登记到 src/scatter_aicpu_kernel.cmake",
                    "★ 最高频接线 bug：只加了 host 侧，AICPU 侧运行时找不到符号。\n"
                    "加到 add_library(scatter_aicpu_kernel SHARED ...) 块里：\n"
                    "    ${CMAKE_CURRENT_SOURCE_DIR}/%s" % needle,
                    rid="R-CMAKE-002",
                )
        else:
            rep.add(ERROR, "找不到 %s" % kernel_cmake, rid="R-CMAKE-002")

    # ---------- executor 是否用上了它 ----------
    hits = []
    resource_consumers = []
    for path in globs(repo, "executor/*.cc"):
        text = read(path)
        if re.search(
            r"(?:REGISTER_EXEC_V2(?:_MULTI)?|REGISTER_EXECUTOR_BY_(?:TWO|FOUR)_TEMPS)"
            r"[^;]*\b%s\b" % re.escape(cls),
            strip_comments_and_strings(text),
            re.S,
        ):
            hits.append(os.path.relpath(path, repo))
            if re.search(r"(?:\.|->)\s*GetRes\s*\(", strip_comments_and_strings(text)):
                resource_consumers.append(os.path.relpath(path, repo))
    if resource_consumers and derived_from_base:
        implemented = re.search(r"\b%s::GetRes\s*\(" % re.escape(cls), cc) or re.search(
            r"\bGetRes\s*\([^;]*\)[^;]*\{", hh, re.S
        )
        if not implemented:
            rep.add(
                ERROR,
                "executor 消费 GetRes，但模板未实现：%s"
                % ", ".join(resource_consumers),
                "基类 GetRes 返回错误且不填出参。补实现并核对与 CalcRes 的线程/notify 一致；"
                "GetThreadNum 按实际消费者检查。",
                rid="R-IFACE-014",
            )
    if hits:
        rep.add(
            OK, "被 executor 引用：%s" % ", ".join(sorted(set(hits))), rid="R-IFACE-013"
        )
    elif not is_dpu:
        rep.add(
            WARN,
            "没有任何 REGISTER_EXEC_V2 引用这个类",
            "template 只有被 executor 当模板参数注册后才会真正跑起来。\n"
            "在对应 executor 的 .cc 末尾加：\n"
            "REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_XXX, <算法名>, <Executor>, <TopoMatch>, %s);"
            % cls,
            tag="没有任何 REGISTER_EXEC_V2 引用这个类",
            rid="R-IFACE-013",
        )

    return rep


# ---------------------------------------------------------------------------
# 目录布局探测：HCCL 重构过一次（<op>/template → <op>/algorithm/template）。
# 这里实时探测，两种布局都支持，不写死任何一种。
# ---------------------------------------------------------------------------
def globs(repo, tail):
    hits = []
    for pat in (
        os.path.join(repo, "src/ops/*", tail),
        os.path.join(repo, "src/ops/*/*", tail),
    ):
        hits.extend(glob.glob(pat, recursive=True))
    return sorted(set(hits))


def src_rel(repo, path):
    """相对 src/ 的路径 —— scatter_aicpu_kernel.cmake 里登记的就是这个形式。"""
    return os.path.relpath(path, os.path.join(repo, "src")).replace(os.sep, "/")


def load_algo_types(repo):
    """从 $HCCL/src/common/alg_parse.h 实时解析 AlgoType 枚举，**不写死清单**。"""
    path = os.path.join(repo, "src/common/alg_parse.h")
    if not os.path.exists(path):
        return set()
    m = re.search(r"enum\s+class\s+AlgoType\s*:[^{]*\{(.*?)\}", read(path), re.S)
    if not m:
        return set()
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return {v.strip() for v in body.split(",") if v.strip()}


def infer_algo_type(cls, known):
    """按类名推断该写哪个 AlgoType；推不出返回 None。

    **枚举值从 alg_parse.h 实时解析，这里不写死任何拓扑名**（`R-PATH-004`）。
    做法：把枚举名按 `_` 拆成词元，全部出现在类名里才算命中，命中字符最多的那个最具体。
      InsTempAllReduceMesh1DOneShot        -> MESH_ONESHOT（MESH+ONESHOT 比单个 MESH 更具体）
      InsTempReduceScatterAicpuReduceNhr   -> NHR_AICPU_REDUCE
      InsTempAllGatherRing1D               -> RING（仓里加了 AlgoType::RING 就自动生效）
    写死 if 链的老做法会让**枚举里新增的拓扑一律推断不出**，
    于是 `R-IFACE-009`（props 与类名是否一致）被静默跳过——
    「Ring 类却填 AlgoType::MESH」正是这么漏过去的。
    """
    u = re.sub(r"[^A-Z0-9]", "", cls.upper())
    best, score = None, 0
    for enum in sorted(known or ()):
        if enum == "UNKNOWN":
            continue
        words = [w for w in enum.split("_") if w]
        if not words or not all(w in u for w in words):
            continue
        s = sum(len(w) for w in words)
        if s > score:
            best, score = enum, s
    return best


def print_rules(rid=None):
    """打印规则卡；权威定义在 references/07-rules.md。"""
    if rid:
        key = rid.upper()
        if key not in RULES:
            print("没有这条规则：%s" % rid)
            print("已登记（本脚本能自动检查的部分）：%s" % "、".join(sorted(RULES)))
            print("完整规则表见 references/07-rules.md。")
            return 1
        level, scope, text = RULES[key]
        behav = (
            "适用=全量 → %s" % ("ERROR" if level in (BLOCKER, MUST) else "WARN")
            if scope == ALL
            else "适用=%s → 默认 WARN；--strict-new 下 %s"
            % (scope, "ERROR" if level in (BLOCKER, MUST) else "WARN")
        )
        print("%s  [%s]  适用: %s" % (key, level, scope))
        print("  规则: %s" % text)
        print("  本脚本: %s" % behav)
        print("  详情（理由 / 反例 / 出处）: references/07-rules.md")
        return 0
    print("check_template.py 能自动检查的规则（完整表见 references/07-rules.md）：")
    print()
    for key in sorted(RULES):
        level, scope, text = RULES[key]
        print("  %-13s %-8s %-8s %s" % (key, level, scope, text))
    print()
    print("等级映射：适用=全量 时 BLOCKER/MUST → ERROR、SHOULD → WARN；")
    print(
        "          适用=新增/新增+修改 时默认 WARN，--strict-new 下 BLOCKER/MUST → ERROR。"
    )
    return 0


def resolve_repo(arg):
    """解析 HCCL 仓根目录。

    路径只能来自调用方：`--repo` 参数，或环境变量 `HCCL_REPO`。
    **脚本里不带任何默认路径**——缺失时直接返回输入错误，不猜、不扫盘。
    """
    repo = arg or os.environ.get("HCCL_REPO")
    if not repo:
        sys.exit(
            "未指定 HCCL 仓路径。\n"
            "  用 --repo <HCCL 仓根目录> 传入，或先 export HCCL_REPO=<HCCL 仓根目录>。\n"
            "  请由调用方补充显式输入；不要猜或扫盘。"
        )
    repo = os.path.abspath(os.path.expanduser(repo))
    if not os.path.isdir(os.path.join(repo, "src/ops")):
        sys.exit(
            "不像是 HCCL 仓（缺少 src/ops 目录）: %s\n"
            "  请检查上游提供的仓根目录（应该是含 src/ops、build.sh 的那一层）。" % repo
        )
    return repo


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--repo", help="HCCL 仓根目录；缺省读环境变量 HCCL_REPO，两者都没有就报错"
    )
    ap.add_argument("--all", action="store_true", help="扫描全仓所有 AICPU template")
    ap.add_argument(
        "--strict-new",
        action="store_true",
        help="新文件门禁：把「适用=新增」的 BLOCKER/MUST 规则升级为 ERROR",
    )
    ap.add_argument(
        "--explicit-only",
        action="store_true",
        help="仅限单个目标：检查占位成本的无条件非空候选；仍须人工复核显式配置语义",
    )
    ap.add_argument(
        "--rule",
        nargs="?",
        const="",
        metavar="ID",
        help="打印一条规则（如 --rule R-CMAKE-002）；不带参数则列出全部",
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="--all 模式下把只有 WARN 的报告也完整打印",
    )
    ap.add_argument(
        "paths", nargs="*", help="template 的 .cc 或 .h 路径（相对仓根或绝对）"
    )
    args = ap.parse_args()

    if args.explicit_only and (args.all or len(args.paths) != 1):
        ap.error("--explicit-only 必须指定单个目标，不能与 --all 合用")

    if args.rule is not None:
        return print_rules(args.rule or None)

    repo = resolve_repo(args.repo)
    targets = []

    if args.all:
        targets = [
            p
            for p in globs(repo, "template/aicpu/*.cc")
            if "/op_common/" not in p
        ]
        if not targets:
            ap.error(
                "--all 在 %s 下没扫到任何 AICPU template。\n"
                "确认这是 HCCL 仓根目录；若仓库目录结构又变了，需要更新 globs() 的匹配深度。"
                % repo
            )
    for p in args.paths:
        path = p if os.path.isabs(p) else os.path.join(repo, p)
        if path.endswith(".h"):
            path = path[:-2] + ".cc"
        targets.append(path)

    if not targets:
        ap.error("请给出至少一个 template 路径，或使用 --all")

    if args.strict_new and args.all:
        print(
            "提示：--strict-new 是给新文件用的门禁，配 --all 会把存量欠债全部报成 ERROR。\n"
        )

    total_err, warn_files, total_warn = 0, [], 0
    for path in targets:
        rep = check_one(
            repo, path, strict_new=args.strict_new, explicit_only=args.explicit_only
        )
        total_err += rep.errors
        if rep.errors:
            rep.dump()
            continue
        if rep.warns:
            total_warn += rep.warns
            warn_files.append(rep)
            if not args.all or args.verbose:
                rep.dump()
        elif not args.all:
            rep.dump()

    # --all 模式下 WARN 不再被吞掉：按类别聚合，避免逐文件刷屏
    if args.all and warn_files and not args.verbose:
        buckets = {}
        for rep in warn_files:
            for tag in rep.warn_tags:
                buckets.setdefault(tag, []).append(rep.title)
        print("=" * 78)
        print(
            "WARN 汇总：%d 个文件共 %d 条（-v 看逐条明细）"
            % (len(warn_files), total_warn)
        )
        print("=" * 78)
        for tag, files in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            print("  %3d  %s" % (len(files), tag))
            if len(files) <= 3:
                for f in files:
                    print("       - %s" % f)
        print()
        print(
            "这些多半是「适用=新增」规则上的存量欠债，不是规则错了——见 references/07-rules.md §0。"
        )
        print("写新文件时用 --strict-new 把它们卡死。")
        print()

    if total_err:
        print("共 %d 个 ERROR" % total_err, end="")
        print("、%d 个 WARN" % total_warn if total_warn else "")
        return 1
    if total_warn:
        print(
            "没有 ERROR，但有 %d 个文件共 %d 条 WARN。" % (len(warn_files), total_warn)
        )
        return 0
    print("全部检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
