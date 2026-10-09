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

"""实时扫描 HCCL 仓，列出所有 algorithm template 及其关键特征。

用法（--repo 必填，或先 export HCCL_REPO=<HCCL 仓根目录>；下面为省字都写作 $HCCL）：
    python3 scripts/list_templates.py --repo $HCCL                   # 全量索引
    python3 scripts/list_templates.py --repo $HCCL --op all_reduce   # 只看某个算子
    python3 scripts/list_templates.py --repo $HCCL --topo nhr        # 只看 NHR 类
    python3 scripts/list_templates.py --repo $HCCL --grep OneShot    # 类名/文件名模糊匹配
    python3 scripts/list_templates.py --repo $HCCL --blueprint       # 「照着抄哪个」推荐蓝本
    python3 scripts/list_templates.py --repo $HCCL --variants        # 继承已有 template 的变体关系
    python3 scripts/list_templates.py --repo $HCCL --detail InsTempAllReduceNHR   # 单个 template 详情
    python3 scripts/list_templates.py --repo $HCCL --format md       # Markdown 表格输出

扫描的是 src/ops/<op>/[algorithm/]template/** 下继承 InsAlgTemplateBase 的类（AICPU / DPU / host_nic）。
兼容 2026-09 目录重构前后的两种仓布局（template 迁入 <op>/algorithm/ 下）。
CCU（CcuAlgTemplateBase）和 AIV（AivAlgTemplateBase）默认不列，加 --all-engines 一并列出。
"""

import argparse
import glob
import os
import re
import sys

from cmake_utils import has_source

# 两种仓布局下 template / executor 所在的目录段（新布局带 algorithm/，旧布局不带）
TMPL_LAYOUTS = ("algorithm/template", "template")
EXEC_LAYOUTS = ("algorithm/executor", "executor")


def tmpl_headers(repo):
    """两种布局下的全部 template 头文件（去重排序）。"""
    out = []
    for layout in TMPL_LAYOUTS:
        out += glob.glob(
            os.path.join(repo, "src/ops/*/%s/**/*.h" % layout), recursive=True
        )
    return sorted(set(out))


BASES = {
    "InsAlgTemplateBase": "aicpu",
    "CcuAlgTemplateBase": "ccu",
    "AivAlgTemplateBase": "aiv",
    "AlgTemplateBase": "v1",
}

# 「照着抄哪个」：意图 → 首选类名（按优先级），运行时对照实际扫描结果解析
BLUEPRINTS = [
    ("最小可运行骨架（只做同步，不搬数据）", ["InsTempBarrierMesh1D"]),
    ("Mesh 全连接 one-shot（交换整份 + 本地归约）", ["InsTempAllReduceMesh1DOneShot"]),
    ("Mesh two-shot（ReduceScatter + AllGather）", ["InsTempAllReduceMesh1DTwoShot"]),
    ("NHR 多 step / 多 jetty / 读写模式切换", ["InsTempAllReduceNHR"]),
    ("Gather 类（repeat + stride + 对称内存 + 图模式）", ["InsTempAllGatherMesh1D"]),
    ("Scatter 类（root 分发）", ["InsTempScatterMesh1D"]),
    ("ReduceScatter 类（切片 + 归约到本 rank 片）", ["InsTempReduceScatterMesh1D"]),
    (
        "变长算子（*_v，allRankSliceSize / allRankDispls）",
        ["InsTempAllGatherVMesh1D", "InsTempReduceScatterVMesh1D"],
    ),
    (
        "AlltoAll 类（sendCounts / recvCounts / sdispls / rdispls）",
        ["InsTempAlltoAllVMesh1D"],
    ),
    (
        "DPU 展开（需要 REGISTER_TEMPLATE_V2 字符串注册）",
        ["InsTempReduceScatterMesh1dDpu", "InsTempAllGatherNHRDPU"],
    ),
    (
        "变体：继承已有 template 只覆写差异",
        ["InsTempAllGatherOmniPipeMesh1D", "InsTempReduceScatterOmniPipeNHR"],
    ),
]


def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def func_body(text, cls, method):
    """抠出 Cls::method(...) 的函数体（假定收尾 } 顶格），并剥掉注释。"""
    m = re.search(r"\b%s::%s\s*\(" % (re.escape(cls), re.escape(method)), text)
    if not m:
        return None
    open_brace = text.find("{", m.end())
    if open_brace < 0:
        return None
    end = text.find("\n}", open_brace)
    body = text[open_brace:end] if end > 0 else text[open_brace:]
    return strip_comments(body)


def scratch_expr(text, cls):
    body = func_body(text, cls, "CalcScratchMultiple")
    if body is None:
        return "-"
    ret = re.search(r"return\s+([^;]+);", body)
    if not ret:
        return "?"
    val = ret.group(1).strip()
    if re.match(r"^\w+$", val) and not val.isdigit():
        # 返回的是局部变量，回溯它的赋值（取最后一次）
        assigns = re.findall(r"\b%s\s*=\s*([^;]+);" % re.escape(val), body)
        if assigns:
            val = assigns[-1].strip()
            if len(assigns) > 1:
                val += " (branch)"
    return re.sub(r"\s+", " ", val)


def thread_expr(text, cls):
    body = func_body(text, cls, "GetThreadNum")
    if body is not None:
        ret = re.search(r"return\s+([^;]+);", body)
        if ret:
            expr = re.sub(r"\s+", " ", ret.group(1).strip())
            expr = re.sub(r"(\w+) > 1 \? \1 : 1", r"\1", expr)
            return expr
    for method in ("CalcRes", "GetRes"):
        body = func_body(text, cls, method)
        if body:
            m = re.search(r"slaveThreadNum\s*=\s*([^;]+);", body)
            if m:
                expr = re.sub(r"\s+", " ", m.group(1).strip())
                expr = re.sub(
                    r"^(.*?) - 1$", r"\1", expr
                )  # slave = X-1  =>  线程总数 X
                if re.match(r"^\w+$", expr) and not expr.isdigit():
                    local = re.search(
                        r"\b\w+\s+%s\s*=\s*([^;]+);" % re.escape(expr), body
                    )
                    if local:
                        expr = re.sub(r"\s+", " ", local.group(1).strip())
                expr = re.sub(r"(\w+) > 1 \? \1 : 1", r"\1", expr)  # 常见三元退化写法
                return expr
    return "-"


def channel_fns(text, cls):
    body = func_body(text, cls, "CalcRes") or ""
    fns = sorted(
        set(
            re.findall(
                r"\b(CalcChannelRequest\w+|CreateChannelRequestByRankId)\s*\(", body
            )
        )
    )
    return ", ".join(f.replace("CalcChannelRequest", "") for f in fns) or "-"


def split_args(argstr):
    """按顶层逗号切分宏参数。"""
    out, depth, cur = [], 0, ""
    for ch in argstr:
        if ch in "(<":
            depth += 1
        elif ch in ")>":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


# 所有会把 template 绑定到 executor 的注册宏 → 模板参数在 args 里的起始下标
REG_MACROS = {
    "REGISTER_EXEC_V2": 4,
    "REGISTER_EXEC_V2_MULTI": 4,
    "REGISTER_EXECUTOR_BY_TWO_TEMPS": 4,
    "REGISTER_EXECUTOR_BY_FOUR_TEMPS": 4,
    "REGISTER_EXECUTOR_IMPL_NO_TOPOMATCH": 3,
}
REG_RE = re.compile(
    r"\b(%s)\s*\((.*?)\)\s*;" % "|".join(sorted(REG_MACROS, key=len, reverse=True)),
    re.S,
)


def scan_executors(repo):
    """class -> [(algName, executorRelPath)]"""
    result = {}
    exec_paths = []
    for layout in EXEC_LAYOUTS:
        exec_paths += glob.glob(os.path.join(repo, "src/ops/*/%s/*.cc" % layout))
    for path in sorted(set(exec_paths)):
        text = strip_comments(read(path))
        rel = os.path.relpath(path, repo)
        for m in REG_RE.finditer(text):
            macro, args = m.group(1), split_args(m.group(2))
            start = REG_MACROS[macro]
            if len(args) <= start:
                continue
            alg_name = args[1]
            for tmpl in args[start:]:
                if re.match(r"^\w+$", tmpl):
                    result.setdefault(tmpl, []).append((alg_name, rel))
    return result


def cmake_status(repo, op, op_tmpl_prefix, filebase, engine_dir):
    """op_tmpl_prefix: '<op>/algorithm' 或 '<op>'（旧布局）——由文件 rel 路径解析而来。"""
    local = os.path.join(
        repo, "src/ops", op_tmpl_prefix, "template", engine_dir, "CMakeLists.txt"
    )
    if engine_dir == ".":
        local = os.path.join(
            repo, "src/ops", op_tmpl_prefix, "template", "CMakeLists.txt"
        )
    kernel = os.path.join(repo, "src/scatter_aicpu_kernel.cmake")
    host = os.path.exists(local) and has_source(
        read(local), "${CMAKE_CURRENT_SOURCE_DIR}/%s.cc" % filebase, "host"
    )
    sub = engine_dir if engine_dir != "." else ""
    needle = (
        "ops/%s/template/%s/%s.cc" % (op_tmpl_prefix, sub, filebase)
        if sub
        else "ops/%s/template/%s.cc" % (op_tmpl_prefix, filebase)
    )
    dev = os.path.exists(kernel) and has_source(
        read(kernel), "${CMAKE_CURRENT_SOURCE_DIR}/" + needle, "device"
    )
    return ("H" if host else "-") + ("D" if dev else "-")


def scan(repo, all_engines):
    execs = scan_executors(repo)
    rows = []
    all_headers = tmpl_headers(repo)
    for h_path in all_headers:
        rel = os.path.relpath(h_path, repo).replace(os.sep, "/")
        if "/op_common/" in rel or "/kernel/" in rel:
            continue
        hh = read(h_path)
        m = re.search(r"class\s+(\w+)\s*:\s*public\s+(\w+)", hh)
        if not m:
            continue
        cls, parent = m.group(1), m.group(2)
        cc_path = h_path[:-2] + ".cc"
        cc = read(cc_path) if os.path.exists(cc_path) else ""

        # 沿继承链找到引擎归属
        engine, root, chain = None, parent, 0
        seen = {cls}
        while root and chain < 8:
            if root in BASES:
                engine = BASES[root]
                break
            nxt = None
            for other in all_headers:
                mm = re.search(
                    r"class\s+%s\s*:\s*public\s+(\w+)" % re.escape(root), read(other)
                )
                if mm:
                    nxt = mm.group(1)
                    break
            if not nxt or root in seen:
                break
            seen.add(root)
            root = nxt
            chain += 1
        if engine is None:
            continue
        if not all_engines and engine not in ("aicpu",):
            continue

        parts = rel.split("/")
        op = parts[2]
        tmpl_idx = parts.index("template")
        # 'all_reduce/algorithm'（新布局）或 'all_reduce'（旧布局）——CMake needle 与 local 路径都按它拼
        op_tmpl_prefix = "/".join([op] + parts[3:tmpl_idx])
        engine_dir = parts[tmpl_idx + 1] if len(parts) > tmpl_idx + 2 else "."
        filebase = os.path.basename(h_path)[:-2]

        regs = execs.get(cls, [])
        rows.append(
            {
                "op": op,
                "engine_dir": engine_dir,
                "file": filebase,
                "rel": rel[:-2] + ".cc",
                "cls": cls,
                "parent": parent,
                "loc": len(cc.splitlines()),
                "scratch": scratch_expr(cc, cls) if cc else "-",
                "thread": thread_expr(cc, cls) if cc else "-",
                "channel": channel_fns(cc, cls) if cc else "-",
                "algotype": algo_type(hh),
                "strreg": "REGISTER_TEMPLATE_V2" in cc,
                "algnames": sorted(set(n for n, _ in regs)),
                "execs": sorted(set(p for _, p in regs)),
                "cmake": cmake_status(repo, op, op_tmpl_prefix, filebase, engine_dir),
            }
        )
    return rows


def fit(s, width):
    """表格列裁剪：超宽时以 … 结尾，提示这里被截断了（完整值用 --detail 或 --format md 看）。"""
    return s if len(s) <= width else s[: width - 1] + "…"


def print_table(rows, fmt):
    if fmt == "md":
        print(
            "| 算子 | 文件 | 类 | 行数 | scratch 倍数 | thread 数 | channel | CMake | 算法名 |"
        )
        print("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(
                "| %s | `%s` | `%s` | %d | `%s` | `%s` | %s | %s | %s |"
                % (
                    r["op"],
                    r["file"],
                    r["cls"],
                    r["loc"],
                    r["scratch"],
                    r["thread"],
                    r["channel"],
                    r["cmake"],
                    ", ".join(r["algnames"]) or "—",
                )
            )
        return

    hdr = "%-16s %-46s %-38s %5s %-30s %-34s %-24s %-5s" % (
        "OP",
        "FILE",
        "CLASS",
        "LOC",
        "SCRATCH",
        "THREADS",
        "CHANNEL",
        "CMAKE",
    )
    print(hdr)
    print("-" * len(hdr))
    cur_op = None
    for r in rows:
        if r["op"] != cur_op:
            cur_op = r["op"]
            print()
        flags = ""
        if not r["parent"].endswith("TemplateBase"):
            flags += " [变体←%s]" % r["parent"]
        if r["algotype"]:
            flags += " [%s]" % r["algotype"]
        if r["strreg"]:
            flags += " [strReg]"
        if not r["algnames"]:
            flags += " [未被executor引用]"
        print(
            "%-16s %-46s %-38s %5d %-30s %-34s %-24s %-5s%s"
            % (
                r["op"],
                r["file"],
                r["cls"],
                r["loc"],
                fit(r["scratch"], 30),
                fit(r["thread"], 34),
                fit(r["channel"], 24),
                r["cmake"],
                flags,
            )
        )


def print_detail(rows, name):
    hits = [r for r in rows if r["cls"] == name or r["file"] == name]
    if not hits:
        print("没找到: %s" % name)
        return 1
    for r in hits:
        print("=" * 78)
        print("%s" % r["cls"])
        print("=" * 78)
        print("  源文件        : %s" % r["rel"])
        print("  算子/引擎目录 : %s / %s" % (r["op"], r["engine_dir"]))
        print(
            "  父类          : %s%s"
            % (
                r["parent"],
                ""
                if r["parent"].endswith("TemplateBase")
                else "   ← 变体，继承已有 template",
            )
        )
        print("  .cc 行数      : %d" % r["loc"])
        print("  scratch 倍数  : %s" % r["scratch"])
        print("  thread 数     : %s" % r["thread"])
        print("  channel 申请  : %s" % r["channel"])
        print(
            "  props.algoType: %s"
            % (r["algotype"] or "未声明（TemplateProp 默认 UNKNOWN）")
        )
        print(
            "  字符串注册    : %s"
            % (
                "REGISTER_TEMPLATE_V2（DPU 路径）"
                if r["strreg"]
                else "无（走 executor 静态绑定）"
            )
        )
        print(
            "  CMake 接线    : %s   (H=host libhccl.so, D=device libscatter_aicpu_kernel.so)"
            % r["cmake"]
        )
        print(
            "  算法名        : %s"
            % (", ".join(r["algnames"]) or "—  未被任何 REGISTER_EXEC_V2 引用")
        )
        for p in r["execs"]:
            print("  executor      : %s" % p)
        print()
    return 0


def print_blueprints(rows):
    by_cls = {r["cls"]: r for r in rows}
    print("「照着抄哪个」—— 实时对照当前仓库解析\n")
    for intent, candidates in BLUEPRINTS:
        picked = next((by_cls[c] for c in candidates if c in by_cls), None)
        print("  %s" % intent)
        if picked:
            print(
                "      %s   (%d 行, scratch=%s)"
                % (picked["rel"], picked["loc"], picked["scratch"])
            )
        else:
            print(
                "      ⚠ 首选 %s 已不在仓内，用 --op <算子> 自行挑选"
                % " / ".join(candidates)
            )
        print()
    print(
        "复用已有实现时请读相关完整函数及调用关系；自定义算法先读算子契约，无需同类蓝本。"
    )


def print_variants(rows):
    variants = [r for r in rows if not r["parent"].endswith("TemplateBase")]
    if not variants:
        print("当前没有继承已有 template 的变体。")
        return
    print("变体继承关系（只覆写差异方法，CalcRes / CalcScratchMultiple 等直接继承）\n")
    width = max(len(r["cls"]) for r in variants)
    for r in sorted(variants, key=lambda x: (x["parent"], x["cls"])):
        pw = max(len(x["parent"]) for x in variants)
        print("  %-*s  ->  %-*s   %s" % (width, r["cls"], pw, r["parent"], r["rel"]))


def algo_type(hh):
    """解析 static constexpr TemplateProp props = {.algoType = AlgoType::XXX};

    TemplateProp 当前只有 algoType 一个成员（历史上的 isNhr 已被删除）。
    """
    m = re.search(r"TemplateProp\s+props\s*=\s*\{[^}]*AlgoType::(\w+)", hh)
    return m.group(1) if m else ""


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
    ap.add_argument("--op", help="只看某个算子目录，如 all_reduce")
    ap.add_argument("--topo", choices=["mesh", "nhr"], help="按拓扑过滤")
    ap.add_argument("--grep", help="类名/文件名模糊匹配（不区分大小写）")
    ap.add_argument(
        "--all-engines", action="store_true", help="连 CCU / AIV / V1 一起列出"
    )
    ap.add_argument(
        "--blueprint", action="store_true", help="打印「照着抄哪个」推荐蓝本"
    )
    ap.add_argument("--variants", action="store_true", help="打印变体继承关系")
    ap.add_argument("--detail", help="打印单个 template 的详情（类名或文件名）")
    ap.add_argument("--format", default="table", choices=["table", "md"])
    args = ap.parse_args()

    repo = resolve_repo(args.repo)

    rows = scan(repo, args.all_engines)

    if args.detail:
        return print_detail(rows, args.detail)
    if args.blueprint:
        print_blueprints(rows)
        return 0
    if args.variants:
        print_variants(rows)
        return 0

    if args.op:
        rows = [r for r in rows if r["op"] == args.op]
    if args.topo:
        key = "nhr" if args.topo == "nhr" else "mesh"
        rows = [r for r in rows if key in r["cls"].lower() or key in r["file"].lower()]
    if args.grep:
        pat = args.grep.lower()
        rows = [r for r in rows if pat in r["cls"].lower() or pat in r["file"].lower()]

    if not rows:
        print("没有匹配的 template。")
        return 0
    print_table(rows, args.format)
    print(
        "\n共 %d 个 template。CMAKE 列：H=host libhccl.so，D=device libscatter_aicpu_kernel.so，"
        "两者都应为 H D。" % len(rows)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
