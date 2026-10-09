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

"""生成一个新的 HCCL AICPU algorithm template，并完成两处 CMake 接线。

用法：
    python3 scripts/new_template.py --repo <HCCL 仓根目录> \
            --op all_reduce --class InsTempAllReduceMesh1DFoo \
            [--file ins_temp_all_reduce_mesh_1D_foo] \
            [--pattern barebone|all-reduce-mesh-oneshot] \
            [--desc "all reduce (one-shot) 1D Mesh"] \
            [--compat-guard] [--dry-run] [--force]

--compat-guard  把源文件放进 if(NOT HCCL_CANN_COMPAT_850) 分支（需要 CANN 9.x 才编译）
--dry-run       只打印将要做的改动，不落盘
"""

import argparse
import datetime
import os
from pathlib import Path
import re
import sys

from cmake_utils import active_text, compat_guard_lines, has_source

# CamelCase → snake_case 时需要整体保留的词元（长的排前面）
CASE_ATOMS = ["2Die", "1D", "2D", "3D", "1d", "2d"]  # 保留原大小写
LOWER_ATOMS = ["NHR", "DPU", "AIV", "CCU", "UBX", "PCIE"]  # 统一转小写


def to_snake(name: str) -> str:
    """InsTempAllReduceMesh1DOneShot -> ins_temp_all_reduce_mesh_1D_one_shot"""
    reps = {}
    for i, atom in enumerate(CASE_ATOMS + LOWER_ATOMS):
        if atom in name:
            token = "\x00%d\x00" % i
            reps[token] = atom if atom in CASE_ATOMS else atom.lower()
            name = name.replace(atom, token)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = name.lower()
    for token, atom in reps.items():
        name = name.replace(token, "_%s_" % atom)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def load_asset(assets_dir: str, pattern: str, ext: str) -> str:
    path = os.path.join(assets_dir, "%s.%s.tpl" % (pattern, ext))
    if not os.path.exists(path):
        sys.exit("找不到骨架文件: %s" % path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def render(body: str, subs: dict) -> str:
    for key, val in subs.items():
        body = body.replace("{{%s}}" % key, val)
    left = re.findall(r"\{\{(\w+)\}\}", body)
    if left:
        sys.exit("骨架中存在未替换的占位符: %s" % sorted(set(left)))
    return body


def find_block(lines, start_pat, close_pat=r"^\)\s*$", guard=False):
    """返回 (起始行号, 结束行号)，结束行是块的收尾括号行。找不到返回 None。"""
    text = "\n".join(lines)
    active = active_text(text).splitlines()
    guarded = compat_guard_lines(text)
    start = None
    for i, line in enumerate(active):
        if re.search(start_pat, line) and (not guard or i in guarded):
            start = i
            break
    if start is None:
        return None
    for j in range(start + 1, len(active)):
        if re.match(close_pat, active[j]):
            return (start, j)
    return None


def insert_into_block(lines, block, entry, anchors):
    """在块内插入 entry。优先插到最后一条匹配 anchors 的行之后，否则插在块尾。"""
    start, end = block
    active = active_text("\n".join(lines)).splitlines()
    pos = end
    for anchor in anchors:
        hits = [i for i in range(start + 1, end) if anchor in active[i]]
        if hits:
            pos = hits[-1] + 1
            break
    lines.insert(pos, entry)
    return pos


def find_aicpu_prefix(repo, op):
    """探测 <op> 下 AICPU template 目录的布局前缀。

    2026-09 目录重构后为 'algorithm/template/aicpu'，旧仓为 'template/aicpu'；
    优先新布局，两者都不存在返回 None。
    """
    for prefix in ("algorithm/template/aicpu", "template/aicpu"):
        if os.path.isdir(os.path.join(repo, "src/ops", op, prefix)):
            return prefix
    return None


def patch_local_cmake(repo, op, filebase, guard, prefix):
    path = os.path.join(repo, "src/ops", op, prefix, "CMakeLists.txt")
    entry_body = "${CMAKE_CURRENT_SOURCE_DIR}/%s.cc" % filebase
    if not os.path.exists(path):
        return (
            "MISSING",
            path,
            "目录下没有 CMakeLists.txt，需要手工创建（参考同级其它算子）",
            None,
        )

    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    if has_source("\n".join(lines), entry_body, "host"):
        if guard and not has_source(
            "\n".join(lines), entry_body, "host", compat_guard=True
        ):
            return (
                "FAIL",
                path,
                "已有条目不在已核验的 CANN 9.x 守卫内，请先核对其条件",
                None,
            )
        return ("SKIP", path, "条目已存在", None)

    if guard:
        block = find_block(
            lines, r"^\s*list\(APPEND src_list", r"^\s*\)\s*$", guard=True
        )
        if block is None:
            return (
                "FAIL",
                path,
                "找不到 list(APPEND src_list ...) 块，需手工加 if(NOT HCCL_CANN_COMPAT_850) 分支",
                None,
            )
        entry = "        " + entry_body
    else:
        block = find_block(lines, r"^set\(src_list", r"^\)\s*$")
        if block is None:
            return ("FAIL", path, "找不到 set(src_list ...) 块", None)
        entry = "    " + entry_body

    pos = insert_into_block(lines, block, entry, [])
    if not has_source("\n".join(lines), entry_body, "host", compat_guard=guard):
        return ("FAIL", path, "修改后没有有效的 host 源文件登记", None)
    return (
        "OK",
        path,
        "第 %d 行插入 %s" % (pos + 1, entry.strip()),
        "\n".join(lines) + "\n",
    )


def patch_kernel_cmake(repo, op, filebase, guard, prefix):
    path = os.path.join(repo, "src/scatter_aicpu_kernel.cmake")
    if not os.path.exists(path):
        return ("MISSING", path, "文件不存在", None)

    rel = "ops/%s/%s/%s.cc" % (op, prefix, filebase)
    entry_body = "${CMAKE_CURRENT_SOURCE_DIR}/%s" % rel

    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    if has_source("\n".join(lines), entry_body, "device"):
        if guard and not has_source(
            "\n".join(lines), entry_body, "device", compat_guard=True
        ):
            return (
                "FAIL",
                path,
                "已有条目不在已核验的 CANN 9.x 守卫内，请先核对其条件",
                None,
            )
        return ("SKIP", path, "条目已存在", None)

    if guard:
        block = find_block(
            lines,
            r"^\s*target_sources\(scatter_aicpu_kernel PRIVATE",
            r"^\s*\)\s*$",
            guard=True,
        )
        if block is None:
            return (
                "FAIL",
                path,
                "找不到 target_sources(scatter_aicpu_kernel PRIVATE ...) 块，需手工加 if(NOT HCCL_CANN_COMPAT_850) 分支",
                None,
            )
        entry = "        " + entry_body
    else:
        block = find_block(
            lines, r"^add_library\(scatter_aicpu_kernel SHARED", r"^\)\s*$"
        )
        if block is None:
            return (
                "FAIL",
                path,
                "找不到 add_library(scatter_aicpu_kernel SHARED ...) 块",
                None,
            )
        entry = "    " + entry_body

    anchors = ["ops/%s/%s/" % (op, prefix), "ops/%s/" % op]
    pos = insert_into_block(lines, block, entry, anchors)
    if not has_source("\n".join(lines), entry_body, "device", compat_guard=guard):
        return ("FAIL", path, "修改后没有有效的 device 源文件登记", None)
    return (
        "OK",
        path,
        "第 %d 行插入 %s" % (pos + 1, entry.strip()),
        "\n".join(lines) + "\n",
    )


def load_algo_types(repo):
    """从 $HCCL/src/common/alg_parse.h 实时解析 AlgoType 枚举，**不写死清单**。"""
    path = os.path.join(repo, "src/common/alg_parse.h")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        m = re.search(r"enum\s+class\s+AlgoType\s*:[^{]*\{(.*?)\}", f.read(), re.S)
    if not m:
        return set()
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return {v.strip() for v in body.split(",") if v.strip()}


def infer_algo_type(cls, pattern):
    """按类名推断 props 的 AlgoType；推不出时按骨架给个保底值。"""
    u = re.sub(r"[^A-Z0-9]", "", cls.upper())
    if "NHR" in u:
        return "NHR_AICPU_REDUCE" if "AICPUREDUCE" in u else "NHR"
    if "MESHCHUNK" in u:
        return "MESH_CHUNK_TWOSHOT" if "TWOSHOT" in u else "MESH_CHUNK"
    if "MESH" in u:
        if "ONESHOT" in u:
            return "MESH_ONESHOT"
        if "TWOSHOT" in u:
            return "MESH_TWOSHOT"
        # 类名只说了 Mesh 没说 one-shot/two-shot 时，用骨架自身的语义兜底
        return "MESH_ONESHOT" if pattern == "all-reduce-mesh-oneshot" else "MESH"
    return "MESH_ONESHOT" if pattern == "all-reduce-mesh-oneshot" else "UNKNOWN"


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
    ap.add_argument("--op", required=True, help="算子目录名，如 all_reduce")
    ap.add_argument(
        "--class", dest="cls", required=True, help="类名，如 InsTempAllReduceMesh1DFoo"
    )
    ap.add_argument(
        "--file", help="文件名（不带扩展名）。缺省由类名推导，请核对后再落盘"
    )
    ap.add_argument(
        "--pattern",
        default="barebone",
        choices=["barebone", "all-reduce-mesh-oneshot"],
        help="barebone=不预设拓扑/归约的自定义算法骨架（未实现时返回错误）；all-reduce-mesh-oneshot=AllReduce 专用",
    )
    ap.add_argument("--desc", help="Describe() 里的一句话描述")
    ap.add_argument(
        "--algo-type",
        dest="algo_type",
        help="props 的 AlgoType（默认按类名推断），取值见 $HCCL/src/common/alg_parse.h",
    )
    ap.add_argument("--year", default=str(datetime.date.today().year))
    ap.add_argument(
        "--compat-guard",
        action="store_true",
        help="放进 if(NOT HCCL_CANN_COMPAT_850) 分支",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 .h/.cc")
    args = ap.parse_args()

    repo = resolve_repo(args.repo)
    for label, value, pattern in (
        ("--op", args.op, r"[a-z][a-z0-9_]*"),
        ("--class", args.cls, r"[A-Za-z_][A-Za-z0-9_]*"),
    ):
        if not re.fullmatch(pattern, value):
            ap.error(label + " 必须是单个合法标识符，不能包含路径")

    if not args.cls.startswith("InsTemp"):
        print("[warn] 类名建议以 InsTemp 开头（当前: %s）" % args.cls, file=sys.stderr)

    if args.pattern == "all-reduce-mesh-oneshot" and args.op != "all_reduce":
        sys.exit(
            "--pattern all-reduce-mesh-oneshot 是 AllReduce 专用骨架（自带 LocalReduce / reduceOp /\n"
            "one-shot cost model），给 --op %s 用会生成语义错误但能编过的代码。\n"
            "请改用 --pattern barebone。" % args.op
        )

    filebase = args.file or to_snake(args.cls)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", filebase):
        ap.error("--file 必须是无目录、无扩展名的文件基名（字母、数字、下划线）")
    if not filebase.startswith("ins_temp_"):
        print(
            "[warn] 文件名建议以 ins_temp_ 开头（当前: %s）" % filebase, file=sys.stderr
        )

    prefix = find_aicpu_prefix(repo, args.op)
    if prefix is None:
        sys.exit(
            "目录不存在（两种布局都探测过）:\n"
            "    %s\n    %s\n"
            "确认 --op 拼写，或先手工建目录与 CMakeLists.txt"
            % (
                os.path.join(repo, "src/ops", args.op, "algorithm/template/aicpu"),
                os.path.join(repo, "src/ops", args.op, "template/aicpu"),
            )
        )
    outdir = os.path.join(repo, "src/ops", args.op, prefix)

    h_path = os.path.join(outdir, filebase + ".h")
    cc_path = os.path.join(outdir, filebase + ".cc")
    # Refuse symlinks in all write destinations, including the CMake files.
    # --force permits replacing source files, not escaping the selected tree.
    repo_root = Path(repo).resolve()
    destinations = (
        h_path,
        cc_path,
        os.path.join(outdir, "CMakeLists.txt"),
        os.path.join(repo, "src/scatter_aicpu_kernel.cmake"),
    )
    for destination in destinations:
        relative = Path(destination).relative_to(Path(repo))
        current = repo_root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                ap.error("写入路径不能经过符号链接: " + str(current))
        if current.exists() and not current.is_file():
            ap.error("写入目标不是普通文件: " + str(current))
    for path in (h_path, cc_path):
        if os.path.exists(path) and not args.force:
            sys.exit("文件已存在: %s（加 --force 覆盖）" % path)

    assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
    assets = os.path.normpath(assets)
    license_path = os.path.join(assets, "_license.tpl")
    if not os.path.exists(license_path):
        sys.exit("找不到版权头模板: %s" % license_path)
    with open(license_path, encoding="utf-8") as f:
        license_txt = f.read().rstrip("\n").replace("{{YEAR}}", args.year)

    algo_type = args.algo_type or infer_algo_type(args.cls, args.pattern)
    known = load_algo_types(repo)
    if known and algo_type not in known:
        sys.exit(
            "AlgoType::%s 不在 %s/src/common/alg_parse.h 的枚举里。\n可选值：%s"
            % (algo_type, repo, "、".join(sorted(known)))
        )

    subs = {
        "LICENSE": license_txt,
        "CLASS": args.cls,
        "GUARD": filebase.upper() + "_H",
        "HEADER": filebase + ".h",
        "DESC": args.desc or args.cls,
        "YEAR": args.year,
        "ALGOTYPE": algo_type,
    }
    h_body = render(load_asset(assets, args.pattern, "h"), subs)
    cc_body = render(load_asset(assets, args.pattern, "cc"), subs)

    print("类名     : %s" % args.cls)
    print("文件名   : %s.{h,cc}   ← 请核对是否与同目录邻居风格一致" % filebase)
    print("落盘目录 : %s" % outdir)
    print("骨架     : %s" % args.pattern)
    print(
        "algoType : AlgoType::%s%s"
        % (
            algo_type,
            "（--algo-type 指定）" if args.algo_type else "（按类名推断，请核对）",
        )
    )
    print()

    # ---- 先在内存里把全部四处改动算完，任一处失败就整体不落盘 ----
    results = [
        patch_local_cmake(repo, args.op, filebase, args.compat_guard, prefix),
        patch_kernel_cmake(repo, args.op, filebase, args.compat_guard, prefix),
    ]
    for status, path, note, _ in results:
        print("[%-7s] %s\n            %s" % (status, path, note))
    print()

    bad = [(s, p_, n) for s, p_, n, _ in results if s in ("FAIL", "MISSING")]
    if bad:
        print("两处 CMake 登记没能自动完成，**源文件也不会生成**——")
        print(
            "否则会留下「代码在、接线缺一半」的中间状态：host 编得过，运行时 AICPU 侧找不到符号。"
        )
        for s, p_, n in bad:
            print("  [%s] %s\n        %s" % (s, p_, n))
        print("\n处理办法：手工补好上述 CMake 块后重跑，或先 --dry-run 看清楚落点。")
        return 1

    if args.dry_run:
        print(
            "[dry-run] 以上改动均未落盘（源文件 2 个 + CMake %d 处）"
            % sum(1 for s, _, _, _ in results if s == "OK")
        )
        return 0

    if args.force:
        exist = [p_ for p_ in (h_path, cc_path) if os.path.exists(p_)]
        if exist:
            print("[--force] 将覆盖以下已存在的文件：")
            for p_ in exist:
                print("    %s" % p_)
            print()

    for path, body in ((h_path, h_body), (cc_path, cc_body)):
        with open(path, "w", encoding="utf-8") as f:
            f.write(body if body.endswith("\n") else body + "\n")
        print("[写入] %s" % path)
    for status, path, note, content in results:
        if content is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print("[写入] %s" % path)
    print()

    print()
    print("下一步：")
    print("  1. 按已检查的 Dataflow Spec 实现资源、切片、同步、输出落位和选路条件")
    if args.pattern == "barebone":
        print("     barebone 尚未实现算法：CalcRes/KernelRun 返回错误，成本候选为空。")
        print("     完成全部 TODO 后才可解除未实现错误；生成和静态检查不代表功能可用。")
    print(
        "  2. 让 executor 用上它：REGISTER_EXEC_V2(cmdType, <算法名>, <Executor>, <TopoMatch>, %s)"
        % args.cls
    )
    checker = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "check_template.py"
    )
    print(
        "  3. python3 %s --repo %s --strict-new src/ops/%s/%s/%s.cc"
        % (checker, repo, args.op, prefix, filebase)
    )
    print("  4. 将生成清单、静态检查结果和未处理 TODO 返回调用方。")
    print("     完整构建、安装和双轨 Checker 由调用 Agent/Plugin 编排。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
