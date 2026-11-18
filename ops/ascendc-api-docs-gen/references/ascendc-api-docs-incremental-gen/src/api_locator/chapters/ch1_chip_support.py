# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ch1 产品支持情况 — 严格按四维表提取。

四维表映射:
- basic_api: 文件(impl)✅ 函数(Impl)✅ 代码段❌ 注释段❌
- adv_api: 文件(impl+头)✅ 函数(Impl/原名)✅ 代码段(__NPU_ARCH__)✅ 注释段❌
- simt_api: 文件(impl)✅ 函数(原名)✅ 代码段❌ 注释段❌
- tensor_api: 文件(impl)✅ 函数(原名)✅ 代码段❌ 注释段❌
- c_api: 文件(impl+头)✅ 函数(原名)✅ 代码段(__NPU_ARCH__)✅ 注释段❌
- aicpu_api: 无
- utils: 文件(impl)✅ 函数(原名)✅ 代码段❌ 注释段❌
"""

import os
import re

from config import CH_CHIP_SUPPORT
from evidence import Evidence

CHAPTER_NAME = CH_CHIP_SUPPORT


def extract(api_name, api_type, decl_file, impl_files, test_files, extractor, export_root):
    """四维独立提取：文件 + 函数 + 代码段 + 注释段。"""
    evidence = []
    evidence.extend(_file_dimension(api_name, api_type, decl_file, impl_files))
    evidence.extend(_function_dimension(api_name, api_type, decl_file, impl_files, extractor))
    evidence.extend(_code_dimension(api_name, api_type, decl_file, extractor))

    arch_list, arch_sources = _scan_arch_directories(extractor, api_type, api_name, impl_files, decl_file)

    # c_api/adv_api：arch 必须被头文件 __NPU_ARCH__==X 门控，且门控 #include 的目标文件
    # 仍被 locator 定位（在 impl_files 中）才算有效接线。
    # 仅 impl 目录存在但头文件未 #include（如 npu_arch_2201 的 vector_compute_impl 存在
    # 但头文件只门控 3510），或门控目标已被清空/删除（如 vector_compute_impl.h 被掏空
    # 不再引用本 API）都不算支持。
    if api_type in ("c_api", "adv_api") and arch_list:
        wired_arches = _extract_header_wired_arches(decl_file, extractor, impl_files)
        arch_list = [a for a in arch_list if a in wired_arches]
        arch_sources = {a: srcs for a, srcs in arch_sources.items() if a in wired_arches}

    chip_list, chip_sources, arch_from_content, arch_sources_content = _scan_chip_registration(
        extractor, api_name, impl_files
    )

    # arch_from_content 作为补充信号，合并到 arch_list/arch_sources
    for arch in arch_from_content:
        if arch not in set(arch_list):
            arch_list.append(arch)
        arch_sources.setdefault(arch, []).extend(arch_sources_content[arch])
    arch_list = sorted(set(arch_list), key=lambda n: (0, int(n)) if n.isdigit() else (1, n))

    evidence.append(
        Evidence(
            dimension="file",
            file="",
            line=0,
            content=", ".join(arch_list) if arch_list else "",
            metadata={
                "type": "chip_arch",
                "arch_list": arch_list,
                "arch_sources": arch_sources,
                "chip_list": chip_list,
                "chip_sources": chip_sources,
            },
        )
    )
    return evidence


# ==================== 四维独立提取 ====================


def _file_dimension(api_name, api_type, decl_file, impl_files):
    """文件维度：只要文件存在就生成 file anchor。"""
    evidence = []
    if api_type in ("basic_api", "simt_api", "tensor_api", "utils"):
        for impl in impl_files:
            evidence.append(
                Evidence(
                    dimension="file",
                    file=impl,
                    line=1,
                    content=impl,
                    metadata={"type": "impl_file"},
                )
            )
    elif api_type in ("adv_api", "c_api"):
        for impl in impl_files:
            evidence.append(
                Evidence(
                    dimension="file",
                    file=impl,
                    line=1,
                    content=impl,
                    metadata={"type": "impl_file"},
                )
            )
        if decl_file:
            evidence.append(
                Evidence(
                    dimension="file",
                    file=decl_file,
                    line=1,
                    content=decl_file,
                    metadata={"type": "header_file"},
                )
            )
    return evidence


def _function_dimension(api_name, api_type, decl_file, impl_files, e):
    """函数维度：独立搜索函数名/声明。"""
    evidence = []
    if api_type in ("basic_api", "simt_api", "tensor_api", "utils"):
        for impl in impl_files:
            # basic_api: prefer {API}Impl function name
            found_name = None
            for fname in (f"{api_name}Impl", api_name):
                snippets = e.extract_snippets(impl, fname, is_declaration=False)
                if snippets:
                    found_name = fname
                    break
            if found_name:
                ev = snippets[0]
                ev.metadata["type"] = "impl_function"
                ev.metadata["name"] = found_name
                evidence.append(ev)
    elif api_type == "adv_api":
        for impl in impl_files:
            found_name = None
            for fname in (api_name, f"{api_name}Impl"):
                snippets = e.extract_snippets(impl, fname, is_declaration=False)
                if snippets:
                    found_name = fname
                    break
            if found_name:
                ev = snippets[0]
                ev.metadata["type"] = "impl_function"
                ev.metadata["name"] = found_name
                evidence.append(ev)
    elif api_type == "c_api":
        for impl in impl_files:
            snippets = e.extract_snippets(impl, api_name, is_declaration=False)
            if snippets:
                ev = snippets[0]
                ev.metadata["type"] = "impl_function"
                ev.metadata["name"] = api_name
                evidence.append(ev)
        if not impl_files:
            decl_snippets = e.extract_all_decl_snippets(api_name, decl_file)
            if decl_snippets:
                first = decl_snippets[0]
                evidence.append(
                    Evidence(
                        dimension="function",
                        file=decl_file,
                        line=first.line,
                        content=f"{api_name} (位于 {decl_file})",
                        metadata={"type": "impl_function", "name": api_name},
                    )
                )
    return evidence


def _code_dimension(api_name, api_type, decl_file, e):
    """代码段维度：独立提取 __NPU_ARCH__ 条件编译。"""
    if api_type in ("adv_api", "c_api"):
        return e.extract_arch_conditions(decl_file)
    return []


def _scan_chip_registration(extractor, api_name, impl_files):
    """扫描非 arch 目录化文件，提取芯片标识和补充 arch。

    跳过路径含 npu_arch_ / dav_ 的文件（这些已由 _scan_arch_directories 处理）。
    只扫 _def.cpp / _apt.cpp / header 等注册类文件。

    Returns:
        (chip_list, chip_sources, arch_from_content, arch_sources_content)
    """
    chip_patterns = [
        # TYPE1: AICore().AddConfig("ascend950", ...)
        re.compile(r'AICore\(\)\s*\.\s*AddConfig\s*\(\s*"([^"]+)"'),
        # TYPE2: SocVersion::XXX (with optional namespace)
        re.compile(r"SocVersion::(\w+)"),
        # TYPE3: curShortSocName == platform_ascendc::SocVersion::XXX
        re.compile(r"curShortSocName\s*==\s*[^:]+::(\w+)"),
    ]
    arch_pattern = re.compile(r"__NPU_ARCH__\s*==\s*(\d+)")

    chips = set()
    chip_sources: dict[str, list[str]] = {}
    archs_from_content = set()
    arch_sources_content: dict[str, list[str]] = {}
    repo = getattr(extractor, "repo", None)

    for impl in impl_files:
        # 排除 arch 目录化文件
        if any(part.startswith("npu_arch_") or part.startswith("dav_") for part in impl.split("/")):
            continue
        full_path = os.path.join(repo, impl) if repo else impl
        if not os.path.isfile(full_path):
            continue
        try:
            with open(full_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except (IOError, OSError):
            continue

        local_chips = set()
        local_archs = set()

        for pat in chip_patterns:
            for m in pat.finditer(content):
                chip = m.group(1).strip().upper()
                if chip:
                    local_chips.add(chip)

        for m in arch_pattern.finditer(content):
            arch = m.group(1).strip()
            if arch:
                local_archs.add(arch)

        for chip in local_chips:
            chips.add(chip)
            chip_sources.setdefault(chip, []).append(impl)
        for arch in local_archs:
            archs_from_content.add(arch)
            arch_sources_content.setdefault(arch, []).append(impl)

    return (
        sorted(chips),
        chip_sources,
        sorted(archs_from_content),
        arch_sources_content,
    )


def _scan_arch_directories(e, api_type, api_name, impl_files, decl_file=None):
    """从 impl_files 路径提取 chip/arch 标识，规范化为纯数字；并验证每个 arch 下
    {api_name} 的实现完整性，剔除实现不完整的 arch。

    实现完整性（决策 C，三者俱全）：
      a. 底层 impl：arch 目录下存在文件名含 {api_name} 且内容含 {api_name}( 的 impl 文件
      b. 聚合头：arch 目录下 vector_compute_impl.h / reg_vector_impl.h 内容含 {api_name}(
      c. include 声明（仅 c_api/adv_api）：decl_file 内容含 {api_name}(
    c_api/adv_api 要求 a∧b∧c；其他类型要求 a∧b。

    返回 (arch_list, arch_sources)，仅含实现完整的 arch。
    """
    repo = getattr(e, "repo", None)
    api_call_pat = re.compile(rf"\b{re.escape(api_name)}\s*\(")  # 对外接口/声明 (asc_abs()
    api_impl_pat = re.compile(rf"\b{re.escape(api_name)}_impl\s*\(")  # 底层 impl 函数定义 (asc_abs_impl()

    def _read(rel_path: str) -> str:
        full = os.path.join(repo, rel_path) if repo else rel_path
        if not os.path.isfile(full):
            return ""
        try:
            with open(full, encoding="utf-8", errors="ignore") as f:
                return f.read()
        except (IOError, OSError):
            return ""

    # 第一遍：从路径提取所有候选 arch + 收集各 arch 下的 impl 文件
    arch_files: dict[str, list[str]] = {}
    for impl in impl_files:
        for part in impl.split("/"):
            if part.startswith("dav_"):
                arch = part[4:]  # dav_3510 → 3510
            elif part.startswith("npu_arch_"):
                arch = part[9:]  # npu_arch_2201 → 2201
            else:
                continue
            arch_files.setdefault(arch, []).append(impl)

    # 第二遍：对每个 arch 验证实现完整性
    arch_sources: dict[str, list[str]] = {}
    for arch, files in arch_files.items():
        has_impl = False  # a. 底层 impl
        has_aggregator = False  # b. 聚合头转发
        for impl in files:
            basename = os.path.basename(impl)
            content = _read(impl)
            if not content:
                continue
            # a. 底层 impl：文件名含 api_name 且内容含 {api_name}_impl( 的函数定义
            if api_name in basename and api_impl_pat.search(content):
                has_impl = True
            # b. 聚合头：arch 目录内任一文件含对外形态 {api_name}( 即视为已聚合/转发
            #    （原白名单仅 vector_compute_impl.h / reg_vector_impl.h，sys_var 等
            #     家族的聚合头是 sys_var_impl.h，按内容判定兼容各族目录布局；
            #     api_call_pat 只匹配对外形态，不会误匹配 {api_name}_impl(）
            if api_call_pat.search(content):
                has_aggregator = True
        # c. include 声明（仅 c_api/adv_api 检查；其他类型默认通过）
        has_decl = True
        if api_type in ("c_api", "adv_api"):
            has_decl = bool(decl_file) and bool(api_call_pat.search(_read(decl_file)))
        # 判定：c_api/adv_api 要求 a∧b∧c；其他要求 a∧b
        if api_type in ("c_api", "adv_api"):
            complete = has_impl and has_aggregator and has_decl
        else:
            complete = has_impl and has_aggregator
        if complete:
            arch_sources[arch] = files

    arch_list = sorted(arch_sources.keys(), key=lambda n: (0, int(n)) if n.isdigit() else (1, n))
    return arch_list, arch_sources


def _extract_header_wired_arches(decl_file, extractor, impl_files):
    """解析头文件 #if/__NPU_ARCH__==X 块，返回门控 #include 目标仍被定位的 arch 集合。

    逐行跟踪 #if/#elif/#else/#endif 块上下文，收集每块内的 #include 目标路径，
    与 impl_files（locator 定位到的、仍引用本 API 的文件）求交集：
    arch 有效 ⟺ 头文件门控该 arch 且 #include 的目标文件仍在 impl_files 中。
    （门控目标被清空/删除导致 locator 未定位 → 该 arch 不再接线 → 不算支持）
    """
    if not decl_file:
        return set()
    import os

    full_path = os.path.join(extractor.repo, decl_file)
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except (FileNotFoundError, PermissionError):
        return set()

    impl_set = {f for f in impl_files}
    wired = set()
    cur_archs: set = set()  # 当前 #if/#elif 块生效的 arch 号（可能多个 __NPU_ARCH__==X）
    depth = 0  # 嵌套深度，只在最外层 arch 块内收集 include
    for line in lines:
        s = line.strip()
        if s.startswith("#if"):
            depth += 1
            cur_archs = set(re.findall(r"__NPU_ARCH__\s*==\s*(\d+)", s))
        elif s.startswith("#elif"):
            cur_archs = set(re.findall(r"__NPU_ARCH__\s*==\s*(\d+)", s))
        elif s.startswith("#else"):
            cur_archs = set()  # #else 分支不绑定具体 arch 号
        elif s.startswith("#endif") and depth > 0:
            depth -= 1
            cur_archs = set()
        elif s.startswith("#include") and cur_archs and depth > 0:
            m = re.search(r'#include\s*[<"]([^>"]+)[>"]', s)
            if not m:
                continue
            target = m.group(1).strip()
            # 门控 #include 目标被 locator 定位 = 该 arch 仍有效接线
            if target in impl_set:
                wired |= cur_archs
    return wired
