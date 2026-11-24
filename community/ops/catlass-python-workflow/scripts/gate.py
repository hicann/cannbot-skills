#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""catlass-python 主流程门禁检查。

只读脚本：不写状态文件，不执行算子，只校验工作区产物与 ITERATIONS.md 记录。
阶段顺序：design -> develop -> full test #1 -> merge -> full test #2 -> optimize ->
full test #3 -> benchmark -> finish。

用法:
    python3 gate.py design    --workspace <dir>
    python3 gate.py develop   --workspace <dir>
    python3 gate.py fulltest  --workspace <dir>
    python3 gate.py merge     --workspace <dir>
    python3 gate.py optimize  --workspace <dir>
    python3 gate.py benchmark --workspace <dir>
    python3 gate.py finish    --workspace <dir>
    python3 gate.py status    --workspace <dir>

退出码：0 通过，1 未通过。
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

DESIGN_FILES = (
    "DESIGN.md",
    "reference.py",
    "definition.json",
    "workload.jsonl",
    "workload_dev.jsonl",
)
DEV_MIN = 30
DEV_MAX = 40
ORDER = ("design", "develop", "fulltest", "merge", "optimize", "benchmark", "finish")
MODES = ("remote-detached", "remote-connected", "local")
REMOTE_DETACHED = "remote-detached"
MODE_PATTERN = re.compile(r"执行模式[\s*`]*[:：][\s*`]*([A-Za-z\-]+)")
PERF_REQUESTED = "requested"
PERF_NONE = "none"
PERF_PATTERN = re.compile(r"性能要求[\s*`]*[:：][\s*`]*([A-Za-z\-]+)")
HEADING = re.compile(r"^##\s+(.+?)\s*$")


class GateError(Exception):
    """门禁未通过。"""


def fail(message):
    raise GateError(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(path.read_bytes())


def strip_ticks(value):
    return str(value).strip().strip("`").strip()


def sha_match(recorded, actual):
    recorded = strip_ticks(recorded).lower()
    if len(recorded) < 8:
        return False
    actual = actual.lower()
    return (
        recorded == actual or actual.startswith(recorded) or recorded.startswith(actual)
    )


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail("{} 不是合法 JSON: {}".format(path.name, exc))


def load_jsonl(path):
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            fail("{} 第 {} 行不是合法 JSON: {}".format(path.name, lineno, exc))
    return rows


def read_iterations(ws):
    path = ws / "ITERATIONS.md"
    if not path.is_file():
        fail("缺少 ITERATIONS.md")
    text = path.read_text(encoding="utf-8")
    sections = []
    current = None
    for index, line in enumerate(text.splitlines()):
        match = HEADING.match(line)
        if match:
            current = (match.group(1).strip(), [])
            sections.append(current)
        elif current is not None:
            current[1].append((index, line))
    return text, sections


def table_rows(lines):
    rows = []
    for index, line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        if cells[0].lower() in ("iter", "round"):
            continue
        rows.append((index, cells))
    return rows


def section_rows(sections, name):
    name = name.lower()
    for title, lines in sections:
        if name in title.lower():
            return table_rows(lines)
    return []


def milestone_rows(sections):
    rows = section_rows(sections, "milestones")
    for _, cells in rows:
        if len(cells) < 6:
            fail("ITERATIONS.md 里程碑行列数不足 6 列: {}".format(" | ".join(cells)))
    return rows


def normalize_stage(value):
    return re.sub(r"[\s_\-]+", "", value.lower())


def status_of(cells):
    """里程碑表 Result 列的状态裸词，允许列内附带括号说明。

    只取开头的 ASCII 状态词，`deferred（等待远端）` 与 `deferred` 等价，
    说明文字放表格内还是表格外不影响门禁判定。
    """
    text = strip_ticks(cells[4]).strip().lower()
    match = re.match(r"[a-z_\-]+", text)
    return match.group(0) if match else ""


def require_evidence(ws, value, context):
    value = strip_ticks(value)
    if not value:
        fail("{} 缺少证据路径".format(context))
    if "*" in value or value in ("-", "n/a"):
        return
    if not (ws / value).exists():
        fail("{} 的证据路径不存在: {}".format(context, value))


def digest_candidates(path):
    raw = path.read_bytes()
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return {sha256_bytes(raw), sha256_bytes(normalized.encode("utf-8"))}


def design_digest_present(path, design_text):
    return any(digest in design_text.lower() for digest in digest_candidates(path))


def kernel_info(ws):
    path = ws / "solution.json"
    if not path.is_file():
        fail("缺少 solution.json")
    solution = load_json(path)
    if not isinstance(solution, dict):
        fail("solution.json 顶层必须是对象")
    entry = (solution.get("spec") or {}).get("entry_point")
    if not entry:
        fail("solution.json 缺少 spec.entry_point")
    relative = entry.split("::")[0].strip()
    kernel = ws / relative
    if not kernel.is_file():
        fail("solution.json 指向的实现文件不存在: {}".format(relative))

    digest = sha256_file(kernel)
    recorded = solution.get("kernel_sha256") or solution.get("source_sha256")
    if recorded:
        if not sha_match(recorded, digest):
            fail("solution.json 记录的源码摘要与实现文件不一致")
        return kernel, digest

    sources = solution.get("sources") or []
    if not sources:
        fail("solution.json 缺少 sources")
    content = sources[0].get("content")
    if not isinstance(content, str) or not content.strip():
        fail("solution.json sources[0].content 为空")
    actual = kernel.read_text(encoding="utf-8").replace("\r\n", "\n")
    if sha256_bytes(content.replace("\r\n", "\n").encode("utf-8")) != sha256_bytes(
        actual.encode("utf-8")
    ):
        fail("solution.json sources[0].content 与实现文件内容不一致")
    return kernel, digest


def read_mode(ws):
    design = ws / "DESIGN.md"
    if not design.is_file():
        return None
    match = MODE_PATTERN.search(design.read_text(encoding="utf-8"))
    if not match:
        fail(
            "DESIGN.md 缺少执行模式声明，需写明 "
            "`执行模式：remote-detached | remote-connected | local`"
        )
    mode = match.group(1).lower()
    if mode not in MODES:
        fail("执行模式非法: {}（只允许 {}）".format(mode, " / ".join(MODES)))
    return mode


def read_perf_demand(ws):
    design = ws / "DESIGN.md"
    if not design.is_file():
        return PERF_NONE
    match = PERF_PATTERN.search(design.read_text(encoding="utf-8"))
    if not match:
        fail("DESIGN.md 缺少性能要求声明，需写明 `性能要求：requested | none`")
    value = match.group(1).lower()
    if value not in (PERF_REQUESTED, PERF_NONE):
        fail("性能要求非法: {}（只允许 requested / none）".format(value))
    return value


def deferred_row(rows, prefixes, context, exclude=()):
    for _, cells in rows:
        stage = normalize_stage(cells[1])
        if stage in exclude:
            continue
        if not any(stage.startswith(prefix) for prefix in prefixes):
            continue
        if status_of(cells) != "deferred":
            continue
        command = strip_ticks(cells[3])
        if not command or command in ("-", "n/a"):
            fail("{} 标记 deferred 但缺少精确重跑命令".format(context))
        return cells
    return None


def _norm_name(value):
    return re.sub(r"[^0-9a-z]", "", str(value).lower())


def design_has_mmad(ws):
    path = ws / "DESIGN.md"
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    match = re.search(r"含\s*MMAD[：:]\s*`?\s*(是|否)", text, re.IGNORECASE)
    return bool(match and match.group(1) == "是")


MATMUL_LIST_PATTERNS = (
    # 「包含的 matmul 阶段」是概览，「全部 matmul 阶段清单」是逐行明细；
    # 两处都读，任一处漏写阶段都能被发现，避免清单缩水把达到率分母做小。
    r"包含的\s*matmul\s*阶段[^\n]*[:：]\s*(.+)",
    r"全部\s*matmul\s*阶段清单[^\n]*[:：]\s*(.+)",
)

STAGE_NAME = re.compile(r"^([A-Za-z][A-Za-z0-9_\-]*)\s*[`*。.,，；;:：]*$")


def _stage_names_from_line(line):
    """从清单行里取阶段名。

    两种写法都要支持：
    - 明细行：每项是 `名称 / 所在 kernel / 占比 / shape / 来源`，取第一个 `/` 前；
    - 概览行：可能整行就是 `QK / PV / dK / dV` 或 `a、b、c`，直接按名字切。
    模板把整行包在一个反引号里、内部用 `；` 分隔多项，所以先整行切分再剥反引号；
    行尾说明文字不算阶段名。
    """
    parts = re.split(r"[、；;]", line)
    detailed = len(parts) > 1 or any(ch in line for ch in "%(=≈")
    names = []
    if detailed:
        for part in parts:
            part = part.strip().strip("`").strip()
            if not part or part.startswith("<"):
                continue
            head = part.split("/")[0].strip().strip("`").strip()
            match = STAGE_NAME.match(head)
            if match:
                names.append(match.group(1))
        return names
    for token in re.split(r"/", line):
        token = token.strip().strip("`").strip()
        match = STAGE_NAME.match(token) if token and not token.startswith("<") else None
        if match:
            names.append(match.group(1))
    return names


def design_matmul_names(ws):
    """取 DESIGN 登记的 matmul 阶段名（概览行与逐行清单合并去重）。"""
    path = ws / "DESIGN.md"
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    names = []
    seen = set()
    for pattern in MATMUL_LIST_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            for name in _stage_names_from_line(match.group(1)):
                key = _norm_name(name)
                if key and key not in seen:
                    seen.add(key)
                    names.append(name)
    return names


def check_matmul_spec_coverage(ws):
    """DESIGN 的 matmul 清单必须与骨架 spec 一一对应，避免分母被漏登记的阶段缩小。"""
    if not design_has_mmad(ws):
        return
    spec_path = ws / "tools" / "skeleton_spec.json"
    if not spec_path.is_file():
        fail(
            "DESIGN 声明含 MMAD，但缺少 tools/skeleton_spec.json（每份 GEMM 骨架的清单）"
        )
    data = load_json(spec_path)
    stages = [item for item in (data or {}).get("stages", []) if isinstance(item, dict)]
    names = [item.get("name") for item in stages if item.get("name")]
    if not names:
        fail("tools/skeleton_spec.json 没有有效的 stage name")
    design_text = (ws / "DESIGN.md").read_text(encoding="utf-8")
    design_lower = design_text.lower()
    missing = [name for name in names if name.lower() not in design_lower]
    if missing:
        fail(
            "骨架 spec 的 stage 未在 DESIGN 登记：{}；DESIGN 的「全部 matmul 阶段清单」"
            "要逐行列出每个 matmul，名称与 spec 一致".format("、".join(missing))
        )
    spec_norms = [_norm_name(name) for name in names]
    extra = [
        name
        for name in design_matmul_names(ws)
        if _norm_name(name)
        and not any(
            _norm_name(name) in item or item in _norm_name(name) for item in spec_norms
        )
    ]
    if extra:
        fail(
            "DESIGN 登记的 matmul 阶段没有对应的骨架 spec：{}；每个 matmul 都要各自跑一次"
            "骨架评测，否则达成率的分母会被漏测阶段缩小".format("、".join(extra))
        )


def check_design(ws):
    missing = [name for name in DESIGN_FILES if not (ws / name).is_file()]
    if missing:
        fail("设计交付物缺失: {}".format(", ".join(missing)))
    for name in DESIGN_FILES:
        if not (ws / name).read_text(encoding="utf-8").strip():
            fail("设计交付物为空: {}".format(name))

    definition = load_json(ws / "definition.json")
    if not isinstance(definition, dict):
        fail("definition.json 顶层必须是对象")
    if definition.get("axes") not in ({}, None):
        fail("definition.json 的 axes 必须为空对象")
    if not definition.get("outputs"):
        fail("definition.json 缺少 outputs")
    reference = definition.get("reference")
    if not isinstance(reference, str) or not reference.strip():
        fail("definition.json 缺少 reference 源码")

    workload = load_jsonl(ws / "workload.jsonl")
    if not workload:
        fail("workload.jsonl 没有用例")
    uuids = []
    user_uuids = []
    for row in workload:
        if not isinstance(row, dict) or not row.get("uuid"):
            fail("workload.jsonl 存在缺少 uuid 的行")
        if "inputs" not in row:
            fail("用例 {} 缺少 inputs".format(row["uuid"]))
        origin = row.get("origin")
        if origin not in (None, "user", "agent_supplemented"):
            fail("用例 {} 的 origin 非法: {}".format(row["uuid"], origin))
        uuid = str(row["uuid"])
        uuids.append(uuid)
        if origin == "user":
            user_uuids.append(uuid)
    if len(set(uuids)) != len(uuids):
        fail("workload.jsonl 存在重复 uuid")

    dev = load_jsonl(ws / "workload_dev.jsonl")
    dev_uuids = []
    for row in dev:
        if not isinstance(row, dict) or not row.get("uuid"):
            fail("workload_dev.jsonl 存在缺少 uuid 的行")
        uuid = str(row["uuid"])
        if uuid in dev_uuids:
            fail("workload_dev.jsonl 存在重复 uuid: {}".format(uuid))
        origin = row.get("origin")
        if origin not in (None, "user", "agent_supplemented"):
            fail("用例 {} 的 origin 非法: {}".format(uuid, origin))
        if origin == "user":
            if uuid not in uuids:
                fail("用户用例 {} 不在 workload.jsonl 中".format(uuid))
            if row.get("source") != uuid:
                fail("用户用例 {} 的 source 必须等于 uuid".format(uuid))
        if origin == "agent_supplemented" and row.get("source"):
            fail("补充用例 {} 不得声明 source".format(uuid))
        dev_uuids.append(uuid)

    user_case_count = len(user_uuids)
    missing_user = sorted(set(user_uuids) - set(dev_uuids))
    if user_case_count < DEV_MIN:
        if missing_user:
            fail(
                "用户用例只有 {} 条（少于 {}），子集必须全部选入，缺少: {}".format(
                    user_case_count, DEV_MIN, ", ".join(missing_user[:5])
                )
            )
        if len(dev) > DEV_MAX:
            fail(
                "workload_dev.jsonl 最多 {} 条用例，当前 {} 条".format(
                    DEV_MAX, len(dev)
                )
            )
    else:
        if not DEV_MIN <= len(dev) <= DEV_MAX:
            fail(
                "workload_dev.jsonl 需要 {}~{} 条用例，当前 {} 条".format(
                    DEV_MIN, DEV_MAX, len(dev)
                )
            )
        if user_case_count <= DEV_MAX:
            if missing_user:
                fail(
                    "用户用例不超过 {} 条时子集必须全覆盖，缺少: {}".format(
                        DEV_MAX, ", ".join(missing_user[:5])
                    )
                )
        else:
            selected = sum(1 for row in dev if str(row.get("origin")) == "user")
            if selected < DEV_MIN:
                fail(
                    "用户用例超过 {} 条时子集至少包含 {} 条用户用例，当前 {} 条".format(
                        DEV_MAX, DEV_MIN, selected
                    )
                )

    design_text = (ws / "DESIGN.md").read_text(encoding="utf-8")
    lowered = design_text.lower()
    for keyword in ("dsl 版本", "benchmark", "性能要求"):
        if keyword not in lowered:
            fail("DESIGN.md 缺少必备内容: {}".format(keyword))
    if not MERGE_DECLARED.search(design_text):
        fail(
            "DESIGN.md 未登记 merge_required（true / false）；该字段与用户是否要求单 kernel "
            "正交，必须以独立一行写出（自然划分即单 kernel 或用户未要求单 kernel 时写 false），"
            "不要只写在说明句里"
        )
    for name in ("reference.py", "definition.json", "workload.jsonl"):
        if not design_digest_present(ws / name, design_text):
            fail("DESIGN.md 未记录 {} 的 SHA-256".format(name))
    mode = read_mode(ws)
    perf = read_perf_demand(ws)
    return "设计交付物齐备（dev 子集 {} 条，执行模式 {}，性能要求 {}）".format(
        len(dev), mode, perf
    )


def check_develop(ws):
    note = check_design(ws)
    _, digest = kernel_info(ws)
    _, sections = read_iterations(ws)
    rows = milestone_rows(sections)

    stages = []
    for index, cells in rows:
        stages.append(
            (
                index,
                normalize_stage(cells[1]),
                strip_ticks(cells[2]),
                status_of(cells),
                cells,
            )
        )
    if not any(stage == "skeleton" for _, stage, _, _, _ in stages):
        fail("ITERATIONS.md 缺少 skeleton 里程碑")

    check_matmul_spec_coverage(ws)

    focused = [
        (index, cells)
        for index, stage, recorded, result, cells in stages
        if stage.startswith("focusedtest")
        and result == "passed"
        and sha_match(recorded, digest)
    ]
    if focused and read_mode(ws) == REMOTE_DETACHED:
        skeleton_cells = []
        for _, stage, _, _, cells in stages:
            if stage == "skeleton":
                skeleton_cells.append(cells)
        waived = any(
            "不适用" in strip_ticks(cells[4]) or "n/a" in strip_ticks(cells[4]).lower()
            for cells in skeleton_cells
        )
        if not waived and not any(
            status_of(cells) == "passed" for cells in skeleton_cells
        ):
            fail(
                "remote-detached 下 skeleton 未 passed（当前 {}），却已有当前源码的 focused test "
                "passed 记录；远端首跑必须先执行 tools/skeleton_gemm.py，把判定与 Skeleton "
                "Performance 表回填后再验主实现".format(
                    " / ".join(status_of(cells) or "空" for cells in skeleton_cells)
                )
            )
    if not focused:
        if fulltest_rows(rows, digest, SECOND_FULLTEST):
            return "{}；当前源码已有 full test #2 记录".format(note)
        mode = read_mode(ws)
        if mode == REMOTE_DETACHED and deferred_row(
            rows, ("focusedtest",), "focused test"
        ):
            return "{}；focused test 已推迟（代码生成完成，等待远端连接）".format(note)
        fail(
            "缺少当前源码（{}）的 focused test passed 记录；只有 remote-detached 模式可在代码生成"
            "完成后把 focused test 记为 deferred（Command 列附精确重跑命令）。Result 列须以状态"
            "裸词开头（passed / failed / blocked / not_run / deferred）".format(
                digest[:8]
            )
        )
    require_evidence(ws, focused[-1][1][5], "focused test")
    return "{}；focused test 通过".format(note)


def fulltest_rows(rows, digest, stages):
    matched = []
    for index, cells in rows:
        stage = normalize_stage(cells[1])
        result = status_of(cells)
        if stage in stages and result == "passed" and sha_match(cells[2], digest):
            matched.append((index, cells))
    return matched


FIRST_FULLTEST = ("fulltest", "fulltest1", "fulltest#1")
SECOND_FULLTEST = ("fulltest2", "fulltest#2")
THIRD_FULLTEST = ("fulltest3", "fulltest#3")
ALL_FULLTEST = FIRST_FULLTEST + SECOND_FULLTEST + THIRD_FULLTEST


def any_fulltest_rows(rows, stages):
    """不按 SHA 过滤的全量测试记录，用于校验融合前的多 kernel 基线。"""
    return [
        (index, cells)
        for index, cells in rows
        if normalize_stage(cells[1]) in stages and status_of(cells) == "passed"
    ]


BASELINE_SHA = re.compile(
    r"baseline(?:_sha)?\s*[:=]\s*`?([0-9a-fA-F]{8,64})`?", re.IGNORECASE
)


def check_fulltest(ws):
    note = check_design(ws)
    _, digest = kernel_info(ws)
    _, sections = read_iterations(ws)
    rows = milestone_rows(sections)
    matched = fulltest_rows(rows, digest, ALL_FULLTEST)
    if not matched:
        fail(
            "缺少当前源码（{}）的全量测试 passed 记录（full test #1 / #2 / #3）；remote-detached "
            "模式最多只推进到代码生成完成，必须先连接远端并真实执行全量测试".format(
                digest[:8]
            )
        )
    require_evidence(ws, matched[-1][1][5], "full test")
    return "{}；当前 kernel 已有全量测试通过".format(note)


MERGE_REQUIRED = re.compile(
    r"^\s*[-*]?\s*`?merge_required`?\s*[:：]\s*`?(?:true|yes)`?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
MERGE_DECLARED = re.compile(
    r"^\s*[-*]?\s*`?merge_required`?\s*[:：]\s*`?(?:true|false|yes|no)`?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def merge_is_fused(cells):
    """Result passed 且未在 Command/Result 里标注 not_applicable 才算真正做过融合。"""
    return status_of(cells) == "passed" and "not_applicable" not in (
        cells[3] + " " + cells[4]
    ).lower()


def check_merge(ws):
    """merge 阶段验收门禁：应在融合版 focused 与 full test #2 之后执行。"""
    note = check_design(ws)
    _, digest = kernel_info(ws)
    _, sections = read_iterations(ws)
    rows = milestone_rows(sections)
    design_text = (ws / "DESIGN.md").read_text(encoding="utf-8")

    merges = [
        (index, cells)
        for index, cells in rows
        if normalize_stage(cells[1]).startswith("merge")
        and merge_is_fused(cells)
        and sha_match(cells[2], digest)
    ]
    if not MERGE_REQUIRED.search(design_text):
        skipped = [
            (index, cells)
            for index, cells in rows
            if normalize_stage(cells[1]).startswith("merge")
            and status_of(cells) in ("passed", "not_applicable")
            and sha_match(cells[2], digest)
        ]
        if skipped:
            return "{}；DESIGN 未声明 merge_required: true，融合阶段记 not_applicable".format(
                check_fulltest(ws)
            )
        stale = [
            (index, cells)
            for index, cells in rows
            if normalize_stage(cells[1]).startswith("merge")
            and status_of(cells) == "passed"
        ]
        if stale:
            fail(
                "ITERATIONS.md 已有 merge 记录，但 Kernel SHA（{}）与当前源码（{}）不一致；"
                "当前源码相对融合版发生实质变化时必须补跑 full test #3 后再交付".format(
                    strip_ticks(stale[-1][1][2])[:8], digest[:8]
                )
            )
        fail(
            "DESIGN 未声明 merge_required: true，但 ITERATIONS.md 缺少 merge 里程碑；"
            "需追加一行 Stage=merge，Result 写 passed 并在 Command 列注明 "
            "not_applicable（自然划分即单 kernel 或用户未要求单 kernel）"
        )

    if merges:
        cells = merges[-1][1]
        require_evidence(ws, cells[5], "merge")
        focused = [
            (index, cells)
            for index, cells in rows
            if normalize_stage(cells[1]).startswith("focusedtest")
            and status_of(cells) == "passed"
            and sha_match(cells[2], digest)
        ]
        if not focused:
            fail(
                "merge 阶段缺少融合版（{}）的 focused test passed 记录；融合后必须先跑 focused，"
                "再进入 full test #2".format(digest[:8])
            )
        require_evidence(ws, focused[-1][1][5], "merge 阶段的 focused test")
        if not fulltest_rows(rows, digest, SECOND_FULLTEST):
            fail(
                "缺少融合版（{}）的 full test #2 passed 记录；merge 门禁在 FT#2 之后执行，"
                "融合版必须先跑 focused 与 full test #2".format(digest[:8])
            )
        baseline = BASELINE_SHA.search(" ".join(cells[3:5]))
        if baseline:
            baseline_sha = baseline.group(1)
            if baseline_sha.lower() == digest.lower():
                fail(
                    "merge 记录声明的 baseline（{}）与当前源码相同；融合前后 kernel SHA 必须不同，"
                    "请把 baseline 指向 full test #1 通过的多 kernel 版本".format(
                        baseline_sha[:8]
                    )
                )
            if not fulltest_rows(rows, baseline_sha, FIRST_FULLTEST):
                fail(
                    "merge 记录声明 baseline={}，但该 SHA 上没有 full test #1 passed 记录；"
                    "融合前必须先让自然划分的多 kernel 版通过 full test #1".format(
                        baseline_sha[:8]
                    )
                )
        else:
            distinct = [
                (index, row_cells)
                for index, row_cells in any_fulltest_rows(rows, FIRST_FULLTEST)
                if not sha_match(row_cells[2], digest)
            ]
            if not distinct:
                fail(
                    "缺少融合前多 kernel 基线的 full test #1 passed 记录；融合前必须先让自然划分的"
                    "多 kernel 版通过 full test #1，并在 merge 行的 Command 列用 baseline=<sha> "
                    "注明基线 SHA，否则无法证明融合版与已验证基线一致"
                )
        return "{}；融合版 merge、focused、full test #2 与基线 full test #1 齐备".format(note)

    history = [
        (index, cells)
        for index, cells in rows
        if normalize_stage(cells[1]).startswith("merge") and merge_is_fused(cells)
    ]
    if history:
        merge_sha = strip_ticks(history[-1][1][2])
        require_evidence(ws, history[-1][1][5], "merge")
        if not fulltest_rows(rows, merge_sha, SECOND_FULLTEST):
            fail(
                "merge 记录（{}）缺少对应的 full test #2 passed 记录；融合版必须先跑 full test #2".format(
                    merge_sha[:8]
                )
            )
        if fulltest_rows(rows, digest, THIRD_FULLTEST):
            return "{}；merge 在 {} 完成，当前源码 {} 已补 full test #3，按 optimize 复验继承".format(
                note, merge_sha[:8], digest[:8]
            )
        fail(
            "缺少当前源码（{}）的 merge passed 记录；融合已在 {} 完成，当前源码相对融合版发生变化，"
            "必须补跑 full test #3 后再交付".format(digest[:8], merge_sha[:8])
        )

    fail(
        "缺少当前源码（{}）的 merge passed 记录；DESIGN 声明 merge_required: true 时，"
        "必须先以 full test #1 通过的多 kernel 版本为基线完成融合，再进入 full test #2".format(
            digest[:8]
        )
    )


def check_merge_history(ws):
    """finish 侧校验融合确实发生过：只按历史记录，不要求与交付 kernel 同一 SHA。"""
    design_text = (ws / "DESIGN.md").read_text(encoding="utf-8")
    if not MERGE_REQUIRED.search(design_text):
        return "merge 非必需"
    _, sections = read_iterations(ws)
    rows = milestone_rows(sections)
    completed = [
        (index, cells)
        for index, cells in rows
        if normalize_stage(cells[1]).startswith("merge") and merge_is_fused(cells)
    ]
    if not completed:
        fail(
            "DESIGN 声明 merge_required: true，但没有 merge passed 记录；必须先完成"
            "多 kernel 到单 kernel 的融合再交付"
        )
    index, cells = completed[-1]
    require_evidence(ws, cells[5], "merge")
    merge_sha = strip_ticks(cells[2])
    if not fulltest_rows(rows, merge_sha, SECOND_FULLTEST):
        fail(
            "merge 记录（{}）缺少对应的 full test #2 passed 记录；融合版必须先跑 full test #2".format(
                merge_sha[:8]
            )
        )
    return "merge 已完成（融合版 full test #2 通过）"


def optimization_mode(text):
    match = re.search(r"^[-*]\s*mode\s*:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    if not match:
        fail("ITERATIONS.md 缺少 Optimization Policy 的 mode 行（target / explore）")
    value = match.group(1).lower()
    has_target = "target" in value
    has_explore = "explore" in value
    if has_target == has_explore:
        fail(
            "Optimization Policy 的 mode 必须明确写成 target 或 explore，当前为: {}".format(
                match.group(1).strip()
            )
        )
    return "target" if has_target else "explore"


def optimization_rounds(sections):
    rounds = []
    for index, cells in section_rows(sections, "performance rounds"):
        if len(cells) < 7:
            fail("Performance Rounds 行列数不足 7 列: {}".format(" | ".join(cells)))
        decision = cells[5].strip().lower()
        if decision not in ("adopted", "rejected"):
            continue
        if (
            not strip_ticks(cells[1])
            or not strip_ticks(cells[2])
            or not strip_ticks(cells[6])
        ):
            fail("优化轮次记录不完整: {}".format(" | ".join(cells)))
        rounds.append((index, cells, decision))
    return rounds


def check_optimize(ws):
    note = check_design(ws)
    text, sections = read_iterations(ws)
    rows = milestone_rows(sections)
    baseline = [
        (index, cells)
        for index, cells in rows
        if normalize_stage(cells[1]) in FIRST_FULLTEST + SECOND_FULLTEST
        and status_of(cells) == "passed"
    ]
    if not baseline:
        fail(
            "缺少 full test #1 / #2 passed 记录；optimize 必须在多 kernel 版（#1）或"
            "融合版（#2）全量测试通过后开始"
        )
    require_evidence(ws, baseline[-1][1][5], "full test")
    opt_mode = optimization_mode(text)

    rounds = optimization_rounds(sections)
    required = 1 if opt_mode == "target" else 2
    early_stop = re.search(
        r"^[-*]\s*early_stop\s*:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE
    )
    if early_stop:
        if "skeleton_ratio" not in early_stop.group(1).lower():
            fail(
                "early_stop 只支持 skeleton_ratio 口径，当前为: {}".format(
                    early_stop.group(1).strip()
                )
            )
        threshold = re.search(
            r"skeleton_ratio\s*>=\s*([0-9.]+)", early_stop.group(1), re.IGNORECASE
        )
        if not threshold:
            fail(
                "early_stop 必须写成 `skeleton_ratio >= 0.80` 的形式，当前为: {}".format(
                    early_stop.group(1).strip()
                )
            )
        if float(threshold.group(1)) < 0.80:
            fail(
                "early_stop 门槛不能低于 0.80（当前 {}）；门槛由 skill 固定，"
                "不得按实验情况自行降低".format(threshold.group(1))
            )
        required = 0
    if len(rounds) < required:
        fail(
            "模式 {} 需要至少 {} 轮已记录优化，当前 {} 轮；optimize 必须在远端/本机真实执行，"
            "remote-detached 模式不得推迟（命中 early_stop: skeleton_ratio 时该下限为 0）".format(
                opt_mode, required, len(rounds)
            )
        )
    for _, cells, _ in rounds:
        require_evidence(ws, cells[6], "优化轮次 {}".format(strip_ticks(cells[0])))
    return "{}；模式 {} 已记录 {} 轮".format(note, opt_mode, len(rounds))


def check_benchmark(ws):
    note = check_optimize(ws)
    _, digest = kernel_info(ws)
    _, sections = read_iterations(ws)
    rows = milestone_rows(sections)
    if not fulltest_rows(rows, digest, ALL_FULLTEST):
        fail(
            "缺少当前源码（{}）的全量测试 passed 记录（full test #1 / #2 / #3）；优化未采纳、"
            "kernel 未变化时可复用最近一次记录，采纳优化改动了 kernel 的必须补跑 full test #3"
            "（无融合阶段时记全量测试 full test #2）".format(digest[:8])
        )

    if read_perf_demand(ws) != PERF_REQUESTED:
        return "{}；当前 kernel 全量测试通过，用户未要求性能，跳过 benchmark 与 profiling".format(
            note
        )

    benchmarks = [
        (index, cells)
        for index, cells in rows
        if normalize_stage(cells[1]).startswith("benchmark")
        and status_of(cells) == "passed"
        and sha_match(cells[2], digest)
    ]
    if not benchmarks:
        fail(
            "缺少当前源码（{}）的 benchmark passed 记录；benchmark 必须在真实设备执行".format(
                digest[:8]
            )
        )
    require_evidence(ws, benchmarks[-1][1][5], "benchmark")
    return "{}；全量测试与 benchmark 齐备".format(note)


def check_finish(ws):
    note = check_benchmark(ws)
    merge_note = check_merge_history(ws)
    text, sections = read_iterations(ws)
    _, digest = kernel_info(ws)
    rows = milestone_rows(sections)

    deferred = [
        "{}（{}）".format(strip_ticks(cells[1]), strip_ticks(cells[3]))
        for _, cells in rows
        if status_of(cells) == "deferred"
    ]
    if deferred:
        fail(
            "ITERATIONS.md 仍有 {} 条 deferred 里程碑: {}；finish 不允许推迟，必须在真实环境补跑并改为 "
            "passed".format(len(deferred), "、".join(deferred))
        )
    reviews = [
        (index, cells)
        for index, cells in rows
        if normalize_stage(cells[1]).startswith("finalreview")
        and status_of(cells) == "passed"
        and sha_match(cells[2], digest)
    ]
    if not reviews:
        fail("缺少当前源码（{}）的 final review passed 记录".format(digest[:8]))
    if "final verdict" not in text.lower():
        fail("ITERATIONS.md 缺少 Final Verdict 结论")
    return "{}；{}；可以交付".format(note, merge_note)


CHECKS = {
    "design": check_design,
    "develop": check_develop,
    "fulltest": check_fulltest,
    "merge": check_merge,
    "optimize": check_optimize,
    "benchmark": check_benchmark,
    "finish": check_finish,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="catlass-python 工作流门禁检查（只读）")
    parser.add_argument("stage", choices=ORDER + ("status",))
    parser.add_argument("--workspace", default=".")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        print("[gate] 工作区不存在: {}".format(workspace))
        return 1

    if args.stage == "status":
        failed = 0
        for name in ORDER:
            try:
                note = CHECKS[name](workspace)
                print("[gate] {:<9} PASS  {}".format(name, note))
            except GateError as exc:
                failed += 1
                print("[gate] {:<9} FAIL  {}".format(name, exc))
        return 1 if failed else 0

    try:
        note = CHECKS[args.stage](workspace)
    except GateError as exc:
        print("[gate] {} FAIL".format(args.stage))
        print("  - {}".format(exc))
        return 1
    print("[gate] {} PASS  {}".format(args.stage, note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
