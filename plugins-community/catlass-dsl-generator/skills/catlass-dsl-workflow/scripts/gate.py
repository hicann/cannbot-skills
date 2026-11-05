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
"""catlass-dsl 主流程门禁检查。

只读脚本：不写状态文件，不执行算子，只校验工作区产物与 ITERATIONS.md 记录。
阶段顺序：design -> develop -> full test #1 -> optimize -> full test #2 -> benchmark -> finish。

用法:
    python3 gate.py design    --workspace <dir>
    python3 gate.py develop   --workspace <dir>
    python3 gate.py fulltest  --workspace <dir>
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
ORDER = ("design", "develop", "fulltest", "optimize", "benchmark", "finish")
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
        if cells[4].strip().lower() != "deferred":
            continue
        command = strip_ticks(cells[3])
        if not command or command in ("-", "n/a"):
            fail("{} 标记 deferred 但缺少精确重跑命令".format(context))
        return cells
    return None


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
    for row in workload:
        if not isinstance(row, dict) or not row.get("uuid"):
            fail("workload.jsonl 存在缺少 uuid 的行")
        if "inputs" not in row:
            fail("用例 {} 缺少 inputs".format(row["uuid"]))
        uuids.append(str(row["uuid"]))
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

    user_case_count = len(uuids)
    missing = sorted(set(uuids) - set(dev_uuids))
    if user_case_count < DEV_MIN:
        if missing:
            fail(
                "workload.jsonl 只有 {} 条用户用例（少于 {}），子集必须全部选入，缺少: {}".format(
                    user_case_count, DEV_MIN, ", ".join(missing[:5])
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
            if missing:
                fail(
                    "用户用例不超过 {} 条时子集必须全覆盖，缺少: {}".format(
                        DEV_MAX, ", ".join(missing[:5])
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
                cells[4].strip().lower(),
                cells,
            )
        )
    if not any(stage == "skeleton" for _, stage, _, _, _ in stages):
        fail("ITERATIONS.md 缺少 skeleton 里程碑")

    focused = [
        (index, cells)
        for index, stage, recorded, result, cells in stages
        if stage.startswith("focusedtest")
        and result == "passed"
        and sha_match(recorded, digest)
    ]
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
            "完成后把 focused test 记为 deferred（附精确重跑命令），其余情况必须真实执行".format(
                digest[:8]
            )
        )
    require_evidence(ws, focused[-1][1][5], "focused test")
    return "{}；focused test 通过".format(note)


def fulltest_rows(rows, digest, stages):
    matched = []
    for index, cells in rows:
        stage = normalize_stage(cells[1])
        result = cells[4].strip().lower()
        if stage in stages and result == "passed" and sha_match(cells[2], digest):
            matched.append((index, cells))
    return matched


FIRST_FULLTEST = ("fulltest", "fulltest1", "fulltest#1")
SECOND_FULLTEST = ("fulltest2", "fulltest#2")


def check_fulltest(ws):
    note = check_design(ws)
    _, digest = kernel_info(ws)
    _, sections = read_iterations(ws)
    rows = milestone_rows(sections)
    matched = fulltest_rows(rows, digest, FIRST_FULLTEST + SECOND_FULLTEST)
    if not matched:
        fail(
            "缺少当前源码（{}）的全量测试 passed 记录（full test #1 或 #2）；remote-detached "
            "模式最多只推进到代码生成完成，必须先连接远端并真实执行全量测试".format(
                digest[:8]
            )
        )
    require_evidence(ws, matched[-1][1][5], "full test")
    return "{}；当前 kernel 已有全量测试通过".format(note)


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
        if normalize_stage(cells[1]) in FIRST_FULLTEST
        and cells[4].strip().lower() == "passed"
    ]
    if not baseline:
        fail("缺少 full test #1 passed 记录；optimize 必须在 full test #1 通过后开始")
    require_evidence(ws, baseline[-1][1][5], "full test #1")
    opt_mode = optimization_mode(text)

    rounds = optimization_rounds(sections)
    required = 1 if opt_mode == "target" else 3
    if len(rounds) < required:
        fail(
            "模式 {} 需要至少 {} 轮已记录优化，当前 {} 轮；optimize 必须在远端/本机真实执行，"
            "remote-detached 模式不得推迟".format(opt_mode, required, len(rounds))
        )
    for _, cells, _ in rounds:
        require_evidence(ws, cells[6], "优化轮次 {}".format(strip_ticks(cells[0])))
    return "{}；模式 {} 已记录 {} 轮".format(note, opt_mode, len(rounds))


def check_benchmark(ws):
    note = check_optimize(ws)
    _, digest = kernel_info(ws)
    _, sections = read_iterations(ws)
    rows = milestone_rows(sections)
    if not fulltest_rows(rows, digest, FIRST_FULLTEST + SECOND_FULLTEST):
        fail(
            "缺少当前源码（{}）的全量测试 passed 记录（full test #1 或 #2）；优化未采纳、kernel "
            "未变化时可复用 full test #1，采纳优化改动了 kernel 的必须补跑 full test #2".format(
                digest[:8]
            )
        )

    if read_perf_demand(ws) != PERF_REQUESTED:
        return "{}；当前 kernel 全量测试通过，用户未要求性能，跳过 benchmark 与 profiling".format(
            note
        )

    benchmarks = [
        (index, cells)
        for index, cells in rows
        if normalize_stage(cells[1]).startswith("benchmark")
        and cells[4].strip().lower() == "passed"
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
    text, sections = read_iterations(ws)
    _, digest = kernel_info(ws)
    rows = milestone_rows(sections)

    deferred = [
        "{}（{}）".format(strip_ticks(cells[1]), strip_ticks(cells[3]))
        for _, cells in rows
        if cells[4].strip().lower() == "deferred"
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
        and cells[4].strip().lower() == "passed"
        and sha_match(cells[2], digest)
    ]
    if not reviews:
        fail("缺少当前源码（{}）的 final review passed 记录".format(digest[:8]))
    if "final verdict" not in text.lower():
        fail("ITERATIONS.md 缺少 Final Verdict 结论")
    return "{}；可以交付".format(note)


CHECKS = {
    "design": check_design,
    "develop": check_develop,
    "fulltest": check_fulltest,
    "optimize": check_optimize,
    "benchmark": check_benchmark,
    "finish": check_finish,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="catlass-dsl 工作流门禁检查（只读）")
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
