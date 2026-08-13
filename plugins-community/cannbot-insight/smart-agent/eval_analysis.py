#!/usr/bin/env python3

# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""审计质量评估：从轨迹确定性提取 ground truth（骨架/gates/skill_content），
给 analysis JSON 打客观分。用于对比 prompt 优化前后。

用法: python eval_analysis.py <trajectory.md> <analysis.json>
"""
import json
import logging
import sys
from pathlib import Path

from trajectory_parser import (
    extract_skeleton,
    extract_skill_content,
    extract_gates,
    parse_stats,
)
from trajectory_analyzer import _load_json_lenient

_log = logging.getLogger("eval_analysis")
if not _log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)
    _log.propagate = False


def _report_flow_ratio(flow_count: int, sk_call_count: int) -> None:
    ratio = flow_count / sk_call_count if sk_call_count else 0
    _log.info(f"\n[5] flow 节点数 vs skill 调用数: {flow_count} / {sk_call_count} = {ratio:.2f}")
    _log.info("    (1:1 规则期望 ~1.0；verifier dispatch+result 合并 gate 会略低；偏离大=非确定性)")


def _report_occurrences(sk, a_occ: dict) -> None:
    _log.info("\n[6] occurrences 匹配:")
    match = 0
    for skill, cnt in sk.occurrences.items():
        a = a_occ.get(skill)
        if a is not None and a == cnt:
            match += 1
        elif a is not None:
            _log.info(f"    {skill}: ground truth={cnt} 分析={a}  MISMATCH")
    _log.info(f"    匹配 {match}/{len(sk.occurrences)}")


def _report_gates(gates) -> None:
    gt_pass = sum(1 for g in gates if g.result == "PASS")
    gt_fail = sum(1 for g in gates if g.result == "FAIL")
    _log.info(
        f"\n[7] gates(ground truth): PASS={gt_pass} FAIL={gt_fail}  "
        "(分析应在 flow/skillQuality 体现)"
    )


def _report_summary(sn_ok: bool, recall_req: float, recall_sq: float,
                    precision_sq: float, ratio: float) -> None:
    _log.info("\n" + "=" * 60)
    _log.info(f"综合: skillName={'OK' if sn_ok else 'FAIL'} req召回={recall_req:.0f}% "
              f"sq召回={recall_sq:.0f}% sq精度={precision_sq:.0f}% flow比={ratio:.2f}")
    _log.info("=" * 60)


def _report_skillname(sc, wm) -> bool:
    sn_ok = wm.get("skillName") == sc.skill_name and bool(sc.skill_name)
    _log.info(f"\n[1] workflowMeta.skillName 正确: {'PASS' if sn_ok else 'FAIL'}")
    _log.info(f"    ground truth: {sc.skill_name!r}  分析: {wm.get('skillName')!r}")
    return sn_ok


def _report_precision(gt_skills, a_skills) -> float:
    matched = len(gt_skills & a_skills)
    precision_sq = matched / len(a_skills) * 100 if a_skills else 0
    hallucinated = a_skills - gt_skills
    _log.info(f"[4] skillQuality 精度: {precision_sq:.0f}%  "
              f"({matched}/{len(a_skills)} skills)")
    if hallucinated:
        _log.info(f"    幻觉 skill: {hallucinated}")
    return precision_sq


def eval_analysis(traj_path: str, analysis_path: str) -> None:
    text = Path(traj_path).read_text(encoding="utf-8")
    raw = Path(analysis_path).read_text(encoding="utf-8")
    d = _load_json_lenient(raw)

    # ── ground truth ──
    sk = extract_skeleton(text)
    sc = extract_skill_content(text)
    gates = extract_gates(text)
    stats = parse_stats(text)

    gt_skills = set(sk.occurrences.keys())                      # 轨迹里出现的所有 skill
    gt_dispatch = {e.skill for e in sk.skeleton if e.type == "dispatch"}  # dispatch 的 skill
    gt_invoke = {e.skill for e in sk.skeleton if e.type == "invoke"}     # invoke 的 skill
    sk_call_count = len(sk.skeleton)                            # *Skill: 标记总数

    # ── analysis ──
    wm = d.get("workflowMeta", {})
    req_skills = set(wm.get("requiredSkills", []))
    sq = d.get("skillQuality", [])
    a_skills = {s.get("skill") for s in sq}
    a_occ = {s.get("skill"): s.get("occurrences", 0) for s in sq}
    flow = d.get("flow", [])
    flow_count = len(flow)

    # ── 指标 ──
    _log.info("=" * 60)
    _log.info(f"轨迹: {Path(traj_path).name}  ({len(text.encode())//1024}KB)")
    _log.info(f"分析: {Path(analysis_path).name}  ({len(raw)//1024}KB)")
    _log.info("=" * 60)

    # 1. workflowMeta 正确性
    sn_ok = _report_skillname(sc, wm)

    # 2. requiredSkills 召回（dispatch 的 skill 是否都在 requiredSkills）
    if gt_dispatch:
        recall_req = len(gt_dispatch & req_skills) / len(gt_dispatch) * 100
    else:
        recall_req = 100.0
    got_req = len(gt_dispatch & req_skills)
    _log.info(f"\n[2] requiredSkills 召回: {recall_req:.0f}%  "
              f"({got_req}/{len(gt_dispatch)} dispatch skills)")
    missing_req = gt_dispatch - req_skills
    if missing_req:
        _log.info(f"    缺失: {missing_req}")

    # 3. skillQuality 召回（轨迹的 skill 是否都评了）
    recall_sq = len(gt_skills & a_skills) / len(gt_skills) * 100 if gt_skills else 0
    _log.info(f"\n[3] skillQuality 召回: {recall_sq:.0f}%  "
              f"({len(gt_skills & a_skills)}/{len(gt_skills)} skills)")
    missing_sq = gt_skills - a_skills
    if missing_sq:
        _log.info(f"    漏评: {missing_sq}")

    # 4. skillQuality 精度（分析的 skill 是否都在轨迹里，无幻觉）
    precision_sq = _report_precision(gt_skills, a_skills)

    _report_flow_ratio(flow_count, sk_call_count)
    _report_occurrences(sk, a_occ)
    _report_gates(gates)
    _report_summary(sn_ok, recall_req, recall_sq, precision_sq,
                    flow_count / sk_call_count if sk_call_count else 0)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        _log.info("用法: python eval_analysis.py <trajectory.md> <analysis.json>")
        sys.exit(1)
    eval_analysis(sys.argv[1], sys.argv[2])
