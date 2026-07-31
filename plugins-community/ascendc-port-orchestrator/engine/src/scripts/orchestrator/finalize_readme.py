# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Finalize pipeline — archive README rendering (DEBT-201, 2026-07-06).

Extracted verbatim from finalize_pipeline.py: the human-readable archive
README + verification-conclusion rendering helpers. Pure (no finalize_pipeline
module-level dependency), none monkeypatched. finalize_pipeline re-imports these
names (bottom shim) so `finalize_pipeline._write_archive_readme` etc. and the
finalize_op call-site stay valid. Behaviour is byte-identical.
"""
from __future__ import annotations
import logging

import json
from pathlib import Path


def _render_verification_conclusion(vj: dict) -> str:
    """Render the customer-facing 「验证结论」 README block from verification.json.

    PB (2026-07-06, source-migration review): the prior render emitted
    4 fields with customer-broken values — empty `Verdict: —`, precision `PASS` without
    the pass-count strength, `None×` for N/A perf, and raw internal `truth_source` tokens.
    This produces 5 humanized fields. Provenance-honest: the reference truth-source is the
    part BEFORE the first `;` (the `ours=...` capture-method after `;` must NOT be misread
    as an A3-golden reference — e.g. a gelu port whose A3 capture failed + fell back to
    CPU-truth is honestly labeled CPU, not A3-golden).
    """
    prec = vj.get("precision") or {}
    perf = vj.get("performance") or {}
    det = vj.get("determinism") or {}
    prec_status = prec.get("status") or "—"
    pass_a = prec.get("pass_a") or {}

    # 总体 (synthesize when verdict absent)
    verdict = vj.get("verdict")
    perf_status = perf.get("status")
    if not verdict:
        if prec_status == "PASS":
            overall = "通过（精度达标"
            overall += "，性能待优化）" if perf_status == "BELOW_THRESHOLD" else "）"
        elif str(prec_status).startswith("PARTIAL"):
            overall = "部分通过"
        else:
            overall = str(prec_status)
    else:
        overall = str(verdict)

    # 精度 (+ tier1 count strength, tier2 if present)
    t1p, t1t = pass_a.get("tier1_pass"), pass_a.get("total")
    prec_line = str(prec_status)
    if t1p is not None and t1t is not None:
        prec_line += f"（严格 T1 {t1p}/{t1t}"
        t2 = pass_a.get("tier2_pass")
        if t2:
            prec_line += f"，硬件下限 T2 {t2}"
        prec_line += "）"

    # 确定性
    obs = det.get("observed_deterministic")
    det_line = "确定" if obs is True else ("非确定" if obs is False else "不适用")

    # 性能 (never render None×)
    ratio = perf.get("ratio")
    if isinstance(ratio, (int, float)):
        perf_line = f"{ratio:.2f}×（A5 相对参考基线）"
    elif perf_status in (None, "", "N/A", "NA"):
        perf_line = "N/A（移植/反向以 CPU 真值为参考，无性能基线）"
    else:
        perf_line = str(perf_status)

    # 参考真值来源 (reference part only, humanized)
    traw = str(vj.get("truth_source") or "").split(";")[0].strip().lower()
    if "autograd" in traw:
        truth_line = "torch.autograd 精确梯度真值(CPU/fp64)"
    elif "a3_capture" in traw or "aclnn" in traw or "a3_cann" in traw:
        truth_line = "A3-CANN 实测 golden"
    elif "cpu" in traw:
        truth_line = "CPU 真值(fp64，合成边界用例)"
    else:
        truth_line = vj.get("truth_source") or "—"

    return (
        f"\n## 验证结论\n\n"
        f"- **总体**: {overall}\n"
        f"- **精度**: {prec_line}\n"
        f"- **确定性**: {det_line}\n"
        f"- **性能**: {perf_line}\n"
        f"- **参考真值来源**: {truth_line}\n"
    )


def _assemble_readme(op, archive_dir, verdict_block: str) -> str:
    """Assemble the archive README body (extracted so verdict rendering is testable)."""

    # Categorize root files
    customer_files = []
    for entry in sorted(archive_dir.iterdir()):
        if entry.name.startswith(".") or entry.name == "docs":
            continue
        kind = "目录" if entry.is_dir() else "文件"
        customer_files.append(f"- `{entry.name}` ({kind})")

    has_install_caveat = (archive_dir / "INFRA_INSTALL_CAVEAT.md").exists()
    caveat_block = ""
    if has_install_caveat:
        caveat_block = (
            "\n## ⚠️ 安装路径备注\n\n"
            "见 [INFRA_INSTALL_CAVEAT.md](INFRA_INSTALL_CAVEAT.md) — "
            "本归档的 A5 binary 安装路径在 P9 边界, 需 P89 标准化 install agent。\n"
        )

    text = f"""# {op}

A5 (Ascend950PR / V351 / arch35) 端算子归档。
{verdict_block}{caveat_block}
## 根目录文件清单 (客户输出)

{chr(10).join(customer_files)}

## `docs/` 子目录

中文 API 文档 (上游 cann/ops-nn 同款格式)。

## `.harness/` 子目录

**非客户输出** — harness 内部状态 + 审计 + 中间产物:
- 状态机日志: `state_transitions.jsonl` / `orchestrator_events.jsonl`
- 自我审查: `audit_self_critic_post_worker.md` / `self_critic_report.md`
- 知识库反馈: `knowledge_update.md`
- 工作流元数据: `user_decision.md` / `op_classification.json` / `a3_reference_runnable.json`
- 中间产物 (若存在): probe 结果 / a5 captured binary outputs / determinism check / perf measurement scripts 等

阅读 `.harness/` 是 harness 诊断时才需要, 不影响算子使用。

---

*本 README 由 finalize_pipeline 自动生成于归档时。如需更详细的设计说明, 见 `analysis.md` + `PROGRESS.md`。*
"""
    (archive_dir / "README.md").write_text(text)


def _write_archive_readme(archive_dir: Path, op: str, workspace: Path) -> None:
    """Auto-generate README.md for the archive describing each root file/dir.

    Universal output format (2026-05-16): every archive has a README at root
    explaining what each file is + how to reproduce. Manually-curated README
    in workspace takes precedence (this only runs if archive lacks one).
    """
    vj_path = archive_dir / "verification.json"
    verdict_block = ""
    if vj_path.is_file():
        try:
            vj = json.loads(vj_path.read_text())
            verdict_block = _render_verification_conclusion(vj)
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
    return _assemble_readme(op, archive_dir, verdict_block)
