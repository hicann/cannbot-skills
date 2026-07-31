# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Finalize pipeline — GE op_host assembly (DEBT-201, 2026-07-06).

Extracted verbatim from finalize_pipeline.py: FA GE-op_host template assembly
+ its CANN md5-index helpers and template constants. Pure (no finalize_pipeline
module-level dependency), none monkeypatched. finalize_pipeline re-imports these
names (bottom shim) so `finalize_pipeline.assemble_ge_ophost` and all call-sites
stay valid. Behaviour is byte-identical.
"""
from __future__ import annotations
import logging

import hashlib
import json
from pathlib import Path

# Repo root, computed identically to finalize_pipeline._PROJECT_ROOT (both files
# live in src/scripts/orchestrator/, so the .parent.parent.parent.parent chain
# resolves to the same path). Defined locally to avoid importing from
# finalize_pipeline (would reintroduce the import cycle this split removes).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


_FA_GE_OPHOST_TEMPLATE_DIR = (
    _PROJECT_ROOT.parent / "kb" / "target" / "ascendc"
    / "patterns" / "domains" / "fa_class" / "templates" / "op_host"
)

_FA_GE_TEMPLATE_OP_TOKEN = "flash_attention_score"

_FA_GE_CPP_SUFFIXES = ("_def", "_infershape", "_tiling")

_FA_GE_HDR_SUFFIXES = ("_tiling_common",)

_FA_GE_SHARED_HEADERS = (
    "wp_fa_host_tiling.h",
    "ge_host_shim.h",
    "wp_fa_host_cache.h",
)


def _md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _cann_md5_index_for_basenames(basenames: set) -> dict:
    """Build basename -> set-of-md5 over the CANN attention/FA subtree (or whole
    tree fallback). Returns {} when CANN source is absent (customer machine).
    Used by the assembler's idempotency check (don't treat a CANN-copy as a
    legitimate worker emit) and shared with the gate's byte check."""
    import os as _os
    cann_root = Path(_os.environ.get(
        "CANN_SOURCE_ROOT", str(Path.home() / "workspace" / "cann")
    ))
    if not cann_root.is_dir():
        return {}
    fa_subtree = cann_root / "ops-transformer" / "attention"
    root = fa_subtree if fa_subtree.is_dir() else cann_root
    idx: dict = {}
    for cf in root.rglob("*"):
        if not cf.is_file() or cf.suffix not in (".cpp", ".h"):
            continue
        if cf.name not in basenames:
            continue
        skip_current_item = False
        try:
            idx.setdefault(cf.name, set()).add(_md5_bytes(cf.read_bytes()))
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
    return idx


def assemble_ge_ophost(workspace: Path) -> dict:
    """flash_attention_score-ghasm-1 (2026-06-11, owner mandate): DETERMINISTIC
    GE op_host assembler. The last gap for the port_a3 FA blackbox真闭环.

    Problem: the port_a3 FA blackbox reaches `done` + promotes, but the archive
    `op_host/` ships ONLY shared headers — NO GE def/infershape/tiling.cpp. Root
    cause: the kw_brief `_fa_ge_host_gen_block` was a BRIEF INSTRUCTION (LLM-
    skippable), not a hard gate; the worker focused on kernel+pybind+precision
    and skipped emitting the GE op_host .cpp. The KB already carries the correct,
    compile-verified, non-CANN GE op_host templates (commits b29ffb34/1dd88b1e)
    under `fa_class/templates/op_host/` — the pipeline just isn't DELIVERING them.

    Why NOT prestage (phase_o25_a3_ref): the kernel build chain compiles
    op_host/*.cpp, but the GE .cpp need the full GE/op-build framework (absent
    in the kernel build) → staging them at build-time BREAKS the build. So the
    GE .cpp are injected at FINALIZE-PREP (AFTER kernel build+verify, BEFORE
    archive promote + BEFORE the GE_OPHOST_RAW_CANN_COPY gate) — they ride into
    the archive without ever being compiled in the kernel build.

    Behavior (port_a3_to_a5 + FA-class only):
      1. Reads the 3 GE .cpp (def/infershape/tiling) + 3 shared .h from the KB
         fa_class templates/op_host/ dir.
      2. Copies them into workspace `op_host/`, PARAMETERIZING the op name —
         the template token `flash_attention_score` is substituted (in filename
         AND file content) with the actual op name so the step generalizes to
         other FA-class ops. Identity substitution for flash_attention_score.
      3. IDEMPOTENT: if the workspace op_host/ already carries a worker-emitted
         GE .cpp (uses `wfh::`/`wp_fa_host::` AND md5 != any CANN-source file of
         the same basename), the worker's own emit is authoritative — do NOT
         overwrite it. The assembler is the FALLBACK when the worker skipped it.
      4. STALE-ARCH35 CLEANUP: removes any leftover CANN arch35 header under the
         workspace op_host/ (e.g. arch35/<op>_tiling_regbase.h) — the archive
         must not ship a CANN arch35 header.

    HARD INVARIANT: the assembled GE .cpp come ONLY from the KB template (uses
    `wfh::`, md5 != CANN). Never from CANN arch35 source. The companion
    GE_OPHOST_RAW_CANN_COPY gate (run AFTER this) verifies that invariant.

    Returns a report dict (best-effort — never raises): {
      "ran": bool,                 # did the assembler engage (scope match)?
      "skipped_reason": str|None,  # why it didn't engage
      "assembled": [str],          # GE files written by the assembler
      "preserved": [str],          # worker-emitted GE files left untouched
      "stale_arch35_removed": [str],
      "errors": [str],
    }
    """
    import re as _re

    report = {
        "ran": False, "skipped_reason": None, "assembled": [],
        "preserved": [], "stale_arch35_removed": [], "errors": [],
    }

    # --- scope: port_a3_to_a5 + FA-class only (mirror the gate's scope) ---
    from plugins import detect_plugin as _detect_plugin
    _active = _detect_plugin(workspace)
    if _active is None or _active.name != "port_a3_to_a5":
        report["skipped_reason"] = "not port_a3_to_a5 mode"
        return report

    op_name = workspace.name
    op_class = ""
    op_cls_path = workspace / "op_classification.json"
    if op_cls_path.is_file():
        try:
            tags = json.loads(op_cls_path.read_text()).get("op_class_tags") or []
            op_class = " ".join(tags) if isinstance(tags, list) else str(tags)
        except Exception:
            op_class = ""
    try:
        from plugins.base import is_attention_named as _is_fa_named
        from plugins.base import is_fa_class as _is_fa_tag
        if not (_is_fa_named(op_name) or _is_fa_tag(op_class)):
            report["skipped_reason"] = "not FA-class"
            return report
    except Exception as e:
        report["skipped_reason"] = f"FA-class predicate failed: {e}"
        return report

    tmpl_dir = _FA_GE_OPHOST_TEMPLATE_DIR
    if not tmpl_dir.is_dir():
        report["skipped_reason"] = f"KB template dir absent: {tmpl_dir}"
        return report

    report["ran"] = True
    op_host_dir = workspace / "op_host"
    op_host_dir.mkdir(parents=True, exist_ok=True)

    tok = _FA_GE_TEMPLATE_OP_TOKEN  # template's op-name token

    def _subst(text: str) -> str:
        # op-name parameterization: identity for flash_attention_score, real
        # substitution for other FA-class ops. Token-bounded so we don't
        # corrupt substrings (e.g. a longer op name containing the token).
        if op_name == tok:
            return text
        return _re.sub(rf"\b{_re.escape(tok)}\b", op_name, text)

    # --- 1+2: assemble the 3 GE .cpp (idempotent per-file) ---
    # Build a CANN md5 index once for the GE basenames (idempotency: a workspace
    # file that is byte-identical to CANN is NOT a legit worker emit).
    ge_basenames = {f"{op_name}{suf}.cpp" for suf in _FA_GE_CPP_SUFFIXES}
    cann_idx = _cann_md5_index_for_basenames(ge_basenames)

    for suf in _FA_GE_CPP_SUFFIXES:
        tmpl_file = tmpl_dir / f"{tok}{suf}.cpp"
        if not tmpl_file.is_file():
            report["errors"].append(f"template missing: {tmpl_file.name}")
            continue
        dst_name = f"{op_name}{suf}.cpp"
        dst = op_host_dir / dst_name

        # Idempotency: a worker-emitted GE .cpp is authoritative iff it uses the
        # shared layer AND is not a CANN byte-copy. tiling.cpp is the one that
        # carries `wfh::` (def/infershape don't); for def/infershape "worker
        # emit" = present + not-CANN-copy.
        if dst.is_file():
            try:
                existing = dst.read_bytes()
            except Exception:
                existing = b""
            is_cann_copy = (
                _md5_bytes(existing) in cann_idx.get(dst_name, set())
                if existing else False
            )
            existing_txt = existing.decode("utf-8", errors="replace")
            uses_shared = bool(_re.search(r"\b(?:wfh|wp_fa_host)::", existing_txt))
            # tiling.cpp: authoritative only if it uses the shared layer and is
            # not a CANN copy. def/infershape: authoritative if not a CANN copy
            # (they legitimately don't reference wfh::).
            if suf == "_tiling":
                worker_authoritative = uses_shared and not is_cann_copy
            else:
                worker_authoritative = (not is_cann_copy) and len(existing) > 0
            if worker_authoritative:
                report["preserved"].append(f"op_host/{dst_name}")
                continue

        try:
            content = _subst(tmpl_file.read_text(encoding="utf-8"))
            dst.write_text(content, encoding="utf-8")
            report["assembled"].append(f"op_host/{dst_name}")
        except Exception as e:
            report["errors"].append(f"assemble {dst_name}: {e}")

    # --- 1b: op-name-specific GE headers (e.g. <op>_tiling_common.h — the
    # CompileInfo POD). KB-authored from installed platform_ascendc API. Idempotency:
    # a present header is authoritative iff NOT a CANN byte-copy; a CANN
    # source-copy/prestage is OVERWRITTEN by the KB version (owner rule 2026-06-13:
    # CANN source-only headers must not be used — re-author from installed API).
    hdr_basenames = {f"{op_name}{suf}.h" for suf in _FA_GE_HDR_SUFFIXES}
    cann_hdr_idx = _cann_md5_index_for_basenames(hdr_basenames)
    for suf in _FA_GE_HDR_SUFFIXES:
        tmpl_file = tmpl_dir / f"{tok}{suf}.h"
        if not tmpl_file.is_file():
            report["errors"].append(f"template missing: {tmpl_file.name}")
            continue
        dst_name = f"{op_name}{suf}.h"
        dst = op_host_dir / dst_name
        if dst.is_file():
            try:
                existing = dst.read_bytes()
            except Exception:
                existing = b""
            is_cann_copy = (
                _md5_bytes(existing) in cann_hdr_idx.get(dst_name, set())
                if existing else False
            )
            if existing and not is_cann_copy:
                report["preserved"].append(f"op_host/{dst_name}")
                continue
        try:
            content = _subst(tmpl_file.read_text(encoding="utf-8"))
            dst.write_text(content, encoding="utf-8")
            report["assembled"].append(f"op_host/{dst_name}")
        except Exception as e:
            report["errors"].append(f"assemble {dst_name}: {e}")

    # --- 2 (cont): shared-layer headers (op-name-agnostic; copy if absent) ---
    for hdr in _FA_GE_SHARED_HEADERS:
        tmpl_hdr = tmpl_dir / hdr
        if not tmpl_hdr.is_file():
            continue
        dst = op_host_dir / hdr
        if dst.is_file():
            continue  # already prestaged / present — leave it
        try:
            dst.write_text(
                _subst(tmpl_hdr.read_text(encoding="utf-8")), encoding="utf-8"
            )
            report["assembled"].append(f"op_host/{hdr}")
        except Exception as e:
            report["errors"].append(f"assemble header {hdr}: {e}")

    # --- 4: stale-arch35 cleanup (archive must not ship a CANN arch35 header) ---
    arch35_dir = op_host_dir / "arch35"
    if arch35_dir.is_dir():
        for stale in sorted(arch35_dir.rglob("*")):
            if stale.is_file():
                try:
                    stale.unlink()
                    report["stale_arch35_removed"].append(
                        str(stale.relative_to(workspace))
                    )
                except Exception as e:
                    report["errors"].append(f"rm stale {stale.name}: {e}")
        # drop the now-empty arch35 dir tree
        try:
            for d in sorted(arch35_dir.rglob("*"), reverse=True):
                if d.is_dir():
                    d.rmdir()
            arch35_dir.rmdir()
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )

    return report
