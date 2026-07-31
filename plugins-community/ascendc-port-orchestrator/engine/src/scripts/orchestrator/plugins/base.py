# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""OpGen mode plugin contract — the single source of truth for what
a mode plugin must provide.

Per `docs/design/PLUGIN_ARCHITECTURE_DESIGN.md`. Phase 1 of DEBT-094.

Universal invariants are NOT in this protocol — they're core-enforced
because they apply identically to every mode. See §3 of the design doc.

DEBT-201 god-class decompose (2026-07): the ~1400-line god-module was split
into cohesive leaves while keeping this module's public surface identical:
  - `taxonomy.py`  — the `is_*` op-class / op-name predicates (pure leaf).
  - `protocol.py`  — the `@runtime_checkable` `PluginProtocol` contract.
  - `base.py` (this file) — the `BasePlugin` neutral-default implementation.
The four predicates and `PluginProtocol` are re-exported below so every
historical import path (`from plugins.base import PluginProtocol`,
`from plugins.base import is_fa_class`, `from plugins import BasePlugin`,
etc.) keeps working byte-for-byte, including runtime `isinstance`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# ── Re-exports (backward-compat: preserve `plugins.base.<symbol>`) ──────
from .taxonomy import (  # noqa: F401
    is_fa_class,
    is_attention_named,
    is_l4_fused,
    is_backward_class,
)
from .protocol import PluginProtocol  # noqa: F401

__all__ = [
    "is_fa_class",
    "is_attention_named",
    "is_l4_fused",
    "is_backward_class",
    "PluginProtocol",
    "BasePlugin",
]


class BasePlugin:
    """Optional helper base providing neutral defaults for every protocol
    method. Concrete plugins MAY subclass this to override only what
    varies — but subclassing is NOT required (a plain class implementing
    PluginProtocol works equally).

    Subclasses must set `name` (and optionally `cli_flag`) at class level.
    """
    name: str = ""
    cli_flag: Optional[str] = None

    def detect(self, workspace: Path) -> bool:
        return False  # plugin must override; default = "never matches"

    def kernel_logic_files(self) -> tuple[str, ...]:
        """NOT a neutral default — the empty tuple is a FAIL-LOUD default.

        Every other hook here returns "no opinion" and the caller carries
        on. This one cannot: a mode with no declared kernel-logic file is
        precisely the OL-160 silent-`violations=0` hole (DEBT-211), so
        `scan_delegation_cheating` treats empty as a hard failure rather
        than as "nothing to scan". A new plugin therefore starts
        UNSCANNABLE and must declare where its compute lives before it
        can pass the delegation scan. That is the intended pressure —
        do not "fix" a hard failure by deleting the check.
        """
        return ()

    def kernel_cpp_dirs(self) -> tuple[str, ...]:
        """Workspace-relative directories where THIS mode's AscendC C++
        compute lives (kernel + host). `scan_delegation_cheating` walks
        each declared dir recursively for `.cpp/.cc/.cxx/.h/.hpp` and
        applies the universal CPP delegation patterns.

        NOT a neutral default — the empty tuple is a FAIL-LOUD default,
        the directory-level twin of `kernel_logic_files()` (DEBT-211 /
        OL-160). The original OL-160 incident was NAME-level (the scanner
        read a hard-coded *file* name that held no logic and reported
        `violations=0`); its recurrence is DIRECTORY-level: the scanner
        walked a hard-coded `kernel/` dir, but `port_a3_to_a5` puts its
        AscendC C++ in `op_host/` + `op_kernel/` and has no `kernel/`, so
        the walk traversed nothing and the whole mode's host-compute
        delegation shipped unscanned.

        For every supported mode, an empty declaration OR a declaration whose
        dirs are all absent is a COVERAGE-GAP VIOLATION, never a silent
        0-violation pass. A new AscendC plugin therefore starts UNSCANNABLE
        on the C++ side and must declare where its kernel/host C++ lives.

        Examples:
        - backward:             ("kernel",)
        - port_a3_to_a5:        ("op_host", "op_kernel")
        """
        return ()

    def verify_files(self) -> tuple[str, ...]:
        return ()

    def forbidden_patterns(self) -> list[tuple[str, str]]:
        return []

    def scanner_category(self) -> str:
        return f"{self.name}_verify_forbidden"

    def kb_subdirs(self) -> list[str]:
        # Phase 1: legacy flat layout. Phase 2 will populate per-plugin
        # lists once KB physical reorg lands.
        return ["."]

    def extra_finalize_checks(self):
        return []

    def requires_source_architecture_gate(self) -> bool:
        """Whether finalize must compare generated architecture to arch22 source."""
        return False

    def check_binary_provenance(
        self, workspace: Path, vj: dict
    ) -> Optional[str]:
        return None

    def check_verify_path_provenance(
        self, workspace: Path, vj: dict
    ) -> Optional[str]:
        return None

    def archive_layout_mapping(self, workspace: Path) -> dict[str, str]:
        return {}

    def archive_project_subdir(self) -> Optional[str]:
        return None

    def resolve_archive_target(
        self, workspace_rel_path: str, op: str
    ) -> str:
        return workspace_rel_path

    def check_op_host_completeness(self, workspace: Path) -> Optional[str]:
        """Require a complete independently authored AscendC host package."""
        op_host = workspace / "op_host"
        mode_hint = getattr(self, "name", None) or "ascendc"
        if not op_host.is_dir():
            return (
                f"PB-33 op_host completeness ({mode_hint} mode): "
                f"workspace/op_host/ directory is missing entirely. "
                "PR4778 spec requires complete op_host/ mirror with "
                "<op>_def.cpp + <op>_tiling.cpp + <op>_tiling.h + CMakeLists.txt. "
                "Patches are review-aid only — ship complete files. See "
                "briefs/_common.py fixed_layout_block() + kw_brief Phase B.4."
            )
        # Count non-patch, non-config files
        real_files = []
        for f in op_host.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(op_host)
            if rel.parts and rel.parts[0] == "config":
                continue
            if f.suffix == ".patch":
                continue
            real_files.append(rel)
        if len(real_files) < 3:
            return (
                f"PB-33 op_host completeness ({mode_hint} mode): "
                f"only {len(real_files)} non-config / non-patch file(s) in "
                f"workspace/op_host/ — minimum 3 required (<op>_def.cpp + "
                f"<op>_tiling.cpp + <op>_tiling.h). Found: "
                f"{sorted(str(p) for p in real_files)}. PR4778 spec requires "
                f"complete op_host/ package; patches are review-aid only. "
                f"See briefs/_common.py fixed_layout_block()."
            )
        return None

    def kw_brief_phase_a(self) -> Optional[str]:
        return None

    def kw_brief_phase_d(self) -> Optional[str]:
        return None

    def canonical_pass_a_skip_reason(self, workspace: Path) -> Optional[str]:
        # DEBT-161: default = use the generic canonical pass_a evaluator (port_a3
        # overrides to skip it — its two-tier verdict comes from the worker runner).
        return None

    def truth_source(self) -> Optional[str]:
        # DEBT-161 paradigm: plugin declares its O5 truth source. Default None =
        # phase_o5.expected_truth_source falls back to the scoped
        # port_a3/backward pre-dispatch behavior.
        return None

    # ── P96 pass_b shape contract defaults ──────────────────────────────
    def pass_b_required(self) -> bool:
        return True  # conservative default; migration overrides to False

    def pass_b_default_when_skipped(self) -> dict:
        return {}  # only consulted when pass_b_required=False

    def verifier_canonical_filenames(self) -> tuple[str, ...]:
        return ()

    def forbidden_workspace_files(self) -> tuple[str, ...]:
        return ()

    def check_verifier_uses_modelnew(
        self, workspace: Path, vj: dict
    ) -> Optional[str]:
        return None  # neutral default — no enforcement

    # ── Docs subdir contribution (2026-05-16, universal output format) ──
    def docs_source_files(self, workspace: Path) -> list[tuple[str, Path]]:
        """Files to copy into archive/docs/. Returns list of (dest_name, src_path).

        Universal output format: every archive has a `docs/` subdir with at
        least one Chinese-language API doc. Plugin decides source:
        - port_a3: copy upstream `docs/aclnn*.md` from cann/ops-nn/
        - backward: generate from the forward specification when available

        Default (BasePlugin): empty list. Plugins SHOULD override to
        supply at least one doc file.
        """
        return []

    def kw_brief_phase_block(
        self, *, op: str = "", workspace=None,
        iter_cap_remaining: int = 0,
        directive_text: Optional[str] = None,
        handoff_from_prior_agent: Optional[str] = None,
        env=None,
    ) -> Optional[str]:
        return None

    def should_auto_cann_learn_on_gap(
        self, op_class: str, op_complexity: str, worker_signal: str,
        *, workspace=None,
    ) -> bool:
        """Neutral default: unsupported workflows never enter research recovery."""
        return False

    def ko_escalation_threshold(self, op_class: str = "unknown") -> float:
        return 1.0  # paradigm-uniform AscendC default (parity target; owner-directed 2026-07-21, was 0.6)

    # ────────────────────── Phase O5 orchestrator-driven perf re-verify ──────────────────────
    # DEBT-097 wire-in (2026-05-27, owner direction):
    # `phase_o5_perf_capture.measure_op_perf` was shipped 2026-05-18 (`07a770ed`)
    # but zero callers existed in orchestrator.py / finalize_pipeline.py for 9 days
    # — independent perf re-verify was effectively absent for ALL modes. Owner
    # Independent performance re-verification is wired into orchestrator
    # finalize state for supported modes.
    #
    # Default True — every paradigm participates unless it has structural reason
    # to opt out. The `measure_op_perf` module itself returns NOT_VERIFIED_SAME_METHOD
    # for unwired dispatch paths (for example the port_a3 SSH stub), so opt-in here
    # only controls whether the orchestrator INVOKES the module, not whether the
    # module produces a result. Opting out (return False) skips the call entirely
    # and leaves verification.json.performance as worker-authored.

    def should_run_phase_o5_perf_capture(self) -> bool:
        """Whether the orchestrator should run `phase_o5_perf_capture.measure_op_perf`
        in its finalize state. Default True (all paradigms participate).

        Plugins return False ONLY when there is a structural reason the
        symmetric profiler-CSV harness cannot meaningfully measure this paradigm
        (e.g., paradigm has no NPU-side runnable, paradigm uses an oracle
        already known to dominate orchestrator-side re-measurement).
        """
        return True

    # ────────────────────── Stream-silence timeout (per-paradigm) ──────────────────────
