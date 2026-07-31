# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""PluginProtocol — the OpGen mode-plugin contract surface (extracted from
plugins/base.py, DEBT-201 god-class decompose).

Per `docs/design/PLUGIN_ARCHITECTURE_DESIGN.md`. Phase 1 of DEBT-094.

Universal invariants are NOT in this protocol — they're core-enforced
because they apply identically to every mode. See §3 of the design doc.

`plugins.base` re-exports `PluginProtocol` so every historical import path
(`from plugins.base import PluginProtocol`, `from plugins import
PluginProtocol`) keeps working byte-for-byte, including runtime-isinstance
(the class stays `@runtime_checkable`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class PluginProtocol(Protocol):
    """Every op-gen mode plugin MUST satisfy this contract.

    Mode-specific concerns are listed here as method signatures. A plugin
    that doesn't need to vary on a concern returns the neutral default
    (empty tuple / empty dict / None) — the contract is uniform; the
    behavior space includes "no override".
    """

    # ── Identity ────────────────────────────────────────────────────────
    name: str
    """Canonical mode key. MUST match the .opgen_state.json.opgen_mode
    enum value used at workspace creation."""

    cli_flag: Optional[str]
    """CLI argument that activates this mode (``--port-a3`` or
    ``--backward``)."""

    # ── Mode detection ──────────────────────────────────────────────────
    def detect(self, workspace: Path) -> bool:
        """Does `workspace` belong to this plugin?

        Implementation contract:
        1. If `.opgen_state.json` exists with an `opgen_mode` field, that
           value is authoritative — return `opgen_mode == self.name`.
        2. Otherwise, layout heuristic (verify-file presence, ops-nn
           upstream layout, etc.) may infer the mode.

        Mutual exclusivity: at most one plugin's detect() must return
        True for any given workspace. The registry asserts this on
        every call to `detect_plugin()`.
        """
        ...

    # ── Scanner contributions ───────────────────────────────────────────
    def kernel_logic_files(self) -> tuple[str, ...]:
        """Every workspace-root file name that can carry THIS mode's real
        kernel logic. `scan_delegation_cheating` reads each one that
        exists and applies the UNIVERSAL delegation patterns (torch_npu.*
        / aclnn* / tensor-method compute) to it.

        This is the mode's answer to "where did you put the compute?" —
        NOT "which files must exist" (that is
        `finalize_checks_structural._check_universal_entrypoints`) and NOT
        "which files are allowed to exist" (that is
        `forbidden_workspace_paths()`).

        MANDATORY. Returning empty is a HARD SCANNER FAILURE, never a
        silent pass — see DEBT-211. The 2026-05-14 name-coupling incident
        (`docs/postmortem/SAFETY_NET_NAME_COUPLING_2026_05_14.md`) and its
        route-a recurrence both had the same shape: the scanner looked at
        a hard-coded name, that name held no logic for the mode in
        question, and the scan silently reported `violations=0`. A mode
        that cannot say where its compute lives MUST NOT be scannable-by
        -accident; it must be un-shippable until it declares.

        Contract notes:
        - Coverage, not policy. Include every file in which the scoped mode
          can place real compute; a scan that omits a compute-bearing file is
          invalid even when another gate rejects that file later.
        - At least one declared name must EXIST in the workspace, or the
          scan is vacuous and core raises the same hard failure. A scan
          that read nothing is an error, not a pass.

        Both supported modes declare ``model_new_ascendc.py`` as their
        canonical Python compute entry.
        """
        ...

    def kernel_cpp_dirs(self) -> tuple[str, ...]:
        """Workspace-relative directories where THIS mode's AscendC C++
        compute lives. `scan_delegation_cheating` walks each declared dir
        recursively for `.cpp/.cc/.cxx/.h/.hpp` and applies core's universal
        CPP delegation patterns.

        This is the DIRECTORY-level twin of `kernel_logic_files()`. OL-160
        recurred at the directory level (DEBT-211 PR note): the scanner
        walked a hard-coded `kernel/` dir, but `port_a3_to_a5` puts its
        C++ in `op_host/` + `op_kernel/` and has no `kernel/`, so the walk
        traversed nothing and the mode's whole host-compute delegation
        shipped `violations=0`. Declaring the dirs closes that hole.

        MANDATORY: empty, or all-declared-dirs-absent, is a HARD SCANNER
        FAILURE (coverage gap), never a silent pass.

        Examples:
        - backward:             ("kernel",)
        - port_a3_to_a5:        ("op_host", "op_kernel")
        """
        ...

    def verify_files(self) -> tuple[str, ...]:
        """File names this plugin treats as verify-path entry points
        (scanned for `forbidden_patterns`). Empty tuple = no mode-
        specific verify files (universal patterns still apply via core).

        NOTE — distinct from `kernel_logic_files()`: these are the files
        that CHECK the kernel, scanned with this plugin's OWN
        `forbidden_patterns()`. `kernel_logic_files()` are the files that
        ARE the kernel, scanned with core's universal patterns. Do not
        merge the two: port_a3's verify patterns (e.g. the
        CPU-as-reference antipattern) are correct on a verifier and
        nonsense on a kernel source.
        """
        ...

    def forbidden_patterns(self) -> list[tuple[str, str]]:
        """Mode-specific forbidden regex patterns. Applied ONLY to files
        named in `verify_files()`. Format: list of (regex_pattern, description).

        Universal patterns (torch_npu.<api>, aclnn*) live in core
        scanner — they apply to all modes equally and MUST NOT be
        duplicated here.
        """
        ...

    def scanner_category(self) -> str:
        """Tag used in scanner violation dicts when reporting hits from
        this plugin's `forbidden_patterns()`. Downstream consumers
        (REPORT generator, hooks) read this to scope violation context.
        Default '<name>_verify_forbidden' is fine for most modes.
        """
        ...

    def kb_subdirs(self) -> list[str]:
        """KB subdirectory paths (relative to src/skills/references/)
        that this plugin's worker briefs should load.

        Phase 1 (today): all plugins return ["."] (legacy flat layout)
        — additive hook with no functional change. A future KB reorganization
        may populate scoped lists such as:
        - shared/                       (target-agnostic + plugin-agnostic)
        - target/ascendc/               (AscendC-specific reference)
        - plugin-scope/<plugin_name>/   (per-plugin learned lessons)

        Possible per-plugin returns:
        - port_a3: ["shared/", "target/ascendc/", "plugin-scope/port_a3/"]
        - backward: ["shared/", "target/ascendc/", "plugin-scope/backward/"]

        See docs/design/KB_DESIGN_NOTES.md#kb-reorganization-design.
        """
        ...

    def extra_finalize_checks(
        self,
    ) -> list[tuple[str, "Callable[[Path, dict], Optional[str]]"]]:
        """Plugin-defined finalize-gate hooks beyond the standard ones
        (binary_provenance, verify_path_provenance, etc.). Returns
        list of (gate_name, callable). Each callable takes (workspace,
        verification_json_dict) and returns None on accept or an
        error string on reject — same shape as the standard hooks.

        Default (BasePlugin) returns []. Plugins MAY override to add
        new finalize gate types WITHOUT changing the protocol or
        finalize_pipeline dispatch site. This is the option-A
        extensibility path: new check types live entirely inside a
        plugin's own file.

        Caller (finalize_pipeline.check_finalize_eligibility) iterates
        plugin.extra_finalize_checks() AFTER the standard hardcoded
        checks; any rejection aborts with the gate name from the
        returned tuple.
        """
        ...

    def requires_source_architecture_gate(self) -> bool:
        """Return true only when fresh arch22 source is an authoritative input."""
        ...

    # ── Finalize gates ──────────────────────────────────────────────────
    def check_binary_provenance(
        self, workspace: Path, vj: dict
    ) -> Optional[str]:
        """Return error message if binary-provenance proof model is
        required for this mode but missing/invalid. Return None if the
        check passes OR this mode doesn't require this proof model.

        Examples:
        - port_a3: generated-source to built-artifact lineage
        - backward: generated-kernel provenance match
        """
        ...

    def check_verify_path_provenance(
        self, workspace: Path, vj: dict
    ) -> Optional[str]:
        """Mode-specific verify-path provenance check.

        Examples:
        - port_a3: truth_source must contain 'aclnn' + *_runner.cpp present
        - backward: plugin-defined reference provenance
        """
        ...

    # ── Archive layout ──────────────────────────────────────────────────
    def archive_layout_mapping(
        self, workspace: Path
    ) -> dict[str, str]:
        """Map workspace-relative path → archive-relative path.

        Empty dict (default) means flat copy: every workspace file goes
        to the same path under the archive root.

        port_a3 returns a non-empty mapping to place op_kernel/ under
        archive's op_kernel/arch35/, op_host/ under op_host/, etc.,
        mirroring the ops-nn directory shape.
        """
        ...

    def archive_project_subdir(self) -> Optional[str]:
        """Default project subdirectory under `output/` for this mode's
        archived ops (for example, migration → 'a3_to_a5_port' and
        backward → 'backward_ops'). None means caller must provide archive_root
        explicitly (no mode-specific default).
        """
        ...

    def resolve_archive_target(
        self, workspace_rel_path: str, op: str
    ) -> str:
        """Translate a single workspace-relative path to its archive-
        relative target. Default (BasePlugin) is identity (flat copy).

        port_a3 overrides this to remap kernel/ → op_kernel/ and route
        *_migration_plan.md → docs/. Used by the finalize archive
        promotion loop.
        """
        ...

    def check_op_host_completeness(self, workspace: Path) -> Optional[str]:
        """PB-33 op_host completeness gate (plugin-owned, DEBT-094 phase 3).

        Returns error string when the gate fails for this plugin's mode;
        None when it passes (or doesn't apply to this paradigm).

        - BasePlugin (AscendC default): require
          workspace/op_host/ with ≥3 non-config, non-patch files
          (<op>_def.cpp + <op>_tiling.cpp + <op>_tiling.h minimum).

        Previously a boolean-flag pattern (`op_host_required()` +
        `_check_op_host_completeness` in pipeline); flipped to full
        delegate so finalize_pipeline.py has no plugin if/else.
        """
        ...

    # ── Brief overrides (Phase 2 wiring) ────────────────────────────────
    def kw_brief_phase_a(self) -> Optional[str]:
        """Mode-specific Phase A (Analysis) text to inject into worker
        brief. None = core default applies. (Wired in Phase 2 of DEBT-094.)
        """
        ...

    def kw_brief_phase_d(self) -> Optional[str]:
        """Mode-specific Phase D (Verification) text. None = core default.
        (Wired in Phase 2.)
        """
        ...

    def canonical_pass_a_skip_reason(self, workspace: Path) -> Optional[str]:
        """DEBT-161 (plugin-method extraction of a phase_o5 `plugin.name ==` branch).
        Return a non-None reason to SKIP the generic canonical pass_a file-evaluator for
        this mode — i.e. the mode's native pass_a verdict is produced elsewhere (e.g. the
        worker runner) and the generic `model(*group)` evaluator does not apply. None
        (default, in BasePlugin) = the mode uses the generic canonical evaluator.
        """
        ...

    # ── pass_b shape contract (P96, 2026-05-15) ─────────────────────────
    def pass_b_required(self) -> bool:
        """Does this mode require a non-degenerate pass_b verifier?

        - backward mode: plugin-defined according to its generated reference.
        - port_a3_to_a5 mode: False — edge_dataset.pt IS the truth source,
          pass_a IS the dispatch test; pass_b is degenerate by design.

        Used by:
        - finalize_pipeline._check_port_a3_pass_b_schema (rejects
          run_pass_b.py at workspace root when pass_b_required=False)
        - kw_brief phase-E checklist (only mandates pass_b.status fields
          when pass_b_required=True)

        Default (BasePlugin): True.
        """
        ...

    def pass_b_default_when_skipped(self) -> dict:
        """When pass_b_required() returns False, what canonical N/A shape
        should verification.json.precision.pass_b have?

        Used by finalize gate to assert worker wrote the correct shape
        (caught 2026-05-15 gather_elements_v2 kw-2 writing PASS instead
        of N/A — mode-schema drift).

        Default (BasePlugin): empty dict (BasePlugin pass_b_required=True
        so this isn't consulted). Plugins where pass_b_required=False
        should return:
            {
                "status": "N/A",
                "reason": "<mode-specific canonical reason>",
                "method": "<mode-specific n/a method>"
            }
        """
        ...

    def verifier_canonical_filenames(self) -> tuple[str, ...]:
        """File names this plugin uses for verifier scripts at workspace
        root. Used by:
        - phase_o5_runner._verify_runner_independence to know which files
          to scan for verification.json self-citation (cycle gate)
        - finalize_pipeline._check_port_a3_pass_b_schema to know which
          files MUST NOT exist (e.g. run_pass_b.py for port_a3 mode)

        Default (BasePlugin): empty (no canonical verifier files —
        plugin relies on core-default detection).

        Examples:
        - backward: plugin-declared verifier files
        - port_a3_to_a5: ("pass_a_runner.py", "<op>_runner.cpp")
          (NO run_pass_b.py — port_a3 mode pass_b is degenerate)
        """
        ...

    def forbidden_workspace_files(self) -> tuple[str, ...]:
        """File names this plugin's workspaces MUST NOT contain at root.
        Used by finalize gate to reject mode-schema drift.

        Default (BasePlugin): empty (no forbidden files).

        Examples:
        - port_a3_to_a5: ("run_pass_b.py",)  # benchmark-style verifier doesn't belong
        - backward: plugin-defined
        """
        ...

    # ── Anti-decorative-bypass check (P96 follow-up, 2026-05-15) ────────
    def check_verifier_uses_modelnew(
        self, workspace: Path, vj: dict
    ) -> Optional[str]:
        """Mode-specific anti-decorative-bypass check.

        Goal: ensure that the verifier scripts named by `verifier_canonical_filenames()`
        ACTUALLY import + instantiate + call `ModelNew` from the workspace's
        primary kernel module — not just have `model_new_*.py` files present
        as decoration with no runtime invocation (the 2026-05-15 OL-160-gap
        fraud pattern caught in an adaptive_avg_pool3d archive).

        Return error message if check fails, None if it passes OR this plugin
        doesn't enforce the check.

        What "uses ModelNew" means:
        - verifier script imports the canonical ``model_new_ascendc`` module
        - verifier script instantiates `ModelNew()` (object construction)
        - verifier script CALLS the instance (`model_new(...)` somewhere in
          a code line, not just a comment / docstring)

        Default (BasePlugin): None (no enforcement). Plugins that ship
        canonical verifier files should override to do a strict static-text
        check (grep level is acceptable; AST level is stronger but optional).

        Examples:
        - port_a3_to_a5: scan pass_a_runner.py / pass_a_*.py for
          `from model_new_ascendc import` + `ModelNew()` + `model_new(`
        - backward: enforce the verifier files declared by the backward plugin

        Used by:
        - finalize_pipeline gate `GateID.VERIFIER_USES_MODELNEW` (wired on
          main commit 1e89adfe) — rejects archives where canonical verifier
          files exist but don't actually invoke ModelNew (decorative bypass
          fraud pattern caught 2026-05-15).

        Implementation hint (regex approach):
            verifier_path = workspace / "verify.py"
            text = verifier_path.read_text()
            if "ModelNew" not in text:
                return "verify.py does not reference ModelNew (decorative bypass?)"
            if "ModelNew()" not in text and "ModelNew(" not in text:
                return "verify.py references ModelNew but never instantiates it"
            # Could also check that an instance is called: `model_new(`
            # appears as a call, not just an assignment.
            return None
        """
        ...

    # ── Docs subdir contribution (universal output format) ──────────────
    def docs_source_files(self, workspace: Path) -> list:
        """Files to copy into archive/docs/. Returns list of (dest_name, src_path).

        Universal: every archive has a `docs/` subdir with Chinese-language
        API doc(s). Plugin decides where to source it:
        - port_a3: from upstream `cann/ops-nn/<...>/<op>/docs/aclnn*.md`
        - backward: generate from the forward specification and autograd contract

        Default (BasePlugin): empty list.
        """
        ...

    def kw_brief_phase_block(
        self, *, op: str = "", workspace=None,
        iter_cap_remaining: int = 0,
        directive_text: Optional[str] = None,
        handoff_from_prior_agent: Optional[str] = None,
        env=None,
    ) -> Optional[str]:
        """Mode-specific full phase block for the kernel-worker brief."""
        ...

    def should_auto_cann_learn_on_gap(
        self, op_class: str, op_complexity: str, worker_signal: str,
        *, workspace=None,
    ) -> bool:
        """Whether a research gap may enter the bounded prior-art learner.

        The neutral implementation is False; arch22 -> arch35 migration opts
        in while backward generation remains out.
        """
        ...

    def ko_escalation_threshold(self, op_class: str = "unknown") -> float:
        """Perf ratio below which the AscendC workflow routes kw to ko."""
        ...
