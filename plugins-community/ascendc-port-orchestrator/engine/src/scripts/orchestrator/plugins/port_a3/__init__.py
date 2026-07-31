# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""``--port-a3`` plugin for AscendC arch22 to arch35 migration.

Reference baseline: A3-CANN execution via aclnn API (NOT CPU PyTorch).
Source: ~/workspace/cann/ops-nn/<category>/<op>/ ops-nn-shaped dir.
Archive layout: mirrors ops-nn shape (op_kernel/arch35/, op_host/,
<op>_apt.cpp, etc.) per ROADMAP §1.5.

Phase 1 of DEBT-094 (2026-05-15): migrated from
- finalize_pipeline._is_port_a3_mode + _check_binary_provenance +
  _check_a5_verify_path_provenance + _finalize_port_a3_to_a5_layout
- scan_delegation_cheating._is_port_a3_workspace + PORT_A3_VERIFY_FILES +
  PORT_A3_VERIFY_FORBIDDEN + CPU_AS_REFERENCE_ANTIPATTERN
"""
from __future__ import annotations
import logging

import json
from pathlib import Path
from typing import Optional

from ..base import BasePlugin, is_attention_named


# ── Module-level constants (kept here, NOT in core) ─────────────────────

VERIFY_FILES: tuple[str, ...] = (
    "run_a5_verify.py",   # A5-side verification entry
    "pass_a_runner.py",   # Pass A canonical verifier
    "run_pass_b.py",      # Pass B canonical verifier
)

# Sanctioned `truth_source` substrings for the verify-path-provenance gate
# (DEBT-162, 2026-06-21). The verify-path gate accepts a PASS only if
# truth_source names a real-NPU verify path; these are the substrings that
# identify one. This list MUST stay a superset of the vocabulary the worker
# brief sanctions (kw_brief.py D.5: "truth_source MUST contain ...") — else a
# worker that follows the brief gets flagged as fraud by the gate (the exact
# brief↔gate drift that false-flagged gather_elements_v2: its
# `a3_capture_via_pybind_aclrtlaunch_kernel` — a real aclrtlaunch NPU launch —
# was rejected because the gate's list lagged the brief). test_real_archive_
# sanity asserts gate ⊇ brief-sanctioned to prevent the drift recurring
# (OL-160 spirit: key off ONE canonical vocabulary, not duplicated literals).
SANCTIONED_TRUTH_SOURCE_SUBSTRINGS: tuple[str, ...] = (
    "aclnn",                                  # aclnn-direct C++ runner / aclnnGather etc.
    "a3_cann",                                # A3-CANN captured reference
    "a3_capture_via_pybind_aclrtlaunch_kernel",  # pybind + ACLRT_LAUNCH_KERNEL real NPU launch (P140)
)

# Patterns BANNED in port_a3-mode verify files: A5-side verification
# must invoke our built kernel via aclnn-direct C++ runner / ctypes shim
# — NOT through PyTorch dispatcher (which silently falls back to AICPU
# when our .so isn't installed). The exact same `F.*` / `torch._foreach_*`
# / `torch_npu.npu_*` calls ARE legitimate in run_a3_reference.py (A3 has
# working CANN install) but FATAL in run_a5_verify.py / pass_a_runner.py.
VERIFY_FORBIDDEN: list[tuple[str, str]] = [
    (r'\btorch\.nn\.functional\.[a-z_]+\s*\(',
        "torch.nn.functional.<op>() in A5 verify path — falls back to "
        "AICPU when our .so unloaded; runs stock PyTorch instead of our kernel"),
    (r'\bF\.[a-z_]+\s*\(',
        "F.<op>() in A5 verify path — same fallback risk; use "
        "aclnn-direct shim instead"),
    (r'\btorch\._foreach_[a-z]+\s*\(',
        "torch._foreach_<op>() in A5 verify path — routes through "
        "PyTorch foreach dispatcher, NOT our kernel"),
    (r'\btorch_npu\.npu_[a-zA-Z_]+\s*\(',
        "torch_npu.npu_<op>() in A5 verify path — may unbind on A5 "
        "(per OL-68); use aclnn-direct ctypes shim"),
    (r'\btorch\.gather\s*\(.*device\s*=\s*[\'"]npu',
        "torch.gather() on NPU in A5 verify path — uses stock PyTorch "
        "gather, NOT our aclnnGatherElementsV2"),
]

# CPU-as-reference anti-pattern: pass_a_runner / run_a5_verify computes
# reference via CPU `tensor.<op>()` and compares against A3 NPU output —
# this NEVER touches A5 hardware. Caught as a separate sub-pattern so
# the violation message is clear.
CPU_AS_REFERENCE_ANTIPATTERN: list[tuple[str, str]] = [
    (
        r'tensor\.(abs|exp|log|sqrt|rsqrt|sigmoid|tanh|gelu|silu|softmax|sum|mean|'
        r'var|max|min|argmax|argmin|topk|sort|cumsum|gather|scatter)\s*\(',
        "CPU tensor.<op>() as 'reference' in verify path — never "
        "exercises A5 NPU let alone our kernel; verify is meaningless"),
]

# Archive layout path mapping (W7, migrated from finalize_pipeline.py).
# Ops-nn mirror layout (per PR4778): workspace/kernel/ → archive/op_kernel/
ARCHIVE_PATH_MAP: tuple[tuple[str, str], ...] = (
    ("kernel/arch35/", "op_kernel/arch35/"),
    ("kernel/", "op_kernel/"),
    ("op_host/", "op_host/"),
)

# Migration-plan filename suffixes that route to archive's docs/ subdir
ARCHIVE_DOCS_SUFFIXES = (
    "_a5_migration_plan.md",
    "_migration_plan.md",
)


class PortA3Plugin(BasePlugin):
    name = "port_a3_to_a5"
    cli_flag = "--port-a3"

    # ── Mode detection ─────────────────────────────────────────────────
    def detect(self, workspace: Path) -> bool:
        """Detect only an explicit, readable migration state.

        Target-shaped files are outputs, not authorization to enter migration
        mode.  Inspecting them during detection would also violate the source-
        only boundary before the mode is established.
        """
        state_file = workspace / ".opgen_state.json"
        if not state_file.is_file():
            return False
        try:
            declared = json.loads(state_file.read_text()).get("opgen_mode")
        except Exception:
            return False
        return declared == self.name

    # ── Scanner contributions ──────────────────────────────────────────
    def kernel_logic_files(self) -> tuple[str, ...]:
        """port_a3 keeps compute in the canonical entry-point:
        `model_new_ascendc.py` subprocess-invokes the aclnn-runner binary
        we generate, and the Python surface is where a torch_npu/aclnn
        fallback would be smuggled in. `model_new_ascendc.py` is the sole
        supported Python compute surface.

        The C++ side is declared separately in `kernel_cpp_dirs()` (this
        hook is the Python-compute surface). Until that hook existed, this
        mode's C++ lived OUTSIDE the scanner's hard-coded `kernel/` walk and
        was entirely UNSCANNED — the DEBT-211 directory-level recurrence of
        OL-160 that made the finalize POST_WORKER_AUDIT gate inert for the
        whole mode; `kernel_cpp_dirs()` closes it.
        """
        return ("model_new_ascendc.py",)

    def kernel_cpp_dirs(self) -> tuple[str, ...]:
        """port_a3 mirrors the ops-nn tree: its AscendC C++ is in `op_host/`
        (GE host: <op>_def.cpp / <op>_tiling.cpp/.h / <op>_infershape.cpp) and
        `op_kernel/` (device kernel incl. `op_kernel/arch35/`, pybind11.cpp).
        There is NO flat `kernel/` dir. Before this declaration the scanner
        walked only `kernel/`, traversed nothing, and reported `violations=0`
        on every port_a3 archive — host-compute delegation shipped unscanned
        (DEBT-211 directory-level; the finalize POST_WORKER_AUDIT gate was
        inert for the whole mode).

        `op_api/` (the op's own generated aclnn C-API definitions —
        `aclnnXxxGetWorkspaceSize` / `aclnnXxx` DEFINITIONS) is NOT declared:
        it is API plumbing, not host-compute delegation, and core skips any
        `op_api/` layer it encounters so those definitions don't mass-flag.
        The top-level `*_runner.cpp` (the aclnn-direct verify runner, which
        legitimately CALLS aclnn) is likewise outside the declared compute
        dirs and so is not scanned for delegation here.
        """
        return ("op_host", "op_kernel")

    def verify_files(self) -> tuple[str, ...]:
        return VERIFY_FILES

    def forbidden_patterns(self) -> list[tuple[str, str]]:
        return VERIFY_FORBIDDEN + CPU_AS_REFERENCE_ANTIPATTERN

    def scanner_category(self) -> str:
        """Back-compat: existing scanner consumers + tests reference
        `port_a3_verify_forbidden` (not the default `port_a3_to_a5_...`
        derived from plugin.name). Keep the historical tag.
        """
        return "port_a3_verify_forbidden"

    def requires_source_architecture_gate(self) -> bool:
        """Migration must preserve the fresh arch22 source architecture class."""
        return True

    # ── pass_b shape contract (P96, 2026-05-15) ─────────────────────────
    def pass_b_required(self) -> bool:
        """port_a3_to_a5 mode: pass_b is DEGENERATE by design — edge_dataset.pt
        IS the truth source, pass_a IS the dispatch test (per ROADMAP §1.5
        Path-B contract). No separate pass_b verifier.
        """
        return False

    def pass_b_default_when_skipped(self) -> dict:
        """Canonical N/A shape for port_a3 mode pass_b. Worker MUST write
        this exact shape; finalize gate rejects deviations.
        """
        return {
            "status": "N/A",
            "reason": (
                "port_a3_to_a5 mode: pass_b is subsumed by pass_a — "
                "edge_dataset.pt['a3_outputs'] IS the truth source per "
                "ROADMAP §1.5 Path-B contract; pass_a IS the A5-vs-A3-"
                "edge_dataset comparison. pass_b would be degenerate."
            ),
            "method": "n/a — port_a3 mode pass_b not applicable",
        }

    def verifier_canonical_filenames(self) -> tuple[str, ...]:
        """port_a3 mode uses pass_a_runner.py + <op>_runner.cpp; no
        run_pass_b.py (the legacy generic template).
        """
        return ("pass_a_runner.py",)

    def forbidden_workspace_files(self) -> tuple[str, ...]:
        """run_pass_b.py is a legacy generic template — migration pass_b
        is degenerate, this file MUST NOT exist at workspace root. Caught
        2026-05-15 gather_elements_v2 kw-2 (P94 cycle gate tripped on
        self-citing run_pass_b.py).
        """
        return ("run_pass_b.py",)

    def docs_source_files(self, workspace):
        """Copy upstream `cann/ops-nn/<category>/<op>/docs/aclnn*.md` into
        archive/docs/. Universal output-format contribution (2026-05-16).
        """
        import json as _json

        state_file = Path(workspace) / ".opgen_state.json"
        if not state_file.is_file():
            return []
        try:
            state = _json.loads(state_file.read_text())
            from source_arch import verify_source_stage

            valid, _reason, _manifest = verify_source_stage(Path(workspace), state)
            if not valid:
                return []
            port_src = state.get("port_a3_source")
        except Exception:
            return []
        if not port_src:
            return []
        src_docs_dir = Path(port_src) / "docs"
        if not src_docs_dir.is_dir():
            return []
        result = []
        for f in src_docs_dir.glob("aclnn*.md"):
            result.append((f.name, f))
        return result

    def check_verifier_uses_modelnew(self, workspace, vj):
        """P96 follow-up (2026-05-15): anti-decorative-bypass.

        Caught when adaptive_avg_pool3d archive shipped with model_new_ascendc.py
        file present (passing OL-160 filename check) but pass_a_runner.py
        directly subprocess-invoked the runner binary, bypassing the
        ModelNew wrapper layer entirely. OL-160 filename rule = 0-USAGE-rate.

        For port_a3, pass_a_runner.py (canonical verifier) MUST contain
        all 3 of (in CODE, not docstring):
        - `model_new_ascendc` substring (import reference)
        - `ModelNew()` substring (instantiation)
        - `<name>(` or `<name>.<attr>(` where `<name>` came from `<name> = ModelNew()`
          (instance is actually called — not just constructed and discarded)

        This is the shared verifier-call contract for migration work.
        """
        import re

        ws = Path(workspace)
        verifier_path = ws / "pass_a_runner.py"
        if not verifier_path.is_file():
            return None  # no canonical verifier — caller's other gates handle
        try:
            text = verifier_path.read_text(errors="ignore")
        except Exception:
            return "check_verifier_uses_modelnew: cannot read pass_a_runner.py"

        # Strip docstrings — markers must be in actual code.
        # Cheap heuristic: skip lines inside triple-quoted strings.
        code_lines = []
        in_docstring = False
        for raw in text.splitlines():
            triple_count = raw.count('"""') + raw.count("'''")
            if triple_count >= 2:
                continue  # single-line docstring
            if triple_count == 1:
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            stripped = raw.strip()
            if stripped.startswith("#"):
                continue
            code_lines.append(raw)
        code_text = "\n".join(code_lines)

        # Marker 1: import reference
        if "model_new_ascendc" not in code_text:
            return (
                "port_a3 decorative-bypass gate (P96 follow-up): "
                "pass_a_runner.py has no `model_new_ascendc` import "
                "reference in CODE — ModelNew wrapper bypassed (OL-160 "
                "USAGE-gap, adaptive_avg_pool3d 2026-05-15 fraud pattern)."
            )

        # Marker 2: instantiation — find `<name> = ModelNew()`
        instantiation_re = re.compile(r"(\w+)\s*=\s*ModelNew\s*\(")
        m = instantiation_re.search(code_text)
        if m is None:
            return (
                "port_a3 decorative-bypass gate (P96 follow-up): "
                "pass_a_runner.py never instantiates ModelNew (no "
                "`<name> = ModelNew()` line in CODE). ModelNew wrapper "
                "bypassed (OL-160 USAGE-gap)."
            )
        instance_name = m.group(1)

        # Marker 3: instance is actually invoked — `<name>(` OR `<name>.<attr>(`
        # in code AFTER the instantiation site
        post = code_text[m.end():]
        call_re = re.compile(rf"\b{re.escape(instance_name)}\s*(?:\(|\.\w+\s*\()")
        if call_re.search(post) is None:
            return (
                f"port_a3 decorative-bypass gate (P96 follow-up): "
                f"pass_a_runner.py instantiates `{instance_name} = ModelNew()` "
                f"but never invokes it (no `{instance_name}(...)` or "
                f"`{instance_name}.method(...)` call after). Decorative "
                f"instantiation = OL-160 USAGE-gap fraud."
            )
        return None

    # ── Finalize gates ─────────────────────────────────────────────────
    def check_binary_provenance(
        self, workspace: Path, vj: dict
    ) -> Optional[str]:
        """Verify the current workspace build lineage without probing CANN.

        A migration run may only attest artifacts it authored and copied back
        into its workspace.  Installed CANN trees and their target binaries are
        deliberately outside this proof model.
        """
        import hashlib

        prec = vj.get("precision", {}) or {}
        status = prec.get("status")
        if status not in ("PASS", "PASS_WITHIN_TOLERANCE"):
            return None
        build_ev = vj.get("build_evidence", {}) or {}
        cp = build_ev.get("compiled_provenance")
        if not isinstance(cp, dict):
            return (
                f"port_a3 precision.status={status} but build_evidence."
                "compiled_provenance is missing — current-build SHA256 lineage "
                "is mandatory; installed CANN artifacts are not admissible."
            )

        root = workspace.resolve()
        failures: list[str] = []

        def _local_file(field: str) -> Optional[Path]:
            raw = cp.get(field)
            if not isinstance(raw, str) or not raw.strip():
                failures.append(f"{field} missing")
                return None
            candidate = Path(raw)
            candidate = candidate if candidate.is_absolute() else root / candidate
            try:
                if candidate.is_symlink():
                    raise ValueError("symlink")
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
                if not resolved.is_file():
                    raise ValueError("not a regular file")
            except Exception as exc:
                failures.append(f"{field} is not a workspace file ({exc})")
                return None
            return resolved

        source = _local_file("source")
        deployed_source = _local_file("deployed_source")
        object_file = _local_file("object")
        shared_lib = _local_file("shared_lib")

        if source and deployed_source and source.name != deployed_source.name:
            failures.append("deployed_source basename differs from source")
        if source and object_file and object_file.name != f"{source.name}.o":
            failures.append("object basename is not <source>.o")
        if shared_lib and ".so" not in shared_lib.name:
            failures.append("shared_lib is not a .so artifact")

        def _sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        digest_fields = {
            "workspace_source_sha256": source,
            "deploy_source_sha256": deployed_source,
            "object_sha256": object_file,
            "shared_lib_sha256": shared_lib,
        }
        actual: dict[str, str] = {}
        for field, path in digest_fields.items():
            declared = cp.get(field)
            if not isinstance(declared, str) or len(declared) != 64:
                failures.append(f"{field} is not a 64-hex SHA256")
                continue
            declared = declared.lower()
            if any(char not in "0123456789abcdef" for char in declared):
                failures.append(f"{field} is not a 64-hex SHA256")
                continue
            if path is not None:
                actual[field] = _sha256(path)
                if actual[field] != declared:
                    failures.append(f"{field} does not match workspace bytes")

        built_from = cp.get("built_from_source_sha256")
        if not isinstance(built_from, str) or len(built_from) != 64 or any(
            char not in "0123456789abcdef" for char in built_from.lower()
        ):
            failures.append("built_from_source_sha256 is not a 64-hex SHA256")
        else:
            source_digest = cp.get("workspace_source_sha256")
            deploy_digest = cp.get("deploy_source_sha256")
            if (
                isinstance(source_digest, str)
                and isinstance(deploy_digest, str)
                and not (
                    source_digest.lower()
                    == deploy_digest.lower()
                    == built_from.lower()
                )
            ):
                failures.append("source/deploy/built-from SHA256 lineage differs")

        if failures:
            return (
                f"port_a3 precision.status={status} but current-build provenance "
                f"FAILED: {failures[:5]}. Installed CANN target artifacts are "
                "outside the allowed evidence boundary."
            )
        return None

    @staticmethod
    def _snake_to_pascal(snake: str) -> str:
        """top_k_top_p_sample → TopKTopPSample"""
        return "".join(p.capitalize() for p in snake.split("_") if p)

    @staticmethod
    def _derive_op_name(workspace: Path, vj: dict) -> Optional[str]:
        """Best-effort op name lookup: verification.json.op,
        .opgen_state.json.op, or workspace dir name."""
        op = vj.get("op")
        if op:
            return op
        state_path = workspace / ".opgen_state.json"
        if state_path.is_file():
            try:
                op = json.loads(state_path.read_text()).get("op")
                if op:
                    return op
            except Exception as error:
                logging.getLogger(__name__).debug(
                    "Recoverable operation failed.", exc_info=error
                )
        # Last resort: workspace dir name (works for both
        # workspace/<op>/ and output/<project>/src/kernels/<op>/)
        return workspace.name if workspace.is_dir() else None

    # ── Archive layout (migrated from finalize_pipeline.py phase 2) ────
    def archive_project_subdir(self) -> Optional[str]:
        return "a3_to_a5_port"

    def archive_layout_mapping(self, workspace: Path) -> dict[str, str]:
        """Returns path-prefix mapping. Caller uses this dict OR
        calls resolve_archive_target() per-file. Both APIs supported.
        """
        return dict(ARCHIVE_PATH_MAP)

    def resolve_archive_target(
        self, workspace_rel_path: str, op: str
    ) -> str:
        """W7: map workspace-relative path → archive-relative target."""
        # Migration-plan filenames → docs/
        for suffix in ARCHIVE_DOCS_SUFFIXES:
            if workspace_rel_path.endswith(suffix):
                return f"docs/{workspace_rel_path}"
        # Subdir prefix remaps (order matters — longer first)
        for src_prefix, dst_prefix in ARCHIVE_PATH_MAP:
            if workspace_rel_path.startswith(src_prefix):
                return dst_prefix + workspace_rel_path[len(src_prefix):]
        # Default: top-level archive
        return workspace_rel_path

    def check_verify_path_provenance(
        self, workspace: Path, vj: dict
    ) -> Optional[str]:
        """DEBT-NEW (2026-05-14) port_a3-specific verify-path provenance.

        Rejects PASS when workspace can't prove the A5-side verify
        invoked OUR built kernel via aclnn-direct C++ runner / ctypes
        shim (vs PyTorch dispatcher falling back to AICPU).

        Two checks AND'd:
        1. verification.json.truth_source contains 'aclnn' or 'a3_cann'
        2. Workspace has at least one *_runner.cpp / *_shim.cpp /
           aclnn_*.cpp artifact

        """
        prec = vj.get("precision", {})
        status = prec.get("status")
        if status not in ("PASS", "PASS_WITHIN_TOLERANCE"):
            return None

        truth_source = (vj.get("truth_source") or "").lower()
        if not truth_source:
            return (
                "DEBT-NEW a5_verify_path_fraud: precision.status=PASS but "
                "verification.json.truth_source is missing/blank — "
                "port_a3_to_a5 mode requires `truth_source` to identify "
                "the verify path (e.g. 'a3_cann' or "
                "'a3_cann_via_v1_aclnn_direct'). Set truth_source AFTER "
                "verifying via aclnn-direct C++ runner / ctypes shim "
                "(NOT PyTorch dispatcher path)."
            )
        if not any(tok in truth_source for tok in SANCTIONED_TRUTH_SOURCE_SUBSTRINGS):
            return (
                f"DEBT-NEW a5_verify_path_fraud: precision.status=PASS "
                f"but truth_source={truth_source!r} doesn't reference a "
                f"real-NPU verify path. Required substring (one of): "
                f"{', '.join(SANCTIONED_TRUTH_SOURCE_SUBSTRINGS)}. "
                f"Suspect PyTorch dispatcher fallback to AICPU/CPU."
            )

        has_runner = False
        if workspace.is_dir():
            for p in workspace.iterdir():
                if not p.is_file():
                    continue
                name = p.name
                if name.endswith("_runner.cpp"):
                    has_runner = True
                    break
                if name.endswith("_shim.cpp"):
                    has_runner = True
                    break
                if name.startswith("aclnn_") and name.endswith(".cpp"):
                    has_runner = True
                    break
        if not has_runner:
            return (
                "DEBT-NEW a5_verify_path_fraud: precision.status=PASS + "
                f"truth_source={truth_source!r} claims aclnn-direct "
                "verify, but workspace has no `*_runner.cpp` / "
                "`aclnn_*_shim.cpp` / `*_shim.cpp` artifact to prove it."
            )
        return None

    def canonical_pass_a_skip_reason(self, workspace: Path) -> Optional[str]:
        """port_a3's native two-tier verdict comes from the worker's pass_a_runner.py
        (per-op ModelNew/Model live run → precision_eval_port_a3_two_tier.classify_port_a3_case:
        ours on NPU, a3 from edge_dataset, cpu_truth on CPU), NOT the generic canonical
        file-evaluator (which live-recomputes via a generic `model(*group)` call that
        port_a3's per-op call signatures — e.g. FA `m(q,k,v,*args)` — break). Two-tier
        ENGAGEMENT is gated post-runner in ssh_runner._gate_port_a3_two_tier. (task#82;
        DEBT-161 plugin-method extraction of the former phase_o5 `plugin.name ==` branch.)
        """
        return (
            "canonical pass_a skipped: port_a3_to_a5 native two-tier comes "
            "from worker pass_a_runner.py (per-op ModelNew/Model live run); "
            "two-tier engagement gated post-runner"
        )

    def truth_source(self) -> Optional[str]:
        """port_a3 truth = upstream Ascend910 (A3/V220) CANN aclnn binary,
        recorded in edge_dataset (worker pass_a_runner.py two-tier compares
        ours-on-NPU vs a3). Plugin-method dispatch (DEBT-161 paradigm) —
        burns down the phase_o5 `opgen_mode == "port_a3_to_a5"` literal-branch.
        """
        return "a3_cann"

    # FA paradigm = template-assembly (owner 2026-06-07): FA-class L4 routes to
    # the standard kw worker, which derives arch35 output from the admitted
    # arch22 source and public KB templates.

    # A bounded research-recovery pass is useful when the standard arch22 ->
    # arch35 migration researcher cannot close a gap. Its sealed-input scanners
    # and compile/copy-shape/KB-overlap checks remain mandatory; recovered
    # material is advisory and never replaces fresh source-NPU truth.
    def should_auto_cann_learn_on_gap(
        self, op_class: str, op_complexity: str, worker_signal: str,
        *, workspace=None,
    ) -> bool:
        return True


_INSTANCE = PortA3Plugin()


def get_plugin_instance():
    return _INSTANCE


# Self-register at import time
from .. import register_plugin  # noqa: E402
register_plugin(_INSTANCE)
