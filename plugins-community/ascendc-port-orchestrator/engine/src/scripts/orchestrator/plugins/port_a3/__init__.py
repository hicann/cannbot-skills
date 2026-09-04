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

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional

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


TILELANG2ASCENDC_SOURCE_KIND = "port-aclnn-tilelang2ascendc"

# A strict-profile delivery is a source product, not a build directory.  Keep
# this list conservative and explicit: it is used only to identify the
# candidate kernel sources that must be regular files, never to infer source
# semantics or an ABI from their contents.
DIRECT_LAUNCH_CPP_SOURCE_SUFFIXES = frozenset({
    ".asc",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inc",
    ".tpp",
})

# A header-only wrapper is not a kernel delivery.  Keep this separate from
# the broader source suffix set used while walking a candidate so that a
# header cannot satisfy the implementation-unit completeness gate.
DIRECT_LAUNCH_IMPLEMENTATION_SUFFIXES = frozenset({
    ".asc",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
})

# Build products in the direct profile are source-delivery artifacts.  CMake
# is the required primary recipe, but a reproducible kernel can legitimately
# carry an auxiliary Makefile, CMake include, generated-tiling descriptor, or
# a small Python/build helper.  These are deliberately listed here instead of
# relying on a blanket "copy everything under kernel/" rule.
DIRECT_LAUNCH_BUILD_FILENAMES = frozenset({
    "CMakeLists.txt",
    "Makefile",
    "GNUmakefile",
})
DIRECT_LAUNCH_KERNEL_AUXILIARY_SUFFIXES = frozenset({
    ".cmake",
    ".in",
    ".json",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
})
DIRECT_LAUNCH_ROOT_AUXILIARY_FILENAMES = frozenset({
    "LICENSE",
    "MANIFEST.in",
    "NOTICE",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
})

# Binary and controller/runtime artifacts must never become part of the
# customer-facing flat direct-launch archive.  The freshly built extension is
# consumed by NPUKernelBench before finalization, but the archive contains the
# reproducible sources and build recipe instead of a host-specific binary.
DIRECT_LAUNCH_BINARY_SUFFIXES = frozenset({
    ".a",
    ".bin",
    ".dll",
    ".dylib",
    ".exe",
    ".o",
    ".obj",
    ".pyc",
    ".pyd",
    ".so",
})

DIRECT_LAUNCH_ARCHIVE_EXCLUDED_DIRS = frozenset({
    ".lingxi_verify_logs",
    ".npubench_candidate",
    ".source_arch22",
    ".tilelang2ascendc_source",
    "__pycache__",
    "build",
    "dist",
    "install",
    "logs",
    # NPUKernelBench/controller evidence and the graybox plugin snapshot are
    # runtime inputs, not customer-deliverable kernel sources.  These entries
    # belong to the strict TileLang2AscendC delivery profile; the established legacy
    # archive policy remains untouched.
    ".graybox_plugin_runtime",
    "npubench_evidence",
    "output",
    "port_source",
    "reference_inputs",
    "reports",
    "source_stage",
})

DIRECT_LAUNCH_ARCHIVE_EXCLUDED_FILES = frozenset({
    "analysis.md",
    ".DS_Store",
    "PROGRESS.md",
    "a_tier_manifest.json",
    "audit_self_critic_post_worker.md",
    "construction_manifest.json",
    "evaluation.json",
    "evaluation_results.json",
    "failures_ledger.md",
    "finalize_precheck.md",
    "fused_analysis.md",
    "kb_draft_from_user_decision.md",
    "knowledge_update.md",
    "op_classification.json",
    "orchestrator_events.jsonl",
    "optimization_directive.md",
    "optimization_log.md",
    "perf_report.json",
    "performance.json",
    "progress.md",
    "probe_report.md",
    "reference_manifest.jsonl",
    "report.md",
    "self_critic_report.md",
    "state_transitions.jsonl",
    "user_decision.md",
    "verification.json",
})

# These are controller/runtime products, not candidate delivery files.  Keep
# the list intentionally explicit: an unrecognized dot file must be reported
# instead of silently disappearing from an otherwise finalized archive.
DIRECT_LAUNCH_ARCHIVE_EXCLUDED_FILE_PREFIXES = (
    ".active_agent_",
    ".agent_died_at_",
    ".cba_required_routes",
    ".cc_envelope_log",
    ".cc_stream_log_",
    ".critic_invoke_log",
    ".delegation_scan_",
    ".finalize_loop_nonconvergent",
    ".finalized-",
    ".kb_merged",
    ".kernel_worker_active",
    ".npubench_",
    ".opgen",
    ".optimizer_kernel_sig",
    "msprof_opt_",
    ".rollback_history",
    ".stop_gate_",
    ".tilelang2ascendc_source.",
    ".user_decision_",
    "audit_self_critic_post_worker",
)

DIRECT_LAUNCH_REFERENCE_ARTIFACT_PREFIXES = (
    "a3_capture",
    "edge_dataset",
    "reference_capture",
)

DIRECT_PROFILE_LEGACY = "legacy"
TILELANG_PROFILE_VALID = "valid_tilelang2ascendc"
TILELANG_PROFILE_INVALID = "invalid_tilelang2ascendc_claim"

# GE op_host delivery files the finalize GE assembler injects at workspace
# top-level op_host/ (FA-class trio + op-parameterized def/infershape/tiling).
TILELANG_GE_OPHOST_DELIVERY_FILENAMES = frozenset({
    "ge_host_shim.h",
    "wp_fa_host_cache.h",
    "wp_fa_host_tiling.h",
})
TILELANG_GE_OPHOST_DELIVERY_SUFFIXES = (
    "_def.cpp",
    "_infershape.cpp",
    "_tiling.cpp",
    "_tiling_common.h",
)

# GE registration subtree (op_host/config/<soc>/{<op>_binary.json,
# <op>_simplified_key.ini}).  The kw brief has the worker emit these for the
# GE op-registration flow, but the direct-launch (non-GE) delivery contract
# does not consume them — the delivery is the model_new_ascendc.py + kernel/
# direct-launch package, no GE op_host five-piece.  Promotion therefore omits
# the whole subtree with a WARN instead of hard-failing on it (2026-08-29
# 2_FFN_evo: a leftover *_simplified_key.ini blocked an otherwise-PASS
# finalize with a bare exit 7).  Fail-closed stays reserved for genuinely
# unknown file types outside this known GE-registration shape.
DIRECT_LAUNCH_GE_REGISTRATION_SUBTREE = ("op_host", "config")


RUNNER_BUILD_RECEIPT_FILENAMES: tuple[str, ...] = (
    "tilelang2ascendc_build_receipt.json",
)


def _sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest of one file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> Optional[dict[str, Any]]:
    """Return one JSON object read from a regular, non-symlink file.

    ``None`` means "no usable receipt here" — an absent, unsafe, unreadable or
    malformed file is deliberately not an error: the caller then keeps the
    strict worker-attested provenance path.  The suppression is logged at
    DEBUG so a mis-written receipt is still diagnosable.
    """
    if not path.is_file() or path.is_symlink():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logging.getLogger(__name__).debug(
            "Ignoring unreadable build receipt %s.", path, exc_info=error
        )
        return None
    return payload if isinstance(payload, dict) else None


def _runner_build_receipt(evidence: Path) -> Optional[dict[str, Any]]:
    """Return the controlled-build receipt when the runner reported PASS."""
    for name in RUNNER_BUILD_RECEIPT_FILENAMES:
        receipt = _read_json_object(evidence / name)
        if receipt is None:
            continue
        return receipt if receipt.get("status") == "PASS" else None
    return None


def _resolved_workspace_file(root: Path, raw: Any) -> Optional[Path]:
    """Resolve a declared provenance path to a real path inside ``root``."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = Path(raw)
    candidate = candidate if candidate.is_absolute() else root / candidate
    try:
        if candidate.is_symlink():
            return None
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved


def _declared_build_artifact(root: Path, raw: Any) -> Optional[Path]:
    """Return a declared build artifact path when it is a workspace file."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = Path(raw)
    candidate = candidate if candidate.is_absolute() else root / candidate
    try:
        if candidate.is_symlink() or not candidate.is_file():
            return None
        candidate.resolve(strict=True).relative_to(root)
    except (OSError, ValueError):
        return None
    return candidate


def _built_object_path(
    root: Path, build_dir: Path, cp: Mapping[str, Any], source_resolved: Path
) -> Optional[Path]:
    """Prefer the worker-recorded object, else the built ``<source>.o``."""
    object_path = _declared_build_artifact(root, cp.get("object"))
    if object_path is not None:
        return object_path
    matches = [
        item
        for item in build_dir.rglob(source_resolved.name + ".o")
        if item.is_file() and not item.is_symlink()
    ]
    return matches[0] if matches else None


def _built_shared_lib(
    root: Path, build_dir: Path, cp: Mapping[str, Any]
) -> Optional[Path]:
    """Locate the built extension module directly under ``kernel/build``."""
    shared_path = _declared_build_artifact(root, cp.get("shared_lib"))
    if shared_path is not None:
        return shared_path
    so_files = sorted(
        item
        for item in build_dir.rglob("*.so")
        if item.is_file() and not item.is_symlink()
    )
    if not so_files:
        return None
    # Prefer the authored extension module (underscore prefix, e.g.
    # _optimized_flash_attention_ext...so) — some CMake layouts place
    # it under kernel/build/lib/ (audit H3).  A lone candidate is
    # accepted as fallback; ambiguous layouts without a clear
    # extension module stay on the strict worker-attested path.
    extension = [item for item in so_files if item.name.startswith("_")]
    if len(extension) == 1:
        return extension[0]
    if len(so_files) == 1:
        return so_files[0]
    return None


def _delivery_path_is_retained(profile: str, workspace_rel_path: str) -> bool:
    """Classify only explicitly retained strict-delivery files."""
    if profile == DIRECT_PROFILE_LEGACY:
        return True
    if profile in {TILELANG_PROFILE_INVALID}:
        return False
    if not isinstance(workspace_rel_path, str):
        return False
    try:
        relative = PurePosixPath(workspace_rel_path.replace("\\", "/"))
    except TypeError:
        return False
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return False
    parts = relative.parts
    name = relative.name
    if any(part in DIRECT_LAUNCH_ARCHIVE_EXCLUDED_DIRS for part in parts):
        return False
    if any(part.startswith(".") for part in parts):
        return False
    if len(parts) == 1:
        return (
            name in {"model_new_ascendc.py", "README.md"}
            or name in DIRECT_LAUNCH_ROOT_AUXILIARY_FILENAMES
            or Path(name).suffix.lower() == ".py"
        )
    if profile == TILELANG_PROFILE_VALID and parts[0] == "op_host":
        # GE op_host delivery trio: the finalize GE assembler injects these
        # at workspace-top-level op_host/ (they need the GE framework and
        # cannot ride the kernel build).  They ARE required delivery
        # source — the GE_OPHOST_RAW_CANN_COPY gate verifies them — so the
        # TileLang2AscendC delivery profile retains them explicitly.
        return (
            name in TILELANG_GE_OPHOST_DELIVERY_FILENAMES
            or name.endswith(TILELANG_GE_OPHOST_DELIVERY_SUFFIXES)
        )
    if parts[0] != "kernel":
        return False
    if name in DIRECT_LAUNCH_BUILD_FILENAMES:
        return True
    suffix = Path(name).suffix.lower()
    return (
        suffix in DIRECT_LAUNCH_CPP_SOURCE_SUFFIXES
        or suffix in DIRECT_LAUNCH_KERNEL_AUXILIARY_SUFFIXES
    )


def _delivery_path_rejection(
    profile: str, workspace_rel_path: str
) -> Optional[str]:
    """Return a hard-failure reason for a non-excluded direct path.

    ``None`` means either the path is retained or it is an intentional
    controller/runtime exclusion.  The caller can therefore distinguish
    omission from a delivery contract error without changing the legacy
    plugin protocol's bool-only ``should_archive_path`` surface.
    """
    if profile not in {TILELANG_PROFILE_VALID}:
        return None
    delivery_label = "TileLang2AscendC"
    if not isinstance(workspace_rel_path, str):
        return f"{delivery_label} delivery path is not a string"
    try:
        relative = PurePosixPath(workspace_rel_path.replace("\\", "/"))
    except TypeError:
        return f"{delivery_label} delivery path is malformed"
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return f"{delivery_label} delivery path is unsafe: {workspace_rel_path!r}"
    parts = relative.parts
    name = relative.name
    if any(part in DIRECT_LAUNCH_ARCHIVE_EXCLUDED_DIRS for part in parts):
        return None
    if (
        name in DIRECT_LAUNCH_ARCHIVE_EXCLUDED_FILES
        or any(name.startswith(prefix) for prefix in DIRECT_LAUNCH_ARCHIVE_EXCLUDED_FILE_PREFIXES)
        or any(name.startswith(prefix) for prefix in DIRECT_LAUNCH_REFERENCE_ARTIFACT_PREFIXES)
    ):
        return None
    # Option B (2026-08-22, campaign): the strict whitelist previously
    # failed the whole promotion on harness-runtime artifacts OUTSIDE the
    # customer-facing delivery territory (stream logs, markers, docs,
    # manifests at the workspace root).  Those are controller-owned and
    # are now EXCLUDED, not errors — hard-failing on them churned the
    # campaign: the MUSEAttention promotion reported 30 errors, all of
    # them harness artifacts, after a fully precision/perf-PASS
    # candidate.  Fail-closed behavior
    # is preserved only for the delivery territory itself: the kernel/
    # subtree and the top-level op_host/ GE delivery files.
    if parts[0] not in ("kernel", "op_host"):
        return None
    if any(part.startswith(".") for part in parts):
        return (
            f"{delivery_label} delivery contains an unrecognized hidden artifact: "
            f"{relative.as_posix()}"
        )
    if Path(name).suffix.lower() in DIRECT_LAUNCH_BINARY_SUFFIXES:
        return (
            f"{delivery_label} delivery contains a binary/build artifact instead of source: "
            f"{relative.as_posix()}"
        )
    if _delivery_path_is_retained(profile, workspace_rel_path):
        return None
    if parts[:2] == DIRECT_LAUNCH_GE_REGISTRATION_SUBTREE:
        # Known GE-registration leftover (see the constant's comment): omit
        # from the archive and record a WARN instead of erroring — the
        # worker cannot be expected to un-emit what the brief asked for,
        # and the direct-launch delivery never consumes this subtree.
        logging.getLogger(__name__).warning(
            "delivery promotion omits GE registration artifact (not part of "
            "the direct-launch delivery contract): %s",
            relative.as_posix(),
        )
        return None
    return (
        f"{delivery_label} delivery contains an unrecognized file that would be omitted: "
        f"{relative.as_posix()}"
    )


def _delivery_path_is_archived(profile: str, workspace_rel_path: str) -> bool:
    """Return whether a path is explicitly retained by the strict profile."""
    return _delivery_path_rejection(
        profile, workspace_rel_path
    ) is None and _delivery_path_is_retained(profile, workspace_rel_path)


@dataclass(frozen=True)
class _PortA3ArchiveView:
    """A single archive-promotion snapshot for the active Port A3 profile.

    Finalization can touch hundreds of files.  It must not reread mutable
    durable state for every path and allow one promotion to switch profiles
    halfway through.  The view captures the validated profile once and keeps
    all path and layout decisions tied to that decision.
    """

    plugin: "PortA3Plugin"
    profile: str
    reason: str

    @property
    def rejection_reason(self) -> Optional[str]:
        if self.profile == TILELANG_PROFILE_INVALID:
            return f"port-aclnn-tilelang2ascendc delivery profile is invalid: {self.reason}"
        return None

    @property
    def requires_regular_files(self) -> bool:
        return self.profile in {TILELANG_PROFILE_VALID}

    @property
    def strict_delivery(self) -> bool:
        return self.profile in {TILELANG_PROFILE_VALID}

    def should_archive_path(self, workspace_rel_path: str) -> bool:
        return _delivery_path_is_archived(self.profile, workspace_rel_path)

    def rejected_archive_path_reason(self, workspace_rel_path: str) -> Optional[str]:
        """Explain an unrecognized direct-delivery path, when applicable."""
        return _delivery_path_rejection(self.profile, workspace_rel_path)

    def resolve_archive_target(self, workspace_rel_path: str, op: str) -> str:
        if self.profile in {TILELANG_PROFILE_VALID}:
            return workspace_rel_path
        return self.plugin.resolve_archive_target(workspace_rel_path, op)


class PortA3Plugin(BasePlugin):
    name = "port_a3_to_a5"
    cli_flag = "--port-a3"

    # ── Internal helpers ───────────────────────────────────────────────
    # Class-member order is static methods, then the classmethod profile
    # discriminator, then the instance protocol methods below.
    @staticmethod
    def _port_source_kind(workspace: Path) -> Optional[str]:
        """Read the explicit source kind from durable migration state.

        The direct-launch delivery profile is opt-in by design.  In
        particular, a ``kernel/`` directory, a source filename, or a legacy
        top-level state field must never silently switch an existing
        ``ops-nn`` workspace to the flat profile.  Treat an unsafe/malformed
        state file as absent so legacy callers retain their historical
        behavior; the state validation layer remains responsible for the
        fail-closed diagnostic when the workspace is resumed.
        """
        state_file = Path(workspace) / ".opgen_state.json"
        try:
            metadata = state_file.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                return None
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(state, Mapping):
            return None
        source = state.get("port_source")
        if not isinstance(source, Mapping):
            return None
        kind = source.get("kind")
        return kind if isinstance(kind, str) else None

    @staticmethod
    def _load_durable_state(workspace: Path) -> Mapping[str, Any] | None:
        """Load one regular, non-symlink state file for profile binding."""
        state_file = Path(workspace) / ".opgen_state.json"
        try:
            metadata = state_file.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                return None
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        return state if isinstance(state, Mapping) else None

    @staticmethod
    def _reconcile_runner_built_provenance(
        root: Path, cp: Mapping[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Rebuild compiled_provenance from the controlled-build receipt.

        For the build-isolated port-aclnn-tilelang2ascendc route the graybox
        worker has no CANN toolchain and
        can never attest the built artifact bytes: the O5 controlled target
        runner owns the build instead, and _clear_harness_build_artifacts
        removes kernel/build before every worker respawn.  When the runner's
        build receipt is PASS and the artifacts are present, reconcile the
        evidence from the real workspace bytes so the provenance gate verifies
        what the runner actually built.

        Returns None when no runner-owned receipt/artifacts are available; the
        caller then keeps the strict worker-attested path unchanged.
        """
        if _runner_build_receipt(root / "npubench_evidence") is None:
            return None
        build_dir = root / "kernel" / "build"
        if not build_dir.is_dir():
            return None
        source_resolved = _resolved_workspace_file(root, cp.get("source"))
        if source_resolved is None:
            return None
        object_path = _built_object_path(root, build_dir, cp, source_resolved)
        if object_path is None:
            return None
        shared_path = _built_shared_lib(root, build_dir, cp)
        if shared_path is None:
            return None
        source_digest = _sha256_file(source_resolved)
        reconciled = dict(cp)
        reconciled.update(
            {
                "source": str(source_resolved.relative_to(root)),
                "deployed_source": str(source_resolved.relative_to(root)),
                "object": str(object_path.resolve().relative_to(root)),
                "shared_lib": str(shared_path.resolve().relative_to(root)),
                "workspace_source_sha256": source_digest,
                "deploy_source_sha256": source_digest,
                "built_from_source_sha256": source_digest,
                "object_sha256": _sha256_file(object_path),
                "shared_lib_sha256": _sha256_file(shared_path),
                "_reconciled_from_runner_build": True,
            }
        )
        return reconciled

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

    @staticmethod
    def _tilelang2ascendc_delivery_sources(workspace: Path) -> list[str]:
        """Return structural delivery errors for the TileLang2AscendC project layout."""
        root = Path(workspace)
        failures: list[str] = []

        def _regular(path: Path, label: str) -> None:
            try:
                metadata = path.lstat()
            except OSError:
                failures.append(f"{label} is missing")
                return
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_size == 0:
                failures.append(f"{label} must be a non-empty regular non-symlink file")

        _regular(root / "model_new_ascendc.py", "model_new_ascendc.py")
        _regular(root / "kernel" / "CMakeLists.txt", "kernel/CMakeLists.txt")
        _regular(root / "kernel" / "register.cpp", "kernel/register.cpp")
        for name in ("op_host", "op_kernel"):
            directory = root / "kernel" / name
            try:
                metadata = directory.lstat()
            except OSError:
                failures.append(f"kernel/{name}/ is missing")
                continue
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                failures.append(f"kernel/{name}/ must be a real directory")
                continue
            source_files = []
            for path in sorted(directory.rglob("*")):
                try:
                    item = path.lstat()
                except OSError:
                    failures.append(f"kernel/{name}/{path.name} is unreadable")
                    continue
                if stat.S_ISLNK(item.st_mode) or (path.is_file() and item.st_nlink != 1):
                    failures.append(f"kernel/{name}/{path.relative_to(directory).as_posix()} is unsafe")
                elif path.is_file() and path.suffix.lower() in DIRECT_LAUNCH_IMPLEMENTATION_SUFFIXES:
                    source_files.append(path.relative_to(root).as_posix())
            if not source_files:
                failures.append(f"kernel/{name}/ has no authored C/C++ source")
        return failures

    @staticmethod
    def _should_archive_path_for_profile(profile: str, workspace_rel_path: str) -> bool:
        """Return whether a path is explicitly retained by the strict profile."""
        return _delivery_path_is_archived(profile, workspace_rel_path)

    @staticmethod
    def _direct_path_is_retained(profile: str, workspace_rel_path: str) -> bool:
        """Classify only explicitly retained strict-delivery files."""
        return _delivery_path_is_retained(profile, workspace_rel_path)

    @staticmethod
    def _archive_path_rejection_for_profile(
        profile: str, workspace_rel_path: str
    ) -> Optional[str]:
        """Return a hard-failure reason for a non-excluded direct path."""
        return _delivery_path_rejection(profile, workspace_rel_path)

    @staticmethod
    def _npubench_binding(workspace: Path) -> Optional[Mapping[str, Any]]:
        """Return an explicitly selected NPUKernelBench block, if present.

        The detailed completeness and immutable-stage checks stay in the
        provider contract.  This narrow discriminator is deliberately enough
        to route a malformed ``source=npubench`` state to that fail-closed
        contract instead of accidentally applying a live-A3 provenance rule.
        """
        state_path = Path(workspace) / ".opgen_state.json"
        try:
            if state_path.is_symlink() or not state_path.is_file():
                return None
            state = json.loads(state_path.read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(state, Mapping):
            return None
        reference = state.get("reference")
        if not isinstance(reference, Mapping):
            return None
        return reference if reference.get("source") == "npubench" else None

    @classmethod
    def _tilelang2ascendc_profile_status(cls, workspace: Path) -> tuple[str, str]:
        """Return the independent TileLang2AscendC delivery-profile verdict."""
        state = cls._load_durable_state(Path(workspace))
        if state is None:
            return DIRECT_PROFILE_LEGACY, "durable state is absent or unreadable"
        source = state.get("port_source")
        if not isinstance(source, Mapping) or source.get("kind") != TILELANG2ASCENDC_SOURCE_KIND:
            return DIRECT_PROFILE_LEGACY, "source kind is not port-aclnn-tilelang2ascendc"
        candidate = state.get("candidate")
        if not isinstance(candidate, Mapping) or candidate.get("kind") != "tilelang2ascendc_custom_op":
            return TILELANG_PROFILE_INVALID, "candidate.kind must be tilelang2ascendc_custom_op"
        if candidate.get("target_arch") != "arch35":
            return TILELANG_PROFILE_INVALID, "candidate.target_arch must be arch35"
        try:
            from tilelang2ascendc_source import verify_tilelang2ascendc_source_stage

            valid, reason, _manifest = verify_tilelang2ascendc_source_stage(Path(workspace), state)
        except Exception as exc:  # pragma: no cover - import/environment guard
            return TILELANG_PROFILE_INVALID, f"TileLang2AscendC source verifier failed: {type(exc).__name__}: {exc}"
        if not valid:
            return TILELANG_PROFILE_INVALID, f"TileLang2AscendC source stage invalid: {reason}"
        reference = state.get("reference")
        if not isinstance(reference, Mapping) or reference.get("source") != "npubench":
            return TILELANG_PROFILE_INVALID, "port-aclnn-tilelang2ascendc requires reference.source=npubench"
        try:
            from reference_source import resolve_reference_binding
            from npubench.npubench_inputs import verify_npubench_stage

            bound_reference = resolve_reference_binding(state)
            valid, reason, _manifest = verify_npubench_stage(Path(workspace), bound_reference)
        except Exception as exc:  # pragma: no cover - import/environment guard
            return TILELANG_PROFILE_INVALID, f"NPUBench reference verifier failed: {type(exc).__name__}: {exc}"
        if not valid:
            return TILELANG_PROFILE_INVALID, f"NPUBench reference stage invalid: {reason}"
        return TILELANG_PROFILE_VALID, "TileLang2AscendC source, candidate, and NPUBench bindings verified"

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

    def kernel_cpp_dirs_for_workspace(self, workspace: Path) -> tuple[str, ...]:
        """Return the candidate C++ scan roots for one durable workspace.

        ``kernel_cpp_dirs()`` intentionally remains the legacy ops-nn answer
        so protocol users and old archives keep their established behavior.
        New scanner callers should use this workspace-aware hook:
        TileLang2AscendC candidates put all host/device C++ under ``kernel/``
        and must not scan stale ``op_host/`` or ``op_kernel/`` trees that may
        be present as unrelated harness/source material.
        """
        tile_status, _tile_reason = self._tilelang2ascendc_profile_status(workspace)
        if tile_status in {TILELANG_PROFILE_VALID, TILELANG_PROFILE_INVALID}:
            return ("kernel",)
        return self.kernel_cpp_dirs()

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
        """port_a3_to_a5 mode: pass_b is DEGENERATE by design.

        Pass A consumes the immutable, source-specific reference evidence
        (live A3 capture or frozen NPUKernelBench bundle), so a second pass_b verifier
        would only repeat the same fixed corpus.
        """
        return False

    def pass_b_default_when_skipped(self) -> dict:
        """Canonical N/A shape for port_a3 mode pass_b. Worker MUST write
        this exact shape; finalize gate rejects deviations.
        """
        return {
            "status": "N/A",
            "reason": (
                "port_a3_to_a5 mode: pass_b is subsumed by pass_a; "
                "source-specific immutable evidence makes pass_b degenerate."
            ),
            "method": "n/a — port_a3 migration pass_b not applicable",
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

        # Marker 2: instantiation — `<name> = ModelNew()` (named form) OR
        # a one-expression instantiate+call `ModelNew()(inputs)` (audit L1,
        # 2026-08-22: the named-instance regex rejected the legitimate
        # one-expression form as decorative bypass).
        direct_call_re = re.compile(r"ModelNew\s*\([^)]*\)\s*\(")
        instantiation_re = re.compile(r"(\w+)\s*=\s*ModelNew\s*\(")
        direct = direct_call_re.search(code_text)
        m = instantiation_re.search(code_text)
        if m is None and direct is None:
            return (
                "port_a3 decorative-bypass gate (P96 follow-up): "
                "pass_a_runner.py never instantiates ModelNew (no "
                "`<name> = ModelNew()` line in CODE). ModelNew wrapper "
                "bypassed (OL-160 USAGE-gap)."
            )
        if direct is not None:
            # One-expression `ModelNew()(inputs)` — instantiated AND invoked;
            # both markers satisfied in a single expression.
            return None
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
        reconciled = self._reconcile_runner_built_provenance(root, cp)
        if reconciled is not None:
            cp = reconciled

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

    # ── Archive layout (migrated from finalize_pipeline.py phase 2) ────
    def archive_project_subdir(self) -> Optional[str]:
        return "a3_to_a5_port"

    def archive_layout_mapping(self, workspace: Path) -> dict[str, str]:
        """Returns path-prefix mapping. Caller uses this dict OR
        calls resolve_archive_target() per-file. Both APIs supported.
        """
        return dict(ARCHIVE_PATH_MAP)

    def check_tilelang2ascendc_delivery_completeness(self, workspace: Path) -> Optional[str]:
        """Validate the TileLang2AscendC candidate before archive promotion."""
        status, reason = self._tilelang2ascendc_profile_status(Path(workspace))
        if status == DIRECT_PROFILE_LEGACY:
            return None
        if status == TILELANG_PROFILE_INVALID:
            return f"port-aclnn-tilelang2ascendc delivery profile is invalid: {reason}"
        failures = self._tilelang2ascendc_delivery_sources(Path(workspace))
        if not failures:
            try:
                from npubench.npubench_target import _validate_candidate_for_controlled_build

                _validate_candidate_for_controlled_build(
                    Path(workspace), TILELANG2ASCENDC_SOURCE_KIND
                )
            except Exception as exc:
                failures.append(str(exc))
        if failures:
            return "port-aclnn-tilelang2ascendc delivery completeness: " + "; ".join(failures)
        return None

    def check_op_host_completeness(self, workspace: Path) -> Optional[str]:
        """Choose the delivery-completeness gate from durable source state.

        The legacy archive path keeps BasePlugin's PB-33 behavior.
        Only a persisted ``port_source.kind=port-aclnn-tilelang2ascendc``
        replaces it with the TileLang2AscendC project contract above.
        """
        tile_status, _tile_reason = self._tilelang2ascendc_profile_status(workspace)
        if tile_status in {TILELANG_PROFILE_VALID, TILELANG_PROFILE_INVALID}:
            return self.check_tilelang2ascendc_delivery_completeness(workspace)
        return super().check_op_host_completeness(workspace)

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

    def resolve_archive_target_for_workspace(
        self, workspace: Path, workspace_rel_path: str, op: str
    ) -> str:
        """Resolve archive paths with an explicit direct-launch flat branch.

        The direct product preserves ``kernel/`` unchanged — it must never be
        remapped to ``op_kernel/``.  Legacy workspaces still delegate to the
        established mapper above, including migration-plan-to-docs routing.
        """
        return self.freeze_archive_view(workspace, op).resolve_archive_target(
            workspace_rel_path, op
        )

    def freeze_archive_view(self, workspace: Path, op: str) -> _PortA3ArchiveView:
        """Freeze profile-dependent archive policy for one promotion pass."""
        profile, reason = self._tilelang2ascendc_profile_status(workspace)
        return _PortA3ArchiveView(self, profile, reason)

    def should_archive_path(self, workspace: Path, workspace_rel_path: str) -> bool:
        """Return whether a strict-profile workspace path belongs in delivery.

        This is a delivery allow/deny decision, not an evaluator decision.  It
        excludes controller state, frozen source/reference inputs, historical
        reports and build artifacts while retaining candidate sources and their
        explicit ``kernel/CMakeLists.txt`` recipe.  Legacy workspaces
        retain the historical archive-everything policy.
        """
        profile, _reason = self._tilelang2ascendc_profile_status(workspace)
        return self._should_archive_path_for_profile(profile, workspace_rel_path)

    def check_verify_path_provenance(
        self, workspace: Path, vj: dict
    ) -> Optional[str]:
        """DEBT-NEW (2026-05-14) port_a3-specific verify-path provenance.

        Rejects PASS when workspace can't prove the A5-side verify
        invoked OUR built kernel via aclnn-direct C++ runner / ctypes
        shim (vs PyTorch dispatcher falling back to AICPU).

        For ``reference.source=npubench``, delegate to the immutable-provider
        proof below.  Every other port-A3 workspace remains on the live-A3
        branch, where these two checks are
        AND'd:
        1. verification.json.truth_source contains a sanctioned live-A3 token
        2. Workspace has at least one *_runner.cpp / *_shim.cpp /
           aclnn_*.cpp artifact

        """
        if self._npubench_binding(workspace) is not None:
            return self.check_npubench_provenance(workspace, vj)

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

    def extra_finalize_checks(self):
        """Register immutable-provider proofs on the real finalize path.

        The gate is a no-op for legacy/live-A3 state, so it cannot relax
        or alter the pre-existing A3-CANN truth-source contract.
        """
        return [
            (
                "port_a3_npubench_provenance",
                self.check_npubench_provenance,
            )
        ]

    def check_npubench_provenance(
        self, workspace: Path, vj: dict
    ) -> Optional[str]:
        """Validate runner-owned NPUKernelBench evidence without A3 fallback.

        NPUKernelBench has one immutable old-format task bundle and an
        orchestrator-owned precision/performance result.  It does not use the
        historical live-A3 runner-file or truth-source vocabulary.
        """
        if self._npubench_binding(workspace) is None:
            return None
        try:
            from npubench.npubench_finalize_contract import (
                resolve_npubench_workspace,
                validate_npubench_finalize_evidence,
            )

            is_npubench, source_error = resolve_npubench_workspace(Path(workspace))
        except Exception as exc:
            return (
                "npubench provenance: could not resolve provider state: "
                f"{type(exc).__name__}: {exc}"
            )
        if not is_npubench:
            return None
        if source_error:
            return source_error
        return validate_npubench_finalize_evidence(Path(workspace), vj)

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
