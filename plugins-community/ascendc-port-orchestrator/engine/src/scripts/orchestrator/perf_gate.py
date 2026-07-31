# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Perf-gate profile foundation — Phase A of PERF_GATE_PROFILE_DESIGN_2026_05_20.

Single source of truth for all perf-related gates in the orchestrator.

USAGE (consumer side — each gate is ONE line):

    from perf_gate import resolve_profile

    profile = resolve_profile(workspace)
    if profile.measure_reference_perf:
        ...  # run Phase O2 reference perf measurement

    if profile.allow_ko_escalation and ratio < threshold:
        route_to_ko()

USAGE (orchestrator entry — once per run):

    from perf_gate import write_profile_marker

    if perf_threshold is not None:
        write_profile_marker(workspace, perf_threshold)

CLI MAPPING (driven from orchestrator.py --perf-threshold):

    --perf-threshold=None   → DEFAULT profile (band-aware via plugin)
    --perf-threshold=0      → PRECISION_ONLY profile
    --perf-threshold=0.9    → synthesized "custom_t0.90" cloning DEFAULT,
                              finalize_threshold=0.9
    --perf-threshold=0.3    → synthesized "custom_t0.30" cloning DEFAULT,
                              finalize_threshold=0.3

For the YAML side (state_machine.yaml condition primitives), see
`workflow/state_machine.py:_eval_condition` Phase B wire-up.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional


# Marker file name — workspace-local; resolve_profile() reads this at every
# gate call. Written by orchestrator at run start; never edited mid-run.
MARKER_FILENAME = ".perf_gate_profile.json"


@dataclass(frozen=True)
class PerfGateProfile:
    """A named bundle of decisions across the perf-gate touchpoints.

    All 9 gates consult this profile via a single resolver. Profile selection
    happens ONCE at orchestrator entry; the profile flows down via workspace
    state. No gate consults marker files directly.

    See docs/design/PERF_METHODOLOGY_NOTES.md#perf-gate-profile-design-2026-05-20 §3 + §4.
    """

    name: str
    """Human-readable: "default" / "precision_only" / "hero_op_strict" /
    "custom_t<N>" (synthesized for non-built-in --perf-threshold values)."""

    measure_reference_perf: bool
    """Phase O2 reference perf measurement on the vendor ref op. False → skip
    entirely (saves wall + container cycles)."""

    include_perf_in_brief: bool
    """kw_brief Phase D perf instructions. False → omit perf section from
    brief; worker doesn't measure or report ratio."""

    finalize_threshold: Optional[float]
    """schema_norm finalize gate threshold:
    - None: defer to plugin.ko_escalation_threshold(op_class) (band-aware)
    - 0.0: accept ANY ratio (including null/missing) when precision PASS
    - >0: global override of band-aware (e.g. 0.9 for hero op).
    """

    allow_ko_escalation: bool
    """state_machine kw→ko route on perf<threshold. False suppresses the
    transition regardless of ratio."""

    allow_ar_escalation: bool
    """V3.7.11 ko-plateau → researcher. False suppresses."""

    allow_fo_escalation: bool
    """fused-optimizer plateau path. False suppresses."""

    require_ratio_in_verification: bool
    """verification.json schema gate. False allows null/missing ratio."""

    require_perf_artifacts: bool
    """Finalize archive requires perf.json + perf.md. False makes them
    optional."""

    escalation_overrides: dict = field(default_factory=dict)
    """Per-state profile auto-switch: `dict[state_id, profile_name]`. When the
    orchestrator enters one of these states, the resolver returns the mapped
    sub-profile for that state's duration; the outer paradigm's profile resumes
    after the state exits. Currently unused (empty) — kept as a generic hook for
    future per-state profile overrides.
    """

    # ---- Roofline-mode fields (NODE-9 Phase B2, 2026-05-28) ----

    enable_roofline_eval: bool = False
    """When True, consult roofline_eval.analyze() before ko escalation.
    If efficiency > roofline_skip_threshold_pct, skip optimizer entirely."""

    roofline_skip_threshold_pct: float = 80.0
    """Efficiency % above which optimizer is skipped (default 80 per design doc §2.2)."""

    allow_partial_on_subfloor: bool = False
    """route-a first e2e (2026-07-14, --defer-perf-opt): when perf < floor but
    precision + det PASS, finalize PARTIAL_PERSIST (honest sub-floor perf recorded)
    instead of escalating a perf optimizer OR aborting. Perf-optimization itself is
    deferred to a later rung. False for all other profiles (escalate-or-pass behavior
    unchanged)."""


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

PRECISION_ONLY = PerfGateProfile(
    name="precision_only",
    measure_reference_perf=False,
    include_perf_in_brief=False,
    finalize_threshold=0.0,
    allow_ko_escalation=False,
    allow_ar_escalation=False,
    allow_fo_escalation=False,
    require_ratio_in_verification=False,
    require_perf_artifacts=False,
    escalation_overrides={},
)

HERO_OP_STRICT = PerfGateProfile(
    name="hero_op_strict",
    measure_reference_perf=True,
    include_perf_in_brief=True,
    finalize_threshold=0.9,
    allow_ko_escalation=True,
    allow_ar_escalation=True,
    allow_fo_escalation=True,
    require_ratio_in_verification=True,
    require_perf_artifacts=True,
    escalation_overrides={},
)

DEFAULT = PerfGateProfile(
    name="default",
    measure_reference_perf=True,
    include_perf_in_brief=True,
    finalize_threshold=None,  # → plugin band-aware
    allow_ko_escalation=True,
    allow_ar_escalation=True,
    allow_fo_escalation=True,
    require_ratio_in_verification=True,
    require_perf_artifacts=True,
    escalation_overrides={},
)

PERF_DEFERRED = PerfGateProfile(
    name="perf_deferred",
    # Distinct from PRECISION_ONLY: perf IS still measured + recorded (honest
    # sub-floor ratio in verification.json), but perf-optimization is DEFERRED —
    # no ko/ar/fo escalation, and a sub-floor ratio finalizes PARTIAL_PERSIST
    # instead of escalate→abort. route-a first e2e (2026-07-14): correctness +
    # det PASS, perf 0.13× honestly recorded, perf-opt is a later rung.
    measure_reference_perf=True,
    include_perf_in_brief=True,
    finalize_threshold=None,          # band-aware → sub-floor correctly detected + recorded
    allow_ko_escalation=False,        # no perf-optimizer escalation (deferred)
    allow_ar_escalation=False,
    allow_fo_escalation=False,
    require_ratio_in_verification=True,   # ratio MUST be present (honest record, not hidden)
    require_perf_artifacts=True,
    allow_partial_on_subfloor=True,   # THE lever: sub-floor perf → PARTIAL_PERSIST finalize
    escalation_overrides={},
)

_BUILT_IN_BY_NAME: dict[str, PerfGateProfile] = {
    "default": DEFAULT,
    "precision_only": PRECISION_ONLY,
    "hero_op_strict": HERO_OP_STRICT,
    "perf_deferred": PERF_DEFERRED,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_profile(workspace: Path, *, current_state: Optional[str] = None) -> PerfGateProfile:
    """Return the active perf-gate profile for `workspace`.

    Resolution order:
    1. Read workspace/.perf_gate_profile.json (written by orchestrator at
       run start). If absent → DEFAULT.
    2. If `current_state` is provided AND the resolved profile has an
       `escalation_overrides[current_state]` entry, recursively resolve
       that override.

    The recursion is one level deep by convention — overrides should map to
    built-in profiles, not other override-bearing profiles. Cycles are
    impossible because override values are profile NAMES (strings), not
    dataclass instances.

    Args:
        workspace: workspace directory containing the marker
        current_state: state_machine state name (e.g. "await_probe").
            When provided, escalation_overrides is consulted.

    Returns:
        PerfGateProfile (always; never None — DEFAULT is the safe fallback)
    """
    profile = _read_marker_or_default(workspace)
    if current_state and current_state in profile.escalation_overrides:
        override_name = profile.escalation_overrides[current_state]
        override_profile = _BUILT_IN_BY_NAME.get(override_name)
        if override_profile is not None:
            return override_profile
    return profile


def write_profile_marker(
    workspace: Path,
    perf_threshold: Optional[float],
    *,
    roofline_mode: bool = False,
    roofline_skip_threshold_pct: float = 80.0,
    defer_perf_opt: bool = False,
) -> PerfGateProfile:
    """Write `.perf_gate_profile.json` marker per the CLI `--perf-threshold` arg.

    CLI mapping:
    - defer_perf_opt=True → PERF_DEFERRED profile (takes precedence; measures +
      records perf but defers optimization → sub-floor finalizes PARTIAL_PERSIST).
    - perf_threshold is None → no marker (caller should remove stale marker
      if any). Returns DEFAULT.
    - perf_threshold == 0 → PRECISION_ONLY profile
    - perf_threshold == 0.9 → HERO_OP_STRICT (recognized shortcut)
    - perf_threshold > 0 (other) → synthesize "custom_t<N>" profile

    Roofline-mode fields (NODE-9 Phase B3, 2026-05-28):
    - roofline_mode: enable roofline self-eval
    - roofline_skip_threshold_pct: efficiency % above which optimizer is skipped

    Args:
        workspace: workspace directory
        perf_threshold: from --perf-threshold CLI arg (or None if unset)
        roofline_mode: from --roofline-mode CLI flag
        roofline_skip_threshold_pct: from --roofline-skip-threshold CLI arg

    Returns:
        The PerfGateProfile that was selected + written (for caller logging)
    """
    if defer_perf_opt:
        profile = PERF_DEFERRED
        marker = workspace / MARKER_FILENAME
        data = {"name": profile.name, "finalize_threshold": profile.finalize_threshold}
        if roofline_mode:
            data["roofline_mode"] = True
            data["roofline_skip_threshold_pct"] = roofline_skip_threshold_pct
        marker.write_text(json.dumps(data))
        return profile

    if perf_threshold is None:
        # Caller responsible for removing stale marker; return DEFAULT.
        return DEFAULT

    profile = _profile_for_threshold(perf_threshold)
    marker = workspace / MARKER_FILENAME
    data: dict = {
        "name": profile.name,
        "finalize_threshold": profile.finalize_threshold,
    }
    if roofline_mode:
        data["roofline_mode"] = True
        data["roofline_skip_threshold_pct"] = roofline_skip_threshold_pct
    marker.write_text(json.dumps(data))
    return profile


def remove_profile_marker(workspace: Path) -> bool:
    """Remove the marker if present. Returns True if a marker was removed."""
    marker = workspace / MARKER_FILENAME
    if marker.exists():
        marker.unlink()
        return True
    return False


def synthesize_custom(threshold: float) -> PerfGateProfile:
    """Synthesize an anonymous profile cloning DEFAULT with a custom
    finalize_threshold. Used when --perf-threshold=<N> doesn't match a
    built-in named profile.

    Name format: `custom_t{N:.2f}` per main agent's Q2 §7 (2026-05-20).
    """
    return replace(
        DEFAULT,
        name=f"custom_t{threshold:.2f}",
        finalize_threshold=float(threshold),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _read_marker_or_default(workspace: Path) -> PerfGateProfile:
    """Read the marker file and return the corresponding profile.

    Marker format: `{"name": "<profile_name>", "finalize_threshold": <float|null>}`.

    Built-in profile names map directly. "custom_t<N>" names re-synthesize
    via `synthesize_custom(N)`. Unknown names fall back to DEFAULT (safe).
    """
    marker = workspace / MARKER_FILENAME
    if not marker.is_file():
        return DEFAULT
    try:
        data = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return DEFAULT

    name = data.get("name", "default")
    if name in _BUILT_IN_BY_NAME:
        profile = _BUILT_IN_BY_NAME[name]
    elif name.startswith("custom_t"):
        threshold = data.get("finalize_threshold")
        if threshold is not None:
            profile = synthesize_custom(float(threshold))
        else:
            return DEFAULT
    else:
        return DEFAULT

    # NODE-9 Phase B3: apply roofline-mode fields from marker
    if data.get("roofline_mode"):
        profile = replace(profile,
                          enable_roofline_eval=True,
                          roofline_skip_threshold_pct=float(
                              data.get("roofline_skip_threshold_pct", 80.0)))
    return profile


def _profile_for_threshold(threshold: float) -> PerfGateProfile:
    """Map a numeric threshold to a profile (built-in if recognized, else
    synthesized)."""
    if threshold == 0:
        return PRECISION_ONLY
    if threshold == 0.9:
        # Shortcut: --perf-threshold=0.9 picks the named HERO_OP_STRICT
        # profile (full perf chain, just stricter threshold).
        return HERO_OP_STRICT
    return synthesize_custom(threshold)


# ---------------------------------------------------------------------------
# Roofline-mode integration (NODE-9 Phase B2, 2026-05-28)
# ---------------------------------------------------------------------------

def resolve_roofline_gate(
    workspace: Path,
    *,
    ratio: float,
    profile: PerfGateProfile,
    op_type: str = "",
    soc_target: str = "a5",
    dtype_itemsize: int = 2,
    measured_time_s: float = 0.0,
    op_shape: Optional[dict] = None,
) -> bool:
    """Pre-ko roofline gate: should we escalate to optimizer?

    Consults roofline_eval when profile.enable_roofline_eval is True.
    If efficiency > profile.roofline_skip_threshold_pct, returns False
    (skip optimizer — near hardware ceiling, diminishing returns).

    When roofline eval is disabled or non-actionable, falls back to the
    existing band-aware threshold logic (ratio < threshold → escalate).

    Returns:
        True if optimizer SHOULD be invoked, False if it should be skipped.
    """
    if not profile.allow_ko_escalation:
        return False

    # Traditional gate: ratio-based.
    if not profile.enable_roofline_eval:
        threshold = profile.finalize_threshold
        if threshold is None:
            threshold = 1.0  # safe default (parity) when band-aware not resolved (owner-directed 2026-07-21, was 0.6)
        return ratio < threshold

    # Roofline-aware gate.
    try:
        from roofline_eval import analyze as _roofline_analyze
    except ImportError:
        # roofline_eval not available → fall back to ratio-based
        threshold = profile.finalize_threshold or 1.0
        return ratio < threshold

    result = _roofline_analyze(
        op_type=op_type, soc_target=soc_target,
        dtype_itemsize=dtype_itemsize,
        measured_time_s=measured_time_s,
        op_shape=op_shape,
    )

    if not result.actionable:
        # Can't compute roofline → fall back to ratio-based
        threshold = profile.finalize_threshold or 1.0
        return ratio < threshold

    # Roofline says skip → suppress ko escalation
    if result.efficiency_pct >= profile.roofline_skip_threshold_pct:
        return False

    # Roofline says optimize → proceed
    return True
