# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Common brief sections used by all agent briefs.

Provides templated chunks for: env config (lane env, A5 host, container, CANN
paths), hard floors (current verification.json baseline), exit handoff
options (per agent_type from YAML), schema canonical contract, KB manifest
load list (from op_taxonomy).

Goal: 90% deterministic Python templating + 10% optional LLM-synth for
op-specific deep dives. Per codex C1 we target 100% Python lookup via
op_taxonomy.

Usage:
    from briefs._common import (
        load_env, env_block, hard_floors_block, schema_contract_block,
        kb_manifest_block, exit_handoff_block,
    )
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from briefs.brief_kb import (  # re-export: KB cluster moved to brief_kb (behavior-neutral, 2026-07-05)
    KBEntry, kb_inject_filtered, _target_for_opgen_mode, _parse_kb_index_rows,
    _detect_forced_architecture, kb_manifest_block, _FORCED_ARCH_TAGS,
    # cannbot c>b>a blocks (re-exported so `from briefs._common import ...` stays stable
    # for the c-tier read-wiring test + any cannbot caller after the 2026-07-05 decomposition).
    _cba_tier_a_routes_block, _c_tier_lessons_block,
    # Keep the local OKF-default contract stable; remote resyncs must not
    # silently replace it with the legacy KB manifest format.
    _okf_reference_block, _okf_enabled, _kb_discipline_scaffold,
)


_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent.parent.parent  # → repo root
DEFAULT_ASCENDC_ENV = _PROJECT_ROOT / "workspace" / ".ascendc_env"


# ---------------------------------------------------------------------------
# Env loading
# ---------------------------------------------------------------------------
@dataclass
class AscendCEnv:
    target: str                # "a5" | "a3" | "a2"
    host: str
    user: str
    password: str
    container: str
    cann_path: str
    soc_version: str
    benchmark_root: str
    local_benchmark: str
    local_project: str
    archive_project: str       # mode-owned output project (migration/backward)
    build_archive_enabled: bool
    # Customer mode is seeded by the scoped CLI and then loaded from durable
    # state. Empty means "not resolved yet", never an implicit third mode.
    opgen_mode: str = ""
    # When opgen_mode=="port_a3_to_a5", path to the ops-nn op directory
    # (e.g. ~/workspace/cann/ops-nn/loss/ctc_loss_v3). Empty otherwise.
    port_a3_source: str = ""
    # Per-target A3 reference-run config (used by phase_o25_a3_ref when
    # active target is a5 and reference baseline is A3-CANN).
    a3_host: str = ""
    a3_user: str = ""
    a3_container: str = ""
    a3_cann_path: str = ""
    a3_soc_version: str = ""
    a3_workspace: str = ""
    # task#24-item2 (2026-06-01): dedicated A5 build-target config for port_a3
    # mode (A3-source → A5-build, split-host). Read ONLY from A5_* keys — never
    # the generic `host`/`cann_path` resolution — so an agent whose primary
    # TARGET=a3 can still point port_a3 at a separate A5 build host without
    # flipping TARGET. Empty when A5_* not configured → no behavior change for
    # callers that do not configure an A5 host.
    a5_host: str = ""
    a5_container: str = ""
    a5_cann_path: str = ""
    a5_soc_version: str = ""
    # P129 (2026-05-17): host-side slice paths for A3 multi-agent isolation.
    # See feedback_a3_per_agent_sliced_container_convention. Container path
    # `/home/REDACTED_USER` is bound to host path `${a3_host_home}` (e.g.
    # `/home/REDACTED_USER_opus`). orchestrator validates this mapping at port_a3
    # dispatch time via `docker inspect` against the live container.
    a3_host_home: str = ""
    a3_host_backup: str = ""
    a3_deploy_stage_host: str = ""
    a3_deploy_stage_container: str = ""
    # The customer workflow has one kernel programming model: AscendC.
    backend: str = "ascendc"
    # P0xxx xulei commit 12a1b71e (2026-05-13): per-subagent settings file.
    # Reads SUBAGENT_SETTINGS_<AGENT_TYPE_UPPER>=<path> from .ascendc_env.
    # Used by agent_dispatch._build_extra_args to pass --settings to claude CLI
    # per agent type. Empty dict means default settings.json applies.
    # xulei's original commit added the caller but forgot the storage + method;
    # added here 2026-05-15 to unblock cold-starts (a5 audit catch).
    subagent_settings: dict[str, str] = None

    def get_subagent_settings(self, agent_type: str) -> Optional[str]:
        """Return settings-file path for given agent_type (or None).

        agent_type is e.g. 'aog-kernel-worker' / 'aog-precision-probe'.
        .ascendc_env keys are uppercase + dash→underscore:
            SUBAGENT_SETTINGS_AOG_KERNEL_WORKER=/path/to/settings.json
        """
        if not self.subagent_settings:
            return None
        key = agent_type.upper().replace("-", "_")
        return self.subagent_settings.get(key)


def load_env(env_file: Optional[Path] = None) -> AscendCEnv:
    """Parse workspace/.ascendc_env (shell-style KEY=VALUE).

    DEBT-101: `env_file` defaults to None (NOT DEFAULT_ASCENDC_ENV) so that
    `tests.monkeypatch(_common.DEFAULT_ASCENDC_ENV, ...)` is honored by
    `load_env()` callers across the orchestrator (batch.py + multiple
    orchestrator.py call sites previously passed no arg). A function default
    `= DEFAULT_ASCENDC_ENV` would bind the module-level value at def time,
    making monkeypatches no-ops for default-arg callers. Resolving the
    default at call time fixes the test-env-bypass class of bug.

    Subprocess override: `ASCENDC_ENV_PATH` env var, if set, takes precedence
    over DEFAULT_ASCENDC_ENV. Lets subprocess-based tests (the only callers
    where Python-level monkeypatches don't carry across the fork boundary)
    point load_env at a tmp .ascendc_env without depending on the real
    workspace env's contents. No effect on default-arg callers when the
    var is unset.
    """
    if env_file is None:
        override = os.environ.get("ASCENDC_ENV_PATH")
        env_file = Path(override) if override else DEFAULT_ASCENDC_ENV
    if not env_file.exists():
        raise FileNotFoundError(f".ascendc_env not found at {env_file}")
    raw = env_file.read_text()
    kv: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        # Strip optional matching quotes
        if (val.startswith("'") and val.endswith("'")) or (
            val.startswith('"') and val.endswith('"')
        ):
            val = val[1:-1]
        kv[key] = val

    target = kv.get("TARGET", "a5")
    backend = resolve_backend_from_env(kv.get("BACKEND", "ascendc"), None)
    archive_project = {
        "port_a3_to_a5": "a3_to_a5_port",
        "backward": "backward_ops",
    }.get(kv.get("OPGEN_MODE", ""), "generated_ops")

    # Resolve per-target keys (TARGET=a3 → try A3_HOST, A3_SOC_VERSION, etc.)
    _tp = target.upper().replace("-", "_")  # a3 → A3, a3-ds → A3_DS
    _tp_base = _tp.split("_")[0]  # A3_DS → A3

    return _build_env_from_kv(kv, target, backend, archive_project, _tp_base)


def resolve_backend_from_env(backend: str, env) -> str:
    """Validate and return the sole supported kernel-authoring backend."""
    if backend != "ascendc":
        raise ValueError("only the AscendC backend is supported")
    return "ascendc"


def resolve_plugin_for_brief(
    env: "AscendCEnv",
    workspace: Optional[Path] = None,
    backend_override: Optional[str] = None,
):
    """Resolve a scoped plugin from persisted workspace state only."""
    if backend_override not in (None, "ascendc"):
        raise ValueError("only the AscendC backend is supported")
    from plugins import detect_plugin  # avoid import-time cycle
    if workspace is not None and workspace.exists():
        return detect_plugin(workspace)
    return None


def _resolve_target_first(kv: dict, target_key: str, generic_key: str,
                          fallback_key: str, target: str) -> str:
    """Resolve a target-specific config key with generic + fallback, TARGET-SPECIFIC WINS.

    Precedence (fixes the DEBT-227 config bug): the target-specific key
    (e.g. ``A3_SOC_VERSION``) takes precedence over the generic key
    (e.g. ``SOC_VERSION``). The generic key is only a fallback for when the
    target-specific one is absent.

    Historically ``soc_version`` resolved the GENERIC ``SOC_VERSION`` FIRST, so a
    stale/leftover generic ``SOC_VERSION`` (or ``CANN_PATH``) in ``.ascendc_env``
    would SILENTLY override the correct ``A3_SOC_VERSION`` — the customer-repro
    agent hit exactly this: TARGET=a3 but the build used the A5 SoC because a
    generic ``SOC_VERSION`` shadowed ``A3_SOC_VERSION`` with no diagnostic.

    When BOTH the target-specific and the generic key are present AND they
    DIFFER, emit a loud stderr WARNING (fail-loud on conflict) so the override
    is never silent — then return the target-specific value.

    ``fallback_key`` (e.g. ``A5_SOC_VERSION``) is the last resort, preserving the
    pre-existing default-to-A5 behavior for envs that only populate the A5 block.
    """
    target_val = kv.get(target_key)
    generic_val = kv.get(generic_key)
    if target_val and generic_val is not None and generic_val != target_val:
        print(
            f"[briefs] WARNING: both {target_key}={target_val!r} and generic "
            f"{generic_key}={generic_val!r} are set and DIFFER (TARGET={target}). "
            f"Using the target-specific {target_key} — the generic {generic_key} "
            f"is IGNORED (it used to silently override; DEBT-227). Remove the "
            f"generic {generic_key} from workspace/.ascendc_env to silence this.",
            file=sys.stderr,
        )
    return target_val or generic_val or kv.get(fallback_key, "")


def _build_env_from_kv(kv: dict, target: str, backend: str, archive_project: str, _tp_base: str):
    """Internal helper: construct AscendCEnv from parsed key-value dict."""
    env_mode = kv.get("OPGEN_MODE", "")
    if env_mode not in ("", "port_a3_to_a5", "backward"):
        raise ValueError(
            "OPGEN_MODE may only be port_a3_to_a5 or backward; "
            f"got {env_mode!r}"
        )
    return AscendCEnv(
        target=target,
        host=kv.get(f"{_tp_base}_HOST") or kv.get("A5_HOST", ""),
        user=kv.get(f"{_tp_base}_USER") or kv.get("A5_USER", "root"),
        password=kv.get(f"{_tp_base}_PASSWORD") or kv.get("A5_PASSWORD", ""),
        container=kv.get(f"{_tp_base}_CONTAINER") or kv.get("A5_CONTAINER", ""),
        # DEBT-227: TARGET-SPECIFIC wins over generic (was already correct for
        # cann_path; soc_version had the generic-first bug — now unified).
        cann_path=_resolve_target_first(
            kv, f"{_tp_base}_CANN_PATH", "CANN_PATH", "A5_CANN_PATH", target),
        soc_version=_resolve_target_first(
            kv, f"{_tp_base}_SOC_VERSION", "SOC_VERSION", "A5_SOC_VERSION", target),
        benchmark_root=kv.get("BENCHMARK_ROOT", ""),
        local_benchmark=kv.get("LOCAL_BENCHMARK", ""),
        local_project=kv.get("LOCAL_PROJECT", ""),
        archive_project=archive_project,
        build_archive_enabled=kv.get("BUILD_ARCHIVE_ENABLED", "0") == "1",
        # The scoped CLI and durable workspace state own the customer mode.
        # Environment loading must not invent a default generation route.
        opgen_mode=env_mode,
        port_a3_source=kv.get("PORT_A3_SOURCE", ""),
        a3_host=kv.get("A3_HOST", ""),
        a3_user=kv.get("A3_USER", "root"),
        a3_container=kv.get("A3_CONTAINER", ""),
        a3_cann_path=kv.get("A3_CANN_PATH", ""),
        a3_soc_version=kv.get("A3_SOC_VERSION", ""),
        a3_workspace=kv.get("A3_WORKSPACE", ""),
        a3_host_home=kv.get("A3_HOST_HOME", ""),
        a3_host_backup=kv.get("A3_HOST_BACKUP", ""),
        a3_deploy_stage_host=kv.get("A3_DEPLOY_STAGE_HOST", ""),
        a3_deploy_stage_container=kv.get("A3_DEPLOY_STAGE_CONTAINER", ""),
        # task#24-item2: explicit A5 build-target keys (port_a3 split-host).
        a5_host=kv.get("A5_HOST", ""),
        a5_container=kv.get("A5_CONTAINER", ""),
        a5_cann_path=kv.get("A5_CANN_PATH", ""),
        a5_soc_version=kv.get("A5_SOC_VERSION", ""),
        backend=backend,
        # P0xxx xulei (commit 12a1b71e 2026-05-13, completed 2026-05-15 by a5):
        # parse SUBAGENT_SETTINGS_<AGENT_TYPE_UPPER>=<path> keys into dict.
        # Empty dict if no SUBAGENT_SETTINGS_* keys present.
        subagent_settings={
            k[len("SUBAGENT_SETTINGS_"):]: v
            for k, v in kv.items() if k.startswith("SUBAGENT_SETTINGS_")
        } or None,
    )


# ---------------------------------------------------------------------------
# Brief sections (templated)
# ---------------------------------------------------------------------------
def env_block(
    env: AscendCEnv,
    *,
    lane: int,
    op: str,
    workspace: Path,
    deploy_stage_dir: Optional[str] = None,
    build_batch_id: Optional[str] = None,
) -> str:
    """Lane / NPU / target / paths block. Same fields the LLM-orchestrator
    used to write at the top of every brief, but deterministic.
    """
    if deploy_stage_dir is None:
        # Stable per-spawn dir
        import time
        deploy_stage_dir = f"op_{_op_slug(op)}_lane{lane}_{int(time.time())}"
    if build_batch_id is None:
        build_batch_id = f"op_{_op_slug(op)}"

    a3_lines = ""
    if env.target.startswith("a3") or env.a3_host_home or env.a3_deploy_stage_host:
        if env.a3_host_home:
            a3_lines += f"\nA3_HOST_HOME: {env.a3_host_home}"
        if env.a3_host_backup:
            a3_lines += f"\nA3_HOST_BACKUP: {env.a3_host_backup}"
        if env.a3_deploy_stage_host:
            a3_lines += f"\nA3_DEPLOY_STAGE_HOST: {env.a3_deploy_stage_host}"
        if env.a3_deploy_stage_container:
            a3_lines += f"\nA3_DEPLOY_STAGE_CONTAINER: {env.a3_deploy_stage_container}"

    return f"""OP: {op}
WORKSPACE: {workspace}
TARGET: {env.target}
SOC_VERSION: {env.soc_version}
PLATFORM_SIMT: {"true" if env.target == "a5" else "false"}
ARCHIVE_PROJECT: {env.archive_project}
LANE: {lane} (NPU {lane} — A5 has 3 NPUs total IDs 0/1/2)
A5_HOST: {env.host} (user={env.user}, container={env.container}, CANN={env.cann_path})
BENCHMARK_ROOT: {env.benchmark_root}
LOCAL_BENCHMARK: {env.local_benchmark}
DEPLOY_STAGE_DIR: {deploy_stage_dir}
BUILD_BATCH_ID: {build_batch_id}
ASCENDC_WORKSPACE: {workspace}
BUILD_ARCHIVE_ENABLED: {1 if env.build_archive_enabled else 0}{a3_lines}"""


def rollback_context_block(workspace: Path) -> str:
    """P0abe (2026-05-07): inject prior-rollback signature into next brief.

    When a finalize-rollback gate fires (P0abd coverage / P0aax KB / P0aba
    audit / P0ee pass-count / P0ff persist-without-evidence), the rollback
    reason is recorded in `.rollback_history.jsonl`. Without injecting it
    into the next worker's brief, kw-N+1 reads the same generic Phase D
    text kw-N already ignored → emits same shape → infinite loop until
    TOTAL_SPAWN_CAP_PER_OP fires.

    This block surfaces the LAST rollback signature with concrete remediation
    so the next worker spawn knows EXACTLY what kw-N got rejected for.

    Returns empty string if no rollback history (cold start path).
    """
    history = workspace / ".rollback_history.jsonl"
    if not history.exists():
        return ""
    try:
        lines = [
            json.loads(ln) for ln in history.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except (json.JSONDecodeError, OSError):
        return ""
    if not lines:
        return ""
    last = lines[-1]
    sig = last.get("signature", "?")
    repeats = sum(1 for e in lines if e.get("signature") == sig)
    gate = last.get("gate", "<unknown>")
    rollback_state = last.get("rollback_state", "<unknown>")
    reason = last.get("reason", "")
    repeat_hint = ""
    if repeats >= 2:
        repeat_hint = (
            f"\n**THIS IS YOUR {repeats}TH ATTEMPT WITH SAME SIGNATURE**. "
            f"The next finalize attempt that produces the same gap will route "
            f"to `await_user_decision` (loop-break, P0abe) instead of looping. "
            f"You must produce a DIFFERENT shape this spawn — re-read the "
            f"gate's remediation hint below carefully and act on it."
        )
    body = f"""# CRITICAL: Previous spawn rejected by finalize gate (P0abe)

The orchestrator's finalize gate REJECTED the prior worker's handoff and rolled
the FSM back to {rollback_state}. You are the next spawn; the same workspace
is loaded but the prior gate REASON is below. Do NOT emit the same shape.

**Gate that fired**: `{gate}`
**Signature** (normalized): `{sig}`
**Reason given to prior worker**:

{reason}
{repeat_hint}

## Concrete remediation (act on this BEFORE Phase D verify)

The reason text above contains the exact remediation. Common cases:
- **P0abd_pass_a_coverage** — your prior `precision.pass_a.total` was less
  than the benchmark `<op>.json` JSONL line count. Either:
  (a) re-run verification iterating ALL benchmark cases (line count =
      `wc -l workspace/{{op}}/{{op}}.json` — use Path-A `get_input_groups()`
      that loads `<op>.json` and yields one input group per JSONL line);
  (b) populate `precision.pass_a.skipped_cases` as a list of
      `{{case_idx, reason}}` entries (one per skipped case) with explicit
      hardware/CANN-rejection reason.
  Bare reduction of `total` is silent coverage fraud — gate will reject again.

- **P0aax_kb_missing / kb_thin** — write/expand `knowledge_update.md` per
  Phase E required structure (Findings, KB-promotable patterns with scope
  tags, Cited KB, Anti-patterns).

- **P0aba_audit** — `audit_self_critic_post_worker.md` /
  `.delegation_scan_passed` / explicit Pass-B + perf re-measure status
  missing. Orchestrator produces audit doc + scan automatically; ensure your
  verification.json sets `precision.pass_b.status` and
  `performance.independent_re_measure` to concrete values (or N/A with
  reason).

- **P0ee_pass_count** — claimed `precision.status=PASS` but
  `pass_a.tier1_pass < pass_a.total`. Either reconcile the counts or change
  status to PARTIAL_PERSIST + emit Tier-2 evidence per OL-109.

If the gate's reason text doesn't match the patterns above, treat the
verbatim reason as your authoritative remediation hint.
"""
    return body


def _no_edge_cases(workspace: Path) -> bool:
    """Check if --no-edge-cases flag is active for this op."""
    return (workspace / ".no_edge_cases").exists()


def hard_floors_block(workspace: Path) -> str:
    """Read current verification.json, emit "preserve these floors" block.

    For cold-start ops (no verification.json), emits the goal floor instead.
    When .no_edge_cases marker exists, Pass B is marked N/A.
    """
    no_ec = _no_edge_cases(workspace)
    pass_b_line = ("- Pass B: N/A (edge cases disabled)"
                   if no_ec
                   else "- Pass B (edge_dataset): bit-exact vs CPU truth")

    vj_path = workspace / "verification.json"
    if not vj_path.exists():
        return f"""# Hard gate floors (cold start — no prior baseline)

GOAL:
- Pass A: 50/50 OVERALL_T1 (or OVERALL_T2 with documented evidence per OL-109)
{pass_b_line}
- Det: 50/50 (DET_POLICY=required satisfied)
- Perf: ≥ 0.6× CANN ratio (V3.8.4 escalation rule)"""

    try:
        vj = json.loads(vj_path.read_text())
    except Exception:
        return "# Hard floors: verification.json malformed; cold-start defaults apply."

    prec = vj.get("precision", {}) or {}
    perf = vj.get("performance", {}) or {}
    pa = prec.get("pass_a", {}) or {}
    pb = prec.get("pass_b", {}) or {}
    det = vj.get("determinism", {}) or {}

    ratio = (perf.get("ratio")
             or perf.get("ratio_mean")
             or perf.get("overall_speedup")
             or perf.get("sum_ratio")
             or perf.get("ratio_median")
             or perf.get("median_ratio"))

    pa_pass = pa.get("tier1_pass", pa.get("n_pass", "?"))
    pa_total = pa.get("total", pa.get("n_total", "?"))
    if no_ec:
        pb_pass = "N/A"
        pb_total = "N/A"
        pb_status = "SKIPPED"
    else:
        pb_pass = pb.get("tier1_pass", pb.get("n_pass", "?"))
        pb_total = pb.get("total", pb.get("n_total", "?"))
        pb_status = pb.get("status", "?")
    det_id = det.get("n_identical_cases", det.get("n_identical", "?"))
    det_total = det.get("n_cases_checked", det.get("n_total", "?"))

    return f"""# Hard gate floors (DO NOT REGRESS — preserve byte-identical or beat)

- Pass A: {pa_pass}/{pa_total} {pa.get("status", "?")} (kw baseline preserved)
- Pass B: {pb_pass}/{pb_total} {pb_status}
- Det: {det_id}/{det_total} policy_satisfied={det.get("policy_satisfied", "?")}
- Perf: ratio={ratio} (V3.8.4: < 0.6× → ko escalation)

Top-level precision.status: {prec.get("status", "?")} (V3.8.3 fallback applies if absent)"""


def env_quirks_block(target: str) -> str:
    """P0aas (2026-05-07): per-TARGET infra-quirks preamble for kw/ko/pp/ar briefs.

    Reads `src/scripts/env_quirks_<target>.json` (DS-shipped, msg
    DISCORD_ID_REDACTED). Returns the file's `preamble_text` formatted as a
    brief section, so the worker frontloads target-environment-specific
    infrastructure quirks instead of rediscovering them each spawn (DS
    finding 2026-05-07: workers spent 70%+ wall-clock on infra
    rediscovery — sandbox/LOCAL_TASK/SSH-permission patterns).

    Empty string when no JSON exists for the target — the function is
    a no-op for targets without a quirks catalog (e.g. `a5` currently).
    """
    if not target:
        return ""
    # Resolve repo root from this file's location: briefs/ → orchestrator/ → scripts/ → root
    repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    # File-naming convention: env_quirks_<target>.json with target normalized to
    # underscores (e.g. TARGET=a3-ds → env_quirks_a3_ds.json). Try both forms.
    target_norm = target.replace("-", "_")
    quirks_file = repo_root / "src" / "scripts" / f"env_quirks_{target_norm}.json"
    if not quirks_file.exists():
        # Fall back to literal target name (legacy)
        quirks_file = repo_root / "src" / "scripts" / f"env_quirks_{target}.json"
        if not quirks_file.exists():
            return ""
    try:
        data = json.loads(quirks_file.read_text())
    except (json.JSONDecodeError, OSError):
        return ""
    preamble = data.get("preamble_text", "").strip()
    if not preamble:
        return ""
    quirks = data.get("quirks", [])
    quirks_summary = ""
    if quirks:
        lines = []
        for q in quirks:
            cat = q.get("category", "?")
            rule = q.get("rule", "").strip()
            if rule:
                lines.append(f"  - [{cat}] {rule}")
                # Surface the most actionable workaround field if present
                wa = q.get("workaround") or q.get("export_cmd") or q.get("wrapper_usage") or ""
                if wa:
                    lines.append(f"    → {wa}")
        if lines:
            quirks_summary = "\n\n## Quirks (full catalog)\n" + "\n".join(lines)
    return f"""# ENV QUIRKS — {target}

{preamble}{quirks_summary}

(See `src/scripts/env_quirks_{target}.json` for the canonical structured catalog.)"""


# ---------------------------------------------------------------------------
# NODE-21 Phase C (2026-05-28): tag-aware KB injection for brief construction.
# ---------------------------------------------------------------------------


# Forced-architecture markers in op_classification.json. The architecture
# (SIMT / SIMD) is FIXED at classification time and is carried through
# forced-SIMT classification before `orch --port`); kw must HONOR it and
# implement only — it must NOT run the SIMT_VS_SIMD decision tree itself nor
# override the forced choice during Phase A authoring. Architecture-change is
# a ko-stage (optimizer) decision, AFTER precision passes, for performance.
#
# Lives here (the lower-level module) — NOT in kw_brief — because both
# `kb_manifest_block` (this module) and `kw_brief._forced_architecture_block`
# need it. Defining it in kw_brief and importing it back here would form a
# circular import (_common↔kw_brief), which caused an order-dependent
# ImportError under combined pytest collection (review 2026-06-16). The
# dependency edge is one-way: kw_brief → _common.
#
# Marker robustness for pre-injected classification:
#   1. explicit `force_simt: true` / `force_simd: true` boolean key (preferred)
#   2. explicit `forced_arch: "SIMT" | "SIMD"` string key
#   3. forced-arch convention: a bare "SIMT" / "SIMD" tag in op_class_tags
#      (the merge preserves an explicit "SIMT" classification tag)


def fixed_layout_block() -> str:
    """PB-33 (2026-05-14): canonical complete-NPU-op layout contract.

    Every agent that produces or modifies op output (kw / ko / fo / probe-fix
    iteration / researcher-directive-emit) MUST conform to this layout.
    finalize_pipeline._check_port_a3_op_host_completeness() enforces; agents
    that ship incomplete op_host/ get ROLLBACK.

    This is the SAME layout for the two scoped modes (port_a3_to_a5 and
    backward) — the input source differs but
    the OUTPUT shape is invariant. A complete AscendC NPU op IS the full
    PR4778 mirror. Patches and snippets are review-aid, not ship artifacts.
    """
    return """# FIXED OUTPUT LAYOUT CONTRACT (PB-33, 2026-05-14)

Every produced AscendC op MUST ship a COMPLETE PR4778 mirror in
`workspace/{op}/`. Patches alone are insufficient — they lose context once
detached from the master they were generated against, and downstream
consumers (CANN team, reviewers, finalize pipeline) cannot apply them
without the original anchor.

Required structure (workspace-relative paths):

  op_host/
    <op>_def.cpp           # MANDATORY — complete file, NOT patch
    <op>_tiling.cpp        # MANDATORY — host tiling implementation
    <op>_tiling.h          # MANDATORY — host tiling header
    CMakeLists.txt         # MANDATORY — build registration
    config/
      ascend950/
        <op>_binary.json   # MANDATORY — A5 binary registration
        <op>_simplified_key.ini  # MANDATORY — A5 simplified-key map
    # Optional (only if upstream has them):
    <op>_infershape.cpp    # if op has bespoke shape-inference
    op_api/
      <op>.cpp             # if op needs aclnn L0 wrapper
      <op>.h
    # Optional (review-aid, NOT ship artifact):
    <op>_def.cpp.patch     # diff vs upstream — for review trail only

  op_kernel/
    arch35/
      <op>.h               # MANDATORY — A5 kernel implementation
      <op>.cpp             # if upstream has cpp dispatcher in arch35/
      <sibling_variants>.h # if multi-variant (per OL-153 L2+L3 hybrid)
    <op>_apt.cpp           # MANDATORY — A5 entry-point wrapper

Hard rules:
- For port_a3_to_a5 mode: independently author the required op_host files
  from the detector-admitted, top-level arch22 host interfaces and repository
  templates, plus the config/ascend950 additions. Never recursively mirror
  `<port_source>/op_host/`; target subdirectories are not inputs.
  finalize_pipeline checks count ≥ 3 non-config / non-patch files.
- For backward mode: the same layout is the target. If the forward spec doesn't
  provide op_host scaffolding, GENERATE the minimum required set:
  <op>_def.cpp + <op>_tiling.{cpp,h} + CMakeLists.txt + binary.json +
  simplified_key.ini per PR4778 template.
- Patches CAN be produced as review-aids in addition to complete files,
  but NEVER as a substitute. finalize_pipeline.OP_HOST_COMPLETENESS gate
  rolls back ops that ship patches instead of complete files.
- Cross-agent consistency: kw writes the initial layout. ko / fo / probe
  iter editors MUST preserve every file kw produced; deletions of any
  layout file is REJECT.
"""


def schema_contract_block() -> str:
    """Canonical schema contract — every brief should remind the worker."""
    return """# OUTPUT SCHEMA CONTRACT (V3.8.x canonical — DO NOT improvise)

verification.json:
- top-level `precision.status` ∈ {PASS, PASS_WITHIN_TOLERANCE, PARTIAL, FAIL}
  RECOMMENDED. If absent, V3.8.3 fallback derives it: pass_a ∈ {PASS, N/A} +
  pass_b == PASS → PASS (Path A / OL-68 case A pattern)
- per-pass: `precision.pass_a.{status, tier1_pass, total}` +
  `precision.pass_b.{status, tier1_pass, total}` (legacy `n_pass`/`n_total`
  also accepted via fallback but `tier1_pass`/`total` is canonical)
- top-level `performance.ratio` (NOT `overall_speedup` — schema_norm aliases
  but canonical is `ratio`)
- `determinism.{policy_satisfied, n_identical_cases, n_cases_checked}`
  (legacy `n_identical`/`n_total` also accepted via fallback)

state_transitions.jsonl is OWNED BY THE ORCHESTRATOR — you do NOT write to it.
The orchestrator records your handoff line + computes the routing transition.
If you OBSERVE entries (e.g. read prior workspace state), the canonical schema is:
- keys: `from_state`, `to_state`, `handoff`, `matched_transition_index`,
  `rationale`, `iter_counts_snapshot`, `ts`
- valid `to_state` values: await_worker / await_optimizer / await_probe /
  await_fused_optimizer / await_researcher / await_det_analyzer / finalize / abort

Your handoff line in stdout (and in PROGRESS.md tail) MUST be one of the
EXIT HANDOFF OPTIONS listed in your agent's brief. Do NOT improvise free-form
state names like `done` / `partial_persist` / `await_orchestrator` — those
are not YAML states; the orchestrator's schema_norm will REJECT or DROP them."""


def self_introspection_block() -> str:
    """P0qq (2026-05-06): IN-CONTEXT self-introspection checkpoint.

    User correction (2026-05-06): the previous aog-self-critic skill was
    spawned as a SEPARATE process — fresh model, no access to the agent's
    reasoning trace. That's "post-hoc audit" not introspection. Real
    self-critic = same model in same context reflecting on its own work
    BEFORE emitting decisions.

    This block goes into every agent brief. Agents must execute it AS
    PART OF their own reasoning, not as a separate spawn. Output goes
    to PROGRESS.md `## Self-introspection` section. Schema validator
    rejects `done` / `PARTIAL_PERSIST` handoffs that lack this section.
    """
    return """# SELF-INTROSPECTION CHECKPOINT (mandatory before terminal handoff)

Before emitting `→ orchestrator: done` OR `→ orchestrator: PARTIAL_PERSIST`,
you MUST write a `## Self-introspection` section in PROGRESS.md tail
with at minimum these subsections:

```
[[H2]] Self-introspection ({{your-slug}})

### Pressure modes I felt
List which P1-P8 from ANTI_PRESSURE_PROTOCOLS.md you experienced during
this spawn. Be honest — naming them in-context is the actual mechanism;
hiding them is P5 + P7 in costume. Format:
  - **Px (Name)**: <one sentence on when/where it pulled at you>

### Decisions I almost rationalized
For each pressure, name a decision you almost made under it that would
have violated the rule. Format:
  - Almost: <bad action>. Rule that stopped me: <Px counter-instruction>

### Verifications I might have skipped
List concrete steps I ran that the urge said to skip:
  - <verification step>: ran (or: skipped because <legitimate reason>)

### Confidence calibration
Rate my own confidence in this handoff being correct, on:
  - precision: HIGH | MEDIUM | LOW (cite Tier-2 evidence file or
    reasoning if MEDIUM/LOW)
  - perf: HIGH | MEDIUM | LOW
  - architectural fit: HIGH | MEDIUM | LOW
LOW on any axis without a probe / researcher / KB cite is a smell —
state it explicitly so reviewer sees the gap.
```

This is NOT a separate spawn. NOT a tool call. Write it in the SAME
context where you've been reasoning about this op. It's reflection on
your own trace.

If you skip this section before a terminal handoff, schema_norm
rejects the handoff (P0qq) and routes you back to await_<state>.
""".replace("[[H2]]", "##")


def safety_block(env: AscendCEnv) -> str:
    return """# SAFETY (CLAUDE.md rules)

- NO torch_npu / aclnn delegation in kernel — pure AscendC primitives only
- NO CANN op_impl source reading (source-isolation rule)
- After each iter: independent perf re-measure per CLAUDE.md "Independent
  Performance Verification" rule
- Build via deploy_to_npu_lane.sh (sets BUILD_BATCH_ID)
- Lane release best-effort at exit (wrap in try/except; if lane_release.py
  not present, that's OK — orchestrator handles release on its side)
- NPU IDs available: 0, 1, 2 only (lane 3 = phantom; A5 has 3 real NPUs).
  If your script tries to use ASCEND_VISIBLE_DEVICES≥3, that's a bug."""


def _op_slug(op: str) -> str:
    """Convert workspace dir name to G7-compliant slug.

    G7 regex: ^[a-z][a-z0-9_]*-(kw|pp|ko|fo|ar|da|bs|td|tt|cl)-\\d+\\b
    The slug is the leading portion (before -kw-1 etc.).
    Strip numeric prefix; lowercase; preserve underscores.
    """
    s = op.lower()
    # Strip leading "<digits>_"
    s = re.sub(r"^\d+_", "", s)
    # Replace remaining hyphens with underscore (G7 allows _ but not -)
    s = s.replace("-", "_")
    # Drop chars not in [a-z0-9_]
    s = re.sub(r"[^a-z0-9_]", "", s)
    # Must start with letter
    if not s or not s[0].isalpha():
        s = "op_" + s
    return s


def g7_slug(op: str, agent_type: str, spawn_index: int) -> str:
    """Build G7-compliant agent name slug: {op_slug}-{code}-{N}.

    Keep the codes in lockstep with `workflow_critic._AGENT_NAME_PATTERN` so
    post-hoc audits don't reject spawns.
    """
    code = {
        "aog-kernel-worker": "kw",
        "aog-precision-probe": "pp",
        "aog-kernel-optimizer": "ko",
        "aog-fused-optimizer": "fo",
        "aog-researcher": "ar",
        "aog-determinism-analyzer": "da",
        "aog-cann-learner": "cl",
    }.get(agent_type)
    if code is None:
        raise ValueError(f"unknown agent_type for G7 slug: {agent_type}")
    return f"{_op_slug(op)}-{code}-{spawn_index}"
