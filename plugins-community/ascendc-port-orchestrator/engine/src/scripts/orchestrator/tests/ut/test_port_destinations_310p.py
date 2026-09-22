# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""310P port-destination behavior pins.

310P is a DOWN-port destination (arch22 → arch20 / dav-m200, Ascend310P3).
These tests pin the destination-aware seams so the silent-wrong-chip class
(the DEBT-336 family: a digit-led ``310P_*`` env prefix no parser populates
→ lookup silently misses → wrong-chip ``A5_*`` fallback) cannot regress:

  1. prefix / destination map anti-drift (briefs and capability layers must
     import the SAME single map object)
  2. destination env resolution NEVER falls back to A5_* for 310p, and the
     a3-source / non-port behaviors stay byte-identical
  3. SoC version + limited-SoC gates are 310p-aware
  4. migration metadata (target_arch) is destination-derived — never a
     literal "arch35" stamped onto a 310P port
  5. the 310P KB spec page is mapped to its external cannbot-knowledge path
  6. bf16-bearing npubench sidecars are rejected at binding time (310P has
     no bf16 hardware; compiler-measured — see
     cannbot-knowledge target_ascend310p.md)
  7. the staged-reference endpoint gate is destination-aware: a 310p port
     with ASCEND310P_CONTAINER=local (or ASCEND310P_HOST) passes the npubench
     gate exactly like the a5 endpoint does
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import a5_target_capability as cap
import a3_ref_common
import finalize_pipeline  # noqa: F401 - initializes parent before dispatch re-export
import finalize_dispatch

# DEBT-47 hazard (bare-name collision): under some collection orders a
# conftest-imported module (kb_invoke) prepends <scripts> to sys.path, so the
# bare name ``orchestrator`` binds the PACKAGE (which re-exports nothing)
# instead of orchestrator.py, and the enforce tests below die on
# AttributeError when this file runs alone.  Undo a package binding and MOVE
# the module dir to the front of sys.path — a plain ``not in sys.path`` guard
# is useless here because the dir already sits DEEP in the list (behind
# <scripts>), so only reordering decides which entry wins.  Import-time only,
# so the parent conftest's per-test snapshots are unaffected; nothing in the
# tree depends on the bare package binding (tests/conftest.py DEBT-47 note).

_orch_dir = str(Path(__file__).resolve().parents[2])
_bound = sys.modules.get("orchestrator")
if _bound is not None and not str(getattr(_bound, "__file__", "")).endswith("orchestrator.py"):
    del sys.modules["orchestrator"]
if _orch_dir in sys.path:
    sys.path.remove(_orch_dir)
sys.path.insert(0, _orch_dir)

import orchestrator as orch  # noqa: E402
import phase_o5_runner  # noqa: E402
from a3_ref_common import PORT_DESTINATIONS
from briefs._common import TARGET_ENV_PREFIXES as BRIEF_PREFIXES
from briefs._common import load_env, target_env_prefix
from briefs.op_taxonomy import TARGET_HW_SPEC_MAP
from npubench.npubench_inputs import (
    NpubenchInputError,
    assert_sidecar_dtype_supported,
    validate_cli_npubench_args,
)

# These tests pin MODULE-PRIVATE seams of phase_o5_runner / finalize_dispatch
# (the resolvers are internal by design). Naming module._attr makes the
# checker's protected-access rule (G.CLS.11) fire per expression, so bind each
# once here via getattr — the same convention as test_stop_gate_dispatch.py.
# The bound objects resolve their module globals at call time, so behavior is
# identical to attribute access.
_port_destination_prefix = getattr(phase_o5_runner, "_port_destination_prefix")
_normalise_target = getattr(phase_o5_runner, "_normalise_target")
_a5_build_host = getattr(phase_o5_runner, "_a5_build_host")
_a5_build_container = getattr(phase_o5_runner, "_a5_build_container")
_a5_build_cann_path = getattr(phase_o5_runner, "_a5_build_cann_path")
_inject_migration_metadata = getattr(finalize_dispatch, "_inject_migration_metadata")


def _seed_port_workspace(tmp_path: Path) -> Path:
    """Seed a workspace the port_a3 plugin claims (opgen_mode marker only)."""
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"opgen_mode": "port_a3_to_a5"})
    )
    return tmp_path


# ---------------------------------------------------------------------------
# 1. map anti-drift
# ---------------------------------------------------------------------------
def test_prefix_map_anti_drift_between_copies():
    """briefs and the capability module must expose the SAME map object.

    Single source of truth is briefs/target_env_map.py (the map used to
    exist as two literal copies, which the code checker flagged as
    redundant code); both layers import it, so identity — not just equal
    values — is the property under test.
    """
    assert cap.TARGET_ENV_PREFIXES is BRIEF_PREFIXES
    assert dict(cap.TARGET_ENV_PREFIXES) == dict(BRIEF_PREFIXES)
    assert cap.TARGET_ENV_PREFIXES["310p"] == "ASCEND310P"


def test_target_env_prefix_310p_spellings():
    assert target_env_prefix("310p") == "ASCEND310P"
    assert target_env_prefix("310P") == "ASCEND310P"
    assert target_env_prefix("310p-ds") == "ASCEND310P"


def test_target_env_prefix_never_uppercases_310p():
    """The DEBT-336 core: 310p.upper() is the illegal digit-led 310P_* prefix."""
    assert target_env_prefix("310p") != "310P"


def test_target_env_prefix_rejects_unknown():
    with pytest.raises(ValueError):
        target_env_prefix("310b")


def test_port_destinations_contents():
    assert PORT_DESTINATIONS == frozenset({"a5", "310p"})
    # a3 is the port SOURCE, never a destination.
    assert "a3" not in PORT_DESTINATIONS


def test_port_destination_prefix_predicate():
    assert _port_destination_prefix("ASCEND310P")
    assert _port_destination_prefix("A5")
    assert not _port_destination_prefix("A3")
    assert not _port_destination_prefix("A2")


# ---------------------------------------------------------------------------
# 2. destination env resolution — never an A5_* fallback for 310p
# ---------------------------------------------------------------------------
def test_normalise_target_310p_prefix():
    assert _normalise_target({"TARGET": "310p"}) == (
        "310p", "ASCEND310P",
    )
    assert _normalise_target({"TARGET": "310P-DS"}) == (
        "310p", "ASCEND310P",
    )


def test_port_310p_reads_own_keys_never_a5(tmp_path):
    ws = _seed_port_workspace(tmp_path)
    env = {
        "A5_HOST": "a5-host",
        "A5_CONTAINER": "npu_dev3",
        "A5_CANN_PATH": "/a5/cann",
        "ASCEND310P_HOST": "310p-host",
        "ASCEND310P_CONTAINER": "local",
        "ASCEND310P_CANN_PATH": "/usr/local/Ascend/cann",
    }
    assert _a5_build_host(env, ws, "ASCEND310P") == "310p-host"
    assert (
        _a5_build_container(env, ws, "ASCEND310P") == "local"
    )
    assert (
        _a5_build_cann_path(env, ws, "ASCEND310P")
        == "/usr/local/Ascend/cann"
    )


def test_port_310p_unconfigured_fails_loud_not_a5(tmp_path):
    """A 310p destination with no ASCEND310P_* keys must resolve EMPTY.

    Falling back to the configured A5_* host here is the silent-wrong-chip
    run: the kernel builds/verifies on the wrong chip and the campaign
    reports a pass that never touched 310P hardware.
    """
    ws = _seed_port_workspace(tmp_path)
    env = {
        "A5_HOST": "a5-host",
        "A5_CONTAINER": "npu_dev3",
        "A5_CANN_PATH": "/a5/cann",
    }
    assert _a5_build_host(env, ws, "ASCEND310P") == ""
    assert _a5_build_container(env, ws, "ASCEND310P") == ""
    assert _a5_build_cann_path(env, ws, "ASCEND310P") == ""


def test_port_a5_destination_unchanged(tmp_path):
    ws = _seed_port_workspace(tmp_path)
    env = {
        "A5_HOST": "a5-host",
        "A5_CONTAINER": "npu_dev3",
        "A5_CANN_PATH": "/a5/cann",
    }
    assert _a5_build_host(env, ws, "A5") == "a5-host"
    assert _a5_build_container(env, ws, "A5") == "npu_dev3"
    assert _a5_build_cann_path(env, ws, "A5") == "/a5/cann"


def test_port_a3_source_keeps_split_host_precedence(tmp_path):
    """task#24-item2 legacy: TARGET=a3 agent builds on the A5 host — unchanged."""
    ws = _seed_port_workspace(tmp_path)
    env = {
        "A3_HOST": "a3-host",
        "A5_HOST": "a5-host",
        "A3_CANN_PATH": "/a3/cann",
        "A5_CANN_PATH": "/a5/cann",
    }
    assert _a5_build_host(env, ws, "A3") == "a5-host"
    assert _a5_build_cann_path(env, ws, "A3") == "/a5/cann"


def test_non_port_mode_legacy_unchanged(tmp_path):
    """Outside port_a3 the legacy `{target}_*`-first resolution is untouched."""
    env = {"A3_HOST": "a3-host", "A5_HOST": "a5-host"}
    assert _a5_build_host(env, tmp_path, "A3") == "a3-host"
    assert _a5_build_host(env, tmp_path, "A5") == "a5-host"


# ---------------------------------------------------------------------------
# enforce_port_a3_target — the guard and the resolvers must agree
# ---------------------------------------------------------------------------
def test_enforce_accepts_310p_destination():
    assert orch.enforce_port_a3_target("port_a3_to_a5", "310p") == ("310p", None)
    assert orch.enforce_port_a3_target("port_a3_to_a5", "310P") == ("310p", None)


def test_enforce_forces_a5_for_source_or_unknown():
    effective, warning = orch.enforce_port_a3_target("port_a3_to_a5", "a3")
    assert effective == "a5"
    assert warning
    effective, warning = orch.enforce_port_a3_target("port_a3_to_a5", "310b")
    assert effective == "a5"
    assert warning


# ---------------------------------------------------------------------------
# 3. SoC gates
# ---------------------------------------------------------------------------
def test_soc_version_310p_reads_own_prefix():
    env = {"ASCEND310P_SOC_VERSION": "Ascend310P3", "SOC_VERSION": "Ascend910"}
    assert cap.soc_version_for_target(env, "310p") == "Ascend310P3"


def test_soc_version_a5_generic_fallback_unchanged():
    env = {"SOC_VERSION": "Ascend950PR"}
    assert cap.soc_version_for_target(env, "a5") == "Ascend950PR"


def test_is_supported_310p_soc():
    for ok in ("Ascend310P3", "ascend310p", "310P3", "ASCEND310P"):
        assert cap.is_supported_310p_soc(ok), ok
    for bad in ("Ascend310B", "Ascend910", "Ascend950PR", "310b", "", None):
        assert not cap.is_supported_310p_soc(bad), bad


def test_is_limited_soc_target_aware():
    # For a 310p destination, anything outside the 310P family is limited.
    assert cap.is_limited_soc("Ascend310B", "310p")
    assert cap.is_limited_soc("Ascend910", "310p")
    assert not cap.is_limited_soc("Ascend310P3", "310p")
    # a5 keeps its legacy limited-SoC semantics (910 = codegen-smoke-only).
    assert cap.is_limited_soc("Ascend910", "a5")
    assert not cap.is_limited_soc("Ascend950PR", "a5")


def test_limited_target_validation_error_310p():
    message = cap.limited_target_validation_error("Ascend310B", "310p")
    assert "310P_SOC_UNSUPPORTED_FOR_VALIDATION" in message
    assert "Ascend310B" in message


# ---------------------------------------------------------------------------
# 4. destination-derived arch metadata
# ---------------------------------------------------------------------------
def test_arch_by_target_map():
    assert cap.ARCH_BY_TARGET == {"a5": "arch35", "310p": "arch20"}


def test_arch_for_target_spellings():
    assert cap.arch_for_target("a5") == "arch35"
    assert cap.arch_for_target("A5") == "arch35"
    assert cap.arch_for_target("310p") == "arch20"
    assert cap.arch_for_target("ASCEND310P") == "arch20"
    assert cap.arch_for_target("310p-ds") == "arch20"
    # Legacy/unknown spellings stay byte-identical with historical metadata.
    assert cap.arch_for_target(None) == "arch35"
    assert cap.arch_for_target("") == "arch35"
    assert cap.arch_for_target("a3") == "arch35"


def _write_state(workspace: Path, **extra) -> None:
    payload = {
        "opgen_mode": "port_a3_to_a5",
        "source_arch": "arch22",
        "target_arch": "arch35",
        "source_arch_detection": {},
    }
    payload.update(extra)
    (workspace / ".opgen_state.json").write_text(json.dumps(payload))


def test_migration_block_310p_uses_arch20(tmp_path):
    _write_state(tmp_path, target="310p", target_arch="arch20")
    (tmp_path / "verification.json").write_text(
        json.dumps({"precision": {"status": "PASS"}})
    )
    _inject_migration_metadata(tmp_path)
    migration = json.loads((tmp_path / "verification.json").read_text())[
        "migration"
    ]
    assert migration["source_arch"] == "arch22"
    assert migration["target_arch"] == "arch20"


def test_migration_block_a5_unchanged(tmp_path):
    _write_state(tmp_path, target="a5", target_arch="arch35")
    (tmp_path / "verification.json").write_text(json.dumps({}))
    _inject_migration_metadata(tmp_path)
    migration = json.loads((tmp_path / "verification.json").read_text())[
        "migration"
    ]
    assert migration["target_arch"] == "arch35"


def test_migration_block_stale_arch35_not_injected_for_310p(tmp_path):
    """A 310p destination with stale arch35 metadata must inject NOTHING.

    A wrong-chip migration block in the customer-visible verification.json is
    worse than a missing one; the gate fails closed instead.
    """
    _write_state(tmp_path, target="310p", target_arch="arch35")
    (tmp_path / "verification.json").write_text(json.dumps({}))
    _inject_migration_metadata(tmp_path)
    assert "migration" not in json.loads(
        (tmp_path / "verification.json").read_text()
    )


# ---------------------------------------------------------------------------
# 5. KB spec page mapped and present
# ---------------------------------------------------------------------------
def test_kb_spec_page_mapped_to_external_canonical_path():
    # 310p routes to its own card in cannbot-knowledge, same canonical layout
    # as a5/a3/a2. The in-plugin kb/okf/runbooks/hardware/ cards were removed by
    # the KB-externalization refactor, so the map must point at the external
    # canonical path — a silent fallback to the a5 page is forbidden (a wrong
    # page that loads cleanly is worse than a missing one).
    assert (
        TARGET_HW_SPEC_MAP.get("310p")
        == "knowledge/common/platforms/concepts/target_ascend310p.md"
    )
    assert "okf/runbooks" not in TARGET_HW_SPEC_MAP["310p"]


def test_archive_project_destination_scoped(tmp_path):
    """310p ports archive under their own project, not the a5 namespace."""
    env_file = tmp_path / "env_310p"
    env_file.write_text(
        "\n".join(
            [
                "TARGET=310p",
                "OPGEN_MODE=port_a3_to_a5",
                "ASCEND310P_HOST=localhost",
                "ASCEND310P_USER=npu_user",
                "ASCEND310P_CONTAINER=local",
                "ASCEND310P_CANN_PATH=/usr/local/Ascend/cann",
                "ASCEND310P_SOC_VERSION=Ascend310P3",
            ]
        )
        + "\n"
    )
    env = load_env(env_file)
    assert env.archive_project == "a3_to_310p_port"

    env_file_a5 = tmp_path / "env_a5"
    env_file_a5.write_text(
        "\n".join(
            [
                "TARGET=a5",
                "OPGEN_MODE=port_a3_to_a5",
                "A5_HOST=a5-host",
                "A5_USER=root",
                "A5_CONTAINER=npu_dev3",
                "A5_CANN_PATH=/a5/cann",
                "A5_SOC_VERSION=Ascend950PR",
            ]
        )
        + "\n"
    )
    env_a5 = load_env(env_file_a5)
    assert env_a5.archive_project == "a3_to_a5_port"


def test_archive_project_claim_yields_a5_namespace_only_for_a5():
    """The plugin claims the a5 archive project ONLY for a5 destinations.

    For 310p it yields None so the env-derived, destination-aware project in
    briefs._common decides (a3_to_310p_port) — claiming a3_to_a5_port for a
    310p port would archive it under a wrong-chip namespace.
    """
    from plugins import all_plugins

    port_plugin = next(p for p in all_plugins() if p.name == "port_a3_to_a5")

    class _Env:
        def __init__(self, target):
            self.target = target

    assert port_plugin.archive_project_claim(_Env("a5")) == "a3_to_a5_port"
    assert port_plugin.archive_project_claim(_Env("A5")) == "a3_to_a5_port"
    assert port_plugin.archive_project_claim(_Env("310p")) is None
    assert port_plugin.archive_project_claim(_Env("310P-DS")) is None


# ---------------------------------------------------------------------------
# 6. bf16 sidecar policy (no bf16 hardware on 310P)
# ---------------------------------------------------------------------------
def _sidecar_args(tmp_path: Path, sidecar_text: str):
    task = tmp_path / "task.py"
    task.write_text("import torch\n")
    (tmp_path / "task.json").write_text(sidecar_text)
    args = validate_cli_npubench_args(task, tmp_path)
    assert args is not None
    return args


def test_bf16_sidecar_rejected_at_binding(tmp_path):
    args = _sidecar_args(tmp_path, '{"inputs": [{"dtype": "bfloat16"}]}')
    with pytest.raises(NpubenchInputError) as excinfo:
        assert_sidecar_dtype_supported(
            args, frozenset({"bfloat16"}), "no bf16 hardware on Ascend310P"
        )
    assert "bfloat16" in str(excinfo.value)
    assert "no bf16 hardware" in str(excinfo.value)


def test_fp16_sidecar_passes(tmp_path):
    args = _sidecar_args(tmp_path, '{"inputs": [{"dtype": "float16"}]}')
    assert_sidecar_dtype_supported(
        args, frozenset({"bfloat16"}), "no bf16 hardware on Ascend310P"
    )  # must not raise


def test_empty_unsupported_set_is_noop(tmp_path):
    args = _sidecar_args(tmp_path, '{"inputs": [{"dtype": "bfloat16"}]}')
    assert_sidecar_dtype_supported(args, frozenset(), "unused")  # must not raise
