# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Single durable registry for migration reference providers.

``port_a3_to_a5`` is a generation mode, not a truth-provider choice.  Every
new workspace therefore persists a complete ``reference`` object and consumers
must resolve its ``source`` through this module.  In particular, a missing
reference is *not* inferred to mean live A3: operators must invoke an explicit
legacy migration before resuming such a workspace.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


A3_LIVE = "a3_live"
NPUBENCH = "npubench"
CANNBENCH = "cannbench"


@dataclass(frozen=True)
class ReferenceSourceSpec:
    """Provider properties that must not be re-inferred in each FSM phase."""

    source: str
    uses_a3_truth: bool
    lifecycle: str


REFERENCE_SOURCE_REGISTRY = MappingProxyType(
    {
        A3_LIVE: ReferenceSourceSpec(
            source=A3_LIVE, uses_a3_truth=True, lifecycle="live_capture"
        ),
        NPUBENCH: ReferenceSourceSpec(
            source=NPUBENCH, uses_a3_truth=False, lifecycle="staged_npubench"
        ),
        CANNBENCH: ReferenceSourceSpec(
            source=CANNBENCH, uses_a3_truth=False, lifecycle="unsupported"
        ),
    }
)
VALID_REFERENCE_SOURCES = frozenset(REFERENCE_SOURCE_REGISTRY)

# The immutable NPUKernelBench binding fields, grouped by what they pin down:
# provider identity, the bundle bytes, the task bytes, and the sidecar bytes.
# ``npubench_inputs`` writes the same field set; keeping the canonical grouping
# here lets that writer import it instead of repeating the flat name list.
NPUBENCH_BINDING_FIELD_GROUPS = (
    ("schema_version", "source", "semantic_binding", "runner_contract_version"),
    ("bundle_manifest_path", "bundle_manifest_sha256", "bundle_sha256"),
    ("task_relative_path", "task_sha256"),
    ("sidecar_relative_path", "sidecar_sha256", "sidecar_encoding"),
)
_NPUBENCH_REQUIRED_FIELDS = frozenset(
    field for group in NPUBENCH_BINDING_FIELD_GROUPS for field in group
)


class ReferenceSourceError(ValueError):
    """The durable reference discriminator is missing or unsafe to use."""


def resolve_reference_binding(workspace: Path | Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the explicit immutable provider binding or fail closed.

    It validates the immutable binding *shape* for every registered provider;
    provider-specific byte/content verification (for example an NPUKernelBench
    bundle manifest) remains with that provider before execution.  This keeps
    source-only and half-written state from selecting an accidental A3 path.
    """
    state = _load_state(workspace)
    if "reference" not in state or state.get("reference") is None:
        raise ReferenceSourceError(
            "durable state has no explicit reference binding; run the explicit "
            "a3_live migration before resume"
        )
    reference = state.get("reference")
    if not isinstance(reference, Mapping):
        raise ReferenceSourceError("durable state reference block is not an object")
    source = reference.get("source")
    if source not in VALID_REFERENCE_SOURCES:
        raise ReferenceSourceError(
            "durable state reference.source must be one of "
            f"{', '.join(sorted(VALID_REFERENCE_SOURCES))}; got {source!r}"
        )
    _validate_complete_binding(reference, str(source))
    return reference


def load_durable_state(workspace: Path | Mapping[str, Any]) -> Mapping[str, Any]:
    """Load a durable state through the provider's no-symlink boundary.

    A few lifecycle callers need ``opgen_mode`` before they can resolve a
    reference binding.  Exporting this narrow loader keeps those callers from
    reintroducing a direct ``Path.read_text()`` path that follows a
    task-created ``.opgen_state.json`` symlink.
    """
    return _load_state(workspace)


def resolve_reference_source(workspace: Path | Mapping[str, Any]) -> str:
    """Return a registered provider source from a complete explicit binding."""
    return str(resolve_reference_binding(workspace)["source"])


def uses_live_a3_reference(workspace: Path | Mapping[str, Any]) -> bool:
    """True only for the single provider that may read live A3 truth."""
    return resolve_reference_source(workspace) == A3_LIVE


def uses_npubench_reference(workspace: Path | Mapping[str, Any]) -> bool:
    """True only when state explicitly binds an original NPUKernelBench task."""
    return resolve_reference_source(workspace) == NPUBENCH


def uses_cannbench_reference(workspace: Path | Mapping[str, Any]) -> bool:
    """True only for the reserved CannBench provider interface."""
    return resolve_reference_source(workspace) == CANNBENCH


def explicit_a3_live_binding() -> dict[str, Any]:
    """Return the migration result for an operator-approved legacy workspace.

    This helper deliberately does not inspect or write a state file.  Callers
    must expose an explicit migration command and atomically persist the result;
    normal O2.5 resume never upgrades a missing binding on its own.
    """
    return {
        "schema_version": 3,
        "source": A3_LIVE,
        "semantic_binding": "a3_cann_live",
        "runner_contract_version": "a3_live/v1",
    }


def explicit_cannbench_binding() -> dict[str, Any]:
    """Return the complete reserved CannBench binding (no evaluator implied)."""
    return {
        "schema_version": 3,
        "source": CANNBENCH,
        "semantic_binding": "cannbench_reserved",
        "runner_contract_version": "cannbench/unimplemented",
    }


def migrate_legacy_a3_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Build, but do not persist, an explicit binding for a legacy workspace.

    A lifecycle command may call this helper after independently validating the
    workspace and atomically write the returned state.  Keeping it side-effect
    free makes accidental migration from the normal O2.5 resume path impossible.
    """
    existing = state.get("reference")
    if existing is not None:
        if not isinstance(existing, Mapping) or existing.get("source") != A3_LIVE:
            raise ReferenceSourceError(
                "legacy migration refuses a non-A3 or malformed reference binding"
            )
        if dict(existing) == explicit_a3_live_binding():
            raise ReferenceSourceError("state already has the canonical a3_live binding")
    migrated = dict(state)
    migrated["reference"] = explicit_a3_live_binding()
    return migrated


def _validate_complete_binding(reference: Mapping[str, Any], source: str) -> None:
    """Reject a source-only, half-written, or substituted provider binding."""
    if source == A3_LIVE:
        expected = explicit_a3_live_binding()
        if dict(reference) != expected:
            raise ReferenceSourceError(
                "a3_live reference binding is incomplete or differs from the explicit "
                "migration binding"
            )
        return
    if source == CANNBENCH:
        expected = explicit_cannbench_binding()
        # CannBench's O2.5 terminal status is intentionally append-only.
        allowed = {**expected}
        status = reference.get("provisioning_status")
        if status is not None:
            allowed["provisioning_status"] = status
        if dict(reference) != allowed:
            raise ReferenceSourceError(
                "cannbench reference binding is incomplete or has unsupported fields"
            )
        if status is not None and not _is_valid_cannbench_status(status):
            raise ReferenceSourceError("cannbench unsupported-provider status is malformed")
        return
    if source == NPUBENCH:
        _require_fields(reference, _NPUBENCH_REQUIRED_FIELDS, source)
        if reference.get("schema_version") != 3:
            raise ReferenceSourceError("npubench binding has unsupported schema")
        if reference.get("semantic_binding") != "npubench_old_format_task_bundle":
            raise ReferenceSourceError("npubench semantic binding mismatch")
        if reference.get("runner_contract_version") != "npubench/v1":
            raise ReferenceSourceError("npubench runner contract mismatch")
        return
    raise ReferenceSourceError(f"unhandled registered reference source: {source!r}")


def _is_valid_cannbench_status(status: Any) -> bool:
    """Return whether the append-only CannBench provisioning status is well formed."""
    if not isinstance(status, Mapping):
        return False
    return (
        status.get("verdict") == "UNSUPPORTED_REFERENCE_SOURCE"
        and status.get("source") == CANNBENCH
        and status.get("retryable") is False
    )


def _require_fields(
    reference: Mapping[str, Any], fields: frozenset[str], source: str
) -> None:
    missing = sorted(field for field in fields if reference.get(field) is None)
    if missing:
        raise ReferenceSourceError(
            f"{source} reference binding is incomplete; missing: {', '.join(missing)}"
        )


def _load_state(workspace: Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(workspace, Mapping):
        return workspace
    state_path = Path(workspace) / ".opgen_state.json"
    if state_path.is_symlink():
        raise ReferenceSourceError("durable state must be a regular non-symlink file")
    try:
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        state = json.loads(state_path.read_text())
    except FileNotFoundError as exc:
        raise ReferenceSourceError("durable state is missing") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceSourceError(f"durable state is unreadable: {exc}") from exc
    if not isinstance(state, Mapping):
        raise ReferenceSourceError("durable state is not a JSON object")
    return state
