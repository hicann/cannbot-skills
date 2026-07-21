# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""phase_machine package — facade over four single-concern submodules.

Dependency direction (top depends on lower):
    guidance, phase_policy
        → validators
            → state_store

Public surface centers on a single per-task state record
(<task_dir>/.ar_state/state.json). Every piece of control state lives
in state.json, atomic write of state.json IS the transaction commit,
and cross-file consistency with the two durable artifacts (plan.md,
history.jsonl) is checked via state.expected_* fields.
"""

__all__ = (
    "Progress",
    "INIT", "BASELINE", "PLAN", "EDIT", "DIAGNOSE", "REPLAN", "FINISH",
    "ALL_PHASES",
    "STATE_FILE", "HISTORY_FILE", "PLAN_FILE", "PLAN_ITEMS_FILE",
    "EDIT_MARKER_FILE", "INTENT_FILE",
    "DIAGNOSE_ARTIFACT_TEMPLATE", "DIAGNOSE_MARKER_TEMPLATE",
    "DIAGNOSE_ATTEMPTS_CAP",
    "state_path", "state_record_path", "plan_path", "history_path",
    "edit_marker_path", "intent_path", "diagnose_artifact_path",
    "diagnose_marker",
    "write_intent", "read_intent", "clear_intent", "replay_intent",
    "load_state", "save_state", "state_transaction", "read_phase",
    "load_progress", "save_progress", "append_history", "update_progress",
    "get_task_dir", "set_task_dir", "clear_active_task", "touch_heartbeat",
    "find_active_task_dir", "current_session_task_dir",
    "check_state_consistency", "format_state_inconsistency",
    "require_state_consistency",
    "task_summary", "is_task_active", "task_owner_info",
    "validate_plan", "DiagnoseState", "diagnose_state",
    "DIAGNOSE_NEED_DIAGNOSIS", "DIAGNOSE_READY",
    "DIAGNOSE_MANUAL_FALLBACK", "get_plan_items", "parse_plan_text",
    "has_pending_items", "get_active_item", "is_settled_table_header",
    "_PLAN_ITEM_RE",
    "classify", "parse_script_names", "parse_invoked_ar_script",
    "is_single_foreground_ar_invocation", "check_bash", "check_edit",
    "get_guidance",
)

from .guidance import get_guidance
from .models import Progress
from .phase_policy import (
    check_bash,
    check_edit,
    classify,
    is_single_foreground_ar_invocation,
    parse_invoked_ar_script,
    parse_script_names,
)
from .state_store import (
    ALL_PHASES,
    BASELINE,
    DIAGNOSE,
    DIAGNOSE_ARTIFACT_TEMPLATE,
    DIAGNOSE_ATTEMPTS_CAP,
    DIAGNOSE_MARKER_TEMPLATE,
    EDIT,
    EDIT_MARKER_FILE,
    FINISH,
    HISTORY_FILE,
    # Phase constants
    INIT,
    INTENT_FILE,
    PLAN,
    PLAN_FILE,
    PLAN_ITEMS_FILE,
    REPLAN,
    # File constants
    STATE_FILE,
    append_history,
    # Cross-file consistency gate
    check_state_consistency,
    clear_active_task,
    clear_intent,
    current_session_task_dir,
    diagnose_artifact_path,
    diagnose_marker,
    edit_marker_path,
    find_active_task_dir,
    format_state_inconsistency,
    # Ownership (per-task owner field in state.json)
    get_task_dir,
    history_path,
    intent_path,
    is_task_active,
    load_progress,
    # State record I/O — single source of truth
    load_state,
    plan_path,
    read_intent,
    # Typed views over state.json
    read_phase,
    replay_intent,
    require_state_consistency,
    save_progress,
    save_state,
    set_task_dir,
    # Path builders
    state_path,
    state_record_path,
    state_transaction,
    task_owner_info,
    # Outward-facing facade — preferred over direct state.json reads in
    # batch / resume / dashboard / scaffold.
    task_summary,
    touch_heartbeat,
    update_progress,
    # Journal / write-ahead intent (closes the bodies-without-state crash window)
    write_intent,
)
from .validators import (
    _PLAN_ITEM_RE,
    DIAGNOSE_MANUAL_FALLBACK,
    DIAGNOSE_NEED_DIAGNOSIS,
    DIAGNOSE_READY,
    DiagnoseState,
    diagnose_state,
    get_active_item,
    get_plan_items,
    has_pending_items,
    is_settled_table_header,
    parse_plan_text,
    validate_plan,
)
