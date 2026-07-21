#!/usr/bin/env bash
# loop-subagent-stop.sh — SubagentStop hook.
#
# Triggered every time a subagent (spawned via Task tool) terminates.
# Validates that evolution partial subagents (lingxi-partial / ops-partial)
# actually executed the work they claim, by auditing their transcript JSONL.
#
# Routing by agent_type (from stdin JSON):
#   lingxi-partial / ops-partial   →  strict audit (S1, S6)
#   *                              →  allow (no enforcement)
#
# Audit relies on `transcript_audit.py` which inspects the subagent's full
# tool-call history (cannot be lied about — Claude Code's own log).
#
# Escape hatch: LINGXI_LOOP_HOOK_DISABLE=1 downgrades all blocks to warn-only.

set -uo pipefail

HOOK_NAME="loop-subagent-stop"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

read_stdin_json

# Extract identifying fields from SubagentStop stdin JSON
agent_type="$(json_get "agent_type")"
agent_transcript="$(json_get "agent_transcript_path")"
last_msg="$(json_get "last_assistant_message")"
cwd_field="$(json_get "cwd")"
stop_hook_active="$(json_get "stop_hook_active")"

# Defensive: stop_hook_active=true means we already blocked once on the same
# subagent; allow on second visit to avoid infinite loops.
if [[ "${stop_hook_active}" == "True" || "${stop_hook_active}" == "true" ]]; then
    hook_allow
fi

# Resolve agent_type via meta.json if stdin's field is empty.
# Each subagent transcript has a sibling <transcript>.meta.json with authoritative agentType.
if [[ -z "${agent_type}" && -n "${agent_transcript}" ]]; then
    meta_path="${agent_transcript%.jsonl}.meta.json"
    if [[ -f "${meta_path}" ]]; then
        agent_type="$(python3 -c "
import json, sys
try:
    m = json.load(open('${meta_path}'))
    print(m.get('agentType', ''))
except Exception:
    pass
" 2>/dev/null)"
    fi
fi

# Route: only enforce on evolution partials. Other subagent types pass through.
case "${agent_type}" in
    lingxi-partial|ops-partial)
        : # fall through to strict audit
        ;;
    *)
        hook_allow
        ;;
esac

# Transcript missing → can't audit; fail open (main Stop hook + filesystem
# evidence will catch downstream problems via existing R-rules).
if [[ -z "${agent_transcript}" ]] || [[ ! -f "${agent_transcript}" ]]; then
    hook_log "transcript missing/empty for ${agent_type} subagent; skipping audit"
    hook_allow
fi

# Run the transcript audit. transcript_audit.py prints violations to stderr
# (one per line) and exits 1 if any.
audit_py="${PROJECT_ROOT}/.claude/skills/evolution-world-model/scripts/transcript_audit.py"
if [[ ! -f "${audit_py}" ]]; then
    hook_log "transcript_audit.py not found at ${audit_py}; skipping audit"
    hook_allow
fi

audit_output="$(python3 "${audit_py}" audit-partial \
    --transcript "${agent_transcript}" \
    --partial-type "${agent_type}" \
    --last-msg "${last_msg:-}" \
    --cwd "${cwd_field:-${PWD}}" 2>&1 1>/dev/null)"
audit_rc=$?

# ── v0.5: Write subagent exit status marker and update parent state ──
# Record the exit outcome BEFORE blocking so _infer_state_from_filesystem
# can distinguish crashed/killed subagents from still-running ones.
_write_subagent_exit_marker() {
    local transcript="$1"
    local agent_type="$2"
    local cwd="$3"
    local audit_passed="$4"   # "true" or "false"

    # Determine the target parallel dir from eval --output paths.
    # transcript_audit.py list-calls extracts all Bash invocations.
    local audit_script="${PROJECT_ROOT}/.claude/skills/evolution-world-model/scripts/transcript_audit.py"
    local output_paths
    output_paths="$(python3 "${audit_script}" list-calls \
        --transcript "${transcript}" --tool Bash 2>/dev/null \
        | grep -oP -- '--output[=\s]+\K\S+' \
        | head -5)"  # usually only 1-2 invocations

    # Determine exit status based on evidence
    local exit_status="unknown"
    local eval_exists="false"

    # Check if any of the claimed --output paths actually exist on disk
    local found_parallel_dir=""
    while IFS= read -r op; do
        [[ -z "${op}" ]] && continue
        op="${op//\"/}"   # strip quotes
        op="${op//\'/}"
        # Resolve relative paths
        local abs_op="${op}"
        [[ "${op:0:1}" != "/" ]] && abs_op="${cwd}/${op}"
        local parent_dir
        parent_dir="$(dirname "${abs_op}")"
        if [[ -d "${parent_dir}" ]]; then
            found_parallel_dir="${parent_dir}"
            if [[ -f "${abs_op}" ]] && [[ -s "${abs_op}" ]]; then
                eval_exists="true"
            fi
            break
        fi
    done <<< "${output_paths}"

    # Determine exit_status from audit result + filesystem evidence
    if [[ "${audit_passed}" == "true" ]] && [[ "${eval_exists}" == "true" ]]; then
        exit_status="completed"
    elif [[ "${audit_passed}" == "true" ]] && [[ "${eval_exists}" == "false" ]]; then
        # Audit passed (script was invoked) but no results — likely killed/timeout
        exit_status="timeout"
    else
        exit_status="error"
    fi

    # Map to valid state_ops partial_status value
    local exit_status_mapped="running"
    case "${exit_status}" in
        completed) exit_status_mapped="completed" ;;
        timeout|error|killed) exit_status_mapped="failed" ;;
    esac

    # Write marker file into the parallel directory
    if [[ -n "${found_parallel_dir}" ]]; then
        local marker_file="${found_parallel_dir}/.subagent_exit_status"
        python3 -c "
import json
from datetime import datetime, timezone, timedelta
tz = timezone(timedelta(hours=8))
now = datetime.now(tz).strftime('%Y-%m-%dT%H:%M:%S%z')
marker = {
    'exit_status': '${exit_status}',
    'timestamp': now,
    'agent_type': '${agent_type}',
    'audit_passed': True,
    'evaluation_results_exist': False,
}
if '${audit_passed}' in ('true','false'):
    marker['audit_passed'] = True if '${audit_passed}' == 'true' else False
if '${eval_exists}' in ('true','false'):
    marker['evaluation_results_exist'] = True if '${eval_exists}' == 'true' else False
marker['parallel_dir'] = '${found_parallel_dir}'
with open('${marker_file}', 'w', encoding='utf-8') as f:
    json.dump(marker, f, ensure_ascii=False, indent=2)
" 2>/dev/null
        hook_log "wrote subagent exit marker: exit_status=${exit_status} marker=${marker_file}"

        # Update parent state.json partial_status via session anchor
        local anchor_pattern="${PROJECT_ROOT}/output/.ops-evo_current_session_*.json"
        local state_ops="${PROJECT_ROOT}/.claude/skills/evolution-world-model/scripts/state_ops.py"
        for af in ${anchor_pattern}; do
            [[ -f "${af}" ]] || continue
            local evo_dir
            evo_dir="$(python3 -c "
import json, os
try:
    a = json.load(open('${af}'))
    d = a.get('evo_dir', '')
    if d and os.path.isdir(d) and os.path.isfile(os.path.join(d, 'state.json')):
        print(d)
except Exception:
    pass
" 2>/dev/null)"
            [[ -z "${evo_dir}" ]] && continue

            # Verify this parallel dir actually belongs to this evo_dir
            # (prevents cross-contamination when multiple ops are running)
            [[ "${found_parallel_dir}" == "${evo_dir}"/* ]] || continue

            # Extract parallel_idx from found_parallel_dir (e.g., .../round_2/parallel_0)
            local parallel_idx
            parallel_idx="$(basename "${found_parallel_dir}")"
            parallel_idx="${parallel_idx#parallel_}"

            if [[ -f "${state_ops}" ]] && [[ -n "${parallel_idx}" ]]; then
                python3 "${state_ops}" write-partial \
                    --evo-dir "${evo_dir}" \
                    --parallel-idx "${parallel_idx}" \
                    --status "${exit_status_mapped}" 2>/dev/null && \
                    hook_log "updated state.json partial[${parallel_idx}] = ${exit_status_mapped} (via anchor ${af})"
            fi
            break  # only update one matching evo_dir
        done
    fi
}

# Determine if audit passed
audit_passed_flag="true"
if [[ ${audit_rc} -ne 0 ]]; then
    audit_passed_flag="false"
fi

_write_subagent_exit_marker \
    "${agent_transcript}" "${agent_type}" "${cwd_field:-${PWD}}" "${audit_passed_flag}"

# Now enforce the audit result.  Marker is already on disk, so even if
# we block here, _infer_state_from_filesystem will see the failure.
if [[ ${audit_rc} -ne 0 && -n "${audit_output}" ]]; then
    hook_block "SubagentStop audit failed for ${agent_type}:
${audit_output}

Hint: this subagent's transcript shows it claimed completion but did not
actually invoke the required evaluation/build scripts. Inspect transcript at:
  ${agent_transcript}
To bypass in emergency only: export LINGXI_LOOP_HOOK_DISABLE=1"
fi

hook_allow
