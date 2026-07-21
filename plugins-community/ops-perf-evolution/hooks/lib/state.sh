#!/usr/bin/env bash
# state.sh — Stop hook rule checks against state.json.
#
# Sourced by .claude/hooks/loop-stop.sh.
# Exposes:
#   check_stop_rules <evo_dir>
# which echoes a list of violation reasons (one per line) and returns 0 if no
# violations, or 1 if any.

if [[ -n "${LOOP_HOOK_STATE_SOURCED:-}" ]]; then
    return 0
fi
LOOP_HOOK_STATE_SOURCED=1

# Detect concrete violations of state.json invariants.
# Echoes reasons to stdout (caller assembles them); returns 1 if any.
check_stop_rules() {
    local evo_dir="$1"
    local state_file="${evo_dir}/state.json"

    if [[ ! -f "${state_file}" ]]; then
        # No state.json → no active evolution loop → noop
        return 0
    fi

    # Use the Python validator for the heavy lifting; it owns the schema.
    local issues_output
    issues_output="$(python3 "${STATE_OPS_PY}" validate \
        --evo-dir "${evo_dir}" \
        --check-stage-artifacts 2>&1 1>/dev/null)"
    local validate_rc=$?

    if [[ ${validate_rc} -ne 0 ]]; then
        # state_ops.py validate prints issues to stderr, one per line prefixed with "  - "
        echo "${issues_output}" | sed -n 's/^  - /R2: /p'
    fi

    # Additional rules that need stage-aware logic beyond schema validation

    local stage
    stage="$(read_state_field "${evo_dir}" "stage")"

    local drift
    drift="$(read_state_field "${evo_dir}" "drift_status")"

    local must_run_json
    must_run_json="$(read_state_field "${evo_dir}" "must_run_before_next_round")"

    local partial_json
    partial_json="$(read_state_field "${evo_dir}" "partial_status")"

    local state_evo_dir
    state_evo_dir="$(read_state_field "${evo_dir}" "evo_dir")"

    # R3: drift_status=replan_required but agent is trying to stop in mid-round_select
    if [[ "${drift}" == "replan_required" ]]; then
        case "${stage}" in
            round_select|round_generate)
                echo "R3: drift_status=replan_required but stage=${stage}; agent must run drift_replan before stopping"
                ;;
        esac
    fi

    # R4: must_run_before_next_round non-empty and agent entered round_select
    if [[ "${stage}" == "round_select" ]] && [[ -n "${must_run_json}" && "${must_run_json}" != "[]" ]]; then
        echo "R4: must_run_before_next_round=${must_run_json} but stage already=round_select"
    fi

    # R5: session drift (state.evo_dir disagrees with hook-detected evo_dir)
    if [[ -n "${state_evo_dir}" ]]; then
        local state_abs hook_abs
        state_abs="$(python3 -c "import os,sys;print(os.path.abspath('${state_evo_dir}'))" 2>/dev/null)"
        hook_abs="$(python3 -c "import os,sys;print(os.path.abspath('${evo_dir}'))" 2>/dev/null)"
        if [[ -n "${state_abs}" && -n "${hook_abs}" && "${state_abs}" != "${hook_abs}" ]]; then
            echo "R5: state.evo_dir=${state_abs} disagrees with hook-detected evo_dir=${hook_abs}"
        fi
    fi

    # R2.1: stage=round_generate and at least one partial is running/pending
    if [[ "${stage}" == "round_generate" ]]; then
        if echo "${partial_json}" | grep -qE '"(running|pending)"'; then
            echo "R2.1: stage=round_generate but partial_status has running/pending: ${partial_json}"
        fi
    fi

    # -----------------------------------------------------------------
    # Anti-skip rules — block agent from claiming "done" without
    # actually running the required rounds / partials / refine / msprof.
    # All rules rely on filesystem-derived state (post-infer), not LLM-written
    # fields, so they cannot be bypassed by agent lying.
    # -----------------------------------------------------------------

    # Read world_model.json's session for requested_rounds / actual_rounds_completed
    local wm_path="${evo_dir}/world_model.json"
    local requested_rounds=""
    local actual_rounds=""
    if [[ -f "${wm_path}" ]]; then
        requested_rounds="$(python3 -c "
import json, sys
try:
    w = json.load(open('${wm_path}'))
    print(w.get('session',{}).get('requested_rounds',''))
except Exception:
    pass
" 2>/dev/null)"
        actual_rounds="$(python3 -c "
import json, sys
try:
    w = json.load(open('${wm_path}'))
    print(w.get('session',{}).get('actual_rounds_completed',''))
except Exception:
    pass
" 2>/dev/null)"
    fi

    local expected_parallel
    expected_parallel="$(read_state_field "${evo_dir}" "expected_parallel_num")"

    # R6: agent tries to stop before completing requested_rounds.
    # Only enforce when stage is in a round-active phase. Pre-round stages
    # (init/shared_prep/baseline_build/wm_init) and terminal/finalize stages
    # are exempt — pre-round means rounds haven't even started, terminal means
    # the user accepted early exit.
    if [[ -n "${requested_rounds}" && -n "${actual_rounds}" ]]; then
        if [[ "${actual_rounds}" =~ ^[0-9]+$ && "${requested_rounds}" =~ ^[0-9]+$ ]]; then
            if (( actual_rounds < requested_rounds )); then
                case "${stage}" in
                    init|shared_prep|baseline_build|wm_init|done|aborted|finalize|report)
                        # Pre-round (agent hasn't started yet) or terminal — allow.
                        ;;
                    *)
                        echo "R6: actual_rounds_completed=${actual_rounds} < requested_rounds=${requested_rounds}; finish remaining rounds or cancel explicitly"
                        ;;
                esac
            fi
        fi
    fi

    # R7: number of parallel partials actually produced < expected_parallel_num.
    # Only enforced once we've started a round (stage past wm_init/shared_prep)
    # and expected_parallel_num was set at init time (0 = skip check).
    if [[ -n "${expected_parallel}" && "${expected_parallel}" =~ ^[0-9]+$ && "${expected_parallel}" -gt 0 ]]; then
        case "${stage}" in
            round_generate|round_refine|round_react|round_checkpoint)
                local actual_parallel
                actual_parallel="$(python3 -c "
import json
try:
    s = json.load(open('${evo_dir}/state.json'))
    print(len(s.get('partial_status', {})))
except Exception:
    print(0)
" 2>/dev/null)"
                if [[ "${actual_parallel}" =~ ^[0-9]+$ && "${actual_parallel}" -lt "${expected_parallel}" ]]; then
                    echo "R7: only ${actual_parallel} partial(s) produced, expected ${expected_parallel}; start the missing ones"
                fi
                ;;
        esac
    fi

    # R8: all partials done but refine never ran (wm.actual_rounds < max_round seen on disk).
    # This catches "agent collected partial results but skipped wm_ops refine and tried to stop".
    if [[ "${stage}" == "round_generate" && -n "${partial_json}" && "${partial_json}" != "{}" ]]; then
        # No running/pending partials (otherwise R2.1 fires first)
        if ! echo "${partial_json}" | grep -qE '"(running|pending)"'; then
            # Compare actual_rounds vs current_round
            local current_round
            current_round="$(read_state_field "${evo_dir}" "current_round")"
            if [[ "${current_round}" =~ ^[0-9]+$ && "${actual_rounds}" =~ ^[0-9]+$ ]]; then
                if (( actual_rounds < current_round )); then
                    echo "R8: round ${current_round} partials all completed but wm.actual_rounds_completed=${actual_rounds}; run wm_ops.py refine before stopping"
                fi
            fi
        fi
    fi

    # R13 (v3.2 C8-T4): refine 完成后所有 passed 节点必须有 diagnosis 字段。
    # 仅检查"字段存在"，不强制 18 词表内（容错给 LLM 学习空间）。
    # 触发条件：stage 在 round_refine/round_react/round_checkpoint 或 done 之前
    local wm_path="${evo_dir}/world_model.json"
    if [[ -f "${wm_path}" ]] && [[ -f "${PROJECT_ROOT}/.claude/skills/evolution-world-model/scripts/wm_ops.py" ]]; then
        case "${stage}" in
            round_refine|round_react|round_checkpoint|round_select)
                local r13_output
                r13_output="$(python3 "${PROJECT_ROOT}/.claude/skills/evolution-world-model/scripts/wm_ops.py" \
                    validate-diagnosis --wm-path "${wm_path}" 2>/dev/null)"
                if [[ -n "${r13_output}" ]]; then
                    # 解析 issues_count > 0 表示有缺 diagnosis 的节点
                    local issues_count
                    issues_count="$(echo "${r13_output}" | python3 -c "
import sys, json
try: data = json.loads(sys.stdin.read())
except Exception: print(0); sys.exit()
print(data.get('issues_count', 0))
" 2>/dev/null)"
                    # F-G3: 严格用 status=passed + solution_ref 以 "round_" 开头判定。
                    #   排除 status=open(未评估) / root(solution_ref="baseline") / score=0 误报
                    local missing_passed
                    missing_passed="$(python3 -c "
import json
wm = json.load(open('${wm_path}'))
nodes = wm.get('decision_tree', {}).get('nodes', {})
miss = [nid for nid, n in nodes.items()
        if n.get('status') == 'passed'
        and isinstance(n.get('solution_ref'), str)
        and n.get('solution_ref').startswith('round_')
        and not n.get('diagnosis')]
print(len(miss))
print(','.join(miss[:3]))
" 2>/dev/null)"
                    local miss_count
                    miss_count="$(echo "${missing_passed}" | head -1)"
                    local miss_sample
                    miss_sample="$(echo "${missing_passed}" | tail -1)"
                    if [[ -n "${miss_count}" ]] && (( miss_count > 0 )); then
                        echo "R13: ${miss_count} passed node(s) lack diagnosis field (sample: ${miss_sample}); run Stage 2 LLM diagnosis to populate node.diagnosis = {diagnosis_text, bottleneck_labels, confidence} before stopping"
                    fi
                fi
                ;;
        esac
    fi

    # R14: ledger must reflect latest diagnoses. After LLM populates
    # node.diagnosis, ledger artifacts (lineage.jsonl / attempt-ledger.md)
    # still hold the refine-time snapshot where diagnosis was null. Agent
    # must call `wm_ops.py finalize-ledger` to reconcile.
    local lineage_path="${evo_dir}/artifacts/lineage.jsonl"
    if [[ -f "${wm_path}" ]] && [[ -f "${lineage_path}" ]]; then
        case "${stage}" in
            round_checkpoint|round_select|finalize|done)
                local stale_count
                stale_count="$(python3 -c "
import json
wm = json.load(open('${wm_path}'))
nodes = wm.get('decision_tree', {}).get('nodes', {})
stale = 0
for line in open('${lineage_path}'):
    line = line.strip()
    if not line: continue
    try: e = json.loads(line)
    except Exception: continue
    nid = e.get('node_id')
    wm_diag = (nodes.get(nid) or {}).get('diagnosis')
    if wm_diag and not e.get('diagnosis'):
        stale += 1
print(stale)
" 2>/dev/null)"
                if [[ -n "${stale_count}" ]] && (( stale_count > 0 )); then
                    echo "R14: ledger has ${stale_count} stale entry/entries (diagnosis missing in lineage.jsonl but present in world_model.json); run 'wm_ops.py finalize-ledger --wm-path ${wm_path} --evo-dir ${evo_dir}' before stopping"
                fi
                ;;
        esac
    fi

    # R15/R16 are stage-independent wm integrity checks; loop-stop.sh invokes
    # check_wm_integrity_rules separately (so they fire even at terminal
    # stages). Don't duplicate here.

    return 0
}


# Stage-independent rules on world_model.json itself. Safe to call regardless
# of stage (including terminal stages done/aborted), since these check whether
# the saved wm has structurally broken nodes — that's a bug at any time.
check_wm_integrity_rules() {
    local evo_dir="$1"
    local wm_path="${evo_dir}/world_model.json"
    [[ -f "${wm_path}" ]] || return 0

    local oe_issues
    oe_issues="$(python3 -c "
import json
wm = json.load(open('${wm_path}'))
nodes = wm.get('decision_tree', {}).get('nodes', {})
r15 = [nid for nid, n in nodes.items()
       if n.get('mode') == 'open_exploration' and (n.get('strategy_combination') or [])]
r16 = [nid for nid, n in nodes.items()
       if n.get('mode') == 'open_exploration'
       and len((n.get('description') or '').strip()) < 80]
print(','.join(r15[:3]))
print(','.join(r16[:3]))
" 2>/dev/null)"
    local r15_sample r16_sample
    r15_sample="$(echo "${oe_issues}" | head -1)"
    r16_sample="$(echo "${oe_issues}" | tail -1)"
    if [[ -n "${r15_sample}" ]]; then
        echo "R15: open_exploration node(s) have non-empty strategy_combination (sample: ${r15_sample}); open nodes must keep sc=[] — strategies must NOT be injected. Clear sc or change mode."
    fi
    if [[ -n "${r16_sample}" ]]; then
        echo "R16: open_exploration node(s) lack a concrete direction (sample: ${r16_sample}); description must be ≥80 chars referencing profiling facts + parent hint rationale, not a placeholder."
    fi
}

export -f check_stop_rules
export -f check_wm_integrity_rules
