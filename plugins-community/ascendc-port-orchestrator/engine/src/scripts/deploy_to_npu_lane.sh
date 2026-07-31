#!/bin/bash
# deploy_to_npu_lane.sh — wrapper around deploy_to_npu.sh with per-lane env overrides.
# Lane 0 is the default (no overrides). Lanes 1..7 use AscendOpGenAgent_lane<N> dirs.
#
# Usage:
#   bash deploy_to_npu_lane.sh --lane 0           # equivalent to deploy_to_npu.sh
#   bash deploy_to_npu_lane.sh --lane 2 --build
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LANE=0
ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --lane) LANE="$2"; shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

# P0l (2026-05-05): sync ASCENDC_WORKSPACE → LOCAL_TASK before deploy.
# Bug: LOCAL_TASK is a shared singleton dir (lane 0 default
# $HOME/workspace/AscendOpGenAgent/current_task; lane N+ dedicated). Nothing
# in the deploy chain populates it from the active op's workspace, so
# whatever the prior op left in LOCAL_TASK gets re-deployed. Cold-starting
# op#10 LayerNorm ran the prior op's _gelu_ext build → wrong .so.
# Fix: rsync workspace contents (kernel + benchmark inputs) into LOCAL_TASK
# before invoking deploy_to_npu.sh. Workspace path comes from
# $ASCENDC_WORKSPACE env var (set by orchestrator brief).
_sync_workspace_to_local_task() {
    local src="$1"
    local dst="$2"
    if [ -z "$src" ] || [ ! -d "$src" ]; then
        return 0  # No workspace to sync — caller falls through to legacy behavior
    fi
    mkdir -p "$dst"
    # Sync kernel/ (kernel source files), model.py, model_new_ascendc.py,
    # benchmark JSON, edge_dataset prep artifacts. Exclude logs/state files
    # (workspace internal bookkeeping) from being deployed.
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete --delete-excluded \
            --include='kernel/' --include='kernel/**' \
            --exclude='.*' \
            --include='*.py' --include='*.json' --include='*.pt' \
            --exclude='probes/' --exclude='probe_*' \
            --exclude='msprof_*' --exclude='self_critic_report.md' \
            --exclude='PROGRESS*.md' --exclude='analysis.md' \
            --exclude='knowledge_update.md' --exclude='failures_ledger.md' \
            --exclude='*.bak*' --exclude='*.backup*' \
            --exclude='*' \
            "$src/" "$dst/" 2>&1 | tail -5
    else
        # Fallback: cp -a for environments without rsync
        find "${dst:?}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
        cp -a "$src/kernel" "$dst/" 2>/dev/null || true
        find "$src" -maxdepth 1 \( -name '*.py' -o -name '*.json' -o -name '*.pt' \) \
            ! -name 'self_critic_report.md' ! -name 'knowledge_update.md' \
            -exec cp {} "$dst/" \; 2>/dev/null || true
    fi
    echo "[deploy/lane${LANE}] synced ASCENDC_WORKSPACE=$src → LOCAL_TASK=$dst"
}

if [ "$LANE" = "0" ]; then
    # Default — no per-lane overrides, but still sync workspace into the SAME
    # path deploy_to_npu.sh will tar. DEBT-STAGE-COLLISION (2026-06-16): the
    # lane-0 default must match deploy_to_npu.sh's isolated default
    # (current_task_${USER}_lane${LANE}) EXACTLY, else the writer syncs to one
    # dir and deploy tars another (empty) dir. An explicit LOCAL_TASK override
    # still wins (kept the `${LOCAL_TASK:-...}` form). LANE=0 here, $USER may be
    # empty under bare env → fall back to `id -un` (mirrors deploy_to_npu.sh).
    _STAGE_USER="${USER:-$(id -un 2>/dev/null || echo unknown)}"
    DEFAULT_LOCAL_TASK="${LOCAL_TASK:-$HOME/workspace/AscendOpGenAgent/current_task_${_STAGE_USER}_lane${LANE}}"
    # Export the resolved path so deploy_to_npu.sh's _PRE_LOCAL_TASK capture
    # uses the EXACT same string we synced to (zero chance of writer/deploy
    # divergence). An explicit caller LOCAL_TASK already flowed into
    # DEFAULT_LOCAL_TASK above, so this preserves the override too.
    export LANE
    export LOCAL_TASK="$DEFAULT_LOCAL_TASK"
    # If the caller sourced .ascendc_env before invoking this wrapper, a shared
    # BENCHMARK_ROOT=/root/AscendOpGenAgent can look like an explicit override
    # to deploy_to_npu.sh and bypass its lane-0 isolation rule. Normalize that
    # shared root here so deploy/build writes to the same lane root that O5 reads.
    _SHARED_REMOTE_DEFAULT="/root/AscendOpGenAgent"
    _BENCHMARK_ROOT_IN="${BENCHMARK_ROOT:-}"
    _BENCHMARK_ROOT_NORM="${_BENCHMARK_ROOT_IN%/}"
    if [ -z "$_BENCHMARK_ROOT_NORM" ] || [ "$_BENCHMARK_ROOT_NORM" = "$_SHARED_REMOTE_DEFAULT" ]; then
        export BENCHMARK_ROOT="/home/npu_user/workspace/AscendOpGenAgent_lane${LANE}"
    fi
    _sync_workspace_to_local_task "${ASCENDC_WORKSPACE:-}" "$DEFAULT_LOCAL_TASK"
    exec bash "$SCRIPT_DIR/deploy_to_npu.sh" ${ARGS[@]+"${ARGS[@]}"}
fi

# Lane N>0 — set per-lane env then call
export LANE
export ASCEND_VISIBLE_DEVICES=$LANE
# Note: ASCEND_VISIBLE_DEVICES is for runtime (verifier / performance.py), not build.
# The deploy_to_npu.sh build step doesn't need ASCEND_VISIBLE_DEVICES.

# P0aaz (2026-05-12): detect local A2 mode — use project-local lane dirs
# instead of remote /home/npu_user/ paths. Mirrors deploy_to_npu.sh local detection.
_IS_LOCAL=false
_ENV_FILE="$SCRIPT_DIR/../../workspace/.ascendc_env"
if [ -f "$_ENV_FILE" ]; then
    _TARGET=$(grep "^TARGET=" "$_ENV_FILE" | cut -d= -f2)
    _TARGET_UPPER=$(echo "${_TARGET:-a5}" | tr '[:lower:]' '[:upper:]')
    _HOST=$(grep "^${_TARGET_UPPER}_HOST=" "$_ENV_FILE" | cut -d= -f2)
    _CONTAINER=$(grep "^${_TARGET_UPPER}_CONTAINER=" "$_ENV_FILE" | cut -d= -f2)
    if [ "$_HOST" = "localhost" ] && [ "$_CONTAINER" = "local" ]; then
        _IS_LOCAL=true
    fi
fi

if [ "$_IS_LOCAL" = "true" ]; then
    # Local-mode lane: use project-local .lanes/ directory
    _PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    export LOCAL_TASK="$_PROJECT_ROOT/workspace/.lanes/lane${LANE}/current_task"
    export BENCHMARK_ROOT="$_PROJECT_ROOT/vendor/AscendOpGenAgent"
    export DEPLOY_STAGE_DIR=""
    mkdir -p "$LOCAL_TASK"
else
    # Remote-mode lane: use A3/A5 remote paths
    export LOCAL_TASK="$HOME/workspace/AscendOpGenAgent_lane${LANE}/current_task"
    export BENCHMARK_ROOT="/home/npu_user/workspace/AscendOpGenAgent_lane${LANE}"
    # Per-lane stage dir so concurrent uploads don't clobber each other's tarball.
    export DEPLOY_STAGE_DIR="ascendc_op_gen_stage_lane${LANE}"
fi

if [ ! -d "$LOCAL_TASK" ]; then
    echo "ERROR: lane $LANE host dir $LOCAL_TASK missing — run setup_lanes.sh --max-lane $LANE first" >&2
    exit 2
fi

# P0l: sync workspace → LOCAL_TASK before deploy (lane N+ path)
_sync_workspace_to_local_task "${ASCENDC_WORKSPACE:-}" "$LOCAL_TASK"

echo "[deploy/lane$LANE] LOCAL_TASK=$LOCAL_TASK BENCHMARK_ROOT=$BENCHMARK_ROOT DEPLOY_STAGE_DIR=$DEPLOY_STAGE_DIR"
exec bash "$SCRIPT_DIR/deploy_to_npu.sh" ${ARGS[@]+"${ARGS[@]}"}
