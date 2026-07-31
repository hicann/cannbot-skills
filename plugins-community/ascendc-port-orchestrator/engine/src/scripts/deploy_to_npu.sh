#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# Deploy current_task to NPU container — multi-target NPU container deploy.
# Usage: bash src/scripts/deploy_to_npu.sh [--build]
#
# Reads TARGET from workspace/.ascendc_env and resolves HOST/USER/PASSWORD/CONTAINER/
# CANN_PATH/SOC_VERSION via src/scripts/resolve_target.sh. Auth mode switches between
# sshpass (when $PASSWORD non-empty, A5 default) and key-based (A3 default).
#
# 1. Cleans remote current_task/
# 2. Tars local current_task/
# 3. Scps + container-extracts (no docker cp — use bind-mounted stage path)
# 4. Verifies file count matches
# 5. Optionally builds (--build flag)
#
# Exit codes (Fault Tolerance spec — see aog-kernel-worker.md "Fault Tolerance"):
#   0  = success
#   1  = compile error (build step failed with non-infra stderr)
#   10 = infra failure (after 3× retry with 10s backoff)
#   2  = config / usage error
#
# Marker paths (per-workspace, with /tmp fallback):
#   - If env ASCENDC_WORKSPACE=path is set: writes $ASCENDC_WORKSPACE/.last_build.{stderr,class}
#   - Otherwise: writes /tmp/ascendc_last_build.{stderr,class}
# PreToolUse hook block_edit_on_infra.sh reads the same markers.
#
# Build archive (GEPA design §15.2, opt-in via env, additive — no default behavior change):
#   - BUILD_ARCHIVE_ENABLED=1                  → archive each build to A5 /data/build_archive/
#   - BUILD_ARCHIVE_ROOT=/data/build_archive   → container-side archive root (default)
#   - BUILD_BATCH_ID=orch_<UTC>                → group multiple iters under one batch
#   - CANDIDATE_BRIEF_HASH=<sha>               → tag for GEPA prompt-evolution candidate
# When enabled, every build (pass or fail) uploads {kernel snapshot, stderr.log on failure,
# meta.json} to $ARCHIVE_ROOT/<op>/<batch>/iter_NNN/. Workspace breadcrumb in
# $_WS_DIR/.last_build_archive_path so trace_serializer.py can locate it.

# intentionally NOT -e: we capture rc and classify
# intentionally NOT -u: .ascendc_env may have passwords with '$'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# ENV_FILE honors ASCENDC_ENV_FILE (same override resolve_target.sh uses) so the
# env-resolution path can be exercised against a crafted env (DEPLOY_RESOLVE_ONLY
# tests); defaults to the project workspace env.
ENV_FILE="${ASCENDC_ENV_FILE:-$PROJECT_ROOT/workspace/.ascendc_env}"

# ─── Resolve marker location ──────────────────────────────────────────
if [ -n "${ASCENDC_WORKSPACE:-}" ]; then
    if [ "${ASCENDC_WORKSPACE:0:1}" = "/" ]; then
        _WS_DIR="$ASCENDC_WORKSPACE"
    else
        _WS_DIR="$PROJECT_ROOT/$ASCENDC_WORKSPACE"
    fi
    if [ ! -d "$_WS_DIR" ]; then
        echo "[deploy] WARN: ASCENDC_WORKSPACE=$ASCENDC_WORKSPACE not a directory, falling back to /tmp markers" >&2
        STDERR_MARKER=/tmp/ascendc_last_build.stderr
        CLASS_MARKER=/tmp/ascendc_last_build.class
    else
        STDERR_MARKER="$_WS_DIR/.last_build.stderr"
        CLASS_MARKER="$_WS_DIR/.last_build.class"
    fi
else
    STDERR_MARKER=/tmp/ascendc_last_build.stderr
    CLASS_MARKER=/tmp/ascendc_last_build.class
fi
: > "$STDERR_MARKER"
: > "$CLASS_MARKER"

# ─── DEBT-197 self-heal (local mode): provision the full build baseline TRIO ──
# build_ascendc.py (entrypoint) does `from build_capabilities import ...` after adding
# its own dir to sys.path (patches/build_ascendc.py:15-29), and consumes cann_stubs/ as
# the DEBT-110 baseline include. Provisioning ONLY the entrypoint (the old behavior) left
# a truly fresh container without those siblings → `ModuleNotFoundError: No module named
# 'build_capabilities'` surfacing as an empty-stderr phantom compile failure. This helper
# provisions the whole dependency closure — build_ascendc.py + build_capabilities/ (package
# dir, recursive) + cann_stubs/ (dir, recursive) — into $BUILD_ROOT/utils/. Idempotent: it
# skips any piece already present, so it fires when ANY of the trio is missing (a container
# can have the file but not the deps). Fail-soft to the existing INFRA_BASELINE_VIOLATED
# path with a CLEAR stderr marker (never an empty-stderr phantom failure).
# Args: $1=BUILD_ROOT  $2=patch source dir (patches/). Uses globals CLASS_MARKER,
# STDERR_MARKER, TARGET. Returns 0 on success, 2 on hard-fail (a required piece absent in
# utils/ AND absent from the shipped bundle, or the post-provision guard fails).
_debt197_provision_trio_local() {
    local _build_root="$1"
    local _patch_src="$2"
    local _utils="$_build_root/utils"
    local _piece _dst _src
    mkdir -p "$_utils"
    for _piece in build_ascendc.py build_capabilities cann_stubs; do
        _dst="$_utils/$_piece"
        _src="$_patch_src/$_piece"
        if [ -e "$_dst" ]; then
            continue   # idempotent — already provisioned, leave it untouched
        fi
        if [ -e "$_src" ]; then
            cp -r "$_src" "$_dst"
            echo "[deploy/${TARGET:-local}/local] DEBT-197 self-heal: provisioned bundle $_piece → $_utils/"
        else
            echo "ERROR: $_dst absent AND no bundled $_piece at $_src" >&2
            echo "compile" > "$CLASS_MARKER"
            echo "build baseline piece '$_piece' missing and no bundled copy to self-heal (DEBT-197 trio)" > "$STDERR_MARKER"
            return 2
        fi
    done
    # Guard: build_capabilities MUST be an importable package DIR (not just the file)
    # before build_ascendc.py runs, else the phantom empty-stderr ModuleNotFoundError
    # returns. Test the directory, and emit a clear diagnostic if it is still missing.
    if [ ! -d "$_utils/build_capabilities" ]; then
        echo "ERROR: $_utils/build_capabilities is not a package directory after self-heal" >&2
        echo "compile" > "$CLASS_MARKER"
        echo "DEBT-197 guard: build_capabilities package dir missing at $_utils/build_capabilities after self-heal — build_ascendc.py would raise ModuleNotFoundError: No module named 'build_capabilities'" > "$STDERR_MARKER"
        return 2
    fi
    return 0
}

# Test hook: exercise ONLY the local self-heal + guard, then exit before any target
# resolution / real build. Consumed by src/scripts/tests/test_deploy_selfheal_trio.py
# (no NPU / remote host required). DEBT197_SELFHEAL_TEST_BUILD_ROOT + optional
# DEBT197_SELFHEAL_TEST_PATCH_DIR (defaults to $SCRIPT_DIR/patches).
if [ "${DEBT197_SELFHEAL_TEST_ONLY:-}" = "1" ]; then
    _debt197_provision_trio_local \
        "${DEBT197_SELFHEAL_TEST_BUILD_ROOT:?DEBT197_SELFHEAL_TEST_BUILD_ROOT required}" \
        "${DEBT197_SELFHEAL_TEST_PATCH_DIR:-$SCRIPT_DIR/patches}"
    exit $?
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found. Run this plugin's init.sh, then fill the scaffolded workspace/.ascendc_env." >&2
    exit 2
fi

# 2026-04-30 lane wrapper fix: capture caller-injected env BEFORE any source of
# .ascendc_env (resolve_target.sh and the re-source below both call `set -a; . ENV;`
# which would otherwise clobber lane-specific BENCHMARK_ROOT/LOCAL_TASK/DEPLOY_STAGE_DIR
# back to the file's defaults). The earlier "preserve after re-source" attempt was
# too late — resolve_target.sh's source already nuked the values.
_PRE_BENCHMARK_ROOT="${BENCHMARK_ROOT:-}"
_PRE_LOCAL_TASK="${LOCAL_TASK:-}"
_PRE_DEPLOY_STAGE_DIR="${DEPLOY_STAGE_DIR:-}"
_PRE_LANE="${LANE:-}"
# DEBT-STAGE-COLLISION (2026-06-16): capture the LOCAL OS account BEFORE
# resolve_target.sh runs — that source sets `USER` to the REMOTE ssh-login user
# (A5_USER, default "root"), which is NOT a per-instance discriminator. We want
# the local account that owns $HOME so two accounts sharing one $HOME get
# distinct staging dirs. `id -un` is independent of the $USER clobber.
_LOCAL_ACCOUNT="${USER:-$(id -un 2>/dev/null || echo unknown)}"

# ─── Resolve active TARGET → flat HOST/USER/PASSWORD/CONTAINER/CANN_PATH/SOC_VERSION ──
# shellcheck disable=SC1091
source "$SCRIPT_DIR/resolve_target.sh" || {
    echo "ERROR: resolve_target.sh failed. Check $ENV_FILE has TARGET=… and the matching ${TARGET^^}_HOST/CONTAINER/SOC_VERSION set." >&2
    exit 2
}

# DEBT-DEPLOY-ENV-CLOBBER (back-agent 2026-05-30): the re-source below re-reads
# the file's TOP-LEVEL CANN_PATH/SOC_VERSION — which are typically the a5
# defaults — and would clobber the PER-TARGET values resolve_target.sh just
# resolved (e.g. TARGET=a3 → A3_CANN_PATH/A3_SOC_VERSION). Capture the resolved
# values here so they can be restored after the re-source (same pattern as the
# _PRE_* lane overrides above). For default-target deploys (top-level ==
# resolved) the restore is a harmless no-op. Found during the first non-a5
# backward-op hardware loop on npu-a3-back.
_RESOLVED_CANN_PATH="${CANN_PATH:-}"
_RESOLVED_SOC_VERSION="${SOC_VERSION:-}"

# BENCHMARK_ROOT is a legacy harness path (set in .ascendc_env directly, not per-target);
# also re-source the env file to pick it up (resolve_target.sh sources it but doesn't export
# all keys uniformly). Safe re-source.
set -a; source "$ENV_FILE"; set +a

# Restore caller-injected lane overrides (after both sources have run).
if [ -n "$_PRE_BENCHMARK_ROOT" ]; then export BENCHMARK_ROOT="$_PRE_BENCHMARK_ROOT"; fi
if [ -n "$_PRE_LOCAL_TASK" ];     then export LOCAL_TASK="$_PRE_LOCAL_TASK"; fi
if [ -n "$_PRE_DEPLOY_STAGE_DIR" ]; then export DEPLOY_STAGE_DIR="$_PRE_DEPLOY_STAGE_DIR"; fi
if [ -n "$_PRE_LANE" ];           then export LANE="$_PRE_LANE"; fi
# Restore resolve_target.sh's per-target CANN_PATH/SOC_VERSION (DEBT-DEPLOY-ENV-CLOBBER).
if [ -n "$_RESOLVED_CANN_PATH" ];   then export CANN_PATH="$_RESOLVED_CANN_PATH"; fi
if [ -n "$_RESOLVED_SOC_VERSION" ]; then export SOC_VERSION="$_RESOLVED_SOC_VERSION"; fi

# ─── Resolve LOCAL_TASK / REMOTE_TASK staging paths ───────────────────
# LOCAL_TASK can be overridden via env (multi-CC instance support, 2026-04-27).
# An explicit LOCAL_TASK env override ALWAYS wins (the `${LOCAL_TASK:-...}` form) —
# Kimi-style overrides and the lane-wrapper's per-lane LOCAL_TASK (lane N>0) must
# not break.
# For Kimi/parallel CC: set LOCAL_TASK=$HOME/workspace/AscendOpGenAgent_kimi/current_task
# in workspace/.ascendc_env to isolate from other CC instances.
#
# DEBT-STAGE-COLLISION (2026-06-16, account-01 celu clobbered account-02 selective_scan):
# the prior default `$HOME/workspace/AscendOpGenAgent/current_task` was BOTH
# account-agnostic AND lane-agnostic. Two parallel op-gen runs sharing the same
# $HOME (different OS accounts and/or different NPU lanes, both at lane 0) both
# resolved LOCAL_TASK to that single shared dir and clobbered each other's local
# tar/scp staging → both runs corrupted. The default now ENFORCES isolation by
# baking BOTH the account ($USER) AND the lane (LANE) into the path, so parallel
# runs never collide WITHOUT requiring a manual override. The writer side
# (deploy_to_npu_lane.sh lane-0 _sync_workspace_to_local_task) computes the SAME
# default via this exact shape — keep them in sync.
#   _LOCAL_ACCOUNT was captured pre-source (the post-source $USER is the remote
#   ssh user, not the local account — see capture above).
#   LANE defaults to 0 (single-CC / direct deploy_to_npu.sh invocation).
# Resolved BEFORE the dry-run hook so DEPLOY_RESOLVE_ONLY can report the paths.
_STAGE_LANE="${LANE:-0}"
LOCAL_TASK="${LOCAL_TASK:-$HOME/workspace/AscendOpGenAgent/current_task_${_LOCAL_ACCOUNT}_lane${_STAGE_LANE}}"

# DEBT-159 (2026-06-16, celu lane2 cleaned BOTH _lane2/current_task AND the
# shared /root/AscendOpGenAgent/current_task): the REMOTE benchmark root must be
# lane-isolated EXACTLY like LOCAL_TASK above, else a lane-N run that reaches
# this script without an explicit BENCHMARK_ROOT (e.g. the worker's manual
# `LOCAL_TASK=… deploy_to_npu.sh` fallback to dodge the git-tracked current_task
# symlink — which sets LOCAL_TASK but NOT BENCHMARK_ROOT) falls back to the
# SHARED /root/AscendOpGenAgent/current_task and clobbers other lanes. That
# shared touch is what forced two parallel agents to serialize the whole
# container instead of just the NPU.
#
# Resolution rule (single source of truth — phase_o5_runner._lane_aware_benchmark_root
# MIRRORS this EXACTLY so deploy/build/O5 all resolve the SAME remote root for a
# given lane). Precedence, mirroring O5's lane!=0 vs lane==0 split:
#   1. _PRE_BENCHMARK_ROOT (caller EXPORTED BENCHMARK_ROOT before invoking us — the
#      lane wrapper lane>0, host-mode, local-mode, Kimi-style override)  → verbatim;
#      the caller asserted an exact root.
#   2. else, LANE set AND > 0 (lane-managed run on a real lane)  → isolate to the
#      canonical REMOTE lane base /home/npu_user/workspace/AscendOpGenAgent_lane${LANE}
#      (the SAME string the lane wrapper exports for lane>0 AND O5 returns), IGNORING
#      any env-file base — so a per-lane run is ALWAYS lane-isolated even if the env
#      carries a shared base (/root) or a per-instance base (/data/...). npu_user is
#      the CONTAINER/remote home (REMOTE_TASK is a container path), not local $HOME.
#   3. else (lane 0 OR no lane): honor a NON-shared env-file BENCHMARK_ROOT verbatim
#      (the per-instance configs /data/.../AscendOpGenAgent, .../AscendOpGenAgent_kimi
#      / _ds / _faregen — these are already unique, and O5 honors them at lane 0 too).
#   4. else, LANE explicitly == 0 with a SHARED-or-empty env (lane-managed lane-0 run;
#      the lane wrapper exports LANE=0)  → isolate to AscendOpGenAgent_lane0 so two
#      lane-0 runs never share /root/AscendOpGenAgent/current_task (the DEBT-159 core
#      goal). Guardrail: we make the step lane-aware, not mutate the shared-default env.
#   5. else (LANE unset → bare single-CC `deploy_to_npu.sh`, no lane management)  →
#      legacy /root/AscendOpGenAgent (single-CC backward-compat preserved).
_SHARED_REMOTE_DEFAULT="/root/AscendOpGenAgent"
_ENVFILE_BENCHMARK_ROOT="${BENCHMARK_ROOT:-}"
_ENVFILE_BENCHMARK_ROOT="${_ENVFILE_BENCHMARK_ROOT%/}"
if [ -n "${_PRE_BENCHMARK_ROOT:-}" ]; then
    _BENCHMARK_ROOT_RESOLVED="$_PRE_BENCHMARK_ROOT"
elif [ -n "${_PRE_LANE:-}" ] && [ "$_STAGE_LANE" != "0" ]; then
    _BENCHMARK_ROOT_RESOLVED="/home/npu_user/workspace/AscendOpGenAgent_lane${_STAGE_LANE}"
elif [ -n "$_ENVFILE_BENCHMARK_ROOT" ] && [ "$_ENVFILE_BENCHMARK_ROOT" != "$_SHARED_REMOTE_DEFAULT" ]; then
    _BENCHMARK_ROOT_RESOLVED="$_ENVFILE_BENCHMARK_ROOT"
elif [ -n "${_PRE_LANE:-}" ]; then
    _BENCHMARK_ROOT_RESOLVED="/home/npu_user/workspace/AscendOpGenAgent_lane0"
else
    _BENCHMARK_ROOT_RESOLVED="$_SHARED_REMOTE_DEFAULT"
fi
# Export so the build step's `${BENCHMARK_ROOT:-…}` (BUILD_ROOT, both local +
# remote paths) resolves to the SAME root REMOTE_TASK was deployed under — a
# lane-isolated REMOTE_TASK with a /root BUILD_ROOT would build the wrong (shared)
# current_task.
export BENCHMARK_ROOT="$_BENCHMARK_ROOT_RESOLVED"
REMOTE_TASK="$_BENCHMARK_ROOT_RESOLVED/current_task"

# ─── Per-target deploy stage path ─────────────────────────────────────
# A5 host has its own /root/openeuler/data/ pool that bind-mounts as /data inside the container.
# A3 90.* hosts have historically used /home/npu_user, but shared/team containers
# may expose a real account home at the same in-container path. Allow stage roots
# to be overridden via env before the DEPLOY_RESOLVE_ONLY dry-run.
DEPLOY_STAGE_DIR="${DEPLOY_STAGE_DIR:-ascendc_op_gen_stage}"
_DEPLOY_TARGET="$TARGET"
case "$_DEPLOY_TARGET" in
    *-ds) _DEPLOY_TARGET="${_DEPLOY_TARGET%-ds}" ;;
esac
case "$_DEPLOY_TARGET" in
    a5)
        DEPLOY_STAGE_HOST=${A5_DEPLOY_STAGE_HOST:-/root/openeuler/data}/${DEPLOY_STAGE_DIR}
        DEPLOY_STAGE_CONTAINER=${A5_DEPLOY_STAGE_CONTAINER:-/data}/${DEPLOY_STAGE_DIR}
        ;;
    a3|a2)
        DEPLOY_STAGE_HOST=${A3_DEPLOY_STAGE_HOST:-${A3_HOST_HOME:-/home/npu_user}}/${DEPLOY_STAGE_DIR}
        DEPLOY_STAGE_CONTAINER=${A3_DEPLOY_STAGE_CONTAINER:-/home/npu_user}/${DEPLOY_STAGE_DIR}
        ;;
    *)
        echo "ERROR: unknown TARGET='$TARGET' for deploy_to_npu.sh" >&2
        exit 2
        ;;
esac

# Dry-run hook (DEPLOY_RESOLVE_ONLY=1): print the fully-resolved target env +
# staging paths and exit 0 BEFORE any SSH/build. Lets the env-resolution path
# (incl. the clobber-fix above and the staging-isolation default) be unit-tested
# without hardware. Default off → no behavior change for real deploys.
if [ "${DEPLOY_RESOLVE_ONLY:-0}" = "1" ]; then
    echo "RESOLVED TARGET=$TARGET HOST=$HOST CONTAINER=$CONTAINER CANN_PATH=$CANN_PATH SOC_VERSION=$SOC_VERSION"
    echo "RESOLVED_STAGING LOCAL_TASK=$LOCAL_TASK REMOTE_TASK=$REMOTE_TASK"
    echo "RESOLVED_DEPLOY_STAGE DEPLOY_STAGE_HOST=$DEPLOY_STAGE_HOST DEPLOY_STAGE_CONTAINER=$DEPLOY_STAGE_CONTAINER"
    exit 0
fi

# ─── Optional sudo-gated docker host (A5_DOCKER_SUDO / {TARGET}_DOCKER_SUDO) ────
# Some shared A5/A3 boxes give the ssh user NOPASSWD only for `sudo su` (not `sudo docker`).
# When the flag is set, wrap the remote docker command as `sudo su -c '<cmd>'`; the embedded
# single-quotes are POSIX-escaped (' -> '\'') so `su -c` gets the command as ONE arg.
# Default OFF -> byte-identical to prior behavior. Mirrors phase_o5_helpers._maybe_sudo_wrap_remote.
# Sanitize TARGET for use as a shell var name: '-' is legal in a target slug
# (e.g. "a3-ds") but illegal in a bash identifier, so an indirect expansion on
# "A3-DS_DOCKER_SUDO" hard-errors ("invalid variable name") under set -u and
# aborts deploy. Map any non-alnum char to '_' first. Non-hyphen targets
# (a5/a3/a2) are byte-identical to the prior behavior.
_TDS_VAR="${TARGET^^}_DOCKER_SUDO"
_TDS_VAR="${_TDS_VAR//[^A-Z0-9_]/_}"
_DOCKER_SUDO_FLAG="${!_TDS_VAR:-${A5_DOCKER_SUDO:-}}"
_docker_sudo_on() {
    case "$_DOCKER_SUDO_FLAG" in 1|true|yes|on|TRUE|Yes|YES|On) return 0 ;; *) return 1 ;; esac
}
_wrap_sudo() {  # $1 = remote command string; echoes wrapped-or-unchanged
    if _docker_sudo_on; then
        local c="$1"
        c="${c//\'/\'\\\'\'}"   # POSIX single-quote escape: ' -> '\''
        printf "sudo su -c '%s'" "$c"
    else
        printf '%s' "$1"
    fi
}

if [ ! -d "$LOCAL_TASK" ] || [ -z "$(ls -A "$LOCAL_TASK" 2>/dev/null)" ]; then
    echo "ERROR: $LOCAL_TASK is empty. Write files first." >&2
    exit 2
fi

# ─── Local-mode fast path (A2 CONTAINER=local, no SSH/SCP/docker) ────
# P0aaz (2026-05-12): when HOST=localhost AND CONTAINER=local, the NPU is
# on this machine. Skip SSH/SCP/docker entirely — rsync files and build
# directly. Mirrors phase_o5_runner.py P0aay local detection.
if [ "$HOST" = "localhost" ] && [ "$CONTAINER" = "local" ]; then
    echo "[deploy/$TARGET/local] local-mode: HOST=$HOST CONTAINER=$CONTAINER"
    echo "[deploy/$TARGET/local] LOCAL_TASK=$LOCAL_TASK → REMOTE_TASK=$REMOTE_TASK"

    # Clean
    mkdir -p "$REMOTE_TASK"
    find "${REMOTE_TASK:?}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +

    # Sync
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "$LOCAL_TASK/" "$REMOTE_TASK/"
    else
        cp -a "$LOCAL_TASK/." "$REMOTE_TASK/"
    fi
    LOCAL_COUNT=$(find "$LOCAL_TASK" -type f | grep -v __pycache__ | grep -v '.pyc' | wc -l)
    REMOTE_COUNT=$(find "$REMOTE_TASK" -type f | grep -v __pycache__ | grep -v '.pyc' | grep -v build | wc -l)
    echo "[deploy/$TARGET/local] Verify: local=$LOCAL_COUNT files, remote=$REMOTE_COUNT files"
    if [ "$LOCAL_COUNT" != "$REMOTE_COUNT" ]; then
        echo "WARNING: file count mismatch! Check for stale files." >&2
    fi

    # Build (if --build)
    if [ "${1-}" = "--build" ]; then
        echo "[deploy/$TARGET/local] Building with SOC=$SOC_VERSION CANN=$CANN_PATH ..."
        export PATH="$CANN_PATH/bin:$PATH"
        if [ -d "$CANN_PATH/x86_64-linux/lib64" ]; then
            export LD_LIBRARY_PATH="$CANN_PATH/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"
        elif [ -d "$CANN_PATH/aarch64-linux/lib64" ]; then
            export LD_LIBRARY_PATH="$CANN_PATH/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}"
        else
            export LD_LIBRARY_PATH="$CANN_PATH/lib64:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}"
        fi
        export ASCEND_HOME_PATH="$CANN_PATH"
        export ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp"

        BUILD_ROOT="${BENCHMARK_ROOT:-/root/AscendOpGenAgent}"
        # DEBT-197 self-heal: provisions the full build baseline trio (entrypoint +
        # build_capabilities package + cann_stubs) at $BUILD_ROOT/utils/. These are
        # normally installed already (setup_a3_isolated_container.sh provisions them at
        # container-setup). A redirected BENCHMARK_ROOT (e.g. off a full-disk host) or a
        # fresh cannbot customer container that skipped that setup would hit
        # INFRA_BASELINE_VIOLATED (exit 2) here. Auto-provision the SHIPPED bundle-owned
        # trio so a customer's port_a3 run is turnkey with zero manual provisioning.
        # Idempotent — skips pieces already present, fires when ANY of the trio is
        # missing (build_ascendc.py imports build_capabilities + needs cann_stubs).
        if ! _debt197_provision_trio_local "$BUILD_ROOT" "$SCRIPT_DIR/patches"; then
            exit 2
        fi
        python3 "$BUILD_ROOT/utils/build_ascendc.py" current_task -v "$SOC_VERSION" --build-type Release 2>"$STDERR_MARKER"
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "compile" > "$CLASS_MARKER"
            cat "$STDERR_MARKER" >&2
            echo "ERROR: Build failed — see $STDERR_MARKER" >&2
            exit 1
        fi

        SO_CHECK=$(find "$REMOTE_TASK/kernel/build" -name '*.so' 2>/dev/null | wc -l)
        if [ "$SO_CHECK" = "0" ]; then
            echo "ERROR: Build returned OK but no .so produced." >&2
            echo "compile" > "$CLASS_MARKER"
            echo "Build step produced 0 .so files" > "$STDERR_MARKER"
            exit 1
        fi
        echo "[deploy/$TARGET/local] Build OK (SOC=$SOC_VERSION)"
    fi

    : > "$STDERR_MARKER"
    echo "pass" > "$CLASS_MARKER"
    echo "[deploy/$TARGET/local] Done."
    exit 0
fi

# ─── Auth mode: sshpass vs SSH key ────────────────────────────────────
# A5 uses password auth (.ascendc_env has A5_PASSWORD=…)
# A3 uses key auth (A3_PASSWORD is empty)
# 2026-06-20 (FA-grad .141 V351 lane): some A5/V351 lanes (e.g. the .141
# clean-runtime hosts use KEY auth, not password — the host has
# no password set, only an ssh key (id_ca_team). Honor an OPTIONAL explicit key via
# `{TARGET}_SSH_KEY` (resolved by resolve_target.sh to the flat `SSH_KEY`) so a
# key-auth A5 lane works without a global ~/.ssh/config (which may be read-only in a
# sandbox). When SSH_KEY is set it takes precedence over PASSWORD (an explicit key is
# the stronger signal). Empty SSH_KEY → unchanged legacy behavior (password or default key).
_SSH_KEY="${SSH_KEY:-}"
_KEY_OPT=()
if [ -n "$_SSH_KEY" ]; then
    _KEY_OPT=(-i "$_SSH_KEY")
fi
if [ -n "$_SSH_KEY" ]; then
    SSH_CMD=(ssh "${_KEY_OPT[@]}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 "${USER}@${HOST}")
    SCP_CMD=(scp "${_KEY_OPT[@]}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10)
elif [ -n "$PASSWORD" ]; then
    SSH_CMD=(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${USER}@${HOST}")
    SCP_CMD=(sshpass -p "$PASSWORD" scp -o StrictHostKeyChecking=no -o ConnectTimeout=10)
else
    SSH_CMD=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 "${USER}@${HOST}")
    SCP_CMD=(scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10)
fi

# ─── Infra classification ────────────
is_infra_stderr() {
    local s="$1"
    echo "$s" | grep -qiE "Connection refused|Connection timed out|Connection reset by peer|No route to host|Could not resolve hostname|Permission denied \((publickey|password)|Permission denied, please try again|Error response from daemon: No such container|Container .* is not running|sshpass: Failed to run command: Connection|ssh: connect to host|ssh_exchange_identification"
}
is_container_not_running() {
    local s="$1"
    echo "$s" | grep -qiE "Container .* is not running|Error response from daemon: No such container"
}

run_with_retry() {
    local label="$1"; shift
    local attempt=0 rc=0 stderr_file
    stderr_file=$(mktemp "${TMPDIR:-/tmp}/ascendc_attempt.XXXXXX.stderr")
    while [ $attempt -lt 3 ]; do
        attempt=$((attempt + 1))
        "$@" 2>"$stderr_file"
        rc=$?
        if [ $rc -eq 0 ]; then
            rm -f "$stderr_file"
            return 0
        fi
        local err
        err=$(cat "$stderr_file")
        if is_infra_stderr "$err"; then
            echo "[deploy/$TARGET] $label attempt $attempt/3 infra failure:" >&2
            echo "$err" | head -5 >&2
            if [ $attempt -eq 1 ] && is_container_not_running "$err"; then
                echo "[deploy/$TARGET] attempting 'docker start $CONTAINER' ..." >&2
                local start_err
                start_err=$("${SSH_CMD[@]}" "docker start $CONTAINER" 2>&1 >/dev/null) || true
                sleep 5
                if is_infra_stderr "$start_err"; then
                    echo "[deploy/$TARGET] docker start also hit infra error: $start_err" >&2
                else
                    echo "[deploy/$TARGET] docker start ok, retrying $label immediately" >&2
                    continue
                fi
            fi
            if [ $attempt -lt 3 ]; then
                echo "[deploy/$TARGET] sleep 10 then retry" >&2
                sleep 10
                continue
            fi
            printf '%s\n' "$err" > "$STDERR_MARKER"
            echo "infra" > "$CLASS_MARKER"
            rm -f "$stderr_file"
            echo "[deploy/$TARGET] $label: infra failure after 3 retries. Writing markers." >&2
            return 10
        else
            printf '%s\n' "$err" > "$STDERR_MARKER"
            echo "compile" > "$CLASS_MARKER"
            cat "$stderr_file" >&2
            rm -f "$stderr_file"
            return $rc
        fi
    done
    rm -f "$stderr_file"
    return $rc
}

# ─── Pipeline steps ───────────────────────────────────────────────────
echo "[deploy/$TARGET] Cleaning $HOST:$REMOTE_TASK/ ..."
run_with_retry "ssh clean" "${SSH_CMD[@]}" "$(_wrap_sudo "docker exec $CONTAINER bash -c 'mkdir -p \"$REMOTE_TASK\" && find \"$REMOTE_TASK\" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +'")"
rc=$?
[ $rc -eq 10 ] && exit 10
[ $rc -ne 0 ] && { echo "ERROR: clean step failed (non-infra)" >&2; exit 1; }

# V3.7.8 (2026-05-02): per-lane + per-PID tar path to prevent shared-tar race under
# parallel-batch workloads. Prior hardcoded `/tmp/current_task_deploy.tar.gz` was
# clobbered between `tar czf` and `scp` when 4+ concurrent lanes ran deploys
# simultaneously — caused silent extract failures + exit-2 with empty stderr_marker
# (caught op#12 KvRmsnormRopeCache 2026-05-02; worker had to roll its own isolated
# deploy script with PID-tagged tar to work around). Fix: tag the local tar AND the
# remote scp target name with `${LANE:-0}_$$` so concurrent workers don't collide.
TAR_TAG="${LANE:-0}_$$"
TAR_LOCAL="${TMPDIR:-/tmp}/current_task_deploy_${TAR_TAG}.tar.gz"
TAR_NAME="current_task_deploy_${TAR_TAG}.tar.gz"

echo "[deploy/$TARGET] Packaging local $LOCAL_TASK/ → $TAR_LOCAL ..."
tar czf "$TAR_LOCAL" -C "$LOCAL_TASK" .

echo "[deploy/$TARGET] Ensuring stage dir $HOST:$DEPLOY_STAGE_HOST ..."
run_with_retry "ssh mkdir stage" "${SSH_CMD[@]}" "mkdir -p $DEPLOY_STAGE_HOST"
rc=$?
[ $rc -eq 10 ] && exit 10
[ $rc -ne 0 ] && { echo "ERROR: mkdir stage failed (non-infra)" >&2; exit 1; }

echo "[deploy/$TARGET] Uploading $TAR_NAME to $HOST:$DEPLOY_STAGE_HOST/ ..."
run_with_retry "scp upload" "${SCP_CMD[@]}" "$TAR_LOCAL" "$USER@$HOST:$DEPLOY_STAGE_HOST/$TAR_NAME"
rc=$?
[ $rc -eq 10 ] && exit 10
[ $rc -ne 0 ] && { echo "ERROR: upload step failed (non-infra)" >&2; exit 1; }

echo "[deploy/$TARGET] Extracting $TAR_NAME in $CONTAINER from $DEPLOY_STAGE_CONTAINER/ ..."
run_with_retry "ssh extract" "${SSH_CMD[@]}" "$(_wrap_sudo "docker exec $CONTAINER bash -c 'mkdir -p $REMOTE_TASK && cd $REMOTE_TASK && tar xzf $DEPLOY_STAGE_CONTAINER/$TAR_NAME && rm -f $DEPLOY_STAGE_CONTAINER/$TAR_NAME'")"
rc=$?
[ $rc -eq 10 ] && exit 10
[ $rc -ne 0 ] && { echo "ERROR: extract step failed (non-infra)" >&2; exit 1; }

# Verify file counts
LOCAL_COUNT=$(find "$LOCAL_TASK" -type f | grep -v __pycache__ | grep -v '.pyc' | wc -l)
REMOTE_COUNT=$("${SSH_CMD[@]}" "$(_wrap_sudo "docker exec $CONTAINER bash -c 'find $REMOTE_TASK -type f | grep -v __pycache__ | grep -v .pyc | grep -v build | wc -l'")" 2>/dev/null)
echo "[deploy/$TARGET] Verify: local=$LOCAL_COUNT files, remote=$REMOTE_COUNT files"
if [ "$LOCAL_COUNT" != "$REMOTE_COUNT" ]; then
    echo "WARNING: file count mismatch! Check for stale files." >&2
fi

# ─── Optional build ───────────────────────────────────────────────────
if [ "${1-}" = "--build" ]; then
    echo "[deploy/$TARGET] Building with SOC=$SOC_VERSION CANN=$CANN_PATH ..."

    # Per-target lib path quirk: A5 has $CANN_PATH/x86_64-linux/lib64 (CANN 9.0.T501 layout);
    # A3 has $CANN_PATH/lib64 directly (CANN 9.0.0 layout). Probe at runtime.
    # NPU_PYTHON_BIN (env / .ascendc_env): container python3 dir for non-login docker exec
    # (2026-06-09 npu_dev3: python3 lives in /root/miniconda3, conda-init only in
    # interactive .bashrc, so docker exec sees no python3 → exit 127 without this).
    # Source vendor set_env.sh when present (torch_npu autoload at `import torch` needs the
    # FULL CANN env — manual LD_LIBRARY_PATH exports alone fail at cmake torch probing).
    # NPU_PYTHON_BIN env (e.g. py311 on .171) ships cmake as a pip package, not
    # on default PATH → 'cmake not found' at build (2026-06-11). Add the env's
    # cmake/data/bin to PATH alongside the python bin.
    # DEBT-197 self-heal (remote/docker mode): mirror the local-mode provision — ensure
    # the full build baseline TRIO (build_ascendc.py + build_capabilities/ package +
    # cann_stubs/) exists inside the container before build. build_ascendc.py imports
    # build_capabilities and consumes cann_stubs; provisioning only the entrypoint left a
    # fresh container with `ModuleNotFoundError: No module named 'build_capabilities'`
    # surfacing as an empty-stderr phantom failure. GUARDED per-piece (only provisions a
    # piece ABSENT in the container → zero behavior change for prepared containers, so
    # it cannot regress existing working remote deploys) + FAIL-SOFT (a provision failure
    # falls through to the normal build error, never harden-breaks a working deploy).
    # Provisions the SHIPPED bundle-owned trio via scp -r (handles file + dir uniformly)
    # into a host stage dir, then docker cp into the container's utils/.
    _DEBT197_BUILD_ROOT="${BENCHMARK_ROOT:-/root/AscendOpGenAgent}"
    _DEBT197_STAGE="/tmp/_debt197_stage_$$"
    for _DEBT197_PIECE in build_ascendc.py build_capabilities cann_stubs; do
        _DEBT197_SRC="$SCRIPT_DIR/patches/$_DEBT197_PIECE"
        # Idempotent: skip pieces already present in the container.
        if "${SSH_CMD[@]}" "$(_wrap_sudo "docker exec $CONTAINER test -e $_DEBT197_BUILD_ROOT/utils/$_DEBT197_PIECE")" >/dev/null 2>&1; then
            continue
        fi
        # Nothing bundled to provision → fall through to the normal build error (fail-soft).
        [ -e "$_DEBT197_SRC" ] || continue
        "${SSH_CMD[@]}" "mkdir -p $_DEBT197_STAGE" >/dev/null 2>&1 \
          && "${SCP_CMD[@]}" -r "$_DEBT197_SRC" "$USER@$HOST:$_DEBT197_STAGE/$_DEBT197_PIECE" >/dev/null 2>&1 \
          && "${SSH_CMD[@]}" "$(_wrap_sudo "docker exec $CONTAINER mkdir -p $_DEBT197_BUILD_ROOT/utils && docker cp $_DEBT197_STAGE/$_DEBT197_PIECE $CONTAINER:$_DEBT197_BUILD_ROOT/utils/$_DEBT197_PIECE && rm -rf $_DEBT197_STAGE/$_DEBT197_PIECE")" >/dev/null 2>&1 \
          && echo "[deploy/$TARGET] DEBT-197 self-heal: provisioned bundle $_DEBT197_PIECE into $CONTAINER:$_DEBT197_BUILD_ROOT/utils/"
    done
    "${SSH_CMD[@]}" "rm -rf $_DEBT197_STAGE" >/dev/null 2>&1 || true
    _CMAKE_BIN="${NPU_PYTHON_BIN:+$(dirname $NPU_PYTHON_BIN)/lib/python3.11/site-packages/cmake/data/bin:}"
    BUILD_CMD="[ -f $CANN_PATH/set_env.sh ] && source $CANN_PATH/set_env.sh; export PATH=${NPU_PYTHON_BIN:+$NPU_PYTHON_BIN:}${_CMAKE_BIN}$CANN_PATH/bin:\$PATH && \
        if [ -d $CANN_PATH/x86_64-linux/lib64 ]; then export LD_LIBRARY_PATH=$CANN_PATH/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:\$LD_LIBRARY_PATH; \
        elif [ -d $CANN_PATH/aarch64-linux/lib64 ]; then export LD_LIBRARY_PATH=$CANN_PATH/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64:\$LD_LIBRARY_PATH; \
        else export LD_LIBRARY_PATH=$CANN_PATH/lib64:/usr/local/Ascend/driver/lib64:\$LD_LIBRARY_PATH; fi && \
        export ASCEND_HOME_PATH=$CANN_PATH && \
        export ASCEND_OPP_PATH=\$ASCEND_HOME_PATH/opp && \
        cd ${BENCHMARK_ROOT:-/root/AscendOpGenAgent} && \
        python3 utils/build_ascendc.py current_task -v $SOC_VERSION --build-type Release"

    run_with_retry "build" "${SSH_CMD[@]}" "$(_wrap_sudo "docker exec $CONTAINER bash -c '$BUILD_CMD'")"
    rc=$?

    # ─── Archive build artifacts (GEPA design §15.2) ──────────────────
    # Gated by env: BUILD_ARCHIVE_ENABLED=1 (default off — turned on once A5
    # archive root /data/build_archive/ is provisioned). Pure additive: when
    # disabled, no behavior change. On enable, every build (pass or fail)
    # uploads {kernel snapshot, stderr, meta.json} to A5 archive for GEPA
    # trace_serializer + the regen-sweep training corpus.
    # V1 limitations: stdout NOT captured (would need tee in docker exec).
    # rc=10 (infra) skips archive — nothing useful captured.
    #
    # V3.7.6 (2026-05-02): if BUILD_ARCHIVE_ENABLED=1 but ASCENDC_WORKSPACE is
    # NOT set (common when agent runs `bash deploy_to_npu_lane.sh ...` from a
    # fresh shell without `export X=Y &&` chain), warn loudly and skip archive
    # so users can see + fix the gap. Previously this silently skipped, which
    # caused op#10 ko-1's 5 perf-iter trajectories to be lost from build_archive
    # despite the optimizer brief setting BUILD_BATCH_ID. Caught 2026-05-02
    # via /aog-self-critic C13 verification.
    if [ "${BUILD_ARCHIVE_ENABLED:-0}" = "1" ] && [ -z "${ASCENDC_WORKSPACE:-}" ]; then
        echo "[deploy/$TARGET] ⚠️ BUILD_ARCHIVE_ENABLED=1 but ASCENDC_WORKSPACE empty — archive SKIPPED. Agent should export ASCENDC_WORKSPACE in same shell as deploy command (use 'export X=Y && bash deploy...' chain)." >&2
    fi
    if [ "${BUILD_ARCHIVE_ENABLED:-0}" = "1" ] && [ -n "${ASCENDC_WORKSPACE:-}" ] && [ "$rc" != "10" ]; then
        # P0i (Day 4 op#10 finding): ${var##*/} returns empty when var ends
        # with `/`. Caused op#10 archives to land at /build_archive//<batch>/
        # (orphan top-level, missing op-name level). Use `basename` which
        # handles trailing slashes correctly. Also fail-fast on empty slug
        # so the bug never silently writes to a malformed path.
        OP_SLUG="$(basename "$ASCENDC_WORKSPACE" 2>/dev/null)"
        if [ -z "$OP_SLUG" ] || [ "$OP_SLUG" = "/" ] || [ "$OP_SLUG" = "." ]; then
            echo "[deploy/$TARGET] ⚠️ BUILD_ARCHIVE_ENABLED=1 but ASCENDC_WORKSPACE='$ASCENDC_WORKSPACE' resolves to empty op slug — archive SKIPPED. Check trailing slash / pure-root path." >&2
            : # fall through to skip archive section
        else
        ARCHIVE_ROOT_CONTAINER="${BUILD_ARCHIVE_ROOT:-/data/build_archive}"
        BATCH_ID="${BUILD_BATCH_ID:-orch_$(date -u +%Y%m%dT%H%M%SZ)}"

        # On A5, /data/ inside the container is bind-mounted from the host's
        # /root/openeuler/data/. We compute iter on host (not via docker exec)
        # because the remote reference host reads its own mounted data path.
        case "$TARGET" in
            a5)    ARCHIVE_HOST_BASE="/root/openeuler${ARCHIVE_ROOT_CONTAINER}/$OP_SLUG/$BATCH_ID" ;;
            a3|a2) ARCHIVE_HOST_BASE="${ARCHIVE_ROOT_CONTAINER}/$OP_SLUG/$BATCH_ID" ;;
            *)     ARCHIVE_HOST_BASE="${ARCHIVE_ROOT_CONTAINER}/$OP_SLUG/$BATCH_ID" ;;
        esac

        NEXT_ITER=$("${SSH_CMD[@]}" "
            mkdir -p '$ARCHIVE_HOST_BASE' 2>/dev/null
            n=\$(ls -d '$ARCHIVE_HOST_BASE'/iter_* 2>/dev/null | wc -l)
            printf 'iter_%03d' \$((n + 1))
        " 2>/dev/null)
        NEXT_ITER="${NEXT_ITER:-iter_$(date -u +%H%M%S)}"
        ARCHIVE_DIR="$ARCHIVE_HOST_BASE/$NEXT_ITER"

        "${SSH_CMD[@]}" "mkdir -p '$ARCHIVE_DIR'" 2>/dev/null

        # Kernel snapshot from local LOCAL_TASK
        if [ -d "$LOCAL_TASK/kernel" ]; then
            tar czf "${TMPDIR:-/tmp}/.kernel_snapshot.tgz" -C "$LOCAL_TASK" kernel/ 2>/dev/null
            "${SCP_CMD[@]}" "${TMPDIR:-/tmp}/.kernel_snapshot.tgz" "$USER@$HOST:$ARCHIVE_DIR/kernel_snapshot.tgz" 2>/dev/null || true
            rm -f "${TMPDIR:-/tmp}/.kernel_snapshot.tgz"
        fi

        # stderr (run_with_retry populated STDERR_MARKER on failure; empty on success)
        if [ -s "$STDERR_MARKER" ]; then
            "${SCP_CMD[@]}" "$STDERR_MARKER" "$USER@$HOST:$ARCHIVE_DIR/stderr.log" 2>/dev/null || true
        fi

        case "$rc" in
            0)  ARCH_VERDICT="PASS";;
            1)  ARCH_VERDICT="COMPILE_ERROR";;
            *)  ARCH_VERDICT="UNKNOWN_$rc";;
        esac

        ARCH_META=$(cat <<META_EOF
{
  "op": "$OP_SLUG",
  "batch_id": "$BATCH_ID",
  "iter_id": "$NEXT_ITER",
  "ts_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "host": "$HOST",
  "container": "$CONTAINER",
  "soc": "$SOC_VERSION",
  "lane": "${LANE:-0}",
  "npu": "${ASCEND_VISIBLE_DEVICES:-0}",
  "candidate_brief_hash": "${CANDIDATE_BRIEF_HASH:-default}",
  "build_exit_code": $rc,
  "verdict": "$ARCH_VERDICT"
}
META_EOF
)
        "${SSH_CMD[@]}" "cat > '$ARCHIVE_DIR/meta.json' <<META_REMOTE
$ARCH_META
META_REMOTE" 2>/dev/null

        # Workspace breadcrumb for trace_serializer.py
        echo "$ARCHIVE_DIR" > "$_WS_DIR/.last_build_archive_path" 2>/dev/null || true
        echo "[deploy/$TARGET] Archived build to $ARCHIVE_DIR ($ARCH_VERDICT)"
        fi  # close P0i else (OP_SLUG non-empty)
    fi

    if [ $rc -ne 0 ]; then
        [ $rc -eq 10 ] && exit 10
        echo "ERROR: Build failed — see $STDERR_MARKER" >&2
        exit 1
    fi

    SO_CHECK=$("${SSH_CMD[@]}" "$(_wrap_sudo "docker exec $CONTAINER bash -c 'ls $REMOTE_TASK/kernel/build/*.so 2>/dev/null | wc -l'")" 2>/dev/null)
    if [ "$SO_CHECK" = "0" ]; then
        echo "ERROR: Build returned OK but no .so produced." >&2
        echo "compile" > "$CLASS_MARKER"
        echo "Build step produced 0 .so files" > "$STDERR_MARKER"
        exit 1
    fi
    echo "[deploy/$TARGET] Build OK (SOC=$SOC_VERSION)"
fi

# V3.7.7 (2026-05-02): on success, write "pass" sentinel into CLASS_MARKER (was: clear to empty).
# Reason: agent wait-loops use `[ -s class_file ]` (non-empty) check to know "result available";
# clearing on success caused those loops to hang indefinitely (caught op 10 ko-1 hung 20+ min on
# this exact pattern, 2026-05-02). With "pass" sentinel: file is non-empty (5 bytes) so wait-loop
# exits, agent reads "pass", knows build PASSed. STDERR_MARKER stays cleared on success since
# nothing useful to record there.
: > "$STDERR_MARKER"
echo "pass" > "$CLASS_MARKER"
# V3.7.8: clean up local per-PID tar
rm -f "$TAR_LOCAL"
echo "[deploy/$TARGET] Done."
