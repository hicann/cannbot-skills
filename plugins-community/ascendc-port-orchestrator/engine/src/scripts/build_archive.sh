#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# build_archive.sh — per-iter capture of build stdout/stderr + kernel snapshot to A5 archive
#
# Wraps a build invocation: runs the build remotely, captures both streams + kernel snapshot
# to /data/build_archive/<op>/<batch>/<iter>/ on A5 host. Returns the build's exit code.
#
# Usage (called from deploy_to_npu_lane.sh after the existing build command):
#
#   bash src/scripts/build_archive.sh \
#     --ssh-cmd "$SSH_CMD" \
#     --remote-task "$REMOTE_TASK" \
#     --build-cmd "$BUILD_CMD" \
#     --op-slug "$ASCENDC_WORKSPACE" \
#     --soc "$SOC_VERSION" \
#     --lane "${LANE:-0}" \
#     --npu "${ASCEND_VISIBLE_DEVICES:-0}" \
#     --workspace-marker "$_WS_DIR/.last_build_archive_path"
#
# Optional env: BUILD_ARCHIVE_ROOT (default /data/build_archive)
#               BUILD_BATCH_ID (default orch_<UTC ts>)
#               CANDIDATE_BRIEF_HASH (default "default")
#
# Standalone test (without integration):
#   BUILD_ARCHIVE_ROOT=/tmp/build_archive_test bash src/scripts/build_archive.sh --dry-run
#
# Created 2026-05-02 per docs/research/GEPA_skill_design_2026_05_02.md §15.2

set -uo pipefail

# ─── parse args ─────────────────────────────────────────────────────────
SSH_CMD=""
REMOTE_TASK=""
BUILD_CMD=""
OP_SLUG=""
SOC=""
LANE="0"
NPU="0"
WORKSPACE_MARKER=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --ssh-cmd)           SSH_CMD="$2"; shift 2;;
        --remote-task)       REMOTE_TASK="$2"; shift 2;;
        --build-cmd)         BUILD_CMD="$2"; shift 2;;
        --op-slug)           OP_SLUG="$2"; shift 2;;
        --soc)               SOC="$2"; shift 2;;
        --lane)              LANE="$2"; shift 2;;
        --npu)               NPU="$2"; shift 2;;
        --workspace-marker)  WORKSPACE_MARKER="$2"; shift 2;;
        --dry-run)           DRY_RUN=1; shift;;
        *)                   echo "unknown arg: $1" >&2; exit 2;;
    esac
done

# Strip trailing slash from OP_SLUG path; take basename
OP_SLUG="${OP_SLUG##*/}"
OP_SLUG="${OP_SLUG:-unknown_op}"

ARCHIVE_ROOT="${BUILD_ARCHIVE_ROOT:-/data/build_archive}"
BATCH_ID="${BUILD_BATCH_ID:-orch_$(date -u +%Y%m%dT%H%M%SZ)}"
BRIEF_HASH="${CANDIDATE_BRIEF_HASH:-default}"

if [ "$DRY_RUN" = "1" ]; then
    cat <<EOF
[dry-run] Would archive build for:
  archive_root: $ARCHIVE_ROOT
  op_slug:      $OP_SLUG
  batch_id:     $BATCH_ID
  brief_hash:   $BRIEF_HASH
  lane:         $LANE
  npu:          $NPU
  soc:          $SOC
  build_cmd:    $BUILD_CMD
EOF
    exit 0
fi

if [ -z "$SSH_CMD" ] || [ -z "$REMOTE_TASK" ] || [ -z "$BUILD_CMD" ]; then
    echo "[build_archive] ERROR: missing required args (--ssh-cmd, --remote-task, --build-cmd)" >&2
    exit 2
fi

# ─── compute next iter_id atomically on remote ──────────────────────────
ARCHIVE_DIR_BASE="$ARCHIVE_ROOT/$OP_SLUG/$BATCH_ID"

# Use shell arithmetic on remote to find the next iter number.
# Race-safe-ish: ls + wc on existing dirs. Two concurrent builds on same op+batch
# can collide, but that's a degenerate case (same op, same batch_id, parallel)
# and we'd just lose one iter's logs.
NEXT_ITER=$($SSH_CMD "
    mkdir -p '$ARCHIVE_DIR_BASE' 2>/dev/null
    n=\$(ls -d '$ARCHIVE_DIR_BASE'/iter_* 2>/dev/null | wc -l)
    printf 'iter_%03d' \$((n + 1))
" 2>/dev/null)

if [ -z "$NEXT_ITER" ]; then
    echo "[build_archive] WARN: could not compute iter_id; falling back to ts-based" >&2
    NEXT_ITER="iter_$(date -u +%H%M%S)"
fi

ARCHIVE_DIR="$ARCHIVE_DIR_BASE/$NEXT_ITER"

# ─── prepare archive dir on remote ──────────────────────────────────────
$SSH_CMD "mkdir -p '$ARCHIVE_DIR'" 2>/dev/null

# ─── run build, capture both streams ────────────────────────────────────
# We run the build inside the docker container (already implicit in $BUILD_CMD)
# and capture into the archive dir on the host filesystem (which is bind-mounted
# to /data inside the container — both paths see the same files).
START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Pass exit code back via $?; capture into ARCHIVE_DIR/exit_code on remote
$SSH_CMD "
    cd '$REMOTE_TASK' 2>/dev/null
    ($BUILD_CMD) > '$ARCHIVE_DIR/stdout.log' 2> '$ARCHIVE_DIR/stderr.log'
    rc=\$?
    echo \$rc > '$ARCHIVE_DIR/exit_code'
    exit \$rc
"
BUILD_RC=$?

END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# ─── snapshot kernel files ──────────────────────────────────────────────
# Tar the kernel/ directory at build-time to ARCHIVE_DIR/kernel_snapshot.tgz
$SSH_CMD "
    cd '$REMOTE_TASK' 2>/dev/null
    if [ -d kernel ]; then
        tar czf '$ARCHIVE_DIR/kernel_snapshot.tgz' kernel/ 2>/dev/null
    fi
" 2>/dev/null

# ─── classify exit code into a one-line verdict ─────────────────────────
case "$BUILD_RC" in
    0)   VERDICT="PASS";;
    1)   VERDICT="COMPILE_ERROR";;
    2)   VERDICT="USAGE_ERROR";;
    10)  VERDICT="INFRA_ERROR";;
    *)   VERDICT="UNKNOWN_$BUILD_RC";;
esac

# ─── compress logs if large (>100 KB) ───────────────────────────────────
$SSH_CMD "
    for f in '$ARCHIVE_DIR/stdout.log' '$ARCHIVE_DIR/stderr.log'; do
        if [ -f \"\$f\" ] && [ \$(stat -c%s \"\$f\" 2>/dev/null || echo 0) -gt 102400 ]; then
            gzip \"\$f\" 2>/dev/null
        fi
    done
"

# ─── write meta.json ────────────────────────────────────────────────────
META_JSON=$(cat <<EOF
{
  "op": "$OP_SLUG",
  "batch_id": "$BATCH_ID",
  "iter_id": "$NEXT_ITER",
  "ts_start_utc": "$START_TS",
  "ts_end_utc": "$END_TS",
  "host": "${HOST:-unknown}",
  "container": "${CONTAINER:-unknown}",
  "soc": "$SOC",
  "lane": "$LANE",
  "npu": "$NPU",
  "candidate_brief_hash": "$BRIEF_HASH",
  "build_exit_code": $BUILD_RC,
  "verdict": "$VERDICT"
}
EOF
)
$SSH_CMD "cat > '$ARCHIVE_DIR/meta.json' <<'METAEOF'
$META_JSON
METAEOF" 2>/dev/null

# ─── workspace-side breadcrumb (so trace_serializer can find the dir) ───
if [ -n "$WORKSPACE_MARKER" ]; then
    echo "$ARCHIVE_DIR" > "$WORKSPACE_MARKER" 2>/dev/null || true
fi

# Convenience symlinks (preserve existing /tmp/ascendc_last_build.* contract)
$SSH_CMD "
    ln -sf '$ARCHIVE_DIR/stderr.log' /tmp/ascendc_last_build.stderr 2>/dev/null || \
    ln -sf '$ARCHIVE_DIR/stderr.log.gz' /tmp/ascendc_last_build.stderr 2>/dev/null
    ln -sf '$ARCHIVE_DIR/stdout.log' /tmp/ascendc_last_build.stdout 2>/dev/null || \
    ln -sf '$ARCHIVE_DIR/stdout.log.gz' /tmp/ascendc_last_build.stdout 2>/dev/null
" 2>/dev/null

# ─── stdout summary line ────────────────────────────────────────────────
echo "[build_archive] $OP_SLUG/$BATCH_ID/$NEXT_ITER -> $VERDICT (rc=$BUILD_RC)"

exit $BUILD_RC
