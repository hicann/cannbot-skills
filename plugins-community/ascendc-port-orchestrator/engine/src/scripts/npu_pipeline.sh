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
# npu_pipeline.sh — Multi-target execution substrate for AscendC kernels (V3.4)
# Wraps deploy, build, precision test, benchmark, and msprof profiling.
# Outputs JSON for machine parsing by QA skill.
#
# This script is target-agnostic: it sources `resolve_target.sh` to pick up
# HOST / CONTAINER / CANN_PATH / SOC_VERSION based on $TARGET in the active
# .ascendc_env. The legacy `a5_pipeline.sh` is now a thin wrapper that
# forces TARGET=a5 before delegating here.
#
# Override TARGET on the command line:
#   TARGET=a3 bash src/scripts/npu_pipeline.sh build
#   TARGET=a5 bash src/scripts/npu_pipeline.sh full ./npu_benchmark /tmp/data
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/resolve_target.sh"

# Pick the right exec wrapper per target.
# a5  → a5_op skill (~/.claude/skills/a5_op/scripts/a5_exec.py)
# a3  → a3_op skill (~/.claude/skills/a3_op/scripts/a3_exec.py)
# a2  → a2_op skill (when A2 server is wired up)
case "$TARGET" in
    a5)
        EXEC_SCRIPT="$HOME/.claude/skills/a5_op/scripts/a5_exec.py"
        REMOTE_DIR="${REMOTE_DIR:-/root/a5_ops}"
        ;;
    a3)
        EXEC_SCRIPT="$HOME/.claude/skills/a3_op/scripts/a3_exec.py"
        # A3 container remote workspace on a persistent mount (not the container overlay).
        # Config-driven: honor REMOTE_DIR, then A3_REMOTE_DIR (genericize — set for your A3
        # container), then the a5 default fallback. Overridable so a scrubbed / non-default
        # deployment sets its own without touching the source literal.
        REMOTE_DIR="${REMOTE_DIR:-${A3_REMOTE_DIR:-/data2/npu_user/workspace/a5_ops_remote}}"
        ;;
    a2)
        EXEC_SCRIPT="$HOME/.claude/skills/a2_op/scripts/a2_exec.py"
        REMOTE_DIR="${REMOTE_DIR:-/root/a5_ops}"
        ;;
esac

if [ ! -x "$EXEC_SCRIPT" ] && [ ! -f "$EXEC_SCRIPT" ]; then
    echo "npu_pipeline: missing exec wrapper $EXEC_SCRIPT for TARGET=$TARGET" >&2
    echo "  Install the matching {a5,a3,a2}_op skill or set TARGET to one that has the skill installed." >&2
    exit 1
fi

# Drivers under CANN 9.0 differ from 8.x — the set_env path is the same,
# but ASCEND_HOME_PATH may default to a baked-in CANN inside the container.
# Always pin it to $CANN_PATH when entering the container.
SETENV="export ASCEND_HOME_PATH=$CANN_PATH && source $CANN_PATH/set_env.sh && export ASCEND_HOME_PATH=$CANN_PATH"

# CANN driver lib path varies a bit per generation — use the one in the
# install if present, else fall back to /usr/local/Ascend/driver/lib64.
DRIVER_LIB="${DRIVER_LIB:-/usr/local/Ascend/driver/lib64}"

usage() {
    cat <<EOF
Usage: $0 <command> [options]

Active config (from .ascendc_env, TARGET=$TARGET):
  HOST=$HOST
  CONTAINER=$CONTAINER
  CANN_PATH=$CANN_PATH
  SOC_VERSION=$SOC_VERSION
  PLATFORM_SIMT=$PLATFORM_SIMT (UB/AIV=${UB_PER_AIV_KB}KB, arch=$ARCH_CODE)
  REMOTE_DIR=$REMOTE_DIR

Commands:
  deploy                     Package, upload, and build on remote container
  build                      Build only (assumes code already deployed)
  precision <binary> <data> <dtype>  Run precision tests, emit JSON
  benchmark <binary> <data>  Run performance benchmark, emit JSON
  profile <binary>           Run msprof profiling, emit JSON summary
  full <binary> <data>       Run precision + benchmark + profile
  health                     Container health check (temp, zombies, disk)

Override TARGET on the CLI:
  TARGET=a3 $0 build
EOF
    exit 1
}

remote() {
    "$EXEC_SCRIPT" "$1" 2>&1
}

# ============================================================
# deploy: package + upload + build
# ============================================================
cmd_deploy() {
    echo "=== Deploy to $TARGET ($HOST / $CONTAINER) ==="

    echo "Packaging..."
    tar czf /tmp/a5_ops.tar.gz -C "$REPO_DIR" --exclude=.git --exclude=CLAUDE.md .

    echo "Uploading..."
    case "$TARGET" in
        a5)
            scp -o StrictHostKeyChecking=no /tmp/a5_ops.tar.gz "root@$HOST:/tmp/"
            ssh -o StrictHostKeyChecking=no "root@$HOST" \
                "docker cp /tmp/a5_ops.tar.gz $CONTAINER:/tmp/ && \
                 docker exec $CONTAINER bash -c 'rm -rf $REMOTE_DIR && mkdir -p $REMOTE_DIR && cd $REMOTE_DIR && tar xzf /tmp/a5_ops.tar.gz'"
            ;;
        a3|a2)
            # A3 uses key-based SSH; container has /data2 mounted, so we tar
            # into /data2 directly (no docker cp needed — bind-mount sees it).
            scp -o StrictHostKeyChecking=no /tmp/a5_ops.tar.gz "root@$HOST:/tmp/"
            ssh -o StrictHostKeyChecking=no "root@$HOST" \
                "docker exec $CONTAINER bash -c 'mkdir -p $REMOTE_DIR && cd $REMOTE_DIR && tar xzf /tmp/a5_ops.tar.gz' && cp /tmp/a5_ops.tar.gz \"$(dirname \"$REMOTE_DIR\")/a5_ops.tar.gz\" 2>/dev/null || true"
            ;;
    esac

    echo "Building..."
    cmd_build

    echo "=== Deploy complete ==="
}

# ============================================================
# build: cmake + make on remote
# ============================================================
cmd_build() {
    echo "=== Build for $TARGET (SOC=$SOC_VERSION) ==="
    local output rc
    output=$(remote "$SETENV && cd $REMOTE_DIR && rm -rf build && mkdir build && cd build && cmake .. -DRUN_MODE=npu -DSOC_VERSION=$SOC_VERSION -DASCEND_CANN_PACKAGE_PATH=$CANN_PATH 2>&1 && make -j\$(nproc) 2>&1") || rc=$?
    rc=${rc:-0}

    if echo "$output" | grep -q "Built target a5ops_kernels_npu"; then
        echo '{"build": "SUCCESS", "target": "'$TARGET'", "soc": "'$SOC_VERSION'"}'
        # Also build benchmark binary
        remote "cd $REMOTE_DIR && g++ -std=c++17 -O2 tests/npu_benchmark.cpp -I$CANN_PATH/include -L$CANN_PATH/lib64 -L$DRIVER_LIB -lascendcl -L./build/lib -la5ops_kernels_npu -Wl,-rpath,$CANN_PATH/lib64:./build/lib:$DRIVER_LIB -o npu_benchmark -lm 2>&1" >/dev/null
        echo "Benchmark binary built."
    else
        echo '{"build": "FAIL", "target": "'$TARGET'", "soc": "'$SOC_VERSION'"}'
        echo "$output" | tail -30
        return 1
    fi
}

# ============================================================
# precision: run precision tests, emit JSON
# ============================================================
cmd_precision() {
    local binary="${1:?binary required}" data="${2:?data path required}" dtype="${3:-fp32}"
    echo "=== Precision Test ($dtype, $TARGET) ==="

    local output
    output=$(remote "cd $REMOTE_DIR && ./$binary $data --dtype $dtype 2>&1")

    local pass_count fail_count
    pass_count=$(echo "$output" | grep -c "PASS" || true)
    fail_count=$(echo "$output" | grep -c "FAIL" || true)

    cat <<JSONEOF
{
  "test": "precision",
  "target": "$TARGET",
  "dtype": "$dtype",
  "pass_count": $pass_count,
  "fail_count": $fail_count,
  "gate": "$([ "$fail_count" -eq 0 ] && echo "PASS" || echo "FAIL")",
  "raw_output_tail": "$(echo "$output" | tail -30 | sed 's/"/\\"/g' | tr '\n' '|')"
}
JSONEOF
    echo ""
    echo "$output"
}

# ============================================================
# benchmark: run timing, emit JSON
# ============================================================
cmd_benchmark() {
    local binary="${1:?binary required}" data="${2:?data path required}"
    echo "=== Performance Benchmark ($TARGET) ==="

    local output
    output=$(remote "cd $REMOTE_DIR && ./$binary $data 2>&1")

    local fwd_ms bwd_ms
    fwd_ms=$(echo "$output" | grep "Forward:" | tail -1 | grep -oP '[\d.]+(?= ms)')
    bwd_ms=$(echo "$output" | grep "Backward:" | tail -1 | grep -oP '[\d.]+(?= ms)')

    cat <<JSONEOF
{
  "test": "benchmark",
  "target": "$TARGET",
  "forward_ms": "${fwd_ms:-N/A}",
  "backward_ms": "${bwd_ms:-N/A}",
  "raw_output_tail": "$(echo "$output" | tail -20 | sed 's/"/\\"/g' | tr '\n' '|')"
}
JSONEOF
    echo ""
    echo "$output"
}

# ============================================================
# profile: run msprof, parse op_statistic + op_summary, emit JSON
# ============================================================
cmd_profile() {
    local binary="${1:?binary required}"
    local prof_dir="/tmp/msprof_pipeline_$$"
    echo "=== msprof Profiling ($TARGET) ==="

    # msprof location varies by CANN gen; try a couple
    local msprof_bin
    for candidate in \
        "$CANN_PATH/tools/profiler/bin/msprof" \
        "$CANN_PATH/aarch64-linux/tools/profiler/bin/msprof" \
        "$CANN_PATH/x86_64-linux/tools/profiler/bin/msprof"; do
        if remote "test -x $candidate && echo OK" | grep -q OK; then
            msprof_bin="$candidate"
            break
        fi
    done
    if [ -z "${msprof_bin:-}" ]; then
        echo '{"test": "profile", "error": "msprof not found in any known CANN layout"}'
        return 1
    fi

    remote "rm -rf $prof_dir && export LD_LIBRARY_PATH=$CANN_PATH/lib64:\$LD_LIBRARY_PATH && $msprof_bin --output=$prof_dir -- cd $REMOTE_DIR && ./$binary 2>&1" || true

    echo "--- op_statistic ---"
    local stat
    stat=$(remote "cat $prof_dir/PROF_*/mindstudio_profiler_output/op_statistic_*.csv 2>/dev/null || echo 'NO_PROFILE_DATA'")
    echo "$stat"

    echo "--- op_summary (key metrics) ---"
    local summary
    summary=$(remote "grep -v '^Device' $prof_dir/PROF_*/mindstudio_profiler_output/op_summary_*.csv 2>/dev/null | awk -F',' '{printf \"%s: dur=%.1fus vec=%.3f scl=%.3f mte=%.3f\\n\", \$5, \$10, \$38, \$40, \$42}' | head -30 || echo 'NO_SUMMARY_DATA'")
    echo "$summary"

    cat <<JSONEOF
{
  "test": "profile",
  "target": "$TARGET",
  "msprof": "$msprof_bin",
  "op_statistic": "$(echo "$stat" | head -5 | sed 's/"/\\"/g' | tr '\n' '|')",
  "op_summary_sample": "$(echo "$summary" | head -10 | sed 's/"/\\"/g' | tr '\n' '|')"
}
JSONEOF
}

# ============================================================
# health: container health check
# ============================================================
cmd_health() {
    echo "=== Container Health Check ($TARGET / $CONTAINER) ==="

    echo "--- NPU Status ---"
    remote "npu-smi info 2>/dev/null | head -20 || echo 'npu-smi not available'"

    echo "--- Zombie Processes ---"
    local zombies
    zombies=$(remote "ps aux | grep -c defunct || echo 0")
    echo "Zombie count: $zombies"

    echo "--- Disk Space ---"
    remote "df -h /tmp | tail -1"

    cat <<JSONEOF
{
  "test": "health",
  "target": "$TARGET",
  "container": "$CONTAINER",
  "zombie_count": "$zombies"
}
JSONEOF
}

# ============================================================
# full: precision + benchmark + profile
# ============================================================
cmd_full() {
    local binary="${1:?binary required}" data="${2:?data path required}"
    cmd_health
    echo ""
    cmd_precision "$binary" "$data" fp32
    echo ""
    cmd_precision "$binary" "$data" fp16
    echo ""
    cmd_precision "$binary" "$data" bf16
    echo ""
    cmd_benchmark "$binary" "$data"
    echo ""
    cmd_profile "$binary"
}

# Main dispatch
[ $# -lt 1 ] && usage
CMD="$1"; shift

case "$CMD" in
    deploy)    cmd_deploy "$@" ;;
    build)     cmd_build "$@" ;;
    precision) cmd_precision "$@" ;;
    benchmark) cmd_benchmark "$@" ;;
    profile)   cmd_profile "$@" ;;
    full)      cmd_full "$@" ;;
    health)    cmd_health "$@" ;;
    *)         echo "Unknown command: $CMD"; usage ;;
esac
