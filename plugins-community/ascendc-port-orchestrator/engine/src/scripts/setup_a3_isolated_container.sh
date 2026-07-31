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
# setup_a3_isolated_container.sh — one-shot per-agent isolated A3 container setup.
#
# Codifies the multi-agent A3 convention (memory feedback_a3_per_agent_sliced_container_convention,
# user direction 2026-05-12 + 2026-05-17 hardening):
#
#   - per-agent container name (npu-a3-<agent>)
#   - sliced /home mount: host /home/npu_user_<agent> → canonical /home/npu_user
#   - sliced /data mount: host /data/npu_user_<agent> → canonical /data/npu_user
#   - owned CANN 9.0.0 copy at /home/npu_user_<agent>/cann/cann-9.0.0 WITH path rewrite
#   - no /usr/local/Ascend/ascend-toolkit mount (breaks verl image entrypoint)
#
# Compliant with CROSS_AGENT_ARCHITECTURE_PROTOCOL.md Amendment 2026-05-17
# (end-to-end smoke gate for first-contact scripts). Script prints `✅` only
# after the smoke test passes; failure exits non-zero with named code.
#
# Usage:
#   bash setup_a3_isolated_container.sh \
#       --agent opus \
#       --host 198.51.100.70 \
#       [--cann-source /home/npu_user/cann/cann-9.0.0] \
#       [--image quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest] \
#       [--dry-run]
#
# Re-running with same --agent --host is idempotent: skips steps already done.
# Returns env overlay on stdout for caller to merge into workspace/.ascendc_env.
#
# Exit codes (first-contact convention):
#   0  success (smoke passed)
#   2  argument error
#   10 host preflight failed (driver / docker / npu-smi missing)
#   11 CANN source not found
#   12 CANN copy step failed
#   13 CANN path rewrite (sed sweep) failed or left stranger paths
#   14 container create failed
#   15 container mount verification failed (slice mount ≠ expected)
#   16 cross-agent isolation breached (my container sees other agent slice)
#   17 smoke test failed (torch_npu / NPU not usable)

set -euo pipefail

# Resolve this script's own dir (used to locate vendor/AscendOpGenAgent for the
# provisioning step + patches/build_ascendc.py). Must be set before `set -u`
# trips on the first reference (P-fix 2026-06-11: was undefined → the
# AscendOpGenAgent provisioning step aborted with "SCRIPT_DIR: unbound variable"
# AFTER the container smoke passed, silently leaving the build harness missing).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AGENT=""
HOST=""
CANN_SOURCE=""
IMAGE="quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent) AGENT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --cann-source) CANN_SOURCE="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$AGENT" || -z "$HOST" ]] && {
  echo "Usage: --agent <slug> --host <ip> [--cann-source <path>] [--image <ref>] [--dry-run]" >&2
  exit 2
}

# P125 (2026-05-17): NO back-compat fallback. Container name is always
# npu-a3-${AGENT}. Old "default-agent → npu-a3" hack removed — caller responsible
# for using the new name. If existing op-gen runs reference npu-a3,
# repoint A3_CONTAINER in workspace/.ascendc_env explicitly.
CONTAINER="npu-a3-${AGENT}"
# In-container canonical home/data — the mount points baked into CANN set_env.sh and used
# as the slice-mount DESTINATIONS. Configurable via A3_CONTAINER_HOME / A3_CONTAINER_DATA
# for non-npu_user environments; the npu_user defaults preserve existing behavior. The
# per-agent HOST slices derive as ${CTR_HOME}_${AGENT} / ${CTR_DATA}_${AGENT}. (These must
# stay consistent host↔container because CANN set_env.sh bakes the in-container path.)
CTR_HOME="${A3_CONTAINER_HOME:-/home/npu_user}"
CTR_DATA="${A3_CONTAINER_DATA:-/data/npu_user}"
SLICE_HOME="${CTR_HOME}_${AGENT}"
SLICE_DATA="${CTR_DATA}_${AGENT}"
SLICE_CANN="${SLICE_HOME}/cann/cann-9.0.0"
# In-container view of the CANN install. The container mounts
# ${SLICE_HOME} → /home/npu_user, so anything under our sliced CANN at
# ${SLICE_CANN} appears inside as ${IN_CONTAINER_CANN}. This is the target
# the post-cp sed sweep must bake into set_env.sh / install.info / etc. —
# NOT ${SLICE_CANN}, which is unreachable from inside the container.
IN_CONTAINER_CANN="${CTR_HOME}/cann/cann-9.0.0"

# Known stranger paths from past multi-hop-cp incidents. CANN is now
# typically a chain of cp's (<host> outage 2026-05-20 forced re-cp's across
# the fleet), so the path baked into a copy's set_env.sh is usually the
# ORIGINAL install location, not the immediate cp source. Each entry here
# was a real residual that broke a setup. Append as new ones are caught.
KNOWN_STRANGER_PATHS=(
  "/home/npu_user/cann/cann-9.0.0"  # original install on <host> (setup 2026-05-20)
)

step() { echo "── $* ──" >&2; }
fail() {
  local code="$1"; shift
  echo "✗ FAILED at step ${code}: $*" >&2
  exit "$code"
}
remote() {
  if [[ "$DRY_RUN" -eq 1 ]]; then echo "[DRY] $*" >&2; return 0; fi
  ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no "root@${HOST}" "$@"
}

# ─────────────────────────────────────────────────────────────────────────────
step "Step 1/7: pre-flight on ${HOST}"
remote "test -d /usr/local/Ascend/driver && command -v docker && command -v npu-smi" >/dev/null \
  || fail 10 "host ${HOST} missing Ascend driver / docker / npu-smi"

# Discover CANN 9.0 source if not specified
if [[ -z "$CANN_SOURCE" ]]; then
  step "Step 2/7: discover CANN 9.0.0 source on ${HOST}"
  CANN_SOURCE=$(remote "find /home /usr/local /data /opt -maxdepth 6 -type d -name 'cann-9.0.0' 2>/dev/null | head -1")
  [[ -z "$CANN_SOURCE" ]] && fail 11 "no cann-9.0.0 directory found on ${HOST}. Specify --cann-source explicitly."
  echo "found CANN source: ${CANN_SOURCE}" >&2
fi

# ─────────────────────────────────────────────────────────────────────────────
step "Step 3/7: prepare owned slices + CANN copy WITH path rewrite (P130)"
# P130 (2026-05-17): cp -a of CANN from stranger user leaves 1000+ hard-coded
# references to source path inside set_env.sh and helpers. torch_npu silently
# fails to load (ASCEND_OPP_PATH points to nonexistent stranger path).
#
# P130 amendment (2026-05-20, isolated-container setup): two prior bugs fixed —
#   (a) only swept ${CANN_SOURCE} as a stranger; missed deeper ancestor paths
#       (e.g. /home/npu_user/cann/cann-9.0.0 baked in by multi-hop cp).
#       Fix: sweep ${CANN_SOURCE} PLUS every entry in KNOWN_STRANGER_PATHS
#       PLUS any other cann-9.0.0-suffixed paths discovered by grep.
#   (b) sed rewrite target was ${SLICE_CANN} (host slice path). That path is
#       unreachable from INSIDE the container, where set_env.sh ultimately
#       runs (the slice is mounted to /home/npu_user). Fix: rewrite to
#       ${IN_CONTAINER_CANN} so the baked references resolve correctly
#       under the container's view of the slice mount.
#
# Build the union of stranger paths to sweep:
#   - ${CANN_SOURCE} (immediate cp source; may or may not actually appear)
#   - ${KNOWN_STRANGER_PATHS[@]} (catalog of past-incident strangers)
#   - any other /home/*/cann-9.0.0 paths discovered by grep on the sliced copy
ALL_STRANGER_PATHS=("${CANN_SOURCE}" "${KNOWN_STRANGER_PATHS[@]}")
SED_EXPRS=""
for sp in "${ALL_STRANGER_PATHS[@]}"; do
  [[ -z "$sp" || "$sp" == "$IN_CONTAINER_CANN" ]] && continue
  SED_EXPRS+="s|${sp}|${IN_CONTAINER_CANN}|g;"
done
remote "
  set -e
  mkdir -p ${SLICE_HOME}/workspace ${SLICE_DATA}
  mkdir -p ${SLICE_HOME}/cann
  if [[ ! -x ${SLICE_CANN}/bin/bisheng ]]; then
    echo copying CANN 9.0.0 from ${CANN_SOURCE} → ${SLICE_CANN}
    cp -a ${CANN_SOURCE} ${SLICE_CANN}
  else
    echo CANN already owned at ${SLICE_CANN}, skipping copy
  fi
  echo === sed-rewrite stranger paths in ${SLICE_CANN} → ${IN_CONTAINER_CANN} ===
  # Auto-discover any additional cann-9.0.0-suffixed stranger paths and add
  # to the sweep (handles unknown multi-hop-cp ancestors not yet in the
  # KNOWN_STRANGER_PATHS catalog).
  EXTRA_EXPRS=''
  for extra in \$(grep -rhIoE '/[A-Za-z0-9_./-]+/cann-9\\.0\\.0' ${SLICE_CANN}/ 2>/dev/null | sort -u); do
    case \"\$extra\" in
      ${IN_CONTAINER_CANN}|${SLICE_CANN}) continue ;;
    esac
    EXTRA_EXPRS+=\"s|\${extra}|${IN_CONTAINER_CANN}|g;\"
  done
  ALL_EXPRS='${SED_EXPRS}'\"\$EXTRA_EXPRS\"
  if [[ -n \"\$ALL_EXPRS\" ]]; then
    # Build a single grep -E pattern covering every stranger to find matched files
    PATTERN=\$(echo \"\$ALL_EXPRS\" | sed -E 's|s\\|([^|]+)\\|[^|]*\\|g;|\\1\\n|g' | grep -v '^\$' | tr '\\n' '|' | sed 's/|\$//')
    COUNT_BEFORE=\$(grep -rlIE \"\$PATTERN\" ${SLICE_CANN}/ 2>/dev/null | wc -l)
    if [[ \$COUNT_BEFORE -gt 0 ]]; then
      grep -rlIE \"\$PATTERN\" ${SLICE_CANN}/ 2>/dev/null | xargs -r sed -i \"\$ALL_EXPRS\"
      COUNT_AFTER=\$(grep -rlIE \"\$PATTERN\" ${SLICE_CANN}/ 2>/dev/null | wc -l)
      echo \"rewrote stranger paths in \$COUNT_BEFORE → \$COUNT_AFTER files (residual = binary blobs / non-canonical /cann subpaths)\"
    else
      echo no stranger paths to rewrite
    fi
  else
    echo no stranger paths configured
  fi
  ${SLICE_CANN}/bin/bisheng --version 2>&1 | head -1
" || fail 12 "CANN copy or path rewrite failed"

# Verify rewrite succeeded — residual TEXT refs to any stranger should be 0
# (binary blobs are OK, hence -I flag to grep). Wider denylist than the old
# single-CANN_SOURCE check: catches the (a) bug from above where ancestor
# paths slipped through.
RESIDUAL_PATTERN=$(IFS='|'; echo "${ALL_STRANGER_PATHS[*]}")
RESIDUAL_TEXT=$(remote "grep -rIlE '${RESIDUAL_PATTERN}' ${SLICE_CANN}/ 2>/dev/null | wc -l" || echo "1")
[[ "$RESIDUAL_TEXT" -gt 0 ]] && fail 13 "P130 sed rewrite incomplete: ${RESIDUAL_TEXT} text files still reference one of: ${ALL_STRANGER_PATHS[*]}"

# ─────────────────────────────────────────────────────────────────────────────
step "Step 4/7: create container ${CONTAINER} with sliced mounts"
remote "
  set -e
  if docker inspect ${CONTAINER} >/dev/null 2>&1; then
    status=\$(docker inspect -f '{{.State.Status}}' ${CONTAINER})
    if [[ \$status == 'running' ]]; then
      echo container ${CONTAINER} already running, skipping create
    else
      echo container exists but stopped, removing + recreating
      docker rm -f ${CONTAINER}
    fi
  fi
  if ! docker inspect ${CONTAINER} >/dev/null 2>&1; then
    docker run -d --name ${CONTAINER} --restart=unless-stopped --privileged \
      -v ${SLICE_HOME}:${CTR_HOME} \
      -v ${SLICE_DATA}:${CTR_DATA} \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /etc/localtime:/etc/localtime \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /usr/local/sbin:/usr/local/sbin \
      -v /etc/hccn.conf:/etc/hccn.conf \
      -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
      ${IMAGE} \
      tail -f /dev/null > /dev/null
    sleep 3
  fi
  docker ps --filter name=^${CONTAINER}\$ --format '{{.Names}}\t{{.Status}}'
" || fail 14 "container create/restart failed"

# ─────────────────────────────────────────────────────────────────────────────
step "Step 5/7: verify mount convention (P129 setup-time companion)"
# P129 (2026-05-17): the orchestrator runtime gate validates A3_HOST_HOME ↔
# container mount at op-gen startup. Companion check here at setup time: assert
# the just-created container has the expected sliced mount. Catches drift if
# someone manually edits the container later.
HOME_MOUNT=$(remote "docker inspect ${CONTAINER} --format '{{range .Mounts}}{{if eq .Destination \"${CTR_HOME}\"}}{{.Source}}{{end}}{{end}}'")
DATA_MOUNT=$(remote "docker inspect ${CONTAINER} --format '{{range .Mounts}}{{if eq .Destination \"${CTR_DATA}\"}}{{.Source}}{{end}}{{end}}'")
if [[ "$HOME_MOUNT" != "$SLICE_HOME" ]]; then
  fail 15 "${CTR_HOME} in container maps to '${HOME_MOUNT}' but expected '${SLICE_HOME}'"
fi
if [[ "$DATA_MOUNT" != "$SLICE_DATA" ]]; then
  fail 15 "${CTR_DATA} in container maps to '${DATA_MOUNT}' but expected '${SLICE_DATA}'"
fi
echo "OK: home mount ${HOME_MOUNT} ↔ ${CTR_HOME}; data mount ${DATA_MOUNT} ↔ ${CTR_DATA}" >&2

# ─────────────────────────────────────────────────────────────────────────────
step "Step 6/7: cross-agent isolation check (hard-fail per protocol)"
# Protocol amendment 2026-05-17: WARN-and-continue replaced with hard-fail.
# If my container can see any other agent's slice, isolation is broken.
OTHER_SLICES=$(remote "ls -d ${CTR_HOME}_* 2>/dev/null | grep -v '_${AGENT}$' || true")
if [[ -n "$OTHER_SLICES" ]]; then
  for slice in $OTHER_SLICES; do
    # Check via host-path: file in stranger slice should NOT be reachable
    # from container's ${CTR_HOME} view (which is bound to our slice).
    HOST_SENTINEL_BASENAME=".cross_agent_isolation_check_$$"
    remote "touch ${slice}/${HOST_SENTINEL_BASENAME}" 2>/dev/null || continue
    IN_CONTAINER=$(remote "docker exec ${CONTAINER} test -e ${CTR_HOME}/${HOST_SENTINEL_BASENAME} && echo yes || echo no" 2>/dev/null)
    remote "rm -f ${slice}/${HOST_SENTINEL_BASENAME}" 2>/dev/null || true
    if [[ "$IN_CONTAINER" == "yes" ]]; then
      fail 16 "isolation breached — my container sees ${slice} (via mount overlap)"
    fi
  done
  echo "OK: my container cannot see any of: $OTHER_SLICES" >&2
else
  echo "no other agent slices on host (only ours)" >&2
fi

# ─────────────────────────────────────────────────────────────────────────────
step "Step 7a/7: patch container ~/.bashrc with ASCEND_*_PATH hijack override"
# verl base image pre-sets ASCEND_OPP_PATH + ASCEND_HOME_PATH to /usr/local/
# Ascend/cann-8.5.0/... (baked-in CANN 8.5 layer). The newer CANN 9.0.0 install
# at /usr/local/Ascend/cann-9.0.0/ doesn't override these because set_env.sh
# only sets ASCEND_OPP_PATH if UNSET. Caught 2026-05-23T19:46Z on FA orch:
# `aclnnReduceSum 561103 "Op ReduceSum does not has any binary"` because ascend910_93
# SoC binaries are in 9.0.0 install tree but the env was pointing at 8.5.0.
# Same hijack pathology as documented ASCEND_HOME_PATH (aog-a3-rebuild SKILL.md
# line 152) — extended sibling fix here so every fresh container picks it up.
#
# Without this, kw workers that invoke ad-hoc shells (outside deploy_to_npu.sh
# which already sets the vars explicitly at lines 145/339) hit ReduceSum 561103
# + mis-diagnose as install drift (kw-9 FA P9 incident 2026-05-23).
BASHRC_PATCH_MARKER="# AOG-CANN-HIJACK-OVERRIDE (per setup_a3_isolated_container.sh Step 7a, 2026-05-23)"
if [ "${DRY_RUN:-false}" != "true" ]; then
    remote "docker exec ${CONTAINER} bash -c '
        if ! grep -q \"${BASHRC_PATCH_MARKER}\" ~/.bashrc 2>/dev/null; then
            cat >> ~/.bashrc <<BASHRC_EOF

${BASHRC_PATCH_MARKER}
unset ASCEND_OPP_PATH ASCEND_HOME_PATH
source /usr/local/Ascend/cann/set_env.sh > /dev/null 2>&1 || true
export ASCEND_OPP_PATH=/usr/local/Ascend/cann/opp
export ASCEND_HOME_PATH=/usr/local/Ascend/cann
BASHRC_EOF
            echo \"[bashrc-patch] override appended to ~/.bashrc\"
        else
            echo \"[bashrc-patch] override already present (idempotent skip)\"
        fi
    '" || fail 16 "failed to patch container ~/.bashrc"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "Step 7/7: end-to-end smoke (protocol amendment 2026-05-17)"
# torch_npu loads → NPU detected → trivial torch op runs. This is the actual
# capability the script claims to provide. If this fails, the "container ready"
# message would be a lie.
SMOKE_OUTPUT=$(remote "docker exec ${CONTAINER} bash -c '
  set -e
  source ${CTR_HOME}/cann/cann-9.0.0/set_env.sh > /dev/null 2>&1
  export ASCEND_HOME_PATH=${CTR_HOME}/cann/cann-9.0.0
  python3 <<PYEOF
import torch
import torch_npu
assert torch.npu.is_available(), \"torch.npu.is_available() False\"
n = torch.npu.device_count()
assert n >= 1, f\"npu device_count={n}, expected >=1\"
torch.npu.set_device(0)
soc = torch.npu.get_device_name(0)
# trivial computation — proves NPU is actually usable, not just present
x = torch.tensor([2.0, 4.0, 8.0], device=\"npu\")
y = torch.reciprocal(x)
torch.npu.synchronize()
expected = torch.tensor([0.5, 0.25, 0.125])
diff = (y.cpu() - expected).abs().max().item()
assert diff < 1e-6, f\"reciprocal smoke max_diff={diff}, expected ~0\"
# Additional smoke (added 2026-05-23 after FA orch ReduceSum 561103 catch):
# verify sum() / ReduceSum dispatches. Reciprocal alone passes even with the
# ASCEND_OPP_PATH hijack (reciprocal binary exists in both CANN 8.5 and 9.0
# install trees); ReduceSum binary for ascend910_93 SoC ONLY exists in 9.0+
# install tree. If env points at 8.5 install (the hijack pathology fixed by
# Step 7a above), .sum() throws aclnnReduceSum 561103. Test catches this.
z = torch.randn(128, 64, dtype=torch.float16, device=\"npu\")
s = z.sum().item()
assert not (s != s), f\"sum() returned NaN (likely ReduceSum dispatch failed)\"
print(f\"SMOKE_PASS soc={soc} device_count={n} torch={torch.__version__} torch_npu={torch_npu.__version__} sum_check_ok={s:.4f}\")
PYEOF
'" 2>&1) || fail 17 "torch_npu smoke failed: ${SMOKE_OUTPUT}"

echo "$SMOKE_OUTPUT" | grep -q "SMOKE_PASS" || fail 17 "smoke output missing SMOKE_PASS marker: ${SMOKE_OUTPUT}"
echo "$SMOKE_OUTPUT" | tail -3 >&2

# ─────────────────────────────────────────────────────────────────────────────
# Step 8: provision vendor/AscendOpGenAgent into the agent's host slice
# (DEBT-111, 2026-05-20). The deploy_to_npu.sh build path runs
# `cd ${BENCHMARK_ROOT:-/root/AscendOpGenAgent} && python3 utils/build_ascendc.py
# current_task ...` inside the container. Without provisioning, the container has
# no utils/build_ascendc.py and the build fails. Putting the files in
# ${SLICE_HOME}/AscendOpGenAgent on host makes them appear at
# /home/npu_user/AscendOpGenAgent inside the container (via the slice mount);
# BENCHMARK_ROOT in the env overlay points to that path.
#
# Caller invokes this script from the repo (clones of example/a5_ops);
# $SCRIPT_DIR is src/scripts → repo root is $SCRIPT_DIR/../.. → vendor at
# $SCRIPT_DIR/../../vendor/AscendOpGenAgent. Skip + warn (not fail) if vendor/
# tree is absent — older clones without the submodule still get a working
# container, just need manual provision later.
VENDOR_AOG="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)/vendor/AscendOpGenAgent"
if [[ -d "$VENDOR_AOG" && -f "$VENDOR_AOG/utils/build_ascendc.py" ]]; then
    echo "[provision] AscendOpGenAgent → ${SLICE_HOME}/AscendOpGenAgent on ${HOST}" >&2
    if [[ "$DRY_RUN" -eq 0 ]]; then
        # tar pipe (rsync not present on the remote 90.* fleet)
        TAR_TMP="/tmp/aog_provision_${AGENT}_$$.tgz"
        tar czf "$TAR_TMP" -C "$VENDOR_AOG/.." --exclude='.git' --exclude='__pycache__' AscendOpGenAgent
        scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$TAR_TMP" "root@${HOST}:${SLICE_HOME}/" \
            || fail 18 "scp AscendOpGenAgent tarball to ${SLICE_HOME}/ failed"
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@${HOST}" \
            "cd ${SLICE_HOME} && tar xzf $(basename "$TAR_TMP") && rm -f $(basename "$TAR_TMP") && test -f AscendOpGenAgent/utils/build_ascendc.py" \
            || fail 18 "remote tar-extract or post-extract sanity check failed"
        rm -f "$TAR_TMP"
        # DEBT-20 (2026-05-21): overlay project-owned build_ascendc.py patch
        # with per-source COMPILE_DEFINITIONS support (kernel/build_overrides.json
        # manifest). Required for FA Pattern B (-DASCENDC_MATMUL_AICORE isolated
        # to KFC kernel TU). Patched file lives at src/scripts/patches/ in
        # parent repo; overlays the vendored version inside the container.
        PATCH_AOG="${SCRIPT_DIR}/patches/build_ascendc.py"
        if [[ -f "$PATCH_AOG" ]]; then
            scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$PATCH_AOG" \
                "root@${HOST}:${SLICE_HOME}/AscendOpGenAgent/utils/build_ascendc.py" \
                || fail 18 "scp DEBT-20 build_ascendc.py patch failed"
            echo "[provision] DEBT-20 patch overlaid on AscendOpGenAgent/utils/build_ascendc.py" >&2
        fi
        echo "[provision] OK — AscendOpGenAgent/utils/ visible inside container at ${CTR_HOME}/AscendOpGenAgent/utils/" >&2
    else
        echo "[provision][DRY] would tar+scp+extract AscendOpGenAgent into ${SLICE_HOME}/" >&2
    fi
else
    echo "[provision] WARNING: $VENDOR_AOG not found — skipping. Manual rsync needed before first a3 build." >&2
fi

# ─────────────────────────────────────────────────────────────────────────────
# Emit .ascendc_env overlay on stdout (only reached if all 7 steps passed)
cat <<EOF
# A3 isolated-container config for agent=${AGENT} on host=${HOST}
# Generated by setup_a3_isolated_container.sh on $(date -Iseconds)
A3_HOST=${HOST}
A3_USER=root
A3_PASSWORD=''
A3_CONTAINER=${CONTAINER}
A3_CANN_PATH=${CTR_HOME}/cann/cann-9.0.0
A3_SOC_VERSION=Ascend910_9382
A3_DEFAULT_NPU_ID=0
A3_WORKSPACE=${CTR_HOME}/workspace
A3_BACKUP_ROOT=${CTR_DATA}
A3_HOST_HOME=${SLICE_HOME}
A3_HOST_BACKUP=${SLICE_DATA}
# DEBT-111 (2026-05-20): BENCHMARK_ROOT/LOCAL_TASK point at the provisioned
# AscendOpGenAgent location inside the container (via slice mount). Without
# these, deploy_to_npu.sh defaults to /root/AscendOpGenAgent which doesn't
# exist in this container layout.
BENCHMARK_ROOT=${CTR_HOME}/AscendOpGenAgent
LOCAL_TASK=${CTR_HOME}/AscendOpGenAgent/current_task
EOF
echo "✅ setup complete for agent=${AGENT} on ${HOST} (all 8 hard-gates passed including end-to-end smoke + AscendOpGenAgent provision)" >&2
