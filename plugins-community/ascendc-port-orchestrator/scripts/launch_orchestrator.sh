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
# Shared entry-layer launcher — the SINGLE place that knows how to start the engine.
#
# Both entry surfaces are thin wrappers over this script:
#   * Claude Code : skills/ascendc-{cross-gen-port,backward-gen}/SKILL.md
#   * opencode    : .opencode/command/ascendc-{cross-gen-port,backward-gen}.md
#
# Keeping resolve/validate/harness-detect/launch here is deliberate: the launch snippet used
# to be copy-pasted inside each SKILL.md, so adding a second harness would have produced a
# third divergent copy — and the parts most likely to drift (which config-dir variable to
# export, whether AOG_HARNESS_BACKEND is set at all) are exactly the parts whose failure is
# SILENT. A missing AOG_HARNESS_BACKEND does not error; it just falls back to Claude Code
# and spawns `claude` from inside an opencode session.
#
# Usage:
#   launch_orchestrator.sh --skill-base <dir> --mode port-a3-ops --source <ops-nn op dir> [--lane N]
#       (CANN ops 仓通用格式来源)
#   launch_orchestrator.sh --skill-base <dir> --mode port-a3-tilelang2ascendc \
#       --source <TileLang2AscendC project> [--lane N]
#       (TileLang2AscendC 插件输出格式来源)
#       [--reference-source npubench|a3_live|cannbench]
#       [--npubench-task <task.py> [--npubench-root <task-root>]]]
#       [--source-kind port-aclnn-tilelang2ascendc]
#   launch_orchestrator.sh --skill-base <dir> --mode backward --source <forward_spec.py> [--lane N]
#   launch_orchestrator.sh --skill-base <dir> --resume <op> [--lane N]
#
# Optional: --harness claude|opencode  (default: auto-detect, see resolve_harness below)
# Lifecycle: --cold-start archives an existing scoped workspace before starting a new run.
#   --resume <op> continues an interrupted scoped workspace.  In the engine CLI it is a
#   standalone lifecycle command (python3 -m orchestrator --resume <op>) whose operand is
#   the scoped workspace name, so it is mutually exclusive with --cold-start and with the
#   creation arguments (--mode/--source/reference/source-kind).
#
# The caller is responsible for streaming/backgrounding. This script execs the orchestrator
# in the foreground with unbuffered output so the host's task viewer sees it live.
set -euo pipefail

SKILL_BASE=""; MODE=""; SOURCE=""; LANE="0"; HARNESS=""
REFERENCE_SOURCE=""
NPUBENCH_TASK=""; NPUBENCH_ROOT=""
SOURCE_KIND=""; SOURCE_ARCH=""; CANDIDATE_KIND=""
COLD_START=0; RESUME=""

while [ $# -gt 0 ]; do
  case "$1" in
    --skill-base) SKILL_BASE="${2:-}"; shift 2 ;;
    --mode)       MODE="${2:-}";       shift 2 ;;
    --source)     SOURCE="${2:-}";     shift 2 ;;
    --lane)       LANE="${2:-0}";      shift 2 ;;
    --harness)    HARNESS="${2:-}";    shift 2 ;;
    --reference-source)      REFERENCE_SOURCE="${2:-}";      shift 2 ;;
    --npubench-task)         NPUBENCH_TASK="${2:-}";         shift 2 ;;
    --npubench-root)         NPUBENCH_ROOT="${2:-}";         shift 2 ;;
    --source-kind)           SOURCE_KIND="${2:-}";           shift 2 ;;
    --source-arch)           SOURCE_ARCH="${2:-}";           shift 2 ;;
    --candidate-kind)        CANDIDATE_KIND="${2:-}";        shift 2 ;;
    --cold-start)            COLD_START=1;                    shift ;;
    --resume)
      RESUME="${2:-}"
      [ -n "$RESUME" ] || { echo "launch_orchestrator: --resume requires a scoped workspace name (op)" >&2; exit 2; }
      shift 2 ;;
    -h|--help)
      sed -n '/^# Usage:/,/^# Optional:/p' "$0"; exit 0 ;;
    *) echo "launch_orchestrator: unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$SKILL_BASE" ] || { echo "launch_orchestrator: --skill-base is required" >&2; exit 2; }

# --resume is a standalone engine lifecycle command (python3 -m orchestrator
# --resume <op>), not a modifier of a new run: the engine CLI rejects it in
# combination with --port-a3-ops/--backward/--cold-start.  Mirror that here and
# fail closed instead of silently dropping creation arguments.
if [ -n "$RESUME" ]; then
  [ "$COLD_START" -eq 0 ] || {
    echo "launch_orchestrator: --resume and --cold-start are mutually exclusive" >&2; exit 2; }
  if [ -n "$MODE$SOURCE$REFERENCE_SOURCE$NPUBENCH_TASK$NPUBENCH_ROOT$SOURCE_KIND$SOURCE_ARCH$CANDIDATE_KIND" ]; then
    echo "launch_orchestrator: --resume cannot be combined with --mode/--source/reference/source-kind arguments" >&2
    exit 2
  fi
else
  [ -n "$SOURCE" ] || { echo "launch_orchestrator: --source is required" >&2; exit 2; }
fi

# Mode names encode the source format category:
#   port-a3-ops              = CANN ops 仓通用格式（arch22 ops-nn 源算子目录）
#   port-a3-tilelang2ascendc = TileLang2AscendC 插件输出格式（model_new_ascendc.py + kernel/）

# The friendly mode is an explicit profile alias over the existing
# port_a3_to_a5 state machine.  Keep the durable source kind distinct from
# both direct-launch and the existing ops-nn ACLNN route.
if [ "$MODE" = "port-a3-tilelang2ascendc" ]; then
  if [ -n "$SOURCE_KIND" ] && [ "$SOURCE_KIND" != "port-aclnn-tilelang2ascendc" ] && [ "$SOURCE_KIND" != "port_aclnn_tilelang2ascendc" ]; then
    echo "launch_orchestrator: port-a3-tilelang2ascendc mode cannot be combined with a different --source-kind" >&2
    exit 2
  fi
  MODE="port-a3-ops"
  SOURCE_KIND="port-aclnn-tilelang2ascendc"
fi
if [ -z "$RESUME" ]; then
case "$MODE" in
  port-a3-ops|backward) ;;
  *) echo "launch_orchestrator: --mode must be port-a3-ops, port-a3-tilelang2ascendc, or backward (got '${MODE}')" >&2; exit 2 ;;
esac
fi

# Reference-provider arguments are accepted only for port-a3-ops mode.  The
# provider must be chosen explicitly: npubench (with --npubench-task) for the
# frozen NPUKernelBench golden, or a3_live for a fresh A3 CANN capture.  A bare
# invocation without --reference-source fails closed in the engine CLI.
if [ -n "$REFERENCE_SOURCE$NPUBENCH_TASK$NPUBENCH_ROOT" ]; then
  [ "$MODE" = "port-a3-ops" ] || {
    echo "launch_orchestrator: reference arguments require --mode port-a3-ops" >&2; exit 2; }
  case "$REFERENCE_SOURCE" in
    ""|a3_live|npubench|cannbench) ;;
    *) echo "launch_orchestrator: --reference-source must be npubench, a3_live, or cannbench" >&2; exit 2 ;;
  esac
  case "$REFERENCE_SOURCE" in
    "")
      if [ -n "$NPUBENCH_TASK$NPUBENCH_ROOT" ]; then
        echo "launch_orchestrator: --npubench-task/--npubench-root require --reference-source npubench" >&2; exit 2
      fi
      ;;
    a3_live)
      if [ -n "$NPUBENCH_TASK$NPUBENCH_ROOT" ]; then
        echo "launch_orchestrator: a3_live cannot be combined with NPUKernelBench arguments" >&2; exit 2
      fi
      ;;
    npubench)
      [ -n "$NPUBENCH_TASK" ] || {
        echo "launch_orchestrator: --reference-source npubench requires --npubench-task" >&2; exit 2; }
      ;;
    cannbench)
      if [ -n "$NPUBENCH_TASK$NPUBENCH_ROOT" ]; then
        echo "launch_orchestrator: reserved cannbench cannot be combined with provider input arguments" >&2; exit 2
      fi
      ;;
  esac
fi

if [ -n "$SOURCE_KIND$SOURCE_ARCH$CANDIDATE_KIND" ]; then
  [ "$MODE" = "port-a3-ops" ] || {
    echo "launch_orchestrator: source-kind arguments require --mode port-a3-ops" >&2; exit 2; }
  if [ "$SOURCE_KIND" = "port-aclnn-tilelang2ascendc" ] || [ "$SOURCE_KIND" = "port_aclnn_tilelang2ascendc" ]; then
    if [ -n "$SOURCE_ARCH" ] && [ "$SOURCE_ARCH" != "arch35" ]; then
      echo "launch_orchestrator: TileLang2AscendC source architecture must be arch35" >&2
      exit 2
    fi
    if [ -n "$CANDIDATE_KIND" ] && [ "$CANDIDATE_KIND" != "tilelang2ascendc_custom_op" ]; then
      echo "launch_orchestrator: TileLang2AscendC candidate kind must be tilelang2ascendc_custom_op" >&2
      exit 2
    fi
    [ "$REFERENCE_SOURCE" = "npubench" ] && [ -n "$NPUBENCH_TASK" ] || {
      echo "launch_orchestrator: TileLang2AscendC requires --reference-source npubench and --npubench-task" >&2
      exit 2
    }
  else
    echo "launch_orchestrator: unsupported --source-kind: $SOURCE_KIND (only port-aclnn-tilelang2ascendc is supported)" >&2; exit 2
  fi
fi

# --- resolve the engine (never guess from cwd, never search the filesystem) ---------------
PLUGIN_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLVER="$PLUGIN_SCRIPTS/resolve_engine.py"
[ -f "$RESOLVER" ] || { echo "launch_orchestrator: resolver missing: $RESOLVER" >&2; exit 2; }
ENGINE_DIR="$(python3 "$RESOLVER" --base-dir "$SKILL_BASE")" || exit $?
[ -f "$ENGINE_DIR/src/scripts/orchestrator/__main__.py" ] || {
  echo "launch_orchestrator: resolved engine has no orchestrator: $ENGINE_DIR" >&2; exit 2; }
[ -f "$ENGINE_DIR/workspace/.ascendc_env" ] || {
  echo "launch_orchestrator: configure $ENGINE_DIR/workspace/.ascendc_env first (docs/USAGE.md §2)" >&2
  exit 2; }
# The resolved plugin workspace is the launch contract.  Do not let a stale
# host-level ASCENDC_ENV_PATH/FILE silently redirect a marketplace-installed
# run to another target (in particular, turning the documented local target
# into an unrelated SSH target).  Remote validation remains an explicit edit
# of this file: set A5_HOST and replace A5_CONTAINER=local there.
export ASCENDC_ENV_PATH="$ENGINE_DIR/workspace/.ascendc_env"
export ASCENDC_ENV_FILE="$ENGINE_DIR/workspace/.ascendc_env"

# --- which harness is driving us? --------------------------------------------------------
# Order: explicit flag > already-exported backend > **fingerprint of the harness we are
# RUNNING INSIDE** > install manifest > claude_code.
#
# The running-host fingerprint MUST outrank the install manifest. An earlier version asked
# the manifests first, in the fixed order opencode-then-claude, which answers "what was
# installed on this machine" — not "what is driving this process". On a box where the user
# had ever run `init.sh global opencode`, EVERY Claude Code invocation then resolved to
# opencode, and every consequence of that is silent: AOG_HARNESS_BACKEND=opencode makes the
# dispatch sites bind OpencodeBackend and spawn `opencode run` from inside a CC session,
# CLAUDE_CONFIG_DIR stops being exported (breaking the self-containment rule this script
# exists to enforce), and Phase O0 skips the Claude Code hook-registration check entirely.
resolve_harness() {
  if [ -n "$HARNESS" ]; then echo "$HARNESS"; return; fi
  if [ -n "${AOG_HARNESS_BACKEND:-}" ]; then echo "$AOG_HARNESS_BACKEND"; return; fi

  # Running-host fingerprints. MEASURED on opencode 1.18.18: a `opencode run` child sees
  # exactly `OPENCODE=1` and `OPENCODE_PID=<pid>`. OPENCODE_CONFIG / OPENCODE_CONFIG_DIR /
  # OPENCODE_CONFIG_CONTENT are INPUT variables a user or caller may export anywhere — they
  # are not evidence that opencode is driving us. Keying off them got it wrong in both
  # directions: a real opencode session (which sets neither) resolved to claude_code, and a
  # Claude Code session belonging to a user who exports OPENCODE_CONFIG in their shell
  # profile resolved to opencode.
  if [ -n "${OPENCODE:-}${OPENCODE_PID:-}" ]; then
    echo opencode; return
  fi
  if [ -n "${CLAUDECODE:-}${CLAUDE_CODE_ENTRYPOINT:-}" ]; then
    echo claude_code; return
  fi

  # No fingerprint (e.g. invoked from a bare shell): fall back to what was installed. Only
  # consult a manifest whose config root exists, and prefer the CC one when both are present
  # so the historical default stays Claude Code.
  local manifest tool
  for manifest in "${CLAUDE_CONFIG_DIR:-$HOME/.claude}/cannbot-manifest.json" \
                  "$HOME/.config/opencode/cannbot-manifest.json"; do
    [ -f "$manifest" ] || continue
    tool="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('tool',''))" \
            "$manifest" 2>/dev/null || true)"
    case "$tool" in
      opencode) echo opencode; return ;;
      claude)   echo claude_code; return ;;
    esac
  done
  echo claude_code
}

case "$(resolve_harness)" in
  opencode|open_code|open-code) BACKEND="opencode" ;;
  codex|codex_cli|codex-cli)    BACKEND="codex" ;;
  *)                            BACKEND="claude_code" ;;
esac
export AOG_HARNESS_BACKEND="$BACKEND"

# The 9 orchestrator dispatch sites bind `_backend = get_backend()` at MODULE IMPORT time,
# so the selection must be in the environment BEFORE the orchestrator process starts — it
# cannot be switched afterwards. That is why this is exported here and not inside Python.
if [ "$BACKEND" = "claude_code" ]; then
  # Self-containment: the spawned worker resolves skills/agents/KB from CLAUDE_CONFIG_DIR.
  # If an upstream layer dropped it, the worker would silently fall back to ~/.claude and
  # could pick up a different operator-generation suite.
  export CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
  # Anthropic-compatible endpoints (including Kimi Coding) do not use the
  # interactive Claude login flow.  Fail before O2/O4 if the private key was
  # not injected, instead of letting a graybox worker report a late and
  # ambiguous "Not logged in" error.  Ordinary Claude deployments without an
  # explicit endpoint retain their existing login behavior.
  if [ -n "${ANTHROPIC_BASE_URL:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "launch_orchestrator: ANTHROPIC_BASE_URL is set but ANTHROPIC_API_KEY is empty; inject the key through private environment management" >&2
    exit 2
  fi
fi

case "$MODE" in
  port-a3-ops) FLAG="--port-a3-ops" ;;
  backward)    FLAG="--backward" ;;
esac

# Resolve --source against the CALLER's cwd before cd: a relative --source would
# otherwise be interpreted relative to the engine dir, sending the orchestrator at a
# path the caller never meant (or a nonexistent one).
case "$SOURCE" in
  /*) : ;;
  *) [ -z "$SOURCE" ] || SOURCE="$(pwd)/$SOURCE" ;;
esac

# --resume switches the invocation form to the engine lifecycle command; its
# operand is the scoped workspace name, so no source path is resolved or passed.
RUN_ARGS=()
if [ -n "$RESUME" ]; then
  RUN_ARGS=(--resume "$RESUME")
else
  RUN_ARGS=("$FLAG" "$SOURCE")
fi

# NPUKernelBench task paths have the same caller-cwd semantics as --source.
# Resolve both before ``cd "$ENGINE_DIR"`` so a relative task/root remains
# anchored at the location from which the user invoked the entry skill.
if [ -n "$NPUBENCH_TASK" ]; then
  case "$NPUBENCH_TASK" in
    /*) : ;;
    *) NPUBENCH_TASK="$(pwd)/$NPUBENCH_TASK" ;;
  esac
fi
if [ -n "$NPUBENCH_ROOT" ]; then
  case "$NPUBENCH_ROOT" in
    /*) : ;;
    *) NPUBENCH_ROOT="$(pwd)/$NPUBENCH_ROOT" ;;
  esac
fi

REFERENCE_ARGS=()
LIFECYCLE_ARGS=()
if [ "$COLD_START" -eq 1 ]; then
  LIFECYCLE_ARGS+=(--cold-start)
fi
if [ "$SOURCE_KIND" = "port-aclnn-tilelang2ascendc" ] || [ "$SOURCE_KIND" = "port_aclnn_tilelang2ascendc" ]; then
  REFERENCE_ARGS+=(--source-kind "$SOURCE_KIND")
  if [ -n "$SOURCE_ARCH" ]; then
    REFERENCE_ARGS+=(--source-arch "$SOURCE_ARCH")
  fi
  if [ -n "$CANDIDATE_KIND" ]; then
    REFERENCE_ARGS+=(--candidate-kind "$CANDIDATE_KIND")
  fi
fi
if [ -n "$REFERENCE_SOURCE" ]; then
  REFERENCE_ARGS+=(--reference-source "$REFERENCE_SOURCE")
fi
if [ -n "$NPUBENCH_TASK" ]; then
  REFERENCE_ARGS+=(--npubench-task "$NPUBENCH_TASK")
  if [ -n "$NPUBENCH_ROOT" ]; then
    REFERENCE_ARGS+=(--npubench-root "$NPUBENCH_ROOT")
  fi
fi
echo "[launch] harness=$BACKEND mode=${MODE:-resume} lane=$LANE engine=$ENGINE_DIR source=${SOURCE:-$RESUME}"
cd "$ENGINE_DIR"

# A provider outage must not leave the foreground port command retrying forever.
# Keep the watchdog in the shared launcher (rather than relying only on a usage
# snippet), and let callers tune the bounded deadline for unusually large ops.
# ``--foreground`` keeps terminal signals visible to the orchestrator; the
# kill-after grace period gives its marker/evidence finally blocks a chance to
# persist an honest interrupted state before the hard stop.
if ! command -v timeout >/dev/null 2>&1; then
  echo "launch_orchestrator: GNU timeout is required for bounded execution" >&2
  exit 2
fi
PORT_TIMEOUT_SEC="${CANNBOT_PORT_TIMEOUT_SEC:-5400}"
case "$PORT_TIMEOUT_SEC" in
  ''|*[!0-9]*)
    echo "launch_orchestrator: CANNBOT_PORT_TIMEOUT_SEC must be a positive integer" >&2
    exit 2
    ;;
esac
if [ "$PORT_TIMEOUT_SEC" -le 0 ]; then
  echo "launch_orchestrator: CANNBOT_PORT_TIMEOUT_SEC must be a positive integer" >&2
  exit 2
fi
exec timeout --foreground --kill-after=30s "${PORT_TIMEOUT_SEC}s" \
  env PYTHONUNBUFFERED=1 PYTHONPATH=src/scripts \
  python3 -m orchestrator "${RUN_ARGS[@]}" --lane "$LANE" \
  "${LIFECYCLE_ARGS[@]}" "${REFERENCE_ARGS[@]}"
