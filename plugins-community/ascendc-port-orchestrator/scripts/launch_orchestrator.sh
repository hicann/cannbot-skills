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
#   launch_orchestrator.sh --skill-base <dir> --mode port-a3  --source <ops-nn op dir> [--lane N]
#   launch_orchestrator.sh --skill-base <dir> --mode backward --source <forward_spec.py> [--lane N]
#
# Optional: --harness claude|opencode  (default: auto-detect, see resolve_harness below)
#
# The caller is responsible for streaming/backgrounding. This script execs the orchestrator
# in the foreground with unbuffered output so the host's task viewer sees it live.
set -euo pipefail

SKILL_BASE=""; MODE=""; SOURCE=""; LANE="0"; HARNESS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --skill-base) SKILL_BASE="${2:-}"; shift 2 ;;
    --mode)       MODE="${2:-}";       shift 2 ;;
    --source)     SOURCE="${2:-}";     shift 2 ;;
    --lane)       LANE="${2:-0}";      shift 2 ;;
    --harness)    HARNESS="${2:-}";    shift 2 ;;
    -h|--help)
      sed -n '12,32p' "$0"; exit 0 ;;
    *) echo "launch_orchestrator: unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$SKILL_BASE" ] || { echo "launch_orchestrator: --skill-base is required" >&2; exit 2; }
[ -n "$SOURCE" ]     || { echo "launch_orchestrator: --source is required" >&2; exit 2; }
case "$MODE" in
  port-a3|backward) ;;
  *) echo "launch_orchestrator: --mode must be port-a3 or backward (got '${MODE}')" >&2; exit 2 ;;
esac

# --- resolve the engine (never guess from cwd, never search the filesystem) ---------------
PLUGIN_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLVER="$PLUGIN_SCRIPTS/resolve_engine.py"
[ -f "$RESOLVER" ] || { echo "launch_orchestrator: resolver missing: $RESOLVER" >&2; exit 2; }
ENGINE_DIR="$(python3 "$RESOLVER" --base-dir "$SKILL_BASE")" || exit $?
[ -f "$ENGINE_DIR/src/scripts/orchestrator/__main__.py" ] || {
  echo "launch_orchestrator: resolved engine has no orchestrator: $ENGINE_DIR" >&2; exit 2; }
[ -f "$ENGINE_DIR/workspace/.ascendc_env" ] || {
  echo "launch_orchestrator: configure $ENGINE_DIR/workspace/.ascendc_env first (docs/USAGE.md §3)" >&2
  exit 2; }

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
fi

case "$MODE" in
  port-a3)  FLAG="--port-a3" ;;
  backward) FLAG="--backward" ;;
esac

# Resolve --source against the CALLER's cwd before cd: a relative --source would
# otherwise be interpreted relative to the engine dir, sending the orchestrator at a
# path the caller never meant (or a nonexistent one).
case "$SOURCE" in
  /*) : ;;
  *) SOURCE="$(pwd)/$SOURCE" ;;
esac
echo "[launch] harness=$BACKEND mode=$MODE lane=$LANE engine=$ENGINE_DIR source=$SOURCE"
cd "$ENGINE_DIR"
exec env PYTHONUNBUFFERED=1 PYTHONPATH=src/scripts \
  python3 -m orchestrator "$FLAG" "$SOURCE" --lane "$LANE"
