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

set -euo pipefail

MODE="link"; ACTION="install"
for arg in "$@"; do
  case "$arg" in
    --copy)      MODE="copy" ;;
    --uninstall) ACTION="uninstall" ;;
    -h|--help)   grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $arg (try --help)" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # skills/
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "$REPO_ROOT" ] || REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEST_SKILLS="$REPO_ROOT/.opencode/skills"
DEST_AGENTS="$REPO_ROOT/.opencode/agents"
MARKER=".cannbotdsl-managed"

log() { printf '  %s\n' "$*"; }

remove_managed() {
  local t="$1"
  if [ -L "$t" ]; then
    rm -f "$t"; return 0
  fi
  if [ -d "$t" ] && [ -e "$t/$MARKER" ]; then
    rm -rf "$t"; return 0
  fi
  if [ -f "$t" ] && [ -e "$t.$MARKER" ]; then
    rm -f "$t" "$t.$MARKER"; return 0
  fi
  if [ -e "$t" ]; then
    log "SKIP (not managed by installer): ${t#$REPO_ROOT/}"; return 1
  fi
  return 0
}

install_one() {   # $1=src  $2=dest
  local src="$1" dest="$2"
  remove_managed "$dest" || return 0
  rm -f "$dest.$MARKER"
  if [ "$MODE" = "link" ]; then
    ln -s "$src" "$dest"
  else
    cp -r "$src" "$dest"
    if [ -d "$dest" ]; then touch "$dest/$MARKER"; else touch "$dest.$MARKER"; fi
  fi
  log "${dest#$REPO_ROOT/}"
}

if [ "$ACTION" = "uninstall" ]; then
  echo "Uninstalling cannbot-dsl skills from $REPO_ROOT/.opencode ..."
  for d in "$DEST_SKILLS"/cannbotdsl-* "$DEST_AGENTS"/cannbotdsl-*.md; do
    [ -e "$d" ] || [ -L "$d" ] || continue
    remove_managed "$d" && log "removed ${d#$REPO_ROOT/}" || true
  done
  # Also remove the orchestrator entry
  if [ -e "$DEST_SKILLS/cannbotdsl-op-orchestrator" ] || [ -L "$DEST_SKILLS/cannbotdsl-op-orchestrator" ]; then
    remove_managed "$DEST_SKILLS/cannbotdsl-op-orchestrator" && log "removed cannbotdsl-op-orchestrator" || true
  fi
  echo "Done."
  exit 0
fi

echo "Installing cannbot-dsl skills into $REPO_ROOT/.opencode ($MODE mode) ..."
mkdir -p "$DEST_SKILLS" "$DEST_AGENTS"

# 1) Skills: skills/{core,debug,op}-skills/cannbotdsl-*/ and skills/orchestrator/
n_skills=0
for group in core-skills debug-skills op-skills; do
  for src in "$SCRIPT_DIR"/$group/cannbotdsl-*/; do
    [ -d "$src" ] || continue
    install_one "${src%/}" "$DEST_SKILLS/$(basename "$src")" && n_skills=$((n_skills+1))
  done
done

# Orchestrator: skills/orchestrator/ -> .opencode/skills/cannbotdsl-op-orchestrator/
if [ -f "$SCRIPT_DIR/orchestrator/SKILL.md" ]; then
  orch_dest="$DEST_SKILLS/cannbotdsl-op-orchestrator"
  remove_managed "$orch_dest" && {
    mkdir -p "$orch_dest"; touch "$orch_dest/$MARKER"
    if [ "$MODE" = "link" ]; then ln -s "$SCRIPT_DIR/orchestrator/SKILL.md" "$orch_dest/SKILL.md"
    else cp "$SCRIPT_DIR/orchestrator/SKILL.md" "$orch_dest/SKILL.md"; fi
    log "${orch_dest#$REPO_ROOT/}/SKILL.md  (orchestrator)"
    n_skills=$((n_skills+1))
  } || true
fi

# 2) Agents: skills/agents/cannbotdsl-*.md -> .opencode/agents/cannbotdsl-*.md
n_agents=0
for src in "$SCRIPT_DIR"/agents/cannbotdsl-*.md; do
  [ -f "$src" ] || continue
  install_one "$src" "$DEST_AGENTS/$(basename "$src")" && n_agents=$((n_agents+1))
done

echo ""
echo "Installed $n_skills skills + $n_agents agents."
echo "Restart opencode in $REPO_ROOT to pick them up."
