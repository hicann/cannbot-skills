#!/bin/bash
set -e

# Copy run-hook.cmd from ops-direct-invoke (same pattern)
# This script is called by hooks.json to dispatch hook commands
HOOK_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$HOOK_NAME" in
  session-start-ascendc-ops-lab-developer)
    source "$SCRIPT_DIR/session-start-ascendc-ops-lab-developer"
    ;;
  *)
    echo "Unknown hook: $HOOK_NAME"
    exit 1
    ;;
esac
