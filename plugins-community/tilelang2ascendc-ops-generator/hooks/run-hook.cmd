#!/bin/bash
set -e

# Copy run-hook.cmd from ops-direct-invoke (same pattern)
# This script is called by hooks.json to dispatch hook commands
HOOK_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$HOOK_NAME" in
  session-start-tilelang2ascendc-ops-generator)
    source "$SCRIPT_DIR/session-start-tilelang2ascendc-ops-generator"
    ;;
  *)
    echo "Unknown hook: $HOOK_NAME"
    exit 1
    ;;
esac
