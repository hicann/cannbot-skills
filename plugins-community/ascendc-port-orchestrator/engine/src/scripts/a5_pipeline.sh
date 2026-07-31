#!/bin/bash
# a5_pipeline.sh — Backwards-compatibility wrapper around npu_pipeline.sh (V3.4)
# Pre-V3.4 callers expect this script and an A5-only execution path.
# We now delegate to npu_pipeline.sh with TARGET forced to a5.
#
# New code should invoke `bash src/scripts/npu_pipeline.sh <cmd>` directly so
# `--target=a3` / TARGET=a3 overrides work without going through this shim.
exec env TARGET=a5 bash "$(dirname "$0")/npu_pipeline.sh" "$@"
