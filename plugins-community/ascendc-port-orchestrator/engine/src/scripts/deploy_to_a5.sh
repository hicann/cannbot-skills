#!/bin/bash
# deploy_to_a5.sh — Backwards-compatibility wrapper. V3.4 routes deploys
# through deploy_to_npu.sh (multi-target). Pre-V3.4 callers (aog-kernel-worker
# briefs, ad-hoc scripts) still invoke this name; we honor it by forcing
# TARGET=a5 and delegating.
#
# The original A5-only deploy logic is preserved at deploy_to_a5.sh.legacy
# for the (rare) case where the multi-target plumbing has a regression.
exec env TARGET=a5 bash "$(dirname "$0")/deploy_to_npu.sh" "$@"
