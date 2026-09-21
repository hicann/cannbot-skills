#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# Read-only compatibility entrypoint for the project-installed cannbot-knowledge.
#
# The cannbot-knowledge installer owns checkout acquisition, updates, dependency
# installation and index builds. Port only resolves that install result and
# invokes its query CLI. There is deliberately no plugin-local KB fallback and
# no build/lint command here.
set -euo pipefail

HERE="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve() { python3 "$HERE/okf_engine.py" "$1"; }

ROOT="$(resolve root)" || exit $?
QUERY="$(resolve query)" || exit $?
INDEX="$ROOT/artifacts/indexes/knowledge.sqlite3"
if [[ ! -f "$INDEX" ]]; then
  echo "okf_kb: FAIL — cannbot-knowledge index missing: $INDEX" >&2
  echo "Re-run the cannbot-knowledge installer for this project to build it." >&2
  exit 2
fi

cmd="${1:-}"
shift || true
case "$cmd" in
  search)
    exec python3 "$QUERY" --knowledge-root "$ROOT" search \
      --domain ops --technology ascendc --include-shared --status all "$@"
    ;;
  discover)
    exec python3 "$QUERY" --knowledge-root "$ROOT" discover "$@"
    ;;
  *)
    echo "usage: okf_kb.sh {search|discover} [args]" >&2
    echo "Port is a read-only consumer; build/lint belong to cannbot-knowledge." >&2
    exit 64
    ;;
esac
