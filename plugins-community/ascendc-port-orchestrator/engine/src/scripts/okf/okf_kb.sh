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
# okf_kb.sh — fail-closed entrypoint for the EXTERNAL cannbot-knowledge OKF engine.
#
# The port plugin no longer vendors the OKF engine (RFC #381). This wrapper resolves the INSTALLED
# cannbot-knowledge scripts (via okf_engine.py) and the KB root (this plugin's kb/okf), then dispatches:
#   retrieval  (build/search/pipeline/…) → knowledge_query.py --knowledge-root <kb>   (consumer tier)
#   governance (lint/lint-cards)         → knowledge_lint.py  --knowledge-root <kb>   (contributor tier)
# It keeps the two fail-closed guards from the vendored era: (1) the KB root must look like an OKF KB,
# (2) lint refuses to run on a 0-node graph — so a mis-set/empty root can NEVER silently pass card
# checks on an empty node set (the vacuous-lint failure mode found in M0). All porter code should call
# the engine through this wrapper, not knowledge_query.py/knowledge_lint.py directly.
#
# Usage:  engine/src/scripts/okf/okf_kb.sh build
#         engine/src/scripts/okf/okf_kb.sh search --query "..." --scope runbooks/
#         engine/src/scripts/okf/okf_kb.sh lint
#         OKF_KB_ROOT=/some/kb  CANNBOT_OKF_ENGINE_ROOT=/path/to/cannbot-knowledge  okf_kb.sh build
set -euo pipefail

HERE="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# this script lives at <plugin_root>/engine/src/scripts/okf/ → 4 levels below the plugin root
PLUGIN_ROOT="$(cd -P "$HERE/../../../.." && pwd)"
KB_ROOT="$(cd -P "${OKF_KB_ROOT:-$PLUGIN_ROOT/kb/okf}" 2>/dev/null && pwd -P || true)"

# fail-closed: root must exist and look like an OKF KB
if [ -z "$KB_ROOT" ] || [ ! -d "$KB_ROOT/reference" ] || [ ! -d "$KB_ROOT/runbooks" ]; then
  echo "okf_kb: FAIL — KB root '${OKF_KB_ROOT:-$PLUGIN_ROOT/kb/okf}' missing reference/ + runbooks/" >&2
  exit 2
fi
# community knowledge_query/knowledge_lint/okf_graph resolve their root from these env vars
# (as well as the --knowledge-root flag passed explicitly below).
export CANNBOT_KNOWLEDGE_ROOT="$KB_ROOT" OKF_KNOWLEDGE_ROOT="$KB_ROOT" KNOWLEDGE_ROOT="$KB_ROOT"

# resolve an external engine script; prints install guidance + nonzero exit if cannbot-knowledge is absent.
resolve() { python3 "$HERE/okf_engine.py" "$1"; }

cmd="${1:-}"; shift || true
case "$cmd" in
  build|search|pipeline|verify|get|grep|recall|rerank|neighbors|overview|browse|plan|preflight|discover)
    QUERY="$(resolve query)" || exit $?
    exec python3 "$QUERY" "$cmd" --knowledge-root "$KB_ROOT" "$@"
    ;;
  lint|lint-cards)
    # lint/graph live in the CONTRIBUTOR tier (maintainer only); resolve fails loudly if absent.
    GRAPH_DIR="$(resolve graph-dir)" || exit $?
    LINT="$(resolve lint)" || exit $?
    # fail-closed: refuse to run lint if okf_graph sees 0 nodes (wrong/empty root),
    # which would make its per-card checks vacuously pass.
    n="$(python3 - "$GRAPH_DIR" "$KB_ROOT" <<'PY'
import sys, os
graph_dir, root = sys.argv[1], sys.argv[2]
sys.path.insert(0, graph_dir)
for _v in ("CANNBOT_KNOWLEDGE_ROOT", "OKF_KNOWLEDGE_ROOT", "KNOWLEDGE_ROOT"):
    os.environ[_v] = root
# okf_graph computes ROOT at import from `--knowledge-root` in argv FIRST (env, then cwd, are fallbacks).
# Pass it explicitly so a missing/ignored env can NEVER silently make it count the cwd KB (fail-open).
sys.argv = ["okf_graph", "--knowledge-root", root]
import okf_graph as G
# defense-in-depth: the engine must have resolved OUR root, not a default/cwd KB, before we trust nodes>0.
if os.path.abspath(getattr(G, "ROOT", "")) != os.path.abspath(root):
    sys.stderr.write("okf_graph resolved root=%r != %r\n" % (getattr(G, "ROOT", None), root)); sys.exit(9)
print(len(G.load_nodes()))
PY
)"
    if ! [ "$n" -gt 0 ] 2>/dev/null; then
      echo "okf_kb: FAIL-CLOSED — okf_graph sees $n nodes under $KB_ROOT; refusing a vacuous lint." >&2
      exit 3
    fi
    if [ "$cmd" = "lint" ]; then
      exec python3 "$LINT" --knowledge-root "$KB_ROOT" "$@"
    fi
    # lint-cards: gate on card-level (.md) blockers AND real aggregate blockers (e.g. knowledge_query index
    # verify FAIL). ONLY the opt-in graph "cards never judged" blocker (needs the LLM-judge build) is
    # tolerated as informational. Fail-closed on unparseable/schema/accounting mismatch (exit 2).
    # This is the M1 card-governance gate (迁移OKF-设计方案 §3.6 完成判据, card side).
    exec python3 - "$LINT" "$KB_ROOT" <<'PY'
import json, os, subprocess, sys
lint, root = sys.argv[1], sys.argv[2]
env = dict(os.environ, CANNBOT_KNOWLEDGE_ROOT=root, OKF_KNOWLEDGE_ROOT=root, KNOWLEDGE_ROOT=root)
p = subprocess.run(["python3", lint, "--knowledge-root", root, "--json"],
                   capture_output=True, text=True, env=env)
# fail-closed: knowledge_lint returns exactly `1 if any blocker else 0`. Any OTHER exit code
# (crash / traceback / segfault) is a runtime error whose JSON — if any — can't be trusted.
if p.returncode not in (0, 1):
    sys.stderr.write("okf_kb lint-cards: FAIL-CLOSED — knowledge_lint exited %s (runtime error)\n%s\n"
                     % (p.returncode, (p.stderr or p.stdout or "")[:400])); sys.exit(2)
# fail-closed: unparseable or unexpected schema -> exit 2 (never silently pass).
try:
    d = json.loads(p.stdout)
except Exception:
    sys.stderr.write("okf_kb lint-cards: FAIL-CLOSED — knowledge_lint --json unparseable (rc=%s)\n%s\n"
                     % (p.returncode, (p.stdout or p.stderr)[:400])); sys.exit(2)
if not (isinstance(d, dict) and isinstance(d.get("findings"), list)
        and isinstance(d.get("blocker"), int)):
    sys.stderr.write("okf_kb lint-cards: FAIL-CLOSED — unexpected knowledge_lint JSON schema\n"); sys.exit(2)
# fail-closed: exit code must agree with the reported blocker count (guards a broken lint that
# prints blocker>0 but returns 0, or vice versa).
if (p.returncode == 0) != (d["blocker"] == 0):
    sys.stderr.write("okf_kb lint-cards: FAIL-CLOSED — rc=%s inconsistent with blocker=%s\n"
                     % (p.returncode, d["blocker"])); sys.exit(2)
findings = [f for f in d["findings"] if isinstance(f, dict)]
AGG = "F 聚合 verify"   # the aggregate-verify group (graph verify + knowledge_query index verify)
# tolerate ONLY the opt-in graph "N cards never judged" blocker — match the specific phrase (not a bare
# "never judged" substring) so a different aggregate failure that happens to contain those words is NOT
# waved through. See knowledge_lint run_aggregate → okf_graph verify "FAIL: N cards never judged: …".
NEVER_JUDGED = "cards never judged"
blockers = [f for f in findings if f.get("severity") == "blocker"]
# Classify by the STRUCTURED `group` field, not path/message text:
#   card_block = blockers NOT in the aggregate group (real per-card frontmatter/structure issues)
#   agg_real   = aggregate blockers that are NOT the opt-in graph "cards never judged"
#                (e.g. knowledge_query index verify FAIL — a real problem that MUST fail the gate)
# Only the opt-in graph "never judged" (unbuilt LLM-judge graph) is tolerated.
card_block = [f for f in blockers if f.get("group") != AGG]
agg_real   = [f for f in blockers if f.get("group") == AGG and NEVER_JUDGED not in str(f.get("msg", ""))]
fail = card_block + agg_real
# consistency guard: our accounting of blockers must equal knowledge_lint's own count minus tolerated ones.
tolerated = [f for f in blockers if f.get("group") == AGG and NEVER_JUDGED in str(f.get("msg", ""))]
if len(blockers) != len(fail) + len(tolerated) or len(blockers) != d["blocker"]:
    sys.stderr.write("okf_kb lint-cards: FAIL-CLOSED — blocker accounting mismatch "
                     "(reported=%s counted=%s)\n" % (d["blocker"], len(blockers))); sys.exit(2)
graph_unbuilt = bool(tolerated)
card_warn = [f for f in findings if f.get("severity") == "warn" and f.get("group") != AGG]
print("okf_kb lint-cards: card-blockers=%d real-aggregate=%d warns=%d | graph-verify=%s"
      % (len(card_block), len(agg_real), len(card_warn),
         "unbuilt(opt-in, tolerated)" if graph_unbuilt else "ok"))
for f in fail[:30]:
    print("  BLOCKER [%s] %s — %s" % (f.get("group"), f.get("card"), str(f.get("msg", ""))[:80]))
sys.exit(1 if fail else 0)
PY
    ;;
  *)
    echo "usage: okf_kb.sh {build|search|pipeline|verify|get|grep|recall|rerank|neighbors|overview|plan|preflight|lint|lint-cards} [args]" >&2
    exit 64
    ;;
esac
