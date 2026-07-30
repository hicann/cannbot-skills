#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# ascendc-port-orchestrator plugin installer (bundle-orch).
#
# Mirrors the cannbot per-plugin install convention (cf. catlass-op-generator/init.sh):
#   - Product-owned Skills come from this plugin's local `skills/` directory.
#   - Shared ops Skills and community `knowledge-query` stay in their canonical
#     packages and are resolved from the checkout or marketplace dependencies.
#   - AGENTS come from this plugin's local `agents/`, filtered by INCLUDED_AGENT_PATTERN.
#   - Both installed as per-item symlinks under <CONFIG_ROOT>/{skills,agents}.
# Differences from catlass:
#   - For Claude we honor $CLAUDE_CONFIG_DIR (multi-agent hosts do NOT use ~/.claude).
#   - No asc-devkit step (the op-gen engine is bundled under engine/).
#   - Also creates the user-side KB(c) root + scaffolds engine/workspace/.ascendc_env.
# Engine packaged under engine/ (PYTHONPATH=engine/src/scripts python3 -m orchestrator …).
# Currently coupled to Claude Code (skill format / hooks / sub-agent via `claude`); see docs/ARCHITECTURE.md §8.
set -euo pipefail

if [ -t 1 ]; then
  GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
else
  GREEN=''; YELLOW=''; RED=''; CYAN=''; BOLD=''; DIM=''; NC=''
fi
ok()   { echo -e "  ${DIM}${GREEN}✓${NC}${DIM} $*${NC}"; }
warn() { echo -e "  ${YELLOW}⚠${NC}${DIM} $*${NC}"; }
err()  { echo -e "  ${RED}✗${NC}${DIM} $*${NC}"; }
step() { echo -e "${DIM}$*${NC}"; }

# safe_link SRC DST OURROOT — symlink SRC→DST without silently clobbering a
# non-plugin target (name collision, e.g. a co-installed a5_ops skill of the same
# name, or a user's own skill dir). Rules:
#   - DST is our own prior symlink (→ OURROOT) → silent re-link (idempotent).
#   - DST is a symlink pointing ELSEWHERE → warn (cross-plugin collision), then
#     overwrite (plugin takes precedence, but the clobber is surfaced, not silent).
#   - DST is a REAL file/dir (not a symlink) → SKIP + error. Never `rm -rf` real
#     content — that would destroy a user's / another tool's data.
# Returns 0 if linked, 1 if skipped. COLLISIONS counts warns+skips for the summary.
COLLISIONS=0
safe_link() {
  local src="$1" dst="$2" ourroot="$3"
  if [ -L "$dst" ]; then
    local cur; cur="$(readlink -f "$dst" 2>/dev/null || true)"
    case "$cur" in
      "$ourroot"/*) : ;;  # our own prior link — idempotent re-link, silent
      *) warn "collision: $(basename "$dst") already links elsewhere (${cur:-?}) — overwriting with plugin version"; COLLISIONS=$((COLLISIONS + 1)) ;;
    esac
    rm -f "$dst"
  elif [ -e "$dst" ]; then
    err "collision: $(basename "$dst") exists as a real path (not a plugin symlink) — SKIPPING to avoid data loss; remove it manually to install"
    COLLISIONS=$((COLLISIONS + 1))
    return 1
  fi
  ln -sfn "$src" "$dst"
}

BRAND="cannbot"
VERSION="0.1.4"                 # keep in sync with plugin.json
PLUGIN="ascendc-port-orchestrator"

# Self-contained customer implementation: 2 customer entries + 8 aog-* Skills.
# Reusable ops Skills keep a single canonical copy under repository ops/ and are
# supplied by the ascendc-port-orchestrator-shared-skills marketplace dependency.
LOCAL_SKILLS="ascendc-cross-gen-port ascendc-backward-gen aog-op-classify aog-input-gen-builder aog-knowledge-maintain aog-perf-eval aog-self-critic aog-a3-author aog-prior-art-verify aog-report-gen"
SHARED_SKILLS="ops-precision-standard ascendc-docs-search ascendc-simt-best-practices"
# OKF query is owned by plugins-community/cannbot-knowledge.
KNOWLEDGE_SKILLS="knowledge-query"
# Keep this literal union in sync with the three lists above: the repository's
# dependency validator and third-party installers consume this declaration without
# evaluating shell variable expansion.
INCLUDED_SKILLS="ascendc-cross-gen-port ascendc-backward-gen aog-op-classify aog-input-gen-builder aog-knowledge-maintain aog-perf-eval aog-self-critic aog-a3-author aog-prior-art-verify aog-report-gen ops-precision-standard ascendc-docs-search ascendc-simt-best-practices knowledge-query"
# Customer agents, kept CONSISTENT with plugin.json agents[] (9).
# Both installer and manifest must expose the same set: a missing dispatched agent crashes,
# while every advertised agent must have its customer Skill installed. The
# a3-port/backward escalation chain is a subset (kernel-worker + precision-probe + kernel-
# optimizer + fused-optimizer + researcher + determinism-analyzer + cann-learner);
# hardware-probe and report-gen support that same migration/backward lifecycle.
INCLUDED_AGENTS="aog-kernel-worker aog-precision-probe aog-kernel-optimizer aog-fused-optimizer aog-researcher aog-determinism-analyzer aog-cann-learner aog-hardware-probe aog-report-gen"
# All customer agents use the conventional aog-* name and resolve to the same
# explicit 9-agent closure above.
INCLUDED_AGENT_PATTERN="aog-*"

LEVEL="project"; TOOL="claude"
for arg in "${@:-}"; do
  case "$arg" in
    --help) echo "Usage: init.sh [project|global] [claude|opencode|cursor|copilot]  (Claude honors \$CLAUDE_CONFIG_DIR)"; exit 0 ;;
    global|project) LEVEL="$arg" ;;
    claude)   TOOL="claude" ;;
    opencode) TOOL="opencode" ;;
    cursor)   TOOL="cursor" ;;
    copilot)  TOOL="copilot" ;;
  esac
done

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_AGENT_ROOT="$PLUGIN_DIR/agents"
LOCAL_SKILL_ROOT="$PLUGIN_DIR/skills"
SHARED_SKILL_ROOT="$PLUGIN_DIR/../../ops"
# A checkout can link canonical shared Skills directly. A marketplace copy has no
# repository root and relies on the declared shared-skills dependency instead.
if [ -d "$SHARED_SKILL_ROOT" ]; then DIRECT_CHECKOUT=1; else DIRECT_CHECKOUT=0; fi

claude_root() { echo "${CLAUDE_CONFIG_DIR:-$HOME/.claude}"; }
marketplace_skill_path() {
  local skill="$1" cache_root hit
  cache_root="$(claude_root)/plugins/cache"
  [ -d "$cache_root" ] || return 1
  hit="$(find "$cache_root" -type f -path "*/$skill/SKILL.md" -print -quit 2>/dev/null || true)"
  [ -n "$hit" ] || return 1
  dirname "$hit"
}
marketplace_skill_present() {
  marketplace_skill_path "$1" >/dev/null
}
if [ "$LEVEL" = "global" ]; then
  if [ "$TOOL" = "opencode" ]; then
    CONFIG_ROOT="$HOME/.config/opencode"
  elif [ "$TOOL" = "copilot" ]; then
    CONFIG_ROOT="$HOME/.copilot"
  elif [ "$TOOL" = "cursor" ]; then
    CONFIG_ROOT="$HOME/.cursor"
  else
    if [ -n "${CLAUDE_CONFIG_DIR:-}" ]; then
      CONFIG_ROOT="$CLAUDE_CONFIG_DIR"
    else
      CONFIG_ROOT="$HOME/.claude"
    fi
  fi
else
  if [ "$TOOL" = "opencode" ]; then
    CONFIG_ROOT="$PWD/.opencode"
  elif [ "$TOOL" = "copilot" ]; then
    CONFIG_ROOT="$PWD/.github"
  elif [ "$TOOL" = "cursor" ]; then
    CONFIG_ROOT="$PWD/.cursor"
  else
    CONFIG_ROOT="$PWD/.claude"
  fi
fi

echo ""
echo -e "  ${BOLD}CANNBot · ${PLUGIN} (bundle-orch)${NC}"
echo "  Tool: $TOOL   Level: $LEVEL   Path: $CONFIG_ROOT"
echo "  Engine: $PLUGIN_DIR/engine"
echo ""

# --- Step 1: link plugin-local skills, shared dependencies, and local agents ---
step "[1/4] Installing skills + agents (per-item symlinks)..."
mkdir -p "$CONFIG_ROOT/skills" "$CONFIG_ROOT/agents"

# A direct checkout install needs the primary instructions/workflow assets in
# its config root. Claude reads CLAUDE.md; the other supported tools read
# AGENTS.md. Marketplace installs already receive the same instructions via the
# plugin SessionStart hook, so they must not inject duplicate global context.
# safe_link preserves real user-owned paths.
if [ "$DIRECT_CHECKOUT" = "1" ]; then
  if [ -f "$PLUGIN_DIR/AGENTS.md" ]; then
    if [ "$TOOL" = "claude" ]; then
      context_file="CLAUDE.md"
    else
      context_file="AGENTS.md"
    fi
    if safe_link "$(realpath "$PLUGIN_DIR/AGENTS.md")" "$CONFIG_ROOT/$context_file" "$PLUGIN_DIR"; then
      ok "$context_file"
    fi
  fi
  if [ -d "$PLUGIN_DIR/workflows" ]; then
    if safe_link "$(realpath "$PLUGIN_DIR/workflows")" "$CONFIG_ROOT/workflows" "$PLUGIN_DIR"; then
      ok "workflows"
    fi
  fi
fi

sc=0
for want in $LOCAL_SKILLS; do
  [ -d "$LOCAL_SKILL_ROOT/$want" ] || warn "plugin-local skill missing: $want"
done
for skill_dir in "$LOCAL_SKILL_ROOT"/*/; do
  [ -d "$skill_dir" ] || continue
  name=$(basename "$skill_dir")
  echo "$LOCAL_SKILLS" | grep -qw "$name" || continue
  target="$CONFIG_ROOT/skills/$name"
  if safe_link "$(realpath "$skill_dir")" "$target" "$LOCAL_SKILL_ROOT"; then sc=$((sc + 1)); fi
done

for want in $SHARED_SKILLS; do
  if [ "$DIRECT_CHECKOUT" = "1" ] && [ -d "$SHARED_SKILL_ROOT/$want" ]; then
    shared_source="$(realpath "$SHARED_SKILL_ROOT/$want")"
    if safe_link "$shared_source" "$CONFIG_ROOT/skills/$want" "$(realpath "$SHARED_SKILL_ROOT")"; then
      sc=$((sc + 1))
    fi
  elif shared_source="$(marketplace_skill_path "$want")"; then
    # Link dependency skills explicitly as well.  This keeps them loadable from
    # CONFIG_ROOT even when a Claude version rejects the dependency's synthetic
    # marketplace manifest after caching it.
    if safe_link "$(realpath "$shared_source")" "$CONFIG_ROOT/skills/$want" \
        "$(realpath "$(dirname "$shared_source")")"; then
      sc=$((sc + 1))
    fi
  elif [ ! -e "$CONFIG_ROOT/skills/$want" ]; then
    warn "$want is not locally resolvable; marketplace dependency must provide it"
  fi
done

KNOWLEDGE_CHECKOUT_ROOT="$PLUGIN_DIR/../cannbot-knowledge/skills"
if [ -d "$KNOWLEDGE_CHECKOUT_ROOT/knowledge-query" ]; then
  knowledge_source="$(realpath "$KNOWLEDGE_CHECKOUT_ROOT/knowledge-query")"
  if safe_link "$knowledge_source" "$CONFIG_ROOT/skills/knowledge-query" "$KNOWLEDGE_CHECKOUT_ROOT"; then
    sc=$((sc + 1))
  fi
elif knowledge_source="$(marketplace_skill_path knowledge-query)"; then
  # The consumer bundle can be present in the cache but rejected as a plugin by
  # stricter Claude manifest validation.  The canonical Skill itself remains a
  # valid dependency, so expose it through the normal config-root load path.
  if safe_link "$(realpath "$knowledge_source")" "$CONFIG_ROOT/skills/knowledge-query" \
      "$(realpath "$(dirname "$knowledge_source")")"; then
    sc=$((sc + 1))
  fi
elif [ ! -e "$CONFIG_ROOT/skills/knowledge-query" ]; then
  warn "knowledge-query is not locally resolvable; marketplace dependency must provide it"
fi

# --- a-tier route dependency resolution (§5.2 c>b>a) ---
# The engine ships a route config (cba_routes.json) declaring the COMMUNITY (tier-a)
# skills this plugin ROUTES worker briefs to. A route whose target skill is absent =
# the brief surfaces the route but the worker cannot invoke the Skill → LOAD-without-USE
# (a real single-plugin deploy gap: the plugin ships a route but not a guarantee its
# target is present). The two routes owned by this product are bundled locally;
# other routes may be supplied by a separately installed plugin. Idempotent: an
# already-linked or marketplace-provided target is accepted, never double-linked.
CBA_ROUTES="$PLUGIN_DIR/engine/src/scripts/orchestrator/cba_routes.json"
if [ -f "$CBA_ROUTES" ]; then
  ROUTE_SKILLS="$(python3 - "$CBA_ROUTES" 2>/dev/null <<'PY'
import json
import sys

with open(sys.argv[1]) as route_file:
    routes = json.load(route_file)
print(" ".join(sorted({entry["skill"] for entry in routes
                       if isinstance(entry, dict) and entry.get("skill")})))
PY
  )" || ROUTE_SKILLS=""
  for rs in $ROUTE_SKILLS; do
    [ -e "$CONFIG_ROOT/skills/$rs" ] && continue         # already installed → idempotent skip
    if marketplace_skill_present "$rs"; then
      ok "a-tier route dep resolved by marketplace: $rs"
      continue
    fi
    warn "a-tier route target not installed: $rs (install its owning plugin before using this optional route)"
  done
fi

for want in $INCLUDED_AGENTS; do
  [ -f "$LOCAL_AGENT_ROOT/$want.md" ] || warn "whitelisted agent NOT in plugin agents/: $want"
done
ac=0
for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
  [ -e "$agent_entry" ] || continue
  name=$(basename "$agent_entry"); base="${name%.md}"
  echo "$INCLUDED_AGENTS" | grep -qw "$base" || continue
  target="$CONFIG_ROOT/agents/$name"
  if safe_link "$(realpath "$agent_entry")" "$target" "$LOCAL_AGENT_ROOT"; then ac=$((ac + 1)); fi
done
ok "skills: $sc symlinks   agents: $ac symlinks"
if [ "$COLLISIONS" -gt 0 ]; then
  warn "$COLLISIONS name collision(s) — a same-named skill/agent (e.g. a co-installed a5_ops or another plugin) was overwritten or skipped; see above"
fi
echo ""

# --- Step 1b: ensure one safety-net hook registration surface ---
# The op-gen quality guarantee (anti-reward-hacking) needs CC PreToolUse/PostToolUse
# hooks that gate the SUB-AGENT's individual tool calls (workflow_critic = state-machine
# + G7-slug spawn gate; output_read_guard = force reading tool output; ship_claim_audit =
# audit done/PASS claims). These fire INSIDE the `claude --agent` subprocess, which the
# Python engine can't introspect — so they must be CC hooks, not pipeline events.
# Marketplace installs use the plugin-native hooks/hooks.json. A direct checkout
# has no enabled plugin registration, so only that mode writes owner-tagged entries
# into its config root and engine cwd. Scripts self-resolve from __file__ and pass
# through when no op-gen workspace is active. Step 4 validates the selected
# declaration and proves the output-read guard's deny/allow behavior.
step "[1b/4] Registering safety-net hooks..."
if [ "$TOOL" = "claude" ]; then
  if [ "$DIRECT_CHECKOUT" = "1" ]; then
    WF="$PLUGIN_DIR/engine/src/scripts/workflow"
    ENGINE_SETTINGS="$PLUGIN_DIR/engine/.claude/settings.json"
    mkdir -p "$(dirname "$ENGINE_SETTINGS")"
    for SETTINGS in "$CONFIG_ROOT/settings.json" "$ENGINE_SETTINGS"; do
  ASCENDC_PORT_WF="$WF" python3 - "$SETTINGS" <<'PYHOOK'
import json, os, sys
settings_path = sys.argv[1]
wf = os.environ["ASCENDC_PORT_WF"]
OWNER = "ascendc-port-orchestrator"
plugin_root = os.path.abspath(os.path.join(wf, "../../../.."))
critic = f"python3 {wf}/workflow_critic.py"
guard  = f"python3 {wf}/output_read_guard.py"
ship   = f"python3 {wf}/ship_claim_audit.py"
agent_gate = f"python3 {plugin_root}/hooks/agent-gate-dispatch.py"
def H(cmd, t): return {"type": "command", "command": cmd, "timeout": t, "_owner": OWNER}
pre = [
    ("Task|Agent", critic, 30), ("Edit|Write|MultiEdit", critic, 15),
    ("Edit|Write|MultiEdit", f"{agent_gate} pretool", 10),
    ("Bash", critic, 10), ("Read|Grep|Glob|Bash", guard, 10), ("WebFetch", critic, 5),
]
post = [("Task|Agent", ship, 30)]
subagent_stop = [("", f"{agent_gate} stop", 60)]
try:
    s = json.load(open(settings_path))
except Exception:
    s = {}
hooks = s.setdefault("hooks", {})
def strip_owned(lst):
    out = []
    for entry in lst:
        kept = [h for h in entry.get("hooks", []) if h.get("_owner") != OWNER]
        if kept:
            entry = dict(entry); entry["hooks"] = kept; out.append(entry)
        elif not entry.get("hooks"):
            out.append(entry)
    return out
hooks["PreToolUse"] = strip_owned(hooks.get("PreToolUse", [])) + [{"matcher": m, "hooks": [H(c, t)]} for m, c, t in pre]
hooks["PostToolUse"] = strip_owned(hooks.get("PostToolUse", [])) + [{"matcher": m, "hooks": [H(c, t)]} for m, c, t in post]
hooks["SubagentStop"] = strip_owned(hooks.get("SubagentStop", [])) + [
    {"hooks": [H(c, t)]} for _, c, t in subagent_stop
]
json.dump(s, open(settings_path, "w"), indent=2)
print(
    f"  registered {len(pre)} PreToolUse + {len(post)} PostToolUse + "
    f"{len(subagent_stop)} SubagentStop hooks → {settings_path}"
)
PYHOOK
    done
    HOOK_REGISTRATION="direct-settings"
    HOOK_SETTINGS_ENGINE="$ENGINE_SETTINGS"
    ok "direct-checkout hooks → $CONFIG_ROOT/settings.json + $ENGINE_SETTINGS"
  else
    # Claude installs hooks/hooks.json as part of the enabled marketplace plugin.
    # Rewriting the same commands into user + engine settings here would execute
    # every gate multiple times and leave stale cache-version absolute paths after
    # an upgrade or uninstall.
    HOOK_REGISTRATION="plugin"
    HOOK_SETTINGS_ENGINE=""
    ok "safety-net hooks provided by the enabled marketplace plugin"
  fi
else
  HOOK_REGISTRATION="unsupported-$TOOL"
  HOOK_SETTINGS_ENGINE=""
  warn "hook registration skipped (tool=$TOOL; hooks are Claude Code PreToolUse/PostToolUse)"
fi
echo ""

# --- Step 2: user-side KB(c) root + index (ARCH §5.1/§5.3) ---
step "[2/4] User KB (c-tier) root..."
USER_KB="${ASCENDC_PORT_USER_KB:-$HOME/.ascendc-port/user_kb}"
if [ ! -d "$USER_KB" ]; then
  mkdir -p "$USER_KB"
  cat > "$USER_KB/INDEX.md" <<'IDX'
# 用户本地 KB 索引 (c 层) — canonical-topic → 条目文件
# 由流水线「生成后沉淀」自动更新；用户可手工增删改。优先级 c > b > a。
IDX
  ok "created user-KB(c): $USER_KB (+ INDEX.md)"
else
  ok "user-KB(c) exists: $USER_KB"
fi
echo ""

# Build the packaged b-tier OKF index during installation.  The index is a
# generated cache and is intentionally not committed, but the deterministic
# orchestrator reads it directly; leaving it absent makes a clean install look
# healthy while every official-KB query returns no evidence.
OFFICIAL_OKF_ROOT="$PLUGIN_DIR/kb/okf"
OFFICIAL_OKF_INDEX_READY=false
if [ -d "$OFFICIAL_OKF_ROOT" ]; then
  QUERY_SCRIPT="$CONFIG_ROOT/skills/knowledge-query/scripts/knowledge_query.py"
  if [ ! -f "$QUERY_SCRIPT" ]; then
    err "knowledge-query script is unavailable; cannot build the official OKF index"
    exit 1
  fi
  if python3 "$QUERY_SCRIPT" --knowledge-root "$OFFICIAL_OKF_ROOT" build >/dev/null; then
    OFFICIAL_OKF_INDEX_READY=true
    ok "official OKF index built: $OFFICIAL_OKF_ROOT/search/okf.index.json"
  else
    err "failed to build official OKF index under $OFFICIAL_OKF_ROOT"
    exit 1
  fi
else
  warn "official OKF root is not present in this checkout; aggregate package must provide kb/okf"
fi
echo ""

# --- Step 3: scaffold engine/workspace/.ascendc_env (NPU host/container/mode single source) ---
step "[3/4] NPU env scaffold..."
ENV_TEMPLATE="$PLUGIN_DIR/engine/workspace/.ascendc_env.template"
ENV_FILE="$PLUGIN_DIR/engine/workspace/.ascendc_env"
if [ -f "$ENV_TEMPLATE" ] && [ ! -f "$ENV_FILE" ]; then
  cp "$ENV_TEMPLATE" "$ENV_FILE"
  warn "scaffolded $ENV_FILE — fill NPU host/container/credentials before running"
elif [ -f "$ENV_FILE" ]; then
  ok ".ascendc_env exists"
else
  warn ".ascendc_env(.template) not found under engine/workspace/ — a3-port needs it (docs/USAGE.md)"
fi
echo ""

# --- Step 4: health check + manifest ---
step "[4/4] Health check + manifest..."
health_ok=true
for want in $INCLUDED_SKILLS; do
  if [ -e "$CONFIG_ROOT/skills/$want" ] || marketplace_skill_present "$want"; then
    continue
  fi
  err "skill not installed or present in a marketplace dependency: $want"
  health_ok=false
done
EFFECTIVE_AGENTS=""
for want in $INCLUDED_AGENTS; do
  if [ -e "$CONFIG_ROOT/agents/$want.md" ]; then
    EFFECTIVE_AGENTS="$EFFECTIVE_AGENTS $want"
  else
    err "agent not installed: $want"
    health_ok=false
  fi
done

# Hook LIVENESS proof (DEBT-253). Counting files is what let a disarmed install tick ✓:
# skills/agents were present, hooks were written — to a path CC never reads. So prove two
# things instead of counting:
#   (a) LOCATION — our hooks are in `engine/.claude/settings.json`, the only file CC is
#       guaranteed to load for subagent tool calls (it reads `<cwd>/.claude/`, and the
#       engine's cwd IS engine/; it does not walk up, not even to the git root);
#   (b) BEHAVIOR — the guard actually runs and DISCRIMINATES: a subagent payload touching
#       output/ must be denied (exit 2) AND a main-agent payload must pass (exit 0). A
#       guard that always-allows or always-denies fails this, a missing python3 fails it.
# Failing either is a hard install failure — a disarmed install must never print ✓.
hooks_live=false
if [ "$TOOL" = "claude" ]; then
  hooks_check_ok=true
  if [ "$DIRECT_CHECKOUT" = "1" ]; then
    HOOK_DECLARATION="$PLUGIN_DIR/engine/.claude/settings.json"
  else
    HOOK_DECLARATION="$PLUGIN_DIR/hooks/hooks.json"
  fi
  python3 - "$HOOK_DECLARATION" <<'PYCHK' || hooks_check_ok=false
import json, sys
try:
    h = json.load(open(sys.argv[1])).get("hooks", {})
except Exception as e:
    print(f"  hook settings unreadable at {sys.argv[1]}: {e}"); sys.exit(1)
cmds = [x.get("command", "") for ev in ("PreToolUse", "PostToolUse", "SubagentStop")
        for m in h.get(ev, []) for x in m.get("hooks", [])]
missing = [n for n in ("output_read_guard.py", "workflow_critic.py", "ship_claim_audit.py",
                       "agent-gate-dispatch.py")
           if not any(n in c for c in cmds)]
if missing:
    print(f"  safety-net hooks missing from {sys.argv[1]}: {missing}"); sys.exit(1)
stop_cmds = [x.get("command", "") for m in h.get("SubagentStop", [])
             for x in m.get("hooks", [])]
if not any("agent-gate-dispatch.py" in c and " stop" in c for c in stop_cmds):
    print(f"  agent gates are not registered on SubagentStop in {sys.argv[1]}")
    sys.exit(1)
PYCHK
  [ "$hooks_check_ok" = true ] || err "hook declaration check failed → subagents would run unguarded"
  GUARD="$PLUGIN_DIR/engine/src/scripts/workflow/output_read_guard.py"
  PROBE_PATH="$PLUGIN_DIR/engine/output/_install_selfcheck/src/kernels/_probe/verification.json"
  # NOTE: a DENY is exit 2, i.e. "nonzero" — under this script's `set -e` an unguarded
  # invocation would kill the installer instead of being read as the expected verdict.
  # `|| rc=$?` captures it. (Same shape as the vendor-script failure DEBT-247 chased: a
  # meaningful nonzero exit swallowed by `set -e` before anyone could inspect it.)
  rc_deny=0
  printf '{"agent_id":"install-selfcheck","tool_name":"Read","tool_input":{"file_path":"%s"},"cwd":"%s"}' \
    "$PROBE_PATH" "$PLUGIN_DIR/engine" | python3 "$GUARD" >/dev/null 2>&1 || rc_deny=$?
  [ "$rc_deny" -eq 2 ] || { err "output_read_guard did NOT deny a subagent output/ read (got $rc_deny, expected 2)"; hooks_check_ok=false; }
  rc_allow=0
  printf '{"tool_name":"Read","tool_input":{"file_path":"%s"},"cwd":"%s"}' \
    "$PROBE_PATH" "$PLUGIN_DIR/engine" | python3 "$GUARD" >/dev/null 2>&1 || rc_allow=$?
  [ "$rc_allow" -eq 0 ] || { err "output_read_guard denied the MAIN agent (got $rc_allow, expected 0 — it must only bind subagents)"; hooks_check_ok=false; }
  if [ "$hooks_check_ok" = true ]; then
    hooks_live=true
    ok "hooks verified live (denies subagent output/ read, passes main agent)"
  else health_ok=false; fi
fi

MANIFEST="$CONFIG_ROOT/cannbot-manifest.json"
EFFECTIVE_SKILLS=""
for want in $INCLUDED_SKILLS; do
  if [ -e "$CONFIG_ROOT/skills/$want" ] || marketplace_skill_present "$want"; then
    EFFECTIVE_SKILLS="$EFFECTIVE_SKILLS $want"
  fi
done
SKILLS_JSON=$(printf '%s\n' $EFFECTIVE_SKILLS | sed '/^$/d' | sort -u | python3 -c "import sys,json;print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))")
AGENTS_JSON=$(printf '%s\n' $EFFECTIVE_AGENTS | sed '/^$/d' | sort -u | python3 -c "import sys,json;print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))")
cat > "$MANIFEST" <<MANIFEST_EOF
{
  "brand": "CANNBot",
  "plugin": "$PLUGIN",
  "version": "$VERSION",
  "level": "$LEVEL",
  "tool": "$TOOL",
  "config_root": "$CONFIG_ROOT",
  "plugin_root": "$PLUGIN_DIR",
  "engine_root": "$PLUGIN_DIR/engine",
  "installed_skills": $SKILLS_JSON,
  "installed_agents": $AGENTS_JSON,
  "linked_skills_count": $sc,
  "effective_skills_count": $(printf '%s\n' $EFFECTIVE_SKILLS | sed '/^$/d' | sort -u | wc -l | tr -d ' '),
  "installed_agents_count": $ac,
  "user_kb_c": "$USER_KB",
  "official_kb_b": "$OFFICIAL_OKF_ROOT",
  "official_okf_index_ready": $OFFICIAL_OKF_INDEX_READY,
  "hooks_registration": "$HOOK_REGISTRATION",
  "hooks_settings_engine": "$HOOK_SETTINGS_ENGINE",
  "hooks_verified_live": $( [ "${hooks_live:-false}" = true ] && echo true || echo false ),
  "install_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
MANIFEST_EOF
[ -f "$MANIFEST" ] || { err "manifest failed"; health_ok=false; }

echo ""
if [ "$health_ok" = true ]; then
  if [ "$hooks_live" = true ]; then
    echo -e "  ${GREEN}${BOLD}✓ ${PLUGIN} installed${NC} (skills=$sc agents=$ac → $CONFIG_ROOT)"
  else
    echo -e "  ${YELLOW}${BOLD}⚠ ${PLUGIN} installed without live safety hooks${NC} (tool=$TOOL; skills=$sc agents=$ac → $CONFIG_ROOT)"
  fi
else
  echo -e "  ${RED}${BOLD}✗ install had errors, see above${NC}"; exit 1
fi
echo ""
echo -e "  ${BOLD}Quick start (Claude Code):${NC}"
echo -e "  ${CYAN}1.${NC} fill ${GREEN}engine/workspace/.ascendc_env${NC} (A3/A5 host+container) — docs/USAGE.md"
echo -e "  ${CYAN}2.${NC} launch ${GREEN}claude${NC}, then a customer entry skill:"
echo -e "       ${GREEN}/ascendc-cross-gen-port <ops-nn op dir>${NC}   (→ orch --port-a3)"
echo -e "       ${GREEN}/ascendc-backward-gen <forward spec>${NC}      (→ orch --backward)"
echo -e "  ${DIM}Pipeline logic lives entirely in engine/; the entry skills are thin NL front-ends.${NC}"
echo ""
