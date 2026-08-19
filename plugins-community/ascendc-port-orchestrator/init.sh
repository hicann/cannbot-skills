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
# Supports independent Claude Code and OpenCode harnesses; see docs/ARCHITECTURE.md §8.
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

# Run a command bounded by coreutils `timeout` where it exists (GNU/Linux, the plugin's
# target platform). On hosts without it (macOS) run unbounded rather than failing the
# probe setup: the bound exists to stop opencode's first-run plugin dependency
# resolution from hanging the installer forever, not to add a hard platform requirement.
_oc_timeout() {
  if command -v timeout >/dev/null 2>&1; then
    timeout "$@"
  else
    shift
    "$@"
  fi
}

# JS runtime for opencode-side probes: node preferred, bun as fallback. This is the
# canonical rule, shared with the engine's runtime self-check
# (backends/opencode_runtime.pick_js_runtime) — keep both sides in this order.
js_runtime() {
  command -v node >/dev/null 2>&1 && { echo "node"; return 0; }
  command -v bun >/dev/null 2>&1 && { echo "bun"; return 0; }
  return 1
}

# Match Python's `.strip()` normalization before comparing the probe's one-line
# protocol token. This keeps the installer and runtime verdict table identical
# even when a runner emits harmless leading/trailing whitespace or CRLF.
trim_probe_output() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

# Harness dependency preflight (G4). Each harness declares ONLY its own runtime deps:
#   claude   → the claude CLI (honors CLAUDE_BIN)
#   opencode → the opencode CLI + a JS runtime (node or bun)
# Default is warn-only (exit 0); --strict-deps turns every miss into a hard error.
# cursor/copilot config-only installs have no runtime dependency to check.
check_harness_deps() {
  local dep_ok=true missing=""
  case "$TOOL" in
    claude)
      if ! command -v "${CLAUDE_BIN:-claude}" >/dev/null 2>&1; then
        dep_ok=false; missing="${CLAUDE_BIN:-claude} CLI"
      fi ;;
    opencode)
      if ! command -v "${AOG_OPENCODE_BIN:-opencode}" >/dev/null 2>&1; then
        dep_ok=false; missing="${AOG_OPENCODE_BIN:-opencode} CLI"
      fi
      if [ -z "$(js_runtime)" ]; then
        dep_ok=false; missing="${missing:+$missing, }node/bun runtime"
      fi ;;
  esac
  if [ "$dep_ok" = false ]; then
    if [ "$STRICT_DEPS" = 1 ]; then
      err "missing harness dependency for tool=$TOOL: $missing — refusing under --strict-deps"
      exit 1
    fi
    warn "missing harness dependency for tool=$TOOL: $missing"
    warn "engine dispatch for tool=$TOOL will fail until it is installed (see docs/USAGE.md §Dependencies)"
    # This is explicitly warn-only unless --strict-deps was requested.  Returning
    # non-zero here would trigger the top-level errexit and abort the installer.
    return 0
  fi
  ok "harness deps for tool=$TOOL present ($(case "$TOOL" in claude) echo "${CLAUDE_BIN:-claude}";; opencode) echo "${AOG_OPENCODE_BIN:-opencode} + $(js_runtime)";; esac))"
  return 0
}

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
SHARED_SKILLS="ops-precision-standard ascendc-docs-search ascendc-simt-best-practices ascendc-api-best-practices"
# OKF query is owned by plugins-community/cannbot-knowledge.
KNOWLEDGE_SKILLS="knowledge-query"
# Keep this literal union in sync with the three lists above: the repository's
# dependency validator and third-party installers consume this declaration without
# evaluating shell variable expansion.
INCLUDED_SKILLS="ascendc-cross-gen-port ascendc-backward-gen aog-op-classify aog-input-gen-builder aog-knowledge-maintain aog-perf-eval aog-self-critic aog-a3-author aog-prior-art-verify aog-report-gen ops-precision-standard ascendc-docs-search ascendc-simt-best-practices ascendc-api-best-practices knowledge-query"
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

LEVEL="project"; TOOL="claude"; STRICT_DEPS=0
for arg in "${@:-}"; do
  case "$arg" in
    --help) echo "Usage: init.sh [project|global] [claude|opencode|cursor|copilot] [--strict-deps]  (Claude honors \$CLAUDE_CONFIG_DIR)"; exit 0 ;;
    global|project) LEVEL="$arg" ;;
    claude)   TOOL="claude" ;;
    opencode) TOOL="opencode" ;;
    cursor)   TOOL="cursor" ;;
    copilot)  TOOL="copilot" ;;
    --strict-deps) STRICT_DEPS=1 ;;   # missing harness deps become hard errors instead of warnings
  esac
done

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_AGENT_ROOT="$PLUGIN_DIR/agents"
LOCAL_SKILL_ROOT="$PLUGIN_DIR/skills"
SHARED_SKILL_ROOT="$PLUGIN_DIR/../../ops"
# A checkout can link canonical shared Skills directly. A Claude marketplace copy has no
# repository root and relies on the declared shared-skills dependency instead.
if [ -d "$SHARED_SKILL_ROOT" ]; then DIRECT_CHECKOUT=1; else DIRECT_CHECKOUT=0; fi

# OpenCode intentionally never treats Claude Code's marketplace cache as a dependency source:
# doing so would reintroduce a hidden ~/.claude read. The currently packaged OpenCode setup
# therefore requires a full repository checkout, where all canonical shared Skills are present.
if [ "$TOOL" = "opencode" ] && [ "$DIRECT_CHECKOUT" != "1" ]; then
  err "OpenCode setup requires a full cannbot-skills checkout (shared skills are not read from Claude marketplace cache)"
  err "Run init.sh from the repository checkout, not from a Claude marketplace cache path."
  exit 1
fi

claude_root() { echo "${CLAUDE_CONFIG_DIR:-$HOME/.claude}"; }
marketplace_skill_path() {
  local skill="$1" cache_root hit
  # OpenCode setup must be self-contained with respect to Claude Code: even a
  # best-effort cache lookup would read a user's ~/.claude tree. Direct checkout
  # dependencies remain usable; marketplace-only skills are reported normally.
  [ "$TOOL" = "opencode" ] && return 1
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

# G4: per-harness dependency preflight — warn by default, hard-fail under --strict-deps.
check_harness_deps

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
  # Normalise ourroot too: safe_link compares it against readlink -f output, and a
  # ../-bearing root would make our own prior links look like foreign collisions on
  # every idempotent re-run.
  if safe_link "$knowledge_source" "$CONFIG_ROOT/skills/knowledge-query" "$(realpath "$KNOWLEDGE_CHECKOUT_ROOT")"; then
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
# opencode MUST NOT receive the Claude-Code agent files. Their frontmatter carries
# `tools:` as a YAML LIST, while opencode's agent schema expects a record; a single such
# file makes opencode reject its whole configuration ("Configuration is invalid ...
# Expected object | undefined, got [...]") and EVERY opencode invocation on that machine
# fails, not just this plugin's. opencode gets its agents from the process-private
# OPENCODE_CONFIG_CONTENT the backend injects at dispatch time (already converted to
# `mode: primary` + a record-shaped tool map), so no install-time agent files are needed.
if [ "$TOOL" = "opencode" ]; then
  ok "agents: provided at dispatch time via OPENCODE_CONFIG_CONTENT (no agent files installed)"
  # Entry layer. opencode scans {command,commands}/**/*.md under its config dirs. The
  # shipped templates carry an @@PLUGIN_DIR@@ placeholder because a command file is a
  # prompt, not a script: it cannot resolve its own location the way the Claude Code skill
  # loader does (which echoes a base directory). Materialise it with the real path so the
  # command can invoke the SHARED launcher rather than re-deriving how to start the engine.
  cc=0
  cmd_fail=0
  if [ -d "$PLUGIN_DIR/.opencode/command" ]; then
    mkdir -p "$CONFIG_ROOT/command"
    for cmd_src in "$PLUGIN_DIR/.opencode/command"/*.md; do
      [ -f "$cmd_src" ] || continue
      cmd_dst="$CONFIG_ROOT/command/$(basename "$cmd_src")"
      # `init.sh project opencode` puts CONFIG_ROOT at $PWD/.opencode, so running it from
      # inside the plugin directory makes destination and SOURCE the same file. The shell
      # truncates the redirect target before sed ever reads it, so that would empty the
      # template — a tracked file in this repo — and then count it as installed. Refuse: a
      # "project install" into the plugin's own source tree is not an install.
      if [ "$cmd_dst" -ef "$cmd_src" ]; then
        err "refusing to install into the plugin's own source tree ($cmd_dst)"
        err "run 'init.sh project opencode' from YOUR project directory, or use 'global'"
        cmd_fail=$((cmd_fail + 1))
        continue
      fi
      # -F (fixed strings): the plugin path is data, not a BRE — dots/plus/pipes in it
      # must not turn the probe into a wrong match or an invalid pattern.
      if [ -e "$cmd_dst" ] && ! grep -Fq "@@PLUGIN_DIR@@" "$cmd_dst" 2>/dev/null \
          && ! grep -Fq "$PLUGIN_DIR" "$cmd_dst" 2>/dev/null; then
        warn "user-owned command preserved: $(basename "$cmd_dst")"
        continue
      fi
      # Via a temp file in the destination directory, then mv: a redirect straight onto
      # cmd_dst truncates it first, so an interrupted install leaves an empty command rather
      # than the previous working one.
      cmd_tmp="$(mktemp "${cmd_dst}.XXXXXX")"
      # Escape the replacement string: a plugin path containing & (or \, |) would be
      # reinterpreted by sed's replacement syntax otherwise.
      escaped_plugin_dir="$(printf '%s' "$PLUGIN_DIR" | sed 's/[&|\\]/\\&/g')"
      if sed "s|@@PLUGIN_DIR@@|$escaped_plugin_dir|g" "$cmd_src" > "$cmd_tmp" && mv -f "$cmd_tmp" "$cmd_dst"; then
        cc=$((cc + 1))
      else
        rm -f "$cmd_tmp"
        err "failed to install command $(basename "$cmd_src")"
        cmd_fail=$((cmd_fail + 1))
      fi
    done
  fi
  if [ "$cc" -gt 0 ]; then
    ok "commands: $cc installed → $CONFIG_ROOT/command (entry layer)"
  else
    warn "no opencode commands installed — users have no /ascendc-* entry point"
  fi
  # Fail loud if a placeholder survived: a command still saying @@PLUGIN_DIR@@ would tell
  # the agent to run a path that does not exist.
  # NOTE: health_ok is initialised to true further down (step 4), so setting it here would
  # be silently overwritten. Carry the verdict in ENTRY_LAYER_OK and fold it in at the gate.
  # cmd_fail joins the verdict too: "1 of N commands installed" was previously reported as
  # a clean entry layer (cc>0), i.e. exactly the armed-vs-disarmed confusion this script
  # exists to prevent.
  if grep -rl "@@PLUGIN_DIR@@" "$CONFIG_ROOT/command" >/dev/null 2>&1; then
    err "command install left an unsubstituted @@PLUGIN_DIR@@ placeholder"
    ENTRY_LAYER_OK=false
  elif [ "$cmd_fail" -gt 0 ]; then
    err "failed to install $cmd_fail opencode command file(s) — entry layer incomplete"
    ENTRY_LAYER_OK=false
  elif [ "$cc" -gt 0 ]; then
    ENTRY_LAYER_OK=true
  else
    ENTRY_LAYER_OK=false
  fi
else
  for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
    [ -e "$agent_entry" ] || continue
    name=$(basename "$agent_entry"); base="${name%.md}"
    echo "$INCLUDED_AGENTS" | grep -qw "$base" || continue
    target="$CONFIG_ROOT/agents/$name"
    if safe_link "$(realpath "$agent_entry")" "$target" "$LOCAL_AGENT_ROOT"; then ac=$((ac + 1)); fi
  done
fi
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
if [ "$TOOL" = "opencode" ] && [ "${ENTRY_LAYER_OK:-false}" != true ]; then
  err "opencode entry layer not installed — users would have no /ascendc-* command"
  health_ok=false
fi
if [ "$TOOL" = "opencode" ]; then
  # opencode resolves agents from the backend's process-private OPENCODE_CONFIG_CONTENT,
  # not from files under CONFIG_ROOT. Prove the CONVERSION works rather than counting
  # files: every whitelisted agent must appear in the generated config as mode=primary
  # (a subagent-mode entry would make `opencode run --agent` silently fall back).
  if OPENCODE_AGENT_CHECK="$(ASCENDC_PORT_WANT="$INCLUDED_AGENTS" python3 - "$PLUGIN_DIR" <<'PYOC'
import json, os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "engine", "src", "scripts", "orchestrator"))
from backends.opencode_backend import OpencodeBackend as B
cfg = json.loads(B._opencode_config_content() or "{}")
agents = cfg.get("agent", {})
missing = [w for w in os.environ["ASCENDC_PORT_WANT"].split() if w not in agents]
bad = [n for n, v in agents.items() if v.get("mode") != "primary"]
if missing or bad:
    print("missing=%s not_primary=%s" % (missing, bad)); sys.exit(1)
print(" ".join(sorted(agents)))
PYOC
  )"; then
    EFFECTIVE_AGENTS="$OPENCODE_AGENT_CHECK"
    ok "agents: $(echo "$EFFECTIVE_AGENTS" | wc -w | tr -d ' ') resolvable as mode=primary via OPENCODE_CONFIG_CONTENT"
  else
    err "opencode agent config invalid: $OPENCODE_AGENT_CHECK"
    health_ok=false
  fi
else
  for want in $INCLUDED_AGENTS; do
    if [ -e "$CONFIG_ROOT/agents/$want.md" ]; then
      EFFECTIVE_AGENTS="$EFFECTIVE_AGENTS $want"
    else
      err "agent not installed: $want"
      health_ok=false
    fi
  done
fi

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

# --- opencode: prove the safety net against the REAL binary -------------------------------
# The adapter is registered through the process-private OPENCODE_CONFIG_CONTENT the backend
# injects at dispatch time, so there is no declaration file to inspect. Two levels of proof,
# and `hooks_verified_live` is only ever set by the level that actually proves enforcement:
#
#   STRUCTURAL (always, offline-safe) — the real opencode binary must RESOLVE our injected
#     config: the agents must exist as mode=primary and the plugin skills must be visible.
#     This catches a config opencode silently refuses, which is the failure that would make
#     every later dispatch fall back to the default agent.
#   BEHAVIOURAL (src/opencode/probe_safety_net.mjs) — the adapter's real `tool.execute.before`
#     must REFUSE a kernel-worker's cross-workspace read and ALLOW its own-workspace read.
#     Enforcement, not resolution. It needs no model, so it runs on ordinary installs.
#
# Counting files and calling it armed is precisely the DEBT-253 failure this gate exists to
# prevent, so `hooks_verified_live` is set only by the behavioural level. What the behavioural
# level still does NOT cover is whether opencode itself invokes the hook for a model-driven
# tool call. Phase O0 does not close that either — it drives the same entry point through node.
# Only a real model turn shows it, which is what src/scripts/tests/test_opencode_e2e_live.py
# does when an operator points AOG_E2E_OPENCODE_MODEL at a configured model.
if [ "$TOOL" = "opencode" ]; then
  oc_ok=true
  OC_BIN="${AOG_OPENCODE_BIN:-opencode}"
  if ! command -v "$OC_BIN" >/dev/null 2>&1; then
    warn "opencode binary not found ($OC_BIN) — cannot verify the safety net"
    oc_ok=false
  else
    OC_CFG="$(cd "$PLUGIN_DIR/engine" && PYTHONPATH=src/scripts python3 -c "
import sys; sys.path.insert(0,'src/scripts/orchestrator')
from backends.opencode_backend import OpencodeBackend as B
sys.stdout.write(B._opencode_config_content() or '')" 2>/dev/null || true)"
    if [ -z "$OC_CFG" ]; then
      err "backend produced no opencode config — agents/skills/adapter would all be absent"
      oc_ok=false
    else
      # First run against a pristine opencode config dir triggers opencode's plugin
      # dependency resolution (a bun install) before any debug output: measured >90s
      # online, unbounded in the offline containers this plugin targets. Bound it and
      # say so — a probe that hangs forever is indistinguishable from a broken install.
      step "probing opencode plugin resolution — first run can take 1-2 minutes..."
      if ! _oc_timeout 180 env OPENCODE_CONFIG_CONTENT="$OC_CFG" "$OC_BIN" debug agent aog-kernel-worker \
             >/dev/null 2>&1; then
        err "opencode did not resolve agent aog-kernel-worker from the injected config (or the probe timed out)"
        oc_ok=false
      fi
      # G4: assert the FULL aog-* agent closure, not just the first agent. A config
      # that resolves kernel-worker but silently drops a later agent would fail at
      # dispatch time, deep inside a run — catch it at install time instead.
      if [ "$oc_ok" = true ]; then
        for _ag in "$PLUGIN_DIR"/agents/aog-*.md; do
          [ -e "$_ag" ] || continue
          _agname="$(basename "$_ag" .md)"
          # The structural probe above already checked this first agent.  Do not
          # spend a second 60-second timeout budget on the same resolution.
          [ "$_agname" = "aog-kernel-worker" ] && continue
          if ! _oc_timeout 60 env OPENCODE_CONFIG_CONTENT="$OC_CFG" "$OC_BIN" debug agent "$_agname" \
                 >/dev/null 2>&1; then
            err "opencode did not resolve agent $_agname from the injected config (or the probe timed out)"
            oc_ok=false
            break
          fi
        done
      fi
      # `debug skill` emits ~350 KB of JSON. Piping that into `grep -q` is NOT a sound
      # probe: grep exits at the first match and the resulting SIGPIPE truncates the
      # stream, so the verdict depends on scheduling — it reported "skills missing" for a
      # config whose skills in fact all resolve. Redirect to a file, then inspect.
      # Explicit XXXXXX template, not `mktemp -t PREFIX`: BSD/macOS treats -t's argument as
      # a PREFIX, GNU coreutils treats it as a TEMPLATE and fails with "too few X's in
      # template". Under `set -e` that aborted the install before the manifest was written —
      # invisible on macOS, fatal in the Linux containers this plugin actually targets.
      if [ "$oc_ok" = true ]; then
        OC_SKILLS="$(mktemp "${TMPDIR:-/tmp}/cannbot_oc_skills.XXXXXX")"
        if ! _oc_timeout 180 env OPENCODE_CONFIG_CONTENT="$OC_CFG" "$OC_BIN" debug skill >"$OC_SKILLS" 2>/dev/null; then
          err "opencode skill listing probe failed or timed out"
          oc_ok=false
        fi
      fi
      if [ "$oc_ok" = true ] && ! python3 - "$OC_SKILLS" <<'PYSKILL'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception as exc:
    print(f"  could not read opencode skill listing: {exc}"); sys.exit(1)
names = {s.get("name") for s in data if isinstance(s, dict)}
missing = [w for w in ("ascendc-cross-gen-port", "ascendc-backward-gen") if w not in names]
if missing:
    print(f"  opencode resolved {len(names)} skills but not: {missing}"); sys.exit(1)
PYSKILL
      then
        err "opencode did not resolve the plugin entry skills from the injected config"
        oc_ok=false
      fi
      [ -n "${OC_SKILLS:-}" ] && rm -f "$OC_SKILLS"
    fi
  fi
  if [ "$oc_ok" = true ]; then
    ok "opencode resolves injected agents + skills (structural)"
    # BEHAVIOURAL. Driven by src/opencode/probe_safety_net.mjs, which calls the adapter's real
    # `tool.execute.before` in the real JS runtime and requires a deny/allow PAIR. It needs no
    # model, so unlike the earlier design it runs on virtually every install rather than
    # never — the previous block described this level in its comments but never implemented
    # it, so every opencode install fell through to the refusal below and the override turned
    # into a required incantation. A bypass everyone is forced to set protects nothing.
    OC_JS="$(js_runtime)" || OC_JS=""   # canonical node→bun rule, shared with the engine self-check
    if [ -z "$OC_JS" ]; then
      # opencode itself ships as a bun/node program, so this is close to unreachable in a
      # working install; it stays a warning rather than a failure because the runtime gate is
      # the load-bearing one and refusing here would only strand the operator.
      warn "no node/bun runtime found — cannot run the behavioural safety-net proof"
      warn "OpenCode runtime self-check will refuse dispatch until node/bun is installed"
    else
      # Keep the exact verdict contract in lockstep with the backend runtime
      # self-check: (rc=0, output=OK) passes; (rc=2, output=SKIP:...) is the
      # only non-fatal setup result. Capture rc inside an `if` so errexit does
      # not discard it before we can make the same decision as Python.
      if OC_PROBE_OUT="$("$OC_JS" "$PLUGIN_DIR/engine/src/opencode/probe_safety_net.mjs" 2>&1)"; then
        OC_PROBE_RC=0
      else
        OC_PROBE_RC=$?
      fi
      OC_PROBE_OUT="$(trim_probe_output "$OC_PROBE_OUT")"
      if [ "$OC_PROBE_RC" -eq 0 ] && [ "$OC_PROBE_OUT" = "OK" ]; then
        ok "safety net ENFORCES (cross-workspace read refused, own-workspace read allowed)"
        hooks_live=true
      elif [ "$OC_PROBE_RC" -eq 2 ] && [[ "$OC_PROBE_OUT" == SKIP:* ]]; then
        # The probe could not be SET UP on this machine (read-only temp, no symlink
        # permission). That is not evidence about the guards, so it must not be reported as
        # their failure — and blocking the install here would strand the operator over
        # something they cannot fix.
        warn "behavioural safety-net proof could not run here: ${OC_PROBE_OUT#SKIP: }"
        warn "Phase O0 re-proves it, with a deny/allow pair, before the first agent of every run"
      else
        # Distinct from the structural failure above: opencode accepted the config, but the
        # guards do not enforce. Installing that is installing a decoration.
        err "behavioural safety-net proof FAILED — the adapter does not enforce"
        err "  ${OC_PROBE_OUT}"
        health_ok=false
      fi
    fi
  else
    health_ok=false
  fi
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
# G4: harness dependency checklist (warn-level misses were already surfaced above).
case "$TOOL" in
  claude)
    if command -v "${CLAUDE_BIN:-claude}" >/dev/null 2>&1; then
      echo -e "  ${DIM}harness deps: ${GREEN}claude ✓${NC}"
    else
      echo -e "  ${DIM}harness deps: ${RED}claude ✗ (engine dispatch will fail until installed)${NC}"
    fi ;;
  opencode)
    _js="$(js_runtime)" || _js=""
    if command -v "${AOG_OPENCODE_BIN:-opencode}" >/dev/null 2>&1 && [ -n "$_js" ]; then
      echo -e "  ${DIM}harness deps: ${GREEN}opencode ✓ + $_js ✓${NC}"
    else
      echo -e "  ${DIM}harness deps: ${RED}opencode/js-runtime ✗ (engine dispatch will fail until installed)${NC}"
    fi ;;
esac
echo ""
if [ "$TOOL" = "opencode" ]; then
  echo -e "  ${BOLD}Quick start (opencode):${NC}"
  echo -e "  ${CYAN}1.${NC} fill ${GREEN}engine/workspace/.ascendc_env${NC} (A3/A5 host+container) — docs/USAGE.md"
  echo -e "  ${CYAN}2.${NC} launch ${GREEN}opencode${NC} in your project, then a customer entry command:"
  echo -e "       ${GREEN}/ascendc-cross-gen-port <ops-nn op dir>${NC}   (→ orch --port-a3)"
  echo -e "       ${GREEN}/ascendc-backward-gen <forward spec>${NC}      (→ orch --backward)"
else
  echo -e "  ${BOLD}Quick start (Claude Code):${NC}"
  echo -e "  ${CYAN}1.${NC} fill ${GREEN}engine/workspace/.ascendc_env${NC} (A3/A5 host+container) — docs/USAGE.md"
  echo -e "  ${CYAN}2.${NC} launch ${GREEN}claude${NC}, then a customer entry skill:"
  echo -e "       ${GREEN}/ascendc-cross-gen-port <ops-nn op dir>${NC}   (→ orch --port-a3)"
  echo -e "       ${GREEN}/ascendc-backward-gen <forward spec>${NC}      (→ orch --backward)"
fi
echo -e "  ${DIM}Pipeline logic lives entirely in engine/; the entry skills are thin NL front-ends.${NC}"
echo ""
