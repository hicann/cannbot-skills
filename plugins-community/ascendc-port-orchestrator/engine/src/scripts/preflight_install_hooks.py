#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Compatibility checker for ascendc-port-orchestrator hook registration.

The community plugin has two supported registration surfaces:

* marketplace install: plugin-native ``hooks/hooks.json``;
* direct checkout: owner-tagged settings written by ``init.sh`` into the
  engine cwd (and the selected Claude config root).

The legacy a5_ops writer remains below only so old registrations can be
removed. New installs must run the plugin's ``init.sh``; the default command
fails closed instead of writing the obsolete five-hook/stamp contract.
``--check`` is read-only and validates the complete active surface:

* marketplace mode: the enabled plugin path and all nine native hooks;
* direct-checkout mode: all eight owner-tagged project hooks.

CLI:
  python3 preflight_install_hooks.py --check
  python3 preflight_install_hooks.py --uninstall [--dry-run] [--non-interactive]

  --check              Validate current registration without modifying files
  --uninstall          Remove hooks + version stamp + restore backups; do NOT install
  --dry-run            Show what would be done; don't write
  --non-interactive    Auto-yes on user-global path/hash matches (USE CAREFULLY)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ROOT = REPO_ROOT.parent
PLUGIN_HOOKS = PLUGIN_ROOT / "hooks" / "hooks.json"
REPO_OPS_ROOT = (PLUGIN_ROOT / "../../ops").resolve()
# 2026-07-05: hooks relocated out of the (now-removed) engine/src/skills/ tree.
SKILL_DIR = REPO_ROOT / "src" / "scripts" / "hooks"
MANIFEST_PATH = SKILL_DIR / "manifest.json"
VERSION_PATH = SKILL_DIR / "version"
LEGACY_PATTERNS_PATH = SKILL_DIR / "legacy_patterns.json"

PROJECT_SETTINGS = REPO_ROOT / ".claude" / "settings.json"
PROJECT_VERSION_STAMP = REPO_ROOT / ".claude" / ".ascendc_hooks_version"
USER_GLOBAL_SETTINGS = Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))) / "settings.json"

ENV_FILE = REPO_ROOT / "workspace" / ".ascendc_env"

OWNER_TAG = "ascendc-op-gen"


def log(msg: str, indent: int = 0):
    print(("  " * indent) + msg, flush=True)


def warn(msg: str):
    print(f"  ⚠ {msg}", file=sys.stderr, flush=True)


def err(msg: str):
    print(f"  ❌ {msg}", file=sys.stderr, flush=True)


def read_json(p: Path) -> dict:
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        warn(f"settings.json malformed at {p}: {e}; treating as empty")
        return {}


def write_json(p: Path, data: dict, dry_run: bool):
    if dry_run:
        log(f"[dry-run] would write {p}", 1)
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n")


def backup_file(p: Path, dry_run: bool):
    if not p.is_file():
        return None
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = p.with_name(p.name + f".bak.{ts}")
    if dry_run:
        log(f"[dry-run] would backup {p} → {backup}", 1)
        return backup
    shutil.copy2(p, backup)
    log(f"backed up {p.name} → {backup.name}", 1)
    return backup


def _configured_benchmark_branch() -> str:
    """Read the optional benchmark branch without changing its legacy default."""
    branch = "br_430"
    if not ENV_FILE.is_file():
        return branch
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith("BENCHMARK_BRANCH="):
            configured_branch = line.split("=", 1)[1].strip().strip('"').strip("'")
            return configured_branch or branch
    return branch


def _run_submodule_checkout(git_executable: str, branch: str):
    """Update the legacy submodule using the resolved Git executable."""
    submodule_path = REPO_ROOT / "vendor" / "AscendOpGenAgent"
    subprocess.run(
        [git_executable, "submodule", "update", "--init", "vendor/AscendOpGenAgent"],
        cwd=str(REPO_ROOT),
        check=True,
    )
    subprocess.run(
        [git_executable, "-C", str(submodule_path), "fetch", "origin"],
        check=False,
    )
    subprocess.run(
        [git_executable, "-C", str(submodule_path), "checkout", branch],
        check=True,
    )


def _legacy_git_error(error: subprocess.CalledProcessError, git_executable: str) -> str:
    """Render a resolved Git failure as the legacy command-name diagnostic."""
    command = error.cmd
    if isinstance(command, (list, tuple)) and command and command[0] == git_executable:
        error = subprocess.CalledProcessError(
            error.returncode,
            ["git", *command[1:]],
            output=error.output,
            stderr=error.stderr,
        )
    return str(error)


def init_submodule(dry_run: bool) -> int:
    log("step 1: initialize vendor/AscendOpGenAgent submodule + checkout configured branch")
    if dry_run:
        log("[dry-run] would run: git submodule update --init vendor/AscendOpGenAgent", 1)
        return 0
    branch = _configured_benchmark_branch()
    git_executable = shutil.which("git") or "git"
    try:
        _run_submodule_checkout(git_executable, branch)
        log(f"submodule on branch {branch}", 1)
    except subprocess.CalledProcessError as e:
        err(f"submodule init failed: {_legacy_git_error(e, git_executable)}")
        return 1
    return 0


def symlink_shim(dry_run: bool):
    log("step 2: symlink ~/workspace/AscendOpGenAgent → vendor/AscendOpGenAgent")
    target = REPO_ROOT / "vendor" / "AscendOpGenAgent"
    link = Path.home() / "workspace" / "AscendOpGenAgent"
    if not target.is_dir():
        err(f"target {target} does not exist; submodule init failed")
        return
    if link.is_symlink():
        if link.resolve() == target.resolve():
            log("symlink already correct, skipping", 1)
            return
        if dry_run:
            log(f"[dry-run] would replace existing symlink {link} (currently → {link.resolve()})", 1)
        else:
            link.unlink()
            link.symlink_to(target)
            log(f"replaced symlink: {link} → {target}", 1)
        return
    if link.is_dir():
        # User already has a real dir there; back it up
        bak = link.with_name(link.name + f".bak.{time.strftime('%Y%m%d_%H%M%S')}")
        if dry_run:
            log(f"[dry-run] would backup {link} → {bak}, then symlink to {target}", 1)
        else:
            shutil.move(str(link), str(bak))
            link.symlink_to(target)
            log(f"backed up existing {link.name} → {bak.name}; symlinked to {target}", 1)
        return
    # Doesn't exist
    if dry_run:
        log(f"[dry-run] would create symlink {link} → {target}", 1)
    else:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
        log(f"created symlink {link} → {target}", 1)


def is_ours(hook_entry: dict) -> bool:
    return hook_entry.get("_owner") == OWNER_TAG


def matches_legacy_path(hook_entry: dict, patterns: list[str]) -> str | None:
    cmd = hook_entry.get("command", "")
    for p in patterns:
        if p in cmd:
            return p
    return None


def _remove_legacy_hook(hook_entry: dict, kind: str, matcher: str, file_label: str,
                        legacy_patterns: list[str], interactive: bool) -> bool:
    """Report and classify one obsolete hook without modifying its containing block."""
    if is_ours(hook_entry):
        log(f"[L1 tagged] removing {kind} {matcher} from {file_label}", 2)
        return True
    pattern = matches_legacy_path(hook_entry, legacy_patterns)
    if not pattern:
        return False
    if file_label == "project-local":
        log(
            f"[L2 path] removing {kind} {matcher} (matches {pattern}) from "
            f"{file_label}",
            2,
        )
        return True
    if not interactive:
        log(f"[L2 path SKIPPED non-interactive] {kind} {matcher} in {file_label}", 2)
        return False
    answer = input(
        f"  Remove {kind} hook from ~/.claude/settings.json? "
        f"matched pattern: {pattern}\n    command: "
        f"{hook_entry.get('command', '')[:120]}\n    [y/N]: "
    ).strip().lower()
    return answer == "y"


def _cleanup_hook_block(block: dict, kind: str, legacy_patterns: list[str],
                        file_label: str, interactive: bool) -> tuple[dict | None, int]:
    """Return the retained hook block and number of entries removed from it."""
    matcher = block.get("matcher", "-")
    retained_hooks = []
    removed = 0
    for hook_entry in block.get("hooks", []):
        if _remove_legacy_hook(
            hook_entry, kind, matcher, file_label, legacy_patterns, interactive
        ):
            removed += 1
        else:
            retained_hooks.append(hook_entry)
    if not retained_hooks:
        return None, removed
    block["hooks"] = retained_hooks
    return block, removed


def cleanup_settings(settings: dict, legacy_patterns: list[str], file_label: str,
                     interactive: bool, dry_run: bool) -> int:
    """Remove obsolete entries from settings dict (mutates in place). Returns count removed.

    L1 (tagged _owner=ascendc-op-gen): auto-remove
    L2 (path matches legacy_patterns): interactive confirm if file_label=user-global,
        auto-remove if file_label=project-local
    L3 (hash match): TODO when body_hashes populated
    """
    if "hooks" not in settings:
        return 0
    removed = 0
    for kind in list(settings["hooks"].keys()):
        retained_blocks = []
        for block in settings["hooks"][kind]:
            retained_block, block_removed = _cleanup_hook_block(
                block, kind, legacy_patterns, file_label, interactive
            )
            removed += block_removed
            if retained_block is not None:
                retained_blocks.append(retained_block)
        if retained_blocks:
            settings["hooks"][kind] = retained_blocks
        else:
            del settings["hooks"][kind]
    if not settings["hooks"]:
        del settings["hooks"]
    return removed


def merge_manifest(settings: dict, manifest: dict):
    """Merge canonical hook entries from manifest into settings (in place).

    Deduplication: drop any existing hook in the matched block whose `command`
    matches a new one's `command` (whether tagged or not). This handles the
    pre-v3.7.0 untagged entries that share the same path as the new tagged
    entries — without this, a re-install would create duplicates.
    """
    if "hooks" not in manifest:
        return
    settings.setdefault("hooks", {})
    for kind, blocks in manifest["hooks"].items():
        existing = settings["hooks"].setdefault(kind, [])
        for new_block in blocks:
            existing_block = next((b for b in existing if b.get("matcher") == new_block.get("matcher")), None)
            if existing_block is None:
                existing.append(new_block)
                continue
            new_commands = {h.get("command", "") for h in new_block.get("hooks", [])}
            # Drop existing entries whose command is being added by this manifest
            existing_block["hooks"] = [
                h for h in existing_block.get("hooks", [])
                if h.get("command", "") not in new_commands
            ]
            existing_block["hooks"].extend(new_block.get("hooks", []))


def manifest_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _legacy_cmd_install(args) -> int:
    if not MANIFEST_PATH.is_file():
        err(f"missing canonical manifest at {MANIFEST_PATH}; aborting")
        return 1
    if not VERSION_PATH.is_file():
        err(f"missing version file at {VERSION_PATH}; aborting")
        return 1

    version = VERSION_PATH.read_text().strip()
    manifest = json.loads(MANIFEST_PATH.read_text())
    legacy = json.loads(LEGACY_PATTERNS_PATH.read_text()).get(
        "path_patterns", []) if LEGACY_PATTERNS_PATH.is_file() else []

    log(f"=== preflight_install_hooks ({version}, dry_run={args.dry_run}) ===")
    if init_submodule(args.dry_run) != 0:
        return 1
    symlink_shim(args.dry_run)

    # 3. Backup
    log("step 3: backup settings.json files")
    backup_file(PROJECT_SETTINGS, args.dry_run)
    backup_file(USER_GLOBAL_SETTINGS, args.dry_run)

    # 4. Cleanup
    log("step 4: cleanup obsolete entries")
    proj = read_json(PROJECT_SETTINGS)
    user = read_json(USER_GLOBAL_SETTINGS)
    proj_removed = cleanup_settings(proj, legacy, "project-local", interactive=False, dry_run=args.dry_run)
    user_removed = cleanup_settings(user, legacy, "user-global",
                                     interactive=not args.non_interactive, dry_run=args.dry_run)
    log(f"removed {proj_removed} project-local + {user_removed} user-global stale entries", 1)

    # 5. Install canonical
    log("step 5: install canonical hooks at project-local .claude/settings.json")
    merge_manifest(proj, manifest)
    write_json(PROJECT_SETTINGS, proj, args.dry_run)
    if user_removed > 0:
        write_json(USER_GLOBAL_SETTINGS, user, args.dry_run)

    # 6. Stamp version
    log("step 6: write .claude/.ascendc_hooks_version stamp")
    stamp = {"version": version, "manifest_sha256": manifest_sha256(MANIFEST_PATH)}
    write_json(PROJECT_VERSION_STAMP, stamp, args.dry_run)

    # 7. Final message
    log("=== preflight install complete ===")
    log(f"   hooks installed at: {PROJECT_SETTINGS}")
    log(f"   version stamp:      {PROJECT_VERSION_STAMP}")
    log("")
    log("⚠ Restart Claude Code NOW for hooks to activate (settings.json is read at session start).")
    return 0


def _hook_entries(hooks: dict, event: str, owner: str | None = None) -> list[tuple[str, dict]]:
    entries = []
    for matcher in hooks.get(event, []):
        matcher_name = matcher.get("matcher", "")
        for hook in matcher.get("hooks", []):
            if owner is None or hook.get("_owner") == owner:
                entries.append((matcher_name, hook))
    return entries


def _project_context_root() -> Path:
    """Return the project whose Claude settings spawned this orchestrator.

    Claude Code exposes ``CLAUDE_PROJECT_DIR`` to plugin processes.
    ``ASCENDC_PORT_PROJECT_ROOT`` is a fallback for non-interactive launchers
    that do not receive Claude's project variable. Direct CLI use falls back to
    its current directory, so callers launched from ``engine/`` must set that
    fallback to the customer project explicitly.
    """
    raw = (
        os.environ.get("CLAUDE_PROJECT_DIR")
        or os.environ.get("ASCENDC_PORT_PROJECT_ROOT")
        or os.getcwd()
    )
    return Path(raw).expanduser().resolve()


def _registered_marketplace_plugins(plugin_id: str) -> tuple[list[dict], str | None]:
    """Read Claude's install registry when the checker runs outside a project cwd.

    ``claude plugin list --json`` is project-cwd scoped.  The orchestrator runs
    from the packaged ``engine/`` directory, so a healthy *project*-scope plugin
    otherwise appears absent even though Claude has recorded both its install and
    the project ``enabledPlugins`` switch.  The registry is Claude's durable
    install record; pair it with the switch for the *running project* before
    accepting it.  A switch from another project must not arm hooks for a
    worker that Claude launches from this project.
    """
    config_root = Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))
    registry = config_root / "plugins" / "installed_plugins.json"
    try:
        raw = json.loads(registry.read_text())
        entries = raw.get("plugins", {}).get(plugin_id, [])
    except (OSError, json.JSONDecodeError) as exc:
        return [], f"marketplace install registry unreadable: {exc}"
    if not isinstance(entries, list):
        return [], "marketplace install registry has an invalid plugin entry"

    expected_project = _project_context_root()
    observed: list[dict] = []
    context_errors: list[str] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        candidate = dict(item)
        if item.get("scope") == "project":
            project_path = item.get("projectPath")
            if not isinstance(project_path, str) or not project_path:
                candidate["enabled"] = False
                context_errors.append("project-scope registry entry has no projectPath")
                observed.append(candidate)
                continue
            registered_project = Path(project_path).expanduser().resolve()
            if registered_project != expected_project:
                candidate["enabled"] = False
                context_errors.append(
                    "project-scope registry path "
                    f"{registered_project} does not match running project {expected_project}"
                )
                observed.append(candidate)
                continue
            settings = registered_project / ".claude" / "settings.json"
        else:
            settings = config_root / "settings.json"
        enabled_plugins = read_json(settings).get("enabledPlugins", {}) if settings else {}
        candidate["enabled"] = enabled_plugins.get(plugin_id) is True
        observed.append(candidate)
    return observed, "; ".join(context_errors) if context_errors else None


def _registration_surface() -> tuple[bool, str, Path]:
    """Select the hook declaration matching the installed registration mode."""
    direct_checkout = REPO_OPS_ROOT.is_dir()
    registration = "direct-settings" if direct_checkout else "plugin"
    declaration = PROJECT_SETTINGS if direct_checkout else PLUGIN_HOOKS
    return direct_checkout, registration, declaration


def _plugin_manifest_errors() -> tuple[list[str], bool]:
    """Return manifest diagnostics and whether the declaration check must stop."""
    manifest_path = PLUGIN_ROOT / ".claude-plugin" / "plugin.json"
    try:
        plugin_manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"plugin manifest unreadable: {exc}"], True
    if plugin_manifest.get("name") != "ascendc-port-orchestrator":
        return ["plugin manifest name is not ascendc-port-orchestrator"], False
    return [], False


def _load_hooks(declaration: Path) -> tuple[dict | None, str | None]:
    """Read one hook declaration while retaining the public diagnostic text."""
    try:
        return json.loads(declaration.read_text()).get("hooks", {}), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"declaration unreadable: {exc}"


def _observed_hooks(hooks: dict, owner: str | None) -> list[tuple[str, str, dict]]:
    """Flatten hook declarations across the events checked by this gate."""
    observed = []
    for event in ("SessionStart", "PreToolUse", "PostToolUse", "SubagentStop"):
        observed.extend((event, matcher, hook) for matcher, hook in _hook_entries(hooks, event, owner))
    return observed


def _expected_hook_entries(direct_checkout: bool) -> list[tuple[str, str, str, int | None]]:
    """Return the immutable hook closure required by each registration mode."""
    expected = [
        ("PreToolUse", "Task|Agent", "workflow_critic.py", 30),
        ("PreToolUse", "Edit|Write|MultiEdit", "workflow_critic.py", 15),
        ("PreToolUse", "Edit|Write|MultiEdit", "agent-gate-dispatch.py pretool", 10),
        ("PreToolUse", "Bash", "workflow_critic.py", 10),
        ("PreToolUse", "Read|Grep|Glob|Bash", "output_read_guard.py", 10),
        ("PreToolUse", "WebFetch", "workflow_critic.py", 5),
        ("PostToolUse", "Task|Agent", "ship_claim_audit.py", 30),
        ("SubagentStop", "", "agent-gate-dispatch.py stop", 60),
    ]
    if not direct_checkout:
        expected.insert(
            0,
            (
                "SessionStart",
                "startup|resume|clear|compact",
                "session-start-ascendc-port-orchestrator",
                None,
            ),
        )
    return expected


def _matches_expected_hook(item: tuple[str, str, dict],
                           wanted: tuple[str, str, str, int | None]) -> bool:
    """Return whether one observed hook fulfills one required hook entry."""
    event, matcher, hook = item
    want_event, want_matcher, needle, timeout = wanted
    command = hook.get("command", "")
    return (
        event == want_event
        and matcher == want_matcher
        and all(part in command for part in needle.split())
        and (timeout is None or hook.get("timeout") == timeout)
    )


def _expected_hook_errors(observed: list[tuple[str, str, dict]],
                          expected: list[tuple[str, str, str, int | None]]) -> list[str]:
    """Validate required entries and retain the prior error ordering and wording."""
    errors = []
    for wanted in expected:
        count = sum(_matches_expected_hook(item, wanted) for item in observed)
        if count != 1:
            errors.append(
                f"expected exactly one {wanted[0]} matcher={wanted[1]!r} "
                f"command={wanted[2]!r}, found {count}"
            )
    if len(observed) != len(expected):
        errors.append(f"hook closure size is {len(observed)}, expected {len(expected)}")
    return errors


def _hook_type_errors(observed: list[tuple[str, str, dict]]) -> list[str]:
    """Check command-only and synchronous-session invariants."""
    errors = []
    for event, _, hook in observed:
        if hook.get("type") != "command":
            errors.append(f"{event} contains a non-command hook")
        if event == "SessionStart" and hook.get("async") is not False:
            errors.append("SessionStart hook must be synchronous")
    return errors


def _direct_checkout_errors(hooks: dict) -> list[str]:
    """Return direct-checkout-only legacy-owner violations."""
    legacy = [
        hook.get("command", "")
        for event in hooks
        for _, hook in _hook_entries(hooks, event)
        if hook.get("_owner") == "ascendc-op-gen"
    ]
    if legacy:
        return [f"legacy ascendc-op-gen owner hooks remain: {len(legacy)}"]
    return []


def _plugin_cli_entries() -> tuple[list[dict], str | None]:
    """Inspect the CLI's current plugin view without making it authoritative alone."""
    claude = shutil.which("claude")
    if not claude:
        return [], "claude CLI unavailable; cannot verify plugin activation"
    try:
        result = subprocess.run(
            [claude, "plugin", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        listed = json.loads(result.stdout) if result.returncode == 0 else []
        if not isinstance(listed, list):
            listed = []
        entries = [
            item for item in listed
            if isinstance(item, dict) and item.get("id") == "ascendc-port-orchestrator@cannbot"
        ]
        error = None if result.returncode == 0 else "claude plugin list --json failed"
        return entries, error
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return [], f"failed to inspect enabled plugins: {exc}"


def _active_plugin_candidates() -> tuple[list[dict], str | None, str | None]:
    """Use the registry fallback only when the CLI has no enabled target."""
    cli_entries, cli_error = _plugin_cli_entries()
    registry_error = None
    if any(item.get("enabled") is True for item in cli_entries):
        return cli_entries, cli_error, registry_error
    registry_entries, registry_error = _registered_marketplace_plugins(
        "ascendc-port-orchestrator@cannbot"
    )
    return registry_entries or cli_entries, cli_error, registry_error


def _plugin_activation_errors() -> list[str]:
    """Validate the one enabled marketplace installation expected by this package."""
    candidates, cli_error, registry_error = _active_plugin_candidates()
    active = [item for item in candidates if item.get("enabled") is True]
    if len(active) != 1:
        detail = "ascendc-port-orchestrator@cannbot is not installed and enabled exactly once"
        details = [detail]
        if registry_error:
            details.append(registry_error)
        if cli_error:
            details.append(cli_error)
        detail = " (".join([details[0], "; ".join(details[1:]) + ")"]) \
            if len(details) > 1 else details[0]
        return [detail]
    try:
        install_path = active[0]["installPath"]
        if not isinstance(install_path, str) or not install_path:
            raise TypeError("installPath is not a non-empty string")
        active_root = Path(install_path).resolve()
    except (KeyError, OSError, TypeError) as exc:
        return [f"enabled plugin installPath is invalid: {exc}"]
    if active_root != PLUGIN_ROOT.resolve():
        return [
            f"enabled plugin path is {active_root}, current package is "
            f"{PLUGIN_ROOT.resolve()}"
        ]
    return []


def check_current_registration() -> tuple[str, Path, list[str]]:
    """Return ``(mode, declaration, errors)`` without modifying any file."""
    direct_checkout, registration, declaration = _registration_surface()
    errors, manifest_unreadable = _plugin_manifest_errors()
    if manifest_unreadable:
        return registration, declaration, errors

    hooks, declaration_error = _load_hooks(declaration)
    if declaration_error:
        return registration, declaration, errors + [declaration_error]
    owner = "ascendc-port-orchestrator" if direct_checkout else None
    observed = _observed_hooks(hooks, owner)
    errors.extend(_expected_hook_errors(observed, _expected_hook_entries(direct_checkout)))
    errors.extend(_hook_type_errors(observed))

    if _hook_entries(hooks, "Stop", owner=owner):
        errors.append("agent gate incorrectly registered on Stop (must be SubagentStop)")
    if direct_checkout:
        errors.extend(_direct_checkout_errors(hooks))
    else:
        errors.extend(_plugin_activation_errors())

    return registration, declaration, errors


class LegacyInstallerDisabled(RuntimeError):
    """The obsolete hook writer was invoked; init.sh owns installation now."""


def cmd_install(args) -> int:
    """Refuse the obsolete writer; current installs are owned by init.sh."""
    init_sh = PLUGIN_ROOT / "init.sh"
    message = (
        "legacy hook installer is disabled for this community plugin; "
        f"run `bash {init_sh} global claude`, then restart Claude Code"
    )
    print(f"ERROR {message}")
    # A library function must not terminate the interpreter; ``main`` is the
    # only entry allowed to turn this into a process exit status.
    raise LegacyInstallerDisabled(message)


def cmd_check(args) -> int:
    """Validate the single hook surface selected by ``init.sh``.

    The historic verdict token ``STAMP_OK`` is retained because Mode 5 parses
    it. No legacy stamp is required: the package declaration itself is the
    marketplace source of truth, while direct checkouts are validated against
    the engine-cwd settings Claude actually loads for spawned agents.
    """

    registration, declaration, errors = check_current_registration()
    if errors:
        print(
            f"CONTENT_DRIFT registration={registration} declaration={declaration} "
            f"missing_or_invalid={errors}; run plugin init.sh and restart Claude Code"
        )
        return 2

    digest = hashlib.sha256(declaration.read_bytes()).hexdigest()[:12]
    print(
        f"STAMP_OK registration={registration} declaration={declaration} "
        f"sha={digest}"
    )
    return 0


def cmd_uninstall(args) -> int:
    log(f"=== preflight_install_hooks UNINSTALL (dry_run={args.dry_run}) ===")
    backup_file(PROJECT_SETTINGS, args.dry_run)
    backup_file(USER_GLOBAL_SETTINGS, args.dry_run)
    legacy = json.loads(LEGACY_PATTERNS_PATH.read_text()).get(
        "path_patterns", []) if LEGACY_PATTERNS_PATH.is_file() else []
    proj = read_json(PROJECT_SETTINGS)
    user = read_json(USER_GLOBAL_SETTINGS)
    proj_removed = cleanup_settings(proj, legacy, "project-local", interactive=False, dry_run=args.dry_run)
    user_removed = cleanup_settings(user, legacy, "user-global",
                                     interactive=not args.non_interactive, dry_run=args.dry_run)
    log(f"removed {proj_removed} project-local + {user_removed} user-global entries", 1)
    write_json(PROJECT_SETTINGS, proj, args.dry_run)
    if user_removed > 0:
        write_json(USER_GLOBAL_SETTINGS, user, args.dry_run)
    if PROJECT_VERSION_STAMP.is_file() and not args.dry_run:
        PROJECT_VERSION_STAMP.unlink()
        log(f"removed {PROJECT_VERSION_STAMP}", 1)
    log("=== uninstall complete; restart Claude Code ===")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="Diagnose drift without modifying anything. Exits 0 if "
                         "STAMP_OK, or 2 if CONTENT_DRIFT (run init.sh and "
                         "restart Claude Code).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--non-interactive", action="store_true")
    args = ap.parse_args()
    if args.check:
        return cmd_check(args)
    if args.uninstall:
        return cmd_uninstall(args)
    try:
        return cmd_install(args)
    except LegacyInstallerDisabled:
        # cmd_install already printed the operator-facing ERROR line.
        return 2


if __name__ == "__main__":
    sys.exit(main())
