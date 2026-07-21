#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# Runtime dependency: pyyaml (declared in the plugin requirements.txt).
"""
Unified CLI for remote NPU development.

Supports multiple backends (docker, hdspace) via .npus.yaml config.

Usage:
    npu.py sync push [TARGET] [PATH]        # Local → Remote
    npu.py sync pull TARGET PATH            # Remote → Local
    npu.py sync diff [TARGET]               # Show stale files
    npu.py sync clean TARGET PATH [--force] # Delete remote file
    npu.py exec TARGET COMMAND              # Run command on remote
    npu.py exec TARGET info                 # Probe NPU platform
    npu.py list                             # List configured remotes
    npu.py -w .claude/worktrees/atan2 sync push  # Worktree isolation
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml  # pip install pyyaml

logger = logging.getLogger(__name__)


class NpuError(Exception):
    """Raised by helper functions for fatal CLI errors.

    Carries the message (logged at the top-level entry point) and the process
    exit code to use (defaults to 1). The CLI entry point (`main`) catches this,
    logs the message, and exits with `code`, preserving prior CLI behavior.
    """

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_./\-][A-Za-z0-9_./\- ]*$")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def find_repo_root() -> Path:
    """Find git repo root, handling both normal repos (.git dir) and worktrees (.git file)."""
    p = Path.cwd()
    while p != p.parent:
        git = p / ".git"
        if git.is_dir() or git.is_file():
            return p
        p = p.parent
    return Path.cwd()


def find_main_repo_root() -> Path:
    """Find the main repo root (not worktree). Used for loading .npus.yaml."""
    root = find_repo_root()
    git = root / ".git"
    if git.is_file():
        # Worktree: .git is a file like "gitdir: /path/to/main/.git/worktrees/name"
        content = git.read_text().strip()
        if content.startswith("gitdir:"):
            gitdir = Path(content.split(":", 1)[1].strip())
            if not gitdir.is_absolute():
                gitdir = (root / gitdir).resolve()
            # gitdir is like /main/.git/worktrees/name → main repo root is gitdir/../../..
            # But more robust: use commondir
            commondir_file = gitdir / "commondir"
            if commondir_file.exists():
                commondir = commondir_file.read_text().strip()
                main_git = (gitdir / commondir).resolve()
                return main_git.parent
            # Fallback: strip /worktrees/<name> from path
            if "worktrees" in gitdir.parts:
                idx = gitdir.parts.index("worktrees")
                return Path(*gitdir.parts[:idx]).parent
    return root


def load_config(main_root: Path | None = None) -> dict:
    root = main_root or find_main_repo_root()
    cfg_path = root / ".npus.yaml"
    if not cfg_path.exists():
        # Backward compatibility: fall back to .remotes.yaml
        cfg_path = root / ".remotes.yaml"
    if not cfg_path.exists():
        example = Path(__file__).parent / "npus.example.yaml"
        raise NpuError(f"Error: {root / '.npus.yaml'} not found.\nCopy from: {example}")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise NpuError(f"Error: {cfg_path} must contain a YAML mapping at the top level.")
    return config


def get_remote(config: dict, name: str | None) -> tuple[str, dict]:
    """Resolve remote by name or default."""
    remotes = config.get("remotes", {})
    if not name:
        name = config.get("default")
    if not name:
        if len(remotes) == 1:
            name = next(iter(remotes))
        else:
            raise NpuError(
                "Error: specify TARGET or set 'default' in .npus.yaml\n"
                f"Available: {', '.join(remotes.keys())}"
            )
    if name not in remotes:
        raise NpuError(f"Error: remote '{name}' not found. Available: {', '.join(remotes.keys())}")
    return name, remotes[name]


# ---------------------------------------------------------------------------
# Backend: resolve SSH host
# ---------------------------------------------------------------------------

def resolve_ssh_host(remote: dict) -> str:
    """Resolve SSH host alias. For hdspace, looks up ~/.devenv/.ssh/config."""
    backend = remote.get("backend", "docker")
    if backend == "docker":
        return remote["host"]
    elif backend == "hdspace":
        config_path = Path.home() / ".devenv" / ".ssh" / "config"
        if not config_path.exists():
            raise NpuError(f"Error: {config_path} not found. Run hdspace tunnel first.")
        name = remote.get("name")
        if not isinstance(name, str) or not name.strip():
            raise NpuError("Error: hdspace remote requires a non-empty 'name' in .npus.yaml")
        name = name.strip()
        name_lower = name.lower()
        with open(config_path) as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 2 or parts[0].lower() != "host":
                    continue
                # Match any alias on the `Host` line exactly, or match the container-name
                # prefix (hdspace aliases look like `{name}.{instance_id}.0`).
                for alias in parts[1:]:
                    alias_lower = alias.lower()
                    if alias_lower == name_lower or alias_lower.startswith(name_lower + "."):
                        return alias
        raise NpuError(f"Error: no hdspace host alias matching '{name}' in {config_path}")
    else:
        raise NpuError(f"Error: unknown backend '{backend}'")


def ssh_port_args(remote: dict) -> list[str]:
    """Return ['-p', 'PORT'] if port is explicitly configured, else [].

    hdspace's ~/.devenv/.ssh/config Port is unreliable (not updated on tunnel restart,
    overwritten on ssh-key-reset). Explicit port in .npus.yaml takes precedence.
    """
    port = remote.get("port")
    if port:
        return ["-p", str(port)]
    return []


def resolve_remote_repo(remote: dict, repo_name: str, worktree_name: str | None = None) -> str:
    """Resolve the container-side (or direct SSH) path for the repo.

    workdir is always the parent directory; repo_name is always appended.
    e.g. workdir=/workspace, repo=ops-math → /workspace/ops-math
    """
    workdir = remote.get("workdir", "/mnt/workspace").rstrip("/")
    base = f"{workdir}/{repo_name}"
    if worktree_name:
        return f"{base}.worktrees/{worktree_name}"
    return base


def resolve_host_repo(remote: dict, repo_name: str, worktree_name: str | None = None) -> str | None:
    """For docker backend: resolve the host-side path if host_workdir is configured.

    Same convention as resolve_remote_repo: host_workdir is parent, repo_name appended.
    Returns None if host_workdir not configured (fall back to tar-through-docker).
    """
    host_workdir = remote.get("host_workdir")
    if not host_workdir:
        return None
    base = f"{host_workdir.rstrip('/')}/{repo_name}"
    if worktree_name:
        return f"{base}.worktrees/{worktree_name}"
    return base


# ---------------------------------------------------------------------------
# SSH command builders
# ---------------------------------------------------------------------------

def _ssh_base(host: str, remote: dict) -> list[str]:
    """Build ['ssh', '-p', PORT, host] or ['ssh', host] base command."""
    return ["ssh", *ssh_port_args(remote), host]


def _rsync_ssh_opt(remote: dict) -> list[str]:
    """Build ['-e', 'ssh -p PORT'] for rsync, or [] if no explicit port."""
    port = remote.get("port")
    if port:
        return ["-e", f"ssh -p {port}"]
    return []


def ssh_exec(host: str, command: str, remote: dict, capture: bool = False) -> subprocess.CompletedProcess:
    """Execute command on remote via SSH."""
    backend = remote.get("backend", "docker")
    container = remote.get("container", "")
    login_shell = remote.get("login_shell", False)
    base = _ssh_base(host, remote)

    if backend == "docker" and container:
        # SSH to host, then docker exec
        inner = command.replace("'", "'\\''")
        if login_shell:
            ssh_cmd = [*base, f"docker exec -i '{container}' bash -l -c '{inner}'"]
        else:
            ssh_cmd = [*base, f"docker exec -i '{container}' bash -c '{inner}'"]
    elif backend == "hdspace" or (backend == "docker" and not container):
        # Direct SSH
        if login_shell or backend == "hdspace":
            ssh_cmd = [*base, "bash", "-l"]
        else:
            ssh_cmd = [*base, "bash"]
    else:
        ssh_cmd = [*base, command]

    if backend == "hdspace" or (backend == "docker" and not container):
        return subprocess.run(
            ssh_cmd,
            input=command,
            text=True,
            errors="replace",
            capture_output=capture,
        )
    else:
        return subprocess.run(ssh_cmd, capture_output=capture, text=True, errors="replace")


def build_cann_preamble(remote: dict) -> str:
    """Build CANN environment setup commands."""
    cann_path = remote.get("cann", "")
    lines = []
    if cann_path:
        lines.append(f'[ -f "{cann_path}/bin/setenv.bash" ] && source "{cann_path}/bin/setenv.bash"')
        lines.append(
            f'[ -f "{cann_path}/aarch64-linux/bin/setenv.bash" ] && source "{cann_path}/aarch64-linux/bin/setenv.bash"')
        lines.append(f'export ASCEND_HOME_PATH="{cann_path}"')
    else:
        # Auto-detect
        lines.append(
            'for p in /home/developer/Ascend/cann-*/aarch64-linux/bin/setenv.bash /usr/local/Ascend/cann-*/bin/setenv.bash /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash; do [ -f "$p" ] && source "$p" && break; done')
    lines.append(
        'export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}')
    device = remote.get("device_isolate")
    if device is not None:
        lines.append(f'export ASCEND_RT_VISIBLE_DEVICES={device}')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validate path
# ---------------------------------------------------------------------------

def run_checked(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command and exit on failure."""
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise NpuError(
            f"Error: command failed (exit {result.returncode}): {' '.join(cmd[:4])}...",
            code=result.returncode,
        )
    return result


def validate_path(path: str):
    if not path:
        raise NpuError("Error: path is empty")
    if os.path.isabs(path):
        raise NpuError(f"Error: path must be relative to repo root: {path}")
    parts = path.split("/")
    if ".." in parts:
        raise NpuError(f"Error: path must not contain '..': {path}")
    # Defend against option-injection (leading '-') and shell metacharacters
    # that could break the interpolated rsync/ssh/docker-cp commands.
    if path.startswith("-"):
        raise NpuError(f"Error: path must not start with '-': {path}")
    if not _SAFE_PATH_RE.match(path):
        raise NpuError(
            f"Error: path contains disallowed characters (allowed: letters, digits, '_', '-', '.', '/', space): {path}",
        )


# ---------------------------------------------------------------------------
# Sync commands
# ---------------------------------------------------------------------------

RSYNC_EXCLUDES = [
    # Protect cmake/third_party/ from .gitignore's `third_party/` pattern
    "--include=cmake/",
    "--include=cmake/third_party/",
    "--include=cmake/third_party/**",
    "--filter=:- .gitignore",
    "--exclude=.git/",
    "--exclude=__pycache__/",
    "--exclude=.claude/",
    "--exclude=.clangd",
    "--exclude=build/",
    "--exclude=build_out/",
    "--exclude=opp/",
]


def _rsync_push(host: str, remote: dict, src: Path, dst: str, is_dir: bool = True, excludes: list[str] | None = None):
    """Direct rsync to host:dst. Simple and fast."""
    run_checked([*_ssh_base(host, remote), f"mkdir -p {shlex.quote(dst)}"])
    cmd = ["rsync", "-az", *_rsync_ssh_opt(remote)]
    if excludes:
        cmd.extend(excludes)
    # `--` terminates option parsing so leading-dash paths can't be interpreted as flags.
    if is_dir:
        cmd.extend(["--", f"{src}/", f"{host}:{dst}/"])
    else:
        cmd.extend(["--", str(src), f"{host}:{dst}"])
    run_checked(cmd)


def cmd_sync_push(host: str, remote: dict, repo_root: Path, target_path: str | None, wt_name: str | None = None):
    repo_name = find_main_repo_root().name if wt_name else repo_root.name
    remote_repo = resolve_remote_repo(remote, repo_name, wt_name)
    backend = remote.get("backend", "docker")
    container = remote.get("container", "")
    # For docker: prefer direct rsync to host_workdir if configured
    host_repo = resolve_host_repo(remote, repo_name, wt_name)

    if target_path:
        validate_path(target_path)
        src = repo_root / target_path
        if not src.exists():
            raise NpuError(f"Error: {src} does not exist locally")

        if host_repo:
            # Direct rsync to host-side mount
            dst = f"{host_repo}/{target_path}"
            parent = os.path.dirname(dst) if src.is_file() else dst
            logger.info(f"Pushing {target_path} → {host}:{dst}")
            _rsync_push(host, remote, src, parent if src.is_file() else dst, is_dir=src.is_dir())
        elif backend == "docker" and container:
            # Fallback: rsync to staging, then docker cp
            dst = f"{remote_repo}/{target_path}"
            staging = f"/tmp/_npu_sync_{repo_name}"
            staging_target = f"{staging}/{target_path}"
            staging_parent = os.path.dirname(staging_target) or staging
            run_checked([*_ssh_base(host, remote), f"mkdir -p {shlex.quote(staging_parent)}"])
            if src.is_dir():
                run_checked(["rsync", "-az", *_rsync_ssh_opt(remote), "--", f"{src}/", f"{host}:{staging_target}/"])
            else:
                run_checked(["rsync", "-az", *_rsync_ssh_opt(remote), "--", str(src), f"{host}:{staging_target}"])
            # Ensure destination parent exists inside the container; otherwise `docker cp`
            # fails when the subdirectory hasn't been created yet (fresh container/new path).
            dst_parent = os.path.dirname(dst) or "/"
            run_checked([*_ssh_base(host, remote),
                         f"docker exec {shlex.quote(container)} mkdir -p {shlex.quote(dst_parent)} && "
                         f"docker cp {shlex.quote(staging_target)} {shlex.quote(container)}:{shlex.quote(dst)}"])
        else:
            dst = f"{remote_repo}/{target_path}"
            logger.info(f"Pushing {target_path} → {host}:{dst}")
            parent = os.path.dirname(dst) if src.is_file() else dst
            _rsync_push(host, remote, src, parent if src.is_file() else dst, is_dir=src.is_dir())
    else:
        if host_repo:
            # Direct rsync to host-side mount — fast, incremental
            logger.info(f"Syncing {repo_root} → {host}:{host_repo}")
            _rsync_push(host, remote, repo_root, host_repo, excludes=RSYNC_EXCLUDES)
        elif backend == "docker" and container:
            # Fallback: rsync → staging → tar into container
            logger.info(f"Syncing {repo_root} → {host}:{remote_repo} (via tar)")
            staging = f"/tmp/_npu_sync_{repo_name}"
            run_checked([*_ssh_base(host, remote), f"mkdir -p {staging}"])
            cmd = ["rsync", "-az", *_rsync_ssh_opt(remote), *RSYNC_EXCLUDES, f"{repo_root}/", f"{host}:{staging}/"]
            run_checked(cmd)
            run_checked([*_ssh_base(host, remote),
                         f"docker exec '{container}' mkdir -p '{remote_repo}' && "
                         f"cd '{staging}' && tar cf - . | docker exec -i '{container}' tar --no-same-owner -xf - -C '{remote_repo}'"])
        else:
            logger.info(f"Syncing {repo_root} → {host}:{remote_repo}")
            _rsync_push(host, remote, repo_root, remote_repo, excludes=RSYNC_EXCLUDES)
    logger.info("Push complete.")


def cmd_sync_pull(host: str, remote: dict, repo_root: Path, target_path: str, wt_name: str | None = None):
    validate_path(target_path)
    repo_name = find_main_repo_root().name if wt_name else repo_root.name
    remote_repo = resolve_remote_repo(remote, repo_name, wt_name)
    host_repo = resolve_host_repo(remote, repo_name, wt_name)
    backend = remote.get("backend", "docker")
    container = remote.get("container", "")

    if host_repo:
        # Direct rsync from host-side mount
        logger.info(f"Pulling {host}:{host_repo}/{target_path} → {repo_root / target_path}")
        run_checked(["rsync",
                     "-az",
                     *_rsync_ssh_opt(remote),
                     f"{host}:{host_repo}/{target_path}",
                     str(repo_root / target_path)])
    elif backend == "docker" and container:
        # Fallback: docker cp to staging, then rsync from host
        logger.info(f"Pulling {host}:{remote_repo}/{target_path} → {repo_root / target_path}")
        staging = f"/tmp/_npu_pull_{repo_name}"
        run_checked([*_ssh_base(host, remote),
                    f"mkdir -p {staging} && docker cp {container}:{remote_repo}/{target_path} {staging}/"])
        run_checked(["rsync",
                     "-az",
                     *_rsync_ssh_opt(remote),
                     f"{host}:{staging}/{os.path.basename(target_path)}",
                     str(repo_root / target_path)])
    else:
        logger.info(f"Pulling {host}:{remote_repo}/{target_path} → {repo_root / target_path}")
        run_checked(["rsync",
                     "-az",
                     *_rsync_ssh_opt(remote),
                     f"{host}:{remote_repo}/{target_path}",
                     str(repo_root / target_path)])
    logger.info("Pull complete.")


def cmd_sync_diff(host: str, remote: dict, repo_root: Path, wt_name: str | None = None):
    repo_name = find_main_repo_root().name if wt_name else repo_root.name
    # For docker with host_workdir, diff against host path (rsync doesn't go through docker)
    host_repo = resolve_host_repo(remote, repo_name, wt_name)
    remote_repo = host_repo or resolve_remote_repo(remote, repo_name, wt_name)
    logger.info(f"Comparing {repo_root.name} ↔ {host}:{remote_repo}...")
    cmd = [
        "rsync", "-az", *_rsync_ssh_opt(remote), "--dry-run", "--delete", "--itemize-changes",
        *RSYNC_EXCLUDES,
        "--exclude=*.log", "--exclude=*.o",
        f"{repo_root}/", f"{host}:{remote_repo}/"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        msg = f"Error: rsync diff failed (exit {result.returncode})"
        if result.stderr:
            msg += "\n" + result.stderr.rstrip()
        raise NpuError(msg, code=result.returncode or 1)

    added = []     # + local has, remote doesn't
    modified = []  # ~ local modified, remote outdated
    deleted = []   # - remote has, local doesn't

    for line in result.stdout.splitlines():
        if "*deleting" in line:
            deleted.append(line.split("*deleting ")[-1].strip())
        elif line.startswith(".."):
            continue  # directory metadata, skip
        elif len(line) > 12 and line[1] in "fdLDS":
            # Itemize format: YXcstpoguax path (Y=type: f=file, d=dir, L=link, etc.)
            flags = line[:11]
            path = line[12:].strip()
            if not path:
                continue
            if "f++++++++" in flags:
                added.append(path)  # new file
            elif flags[0] in (">", "c") and ("s" in flags or "c" in flags[2:3]):
                modified.append(path)  # content changed

    any_diff = added or modified or deleted
    if added:
        logger.info(f"\n  + New ({len(added)}):")
        for f in added[:20]:
            logger.info(f"    + {f}")
        if len(added) > 20:
            logger.info(f"    ... and {len(added) - 20} more")
    if modified:
        logger.info(f"\n  ~ Modified ({len(modified)}):")
        for f in modified[:20]:
            logger.info(f"    ~ {f}")
        if len(modified) > 20:
            logger.info(f"    ... and {len(modified) - 20} more")
    if deleted:
        logger.info(f"\n  - Remote only ({len(deleted)}):")
        for f in deleted[:20]:
            logger.info(f"    - {f}")
        if len(deleted) > 20:
            logger.info(f"    ... and {len(deleted) - 20} more")
    if not any_diff:
        logger.info("  In sync.")


def cmd_sync_clean(host: str, remote: dict, repo_root: Path, target_path: str, force: bool, wt_name: str | None = None):
    validate_path(target_path)
    repo_name = find_main_repo_root().name if wt_name else repo_root.name
    remote_repo = resolve_remote_repo(remote, repo_name, wt_name)
    target = f"{remote_repo}/{target_path}"
    logger.info(f"Will delete: {host}:{target}")
    if not force:
        confirm = input("Confirm? [y/N] ").strip().lower()
        if confirm != "y":
            logger.info("Cancelled.")
            return
    ssh_exec(host, f"rm -rf '{target}'", remote)
    logger.info("Deleted.")


# ---------------------------------------------------------------------------
# Exec commands
# ---------------------------------------------------------------------------

def cmd_exec(host: str, remote: dict, repo_root: Path, command: str, wt_name: str | None = None) -> int:
    repo_name = find_main_repo_root().name if wt_name else repo_root.name
    remote_repo = resolve_remote_repo(remote, repo_name, wt_name)
    preamble = build_cann_preamble(remote)

    if command == "info":
        info_cmd = f"""{preamble}
echo "=== NPU Platform Info ==="
npu-smi info 2>/dev/null | head -15
echo "--- CANN ---"
echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
echo "--- Workspace ---"
df -h {remote.get('workdir', '/mnt/workspace')} 2>/dev/null
echo "--- Python ---"
which python3 2>/dev/null; python3 --version 2>&1
"""
        result = ssh_exec(host, info_cmd, remote)
    else:
        full_cmd = f"""{preamble}
cd "{remote_repo}" || exit 1
{command}
"""
        result = ssh_exec(host, full_cmd, remote)
    return result.returncode


def cmd_list(config: dict):
    default = config.get("default", "")
    remotes = config.get("remotes", {})
    logger.info(f"{'NAME':<10} {'BACKEND':<10} {'SOC':<15} {'HOST/NAME':<20} {'DEFAULT'}")
    for name, r in remotes.items():
        backend = r.get("backend", "docker")
        soc = r.get("soc", "?")
        identifier = r.get("host", "") or r.get("name", "")
        is_default = "  *" if name == default else ""
        logger.info(f"{name:<10} {backend:<10} {soc:<15} {identifier:<20} {is_default}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(
        prog="npu",
        description="Unified CLI for remote NPU development",
    )
    parser.add_argument(
        "-w", "--worktree",
        help="Use a git worktree as sync source (path to worktree root)",
    )
    sub = parser.add_subparsers(dest="command")

    # sync
    sync_p = sub.add_parser("sync", help="Sync code with remote")
    sync_sub = sync_p.add_subparsers(dest="action")

    push_p = sync_sub.add_parser("push", help="Local → Remote")
    push_p.add_argument("target", nargs="?", help="Remote name (default from config)")
    push_p.add_argument("path", nargs="?", help="Specific path to push")

    pull_p = sync_sub.add_parser("pull", help="Remote → Local")
    pull_p.add_argument("target", help="Remote name")
    pull_p.add_argument("path", help="Remote path to pull")

    diff_p = sync_sub.add_parser("diff", help="Show stale files")
    diff_p.add_argument("target", nargs="?", help="Remote name")

    clean_p = sync_sub.add_parser("clean", help="Delete remote file")
    clean_p.add_argument("target", help="Remote name")
    clean_p.add_argument("path", help="Path to delete")
    clean_p.add_argument("--force", action="store_true", help="Skip confirmation")

    # exec
    exec_p = sub.add_parser("exec", help="Execute command on remote")
    exec_p.add_argument("target", help="Remote name")
    exec_p.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run")

    # list
    sub.add_parser("list", help="List configured remotes")

    args = parser.parse_args()

    # Single CLI entry point: helper functions raise NpuError for fatal errors;
    # we log the message to the original channel (logger.error) and exit with the
    # carried code (default 1), preserving prior CLI behavior. This is the only
    # place a fatal error path is allowed to call sys.exit (CodeCheck G.ERR.11).
    try:
        rc = _dispatch(args, parser)
    except NpuError as e:
        logger.error(str(e))
        sys.exit(e.code)
    if rc:
        sys.exit(rc)


def _dispatch(args, parser) -> int:
    """Run the requested command. Returns a process exit code (0 = success).

    Fatal errors raise NpuError (caught by `main`). Returning a non-zero code
    propagates a remote command's exit status without treating it as an error.
    """
    if args.command == "list":
        config = load_config()
        cmd_list(config)
        return 0

    if not args.command:
        parser.print_help()
        return 0

    # Resolve worktree:
    #   - `-w` explicitly overrides repo_root for sync source.
    #   - Otherwise, auto-detect: if find_repo_root() lands in a git worktree
    #     (.git is a file, main repo root differs), treat cwd's worktree as
    #     the sync source so the remote side lands under {repo}.worktrees/{name}/
    #     and multiple worktrees of the same repo don't overwrite each other.
    wt_name: str | None = None
    if args.worktree:
        wt = Path(args.worktree).resolve()
        if not wt.is_dir():
            raise NpuError(f"Error: worktree path does not exist: {wt}")
        repo_root = wt
        wt_name = wt.name  # e.g. "atan2" from .claude/worktrees/atan2
        config = load_config(find_main_repo_root())
        logger.info(f"🌿 Worktree mode: {wt_name} (local: {wt})")
    else:
        config = load_config()
        repo_root = find_repo_root()
        main_root = find_main_repo_root()
        git_marker = repo_root / ".git"
        if git_marker.is_file() and repo_root.resolve() != main_root.resolve():
            wt_name = repo_root.name
            logger.info(f"🌿 Auto-detected worktree: {wt_name} (local: {repo_root})")

    if args.command == "sync":
        if args.action == "push":
            # Disambiguate: if target looks like a path and no path given, it's actually a path
            target = args.target
            path = args.path
            target_looks_like_path = (
                bool(target)
                and "/" in target
                and not path
                and target not in config.get("remotes", {})
            )
            if target_looks_like_path:
                path = target
                target = None
            name, remote = get_remote(config, target)
            host = resolve_ssh_host(remote)
            cmd_sync_push(host, remote, repo_root, path, wt_name)

        elif args.action == "pull":
            name, remote = get_remote(config, args.target)
            host = resolve_ssh_host(remote)
            cmd_sync_pull(host, remote, repo_root, args.path, wt_name)

        elif args.action == "diff":
            name, remote = get_remote(config, args.target)
            host = resolve_ssh_host(remote)
            cmd_sync_diff(host, remote, repo_root, wt_name)

        elif args.action == "clean":
            name, remote = get_remote(config, args.target)
            host = resolve_ssh_host(remote)
            cmd_sync_clean(host, remote, repo_root, args.path, args.force, wt_name)

    elif args.command == "exec":
        name, remote = get_remote(config, args.target)
        host = resolve_ssh_host(remote)
        command = " ".join(args.cmd) if args.cmd else "info"
        rc = cmd_exec(host, remote, repo_root, command, wt_name)
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    main()
