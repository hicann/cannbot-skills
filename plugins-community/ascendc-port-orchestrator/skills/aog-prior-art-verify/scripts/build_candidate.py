# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Phase 3 of /aog-prior-art-verify — build a staged arch35 candidate.

Given `workspace/<op>/.prior_art_candidate/` (Phase 2 output), this module
copies the candidate tree into a unique container path, overlays its
digest-bound files onto an isolated copy of ops-nn, invokes
`build.sh --pkg --ops=<op> --soc=ascend950`, and pulls only that build's fresh
`.so` + `binary.json` artifacts back to `candidate_dir/build/`.

Build failure is a valid candidate outcome
(BUILD_FAILED → Phase 5 → CANDIDATE_BUILD_GAP).
Timeout is also handled as a verdict, not an exception.

Usage (CLI):
  python3 build_candidate.py --op <name> --workspace <dir> \\
    --a5-host 198.51.100.35 --a5-container npu_dev3 \\
    --ops-nn-path /home/.../cann/ops-nn

Usage (programmatic):
  from build_candidate import build, BuildReport
  rep = build(op, workspace, ops_nn_path, a5_host=..., a5_user=..., a5_container=...)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from stage_candidate import compute_candidate_digest


# Same DI alias as phase_o25_a3_ref — caller swaps in a mock for tests.
RunRemote = Callable[[str], tuple[int, str, str]]
PushDir = Callable[[Path, str, str, str, str, str], Optional[str]]
PullFiles = Callable[[str, str, str, str, str, Path], Optional[str]]


@dataclass
class BuildReport:
    op: str
    verdict: str = "UNKNOWN"
    # SUCCESS / BUILD_FAILED / SCP_PUSH_FAILED / SCP_PULL_FAILED / TIMEOUT /
    # NO_CANDIDATE / CANDIDATE_INVALID
    candidate_dir: Optional[Path] = None
    candidate_digest: Optional[str] = None
    manifest_sha256: Optional[str] = None
    build_id: Optional[str] = None
    container_candidate_path: Optional[str] = None
    container_repo_path: Optional[str] = None
    build_log: str = ""
    so_path: Optional[Path] = None       # pulled-back .so file
    so_sha256: Optional[str] = None
    binary_json_path: Optional[Path] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Default SSH/SCP helpers (mirror phase_o25_a3_ref shape)
# ---------------------------------------------------------------------------
def _default_run_remote(a5_user: str, a5_host: str, a5_container: str,
                        timeout_s: int = 1500) -> RunRemote:
    def run(cmd: str) -> tuple[int, str, str]:
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            f"{a5_user}@{a5_host}",
            f"docker exec {shlex.quote(a5_container)} bash -lc {shlex.quote(cmd)}",
        ]
        try:
            r = subprocess.run(ssh_cmd, capture_output=True, text=True,
                               timeout=timeout_s)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return 124, "", f"TimeoutExpired: build exceeded {timeout_s}s"
        except Exception as e:
            return 1, "", f"subprocess error: {e!r}"
    return run


def _default_push_dir(local: Path, user: str, host: str, container: str,
                      host_root: str, container_candidate: str) -> Optional[str]:
    """Copy the candidate through the host into the named container.

    A host-side SCP alone is insufficient: ``run_remote`` executes with
    ``docker exec`` and therefore cannot assume that the host ``/tmp`` is a
    bind mount.  The explicit ``docker cp`` is part of the object binding.
    """
    try:
        host_candidate = f"{host_root}/candidate"
        mk = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no",
             f"{user}@{host}",
             " && ".join([
                 f"rm -rf -- {shlex.quote(host_root)}",
                 f"mkdir -p {shlex.quote(host_candidate)}",
             ])],
            capture_output=True, text=True, timeout=30,
        )
        if mk.returncode != 0:
            return f"prepare host staging failed: {mk.stderr[:400]}"
        r = subprocess.run(
            ["scp", "-r", "-o", "StrictHostKeyChecking=no",
             f"{local}/.", f"{user}@{host}:{host_candidate}/"],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            return f"scp push failed: {r.stderr[:400]}"
        copy = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"{user}@{host}",
             " && ".join([
                 f"docker exec {shlex.quote(container)} rm -rf -- "
                 f"{shlex.quote(str(Path(container_candidate).parent))}",
                 f"docker exec {shlex.quote(container)} mkdir -p "
                 f"{shlex.quote(container_candidate)}",
                 f"docker cp {shlex.quote(host_candidate)}/. "
                 f"{shlex.quote(container)}:{shlex.quote(container_candidate)}/",
             ])],
            capture_output=True, text=True, timeout=120,
        )
        if copy.returncode != 0:
            return f"docker cp candidate failed: {copy.stderr[:400]}"
        return None
    except subprocess.TimeoutExpired:
        return "scp push timeout"
    except Exception as e:
        return f"scp push exception: {e!r}"


def _default_pull_files(user: str, host: str, container: str,
                        container_artifacts: str, host_artifacts: str,
                        local_dir: Path) -> Optional[str]:
    """Copy only this build's isolated artifact directory out of the container."""
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        copy = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"{user}@{host}",
             " && ".join([
                 f"rm -rf -- {shlex.quote(host_artifacts)}",
                 f"mkdir -p {shlex.quote(host_artifacts)}",
                 f"docker cp {shlex.quote(container)}:"
                 f"{shlex.quote(container_artifacts)}/. "
                 f"{shlex.quote(host_artifacts)}/",
             ])],
            capture_output=True, text=True, timeout=60,
        )
        if copy.returncode != 0:
            return f"docker cp artifacts failed: {copy.stderr[:400]}"
        r = subprocess.run(
            ["scp", "-r", "-o", "StrictHostKeyChecking=no",
             f"{user}@{host}:{host_artifacts}/.", str(local_dir) + "/"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return f"scp pull failed: {r.stderr[:400]}"
        return None
    except subprocess.TimeoutExpired:
        return "scp pull timeout"
    except Exception as e:
        return f"scp pull exception: {e!r}"


# ---------------------------------------------------------------------------
# Build orchestration
# ---------------------------------------------------------------------------
_REMOTE_BUILD_ROOT = "/tmp/aog_prior_art_build"
_HOST_TRANSFER_ROOT = "/tmp/aog_prior_art_transfer"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class CandidateBinding:
    candidate_digest: str
    manifest_sha256: str
    repo_name: str
    files: tuple[dict, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_relative(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} missing")
    path = Path(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe {label}: {value!r}")
    return path


def _load_candidate_binding(candidate_dir: Path, op: str,
                            expected_repo_name: Optional[str] = None,
                            ) -> tuple[Optional[CandidateBinding], list[str]]:
    """Revalidate the staged object immediately before build or verify."""
    errors: list[str] = []
    manifest_path = candidate_dir / "manifest.json"
    if not manifest_path.is_file():
        return None, ["candidate manifest.json missing"]
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        return None, [f"candidate manifest parse error: {exc}"]
    if manifest.get("schema_version") != 2:
        errors.append("candidate manifest schema_version must be 2")
    if manifest.get("op") != op:
        errors.append(
            f"candidate manifest op mismatch: {manifest.get('op')!r} != {op!r}"
        )
    if manifest.get("errors"):
        errors.append("candidate manifest records staging errors")
    repo_name = manifest.get("repo_name")
    if not isinstance(repo_name, str) or not repo_name:
        errors.append("candidate manifest repo_name missing")
        repo_name = ""
    if expected_repo_name and repo_name != expected_repo_name:
        errors.append(
            f"candidate repo mismatch: manifest={repo_name!r}, "
            f"build checkout={expected_repo_name!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        errors.append("candidate manifest files must be a non-empty list")
        files = []
    if manifest.get("file_count") != len(files):
        errors.append("candidate manifest file_count mismatch")

    normalized_files: list[dict] = []
    seen_rel: set[Path] = set()
    seen_build: set[Path] = set()
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            errors.append(f"candidate file entry {index} is not an object")
            continue
        try:
            rel_path = _manifest_relative(entry.get("rel_path"), "rel_path")
            build_rel_path = _manifest_relative(
                entry.get("build_rel_path"), "build_rel_path"
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        recorded_sha = entry.get("sha")
        if not isinstance(recorded_sha, str) or len(recorded_sha) != 64:
            errors.append(f"invalid staged SHA-256 for {rel_path}")
            continue
        if rel_path in seen_rel or build_rel_path in seen_build:
            errors.append(f"duplicate candidate overlay mapping for {rel_path}")
            continue
        seen_rel.add(rel_path)
        seen_build.add(build_rel_path)
        staged_path = candidate_dir / rel_path
        if not staged_path.is_file():
            errors.append(f"manifested candidate file missing: {rel_path}")
            continue
        actual_sha = _sha256_file(staged_path)
        if actual_sha != recorded_sha:
            errors.append(
                f"manifested candidate file changed: {rel_path} "
                f"(expected {recorded_sha}, got {actual_sha})"
            )
            continue
        normalized_files.append({
            "rel_path": str(rel_path),
            "build_rel_path": str(build_rel_path),
            "sha": recorded_sha,
            "source_type": entry.get("source_type", ""),
        })

    recorded_digest = manifest.get("candidate_digest")
    if not isinstance(recorded_digest, str) or len(recorded_digest) != 64:
        errors.append("candidate_digest missing or invalid")
    else:
        actual_digest = compute_candidate_digest(op, repo_name, normalized_files)
        if actual_digest != recorded_digest:
            errors.append(
                f"candidate_digest mismatch: expected {recorded_digest}, "
                f"got {actual_digest}"
            )

    if errors:
        return None, errors
    return CandidateBinding(
        candidate_digest=recorded_digest,
        manifest_sha256=_sha256_file(manifest_path),
        repo_name=repo_name,
        files=tuple(normalized_files),
    ), []


def _construct_remote_build_layout(op: str, candidate_digest: str,
                                   build_id: str) -> dict:
    """Return unique host/container paths for exactly one candidate build."""
    if not _SAFE_COMPONENT.fullmatch(op):
        raise ValueError(f"unsafe operator name: {op!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", candidate_digest):
        raise ValueError("candidate_digest must be a lowercase SHA-256")
    if not re.fullmatch(r"[0-9a-f]{12,64}", build_id):
        raise ValueError("build_id must be 12-64 lowercase hex characters")
    suffix = f"{op}-{candidate_digest[:12]}-{build_id[:12]}"
    container_root = f"{_REMOTE_BUILD_ROOT}/{suffix}"
    host_root = f"{_HOST_TRANSFER_ROOT}/{suffix}"
    return {
        "host_root": host_root,
        "host_artifacts": f"{host_root}/artifacts",
        "container_root": container_root,
        "candidate": f"{container_root}/candidate",
        "repo": f"{container_root}/repo",
        "artifacts": f"{container_root}/artifacts",
        "build_started": f"{container_root}/build.started",
        "log": f"{container_root}/build.log",
    }


def _construct_build_command(op: str, ops_nn_path: str,
                             binding: CandidateBinding, layout: dict,
                             soc: str = "ascend950") -> str:
    """Build a SHA-checked overlay in an isolated container-side checkout."""
    candidate_root = layout["candidate"]
    repo_root = layout["repo"]
    artifacts = layout["artifacts"]
    parts = [
        "set -euo pipefail",
        f"test -f {shlex.quote(candidate_root + '/manifest.json')}",
        f"rm -rf -- {shlex.quote(repo_root)} {shlex.quote(artifacts)}",
        f"cp -a -- {shlex.quote(ops_nn_path)} {shlex.quote(repo_root)}",
        f"rm -rf -- {shlex.quote(repo_root + '/build_out')}",
        f"mkdir -p {shlex.quote(artifacts)}",
    ]
    for entry in binding.files:
        source = f"{candidate_root}/{entry['rel_path']}"
        destination = f"{repo_root}/{entry['build_rel_path']}"
        expected_sha = entry["sha"]
        parts.extend([
            f"test -f {shlex.quote(source)}",
            f"test \"$(sha256sum {shlex.quote(source)} | awk '{{print $1}}')\" = "
            f"{shlex.quote(expected_sha)}",
            f"mkdir -p {shlex.quote(str(Path(destination).parent))}",
            f"cp -- {shlex.quote(source)} {shlex.quote(destination)}",
            f"test \"$(sha256sum {shlex.quote(destination)} | awk '{{print $1}}')\" = "
            f"{shlex.quote(expected_sha)}",
        ])
    parts.extend([
        f"touch {shlex.quote(layout['build_started'])}",
        f"cd {shlex.quote(repo_root)}",
        f"bash build.sh --pkg --ops={shlex.quote(op)} "
        f"--soc={shlex.quote(soc)} 2>&1 | tee {shlex.quote(layout['log'])}",
        f"find {shlex.quote(repo_root + '/build_out')} -type f "
        f"-newer {shlex.quote(layout['build_started'])} "
        f"\\( -name {shlex.quote('*' + op + '*.so')} -o "
        f"-name {shlex.quote('*' + op + '*binary.json')} \\) "
        f"-exec cp -- {{}} {shlex.quote(artifacts + '/')} \\;",
        f"test -n \"$(find {shlex.quote(artifacts)} -type f "
        f"-name {shlex.quote('*' + op + '*.so')} -print -quit)\"",
        f"printf '%s\\n' {shlex.quote(binding.candidate_digest)} > "
        f"{shlex.quote(artifacts + '/candidate_digest.txt')}",
    ])
    return " && ".join(parts)


def build(op: str, workspace: Path, ops_nn_path: str,
          *,
          a5_host: str,
          a5_user: str = "root",
          a5_container: str = "npu_dev3",
          soc: str = "ascend950",
          run_remote: Optional[RunRemote] = None,
          push_dir: Optional[PushDir] = None,
          pull_files: Optional[PullFiles] = None,
          timeout_s: int = 1500,
          ) -> BuildReport:
    """Phase 3 entry. Returns BuildReport with verdict + pulled-back artifacts."""
    import time
    rep = BuildReport(op=op)
    candidate_dir = workspace / ".prior_art_candidate"
    rep.candidate_dir = candidate_dir
    if not candidate_dir.is_dir():
        rep.verdict = "NO_CANDIDATE"
        rep.errors.append(
            f"{candidate_dir} missing — run stage_candidate first"
        )
        return rep

    binding, binding_errors = _load_candidate_binding(
        candidate_dir, op, expected_repo_name=Path(ops_nn_path).name
    )
    if binding is None:
        rep.verdict = "CANDIDATE_INVALID"
        rep.errors.extend(binding_errors)
        return rep
    rep.candidate_digest = binding.candidate_digest
    rep.manifest_sha256 = binding.manifest_sha256
    rep.build_id = uuid.uuid4().hex
    try:
        layout = _construct_remote_build_layout(
            op, binding.candidate_digest, rep.build_id
        )
    except ValueError as exc:
        rep.verdict = "CANDIDATE_INVALID"
        rep.errors.append(str(exc))
        return rep
    rep.container_candidate_path = layout["candidate"]
    rep.container_repo_path = layout["repo"]

    # A prior local pull must never satisfy this build's artifact checks.
    build_out_local = candidate_dir / "build"
    if build_out_local.exists():
        shutil.rmtree(build_out_local)

    _run = run_remote or _default_run_remote(a5_user, a5_host, a5_container,
                                              timeout_s=timeout_s)
    _push = push_dir or _default_push_dir
    _pull = pull_files or _default_pull_files

    # Step 1: copy the digest-bound candidate into this build's container path.
    err = _push(candidate_dir, a5_user, a5_host, a5_container,
                layout["host_root"], layout["candidate"])
    if err:
        rep.verdict = "SCP_PUSH_FAILED"
        rep.errors.append(err)
        return rep

    # Step 2: invoke build inside container
    cmd = _construct_build_command(
        op, ops_nn_path, binding, layout, soc=soc
    )
    start = time.time()
    rc, stdout, stderr = _run(cmd)
    rep.duration_s = time.time() - start
    rep.build_log = (stdout + ("\n--STDERR--\n" + stderr if stderr else ""))[-8000:]

    if rc == 124:
        rep.verdict = "TIMEOUT"
        rep.errors.append(stderr or "build exceeded timeout")
        return rep
    if rc != 0:
        rep.verdict = "BUILD_FAILED"
        rep.errors.append(f"build returned rc={rc}")
        return rep

    # Step 3: pull only this unique build's isolated artifact directory.
    pull_error = _pull(
        a5_user, a5_host, a5_container, layout["artifacts"],
        layout["host_artifacts"], build_out_local
    )
    if pull_error:
        rep.verdict = "SCP_PULL_FAILED"
        rep.errors.append(pull_error)
        return rep

    digest_marker = build_out_local / "candidate_digest.txt"
    if (not digest_marker.is_file()
            or digest_marker.read_text().strip() != binding.candidate_digest):
        rep.verdict = "BUILD_FAILED"
        rep.errors.append(
            "pulled artifacts are not bound to this candidate digest"
        )
        return rep

    # Locate pulled artifacts
    so_files = sorted(build_out_local.glob(f"*{op}*.so"))
    json_files = sorted(build_out_local.glob(f"*{op}*binary.json"))
    if so_files:
        rep.so_path = so_files[0]
        rep.so_sha256 = _sha256_file(rep.so_path)
    else:
        rep.warnings.append(".so file not found in pulled artifacts")
    if json_files:
        rep.binary_json_path = json_files[0]
    else:
        rep.warnings.append("binary.json not found in pulled artifacts")

    if rep.so_path is None:
        # build succeeded but no .so → treat as failed
        rep.verdict = "BUILD_FAILED"
        rep.errors.append("build rc=0 but no .so produced (artifact mismatch)")
        return rep

    rep.verdict = "SUCCESS"
    return rep


def write_build_report(rep: BuildReport, workspace: Path) -> Path:
    """Persist BuildReport to `workspace/.prior_art_build.json` for downstream
    phases (verify / classify) to read."""
    out = workspace / ".prior_art_build.json"
    out.write_text(json.dumps({
        "schema_version": 2,
        "op": rep.op,
        "verdict": rep.verdict,
        "duration_s": rep.duration_s,
        "candidate_digest": rep.candidate_digest,
        "manifest_sha256": rep.manifest_sha256,
        "build_id": rep.build_id,
        "container_candidate_path": rep.container_candidate_path,
        "container_repo_path": rep.container_repo_path,
        "so_path": str(rep.so_path) if rep.so_path else None,
        "so_sha256": rep.so_sha256,
        "binary_json_path": str(rep.binary_json_path) if rep.binary_json_path else None,
        "build_log_tail": rep.build_log[-2000:],
        "errors": rep.errors,
        "warnings": rep.warnings,
    }, indent=2, sort_keys=True) + "\n")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--op", required=True)
    p.add_argument("--workspace", required=True, type=Path)
    p.add_argument("--ops-nn-path", required=True,
                   help="path to ops-nn checkout on A5 host (e.g. /home/.../cann/ops-nn)")
    p.add_argument("--a5-host", required=True)
    p.add_argument("--a5-user", default="root")
    p.add_argument("--a5-container", default="npu_dev3")
    p.add_argument("--soc", default="ascend950")
    p.add_argument("--timeout-s", type=int, default=1500)
    args = p.parse_args(argv)
    rep = build(args.op, args.workspace, args.ops_nn_path,
                a5_host=args.a5_host, a5_user=args.a5_user,
                a5_container=args.a5_container, soc=args.soc,
                timeout_s=args.timeout_s)
    out = write_build_report(rep, args.workspace)
    print(f"verdict={rep.verdict} duration={rep.duration_s:.1f}s → {out}")
    if rep.errors:
        for e in rep.errors:
            print(f"  ERROR: {e}", file=sys.stderr)
    return 0 if rep.verdict == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
