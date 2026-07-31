# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Optional sudo-gated-docker-host support (2026-07-08, owner-directed).

`A5_DOCKER_SUDO` / `{TARGET}_DOCKER_SUDO` is an OPTIONAL, opt-in, per-target-resource
flag: when set, the engine wraps its post-SSH remote docker commands under `sudo su -c`
(needed on hosts where docker requires root AND the SSH user's only NOPASSWD path is
`sudo su`, e.g. some shared multi-user A5 boxes). Default OFF → byte-identical behavior
(backward compatible).
"""
import sys
from pathlib import Path

_ORCH = Path(__file__).resolve().parents[2]  # src/scripts/orchestrator
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from phase_o5_helpers import _docker_sudo_enabled, _maybe_sudo_wrap_remote  # noqa: E402


def test_default_off_is_byte_identical():
    """Unset flag → command returned UNCHANGED (backward compat)."""
    cmd = "docker exec c bash -c 'echo hi'"
    assert _maybe_sudo_wrap_remote(cmd, {}) == cmd
    assert _maybe_sudo_wrap_remote(cmd, {"A5_DOCKER_SUDO": ""}) == cmd
    assert _maybe_sudo_wrap_remote(cmd, {"A5_DOCKER_SUDO": "0"}) == cmd
    assert _maybe_sudo_wrap_remote(cmd, {"A5_DOCKER_SUDO": "false"}) == cmd
    assert _docker_sudo_enabled({}) is False


def test_enabled_wraps_in_sudo_su_c():
    """Truthy flag → wrapped as `sudo su -c <single-quoted whole cmd>`."""
    cmd = "docker exec c ls /x"
    for truthy in ("1", "true", "yes", "on", "TRUE", "Yes"):
        out = _maybe_sudo_wrap_remote(cmd, {"A5_DOCKER_SUDO": truthy})
        assert out.startswith("sudo su -c "), out
        assert _docker_sudo_enabled({"A5_DOCKER_SUDO": truthy}) is True


def test_nested_single_quotes_are_escaped():
    """The inner docker command's single quotes must survive `su -c` single-quoting."""
    cmd = "docker exec c bash -c 'cd /x && tar -xf a.tar'"
    out = _maybe_sudo_wrap_remote(cmd, {"A5_DOCKER_SUDO": "1"})
    # POSIX single-quote escaping: embedded ' becomes '\'' — no bare unescaped inner quote
    assert out.startswith("sudo su -c '")
    assert out.endswith("'")
    assert "'\\''" in out  # the inner single quotes were escaped, not left bare


def test_per_target_override_beats_legacy():
    """`{TARGET}_DOCKER_SUDO` is honored (target-specific), plus legacy A5_DOCKER_SUDO."""
    assert _docker_sudo_enabled({"A5_DOCKER_SUDO": "1"}, target="a5") is True
    assert _docker_sudo_enabled({"A5_DOCKER_SUDO": "1"}, target="A5") is True
    # a non-a5 target with only the target-specific flag set
    assert _docker_sudo_enabled({"A3_DOCKER_SUDO": "1"}, target="a3") is True
    assert _docker_sudo_enabled({"A3_DOCKER_SUDO": "1"}, target="a5") is False


# --- Coverage guard: every op-gen-critical-path docker site stays sudo-wrapped ------------
# The sudo wrap is all-or-none per remote-exec path: an unwrapped docker site on a sudo-gated
# host fails there (silently un-covered). This guards against silent removal of the wrap.
def test_all_critical_docker_sites_wrapped():
    scripts_root = _ORCH.parent  # src/scripts
    checks = [
        # (file, marker that must be present)
        (_ORCH / "phase_o5_runner.py", "_maybe_sudo_wrap_remote"),
        (_ORCH / "phase_o5_verify.py", "_maybe_sudo_wrap_remote"),
        (_ORCH / "a3_ref_provision.py", "sudo su -c"),
        (scripts_root / "deploy_to_npu.sh", "_wrap_sudo"),
    ]
    for path, marker in checks:
        assert path.is_file(), f"missing {path}"
        assert marker in path.read_text(), f"sudo-wrap marker {marker!r} removed from {path.name}"
    # deploy_to_npu.sh: every `docker exec`/`docker cp` remote-exec must go through _wrap_sudo.
    deploy = (scripts_root / "deploy_to_npu.sh").read_text()
    for ln in deploy.splitlines():
        if ('"${SSH_CMD[@]}"' in ln) and ("docker exec" in ln or "docker cp" in ln):
            assert "_wrap_sudo" in ln, f"unwrapped docker site in deploy_to_npu.sh: {ln.strip()[:80]}"
