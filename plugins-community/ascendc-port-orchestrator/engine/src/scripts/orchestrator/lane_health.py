# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""lane_health — device-health-aware NPU lane routing for shared A5 hosts.

Motivation (2026-06-18, selective_scan + GDN recurrent runs): the A5 host
`.171` is shared infrastructure. An external tenant can saturate a physical
NPU (100% AICore util, ~98% HBM, watchdog 507034) while other chips stay
healthy. `phase_o5` (and the worker deploy/verify) historically hardcoded
`ASCEND_RT_VISIBLE_DEVICES=<lane>` with no health check, so a run pinned to a
wedged device would HANG at verify — and the hang risks being mis-read as a
kernel/precision defect rather than the shared-NPU env condition it is.

This module provides a pure, testable parser for `npu-smi info` output (the
Ascend950PR 2-line-per-device layout) plus a picker that returns a healthy
physical device, preferring the requested lane when it is itself healthy.

Design rules:
- FAIL-OPEN: if npu-smi is unavailable / unparseable, return the requested
  lane unchanged (never block a run on a best-effort health probe).
- NEVER select a device we'd have to reset or that holds another tenant's
  work — we only *route around* busy devices, never reclaim them.
- Decouple from deploy-lane: the chosen device only sets
  ASCEND_RT_VISIBLE_DEVICES (which physical NPU runs the .so). The .so
  location (BENCHMARK_ROOT_lane{N}) stays keyed to the deploy lane — the two
  are independent (see phase_o5_runner notes).

The Ascend950PR `npu-smi info` device block (verified on .171, CANN 25.7.rc1):

    | NPU ID | Name             | Health        | Power(W) Temp(C) Hugepages-Usage |
    |        |                  | Bus-Id        | NPU Util(%) Memory-Usage HBM-Usage|
    +========+==================+===============+===================================+
    | 0      | Ascend950PR      | Alarm         | 205.7    71      0     / 0       |
    |        |                  | 0000:71:00.0  | 100      0  / 0   90633 / 114688 |
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class DeviceHealth:
    device_id: int
    health: str            # "OK" / "Alarm" / other
    util_pct: float        # AICore util %
    hbm_used_mb: int
    hbm_total_mb: int

    @property
    def hbm_frac(self) -> float:
        if self.hbm_total_mb <= 0:
            return 0.0
        return self.hbm_used_mb / self.hbm_total_mb

    def is_healthy(self, *, hbm_frac_max: float = 0.5, util_max: float = 80.0) -> bool:
        """Healthy = reported OK AND HBM headroom AND not AICore-saturated.

        util_max defaults to 80 (not 50): a transient util spike is fine; the
        wedge signature that matters is sustained 100% + high HBM. HBM is the
        harder gate (a launch OOMs / queues behind a memory-hog tenant).
        """
        return (
            self.health.upper() == "OK"
            and self.hbm_frac < hbm_frac_max
            and self.util_pct < util_max
        )


# Device summary row: | <id> | <Name> | <Health> | <power> <temp> <hugepages> |
_DEV_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*(\S+)\s*\|\s*([A-Za-z]+)\s*\|")
# Chip detail row (immediately follows the summary). Layout has TWO empty
# leading columns (NPU-ID + Name) then Bus-Id then the data column:
#   |        |                  | 0000:71:00.0  | 100  0 / 0  90633 / 114688 |
# Bus-Id looks like 0000:71:00.0 ; capture util int + the LAST "<used> / <total>" (HBM).
_CHIP_ROW_RE = re.compile(
    r"^\|\s*\|\s*\|\s*[0-9A-Fa-f:.]+\s*\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s*\|"
)


def parse_npu_health(npu_smi_output: str) -> dict[int, DeviceHealth]:
    """Parse `npu-smi info` (Ascend950PR layout) → {device_id: DeviceHealth}.

    Tolerant: a device whose detail row doesn't parse is skipped (not crashed).
    The process-list table at the bottom is ignored (its rows don't match the
    device-summary shape: a process row is `| 0 | <pid> | <name> | <mem> |`,
    but the chip-detail row that must immediately follow a device-summary row
    is absent, so the pending summary is discarded).
    """
    health: dict[int, DeviceHealth] = {}
    pending: Optional[tuple[int, str]] = None  # (id, health) awaiting its chip row
    for line in npu_smi_output.splitlines():
        mdev = _DEV_ROW_RE.match(line)
        if mdev:
            # A new device-summary row. Any unmatched pending summary is dropped
            # (e.g. a process-table row that looked summary-shaped).
            pending = (int(mdev.group(1)), mdev.group(3))
            continue
        if pending is not None:
            mchip = _CHIP_ROW_RE.match(line)
            if mchip:
                dev_id, h = pending
                health[dev_id] = DeviceHealth(
                    device_id=dev_id,
                    health=h,
                    util_pct=float(mchip.group(1)),
                    hbm_used_mb=int(mchip.group(2)),
                    hbm_total_mb=int(mchip.group(3)),
                )
            # whether or not the chip row matched, this summary is consumed
            pending = None
    return health


def pick_healthy_device(
    requested: int,
    health: dict[int, DeviceHealth],
    *,
    hbm_frac_max: float = 0.5,
    util_max: float = 80.0,
) -> tuple[int, str]:
    """Choose a physical device, preferring `requested` when it is healthy.

    Returns (device_id, reason). FAIL-OPEN: if health is empty or the requested
    device is absent from the map, return `requested` unchanged (best-effort).
    """
    if not health:
        return requested, "no health data (npu-smi empty/unparsed) — keep requested lane"
    req = health.get(requested)
    if req is not None and req.is_healthy(hbm_frac_max=hbm_frac_max, util_max=util_max):
        return requested, f"requested lane {requested} healthy (hbm={req.hbm_frac:.0%}, util={req.util_pct:.0f}%)"
    if req is None:
        # requested device not visible in npu-smi — don't second-guess, keep it.
        return requested, f"requested lane {requested} not in npu-smi — keep (fail-open)"
    # Requested is unhealthy: pick the healthiest alternative (lowest HBM frac
    # among OK + util-ok devices). Deterministic tie-break by device id.
    candidates = [
        d for d in health.values()
        if d.is_healthy(hbm_frac_max=hbm_frac_max, util_max=util_max)
    ]
    if not candidates:
        return requested, (
            f"requested lane {requested} unhealthy "
            f"(hbm={req.hbm_frac:.0%}, util={req.util_pct:.0f}%, health={req.health}) "
            f"but NO healthy device available — keep requested (fail-open)"
        )
    best = min(candidates, key=lambda d: (d.hbm_frac, d.device_id))
    return best.device_id, (
        f"requested lane {requested} unhealthy "
        f"(hbm={req.hbm_frac:.0%}, util={req.util_pct:.0f}%, health={req.health}); "
        f"rerouted to device {best.device_id} (hbm={best.hbm_frac:.0%}, util={best.util_pct:.0f}%)"
    )


def resolve_healthy_device(
    requested: int,
    host: str,
    user: str = "root",
    *,
    run_npu_smi: Optional[Callable[[str, str], tuple[int, str, str]]] = None,
    hbm_frac_max: float = 0.5,
    util_max: float = 80.0,
) -> tuple[int, str]:
    """ssh `npu-smi info` on `host`, parse, pick a healthy device.

    FAIL-OPEN on any error: returns (requested, reason). `run_npu_smi` is an
    injection point for tests; default does a real host-level ssh (NOT
    docker-exec — npu-smi must run host-side on A5).
    """
    runner = run_npu_smi or _ssh_npu_smi
    try:
        rc, out, err = runner(user, host)
    except Exception as e:  # noqa: BLE001
        return requested, f"npu-smi invocation error ({e!r}) — keep requested lane (fail-open)"
    if rc != 0:
        return requested, f"npu-smi rc={rc} ({err.strip()[:80]}) — keep requested lane (fail-open)"
    health = parse_npu_health(out)
    return pick_healthy_device(
        requested, health, hbm_frac_max=hbm_frac_max, util_max=util_max
    )


def _ssh_npu_smi(user: str, host: str) -> tuple[int, str, str]:
    """Host-level `npu-smi info` via ssh (NOT docker-exec — npu-smi resolves
    on the A5 host, and docker-exec swallows its output under nested quoting)."""
    import subprocess
    try:
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null",
             f"{user}@{host}", "npu-smi info"],
            capture_output=True, text=True, timeout=15,
        )
        return r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired:
        return 1, "", "npu-smi timed out after 15s"
    except Exception as e:  # noqa: BLE001
        return 1, "", f"npu-smi invocation error: {e!r}"
