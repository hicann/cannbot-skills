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

"""platform_config_accessor — read AUTHORITATIVE per-SoC hardware constants from the
CANN install's platform_config, instead of hand-copying (drift-prone) numbers into KB.

Rationale (owner-directed 2026-06-27): every real op-gen run REQUIRES CANN (build =
bisheng, run = NPU), so `${ASCEND_HOME_PATH}/<arch>/data/platform_config/<soc>.ini`
is always present on disk — it is the canonical, version-correct source for core
counts / sizes / freq / bandwidth ratios. roofline_eval/MFU should read these, not
carry hardcoded peaks (which drifted before — VEC=56 mis-used as fp16 ceiling).
INTERPRETED knowledge (what the constants mean for tiling/optimization) still lives
in KB; raw constants come from here.

Usage:
    python3 platform_config_accessor.py                 # list all SoCs in the install
    python3 platform_config_accessor.py Ascend950PR_957b   # constants + derived peaks
    python3 platform_config_accessor.py Ascend950PR_957b --json
"""
from __future__ import annotations
import configparser
import glob
import json
import os
import sys


def _platform_config_dir() -> str | None:
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not home:
        return None
    # arch subdir varies (x86_64-linux on dev hosts, arm64-linux on some); take whichever exists
    for arch in ("x86_64-linux", "arm64-linux", "aarch64-linux"):
        d = os.path.join(home, arch, "data", "platform_config")
        if os.path.isdir(d):
            return d
    # fallback: any data/platform_config under the install
    hits = glob.glob(os.path.join(home, "*", "data", "platform_config"))
    return hits[0] if hits else None


def list_socs() -> list[str]:
    d = _platform_config_dir()
    if not d:
        return []
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(d, "*.ini")))


def read_soc(soc: str) -> dict:
    """Return the parsed hardware constants + derived peaks for one SoC, or {} if absent."""
    d = _platform_config_dir()
    if not d:
        return {}
    path = os.path.join(d, f"{soc}.ini")
    if not os.path.isfile(path):
        return {}
    cp = configparser.ConfigParser(strict=False)
    cp.read(path)
    info = {s: dict(cp.items(s)) for s in cp.sections()}

    soc_info = info.get("SoCInfo", {})
    spec = info.get("AICoreSpec", {})

    def _i(d_, k):  # tolerant int read
        try:
            return int(d_.get(k, ""))
        except ValueError:
            return None

    cube_cnt = _i(soc_info, "cube_core_cnt") or _i(soc_info, "ai_core_cnt")
    vec_cnt = _i(soc_info, "vector_core_cnt")
    freq = _i(spec, "cube_freq")  # MHz
    m, n, k = _i(spec, "cube_m_size"), _i(spec, "cube_n_size"), _i(spec, "cube_k_size")

    derived = {}
    if cube_cnt and freq and m and n and k:
        # cube peak: cores × freq(Hz) × MACs/cycle(m·n·k) × 2 flop/MAC
        macs_per_cycle = m * n * k
        peak_flops = cube_cnt * (freq * 1e6) * macs_per_cycle * 2
        derived["cube_macs_per_cycle"] = macs_per_cycle
        derived["cube_peak_tflops"] = round(peak_flops / 1e12, 2)
    if soc_info.get("memory_size"):
        try:
            derived["hbm_gib"] = round(int(soc_info["memory_size"]) / (1024**3), 1)
        except ValueError:
            pass
    if soc_info.get("l2_size"):
        try:
            derived["l2_mib"] = round(int(soc_info["l2_size"]) / (1024**2), 1)
        except ValueError:
            pass

    return {
        "soc": soc,
        "source": path,
        "cube_core_cnt": cube_cnt,
        "vector_core_cnt": vec_cnt,
        "cube_freq_mhz": freq,
        "cube_mnk": [m, n, k],
        "ub_size": _i(spec, "ub_size"),
        "l1_size": _i(spec, "l1_size"),
        "l0c_size": _i(spec, "l0_c_size"),
        "derived": derived,
        "raw_sections": info,
    }


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    as_json = "--json" in argv

    d = _platform_config_dir()
    if not d:
        print("ERROR: ASCEND_HOME_PATH unset or no platform_config dir found "
              "(needs a CANN install).", file=sys.stderr)
        return 2

    if not args:
        socs = list_socs()
        if as_json:
            print(json.dumps(socs, indent=2))
        else:
            print(f"platform_config: {d}\n{len(socs)} SoC configs:")
            for s in socs:
                print(f"  {s}")
        return 0

    soc = args[0]
    data = read_soc(soc)
    if not data:
        print(f"ERROR: SoC '{soc}' not found in {d}", file=sys.stderr)
        print("Available: " + ", ".join(list_socs()[:8]) + " ...", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(data, indent=2))
    else:
        de = data["derived"]
        print(f"=== {soc} (authoritative, from {data['source']}) ===")
        print(f"  cube cores : {data['cube_core_cnt']}")
        print(f"  vector cores: {data['vector_core_cnt']}")
        print(f"  cube freq  : {data['cube_freq_mhz']} MHz")
        print(f"  cube m×n×k : {data['cube_mnk']}")
        print(f"  UB / L1 / L0C: {data['ub_size']} / {data['l1_size']} / {data['l0c_size']} bytes")
        print(f"  derived cube peak: {de.get('cube_peak_tflops')} TFLOPS")
        print(f"  HBM / L2  : {de.get('hbm_gib')} GiB / {de.get('l2_mib')} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
