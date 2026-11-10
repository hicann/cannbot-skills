#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check profiles against read-only repo checkouts, executing their UT globs without CANN."""

import argparse
import fnmatch
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from migration_contract import cmake_calls, executable, profile, read, words

LOGGER = logging.getLogger(__name__)

# Read-only witnesses for repository rules; never source identities for migration.
SAMPLES = {
    "ops-math": ("math/abs", "OPS_CATEGORY_LIST"),
    "ops-nn": ("matmul/mat_mul_v3", "OP_CATEGORY_LIST"),
    "ops-cv": ("objdetect/roi_align", "OPS_CATEGORY_LIST"),
    "ops-transformer": ("attention/flash_attention_score", "ASCEND_ALL_MODULE_NAME"),
}

PROBE_PRELUDE = """cmake_minimum_required(VERSION 3.16)
function(add_library)
endfunction()
function(add_optiling_ut_modules)
endfunction()
function(target_sources)
    foreach(item IN LISTS ARGN)
        if(item MATCHES "[.]cpp$")
            file(APPEND "${COLLECTED}" "${item}\\n")
        endif()
    endforeach()
endfunction()
set(OP_TILING_MODULE_NAME tiling)
set(OP_INFERSHAPE_MODULE_NAME infershape)
set(OP_API_MODULE_NAME op_api)
set(OP_GRAPH_MODULE_NAME op_graph)
set(OP_FRAMEWORK_MODULE_NAME framework)
set(OP_ONNX_MODULE_NAME onnx)
set(ASCEND_OP_NAME probe)
"""


def definition(text, name, kind="function"):
    match = re.search(
        rf"(?ims)^\s*{kind}\s*\(\s*{name}\b.*?^\s*end{kind}\s*\([^)]*\)", text
    )
    if not match:
        raise ValueError(f"missing {kind}: {name}")
    return match[0]


def create_probe_files(host, arch):
    names = [
        "test_probe_tiling.cpp",
        f"{arch}/test_probe_tiling.cpp",
        f"{arch}/test_probe_tiling_{arch}.cpp",
        f"{'arch22' if arch == 'arch35' else 'arch35'}/test_wrong_tiling.cpp",
    ]
    for file in names:
        path = host / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return names


def probe_ut_globs(repo, name, config, soc, arch):
    function = definition(read(repo / "cmake/ut.cmake"), "add_modules_ut_sources")
    if name == "ops-nn":
        common = read(repo / "cmake/func.cmake")
        function += "\n" + definition(common, "map_compute_unit")
        function += "\n" + definition(common, "get_target_dir")
    with tempfile.TemporaryDirectory(prefix="migration-cmake-") as temporary:
        temp = Path(temporary)
        host = temp / "category/probe/tests/ut/op_host"
        names = create_probe_files(host, arch)
        collected = temp / "collected.txt"
        collected.touch()
        prelude = PROBE_PRELUDE + (
            f'set(COLLECTED "{collected}")\nset(CMAKE_CURRENT_SOURCE_DIR "{host}")\n'
            f"set(ASCEND_COMPUTE_UNIT {soc})\nset(ARCH_DIRECTORY {arch})\n"
        )
        key = "HOSTNAME" if name == "ops-nn" else "UT_NAME"
        call = f'add_modules_ut_sources({key} tiling MODE PRIVATE DIR "{host}"'
        call += f" TILING_DIR {arch})\n" if name == "ops-math" else ")\n"
        if config["explicit_tests"]:
            call += f'add_modules_ut_sources({key} tiling MODE PRIVATE DIR "{host}/{arch}")\n'
        script = temp / "probe.cmake"
        script.write_text(prelude + function + "\n" + call)
        result = subprocess.run(
            [executable("cmake"), "-P", str(script)],
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
        if result.returncode:
            raise ValueError(f"{name} CMake probe failed: {result.stderr}")
        actual = sorted(
            {
                str(Path(line).relative_to(host))
                for line in collected.read_text().splitlines()
            }
        )
        expected = [names[0], names[1]]
        if fnmatch.fnmatch(Path(names[2]).name, config["tiling_arch_glob"]):
            expected.append(names[2])
        if actual != sorted(expected):
            raise ValueError(
                f"{name} UT glob drift: actual={actual}, expected={sorted(expected)}"
            )
        return actual


def category_list(text, variable):
    for body in cmake_calls(text, "set"):
        args = words(body)
        if args and args[0] == variable:
            end = args.index("CACHE") if "CACHE" in args else len(args)
            return args[1:end]
    return None


def check_categories(name, repo, config, variables):
    categories = category_list(variables, SAMPLES[name][1])
    if set(categories or []) - {"ge"} != set(config["classes"]):
        raise ValueError(f"{name} class allowlist drift: {categories}")
    for category in config["classes"]:
        if not (repo / category).is_dir():
            raise ValueError(f"{name} missing class directory: {category}")


def check_sample(name, repo, config):
    sample = SAMPLES[name][0]
    cmake = read(repo / "cmake/func.cmake") + read(repo / "cmake/obj_func.cmake")
    macro = definition(cmake, config["cmake_macro"], "macro")
    auto_ut = bool(cmake_calls(macro, "add_all_ut_sources"))
    if auto_ut == config["explicit_tests"]:
        raise ValueError(f"{name} automatic/explicit UT contract drift")
    sample_dir = repo / sample
    if not cmake_calls(read(sample_dir / config["cmake_file"]), config["cmake_macro"]):
        raise ValueError(
            f"{name} representative no longer uses the documented CMake entry: {sample}"
        )
    if "CANN Open Software License Agreement Version 2.0" not in read(
        sample_dir / config["cmake_file"]
    ):
        raise ValueError(f"{name} representative license changed")
    examples = list(sample_dir.glob("examples/**/test_aclnn_*.cpp"))
    if not examples:
        raise ValueError(f"{name} representative examples missing: {sample}")


def check_packaging(name, repo, config, variables):
    # Inspect the branch used by build.sh --pkg --ops=... --vendor_name=...;
    # transformer also contains an unused pack_custom() with a different name.
    packaging = read(repo / config["package_cmake"])
    package = (
        config["package_glob"]
        .replace("{vendor}", "${VENDOR_NAME}")
        .replace("*", "${ARCH}")
        .removesuffix(".run")
    )
    if package not in packaging or "${VENDOR_NAME}" + config["vendor_suffix"] not in (
        packaging + variables
    ):
        raise ValueError(f"{name} package/vendor convention drift")


def audit(name, repo):
    config = profile(name)
    variables = read(repo / "cmake/variables.cmake") + read(repo / "CMakeLists.txt")
    check_categories(name, repo, config, variables)
    check_sample(name, repo, config)
    check_packaging(name, repo, config, variables)
    sha = subprocess.check_output(
        [executable("git"), "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        shell=False,
    ).strip()
    return {
        "repo": name,
        "commit": sha,
        "class_count": len(config["classes"]),
        "representative": SAMPLES[name][0],
        "explicit_tests": config["explicit_tests"],
        "ut_glob_probes": {
            soc: probe_ut_globs(repo, name, config, soc, arch)
            for soc, arch in [("ascend910b", "arch22"), ("ascend950", "arch35")]
        },
        "status": "PASS",
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repositories", nargs="+", metavar="REPO=PATH")
    args = parser.parse_args()
    results = []
    for item in args.repositories:
        try:
            name, path = item.split("=", 1)
            results.append(audit(name, Path(path).resolve()))
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            parser.exit(1, f"[FAIL] {item}: {exc}\n")
    LOGGER.info("%s", json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
