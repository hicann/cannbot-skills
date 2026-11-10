#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Read shared repo profiles and check the documented migration layout (no build)."""

import argparse
import json
import logging
import re
import shlex
import shutil
import sys
from pathlib import Path

PROFILES = Path(__file__).resolve().parent.parent / "references/repo-profiles.json"
LOGGER = logging.getLogger(__name__)


def profiles():
    return json.loads(PROFILES.read_text(encoding="utf-8"))["repositories"]


def profile(name):
    try:
        return profiles()[name]
    except KeyError as exc:
        raise ValueError(f"unsupported target_repo: {name}") from exc


def validate_decision(data):
    if not isinstance(data, dict) or data.get("decision") not in {"RUN", "SKIP"}:
        raise ValueError("decision must be RUN or SKIP")
    if not {"target_repo", "op_class"} <= data.keys():
        raise ValueError("target_repo and op_class must be recorded, even when null")
    for key in ("reason", "reason_code"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"non-empty {key} required")
    evidence = data.get("evidence")
    if (
        not isinstance(evidence, dict)
        or not evidence.get("category")
        or not evidence.get("routing_rule")
    ):
        raise ValueError("evidence.category and evidence.routing_rule required")
    if data["decision"] == "RUN":
        validate_run_decision(data)
    else:
        validate_skip_decision(data)
    return data


def validate_run_decision(data):
    config = profile(data.get("target_repo"))
    if data.get("op_class") not in config["classes"]:
        raise ValueError("op_class is not a supported class of target_repo")
    if data["reason_code"] not in {
        "explicit_target",
        "category_match",
        "domain_match",
    }:
        raise ValueError("RUN requires explicit_target, category_match or domain_match")
    if (
        not isinstance(data.get("source_package"), str)
        or not Path(data["source_package"]).is_absolute()
    ):
        raise ValueError("RUN requires an absolute source_package path")


def validate_skip_decision(data):
    if data["reason_code"] not in {
        "user_opt_out",
        "unsupported_target",
        "unsupported_category",
        "ambiguous_target",
    }:
        raise ValueError("invalid SKIP reason_code")
    if data.get("op_class") is not None:
        raise ValueError("SKIP op_class must be null")
    if data["reason_code"] == "unsupported_target":
        target = data.get("target_repo")
        if not isinstance(target, str) or not target or target in profiles():
            raise ValueError("unsupported_target must retain the unsupported repo name")


def executable(name):
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"required executable not found: {name}")
    return str(Path(path).resolve())


def read(path):
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def registration_types(text, macro):
    # Literal registrations are required by the migration layout. Ignore names
    # in comments and ordinary string/character literals, without guessing case.
    code = re.sub(
        r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        " ",
        text,
        flags=re.S,
    )
    return re.findall(rf"\b{macro}\s*\(\s*([A-Za-z_]\w*)\s*\)", code)


def example_files(directory, soc, config, route):
    directories = config["example_dirs"].get(soc, config["example_dirs"]["default"])
    files = []
    for subdir in directories:
        pattern = f"{subdir}/test_{route}_*.cpp" if subdir else f"test_{route}_*.cpp"
        matches = (directory / "examples").glob(pattern)
        files.extend(path for path in matches if path.is_file())
    return files


def strip_cmake_comments(text):
    # CMake bracket comments use matching equals counts; quoted # characters
    # remain arguments. An unfinished comment must never expose its body as code.
    token = re.compile(r'"(?:\\.|[^"\\])*"|#\[(=*)\[|#[^\n]*', re.S)
    parts = []
    offset = 0
    while match := token.search(text, offset):
        start = match.start()
        parts.append(text[offset:start])
        offset = match.end()
        if match[0].startswith('"'):
            parts.append(match[0])
            continue
        if match[1] is not None:
            closing = "]" + match[1] + "]"
            end = text.find(closing, offset)
            if end < 0:
                raise ValueError("unterminated CMake bracket comment")
            offset = end + len(closing)
        parts.append(" " + "\n" * text[start:offset].count("\n"))
    parts.append(text[offset:])
    return "".join(parts)


def cmake_calls(text, name):
    # The migration recipe uses literal arguments and set() lists, not a full
    # CMake evaluator. Runtime conditions still need real source collection/UT.
    text = strip_cmake_comments(text)
    return re.findall(rf"\b{re.escape(name)}\s*\(([^()]*)\)", text, re.I)


def words(text):
    return [
        part for word in shlex.split(text, comments=True) for part in word.split(";")
    ]


def cmake_values(text, call, key):
    variables = {}
    for body in cmake_calls(text, "set"):
        args = words(body)
        if args:
            variables[args[0]] = args[1:]
    keys = {
        "OPTYPE",
        "ACLNNTYPE",
        "COMPUTE_UNIT",
        "TILING_DIR",
        "DEPENDENCIES",
        "DISABLE_IN_OPP",
        "OP_API_INDEPENDENT",
        "OP_API_DIR",
        "HOSTCPU",
        "NPU_ARCH",
    }
    args = words(call)
    if key not in args:
        return []
    result = []
    start = args.index(key) + 1
    for arg in args[start:]:
        if arg in keys:
            break
        match = re.fullmatch(r"\$\{(\w+)\}", arg)
        result.extend(variables.get(match[1], [arg]) if match else [arg])
    if any("$" in x for x in result):
        raise ValueError(
            f"cannot statically resolve {key}; use literal lists per the recipe"
        )
    return result


class Check:
    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def require(self, condition, message):
        LOGGER.info("[%s] %s", " OK " if condition else "FAIL", message)
        if not condition:
            self.failures += 1
        return bool(condition)

    def warn(self, message):
        self.warnings += 1
        LOGGER.warning("[WARN] %s", message)


def check_source_layout(check, directory):
    for path in directory.rglob("*"):
        if path.is_file() and (
            path.suffix
            in {
                ".c",
                ".cc",
                ".cpp",
                ".cxx",
                ".h",
                ".hh",
                ".hpp",
                ".hxx",
                ".asc",
                ".py",
                ".cmake",
            }
            or path.name == "CMakeLists.txt"
        ):
            check.require(
                "Copyright" in "\n".join(read(path).splitlines()[:8]),
                f"license header: {path.relative_to(directory)}",
            )
    for name in [
        "build",
        "build.sh",
        "tests/aclnn",
        "tests/geir",
        "tests/tiling",
        "tests/tensorflow",
        "tests/torchair",
        "tests/torch",
    ]:
        check.require(not (directory / name).exists(), f"no workflow leftover: {name}")
    check.require(
        not list(directory.glob("op_graph/*_proto.cc")),
        "registration is in op_graph/*_proto.h",
    )
    check.require(
        bool(list(directory.glob("op_graph/*_proto.h"))),
        "graph prototype header exists",
    )


def check_registration(check, directory, soc, source_optype):
    op = directory.name
    definitions = list(directory.glob("op_host/*_def.cpp"))
    check.require(
        len(definitions) == 1, "exactly one OpDef file for the migrated operator"
    )
    definition = read(definitions[0]) if len(definitions) == 1 else ""
    check.require(
        definitions == [directory / "op_host" / f"{op}_def.cpp"],
        "OpDef filename matches the operator directory used by the build",
    )
    check.require(
        len(re.findall(r"OP_ADD\s*\(", definition)) == 1,
        "one OP_ADD occurrence (including comments)",
    )
    op_types = registration_types(definition, "OP_ADD")
    check.require(len(op_types) == 1, "one literal OP_ADD registration type")
    graph_types = set()
    for path in directory.glob("op_graph/*_proto.h"):
        graph_types.update(registration_types(read(path), "REG_OP"))
    check.require(
        len(op_types) == 1 and op_types[0] in graph_types,
        "REG_OP contains the actual OP_ADD type (case preserved)",
    )
    if source_optype is not None:
        check.require(
            op_types == [source_optype],
            f"registration preserves source type: {source_optype}",
        )
    else:
        check.warn(
            "source type not supplied; compare against the source before acceptance"
        )
    check.require(f'AddConfig("{soc}"' in definition, f'explicit AddConfig("{soc}")')
    return definition


def check_kernel_entries(check, directory, definition, config):
    op = directory.name
    entries = re.findall(r'opFile\.value"\s*,\s*"([\w]+)"', definition)
    check.require(bool(entries), "opFile.value is present")
    for entry in set(entries):
        check.require(
            entry in {op + suffix for suffix in config["kernel_entry_suffixes"]},
            f"opFile.value={entry} resolves to the staged operator directory {op}",
        )
        check.require(
            (directory / "op_kernel" / f"{entry}.cpp").is_file(),
            f"kernel entry is directly under op_kernel: {entry}.cpp",
        )


def check_tiling_lists(check, directory, soc, cmake, call):
    units = cmake_values(cmake, call, "COMPUTE_UNIT")
    tiling = cmake_values(cmake, call, "TILING_DIR")
    valid = len(units) == len(tiling) and bool(units) and len(set(units)) == len(units)
    check.require(
        valid,
        "COMPUTE_UNIT / TILING_DIR are non-empty, aligned, unambiguous lists",
    )
    check.require(soc in units, f"COMPUTE_UNIT contains {soc}")
    if not valid or soc not in units:
        return None, []
    arch = tiling[units.index(soc)]
    check.require(
        (directory / "op_host" / arch).is_dir(), f"tiling directory exists: {arch}"
    )
    return arch, tiling


def has_definition_source(cmake, bucket, op):
    for body in cmake_calls(cmake, "target_sources"):
        args = words(body)
        if bucket in args and f"{op}_def.cpp" in args:
            return True
    return False


def check_transformer_build(check, directory, cmake, call, manual_api):
    if directory.parent.name in {"attention", "mhc", "mamba"}:
        check.require(
            len(cmake_calls(cmake, "add_op_to_compiled_list")) == 1,
            "transformer registers its directory for kernel compilation and UT",
        )
    bucket = "op_host_aclnnExc" if manual_api else "op_host_aclnn"
    check.require(
        has_definition_source(cmake, bucket, directory.name),
        "transformer OpDef is also registered in the open-project target",
    )
    check.require(
        cmake_values(cmake, call, "OP_API_INDEPENDENT") == ["ON"]
        if manual_api
        else True,
        "transformer keeps independent op_api wired into the host macro",
    )
    if manual_api:
        check.require(
            bool(
                re.search(
                    r'OP_API_DIR\s+"?\$\{CMAKE_CURRENT_SOURCE_DIR\}/\.\./op_api', call
                )
            ),
            "transformer OP_API_DIR points to ../op_api",
        )


def check_build_contract(check, directory, soc, config):
    op = directory.name
    cmake = read(directory / config["cmake_file"])
    calls = cmake_calls(cmake, config["cmake_macro"])
    check.require(
        len(calls) == 1, f"one {config['cmake_macro']} call in {config['cmake_file']}"
    )
    arch = None
    source_arch = None
    supported_arches = set(config["soc_arch"].values())
    if len(calls) == 1:
        call = calls[0]
        optypes = cmake_values(cmake, call, "OPTYPE")
        check.require(
            optypes in ([op], []), "OPTYPE matches the directory (or uses its default)"
        )
        aclnn = cmake_values(cmake, call, "ACLNNTYPE")
        manual_api = list(directory.glob("op_api/aclnn_*.cpp"))
        check.require(
            aclnn == (["aclnn_exclude"] if manual_api else ["aclnn"]),
            "ACLNNTYPE matches handwritten/autogenerated ACLNN",
        )
        if config["tiling_lists"]:
            arch, tiling = check_tiling_lists(check, directory, soc, cmake, call)
            source_arch = arch
            supported_arches.update(tiling)
        else:
            check_transformer_build(check, directory, cmake, call, manual_api)
    if config["ut_arch_source"] == "soc":
        arch = config["soc_arch"].get(soc)
        check.require(bool(arch), f"UT architecture mapping exists for {soc}")
    if not config["tiling_lists"]:
        source_arch = arch
    check_tiling_sources(check, directory, source_arch)
    return arch, supported_arches


def check_tiling_sources(check, directory, source_arch):
    tiling_sources = list(directory.glob("op_host/*_tiling*.cpp"))
    if source_arch:
        tiling_sources += list(
            (directory / "op_host" / source_arch).glob("*_tiling*.cpp")
        )
    check.require(
        bool(tiling_sources), "tiling implementation is in a collected source directory"
    )


def has_subdirectory(content, parent, child):
    expected = (parent / child).resolve()
    for body in cmake_calls(content, "add_subdirectory"):
        args = words(body)
        if not args:
            continue
        source = args[0]
        for variable in ["CMAKE_CURRENT_SOURCE_DIR", "CMAKE_CURRENT_LIST_DIR"]:
            source = source.replace("${" + variable + "}", str(parent))
        if "$" not in source and (parent / source).resolve() == expected:
            return True
    return False


def check_ut_cmake(check, directory, config):
    if config["explicit_tests"]:
        edges = [(".", "tests"), ("tests", "ut"), ("tests/ut", "op_host")]
        if list(directory.glob("op_api/aclnn_*.cpp")):
            edges.append(("tests/ut", "op_api"))
        for parent, child in edges:
            content = read(directory / parent / "CMakeLists.txt")
            check.require(
                has_subdirectory(content, directory / parent, child),
                f"CMake traversal: {parent}/CMakeLists.txt -> {child}",
            )
        host_cmake = read(directory / "tests/ut/op_host/CMakeLists.txt")
        check.require(
            bool(cmake_calls(host_cmake, "add_modules_ut_sources")),
            "op_host UT registered explicitly",
        )
        check.warn(
            "CMake condition reachability must also be proven by collected sources and non-zero UT execution"
        )
    else:
        check.require(
            not (directory / "tests/CMakeLists.txt").exists(),
            "automatic UT collection: no tests/CMakeLists.txt",
        )


def check_host_ut_sources(check, directory, config, arch, supported_arches):
    host = directory / "tests/ut/op_host"
    tiling_tests = list(host.glob("test_*_tiling.cpp"))
    shape_tests = list(host.glob("test_*_infershape.cpp"))
    if arch:
        tiling_tests += list((host / arch).glob(config["tiling_arch_glob"]))
        if config["infershape_in_arch"]:
            shape_tests += list((host / arch).glob("test_*_infershape.cpp"))
    check.require(bool(tiling_tests), "at least one tiling UT matches this soc's glob")
    check.require(
        bool(shape_tests), "at least one InferShape UT matches this soc's glob"
    )
    collected = set(tiling_tests + shape_tests)
    for path in host.rglob("test_*.cpp"):
        if "tiling" not in path.stem and "infershape" not in path.stem:
            continue
        relative = path.relative_to(host)
        if (
            len(relative.parts) == 2
            and relative.parts[0] in supported_arches
            and relative.parts[0] != arch
        ):
            continue
        check.require(path in collected, f"UT is collected: {relative}")


def check_api_and_ut_main(check, directory, config):
    if list(directory.glob("op_api/aclnn_*.cpp")):
        check.require(
            bool(list(directory.glob("tests/ut/op_api/test_aclnn_*.cpp"))),
            "handwritten ACLNN has op_api UT",
        )
        if config["explicit_tests"]:
            api_cmake = read(directory / "tests/ut/op_api/CMakeLists.txt")
            check.require(
                bool(cmake_calls(api_cmake, "add_modules_ut_sources")),
                "op_api UT registered explicitly",
            )
    for path in directory.glob("tests/ut/**/*.cpp"):
        check.require(
            not re.search(r"\bint\s+main\s*\(", read(path)),
            f"no standalone main in {path.relative_to(directory)}",
        )


def check_operator(directory, soc, repo, source_optype=None):
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError(f"operator directory does not exist: {directory}")
    if not soc:
        raise ValueError("soc must be explicitly selected from the source contract")
    config = profile(repo)
    check = Check()
    LOGGER.info(
        "== %s: %s/%s, soc=%s ==", repo, directory.parent.name, directory.name, soc
    )
    check.require(directory.parent.name in config["classes"], "supported op_class")
    check_source_layout(check, directory)
    definition = check_registration(check, directory, soc, source_optype)
    check_kernel_entries(check, directory, definition, config)
    arch, supported_arches = check_build_contract(check, directory, soc, config)
    check_ut_cmake(check, directory, config)
    check_host_ut_sources(check, directory, config, arch, supported_arches)
    check_api_and_ut_main(check, directory, config)
    check.require(
        bool(example_files(directory, soc, config, "aclnn")),
        "ACLNN example exists in a directory collected for this soc",
    )
    if not example_files(directory, soc, config, "geir"):
        check.warn(
            "GEIR example absent; report N/A only when the source has no GEIR route"
        )
    LOGGER.info("FAIL=%s WARN=%s", check.failures, check.warnings)
    return 1 if check.failures else 0


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("profile").add_argument("repo")
    commands.add_parser("decision").add_argument("path", type=Path)
    check = commands.add_parser("check")
    check.add_argument("directory", type=Path)
    check.add_argument("--soc", required=True)
    check.add_argument("--repo")
    check.add_argument(
        "--op-type", help="actual OP_ADD / REG_OP type from the source package"
    )
    args = parser.parse_args()
    try:
        if args.command == "profile":
            LOGGER.info(
                "%s", json.dumps(profile(args.repo), ensure_ascii=False, indent=2)
            )
        elif args.command == "decision":
            validate_decision(json.loads(args.path.read_text(encoding="utf-8")))
            LOGGER.info("migration decision: OK")
        else:
            repo = args.repo or args.directory.resolve().parent.parent.name
            return check_operator(args.directory, args.soc, repo, args.op_type)
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(2, f"[ERROR] {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
