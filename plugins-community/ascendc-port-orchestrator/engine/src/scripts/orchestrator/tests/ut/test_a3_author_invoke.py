# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Unit tests for a3_author_invoke (Gap-A skill wrapper, 2026-05-13).

Covers:
- should_trigger gate (wrong mode / scripts already present / fires)
- author_one trigger logic + verdict propagation (mocks subprocess.run)
- _validate_authored_scripts AST + invariant gates
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import a3_author_invoke as aai  # noqa: E402
from source_arch import stage_source_tree  # noqa: E402


# ---------------------------------------------------------------------------
# should_trigger gate
# ---------------------------------------------------------------------------
def test_should_trigger_refuses_wrong_mode(tmp_path):
    """Non-port_a3_to_a5 modes must NEVER trigger the skill."""
    for mode in ["backward", "unsupported", ""]:
        fire, reason = aai.should_trigger(tmp_path, mode)
        assert fire is False, f"mode={mode!r} should refuse"
        assert "port_a3_to_a5" in reason


def test_should_trigger_refuses_when_runner_exists(tmp_path):
    """If run_a3_reference.py is already there, skill must NOT overwrite."""
    (tmp_path / "run_a3_reference.py").write_text("# already authored")
    fire, reason = aai.should_trigger(tmp_path, "port_a3_to_a5")
    assert fire is False
    assert "already present" in reason


def test_should_trigger_fires_when_mode_correct_and_runner_missing(tmp_path):
    """Both conditions met → trigger fires."""
    fire, reason = aai.should_trigger(tmp_path, "port_a3_to_a5")
    assert fire is True
    assert "port_a3_to_a5" in reason and "absent" in reason


# ---------------------------------------------------------------------------
# author_one: trigger gating
# ---------------------------------------------------------------------------
def test_author_one_short_circuits_when_runner_present(tmp_path, monkeypatch):
    """Idempotence: existing runner_a3_reference → ALREADY_AUTHORED, no subprocess."""
    (tmp_path / "run_a3_reference.py").write_text("# already")
    fired = []
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: fired.append(a) or pytest.fail("subprocess fired"),
    )
    result = aai.author_one(
        workspace=tmp_path, op_dir=tmp_path,
        opgen_mode="port_a3_to_a5",
    )
    assert result["success"] is True
    assert result["verdict"] == "ALREADY_AUTHORED"
    assert fired == []


def test_author_one_refuses_wrong_mode(tmp_path, monkeypatch):
    """Any non-port_a3_to_a5 mode → WRONG_MODE without spawning claude."""
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: pytest.fail("subprocess fired on wrong mode"),
    )
    result = aai.author_one(
        workspace=tmp_path, op_dir=tmp_path,
        opgen_mode="backward",
    )
    assert result["success"] is False
    assert result["verdict"] == "WRONG_MODE"


def test_author_one_refuses_missing_cpp(tmp_path, monkeypatch):
    """No examples/test_aclnn_<op>.cpp in op_dir → MISSING_CPP without claude call."""
    op_dir = tmp_path / "my_op"
    op_dir.mkdir()
    workspace = tmp_path / "workspace" / "my_op"
    workspace.mkdir(parents=True)
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: pytest.fail("subprocess fired despite missing cpp"),
    )
    result = aai.author_one(
        workspace=workspace, op_dir=op_dir,
        opgen_mode="port_a3_to_a5",
    )
    assert result["success"] is False
    assert result["verdict"] == "MISSING_CPP"
    assert any("does not exist" in e for e in result["errors"])


def test_staged_snapshot_prompt_uses_manifest_op_name(tmp_path):
    op_dir = tmp_path / "heaviside"
    (op_dir / "op_kernel" / "arch22").mkdir(parents=True)
    (op_dir / "op_kernel" / "arch22" / "heaviside.h").write_text(
        "class Heaviside { void Process() {} };\n"
    )
    (op_dir / "examples").mkdir()
    (op_dir / "examples" / "test_aclnn_heaviside.cpp").write_text("// fixture\n")
    workspace = tmp_path / "workspace" / "heaviside"
    stage = stage_source_tree(op_dir, workspace)

    prompt = getattr(aai, '_build_prompt')(stage.root, workspace)

    assert "op `heaviside`" in prompt
    assert "test_aclnn_heaviside.cpp" in prompt
    assert "heaviside_runner.cpp" in prompt
    assert "test_aclnn_.source_arch22.cpp" not in prompt
    assert "one source-supported representative `rank`" in prompt
    assert "MAX_CASE_TENSOR_BYTES = 100 * 1024 * 1024" in prompt
    assert "MAX_DATASET_TENSOR_BYTES = 1024 * 1024 * 1024" in prompt
    assert "ascendcl + nnopbase + opapi" in prompt
    assert "do not link ascend_hal" in prompt
    assert "Do not define generic" in prompt and "CHECK_RET" in prompt


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------
def _write_valid_runner(workspace: Path):
    (workspace / "run_a3_reference.py").write_text(
        '"""runner stub."""\n'
        'from __future__ import annotations\n'
        'import torch, torch_npu, time, json\n'
        'from pathlib import Path\n'
        '_HERE = Path(__file__).resolve().parent\n'
        'def main():\n'
        '    torch.npu.set_device(0)\n'
        '    cases = torch.load(_HERE / "edge_inputs.pt", weights_only=False)\n'
        '    out = []\n'
        '    perf = {}\n'
        '    for c in cases:\n'
        '        out.append(c)\n'
        '        perf[str(c["case_id"])] = 1.0\n'
        '    torch.save(out, _HERE / "edge_dataset.pt")\n'
        '    (_HERE / "a3_baseline_perf.json").write_text(json.dumps(perf))\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
    )


def _write_valid_input_gen(workspace: Path, schema_limit: str | None = None):
    schema_entry = (
        f'"max_case_tensor_bytes": {schema_limit}' if schema_limit else ""
    )
    (workspace / "input_gen.py").write_text(
        '"""input gen stub."""\n'
        'import torch, json\n'
        'from pathlib import Path\n'
        'from case_gen import generate_cases\n'
        'MAX_CASE_TENSOR_BYTES = 100 * 1024 * 1024\n'
        'MAX_DATASET_TENSOR_BYTES = 1024 * 1024 * 1024\n'
        f'SCHEMA = {{{schema_entry}}}\n'
        '_HERE = Path(__file__).resolve().parent\n'
        'def _tensor_payload_bytes(value):\n'
        '    return 0\n'
        'def main():\n'
        '    configured_case_limit = int(SCHEMA.get(\n'
        '        "max_case_tensor_bytes", MAX_CASE_TENSOR_BYTES))\n'
        '    if not 0 < configured_case_limit <= MAX_CASE_TENSOR_BYTES:\n'
        '        raise RuntimeError("case payload too large")\n'
        '    SCHEMA["max_case_tensor_bytes"] = configured_case_limit\n'
        '    cases = generate_cases(SCHEMA)\n'
        '    dataset_tensor_bytes = sum(_tensor_payload_bytes(c) for c in cases)\n'
        '    if dataset_tensor_bytes > MAX_DATASET_TENSOR_BYTES:\n'
        '        raise RuntimeError("dataset payload too large")\n'
        '    torch.save(cases, _HERE / "edge_inputs.pt")\n'
        '    (_HERE / "manifest.json").write_text(json.dumps({"n": 8}))\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
    )


def test_validate_authored_scripts_passes_with_good_scripts(tmp_path):
    _write_valid_runner(tmp_path)
    _write_valid_input_gen(tmp_path)
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "AUTHORED"
    assert errors == []


def test_validate_authored_scripts_fails_when_missing(tmp_path):
    # only one of the two scripts present
    _write_valid_runner(tmp_path)
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "PARSE_FAILED"
    assert any("input_gen.py missing" in e for e in errors)


def test_validate_authored_scripts_fails_on_syntax_error(tmp_path):
    _write_valid_runner(tmp_path)
    (tmp_path / "input_gen.py").write_text("def main(:\n    pass\n")  # syntax error
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "PARSE_FAILED"
    assert any("input_gen.py ast.parse failed" in e for e in errors)


def test_validate_authored_scripts_fails_on_missing_invariants(tmp_path):
    # parses but missing torch_npu import in runner
    (tmp_path / "run_a3_reference.py").write_text(
        'import json\n'
        'def main():\n'
        '    pass\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
    )
    _write_valid_input_gen(tmp_path)
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "INVARIANT_FAILED"
    # missing several tokens — torch_npu / torch.npu.set_device etc
    assert any("torch_npu" in e for e in errors)


def test_validate_input_gen_rejects_budget_tokens_only_in_comments(tmp_path):
    """Comments cannot masquerade as assignments, calls, or payload gates."""
    _write_valid_runner(tmp_path)
    (tmp_path / "input_gen.py").write_text(
        'import torch, json\n'
        '# MAX_CASE_TENSOR_BYTES = 100 * 1024 * 1024\n'
        '# MAX_DATASET_TENSOR_BYTES = 1024 * 1024 * 1024\n'
        'def main():\n'
        '    # cases = generate_cases(SCHEMA)\n'
        '    cases = []\n'
        '    torch.save(cases, "edge_inputs.pt")\n'
        '    open("manifest.json", "w").write(json.dumps({}))\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
    )
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "INVARIANT_FAILED"
    assert any("must assign MAX_CASE_TENSOR_BYTES" in error for error in errors)
    assert any("must assign MAX_DATASET_TENSOR_BYTES" in error for error in errors)
    assert any("main() must call generate_cases" in error for error in errors)


def test_validate_input_gen_rejects_large_schema_case_override(tmp_path):
    """A static per-op override cannot raise the canonical 100 MiB ceiling."""
    _write_valid_runner(tmp_path)
    _write_valid_input_gen(tmp_path, "4 * 1024 * 1024 * 1024")
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "INVARIANT_FAILED"
    assert any("override exceeds the 100 MiB" in error for error in errors)


# ---------------------------------------------------------------------------
# End-to-end author_one with mocked subprocess that writes scripts
# ---------------------------------------------------------------------------
def test_author_one_success_path(tmp_path, monkeypatch):
    """Skill subprocess returns 0 + writes both valid scripts → AUTHORED."""
    op_dir = tmp_path / "my_op"
    (op_dir / "examples").mkdir(parents=True)
    (op_dir / "examples" / "test_aclnn_my_op.cpp").write_text("// dummy aclnn")
    workspace = tmp_path / "workspace" / "my_op"
    workspace.mkdir(parents=True)

    def fake_run(cmd, **kwargs):
        # Simulate the LLM-emitted scripts
        _write_valid_runner(workspace)
        _write_valid_input_gen(workspace)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout='{"result":"ok"}', stderr="",
        )
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = aai.author_one(
        workspace=workspace, op_dir=op_dir,
        opgen_mode="port_a3_to_a5",
    )
    assert result["success"] is True
    assert result["verdict"] == "AUTHORED"
    assert (workspace / ".a3_authored").exists()
    assert (workspace / "run_a3_reference.py").exists()
    assert (workspace / "input_gen.py").exists()
    # Log written
    assert (workspace / ".a3_author_log.jsonl").exists()


def test_author_one_proxy_error(tmp_path, monkeypatch):
    op_dir = tmp_path / "my_op"
    (op_dir / "examples").mkdir(parents=True)
    (op_dir / "examples" / "test_aclnn_my_op.cpp").write_text("// dummy")
    workspace = tmp_path / "workspace" / "my_op"
    workspace.mkdir(parents=True)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0,
            stdout='{"is_error":true,"result":"API Error: corporate proxy Notification"}',
            stderr="",
        )
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = aai.author_one(
        workspace=workspace, op_dir=op_dir,
        opgen_mode="port_a3_to_a5",
    )
    assert result["success"] is False
    assert result["verdict"] == "LLM_PROXY_ERROR"
    assert (workspace / ".a3_author_FAILED-llm_proxy_error").exists()


def test_author_one_timeout(tmp_path, monkeypatch):
    op_dir = tmp_path / "my_op"
    (op_dir / "examples").mkdir(parents=True)
    (op_dir / "examples" / "test_aclnn_my_op.cpp").write_text("// dummy")
    workspace = tmp_path / "workspace" / "my_op"
    workspace.mkdir(parents=True)

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=300)
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = aai.author_one(
        workspace=workspace, op_dir=op_dir,
        opgen_mode="port_a3_to_a5", timeout_sec=300,
    )
    assert result["success"] is False
    assert result["verdict"] == "LLM_TIMEOUT"


def test_author_one_invariant_failure_persists_marker(tmp_path, monkeypatch):
    """Skill ran but emitted scripts missing required tokens → INVARIANT_FAILED
    + failure marker dropped for audit trail.
    """
    op_dir = tmp_path / "my_op"
    (op_dir / "examples").mkdir(parents=True)
    (op_dir / "examples" / "test_aclnn_my_op.cpp").write_text("// dummy")
    workspace = tmp_path / "workspace" / "my_op"
    workspace.mkdir(parents=True)

    def fake_run(cmd, **kwargs):
        # Parses but missing torch_npu
        (workspace / "run_a3_reference.py").write_text(
            'import json\n'
            'def main(): pass\n'
            'if __name__ == "__main__": main()\n'
        )
        _write_valid_input_gen(workspace)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = aai.author_one(
        workspace=workspace, op_dir=op_dir,
        opgen_mode="port_a3_to_a5",
    )
    assert result["success"] is False
    assert result["verdict"] == "INVARIANT_FAILED"
    assert (workspace / ".a3_author_FAILED-invariant_failed").exists()


# ---------------------------------------------------------------------------
# Path B (cpp-binary) validation tests (2026-05-13, Task #49)
# ---------------------------------------------------------------------------
def _write_path_b_runner(workspace: Path):
    """Path B runner: subprocess-based, no torch_npu wrapper required."""
    (workspace / "run_a3_reference.py").write_text(
        '"""Path-B runner stub."""\n'
        'import subprocess, torch, json\n'
        'from pathlib import Path\n'
        '_HERE = Path(__file__).resolve().parent\n'
        'def main():\n'
        '    cases = torch.load(_HERE / "edge_inputs.pt", weights_only=False)\n'
        '    out = []\n'
        '    perf = {}\n'
        '    for c in cases:\n'
        '        subprocess.run(["./build/foo_runner"], check=True)\n'
        '        out.append(c)\n'
        '        perf[str(c["case_id"])] = 1.0\n'
        '    torch.save(out, _HERE / "edge_dataset.pt")\n'
        '    (_HERE / "a3_baseline_perf.json").write_text(json.dumps(perf))\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
    )


def _write_path_b_build_artifacts(workspace: Path, libraries: str = "ascendcl nnopbase opapi"):
    (workspace / "CMakeLists.txt").write_text(
        "add_executable(foo_runner foo_runner.cpp)\n"
        f"target_link_libraries(foo_runner PRIVATE {libraries})\n"
    )
    (workspace / "build_runner.sh").write_text(
        "#!/bin/bash\ncmake -S . -B build && cmake --build build --clean-first\n"
    )
    (workspace / "foo_runner.cpp").write_text("int main() { return 0; }\n")


def test_validate_path_b_runner_passes_with_cmake_present(tmp_path):
    """Path B: CMakeLists.txt + build_runner.sh present → validator selects
    Path B token set (subprocess instead of torch_npu).
    """
    _write_path_b_runner(tmp_path)
    _write_valid_input_gen(tmp_path)
    _write_path_b_build_artifacts(tmp_path)
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "AUTHORED"
    assert errors == []


def test_validate_path_b_runner_fails_without_subprocess(tmp_path):
    """Path B token set requires 'subprocess' import in runner."""
    # Path A-style runner (no subprocess), but with cmake present → mismatch
    _write_valid_runner(tmp_path)  # has torch_npu, no subprocess
    _write_valid_input_gen(tmp_path)
    _write_path_b_build_artifacts(tmp_path)
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "INVARIANT_FAILED"
    # Specifically: Path-B was selected (cmake present), but runner has no subprocess
    assert any("Path-B" in e and "subprocess" in e for e in errors)


def test_validate_path_b_rejects_driver_hal_link(tmp_path):
    """Path B must use runtime libraries, not the driver/devlib-only hal library."""
    _write_path_b_runner(tmp_path)
    _write_valid_input_gen(tmp_path)
    _write_path_b_build_artifacts(tmp_path, "ascendcl ascend_hal nnopbase opapi")
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "INVARIANT_FAILED"
    assert any("must not link ascend_hal" in error for error in errors)


def test_validate_path_b_requires_all_runtime_libraries(tmp_path):
    """A direct aclnn host runner needs all three public runtime libraries."""
    _write_path_b_runner(tmp_path)
    _write_valid_input_gen(tmp_path)
    _write_path_b_build_artifacts(tmp_path, "ascendcl opapi")
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "INVARIANT_FAILED"
    assert any("missing required runtime library: nnopbase" in error for error in errors)


@pytest.mark.parametrize(
    "libraries",
    [
        "ascendcl nnopbase opapi",
        "-lascendcl -lnnopbase -lopapi",
        "libascendcl.so libnnopbase.so libopapi.so",
        (
            "/opt/cann/lib64/libascendcl.so.1 "
            "/opt/cann/lib64/libnnopbase.so.2.3 "
            "/opt/cann/lib64/libopapi.so.4"
        ),
    ],
)
def test_validate_path_b_accepts_supported_runtime_library_forms(
    tmp_path, libraries
):
    """Bare, -l, exact .so paths, and versioned SONAMEs are valid."""
    _write_path_b_runner(tmp_path)
    _write_valid_input_gen(tmp_path)
    _write_path_b_build_artifacts(tmp_path, libraries)
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "AUTHORED"
    assert errors == []


def test_validate_path_b_ignores_commented_link_items(tmp_path):
    """A commented library must neither satisfy nor violate the contract."""
    _write_path_b_runner(tmp_path)
    _write_valid_input_gen(tmp_path)
    _write_path_b_build_artifacts(tmp_path, "opapi")
    cmake = tmp_path / "CMakeLists.txt"
    cmake.write_text(
        "# target_link_libraries(foo_runner PRIVATE ascendcl nnopbase)\n"
        "#[[ target_link_libraries(foo_runner PRIVATE ascend_hal) ]]\n"
        + cmake.read_text()
        + "# -lascend_hal /driver/libascend_hal.so.1\n"
    )
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "INVARIANT_FAILED"
    assert any("missing required runtime library: ascendcl" in e for e in errors)
    assert any("missing required runtime library: nnopbase" in e for e in errors)
    assert not any("must not link ascend_hal" in e for e in errors)


@pytest.mark.parametrize(
    "hal_item",
    [
        "ascend_hal",
        "-lascend_hal",
        "libascend_hal.so",
        "/usr/local/Ascend/driver/lib64/libascend_hal.so.1.2",
    ],
)
def test_validate_path_b_rejects_all_driver_hal_link_forms(tmp_path, hal_item):
    """Every supported spelling of the forbidden driver library is rejected."""
    _write_path_b_runner(tmp_path)
    _write_valid_input_gen(tmp_path)
    _write_path_b_build_artifacts(
        tmp_path, f"ascendcl nnopbase opapi {hal_item}"
    )
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "INVARIANT_FAILED"
    assert any("must not link ascend_hal" in error for error in errors)


def test_validate_path_b_does_not_match_library_name_substrings(tmp_path):
    """Lookalike item names cannot satisfy required libs or trip the HAL ban."""
    _write_path_b_runner(tmp_path)
    _write_valid_input_gen(tmp_path)
    _write_path_b_build_artifacts(
        tmp_path,
        (
            "myascendcl nnopbase_extra libopapi_helper.so "
            "ascend_hal_helper -lascend_hal_extra libmyascend_hal.so"
        ),
    )
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "INVARIANT_FAILED"
    for runtime_lib in ("ascendcl", "nnopbase", "opapi"):
        assert any(
            f"missing required runtime library: {runtime_lib}" in error
            for error in errors
        )
    assert not any("must not link ascend_hal" in error for error in errors)


def test_validate_path_b_rejects_exit_zero_build_stub(tmp_path):
    """A generated Path-B build helper cannot be an unconditional no-op."""
    _write_path_b_runner(tmp_path)
    _write_valid_input_gen(tmp_path)
    _write_path_b_build_artifacts(tmp_path)
    (tmp_path / "build_runner.sh").write_text("#!/bin/bash\nexit 0\n")
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "INVARIANT_FAILED"
    assert any("empty exit 0 stub" in error for error in errors)


def test_validate_path_b_requires_executable_clean_first_gate(tmp_path):
    """A comment mentioning --clean-first is not a clean-build gate."""
    _write_path_b_runner(tmp_path)
    _write_valid_input_gen(tmp_path)
    _write_path_b_build_artifacts(tmp_path)
    (tmp_path / "build_runner.sh").write_text(
        "#!/bin/bash\n"
        "cmake -S . -B build && cmake --build build\n"
        "# cmake --build build --clean-first\n"
    )
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "INVARIANT_FAILED"
    assert any("must execute cmake --build ... --clean-first" in e for e in errors)


def test_validate_path_b_rejects_generic_cann_macro(tmp_path):
    """Generated helpers must not redefine generic macros from CANN headers."""
    _write_path_b_runner(tmp_path)
    _write_valid_input_gen(tmp_path)
    _write_path_b_build_artifacts(tmp_path)
    (tmp_path / "foo_runner.cpp").write_text(
        "#define CHECK_RET(condition, value) do { } while (0)\n"
        "int main() { return 0; }\n"
    )
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "INVARIANT_FAILED"
    assert any("defines generic CHECK_RET" in error for error in errors)


def test_validate_path_a_still_works_when_no_cmake(tmp_path):
    """Backward compat: ops without cmake (Path A) still validate as
    Path A — torch_npu + set_device tokens required.
    """
    _write_valid_runner(tmp_path)
    _write_valid_input_gen(tmp_path)
    # No CMakeLists.txt + no build_runner.sh — Path A
    verdict, errors = getattr(aai, '_validate_authored_scripts')(tmp_path)
    assert verdict == "AUTHORED"
    assert errors == []
