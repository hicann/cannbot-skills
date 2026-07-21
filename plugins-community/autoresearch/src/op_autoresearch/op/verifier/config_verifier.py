# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verification pipeline for Triton autotune config candidates."""

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigVerificationRequest:
    target_code: str
    verify_dir: str
    device_id: int
    timeout: int
    current_step: int


@dataclass(frozen=True)
class ConfigVerificationResult:
    passed: Optional[bool]
    logs: str
    final_code: str
    total_configs: int = 0
    valid_configs: int = 0
    should_record: bool = False


@dataclass
class _VerificationState:
    request: ConfigVerificationRequest
    configs: List[str]
    total_configs: int
    valid_configs: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    merged_program_passed: bool = True


def has_autotune_configs(code: str) -> bool:
    return "@triton.autotune" in code or "@autotune" in code


def _extract_configs(code: str) -> List[str]:
    pattern = r"@triton\.autotune\s*\(\s*configs\s*=\s*\[(.*?)\]"
    match = re.search(pattern, code, re.DOTALL)
    if not match:
        return []
    config_text = match.group(1)
    config_pattern = r"triton\.Config\s*\([^)]*\{[^}]+\}[^)]*\)"
    configs = []
    for config_match in re.finditer(
        config_pattern, config_text, re.DOTALL
    ):
        line_start = config_text.rfind("\n", 0, config_match.start()) + 1
        if "#" not in config_text[line_start:config_match.start()]:
            configs.append(config_match.group(0))
    return configs


def _count_configs(code: str) -> int:
    pattern = r"@triton\.autotune\s*\(\s*configs\s*=\s*\[(.*?)\]"
    match = re.search(pattern, code, re.DOTALL)
    return match.group(1).count("triton.Config") if match else 0


def _single_config_code(code: str, config: str) -> str:
    block = f"configs=[\n        {config},\n    ]"
    return re.sub(
        r"configs\s*=\s*\[(.*?)\]",
        block,
        code,
        count=1,
        flags=re.DOTALL,
    )


def _commented_config(config: str) -> str:
    lines = (
        f"        # {line}" if line.strip() else line
        for line in config.split("\n")
    )
    return "\n".join(lines) + ",  # Failed verification"


def _final_code(state: _VerificationState) -> str:
    if not state.configs:
        return state.request.target_code
    lines = []
    for config in state.configs:
        if config in state.valid_configs:
            lines.append(f"        {config},")
        else:
            lines.append(_commented_config(config))
    block = "configs=[\n" + "\n".join(lines) + "\n    ]"
    return re.sub(
        r"configs\s*=\s*\[(.*?)\]",
        block,
        state.request.target_code,
        count=1,
        flags=re.DOTALL,
    )


def _missing_config_result(
    verifier: Any, state: _VerificationState
) -> Optional[ConfigVerificationResult]:
    if state.configs:
        return None
    request = state.request
    if state.total_configs == 0:
        logger.warning("[%s] no autotune configs found", verifier.op_name)
        return ConfigVerificationResult(
            passed=None,
            logs="",
            final_code=request.target_code,
        )
    state.logs.extend([
        "=== Autotune Config Verification ===\n",
        f"Detected {state.total_configs} commented configs.\n",
        "No active config remains to verify.\n",
    ])
    try:
        verifier.gen_verify_project(
            request.target_code, request.verify_dir, request.device_id
        )
    except Exception as error:
        state.logs.append(
            f"Failed to generate verification project: {error}\n"
        )
    return ConfigVerificationResult(
        passed=False,
        logs="".join(state.logs),
        final_code=request.target_code,
        total_configs=state.total_configs,
        should_record=True,
    )


def _initialize_logs(state: _VerificationState) -> None:
    skipped = state.total_configs - len(state.configs)
    state.logs.extend([
        "=== Autotune Config Verification ===\n",
        f"Active configs: {len(state.configs)}, commented: {skipped}\n\n",
    ])


def _timed_out(logs: str) -> bool:
    markers = (
        "timed out",
        "timeout after",
        "timeouterror",
        "计算超时",
    )
    lowered = logs.lower()
    return any(marker in lowered for marker in markers)


async def _verify_one(
    verifier: Any,
    state: _VerificationState,
    index: int,
    config: str,
) -> tuple[bool, bool]:
    request = state.request
    number = index + 1
    verify_dir = os.path.join(
        request.verify_dir, f"config_{number}_verify"
    )
    state.logs.append(f"--- Config {number} ---\n{config}\n")
    os.makedirs(verify_dir, exist_ok=True)
    try:
        verifier.gen_verify_project(
            _single_config_code(request.target_code, config),
            verify_dir,
            request.device_id,
        )
        passed, logs = await verifier.run_verify(
            verify_dir,
            timeout=request.timeout,
            device_id=request.device_id,
        )
        if passed:
            state.valid_configs.append(config)
            state.logs.append("Verification passed\n\n")
            return True, False
        state.logs.append(
            f"Verification failed\nError log:\n{logs}\n\n"
        )
        return False, _timed_out(logs)
    except Exception as error:
        state.logs.append(f"Verification raised: {error}\n\n")
        logger.error(
            "[%s] config %s verification raised: %s",
            verifier.op_name,
            number,
            error,
        )
        return False, False
    finally:
        shutil.rmtree(verify_dir, ignore_errors=True)


async def _verify_candidates(
    verifier: Any, state: _VerificationState
) -> None:
    consecutive_timeouts = 0
    for index, config in enumerate(state.configs):
        passed, timed_out = await _verify_one(
            verifier, state, index, config
        )
        consecutive_timeouts = (
            consecutive_timeouts + 1 if timed_out else 0
        )
        if passed:
            consecutive_timeouts = 0
        if consecutive_timeouts >= 2:
            remaining = len(state.configs) - index - 1
            message = (
                "Fail-fast after two consecutive timeouts; "
                f"skipped {remaining} configs"
            )
            logger.warning("[%s] %s", verifier.op_name, message)
            state.logs.append(message + "\n\n")
            break
    state.logs.append(
        f"Passed configs: {len(state.valid_configs)}/"
        f"{len(state.configs)}\n"
    )


async def _verify_merged_program(
    verifier: Any, state: _VerificationState
) -> None:
    all_passed = (
        len(state.configs) > 1
        and len(state.valid_configs) == len(state.configs)
    )
    if not all_passed:
        return
    request = state.request
    state.logs.append("\n=== Merged Config Regression ===\n")
    verify_dir = os.path.join(request.verify_dir, "full_code_verify")
    os.makedirs(verify_dir, exist_ok=True)
    try:
        verifier.gen_verify_project(
            request.target_code, verify_dir, request.device_id
        )
        passed, logs = await verifier.run_verify(
            verify_dir,
            timeout=request.timeout,
            device_id=request.device_id,
        )
        state.merged_program_passed = passed
        if passed:
            state.logs.append("Merged program verification passed\n")
        else:
            state.logs.extend([
                f"Merged program verification failed:\n{logs}\n\n",
                "Each config passed alone, but the merged autotune "
                "program failed. Check restore_value for every output.\n",
            ])
    except Exception as error:
        state.merged_program_passed = False
        state.logs.append(
            f"Merged program verification raised: {error}\n"
        )
    finally:
        shutil.rmtree(verify_dir, ignore_errors=True)


def _finish(
    verifier: Any, state: _VerificationState
) -> ConfigVerificationResult:
    code = _final_code(state)
    passed = bool(state.valid_configs) and state.merged_program_passed
    if passed:
        state.logs.append(
            f"Retained {len(state.valid_configs)} verified configs.\n"
        )
    elif state.valid_configs:
        state.logs.append(
            "Individual configs passed, but merged verification failed.\n"
        )
    else:
        state.logs.append("No config passed verification.\n")
    try:
        verifier.gen_verify_project(
            code,
            state.request.verify_dir,
            state.request.device_id,
        )
    except Exception as error:
        state.logs.append(
            f"Failed to generate final verification project: {error}\n"
        )
        passed = False
    return ConfigVerificationResult(
        passed=passed,
        logs="".join(state.logs),
        final_code=code,
        total_configs=len(state.configs),
        valid_configs=len(state.valid_configs),
        should_record=True,
    )


async def verify_autotune_configs(
    verifier: Any, request: ConfigVerificationRequest
) -> ConfigVerificationResult:
    """Run active configs independently, then validate their merged code."""
    state = _VerificationState(
        request=request,
        configs=_extract_configs(request.target_code),
        total_configs=_count_configs(request.target_code),
    )
    missing = _missing_config_result(verifier, state)
    if missing is not None:
        return missing
    _initialize_logs(state)
    await _verify_candidates(verifier, state)
    await _verify_merged_program(verifier, state)
    return _finish(verifier, state)
