# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Conservative source-list checks for the two supported CMake targets."""

import re


def active_text(text):
    """Remove line/bracket comments while preserving quoted arguments and lines."""
    pattern = r'"(?:[^"\\]|\\.)*"|#\[(=*)\[.*?\]\1\]|#[^\n]*'
    return re.sub(
        pattern,
        lambda m: m.group(0)
        if m.group(0).startswith('"')
        else "\n" * m.group(0).count("\n"),
        text,
        flags=re.S,
    )


def compat_guard_lines(text):
    """Line indices inside the positive NOT HCCL_CANN_COMPAT_850 branch.

    Only recognize a standalone, single-line guard. Unknown conditions are not
    evidence of compatibility; else/elseif leave the recognized branch.
    """
    stack, guarded = [], set()
    for index, line in enumerate(active_text(text).splitlines()):
        if re.match(r"^\s*if\s*\(", line, re.I):
            stack.append(
                bool(
                    re.fullmatch(
                        r"\s*if\s*\(\s*NOT\s+HCCL_CANN_COMPAT_850\s*\)\s*", line, re.I
                    )
                )
            )
        elif re.match(r"^\s*(?:else|elseif)\s*\(", line, re.I):
            if stack:
                stack[-1] = False
        elif re.match(r"^\s*endif\s*\(", line, re.I):
            if stack:
                stack.pop()
        if any(stack):
            guarded.add(index)
    return guarded


def has_source(text, entry, target, compat_guard=False):
    """Require an exact argument in a relevant set/list or library command.

    This checks registration, not whether enclosing CMake conditions evaluate true.
    Build both host and device targets to verify conditional wiring.
    """
    active = active_text(text)
    guarded = compat_guard_lines(text) if compat_guard else set()
    for match in re.finditer(
        r"\b(set|list|add_library|target_sources)\s*\(([^)]*)\)", active, re.I | re.S
    ):
        if compat_guard and active.count("\n", 0, match.start()) not in guarded:
            continue
        command, body = match.groups()
        args = [word.strip('"') for word in re.findall(r'"[^"\n]*"|[^\s]+', body)]
        if target == "host":
            relevant = (command.lower() == "set" and args[:1] == ["src_list"]) or (
                command.lower() == "list" and args[:2] == ["APPEND", "src_list"]
            )
        else:
            relevant = command.lower() in ("add_library", "target_sources") and args[
                :1
            ] == ["scatter_aicpu_kernel"]
        if relevant and entry in args:
            return True
    return False
