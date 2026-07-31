# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Built-kernel checks for the two supported AscendC workflow modes."""
from pathlib import Path

from plugins import get_plugin
from workspace_lifecycle import _optimize_built_kernel_present


def _touch(workspace: Path, *relative_files: str) -> None:
    for relative in relative_files:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")


def test_migration_populated_cpp_dirs_proceed(tmp_path):
    plugin = get_plugin("port_a3_to_a5")
    assert plugin is not None
    _touch(tmp_path, "op_host/def.cpp", "op_kernel/kernel.h")
    assert _optimize_built_kernel_present(tmp_path, plugin) is True


def test_migration_requires_every_declared_cpp_dir(tmp_path):
    plugin = get_plugin("port_a3_to_a5")
    assert plugin is not None
    _touch(tmp_path, "op_host/def.cpp")
    assert _optimize_built_kernel_present(tmp_path, plugin) is False


def test_migration_rejects_empty_cpp_dirs(tmp_path):
    plugin = get_plugin("port_a3_to_a5")
    assert plugin is not None
    (tmp_path / "op_host").mkdir()
    (tmp_path / "op_kernel").mkdir()
    assert _optimize_built_kernel_present(tmp_path, plugin) is False


def test_backward_populated_kernel_dir_proceeds(tmp_path):
    plugin = get_plugin("backward")
    assert plugin is not None
    _touch(tmp_path, "kernel/grad_kernel.cpp")
    assert _optimize_built_kernel_present(tmp_path, plugin) is True


def test_backward_empty_kernel_dir_rejects(tmp_path):
    plugin = get_plugin("backward")
    assert plugin is not None
    (tmp_path / "kernel").mkdir()
    assert _optimize_built_kernel_present(tmp_path, plugin) is False


def test_legacy_workspace_keeps_kernel_fallback(tmp_path):
    _touch(tmp_path, "kernel/kernel.cpp")
    assert _optimize_built_kernel_present(tmp_path, None) is True
