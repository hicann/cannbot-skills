# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P96 regression — PluginProtocol pass_b shape contract.

Verifies port_a3_to_a5 plugin overrides BasePlugin defaults so:
- pass_b_required() returns False (port_a3 mode pass_b is degenerate)
- pass_b_default_when_skipped() returns canonical N/A shape
- verifier_canonical_filenames() excludes run_pass_b.py
- forbidden_workspace_files() includes run_pass_b.py

See:
- src/scripts/orchestrator/plugins/base.py
- src/scripts/orchestrator/plugins/port_a3/__init__.py
- ANTI_PRESSURE_PROTOCOLS.md §P9
- aog-self-critic C-PORT-A3-PASS-B-SCHEMA
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))


def test_base_plugin_pass_b_required_default_true():
    """BasePlugin default: benchmark-like contract requires pass_b."""
    from plugins.base import BasePlugin
    bp = BasePlugin()
    assert bp.pass_b_required() is True


def test_base_plugin_default_skipped_shape_empty():
    """BasePlugin's default skipped shape is empty (only consulted when
    pass_b_required=False).
    """
    from plugins.base import BasePlugin
    bp = BasePlugin()
    assert bp.pass_b_default_when_skipped() == {}


def test_base_plugin_verifier_filenames_empty():
    """BasePlugin: empty (plugin should override with canonical names)."""
    from plugins.base import BasePlugin
    bp = BasePlugin()
    assert bp.verifier_canonical_filenames() == ()


def test_base_plugin_forbidden_workspace_files_empty():
    """BasePlugin: no forbidden files by default."""
    from plugins.base import BasePlugin
    bp = BasePlugin()
    assert bp.forbidden_workspace_files() == ()


def test_port_a3_pass_b_not_required():
    """port_a3 plugin: pass_b is degenerate."""
    from plugins.port_a3 import PortA3Plugin
    p = PortA3Plugin()
    assert p.pass_b_required() is False


def test_port_a3_canonical_skipped_shape():
    """port_a3 default skipped shape: N/A status + canonical port_a3 reason."""
    from plugins.port_a3 import PortA3Plugin
    p = PortA3Plugin()
    shape = p.pass_b_default_when_skipped()
    assert shape.get("status") == "N/A"
    assert "port_a3_to_a5 mode" in shape.get("reason", "")
    assert "subsumed by pass_a" in shape.get("reason", "")
    assert "edge_dataset" in shape.get("reason", "")
    assert "n/a" in shape.get("method", "").lower()


def test_port_a3_verifier_filenames_excludes_pass_b():
    """port_a3 plugin: verifier names do NOT include run_pass_b.py."""
    from plugins.port_a3 import PortA3Plugin
    p = PortA3Plugin()
    fnames = p.verifier_canonical_filenames()
    assert "pass_a_runner.py" in fnames
    assert "run_pass_b.py" not in fnames


def test_port_a3_forbidden_files_include_run_pass_b():
    """port_a3 plugin: run_pass_b.py is FORBIDDEN at workspace root."""
    from plugins.port_a3 import PortA3Plugin
    p = PortA3Plugin()
    forbidden = p.forbidden_workspace_files()
    assert "run_pass_b.py" in forbidden


def test_port_a3_plugin_implements_protocol_with_new_fields():
    """PortA3Plugin still satisfies PluginProtocol after adding 4 new methods."""
    from plugins.base import PluginProtocol
    from plugins.port_a3 import PortA3Plugin
    p = PortA3Plugin()
    # runtime_checkable Protocol check
    assert isinstance(p, PluginProtocol)
    # Spot-check new methods are callable
    assert callable(p.pass_b_required)
    assert callable(p.pass_b_default_when_skipped)
    assert callable(p.verifier_canonical_filenames)
    assert callable(p.forbidden_workspace_files)
