# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression tests for pre_build_check.py static analysis gate.

Tests that the checker catches all the bug patterns found in
LightningIndexerGrad P128-P134, plus other known failure modes.

Run: python3 -m pytest src/scripts/orchestrator/tests/test_pre_build_check.py -v
     or: python3 src/scripts/orchestrator/tests/test_pre_build_check.py
"""
import os
import sys
import tempfile
try:
    import pytest
except ImportError:
    pytest = None  # fallback: run as standalone script

# Add orchestrator dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pre_build_check import (
    parse_kernel_header,
    check_ub_layout,
    check_sync_audit,
    check_event_lifecycle,
    check_alignment,
    Finding,
)


def _make_header(content: str) -> str:
    """Write content to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".h", delete=False)
    f.write(content)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# UB Layout tests (OL-173)
# ---------------------------------------------------------------------------

def test_ub_layout_clean_chain():
    """Valid offset chain should produce no errors."""
    content = """
constexpr static int64_t TOTAL_SIZE = 193536;
constexpr static int64_t LIMIT_TOPK = 2048;
constexpr static int64_t buf1Offset = 0;
constexpr static int64_t buf1Size = 4096;
constexpr static int64_t buf2Offset = buf1Offset + buf1Size;
constexpr static int64_t buf2Size = 8192;
constexpr static int64_t buf3Offset = buf2Offset + buf2Size;
constexpr static int64_t buf3Size = 32768;
"""
    path = _make_header(content)
    try:
        result = parse_kernel_header(path)
        check_ub_layout(result)
        assert result.passed, f"Expected PASS, got: {result.findings}"
        assert len(result.buffers) == 3
        assert result.buffers[0].offset == 0
        assert result.buffers[1].offset == 4096
        assert result.buffers[2].offset == 12288  # 4096 + 8192
    finally:
        os.unlink(path)


def test_ub_layout_p134_overlap():
    """P134: reluInPingUbOffset = 0 + indicesUbSize should be indicesUbOffset + indicesUbSize."""
    content = """
constexpr static int64_t TOTAL_SIZE = 189 * 1024;
constexpr static int64_t LIMIT_TOPK = 2048;
constexpr static int64_t gatherPingUbOffset = 0;
constexpr static int64_t gatherPingUbSize = 128 * 4 * 8;
constexpr static int64_t indicesUbOffset = gatherPingUbOffset + gatherPingUbSize;
constexpr static int64_t indicesUbSize = LIMIT_TOPK * 4;
// BUG: 0 + indicesUbSize = 8192, should be indicesUbOffset + indicesUbSize = 16384
constexpr static int64_t reluInPingUbOffset = 0 + indicesUbSize;
constexpr static int64_t reluInPingUbSize = 4 * LIMIT_TOPK * 4;
"""
    path = _make_header(content)
    try:
        result = parse_kernel_header(path)
        check_ub_layout(result)
        assert not result.passed, "Expected FAIL for P134 overlap"
        overlap_errors = [f for f in result.findings
                          if f.severity == "ERROR" and "OVERLAP" in f.message]
        assert len(overlap_errors) >= 1, f"Expected overlap error, got: {result.findings}"
    finally:
        os.unlink(path)


def test_ub_layout_p132_max_ub_size_overflow():
    """P132: MAX_UB_SIZE exceeds available workspace by 2.5x."""
    content = """
constexpr static int64_t TOTAL_SIZE = 193536;
constexpr static int64_t buf1Offset = 0;
constexpr static int64_t buf1Size = 154624;
// Only 38912 bytes remaining, but MAX_UB_SIZE = 24192 floats = 96768 bytes = 2.5x overflow
constexpr static int64_t MAX_UB_SIZE = 24192;
"""
    path = _make_header(content)
    try:
        result = parse_kernel_header(path)
        check_ub_layout(result)
        assert not result.passed, "Expected FAIL for MAX_UB_SIZE overflow"
        overflow_errors = [f for f in result.findings
                           if f.severity == "ERROR" and "exceeds available" in f.message]
        assert len(overflow_errors) >= 1, f"Expected overflow error, got: {result.findings}"
    finally:
        os.unlink(path)


def test_ub_layout_total_overflow():
    """Declared buffers exceed TOTAL_SIZE."""
    content = """
constexpr static int64_t TOTAL_SIZE = 4096;
constexpr static int64_t buf1Offset = 0;
constexpr static int64_t buf1Size = 3000;
constexpr static int64_t buf2Offset = buf1Offset + buf1Size;
constexpr static int64_t buf2Size = 2000;
"""
    path = _make_header(content)
    try:
        result = parse_kernel_header(path)
        check_ub_layout(result)
        assert not result.passed, "Expected FAIL for TOTAL_SIZE overflow"
        overflow_errors = []
        for finding in result.findings:
            message = finding.message.lower()
            if finding.severity == "ERROR" and (
                "exceed" in message or "overflow" in message
            ):
                overflow_errors.append(finding)
        assert len(overflow_errors) >= 1, f"Expected overflow error, got: {result.findings}"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Sync audit tests (OL-171)
# ---------------------------------------------------------------------------

def test_sync_audit_aicore_only_syncall():
    """SyncAll in AiCore-only pipeline should be flagged."""
    content = """// AiCore-only kernel (no MIX_AIC)
void Process() {
    SyncAll();
}
"""
    path = _make_header(content)
    try:
        result = parse_kernel_header(path)
        check_sync_audit(result)
        assert not result.passed, "Expected FAIL for SyncAll in AiCore-only"
        sync_errors = [f for f in result.findings
                       if f.check == "SYNC_AUDIT"]
        assert len(sync_errors) >= 1, f"Expected sync error, got: {result.findings}"
    finally:
        os.unlink(path)


def test_sync_audit_mix_aic_ok():
    """SyncAll in MIX_AIC pipeline should NOT be flagged (cross-core needed)."""
    content = """#define MIX_AIC 1
void Process() {
    SyncAll();
}
"""
    path = _make_header(content)
    try:
        result = parse_kernel_header(path)
        check_sync_audit(result)
        assert result.passed, f"Expected PASS for SyncAll in MIX_AIC, got: {result.findings}"
    finally:
        os.unlink(path)


def test_sync_audit_loop_syncall():
    """SyncAll inside a loop is the most dangerous pattern."""
    content = """// AiCore-only kernel
void Process() {
    for (int b = 0; b < batch; b++) {
        DoWork();
        SyncAll();
    }
}
"""
    path = _make_header(content)
    try:
        result = parse_kernel_header(path)
        check_sync_audit(result)
        loop_errors = [f for f in result.findings
                       if "loop" in f.message.lower()]
        assert len(loop_errors) >= 1, f"Expected in-loop SyncAll error, got: {result.findings}"
    finally:
        os.unlink(path)


def test_sync_audit_pipebarrier_ok():
    """PipeBarrier<PIPE_ALL> in AiCore-only pipeline should pass."""
    content = """// AiCore-only kernel
void Process() {
    PipeBarrier<PIPE_ALL>();
    for (int b = 0; b < batch; b++) {
        DoWork();
        PipeBarrier<PIPE_ALL>();
    }
}
"""
    path = _make_header(content)
    try:
        result = parse_kernel_header(path)
        check_sync_audit(result)
        assert result.passed, f"Expected PASS for PipeBarrier, got: {result.findings}"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Event lifecycle tests (OL-171)
# ---------------------------------------------------------------------------

def test_event_free_inside_loop():
    """FreeEvent inside a loop with alloc outside should warn."""
    content = """// kernel
void Process() {
    AllocEvent(e1);
    for (int b = 0; b < batch; b++) {
        DoWork();
        FreeEvent(e1);
    }
}
"""
    path = _make_header(content)
    try:
        result = parse_kernel_header(path)
        check_event_lifecycle(result)
        event_warns = [f for f in result.findings
                       if f.check == "EVENT_LIFECYCLE"]
        assert len(event_warns) >= 1, f"Expected event lifecycle warning, got: {result.findings}"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Alignment tests
# ---------------------------------------------------------------------------

def test_alignment_32b_datacopy():
    """DataCopy with non-32B-aligned byte count should warn."""
    content = """
void Process() {
    DataCopy(dst, src, 4);  // 4 half = 8B, not 32B-aligned
}
"""
    path = _make_header(content)
    try:
        result = parse_kernel_header(path)
        check_alignment(result)
        align_warns = [f for f in result.findings
                       if f.check == "ALIGNMENT"]
        assert len(align_warns) >= 1, f"Expected alignment warning, got: {result.findings}"
    finally:
        os.unlink(path)


def test_alignment_does_not_assume_literal_eight_is_half():
    """A simple count of eight is valid for fp32; dtype is not inferable here."""
    path = _make_header("void Process() { DataCopy(dst, src, 8); }\n")
    try:
        result = parse_kernel_header(path)
        check_alignment(result)
        assert [f for f in result.findings if f.check == "ALIGNMENT"] == []
    finally:
        os.unlink(path)


def test_alignment_does_not_treat_all_blocklen_fields_as_bytes():
    """Params uses 32B blocks while the Pad extension uses bytes."""
    content = """
void Process() {
    DataCopyParams cp;
    cp.blockCount = 1;
    cp.blockLen = 1;
    DataCopy(dst, src, cp);
    DataCopyExtParams ext;
    ext.blockLen = 7;
    DataCopyPad(dst, src, ext);
}
"""
    path = _make_header(content)
    try:
        result = parse_kernel_header(path)
        check_alignment(result)
        assert [f for f in result.findings if f.check == "ALIGNMENT"] == []
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Integration: full pre_build_check run
# ---------------------------------------------------------------------------

def test_full_pre_build_check_clean():
    """A clean header should pass all checks."""
    content = """
constexpr static int64_t TOTAL_SIZE = 193536;
constexpr static int64_t buf1Offset = 0;
constexpr static int64_t buf1Size = 4096;
constexpr static int64_t buf2Offset = buf1Offset + buf1Size;
constexpr static int64_t buf2Size = 8192;
// MIX_AIC pipeline
void Process() {
    PipeBarrier<PIPE_ALL>();
}
"""
    path = _make_header(content)
    try:
        result = parse_kernel_header(path)
        check_ub_layout(result)
        check_sync_audit(result)
        check_event_lifecycle(result)
        check_alignment(result)
        # Should pass all checks
        errors = [f for f in result.findings if f.severity == "ERROR"]
        assert len(errors) == 0, f"Expected no errors, got: {errors}"
    finally:
        os.unlink(path)
