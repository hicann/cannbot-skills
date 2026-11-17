#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Compatible AllGather entry; implementation shared with preview_cases.py."""

import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True

_script = Path(__file__).resolve().with_name("preview_cases.py")
_spec = importlib.util.spec_from_file_location("_ccu_collective_preview", _script)
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)
PROFILE = (
    Path(__file__).resolve().parents[1] / "references/hccl-test-allgather-profile.json"
)
DTYPE_BYTES, U64_MAX = _core.DTYPE_BYTES, _core.U64_MAX
digest, positive = _core.digest, _core.positive


def preview(root, rank_count, dtype, sizes, *, profile_path=PROFILE, **kwargs):
    return _core.preview(
        root,
        rank_count,
        dtype,
        sizes,
        op="all_gather",
        profile_path=profile_path,
        **kwargs,
    )


if __name__ == "__main__":
    sys.exit(_core.main(default_op="all_gather", profile_path=PROFILE))
