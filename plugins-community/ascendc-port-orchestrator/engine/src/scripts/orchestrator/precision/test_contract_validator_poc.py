# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Selection-policy tests — the anti-metric-shop core (selection is contract-fixed, not agent-choosable)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract_validator_poc import Contract, select_validator, grade_contract


def test_int_bit_exact():
    assert select_validator(Contract("cast", "int", "int32"))[0] == "bit_exact"


def test_elementwise_same_dtype():
    assert select_validator(Contract("gelu", "elementwise", "float16", refs_available=("model_forward",)))[
                            0] == "same_dtype_threshold"


def test_numhard_fp64_independent_double_ratio():
    v, _ = select_validator(Contract("fa", "numerically_hard", "float16",
                            "L1", ("fp64_golden", "independent_baseline")))
    assert v == "double_baseline_ratio"


def test_numhard_fp64_only_is_ecosystem_not_lenient_ratio():
    # no independent baseline → fall to single-baseline threshold, not a ratio
    v, _ = select_validator(Contract("fa", "numerically_hard", "float16", "L1", ("fp64_golden",)))
    assert v == "single_baseline_threshold"


def test_numhard_vendor_declares_circular():
    v, rule = select_validator(Contract("fa", "numerically_hard", "float16", "L1", ("same_dtype_vendor",)))
    assert v == "same_dtype_threshold" and "circular" in rule


def test_selection_is_deterministic_not_choosable():
    c = Contract("fa", "numerically_hard", "float16", "L1", ("fp64_golden", "independent_baseline"))
    assert select_validator(c) == select_validator(c)  # fixed table, no runtime freedom


def test_tier_changes_thresholds_not_validator():
    a = select_validator(Contract("fa", "numerically_hard", "float16", "L0", ("fp64_golden", "independent_baseline")))
    b = select_validator(Contract("fa", "numerically_hard", "float16", "L2", ("fp64_golden", "independent_baseline")))
    assert a[0] == b[0] == "double_baseline_ratio"  # tier affects thresholds inside the validator, not the choice


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
