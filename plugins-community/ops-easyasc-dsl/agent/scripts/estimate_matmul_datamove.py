# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
from typing import Any, Dict, List, Optional, Sequence, Tuple

from prettytable import PrettyTable


MODE_LEFT_FIRST = "left_first"
MODE_RIGHT_FIRST = "right_first"
MODE_BALANCED = "balanced"
MAX_TOTAL_TILE_ELEMENTS = 128 * 1024
MAX_L0C_TILE_ELEMENTS_DBUF = 32 * 1024
MAX_L0C_TILE_ELEMENTS_TENSOR = 64 * 1024
SPLIT_MODE_SPLIT_M = "split_m"
SPLIT_MODE_SPLIT_N = "split_n"
SPLIT_MODE_MIX = "mix"
TILE_CANDIDATES = (32, 64, 128, 256, 512)

VALID_MODES = (
    MODE_LEFT_FIRST,
    MODE_RIGHT_FIRST,
    MODE_BALANCED,
)

VALID_SPLIT_MODES = (
    SPLIT_MODE_SPLIT_M,
    SPLIT_MODE_SPLIT_N,
    SPLIT_MODE_MIX,
)

StrategyItem = Dict[str, Any]
SplitCandidate = Tuple[int, int]

__all__ = [
    "MODE_LEFT_FIRST",
    "MODE_RIGHT_FIRST",
    "MODE_BALANCED",
    "MAX_TOTAL_TILE_ELEMENTS",
    "MAX_L0C_TILE_ELEMENTS_DBUF",
    "MAX_L0C_TILE_ELEMENTS_TENSOR",
    "SPLIT_MODE_SPLIT_M",
    "SPLIT_MODE_SPLIT_N",
    "SPLIT_MODE_MIX",
    "TILE_CANDIDATES",
    "VALID_MODES",
    "VALID_SPLIT_MODES",
    "align_tile_k",
    "ceil_div",
    "estimate_effective_k_datamove",
    "estimate_l0c_tile_elements",
    "estimate_multi_core",
    "estimate_strategy",
    "get_l0c_tile_element_limit",
    "estimate_total_tile_elements",
    "estimate_percore_datamove",
    "format_datamove",
]


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("b must be positive")
    return (a + b - 1) // b


def format_datamove(datamove: int) -> str:
    return "{:,}".format(datamove)


def format_element_count(element_count: float) -> str:
    if element_count.is_integer():
        return "{:,}".format(int(element_count))
    return "{:,.1f}".format(element_count)


def apply_dbuf_size(size: int, use_dbuf: bool) -> float:
    if use_dbuf:
        return float(size)
    return float(size) / 2.0


def get_l0c_tile_element_limit(dbuf_l0c: bool) -> int:
    if dbuf_l0c:
        return MAX_L0C_TILE_ELEMENTS_DBUF
    return MAX_L0C_TILE_ELEMENTS_TENSOR


def estimate_l0c_tile_elements(TILEM: int, TILEN: int) -> int:
    return TILEM * TILEN


def align_tile_k(TILEK: int, k: int) -> int:
    if TILEK != k:
        return ceil_div(TILEK, 256) * 256
    return TILEK


def estimate_effective_k_datamove(k: int, TILEK: int) -> int:
    if TILEK == k:
        return k
    return ceil_div(k, TILEK) * align_tile_k(TILEK, k)


def estimate_total_tile_elements(
    TILEM: int, TILEN: int, TILEK: int,
    k: int, mode: str,
    dbuf_left: bool = True,
    dbuf_right: bool = True,
) -> float:
    normalized_mode = mode.strip().lower()

    if normalized_mode == MODE_LEFT_FIRST:
        left_size = TILEM * k
        right_size = TILEN * TILEK
        return (
            apply_dbuf_size(left_size, dbuf_left)
            + apply_dbuf_size(right_size, dbuf_right)
        )
    if normalized_mode == MODE_RIGHT_FIRST:
        left_size = TILEM * TILEK
        right_size = TILEN * k
        return (
            apply_dbuf_size(left_size, dbuf_left)
            + apply_dbuf_size(right_size, dbuf_right)
        )
    if normalized_mode == MODE_BALANCED:
        left_size = TILEM * TILEK
        right_size = TILEN * TILEK
        return (
            apply_dbuf_size(left_size, dbuf_left)
            + apply_dbuf_size(right_size, dbuf_right)
        )

    raise ValueError(
        "Unsupported mode: {mode}. Valid modes are: {modes}".format(
            mode=mode,
            modes=", ".join(VALID_MODES),
        )
    )


def estimate_percore_datamove(
    m: int, n: int, k: int,
    TILEM: int, TILEN: int, TILEK: int,
    mode: str,
    dbuf_left: bool = True,
    dbuf_right: bool = True,
    dbuf_l0c: bool = True,
) -> int:
    if m < 0 or n < 0 or k < 0:
        raise ValueError("m, n, and k must be non-negative")
    if TILEM <= 0 or TILEN <= 0 or TILEK <= 0:
        raise ValueError("TILEM, TILEN, and TILEK must be positive")

    normalized_mode = mode.strip().lower()
    total_tile_elements = estimate_total_tile_elements(
        TILEM, TILEN, TILEK,
        k, normalized_mode,
        dbuf_left=dbuf_left,
        dbuf_right=dbuf_right,
    )

    if total_tile_elements > MAX_TOTAL_TILE_ELEMENTS:
        raise ValueError(
            "Total tile elements exceed limit: {total} > {limit}".format(
                total=format_element_count(total_tile_elements),
                limit=format_datamove(MAX_TOTAL_TILE_ELEMENTS),
            )
        )
    l0c_tile_elements = estimate_l0c_tile_elements(TILEM, TILEN)
    l0c_tile_limit = get_l0c_tile_element_limit(dbuf_l0c)
    if l0c_tile_elements > l0c_tile_limit:
        raise ValueError(
            "L0C tile elements exceed limit: {total} > {limit}".format(
                total=format_datamove(l0c_tile_elements),
                limit=format_datamove(l0c_tile_limit),
            )
        )

    effective_k_datamove = estimate_effective_k_datamove(k, TILEK)

    # Only the operand visited through the TILEK loop uses the aligned effective K span.
    if normalized_mode == MODE_LEFT_FIRST:
        return m * k + ceil_div(m, TILEM) * n * effective_k_datamove
    if normalized_mode == MODE_RIGHT_FIRST:
        return ceil_div(n, TILEN) * m * effective_k_datamove + n * k
    if normalized_mode == MODE_BALANCED:
        return (
            ceil_div(n, TILEN) * m * effective_k_datamove
            + ceil_div(m, TILEM) * n * effective_k_datamove
        )

    raise ValueError(
        "Unsupported mode: {mode}. Valid modes are: {modes}".format(
            mode=mode,
            modes=", ".join(VALID_MODES),
        )
    )


def estimate_multi_core_with_mode(
    m: int, n: int, k: int,
    m_split: int, n_split: int,
    TILEM: int, TILEN: int, TILEK: int,
    mode: str,
    nonempty_only: bool = False,
    dbuf_left: bool = True,
    dbuf_right: bool = True,
    dbuf_l0c: bool = True,
) -> int:
    if m_split <= 0 or n_split <= 0:
        raise ValueError("m_split and n_split must be positive")

    tile_m = ceil_div(m, TILEM)
    tile_n = ceil_div(n, TILEN)
    single_m = ceil_div(tile_m, m_split) * TILEM
    single_n = ceil_div(tile_n, n_split) * TILEN
    single_k = k
    active_m_split = min(m_split, tile_m)
    active_n_split = min(n_split, tile_n)

    percore_datamove = estimate_percore_datamove(
        single_m, single_n, single_k,
        TILEM, TILEN, TILEK,
        mode,
        dbuf_left=dbuf_left,
        dbuf_right=dbuf_right,
        dbuf_l0c=dbuf_l0c,
    )

    if nonempty_only:
        return percore_datamove * active_m_split * active_n_split
    return percore_datamove * m_split * n_split


def estimate_multi_core(
    m: int, n: int, k: int,
    m_split: int, n_split: int,
    TILEM: int, TILEN: int, TILEK: int,
    nonempty_only: bool = False,
    dbuf_left: bool = True,
    dbuf_right: bool = True,
    dbuf_l0c: bool = True,
) -> int:
    best_total_datamove: Optional[int] = None
    errors: List[str] = []

    for mode in VALID_MODES:
        try:
            total_datamove = estimate_multi_core_with_mode(
                m, n, k,
                m_split, n_split,
                TILEM, TILEN, TILEK,
                mode,
                nonempty_only=nonempty_only,
                dbuf_left=dbuf_left,
                dbuf_right=dbuf_right,
                dbuf_l0c=dbuf_l0c,
            )
        except ValueError as err:
            errors.append("{mode}: {msg}".format(mode=mode, msg=err))
            continue

        if best_total_datamove is None or total_datamove < best_total_datamove:
            best_total_datamove = total_datamove

    if best_total_datamove is None:
        raise ValueError(
            "No valid mode for multi-core estimate. {errors}".format(
                errors="; ".join(errors),
            )
        )

    return best_total_datamove


def generate_split_candidates(num_core: int, split_mode: str) -> List[SplitCandidate]:
    if num_core <= 0:
        raise ValueError("num_core must be positive")

    normalized_split_mode = split_mode.strip().lower()

    if normalized_split_mode == SPLIT_MODE_SPLIT_M:
        return [(num_core, 1)]
    if normalized_split_mode == SPLIT_MODE_SPLIT_N:
        return [(1, num_core)]
    if normalized_split_mode == SPLIT_MODE_MIX:
        candidates = []
        for m_split in range(1, num_core + 1):
            if num_core % m_split == 0:
                candidates.append((m_split, num_core // m_split))
        return candidates

    raise ValueError(
        "Unsupported split_mode: {mode}. Valid split modes are: {modes}".format(
            mode=split_mode,
            modes=", ".join(VALID_SPLIT_MODES),
        )
    )


def generate_dbuf_candidates(mode: str) -> List[Tuple[bool, bool]]:
    normalized_mode = mode.strip().lower()

    if normalized_mode == MODE_BALANCED:
        return [(True, True)]
    if normalized_mode == MODE_LEFT_FIRST:
        return [(True, True), (False, True)]
    if normalized_mode == MODE_RIGHT_FIRST:
        return [(True, True), (True, False)]

    raise ValueError(
        "Unsupported mode: {mode}. Valid modes are: {modes}".format(
            mode=mode,
            modes=", ".join(VALID_MODES),
        )
    )


def build_strategy_table(results: Sequence[StrategyItem]) -> PrettyTable:
    table = PrettyTable()
    table.field_names = [
        "datamove",
        "expansion_ratio",
        "m_split",
        "n_split",
        "TILEM",
        "TILEN",
        "TILEK",
        "dbuf_left",
        "dbuf_right",
        "dbuf_l0c",
        "mode",
    ]

    for result in results:
        table.add_row(
            [
                format_datamove(result["datamove"]),
                "{:.6f}".format(result["expansion_ratio"]),
                result["m_split"],
                result["n_split"],
                result["TILEM"],
                result["TILEN"],
                result["TILEK"],
                result["dbuf_left"],
                result["dbuf_right"],
                result["dbuf_l0c"],
                result["mode"],
            ]
        )

    return table


def estimate_strategy(
    m: int, n: int, k: int,
    num_core: int,
    split_mode: str,
    min_tile_m: Optional[int] = None,
    min_tile_n: Optional[int] = None,
    dbuf_l0c: bool = True,
) -> Dict[str, Any]:
    split_candidates = generate_split_candidates(num_core, split_mode)
    baseline_datamove = m * k + n * k
    tile_m_candidates = [
        tile for tile in TILE_CANDIDATES if min_tile_m is None or tile >= min_tile_m
    ]
    tile_n_candidates = [
        tile for tile in TILE_CANDIDATES if min_tile_n is None or tile >= min_tile_n
    ]
    tile_k_candidates = sorted(
        set([tile for tile in TILE_CANDIDATES if tile <= k] + [k])
    )

    if not tile_m_candidates:
        raise ValueError("No TILEM candidates remain after min_tile_m filtering")
    if not tile_n_candidates:
        raise ValueError("No TILEN candidates remain after min_tile_n filtering")

    best_datamove: Optional[int] = None
    best_results: List[StrategyItem] = []

    for m_split, n_split in split_candidates:
        for TILEM in tile_m_candidates:
            for TILEN in tile_n_candidates:
                for TILEK in tile_k_candidates:
                    for mode in VALID_MODES:
                        for dbuf_left, dbuf_right in generate_dbuf_candidates(mode):
                            try:
                                datamove = estimate_multi_core_with_mode(
                                    m, n, k,
                                    m_split, n_split,
                                    TILEM, TILEN, TILEK,
                                    mode,
                                    nonempty_only=False,
                                    dbuf_left=dbuf_left,
                                    dbuf_right=dbuf_right,
                                    dbuf_l0c=dbuf_l0c,
                                )
                            except ValueError:
                                continue

                            if baseline_datamove == 0:
                                expansion_ratio = 0.0 if datamove == 0 else float("inf")
                            else:
                                expansion_ratio = (
                                    float(datamove) / float(baseline_datamove)
                                )

                            result = {
                                "datamove": datamove,
                                "expansion_ratio": expansion_ratio,
                                "m_split": m_split,
                                "n_split": n_split,
                                "TILEM": TILEM,
                                "TILEN": TILEN,
                                "TILEK": TILEK,
                                "dbuf_left": dbuf_left,
                                "dbuf_right": dbuf_right,
                                "dbuf_l0c": dbuf_l0c,
                                "mode": mode,
                            }

                            if best_datamove is None or datamove < best_datamove:
                                best_datamove = datamove
                                best_results = [result]
                            elif datamove == best_datamove:
                                best_results.append(result)

    if best_datamove is None:
        raise ValueError("No valid strategy found")

    best_results.sort(
        key=lambda item: (
            item["m_split"],
            item["n_split"],
            item["TILEM"],
            item["TILEN"],
            item["TILEK"],
            item["dbuf_left"],
            item["dbuf_right"],
            item["dbuf_l0c"],
            item["mode"],
        )
    )
    table = build_strategy_table(best_results)

    return {
        "baseline_datamove": baseline_datamove,
        "best_datamove": best_datamove,
        "best_results": best_results,
        "table": table,
    }
