# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check finite byte-layout examples in layout-check JSON blocks, without eval.

This does not parse C++, infer expected offsets, or prove synchronization correctness.
"""

import argparse
import ast
import json
from pathlib import Path
import re

U64_MAX = (1 << 64) - 1


def integer(value):
    if type(value) is not int or not 0 <= value <= U64_MAX:
        raise ValueError("values must be unsigned 64-bit integers")
    return value


def offset(expression, variables):
    if not isinstance(expression, str) or len(expression) > 512:
        raise ValueError("formula must be a string of at most 512 characters")
    tree = ast.parse(expression, mode="eval")

    def visit(node):
        if isinstance(node, ast.Constant):
            return integer(node.value)
        if isinstance(node, ast.Name) and node.id in variables:
            return integer(variables[node.id])
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult)
        ):
            left, right = visit(node.left), visit(node.right)
            value = (
                left + right
                if isinstance(node.op, ast.Add)
                else left - right
                if isinstance(node.op, ast.Sub)
                else left * right
            )
            return integer(value)
        raise ValueError(
            "formula only allows integer constants, declared variables, +, -, *"
        )

    return visit(tree.body)


def validate(document):
    errors = []
    blocks = re.findall(r"^```layout-check\s*\n(.*?)^```\s*$", document, re.M | re.S)
    if not blocks:
        return ["missing layout-check JSON block"]
    for block_no, block in enumerate(blocks, 1):
        try:
            data = json.loads(block)
            if (
                not isinstance(data, dict)
                or not isinstance(data.get("cases"), list)
                or not data["cases"]
            ):
                raise ValueError("each block needs a nonempty cases list")
        except (ValueError, TypeError) as exc:
            errors.append("block %d: %s" % (block_no, exc))
            continue
        for index, case in enumerate(data["cases"], 1):
            try:
                if not isinstance(case, dict):
                    raise ValueError("case must be an object")
                variables = case["vars"]
                if not isinstance(variables, dict):
                    raise ValueError("vars must be an object")
                for value in variables.values():
                    integer(value)
                actual = offset(case["formula"], variables)
                expected = integer(case["expected"])
                length = integer(case["length"])
                capacity = integer(case["capacity"])
                element = integer(case["element_size"])
                if element == 0:
                    raise ValueError("element_size must be positive")
                if actual != expected:
                    raise ValueError("offset %d != expected %d" % (actual, expected))
                if integer(actual + length) > capacity:
                    raise ValueError("slice exceeds buffer capacity")
                if actual % element or length % element:
                    raise ValueError("offset/length is not element-aligned")
            except (
                KeyError,
                ValueError,
                TypeError,
                SyntaxError,
                RecursionError,
            ) as exc:
                errors.append("block %d case %d: %s" % (block_no, index, exc))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", nargs="+")
    args = parser.parse_args()
    errors = []
    for path in args.spec:
        try:
            errors.extend(
                "%s: %s" % (path, error) for error in validate(Path(path).read_text())
            )
        except OSError as exc:
            errors.append(str(exc))
    for error in errors:
        print("ERROR:", error)
    if not errors:
        print("量化样例通过；仍需核对实际代码、独立布局与完整构建及双轨验收。")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
