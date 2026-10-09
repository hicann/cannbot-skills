#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""只读检查 executor 的 CalcCostCoeffParam 是否显式传入本次依赖的字段。

仅支持可解析的普通结构体字段与 Type{...} 初始化；未知语法返回 UNVERIFIED。
PASS 只证明各初始化点使用指定表达式，不证明选路、条件分支或算法正确。
"""

import argparse
import hashlib
import json
from pathlib import Path
import re


def without_comments(text):
    token = r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(
        token,
        lambda m: re.sub(r"[^\n]", " ", m[0])
        if m[0].startswith(("//", "/*"))
        else m[0],
        text,
        flags=re.S,
    )


def split_group(text, start, separator=","):
    """Read one balanced brace group, ignoring delimiters in string literals."""
    stack, parts, begin = ["}"], [], start + 1
    quote, escaped = None, False
    pairs = {"{": "}", "(": ")", "[": "]"}
    for i in range(start + 1, len(text)):
        char = text[i]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ('"', "'"):
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif char in "})]":
            if char != stack.pop():
                raise ValueError("括号不匹配")
            if not stack:
                last = text[begin:i].strip()
                if last:
                    parts.append(last)
                return parts, i
        elif char == separator and len(stack) == 1:
            parts.append(text[begin:i].strip())
            begin = i + 1
    raise ValueError("初始化或结构体未闭合")


def field_names(header):
    header = without_comments(header)
    match = re.search(r"\bstruct\s+CalcCostCoeffParam\s*\{", header)
    if not match:
        raise ValueError("找不到普通 CalcCostCoeffParam 结构体定义")
    declarations, _ = split_group(header, match.end() - 1, ";")
    fields = []
    for declaration in declarations:
        # Refuse methods, macros, multiple declarators and conditional layouts.
        match = re.fullmatch(r"[\w:\s<>*&]+\s+(\w+)\s*(?:=\s*[^;]+)?", declaration)
        if not match or any(c in declaration.split("=")[0] for c in "#(),{}"):
            raise ValueError("字段布局需人工核对：" + declaration)
        fields.append(match[1])
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("结构体字段为空或不唯一")
    return fields


def requirements(values):
    """Parse explicit field=expression pairs; never silently replace duplicates."""
    result = {}
    for value in values:
        field, separator, expression = value.partition("=")
        field, expression = field.strip(), expression.strip()
        if (
            not separator
            or not re.fullmatch(r"[A-Za-z_]\w*", field)
            or not expression
            or expression in ("nullptr", "NULL", "0", "{}")
            or field in result
        ):
            raise ValueError("字段要求须为不重复的 field=非空来源表达式: " + value)
        result[field] = expression
    return result


def inspect(
    executor, header, expected_calls, name_expression="algName", required_fields=None
):
    required = (
        {"algName": name_expression} if required_fields is None else required_fields
    )
    findings = []
    try:
        if expected_calls < 1 or not required:
            raise ValueError("调用数和需要核对的字段均不得为空")
        fields = field_names(header)
        for field in required:
            if field not in fields:
                raise ValueError("结构体中缺少所需字段: " + field)
        source = without_comments(executor)
        # Mask literals before locating type names so log text is not a call site.
        searchable = re.sub(
            r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', lambda m: " " * len(m[0]), source
        )
        sites = list(re.finditer(r"\bCalcCostCoeffParam\s*\{", searchable))
        if len(sites) != expected_calls:
            raise ValueError(
                "预期 %d 个直接聚合初始化点，实际 %d；定向核对调用写法/数量"
                % (expected_calls, len(sites))
            )
        for site in sites:
            values, _ = split_group(source, site.end() - 1)
            for field, expression in required.items():
                index = fields.index(field)
                value = values[index] if len(values) > index else None
                if any(re.match(r"\.\w+\s*=", v) for v in values) or len(values) > len(
                    fields
                ):
                    status, detail = (
                        "UNVERIFIED",
                        "指定成员初始化或参数布局超出解析范围",
                    )
                elif value is None or value in ("nullptr", "NULL", "0", "{}"):
                    status, detail = "FAIL", field + " 被省略或显式置空"
                elif re.sub(r"\s+", "", value) != re.sub(r"\s+", "", expression):
                    status, detail = "UNVERIFIED", "表达式不同，需核对别名/赋值来源"
                else:
                    status, detail = "PASS", "显式使用指定字段来源表达式"
                findings.append(
                    dict(
                        line=executor.count("\n", 0, site.start()) + 1,
                        field=field,
                        status=status,
                        expression=value,
                        detail=detail,
                    )
                )
    except ValueError as exc:
        findings.append(dict(status="UNVERIFIED", detail=str(exc)))
    status = (
        "FAIL"
        if any(f["status"] == "FAIL" for f in findings)
        else "UNVERIFIED"
        if any(f["status"] == "UNVERIFIED" for f in findings)
        else "PASS"
    )
    return dict(
        status=status,
        required_fields=required,
        findings=findings,
        limitation="仅检查直接聚合初始化及指定字段表达式；不证明实际值非空、来源有效、调用可达、lambda 捕获或最终算法命中。",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--cost-header", type=Path, required=True)
    parser.add_argument(
        "--expected-calls",
        type=int,
        required=True,
        help="按 executor 契约填写，不根据扫描结果自动接受",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--name-expression", help="兼容原算法名检查，默认 algName")
    selection.add_argument(
        "--require-field",
        action="append",
        default=[],
        metavar="FIELD=EXPR",
        help="可重复；只检查列出的依赖字段；不指定时检查 algName=algName",
    )
    args = parser.parse_args()
    if args.expected_calls < 1:
        parser.error("expected-calls 须为正")
    try:
        required = requirements(
            args.require_field
            or [
                "algName="
                + (
                    args.name_expression
                    if args.name_expression is not None
                    else "algName"
                )
            ]
        )
        data = {
            key: path.read_bytes()
            for key, path in [
                ("executor", args.executor),
                ("cost_header", args.cost_header),
            ]
        }
        result = inspect(
            data["executor"].decode(),
            data["cost_header"].decode(),
            args.expected_calls,
            required_fields=required,
        )
        result["sources"] = {
            key: dict(
                path=str(path.resolve()), sha256=hashlib.sha256(data[key]).hexdigest()
            )
            for key, path in [
                ("executor", args.executor),
                ("cost_header", args.cost_header),
            ]
        }
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return {"PASS": 0, "FAIL": 1, "UNVERIFIED": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
