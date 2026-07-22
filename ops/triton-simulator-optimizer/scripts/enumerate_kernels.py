#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
"""枚举实现文件中全部 @triton.jit / @jit 装饰的 kernel 函数名。

供 triton-simulator-optimizer 「全 kernel 采集覆盖门禁」步骤 1 使用：
多 kernel 算子的瓶颈可能在任意一个 kernel，手工枚举易漏，本脚本用 AST 自动提取。

用法:
    python3 enumerate_kernels.py <impl.py> [--help-skip]

排除规则：framework 辅助 kernel（ZerosLike / Empty / OnesLike / Contiguous 等）非算子逻辑，
即便被 @triton.jit 装饰也跳过（名单见 SKIP_NAMES）。
"""
import argparse
import ast
import sys

# framework 辅助 kernel（非算子逻辑），即使被 @triton.jit 装饰也排除
SKIP_NAMES = {
    "ZerosLike", "Empty", "OnesLike", "Contiguous",
    "Zeros", "Ones", "Full", "EmptyLike",
}


def _is_jit_decorator(d) -> bool:
    """判断装饰器是否为 @triton.jit / @triton.jit(...) / @jit / @jit(...)。"""
    func = d.func if isinstance(d, ast.Call) else d
    # triton.jit 形式
    if isinstance(func, ast.Attribute) and func.attr == "jit":
        return True
    # 裸 @jit 形式
    if isinstance(func, ast.Name) and func.id == "jit":
        return True
    return False


def enumerate_kernels(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=path)
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any(isinstance(d, (ast.Attribute, ast.Name, ast.Call)) and _is_jit_decorator(d)
                   for d in node.decorator_list):
            continue
        if node.name in SKIP_NAMES:
            continue
        names.append(node.name)
    # 去重保序
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="枚举 @triton.jit kernel 函数名")
    p.add_argument("file", help="Triton 实现文件路径 (.py)")
    p.add_argument("--help-skip", action="store_true", help="打印排除名单后退出")
    args = p.parse_args()
    if args.help_skip:
        print("SKIP_NAMES:", ", ".join(sorted(SKIP_NAMES)))
        return 0
    names = enumerate_kernels(args.file)
    if not names:
        print("[]  # 未找到 @triton.jit kernel", file=sys.stderr)
        return 1
    # 一行一个，便于 shell 循环: for k in $(python3 enumerate_kernels.py impl.py); do ...
    for n in names:
        print(n)
    print(f"# 共 {len(names)} 个算子 kernel（已排除 framework 辅助 kernel）", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
