#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Read Python API information directly from a wheel without importing it."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import logging
import os
import re
import sys
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

INTERNAL_PREFIXES = ("_mlir/", "core/", "aicpu/")

CATEGORY_RULES = (
    {
        "id": "language-and-kernel",
        "title": "语言、JIT 与 Kernel",
        "description": "DSL 语言结构、constexpr、控制流、JIT、kernel 装饰器与规格声明。",
        "prefixes": ("lang",),
    },
    {
        "id": "tensor-types-and-layout",
        "title": "Tensor、类型与布局",
        "description": "Tensor、dtype、layout、tiler、数据结构与复合类型。",
        "prefixes": ("tensor", "types"),
    },
    {
        "id": "memory-and-dataflow",
        "title": "Buffer、Channel 与数据流",
        "description": "片上 Buffer、Channel 及其 acquire/commit/wait/release 数据流接口。",
        "prefixes": ("buffer", "channel"),
    },
    {
        "id": "data-movement",
        "title": "数据搬运与格式转换",
        "description": "内存复制、DMA、格式转换和搬运引擎描述。",
        "prefixes": ("ops.memcpy",),
    },
    {
        "id": "vector-register",
        "title": "Vector/寄存器操作",
        "description": "VF/寄存器算术、比较、装载、存储、规约、排列、mask 与 gather/scatter。",
        "prefixes": ("ops.reg",),
    },
    {
        "id": "scalar-and-simt",
        "title": "Scalar 与 SIMT",
        "description": "标量运算、标量指针、类型转换和 SIMT 接口。",
        "prefixes": ("ops.scalar", "ops.simt"),
    },
    {
        "id": "cube-matmul-and-convolution",
        "title": "Cube、Matmul 与 Convolution",
        "description": "Cube 配置、矩阵乘和卷积描述、装载及存储接口。",
        "prefixes": ("ops.cube", "ops.matmul", "ops.conv"),
    },
    {
        "id": "synchronization",
        "title": "同步与缓存控制",
        "description": "流水同步、barrier、事件、buffer 同步和缓存控制接口。",
        "prefixes": ("ops.sync",),
    },
    {
        "id": "distributed",
        "title": "分布式与通信",
        "description": "通信上下文、通信引擎和分布式设备指针接口。",
        "prefixes": ("ops.distributed",),
    },
    {
        "id": "architecture-and-platform",
        "title": "架构与平台信息",
        "description": "核索引、硬件状态、平台信息和内存容量查询。",
        "prefixes": ("ops.arch", "ops.info"),
    },
    {
        "id": "native-packaging-and-runtime",
        "title": "Native 打包与运行时",
        "description": "Native 产物发布、存储、注册和运行时查找接口。",
        "prefixes": ("package",),
    },
    {
        "id": "other-operations",
        "title": "其他算子接口",
        "description": "未归入专门类别的公开 ops 子模块接口。",
        "prefixes": ("ops",),
    },
    {
        "id": "internal-implementation",
        "title": "内部编译与运行实现",
        "description": "仅在显式包含内部模块时生成的 core、_mlir 与 aicpu 接口。",
        "prefixes": ("core", "_mlir", "aicpu"),
    },
    {
        "id": "other",
        "title": "其他 API",
        "description": "无法由定义模块归入上述类别的 API。",
        "prefixes": (),
    },
)


@dataclass
class Definition:
    module: str
    qualname: str
    kind: str
    signature: str
    file: str
    line: int
    end_line: int
    doc: str
    decorators: list[str] = field(default_factory=list)
    guards: list[dict[str, Any]] = field(default_factory=list)
    source_excerpt: str | None = None

    @property
    def top_name(self) -> str:
        return self.qualname.split(".", 1)[0]


@dataclass
class ImportEdge:
    export_module: str
    export_name: str
    source_module: str
    source_name: str
    file: str
    line: int
    star: bool = False


@dataclass
class ModuleInfo:
    name: str
    file: str
    is_package: bool
    definitions: list[Definition] = field(default_factory=list)
    imports: list[ImportEdge] = field(default_factory=list)
    all_names: list[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 wheel 全量提取 Python API，并生成中文 Markdown 文档和 YAML 索引。"
    )
    parser.add_argument(
        "--wheel", required=True, type=Path, help="wheel 文件路径，可使用相对路径"
    )
    parser.add_argument(
        "--package", default="cannbotdsl", help="wheel 内的 Python 包名"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="调用方指定的产物根目录；其下生成 cannbot-dsl-docs-search",
    )
    parser.add_argument(
        "--include-source", action="store_true", help="为每个 API 附带有限源码片段"
    )
    return parser.parse_args()


def node_text(source: str, node: ast.AST | None) -> str:
    if node is None:
        return ""
    return ast.get_source_segment(source, node) or ""


def format_arg(source: str, arg: ast.arg, default: ast.AST | None = None) -> str:
    text = arg.arg
    annotation = node_text(source, arg.annotation)
    if annotation:
        text += f": {annotation}"
    if default is not None:
        text += f"={node_text(source, default) or '...'}"
    return text


def format_parameters(
    source: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    drop_first: bool = False,
) -> str:
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.AST | None] = [None] * (
        len(positional) - len(args.defaults)
    ) + list(args.defaults)
    if drop_first and positional:
        positional = positional[1:]
        defaults = defaults[1:]
    parts: list[str] = []
    posonly_count = max(
        len(args.posonlyargs) - (1 if drop_first and args.posonlyargs else 0), 0
    )
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        parts.append(format_arg(source, arg, default))
        if posonly_count and index + 1 == posonly_count:
            parts.append("/")
    if args.vararg:
        parts.append("*" + format_arg(source, args.vararg))
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(format_arg(source, arg, default))
    if args.kwarg:
        parts.append("**" + format_arg(source, args.kwarg))
    return ", ".join(parts)


def format_function_signature(
    source: str, node: ast.FunctionDef | ast.AsyncFunctionDef
) -> str:
    returns = node_text(source, node.returns)
    suffix = f" -> {returns}" if returns else ""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({format_parameters(source, node)}){suffix}"


def format_class_signature(source: str, node: ast.ClassDef) -> str:
    init = next(
        (
            item
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == "__init__"
        ),
        None,
    )
    if init is not None:
        return f"class {node.name}({format_parameters(source, init, drop_first=True)})"
    bases = [node_text(source, base) for base in node.bases]
    return f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"


def module_name(member: str) -> tuple[str, bool]:
    path = member[:-4] if member.endswith(".pyi") else member[:-3]
    dotted = path.replace("/", ".")
    if dotted.endswith(".__init__"):
        return dotted[:-9], True
    return dotted, False


def limited_source(source: str, node: ast.AST, max_lines: int = 48) -> str:
    lines = source.splitlines()
    start = max(getattr(node, "lineno", 1) - 1, 0)
    end = min(getattr(node, "end_lineno", start + 1), start + max_lines)
    return "\n".join(lines[start:end])


class GuardVisitor(ast.NodeVisitor):
    def __init__(self, root: ast.AST, source: str, limit: int = 24) -> None:
        self.root = root
        self.source = source
        self.limit = limit
        self.items: list[dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is self.root:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node is self.root:
            self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node is self.root:
            for child in node.body:
                if not isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    self.visit(child)

    def visit_Assert(self, node: ast.Assert) -> None:
        if len(self.items) < self.limit:
            self.items.append(
                {
                    "line": node.lineno,
                    "kind": "assert",
                    "source": node_text(self.source, node)[:600],
                }
            )

    def visit_Raise(self, node: ast.Raise) -> None:
        if len(self.items) < self.limit:
            self.items.append(
                {
                    "line": node.lineno,
                    "kind": "raise",
                    "source": node_text(self.source, node)[:600],
                }
            )


def guard_evidence(source: str, node: ast.AST) -> list[dict[str, Any]]:
    visitor = GuardVisitor(node, source)
    visitor.visit(node)
    return visitor.items


def decorator_names(source: str, node: ast.AST) -> list[str]:
    return [
        text
        for item in getattr(node, "decorator_list", [])
        if (text := node_text(source, item))
    ]


def collect_definitions(
    module: str,
    member: str,
    source: str,
    tree: ast.Module,
    include_source: bool,
) -> list[Definition]:
    definitions: list[Definition] = []

    def assignment_definition(node: ast.Assign | ast.AnnAssign) -> Definition | None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if len(names) != 1 or names[0] == "__all__" or node.value is None:
            return None
        name = names[0]
        value = node_text(source, node.value).strip()
        annotation = (
            node_text(source, node.annotation).strip()
            if isinstance(node, ast.AnnAssign)
            else ""
        )
        declaration = (
            f"{name}: {annotation} = {value}" if annotation else f"{name} = {value}"
        )
        if len(declaration) > 500:
            declaration = declaration[:497] + "..."
        return Definition(
            module=module,
            qualname=name,
            kind="attribute",
            signature=declaration,
            file=member,
            line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            doc="",
            source_excerpt=limited_source(source, node) if include_source else None,
        )

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{prefix}.{node.name}" if prefix else node.name
                definitions.append(
                    Definition(
                        module=module,
                        qualname=qualname,
                        kind="method" if prefix else "function",
                        signature=format_function_signature(source, node),
                        file=member,
                        line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        doc=ast.get_docstring(node, clean=True) or "",
                        decorators=decorator_names(source, node),
                        guards=guard_evidence(source, node),
                        source_excerpt=limited_source(source, node)
                        if include_source
                        else None,
                    )
                )
            elif isinstance(node, ast.ClassDef):
                qualname = f"{prefix}.{node.name}" if prefix else node.name
                definitions.append(
                    Definition(
                        module=module,
                        qualname=qualname,
                        kind="class",
                        signature=format_class_signature(source, node),
                        file=member,
                        line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        doc=ast.get_docstring(node, clean=True) or "",
                        decorators=decorator_names(source, node),
                        guards=guard_evidence(source, node),
                        source_excerpt=limited_source(source, node)
                        if include_source
                        else None,
                    )
                )
                visit(node.body, qualname)
            elif not prefix and isinstance(node, (ast.Assign, ast.AnnAssign)):
                definition = assignment_definition(node)
                if definition is not None:
                    definitions.append(definition)

    visit(tree.body)
    return definitions


def resolve_import_module(
    current: str, is_package: bool, level: int, imported: str | None
) -> str:
    if level == 0:
        return imported or ""
    package_parts = current.split(".") if is_package else current.split(".")[:-1]
    keep = len(package_parts) - (level - 1)
    parts = package_parts[: max(keep, 0)]
    if imported:
        parts.extend(imported.split("."))
    return ".".join(parts)


def parse_all_names(node: ast.Assign | ast.AnnAssign) -> list[str] | None:
    target = (
        node.target
        if isinstance(node, ast.AnnAssign)
        else (node.targets[0] if len(node.targets) == 1 else None)
    )
    if not isinstance(target, ast.Name) or target.id != "__all__" or node.value is None:
        return None
    try:
        value = ast.literal_eval(node.value)
    except (TypeError, ValueError):
        return None
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        return list(value)
    return None


def collect_imports(
    module: str,
    member: str,
    is_package: bool,
    tree: ast.Module,
) -> tuple[list[ImportEdge], list[str] | None]:
    imports: list[ImportEdge] = []
    all_names: list[str] | None = None

    for node in module_statements(tree.body):
        if isinstance(node, ast.ImportFrom):
            source_module = resolve_import_module(
                module, is_package, node.level, node.module
            )
            for alias in node.names:
                if alias.name == "*":
                    imports.append(
                        ImportEdge(
                            module, "*", source_module, "*", member, node.lineno, True
                        )
                    )
                    continue
                if node.module is None:
                    imported_module = (
                        f"{source_module}.{alias.name}" if source_module else alias.name
                    )
                else:
                    imported_module = source_module
                imports.append(
                    ImportEdge(
                        module,
                        alias.asname or alias.name,
                        imported_module,
                        alias.name,
                        member,
                        node.lineno,
                    )
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    ImportEdge(
                        module,
                        alias.asname or alias.name.split(".", 1)[0],
                        alias.name,
                        alias.name.rsplit(".", 1)[-1],
                        member,
                        node.lineno,
                    )
                )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            parsed = parse_all_names(node)
            if parsed is not None:
                all_names = parsed
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "__getattr__"
        ):
            imports.extend(collect_getattr_imports(module, member, is_package, node))
    return imports, all_names


def condition_export_names(test: ast.AST) -> list[str]:
    if (
        not isinstance(test, ast.Compare)
        or len(test.ops) != 1
        or len(test.comparators) != 1
    ):
        return []
    if not isinstance(test.left, ast.Name) or test.left.id != "name":
        return []
    comparator = test.comparators[0]
    if (
        isinstance(test.ops[0], ast.Eq)
        and isinstance(comparator, ast.Constant)
        and isinstance(comparator.value, str)
    ):
        return [comparator.value]
    if isinstance(test.ops[0], ast.In) and isinstance(
        comparator, (ast.Tuple, ast.List, ast.Set)
    ):
        values = [
            item.value
            for item in comparator.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        ]
        return values if len(values) == len(comparator.elts) else []
    return []


def collect_getattr_imports(
    module: str,
    member: str,
    is_package: bool,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ImportEdge]:
    edges: list[ImportEdge] = []
    for branch in (node for node in ast.walk(function) if isinstance(node, ast.If)):
        names = condition_export_names(branch.test)
        if not names:
            continue
        module_aliases, direct_imports = getattr_aliases(
            module, is_package, branch.body
        )
        for name in names:
            if name in direct_imports:
                source_module, source_name = direct_imports[name]
                edges.append(
                    ImportEdge(
                        module, name, source_module, source_name, member, branch.lineno
                    )
                )
                continue
            for node in ast.walk(ast.Module(body=branch.body, type_ignores=[])):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in module_aliases
                    and isinstance(node.args[1], ast.Name)
                    and node.args[1].id == "name"
                ):
                    continue
                edges.append(
                    ImportEdge(
                        module,
                        name,
                        module_aliases[node.args[0].id],
                        name,
                        member,
                        branch.lineno,
                    )
                )
                break
    return edges


def parse_modules(
    archive: zipfile.ZipFile,
    package: str,
    include_internal: bool,
    include_source: bool,
) -> tuple[dict[str, ModuleInfo], list[str], list[dict[str, Any]]]:
    package_path = package.replace(".", "/").rstrip("/") + "/"
    modules: dict[str, ModuleInfo] = {}
    binaries: list[str] = []
    parse_errors: list[dict[str, Any]] = []
    for member in archive.namelist():
        if not member.startswith(package_path):
            continue
        if member.endswith((".so", ".pyd")):
            binaries.append(member)
        if not member.endswith((".py", ".pyi")):
            continue
        module, is_package = module_name(member)
        existing = modules.get(module)
        if (
            existing is not None
            and existing.file.endswith(".py")
            and member.endswith(".pyi")
        ):
            continue
        source = archive.read(member).decode("utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=member, type_comments=True)
        except SyntaxError as exc:
            parse_errors.append(
                {"file": member, "line": exc.lineno, "message": exc.msg}
            )
            continue
        definitions = collect_definitions(module, member, source, tree, include_source)
        imports, all_names = collect_imports(module, member, is_package, tree)
        modules[module] = ModuleInfo(
            module, member, is_package, definitions, imports, all_names
        )
    return modules, sorted(binaries), parse_errors


def read_metadata(archive: zipfile.ZipFile) -> dict[str, Any]:
    members = [
        name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
    ]
    if len(members) != 1:
        raise ValueError(
            f"wheel 必须且只能包含一个 .dist-info/METADATA，实际为 {len(members)}"
        )
    text = archive.read(members[0]).decode("utf-8", errors="replace")
    result: dict[str, Any] = {"metadata_member": members[0]}
    for field_name in ("Name", "Version"):
        match = re.search(
            rf"^{field_name}:\s*(.+)$", text, flags=re.MULTILINE | re.IGNORECASE
        )
        result[field_name.lower()] = match.group(1).strip() if match else None
    return result


def exported_names(
    module_name_: str,
    modules: dict[str, ModuleInfo],
    cache: dict[str, set[str]],
    visiting: set[str],
) -> set[str]:
    if module_name_ in cache:
        return cache[module_name_]
    if module_name_ in visiting or module_name_ not in modules:
        return set()
    module = modules[module_name_]
    if module.all_names is not None:
        names = set(module.all_names)
    else:
        names = {
            definition.top_name
            for definition in module.definitions
            if "." not in definition.qualname
            and not definition.top_name.startswith("_")
        }
        names.update(
            edge.export_name
            for edge in module.imports
            if edge.export_name != "*" and not edge.export_name.startswith("_")
        )
        for edge in module.imports:
            if edge.star:
                names.update(
                    exported_names(
                        edge.source_module, modules, cache, {*visiting, module_name_}
                    )
                )
    cache[module_name_] = names
    return names


def build_public_import_index(
    modules: dict[str, ModuleInfo],
) -> dict[tuple[str, str], list[str]]:
    definitions = set()
    for module in modules.values():
        for definition in module.definitions:
            if "." not in definition.qualname:
                definitions.add((definition.module, definition.top_name))
    export_cache: dict[str, set[str]] = {}

    def resolve(
        module_name_: str,
        name: str,
        seen: set[tuple[str, str]],
    ) -> tuple[str, str] | None:
        key = (module_name_, name)
        if key in seen:
            return None
        if key in definitions:
            return key
        module = modules.get(module_name_)
        if module is None:
            return None
        next_seen = {*seen, key}
        for edge in module.imports:
            if edge.star:
                names = exported_names(edge.source_module, modules, export_cache, set())
                if name not in names:
                    continue
                source_name = name
            elif edge.export_name == name:
                source_name = edge.source_name
            else:
                continue
            resolved = resolve(edge.source_module, source_name, next_seen)
            if resolved:
                return resolved
        return None

    index: dict[tuple[str, str], list[str]] = {}
    for module_name_ in modules:
        if any(part.startswith("_") for part in module_name_.split(".")[1:]):
            continue
        for name in exported_names(module_name_, modules, export_cache, set()):
            target = resolve(module_name_, name, set())
            if target:
                index.setdefault(target, []).append(f"{module_name_}.{name}")
    for paths in index.values():
        paths.sort(key=lambda value: (value.count("."), len(value), value))
    return index


def definition_imports(
    definition: Definition,
    index: dict[tuple[str, str], list[str]],
    package: str,
) -> list[str]:
    top_paths = index.get((definition.module, definition.top_name), [])
    suffix = definition.qualname[len(definition.top_name) :]
    # Class members retain their suffix under the resolved top-level alias.
    return [path + suffix for path in top_paths if path.rpartition(".")[0] == package]


def is_private(definition: Definition) -> bool:
    return any(part.startswith("_") for part in definition.qualname.split("."))


def is_internal_module(module: str, package: str) -> bool:
    relative = (
        module[len(package) + 1 :] if module.startswith(package + ".") else module
    )
    prefixes = tuple(
        prefix.rstrip("/").replace("/", ".") for prefix in INTERNAL_PREFIXES
    )
    return any(
        relative == prefix or relative.startswith(prefix + ".") for prefix in prefixes
    )


def is_internal_import_path(path: str, package: str) -> bool:
    prefixes = tuple(
        prefix.rstrip("/").replace("/", ".") for prefix in INTERNAL_PREFIXES
    )
    return any(
        path == f"{package}.{prefix}" or path.startswith(f"{package}.{prefix}.")
        for prefix in prefixes
    )


def category_for_module(module: str, package: str) -> dict[str, Any]:
    relative = (
        module[len(package) + 1 :] if module.startswith(package + ".") else module
    )
    for category in CATEGORY_RULES[:-1]:
        for prefix in category["prefixes"]:
            if relative == prefix or relative.startswith(prefix + "."):
                return category
    return CATEGORY_RULES[-1]


def api_record(
    definition: Definition,
    public_imports: list[str],
    category: dict[str, Any],
) -> dict[str, Any]:
    record = asdict(definition)
    if record["source_excerpt"] is None:
        del record["source_excerpt"]
    preferred = (
        public_imports[0]
        if public_imports
        else f"{definition.module}.{definition.qualname}"
    )
    private_member = any(
        part.startswith("_") for part in definition.qualname.split(".")[1:]
    )
    if public_imports and not private_member:
        visibility = "public"
    elif is_private(definition):
        visibility = "private"
    else:
        visibility = "module-local"
    record.update(
        {
            "symbol": preferred,
            "definition": f"{definition.module}.{definition.qualname}",
            "public_imports": public_imports,
            "visibility": visibility,
            "category": category["id"],
        }
    )
    return record


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def first_doc_line(doc: str) -> str:
    for line in doc.splitlines():
        if line.strip():
            return line.strip()
    return ""


def normalize_doc_markup(doc: str) -> str:
    """Normalize common RST/docstring markup for readable Markdown output."""
    text = html.unescape(doc).replace("\u00a0", " ")
    text = re.sub(
        r":(?:func|class|meth|mod|attr|data|exc|obj):`~?([^`]+)`",
        lambda match: f"`{match.group(1).rsplit('.', 1)[-1]}`",
        text,
    )
    text = re.sub(r"``([^`\n]+)``", r"`\1`", text)
    lines = text.splitlines()
    normalized: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if line.endswith("::"):
            line = line[:-1]
            normalized.append(line)
            index += 1
            code, index = consume_doc_code(lines, index)
            if code:
                normalized.extend(["", "```python", *code, "```"])
            continue
        normalized.append(re.sub(r"^(\s*)\*\s+", r"\1- ", line))
        index += 1
    return "\n".join(normalized).strip()


def doc_summary(doc: str) -> str:
    normalized = normalize_doc_markup(doc)
    paragraphs = re.split(r"\n\s*\n", normalized, maxsplit=1)
    return " ".join(paragraphs[0].splitlines()).strip() if paragraphs else ""


KIND_LABELS = {
    "function": "函数",
    "class": "类",
    "method": "方法",
    "attribute": "类型别名/常量",
}

VISIBILITY_LABELS = {
    "public": "公开",
    "private": "私有",
    "module-local": "模块内",
}


def chinese_bool(value: bool) -> str:
    return "是" if value else "否"


def render_category_reference(
    wheel: dict[str, Any],
    scope: dict[str, bool],
    category: dict[str, Any],
    apis: list[dict[str, Any]],
) -> str:
    kind_counts = Counter(item["kind"] for item in apis)
    lines = [
        f"# {category['title']} API 参考",
        "",
        category["description"],
        "",
        "## Wheel 信息",
        "",
        f"- 文件名：`{Path(wheel['path']).name}`",
        f"- 分发名称：`{wheel.get('distribution')}`",
        f"- 版本：`{wheel.get('version')}`",
        f"- SHA256：`{wheel['sha256']}`",
        "",
        "## 提取范围",
        "",
        f"- 本分类 API 数量：**{len(apis)}**",
        f"- 函数：{kind_counts.get('function', 0)}",
        f"- 类：{kind_counts.get('class', 0)}",
        f"- 方法：{kind_counts.get('method', 0)}",
        f"- 类型别名/常量：{kind_counts.get('attribute', 0)}",
        "- 产出范围：`cannbotdsl` 顶层公开 API 及其公开类成员",
        f"- 包含源码片段：`{chinese_bool(scope['include_source'])}`",
        f"- 分类标识：`{category['id']}`",
    ]
    lines.extend(["", "## API 简表", "", "| API | 类型 | 功能摘要 |", "|---|---|---|"])
    for item in apis:
        lines.append(
            f"| `{markdown_cell(item['symbol'])}` | {KIND_LABELS.get(item['kind'], item['kind'])} "
            f"| {markdown_cell(doc_summary(item['doc']))} |"
        )
    lines.extend(["", "## API 详情", ""])
    for item in apis:
        lines.extend(render_api_detail(item))
    lines.extend(
        [
            "",
            "## 静态解析边界",
            "",
            "动态导出、运行时 monkey patch 和无 Python stub 的二进制扩展符号可能无法恢复。",
            "公开路径来自静态 import/alias/`__all__` 链；定义和签名来自 wheel 内 `.py`/`.pyi`。",
            "",
        ]
    )
    return "\n".join(lines)


def yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def relative_display_path(path: Path) -> str:
    return os.path.relpath(path, Path.cwd())


def render_yaml_index(
    wheel: dict[str, Any],
    package: str,
    apis: list[dict[str, Any]],
    coverage: dict[str, int],
) -> str:
    lines = [
        "schema_version: 1",
        "wheel:",
        f"  distribution: {yaml_string(str(wheel.get('distribution') or ''))}",
        f"  version: {yaml_string(str(wheel.get('version') or ''))}",
        f"  sha256: {yaml_string(wheel['sha256'])}",
        f"package: {yaml_string(package)}",
        f"api_count: {len(apis)}",
        "coverage:",
        f"  definitions_parsed: {coverage['definitions_parsed']}",
        f"  parse_error_count: {coverage['parse_error_count']}",
        f"  binary_member_count: {coverage['binary_member_count']}",
        f"  static_export_unresolved_count: {coverage['static_export_unresolved_count']}",
        f"  module_namespace_export_count: {coverage['module_namespace_export_count']}",
        f"  symbol_collision_count: {coverage['symbol_collision_count']}",
        "apis:",
    ]
    for item in apis:
        lines.extend(
            [
                f"  - category: {yaml_string(item['category'])}",
                f"    symbol: {yaml_string(item['symbol'])}",
                f"    kind: {yaml_string(item['kind'])}",
                f"    definition: {yaml_string(item['definition'])}",
                f"    signature: {yaml_string(item['signature'])}",
                f"    module: {yaml_string(item['module'])}",
                f"    visibility: {yaml_string(item['visibility'])}",
            ]
        )
        if item["public_imports"]:
            lines.append("    public_imports:")
            lines.extend(
                f"      - {yaml_string(value)}" for value in item["public_imports"]
            )
        else:
            lines.append("    public_imports: []")
        if item["decorators"]:
            lines.append("    decorators:")
            lines.extend(
                f"      - {yaml_string(value)}" for value in item["decorators"]
            )
        else:
            lines.append("    decorators: []")
        lines.extend(
            [
                f"    guard_count: {len(item['guards'])}",
                f"    doc_summary: {yaml_string(doc_summary(item['doc']))}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    logging.basicConfig(format="%(message)s", level=logging.ERROR)
    locations = validated_locations(args)
    if locations is None:
        return 2
    wheel, output_dir = locations

    digest = wheel_digest(wheel)
    try:
        metadata, modules, binaries, parse_errors = parse_wheel(
            wheel, args.package, args.include_source
        )
    except (zipfile.BadZipFile, ValueError) as exc:
        logging.error(f"error: 无法解析 wheel: {exc}")
        return 2

    if not modules:
        logging.error(f"error: wheel 中未找到包 {args.package!r} 的 Python 模块")
        return 2

    public_index = build_public_import_index(modules)
    unresolved_static_exports, module_namespace_exports = unresolved_exports(
        modules, public_index, args.package
    )
    parsed_definitions = [
        definition for module in modules.values() for definition in module.definitions
    ]
    api_items, private_excluded, non_top_level_excluded = collect_api_records(
        parsed_definitions, public_index, args.package
    )
    api_items, symbol_collision_count = deduplicate_api_records(api_items)
    by_category, category_summaries = group_api_categories(api_items)
    scope = {
        "include_source": args.include_source,
    }
    wheel_info = {
        "path": str(wheel),
        "distribution": metadata.get("name"),
        "version": metadata.get("version"),
        "sha256": digest.hexdigest(),
        "metadata_member": metadata.get("metadata_member"),
    }
    coverage = {
        "definitions_parsed": len(parsed_definitions),
        "parse_error_count": len(parse_errors),
        "binary_member_count": len(binaries),
        "static_export_unresolved_count": len(unresolved_static_exports),
        "module_namespace_export_count": len(module_namespace_exports),
        "symbol_collision_count": symbol_collision_count,
    }

    catalog = ApiCatalog(api_items, by_category, category_summaries, scope)
    product_dir, yaml_path, document_paths = write_api_files(
        output_dir, wheel_info, args.package, coverage, catalog
    )

    coverage["private_definitions_excluded"] = private_excluded
    coverage["non_top_level_definitions_excluded"] = non_top_level_excluded
    emit_summary(product_dir, api_items, coverage, yaml_path, document_paths)
    return 0


def unresolved_exports(modules, public_index, package):
    resolved_paths = {path for paths in public_index.values() for path in paths}
    unresolved_static_exports: list[str] = []
    module_namespace_exports: list[str] = []
    for module_name_, module in modules.items():
        if module.all_names is None or module_name_ != package:
            continue
        for name in module.all_names:
            export_path = f"{module_name_}.{name}"
            if export_path in resolved_paths:
                continue
            module_candidates = {export_path}
            for edge in module.imports:
                if not edge.star and edge.export_name == name:
                    module_candidates.add(f"{edge.source_module}.{edge.source_name}")
                    module_candidates.add(edge.source_module)
            if any(candidate in modules for candidate in module_candidates):
                module_namespace_exports.append(export_path)
            else:
                unresolved_static_exports.append(export_path)
    return unresolved_static_exports, module_namespace_exports


def collect_api_records(parsed_definitions, public_index, package):
    api_items: list[dict[str, Any]] = []
    private_excluded = 0
    non_top_level_excluded = 0
    for definition in parsed_definitions:
        imports = definition_imports(definition, public_index, package)
        if not imports:
            non_top_level_excluded += 1
            continue
        private_member = any(
            part.startswith("_") for part in definition.qualname.split(".")[1:]
        )
        if private_member or (is_private(definition) and not imports):
            private_excluded += 1
            continue
        category = category_for_module(definition.module, package)
        api_items.append(api_record(definition, imports, category))

    return api_items, private_excluded, non_top_level_excluded


def deduplicate_api_records(api_items):
    kind_priority = {"class": 0, "function": 1, "method": 2, "attribute": 3}
    deduplicated: dict[str, dict[str, Any]] = {}
    symbol_collision_count = 0
    for item in api_items:
        existing = deduplicated.get(item["symbol"])
        if existing is None:
            deduplicated[item["symbol"]] = item
            continue
        symbol_collision_count += 1
        chosen = min(
            (existing, item),
            key=lambda value: (
                kind_priority.get(value["kind"], 99),
                value["definition"].count("."),
                value["definition"],
            ),
        )
        chosen["public_imports"] = sorted(
            set(existing["public_imports"]) | set(item["public_imports"]),
            key=lambda value: (value.count("."), len(value), value),
        )
        deduplicated[item["symbol"]] = chosen
    api_items = list(deduplicated.values())

    return api_items, symbol_collision_count


def group_api_categories(api_items):
    category_order = {
        category["id"]: index for index, category in enumerate(CATEGORY_RULES)
    }
    api_items.sort(
        key=lambda item: (
            category_order[item["category"]],
            item["symbol"],
            item["definition"],
        )
    )
    by_category: dict[str, list[dict[str, Any]]] = {
        category["id"]: [] for category in CATEGORY_RULES
    }
    for item in api_items:
        by_category[item["category"]].append(item)
    category_summaries = [
        {
            "id": category["id"],
            "title": category["title"],
            "description": category["description"],
            "prefixes": list(category["prefixes"]),
            "count": len(by_category[category["id"]]),
        }
        for category in CATEGORY_RULES
        if by_category[category["id"]]
    ]
    return by_category, category_summaries


def module_statements(body: list[ast.stmt]):
    for statement in body:
        yield statement
        if isinstance(statement, ast.If):
            yield from module_statements(statement.body)
            yield from module_statements(statement.orelse)
        elif isinstance(
            statement, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)
        ):
            yield from module_statements(statement.body)
            yield from module_statements(getattr(statement, "orelse", []))
        elif isinstance(statement, ast.Try):
            yield from module_statements(statement.body)
            for handler in statement.handlers:
                yield from module_statements(handler.body)
            yield from module_statements(statement.orelse)
            yield from module_statements(statement.finalbody)


def getattr_aliases(module, is_package, body):
    module_aliases: dict[str, str] = {}
    direct_imports: dict[str, tuple[str, str]] = {}
    for node in body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_aliases[alias.asname or alias.name.split(".", 1)[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            source_module = resolve_import_module(
                module, is_package, node.level, node.module
            )
            for alias in node.names:
                local_name = alias.asname or alias.name
                direct_imports[local_name] = (source_module, alias.name)
                module_aliases[local_name] = f"{source_module}.{alias.name}"
    return module_aliases, direct_imports


def consume_doc_code(lines, index):
    while index < len(lines) and not lines[index].strip():
        index += 1
    code: list[str] = []
    while index < len(lines):
        candidate = lines[index]
        if candidate.startswith(("    ", "\t")):
            code.append(
                candidate[4:] if candidate.startswith("    ") else candidate[1:]
            )
            index += 1
            continue
        break
    return code, index


def render_api_detail(item):
    lines = []
    lines.extend(
        [
            f"### `{item['symbol']}`",
            "",
            f"- 类型：`{KIND_LABELS.get(item['kind'], item['kind'])}`",
            f"- 定义符号：`{item['definition']}`",
            f"- 可见性：`{VISIBILITY_LABELS.get(item['visibility'], item['visibility'])}`",
        ]
    )
    if item["public_imports"]:
        lines.append(
            "- 公开导入路径："
            + ", ".join(f"`{path}`" for path in item["public_imports"])
        )
    if item["decorators"]:
        lines.append(
            "- 装饰器：" + ", ".join(f"`@{value}`" for value in item["decorators"])
        )
    lines.extend(["", "```python", item["signature"], "```", ""])
    if item["doc"]:
        summary = doc_summary(item["doc"])
        if summary:
            lines.extend(["#### 功能摘要", "", summary, ""])
    if item["guards"]:
        lines.extend(["#### 参数校验与异常", ""])
        for guard in item["guards"]:
            source = " ".join(str(guard["source"]).replace("`", "\\`").split())
            guard_label = {"assert": "断言", "raise": "抛出异常"}.get(
                guard["kind"], guard["kind"]
            )
            lines.append(f"- {guard_label}：`{source}`")
        lines.append("")
    if "source_excerpt" in item:
        lines.extend(
            ["#### 源码片段", "", "```python", item["source_excerpt"], "```", ""]
        )
    return lines


@dataclass
class ApiCatalog:
    items: list
    by_category: dict
    summaries: list
    scope: dict


def write_api_files(output_dir, wheel_info, package, coverage, catalog):
    api_items, by_category, category_summaries = (
        catalog.items,
        catalog.by_category,
        catalog.summaries,
    )
    scope = catalog.scope
    product_dir = output_dir / "cannbot-dsl-docs-search"
    api_dir = product_dir / "api-info"
    api_dir.mkdir(parents=True, exist_ok=True)
    expected_documents = {f"{summary['id']}.md" for summary in category_summaries}
    for stale in api_dir.glob("*.md"):
        if stale.name not in expected_documents:
            stale.unlink()

    yaml_path = product_dir / "api_index.yaml"
    yaml_path.write_text(
        render_yaml_index(wheel_info, package, api_items, coverage), encoding="utf-8"
    )
    category_by_id = {category["id"]: category for category in CATEGORY_RULES}
    document_paths = []
    for summary in category_summaries:
        category = category_by_id[summary["id"]]
        document_path = api_dir / f"{category['id']}.md"
        document_path.write_text(
            render_category_reference(
                wheel_info, scope, category, by_category[category["id"]]
            ),
            encoding="utf-8",
        )
        document_paths.append(str(document_path))

    return product_dir, yaml_path, document_paths


def validated_locations(args):
    wheel = args.wheel.expanduser()
    wheel = wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        logging.error(f"error: wheel 不存在或扩展名不是 .whl: {wheel}")
        return None
    if not args.package or any(
        not part.isidentifier() for part in args.package.split(".")
    ):
        logging.error(f"error: 无效包名: {args.package!r}")
        return None
    output_dir = args.output_dir.expanduser()
    output_dir = output_dir.resolve()
    if output_dir == Path(output_dir.anchor):
        logging.error("error: --output-dir 不能是文件系统根目录")
        return None

    return wheel, output_dir


def wheel_digest(wheel):
    digest = hashlib.sha256()
    with wheel.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest


def emit_summary(product_dir, api_items, coverage, yaml_path, document_paths):
    sys.stdout.write(
        json.dumps(
            {
                "status": "ok",
                "output_dir": relative_display_path(product_dir),
                "api_count": len(api_items),
                "category_count": len(document_paths),
                "definitions_parsed": coverage["definitions_parsed"],
                "private_definitions_excluded": coverage[
                    "private_definitions_excluded"
                ],
                "non_top_level_definitions_excluded": coverage[
                    "non_top_level_definitions_excluded"
                ],
                "binary_member_count": coverage["binary_member_count"],
                "parse_error_count": coverage["parse_error_count"],
                "static_export_unresolved_count": coverage[
                    "static_export_unresolved_count"
                ],
                "module_namespace_export_count": coverage[
                    "module_namespace_export_count"
                ],
                "symbol_collision_count": coverage["symbol_collision_count"],
                "api_documents": [
                    relative_display_path(Path(path)) for path in document_paths
                ],
                "yaml_index": relative_display_path(yaml_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def parse_wheel(wheel, package, include_source):
    with zipfile.ZipFile(wheel) as archive:
        metadata = read_metadata(archive)
        modules, binaries, parse_errors = parse_modules(
            archive,
            package,
            False,
            include_source,
        )
    return metadata, modules, binaries, parse_errors


if __name__ == "__main__":
    raise SystemExit(main())
