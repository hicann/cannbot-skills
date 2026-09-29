# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Manage the read-only CATLASS C++ OKF v0.2 knowledge bundle."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import stat
import sys
from pathlib import Path
from typing import Any

import yaml


OKF_VERSION = "0.2"
VOCAB_VERSION = "1"
PLUGIN_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_KNOWLEDGE = (PLUGIN_ROOT / "knowledge").resolve()
PROJECT_KNOWLEDGE = Path(".catlass-cpp") / "knowledge"
BUSINESS_PARTITIONS = {"catlass", "workflow", "operator"}
WORKFLOW_CONSUMERS = {
    "catlass-cpp-interface",
    "catlass-cpp-reference",
    "catlass-cpp-design",
    "catlass-cpp-develop",
    "catlass-cpp-test",
}
REQUIRED_FIELDS = {
    "type",
    "title",
    "description",
    "tags",
    "status",
    "generated",
    "verified",
    "sources",
}
COMMON_HEADINGS = (
    "# 接口与概念",
    "# 用法",
    "# 代码模式",
    "# 约束",
    "# 失败表现",
    "# 验证方法",
)
OPERATOR_HEADINGS = (
    "## 算子算法",
    "## 分核策略与基本块切分",
    "## 数据路径与存储层级",
    "## 流水排布、同步关系与数值精度",
)
SAFE_PATH = re.compile(r"^[A-Za-z0-9._/-]+$")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _reject_symlink_chain(path: Path, stop: Path) -> None:
    current = path.absolute()
    stop = stop.absolute()
    while True:
        if current.exists() and stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError(f"path must not traverse a symlink: {current}")
        if current == stop or current == current.parent:
            break
        current = current.parent


def _safe_relative(value: str) -> Path:
    if not value or not SAFE_PATH.fullmatch(value):
        raise ValueError("path must be a safe relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("path must be a safe relative path")
    return path


def _frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError(f"{path}: unclosed YAML frontmatter")
    try:
        data = yaml.safe_load(text[4:end])
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML frontmatter: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: frontmatter must be a mapping")
    return data, text[end + 5 :]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: YAML root must be a mapping")
    return value


def _concept_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for partition in sorted(BUSINESS_PARTITIONS):
        base = root / partition
        if base.is_dir():
            paths.extend(
                path
                for path in base.rglob("*.md")
                if path.name != "index.md" and path.is_file()
            )
    return sorted(paths)


def initialize(bundle: Path, target: Path) -> list[str]:
    bundle = bundle.resolve()
    target = target.resolve()
    if bundle != PLUGIN_KNOWLEDGE:
        raise ValueError("bundle must be the built-in CATLASS C++ knowledge directory")
    if _within(target, bundle) or _within(bundle, target):
        raise ValueError("built-in and project knowledge directories must not overlap")
    _reject_symlink_chain(target, target.anchor and Path(target.anchor) or target.parent)
    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source in sorted(bundle.rglob("*")):
        relative = source.relative_to(bundle)
        destination = target / relative
        if source.is_dir():
            destination.mkdir(exist_ok=True)
            continue
        if not source.is_file():
            raise ValueError(f"bundle contains a non-regular entry: {relative}")
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append(relative.as_posix())
    return copied


def _validate_root(root: Path, errors: list[str]) -> None:
    if not root.is_dir():
        errors.append(f"knowledge root not found: {root}")
        return
    try:
        metadata, _ = _frontmatter(root / "index.md")
        if metadata != {"okf_version": OKF_VERSION}:
            errors.append("root index frontmatter must contain only okf_version: 0.2")
    except (OSError, ValueError) as exc:
        errors.append(str(exc))

    allowed = BUSINESS_PARTITIONS | {
        "index.md",
        "query-vocabulary.yaml",
        "bundle-profile.yaml",
    }
    actual = {path.name for path in root.iterdir()}
    extra = sorted(actual - allowed)
    if extra:
        errors.append(f"unexpected root entries: {', '.join(extra)}")
    missing = sorted(allowed - actual)
    if missing:
        errors.append(f"missing root entries: {', '.join(missing)}")

    try:
        profile = _load_yaml(root / "bundle-profile.yaml")
        if profile.get("okf_version") != OKF_VERSION:
            errors.append("bundle profile okf_version must be 0.2")
        if set(profile.get("business_partitions", [])) != BUSINESS_PARTITIONS:
            errors.append("bundle profile business_partitions must be catlass, workflow, and operator")
        if profile.get("runtime_root") != ".catlass-cpp/knowledge":
            errors.append("bundle profile runtime_root must be .catlass-cpp/knowledge")
    except ValueError as exc:
        errors.append(str(exc))


def _validate_vocabulary(root: Path, errors: list[str]) -> dict[str, Any]:
    try:
        vocabulary = _load_yaml(root / "query-vocabulary.yaml")
    except ValueError as exc:
        errors.append(str(exc))
        return {}
    if str(vocabulary.get("version")) != VOCAB_VERSION:
        errors.append("query vocabulary version must be 1")
    workflow_topics = vocabulary.get("workflow_topics")
    if not isinstance(workflow_topics, dict) or not workflow_topics:
        errors.append("workflow_topics must be a non-empty mapping")
    families = vocabulary.get("operator_families")
    if not isinstance(families, dict) or not families:
        errors.append("operator_families must be a non-empty mapping")
        return vocabulary
    seen: dict[str, str] = {}
    for family, entry in families.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("aliases"), list):
            errors.append(f"family {family} must declare aliases")
            continue
        if not (root / "operator" / family).is_dir():
            errors.append(f"registered family directory missing: {family}")
        for alias in [family, *entry["aliases"]]:
            normalized = str(alias).lower().replace("_", "-")
            previous = seen.get(normalized)
            if previous and previous != family:
                errors.append(f"family alias conflict: {alias}")
            seen[normalized] = family
    return vocabulary


def _validate_concept(path: Path, root: Path, errors: list[str]) -> None:
    relative = path.relative_to(root).as_posix()
    try:
        data, body = _frontmatter(path)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        return
    missing = sorted(REQUIRED_FIELDS - set(data))
    if missing:
        errors.append(f"{relative}: missing fields: {', '.join(missing)}")
    concept_type = data.get("type")
    if concept_type not in BUSINESS_PARTITIONS:
        errors.append(f"{relative}: type must be catlass, workflow, or operator")
    if relative.split("/", 1)[0] != concept_type:
        errors.append(f"{relative}: type does not match partition")
    if data.get("status") not in {"draft", "stable", "deprecated"}:
        errors.append(f"{relative}: invalid status")
    for field in ("title", "description"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            errors.append(f"{relative}: {field} must be non-empty")
    if not isinstance(data.get("tags"), list) or not data["tags"]:
        errors.append(f"{relative}: tags must be a non-empty list")
    if not isinstance(data.get("generated"), dict):
        errors.append(f"{relative}: generated must be a mapping")
    if not isinstance(data.get("verified"), list):
        errors.append(f"{relative}: verified must be a list")
    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append(f"{relative}: sources must be a non-empty list")
    else:
        ids: set[str] = set()
        for source in sources:
            if not isinstance(source, dict) or not {
                "id",
                "resource",
                "title",
                "kind",
            }.issubset(source):
                errors.append(f"{relative}: invalid source entry")
                continue
            source_id = str(source["id"])
            if source_id in ids:
                errors.append(f"{relative}: duplicate source id {source_id}")
            ids.add(source_id)
            if f"[^{source_id}]:" not in body:
                errors.append(f"{relative}: missing source footnote {source_id}")
    for heading in COMMON_HEADINGS:
        if not re.search(rf"(?m)^{re.escape(heading)}\s*$", body):
            errors.append(f"{relative}: missing heading {heading}")
    if concept_type == "operator":
        families = data.get("operator_families")
        if not isinstance(families, list) or not families:
            errors.append(f"{relative}: operator_families must be non-empty")
        if not isinstance(data.get("architectures"), list) or not data["architectures"]:
            errors.append(f"{relative}: architectures must be non-empty")
        for heading in OPERATOR_HEADINGS:
            if not re.search(rf"(?m)^{re.escape(heading)}\s*$", body):
                errors.append(f"{relative}: missing heading {heading}")
    elif concept_type == "workflow":
        consumers = data.get("consumers")
        if not isinstance(consumers, list) or not consumers:
            errors.append(f"{relative}: consumers must be a non-empty list")
        elif any(item not in WORKFLOW_CONSUMERS for item in consumers):
            errors.append(f"{relative}: consumers must name existing stage skills")
        if Path(relative).parent != Path("workflow"):
            errors.append(f"{relative}: workflow concepts must be direct children of workflow/")


def _markdown_links(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return set(re.findall(r"\[[^\]]+\]\(([^)#]+)", path.read_text(encoding="utf-8")))


def _validate_indexes(root: Path, concepts: list[Path], errors: list[str]) -> None:
    root_links = _markdown_links(root / "index.md")
    for required in ("catlass/index.md", "workflow/index.md", "operator/index.md"):
        if required not in root_links:
            errors.append(f"root index missing link: {required}")
    catlass_links = _markdown_links(root / "catlass" / "index.md")
    for concept in [p for p in concepts if p.parent == root / "catlass"]:
        if concept.name not in catlass_links:
            errors.append(f"catlass index missing link: {concept.name}")
    workflow_links = _markdown_links(root / "workflow" / "index.md")
    for concept in [p for p in concepts if p.parent == root / "workflow"]:
        if concept.name not in workflow_links:
            errors.append(f"workflow index missing link: {concept.name}")
    operator_links = _markdown_links(root / "operator" / "index.md")
    for family_dir in sorted(path for path in (root / "operator").iterdir() if path.is_dir()):
        required = f"{family_dir.name}/index.md"
        if required not in operator_links:
            errors.append(f"operator index missing link: {required}")
        family_links = _markdown_links(family_dir / "index.md")
        for concept in [p for p in concepts if p.parent == family_dir]:
            if concept.name not in family_links:
                errors.append(f"{family_dir.name} index missing link: {concept.name}")


def validate_bundle(root: Path) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    _validate_root(root, errors)
    if errors and not root.is_dir():
        return {"status": "failed", "okf_version": OKF_VERSION, "errors": errors}
    _validate_vocabulary(root, errors)
    concepts = _concept_paths(root)
    for concept in concepts:
        _validate_concept(concept, root, errors)
    _validate_indexes(root, concepts, errors)
    return {
        "status": "failed" if errors else "passed",
        "okf_version": OKF_VERSION,
        "count": len(concepts),
        "errors": errors,
    }


def _canonical_family(vocabulary: dict[str, Any], value: str) -> str:
    normalized = value.lower().replace("_", "-")
    for family, entry in vocabulary.get("operator_families", {}).items():
        aliases = [family, *entry.get("aliases", [])]
        if normalized in {str(alias).lower().replace("_", "-") for alias in aliases}:
            return family
    return normalized


def query_bundle(
    root: Path,
    concept_type: str | None,
    tags: list[str],
    family: str | None,
    arch: str | None,
    text: str | None,
    compact: bool,
) -> dict[str, Any]:
    report = validate_bundle(root)
    if report["status"] != "passed":
        raise ValueError("knowledge bundle validation failed: " + "; ".join(report["errors"]))
    vocabulary = _load_yaml(root / "query-vocabulary.yaml")
    wanted_family = _canonical_family(vocabulary, family) if family else None
    terms = [term.lower() for term in (text or "").split() if term]
    results: list[dict[str, Any]] = []
    for path in _concept_paths(root):
        data, body = _frontmatter(path)
        if concept_type and data.get("type") != concept_type:
            continue
        concept_tags = [str(tag).lower() for tag in data.get("tags", [])]
        if tags and any(tag.lower() not in concept_tags for tag in tags):
            continue
        families = [str(item) for item in data.get("operator_families", [])]
        if wanted_family and wanted_family not in families:
            continue
        architectures = [str(item).lower() for item in data.get("architectures", [])]
        if arch and arch.lower() not in architectures:
            continue
        haystack = " ".join(
            [
                str(data.get("title", "")),
                str(data.get("description", "")),
                " ".join(concept_tags),
                " ".join(families),
                path.relative_to(root).as_posix(),
                body,
            ]
        ).lower()
        if terms and not all(term in haystack for term in terms):
            continue
        item = {
            "path": path.relative_to(root).as_posix(),
            "type": data["type"],
            "title": data["title"],
            "description": data["description"],
            "status": data["status"],
        }
        if not compact:
            item["tags"] = data["tags"]
            item["operator_families"] = data.get("operator_families", [])
            item["consumers"] = data.get("consumers", [])
        results.append(item)
    return {"status": "passed", "okf_version": OKF_VERSION, "count": len(results), "results": results}


def get_concept(root: Path, value: str) -> dict[str, Any]:
    relative = _safe_relative(value)
    root = root.resolve()
    path = root / relative
    _reject_symlink_chain(path, root)
    if not _within(path, root) or not path.is_file() or path.suffix != ".md":
        raise ValueError("concept path must resolve to a Markdown file inside knowledge root")
    if path.name == "index.md":
        raise ValueError("get only accepts concept files, not indexes")
    return {
        "status": "passed",
        "okf_version": OKF_VERSION,
        "path": relative.as_posix(),
        "content": path.read_text(encoding="utf-8"),
    }


def _write_index(path: Path, title: str, links: list[tuple[str, str]]) -> None:
    body = [f"# {title}", ""]
    body.extend(f"- [{label}]({target})" for label, target in links)
    body.append("")
    path.write_text("\n".join(body), encoding="utf-8")


def reindex(root: Path) -> dict[str, Any]:
    root = root.resolve()
    concepts = _concept_paths(root)
    catlass_links = []
    for path in concepts:
        if path.parent == root / "catlass":
            data, _ = _frontmatter(path)
            catlass_links.append((data["title"], path.name))
    _write_index(root / "catlass" / "index.md", "CATLASS C++ API", sorted(catlass_links))

    workflow_links = []
    for path in concepts:
        if path.parent == root / "workflow":
            data, _ = _frontmatter(path)
            workflow_links.append((data["title"], path.name))
    _write_index(root / "workflow" / "index.md", "Workflow Knowledge", sorted(workflow_links))

    family_links = []
    for family_dir in sorted(path for path in (root / "operator").iterdir() if path.is_dir()):
        links = []
        for path in [item for item in concepts if item.parent == family_dir]:
            data, _ = _frontmatter(path)
            links.append((data["title"], path.name))
        _write_index(family_dir / "index.md", family_dir.name, sorted(links))
        family_links.append((family_dir.name, f"{family_dir.name}/index.md"))
    _write_index(root / "operator" / "index.md", "Operator Families", family_links)
    return {"status": "passed", "okf_version": OKF_VERSION, "count": len(concepts)}


def _root(args: argparse.Namespace) -> Path:
    if getattr(args, "builtin", False):
        return PLUGIN_KNOWLEDGE
    return Path(args.project_root).resolve() / PROJECT_KNOWLEDGE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize_parser = subparsers.add_parser("initialize")
    initialize_parser.add_argument("--project-root", required=True)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--project-root", required=True)
    query_parser.add_argument("--type", choices=sorted(BUSINESS_PARTITIONS))
    query_parser.add_argument("--tag", action="append", default=[])
    query_parser.add_argument("--family")
    query_parser.add_argument("--arch")
    query_parser.add_argument("--text")
    query_parser.add_argument("--compact", action="store_true")

    get_parser = subparsers.add_parser("get")
    get_parser.add_argument("--project-root", required=True)
    get_parser.add_argument("--path", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--project-root")
    validate_parser.add_argument("--builtin", action="store_true")

    reindex_parser = subparsers.add_parser("reindex")
    reindex_parser.add_argument("--project-root", required=True)

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--project-root", required=True)
    record_parser.add_argument("--entry")

    args = parser.parse_args()
    try:
        if args.command == "initialize":
            validation = validate_bundle(PLUGIN_KNOWLEDGE)
            if validation["status"] != "passed":
                raise ValueError("built-in bundle is invalid: " + "; ".join(validation["errors"]))
            project = Path(args.project_root).resolve()
            result = {
                "status": "passed",
                "okf_version": OKF_VERSION,
                "copied": initialize(PLUGIN_KNOWLEDGE, project / PROJECT_KNOWLEDGE),
            }
        elif args.command == "query":
            result = query_bundle(
                _root(args), args.type, args.tag, args.family, args.arch, args.text, args.compact
            )
        elif args.command == "get":
            result = get_concept(_root(args), args.path)
        elif args.command == "validate":
            if args.builtin == bool(args.project_root):
                raise ValueError("validate requires exactly one of --builtin or --project-root")
            result = validate_bundle(_root(args))
        elif args.command == "reindex":
            result = reindex(_root(args))
        else:
            result = {
                "status": "unsupported",
                "okf_version": OKF_VERSION,
                "reason": "CATLASS C++ profile has no learned partition; edit the static bundle through review",
            }
            print(_json(result))
            return 2
    except (OSError, ValueError) as exc:
        print(_json({"status": "failed", "okf_version": OKF_VERSION, "errors": [str(exc)]}))
        return 1
    print(_json(result))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
