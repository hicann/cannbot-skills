#!/usr/bin/env python3
"""Initialize an OKF bundle and append evidence-backed learned concepts."""

import argparse
import datetime
import difflib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from urllib.parse import urlparse
from pathlib import Path

import yaml

try:
    import fcntl
except ImportError:  # pragma: no cover - POSIX is required for recording
    fcntl = None


OKF_VERSION = "0.2"
SAFE_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_ARCHES = {"c310"}
PROVEN_STATUSES = {"有效", "无效", "条件有效"}
EVIDENCE_KINDS = {"test", "profiling"}
REQUIRED_FIELDS = {
    "operator_family",
    "topic",
    "arch",
    "versions",
    "applicability",
    "hypothesis",
    "actual_change",
    "correctness_before",
    "correctness_after",
    "performance_before",
    "performance_after",
    "profiling_observation",
    "result",
    "status",
    "evidence",
    "kernel_sha256",
}
APPLICABILITY_FIELDS = {"shape", "dtype", "layout", "repository_integration"}
MEASUREMENT_FIELDS = {"status", "summary", "reason"}
PROFILING_FIELDS = {"status", "observation", "reason"}
PLUGIN_KNOWLEDGE = (Path(__file__).resolve().parents[3] / "knowledge").resolve()
PROJECT_KNOWLEDGE = Path(".catlass-dsl") / "knowledge"
TEMPLATE = (Path(__file__).resolve().parents[1] / "templates" / "learned-entry.md").resolve()
OKF_STATUSES = {"draft", "stable", "deprecated"}
ACTOR = re.compile(
    r"^(?:human:[A-Za-z0-9._-]+|process:[A-Za-z0-9._-]+|"
    r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+)$"
)
SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
QUERY_VOCABULARY = "query-vocabulary.yaml"
QUERY_VOCABULARY_VERSION = "1"
QUERY_FIELD_WEIGHTS = {
    "title": 12,
    "operator_families": 10,
    "tags": 8,
    "description": 6,
    "path": 4,
    "body": 2,
    "source_titles": 1,
}
QUERY_STATUS_ORDER = {"stable": 0, "draft": 1, "deprecated": 2}
QUERY_FIELD_ORDER = tuple(QUERY_FIELD_WEIGHTS)
POSITIVE_INT = re.compile(r"^[1-9][0-9]*$")
QUERY_SNIPPET_LIMIT = 3
QUERY_SNIPPET_CHARS = 240


def _is_within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _reject_symlink_chain(path, stop=None):
    current = Path(path).absolute()
    stop = Path(stop).absolute() if stop is not None else None
    while True:
        if current.exists() and stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError("路径不得经过符号链接：{}".format(current))
        if current == current.parent or current == stop:
            break
        current = current.parent


def _safe_relative_path(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError("{} 必须是非空项目相对路径".format(label))
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("{} 必须是安全项目相对路径".format(label))
    return path


def _validate_measurement(field, value, statuses):
    if not isinstance(value, dict) or set(value) != MEASUREMENT_FIELDS:
        return ["{} 必须包含 status、summary 和 reason".format(field)]
    if value.get("status") not in statuses:
        return ["{} status 非法".format(field)]
    if not isinstance(value.get("summary"), str) or not isinstance(value.get("reason"), str):
        return ["{} summary/reason 必须是字符串".format(field)]
    if value["status"] == "not_run" and not value["reason"].strip():
        return ["{} 为 not_run 时必须记录 reason".format(field)]
    if value["status"] != "not_run" and not value["summary"].strip():
        return ["{} 已运行时必须记录 summary".format(field)]
    return []


def validate_entry(entry, project_root):
    project = Path(project_root).resolve()
    errors = []
    if not isinstance(entry, dict):
        return ["entry 必须是对象"]
    if set(entry) != REQUIRED_FIELDS:
        errors.append("entry 字段集合非法")

    for field in ("operator_family", "topic"):
        value = entry.get(field)
        if not isinstance(value, str) or not SAFE_SLUG.fullmatch(value):
            errors.append("{} 必须是安全 slug".format(field))

    arches = entry.get("arch")
    if (
        not isinstance(arches, list)
        or not arches
        or any(not isinstance(arch, str) or arch not in SUPPORTED_ARCHES for arch in arches)
        or len(set(arches)) != len(arches)
    ):
        errors.append("arch 必须是只包含 c310 字符串的非空列表")

    versions = entry.get("versions")
    if (
        not isinstance(versions, dict)
        or set(versions) != {"catlass", "cann"}
        or any(not isinstance(versions.get(key), str) or not versions[key].strip() for key in versions)
    ):
        errors.append("versions 必须记录 CATLASS DSL 和 CANN 版本条件")

    applicability = entry.get("applicability")
    if not isinstance(applicability, dict) or set(applicability) != APPLICABILITY_FIELDS:
        errors.append("applicability 必须包含 shape、dtype、layout 和 repository_integration")
    elif any(not isinstance(applicability[key], str) or not applicability[key].strip() for key in APPLICABILITY_FIELDS):
        errors.append("applicability 字段必须是非空字符串")

    for field in ("hypothesis", "actual_change", "result"):
        if not isinstance(entry.get(field), str) or not entry[field].strip():
            errors.append("{} 必须是非空字符串".format(field))

    errors.extend(_validate_measurement("correctness_before", entry.get("correctness_before"), {"passed", "failed", "not_run"}))
    errors.extend(_validate_measurement("correctness_after", entry.get("correctness_after"), {"passed", "failed", "not_run"}))
    errors.extend(_validate_measurement("performance_before", entry.get("performance_before"), {"passed", "not_run"}))
    errors.extend(_validate_measurement("performance_after", entry.get("performance_after"), {"passed", "not_run"}))

    profiling = entry.get("profiling_observation")
    if not isinstance(profiling, dict) or set(profiling) != PROFILING_FIELDS:
        errors.append("profiling_observation 必须包含 status、observation 和 reason")
    elif profiling.get("status") not in {"passed", "not_run"}:
        errors.append("profiling_observation status 非法")
    elif not isinstance(profiling.get("observation"), str) or not isinstance(profiling.get("reason"), str):
        errors.append("profiling_observation observation/reason 必须是字符串")
    elif profiling["status"] == "passed" and not profiling["observation"].strip():
        errors.append("profiling_observation passed 时必须记录 observation")
    elif profiling["status"] == "not_run" and not profiling["reason"].strip():
        errors.append("profiling_observation not_run 时必须记录 reason")

    if entry.get("status") not in PROVEN_STATUSES:
        errors.append("status 只能是有效、无效或条件有效")
    if isinstance(entry.get("correctness_after"), dict) and entry["correctness_after"].get("status") != "passed":
        errors.append("correctness_after 必须为 passed")

    evidence = entry.get("evidence")
    evidence_kinds = set()
    if not isinstance(evidence, list) or not evidence:
        errors.append("evidence 必须是非空列表")
    else:
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {"kind", "path"}:
                errors.append("每条 evidence 只能包含 kind 和 path")
                continue
            if item.get("kind") not in EVIDENCE_KINDS:
                errors.append("evidence kind 只能是 test 或 profiling")
                continue
            evidence_kinds.add(item["kind"])
            try:
                relative = _safe_relative_path(item.get("path"), "evidence.path")
                candidate = project / relative
                _reject_symlink_chain(candidate, project)
                if not candidate.is_file():
                    errors.append("evidence 必须指向项目内普通文件：{}".format(relative))
            except ValueError as exc:
                errors.append(str(exc))

    if isinstance(profiling, dict):
        if profiling.get("status") == "passed" and "profiling" not in evidence_kinds:
            errors.append("profiling passed 必须链接 profiling evidence")
        if "profiling" in evidence_kinds and profiling.get("status") != "passed":
            errors.append("profiling evidence 与 profiling_observation 状态不一致")

    if not isinstance(entry.get("kernel_sha256"), str) or not SHA256.fullmatch(entry["kernel_sha256"]):
        errors.append("kernel_sha256 必须是 64 位小写 SHA-256")
    return list(dict.fromkeys(errors))


def initialize(bundle, target):
    bundle = Path(bundle).resolve()
    target = Path(target).resolve()
    if bundle != PLUGIN_KNOWLEDGE:
        raise ValueError("bundle 必须是插件内置 knowledge OKF bundle")
    if _is_within(target, bundle) or _is_within(bundle, target):
        raise ValueError("目标 knowledge 与插件内置 bundle 不得重叠")
    _reject_symlink_chain(target)
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in sorted(bundle.rglob("*")):
        relative = source.relative_to(bundle)
        destination = target / relative
        if source.is_dir():
            destination.mkdir(exist_ok=True)
            continue
        if not source.is_file():
            raise ValueError("内置 bundle 只允许普通文件和目录：{}".format(relative))
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(source), str(destination))
        copied.append(str(relative))
    return copied


def _yaml(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _measurement(value):
    return "{}; {}; {}".format(value["status"], value["summary"], value["reason"]).strip("; ")


def _render(entry, date_value):
    template = TEMPLATE.read_text(encoding="utf-8")
    sources = []
    evidence_links = []
    for number, item in enumerate(entry["evidence"], 1):
        resource = "project-evidence:{}?kernel-sha256={}".format(item["path"], entry["kernel_sha256"])
        sources.append(
            {
                "id": "evidence-{}".format(number),
                "resource": resource,
                "title": "{} evidence: {}".format(item["kind"], item["path"]),
                "kind": item["kind"],
            }
        )
        evidence_links.append("- [{}](../../../{})".format(item["kind"], item["path"]))

    replacements = {
        "{{title_json}}": _yaml("{}: {}".format(entry["operator_family"], entry["topic"])),
        "{{description_json}}": _yaml(entry["result"]),
        "{{tags_json}}": _yaml(["catlass-dsl", "learned", entry["operator_family"], entry["status"]]),
        "{{generated_json}}": _yaml({"by": "process:catlass-dsl-knowledge", "at": "{}T00:00:00Z".format(date_value)}),
        "{{verified_json}}": _yaml([{"by": "process:catlass-dsl-evidence-gate", "at": "{}T00:00:00Z".format(date_value)}]),
        "{{sources_json}}": _yaml(sources),
        "{{operator_family_json}}": _yaml(entry["operator_family"]),
        "{{topic_json}}": _yaml(entry["topic"]),
        "{{arch_json}}": _yaml(entry["arch"]),
        "{{versions_json}}": _yaml(entry["versions"]),
        "{{applicability_json}}": _yaml(entry["applicability"]),
        "{{result_status_json}}": _yaml(entry["status"]),
        "{{kernel_sha256_json}}": _yaml(entry["kernel_sha256"]),
        "{{operator_family}}": entry["operator_family"],
        "{{topic}}": entry["topic"],
        "{{arch}}": ", ".join(entry["arch"]),
        "{{catlass_dsl_version}}": entry["versions"]["catlass"],
        "{{cann_version}}": entry["versions"]["cann"],
        "{{shape}}": entry["applicability"]["shape"],
        "{{dtype}}": entry["applicability"]["dtype"],
        "{{layout}}": entry["applicability"]["layout"],
        "{{repository_integration}}": entry["applicability"]["repository_integration"],
        "{{hypothesis}}": entry["hypothesis"],
        "{{actual_change}}": entry["actual_change"],
        "{{correctness_before}}": _measurement(entry["correctness_before"]),
        "{{correctness_after}}": _measurement(entry["correctness_after"]),
        "{{performance_before}}": _measurement(entry["performance_before"]),
        "{{performance_after}}": _measurement(entry["performance_after"]),
        "{{profiling_observation}}": "{}; {}; {}".format(
            entry["profiling_observation"]["status"],
            entry["profiling_observation"]["observation"],
            entry["profiling_observation"]["reason"],
        ).strip("; "),
        "{{result}}": entry["result"],
        "{{status}}": entry["status"],
        "{{kernel_sha256}}": entry["kernel_sha256"],
        "{{evidence}}": "\n".join(evidence_links),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    if "{{" in template or "}}" in template:
        raise ValueError("learned OKF 模板仍有未替换占位符")
    return template


def _frontmatter_value(text, key):
    if not text.startswith("---\n"):
        raise ValueError("OKF concept 缺少 YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("OKF concept frontmatter 未闭合")
    prefix = "{}:".format(key)
    for line in text[4:end].splitlines():
        if line.startswith(prefix):
            raw = line[len(prefix):].strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw.strip("\"'")
    raise ValueError("OKF concept 缺少 {} 字段".format(key))


def _frontmatter(text, path="<concept>"):
    if not text.startswith("---\n"):
        raise ValueError("{} 缺少 YAML frontmatter".format(path))
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("{} frontmatter 未闭合".format(path))
    try:
        value = yaml.safe_load(text[4:end])
    except yaml.YAMLError as exc:
        raise ValueError("{} frontmatter YAML 非法：{}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("{} frontmatter 必须是 mapping".format(path))
    return value, text[end + 5:]


def _json_safe(value):
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _validate_timestamp(value, label):
    if not isinstance(value, str):
        raise ValueError("{} 必须是 ISO 8601 字符串".format(label))
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("{} 必须是合法 ISO 8601 时间".format(label)) from exc
    if parsed.tzinfo is None:
        raise ValueError("{} 必须包含时区".format(label))


def _validate_actor(value, label):
    if not isinstance(value, str) or not ACTOR.fullmatch(value):
        raise ValueError(
            "{} 必须使用 producer/version、human:<id> 或 process:<id>".format(label)
        )


def _validate_action(value, label):
    if not isinstance(value, dict) or set(value) != {"by", "at"}:
        raise ValueError("{} 必须只包含 by 和 at".format(label))
    _validate_actor(value["by"], "{}.by".format(label))
    _validate_timestamp(value["at"], "{}.at".format(label))


def _validate_source_resource(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} 必须是非空字符串".format(label))
    if value.startswith("project-evidence:"):
        return
    parsed = urlparse(value)
    if parsed.scheme in {"https", "http"} and parsed.netloc:
        return
    if parsed.scheme:
        raise ValueError("{} 使用了不支持的 source scheme".format(label))
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("{} 必须是安全相对路径或可访问 URL".format(label))


def _validate_index_links(root):
    errors = []
    for index in sorted(root.rglob("index.md")):
        text = index.read_text(encoding="utf-8")
        for target in MARKDOWN_LINK.findall(text):
            target = target.split("#", 1)[0].strip()
            if not target or urlparse(target).scheme:
                continue
            candidate = (index.parent / target).resolve()
            if not _is_within(candidate, root):
                errors.append("{} 索引链接越出 bundle：{}".format(index.relative_to(root), target))
            elif not candidate.exists():
                errors.append("{} 索引链接不存在：{}".format(index.relative_to(root), target))
            elif candidate.is_dir():
                errors.append(
                    "{} 子目录链接必须显式指向 index.md：{}".format(
                        index.relative_to(root), target
                    )
                )
    return errors


def _normalized_words(value):
    """Return deterministic Unicode words for identifiers and natural-language text."""
    return tuple(
        word for word in re.sub(r"[_\W]+", " ", str(value).casefold()).split()
        if word
    )


def _load_query_vocabulary(root):
    path = Path(root) / QUERY_VOCABULARY
    if not path.is_file():
        raise ValueError("OKF bundle 缺少 {}".format(QUERY_VOCABULARY))
    try:
        vocabulary = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError("{} YAML 非法：{}".format(QUERY_VOCABULARY, exc)) from exc
    if not isinstance(vocabulary, dict) or set(vocabulary) != {
        "schema_version", "operator_families"
    }:
        raise ValueError("{} 字段集合非法".format(QUERY_VOCABULARY))
    if vocabulary["schema_version"] != QUERY_VOCABULARY_VERSION:
        raise ValueError(
            "{} schema_version 必须为 {}".format(
                QUERY_VOCABULARY, QUERY_VOCABULARY_VERSION
            )
        )
    families = vocabulary["operator_families"]
    if not isinstance(families, dict):
        raise ValueError("{} operator_families 必须是 mapping".format(QUERY_VOCABULARY))
    return vocabulary


def _query_aliases(vocabulary, known_families=None):
    aliases = {}
    for canonical, config in vocabulary["operator_families"].items():
        if not isinstance(canonical, str) or not SAFE_SLUG.fullmatch(canonical):
            raise ValueError("查询词表规范算子族必须是安全 slug")
        if known_families is not None and canonical not in known_families:
            raise ValueError("查询词表规范算子族不存在：{}".format(canonical))
        if not isinstance(config, dict) or set(config) != {"aliases"}:
            raise ValueError("查询词表 {} 必须只包含 aliases".format(canonical))
        values = config["aliases"]
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError("查询词表 {} aliases 必须是字符串列表".format(canonical))
        candidates = [canonical] + values
        local = set()
        for value in candidates:
            key = _normalized_words(value)
            if not key:
                raise ValueError("查询词表别名规范化后不能为空")
            if key in local:
                raise ValueError("查询词表 {} 存在重复别名：{}".format(canonical, value))
            local.add(key)
            if key in aliases and aliases[key] != canonical:
                raise ValueError(
                    "查询词表别名冲突：{} 同时属于 {} 和 {}".format(
                        value, aliases[key], canonical
                    )
                )
            aliases[key] = canonical
    return aliases


def _normalize_query(value, aliases):
    words = list(_normalized_words(value))
    if not words:
        return [], ""
    alias_lengths = sorted({len(key) for key in aliases}, reverse=True)
    terms = []
    index = 0
    while index < len(words):
        matched = None
        for length in alias_lengths:
            key = tuple(words[index:index + length])
            if len(key) == length and key in aliases:
                matched = (length, aliases[key])
                break
        if matched:
            length, canonical = matched
            if canonical not in terms:
                terms.append(canonical)
            index += length
        else:
            if words[index] not in terms:
                terms.append(words[index])
            index += 1
    normalized = " ".join(terms)
    return terms, normalized


def _term_words(term):
    return _normalized_words(term)


def _contains_term(field_words, term):
    needle = _term_words(term)
    if not needle:
        return False
    return " ".join(needle) in " ".join(field_words)


def _score_concept(relative, metadata, body, terms, normalized_text):
    families = metadata.get("operator_families", [])
    learned_family = metadata.get("operator_family")
    family_values = list(families) + ([learned_family] if learned_family else [])
    source_titles = [
        source.get("title", "") for source in metadata.get("sources", [])
        if isinstance(source, dict)
    ]
    raw_fields = {
        "title": metadata.get("title", ""),
        "operator_families": " ".join(str(value) for value in family_values),
        "tags": " ".join(str(value) for value in metadata.get("tags", [])),
        "description": metadata.get("description", ""),
        "path": str(relative),
        "body": body,
        "source_titles": " ".join(str(value) for value in source_titles),
    }
    fields = {name: _normalized_words(value) for name, value in raw_fields.items()}
    matched_fields = []
    matched_terms = []
    score = 0
    for field_name in QUERY_FIELD_ORDER:
        field_matches = [
            term for term in terms if _contains_term(fields[field_name], term)
        ]
        if field_matches:
            matched_fields.append(field_name)
            score += QUERY_FIELD_WEIGHTS[field_name] * len(field_matches)
            for term in field_matches:
                if term not in matched_terms:
                    matched_terms.append(term)
    phrase_words = _normalized_words(normalized_text)
    phrase_match = bool(phrase_words) and any(
        _contains_term(field_words, normalized_text) for field_words in fields.values()
    )
    if phrase_match:
        score += 20
    if terms and len(matched_terms) == len(terms):
        score += 10
    return score, matched_fields, matched_terms, phrase_match


def _body_snippets(body, terms, line_offset=0, limit=QUERY_SNIPPET_LIMIT):
    """Return bounded, line-addressable snippets for query terms found in a body."""
    if not terms or limit <= 0:
        return []
    snippets = []
    for line_number, raw_line in enumerate(body.splitlines(), 1):
        text = raw_line.strip()
        if not text:
            continue
        words = _normalized_words(text)
        matched = [term for term in terms if _contains_term(words, term)]
        if not matched:
            continue
        if len(text) > QUERY_SNIPPET_CHARS:
            text = text[:QUERY_SNIPPET_CHARS - 1].rstrip() + "…"
        snippets.append({
            "line": line_number + line_offset,
            "text": text,
            "matched_terms": matched,
        })
        if len(snippets) == limit:
            break
    return snippets


def get_concept(knowledge_root, concept_path):
    """Return one validated concept selected by bundle- or project-relative path."""
    root = Path(knowledge_root).resolve()
    validate_bundle(root)
    relative = _safe_relative_path(concept_path, "path")
    project_prefix = PROJECT_KNOWLEDGE.parts
    if relative.parts[:len(project_prefix)] == project_prefix:
        relative = Path(*relative.parts[len(project_prefix):])
    if (
        not relative.parts
        or relative.suffix != ".md"
        or relative.name in {"index.md", "log.md"}
    ):
        raise ValueError("path 必须指向 bundle 内的非保留 Markdown concept")
    path = root / relative
    _reject_symlink_chain(path, root)
    if not _is_within(path, root) or not path.is_file():
        raise ValueError("concept 不存在：{}".format(relative))
    metadata, body = _frontmatter(path.read_text(encoding="utf-8"), str(relative))
    return {
        "path": str(relative),
        "metadata": _json_safe(metadata),
        "body": body,
    }


def _query_suggestions(text, operator_family, aliases):
    raw = operator_family or text
    words = _normalized_words(raw) if raw else ()
    if not words:
        return []
    display = {" ".join(key): canonical for key, canonical in aliases.items()}
    candidates = list(display)
    suggestions = []
    seen = set()
    for word in words:
        for match in difflib.get_close_matches(word, candidates, n=3, cutoff=0.6):
            canonical = display[match]
            if canonical in seen:
                continue
            seen.add(canonical)
            suggestions.append({
                "term": match,
                "normalized_operator_family": canonical,
                "filters": {"operator_family": canonical},
            })
    return suggestions[:3]


def validate_bundle(knowledge_root):
    root = Path(knowledge_root).resolve()
    vocabulary = _load_query_vocabulary(root)
    root_index = root / "index.md"
    if not root_index.is_file():
        raise ValueError("OKF bundle 缺少 index.md")
    root_text = root_index.read_text(encoding="utf-8")
    try:
        root_metadata, _root_body = _frontmatter(root_text, "index.md")
    except ValueError as exc:
        raise ValueError("OKF bundle 根索引非法：{}".format(exc)) from exc
    if root_metadata != {"okf_version": OKF_VERSION}:
        raise ValueError("OKF bundle 版本不是 {}".format(OKF_VERSION))
    concepts = []
    errors = []
    known_families = set()
    required = {"type", "title", "description", "tags", "status", "generated", "sources"}
    for path in sorted(root.rglob("*.md")):
        relative = path.relative_to(root)
        if path.name in {"index.md", "log.md"}:
            continue
        try:
            metadata, body = _frontmatter(
                path.read_text(encoding="utf-8"), str(relative)
            )
            missing = sorted(required - set(metadata))
            if missing:
                raise ValueError(
                    "{} 缺少 OKF 字段：{}".format(relative, ", ".join(missing))
                )
            if not isinstance(metadata["type"], str) or not metadata["type"].strip():
                raise ValueError("{} type 必须是非空字符串".format(relative))
            for key in ("title", "description"):
                if not isinstance(metadata[key], str) or not metadata[key].strip():
                    raise ValueError("{} {} 必须是非空字符串".format(relative, key))
            if (
                not isinstance(metadata["tags"], list)
                or not metadata["tags"]
                or any(not isinstance(tag, str) or not tag.strip() for tag in metadata["tags"])
            ):
                raise ValueError("{} tags 必须是非空字符串列表".format(relative))
            if metadata["status"] not in OKF_STATUSES:
                raise ValueError("{} status 必须是 draft、stable 或 deprecated".format(relative))
            _validate_action(metadata["generated"], "{} generated".format(relative))
            verified = metadata.get("verified")
            if not isinstance(verified, list) or not verified:
                raise ValueError("{} verified 必须是非空列表".format(relative))
            for index, action in enumerate(verified):
                _validate_action(action, "{} verified[{}]".format(relative, index))
            sources = metadata["sources"]
            if not isinstance(sources, list) or not sources:
                raise ValueError("{} sources 必须是非空列表".format(relative))
            source_ids = set()
            for index, source in enumerate(sources):
                label = "{} sources[{}]".format(relative, index)
                if not isinstance(source, dict) or "resource" not in source:
                    raise ValueError("{} 必须是包含 resource 的 mapping".format(label))
                _validate_source_resource(source["resource"], "{}.resource".format(label))
                source_id = source.get("id")
                if source_id is not None:
                    if not isinstance(source_id, str) or not SOURCE_ID.fullmatch(source_id):
                        raise ValueError("{} id 非法".format(label))
                    if source_id in source_ids:
                        raise ValueError("{} source id 重复：{}".format(relative, source_id))
                    source_ids.add(source_id)
            if relative.parts[0] in {
                "dsl", "operator", "debug", "profiler", "optimization"
            }:
                if not source_ids:
                    raise ValueError("{} 静态 concept 必须声明 source id".format(relative))
                for source_id in source_ids:
                    if "[^{}]".format(source_id) not in body:
                        raise ValueError("{} 未引用 source id {}".format(relative, source_id))
                    if "[^{}]:".format(source_id) not in body:
                        raise ValueError("{} 缺少 source footnote {}".format(relative, source_id))
            families = metadata.get("operator_families", [])
            if isinstance(families, list):
                known_families.update(
                    family for family in families if isinstance(family, str)
                )
            learned_family = metadata.get("operator_family")
            if isinstance(learned_family, str):
                known_families.add(learned_family)
            concepts.append(str(relative))
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(str(exc))
    errors.extend(_validate_index_links(root))
    try:
        _query_aliases(vocabulary, known_families)
    except ValueError as exc:
        errors.append(str(exc))
    if errors:
        raise ValueError("; ".join(errors))
    return {"okf_version": OKF_VERSION, "concepts": concepts, "count": len(concepts)}


def query_bundle_report(
    knowledge_root, concept_type=None, tags=None, status=None,
    operator_family=None, arch=None, text=None, limit=None, compact=False,
):
    root = Path(knowledge_root).resolve()
    validate_bundle(root)
    aliases = _query_aliases(_load_query_vocabulary(root))
    required_tags = set(tags or [])
    terms, normalized_text = _normalize_query(text, aliases) if text else ([], "")
    family_terms, _normalized_family_text = (
        _normalize_query(operator_family, aliases) if operator_family else ([], "")
    )
    canonical_families = set(aliases.values())
    resolved_families = [term for term in family_terms if term in canonical_families]
    normalized_family = None
    if operator_family:
        normalized_family = (
            resolved_families[0]
            if len(set(resolved_families)) == 1
            else operator_family
        )
    candidates = []

    for path in sorted(root.rglob("*.md")):
        if path.name in {"index.md", "log.md"}:
            continue
        relative = path.relative_to(root)
        raw = path.read_text(encoding="utf-8")
        metadata, body = _frontmatter(raw, str(relative))
        body_line_offset = raw[:len(raw) - len(body)].count("\n")
        if concept_type and metadata.get("type") != concept_type:
            continue
        if status and metadata.get("status") != status and metadata.get("result_status") != status:
            continue
        if required_tags and not required_tags.issubset(set(metadata.get("tags", []))):
            continue
        families = metadata.get("operator_families", [])
        learned_family = metadata.get("operator_family")
        if normalized_family and normalized_family not in families and normalized_family != learned_family:
            continue
        arches = metadata.get("arch", [])
        if arch and arch not in arches:
            continue
        score, matched_fields, matched_terms, phrase_match = _score_concept(
            relative, metadata, body, terms, normalized_text
        )
        result = {
            "path": str(relative),
            "type": metadata["type"],
            "title": metadata["title"],
            "description": metadata["description"],
            "status": metadata["status"],
            "score": score,
            "matched_fields": matched_fields,
            "matched_terms": matched_terms,
            "body_snippets": _body_snippets(body, terms, line_offset=body_line_offset),
        }
        if not compact:
            result.update({
                "tags": metadata["tags"],
                "verified": _json_safe(metadata.get("verified", [])),
                "sources": _json_safe(metadata["sources"]),
            })
        candidates.append(
            {"result": result, "phrase_match": phrase_match}
        )

    if not terms:
        selected = candidates
        match_mode = "exact"
    else:
        selected = [
            item for item in candidates
            if len(item["result"]["matched_terms"]) == len(terms)
        ]
        if selected:
            match_mode = (
                "exact" if any(item["phrase_match"] for item in selected)
                else "all_terms"
            )
        else:
            minimum = max(1, int(math.ceil(len(terms) / 2.0)))
            selected = [
                item for item in candidates
                if len(item["result"]["matched_terms"]) >= minimum
            ]
            match_mode = "relaxed"

    results = [item["result"] for item in selected]
    results.sort(key=lambda item: (
        -item["score"],
        QUERY_STATUS_ORDER.get(item["status"], len(QUERY_STATUS_ORDER)),
        item["path"],
    ))
    total_count = len(results)
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit 必须是正整数")
        results = results[:limit]
    return {
        "results": results,
        "total_count": total_count,
        "query": {
            "normalized_text": normalized_text or None,
            "normalized_operator_family": normalized_family,
            "match_mode": match_mode,
        },
        "suggestions": (
            [] if results else _query_suggestions(text, operator_family, aliases)
        ),
    }


def query_bundle(
    knowledge_root, concept_type=None, tags=None, status=None,
    operator_family=None, arch=None, text=None, limit=None, compact=False,
):
    return query_bundle_report(
        knowledge_root,
        concept_type=concept_type,
        tags=tags,
        status=status,
        operator_family=operator_family,
        arch=arch,
        text=text,
        limit=limit,
        compact=compact,
    )["results"]


def _regenerate_learned_index(learned):
    rows = []
    for path in sorted(learned.glob("*.md")):
        if path.name == "index.md":
            continue
        text = path.read_text(encoding="utf-8")
        if _frontmatter_value(text, "type") != "CATLASS DSL Learned Result":
            raise ValueError("learned 目录包含未知 OKF type：{}".format(path.name))
        rows.append(
            (
                str(_frontmatter_value(text, "title")),
                path.name,
                str(_frontmatter_value(text, "description")),
            )
        )
    lines = ["# CATLASS DSL Learned Result", ""]
    if rows:
        for title, filename, description in rows:
            lines.append("* [{}]({}) - {}".format(title, filename, description))
    else:
        lines.append("No project-verified concepts have been recorded.")
    content = "\n".join(lines) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".index.", suffix=".md", dir=str(learned))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(learned / "index.md"))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _require_initialized_project(project):
    root_index = project / PROJECT_KNOWLEDGE / "index.md"
    if (
        not root_index.is_file()
        or 'okf_version: "0.2"' not in root_index.read_text(encoding="utf-8")
    ):
        raise ValueError(
            "目标项目 .catlass-dsl/knowledge 不是已初始化的 OKF v0.2 bundle"
        )


def _prepare_record(entry, project, date_value):
    errors = validate_entry(entry, project)
    if errors:
        raise ValueError("; ".join(errors))
    entry = dict(entry)
    filename = "{}-{}-{}.md".format(date_value, entry["operator_family"], entry["topic"])
    if not SAFE_SLUG.fullmatch(entry["operator_family"]) or not SAFE_SLUG.fullmatch(entry["topic"]):
        raise ValueError("learned 文件名 slug 非法")
    return filename, _render(entry, date_value)


def _write_exclusive(destination, content):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(destination), flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _record_result(destination):
    return {
        "status": "passed",
        "filename": destination.name,
        "path": str(destination),
        "okf_version": OKF_VERSION,
    }


def record_entry(entry, project_root, date_value=None, regenerate_index=True):
    project = Path(project_root).resolve()
    _require_initialized_project(project)
    date_value = date_value or datetime.date.today().isoformat()
    datetime.date.fromisoformat(date_value)
    filename, content = _prepare_record(entry, project, date_value)

    learned = project / PROJECT_KNOWLEDGE / "learned"
    _reject_symlink_chain(learned, project)
    learned.mkdir(parents=True, exist_ok=True)
    destination = learned / filename

    if fcntl is None:
        raise RuntimeError("unsupported filesystem: 缺少 POSIX fcntl.flock")
    lock_path = learned / ".record.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _write_exclusive(destination, content)
        try:
            if regenerate_index:
                _regenerate_learned_index(learned)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    return _record_result(destination)


def reindex(knowledge_root):
    root = Path(knowledge_root).resolve()
    validate_bundle(root)
    learned = root / "learned"
    _reject_symlink_chain(learned, root)
    learned.mkdir(parents=True, exist_ok=True)
    if fcntl is None:
        raise RuntimeError("unsupported filesystem: 缺少 POSIX fcntl.flock")
    with (learned / ".record.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _regenerate_learned_index(learned)
    return {"status": "passed", "path": str(learned / "index.md")}


def record_entries(entries, project_root, date_value=None):
    if not isinstance(entries, list) or not entries:
        raise ValueError("批量 entry 必须是非空列表")
    project = Path(project_root).resolve()
    _require_initialized_project(project)
    errors = []
    filenames = set()
    date_value = date_value or datetime.date.today().isoformat()
    datetime.date.fromisoformat(date_value)
    prepared = []
    for index, entry in enumerate(entries):
        entry_errors = validate_entry(entry, project)
        errors.extend("entry[{}]: {}".format(index, error) for error in entry_errors)
        if isinstance(entry, dict):
            filename = "{}-{}-{}.md".format(
                date_value, entry.get("operator_family", ""), entry.get("topic", "")
            )
            if filename in filenames:
                errors.append("entry[{}]: 批次内 learned 文件名重复".format(index))
            filenames.add(filename)
            if not entry_errors:
                prepared.append((filename, _render(dict(entry), date_value)))
    if errors:
        raise ValueError("; ".join(errors))

    learned = project / PROJECT_KNOWLEDGE / "learned"
    _reject_symlink_chain(learned, project)
    learned.mkdir(parents=True, exist_ok=True)
    if fcntl is None:
        raise RuntimeError("unsupported filesystem: 缺少 POSIX fcntl.flock")

    destinations = [learned / filename for filename, _content in prepared]
    created = []
    with (learned / ".record.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        conflicts = [path.name for path in destinations if path.exists()]
        if conflicts:
            raise FileExistsError(
                "learned concept 已存在：{}".format(", ".join(sorted(conflicts)))
            )
        try:
            for destination, (_filename, content) in zip(destinations, prepared):
                _write_exclusive(destination, content)
                created.append(destination)
            _regenerate_learned_index(learned)
        except BaseException:
            for destination in reversed(created):
                destination.unlink(missing_ok=True)
            raise

    results = [_record_result(destination) for destination in destinations]
    return {
        "status": "passed",
        "count": len(results),
        "entries": results,
        "okf_version": OKF_VERSION,
    }


def _positive_int(value):
    if not isinstance(value, str) or not POSITIVE_INT.fullmatch(value):
        raise argparse.ArgumentTypeError("必须是正整数")
    return int(value)


def build_parser():
    parser = argparse.ArgumentParser(description="初始化或追加 CATLASS DSL OKF 知识库")
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize_parser = subparsers.add_parser("initialize")
    initialize_parser.add_argument("--project-root", required=True)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--entry", required=True)
    record_parser.add_argument("--project-root", required=True)
    record_parser.add_argument("--date")
    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--project-root", required=True)
    query_parser.add_argument("--type")
    query_parser.add_argument("--tag", action="append", default=[])
    query_parser.add_argument("--status")
    query_parser.add_argument("--operator-family")
    query_parser.add_argument("--arch")
    query_parser.add_argument("--text")
    query_parser.add_argument("--limit", type=_positive_int, default=20)
    query_parser.add_argument(
        "--compact", action="store_true",
        help="omit full tags/verification/source metadata; use get for a selected concept",
    )
    get_parser = subparsers.add_parser("get")
    get_parser.add_argument("--project-root", required=True)
    get_parser.add_argument(
        "--path", required=True,
        help="path returned by query, relative to the project or knowledge bundle",
    )
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--project-root", required=True)
    reindex_parser = subparsers.add_parser("reindex")
    reindex_parser.add_argument("--project-root", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "initialize":
            project = Path(args.project_root).resolve()
            copied = initialize(PLUGIN_KNOWLEDGE, project / PROJECT_KNOWLEDGE)
            result = {"status": "passed", "copied": copied, "okf_version": OKF_VERSION}
        elif args.command == "record":
            entry = json.loads(Path(args.entry).read_text(encoding="utf-8"))
            result = (
                record_entries(entry, args.project_root, args.date)
                if isinstance(entry, list)
                else record_entry(entry, args.project_root, args.date)
            )
        elif args.command == "query":
            report = query_bundle_report(
                Path(args.project_root) / PROJECT_KNOWLEDGE,
                concept_type=args.type,
                tags=args.tag,
                status=args.status,
                operator_family=args.operator_family,
                arch=args.arch,
                text=args.text,
                limit=args.limit,
                compact=args.compact,
            )
            for item in report["results"]:
                item["path"] = str(PROJECT_KNOWLEDGE / item["path"])
            result = {
                "status": "passed",
                "count": len(report["results"]),
                "total_count": report["total_count"],
                "results": report["results"],
                "query": report["query"],
                "suggestions": report["suggestions"],
                "okf_version": OKF_VERSION,
            }
        elif args.command == "get":
            concept = get_concept(
                Path(args.project_root) / PROJECT_KNOWLEDGE,
                args.path,
            )
            concept["path"] = str(PROJECT_KNOWLEDGE / concept["path"])
            result = {
                "status": "passed",
                "concept": concept,
                "okf_version": OKF_VERSION,
            }
        elif args.command == "validate":
            result = {
                "status": "passed",
                **validate_bundle(Path(args.project_root) / PROJECT_KNOWLEDGE),
            }
        else:
            result = reindex(Path(args.project_root) / PROJECT_KNOWLEDGE)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        result = {"status": "failed", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
