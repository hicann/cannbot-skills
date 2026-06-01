#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Select relevant kernel examples from the generated kernel index."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
KERNEL_INDEX = ROOT / "agent" / "index" / "kernels.json"

TOPOLOGY_CHOICES = [
    "cube-only",
    "cube->vec",
    "vec->cube",
    "vec->cube->vec",
    "cube->vec->cube->vec",
    "vec-only",
    "micro-only",
]
TAG_CHOICES = [
    "splitk",
    "splitn",
    "fp8",
    "dual-output",
    "delayed-stage",
    "rowwise-norm",
    "quant",
]
FEATURE_CHOICES = [
    "vec-postprocess",
    "vec-preprocess",
    "atomic-add",
    "two-pass",
]
WORD_RE = re.compile(r"[a-z0-9]+")


def _flatten_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _normalize_text(text: str) -> str:
    text = text.lower().replace("`", "")
    text = text.replace("→", "->")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> List[str]:
    return WORD_RE.findall(_normalize_text(text))


def _expand_query_tokens(tokens: Sequence[str]) -> List[str]:
    expanded = []
    seen = set()
    for token in tokens:
        candidates = [token]
        if token.startswith("add") and any(ch.isdigit() for ch in token[3:]):
            candidates.append("add")
        if token == "quant":
            candidates.append("quantized")
        if token == "norm":
            candidates.extend(["normalize", "normalization"])
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            expanded.append(candidate)
    return expanded


def _canonical_topology(text: str) -> str:
    normalized = _normalize_text(text).replace(" ", "")
    if "cube->vec->cube->vec" in normalized:
        return "cube->vec->cube->vec"
    if "vec->cube->vec" in normalized:
        return "vec->cube->vec"
    if "vec->cube" in normalized:
        return "vec->cube"
    if "cube->vec" in normalized:
        return "cube->vec"
    if "cube-only" in normalized or normalized == "cubeonly":
        return "cube-only"
    if "vec-only" in normalized or normalized == "veconly":
        return "vec-only"
    if "micro-only" in normalized or normalized == "microonly":
        return "micro-only"
    return _normalize_text(text)


def _collect_entry_text(entry: Dict[str, Any], include_negative_guidance: bool = False) -> str:
    parts = [
        _flatten_text(entry.get("name", "")),
        _flatten_text(entry.get("path", "")),
        _flatten_text(entry.get("category", "")),
        _flatten_text(entry.get("formula", "")),
        _flatten_text(entry.get("topology", "")),
        _flatten_text(entry.get("study_for", [])),
    ]
    if include_negative_guidance:
        parts.append(_flatten_text(entry.get("do_not_copy_when", [])))
    return _normalize_text(" ".join(part for part in parts if part))


def _derive_tags(entry: Dict[str, Any], entry_text: str) -> Set[str]:
    tags = set()
    text = entry_text
    if "splitk" in text or "split-k" in text:
        tags.add("splitk")
    if "splitn" in text or "split-n" in text:
        tags.add("splitn")
    if any(token in text for token in ["fp8", "float8", "e5m2", "e4m3"]):
        tags.add("fp8")
    if any(token in text for token in ["dual output", "dual-output", "two outputs", "dual-fp8-output"]):
        tags.add("dual-output")
    if any(token in text for token in ["delayed", "lookahead", "warmup and drain"]):
        tags.add("delayed-stage")
    if "rowwise" in text and any(token in text for token in ["norm", "normalize"]):
        tags.add("rowwise-norm")
    if any(token in text for token in ["quant", "quantized", "absmax", "scale = absmax"]):
        tags.add("quant")
    return tags


def _derive_features(entry: Dict[str, Any], entry_text: str, topology: str) -> Set[str]:
    features = set()
    text = entry_text
    if topology in ["cube->vec", "vec->cube->vec", "cube->vec->cube->vec"]:
        features.add("vec-postprocess")
    if topology in ["vec->cube", "vec->cube->vec"]:
        features.add("vec-preprocess")
    if "atomic" in text:
        features.add("atomic-add")
    if "two-pass" in text:
        features.add("two-pass")
    return features


def _field_token_sets(entry: Dict[str, Any]) -> Dict[str, Set[str]]:
    return {
        "name": set(_tokenize(entry.get("name", ""))),
        "path": set(_tokenize(entry.get("path", ""))),
        "category": set(_tokenize(entry.get("category", ""))),
        "formula": set(_tokenize(_flatten_text(entry.get("formula", "")))),
        "study_for": set(_tokenize(_flatten_text(entry.get("study_for", [])))),
    }


def _find_token_overlap(tokens: Sequence[str], token_set: Set[str]) -> Set[str]:
    overlap = set()
    for token in tokens:
        if token in token_set:
            overlap.add(token)
            continue
        if len(token) < 3:
            continue
        if any(candidate.startswith(token) or token.startswith(candidate) for candidate in token_set):
            overlap.add(token)
    return overlap


def _match_query(tokens: Sequence[str], field_tokens: Dict[str, Set[str]]) -> Tuple[int, List[str], Set[str]]:
    if not tokens:
        return 0, [], set()

    weights = {
        "name": 3,
        "path": 2,
        "formula": 3,
        "study_for": 2,
        "category": 1,
    }
    score = 0
    matched_fields = []
    matched_tokens = set()

    for field_name, token_set in field_tokens.items():
        overlap = _find_token_overlap(tokens, token_set)
        if not overlap:
            continue
        matched_tokens.update(overlap)
        score += len(overlap) * weights[field_name]
        matched_fields.append("%s=%s" % (field_name, ", ".join(sorted(overlap))))

    return score, matched_fields, matched_tokens


def _score_query_intent(query_tokens: Sequence[str], entry_tokens: Set[str]) -> Tuple[int, List[str]]:
    score = 0
    reasons = []
    query_token_set = set(query_tokens)

    if query_token_set.intersection(["softmax", "normalized"]):
        matched = sorted(_find_token_overlap(["softmax", "normalized"], entry_tokens))
        if matched:
            score += len(matched) * 2
            reasons.append("normalized/softmax intent matched: %s" % ", ".join(matched))

    specialization_groups = [
        ("causal specialization", {"causal"}),
        ("fp8 specialization", {"fp8", "float8", "e5m2", "e4m3"}),
        ("hif8 specialization", {"hif8"}),
        ("block32 specialization", {"block32"}),
    ]
    for label, group_tokens in specialization_groups:
        if query_token_set.intersection(group_tokens):
            continue
        matched = sorted(entry_tokens.intersection(group_tokens))
        if matched:
            score -= 3
            reasons.append("%s unrequested: %s" % (label, ", ".join(matched)))

    return score, reasons


def load_index(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("Kernel index not found: %s. Run `python3 tools/build_agent_index.py` first." % path)
    return json.loads(path.read_text(encoding="utf-8"))


def score_entries(
    entries: Iterable[Dict[str, Any]], args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    strong_results = []
    weak_results = []
    query_text = " ".join(part for part in [args.query, args.formula] if part)
    query_tokens = _expand_query_tokens(_tokenize(query_text))

    for entry in entries:
        entry_text = _collect_entry_text(entry)
        entry_tokens = set(_tokenize(entry_text))
        topology = _canonical_topology(entry.get("topology", ""))
        tags = _derive_tags(entry, entry_text)
        features = _derive_features(entry, entry_text, topology)
        field_tokens = _field_token_sets(entry)
        reasons = []
        score = 0

        if args.topology:
            requested_topology = _canonical_topology(args.topology)
            if topology != requested_topology:
                continue
            score += 10
            reasons.append("topology matched: %s" % requested_topology)

        missing_tags = [tag for tag in args.tag if tag not in tags]
        if missing_tags:
            continue
        for tag in args.tag:
            score += 4
            reasons.append("tag matched: %s" % tag)

        missing_features = [feature for feature in args.has if feature not in features]
        if missing_features:
            continue
        for feature in args.has:
            score += 3
            reasons.append("feature matched: %s" % feature)

        if args.dtype:
            dtype_token = _normalize_text(args.dtype)
            if dtype_token not in entry_text:
                continue
            score += 2
            reasons.append("dtype matched: %s" % dtype_token)

        query_score, matched_fields, matched_tokens = _match_query(query_tokens, field_tokens)
        score += query_score
        if matched_fields:
            reasons.append("query matched: %s" % "; ".join(matched_fields))

        intent_score, intent_reasons = _score_query_intent(query_tokens, entry_tokens)
        score += intent_score
        reasons.extend(intent_reasons)

        result = {
            "path": entry["path"],
            "name": entry.get("name", Path(entry["path"]).name),
            "category": entry.get("category", ""),
            "formula": entry.get("formula", ""),
            "topology": topology,
            "study_for": entry.get("study_for", []),
            "deep_note": entry.get("deep_note", ""),
            "do_not_copy_when": entry.get("do_not_copy_when", []),
            "score": score,
            "why": reasons,
            "order": entry.get("order", 0),
            "tags": sorted(tags),
            "features": sorted(features),
            "matched_query_tokens": sorted(matched_tokens),
        }

        is_strong = True
        if query_tokens:
            is_strong = bool(matched_tokens)
        if is_strong:
            strong_results.append(result)
        else:
            weak_results.append(result)

    key = lambda item: (-item["score"], item["order"], item["path"])
    strong_results.sort(key=key)
    weak_results.sort(key=key)
    return strong_results, weak_results


def _print_list_block(title: str, items: Any) -> None:
    text_items = items if isinstance(items, list) else [items]
    print("   %s:" % title)
    for item in text_items:
        print("   - %s" % item)


def print_text_results(results: List[Dict[str, Any]], args: argparse.Namespace, weak: bool = False) -> None:
    if args.path_only:
        for item in results:
            print(item["path"])
        return

    if weak:
        print("No strong match found. Showing weaker candidates:\n")

    for index, item in enumerate(results, start=1):
        print("%d. %s" % (index, item["path"]))
        print("   category: %s" % item["category"])
        print("   topology: %s" % item["topology"])
        if args.show_score:
            print("   score: %s" % item["score"])
        if args.show_why:
            print("   why: %s" % (" | ".join(item["why"]) if item["why"] else "no specific match reason"))
        _print_list_block("study_for", item["study_for"])
        _print_list_block("do_not_copy_when", item["do_not_copy_when"])
        if args.catalog:
            _print_list_block("formula", item["formula"])
            if item["deep_note"]:
                print("   deep_note: %s" % item["deep_note"])
            if item["tags"]:
                print("   tags: %s" % ", ".join(item["tags"]))
            if item["features"]:
                print("   features: %s" % ", ".join(item["features"]))
        print()


def print_json_results(results: List[Dict[str, Any]], args: argparse.Namespace, weak: bool = False) -> None:
    payload = {
        "query": {
            "query": args.query,
            "formula": args.formula,
            "topology": _canonical_topology(args.topology) if args.topology else None,
            "tags": list(args.tag),
            "has": list(args.has),
            "dtype": args.dtype,
            "weak_only": weak,
        },
        "result_count": len(results),
        "results": results,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select relevant kernel examples from agent/index/kernels.json.")
    parser.add_argument("--query", default="", help="Free-form task description or formula keywords.")
    parser.add_argument("--formula", default="", help="Formula-focused query text.")
    parser.add_argument("--topology", choices=TOPOLOGY_CHOICES, help="Canonical topology filter.")
    parser.add_argument("--tag", action="append", default=[], choices=TAG_CHOICES, help="Repeatable tag filter.")
    parser.add_argument(
        "--has", action="append", default=[], choices=FEATURE_CHOICES,
        help="Repeatable feature filter.",
    )
    parser.add_argument("--dtype", default="", help="Optional dtype keyword filter.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to print.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--show-score", action="store_true", help="Show per-result match score in text mode.")
    parser.add_argument("--show-why", action="store_true", help="Show short reasoning in text mode.")
    parser.add_argument("--path-only", action="store_true", help="Print only ranked source paths.")
    parser.add_argument("--catalog", action="store_true", help="Print richer catalog-style fields in text mode.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.limit <= 0:
        parser.error("--limit must be > 0")

    try:
        payload = load_index(KERNEL_INDEX)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    strong_results, weak_results = score_entries(payload.get("entries", []), args)
    results = strong_results[: args.limit]
    weak = False

    if not results and weak_results:
        results = weak_results[: args.limit]
        weak = True

    if args.json:
        print_json_results(results, args, weak=weak)
        return 0

    if not results:
        print("No match found.")
        return 0

    print_text_results(results, args, weak=weak)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
