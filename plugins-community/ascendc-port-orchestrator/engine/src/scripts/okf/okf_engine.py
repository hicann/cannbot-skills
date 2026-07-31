#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""okf_engine.py — locate the EXTERNAL cannbot-knowledge OKF engine (thin interface, no vendoring).

The port plugin no longer vendors the OKF retrieval/lint engine. Per RFC #381 it DEPENDS on the
community plugin `cannbot-knowledge` (marketplace: cann/cannbot-skills), which ships the deterministic
`knowledge_query.py` / `knowledge_lint.py` / `okf_graph.py`. The port plugin only self-maintains its
own cards under `kb/okf/` and calls the shared engine against them via `--knowledge-root`.

Package split (marketplace `cannbot`):
  - runtime retrieval  → `cannbot-knowledge-consumer-skills`   (ships only skills/knowledge-query)
  - card authoring/lint → `cannbot-knowledge-contributor-skills` (ships all 7 skills incl. knowledge-lint,
                          ops-knowledge-ingest/okf_graph.py) — maintainer side only.

Resolution order (fail-closed, never a silent guess). Each capability (query / lint / graph) is resolved
INDEPENDENTLY by its own marker file, across ALL candidate dirs — so a consumer-only install (has
knowledge-query but not knowledge-lint) does NOT shadow a contributor install that has lint:
  1. $CANNBOT_OKF_ENGINE_ROOT — explicit path to the cannbot-knowledge plugin dir (or its skills/ dir).
  2. sibling-plugin discovery relative to this plugin's install location (marketplace / monorepo layouts).
  3. active install paths from Claude's installed_plugins.json registry.
  4. unambiguous legacy/cache fallback only when no registry exists.
Candidate DIR names match both the canonical `cannbot-knowledge` and marketplace-install variants like
`cannbot-knowledge-consumer-skills` / `cannbot-knowledge-contributor-skills` (validated by the marker file,
never by name alone), including versioned cache paths
(`<plugins>/cache/<marketplace>/cannbot-knowledge*/<version>`).
"""
import json
import os
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# per-capability marker files (relative to a cannbot-knowledge plugin root):
_QUERY_REL = "skills/knowledge-query/scripts/knowledge_query.py"     # consumer tier (always present)
_QUERY_RELS = (_QUERY_REL, "knowledge-query/scripts/knowledge_query.py")
_LINT_REL = "skills/knowledge-lint/scripts/knowledge_lint.py"        # contributor tier
_GRAPH_REL = "skills/ops-knowledge-ingest/scripts/okf_graph.py"      # contributor tier

_DIR_GLOBS = ("cannbot-knowledge", "cannbot-knowledge*",
              "*/cannbot-knowledge", "*/cannbot-knowledge*")
_CACHE_GLOBS = ("cache/*/cannbot-knowledge*/*",)
_REGISTRY_ABSENT = "absent"
_REGISTRY_INVALID = "invalid"
_REGISTRY_VALID = "valid"

INSTALL_HINT = (
    "cannbot-knowledge OKF engine not found. The port plugin depends on it (RFC #381) — install it:\n"
    "  /plugin marketplace add https://gitcode.com/cann/cannbot-skills.git\n"
    "  /plugin install cannbot-knowledge-consumer-skills@cannbot     # runtime retrieval\n"
    "  /plugin install cannbot-knowledge-contributor-skills@cannbot  # + lint (maintainer only)\n"
    "or export CANNBOT_OKF_ENGINE_ROOT=/path/to/cannbot-knowledge (its plugin dir)."
)


def _plugin_root() -> Path:
    # this file lives at <plugin>/engine/src/scripts/okf/okf_engine.py → 4 parents up = plugin root.
    return Path(__file__).resolve().parents[4]


def _read_registry(plugins_dir: Path) -> Tuple[str, List[Path]]:
    """Return registry state and active cannbot-knowledge install paths."""
    registry = plugins_dir / "installed_plugins.json"
    try:
        raw = registry.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _REGISTRY_ABSENT, []
    except (OSError, UnicodeError):
        return _REGISTRY_INVALID, []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _REGISTRY_INVALID, []

    if not isinstance(data, dict):
        return _REGISTRY_INVALID, []
    plugins = data.get("plugins")
    if not isinstance(plugins, dict):
        return _REGISTRY_INVALID, []

    candidates = []
    for key in sorted(plugins):
        package = key.split("@", 1)[0]
        if package != "cannbot-knowledge" and not package.startswith("cannbot-knowledge-"):
            continue
        records = plugins[key]
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            install_path = record.get("installPath")
            if not isinstance(install_path, str) or not install_path:
                continue
            try:
                candidate = Path(install_path).expanduser()
            except (OSError, RuntimeError, ValueError):
                continue
            if not candidate.is_absolute():
                candidate = plugins_dir / candidate
            candidates.append(candidate)
    return _REGISTRY_VALID, candidates


def _unique_version_candidates(base: Path, pattern: str) -> Iterator[Path]:
    """Yield a cache package only when exactly one version exists for that package."""
    by_package = {}
    try:
        for candidate in base.glob(pattern):
            if candidate.is_dir():
                by_package.setdefault(candidate.parent, set()).add(candidate)
    except OSError:
        return
    for package in sorted(by_package):
        versions = sorted(by_package[package])
        if len(versions) == 1:
            yield versions[0]


def _versioned_cache_base(plugin_root: Path) -> Optional[Path]:
    """Return <plugins> for <plugins>/cache/<marketplace>/<package>/<version>."""
    try:
        if plugin_root.parents[2].name == "cache":
            return plugin_root.parents[3]
    except IndexError:
        pass
    return None


def _dir_candidates() -> Iterator[Path]:
    """Yield candidate cannbot-knowledge plugin roots, most-specific first, deduped, deterministic."""
    seen = set()

    def _emit(p: Path) -> Iterator[Path]:
        rp = p
        if rp not in seen:
            seen.add(rp)
            yield rp

    env = os.environ.get("CANNBOT_OKF_ENGINE_ROOT")
    if env:
        e = Path(env).expanduser()
        yield from _emit(e)            # user pointed at the plugin dir
        yield from _emit(e.parent)     # …or at its skills/ dir

    pr = _plugin_root()
    cfg = os.environ.get("CLAUDE_CONFIG_DIR")
    plugin_bases = []
    if cfg:
        try:
            plugin_bases.append(Path(cfg).expanduser() / "plugins")
        except (OSError, RuntimeError, ValueError):
            pass
    else:
        plugin_bases.append(Path(os.path.expanduser("~")) / ".claude" / "plugins")
    cache_base = _versioned_cache_base(pr)
    if cache_base is not None:
        plugin_bases.append(cache_base)
    plugin_bases = list(dict.fromkeys(plugin_bases))

    # Source checkouts and non-Claude legacy layouts are authoritative siblings. A sibling base
    # that is itself a Claude plugins dir is handled by its registry below.
    direct_bases = [pr.parents[1] / "plugins-community"]
    if pr.parent not in plugin_bases:
        direct_bases.insert(0, pr.parent)
    for base in direct_bases:
        try:
            if not base.is_dir():
                continue
            for pat in _DIR_GLOBS:
                for cand in sorted(base.glob(pat)):
                    if cand.is_dir():
                        yield from _emit(cand)
        except OSError:
            continue

    registry_states = {}
    for base in plugin_bases:
        state, candidates = _read_registry(base)
        registry_states[base] = state
        for cand in candidates:
            yield from _emit(cand)

    # A valid or malformed registry is authoritative: never revive an orphaned cache entry that
    # Claude no longer considers active. Registry-less pre-populated caches remain supported, but
    # only when a package has exactly one version (multiple versions are ambiguous and skipped).
    for base in plugin_bases:
        if registry_states[base] != _REGISTRY_ABSENT:
            continue
        try:
            if not base.is_dir():
                continue
            for pat in _DIR_GLOBS:
                for cand in sorted(base.glob(pat)):
                    if cand.is_dir():
                        yield from _emit(cand)
        except OSError:
            continue
        for pat in _CACHE_GLOBS:
            for cand in _unique_version_candidates(base, pat):
                yield from _emit(cand)


def _resolve_with_markers(marker_rels: Tuple[str, ...]) -> Optional[Tuple[Path, str]]:
    """Return the first candidate and marker it contains, preserving candidate priority."""
    for c in _dir_candidates():
        for marker_rel in marker_rels:
            try:
                if (c / marker_rel).is_file():
                    return c, marker_rel
            except OSError:
                continue
    return None


def _resolve_with_marker(marker_rel: str) -> Optional[Path]:
    """Return the first candidate dir that actually contains marker_rel."""
    resolved = _resolve_with_markers((marker_rel,))
    return resolved[0] if resolved else None


def resolve_engine_root() -> Optional[Path]:
    """Return a cannbot-knowledge root that has the (consumer-tier) knowledge-query engine, else None."""
    resolved = _resolve_with_markers(_QUERY_RELS)
    return resolved[0] if resolved else None


def knowledge_query_script() -> Optional[Path]:
    resolved = _resolve_with_markers(_QUERY_RELS)
    return (resolved[0] / resolved[1]) if resolved else None


def knowledge_lint_script() -> Optional[Path]:
    # search ALL candidates for a contributor-tier root with lint — NOT shadowed by a consumer-only root.
    r = _resolve_with_marker(_LINT_REL)
    return (r / _LINT_REL) if r else None


def okf_graph_dir() -> Optional[Path]:
    r = _resolve_with_marker(_GRAPH_REL)
    return (r / _GRAPH_REL).parent if r else None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Locate the external cannbot-knowledge OKF engine.")
    ap.add_argument("what", choices=["root", "query", "lint", "graph-dir"],
                    help="which resolved path to print")
    a = ap.parse_args()
    fn = {"root": resolve_engine_root, "query": knowledge_query_script,
          "lint": knowledge_lint_script, "graph-dir": okf_graph_dir}[a.what]
    val = fn()
    if val is None:
        # distinguish "engine missing entirely" from "contributor tier missing" for a clearer message.
        if a.what in ("lint", "graph-dir") and resolve_engine_root() is not None:
            sys.stderr.write(
                "cannbot-knowledge found but the '%s' tool (contributor tier) is not installed. "
                "Install: /plugin install cannbot-knowledge-contributor-skills@cannbot\n" % a.what)
            return 4
        sys.stderr.write(INSTALL_HINT + "\n")
        return 3
    print(val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
