#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
MD-01: Markdown relative-link validator.

Scans every Markdown file under the scan root and verifies that each relative
link target (inline links, images and reference-style definitions) resolves
to an existing file or directory on disk.

The default scan root is the ascendc-port-orchestrator plugin directory
(auto-detected from this script's location); pass --root to scan the whole
repository instead.

Rules:
  - Skipped: absolute URLs (http/https/mailto/ftp), pure in-page anchors
    (#...), autolinks (<...>), and template placeholders (targets containing
    '{', '<', regex-like patterns such as '[^/]+', or bare '...').
  - URL-escaped characters (%20 etc.) are decoded before resolving.
  - Fragment (#anchor) and query (?...) parts are stripped; only the path
    must exist. Anchor *validity* inside the target file is not checked.
  - Links inside fenced code blocks and inline code spans are ignored.
  - Root-relative links (/...) resolve against the scan root.

Exit code: 0 when no broken links are found, 1 otherwise.
"""

import argparse
import json
import logging
import os
import re
import sys
import urllib.parse
from collections import Counter

logger = logging.getLogger("md_link_validator")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

EXCLUDE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".version-state",
    "CI",
    "asc-devkit",
    "pypto",
    "cann-recipes-infer",
    "operators",
    ".opencode",
    ".claude",
    ".trae",
    ".marscode",
    ".traecli",
    ".cursor",
}

# Path components whose Markdown files are test data, not documentation
# (e.g. engine test fixtures that intentionally reference nonexistent paths).
EXCLUDE_PATH_PARTS = {"fixtures"}

SKIP_SCHEMES = ("http://", "https://", "mailto:", "ftp://", "tel:", "data:", "skill:")

# Inline link or image: [text](url "optional title") / ![alt](url)
INLINE_LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*([^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\s*\)")
# Reference-style link definition: [label]: url
# Footnote definitions ([^label]: text) are NOT links and are excluded.
REF_DEF_RE = re.compile(r"^[ ]{0,3}\[([^\]]+)\]:\s*(\S+)", re.MULTILINE)
# Fenced code block (``` or ~~~), allowing up to 3 leading spaces per CommonMark
FENCE_RE = re.compile(r"^[ ]{0,3}(```|~~~)", re.MULTILINE)
# Inline code span
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


def strip_ignored_spans(text):
    """Blank out fenced code blocks and inline code spans (keep line count)."""
    lines = text.split("\n")
    in_fence = False
    fence_marker = None
    for i, line in enumerate(lines):
        m = FENCE_RE.match(line)
        if m:
            marker = m.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
            lines[i] = ""
        elif in_fence:
            lines[i] = ""
    text = "\n".join(lines)
    return INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), text)


def is_placeholder(target):
    """Template/regex placeholders that can never resolve on disk."""
    if target in (".", "..", "...", "…") or set(target) <= {".", "…"}:
        return True
    if any(c in target for c in "{}<>"):
        return True
    # Regex-like patterns used in docs, e.g. [^/]+
    if target.startswith("[") and ("]" in target):
        return True
    return False


def classify_target(raw_url):
    """Return None to skip, or the decoded filesystem path to check."""
    url = raw_url.strip().strip("<>")
    if not url or url.startswith("#"):
        return None
    if url.lower().startswith(SKIP_SCHEMES):
        return None
    path = url.split("#", 1)[0].split("?", 1)[0]
    if not path:
        return None
    path = urllib.parse.unquote(path)
    if is_placeholder(path):
        return None
    return path


def _broken_in_file(root, dirpath, abs_path, rel_path):
    """Yield (rel_file, line_no, raw_url, resolved_path) for each broken link in one file."""
    try:
        with open(abs_path, encoding="utf-8") as f:
            raw_text = f.read()
    except (OSError, UnicodeDecodeError):
        return
    text = strip_ignored_spans(raw_text)
    candidates = []  # (line_no, raw_url)
    for m in INLINE_LINK_RE.finditer(text):
        candidates.append((text.count("\n", 0, m.start()) + 1, m.group(1)))
    for m in REF_DEF_RE.finditer(text):
        if m.group(1).startswith("^"):
            continue  # footnote definition, not a link
        candidates.append((text.count("\n", 0, m.start()) + 1, m.group(2)))
    for line_no, raw_url in candidates:
        path = classify_target(raw_url)
        if path is None:
            continue
        if path.startswith("/"):
            resolved = os.path.normpath(os.path.join(root, path.lstrip("/")))
        else:
            resolved = os.path.normpath(os.path.join(dirpath, path))
        if not os.path.exists(resolved):
            yield (rel_path, line_no, raw_url, os.path.relpath(resolved, root))


def find_broken(root):
    """Yield (rel_file, line_no, raw_url, resolved_path) for each broken link."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        for fn in sorted(filenames):
            if not fn.endswith(".md"):
                continue
            abs_path = os.path.join(dirpath, fn)
            rel_path = os.path.relpath(abs_path, root)
            if EXCLUDE_PATH_PARTS & set(rel_path.split(os.sep)):
                continue
            yield from _broken_in_file(root, dirpath, abs_path, rel_path)


def main():
    parser = argparse.ArgumentParser(description="MD-01: validate relative links in Markdown files")
    parser.add_argument("--root", default=None,
                        help="scan root (default: the plugin directory, auto-detected from script location)")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--summary-only", action="store_true", help="only print the per-area summary")
    args = parser.parse_args()

    if args.root:
        root = os.path.abspath(args.root)
    else:
        root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    broken = list(find_broken(root))

    if args.format == "json":
        logger.info(json.dumps([
            {"file": f, "line": ln, "url": url, "resolved": res}
            for f, ln, url, res in broken
        ], ensure_ascii=False, indent=2))
    else:
        by_area = Counter(f.split(os.sep)[0] for f, _, _, _ in broken)
        if not args.summary_only:
            for f, ln, url, res in broken:
                logger.info(f"{f}:{ln}: broken link [{url}] -> {res}")
        logger.info(f"\nScanned root: {root}")
        logger.info(f"Broken links: {len(broken)}")
        for area, n in by_area.most_common():
            logger.info(f"  {area}: {n}")

    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
