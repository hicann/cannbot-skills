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
"""Markdown intra-repo anchor link validator (rule ANCHOR).

Guards the class of defect fixed in PR #915: a markdown link
``[text](file.md#anchor)`` / ``[text](#anchor)`` / ``[id]: file.md#anchor``
whose ``#anchor`` no longer matches any heading in the target .md file,
because a heading was renamed/restructured and no back-reference was updated.

Anchors are generated with the GitHub/GitCode autolink-heading slug rules so
the check matches what the renderer actually produces:
  * lowercase
  * drop every character that is not a word char, whitespace, or hyphen
  * trim, then replace EACH whitespace char with a hyphen (no collapsing:
    ``a  b`` -> ``a--b``)
  * CJK letters/digits preserved
  * duplicate headings get ``-1``, ``-2`` ... suffixes (first has no suffix)

Scope (kept tight to the #915 defect, to stay precise and avoid noise):
  * only links that carry a non-empty ``#fragment`` are checked
  * target must be empty (same file) or resolve to an EXISTING .md file
    inside the repo root; missing files / non-.md targets / external URLs
    are skipped (those are a different guard)
  * images ``![](...)`` and links inside fenced code, inline code, and HTML
    comments are not parsed

Output: one JSON object per line to stdout, then a summary object. Exit 1 if
any error, else 0. Consumed by tests/unit/test-markdown-anchors.sh.
"""

import difflib
import json
import logging
import os
import posixpath
import re
import sys

logger = logging.getLogger("anchor_validator")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def _emit(obj):
    logger.info(json.dumps(obj, ensure_ascii=False))

EXCLUDE_DIRS = {".git", "node_modules", "__pycache__", ".version-state"}

SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")
HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+(.*?))?\s*$")
OPEN_FENCE_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
SETEXT_RE = re.compile(r"^ {0,3}(=+|-+)\s*$")
STRUCTURAL_RE = re.compile(r"^\s{0,3}([-+*>]|\d+\.|\||>|#)")
LINK_RE = re.compile(r"(?<!\!)\[([^\]]*)\]\(\s*([^)]*?)\s*\)")
REFDEF_RE = re.compile(
    r'^\s*\[([^\]]+)\]:\s*(\S+)(?:\s+(?:"[^"]*"|\'[^\']*\'|\([^)]*\)))?\s*$'
)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
CODESPAN_RE = re.compile(r"`([^`]*?)`")
EMPH_RE = re.compile(r"\*\*|__|\*")
HTML_TAG_RE = re.compile(
    r"</?(a|abbr|address|article|aside|b|bdi|bdo|blockquote|body|br|button|"
    r"caption|cite|code|col|colgroup|data|dd|del|details|dfn|dialog|div|dl|dt|"
    r"em|fieldset|figcaption|figure|footer|form|h[1-6]|head|header|hr|html|i|"
    r"iframe|img|input|ins|kbd|label|legend|li|main|map|mark|meter|nav|"
    r"noscript|object|ol|optgroup|option|output|p|param|picture|pre|progress|"
    r"q|rp|rt|ruby|s|samp|script|section|select|slot|small|source|span|strong|"
    r"style|sub|summary|sup|svg|table|tbody|td|template|tfoot|th|thead|time|"
    r"title|tr|track|u|ul|var|video|wbr)\b[^>]*>",
    re.IGNORECASE,
)
HTML_ID_RE = re.compile(
    r"""<[a-zA-Z][^>]*?\bid\s*=\s*(?:"([^"]+)"|'([^']+)')""",
    re.IGNORECASE,
)
HTML_A_NAME_RE = re.compile(
    r"""<a\b[^>]*?\bname\s*=\s*(?:"([^"]+)"|'([^']+)')""",
    re.IGNORECASE,
)

REMOVE_RE = re.compile(r"[^\w\s-]|[\u00b2\u00b3\u00b9\u2070-\u209f]", re.UNICODE)
WS_RE = re.compile(r"\s")


def slugify(text):
    s = text.lower()
    s = REMOVE_RE.sub("", s)
    s = s.strip()
    s = WS_RE.sub("-", s)
    return s


def strip_heading_inline(text):
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = CODESPAN_RE.sub(r"\1", text)
    text = EMPH_RE.sub("", text)
    text = HTML_TAG_RE.sub("", text)
    return text


class MdFile:
    __slots__ = ("rel", "anchors", "refdefs", "links")

    def __init__(self, rel):
        self.rel = rel
        self.anchors = set()
        self.refdefs = {}
        self.links = []

    def add_heading(self, text):
        clean = strip_heading_inline(text)
        if not clean.strip():
            return
        base = slugify(clean)
        if not base:
            return
        if base not in self.anchors:
            self.anchors.add(base)
            return
        n = 1
        cand = "%s-%d" % (base, n)
        while cand in self.anchors:
            n += 1
            cand = "%s-%d" % (base, n)
        self.anchors.add(cand)


def iter_md_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn.lower().endswith(".md"):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                yield full, rel


def parse_md(full, rel):
    md = MdFile(rel)
    try:
        with open(full, encoding="utf-8") as f:
            raw = f.read()
    except UnicodeDecodeError:
        with open(full, encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError:
        return md

    raw = HTML_COMMENT_RE.sub(lambda m: re.sub(r"[^\n]", "", m.group(0)), raw)
    _scan_lines(md, raw.split("\n"))
    return md


def _scan_lines(md, lines):
    fence_char = None
    fence_len = 0
    prev_content = None
    for i, line in enumerate(lines, 1):
        in_fence = fence_char is not None
        if not in_fence:
            om = OPEN_FENCE_RE.match(line)
            if om and not (om.group(2)[0] == "`" and "`" in om.group(3)):
                fence_char = om.group(2)[0]
                fence_len = len(om.group(2))
                prev_content = None
                continue
        else:
            close_re = re.compile(
                r"^ {0,3}" + re.escape(fence_char)
                + "{" + str(fence_len) + r",}\s*$"
            )
            if close_re.match(line):
                fence_char, fence_len = None, 0
            prev_content = None
            continue

        if prev_content is not None and SETEXT_RE.match(line):
            md.add_heading(
                strip_heading_inline(prev_content.rstrip("#").strip())
            )
            prev_content = None
            continue

        prev_content = _process_line(md, line, i, prev_content)


def _process_line(md, line, i, prev_content):
    for m in HTML_ID_RE.finditer(line):
        v = m.group(1) or m.group(2)
        if v:
            md.anchors.add(v)
    for m in HTML_A_NAME_RE.finditer(line):
        v = m.group(1) or m.group(2)
        if v:
            md.anchors.add(v)

    hm = HEADING_RE.match(line)
    if hm:
        ht = hm.group(1)
        if ht is not None:
            md.add_heading(ht.rstrip("#").strip())
        return None

    rdm = REFDEF_RE.match(line)
    if rdm:
        md.refdefs[rdm.group(1)] = (rdm.group(2), i)
        return None

    scan = CODESPAN_RE.sub("", line)
    for lm in LINK_RE.finditer(scan):
        md.links.append((i, lm.group(2)))

    if line.strip() and not SETEXT_RE.match(line) \
            and not STRUCTURAL_RE.match(line):
        return line
    return None


def is_external(path):
    if path == "":
        return False
    if path.startswith("//"):
        return True
    if SCHEME_RE.match(path) and not path.lower().endswith(".md"):
        return True
    return False


def parse_target(raw):
    raw = raw.strip()
    url = raw.split(None, 1)[0] if raw else ""
    if "#" in url:
        p, f = url.split("#", 1)
    else:
        p, f = url, None
    return p, f


def resolve(link_rel, path):
    if path == "":
        return link_rel
    if path.startswith("./"):
        path = path[2:]
    if path.startswith("/"):
        target = posixpath.normpath(path.lstrip("/"))
    else:
        link_dir = posixpath.dirname(link_rel)
        target = posixpath.normpath(posixpath.join(link_dir, path)) if link_dir else posixpath.normpath(path)
    if target == "" or target.startswith("..") or target.startswith("/"):
        return None
    if not target.lower().endswith(".md"):
        return None
    return target


def suggest(fragment, anchors):
    if not anchors:
        return ""
    best = difflib.get_close_matches(fragment, sorted(anchors), n=1, cutoff=0.6)
    return best[0] if best else ""


def _check_link(link_rel, line, raw, files):
    path, frag = parse_target(raw)
    if frag is None or frag == "":
        return 0, 0
    if is_external(path):
        return 0, 0
    target_rel = resolve(link_rel, path)
    if target_rel is None or target_rel not in files:
        return 0, 0
    tanchors = files[target_rel].anchors
    if frag in tanchors or frag.lower() in tanchors:
        return 1, 0
    obj = {
        "level": "error",
        "rule": "ANCHOR",
        "file": link_rel,
        "line": line,
        "target": target_rel,
        "anchor": frag,
        "raw": raw,
    }
    s = suggest(frag, tanchors)
    if s:
        obj["suggest"] = s
    _emit(obj)
    return 1, 1


def main():
    if len(sys.argv) < 2:
        _emit({"level": "error", "rule": "ANCHOR",
               "msg": "usage: markdown_anchor_validator.py <repo-root>"})
        return 2
    root = os.path.abspath(sys.argv[1])

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass  # stdout does not support reconfigure (non-CPython or pipe)

    files = {}
    for full, rel in iter_md_files(root):
        files[rel] = parse_md(full, rel)

    errors = 0
    checked = 0

    for rel, md in files.items():
        for line, raw in md.links:
            c, e = _check_link(rel, line, raw, files)
            checked += c
            errors += e
        for _id, (url, line) in md.refdefs.items():
            c, e = _check_link(rel, line, url, files)
            checked += c
            errors += e

    _emit({
        "summary": {
            "md_files": len(files),
            "links_checked": checked,
            "errors": errors,
        }
    })
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
