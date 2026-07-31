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

"""Cut-tag / VERSION.md banner consistency gate (pure stdlib).

Origin: 2026-07-23 — the internal `v3.17.0` tag was pushed while VERSION.md's
banner still read `V3.16.1`, i.e. a release tag was cut WITHOUT its
release-notes entry. The omission was discoverable only via commit archaeology.
VERSION.md (repo root) is the canonical release-notes source of truth; this
module binds an internal version tag to that banner so the mismatch is caught
at push time (see .githooks/pre-push).

Design: FAIL-SAFE. We block ONLY a confirmed contradiction (a version tag whose
version differs from a parseable banner). Anything we cannot read or parse is a
no-op pass — inability to verify never blocks a push.

CLI:
    python3 src/scripts/check_tag_version_consistency.py <tag> [<version_md_path>]
    exit 0 if ok (or not our concern / unverifiable), exit 1 on a confirmed
    tag-vs-banner mismatch.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Banner line: "## Current version: **V3.17.0** (2026-07-23)". We accept an
# optional leading V/v and 2- or 3-segment versions. Case-insensitive on the
# "Current version" label; tolerant of surrounding whitespace.
_BANNER_RE = re.compile(
    r"^\s*##\s*Current version:\s*\*\*\s*[vV]?(\d+\.\d+(?:\.\d+)?)\s*\*\*",
    re.MULTILINE,
)

# A version tag: v-prefixed 2- or 3-segment dotted number. May arrive as a bare
# name (`v3.17.0`) or a full ref (`refs/tags/v3.17.0`).
_VERSION_TAG_RE = re.compile(r"^v(\d+\.\d+(?:\.\d+)?)$")


def banner_version(version_md_text: str) -> str | None:
    """Return the version from the VERSION.md banner, normalized WITHOUT the
    leading V (e.g. "3.17.0"), or None if no banner is found.
    """
    if not version_md_text:
        return None
    m = _BANNER_RE.search(version_md_text)
    if not m:
        return None
    return m.group(1)


def tag_to_version(tag_ref_or_name: str) -> str | None:
    """Given `refs/tags/v3.17.0` or `v3.17.0`, return "3.17.0". Return None if
    the ref is NOT a version tag (non-version tags, branch refs, garbage).
    """
    if not tag_ref_or_name:
        return None
    name = tag_ref_or_name.strip()
    # Strip a refs/tags/ prefix; a refs/heads/ (branch) ref will not match the
    # version-tag pattern below, so it correctly falls through to None.
    prefix = "refs/tags/"
    if name.startswith(prefix):
        name = name[len(prefix):]
    m = _VERSION_TAG_RE.match(name)
    if not m:
        return None
    return m.group(1)


def _version_tuple(version: str) -> tuple[int, int, int]:
    """Normalize "3.17" / "3.17.0" to a (major, minor, patch) tuple, missing
    patch treated as 0 — so `v3.17` and `V3.17.0` compare equal."""
    parts = version.split(".")
    while len(parts) < 3:
        parts.append("0")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def check(tag: str, version_md_text: str) -> tuple[bool, str]:
    """Return (ok, message). FAIL-SAFE — block ONLY a confirmed mismatch.

    - tag is not a version tag            -> (True,  no-op)
    - banner missing / unparseable        -> (True,  could-not-verify note)
    - tag version == banner version       -> (True,  match)
    - tag version != banner version       -> (False, names both — the block)
    """
    tag_ver = tag_to_version(tag)
    if tag_ver is None:
        return True, f"'{tag}' is not a version tag; nothing to check."

    banner_ver = banner_version(version_md_text)
    if banner_ver is None:
        return True, (
            f"tag {tag} is v{tag_ver} but VERSION.md has no parseable "
            f"'## Current version: **V...**' banner — could not verify "
            f"(fail-safe: not blocking)."
        )

    if _version_tuple(tag_ver) == _version_tuple(banner_ver):
        return True, (
            f"OK: tag v{tag_ver} matches VERSION.md banner V{banner_ver}."
        )

    return False, (
        f"MISMATCH: tag version v{tag_ver} != VERSION.md banner V{banner_ver}. "
        f"The tag is being cut without its release-notes entry. Run "
        f"/aog-version-bump to add the VERSION.md row for v{tag_ver} BEFORE "
        f"cutting/pushing the tag."
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "usage: check_tag_version_consistency.py <tag> [<version_md_path>]",
            file=sys.stderr,
        )
        return 2
    tag = argv[0]
    if len(argv) >= 2:
        version_md_path = Path(argv[1])
    else:
        # Default: repo-root VERSION.md (this file lives at src/scripts/).
        version_md_path = Path(__file__).resolve().parents[2] / "VERSION.md"

    try:
        text = version_md_path.read_text(encoding="utf-8")
    except OSError as exc:
        # Fail-safe: cannot read the file -> cannot confirm a contradiction.
        print(
            f"could not read {version_md_path}: {exc} — not blocking "
            f"(fail-safe).",
            file=sys.stderr,
        )
        return 0

    ok, message = check(tag, text)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
