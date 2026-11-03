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
"""Validate changelog fragments and aggregate them at release time."""

import argparse
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple

FRAGMENT_DIR = "docs/changelog.d"
CHANGELOG = "CHANGELOG.md"
GENERATED_HEADER = (
    "<!-- GENERATED FILE: 由 scripts/changelog.py 从 docs/changelog.d 聚合；请勿直接修改。 -->"
)
NAME_PATTERN = re.compile(r"(?P<pr>[1-9]\d*)\.(?P<kind>added|fixed|docs|chore)\.md")
DATE_PATTERN = re.compile(r"^### 【(?P<date>\d{4}-\d{2}-\d{2})】", re.MULTILINE)
CATEGORIES = (
    ("added", "新特性 New Features"),
    ("fixed", "缺陷修复 Bug Fixes"),
    ("docs", "文档 Documentation"),
    ("chore", "变更 Changes"),
)


@dataclass(frozen=True)
class Fragment:
    path: Path
    pr_number: int
    kind: str
    text: str


def load_fragments(repo: Path, expected_pr: Optional[int] = None) -> List[Fragment]:
    directory = repo / FRAGMENT_DIR
    if not directory.exists():
        return []
    fragments = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.name == "README.md" or path.name.startswith("."):
            continue
        if not path.is_file():
            raise ValueError(f"fragment 目录只允许文件: {path}")
        match = NAME_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(
                f"fragment 文件名须为 <PR号>.<added|fixed|docs|chore>.md: {path.name}"
            )
        pr_number = int(match.group("pr"))
        if expected_pr is not None and pr_number != expected_pr:
            raise ValueError(f"fragment PR 号 {pr_number} 与当前 PR {expected_pr} 不一致")
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(lines) != 1 or not lines[0].startswith("- "):
            raise ValueError(f"fragment 只能包含一条以 '- ' 开头的变更说明: {path.name}")
        text = lines[0]
        if len(text[2:].strip()) == 0:
            raise ValueError(f"fragment 内容不能为空: {path.name}")
        if len(text[2:].strip()) > 200:
            raise ValueError(f"fragment 正文不得超过 200 字: {path.name}")
        fragments.append(Fragment(path, pr_number, match.group("kind"), text))
    return fragments


def changed_files(repo: Path, base: str, head: str) -> List[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--no-renames",
            "--name-only",
            f"{base}...{head}",
            "--",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return [line for line in result.stdout.splitlines() if line]


def changed_file_statuses(repo: Path, base: str, head: str) -> List[Tuple[str, str]]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--no-renames",
            "--name-status",
            f"{base}...{head}",
            "--",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    changes = []
    for line in result.stdout.splitlines():
        status, filename = line.split("\t", 1)
        changes.append((status, filename))
    return changes


def restore_tree(repo: Path, commit: str, destination: Path) -> None:
    paths = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            "--",
            CHANGELOG,
            FRAGMENT_DIR,
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).splitlines()
    for filename in paths:
        content = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{commit}:{filename}"]
        )
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def check_changed_fragment_paths(files: List[str], expected_pr: Optional[int]) -> None:
    for filename in files:
        if not filename.startswith(f"{FRAGMENT_DIR}/"):
            continue
        name = PurePosixPath(filename).name
        if name == "README.md":
            continue
        match = NAME_PATTERN.fullmatch(name)
        if not match:
            raise ValueError(f"变更的 fragment 文件名不合法: {filename}")
        pr_number = int(match.group("pr"))
        if expected_pr is not None and pr_number != expected_pr:
            raise ValueError(
                f"变更的 fragment PR 号 {pr_number} 与当前 PR {expected_pr} 不一致"
            )


def check(repo: Path, base: str, head: str, expected_pr: Optional[int]) -> None:
    files = changed_files(repo, base, head)
    if CHANGELOG in files:
        raise ValueError("禁止直接修改 CHANGELOG.md；请新增 docs/changelog.d fragment")
    check_changed_fragment_paths(files, expected_pr)
    fragments = load_fragments(repo)
    print(f"fragment 校验通过：{len(fragments)} 个；CHANGELOG.md 未被直接修改")


def build_section(fragments: List[Fragment], release_date: str) -> str:
    lines = [f"### 【{release_date}】"]
    for kind, heading in CATEGORIES:
        entries = [fragment for fragment in fragments if fragment.kind == kind]
        if not entries:
            continue
        lines.extend(["", f"#### {heading}"])
        lines.extend(
            fragment.text
            for fragment in sorted(entries, key=lambda item: item.pr_number)
        )
    return "\n".join(lines) + "\n"


def aggregate(repo: Path, release_date: str) -> bool:
    date.fromisoformat(release_date)
    fragments = load_fragments(repo)
    if not fragments:
        print("没有待聚合 fragment，无需修改")
        return False
    changelog = repo / CHANGELOG
    original = changelog.read_text(encoding="utf-8")
    if release_date in DATE_PATTERN.findall(original):
        raise ValueError(f"日期 {release_date} 已存在，拒绝重复聚合")
    text = original
    if GENERATED_HEADER not in text:
        text = f"{GENERATED_HEADER}\n\n{text}"
    section = build_section(fragments, release_date)
    match = DATE_PATTERN.search(text)
    if match:
        text = f"{text[:match.start()]}{section}\n{text[match.start():]}"
    else:
        text = f"{text.rstrip()}\n\n{section}"
    changelog.write_text(text, encoding="utf-8")
    for fragment in fragments:
        fragment.path.unlink()
    print(f"已聚合 {len(fragments)} 个 fragment 到 {CHANGELOG}（{release_date}）")
    return True


def verify_release(repo: Path, base: str, head: str) -> None:
    changes = changed_file_statuses(repo, base, head)
    for status, filename in changes:
        if filename != CHANGELOG and not filename.startswith(f"{FRAGMENT_DIR}/"):
            raise ValueError(f"聚合 PR 含范围外改动: {filename}")
        if filename.startswith(f"{FRAGMENT_DIR}/") and status != "D":
            raise ValueError(f"聚合 PR 只能删除已消费 fragment: {filename}")
    if ("M", CHANGELOG) not in changes:
        raise ValueError("聚合 PR 必须修改 CHANGELOG.md")
    deleted = {
        filename
        for status, filename in changes
        if status == "D" and filename.startswith(f"{FRAGMENT_DIR}/")
    }
    if not deleted:
        raise ValueError("聚合 PR 至少要删除一个已消费 fragment，不能裸改 CHANGELOG.md")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base_repo = root / "base"
        head_repo = root / "head"
        restore_tree(repo, base, base_repo)
        restore_tree(repo, head, head_repo)
        if load_fragments(head_repo):
            raise ValueError("聚合 PR 仍包含未消费 fragment")
        base_text = (base_repo / CHANGELOG).read_text(encoding="utf-8")
        head_text = (head_repo / CHANGELOG).read_text(encoding="utf-8")
        match = DATE_PATTERN.search(head_text)
        if not match:
            raise ValueError("聚合后的 CHANGELOG.md 缺少日期标题")
        release_date = match.group("date")
        if release_date in DATE_PATTERN.findall(base_text):
            raise ValueError(f"日期 {release_date} 已存在于基线")

        base_fragments = load_fragments(base_repo)
        expected_deleted = {
            fragment.path.relative_to(base_repo).as_posix()
            for fragment in base_fragments
        }
        if deleted != expected_deleted:
            raise ValueError("删除的 fragment 集合与基线待聚合集合不一致")
        aggregate(base_repo, release_date)
        expected = (base_repo / CHANGELOG).read_text(encoding="utf-8")
    if expected != head_text:
        raise ValueError("CHANGELOG.md 与基线重放聚合结果不一致")
    print(f"聚合 PR 校验通过：{release_date}，{len(deleted)} 个 fragment")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("check", help="校验 PR fragments 与 CHANGELOG 禁改")
    check_parser.add_argument("--base", required=True)
    check_parser.add_argument("--head", required=True)
    check_parser.add_argument("--pr-number", type=int)
    release_parser = subparsers.add_parser(
        "verify-release", help="重放基线聚合并校验发版 PR"
    )
    release_parser.add_argument("--base", required=True)
    release_parser.add_argument("--head", required=True)
    validate_parser = subparsers.add_parser("validate", help="仅校验当前 fragments")
    validate_parser.add_argument("--pr-number", type=int)
    aggregate_parser = subparsers.add_parser(
        "aggregate", help="聚合 fragments 并清空目录"
    )
    aggregate_parser.add_argument("--date", default=date.today().isoformat())
    for subparser in (check_parser, release_parser, validate_parser, aggregate_parser):
        subparser.add_argument("--repo", default=".")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    try:
        if args.command == "check":
            check(repo, args.base, args.head, args.pr_number)
        elif args.command == "verify-release":
            verify_release(repo, args.base, args.head)
        elif args.command == "validate":
            fragments = load_fragments(repo, args.pr_number)
            print(f"fragment 校验通过：{len(fragments)} 个")
        else:
            aggregate(repo, args.date)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
