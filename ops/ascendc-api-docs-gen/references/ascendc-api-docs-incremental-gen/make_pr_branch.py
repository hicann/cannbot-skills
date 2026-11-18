#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""按测试 PR 的 base 建一个「可提 PR」的分支：PR 的代码改动 + 我们生成的文档。

背景：run_by_pr_test 会把 PR 的 before/now 工作区跑一遍生成文档，产物落在
``data/pr_runs_test/pr_<id>/data/outputs/<Api>_new.md``，对应关系记在
``report_<id>.json``（api_name / doc_rel / output / base_sha / head_sha）。

本脚本把这些拼成一个新分支，口径是：

    分支起点 = PR 的 base（merge-base 解析，与生成时同一口径）
    分支内容 = PR 的全部改动（代码/测试/示例/它自己的文档改动）
               + 用「我们生成的文档」覆盖同名 doc_rel

即：代码是 PR 的，文档是模型生成的；两者合成一个新的 PR 头。

用法::

    # 只建本地分支（默认，不 push）
    uv run --no-sync make_pr_branch.py --pr_ids 6620

    # 建完并推到 fork（需要该仓库推送权限）
    uv run --no-sync make_pr_branch.py --pr_ids 6620 --push-url https://gitcode.com/<user>/asc-devkit.git

    # 只看看会改什么
    uv run --no-sync make_pr_branch.py --pr_ids 6620 --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_by_pr_test import (  # noqa: E402
    changed_files_between,
    fetch_pr_refs,
    git,
    resolve_base_by_merge,
    sha_exists,
)

RUNS_DIR = Path(__file__).resolve().parent / "data" / "pr_runs_test"
BRANCH_PREFIX = "docs/ai-gen-"


def _git_identity(key: str) -> str:
    """取本仓配置的提交身份（CLA 校验用的是 commit 邮箱，必须与签署邮箱一致）。"""
    out = subprocess.run(
        ["git", "config", f"user.{key}"], capture_output=True, text=True, encoding="utf-8", errors="replace"
    ).stdout.strip()
    if out:
        return out
    raise SystemExit(f"未配置 git user.{key}：请先 git config user.{key} '...'（提交邮箱需与 CLA 签署邮箱一致）")


def load_report(pr_id: str) -> dict:
    p = RUNS_DIR / f"report_{pr_id}.json"
    if not p.is_file():
        raise SystemExit(f"找不到 {p}，先跑 run_by_pr_test.py --pr_ids {pr_id}")
    data = json.loads(p.read_text(encoding="utf-8"))
    prs = data.get("prs") or []
    if not prs:
        raise SystemExit(f"{p} 里没有 PR 记录")
    return prs[0]


def ensure_shas(main_repo: Path, pr_id: str, base_sha: str, head_sha: str) -> str:
    """确保 base/head 都在本地，返回最终使用的 base_sha。"""
    fetch_pr_refs(main_repo, pr_id, head_sha)
    if base_sha and sha_exists(main_repo, base_sha):
        return base_sha
    print(f"  [warn] base {base_sha[:10]} 不可达，用 merge-base 重算")
    return resolve_base_by_merge(main_repo, head_sha, base_sha or None)


def prepare_worktree(main_repo: Path, worktree: Path, branch: str, base_sha: str) -> None:
    """在 base_sha 上建分支工作区；同名分支/目录先清掉（幂等）。"""
    git(main_repo, "worktree", "prune", check=False)
    if worktree.exists():
        git(main_repo, "worktree", "remove", str(worktree), "--force", check=False)
        if worktree.exists():
            shutil.rmtree(worktree, ignore_errors=True)
    git(main_repo, "branch", "-D", branch, check=False)
    git(main_repo, "worktree", "prune", check=False)
    worktree.parent.mkdir(parents=True, exist_ok=True)
    git(main_repo, "worktree", "add", "-B", branch, str(worktree), base_sha)


def apply_pr_changes(worktree: Path, base_sha: str, head_sha: str, changed: list[str]) -> list[str]:
    """把 PR 的全部改动搬到 base 上（文档也搬，后面再用生成的覆盖）。"""
    done: list[str] = []
    for rel in changed:
        if not (worktree / rel).parent.exists() and not (worktree / rel).exists():
            (worktree / rel).parent.mkdir(parents=True, exist_ok=True)
        exists_at_head = (
            subprocess.run(
                ["git", "-C", str(worktree), "cat-file", "-e", f"{head_sha}:{rel}"],
                capture_output=True,
            ).returncode
            == 0
        )
        if exists_at_head:
            git(worktree, "checkout", head_sha, "--", rel)
        else:  # PR 删除的文件
            git(worktree, "rm", "--quiet", "--ignore-unmatch", rel, check=False)
        done.append(rel)
    return done


def write_generated_docs(pr_dir: Path, worktree: Path, apis: list[dict], main_repo: Path, head_sha: str) -> list[str]:
    """用生成产物覆盖对应文档；返回被覆盖的相对路径。

    产物与文档的对应关系有两个来源：
      1) report 里的 ``api_name -> doc_rel``（权威）；
      2) 目录里存在但 report 没记的 ``<Api>_new.md``（PR 被单接口复跑过，
         report 只剩那一次的结果，全量产物还在）—— 按 ``<Api>.md`` 在 head
         的 docs 树里唯一定位，定不唯一就不猜、跳过。
    """
    outputs = pr_dir / "data" / "outputs"
    doc_by_stem: dict[str, str] = {}
    by_api: dict[str, str] = {}
    for api in apis:
        doc_rel = (api.get("doc_rel") or "").strip()
        name = (api.get("api_name") or "").strip()
        if doc_rel and name:
            by_api[name] = doc_rel
    for rel in git(main_repo, "ls-tree", "-r", "--name-only", head_sha, "--", "docs").splitlines():
        rel = rel.strip()
        if rel.endswith(".md"):
            doc_by_stem.setdefault(rel.rsplit("/", 1)[-1][:-3], []).append(rel)

    written: list[str] = []
    seen: dict[str, str] = {}
    for api in apis:
        out_rel = (api.get("output") or "").strip()
        if api.get("status") != "ok" or not out_rel:
            continue
        src = pr_dir / out_rel
        if not src.is_file():
            print(f"  [warn] {api.get('api_name')}: 产物不存在 {out_rel}，跳过")
            continue
        stem = Path(out_rel).name[:-3]
        stem = stem[:-4] if stem.endswith("_new") else stem
        doc_rel = by_api.get(api.get("api_name") or "") or _unique_doc(doc_by_stem, stem)
        if not doc_rel:
            print(f"  [warn] {api.get('api_name')}: 无法唯一定位文档路径，跳过")
            continue
        if doc_rel in seen:
            print(f"  [warn] {api.get('api_name')}: 与 {seen[doc_rel]} 共用 {doc_rel}，后者被覆盖")
        seen[doc_rel] = api.get("api_name") or stem
        dst = worktree / doc_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        written.append(doc_rel)

    # 产物有、report 没记的（PR 被单接口复跑过）：按 <Api>.md 唯一定位补齐
    if outputs.is_dir():
        known = {(api.get("output") or "").strip() for api in apis}
        for f in sorted(outputs.glob("*_new.md")):
            rel = str(f.relative_to(pr_dir)).replace("\\", "/")
            if rel in known:
                continue
            stem = f.name[:-3][:-4]
            doc_rel = _unique_doc(doc_by_stem, stem)
            if not doc_rel or doc_rel in seen:
                continue
            seen[doc_rel] = stem
            dst = worktree / doc_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
            written.append(doc_rel)
            print(f"  [补] report 未记录，按文件名定位: {f.name} -> {doc_rel}")
    return written


def _unique_doc(doc_by_stem: dict[str, list[str]], stem: str) -> str:
    cands = doc_by_stem.get(stem) or []
    return cands[0] if len(cands) == 1 else ""


def build_commit_message(pr_id: str, title: str, apis: list[dict], base_sha: str, n_code: int, docs: list[str]) -> str:
    names = ", ".join(a.get("api_name", "?") for a in apis if a.get("status") == "ok")
    head = f"docs(api): regenerate {names} docs for #{pr_id}" if names else f"docs(api): regenerate docs for #{pr_id}"
    lines = [
        "",
        f"Source PR: #{pr_id} {title}".rstrip(),
        f"Base (merge-base): {base_sha}",
        f"Code changes carried over unchanged: {n_code} file(s)",
        "",
        "Generated docs (replaces the PR's own doc edits for these paths):",
    ]
    lines += [f"- {d}" for d in docs] or ["- (none)"]
    return head + "\n\n" + "\n".join(lines[1:]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="按 PR base 合成「代码+生成文档」的新分支")
    ap.add_argument("--pr_ids", required=True, help="逗号分隔的 PR id，如 6620 或 6620,5428")
    ap.add_argument("--branch_prefix", default=BRANCH_PREFIX)
    ap.add_argument("--push_url", default="", help="给了就 git push <url> <branch>")
    ap.add_argument("--dry_run", action="store_true", help="只打印计划，不建分支")
    ap.add_argument("--keep_worktree", action="store_true", help="保留分支工作区便于检查")
    args = ap.parse_args()

    main_repo = RUNS_DIR / "main_repo"
    if not (main_repo / ".git").exists():
        raise SystemExit(f"{main_repo} 不存在，先跑一次 run_by_pr_test.py")

    for raw in [x.strip() for x in args.pr_ids.split(",") if x.strip()]:
        pr_id = raw
        print(f"\n{'=' * 70}\nPR #{pr_id}")
        rep = load_report(pr_id)
        head_sha = rep.get("head_sha") or ""
        base_sha = rep.get("base_sha") or ""
        if not head_sha:
            raise SystemExit(f"report_{pr_id}.json 缺 head_sha")
        base_sha = ensure_shas(main_repo, pr_id, base_sha, head_sha)
        changed = changed_files_between(main_repo, base_sha, head_sha)
        if not changed:
            print("  [skip] base..head 无差异")
            continue
        print(f"  base={base_sha[:10]} head={head_sha[:10]} 改动={len(changed)} 个文件")

        pr_dir = RUNS_DIR / f"pr_{pr_id}"
        apis = [a for a in (rep.get("apis") or []) if a.get("status") == "ok"]
        docs = [a["doc_rel"] for a in apis if a.get("doc_rel")]
        print(f"  生成文档 {len(apis)} 篇: {docs}")
        if args.dry_run:
            for a in apis:
                print(f"    - {a.get('api_name')}: {a.get('doc_rel')} ← {a.get('output')}")
            continue

        branch = f"{args.branch_prefix}{pr_id}"
        worktree = pr_dir / "pr_branch"
        prepare_worktree(main_repo, worktree, branch, base_sha)
        apply_pr_changes(worktree, base_sha, head_sha, changed)
        written = write_generated_docs(pr_dir, worktree, apis, main_repo, head_sha)
        if not written:
            print("  [warn] 没有写入任何生成文档，提交可能只是 PR 代码的复制")

        git(worktree, "add", "-A")
        msg = build_commit_message(pr_id, rep.get("title") or "", apis, base_sha, len(changed), written)
        msg_file = pr_dir / "pr_branch_msg.txt"
        msg_file.write_text(msg, encoding="utf-8")
        # 提交身份必须与 CLA 签署邮箱一致：默认沿用本仓 git 配置，
        # 需要覆盖时用 DOCS_BOT_NAME / DOCS_BOT_EMAIL，不要写死机器人身份。
        _name = os.environ.get("DOCS_BOT_NAME") or _git_identity("name")
        _email = os.environ.get("DOCS_BOT_EMAIL") or _git_identity("email")
        git(worktree, "-c", f"user.name={_name}", "-c", f"user.email={_email}", "commit", "-q", "-F", str(msg_file))
        print(f"  提交身份: {_name} <{_email}>")

        stat = git(worktree, "show", "--stat", "--oneline", "-1")
        print(f"  分支 {branch} 已建：{worktree}")
        print("\n".join("  " + ln for ln in stat.strip().splitlines()[:12]))

        if args.push_url:
            print(f"  push -> {args.push_url} ({branch})")
            git(worktree, "push", args.push_url, f"{branch}:{branch}")
            print(f"  已推送；到目标仓库发起 PR（base=master）即可")
        else:
            print(f"  未推送（加 --push_url <fork-url> 才会 push）")

        if not args.keep_worktree:
            print(f"  保留工作区：{worktree}（删除：git worktree remove --force）")


if __name__ == "__main__":
    main()
