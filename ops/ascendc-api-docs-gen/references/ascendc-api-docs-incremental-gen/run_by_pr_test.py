#!/usr/bin/env python3
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

"""按 gitcode PR 测试文档生成：PR 代码改动 → 检测受影响 API → 自动生成新文档 markdown。

针对 cann/asc-devkit 的 PR 做文档生成测试（沿用早期一致性脚本的整体骨架）。

目标：对每个 PR，
  1. 通过 gitcode API（不可用时降级 git ls-remote）获取 PR head.sha，
     base（分叉点）一律用 git merge-base 解析 —— gitcode API 的 base.sha
     实测是 master 当前动态 tip（open/merged 均如此），做 base 会让 diff
     混入 master 上数千个无关合入文件
  2. git worktree 分别挂载 original（base）与 modified（head）工作区
  3. 调 src/api_change_detect.pipeline.detect_api_changes_lightweight 检测受影响 API
     （PR 无代码改动时，回退用 PR 改动的 docs/**.md 文档名作受影响 API）
  4. 对每个受影响 API 的主文档，subprocess 调用 src/run_all_cli.py 对比 before/now，
     自动生成更新后的文档 markdown 并收集到 outputs 目录

与早期一致性脚本的差异（修复「所有 API 被 skip」的根源）：
  1. 不做「.md 文件名相似度选 1 个」过滤 —— 原 select_most_similar_api 在
     PR 只改代码（无 .md 改动、title 无 .md 文件名）时返回空 dict，导致全部 skip。
     本脚本直接跑 detect 检出的全部受影响 API（detect 本身已按代码改动做关联）。
  2. 不用 base_sha 检查 include/ 声明 —— 新增接口在 base 里没有声明，
     `has_include_decl(base)` 会把新增接口全部 skip。改用 head_sha 检查
     （只用于提前排除聚合头，避免 run_all_cli 内部 ValueError）。
  3. before 缺文档不再 skip —— PR 新增文档（before 无、now 有）时，
     用 now 的文档复制到 before 同路径做占位，让流水线正常跑
     （变更检测对比的是两侧代码快照，不受文档占位影响）。
  4. fetch PR head 用 refs/pull/<id>/head —— PR head 常在贡献者 fork 仓库，
     从 cann/asc-devkit 直接 fetch head_sha 会失败；refs/pull/<id>/head 是
     主仓上的正式 ref，fork 场景同样有效。
  5. LLM 凭证直接继承 os.environ —— 不再读 ~/.zshrc（Windows 无此文件）。

测试 PR 列表（内置，--pr_ids 可过滤）：
  1. PR 6819 - tensor api新增gather/iota/pack/unpack/compress/expand接口
  2. PR 5826 - asc_muls_v2代码中接口废弃但是资料没同步
  3. PR 5091 - sc_copy_l0c2l1新增接口重载但未同步新增资料
  4. PR 5557 - 提供统一的asc_cast接口
  5. PR 4970 - 变更了同步控制接口，通过if判断限定了源流出、目的流水只能取特定组合，
     但相应的资料并没有列出这些有效组合

输出：data/pr_runs_test/ 下：
  - main_repo/         临时 git 仓库（fetch master 历史 + 各 PR head）
  - pr_<id>/           每个 PR 一个工作区
    - before/          original 工作区（base.sha）
    - now/             modified 工作区（head.sha）
    - data/outputs/    生成的 {api}_new.md 文档产物（按算子名）
  - report.json / report.md   汇总报告（机读 + 人读）

用法：
  # dry_run 全部：fetch + worktree + 检测受影响 API（不跑文档生成）
  uv run run_by_pr_test.py --dry_run
  # 只跑指定 PR（含文档生成）
  uv run run_by_pr_test.py --pr_ids 6819
  # 保留 worktree 便于检查
  uv run run_by_pr_test.py --pr_ids 5826 --keep_worktrees

依赖：uv run 会使用项目虚拟环境（脚本本体只用标准库）；
  run_all_cli 的 LLM 凭证（ALIYUN_*/OPENAI_* 等）需已在环境变量中。
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_INSTANCE_LOCK = None  # 进程存活期持有，退出/被杀时锁自动释放


def acquire_instance_lock(runs_dir: Path) -> None:
    """单实例互斥：同一 runs_dir 只允许一个脚本实例运行。

    并发实例会在 setup_worktrees 的幂等清理里互相删除对方正在使用的
    worktree（实测：后启动实例 rm 掉先启动实例的 before/，其 run_all_cli
    子进程随即 FileNotFoundError）。用锁文件 + 独占字节锁实现：
    Windows 用 msvcrt.locking / POSIX 用 fcntl.flock，锁随进程死亡自动
    释放，无残留死锁。
    """
    global _INSTANCE_LOCK
    runs_dir.mkdir(parents=True, exist_ok=True)
    lock_path = runs_dir / ".instance.lock"
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        try:
            fh.seek(0)
            pid = fh.read(64).strip()
        except OSError:
            pid = ""  # 锁区域不可读（被持锁实例锁定），PID 未知
        fh.close()
        raise SystemExit(
            f"[pr_test] 已有实例正在运行 (PID={pid or '未知'}, 锁: {lock_path})。\n"
            f"[pr_test] 并发运行会互相破坏 worktree，请等待其结束、或终止旧实例后再重跑。"
        )
    fh.seek(0)
    fh.truncate()
    fh.write(str(os.getpid()))
    fh.flush()
    _INSTANCE_LOCK = fh


def _setup_win_console() -> None:
    """Windows 下统一输出编码为 UTF-8 并启用 VT 转义。

    Python 对管道/重定向输出默认跟随 locale（中文 Windows = GBK/cp936），
    而现代终端（VS Code / Windows Terminal / PowerShell 7）按 UTF-8 解码，
    中文会变乱码。stdout 为真实控制台时 Python 走 WriteConsoleW（宽字符，
    无编码问题），reconfigure 对其无副作用，统一设置无害。
    os.system("") 是启用旧版 conhost VT 转义序列支持的标准技巧
    （新终端默认已开启，重复设置无害）。
    """
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass  # 个别包装流不支持 reconfigure，保持原状
    os.system("")  # noqa: S605  启用 VT（空命令副作用）


_setup_win_console()

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_ALL_CLI = PROJECT_ROOT / "src" / "run_all_cli.py"
# 当前 PR 的改动清单文件（每行一个相对路径），由 run_one_pr 写入，
# run_all_cli() 拼进子进程参数，供快照补拷（R1/R2/R3）使用。
_CURRENT_CHANGED_FILES: Path | None = None
REPO_URL = "https://gitcode.com/cann/asc-devkit.git"
API_BASE = "https://gitcode.com/api/v5/repos/cann/asc-devkit/pulls"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "data" / "pr_runs_test"
# fetch master 历史的起始日期（需覆盖 PR 分叉点；不够时 resolve_base_by_merge
# 会自动加深，但首次就给足可省去加深开销）
DEFAULT_SINCE = "2026-05-01"

# 内置测试 PR 列表（用户指定 + 一句话描述，描述写入报告便于人工核对场景覆盖）
PR_TEST_LIST: list[dict] = [
    {"pr_id": "6819", "desc": "tensor api新增gather/iota/pack/unpack/compress/expand接口"},
    # {"pr_id": "5826", "desc": "asc_muls_v2代码中接口废弃但是资料没同步"},
    {"pr_id": "5091", "desc": "sc_copy_l0c2l1新增接口重载但未同步新增资料"},
    # {"pr_id": "5557", "desc": "提供统一的asc_cast接口"},  # 不在业务范围
    {
        "pr_id": "4970",
        "desc": "变更了同步控制接口，通过if判断限定了源流出、目的流水只能取特定组合，"
        "但相应的资料并没有列出这些有效组合",
    },
    {"pr_id": "5559", "desc": "新增mmad、mmadmx支持传入bias的接口及其对应资料，资料为增量修改"},
    {"pr_id": "5428", "desc": "Load和Store接口支持RegTraitNumTwo"},
    {"pr_id": "5081", "desc": "修改HF32模式支持的枚举值的命名和资料描述"},
    {"pr_id": "6512", "desc": "feat: 为950产品新增asc_dcci_entire_ub接口支持并修改资料中对应的约束"},
    {"pr_id": "6620", "desc": "为basic_api的SetLoopModePara/ResetLoopModePara增加OUT_TO_L1支持"},
]


# ---------------------------------------------------------------------------
# Python 层 stdout/stderr 双写（tee）：包装 sys.stdout/sys.stderr，print 与
# 日志输出同时写终端和日志文件（日志为 UTF-8 且剥 ANSI 转义）。
#
# 不用 fd 级 os.dup2 重定向：Windows 真实控制台下 sys.stdout 是
# _WindowsConsoleIO（创建时缓存控制台句柄），dup2 重定向 fd 后输出链路
# 不可控（实测输出会被静默吞掉：终端与日志都收不到）。macOS 参考脚本的
# fd-tee 方案在此不可移植，故统一走 Python 层包装。
# 子进程（run_all_cli）的输出用 stdout=PIPE 逐行读取后经 print 双写。
# ---------------------------------------------------------------------------
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?(?:\x07|\x1b\\)")
_log_fh = None
_tee_lock = threading.Lock()


class _TeeStream:
    """把写入同时转发到原 stream 与日志文件（日志剥 ANSI、UTF-8）。"""

    def __init__(self, stream, log_fh):
        self._stream = stream
        self._log = log_fh

    def write(self, s: str) -> int:
        with _tee_lock:
            try:
                self._stream.write(s)
                self._stream.flush()
            except OSError:
                pass  # 终端写失败（如 console 句柄异常）不阻断日志
            try:
                self._log.write(_ANSI_RE.sub("", s))
                self._log.flush()
            except OSError:
                pass
        return len(s)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except OSError:
            pass

    # 委托常用属性：rich/questionary 等库会探测 isatty/fileno/encoding
    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def _setup_python_tee(log_path: Path) -> None:
    """把 sys.stdout/sys.stderr 包成终端+日志双写。日志打不开时降级为不 tee。"""
    global _log_fh
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
    except OSError as e:
        print(f"[pr_test] 警告: 日志文件打开失败（{e}），仅输出到终端", flush=True)
        return
    sys.stdout = _TeeStream(sys.stdout, _log_fh)
    sys.stderr = _TeeStream(sys.stderr, _log_fh)


def log(msg: str) -> None:
    print(f"[pr_test] {msg}", flush=True)


# ---------------------------------------------------------------------------
# gitcode API：单 PR 元信息（base.sha / head.sha / title / state）
# ---------------------------------------------------------------------------


def _ls_remote_sha(ref_pattern: str) -> str:
    """git ls-remote 拿指定 ref 的 sha（不走 gitcode API 额度）。

    Raises:
        RuntimeError: ref 不存在或 git 失败
    """
    result = subprocess.run(
        ["git", "ls-remote", REPO_URL, ref_pattern],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-remote 失败: {result.stderr[:300]}")
    for ln in result.stdout.splitlines():
        sha, _, ref = ln.partition("\t")
        if ref.strip() and sha.strip():
            return sha.strip()
    raise RuntimeError(f"ls-remote 未找到 ref: {ref_pattern}")


def _fetch_pr_meta_via_api(pr_id: str) -> dict:
    """curl 调 gitcode API 拿单个 PR 元信息（带 429 限流退避重试）。"""
    url = f"{API_BASE}/{pr_id}"
    last_err: Exception | None = None
    for attempt in range(3):
        result = subprocess.run(
            ["curl", "-sS", "--max-time", "60", "-H", "User-Agent: pr-test/1.0", url],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(f"curl 失败 (rc={result.returncode}): {result.stderr[:500]}")
        data = json.loads(result.stdout)
        if isinstance(data, dict) and data.get("error_code_name"):
            err = f"API 返回错误: {data.get('error_code_name')} - {data.get('error_message', '')}"
            if "429" in err and attempt < 2:
                wait = 15 * (attempt + 1)
                log(f"  API 限流 429，{wait}s 后重试 ({attempt + 1}/2)...")
                time.sleep(wait)
                last_err = RuntimeError(err)
                continue
            raise RuntimeError(err)
        base_sha = (data.get("base") or {}).get("sha", "")
        head_sha = (data.get("head") or {}).get("sha", "")
        if not base_sha or not head_sha:
            raise RuntimeError(f"API 返回缺少 base/head sha: keys={sorted(data.keys())}")
        return {
            "base_sha": base_sha,
            "head_sha": head_sha,
            "head_ref": (data.get("head") or {}).get("ref", ""),
            "head_repo": ((data.get("head") or {}).get("repo") or {}).get("full_name", ""),
            "title": data.get("title", ""),
            "state": data.get("state", ""),
            "source": "api",
        }
    raise last_err or RuntimeError("API 请求失败")


def fetch_pr_meta(pr_id: str, runs_dir: Path) -> dict:
    """拿 PR 元信息：本地缓存 → gitcode API（429 退避重试）→ ls-remote 降级。

    - 缓存：runs_dir/pr_meta/<id>.json，PR 元信息一次查询后不变（对测试而言），
      复用可避免重复消耗 API 额度、也避免 base.sha（master 动态 tip）漂移。
    - 降级：API 不可用（如匿名共享额度被 429 占满）时，用 git ls-remote
      直接拿 refs/merge-requests/<id>/head（head_sha）与 refs/heads/master
      （base_sha=master 当前 tip）。此时 base 非精确 PR 基线，diff 可能混入
      master 上其他合入，报告 meta.source=ls_remote 可识别。
    """
    cache = runs_dir / "pr_meta" / f"{pr_id}.json"
    if cache.is_file():
        meta = json.loads(cache.read_text(encoding="utf-8"))
        # ls_remote 降级结果不缓存 base：每次用 merge-base 重新解析分叉点
        # （master 浅历史更新后 merge-base 更准，且避免旧缓存固化 master tip）
        if meta.get("source") == "ls_remote":
            meta["base_sha"] = None
        log(f"  PR 元信息命中缓存: head={meta['head_sha'][:10]} (source={meta.get('source', 'api')})")
        return meta
    try:
        meta = _fetch_pr_meta_via_api(pr_id)
    except Exception as e:
        log(f"  API 不可用（{str(e)[:120]}），降级 ls-remote...")
        # base 置 None：等 head 对象 fetch 到本地后用 merge-base 解析分叉点
        # （比 master tip 精确得多，老 PR 用 master tip 做 diff 会混入数千无关文件）
        meta = {
            "base_sha": None,
            "head_sha": _ls_remote_sha(f"refs/merge-requests/{pr_id}/head"),
            "head_ref": "master",
            "head_repo": "",
            "title": "",
            "state": "unknown",
            "source": "ls_remote",
        }
        log(f"  降级成功: head={meta['head_sha'][:10]}（base 待 merge-base 解析）")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
# git 工具函数
# ---------------------------------------------------------------------------


def git(repo_dir: Path, *args: str, check: bool = True) -> str:
    cmd = ["git", "-C", str(repo_dir)]
    if args and args[0] in ("diff", "log", "show"):
        cmd += ["-c", "core.quotepath=false"]
    cmd += list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and result.returncode != 0:
        raise RuntimeError(f"git {args[0]} 失败 (rc={result.returncode}): {result.stderr[:800]}")
    return result.stdout


def sha_exists(main_repo: Path, sha: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(main_repo), "cat-file", "-t", sha],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def init_main_repo(runs_dir: Path, since: str) -> Path:
    main_repo = runs_dir / "main_repo"
    if not (main_repo / ".git").exists():
        main_repo.mkdir(parents=True, exist_ok=True)
        git(main_repo, "init")
    remotes = git(main_repo, "remote").split()
    if "origin" not in remotes:
        git(main_repo, "remote", "add", "origin", REPO_URL)
    # fetch master 历史（覆盖 PR base 所在时间窗）；幂等，已存在则增量更新
    log(f"fetch master 历史 (since={since})，首次较慢...")
    git(main_repo, "fetch", f"--shallow-since={since}", "origin", "master")
    return main_repo


def resolve_base_by_merge(main_repo: Path, head_sha: str, fallback_sha: str = None) -> str:
    """解析 PR 的 base（分叉点）。gitcode API 的 base.sha 一律是当前 master
    动态 tip（对 open/merged PR 均如此，实测），做 base 会让 diff 混入 master
    上数千个无关合入文件——因此 base 统一用 merge-base 解析。

    浅历史（--shallow-since）下若真实分叉点早于浅边界，merge-base 找不到
    公共祖先——此时加深 head 分支历史（--depth=30，单分支少量 commit，开销
    小）使两侧历史相交后重试；仍失败依次回退 API 返回的 base.sha、master tip。
    """
    out = git(main_repo, "merge-base", "origin/master", head_sha, check=False).strip()
    if out and sha_exists(main_repo, out):
        log(f"  base 用 merge-base: {out[:10]}")
        return out
    # 第 1 级：加深 head 分支历史后重试（分叉点通常在 head 分支前几十个 commit 内）
    log("  merge-base 未命中，加深 head 分支历史重试...")
    git(main_repo, "fetch", "--depth=30", "origin", head_sha, check=False)
    out = git(main_repo, "merge-base", "origin/master", head_sha, check=False).strip()
    if out and sha_exists(main_repo, out):
        log(f"  base 用 merge-base（加深 head 后）: {out[:10]}")
        return out
    # 第 2 级：加深 master 历史（git 对已 shallow 的仓库不会随增量 fetch 自动
    # 加深，老 PR 分叉点可能早于当前浅边界——deepen master 后重试）
    log("  merge-base 仍未命中，加深 master 历史重试...")
    git(main_repo, "fetch", "--shallow-since=2025-11-01", "origin", "master", check=False)
    out = git(main_repo, "merge-base", "origin/master", head_sha, check=False).strip()
    if out and sha_exists(main_repo, out):
        log(f"  base 用 merge-base（加深 master 后）: {out[:10]}")
        return out
    for label, cand in (("API base.sha", fallback_sha), ("master tip", None)):
        if not cand:
            cand = _ls_remote_sha("refs/heads/master")
        if sha_exists(main_repo, cand):
            log(
                f"  警告: merge-base 仍失败，回退 {label} {cand[:10]}"
                f"（diff 可能混入大量无关改动，建议调大 --since 后重跑）"
            )
            return cand
    raise RuntimeError(f"base 不可解析（merge-base 与回退均失败）")


def affected_from_doc_changes(changed_files: list[str], now_dir: Path) -> dict:
    """PR 只改文档（无代码改动）时的 fallback：改动文档名即受影响 API。

    detect 基于代码变化（include/impl/tests/examples），PR 只同步资料时
    检出 0 个 API。此时用 PR 改动的 docs/**.md 文件 stem 作 API 名——
    文档被本次 PR 同步过，正是需要用生成流水线验证的对象。
    附录/索引页（如 deprecated_interface.md）由调用方 has_include_decl 过滤。
    """
    result: dict = {}
    for f in changed_files:
        if f.startswith("docs/") and f.endswith(".md") and (now_dir / f).is_file():
            result.setdefault(Path(f).stem, f)
    return result


def fetch_pr_refs(main_repo: Path, pr_id: str, head_sha: str) -> None:
    """确保 PR head commit 在本地可达（base 由 merge-base 从 master 历史解析）。

    head fetch 依次尝试：refs/merge-requests/<id>/head（gitcode 的 GitLab 风格
    PR ref，实测可用）→ refs/pull/<id>/head（Gitea 风格兜底）→ 直接 fetch
    head_sha（服务端允许 fetch-by-sha，实测可用，PR head 在 fork 时也有效）。
    fetch 一律 --depth=30：merge-base 需要沿 head 父链回溯找公共祖先，
    depth=1（无父链）时 merge-base 结果不可靠（实测同一 PR 两次结果不同，
    一次假值导致 diff 虚大到 14711 个文件）。单分支 30 层开销极小。
    fetch 失败直接报错（而非静默跳过），避免后续 worktree add 报 invalid reference。
    """
    if not sha_exists(main_repo, head_sha):
        for ref in (f"refs/merge-requests/{pr_id}/head", f"refs/pull/{pr_id}/head"):
            log(f"  fetch PR #{pr_id} head via {ref}")
            result = subprocess.run(
                ["git", "-C", str(main_repo), "fetch", "--depth=30", "origin", ref],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode == 0 and sha_exists(main_repo, head_sha):
                break
            log(f"  {ref} fetch 失败 ({(result.stderr or '').strip()[:120]})")
        if not sha_exists(main_repo, head_sha):
            log("  回退 fetch by sha")
            git(main_repo, "fetch", "--depth=30", "origin", head_sha)
    if not sha_exists(main_repo, head_sha):
        raise RuntimeError(f"head {head_sha} 不可达（PR ref 与 fetch-by-sha 均失败）")


def setup_worktrees(main_repo: Path, before_dir: Path, now_dir: Path, base_sha: str, head_sha: str) -> None:
    # 幂等清理：先 prune 清 prunable 记录，再删残留目录
    git(main_repo, "worktree", "prune", check=False)
    for d in (before_dir, now_dir):
        if d.exists():
            git(main_repo, "worktree", "remove", str(d), "--force", check=False)
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
    git(main_repo, "worktree", "prune", check=False)
    before_dir.parent.mkdir(parents=True, exist_ok=True)
    git(main_repo, "worktree", "add", "--detach", str(before_dir), base_sha)
    git(main_repo, "worktree", "add", "--detach", str(now_dir), head_sha)


# ---------------------------------------------------------------------------
# 变更检测：代码改动 → 受影响 API（含主文档路径）
# ---------------------------------------------------------------------------


def detect_affected_apis(before_dir: Path, now_dir: Path, main_repo: Path, base_sha: str, head_sha: str) -> dict:
    """调 src/api_change_detect 检测代码变化 → 受影响 API 及其主文档路径。

    使用轻量版 detect_api_changes_lightweight：不建全量索引，只对改动文件
    做内容扫描 + 对称差集（~5-10s，而非全量索引的 ~100s）。
    改动文件列表用 `git diff --name-only base_sha head_sha` 直接获取，
    跳过轻量版内部的全量 os.walk + MD5 对比（大仓库上要读两份完整工作区）。

    返回的 {api: doc_rel} 中 doc_rel 一定非空（lightweight 从 now 的 docs/
    收集 API→文档映射，无文档的 API 已被过滤）。

    注意模块冲突：api_change_detect 与 api_locator 都有顶层 config/evidence
    等模块名，用完必须清理 sys.modules，否则同进程连跑多个 PR 时 ImportError。
    """
    import importlib

    out = git(main_repo, "diff", "--name-only", base_sha, head_sha)
    changed_files = [ln.strip().replace("\\", "/") for ln in out.splitlines() if ln.strip()]
    log(f"  git diff 改动文件: {len(changed_files)} 个")
    if not changed_files:
        return {}
    api_cd_dir = PROJECT_ROOT / "src" / "api_change_detect"
    sys.path.insert(0, str(api_cd_dir))
    # 清掉可能被缓存的顶层 config（模块名冲突），确保 pipeline 重新找到
    # api_change_detect/config（含 DEFAULT_PATH_CONFIG/PathConfig）
    sys.modules.pop("config", None)
    try:
        pipeline = importlib.import_module("pipeline")
        pc = pipeline.PathConfig(doc_dir="docs/zh/api")
        return pipeline.detect_api_changes_lightweight(
            str(before_dir),
            str(now_dir),
            path_config=pc,
            changed_files=changed_files,
        )
    finally:
        sys.path.remove(str(api_cd_dir))
        for mod_name in list(sys.modules):
            mod = sys.modules[mod_name]
            mod_file = getattr(mod, "__file__", None) or ""
            if str(api_cd_dir) in mod_file:
                del sys.modules[mod_name]


def has_include_decl(main_repo: Path, sha: str, api_name: str) -> bool:
    """检查 api_name 在指定 commit 的 include/ 是否有函数/模板声明（Name( 或 Name<）。

    调用方做两侧检查（run_all_cli 的 Original/Modified 定位分别在
    before/now worktree 上执行，两侧都必须能定位）：
      - head_sha 无声明 → 聚合头/类别名（determine_api_type 会抛 ValueError）
      - base_sha 无声明 → 新增接口（before 侧定位必然失败，属「无原文档
        生成」场景，当前流水线不支持，需明确 skip 而非放行崩溃）
    """
    pattern = rf"\b{re.escape(api_name)}\s*[<(]"
    result = subprocess.run(
        ["git", "-C", str(main_repo), "grep", "-l", pattern, sha, "--", "include/"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode == 0


def changed_files_between(main_repo: Path, base_sha: str, head_sha: str) -> list[str]:
    """base→head 的改动文件相对路径列表（正斜杠）。"""
    out = git(main_repo, "diff", "--name-only", base_sha, head_sha)
    return [ln.strip().replace("\\", "/") for ln in out.splitlines() if ln.strip()]


def _api_path_variants(api_name: str) -> set[str]:
    """API 名的路径匹配变体（小写）：原名 + PascalCase（Gather → gather）。

    snake 拆分后重拼的 PascalCase 变体覆盖「文档名 Gather / 代码名 gather」
    的命名差。短 stem 不额外生成变体，避免 asc_abs → abs 之类的过宽匹配。
    """
    variants = {api_name.lower()}
    pascal = "".join(p.capitalize() for p in api_name.split("_") if p)
    if pascal:
        variants.add(pascal.lower())
    return variants


def filter_relevant_apis(
    affected: dict,
    changed_files: list[str],
    main_repo: Path,
    base_sha: str,
    head_sha: str,
    now_dir: Path = None,
    api_override: list[str] = None,
) -> dict:
    """从 detect 结果中筛选与 PR 改动相关的 API。

    detect 的 ±100 行内容扫描会把改动代码「顺带调用」的通用函数
    （abs/cosf/sqrtf 等）和目录名撞上的聚合索引页（tensor_api.md 等）一并
    检出（如 PR 6819 检出 137 个，绝大多数与 PR 主题无关）。本函数按两级
    信号筛选真正相关的 API：

      a. 路径词边界命中：API 名变体（长度>=5）按 `_` 词边界出现在任一改动
         文件路径中 —— 该 API 的定义/实现/测试文件被本次 PR 改动。
         `gather` 能命中 `reg_gather_impl.h`、`asc_gather_impl.h`；
         `abs` 不会误命中 `fabsf`（无词边界）；`all`（过短）不参与。
      b. 头文件新增行声明特征：在**被改动的 include/ 头文件**的新增行（+行）
         中出现 `API名(` 或 `API名<` —— 抓「多个新接口声明集中在同一个
         新增头文件」的场景（如 PR 6819 的 pack/unpack/compress/expand 与
         gather 同在 gather_compute.h 中新增）。限定 include/ 是为了避免
         impl 调用处（如 math_functions_impl.h 里调用 cosf）造成的误匹配。

    Args:
        affected: detect_api_changes_lightweight 结果 {api: doc_rel}
        changed_files: PR 改动文件相对路径列表
        main_repo/base_sha/head_sha: 用于取 include/ 头文件的新增行（b 信号）
        api_override: 手动指定 API 名（--apis），直接按名过滤（大小写不敏感）

    Returns:
        筛选后的 {api: doc_rel}；两级信号都为空时**回退全量 detect 结果**
        （保证不全 skip——极少数 PR 只改调用方代码时仍可跑内容相关 API）。
        聚合索引页混入时由调用方的 has_include_decl 过滤兜底。
    """
    if api_override:
        wanted = set()
        picked = {}
        # 语法:「名」(按 stem 匹配注入)或「符号=doc相对路径」(显式指定 API 名
        # 与文档,用于家族共文档等「文档名 ≠ API 名」场景,如
        # asc_dcci_entire_ub=docs/zh/api/SIMD-API/c_api/cache_ctrl/asc_dcci.md)
        for item in api_override:
            item = item.strip().lstrip("#")
            if not item:
                continue
            if "=" in item:
                sym, _, doc_rel = item.partition("=")
                sym = sym.strip()
                doc_rel = doc_rel.strip().replace("\\", "/")
                if sym and now_dir is not None and (now_dir / doc_rel).is_file():
                    picked[sym] = doc_rel
                    log(f"  --apis 显式指定: {sym} <- {doc_rel}")
                elif sym:
                    log(f"  --apis 显式指定的文档不存在: {sym} <- {doc_rel}")
            else:
                wanted.add(item.lower())
        picked.update({api: doc for api, doc in affected.items() if api.lower() in wanted})
        missed = wanted - {api.lower() for api in picked}
        # 注入:detect 未检出的 API(如家族前缀漏检:asc_dcci_entire_ub 的改动
        # 词边界匹配不到家族文档 asc_dcci),从 now 的 docs 树按 stem 搜同名
        # 接口文档构造 doc_rel,手动纳入处理范围
        for api in sorted(missed):
            if now_dir is None or not (now_dir / "docs").is_dir():
                break
            matches = [p for p in (now_dir / "docs").rglob(f"{api}.md") if _is_api_doc(p)]
            if matches:
                doc_rel = str(matches[0].relative_to(now_dir)).replace("\\", "/")
                picked[api] = doc_rel
                others = [str(m.relative_to(now_dir)).replace("\\", "/") for m in matches[1:]]
                log(
                    f"  --apis 注入: {api} <- {doc_rel}（detect 未检出，手动指定）"
                    + (f"；同名接口文档还有 {others}" if others else "")
                )
            else:
                log(f"  --apis 注入失败: {api} 在 now 的 docs/ 下无同名接口文档")
        return picked

    # ---- b 信号素材：被改动的 include/ 头文件的新增行 ----
    decl_text = ""
    include_files = [f for f in changed_files if f.startswith("include/")]
    for f in include_files:
        out = git(main_repo, "diff", "-U0", base_sha, head_sha, "--", f)
        added = [ln[1:] for ln in out.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
        decl_text += "\n".join(added) + "\n"

    flat_paths = [f.lower().replace("/", "_").rsplit(".", 1)[0] for f in changed_files]
    picked: dict[str, dict] = {}
    for api, doc in affected.items():
        hit_files: list[str] = []
        decl_hit = False
        for v in _api_path_variants(api):
            # a. 路径词边界（变体过短不参与，误命中率高）
            if len(v) >= 5:
                pat = re.compile(rf"(?:^|_){re.escape(v)}(?:_|$)")
                for f, flat in zip(changed_files, flat_paths):
                    if pat.search(flat):
                        hit_files.append(f)
            # b. include 头文件新增行中的声明/模板特征
            if not decl_hit and decl_text:
                decl_hit = re.search(rf"\b{re.escape(v)}\s*[<(]", decl_text) is not None
        if hit_files or decl_hit:
            picked[api] = {"doc_rel": doc, "hit_files": sorted(set(hit_files)), "decl_hit": decl_hit}
    if not picked:
        log(f"  相关度筛选: 路径/声明均无命中，回退全量 {len(affected)} 个")
        return dict(affected)
    for api, info in sorted(picked.items()):
        sig = []
        if info["hit_files"]:
            sig.append("path: " + ", ".join(info["hit_files"][:2]) + ("..." if len(info["hit_files"]) > 2 else ""))
        if info["decl_hit"]:
            sig.append("decl in new include lines")
        log(f"  relevant: {api} <- {'; '.join(sig)}")
    return {api: info["doc_rel"] for api, info in picked.items()}


# ---------------------------------------------------------------------------
# 文档生成：subprocess 调 run_all_cli + 产物收集
# ---------------------------------------------------------------------------


def _is_api_doc(md_path: Path) -> bool:
    """判断文档是否为接口主文档（存在 `##` 级章节）。

    asc-devkit 的 docs/ 下存在聚合索引页（如 c_api/reg_gather/gather.md，
    内容仅是「# 聚合操作 + 链接」，gather 接口已改名 asc_gather 后遗留），
    stem 与接口同名但不含任何 `##` 级章节。detect 按 stem 匹配会把它们
    误绑为「受影响 API 的文档」——run_all_cli 对这类文档定位/回填均无意义，
    需在进入生成前排除。

    不按章节名（函数原型/原型定义等）正向判定：各家族模板用词不一
    （实测 tensor_api 用「原型定义」、c_api 用「函数原型」），而
    「索引页无 ## 章节、接口文档必有参数说明等 ## 章节」在两类模板下
    均成立。残留的富索引页由 has_include_decl 兜底（索引页 stem 无
    include 声明）。
    """
    try:
        return re.search(r"^##\s", md_path.read_text(encoding="utf-8", errors="replace"), re.M) is not None
    except OSError:
        return False


# run_all_cli 输出中的「无需变更」标志行（doc_parts 为空、预校验全部通过时
# 流水线以 rc=0 正常结束且不回填——这是正常结果而非失败）
_NOCHANGE_MARKERS = ("无需处理任何 doc_parts", "（无需变更）")


def _catalog_children(api_name: str, now_doc: Path) -> list[Path]:
    """目录页展开：同名文档无 ## 章节时，找同目录下的正经接口子文档。

    asc-devkit 存在「目录页 + 按芯片架构拆分子文档」的组织形式（实测
    asc_copy_l0c2l1：同名 .md 只有两个链接，真正的接口文档是
    asc_copy_l0c2l1_arch_2201.md / _arch_3510.md）。子文档识别规则：
    stem 以「API 名 + _」为前缀（排除其他家族同名前缀的误伤），且含
    ## 级章节（正经接口文档）。
    """
    kids: list[Path] = []
    for p in sorted(now_doc.parent.glob("*.md")):
        if p.name == now_doc.name:
            continue
        if not p.stem.lower().startswith(api_name.lower()):
            continue
        if p.stem[len(api_name): len(api_name) + 1] not in ("_", "-"):
            continue
        if _is_api_doc(p):
            kids.append(p)
    return kids


def _expand_catalog_tree(page: Path, now_dir: Path, seen: set = None, depth: int = 0) -> list[Path]:
    """递归解析导航/分类页的 markdown 链接，返回树中所有「正经接口文档」。

    判定标准与 _is_api_doc 一致（含 ## 级章节）：接口文档 → 收集为终点；
    仅含链接的目录页/分类页 → 递归跟进。相对链接相对当前页面所在目录
    解析（支持 ../），越出 now_dir 的链接忽略；seen 防环、depth 限深防爆炸。
    """
    if seen is None:
        seen = set()
    key = page.resolve()
    if key in seen or depth > 4:
        return []
    seen.add(key)
    if _is_api_doc(page):
        return [page]
    try:
        text = page.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    result: list[Path] = []
    now_root = now_dir.resolve()
    for m in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
        rel = m.group(1).split("#", 1)[0].strip()
        if not rel.endswith(".md"):
            continue
        target = (page.parent / rel.replace("/", os.sep)).resolve()
        if not target.is_file() or not target.is_relative_to(now_root):
            continue
        result.extend(_expand_catalog_tree(target, now_dir, seen, depth + 1))
    return result


def _extract_proto_symbols(doc_path: Path) -> list[str]:
    """从文档「## 函数原型」章节提取声明的符号名(家族共文档的多接口)。"""
    try:
        text = doc_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    m = re.search(r"##\s*函数原型(.*?)(?=\n##\s|\Z)", text, re.S)
    if not m:
        return []
    syms: list[str] = []
    seen: set = set()
    for sm in re.finditer(r"\b([A-Za-z_]\w+)\s*[<(]", m.group(1)):
        s = sm.group(1)
        if s not in seen and not s.startswith("__"):  # 排除 __aicore__ 等修饰
            seen.add(s)
            syms.append(s)
    return syms


def _family_doc_symbols(
    doc_now: Path, doc_before: Path, main_repo: Path, head_sha: str, base_sha: str, changed_files: list[str]
) -> list[str]:
    """家族共文档(asc_dcci 型:stem 非代码符号,一份文档描述 N 个接口)的
    相关符号解析:① 函数原型章节提取符号 → ② include 声明交叉(真实符号)
    → ③ PR 改动文件的 +行 内容交叉(本 PR 真正触碰的符号)。"""
    syms = _extract_proto_symbols(doc_now)
    real = [s for s in syms if has_include_decl(main_repo, head_sha, s)]
    if not real:
        return []
    plus_text = ""
    for f in changed_files:
        out = git(main_repo, "diff", "-U0", base_sha, head_sha, "--", f)
        for ln in out.splitlines():
            if ln.startswith("+") and not ln.startswith("+++"):
                plus_text += ln + "\n"
    relevant = [
        s
        for s in real
        if re.search(rf"\b{re.escape(s)}\b".encode(), plus_text.encode("utf-8", errors="replace"))
    ]
    return relevant


def _collect_latest_new(workspace_dir: Path, api_name: str) -> Path | None:
    """取 workspace 下该 API 最新的 [new] 回填产物（不复制）。"""
    results_dir = workspace_dir / "data" / "outputs" / "snapshot" / "Original" / api_name / "results"
    if not results_dir.is_dir():
        return None
    candidates = sorted(results_dir.glob(f"[[]new]{api_name}_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def run_all_cli(original_md: Path, modified_md: Path, workspace_dir: Path) -> tuple[int, bool]:
    """对 before/now 两个版本的算子主文档跑完整生成/校验流水线。

    LLM 凭证直接继承当前环境（os.environ）——Windows 无 ~/.zshrc，
    不做 shell rc 解析。LLM 连通失败时 run_all_cli 以 rc=0 静默退出且无产物，
    因此产物收集（collect_output）以实际 [new].md 文件为准，不以 rc 为准。

    返回 (rc, no_change)：no_change 表示输出中出现「无需变更」标志
    （文档与代码一致，流水线无章节可更新）——该情况下无产物是正常结果。

    子进程输出用 PIPE 逐行读取后经 print 双写（终端 + run.log）——
    直接继承控制台时输出不落盘，PIPE 化后 rich 的 ANSI 会被日志层剥掉。
    子进程强制 PYTHONIOENCODING/PYTHONUTF8=utf-8：否则其 stdout 跟随
    Windows locale（GBK），被本脚本的 UTF-8 泵解码后中文全部变成替换符
    （实测 run_all_cli 的 rich 横幅/章节名全灭）。
    """
    cmd = [
        sys.executable,
        str(RUN_ALL_CLI),
        "--original_op_dir",
        str(original_md),
        "--modified_op_dir",
        str(modified_md),
        "--workspace_dir",
        str(workspace_dir),
        "--dashboard",
    ]
    if _CURRENT_CHANGED_FILES is not None and _CURRENT_CHANGED_FILES.is_file():
        cmd += ["--changed_files", str(_CURRENT_CHANGED_FILES)]
    log(f"run_all_cli: {original_md.name} | ws={workspace_dir.name}")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    no_change = False
    try:
        for line in proc.stdout:
            text = line.rstrip("\r\n")
            if any(m in text for m in _NOCHANGE_MARKERS):
                no_change = True
            print(text, flush=True)
    finally:
        rc = proc.wait()
    return rc, no_change


def _save_output(src: Path, outputs_dir: Path, name: str) -> Path:
    """把 [new] 产物落到 outputs_dir/{name}_new.md（目录缺失时自动建）。

    6512 实测：家族共文档（asc_dcci → asc_dcci_entire_ub）走的是展开分支，
    该分支从不建 outputs_dir，产物拷贝直接 FileNotFoundError 报 ERROR。
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    dst = outputs_dir / f"{name}_new.md"
    shutil.copy2(src, dst)
    return dst


def collect_output(workspace_dir: Path, api_name: str, outputs_dir: Path) -> Path | None:
    """收集 run_all_cli 的 [new] 产物 → outputs_dir/{api_name}_new.md。

    产物路径：<ws>/data/outputs/snapshot/Original/<api>/results/[new]<api>_<ts>.md
    （_step_fill_back 的 results_dir = ctx.snapshot_op_dir / "results"，
    而 snapshot_op_dir 指向 snapshot/Original/<op>），取最新一份。
    """
    results_dir = workspace_dir / "data" / "outputs" / "snapshot" / "Original" / api_name / "results"
    if not results_dir.is_dir():
        return None
    candidates = sorted(results_dir.glob(f"[[]new]{api_name}_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    outputs_dir.mkdir(parents=True, exist_ok=True)
    final_md = outputs_dir / f"{api_name}_new.md"
    shutil.copy2(candidates[0], final_md)
    return final_md


# ---------------------------------------------------------------------------
# 单 PR 全流程
# ---------------------------------------------------------------------------


def run_one_pr(
    pr: dict,
    runs_dir: Path,
    main_repo: Path,
    keep_worktrees: bool = False,
    dry_run: bool = False,
    unsync_now_doc: bool = False,
    api_override: list[str] = None,
) -> dict:
    """跑单个 PR 全流程，返回报告 dict。

    skip 语义（逐条明确理由，不再整 PR 盲跳）：
      - now 缺文档（PR 删除文档）：skip
      - include/ 无声明（聚合头/类别名，按 head_sha 判断）：skip
      - before 缺文档（PR 新增文档）：不 skip！用 now 文档占位后照跑
    """
    pr_id = pr["pr_id"]
    pr_desc = pr.get("desc", "")
    t0 = time.time()
    stages: dict[str, float] = {}

    @contextmanager
    def _stage(name: str):
        """阶段计时（try/finally 保证 skip/异常路径同样记录）。"""
        t = time.time()
        try:
            yield
        finally:
            stages[name] = stages.get(name, 0.0) + time.time() - t

    def _finalize() -> dict:
        """收尾：记录总耗时，并把分项之外的残差归入 overhead，
        保证 sum(stages) == elapsed（时间统计完全覆盖）。"""
        total = time.time() - t0
        residual = total - sum(stages.values())
        if residual > 0.0005:
            stages["overhead"] = stages.get("overhead", 0.0) + residual
        report["elapsed"] = total
        report["stages"] = {k: round(v, 3) for k, v in sorted(stages.items())}
        return report

    def _run_cli_maybe_unsync(before_doc: Path, now_doc: Path, ws: Path):
        """跑生成流水线。unsync_now_doc(闭卷)模式：临时把 now worktree 的目标
        文档替换为 before 版内容（模拟「资料未同步」世界），now 真实文档全程
        不参与定位/生成，跑完恢复——now 文档只留给事后对比验证。

        返回 (rc, no_change, unsync_applied)。
        """
        backup = None
        applied = False
        if unsync_now_doc and before_doc.is_file() and now_doc.is_file():
            if before_doc.read_bytes() != now_doc.read_bytes():
                backup = now_doc.read_bytes()
                shutil.copy2(before_doc, now_doc)
                applied = True
                log(f"    [unsync] {now_doc.name} 已临时替换为 before 版（闭卷生成）")
        try:
            rc, no_change = run_all_cli(before_doc, now_doc, ws)
        finally:
            if backup is not None:
                now_doc.write_bytes(backup)
        return rc, no_change, applied

    log(f"=== PR #{pr_id} | {pr_desc} ===")
    report: dict = {
        "pr_id": pr_id,
        "desc": pr_desc,
        "title": "",
        "state": "",
        "base_sha": "",
        "head_sha": "",
        "status": "ok",
        "skip_reason": None,
        "elapsed": 0.0,
        "stages": {},
        "changed_files": 0,
        "detected": 0,
        "relevant": 0,
        "generated": 0,
        "nochange": 0,
        "skipped": 0,
        "apis": [],
    }

    # ---- 1. PR 元信息 ----
    try:
        with _stage("fetch_meta"):
            meta = fetch_pr_meta(pr_id, runs_dir)
    except Exception as e:
        report.update(status="skip", skip_reason=f"API 获取 PR 元信息失败: {e}")
        log(f"  SKIP: {report['skip_reason']}")
        return _finalize()
    head_sha = meta["head_sha"]
    report.update(
        title=meta["title"] or pr_desc, state=meta["state"], head_sha=head_sha, meta_source=meta.get("source", "api")
    )
    log(
        f"  state={meta['state']} head={head_sha[:10]}"
        f" (head repo: {meta['head_repo'] or '同仓'}, source={meta.get('source', 'api')})"
    )

    # ---- 2. fetch refs + worktree ----
    # base 一律用 merge-base 解析分叉点（API 的 base.sha 是动态 master tip，不可信）
    pr_dir = runs_dir / f"pr_{pr_id}"
    before_dir = pr_dir / "before"
    now_dir = pr_dir / "now"
    try:
        with _stage("fetch_worktree"):
            fetch_pr_refs(main_repo, pr_id, head_sha)
            base_sha = resolve_base_by_merge(main_repo, head_sha, fallback_sha=meta["base_sha"])
            report["base_sha"] = base_sha
            setup_worktrees(main_repo, before_dir, now_dir, base_sha, head_sha)
    except Exception as e:
        report.update(status="skip", skip_reason=f"fetch/worktree 失败: {e}")
        log(f"  SKIP: {report['skip_reason']}")
        return _finalize()

    # ---- 3. 变更检测（detect + 改动文件清单 + 文档级 fallback）----
    try:
        with _stage("detect_affected"):
            affected = detect_affected_apis(before_dir, now_dir, main_repo, base_sha, head_sha)
            changed_files = changed_files_between(main_repo, base_sha, head_sha)
            report["changed_files"] = len(changed_files)
            # 供子进程做快照补拷（R1/R2/R3）：改动清单落盘一次，多个 API 复用
            global _CURRENT_CHANGED_FILES
            _changed_list_path = pr_dir / "changed_files.txt"
            pr_dir.mkdir(parents=True, exist_ok=True)
            _changed_list_path.write_text("\n".join(changed_files) + "\n", encoding="utf-8")
            _CURRENT_CHANGED_FILES = _changed_list_path
            report["detected"] = len(affected)
            if not affected:
                # 代码级检出 0 个（PR 只改文档/资料同步）：改动文档名即受影响 API
                affected = affected_from_doc_changes(changed_files, now_dir)
                if affected:
                    report["detected_source"] = "doc_changes"
                    log(f"  代码级检出 0 个（无代码改动），文档级 fallback: {len(affected)} 个改动文档")
    except Exception as e:
        report.update(status="skip", skip_reason=f"detect_affected_apis 失败: {e}")
        log(f"  SKIP: {report['skip_reason']}")
        if not keep_worktrees:
            with _stage("cleanup_worktrees"):
                _cleanup_worktrees(main_repo, before_dir, now_dir)
        return _finalize()
    log(f"  检出受影响 API: {len(affected)} 个（改动文件 {len(changed_files)} 个）")

    # ---- 3.5 相关度筛选：路径命中 / include 头文件新增行声明 ----
    with _stage("filter_relevant"):
        affected = filter_relevant_apis(
            affected, changed_files, main_repo, base_sha, head_sha, now_dir=now_dir, api_override=api_override
        )
    report["relevant"] = len(affected)
    log(f"  相关度筛选后: {len(affected)} 个 API（检出 {report['detected']}）")
    for api_name, doc_rel in sorted(affected.items()):
        log(f"    - {api_name}: {doc_rel}")

    # ---- 4. 逐 API 生成文档 ----
    outputs_dir = pr_dir / "data" / "outputs"
    if dry_run:
        with _stage("dry_run_check"):
            for api_name, doc_rel in sorted(affected.items()):
                before_has = (before_dir / doc_rel).is_file()
                now_has = (now_dir / doc_rel).is_file()
                status = "dry_run"
                note = ""
                if not now_has:
                    status, note = "skip", "now 中无文档（PR 删除文档）"
                elif not has_include_decl(main_repo, head_sha, api_name):
                    # 去重:已独立检出的接口不重复处理;但排除当前 API 自身的文档
                    # (注入的 asc_dcci 就在聚合头自己的树下,排除它会让 kids 恒空)
                    handled_rels = {rel for rel in affected.values() if rel != doc_rel}
                    tree_hits = sorted(
                        str(d.relative_to(now_dir)).replace("\\", "/")
                        for d in _expand_catalog_tree(now_dir / doc_rel, now_dir)
                        if str(d.relative_to(now_dir)).replace("\\", "/") in set(changed_files)
                        and str(d.relative_to(now_dir)).replace("\\", "/") not in handled_rels
                    )
                    if tree_hits:
                        status = "dry_run"
                        note = (
                            f"聚合头目录树展开: {len(tree_hits)} 个受影响接口 "
                            f"{[Path(x).stem for x in tree_hits[:6]]} 将分别跑流水线"
                        )
                    else:
                        status, note = "skip", "聚合头/类别名（目录树下无受本 PR 影响的接口文档）"
                elif not has_include_decl(main_repo, base_sha, api_name):
                    status, note = "skip", "before 的 include/ 无声明（新增接口，流水线无法定位原始侧）"
                elif not _is_api_doc(now_dir / doc_rel):
                    kids = _catalog_children(api_name, now_dir / doc_rel)
                    if kids:
                        status = "dry_run"
                        note = f"目录页展开: {len(kids)} 个子文档 {[k.stem for k in kids]} 将分别跑流水线"
                    else:
                        status, note = "skip", "聚合索引页/非接口文档（无『函数原型』章节且无子文档可展开）"
                report["apis"].append(
                    {
                        "api_name": api_name,
                        "doc_rel": doc_rel,
                        "before_doc_exists": before_has,
                        "now_doc_exists": now_has,
                        "status": status,
                        "note": note,
                    }
                )
        if not keep_worktrees:
            with _stage("cleanup_worktrees"):
                _cleanup_worktrees(main_repo, before_dir, now_dir)
        return _finalize()

    with _stage("generate_loop"):
        _done_lower: dict[str, str] = {}  # 小写名 -> 已完成的名字（避免大小写重复跑）
        for api_name, doc_rel in sorted(affected.items()):
            rec: dict = {"api_name": api_name, "doc_rel": doc_rel, "elapsed": 0.0, "rc": None, "status": "", "note": ""}
            _dup = _done_lower.get(api_name.lower())
            if _dup and _dup != api_name:
                # store / Store 这类仅大小写不同的重复项：前一个已跑过就不再跑。
                # 否则 lowercase 那次会因文档 stem 与 API 名不一致而入错分支，
                # 空跑（gen_val=0、backfill_map={}）并覆盖掉正常产物。
                rec.update(status="skip", note=f"与已处理的 {_dup} 重复（仅大小写差异），跳过", elapsed=0.0)
                report["skipped"] += 1
                log(f"  skip {api_name}: {rec['note']}")
                report["apis"].append(rec)
                continue
            _done_lower[api_name.lower()] = api_name
            api_t0 = time.time()
            now_doc = now_dir / doc_rel
            before_doc = before_dir / doc_rel
            if not now_doc.is_file():
                rec.update(status="skip", note="now 中无文档（PR 删除文档）", elapsed=time.time() - api_t0)
                report["skipped"] += 1
                log(f"  skip {api_name}: {rec['note']}")
                report["apis"].append(rec)
                continue
            # include 声明检查必须在文档形态检查之前：聚合头/类别名（如 basic_api、
            # c_api）的目录页即使能展开出子文档，API 名本身不是代码符号，
            # run_all_cli 定位必然失败（实测 ValueError: not found in include/）
            if not has_include_decl(main_repo, head_sha, api_name):
                # 聚合头/类别名（basic_api/c_api 等导航页，include 无声明）：
                # 递归展开其目录树，找「文档被本 PR 改动」且尚未被独立检出的
                # 真实接口文档，逐个走完整流水线（树内接口的文档路径出现在
                # 改动清单 = 资料被本 PR 触碰，是相关性依据）。
                # 去重排除当前 API 自身的文档（注入的 asc_dcci 就在聚合头自己的
                # 树下，不排除会让 kids 恒空、注入的 API 被自己 skip）
                handled_rels = {rel for rel in affected.values() if rel != doc_rel}
                changed_set = set(changed_files)
                kids = []
                seen_rels = set()
                for d in _expand_catalog_tree(now_doc, now_dir):
                    rel = str(d.relative_to(now_dir)).replace("\\", "/")
                    if rel in changed_set and rel not in handled_rels and rel not in seen_rels:
                        seen_rels.add(rel)
                        kids.append(d)
                if not kids:
                    rec.update(
                        status="skip",
                        note="聚合头/类别名（include 无声明），其目录树下无受本 PR 影响的接口文档",
                        elapsed=time.time() - api_t0,
                    )
                    report["skipped"] += 1
                    log(f"  skip {api_name}: {rec['note']}")
                    report["apis"].append(rec)
                    continue
                log(f"  {api_name}: 聚合头目录树展开 -> {len(kids)} 个受影响接口 {[k.stem for k in kids]}")
                for sub_now in kids:
                    sub_stem = sub_now.stem
                    sub_rel = str(sub_now.relative_to(now_dir)).replace("\\", "/")
                    sub_before = before_dir / sub_rel
                    rec_sub: dict = {
                        "api_name": sub_stem,
                        "doc_rel": sub_rel,
                        "elapsed": 0.0,
                        "rc": None,
                        "status": "",
                        "note": f"由聚合头 {api_name} 目录树展开",
                    }
                    sub_t0 = time.time()
                    if not has_include_decl(main_repo, head_sha, sub_stem):
                        # 家族共文档（asc_dcci 型）：stem 非代码符号，但函数原型
                        # 章节描述了真实符号（asc_dcci_entire_ub 等）——提取与 PR
                        # 改动交叉的相关符号，把家族文档内容临时装入「符号名.md」
                        # 逐个跑流水线，产物按符号名落盘后清理临时文件
                        syms = _family_doc_symbols(sub_now, sub_before, main_repo, head_sha, base_sha, changed_files)
                        if not syms:
                            rec_sub.update(
                                status="skip",
                                note="include/ 无函数声明（聚合头/类别名，且非家族共文档）",
                                elapsed=time.time() - sub_t0,
                            )
                            report["apis"].append(rec_sub)
                            continue
                        log(f"    {sub_stem}: 家族共文档展开 -> {len(syms)} 个相关符号 {syms}")
                        for sym in syms:
                            rec_sym: dict = {
                                "api_name": sym,
                                "doc_rel": sub_rel,
                                "elapsed": 0.0,
                                "rc": None,
                                "status": "",
                                "note": f"由家族文档 {sub_stem}.md 展开",
                            }
                            sym_t0 = time.time()
                            tmp_now = sub_now.parent / f"{sym}.md"
                            tmp_before = sub_before.parent / f"{sym}.md"
                            try:
                                shutil.copy2(sub_now, tmp_now)
                                shutil.copy2(sub_before, tmp_before)
                                ws_sym = pr_dir / f"ws_{sym}"
                                rc, no_change, unsynced = _run_cli_maybe_unsync(tmp_before, tmp_now, ws_sym)
                                rec_sym["rc"] = rc
                                latest = _collect_latest_new(ws_sym, sym)
                                rec_sym["elapsed"] = time.time() - sym_t0
                                if latest is not None:
                                    # 产物以「文档」视角命名:更新后的 asc_dcci.md
                                    # 产物按「文档」视角命名：该符号所属的
                                    # 文档是 sub_stem（asc_dcci），不是聚合页 now_doc
                                    final_md = _save_output(latest, outputs_dir, sub_stem)
                                    rec_sym.update(
                                        status="ok", output=str(final_md.relative_to(pr_dir)).replace("\\", "/")
                                    )
                                    report["generated"] += 1
                                    log(f"  done {sym}: rc={rc}, 产物 {rec_sym['output']} ({rec_sym['elapsed']:.1f}s)")
                                elif no_change:
                                    if unsynced:
                                        rec_sym.update(
                                            status="fail", note="闭卷(unsync)模式下仍判无需变更——疑似变更检测漏检"
                                        )
                                    else:
                                        rec_sym.update(status="nochange", note="文档与代码一致，无需变更（非失败）")
                                    report["nochange"] = report.get("nochange", 0) + 1
                                    log(f"  nochange {sym}: 文档无需变更 ({rec_sym['elapsed']:.1f}s)")
                                else:
                                    rec_sym.update(
                                        status="fail",
                                        note="run_all_cli 未产出 [new] 文档"
                                        + (
                                            "（rc=0，无无需变更标志，疑似 LLM 中途异常）" if rc == 0 else f"（rc={rc}）"
                                        ),
                                    )
                                    log(f"  FAIL {sym}: {rec_sym['note']} ({rec_sym['elapsed']:.1f}s)")
                            except Exception as e:
                                rec_sym.update(status="fail", note=str(e)[:300], elapsed=time.time() - sym_t0)
                                log(f"  ERROR {sym}: {e}")
                            finally:
                                tmp_now.unlink(missing_ok=True)
                                tmp_before.unlink(missing_ok=True)
                            report["apis"].append(rec_sym)
                        continue
                    if not has_include_decl(main_repo, base_sha, sub_stem):
                        rec_sub.update(
                            status="skip", note="before 的 include/ 无声明（新增接口）", elapsed=time.time() - sub_t0
                        )
                        report["apis"].append(rec_sub)
                        continue
                    if not sub_before.is_file():
                        sub_before.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(sub_now, sub_before)
                        rec_sub["note"] += "；before 缺文档，已用 now 占位"
                    ws_sub = pr_dir / f"ws_{sub_stem}"
                    try:
                        rc, no_change, unsynced = _run_cli_maybe_unsync(sub_before, sub_now, ws_sub)
                        rec_sub["rc"] = rc
                        latest = _collect_latest_new(ws_sub, sub_stem)
                        rec_sub["elapsed"] = time.time() - sub_t0
                        if latest is not None:
                            final_md = _save_output(latest, outputs_dir, sub_stem)
                            rec_sub.update(status="ok", output=str(final_md.relative_to(pr_dir)).replace("\\", "/"))
                            report["generated"] += 1
                            log(f"  done {sub_stem}: rc={rc}, 产物 {rec_sub['output']} ({rec_sub['elapsed']:.1f}s)")
                        elif no_change:
                            if unsynced:
                                rec_sub.update(
                                    status="fail",
                                    note="闭卷(unsync)模式下仍判无需变更——文档缺新内容却未触发生成，疑似变更检测漏检",
                                )
                            else:
                                rec_sub.update(status="nochange", note="文档与代码一致，无需变更（非失败）")
                            report["nochange"] = report.get("nochange", 0) + 1
                            log(f"  nochange {sub_stem}: 文档无需变更 ({rec_sub['elapsed']:.1f}s)")
                        else:
                            rec_sub.update(
                                status="fail",
                                note="run_all_cli 未产出 [new] 文档"
                                + ("（rc=0，无无需变更标志，疑似 LLM 中途异常）" if rc == 0 else f"（rc={rc}）"),
                            )
                            log(f"  FAIL {sub_stem}: {rec_sub['note']} ({rec_sub['elapsed']:.1f}s)")
                    except Exception as e:
                        rec_sub.update(status="fail", note=str(e)[:300], elapsed=time.time() - sub_t0)
                        log(f"  ERROR {sub_stem}: {e}")
                    report["apis"].append(rec_sub)
                continue
            if not has_include_decl(main_repo, base_sha, api_name):
                # 新增接口：before 的 include/ 无声明，run_all_cli 的 Original 定位
                # （在 before worktree 找文档 stem 同名声明）必然失败——属「无原文档
                # 生成」场景，当前流水线不支持，明确 skip 而非放行崩溃
                rec.update(
                    status="skip",
                    note="before 的 include/ 无声明（新增接口，『更新文档』流水线"
                    "无法定位原始侧，需无原文档生成能力才能覆盖）",
                    elapsed=time.time() - api_t0,
                )
                report["skipped"] += 1
                log(f"  skip {api_name}: {rec['note']}")
                report["apis"].append(rec)
                continue
            if not _is_api_doc(now_doc):
                # 目录页场景：同目录存在「API名_*.md」正经子文档时逐个展开处理；
                # 临时把子文档内容装入与 API 同名的文件（run_all_cli 的定位、
                # 回填都以文件 stem 推导 API 名），跑完恢复目录页原内容。
                children = _catalog_children(api_name, now_doc)
                if not children:
                    rec.update(
                        status="skip",
                        note="同名文档是聚合索引页/非接口文档（无『函数原型』章节），与本次改动仅为 stem 同名，不更新",
                        elapsed=time.time() - api_t0,
                    )
                    report["skipped"] += 1
                    log(f"  skip {api_name}: {rec['note']}")
                    report["apis"].append(rec)
                    continue
                log(f"  {api_name}: 目录页展开 -> {len(children)} 个子文档 {[c.stem for c in children]}")
                for sub_now in children:
                    sub_stem = sub_now.stem
                    sub_before = before_doc.parent / sub_now.name
                    rec_sub: dict = {
                        "api_name": sub_stem,
                        "doc_rel": str(sub_now.relative_to(now_dir)).replace("\\", "/"),
                        "elapsed": 0.0,
                        "rc": None,
                        "status": "",
                        "note": f"由目录页 {now_doc.name} 展开",
                    }
                    sub_t0 = time.time()
                    # 架构作用域：子文档名带 arch_X，但本次改动完全没碰 X → 不改写该文档。
                    # 目录页展开会把子文档内容装进“目录页文件名”再跑流水线，文档的
                    # arch 归属在这一步丢失，只能在这里按子文档名判定。
                    _sub_arch = re.search(r"arch_(\d+)", sub_stem)
                    if _sub_arch:
                        _scope = sorted(
                            {m.group(1) for f in changed_files if (m := re.search(r"(?:npu_arch_|dav_)(\d+)", f))}
                        )
                        if _scope and _sub_arch.group(1) not in _scope:
                            rec_sub.update(
                                status="skip",
                                note=(
                                    f"架构作用域：文档归属 npu_arch_{_sub_arch.group(1)}，"
                                    f"本次变更仅涉及 {'、'.join(_scope)}"
                                ),
                                elapsed=time.time() - sub_t0,
                            )
                            report["skipped"] += 1
                            log(f"  skip {sub_stem}: {rec_sub['note']}")
                            report["apis"].append(rec_sub)
                            continue
                    if not sub_before.is_file():
                        sub_before.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(sub_now, sub_before)
                        rec_sub["note"] += "；before 缺子文档，已用 now 占位"
                    # 临时替换两侧目录页为子文档内容（文件名保持 API 名，stem 推导才正确）
                    page_now_backup = now_doc.read_bytes()
                    page_before_backup = before_doc.read_bytes()
                    try:
                        shutil.copy2(sub_now, now_doc)
                        shutil.copy2(sub_before, before_doc)
                        ws_sub = pr_dir / f"ws_{sub_stem}"
                        rc, no_change, unsynced = _run_cli_maybe_unsync(before_doc, now_doc, ws_sub)
                        rec_sub["rc"] = rc
                        # 这条分支跑的是「目录页文档」（now_doc），快照算子名 = 目录页
                        # stem = api_name，所以产物按 api_name 取；落盘文件名才用
                        # sub_stem（写到对应子文档上）。
                        latest = _collect_latest_new(ws_sub, api_name)
                        rec_sub["elapsed"] = time.time() - sub_t0
                        if latest is not None:
                            final_md = _save_output(latest, outputs_dir, sub_stem)
                            rec_sub.update(status="ok", output=str(final_md.relative_to(pr_dir)).replace("\\", "/"))
                            report["generated"] += 1
                            log(f"  done {sub_stem}: rc={rc}, 产物 {rec_sub['output']} ({rec_sub['elapsed']:.1f}s)")
                        elif no_change:
                            if unsynced:
                                rec_sub.update(
                                    status="fail",
                                    note="闭卷(unsync)模式下仍判无需变更——文档缺新内容却未触发生成，疑似变更检测漏检",
                                )
                            else:
                                rec_sub.update(status="nochange", note="文档与代码一致，无需变更（非失败）")
                            report["nochange"] = report.get("nochange", 0) + 1
                            log(f"  nochange {sub_stem}: 文档无需变更 ({rec_sub['elapsed']:.1f}s)")
                        else:
                            rec_sub.update(
                                status="fail",
                                note="run_all_cli 未产出 [new] 文档"
                                + ("（rc=0，无无需变更标志，疑似 LLM 中途异常）" if rc == 0 else f"（rc={rc}）"),
                            )
                            log(f"  FAIL {sub_stem}: {rec_sub['note']} ({rec_sub['elapsed']:.1f}s)")
                    finally:
                        # 恢复目录页原内容（keep_worktrees 时保留仓库原貌）
                        now_doc.write_bytes(page_now_backup)
                        before_doc.write_bytes(page_before_backup)
                    report["apis"].append(rec_sub)
                continue
            if not before_doc.is_file():
                # 核心修复：PR 新增文档场景不再 skip —— 用 now 文档占位，
                # 变更检测对比的是两侧代码快照，占位文档不影响检出与生成
                before_doc.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(now_doc, before_doc)
                rec["note"] = "before 缺文档，已用 now 文档占位"
                log(f"  {api_name}: {rec['note']}")
            # 「API 名 ≠ 文档 stem」场景(--apis 显式指定,家族共文档等):
            # 把文档内容临时装入「API名.md」让 stem 推导得到正确符号,
            # 产物按 API 名归位;处理完删除临时文件
            if now_doc.stem.lower() != api_name.lower():
                tmp_now = now_doc.parent / f"{api_name}.md"
                tmp_before = before_doc.parent / f"{api_name}.md"
                page_now_bak = now_doc.read_bytes()
                page_before_bak = before_doc.read_bytes()
                log(f"  {api_name}: 文档名({now_doc.stem}.md)≠API名,临时装名处理")
                try:
                    shutil.copy2(now_doc, tmp_now)
                    shutil.copy2(before_doc, tmp_before)
                    rc, no_change, unsynced = _run_cli_maybe_unsync(tmp_before, tmp_now, pr_dir)
                    rec["rc"] = rc
                    output = None
                    latest = _collect_latest_new(pr_dir, api_name)
                    if latest is not None:
                        output = _save_output(latest, outputs_dir, api_name)
                    rec["elapsed"] = time.time() - api_t0
                    if output is not None:
                        rec.update(status="ok", output=str(output.relative_to(pr_dir)).replace("\\", "/"))
                        report["generated"] += 1
                        log(f"  done {api_name}: rc={rc}, 产物 {rec['output']} ({rec['elapsed']:.1f}s)")
                    elif no_change:
                        if unsynced:
                            rec.update(status="fail", note="闭卷(unsync)模式下仍判无需变更——疑似变更检测漏检")
                        else:
                            rec.update(status="nochange", note="文档与代码一致，无需变更（非失败）")
                        report["nochange"] = report.get("nochange", 0) + 1
                        log(f"  nochange {api_name}: 文档无需变更 ({rec['elapsed']:.1f}s)")
                    else:
                        rec.update(
                            status="fail",
                            note="run_all_cli 未产出 [new] 文档"
                            + ("（rc=0，无无需变更标志，疑似 LLM 中途异常）" if rc == 0 else f"（rc={rc}）"),
                        )
                        log(f"  FAIL {api_name}: {rec['note']} ({rec['elapsed']:.1f}s)")
                    report["apis"].append(rec)
                finally:
                    tmp_now.unlink(missing_ok=True)
                    tmp_before.unlink(missing_ok=True)
                    now_doc.write_bytes(page_now_bak)
                    before_doc.write_bytes(page_before_bak)
                continue

            # 正常路径：文档名 == API 名，直接用 PR 工作区跑流水线
            # （此处曾因重构丢失，导致此类 API 静默不产出、计数与日志全为空）
            try:
                rc, no_change, unsynced = _run_cli_maybe_unsync(before_doc, now_doc, pr_dir)
            except Exception as e:
                rec.update(status="fail", note=str(e)[:300], elapsed=time.time() - api_t0)
                log(f"  ERROR {api_name}: {e}")
                report["apis"].append(rec)
                continue
            rec["rc"] = rc
            rec["elapsed"] = time.time() - api_t0
            latest = _collect_latest_new(pr_dir, api_name)
            if latest is not None:
                output = _save_output(latest, outputs_dir, api_name)
                rec.update(status="ok", output=str(output.relative_to(pr_dir)).replace("\\", "/"))
                report["generated"] += 1
                log(f"  done {api_name}: rc={rc}, 产物 {rec['output']} ({rec['elapsed']:.1f}s)")
            elif no_change:
                if unsynced:
                    rec.update(status="fail", note="闭卷(unsync)模式下仍判无需变更——疑似变更检测漏检")
                else:
                    rec.update(status="nochange", note="文档与代码一致，无需变更（非失败）")
                report["nochange"] = report.get("nochange", 0) + 1
                log(f"  nochange {api_name}: 文档无需变更 ({rec['elapsed']:.1f}s)")
            else:
                rec.update(
                    status="fail",
                    note="run_all_cli 未产出 [new] 文档"
                    + ("（rc=0，无无需变更标志，疑似 LLM 中途异常）" if rc == 0 else f"（rc={rc}）"),
                )
                log(f"  FAIL {api_name}: {rec['note']} ({rec['elapsed']:.1f}s)")
            report["apis"].append(rec)

    if not keep_worktrees:
        with _stage("cleanup_worktrees"):
            _cleanup_worktrees(main_repo, before_dir, now_dir)

    report = _finalize()
    stage_str = " | ".join(f"{k}={v:.1f}s" for k, v in report["stages"].items())
    log(
        f"  PR #{pr_id} done: generated={report['generated']} "
        f"nochange={report['nochange']} skipped={report['skipped']} "
        f"total={report['elapsed']:.1f}s [{stage_str}]"
    )
    return report


def _cleanup_worktrees(main_repo: Path, before_dir: Path, now_dir: Path) -> None:
    git(main_repo, "worktree", "remove", str(before_dir), "--force", check=False)
    git(main_repo, "worktree", "remove", str(now_dir), "--force", check=False)


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------


def write_report(reports: list[dict], runs_dir: Path, total_elapsed: float, name: str = "report") -> tuple[Path, Path]:
    """写出汇总报告；name 为报告基名，部分跑传 report_<ids> 以免覆盖全量报告。"""
    runs_dir.mkdir(parents=True, exist_ok=True)
    json_path = runs_dir / f"{name}.json"
    md_path = runs_dir / f"{name}.md"

    summary = {
        "total_prs": len(reports),
        "generated_prs": sum(1 for r in reports if r.get("generated", 0) > 0),
        "skipped_prs": sum(1 for r in reports if r.get("status") == "skip"),
        "total_detected": sum(r.get("detected", 0) for r in reports),
        "total_relevant": sum(r.get("relevant", 0) for r in reports),
        "total_generated": sum(r.get("generated", 0) for r in reports),
        "total_nochange": sum(r.get("nochange", 0) for r in reports),
        "total_skipped_apis": sum(r.get("skipped", 0) for r in reports),
        "total_elapsed": total_elapsed,
        "prs": reports,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# run_by_pr_test 文档生成测试报告")
    lines.append("")
    lines.append(
        f"- PR 数: {summary['total_prs']}（有产出 {summary['generated_prs']}，整 PR 跳过 {summary['skipped_prs']}）"
    )
    lines.append(
        f"- 检出受影响 API 总数: {summary['total_detected']}"
        f"（相关度筛选后 {summary['total_relevant']}，"
        f"生成文档 {summary['total_generated']}，"
        f"无需变更 {summary['total_nochange']}，skip {summary['total_skipped_apis']}）"
    )
    lines.append(f"- 总耗时: {total_elapsed:.1f}s")
    lines.append("")
    lines.append("| PR | 描述 | state | 检出 | 相关 | 生成 | 无需变更 | skip | 耗时 (s) |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in reports:
        lines.append(
            f"| #{r['pr_id']} | {r.get('desc', '')} | {r.get('state', '')} | "
            f"{r.get('detected', 0)} | {r.get('relevant', 0)} | "
            f"{r.get('generated', 0)} | {r.get('nochange', 0)} | "
            f"{r.get('skipped', 0)} | {r.get('elapsed', 0):.1f} |"
        )
    lines.append("")

    # ---- 时间统计：每 PR 阶段明细（分项之和 == 该 PR 总耗时，overhead 为残差）----
    lines.append("## 每 PR 阶段耗时明细（s）")
    lines.append("")
    lines.append("分项之和等于该 PR 总耗时（overhead 为日志/报告组装等未计入分项的残差）。")
    lines.append("")
    stage_names: list[str] = []
    for r in reports:
        for k in r.get("stages", {}):
            if k not in stage_names:
                stage_names.append(k)
    header = ["PR"] + stage_names + ["合计", "总耗时"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |")
    for r in reports:
        st = r.get("stages", {})
        cells = [f"#{r['pr_id']}"]
        for k in stage_names:
            cells.append(f"{st[k]:.1f}" if k in st else "-")
        stage_sum = sum(st.values())
        cells.append(f"{stage_sum:.1f}")
        cells.append(f"{r.get('elapsed', 0):.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ---- 时间统计：跨 PR 阶段累计 ----
    if stage_names:
        lines.append("## 阶段耗时累计（所有 PR 汇总，s）")
        lines.append("")
        lines.append("| 阶段 | 累计 (s) | 占比 |")
        lines.append("| --- | ---: | ---: |")
        pr_total = sum(r.get("elapsed", 0) for r in reports)
        for k in stage_names:
            acc = sum(r.get("stages", {}).get(k, 0) for r in reports)
            pct = f"{acc / pr_total * 100:.1f}%" if pr_total > 0 else "-"
            lines.append(f"| {k} | {acc:.1f} | {pct} |")
        lines.append(f"| **PR 总耗时合计** | **{pr_total:.1f}** | 100% |")
        lines.append("")
    # ---- 时间统计：算子级耗时（每个算子从进入处理到完成的时长）----
    lines.append("## 算子级耗时")
    lines.append("")
    lines.append(
        "每算子耗时含该算子的准入检查、流水线调用与产物收集；各 PR 算子耗时之和 ≈ 该 PR 的 generate_loop 阶段耗时。"
    )
    lines.append("")
    lines.append("| PR | 算子 | 状态 | 耗时 (s) | 产物 |")
    lines.append("| --- | --- | --- | ---: | --- |")
    for r in reports:
        for a in r.get("apis", []):
            out = f"`{a['output']}`" if a.get("output") else "-"
            lines.append(
                f"| #{r['pr_id']} | {a.get('api_name')} | {a.get('status', '')} | {a.get('elapsed', 0):.1f} | {out} |"
            )
    lines.append("")

    lines.append("## API 明细")
    lines.append("")
    for r in reports:
        if r.get("status") == "skip":
            lines.append(f"- PR #{r['pr_id']}: **SKIP** — {r.get('skip_reason', '')}")
            continue
        lines.append(f"### PR #{r['pr_id']} — {r.get('desc', '')}")
        lines.append("")
        if not r.get("apis"):
            lines.append("- 检出 0 个受影响 API（改动文件均未关联到已有文档的 API）")
        for a in r.get("apis", []):
            out = f"，产物 `{a['output']}`" if a.get("output") else ""
            note = f"，{a['note']}" if a.get("note") else ""
            elapsed = f"（{a.get('elapsed', 0):.1f}s）"
            lines.append(
                f"- `{a['api_name']}` → `{a.get('doc_rel', '')}` — **{a.get('status', '')}**{elapsed}{note}{out}"
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="按 gitcode PR 测试文档生成：检测受影响 API 并自动生成文档 markdown")
    parser.add_argument("--pr_ids", default="", help="只跑指定 PR id，逗号分隔（默认跑内置的全部 5 个）")
    parser.add_argument("--runs_dir", default=str(DEFAULT_RUNS_DIR), help="运行/输出根目录")
    parser.add_argument(
        "--since", default=DEFAULT_SINCE, help="fetch master 历史的起始日期（YYYY-MM-DD），需覆盖 PR base 所在时间"
    )
    parser.add_argument("--keep_worktrees", action="store_true", help="保留 worktree（默认跑完删除以省磁盘）")
    parser.add_argument("--dry_run", action="store_true", help="只做 fetch+worktree+变更检测，不跑文档生成")
    parser.add_argument(
        "--api",
        default="",
        help="手动指定要处理的 API 符号名(如 asc_dcci_entire_ub)。"
        "配合 --doc 使用:对该符号跑流水线,以指定的文档为载体回填",
    )
    parser.add_argument(
        "--doc",
        default="",
        help="与 --api 配对的文档相对路径(相对仓库根,如 "
        "docs/zh/api/SIMD-API/c_api/cache_ctrl/asc_dcci.md)。"
        "文档名可与 API 名不同(家族共文档场景),生成结果以该文档为载体",
    )
    parser.add_argument(
        "--sync_now_doc",
        action="store_true",
        help="开卷模式（默认闭卷）：允许流水线参考 now 的文档参与定位与变更评估。"
        "默认闭卷——now 文档临时替换为 before 版，生成结果只依据 before 文档"
        "+ now 代码证据，now 文档仅用于事后对比验证准确性。"
        "仅在需要校验「作者同步的资料是否与代码一致」时使用",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()  # 必须绝对:git -C main_repo 会把相对路径
    # 的 worktree add 目标解析到 main_repo 内部,导致 detect 读到空目录
    prs = list(PR_TEST_LIST)
    if args.pr_ids:
        wanted = [x.strip().lstrip("#") for x in args.pr_ids.split(",") if x.strip()]
        by_id = {p["pr_id"]: p for p in prs}
        merged: list[dict] = []
        for wid in wanted:
            if wid in by_id:
                merged.append(by_id[wid])
            else:
                # 支持任意 PR id（不在内置列表）：描述留空，报告标题取 API 返回的 title
                merged.append({"pr_id": wid, "desc": ""})
        prs = merged
    if not prs:
        log("没有匹配的 PR（--pr_ids 过滤后为空）")
        sys.exit(2)
    api_override = None
    if args.api:
        # --api 一对一指定;配 --doc 时为「API 名 ≠ 文档名」的显式对(家族共文档),
        # 不配 --doc 时按 stem 在 now docs 中注入
        api_override = [f"{args.api}={args.doc}" if args.doc else args.api]

    # 报告命名：--pr_ids 部分跑时按 id 命名（report_5557 / report_5559_6620），
    # 不覆盖全量报告 report.md；全量跑维持 report（覆盖式更新）
    report_name = "report"
    if args.pr_ids:
        report_name = "report_" + "_".join(p["pr_id"] for p in prs)

    log_path = runs_dir / "run.log"
    runs_dir.mkdir(parents=True, exist_ok=True)
    # 单实例互斥必须在写日志/操作 worktree 之前：并发实例会互删对方 worktree
    acquire_instance_lock(runs_dir)
    _setup_python_tee(log_path)
    log(f"日志文件: {log_path}")

    main_repo = init_main_repo(runs_dir, args.since)
    log(f"main_repo: {main_repo}")
    log(f"待跑 PR 数: {len(prs)} -> {','.join(p['pr_id'] for p in prs)}")

    reports: list[dict] = []
    overall_t0 = time.time()
    for idx, pr in enumerate(prs, 1):
        try:
            reports.append(
                run_one_pr(
                    pr,
                    runs_dir,
                    main_repo,
                    keep_worktrees=args.keep_worktrees,
                    dry_run=args.dry_run,
                    unsync_now_doc=not args.sync_now_doc,
                    api_override=api_override,
                )
            )
        except Exception as e:
            # 兜底：单 PR 未预期异常不中断整体，报告里记 skip
            log(f"PR #{pr['pr_id']} 未预期异常（已兜底为 skip）: {e}")
            reports.append(
                {
                    "pr_id": pr["pr_id"],
                    "desc": pr.get("desc", ""),
                    "status": "skip",
                    "skip_reason": f"未预期异常: {e}",
                    "elapsed": 0.0,
                    "stages": {},
                    "detected": 0,
                    "relevant": 0,
                    "generated": 0,
                    "skipped": 0,
                    "apis": [],
                }
            )
        # 循环内增量写报告：中途中断（kill/崩溃）时保留已跑 PR 的数据
        try:
            write_report(reports, runs_dir, time.time() - overall_t0, name=report_name)
        except Exception as e:
            log(f"  增量报告写入失败（忽略，继续）: {e}")
        log(f"进度: {idx}/{len(prs)} PR 已处理")

    total_elapsed = time.time() - overall_t0
    json_path, md_path = write_report(reports, runs_dir, total_elapsed, name=report_name)
    log(f"报告已写出: {json_path} / {md_path} (总耗时 {total_elapsed:.1f}s)")


if __name__ == "__main__":
    main()
