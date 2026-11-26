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
docs-common-review PR 检视模式辅助脚本（自包含，不依赖其他 skill）

子命令：
  filter   解析 unified diff（git diff -U0 输出）的变更行区间，把 aggregate.js 聚合结果
           过滤为仅 PR 引入的问题（存量问题剔除），并渲染行内评论清单
  submit   把渲染好的评论提交到 GitCode PR。安全流程：提交前与 PR 已有评论去重
           （拉取失败即中止，不假定没有重复）→ head SHA 核对 → 首条 smoke test →
           GET 回读验证行号 → 错位 DELETE 撤回并中止批量 → 逐条发送前刷新判重 →
           行内评论 POST 400 自动降级为普通评论 → 发布记录防重发

Token 获取优先级（对齐 pypto-doc-reviewer 约定）：
  1. --config <json 文件>，格式 {"token": "<令牌>", "baseUrl": "https://api.gitcode.com/api/v5"}
     （baseUrl 可省略，缺省即官方公网地址）；默认路径
     ~/.config/opencode/skills/docs-common-review/config/gitcode-config.json
  2. 环境变量 GITCODE_TOKEN
  Token 全程不回显、不打印、不写入任何产物。

用法：
  python3 pr_review_filter.py filter --diff=<diff文件> --result=<聚合报告.json> \
      --output=<过滤结果.json> [--comments=<评论清单.json>] [--context=3]
  python3 pr_review_filter.py submit --comments=<评论清单.json> --repo=<owner/repo> \
      --pr=<编号> [--expect-sha=<检视head_sha>] [--proximity=5] [--record=<发布记录.json>] \
      [--config=<token配置.json>] [--base-url=<API根地址>]

退出码：0 成功；1 用法错误；2 环境缺失（token / 输入文件 / 去重前置失败 / head SHA 不一致）；
        3 smoke test 错位中止；4 存在提交失败；5 提交过程中版本变化或无法核对，停止剩余提交
单测：python3 -m unittest scripts.test_pr_review_filter（本 skill 根目录内执行）
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
DEFAULT_CONFIG = os.path.join(
    os.path.expanduser("~"),
    ".config/opencode/skills/docs-common-review/config/gitcode-config.json",
)
DEFAULT_BASE_URL = "https://api.gitcode.com/api/v5"
# aggregate.js 聚合规范化后 rule_group 恒为该值；邻近判重的规则标识不采纳它，避免泛化误配
GENERIC_RULE_GROUP = "通用规则"


def unquote_path(s):
    """还原 git 对含特殊字符路径的 C 风格引号转义（+++ b/"docs/\\346\\226\\207.md"）。

    git 按字节转义 UTF-8 多字节字符（如 新 → \\346\\226\\260），须做字节级解码。
    """
    if not (len(s) >= 2 and s.startswith('"') and s.endswith('"')):
        return s
    body = s[1:-1]
    out = bytearray()
    i = 0
    simple = {"n": 10, "t": 9, "r": 13, '"': 34, "\\": 92}
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt in "01234567":
                start, end = i + 1, i + 4
                out.append(int(body[start:end], 8) & 0xFF)
                i += 4
                continue
            out.append(simple.get(nxt, ord(nxt) & 0xFF))
            i += 2
            continue
        out.append(ord(c) & 0xFF)
        i += 1
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return out.decode("latin-1")


def parse_diff(diff_path):
    """解析 unified diff，返回 {head侧文件路径: [[起始行, 结束行], ...]}。

    纯删除 hunk（+侧 count=0）记为 [start, start-1]：context 匹配时仍覆盖插入点邻域，
    严格匹配（context=0）时天然不命中（head 侧无变更行）。
    """
    ranges = {}
    current = None
    with open(diff_path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("diff --git "):
                current = None
            elif line.startswith("+++ "):
                rest = line[4:].strip()
                if rest.startswith("b/"):
                    rest = rest[2:]
                if rest == "/dev/null":
                    current = None
                else:
                    current = unquote_path(rest)
                    ranges.setdefault(current, [])
            elif line.startswith("@@") and current is not None:
                m = HUNK_RE.match(line)
                if m:
                    start = int(m.group(1))
                    count = int(m.group(2)) if m.group(2) is not None else 1
                    ranges[current].append([start, start + count - 1])
    return ranges


def norm_path(p):
    p = str(p).replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def file_matches(issue_file, diff_path):
    """问题文件名与 diff 内仓库相对路径做后缀匹配。"""
    f = norm_path(issue_file)
    d = norm_path(diff_path)
    return f == d or f.endswith("/" + d)


def parse_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def in_ranges(line, ranges, context):
    return any(s - context <= line <= e + context for s, e in ranges)


def in_strict(line, ranges):
    return any(s <= line <= e for s, e in ranges)


def issue_path(issue):
    """问题文件的匹配键：优先 location 的路径部分（聚合层保证 绝对路径[:行号]，而
    file_name 可能是裸文件名无法做路径匹配），退化用 file_name。
    """
    loc = str(issue.get("location") or "").strip()
    if loc:
        head, sep, tail = loc.rpartition(":")
        if sep and parse_int(tail) is not None:
            loc = head
        if loc and not loc.isdigit():
            return loc
    return str(issue.get("file_name") or "")


def issue_kept(issue, ranges, context):
    """单条问题的红线判定：返回 (kept, matched_path)。"""
    matched = next((p for p in ranges if file_matches(issue_path(issue), p)), None)
    if matched is None:
        return False, None
    line = parse_int(issue.get("line_number"))
    if line is None:
        return True, matched
    return in_ranges(line, ranges[matched], context), matched


def one_line(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def render_body(issue, diff_path, line, inline):
    """评论正文：首行 [级别] [问题大类] [规则项名称] [规则名称]，
    四个中括号分别取问题契约的 rule_level / rule_group / rule_item / rule_name
    （问题大类或规则名称为空时省略对应中括号）；正文仅 问题/建议 两段。
    行内评论锚定在对应行上，正文不重复文件与行号；降级普通评论（无锚定）时
    首行保留 文件(:行号)，位置信息自含。
    """
    level = one_line(issue.get("rule_level")) or "提示"
    item = one_line(issue.get("rule_item")) or "检查项"
    group = one_line(issue.get("rule_group"))
    rule = one_line(issue.get("rule_name"))
    desc = one_line(issue.get("description"))
    sug = one_line(issue.get("suggestion"))
    head = f"[{level}]"
    if group:
        head += f" [{group}]"
    head += f" [{item}]"
    if rule:
        head += f" [{rule}]"
    if not inline:
        loc = f"{diff_path}:{line}" if line else diff_path
        head = f"{head} {loc}"
    return f"{head}\n\n**问题**：{desc}\n\n**建议**：{sug}"


def dedup_issues(issues):
    """11 字段全等的问题视为重复（原子 skill 可能对同一行重复上报），保留首条。
    返回 (kept, dedup_removed)。
    """
    seen, kept, dup = set(), [], 0
    for issue in issues:
        key = tuple(sorted((k, str(v)) for k, v in issue.items()))
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        kept.append(issue)
    return kept, dup


def dedup_comments(comments):
    """正文 + 文件 + 行号 全等的评论视为重复，保留首条。"""
    seen, kept = set(), []
    for c in comments:
        key = (c.get("file"), c.get("line"), str(c.get("body")))
        if key in seen:
            continue
        seen.add(key)
        kept.append(c)
    return kept


def cmd_filter(args):
    try:
        ranges = parse_diff(args.diff)
    except OSError as e:
        print(f"读取 diff 失败: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        with open(args.result, encoding="utf-8") as fh:
            result = json.load(fh)
    except (OSError, ValueError) as e:
        print(f"读取聚合报告失败: {e}", file=sys.stderr)
        sys.exit(2)
    issues = result.get("issues") or []
    issues, dedup_removed = dedup_issues(issues)
    kept, dropped, comments = [], 0, []
    for issue in issues:
        ok, matched = issue_kept(issue, ranges, args.context)
        if not ok:
            dropped += 1
            continue
        line = parse_int(issue.get("line_number"))
        kept.append(issue)
        inline = bool(line) and in_strict(line, ranges[matched])
        comments.append(
            {
                "file": matched,
                "line": line,
                "inline": inline,
                "body": render_body(issue, matched, line, inline),
                "rule_item": str(issue.get("rule_item") or ""),
                "rule_name": str(issue.get("rule_name") or ""),
            }
        )
    comments = dedup_comments(comments)
    out = {
        "issues": kept,
        "stats": {
            "total": len(result.get("issues") or []),
            "kept": len(kept),
            "dedup_removed": dedup_removed,
            "dropped_existing": dropped,
            "files_in_diff": len(ranges),
        },
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    if args.comments:
        with open(args.comments, "w", encoding="utf-8") as fh:
            json.dump(comments, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    print(
        f"kept={len(kept)} dropped_existing={dropped} total={len(issues)} files_in_diff={len(ranges)}"
    )


def resolve_token(args):
    """Token 获取优先级：--config / 默认配置文件 → 环境变量；返回 (token, base_url)。"""
    config_path = args.config or DEFAULT_CONFIG
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, ValueError) as e:
            print(f"读取 token 配置失败: {config_path}: {e}", file=sys.stderr)
            sys.exit(2)
        token = str(cfg.get("token") or "").strip()
        base_url = str(cfg.get("baseUrl") or "").strip()
        if token:
            return token, (base_url or DEFAULT_BASE_URL)
        print(f"token 配置存在但缺少 token 字段: {config_path}", file=sys.stderr)
        sys.exit(2)
    env_token = os.environ.get("GITCODE_TOKEN", "").strip()
    if env_token:
        return env_token, (args.base_url or DEFAULT_BASE_URL)
    print(
        "提交评论需要 GitCode 私人令牌，二选一配置后重试：\n"
        "  1. 创建配置文件 " + config_path + '，内容 {"token": "<令牌>"}\n'
        "     （令牌在 GitCode「设置 → 私人令牌」生成，需具备 PR 评论写权限）\n"
        "  2. 自行执行 export GITCODE_TOKEN=<令牌>（令牌不得出现在对话上下文中）",
        file=sys.stderr,
    )
    sys.exit(2)


def api_request(method, url, token, payload=None, timeout=30):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("PRIVATE-TOKEN", token)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def submit_one(base, repo, pr, token, entry):
    """提交单条评论。返回 (status, note_id, detail)；status ∈ ok/fallback/failed/mismatch。"""
    url = f"{base}/repos/{repo}/pulls/{pr}/comments"
    if entry.get("inline"):
        try:
            resp = api_request(
                "POST",
                url,
                token,
                {
                    "body": entry["body"],
                    "path": entry["file"],
                    "position": entry["line"],
                    "side": "RIGHT",
                },
            )
            note_id = resp.get("id")
            if not note_id:
                return "failed", None, "响应缺少 note_id"
            check = api_request(
                "GET", f"{base}/repos/{repo}/pulls/comments/{note_id}", token
            )
            new_line = (check.get("position") or {}).get("new_line")
            if new_line != entry["line"]:
                api_request(
                    "DELETE", f"{base}/repos/{repo}/pulls/comments/{note_id}", token
                )
                return (
                    "mismatch",
                    note_id,
                    f"期望行 {entry['line']}，落点 {new_line}，已撤回",
                )
            return "ok", note_id, ""
        except urllib.error.HTTPError as e:
            if e.code == 400:
                try:
                    resp = api_request("POST", url, token, {"body": entry["body"]})
                    return "fallback", resp.get("id"), "行内锚定失败，已降级普通评论"
                except urllib.error.HTTPError as e2:
                    return "failed", None, f"{e2.code} {e2.reason}"
            return "failed", None, f"{e.code} {e.reason}"
    try:
        resp = api_request("POST", url, token, {"body": entry["body"]})
        return "ok", resp.get("id"), "plain"
    except urllib.error.HTTPError as e:
        return "failed", None, f"{e.code} {e.reason}"


def get_existing_comments(base, repo, pr, token):
    """拉取 PR 全部已有评论（分页）。失败时返回 None——调用方必须中止提交，
    不假定没有重复（去重是硬门禁，不做尽力而为降级）。
    """
    out, page = [], 1
    while True:
        url = f"{base}/repos/{repo}/pulls/{pr}/comments?per_page=100&page={page}"
        try:
            batch = api_request("GET", url, token)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"已有评论拉取失败: {e}", file=sys.stderr)
            return None
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return out


def get_pull(base, repo, pr, token):
    """GET PR 元数据；失败抛异常，由调用方决定中止。"""
    return api_request("GET", f"{base}/repos/{repo}/pulls/{pr}", token)


def entry_rule_tags(entry):
    """条目的规则标识（rule_name / rule_item 优先），清单旧条目缺失时从正文首行
    `[级别] [问题大类] [规则项名称] [规则名称]` 解析（问题大类不参与标识，
    避免邻近判重被泛化值误配）。供邻近判重做同风险近似比对。
    """
    tags = []
    for key in ("rule_name", "rule_item"):
        val = str(entry.get(key) or "").strip()
        if val:
            tags.append(val)
    if tags:
        return tags
    body = str(entry.get("body") or "")
    head = body.splitlines()[0] if body else ""
    segs = re.findall(r"\[([^\[\]]+)\]", head)
    if len(segs) >= 4:
        raw = (segs[2], segs[3])
    elif len(segs) == 3:
        raw = (segs[1], segs[2])
    elif len(segs) == 2:
        raw = tuple(segs[1].split("|"))
    else:
        raw = ()
    tags = [t.strip() for t in raw if t.strip() and t.strip() != GENERIC_RULE_GROUP]
    return tags


def dedup_entry(entry, existing, proximity=5):
    """单条候选与已有评论判重，返回跳过原因；不重复返回 None。判重规则：
    1. file+line 命中已有行内评论；
    2. 正文指纹（去空白）相同；
    3. 同文件、落点距离 ≤proximity 行且已有评论正文含本条规则标识
       （同一风险的确定性近似；proximity=0 关闭）。
    """

    def body_key(b):
        return re.sub(r"\s+", "", str(b or ""))

    ekey = body_key(entry.get("body"))
    tags = entry_rule_tags(entry)
    file, line = entry.get("file"), entry.get("line")
    for c in existing or []:
        pos = c.get("position") or {}
        cf = c.get("path") or pos.get("new_path") or pos.get("path") or ""
        cl = pos.get("new_line")
        cbody = body_key(c.get("body"))
        cid = c.get("id") or "?"
        if file and cf == file and line is not None and cl == line:
            return f"file+line 已有评论（comment {cid}）"
        if ekey and ekey == cbody:
            return f"正文指纹与已有评论相同（comment {cid}）"
        if (
            proximity > 0
            and file
            and cf == file
            and isinstance(line, int)
            and isinstance(cl, int)
            and abs(cl - line) <= proximity
            and any(t and t in cbody for t in tags)
        ):
            return (
                f"落点距离 {abs(cl - line)} 行且规则标识相同"
                f"（comment {cid}），疑似同一问题"
            )
    return None


def dedup_against_existing(comments, existing, proximity=5):
    """与 PR 已有评论去重：file+line 命中、正文指纹相同、或同文件邻近且同规则
    → 跳过（语义与 ascendc-code-review 对齐，确定性近似）。返回 (kept, skipped)。
    """
    kept, skipped = [], []
    for e in comments:
        reason = dedup_entry(e, existing, proximity)
        if reason:
            e = dict(e)
            e["skip_reason"] = reason
            skipped.append(e)
        else:
            kept.append(e)
    return kept, skipped


def load_record(path):
    """读取发布记录（防重发）。缺失或损坏按无记录处理，不阻塞提交。"""
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
        except (OSError, ValueError) as e:
            print(f"发布记录读取失败（按无记录处理）: {e}", file=sys.stderr)
    return []


def dump_record(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def cmd_submit(args):
    token, base_url = resolve_token(args)
    base = base_url.rstrip("/")

    def label_of(entry):
        return (
            f"{entry.get('file')}:{entry.get('line')}"
            if entry.get("line")
            else str(entry.get("file"))
        )

    try:
        with open(args.comments, encoding="utf-8") as fh:
            comments = json.load(fh)
    except (OSError, ValueError) as e:
        print(f"读取评论清单失败: {e}", file=sys.stderr)
        sys.exit(2)

    # 发布记录（防重发）：默认与评论清单同目录 <pr>_inline_comment_results.json
    record_path = args.record or os.path.join(
        os.path.dirname(os.path.abspath(args.comments)),
        f"{args.pr}_inline_comment_results.json",
    )
    records = load_record(record_path)

    def body_key(b):
        return re.sub(r"\s+", "", str(b or ""))

    done_keys = {
        (r.get("file"), r.get("line"), body_key(r.get("body")))
        for r in records
        if r.get("status") in ("ok", "fallback")
    }
    pending, skipped = [], 0
    for entry in comments:
        key = (entry.get("file"), entry.get("line"), body_key(entry.get("body")))
        if key in done_keys:
            skipped += 1
            print(f"SKIP {label_of(entry)} 发布记录中已提交过（防重发）")
        else:
            pending.append(entry)

    # head SHA 基准：--expect-sha 优先（阶段 1 记录值），否则取当前值；
    # 提交过程中每条发送前复核，版本漂移立即停止
    try:
        pull = get_pull(base, args.repo, args.pr, token)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f"PR 信息获取失败，无法核对 head SHA，中止提交: {e}", file=sys.stderr)
        sys.exit(2)
    expect_sha = args.expect_sha or (pull.get("head") or {}).get("sha")
    if not expect_sha:
        print("无法获取 PR head SHA，中止提交（不沿用旧行号）", file=sys.stderr)
        sys.exit(2)
    current_sha = (pull.get("head") or {}).get("sha")
    if args.expect_sha and current_sha != args.expect_sha:
        print(
            f"PR head SHA 已变化（检视 {args.expect_sha}，当前 {current_sha}），中止提交",
            file=sys.stderr,
        )
        sys.exit(2)

    # 提交前去重（硬门禁）：拉取失败即中止，不假定没有重复
    existing = get_existing_comments(base, args.repo, args.pr, token)
    if existing is None:
        print(
            "已有评论拉取失败，中止提交（不假定没有重复，避免重复评论）",
            file=sys.stderr,
        )
        sys.exit(2)
    pending, dup_skipped = dedup_against_existing(pending, existing, args.proximity)
    for e in dup_skipped:
        print(f"SKIP {label_of(e)} {e.get('skip_reason')}")
    skipped += len(dup_skipped)
    if not pending:
        print(
            f"submitted ok=0 fallback=0 failed=0 dup_skipped={skipped}"
            "（全部与已有评论/发布记录重复，未提交新评论）"
        )
        return

    ok = fallback = failed = refreshed_skip = 0
    for idx, entry in enumerate(pending):
        label = label_of(entry)
        # 每条发送前：核对 head SHA 并刷新已有评论重新判重（严格模式）
        try:
            head_now = (
                get_pull(base, args.repo, args.pr, token).get("head") or {}
            ).get("sha")
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"FAIL {label} head SHA 核对失败，停止剩余提交: {e}", file=sys.stderr)
            sys.exit(5)
        if head_now != expect_sha:
            print(
                f"PR head SHA 已变化（基准 {expect_sha}，当前 {head_now}），停止剩余提交",
                file=sys.stderr,
            )
            sys.exit(5)
        fresh = get_existing_comments(base, args.repo, args.pr, token)
        if fresh is None:
            print(
                f"FAIL {label} 已有评论刷新失败，中止提交（不假定没有重复）",
                file=sys.stderr,
            )
            sys.exit(2)
        reason = dedup_entry(entry, fresh, args.proximity)
        if reason:
            refreshed_skip += 1
            skipped += 1
            print(f"SKIP {label} {reason}")
            continue
        try:
            status, note_id, detail = submit_one(base, args.repo, args.pr, token, entry)
        except urllib.error.URLError as e:
            status, note_id, detail = "failed", None, str(e)
        if status == "mismatch" and idx == 0:
            print(
                f"SMOKE-MISMATCH {label} {detail}；已中止批量提交，请修正行号后重试",
                file=sys.stderr,
            )
            sys.exit(3)
        if status in ("ok", "fallback"):
            records.append(
                {
                    "repo": args.repo,
                    "pr": args.pr,
                    "head_sha": expect_sha,
                    "file": entry.get("file"),
                    "line": entry.get("line"),
                    "body": entry.get("body"),
                    "status": status,
                    "note_id": note_id,
                    "detail": detail,
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
            dump_record(record_path, records)
        if status == "ok":
            ok += 1
            print(
                f"OK {label} note_id={note_id}{' plain' if not entry.get('inline') else ''}"
            )
        elif status == "fallback":
            fallback += 1
            print(f"FALLBACK {label} note_id={note_id} {detail}")
        elif status == "mismatch":
            failed += 1
            print(f"DROP {label} {detail}")
        else:
            failed += 1
            print(f"FAIL {label} {detail}")
        time.sleep(0.2)
    print(
        f"submitted ok={ok} fallback={fallback} failed={failed} dup_skipped={skipped}"
        f"（其中逐条复查阶段跳过 {refreshed_skip} 条）"
    )
    if failed:
        sys.exit(4)


def build_parser():
    parser = argparse.ArgumentParser(
        description="docs-common-review PR 检视模式辅助脚本"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_filter = sub.add_parser(
        "filter", help="解析 diff 变更行并过滤聚合结果，渲染行内评论"
    )
    p_filter.add_argument(
        "--diff", required=True, help="unified diff 文件（git diff -U0 输出）"
    )
    p_filter.add_argument(
        "--result", required=True, help="aggregate.js 产出的质检报告 JSON"
    )
    p_filter.add_argument("--output", required=True, help="过滤后问题清单输出路径")
    p_filter.add_argument("--comments", help="行内评论清单输出路径（可选）")
    p_filter.add_argument(
        "--context", type=int, default=3, help="变更行上下文扩展行数（默认 3）"
    )
    p_filter.set_defaults(func=cmd_filter)

    p_submit = sub.add_parser(
        "submit", help="提交评论到 GitCode PR（需 token 配置，见模块说明）"
    )
    p_submit.add_argument(
        "--comments", required=True, help="filter 生成的评论清单 JSON"
    )
    p_submit.add_argument("--repo", required=True, help="仓库（owner/repo）")
    p_submit.add_argument("--pr", required=True, type=int, help="PR 编号")
    p_submit.add_argument(
        "--config",
        help="token 配置 JSON 路径（缺省用 ~/.config/opencode/skills/"
        "docs-common-review/config/gitcode-config.json）",
    )
    p_submit.add_argument(
        "--base-url",
        help="GitCode API 根地址（默认官方公网地址，config 内 baseUrl 优先）",
    )
    p_submit.add_argument(
        "--expect-sha",
        help="检视时的 PR head SHA（提供则先核对，不一致中止；提交过程中逐条复核）",
    )
    p_submit.add_argument(
        "--proximity",
        type=int,
        default=5,
        help="邻近判重窗口：同文件落点距离 ≤N 行且规则标识出现在已有评论正文视为重复"
        "（默认 5，0 关闭）",
    )
    p_submit.add_argument(
        "--record",
        help="发布记录 JSON 路径（默认与评论清单同目录 <pr>_inline_comment_results.json），"
        "成功项实时写入，重跑时防重发",
    )
    p_submit.set_defaults(func=cmd_submit)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
