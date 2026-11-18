# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from llm import config as llm_config
from llm.client import LLMClient
from src.logger import debug, error, info, warning

console = Console(force_terminal=True)

# LLM client 延迟初始化：API key 缺失时优雅降级而非崩溃
_client = None
_llm_available = False


def _init_llm_client():
    """延迟初始化 LLM client，检查 API key 是否可用。"""
    global _client, _llm_available
    if _client is not None:
        return _llm_available
    try:
        if os.getenv(llm_config.API_KEY_ENV) and os.getenv(llm_config.BASE_URL_ENV):
            _client = LLMClient(model=os.getenv(llm_config.LLM_MODELS_ENV, "qwen3.7-max"))
            _llm_available = True
        else:
            warning("LLM API key 缺失 (ALIYUN_KEY/ALIYUN_URL)，file 阶段 Part2/Part3 降级为纯算法匹配")
            _llm_available = False
    except Exception as e:
        warning(f"LLM client 初始化失败: {e}，file 阶段 Part2/Part3 降级为纯算法匹配")
        _llm_available = False
    return _llm_available


# =========================
# 缓存（进程内复用，省重复 os.walk/读文件/LLM 调用）
# =========================
_summary_cache: dict[str, str] = {}  # content_hash -> 功能总结（省 LLM）
_tree_cache: dict[str, str] = {}  # repo_path -> 目录树文本（省 os.walk）
_code_map_cache: dict[tuple[str, str], dict] = {}  # (base, file_name) -> code_map（省文件读）
_token_rank_cache: dict[str, list] = {}  # origin_token_hash -> token_ranked 排序结果（省同文件多 anchor 重复算）
_repo_cache: dict[str, list] = {}  # change_repo_path -> repo 文件列表（省跨 process 重复 os.walk/读）

# Part3 batch-verify LLM 结果持久化缓存目录（跨 process 复用，命中则免 LLM 调用）
CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "output" / "cache"


# ---- Part3 batch-verify LLM 结果持久化缓存 ----
# key = (tree_id, tree_structure, origin_token_hash)。命中即"同文件+同仓库结构+同仓库语义"，
# candidates 必同（同 origin_token + 同 repo），response 可安全复用。origin_token 是文件全文，
# 内容变必变 hash，跨 process 也安全（避免 file_功能 摘要碰撞误命中）。
_part3_cache_mem: dict[str, dict] = {}  # h -> parsed（内存层，免文件读）


def _part3_cache_key(tree_id: str, tree_structure: str, origin_token_hash: str) -> tuple[str, Path]:
    """生成缓存 key 的 sha1 短哈希和落盘文件路径。"""
    key_str = json.dumps([tree_id, tree_structure, origin_token_hash], ensure_ascii=False, sort_keys=True)
    h = hashlib.sha1(key_str.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return h, CACHE_DIR / f"part3_{h}.json"


def _part3_cache_get(tree_id: str, tree_structure: str, origin_token_hash: str):
    """返回 (parsed_or_None, h, path, source)。source ∈ {mem, file, miss}。"""
    h, path = _part3_cache_key(tree_id, tree_structure, origin_token_hash)
    if h in _part3_cache_mem:
        return _part3_cache_mem[h], h, path, "mem"
    if path.exists():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            _part3_cache_mem[h] = parsed
            return parsed, h, path, "file"
        except Exception:
            pass
    return None, h, path, "miss"


def _part3_cache_set(h: str, path: Path, parsed: dict) -> None:
    """写内存层 + 落盘（失败只 warning 不崩）。"""
    _part3_cache_mem[h] = parsed
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        warning(f"Part3 缓存写入失败 {path}: {e}")


# n-gram Jaccard 快速预筛选（O(n)，替代 SequenceMatcher O(n²) 做预筛排序）
_PREFILTER_MODE = "fast"  # "fast"=n-gram Jaccard; "seq"=SequenceMatcher 回退
_NGRAM_N = 4


def _ngram_set(s: str, n: int = _NGRAM_N) -> set:
    """字符 n-gram 集合。len(s)<n 时退化为字符集（避免短串越界）。"""
    if not s:
        return set()
    if len(s) < n:
        return set(s)
    return {s[i: i + n] for i in range(len(s) - n + 1)}


def _jaccard(sa: set, sb: set) -> float:
    """集合 Jaccard 相似度。"""
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    return inter / len(sa | sb)


def _get_file_summary(content: str) -> str:
    """带缓存地获取文件功能总结。同一 content 只调一次 LLM。

    rebuild_anchor 在 Part1内容变/Part2命中/Part3命中后都会对命中文件调
    build_summary_prompt；多个 anchor 命中同一新文件时，同一份 content 会被
    重复总结 N 次。LLMClient 无内置缓存，故这里按 content 的 sha1 做进程内缓存。
    LLM 不可用时用 "unknown" 兜底，保持与原 rebuild_anchor 降级语义一致。
    """
    key = hashlib.sha1(content.encode("utf-8", errors="ignore"), usedforsecurity=False).hexdigest()
    cached = _summary_cache.get(key)
    if cached is not None:
        dprint(f"[SummaryCache] hit (content len={len(content)})")
        return cached
    _init_llm_client()
    if _llm_available:
        prompt = build_summary_prompt(content)
        summary = _client.chat(prompt, temperature=0.1, step_label="重建-总结文件功能").strip()
    else:
        # LLM 不可用时用占位符（与原 rebuild_anchor 降级语义一致）
        summary = "unknown"
    _summary_cache[key] = summary
    return summary


DEBUG = False


def dprint(*args):
    if DEBUG:
        print(*args)


# =========================
#  LLM封装
# =========================
def llm_infer(messages):
    """LLM 推理封装，API key 缺失时返回空字符串。"""
    _init_llm_client()
    if not _llm_available:
        warning("llm_infer: LLM 不可用，返回空结果")
        return ""
    return _client.chat(messages, temperature=0.1).strip()


# llm判断两个文件是不是一个 打分


def build_similarity_prompt(origin_func, new_content):
    return [
        {"role": "system", "content": "你是代码相似度判断专家"},
        {
            "role": "user",
            "content": f"""
判断两个文件是否是同一个算子文件（可能改名或移动）

原文件功能：
{origin_func}

候选文件代码：
{new_content[:1000]}

输出JSON：
{{
  "similarity": 0~1,
  "is_same_file": true/false,
  "reason": "简要原因"
}}
""",
        },
    ]


# llm总结文件内容


def build_summary_prompt(content):
    return [
        {"role": "system", "content": "你是代码分析助手"},
        {
            "role": "user",
            "content": f"""
总结代码功能（50字以内）：

{content[:1000]}
""",
        },
    ]


# =======================
# 文件目录树tree
# =======================
def get_tree_structure(repo_path):
    cached = _tree_cache.get(repo_path)
    if cached is not None:
        dprint(f"[TreeCache] hit: {repo_path}")
        return cached
    lines = []
    for root, dirs, files in os.walk(repo_path):
        dirs.sort()
        rel = os.path.relpath(root, repo_path)
        depth = 0 if rel == "." else rel.replace("\\", "/").count("/") + 1
        indent = "    " * depth
        folder = os.path.basename(root)
        lines.append(f"{indent}{folder}/")
        for f in sorted(files):
            if f.endswith(".cpp") or f.endswith(".h"):
                lines.append(f"{indent}    {f}")
    result = "\n".join(lines)
    _tree_cache[repo_path] = result
    return result


# =========================
# 变更仓库加载
# =========================
def load_repo(repo_path):
    # 按 change_repo_path 缓存：anchor_redirect 对每个 anchor_file 各调一次 process，
    # 同一变更仓的 load_repo 只跑 1 次（os.walk + 读所有文件），跨 process 复用。
    cached = _repo_cache.get(repo_path)
    if cached is not None:
        dprint(f"[Repo] cache hit: {repo_path} ({len(cached)} files)")
        return cached
    dprint(f"\n[Repo] Loading repo: {repo_path}")

    files = []
    for root, dirs, names in os.walk(repo_path):
        dirs.sort()
        for n in sorted(names):
            if n.endswith(".cpp") or n.endswith(".h") or n.endswith(".asc"):
                p = os.path.join(root, n)
                with open(p, "r", errors="ignore") as f:
                    content = f.read()

                rel_inner = os.path.relpath(p, repo_path)
                rel = rel_inner.replace("\\", "/")  # 路径归一化，与 api_locator 格式一致

                files.append({"file_path": rel, "file_name": n, "content": content})

    _repo_cache[repo_path] = files
    dprint(f"[Repo] Total files loaded: {len(files)}")

    return files


# =========================
# code_map读取  （包含变更前后的code_map 这部分假设数据已有）
# =========================
def load_code_map(base, file_name):
    cache_key = (base, file_name)
    cached = _code_map_cache.get(cache_key)
    if cached is not None:
        dprint(f"[CodeMap] hit: {file_name}")
        return cached

    p = os.path.join(base, "files", f"{file_name}_map.json")

    if not os.path.exists(p):
        dprint(f"[CodeMap]  Missing: {file_name}")
        result = {"nodes": [], "edges": []}
    else:
        dprint(f"[CodeMap] Loading: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("nodes"):
            dprint(f"[CodeMap]  Empty nodes: {file_name}")
        result = data

    _code_map_cache[cache_key] = result
    return result


# =========================
#  相似度
# =========================
#
# token相似度
def token_sim(a, b):
    return SequenceMatcher(None, a, b).ratio()


# 节点连边相似度


def code_map_sim(m1, m2):
    def extract_nodes(m):
        raw = m.get("nodes", [])
        if raw and isinstance(raw[0], dict):
            return set(n.get("name") or n.get("id") or str(n) for n in raw)
        return set(raw)

    n1, n2 = extract_nodes(m1), extract_nodes(m2)

    def extract_edges(m):
        result = set()
        for x in m.get("edges", []):
            try:
                result.add((x["type"], x["target"]))
            except KeyError:
                pass
        return result

    e1 = extract_edges(m1)
    e2 = extract_edges(m2)

    node_sim = len(n1 & n2) / max(len(n1 | n2), 1)

    if len(e1) == 0 or len(e2) == 0:
        return node_sim

    edge_sim = len(e1 & e2) / max(len(e1 | e2), 1)

    return 0.6 * node_sim + 0.4 * edge_sim


# =========================
#  路径截断工具
# =========================
_truncate_keywords = []  # 路径格式已统一，不再需要截断


def _truncate_path(path: str | list) -> str:
    """路径归一化：统一使用正斜杠，兼容 Windows/Linux/macOS。

    注意：原截断逻辑（op_api/op_host/op_kernel/op_graph）已不适用新路径格式，
    现在路径格式与 api_locator 一致，都是纯相对路径。

    兼容 str 和 list 两种输入格式（多架构文件聚合时 file_path 可能为 list）：
    - str: 直接归一化返回
    - list: 归一化每个元素，返回第一个元素（用于 unchanged_file_paths 集合匹配）
    """
    if isinstance(path, list):
        normalized = [p.replace("\\", "/") for p in path]
        return normalized[0] if normalized else ""
    else:
        return path.replace("\\", "/")


# =========================
# part3 本地预筛选 + LLM 批量验证（1 次 LLM 调用）
# =========================
def build_batch_verify_prompt(tree_id, tree_structure, file_功能, candidates_info):
    """一次 LLM 调用对多个候选文件输出语义相似度判断。

    合并原 build_candidate_guess_prompt（猜候选）+ _batch_verify_candidates_by_llm
    （批量验证），把 Part3 的 2 次 LLM 调用降为 1 次。候选由调用方用 token_sim
    /n-gram 预筛选提供，LLM 同时看到目录树（语义上下文）和候选内容（打分依据），
    输出每个候选的 similarity。
    """
    candidates_text = ""
    for i, c in enumerate(candidates_info, 1):
        candidates_text += (
            f"\n--- 候选{i} ---\n"
            f"文件名: {c['file_name']}\n"
            f"路径: {c['file_path']}\n"
            f"内容前1000字符:\n{c['content'][:1000]}\n"
        )

    return [
        {"role": "system", "content": "你是代码相似度判断专家"},
        {
            "role": "user",
            "content": f"""
判断以下候选文件中哪些与原始算子文件是同一个文件（可能改名或移动）。

变更前算子仓整体说明：
{tree_id}

变更后算子仓目录结构：
{tree_structure}

原始文件功能：
{file_功能}

候选文件列表：
{candidates_text}

对每个候选只输出文件名和相似度分值, 按相似度从高到低排序。只输出纯JSON:
{{
  "results": [
    {{"file_name": "xxx.cpp", "similarity": 0.0}}
  ]
}}
""",
        },
    ]


# 计算候选和旧文件的三种相似度（本地预筛选 top-K + 1 次 LLM 批量验证）
def match_by_tree_llm(
    anchor, repo, change_repo_path, change_base, origin_base, progress=None, anchor_label="", changed_files=None
):
    """Part3：本地预筛选 top-K + 1 次 LLM 批量验证。

    原实现为 1 次 build_candidate_guess_prompt（猜候选）+ 1 次
    _batch_verify_candidates_by_llm（批量验证），共 2 次 LLM 调用。
    现合并为：token_sim/n-gram 本地预筛选 top-K → 并发算 code_map_sim →
    1 次 LLM 批量打分，LLM 调用降为 1 次（LLM 不可用则 0 次，纯本地兜底）。

    保留 changed_files 参数：当传入变更文件列表时，用它替代全量目录树以
    减小 LLM prompt（本项目独有优化）。
    """
    level2 = anchor["anchor_information"]["level2"]
    level3 = anchor["anchor_information"].get("level3", {})

    tree_id = level3.get("tree_id", "")
    file_功能 = level2.get("file_功能", "")

    part3_detail = {
        "result": "miss",
        "llm_candidates_returned": [],
        "verified": [],
        "changed_files_mode": changed_files is not None,
    }

    if not tree_id:
        dprint("[Part3-LLM]  tree_id 为空，跳过")
        part3_detail["result"] = "skipped"
        part3_detail["reason"] = "tree_id 为空"
        return None, part3_detail

    # tree_structure：优先用 changed_files 缩小范围（本项目独有优化）
    if changed_files is not None and len(changed_files) > 0:
        tree_structure = "\n".join(sorted(changed_files))
        dprint(f"[Part3-LLM] 使用变更文件列表（{len(changed_files)} 个），取代全量目录树")
    else:
        tree_structure = get_tree_structure(change_repo_path)
        dprint(f"[Part3-LLM] tree_structure length: {len(tree_structure)}（全量目录树）")

    if not file_功能:
        dprint("[Warn]  missing file_功能, fallback to token")
        file_功能 = level2.get("token_hash", "")[:1000]

    origin_name = anchor["anchor_information"]["level1"]["file_name"]
    origin_token = level2.get("token_hash", "")
    origin_map = load_code_map(origin_base, origin_name)

    # Step 1: 预筛选 top-K（n-gram Jaccard 或 SequenceMatcher，跨 anchor 缓存）
    _pre_filter_top_k = 15
    origin_token_hash = hashlib.sha1(origin_token.encode("utf-8", errors="ignore"), usedforsecurity=False).hexdigest()
    _rank_cache_key = f"{_PREFILTER_MODE}:{origin_token_hash}"
    token_ranked = _token_rank_cache.get(_rank_cache_key)
    if token_ranked is None:
        token_ranked = []
        origin_ngram = _ngram_set(origin_token) if _PREFILTER_MODE == "fast" else None
        if progress is not None:
            prefilter_task = progress.add_task(f"  Part3 预筛选[{_PREFILTER_MODE}] {anchor_label}", total=len(repo))
            for f in repo:
                if _PREFILTER_MODE == "fast":
                    # f["_ngram_set"] 懒加载：首次算，随 repo 缓存跨 anchor 复用
                    if "_ngram_set" not in f:
                        f["_ngram_set"] = _ngram_set(f["content"])
                    t_sim = _jaccard(origin_ngram, f["_ngram_set"])
                else:
                    t_sim = token_sim(origin_token, f["content"])
                token_ranked.append((t_sim, f))
                progress.update(prefilter_task, advance=1)
            progress.remove_task(prefilter_task)
        else:
            for f in repo:
                if _PREFILTER_MODE == "fast":
                    if "_ngram_set" not in f:
                        f["_ngram_set"] = _ngram_set(f["content"])
                    t_sim = _jaccard(origin_ngram, f["_ngram_set"])
                else:
                    t_sim = token_sim(origin_token, f["content"])
                token_ranked.append((t_sim, f))
        token_ranked.sort(key=lambda x: x[0], reverse=True)
        _token_rank_cache[_rank_cache_key] = token_ranked
    top_candidates = token_ranked[:_pre_filter_top_k]
    dprint(f"[Part3-LLM] 预筛选: {len(repo)} → {len(top_candidates)} (top-{_pre_filter_top_k})")

    if not top_candidates:
        dprint("[Part3-LLM]  无候选文件")
        return None, part3_detail

    part3_detail["llm_candidates_returned"] = [f["file_name"] for _, f in top_candidates]

    # Step 2: 并发计算 top-K 的 code_map_sim（文件 I/O，读 JSON）
    scored = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        fut_map = {}
        for t_sim, f in top_candidates:
            fut = pool.submit(_compute_code_map_sim, f, origin_map, change_base)
            fut_map[fut] = (t_sim, f)
        if progress is not None:
            cmap_task = progress.add_task(f"  Part3 code_map 计算 {anchor_label}", total=len(fut_map))
            for future in as_completed(fut_map):
                t_sim, f = fut_map[future]
                c_sim = future.result()
                scored.append(
                    {
                        "file": f,
                        "name_after_basename": os.path.basename(f["file_name"]),
                        "found_in_repo": True,
                        "file_path": f["file_path"],
                        "token_sim": round(t_sim, 4),
                        "code_map_sim": round(c_sim, 4),
                        "llm_sim": 0.0,
                        "final_score": 0.0,
                    }
                )
                progress.update(cmap_task, advance=1)
            progress.remove_task(cmap_task)
        else:
            with Progress(
                TextColumn("    "),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TextColumn("("),
                TimeRemainingColumn(),
                TextColumn(","),
                TimeElapsedColumn(),
                TextColumn(")"),
                console=console,
                disable=not sys.stdout.isatty(),
            ) as local_progress:
                task = local_progress.add_task("Part3 code_map 计算", total=len(fut_map))
                for future in as_completed(fut_map):
                    t_sim, f = fut_map[future]
                    c_sim = future.result()
                    scored.append(
                        {
                            "file": f,
                            "name_after_basename": os.path.basename(f["file_name"]),
                            "found_in_repo": True,
                            "file_path": f["file_path"],
                            "token_sim": round(t_sim, 4),
                            "code_map_sim": round(c_sim, 4),
                            "llm_sim": 0.0,
                            "final_score": 0.0,
                        }
                    )
                    local_progress.update(task, advance=1)

    # Step 3: LLM 批量验证（持久化缓存, key=tree_id+tree_structure+origin_token_hash）
    # 命中即同文件+同仓库结构, candidates 必同, 可安全复用 response; 未命中才请求 LLM 并写入。
    candidates_info = [
        {"file_name": s["file"]["file_name"], "file_path": s["file_path"], "content": s["file"]["content"]}
        for s in scored
    ]
    _init_llm_client()
    llm_results = []
    if _llm_available:
        if progress is not None:
            verify_task = progress.add_task(f"  Part3 LLM 批量验证 {anchor_label}", total=1)
        # 缓存 key 用 origin_token_hash（文件身份精确指纹, 避免不同文件 file_功能 摘要碰撞误命中）
        cached, ch, cpath, csrc = _part3_cache_get(tree_id, tree_structure, origin_token_hash)
        if cached is not None:
            parsed = cached
            debug(f"[Part3-LLM] 缓存命中({csrc}): part3_{ch}.json")
        else:
            prompt = build_batch_verify_prompt(tree_id, tree_structure, file_功能, candidates_info)
            parsed = _client.chat_with_json(prompt, temperature=0.1, step_label="Part3-批量验证候选")
            _part3_cache_set(ch, cpath, parsed)
            debug(f"[Part3-LLM] 缓存未命中, 已请求 LLM 并写入: part3_{ch}.json")
        llm_results = parsed.get("results", [])
        if progress is not None:
            progress.update(verify_task, advance=1)
            progress.remove_task(verify_task)
    else:
        dprint("[Part3-LLM] LLM 不可用，跳过语义相似度评分")

    # 建 file_name → llm_sim 映射
    llm_sim_map = {}
    for r in llm_results:
        llm_sim_map[r.get("file_name", "")] = r.get("similarity", 0.0)

    # Step 4: 加权打分（权重与原逻辑一致：0.4 code_map + 0.3 token + 0.3 llm；阈值 0.8）
    for s in scored:
        s["llm_sim"] = round(llm_sim_map.get(s["file"]["file_name"], 0.0), 4)
        s["final_score"] = round(0.4 * s["code_map_sim"] + 0.3 * s["token_sim"] + 0.3 * s["llm_sim"], 4)
        s["passed"] = s["final_score"] > 0.8

    # 按分数降序排列，取最高分且通过阈值的候选
    scored.sort(key=lambda x: x["final_score"], reverse=True)
    for s in scored:
        dprint(f"[Part3-LLM] 验证候选: {s['file_path']}")
        dprint(
            f"  token={s['token_sim']:.4f} code_map={s['code_map_sim']:.4f} llm={s['llm_sim']:.4f} "
            f"final={s['final_score']:.4f}"
        )
        vr_for_detail = {k: v for k, v in s.items() if k != "file"}
        part3_detail["verified"].append(vr_for_detail)
        if s["passed"]:
            dprint(f"[Part3-LLM]  命中: {s['file_path']}")
            part3_detail["result"] = "hit"
            part3_detail["selected"] = s["file_path"]
            return s["file"], part3_detail

    dprint("[Part3-LLM]  所有候选均未通过")
    return None, part3_detail


# =========================
# part2  匹配主流程
# =========================
def _score_single_candidate(f, origin_func, origin_token, origin_map, change_base):
    """计算单个候选文件的三种相似度（token / code_map / LLM）"""
    t_sim = token_sim(origin_token, f["content"])
    c_map = load_code_map(change_base, f["file_name"])
    c_sim = code_map_sim(origin_map, c_map)
    s_sim = 0.0
    _init_llm_client()
    if _llm_available:
        prompt = build_similarity_prompt(origin_func, f["content"])
        sem = _client.chat_with_json(prompt, temperature=0.1, step_label="Part2-验证单个候选")
        s_sim = sem.get("similarity", 0.0)
    final = 0.4 * c_sim + 0.3 * t_sim + 0.3 * s_sim
    return {
        "file": f,
        "scores": {
            "file_path": f["file_path"],
            "token_sim": round(t_sim, 4),
            "code_map_sim": round(c_sim, 4),
            "llm_sim": round(s_sim, 4),
            "final_score": round(final, 4),
        },
    }


def _compute_code_map_sim(f, origin_map, change_base):
    """计算单个候选文件的 code_map 相似度（不含 LLM）"""
    c_map = load_code_map(change_base, f["file_name"])
    return code_map_sim(origin_map, c_map)


def match_candidates(anchor, candidates, origin_base, change_base, progress=None, anchor_label=""):
    level2 = anchor["anchor_information"]["level2"]

    origin_func = level2.get("file_功能", "")
    origin_token = level2.get("token_hash", "")
    origin_name = anchor["anchor_information"]["level1"]["file_name"]

    if not origin_func:
        dprint("[Warn]  missing file_功能, fallback to token")
        origin_func = origin_token[:1000]

    dprint(f"\n[Match] Anchor file: {origin_name}")
    dprint(f"[Match] Candidate count: {len(candidates)}")

    origin_map = load_code_map(origin_base, origin_name)

    # Step 1: token_sim 预筛选（纯内存，毫秒级）
    _pre_filter_top_k = 8
    if progress is not None:
        token_task = progress.add_task(f"Part2 token_sim 预筛选 {anchor_label}", total=len(candidates))
        token_ranked = []
        for f in candidates:
            t_sim = token_sim(origin_token, f["content"])
            token_ranked.append((t_sim, f))
            progress.update(token_task, advance=1)
        progress.remove_task(token_task)
    else:
        with Progress(
            TextColumn("    "),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TextColumn("("),
            TimeRemainingColumn(),
            TextColumn(","),
            TimeElapsedColumn(),
            TextColumn(")"),
            console=console,
            disable=not sys.stdout.isatty(),
        ) as local_progress:
            task = local_progress.add_task("Part2 token_sim 预筛选", total=len(candidates))
            token_ranked = []
            for f in candidates:
                t_sim = token_sim(origin_token, f["content"])
                token_ranked.append((t_sim, f))
                local_progress.update(task, advance=1)
    token_ranked.sort(key=lambda x: x[0], reverse=True)
    pre_filtered = [(t_sim, f) for t_sim, f in token_ranked[:_pre_filter_top_k]]
    dprint(f"[Match] token_sim 预筛选: {len(candidates)} → {len(pre_filtered)} (top-{_pre_filter_top_k})")

    # Step 2: 并发计算 top-K 的 code_map_sim（文件 I/O，读 JSON）
    scored = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        fut_map = {}
        for t_sim, f in pre_filtered:
            fut = pool.submit(_compute_code_map_sim, f, origin_map, change_base)
            fut_map[fut] = (t_sim, f)
        if progress is not None:
            cmap_task = progress.add_task(f"Part2 code_map 计算 {anchor_label}", total=len(fut_map))
            for future in as_completed(fut_map):
                t_sim, f = fut_map[future]
                c_sim = future.result()
                scored.append(
                    {
                        "file": f,
                        "scores": {
                            "file_path": f["file_path"],
                            "token_sim": round(t_sim, 4),
                            "code_map_sim": round(c_sim, 4),
                            "llm_sim": 0.0,
                            "final_score": 0.0,
                        },
                    }
                )
                progress.update(cmap_task, advance=1)
            progress.remove_task(cmap_task)
        else:
            with Progress(
                TextColumn("    "),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TextColumn("("),
                TimeRemainingColumn(),
                TextColumn(","),
                TimeElapsedColumn(),
                TextColumn(")"),
                console=console,
                disable=not sys.stdout.isatty(),
            ) as local_progress:
                task = local_progress.add_task("Part2 code_map 计算", total=len(fut_map))
                for future in as_completed(fut_map):
                    t_sim, f = fut_map[future]
                    c_sim = future.result()
                    scored.append(
                        {
                            "file": f,
                            "scores": {
                                "file_path": f["file_path"],
                                "token_sim": round(t_sim, 4),
                                "code_map_sim": round(c_sim, 4),
                                "llm_sim": 0.0,
                                "final_score": 0.0,
                            },
                        }
                    )
                    local_progress.update(task, advance=1)

    # Step 3: 纯本地加权打分（不调 LLM）—— code_map 看函数节点/边结构稳定性，
    # token 看文本相似度。权重偏 code_map(0.6)：文件改名后 token 变化大、code_map 更稳。
    # 阈值从含 LLM 时的 0.8 降到 0.6，miss 则 fallthrough 到 Part3 兜底（Part3 有持久化缓存）。
    for s in scored:
        s["scores"]["final_score"] = round(0.6 * s["scores"]["code_map_sim"] + 0.4 * s["scores"]["token_sim"], 4)
    scored.sort(key=lambda x: x["scores"]["final_score"], reverse=True)

    dprint("\n[Match] Top candidates:")
    for item in scored[:5]:
        dprint(f"  {item['file']['file_path']} -> {item['scores']['final_score']:.4f}")

    if scored and scored[0]["scores"]["final_score"] > 0.6:
        dprint(f"[Match]  Selected: {scored[0]['file']['file_path']}")
        return scored[0]["file"], scored

    dprint("[Match]  No match passed threshold, fallthrough to Part3")
    return None, scored


# =========================
#  Anchor重建
# =========================
def rebuild_anchor(anchor, new_file, change_base):
    dprint(f"\n[Rebuild] File: {new_file['file_path']}")

    # deepcopy: load_code_map 现返回缓存共享引用；存入 anchor 前复制一份，
    # 避免多个命中同一文件的 anchor 共享同一 code_map dict 被下游交叉修改。
    from copy import deepcopy

    new_map = deepcopy(load_code_map(change_base, new_file["file_name"]))

    new_func = _get_file_summary(new_file["content"])
    # LLM 不可用时 _get_file_summary 返回 "unknown"，fallback 到文件名保持原降级语义
    if new_func == "unknown":
        new_func = new_file.get("file_name", "unknown")

    dprint(f"[Rebuild] New function: {new_func}")

    anchor["anchor_information"]["level1"] = {"file_path": new_file["file_path"], "file_name": new_file["file_name"]}

    anchor["anchor_information"]["level2"] = {
        "code_map": new_map,
        "token_hash": new_file["content"],
        "file_功能": new_func,
    }

    anchor["is_redirect"] = True

    for ev in anchor["evidence"]:
        ev["evidence_information"]["file_path"] = new_file["file_path"]
        ev["evidence_information"]["file_name"] = new_file["file_name"]
        ev["evidence_information"]["file_content"] = new_file["content"]
        ev["evidence_information"]["file_功能"] = new_func

    return anchor


# =========================
#  构建报告
# =========================
def build_report(anchors, file_anchors, change_log):
    unchanged = [r for r in change_log if r["status"] == "unchanged"]
    changed = [r for r in change_log if r["status"] == "changed"]
    failed = [r for r in change_log if r["status"] == "change_failed"]

    return {
        "summary": {
            "total_anchors": len(anchors),
            "file_anchors": len(file_anchors),
            "unchanged_count": len(unchanged),
            "changed_count": len(changed),
            "failed_count": len(failed),
        },
        "unchanged": [{"anchor_id": r.get("anchor_id", ""), "origin_path": r["origin_path"]} for r in unchanged],
        "changed": [
            {
                "anchor_id": r.get("anchor_id", ""),
                "origin_path": r["origin_path"],
                "new_path": r["new_path"],
                "via": r["via"],
                "match_detail": r["detail"],
            }
            for r in changed
        ],
        "failed": [
            {"anchor_id": r.get("anchor_id", ""), "origin_path": r["origin_path"], "match_detail": r["detail"]}
            for r in failed
        ],
    }


# =========================
#  打印变更结论
# =========================
def dprint_change_report(change_log):
    dprint("\n")
    dprint("=" * 60)
    dprint(" 变更结论报告")
    dprint("=" * 60)

    unchanged = [r for r in change_log if r["status"] == "unchanged"]
    changed = [r for r in change_log if r["status"] == "changed"]
    change_failed = [r for r in change_log if r["status"] == "change_failed"]

    dprint(f"\n 未变更: {len(unchanged)} 个")
    for r in unchanged:
        dprint(f"   {r['origin_path']}")
        dprint(f"     Part1: {r['detail']['part1']}")

    dprint(f"\n 已变更: {len(changed)} 个")
    for r in changed:
        dprint(f"   {r['origin_path']}")
        dprint(f"     Part1: {r['detail']['part1']}")
        dprint(f"     Part2: {r['detail']['part2']}")
        dprint(f"     Part3: {r['detail']['part3']}")
        dprint(f"     → 新路径: {r['new_path']}  (via {r['via']})")

    dprint(f"\n 变更失败: {len(change_failed)} 个")
    for r in change_failed:
        dprint(f"   {r['origin_path']}")
        dprint(f"     Part1: {r['detail']['part1']}")
        dprint(f"     Part2: {r['detail']['part2']}")
        dprint(f"     Part3: {r['detail']['part3']}")

    dprint("\n" + "=" * 60)


def _process_single_anchor(
    idx, i, anchor, repo, origin_map_base, change_map_base, change_repo_path, changed_files=None
):
    """处理单个 anchor 的 Part2 + Part3 匹配逻辑（用于并发执行）

    如果提供了 changed_files，Part2 的候选集和 Part3 的搜索范围会缩小到变更文件。

    Returns:
        dict: {"change_log_entry": ..., "new_anchor": ... or None}
    """
    origin_path = anchor["anchor_information"]["level1"]["file_path"]
    anchor_id = anchor.get("anchor_id") or anchor.get("id") or f"anchor_{i}"

    norm_path = origin_path.replace("\\", "/")
    parts = norm_path.split("/")
    parent_dir = parts[-2] if len(parts) >= 2 else ""
    parent_subpath = "/".join(parts[:-1])

    detail = {"part1": f"路径 '{origin_path}' 在变更仓中不存在", "part2": {}, "part3": {}}

    matched_file = None
    via = ""

    # Part2：父目录路径匹配，用 changed_files 过滤候选以缩小搜索范围
    same_dir = []
    if parent_subpath:
        all_same_dir = [f for f in repo if os.path.dirname(f["file_path"].replace("\\", "/")) == parent_subpath]
        if changed_files is not None:
            same_dir = [f for f in all_same_dir if f["file_path"] in changed_files]
            # 同目录无变更文件 → 跳过 Part2，让 Part3 全局搜索
            # （同目录全是未变更文件，Part2 不可能命中）
            if not same_dir:
                dprint(f"[Part2] 同目录无变更文件，跳过 Part2，交由 Part3 全局搜索")
        else:
            same_dir = all_same_dir

    if same_dir:
        matched_file, part2_scored = match_candidates(anchor, same_dir, origin_map_base, change_map_base)
        if matched_file:
            via = "Part2"
            anchor["change_type"] = "part2_hit"
            detail["part2"] = {
                "result": "hit",
                "parent_subpath": parent_subpath,
                "candidate_count": len(same_dir),
                "candidates_scored": [s["scores"] for s in part2_scored],
                "selected": matched_file["file_path"],
            }
            detail["part3"] = {"result": "skipped", "reason": "Part2 已命中"}
        else:
            detail["part2"] = {
                "result": "miss",
                "parent_subpath": parent_subpath,
                "candidate_count": len(same_dir),
                "candidates_scored": [s["scores"] for s in part2_scored],
            }
    else:
        detail["part2"] = {"result": "skipped", "reason": f"parent_subpath='{parent_subpath}' 在变更仓中无任何文件"}

    # Part3：tree_id + LLM 猜候选，批量验证
    if not matched_file:
        matched_file, part3_detail = match_by_tree_llm(
            anchor, repo, change_repo_path, change_map_base, origin_map_base, changed_files=changed_files
        )
        detail["part3"] = part3_detail
        if matched_file:
            via = "Part3"
            anchor["change_type"] = "part3_hit"

    # 记录结论 + 重建
    new_anchor = None
    if matched_file:
        change_log_entry = {
            "anchor_id": anchor_id,
            "status": "changed",
            "origin_path": origin_path,
            "new_path": matched_file["file_path"],
            "via": via,
            "score": 0.0,
            "detail": detail,
        }
        new_anchor = rebuild_anchor(anchor, matched_file, change_map_base)
    else:
        change_log_entry = {
            "anchor_id": anchor_id,
            "status": "change_failed",
            "origin_path": origin_path,
            "new_path": None,
            "via": None,
            "score": 0.0,
            "detail": detail,
        }

    return {"change_log_entry": change_log_entry, "new_anchor": new_anchor}


# =========================
#  主流程
# =========================
def process(anchor_path, origin_map_base, change_map_base, change_repo_path, output_dir, changed_files=None):

    dprint("\n========== PROCESS START ==========")

    with open(anchor_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    from copy import deepcopy

    additional_tuples = deepcopy(data)
    additional_tuples["anchor_tuple"] = []
    anchors = data["anchor_tuple"]

    dprint(f"\n[Stats] Total anchors: {len(anchors)}")

    file_anchors = [a for a in anchors if a.get("anchor_information", {}).get("type") == "file"]
    info(f"找到 {len(file_anchors)} 个 file 类型的 anchor")
    dprint(f"[Stats] File anchors: {len(file_anchors)}")

    repo = load_repo(change_repo_path)
    by_path = {f["file_path"]: f for f in repo}

    change_log = []

    dprint("========== Part1 预检查 ==========")
    part1_hit = 0
    part1_miss_list = []

    dprint("\n========== Part1 Precheck ==========")

    for i, anchor in enumerate(file_anchors):
        origin_path = str(Path(anchor["anchor_information"]["level1"]["file_path"]))
        anchor_id = anchor.get("anchor_id") or anchor.get("id") or f"anchor_{i}"

        # ===== [DEBUG] Part1 截断路径对比 =====
        origin_truncated = _truncate_path(origin_path)
        matched_new_path = None
        debug(f"\n{'=' * 60}")
        debug(f"  [Part1 截断路径匹配] anchor #{i}")
        debug(f"  origin_path (旧文件路径):   {origin_path}")
        debug(f"  截断后:                     {origin_truncated}")
        debug(f"  by_path 中共 {len(by_path)} 个文件 (截断后对比):")
        for p in sorted(by_path.keys()):
            p_trunc = _truncate_path(p)
            is_match = p_trunc == origin_truncated
            if is_match:
                matched_new_path = p
            debug(f"    → {p}")
            debug(f"        截断: {p_trunc}{' ← MATCH' if is_match else ''}")
        debug(f"{'=' * 60}")
        if matched_new_path:
            part1_hit += 1
            new_file = by_path[matched_new_path]
            # ===== 内容一致性判断 =====
            old_token = anchor["anchor_information"]["level2"].get("token_hash", "")
            new_content = new_file["content"]

            sim = token_sim(old_token, new_content)
            dprint(f"[Part1]hit file = {origin_path}")
            dprint(f"[Part1] content similarity = {sim:.4f}")
            if sim > 0.98:
                dprint("[Part1]hit file 内容没变")
                #  内容没变，仍用 modified 仓数据更新 anchor（file_content/token_hash/code_map/path）
                new_anchor = rebuild_anchor(anchor, new_file, change_map_base)
                new_anchor["change_type"] = "unchanged"
                additional_tuples["anchor_tuple"].append(new_anchor)
                change_log.append(
                    {
                        "anchor_id": anchor_id,
                        "status": "unchanged",
                        "origin_path": origin_path,
                        "new_path": origin_path,
                        "via": "Part1",
                        "score": 1.0,
                        "detail": {
                            "part1": f"路径一致 + 内容一致(sim={sim:.4f})",
                            "part2": {"result": "skipped", "reason": "Part1 已命中"},
                            "part3": {"result": "skipped", "reason": "Part1 已命中"},
                        },
                    }
                )
            else:
                # 内容变了 → 走 rebuild
                dprint(f"[Part1] 内容发生变化，执行重建: {origin_path}")
                anchor["change_type"] = "part1_hit_content_change"

                new_anchor = rebuild_anchor(anchor, new_file, change_map_base)
                additional_tuples["anchor_tuple"].append(new_anchor)
                change_log.append(
                    {
                        "anchor_id": anchor_id,
                        "status": "changed",
                        "origin_path": origin_path,
                        "new_path": origin_path,
                        "via": "Part1_content_changed",
                        "score": sim,
                        "detail": {
                            "part1": f"路径一致但内容变化(sim={sim:.4f})，已重建",
                            "part2": {"result": "skipped", "reason": "Part1 已处理"},
                            "part3": {"result": "skipped", "reason": "Part1 已处理"},
                        },
                    }
                )
        ##########################

        else:
            part1_miss_list.append((i, anchor))

    dprint(f"[Part1]  hit: {part1_hit}")
    dprint(f"[Part1]  miss: {len(part1_miss_list)}")

    # ========== 正式匹配（并发）==========
    dprint("\n========== Matching Start ==========")
    info(f"共 {len(part1_miss_list)} 个 anchor 需要重定向匹配（Part2 + Part3）")

    with Progress(
        TextColumn("    "),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TextColumn("("),
        TimeRemainingColumn(),
        TextColumn(","),
        TimeElapsedColumn(),
        TextColumn(")"),
        console=console,
        disable=not sys.stdout.isatty(),
    ) as outer_progress:
        file_label = Path(anchor_path).stem
        overall_task = outer_progress.add_task(f"文件重定向匹配 ({file_label})", total=len(part1_miss_list))

        # 并发处理所有 anchor：LLM 调用是网络 I/O，线程池效率高
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {}
            for idx, (i, anchor) in enumerate(part1_miss_list):
                anchor_label = f"[{idx + 1}/{len(part1_miss_list)}]"
                fut = pool.submit(
                    _process_single_anchor,
                    idx,
                    i,
                    anchor,
                    repo,
                    origin_map_base,
                    change_map_base,
                    change_repo_path,
                    changed_files,
                )
                futures[fut] = (idx, anchor_label)

            for future in as_completed(futures):
                idx, anchor_label = futures[future]
                result = future.result()
                change_log.append(result["change_log_entry"])
                if result["new_anchor"]:
                    additional_tuples["anchor_tuple"].append(result["new_anchor"])
                outer_progress.update(overall_task, advance=1)

    # ========== 输出 report.json ==========
    report = build_report(anchors, file_anchors, change_log)

    # with open(os.path.join(output_dir, "file_change_report.json"), "w") as f:
    #     json.dump(report, f, indent=2, ensure_ascii=False)
    # dprint("\n[Report]  change_report.json 已输出")

    # data["change_log"] = change_log

    return report, additional_tuples


# =========================
#  运行入口
# =========================
if __name__ == "__main__":
    # 示例入口：路径由环境变量提供，仓库内不保留个人目录
    import os

    base = os.environ.get("ANCHOR_DEMO_BASE", "")
    anchor_path = os.path.join(base, "test_data/abs_aclnnAbs_support_evidence.json")
    origin_map_base = os.path.join(base, "operator/code_map_result/abs_origin")
    change_map_base = os.path.join(base, "operator/code_map_result/abs")
    change_repo_path = os.path.join(base, "operator/abs")

    result = process(anchor_path, origin_map_base, change_map_base, change_repo_path, "result_output")

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    dprint("\n Done. 输出 result.json")
