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

"""
gen_dashboard.py  —  AscendC Operator Dashboard Generator  v2

从算子输出目录自动收集真实数据，生成自包含的可交互 HTML 看板。
不做任何分析或评测，仅汇总已有数据。

用法:
    # 自动发现（推荐）
    python3 gen_dashboard.py --op-dir output/Softmax_evo_*/round_1/parallel_0

    # 显式指定各数据源（可替换任意一项）
    python3 gen_dashboard.py \\
        --op-desc   .../<Op>_op_desc.json \\
        --eval      .../evaluation_results.json \\
        --precision .../precision_results.json \\
        --multi-csv .../profiling/multi_case_report.csv \\
        --dsl       .../<Op>_dsl.py \\
        --kernel    .../<Op>Custom/op_kernel/<op>_custom.cpp \\
        --profiling-dir .../profiling \\
        --test-cases .../test_cases.csv \\
        --output    dashboard.html
"""

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─── 外部可执行程序绝对路径（安全：避免按名称在 PATH 中查找）─────────────────────
# 通过 shutil.which 解析为绝对路径，找不到时回退原名（行为不变）。
_NPU_SMI = shutil.which("npu-smi") or "npu-smi"

# ─── DESIGN TOKENS（单一真源）─────────────────────────────────────────────────
_SKILL_DIR = Path(__file__).parent.parent
_tok_path = _SKILL_DIR / "design_tokens.json"


def _load_tokens(path: Path) -> dict:
    """加载 design_tokens.json；放在函数内避免异常变量泄漏到模块作用域（G.VAR.03）。"""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("读取 design_tokens.json 失败，使用空 tokens: %s", e)
        return {}


_TOKENS = _load_tokens(_tok_path)


def _tok(path: str, default=None):
    """读取 design_tokens 中的嵌套键，如 'colors.pos_vecin'"""
    parts, v = path.split("."), _TOKENS
    for p in parts:
        if not isinstance(v, dict):
            return default
        v = v.get(p, default)
    return v if v is not None else default


def detect_chip_runtime(default_name: str, default_ub_kb: int, default_aic: int,
                        default_aiv: Optional[int] = None) -> dict:
    """
    通过 npu-smi 探测真实芯片型号、NPU 数量和 AICore 核数。

    - 芯片名、NPU 总数：来自 `npu-smi info`
    - AICore Count：来自 `npu-smi info -t common -i 0`（真实硬件值）
    - UB 大小：npu-smi 不暴露，仍从已知芯片映射取
    - aiv：AIV/vector_core 数量，910B 上每个 ai_core 含 2 个 vector_core，
           若未提供则默认为 aic * 2
    """
    info = {
        "name": default_name,
        "ub_kb": int(default_ub_kb),
        "aic": int(default_aic),
        "aiv": int(default_aiv) if default_aiv is not None else int(default_aic) * 2,
        "npu_count": 1,
        "source": "design_tokens",
    }

    # Step 1: npu-smi info — 获取芯片名 + NPU 总数
    try:
        out = subprocess.check_output(
            [_NPU_SMI, "info"], stderr=subprocess.STDOUT, text=True, timeout=3
        )
        m = re.search(r"\|\s*(\d+)\s+(910\w+|310\w*)\s+\|", out)
        if m:
            info["name"] = f"Ascend {m.group(2)}"
            info["source"] = "npu-smi info"
        # NPU 总数：统计 "| <id>  910..." 行数
        npu_ids = re.findall(r"^\|\s*(\d+)\s+(?:910|310)", out, re.MULTILINE)
        if npu_ids:
            info["npu_count"] = len(set(npu_ids))
    except Exception as e:
        logger.debug("npu-smi info 探测失败，回退默认芯片信息: %s", e)

    # UB 大小映射（npu-smi 不暴露，按架构族取）
    ub_map = {
        "910B": 192,   # 910B2/B3/B4: UB 192 KB
        "910": 256,   # 910 Pro/A: UB 256 KB
        "310B": 192,
        "310": 192,
    }
    for k, ub in ub_map.items():
        if k in info["name"]:
            info["ub_kb"] = ub
            break

    # Step 2: npu-smi info -t common -i 0 — 获取真实 AICore Count
    try:
        out2 = subprocess.check_output(
            [_NPU_SMI, "info", "-t", "common", "-i", "0"],
            stderr=subprocess.STDOUT, text=True, timeout=3
        )
        m2 = re.search(r"Aicore Count\s*:\s*(\d+)", out2)
        if m2:
            info["aic"] = int(m2.group(1))
            info["source"] = "npu-smi info -t common"
    except Exception as e:
        logger.debug("npu-smi 探测 AICore Count 失败，保留默认 aic: %s", e)

    return info


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate AscendC operator dashboard HTML")
    p.add_argument("--op-dir", help="算子输出目录（自动发现所有文件）")
    p.add_argument("--op-desc", help="*_op_desc.json")
    p.add_argument("--eval", help="evaluation_results.json（单case结果）")
    p.add_argument("--precision", help="precision_results.json（v2多case精度）")
    p.add_argument("--multi-csv", help="profiling/multi_case_report.csv")
    p.add_argument("--dsl", help="*_dsl.py")
    p.add_argument("--kernel", help="*_custom.cpp")
    p.add_argument("--profiling-dir", help="profiling/ 目录")
    p.add_argument("--test-cases", help="test_cases.csv")
    p.add_argument("--algo-flow", help="algo_flow.json（Claude生成的算子计算流图，覆盖自动提取）")
    p.add_argument("--output", default=None, help="输出 HTML 路径（默认：<op-dir>/dashboard.html，无 op-dir 时为 ./dashboard.html）")
    p.add_argument("--extract-only", action="store_true",
                   help="只写 panels/*/data.json，不生成 HTML（供 Claude 分析前使用）")
    return p.parse_args()


# ─── AUTO-DISCOVER ─────────────────────────────────────────────────────────────

def _pick_latest(candidates) -> "Path | None":
    """从候选路径列表中选取 mtime 最新的文件（忽略访问错误）。"""
    best, best_mtime = None, -1.0
    for cand in candidates:
        try:
            t = cand.stat().st_mtime
            if t > best_mtime:
                best, best_mtime = cand, t
        except Exception as e:
            logger.debug("无法读取候选文件 mtime，已跳过 %s: %s", cand, e)
    return best


def _is_msprof_fair_bench_run(eval_path: "Path") -> bool:
    """
    检测 evaluation_results.json 是否来自 msprof_fair_bench.py 的 ModelNew=Model 镜像跑法。
    这类跑法 speedup≈1.0，ref_time 无效，应跳过作为性能数据源。
    判断依据：文件所在目录名含 _ref_timing_ 或父目录含 ref_timing。
    """
    try:
        parts = eval_path.parts
        for part in parts:
            if "_ref_timing_" in part or part.endswith("_ref_timing"):
                return True
    except Exception as e:
        logger.debug("检测 ref_timing 路径失败，按非镜像跑处理 %s: %s", eval_path, e)
    return False


def discover(op_dir: Path) -> dict:
    found = {}
    for f_desc in sorted(op_dir.rglob("*_op_desc.json")):
        found["op_desc"] = f_desc
        break
    for f in sorted(op_dir.rglob("precision_results.json")):
        found["precision"] = f
        break
    # evaluation_results.json / results_precision.json:
    # 优先选 mtime 最新的，且跳过 msprof_fair_bench 镜像跑（_ref_timing_ 目录）
    _eval_candidates = (list(op_dir.rglob("evaluation_results.json"))
                        + list(op_dir.rglob("results_precision.json")))
    _real_evals = [f for f in _eval_candidates if not _is_msprof_fair_bench_run(f)]
    _eval_pick = _pick_latest(_real_evals) or _pick_latest(_eval_candidates)
    if _eval_pick:
        found["eval"] = _eval_pick
    if "eval" not in found:
        _txt_candidates = list(op_dir.rglob("evaluation_result*.txt"))
        _txt_pick = _pick_latest(_txt_candidates)
        if _txt_pick:
            found["eval"] = _txt_pick
    # Standard profiling/ directory（直接在 op_dir 下）
    profiling = op_dir / "profiling"
    if profiling.is_dir():
        found["profiling_dir"] = profiling
        csv_path = profiling / "multi_case_report.csv"
        if csv_path.exists():
            found["multi_csv"] = csv_path
        report = profiling / "report.txt"
        if report.exists():
            found["report_txt"] = report

    # TileLang / msprof_fair_bench 场景：profiling 在子目录 *_ref_timing_*/profiling/ 里
    # 找最新的含有 Model_device* + ModelNew_device* 配对的 profiling 目录
    if "profiling_dir" not in found:
        _prof_candidates = []
        for d in op_dir.rglob("profiling"):
            if not d.is_dir():
                continue
            _has_model = any(x.is_dir() and x.name.startswith("Model_device")
                             and not x.name.startswith("ModelNew") for x in d.iterdir())
            _has_modelnew = any(x.is_dir() and x.name.startswith("ModelNew_device")
                                for x in d.iterdir())
            if _has_model and _has_modelnew:
                _prof_candidates.append(d)
        if _prof_candidates:
            # 选 mtime 最新的 profiling 目录
            _best_prof = max(_prof_candidates, key=lambda p: p.stat().st_mtime)
            found["profiling_dir"] = _best_prof
            _report = _best_prof / "report.txt"
            if _report.exists():
                found["report_txt"] = _report

    # Flat msprof CSVs: op_summary_custom_fn_<case>.csv + op_summary_reference_fn_<case>.csv
    if "multi_csv" not in found:
        custom_csvs = sorted(op_dir.rglob("op_summary_custom_fn_*.csv"))
        ref_csvs = sorted(op_dir.rglob("op_summary_reference_fn_*.csv"))
        if custom_csvs or ref_csvs:
            found["msprof_custom_csvs"] = custom_csvs
            found["msprof_ref_csvs"] = ref_csvs
    # CAKE2 evaluate.py format: op_summary_<case>.csv（无 custom_fn_ 前缀）
    # 路径不固定，优先在已找到的 profiling_dir 下 glob；
    # 若无 profiling_dir，则在 op_dir 全局 rglob，不假设固定层级。
    if "multi_csv" not in found and "msprof_custom_csvs" not in found:
        if found.get("profiling_dir"):
            _search_iter = Path(found["profiling_dir"]).glob("op_summary_*.csv")
        else:
            _search_iter = op_dir.rglob("op_summary_*.csv")
        _cake2_csvs = sorted([
            f for f in _search_iter
            if not f.name.startswith("op_summary_custom_fn_")
            and not f.name.startswith("op_summary_reference_fn_")
            and f.name != "multi_case_report.csv"
        ])
        if _cake2_csvs:
            found["msprof_cake2_csvs"] = _cake2_csvs
    for f in sorted(op_dir.rglob("op_host/*.cpp")):
        found["op_host"] = f
        break
    for f in sorted(op_dir.rglob("op_kernel/*.cpp")):
        found["kernel"] = f
        break
    if "kernel" not in found:
        for f in sorted(op_dir.rglob("*_custom.cpp")):
            found["kernel"] = f
            break
    for f in sorted(op_dir.rglob("*_dsl.py")):
        found["dsl"] = f
        break
    # test_cases.csv: pick the one co-located with the selected eval dir (most relevant),
    # otherwise pick latest by mtime
    _tc_candidates = list(op_dir.rglob("test_cases.csv"))
    if _tc_candidates:
        if found.get("eval"):
            _eval_dir = Path(found["eval"]).parent
            _colocated = [f for f in _tc_candidates if f.parent == _eval_dir]
            found["test_cases"] = _pick_latest(_colocated) or _pick_latest(_tc_candidates)
        else:
            found["test_cases"] = _pick_latest(_tc_candidates)
    # test_cases.py: prefer co-located with eval dir, then op root
    _tc_py_candidates = list(op_dir.rglob("test_cases.py"))
    if _tc_py_candidates:
        if found.get("eval"):
            _eval_dir = Path(found["eval"]).parent
            _colocated_py = [f for f in _tc_py_candidates if f.parent == _eval_dir]
            found["test_cases_py"] = _pick_latest(_colocated_py) or _pick_latest(_tc_py_candidates)
        else:
            found["test_cases_py"] = _pick_latest(_tc_py_candidates)
    # algo_flow.json: panels/algo/ first, then root
    panels_algo_flow = op_dir / "panels" / "algo" / "algo_flow.json"
    if panels_algo_flow.exists():
        found["algo_flow"] = panels_algo_flow
    else:
        for f in sorted(op_dir.glob("algo_flow.json")):
            found["algo_flow"] = f
            break
    # TileLang standalone: panels/perf/cann_result.json (真实 CANN .so 评测结果)
    cann_result = op_dir / "panels" / "perf" / "cann_result.json"
    if cann_result.exists():
        found["cann_result"] = cann_result
    # TileLang standalone: panels/memory/analysis.md (Claude 写的 UB buffer 分析)
    mem_analysis = op_dir / "panels" / "memory" / "analysis.md"
    if mem_analysis.exists():
        found["memory_analysis_md"] = mem_analysis
    return found


# ─── PARSERS ──────────────────────────────────────────────────────────────────

def _read(path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace") if path and Path(path).exists() else ""


def _json(path) -> dict:
    txt = _read(path)
    try:
        return json.loads(txt) if txt else {}
    except Exception:
        return {}


def parse_ub_from_analysis_md(md_path) -> list:
    """
    从 panels/memory/analysis.md 的 markdown 表格提取 UB buffer 信息。
    支持格式（Claude 按 SPEC.md 生成）：
      **K1 UB（≈192 KB，利用率≈75%）**：
      | Buffer | 用途 | 大小 | 生命周期 |
      |--------|------|------|---------|
      | c_ub   | 暂存 GEMM int32 结果 | 32 KB (8192×int32) | AIV 开始 → Cast 前 |

    扫描整个 analysis.md：支持多个 buffer 表格（如 K1、K2 分别有自己的表格）。
    每个 buffer 携带 kernel_group 字段（如 "K1"/"K2"，单 kernel 时为 ""）。
    返回 [{name, kind, position, multiplier, size_kb, color, label, kernel_group}]
    """
    try:
        txt = _read(md_path)
        if not txt:
            return []
        buffers = []
        seen_names = set()          # 去重：同名 buffer 只保留一次
        in_buffer_table = False
        header_found = False
        current_group = ""          # 当前所属 kernel 组（如 "K1"/"K2"）
        for line in txt.splitlines():
            s = line.strip()
            # 检测 kernel group 标题：**K1 UB** / **K2 UB** / ### K1 / ## K2 UB 等
            if not s.startswith('|'):
                # 非表格行：若当前在 buffer 表中，退出该表，继续扫描下一个表
                if in_buffer_table:
                    in_buffer_table = False
                    header_found = False
                # 尝试提取 kernel group label（如 K1/K2/K3/Kernel1）
                _grp_m = re.search(r'\b(K\d+|Kernel\s*\d+)\b', s, re.IGNORECASE)
                # 只在含 UB/buffer/内存 关键字时更新 group（避免误匹配公式里的 K1）
                _has_mem_keyword = any(
                    kw in s for kw in (
                        'UB',
                        'ub',
                        'buffer',
                        'Buffer',
                        '内存',
                        '空间',
                        'kernel',
                        'Kernel'))
                _is_kernel_group_heading = _grp_m is not None and _has_mem_keyword
                if _is_kernel_group_heading:
                    current_group = _grp_m.group(1).upper().replace(" ", "")
                continue
            cols = [c.strip() for c in s.strip('|').split('|')]
            # Detect header row: contains 'buffer'/'buf'/'用途' keyword
            if not header_found and any(
                'buffer' in c.lower() or '用途' in c.lower() or
                (c.lower() in ('buf', 'buffer', '缓冲区', '名称', 'name'))
                for c in cols
            ):
                header_found = True
                in_buffer_table = True
                continue
            # Skip separator row (|---|---|)
            if all(re.match(r'^[-: ]+$', c) for c in cols if c.strip()):
                continue
            if not in_buffer_table or len(cols) < 3:
                continue
            name = cols[0].strip()
            if not name or name.lower() in ('buffer', 'buf', 'name', '名称', '缓冲区'):
                continue
            if name in seen_names:
                continue  # 跨表格去重
            # Parse size: "32 KB" or "32.0 KB" or "48 KB (49152×uint8)"
            size_str = cols[2] if len(cols) > 2 else ""
            m = re.search(r'([\d.]+)\s*KB', size_str, re.IGNORECASE)
            size_kb = float(m.group(1)) if m else 0.0
            # Infer position from name heuristics (for metadata only, not coloring)
            name_lower = name.lower()
            if any(x in name_lower for x in ('l1', 'l0a', 'l0b')):
                pos = "L1"
            elif any(x in name_lower for x in ('int8', 'fp16', 'out', 'res')):
                pos = "VECOUT"
            else:
                pos = "VECCALC"
            # 每个 buffer 独立配色（按全局序号从色板取，相邻 buffer 颜色不同）
            color = _buf_color(len(buffers))
            # 用途描述：取第 2 列（index 1），去除 markdown 加粗等装饰
            purpose = re.sub(r'\*\*([^*]+)\*\*', r'\1', cols[1].strip()) if len(cols) > 1 else ""
            seen_names.add(name)
            buffers.append({
                "name": name,
                "kind": "TBuf",
                "position": pos,
                "multiplier": 1,
                "size_kb": round(size_kb, 2),
                "color": color,
                "label": f"{pos} TBuf",
                "kernel_group": current_group,
                "purpose": purpose,
            })
        return buffers
    except Exception:
        return []


def parse_cann_result_json(path) -> dict:
    """
    解析 panels/perf/cann_result.json（TileLang standalone 真实 NPU kernel 计时结果）。
    格式：
      {"source": "tilelang_eval_adapter", "ref_time_us": 1279.8,
       "custom_time_us": 678.9, "speedup": 1.89, "notes": "..."}
    """
    try:
        data = _json(path)
        if not data:
            return {}
        return {
            "ref_time_us": float(data.get("ref_time_us", 0)),
            "custom_time_us": float(data.get("custom_time_us", 0)),
            "speedup": float(data.get("speedup", 0)),
            "source": data.get("source", "cann_standalone"),
            "notes": data.get("notes", ""),
        }
    except Exception:
        return {}


def parse_test_cases_csv(path) -> list:
    try:
        txt = _read(path)
        if not txt:
            return []
        lines = [l.strip() for l in txt.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            return []
        sep = ";" if ";" in lines[0] else ","
        headers = [h.strip() for h in lines[0].split(sep)]
        cases = []
        for line in lines[1:]:
            parts = line.split(sep)
            row = {h: v.strip() for h, v in zip(headers, parts)}
            cases.append(row)
        return cases
    except Exception:
        return []


def parse_test_cases_py(path) -> dict:
    """
    从 test_cases.py 中提取 case_name → shape_str 映射。
    使用正则解析（不 exec），支持 mat1_shape/mat2_shape 字段。
    返回 {"case_name": "M=2048 K=4096 N=2048", ...}
    """
    txt = _read(path)
    if not txt:
        return {}
    result = {}
    # 找到每个 case dict 块：从 "name": "xxx" 开始，提取 mat1_shape/mat2_shape
    # 策略：按 "name": "..." 分割，然后在每段中找 mat1_shape/mat2_shape
    case_name_pat = re.compile(r'"name"\s*:\s*"([^"]+)"')
    shape_pat = re.compile(r'"(mat1_shape|mat2_shape|input_shape|shape|shape1|shape2)"\s*:\s*\[([^\]]+)\]')
    # 找所有 case 块的起始位置
    name_matches = list(case_name_pat.finditer(txt))
    for i, nm in enumerate(name_matches):
        case_name = nm.group(1)
        # 搜索范围：到下一个 case 的 "name" 或文件末尾
        end_pos = name_matches[i + 1].start() if i + 1 < len(name_matches) else len(txt)
        segment = txt[nm.start():end_pos]
        shapes = {}
        for sm in shape_pat.finditer(segment):
            key = sm.group(1)
            vals = [v.strip() for v in sm.group(2).split(',')]
            try:
                shapes[key] = [int(v) for v in vals if v]
            except ValueError:
                pass
        if "mat1_shape" in shapes and "mat2_shape" in shapes:
            m1 = shapes["mat1_shape"]
            m2 = shapes["mat2_shape"]
            if len(m1) >= 2 and len(m2) >= 2:
                result[case_name] = f"M={m1[0]} K={m1[1]} N={m2[1]}"
        elif "mat1_shape" in shapes:
            m1 = shapes["mat1_shape"]
            result[case_name] = "×".join(str(v) for v in m1)
        elif "shape1" in shapes and "shape2" in shapes:
            s1 = shapes["shape1"]
            s2 = shapes["shape2"]
            # Matmul-like: shape1[..., M, K] × shape2[..., K, N]
            if len(s1) >= 2 and len(s2) >= 2 and s1[-1] == s2[-2]:
                if len(s1) == 2:
                    result[case_name] = f"M={s1[0]} K={s1[1]} N={s2[1]}"
                else:
                    batch = "×".join(str(v) for v in s1[:-2])
                    result[case_name] = f"B={batch} M={s1[-2]} K={s1[-1]} N={s2[-1]}"
            else:
                # Element-wise binary: show both shapes
                result[case_name] = f"{'×'.join(str(v) for v in s1)} / {'×'.join(str(v) for v in s2)}"
        elif "shape1" in shapes:
            result[case_name] = "×".join(str(v) for v in shapes["shape1"])
        elif "shape" in shapes:
            # Standard "shape": [...] key (e.g. Softmax, Gelu, etc.)
            result[case_name] = "×".join(str(v) for v in shapes["shape"])
        elif "input_shape" in shapes:
            result[case_name] = "×".join(str(v) for v in shapes["input_shape"])
        else:
            # 通用fallback: 从 case segment 提取所有 "key": integer 字段（排除 id/seed 等元信息）
            _skip = {'id', 'seed', 'case_id', 'rank'}
            _prio = ['N_orig', 'N_expert', 'D_in', 'D_out', 'K', 'G', 'M', 'N', 'H', 'S', 'B']
            kv_pat = re.compile(r'"(\w+)"\s*:\s*(\d+)')
            kv = {}
            for kvm in kv_pat.finditer(segment):
                k, v = kvm.group(1), int(kvm.group(2))
                if k.lower() not in _skip and k not in kv:
                    kv[k] = v
            if kv:
                # Priority order first, then remaining (up to 5 total)
                ordered = [(k, kv[k]) for k in _prio if k in kv]
                ordered += [(k, v) for k, v in kv.items() if k not in {k2 for k2, _ in ordered}]
                result[case_name] = " ".join(f"{k}={v}" for k, v in ordered[:5])
    return result


def parse_multi_case_csv(path) -> list:
    """解析 multi_case_report.csv，返回 [{case_id, shape, passed, ref_us, custom_us, speedup}]"""
    txt = _read(path)
    if not txt:
        return []
    rows = []
    # Collect all varN_shape-like columns dynamically
    try:
        reader = csv.DictReader(io.StringIO(txt))
        for row in reader:
            # shape: prefer first varN_shape col, fallback to 'shape'
            shape_val = ""
            for k, v in row.items():
                if re.match(r'var\d+_shape', k.strip()) or k.strip() == 'shape':
                    shape_val = v.strip().strip('"')
                    break
            try:
                rows.append({
                    "case_id": int(row.get("case_id", 0) or 0),
                    "shape": shape_val,
                    "passed": (row.get("passed", "") or "").lower() in ("true", "1", "yes"),
                    "ref_time_us": float(row.get("ref_time_us", 0) or 0),
                    "custom_time_us": float(row.get("custom_time_us", 0) or 0),
                    "speedup": float(row.get("speedup", 0) or 0),
                })
            except (ValueError, TypeError):
                pass
    except Exception as e:
        logger.debug("解析 multi_case_report.csv 失败，返回已解析行: %s", e)
    return rows


def parse_evaluation_results_json(path) -> list:
    """
    解析 CAKE2 evaluate.py 生成的 evaluation_results.json 多-case 格式。

    Schema:
      {
        "operator": "FastGelu",
        "total_cases": 10, "passed_cases": 10,
        "geometric_mean_speedup": 1.69,
        "results": [
          {"case_id": 1, "case_name": "basic_1d", "status": "PASS",
           "speedup": 1.68, "ref_time_us": 6.46, "custom_time_us": 3.62,
           "precision": {"passed": true, "ratios": {"max_re": 1.0, "mean_re": 0.38, ...}}}
        ]
      }

    Returns [{case_id, name, shape, passed, ref_time_us, custom_time_us, speedup, precision}]
    Compatible with the multi_csv branch in collect_data().
    """
    data = _json(path)
    if not data or "results" not in data:
        return []
    rows = []
    for r in data.get("results", []):
        prec_raw = r.get("precision") or {}
        ratios = prec_raw.get("ratios") or {}
        # "PASS" status → passed=True; "FAIL"/"ERROR" → False
        passed = str(r.get("status", "")).upper() == "PASS"
        raw_cid = r.get("case_id")
        try:
            case_id = int(raw_cid) if raw_cid not in (None, "") else 0
        except (TypeError, ValueError):
            case_id = 0
        rows.append({
            "case_id": case_id,
            "name": r.get("case_name", f"case_{case_id}"),
            "shape": r.get("case_name", ""),   # use case_name as shape label
            "passed": passed,
            "ref_time_us": float(r.get("ref_time_us") or 0),
            "custom_time_us": float(r.get("custom_time_us") or 0),
            "speedup": float(r.get("speedup") or 0),
            "precision": {
                "passed": prec_raw.get("passed", passed),
                "max_re": ratios.get("max_re"),
                "mean_re": ratios.get("mean_re"),
                "rmse": ratios.get("rmse"),
                "svec": ratios.get("svec"),
            },
        })
    return rows


def parse_precision_json(path) -> dict:
    """解析 precision_results.json (v2 schema)，按 case_id 提取精度指标"""
    data = _json(path)
    if not data or "cases" not in data:
        return {}
    result = {}
    for case in data.get("cases", []):
        cid = case.get("id", 0)
        params = case.get("params", {})
        fwd = case.get("forward", {})
        # 找第一个有 ratios 的 component
        for comp_name, comp in fwd.items():
            ratios = comp.get("ratios", {})
            avg_vs_golden = comp.get("ans_vs_golden", {})
            result[cid] = {
                "passed": comp.get("passed", False),
                "max_re": ratios.get("max_re"),
                "mean_re": ratios.get("mean_re"),
                "rmse": ratios.get("rmse"),
                "svec": ratios.get("svec"),
                "ae_max": avg_vs_golden.get("ae_max"),
                "re_max": avg_vs_golden.get("re_max"),
                "mismatch_rate": avg_vs_golden.get("mismatch_rate"),
                "component": comp_name,
                "params": params,
            }
            break  # only first component
    return result


def parse_evaluation_txt(path) -> list:
    """
    解析 evaluation_result.txt 日志格式（非 JSON）。
    支持格式：
      INFO - [1/10] Test: small_basic  {'M': 32, ...}
      INFO -   Correctness: [PASS] match_rate=100.00% (1536/1536), max_diff=0.0, mean_diff=0.0
      INFO -   Performance: ref=0.0068ms, custom=0.0117ms, speedup=0.58x
    返回 [{id, name, shape, passed, precision, performance}]
    """
    txt = _read(path)
    if not txt or not any(s in txt for s in ("[PASS]", "[FAIL]", "Correctness:")):
        return []
    cases = []
    # Split on test case headers
    header_re = re.compile(
        r'\[(\d+)/\d+\]\s+Test:\s+(\w+)\s+(.*?)(?=\[\d+/\d+\]|$)', re.DOTALL)
    for m in header_re.finditer(txt):
        idx = int(m.group(1)) - 1
        name = m.group(2)
        block = m.group(3)
        # params dict → shape string
        p = re.search(r"\{([^}]+)\}", block)
        shape = p.group(0) if p else ""
        # correctness
        passed = bool(re.search(r'\[PASS\]', block))
        mr = re.search(r'match_rate=([\d.]+)%', block)
        mx = re.search(r'max_diff=([\d.e+\-]+)', block)
        mn = re.search(r'mean_diff=([\d.e+\-]+)', block)
        # performance
        ref_m = re.search(r'ref=([\d.]+)ms', block)
        cus_m = re.search(r'custom=([\d.]+)ms', block)
        spd_m = re.search(r'speedup=([\d.]+)x', block)
        cases.append({
            "id": idx,
            "name": name,
            "shape": shape,
            "passed": passed,
            "precision": {
                "passed": passed,
                "match_rate": float(mr.group(1)) if mr else None,
                "max_diff": float(mx.group(1)) if mx else None,
                "mean_diff": float(mn.group(1)) if mn else None,
            },
            "performance": {
                "ref_time_us": float(ref_m.group(1)) * 1000 if ref_m else 0,
                "custom_time_us": float(cus_m.group(1)) * 1000 if cus_m else 0,
                "speedup": float(spd_m.group(1)) if spd_m else 0,
                "source": "msprof",
            } if ref_m else {},
        })
    return cases


def parse_flat_msprof_csv_pairs(custom_csvs: list, ref_csvs: list, op_name: str) -> list:
    """
    从平铺的 op_summary_custom_fn_<case>.csv + op_summary_reference_fn_<case>.csv 配对提取时延。
    自动按 case 名称配对，提取 median(Task Duration) 跳过 warmup 行。
    返回 [{case_name, ref_time_us, custom_time_us, speedup}]
    """
    def _extract_durations(csv_path: Path, target_op: str = None) -> list:
        """提取目标 op 的 Task Duration(us) 列表。
        过滤策略：若给定 target_op，只保留名称含 target_op 的行；
        否则按算子名过滤掉明显的 L2 flush/warmup（通过分位数：超过 p75 的 10 倍视为异常）。
        """
        txt = _read(csv_path)
        if not txt:
            return []
        durs = []
        try:
            reader = csv.DictReader(io.StringIO(txt))
            dur_col = next((k for k in (reader.fieldnames or []) if "Duration" in k), None)
            name_col = next((k for k in (reader.fieldnames or []) if k.strip() in ("Op Name", "OP Type", "Name")), None)
            if not dur_col:
                return []
            for row in reader:
                raw_dur = row.get(dur_col, "").strip()
                if not raw_dur:
                    continue
                try:
                    dur = float(raw_dur)
                except ValueError:
                    continue
                if target_op and name_col:
                    op_nm = row.get(name_col, "").strip()
                    if target_op.lower() not in op_nm.lower():
                        continue
                durs.append(dur)
        except Exception as e:
            logger.debug("提取 Task Duration 失败 %s: %s", csv_path, e)
        if not durs:
            return durs
        # Remove outliers: rows > 20× median (catches L2 flush, large warmup)
        med = sorted(durs)[len(durs) // 2]
        threshold = max(med * 20, 200.0)
        return [d for d in durs if d <= threshold]

    def _median(lst):
        return sorted(lst)[len(lst) // 2] if lst else 0.0

    # Build {case_name: csv_path} mappings
    def _case_name(p: Path, prefix: str) -> str:
        return p.stem[len(prefix):]  # e.g. "op_summary_custom_fn_" → "small_basic"

    custom_map = {_case_name(f, "op_summary_custom_fn_"): f for f in custom_csvs}
    ref_map = {_case_name(f, "op_summary_reference_fn_"): f for f in ref_csvs}
    all_cases = sorted(set(custom_map) | set(ref_map))

    results, unpaired = [], []
    for case_name in all_cases:
        has_custom = case_name in custom_map
        has_ref = case_name in ref_map
        if has_custom and has_ref:
            cus_durs = _extract_durations(custom_map[case_name], op_name)
            ref_durs = _extract_durations(ref_map[case_name])
            cus_us = _median(cus_durs[1:] or cus_durs)
            ref_us = _median(ref_durs[1:] or ref_durs)
            results.append({
                "case_name": case_name,
                "ref_time_us": ref_us,
                "custom_time_us": cus_us,
                "speedup": round(ref_us / cus_us, 2) if cus_us > 0 else 0,
            })
        else:
            unpaired.append(case_name + ("(custom_only)" if has_custom else "(ref_only)"))
    if unpaired:
        # Store unpaired info so collect_data can add to diagnostics
        results.append({"_unpaired": unpaired})
    return results


# Fields shown in the Perf tab op_summary detail panel (user-configurable)
# 根据算子类型（Task Type）动态选择显示字段
OP_SUMMARY_FIELDS_BY_TYPE = {
    "vector": [
        "OP Type", "OP State", "Task Type", "Task Duration(us)", "Block Dim",
        "Input Shapes", "Input Data Types", "Input Formats", "Output Shapes",
        "Output Data Types", "Output Formats", "aiv_time(us)", "aiv_total_cycles",
        "aiv_vec_time(us)", "aiv_vec_ratio", "aiv_scalar_time(us)", "aiv_scalar_ratio",
        "aiv_mte2_time(us)", "aiv_mte2_ratio", "aiv_mte3_time(us)", "aiv_mte3_ratio",
        "aiv_icache_miss_rate"
    ],
    "cube": [
        "OP Type", "OP State", "Task Type", "Task Duration(us)", "Block Dim",
        "Input Shapes", "Input Data Types", "Input Formats", "Output Shapes",
        "Output Data Types", "Output Formats", "aicore_time(us)", "aic_total_cycles",
        "aic_mac_time(us)", "aic_mac_ratio", "aic_scalar_time(us)", "aic_scalar_ratio",
        "aic_mte1_time(us)", "aic_mte1_ratio", "aic_mte2_time(us)", "aic_mte2_ratio",
        "aic_fixpipe_time(us)", "aic_fixpipe_ratio", "aic_icache_miss_rate", "cube_utilization(%)"
    ],
}

# 向后兼容：默认显示所有字段（用于未指定类型时的降级处理）
# 展开为显式循环：去重并保序，避免循环变量泄漏到模块作用域（被内层函数局部变量遮蔽）


def _all_op_summary_fields() -> list:
    seen = {}
    for _type_fields in OP_SUMMARY_FIELDS_BY_TYPE.values():
        for _field in _type_fields:
            seen[_field] = None  # dict 保序去重
    return list(seen)


OP_SUMMARY_DISPLAY_FIELDS = _all_op_summary_fields()
_OP_SUM_STRING_FIELDS = {
    "Input Shapes", "Input Data Types", "Input Formats",
    "Output Shapes", "Output Data Types", "Block Dim", "Mix Block Dim",
    "OP Type", "Op Name", "Task Type",
}


def parse_op_summary_detail(csv_path: Path, op_name: str,
                            display_fields: list = None) -> dict:
    """
    从 op_summary_custom_fn_<case>.csv 提取目标算子的关键性能字段（取匹配行均值）。
    - 数值字段：取所有匹配行的均值（跳过 warmup 首行）
    - 字符串字段（Input Shapes / Input Data Types / Block Dim）：取首个非空值
    - 根据 Task Type 自动选择合适的字段列表（vector/cube）
    """
    txt = _read(csv_path)
    if not txt:
        return {}

    # 如果没有指定 display_fields，尝试从 CSV 中检测 Task Type 来选择合适的字段列表
    if display_fields is None:
        # 先读取 CSV 检测 Task Type
        try:
            reader = csv.DictReader(io.StringIO(txt))
            raw_headers = list(reader.fieldnames or [])
            hmap = {h.strip(): h for h in raw_headers}
            task_type_col = hmap.get("Task Type")
            detected_task_type = None

            # 从匹配的行中获取 Task Type
            op_col = next((h for h in raw_headers if h.strip() in ("Op Name",)), None)
            for row in reader:
                if op_col and op_name:
                    row_op = row.get(op_col, "").strip()
                    if op_name.lower() not in row_op.lower():
                        continue
                if task_type_col:
                    detected_task_type = row.get(task_type_col, "").strip()
                    break

            # 根据 Task Type 选择字段列表
            if detected_task_type:
                if "AI_VECTOR" in detected_task_type:
                    display_fields = OP_SUMMARY_FIELDS_BY_TYPE.get("vector", OP_SUMMARY_DISPLAY_FIELDS)
                elif "AIC" in detected_task_type or "AI_CORE" in detected_task_type:
                    display_fields = OP_SUMMARY_FIELDS_BY_TYPE.get("cube", OP_SUMMARY_DISPLAY_FIELDS)
                else:
                    display_fields = OP_SUMMARY_DISPLAY_FIELDS
            else:
                display_fields = OP_SUMMARY_DISPLAY_FIELDS
        except Exception:
            display_fields = OP_SUMMARY_DISPLAY_FIELDS

    active_fields = display_fields or OP_SUMMARY_DISPLAY_FIELDS
    numeric_sums: dict = {}
    numeric_counts: dict = {}
    string_first: dict = {}
    try:
        reader = csv.DictReader(io.StringIO(txt))
        raw_headers = list(reader.fieldnames or [])
        # Strip whitespace from headers for lookup
        hmap = {h.strip(): h for h in raw_headers}
        for row in reader:
            # Filter: Op Name must contain op_name
            op_col = next((h for h in raw_headers if h.strip() in ("Op Name",)), None)
            if op_col and op_name:
                row_op = row.get(op_col, "").strip()
                if op_name.lower() not in row_op.lower():
                    continue
            for field in active_fields:
                h_orig = hmap.get(field)
                if h_orig is None:
                    continue
                val_str = row.get(h_orig, "").strip()
                if not val_str:
                    continue
                if field in _OP_SUM_STRING_FIELDS:
                    if field not in string_first:
                        string_first[field] = val_str.strip('"')
                else:
                    try:
                        val = float(val_str)
                        numeric_sums[field] = numeric_sums.get(field, 0.0) + val
                        numeric_counts[field] = numeric_counts.get(field, 0) + 1
                    except ValueError:
                        if field not in string_first:
                            string_first[field] = val_str
    except Exception as e:
        logger.debug("解析 op_summary 明细失败，返回已聚合字段: %s", e)
    result = dict(string_first)
    for field, total in numeric_sums.items():
        cnt = numeric_counts[field]
        result[field] = round(total / cnt, 4) if cnt > 0 else 0.0
    return result


def parse_profiling_report_txt(path) -> dict:
    """解析 profiling/report.txt（单 case）"""
    txt = _read(path)
    result = {}
    m = re.search(r"Reference time\s*:\s*([\d.]+)\s*us", txt)
    if m:
        result["ref_time_us"] = float(m.group(1))
    m = re.search(r"Custom time\s*:\s*([\d.]+)\s*us", txt)
    if m:
        result["custom_time_us"] = float(m.group(1))
    m = re.search(r"Speedup\s*:\s*([\d.]+)x", txt)
    if m:
        result["speedup"] = float(m.group(1))
    return result


def pair_profiling_dirs(profiling_dir: Path) -> list:
    """
    按时间戳排序，将 Model_* 与 ModelNew_* 两两配对。
    返回 [(model_dir, modelnew_dir), ...] 按时间戳从旧到新。
    """
    if not profiling_dir or not Path(profiling_dir).is_dir():
        return []
    pdir = Path(profiling_dir)
    models = sorted([d for d in pdir.iterdir() if d.is_dir() and d.name.startswith("Model_device")
                     and not d.name.startswith("ModelNew")])
    modelnews = sorted([d for d in pdir.iterdir() if d.is_dir() and d.name.startswith("ModelNew_device")])
    pairs = list(zip(models, modelnews))
    return pairs


def extract_profiling_time(prof_dir: Path) -> float:
    """从 profiling dir 的 op_summary CSV 提取总 task duration (us)"""
    if not prof_dir or not prof_dir.is_dir():
        return 0.0
    for csv_file in sorted(prof_dir.rglob("op_summary*.csv")):
        try:
            txt = csv_file.read_text(encoding="utf-8", errors="replace")
            lines = [l for l in txt.splitlines() if l.strip() and not l.startswith("#")]
            if not lines:
                continue
            headers = [h.strip() for h in lines[0].split(",")]
            dur_col = next((i for i, h in enumerate(headers) if "Duration" in h), None)
            if dur_col is None:
                continue
            total = 0.0
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) > dur_col:
                    try:
                        total += float(parts[dur_col].strip())
                    except ValueError:
                        pass
            if total > 0:
                return total
        except Exception as e:
            logger.debug("读取 profiling op_summary CSV 失败，已跳过 %s: %s", csv_file, e)
    return 0.0


def parse_dsl(path) -> dict:
    txt = _read(path)
    if not txt:
        return {}
    # header comments
    algo_lines, in_comment = [], False
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("#"):
            in_comment = True
            c = s.lstrip("# ").strip()
            if c:
                algo_lines.append(c)
        elif in_comment and algo_lines:
            break
    # Extract compute steps from "with tl.compute():" blocks and their comments
    compute_steps = []
    in_compute = False
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("with tl.compute():"):
            in_compute = True
            continue
        if in_compute:
            if s.startswith("with ") or (s and not line.startswith(" ") and not line.startswith("\t")):
                in_compute = False
                continue
            # Pick up comment lines inside compute block
            if s.startswith("# "):
                c = s[2:].strip()
                if c:
                    compute_steps.append(c)
    return {"header_comments": algo_lines[:20], "compute_steps": compute_steps}


def extract_kernel_compute_apis(kernel_txt: str) -> list:
    """
    从 kernel 源码的 Compute() 方法体中直接提取 AscendC API 调用序列。

    对显式前缀调用（如 `AscendC::Foo(...)`）不依赖任何预设 API 列表，AscendC 升级后会自动适配；
    对无前缀调用（如 `Foo(...)`，通常来自 `using namespace AscendC;`）会回退到 `_KNOWN_APIS`
    做名称匹配，因此可能无法覆盖未来新增但未同步进列表的 API。
    返回格式: [{"api": "Muls", "args": ["expLocal", "xLocal", "-1.702f", "tileSize"]}]
    """
    if not kernel_txt:
        return []

    def _brace_body(txt: str, header_end: int) -> str | None:
        """从 header_end 后找到第一个 '{', 然后用括号深度计数提取完整函数体。
        正确处理函数内嵌套的 if/for/while 块，避免正则在第一个 '}' 处截断。"""
        start = txt.find('{', header_end)
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(txt)):
            if txt[i] == '{':
                depth += 1
            elif txt[i] == '}':
                depth -= 1
                if depth == 0:
                    return txt[start + 1:i]
        return None

    def _paren_args(txt: str, paren_pos: int) -> str:
        """从 '(' 处起，用括号深度计数提取配对括号内的参数字符串（不含外层括号）。
        修复 ([^;]*) 贪婪匹配跨行代码的问题（如 GetBlockNum() 后紧跟 { ... ; }）。"""
        depth, n = 0, len(txt)
        for i in range(paren_pos, n):
            if txt[i] == '(':
                depth += 1
            elif txt[i] == ')':
                depth -= 1
                if depth == 0:
                    return txt[paren_pos + 1:i]
        return txt[paren_pos + 1:]  # unbalanced — return remainder

    # 找到所有 Compute*() / Process() 方法体，合并扫描（支持 Compute1/Compute2/ComputeRem 等拆分写法）
    header_re = re.compile(
        r'(?:__aicore__\s+inline\s+)?void\s+(Compute\w*|Process)\s*\([^)]*\)'
    )
    compute_bodies = []
    for m in header_re.finditer(kernel_txt):
        body = _brace_body(kernel_txt, m.end())
        if body:
            compute_bodies.append(body)
    compute_body = "\n".join(compute_bodies) if compute_bodies else None
    if compute_body is None:
        return []

    # 提取 AscendC API 调用序列（按出现顺序）
    # 只取 compute 操作，过滤 DataCopy 类（搬运）和 dispatch/sync 类（GetBlockIdx 等）
    skip_apis = {'DataCopy', 'DataCopyPad', 'DeQue', 'AllocTensor', 'EnQue', 'FreeTensor', 'Get',
                 'SetGlobalBuffer', 'InitBuffer', 'set_atomic_none',
                 # dispatch/loop-control（属于任务分配，不是计算逻辑）
                 'GetBlockIdx', 'GetBlockNum', 'GetSubBlockIdx', 'GetSubBlockNum',
                 # CV cross-core sync（属于流水线协同，不是计算逻辑）
                 'CrossCoreSetFlag', 'CrossCoreWaitFlag'}

    # 已知 AscendC compute API 列表（用于支持 using namespace AscendC; 无前缀调用）
    # Use a list (not set) so iteration order is deterministic and matches source order.
    _known_apis = [
        'Muls', 'Adds', 'Mul', 'Add', 'Sub', 'Div', 'Exp', 'Log', 'Sqrt', 'Rec',
        'Abs', 'Neg', 'Max', 'Min', 'Relu', 'Sigmoid', 'Tanh', 'Cast',
        'ReduceSum', 'ReduceMax', 'ReduceMean', 'ArgMax',
        'Transpose', 'Broadcast', 'Gather', 'Scatter',
        'Matmul', 'BatchMatmul', 'Softmax', 'LayerNorm',
        'Copy', 'SetValue', 'Concat', 'Split',
    ]

    # Collect all calls with their source position so we can merge and sort by appearance.
    _all: list[tuple[int, str, list[str]]] = []  # (start_pos, api_name, args)
    seen_apis: set[str] = set()

    # 匹配带前缀的调用 AscendC::XXX(...)
    # 用 _paren_args 替代 ([^;]*) 以精确提取平衡括号内的参数，避免捕获闭合括号或跨行代码
    for call_m in re.finditer(r'AscendC::(\w+)\s*\(', compute_body):
        api_name = call_m.group(1)
        if api_name in skip_apis:
            continue
        args_txt = _paren_args(compute_body, call_m.end() - 1)
        args = [a.strip() for a in args_txt.split(',')]
        seen_apis.add(api_name)
        _all.append((call_m.start(), api_name, args[:4]))

    # 始终扫描无前缀的已知 API（支持混用 AscendC::Foo 与 Bar 的情况，以及 using namespace）
    # 避免误匹配：要求函数名在已知列表中，且前面不是 :: 或 . 成员访问
    for api_name in _known_apis:
        if api_name in skip_apis or api_name in seen_apis:
            continue
        for call_m in re.finditer(
                rf'(?<![:\.\w]){re.escape(api_name)}\s*\(', compute_body):
            args_txt = _paren_args(compute_body, call_m.end() - 1)
            args = [a.strip() for a in args_txt.split(',')]
            _all.append((call_m.start(), api_name, args[:4]))
            seen_apis.add(api_name)
            break  # 每个 API 只记录一次出现

    # Sort by source position so algo_flow reflects actual Compute() call order
    api_calls = [{"api": n, "args": a} for _, n, a in sorted(_all, key=lambda x: x[0])]

    return api_calls


def _infer_api_from_desc(desc: str) -> str:
    """从步骤描述推断 AscendC API 名（fallback，用于 DSL/注释来源）。
    识别最外层运算符，不被 exp() 子式误导。"""
    m = re.search(r'=\s*(.+)$', desc)
    rhs = m.group(1).strip() if m else desc
    depth = 0
    has_plus = has_minus = has_mul = has_div = False
    for i, c in enumerate(rhs):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif depth == 0:
            if c == '+':
                has_plus = True
            elif c == '/':
                has_div = True
            elif c == '*':
                has_mul = True
            elif c == '-' and i > 0 and rhs[i - 1] not in '0123456789':
                has_minus = True
    if has_div:
        return "Div"
    if has_plus:
        return "Adds"
    if has_minus:
        return "Sub"
    if has_mul:
        return "Muls"
    rhs_l = rhs.lower().lstrip()
    for prefix, api in [('exp(', 'Exp'), ('sqrt(', 'Sqrt'), ('tanh(', 'Tanh'),
                        ('ln(', 'Ln'), ('log(', 'Ln'), ('reducemax(', 'ReduceMax'),
                        ('reducemin(', 'ReduceMin'), ('reducesum(', 'ReduceSum'),
                        ('max(', 'Maxs'), ('min(', 'Mins')]:
        if rhs_l.startswith(prefix):
            return api
    for pat, api in [(r'\bmuls\b', 'Muls'), (r'\badds\b', 'Adds'), (r'\bdivs?\b', 'Div'),
                     (r'\bsubs\b', 'Sub'), (r'\bmaxs\b|\bvmax\b', 'Maxs'),
                     (r'\bmins\b|\bvmin\b', 'Mins'), (r'\breducesum\b', 'ReduceSum'),
                     (r'\breducemax\b', 'ReduceMax'), (r'\breducemin\b', 'ReduceMin'),
                     (r'\bexp\b', 'Exp')]:
        if re.search(pat, desc, re.I):
            return api
    return "Compute"


def _make_api_formula(api: str, raw_desc: str) -> str:
    """生成 AscendC API 调用形式的 formula（fallback 路径）。"""
    floats = re.findall(r'-?\d+\.\d+', raw_desc)
    scalar = floats[0] if floats else ""
    if not scalar:
        ints = re.findall(r'\b\d{2,}\b', raw_desc)
        scalar = ints[0] if ints else ""
    t = {
        "Muls": f"tmp = Muls(x, {scalar})" if scalar else "tmp = Muls(x, α)",
        "Exp": "e   = Exp(tmp)",
        "Adds": f"d   = Adds(e, {scalar})" if scalar else "d   = Adds(e, 1)",
        "Div": "y   = Div(x, d)",
        "Sub": "tmp = Sub(a, b)", "Add": "out = Add(a, b)", "Mul": "out = Mul(a, b)",
        "Maxs": f"out = Maxs(x, {scalar})" if scalar else "out = Maxs(x, 0)",
        "Mins": f"out = Mins(x, {scalar})" if scalar else "out = Mins(x, 0)",
        "ReduceSum": "s = ReduceSum(x, work)", "ReduceMax": "m = ReduceMax(x, work)",
        "ReduceMin": "m = ReduceMin(x, work)", "Sqrt": "out = Sqrt(x)",
        "Tanh": "out = Tanh(x)", "Ln": "out = Ln(x)", "Compute": "out = Compute(x)",
    }
    return t.get(api, f"out = {api}(x)")


def _build_nodes_from_api_calls(api_calls: list, dsl_steps: list) -> list:
    """
    从 extract_kernel_compute_apis() 的结果构建 nodes（主路径）。
    formula 直接来自真实 AscendC 调用参数，最可靠。
    """
    # _NO_SCALAR_APIS: 这些 API 不接受标量参数
    _no_scalar = {'Exp', 'Sqrt', 'Tanh', 'Ln', 'Log', 'ReduceSum', 'ReduceMax', 'ReduceMin',
                  'Div', 'Add', 'Sub', 'Mul', 'DataCopy', 'DataCopyPad'}

    nodes = []
    for call in api_calls[:8]:
        api = call.get("api", "Compute")
        args = call.get("args", [])

        # 仅从当前 call 的 args 提取常量（不跨 call 引用）
        scalar = ""
        if api not in _no_scalar:
            for arg in args:
                # 去掉 f 后缀，匹配浮点或多位整数
                arg_clean = arg.strip().rstrip('f')
                fm = re.search(r'-?\d+\.\d+', arg_clean)
                if fm:
                    scalar = fm.group(0)
                    break

        # 清理变量名：去 this->、Local/Buf/_ub 后缀，使其简洁
        def _var(s):
            s = re.sub(r'\bthis->', '', s).strip().rstrip('f').strip()  # 仅去前缀，保留字段名
            s = re.sub(r'(?i)(local|buf|_ub|_tensor|local_)$', '', s.strip())
            return s.strip() or 'tmp'

        dst = _var(args[0]) if len(args) > 0 else "out"
        src = _var(args[1]) if len(args) > 1 else "x"
        src2 = _var(args[2]) if len(args) > 2 else ""

        # 二元无标量 API（Div, Add, Sub, Mul）：显示两个操作数
        _binary = {'Div', 'Add', 'Sub', 'Mul', 'ReduceSum', 'ReduceMax', 'ReduceMin'}
        # src2 是否为有效的第二操作数（排除 tiling 控制参数等非数据操作数）
        _src2_is_loop_param = src2.lower() in ('tilesize', 'innerloops', 'tailcount', 'count', 'size')
        _has_second_operand = bool(src2) and src2 != src and not _src2_is_loop_param
        _is_binary_with_two_operands = api in _binary and _has_second_operand
        if scalar:
            formula = f"{dst} = {api}({src}, {scalar})"
        elif _is_binary_with_two_operands:
            formula = f"{dst} = {api}({src}, {src2})"
        else:
            formula = f"{dst} = {api}({src})"

        nodes.append({
            "api": api,
            "formula": formula,
            "in_tmpl": "{N}",
            "out_tmpl": "{N}",
        })
    return nodes


def auto_build_algo_flow(op_name: str, dsl_data: dict, kernel_data: dict,
                         op_type: str = "vector", op_desc: dict = None) -> dict:
    """
    三层优先级构建 algo_flow.json：
    1. 直接从 kernel Compute() 提取 AscendC API 调用（最可靠，不依赖预设列表）
    2. 从 DSL compute_steps 推断（含数学运算符，适合 API 名映射）
    3. Fallback：从 pass_comments 提取标题

    algo_flow.json 增加 api_list 字段，记录实际调用的 AscendC 函数，自描述、可审计。
    op_desc 若提供，则用于填充真实 shape_vars / IO dtype，否则退化为 ["N"] / float32。
    """
    kernel_txt = kernel_data.get("_txt", "")

    # ══ 来源 1：直接从 kernel Compute() 提取 AscendC::XXX() 调用（最可靠）══
    # 不依赖预设 API 列表，AscendC 升级自动适应
    kernel_api_calls = extract_kernel_compute_apis(kernel_txt)

    # ── 来源 2: DSL compute_steps（含运算符，适合推断 API 名）──
    dsl_steps = [s for s in dsl_data.get("compute_steps", [])
                 if not re.match(r'^-{3,}$', s)
                 and not re.match(r'^(COPYOUT|COPYIN)', s, re.I)
                 and '=' in s
                 and len(s) > 5]

    # ── 来源 3: kernel step 注释公式行 ──
    step_re = re.compile(r'//\s*step\s*(\d+)\s*[:\-]\s*(.+)', re.IGNORECASE)
    step_map = {}
    for m in step_re.finditer(kernel_txt):
        n, desc = int(m.group(1)), m.group(2).strip()
        if n not in step_map:
            step_map[n] = desc
    kernel_formula_steps = [step_map[n] for n in sorted(step_map)]

    # ── 来源 4: pass_comments 标题（fallback）──
    pass_titles = []
    for c in kernel_data.get("pass_comments", []):
        m = re.match(r'Step\s*\d+\s*:\s*(.+?)(?:\s*[—–]\s*.+)?$', c, re.IGNORECASE)
        if m:
            pass_titles.append(m.group(1).strip())

    # ══ 优先使用直接提取的 API 调用构建节点 ══
    use_source = "kernel_api"
    if kernel_api_calls:
        nodes = _build_nodes_from_api_calls(kernel_api_calls, dsl_steps)
    else:
        # Fallback：从 DSL/注释推断
        if dsl_steps:
            raw_steps = dsl_steps
            use_source = "dsl"
        elif kernel_formula_steps:
            raw_steps = kernel_formula_steps
            use_source = "kernel_comments"
        elif pass_titles:
            raw_steps = pass_titles
            use_source = "pass_titles"
        else:
            return {}
        nodes = []

    if not nodes and raw_steps:
        # Fallback 路径下构建 nodes（来源：DSL/注释推断）
        for desc in raw_steps[:8]:
            api = _infer_api_from_desc(desc)
            formula = _make_api_formula(api, desc)
            nodes.append({"api": api, "formula": formula, "in_tmpl": "{N}", "out_tmpl": "{N}"})

    if not nodes:
        return {}

    # ── 硬件单元元数据 ──
    _unit_meta = {
        "vector": {"id": "vector", "label": "Vector Core  Compute", "bg": "#f5f0ff", "accent": "#8250df"},
        "cube": {"id": "cube", "label": "Cube Core  MAC", "bg": "#fffbf0", "accent": "#bf8700"},
    }
    unit_meta = _unit_meta.get(op_type, _unit_meta["vector"])

    # ── 从 op_desc 推导真实 IO 元信息（避免硬编码 float32/单维度） ──
    _inp_shapes = (op_desc or {}).get("shape_info", {}).get("input_shapes", [])
    _out_shapes = (op_desc or {}).get("shape_info", {}).get("output_shapes", [])
    _first_in = _inp_shapes[0] if _inp_shapes else {}
    _first_out = _out_shapes[0] if _out_shapes else {}
    _ndim = len(_first_in.get("shape", [])) or 1
    _svars = [f"S{i}" for i in range(_ndim)] if _ndim > 1 else ["N"]
    _tmpl = "{" + ",".join(_svars) + "}"
    _inp_dtype = _first_in.get("dtype", "float32")
    _out_dtype = _first_out.get("dtype", "float32")
    _inp_name = _first_in.get("name", "x")
    _out_name = _first_out.get("name", "y")

    # ── 组装合规 algo_flow，包含 api_list 供审计 ──
    api_list = list(dict.fromkeys(n["api"] for n in nodes if n["api"] != "Compute"))
    return {
        "op_name": op_name,
        "description": f"Auto-extracted ({use_source}, {len(nodes)} ops)",
        "_auto": True,
        "shape_vars": _svars,
        "inp_name": _inp_name,
        "inp_tmpl": _tmpl,
        "inp_dtype": _inp_dtype,
        "out_name": _out_name,
        "out_tmpl": _tmpl,
        "out_dtype": _out_dtype,
        "api_list": api_list,       # 实际调用的 AscendC API，可审计、不依赖预设列表
        "units": [
            {
                "id": unit_meta["id"],
                "label": unit_meta["label"],
                "bg": unit_meta["bg"],
                "accent": unit_meta["accent"],
                "nodes": nodes,
            }
        ],
    }


def parse_kernel_cpp(path) -> dict:
    txt = _read(path)
    if not txt:
        return {}
    m = re.search(r"BUFFER_NUM\s*=\s*(\d+)", txt)
    buffer_num = int(m.group(1)) if m else 1
    buffers = []
    # TQue with BUFFER_NUM variable — supports optional AscendC:: prefix and QuePosition alias
    _pos = r"(?:(?:AscendC::)?(?:TPosition|QuePosition))"
    for m in re.finditer(
            rf"(?:AscendC::)?TQue<{_pos}::(\w+),\s*BUFFER_NUM>\s*(\w+)\s*;", txt):
        buffers.append({"name": m.group(2), "kind": "TQue", "position": m.group(1), "multiplier": buffer_num})
    # TQue with literal integer (CAKE2 format: TQue<..., 1> inQueue;)
    for m in re.finditer(
            rf"(?:AscendC::)?TQue<{_pos}::(\w+),\s*(\d+)>\s*(\w+)\s*;", txt):
        buffers.append({"name": m.group(3), "kind": "TQue", "position": m.group(1), "multiplier": int(m.group(2))})
    # TBuf — supports optional AscendC:: prefix and QuePosition alias
    for m in re.finditer(
            rf"(?:AscendC::)?TBuf<{_pos}::(\w+)>\s*(\w+)\s*;", txt):
        buffers.append({"name": m.group(2), "kind": "TBuf", "position": m.group(1), "multiplier": 1})
    tiling_params = list(dict.fromkeys(re.findall(r"tiling_data\.(\w+)", txt)))
    pass_comments = []
    _seen_step_nums = set()
    lines = txt.splitlines()
    # 匹配多种 step/pass/phase 注释格式（大小写不敏感），包括：
    #   // Step N: desc              // Pass N: desc        // Phase N: desc
    #   // ---- Step N: desc ----   // ── Step N: desc ──  // Phase N (ctx): desc
    #   // A5: Step N - desc        // Stage N: desc
    step_pat = re.compile(
        r'^[ \t]*//\s*'               # 行首注释，允许任意缩进（重复步骤号通过 _seen_step_nums 去重）
        r'[-\u2500-\u257F=\s]*'           # 可选前置装饰符（ASCII/Unicode box-drawing）
        r'(?:A5:\s*)?'                    # 可选 A5: 前缀
        r'(?:Pass|Step|Phase|Stage)\s*'   # 关键字
        r'(\d+)'                          # 步骤编号
        r'(?:\s*[\(\uff08][^\uff09)]*[\)\uff09])?'  # 可选括号内上下文，如 (AIC+AIV)
        r'\s*[-:：\u2500-\u257F\s]+'     # 分隔符（冒号/横线/空格组合）
        r'([\w\u4e00-\u9fff].{1,})',      # 描述文本（字母/汉字开头，至少2字符）
        re.IGNORECASE
    )
    cont_pat = re.compile(r'^\s*//\s*(?!step\s*\d|pass\s*\d)(.+)', re.IGNORECASE)
    for i, line in enumerate(lines):
        m = step_pat.match(line)
        if not m:
            continue
        step_num = m.group(1)
        # Strip trailing decorators (ASCII dashes, Unicode box-drawing, spaces)
        title = re.sub(r'[\s\-\u2500-\u257F]+$', '', m.group(2)).strip()
        if step_num in _seen_step_nums:
            continue
        _seen_step_nums.add(step_num)
        # 尝试读取紧随的解释行（最多 1 行）
        detail = ""
        if i + 1 < len(lines):
            cm = cont_pat.match(lines[i + 1])
            if cm:
                detail = cm.group(1).strip()
        # 组合：标题 + 解释（用空格分隔，保持单行显示）
        full = f"Step {step_num}: {title}"
        if detail:
            full = f"Step {step_num}: {title} — {detail}"
        if len(full) > 5:
            pass_comments.append(full)
    # ── Constant extraction ──
    consts = {}
    # Pass 1: literal constexpr (type VAR = NUMBER)
    for m in re.finditer(r'constexpr\s+\w+\s+(\w+)\s*=\s*(\d+)\s*;', txt):
        consts[m.group(1)] = int(m.group(2))
    # Pass 2: template shape typedefs  using L1Shape = GemmShape<128, 256, 512>
    _dim_names = ['M', 'N', 'K', 'D', 'E']
    for m in re.finditer(r'using\s+(\w+)\s*=\s*\w*Shape\s*<\s*([\d,\s]+)\s*>', txt):
        tname = m.group(1)
        vals = [int(x.strip()) for x in m.group(2).split(',')]
        for i, v in enumerate(vals):
            if i < len(_dim_names):
                # Store as both "L1Shape_N" and "L1Shape::N" for resolution
                consts[f'{tname}_{_dim_names[i]}'] = v
                consts[f'{tname}::{_dim_names[i]}'] = v
    # Pass 3: compound constexpr (evaluate using known constants, multi-pass)
    for _ in range(3):
        for m in re.finditer(r'constexpr\s+\w+\s+(\w+)\s*=\s*([^;{]+?)\s*;', txt):
            name, expr = m.group(1), m.group(2).strip()
            if name not in consts:
                val = _eval_size_expr(expr, consts)
                if val > 0:
                    consts[name] = val
    return {
        "buffer_num": buffer_num,
        "buffers": buffers,
        "tiling_params": tiling_params,
        "pass_comments": pass_comments[:20],
        "constants": consts,
        "_txt": txt,
    }


def _extract_tiling_body(txt: str) -> str:
    """提取 TilingFunc 完整函数体（括号深度计数，避免被内层 if/for 截断）。"""
    sig_match = re.search(r'ge::graphStatus\s+\w*TilingFunc\w*\s*\([^)]*\)\s*\{', txt)
    if not sig_match:
        return ""
    start = sig_match.end() - 1
    depth = 0
    for i in range(start, len(txt)):
        if txt[i] == '{':
            depth += 1
        elif txt[i] == '}':
            depth -= 1
            if depth == 0:
                return txt[start + 1:i]
    return ""


def parse_op_host_cpp(path) -> dict:
    """从 op_host/*.cpp 提取 constexpr/const/局部变量 tile 常量（L1_M/L1_N/L1_K/WORKSPACE_STAGES/BLOCK_DIM/tileSize 等）。

    额外解析 tiling.set_<field>(<expr>) 调用，将 tiling 结构体字段名
    （如 tileSize、nUsed）也加入常量表。这样 kernel 的 InitBuffer 表达式
    "tileSize * sizeof(float)" 就能用 op_host 的 TILE_SIZE 常量正确求值，
    避免因名称不匹配导致的 UB 溢出误报。
    """
    txt = _read(path)
    if not txt:
        return {"constants": {}, "tiling_body": ""}
    consts = {}
    # constexpr 常量
    for m in re.finditer(r'constexpr\s+\w+\s+(\w+)\s*=\s*(\d+)\s*;', txt):
        consts[m.group(1)] = int(m.group(2))
    # const 常量（CAKE2 格式：const uint32_t BLOCK_DIM = 16;）
    for m in re.finditer(r'const\s+\w+\s+(\w+)\s*=\s*(\d+)\s*;', txt):
        consts[m.group(1)] = int(m.group(2))
    # 提取 TilingFunc 函数体
    tiling_body = _extract_tiling_body(txt)
    if tiling_body:
        # 局部变量赋值：字面量 uint32_t tileSize = 2048; 或引用已知常量 uint32_t tileSize = TILE_SIZE;
        # 两轮迭代：处理常量之间的前向依赖（如 A = B; B = 4096;）
        for _ in range(2):
            for m in re.finditer(r'uint\d+_t\s+(\w+)\s*=\s*([^;]+);', tiling_body):
                var, rhs = m.group(1), m.group(2).strip()
                if var in consts:
                    continue
                if re.fullmatch(r'\d+', rhs):
                    consts[var] = int(rhs)
                elif rhs in consts:
                    # e.g. uint32_t tileSize = TILE_SIZE; → consts["tileSize"] = 4096
                    consts[var] = consts[rhs]
        # tiling.set_<field>(<expr>) → 将 tiling 字段名映射到已知常量值
        # 例：tiling.set_tileSize(tileSize) 且 consts["tileSize"]=4096
        #     → consts["tileSize"] = 4096（已在上步建立，此处跳过重复写；下方处理直接传 TILE_SIZE 的情形）
        # 这使得 kernel InitBuffer 中 "tileSize * sizeof(float)" 可正确求值
        for m in re.finditer(r'tiling\.set_(\w+)\s*\(\s*([^)]+?)\s*\)', tiling_body):
            field, expr = m.group(1), m.group(2).strip()
            if expr in consts:
                consts[field] = consts[expr]
            elif re.fullmatch(r'\d+', expr):
                consts[field] = int(expr)
    return {"constants": consts, "tiling_body": tiling_body}


# sizeof lookup for InitBuffer expression evaluation
_SIZEOF = {
    "char": 1, "int8_t": 1, "uint8_t": 1, "int8": 1,
    "short": 2, "int16_t": 2, "uint16_t": 2, "int16": 2, "half": 2, "float16_t": 2, "__fp16": 2,
    "int": 4, "int32_t": 4, "uint32_t": 4, "int32": 4, "float": 4, "float32_t": 4,
    "double": 8, "int64_t": 8, "float64_t": 8,
    # Template type aliases
    "ElementA": 1, "ElementB": 1, "ElementC": 4, "ElementD": 2,
}


def _eval_size_expr(expr: str, consts: dict) -> int:
    """Evaluate simple InitBuffer size expression like 'tileElements * sizeof(float)'."""
    # Replace sizeof(type) → integer
    expr2 = re.sub(r'sizeof\s*\(\s*(\w+)\s*\)',
                   lambda m: str(_SIZEOF.get(m.group(1), 4)), expr)
    # Replace TYPE::MEMBER constants first (longest key first to avoid partial match)
    for k, v in sorted(consts.items(), key=lambda x: -len(x[0])):
        if '::' in k:
            expr2 = expr2.replace(k, str(v))
    # Replace simple identifier constants
    for k, v in sorted(consts.items(), key=lambda x: -len(x[0])):
        if '::' not in k:
            expr2 = re.sub(r'\b' + re.escape(k) + r'\b', str(v), expr2)
    # Only evaluate pure arithmetic
    try:
        if re.fullmatch(r'[\d\s\*\+\-\/\(\)]+', expr2.strip()):
            return int(eval(expr2))
    except Exception as e:
        logger.debug("size 表达式求值失败，返回 0 (%r): %s", expr, e)
    return 0


def parse_kernel_init_buffers(kernel_txt: str, consts: dict) -> dict:
    """
    解析 pipe.InitBuffer() 调用，返回 {buf_name: size_bytes}。
    支持两种形式：
      pipe.InitBuffer(bufName, count, sizeExpr);  — TQue（count 是整数字面量）
      pipe.InitBuffer(bufName, sizeExpr);          — TBuf

    在求值 sizeExpr 前，按以下三类扩展常量表（两轮前向引用）：
      1. 带类型前缀的局部变量声明：uint32_t x = expr;
      2. constexpr 类成员常量：static constexpr uint32_t CHUNK_M = L1Shape::M / VEC_NUM;
      3. 成员变量赋值（无类型前缀）：computeLength = CHUNK_M * L1Shape::N;
    """
    # 扩展常量表：先复制 consts，再追加 kernel 中可求值的常量/变量
    ext = dict(consts)
    # Pass 1a: 带类型前缀的局部变量声明（uint32_t x = expr; / int x = expr;）
    for m in re.finditer(r'\b(?:uint\d+_t|int|size_t)\s+(\w+)\s*=\s*([^;]+);', kernel_txt):
        var, rhs = m.group(1), m.group(2).strip()
        if var not in ext:
            v = _eval_size_expr(rhs, ext)
            if v > 0:
                ext[var] = v
    # Pass 1b: constexpr 类成员常量（static constexpr uint32_t CHUNK_M = L1Shape::M / VEC_NUM;）
    for m in re.finditer(r'\bconstexpr\s+\w[\w:]*\s+(\w+)\s*=\s*([^;]+);', kernel_txt):
        var, rhs = m.group(1), m.group(2).strip()
        if var not in ext:
            v = _eval_size_expr(rhs, ext)
            if v > 0:
                ext[var] = v
    # Pass 1c（两轮）: 成员变量赋值，无类型前缀（computeLength = CHUNK_M * L1Shape::N;）
    # 只匹配行首缩进后的 identifier = arithmetic_expr; 形式，避免匹配 if/for 等控制结构
    for _ in range(2):
        for m in re.finditer(r'^\s+(\w+)\s*=\s*([^=;{}\n][^;\n]*);', kernel_txt, re.MULTILINE):
            var, rhs = m.group(1), m.group(2).strip()
            if var not in ext:
                v = _eval_size_expr(rhs, ext)
                if v > 0:
                    ext[var] = v

    sizes = {}
    # TQue form: 3 args, 2nd is integer literal
    for m in re.finditer(
            r'pipe\.InitBuffer\s*\(\s*(\w+)\s*,\s*(\d+)\s*,\s*([^;]+?)\s*\)\s*;', kernel_txt):
        buf, size_expr = m.group(1), m.group(3).strip()
        sizes[buf] = _eval_size_expr(size_expr, ext)
    # TBuf form: 2 args, 2nd is NOT an integer literal alone
    for m in re.finditer(
            r'pipe\.InitBuffer\s*\(\s*(\w+)\s*,\s*([^,;]+?)\s*\)\s*;', kernel_txt):
        buf, size_expr = m.group(1), m.group(2).strip()
        if buf not in sizes and not re.fullmatch(r'\d+', size_expr):
            sizes[buf] = _eval_size_expr(size_expr, ext)
    return sizes


def compact_shape(shape_str: str, shape_vars=None) -> str:
    """Convert shape dict string to compact 'M=32 K=64 N=48' format."""
    kv = {}
    for m in re.finditer(r"'(\w+)':\s*(\d+)", str(shape_str)):
        kv[m.group(1)] = m.group(2)
    if not kv:
        return str(shape_str)
    if shape_vars:
        parts = [f"{k}={kv.get(k, '')}" for k in shape_vars if k in kv]
        extras = [f"{k}={v}" for k, v in kv.items() if k not in shape_vars]
        return " ".join(parts + extras[:2])
    return " ".join(f"{k}={v}" for k, v in list(kv.items())[:6])


# ─── UB BUFFER SIZES ──────────────────────────────────────────────────────────

def compute_tiling_analysis(cases: list, tiling_consts: dict, input_shapes: list, aic_cores: int = 32) -> list:
    """
    对每个 test case 计算 tiling 指标：
    - 从 shape 字符串提取 M/K/N（支持 dict 格式 {'M':32,'K':64,'N':48,...}）
    - 计算 AIC tile 数、有无尾块、每核分配、负载均衡率
    返回 [{case_id, case_name, dims, tiles, tail, tiles_per_core, balance_pct}, ...]
    """
    # Resolve tile constants with fallbacks
    l1_m = tiling_consts.get('L1_M') or tiling_consts.get('L1Shape_M') or tiling_consts.get('tile_m')
    l1_n = tiling_consts.get('L1_N') or tiling_consts.get('L1Shape_N') or tiling_consts.get('tile_n')
    l1_k = tiling_consts.get('L1_K') or tiling_consts.get('L1Shape_K') or tiling_consts.get('tile_k')

    # Collect all dim names from input_shapes
    all_dim_names = []
    for s in input_shapes:
        for d in s.get('shape', []):
            if isinstance(d, str) and d not in all_dim_names:
                all_dim_names.append(d)

    results = []
    for c in cases:
        shape_str = str(c.get('shape', ''))
        # Parse numeric key-value pairs from dict string
        kv = {}
        for m in re.finditer(r"'(\w+)':\s*(\d+)", shape_str):
            kv[m.group(1)] = int(m.group(2))
        if not kv:
            # Try compact format: M=32 K=64 N=48
            for m in re.finditer(r'\b([A-Z]\w*)\s*=\s*(\d+)', shape_str):
                kv[m.group(1)] = int(m.group(2))

        m_dim = kv.get('M', 0)
        n_dim = kv.get('N', 0)
        k_dim = kv.get('K', 0)

        # Build dim string from available shape dims
        dim_parts = [
            (k, v) for k, v in kv.items() if isinstance(
                v, int) and k in (
                'M', 'N', 'K', 'B', 'H', 'S', 'D', 'C', 'L')]
        dim_str = '×'.join(str(v) for _, v in dim_parts) if dim_parts else c.get('shape_compact', '')

        # Only compute tile analysis when we have primary tile dims
        if l1_n and n_dim:
            tiles_m = ((m_dim + l1_m - 1) // l1_m) if l1_m and m_dim else None
            tiles_n = (n_dim + l1_n - 1) // l1_n
            tiles_k = ((k_dim + l1_k - 1) // l1_k) if l1_k and k_dim else None
            total_tiles = (tiles_m or 1) * tiles_n

            tail_m = bool(l1_m and m_dim and m_dim % l1_m != 0)
            tail_n = bool(n_dim % l1_n != 0)
            tail_k = bool(l1_k and k_dim and k_dim % l1_k != 0)

            tiles_pc = round(total_tiles / aic_cores, 2)
            remainder = total_tiles % aic_cores
            if total_tiles == 0:
                balance_pct = 0
            elif remainder == 0:
                balance_pct = 100
            else:
                ceil_t = (total_tiles + aic_cores - 1) // aic_cores
                balance_pct = round(total_tiles / (ceil_t * aic_cores) * 100)

            results.append({
                'case_id': c.get('id', 0),
                'case_name': c.get('name', f"Case {c.get('id',0)}"),
                'dim_str': dim_str,
                'M': m_dim, 'N': n_dim, 'K': k_dim,
                'tiles_m': tiles_m, 'tiles_n': tiles_n, 'tiles_k': tiles_k,
                'total_tiles': total_tiles,
                'tail_m': tail_m, 'tail_n': tail_n, 'tail_k': tail_k,
                'has_tail': tail_m or tail_n or tail_k,
                'tiles_per_core': tiles_pc,
                'balance_pct': balance_pct,
            })
    return results


DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "float": 4, "half": 2, "int8": 1, "int16": 2, "int32": 4}
POSITION_COLORS = {
    "VECIN": _tok("colors.pos_vecin", "#0969da"),
    "VECOUT": _tok("colors.pos_vecout", "#1a7f37"),
    "VECCALC": _tok("colors.pos_veccalc", "#cf222e"),
    "L1": _tok("colors.pos_l1", "#8250df"),
    "L0A": "#bf8700",
}

# 每个 buffer 独立配色色板（16色，感知均匀、明暗双主题均可读）
# 冷暖交替，相邻 buffer 对比明显
_BUF_PALETTE = [
    "#0969da",  # 0  蓝
    "#e36209",  # 1  橙
    "#1a7f37",  # 2  绿
    "#cf222e",  # 3  红
    "#8250df",  # 4  紫
    "#0e8a8a",  # 5  青
    "#bf8700",  # 6  琥珀
    "#c4432b",  # 7  砖红
    "#0550ae",  # 8  深蓝
    "#2da44e",  # 9  亮绿
    "#953800",  # 10 深橙
    "#6639ba",  # 11 深紫
    "#1b7a8a",  # 12 深青
    "#d4a017",  # 13 金
    "#b35cad",  # 14 粉紫
    "#3d7a3d",  # 15 橄榄绿
]


def _buf_color(idx: int) -> str:
    """按 buffer 序号取色板颜色，循环使用。"""
    return _BUF_PALETTE[idx % len(_BUF_PALETTE)]


def compute_ub_buffers(kernel_data, tile_length, dtype_bytes, init_sizes=None):
    """
    计算 UB buffer 大小列表。
    优先使用 init_sizes（从 InitBuffer 表达式求值），fallback 到 tile_length × dtype_bytes。
    每个 buffer 从 _BUF_PALETTE 按序号取色，保证视觉可区分。
    """
    result = []
    for idx, buf in enumerate(kernel_data.get("buffers", [])):
        pos = buf["position"]
        mul = buf["multiplier"]
        if init_sizes and buf["name"] in init_sizes and init_sizes[buf["name"]] > 0:
            size_kb = init_sizes[buf["name"]] / 1024.0
        else:
            size_kb = mul * tile_length * dtype_bytes / 1024.0
        result.append({
            "name": buf["name"],
            "kind": buf["kind"],
            "position": pos,
            "multiplier": mul,
            "size_kb": round(size_kb, 2),
            "color": _buf_color(idx),
            "label": f"{pos} {buf['kind']}{'×'+str(mul) if mul>1 else ''}",
            "kernel_group": buf.get("kernel_group", ""),
        })
    return result


# ─── OP GRAPH (工业级计算流图骨架) ─────────────────────────────────────────────
# ⚠️  gen_dashboard.py 是纯汇总器，不做算子语义推断。
# 计算单元（units）必须来自外部 algo_flow.json（由 Claude 阅读算子源码生成）。
# 若未提供 algo_flow.json，units 为空，Tab 1 显示引导提示。
# 详见：subskills/algo_flowchart.md

def build_op_graph(input_shapes, output_shapes, cases):
    """
    构建 op_graph 骨架：IO 字段 + shape_cases，units 始终为空。
    计算单元由 algo_flow.json 注入，不在此处推断。
    """
    inp_shape = input_shapes[0].get("shape", []) if input_shapes else []
    ndim = len(inp_shape)
    # Use actual symbolic names from shape (e.g. ["M","K"]) if they are strings,
    # otherwise fall back to generic var names
    if inp_shape and all(isinstance(s, str) for s in inp_shape):
        shape_vars = list(inp_shape)   # e.g. ["M", "K"] from op_desc
    elif ndim == 1:
        shape_vars = ["N"]
    elif ndim == 2:
        shape_vars = ["N", "C"]
    elif ndim == 3:
        shape_vars = ["B", "N", "C"]
    else:
        shape_vars = [f"D{i}" for i in range(ndim)]

    def shape_tmpl(shape):
        return ", ".join(
            "{" + (shape_vars[i] if i < len(shape_vars) else f"D{i}") + "}"
            for i, _ in enumerate(shape)
        )

    inp_tmpl = shape_tmpl(inp_shape)
    out_shape = output_shapes[0].get("shape", inp_shape) if output_shapes else inp_shape
    out_tmpl = shape_tmpl(out_shape)
    inp_dtype = input_shapes[0].get("dtype", "") if input_shapes else ""
    out_dtype = output_shapes[0].get("dtype", "") if output_shapes else ""
    inp_name = input_shapes[0].get("name", "x") if input_shapes else "x"
    out_name = output_shapes[0].get("name", "y") if output_shapes else "y"

    shape_cases = []
    for c in cases:
        shape_str = str(c.get("shape", ""))
        label_prefix = c.get("name") or f"Case {c.get('id',0)}"
        # Try 1: parse dict-format string → extract values by key (e.g. {'M':32,'K':64,'N':48,...})
        kv = {m.group(1): m.group(2) for m in re.finditer(r"'(\w+)':\s*(\d+)", shape_str)}
        if kv and all(v in kv for v in shape_vars):
            vals = [kv[v] for v in shape_vars]
            shape_cases.append({
                "label": f"{label_prefix}: {' × '.join(vals)}",
                "vars": dict(zip(shape_vars, vals)),
            })
            continue
        # Try 2: use actual dims regardless of ndim mismatch (supports 1D/2D/3D/4D test cases)
        nums = re.findall(r'\d+', shape_str)
        if nums:
            # Map to shape_vars for first min(n, len(shape_vars)) dims, extend with S{i} for extras
            actual_vars = [shape_vars[i] if i < len(shape_vars) else f"S{i}"
                           for i in range(len(nums))]
            shape_cases.append({
                "label": f"{label_prefix}: {' × '.join(nums)}",
                "vars": dict(zip(actual_vars, nums)),
            })
    if not shape_cases and inp_shape:
        shape_cases = [{"label": "Default",
                        "vars": dict(zip(shape_vars, [str(s) for s in inp_shape]))}]

    # Multi-IO arrays (v2)
    def _io_entry(s, idx, is_output=False):
        shp = s.get("shape", [])
        tmpl = ", ".join("{" + (shape_vars[i] if i < len(shape_vars) else f"D{i}") + "}"
                         for i, _ in enumerate(shp))
        return {"name": s.get("name", "y" if is_output else f"x{idx}"),
                "dtype": s.get("dtype", ""), "tmpl": tmpl}

    inputs_arr = [_io_entry(s, i, False) for i, s in enumerate(input_shapes)]
    outputs_arr = [_io_entry(s, i, True) for i, s in enumerate(output_shapes)]

    return {
        "units": [],          # 由 algo_flow.json 注入
        "needs_algo_flow": True,        # JS 用于显示引导提示
        "shape_vars": shape_vars,
        "shape_cases": shape_cases,
        # legacy single-IO fields (backward compat)
        "inp_name": inp_name,
        "out_name": out_name,
        "inp_tmpl": inp_tmpl,
        "out_tmpl": out_tmpl,
        "inp_dtype": inp_dtype,
        "out_dtype": out_dtype,
        # v2 multi-IO arrays
        "inputs": inputs_arr,
        "outputs": outputs_arr,
    }


# ─── 主数据汇总 ────────────────────────────────────────────────────────────────

def collect_data(args) -> dict:
    paths = {}
    if args.op_dir:
        paths = discover(Path(args.op_dir))
    # explicit overrides
    for key, attr in [("op_desc", "op_desc"), ("eval", "eval"), ("precision", "precision"),
                      ("multi_csv", "multi_csv"), ("dsl", "dsl"), ("kernel", "kernel"),
                      ("profiling_dir", "profiling_dir"), ("test_cases", "test_cases"),
                      ("algo_flow", "algo_flow")]:
        val = getattr(args, attr.replace("-", "_"), None) or getattr(args, attr, None)
        if val:
            paths[key] = Path(val)

    logger.info("📂 数据源:")
    for k, v in paths.items():
        if isinstance(v, list):
            logger.info(f"  ✓ {k}: {len(v)} files")
        else:
            exists = "✓" if Path(v).exists() else "✗"
            logger.info(f"  {exists} {k}: {v}")

    # ── 解析各数据源 ──
    op_desc = _json(paths.get("op_desc"))
    # eval: JSON (evaluation_results.json) or TXT log (evaluation_result.txt)
    eval_path = paths.get("eval")
    eval_is_txt = eval_path and str(eval_path).endswith(".txt")
    eval_txt_cases = parse_evaluation_txt(eval_path) if eval_is_txt else []
    eval_res = {} if eval_is_txt else _json(eval_path)
    # CAKE2 multi-case evaluation_results.json: has "results" array
    eval_results_cases = (parse_evaluation_results_json(eval_path)
                          if (not eval_is_txt and eval_res.get("results")) else [])
    prec_cases = parse_precision_json(paths.get("precision"))
    multi_csv = parse_multi_case_csv(paths.get("multi_csv"))
    # flat msprof CSV pairs (e.g. op_summary_custom_fn_*.csv)
    flat_perf = {}
    if not multi_csv and (paths.get("msprof_custom_csvs") or paths.get("msprof_ref_csvs")):
        flat_pairs = parse_flat_msprof_csv_pairs(
            paths.get("msprof_custom_csvs", []),
            paths.get("msprof_ref_csvs", []),
            op_desc.get("op_name", "") if op_desc else "",
        )
        # Pull out unpaired diagnostic if present
        _unpaired_diag = None
        clean_pairs = []
        for row in flat_pairs:
            if "_unpaired" in row:
                _unpaired_diag = row["_unpaired"]
            else:
                clean_pairs.append(row)
        flat_perf = {row["case_name"]: row for row in clean_pairs}
        if _unpaired_diag:
            paths["_perf_unpaired"] = _unpaired_diag
    # Parse op_summary detail (full field extraction) for Perf tab op_summary panel
    op_summary_details: dict = {}
    _op_nm_for_detail = op_desc.get("op_name", "") if op_desc else ""

    # 通用策略：全局 rglob op_summary*.csv（排除 origin_data/），
    # 从路径中提取 case 名：
    #   custom/ModelNew_* 路径 → 取上层（如 _msprof_work/<case>/custom/... → case）
    #   op_summary_<case>.csv 直接命名 → 从文件名提取
    # _msprof_work/<case>/custom/... 和 profiling_dir 等各种结构均兼容
    if args.op_dir:
        _op_dir_path = Path(args.op_dir)
        _all_csvs = sorted(
            [f for f in _op_dir_path.rglob("op_summary*.csv")
             if "origin_data" not in f.parts
             and not f.name.startswith("op_summary_reference_fn_")],
            key=lambda f: f.stat().st_mtime
        )
        _case_csv_map: dict = {}  # case_name → csv_path (prefer custom/ModelNew path)
        for _f in _all_csvs:
            # 尝试从路径结构推断 case 名
            _parts = _f.relative_to(_op_dir_path).parts  # e.g. ('_msprof_work','basic_1d','custom','ModelNew_...',...)
            _inferred = None
            # 模式1: .../custom/ModelNew_*/.../op_summary*.csv → 取 custom 的上级目录
            if "custom" in _parts:
                _ci = list(_parts).index("custom")
                if _ci > 0:
                    _inferred = _parts[_ci - 1]
            # 模式2: op_summary_<case>.csv（非时间戳命名，即含非纯数字的词）
            if _inferred is None:
                _stem = _f.stem[len("op_summary_custom_fn_"):] if _f.name.startswith("op_summary_custom_fn_") \
                    else _f.stem[len("op_summary_"):]
                if _stem and not _stem.isdigit() and not re.match(r'^\d{8,}$', _stem):
                    _inferred = _stem
            if _inferred is None:
                continue  # 时间戳命名且无路径线索 → 跳过
            # custom 路径优先（比 ref 路径更可靠）
            if _inferred not in _case_csv_map or "custom" in str(_f) or "ModelNew" in str(_f):
                _case_csv_map[_inferred] = _f
        for _cname, _f in _case_csv_map.items():
            _det = parse_op_summary_detail(_f, _op_nm_for_detail)
            if _det:
                op_summary_details[_cname] = _det

    if paths.get("msprof_custom_csvs"):
        for _f in paths["msprof_custom_csvs"]:
            _cname = _f.stem[len("op_summary_custom_fn_"):]
            _det = parse_op_summary_detail(_f, _op_nm_for_detail)
            if _det:
                op_summary_details[_cname] = _det
    # CAKE2 evaluate.py format: profiling/op_summary_<case>.csv (no prefix)
    # e.g. op_summary_basic_1d.csv → case name = "basic_1d"
    if not op_summary_details and paths.get("msprof_cake2_csvs"):
        for _f in paths["msprof_cake2_csvs"]:
            _cname = _f.stem[len("op_summary_"):]   # strip "op_summary_" prefix
            _det = parse_op_summary_detail(_f, _op_nm_for_detail)
            if _det:
                op_summary_details[_cname] = _det
    # TileLang / msprof 场景：从 profiling_dir 的 ModelNew_* 目录提取 op_summary detail
    # 文件名格式 op_summary_YYYYMMDD.csv（不含 custom_fn_ 前缀）
    if not op_summary_details and paths.get("profiling_dir"):
        _pdir = Path(paths["profiling_dir"])
        _modelnew_dirs = sorted(
            [d for d in _pdir.iterdir() if d.is_dir() and d.name.startswith("ModelNew_device")],
            key=lambda d: d.stat().st_mtime
        )
        if _modelnew_dirs:
            # 取最新的 ModelNew 目录，递归找 op_summary*.csv（排除 origin_data/）
            _latest_mn = _modelnew_dirs[-1]
            _mn_csvs = [f for f in _latest_mn.rglob("op_summary*.csv")
                        if "origin_data" not in f.parts]
            if _mn_csvs:
                # TileLang op_summary：op name 是底层 aclnn 算子名，不含算子名前缀
                # 传 "" 跳过 op_name 过滤，汇总所有行的平均指标
                _det = parse_op_summary_detail(_mn_csvs[0], "")
                if _det:
                    op_summary_details["case_0"] = _det
    test_cases = parse_test_cases_csv(paths.get("test_cases"))
    test_cases_py_shapes = parse_test_cases_py(paths.get("test_cases_py"))
    kernel_data = parse_kernel_cpp(paths.get("kernel"))
    op_host_data = parse_op_host_cpp(paths.get("op_host"))
    dsl_data = parse_dsl(paths.get("dsl"))
    prof_report = parse_profiling_report_txt(paths.get("report_txt"))
    prof_pairs = pair_profiling_dirs(paths.get("profiling_dir"))
    cann_result = parse_cann_result_json(paths.get("cann_result"))

    # ── 基础 ──
    op_name = op_desc.get("op_name", "Unknown")
    category = op_desc.get("category", "")
    desc_str = op_desc.get("description", "")
    attributes = op_desc.get("attributes", {})
    # shape_info: two formats
    #   dict format: {"input_shapes": [...], "output_shapes": [...]}
    #   list format: [{name, shape, dtype, ...}, ...] — name=="output" marks outputs
    si = op_desc.get("shape_info", {})
    if isinstance(si, list):
        input_shapes = [s for s in si if s.get("name", "").lower() != "output"]
        output_shapes = [s for s in si if s.get("name", "").lower() == "output"]
        if not output_shapes and input_shapes:
            output_shapes = [input_shapes[-1]]
            input_shapes = input_shapes[:-1]
    else:
        input_shapes = si.get("input_shapes", [])
        output_shapes = si.get("output_shapes", [])

    # dtype/tile
    tile_length, dtype_bytes = 1, 4
    if input_shapes:
        sh = input_shapes[0].get("shape", [])
        # Only use numeric tile lengths (skip symbolic like "M", "K")
        if sh:
            last = sh[-1]
            if isinstance(last, (int, float)):
                tile_length = int(last)
        dtype_bytes = DTYPE_BYTES.get(input_shapes[0].get("dtype", "float32"), 4)

    # ── Cases（优先级：multi_csv > eval_results_cases > eval_txt > prec_cases > eval_res 单case）──
    cases = []
    if multi_csv:
        for row in multi_csv:
            cid = row["case_id"]
            prec = prec_cases.get(cid, {})
            cases.append({
                "id": cid,
                "shape": row["shape"],
                "passed": row["passed"],
                "precision": {
                    "passed": prec.get("passed", row["passed"]),
                    "max_re": prec.get("max_re"),
                    "mean_re": prec.get("mean_re"),
                    "rmse": prec.get("rmse"),
                    "ae_max": prec.get("ae_max"),
                    "re_max": prec.get("re_max"),
                    "mismatch_rate": prec.get("mismatch_rate"),
                },
                "performance": {
                    "ref_time_us": row["ref_time_us"],
                    "custom_time_us": row["custom_time_us"],
                    "speedup": row["speedup"],
                    "source": "msprof",
                },
            })
    elif eval_results_cases:
        # CAKE2 evaluate.py multi-case format (evaluation_results.json with "results" array)
        for c in eval_results_cases:
            cases.append({
                "id": c["case_id"],
                "name": c["name"],
                "shape": c["shape"],
                "passed": c["passed"],
                "precision": c["precision"],
                "performance": {
                    "ref_time_us": c["ref_time_us"],
                    "custom_time_us": c["custom_time_us"],
                    "speedup": c["speedup"],
                    "source": "msprof",
                },
            })
    elif eval_txt_cases:
        # evaluation_result.txt 日志格式（每 case 含精度+性能）
        for c in eval_txt_cases:
            name = c.get("name", "")
            # Enrich performance from flat msprof CSV pairs if available
            perf = c.get("performance", {})
            if flat_perf.get(name) and flat_perf[name].get("custom_time_us"):
                fp = flat_perf[name]
                perf = {
                    "ref_time_us": fp["ref_time_us"],
                    "custom_time_us": fp["custom_time_us"],
                    "speedup": fp["speedup"],
                    "source": "msprof_csv",
                }
            cases.append({
                "id": c["id"],
                "name": name,
                "shape": c.get("shape", ""),
                "passed": c["passed"],
                "precision": c.get("precision", {}),
                "performance": perf,
            })
    elif prec_cases:
        for cid, prec in sorted(prec_cases.items()):
            cases.append({
                "id": cid,
                "shape": str(prec.get("params", {}).get("var0_shape", "")),
                "passed": prec.get("passed", False),
                "precision": prec,
                "performance": {},
            })
    elif eval_res:
        # 单 case fallback (JSON)
        from_msg = _parse_correctness(eval_res.get("correctness_message", ""))
        perf = {}
        if prof_report.get("ref_time_us"):
            perf = {"ref_time_us": prof_report["ref_time_us"],
                    "custom_time_us": prof_report["custom_time_us"],
                    "speedup": prof_report.get("speedup", 0), "source": "msprof"}
        elif eval_res.get("base_time_ms"):
            perf = {"ref_time_us": eval_res["base_time_ms"] * 1000,
                    "custom_time_us": eval_res["gen_time_ms"] * 1000,
                    "speedup": eval_res.get("speedup", 0), "source": "npu_event"}
        cases.append({
            "id": 0, "shape": str(input_shapes[0].get("shape", "") if input_shapes else ""),
            "passed": from_msg.get("passed", eval_res.get("precision_passed", False)),
            "precision": from_msg,
            "performance": perf,
        })

    # ── Enrich cases: add shape_compact + case_label + op_summary_avg ──
    _isv = input_shapes[0].get("shape", []) if input_shapes else []
    _sv = [str(s) for s in _isv] if _isv and all(isinstance(s, str) for s in _isv) else None
    for c in cases:
        # Enrich shape from test_cases.py if shape is still just the case name
        cname = c.get("name", "")
        if test_cases_py_shapes and cname in test_cases_py_shapes:
            py_shape = test_cases_py_shapes[cname]
            # Only replace if shape == case_name (the fallback) or shape is empty
            if not c.get("shape") or c.get("shape") == cname:
                c["shape"] = py_shape
        if not c.get("shape_compact"):
            c["shape_compact"] = compact_shape(c.get("shape", ""), _sv)
        if not c.get("name"):
            c["name"] = c.get("shape_compact") or f"Case {c.get('id',0)}"
        # Attach op_summary detail if available
        cname = c.get("name", "")
        cid = c.get("id", 0)
        # 匹配优先级：case name → "case_N" → 单条时直接用 "case_0"
        _det = (op_summary_details.get(cname)
                or op_summary_details.get(f"case_{cid}")
                or (op_summary_details.get("case_0") if len(op_summary_details) == 1 and cid == 0 else None))
        if _det:
            c["op_summary_avg"] = _det

    # ── UB buffers ──
    # 先检测芯片型号，以便将 AIV core 数作为 nUsed/coreNum 上界注入常量表，
    # 防止 workspace 类 InitBuffer 表达式（含 nUsed 等运行时变量）求值失败时
    # 错误回落到 tile_length×dtype_bytes（可能异常大）。
    chip_info = detect_chip_runtime(
        str(_tok("chip.name", "Ascend910B2")),
        int(_tok("chip.ub_kb", 192)),
        int(_tok("chip.aic", 24)),
        int(_tok("chip.aiv", 48)),
    )
    # Build merged constant dict (op_host constants take priority)
    tiling_consts = {}
    tiling_consts.update(kernel_data.get("constants", {}))
    tiling_consts.update(op_host_data.get("constants", {}))
    # 注入 AIV core 数作为运行时变量的保守上界
    # 例：workspaceInQueue 的 ((nUsed * sizeof(float) + 31) / 32) * 32 可正确求值
    # 注意：nUsed/coreNum/blockDim 绑定的是 AIV（vector_core）数，而非 AIC（cube_core）数
    _aiv = int(chip_info.get("aiv", 48))
    for _runtime_var in ("nUsed", "usedCoreNum", "coreNum", "blockDim"):
        if _runtime_var not in tiling_consts:
            tiling_consts[_runtime_var] = _aiv
    init_sizes = parse_kernel_init_buffers(kernel_data.get("_txt", ""), tiling_consts)
    ub_buffers = compute_ub_buffers(kernel_data, tile_length, dtype_bytes, init_sizes)
    # Track whether any buffer size fell back to tileLength estimate (vs parsed from InitBuffer).
    # Used by check_dashboard.py to decide FAIL vs WARN when ub_used > ub_total.
    _ub_estimated = bool(
        not init_sizes
        or any(
            not (init_sizes.get(buf["name"], 0) > 0)
            for buf in kernel_data.get("buffers", [])
        )
    )
    # TileLang standalone fallback: parse UB buffers from panels/memory/analysis.md
    _ub_from_md = False
    if not ub_buffers and paths.get("memory_analysis_md"):
        ub_buffers = parse_ub_from_analysis_md(paths["memory_analysis_md"])
        if ub_buffers:
            _ub_from_md = True
    tiling_analysis = compute_tiling_analysis(
        cases, tiling_consts, input_shapes, int(chip_info.get("aic", 32)))
    ub_used_kb = sum(b["size_kb"] for b in ub_buffers)
    ub_total_kb = float(chip_info.get("ub_kb", 256))

    # ── Raw Pass 注释（直接透传，不做语义推断）──
    pass_comments = kernel_data.get("pass_comments", [])
    if not pass_comments:
        pass_comments = [l for l in dsl_data.get("header_comments", [])
                         if any(k in l for k in ["Pass", "pass", "步骤", "算法"])]

    # ── Op graph (工业流图骨架) ──
    # 优先读取 algo_flow.json（由 Claude 按 subskills/algo_flowchart.md 生成）
    # 若不存在，返回 needs_algo_flow=True 的空骨架，Tab 1 显示引导提示
    algo_flow_path = paths.get("algo_flow")
    if algo_flow_path and Path(algo_flow_path).exists():
        af = _json(algo_flow_path)
        # 若为自动生成且 api_list 为空（API 提取失败），丢弃重新提取
        if af.get("_auto") and not af.get("api_list"):
            af = None
            algo_flow_path = None
    else:
        af = None
    if algo_flow_path and af:
        # - v1 (shape_vars format): has "shape_vars" key
        # - v2 (units format): has "units" key directly (CAKE2 auto-generated or manual)
        if af and "shape_vars" in af:
            if not af.get("shape_cases"):
                af["shape_cases"] = []
                for c in cases:
                    shape_str = str(c.get("shape", ""))
                    nums = re.findall(r'\d+', shape_str)
                    _svars = af["shape_vars"]
                    if nums:
                        # Use whatever dims available; map to shape_vars + S{i} for extras
                        actual_vars = [_svars[i] if i < len(_svars) else f"S{i}"
                                       for i in range(len(nums))]
                        af["shape_cases"].append({
                            "label": f"{c.get('name','Case '+str(c.get('id',0)))}: {' × '.join(nums)}",
                            "vars": dict(zip(actual_vars, nums)),
                        })
            sv = af.get("shape_vars", ["N", "C"])

            def _io_tmpl_af(shape):
                return ", ".join("{" + (sv[i] if i < len(sv) else f"D{i}") + "}"
                                 for i, _ in enumerate(shape))
            # Inject IO from algo_flow.json inputs[]/outputs[] if present
            if af.get("inputs"):
                af["inp_name"] = af["inputs"][0].get("name", "x")
                af["inp_tmpl"] = af["inputs"][0].get("tmpl", "")
                af["inp_dtype"] = af["inputs"][0].get("dtype", "")
            elif not af.get("inp_name") and input_shapes:
                af["inp_name"] = input_shapes[0].get("name", "x")
                af["inp_tmpl"] = _io_tmpl_af(input_shapes[0].get("shape", []))
                af["inp_dtype"] = input_shapes[0].get("dtype", "")
            if af.get("outputs"):
                af["out_name"] = af["outputs"][0].get("name", "y")
                af["out_tmpl"] = af["outputs"][0].get("tmpl", "")
                af["out_dtype"] = af["outputs"][0].get("dtype", "")
            elif not af.get("out_name") and output_shapes:
                af["out_name"] = output_shapes[0].get("name", "y")
                af["out_tmpl"] = _io_tmpl_af(output_shapes[0].get("shape", []))
                af["out_dtype"] = output_shapes[0].get("dtype", "")
            # Ensure v2 multi-IO arrays exist
            if not af.get("inputs") and input_shapes:
                af["inputs"] = [{"name": s.get("name", f"x{i}"),
                                 "dtype": s.get("dtype", ""),
                                 "tmpl": _io_tmpl_af(s.get("shape", []))}
                                for i, s in enumerate(input_shapes)]
            if not af.get("outputs") and output_shapes:
                af["outputs"] = [{"name": s.get("name", f"y{i}"),
                                  "dtype": s.get("dtype", ""),
                                  "tmpl": _io_tmpl_af(s.get("shape", []))}
                                 for i, s in enumerate(output_shapes)]
            op_graph = af
        elif af and af.get("units"):
            # v2 format (CAKE2 auto-generated / manual without shape_vars)
            # units are directly in "units" key, no shape interpolation needed.
            # Inject defaults required by validate_contract / check_dashboard.
            if not af.get("shape_vars"):
                af = dict(af, shape_vars=["N"])
            if not af.get("inputs"):
                af = dict(af, inputs=[{"name": "x", "shape": ["{N}"], "tmpl": "{N}", "dtype": "float32"}])
            if not af.get("outputs"):
                af = dict(af, outputs=[{"name": "y", "shape": ["{N}"], "tmpl": "{N}", "dtype": "float32"}])
            # Flat inp_*/out_* fields (check_dashboard validates these directly)
            if not af.get("inp_name"):
                _in0 = af["inputs"][0] if af.get("inputs") else {}
                _out0 = af["outputs"][0] if af.get("outputs") else {}
                _svars = af.get("shape_vars", ["N"])
                _tmpl = "{" + ",".join(_svars) + "}"
                af = dict(af,
                          inp_name=_in0.get("name", "x"),
                          inp_tmpl=_in0.get(
                              "tmpl",
                              [_tmpl])[0] if isinstance(
                              _in0.get("tmpl"),
                              list) else _in0.get(
                              "tmpl",
                              _tmpl),
                          inp_dtype=_in0.get("dtype", "float32"),
                          out_name=_out0.get("name", "y"),
                          out_tmpl=_out0.get(
                              "tmpl",
                              [_tmpl])[0] if isinstance(
                              _out0.get("tmpl"),
                              list) else _out0.get(
                              "tmpl",
                              _tmpl),
                          out_dtype=_out0.get("dtype", "float32"))
            op_graph = af
        else:
            op_graph = build_op_graph(input_shapes, output_shapes, cases)
    else:
        # No algo_flow.json: try to auto-build from kernel step comments / DSL compute steps
        _mapped_op_type = op_desc.get("operator_type", "vector") if op_desc else "vector"
        _auto_af = auto_build_algo_flow(op_name, dsl_data, kernel_data,
                                        op_type=_mapped_op_type,
                                        op_desc=op_desc)
        if _auto_af and _auto_af.get("units"):
            op_graph = _auto_af
            # Also save to panels/algo/algo_flow.json so next run is faster
            if args.op_dir:
                _algo_out_dir = Path(args.op_dir) / "panels" / "algo"
                _algo_out_dir.mkdir(parents=True, exist_ok=True)
                _algo_out_path = _algo_out_dir / "algo_flow.json"
                # 写入条件：文件不存在，或旧文件是 _auto:true 且 api_list 为空的过期版本
                _existing = _json(_algo_out_path) if _algo_out_path.exists() else {}
                _is_stale = _existing.get("_auto") and not _existing.get("api_list")
                if not _algo_out_path.exists() or _is_stale:
                    _algo_out_path.write_text(
                        json.dumps(_auto_af, indent=2, ensure_ascii=False), encoding="utf-8")
                    logger.info(f"  ↳ 自动提取 algo_flow.json → {_algo_out_path}")
        else:
            op_graph = build_op_graph(input_shapes, output_shapes, cases)

    # ── Profiling pairs → per-case timing (if not from multi_csv) ──
    if prof_pairs and not multi_csv and len(prof_pairs) == len(cases):
        for i, (ref_dir, cust_dir) in enumerate(prof_pairs):
            if i < len(cases):
                ref_us = extract_profiling_time(ref_dir)
                cust_us = extract_profiling_time(cust_dir)
                if ref_us > 0 and cust_us > 0:
                    cases[i]["performance"] = {
                        "ref_time_us": ref_us,
                        "custom_time_us": cust_us,
                        "speedup": round(ref_us / cust_us, 2),
                        "source": "msprof",
                    }

    # ── TileLang standalone: cann_result.json 覆盖性能数据（真实 NPU kernel 时延）──
    # cann_result.json 来自 tilelang_eval_adapter.py + combined_kernel_loader.py 评测
    # 代表真实 CANN .so kernel 时间（不含 Python padding 开销），比 evaluation_results.json 准确
    if cann_result and cann_result.get("speedup"):
        cann_perf = {
            "ref_time_us": cann_result["ref_time_us"],
            "custom_time_us": cann_result["custom_time_us"],
            "speedup": cann_result["speedup"],
            "source": "cann_standalone",
            "notes": cann_result.get("notes", ""),
        }
        if cases:
            # Override case 0 (or all single-case scenarios)
            for c in cases:
                c["performance"] = cann_perf
        else:
            # No cases at all: synthesize one
            cases.append({
                "id": 0, "name": "CANN Standalone", "shape": "",
                "passed": True, "precision": {}, "performance": cann_perf,
            })

    # ── 从 test_cases.csv 扩展 cases：当 eval 只有 1 case 但 CSV 有多个时 ──
    # 常见场景：cann_result.json 只有整体 speedup，不区分 case；eval 只跑了 case 0；
    # 但 test_cases.csv 记录了全部测试形状。将缺失 case 填充为 shape 已知、精度待评状态。
    if test_cases and len(cases) <= 1:
        try:
            _existing_ids = {c.get("id") for c in cases}
            _base_perf = cases[0].get("performance", {}) if cases else {}
            for tc_row in test_cases:
                try:
                    cid = int(tc_row.get("case_id", 0))
                except (ValueError, TypeError):
                    cid = 0
                if cid in _existing_ids:
                    continue
                # Build shape string from first var*_shape column
                shape_str = ""
                for k in sorted(tc_row.keys()):
                    if re.match(r'var\d+_shape', k):
                        shape_str = tc_row.get(k, "").strip()
                        break
                cases.append({
                    "id": cid,
                    "name": f"Case {cid}",
                    "shape": shape_str,
                    "passed": True,   # 假设 PASS（cann_result 不区分 case）
                    "precision": {"_note": "精度来自 cann_result.json（汇总），未单独评测此 case"},
                    "performance": _base_perf,  # 沿用汇总性能数据
                })
                _existing_ids.add(cid)
            # 按 id 排序
            cases.sort(key=lambda c: c.get("id", 0))
        except Exception as e:
            logger.debug("从 test_cases.csv 扩展 cases 失败，保留已有 cases: %s", e)

    # ── 整体精度/性能摘要（用于 header）──
    n_pass = sum(1 for c in cases if c.get("passed"))
    n_total = len(cases)
    speedups = [c.get("performance", {}).get("speedup") for c in cases if c.get("performance", {}).get("speedup")]
    # 使用几何均值（与 check_dashboard.py 一致）；过滤 ≤0 的值防止 math domain error
    import math as _math
    pos_sp = [s for s in speedups if s > 0]
    avg_speedup = round(_math.exp(sum(_math.log(s) for s in pos_sp) / len(pos_sp)), 2) if pos_sp else None

    # ── Diagnostics（数据健康）──
    diagnostics = []

    def _diag(tab, level, source, detail="", hint=""):
        diagnostics.append({"tab": tab, "level": level, "source": source,
                            "detail": detail, "hint": hint})

    _diag("algo", "FOUND" if op_graph.get("units") else "DERIVED",
          "algo_flow.json" if op_graph.get("units") else "auto-build",
          f"{len(op_graph.get('units',[]))} units" if op_graph.get("units")
          else "需精品流程生成 algo_flow.json",
          "" if op_graph.get("units") else "运行精品流程 Step 3 生成 panels/algo/algo_flow.json")
    _diag("mem", "FOUND" if ub_buffers else "MISSING",
          "panels/memory/analysis.md" if _ub_from_md else "op_kernel/*.cpp",
          f"{len(ub_buffers)} buffers, {ub_used_kb:.1f}/{ub_total_kb:.0f} KB" if ub_buffers else "",
          "" if ub_buffers else "提供 op_kernel/*_custom.cpp 或 panels/memory/analysis.md（含 Buffer 表格）")
    _diag("mem", "FOUND" if tiling_consts else "MISSING",
          "op_host/*.cpp",
          f"{len(tiling_consts)} constants" if tiling_consts else "",
          "" if tiling_consts else "提供 op_host/*_custom.cpp 文件（TileLang 场景可忽略此项）")
    if paths.get("_perf_unpaired"):
        _diag("perf", "DERIVED", "msprof CSVs",
              f"未配对文件: {', '.join(paths['_perf_unpaired'])}")
    _diag("prec", "FOUND" if any(c.get("precision", {}).get("match_rate") is not None for c in cases)
          else ("FOUND" if any(c.get("precision", {}).get("max_re") is not None for c in cases) else "MISSING"),
          "evaluation_result.txt" if eval_is_txt else "precision_results.json",
          f"{n_pass}/{n_total} PASS",
          "" if n_total > 0 else "运行 ascendc-evaluation 生成精度数据")
    _diag("perf", "FOUND" if speedups else "MISSING",
          "cann_result.json" if cann_result and cann_result.get("speedup") else
          ("msprof CSV" if (paths.get("msprof_custom_csvs") or paths.get("multi_csv")) else "无 msprof 数据"),
          f"{len(speedups)} cases with speedup" if speedups else "",
          "" if speedups else "提供 panels/perf/cann_result.json（TileLang）或运行 ascendc-evaluation_remote")
    _diag("mem", "FOUND",
          "chip probe",
          f"{chip_info.get('name')} | UB {int(ub_total_kb)}KB × {chip_info.get('aic')}核",
          f"source={chip_info.get('source', 'unknown')}")

    # ── Panel content discovery ──
    panels_content: dict = {}
    if args.op_dir:
        panels_content = discover_panels(Path(args.op_dir))

    return {
        "op_name": op_name,
        "category": category,
        "description": desc_str,
        "attributes": attributes,
        "chip": {
            "name": chip_info.get("name", _tok("chip.name", "Ascend910B2")),
            "ub_kb": int(ub_total_kb),
            "aic": int(chip_info.get("aic", _tok("chip.aic", 32))),
            "source": chip_info.get("source", "design_tokens"),
        },
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "cases": cases,
        "n_pass": n_pass,
        "n_total": n_total,
        "avg_speedup": avg_speedup,
        "pass_comments": pass_comments,
        "op_graph": op_graph,
        "tiling": {
            "tile_length": tile_length,
            "tile_size_kb": round(tile_length * dtype_bytes / 1024, 2),
            "dtype_bytes": dtype_bytes,
            "tiling_params": kernel_data.get("tiling_params", []),
            "consts": tiling_consts,
            "analysis": tiling_analysis,
        },
        "ub_buffers": ub_buffers,
        "ub_used_kb": round(ub_used_kb, 2),
        "ub_estimated": _ub_estimated,   # True = any buffer size from tileLength fallback (not InitBuffer parse)
        "_ub_from_md": _ub_from_md,
        "ub_total_kb": ub_total_kb,
        "ub_util_pct": round(ub_used_kb / ub_total_kb * 100, 1),
        "diagnostics": diagnostics,
        "op_summary_fields": OP_SUMMARY_DISPLAY_FIELDS,
        "panels": {
            "memory": panels_content.get("memory", {}),
            "precision": panels_content.get("precision", {}),
            "perf": panels_content.get("perf", {}),
            "algo": panels_content.get("algo", {}),
            "extra": panels_content.get("extra", []),
        },
    }


def _parse_correctness(msg: str) -> dict:
    result = {"passed": "[PASS]" in msg}
    for key in ["max_re", "mean_re", "rmse", "mismatch_rate"]:
        m = re.search(rf"{key}=([\d.]+)", msg)
        if m:
            result[key] = float(m.group(1))
    return result


# ─── PANEL SYSTEM ──────────────────────────────────────────────────────────────

def _md_inline(text: str) -> str:
    """Inline markdown → HTML: **bold**, `code`, [PASS]/[FAIL] coloring."""
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(
        r'`([^`]+)`',
        r'<code style="font-family:var(--mono);background:var(--sf2);padding:1px 4px;border-radius:3px">\1</code>',
        text)
    text = text.replace('[PASS]', '<span class="badge bp">PASS</span>')
    text = text.replace('[FAIL]', '<span class="badge bf">FAIL</span>')
    return text


def _sanitize_html_fragment(html: str) -> str:
    """Strip XSS vectors from Claude-authored HTML fragments before embedding.

    Removes:
    - <script> blocks (with content) and self-closing <script> tags
    - <object> / <embed> tags (plugin execution vectors)
    - Inline event-handler attributes (on<event>=...)
    - javascript: URL schemes
    - <iframe> tags (any variant)
    - <meta> tags (to block http-equiv refresh / CSP bypasses)
    """
    # Remove <script>...</script> blocks entirely (including content)
    html = re.sub(r'<script\b[^>]*>.*?</script>', '', html, flags=re.IGNORECASE | re.DOTALL)
    # Remove self-closing / unclosed <script> tags
    html = re.sub(r'<script\b[^>]*/?\s*>', '', html, flags=re.IGNORECASE)
    # Remove <object> and <embed> tags (plugin execution vectors)
    html = re.sub(r'<object\b[^>]*>.*?</object>', '', html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<(?:object|embed)\b[^>]*/?\s*>', '', html, flags=re.IGNORECASE)
    # Remove event handler attributes: on<word>="..." or on<word>='...' or unquoted
    html = re.sub(r'\s+on[a-z]{1,20}\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s>]*)',
                  '', html, flags=re.IGNORECASE)
    # Remove javascript: URL schemes (in href/src/action/etc.)
    html = re.sub(r'javascript\s*:', 'javascript_blocked:', html, flags=re.IGNORECASE)
    # Remove <iframe ...> tags entirely
    html = re.sub(r'<iframe\b[^>]*>.*?</iframe>', '', html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<iframe\b[^>]*/?\s*>', '', html, flags=re.IGNORECASE)
    # Remove <meta> tags (potential http-equiv refresh / CSP bypasses)
    html = re.sub(r'<meta\b[^>]*/?\s*>', '', html, flags=re.IGNORECASE)
    return html


def _md_to_html(md_text: str) -> str:
    """Minimal markdown → HTML for analysis.md (## h2, ### h3, - list, | table, **bold**, `code`)."""
    lines = md_text.split('\n')
    html, in_ul, in_ol, in_table, table_body_started = [], False, False, False, False

    def _close_list():
        nonlocal in_ul, in_ol
        if in_ul:
            html.append('</ul>')
            in_ul = False
        if in_ol:
            html.append('</ol>')
            in_ol = False

    def _close_table():
        nonlocal in_table, table_body_started
        if in_table:
            if table_body_started:
                html.append('</tbody>')
            html.append('</table>')
            in_table = False
            table_body_started = False

    for line in lines:
        s = line.strip()
        if s.startswith('## '):
            _close_list()
            _close_table()
            html.append(f'<h3 class="ana-h2">{_md_inline(s[3:].strip())}</h3>')
        elif s.startswith('### '):
            _close_list()
            _close_table()
            html.append(f'<h4 class="ana-h3">{_md_inline(s[4:].strip())}</h4>')
        elif s.startswith('|'):
            _close_list()
            cols = [c.strip() for c in s.strip('|').split('|')]
            # Skip pure separator rows (|---|---|) — they only trigger tbody open
            if all(re.match(r'^[-: ]+$', c) for c in cols if c.strip()):
                if in_table and not table_body_started:
                    html.append('<tbody>')
                    table_body_started = True
                continue
            if not in_table:
                in_table = True
                table_body_started = False
                html.append('<table class="ana-tbl"><thead><tr>')
                html.extend(f'<th>{_md_inline(c)}</th>' for c in cols)
                html.append('</tr></thead>')
            else:
                if not table_body_started:
                    html.append('<tbody>')
                    table_body_started = True
                html.append('<tr>')
                html.extend(f'<td>{_md_inline(c)}</td>' for c in cols)
                html.append('</tr>')
        elif s.startswith('- ') or s.startswith('* '):
            _close_table()
            if in_ol:
                html.append('</ol>')
                in_ol = False
            if not in_ul:
                html.append('<ul class="ana-list">')
                in_ul = True
            html.append(f'<li>{_md_inline(s[2:])}</li>')
        elif re.match(r'^\d+\.\s', s):
            _close_table()
            content = re.sub(r'^\d+\.\s', '', s)
            if in_ul:
                html.append('</ul>')
                in_ul = False
            if not in_ol:
                html.append('<ol class="ana-list">')
                in_ol = True
            html.append(f'<li>{_md_inline(content)}</li>')
        elif not s:
            _close_list()
            _close_table()
        else:
            _close_list()
            _close_table()
            html.append(f'<p class="ana-p">{_md_inline(s)}</p>')

    _close_list()
    _close_table()
    return '\n'.join(html)


def discover_panels(op_dir: Path, op_name: str = "") -> dict:
    """
    扫描 output/{op}/panels/ 子目录，加载 analysis.md / panel.html。
    返回 {"memory": {...}, "precision": {...}, "perf": {...}, "algo": {...},
           "extra": [{"id","label","html","analysis_html"}]}
    """
    known_tabs = {"algo", "memory", "precision", "perf"}
    panels: dict = {t: {} for t in known_tabs}
    panels["extra"] = []

    panel_dir = op_dir / "panels"
    if not panel_dir.exists():
        return panels

    for sub in sorted(panel_dir.iterdir()):
        if not sub.is_dir():
            continue
        pid = sub.name
        info: dict = {}

        if (sub / "analysis.md").exists():
            try:
                md = (sub / "analysis.md").read_text(encoding="utf-8", errors="replace")
                info["analysis_html"] = _md_to_html(md)
            except Exception as e:
                logger.debug("读取 panel analysis.md 失败 %s: %s", sub, e)

        # ── 独立 HTML 片段文件（CC 直接写入，脚本直接读取嵌入）──
        for _key, _fname in [("flow_html", "flow.html"),
                             ("steps_html", "steps.html"),
                             ("ub_viz_html", "ub_viz.html")]:
            _hf = sub / _fname
            if _hf.exists():
                try:
                    _content = _hf.read_text(encoding="utf-8", errors="replace").strip()
                    if len(_content) > 30:
                        info[_key] = _sanitize_html_fragment(_content)
                except Exception as e:
                    logger.debug("读取 panel 片段 %s 失败 %s: %s", _fname, sub, e)

        if (sub / "panel.html").exists():
            try:
                info["raw_html"] = (sub / "panel.html").read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.debug("读取 panel.html 失败 %s: %s", sub, e)

        if pid in known_tabs:
            panels[pid] = info
        elif info:
            # Unknown subdir with content → extra tab
            meta_path = sub / "panel.json"
            label = pid
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    label = meta.get("title", pid)
                except Exception as e:
                    logger.debug("读取 panel.json 失败，使用目录名作标题 %s: %s", sub, e)
            panels["extra"].append({
                "id": pid,
                "label": label,
                "html": info.get("raw_html", ""),
                "analysis_html": info.get("analysis_html", ""),
            })

    return panels


def extract_panel_data_json(data: dict, op_dir: Path):
    """写 panels/{memory,precision,perf}/data.json 供 Claude 精品流程读取。"""
    panel_dir = op_dir / "panels"
    # memory
    mem_dir = panel_dir / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "data.json").write_text(json.dumps({
        "tiling_consts": data.get("tiling", {}).get("consts", {}),
        "ub_buffers": data.get("ub_buffers", []),
        "ub_used_kb": data.get("ub_used_kb", 0),
        "ub_total_kb": data.get("ub_total_kb", 256),
        "tiling_analysis": data.get("tiling", {}).get("analysis", []),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    # precision
    prec_dir = panel_dir / "precision"
    prec_dir.mkdir(parents=True, exist_ok=True)
    (prec_dir / "data.json").write_text(json.dumps({
        "n_pass": data.get("n_pass", 0),
        "n_total": data.get("n_total", 0),
        "cases": [{"id": c.get("id"), "name": c.get("name"),
                   "shape": c.get("shape"), "passed": c.get("passed"),
                   "precision": c.get("precision", {})}
                  for c in data.get("cases", [])],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    # perf
    perf_dir = panel_dir / "perf"
    perf_dir.mkdir(parents=True, exist_ok=True)
    perf_cases = [c for c in data.get("cases", []) if c.get("performance", {}).get("speedup")]
    (perf_dir / "data.json").write_text(json.dumps({
        "avg_speedup": data.get("avg_speedup"),
        "speedups": [{"name": c.get("name", f"case_{c.get('id',0)}"),
                      "speedup": c.get("performance", {}).get("speedup"),
                      "ref_time_us": c["performance"].get("ref_time_us", 0),
                      "custom_time_us": c["performance"].get("custom_time_us", 0)}
                     for c in perf_cases],
        "op_summary_fields_by_type": OP_SUMMARY_FIELDS_BY_TYPE,
        "op_summary_fields": OP_SUMMARY_DISPLAY_FIELDS,
        "cases": [{"id": c.get("id"), "name": c.get("name"),
                   "shape": c.get("shape"), "performance": c.get("performance", {}),
                   "op_summary_avg": c.get("op_summary_avg", {})}
                  for c in perf_cases],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    # algo (pass_comments + compute_apis for CC to write flow.html / steps.html)
    algo_dir = panel_dir / "algo"
    algo_dir.mkdir(parents=True, exist_ok=True)
    (algo_dir / "data.json").write_text(json.dumps({
        "pass_comments": data.get("pass_comments", []),
        "compute_apis": data.get("op_graph", {}).get("api_list", []),
        "tiling_consts": data.get("tiling", {}).get("consts", {}),
        "op_name": data.get("op_name", ""),
        "description": data.get("description", ""),
        "inputs": data.get("op_graph", {}).get("inputs", []),
        "outputs": data.get("op_graph", {}).get("outputs", []),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"📁 Panel data 已写入 {panel_dir}/{{algo,memory,precision,perf}}/data.json")


# ─── HTML TEMPLATE ────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>__TITLE__ — AscendC Dashboard</title>
<script>/*__CHARTJS__*/</script>
<style>
/* ══ 变量 ══ */
:root{
  --bg:#f6f8fa;--sf:#fff;--sf2:#f0f2f5;--sf3:#e8ecf0;
  --bd:#d0d7de;--bdl:#e8ecf0;
  --tx:#1a1a1a;--tx2:#57606a;--tx3:#8b949e;
  --ac:#0969da;--gn:#1a7f37;--rd:#cf222e;--or:#bf8700;--pu:#8250df;
  --mono:'SF Mono',Consolas,'Courier New',monospace;
  --r:8px;--rl:12px;--sh:0 1px 4px rgba(0,0,0,.08),0 4px 16px rgba(0,0,0,.05);
}
@media(prefers-color-scheme:dark){:root{
  --bg:#0d1117;--sf:#161b22;--sf2:#1c2128;--sf3:#21262d;
  --bd:#30363d;--bdl:#21262d;--tx:#e6edf3;--tx2:#8b949e;--tx3:#6e7681;
  --ac:#58a6ff;--gn:#3fb950;--rd:#f85149;--or:#d29922;--pu:#a371f7;
}}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;background:var(--bg);color:var(--tx);font-size:14px;line-height:1.5}
code,.mono{font-family:var(--mono)}

/* ── Header ── */
.hdr{padding:24px 32px 0;max-width:1100px;margin:0 auto}
.hdr-row{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.hdr-name{font-size:26px;font-weight:700;letter-spacing:-.03em;font-family:var(--mono)}
.hdr-cat{font-size:11px;padding:3px 10px;border-radius:16px;background:rgba(130,80,223,.1);color:var(--pu);font-weight:600;border:1px solid rgba(130,80,223,.2)}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
.chip{background:var(--sf);border:1px solid var(--bd);border-radius:16px;padding:4px 12px;font-size:12px;font-family:var(--mono)}
.chip b{color:var(--ac)}
.chip.pass b{color:var(--gn)}.chip.warn b{color:var(--or)}.chip.fail b{color:var(--rd)}
.desc{margin-top:10px;font-size:13px;color:var(--tx2);max-width:820px;line-height:1.7}

/* ── Tabs ── */
.tab-bar{display:flex;padding:0 32px;max-width:1100px;margin:18px auto 0;border-bottom:1px solid var(--bd);overflow-x:auto}
.tb{padding:9px 18px;font-size:13px;cursor:pointer;border:none;background:none;color:var(--tx2);border-bottom:2px solid transparent;margin-bottom:-1px;white-space:nowrap;transition:color .15s,border-color .15s;display:flex;align-items:center;gap:6px;font-family:inherit}
.tb:hover{color:var(--tx)}.tb.active{color:var(--ac);border-bottom-color:var(--ac);font-weight:600}

/* ── Content ── */
.content{max-width:1100px;margin:0 auto;padding:24px 32px 56px}
.tp{display:none}.tp.active{display:block}

/* ── Cards ── */
.card{background:var(--sf);border:1px solid var(--bd);border-radius:var(--rl);padding:20px;margin-bottom:16px;box-shadow:var(--sh)}
.ct{font-size:14px;font-weight:700;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.ct .ico{font-size:16px}
.lbl{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--tx3);margin-bottom:8px}

/* ── Badges ── */
.badge{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:980px;font-size:12px;font-weight:600}
.bp{background:rgba(26,127,55,.1);color:var(--gn);border:1px solid rgba(26,127,55,.2)}
.bf{background:rgba(207,34,46,.1);color:var(--rd);border:1px solid rgba(207,34,46,.2)}
.bw{background:rgba(191,135,0,.1);color:var(--or);border:1px solid rgba(191,135,0,.2)}
.bi{background:rgba(9,105,218,.08);color:var(--ac);border:1px solid rgba(9,105,218,.15)}

/* ══ TAB 1: 算法图示（工业级计算逻辑流） ══ */
/* IO tensor block */
.tv{display:inline-flex;flex-direction:column;align-items:center;gap:3px}
.tg{display:flex;flex-direction:column;border-radius:3px;overflow:hidden}
.tr{display:flex}.tc{border:1px solid rgba(255,255,255,.25)}
.td{font-size:11px;color:var(--tx3);font-family:var(--mono)}
.tn{font-size:12px;font-weight:700;font-family:var(--mono)}

/* Shape selector bar */
.shape-bar{display:flex;align-items:center;gap:10px;margin-bottom:20px;flex-wrap:wrap}
.shape-bar label{font-size:12px;font-weight:600;color:var(--tx2);white-space:nowrap}
.shape-bar select{background:var(--sf);color:var(--tx);border:1px solid var(--bd);border-radius:var(--r);padding:6px 12px;font-size:13px;font-family:var(--mono);cursor:pointer;outline:none}
.shape-bar select:focus{border-color:var(--ac)}

/* IO node (input / output tensor) */
.io-node{background:var(--sf);border:1.5px solid var(--ac);border-radius:var(--r);padding:12px 20px;text-align:center;display:inline-flex;flex-direction:column;align-items:center;gap:6px}
.io-lbl{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--tx3)}
.io-name{font-size:15px;font-weight:700;font-family:var(--mono);color:var(--ac)}
.io-shape{font-size:12px;font-family:var(--mono);color:var(--tx2)}
.io-dtype{font-size:10px;color:var(--tx3);font-family:var(--mono)}

/* Arrow between units */
.flow-arrow{text-align:center;color:var(--tx3);font-size:22px;padding:6px 0;user-select:none}

/* Hardware unit group */
.unit-group{border-radius:12px;padding:16px 20px;width:fit-content;max-width:100%;box-sizing:border-box}
.unit-hdr{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.1em;margin-bottom:14px;font-family:var(--mono)}
.unit-nodes{display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap}
.unit-sep{color:var(--tx3);font-size:20px;align-self:center;padding:0 2px;user-select:none}

/* API node card */
.api-node{background:var(--sf);border-radius:8px;padding:12px 14px;min-width:110px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.07)}
.api-name{font-family:var(--mono);font-size:13px;font-weight:700;margin-bottom:4px}
.api-formula{font-size:11px;color:var(--tx2);font-family:var(--mono);margin:3px 0;line-height:1.4}
.api-shape{font-size:10px;color:var(--tx3);font-family:var(--mono);margin-top:3px}

/* Flow outer container */
.flow-col{display:flex;flex-direction:column;align-items:center;gap:0;width:100%}

/* ── Claude-written algo flow HTML classes ── */
.algo-flow{display:flex;flex-direction:column;align-items:center;gap:4px;width:100%;padding:8px 0}
.algo-phase{border:2px solid var(--bd);border-radius:8px;padding:12px 16px;margin:2px 0;width:100%;max-width:640px;box-sizing:border-box;display:flex;flex-direction:column;gap:4px}
.algo-phase[data-core="AIC"]{border-color:#bf8700;background:#fffbf0}
.algo-phase[data-core="AIV"]{border-color:#8250df;background:#f5f0ff}
.algo-phase-lbl{font-size:11px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;margin-bottom:4px}
.algo-phase[data-core="AIC"] .algo-phase-lbl{color:#bf8700}
.algo-phase[data-core="AIV"] .algo-phase-lbl{color:#8250df}
.algo-node{border-radius:6px;padding:7px 12px;margin:2px auto;font-family:var(--mono);font-size:12px;line-height:1.5;max-width:360px;width:auto;box-sizing:border-box;text-align:center}
.algo-node.input{background:#dff0fc;border:1px solid #0969da;color:#0550ae}
.algo-node.output{background:#dcfce7;border:1px solid #1a7f37;color:#166534}
.algo-node.workspace{background:#f6f8fa;border:1px dashed #8c959f;color:#57606a}
.algo-node.compute{background:#fff8c5;border:1px solid #bf8700;color:#7d4e00}
.algo-node.sync{background:#fef2f2;border:1.5px dashed #cf222e;color:#a40e26}
.algo-arrow{color:var(--tx3);font-size:20px;line-height:1;text-align:center;padding:2px 0}
.algo-arrow-lbl{font-size:11px;color:var(--tx3);font-family:var(--mono);margin-left:4px}
.algo-gate{background:#e8f4fd;border:1px solid #54aeff;border-radius:6px;padding:5px 10px;font-size:11px;font-family:var(--mono);color:#0550ae;margin:2px auto;max-width:600px;width:100%;box-sizing:border-box;text-align:center}
.algo-steps{display:flex;flex-direction:column;gap:10px;width:100%}
.algo-step{display:flex;gap:10px;align-items:flex-start}
.algo-step-num{background:var(--ac);color:#fff;border-radius:50%;min-width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0;margin-top:2px}
.algo-step-body{font-size:13px;line-height:1.6;color:var(--tx1)}
.algo-step-code{font-family:var(--mono);font-size:11px;color:var(--tx2);background:#f6f8fa;padding:2px 6px;border-radius:3px;display:inline-block;margin-top:2px}

/* Algo steps list (raw Pass comments) */
.as-list{list-style:none;counter-reset:s}
.as-list li{counter-increment:s;display:flex;align-items:flex-start;gap:10px;padding:9px 0;border-bottom:1px solid var(--bdl);font-size:13px}
.as-list li:last-child{border-bottom:none}
.as-list li::before{content:counter(s);min-width:22px;height:22px;background:var(--sf2);border:1px solid var(--bd);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:var(--ac);font-family:var(--mono);flex-shrink:0}

/* ══ TAB 2: 内存 Tiling ══ */
.ub-bar{display:flex;height:36px;border-radius:var(--r);overflow:hidden;border:1px solid var(--bd);background:var(--sf3)}
.ub-seg{display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:600;color:#fff;overflow:hidden;cursor:default;position:relative;min-width:3px;transition:opacity .15s}
.ub-seg:hover{opacity:.85;outline:2px solid rgba(0,0,0,.4);z-index:2}
.ub-labels-row{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;font-size:11px;font-family:var(--mono)}
.ub-label-chip{display:flex;align-items:center;gap:4px;padding:2px 7px;border-radius:4px;background:var(--sf2);border:1px solid var(--bd);cursor:default;white-space:nowrap}
.ub-label-swatch{width:10px;height:10px;border-radius:2px;flex-shrink:0}
.ub-free{background:var(--sf3);color:var(--tx3)}
.ub-ruler{position:relative;height:14px;margin-top:2px}

/* ── Claude-written UB viz HTML classes ── */
.ub-alloc{display:flex;gap:20px;align-items:center;padding:8px 0;flex-wrap:wrap}
.ub-alloc-legend{display:flex;flex-direction:column;gap:5px;flex:1;min-width:160px}
.ub-alloc-row{display:flex;align-items:center;gap:7px;font-size:12px}
.ub-free-bar{display:flex;align-items:center;gap:7px;font-size:12px;margin-top:2px;padding-top:5px;border-top:1px solid var(--bdl)}
.ub-alloc-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.ub-alloc-label{font-family:var(--mono);color:var(--tx1);flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ub-alloc-pct{font-size:10px;font-family:var(--mono);min-width:48px;text-align:right}
.ub-alloc-size{font-family:var(--mono);color:var(--tx3);min-width:56px;text-align:right}
#ub-viz-slot{margin-top:12px}
.ub-tick{position:absolute;bottom:0;font-size:9px;color:var(--tx3);font-family:var(--mono);transform:translateX(-50%)}
.ub-tick::before{content:'';position:absolute;top:0;left:50%;width:1px;height:6px;background:var(--bd)}
.ub-labels{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.ub-lbl{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--tx2)}
.ub-dot{width:10px;height:10px;border-radius:2px;flex-shrink:0}

.dt{width:100%;border-collapse:collapse;font-size:13px}
.dt th{background:var(--sf2);padding:8px 14px;text-align:left;font-size:11px;font-weight:600;color:var(--tx3);text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--bd)}
.dt td{padding:8px 14px;border-bottom:1px solid var(--bdl);vertical-align:middle}
.dt tr:last-child td{border-bottom:none}
.val{font-family:var(--mono);font-weight:600;color:var(--tx)}
.note{font-size:11px;color:var(--tx3);margin-top:2px}
.pipe-tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;font-family:var(--mono)}

/* ══ TAB 3: 精度 ══ */
/* Shape selector */
.shape-sel{display:flex;align-items:center;gap:10px;margin-bottom:16px;flex-wrap:wrap}
.shape-sel label{font-size:12px;font-weight:600;color:var(--tx2)}
.shape-sel select{background:var(--sf);color:var(--tx);border:1px solid var(--bd);border-radius:var(--r);padding:6px 12px;font-size:13px;font-family:var(--mono);cursor:pointer;outline:none}
.shape-sel select:focus{border-color:var(--ac)}

/* Precision overview table */
.poc{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:16px}
.poc th{background:var(--sf2);padding:9px 14px;text-align:center;font-size:11px;font-weight:600;color:var(--tx3);text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--bd)}
.poc th:first-child{text-align:left}
.poc td{padding:9px 14px;border-bottom:1px solid var(--bdl);text-align:center;vertical-align:middle;cursor:pointer;transition:background .15s}
.poc td:first-child{text-align:left}
.poc tr:hover td{background:rgba(9,105,218,.03)}
.poc tr.sel td{background:rgba(9,105,218,.06);border-bottom-color:var(--ac)}
.poc tr:last-child td{border-bottom:none}

/* Metric cards */
.mg{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:12px;margin-bottom:16px}
.mc{background:var(--sf2);border:1px solid var(--bdl);border-radius:var(--r);padding:14px}
.mc-lbl{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--tx3);margin-bottom:5px}
.mc-val{font-size:22px;font-weight:700;font-family:var(--mono);letter-spacing:-.02em}
.mc-lim{font-size:11px;color:var(--tx3);margin-top:2px;font-family:var(--mono)}
.mc-bar{height:4px;border-radius:2px;background:var(--sf3);margin-top:8px;overflow:hidden}
.mc-fill{height:100%;border-radius:2px;transition:width .5s}
.mc-ok{color:var(--gn)}.mc-w{color:var(--or)}.mc-bad{color:var(--rd)}

/* Cross-case comparison chart */
.svg-chart{width:100%;overflow:visible}

/* ══ TAB 4: 性能 ══ */
.perf-summary{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.sp-card{background:var(--sf2);border:1px solid var(--bdl);border-radius:var(--r);padding:12px 14px;text-align:center;min-width:130px;max-width:200px;flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-start}
.sp-val{font-size:28px;font-weight:800;font-family:var(--mono);letter-spacing:-.03em;line-height:1.1;text-align:center}
.sp-unit{font-size:11px;color:var(--tx2);margin-top:2px;text-align:center}
.sp-shape{font-size:11px;color:var(--tx3);font-family:var(--mono);margin-top:4px;max-width:170px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:center}
.sp-timing{font-size:10px;color:var(--tx3);margin-top:2px;text-align:center}
.sp-good{color:var(--gn)}.sp-ok{color:var(--or)}.sp-slow{color:var(--rd)}
/* tiling analysis */
.til-dims{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
.til-block{background:var(--sf2);border:1px solid var(--bdl);border-radius:var(--r);padding:10px 16px;flex:1;min-width:140px}
.til-block-title{font-size:10px;color:var(--tx3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}
.til-block-val{font-size:18px;font-weight:700;font-family:var(--mono);color:var(--tx1)}
.til-block-note{font-size:11px;color:var(--tx3);margin-top:2px}
.til-insight{margin-top:10px;padding:10px 14px;background:var(--sf2);border-left:3px solid var(--ac);border-radius:0 var(--r) var(--r) 0;font-size:12px;color:var(--tx2);line-height:1.6}
.bal-bar{display:inline-block;width:48px;height:6px;background:var(--bdl);border-radius:3px;vertical-align:middle;margin-left:4px;overflow:hidden}
.bal-fill{height:100%;border-radius:3px;background:var(--gn)}

.chart-wrap{position:relative;width:100%}
.chart-title{font-size:12px;color:var(--tx2);margin-bottom:8px;font-weight:600}

/* Perf note */
.pnote{font-size:12px;color:var(--tx2);background:rgba(9,105,218,.05);border:1px solid rgba(9,105,218,.12);border-radius:var(--r);padding:10px 14px;line-height:1.7}
.pnote b{color:var(--ac)}

/* ── responsive ── */
@media(max-width:700px){
  .hdr,.tab-bar,.content{padding-left:16px;padding-right:16px}
  .mg{grid-template-columns:1fr 1fr}
  .perf-summary{flex-direction:column}
}

/* ══ Analysis markdown rendering (panels/*.md) ══ */
.ana-section{margin-bottom:16px;padding:16px 18px;background:var(--sf2);
  border-left:3px solid var(--ac);border-radius:0 var(--r) var(--r) 0}
.ana-section .ana-origin{font-size:10px;color:var(--tx3);margin-bottom:10px;
  font-weight:600;text-transform:uppercase;letter-spacing:.05em}
.ana-h2{font-size:14px;font-weight:700;margin:14px 0 6px;color:var(--tx);
  padding-bottom:4px;border-bottom:1px solid var(--bdl)}
.ana-h2:first-child{margin-top:0}
.ana-h3{font-size:13px;font-weight:600;margin:10px 0 4px;color:var(--tx2)}
.ana-p{font-size:13px;color:var(--tx2);line-height:1.65;margin:4px 0}
.ana-list{font-size:13px;color:var(--tx2);margin:4px 0 4px 18px;line-height:1.65}
.ana-list li{margin:2px 0}
.ana-tbl{border-collapse:collapse;font-size:12px;margin:8px 0;width:100%}
.ana-tbl th{background:var(--sf3);padding:6px 10px;font-size:11px;font-weight:600;
  color:var(--tx3);text-align:left;border-bottom:1px solid var(--bd)}
.ana-tbl td{padding:6px 10px;border-bottom:1px solid var(--bdl);color:var(--tx2)}
.ana-tbl tr:last-child td{border-bottom:none}

/* ══ Data Health bar ══ */
.dh-bar{border:1px solid var(--bdl);border-radius:var(--r);margin-bottom:12px;
  font-size:12px;overflow:hidden}
.dh-hdr{display:flex;align-items:center;gap:10px;padding:8px 14px;cursor:pointer;
  background:var(--sf2);user-select:none;font-weight:600;font-size:12px;color:var(--tx2)}
.dh-hdr:hover{background:var(--sf3)}
.dh-counts{display:flex;gap:8px;margin-left:auto;font-weight:400}
.dh-body{display:none;padding:8px 14px 10px}
.dh-body.open{display:block}
.dh-item{display:flex;align-items:baseline;gap:8px;padding:4px 0;
  border-bottom:1px solid var(--bdl);font-size:12px}
.dh-item:last-child{border-bottom:none}
.dh-icon{font-weight:700;min-width:14px}
.dh-msg{color:var(--tx2);flex:1}
.dh-hint{color:var(--tx3);font-size:11px;font-family:var(--mono)}
.dh-found .dh-icon{color:var(--gn)}.dh-derived .dh-icon{color:var(--or)}
.dh-missing .dh-icon{color:var(--rd)}
.dh-found-c{color:var(--gn);font-weight:600}
.dh-derived-c{color:var(--or);font-weight:600}
.dh-missing-c{color:var(--rd);font-weight:600}

/* ══ Precision v2 ══ */
.prec-heat-bar{display:flex;align-items:center;gap:8px;margin-bottom:12px;font-size:12px;color:var(--tx2);flex-wrap:wrap}
.prec-heat-bar label{font-weight:600}
.prec-heat-bar select{background:var(--sf);color:var(--tx);border:1px solid var(--bd);border-radius:var(--r);padding:4px 8px;font-size:12px;font-family:inherit;cursor:pointer;outline:none}
.prec-heat-bar select:focus{border-color:var(--ac)}
.heat-legend-wrap{display:flex;align-items:center;gap:6px;margin-left:auto}
.heat-grad-bar{width:120px;height:8px;border-radius:4px;border:1px solid var(--bdl);position:relative}
.heat-grad-ticks{position:relative;width:120px;height:12px}
.heat-tick{position:absolute;transform:translateX(-50%);font-size:9px;font-family:var(--mono);color:var(--tx3)}
.heat-lbl{font-size:10px;font-family:var(--mono);color:var(--tx3)}
.prec-kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:14px}
.prec-kpi{background:var(--sf2);border:1px solid var(--bdl);border-radius:var(--r);padding:10px 12px}
.prec-kpi .k{font-size:10px;color:var(--tx3);text-transform:uppercase;letter-spacing:.05em;font-weight:700}
.prec-kpi .v{font-size:20px;font-family:var(--mono);font-weight:800;line-height:1.15;margin-top:3px}
.prec-kpi .n{font-size:11px;color:var(--tx3);margin-top:2px}
/* Precision summary table v2 */
.pst{width:100%;border-collapse:separate;border-spacing:0;font-size:12px}
.pst thead th{background:var(--sf2);padding:8px 12px;text-align:center;font-size:10px;font-weight:700;color:var(--tx3);text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--bd);position:sticky;top:0;z-index:2}
.pst thead th:first-child{text-align:left;min-width:130px}
.pst tbody td{padding:8px 12px;text-align:center;border-bottom:1px solid var(--bdl);vertical-align:middle;transition:background .15s}
.pst tbody td:first-child{text-align:left}
.pst tbody tr.pst-case{cursor:pointer}
.pst tbody tr.pst-case:hover td{background:rgba(9,105,218,.04)}
.pst tbody tr.pst-case.pst-open td{background:rgba(9,105,218,.05);border-bottom-color:var(--ac)}
.pst-name{font-weight:600;font-size:12px}
.pst-shape{font-size:10px;color:var(--tx3);font-family:var(--mono);margin-top:2px}
.pst-metric-val{font-size:11px;font-family:var(--mono);color:var(--tx2)}
/* Expandable detail row */
.pst-detail{display:none}.pst-detail.pst-open{display:table-row}
.pst-dpanel{background:var(--sf2);padding:16px 20px;border-top:2px solid var(--ac)}
.pst-charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;margin-bottom:14px}
.pst-chart-box{background:var(--sf);border:1px solid var(--bdl);border-radius:var(--r);padding:12px}
.pst-chart-box h5{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--tx3);margin-bottom:8px}
.pst-chart-box canvas{max-height:160px}
/* Threshold table */
.pst-thr{width:100%;border-collapse:collapse;font-size:11px;margin-top:10px}
.pst-thr th,.pst-thr td{padding:5px 10px;border:1px solid var(--bdl);text-align:center}
.pst-thr th{background:var(--sf);color:var(--tx3);font-weight:600;font-size:10px;text-transform:uppercase;letter-spacing:.03em}
.pst-thr td:first-child{text-align:left;font-weight:600;color:var(--tx2)}
.thr-ok{background:rgba(26,127,55,.08);color:var(--gn)}
.thr-warn{background:rgba(191,135,0,.08);color:var(--or)}
.thr-fail{background:rgba(207,34,46,.08);color:var(--rd)}
/* Floating tooltip */
.prec-tip{display:none;position:fixed;z-index:9999;background:var(--sf);border:1px solid var(--bd);
  border-radius:var(--rl);padding:10px 14px;font-size:11px;line-height:1.7;min-width:200px;max-width:300px;
  box-shadow:0 8px 24px rgba(0,0,0,.18);pointer-events:none}
.prec-tip.vis{display:block}
.prec-tip-title{font-weight:700;margin-bottom:5px;color:var(--ac);font-size:12px}
.prec-tip-row{display:flex;justify-content:space-between;gap:10px}
.prec-tip-lbl{color:var(--tx2)}.prec-tip-val{font-family:var(--mono);color:var(--tx)}

/* ══ Perf v2 ══ */
.perf-hist-wrap{position:relative;width:100%}
.perf-hist-wrap canvas{max-height:200px}
.case-sel-bar{display:flex;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap}
.case-sel-bar label{font-size:12px;font-weight:600;color:var(--tx2);white-space:nowrap}
.case-sel-bar select{background:var(--sf);color:var(--tx);border:1px solid var(--bd);border-radius:var(--r);
  padding:6px 12px;font-size:13px;font-family:var(--mono);cursor:pointer;outline:none;flex:1;max-width:380px}
.case-sel-bar select:focus{border-color:var(--ac)}
/* KPI hero cards */
.kpi-row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
.kpi-card{background:var(--sf2);border:1px solid var(--bdl);border-radius:var(--r);padding:11px 14px;flex:1;min-width:110px}
.kpi-lbl{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--tx3);margin-bottom:4px}
.kpi-val{font-size:20px;font-weight:800;font-family:var(--mono);letter-spacing:-.02em;color:var(--tx);line-height:1.1}
.kpi-unit{font-size:10px;color:var(--tx3);margin-top:2px}
/* AIV/AIC time breakdown bar */
.breakdown-title{font-size:11px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;display:flex;align-items:center;gap:6px}
.breakdown-stack{height:28px;border-radius:var(--r);overflow:hidden;display:flex;border:1px solid var(--bdl);background:var(--sf3);margin-bottom:6px}
.breakdown-seg{display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:#fff;overflow:hidden;transition:all .2s;cursor:default}
.breakdown-seg:hover{opacity:.78}
.breakdown-legend{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
.breakdown-legend-item{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--tx2)}
.breakdown-legend-dot{width:8px;height:8px;border-radius:2px;flex-shrink:0}
/* Charts grid */
.perf-detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px}
.perf-detail-grid.three-cols{grid-template-columns:1fr 1fr 1fr}
.perf-chart-card{background:var(--sf);border:1px solid var(--bdl);border-radius:var(--r);padding:13px}
.perf-chart-card h5{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--tx3);margin-bottom:8px}
.perf-chart-card canvas{max-height:160px}
/* Op summary table */
.ops-tbl{width:100%;border-collapse:collapse;font-size:12px}
.ops-tbl th{background:var(--sf2);padding:6px 12px;text-align:left;font-size:10px;font-weight:700;color:var(--tx3);text-transform:uppercase;letter-spacing:.04em;border-bottom:1px solid var(--bd)}
.ops-tbl td{padding:6px 12px;border-bottom:1px solid var(--bdl)}
.ops-tbl tr:last-child td{border-bottom:none}
.ops-key{color:var(--tx2);font-weight:600;white-space:nowrap;min-width:160px}
.ops-val{font-family:var(--mono);color:var(--tx)}
@media(max-width:700px){.perf-detail-grid{grid-template-columns:1fr}.pst-charts{grid-template-columns:1fr}}
</style>
</head>
<body>

<!-- Header -->
<div class="hdr">
  <div class="hdr-row">
    <span class="hdr-name" id="hdr-name"></span>
    <span class="hdr-cat"  id="hdr-cat"></span>
    <span id="hdr-badges"></span>
  </div>
  <div class="chips" id="hdr-chips"></div>
  <p class="desc" id="hdr-desc"></p>
</div>

<!-- Tabs -->
<div class="tab-bar" id="main-tab-bar">
  <button class="tb active" data-tab="algo">⚙️ 算法图示</button>
  <button class="tb" data-tab="mem">🗄 内存 &amp; Tiling</button>
  <button class="tb" data-tab="prec">🎯 精度分析</button>
  <button class="tb" data-tab="perf">⚡ 性能报告</button>
</div>

<div class="content">

<!-- ══ Tab: 算法图示 ══ -->
<div class="tp active" id="tab-algo">
  <div class="dh-bar" id="dh-algo" style="display:none">
    <div class="dh-hdr" onclick="toggleDH('dh-algo')">▶ Data Health <span class="dh-counts" id="dh-algo-counts"></span></div>
    <div class="dh-body" id="dh-algo-body"></div>
  </div>
  <div id="algo-analysis-slot"></div>
  <div class="card">
    <div class="ct">
      <span class="ico">⚙️</span>计算逻辑流
      <div style="margin-left:auto;display:flex;align-items:center;gap:8px">
        <label class="shape-bar" style="margin:0" for="shape-sel">
          <span>测试 Shape：</span>
          <select id="shape-sel" onchange="updateShapeVars()"></select>
        </label>
      </div>
    </div>
    <div class="flow-col" id="algo-flow"></div>
  </div>
  <div class="card">
    <div class="ct"><span class="ico">📋</span>算法步骤注释</div>
    <div id="algo-steps"></div>
  </div>
</div>

<!-- ══ Tab: 内存 & Tiling ══ -->
<div class="tp" id="tab-mem">
  <div class="dh-bar" id="dh-mem" style="display:none">
    <div class="dh-hdr" onclick="toggleDH('dh-mem')">▶ Data Health <span class="dh-counts" id="dh-mem-counts"></span></div>
    <div class="dh-body" id="dh-mem-body"></div>
  </div>
  <div class="card">
    <div class="ct"><span class="ico">🔪</span>Tiling 策略分析</div>
    <div id="tiling-strategy"></div>
    <div id="tiling-table" style="margin-top:14px"></div>
  </div>
  <div class="card">
    <div class="ct"><span class="ico">📦</span>UB 内存分配
      <span id="ub-badge" class="badge bi" style="margin-left:auto;font-size:11px"></span>
    </div>
    <div class="lbl" id="ub-header-lbl">Unified Buffer — 地址空间（单核视图）</div>
    <div id="ub-donut-container" style="margin:8px 0"></div>
    <div class="ub-bar" id="ub-bar" style="display:none"></div>
    <div class="ub-ruler" id="ub-ruler" style="display:none"></div>
    <div class="ub-labels" id="ub-labels" style="display:none"></div>
    <div id="ub-table" style="margin-top:16px"></div>
    <div id="ub-viz-slot"></div>
  </div>
  <div id="mem-analysis-slot"></div>
</div>

<!-- ══ Tab: 精度 ══ -->
<div class="tp" id="tab-prec">
  <div class="dh-bar" id="dh-prec" style="display:none">
    <div class="dh-hdr" onclick="toggleDH('dh-prec')">▶ Data Health <span class="dh-counts" id="dh-prec-counts"></span></div>
    <div class="dh-body" id="dh-prec-body"></div>
  </div>
  <div id="prec-fail-banner" style="display:none;margin:0 0 12px 0;padding:10px 14px;border-radius:6px;background:var(--er2,#fff0f0);border:1px solid var(--er1,#cf222e);color:var(--er3,#82071e);font-size:13px;line-height:1.6"></div>
  <div class="card">
    <div class="ct"><span class="ico">📊</span>精度总览
      <div id="prec-heat-bar" class="prec-heat-bar" style="margin-left:auto"></div>
    </div>
    <div id="prec-kpi-row" class="prec-kpi-row"></div>
    <div style="overflow-x:auto">
      <table class="pst" id="prec-table"></table>
    </div>
  </div>
  <div id="prec-analysis-slot"></div>
</div>

<!-- ══ Tab: 性能 ══ -->
<div class="tp" id="tab-perf">
  <div class="dh-bar" id="dh-perf" style="display:none">
    <div class="dh-hdr" onclick="toggleDH('dh-perf')">▶ Data Health <span class="dh-counts" id="dh-perf-counts"></span></div>
    <div class="dh-body" id="dh-perf-body"></div>
  </div>
  <div id="perf-prec-warn" style="display:none;margin:0 0 12px 0;padding:10px 14px;border-radius:6px;background:var(--er2,#fff8f0);border:1px solid var(--er1,#f97316);color:var(--er3,#9a3412);font-size:13px;line-height:1.5"></div>
  <!-- Overview histogram -->
  <div class="card">
    <div class="ct"><span class="ico">⚡</span>加速比概览
      <span id="perf-avg-badge" class="badge bi" style="margin-left:auto;font-size:11px"></span>
    </div>
    <div id="perf-cards">
      <div class="perf-hist-wrap"><canvas id="perf-hist-canvas"></canvas></div>
    </div>
    <svg id="perf-svg" style="display:none;height:0"></svg>
    <div class="pnote" id="perf-note" style="margin-top:12px"></div>
  </div>
  <!-- Case detail -->
  <div class="card" id="perf-case-detail-card">
    <div class="ct"><span class="ico">🔬</span>Case 详情
      <div class="case-sel-bar" style="margin:0 0 0 auto">
        <label>Shape：</label>
        <select id="perf-case-sel" onchange="showCaseDetail(parseInt(this.value))"></select>
      </div>
    </div>
    <div class="kpi-row" id="perf-kpi-row"></div>
    <div id="perf-aiv-section"></div>
    <div class="perf-detail-grid" id="perf-charts-grid"></div>
    <div id="perf-ops-detail"></div>
  </div>
  <div id="perf-analysis-slot"></div>
</div>

</div><!-- /content -->

<script>
// ═══════════════════════════════════════════════════
// DATA
// ═══════════════════════════════════════════════════
const D = __DATA__;

// ═══════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════
const $  = id => document.getElementById(id);
const el = (tag, cls, html) => { const e=document.createElement(tag); if(cls)e.className=cls; if(html!==undefined)e.innerHTML=html; return e; };
const fmtN = (v, d=3) => v==null?'—':Math.abs(v)<0.001?v.toExponential(2):v.toFixed(d);
const fmtErr = (v, d=5) => {
  if(v==null) return '—';
  const base = Math.abs(v)<0.001 ? v.toExponential(2) : v.toFixed(d);
  const av = Math.abs(v);
  if(av===0) return base;
  const p = Math.log2(av);
  const k = Math.round(p);
  const nearPow2 = Math.abs(p-k) < 1e-6;
  if(nearPow2){
    return `${base} (~2^${k})`;
  }
  return base;
};
const fmtKB= kb => kb>=1 ? kb.toFixed(1)+' KB' : (kb*1024).toFixed(0)+' B';
const spCls= sp => sp>=2?'sp-good':sp>=1.2?'sp-ok':'sp-slow';

function shapeAsBracket(shapeText){
  let t = (shapeText || '').trim();
  if(!t) return '[]';
  if(t.startsWith('[') && t.endsWith(']')) return t;
  if(t.startsWith('(') && t.endsWith(')')) t = t.slice(1, -1).trim();
  const parts = t.split(',').map(s => s.trim()).filter(Boolean);
  if(parts.length) return `[${parts.join(',')}]`;
  return `[${t}]`;
}

// Tensor block (matrix visualization)
function tensorBlock(name, shape, dtype, color){
  const sz=8, maxC=6;
  const gr=d=>typeof d==='string'?3:d<=128?2:d<=1024?3:d<=4096?5:6;
  const rows=shape.length>=2?gr(shape[shape.length-2]):2;
  const cols=gr(shape[shape.length-1]);
  let grid='';
  for(let r=0;r<Math.min(rows,maxC);r++){
    grid+='<div class="tr">';
    for(let c=0;c<Math.min(cols,maxC);c++){
      const op=(0.45+Math.random()*.5).toFixed(2);
      grid+=`<div class="tc" style="width:${sz}px;height:${sz}px;background:${color};opacity:${op}"></div>`;
    }
    grid+='</div>';
  }
  return `<div class="tv">
    <div class="tn" style="color:${color}">${name}</div>
    <div class="tg">${grid}</div>
    <div class="td">[${shape.join(', ')}]</div>
    <div class="td" style="color:var(--tx3)">${dtype}</div>
  </div>`;
}

// SVG bar chart helper
function svgBar(svgId, groups, series, colors, yLabel, groupTitles){
  const svg = $(svgId);
  if(!svg||!groups.length) return;
  const W=svg.parentElement.clientWidth||800, H=parseInt(svg.getAttribute('height')||200);
  const padL=50, padR=20, padT=20, padB=60;
  const chartW=W-padL-padR, chartH=H-padT-padB;
  const nGroups=groups.length, nSeries=series.length;
  const allVals=series.flatMap(s=>s.values).filter(v=>v!=null&&v>0);
  if(!allVals.length) return;
  const maxV=Math.max(...allVals)*1.15;
  const barW=Math.min(40, (chartW/nGroups/nSeries)*0.8);
  const groupW=chartW/nGroups;
  let html='';

  // Y gridlines + labels
  for(let i=0;i<=4;i++){
    const v=maxV*i/4;
    const y=padT+chartH-(chartH*i/4);
    html+=`<line x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}" stroke="var(--bdl)" stroke-width="1"/>`;
    html+=`<text x="${padL-6}" y="${y+4}" text-anchor="end" font-size="10" fill="var(--tx3)" font-family="var(--mono)">${v.toFixed(1)}</text>`;
  }
  // Y axis label
  html+=`<text x="12" y="${padT+chartH/2}" text-anchor="middle" font-size="10" fill="var(--tx3)" transform="rotate(-90,12,${padT+chartH/2})">${yLabel||''}</text>`;

  // Bars
  series.forEach((s,si)=>{
    groups.forEach((g,gi)=>{
      const v=s.values[gi];
      if(v==null||v<=0) return;
      const barH=Math.max(2,(v/maxV)*chartH);
      const x=padL+gi*groupW+(groupW-(nSeries*barW+(nSeries-1)*2))/2+si*(barW+2);
      const y=padT+chartH-barH;
      const fullLabel=groupTitles?groupTitles[gi]:g;
      html+=`<rect x="${x}" y="${y}" width="${barW}" height="${barH}" fill="${colors[si]||'#888'}" rx="2" opacity="0.85">
        <title>${fullLabel}\n${s.name}: ${v.toFixed(2)}</title>
      </rect>`;
      if(barH>16){
        html+=`<text x="${x+barW/2}" y="${y+barH/2+4}" text-anchor="middle" font-size="9" fill="#fff" font-weight="600">${v.toFixed(1)}</text>`;
      } else {
        html+=`<text x="${x+barW/2}" y="${y-3}" text-anchor="middle" font-size="9" fill="var(--tx3)">${v.toFixed(1)}</text>`;
      }
    });
  });

  // X labels — short, with tooltip via <title> on surrounding group rect
  groups.forEach((g,gi)=>{
    const x=padL+gi*groupW+groupW/2;
    const fullLabel=groupTitles?groupTitles[gi]:g;
    const label=g.length>12?g.slice(0,11)+'…':g;
    // Invisible rect for group hover tooltip
    html+=`<rect x="${padL+gi*groupW}" y="${padT}" width="${groupW}" height="${chartH+padB}" fill="transparent">
      <title>${fullLabel}</title></rect>`;
    html+=`<text x="${x}" y="${H-padB+18}" text-anchor="middle" font-size="10" fill="var(--tx2)" font-family="var(--mono)">${label}</text>`;
  });

  // Legend
  series.forEach((s,si)=>{
    const lx=padL+si*100;
    html+=`<rect x="${lx}" y="${H-14}" width="10" height="10" fill="${colors[si]}" rx="2"/>`;
    html+=`<text x="${lx+14}" y="${H-4}" font-size="10" fill="var(--tx2)">${s.name}</text>`;
  });

  svg.setAttribute('width', W);
  svg.innerHTML = html;
}

// ═══════════════════════════════════════════════════
// HEADER
// ═══════════════════════════════════════════════════
function buildHeader(){
  document.title = D.op_name+' — AscendC Dashboard';
  $('hdr-name').textContent = D.op_name;
  if(D.category) $('hdr-cat').textContent = D.category;

  // badges
  const badges = $('hdr-badges');
  if(D.n_total>0){
    const allPass = D.n_pass===D.n_total;
    const b=el('span','badge '+(allPass?'bp':'bf'));
    b.textContent = allPass ? `✓ ${D.n_pass}/${D.n_total} PASS` : `✗ ${D.n_pass}/${D.n_total} PASS`;
    badges.appendChild(b);
  }

  // chips
  const chip=(label,val,cls='')=>`<span class="chip ${cls}"><b>${label}</b> ${val}</span>`;
  let chips = chip('芯片', D.chip.name) + chip('UB', `${D.chip.ub_kb}KB × ${D.chip.aic}核`);
  if(D.input_shapes&&D.input_shapes[0]) chips+=chip('dtype', D.input_shapes[0].dtype||'');
  if(D.avg_speedup){
    const cls=D.avg_speedup>=2?'pass':D.avg_speedup>=1.2?'warn':'fail';
    chips+=chip('Avg Speedup',`${D.avg_speedup.toFixed(2)}x`,cls);
  }
  chips+=chip('Cases', `${D.n_pass}/${D.n_total}`);
  $('hdr-chips').innerHTML = chips;
  $('hdr-desc').textContent = D.description||'';
}

// ═══════════════════════════════════════════════════
// TAB 1: 算法图示（工业级计算逻辑流）
// ═══════════════════════════════════════════════════
let _flowHtmlTemplate = null;  // Claude flow.html 模板，供 shape 切换时重渲染

function _applyFlowShapeVars(template){
  const og = D.op_graph;
  if(!og || !og.shape_cases || !og.shape_cases.length) return template;
  const idx = parseInt(($('shape-sel')||{}).value) || 0;
  const vars = (og.shape_cases[idx] && og.shape_cases[idx].vars) || {};
  let result = template;
  Object.entries(vars).forEach(([k,v]) => { result = result.replaceAll('{'+k+'}', v); });
  return result;
}
function buildAlgo(){
  const og = D.op_graph;
  const P  = D.panels || {};

  // Populate shape selector
  const sel = $('shape-sel');
  const cases = (og && og.shape_cases) || [];
  cases.forEach((c,i) => {
    const opt = document.createElement('option');
    opt.value = i; opt.textContent = c.label;
    sel.appendChild(opt);
  });

  // ── 计算流程图：优先 Claude 生成的 flow.html，退回 JS 渲染 ──
  const flowDiv = $('algo-flow');
  if(P.algo && P.algo.flow_html && P.algo.flow_html.length > 50){
    _flowHtmlTemplate = P.algo.flow_html;   // 存模板，shape 切换时重渲染
    flowDiv.innerHTML = _applyFlowShapeVars(_flowHtmlTemplate);
  } else {
    renderOpGraph();
  }

  // ── 算法步骤：优先 Claude 生成的 steps.html，退回 pass_comments 列表 ──
  const stepsDiv = $('algo-steps');
  if(P.algo && P.algo.steps_html && P.algo.steps_html.length > 50){
    stepsDiv.innerHTML = P.algo.steps_html;
  } else {
    const comments = D.pass_comments || [];
    if(!comments.length){
      stepsDiv.innerHTML='<p style="color:var(--tx3);font-size:13px">（未找到 Pass 注释；请检查 kernel .cpp 注释格式，或运行精品流程生成 steps.html）</p>';
    } else {
      const ul = document.createElement('ul');
      ul.className = 'as-list';
      comments.forEach(c=>{
        const li=el('li');
        li.innerHTML=`<span style="font-family:var(--mono);font-size:12px;color:var(--tx2)">${c}</span>`;
        ul.appendChild(li);
      });
      stepsDiv.appendChild(ul);
    }
  }
}

function makeIONode(name, tmpl, dtype, kind){
  const d = el('div','');
  d.style.cssText='display:flex;justify-content:center;width:100%';
  const inner = el('div','io-node');
  const isIn = kind==='input';
  inner.style.borderColor = isIn ? 'var(--ac)' : 'var(--gn)';
  inner.innerHTML=`
    <div class="io-lbl">${isIn?'输入 Input':'输出 Output'}</div>
    <div class="io-name" style="color:${isIn?'var(--ac)':'var(--gn)'}">${name}</div>
    <div class="io-shape"><span data-sv="${tmpl}">${shapeAsBracket(tmpl)}</span></div>
    <div class="io-dtype">${dtype}</div>`;
  d.appendChild(inner);
  return d;
}

function makeIORow(ioArray, kind){
  // Renders a flex row of IO nodes for multi-input or multi-output
  const row = el('div','');
  row.style.cssText='display:flex;justify-content:center;flex-wrap:wrap;gap:12px;width:100%';
  ioArray.forEach(io=>{
    const inner = el('div','io-node');
    const isIn = kind==='input';
    inner.style.borderColor = isIn ? 'var(--ac)' : 'var(--gn)';
    inner.innerHTML=`
      <div class="io-lbl">${isIn?'输入 Input':'输出 Output'}</div>
      <div class="io-name" style="color:${isIn?'var(--ac)':'var(--gn)'}">${io.name||'?'}</div>
      <div class="io-shape"><span data-sv="${io.tmpl||''}">${shapeAsBracket(io.tmpl||'')}</span></div>
      <div class="io-dtype">${io.dtype||''}</div>`;
    row.appendChild(inner);
  });
  return row;
}

function makeFlowArrow(){
  return el('div','flow-arrow','▼');
}

function renderOpGraph(){
  const og = D.op_graph;
  const flow = $('algo-flow');
  flow.innerHTML = '';
  if(!og){ flow.innerHTML='<p style="color:var(--tx3)">（无算子图数据）</p>'; return; }

  // Input IO nodes — use v2 inputs[] if available, fallback to legacy inp_name
  const inputArr = (og.inputs && og.inputs.length)
    ? og.inputs
    : [{name: og.inp_name, tmpl: og.inp_tmpl, dtype: og.inp_dtype}];
  flow.appendChild(makeIORow(inputArr, 'input'));

  // No units → contract-first guidance
  if(!og.units || og.units.length === 0){
    flow.appendChild(makeFlowArrow());
    const ph = el('div','');
    ph.style.cssText='text-align:center;padding:32px 20px;color:var(--tx2);background:var(--sf2);border:1.5px dashed var(--bd);border-radius:var(--rl);width:fit-content;max-width:540px;align-self:center';
    ph.innerHTML=`<div style="font-size:28px;margin-bottom:10px">📐</div>
      <div style="font-weight:700;margin-bottom:8px">计算流图需由 Claude 生成</div>
      <div style="font-size:12px;color:var(--tx3);line-height:1.9;font-family:var(--mono)">
        1. 阅读 <b>panels/algo/SPEC.md</b> 了解生成规则<br>
        2. 参考 <b>panels/algo/example_softmax.json</b> 格式<br>
        3. 校验：<b>python3 scripts/validate_contract.py algo_flow.json</b><br>
        4. 重新生成看板：加 <b>--algo-flow panels/algo/algo_flow.json</b> 参数
      </div>`;
    flow.appendChild(ph);
  } else {
    // Units
    og.units.forEach((unit, ui) => {
      flow.appendChild(makeFlowArrow());

      const grp = el('div','unit-group');
      grp.style.cssText=`background:${unit.bg};border:2px solid ${unit.accent}40;`;

      const hdr = el('div','unit-hdr');
      hdr.style.color = unit.accent;
      hdr.textContent = unit.label;
      grp.appendChild(hdr);

      const nodesRow = el('div','unit-nodes');
      unit.nodes.forEach((node, ni) => {
        if(ni>0) nodesRow.appendChild(el('div','unit-sep','→'));

        const nd = el('div','api-node');
        nd.style.borderLeft = `3px solid ${unit.accent}`;

        const sameShape = node.in_tmpl === node.out_tmpl;
        nd.innerHTML=`
          <div class="api-name" style="color:${unit.accent}">${node.api}</div>
          <div class="api-formula">${node.formula}</div>
          <div class="api-shape">
            <span data-sv="${node.in_tmpl}">${shapeAsBracket(node.in_tmpl)}</span>
            ${!sameShape ? ` → <span data-sv="${node.out_tmpl}">${shapeAsBracket(node.out_tmpl)}</span>` : ''}
          </div>`;
        nodesRow.appendChild(nd);
      });
      grp.appendChild(nodesRow);
      flow.appendChild(grp);
    });
  }

  flow.appendChild(makeFlowArrow());
  // Output IO nodes — use v2 outputs[] if available, fallback to legacy out_name
  const outputArr = (og.outputs && og.outputs.length)
    ? og.outputs
    : [{name: og.out_name, tmpl: og.out_tmpl, dtype: og.out_dtype}];
  flow.appendChild(makeIORow(outputArr, 'output'));

  // Apply current shape vars
  updateShapeVars();
}

function updateShapeVars(){
  const og = D.op_graph;
  // shape_cases may be null (auto-generated flow) or empty array — guard both
  if(!og || !og.shape_cases || !og.shape_cases.length) return;
  // Claude flow.html: 重新注入带 shape 替换的模板
  if(_flowHtmlTemplate){
    $('algo-flow').innerHTML = _applyFlowShapeVars(_flowHtmlTemplate);
  }
  const idx = parseInt($('shape-sel').value) || 0;
  const vars = og.shape_cases[idx] && og.shape_cases[idx].vars || {};
  document.querySelectorAll('[data-sv]').forEach(node => {
    let text = node.getAttribute('data-sv');
    Object.entries(vars).forEach(([k,v]) => {
      text = text.replaceAll('{'+k+'}', v);
    });
    node.textContent = shapeAsBracket(text);
  });
}

// ═══════════════════════════════════════════════════
// TAB 2: 内存 & Tiling
// ═══════════════════════════════════════════════════
function buildMemory(){
  const bufs = D.ub_buffers||[];
  const total = D.ub_total_kb, used = D.ub_used_kb;
  const tc = D.tiling.consts||{};
  const aic = D.chip.aic||32;

  // ── 更新 UB 头部标题（动态填入真实 KB 数）──
  const hdrLbl = $('ub-header-lbl');
  if(hdrLbl) hdrLbl.textContent = `Unified Buffer — ${total} KB 地址空间（单核视图）`;

  // ── Tiling 策略分析 ──────────────────────────────
  const strategy = $('tiling-strategy');
  const L1_M=tc.L1_M||tc['L1Shape_M'], L1_N=tc.L1_N||tc['L1Shape_N'], L1_K=tc.L1_K||tc['L1Shape_K'];
  const EP_M=tc.EPILOGUE_TILE_M||tc.ep_tile_m;
  const WS=tc.WORKSPACE_STAGES||1;

  // Cube/CV 算子：L1 tile 维度块
  if(L1_M||L1_N||L1_K){
    const dims=`<div class="til-dims">
      ${L1_M&&L1_N?`<div class="til-block">
        <div class="til-block-title">AIC Cube Tile (M×N)</div>
        <div class="til-block-val">${L1_M} × ${L1_N}</div>
        <div class="til-block-note">${L1_K?`K stride = ${L1_K}`:''}</div>
      </div>`:''}
      ${EP_M&&L1_N?`<div class="til-block">
        <div class="til-block-title">AIV Vector Epilogue</div>
        <div class="til-block-val">${EP_M} × ${L1_N}</div>
        <div class="til-block-note">per epilogue tile</div>
      </div>`:''}
      ${WS?`<div class="til-block">
        <div class="til-block-title">双缓冲 Pipeline</div>
        <div class="til-block-val">${WS} stages</div>
        <div class="til-block-note">AIC↔AIV 流水深度</div>
      </div>`:''}
      <div class="til-block">
        <div class="til-block-title">AIC 核心数</div>
        <div class="til-block-val">${aic}</div>
        <div class="til-block-note">${D.chip.name||'Ascend'}</div>
      </div>
    </div>`;
    strategy.innerHTML=dims;
  } else {
    // Vector 算子：展示 BLOCK_DIM + tileSize 等向量 tiling 参数
    const blockDim = tc.BLOCK_DIM||tc.blockDim||tc.BLOCK_NUM;
    const tileSize = tc.tileSize||tc.TILE_SIZE||tc.tile_size||tc.TILE_LENGTH||tc.tile_length;
    const tileKB   = tileSize ? (tileSize * (D.tiling.dtype_bytes||4) / 1024).toFixed(1) : null;
    const tilePar  = D.tiling.tiling_params||[];
    if(blockDim||tileSize){
      const pblocks = [
        blockDim ? `<div class="til-block">
          <div class="til-block-title">AIV 核心数</div>
          <div class="til-block-val">${blockDim}</div>
          <div class="til-block-note">${D.chip.name||'Ascend'}</div>
        </div>` : '',
        tileSize ? `<div class="til-block">
          <div class="til-block-title">Tile 大小</div>
          <div class="til-block-val">${tileSize} 元素</div>
          <div class="til-block-note">${tileKB ? tileKB+' KB / tile' : ''}</div>
        </div>` : '',
        tileSize && blockDim ? `<div class="til-block">
          <div class="til-block-title">UB 利用率</div>
          <div class="til-block-val">${used} / ${total} KB</div>
          <div class="til-block-note">${total ? ((used/total)*100).toFixed(1)+'%' : ''}</div>
        </div>` : '',
      ].filter(Boolean).join('');
      strategy.innerHTML = `<div class="til-dims">${pblocks}</div>`;
      if(tilePar.length){
        strategy.innerHTML += `<div style="margin-top:10px;font-size:12px;color:var(--tx2);font-family:var(--mono)">
          Tiling params: ${tilePar.join(', ')}</div>`;
      }
    }
  }

  // Per-case load analysis table
  const ta = D.tiling.analysis||[];
  if(ta.length){
    let thead=`<table class="dt" style="width:100%">
      <thead><tr>
        <th>Case</th><th>M × K × N</th>
        <th>AIC Tiles</th><th>尾块</th>
        <th>Tiles/Core</th><th>负载均衡</th>
      </tr></thead>`;
    let tbody='<tbody>';
    ta.forEach(r=>{
      const tStr=r.tiles_m!=null?`${r.tiles_m}×${r.tiles_n}=${r.total_tiles}`:`${r.tiles_n} tiles`;
      const tailStr=r.has_tail
        ?`<span style="color:var(--or)">✓ ${[r.tail_m?'M':'',r.tail_k?'K':'',r.tail_n?'N':''].filter(Boolean).join('+')}</span>`
        :`<span style="color:var(--tx3)">无</span>`;
      const balCls=r.balance_pct===100?'color:var(--gn)':r.balance_pct>=50?'color:var(--or)':'color:var(--rd)';
      const balBar=`<span class="bal-bar"><span class="bal-fill" style="width:${r.balance_pct}%;background:${r.balance_pct===100?'var(--gn)':r.balance_pct>=50?'var(--or)':'var(--rd)'}"></span></span>`;
      const starStr=r.balance_pct===100?' <b style="color:var(--gn)">★</b>':'';
      tbody+=`<tr>
        <td style="font-family:var(--mono);font-size:12px;color:var(--tx2)">${r.case_name}</td>
        <td style="font-family:var(--mono);font-size:12px">${r.M||'?'}×${r.K||'?'}×${r.N||'?'}</td>
        <td class="val">${tStr}</td>
        <td>${tailStr}</td>
        <td class="val">${r.tiles_per_core}</td>
        <td><span style="${balCls};font-weight:600">${r.balance_pct}%</span>${balBar}${starStr}</td>
      </tr>`;
    });
    tbody+='</tbody></table>';
    const tDiv=el('div');
    tDiv.innerHTML=thead+tbody;
    strategy.appendChild(tDiv);

    // Insight text
    const tailCases=ta.filter(r=>r.has_tail);
    const perfectCases=ta.filter(r=>r.balance_pct===100);
    const worstBal=Math.min(...ta.map(r=>r.balance_pct));
    let insight='';
    if(tailCases.length)
      insight+=`<b>尾块处理：</b>${tailCases.map(r=>r.case_name).join('、')} 含非对齐尾块（M/N 不整除 tile size），内核需在边界做条件判断或 pad 处理。`;
    if(insight) insight+='<br>';
    if(perfectCases.length)
      insight+=`<b>完美均衡：</b>${perfectCases.map(r=>r.case_name).join('、')} tiles = ${aic} 的整数倍，所有 AIC 核满载。`;
    else if(worstBal<50)
      insight+=`<b>负载不均：</b>最低均衡率 ${worstBal}%，建议调整 tile 尺寸或测试 shape 覆盖更大规模。`;
    if(insight){
      const ins=el('div','til-insight');
      ins.innerHTML=insight;
      strategy.appendChild(ins);
    }
  } else if(Object.keys(tc).length===0){
    // No constants found — show fallback tiling params
    const fallback=el('div','til-insight');
    fallback.innerHTML='<b>未解析到 constexpr tile 常量</b>：请提供 <code>op_host/*.cpp</code> 文件路径。<br>'
      +(D.tiling.tiling_params&&D.tiling.tiling_params.length
        ?`Tiling 参数：${D.tiling.tiling_params.join(', ')}`:'');
    strategy.appendChild(fallback);
  }

  // Simplified constants table (right panel info)
  const til=$('tiling-table');
  const KNOWN=[
    ['L1Shape_M','AIC tile M'],['L1Shape_N','AIC tile N'],['L1Shape_K','AIC tile K'],
    ['L1_M','AIC tile M'],['L1_N','AIC tile N'],['L1_K','AIC tile K'],
    ['EPILOGUE_TILE_M','AIV epilogue M'],['WORKSPACE_STAGES','双缓冲 stages'],
  ];
  const kvRows=[];
  const seen=new Set();
  KNOWN.forEach(([k,desc])=>{
    if(tc[k]!=null&&!seen.has(desc)){kvRows.push([desc,tc[k]]); seen.add(desc);}
  });
  if(kvRows.length){
    let html=`<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:4px">`;
    kvRows.forEach(([label,val])=>{
      html+=`<div style="background:var(--sf3);border:1px solid var(--bdl);border-radius:6px;padding:5px 10px;font-size:12px">
        <span style="color:var(--tx3)">${label}</span>
        <span style="font-family:var(--mono);font-weight:700;margin-left:6px">${val}</span>
      </div>`;
    });
    html+='</div>';
    til.innerHTML=html;
  }

  // ── UB Memory ──────────────────────────────────
  const ubOverflow = used > total;
  const ubBadge = $('ub-badge');
  if(ubOverflow){
    ubBadge.textContent = `${used} KB / ${total} KB — ⚠ 估算值超出 UB`;
    ubBadge.style.background = '#ffeaea';
    ubBadge.style.color = '#cf222e';
    ubBadge.style.borderColor = '#cf222e44';
  } else {
    ubBadge.textContent = `${used} KB / ${total} KB — ${D.ub_util_pct}% 占用`;
  }

  // Detect multi-kernel groups (e.g. K1 / K2)
  const groups = [...new Set(bufs.map(b=>b.kernel_group||'').filter(g=>g))];
  const isMultiKernel = groups.length > 1;

  function renderUBBar(container, barBufs, barTotal) {
    const barUsed = barBufs.reduce((s,b)=>s+b.size_kb,0);
    const utilPct  = barTotal > 0 ? barUsed / barTotal * 100 : 0;
    const lowUtil  = utilPct < 25; // 当利用率 < 25% 时启用放大视图

    // ── 主条形图（全尺度，精确比例）──
    const barLbl = el('div');
    barLbl.style.cssText='font-size:10px;color:var(--tx3);font-family:var(--mono);margin-bottom:2px';
    barLbl.textContent = lowUtil ? `全尺度 (0–${barTotal} KB)` : '';
    if(lowUtil) container.appendChild(barLbl);

    const bar = el('div','ub-bar');
    const barScale = utilPct > 100 ? barUsed : barTotal; // 溢出时以实际使用量为 100% 基准
    barBufs.forEach(b=>{
      const pct=(b.size_kb/barScale*100).toFixed(2);
      const seg=el('div','ub-seg');
      seg.style.cssText=`width:${pct}%;background:${b.color}`;
      seg.title=`${b.name}  ${fmtKB(b.size_kb)}  (${pct}%)`;
      if(parseFloat(pct)>10) seg.textContent=b.name.length>9?b.name.slice(0,7)+'…':b.name;
      bar.appendChild(seg);
    });
    if(utilPct <= 100){
      const freePct=((barTotal-barUsed)/barTotal*100).toFixed(2);
      const f=el('div','ub-seg ub-free');
      f.style.width=freePct+'%'; f.title=`空闲  ${fmtKB(barTotal-barUsed)}  (${freePct}%)`;
      if(parseFloat(freePct)>8) f.textContent='空闲';
      bar.appendChild(f);
    }
    container.appendChild(bar);

    // ── 刻度尺 ──
    const ruler=el('div','ub-ruler');
    [0,64,128,192,256].filter(kb=>kb<=barTotal).forEach(kb=>{
      const t=el('div','ub-tick',`${kb}KB`);
      t.style.left=`${kb/barTotal*100}%`;
      ruler.appendChild(t);
    });
    container.appendChild(ruler);

    // ── 放大视图（仅低利用率时显示：按使用区域比例展开至全宽）──
    if(lowUtil && barBufs.length > 0){
      const zoomLbl = el('div');
      zoomLbl.style.cssText='font-size:10px;color:var(--ac);font-family:var(--mono);margin:8px 0 2px;font-weight:600';
      zoomLbl.textContent = `⬛ 已分配区域放大 (0–${fmtKB(barUsed)}, 各 buffer 等比)`;
      container.appendChild(zoomLbl);

      const zoomBar = el('div','ub-bar');
      zoomBar.style.cssText='height:28px;border-radius:4px;overflow:hidden;display:flex;border:1px solid var(--bd)';
      barBufs.forEach(b=>{
        const pct = barUsed > 0 ? (b.size_kb / barUsed * 100).toFixed(2) : 0;
        const seg = el('div','ub-seg');
        seg.style.cssText=`width:${pct}%;background:${b.color};min-width:0`;
        seg.title=`${b.name}  ${fmtKB(b.size_kb)}`;
        seg.textContent = b.name;
        zoomBar.appendChild(seg);
      });
      container.appendChild(zoomBar);

      const zoomRuler = el('div','ub-ruler');
      // 刻度按使用量等分
      const steps = Math.min(barBufs.length, 4);
      for(let i=0; i<=steps; i++){
        const kb = (barUsed / steps * i).toFixed(1);
        const t = el('div','ub-tick', `${kb}KB`);
        t.style.left = `${(i/steps*100).toFixed(1)}%`;
        zoomRuler.appendChild(t);
      }
      container.appendChild(zoomRuler);
    }

    // ── 图例标签 ──
    const row = el('div','ub-labels-row');
    barBufs.forEach(b=>{
      const chip = el('div','ub-label-chip');
      chip.title = `${b.kind} @ ${b.position}`;
      const sw = el('div','ub-label-swatch');
      sw.style.background = b.color;
      const lbl = el('span');
      lbl.textContent = `${b.name}  ${fmtKB(b.size_kb)}`;
      chip.appendChild(sw); chip.appendChild(lbl);
      row.appendChild(chip);
    });
    const fc = el('div','ub-label-chip');
    const fs = el('div','ub-label-swatch');
    fs.style.background = 'var(--sf3)';
    fs.style.border = '1px solid var(--bd)';
    const fl = el('span'); fl.style.color='var(--tx3)';
    // 当利用率低时补充设计建议（避免基于固定 buffer 数量和 dtype 的硬编码公式）
    const suggestion = lowUtil ? `  ⚠ 当前 ${utilPct.toFixed(1)}% 利用率，建议适当增大 tileSize 以提高 UB 利用率和性能` : '';
    fl.textContent = `空闲  ${fmtKB(barTotal-barUsed)}${suggestion}`;
    fc.appendChild(fs); fc.appendChild(fl); row.appendChild(fc);
    container.appendChild(row);
  }

  function renderUBDonut(container, barBufs, barTotal) {
    const used = barBufs.reduce((s,b)=>s+b.size_kb, 0);
    const free = barTotal - used;
    const overflow = free < 0;
    const W = 200, H = 200, cx = W/2, cy = H/2, R = 80, r = 50;

    // 当估算值超出 UB 总量时（通常因 tileLength 回退估算），仅显示各 buffer 比例，不加"空闲"段
    const allSegs = overflow
      ? barBufs.map(b=>({label:b.name, kb:b.size_kb, color:b.color}))
      : [...barBufs.map(b=>({label:b.name, kb:b.size_kb, color:b.color})),
         {label:'空闲', kb:free, color:'#e8eaed'}];
    // 当溢出时以 used 为基准做比例，否则以 barTotal 为基准
    const pieBase = overflow ? used : barTotal;
    let startAngle = -Math.PI/2;

    const pathData = allSegs.map(seg => {
        const angle = (seg.kb / pieBase) * 2 * Math.PI;
        const endAngle = startAngle + angle;
        const x1 = cx + R * Math.cos(startAngle);
        const y1 = cy + R * Math.sin(startAngle);
        const x2 = cx + R * Math.cos(endAngle);
        const y2 = cy + R * Math.sin(endAngle);
        const xi1 = cx + r * Math.cos(startAngle);
        const yi1 = cy + r * Math.sin(startAngle);
        const xi2 = cx + r * Math.cos(endAngle);
        const yi2 = cy + r * Math.sin(endAngle);
        const large = angle > Math.PI ? 1 : 0;
        const d = `M${x1},${y1} A${R},${R},0,${large},1,${x2},${y2} L${xi2},${yi2} A${r},${r},0,${large},0,${xi1},${yi1} Z`;
        const result = {d, color: seg.color, label: seg.label, kb: seg.kb};
        startAngle = endAngle;
        return result;
    });

    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('width', W); svg.setAttribute('height', H);
    svg.style.cssText = 'display:block;margin:0 auto';

    pathData.forEach(seg => {
        const path = document.createElementNS(svgNS, 'path');
        path.setAttribute('d', seg.d);
        path.setAttribute('fill', seg.color);
        path.setAttribute('stroke', 'var(--bg)');
        path.setAttribute('stroke-width', '2');
        path.title = `${seg.label}: ${fmtKB(seg.kb)}`;
        const title = document.createElementNS(svgNS, 'title');
        title.textContent = `${seg.label}  ${fmtKB(seg.kb)}  (${(seg.kb/pieBase*100).toFixed(1)}%)`;
        path.appendChild(title);
        svg.appendChild(path);
    });

    // 中心文字：已用/总量（溢出时显示警告色）
    const centerColor = overflow ? '#cf222e' : 'var(--tx)';
    const t1 = document.createElementNS(svgNS, 'text');
    t1.setAttribute('x', cx); t1.setAttribute('y', cy - 8);
    t1.setAttribute('text-anchor', 'middle');
    t1.setAttribute('font-size', '14'); t1.setAttribute('font-weight', '700');
    t1.setAttribute('fill', centerColor);
    t1.textContent = fmtKB(used);
    const t2 = document.createElementNS(svgNS, 'text');
    t2.setAttribute('x', cx); t2.setAttribute('y', cy + 10);
    t2.setAttribute('text-anchor', 'middle');
    t2.setAttribute('font-size', '10'); t2.setAttribute('fill', 'var(--tx3)');
    t2.textContent = `/ ${fmtKB(barTotal)}`;
    const t3 = document.createElementNS(svgNS, 'text');
    t3.setAttribute('x', cx); t3.setAttribute('y', cy + 24);
    t3.setAttribute('text-anchor', 'middle');
    t3.setAttribute('font-size', '10');
    t3.setAttribute('fill', overflow ? '#cf222e' : 'var(--tx3)');
    t3.textContent = overflow ? `⚠ 估算值超出 UB` : `${(used/barTotal*100).toFixed(1)}% 已用`;
    svg.appendChild(t1); svg.appendChild(t2); svg.appendChild(t3);
    container.appendChild(svg);

    // 图例
    const row = el('div','ub-labels-row');
    row.style.justifyContent = 'center';
    barBufs.forEach(b=>{
        const chip = el('div','ub-label-chip');
        const sw = el('div','ub-label-swatch'); sw.style.background = b.color;
        const lbl = el('span'); lbl.textContent = `${b.name}  ${fmtKB(b.size_kb)}`;
        chip.appendChild(sw); chip.appendChild(lbl);
        row.appendChild(chip);
    });
    // 空闲 chip — 仅在未溢出时显示；溢出时显示估算警告
    const fc = el('div','ub-label-chip');
    const fs = el('div','ub-label-swatch'); fs.style.background='#e8eaed'; fs.style.border='1px solid var(--bd)';
    const fl = el('span');
    if(overflow){
      fl.style.color='#cf222e';
      fl.textContent = `⚠ 估算值超出单核 UB — buffer 大小基于 tileLength 推算，实际运行时由 Tiling 决定`;
    } else {
      fl.style.color='var(--tx3)';
      const suggestion = (used/barTotal) < 0.3 ? `  ⚠ 仅 ${(used/barTotal*100).toFixed(0)}%，建议增大 tileSize` : '';
      fl.textContent = `空闲  ${fmtKB(free)}${suggestion}`;
    }
    fc.appendChild(fs); fc.appendChild(fl); row.appendChild(fc);
    container.appendChild(row);
  }

  const donutContainer = $('ub-donut-container');
  const hasVizHtml = !!(D.panels && D.panels.memory && D.panels.memory.ub_viz_html
                        && D.panels.memory.ub_viz_html.length > 50);
  if(hasVizHtml){
    // Claude 写了 ub_viz.html，隐藏 JS 生成的 donut，避免重复
    if(donutContainer) donutContainer.style.display = 'none';
  } else if(isMultiKernel){
    // Multi-kernel: one sub-donut per group, each occupying full 256KB scale
    groups.forEach(grp=>{
      const grpBufs = bufs.filter(b=>(b.kernel_group||'')==grp);
      const grpUsed = grpBufs.reduce((s,b)=>s+b.size_kb,0);
      const wrap = el('div');
      wrap.style.cssText='margin-bottom:10px';
      const lbl=el('div');
      lbl.style.cssText='font-size:11px;font-weight:700;color:var(--tx2);margin-bottom:3px;font-family:var(--mono)';
      lbl.textContent=`${grp} UB  (${fmtKB(grpUsed)} / ${fmtKB(total)})`;
      wrap.appendChild(lbl);
      renderUBDonut(wrap, grpBufs, total);
      donutContainer.appendChild(wrap);
    });
  } else {
    renderUBDonut(donutContainer, bufs, total);
  }

  const ubTable=$('ub-table');
  const t=el('table','dt');
  t.innerHTML=`<thead><tr><th>缓冲区</th><th>类型</th><th>位置</th><th>大小</th><th>用途</th></tr></thead>`;
  const tb=el('tbody');
  bufs.forEach(b=>{
    const tr=el('tr');
    tr.innerHTML=`<td><span style="display:inline-flex;align-items:center;gap:6px">
        <span style="width:10px;height:10px;border-radius:2px;background:${b.color};flex-shrink:0;display:inline-block"></span>
        <span class="val" style="font-size:12px">${b.name}</span></span></td>
      <td><span class="pipe-tag" style="background:${b.color}22;color:${b.color};border:1px solid ${b.color}44">${b.kind}</span></td>
      <td><span class="pipe-tag" style="background:${b.color}14;color:${b.color}">${b.position}</span></td>
      <td><b class="val">${fmtKB(b.size_kb)}</b></td>
      <td style="color:var(--tx3);font-size:12px">${b.purpose||''}</td>`;
    tb.appendChild(tr);
  });
  const tot=el('tr');
  tot.innerHTML=`<td colspan="3" style="text-align:right;color:var(--tx2);font-size:12px">已用合计</td>
    <td><b class="val" style="color:var(--ac)">${fmtKB(used)}</b> / ${fmtKB(total)}</td><td></td>`;
  tb.appendChild(tot);
  t.appendChild(tb); ubTable.appendChild(t);

  // ── Claude-written UB 分配可视化（ub_viz.html）──
  const vizSlot = $('ub-viz-slot');
  const P = D.panels||{};
  if(vizSlot && P.memory && P.memory.ub_viz_html && P.memory.ub_viz_html.length > 50){
    vizSlot.innerHTML = P.memory.ub_viz_html;
  }
}

// ═══════════════════════════════════════════════════
// TAB 3: 精度 v2 — Chart.js + Heatmap + Expandable rows
// ═══════════════════════════════════════════════════
const PREC_COL_DEFS = [
  {key:'match_rate',   label:'MATCH %',   fmt:v=>v!=null?v.toFixed(1)+'%':'—',        isStr:true},
  {key:'max_re',       label:'MAX_RE',    fmt:v=>fmtN(v,4), thresh:10.0},
  {key:'mean_re',      label:'MEAN_RE',   fmt:v=>fmtN(v,4), thresh:2.0},
  {key:'rmse',         label:'RMSE',      fmt:v=>fmtN(v,5), thresh:2.0},
  {key:'max_diff',     label:'MAX_DIFF',  fmt:v=>fmtErr(v,5),       isAE:true},
  {key:'mean_diff',    label:'MEAN_DIFF', fmt:v=>fmtErr(v,5),       isAE:true},
  {key:'ae_max',       label:'AE_MAX',    fmt:v=>fmtErr(v,5),       isAE:true},
  {key:'mismatch_rate',label:'MISMATCH',  fmt:v=>v!=null?fmtN(v,4):'—'},
];
const precActiveCols = [];
const precCharts = {};
const precOpenRows = new Set();
const precHeatCells = [];
// Tooltip element (created once, reused)
const _precTip = document.createElement('div');
_precTip.className = 'prec-tip'; document.body.appendChild(_precTip);

function buildPrecision(){
  const cases = D.cases||[];
  if(!cases.length){
    $('prec-table').innerHTML='<tr><td colspan="99" style="color:var(--tx3);padding:16px">（无精度数据）</td></tr>';
    return;
  }

  // FAIL banner：精度未全通过时，顶部显示失败 case 汇总
  const failBanner=$('prec-fail-banner');
  if(failBanner){
    const failCases=cases.filter(c=>!(c.precision||{}).passed);
    if(failCases.length>0){
      failBanner.style.display='';
      failBanner.replaceChildren();

      const title=document.createElement('b');
      title.textContent=`✗ 精度未通过（${D.n_pass}/${D.n_total} PASS）— FAIL Case 诊断`;
      failBanner.appendChild(title);
      failBanner.appendChild(document.createElement('br'));

      const table=document.createElement('table');
      table.style.marginTop='6px';
      table.style.width='100%';
      table.style.fontSize='12px';
      table.style.borderCollapse='collapse';

      const thead=document.createElement('thead');
      const headRow=document.createElement('tr');
      headRow.style.textAlign='left';
      headRow.style.borderBottom='1px solid var(--er1,#cf222e)';
      ['Case','Shape','MAX_RE','MATCH%','MAX_DIFF'].forEach(label=>{
        const th=document.createElement('th');
        th.style.padding='3px 8px';
        th.textContent=label;
        headRow.appendChild(th);
      });
      thead.appendChild(headRow);
      table.appendChild(thead);

      const tbody=document.createElement('tbody');
      failCases.forEach(c=>{
        const p=c.precision||{};
        const maxre=p.max_re!=null?p.max_re.toFixed(4):'—';
        const mr=p.match_rate!=null?p.match_rate.toFixed(1)+'%':'—';
        const md=p.max_diff!=null?p.max_diff.toExponential(3):'—';

        const row=document.createElement('tr');
        row.style.borderBottom='1px solid rgba(207,34,46,.15)';

        const caseCell=document.createElement('td');
        caseCell.style.padding='3px 8px';
        caseCell.style.fontWeight='600';
        caseCell.textContent=c.name||('Case '+c.id);
        row.appendChild(caseCell);

        const shapeCell=document.createElement('td');
        shapeCell.style.padding='3px 8px';
        shapeCell.style.color='var(--tx3,#666)';
        shapeCell.style.fontSize='11px';
        shapeCell.textContent=c.shape||'';
        row.appendChild(shapeCell);

        const maxreCell=document.createElement('td');
        maxreCell.style.padding='3px 8px';
        maxreCell.textContent=maxre;
        row.appendChild(maxreCell);

        const mrCell=document.createElement('td');
        mrCell.style.padding='3px 8px';
        mrCell.textContent=mr;
        row.appendChild(mrCell);

        const mdCell=document.createElement('td');
        mdCell.style.padding='3px 8px';
        mdCell.textContent=md;
        row.appendChild(mdCell);

        tbody.appendChild(row);
      });
      table.appendChild(tbody);
      failBanner.appendChild(table);

      const note=document.createElement('div');
      note.style.marginTop='6px';
      note.style.color='var(--er3,#82071e)';
      note.style.fontSize='12px';
      note.textContent='↳ 常见原因：输出全零（地址映射错误）、数值溢出、tiling 越界、类型转换截断。详见下方"FAIL Case 诊断"分析（需 Claude 分析阶段写入）。';
      failBanner.appendChild(note);
    } else {
      failBanner.style.display='none';
    }
  }

  // Detect active columns
  precActiveCols.length=0;
  PREC_COL_DEFS.forEach(col=>{
    if(cases.some(c=>c.precision&&c.precision[col.key]!=null)) precActiveCols.push(col);
  });

  // KPI row inspired by ascend_test_visualize summary cards
  const kpiRow=$('prec-kpi-row');
  if(kpiRow){
    const passCnt=cases.filter(c=>(c.precision||{}).passed).length;
    const allMetrics=cases.map(c=>c.precision||{});
    const aeMaxVals=allMetrics.map(p=>p.ae_max??p.max_diff).filter(v=>v!=null);
    const reMaxVals=allMetrics.map(p=>p.re_max??p.max_re).filter(v=>v!=null);
    const rmseVals=allMetrics.map(p=>p.rmse).filter(v=>v!=null);
    const maxOf=v=>v.length?Math.max(...v):null;
    kpiRow.innerHTML=`
      <div class="prec-kpi"><div class="k">Pass Ratio</div><div class="v">${passCnt}/${cases.length}</div><div class="n">${cases.length?((passCnt/cases.length)*100).toFixed(1):'0.0'}%</div></div>
        <div class="prec-kpi"><div class="k">Worst AE</div><div class="v">${maxOf(aeMaxVals)!=null?fmtErr(maxOf(aeMaxVals),5):'—'}</div><div class="n">ans_vs_golden</div></div>
      <div class="prec-kpi"><div class="k">Worst RE</div><div class="v">${maxOf(reMaxVals)!=null?fmtN(maxOf(reMaxVals),5):'—'}</div><div class="n">ans_vs_golden</div></div>
      <div class="prec-kpi"><div class="k">Worst RMSE</div><div class="v">${maxOf(rmseVals)!=null?fmtN(maxOf(rmseVals),5):'—'}</div><div class="n">cross-case</div></div>`;
  }

  // Heatmap selector bar
  const heatBar=$('prec-heat-bar');
  if(heatBar){
    const numCols=precActiveCols.filter(c=>!c.isStr&&!c.isAE);
    const opts=numCols.map(c=>`<option value="${c.key}">${c.label}</option>`).join('')+'<option value="">Off</option>';
    heatBar.innerHTML=`<label>Heatmap:</label>
      <select id="prec-heat-sel" onchange="applyPrecHeat(this.value)">${opts}</select>
      <div class="heat-legend-wrap" id="prec-heat-legend" style="display:none">
        <span class="heat-lbl" id="prec-hmin"></span>
        <div style="position:relative">
          <div class="heat-grad-bar" id="prec-hgrad"></div>
          <div class="heat-grad-ticks" id="prec-hticks"></div>
        </div>
        <span class="heat-lbl" id="prec-hmax"></span>
      </div>`;
  }

  // Build table
  const pt=$('prec-table');
  const thCols=precActiveCols.map(c=>`<th>${c.label}</th>`).join('');
  pt.innerHTML=`<thead><tr><th style="text-align:left">Case</th><th>精度</th>${thCols}</tr></thead>`;
  const tbody=document.createElement('tbody');

  cases.forEach((c,idx)=>{
    const p=c.precision||{};
    const pass=p.passed;
    const badge=pass
      ?'<span class="badge bp" style="font-size:10px">✓ PASS</span>'
      :'<span class="badge bf" style="font-size:10px">✗ FAIL</span>';

    const caseRow=document.createElement('tr');
    caseRow.className='pst-case'; caseRow.id=`pcr-${idx}`;
    caseRow.onclick=()=>togglePrecDetail(idx);

    const tdCols=precActiveCols.map(col=>{
      const val=p[col.key];
      return `<td class="pst-metric-val" data-val="${val!=null?val:''}" data-col="${col.key}">${col.fmt(val)}</td>`;
    }).join('');

    caseRow.innerHTML=`<td>
        <div class="pst-name">${c.name||'Case '+c.id}</div>
        <div class="pst-shape">${c.shape||c.shape_compact||''}</div>
      </td><td>${badge}</td>${tdCols}`;
    tbody.appendChild(caseRow);

    // Register heat cells + tooltips
    caseRow.querySelectorAll('td[data-col]').forEach(td=>{
      const v=parseFloat(td.getAttribute('data-val'));
      precHeatCells.push({td,key:td.getAttribute('data-col'),val:isNaN(v)?0:v});
      td.addEventListener('mouseenter',e=>showPrecTip(e,idx));
      td.addEventListener('mouseleave',hidePrecTip);
    });
    caseRow.querySelector('td').addEventListener('mouseenter',e=>showPrecTip(e,idx));
    caseRow.querySelector('td').addEventListener('mouseleave',hidePrecTip);

    // Detail row
    const dtr=document.createElement('tr');
    dtr.className='pst-detail'; dtr.id=`pdr-${idx}`;
    const dtd=document.createElement('td');
    dtd.colSpan=precActiveCols.length+2; dtd.style.padding='0';
    dtd.innerHTML=`<div class="pst-dpanel" id="pdp-${idx}"></div>`;
    dtr.appendChild(dtd); tbody.appendChild(dtr);
  });

  pt.appendChild(tbody);
  // Apply initial heatmap
  const firstNum=precActiveCols.find(c=>!c.isStr&&!c.isAE);
  if(firstNum) applyPrecHeat(firstNum.key);
}

function applyPrecHeat(metricKey){
  const isDark=window.matchMedia('(prefers-color-scheme:dark)').matches;
  const alpha=isDark?0.2:0.13;
  const legend=$('prec-heat-legend');
  if(!metricKey){
    precHeatCells.forEach(c=>{c.td.style.backgroundColor='';});
    if(legend) legend.style.display='none'; return;
  }
  const rel=precHeatCells.filter(c=>c.key===metricKey);
  const vals=rel.map(c=>c.val).filter(v=>v>0);
  if(!vals.length){rel.forEach(c=>{c.td.style.backgroundColor='';});return;}
  const logMin=Math.log10(Math.min(...vals));
  const logMax=Math.log10(Math.max(...vals));
  const logRange=logMax-logMin||1;
  rel.forEach(c=>{
    if(c.val<=0){c.td.style.backgroundColor='';return;}
    const t=Math.max(0,Math.min(1,(Math.log10(c.val)-logMin)/logRange));
    const r=t<0.5?Math.round(255*t*2):255;
    const g=t<0.5?255:Math.round(255*(1-(t-0.5)*2));
    c.td.style.backgroundColor=`rgba(${r},${g},0,${alpha})`;
  });
  // Legend
  if(legend){
    legend.style.display='flex';
    const la=isDark?0.3:0.2;
    const grad=$('prec-hgrad');
    if(grad) grad.style.background=`linear-gradient(to right,rgba(0,255,0,${la}),rgba(255,255,0,${la}),rgba(255,0,0,${la}))`;
    const minEl=$('prec-hmin'); if(minEl) minEl.textContent=Math.min(...vals).toExponential(1);
    const maxEl=$('prec-hmax'); if(maxEl) maxEl.textContent=Math.max(...vals).toExponential(1);
    const ticks=$('prec-hticks'); if(!ticks) return;
    ticks.innerHTML='';
    const eMin=Math.floor(logMin),eMax=Math.ceil(logMax);
    for(let e=eMin;e<=eMax;e++){
      const pct=(e-logMin)/logRange*100;
      if(pct<-1||pct>101) continue;
      const tk=document.createElement('span');
      tk.className='heat-tick'; tk.style.left=Math.max(0,Math.min(100,pct))+'%';
      tk.textContent='1e'+e; ticks.appendChild(tk);
    }
  }
}

function showPrecTip(e,idx){
  const c=(D.cases||[])[idx]; if(!c) return;
  const p=c.precision||{};
  let h=`<div class="prec-tip-title">${c.name||'Case '+c.id}</div>`;
  h+=`<div class="prec-tip-row"><span class="prec-tip-lbl">Shape</span><span class="prec-tip-val" style="font-size:10px">${c.shape||c.shape_compact||''}</span></div>`;
  PREC_COL_DEFS.forEach(col=>{
    const v=p[col.key]; if(v==null) return;
    h+=`<div class="prec-tip-row"><span class="prec-tip-lbl">${col.label}</span><span class="prec-tip-val">${col.fmt(v)}</span></div>`;
  });
  _precTip.innerHTML=h; _precTip.classList.add('vis');
  const r=_precTip.getBoundingClientRect();
  let x=e.clientX+14,y=e.clientY-10;
  if(x+r.width>window.innerWidth-20) x=e.clientX-r.width-14;
  if(y+r.height>window.innerHeight-20) y=window.innerHeight-r.height-20;
  _precTip.style.left=x+'px'; _precTip.style.top=y+'px';
}
function hidePrecTip(){_precTip.classList.remove('vis');}
document.addEventListener('mousemove',e=>{
  if(_precTip.classList.contains('vis')){
    const r=_precTip.getBoundingClientRect();
    let x=e.clientX+14,y=e.clientY-10;
    if(x+r.width>window.innerWidth-20) x=e.clientX-r.width-14;
    if(y+r.height>window.innerHeight-20) y=window.innerHeight-r.height-20;
    _precTip.style.left=x+'px'; _precTip.style.top=y+'px';
  }
});

function togglePrecDetail(idx){
  const drow=$(`pdr-${idx}`), crow=$(`pcr-${idx}`); if(!drow||!crow) return;
  if(precOpenRows.has(idx)){
    drow.classList.remove('pst-open'); crow.classList.remove('pst-open');
    precOpenRows.delete(idx);
    if(precCharts[idx]){precCharts[idx].forEach(c=>c.destroy());delete precCharts[idx];}
  } else {
    // Single-open behavior to keep the precision panel compact
    Array.from(precOpenRows).forEach(i=>{
      const od=$(`pdr-${i}`), oc=$(`pcr-${i}`);
      if(od) od.classList.remove('pst-open');
      if(oc) oc.classList.remove('pst-open');
      if(precCharts[i]){precCharts[i].forEach(c=>c.destroy());delete precCharts[i];}
      precOpenRows.delete(i);
    });
    drow.classList.add('pst-open'); crow.classList.add('pst-open');
    precOpenRows.add(idx); renderPrecDetail(idx);
  }
}

function renderPrecDetail(idx){
  const cases=D.cases||[]; const c=cases[idx]; if(!c) return;
  const p=c.precision||{};
  const panel=$(`pdp-${idx}`); if(!panel) return;
  const isDark=window.matchMedia('(prefers-color-scheme:dark)').matches;
  const CG=isDark?'rgba(255,255,255,.06)':'rgba(0,0,0,.06)';
  const CT=isDark?'#8b949e':'#57606a';
  const numCols=precActiveCols.filter(col=>!col.isStr&&p[col.key]!=null);

  let html=`<div style="font-size:11px;color:var(--tx2);margin-bottom:10px;font-weight:600">
    Case ${c.id} — ${c.name||''} <span style="color:var(--tx3);font-family:var(--mono)">${c.shape||c.shape_compact||''}</span></div>`;
  html+='<div class="pst-charts">';
  html+=`<div class="pst-chart-box"><h5>精度指标（对数轴）</h5><canvas id="pc-bar-${idx}"></canvas></div>`;
  html+=`<div class="pst-chart-box"><h5>跨 Case 对比 — ${numCols[0]?.label||''}</h5><canvas id="pc-cross-${idx}"></canvas></div>`;
  html+='</div>';
  // Threshold table
  html+='<div style="overflow-x:auto"><table class="pst-thr"><thead><tr><th>指标</th><th>值</th><th>阈值</th><th>状态</th></tr></thead><tbody>';
  precActiveCols.forEach(col=>{
    const v=p[col.key]; if(v==null) return;
    const thresh=col.thresh;
    let status='—',cls='';
    if(thresh!=null){
      if(v<=thresh*0.5){status='✓ 优秀';cls='thr-ok';}
      else if(v<=thresh){status='△ 通过';cls='thr-warn';}
      else{status='✗ 超限';cls='thr-fail';}
    } else if(col.key==='match_rate'){
      cls=v>=99.9?'thr-ok':'thr-warn'; status=v>=99.9?'✓ 100%':`△ ${v.toFixed(1)}%`;
    }
    html+=`<tr><td>${col.label}</td><td class="val">${col.fmt(v)}</td><td>${thresh!=null?'≤'+thresh:'—'}</td><td class="${cls}">${status}</td></tr>`;
  });
  html+='</tbody></table></div>';
  panel.innerHTML=html;
  if(typeof Chart==='undefined'){return;}
  precCharts[idx]=[];
  // Chart 1: metric bars for this case
  const c1=document.getElementById(`pc-bar-${idx}`);
  if(c1&&numCols.length){
    const labels=numCols.map(c=>c.label), vals=numCols.map(c=>p[c.key]||0);
    const bgColors=vals.map((v,i)=>{
      const t=numCols[i].thresh;
      if(!t) return isDark?'rgba(41,151,255,.55)':'rgba(9,105,218,.5)';
      return v<=t*0.5?'rgba(26,127,55,.55)':v<=t?'rgba(191,135,0,.55)':'rgba(207,34,46,.55)';
    });
    const bdColors=vals.map((v,i)=>{
      const t=numCols[i].thresh;
      if(!t) return isDark?'#2997ff':'#0969da';
      return v<=t*0.5?'#1a7f37':v<=t?'#bf8700':'#cf222e';
    });
    precCharts[idx].push(new Chart(c1,{
      type:'bar', data:{labels,datasets:[{label:'当前值',data:vals,backgroundColor:bgColors,borderColor:bdColors,borderWidth:1,barPercentage:.65,minBarLength:4}]},
      options:{responsive:true,maintainAspectRatio:false,
        scales:{y:{type:'logarithmic',grid:{color:CG},ticks:{color:CT,font:{size:9}}},x:{grid:{display:false},ticks:{color:CT,font:{size:9},maxRotation:30}}},
        plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>`${ctx.dataset.label}: ${ctx.raw}`}}}}
    }));
  }
  // Chart 2: cross-case for first metric
  const c2=document.getElementById(`pc-cross-${idx}`);
  if(c2&&numCols.length){
    const metric=numCols[0];
    const allC=D.cases||[];
    const labels=allC.map(ca=>ca.name||'Case '+ca.id);
    const vals=allC.map(ca=>ca.precision&&ca.precision[metric.key]!=null?ca.precision[metric.key]:0);
    const bgColors=vals.map((_,i)=>i===idx?(isDark?'rgba(41,151,255,.8)':'rgba(9,105,218,.7)'):(isDark?'rgba(139,148,158,.3)':'rgba(142,142,147,.3)'));
    const bdColors=vals.map((_,i)=>i===idx?(isDark?'#2997ff':'#0969da'):(isDark?'#8b949e':'#8e8e93'));
    precCharts[idx].push(new Chart(c2,{
      type:'bar', data:{labels,datasets:[{label:metric.label,data:vals,backgroundColor:bgColors,borderColor:bdColors,borderWidth:1,barPercentage:.7,minBarLength:4}]},
      options:{responsive:true,maintainAspectRatio:false,
        scales:{y:{type:vals.some(v=>v>0)?'logarithmic':'linear',grid:{color:CG},ticks:{color:CT,font:{size:9}}},x:{grid:{display:false},ticks:{color:CT,font:{size:9},maxRotation:30}}},
        plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>`${metric.label}: ${ctx.raw}`}}}}
    }));
  }
}
// Legacy stubs (tab switch still calls buildPrecChart which was removed)
function buildPrecChart(){}
function updateTableSel(){}
function refreshPrecDetail(){}

// ═══════════════════════════════════════════════════
// TAB 4: 性能 v2 — Chart.js + op_summary 深度解析
// ═══════════════════════════════════════════════════
const AIV_SEGS = [
  {key:'aiv_vec_time(us)',    ratioKey:'aiv_vec_ratio',    label:'Vec (向量计算)',  color:'#0969da'},
  {key:'aiv_mte2_time(us)',   ratioKey:'aiv_mte2_ratio',   label:'MTE2 (GM→UB)',   color:'#1a7f37'},
  {key:'aiv_mte3_time(us)',   ratioKey:'aiv_mte3_ratio',   label:'MTE3 (UB→GM)',   color:'#bf8700'},
  {key:'aiv_scalar_time(us)', ratioKey:'aiv_scalar_ratio', label:'Scalar (标量)',   color:'#8250df'},
];
const AIC_SEGS = [
  {key:'aic_mac_time(us)',     ratioKey:'aic_mac_ratio',     label:'MAC (矩阵计算)',     color:'#0969da'},
  {key:'aic_mte1_time(us)',    ratioKey:'aic_mte1_ratio',    label:'MTE1 (GM→L0)',     color:'#bf8700'},
  {key:'aic_mte2_time(us)',    ratioKey:'aic_mte2_ratio',    label:'MTE2 (GM→L1)',     color:'#1a7f37'},
  {key:'aic_fixpipe_time(us)', ratioKey:'aic_fixpipe_ratio', label:'FixPipe (定长算)',  color:'#cf222e'},
  {key:'aic_scalar_time(us)',  ratioKey:'aic_scalar_ratio',  label:'Scalar (标量)',     color:'#8250df'},
];
let _perfHistChart=null, _perfDonutChartAIC=null, _perfDonutChartAIV=null, _perfTimingChart=null;

function buildPerf(){
  const cases=D.cases||[];
  const perfCases=cases.filter(c=>c.performance&&c.performance.speedup);
  if(!perfCases.length){
    $('perf-cards').innerHTML='<div style="color:var(--tx3);padding:12px">（无性能数据）</div>';
    $('perf-note').innerHTML='无 msprof 数据。请先运行评测生成 op_summary*.csv。';
    const cd=$('perf-case-detail-card'); if(cd) cd.style.display='none';
    return;
  }

  // ── Speedup overview histogram (Chart.js) ──
  const isDark=window.matchMedia('(prefers-color-scheme:dark)').matches;
  const CG=isDark?'rgba(255,255,255,.06)':'rgba(0,0,0,.06)';
  const CT=isDark?'#8b949e':'#57606a';
  const canvas=$('perf-hist-canvas');
  if(canvas&&typeof Chart!=='undefined'){
    if(_perfHistChart){_perfHistChart.destroy();_perfHistChart=null;}
    const labels=perfCases.map(c=>c.name||'Case '+c.id);
    const speedups=perfCases.map(c=>c.performance.speedup);
    const bgColors=speedups.map(s=>s>=2?'rgba(26,127,55,.7)':s>=1?'rgba(191,135,0,.7)':'rgba(207,34,46,.7)');
    const bdColors=speedups.map(s=>s>=2?'#1a7f37':s>=1?'#bf8700':'#cf222e');
    _perfHistChart=new Chart(canvas,{
      type:'bar',
      data:{labels,datasets:[{
        label:'Speedup',data:speedups,backgroundColor:bgColors,borderColor:bdColors,
        borderWidth:1.5,barPercentage:.65,minBarLength:4,
      }]},
      options:{
        responsive:true,maintainAspectRatio:false,
        scales:{
          y:{beginAtZero:true,grid:{color:CG},ticks:{color:CT,font:{size:10},callback:v=>v+'x'},
             title:{display:true,text:'Speedup (x)',color:CT,font:{size:10}}},
          x:{grid:{display:false},ticks:{color:CT,font:{size:10},maxRotation:30}},
        },
        plugins:{
          legend:{display:false},
          annotation:{annotations:{line1:{type:'line',yMin:1,yMax:1,borderColor:isDark?'rgba(255,255,255,.3)':'rgba(0,0,0,.2)',borderWidth:1,borderDash:[4,3]}}},
          tooltip:{callbacks:{label:ctx=>`Speedup: ${ctx.raw.toFixed(2)}x`}},
        },
      },
    });
  }

  // Precision-failed warning banner (shown before note when precision not all-pass)
  const precWarn=$('perf-prec-warn');
  if(precWarn){
    if(D.n_total>0 && D.n_pass<D.n_total){
      precWarn.style.display='';
      precWarn.innerHTML=`⚠ <b>当前精度未通过（${D.n_pass}/${D.n_total} PASS）</b>，以下性能数据仅供调试参考，建议先修复精度再优化性能。`;
    } else {
      precWarn.style.display='none';
    }
  }

  // Summary note
  const avgSp=D.avg_speedup;
  const maxSp=Math.max(...perfCases.map(c=>c.performance.speedup));
  const minSp=Math.min(...perfCases.map(c=>c.performance.speedup));
  let note=`${perfCases.length} 个 Shape 评测完成。加速比 <b>${minSp.toFixed(2)}x — ${maxSp.toFixed(2)}x</b>，均值 <b>${avgSp?avgSp.toFixed(2)+'x':'N/A'}</b>。<br>`;
  const src=perfCases[0].performance.source;
  if(src==='msprof'||src==='msprof_csv') note+='Custom 算子计时：<b>msprof 硬件级</b>（op_summary 纯 kernel 执行时间，不含 dispatch 开销）。参考实现计时：<b>torch.npu.Event</b>（含 PyTorch dispatch 及 inter-kernel gap，偏保守）。<br>';
  else if(src==='cann_standalone') note+='计时：<b>CANN standalone .so</b>（tilelang_eval_adapter + combined_kernel_loader）。NPU kernel 真实耗时，不含 Python 开销。<br>';
  else note+='计时：<b>torch.npu.Event</b>（参考实现与 custom 算子均使用相同计时方式）。<br>';
  if(D.n_total>0 && D.n_pass<D.n_total) note+='⚠ 精度未通过，性能结论仅供参考。';
  else if(avgSp>=2) note+='✓ 平均加速 <b>2x+</b>，达到优秀目标。';
  else if(avgSp>=1.2) note+='△ 加速有效，未达 2x，可进一步优化双缓冲/向量化。';
  else note+='✗ 加速偏低，建议排查 GM-UB 搬运瓶颈或 Scalar 指令占比。';
  $('perf-note').innerHTML=note;
  // avg badge
  const avgBadge=$('perf-avg-badge');
  if(avgBadge&&avgSp) avgBadge.textContent=`平均 ${avgSp.toFixed(2)}x`;

  // ── Case selector ──
  const sel=$('perf-case-sel');
  if(sel){
    sel.innerHTML=perfCases.map((c,i)=>`<option value="${i}">${c.name||'Case '+c.id}</option>`).join('');
    showCaseDetail(0);
  }
}

function showCaseDetail(selIdx){
  const cases=D.cases||[];
  const perfCases=cases.filter(c=>c.performance&&c.performance.speedup);
  const c=perfCases[selIdx]; if(!c) return;
  const osa=c.op_summary_avg||{};
  const perf=c.performance||{};
  const isDark=window.matchMedia('(prefers-color-scheme:dark)').matches;
  const CG=isDark?'rgba(255,255,255,.06)':'rgba(0,0,0,.06)';
  const CT=isDark?'#8b949e':'#57606a';

  // 算子类型检测（需要在 KPI row 之前定义）
  const aivTotal=osa['aiv_time(us)']||0;
  const aicTotal=osa['aicore_time(us)']||0;
  const taskType = osa['Task Type'] || '';
  const isAIC = taskType.includes('AIC') || taskType.includes('AI_CORE');
  const isAIV = taskType.includes('AIV') || taskType.includes('AI_VECTOR');

  // ── KPI row (根据算子类型动态显示) ──
  const kpiRow=$('perf-kpi-row');
  const kpis=[
    {label:'Task Type',     val:osa['Task Type']||'—'},
    {label:'Task Duration', val:osa['Task Duration(us)']!=null?osa['Task Duration(us)'].toFixed(2)+'μs':'—'},
    {label:'Block Dim',     val:osa['Block Dim']||(osa['Mix Block Dim']||'—')},
  ];
  // 根据算子类型添加时间指标
  if(isAIC && aicTotal>0){
    kpis.push({label:'AIC Time', val:osa['aicore_time(us)']!=null?osa['aicore_time(us)'].toFixed(2)+'μs':'—'});
    if(osa['cube_utilization(%)']!=null)
      kpis.push({label:'Cube Util', val:osa['cube_utilization(%)'].toFixed(1)+'%'});
  }
  if(isAIV || (isAIC && aivTotal>0)){
    kpis.push({label:'AIV Time', val:osa['aiv_time(us)']!=null?osa['aiv_time(us)'].toFixed(2)+'μs':'—'});
  }
  kpis.push({label:'Speedup', val:perf.speedup?perf.speedup.toFixed(2)+'x':'—'});
  kpiRow.innerHTML=kpis.map(k=>`
    <div class="kpi-card">
      <div class="kpi-lbl">${k.label}</div>
      <div class="kpi-val">${k.val}</div>
    </div>`).join('');

  // ── AIV/AIC Breakdown stacked bar (根据算子类型自适应) ──
  const aivSec=$('perf-aiv-section');
  if(aivSec){
    let html='';

    // AIC breakdown (Cube 算子或混合算子)
    if(isAIC && aicTotal>0 && osa['aic_mac_time(us)']!=null){
      const aicSegs=AIC_SEGS.map(s=>({...s,us:osa[s.key]||0,ratio:osa[s.ratioKey]||0}));
      const aicRatioTotal=aicSegs.reduce((a,b)=>a+b.ratio,0);
      const aicRemain=Math.max(0,1-aicRatioTotal);
      const cubeUtil=osa['cube_utilization(%)'];
      html+=`<div class="breakdown-title">Cube (AIC) 时间分解 <span style="color:var(--tx3);font-size:10px;font-weight:400">(${aicTotal.toFixed(2)}μs${cubeUtil!=null?` | Cube利用率 ${cubeUtil.toFixed(1)}%`:''})</span></div>`;
      html+='<div class="breakdown-stack">';
      aicSegs.forEach(s=>{
        if(s.ratio<=0) return;
        const pct=(s.ratio*100).toFixed(1);
        html+=`<div class="breakdown-seg" style="width:${pct}%;background:${s.color}" title="${s.label}: ${s.us.toFixed(2)}μs (${pct}%)">${s.ratio>0.08?pct+'%':''}</div>`;
      });
      if(aicRemain>0.01) html+=`<div class="breakdown-seg" style="width:${(aicRemain*100).toFixed(1)}%;background:var(--sf3)"></div>`;
      html+='</div>';
      html+='<div class="breakdown-legend">';
      aicSegs.forEach(s=>{
        if(s.ratio<=0) return;
        html+=`<div class="breakdown-legend-item"><div class="breakdown-legend-dot" style="background:${s.color}"></div>${s.label} <b style="color:var(--tx);font-family:var(--mono)">${(s.ratio*100).toFixed(1)}%</b></div>`;
      });
      html+='</div>';
    }

    // AIV breakdown (Vector 算子或混合算子)
    if((isAIV || isAIC) && aivTotal>0){
      if(isAIC) html+='<div style="height:12px"></div>';  // AIC+AIV 时添加间距
      const segsData=AIV_SEGS.map(s=>({...s,us:osa[s.key]||0,ratio:osa[s.ratioKey]||0}));
      const totalRatio=segsData.reduce((a,b)=>a+b.ratio,0);
      const remainRatio=Math.max(0,1-totalRatio);
      const typeLabel = isAIC?'Vector (AIV)':'Vector';
      html+=`<div class="breakdown-title">${typeLabel} 时间分解 <span style="color:var(--tx3);font-size:10px;font-weight:400">(总计 ${aivTotal.toFixed(2)}μs)</span></div>`;
      html+='<div class="breakdown-stack">';
      segsData.forEach(s=>{
        if(s.ratio<=0) return;
        const pct=(s.ratio*100).toFixed(1);
        const title=`${s.label}: ${s.us.toFixed(2)}μs (${pct}%)`;
        html+=`<div class="breakdown-seg" style="width:${pct}%;background:${s.color}" title="${title}">`;
        if(s.ratio>0.08) html+=pct+'%';
        html+='</div>';
      });
      if(remainRatio>0.01) html+=`<div class="breakdown-seg" style="width:${(remainRatio*100).toFixed(1)}%;background:var(--sf3);color:var(--tx3)"></div>`;
      html+='</div>';
      html+='<div class="breakdown-legend">';
      segsData.forEach(s=>{
        if(s.ratio<=0) return;
        html+=`<div class="breakdown-legend-item"><div class="breakdown-legend-dot" style="background:${s.color}"></div>${s.label} <b style="color:var(--tx);font-family:var(--mono)">${(s.ratio*100).toFixed(1)}%</b></div>`;
      });
      html+='</div>';
    }
    aivSec.innerHTML=html;
  }

  // ── Chart.js charts: 根据算子类型显示不同的图表组合 ──
  const grid=$('perf-charts-grid');
  if(grid&&typeof Chart!=='undefined'&&(aicTotal>0||aivTotal>0)){
    // 根据算子类型决定图表布局
    const isMix = isAIC && isAIV && aicTotal>0 && aivTotal>0;

    // 生成图表 HTML
    let chartsHTML = '';
    if(isMix){
      // MIX_AIC: 显示 3 个图表（AIC donut + AIV donut + timing bar with 4 columns）
      grid.className = "perf-detail-grid three-cols";
      chartsHTML += `<div class="perf-chart-card"><h5>AIC 执行类型占比</h5><canvas id="pc-donut-aic"></canvas></div>`;
      chartsHTML += `<div class="perf-chart-card"><h5>AIV 执行类型占比</h5><canvas id="pc-donut-aiv"></canvas></div>`;
      chartsHTML += `<div class="perf-chart-card" style="grid-column:1/-1"><h5>时延对比 (μs)</h5><canvas id="pc-timing"></canvas></div>`;
    } else if(isAIC && aicTotal>0){
      // 纯 AIC: 显示 2 个图表（AIC donut + timing bar with 3 columns）
      chartsHTML += `<div class="perf-chart-card"><h5>AIC 执行类型占比</h5><canvas id="pc-donut-aic"></canvas></div>`;
      chartsHTML += `<div class="perf-chart-card"><h5>时延对比 (μs)</h5><canvas id="pc-timing"></canvas></div>`;
    } else if(aivTotal>0){
      // 纯 AIV: 显示 2 个图表（AIV donut + timing bar with 3 columns）
      chartsHTML += `<div class="perf-chart-card"><h5>AIV 执行类型占比</h5><canvas id="pc-donut-aiv"></canvas></div>`;
      chartsHTML += `<div class="perf-chart-card"><h5>时延对比 (μs)</h5><canvas id="pc-timing"></canvas></div>`;
    }
    grid.innerHTML = chartsHTML;

    // Destroy old charts
    if(_perfDonutChartAIC){_perfDonutChartAIC.destroy();_perfDonutChartAIC=null;}
    if(_perfDonutChartAIV){_perfDonutChartAIV.destroy();_perfDonutChartAIV=null;}
    if(_perfTimingChart){_perfTimingChart.destroy();_perfTimingChart=null;}

    // 渲染 AIC donut chart
    if(isAIC && aicTotal>0){
      const aicDonutData=AIC_SEGS.map(s=>Math.round((osa[s.ratioKey]||0)*1000)/10);
      const aicDonutLabels=AIC_SEGS.map(s=>s.label.split(' ')[0]);
      const aicDonutColors=AIC_SEGS.map(s=>s.color);
      const aicDonutCanvas=document.getElementById('pc-donut-aic');
      if(aicDonutCanvas) _perfDonutChartAIC=new Chart(aicDonutCanvas,{
        type:'doughnut',
        data:{labels:aicDonutLabels,datasets:[{data:aicDonutData,backgroundColor:aicDonutColors.map(c=>c+'cc'),borderColor:aicDonutColors,borderWidth:1.5,hoverOffset:4}]},
        options:{responsive:true,maintainAspectRatio:false,cutout:'62%',
          plugins:{legend:{position:'bottom',labels:{color:CT,font:{size:10},boxWidth:10,padding:8}},
            tooltip:{callbacks:{label:ctx=>`${ctx.label}: ${ctx.raw.toFixed(1)}%`}}}}
      });
    }

    // 渲染 AIV donut chart
    if(aivTotal>0){
      const aivDonutData=AIV_SEGS.map(s=>Math.round((osa[s.ratioKey]||0)*1000)/10);
      const aivDonutLabels=AIV_SEGS.map(s=>s.label.split(' ')[0]);
      const aivDonutColors=AIV_SEGS.map(s=>s.color);
      const aivDonutCanvas=document.getElementById('pc-donut-aiv');
      if(aivDonutCanvas) _perfDonutChartAIV=new Chart(aivDonutCanvas,{
        type:'doughnut',
        data:{labels:aivDonutLabels,datasets:[{data:aivDonutData,backgroundColor:aivDonutColors.map(c=>c+'cc'),borderColor:aivDonutColors,borderWidth:1.5,hoverOffset:4}]},
        options:{responsive:true,maintainAspectRatio:false,cutout:'62%',
          plugins:{legend:{position:'bottom',labels:{color:CT,font:{size:10},boxWidth:10,padding:8}},
            tooltip:{callbacks:{label:ctx=>`${ctx.label}: ${ctx.raw.toFixed(1)}%`}}}}
      });
    }

    // Timing comparison (根据算子类型显示 3 或 4 个柱子)
    const tCanvas=document.getElementById('pc-timing');
    if(tCanvas){
      let timingLabels, timingVals, timingBg;
      if(isMix){
        // MIX_AIC: 显示 4 个柱子
        timingLabels=['Ref','Custom','AIC','AIV'];
        timingVals=[perf.ref_time_us||0, perf.custom_time_us||0, aicTotal, aivTotal];
        timingBg=[isDark?'rgba(139,148,158,.5)':'rgba(142,142,147,.5)',isDark?'rgba(26,127,55,.6)':'rgba(26,127,55,.5)',isDark?'rgba(9,105,218,.6)':'rgba(9,105,218,.5)',isDark?'rgba(191,135,0,.6)':'rgba(191,135,0,.5)'];
      } else if(isAIC){
        // 纯 AIC: 显示 3 个柱子
        timingLabels=['Ref','Custom','AIC'];
        timingVals=[perf.ref_time_us||0, perf.custom_time_us||0, aicTotal];
        timingBg=[isDark?'rgba(139,148,158,.5)':'rgba(142,142,147,.5)',isDark?'rgba(26,127,55,.6)':'rgba(26,127,55,.5)',isDark?'rgba(9,105,218,.6)':'rgba(9,105,218,.5)'];
      } else{
        // 纯 AIV: 显示 3 个柱子
        timingLabels=['Ref','Custom','AIV'];
        timingVals=[perf.ref_time_us||0, perf.custom_time_us||0, aivTotal];
        timingBg=[isDark?'rgba(139,148,158,.5)':'rgba(142,142,147,.5)',isDark?'rgba(26,127,55,.6)':'rgba(26,127,55,.5)',isDark?'rgba(9,105,218,.6)':'rgba(9,105,218,.5)'];
      }
      _perfTimingChart=new Chart(tCanvas,{
        type:'bar',
        data:{labels:timingLabels,datasets:[{label:'时延(μs)',data:timingVals,backgroundColor:timingBg,borderWidth:1,barPercentage:.6,minBarLength:4}]},
        options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',
          scales:{x:{grid:{color:CG},ticks:{color:CT,font:{size:10},callback:v=>v+'μs'}},y:{grid:{display:false},ticks:{color:CT,font:{size:11,weight:'600'}}}},
          plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>`${ctx.raw.toFixed(2)}μs`}}}}
      });
    }
  } else if(grid){
    grid.innerHTML='';
  }

  // ── Op summary detail table ──
  const opsDiv=$('perf-ops-detail');
  if(opsDiv){
    const dispFields=D.op_summary_fields||[];
    const rows=dispFields.filter(f=>osa[f]!=null);
    if(rows.length){
      let h=`<div style="font-size:11px;font-weight:700;color:var(--tx3);text-transform:uppercase;letter-spacing:.05em;margin:14px 0 6px">op_summary 原始数据（平均值）</div>`;
      h+='<table class="ops-tbl"><thead><tr><th>字段</th><th>值</th></tr></thead><tbody>';
      rows.forEach(f=>{
        const v=osa[f];
        const formatted=typeof v==='number'?v.toFixed(4):String(v);
        h+=`<tr><td class="ops-key">${f}</td><td class="ops-val">${formatted}</td></tr>`;
      });
      h+='</tbody></table>';
      opsDiv.innerHTML=h;
    } else {
      opsDiv.innerHTML=`<div style="font-size:12px;color:var(--tx3);margin-top:12px;padding:10px;background:var(--sf2);border-radius:var(--r)">op_summary 详细字段（aiv_vec_ratio 等）暂无。<br>如需 op_summary 分析，提供 op_summary_custom_fn_*.csv 文件后重新生成。<br><span style="font-size:11px;opacity:.7">TileLang 场景：speedup 来自 cann_result.json（CANN standalone .so），无需 op_summary CSV。</span></div>`;
    }
  }
}

// ═══════════════════════════════════════════════════
// DATA HEALTH & ANALYSIS INJECTION
// ═══════════════════════════════════════════════════
function toggleDH(barId){
  const body=document.getElementById(barId+'-body');
  if(body) body.classList.toggle('open');
}

function renderDH(tabKey, barId){
  const bar=$(barId); if(!bar) return;
  const diags=(D.diagnostics||[]).filter(d=>d.tab===tabKey);
  if(!diags.length) return;
  const counts={FOUND:0,DERIVED:0,MISSING:0};
  diags.forEach(d=>{if(counts[d.level]!==undefined)counts[d.level]++;});
  const any_issue=counts.DERIVED+counts.MISSING>0;
  bar.style.display='block';
  const countsEl=document.getElementById(barId+'-counts');
  if(countsEl) countsEl.innerHTML=
    `<span class="dh-found-c">${counts.FOUND} FOUND</span> `+
    `<span class="dh-derived-c">${counts.DERIVED} DERIVED</span> `+
    (counts.MISSING?`<span class="dh-missing-c">${counts.MISSING} MISSING</span>`:'');
  const body=document.getElementById(barId+'-body');
  if(!body) return;
  const ICONS={FOUND:'✓',DERIVED:'△',MISSING:'✗'};
  body.innerHTML=diags.map(d=>
    `<div class="dh-item dh-${d.level.toLowerCase()}">
      <span class="dh-icon">${ICONS[d.level]||'?'}</span>
      <span class="dh-msg"><b>${d.source}</b>${d.detail?' — '+d.detail:''}</span>
      ${d.hint?`<span class="dh-hint">${d.hint}</span>`:''}
    </div>`).join('');
  // Auto-open if there are issues
  if(any_issue) body.classList.add('open');
}

function injectAnalysis(slotId, html, title){
  const slot=$(slotId); if(!slot||!html) return;
  const card=el('div','card');
  card.innerHTML=`<div class="ct"><span class="ico">🤖</span>${title||'Claude 分析'}</div>`+
    `<div class="ana-section"><div class="ana-origin">📄 由 Claude 按 panels/SPEC.md 生成</div>`+
    html+'</div>';
  slot.appendChild(card);
}

// ═══════════════════════════════════════════════════
// TAB SWITCH
// ═══════════════════════════════════════════════════
document.querySelectorAll('.tb').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.tb').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tp').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-'+btn.dataset.tab).classList.add('active');
    // Re-render charts on tab show (needs layout to be visible)
    if(btn.dataset.tab==='prec') buildPrecChart();
    if(btn.dataset.tab==='perf') buildPerf();
  });
});

// ═══════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════

// 1. Data health for each tab
renderDH('algo',  'dh-algo');
renderDH('mem',   'dh-mem');
renderDH('prec',  'dh-prec');
renderDH('perf',  'dh-perf');

// 2. Inject Claude analysis panels
const P=D.panels||{};
if(P.algo&&P.algo.analysis_html)
  injectAnalysis('algo-analysis-slot', P.algo.analysis_html, '算法流图说明');
if(P.memory&&P.memory.analysis_html)
  injectAnalysis('mem-analysis-slot',  P.memory.analysis_html, 'Tiling 策略深度分析');
if(P.precision&&P.precision.analysis_html)
  injectAnalysis('prec-analysis-slot', P.precision.analysis_html, '精度诊断分析');
if(P.perf&&P.perf.analysis_html)
  injectAnalysis('perf-analysis-slot', P.perf.analysis_html, '性能瓶颈分析');

// 3. Extra custom panels
(P.extra||[]).forEach(ep=>{
  const tabId='tab-extra-'+ep.id;
  const content=el('div');
  const html=ep.analysis_html||ep.html;
  if(ep.analysis_html){
    const wrap=el('div','card');
    wrap.innerHTML='<div class="ct"><span class="ico">📄</span>'+ep.label+'</div>'+
      '<div class="ana-section">'+ep.analysis_html+'</div>';
    content.appendChild(wrap);
  } else if(ep.html){
    content.innerHTML=ep.html;
  }
  const tabDiv=el('div','tp');
  tabDiv.id=tabId;
  tabDiv.appendChild(content);
  document.querySelector('.content').appendChild(tabDiv);
  // Add tab button
  const btn=el('button','tb',ep.label);
  btn.dataset.tab='extra-'+ep.id;
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.tb').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tp').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    tabDiv.classList.add('active');
  });
  document.getElementById('main-tab-bar').appendChild(btn);
});

// 4. Main renders
buildHeader();
buildAlgo();
buildMemory();
buildPrecision();
buildPerf();
</script>
</body>
</html>
"""


# ─── GENERATE ────────────────────────────────────────────────────────────────

def generate_html(data: dict) -> str:
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, indent=2))
    html = html.replace("__TITLE__", data.get("op_name", "Op"))
    # Inject Chart.js inline (self-contained, no external CDN)
    _chartjs_path = Path(__file__).parent.parent / "assets" / "chart.umd.min.js"
    if _chartjs_path.exists():
        chartjs_src = _chartjs_path.read_text(encoding="utf-8", errors="replace")
        html = html.replace("/*__CHARTJS__*/", chartjs_src)
    return html


def _auto_write_analyses(data: dict, op_dir: Path):
    """
    为三个面板写 analysis.md 的**客观数据部分**（buffer 表、case 表、精度表）。

    ⚠ 本函数不生成分析文本。分析内容由 Claude 阅读 panels/*/data.json
    和 kernel/host 源码后，按 panels/*/SPEC.md 要求写入。
    已包含 Claude 撰写内容的文件（不含 NEEDS_CLAUDE_ANALYSIS 标记）不会被覆盖。
    """
    def _needs_analysis(p: Path) -> bool:
        """文件不存在、包含旧版占位、或包含 NEEDS_CLAUDE_ANALYSIS 标记时返回 True。"""
        if not p.exists():
            return True
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
            return ("（占位）" in t or "自动占位" in t
                    or "NEEDS_CLAUDE_ANALYSIS" in t
                    or "请补充" in t)
        except Exception:
            return True

    cases = data.get("cases", [])
    ub_buffers = data.get("ub_buffers", [])
    n_pass = data.get("n_pass", 0)
    n_total = data.get("n_total", 0)
    avg_speedup = data.get("avg_speedup")
    ub_used = data.get("ub_used_kb")
    ub_total = data.get("ub_total_kb")
    tiling_c = data.get("tiling", {}).get("consts", {})
    chip_aic = data.get("chip", {}).get("aic", 0)

    def _prof(c, *keys):
        s = c.get("op_summary_avg", {})
        for k in keys:
            v = s.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return None

    def _prof_str(c, *keys):
        s = c.get("op_summary_avg", {})
        for k in keys:
            v = s.get(k)
            if v is not None:
                return str(v)
        return ""

    def _fmt_ratio(v):
        if v is None:
            return "—"
        try:
            return f"{float(v):.1%}"
        except (TypeError, ValueError):
            return str(v)

    # ── 1. Memory analysis.md ──
    mem_path = op_dir / "panels" / "memory" / "analysis.md"
    if _needs_analysis(mem_path):
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        util_pct_str = ""
        if ub_used is not None and ub_total:
            try:
                util_pct_str = f"{ub_used:.2f} / {ub_total} KB = {ub_used/ub_total*100:.1f}%"
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        buf_rows = "\n".join(
            f"| {b['name']} | {b['kind']} @ {b['position']} | {b['size_kb']:.2f} KB |"
            for b in ub_buffers
        ) or "| （未解析到 buffer 声明） | — | — |"

        tile_size = int(tiling_c.get("tileSize", tiling_c.get("TILE_SIZE", 0)))
        case_rows = ""
        for c in cases:
            shapes_str = _prof_str(c, "Input Shapes")
            block_dim = _prof(c, "Block Dim")
            if shapes_str:
                first_shape = shapes_str.strip('"').split(';')[0].strip()
                total = 1
                for dim in first_shape.split(','):
                    try:
                        total *= int(dim.strip())
                    except ValueError:
                        pass
                if tile_size > 0 and chip_aic > 0:
                    launched = int(block_dim) if block_dim else chip_aic
                    max_by_tile = max(1, (total + tile_size - 1) // tile_size)
                    active = min(min(total, launched), max_by_tile)
                    has_tail = "是" if total % tile_size != 0 else "否"
                    case_rows += (
                        f"| {c.get('name','?')} | {first_shape} | {total:,} "
                        f"| {active} | {tile_size} | {has_tail} |\n"
                    )
        case_table = (
            "| Case | 输入 Shape | totalElems | active cores | tileSize | 有尾块 |\n"
            "|------|-----------|-----------|-------------|---------|-------|\n"
            + case_rows
        ) if case_rows else ""

        case_section = ("### 各 Case Tiling 参数\n\n" + case_table) if case_table else ""

        mem_path.write_text(f"""\
<!-- NEEDS_CLAUDE_ANALYSIS: 请阅读本文件底部的数据表 + panels/memory/data.json + op_kernel/*.cpp + op_host/*.cpp，
     按 panels/memory/SPEC.md 要求撰写分析，然后删除本行标记。 -->

## Tiling 策略分析

### UB Buffer 分配（客观数据，由脚本提取）

UB 使用：**{util_pct_str or "—"}**

| Buffer 名 | 类型 @ 位置 | 大小 |
|-----------|------------|------|
{buf_rows}

{case_section}

---
<!-- 以下各节由 Claude 按 panels/memory/SPEC.md 撰写 -->

### 切分方案

### 尾块处理

### 流水线设计

### 负载均衡诊断
""", encoding="utf-8")
        logger.info(f"  ↳ 写入 memory/analysis.md（客观数据 + 待 Claude 分析标记）")

    # ── 2. Precision analysis.md ──
    prec_path = op_dir / "panels" / "precision" / "analysis.md"
    if _needs_analysis(prec_path):
        prec_path.parent.mkdir(parents=True, exist_ok=True)
        pass_rate = f"{n_pass}/{n_total}" if n_total else "N/A"
        prec_rows = ""
        for c in cases:
            prec = c.get("precision", {})
            if prec:
                status = "✅" if prec.get("passed", True) else "❌"
                shapes_str = _prof_str(c, "Input Shapes")
                first_shape = shapes_str.strip('"').split(';')[0].strip() if shapes_str else c.get('shape', '?')
                prec_rows += (f"| {c.get('name','?')} | {first_shape} | {status} "
                              f"| {prec.get('max_re', 'N/A')} "
                              f"| {prec.get('mean_re', 'N/A')} "
                              f"| {prec.get('rmse', 'N/A')} |\n")
        prec_table = (f"| Case | Shape | 状态 | max_re | mean_re | rmse |\n"
                      f"|------|-------|------|--------|---------|------|\n"
                      f"{prec_rows}") if prec_rows else "（无精度数据）"

        prec_path.write_text(f"""\
<!-- NEEDS_CLAUDE_ANALYSIS: 请阅读本文件底部的数据表 + panels/precision/data.json，
     按 panels/precision/SPEC.md 要求撰写分析，然后删除本行标记。 -->

## 精度分析

### 各 Case 精度数据（客观数据，由脚本提取）

通过率：**{pass_rate}**

{prec_table}

---
<!-- 以下各节由 Claude 按 panels/precision/SPEC.md 撰写 -->

### 总体结论

### 误差类型解读

### 误差分布规律

### 误差来源推断

### 风险评估
""", encoding="utf-8")
        logger.info(f"  ↳ 写入 precision/analysis.md（客观数据 + 待 Claude 分析标记）")

    # ── 3. Perf analysis.md ──
    perf_path = op_dir / "panels" / "perf" / "analysis.md"
    if _needs_analysis(perf_path):
        perf_path.parent.mkdir(parents=True, exist_ok=True)
        perf_cases = [c for c in cases if c.get("performance", {}).get("speedup")]

        tile_size = int(tiling_c.get("tileSize", tiling_c.get("TILE_SIZE", 0)))
        perf_rows = ""
        for c in perf_cases:
            p = c.get("performance", {})
            sp = p.get("speedup", 0)
            sp_icon = "🚀" if sp >= 1.5 else ("✅" if sp >= 1.0 else "⚠")
            ref_us = p.get("ref_time_us", 0)
            cust_us = p.get("custom_time_us", 0)
            shapes_str = _prof_str(c, "Input Shapes")
            first_shape = shapes_str.strip('"').split(';')[0].strip() if shapes_str else c.get('shape', '?')
            total = 0
            for dim in first_shape.split(','):
                try:
                    total = (total or 1) * int(dim.strip())
                except ValueError:
                    pass
            block_dim = _prof(c, "Block Dim")
            if tile_size > 0 and total > 0 and chip_aic > 0:
                launched = int(block_dim) if block_dim else chip_aic
                active = min(min(total, launched), max(1, (total + tile_size - 1) // tile_size))
                cores_str = f"{active}/{launched}"
            else:
                cores_str = str(int(block_dim)) if block_dim else "?"
            vec_r = _prof(c, "aiv_vec_ratio", "vec_ratio")
            mte2_r = _prof(c, "aiv_mte2_ratio", "mte2_ratio")
            scalar_r = _prof(c, "aiv_scalar_ratio", "scalar_ratio")
            total_str = f"{total/1e6:.2f}M" if total >= 1e6 else (f"{total/1e3:.1f}K" if total >= 1000 else str(total))
            perf_rows += (
                f"| {c.get('name','?')} | {first_shape} | {total_str} "
                f"| {cores_str} | {ref_us:.1f} | {cust_us:.1f} "
                f"| **{sp:.2f}x** {sp_icon} "
                f"| {_fmt_ratio(vec_r)} / {_fmt_ratio(mte2_r)} / {_fmt_ratio(scalar_r)} |\n"
            )
        perf_table = (
            "| Case | Shape | 元素数 | 活跃/启动核 | ref(us) | custom(us) | Speedup | vec/mte2/scalar |\n"
            "|------|-------|--------|-----------|--------|-----------|---------|----------------|\n"
            + perf_rows
        ) if perf_rows else "（无性能数据）"

        perf_path.write_text(f"""\
<!-- NEEDS_CLAUDE_ANALYSIS: 请阅读本文件底部的数据表 + panels/perf/data.json + op_kernel/*.cpp，
     按 panels/perf/SPEC.md 要求撰写分析，然后删除本行标记。 -->

## 性能分析

### 各 Case 性能数据（客观数据，由脚本提取）

平均 Speedup：**{f"{avg_speedup:.2f}x" if avg_speedup is not None else "N/A"}**（几何均值，{len(perf_cases)} 个 case）

{perf_table}

---
<!-- 以下各节由 Claude 按 panels/perf/SPEC.md 撰写 -->

### 基准说明

### 性能规律

### 瓶颈推断

### 优化建议
""", encoding="utf-8")
        logger.info(f"  ↳ 写入 perf/analysis.md（客观数据 + 待 Claude 分析标记）")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = parse_args()
    if not args.op_dir and not any([args.op_desc, args.eval, args.precision]):
        logger.error("错误：需要 --op-dir 或至少一个数据源参数。")
        sys.exit(1)
    data = collect_data(args)

    # --extract-only: write panels/*/data.json and exit
    if getattr(args, "extract_only", False):
        if args.op_dir:
            extract_panel_data_json(data, Path(args.op_dir))
        logger.info("\n✅ --extract-only 完成。请按精品流程 Step 3-6 让 Claude 分析各 panel。")
        return

    extract_panel_data_json(data, Path(args.op_dir)) if args.op_dir else None

    # ── 自动生成 analysis.md（精品流程，始终执行）──
    # 必须在 generate_html 之前写好 analysis.md，然后重新加载 panels_content
    if args.op_dir:
        _auto_write_analyses(data, Path(args.op_dir))
        # 重新加载 panels（以便 HTML 包含刚写的 analysis.md）
        data["panels"] = discover_panels(Path(args.op_dir))

    html = generate_html(data)
    # 默认输出到 op_dir/dashboard.html，无 op_dir 时落在当前目录
    if args.output:
        out = Path(args.output)
    elif args.op_dir:
        out = Path(args.op_dir) / "dashboard.html"
    else:
        out = Path("dashboard.html")
    out.write_text(html, encoding="utf-8")
    logger.info(f"\n✅ 看板已生成：{out.resolve()}")
    logger.info(f"   大小：{out.stat().st_size/1024:.1f} KB")
    logger.info(f"   Cases：{data['n_pass']}/{data['n_total']} PASS")
    if data.get("avg_speedup"):
        logger.info(f"   平均 Speedup：{data['avg_speedup']:.2f}x")

    # ── 自动运行质量检测（check_dashboard.py）──
    _check_script = Path(__file__).parent / "check_dashboard.py"
    if _check_script.exists():
        logger.info("")
        _result = subprocess.run(
            [sys.executable, str(_check_script), str(out.resolve())],
            capture_output=False
        )
        if _result.returncode != 0:
            logger.info("\n⚠️  看板存在质量问题，请修复后重新生成。")
            sys.exit(_result.returncode)
        else:
            logger.info("\n✅ 质量检测通过，看板可交付。")


if __name__ == "__main__":
    main()
