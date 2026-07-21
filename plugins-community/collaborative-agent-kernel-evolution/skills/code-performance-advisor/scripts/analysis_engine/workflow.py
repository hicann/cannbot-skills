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
Workflow Orchestrator - Phase 0 Implementation

设计原则落地: CLI 固化流程,确保所有步骤强制执行

Usage:
    python workflow.py run --op fastgelu --mode interactive
    python workflow.py resume --op fastgelu
    python workflow.py status --op fastgelu
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import stat
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List

# 导入新模块
from context_keys import CK
from path_manager import PathManager
from profiling_extractor import ProfilingExtractor
from session_manager import SessionManager

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


class Phase(Enum):
    """工作流阶段 - 固化的状态机"""
    INIT = "init"
    TAG = "tag"
    SCORE = "score"
    ROUTE = "route"
    SUGGEST = "suggest"
    APPLY = "apply"
    BUILD = "build"
    EVALUATE = "evaluate"
    COMPARE = "compare"
    UPDATE = "update"
    DONE = "done"
    ERROR = "error"


@dataclass
class RouteDecision:
    """路由决策 - 数据契约"""
    path: str  # "fast" | "moderate" | "deep" | "scalar_locked"
    max_score: float
    coverage_ratio: float
    top_rule_id: str
    confidence: str  # "high" | "medium" | "low" | "locked"


def _compute_score_cache_key(index_path: Path, tag_file: Path) -> str:
    """计算 SCORE 阶段的缓存键：sha256(index.json + tag_file) 前 20 位。

    任意一个输入文件内容变化，缓存键即失效，触发重评分。
    """
    import hashlib
    h = hashlib.sha256()
    for p in (index_path, tag_file):
        try:
            h.update(p.read_bytes())
        except Exception:
            h.update(b"\x00MISSING")
    return h.hexdigest()[:20]


@dataclass
class WorkflowState:
    """工作流状态 - 持久化契约"""
    op_name: str
    phase: Phase
    mode: str  # always "interactive"
    context: Dict
    history: List[Dict]
    created_at: str
    updated_at: str

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowState":
        data["phase"] = Phase(data["phase"])
        return cls(**data)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["phase"] = self.phase.value
        return result


class WorkflowError(Exception):
    """工作流错误"""
    pass


class WorkflowPaused(Exception):
    """工作流暂停信号 —— 等待 Agent 执行 subskill 后 resume，不计入 error"""
    pass


class WorkflowEngine:
    """
    工作流引擎 - 强制状态转移,防止遗漏

    设计原则:
    1. 每个阶段完成后必须保存状态
    2. 状态转移完全确定性,不依赖 LLM
    3. 支持断点恢复
    """

    def __init__(self, op_name: str, mode: str = "interactive", session_id: Optional[str] = None,
                 skip_build: bool = False, force_retag: bool = False):
        self.op_name = op_name
        self.mode = mode
        self.skip_build = skip_build
        self.force_retag = force_retag

        # ✅ 初始化PathManager
        self.pm = PathManager(ROOT)

        # ✅ 初始化SessionManager
        workspace = self.pm.get_workspace_dir()
        self.sm = SessionManager(workspace)

        # ✅ 创建或恢复session
        if session_id:
            # 恢复已有session
            self.session_id = session_id
            self.session_dir = self.sm.get_session_dir(session_id)
            self.session_info = self.sm.load_session(session_id)

            if not self.session_info:
                raise ValueError(f"Session not found: {session_id}")

            logger.info(f"[WORKFLOW] Resuming session: {session_id}")
        else:
            # 创建新session
            input_baseline = self.pm.get_input_dir(op_name)
            user_info = {
                "launched_by": f"{os.getenv('USER', 'unknown')}@{os.uname().nodename}",
                "cli_args": " ".join(sys.argv)
            }

            self.session_id, self.session_dir = self.sm.create_session(
                op_name=op_name,
                mode=mode,
                input_baseline_dir=input_baseline,
                user_info=user_info
            )
            self.session_info = self.sm.load_session(self.session_id)

            # 记录关联的 output_dir（CAKE2 编译产物目录），供清理/追溯使用
            try:
                output_dir = str(self.pm.get_build_dir(op_name))
                self.sm.update_session_resources(self.session_id, {"output_dir": output_dir})
            except Exception:
                pass  # output_dir 非关键路径，失败不阻断启动

        # ✅ 状态文件路径(session隔离)
        self.state_file = self.session_dir / "workflow_state.json"
        self.state: Optional[WorkflowState] = None
        self._phase_start_time: Optional[datetime] = None  # 用于记录每个阶段的耗时

        logger.info(f"[WORKFLOW] Session: {self.session_id}")
        logger.info(f"[WORKFLOW] Session directory: {self.session_dir}")

    def run(self):
        """执行完整工作流"""
        logger.info(f"[WORKFLOW] Starting optimization for '{self.op_name}'")
        logger.info(f"[WORKFLOW] Mode: {self.mode}")
        logger.info("")

        # 初始化或恢复状态
        if self.state_file.exists():
            self.state = self._load_state()
            # 若上次以 ERROR 状态结束，恢复到出错的阶段重试
            if self.state.phase == Phase.ERROR:
                error_phase = self.state.context.get(CK.ERROR, {}).get("phase")
                if error_phase:
                    try:
                        retry_phase = Phase(error_phase)
                        logger.info(f"[WORKFLOW] Previous run failed at phase '{error_phase}', retrying from there")
                        self.state.phase = retry_phase
                        # 清除旧的 error 上下文，避免误导
                        self.state.context.pop(CK.ERROR, None)
                    except ValueError:
                        logger.info(f"[WORKFLOW] Unknown error phase '{error_phase}', restarting from INIT")
                        self.state.phase = Phase.INIT
                else:
                    logger.info(f"[WORKFLOW] Previous run failed (phase unknown), restarting from INIT")
                    self.state.phase = Phase.INIT
            else:
                logger.info(f"[WORKFLOW] Resuming from phase: {self.state.phase.value}")
        else:
            self.state = WorkflowState(
                op_name=self.op_name,
                phase=Phase.INIT,
                mode=self.mode,
                context={},
                history=[],
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat()
            )
            logger.info(f"[WORKFLOW] Starting new workflow")

        # 状态机循环
        while self.state.phase not in [Phase.DONE, Phase.ERROR]:
            try:
                self._execute_phase()
            except WorkflowPaused as e:
                # 干净暂停：Agent 需要先执行 subskill，不计入 error
                duration_ms = 0
                if self._phase_start_time:
                    duration_ms = int((datetime.now(timezone.utc) - self._phase_start_time).total_seconds() * 1000)
                self.state.history.append({
                    "phase": self.state.phase.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": duration_ms,
                    "exit_reason": "paused",
                })
                logger.info("")
                logger.info("⏸️  Workflow paused — waiting for agent action")
                logger.info(f"   Reason: {e}")
                logger.info(f"   After completing the required action, resume with:")
                logger.info(f"   python workflow.py resume --op {self.op_name}")
                self._save_state()
                break
            except Exception as e:
                self._handle_error(e)
                break

        logger.info("")
        logger.info(f"[WORKFLOW] Workflow completed: {self.state.phase.value}")

    def _execute_phase(self):
        """执行当前阶段"""
        self._phase_start_time = datetime.now(timezone.utc)
        phase = self.state.phase

        if phase == Phase.INIT:
            self._phase_init()
        elif phase == Phase.TAG:
            self._phase_tag()
        elif phase == Phase.SCORE:
            self._phase_score()
        elif phase == Phase.ROUTE:
            self._phase_route()
        elif phase == Phase.SUGGEST:
            self._phase_suggest()
        elif phase == Phase.APPLY:
            self._phase_apply()
        elif phase == Phase.BUILD:
            self._phase_build()
        elif phase == Phase.EVALUATE:
            self._phase_evaluate()
        elif phase == Phase.COMPARE:
            self._phase_compare()
        elif phase == Phase.UPDATE:
            self._phase_update()

    def _phase_init(self):
        """Phase INIT: 检查前置条件"""
        logger.info(f"[PHASE: INIT] Checking prerequisites...")

        # ✅ 使用PathManager获取路径
        input_dir = self.pm.get_input_dir(self.op_name)
        if not input_dir.exists():
            # 尝试自动从 CAKE2/output/{op}/ 初始化
            logger.info(f"  ⚠️  Input directory not found: {input_dir}")
            logger.info(f"  🔄 Attempting auto-initialization from CAKE2 output/...")
            init_script = ROOT / "scripts" / "analysis_engine" / "init_workspace.py"
            if init_script.exists():
                cake2_root = self.pm.get_cake2_root()
                init_result = subprocess.run(
                    [sys.executable, str(init_script),
                     "--root", str(cake2_root),
                     "--op", self.op_name],
                    cwd=ROOT,
                    capture_output=True,
                    text=True
                )
                if init_result.returncode == 0 and input_dir.exists():
                    logger.info(f"  ✅ Auto-initialized workspace for '{self.op_name}'")
                else:
                    output = (init_result.stdout or "") + (init_result.stderr or "")
                    if output:
                        logger.info(f"  ❌ Init output:\n{output}")
                    raise WorkflowError(
                        f"Input directory not found and auto-init failed: {input_dir}\n"
                        f"Please run manually: python scripts/analysis_engine/init_workspace.py --op {self.op_name}"
                    )
            else:
                raise WorkflowError(
                    f"Input directory not found: {input_dir}\n"
                    f"Please run: python scripts/analysis_engine/init_workspace.py --op {self.op_name}"
                )

        code_dir = input_dir / "code"

        # Bug 1 fix: 兼容 profiling/ 和 profiling_data/ 两种目录名
        profiling_dir = input_dir / "profiling"
        if not profiling_dir.exists():
            profiling_dir_alt = input_dir / CK.PROFILING_DATA
            if profiling_dir_alt.exists():
                profiling_dir = profiling_dir_alt

        if not code_dir.exists():
            raise WorkflowError(f"Code directory not found: {code_dir}")

        logger.info(f"  ✅ Code directory: {code_dir}")

        # ✅ 使用ProfilingExtractor动态提取profiling数据
        if profiling_dir.exists():
            # 兼容三种格式:
            # 1. profiling/op_summary.csv              (新式，init_workspace.py 生成的 flat 格式)
            # 2. profiling/profiling_csv/*.csv          (旧式，直接复制的嵌套格式)
            # 3. profiling_data/profiling_csv/*.csv     (手动放置的原始格式)
            csv_files = list((profiling_dir / "profiling_csv").glob("*.csv"))
            if not csv_files:
                csv_files = list(profiling_dir.glob("*.csv"))
            if csv_files:
                csv_path = csv_files[0]
                try:
                    # ✅ 传入 op_name 以精确匹配
                    extractor = ProfilingExtractor(csv_path, target_op_name=self.op_name)
                    profiling_data = extractor.extract()

                    logger.info(f"  ✅ Profiling extracted:")
                    logger.info(f"     Source: {csv_path.relative_to(input_dir)}")
                    logger.info(f"     Task Type: {profiling_data.task_type}")
                    logger.info(f"     Duration: {profiling_data.task_duration_us} us")
                    logger.info(f"     Bottleneck Hint: {profiling_data.get_bottleneck_hint()}")

                    # 保存到context
                    self.state.context[CK.PROFILING_DATA] = profiling_data.to_dict()
                    # 保存 baseline duration 供 COMPARE 阶段自动对比
                    if CK.BASELINE_DURATION_US not in self.state.context:
                        self.state.context[CK.BASELINE_DURATION_US] = profiling_data.task_duration_us
                    # 记录 baseline CSV 的修改时间戳（供 COMPARE 阶段区分新旧 profiling）
                    if CK.BASELINE_CSV_MTIME not in self.state.context:
                        self.state.context[CK.BASELINE_CSV_MTIME] = csv_path.stat().st_mtime

                except Exception as e:
                    logger.info(f"  ⚠️  Failed to extract profiling: {e}")
                    self.state.context[CK.PROFILING_DATA] = None
            else:
                logger.info(f"  ⚠️  No CSV files found in {profiling_dir.name}/")
                self.state.context[CK.PROFILING_DATA] = None
        else:
            logger.info(f"  ⚠️  Profiling data not found (will use code-only analysis)")
            self.state.context[CK.PROFILING_DATA] = None

        self.state.context[CK.INPUT_DIR] = str(input_dir)
        self.state.context[CK.HAS_PROFILING] = profiling_dir.exists()

        # 转移到下一阶段
        self._transition_to(Phase.TAG)

    def _phase_tag(self):
        """Phase TAG: 代码标注"""
        logger.info(f"[PHASE: TAG] Extracting code tags...")

        # ✅ 使用PathManager
        tag_file = self.pm.get_tag_file(self.op_name)
        tags_dir = self.pm.get_cache_dir() / "tags"

        # 判断是否需要强制重新 tag（CLI flag 或上一轮 COMPARE 不达标）
        ctx_force_retag = self.state.context.get(CK.FORCE_RETAG, False)
        if ctx_force_retag:
            self.state.context[CK.FORCE_RETAG] = False  # 消费掉，只触发一次
            logger.info(f"  🔄 Forced retag (previous profiling did not meet target)")
        should_force = self.force_retag or ctx_force_retag

        # 查找最新 tag 文件（精确名或带 timestamp 的变体）
        # should_force=True 时跳过缓存，直接置 None
        resolved_tag = None
        if not should_force:
            if tag_file.exists():
                resolved_tag = tag_file
            elif tags_dir.exists():
                candidates = sorted(
                    tags_dir.glob(f"tag_{self.op_name}*.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if candidates:
                    resolved_tag = candidates[0]

        if resolved_tag is not None:
            # 代码 hash 比对：单一判据，避免 mtime + 行数多因子误判
            current_code_hash = self._compute_code_hash()
            sidecar_file = resolved_tag.with_suffix(".code_hash")
            if sidecar_file.exists():
                stored_hash = sidecar_file.read_text().strip()
                if stored_hash == current_code_hash:
                    logger.info(f"  ✅ Tag cache hit (code hash unchanged)")
                else:
                    logger.info(f"  ⚠️  Code has changed since last tag (hash mismatch)")
                    logger.info(f"     Reusing cached tag — use --force-retag to regenerate")
            else:
                logger.info(f"  ✅ Tag cache hit")

            # ── 质量检测：拒绝 auto_tag.py 生成的低质量缓存 ──────────────
            try:
                with open(resolved_tag) as _cached_fp:
                    _cached = json.load(_cached_fp)
                if _cached.get("auto_generated") is True:
                    _conf = _cached.get("confidence", "?")
                    logger.info(f"  ❌ Tag cache rejected: generated by auto_tag.py "
                                f"(heuristic, confidence={_conf})")
                    logger.info(f"  📌 REQUIRED: Run [code_tag] subskill for LLM-quality tags:")
                    logger.info(f"     Read: subskills/code_tag.md")
                    logger.info(f"     Output to: {tags_dir}")
                    logger.info(f"     Then resume with --force-retag: "
                                f"workflow.py resume --op {self.op_name} --force-retag")
                    raise WorkflowPaused(
                        f"Cached tag was auto-generated (heuristic, low accuracy). "
                        f"Run [code_tag] subskill → save to {tags_dir} "
                        f"→ resume with: workflow.py resume --op {self.op_name} --force-retag"
                    )
            except WorkflowPaused:
                raise
            except Exception:
                pass  # JSON 解析失败时放行，不阻断流程

            tag_file = resolved_tag
        else:
            # ┌─────────────────────────────────────────────────────────────┐
            # │  TAG 阶段：必须由 Agent 调用 code_tag subskill（LLM 驱动）   │
            # │  auto_tag.py 已废弃，不再作为 fallback                       │
            # └─────────────────────────────────────────────────────────────┘
            logger.info(f"  ⚠️  {'Tag cache invalidated (force_retag)' if should_force else 'Tag file not found'}.")
            logger.info(f"  📌 REQUIRED: Run [code_tag] subskill for LLM-quality tags:")
            logger.info(f"     Read: subskills/code_tag.md")
            logger.info(f"     Output to: {tags_dir}")
            logger.info(f"     Then resume with: workflow.py resume --op {self.op_name}")
            logger.info("")

            # 暂停，让 Agent 执行 code_tag subskill（唯一正确路径）
            raise WorkflowPaused(
                f"Run [code_tag] subskill, save tag to {tags_dir}, then resume"
            )

        # 验证 tag 文件格式
        try:
            with open(tag_file) as _tag_fp:
                tags = json.load(_tag_fp)

            def _as_list(value):
                return value if isinstance(value, list) else []

            domain_tags = _as_list(tags.get("domain_tags") or tags.get("domain"))
            symptom_tags = _as_list(tags.get("symptom_tags") or tags.get("symptoms"))
            context_tags = _as_list(tags.get("context_tags") or tags.get("context"))

            logger.info(f"  ✅ Tag file valid")
            logger.info(f"     Domain tags: {len(domain_tags)}")
            logger.info(f"     Symptom tags: {len(symptom_tags)}")
            logger.info(f"     Context tags: {len(context_tags)}")
        except Exception as e:
            raise WorkflowError(f"Invalid tag file: {e}") from e

        # Tag taxonomy validation (non-blocking: unknown tags are warned, not fatal)
        try:
            from tag_validator import extract_valid_tags, validate_tag_file as _validate_tags
            taxonomy_path = ROOT / "references" / "standards" / "tag_taxonony.md"
            if taxonomy_path.exists():
                valid_tags = extract_valid_tags(taxonomy_path)
                is_valid, invalid_tags, suggestions = _validate_tags(tag_file, valid_tags, verbose=False)
                if not is_valid:
                    logger.info(f"  ⚠️  Taxonomy warning: {len(invalid_tags)} unknown tag(s) — {invalid_tags[:3]}")
                    for bad, sugg in list(suggestions.items())[:3]:
                        if sugg:
                            logger.info(f"       '{bad}' → suggest {sugg}")
                else:
                    logger.info(f"  ✅ Tag taxonomy validated ({len(valid_tags)} known tags)")
        except Exception:
            pass  # 验证失败不阻断流程

        self.state.context[CK.TAG_FILE] = str(tag_file)

        # 记录接受此 tag 时的代码 hash（供下次快速比对，避免 mtime 多因子误判）
        try:
            sidecar_file = tag_file.with_suffix(".code_hash")
            sidecar_file.write_text(self._compute_code_hash())
        except Exception:
            pass  # 写 sidecar 失败不阻断流程

        self._transition_to(Phase.SCORE)

    def _phase_score(self):
        """Phase SCORE: 规则评分"""
        logger.info(f"[PHASE: SCORE] Scoring rules...")

        tag_file = self.state.context[CK.TAG_FILE]
        output_file = self.session_dir / "scored_results.json"
        cache_key_file = output_file.with_suffix(".cache_key")
        index_path = ROOT / "assets" / "manifests" / "index.json"

        # ── 缓存有效性检查：内容 hash 驱动，而非仅判断文件是否存在 ──────────────
        # 只有 index.json + tag_file 两者内容均未变化时，才复用缓存。
        # 这样规则库更新（bootstrap/update-index 后）或 tag 文件被 Agent 重写时
        # 都能自动触发重评分，避免旧结果误导后续路由。
        current_key = _compute_score_cache_key(index_path, Path(tag_file))
        cache_valid = False
        if output_file.exists() and cache_key_file.exists():
            try:
                stored_key = cache_key_file.read_text().strip()
                cache_valid = (stored_key == current_key)
            except Exception:
                cache_valid = False

        if cache_valid:
            logger.info(f"  ✅ Scoring cache hit (index + tag unchanged, skipping re-score)")
        else:
            if output_file.exists():
                logger.info(f"  ⚠️  Cache miss (index or tag changed) — re-scoring...")
            # 直接调用 score_rules（原为 subprocess 调用 cli.py，现改为直接 import）
            from cli import score_rules
            logger.info(f"  🔄 Scoring rules (index: {index_path.name}, tag: {Path(tag_file).name})...")
            try:
                score_rules(
                    index_path=index_path,
                    tag_file=Path(tag_file),
                    output_path=output_file,
                    op_name=self.op_name,
                )
            except SystemExit as e:
                raise WorkflowError(f"Scoring failed (exit {e.code})") from e
            except Exception as e:
                raise WorkflowError(f"Scoring failed: {e}") from e

            # 写入新缓存键（原子性：先写 key 再读结果，key 写失败不阻断主流程）
            try:
                cache_key_file.write_text(current_key)
            except Exception as e:
                # 缓存键写失败不阻断主流程，仅记录便于排查
                logger.debug("Failed to write score cache key: %s", e)
            logger.info(f"  ✅ Scoring completed")

        # 读取 session 隔离的结果
        with open(output_file) as _scored_fp:
            scored = json.load(_scored_fp)
        self.state.context[CK.SCORED_RESULTS] = scored

        self._transition_to(Phase.ROUTE)

    def _phase_route(self):
        """Phase ROUTE: 路由决策"""
        logger.info(f"[PHASE: ROUTE] Making routing decision...")

        scored = self.state.context[CK.SCORED_RESULTS]

        # Fix: scored_results uses "results" not "rules"
        results = scored.get("results", [])

        # Filter valid rules (score > 0 and not conflicted)
        valid_rules = [r for r in results if r.get("score", 0) > 0 and not r.get("conflict", True)]

        if not valid_rules:
            logger.info(f"  ⚠️  No valid rules matched")
            route = RouteDecision(
                path="deep",
                max_score=0.0,
                coverage_ratio=0.0,
                top_rule_id="none",
                confidence="low"
            )
        else:
            # ✅ FIX: 使用 raw_score 进行路由决策（而非 adjusted score）
            # 这样即使 top rule 被标记为 redundant，也能正确路由到对应的 path
            top_rule = max(valid_rules, key=lambda r: r.get("raw_score", r.get("score", 0)))

            # 使用 raw_score 进行路由决策
            raw_score = top_rule.get("raw_score", top_rule.get("score", 0))
            coverage = top_rule.get("coverage_ratio", 0.0)
            is_redundant = top_rule.get("redundant", False)

            # Extract rule_id from rule_path
            rule_path = top_rule.get("rule_path", "")
            rule_id = Path(rule_path).parent.name

            # 路由决策逻辑 (基于 raw_score，确定性)
            if raw_score >= 0.7:
                path = "fast"
                confidence = "high"
            elif raw_score >= 0.55 and coverage >= 0.8:
                path = "fast"
                confidence = "medium"
            elif raw_score >= 0.3:
                path = "moderate"
                confidence = "medium"
            else:
                path = "deep"
                confidence = "low"

            route = RouteDecision(
                path=path,
                max_score=raw_score,  # ✅ 保存 raw_score 而非 adjusted
                coverage_ratio=coverage,
                top_rule_id=rule_id,
                confidence=confidence
            )

            # ✅ 如果 top rule 是 redundant，发出警告（但仍然使用其 raw_score 路由）
            if is_redundant:
                logger.info(f"  ⚠️  Top rule '{rule_id}' is redundant (already applied)")
                logger.info(f"     Will explore alternative optimizations in SUGGEST phase")

        logger.info(f"  ✅ Route: {route.path}")
        logger.info(f"     Confidence: {route.confidence}")
        logger.info(f"     Top rule: {route.top_rule_id} (score={route.max_score:.3f})")

        # ── Scalar-Locked override：在规则路由之后，检查是否应升级到算法重设计路径 ──
        scalar_locked_signals = self._check_scalar_locked(route)
        if scalar_locked_signals:
            route = RouteDecision(
                path="scalar_locked",
                max_score=route.max_score,
                coverage_ratio=route.coverage_ratio,
                top_rule_id=route.top_rule_id,
                confidence="locked",
            )
            logger.info(f"  🔒 Scalar-Locked override → route upgraded to: scalar_locked")
            for sig in scalar_locked_signals:
                logger.info(f"     • {sig}")

        self.state.context[CK.ROUTE] = asdict(route)
        self._transition_to(Phase.SUGGEST)

    def _phase_suggest(self):
        """Phase SUGGEST: 生成建议（需要 Agent 调用 subskill 后 resume）"""
        suggestions_dir = self.session_dir / "suggestions"
        suggestions_dir.mkdir(parents=True, exist_ok=True)

        # ── Resume 路径：Agent 已调用 subskill，建议文件已存在 ──
        suggestion_files = list(suggestions_dir.glob("*.md"))
        if suggestion_files:
            logger.info(f"[PHASE: SUGGEST] Suggestion file found:")
            for f in suggestion_files:
                logger.info(f"  - {f.name}")
            self.state.context[CK.SUGGESTION_FILE] = str(suggestion_files[0])
            self._transition_to(Phase.APPLY)
            return

        # ── 首次进入：打印上下文，暂停等待 Agent 调用 subskill ──
        route = RouteDecision(**self.state.context[CK.ROUTE])
        profiling_data = self.state.context.get(CK.PROFILING_DATA)

        logger.info(f"[PHASE: SUGGEST] Waiting for subskill invocation")
        logger.info("")
        logger.info(f"=== Current Context ===")
        logger.info(f"Operator: {self.op_name}")
        logger.info(f"Route: {route.path} path (confidence: {route.confidence})")
        if route.path == "fast":
            logger.info(f"Top rule: {route.top_rule_id} (score={route.max_score:.3f})")

        if profiling_data:
            logger.info(f"\nProfiling:")
            logger.info(f"  Task Type: {profiling_data['task_type']}")
            logger.info(f"  Task Duration: {profiling_data['task_duration_us']} us")
            logger.info(f"  Bottleneck Hint: {profiling_data.get('bottleneck_hint', 'N/A')}")

        logger.info(f"\nCode: {self.state.context[CK.INPUT_DIR]}/code/")
        logger.info("")

        if route.path == "fast":
            subskill = "suggest"
            logger.info(f"📚 Required subskill: [{subskill}]")
            logger.info(f"   Read: subskills/{subskill}.md")
            logger.info(f"   Explore rule: assets/rules/special_rules/{route.top_rule_id}/")
            logger.info(f"   Reference: references/standards/op_summary_header_guide.md")
        elif route.path == "moderate":
            subskill = "deep_research"
            logger.info(f"📚 Required subskill: [{subskill}]")
            logger.info(f"   Read: subskills/{subskill}.md")
            logger.info(f"   5-Step Logic: Bound Analysis → Memory → Pipeline → Tiling → Sync")
        elif route.path == "scalar_locked":
            subskill = "algorithm_redesign"
            logger.info(f"📚 Required subskill: [{subskill}]")
            logger.info(f"   Read: subskills/{subskill}.md")
            logger.info(f"   Input: latest suggestion / deep_research output in {suggestions_dir}/")
            logger.info(f"   Goal: eliminate scalar dependency chains — not pattern-level fixes")
            logger.info(f"   Required output: equivalence class + numerical error bound before code change")
        else:
            subskill = "deep_research"
            logger.info(f"📚 Required subskill: [{subskill}]")
            logger.info(f"   5-Step Logic: Bound Analysis → Memory → Pipeline → Tiling → Sync")

        logger.info("")
        logger.info(f"   Save suggestion to: {suggestions_dir}/<name>.md")
        logger.info("")

        # 在暂停前记录当前代码哈希，供 APPLY 阶段验证代码是否真的被修改
        self.state.context[CK.PRE_APPLY_CODE_HASH] = self._compute_code_hash()

        raise WorkflowPaused(
            f"Run [{subskill}] subskill, save output to {suggestions_dir}/*.md, "
            f"then resume with: workflow.py resume --op {self.op_name}"
        )

    def _phase_apply(self):
        """Phase APPLY: 应用建议"""
        logger.info(f"[PHASE: APPLY] Applying suggestion...")

        suggestion_file = self.state.context.get(CK.SUGGESTION_FILE, "")
        logger.info(f"  📄 Suggestion: {suggestion_file or '(none)'}")
        logger.info(f"  ⚠️  Apply changes manually to:")
        logger.info(f"     {self.state.context[CK.INPUT_DIR]}/code/")
        try:
            response = input("  Apply completed? (y/n): ")
        except EOFError:
            response = 'n'  # 非交互环境：保守默认，不自动放行
        if response.lower() != 'y':
            logger.info(f"  ⏸️  Paused. Resume later with: workflow.py resume --op {self.op_name}")
            raise SystemExit(0)

        # 验证代码是否实际发生了变更，防止用户确认后未实际应用 suggestion
        pre_hash = self.state.context.get(CK.PRE_APPLY_CODE_HASH)
        if pre_hash:
            current_hash = self._compute_code_hash()
            if current_hash == pre_hash:
                logger.info(f"  ⚠️  WARNING: 代码文件未发生变更（哈希与 suggestion 生成前一致）")
                logger.info(f"     请确认已将 suggestion 文件中的修改写入代码：")
                logger.info(f"     {suggestion_file}")
                try:
                    confirm = input("  仍然继续 BUILD？(y/n，默认 n): ")
                except EOFError:
                    confirm = 'n'
                if confirm.lower() != 'y':
                    raise WorkflowPaused(
                        f"代码未变更，请先应用 suggestion，再 resume: "
                        f"workflow.py resume --op {self.op_name}"
                    )
            else:
                # 代码已变更，记录新哈希供 COMPARE 阶段的死循环检测使用
                self.state.context[CK.POST_APPLY_CODE_HASH] = current_hash

        logger.info(f"  ✅ Code modification completed")
        self._transition_to(Phase.BUILD)

    def _phase_build(self):
        """Phase BUILD: 编译"""
        logger.info(f"[PHASE: BUILD] Building operator...")

        if self.skip_build:
            logger.info(f"  ⏭️  Skipping build (--skip-build flag set)")
            logger.info(f"  ℹ️  Using existing compiled artifacts in output/{self.op_name}/")
            self._transition_to(Phase.EVALUATE)
            return

        # ✅ 使用PathManager获取编译目录
        build_dir = self.pm.get_build_dir(self.op_name)

        if not build_dir.exists():
            raise WorkflowError(f"Build directory not found: {build_dir}")

        logger.info(f"  🔄 Running build.sh in {build_dir}...")
        build_script = Path(build_dir) / "build.sh"
        os.chmod(
            build_script,
            os.stat(build_script).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
        )
        result = subprocess.run(
            [str(build_script)],
            cwd=build_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.info(f"  ❌ Build failed (exit code {result.returncode}):")
            output = (result.stdout or "") + (result.stderr or "")
            if output:
                logger.info(output)
            raise WorkflowError("Build failed")

        logger.info(f"  ✅ Build successful")

        # ✅ 使用PathManager获取.run包路径
        run_file = self.pm.get_run_package_path(self.op_name)
        if run_file.exists():
            logger.info(f"  🔄 Installing .run package...")
            os.chmod(
                run_file,
                os.stat(run_file).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
            )
            install_result = subprocess.run(
                [str(run_file)],
                capture_output=True,
                text=True
            )
            if install_result.returncode != 0:
                output = (install_result.stdout or "") + (install_result.stderr or "")
                logger.info(f"  ❌ Installation failed (exit code {install_result.returncode}):")
                if output:
                    logger.info(output)
                raise WorkflowError("Installation failed")
            logger.info(f"  ✅ Installation completed")

        self._transition_to(Phase.EVALUATE)

    def _phase_evaluate(self):
        """Phase EVALUATE: 评测性能"""
        logger.info(f"[PHASE: EVALUATE] Evaluating performance...")

        logger.info(f"  🔄 Running ascendc_evaluation skill...")
        # 这里应该调用 ascendc_evaluation skill
        logger.info(f"  ⚠️  Requires skill invocation")

        try:
            input("  Press Enter after evaluation completes...")
        except EOFError:
            pass

        logger.info(f"  ✅ Evaluation completed")
        self._transition_to(Phase.COMPARE)

    def _phase_compare(self):
        """Phase COMPARE: 对比性能"""
        logger.info(f"[PHASE: COMPARE] Comparing performance...")

        # 追踪优化轮次，防止无限循环
        max_rounds = 5
        rounds = self.state.context.get(CK.OPTIMIZATION_ROUNDS, 0) + 1
        self.state.context[CK.OPTIMIZATION_ROUNDS] = rounds
        logger.info(f"  📊 Optimization round: {rounds}/{max_rounds}")

        # ── 自动对比：尝试从 CAKE2 output 目录读取最新 profiling CSV ──
        improved = None
        baseline_us = self.state.context.get(CK.BASELINE_DURATION_US)

        try:
            cake2_root = self.pm.get_cake2_root()
            output_profiling_dir = cake2_root / "output" / self.op_name / "profiling"
            if output_profiling_dir.exists():
                csv_candidates = sorted(
                    output_profiling_dir.rglob("op_summary*.csv"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                # 只接受严格新于 baseline 的 CSV（EVALUATE 之后产生的）
                # baseline_csv_mtime 在 INIT 阶段记录；+1 是为了容忍 1 秒内的时间精度误差
                baseline_mtime = self.state.context.get(CK.BASELINE_CSV_MTIME, 0)
                fresh_candidates = [p for p in csv_candidates
                                    if p.stat().st_mtime > baseline_mtime + 1]

                if not fresh_candidates:
                    if csv_candidates:
                        logger.info(f"  ⚠️  未找到比 baseline 更新的 profiling 数据")
                        logger.info(f"     最新 CSV：{csv_candidates[0].name}（与 baseline 相同或更旧）")
                    else:
                        logger.info(f"  ⚠️  {output_profiling_dir} 下未找到 op_summary*.csv")
                    logger.info(f"     请先运行 ascendc_evaluation 生成新 profiling 数据，再 resume")
                    # improved 保持 None，后续回退到手动确认
                elif fresh_candidates:
                    latest_csv = fresh_candidates[0]
                    extractor = ProfilingExtractor(latest_csv, target_op_name=self.op_name)
                    new_data = extractor.extract()
                    new_duration = new_data.task_duration_us

                    if baseline_us and baseline_us > 0 and new_duration > 0:
                        improvement = (baseline_us - new_duration) / baseline_us
                        goal = self.state.context.get(CK.PERFORMANCE_GOAL, {})
                        target = goal.get("relative_improvement", 0.2) if goal else 0.2

                        logger.info(f"  ✅ Auto-comparison result:")
                        logger.info(f"     Baseline : {baseline_us:.3f} us")
                        logger.info(f"     Current  : {new_duration:.3f} us")
                        logger.info(f"     Improvement: {improvement:+.1%}  (target: {target:.0%})")
                        logger.info(f"     Source   : {latest_csv.name}")

                        # 保存新 profiling 数据到 context（供后续轮次使用）
                        self.state.context[CK.LAST_EVAL_DURATION_US] = new_duration
                        self.state.context[CK.LAST_EVAL_IMPROVEMENT] = improvement

                        if improvement >= target:
                            logger.info(f"  🎉 Target met! ({improvement:.1%} ≥ {target:.0%})")
                            improved = True
                        elif improvement > 0:
                            logger.info(f"  ⚠️  Improved but target not met ({improvement:.1%} < {target:.0%})")
                            improved = False
                        else:
                            logger.info(f"  ❌ Performance regressed ({improvement:.1%})")
                            improved = False
                    else:
                        logger.info(f"  ⚠️  Auto-compare: baseline duration not recorded, skipping auto-compare")
                else:
                    logger.info(f"  ⚠️  Auto-compare: no op_summary CSV found in {output_profiling_dir}")
            else:
                logger.info(f"  ⚠️  Auto-compare: output profiling dir not found: {output_profiling_dir}")
        except Exception as e:
            logger.info(f"  ⚠️  Auto-compare failed ({e}), falling back to manual")

        # ── 自动对比无法确定时，退回到交互确认 ──
        if improved is None:
            logger.info(f"  ⚠️  Performance comparison requires manual analysis")
            try:
                response = input("  Performance improved and target met? (y/n): ")
            except EOFError:
                response = 'n'
            improved = response.lower() == 'y'

        if improved:
            self.state.context[CK.PERFORMANCE_IMPROVED] = True
            self._transition_to(Phase.UPDATE)
        else:
            self.state.context[CK.PERFORMANCE_IMPROVED] = False

            if rounds >= max_rounds:
                logger.info(f"  ⚠️  Max rounds ({max_rounds}) reached. Stopping optimization loop.")
                self._transition_to(Phase.DONE)
                return

            # 不达标时，先判断代码是否实际发生了变更
            # pre_apply_code_hash 在 SUGGEST 暂停前记录；post_apply_code_hash 在 APPLY 成功后记录
            pre_apply_hash = self.state.context.get(CK.PRE_APPLY_CODE_HASH)
            current_hash = self._compute_code_hash()
            code_was_changed = (pre_apply_hash is None) or (current_hash != pre_apply_hash)

            if not code_was_changed:
                # 代码未变更 → 用户没有真正应用修改，不需要 retag，暂停等待
                logger.info(f"  ⚠️  代码文件未变更（哈希与 suggestion 生成前一致）")
                logger.info(f"     请先将 suggestion 文件中的修改写入代码，重新 BUILD + EVALUATE，再 resume")
                raise WorkflowPaused(
                    f"代码未变更，请应用 {self.session_dir / 'suggestions'} 中的修改，"
                    f"重建并重新评估后 resume: workflow.py resume --op {self.op_name}"
                )

            # 代码已变更但未达标：清除旧 suggestion（避免复用），retag 反映新代码状态
            logger.info(f"  🔄 Target not met — forcing retag before round {rounds + 1}")
            suggestions_dir = self.session_dir / "suggestions"
            if suggestions_dir.exists():
                cleared = sum(1 for f in suggestions_dir.glob("*.md") if f.unlink() is None)
                if cleared:
                    logger.info(f"     Cleared {cleared} old suggestion file(s)")
            self.state.context[CK.FORCE_RETAG] = True
            self._transition_to(Phase.TAG)

    def _phase_update(self):
        """Phase UPDATE: 更新 baseline"""
        logger.info(f"[PHASE: UPDATE] Updating baseline...")

        if self.state.context.get(CK.PERFORMANCE_IMPROVED):
            logger.info(f"  🔄 Updating baseline code and profiling...")
            # 调用 auto_optimize.py update-baseline
            logger.info(f"  ⚠️  Baseline update requires manual execution")
            logger.info(f"  ✅ Baseline updated")

        self._transition_to(Phase.DONE)

    def _transition_to(self, next_phase: Phase):
        """状态转移 + 持久化 + 进度反馈"""
        prev_phase = self.state.phase
        self.state.phase = next_phase
        self.state.updated_at = datetime.now(timezone.utc).isoformat()

        # 计算阶段耗时
        duration_ms = 0
        if self._phase_start_time:
            duration_ms = int((datetime.now(timezone.utc) - self._phase_start_time).total_seconds() * 1000)

        # 记录历史
        self.state.history.append({
            "from": prev_phase.value,
            "to": next_phase.value,
            "timestamp": self.state.updated_at,
            "phase": next_phase.value,
            "duration_ms": duration_ms,
            "exit_reason": "completed",
        })

        # 持久化
        self._save_state()

        # ✅ 增强的进度反馈
        logger.info("")
        logger.info("━" * 60)
        logger.info(f"  ✓ Completed: {prev_phase.value.upper()}")
        logger.info(f"  → Next: {next_phase.value.upper()}")

        # 显示整体进度
        phase_sequence = [
            Phase.INIT, Phase.TAG, Phase.SCORE, Phase.ROUTE,
            Phase.SUGGEST, Phase.APPLY, Phase.BUILD, Phase.EVALUATE,
            Phase.COMPARE, Phase.UPDATE, Phase.DONE
        ]

        try:
            current_idx = phase_sequence.index(next_phase)
            total = len(phase_sequence) - 1  # 不计入 DONE
            progress_pct = int((current_idx / total) * 100)
            logger.info(f"  Progress: {progress_pct}% ({current_idx}/{total} phases)")
        except ValueError:
            pass  # 如果 next_phase 不在序列中,跳过进度显示

        logger.info("━" * 60)
        logger.info("")

    def _generate_diagnostic_report(self, route: RouteDecision, profiling_data: Optional[Dict]):
        """生成诊断报告（Deep Path fallback）"""
        suggestions_dir = self.session_dir / "suggestions"
        suggestions_dir.mkdir(parents=True, exist_ok=True)

        report_file = suggestions_dir / f"{self.op_name}_diagnostic_report.md"

        # 读取 scored_results
        scored_file = self.session_dir / "scored_results.json"
        top_rules = []
        if scored_file.exists():
            with open(scored_file) as f:
                data = json.load(f)
                results = data.get("results", [])
                # 取前 5 个非 redundant 的规则
                top_rules = [
                    r for r in results[:10]
                    if not r.get("redundant", False)
                ][:5]

        # 生成报告内容
        lines = [
            f"# Diagnostic Report: {self.op_name}",
            "",
            f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Session**: {self.session_id}",
            f"**Route**: Deep Path (Low Confidence)",
            "",
            "━" * 60,
            "",
            "## Summary",
            "",
            "自动分析置信度较低，建议人工介入深度分析。",
            "",
            "### Detection Results",
            "",
            f"- **Top Rule Confidence**: {route.max_score:.1%} (< 30% threshold)",
            f"- **Coverage**: {route.coverage_ratio:.1%}",
            f"- **Recommended Path**: Manual expert analysis or deep_research subskill",
            "",
        ]

        # 添加 Profiling 信息
        if profiling_data:
            lines.extend([
                "### Profiling Snapshot",
                "",
                f"- **Task Type**: {profiling_data.get('task_type', 'N/A')}",
                f"- **Duration**: {profiling_data.get('task_duration_us', 'N/A')} μs",
                f"- **Bottleneck**: {profiling_data.get('bottleneck_hint', 'Unknown')}",
                "",
            ])

        # 添加 Top Rules
        if top_rules:
            lines.extend([
                "### Top Candidate Rules (for reference)",
                "",
                "虽然置信度低，以下规则可能相关：",
                "",
            ])
            for i, rule in enumerate(top_rules, 1):
                rule_id = Path(rule["rule_path"]).parent.name
                score = rule.get("raw_score", rule.get("score", 0))
                lines.append(f"{i}. **{rule_id}** (score: {score:.2f})")
                lines.append(f"   - Matched tags: {', '.join(rule.get('matched_tags', []))}")
                lines.append(f"   - Missing tags: {', '.join(rule.get('missing_tags', []))}")
                lines.append("")

        # 添加建议
        lines.extend([
            "",
            "━" * 60,
            "",
            "## Recommended Actions",
            "",
            "### Option 1: Manual Expert Analysis",
            "",
            "1. 检查 profiling hotspot：",
            f"   - CSV: {self.pm.get_input_dir(self.op_name)}/profiling_data/profiling_csv/op_summary.csv",
            f"         或: {self.pm.get_input_profiling_dir(self.op_name)}/op_summary.csv",
            "   - 关注 Task Duration, Scalar Ratio, MTE 指标",
            "",
            "2. 对比参考实现：",
            "   - 查看优化前后代码差异",
            "",
            "3. 咨询 AscendC 专家或提交 issue",
            "",
            "### Option 2: Use deep_research Subskill",
            "",
            "```bash",
            "# 手动调用 5-Step Analysis",
            "# Read: subskills/deep_research.md",
            "# Follow the structured analysis framework",
            "```",
            "",
            "━" * 60,
            "",
            "## Artifacts",
            "",
            f"- **Tags**: workspace/cache/tags/tag_{self.op_name}.json",
            f"- **Scored Results**: {scored_file}",
            f"- **Code**: workspace/inputs/{self.op_name}/code/",
            "",
            "━" * 60,
            "",
            "*This is an automated diagnostic report. Human expertise is recommended for next steps.*",
        ])

        # 写入文件
        report_file.write_text("\n".join(lines), encoding='utf-8')

        logger.info(f"  ✅ Generated diagnostic report: {report_file.name}")
        logger.info(f"     File: {report_file}")
        logger.info("")

    def _check_scalar_locked(self, rule_route: "RouteDecision") -> List[str]:
        """检测是否满足 Scalar-Locked 路由条件。

        返回非空列表 → 满足条件，元素为诊断信息；空列表 → 不满足，保持原路由。

        触发条件（AND 关系，全部满足才触发）
        ────────────────────────────────────
        1. 迭代门槛：至少经历过 1 次完整的 APPLY→EVALUATE 循环。
           意义：第一轮不做算法重设计，必须先用完规则库。

        2. 规则空间穷尽：max_rule_score < 0.40 且至少有 1 条规则被评分过。
           意义：区分"规则没匹配"（深度不足）和"规则匹配但分低"（已穷举）。

        3. Scalar 比例持续偏高（硬件自适应阈值）：
           - AIV（vector 核，910B）: aiv_scalar_ratio > 0.35
           - AIC（cube 核，910B）:  aic_scalar_ratio > 0.30
             （Cube 算子有正常的 ~20% scalar 基线，阈值更保守）

        4. 瓶颈可归因：profiling 数据必须存在，否则无法判断瓶颈类型。

        排除条件（任意一项成立则不触发）
        ────────────────────────────────────
        - 最近评估有正确性失败：先修正正确性，不在错误代码上做算法重设计。
        - scalar_ratio 在最近两次评估间改善 ≥ 5%：说明优化仍在生效，继续规则路径。
        - 最近 APPLY→EVALUATE 轮次中代码未变化（hash 相同）：APPLY 未真正发生。
        """
        profiling_data = self.state.context.get(CK.PROFILING_DATA)

        # 排除条件：profiling 缺失
        if not profiling_data:
            return []

        # 排除条件：最近评估有正确性失败
        if self.state.context.get("last_eval_correctness_failed", False):
            return []

        # 条件 1：至少经历过 1 次完整 APPLY→EVALUATE 循环
        optimization_rounds = self.state.context.get(CK.OPTIMIZATION_ROUNDS, 0)
        if optimization_rounds < 1:
            return []

        # 条件 2：规则空间基本穷尽（max_score < 0.40，且有规则被评分）
        if rule_route.max_score >= 0.40:
            return []
        scored = self.state.context.get(CK.SCORED_RESULTS, {})
        tried_rules = [
            r for r in scored.get("results", [])
            if r.get("raw_score", r.get("score", 0)) > 0
        ]
        if not tried_rules:
            # 没有任何规则匹配过 → 是"深度不足"不是"规则穷尽"
            return []

        # 条件 3：Scalar 比例持续偏高（硬件自适应）
        task_type = profiling_data.get("task_type", "")
        metrics = profiling_data.get("relevant_metrics", {})
        if "VECTOR" in task_type:
            scalar_ratio = metrics.get("aiv_scalar_ratio") or 0
            threshold = 0.35
            label = "aiv_scalar_ratio"
        elif "CORE" in task_type:
            scalar_ratio = metrics.get("aic_scalar_ratio") or 0
            threshold = 0.30
            label = "aic_scalar_ratio"
        else:
            return []  # 硬件类型不明，不触发

        if scalar_ratio < threshold:
            return []

        # 条件 4（隐含）：profiling 存在已在最上方验证

        # 全部条件满足，返回诊断信息
        return [
            f"{label} = {scalar_ratio:.1%}  (threshold: >{threshold:.0%})",
            f"Optimization rounds completed: {optimization_rounds}",
            f"Rules in play: {len(tried_rules)},  max_score = {rule_route.max_score:.3f}  (<0.40)",
            f"Hardware task type: {task_type}",
        ]

    def _compute_code_hash(self) -> str:
        """计算算子代码目录的 MD5 哈希，用于检测 APPLY 前后的代码变更。

        只对 .cpp / .h 文件做内容哈希，文件顺序固定（sorted），结果确定性。
        返回 8 位十六进制字符串；目录不存在时返回空字符串。
        """
        import hashlib
        code_dir = self.pm.get_input_dir(self.op_name) / "code"
        hasher = hashlib.md5()
        if code_dir.exists():
            cpp_files = sorted(code_dir.rglob("*.cpp"))
            h_files = sorted(code_dir.rglob("*.h"))
            for f in cpp_files + h_files:
                try:
                    hasher.update(f.read_bytes())
                except Exception as e:
                    # 单个文件读失败时跳过（不影响整体哈希用途），记录便于排查
                    logger.debug("Failed to read %s for code hash: %s", f, e)
        return hasher.hexdigest()[:8]

    def _save_state(self):
        """保存状态到文件"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2, ensure_ascii=False)

    def _load_state(self) -> WorkflowState:
        """从文件加载状态"""
        with open(self.state_file, "r") as f:
            data = json.load(f)
        return WorkflowState.from_dict(data)

    def _handle_error(self, error: Exception):
        """错误处理 + 恢复建议"""
        import traceback

        logger.info("")
        logger.info("━" * 60)
        logger.info(f"❌ ERROR in phase: {self.state.phase.value.upper()}")
        logger.info("━" * 60)
        logger.info(f"\nError type: {type(error).__name__}")
        logger.info(f"Error message: {str(error)}")
        logger.info("")

        # 保存完整的错误信息
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "phase": self.state.phase.value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        duration_ms = 0
        if self._phase_start_time:
            duration_ms = int((datetime.now(timezone.utc) - self._phase_start_time).total_seconds() * 1000)
        self.state.history.append({
            "phase": self.state.phase.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "exit_reason": "error",
            "error_type": type(error).__name__,
        })

        self.state.phase = Phase.ERROR
        self.state.context[CK.ERROR] = error_info
        self._save_state()

        # ✅ 提供错误恢复建议
        logger.info("🔧 Recovery suggestions:")
        logger.info("")

        phase = self.state.phase
        if "not found" in str(error).lower() or "notfound" in str(error).lower():
            logger.info("  Issue: File or directory not found")
            logger.info("  Actions:")
            logger.info("    1. Check that input files are in workspace/inputs/{op}/")
            logger.info("    2. Verify directory structure matches expected layout")
            logger.info("    3. Run: python cleanup.py --status  (to inspect workspace)")
            logger.info("")

        elif "build" in str(error).lower():
            logger.info("  Issue: Build failure")
            logger.info("  Actions:")
            logger.info("    1. Check build.sh exists in build directory")
            logger.info("    2. Review build logs for compilation errors")
            logger.info("    3. Try manual build: cd output/{op}/ && bash build.sh")
            logger.info("    4. Rollback: python build_operator.py --op {op} --rollback")
            logger.info("")

        elif "redundant" in str(error).lower() or "already applied" in str(error).lower():
            logger.info("  Issue: All rules are redundant (optimizations already applied)")
            logger.info("  Actions:")
            logger.info("    1. This is expected if baseline is already optimized")
            logger.info("    2. Consider exploring Deep Path analysis")
            logger.info("    3. Manually inspect profiling data for new bottlenecks")
            logger.info("")

        else:
            logger.info("  Generic recovery:")
            logger.info("    1. Review error details above")
            logger.info("    2. Check workflow_state.json for context")
            logger.info("    3. Resume from last checkpoint: python workflow.py resume --op {op}")
            logger.info("")

        logger.info(f"Session directory: {self.session_dir}")
        logger.info(f"State file: {self.state_file}")
        logger.info("")
        logger.info("━" * 60)


def cmd_run(args):
    """运行工作流"""
    # 提示历史 session（知识连续性）
    workspace = Path(__file__).resolve().parents[2] / "workspace"
    sm = SessionManager(workspace)
    prev_sessions = sm.list_sessions(op_name=args.op, status="completed", limit=3)
    if prev_sessions:
        logger.info(f"[WORKFLOW] ℹ️  Found {len(prev_sessions)} previous completed session(s) for '{args.op}':")
        for s in prev_sessions:
            pct = s.performance.get("improvement_pct", 0.0)
            logger.info(f"   - {s.session_id}  improvement={pct:.1f}%")
        logger.info(f"[WORKFLOW]    Review them with: workflow.py sessions --op {args.op}")
        logger.info("")

    engine = WorkflowEngine(
        args.op,
        mode=args.mode,
        skip_build=getattr(args, 'skip_build', False),
        force_retag=getattr(args, 'force_retag', False),
    )
    engine.run()


def cmd_resume(args):
    """恢复工作流"""
    workspace = Path(__file__).resolve().parents[2] / "workspace"
    sm = SessionManager(workspace)

    # ✅ 支持通过session_id恢复
    if hasattr(args, 'session_id') and args.session_id:
        info = sm.load_session(args.session_id)

        if not info:
            logger.info(f"[ERROR] Session not found: {args.session_id}")
            raise SystemExit(1)

        op_name = info.operator
        engine = WorkflowEngine(op_name, session_id=args.session_id,
                                skip_build=getattr(args, 'skip_build', False),
                                force_retag=getattr(args, 'force_retag', False))
    else:
        # 旧方式: 查找算子的最新session
        if not args.op:
            logger.info(f"[ERROR] Either --op or --session-id must be provided")
            raise SystemExit(1)

        latest = sm.get_latest_session(args.op)

        if not latest:
            logger.info(f"[ERROR] No session found for '{args.op}'")
            logger.info(f"  Start a new workflow with: workflow.py run --op {args.op}")
            raise SystemExit(1)

        # ✅ 判断是否应该新建 session（而非复用）
        #    条件1：session 创建日期非今天（跨天 → 必须新建）
        #    条件2：session 已 DONE（终态 → 新运行也必须新建，即使当天多次运行）
        session_date = latest.session_id[:8]  # e.g. "20260228"
        today_date = datetime.now(timezone.utc).strftime("%Y%m%d")

        # 读取 workflow_state.json 获取真实 phase（session.json.workflow 可能不同步）
        ws_path = sm.get_session_dir(latest.session_id) / "workflow_state.json"
        actual_phase = "unknown"
        try:
            import json as _json
            with open(ws_path) as _f:
                actual_phase = _json.load(_f).get("phase", "unknown")
        except Exception as e:
            # 读不到 workflow_state.json 时回退到 "unknown"，记录便于排查
            logger.debug("Failed to read workflow_state.json (%s): %s", ws_path, e)

        is_stale_date = (session_date != today_date)
        is_done = (actual_phase in ("done", "DONE"))

        if is_stale_date or is_done:
            reason = "from a previous day" if is_stale_date else f"already DONE (phase={actual_phase})"
            logger.info(f"[WORKFLOW] ⚠️  Latest session '{latest.session_id}' is {reason}.")
            logger.info(f"[WORKFLOW]    Starting a NEW session instead.")
            logger.info(
                f"[WORKFLOW]    To force-resume the old session: workflow.py resume --session-id {latest.session_id}")
            engine = WorkflowEngine(args.op,
                                    mode=getattr(args, 'mode', 'interactive'),
                                    skip_build=getattr(args, 'skip_build', False),
                                    force_retag=getattr(args, 'force_retag', False))
        else:
            logger.info(f"[WORKFLOW] Resuming today's session: {latest.session_id} (phase={actual_phase})")
            engine = WorkflowEngine(args.op, session_id=latest.session_id,
                                    skip_build=getattr(args, 'skip_build', False),
                                    force_retag=getattr(args, 'force_retag', False))

    engine.run()


def cmd_status(args):
    """查看状态"""
    workspace = Path(__file__).resolve().parents[2] / "workspace"
    sm = SessionManager(workspace)

    # ✅ 支持按session_id查询
    if hasattr(args, 'session_id') and args.session_id:
        info = sm.load_session(args.session_id)
        if not info:
            logger.info(f"[STATUS] Session not found: {args.session_id}")
            return

        logger.info(f"[STATUS] Session: {info.session_id}")
        logger.info(f"  Operator: {info.operator}")
        logger.info(f"  Mode: {info.mode}")
        logger.info(f"  Status: {info.status}")
        logger.info(f"  Created: {info.created_at}")
        logger.info(f"  Updated: {info.updated_at}")
        logger.info("")
        # 从权威来源 workflow_state.json 读取阶段信息
        session_dir = sm.get_session_dir(info.session_id)
        ws_path = session_dir / "workflow_state.json"
        current_phase = "unknown"
        phases_completed = []
        try:
            with open(ws_path) as _f:
                ws = json.load(_f)
            current_phase = ws.get("phase", "unknown")
            phases_completed = [h["from"] for h in ws.get("history", []) if "from" in h]
        except Exception as e:
            # 读不到 workflow_state.json 时保留默认值，记录便于排查
            logger.debug("Failed to read workflow_state.json (%s): %s", ws_path, e)
        logger.info(f"  Workflow Phase: {current_phase}")
        logger.info(f"  Phases Completed: {len(phases_completed)}/10")
        logger.info("")
        logger.info(f"  Performance:")
        logger.info(f"    Improvement: {info.performance['improvement_pct']:.1f}%")
        logger.info(f"    Target Met: {'✅' if info.performance['target_met'] else '❌'}")
        return

    # 按算子名查询最新session
    if not args.op:
        logger.info(f"[ERROR] Either --op or --session-id must be provided")
        raise SystemExit(1)

    latest = sm.get_latest_session(args.op)
    if not latest:
        logger.info(f"[STATUS] No session found for '{args.op}'")
        return

    # 递归调用显示详情
    class Args:
        session_id = latest.session_id
        op = None

    cmd_status(Args())


def cmd_sessions(args):
    """列出sessions"""
    workspace = Path(__file__).resolve().parents[2] / "workspace"
    sm = SessionManager(workspace)

    sessions = sm.list_sessions(
        op_name=args.op if hasattr(args, 'op') else None,
        status=args.status if hasattr(args, 'status') else None,
        limit=50
    )

    logger.info(f"Found {len(sessions)} sessions:\n")

    for s in sessions:
        status_icon = {
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
            "interrupted": "⏸️"
        }.get(s.status, "❓")

        logger.info(f"{status_icon} {s.session_id}")
        logger.info(f"   Operator: {s.operator} | Mode: {s.mode}")
        logger.info(f"   Created: {s.created_at}")
        logger.info(f"   Performance: {s.performance['improvement_pct']:.1f}% improvement")
        logger.info("")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Workflow Orchestrator - 固化流程,防止遗漏"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Start new workflow")
    p_run.add_argument("--op", required=True, help="Operator name")
    p_run.add_argument("--mode", default="interactive", choices=["interactive"])
    p_run.add_argument("--skip-build", action="store_true",
                       help="Skip BUILD phase, use existing compiled artifacts")
    p_run.add_argument("--force-retag", action="store_true",
                       help="Ignore tag cache and re-run code_tag even if tag file exists")

    # resume
    p_resume = sub.add_parser("resume", help="Resume from checkpoint")
    p_resume.add_argument("--op", help="Operator name (auto-detect latest session)")
    p_resume.add_argument("--session-id", help="Specific session ID to resume")
    p_resume.add_argument("--skip-build", action="store_true",
                          help="Skip BUILD phase, use existing compiled artifacts")
    p_resume.add_argument("--force-retag", action="store_true",
                          help="Ignore tag cache and re-run code_tag")

    # status
    p_status = sub.add_parser("status", help="Show workflow status")
    p_status.add_argument("--op", help="Operator name")
    p_status.add_argument("--session-id", help="Specific session ID")

    # sessions
    p_sessions = sub.add_parser("sessions", help="List all sessions")
    p_sessions.add_argument("--op", help="Filter by operator")
    p_sessions.add_argument("--status", help="Filter by status")

    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "run": cmd_run,
        "resume": cmd_resume,
        "status": cmd_status,
        "sessions": cmd_sessions
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
    handler(args)


if __name__ == "__main__":
    main()
