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
Session管理器 - 并发与多算子支持

设计原则:
1. 每次workflow运行生成唯一session_id
2. 完全隔离不同session的数据
3. 支持历史追溯和清理策略
4. 并发安全(无文件竞争)
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Session元数据"""
    session_id: str
    operator: str
    mode: str
    created_at: str
    updated_at: str
    status: str  # running / completed / failed / interrupted

    input_baseline: Dict
    resources: Dict
    performance: Dict
    iterations: Dict
    user_info: Dict

    @classmethod
    def from_dict(cls, data: dict) -> SessionInfo:
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


class SessionManager:
    """
    Session管理器 - 支持并发和多算子

    主要功能:
    1. 生成唯一session_id
    2. 创建隔离的session目录
    3. 管理session生命周期
    4. 清理过期session
    5. 查询历史session
    """

    def __init__(self, workspace_root: Path):
        """
        初始化Session管理器

        Args:
            workspace_root: workspace根目录
        """
        self.workspace_root = workspace_root
        self.sessions_dir = workspace_root / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # 加载配置
        self.config = self._load_config()

    # ========== Session创建 ==========

    def create_session(
        self,
        op_name: str,
        mode: str,
        input_baseline_dir: Path,
        user_info: Optional[Dict] = None
    ) -> tuple[str, Path]:
        """
        创建新session

        Args:
            op_name: 算子名
            mode: 运行模式(auto/interactive)
            input_baseline_dir: 输入baseline目录
            user_info: 用户信息(可选)

        Returns:
            (session_id, session_dir)
        """
        # 1. 生成session_id
        session_id = self._generate_session_id(op_name, mode)

        # 2. 创建session目录
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # 3. 创建子目录
        (session_dir / "working_code").mkdir(exist_ok=True)
        (session_dir / "baseline_snapshot").mkdir(exist_ok=True)
        (session_dir / "suggestions").mkdir(exist_ok=True)
        (session_dir / "iterations").mkdir(exist_ok=True)
        (session_dir / "reports").mkdir(exist_ok=True)

        # 4. 计算输入baseline的hash
        code_hash = self._compute_code_hash(input_baseline_dir / "code")

        # 5. 创建session元数据
        now = datetime.now(timezone.utc).isoformat()
        session_info = SessionInfo(
            session_id=session_id,
            operator=op_name,
            mode=mode,
            created_at=now,
            updated_at=now,
            status="running",
            input_baseline={
                "code_snapshot": str(input_baseline_dir / "code"),
                "code_hash": code_hash,
                "profiling_snapshot": str(input_baseline_dir / "profiling"),
                "profiling_timestamp": self._get_profiling_timestamp(input_baseline_dir)
            },
            resources={
                "tag_file": "",
                "scored_results": str(session_dir / "scored_results.json"),
                "workflow_state": str(session_dir / "workflow_state.json")
            },
            performance={
                "initial_duration_us": 0.0,
                "current_duration_us": 0.0,
                "improvement_pct": 0.0,
                "target_improvement_pct": 20.0,
                "target_met": False
            },
            iterations={
                "total": 0,
                "successful": 0,
                "failed": 0
            },
            user_info=user_info or {}
        )

        # 6. 保存元数据
        self._save_session_info(session_dir, session_info)

        # 7. 保存不可变 baseline 快照（用于回溯/恢复）
        self._snapshot_input_baseline(input_baseline_dir, session_dir)

        # 8. 复制 baseline 到 working_code（可变工作副本）
        self._copy_baseline_code(input_baseline_dir, session_dir)

        logger.info(f"✅ Session created: {session_id}")
        logger.info(f"   Directory: {session_dir}")

        return session_id, session_dir

    # ========== Session查询 ==========

    def load_session(self, session_id: str) -> Optional[SessionInfo]:
        """加载session元数据"""
        session_dir = self.sessions_dir / session_id
        session_file = session_dir / "session.json"

        if not session_file.exists():
            return None

        with open(session_file) as f:
            data = json.load(f)
        return SessionInfo.from_dict(data)

    def get_session_dir(self, session_id: str) -> Path:
        """获取session目录"""
        return self.sessions_dir / session_id

    def list_sessions(
        self,
        op_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[SessionInfo]:
        """
        列出sessions

        Args:
            op_name: 筛选算子名
            status: 筛选状态
            limit: 最多返回数量

        Returns:
            Session列表(按创建时间倒序)
        """
        sessions = []

        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue

            info = self.load_session(session_dir.name)
            if not info:
                continue

            if op_name and info.operator != op_name:
                continue
            if status and info.status != status:
                continue

            sessions.append(info)

        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions[:limit]

    def get_latest_session(self, op_name: str) -> Optional[SessionInfo]:
        """获取算子的最新session"""
        sessions = self.list_sessions(op_name=op_name, limit=1)
        return sessions[0] if sessions else None

    # ========== Session清理 ==========

    def cleanup_old_sessions(self, dry_run: bool = False) -> Dict:
        """
        清理过期sessions

        Args:
            dry_run: 只模拟,不实际删除

        Returns:
            清理统计信息
        """
        if not self.config["retention"]["auto_cleanup"]:
            return {"skipped": "auto_cleanup disabled"}

        now = datetime.now(timezone.utc)
        retention = self.config["retention"]

        stats = {
            "total_scanned": 0,
            "to_delete": [],
            "kept": [],
            "errors": []
        }

        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue

            stats["total_scanned"] += 1

            info = self.load_session(session_dir.name)
            if not info:
                stats["errors"].append(f"Failed to load: {session_dir.name}")
                continue

            # 计算session年龄
            updated_at = datetime.fromisoformat(info.updated_at)
            age_days = (now - updated_at).days

            # 根据状态判断是否删除
            should_delete = False

            if info.status == "completed":
                if age_days > retention["keep_completed_days"]:
                    should_delete = True
            elif info.status == "failed":
                if age_days > retention["keep_failed_days"]:
                    should_delete = True
            elif info.status == "interrupted":
                if age_days > retention["keep_interrupted_days"]:
                    should_delete = True

            if should_delete:
                stats["to_delete"].append({
                    "session_id": info.session_id,
                    "operator": info.operator,
                    "status": info.status,
                    "age_days": age_days
                })

                if not dry_run:
                    shutil.rmtree(session_dir)
            else:
                stats["kept"].append(info.session_id)

        stats["deleted_count"] = len(stats["to_delete"])
        stats["kept_count"] = len(stats["kept"])

        return stats

    # ========== Session更新 ==========

    def update_session_status(
        self,
        session_id: str,
        status: str,
        performance: Optional[Dict] = None,
        iterations: Optional[Dict] = None
    ):
        """更新session状态"""
        info = self.load_session(session_id)
        if not info:
            raise ValueError(f"Session not found: {session_id}")

        info.status = status
        info.updated_at = datetime.now(timezone.utc).isoformat()

        if performance:
            info.performance.update(performance)

        if iterations:
            info.iterations.update(iterations)

        session_dir = self.get_session_dir(session_id)
        self._save_session_info(session_dir, info)

    def update_session_resources(self, session_id: str, resources: Dict):
        """更新session资源引用（如 output_dir）"""
        info = self.load_session(session_id)
        if not info:
            return
        info.resources.update(resources)
        info.updated_at = datetime.now(timezone.utc).isoformat()
        session_dir = self.get_session_dir(session_id)
        self._save_session_info(session_dir, info)

    # ========== 内部辅助方法 ==========

    def _load_config(self) -> Dict:
        """加载session配置"""
        config_file = self.workspace_root.parent / "assets" / "configs" / "session_config.yaml"

        if not config_file.exists():
            return self._default_config()

        import yaml
        try:
            with open(config_file) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.info(f"⚠️  Failed to load session config ({e}); using defaults")
            return self._default_config()

    @staticmethod
    def _default_config() -> Dict:
        """默认配置"""
        return {
            "retention": {
                "keep_completed_days": 30,
                "keep_failed_days": 7,
                "keep_interrupted_days": 3,
                "auto_cleanup": True
            },
            "limits": {
                "max_sessions_per_operator": 50,
                "max_total_sessions": 200
            }
        }

    @staticmethod
    def _generate_session_id(op_name: str, mode: str) -> str:
        """生成唯一session_id"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        uuid_short = uuid.uuid4().hex[:4]
        return f"{timestamp}_{op_name}_{mode}_{uuid_short}"

    @staticmethod
    def _compute_code_hash(code_dir: Path) -> str:
        """计算代码目录的hash"""
        if not code_dir.exists():
            return "no_code"

        files = sorted(code_dir.rglob("*.cpp")) + sorted(code_dir.rglob("*.h"))
        content = b""
        for f in files:
            try:
                content += f.read_bytes()
            except Exception as e:
                logger.warning(f"Failed to read {f} for code hash: {e}")

        hash_obj = hashlib.sha256(content)
        return f"sha256:{hash_obj.hexdigest()[:16]}"

    @staticmethod
    def _get_profiling_timestamp(baseline_dir: Path) -> str:
        """获取profiling数据的时间戳"""
        # v2 layout: workspace/inputs/<op>/profiling/op_summary*.csv
        profiling_dir = baseline_dir / "profiling"
        if not profiling_dir.exists():
            return "no_profiling"

        csv_files = list(profiling_dir.rglob("*.csv"))
        if not csv_files:
            return "no_profiling"

        latest = max(csv_files, key=lambda f: f.stat().st_mtime)
        return datetime.fromtimestamp(latest.stat().st_mtime).isoformat()

    @staticmethod
    def _snapshot_input_baseline(baseline_dir: Path, session_dir: Path) -> None:
        """在 session 下保存不可变 baseline 快照（code + profiling）。"""
        snapshot_dir = session_dir / "baseline_snapshot"
        src_code = baseline_dir / "code"
        src_profiling = baseline_dir / "profiling"

        # Code snapshot
        dst_code = snapshot_dir / "code"
        if dst_code.exists():
            shutil.rmtree(dst_code)
        if src_code.exists():
            shutil.copytree(src_code, dst_code)

        # Profiling snapshot (optional)
        dst_profiling = snapshot_dir / "profiling"
        if dst_profiling.exists():
            shutil.rmtree(dst_profiling)
        if src_profiling.exists():
            shutil.copytree(src_profiling, dst_profiling)

    @staticmethod
    def _copy_baseline_code(baseline_dir: Path, session_dir: Path):
        """复制baseline代码到session工作区"""
        src_code = baseline_dir / "code"
        dst_code = session_dir / "working_code"

        if src_code.exists():
            if dst_code.exists():
                shutil.rmtree(dst_code)
            shutil.copytree(src_code, dst_code)

    @staticmethod
    def _save_session_info(session_dir: Path, info: SessionInfo):
        """保存session元数据"""
        session_file = session_dir / "session.json"
        with open(session_file, "w") as f:
            json.dump(info.to_dict(), f, indent=2, ensure_ascii=False)


# ========== CLI工具 ==========

def cmd_cleanup(args):
    """清理过期sessions"""
    workspace = Path("workspace")
    sm = SessionManager(workspace)

    logger.info("🧹 Cleaning up old sessions...")
    logger.info(f"   Dry run: {args.dry_run}\n")

    stats = sm.cleanup_old_sessions(dry_run=args.dry_run)

    if "skipped" in stats:
        logger.info(f"⚠️  {stats['skipped']}")
        return

    logger.info(f"Scanned: {stats['total_scanned']} sessions")
    logger.info(f"To delete: {stats['deleted_count']}")
    logger.info(f"Kept: {stats['kept_count']}")

    if stats["to_delete"]:
        logger.info("\nSessions to delete:")
        for item in stats["to_delete"][:10]:  # 最多显示10个
            logger.info(f"  - {item['session_id']} ({item['operator']}, {item['status']}, {item['age_days']} days old)")

        if len(stats["to_delete"]) > 10:
            logger.info(f"  ... and {len(stats['to_delete']) - 10} more")

    if stats["errors"]:
        logger.info(f"\n⚠️  Errors: {len(stats['errors'])}")
        for err in stats["errors"][:5]:
            logger.info(f"  - {err}")

    if not args.dry_run and stats["deleted_count"] > 0:
        logger.info(f"\n✅ Deleted {stats['deleted_count']} sessions")


def cmd_list(args):
    """列出sessions"""
    workspace = Path("workspace")
    sm = SessionManager(workspace)

    sessions = sm.list_sessions(
        op_name=args.op,
        status=args.status,
        limit=args.limit
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


def cmd_info(args):
    """显示session详细信息"""
    workspace = Path("workspace")
    sm = SessionManager(workspace)

    info = sm.load_session(args.session_id)
    if not info:
        logger.info(f"❌ Session not found: {args.session_id}")
        return

    logger.info(json.dumps(info.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    parser = argparse.ArgumentParser(description="Session管理工具")
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="列出sessions")
    p_list.add_argument("--op", help="筛选算子")
    p_list.add_argument("--status", help="筛选状态")
    p_list.add_argument("--limit", type=int, default=50, help="最多显示数量")

    # cleanup
    p_cleanup = sub.add_parser("cleanup", help="清理过期sessions")
    p_cleanup.add_argument("--dry-run", action="store_true", help="只模拟,不删除")

    # info
    p_info = sub.add_parser("info", help="显示session详情")
    p_info.add_argument("session_id", help="Session ID")

    args = parser.parse_args()

    commands = {
        "list": cmd_list,
        "cleanup": cmd_cleanup,
        "info": cmd_info
    }

    handler = commands[args.command]
    handler(args)
