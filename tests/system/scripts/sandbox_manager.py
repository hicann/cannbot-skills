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
SandboxManager - 管理测试用例的隔离沙箱

为每个测试用例创建独立的执行环境（沙箱），确保用例间文件系统状态不互相干扰。
"""

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def _rmtree_ignore_errors(path: Path) -> None:
    """删除目录树，忽略所有错误（用于清理失败的沙箱）。"""
    shutil.rmtree(str(path), ignore_errors=True)


class SandboxManager:
    """
    管理测试用例的隔离沙箱

    沙箱目录结构：
    tests/system/sandboxes/
    ├── <skill_name>_eval_<id>/
    │   ├── .opencode/skills/<skill_name>/  # skill 独立副本（opencode 自动加载）
    │   └── logs/                           # 该用例的 session 日志
    """

    SANDBOX_DIR_NAME = "sandboxes"

    def __init__(self, framework_dir: Path, use_symlink: bool = True):
        """
        初始化沙箱管理器

        Args:
            framework_dir: tests/system 目录路径
            use_symlink: 是否使用软链接代替复制（默认 True，可通过 SKILL_SANDBOX_COPY=1 切回复制模式）
        """
        self.framework_dir = Path(framework_dir)
        self.sandbox_root = self.framework_dir / self.SANDBOX_DIR_NAME
        self.use_symlink = use_symlink

    @staticmethod
    def cleanup_sandbox(sandbox_path: Path) -> None:
        """
        清理单个沙箱目录

        Args:
            sandbox_path: 沙箱目录路径
        """
        if sandbox_path.exists():
            shutil.rmtree(sandbox_path)

    @staticmethod
    def get_logs_dir(sandbox_path: Path) -> Path:
        """获取沙箱的 logs 目录路径"""
        return sandbox_path / "logs"

    def create_skill_link(self, sandbox_path: Path, skill_dir: Path) -> Path:
        """
        在沙箱的 .opencode/skills/ 下部署 skill 目录

        默认复制独立副本确保文件隔离；设置 use_symlink=True 后改用软链接指向源目录。

        Args:
            sandbox_path: 沙箱目录路径
            skill_dir: skill 源目录路径

        Returns:
            部署后的 skill 目录路径
        """
        skill_name = skill_dir.name
        link_path = sandbox_path / ".opencode" / "skills" / skill_name

        # 如果已存在则删除
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_dir() and not link_path.is_symlink():
                shutil.rmtree(link_path)
            else:
                link_path.unlink()

        abs_skill_dir = Path(skill_dir).resolve()
        link_path.parent.mkdir(parents=True, exist_ok=True)

        if self.use_symlink:
            link_path.symlink_to(abs_skill_dir, target_is_directory=True)
        else:
            shutil.copytree(abs_skill_dir, link_path)

        return link_path

    def ensure_sandbox_root(self) -> None:
        """确保沙箱根目录存在"""
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        logger.info("[Sandbox] 沙箱根目录: %s", self.sandbox_root)

    # opencode 工具权限白名单：允许评测必需的工具；external_directory 设为 allow
    # 因为 bash 已开放且 --dangerously-skip-permissions 已启用，deny 无实际安全收益
    # 但会阻塞 init.sh 转换为绝对路径的工作流模板读取（见 init.sh sed 替换逻辑）
    OPENCODE_SAFE_CONFIG = {
        "permission": {
            "bash": "allow",
            "websearch": "deny",
            "webfetch": "deny",
            "repo_clone": "deny",
            "external_directory": "allow",
            "question": "deny",
            "read": "allow",
            "write": "allow",
            "edit": "allow",
            "glob": "allow",
            "grep": "allow",
            "list": "allow",
            "skill": "allow",
        }
    }

    def create_sandbox(self, skill_name: str, eval_id: int) -> Path:
        """
        创建用例沙箱目录

        Args:
            skill_name: skill 名称
            eval_id: 评测用例 ID

        Returns:
            沙箱目录路径
        """
        sandbox_name = f"{skill_name}_eval_{eval_id}"
        sandbox_path = self.sandbox_root / sandbox_name

        # 创建沙箱目录（如果已存在则先清理）
        if sandbox_path.exists():
            shutil.rmtree(sandbox_path)
        sandbox_path.mkdir(parents=True, exist_ok=True)

        # 创建 logs 子目录
        logs_dir = sandbox_path / "logs"
        logs_dir.mkdir(exist_ok=True)

        # 写入 opencode 安全配置：限制危险工具，防止不可信 prompt 利用
        opencode_dir = sandbox_path / ".opencode"
        opencode_dir.mkdir(parents=True, exist_ok=True)
        config_path = opencode_dir / "opencode.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.OPENCODE_SAFE_CONFIG, f, ensure_ascii=False, indent=2)

        return sandbox_path

    def create_team_sandbox(self, team_name: str, eval_id: int,
                             repo_root: Path, team_dir: Path) -> Path:
        """为 Team 创建沙箱并通过 init.sh 安装 Team。

        1. 创建 sandboxes/<team_name>_eval_<id>/ 目录
        2. 写入 .opencode/opencode.json 安全配置
        3. 执行 init.sh project opencode <sandbox_path> 安装 team

        Args:
            team_name: team 名称
            eval_id: 评测用例 ID
            repo_root: 仓库根目录路径
            team_dir: team 源目录路径

        Returns:
            沙箱目录路径

        Raises:
            RuntimeError: init.sh 不存在、执行失败、或超时
        """
        sandbox_path = self.create_sandbox(team_name, eval_id)

        init_sh = team_dir / "init.sh"
        if not init_sh.exists():
            # 清理已创建的沙箱目录
            _rmtree_ignore_errors(sandbox_path)
            raise RuntimeError(
                f"[Sandbox] init.sh not found at {init_sh} for team {team_name}"
            )

        logger.info("[Sandbox] 安装 Team %s 到沙箱 %s ...", team_name, sandbox_path.name)
        try:
            proc = subprocess.run(
                ["bash", str(init_sh), "project", "opencode", str(sandbox_path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300,
                cwd=str(repo_root),
            )
            if proc.returncode != 0:
                stderr_tail = proc.stderr[-1000:] if proc.stderr else "(no stderr)"
                raise RuntimeError(
                    f"[Sandbox] init.sh exited with {proc.returncode} for team {team_name}. "
                    f"stderr: {stderr_tail}"
                )
            logger.info("[Sandbox] Team %s installed successfully in %s", team_name, sandbox_path.name)
        except subprocess.TimeoutExpired:
            _rmtree_ignore_errors(sandbox_path)
            raise RuntimeError(
                f"[Sandbox] init.sh timed out (300s) for team {team_name}"
            ) from None
        except RuntimeError:
            _rmtree_ignore_errors(sandbox_path)
            raise
        except Exception as e:
            _rmtree_ignore_errors(sandbox_path)
            raise RuntimeError(
                f"[Sandbox] init.sh failed for team {team_name}: {e}"
            ) from e

        return sandbox_path

    def cleanup_all(self) -> None:
        """清理所有沙箱目录"""
        if self.sandbox_root.exists():
            shutil.rmtree(self.sandbox_root)
            logger.info("[Sandbox] 清理所有沙箱目录: %s", self.sandbox_root)

    def list_sandboxes(self) -> list:
        """列出所有沙箱目录"""
        if not self.sandbox_root.exists():
            return []
        return [d for d in self.sandbox_root.iterdir() if d.is_dir()]

    def get_sandbox_path(self, skill_name: str, eval_id: int) -> Path:
        """获取指定用例的沙箱路径（不创建）"""
        return self.sandbox_root / f"{skill_name}_eval_{eval_id}"