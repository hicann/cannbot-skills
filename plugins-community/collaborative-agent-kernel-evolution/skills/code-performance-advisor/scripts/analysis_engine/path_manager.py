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
路径管理器 - 配置驱动,实现skill独立部署

设计原则:
1. 所有路径从assets/configs/paths.yaml读取
2. 支持相对路径和绝对路径
3. Skill可以独立发布,不绑定CAKE2
4. 用户可以自定义配置
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

logger = logging.getLogger(__name__)


class PathManager:
    """
    路径管理器 - 解耦skill与外部项目

    使用示例:
    ```python
    pm = PathManager(skill_root)
    workspace = pm.get_workspace_dir()
    build_dir = pm.get_build_dir("fastgelu")
    ```
    """

    def __init__(self, skill_root: Path, workspace_override: Optional[Path] = None):
        """
        初始化路径管理器

        Args:
            skill_root: code-performance-advisor skill的根目录
            workspace_override: 可选，覆盖默认workspace目录（绝对路径或相对于skill_root的路径）
        """
        self.skill_root = skill_root.resolve()
        self._workspace_override = workspace_override.resolve() if workspace_override and workspace_override.is_absolute() else (
            skill_root / workspace_override).resolve() if workspace_override else None
        self.config = self._load_config()

    @staticmethod
    def _default_config() -> Dict:
        """默认配置(当paths.yaml缺失时)"""
        return {
            "workspace": {
                "base": "workspace"
            },
            "external": {
                "cake2_root": "../../../",
                "build_output": "output/{op_name}/{OpName}Custom"
            },
            "defaults": {
                "auto_create_dirs": True
            }
        }

    # ========== Workspace路径(skill内部) ==========

    def get_workspace_dir(self) -> Path:
        """获取workspace根目录"""
        if self._workspace_override:
            return self._workspace_override
        base = self.config["workspace"]["base"]
        return self.skill_root / base

    # ===== inputs/ - 用户输入（只读）=====

    def get_input_dir(self, op_name: Optional[str] = None) -> Path:
        """获取用户输入目录 (workspace/inputs/{op}/)"""
        base = self.get_workspace_dir() / "inputs"
        if op_name:
            return base / op_name
        return base

    def get_input_code_dir(self, op_name: str) -> Path:
        """获取输入代码目录 (workspace/inputs/{op}/code/)"""
        return self.get_input_dir(op_name) / "code"

    def get_input_profiling_dir(self, op_name: str) -> Path:
        """获取输入profiling目录 (workspace/inputs/{op}/profiling/)"""
        return self.get_input_dir(op_name) / "profiling"

    # ===== sessions/ - 工作流执行记录 =====

    def get_sessions_dir(self) -> Path:
        """获取sessions根目录"""
        return self.get_workspace_dir() / "sessions"

    def get_session_dir(self, session_id: str) -> Path:
        """获取特定session目录"""
        return self.get_sessions_dir() / session_id

    def get_suggestions_dir(self, session_id: str) -> Path:
        """获取session的suggestions目录"""
        return self.get_session_dir(session_id) / "suggestions"

    def get_working_code_dir(self, session_id: str) -> Path:
        """获取session的working_code目录"""
        return self.get_session_dir(session_id) / "working_code"

    def get_session_reports_dir(self, session_id: str) -> Path:
        """获取session的reports目录"""
        return self.get_session_dir(session_id) / "reports"

    # ===== cache/ - 可复用中间结果 =====

    def get_cache_dir(self) -> Path:
        """获取缓存根目录"""
        return self.get_workspace_dir() / "cache"

    def get_tag_file(self, op_name: str, code_hash: Optional[str] = None) -> Path:
        """
        获取tag文件路径 (workspace/cache/tags/tag_{op}.json)

        Args:
            op_name: 算子名
            code_hash: 代码hash(可选,用于区分版本)
        """
        tags_dir = self.get_cache_dir() / "tags"

        if code_hash:
            return tags_dir / f"tag_{op_name}_{code_hash}.json"
        else:
            return tags_dir / f"tag_{op_name}.json"

    # ===== 向后兼容（废弃，保留1个版本）=====

    def get_input_raw_dir(self, op_name: Optional[str] = None) -> Path:
        """⚠️ DEPRECATED: 使用 get_input_dir() 替代"""
        import warnings
        warnings.warn("get_input_raw_dir() is deprecated, use get_input_dir()", DeprecationWarning)
        return self.get_input_dir(op_name)

    # ========== External路径(CAKE2等外部项目) ==========

    def get_cake2_root(self) -> Path:
        """
        获取CAKE2根目录

        支持:
        - 相对路径(相对于skill根目录)
        - 绝对路径
        """
        cake2_path = Path(self.config["external"]["cake2_root"])

        if cake2_path.is_absolute():
            return cake2_path
        else:
            # 相对路径,resolve to absolute
            return (self.skill_root / cake2_path).resolve()

    def get_build_dir(self, op_name: str) -> Path:
        """
        获取编译目录(CAKE2外部)

        自动兼容两种项目格式:
        - 新式(flat): output/{op_name}/build.sh  (如 aten__fused_adamw_)
        - 旧式(Custom子目录): output/{op_name}/{CustomSubdir}/build.sh  (如 fastgelu → FastgeluCustom)

        优先使用显式配置的模板；若模板路径不存在build.sh，则自动探测。

        Args:
            op_name: 算子名(小写,如 fastgelu)

        Returns:
            包含 build.sh 的目录路径
        """
        cake2_root = self.get_cake2_root()
        op_root = cake2_root / "output" / op_name

        # 1. 优先检查新式 flat 格式: output/{op_name}/build.sh
        if (op_root / "build.sh").exists():
            return op_root

        # 2. 探测旧式 Custom 子目录格式: output/{op_name}/*/build.sh
        if op_root.exists():
            for subdir in sorted(op_root.iterdir()):
                if subdir.is_dir() and (subdir / "build.sh").exists():
                    return subdir

        # 3. Fallback: 使用配置模板（允许用户覆盖）
        template = self.config["external"].get("build_output", "output/{op_name}")
        rel_path = template.format(
            op_name=op_name,
            OpName=op_name.capitalize()
        )
        return cake2_root / rel_path

    def get_run_package_path(self, op_name: str) -> Path:
        """获取.run安装包路径"""
        build_dir = self.get_build_dir(op_name)
        run_rel = self.config["external"].get(
            "run_package",
            "build_out/custom_opp_ubuntu_aarch64.run"
        )
        return build_dir / run_rel

    # ========== Helper Methods ==========

    def ensure_dir_exists(self, path: Path) -> Path:
        """确保目录存在(根据配置自动创建)"""
        if self.config["defaults"].get("auto_create_dirs", True):
            path.mkdir(parents=True, exist_ok=True)
        return path

    def validate_external_paths(self) -> Dict[str, bool]:
        """验证外部路径是否存在"""
        cake2_root = self.get_cake2_root()

        return {
            "cake2_root_exists": cake2_root.exists(),
            "cake2_root_path": str(cake2_root)
        }

    def get_skill_dependencies(self) -> list:
        """获取skill依赖列表"""
        return self.config.get("skill_dependencies", [])

    def _load_config(self) -> Dict:
        """加载配置文件(支持默认配置)"""
        config_file = self.skill_root / "assets" / "configs" / "paths.yaml"

        if not config_file.exists():
            default = self._default_config()
            cake2_resolved = (self.skill_root / default["external"]["cake2_root"]).resolve()
            logger.info(f"⚠️  Config not found: {config_file}")
            logger.info(f"   Using defaults — CAKE2 root will be: {cake2_resolved}")
            logger.info(f"   To customize, create: {config_file}")
            return default

        if yaml is None:
            default = self._default_config()
            cake2_resolved = (self.skill_root / default["external"]["cake2_root"]).resolve()
            logger.info(f"⚠️  PyYAML not available; using defaults — CAKE2 root: {cake2_resolved}")
            return default

        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            default = self._default_config()
            cake2_resolved = (self.skill_root / default["external"]["cake2_root"]).resolve()
            logger.info(f"⚠️  Failed to load config ({e}); using defaults — CAKE2 root: {cake2_resolved}")
            return default


# 使用示例
if __name__ == "__main__":
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    # 假设当前在 scripts/analysis_engine/
    skill_root = Path(__file__).resolve().parents[2]
    pm = PathManager(skill_root)

    logger.info("=== Workspace Paths ===")
    logger.info(f"Workspace: {pm.get_workspace_dir()}")
    logger.info(f"Tag file: {pm.get_tag_file('fastgelu')}")
    logger.info(f"Sessions dir: {pm.get_sessions_dir()}")

    logger.info("\n=== External Paths ===")
    logger.info(f"CAKE2 root: {pm.get_cake2_root()}")
    logger.info(f"Build dir: {pm.get_build_dir('fastgelu')}")
    logger.info(f"Run package: {pm.get_run_package_path('fastgelu')}")

    logger.info("\n=== Validation ===")
    validation = pm.validate_external_paths()
    for k, v in validation.items():
        status = "✅" if v else "❌"
        logger.info(f"{status} {k}: {v}")

    logger.info("\n=== Skill Dependencies ===")
    deps = pm.get_skill_dependencies()
    logger.info(f"Dependencies: {', '.join(deps)}")
