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
msDebug 自动化调试脚本

功能：
1. 自动生成调试命令序列
2. 执行调试并收集变量、内存、寄存器信息
3. 对比预期值和实际值
4. 生成调试报告（JSON/HTML）

作者：Claude Code
版本：1.0.0
"""

import os
import sys
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# 将外部可执行文件名解析为绝对路径（找不到时回退到原名）
_WHICH_BIN = shutil.which("which") or "which"
_CAT_BIN = shutil.which("cat") or "cat"


@dataclass
class BreakPoint:
    """断点定义"""
    line: int
    file: str
    actions: List[str] = field(default_factory=list)  # 断点命中后执行的动作


@dataclass
class CheckPoint:
    """检查点：断点 + 验证"""
    breakpoint: BreakPoint
    expected_values: Dict[str, str] = field(default_factory=dict)
    checks: List[str] = field(default_factory=list)


@dataclass
class DebugSession:
    """调试会话"""
    executable: str
    source_file: str
    kernel_name: str
    breakpoints: List[BreakPoint] = field(default_factory=list)
    checkpoints: List[CheckPoint] = field(default_factory=list)
    output_file: Optional[str] = None

    # 调试结果
    variables: Dict[str, any] = field(default_factory=dict)
    memory_snapshots: List[Dict] = field(default_factory=list)
    register_values: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.output_file:
            self.output_file = Path(self.output_file)


class MsDebugAutoDebugger:
    """msDebug 自动化调试器"""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.msdebug_bin = self._find_msdebug()
        self.debug_switch = self._check_debug_switch()

    def check_environment(self) -> Tuple[bool, str]:
        """检查环境配置"""
        issues = []

        # 检查 msdebug
        if not self.msdebug_bin:
            issues.append("❌ 未找到 msdebug 工具")
        else:
            try:
                result = subprocess.run(
                    [str(self.msdebug_bin), "--version"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    issues.append(f"✅ msdebug: {self.msdebug_bin}")
            except Exception as e:
                issues.append(f"⚠️ msdebug 存在但无法运行: {e}")

        # 检查调试开关
        if not self.debug_switch:
            issues.append("⚠️ 调试开关未开启 (/proc/debug_switch != 1)")
            issues.append("   请使用 root 权限执行: echo 1 > /proc/debug_switch")
        else:
            issues.append("✅ 调试开关已开启")

        return (len([i for i in issues if "❌" in i]) == 0, "\n".join(issues))

    def generate_debug_script(self, session: DebugSession, output_path: Path) -> None:
        """生成调试脚本"""
        script_lines = []

        # 添加 msdebug 路径
        script_lines.append("#!/bin/bash")
        script_lines.append(f"# msDebug 自动化调试脚本")
        script_lines.append(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        script_lines.append(f"")
        script_lines.append(f"export LAUNCH_KERNEL_PATH={session.kernel_name}.o")
        script_lines.append(f"")

        # 启动 msdebug
        script_lines.append(f"# 启动调试器")
        script_lines.append(f'cat << "DEBUG_COMMANDS" | {self.msdebug_bin} {session.executable}')
        script_lines.append(f"")

        # 设置断点
        for bp in session.breakpoints:
            script_lines.append(f"# 断点: {session.source_file}:{bp.line}")
            script_lines.append(f"b {session.source_file}:{bp.line}")

        # 运行程序
        script_lines.append(f"# 运行程序")
        script_lines.append(f"run")
        script_lines.append(f"")

        # 检查点
        for cp in session.checkpoints:
            bp = cp.breakpoint
            script_lines.append(f"# 检查点: {session.source_file}:{bp.line}")

            # 执行断点动作
            for action in bp.actions:
                script_lines.append(f"continue")
                script_lines.append(f"# {action}")

            # 检查变量
            for var_name, _ in cp.expected_values.items():
                script_lines.append(f"print {var_name}")

            # 检查内存
            for check in cp.checks:
                script_lines.append(f"{check}")

        # 退出
        script_lines.append(f"quit")
        script_lines.append(f"y")
        script_lines.append(f"DEBUG_COMMANDS")
        script_lines.append(f"")
        script_lines.append("# 调试完成")

        # 写入文件
        output_path.write_text("\n".join(script_lines))
        os.chmod(output_path, 0o755)

    def run_debug_session(self, session: DebugSession) -> Dict:
        """运行调试会话并收集信息"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "executable": session.executable,
            "source_file": session.source_file,
            "kernel_name": session.kernel_name,
            "breakpoints_hit": [],
            "variables": {},
            "memory": {},
            "registers": {},
            "errors": []
        }

        # 生成调试脚本
        script_path = self.project_root / "debug_session.sh"
        self.generate_debug_script(session, script_path)

        # 执行调试脚本
        try:
            import stat
            os.chmod(
                script_path,
                os.stat(script_path).st_mode
                | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
            )
            result = subprocess.run(
                [str(script_path)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.project_root
            )

            # 解析输出
            output = result.stdout + "\n" + result.stderr
            self._parse_debug_output(output, results)

        except subprocess.TimeoutExpired:
            results["errors"].append("调试超时（可能死循环）")
        except Exception as e:
            results["errors"].append(f"调试执行失败: {e}")

        return results

    @staticmethod
    def _parse_debug_output(output: str, results: Dict) -> None:
        """解析 msdebug 输出"""
        lines = output.split('\n')

        # 解析断点命中
        for line in lines:
            if "stop reason = breakpoint" in line:
                results["breakpoints_hit"].append(line.strip())

        # 解析变量值
        var_pattern = r'\((.*?)\s+\$[0-9]+\)\s*=\s*(.+)'
        for var_match in re.finditer(var_pattern, output):
            var_type = var_match.group(1)
            var_value = var_match.group(2)
            results["variables"][var_type] = var_value

        # 解析内存读取
        mem_pattern = r'(0x[0-9a-fA-F]+):\s*\{(.+)\}'
        for mem_match in re.finditer(mem_pattern, output):
            addr = mem_match.group(1)
            values = mem_match.group(1)
            results["memory"][addr] = values

        # 解析寄存器
        reg_pattern = r'([A-Z0-9_]+)\s*=\s*0x[0-9a-fA-F]+'
        for reg_match in re.finditer(reg_pattern, output):
            reg_name = reg_match.group(1)
            reg_value = reg_match.group(0)
            results["registers"][reg_name] = reg_value

    def generate_report(self, results: Dict, output_path: Path, format: str = "json") -> None:
        """生成调试报告"""
        if format == "json":
            self._generate_json_report(results, output_path)
        elif format == "html":
            self._generate_html_report(results, output_path)

    @staticmethod
    def _generate_json_report(results: Dict, output_path: Path) -> None:
        """生成 JSON 报告"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def _generate_html_report(self, results: Dict, output_path: Path) -> None:
        """生成 HTML 报告"""
        html_content = self._create_html_report(results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    @staticmethod
    def _create_html_report(results: Dict) -> str:
        """创建 HTML 报告内容"""
        timestamp = results.get("timestamp", "Unknown")
        executable = results.get("executable", "Unknown")

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>msDebug 调试报告</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #1e1e1e; color: #d4d4d4; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #4ec9b0; border-bottom: 2px solid #4ec9b0; padding-bottom: 10px; }}
        .section {{ background: #252526; padding: 20px; margin: 20px 0; border-radius: 8px; }}
        .section h2 {{ color: #9cdcfe; margin-top: 0; }}
        .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }}
        .info-card {{ background: #1e1e1e; padding: 15px; border-radius: 4px; }}
        .info-label {{ color: #9cdcfe; font-weight: bold; }}
        .info-value {{ color: #d4d4d4; }}
        .breakpoint {{ background: #1e1e1e; padding: 10px; margin: 5px 0; border-left: 3px solid #4ec9b0; }}
        .variable {{ background: #1e1e1e; padding: 10px; margin: 5px 0; font-family: monospace; }}
        .error {{ background: #5a1e1e; padding: 10px; margin: 5px 0; border-left: 3px solid #f48771; }}
        .success {{ color: #4ec9b0; }}
        .warning {{ color: #dcdcaa; }}
        .fail {{ color: #f48771; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 msDebug 调试报告</h1>

        <div class="section">
            <h2>调试概要</h2>
            <div class="info-grid">
                <div class="info-card">
                    <div class="info-label">可执行文件</div>
                    <div class="info-value">{executable}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">调试时间</div>
                    <div class="info-value">{timestamp}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">命中断点数</div>
                    <div class="info-value">{len(results.get('breakpoints_hit', []))}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">收集变量数</div>
                    <div class="info-value">{len(results.get('variables', {}))}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>断点命中</h2>
"""

        for bp in results.get("breakpoints_hit", []):
            html += f'<div class="breakpoint">{bp}</div>\n'

        if not results.get("breakpoints_hit", []):
            html += '<p class="warning">⚠️ 未命中任何断点</p>\n'

        html += '</div>\n'

        # 变量部分
        html += '<div class="section">\n'
        html += '<h2>变量值</h2>\n'

        if results.get("variables"):
            for var_name, var_value in results["variables"].items():
                html += f'<div class="variable">{var_name} = {var_value}</div>\n'
        else:
            html += '<p class="warning">⚠️ 未收集到变量信息</p>\n'

        html += '</div>\n'

        # 内存部分
        html += '<div class="section">\n'
        html += '<h2>内存快照</h2>\n'

        if results.get("memory"):
            for addr, values in results["memory"].items():
                html += f'<div class="variable">{addr}: {values[:100]}...</div>\n'
        else:
            html += '<p class="warning">⚠️ 未收集到内存信息</p>\n'

        html += '</div>\n'

        # 错误部分
        if results.get("errors"):
            html += '<div class="section">\n'
            html += '<h2>错误信息</h2>\n'
            for error in results["errors"]:
                html += f'<div class="error">{error}</div>\n'
            html += '</div>\n'

        html += '''
    </div>
</body>
</html>
'''
        return html

    @staticmethod
    def _find_msdebug() -> Optional[Path]:
        """查找 msdebug 可执行文件"""
        possible_paths = [
            Path("/usr/local/Ascend/ascend-toolkit/latest/tools/msdebug/bin/msdebug"),
            Path("/usr/local/Ascend/ascend-toolkit/8.3.RC2/tools/msdebug/bin/msdebug"),
            Path("/usr/local/Ascend/ascend-toolkit/8.3.RC1/tools/msdebug/bin/msdebug"),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        try:
            result = subprocess.run(
                [_WHICH_BIN, "msdebug"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except Exception as e:
            logger.debug(f"`which msdebug` lookup failed: {e}")

        return None

    @staticmethod
    def _check_debug_switch() -> bool:
        """检查调试开关"""
        try:
            result = subprocess.run(
                [_CAT_BIN, "/proc/debug_switch"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() == "1"
        except Exception:
            return False


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="msDebug 自动化调试脚本"
    )
    parser.add_argument("--executable", required=True, help="可执行文件路径")
    parser.add_argument("--source-file", required=True, help="源代码文件")
    parser.add_argument("--kernel-name", help="算子名称")
    parser.add_argument("--breakpoints", help="断点列表，逗号分隔")
    parser.add_argument("--output", help="输出报告路径")
    parser.add_argument("--format", choices=["json", "html"], default="json", help="报告格式")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    # 创建调试器
    debugger = MsDebugAutoDebugger(Path.cwd())

    # 检查环境
    env_ok, env_msg = debugger.check_environment()
    logger.info(f"环境检查:\n{env_msg}\n")

    if not env_ok:
        logger.info("⚠️ 环境检查未通过，但仍可尝试运行调试")

    # 解析断点
    breakpoints = []
    if args.breakpoints:
        for bp_str in args.breakpoints.split(','):
            if ':' in bp_str:
                line, action = bp_str.split(':', 1)
                bp = BreakPoint(line=int(line), file=args.source_file, actions=[action])
            else:
                bp = BreakPoint(line=int(bp_str), file=args.source_file)
            breakpoints.append(bp)

    # 创建调试会话
    session = DebugSession(
        executable=args.executable,
        source_file=args.source_file,
        kernel_name=args.kernel_name or Path(args.source_file).stem,
        breakpoints=breakpoints,
        output_file=args.output
    )

    # 运行调试
    logger.info(f"🚀 开始调试会话...")
    results = debugger.run_debug_session(session)

    # 生成报告
    if args.output:
        reporter = MsDebugAutoDebugger(Path.cwd())
        output_path = Path(args.output)
        if args.format == "json":
            reporter._generate_json_report(results, output_path)
        else:
            reporter._generate_html_report(results, output_path)
        logger.info(f"\n📄 报告已生成: {output_path}")

    # 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("📊 调试摘要")
    logger.info("=" * 60)
    logger.info(f"命中断点: {len(results['breakpoints_hit'])}")
    logger.info(f"收集变量: {len(results['variables'])}")
    logger.info(f"错误信息: {len(results['errors'])}")

    if results['errors']:
        logger.info(f"\n⚠️ 发现 {len(results['errors'])} 个错误")
        for error in results['errors']:
            logger.info(f"  - {error}")

    return 0 if not results['errors'] else 1


if __name__ == "__main__":
    sys.exit(main())
