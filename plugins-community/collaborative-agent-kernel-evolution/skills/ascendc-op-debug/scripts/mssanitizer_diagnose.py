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
msSanitizer 诊断助手 - AscendC 算子异常诊断专家

核心能力：
1. 自动检测 mssanitizer 工具和环境
2. 针对性诊断方案生成（synccheck/racecheck/memcheck）
3. 报告解析和根因分析
4. 修复建议生成
5. 闭环验证

作者：Claude Code
版本：1.0.0
"""

import os
import sys
import json
import re
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class CheckType(Enum):
    """msSanitizer 检测类型"""
    MEMCHECK = "memcheck"
    RACECHECK = "racecheck"
    INITCHECK = "initcheck"
    SYNCCHECK = "synccheck"


class Severity(Enum):
    """异常严重级别"""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class DiagnosticIssue:
    """诊断问题"""
    check_type: CheckType
    severity: Severity
    location: str  # file:line
    message: str
    evidence: str = ""
    fix_suggestion: str = ""
    code_snippet: str = ""


@dataclass
class DiagnosticResult:
    """诊断结果"""
    success: bool
    issues: List[DiagnosticIssue] = field(default_factory=list)
    raw_output: str = ""
    analysis: str = ""
    recommended_actions: List[str] = field(default_factory=list)


class MsSanitizerHelper:
    """msSanitizer 诊断助手"""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.mssanitizer_bin = self._find_mssanitizer()
        self.cann_home = os.environ.get("ASCEND_TOOLKIT_HOME", "")
        self.log_dir = self.project_root / "mindstudio_sanitizer_log"

    def _find_mssanitizer(self) -> Optional[Path]:
        """查找 mssanitizer 可执行文件"""
        possible_paths = [
            Path("/usr/local/Ascend/ascend-toolkit/latest/tools/mssanitizer/bin/mssanitizer"),
            Path("/usr/local/Ascend/ascend-toolkit/8.3.RC2/tools/mssanitizer/bin/mssanitizer"),
            Path("/usr/local/Ascend/ascend-toolkit/8.3.RC1/tools/mssanitizer/bin/mssanitizer"),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # 尝试 which
        try:
            result = subprocess.run(["which", "mssanitizer"],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except Exception:
            pass

        return None

    def check_environment(self) -> Tuple[bool, str]:
        """检查环境配置"""
        issues = []

        # 检查 mssanitizer
        if not self.mssanitizer_bin:
            issues.append("❌ 未找到 mssanitizer 工具")
        else:
            try:
                result = subprocess.run(
                    [str(self.mssanitizer_bin), "--version"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    issues.append(f"✅ mssanitizer: {result.stdout.strip()}")
                else:
                    issues.append(f"⚠️ mssanitizer 存在但无法获取版本")
            except Exception as e:
                issues.append(f"⚠️ mssanitizer 检查失败: {e}")

        # 检查 CANN 环境
        if not self.cann_home:
            issues.append("⚠️ 未设置 ASCEND_TOOLKIT_HOME 环境变量")
        else:
            issues.append(f"✅ CANN: {self.cann_home}")

        # 检查必要的环境变量
        if os.environ.get("PYTORCH_NO_NPU_MEMORY_CACHING") == "1":
            issues.append("✅ PyTorch 内存池已关闭（推荐）")

        return (len([i for i in issues if "❌" in i]) == 0,
                "\n".join(issues))

    def diagnose_sync_check(self, executable: str,
                            args: List[str] = None) -> DiagnosticResult:
        """
        诊断同步问题（SetFlag/WaitFlag 配对）

        适用场景：
        - 算子超时（error 507034）
        - 怀疑同步指令配对错误
        - 事件ID复用导致的状态机混乱
        """
        cmd = [str(self.mssanitizer_bin), "--tool=synccheck",
               "--log-level=info", executable]
        if args:
            cmd.extend(args)

        return self._run_and_parse(cmd, CheckType.SYNCCHECK)

    def diagnose_race_check(self, executable: str,
                            args: List[str] = None) -> DiagnosticResult:
        """
        诊断数据竞争

        适用场景：
        - 多核踩踏
        - 流水间竞争（WAW/WAR/RAW）
        - 结果不确定（有时正确有时错误）
        """
        cmd = [str(self.mssanitizer_bin), "--tool=racecheck",
               "--log-level=info", executable]
        if args:
            cmd.extend(args)

        return self._run_and_parse(cmd, CheckType.RACECHECK)

    def diagnose_mem_check(self, executable: str,
                           args: List[str] = None,
                           check_leak: bool = False,
                           check_uninit: bool = False) -> DiagnosticResult:
        """
        诊断内存问题

        适用场景：
        - 非法读写
        - 多核踩踏
        - 非对齐访问
        - 内存泄漏（check_leak=True）
        - 未初始化读取（check_uninit=True, 实际用 initcheck）
        """
        cmd = [str(self.mssanitizer_bin), "--tool=memcheck",
               "--log-level=info"]

        if check_leak:
            cmd.append("--leak-check=yes")

        cmd.append(executable)
        if args:
            cmd.extend(args)

        return self._run_and_parse(cmd, CheckType.MEMCHECK)

    def diagnose_init_check(self, executable: str,
                            args: List[str] = None) -> DiagnosticResult:
        """
        诊断未初始化内存读取

        适用场景：
        - 脏数据导致结果错误
        - 不确定是否初始化的缓冲区
        """
        cmd = [str(self.mssanitizer_bin), "--tool=initcheck",
               "--log-level=info", executable]
        if args:
            cmd.extend(args)

        return self._run_and_parse(cmd, CheckType.INITCHECK)

    def diagnose_comprehensive(self, executable: str,
                               args: List[str] = None) -> Dict[CheckType, DiagnosticResult]:
        """
        综合诊断：运行所有检测

        推荐顺序：
        1. memcheck（基础内存问题）
        2. synccheck（同步配对）
        3. racecheck（数据竞争）
        4. initcheck（未初始化）
        """
        results = {}

        # 检查环境
        env_ok, env_msg = self.check_environment()
        if not env_ok:
            print(f"⚠️ 环境检查未通过:\n{env_msg}")
            return results

        print(f"✅ 环境检查通过:\n{env_msg}\n")

        # 按顺序执行检测
        checks = [
            (CheckType.MEMCHECK, "内存检测"),
            (CheckType.SYNCCHECK, "同步检测"),
            (CheckType.RACECHECK, "竞争检测"),
            (CheckType.INITCHECK, "未初始化检测"),
        ]

        for check_type, check_name in checks:
            print(f"\n{'='*60}")
            print(f"🔍 执行 {check_name} ({check_type.value})")
            print(f"{'='*60}")

            if check_type == CheckType.MEMCHECK:
                result = self.diagnose_mem_check(executable, args)
            elif check_type == CheckType.SYNCCHECK:
                result = self.diagnose_sync_check(executable, args)
            elif check_type == CheckType.RACECHECK:
                result = self.diagnose_race_check(executable, args)
            elif check_type == CheckType.INITCHECK:
                result = self.diagnose_init_check(executable, args)

            results[check_type] = result

            # 打印摘要
            error_count = len([i for i in result.issues if i.severity == Severity.ERROR])
            warn_count = len([i for i in result.issues if i.severity == Severity.WARNING])

            if error_count > 0:
                print(f"❌ 发现 {error_count} 个错误")
            elif warn_count > 0:
                print(f"⚠️ 发现 {warn_count} 个警告")
            else:
                print(f"✅ 未发现问题")

        return results

    def _run_and_parse(self, cmd: List[str],
                       check_type: CheckType) -> DiagnosticResult:
        """运行检测并解析输出"""
        try:
            print(f"🚀 执行命令: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.project_root
            )

            raw_output = result.stdout + "\n" + result.stderr

            print(f"\n原始输出:\n{raw_output[:1000]}...")  # 前1000字符

            return self._parse_output(raw_output, check_type)

        except subprocess.TimeoutExpired:
            return DiagnosticResult(
                success=False,
                raw_output="检测超时（可能算子本身有死循环）",
                analysis="检测工具运行超时，这可能意味着被测算子存在死循环或严重阻塞"
            )
        except Exception as e:
            return DiagnosticResult(
                success=False,
                raw_output=str(e),
                analysis=f"检测执行失败: {e}"
            )

    def _parse_output(self, output: str,
                      check_type: CheckType) -> DiagnosticResult:
        """解析 mssanitizer 输出"""
        issues = []
        lines = output.split('\n')

        if check_type == CheckType.SYNCCHECK:
            issues = self._parse_sync_check(output)
        elif check_type == CheckType.RACECHECK:
            issues = self._parse_race_check(output)
        elif check_type == CheckType.MEMCHECK:
            issues = self._parse_mem_check(output)
        elif check_type == CheckType.INITCHECK:
            issues = self._parse_init_check(output)

        analysis = self._generate_analysis(issues, check_type)

        return DiagnosticResult(
            success=True,
            issues=issues,
            raw_output=output,
            analysis=analysis
        )

    def _parse_sync_check(self, output: str) -> List[DiagnosticIssue]:
        """解析同步检测输出"""
        issues = []

        # 匹配模式：Unpaired set_flag
        pattern = r'(WARNING|ERROR):\s*(Unpaired set_flag|Redundant set_flag).*?in\s+(.+?):(\d+)'

        for match in re.finditer(pattern, output, re.MULTILINE | re.DOTALL):
            severity = Severity.ERROR if "ERROR" in match.group(1) else Severity.WARNING
            issue_type = match.group(2)
            file_line = match.group(3)
            line_num = match.group(4)

            issue = DiagnosticIssue(
                check_type=CheckType.SYNCCHECK,
                severity=severity,
                location=f"{file_line}:{line_num}",
                message=f"{issue_type}: 同步指令配对错误",
                evidence=match.group(0),
                fix_suggestion=self._get_sync_fix_suggestion(issue_type)
            )
            issues.append(issue)

        return issues

    def _parse_race_check(self, output: str) -> List[DiagnosticIssue]:
        """解析竞争检测输出"""
        issues = []

        # 匹配模式：Potential RAW/WAR/WAW hazard
        pattern = r'ERROR:\s*Potential\s+(RAW|WAR|WAW)\s+hazard.*?at\s+(\w+).*?in\s+block\s+(\d+).*?#3\s+(.+?):(\d+):\d+'

        for match in re.finditer(pattern, output, re.MULTILINE):
            hazard_type = match.group(1)  # RAW/WAR/WAW
            addr_space = match.group(2)   # GM/UB/etc
            block_id = match.group(3)
            file_path = match.group(4)
            line_num = match.group(5)

            issue = DiagnosticIssue(
                check_type=CheckType.RACECHECK,
                severity=Severity.ERROR,
                location=f"{file_path}:{line_num}",
                message=f"数据竞争 ({hazard_type}) on {addr_space} in block {block_id}",
                evidence=match.group(0),
                fix_suggestion=self._get_race_fix_suggestion(hazard_type)
            )
            issues.append(issue)

        return issues

    def _parse_mem_check(self, output: str) -> List[DiagnosticIssue]:
        """解析内存检测输出"""
        issues = []

        # 匹配多种内存错误模式
        patterns = [
            (r'ERROR:\s*illegal (read|write) of size (\d+).*?#3\s+(.+?):(\d+):\d+', "非法读写"),
            (r'WARNING:\s*out of bounds of size (\d+).*?#3\s+(.+?):(\d+):\d+', "多核踩踏"),
            (r'ERROR:\smisaligned access of size (\d+).*?#3\s+(.+?):(\d+):\d+', "非对齐访问"),
            (r'ERROR:\s*LeakCheck.*?#3\s+(.+?):(\d+):\d+', "内存泄漏"),
            (r'ERROR:\s*illegal free\(\).*?#3\s+(.+?):(\d+):\d+', "非法释放"),
        ]

        for pattern, error_type in patterns:
            for match in re.finditer(pattern, output, re.MULTILINE):
                file_path = match.group(2) if "bounds" in error_type else match.group(3)
                line_num = match.group(3) if "bounds" in error_type else match.group(4)

                issue = DiagnosticIssue(
                    check_type=CheckType.MEMCHECK,
                    severity=Severity.ERROR if "ERROR" in match.group(0) else Severity.WARNING,
                    location=f"{file_path}:{line_num}",
                    message=error_type,
                    evidence=match.group(0),
                    fix_suggestion=self._get_mem_fix_suggestion(error_type)
                )
                issues.append(issue)

        return issues

    def _parse_init_check(self, output: str) -> List[DiagnosticIssue]:
        """解析未初始化检测输出"""
        issues = []

        pattern = r'ERROR:\s*uninitialized read of size (\d+).*?#3\s+(.+?):(\d+):\d+'

        for match in re.finditer(pattern, output, re.MULTILINE):
            size = match.group(1)
            file_path = match.group(2)
            line_num = match.group(3)

            issue = DiagnosticIssue(
                check_type=CheckType.INITCHECK,
                severity=Severity.ERROR,
                location=f"{file_path}:{line_num}",
                message=f"读取未初始化内存 ({size} bytes)",
                evidence=match.group(0),
                fix_suggestion="在使用内存前先初始化（Duplicate/DataCopy/SetValue）"
            )
            issues.append(issue)

        return issues

    def _get_sync_fix_suggestion(self, issue_type: str) -> str:
        """获取同步问题修复建议"""
        suggestions = {
            "Unpaired set_flag": "检查 SetFlag/WaitFlag 是否成对出现。每个 SetFlag 必须有对应的 WaitFlag。",
            "Redundant set_flag": "移除冗余的 SetFlag。两个完全相同的 SetFlag 会导致事件计数器状态错误。"
        }
        return suggestions.get(issue_type, "请检查同步指令的配对关系")

    def _get_race_fix_suggestion(self, hazard_type: str) -> str:
        """获取竞争问题修复建议"""
        suggestions = {
            "RAW": "Read-After-Write: 读操作在写操作前完成。添加 SetFlag/WaitFlag 同步。",
            "WAR": "Write-After-Read: 写操作在读操作前完成。添加同步或调整内存访问顺序。",
            "WAW": "Write-After-Write: 两个写操作顺序不确定。添加同步确保写入顺序。"
        }
        return suggestions.get(hazard_type, "添加同步指令或调整内存访问顺序")

    def _get_mem_fix_suggestion(self, error_type: str) -> str:
        """获取内存问题修复建议"""
        suggestions = {
            "非法读写": "检查 DataCopy/LocalTensor 的偏移和大小，确保不越界",
            "多核踩踏": "添加核间同步或确保各核访问的内存区域不重叠",
            "非对齐访问": "确保 DMA 传输地址满足对齐要求（通常32字节）",
            "内存泄漏": "确保每次 aclrtMalloc 都有对应的 aclrtFree",
            "非法释放": "检查释放的指针是否有效，避免重复释放"
        }
        return suggestions.get(error_type, "请检查内存操作的正确性")

    def _generate_analysis(self, issues: List[DiagnosticIssue],
                           check_type: CheckType) -> str:
        """生成分析报告"""
        if not issues:
            return f"✅ {check_type.value} 未发现异常"

        error_count = len([i for i in issues if i.severity == Severity.ERROR])
        warn_count = len([i for i in issues if i.severity == Severity.WARNING])

        analysis = f"📊 {check_type.value} 发现 {error_count} 个错误，{warn_count} 个警告\n\n"

        # 按位置分组
        issues_by_location = {}
        for issue in issues:
            loc = issue.location
            if loc not in issues_by_location:
                issues_by_location[loc] = []
            issues_by_location[loc].append(issue)

        for loc, loc_issues in sorted(issues_by_location.items()):
            analysis += f"📍 {loc}:\n"
            for issue in loc_issues:
                icon = "❌" if issue.severity == Severity.ERROR else "⚠️"
                analysis += f"  {icon} {issue.message}\n"
                if issue.fix_suggestion:
                    analysis += f"     💡 {issue.fix_suggestion}\n"

        return analysis

    def generate_html_report(self, results: Dict[CheckType, DiagnosticResult],
                             output_path: Path) -> None:
        """生成 HTML 可视化报告"""
        html_content = self._create_html_report(results)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"\n📄 HTML 报告已生成: {output_path}")

    def _create_html_report(self, results: Dict[CheckType, DiagnosticResult]) -> str:
        """创建 HTML 报告内容"""
        total_errors = sum(len([i for i in r.issues if i.severity == Severity.ERROR])
                           for r in results.values() if r.issues)
        total_warnings = sum(len([i for i in r.issues if i.severity == Severity.WARNING])
                             for r in results.values() if r.issues)

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>msSanitizer 诊断报告</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #1e1e1e; color: #d4d4d4; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #4ec9b0; border-bottom: 2px solid #4ec9b0; padding-bottom: 10px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .summary-card {{ background: #252526; padding: 15px; border-radius: 8px; border-left: 4px solid #4ec9b0; }}
        .summary-card.error {{ border-left-color: #f48771; }}
        .summary-card.warning {{ border-left-color: #dcdcaa; }}
        .summary-card h3 {{ margin: 0 0 10px 0; color: #9cdcfe; }}
        .summary-card .value {{ font-size: 24px; font-weight: bold; }}
        .check-section {{ background: #252526; margin: 20px 0; padding: 20px; border-radius: 8px; }}
        .check-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
        .check-type {{ font-size: 18px; font-weight: bold; color: #4ec9b0; }}
        .badge {{ padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
        .badge.pass {{ background: #4ec9b0; color: #1e1e1e; }}
        .badge.error {{ background: #f48771; color: #1e1e1e; }}
        .badge.warning {{ background: #dcdcaa; color: #1e1e1e; }}
        .issue {{ background: #1e1e1e; padding: 12px; margin: 10px 0; border-radius: 4px; border-left: 4px solid #f48771; }}
        .issue.warning {{ border-left-color: #dcdcaa; }}
        .issue-header {{ display: flex; justify-content: space-between; margin-bottom: 8px; }}
        .issue-location {{ font-family: monospace; color: #9cdcfe; }}
        .issue-message {{ font-weight: bold; margin-bottom: 8px; }}
        .issue-evidence {{ font-family: monospace; background: #2d2d2d; padding: 8px; border-radius: 4px; font-size: 12px; overflow-x: auto; white-space: pre-wrap; margin: 8px 0; }}
        .issue-fix {{ background: #264f78; padding: 8px 12px; border-radius: 4px; margin-top: 8px; }}
        .issue-fix::before {{ content: "💡 "; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 msSanitizer 诊断报告</h1>

        <div class="summary">
            <div class="summary-card error">
                <h3>错误总数</h3>
                <div class="value">{total_errors}</div>
            </div>
            <div class="summary-card warning">
                <h3>警告总数</h3>
                <div class="value">{total_warnings}</div>
            </div>
            <div class="summary-card">
                <h3>检测类型</h3>
                <div class="value">{len(results)}</div>
            </div>
        </div>
"""

        for check_type, result in results.items():
            if not result.issues:
                status_badge = '<span class="badge pass">PASS</span>'
                issues_html = '<p style="color: #4ec9b0;">✅ 未发现问题</p>'
            else:
                error_count = len([i for i in result.issues if i.severity == Severity.ERROR])
                warn_count = len([i for i in result.issues if i.severity == Severity.WARNING])
                status_badge = f'<span class="badge {"error" if error_count > 0 else "warning"}">{error_count}E/{warn_count}W</span>'

                issues_html = ""
                for issue in result.issues:
                    severity_class = "warning" if issue.severity == Severity.WARNING else ""
                    issues_html += f"""
                    <div class="issue {severity_class}">
                        <div class="issue-header">
                            <span class="issue-location">📍 {issue.location}</span>
                        </div>
                        <div class="issue-message">{"⚠️" if issue.severity == Severity.WARNING else "❌"} {issue.message}</div>
                        <div class="issue-evidence">{issue.evidence[:300]}</div>
                        <div class="issue-fix">{issue.fix_suggestion}</div>
                    </div>
                    """

            html += f"""
        <div class="check-section">
            <div class="check-header">
                <span class="check-type">{check_type.value}</span>
                {status_badge}
            </div>
            {issues_html}
        </div>
"""

        html += """
    </div>
</body>
</html>
"""
        return html


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="msSanitizer 诊断助手 - AscendC 算子异常诊断专家"
    )
    parser.add_argument("executable", help="待检测的可执行文件或脚本")
    parser.add_argument("--project-root", default=".",
                        help="项目根目录（默认：当前目录）")
    parser.add_argument("--check", "-c",
                        choices=["memcheck", "racecheck", "synccheck", "initcheck", "all"],
                        default="all", help="检测类型（默认：all）")
    parser.add_argument("--output", "-o", help="HTML 报告输出路径")
    parser.add_argument("--args", nargs="*", help="传递给可执行文件的参数")

    args = parser.parse_args()

    # 创建助手
    helper = MsSanitizerHelper(Path(args.project_root))

    # 检查环境
    env_ok, env_msg = helper.check_environment()
    print(f"环境检查:\n{env_msg}\n")

    if not env_ok:
        print("⚠️ 环境检查未通过，但仍可尝试运行检测")

    # 执行检测
    if args.check == "all":
        results = helper.diagnose_comprehensive(args.executable, args.args)
    else:
        check_type = CheckType(args.check)
        if check_type == CheckType.MEMCHECK:
            results = {CheckType.MEMCHECK: helper.diagnose_mem_check(args.executable, args.args)}
        elif check_type == CheckType.RACECHECK:
            results = {CheckType.RACECHECK: helper.diagnose_race_check(args.executable, args.args)}
        elif check_type == CheckType.SYNCCHECK:
            results = {CheckType.SYNCCHECK: helper.diagnose_sync_check(args.executable, args.args)}
        elif check_type == CheckType.INITCHECK:
            results = {CheckType.INITCHECK: helper.diagnose_init_check(args.executable, args.args)}

    # 生成报告
    if args.output:
        helper.generate_html_report(results, Path(args.output))

    # 打印摘要
    print("\n" + "=" * 60)
    print("📊 诊断摘要")
    print("=" * 60)

    total_errors = sum(len([i for i in r.issues if i.severity == Severity.ERROR])
                       for r in results.values() if r.issues)
    total_warnings = sum(len([i for i in r.issues if i.severity == Severity.WARNING])
                         for r in results.values() if r.issues)

    print(f"总错误数: {total_errors}")
    print(f"总警告数: {total_warnings}")

    if total_errors == 0 and total_warnings == 0:
        print("\n✅ 所有检测通过！未发现异常。")
        return 0
    else:
        print(f"\n⚠️ 发现异常，请查看详细报告和修复建议。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
