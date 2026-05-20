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
OpencodeRunner 启动参数示例

本文件展示 OpencodeRunner 的各种启动方式和参数配置
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from opencode_runner import OpencodeRunner

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def example_basic_run():
    logger.info("=" * 60)
    logger.info("示例1: 基本运行")
    logger.info("=" * 60)

    runner = OpencodeRunner(verbose=True)

    if not runner.is_available():
        logger.info("[FAIL] opencode 未安装或未在系统 PATH 中")
        logger.info("\n解决方案:")
        logger.info("  方案1: 安装 opencode 并添加到 PATH")
        logger.info("  方案2: 指定 opencode_path 参数")
        logger.info("    runner = OpencodeRunner(opencode_path='D:/tools/opencode.exe')")
        return

    result = runner.run_stream("请帮我写一个简单的 Python 函数", '“')

    logger.info("成功: {result.success}")
    logger.info("输出: {result.output}")
    if result.error:
        logger.error("错误: {result.error}")


def example_with_opencode_path():
    logger.info("=" * 60)
    logger.info("示例2: 指定 opencode 路径 (Windows)")
    logger.info("=" * 60)

    opencode_path = "D:/tools/opencode.exe"

    runner = OpencodeRunner(
        opencode_path=opencode_path,
        verbose=True
    )

    logger.info("opencode 路径: {runner.get_opencode_path()}")
    logger.info("是否可用: {runner.is_available()}")


def example_check_availability():
    logger.info("=" * 60)
    logger.info("示例3: 检查 opencode 是否可用")
    logger.info("=" * 60)

    runner = OpencodeRunner(verbose=True)

    if runner.is_available():
        logger.info("[OK] opencode 可用: {runner.get_opencode_path()}")
    else:
        logger.info("[FAIL] opencode 不可用")
        logger.info("\n可能的解决方案:")
        logger.info("  1. 安装 opencode: pip install opencode")
        logger.info("  2. 添加到 PATH 环境变量")
        logger.info("  3. 使用 opencode_path 参数指定完整路径")


def example_with_model():
    logger.info("=" * 60)
    logger.info("示例4: 指定模型运行")
    logger.info("=" * 60)

    runner = OpencodeRunner(
        model="GLM-5",
        verbose=True
    )

    if not runner.is_available():
        logger.info("opencode 不可用，跳过此示例")
        return

    result = runner.run("解释什么是 pytest")

    logger.info("成功: {result.success}")
    logger.info("输出: {result.output}")


def example_streaming():
    logger.info("=" * 60)
    logger.info("示例5: 流式输出", flush=True)
    logger.info("=" * 60)

    runner = OpencodeRunner(verbose=True)

    if not runner.is_available():
        logger.info("opencode 不可用，跳过此示例", flush=True)
        return

    logger.info("\n流式输出开始:", flush=True)
    for chunk in runner.run_stream("请写一个 hello world 程序"):
        if chunk["type"] == "json_output":
            data = chunk["data"]
            logger.debug("[JSON] {json.dumps(data, indent=2)}")
        elif chunk["type"] == "raw_output":
            logger.debug("[RAW] {chunk['data']}")
        elif chunk["type"] == "complete":
            logger.info("\n完成! Session: {chunk.get('session_file')}")


def example_keep_session():
    logger.info("=" * 60)
    logger.info("示例6: 保持 Session 文件")
    logger.info("=" * 60)

    runner = OpencodeRunner(
        keep_session=True,
        verbose=True
    )

    if not runner.is_available():
        logger.info("opencode 不可用，跳过此示例")
        logger.info("\nSession 文件管理演示 (不需要真实运行):")
        logger.info("Session 目录: {runner.session_dir}")
        logger.info("Session 列表: {runner.list_sessions()}")
        return

    result = runner.run("第一次对话")

    logger.info("Session 文件: {result.session_file}")
    logger.info("Sessions 列表: {runner.list_sessions()}")


def example_resume_session():
    logger.info("=" * 60)
    logger.info("示例7: 恢复 Session")
    logger.info("=" * 60)

    runner = OpencodeRunner(keep_session=True, verbose=True)

    if not runner.is_available():
        logger.info("opencode 不可用，跳过此示例")
        return

    runner1 = OpencodeRunner(keep_session=True, verbose=True)
    result1 = runner1.run("请记住我的名字是 Alice")
    session_file = result1.session_file

    logger.info("第一次 Session: {session_file}")

    runner2 = OpencodeRunner(keep_session=True, verbose=True)
    result2 = runner2.resume_session(session_file, "你还记得我的名字吗？")

    logger.info("恢复成功: {result2.success}")
    logger.info("输出: {result2.output}")


def example_with_skill():
    logger.info("=" * 60)
    logger.info("示例8: 使用 Skill")
    logger.info("=" * 60)

    runner = OpencodeRunner(verbose=True)

    if not runner.is_available():
        logger.info("opencode 不可用，跳过此示例")
        return

    result = runner.run(
        prompt="请生成测试用例",
        skill="ascendc-st-design"
    )

    logger.info("成功: {result.success}")
    logger.info("输出: {result.output}")


def example_with_timeout():
    logger.info("=" * 60)
    logger.info("示例7: 设置超时")
    logger.info("=" * 60)

    runner = OpencodeRunner(
        timeout=600,
        verbose=True
    )
    result = runner.run("复杂的长时间任务")

    logger.info("成功: {result.success}")
    if not result.success and "timeout" in result.error:
        logger.info("任务超时")


def example_with_workdir():
    logger.info("=" * 60)
    logger.info("示例8: 指定工作目录")
    logger.info("=" * 60)

    workdir = Path(__file__).parent.parent.parent
    runner = OpencodeRunner(
        workdir=str(workdir),
        verbose=True
    )
    result = runner.run("请列出当前目录的文件")

    logger.info("工作目录: {runner.workdir}")
    logger.info("成功: {result.success}")


def example_cleanup_sessions():
    logger.info("=" * 60)
    logger.info("示例9: 清理 Session 文件")
    logger.info("=" * 60)

    runner = OpencodeRunner(
        keep_session=True,
        verbose=True
    )

    # 创建一些 session 文件用于演示清理功能
    # 使用公有方法 run() 来触发 session 创建
    logger.info("Session 文件数量: {len(runner.list_sessions())}")

    runner.cleanup_all_sessions()
    logger.info("清理后数量: {len(runner.list_sessions())}")


def example_cli_usage():
    logger.info("=" * 60)
    logger.info("示例10: CLI 命令行使用方式")
    logger.info("=" * 60)

    examples = [
        "# 检查 opencode 是否可用",
        "python opencode_runner.py --check",
        "",
        "# 基本运行",
        "python opencode_runner.py \"你的提示\"",
        "",
        "# 指定模型",
        "python opencode_runner.py \"你的提示\" --model gpt-4",
        "",
        "# 指定 opencode 路径 (Windows)",
        "python opencode_runner.py \"你的提示\" --opencode-path D:/tools/opencode.exe",
        "",
        "# 保持 session",
        "python opencode_runner.py \"你的提示\" --keep-session",
        "",
        "# 流式输出",
        "python opencode_runner.py \"你的提示\" --stream",
        "",
        "# 详细输出",
        "python opencode_runner.py \"你的提示\" --verbose",
        "",
        "# 设置超时",
        "python opencode_runner.py \"你的提示\" --timeout 600",
        "",
        "# 使用 skill",
        "python opencode_runner.py \"你的提示\" --skill ascendc-st-design",
        "",
        "# 指定工作目录",
        "python opencode_runner.py \"你的提示\" --workdir /path/to/dir",
        "",
        "# 恢复 session",
        "python opencode_runner.py \"你的提示\" --resume-session session.json",
        "",
        "# 清理 session",
        "python opencode_runner.py \"你的提示\" --cleanup",
    ]

    for _ex in examples:
        logger.info("  {ex}")


def example_full_config():
    logger.info("=" * 60)
    logger.info("示例11: 完整配置")
    logger.info("=" * 60)

    runner = OpencodeRunner(
        model="GLM-5",
        keep_session=True,
        session_dir="/tmp/opencode_sessions",
        timeout=600,
        verbose=True,
        workdir=str(Path.cwd())
    )

    result = runner.run(
        prompt="请帮我分析这段代码的质量",
        skill="code-review",
        additional_args=["--max-tokens", "4000"]
    )

    logger.info("成功: {result.success}")
    logger.info("Session: {result.session_file}")
    logger.info("Metadata: {result.metadata}")


def print_usage_guide():
    logger.info(""")
OpencodeRunner 使用指南

启动参数说明:
  --model           指定模型名称 (如 gpt-4, claude-3)
  --keep-session    保持 session 文件 (默认: False)
  --session-dir     session 文件存储目录
  --timeout         超时时间 (秒, 默认: 300)
  --verbose         详细输出模式
  --workdir         工作目录
  --skill           指定 skill
  --stream          使用流式输出
  --resume-session  恢复已有的 session
  --cleanup         清理所有 session 文件
  --opencode-path   指定 opencode 可执行文件路径 (Windows 用户常用)
  --check           检查 opencode 是否可用

Windows 用户特别注意:
  如果 opencode 未添加到 PATH，请使用 --opencode-path 参数指定完整路径:
  
  python opencode_runner.py "你的提示" --opencode-path D:/tools/opencode.exe
  
  或在 Python API 中:
  
  runner = OpencodeRunner(opencode_path="D:/tools/opencode.exe")

Python API 使用:
  from opencode_runner import OpencodeRunner
  
  # 基本使用
  runner = OpencodeRunner()
  
  # 检查是否可用
  if runner.is_available():
      result = runner.run("你的提示")
  
  # 指定 opencode 路径
  runner = OpencodeRunner(opencode_path="D:/tools/opencode.exe")
  result = runner.run("你的提示")
  
  # 指定模型
  runner = OpencodeRunner(model="gpt-4", keep_session=True)
  result = runner.run("你的提示")
  
  # 流式输出
  for chunk in runner.run_stream("你的提示"):
      print(chunk)
  
  # 恢复 session
  runner.resume_session("session.json", "继续对话")
  
  # 清理 session
  runner.cleanup_all_sessions()
""")


EXAMPLE_DISPATCH = {
    "basic": example_basic_run,
    "check": example_check_availability,
    "path": example_with_opencode_path,
    "model": example_with_model,
    "stream": example_streaming,
    "session": example_keep_session,
    "resume": example_resume_session,
    "skill": example_with_skill,
    "timeout": example_with_timeout,
    "workdir": example_with_workdir,
    "cleanup": example_cleanup_sessions,
    "cli": example_cli_usage,
    "full": example_full_config,
}

ALL_EXAMPLES = list(EXAMPLE_DISPATCH.values())


def _run_example(example_name):
    if example_name == "all":
        for fn in ALL_EXAMPLES:
            fn()
    elif example_name in EXAMPLE_DISPATCH:
        EXAMPLE_DISPATCH[example_name]()


def main():
    parser = argparse.ArgumentParser(
        description="OpencodeRunner 启动示例",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--example",
        choices=list(EXAMPLE_DISPATCH.keys()) + ["all"],
        default="basic",
        help="运行指定示例"
    )
    parser.add_argument(
        "--prompt",
        default="请帮我写一个简单的 Python 函数",
        help="测试提示"
    )
    parser.add_argument(
        "--opencode-path",
        help="opencode 可执行文件路径"
    )
    parser.add_argument(
        "--guide",
        action="store_true",
        help="显示使用指南"
    )
    args = parser.parse_args()
    example_with_skill()
    if args.guide:
        print_usage_guide()
        return
    _run_example(args.example)


if __name__ == "__main__":
    main()
