#!/usr/bin/env node
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// ---------------------------------------------------------------------------
// permission-guard —— TraeCode 侧静默问卷拦截 hook
//
// 机制（TraeCode PreToolUse hook）：
//   hook 注册在 .trae/hooks.json（init Step 4+ 幂等写入），matcher 命中
//   AskUserQuestion 问卷工具时本脚本被调用，事件 JSON 经 stdin 传入：
//     { session_id, cwd, hook_event_name: "PreToolUse", tool_use_id,
//       tool_name, llm_tool_name, tool_input }
//   Trae 的 PreToolUse stdin **没有 agent 角色字段**（与 opencode/claude 不同，
//   无法按角色反查当前子 Agent），因此本 hook 只做**静默问卷拦截**兜底：
//   .cannbot/settings.json 的 mode=silent 时阻断任何问卷发送。
//
// 角色写权限隔离由 .trae/agents/*.md 的 frontmatter tools 静态限权承担
// （Trae 原生机制，init Step 3 生成时按角色注入工具白名单）；目录级写权限
// （.cannbot 等）Trae 不支持，由 AGENTS.md prompt 约束（同 codex 降级）。
//
// 违规 → exit 2（阻断本次调用，stderr 原因回传模型）；放行 → exit 0 且无输出。
//
// ★ 静默语义与 hooks/opencode/permission-guard.js、hooks/claude/permission-guard.js
// ★ 及 hooks/dsh/permission-guard.js 保持一致（SILENT_GUARDED_TOOLS 子串匹配 +
// ★ readSilentMode 读 .cannbot/settings.json）。
// ---------------------------------------------------------------------------

"use strict"

const { readFileSync } = require("node:fs")
const { join } = require("node:path")

// 静默模式（.cannbot/settings.json 的 mode=silent）下拦截的询问类工具。
// 按**子串**匹配工具名（小写后），覆盖 AskUserQuestion 等带前后缀的命名。
const SILENT_GUARDED_TOOLS = ["question", "ask"]

function deny(reason) {
  // exit 2：阻断本次工具调用，stderr 内容回传给模型
  console.error(`[permission-guard] ${reason}`)
  process.exit(2)
}

// 读取 .cannbot/settings.json 的静默开关（hook 为一次性进程，天然实时；
// 读失败按非静默处理，避免误伤）
function readSilentMode(projectRoot) {
  try {
    const data = JSON.parse(
      readFileSync(join(projectRoot, ".cannbot", "settings.json"), "utf8"),
    )
    return data && data.mode === "silent"
  } catch {
    return false
  }
}

function main() {
  const raw = readFileSync(0, "utf8") // stdin

  let input
  try {
    input = JSON.parse(raw)
  } catch {
    return // 输入异常不拦（避免误伤）
  }

  // 仅处理 PreToolUse（matcher 已限定，防御性判断）
  if (input.hook_event_name && input.hook_event_name !== "PreToolUse") return

  const tool = (input.tool_name || "").toLowerCase()

  // 静默模式：拦截问卷发送（任何角色都不得绕过）
  if (SILENT_GUARDED_TOOLS.some((t) => tool.includes(t))) {
    // Trae 注入 CLAUDE_PROJECT_DIR（兼容 Claude 环境变量）与 TRAE_PROJECT_DIR；
    // 项目级 hook 工作目录即项目根，cwd 兜底
    const projectRoot =
      process.env.CLAUDE_PROJECT_DIR ||
      process.env.TRAE_PROJECT_DIR ||
      input.cwd ||
      process.cwd()
    if (readSilentMode(projectRoot)) {
      deny(
        "静默模式已启用，问卷发送被拦截：请按静默默认决策执行——不发送问卷，落盘 " +
          '.reply.json（{"mode":"silent","decision":"accepted"}）；如需恢复交互，请让用户关闭静默模式。',
      )
    }
    return
  }

  // 非问卷工具：放行（角色写权限由 Subagent tools 静态限权，不在此拦截）
}

main()
