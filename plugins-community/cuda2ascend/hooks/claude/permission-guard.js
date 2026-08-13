#!/usr/bin/env node
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// ---------------------------------------------------------------------------
// permission-guard —— Claude Code 侧动态权限 hook
//
// 机制（Claude Code PreToolUse hook）：
//   hook 配置在 .claude/settings.json，matcher 命中写类工具时本脚本被调用，
//   事件 JSON 经 stdin 传入；其中 agent_type 字段标识当前子 Agent 名
//   （主线程无该字段，即 PM）。据此对写类工具做按角色限权。
//
// 违规 → exit 2（阻断本次调用，stderr 原因回传模型）；放行 → exit 0 且无输出。
//
// 配置来源（按角色分开文件）：
//   <项目根>/.cannbot/permissions/*.js —— 每角色一文件，ESM export default
//   { categories, exts }。文件名即角色名（去 .js），如 developer-code.js。
//   由 init.sh Step 4.5 从 skills/workflow-agent-permissions/hooks/
//   整体复制生成；子仓 override 该 skill 时自动生效。
//   配置文件由 init 从工作流受控模板生成，与 hook 同属一套信任域；
//   本脚本以文本求值方式读取（兼容各 Node 版本，不依赖 ESM 加载）。
//
// 每次调用重新加载配置（hook 为一次性进程，天然支持热更新）。
// PM 启动闸口负责检测目录异常并阻断任务派发。
//
// ★ 以下 CONFIG 为工作流级约定，不暴露给仓，改这里即可。★
// ★ 与 hooks/opencode/permission-guard.js 保持同一套规则语义。★
// ---------------------------------------------------------------------------

"use strict"

const { readdirSync, readFileSync } = require("node:fs")
const { join, relative, resolve, sep, isAbsolute } = require("node:path")

// 视为主 Agent(PM) 的情形：主线程 hook 输入无 agent_type 字段；
// 经 `claude --agent` 以 PM 身份启动时 agent_type 为 PM/pm。
const PRIMARY_AGENTS = ["PM", "pm"]

// 未知角色（未在规则表中命中）策略：allow-warn | allow | deny
// deny：规则表已枚举全部角色，未命中即为配置异常或越权调用；
// 且 .cannbot 写入在分类阶段已短路放行，deny 只影响代码/测试/文档目录。
const UNKNOWN_ROLE_POLICY = "deny"

// 只对这些写类工具限权，其余工具一律放行（Claude Code 工具名）
const GUARDED_TOOLS = ["write", "edit", "multiedit", "notebookedit"]

// 静默模式（.cannbot/settings.json 的 mode=silent）下拦截的询问类工具。
// 拦截问卷发送是机制兜底；正常流程下 QA 已按静默默认决策执行（prompt 层约束）。
// 按**子串**匹配工具名（小写后），覆盖 AskUserQuestion 等带前后缀的命名。
const SILENT_GUARDED_TOOLS = ["question", "ask"]

// 中间产物区：锚定项目根，只认根下的 .cannbot。
// 不参与下方段级匹配——否则代码树里任意一个同名目录都会拿到
// 「所有角色可写、不限文件类型」的短路放行，成为越权口子。
const INTERMEDIATE_DIR = ".cannbot"

// 其余路径分类：按相对项目根路径的**目录段**命中（段名需完全相等）。
// 用段级匹配而非前缀匹配——测试/文档目录未必在顶层、也未必是单数形式
// （如 <工程目录>/tests/），前缀匹配会把它们兜底成 code，
// 使测试角色写不了自己的目录。code 为兜底类别（无目录段命中者）。
const CATEGORY_DIR_NAMES = {
  test: ["test", "tests"],
  doc: ["doc", "docs"],
}

// 内置默认值（防御性兜底，正常流程不可达）。
// L8 约束：必须与 skills/workflow-agent-permissions/hooks/*.js 保持同步。
const DEFAULT_RULES = {
  PM:               { categories: [], exts: "*" },                          // 只写 .cannbot
  architect:        { categories: [], exts: "*" },                          // 只写 .cannbot
  qa:               { categories: [], exts: "*" },                          // 只写 .cannbot
  developer:        { categories: ["code", "test", "doc"], exts: "*" },
  "developer-code": { categories: ["code"], exts: "*" },
  "developer-test": { categories: ["test"], exts: "*" },
  "developer-doc":  { categories: ["code", "test", "doc"], exts: [".md"] }, // 各目录 md 文档
}

// 解析单个角色配置文件：提取 `export default { ... }` 的对象字面量并求值。
// 失败返回 null，外层跳过该文件。
function parseRuleFile(path) {
  const text = readFileSync(path, "utf8")
  const m = text.match(/export\s+default\s*([\s\S]*?);?\s*$/)
  if (!m) return null
  return new Function(`"use strict"; return (${m[1]})`)()
}

// 加载 .cannbot/permissions/*.js，按文件名（去 .js）建立角色规则。
// 目录缺失或为空 → 返回 null，外层回退到内置默认值。
function loadConfig(projectRoot) {
  const dir = join(projectRoot, ".cannbot", "permissions")
  const agents = {}
  let files
  try {
    files = readdirSync(dir)
  } catch {
    return null
  }
  for (const f of files) {
    if (!f.endsWith(".js")) continue
    const role = f.slice(0, -3)
    try {
      const data = parseRuleFile(join(dir, f))
      if (data && typeof data === "object") {
        agents[role] = {
          categories: Array.isArray(data.categories) ? data.categories : [],
          exts: data.exts !== undefined ? data.exts : "*",
        }
      }
    } catch (e) {
      console.error(`[permission-guard] load ${f} failed: ${e.message}`)
    }
  }
  return Object.keys(agents).length > 0 ? agents : null
}

// 相对项目根的 POSIX 风格路径（统一分隔符、去掉 ./ 前缀）
function relPosix(projectRoot, filePath) {
  let rel = relative(projectRoot, filePath)
  if (sep !== "/") rel = rel.split(sep).join("/")
  return rel
}

// 判定路径类别；写到项目根之外(以 .. 开头)返回 "external"
function classify(rel, categoryDirNames) {
  if (rel.startsWith("../") || rel === "..") return "external"
  if (rel === INTERMEDIATE_DIR || rel.startsWith(INTERMEDIATE_DIR + "/")) return "intermediate"
  const dirs = rel.split("/").slice(0, -1) // 末段是文件名，不参与目录段匹配
  for (const [cat, names] of Object.entries(categoryDirNames)) {
    if (dirs.some((d) => names.includes(d))) return cat
  }
  return "code" // 兜底：其余视为代码目录
}

function extOf(rel) {
  const base = rel.split("/").pop() || ""
  const dot = base.lastIndexOf(".")
  return dot >= 0 ? base.slice(dot).toLowerCase() : ""
}

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

  const tool = (input.tool_name || "").toLowerCase()

  // 静默模式：拦截问卷发送（任何角色都不得绕过）
  if (SILENT_GUARDED_TOOLS.some((t) => tool.includes(t))) {
    const projectRoot = process.env.CLAUDE_PROJECT_DIR || input.cwd || process.cwd()
    if (readSilentMode(projectRoot)) {
      deny(
        "静默模式已启用，问卷发送被拦截：请按静默默认决策执行——不发送问卷，落盘 " +
          '.reply.json（{"mode":"silent","decision":"accepted"}）；如需恢复交互，请让用户关闭静默模式。',
      )
    }
    return
  }

  if (!GUARDED_TOOLS.includes(tool)) return // 非写类工具放行

  const toolInput = input.tool_input || {}
  const rawPath = toolInput.file_path ?? toolInput.notebook_path
  if (!rawPath) return // 拿不到路径，不拦（避免误伤）

  const projectRoot = process.env.CLAUDE_PROJECT_DIR || input.cwd || process.cwd()
  // 相对路径按会话工作目录解析后再分类：直接放行会让相对路径成为绕过口子
  const filePath = isAbsolute(rawPath) ? rawPath : resolve(input.cwd || projectRoot, rawPath)
  const rel = relPosix(projectRoot, filePath)
  const cat = classify(rel, CATEGORY_DIR_NAMES)

  // .cannbot（中间产物区）所有角色均可写，任意文件类型——短路放行。
  // 写哪、怎么命名由任务下发时约定，不在此卡。
  if (cat === "intermediate") return

  const fileRules = loadConfig(projectRoot)
  const rules = fileRules ? { ...DEFAULT_RULES, ...fileRules } : { ...DEFAULT_RULES }

  // 主线程（无 agent_type）视为 PM
  const agent = input.agent_type || null
  const role = agent && PRIMARY_AGENTS.includes(agent) ? "PM" : agent ?? "PM"
  const rule = rules[role]

  // 未知角色（未命中规则表）
  if (!rule) {
    if (UNKNOWN_ROLE_POLICY === "deny") {
      deny(`未知角色 ${agent ?? "?"} 无写权限：${rel}`)
    }
    if (UNKNOWN_ROLE_POLICY === "allow-warn") {
      console.error(`[permission-guard] 未知角色放行(告警): agent=${agent ?? "?"} tool=${tool} path=${rel}`)
    }
    return // allow / allow-warn
  }

  const hint =
    role === "PM"
      ? " 请将此操作派发给对应的子 Agent 执行，PM 只负责调度，不直接执行。"
      : " 如果无此权限无法完成当前任务，请立即结束任务并向主 Agent 上报。"

  // 目录类别校验
  if (!rule.categories.includes(cat)) {
    deny(
      `角色 ${role} 无权写入 ${cat} 目录：${rel}（可写类别：${rule.categories.join("/")}${hint}）`,
    )
  }
  // 文件类型校验
  if (rule.exts !== "*") {
    const e = extOf(rel)
    if (!rule.exts.includes(e)) {
      deny(`角色 ${role} 只能写 ${rule.exts.join("/")} 类型，拒绝：${rel}${hint}`)
    }
  }
  // 通过：exit 0 且无输出
}

main()
