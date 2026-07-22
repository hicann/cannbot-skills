// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// ---------------------------------------------------------------------------
// permission-guard —— opencode 侧动态权限插件
//
// 机制（已由 hook-probe-test 实测坐实）：
//   tool.execute.before 入参有 sessionID → client.session.get() 可拿到当前 agent 名
//   （子 Agent 有独立 sessionID、agent 名可靠）。据此对写类工具做按角色限权。
//
// 违规 → throw（阻断本次调用，原因回传模型）；放行 → 不 throw。
//
// 配置来源（按角色分开文件）：
//   <项目根>/.cannbot/permissions/*.js —— 每角色一文件，ESM export default
//   { categories, exts }。文件名即角色名（去 .js），如 developer-code.js。
//   由 init.sh Step 4.5 从 skills/workflow-agent-permissions/hooks/
//   整体复制生成；子仓 override 该 skill 时自动生效。
//
// 启动加载一次（v1 不支持热更新）。PM 启动闸口负责检测目录异常并阻断任务派发。
//
// ★ 以下 CONFIG 为工作流级约定，不暴露给仓，改这里即可。★
// ---------------------------------------------------------------------------

import { readdirSync } from "node:fs"
import { join, relative, sep } from "node:path"
import { pathToFileURL } from "node:url"

// 视为主 Agent(PM) 的 agent 名。实测主会话跑成 opencode 内置 "build"；
// 设计中主 Agent 是 PM。两者都按 PM 授权，待工作流正式运行确认后收敛。
const PRIMARY_AGENTS = ["build", "PM", "pm"]

// 未知角色（未在规则表中命中）策略：allow-warn | allow | deny
const UNKNOWN_ROLE_POLICY = "allow-warn"

// 只对这些写类工具限权，其余工具一律放行
const GUARDED_TOOLS = ["write", "edit", "patch", "multiedit"]

// 路径分类（相对项目根的前缀）。code 为兜底类别（不匹配下列任何前缀者）。
const CATEGORY_PREFIXES = {
  intermediate: [".cannbot/"],
  test: ["test/"],
  doc: ["docs/", "doc/"],
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

// 加载 .cannbot/permissions/*.js，按文件名（去 .js）建立角色规则。
// 目录缺失或为空 → 返回 null，外层回退到内置默认值。
async function loadConfig(projectRoot) {
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
      const mod = await import(pathToFileURL(join(dir, f)).href)
      const data = mod.default ?? mod
      if (data && typeof data === "object") {
        agents[role] = {
          categories: Array.isArray(data.categories) ? data.categories : [],
          exts: data.exts !== undefined ? data.exts : "*",
        }
      }
    } catch (e) {
      console.warn(`[permission-guard] load ${f} failed: ${e.message}`)
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
function classify(rel, categoryPrefixes) {
  if (rel.startsWith("../") || rel === "..") return "external"
  for (const [cat, prefixes] of Object.entries(categoryPrefixes)) {
    for (const p of prefixes) if (rel === p.replace(/\/$/, "") || rel.startsWith(p)) return cat
  }
  return "code" // 兜底：其余视为代码目录
}

function extOf(rel) {
  const base = rel.split("/").pop() || ""
  const dot = base.lastIndexOf(".")
  return dot >= 0 ? base.slice(dot).toLowerCase() : ""
}

export const PermissionGuard = async ({ client, directory, worktree }) => {
  const projectRoot = worktree || directory || process.cwd()
  const categoryPrefixes = CATEGORY_PREFIXES
  const primaryAgents = PRIMARY_AGENTS
  const unknownPolicy = UNKNOWN_ROLE_POLICY

  // 启动加载一次：文件配置按角色合并到内置默认值（v1 不支持热更新）
  const fileAgents = await loadConfig(projectRoot)
  const rules = fileAgents ? { ...DEFAULT_RULES, ...fileAgents } : { ...DEFAULT_RULES }

  const resolveRole = async (sessionID) => {
    try {
      const r = await client.session.get({ path: { id: sessionID } })
      return (r?.data ?? r)?.agent ?? null
    } catch {
      return null
    }
  }

  // 归一化角色：主 Agent 名统一映射为 "PM"
  const normRole = (agent) => (agent && primaryAgents.includes(agent) ? "PM" : agent)

  return {
    "tool.execute.before": async (input, output) => {
      const tool = (input?.tool || "").toLowerCase()
      if (!GUARDED_TOOLS.includes(tool)) return // 非写类工具放行

      const filePath = output?.args?.filePath ?? output?.args?.path
      if (!filePath) return // 拿不到路径，不拦（避免误伤）

      const rel = relPosix(projectRoot, filePath)
      const cat = classify(rel, categoryPrefixes)

      // .cannbot（中间产物区）所有角色均可写，任意文件类型——短路放行。
      // 写哪、怎么命名由任务下发时约定，不在此卡。
      if (cat === "intermediate") return

      const agent = await resolveRole(input.sessionID)
      const role = normRole(agent)
      const rule = role ? rules[role] : null

      // 未知角色（未命中规则表）
      if (!rule) {
        if (unknownPolicy === "deny") {
          throw new Error(`[permission-guard] 未知角色 ${agent ?? "?"} 无写权限：${rel}`)
        }
        if (unknownPolicy === "allow-warn") {
          console.warn(`[permission-guard] 未知角色放行(告警): agent=${agent ?? "?"} tool=${tool} path=${rel}`)
        }
        return // allow / allow-warn
      }

      const hint = (() => {
        if (role === "PM")
          return " 请将此操作派发给对应的子 Agent 执行，PM 只负责调度，不直接执行。"
        return " 如果无此权限无法完成当前任务，请立即结束任务并向主 Agent 上报。"
      })()

      // 目录类别校验
      if (!rule.categories.includes(cat)) {
        throw new Error(
          `[permission-guard] 角色 ${role} 无权写入 ${cat} 目录：${rel}（可写类别：${rule.categories.join("/")}${hint})`,
        )
      }
      // 文件类型校验
      if (rule.exts !== "*") {
        const e = extOf(rel)
        if (!rule.exts.includes(e)) {
          throw new Error(
            `[permission-guard] 角色 ${role} 只能写 ${rule.exts.join("/")} 类型，拒绝：${rel}${hint}`,
          )
        }
      }
      // 通过：不 throw
    },
  }
}
