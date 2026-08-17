// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// ---------------------------------------------------------------------------
// permission-guard —— dsh（DeepSeek Harness）侧部署级权限守卫（Cordis 插件）
//
// 与 hooks/opencode/permission-guard.js、hooks/claude/permission-guard.js 同一套
// 规则语义（L8 约束：内置默认值与 workflow-agent-permissions skill 同步），
// 挂载点不同：
//   opencode → <install>/.opencode/plugin/（项目级，init 链接）
//   claude   → <install>/.claude/hooks/ + settings.json 注册（项目级，init 链接）
//   dsh      → $DSH_HOME/cordis.patch.yml（home 级 patch 层，对所有 profile 生效；
//              由 hooks/dsh/install.sh 安装，非 init 自动挂载）
//
// 机制（基于 dsh-tools 工具管道）：
//   tools/pre-execute 是工具调用的可扩展 allow/deny 瀑布门。本插件以全局作用域
//   监听该事件：违规 → 返回 { kind: "deny", reason } 短接拒绝（原因回传模型）；
//   合规/非受管 → 调用 next() 委托后续判定（默认放行）。
//
// 工作区识别：仅当会话 cwd 下存在 .cannbot/permissions/（cuda2ascend init 产物）
// 时启用；其它工作区一律放行（本插件挂在用户 home 层，不能影响无关项目）。
//
// 角色识别：
//   主 Agent（session header 无 origin: "subagent"）→ PM；
//   子 Agent → 从持久 label 解析角色。label 即 PM 派发时传入的 description，
//   约定必须含角色名（如「architect：需求分析」）。label 经
//   ctx.subagents.listChildren(parentSessionId) 枚举解析（按子会话 id 匹配）。
//   解析失败/未知角色 → 按 UNKNOWN_ROLE_POLICY（deny）处理（.cannbot 除外）。
//
// 静默问卷拦截：.cannbot/settings.json 的 mode=silent 时，拦截问卷类工具
// （ask_user_question 等，按工具名子串匹配），恢复与 opencode/claude 一致的
// 机制兜底。
//
// 配置来源：<cwd>/.cannbot/permissions/*.js（每角色一文件，ESM export default
//   { categories, exts }），缺失时回退内置默认值（防御兜底）。
// ---------------------------------------------------------------------------

import { readFileSync, readdirSync, statSync } from "node:fs"
import { isAbsolute, join, relative, resolve, sep } from "node:path"
import { pathToFileURL } from "node:url"

// 视为主 Agent（PM）的判定：session header 无 subagent origin 即主 Agent。
// （与 opencode 侧 PRIMARY_AGENTS 语义对齐：主会话统一按 PM 授权）
const PRIMARY_ROLE = "PM"

// 未知角色（未在规则表中命中）策略：allow-warn | allow | deny
// deny：规则表已枚举全部角色，未命中即为配置异常或越权调用；
// 且 .cannbot 写入在分类阶段已短路放行，deny 只影响代码/测试/文档目录。
const UNKNOWN_ROLE_POLICY = "deny"

// 只对这些写类工具限权，其余工具一律放行（与 opencode 侧一致；
// dsh 当前实际提供 write/edit，保留列表以兼容未来新增）
const GUARDED_TOOLS = ["write", "edit", "patch", "multiedit"]

// 静默模式（.cannbot/settings.json 的 mode=silent）下拦截的询问类工具。
// 拦截问卷发送是机制兜底；正常流程下 QA 已按静默默认决策执行（prompt 层约束）。
// 按**子串**匹配工具名（小写后），覆盖带前后缀的同类命名
// （dsh 的 ask_user_question 命中 "ask"）。
const SILENT_GUARDED_TOOLS = ["question", "ask"]

// 中间产物区：锚定项目根，只认根下的 .cannbot。
const INTERMEDIATE_DIR = ".cannbot"

// 其余路径分类：按相对项目根路径的**目录段**命中（段名需完全相等）。
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

// 角色名 token：按最长优先排列（developer-code 必须先于 developer 匹配）
const ROLE_TOKEN_RE = /\b(PM|pm|architect|qa|developer-code|developer-test|developer-doc|developer)\b/

// ---------------------------------------------------------------------------
// 纯决策逻辑（与 opencode 侧逐条对齐；导出便于独立测试）
// ---------------------------------------------------------------------------

// 加载 .cannbot/permissions/*.js，按文件名（去 .js）建立角色规则。
// 目录缺失或为空 → 返回 null，外层回退到内置默认值。
export async function loadConfig(projectRoot) {
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
export function relPosix(projectRoot, filePath) {
  let rel = relative(projectRoot, filePath)
  if (sep !== "/") rel = rel.split(sep).join("/")
  return rel
}

// 判定路径类别；写到项目根之外（以 .. 开头）返回 "external"
export function classify(rel, categoryDirNames) {
  if (rel.startsWith("../") || rel === "..") return "external"
  if (rel === INTERMEDIATE_DIR || rel.startsWith(INTERMEDIATE_DIR + "/")) return "intermediate"
  const dirs = rel.split("/").slice(0, -1) // 末段是文件名，不参与目录段匹配
  for (const [cat, names] of Object.entries(categoryDirNames)) {
    if (dirs.some((d) => names.includes(d))) return cat
  }
  return "code" // 兜底：其余视为代码目录
}

export function extOf(rel) {
  const base = rel.split("/").pop() || ""
  const dot = base.lastIndexOf(".")
  return dot >= 0 ? base.slice(dot).toLowerCase() : ""
}

// 读取 .cannbot/settings.json 的静默开关（每次调用实时读——会话中
// 「关闭静默模式」后立即解除拦截；读失败按非静默处理，避免误伤）
export function readSilentMode(projectRoot) {
  try {
    const data = JSON.parse(
      readFileSync(join(projectRoot, ".cannbot", "settings.json"), "utf8"),
    )
    return data && data.mode === "silent"
  } catch {
    return false
  }
}

// 从持久 label（PM 派发子 Agent 的 description）解析角色名
export function parseRoleFromLabel(label) {
  if (!label) return null
  const m = String(label).match(ROLE_TOKEN_RE)
  if (!m) return null
  return m[1] === "pm" ? "PM" : m[1]
}

// 判定一次写类调用是否合规：返回 null = 放行，返回字符串 = 拒绝原因
export async function decideWrite({ projectRoot, tool, filePath, role, rules }) {
  const rawPath = filePath
  if (!rawPath) return null // 拿不到路径，不拦（避免误伤）

  // 相对路径按项目根解析后再分类：直接放行会让相对路径成为绕过口子
  const abs = isAbsolute(rawPath) ? rawPath : resolve(projectRoot, rawPath)
  const rel = relPosix(projectRoot, abs)
  const cat = classify(rel, CATEGORY_DIR_NAMES)

  // .cannbot（中间产物区）所有角色均可写，任意文件类型——短路放行。
  if (cat === "intermediate") return null

  const rule = role ? rules[role] : null

  // 未知角色（未命中规则表）
  if (!rule) {
    if (UNKNOWN_ROLE_POLICY === "deny") {
      return `[permission-guard] 未知角色 ${role ?? "?"} 无写权限：${rel}`
    }
    if (UNKNOWN_ROLE_POLICY === "allow-warn") {
      console.warn(`[permission-guard] 未知角色放行(告警): role=${role ?? "?"} tool=${tool} path=${rel}`)
    }
    return null // allow / allow-warn
  }

  const hint = (() => {
    if (role === "PM")
      return " 请将此操作派发给对应的子 Agent 执行，PM 只负责调度，不直接执行。"
    return " 如果无此权限无法完成当前任务，请立即结束任务并向主 Agent 上报。"
  })()

  // 目录类别校验
  if (!rule.categories.includes(cat)) {
    return (
      `[permission-guard] 角色 ${role} 无权写入 ${cat} 目录：${rel}` +
      `（可写类别：${rule.categories.join("/")}${hint})`
    )
  }
  // 文件类型校验
  if (rule.exts !== "*") {
    const e = extOf(rel)
    if (!rule.exts.includes(e)) {
      return `[permission-guard] 角色 ${role} 只能写 ${rule.exts.join("/")} 类型，拒绝：${rel}${hint}`
    }
  }
  return null // 通过
}

// ---------------------------------------------------------------------------
// Cordis 插件装配
// ---------------------------------------------------------------------------

const name = "cannbot-permission-guard"

function apply(ctx) {
  // 角色缓存：按 agent session id，仅缓存成功解析的结果（label 持久稳定；
  // 不缓存 null——瞬态失败（listChildren 异常等）不应把角色永久判为未知）
  const roleCache = new Map()
  // 规则缓存：按项目根 + permissions 目录 mtime 失效
  const rulesCache = new Map()

  async function getRules(projectRoot) {
    const dir = join(projectRoot, ".cannbot", "permissions")
    let mtime = 0
    try {
      mtime = statSync(dir).mtimeMs
    } catch {
      return { ...DEFAULT_RULES }
    }
    const hit = rulesCache.get(projectRoot)
    if (hit && hit.mtime === mtime) return hit.rules
    const fileAgents = await loadConfig(projectRoot)
    const rules = fileAgents ? { ...DEFAULT_RULES, ...fileAgents } : { ...DEFAULT_RULES }
    rulesCache.set(projectRoot, { mtime, rules })
    return rules
  }

  // 解析调用方角色：主 Agent → PM；子 Agent → 持久 label → 角色名
  async function resolveRole(ctx, agent, header) {
    if (!agent) return null
    const sid = agent.id
    if (roleCache.has(sid)) return roleCache.get(sid)
    let role = null
    if (!header || header.origin !== "subagent") {
      role = PRIMARY_ROLE // 主 Agent（无 subagent origin）→ PM
    } else {
      const parentId = header.parentSession
      const label = await findLabel(ctx, parentId, sid)
      role = label ? parseRoleFromLabel(label) : null
    }
    // 仅缓存非 null：解析失败（瞬态错误/未命中）下次调用重试，避免永久误判
    if (role) roleCache.set(sid, role)
    return role
  }

  // 通过子 Agent 枚举查持久 label（listChildren 暴露 descriptor 的创建 label）
  async function findLabel(ctx, parentId, childId) {
    const subagents = ctx.get("subagents")
    if (!parentId || !subagents) return null
    try {
      const children = await subagents.listChildren(parentId)
      for (const c of children) {
        if (c.kind === "child" && c.id === childId) return c.label ?? null
      }
      return null
    } catch (e) {
      ctx.logger?.warn?.("[permission-guard] dsh: resolve label failed: %s", e?.message)
      return null
    }
  }

  // tools/pre-execute 瀑布：返回 deny 即短接拒绝；否则委托 next()
  ctx.on("tools/pre-execute", async (exec, next) => {
    try {
      const agent = exec.agent
      const header = agent?.session?.header
      const cwd = header?.cwd
      if (!cwd) return next()

      // 工作区识别：非 cuda2ascend 初始化工作区（无 .cannbot/permissions）一律放行
      const permDir = join(cwd, ".cannbot", "permissions")
      let hasPerm = false
      try {
        statSync(permDir)
        hasPerm = true
      } catch {
        /* 目录缺失 = 非受管工作区 */
      }
      if (!hasPerm) return next()

      const tool = (exec.name || "").toLowerCase()

      // 静默模式：拦截问卷发送（任何角色都不得绕过；与 opencode/claude 同语义）
      if (SILENT_GUARDED_TOOLS.some((t) => tool.includes(t))) {
        if (readSilentMode(cwd)) {
          return {
            kind: "deny",
            reason:
              "[permission-guard] 静默模式已启用，问卷发送被拦截：请按静默默认决策执行——" +
              "不发送问卷，落盘 .reply.json（{\"mode\":\"silent\",\"decision\":\"accepted\"}）；" +
              "如需恢复交互，请让用户关闭静默模式。",
          }
        }
        return next()
      }

      if (!GUARDED_TOOLS.includes(tool)) return next() // 非写类工具放行

      const rawPath = exec.arguments?.file_path ?? exec.arguments?.path
      const role = await resolveRole(ctx, agent, header)
      const rules = await getRules(cwd)
      const denial = await decideWrite({ projectRoot: cwd, tool, filePath: rawPath, role, rules })
      if (denial) return { kind: "deny", reason: denial }
      return next()
    } catch (e) {
      // 意外错误 fail-open（避免拖垮无关调用），记录日志
      ctx.logger?.warn?.("[permission-guard] dsh: unexpected error, allowing: %s", e?.message)
      return next()
    }
  })
}

export { name, apply }
