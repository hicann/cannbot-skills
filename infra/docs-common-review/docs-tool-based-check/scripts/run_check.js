#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
/**
 * docs-tool-based-check 运行器（零第三方依赖，需 Node.js >= 18）
 *
 * 职责：校验配置 → 解析配置优先级（项目 .doctools/ 优先）→ 拉取/复用 pin 版
 *       docs-ci-v2.js → 组装 ciConfig → 执行检查 → result.json 转换为统一
 *       12 字段结果 JSON → 写入 output → 清理临时目录。
 *
 * 用法：
 *   node run_check.js --targets=<路径1,路径2,...> [--checks=<检查项,...>]
 *                     [--project-root=<项目根目录>] [--output=<结果文件路径>]
 *                     [--disable-external] [--proxy=<代理地址>] [--concurrency=<n>]
 *
 * 退出码：0 = 执行完成（含 0 问题）；1 = 配置/环境/执行错误。
 */
"use strict";

const fs = require("fs");
const path = require("path");
const os = require("os");
const { spawn } = require("child_process");

// ---------- 常量 ----------
// 已验证的脚本版本（与 SKILL.md 记录一致）。更新时：修改此值并重跑 evals 验证。
const SCRIPT_SHA = "c5ccfec7c5550b2a8a45feb26e7cd5fac4bd607b";
const SCRIPT_URL =
  "https://raw.gitcode.com/cann/.gitcode/raw/" + SCRIPT_SHA +
  "/actions/staticcheck/dist/staticcheck/docs-ci/docs-ci-v2.js";
const LEGAL_CHECKS = [
  "markdownlint",
  "tag-closed-check",
  "link-validity-check",
  "resource-existence-check",
  "codespell-check",
];
// 注意：docs-ci-v2.js 另支持 toc-check / file-naming-check /
// file-naming-consistency-check，本技能暂不暴露，列为非法输入。
const SNIPPET_MAX = 200;

const SKILL_DIR = path.resolve(__dirname, "..");

// ---------- 工具函数 ----------
function toPosix(p) {
  return String(p).replace(/\\/g, "/");
}
function escapeRegex(s) {
  return String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
function fail(msg) {
  console.error("[error] " + msg);
  process.exit(1);
}
function readJson(file, what) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (e) {
    fail(what + " 不是合法 JSON：" + file + "（" + e.message + "）");
  }
}
function rmDir(dir) {
  fs.rmSync(dir, { recursive: true, force: true });
}
function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--disable-external") {
      args.disableExternal = true;
      continue;
    }
    if (a === "--help" || a === "-h") {
      args.help = true;
      continue;
    }
    if (a.startsWith("--")) {
      const eq = a.indexOf("=");
      let key = eq === -1 ? a.slice(2) : a.slice(2, eq);
      let val = eq === -1 ? argv[++i] : a.slice(eq + 1);
      args[key] = val;
    }
  }
  return args;
}
const HELP = `用法: node run_check.js --targets=<路径1,路径2,...> [选项]
选项:
  --targets         必需。待检查的文件或目录，多个用英文逗号分隔
  --checks          可选。本次检查项，逗号分隔，合法值: ${LEGAL_CHECKS.join("、")}
                    不传时使用技能目录 config.json 的 ci 数组（默认全部 5 项）
  --project-root    可选。项目根目录（用于 .doctools/ 配置解析与结果输出），默认当前工作目录
  --output          可选。结果文件路径，默认 <project-root>/output/tool-based-check-result.json
  --disable-external 可选。跳过外链/外部资源检查（内网或无外网环境使用）
  --proxy           可选。访问外链使用的代理地址
  --concurrency     可选。并发检查的文件数（默认 1，文件多时可调大）`;

// ---------- 参数与环境 ----------
const args = parseArgs(process.argv.slice(2));
if (args.help) {
  console.log(HELP);
  process.exit(0);
}
if (!args.targets) fail("缺少必需参数 --targets（待检查的文件或目录，多个用英文逗号分隔）");
if (typeof globalThis.fetch !== "function") {
  fail("当前 Node.js 不支持 fetch，需要 Node.js 18 及以上（node --version 查看）");
}
const major = Number(process.versions.node.split(".")[0]);
if (major < 18) {
  console.error("[warn] 当前 Node.js 版本为 " + process.versions.node + "，建议升级到 18 及以上");
}

const projectRoot = path.resolve(args["project-root"] || process.cwd());
const skillDir = toPosix(SKILL_DIR);
const outputDefault = path.join(projectRoot, "output", "tool-based-check-result.json");
const outputFile = path.resolve(args.output || outputDefault);

// ---------- 检查目标 ----------
const targetList = String(args.targets).split(",").map((s) => s.trim()).filter(Boolean);
if (targetList.length === 0) fail("--targets 为空");
const absTargets = targetList.map((t) => path.resolve(projectRoot, t));
for (const t of absTargets) {
  if (!fs.existsSync(t)) fail("检查目标不存在：" + t);
}

// 计算公共根目录（repoPath）
function commonRoot(dirs) {
  if (dirs.length === 1) return dirs[0];
  const split = dirs.map((d) => d.split(path.sep));
  const first = split[0];
  let i = 0;
  while (i < first.length && split.every((s) => s[i] === first[i])) i++;
  return first.slice(0, Math.max(1, i)).join(path.sep);
}
const repoRoot = commonRoot(absTargets.map((t) => (fs.statSync(t).isDirectory() ? t : path.dirname(t))));

// 生成 docPath 锚定正则
const patterns = absTargets.map((t) => {
  const st = fs.statSync(t);
  let rel = toPosix(path.relative(repoRoot, t));
  if (rel === "") return ".*";
  return st.isDirectory() ? escapeRegex(rel) + "/.*" : "^" + escapeRegex(rel) + "$";
});
const docPath = patterns.length === 1 ? patterns[0] : "^(" + patterns.join("|") + ")$";

// ---------- 检查项范围 ----------
let scope = null;
if (args.checks) {
  scope = String(args.checks).split(",").map((s) => s.trim()).filter(Boolean);
}
if (!scope) {
  const skillConfigFile = path.join(SKILL_DIR, "config.json");
  if (fs.existsSync(skillConfigFile)) {
    const cfg = readJson(skillConfigFile, "技能目录 config.json");
    const ci = cfg && cfg.config && cfg.config.base && cfg.config.base.ci;
    if (Array.isArray(ci) && ci.length > 0) scope = ci.map(String);
  }
}
if (!scope) scope = LEGAL_CHECKS.slice();
for (const c of scope) {
  if (!LEGAL_CHECKS.includes(c)) {
    fail(
      '非法检查项："' + c + '"。合法值仅：' + LEGAL_CHECKS.join("、") +
      "（toc-check 等其他项暂不暴露）。请修正后重试"
    );
  }
}

// ---------- 配置文件解析（项目 .doctools/ 优先） ----------
const doctoolsDir = path.join(projectRoot, ".doctools");
const projectFile = (name) => path.join(doctoolsDir, name);
const skillFile = (name) => path.join(SKILL_DIR, name);

function fileExistsSafe(f) {
  try {
    return fs.existsSync(f);
  } catch {
    return false;
  }
}

// markdownlint 规则配置：项目存在则整体替换脚本内置默认配置
let markdownlintUrl = null;
const projMd = projectFile("markdownlint.config.json");
if (fileExistsSafe(projMd)) {
  readJson(projMd, "项目 .doctools/markdownlint.config.json");
  markdownlintUrl = toPosix(projMd);
}

// codespell 白名单：内置社区默认（whitelist_words.txt）+ 项目增量，合并为一份
const projCs = projectFile("codespell.whitelist.txt");
const skillCs = skillFile("whitelist_words.txt");
let codespellLines = [];
let csSource = [];
if (fileExistsSafe(skillCs)) {
  codespellLines.push(fs.readFileSync(skillCs, "utf8"));
  csSource.push("内置社区默认 whitelist_words.txt");
}
if (fileExistsSafe(projCs)) {
  codespellLines.push(fs.readFileSync(projCs, "utf8"));
  csSource.push("项目 .doctools/codespell.whitelist.txt");
}

// 链接/资源白名单：内置社区默认（url.whitelist.json）+ 项目增量，合并为一份
const projUrl = projectFile("url.whitelist.txt");
const skillUrlJson = skillFile("url.whitelist.json");
let urlLines = [];
let urlSource = [];
if (fileExistsSafe(skillUrlJson)) {
  const arr = readJson(skillUrlJson, "技能目录 url.whitelist.json");
  if (Array.isArray(arr)) {
    urlLines.push(arr.map(String).join("\n"));
    urlSource.push("内置社区默认 url.whitelist.json");
  } else {
    fail("技能目录 url.whitelist.json 应为字符串数组：" + skillUrlJson);
  }
}
if (fileExistsSafe(projUrl)) {
  urlLines.push(fs.readFileSync(projUrl, "utf8"));
  urlSource.push("项目 .doctools/url.whitelist.txt");
}

// ---------- 准备 pin 版脚本 ----------
const cacheDir = path.join(os.tmpdir(), "docs-tool-based-check");
const cacheFile = path.join(cacheDir, "docs-ci-v2." + SCRIPT_SHA.slice(0, 10) + ".js");
async function ensureScript() {
  if (fileExistsSafe(cacheFile)) {
    console.log("[helper] 复用已缓存的检查脚本（版本 " + SCRIPT_SHA.slice(0, 10) + "）");
    return;
  }
  console.log("[helper] 下载检查脚本（版本 " + SCRIPT_SHA.slice(0, 10) + "）...");
  fs.mkdirSync(cacheDir, { recursive: true });
  let res;
  try {
    res = await fetch(SCRIPT_URL);
  } catch (e) {
    fail("脚本下载失败（" + e.message + "）。请检查网络/代理后重试，" +
      "或用浏览器从以下地址下载后放到：" + toPosix(cacheFile) + "\n" + SCRIPT_URL);
  }
  if (!res.ok) {
    fail("脚本下载失败，HTTP " + res.status + "。下载地址：" + SCRIPT_URL);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  if (buf.length < 1000 || buf.slice(0, 3).toString("utf8").startsWith("<!")) {
    fail("脚本下载内容异常（非 JS 文件），请检查下载地址：" + SCRIPT_URL);
  }
  fs.writeFileSync(cacheFile, buf);
  console.log("[helper] 脚本已缓存至 " + toPosix(cacheFile));
}

// ---------- 组装运行时配置 ----------
const runDir = path.join(cacheDir, "run-" + process.pid + "-" + Date.now());
fs.mkdirSync(runDir, { recursive: true });

const base = {};
if (codespellLines.length > 0) {
  const merged = path.join(runDir, "codespell.whitelist.txt");
  fs.writeFileSync(merged, codespellLines.join("\n"));
  base.codespellCheckRemoteWhitelistUrl = toPosix(merged);
}
if (urlLines.length > 0) {
  const merged = path.join(runDir, "url.whitelist.txt");
  fs.writeFileSync(merged, urlLines.join("\n"));
  base.linkValidityCheckRemoteWhitelistUrl = toPosix(merged);
  base.resourceExistenceCheckRemoteWhitelistUrl = toPosix(merged);
}
if (markdownlintUrl) {
  base.markdownlintRemoteConfigUrl = markdownlintUrl;
}
if (args.disableExternal) {
  base.linkValidityCheckDisableExternalUrl = true;
  base.resourceExistenceCheckDisableExternalUrl = true;
}
base.docPath = docPath;
base.ci = scope;

const ciConfig = {
  name: "docs-tool-based-check",
  config: { base: base },
  repo: { all: { extendConfig: "base" } },
};
const ciConfigFile = path.join(runDir, "ciConfig.json");
fs.writeFileSync(ciConfigFile, JSON.stringify(ciConfig, null, 2));

const resultJsonFile = path.join(runDir, "result.json");
const outputMdFile = path.join(runDir, "output.md");

// ---------- 执行 ----------
console.log("[helper] 检查目标: " + targetList.join("、"));
console.log("[helper] 检查范围: " + scope.join("、"));
console.log("[helper] 检查根目录: " + toPosix(repoRoot));
if (csSource.length) console.log("[helper] codespell 白名单来源: " + csSource.join(" + "));
if (urlSource.length) console.log("[helper] 链接白名单来源: " + urlSource.join(" + "));
if (markdownlintUrl) console.log("[helper] markdownlint 规则配置: 项目 .doctools/markdownlint.config.json（替换默认）");
if (args.disableExternal) console.log("[helper] 已跳过外链/外部资源检查");

function runScript() {
  return new Promise((resolve) => {
    const scriptArgs = [
      cacheFile,
      "--repoPath=" + toPosix(repoRoot),
      "--targetOwnerRepo=all",
      "--ciConfigUrl=" + toPosix(ciConfigFile),
      "--checkAll=true",
      "--outputJson=true",
      "--outputJsonPath=" + toPosix(resultJsonFile),
      "--outputPath=" + toPosix(outputMdFile),
      "--outputCount=200",
    ];
    if (args.proxy) scriptArgs.push("--proxy=" + args.proxy);
    if (args.concurrency) scriptArgs.push("--concurrency=" + args.concurrency);
    const child = spawn(process.execPath, scriptArgs, { stdio: "inherit" });
    child.on("error", (e) => fail("脚本执行失败：" + e.message));
    child.on("close", (code) => resolve(code));
  });
}

// ---------- 结果转换 ----------
function loadReferences() {
  const attrs = readJson(path.join(SKILL_DIR, "references", "check-attributes.json"), "references/check-attributes.json");
  const mdSev = readJson(path.join(SKILL_DIR, "references", "markdownlint-severity.json"), "references/markdownlint-severity.json");
  return { attrs, mdSev };
}
function convertItems(rawItems, refs) {
  const out = [];
  for (const item of rawItems) {
    const checkType = String(item.checkType || "");
    let a = refs.attrs[checkType];
    if (!a) {
      console.error("[warn] 未知检查项 " + checkType + "，按默认属性转换");
      a = { rule_name: checkType, rule_type: "规范性", rule_level: "一般", suggestion: "{message}" };
    }
    const message = String(item.message || "");
    let snippet = String(item.content == null ? "" : item.content).replace(/\r/g, "");
    if (snippet.length > SNIPPET_MAX) snippet = snippet.slice(0, SNIPPET_MAX) + "...";

    let ruleLevel = a.rule_level;
    if (checkType === "markdownlint") {
      const m = message.match(/(MD\d{3})/i);
      const ruleId = m ? m[1].toUpperCase() : null;
      ruleLevel = (ruleId && refs.mdSev[ruleId]) || "一般";
    }
    const lineMatch = String(item.position || "").match(/第\s*(\d+)/);
    const line = lineMatch ? lineMatch[1] : "";
    const absFile = toPosix(path.join(repoRoot, String(item.filePath || "")));
    const suggestion = String(a.suggestion)
      .replace(/\{content\}/g, snippet)
      .replace(/\{message\}/g, message);

    out.push({
      rule_group: "工具检查",
      rule_item: checkType,
      rule_type: a.rule_type,
      rule_name: a.rule_name,
      snippet: snippet,
      description: message,
      rule_level: ruleLevel,
      file_name: String(item.filePath || ""),
      line_number: line,
      location: line ? absFile + ":" + line : absFile,
      suggestion: suggestion,
    });
  }
  return out;
}

// ---------- 主流程 ----------
(async () => {
  await ensureScript();
  const t0 = Date.now();
  const code = await runScript();
  if (code !== 0) {
    // 失败时保留运行目录便于诊断
    console.error("[error] 检查脚本执行失败（退出码 " + code + "）。运行目录保留供诊断：" + toPosix(runDir));
    process.exit(1);
  }
  if (!fileExistsSafe(resultJsonFile)) {
    console.error("[error] 未生成结果文件 " + toPosix(resultJsonFile) + "。运行目录保留供诊断：" + toPosix(runDir));
    process.exit(1);
  }
  const rawItems = readJson(resultJsonFile, "result.json");
  if (!Array.isArray(rawItems)) fail("result.json 不是数组，内容异常");

  const refs = loadReferences();
  const items = convertItems(rawItems, refs);

  fs.mkdirSync(path.dirname(outputFile), { recursive: true });
  fs.writeFileSync(outputFile, JSON.stringify(items, null, 2) + "\n");

  rmDir(runDir);

  const byLevel = {};
  const byCheck = {};
  for (const it of items) {
    byLevel[it.rule_level] = (byLevel[it.rule_level] || 0) + 1;
    byCheck[it.rule_item] = (byCheck[it.rule_item] || 0) + 1;
  }
  console.log("");
  console.log("[helper] 检查完成，耗时 " + ((Date.now() - t0) / 1000).toFixed(1) + "s");
  console.log("[helper] 共检出 " + items.length + " 项问题" +
    (items.length > 0 ? "（" + Object.entries(byLevel).map(([k, v]) => k + " " + v).join("、") + "）" : ""));
  if (Object.keys(byCheck).length > 0) {
    console.log("[helper] 分布: " + Object.entries(byCheck).map(([k, v]) => k + " " + v).join("、"));
  }
  console.log("[helper] 结果已写入 " + toPosix(outputFile));
})();
