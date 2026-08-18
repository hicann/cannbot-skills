// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// Plugin metadata consistency validator.
//
// plugin.json is the single source of truth for the `agents` field. The yml
// files in plugins.d/ must NOT carry an `agents` count or `installAgents`
// list (those would drift from plugin.json). This module enforces that
// invariant plus on-disk existence of every declared agent .md and skill
// source directory, so that drift is caught at build/test time rather than
// surfacing as "Agent 未找到" warnings at install time.
//
// Checks:
//   V1  every plugin yml's target dir has a parseable .claude-plugin/plugin.json (WARN if missing)
//   V2  every discovered agent resolves to agents/<name>.md on disk
//       (discovered from plugin.json agents[], fallback to init.sh INCLUDED_AGENT_PATTERN)
//   V3  every yml installSkills entry's source `dir` and `<dir>/<skill>` exist on disk
//   V4  yml `skills` count == total installSkills entries
//   V5  yml must NOT contain `agents` count (always derived); `installAgents` allowed
//       (acts as Layer 3 fallback + documentation); when plugin.json has agents,
//       yml installAgents must match (consistency check)
//   V6  main package.json optionalDependencies versions == main `version` field
//   V7  yml must NOT contain `version` key; plugin.json must have a non-empty version (SoT)
//
// Usage as a module:
//   const { validatePlugins } = require("./validate-plugins.cjs");
//   const result = validatePlugins(repoRoot, installHelperRoot);
//   if (!result.ok) { console.error(result.report); process.exit(1); }
//
// Usage as a CLI:
//   node scripts/validate-plugins.cjs                 # repoRoot auto-detected
//   node scripts/validate-plugins.cjs --repo <path>   # explicit repo root

const fs = require("fs");
const path = require("path");
const yaml = require("yaml");

function loadScanDirs() {
  try {
    const configPath = path.join(__dirname, "..", "src", "config", "repository.yaml");
    const config = yaml.parse(fs.readFileSync(configPath, "utf-8"));
    if (Array.isArray(config.scanDirs) && config.scanDirs.length > 0) {
      return config.scanDirs;
    }
  } catch {}
  return ["ops", "model", "graph", "infra", "runtime"];
}

const SCAN_DIRS = loadScanDirs();

function readPluginJson(pluginDir) {
  const pjPath = path.join(pluginDir, ".claude-plugin", "plugin.json");
  try {
    return JSON.parse(fs.readFileSync(pjPath, "utf-8"));
  } catch {
    return null;
  }
}

function resolveAgentsFromPluginJson(pj) {
  if (!pj || !Array.isArray(pj.agents)) return [];
  return pj.agents
    .filter((a) => a.startsWith("./agents/"))
    .map((a) => a.replace(/^\.\/agents\//, "").replace(/\.md$/, ""))
    .filter(Boolean);
}

function parseIncludedAgentPattern(initShPath) {
  try {
    const text = fs.readFileSync(initShPath, "utf-8");
    const m = text.match(/INCLUDED_AGENT_PATTERN="([^"]*)"/);
    if (!m) return null;
    return m[1].trim() || null;
  } catch {
    return null;
  }
}

function simpleGlobToRegExp(pattern) {
  let re = "^";
  for (let i = 0; i < pattern.length; i++) {
    const c = pattern[i];
    if (c === "*") re += ".*";
    else if (c === "?") re += ".";
    else re += c.replace(/[.+^${}()|[\]\\]/g, "\\$&");
  }
  return new RegExp(re + "$");
}

function matchAgentsByPattern(agentsDir, pattern) {
  const filePattern = pattern.endsWith(".md") ? pattern : pattern + ".md";
  try {
    if (typeof fs.globSync === "function") {
      return fs.globSync(filePattern, { cwd: agentsDir })
        .map((f) => f.replace(/\.md$/, "").replace(/\\/g, "/"));
    }
  } catch {}
  if (filePattern.includes("@(") || filePattern.includes("+(") || filePattern.includes("!(")) {
    return [];
  }
  try {
    const re = simpleGlobToRegExp(filePattern);
    return fs.readdirSync(agentsDir)
      .filter((f) => f.endsWith(".md") && re.test(f))
      .map((f) => f.replace(/\.md$/, ""));
  } catch {
    return [];
  }
}

function discoverAgents(pluginDir, pj, initShPath, fallbackAgents) {
  const fromPj = resolveAgentsFromPluginJson(pj);
  if (fromPj.length > 0) return fromPj;
  if (pj && Array.isArray(pj.agents) && pj.agents.length === 0) return [];
  const pattern = parseIncludedAgentPattern(initShPath);
  if (pattern) {
    const agentsDir = path.join(pluginDir, "agents");
    if (fs.existsSync(agentsDir)) {
      const fromPattern = matchAgentsByPattern(agentsDir, pattern);
      if (fromPattern.length > 0) return fromPattern;
    }
  }
  // Layer 3: yml installAgents fallback (last resort)
  return fallbackAgents || [];
}

function collectYmlSkillNames(installSkills) {
  const set = new Set();
  if (!installSkills) return set;
  for (const src of installSkills) {
    for (const sk of src.skills) {
      const name = typeof sk === "string" ? sk : sk.name;
      const as = typeof sk === "string" ? sk : (sk.as || sk.name);
      set.add(name);
      if (as !== name) set.add(as);
    }
  }
  return set;
}

function countSkillEntries(installSkills) {
  if (!installSkills) return 0;
  let n = 0;
  for (const src of installSkills) n += src.skills.length;
  return n;
}

function findSkillOnDisk(skillName, repoRoot, pluginDir) {
  for (const dir of SCAN_DIRS) {
    const candidate = path.join(repoRoot, dir, skillName);
    if (fs.existsSync(candidate) && fs.statSync(candidate).isDirectory()) {
      return true;
    }
  }
  if (pluginDir) {
    const localCandidate = path.join(pluginDir, "skills", skillName);
    if (fs.existsSync(localCandidate) && fs.statSync(localCandidate).isDirectory()) {
      return true;
    }
  }
  return false;
}

function loadYml(file) {
  try {
    return yaml.parse(fs.readFileSync(file, "utf-8")) || {};
  } catch {
    return null;
  }
}

/**
 * Validate plugin metadata consistency.
 * @param {string} repoRoot  absolute path to the cannbot-skills repo root
 * @param {string} installHelperRoot  absolute path to install-helper dir (contains plugins.d/ + package.json)
 * @returns {{ ok: boolean, report: string, errors: Array<{plugin:string, check:string, detail:string}> }}
 */
function validatePlugins(repoRoot, installHelperRoot) {
  const pluginsDir = path.join(installHelperRoot, "plugins.d");
  const errors = [];

  function addError(plugin, check, detail) {
    errors.push({ plugin, check, detail });
  }

  for (const file of fs.readdirSync(pluginsDir)) {
    if (file.startsWith("_") || !file.endsWith(".yml")) continue;
    const content = loadYml(path.join(pluginsDir, file));
    if (!content || !content.id) continue;
    const id = content.id;
    const pluginDir = path.join(repoRoot, content.dir);

    // V1: plugin.json parseable (WARN if missing — agents can be discovered via init.sh pattern)
    const pj = readPluginJson(pluginDir);
    if (!pj) {
      console.warn(`  ⚠ [${id}] V1: no .claude-plugin/plugin.json — agents will be discovered via init.sh INCLUDED_AGENT_PATTERN fallback`);
    }

    // V5: agents count is always forbidden (derived at build/runtime).
    // installAgents is allowed (acts as Layer 3 fallback + documentation, like installSkills).
    // When plugin.json provides agents, yml installAgents must match (consistency check).
    // When plugin.json is missing or has no agents field, yml installAgents is the source (no check).
    if (content.agents !== undefined) {
      addError(id, "V5", `yml has 'agents' field (=${content.agents}); remove it — agents count is always derived`);
    }
    if (content.installAgents !== undefined && pj && Array.isArray(pj.agents)) {
      // plugin.json has agents → yml installAgents must match (consistency)
      const pjAgents = resolveAgentsFromPluginJson(pj).sort();
      const ymlAgents = [...content.installAgents].sort();
      if (JSON.stringify(pjAgents) !== JSON.stringify(ymlAgents)) {
        addError(id, "V5", `yml installAgents (${JSON.stringify(content.installAgents)}) does not match plugin.json agents (${JSON.stringify(pjAgents)}); sync them to avoid drift`);
      }
    }

    // V7a: yml must NOT carry version (plugin.json is SoT)
    if (content.version !== undefined) {
      addError(id, "V7", `yml has 'version' field (=${content.version}); remove it — plugin.json is the single source of truth`);
    }
    // V7b: plugin.json must have a non-empty version (so removing yml version leaves a value)
    if (pj && (!pj.version || String(pj.version).trim() === "")) {
      addError(id, "V7", `plugin.json has no non-empty 'version'; plugin.json must provide version since yml no longer does`);
    }

    // V2: every discovered agent resolves to agents/<name>.md on disk.
    // Agents are discovered: Layer 1 plugin.json → Layer 2 init.sh pattern → Layer 3 yml installAgents.
    const initShPath = path.join(pluginDir, "init.sh");
    const agents = discoverAgents(pluginDir, pj, initShPath, content.installAgents);
    const agentsDir = path.join(pluginDir, "agents");
    for (const agentName of agents) {
      const agentFile = agentName.endsWith(".md") ? agentName : `${agentName}.md`;
      const agentPath = path.join(agentsDir, agentFile);
      if (!fs.existsSync(agentPath)) {
        addError(id, "V2", `discovered agent ${agentFile} does not exist at ${content.dir}/agents/`);
      }
    }

    // V3: every installSkills source dir and skill exist on disk
    if (Array.isArray(content.installSkills)) {
      for (const src of content.installSkills) {
        const sourceDir = path.join(repoRoot, src.dir);
        if (!fs.existsSync(sourceDir) || !fs.statSync(sourceDir).isDirectory()) {
          addError(id, "V3", `installSkills source dir '${src.dir}' does not exist on disk`);
          continue;
        }
        for (const sk of src.skills) {
          const skillName = typeof sk === "string" ? sk : sk.name;
          const skillAlias = typeof sk === "string" ? sk : (sk.as || sk.name);
          // skill source can live either under src.dir/<skillName> or in a SCAN_DIR/<skillName>
          const underSourceDir = path.join(sourceDir, skillName);
          if (fs.existsSync(underSourceDir) && fs.statSync(underSourceDir).isDirectory()) {
            continue;
          }
          if (findSkillOnDisk(skillName, repoRoot, pluginDir)) {
            continue;
          }
          addError(id, "V3", `installSkills references skill '${skillName}' (dir: ${src.dir}) but it does not exist on disk`);
          // alias also must exist as the installed symlink target resolves to skillName; checked via skillName above
          void skillAlias;
        }
      }
    }

    // V4: yml skills count == total installSkills entries
    const declared = content.skills || 0;
    const actual = countSkillEntries(content.installSkills);
    if (declared !== actual) {
      addError(id, "V4", `yml 'skills: ${declared}' but installSkills has ${actual} entries`);
    }
  }

  // V6: main package.json optionalDependencies versions == main version
  const mainPkgPath = path.join(installHelperRoot, "package.json");
  try {
    const mainPkg = JSON.parse(fs.readFileSync(mainPkgPath, "utf-8"));
    const mainVersion = mainPkg.version;
    const optDeps = mainPkg.optionalDependencies || {};
    for (const [name, ver] of Object.entries(optDeps)) {
      if (ver !== mainVersion) {
        addError("install-helper", "V6", `optionalDependencies['${name}'] = '${ver}' but package.json version = '${mainVersion}'`);
      }
    }
  } catch (e) {
    addError("install-helper", "V6", `failed to read/parse install-helper package.json: ${e.message}`);
  }

  const ok = errors.length === 0;
  const report = buildReport(errors);
  return { ok, report, errors };
}

function buildReport(errors) {
  if (errors.length === 0) {
    return "OK: all plugin metadata checks passed.";
  }
  const lines = [`FAILED: ${errors.length} plugin metadata check(s) failed:\n`];
  for (const e of errors) {
    lines.push(`  [${e.plugin}] ${e.check}: ${e.detail}`);
  }
  lines.push("");
  lines.push("plugin.json is the single source of truth for agents and version.");
  lines.push("Remove any 'agents'/'installAgents'/'version' fields from plugins.d/*.yml");
  lines.push("and ensure plugin.json agents match the agents/ directory on disk.");
  return lines.join("\n");
}

// ---- CLI entrypoint ----
function autoDetectRoots() {
  const scriptDir = __dirname;
  const installHelperRoot = path.join(scriptDir, "..");
  const repoRoot = path.join(installHelperRoot, "..", "..");
  return { repoRoot, installHelperRoot };
}

if (require.main === module) {
  const args = process.argv.slice(2);
  let repoRoot;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--repo" && args[i + 1]) {
      repoRoot = args[i + 1];
      i++;
    }
  }
  const detected = autoDetectRoots();
  repoRoot = repoRoot || detected.repoRoot;
  const { installHelperRoot } = detected;
  const result = validatePlugins(repoRoot, installHelperRoot);
  if (result.ok) {
    console.log(result.report);
    process.exit(0);
  } else {
    console.error(result.report);
    process.exit(1);
  }
}

module.exports = { validatePlugins, buildReport, autoDetectRoots };
