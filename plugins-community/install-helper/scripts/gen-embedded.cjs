// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

const fs = require("fs");
const path = require("path");
const yaml = require("yaml");

const ROOT = path.join(__dirname, "..");
const PLUGINS_DIR = path.join(ROOT, "plugins.d");
const DEFAULTS_PATH = path.join(PLUGINS_DIR, "_defaults.yml");
const OUT_PATH = path.join(ROOT, "src", "embedded-plugins.json");

// scanDirs must mirror src/config/repository.yaml
const SCAN_DIRS = ["ops", "model", "graph", "infra", "ops-lab", "runtime"];

let defaults = {};
try {
  defaults = yaml.parse(fs.readFileSync(DEFAULTS_PATH, "utf-8")) || {};
} catch {}

// ---- Auto-sync helpers: read plugin self-maintained metadata ----

function readPluginJson(pluginDir) {
  const pjPath = path.join(pluginDir, ".claude-plugin", "plugin.json");
  try {
    return JSON.parse(fs.readFileSync(pjPath, "utf-8"));
  } catch {
    return null;
  }
}

function parseIncludedSkills(pluginDir) {
  const initPath = path.join(pluginDir, "init.sh");
  try {
    const text = fs.readFileSync(initPath, "utf-8");
    const m = text.match(/INCLUDED_SKILLS="([^"]*)"/);
    if (!m) return [];
    return m[1].split(/\s+/).filter(Boolean);
  } catch {
    return [];
  }
}

function findSkillSourceDir(skillName, repoRoot) {
  for (const dir of SCAN_DIRS) {
    const candidate = path.join(repoRoot, dir, skillName);
    if (fs.existsSync(candidate) && fs.statSync(candidate).isDirectory()) {
      return dir;
    }
  }
  return null;
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

function mergeInitSkills(ymlEntry, pluginDir, repoRoot) {
  const ymlSkills = ymlEntry.installSkills || [];
  const seen = collectYmlSkillNames(ymlSkills);
  const initSkills = parseIncludedSkills(pluginDir);

  if (initSkills.length === 0) return ymlSkills;

  const merged = ymlSkills.map((s) => ({ ...s, skills: [...s.skills] }));

  for (const skill of initSkills) {
    if (seen.has(skill)) continue;
    const srcDir = findSkillSourceDir(skill, repoRoot);
    if (!srcDir) continue;
    let bucket = merged.find((b) => b.dir === srcDir);
    if (!bucket) {
      bucket = { dir: srcDir, skills: [] };
      merged.push(bucket);
    }
    bucket.skills.push(skill);
    seen.add(skill);
  }
  return merged;
}

function countSkills(installSkills) {
  if (!installSkills) return 0;
  let n = 0;
  for (const src of installSkills) n += src.skills.length;
  return n;
}

function resolveAgentsFromPluginJson(pj) {
  if (!pj || !Array.isArray(pj.agents)) return [];
  return pj.agents
    .map((a) => a.replace(/^\.\/agents\//, "").replace(/\.md$/, ""))
    .filter(Boolean);
}

// ---- Main ----

function run() {
  const repoRoot = path.join(ROOT, "..", "..");
  const plugins = [];
  for (const file of fs.readdirSync(PLUGINS_DIR)) {
    if (file.startsWith("_") || !file.endsWith(".yml")) continue;
    let content;
    try {
      content = yaml.parse(fs.readFileSync(path.join(PLUGINS_DIR, file), "utf-8"));
    } catch {
      console.warn(`  ! failed to parse ${file}`);
      continue;
    }
    if (!content || !content.id) continue;

    const pluginDir = path.join(repoRoot, content.dir);
    const pj = readPluginJson(pluginDir);

    // version: plugin.json overrides yml hardcoded value
    const version = (pj && pj.version) || content.version;

    // description: yml first, fallback to plugin.json
    const description = content.description || (pj && pj.description) || "";

    // installAgents: yml first, fallback to plugin.json agents
    let installAgents = content.installAgents;
    if (!installAgents || installAgents.length === 0) {
      installAgents = resolveAgentsFromPluginJson(pj);
    }

    // installSkills: yml first, init.sh INCLUDED_SKILLS supplements missing
    const installSkills = mergeInitSkills(content, pluginDir, repoRoot);

    const skillsCount = countSkills(installSkills);
    const agentsCount = installAgents ? installAgents.length : 0;

    plugins.push({
      id: content.id,
      dir: content.dir,
      displayName: content.displayName || content.id,
      script: content.script || defaults.script || "init.sh",
      aliases: content.aliases || [],
      skills: skillsCount,
      agents: agentsCount,
      description,
      version,
      configFile: content.configFile,
      configRootConfigLink: content.configRootConfigLink,
      installSkills,
      installAgents,
      externalRepos: content.externalRepos,
    });
  }
  plugins.sort((a, b) => (a.displayName < b.displayName ? -1 : a.displayName > b.displayName ? 1 : 0));

  fs.writeFileSync(OUT_PATH, JSON.stringify(plugins, null, 2));
  console.log(`Generated embedded-plugins.json with ${plugins.length} plugins`);
}

run();
