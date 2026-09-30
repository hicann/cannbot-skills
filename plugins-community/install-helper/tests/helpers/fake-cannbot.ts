// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You should not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { mkdirSync, writeFileSync } from "fs";
import { join } from "path";

const COMMON_HEADER = `#!/usr/bin/env node
import { mkdirSync, writeFileSync, cpSync, existsSync, readFileSync, rmSync } from "node:fs";
import { join, dirname, basename } from "node:path";
import { fileURLToPath } from "node:url";

const pkgRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const TOOL_LAYOUTS = {
  opencode: { configDirs: [".opencode"], defaultConfigDir: ".opencode", skillsDir: [".agents", "skills"] },
  codex: { configDirs: [".codex"], defaultConfigDir: ".codex", skillsDir: [".agents", "skills"] },
  claude: { configDirs: [".claude"], defaultConfigDir: ".claude" },
  trae: { configDirs: [".traecli", ".marscode", ".trae", ".trae-cn"], defaultConfigDir: ".trae" },
};

`;

// Pre-1.8.0 installer: supports the --bundled flag and the
// FAKE_CANNBOT_SIMULATE_RENAME_GAP simulation of installLatest failures.
const LEGACY_PROLOGUE = `const [command, pluginId, ...rest] = process.argv.slice(2);
if (command !== "install" || !pluginId) {
  console.error("usage: cannbot install <plugin> --tool <tool> [--target <dir>] [--bundled]");
  process.exit(1);
}
const options = { target: process.cwd(), bundled: false };
for (let i = 0; i < rest.length; i++) {
  if (rest[i] === "--bundled") { options.bundled = true; continue; }
  if (rest[i] === "--tool" && rest[i + 1]) { options.tool = rest[i + 1]; i++; continue; }
  if (rest[i] === "--target" && rest[i + 1]) { options.target = rest[i + 1]; i++; continue; }
  console.error("unknown option: " + rest[i]); process.exit(1);
}
// Simulates installers packaged before the cannbot plugins-official rename
// (e140ca6): the bundled sourcePluginDirectory still probes the legacy
// "plugins" dir against a fresh mainline checkout, so installLatest fails
// for official plugins unless --bundled forces the package snapshot.
if (process.env.FAKE_CANNBOT_SIMULATE_RENAME_GAP === "1" && !options.bundled) {
  console.error("Fetching latest CANNBot from https://gitcode.com/cann/cannbot.git...");
  console.error("cannbot: latest installation failed: plugin not found: " + pluginId);
  process.exit(1);
}
if (process.env.FAKE_CANNBOT_BUNDLED_ALSO_FAILS === "1" && options.bundled) {
  console.error("cannbot: bundled installation failed: plugin not found: " + pluginId);
  process.exit(1);
}
`;

const MODERN_PROLOGUE = `const [command, pluginId, ...rest] = process.argv.slice(2);
if (command !== "install" || !pluginId) {
  console.error("usage: cannbot install <plugin> --tool <tool> [--target <dir>] [--source <repository>] [--plugin-dir <dir>] [--override-skills <dir>]");
  process.exit(1);
}
// Modern installer (>= 1.8.0): installs from the bundled dist snapshot by
// default; no mainline fetch, no bundled flag, "plugin not found" on the
// first attempt is a genuine failure.
const options = { target: process.cwd() };
for (let i = 0; i < rest.length; i++) {
  if (rest[i] === "--tool" && rest[i + 1]) { options.tool = rest[i + 1]; i++; continue; }
  if (rest[i] === "--target" && rest[i + 1]) { options.target = rest[i + 1]; i++; continue; }
  if (rest[i] === "--source" && rest[i + 1]) { i++; continue; }
  if (rest[i] === "--plugin-dir" && rest[i + 1]) { i++; continue; }
  if (rest[i] === "--mode" && rest[i + 1]) { i++; continue; }
  if (rest[i] === "--override-skills" && rest[i + 1]) { i++; continue; }
  console.error("unknown option: " + rest[i]); process.exit(1);
}
// Simulates a plugin absent from the modern installer's bundled snapshot
// while the discovery manifest still lists it (stale discovery metadata).
if (process.env.FAKE_CANNBOT_MODERN_NOT_FOUND === "1") {
  console.error("cannbot: plugin not found: " + pluginId);
  process.exit(1);
}
`;

const INSTALL_BODY = `if (!options.tool || !TOOL_LAYOUTS[options.tool]) {
  console.error("--tool must be one of: " + Object.keys(TOOL_LAYOUTS).join(", "));
  process.exit(1);
}

const layout = TOOL_LAYOUTS[options.tool];
const configDir = layout.configDirs.map((n) => join(options.target, n)).find((p) => existsSync(p)) ?? join(options.target, layout.defaultConfigDir);
const skillsDir = layout.skillsDir ? join(options.target, ...layout.skillsDir) : join(configDir, "skills");
const agentsDir = join(configDir, "agents");

const pluginDir = join(pkgRoot, "dist", "plugins", pluginId);
const manifestPath = join(pluginDir, ".claude-plugin", "plugin.json");
if (!existsSync(manifestPath)) {
  console.error("plugin not found: " + pluginId);
  process.exit(1);
}
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));

mkdirSync(skillsDir, { recursive: true });
const installedSkills = [];
const srcSkillsRoot = join(pluginDir, "skills");
for (const name of manifest.skills ?? []) {
  const skillName = basename(name);
  const dest = join(skillsDir, skillName);
  rmSync(dest, { recursive: true, force: true });
  cpSync(join(srcSkillsRoot, skillName), dest, { recursive: true });
  installedSkills.push(dest);
}

mkdirSync(agentsDir, { recursive: true });
const installedAgents = [];
for (const rel of manifest.agents ?? []) {
  const source = join(pluginDir, rel);
  if (options.tool === "codex") {
    const md = readFileSync(source, "utf8");
    const name = md.match(/^name:\\s*(.+)$/m)?.[1]?.trim();
    const description = md.match(/^description:\\s*(.+)$/m)?.[1]?.trim();
    const body = md.split(/^---\\r?\\n[\\s\\S]*?\\r?\\n---\\r?\\n/)[1]?.trim() ?? "";
    const dest = join(agentsDir, name + ".toml");
    writeFileSync(dest, "name = " + JSON.stringify(name) + "\\ndescription = " + JSON.stringify(description) + "\\ndeveloper_instructions = " + JSON.stringify(body) + "\\n");
    installedAgents.push(dest);
  } else {
    const dest = join(agentsDir, basename(source));
    writeFileSync(dest, readFileSync(source, "utf8"));
    installedAgents.push(dest);
  }
}

let instructions = null;
const agentsMd = join(pluginDir, "AGENTS.md");
if (existsSync(agentsMd)) {
  const destination = join(options.target, options.tool === "claude" ? "CLAUDE.md" : "AGENTS.md");
  const current = existsSync(destination) ? readFileSync(destination, "utf8") : "";
  const start = "<!-- cannbot:" + pluginId + ":start -->";
  const end = "<!-- cannbot:" + pluginId + ":end -->";
  const withoutOld = current.replace(new RegExp(start + "[\\\\s\\\\S]*?" + end + "\\\\s*", "g"), "").trimEnd();
  const parts = [withoutOld, start, readFileSync(agentsMd, "utf8").trim(), end, ""].filter(Boolean);
  writeFileSync(destination, parts.join("\\n\\n"));
  instructions = destination;
}

let assets = null;
const workflowsSrc = join(pluginDir, "workflows");
const assetsRoot = join(options.target, ".cannbot", "plugins", pluginId);
if (existsSync(workflowsSrc)) {
  mkdirSync(assetsRoot, { recursive: true });
  cpSync(workflowsSrc, join(assetsRoot, "workflows"), { recursive: true });
  assets = ".cannbot/plugins/" + pluginId;
}

let permissions = null;
let settings = null;
if (manifest.fakeRuntime) {
  permissions = ".cannbot/permissions";
  mkdirSync(join(options.target, permissions), { recursive: true });
  writeFileSync(join(options.target, permissions, "hook.js"), "// fake\\n");
  settings = ".cannbot/settings.json";
  writeFileSync(join(options.target, settings), JSON.stringify({ plugins: [pluginId] }, null, 2));
}

let hookSettings = null;
if (options.tool === "claude" && assets) {
  hookSettings = ".claude/settings.json";
  const sp = join(options.target, hookSettings);
  let obj = {};
  if (existsSync(sp)) { try { obj = JSON.parse(readFileSync(sp, "utf8")); } catch {} }
  obj.hooks = obj.hooks || {};
  obj.hooks.PreToolUse = obj.hooks.PreToolUse || [];
  obj.hooks.PreToolUse.push({ hooks: [{ command: "node " + assetsRoot + "/hooks/permission-guard.js" }] });
  writeFileSync(sp, JSON.stringify(obj, null, 2));
}

const registryPath = join(configDir, "cannbot-plugin.json");
let registry = { schemaVersion: 2, plugins: [] };
if (existsSync(registryPath)) {
  try { registry = JSON.parse(readFileSync(registryPath, "utf8")); } catch {}
}
const rel = (p) => p.slice(options.target.length + 1);
const record = {
  plugin: manifest.name,
  pluginVersion: manifest.version,
  sourcePackage: "@cannbot-plugin/cannbot@9.9.9",
  source: { kind: "package", package: "@cannbot-plugin/cannbot@9.9.9" },
  tool: options.tool,
  skillInstallMode: "copy",
  skills: installedSkills.map(rel),
  agents: installedAgents.map(rel),
  instructions: instructions ? rel(instructions) : null,
  assets,
  permissions,
  settings,
  hookSettings,
  dependencies: [],
  nativePlugin: null,
  unsupported: [],
};
const idx = registry.plugins.findIndex((p) => p.plugin === record.plugin);
if (idx === -1) registry.plugins.push(record); else registry.plugins[idx] = record;
mkdirSync(configDir, { recursive: true });
writeFileSync(registryPath, JSON.stringify(registry, null, 2));
console.log("Installed " + manifest.name + "@" + manifest.version + " for " + options.tool);
`;

const FAKE_CANNBOT_BIN = COMMON_HEADER + LEGACY_PROLOGUE + INSTALL_BODY;
// Mirrors @cannbot-plugin/cannbot >= 1.8.0: bundled-by-default install
// without the legacy bundled flag (that flag string must not appear
// anywhere in this source so the capability probe reads it as unsupported).
const MODERN_CANNBOT_BIN = COMMON_HEADER + MODERN_PROLOGUE + INSTALL_BODY;

function setup() {
  testDir = join(tmpdir(), `ih-cbi-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
  savedEnv = {};
  for (const key of ["CANNBOT_INSTALLER_PATH", "CANNBOT_INSTALLER_CHANNEL"]) {
    savedEnv[key] = process.env[key];
    delete process.env[key];
  }
}

function teardown() {
  rmSync(testDir, { recursive: true, force: true });
  for (const [key, value] of Object.entries(savedEnv)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
}

/**
 * Creates a fake @cannbot-plugin/cannbot installer package layout:
 *   <root>/package.json, bin/cannbot.js, dist/plugins/<id>/{.claude-plugin,skills,agents,AGENTS.md}
 * The bin script faithfully mirrors the real installer's layout contract
 * (skills dir, agents dir, marker-block instructions, .cannbot assets,
 * cannbot-plugin.json registry) so delegation tests exercise real paths.
 * Pass modern: true for the >= 1.8.0 behavior (bundled-by-default install,
 * no bundled flag — a first-attempt "plugin not found" is final).
 */
export function createFakeInstaller(
  root: string,
  pluginId: string,
  options?: { skills?: string[]; agents?: string[]; modern?: boolean }
): string {
  mkdirSync(join(root, "bin"), { recursive: true });
  writeFileSync(
    join(root, "package.json"),
    JSON.stringify({ name: "@cannbot-plugin/cannbot", version: "9.9.9", type: "module" }, null, 2)
  );
  writeFileSync(
    join(root, "bin", "cannbot.js"),
    options?.modern ? MODERN_CANNBOT_BIN : FAKE_CANNBOT_BIN
  );

  const pluginDir = join(root, "dist", "plugins", pluginId);
  mkdirSync(join(pluginDir, ".claude-plugin"), { recursive: true });
  const skills = options?.skills ?? ["fake-skill-a", "fake-skill-b"];
  const agents = options?.agents ?? ["fake-agent"];
  writeFileSync(
    join(pluginDir, ".claude-plugin", "plugin.json"),
    JSON.stringify(
      {
        name: pluginId,
        version: "3.1.0",
        description: `fake cannbot plugin ${pluginId}`,
        agents: agents.map((a) => `./agents/${a}.md`),
        skills: skills.map((s) => `./skills/${s}`),
      },
      null,
      2
    )
  );
  for (const skill of skills) {
    mkdirSync(join(pluginDir, "skills", skill), { recursive: true });
    writeFileSync(join(pluginDir, "skills", skill, "SKILL.md"), `---\nname: ${skill}\ndescription: fake\n---\n# ${skill}\n`);
  }
  mkdirSync(join(pluginDir, "agents"), { recursive: true });
  for (const agent of agents) {
    writeFileSync(
      join(pluginDir, "agents", `${agent}.md`),
      `---\nname: ${agent}\ndescription: fake agent\n---\nbody\n`
    );
  }
  writeFileSync(join(pluginDir, "AGENTS.md"), `# ${pluginId} instructions\nuse me\n`);
  return root;
}
