// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// Migrated-plugin tombstone coverage (issue #842): plugins removed from this
// repo and migrated to the cannbot repo (e.g. triton-op-generator) must stay
// resolvable by discovery (findPlugin/getAllPlugins/getPluginById resolve
// them as cannbot-source), reach the delegated installer, and have their
// legacy install records rerouted to the cannbot source by update. Modern
// installers require the plugin in their published package snapshot.

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdirSync, writeFileSync, existsSync, rmSync, readFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import type { PluginEntry } from "../src/types/index.js";
import { createFakeInstaller } from "./helpers/fake-cannbot.js";

let testDir: string;
let sandboxHome: string;
let savedEnv: Record<string, string | undefined>;
let savedCwd: string;

function setup() {
  testDir = join(tmpdir(), `ih-mig-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
  sandboxHome = join(testDir, "home");
  mkdirSync(join(sandboxHome, ".cannbot"), { recursive: true });
  savedEnv = {
    HOME: process.env.HOME,
    CANNBOT_INSTALLER_PATH: process.env.CANNBOT_INSTALLER_PATH,
    CANNBOT_REPO_PATH: process.env.CANNBOT_REPO_PATH,
  };
  savedCwd = process.cwd();
  process.env.HOME = sandboxHome;
}

async function teardown() {
  // reset the module-level cannbot registry state for the next test
  const { setCannbotPlugins } = await import("../src/core/registry.js");
  setCannbotPlugins([]);
  for (const [key, value] of Object.entries(savedEnv)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  process.chdir(savedCwd);
  rmSync(testDir, { recursive: true, force: true });
}

/** A cannbot-snapshot entry as setCannbotPlugins would receive it. */
function cannbotEntry(id: string, overrides?: Partial<PluginEntry>): PluginEntry {
  return {
    id,
    dir: join(testDir, "snapshot", id),
    displayName: id,
    script: "cannbot",
    aliases: [],
    skills: 3,
    agents: 0,
    description: `snapshot entry ${id}`,
    version: "9.9.9",
    source: "cannbot",
    ...overrides,
  };
}

// A fake @cannbot-plugin/cannbot installer that mirrors the real bin
// bootstrap layer: it follows the cannbot git mainline at install time and
// does NOT consult its own dist/plugins snapshot. triton-op-generator is on
// the mainline although the 1.9.0 npm snapshot (published before the
// migration) does not list it; ghost-plugin does not exist upstream at all.
const MAINLINE_BIN = `#!/usr/bin/env node
import { mkdirSync, writeFileSync, existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

const [command, pluginId, ...rest] = process.argv.slice(2);
if (command !== "install" || !pluginId) {
  console.error("usage: cannbot install <plugin> --tool <tool> --target <dir>");
  process.exit(1);
}
const options = { target: process.cwd() };
for (let i = 0; i < rest.length; i++) {
  if (rest[i] === "--tool" && rest[i + 1]) { options.tool = rest[i + 1]; i++; continue; }
  if (rest[i] === "--target" && rest[i + 1]) { options.target = rest[i + 1]; i++; continue; }
  console.error("unknown option: " + rest[i]);
  process.exit(1);
}
if (!options.tool) {
  console.error("--tool is required");
  process.exit(1);
}

const MAINLINE = {
  "triton-op-generator": {
    version: "2.0.0",
    skills: ["triton-op-coding"],
    agents: [],
    instructions: "# triton-op-generator (cannbot mainline)\\n",
  },
};

const plugin = MAINLINE[pluginId];
if (!plugin) {
  console.error("plugin not found: " + pluginId);
  process.exit(1);
}

const target = options.target;
const installedSkills = [];
for (const skill of plugin.skills) {
  const dest = join(target, ".agents", "skills", skill);
  mkdirSync(dest, { recursive: true });
  writeFileSync(join(dest, "SKILL.md"), "---\\nname: " + skill + "\\ndescription: mainline skill\\n---\\n# " + skill + "\\n");
  installedSkills.push(".agents/skills/" + skill);
}

const configDir = join(target, ".opencode");
mkdirSync(configDir, { recursive: true });
const registryPath = join(configDir, "cannbot-plugin.json");
let registry = { schemaVersion: 2, plugins: [] };
if (existsSync(registryPath)) {
  try { registry = JSON.parse(readFileSync(registryPath, "utf8")); } catch {}
}
const record = {
  plugin: pluginId,
  pluginVersion: plugin.version,
  sourcePackage: "@cannbot-plugin/cannbot@1.9.0",
  source: { kind: "package", package: "@cannbot-plugin/cannbot@1.9.0" },
  tool: options.tool,
  skillInstallMode: "copy",
  skills: installedSkills,
  agents: [],
  instructions: "AGENTS.md",
  assets: null,
  permissions: null,
  settings: null,
  hookSettings: null,
  dependencies: [],
  nativePlugin: null,
  unsupported: [],
};
const idx = registry.plugins.findIndex((p) => p.plugin === pluginId);
if (idx === -1) registry.plugins.push(record); else registry.plugins[idx] = record;
writeFileSync(registryPath, JSON.stringify(registry, null, 2));

const start = "<!-- cannbot:" + pluginId + ":start -->";
const end = "<!-- cannbot:" + pluginId + ":end -->";
const agentsMd = join(target, "AGENTS.md");
const current = existsSync(agentsMd) ? readFileSync(agentsMd, "utf8") : "";
writeFileSync(agentsMd, [current.trimEnd(), start, plugin.instructions.trim(), end, ""].filter(Boolean).join("\\n\\n"));

console.log("Installed " + pluginId + "@" + plugin.version + " for " + options.tool);
`;

/**
 * Creates a fake installer whose discovery snapshot (dist/plugins) holds
 * only ops-x while its bin installs from the git mainline — the exact
 * situation of @cannbot-plugin/cannbot@1.9.0 for triton-op-generator.
 */
function useMainlineInstaller(): string {
  const installerRoot = join(testDir, "installer");
  mkdirSync(join(installerRoot, "bin"), { recursive: true });
  writeFileSync(
    join(installerRoot, "package.json"),
    JSON.stringify({ name: "@cannbot-plugin/cannbot", version: "1.9.0", type: "module" }, null, 2)
  );
  writeFileSync(join(installerRoot, "bin", "cannbot.js"), MAINLINE_BIN);

  const opsX = join(installerRoot, "dist", "plugins", "ops-x", ".claude-plugin");
  mkdirSync(opsX, { recursive: true });
  writeFileSync(
    join(opsX, "plugin.json"),
    JSON.stringify({ name: "ops-x", version: "1.0.0", description: "another plugin", skills: [], agents: [] }, null, 2)
  );
  process.env.CANNBOT_INSTALLER_PATH = installerRoot;
  return installerRoot;
}

/**
 * Creates a fixture skills repo holding the pre-migration local copy of
 * triton-op-generator (plugins.d style: installSkills + configFile, no
 * init.sh — so mergeDynamicPlugins does not re-register it as a legacy
 * plugin) plus the valid-repo markers for RepositoryManager.ensureRepo.
 */
function makeLegacyFixture(): { repoDir: string; projectDir: string } {
  const repoDir = join(testDir, "skills-repo");
  const pluginDir = join(repoDir, "plugins-official", "triton-op-generator");
  mkdirSync(join(pluginDir, "skills", "triton-op-coding"), { recursive: true });
  writeFileSync(
    join(pluginDir, "skills", "triton-op-coding", "SKILL.md"),
    "---\nname: triton-op-coding\ndescription: frozen local copy\n---\n# triton-op-coding\n"
  );
  writeFileSync(join(pluginDir, "AGENTS.md"), "# triton-op-generator legacy instructions\n");
  mkdirSync(join(repoDir, ".git"), { recursive: true });

  const projectDir = join(testDir, "project");
  mkdirSync(projectDir, { recursive: true });
  return { repoDir, projectDir };
}

/** The pre-migration embedded registry entry for triton-op-generator. */
function legacyPluginEntry(): PluginEntry {
  return {
    id: "triton-op-generator",
    dir: "plugins-official/triton-op-generator",
    displayName: "Triton 算子开发",
    script: "init.sh",
    aliases: ["triton"],
    skills: 1,
    agents: 0,
    description: "frozen pre-migration entry",
    configFile: "AGENTS.md",
    installSkills: [{ dir: "plugins-official/triton-op-generator/skills", skills: ["triton-op-coding"] }],
  };
}

/** Runs fn with process.stdout.write captured; returns the written text. */
async function captureStdout(fn: () => Promise<void>): Promise<string> {
  const chunks: string[] = [];
  const spy = vi.spyOn(process.stdout, "write").mockImplementation((chunk: unknown) => {
    chunks.push(typeof chunk === "string" ? chunk : Buffer.from(chunk as Uint8Array).toString());
    return true;
  });
  try {
    await fn();
  } finally {
    spy.mockRestore();
  }
  return chunks.join("");
}

describe("migrated plugins — discovery (findPlugin)", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("resolves the tombstone by exact id, alias and prefix in a clean snapshot state", async () => {
    const { findPlugin, setCannbotPlugins } = await import("../src/core/registry.js");
    setCannbotPlugins([]);

    const exact = findPlugin("triton-op-generator");
    expect(exact?.id).toBe("triton-op-generator");
    expect(exact?.source).toBe("cannbot");

    const alias = findPlugin("triton");
    expect(alias?.id).toBe("triton-op-generator");
    expect(alias?.source).toBe("cannbot");

    const prefix = findPlugin("triton-o");
    expect(prefix?.id).toBe("triton-op-generator");
    expect(prefix?.source).toBe("cannbot");
  });

  it("resolves direct-invoke and its historical aliases only from cannbot", async () => {
    const { findPlugin, getAllPlugins, setCannbotPlugins } = await import("../src/core/registry.js");
    const { isMigratedPlugin } = await import("../src/core/migrated-plugins.js");
    setCannbotPlugins([]);

    for (const name of ["ops-direct-invoke", "ops-direct", "ascendc-direct", "direct", "kernel"]) {
      const plugin = findPlugin(name);
      expect(plugin?.id).toBe("ops-direct-invoke");
      expect(plugin?.source).toBe("cannbot");
      expect(findPlugin(name, { source: "skills" })).toBeUndefined();
    }
    expect(isMigratedPlugin("ops-direct-invoke")).toBe(true);
    expect(getAllPlugins().filter((p) => p.id === "ops-direct-invoke")).toHaveLength(1);

    setCannbotPlugins([cannbotEntry("ops-direct-invoke")]);
    expect(findPlugin("ops-direct-invoke")?.version).toBe("9.9.9");
    expect(getAllPlugins().filter((p) => p.id === "ops-direct-invoke")).toHaveLength(1);
  });

  it("resolves direct-invoke-flash and its historical aliases only from cannbot", async () => {
    const { findPlugin, getAllPlugins, setCannbotPlugins } = await import("../src/core/registry.js");
    const { isMigratedPlugin } = await import("../src/core/migrated-plugins.js");
    setCannbotPlugins([]);

    for (const name of ["ops-direct-invoke-flash", "flash", "kernel-flash", "ops-direct-invoke-fl"]) {
      const plugin = findPlugin(name);
      expect(plugin?.id).toBe("ops-direct-invoke-flash");
      expect(plugin?.source).toBe("cannbot");
      expect(findPlugin(name, { source: "skills" })).toBeUndefined();
    }
    expect(isMigratedPlugin("ops-direct-invoke-flash")).toBe(true);
    expect(getAllPlugins().filter((p) => p.id === "ops-direct-invoke-flash")).toHaveLength(1);

    setCannbotPlugins([cannbotEntry("ops-direct-invoke-flash")]);
    expect(findPlugin("ops-direct-invoke-flash")?.version).toBe("9.9.9");
    expect(getAllPlugins().filter((p) => p.id === "ops-direct-invoke-flash")).toHaveLength(1);
  });

  it("resolves perf-optimize and its historical aliases only from cannbot", async () => {
    const { findPlugin, getAllPlugins, setCannbotPlugins } = await import("../src/core/registry.js");
    const { isMigratedPlugin } = await import("../src/core/migrated-plugins.js");
    setCannbotPlugins([]);

    for (const name of ["ops-perf-optimize", "perf-optimize", "perf-optimization", "performance", "ops-perf-opt"]) {
      const plugin = findPlugin(name);
      expect(plugin?.id).toBe("ops-perf-optimize");
      expect(plugin?.source).toBe("cannbot");
      expect(findPlugin(name, { source: "skills" })).toBeUndefined();
    }
    expect(isMigratedPlugin("ops-perf-optimize")).toBe(true);
    expect(getAllPlugins().filter((p) => p.id === "ops-perf-optimize")).toHaveLength(1);

    setCannbotPlugins([cannbotEntry("ops-perf-optimize")]);
    expect(findPlugin("ops-perf-optimize")?.version).toBe("9.9.9");
    expect(getAllPlugins().filter((p) => p.id === "ops-perf-optimize")).toHaveLength(1);
  });

  it("does not leak tombstones into the --source skills scope", async () => {
    const { findPlugin, setCannbotPlugins } = await import("../src/core/registry.js");
    setCannbotPlugins([]);

    expect(findPlugin("triton-op-generator", { source: "skills" })).toBeUndefined();
    expect(findPlugin("triton", { source: "skills" })).toBeUndefined();
    expect(findPlugin("triton-o", { source: "skills" })).toBeUndefined();
  });

  it("still returns undefined for unknown plugins", async () => {
    const { findPlugin, setCannbotPlugins } = await import("../src/core/registry.js");
    setCannbotPlugins([]);

    expect(findPlugin("totally-unknown-plugin")).toBeUndefined();
  });

  it("getPluginById resolves the tombstone", async () => {
    const { getPluginById, setCannbotPlugins } = await import("../src/core/registry.js");
    setCannbotPlugins([]);

    const plugin = getPluginById("triton-op-generator");
    expect(plugin?.id).toBe("triton-op-generator");
    expect(plugin?.source).toBe("cannbot");
  });
});

describe("migrated plugins — snapshot/tombstone precedence", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("tombstone alias wins over a snapshot prefix match (triton vs triton-auto-evolve)", async () => {
    const { findPlugin, setCannbotPlugins } = await import("../src/core/registry.js");
    setCannbotPlugins([cannbotEntry("triton-auto-evolve")]);

    // historical alias must resolve to the migrated tombstone, not be
    // captured by the prefix match on triton-auto-evolve
    const aliasHit = findPlugin("triton");
    expect(aliasHit?.id).toBe("triton-op-generator");
    expect(aliasHit?.source).toBe("cannbot");

    // snapshot exact match still wins over tombstone alias/prefix
    const exactHit = findPlugin("triton-auto-evolve");
    expect(exactHit?.id).toBe("triton-auto-evolve");
    expect(exactHit?.source).toBe("cannbot");
  });

  it("getAllPlugins keeps uncovered tombstones out of the install list (discovery authoritative)", async () => {
    const { getAllPlugins, setCannbotPlugins } = await import("../src/core/registry.js");

    // snapshot does not know the plugin yet — with live entries present the
    // uncovered tombstone is a dead install target (upstream genuinely does
    // not ship it yet) and stays out of the install list; the CLI by-name
    // path still bridges via the tombstone + fail-open delegation
    setCannbotPlugins([cannbotEntry("triton-auto-evolve")]);
    expect(getAllPlugins().some((p) => p.id === "triton-op-generator")).toBe(false);

    // snapshot caught up with the migration — the snapshot entry is the row
    setCannbotPlugins([cannbotEntry("triton-auto-evolve"), cannbotEntry("triton-op-generator")]);
    const entries = getAllPlugins().filter((p) => p.id === "triton-op-generator");
    expect(entries).toHaveLength(1);
    expect(entries[0].version).toBe("9.9.9");
  });

  it("findPlugin prefers the snapshot entry over the tombstone for the same id", async () => {
    const { findPlugin, setCannbotPlugins } = await import("../src/core/registry.js");
    setCannbotPlugins([cannbotEntry("triton-op-generator")]);

    const hit = findPlugin("triton-op-generator");
    expect(hit?.id).toBe("triton-op-generator");
    expect(hit?.version).toBe("9.9.9");
  });
});

describe("migrated plugins — tombstones with a frozen local copy (ops-registry-invoke / CAKE)", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("ops-registry-invoke: tombstone exact and alias beat the embedded frozen entry", async () => {
    const { findPlugin, setCannbotPlugins } = await import("../src/core/registry.js");
    const { isMigratedPlugin } = await import("../src/core/migrated-plugins.js");
    setCannbotPlugins([]);

    // the embedded registry still carries the frozen pre-migration copy
    // (script init.sh, skills 30); unscoped discovery must resolve the
    // tombstone — migration policy routes installs to cannbot
    const hit = findPlugin("ops-registry-invoke");
    expect(hit?.id).toBe("ops-registry-invoke");
    expect(hit?.source).toBe("cannbot");
    expect(hit?.script).toBe("cannbot");
    expect(hit?.skills).toBe(7);

    // routing follows discovery: the snapshot precheck bypass and update
    // rerouting both key off isMigratedPlugin
    expect(isMigratedPlugin("ops-registry-invoke")).toBe(true);

    // historical aliases resolve to the tombstone as well
    for (const alias of ["ops-registry", "ascendc-registry", "registry"]) {
      const aliasHit = findPlugin(alias);
      expect(aliasHit?.id).toBe("ops-registry-invoke");
      expect(aliasHit?.source).toBe("cannbot");
    }

    // the frozen copy remains reachable explicitly via --source skills
    const legacy = findPlugin("ops-registry-invoke", { source: "skills" });
    expect(legacy?.id).toBe("ops-registry-invoke");
    expect(legacy?.script).toBe("init.sh");
    expect(legacy?.skills).toBe(30);
  });

  it("collaborative-agent-kernel-evolution: tombstone resolves; snapshot cake outranks the tombstone alias", async () => {
    const { findPlugin, setCannbotPlugins } = await import("../src/core/registry.js");

    // clean snapshot: the exact id and the "cake" alias both hit the tombstone
    setCannbotPlugins([]);
    const exact = findPlugin("collaborative-agent-kernel-evolution");
    expect(exact?.id).toBe("collaborative-agent-kernel-evolution");
    expect(exact?.source).toBe("cannbot");
    expect(exact?.script).toBe("cannbot");
    expect(findPlugin("cake")?.id).toBe("collaborative-agent-kernel-evolution");

    // the cannbot repo manifests this plugin under the name "cake": once
    // the npm snapshot lists it, the snapshot exact match wins over the
    // tombstone alias
    setCannbotPlugins([cannbotEntry("cake")]);
    const snapshotHit = findPlugin("cake");
    expect(snapshotHit?.id).toBe("cake");
    expect(snapshotHit?.version).toBe("9.9.9");
    expect(snapshotHit?.source).toBe("cannbot");

    // once the renamed identity is discovered, the directory-name long form
    // routes to the LIVE entry — the delegation must use the name the
    // cannbot installer actually knows; the tombstone alone (offline, no
    // snapshot) still owns the long form
    const longForm = findPlugin("collaborative-agent-kernel-evolution");
    expect(longForm?.id).toBe("cake");
    expect(longForm?.script).toBe("cannbot");
  });

  it("getAllPlugins keeps the ops-registry-invoke tombstone out once discovery is authoritative", async () => {
    const { getAllPlugins, setCannbotPlugins } = await import("../src/core/registry.js");

    // snapshot predates the migration but live entries exist: the uncovered
    // tombstone (dead delegation target — upstream does not ship the plugin
    // yet) is out of the install list; the embedded frozen copy (skills
    // source, locally installable) still coexists
    setCannbotPlugins([cannbotEntry("unrelated-plugin")]);
    const entries = getAllPlugins().filter((p) => p.id === "ops-registry-invoke");
    expect(entries.filter((p) => p.script === "cannbot" && p.skills === 7)).toHaveLength(0);
    expect(entries.some((p) => p.script === "init.sh")).toBe(true);

    // snapshot caught up: the snapshot entry (version 9.9.9) is authoritative
    // for the cannbot source
    setCannbotPlugins([cannbotEntry("unrelated-plugin"), cannbotEntry("ops-registry-invoke")]);
    const caughtUp = getAllPlugins().filter((p) => p.id === "ops-registry-invoke");
    expect(caughtUp.some((p) => p.version === "9.9.9")).toBe(true);
  });
});

describe("migrated plugins — delegated install (snapshot precheck bypass)", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("installs a migrated plugin absent from the installer's dist snapshot (mainline follows)", async () => {
    useMainlineInstaller();
    const target = join(testDir, "project");
    mkdirSync(target, { recursive: true });

    const { delegateInstall } = await import("../src/core/cannbot-delegate.js");
    const result = await delegateInstall({
      pluginId: "triton-op-generator",
      tool: "opencode",
      level: "project",
      target,
    });

    expect(result.errors).toEqual([]);
    expect(result.success).toBe(true);
    expect(result.skillsCount).toBe(1);
    // cannbot layout landed: skills under .agents/skills + registry record
    expect(existsSync(join(target, ".agents", "skills", "triton-op-coding", "SKILL.md"))).toBe(true);
    expect(existsSync(join(target, ".opencode", "cannbot-plugin.json"))).toBe(true);

    // install-helper bookkeeping holds the delegated record
    const recordFile = JSON.parse(
      readFileSync(join(sandboxHome, ".cannbot", "installs", "triton-op-generator.json"), "utf-8")
    );
    const record = Object.values(recordFile.locations)[0] as Record<string, unknown>;
    expect(record.kind).toBe("delegated");
    expect(record.tool).toBe("opencode");
  });

  it("delegates direct-invoke to its package without accessing a local plugin copy", async () => {
    const installerRoot = join(testDir, "direct-installer");
    createFakeInstaller(installerRoot, "ops-direct-invoke", {
      modern: true,
      skills: ["ops-direct-invoke"],
      agents: ["ops-direct-invoke-developer"],
    });
    process.env.CANNBOT_INSTALLER_PATH = installerRoot;
    const target = join(testDir, "direct-project");
    mkdirSync(target, { recursive: true });
    const legacyRepo = join(testDir, "absent-skills-repo");
    const { findPlugin } = await import("../src/core/registry.js");
    const { installPluginRouted } = await import("../src/core/cannbot-delegate.js");
    const plugin = findPlugin("direct")!;
    const result = await installPluginRouted({
      pluginId: plugin.id, plugin,
      tool: "opencode", level: "project",
      repoPath: legacyRepo, installPath: target,
    });

    expect(result.errors).toEqual([]);
    expect(result.success).toBe(true);
    expect(existsSync(legacyRepo)).toBe(false);
    expect(existsSync(join(target, ".agents", "skills", "ops-direct-invoke", "SKILL.md"))).toBe(true);
    expect(existsSync(join(target, ".opencode", "agents", "ops-direct-invoke-developer.md"))).toBe(true);
    expect(existsSync(join(target, ".opencode", "cannbot-plugin.json"))).toBe(true);
  });

  it.each([
    ["opencode", "global"], ["qoder", "project"], ["cursor", "project"],
    ["copilot", "project"], ["codearts", "project"],
  ] as const)("rejects direct-invoke %s/%s instead of falling back to removed assets", async (tool, level) => {
    const { findPlugin } = await import("../src/core/registry.js");
    const { installPluginRouted, boundaryErrors } = await import("../src/core/cannbot-delegate.js");
    const legacyRepo = join(testDir, "absent-skills-repo");
    const target = join(testDir, "unmodified-project");
    for (const source of [undefined, "cannbot"] as const) {
      const plugin = findPlugin("kernel")!;
      const result = await installPluginRouted({
        pluginId: plugin.id, plugin, tool, level, source,
        repoPath: legacyRepo, installPath: target,
      });
      expect(result.success).toBe(false);
      expect(result.errors).toEqual(expect.arrayContaining(boundaryErrors(tool, level)));
      expect(result.warnings).toEqual([]);
      expect(existsSync(legacyRepo)).toBe(false);
      expect(existsSync(target)).toBe(false);
    }
  });

  it("still gates non-migrated plugins on the dist snapshot (regression guard)", async () => {
    useMainlineInstaller();
    const target = join(testDir, "ghost-project");

    const { delegateInstall } = await import("../src/core/cannbot-delegate.js");
    const result = await delegateInstall({
      pluginId: "ghost-plugin",
      tool: "opencode",
      level: "project",
      target,
    });

    // bootstrap semantics (fail-open precheck): the delegate delegates
    // regardless of the local snapshot, and the authoritative mainline
    // installer reports the precise not-found reason from its stdout.
    expect(result.success).toBe(false);
    expect(result.errors.join(" ")).toContain("ghost-plugin");
    expect(result.errors.join(" ")).toMatch(/not found|未找到/i);
    expect(existsSync(join(target, ".opencode", "cannbot-plugin.json"))).toBe(false);
  });
});

describe("migrated plugins — update rerouting (end-to-end)", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("update <name> reroutes a legacy install record of a migrated plugin to the cannbot source", async () => {
    const { repoDir, projectDir } = makeLegacyFixture();
    useMainlineInstaller();

    // seed a real legacy install (pre-migration plugins.d style) — manifest,
    // skills and install record all in place
    const { installPlugin } = await import("../src/core/installer.js");
    const install = await installPlugin({
      pluginId: "triton-op-generator",
      tool: "opencode",
      level: "project",
      repoPath: repoDir,
      installPath: projectDir,
      yes: true,
      plugin: legacyPluginEntry(),
    });
    expect(install.success).toBe(true);
    expect(existsSync(join(projectDir, ".opencode", "triton-op-generator-manifest.json"))).toBe(true);
    expect(existsSync(join(projectDir, ".opencode", "skills", "triton-op-coding"))).toBe(true);

    process.env.CANNBOT_REPO_PATH = repoDir;
    process.chdir(projectDir);
    const { updateCommand } = await import("../src/commands/update.js");

    const output = await captureStdout(() =>
      updateCommand(["triton-op-generator"], { tool: "opencode", level: "project", yes: true })
    );

    // the update went through delegation: the cannbot installer produced the
    // registry record and the mainline layout
    expect(output).toContain("Installed triton-op-generator");
    expect(existsSync(join(projectDir, ".opencode", "cannbot-plugin.json"))).toBe(true);
    expect(existsSync(join(projectDir, ".agents", "skills", "triton-op-coding", "SKILL.md"))).toBe(true);
  });

  it("update (no args) reroutes legacy install records of migrated plugins to the cannbot source", async () => {
    const { repoDir, projectDir } = makeLegacyFixture();
    useMainlineInstaller();

    const { installPlugin } = await import("../src/core/installer.js");
    const install = await installPlugin({
      pluginId: "triton-op-generator",
      tool: "opencode",
      level: "project",
      repoPath: repoDir,
      installPath: projectDir,
      yes: true,
      plugin: legacyPluginEntry(),
    });
    expect(install.success).toBe(true);

    process.env.CANNBOT_REPO_PATH = repoDir;
    process.chdir(projectDir);
    const { updateCommand } = await import("../src/commands/update.js");

    const output = await captureStdout(() =>
      updateCommand([], { tool: "opencode", level: "project", yes: true })
    );

    expect(output).toContain("Installed triton-op-generator");
    expect(existsSync(join(projectDir, ".opencode", "cannbot-plugin.json"))).toBe(true);
    expect(existsSync(join(projectDir, ".agents", "skills", "triton-op-coding", "SKILL.md"))).toBe(true);
  });

  it("MIGRATED_PLUGINS contains no duplicate ids (rebase auto-merge guard)", async () => {
    // A rebase can auto-merge an upstream tombstone addition with our own,
    // yielding two entries for the same id (observed when upstream adopted
    // the catlass tombstone). getAllPlugins/findPlugin would then list the
    // plugin twice. Guard: ids inside MIGRATED_PLUGINS are unique.
    const { MIGRATED_PLUGINS } = await import("../src/core/migrated-plugins.js");
    const ids = MIGRATED_PLUGINS.map((p) => p.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
