// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You should not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdirSync, writeFileSync, existsSync, rmSync, readFileSync, symlinkSync, readdirSync, lstatSync } from "fs";
import { join, dirname } from "path";
import { tmpdir } from "os";
import { fileURLToPath } from "url";
import { createFakeInstaller } from "./helpers/fake-cannbot.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
void __dirname;

let testDir: string;
let sandboxHome: string;
let savedEnv: Record<string, string | undefined>;

function setup() {
  testDir = join(tmpdir(), `ih-cbd-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
  sandboxHome = join(testDir, "home");
  mkdirSync(join(sandboxHome, ".cannbot"), { recursive: true });
  delete process.env.FAKE_CANNBOT_SIMULATE_RENAME_GAP;
  delete process.env.FAKE_CANNBOT_BUNDLED_ALSO_FAILS;
  delete process.env.FAKE_CANNBOT_MODERN_NOT_FOUND;
  savedEnv = {
    HOME: process.env.HOME,
    CANNBOT_INSTALLER_PATH: process.env.CANNBOT_INSTALLER_PATH,
    FAKE_CANNBOT_SIMULATE_RENAME_GAP: process.env.FAKE_CANNBOT_SIMULATE_RENAME_GAP,
    FAKE_CANNBOT_BUNDLED_ALSO_FAILS: process.env.FAKE_CANNBOT_BUNDLED_ALSO_FAILS,
    FAKE_CANNBOT_MODERN_NOT_FOUND: process.env.FAKE_CANNBOT_MODERN_NOT_FOUND,
  };
  process.env.HOME = sandboxHome;
}

function teardown() {
  for (const [key, value] of Object.entries(savedEnv)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  rmSync(testDir, { recursive: true, force: true });
}

function useFakeInstaller(pluginId: string, options?: { skills?: string[]; agents?: string[] }): string {
  const installerRoot = join(testDir, "installer");
  createFakeInstaller(installerRoot, pluginId, options);
  process.env.CANNBOT_INSTALLER_PATH = installerRoot;
  return installerRoot;
}

describe("cannbot-delegate", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("rejects tools the upstream installer does not support", async () => {
    useFakeInstaller("fake-cb-plugin");
    const { delegateInstall } = await import("../src/core/cannbot-delegate.js");

    const result = await delegateInstall({
      pluginId: "fake-cb-plugin",
      tool: "cursor",
      level: "project",
      target: testDir,
    });
    expect(result.success).toBe(false);
    expect(result.errors[0]).toContain("cursor");
  });

  it("rejects global-level installs (upstream is project-only)", async () => {
    useFakeInstaller("fake-cb-plugin");
    const { delegateInstall } = await import("../src/core/cannbot-delegate.js");

    const result = await delegateInstall({
      pluginId: "fake-cb-plugin",
      tool: "opencode",
      level: "global",
      target: testDir,
    });
    expect(result.success).toBe(false);
    expect(result.errors[0]).toContain("project");
  });

  it("reports a missing plugin after forcing one installer refresh", async () => {
    useFakeInstaller("fake-cb-plugin");
    const { delegateInstall } = await import("../src/core/cannbot-delegate.js");

    const result = await delegateInstall({
      pluginId: "not-in-release",
      tool: "opencode",
      level: "project",
      target: testDir,
    });
    expect(result.success).toBe(false);
    expect(result.errors.join(" ")).toContain("not-in-release");
  });

  it("delegates an opencode install producing the cannbot layout and a delegated record", async () => {
    useFakeInstaller("fake-cb-plugin", { skills: ["fake-skill-a", "fake-skill-b"], agents: ["fake-agent"] });
    const target = join(testDir, "project");
    mkdirSync(target, { recursive: true });

    const { delegateInstall } = await import("../src/core/cannbot-delegate.js");
    const result = await delegateInstall({
      pluginId: "fake-cb-plugin",
      displayName: "Fake CB Plugin",
      tool: "opencode",
      level: "project",
      target,
    });

    expect(result.errors).toEqual([]);
    expect(result.success).toBe(true);
    expect(result.skillsCount).toBe(2);
    expect(result.agentsCount).toBe(1);

    // cannbot layout: skills in .agents/skills (opencode/codex shared root)
    expect(existsSync(join(target, ".agents/skills/fake-skill-a/SKILL.md"))).toBe(true);
    expect(existsSync(join(target, ".agents/skills/fake-skill-b/SKILL.md"))).toBe(true);
    // agents under the tool config dir
    expect(existsSync(join(target, ".opencode/agents/fake-agent.md"))).toBe(true);
    // marker-block instructions
    const agentsMd = readFileSync(join(target, "AGENTS.md"), "utf-8");
    expect(agentsMd).toContain("<!-- cannbot:fake-cb-plugin:start -->");
    expect(agentsMd).toContain("<!-- cannbot:fake-cb-plugin:end -->");
    expect(agentsMd).toContain("fake-cb-plugin instructions");
    // registry
    const registry = JSON.parse(readFileSync(join(target, ".opencode/cannbot-plugin.json"), "utf-8"));
    expect(registry.schemaVersion).toBe(2);
    expect(registry.plugins[0].plugin).toBe("fake-cb-plugin");
    expect(registry.plugins[0].skills).toContain(".agents/skills/fake-skill-a");

    // delegated record in install-helper bookkeeping (v2 location-keyed store)
    const recordFile = JSON.parse(
      readFileSync(join(sandboxHome, ".cannbot/installs/fake-cb-plugin.json"), "utf-8")
    );
    const record = Object.values(recordFile.locations)[0] as any;
    expect(record.kind).toBe("delegated");
    expect(record.tool).toBe("opencode");
    expect(record.level).toBe("project");
    expect(record.files.some((f: string) => f.includes("fake-skill-a"))).toBe(true);
  });

  it("delegates a claude install with CLAUDE.md instructions and .claude layout", async () => {
    useFakeInstaller("fake-cb-plugin");
    const target = join(testDir, "claude-project");
    mkdirSync(target, { recursive: true });

    const { delegateInstall } = await import("../src/core/cannbot-delegate.js");
    const result = await delegateInstall({
      pluginId: "fake-cb-plugin",
      tool: "claude",
      level: "project",
      target,
    });
    expect(result.success).toBe(true);

    expect(existsSync(join(target, ".claude/skills/fake-skill-a/SKILL.md"))).toBe(true);
    expect(existsSync(join(target, ".claude/agents/fake-agent.md"))).toBe(true);
    const claudeMd = readFileSync(join(target, "CLAUDE.md"), "utf-8");
    expect(claudeMd).toContain("<!-- cannbot:fake-cb-plugin:start -->");
    expect(existsSync(join(target, ".claude/cannbot-plugin.json"))).toBe(true);
  });

  it("delegates a codex install with toml agents", async () => {
    useFakeInstaller("fake-cb-plugin");
    const target = join(testDir, "codex-project");
    mkdirSync(target, { recursive: true });

    const { delegateInstall } = await import("../src/core/cannbot-delegate.js");
    const result = await delegateInstall({
      pluginId: "fake-cb-plugin",
      tool: "codex",
      level: "project",
      target,
    });
    expect(result.success).toBe(true);

    expect(existsSync(join(target, ".agents/skills/fake-skill-a/SKILL.md"))).toBe(true);
    expect(existsSync(join(target, ".codex/agents/fake-agent.toml"))).toBe(true);
    expect(existsSync(join(target, ".codex/cannbot-plugin.json"))).toBe(true);
  });

  it("migrates a symlinked AGENTS.md to a real file before delegating", async () => {
    useFakeInstaller("fake-cb-plugin");
    const target = join(testDir, "migrate-project");
    mkdirSync(target, { recursive: true });

    // simulate a legacy install owning AGENTS.md as a symlink
    const legacySource = join(testDir, "legacy-plugin-agents.md");
    writeFileSync(legacySource, "# legacy plugin instructions\n");
    symlinkSync(legacySource, join(target, "AGENTS.md"));

    const { delegateInstall } = await import("../src/core/cannbot-delegate.js");
    const result = await delegateInstall({
      pluginId: "fake-cb-plugin",
      tool: "opencode",
      level: "project",
      target,
    });
    expect(result.success).toBe(true);
    expect(result.warnings.join(" ")).toContain("AGENTS.md");

    // AGENTS.md is now a real file keeping legacy content + the marker block
    const stat = lstatSync(join(target, "AGENTS.md"));
    expect(stat.isSymbolicLink()).toBe(false);
    const content = readFileSync(join(target, "AGENTS.md"), "utf-8");
    expect(content).toContain("# legacy plugin instructions");
    expect(content).toContain("<!-- cannbot:fake-cb-plugin:start -->");

    // a migration backup exists
    const entries = readdirSync(target);
    expect(entries.some((e) => e.startsWith("AGENTS.md.cannbot-migration."))).toBe(true);
    // the legacy source file is untouched
    expect(readFileSync(legacySource, "utf-8")).toBe("# legacy plugin instructions\n");
  });

  it("warns when the same skill exists in the legacy opencode layout (scenario B)", async () => {
    useFakeInstaller("fake-cb-plugin");
    const target = join(testDir, "dual-project");
    mkdirSync(join(target, ".opencode/skills/fake-skill-a"), { recursive: true });
    writeFileSync(join(target, ".opencode/skills/fake-skill-a/SKILL.md"), "---\nname: fake-skill-a\n---\n");

    const { delegateInstall } = await import("../src/core/cannbot-delegate.js");
    const result = await delegateInstall({
      pluginId: "fake-cb-plugin",
      tool: "opencode",
      level: "project",
      target,
    });
    expect(result.success).toBe(true);
    expect(result.warnings.join(" ")).toContain("fake-skill-a");
    // legacy copy untouched
    expect(existsSync(join(target, ".opencode/skills/fake-skill-a/SKILL.md"))).toBe(true);
  });

  it("installPluginRouted sends cannbot-source plugins through delegation", async () => {
    useFakeInstaller("fake-cb-plugin");
    const target = join(testDir, "routed-project");
    mkdirSync(target, { recursive: true });

    const { installPluginRouted } = await import("../src/core/cannbot-delegate.js");
    const result = await installPluginRouted({
      pluginId: "fake-cb-plugin",
      tool: "opencode",
      level: "project",
      repoPath: testDir,
      installPath: target,
      plugin: {
        id: "fake-cb-plugin",
        dir: "",
        displayName: "Fake CB Plugin",
        script: "cannbot",
        aliases: [],
        skills: 2,
        agents: 1,
        description: "",
        source: "cannbot",
      },
    });

    expect(result.success).toBe(true);
    expect(existsSync(join(target, ".agents/skills/fake-skill-a/SKILL.md"))).toBe(true);
  });

  it("installPluginRouted keeps legacy plugins on the legacy installer path", async () => {
    // no cannbot source: a fake legacy id routes to installPlugin which
    // reports the standard not-found error
    const { installPluginRouted } = await import("../src/core/cannbot-delegate.js");
    const result = await installPluginRouted({
      pluginId: "totally-unknown-legacy-plugin",
      tool: "opencode",
      level: "project",
      repoPath: testDir,
      source: "skills",
    });
    expect(result.success).toBe(false);
    expect(result.errors.join(" ")).toContain("totally-unknown-legacy-plugin");
  });

  it("installPluginRouted: explicit --source skills wins over a cannbot plugin entry", async () => {
    // regression guard for the review finding: call sites resolving a
    // same-name plugin without the source scope pass the cannbot entry;
    // an explicit source: "skills" must still force the legacy installer
    useFakeInstaller("ops-direct-invoke");
    const { installPluginRouted } = await import("../src/core/cannbot-delegate.js");

    const result = await installPluginRouted({
      pluginId: "ops-direct-invoke",
      tool: "opencode",
      level: "project",
      repoPath: testDir,
      source: "skills",
      plugin: {
        id: "ops-direct-invoke",
        dir: "",
        displayName: "ops-direct-invoke",
        script: "cannbot",
        aliases: [],
        skills: 1,
        agents: 0,
        description: "",
        source: "cannbot",
      },
    });

    // routed to the LEGACY installer: the failure comes from
    // installPlugin's cannbot-entry guard, never from the cannbot
    // delegation layer
    expect(result.success).toBe(false);
    expect(result.errors.join(" ")).not.toContain("cannbot 安装器");
    expect(result.errors.join(" ")).toContain("installPluginRouted");
  });

  it("installPluginRouted: undefined source still delegates cannbot entries", async () => {
    useFakeInstaller("fake-cb-plugin");
    const target = join(testDir, "routed-implicit");
    mkdirSync(target, { recursive: true });

    const { installPluginRouted } = await import("../src/core/cannbot-delegate.js");
    const result = await installPluginRouted({
      pluginId: "fake-cb-plugin",
      tool: "opencode",
      level: "project",
      repoPath: testDir,
      installPath: target,
      plugin: {
        id: "fake-cb-plugin",
        dir: "",
        displayName: "Fake CB Plugin",
        script: "cannbot",
        aliases: [],
        skills: 2,
        agents: 1,
        description: "",
        source: "cannbot",
      },
    });
    expect(result.success).toBe(true);
    expect(existsSync(join(target, ".agents/skills/fake-skill-a/SKILL.md"))).toBe(true);
  });

  it("wizard choice encoding round-trips source and id", async () => {
    const { encodePluginChoice, decodePluginChoice } = await import("../src/ui/wizard.js");
    expect(encodePluginChoice("ops-direct-invoke", "cannbot")).toBe("cannbot::ops-direct-invoke");
    expect(encodePluginChoice("ops-direct-invoke", undefined)).toBe("skills::ops-direct-invoke");

    expect(decodePluginChoice("cannbot::ops-direct-invoke")).toEqual({
      id: "ops-direct-invoke",
      source: "cannbot",
    });
    expect(decodePluginChoice("skills::ops-direct-invoke")).toEqual({
      id: "ops-direct-invoke",
      source: "skills",
    });
    // bare values (backward compat) decode as legacy
    expect(decodePluginChoice("ops-direct-invoke")).toEqual({
      id: "ops-direct-invoke",
      source: "skills",
    });
  });

  it("falls back to --bundled when the installer predates the plugins-official rename", async () => {
    useFakeInstaller("fake-cb-plugin");
    // simulate a pre-rename installer: installLatest fails "plugin not found"
    // against the fresh mainline checkout; --bundled installs from the
    // package's own dist snapshot
    process.env.FAKE_CANNBOT_SIMULATE_RENAME_GAP = "1";

    const target = join(testDir, "gap-project");
    mkdirSync(target, { recursive: true });

    const { delegateInstall } = await import("../src/core/cannbot-delegate.js");
    const result = await delegateInstall({
      pluginId: "fake-cb-plugin",
      tool: "opencode",
      level: "project",
      target,
    });

    expect(result.success).toBe(true);
    expect(result.errors).toEqual([]);
    expect(result.warnings.join(" ")).toContain("plugins-official");
    expect(result.warnings.join(" ")).toContain("--bundled");

    // the bundled-snapshot install produced the standard layout
    expect(existsSync(join(target, ".agents/skills/fake-skill-a/SKILL.md"))).toBe(true);
    expect(existsSync(join(target, ".opencode/cannbot-plugin.json"))).toBe(true);

    // uninstall still works on the fallback-produced record
    const { uninstallCannbotPlugin } = await import("../src/core/cannbot-registry.js");
    const uninstallResult = uninstallCannbotPlugin("fake-cb-plugin", target);
    expect(uninstallResult.success).toBe(true);
    expect(existsSync(join(target, ".agents"))).toBe(false);
  });

  it("reports the delegate failure when the rename-gap fallback also fails", async () => {
    useFakeInstaller("fake-cb-plugin");
    process.env.FAKE_CANNBOT_SIMULATE_RENAME_GAP = "1";
    process.env.FAKE_CANNBOT_BUNDLED_ALSO_FAILS = "1";

    const target = join(testDir, "gap-fail-project");
    mkdirSync(target, { recursive: true });

    const { delegateInstall } = await import("../src/core/cannbot-delegate.js");
    const result = await delegateInstall({
      pluginId: "fake-cb-plugin",
      tool: "opencode",
      level: "project",
      target,
    });

    // both the installLatest attempt and the --bundled retry failed
    expect(result.success).toBe(false);
    expect(result.errors.join(" ")).toContain("cannbot");
    expect(result.warnings.join(" ")).toContain("--bundled");
    expect(existsSync(join(target, ".opencode/cannbot-plugin.json"))).toBe(false);
  });

  it("modern installer (>= 1.8.0): bundled-by-default install succeeds without any fallback", async () => {
    // modern fake: no bundled flag in the bin source, installs from
    // dist/plugins directly — the happy path must work unchanged
    const installerRoot = join(testDir, "installer-modern");
    createFakeInstaller(installerRoot, "fake-cb-plugin", { modern: true });
    process.env.CANNBOT_INSTALLER_PATH = installerRoot;

    const target = join(testDir, "modern-project");
    mkdirSync(target, { recursive: true });

    const { delegateInstall } = await import("../src/core/cannbot-delegate.js");
    const result = await delegateInstall({
      pluginId: "fake-cb-plugin",
      tool: "opencode",
      level: "project",
      target,
    });

    expect(result.success).toBe(true);
    expect(result.errors).toEqual([]);
    expect(result.warnings).toEqual([]);
    expect(existsSync(join(target, ".agents/skills/fake-skill-a/SKILL.md"))).toBe(true);
    expect(existsSync(join(target, ".opencode/cannbot-plugin.json"))).toBe(true);
  });

  it("modern installer: first-attempt plugin-not-found is final (no bundled retry)", async () => {
    // plugin exists in dist (so discovery passes) but the modern installer
    // reports it as absent — mirrors upstream >= 1.8.0 where bundled is the
    // default and "plugin not found" means the plugin is genuinely missing
    const installerRoot = join(testDir, "installer-modern-missing");
    createFakeInstaller(installerRoot, "fake-cb-plugin", { modern: true });
    process.env.CANNBOT_INSTALLER_PATH = installerRoot;
    process.env.FAKE_CANNBOT_MODERN_NOT_FOUND = "1";

    const target = join(testDir, "modern-missing-project");
    mkdirSync(target, { recursive: true });

    const { delegateInstall } = await import("../src/core/cannbot-delegate.js");
    const result = await delegateInstall({
      pluginId: "fake-cb-plugin",
      tool: "opencode",
      level: "project",
      target,
    });

    // no bundled fallback on modern installers — the original failure wins
    expect(result.success).toBe(false);
    expect(result.warnings).toEqual([]);
    expect(result.errors.join(" ")).toContain("cannbot 安装器执行失败");
    expect(existsSync(join(target, ".opencode/cannbot-plugin.json"))).toBe(false);
  });

  it("end-to-end: install then uninstallCannbotPlugin leaves no residue", async () => {
    useFakeInstaller("fake-cb-plugin");
    const target = join(testDir, "e2e-project");
    mkdirSync(target, { recursive: true });

    const { delegateInstall } = await import("../src/core/cannbot-delegate.js");
    const { uninstallCannbotPlugin, scanCannbotInstalled } = await import("../src/core/cannbot-registry.js");

    const installResult = await delegateInstall({
      pluginId: "fake-cb-plugin",
      tool: "opencode",
      level: "project",
      target,
    });
    expect(installResult.success).toBe(true);
    expect(scanCannbotInstalled(target).map((e) => e.id)).toEqual(["fake-cb-plugin"]);

    const uninstallResult = uninstallCannbotPlugin("fake-cb-plugin", target);
    expect(uninstallResult.success).toBe(true);

    expect(scanCannbotInstalled(target)).toEqual([]);
    expect(existsSync(join(target, ".agents"))).toBe(false);
    expect(existsSync(join(target, ".opencode"))).toBe(false);
    expect(existsSync(join(target, "AGENTS.md"))).toBe(false);
    expect(existsSync(join(sandboxHome, ".cannbot/installs/fake-cb-plugin.json"))).toBe(false);
  });
});
