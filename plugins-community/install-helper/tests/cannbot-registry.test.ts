// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdirSync, writeFileSync, existsSync, rmSync, readFileSync, symlinkSync } from "fs";
import { join, dirname } from "path";
import { tmpdir } from "os";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let testDir: string;
let sandboxHome: string;
let savedHome: string | undefined;

function setup() {
  testDir = join(tmpdir(), `ih-cbr-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
  // sandbox ~/.cannbot so delegated-record cleanup never touches the real home
  sandboxHome = join(testDir, "home");
  mkdirSync(join(sandboxHome, ".cannbot"), { recursive: true });
  savedHome = process.env.HOME;
  process.env.HOME = sandboxHome;
}

function teardown() {
  if (savedHome === undefined) delete process.env.HOME;
  else process.env.HOME = savedHome;
  rmSync(testDir, { recursive: true, force: true });
}

interface FakeRecord {
  plugin: string;
  tool?: string;
  skills?: string[];
  agents?: string[];
  instructions?: string | null;
  assets?: string | null;
  permissions?: string | null;
  settings?: string | null;
  hookSettings?: string | null;
  dependencies?: string[];
  schemaVersion?: number;
}

function writeRegistry(target: string, tool: string, records: FakeRecord[], schemaVersion = 2): string {
  const configDirs: Record<string, string> = {
    opencode: ".opencode",
    codex: ".codex",
    claude: ".claude",
    trae: ".trae",
  };
  const configDir = join(target, configDirs[tool]);
  mkdirSync(configDir, { recursive: true });
  const registryPath = join(configDir, "cannbot-plugin.json");
  writeFileSync(
    registryPath,
    JSON.stringify(
      {
        schemaVersion,
        plugins: records.map((r) => ({
          plugin: r.plugin,
          pluginVersion: "1.0.0",
          sourcePackage: "@cannbot-plugin/cannbot@9.9.9",
          source: { kind: "package", package: "@cannbot-plugin/cannbot@9.9.9" },
          tool: r.tool ?? tool,
          skillInstallMode: "copy",
          skills: r.skills ?? [],
          agents: r.agents ?? [],
          instructions: r.instructions ?? null,
          assets: r.assets ?? null,
          permissions: r.permissions ?? null,
          settings: r.settings ?? null,
          hookSettings: r.hookSettings ?? null,
          dependencies: r.dependencies ?? [],
          nativePlugin: null,
          unsupported: [],
        })),
      },
      null,
      2
    )
  );
  return registryPath;
}

function materializePlugin(target: string, pluginId: string, record: FakeRecord): void {
  for (const rel of [...(record.skills ?? []), ...(record.agents ?? [])]) {
    const p = join(target, rel);
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, "fake content\n");
  }
  if (record.assets) {
    mkdirSync(join(target, record.assets, "workflows"), { recursive: true });
    writeFileSync(join(target, record.assets, "workflows", "flow.yaml"), "steps: []\n");
  }
  if (record.permissions) {
    mkdirSync(join(target, record.permissions), { recursive: true });
    writeFileSync(join(target, record.permissions, "hook.js"), "// fake\n");
  }
  if (record.settings) {
    mkdirSync(dirname(join(target, record.settings)), { recursive: true });
    writeFileSync(join(target, record.settings), JSON.stringify({ plugins: [pluginId] }, null, 2));
  }
}

describe("cannbot-registry", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("readCannbotRegistry returns null when no registry exists", async () => {
    const { readCannbotRegistry } = await import("../src/core/cannbot-registry.js");
    expect(readCannbotRegistry(testDir, "opencode")).toBeNull();
  });

  it("readCannbotRegistry parses a valid registry and flags schema support", async () => {
    writeRegistry(testDir, "opencode", [{ plugin: "alpha", skills: [".agents/skills/a"] }]);
    const { readCannbotRegistry } = await import("../src/core/cannbot-registry.js");
    const result = readCannbotRegistry(testDir, "opencode");
    expect(result).not.toBeNull();
    expect(result!.schemaSupported).toBe(true);
    expect(result!.registry.plugins[0].plugin).toBe("alpha");
  });

  it("readCannbotRegistry flags unsupported schemaVersion", async () => {
    writeRegistry(testDir, "opencode", [{ plugin: "alpha" }], 3);
    const { readCannbotRegistry } = await import("../src/core/cannbot-registry.js");
    const result = readCannbotRegistry(testDir, "opencode");
    expect(result!.schemaSupported).toBe(false);
  });

  it("readCannbotRegistry tolerates corrupted JSON", async () => {
    mkdirSync(join(testDir, ".opencode"), { recursive: true });
    writeFileSync(join(testDir, ".opencode", "cannbot-plugin.json"), "{ not json");
    const { readCannbotRegistry } = await import("../src/core/cannbot-registry.js");
    expect(readCannbotRegistry(testDir, "opencode")).toBeNull();
  });

  it("toolConfigDir probes trae variants in order", async () => {
    const { findCannbotRegistryPath } = await import("../src/core/cannbot-registry.js");
    // no variant exists -> default .trae (no registry there either)
    expect(findCannbotRegistryPath(testDir, "trae")).toBeNull();

    // registry inside the .trae-cn variant wins over the default .trae
    mkdirSync(join(testDir, ".trae-cn"), { recursive: true });
    writeFileSync(
      join(testDir, ".trae-cn", "cannbot-plugin.json"),
      JSON.stringify({ schemaVersion: 2, plugins: [{ plugin: "t" }] })
    );
    expect(findCannbotRegistryPath(testDir, "trae")).toBe(join(testDir, ".trae-cn", "cannbot-plugin.json"));
  });

  it("scanCannbotInstalled collects records across tools", async () => {
    writeRegistry(testDir, "opencode", [{ plugin: "alpha", skills: [".agents/skills/a"] }]);
    writeRegistry(testDir, "claude", [{ plugin: "beta", agents: [".claude/agents/b.md"] }]);
    const { scanCannbotInstalled, findCannbotRecord } = await import("../src/core/cannbot-registry.js");

    const installed = scanCannbotInstalled(testDir);
    expect(installed.map((e) => e.id).sort()).toEqual(["alpha", "beta"]);
    expect(installed.find((e) => e.id === "alpha")!.skillsCount).toBe(1);

    expect(findCannbotRecord("beta", testDir)?.tool).toBe("claude");
    expect(findCannbotRecord("missing", testDir)).toBeNull();
  });

  it("removeMarkerBlock strips only the target plugin's block", async () => {
    const { removeMarkerBlock } = await import("../src/core/cannbot-registry.js");
    const content = [
      "# project instructions",
      "",
      "<!-- cannbot:alpha:start -->",
      "alpha content",
      "<!-- cannbot:alpha:end -->",
      "",
      "<!-- cannbot:beta:start -->",
      "beta content",
      "<!-- cannbot:beta:end -->",
      "",
    ].join("\n");

    const cleaned = removeMarkerBlock(content, "alpha");
    expect(cleaned).not.toContain("alpha content");
    expect(cleaned).not.toContain("cannbot:alpha");
    expect(cleaned).toContain("beta content");
    expect(cleaned).toContain("# project instructions");
  });

  it("uninstallCannbotPlugin removes the full install and leaves siblings intact", async () => {
    const alpha = {
      plugin: "alpha",
      skills: [".agents/skills/alpha-skill"],
      agents: [".opencode/agents/alpha-agent.md"],
      instructions: "AGENTS.md",
      assets: ".cannbot/plugins/alpha",
      permissions: ".cannbot/permissions",
      settings: ".cannbot/settings.json",
    };
    const beta = {
      plugin: "beta",
      skills: [".agents/skills/beta-skill"],
      agents: [".opencode/agents/beta-agent.md"],
      instructions: "AGENTS.md",
      assets: ".cannbot/plugins/beta",
      permissions: ".cannbot/permissions",
      settings: ".cannbot/settings.json",
    };
    materializePlugin(testDir, "alpha", alpha);
    materializePlugin(testDir, "beta", beta);
    writeFileSync(
      join(testDir, "AGENTS.md"),
      [
        "# root",
        "",
        "<!-- cannbot:alpha:start -->",
        "alpha instructions",
        "<!-- cannbot:alpha:end -->",
        "",
        "<!-- cannbot:beta:start -->",
        "beta instructions",
        "<!-- cannbot:beta:end -->",
        "",
      ].join("\n")
    );
    writeRegistry(testDir, "opencode", [alpha, beta]);

    // a delegated install record for alpha (should be removed too)
    const installsDir = join(sandboxHome, ".cannbot", "installs");
    mkdirSync(installsDir, { recursive: true });
    writeFileSync(
      join(installsDir, "alpha.json"),
      JSON.stringify({ pluginId: "alpha", kind: "delegated", files: [], directories: [] })
    );

    const { uninstallCannbotPlugin, scanCannbotInstalled } = await import("../src/core/cannbot-registry.js");
    const result = uninstallCannbotPlugin("alpha", testDir);

    expect(result.success).toBe(true);
    expect(result.errors).toEqual([]);

    // skills/agents removed
    expect(existsSync(join(testDir, ".agents/skills/alpha-skill"))).toBe(false);
    expect(existsSync(join(testDir, ".opencode/agents/alpha-agent.md"))).toBe(false);
    // sibling untouched
    expect(existsSync(join(testDir, ".agents/skills/beta-skill"))).toBe(true);
    expect(existsSync(join(testDir, ".opencode/agents/beta-agent.md"))).toBe(true);

    // marker block stripped but file kept for beta
    const agentsMd = readFileSync(join(testDir, "AGENTS.md"), "utf-8");
    expect(agentsMd).not.toContain("alpha instructions");
    expect(agentsMd).toContain("beta instructions");
    expect(agentsMd).toContain("# root");

    // plugin assets removed
    expect(existsSync(join(testDir, ".cannbot/plugins/alpha"))).toBe(false);
    expect(existsSync(join(testDir, ".cannbot/plugins/beta"))).toBe(true);

    // shared runtime paths kept because beta still references them
    expect(existsSync(join(testDir, ".cannbot/permissions"))).toBe(true);
    expect(existsSync(join(testDir, ".cannbot/settings.json"))).toBe(true);

    // registry updated
    const remaining = scanCannbotInstalled(testDir);
    expect(remaining.map((e) => e.id)).toEqual(["beta"]);

    // delegated record deleted
    expect(existsSync(join(installsDir, "alpha.json"))).toBe(false);

    // uninstalling the last plugin removes shared paths and the registry
    const second = uninstallCannbotPlugin("beta", testDir);
    expect(second.success).toBe(true);
    expect(existsSync(join(testDir, ".cannbot/permissions"))).toBe(false);
    expect(existsSync(join(testDir, ".cannbot/settings.json"))).toBe(false);
    expect(existsSync(join(testDir, ".opencode/cannbot-plugin.json"))).toBe(false);
    expect(scanCannbotInstalled(testDir)).toEqual([]);
  });

  it("uninstallCannbotPlugin refuses an unsupported schemaVersion", async () => {
    writeRegistry(testDir, "opencode", [{ plugin: "alpha" }], 99);
    const { uninstallCannbotPlugin } = await import("../src/core/cannbot-registry.js");
    const result = uninstallCannbotPlugin("alpha", testDir);
    expect(result.success).toBe(false);
    expect(result.errors[0]).toContain("schemaVersion");
  });

  it("uninstallCannbotPlugin reports no-record for unknown plugins", async () => {
    const { uninstallCannbotPlugin } = await import("../src/core/cannbot-registry.js");
    const result = uninstallCannbotPlugin("ghost", testDir);
    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
  });

  it("uninstallCannbotPlugin surgically cleans hook settings entries", async () => {
    const record = {
      plugin: "gamma",
      skills: [".claude/skills/gamma-skill"],
      agents: [".claude/agents/gamma-agent.md"],
      instructions: "CLAUDE.md",
      assets: ".cannbot/plugins/gamma",
      hookSettings: ".claude/settings.json",
    };
    materializePlugin(testDir, "gamma", record);
    mkdirSync(join(testDir, ".claude"), { recursive: true });
    writeFileSync(
      join(testDir, "CLAUDE.md"),
      "<!-- cannbot:gamma:start -->\ngamma\n<!-- cannbot:gamma:end -->\n"
    );
    writeFileSync(
      join(testDir, ".claude", "settings.json"),
      JSON.stringify(
        {
          model: "keep-me",
          hooks: {
            PreToolUse: [
              { hooks: [{ command: "node .cannbot/plugins/gamma/hooks/permission-guard.js" }] },
              { hooks: [{ command: "node ~/.other/tool.js" }] },
            ],
          },
        },
        null,
        2
      )
    );
    writeRegistry(testDir, "claude", [record]);

    const { uninstallCannbotPlugin } = await import("../src/core/cannbot-registry.js");
    const result = uninstallCannbotPlugin("gamma", testDir);
    expect(result.success).toBe(true);

    const settings = JSON.parse(readFileSync(join(testDir, ".claude/settings.json"), "utf-8"));
    expect(settings.model).toBe("keep-me");
    expect(settings.hooks.PreToolUse).toHaveLength(1);
    expect(settings.hooks.PreToolUse[0].hooks[0].command).toContain("other");
  });

  it("uninstallCannbotPlugin removes dependency clones", async () => {
    const record = {
      plugin: "delta",
      skills: [".agents/skills/delta-skill"],
      dependencies: [".cannbot/dependencies/delta/asc-devkit"],
    };
    materializePlugin(testDir, "delta", record);
    mkdirSync(join(testDir, ".cannbot/dependencies/delta/asc-devkit"), { recursive: true });
    writeFileSync(join(testDir, ".cannbot/dependencies/delta/asc-devkit/README.md"), "x\n");
    writeRegistry(testDir, "opencode", [record]);

    const { uninstallCannbotPlugin } = await import("../src/core/cannbot-registry.js");
    const result = uninstallCannbotPlugin("delta", testDir);
    expect(result.success).toBe(true);
    expect(existsSync(join(testDir, ".cannbot/dependencies/delta"))).toBe(false);
    expect(existsSync(join(testDir, ".cannbot"))).toBe(false);
  });

  it("uninstall cleans layout roots left empty by 0-agent plugins", async () => {
    // the cannbot installer creates <configDir>/agents even for 0-agent
    // plugins — uninstall must remove it when it ends up empty
    const record = {
      plugin: "solo",
      skills: [".agents/skills/solo-skill"],
      agents: [],
      instructions: null,
      assets: null,
    };
    materializePlugin(testDir, "solo", record);
    mkdirSync(join(testDir, ".opencode", "agents"), { recursive: true });
    writeRegistry(testDir, "opencode", [record]);

    const { uninstallCannbotPlugin, scanCannbotInstalled } = await import("../src/core/cannbot-registry.js");
    const result = uninstallCannbotPlugin("solo", testDir);
    expect(result.success).toBe(true);

    expect(scanCannbotInstalled(testDir)).toEqual([]);
    expect(existsSync(join(testDir, ".opencode"))).toBe(false);
    expect(existsSync(join(testDir, ".agents"))).toBe(false);
  });

  it("uninstall strips the opencode plugin registration when the last cannbot plugin is removed", async () => {
    const record = {
      plugin: "alpha",
      skills: [".agents/skills/alpha-skill"],
      agents: [".opencode/agents/alpha-agent.md"],
      instructions: null,
      assets: null,
      nativePlugin: { package: "@cannbot-plugin/cannbot@1.7.2" },
    };
    materializePlugin(testDir, "alpha", record);
    mkdirSync(join(testDir, ".opencode"), { recursive: true });
    writeFileSync(
      join(testDir, ".opencode", "opencode.json"),
      JSON.stringify({ plugin: ["@cannbot-plugin/cannbot@1.7.2", "some-other-plugin"] }, null, 2)
    );
    writeRegistry(testDir, "opencode", [record]);

    const { uninstallCannbotPlugin } = await import("../src/core/cannbot-registry.js");
    const result = uninstallCannbotPlugin("alpha", testDir);
    expect(result.success).toBe(true);

    // cannbot registration removed, other plugins kept
    const config = JSON.parse(readFileSync(join(testDir, ".opencode/opencode.json"), "utf-8"));
    expect(config.plugin).toEqual(["some-other-plugin"]);
  });

  it("uninstall removes opencode.json entirely when the registration list becomes empty", async () => {
    const record = {
      plugin: "solo",
      skills: [".agents/skills/solo-skill"],
      agents: [],
      instructions: null,
      assets: null,
      nativePlugin: { package: "@cannbot-plugin/cannbot@1.7.2" },
    };
    materializePlugin(testDir, "solo", record);
    mkdirSync(join(testDir, ".opencode"), { recursive: true });
    writeFileSync(
      join(testDir, ".opencode", "opencode.json"),
      JSON.stringify({ plugin: ["@cannbot-plugin/cannbot@1.7.2"] }, null, 2)
    );
    writeRegistry(testDir, "opencode", [record]);

    const { uninstallCannbotPlugin } = await import("../src/core/cannbot-registry.js");
    const result = uninstallCannbotPlugin("solo", testDir);
    expect(result.success).toBe(true);
    expect(existsSync(join(testDir, ".opencode", "opencode.json"))).toBe(false);
  });

  it("uninstall keeps the opencode plugin registration while sibling cannbot plugins remain", async () => {
    const alpha = {
      plugin: "alpha",
      skills: [".agents/skills/alpha-skill"],
      agents: [".opencode/agents/alpha-agent.md"],
      instructions: null,
      assets: null,
      nativePlugin: { package: "@cannbot-plugin/cannbot@1.7.2" },
    };
    const beta = {
      plugin: "beta",
      skills: [".agents/skills/beta-skill"],
      agents: [".opencode/agents/beta-agent.md"],
      instructions: null,
      assets: null,
      nativePlugin: { package: "@cannbot-plugin/cannbot@1.7.2" },
    };
    materializePlugin(testDir, "alpha", alpha);
    materializePlugin(testDir, "beta", beta);
    mkdirSync(join(testDir, ".opencode"), { recursive: true });
    writeFileSync(
      join(testDir, ".opencode", "opencode.json"),
      JSON.stringify({ plugin: ["@cannbot-plugin/cannbot@1.7.2"] }, null, 2)
    );
    writeRegistry(testDir, "opencode", [alpha, beta]);

    const { uninstallCannbotPlugin } = await import("../src/core/cannbot-registry.js");
    const result = uninstallCannbotPlugin("alpha", testDir);
    expect(result.success).toBe(true);

    // beta still installed: registration stays for it
    const config = JSON.parse(readFileSync(join(testDir, ".opencode/opencode.json"), "utf-8"));
    expect(config.plugin).toEqual(["@cannbot-plugin/cannbot@1.7.2"]);

    // removing the last sibling now clears it
    const second = uninstallCannbotPlugin("beta", testDir);
    expect(second.success).toBe(true);
    expect(existsSync(join(testDir, ".opencode", "opencode.json"))).toBe(false);
  });

  it("findOrphanedDelegatedRecords detects stale delegated records", async () => {    writeRegistry(testDir, "opencode", [{ plugin: "alpha" }]);
    const installsDir = join(sandboxHome, ".cannbot", "installs");
    mkdirSync(installsDir, { recursive: true });
    writeFileSync(
      join(installsDir, "alpha.json"),
      JSON.stringify({ pluginId: "alpha", kind: "delegated", files: [], directories: [] })
    );
    writeFileSync(
      join(installsDir, "ghost.json"),
      JSON.stringify({ pluginId: "ghost", kind: "delegated", files: [], directories: [] })
    );

    const { findOrphanedDelegatedRecords } = await import("../src/core/cannbot-registry.js");
    expect(findOrphanedDelegatedRecords(testDir)).toEqual(["ghost"]);
  });
});
