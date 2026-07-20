// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdirSync, writeFileSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

let testDir: string;

function setup() {
  testDir = join(tmpdir(), `ih-meta-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
}

function teardown() {
  rmSync(testDir, { recursive: true, force: true });
}

describe("metadata-sync", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("enriches plugin metadata from plugin.json and init.sh", async () => {
    const pluginDir = join(testDir, "plugins-official", "test-plugin");
    mkdirSync(join(pluginDir, ".claude-plugin"), { recursive: true });
    mkdirSync(join(pluginDir, "agents"), { recursive: true });

    writeFileSync(
      join(pluginDir, ".claude-plugin", "plugin.json"),
      JSON.stringify({
        name: "test-plugin",
        version: "2.0.0",
        description: "Test plugin for metadata sync",
        agents: ["./agents/test-agent.md"],
      })
    );

    writeFileSync(
      join(pluginDir, "init.sh"),
      `#!/bin/bash\nINCLUDED_SKILLS="skill-a skill-b"\n`
    );

    mkdirSync(join(testDir, "ops", "skill-a"), { recursive: true });
    mkdirSync(join(testDir, "ops", "skill-b"), { recursive: true });

    const registryMod = await import("../src/core/registry.js");
    const { enrichPluginMetadata } = await import("../src/core/metadata-sync.js");

    const testPlugin = {
      id: "test-plugin",
      dir: "plugins-official/test-plugin",
      displayName: "test-plugin",
      script: "init.sh",
      aliases: [],
      skills: 0,
      agents: 0,
      description: "",
      version: "1.0.0",
      installSkills: [],
      installAgents: [],
    };

    registryMod.PLUGIN_REGISTRY.push(testPlugin);

    try {
      enrichPluginMetadata(testDir);

      expect(testPlugin.version).toBe("2.0.0");
      expect(testPlugin.description).toBe("Test plugin for metadata sync");
      expect(testPlugin.displayName).toBe("test-plugin");
      expect(testPlugin.installAgents).toEqual(["test-agent"]);
      expect(testPlugin.installSkills).toHaveLength(1);
      expect(testPlugin.installSkills![0].dir).toBe("ops");
      expect(testPlugin.installSkills![0].skills).toEqual(["skill-a", "skill-b"]);
      expect(testPlugin.skills).toBe(2);
      expect(testPlugin.agents).toBe(1);
    } finally {
      const idx = registryMod.PLUGIN_REGISTRY.indexOf(testPlugin);
      if (idx >= 0) registryMod.PLUGIN_REGISTRY.splice(idx, 1);
    }
  });

  it("preserves existing installSkills, appends only init.sh-only skills", async () => {
    const pluginDir = join(testDir, "plugins-official", "merge-plugin");
    mkdirSync(join(pluginDir, ".claude-plugin"), { recursive: true });

    writeFileSync(
      join(pluginDir, ".claude-plugin", "plugin.json"),
      JSON.stringify({ name: "merge-plugin", version: "1.5.0" })
    );

    writeFileSync(
      join(pluginDir, "init.sh"),
      `#!/bin/bash\nINCLUDED_SKILLS="existing-skill new-skill"\n`
    );

    mkdirSync(join(testDir, "ops", "existing-skill"), { recursive: true });
    mkdirSync(join(testDir, "ops", "new-skill"), { recursive: true });

    const registryMod = await import("../src/core/registry.js");
    const { enrichPluginMetadata } = await import("../src/core/metadata-sync.js");

    const testPlugin = {
      id: "merge-plugin",
      dir: "plugins-official/merge-plugin",
      displayName: "Merge Plugin",
      script: "init.sh",
      aliases: [],
      skills: 1,
      agents: 0,
      description: "Existing desc",
      version: "1.0.0",
      installSkills: [{ dir: "ops", skills: ["existing-skill"] }],
      installAgents: [],
    };

    registryMod.PLUGIN_REGISTRY.push(testPlugin);

    try {
      enrichPluginMetadata(testDir);

      const allSkills = testPlugin.installSkills!.flatMap((s) => s.skills);
      expect(allSkills).toContain("existing-skill");
      expect(allSkills).toContain("new-skill");
      expect(allSkills.filter((s) => s === "existing-skill")).toHaveLength(1);
      expect(testPlugin.skills).toBe(2);
      expect(testPlugin.description).toBe("Existing desc");
    } finally {
      const idx = registryMod.PLUGIN_REGISTRY.indexOf(testPlugin);
      if (idx >= 0) registryMod.PLUGIN_REGISTRY.splice(idx, 1);
    }
  });

  it("skips plugins whose dir does not exist in repo", async () => {
    const registryMod = await import("../src/core/registry.js");
    const { enrichPluginMetadata } = await import("../src/core/metadata-sync.js");

    const testPlugin = {
      id: "nonexistent-plugin",
      dir: "plugins-official/nonexistent",
      displayName: "Nonexistent",
      script: "init.sh",
      aliases: [],
      skills: 0,
      agents: 0,
      description: "Should not change",
      version: "1.0.0",
      installSkills: [],
      installAgents: [],
    };

    registryMod.PLUGIN_REGISTRY.push(testPlugin);

    try {
      enrichPluginMetadata(testDir);

      expect(testPlugin.version).toBe("1.0.0");
      expect(testPlugin.description).toBe("Should not change");
      expect(testPlugin.skills).toBe(0);
    } finally {
      const idx = registryMod.PLUGIN_REGISTRY.indexOf(testPlugin);
      if (idx >= 0) registryMod.PLUGIN_REGISTRY.splice(idx, 1);
    }
  });
});
