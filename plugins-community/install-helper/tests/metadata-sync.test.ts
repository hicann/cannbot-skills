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

  it("overwrites stale non-empty installAgents with plugin.json (SoT regression)", async () => {
    // Regression: previously metadata-sync only filled installAgents from
    // plugin.json when it was empty. A stale non-empty list (e.g. abandoned
    // agent names) would silently win. Now plugin.json is always authoritative.
    const pluginDir = join(testDir, "plugins-official", "stale-plugin");
    mkdirSync(join(pluginDir, ".claude-plugin"), { recursive: true });
    mkdirSync(join(pluginDir, "agents"), { recursive: true });

    writeFileSync(
      join(pluginDir, ".claude-plugin", "plugin.json"),
      JSON.stringify({
        name: "stale-plugin",
        version: "3.0.0",
        description: "Stale agents test",
        agents: ["./agents/correct-agent.md"],
      })
    );
    // the correct agent exists on disk
    writeFileSync(join(pluginDir, "agents", "correct-agent.md"), "# correct");

    const registryMod = await import("../src/core/registry.js");
    const { enrichPluginMetadata } = await import("../src/core/metadata-sync.js");

    const testPlugin = {
      id: "stale-plugin",
      dir: "plugins-official/stale-plugin",
      displayName: "Stale Plugin",
      script: "init.sh",
      aliases: [],
      skills: 0,
      agents: 3,
      description: "",
      version: "1.0.0",
      installSkills: [],
      // stale names that do NOT exist in plugin.json — must be overwritten
      installAgents: ["stale-analyst", "stale-developer", "stale-perf-tuner"],
    };

    registryMod.PLUGIN_REGISTRY.push(testPlugin);

    try {
      enrichPluginMetadata(testDir);

      expect(testPlugin.installAgents).toEqual(["correct-agent"]);
      expect(testPlugin.agents).toBe(1);
      expect(testPlugin.installAgents).not.toContain("stale-analyst");
      expect(testPlugin.installAgents).not.toContain("stale-developer");
      expect(testPlugin.installAgents).not.toContain("stale-perf-tuner");
    } finally {
      const idx = registryMod.PLUGIN_REGISTRY.indexOf(testPlugin);
      if (idx >= 0) registryMod.PLUGIN_REGISTRY.splice(idx, 1);
    }
  });

  it("falls back to init.sh INCLUDED_AGENT_PATTERN when plugin.json has no agents field", async () => {
    // plugin.json missing agents field → discoverAgents uses init.sh pattern
    const pluginDir = join(testDir, "plugins-official", "pattern-plugin");
    mkdirSync(join(pluginDir, ".claude-plugin"), { recursive: true });
    mkdirSync(join(pluginDir, "agents"), { recursive: true });

    writeFileSync(
      join(pluginDir, ".claude-plugin", "plugin.json"),
      JSON.stringify({ name: "pattern-plugin", version: "1.0.0" }) // NO agents field
    );
    writeFileSync(
      join(pluginDir, "init.sh"),
      `#!/bin/bash\nINCLUDED_AGENT_PATTERN="pat-*"\nINCLUDED_SKILLS=""\n`
    );
    writeFileSync(join(pluginDir, "agents", "pat-alpha.md"), "# alpha");
    writeFileSync(join(pluginDir, "agents", "pat-beta.md"), "# beta");
    writeFileSync(join(pluginDir, "agents", "other.md"), "# other");

    const registryMod = await import("../src/core/registry.js");
    const { enrichPluginMetadata } = await import("../src/core/metadata-sync.js");

    const testPlugin = {
      id: "pattern-plugin",
      dir: "plugins-official/pattern-plugin",
      displayName: "Pattern Plugin",
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

      expect(testPlugin.installAgents).toContain("pat-alpha");
      expect(testPlugin.installAgents).toContain("pat-beta");
      expect(testPlugin.installAgents).not.toContain("other");
      expect(testPlugin.agents).toBe(2);
    } finally {
      const idx = registryMod.PLUGIN_REGISTRY.indexOf(testPlugin);
      if (idx >= 0) registryMod.PLUGIN_REGISTRY.splice(idx, 1);
    }
  });

  it("does not fall back to pattern when plugin.json has agents:[] (explicit empty)", async () => {
    const pluginDir = join(testDir, "plugins-official", "explicit-empty-plugin");
    mkdirSync(join(pluginDir, ".claude-plugin"), { recursive: true });
    mkdirSync(join(pluginDir, "agents"), { recursive: true });

    writeFileSync(
      join(pluginDir, ".claude-plugin", "plugin.json"),
      JSON.stringify({ name: "explicit-empty", version: "1.0.0", agents: [] })
    );
    writeFileSync(
      join(pluginDir, "init.sh"),
      `#!/bin/bash\nINCLUDED_AGENT_PATTERN="ex-*"\nINCLUDED_SKILLS=""\n`
    );
    writeFileSync(join(pluginDir, "agents", "ex-agent.md"), "# agent");

    const registryMod = await import("../src/core/registry.js");
    const { enrichPluginMetadata } = await import("../src/core/metadata-sync.js");

    const testPlugin = {
      id: "explicit-empty-plugin",
      dir: "plugins-official/explicit-empty-plugin",
      displayName: "Explicit Empty",
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

      expect(testPlugin.installAgents).toEqual([]);
      expect(testPlugin.agents).toBe(0);
    } finally {
      const idx = registryMod.PLUGIN_REGISTRY.indexOf(testPlugin);
      if (idx >= 0) registryMod.PLUGIN_REGISTRY.splice(idx, 1);
    }
  });

  it("Layer 3: falls back to embedded installAgents when plugin.json and pattern both empty", async () => {
    // No plugin.json agents, no init.sh pattern → Layer 3 uses embedded installAgents
    const pluginDir = join(testDir, "plugins-official", "l3-plugin");
    mkdirSync(join(pluginDir, "agents"), { recursive: true });

    writeFileSync(
      join(pluginDir, "init.sh"),
      `#!/bin/bash\nINCLUDED_SKILLS=""\n`
    );
    // NO INCLUDED_AGENT_PATTERN in init.sh
    // NO .claude-plugin/plugin.json
    writeFileSync(join(pluginDir, "agents", "emb-alpha.md"), "# alpha");
    writeFileSync(join(pluginDir, "agents", "emb-beta.md"), "# beta");

    const registryMod = await import("../src/core/registry.js");
    const { enrichPluginMetadata } = await import("../src/core/metadata-sync.js");

    const testPlugin = {
      id: "l3-plugin",
      dir: "plugins-official/l3-plugin",
      displayName: "L3 Plugin",
      script: "init.sh",
      aliases: [],
      skills: 0,
      agents: 2,
      description: "",
      version: "1.0.0",
      installSkills: [],
      // embedded value acts as Layer 3 fallback
      installAgents: ["emb-alpha", "emb-beta"],
    };

    registryMod.PLUGIN_REGISTRY.push(testPlugin);

    try {
      enrichPluginMetadata(testDir);

      // Layer 3 fallback: embedded installAgents preserved
      expect(testPlugin.installAgents).toEqual(["emb-alpha", "emb-beta"]);
      expect(testPlugin.agents).toBe(2);
    } finally {
      const idx = registryMod.PLUGIN_REGISTRY.indexOf(testPlugin);
      if (idx >= 0) registryMod.PLUGIN_REGISTRY.splice(idx, 1);
    }
  });

  it("plugin-local skills/ discovered via findSkillSourceDir fallback", async () => {
    // Skill in init.sh INCLUDED_SKILLS but living in <plugin>/skills/ (not scanDirs)
    const pluginDir = join(testDir, "plugins-official", "local-skill-plugin");
    mkdirSync(join(pluginDir, "skills", "my-local-skill"), { recursive: true });
    writeFileSync(join(pluginDir, "skills", "my-local-skill", "SKILL.md"), "# local");

    writeFileSync(
      join(pluginDir, "init.sh"),
      `#!/bin/bash\nINCLUDED_SKILLS="my-local-skill"\n`
    );

    const registryMod = await import("../src/core/registry.js");
    const { enrichPluginMetadata } = await import("../src/core/metadata-sync.js");

    const testPlugin = {
      id: "local-skill-plugin",
      dir: "plugins-official/local-skill-plugin",
      displayName: "Local Skill Plugin",
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
      const allSkills = testPlugin.installSkills!.flatMap((s) => s.skills);
      expect(allSkills).toContain("my-local-skill");
      // the bucket dir should be the plugin-local path
      const bucket = testPlugin.installSkills!.find((s) =>
        s.skills.some((sk) => (typeof sk === "string" ? sk === "my-local-skill" : sk.name === "my-local-skill"))
      );
      expect(bucket).toBeDefined();
      expect(bucket!.dir).toContain("skills");
    } finally {
      const idx = registryMod.PLUGIN_REGISTRY.indexOf(testPlugin);
      if (idx >= 0) registryMod.PLUGIN_REGISTRY.splice(idx, 1);
    }
  });

  it("ALL_SKILLS static value parsed when INCLUDED_SKILLS absent", async () => {
    const pluginDir = join(testDir, "plugins-official", "all-skills-plugin");
    mkdirSync(join(testDir, "ops", "all-skill-x"), { recursive: true });
    mkdirSync(join(testDir, "ops", "all-skill-y"), { recursive: true });
    mkdirSync(pluginDir, { recursive: true });
    writeFileSync(
      join(pluginDir, "init.sh"),
      `#!/bin/bash\nALL_SKILLS="all-skill-x all-skill-y"\n`
    );
    // NO INCLUDED_SKILLS

    const registryMod = await import("../src/core/registry.js");
    const { enrichPluginMetadata } = await import("../src/core/metadata-sync.js");

    const testPlugin = {
      id: "all-skills-plugin",
      dir: "plugins-official/all-skills-plugin",
      displayName: "All Skills Plugin",
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
      const allSkills = testPlugin.installSkills!.flatMap((s) => s.skills);
      expect(allSkills).toContain("all-skill-x");
      expect(allSkills).toContain("all-skill-y");
      expect(testPlugin.skills).toBeGreaterThanOrEqual(2);
    } finally {
      const idx = registryMod.PLUGIN_REGISTRY.indexOf(testPlugin);
      if (idx >= 0) registryMod.PLUGIN_REGISTRY.splice(idx, 1);
    }
  });

  it("ALL_SKILLS dynamic value ($(...)) not parsed", async () => {
    const pluginDir = join(testDir, "plugins-official", "dyn-all-skills-plugin");
    mkdirSync(pluginDir, { recursive: true });
    writeFileSync(
      join(pluginDir, "init.sh"),
      `#!/bin/bash\nALL_SKILLS=$(echo "dyn-skill" | sort -u)\n`
    );

    const registryMod = await import("../src/core/registry.js");
    const { enrichPluginMetadata } = await import("../src/core/metadata-sync.js");

    const testPlugin = {
      id: "dyn-all-skills-plugin",
      dir: "plugins-official/dyn-all-skills-plugin",
      displayName: "Dyn All Skills",
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
      // Dynamic ALL_SKILLS should NOT be parsed → installSkills stays empty
      expect(testPlugin.installSkills).toEqual([]);
      expect(testPlugin.skills).toBe(0);
    } finally {
      const idx = registryMod.PLUGIN_REGISTRY.indexOf(testPlugin);
      if (idx >= 0) registryMod.PLUGIN_REGISTRY.splice(idx, 1);
    }
  });
});
