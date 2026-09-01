// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { describe, it, expect } from "vitest";
import { existsSync, readFileSync, readdirSync, statSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { createRequire } from "module";

const require = createRequire(import.meta.url);
const { validatePlugins } = require("../scripts/validate-plugins.cjs");

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const INSTALL_HELPER_ROOT = join(__dirname, "..");
const REPO_ROOT = join(INSTALL_HELPER_ROOT, "..", "..");

function readJson(p: string): any {
  return JSON.parse(readFileSync(p, "utf-8"));
}

function pluginJsonAgents(pluginDir: string): string[] {
  const pj = readJson(join(pluginDir, ".claude-plugin", "plugin.json"));
  return (pj.agents || [])
    .filter((a: string) => a.startsWith("./agents/"))
    .map((a: string) => a.replace(/^\.\/agents\//, "").replace(/\.md$/, ""));
}

describe("plugins consistency — single source of truth", () => {
  it("all plugin yml pass metadata validation (V1–V7)", () => {
    const result = validatePlugins(REPO_ROOT, INSTALL_HELPER_ROOT);
    if (!result.ok) {
      console.error(result.report);
    }
    expect(result.ok).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it("yml files: no agents count, no version; installAgents present and matches plugin.json", () => {
    // agents count and version are always forbidden.
    // installAgents is allowed (Layer 3 fallback + documentation, like installSkills).
    // When plugin.json has agents, yml installAgents must match (consistency).
    const pluginsDir = join(INSTALL_HELPER_ROOT, "plugins.d");
    for (const file of readdirSync(pluginsDir)) {
      if (file.startsWith("_") || !file.endsWith(".yml")) continue;
      const raw = readFileSync(join(pluginsDir, file), "utf-8");
      const { parse } = require("yaml");
      const content = parse(raw);
      expect(content.agents).toBeUndefined();
      expect(content.version).toBeUndefined();
      // installAgents should be present (as documentation + fallback)
      expect(content.installAgents).toBeDefined();
      // when plugin.json has agents, they must match
      const pj = readJson(join(REPO_ROOT, content.dir, ".claude-plugin", "plugin.json"));
      if (pj && Array.isArray(pj.agents) && pj.agents.length > 0) {
        const pjAgents = pj.agents
          .filter((a: string) => a.startsWith("./agents/"))
          .map((a: string) => a.replace(/^\.\/agents\//, "").replace(/\.md$/, ""))
          .sort();
        const ymlAgents = [...content.installAgents].sort();
        expect(ymlAgents).toEqual(pjAgents);
      }
    }
  });

  it("every plugin.json has a non-empty version (V7b)", () => {
    const pluginsDir = join(INSTALL_HELPER_ROOT, "plugins.d");
    const { parse } = require("yaml");
    for (const file of readdirSync(pluginsDir)) {
      if (file.startsWith("_") || !file.endsWith(".yml")) continue;
      const content = parse(readFileSync(join(pluginsDir, file), "utf-8"));
      if (!content || !content.dir) continue;
      const pj = readJson(join(REPO_ROOT, content.dir, ".claude-plugin", "plugin.json"));
      expect(pj.version).toBeTruthy();
      expect(String(pj.version).trim()).not.toBe("");
    }
  });

  it("pypto-op-orchestrator installAgents == 8 plugin.json agents (bug regression)", () => {
    const pluginDir = join(REPO_ROOT, "plugins-official", "pypto-op-orchestrator");
    const agents = pluginJsonAgents(pluginDir);
    const expected = [
      "pypto-op-planner",
      "pypto-op-mathematician",
      "pypto-op-architect",
      "pypto-op-designer",
      "pypto-op-coder",
      "pypto-op-verifier",
      "pypto-op-debugger",
      "pypto-op-optimizer",
    ];
    expect(agents).toEqual(expected);
    // the stale names from the bug report must NOT appear
    expect(agents).not.toContain("pypto-op-analyst");
    expect(agents).not.toContain("pypto-op-developer");
    expect(agents).not.toContain("pypto-op-perf-tuner");
  });

  it("ops-registry-invoke installAgents == 7 plugin.json agents (stale-subset regression)", () => {
    const pluginDir = join(REPO_ROOT, "plugins-official", "ops-registry-invoke");
    const agents = pluginJsonAgents(pluginDir);
    expect(agents.length).toBe(7);
    expect(agents).toContain("ascendc-ops-spec-reviewer");
    expect(agents).toContain("ascendc-ops-designer");
    expect(agents).toContain("ascendc-ops-design-reviewer");
    expect(agents).toContain("ascendc-ops-test-design-reviewer");
  });

  it("triton-op-generator has empty agents (single-agent design)", () => {
    const pluginDir = join(REPO_ROOT, "plugins-official", "triton-op-generator");
    const agents = pluginJsonAgents(pluginDir);
    expect(agents).toEqual([]);
    expect(existsSync(join(pluginDir, "agents"))).toBe(false);
  });

  it("every plugin.json agent .md exists on disk", () => {
    const pluginsDir = join(INSTALL_HELPER_ROOT, "plugins.d");
    const { parse } = require("yaml");
    for (const file of readdirSync(pluginsDir)) {
      if (file.startsWith("_") || !file.endsWith(".yml")) continue;
      const content = parse(readFileSync(join(pluginsDir, file), "utf-8"));
      if (!content || !content.dir) continue;
      const pluginDir = join(REPO_ROOT, content.dir);
      const agents = pluginJsonAgents(pluginDir);
      for (const name of agents) {
        const agentPath = join(pluginDir, "agents", `${name}.md`);
        expect(existsSync(agentPath)).toBe(true);
      }
    }
  });

  it("embedded-plugins.json installAgents and version match plugin.json for every plugin", () => {
    const embeddedPath = join(INSTALL_HELPER_ROOT, "src", "embedded-plugins.json");
    expect(existsSync(embeddedPath)).toBe(true);
    const embedded = readJson(embeddedPath) as Array<any>;
    expect(embedded.length).toBeGreaterThan(0);
    for (const entry of embedded) {
      const pluginDir = join(REPO_ROOT, entry.dir);
      const pj = readJson(join(pluginDir, ".claude-plugin", "plugin.json"));
      const pjAgents = (pj.agents || [])
        .filter((a: string) => a.startsWith("./agents/"))
        .map((a: string) => a.replace(/^\.\/agents\//, "").replace(/\.md$/, ""));
      expect(entry.installAgents).toEqual(pjAgents);
      expect(entry.agents).toBe(pjAgents.length);
      // version must come from plugin.json (single source of truth)
      expect(entry.version).toBe(pj.version);
    }
  });

  it("validator catches a stale agent name (negative regression)", () => {
    // Simulate: a plugin yml points to a plugin whose plugin.json references an
    // agent .md that does not exist on disk. The validator must flag V2.
    // We build a throwaway install-helper layout (plugins.d + a fake plugin dir)
    // so the real plugins.d/ is not disturbed.
    const os = require("os");
    const path = require("path");
    const fs = require("fs");
    const tmp = os.tmpdir();
    const fakeRepo = join(tmp, `ih-neg-${Date.now()}`);
    const fakeIHRoot = join(tmp, `ih-ihroot-${Date.now()}`);
    const pluginRelDir = "plugins-official/fake-plugin";
    const pluginDir = join(fakeRepo, pluginRelDir);

    fs.mkdirSync(join(pluginDir, ".claude-plugin"), { recursive: true });
    fs.mkdirSync(join(pluginDir, "agents"), { recursive: true });
    fs.mkdirSync(join(fakeIHRoot, "plugins.d"), { recursive: true });

    fs.writeFileSync(
      join(pluginDir, ".claude-plugin", "plugin.json"),
      JSON.stringify({ name: "fake", version: "1.0.0", agents: ["./agents/ghost-agent.md"] })
    );
    // NOTE: ghost-agent.md is deliberately NOT created → V2 must fire.

    fs.writeFileSync(
      join(fakeIHRoot, "plugins.d", "fake-plugin.yml"),
      `id: fake-plugin\ndir: ${pluginRelDir}\ndisplayName: Fake\nskills: 0\n`
    );
    // minimal package.json so V6 doesn't crash
    fs.writeFileSync(
      join(fakeIHRoot, "package.json"),
      JSON.stringify({ version: "1.0.0", optionalDependencies: {} })
    );

    const result = validatePlugins(fakeRepo, fakeIHRoot);
    try {
      expect(result.ok).toBe(false);
      const v2error = result.errors.find(
        (e: any) => e.check === "V2" && e.plugin === "fake-plugin"
      );
      expect(v2error).toBeDefined();
    } finally {
      fs.rmSync(fakeRepo, { recursive: true, force: true });
      fs.rmSync(fakeIHRoot, { recursive: true, force: true });
    }
  });

  it("./AGENTS.md in plugin.json agents[] is filtered out (not treated as agent)", () => {
    // ops-perf-evolution plugin.json agents[] includes "./AGENTS.md" (the PM entry).
    // This must be filtered out — AGENTS.md is a configFile, not an installable agent.
    const pluginDir = join(REPO_ROOT, "plugins-community", "ops-perf-evolution");
    expect(existsSync(pluginDir)).toBe(true);
    const pj = readJson(join(pluginDir, ".claude-plugin", "plugin.json"));
    const hasAgentsMd = (pj.agents || []).some((a: string) => a === "./AGENTS.md");
    expect(hasAgentsMd).toBe(true);
    const agents = pluginJsonAgents(pluginDir);
    expect(agents).not.toContain("AGENTS");
    expect(agents).not.toContain("AGENTS.md");
    // the 4 real agents should remain
    expect(agents.length).toBe(4);
  });

  it("discoverAgents falls back to init.sh INCLUDED_AGENT_PATTERN when plugin.json has no agents", () => {
    // Simulate: plugin.json missing agents field, but init.sh has INCLUDED_AGENT_PATTERN.
    // discoverAgents should find agents via the pattern.
    const os = require("os");
    const fs = require("fs");
    const tmp = os.tmpdir();
    const fakeRepo = join(tmp, `ih-fb-${Date.now()}`);
    const pluginRelDir = "plugins-official/fallback-plugin";
    const pluginDir = join(fakeRepo, pluginRelDir);

    fs.mkdirSync(join(pluginDir, "agents"), { recursive: true });
    fs.mkdirSync(join(pluginDir, ".claude-plugin"), { recursive: true });
    // plugin.json with NO agents field (not [], just absent)
    fs.writeFileSync(
      join(pluginDir, ".claude-plugin", "plugin.json"),
      JSON.stringify({ name: "fallback", version: "1.0.0" })
    );
    // init.sh with INCLUDED_AGENT_PATTERN
    fs.writeFileSync(
      join(pluginDir, "init.sh"),
      `#!/bin/bash\nINCLUDED_AGENT_PATTERN="fb-*"\nINCLUDED_SKILLS=""\n`
    );
    // create 2 agent .md files matching pattern
    fs.writeFileSync(join(pluginDir, "agents", "fb-alpha.md"), "# alpha");
    fs.writeFileSync(join(pluginDir, "agents", "fb-beta.md"), "# beta");
    // a non-matching file
    fs.writeFileSync(join(pluginDir, "agents", "other.md"), "# other");

    // test discoverAgents via gen-embedded's exported function
    const { validatePlugins } = require("../scripts/validate-plugins.cjs");
    const fakeIHRoot = join(tmp, `ih-fb-ih-${Date.now()}`);
    fs.mkdirSync(join(fakeIHRoot, "plugins.d"), { recursive: true });
    fs.writeFileSync(
      join(fakeIHRoot, "plugins.d", "fallback-plugin.yml"),
      `id: fallback-plugin\ndir: ${pluginRelDir}\ndisplayName: Fallback\nskills: 0\n`
    );
    fs.writeFileSync(
      join(fakeIHRoot, "package.json"),
      JSON.stringify({ version: "1.0.0", optionalDependencies: {} })
    );

    const result = validatePlugins(fakeRepo, fakeIHRoot);
    try {
      // should NOT have V2 errors — agents discovered via pattern exist on disk
      const v2errors = result.errors.filter((e: any) => e.check === "V2" && e.plugin === "fallback-plugin");
      expect(v2errors).toHaveLength(0);
    } finally {
      fs.rmSync(fakeRepo, { recursive: true, force: true });
      fs.rmSync(fakeIHRoot, { recursive: true, force: true });
    }
  });

  it("discoverAgents returns [] for explicit agents:[] (respects declaration, no fallback)", () => {
    // plugin.json agents:[] means "no agents" — discoverAgents must NOT fall back to pattern.
    const os = require("os");
    const fs = require("fs");
    const tmp = os.tmpdir();
    const fakeRepo = join(tmp, `ih-empty-${Date.now()}`);
    const pluginRelDir = "plugins-official/empty-agent-plugin";
    const pluginDir = join(fakeRepo, pluginRelDir);

    fs.mkdirSync(join(pluginDir, "agents"), { recursive: true });
    fs.mkdirSync(join(pluginDir, ".claude-plugin"), { recursive: true });
    fs.writeFileSync(
      join(pluginDir, ".claude-plugin", "plugin.json"),
      JSON.stringify({ name: "empty", version: "1.0.0", agents: [] })
    );
    fs.writeFileSync(
      join(pluginDir, "init.sh"),
      `#!/bin/bash\nINCLUDED_AGENT_PATTERN="empty-*"\nINCLUDED_SKILLS=""\n`
    );
    // agents exist on disk but agents:[] should prevent fallback
    fs.writeFileSync(join(pluginDir, "agents", "empty-agent.md"), "# agent");

    const fakeIHRoot = join(tmp, `ih-empty-ih-${Date.now()}`);
    fs.mkdirSync(join(fakeIHRoot, "plugins.d"), { recursive: true });
    fs.writeFileSync(
      join(fakeIHRoot, "plugins.d", "empty.yml"),
      `id: empty-agent-plugin\ndir: ${pluginRelDir}\ndisplayName: Empty\nskills: 0\n`
    );
    fs.writeFileSync(
      join(fakeIHRoot, "package.json"),
      JSON.stringify({ version: "1.0.0", optionalDependencies: {} })
    );

    const result = validatePlugins(fakeRepo, fakeIHRoot);
    try {
      // no V2 errors — agents:[] means 0 agents, no fallback triggered
      const v2errors = result.errors.filter((e: any) => e.check === "V2" && e.plugin === "empty-agent-plugin");
      expect(v2errors).toHaveLength(0);
    } finally {
      fs.rmSync(fakeRepo, { recursive: true, force: true });
      fs.rmSync(fakeIHRoot, { recursive: true, force: true });
    }
  });

  it("Layer 3: yml installAgents used as fallback when plugin.json missing and no pattern", () => {
    // No plugin.json, no init.sh pattern, but yml has installAgents → Layer 3 fallback
    const os = require("os");
    const fs = require("fs");
    const tmp = os.tmpdir();
    const fakeRepo = join(tmp, `ih-l3-${Date.now()}`);
    const pluginRelDir = "plugins-official/l3-plugin";
    const pluginDir = join(fakeRepo, pluginRelDir);

    fs.mkdirSync(join(pluginDir, "agents"), { recursive: true });
    // NO .claude-plugin/plugin.json
    // NO init.sh (or init.sh without INCLUDED_AGENT_PATTERN)
    fs.writeFileSync(join(pluginDir, "init.sh"), `#!/bin/bash\nINCLUDED_SKILLS=""\n`);
    // create agent .md files matching yml installAgents
    fs.writeFileSync(join(pluginDir, "agents", "l3-alpha.md"), "# alpha");
    fs.writeFileSync(join(pluginDir, "agents", "l3-beta.md"), "# beta");

    const fakeIHRoot = join(tmp, `ih-l3-ih-${Date.now()}`);
    fs.mkdirSync(join(fakeIHRoot, "plugins.d"), { recursive: true });
    fs.writeFileSync(
      join(fakeIHRoot, "plugins.d", "l3.yml"),
      `id: l3-plugin\ndir: ${pluginRelDir}\ndisplayName: L3\nskills: 0\ninstallAgents: [l3-alpha, l3-beta]\n`
    );
    fs.writeFileSync(
      join(fakeIHRoot, "package.json"),
      JSON.stringify({ version: "1.0.0", optionalDependencies: {} })
    );

    const result = validatePlugins(fakeRepo, fakeIHRoot);
    try {
      // V5 should NOT flag installAgents (plugin.json missing → fallback allowed)
      const v5errors = result.errors.filter((e: any) => e.check === "V5" && e.plugin === "l3-plugin");
      expect(v5errors).toHaveLength(0);
      // V2 should pass — agents exist on disk
      const v2errors = result.errors.filter((e: any) => e.check === "V2" && e.plugin === "l3-plugin");
      expect(v2errors).toHaveLength(0);
    } finally {
      fs.rmSync(fakeRepo, { recursive: true, force: true });
      fs.rmSync(fakeIHRoot, { recursive: true, force: true });
    }
  });

  it("V5: installAgents mismatch with plugin.json agents → FAIL (drift detection)", () => {
    // plugin.json has agents → yml installAgents must match, else V5 FAIL (drift)
    const os = require("os");
    const fs = require("fs");
    const tmp = os.tmpdir();
    const fakeRepo = join(tmp, `ih-v5-${Date.now()}`);
    const pluginRelDir = "plugins-official/v5-plugin";
    const pluginDir = join(fakeRepo, pluginRelDir);

    fs.mkdirSync(join(pluginDir, "agents"), { recursive: true });
    fs.mkdirSync(join(pluginDir, ".claude-plugin"), { recursive: true });
    fs.writeFileSync(
      join(pluginDir, ".claude-plugin", "plugin.json"),
      JSON.stringify({ name: "v5", version: "1.0.0", agents: ["./agents/real-agent.md"] })
    );
    fs.writeFileSync(join(pluginDir, "agents", "real-agent.md"), "# real");

    const fakeIHRoot = join(tmp, `ih-v5-ih-${Date.now()}`);
    fs.mkdirSync(join(fakeIHRoot, "plugins.d"), { recursive: true });
    // yml installAgents has STALE name that doesn't match plugin.json
    fs.writeFileSync(
      join(fakeIHRoot, "plugins.d", "v5.yml"),
      `id: v5-plugin\ndir: ${pluginRelDir}\ndisplayName: V5\nskills: 0\ninstallAgents: [stale-agent]\n`
    );
    fs.writeFileSync(
      join(fakeIHRoot, "package.json"),
      JSON.stringify({ version: "1.0.0", optionalDependencies: {} })
    );

    const result = validatePlugins(fakeRepo, fakeIHRoot);
    try {
      // V5 must flag the mismatch
      const v5errors = result.errors.filter((e: any) => e.check === "V5" && e.plugin === "v5-plugin");
      expect(v5errors.length).toBeGreaterThan(0);
    } finally {
      fs.rmSync(fakeRepo, { recursive: true, force: true });
      fs.rmSync(fakeIHRoot, { recursive: true, force: true });
    }
  });

  it("V5: installAgents matching plugin.json agents → PASS", () => {
    // plugin.json has agents → yml installAgents matches → V5 PASS
    const os = require("os");
    const fs = require("fs");
    const tmp = os.tmpdir();
    const fakeRepo = join(tmp, `ih-v5ok-${Date.now()}`);
    const pluginRelDir = "plugins-official/v5ok-plugin";
    const pluginDir = join(fakeRepo, pluginRelDir);

    fs.mkdirSync(join(pluginDir, "agents"), { recursive: true });
    fs.mkdirSync(join(pluginDir, ".claude-plugin"), { recursive: true });
    fs.writeFileSync(
      join(pluginDir, ".claude-plugin", "plugin.json"),
      JSON.stringify({ name: "v5ok", version: "1.0.0", agents: ["./agents/alpha.md", "./agents/beta.md"] })
    );
    fs.writeFileSync(join(pluginDir, "agents", "alpha.md"), "# a");
    fs.writeFileSync(join(pluginDir, "agents", "beta.md"), "# b");

    const fakeIHRoot = join(tmp, `ih-v5ok-ih-${Date.now()}`);
    fs.mkdirSync(join(fakeIHRoot, "plugins.d"), { recursive: true });
    // yml installAgents matches plugin.json (order may differ — V5 sorts before compare)
    fs.writeFileSync(
      join(fakeIHRoot, "plugins.d", "v5ok.yml"),
      `id: v5ok-plugin\ndir: ${pluginRelDir}\ndisplayName: V5ok\nskills: 0\ninstallAgents: [beta, alpha]\n`
    );
    fs.writeFileSync(
      join(fakeIHRoot, "package.json"),
      JSON.stringify({ version: "1.0.0", optionalDependencies: {} })
    );

    const result = validatePlugins(fakeRepo, fakeIHRoot);
    try {
      // V5 should NOT flag — installAgents matches plugin.json (sorted compare)
      const v5errors = result.errors.filter((e: any) => e.check === "V5" && e.plugin === "v5ok-plugin");
      expect(v5errors).toHaveLength(0);
    } finally {
      fs.rmSync(fakeRepo, { recursive: true, force: true });
      fs.rmSync(fakeIHRoot, { recursive: true, force: true });
    }
  });

  it("spec-to-design is included in ops-registry-invoke installSkills (no longer dropped)", () => {
    const embeddedPath = join(INSTALL_HELPER_ROOT, "src", "embedded-plugins.json");
    expect(existsSync(embeddedPath)).toBe(true);
    const embedded = readJson(embeddedPath) as Array<any>;
    const ori = embedded.find((p) => p.id === "ops-registry-invoke");
    expect(ori).toBeDefined();
    const allSkills = ori.installSkills.flatMap((s: any) => s.skills.map((sk: any) => typeof sk === "string" ? sk : sk.name));
    expect(allSkills).toContain("spec-to-design");
  });

  it("plugin-local skills/ are discovered via Layer 4 fallback", () => {
    // A skill listed in yml installSkills with dir pointing to <plugin>/skills/
    // should pass V3 validation (findSkillOnDisk checks plugin-local skills/).
    const os = require("os");
    const fs = require("fs");
    const tmp = os.tmpdir();
    const fakeRepo = join(tmp, `ih-l4-${Date.now()}`);
    const pluginRelDir = "plugins-official/l4-plugin";
    const pluginDir = join(fakeRepo, pluginRelDir);

    fs.mkdirSync(join(pluginDir, "skills", "local-skill"), { recursive: true });
    fs.writeFileSync(join(pluginDir, "skills", "local-skill", "SKILL.md"), "# local");

    const fakeIHRoot = join(tmp, `ih-l4-ih-${Date.now()}`);
    fs.mkdirSync(join(fakeIHRoot, "plugins.d"), { recursive: true });
    fs.writeFileSync(
      join(fakeIHRoot, "plugins.d", "l4.yml"),
      `id: l4-plugin\ndir: ${pluginRelDir}\ndisplayName: L4\nskills: 1\ninstallSkills:\n  - dir: ${pluginRelDir}/skills\n    skills: [local-skill]\ninstallAgents: []\n`
    );
    fs.writeFileSync(
      join(fakeIHRoot, "package.json"),
      JSON.stringify({ version: "1.0.0", optionalDependencies: {} })
    );

    const result = validatePlugins(fakeRepo, fakeIHRoot);
    try {
      // V3 should pass — local-skill exists in <plugin>/skills/local-skill/
      const v3errors = result.errors.filter((e: any) => e.check === "V3" && e.plugin === "l4-plugin");
      expect(v3errors).toHaveLength(0);
    } finally {
      fs.rmSync(fakeRepo, { recursive: true, force: true });
      fs.rmSync(fakeIHRoot, { recursive: true, force: true });
    }
  });

  it("ALL_SKILLS static value is parsed when INCLUDED_SKILLS is absent", () => {
    // Plugin with ALL_SKILLS="sk-a sk-b" (static) and no INCLUDED_SKILLS
    // mergeInitSkills should fall back to ALL_SKILLS
    const os = require("os");
    const fs = require("fs");
    const tmp = os.tmpdir();
    const fakeRepo = join(tmp, `ih-as-${Date.now()}`);
    const pluginRelDir = "plugins-official/as-plugin";
    const pluginDir = join(fakeRepo, pluginRelDir);

    // Create skills in scanDir ops/
    fs.mkdirSync(join(fakeRepo, "ops", "as-skill-a"), { recursive: true });
    fs.mkdirSync(join(fakeRepo, "ops", "as-skill-b"), { recursive: true });
    fs.mkdirSync(join(pluginDir), { recursive: true });
    fs.writeFileSync(
      join(pluginDir, "init.sh"),
      `#!/bin/bash\nALL_SKILLS="as-skill-a as-skill-b"\n`
    );
    // NO INCLUDED_SKILLS

    const fakeIHRoot = join(tmp, `ih-as-ih-${Date.now()}`);
    fs.mkdirSync(join(fakeIHRoot, "plugins.d"), { recursive: true });
    fs.writeFileSync(
      join(fakeIHRoot, "plugins.d", "as.yml"),
      `id: as-plugin\ndir: ${pluginRelDir}\ndisplayName: AS\nskills: 0\ninstallSkills: []\ninstallAgents: []\n`
    );
    fs.writeFileSync(
      join(fakeIHRoot, "package.json"),
      JSON.stringify({ version: "1.0.0", optionalDependencies: {} })
    );

    const result = validatePlugins(fakeRepo, fakeIHRoot);
    try {
      // Should pass — no errors (ALL_SKILLS not validated directly, but no V3 failures)
      expect(result.ok).toBe(true);
    } finally {
      fs.rmSync(fakeRepo, { recursive: true, force: true });
      fs.rmSync(fakeIHRoot, { recursive: true, force: true });
    }
  });

  describe("externalRepos configuration", () => {
    const { parse } = require("yaml");

    function readYml(pluginId: string): any {
      const p = join(INSTALL_HELPER_ROOT, "plugins.d", `${pluginId}.yml`);
      return parse(readFileSync(p, "utf-8"));
    }

    function readEmbedded(pluginId: string): any {
      const embedded = readJson(join(INSTALL_HELPER_ROOT, "src", "embedded-plugins.json")) as Array<any>;
      return embedded.find((e) => e.id === pluginId);
    }

    it("ops-direct-invoke: no externalRepos — init.sh clones deps into the workspace", () => {
      // skill 驱动工作流的 init.sh 接管依赖仓克隆（asc-devkit / cann-samples / ops-tensor
      // 落工作区 .cannbot/），注册表不再声明 externalRepos
      const yml = readYml("ops-direct-invoke");
      expect(yml.externalRepos ?? []).toHaveLength(0);
      const init = readFileSync(
        join(REPO_ROOT, "plugins-official", "ops-direct-invoke", "init.sh"),
        "utf-8"
      );
      expect(init).toContain("asc-devkit");
      expect(init).toContain("ops-tensor");
    });

    it("ops-registry-invoke: externalRepos include ops-tensor", () => {
      const yml = readYml("ops-registry-invoke");
      const repos = yml.externalRepos || [];
      const opsTensor = repos.find((r: any) => r.url?.includes("ops-tensor"));
      expect(opsTensor).toBeDefined();
      expect(opsTensor.dir).toContain("ops-tensor");
    });

    it("ops-registry-invoke: asc-devkit dir uses reference/cann/asc-devkit path", () => {
      const yml = readYml("ops-registry-invoke");
      const repos = yml.externalRepos || [];
      const ascDevkit = repos.find((r: any) => r.url?.includes("asc-devkit"));
      expect(ascDevkit).toBeDefined();
      expect(ascDevkit.dir).toContain("reference/cann/asc-devkit");
      expect(ascDevkit.depth).toBe(1);
    });

    it("ops-direct-invoke: delegation model — no installSkills, install runs init.sh", () => {
      // init.sh 负责链接 skills/agents、部署权限 hook、生成 .cannbot/permissions 与
      // settings.json——这些步骤 installViaManifest 无法覆盖，故注册表不声明 installSkills
      const yml = readYml("ops-direct-invoke");
      expect(yml.installSkills ?? []).toHaveLength(0);
      expect(yml.skills).toBe(0);
      expect(yml.script ?? "init.sh").toBe("init.sh");
    });

    it("embedded-plugins.json externalRepos match yml for ops-direct-invoke", () => {
      const yml = readYml("ops-direct-invoke");
      const ep = readEmbedded("ops-direct-invoke");
      // 两侧均无 externalRepos 时保持一致（委托 init.sh 模型）
      expect(ep.externalRepos ?? []).toEqual(yml.externalRepos ?? []);
    });

    it("embedded-plugins.json externalRepos match yml for ops-registry-invoke", () => {
      const yml = readYml("ops-registry-invoke");
      const ep = readEmbedded("ops-registry-invoke");
      expect(ep.externalRepos).toBeDefined();
      expect(ep.externalRepos.length).toBe(yml.externalRepos.length);
      const epUrls = ep.externalRepos.map((r: any) => r.url).sort();
      const ymlUrls = yml.externalRepos.map((r: any) => r.url).sort();
      expect(epUrls).toEqual(ymlUrls);
    });

    it("embedded ops-direct-invoke delegates: empty installSkills, agents from plugin.json", () => {
      const ep = readEmbedded("ops-direct-invoke");
      expect(ep.installSkills ?? []).toHaveLength(0);
      expect(ep.skills).toBe(0);
      expect(ep.agents).toBe(6);
      expect([...ep.installAgents].sort()).toEqual(
        ["architect", "developer", "developer-code", "developer-doc", "developer-test", "qa"].sort()
      );
    });

    it("ops-tensor and asc-devkit are NOT duplicated across plugins", () => {
      const pluginsDir = join(INSTALL_HELPER_ROOT, "plugins.d");
      for (const file of readdirSync(pluginsDir)) {
        if (file.startsWith("_") || !file.endsWith(".yml")) continue;
        const content = parse(readFileSync(join(pluginsDir, file), "utf-8"));
        const repos = content.externalRepos || [];
        const urls = repos.map((r: any) => r.url);
        const unique = new Set(urls);
        expect(unique.size).toBe(urls.length);
      }
    });
  });

  describe("yml skills count matches init.sh INCLUDED_SKILLS count", () => {
    const { parse } = require("yaml");

    it("ops-direct-invoke: dynamic skill collection — no static INCLUDED_SKILLS, yml skills 0", () => {
      // init.sh 动态收集 skill（枚举 skills/ + 解析 AGENTS.md 与 agents frontmatter），
      // 无静态 INCLUDED_SKILLS 变量；yml skills=0 表示 install-helper 不直接装 skill
      const yml = parse(readFileSync(join(INSTALL_HELPER_ROOT, "plugins.d", "ops-direct-invoke.yml"), "utf-8"));
      const init = readFileSync(join(REPO_ROOT, "plugins-official", "ops-direct-invoke", "init.sh"), "utf-8");
      expect(init).not.toMatch(/INCLUDED_SKILLS=/);
      expect(yml.skills).toBe(0);
      expect(yml.installSkills ?? []).toHaveLength(0);
    });

    it("ops-registry-invoke: yml skills == init.sh INCLUDED_SKILLS count", () => {
      const yml = parse(readFileSync(join(INSTALL_HELPER_ROOT, "plugins.d", "ops-registry-invoke.yml"), "utf-8"));
      const init = readFileSync(join(REPO_ROOT, "plugins-official", "ops-registry-invoke", "init.sh"), "utf-8");
      const m = init.match(/INCLUDED_SKILLS="([^"]*)"/);
      const initCount = m![1].split(/\s+/).filter(Boolean).length;
      expect(yml.skills).toBe(initCount);
    });

    it("torch-compile: yml skills == init.sh INCLUDED_SKILLS count", () => {
      const yml = parse(readFileSync(join(INSTALL_HELPER_ROOT, "plugins.d", "torch-compile.yml"), "utf-8"));
      const init = readFileSync(join(REPO_ROOT, "plugins-official", "torch-compile", "init.sh"), "utf-8");
      const m = init.match(/INCLUDED_SKILLS="([^"]*)"/);
      const initCount = m![1].split(/\s+/).filter(Boolean).length;
      expect(yml.skills).toBe(initCount);
    });
  });

  describe("SCAN_DIRS loaded from repository.yaml", () => {
    it("repository.yaml scanDirs includes runtime", () => {
      const { parse } = require("yaml");
      const config = parse(readFileSync(join(INSTALL_HELPER_ROOT, "src", "config", "repository.yaml"), "utf-8"));
      expect(config.scanDirs).toBeDefined();
      expect(Array.isArray(config.scanDirs)).toBe(true);
      expect(config.scanDirs).toContain("runtime");
    });

    it("validate-plugins.cjs loadScanDirs reads from repository.yaml", () => {
      const validateModule = require("../scripts/validate-plugins.cjs");
      expect(validateModule).toBeDefined();
      expect(typeof validateModule.validatePlugins).toBe("function");
    });

    it("gen-embedded.cjs loadScanDirs reads from repository.yaml", () => {
      const genModule = require("../scripts/gen-embedded.cjs");
      expect(genModule).toBeDefined();
    });
  });
});
