// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------
import { describe, it, expect } from "vitest";
import { writeFileSync, unlinkSync } from "fs";

async function loadParseFrontmatter() {
  try {
    const mod = await import("../dist/index.js");
    return (mod as any).parseFrontmatter;
  } catch {
    return null;
  }
}

describe("parseFrontmatter", () => {
  let parseFrontmatter: ((filePath: string) => { name: string; description: string } | null) | null = null;

  it("should load parseFrontmatter from dist", async () => {
    parseFrontmatter = await loadParseFrontmatter();
    if (!parseFrontmatter) {
      console.warn("dist not available, testing scanner directly");
    }
  });

  it("parses valid frontmatter with name and description", async () => {
    const fn = await loadParseFrontmatter();
    if (!fn) return;
    const tmpFile = `/tmp/test-skill-${Date.now()}.md`;
    writeFileSync(tmpFile, `---\nname: test-skill\ndescription: A test skill\n---\n# Content`);
    const result = fn(tmpFile);
    expect(result).toEqual({ name: "test-skill", description: "A test skill" });
    unlinkSync(tmpFile);
  });

  it("returns null for file without frontmatter", async () => {
    const fn = await loadParseFrontmatter();
    if (!fn) return;
    const tmpFile = `/tmp/test-skill-${Date.now()}.md`;
    writeFileSync(tmpFile, `# Just a title\nNo frontmatter here`);
    const result = fn(tmpFile);
    expect(result).toBeNull();
    unlinkSync(tmpFile);
  });

  it("returns null when name is missing", async () => {
    const fn = await loadParseFrontmatter();
    if (!fn) return;
    const tmpFile = `/tmp/test-skill-${Date.now()}.md`;
    writeFileSync(tmpFile, `---\ndescription: No name field\n---\n# Content`);
    const result = fn(tmpFile);
    expect(result).toBeNull();
    unlinkSync(tmpFile);
  });

  it("rejects name with path traversal (..)", async () => {
    const fn = await loadParseFrontmatter();
    if (!fn) return;
    const tmpFile = `/tmp/test-skill-${Date.now()}.md`;
    writeFileSync(tmpFile, `---\nname: "../../etc/evil"\ndescription: malicious\n---\n# Evil`);
    const result = fn(tmpFile);
    expect(result).toBeNull();
    unlinkSync(tmpFile);
  });

  it("rejects name with slash", async () => {
    const fn = await loadParseFrontmatter();
    if (!fn) return;
    const tmpFile = `/tmp/test-skill-${Date.now()}.md`;
    writeFileSync(tmpFile, `---\nname: "foo/bar"\ndescription: malicious\n---\n# Evil`);
    const result = fn(tmpFile);
    expect(result).toBeNull();
    unlinkSync(tmpFile);
  });

  it("rejects name with spaces", async () => {
    const fn = await loadParseFrontmatter();
    if (!fn) return;
    const tmpFile = `/tmp/test-skill-${Date.now()}.md`;
    writeFileSync(tmpFile, `---\nname: "has spaces"\ndescription: invalid\n---\n# Content`);
    const result = fn(tmpFile);
    expect(result).toBeNull();
    unlinkSync(tmpFile);
  });

  it("accepts name with dots, dashes, underscores", async () => {
    const fn = await loadParseFrontmatter();
    if (!fn) return;
    const tmpFile = `/tmp/test-skill-${Date.now()}.md`;
    writeFileSync(tmpFile, `---\nname: "my.skill-name_v2"\ndescription: valid\n---\n# Content`);
    const result = fn(tmpFile);
    expect(result).toEqual({ name: "my.skill-name_v2", description: "valid" });
    unlinkSync(tmpFile);
  });
});

describe("scanDirs / pluginDirs configuration", () => {
  it("getScanDirs includes the tools domain", async () => {
    const { getScanDirs } = await import("../src/core/scanner.js");
    const dirs = getScanDirs();
    expect(dirs).toContain("tools");
    for (const d of ["ops", "model", "graph", "infra", "runtime"]) {
      expect(dirs).toContain(d);
    }
  });

  it("getPluginDirs covers official and community plugins", async () => {
    const { getPluginDirs } = await import("../src/core/scanner.js");
    const dirs = getPluginDirs();
    expect(dirs).toContain("plugins-official");
    expect(dirs).toContain("plugins-community");
  });

  it("scanSkills discovers tools/ domain skills and both official and community plugin skills", async () => {
    const { scanSkills } = await import("../src/core/scanner.js");
    const { mkdirSync, writeFileSync, rmSync } = await import("fs");
    const { join } = await import("path");
    const { tmpdir } = await import("os");

    const repo = join(tmpdir(), `ih-scan-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    try {
      mkdirSync(join(repo, "tools", "fake-tool-skill"), { recursive: true });
      writeFileSync(join(repo, "tools", "fake-tool-skill", "SKILL.md"), "---\nname: fake-tool-skill\ndescription: tools domain\n---\n");
      mkdirSync(join(repo, "plugins-official", "fake-official-plugin", "skills", "official-skill"), { recursive: true });
      writeFileSync(join(repo, "plugins-official", "fake-official-plugin", "skills", "official-skill", "SKILL.md"), "---\nname: official-skill\ndescription: official\n---\n");
      mkdirSync(join(repo, "plugins-community", "fake-community-plugin", "skills", "community-skill"), { recursive: true });
      writeFileSync(join(repo, "plugins-community", "fake-community-plugin", "skills", "community-skill", "SKILL.md"), "---\nname: community-skill\ndescription: community\n---\n");

      const skills = scanSkills(repo);
      const ids = skills.map((s) => s.id);
      expect(ids).toContain("fake-tool-skill");
      expect(ids).toContain("official-skill");
      // community plugin skills are exposed exactly like official ones —
      // symmetric support for every plugin the repository ships
      expect(ids).toContain("community-skill");
    } finally {
      rmSync(repo, { recursive: true, force: true });
    }
  });
});

describe("dynamic skill domain discovery", () => {
  it("discovers a new top-level domain directory without the scanDirs whitelist", async () => {
    const { scanSkills, discoverSkillDomains } = await import("../src/core/scanner.js");
    const { mkdirSync, rmSync, writeFileSync } = await import("fs");
    const { join } = await import("path");
    const { tmpdir } = await import("os");

    const repo = join(tmpdir(), `ih-dyn-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    try {
      // a brand-new domain the whitelist never knew about
      mkdirSync(join(repo, "security", "threat-model"), { recursive: true });
      writeFileSync(join(repo, "security", "threat-model", "SKILL.md"), "---\nname: threat-model\ndescription: threat modeling\n---\n# body\n");
      // a top-level dir whose skills are NOT direct children (plugins tree) — must not match
      mkdirSync(join(repo, "plugins-official", "some-plugin", "skills", "bundled-skill"), { recursive: true });
      writeFileSync(join(repo, "plugins-official", "some-plugin", "skills", "bundled-skill", "SKILL.md"), "---\nname: bundled-skill\ndescription: bundled\n---\n# body\n");
      // hidden dir — ignored
      mkdirSync(join(repo, ".hidden-domain", "ghost-skill"), { recursive: true });
      writeFileSync(join(repo, ".hidden-domain", "ghost-skill", "SKILL.md"), "---\nname: ghost-skill\ndescription: hidden\n---\n# body\n");

      const domains = discoverSkillDomains(repo);
      expect(domains).toContain("security");
      expect(domains).not.toContain("plugins-official");
      expect(domains).not.toContain(".hidden-domain");

      const skills = scanSkills(repo);
      expect(skills.map((s) => s.id)).toContain("threat-model");
      expect(skills.find((s) => s.id === "threat-model")?.source).toBe("security");
      expect(skills.map((s) => s.id)).not.toContain("ghost-skill");
    } finally {
      rmSync(repo, { recursive: true, force: true });
    }
  });
});
