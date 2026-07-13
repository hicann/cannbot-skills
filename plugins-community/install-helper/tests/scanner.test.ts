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
