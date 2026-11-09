// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "fs";
import { join, relative } from "path";
import { tmpdir } from "os";

const state = vi.hoisted(() => ({ configDir: "", skills: [] as Array<{ filePath: string }> }));
vi.mock("../src/utils/paths.js", () => ({
  getCannbotConfigDir: () => state.configDir,
  getCannbotRepoPath: () => join(state.configDir, "repo"),
}));
vi.mock("../src/core/skill-registry.js", () => ({
  initFromScan: (skills: Array<{ filePath: string }>) => { state.skills = skills; },
}));
vi.mock("../src/core/registry.js", () => ({ mergeDynamicPlugins: () => {} }));
vi.mock("../src/core/metadata-sync.js", () => ({ enrichPluginMetadata: () => {} }));

import { RepositoryManager } from "../src/core/repository.js";

let root: string;
function createRepo(name: string): string {
  const repo = join(root, name);
  mkdirSync(join(repo, ".git"), { recursive: true });
  mkdirSync(join(repo, "plugins-official"));
  mkdirSync(join(repo, "ops", "sample"), { recursive: true });
  writeFileSync(join(repo, ".git", "HEAD"), "0123456789abcdef0123456789abcdef01234567\n");
  writeFileSync(join(repo, "ops", "sample", "SKILL.md"), "---\nname: sample\ndescription: example\n---\n");
  return repo;
}

describe("scan cache repository identity", () => {
  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), "ih-scan-cache-"));
    state.configDir = join(root, "config");
    state.skills = [];
  });
  afterEach(() => { rmSync(root, { recursive: true, force: true }); });

  it("rescans a different checkout at the same commit and reuses that checkout's cache", async () => {
    const first = createRepo("first");
    const second = createRepo("second");
    await new RepositoryManager(first).ensureRepoAndScan();
    expect(state.skills).toHaveLength(1);
    expect(state.skills[0].filePath).toBe(join(first, "ops", "sample", "SKILL.md"));
    await new RepositoryManager(second).ensureRepoAndScan();
    expect(state.skills[0].filePath).toBe(join(second, "ops", "sample", "SKILL.md"));
    const cache = JSON.parse(readFileSync(join(state.configDir, "scan-cache.json"), "utf8"));
    expect(cache.repoPath).toBe(second);
    // Removing the skill proves the second call uses the existing cache.
    rmSync(join(second, "ops", "sample", "SKILL.md"));
    await new RepositoryManager(second).ensureRepoAndScan();
    expect(state.skills[0].filePath).toBe(join(second, "ops", "sample", "SKILL.md"));
  });

  it("stores absolute paths when the checkout is supplied as a relative path", async () => {
    const repo = createRepo("relative-checkout");
    await new RepositoryManager(relative(process.cwd(), repo)).ensureRepoAndScan();
    expect(state.skills).toHaveLength(1);
    expect(state.skills[0].filePath).toBe(join(repo, "ops", "sample", "SKILL.md"));
    await new RepositoryManager(repo).ensureRepoAndScan();
    expect(state.skills[0].filePath).toBe(join(repo, "ops", "sample", "SKILL.md"));
  });

  it("rescans caches created before the repository path was recorded", async () => {
    const repo = createRepo("repo");
    await new RepositoryManager(repo).ensureRepoAndScan();
    const cacheFile = join(state.configDir, "scan-cache.json");
    const cache = JSON.parse(readFileSync(cacheFile, "utf8"));
    delete cache.repoPath;
    cache.skills[0].filePath = "old-path";
    writeFileSync(cacheFile, JSON.stringify(cache));
    await new RepositoryManager(repo).ensureRepoAndScan();
    expect(state.skills[0].filePath).toBe(join(repo, "ops", "sample", "SKILL.md"));
  });
});
