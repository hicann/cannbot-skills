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
import { mkdirSync, mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

const mocks = vi.hoisted(() => ({ pull: vi.fn() }));
vi.mock("execa", () => ({ execa: mocks.pull }));
vi.mock("../src/utils/paths.js", () => ({
  getCannbotConfigDir: () => process.env.TEST_REPO_ROOT!,
  getCannbotRepoPath: () => process.env.TEST_REPO_CACHE!,
}));
vi.mock("../src/core/scanner.js", () => ({
  scanSkills: () => [], getCurrentCommit: () => "fixture", readScanCache: () => null, writeScanCache: () => {},
}));
vi.mock("../src/core/skill-registry.js", () => ({ initFromScan: () => {} }));
vi.mock("../src/core/registry.js", () => ({ mergeDynamicPlugins: () => {} }));
vi.mock("../src/core/metadata-sync.js", () => ({ enrichPluginMetadata: () => {} }));

import { RepositoryManager } from "../src/core/repository.js";

describe("repository auto-update source", () => {
  let root: string;
  let cwd: string;
  let cache: string;
  const oldEnv = process.env.CANNBOT_REPO_PATH;

  function makeRepo(dir: string) {
    mkdirSync(join(dir, ".git"), { recursive: true });
    mkdirSync(join(dir, "plugins-official"));
  }

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), "ih-repo-source-"));
    cwd = join(root, "cwd");
    cache = join(root, "cache");
    mkdirSync(cwd);
    makeRepo(cache);
    process.env.TEST_REPO_ROOT = root;
    process.env.TEST_REPO_CACHE = cache;
    delete process.env.CANNBOT_REPO_PATH;
    vi.spyOn(process, "cwd").mockReturnValue(cwd);
    mocks.pull.mockReset();
    mocks.pull.mockResolvedValue({});
  });

  afterEach(() => {
    vi.restoreAllMocks();
    if (oldEnv === undefined) delete process.env.CANNBOT_REPO_PATH;
    else process.env.CANNBOT_REPO_PATH = oldEnv;
    delete process.env.TEST_REPO_ROOT;
    delete process.env.TEST_REPO_CACHE;
    rmSync(root, { recursive: true, force: true });
  });

  it("does not pull a checkout discovered in the current directory", async () => {
    makeRepo(cwd);
    await new RepositoryManager().ensureRepoAndScan();
    expect(mocks.pull).not.toHaveBeenCalled();
  });

  it("does not auto-pull a valid user-specified checkout", async () => {
    const other = join(root, "other");
    makeRepo(cwd);
    makeRepo(other);
    const result = await new RepositoryManager(other).ensureRepoAndScan();
    expect(result).toBe(other);
    expect(mocks.pull).not.toHaveBeenCalled();
  });

  it("updates a cwd checkout when explicitly requested by the update command", async () => {
    makeRepo(cwd);
    await new RepositoryManager().ensureRepoAndScan({ updateCwd: true });
    expect(mocks.pull).toHaveBeenCalledTimes(1);
    expect(mocks.pull).toHaveBeenCalledWith("git", ["pull", "--quiet"], { cwd, timeout: 30000 });
  });

  it("does not pull twice when an update command selects the managed cache", async () => {
    await new RepositoryManager().ensureRepoAndScan({ updateCwd: true });
    // the managed-cache refresh re-points origin first (CANNBOT_REPO_URL must
    // govern refreshes, not only the first clone) — that set-url is a
    // distinct command; exactly one pull must still happen
    const pullCalls = mocks.pull.mock.calls.filter(
      (args: unknown[]) => (args[1] as string[])?.[0] === "pull"
    );
    expect(pullCalls).toHaveLength(1);
  });

  it("does not auto-pull a valid environment-specified checkout", async () => {
    const other = join(root, "other");
    makeRepo(other);
    process.env.CANNBOT_REPO_PATH = other;
    await new RepositoryManager().ensureRepoAndScan();
    expect(mocks.pull).not.toHaveBeenCalled();
  });

  it("updates the managed cache even if a supplied path is invalid", async () => {
    process.env.CANNBOT_REPO_PATH = join(root, "missing");
    await new RepositoryManager().ensureRepoAndScan();
    expect(mocks.pull).toHaveBeenCalledWith("git", ["pull", "--quiet"], { cwd: cache, timeout: 30000 });
  });

  it("updates the managed cache when a constructor path is invalid", async () => {
    await new RepositoryManager(join(root, "missing")).ensureRepoAndScan();
    expect(mocks.pull).toHaveBeenCalledWith("git", ["pull", "--quiet"], { cwd: cache, timeout: 30000 });
  });

  it("still permits an explicit update of a user checkout", async () => {
    makeRepo(cwd);
    const manager = new RepositoryManager(cwd);
    await manager.ensureRepoAndScan();
    await manager.updateRepo();
    expect(mocks.pull).toHaveBeenCalledTimes(1);
    expect(mocks.pull).toHaveBeenCalledWith("git", ["pull", "--quiet"], { cwd, timeout: 30000 });
  });
});
