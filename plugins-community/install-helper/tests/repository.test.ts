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
import { mkdirSync, writeFileSync, existsSync, rmSync, readFileSync } from "fs";
import { join, dirname } from "path";
import { tmpdir } from "os";
import { fileURLToPath } from "url";
import { execaSync } from "execa";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let testDir: string;

function setup() {
  testDir = join(tmpdir(), `ih-repo-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
}

function teardown() {
  rmSync(testDir, { recursive: true, force: true });
}

describe("repository", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("cloneRepo cleans up corrupted directory before re-cloning (source code verification)", async () => {
    const src = readFileSync(join(__dirname, "..", "src", "core", "repository.ts"), "utf-8");
    expect(src).toContain("rmSync(targetPath");
    expect(src).toContain("recursive: true, force: true");
  });

  it("repo_clone_failed i18n key exists with manual clone hint", async () => {
    const zh = JSON.parse(readFileSync(join(__dirname, "..", "src", "locales", "zh_CN.json"), "utf-8"));
    const en = JSON.parse(readFileSync(join(__dirname, "..", "src", "locales", "en_US.json"), "utf-8"));
    expect(zh.repo_clone_failed).toContain("git clone");
    expect(en.repo_clone_failed).toContain("git clone");
  });

  it("external_repo_update_failed i18n key exists", async () => {
    const zh = JSON.parse(readFileSync(join(__dirname, "..", "src", "locales", "zh_CN.json"), "utf-8"));
    const en = JSON.parse(readFileSync(join(__dirname, "..", "src", "locales", "en_US.json"), "utf-8"));
    expect(zh.external_repo_update_failed).toBeDefined();
    expect(en.external_repo_update_failed).toBeDefined();
  });

  it("updateRepo re-points the managed cache origin at CANNBOT_REPO_URL (firewall after HTTPS cache)", async () => {
    // Same failure class the PR #1671 review found on the mainline cache:
    // the skills-repo cache is cloned once from the URL in effect at clone
    // time; a later CANNBOT_REPO_URL switch (firewall move) must govern the
    // refresh too — and must never rewrite the origin of a user's own
    // checkout. Real local git, no network.
    const git = (cwd: string, args: string[]) =>
      execaSync("git", ["-c", "user.name=test", "-c", "user.email=test@test", ...args], { cwd });

    // skills-a: first source of the managed cache (valid skills repo layout)
    const originA = join(testDir, "skills-a");
    mkdirSync(originA, { recursive: true });
    git(originA, ["init", "--quiet"]);
    mkdirSync(join(originA, "plugins-official", "fa-skill"), { recursive: true });
    writeFileSync(join(originA, "plugins-official", "fa-skill", "SKILL.md"), "---\nname: fa-skill\n---\n");
    git(originA, ["add", "-A"]);
    git(originA, ["commit", "--quiet", "-m", "a"]);

    // skills-b: same history + one newer commit (shares history so pull works)
    const originB = join(testDir, "skills-b");
    git(testDir, ["clone", "--quiet", originA, originB]);
    mkdirSync(join(originB, "plugins-official", "fb-skill"), { recursive: true });
    writeFileSync(join(originB, "plugins-official", "fb-skill", "SKILL.md"), "---\nname: fb-skill\n---\n");
    git(originB, ["add", "-A"]);
    git(originB, ["commit", "--quiet", "-m", "b"]);

    const savedEnv = {
      HOME: process.env.HOME,
      CANNBOT_REPO_PATH: process.env.CANNBOT_REPO_PATH,
      CANNBOT_REPO_URL: process.env.CANNBOT_REPO_URL,
    };
    process.env.HOME = testDir;
    delete process.env.CANNBOT_REPO_PATH;
    try {
      process.env.CANNBOT_REPO_URL = originA;
      const { createRepositoryManager } = await import("../src/core/repository.js");
      await createRepositoryManager().ensureRepoAndScan();
      const cache = join(testDir, ".cannbot", "repo");
      expect(existsSync(join(cache, ".git"))).toBe(true);
      expect(existsSync(join(cache, "plugins-official", "fa-skill"))).toBe(true);

      // firewall move: switch the override and refresh
      process.env.CANNBOT_REPO_URL = originB;
      await createRepositoryManager().ensureRepoAndScan();
      const remoteUrl = execaSync("git", ["remote", "get-url", "origin"], { cwd: cache }).stdout.trim();
      expect(remoteUrl).toBe(originB);
      expect(existsSync(join(cache, "plugins-official", "fb-skill", "SKILL.md"))).toBe(true);
    } finally {
      for (const [key, value] of Object.entries(savedEnv)) {
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
    }
  });
});
