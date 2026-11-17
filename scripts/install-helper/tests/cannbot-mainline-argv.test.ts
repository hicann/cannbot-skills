// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdirSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

// Argv-construction contract for the mainline repo URL (PR #1671 review):
// the value comes from an environment variable and must never be parseable
// as a git option, so every git invocation taking it as a positional must
// guard it behind the `--` end-of-options separator.
vi.mock("execa", () => ({
  execa: vi.fn(async () => ({ stdout: "", stderr: "", exitCode: 0 })),
  execaSync: vi.fn(() => ({ stdout: "", stderr: "", exitCode: 0 })),
}));

import { execa } from "execa";

let testDir: string;
let savedEnv: Record<string, string | undefined>;

describe("cannbot mainline URL argv hardening", () => {
  beforeEach(() => {
    testDir = join(tmpdir(), `ih-mainline-argv-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    mkdirSync(testDir, { recursive: true });
    savedEnv = { HOME: process.env.HOME };
    process.env.HOME = testDir;
    for (const key of ["CANNBOT_INSTALLER_PATH", "CANNBOT_MAINLINE_URL"]) {
      savedEnv[key] = process.env[key];
      delete process.env[key];
    }
    vi.mocked(execa).mockClear();
  });

  afterEach(() => {
    rmSync(testDir, { recursive: true, force: true });
    for (const [key, value] of Object.entries(savedEnv)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  });

  it("clone argv guards a '-'-prefixed CANNBOT_MAINLINE_URL behind --", async () => {
    // no seeded cache → the clone branch runs; a value starting with '-'
    // must land after the end-of-options separator, never in option space
    process.env.CANNBOT_MAINLINE_URL = "-opt-injection";
    try {
      const { ensureCannbotRepoCache } = await import("../src/core/cannbot-installer.js");
      const cachePath = await ensureCannbotRepoCache();
      expect(cachePath).toBeTruthy();

      const cloneCall = vi
        .mocked(execa)
        .mock.calls.find((c) => c[0] === "git" && (c[1] as string[])[0] === "clone");
      expect(cloneCall).toBeDefined();
      const argv = cloneCall![1] as string[];
      const urlIndex = argv.indexOf("-opt-injection");
      expect(urlIndex).toBeGreaterThan(0);
      expect(argv[urlIndex - 1]).toBe("--");
    } finally {
      delete process.env.CANNBOT_MAINLINE_URL;
    }
  });

  it("stale-cache refresh re-points origin with the URL behind --", async () => {
    // valid stale cache → the refresh branch runs; the set-url positional
    // gets the same end-of-options guard
    mkdirSync(join(testDir, ".cannbot", "repo-cannbot", "plugins-official"), { recursive: true });
    // stamp 0 → ancient → refresh branch, not the TTL-fresh early return
    const { writeFileSync } = await import("fs");
    writeFileSync(join(testDir, ".cannbot", "repo-cannbot", ".cannbot-repo-stamp"), "0");
    process.env.CANNBOT_MAINLINE_URL = "-opt-injection";
    try {
      const { ensureCannbotRepoCache } = await import("../src/core/cannbot-installer.js");
      const cachePath = await ensureCannbotRepoCache();
      expect(cachePath).toBeTruthy();

      const setUrlCall = vi
        .mocked(execa)
        .mock.calls.find((c) => c[0] === "git" && (c[1] as string[])[0] === "remote");
      expect(setUrlCall).toBeDefined();
      const argv = setUrlCall![1] as string[];
      expect(argv[0]).toBe("remote");
      expect(argv[1]).toBe("set-url");
      expect(argv[2]).toBe("origin");
      const urlIndex = argv.indexOf("-opt-injection");
      expect(urlIndex).toBeGreaterThan(0);
      expect(argv[urlIndex - 1]).toBe("--");
    } finally {
      delete process.env.CANNBOT_MAINLINE_URL;
    }
  });
});
