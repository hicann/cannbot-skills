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
});
