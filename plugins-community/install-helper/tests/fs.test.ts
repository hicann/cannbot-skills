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
import { mkdirSync, writeFileSync, readFileSync, existsSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

let testDir: string;

function setup() {
  testDir = join(tmpdir(), `ih-fs-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
}

function teardown() {
  rmSync(testDir, { recursive: true, force: true });
}

describe("fs (atomic write)", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("writes file atomically", async () => {
    const { atomicWriteFileSync } = await import("../src/utils/fs.js");
    const filePath = join(testDir, "test.json");
    atomicWriteFileSync(filePath, '{"key":"value"}');
    expect(existsSync(filePath)).toBe(true);
    expect(readFileSync(filePath, "utf-8")).toBe('{"key":"value"}');
  });

  it("overwrites existing file", async () => {
    const { atomicWriteFileSync } = await import("../src/utils/fs.js");
    const filePath = join(testDir, "test.json");
    writeFileSync(filePath, "old", "utf-8");
    atomicWriteFileSync(filePath, "new");
    expect(readFileSync(filePath, "utf-8")).toBe("new");
  });

  it("throws when directory does not exist", async () => {
    const { atomicWriteFileSync } = await import("../src/utils/fs.js");
    const filePath = join(testDir, "nonexistent", "test.json");
    expect(() => atomicWriteFileSync(filePath, "data")).toThrow();
  });
});

describe("fs-helpers", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("isSymlink returns true for symlink", async () => {
    const { isSymlink } = await import("../src/utils/fs-helpers.js");
    const target = join(testDir, "target");
    const link = join(testDir, "link");
    writeFileSync(target, "data");
    const { symlinkSync } = await import("fs");
    symlinkSync(target, link);
    expect(isSymlink(link)).toBe(true);
  });

  it("isSymlink returns false for regular file", async () => {
    const { isSymlink } = await import("../src/utils/fs-helpers.js");
    const file = join(testDir, "file.txt");
    writeFileSync(file, "data");
    expect(isSymlink(file)).toBe(false);
  });

  it("isSymlink returns false for non-existent path", async () => {
    const { isSymlink } = await import("../src/utils/fs-helpers.js");
    expect(isSymlink(join(testDir, "nonexistent"))).toBe(false);
  });

  it("isDirectory returns true for directory", async () => {
    const { isDirectory } = await import("../src/utils/fs-helpers.js");
    expect(isDirectory(testDir)).toBe(true);
  });

  it("isDirectory returns false for file", async () => {
    const { isDirectory } = await import("../src/utils/fs-helpers.js");
    const file = join(testDir, "file.txt");
    writeFileSync(file, "data");
    expect(isDirectory(file)).toBe(false);
  });

  it("isDirectory returns false for non-existent path", async () => {
    const { isDirectory } = await import("../src/utils/fs-helpers.js");
    expect(isDirectory(join(testDir, "nonexistent"))).toBe(false);
  });
});
