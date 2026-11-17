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
import { join, dirname } from "path";
import { tmpdir } from "os";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

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

  it("removePath removes a regular file", async () => {
    const { removePath } = await import("../src/utils/fs-helpers.js");
    const filePath = join(testDir, "to-remove.txt");
    writeFileSync(filePath, "data");
    expect(existsSync(filePath)).toBe(true);
    removePath(filePath);
    expect(existsSync(filePath)).toBe(false);
  });

  it("removePath removes a directory recursively", async () => {
    const { removePath } = await import("../src/utils/fs-helpers.js");
    const dirPath = join(testDir, "to-remove-dir");
    mkdirSync(join(dirPath, "sub"), { recursive: true });
    writeFileSync(join(dirPath, "file.txt"), "data");
    removePath(dirPath);
    expect(existsSync(dirPath)).toBe(false);
  });

  it("removePath does not throw for non-existent path", async () => {
    const { removePath } = await import("../src/utils/fs-helpers.js");
    expect(() => removePath(join(testDir, "nonexistent"))).not.toThrow();
  });

  it("atomicWriteFileSync overwrites existing file without unlink (POSIX rename)", async () => {
    const { atomicWriteFileSync } = await import("../src/utils/fs.js");
    const filePath = join(testDir, "atomic.txt");
    writeFileSync(filePath, "original");
    expect(existsSync(filePath)).toBe(true);

    atomicWriteFileSync(filePath, "updated");
    const content = readFileSync(filePath, "utf-8");
    expect(content).toBe("updated");
    expect(existsSync(`${filePath}.tmp`)).toBe(false);
  });

  it("atomicWriteFileSync uses rename directly with tmp cleanup on failure", async () => {
    const src = readFileSync(join(__dirname, "..", "src", "utils", "fs.ts"), "utf-8");
    expect(src).toContain("renameSync(tmpPath, filePath)");
    expect(src).toContain("unlinkSync(tmpPath)");
  });
});
