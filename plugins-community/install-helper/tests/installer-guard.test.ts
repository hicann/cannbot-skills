// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdirSync, writeFileSync, rmSync, readFileSync, existsSync } from "fs";
import { join, dirname } from "path";
import { tmpdir } from "os";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let testDir: string;

function setup() {
  testDir = join(tmpdir(), `ih-gd-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
}

function teardown() {
  rmSync(testDir, { recursive: true, force: true });
}

describe("installer tool guard", () => {
  beforeEach(setup);
  afterEach(teardown);

  describe("scriptSupportsTool", () => {
    it("detects tool tokens in init.sh arg parsing", async () => {
      const { scriptSupportsTool } = await import("../src/core/installer.js");
      const scriptPath = join(testDir, "init.sh");
      writeFileSync(
        scriptPath,
        `TOOL="opencode"\nfor arg in "$@"; do\n  case "$arg" in\n    opencode|claude|codex) TOOL="$arg" ;;\n  esac\ndone\n`
      );
      expect(scriptSupportsTool(scriptPath, "opencode")).toBe(true);
      expect(scriptSupportsTool(scriptPath, "claude")).toBe(true);
      expect(scriptSupportsTool(scriptPath, "codex")).toBe(true);
      expect(scriptSupportsTool(scriptPath, "codearts")).toBe(false);
    });

    it("does not match codearts inside .codeartsdoer (word boundary)", async () => {
      const { scriptSupportsTool } = await import("../src/core/installer.js");
      const scriptPath = join(testDir, "init.sh");
      writeFileSync(scriptPath, `CONFIG_ROOT="$HOME/.codeartsdoer"\nTOOL="opencode"\n`);
      expect(scriptSupportsTool(scriptPath, "codearts")).toBe(false);
    });

    it("matches codearts when explicitly supported", async () => {
      const { scriptSupportsTool } = await import("../src/core/installer.js");
      const scriptPath = join(testDir, "init.sh");
      writeFileSync(
        scriptPath,
        `case "$arg" in\n  opencode|claude|codearts) TOOL="$arg" ;;\nesac\n`
      );
      expect(scriptSupportsTool(scriptPath, "codearts")).toBe(true);
    });

    it("returns true for unreadable script (defer to execa error)", async () => {
      const { scriptSupportsTool } = await import("../src/core/installer.js");
      expect(scriptSupportsTool(join(testDir, "missing-init.sh"), "codex")).toBe(true);
    });
  });

  describe("guard wiring (source verification)", () => {
    it("installer.ts blocks script-path install when tool is unsupported", async () => {
      const src = readFileSync(join(__dirname, "..", "src", "core", "installer.ts"), "utf-8");
      expect(src).toContain("scriptSupportsTool(scriptPath, opts.tool)");
      expect(src).toContain("error_tool_not_supported");
    });

    it("real community plugin init.sh without codex fails the guard", async () => {
      const { scriptSupportsTool } = await import("../src/core/installer.js");
      const repoRoot = join(__dirname, "..", "..", "..");
      const tritonInit = join(repoRoot, "plugins-community", "triton-optimizer", "init.sh");
      if (!existsSync(tritonInit)) return; // plugin layout changed — nothing to assert
      expect(scriptSupportsTool(tritonInit, "codex")).toBe(false);
      expect(scriptSupportsTool(tritonInit, "opencode")).toBe(true);
    });
  });
});
