// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdirSync, writeFileSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

let testDir: string;
let sandboxHome: string;
let savedEnv: Record<string, string | undefined>;

function setup() {
  testDir = join(tmpdir(), `ih-cbt-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
  sandboxHome = join(testDir, "home");
  mkdirSync(join(sandboxHome, ".cannbot"), { recursive: true });
  savedEnv = {
    HOME: process.env.HOME,
    CANNBOT_INSTALLER_PATH: process.env.CANNBOT_INSTALLER_PATH,
  };
  process.env.HOME = sandboxHome;
  delete process.env.CANNBOT_INSTALLER_PATH;
}

function teardown() {
  for (const [key, value] of Object.entries(savedEnv)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  rmSync(testDir, { recursive: true, force: true });
}

/** Minimal installer package root with an explicit bin/cannbot.js body. */
function writeInstallerRoot(
  name: string,
  binSource: string,
  packageJson?: Record<string, unknown>
): string {
  const root = join(testDir, name);
  mkdirSync(join(root, "bin"), { recursive: true });
  writeFileSync(
    join(root, "package.json"),
    JSON.stringify({ name: "@cannbot-plugin/cannbot", version: "9.9.9", ...packageJson }, null, 2)
  );
  writeFileSync(join(root, "bin", "cannbot.js"), binSource);
  process.env.CANNBOT_INSTALLER_PATH = root;
  return root;
}

// @cannbot-plugin/cannbot >= 1.10.0 declaration: names + layouts in a frozen
// object literal.
const DIRECTORIES_BIN = `#!/usr/bin/env node
const TOOL_DIRECTORIES = Object.freeze({
  opencode: { configDirs: [".opencode"], defaultConfigDir: ".opencode", skillsDir: [".agents", "skills"] },
  codex: { configDirs: [".codex"], defaultConfigDir: ".codex", skillsDir: [".agents", "skills"] },
  claude: { configDirs: [".claude"], defaultConfigDir: ".claude" },
  trae: { configDirs: [".traecli", ".marscode", ".trae", ".trae-cn"], defaultConfigDir: ".trae" },
  dsh: { configDirs: [".dsh"], defaultConfigDir: ".dsh" },
  codearts: { configDirs: [".codeartsdoer"], defaultConfigDir: ".codeartsdoer" },
});
process.exit(0);
`;

// @cannbot-plugin/cannbot <= 1.9.x declaration: names only.
const NAMES_BIN = `#!/usr/bin/env node
const TOOL_NAMES = ['opencode', 'codex', 'claude', 'trae', 'dsh'];
process.exit(0);
`;

// No recognizable declaration at all (e.g. a future upstream refactor).
const UNPARSEABLE_BIN = `#!/usr/bin/env node
const capabilities = computeSupportedClients();
process.exit(0);
`;

describe("cannbot-tools", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("derives tools and layouts from the installer's TOOL_DIRECTORIES declaration", async () => {
    writeInstallerRoot("dirs-installer", DIRECTORIES_BIN);
    const { deriveCannbotTools } = await import("../src/core/cannbot-tools.js");

    const derived = deriveCannbotTools();
    expect(derived.source).toBe("directories");
    expect(derived.tools).toHaveLength(6);
    expect(derived.tools).toContain("codearts");
    expect(derived.layouts?.codearts).toEqual({
      configDirs: [".codeartsdoer"],
      defaultConfigDir: ".codeartsdoer",
    });
    expect(derived.layouts?.trae?.configDirs).toEqual([".traecli", ".marscode", ".trae", ".trae-cn"]);
  });

  it("delegateBoundaryTools returns the derived list when the declaration is parseable", async () => {
    writeInstallerRoot("boundary-installer", DIRECTORIES_BIN);
    const { delegateBoundaryTools } = await import("../src/core/cannbot-tools.js");

    // the boundary set IS the installer's declaration — codearts/dsh flow
    // straight through, no hardcoded list in between
    expect(delegateBoundaryTools()).toEqual([
      "opencode",
      "codex",
      "claude",
      "trae",
      "dsh",
      "codearts",
    ]);
  });

  it("derives tool names only from the legacy TOOL_NAMES declaration", async () => {
    writeInstallerRoot("names-installer", NAMES_BIN);
    const { deriveCannbotTools } = await import("../src/core/cannbot-tools.js");

    const derived = deriveCannbotTools();
    expect(derived.source).toBe("names");
    expect(derived.tools).toEqual(["opencode", "codex", "claude", "trae", "dsh"]);
    expect(derived.tools).not.toContain("codearts");
    expect(derived.layouts).toBeNull();
  });

  it("prefers a package.json contract over source parsing", async () => {
    writeInstallerRoot("contract-installer", DIRECTORIES_BIN, {
      cannbot: {
        toolDirectories: {
          qoder: { configDirs: [".qoder"], defaultConfigDir: ".qoder" },
        },
      },
    });
    const { deriveCannbotTools } = await import("../src/core/cannbot-tools.js");

    const derived = deriveCannbotTools();
    expect(derived.source).toBe("contract");
    expect(derived.tools).toEqual(["qoder"]);
    expect(derived.layouts?.qoder).toEqual({ configDirs: [".qoder"], defaultConfigDir: ".qoder" });
  });

  it("ignores a malformed contract and falls through to source parsing", async () => {
    writeInstallerRoot("bad-contract-installer", DIRECTORIES_BIN, {
      cannbot: { toolDirectories: "not-an-object" },
    });
    const { deriveCannbotTools } = await import("../src/core/cannbot-tools.js");

    const derived = deriveCannbotTools();
    expect(derived.source).toBe("directories");
    expect(derived.tools).toContain("codearts");
  });

  it("falls back to the legacy prior when the installer is absent", async () => {
    const { deriveCannbotTools, delegateBoundaryTools, LEGACY_DELEGATE_TOOLS } = await import(
      "../src/core/cannbot-tools.js"
    );

    const derived = deriveCannbotTools();
    expect(derived.source).toBe("absent");
    expect(derived.tools).toBeNull();
    expect(delegateBoundaryTools()).toEqual(LEGACY_DELEGATE_TOOLS);
  });

  it("falls back to the legacy prior when the declaration is unparseable", async () => {
    writeInstallerRoot("unparseable-installer", UNPARSEABLE_BIN);
    const { deriveCannbotTools, delegateBoundaryTools, LEGACY_DELEGATE_TOOLS } = await import(
      "../src/core/cannbot-tools.js"
    );

    const derived = deriveCannbotTools();
    expect(derived.source).toBe("unparseable");
    expect(derived.tools).toBeNull();
    expect(delegateBoundaryTools()).toEqual(LEGACY_DELEGATE_TOOLS);
    expect(LEGACY_DELEGATE_TOOLS).toContain("opencode");
    expect(LEGACY_DELEGATE_TOOLS).not.toContain("codearts");
  });

  it("tolerates an unfrozen literal with trailing commas and extra keys", async () => {
    const bin = `#!/usr/bin/env node
const TOOL_DIRECTORIES = {
  opencode: { configDirs: [".opencode"], defaultConfigDir: ".opencode", skillsDir: [".agents", "skills"], },
  codearts: { configDirs: [".codeartsdoer"], defaultConfigDir: ".codeartsdoer", extras: true },
};
process.exit(0);
`;
    writeInstallerRoot("tolerant-installer", bin);
    const { deriveCannbotTools } = await import("../src/core/cannbot-tools.js");

    const derived = deriveCannbotTools();
    expect(derived.source).toBe("directories");
    expect(derived.tools).toEqual(["opencode", "codearts"]);
    expect(derived.layouts?.codearts?.defaultConfigDir).toBe(".codeartsdoer");
  });
});
