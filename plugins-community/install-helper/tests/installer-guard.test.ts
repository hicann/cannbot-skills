// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdirSync, writeFileSync, rmSync, readFileSync, existsSync, readdirSync, lstatSync } from "fs";
import { join, dirname } from "path";
import { tmpdir } from "os";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

vi.mock("../src/ui/backup-prompts.js", () => ({
  showOverwriteWarning: vi.fn(async () => "overwrite" as const),
  showUnownedFileWarning: vi.fn(async () => "keep" as const),
  showRestorePrompt: vi.fn(async () => "none" as const),
}));

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

describe("unowned config file protection (project AGENTS.md overwrite)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });
  beforeEach(setup);
  afterEach(teardown);

  const PLUGIN_ID = "zz-unowned-guard";

  function makeFixture() {
    const repoDir = join(testDir, "repo");
    const projectDir = join(testDir, "project");
    const pluginDir = join(repoDir, PLUGIN_ID);
    mkdirSync(join(pluginDir, "skills", "demo-skill"), { recursive: true });
    writeFileSync(join(pluginDir, "skills", "demo-skill", "SKILL.md"), "---\nname: demo-skill\n---\n");
    writeFileSync(join(pluginDir, "AGENTS.md"), "# Plugin agents");
    mkdirSync(join(projectDir, ".opencode"), { recursive: true });
    writeFileSync(join(projectDir, "AGENTS.md"), "# User config");
    return { repoDir, projectDir, pluginDir };
  }

  function pluginEntry(dir: string) {
    return {
      id: PLUGIN_ID,
      dir,
      displayName: "ZZ Unowned Guard",
      script: "init.sh",
      aliases: [],
      skills: 1,
      agents: 0,
      description: "",
      configFile: "AGENTS.md",
      installSkills: [{ dir: `${dir}/skills`, skills: ["demo-skill"] }],
      installAgents: [],
    };
  }

  it("prompts before replacing an unowned project AGENTS.md; cancel leaves it untouched", async () => {
    const { installPlugin } = await import("../src/core/installer.js");
    const { showUnownedFileWarning } = await import("../src/ui/backup-prompts.js");
    const { repoDir, projectDir } = makeFixture();
    vi.mocked(showUnownedFileWarning).mockResolvedValueOnce("cancel");

    const result = await installPlugin({
      pluginId: PLUGIN_ID,
      tool: "opencode",
      level: "project",
      repoPath: repoDir,
      installPath: projectDir,
      plugin: pluginEntry(PLUGIN_ID) as any,
    });

    expect(result.success).toBe(false);
    expect(showUnownedFileWarning).toHaveBeenCalled();
    expect(readFileSync(join(projectDir, "AGENTS.md"), "utf-8")).toBe("# User config");
    expect(existsSync(join(projectDir, ".opencode", "skills", "demo-skill"))).toBe(false);
  });

  it("replace choice backs up beside the file and records the original path", async () => {
    const { installPlugin } = await import("../src/core/installer.js");
    const { showUnownedFileWarning } = await import("../src/ui/backup-prompts.js");
    const { readRecord, deleteRecord } = await import("../src/core/record.js");
    const { repoDir, projectDir } = makeFixture();
    vi.mocked(showUnownedFileWarning).mockResolvedValueOnce("replace");

    try {
      const result = await installPlugin({
        pluginId: PLUGIN_ID,
        tool: "opencode",
        level: "project",
        repoPath: repoDir,
        installPath: projectDir,
        plugin: pluginEntry(PLUGIN_ID) as any,
      });

      expect(result.success).toBe(true);
      // project-root file replaced by symlink to the plugin source
      expect(lstatSync(join(projectDir, "AGENTS.md")).isSymbolicLink()).toBe(true);
      expect(readFileSync(join(projectDir, "AGENTS.md"), "utf-8")).toBe("# Plugin agents");
      // backup preserved beside the replaced file
      const backupFiles = readdirSync(projectDir).filter((f) => f.startsWith("AGENTS.md.cannbot-backup.unowned."));
      expect(backupFiles.length).toBe(1);
      expect(readFileSync(join(projectDir, backupFiles[0]), "utf-8")).toBe("# User config");
      // record carries the original path for uninstall restore
      const record = readRecord(PLUGIN_ID);
      expect(record).not.toBeNull();
      expect(record!.backups?.length).toBe(1);
      expect(record!.backups![0].originalPath).toBe(join(projectDir, "AGENTS.md"));
      expect(record!.backups![0].fromPluginId).toBe("unowned");
    } finally {
      deleteRecord(PLUGIN_ID);
    }
  });

  it("--yes preserves an unowned file without prompting (no explicit confirmation)", async () => {
    const { installPlugin } = await import("../src/core/installer.js");
    const { showUnownedFileWarning } = await import("../src/ui/backup-prompts.js");
    const { deleteRecord } = await import("../src/core/record.js");
    const { repoDir, projectDir } = makeFixture();

    try {
      const result = await installPlugin({
        pluginId: PLUGIN_ID,
        tool: "opencode",
        level: "project",
        repoPath: repoDir,
        installPath: projectDir,
        yes: true,
        plugin: pluginEntry(PLUGIN_ID) as any,
      });

      expect(result.success).toBe(true);
      expect(showUnownedFileWarning).not.toHaveBeenCalled();
      expect(readFileSync(join(projectDir, "AGENTS.md"), "utf-8")).toBe("# User config");
      expect(lstatSync(join(projectDir, "AGENTS.md")).isSymbolicLink()).toBe(false);
      // the rest of the plugin still installs
      expect(existsSync(join(projectDir, ".opencode", "skills", "demo-skill"))).toBe(true);
      expect(existsSync(join(projectDir, ".opencode", "AGENTS.md"))).toBe(true);
    } finally {
      deleteRecord(PLUGIN_ID);
    }
  });

  it("fresh install (no existing config files) does not prompt", async () => {
    const { installPlugin } = await import("../src/core/installer.js");
    const { showUnownedFileWarning, showOverwriteWarning } = await import("../src/ui/backup-prompts.js");
    const { deleteRecord } = await import("../src/core/record.js");
    const { repoDir, projectDir } = makeFixture();
    rmSync(join(projectDir, "AGENTS.md"), { force: true });

    try {
      const result = await installPlugin({
        pluginId: PLUGIN_ID,
        tool: "opencode",
        level: "project",
        repoPath: repoDir,
        installPath: projectDir,
        plugin: pluginEntry(PLUGIN_ID) as any,
      });

      expect(result.success).toBe(true);
      expect(showUnownedFileWarning).not.toHaveBeenCalled();
      expect(showOverwriteWarning).not.toHaveBeenCalled();
      expect(lstatSync(join(projectDir, "AGENTS.md")).isSymbolicLink()).toBe(true);
    } finally {
      deleteRecord(PLUGIN_ID);
    }
  });

  it("keep choice continues installing skills while skipping the config file", async () => {
    const { installPlugin } = await import("../src/core/installer.js");
    const { showUnownedFileWarning } = await import("../src/ui/backup-prompts.js");
    const { deleteRecord } = await import("../src/core/record.js");
    const { repoDir, projectDir } = makeFixture();
    vi.mocked(showUnownedFileWarning).mockResolvedValueOnce("keep");

    try {
      const result = await installPlugin({
        pluginId: PLUGIN_ID,
        tool: "opencode",
        level: "project",
        repoPath: repoDir,
        installPath: projectDir,
        plugin: pluginEntry(PLUGIN_ID) as any,
      });

      expect(result.success).toBe(true);
      expect(readFileSync(join(projectDir, "AGENTS.md"), "utf-8")).toBe("# User config");
      expect(lstatSync(join(projectDir, "AGENTS.md")).isSymbolicLink()).toBe(false);
      expect(existsSync(join(projectDir, ".opencode", "skills", "demo-skill"))).toBe(true);
      expect(existsSync(join(projectDir, ".opencode", "AGENTS.md"))).toBe(true);
    } finally {
      deleteRecord(PLUGIN_ID);
    }
  });
});

describe("preserved file re-protection on reinstall (review finding)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });
  beforeEach(setup);
  afterEach(teardown);

  // Must be a registry-resolvable id: the misclassification happens when
  // detectCurrentPlugin resolves the leftover manifest of the same plugin
  // through the plugin registry.
  const PLUGIN_ID = "ops-direct-invoke-flash";

  function makeFixture() {
    const repoDir = join(testDir, "repo");
    const projectDir = join(testDir, "project");
    const pluginDir = join(repoDir, PLUGIN_ID);
    mkdirSync(join(pluginDir, "skills", "demo-skill"), { recursive: true });
    writeFileSync(join(pluginDir, "skills", "demo-skill", "SKILL.md"), "---\nname: demo-skill\n---\n");
    writeFileSync(join(pluginDir, "AGENTS.md"), "# Plugin agents");
    mkdirSync(join(projectDir, ".opencode"), { recursive: true });
    writeFileSync(join(projectDir, "AGENTS.md"), "# User config");
    return { repoDir, projectDir };
  }

  function pluginEntry() {
    return {
      id: PLUGIN_ID,
      dir: PLUGIN_ID,
      displayName: "ZZ Reinstall Guard",
      script: "init.sh",
      aliases: [],
      skills: 1,
      agents: 0,
      description: "",
      configFile: "AGENTS.md",
      installSkills: [{ dir: `${PLUGIN_ID}/skills`, skills: ["demo-skill"] }],
      installAgents: [],
    };
  }

  function installOpts(repoDir: string, projectDir: string, yes?: boolean) {
    return {
      pluginId: PLUGIN_ID,
      tool: "opencode" as const,
      level: "project" as const,
      repoPath: repoDir,
      installPath: projectDir,
      yes,
      plugin: pluginEntry() as any,
    };
  }

  // Only this test's location may be removed — the same registry plugin can
  // carry records for other locations (real user projects).
  async function cleanupRecord(projectDir: string) {
    const { deleteRecord } = await import("../src/core/record.js");
    deleteRecord(PLUGIN_ID, { tool: "opencode", level: "project", installPath: projectDir });
  }

  it("reinstalling the same plugin re-prompts instead of silently replacing the preserved file", async () => {
    const { installPlugin } = await import("../src/core/installer.js");
    const { showUnownedFileWarning } = await import("../src/ui/backup-prompts.js");
    const { readRecord } = await import("../src/core/record.js");
    const { repoDir, projectDir } = makeFixture();

    try {
      // install #1: user keeps the file (mock default = keep)
      const r1 = await installPlugin(installOpts(repoDir, projectDir));
      expect(r1.success).toBe(true);
      expect(readRecord(PLUGIN_ID)?.preservedTargets).toContain(join(projectDir, "AGENTS.md"));

      // reinstall: the preserved file must be re-confirmed, never silently replaced
      const r2 = await installPlugin(installOpts(repoDir, projectDir));
      expect(r2.success).toBe(true);
      expect(showUnownedFileWarning).toHaveBeenCalledTimes(2);
      expect(readFileSync(join(projectDir, "AGENTS.md"), "utf-8")).toBe("# User config");
      expect(lstatSync(join(projectDir, "AGENTS.md")).isSymbolicLink()).toBe(false);
      // record still marks the file as preserved and never owns it
      const record2 = readRecord(PLUGIN_ID);
      expect(record2?.preservedTargets).toContain(join(projectDir, "AGENTS.md"));
      expect(record2?.files).not.toContain(join(projectDir, "AGENTS.md"));
    } finally {
      await cleanupRecord(projectDir);
    }
  });

  it("--yes reinstall keeps a previously preserved file without prompting", async () => {
    const { installPlugin } = await import("../src/core/installer.js");
    const { showUnownedFileWarning } = await import("../src/ui/backup-prompts.js");
    const { readRecord } = await import("../src/core/record.js");
    const { repoDir, projectDir } = makeFixture();

    try {
      const r1 = await installPlugin(installOpts(repoDir, projectDir));
      expect(r1.success).toBe(true);

      const r2 = await installPlugin(installOpts(repoDir, projectDir, true));
      expect(r2.success).toBe(true);
      expect(showUnownedFileWarning).toHaveBeenCalledTimes(1); // only install #1
      expect(readFileSync(join(projectDir, "AGENTS.md"), "utf-8")).toBe("# User config");
      expect(lstatSync(join(projectDir, "AGENTS.md")).isSymbolicLink()).toBe(false);
      expect(readRecord(PLUGIN_ID)?.preservedTargets).toContain(join(projectDir, "AGENTS.md"));
    } finally {
      await cleanupRecord(projectDir);
    }
  });
});
