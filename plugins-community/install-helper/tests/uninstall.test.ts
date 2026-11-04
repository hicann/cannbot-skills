// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { resolve, sep } from "path";
import { mkdirSync, rmSync } from "fs";
import { tmpdir } from "os";

vi.mock("../src/ui/backup-prompts.js", () => ({
  showOverwriteWarning: vi.fn(async () => "overwrite" as const),
  showUnownedFileWarning: vi.fn(async () => "replace" as const),
  showRestorePrompt: vi.fn(async () => "none" as const),
}));

describe("isSafePath (C2 fix verification)", () => {
  function createIsSafePath(allowedBases: string[]) {
    return (p: string): boolean => {
      const resolved = resolve(p);
      return allowedBases.some(base => resolved === base || resolved.startsWith(base + sep));
    };
  }

  it("allows paths inside configRoot", () => {
    const configRoot = resolve("/home/user/.opencode");
    const isSafePath = createIsSafePath([configRoot]);
    expect(isSafePath(join(configRoot, "skills", "my-skill"))).toBe(true);
    expect(isSafePath(join(configRoot, "agents", "agent.md"))).toBe(true);
    expect(isSafePath(join(configRoot, "cannbot-manifest.json"))).toBe(true);
  });

  it("allows paths inside installPath", () => {
    const installPath = resolve("/home/user/project");
    const isSafePath = createIsSafePath([installPath]);
    expect(isSafePath(join(installPath, "AGENTS.md"))).toBe(true);
    expect(isSafePath(join(installPath, "asc-devkit"))).toBe(true);
  });

  it("rejects paths outside allowed bases", () => {
    const configRoot = resolve("/home/user/.opencode");
    const installPath = resolve("/home/user/project");
    const isSafePath = createIsSafePath([configRoot, installPath]);
    expect(isSafePath("/etc/passwd")).toBe(false);
    expect(isSafePath("/tmp/evil")).toBe(false);
    expect(isSafePath("/home/user/.bashrc")).toBe(false);
  });

  it("rejects path traversal attempts", () => {
    const configRoot = resolve("/home/user/.opencode");
    const isSafePath = createIsSafePath([configRoot]);
    expect(isSafePath("/home/user/.opencode/../../../etc/passwd")).toBe(false);
  });

  it("uses platform separator (C2 fix)", () => {
    const configRoot = resolve("/home/user/.opencode");
    const isSafePath = createIsSafePath([configRoot]);
    expect(isSafePath(join(configRoot, "skills", "test"))).toBe(true);
  });

  it("allows exact base path match", () => {
    const configRoot = resolve("/home/user/.opencode");
    const isSafePath = createIsSafePath([configRoot]);
    expect(isSafePath(configRoot)).toBe(true);
  });
});

describe("codex uninstall safety (source verification)", () => {
  it("uninstall.ts allows the codex skills root outside the .codex config root", async () => {
    const { readFileSync } = await import("fs");
    const { join, dirname } = await import("path");
    const { fileURLToPath } = await import("url");
    const __dirname = dirname(fileURLToPath(import.meta.url));
    const src = readFileSync(join(__dirname, "..", "src", "commands", "uninstall.ts"), "utf-8");
    expect(src).toContain('record.tool === "codex"');
    expect(src).toContain("getSkillsRoot(record.tool, record.level, record.installPath)");
  });

  it("interactive uninstall iterates all registered tools (no hardcoded list)", async () => {
    const { readFileSync } = await import("fs");
    const { join, dirname } = await import("path");
    const { fileURLToPath } = await import("url");
    const __dirname = dirname(fileURLToPath(import.meta.url));
    const src = readFileSync(join(__dirname, "..", "src", "commands", "uninstall.ts"), "utf-8");
    expect(src).toContain("for (const toolName of VALID_TOOLS)");
    expect(src).not.toContain('["opencode", "claude", "trae", "cursor", "copilot", "codearts"]');
  });

  it("falls back to install record for community plugins not in registry", async () => {
    const { readFileSync } = await import("fs");
    const { join, dirname } = await import("path");
    const { fileURLToPath } = await import("url");
    const __dirname = dirname(fileURLToPath(import.meta.url));
    const src = readFileSync(join(__dirname, "..", "src", "commands", "uninstall.ts"), "utf-8");
    expect(src).toContain("if (readRecord(name))");
    // #797: the record fallback honors the requested tool when --tool is given.
    expect(src).toMatch(/const record = tool \? readRecords\(pluginId\)\.find\(\(r\) => r\.tool === tool\) \?\? null : readRecord\(pluginId\);[\s\S]*?displayName: record\.displayName/);
  });
});

describe("uninstall command options", () => {
  it("cli accepts --all flag", async () => {
    const { createCLI } = await import("../src/cli.js");
    const program = createCLI();
    const uninstallCmd = program.commands.find((c: any) => c.name() === "uninstall");
    expect(uninstallCmd).toBeDefined();
    const allOption = uninstallCmd?.options.find((o: any) => o.long === "--all");
    expect(allOption).toBeDefined();
  });

  it("cli accepts --recent flag", async () => {
    const { createCLI } = await import("../src/cli.js");
    const program = createCLI();
    const uninstallCmd = program.commands.find((c: any) => c.name() === "uninstall");
    expect(uninstallCmd).toBeDefined();
    const recentOption = uninstallCmd?.options.find((o: any) => o.long === "--recent");
    expect(recentOption).toBeDefined();
  });

  it("cli accepts --yes flag", async () => {
    const { createCLI } = await import("../src/cli.js");
    const program = createCLI();
    const uninstallCmd = program.commands.find((c: any) => c.name() === "uninstall");
    expect(uninstallCmd).toBeDefined();
    const yesOption = uninstallCmd?.options.find((o: any) => o.long === "--yes");
    expect(yesOption).toBeDefined();
  });

  it("cli uninstall names are optional", async () => {
    const { createCLI } = await import("../src/cli.js");
    const program = createCLI();
    const uninstallCmd = program.commands.find((c: any) => c.name() === "uninstall") as any;
    expect(uninstallCmd).toBeDefined();
    expect(uninstallCmd._name).toBe("uninstall");
    const nameArg = uninstallCmd._args?.[0];
    expect(nameArg?.required).toBeFalsy();
  });
});

function join(...parts: string[]): string {
  return parts.join(sep);
}

describe("skill uninstall by name via install record", () => {
  // Seed pattern: append unique installPath keys instead of whole-file
  // backup/restore — record.test.ts clobbers skills.json in a parallel file,
  // and restore-writes would race with it (clobbering its state).
  // Both files now sandbox HOME per test (below), so each worker owns its
  // own skills.json; the unique-key seeding is kept as defense in depth.
  let sandboxDir: string;
  let savedHome: string | undefined;

  beforeEach(() => {
    savedHome = process.env.HOME;
    sandboxDir = join(tmpdir(), `ih-un-rec-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    mkdirSync(sandboxDir, { recursive: true });
    process.env.HOME = sandboxDir;
  });

  afterEach(() => {
    process.env.HOME = savedHome;
    rmSync(sandboxDir, { recursive: true, force: true });
  });

  it("isRecordedSkill detects skills across tool/level/path records", async () => {
    const { isRecordedSkill } = await import("../src/commands/uninstall.js");
    const { addSkillsToRecord, removeSkillsFromRecord } = await import("../src/core/record.js");

    const keyA = `/tmp/ih-rec-skill-${Date.now()}-a`;
    const keyB = `/tmp/ih-rec-skill-${Date.now()}-b`;
    try {
      addSkillsToRecord(["zz-codex-project-skill"], "codex", "project", keyA);
      addSkillsToRecord(["zz-claude-global-skill"], "claude", "global", keyB);

      expect(isRecordedSkill("zz-codex-project-skill")).toBe(true);
      expect(isRecordedSkill("zz-claude-global-skill")).toBe(true);
      expect(isRecordedSkill("never-installed-skill")).toBe(false);
    } finally {
      removeSkillsFromRecord(["zz-codex-project-skill"], "codex", "project", keyA);
      removeSkillsFromRecord(["zz-claude-global-skill"], "claude", "global", keyB);
    }
  });

  it("plugin install record alone does not classify a name as skill", async () => {
    const { isRecordedSkill } = await import("../src/commands/uninstall.js");
    const { writeRecord, deleteRecord } = await import("../src/core/record.js");

    writeRecord({
      pluginId: "zz-plugin-only",
      displayName: "ZZ",
      tool: "opencode",
      level: "project",
      installPath: "/tmp/ih-plugin-only",
      configRoot: "/tmp/ih-plugin-only/.opencode",
      installTime: "2026-01-01T00:00:00.000Z",
      files: [],
      directories: [],
    });
    try {
      expect(isRecordedSkill("zz-plugin-only")).toBe(false);
    } finally {
      deleteRecord("zz-plugin-only");
    }
  });

  it("uninstallCommand removes a skill that is missing from the static list", async () => {
    const { uninstallCommand } = await import("../src/commands/uninstall.js");
    const { addSkillsToRecord, readSkillRecord, removeSkillsFromRecord } = await import("../src/core/record.js");
    const { mkdirSync, writeFileSync, existsSync, rmSync } = await import("fs");
    const { join } = await import("path");
    const { tmpdir } = await import("os");

    const W = join(tmpdir(), `ih-unskill-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    mkdirSync(join(W, ".opencode", "skills", "zz-static-missing-skill"), { recursive: true });
    writeFileSync(join(W, ".opencode", "skills", "zz-static-missing-skill", "SKILL.md"), "---\nname: zz-static-missing-skill\n---\n");

    const origCwd = process.cwd();
    process.chdir(W);
    try {
      addSkillsToRecord(["zz-static-missing-skill"], "opencode", "project", W);
      expect(existsSync(join(W, ".opencode", "skills", "zz-static-missing-skill"))).toBe(true);

      await uninstallCommand(["zz-static-missing-skill"], { tool: "opencode", level: "project", yes: true });

      expect(existsSync(join(W, ".opencode", "skills", "zz-static-missing-skill"))).toBe(false);
      expect(readSkillRecord().opencode?.project?.[W]?.skills ?? []).not.toContain("zz-static-missing-skill");
    } finally {
      process.chdir(origCwd);
      removeSkillsFromRecord(["zz-static-missing-skill"], "opencode", "project", W);
      rmSync(W, { recursive: true, force: true });
    }
  });

  it("uninstallCommand removes tools-domain skills by name (original repro: asys-toolkit + codex)", async () => {
    const { uninstallCommand } = await import("../src/commands/uninstall.js");
    const { addSkillsToRecord, readSkillRecord, removeSkillsFromRecord } = await import("../src/core/record.js");
    const { mkdirSync, writeFileSync, existsSync, rmSync } = await import("fs");
    const { join } = await import("path");
    const { tmpdir } = await import("os");

    const W = join(tmpdir(), `ih-untools-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    mkdirSync(join(W, ".agents", "skills", "asys-toolkit"), { recursive: true });
    writeFileSync(join(W, ".agents", "skills", "asys-toolkit", "SKILL.md"), "---\nname: asys-toolkit\n---\n");

    const origCwd = process.cwd();
    process.chdir(W);
    try {
      addSkillsToRecord(["asys-toolkit"], "codex", "project", W);

      await uninstallCommand(["asys-toolkit"], { tool: "codex", level: "project", yes: true });

      expect(existsSync(join(W, ".agents", "skills", "asys-toolkit"))).toBe(false);
      expect(existsSync(join(W, ".agents"))).toBe(false);
      expect(readSkillRecord().codex?.project?.[W]?.skills ?? []).not.toContain("asys-toolkit");
    } finally {
      process.chdir(origCwd);
      removeSkillsFromRecord(["asys-toolkit"], "codex", "project", W);
      rmSync(W, { recursive: true, force: true });
    }
  });

  it("uninstallCommand handles mixed static-listed and record-only names in one call", async () => {
    const { uninstallCommand } = await import("../src/commands/uninstall.js");
    const { addSkillsToRecord, removeSkillsFromRecord } = await import("../src/core/record.js");
    const { mkdirSync, writeFileSync, existsSync, rmSync } = await import("fs");
    const { join } = await import("path");
    const { tmpdir } = await import("os");

    const W = join(tmpdir(), `ih-unmix-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    for (const s of ["npu-arch", "msnpureport-toolkit"]) {
      mkdirSync(join(W, ".opencode", "skills", s), { recursive: true });
      writeFileSync(join(W, ".opencode", "skills", s, "SKILL.md"), `---\nname: ${s}\n---\n`);
    }

    const origCwd = process.cwd();
    process.chdir(W);
    try {
      addSkillsToRecord(["npu-arch", "msnpureport-toolkit"], "opencode", "project", W);

      await uninstallCommand(["npu-arch", "msnpureport-toolkit"], { tool: "opencode", level: "project", yes: true });

      expect(existsSync(join(W, ".opencode", "skills", "npu-arch"))).toBe(false);
      expect(existsSync(join(W, ".opencode", "skills", "msnpureport-toolkit"))).toBe(false);
    } finally {
      process.chdir(origCwd);
      removeSkillsFromRecord(["npu-arch", "msnpureport-toolkit"], "opencode", "project", W);
      rmSync(W, { recursive: true, force: true });
    }
  });
});

describe("unowned AGENTS.md backup restore on uninstall", () => {
  // Same HOME sandbox as the record describe above: installPlugin /
  // uninstallCommand write records and config.yaml under
  // getCannbotConfigDir() → HOME; keep the real ~/.cannbot untouched.
  let sandboxDir: string;
  let savedHome: string | undefined;

  beforeEach(() => {
    savedHome = process.env.HOME;
    sandboxDir = join(tmpdir(), `ih-un-bak-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    mkdirSync(sandboxDir, { recursive: true });
    process.env.HOME = sandboxDir;
  });

  afterEach(() => {
    process.env.HOME = savedHome;
    rmSync(sandboxDir, { recursive: true, force: true });
  });

  it("restores a replaced project-root AGENTS.md to its original location", async () => {
    const { installPlugin } = await import("../src/core/installer.js");
    const { showUnownedFileWarning, showRestorePrompt } = await import("../src/ui/backup-prompts.js");
    const { findBackups } = await import("../src/core/backup.js");
    const { deleteRecord } = await import("../src/core/record.js");
    const { uninstallCommand } = await import("../src/commands/uninstall.js");
    const { mkdirSync, writeFileSync, existsSync, rmSync, readFileSync } = await import("fs");
    const { join } = await import("path");
    const { tmpdir } = await import("os");

    const PLUGIN_ID = "zz-restore-plugin";
    const W = join(tmpdir(), `ih-restore-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    const repoDir = join(W, "repo");
    const projectDir = join(W, "project");
    const pluginDir = join(repoDir, PLUGIN_ID);
    mkdirSync(join(pluginDir, "skills", "demo-skill"), { recursive: true });
    writeFileSync(join(pluginDir, "skills", "demo-skill", "SKILL.md"), "---\nname: demo-skill\n---\n");
    writeFileSync(join(pluginDir, "AGENTS.md"), "# Plugin agents");
    mkdirSync(join(projectDir, ".opencode"), { recursive: true });
    writeFileSync(join(projectDir, "AGENTS.md"), "# User config");

    const origCwd = process.cwd();
    try {
      vi.mocked(showUnownedFileWarning).mockResolvedValueOnce("replace");
      const installResult = await installPlugin({
        pluginId: PLUGIN_ID,
        tool: "opencode",
        level: "project",
        repoPath: repoDir,
        installPath: projectDir,
        plugin: {
          id: PLUGIN_ID,
          dir: PLUGIN_ID,
          displayName: "ZZ Restore Plugin",
          script: "init.sh",
          aliases: [],
          skills: 1,
          agents: 0,
          description: "",
          configFile: "AGENTS.md",
          installSkills: [{ dir: `${PLUGIN_ID}/skills`, skills: ["demo-skill"] }],
          installAgents: [],
        } as any,
      });
      expect(installResult.success).toBe(true);
      // replaced by the plugin's config via symlink
      expect(readFileSync(join(projectDir, "AGENTS.md"), "utf-8")).toBe("# Plugin agents");

      const unownedBackup = findBackups([projectDir, join(projectDir, ".opencode")])
        .find((b) => b.pluginId === "unowned");
      expect(unownedBackup).toBeDefined();
      expect(unownedBackup!.originalPath).toBe(join(projectDir, "AGENTS.md"));

      process.chdir(projectDir);
      vi.mocked(showRestorePrompt).mockResolvedValueOnce(unownedBackup!.filePath);
      await uninstallCommand([PLUGIN_ID], { tool: "opencode", level: "project", yes: false });

      // the user's original content is back at the original location
      expect(readFileSync(join(projectDir, "AGENTS.md"), "utf-8")).toBe("# User config");
      // backup consumed after restore; plugin files removed
      expect(existsSync(unownedBackup!.filePath)).toBe(false);
      expect(existsSync(join(projectDir, ".opencode", "skills", "demo-skill"))).toBe(false);
    } finally {
      process.chdir(origCwd);
      deleteRecord(PLUGIN_ID);
      rmSync(W, { recursive: true, force: true });
    }
  });

  it("uninstall never removes a preserved (user-kept) AGENTS.md", async () => {
    const { installPlugin } = await import("../src/core/installer.js");
    const { showUnownedFileWarning } = await import("../src/ui/backup-prompts.js");
    const { deleteRecord } = await import("../src/core/record.js");
    const { uninstallCommand } = await import("../src/commands/uninstall.js");
    const { mkdirSync, writeFileSync, existsSync, rmSync, readFileSync, lstatSync } = await import("fs");
    const { join } = await import("path");
    const { tmpdir } = await import("os");

    const PLUGIN_ID = "zz-preserved-uninstall";
    const W = join(tmpdir(), `ih-preserved-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    const repoDir = join(W, "repo");
    const projectDir = join(W, "project");
    const pluginDir = join(repoDir, PLUGIN_ID);
    mkdirSync(join(pluginDir, "skills", "demo-skill"), { recursive: true });
    writeFileSync(join(pluginDir, "skills", "demo-skill", "SKILL.md"), "---\nname: demo-skill\n---\n");
    writeFileSync(join(pluginDir, "AGENTS.md"), "# Plugin agents");
    mkdirSync(join(projectDir, ".opencode"), { recursive: true });
    writeFileSync(join(projectDir, "AGENTS.md"), "# User config");

    const origCwd = process.cwd();
    try {
      vi.mocked(showUnownedFileWarning).mockResolvedValueOnce("keep");
      const installResult = await installPlugin({
        pluginId: PLUGIN_ID,
        tool: "opencode",
        level: "project",
        repoPath: repoDir,
        installPath: projectDir,
        plugin: {
          id: PLUGIN_ID,
          dir: PLUGIN_ID,
          displayName: "ZZ Preserved Uninstall",
          script: "init.sh",
          aliases: [],
          skills: 1,
          agents: 0,
          description: "",
          configFile: "AGENTS.md",
          installSkills: [{ dir: `${PLUGIN_ID}/skills`, skills: ["demo-skill"] }],
          installAgents: [],
        } as any,
      });
      expect(installResult.success).toBe(true);

      process.chdir(projectDir);
      await uninstallCommand([PLUGIN_ID], { tool: "opencode", level: "project", yes: false });

      // the user-kept file survives the uninstall untouched
      expect(readFileSync(join(projectDir, "AGENTS.md"), "utf-8")).toBe("# User config");
      expect(lstatSync(join(projectDir, "AGENTS.md")).isSymbolicLink()).toBe(false);
      expect(existsSync(join(projectDir, ".opencode", "skills", "demo-skill"))).toBe(false);
      expect(existsSync(join(projectDir, ".opencode", "AGENTS.md"))).toBe(false);
    } finally {
      process.chdir(origCwd);
      deleteRecord(PLUGIN_ID);
      rmSync(W, { recursive: true, force: true });
    }
  });
});

describe("uninstall removal log shows disambiguated paths (display guard)", () => {
  // Two same-named files at different roots (project-root AGENTS.md vs
  // config-root AGENTS.md) previously logged as two identical
  // "移除: AGENTS.md" lines. The display fix renders installPath/configRoot
  // relative paths; guard that the two lines stay distinguishable.
  let savedHome: string | undefined;
  let sandboxDir: string;
  let projectDir: string;

  beforeEach(() => {
    const { mkdirSync, writeFileSync } = require("fs") as typeof import("fs");
    const { join } = require("path") as typeof import("path");
    const { tmpdir } = require("os") as typeof import("os");
    savedHome = process.env.HOME;
    sandboxDir = join(tmpdir(), `ih-un-disp-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    projectDir = join(sandboxDir, "project");
    const configRoot = join(projectDir, ".opencode");
    mkdirSync(join(configRoot, "skills"), { recursive: true });
    writeFileSync(join(projectDir, "AGENTS.md"), "# user config\n");
    writeFileSync(join(configRoot, "AGENTS.md"), "# plugin config\n");
    process.env.HOME = sandboxDir;
  });

  afterEach(() => {
    process.env.HOME = savedHome;
    const { rmSync } = require("fs") as typeof import("fs");
    rmSync(sandboxDir, { recursive: true, force: true });
  });

  it("logs project-root and config-root AGENTS.md as distinct relative paths", async () => {
    const { join } = require("path") as typeof import("path");
    const { writeRecord } = await import("../src/core/record.js");
    const { uninstallCommand } = await import("../src/commands/uninstall.js");
    const configRoot = join(projectDir, ".opencode");

    writeRecord({
      pluginId: "torch-compile",
      displayName: "Torch Compile",
      tool: "opencode",
      level: "project",
      installPath: projectDir,
      configRoot,
      installTime: new Date().toISOString(),
      files: [join(projectDir, "AGENTS.md"), join(configRoot, "AGENTS.md")],
      directories: [join(configRoot, "skills")],
    });

    const lines_out: string[] = [];
    const spy = vi.spyOn(console, "log").mockImplementation((...args: unknown[]) => {
      lines_out.push(args.map(String).join(" "));
    });
    try {
      await uninstallCommand(["torch-compile"], { tool: "opencode", level: "project", yes: true });
    } finally {
      spy.mockRestore();
    }

    const removeLines = lines_out.filter((l) => l.includes("AGENTS.md") && l.includes("移除"));
    expect(removeLines.length).toBe(2);
    expect(removeLines.some((l) => l.includes(".opencode/AGENTS.md") || l.includes(".opencode" + require("path").sep + "AGENTS.md"))).toBe(true);
    expect(removeLines.some((l) => !l.includes(".opencode"))).toBe(true);
  });
});

describe("uninstall --tool isolation (#797)", () => {
  let savedHome: string | undefined;
  let projectDir: string;
  const vkey = process.env.CANNBOT_AUTH_VKEY;

  beforeEach(() => {
    const { mkdirSync, writeFileSync } = require("fs") as typeof import("fs");
    const { join } = require("path") as typeof import("path");
    const { tmpdir } = require("os") as typeof import("os");
    savedHome = process.env.HOME;
    projectDir = join(tmpdir(), `ih-un-tool-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    mkdirSync(join(projectDir, ".opencode", "skills"), { recursive: true });
    mkdirSync(join(projectDir, ".claude", "skills"), { recursive: true });
    for (const root of [".opencode", ".claude"]) {
      writeFileSync(join(projectDir, root, "AGENTS.md"), `# ${root} config\n`);
      for (const s of ["sk-a", "sk-b"]) {
        const d = join(projectDir, root, "skills", s);
        mkdirSync(d, { recursive: true });
        writeFileSync(join(d, "SKILL.md"), `---\nname: ${s}\n---\n`);
      }
    }
    process.env.HOME = join(projectDir, "..", "ih-un-tool-home-" + Math.random().toString(36).slice(2));
  });

  afterEach(() => {
    process.env.HOME = savedHome;
    const { rmSync } = require("fs") as typeof import("fs");
    rmSync(projectDir, { recursive: true, force: true });
  });

  it("uninstall --tool opencode removes only the opencode location of a same-name dual-tool install", async () => {
    const { join } = require("path") as typeof import("path");
    const { existsSync } = require("fs") as typeof import("fs");
    const { writeRecord } = await import("../src/core/record.js");
    const { uninstallCommand } = await import("../src/commands/uninstall.js");

    // Same-name plugin installed for opencode AND claude at the same
    // installPath — claude installed later (its record wins readRecord's
    // byLatest preference, reproducing the #797 removal of the wrong side).
    writeRecord({
      pluginId: "torch-compile",
      displayName: "torch.compile 图模式",
      tool: "opencode",
      level: "project",
      installPath: projectDir,
      configRoot: join(projectDir, ".opencode"),
      installTime: "2026-01-01T00:00:00.000Z",
      files: [join(projectDir, ".opencode", "AGENTS.md")],
      directories: [join(projectDir, ".opencode", "skills")],
    });
    writeRecord({
      pluginId: "torch-compile",
      displayName: "torch.compile 图模式",
      tool: "claude",
      level: "project",
      installPath: projectDir,
      configRoot: join(projectDir, ".claude"),
      installTime: "2026-01-02T00:00:00.000Z",
      files: [join(projectDir, ".claude", "AGENTS.md")],
      directories: [join(projectDir, ".claude", "skills")],
    });

    const origCwd = process.cwd();
    process.chdir(projectDir);
    try {
      await uninstallCommand(["torch-compile"], { tool: "opencode", level: "project", yes: true });
    } finally {
      process.chdir(origCwd);
    }

    // opencode side removed
    expect(existsSync(join(projectDir, ".opencode", "AGENTS.md"))).toBe(false);
    // claude side MUST survive (#797)
    expect(existsSync(join(projectDir, ".claude", "AGENTS.md"))).toBe(true);
    expect(existsSync(join(projectDir, ".claude", "skills", "sk-a"))).toBe(true);

    // and the surviving record is claude's
    const { readRecords } = await import("../src/core/record.js");
    const remaining = readRecords("torch-compile");
    expect(remaining.map((r) => r.tool)).toEqual(["claude"]);
  });

  it("without --tool the legacy cwd-preference resolution is preserved", async () => {
    const { join } = require("path") as typeof import("path");
    const { existsSync } = require("fs") as typeof import("fs");
    const { writeRecord } = await import("../src/core/record.js");
    const { uninstallCommand } = await import("../src/commands/uninstall.js");

    writeRecord({
      pluginId: "torch-compile",
      displayName: "torch.compile 图模式",
      tool: "opencode",
      level: "project",
      installPath: projectDir,
      configRoot: join(projectDir, ".opencode"),
      installTime: "2026-01-01T00:00:00.000Z",
      files: [join(projectDir, ".opencode", "AGENTS.md")],
      directories: [join(projectDir, ".opencode", "skills")],
    });

    const origCwd = process.cwd();
    process.chdir(projectDir);
    try {
      await uninstallCommand(["torch-compile"], { yes: true });
    } finally {
      process.chdir(origCwd);
    }

    // no --tool: the latest matching record (opencode) is removed as before
    expect(existsSync(join(projectDir, ".opencode", "AGENTS.md"))).toBe(false);
  });
});
