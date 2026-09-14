// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { describe, it, expect } from "vitest";
import { resolve, sep } from "path";

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
    expect(src).toMatch(/const record = readRecord\(pluginId\);[\s\S]*?displayName: record\.displayName/);
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
