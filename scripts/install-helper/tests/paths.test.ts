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
import { validateTool, validateLevel, getConfigRoot, getAgentsFileName, getSkillsRoot } from "../src/utils/paths.js";
import { homedir, tmpdir } from "os";
import { join } from "path";
import { mkdirSync, rmSync } from "fs";

describe("paths", () => {
  describe("validateTool", () => {
    it("accepts valid tools", () => {
      expect(validateTool("opencode")).toBe("opencode");
      expect(validateTool("claude")).toBe("claude");
      expect(validateTool("trae")).toBe("trae");
      expect(validateTool("cursor")).toBe("cursor");
      expect(validateTool("codex")).toBe("codex");
      expect(validateTool("copilot")).toBe("copilot");
      expect(validateTool("codearts")).toBe("codearts");
      expect(validateTool("qoder")).toBe("qoder");
    });

    it("throws for invalid tool", () => {
      expect(() => validateTool("vim")).toThrow();
      expect(() => validateTool("")).toThrow();
    });
  });

  describe("validateLevel", () => {
    it("accepts valid levels", () => {
      expect(validateLevel("project")).toBe("project");
      expect(validateLevel("global")).toBe("global");
    });

    it("throws for invalid level", () => {
      expect(() => validateLevel("local")).toThrow();
      expect(() => validateLevel("")).toThrow();
    });
  });

  describe("getConfigRoot", () => {
    it("returns project-level config root", () => {
      const root = getConfigRoot("opencode", "project", "/tmp");
      expect(root).toContain(".opencode");
    });

    it("returns global-level config root", () => {
      const root = getConfigRoot("opencode", "global");
      expect(root).toContain("opencode");
    });

    it("claude global uses .claude", () => {
      const root = getConfigRoot("claude", "global");
      expect(root).toContain(".claude");
    });

    it("codex project uses .codex under base", () => {
      const root = getConfigRoot("codex", "project", "/tmp/proj");
      expect(root).toBe(join("/tmp/proj", ".codex"));
    });

    it("codex global uses ~/.codex", () => {
      const root = getConfigRoot("codex", "global");
      expect(root).toBe(join(homedir(), ".codex"));
    });
  });

  describe("getSkillsRoot", () => {
    it("codex project uses .agents/skills under base", () => {
      const root = getSkillsRoot("codex", "project", "/tmp/proj");
      expect(root).toBe(join("/tmp/proj", ".agents", "skills"));
    });

    it("codex global uses ~/.agents/skills", () => {
      const root = getSkillsRoot("codex", "global");
      expect(root).toBe(join(homedir(), ".agents", "skills"));
    });

    it("non-codex tools use configRoot/skills", () => {
      expect(getSkillsRoot("opencode", "project", "/tmp/proj")).toBe(join("/tmp/proj", ".opencode", "skills"));
      expect(getSkillsRoot("claude", "global")).toBe(join(homedir(), ".claude", "skills"));
    });
  });

  describe("qoder layout (CANNBOT_QODER_LAYOUT forced)", () => {
    it("cli layout: project config root is .qoder", () => {
      process.env.CANNBOT_QODER_LAYOUT = "cli";
      try {
        expect(getConfigRoot("qoder", "project", "/tmp/proj")).toBe(join("/tmp/proj", ".qoder"));
      } finally {
        delete process.env.CANNBOT_QODER_LAYOUT;
      }
    });

    it("cli layout: project skills root is .qoder/skills, agents root is .qoder/agents", () => {
      process.env.CANNBOT_QODER_LAYOUT = "cli";
      try {
        expect(getSkillsRoot("qoder", "project", "/tmp/proj")).toBe(join("/tmp/proj", ".qoder", "skills"));
        expect(join(getConfigRoot("qoder", "project", "/tmp/proj"), "agents")).toBe(join("/tmp/proj", ".qoder", "agents"));
      } finally {
        delete process.env.CANNBOT_QODER_LAYOUT;
      }
    });

    it("cli layout: global config root is ~/.qoder-cn (QODERCN_CONFIG_DIR honored)", () => {
      process.env.CANNBOT_QODER_LAYOUT = "cli";
      try {
        expect(getConfigRoot("qoder", "global")).toBe(join(homedir(), ".qoder-cn"));
        process.env.QODERCN_CONFIG_DIR = "/tmp/qcfg";
        expect(getConfigRoot("qoder", "global")).toBe("/tmp/qcfg");
      } finally {
        delete process.env.QODERCN_CONFIG_DIR;
        delete process.env.CANNBOT_QODER_LAYOUT;
      }
    });

    it("cli layout: global skills root is ~/.agents/skills (shared with codex)", () => {
      process.env.CANNBOT_QODER_LAYOUT = "cli";
      try {
        expect(getSkillsRoot("qoder", "global")).toBe(join(homedir(), ".agents", "skills"));
      } finally {
        delete process.env.CANNBOT_QODER_LAYOUT;
      }
    });

    it("ide layout: project and global roots are .lingma", () => {
      process.env.CANNBOT_QODER_LAYOUT = "ide";
      try {
        expect(getConfigRoot("qoder", "project", "/tmp/proj")).toBe(join("/tmp/proj", ".lingma"));
        expect(getConfigRoot("qoder", "global")).toBe(join(homedir(), ".lingma"));
        expect(getSkillsRoot("qoder", "global")).toBe(join(homedir(), ".lingma", "skills"));
      } finally {
        delete process.env.CANNBOT_QODER_LAYOUT;
      }
    });
  });

  describe("qoder layout (auto-detect: ~/.qoder-cn without qodercn on PATH)", () => {
    // GUI / non-login shells: qodercn is not on PATH (the which/where probe
    // fails) but the CLI's own config dir exists. vi.resetModules() bypasses
    // the module-level qoderCliLayoutCache so each case re-runs detection.
    let sandbox: string;
    let realHome: string | undefined;
    let realPath: string | undefined;

    beforeEach(() => {
      sandbox = join(tmpdir(), `ih-qoder-detect-${Date.now()}-${Math.random().toString(36).slice(2)}`);
      mkdirSync(sandbox, { recursive: true });
      realHome = process.env.HOME;
      realPath = process.env.PATH;
      process.env.HOME = sandbox;
      process.env.PATH = ""; // which/where qodercn must fail
    });

    afterEach(() => {
      if (realHome === undefined) delete process.env.HOME;
      else process.env.HOME = realHome;
      if (realPath === undefined) delete process.env.PATH;
      else process.env.PATH = realPath;
      delete process.env.CANNBOT_QODER_LAYOUT;
      rmSync(sandbox, { recursive: true, force: true });
    });

    it("~/.qoder-cn exists and qodercn not on PATH → CLI layout", async () => {
      mkdirSync(join(sandbox, ".qoder-cn"));
      vi.resetModules();
      const { isQoderCliLayout, getSkillsRoot } = await import("../src/utils/paths.js");
      expect(isQoderCliLayout()).toBe(true);
      expect(getSkillsRoot("qoder", "project", "/tmp/proj")).toBe(join("/tmp/proj", ".qoder", "skills"));
    });

    it("no ~/.qoder-cn and qodercn not on PATH → IDE layout", async () => {
      vi.resetModules();
      const { isQoderCliLayout, getConfigRoot } = await import("../src/utils/paths.js");
      expect(isQoderCliLayout()).toBe(false);
      expect(getConfigRoot("qoder", "project", "/tmp/proj")).toBe(join("/tmp/proj", ".lingma"));
    });

    it("~/.qoder-cn and ~/.lingma both exist and qodercn not on PATH → CLI layout wins", async () => {
      mkdirSync(join(sandbox, ".qoder-cn"));
      mkdirSync(join(sandbox, ".lingma"));
      vi.resetModules();
      const { isQoderCliLayout, getConfigRoot } = await import("../src/utils/paths.js");
      expect(isQoderCliLayout()).toBe(true);
      expect(getConfigRoot("qoder", "project", "/tmp/proj")).toBe(join("/tmp/proj", ".qoder"));
    });
  });

  describe("getAgentsFileName", () => {
    it("returns CLAUDE.md for claude", () => {
      expect(getAgentsFileName("claude")).toBe("CLAUDE.md");
    });

    it("returns AGENTS.md for others", () => {
      expect(getAgentsFileName("opencode")).toBe("AGENTS.md");
      expect(getAgentsFileName("trae")).toBe("AGENTS.md");
      expect(getAgentsFileName("cursor")).toBe("AGENTS.md");
      expect(getAgentsFileName("copilot")).toBe("AGENTS.md");
      expect(getAgentsFileName("qoder")).toBe("AGENTS.md");
    });
  });
});
