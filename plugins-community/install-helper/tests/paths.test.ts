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
import { validateTool, validateLevel, getConfigRoot, getAgentsFileName, getSkillsRoot } from "../src/utils/paths.js";
import { homedir } from "os";
import { join } from "path";

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

  describe("getAgentsFileName", () => {
    it("returns CLAUDE.md for claude", () => {
      expect(getAgentsFileName("claude")).toBe("CLAUDE.md");
    });

    it("returns AGENTS.md for others", () => {
      expect(getAgentsFileName("opencode")).toBe("AGENTS.md");
      expect(getAgentsFileName("trae")).toBe("AGENTS.md");
      expect(getAgentsFileName("cursor")).toBe("AGENTS.md");
      expect(getAgentsFileName("copilot")).toBe("AGENTS.md");
    });
  });
});
