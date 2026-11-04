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
import { findSkill, getAllSkills, getAllCategories } from "../src/core/skill-registry.js";

describe("skill-registry", () => {
  describe("getAllSkills (static fallback)", () => {
    it("returns a non-empty array", () => {
      const skills = getAllSkills();
      expect(skills.length).toBeGreaterThan(0);
    });

    it("all skills have id, description, source", () => {
      const skills = getAllSkills();
      for (const s of skills) {
        expect(s.id).toBeTruthy();
        expect(typeof s.description).toBe("string");
        expect(s.source).toBeTruthy();
      }
    });

    it("no duplicate ids", () => {
      const skills = getAllSkills();
      const ids = skills.map((s) => s.id);
      const unique = new Set(ids);
      expect(unique.size).toBe(ids.length);
    });
  });

  describe("findSkill", () => {
    it("finds existing skill by id", () => {
      const skill = findSkill("npu-arch");
      expect(skill).toBeDefined();
      expect(skill!.id).toBe("npu-arch");
    });

    it("returns undefined for non-existent", () => {
      const skill = findSkill("non-existent-skill");
      expect(skill).toBeUndefined();
    });
  });

  describe("getAllCategories", () => {
    it("returns non-empty categories", () => {
      const categories = getAllCategories();
      expect(categories.length).toBeGreaterThan(0);
    });

    it("all categories have id, name, skills", () => {
      const categories = getAllCategories();
      for (const cat of categories) {
        expect(cat.id).toBeTruthy();
        expect(cat.name).toBeTruthy();
        expect(Array.isArray(cat.skills)).toBe(true);
        expect(cat.skills.length).toBeGreaterThan(0);
      }
    });

    it("includes runtime_migration in runtime category", () => {
      const categories = getAllCategories();
      const runtimeCat = categories.find((c) => c.id === "runtime");
      expect(runtimeCat).toBeDefined();
      expect(runtimeCat!.skills.some((s) => s.id === "runtime_migration")).toBe(true);
    });

    it("categories mirror repository scanDirs — skills grouped by source directory", () => {
      const categories = getAllCategories();
      // A domain skill must be in the category named after its directory.
      const opsCat = categories.find((c) => c.id === "ops");
      expect(opsCat).toBeDefined();
      expect(opsCat!.skills.every((s) => s.source === "ops")).toBe(true);
      expect(opsCat!.skills.some((s) => s.id === "ascendc-tiling-design")).toBe(true);
      expect(opsCat!.skills.some((s) => s.id === "cannbot-dsl-best-practice")).toBe(true);
      expect(opsCat!.skills.some((s) => s.id === "pypto-op-plan")).toBe(true);

      const commCat = categories.find((c) => c.id === "communication");
      expect(commCat).toBeDefined();
      expect(commCat!.skills.some((s) => s.id === "hccl-aicpu-design")).toBe(true);
    });

    it("skills with a source outside the scanDirs land in the Other catch-all, never dropped", () => {
      const categories = getAllCategories();
      const otherCat = categories.find((c) => c.id === "other");
      // science-model-npu-migration's source is a plugins-community path
      expect(otherCat).toBeDefined();
      expect(otherCat!.skills.some((s) => s.id === "science-model-npu-migration")).toBe(true);
    });

    it("does NOT include ops-easyasc-dsl", () => {
      const skills = getAllSkills();
      expect(skills.find((s) => s.id === "ops-easyasc-dsl")).toBeUndefined();
    });

    it("does NOT include ops-registry-invoke-workflow", () => {
      const skills = getAllSkills();
      expect(skills.find((s) => s.id === "ops-registry-invoke-workflow")).toBeUndefined();
    });
  });

  describe("newly added skills (static fallback)", () => {
    it("includes ascendc-simt-tiling-design", () => {
      const skills = getAllSkills();
      expect(skills.find((s) => s.id === "ascendc-simt-tiling-design")).toBeDefined();
    });

    it("includes cann-env-setup", () => {
      const skills = getAllSkills();
      expect(skills.find((s) => s.id === "cann-env-setup")).toBeDefined();
    });

    it("includes torch-ops-profiler", () => {
      const skills = getAllSkills();
      expect(skills.find((s) => s.id === "torch-ops-profiler")).toBeDefined();
    });

    it("includes ascendc-direct-invoke-to-registry-invoke", () => {
      const skills = getAllSkills();
      expect(skills.find((s) => s.id === "ascendc-direct-invoke-to-registry-invoke")).toBeDefined();
    });

    it("includes ascendc-registry-invoke-to-direct-invoke", () => {
      const skills = getAllSkills();
      expect(skills.find((s) => s.id === "ascendc-registry-invoke-to-direct-invoke")).toBeDefined();
    });
  });
});

describe("subgroup separators (wizard ordering hints)", () => {
  it("getSkillSubgroup returns the semantic subgroup when known", async () => {
    const { getSkillSubgroup } = await import("../src/core/skill-registry.js");
    expect(getSkillSubgroup("cannbot-dsl-best-practice")).toBe("cannbot-dsl");
    expect(getSkillSubgroup("hccl-aicpu-design")).toBe("hccl");
    expect(getSkillSubgroup("ops-easyasc-dsl")).toBeUndefined();
  });
});

describe("getCategorySubgroups (wizard second level)", () => {
  it("aggregates the ops category into semantic subgroups sorted by size", async () => {
    const { getCategorySubgroups, getAllCategories } = await import("../src/core/skill-registry.js");
    const groups = getCategorySubgroups("ops");
    expect(groups.length).toBeGreaterThan(5);
    // sorted by size, descending
    for (let i = 1; i < groups.length; i++) {
      expect(groups[i - 1].skills.length).toBeGreaterThanOrEqual(groups[i].skills.length);
    }
    // every group's skills belong to the ops category
    const opsIds = new Set(getAllCategories().find((c) => c.id === "ops")!.skills.map((s) => s.id));
    for (const g of groups) {
      expect(g.name).toBeTruthy();
      for (const s of g.skills) expect(opsIds.has(s.id)).toBe(true);
    }
    // subgroups partition the category: no skill appears in two groups
    const all = groups.flatMap((g) => g.skills.map((s) => s.id));
    expect(new Set(all).size).toBe(all.length);
  });

  it("single-subgroup categories return exactly one group (no extra level)", async () => {
    const { getCategorySubgroups } = await import("../src/core/skill-registry.js");
    const groups = getCategorySubgroups("runtime");
    expect(groups.length).toBe(1);
    expect(groups[0].skills.length).toBeGreaterThan(0);
  });

  it("unknown category returns empty", async () => {
    const { getCategorySubgroups } = await import("../src/core/skill-registry.js");
    expect(getCategorySubgroups("no-such-dir")).toEqual([]);
  });
});
