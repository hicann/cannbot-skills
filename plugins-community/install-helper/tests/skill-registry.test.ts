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

    it("includes science-model-npu-migration", () => {
      const categories = getAllCategories();
      const modelCat = categories.find((c) => c.id === "model");
      expect(modelCat).toBeDefined();
      expect(modelCat!.skills.some((s) => s.id === "science-model-npu-migration")).toBe(true);
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
});
