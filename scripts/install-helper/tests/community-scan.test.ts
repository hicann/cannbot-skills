// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// 嵌套社区域动态发现回归：#1690 将 aiss-tiling-solver 归位至 community/ops/、
// #1680 将 science-model-npu-migration 归位至 community/model/ 后，容器目录
// （community/）之下的一层嵌套域必须被动态领域发现识别——getActiveSkills()
// 在 initFromScan 后只返回动态结果、静态表整体丢弃，域不被发现则技能在
// init/list/install 正常路径下消失。与 #1662 顶层域自发现同一自适精神：
// 仓库新增 community/<域> 无需 install-helper 发版。
import { describe, expect, it, beforeEach, afterEach } from "vitest";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { scanSkills, discoverSkillDomains } from "../src/core/scanner.js";
import { initFromScan, findSkill } from "../src/core/skill-registry.js";

const repositoryRoot = resolve(import.meta.dirname, "../../..");

let fixture: string;

function setup() {
  fixture = join(tmpdir(), `ih-community-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(fixture, { recursive: true });
}

function teardown() {
  rmSync(fixture, { recursive: true, force: true });
}

function skillAt(dir: string, name: string) {
  mkdirSync(join(dir, name), { recursive: true });
  writeFileSync(join(dir, name, "SKILL.md"), `---\nname: ${name}\ndescription: fixture skill\n---\n`);
}

describe("community nested skill domains", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("discovers nested domains inside container directories one level deep", () => {
    skillAt(join(fixture, "ops"), "plain-skill");
    skillAt(join(fixture, "community", "ops"), "c-ops-skill");
    skillAt(join(fixture, "community", "model"), "c-model-skill");
    mkdirSync(join(fixture, "docs", "notes"), { recursive: true });

    const domains = discoverSkillDomains(fixture);
    expect(domains).toContain("ops");
    expect(domains).toContain("community/ops");
    expect(domains).toContain("community/model");
    expect(domains).not.toContain("docs");
    expect(domains).not.toContain("community");
  });

  it("never probes excluded trees (node_modules) for nested domains", () => {
    // vendored packages may carry SKILL.md-like layouts; node_modules stays
    // outside discovery both as a domain and as a container
    skillAt(join(fixture, "node_modules", "some-pkg", "nested"), "vendored-skill");
    expect(discoverSkillDomains(fixture)).toEqual([]);
  });

  it("scanSkills finds skills under community domains with multi-segment sources", () => {
    skillAt(join(fixture, "community", "ops"), "c-ops-skill");
    const skills = scanSkills(fixture);
    const s = skills.find((x) => x.id === "c-ops-skill");
    expect(s).toBeDefined();
    expect(s!.source).toBe("community/ops");
    expect(s!.filePath).toBe(join(fixture, "community", "ops", "c-ops-skill", "SKILL.md"));
  });

  it("keeps the real repository's relocated skills dynamically visible (#1690/#1680 acceptance)", () => {
    expect(existsSync(join(repositoryRoot, "community/ops/aiss-tiling-solver/SKILL.md"))).toBe(true);
    const skills = scanSkills(repositoryRoot);
    initFromScan(skills);

    const aiss = skills.find((s) => s.id === "aiss-tiling-solver");
    expect(aiss).toBeDefined();
    expect(aiss!.source).toBe("community/ops");
    expect(findSkill("aiss-tiling-solver")).toBeDefined();

    expect(skills.find((s) => s.id === "science-model-npu-migration")?.source).toBe("community/model");
    expect(findSkill("science-model-npu-migration")).toBeDefined();
  });
});
