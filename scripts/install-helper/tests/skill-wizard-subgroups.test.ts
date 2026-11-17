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
import { mkdirSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

/**
 * Unit tests for the skill-install wizard's subgroup level, mocking
 * @inquirer/prompts. Real-PTY coverage lives in interactive-install.test.ts
 * (single-subgroup skip, F); these tests assert the call sequence, the exact
 * choice lists rendered at each level, and the returned selection.
 */

// vi.mock factories hoist above top-level consts — create the fns with
// vi.hoisted so they exist when the factory runs.
const { select, checkbox } = vi.hoisted(() => ({
  select: vi.fn(),
  checkbox: vi.fn(),
}));
vi.mock("@inquirer/prompts", () => ({
  select: (...args: unknown[]) => select(...args),
  checkbox: (...args: unknown[]) => checkbox(...args),
  Separator: class Separator {
    constructor(public separator: string) {}
  },
}));

let testDir: string;
let savedHome: string | undefined;

function setup() {
  testDir = join(tmpdir(), `ih-skillwiz-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
  savedHome = process.env.HOME;
  process.env.HOME = testDir;
  select.mockReset();
  checkbox.mockReset();
}
function teardown() {
  rmSync(testDir, { recursive: true, force: true });
  process.env.HOME = savedHome;
}

/** Reads the renderable choice names of a select/checkbox invocation.
 * vitest mock calls store the arguments array directly: calls[i][0] is the
 * first argument (the options object with `choices`). */
function choiceNames(call: unknown[]): string[] {
  const opts = (call?.[0] ?? {}) as { choices?: Array<{ name: string } | { separator: string }> };
  return (opts.choices ?? []).map((c) =>
    "separator" in c ? `| ${c.separator} |` : c.name
  );
}

describe("interactiveSkillSelect — subgroup level", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("ops renders a subgroup level between category and checkbox; picking a subgroup narrows the checkbox", async () => {
    const { interactiveSkillSelect } = await import("../src/core/skill-installer.js");

    // Level 1: category — pick ops
    select.mockImplementationOnce(async () => "ops");
    // Level 2: subgroup — pick the TileLang subgroup
    select.mockImplementationOnce(async () => "tilelang");
    // Level 3: checkbox — pick one skill
    checkbox.mockImplementationOnce(async () => ["tilelang-op-design"]);
    // Post-selection: install now
    select.mockImplementationOnce(async () => "install");

    const result = await interactiveSkillSelect("opencode", "project");
    expect(result).toEqual(["tilelang-op-design"]);

    // Call sequence: category select → subgroup select → checkbox → install select
    expect(select).toHaveBeenCalledTimes(3);
    expect(checkbox).toHaveBeenCalledTimes(1);

    // Category frame: the 7 repository domains + all + back/cancel
    const catNames = choiceNames(select.mock.calls[0] as unknown[]);
    expect(catNames.some((n) => n.includes("ops — 算子开发 ("))).toBe(true);
    expect(catNames.some((n) => n.includes("model — 模型推理 ("))).toBe(true);
    expect(catNames.some((n) => n.includes("communication — 通信与 HCCL ("))).toBe(true);

    // Subgroup frame: ">> 全部 (95)" first, then the 10 subgroups by size
    const subNames = choiceNames(select.mock.calls[1] as unknown[]);
    const firstOption = subNames.find((n) => n.startsWith(">> "));
    expect(firstOption).toContain("个");
    expect(subNames.some((n) => n.includes("AscendC 开发 ("))).toBe(true);
    expect(subNames.some((n) => n.includes("PyPTO 开发 ("))).toBe(true);
    expect(subNames.some((n) => n.includes("调试与诊断 ("))).toBe(true);
    expect(subNames.some((n) => n.includes("TileLang 开发 ("))).toBe(true);

    // Checkbox frame: only the TileLang subgroup's skills
    const boxNames = choiceNames(checkbox.mock.calls[0] as unknown[]);
    expect(boxNames.some((n) => n.includes("tilelang-op-design"))).toBe(true);
    expect(boxNames.some((n) => n.includes("ascendc-tiling-design"))).toBe(false);
    expect(boxNames.some((n) => n.includes("cannbot-dsl-best-practice"))).toBe(false);
  });

  it("subgroup '>> 全部' opens the full ops checkbox (all subgroups' skills listed)", async () => {
    const { interactiveSkillSelect } = await import("../src/core/skill-installer.js");

    select.mockImplementationOnce(async () => "ops"); // category
    select.mockImplementationOnce(async () => "__select_all__"); // subgroup → all
    checkbox.mockImplementationOnce(async () => ["ascendc-tiling-design"]);
    select.mockImplementationOnce(async () => "install");

    const result = await interactiveSkillSelect("opencode", "project");
    expect(result).toEqual(["ascendc-tiling-design"]);

    // The checkbox lists skills from multiple subgroups
    const boxNames = choiceNames(checkbox.mock.calls[0] as unknown[]);
    expect(boxNames.some((n) => n.includes("ascendc-tiling-design"))).toBe(true);
    expect(boxNames.some((n) => n.includes("pypto-op-plan"))).toBe(true);
    expect(boxNames.some((n) => n.includes("cannbot-dsl-best-practice"))).toBe(true);
  });

  it("single-subgroup domains skip the subgroup level entirely (category → checkbox)", async () => {
    const { interactiveSkillSelect } = await import("../src/core/skill-installer.js");

    select.mockImplementationOnce(async () => "runtime"); // category → checkbox directly
    checkbox.mockImplementationOnce(async () => ["runtime_migration"]);
    select.mockImplementationOnce(async () => "install");

    const result = await interactiveSkillSelect("opencode", "project");
    expect(result).toEqual(["runtime_migration"]);

    // Exactly one select (category) before the checkbox, then the install
    // select — no subgroup level.
    expect(select).toHaveBeenCalledTimes(2);
    expect(checkbox).toHaveBeenCalledTimes(1);
    const boxNames = choiceNames(checkbox.mock.calls[0] as unknown[]);
    expect(boxNames.some((n) => n.includes("runtime_migration"))).toBe(true);
  });

  it("back from the subgroup level returns to the category level", async () => {
    const { interactiveSkillSelect } = await import("../src/core/skill-installer.js");

    select.mockImplementationOnce(async () => "ops"); // category
    select.mockImplementationOnce(async () => "__back__"); // subgroup → back to category
    select.mockImplementationOnce(async () => "__cancel__"); // category → cancel

    const result = await interactiveSkillSelect("opencode", "project");
    expect(result).toBe("cancel");
    // After back, the NEXT select (3rd call) is the category frame again:
    // domain names present, subgroup rows absent.
    expect(select).toHaveBeenCalledTimes(3);
    const subFrame = choiceNames(select.mock.calls[1] as unknown[]);
    const catFrame = choiceNames(select.mock.calls[2] as unknown[]);
    expect(subFrame.some((n) => n.includes("AscendC 开发"))).toBe(true); // it WAS the subgroup frame
    expect(catFrame.some((n) => n.includes("ops — 算子开发 ("))).toBe(true);
    expect(catFrame.some((n) => n.includes("AscendC 开发"))).toBe(false); // back to category, not subgroup
  });
});
