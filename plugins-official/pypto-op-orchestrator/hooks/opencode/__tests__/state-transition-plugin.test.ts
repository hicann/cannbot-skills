// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { expect, mock, test } from "bun:test";
import { $ } from "bun";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const optionalSchema = () => ({ optional: optionalSchema });
const tool = Object.assign(
  <T>(definition: T): T => definition,
  {
    schema: {
      number: optionalSchema,
      string: optionalSchema,
    },
  },
);

mock.module("@opencode-ai/plugin", () => ({ tool }));

const { PyptoStateTransitionPlugin } = await import("../pypto-state-transition");

type StateTransitionTool = {
  execute: (
    args: Record<string, unknown>,
    context: Record<string, unknown>,
  ) => Promise<string>;
};

async function makeTool(directory: string, worktree = directory): Promise<StateTransitionTool> {
  const plugin = await PyptoStateTransitionPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory,
    worktree,
    project: {},
  } as never);
  return plugin.tool?.state_transition as unknown as StateTransitionTool;
}

test("init creates a missing operator directory under custom", async () => {
  const worktree = mkdtempSync(join(tmpdir(), "pypto-state-transition-"));
  try {
    const transition = await makeTool(worktree);
    await transition.execute(
      {
        action: "init",
        stage: 1,
        max_stage: 7,
        opDir: "custom/smoke_op",
      },
      { agent: "build" },
    );

    const statePath = join(worktree, "custom", "smoke_op", ".orchestrator_state.json");
    expect(existsSync(statePath)).toBeTrue();
    const state = JSON.parse(readFileSync(statePath, "utf8"));
    expect(state.operator_name).toBe("smoke_op");
    expect(state.stage_status["1"]).toBe("in_progress");
  } finally {
    rmSync(worktree, { recursive: true, force: true });
  }
});

test("init rejects an output directory outside custom before creating it", async () => {
  const worktree = mkdtempSync(join(tmpdir(), "pypto-state-transition-"));
  try {
    const transition = await makeTool(worktree);
    await expect(
      transition.execute(
        {
          action: "init",
          stage: 1,
          max_stage: 7,
          opDir: "other/smoke_op",
        },
        { agent: "build" },
      ),
    ).rejects.toThrow("may only update an operator directory");
    expect(existsSync(join(worktree, "other", "smoke_op"))).toBeFalse();
  } finally {
    rmSync(worktree, { recursive: true, force: true });
  }
});

test("init resolves custom under the active directory rather than a broader worktree", async () => {
  const worktree = mkdtempSync(join(tmpdir(), "pypto-state-transition-worktree-"));
  const directory = join(worktree, "project");
  mkdirSync(directory);
  try {
    const transition = await makeTool(directory, worktree);
    await transition.execute(
      {
        action: "init",
        stage: 1,
        max_stage: 7,
        opDir: "custom/smoke_op",
      },
      { agent: "build" },
    );

    expect(existsSync(join(directory, "custom", "smoke_op", ".orchestrator_state.json"))).toBeTrue();
    expect(existsSync(join(worktree, "custom", "smoke_op", ".orchestrator_state.json"))).toBeFalse();
  } finally {
    rmSync(worktree, { recursive: true, force: true });
  }
});
