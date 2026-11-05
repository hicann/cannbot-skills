// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------

import { describe, it, expect } from "vitest";
import { pluginChoiceLabel } from "../src/ui/wizard.js";
import { truncateForDisplay, getDisplayWidth } from "../src/ui/theme.js";
import type { PluginEntry } from "../src/types/index.js";

function plugin(overrides: Partial<PluginEntry>): PluginEntry {
  return {
    id: "pypto-pro-op-orchestrator",
    dir: "",
    displayName: "PyPTO-Pro 算子开发",
    script: "cannbot",
    aliases: [],
    skills: 0,
    agents: 6,
    description: "",
    source: "cannbot",
    ...overrides,
  };
}

const LONG_DESCRIPTION =
  "PyPTO 算子开发编排工作流：7 阶段状态机（需求规划→Golden→设计→验证准备→按模块编码→E2E 验证→性能调优）经 harness 节点即 agent 编排驱动；Skills/hooks 等资产在安装时从 cannbot-skills 仓拉取并即时软链。";

describe("truncateForDisplay", () => {
  it("keeps text within the width budget as-is", () => {
    expect(truncateForDisplay("短文本", 10)).toBe("短文本");
    expect(truncateForDisplay("ascii text", 20)).toBe("ascii text");
  });

  it("truncates by display width (CJK counts double) and appends an ellipsis", () => {
    const out = truncateForDisplay(LONG_DESCRIPTION, 20);
    expect(out.endsWith("…")).toBe(true);
    expect(getDisplayWidth(out)).toBeLessThanOrEqual(20);
    // the visible part stays a prefix of the original
    expect(LONG_DESCRIPTION.startsWith(out.slice(0, -1))).toBe(true);
  });
});

describe("pluginChoiceLabel", () => {
  it("embeds a truncated description in the choice line", () => {
    const label = pluginChoiceLabel(plugin({ description: LONG_DESCRIPTION }), false);
    expect(label).toContain("PyPTO-Pro 算子开发");
    expect(label).toContain("[cannbot]");
    // inline description present, truncated — the full text stays on the
    // inquirer focus line, the row itself carries the identifying prefix
    expect(label).toContain("—");
    expect(label).toContain("…");
    expect(label).not.toContain(LONG_DESCRIPTION);
  });

  it("keeps a short description inline without truncation", () => {
    const label = pluginChoiceLabel(plugin({ description: "HCCL 通信算子开发" }), false);
    expect(label).toContain("— HCCL 通信算子开发");
    expect(label).not.toContain("…");
  });

  it("omits the description part when the plugin has none", () => {
    const label = pluginChoiceLabel(plugin({ description: "" }), false);
    expect(label).not.toContain("—");
  });

  it("omits the cannbot badge for skills-source plugins", () => {
    const label = pluginChoiceLabel(plugin({ source: "skills" }), false);
    expect(label).not.toContain("[cannbot]");
  });
});
