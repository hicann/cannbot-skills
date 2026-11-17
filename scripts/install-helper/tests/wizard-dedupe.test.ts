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
import type { PluginEntry } from "../src/types/index.js";
import { dedupeForWizardSelection } from "../src/ui/wizard.js";

function entry(id: string, source: "cannbot" | "skills", displayName = id, description = ""): PluginEntry {
  return {
    id,
    dir: "",
    displayName,
    script: "cannbot",
    aliases: [],
    skills: 0,
    agents: 0,
    description,
    source,
  };
}

describe("wizard selection dedupe", () => {
  it("lists each plugin id exactly once — cannbot source wins, repository name shown", () => {
    const dual = [
      entry("ops-direct-invoke", "skills", "AscendC Kernel 直调"), // frozen local yml
      entry("ops-direct-invoke", "cannbot", "ops-direct-invoke"), // cannbot snapshot/mainline
    ];
    const result = dedupeForWizardSelection(dual);
    expect(result).toHaveLength(1);
    expect(result[0].source).toBe("cannbot"); // routing matches findPlugin default
    // the repository's own name is the display authority — users map the
    // wizard row straight onto the repo/PR/docs identity of the plugin
    expect(result[0].displayName).toBe("ops-direct-invoke");
  });

  it("keeps the cannbot display name when it is friendlier than the id echo", () => {
    const result = dedupeForWizardSelection([
      entry("ops-x", "cannbot", "Ops X（主线新版）"),
      entry("ops-x", "skills", "ops-x"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].displayName).toBe("Ops X（主线新版）");
  });

  it("prefers the cannbot-declared display name over the frozen local name", () => {
    // once the cannbot repo manifests carry displayName, the declaration is
    // the authority — the frozen embedded name must not shadow it
    const result = dedupeForWizardSelection([
      entry("pypto-pro-op-orchestrator", "skills", "PyPTO 旧冻结名"),
      entry("pypto-pro-op-orchestrator", "cannbot", "PyPTO-Pro 算子开发"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].displayName).toBe("PyPTO-Pro 算子开发");
  });

  it("leaves single-source plugins untouched", () => {
    const single = [entry("torch-compile", "skills", "torch.compile 图模式")];
    expect(dedupeForWizardSelection(single)).toEqual(single);
  });
});
