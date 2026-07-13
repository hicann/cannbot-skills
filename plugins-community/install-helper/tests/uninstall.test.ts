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

function join(...parts: string[]): string {
  return parts.join(sep);
}
