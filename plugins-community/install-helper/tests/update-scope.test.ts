// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { afterEach, describe, expect, it, vi } from "vitest";

const fixture = vi.hoisted(() => ({
  installed: [
    { id: "plugin-opencode-project", tool: "opencode", level: "project" },
    { id: "plugin-claude-global", tool: "claude", level: "global" },
    { id: "plugin-opencode-global", tool: "opencode", level: "global" },
    { id: "plugin-claude-project", tool: "claude", level: "project" },
  ],
  cannbotInstalled: [] as Array<{ id: string; tool: string }>,
  ensureRepoAndScan: vi.fn(async () => "/fixture/repo"),
  installPluginRouted: vi.fn(async ({ pluginId }: { pluginId: string }) => ({
    pluginId,
    success: true,
    errors: [],
    warnings: [],
    skillsCount: 1,
    agentsCount: 1,
  })),
  printInstallSummary: vi.fn(),
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), success: vi.fn() },
  spinner: { start: vi.fn(), succeed: vi.fn(), fail: vi.fn() },
}));

vi.mock("../src/core/repository.js", () => ({
  createRepositoryManager: () => ({ ensureRepoAndScan: fixture.ensureRepoAndScan }),
}));
vi.mock("../src/core/cannbot-delegate.js", () => ({
  boundaryErrors: () => [],
  installPluginRouted: fixture.installPluginRouted,
}));
vi.mock("../src/core/cannbot-installer.js", () => ({
  ensureCannbotDiscovery: async () => undefined,
  ensureCannbotInstaller: async () => undefined,
}));
vi.mock("../src/core/cannbot-registry.js", () => ({
  scanCannbotInstalled: () => fixture.cannbotInstalled,
}));
vi.mock("../src/core/registry.js", () => ({
  findPlugin: (id: string) => ({ id, displayName: id }),
}));
vi.mock("../src/core/manifest.js", () => ({ scanInstalled: () => fixture.installed }));
vi.mock("../src/ui/display.js", () => ({ printInstallSummary: fixture.printInstallSummary }));
vi.mock("../src/utils/logger.js", () => ({
  logger: fixture.logger,
  createSpinner: () => fixture.spinner,
}));
vi.mock("../src/utils/i18n.js", () => ({ t: (key: string) => key }));
vi.mock("../src/utils/paths.js", () => ({
  validateTool: (value: string) => value,
  validateLevel: (value: string) => value,
}));
vi.mock("chalk", () => ({ default: { cyan: (value: string) => value } }));

import { updateCommand } from "../src/commands/update.js";

describe("update target scope", () => {
  afterEach(() => {
    fixture.cannbotInstalled.length = 0;
    vi.clearAllMocks();
  });

  it("keeps every installed target when no scope options are given", async () => {
    await updateCommand([], {});

    expect(fixture.installPluginRouted).toHaveBeenCalledTimes(4);
  });

  it("filters all-plugin updates by tool and install level", async () => {
    await updateCommand([], { tool: "opencode", level: "project" });

    expect(fixture.installPluginRouted).toHaveBeenCalledTimes(1);
    expect(fixture.installPluginRouted).toHaveBeenCalledWith(
      expect.objectContaining({ pluginId: "plugin-opencode-project", tool: "opencode", level: "project" })
    );
  });

  it("filters all-plugin updates by either option independently", async () => {
    await updateCommand([], { tool: "opencode" });
    expect(fixture.installPluginRouted).toHaveBeenCalledTimes(2);

    vi.clearAllMocks();
    await updateCommand([], { level: "project" });
    expect(fixture.installPluginRouted).toHaveBeenCalledTimes(2);
  });

  it("keeps named-plugin updates scoped by tool and install level", async () => {
    await updateCommand(["plugin-opencode-project"], { tool: "opencode", level: "project" });

    expect(fixture.installPluginRouted).toHaveBeenCalledTimes(1);
    expect(fixture.installPluginRouted).toHaveBeenCalledWith(
      expect.objectContaining({ pluginId: "plugin-opencode-project", tool: "opencode", level: "project" })
    );
  });

  it("applies the project level to cannbot targets", async () => {
    fixture.cannbotInstalled.push(
      { id: "plugin-opencode-cannbot", tool: "opencode" },
      { id: "plugin-claude-cannbot", tool: "claude" }
    );

    await updateCommand([], { tool: "opencode", level: "project" });

    expect(fixture.installPluginRouted).toHaveBeenCalledTimes(2);
    expect(fixture.installPluginRouted).toHaveBeenCalledWith(
      expect.objectContaining({ pluginId: "plugin-opencode-cannbot", tool: "opencode", level: "project" })
    );
  });
});
