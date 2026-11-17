// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

const mocks = vi.hoisted(() => ({ configDir: "", install: vi.fn() }));
vi.mock("execa", () => ({ execa: mocks.install, execaSync: vi.fn() }));
vi.mock("../src/utils/paths.js", () => ({ getCannbotConfigDir: () => mocks.configDir }));
import { ensureCannbotInstaller } from "../src/core/cannbot-installer.js";

let root: string;
let previousChannel: string | undefined;
let previousOverride: string | undefined;
let stateFile: string;
beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "ih-channel-"));
  mocks.configDir = root;
  mocks.install.mockReset().mockResolvedValue({ exitCode: 0 });
  previousChannel = process.env.CANNBOT_INSTALLER_CHANNEL;
  previousOverride = process.env.CANNBOT_INSTALLER_PATH;
  delete process.env.CANNBOT_INSTALLER_PATH;
  const prefix = join(root, "installers", "cannbot");
  const pkg = join(prefix, "node_modules", "@cannbot-plugin", "cannbot");
  mkdirSync(join(pkg, "bin"), { recursive: true });
  writeFileSync(join(pkg, "bin", "cannbot.js"), "// test fixture");
  writeFileSync(join(pkg, "package.json"), JSON.stringify({ version: "1.0.0" }));
  stateFile = join(prefix, "state.json");
  writeFileSync(stateFile, JSON.stringify({ version: "1.0.0", channel: "latest", installedAt: Date.now() }));
});
afterEach(() => {
  if (previousChannel === undefined) delete process.env.CANNBOT_INSTALLER_CHANNEL;
  else process.env.CANNBOT_INSTALLER_CHANNEL = previousChannel;
  if (previousOverride === undefined) delete process.env.CANNBOT_INSTALLER_PATH;
  else process.env.CANNBOT_INSTALLER_PATH = previousOverride;
  rmSync(root, { recursive: true, force: true });
});

describe("managed installer channel changes", () => {
  it("refreshes immediately when a fresh cache belongs to a different channel", async () => {
    process.env.CANNBOT_INSTALLER_CHANNEL = "beta";
    await ensureCannbotInstaller();
    expect(mocks.install).toHaveBeenCalledTimes(1);
    expect(mocks.install.mock.calls[0][1]).toContain("@cannbot-plugin/cannbot@beta");
    expect(JSON.parse(readFileSync(stateFile, "utf8")).channel).toBe("beta");
  });
  it("keeps a fresh cache when the channel is unchanged", async () => {
    process.env.CANNBOT_INSTALLER_CHANNEL = "latest";
    await ensureCannbotInstaller();
    expect(mocks.install).not.toHaveBeenCalled();
  });
  it("keeps the existing offline fallback for an expired cache on the same channel", async () => {
    process.env.CANNBOT_INSTALLER_CHANNEL = "latest";
    const state = JSON.parse(readFileSync(stateFile, "utf8"));
    state.installedAt = Date.now() - 48 * 60 * 60 * 1000;
    writeFileSync(stateFile, JSON.stringify(state));
    mocks.install.mockRejectedValue(new Error("offline"));
    await expect(ensureCannbotInstaller()).resolves.toContain("cannbot");
    expect(mocks.install).toHaveBeenCalledTimes(1);
    expect(JSON.parse(readFileSync(stateFile, "utf8")).channel).toBe("latest");
  });
  it("does not claim a new channel when both installation attempts fail", async () => {
    process.env.CANNBOT_INSTALLER_CHANNEL = "beta";
    mocks.install.mockRejectedValue(new Error("offline"));
    await expect(ensureCannbotInstaller()).rejects.toThrow("offline");
    expect(mocks.install).toHaveBeenCalledTimes(2);
    expect(JSON.parse(readFileSync(stateFile, "utf8")).channel).toBe("latest");
  });
  it("does not assign a channel to an unknown cached package after a failed refresh", async () => {
    process.env.CANNBOT_INSTALLER_CHANNEL = "beta";
    rmSync(stateFile);
    mocks.install.mockRejectedValue(new Error("offline"));
    await ensureCannbotInstaller();
    expect(mocks.install).toHaveBeenCalledTimes(1);
    expect(() => readFileSync(stateFile, "utf8")).toThrow();
    await ensureCannbotInstaller();
    expect(mocks.install).toHaveBeenCalledTimes(2);
  });

  it("does not assign a channel when a damaged state cannot be refreshed", async () => {
    process.env.CANNBOT_INSTALLER_CHANNEL = "beta";
    writeFileSync(stateFile, "not-json");
    mocks.install.mockRejectedValue(new Error("offline"));
    await ensureCannbotInstaller();
    expect(readFileSync(stateFile, "utf8")).toBe("not-json");
  });

  it("treats an incomplete state as unknown and keeps the offline fallback", async () => {
    process.env.CANNBOT_INSTALLER_CHANNEL = "beta";
    const incomplete = JSON.stringify({ installedAt: Date.now() });
    writeFileSync(stateFile, incomplete);
    mocks.install.mockRejectedValue(new Error("offline"));
    await ensureCannbotInstaller();
    expect(readFileSync(stateFile, "utf8")).toBe(incomplete);
  });

  it("records the channel after successfully refreshing an unknown cache", async () => {
    process.env.CANNBOT_INSTALLER_CHANNEL = "beta";
    rmSync(stateFile);
    await ensureCannbotInstaller();
    expect(JSON.parse(readFileSync(stateFile, "utf8")).channel).toBe("beta");
  });
});
