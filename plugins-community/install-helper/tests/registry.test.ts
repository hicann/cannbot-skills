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
import { writeFileSync, mkdirSync, existsSync, rmSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

describe("registry", () => {
  it("getAllPlugins returns plugins from plugins.d", async () => {
    const pluginsDir = join(__dirname, "..", "plugins.d");
    if (!existsSync(pluginsDir)) {
      console.warn("plugins.d not found, skipping registry tests");
      return;
    }
    const { getAllPlugins, findPlugin, getPluginById } = await import("../src/core/registry.js");
    const plugins = getAllPlugins();
    expect(plugins.length).toBeGreaterThan(0);

    for (const p of plugins) {
      expect(p.id).toBeTruthy();
      expect(p.dir).toBeTruthy();
      expect(p.displayName).toBeTruthy();
      expect(typeof p.script).toBe("string");
      expect(Array.isArray(p.aliases)).toBe(true);
    }

    const sorted = [...plugins].sort((a, b) =>
      a.displayName < b.displayName ? -1 : a.displayName > b.displayName ? 1 : 0
    );
    expect(plugins.map((p) => p.id)).toEqual(sorted.map((p) => p.id));
  });

  it("findPlugin finds by exact id", async () => {
    const { findPlugin } = await import("../src/core/registry.js");
    const plugin = findPlugin("ops-direct-invoke");
    expect(plugin).toBeDefined();
    expect(plugin!.id).toBe("ops-direct-invoke");
  });

  it("findPlugin finds by alias", async () => {
    const { findPlugin } = await import("../src/core/registry.js");
    const plugin = findPlugin("flash");
    expect(plugin).toBeDefined();
    expect(plugin!.id).toBe("ops-direct-invoke-flash");
  });

  it("findPlugin finds by prefix", async () => {
    const { findPlugin } = await import("../src/core/registry.js");
    const plugin = findPlugin("ops-direct-invoke-fl");
    expect(plugin).toBeDefined();
    expect(plugin!.id).toBe("ops-direct-invoke-flash");
  });

  it("findPlugin returns undefined for non-existent", async () => {
    const { findPlugin } = await import("../src/core/registry.js");
    const plugin = findPlugin("non-existent-plugin");
    expect(plugin).toBeUndefined();
  });

  it("getPluginById returns plugin by exact id", async () => {
    const { getPluginById } = await import("../src/core/registry.js");
    const plugin = getPluginById("torch-compile");
    expect(plugin).toBeDefined();
    expect(plugin!.id).toBe("torch-compile");
  });

  it("getPluginById returns undefined for non-existent", async () => {
    const { getPluginById } = await import("../src/core/registry.js");
    const plugin = getPluginById("non-existent");
    expect(plugin).toBeUndefined();
  });
});
