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

// Mirrors the PLATFORM_MAP from bin/install-helper.js
const PLATFORM_MAP: Record<string, string | undefined> = {
  "linux-x64": "@cannbot-ai/install-helper-linux-x64",
  "linux-arm64": "@cannbot-ai/install-helper-linux-arm64",
  "darwin-x64": "@cannbot-ai/install-helper-darwin-x64",
  "darwin-arm64": "@cannbot-ai/install-helper-darwin-arm64",
  "win32-x64": "@cannbot-ai/install-helper-windows-x64",
};

function resolvePlatform(platform: string, arch: string): string | undefined {
  return PLATFORM_MAP[`${platform}-${arch}`];
}

function resolveBinaryName(platform: string): string {
  return platform === "win32" ? "install-helper.exe" : "install-helper";
}

function resolveFallback(
  hasBinary: boolean,
  hasJsFallback: boolean
): "binary" | "js-fallback" | "error" {
  if (hasBinary) return "binary";
  if (hasJsFallback) return "js-fallback";
  return "error";
}

describe("bin-wrapper", () => {
  describe("PLATFORM_MAP", () => {
    it("linux-x64 → correct package", () => {
      expect(resolvePlatform("linux", "x64")).toBe("@cannbot-ai/install-helper-linux-x64");
    });

    it("linux-arm64 → correct package", () => {
      expect(resolvePlatform("linux", "arm64")).toBe("@cannbot-ai/install-helper-linux-arm64");
    });

    it("darwin-x64 → correct package", () => {
      expect(resolvePlatform("darwin", "x64")).toBe("@cannbot-ai/install-helper-darwin-x64");
    });

    it("darwin-arm64 → correct package", () => {
      expect(resolvePlatform("darwin", "arm64")).toBe("@cannbot-ai/install-helper-darwin-arm64");
    });

    it("win32-x64 → correct package", () => {
      expect(resolvePlatform("win32", "x64")).toBe("@cannbot-ai/install-helper-windows-x64");
    });

    it("win32-arm64 → undefined (fallback)", () => {
      expect(resolvePlatform("win32", "arm64")).toBeUndefined();
    });
  });

  describe("resolveBinaryName", () => {
    it("win32 → install-helper.exe", () => {
      expect(resolveBinaryName("win32")).toBe("install-helper.exe");
    });

    it("linux → install-helper", () => {
      expect(resolveBinaryName("linux")).toBe("install-helper");
    });
  });

  describe("resolveFallback", () => {
    it("binary exists → spawn binary", () => {
      expect(resolveFallback(true, true)).toBe("binary");
    });

    it("binary missing, JS exists → js-fallback", () => {
      expect(resolveFallback(false, true)).toBe("js-fallback");
    });

    it("both missing → error", () => {
      expect(resolveFallback(false, false)).toBe("error");
    });
  });
});
