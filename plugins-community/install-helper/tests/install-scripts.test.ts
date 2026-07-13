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

// Equivalent of install.ps1 Compare-SemVer and install.sh sort -V comparison
function compareSemVer(a: string, b: string): boolean {
  const aParts = a.split("-", 2);
  const bParts = b.split("-", 2);
  const aVerParts = aParts[0].split(".").map((n) => parseInt(n, 10));
  const bVerParts = bParts[0].split(".").map((n) => parseInt(n, 10));
  if (aVerParts.some(isNaN) || bVerParts.some(isNaN)) return false;
  for (let i = 0; i < Math.max(aVerParts.length, bVerParts.length); i++) {
    const av = aVerParts[i] || 0;
    const bv = bVerParts[i] || 0;
    if (av !== bv) return av > bv;
  }
  if (aParts.length === 1 && bParts.length === 1) return false;
  if (aParts.length === 1) return true;
  if (bParts.length === 1) return false;
  return aParts[1] > bParts[1];
}

function shouldSkipDownload(localVersion: string | null, remoteVersion: string | null): boolean {
  if (!localVersion || !remoteVersion) return false;
  return localVersion.trim() === remoteVersion.trim();
}

function buildMirrorOrder(preferred: string | null, all: string[]): string[] {
  if (!preferred) return [...all];
  const result = [preferred];
  for (const m of all) {
    if (m !== preferred) result.push(m);
  }
  return result;
}

function resolveArchitecture(arch: string): string {
  switch (arch) {
    case "x86_64":
    case "amd64":
      return "x64";
    case "arm64":
    case "aarch64":
      return "arm64";
    default:
      return "x64";
  }
}

function shouldFixExecutionPolicy(policy: string): boolean {
  return policy === "Restricted";
}

function chooseInstallPath(
  hasNode: boolean,
  nodeMajorVersion: number | null,
  os: string
): "npm" | "binary" | "reject" {
  if (os === "darwin" && (!hasNode || (nodeMajorVersion !== null && nodeMajorVersion < 18))) {
    return "reject";
  }
  if (hasNode && nodeMajorVersion !== null && nodeMajorVersion >= 18) {
    return "npm";
  }
  return "binary";
}

describe("install-scripts", () => {
  describe("compareSemVer", () => {
    it("equal versions return false", () => {
      expect(compareSemVer("1.1.2", "1.1.2")).toBe(false);
    });

    it("release > prerelease (1.1.2 > 0.0.2-beta.74)", () => {
      expect(compareSemVer("1.1.2", "0.0.2-beta.74")).toBe(true);
    });

    it("prerelease comparison (beta.74 > beta.73)", () => {
      expect(compareSemVer("0.0.2-beta.74", "0.0.2-beta.73")).toBe(true);
    });

    it("numeric comparison (1.0.1 > 1.0.0)", () => {
      expect(compareSemVer("1.0.1", "1.0.0")).toBe(true);
    });

    it("release > same-number prerelease (1.0.0 > 1.0.0-beta)", () => {
      expect(compareSemVer("1.0.0", "1.0.0-beta")).toBe(true);
    });

    it("invalid version A returns false", () => {
      expect(compareSemVer("invalid", "1.0.0")).toBe(false);
    });

    it("invalid version B returns false", () => {
      expect(compareSemVer("1.0.0", "invalid")).toBe(false);
    });

    it("prerelease string comparison (beta.2 > beta.10 by string order)", () => {
      expect(compareSemVer("1.0.0-beta.2", "1.0.0-beta.10")).toBe(true);
    });
  });

  describe("shouldSkipDownload", () => {
    it("local equals remote → skip", () => {
      expect(shouldSkipDownload("1.1.2", "1.1.2")).toBe(true);
    });

    it("local differs from remote → do not skip", () => {
      expect(shouldSkipDownload("0.0.2-beta.68", "1.1.2")).toBe(false);
    });

    it("no local version → do not skip", () => {
      expect(shouldSkipDownload(null, "1.1.2")).toBe(false);
    });

    it("no remote version → do not skip", () => {
      expect(shouldSkipDownload("1.1.2", null)).toBe(false);
    });
  });

  describe("buildMirrorOrder", () => {
    it("preferred mirror first", () => {
      const all = ["https://registry.npmjs.org", "https://registry.npmmirror.com"];
      const result = buildMirrorOrder("https://registry.npmmirror.com", all);
      expect(result[0]).toBe("https://registry.npmmirror.com");
      expect(result[1]).toBe("https://registry.npmjs.org");
    });

    it("deduplicates preferred mirror", () => {
      const all = ["https://registry.npmjs.org", "https://registry.npmmirror.com"];
      const result = buildMirrorOrder("https://registry.npmjs.org", all);
      expect(result).toHaveLength(2);
      expect(result).toEqual(["https://registry.npmjs.org", "https://registry.npmmirror.com"]);
    });

    it("null preferred returns all mirrors as-is", () => {
      const all = ["https://registry.npmjs.org", "https://registry.npmmirror.com"];
      const result = buildMirrorOrder(null, all);
      expect(result).toEqual(all);
    });
  });

  describe("resolveArchitecture", () => {
    it("x86_64 → x64", () => {
      expect(resolveArchitecture("x86_64")).toBe("x64");
    });

    it("arm64 → arm64", () => {
      expect(resolveArchitecture("arm64")).toBe("arm64");
    });

    it("aarch64 → arm64", () => {
      expect(resolveArchitecture("aarch64")).toBe("arm64");
    });

    it("amd64 → x64", () => {
      expect(resolveArchitecture("amd64")).toBe("x64");
    });

    it("unknown → x64 (default)", () => {
      expect(resolveArchitecture("riscv64")).toBe("x64");
    });
  });

  describe("shouldFixExecutionPolicy", () => {
    it("Restricted → true", () => {
      expect(shouldFixExecutionPolicy("Restricted")).toBe(true);
    });

    it("RemoteSigned → false", () => {
      expect(shouldFixExecutionPolicy("RemoteSigned")).toBe(false);
    });

    it("Unrestricted → false", () => {
      expect(shouldFixExecutionPolicy("Unrestricted")).toBe(false);
    });
  });

  describe("chooseInstallPath", () => {
    it("has Node 18+ → npm", () => {
      expect(chooseInstallPath(true, 18, "linux")).toBe("npm");
    });

    it("has Node 16 → binary", () => {
      expect(chooseInstallPath(true, 16, "linux")).toBe("binary");
    });

    it("no Node → binary", () => {
      expect(chooseInstallPath(false, null, "linux")).toBe("binary");
    });

    it("macOS without Node → reject", () => {
      expect(chooseInstallPath(false, null, "darwin")).toBe("reject");
    });
  });

  describe("channel resolution", () => {
    function resolveChannel(arg: string | undefined): string {
      return arg || "latest";
    }

    function resolveQueryTag(channel: string): string | null {
      if (channel === "latest" || channel === "beta") return channel;
      return null; // explicit version, no tag query needed
    }

    function resolveNpmSpec(channel: string): string {
      return `@cannbot-ai/install-helper@${channel}`;
    }

    it("default channel is latest", () => {
      expect(resolveChannel(undefined)).toBe("latest");
    });

    it("explicit beta channel", () => {
      expect(resolveChannel("beta")).toBe("beta");
    });

    it("explicit version channel", () => {
      expect(resolveChannel("1.1.3-beta.0")).toBe("1.1.3-beta.0");
    });

    it("latest channel queries latest tag only", () => {
      expect(resolveQueryTag("latest")).toBe("latest");
    });

    it("beta channel queries beta tag only", () => {
      expect(resolveQueryTag("beta")).toBe("beta");
    });

    it("explicit version skips tag query", () => {
      expect(resolveQueryTag("1.1.3-beta.0")).toBeNull();
    });

    it("npm spec for latest", () => {
      expect(resolveNpmSpec("latest")).toBe("@cannbot-ai/install-helper@latest");
    });

    it("npm spec for beta", () => {
      expect(resolveNpmSpec("beta")).toBe("@cannbot-ai/install-helper@beta");
    });

    it("npm spec for explicit version", () => {
      expect(resolveNpmSpec("1.1.3-beta.0")).toBe("@cannbot-ai/install-helper@1.1.3-beta.0");
    });
  });

  describe("channel-aware skip download", () => {
    function shouldSkipDownloadChannel(
      localVersion: string | null,
      remoteVersion: string | null,
      channel: string
    ): boolean {
      if (!localVersion || !remoteVersion) return false;
      return localVersion.trim() === remoteVersion.trim();
    }

    it("same beta version → skip", () => {
      expect(shouldSkipDownloadChannel("0.0.2-beta.76", "0.0.2-beta.76", "beta")).toBe(true);
    });

    it("local latest, install beta → do not skip", () => {
      expect(shouldSkipDownloadChannel("1.1.2", "0.0.2-beta.76", "beta")).toBe(false);
    });

    it("local beta, install latest → do not skip", () => {
      expect(shouldSkipDownloadChannel("0.0.2-beta.76", "1.1.2", "latest")).toBe(false);
    });

    it("same latest version → skip", () => {
      expect(shouldSkipDownloadChannel("1.1.2", "1.1.2", "latest")).toBe(true);
    });

    it("no local version → do not skip", () => {
      expect(shouldSkipDownloadChannel(null, "0.0.2-beta.76", "beta")).toBe(false);
    });
  });
});
