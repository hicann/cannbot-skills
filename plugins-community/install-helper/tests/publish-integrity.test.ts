// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// Publish integrity guards — regression tests for the 1.1.11/1.1.12 incident
// where the published npm tarball was missing the entire bin/ directory
// (npm silently omits missing `files` entries) and the global install broke
// with exit 127. These tests pin every link of the accident chain:
//
//   A. package.json declares a bin entry that actually exists in the repo
//   B. package-lock.json stays in sync with package.json (no stale versions)
//   C. build-binaries.sh never reintroduces `rm -rf bin/`
//   D. publish guard (verifyMainPackage) actually rejects broken packages
//   E. real `npm pack` tarball contains all required entry files

import { describe, it, expect, afterAll } from "vitest";
import {
  existsSync,
  readFileSync,
  mkdtempSync,
  rmSync,
  writeFileSync,
  mkdirSync,
} from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { createRequire } from "module";
import { execSync } from "child_process";

const require = createRequire(import.meta.url);
const { verifyMainPackage } = require("../scripts/publish-npm.cjs");

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT = join(__dirname, "..");

function readJson(p: string): any {
  return JSON.parse(readFileSync(p, "utf-8"));
}

const pkg = readJson(join(ROOT, "package.json"));

// ---------------------------------------------------------------------------
// A. package declarations vs repository reality
// ---------------------------------------------------------------------------
describe("publish integrity — package declarations", () => {
  it("bin entry points to a file that exists in the repo", () => {
    const binRel = pkg.bin && pkg.bin["install-helper"];
    expect(binRel).toBe("./bin/install-helper.js");
    expect(existsSync(join(ROOT, binRel))).toBe(true);
  });

  it("files whitelist includes both bin/ wrapper files", () => {
    const files: string[] = pkg.files || [];
    expect(files).toContain("bin/install-helper.js");
    expect(files).toContain("bin/package.json");
  });

  it("dist bundle entry is declared in files", () => {
    expect(pkg.files || []).toContain("dist/");
  });

  it("all 5 platform optionalDependencies match the main version exactly", () => {
    const expected = [
      "@cannbot-ai/install-helper-linux-x64",
      "@cannbot-ai/install-helper-linux-arm64",
      "@cannbot-ai/install-helper-darwin-x64",
      "@cannbot-ai/install-helper-darwin-arm64",
      "@cannbot-ai/install-helper-windows-x64",
    ];
    const optDeps = pkg.optionalDependencies || {};
    expect(Object.keys(optDeps).sort()).toEqual(expected.sort());
    for (const name of expected) {
      expect(optDeps[name]).toBe(pkg.version);
    }
  });
});

// ---------------------------------------------------------------------------
// B. lockfile version sync (guards against stale version residue, e.g. a
//    leftover beta version after a release bump)
// ---------------------------------------------------------------------------
describe("publish integrity — lockfile sync", () => {
  const lock = readJson(join(ROOT, "package-lock.json"));

  it("lockfile top-level version matches package.json", () => {
    expect(lock.version).toBe(pkg.version);
  });

  it("lockfile root package version matches package.json", () => {
    expect(lock.packages && lock.packages[""].version).toBe(pkg.version);
  });

  it("lockfile optionalDependencies match package.json (no stale versions)", () => {
    const lockDeps = (lock.packages && lock.packages[""].optionalDependencies) || {};
    const pkgDeps = pkg.optionalDependencies || {};
    for (const name of Object.keys(pkgDeps)) {
      expect(lockDeps[name]).toBe(pkgDeps[name]);
    }
  });

  it("lockfile contains no version newer or different from package.json for platform sub-packages", () => {
    for (const [p, entry] of Object.entries(lock.packages || {})) {
      if (p.startsWith("node_modules/@cannbot-ai/install-helper-")) {
        const entryVersion = (entry as any).version;
        expect(entryVersion).toBe(pkg.version);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// C. build-binaries.sh safety — the 1.1.11/1.1.12 root cause was
//    `rm -rf bin/` deleting the git-tracked entry wrapper before publish.
// ---------------------------------------------------------------------------
describe("publish integrity — build script safety", () => {
  const scriptPath = join(ROOT, "scripts", "build-binaries.sh");
  const script = readFileSync(scriptPath, "utf-8");

  it("never wipes the whole bin/ directory (root cause regression guard)", () => {
    // `rm -rf bin/`, `rm -rf bin`, and chained variants like
    // `rm -rf bin/; mkdir -p bin/` or `rm -rf bin/ && ...` must never
    // reappear. The exemption is prose in comments (preceded by ` or ').
    const dangerous = script
      .split("\n")
      .map((line) => line.replace(/^(#|\s*\/\/).*$/, ""))
      .join("\n");
    expect(dangerous).not.toMatch(/(^|[^\`'\w])rm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\s+bin\/?(\s|$|;|&)/);
  });

  it("cleans build artifacts selectively (rm -f bin/install-helper-*)", () => {
    expect(script).toMatch(/rm\s+-f\s+bin\/install-helper-/);
  });

  it("keeps the self-heal guard that restores the wrapper from git", () => {
    expect(script).toMatch(/git checkout -- bin\//);
    expect(script).toMatch(/bin\/install-helper\.js/);
  });
});

// ---------------------------------------------------------------------------
// D. publish guard — verifyMainPackage must reject a package missing bin/
//    and accept a complete one. Runs against throwaway package dirs.
// ---------------------------------------------------------------------------
describe("publish integrity — verifyMainPackage guard", () => {
  const tmpDirs: string[] = [];

  function makeFixture(withBin: boolean): string {
    const dir = mkdtempSync(join(__dirname, ".fixture-publish-"));
    tmpDirs.push(dir);
    const fixturePkg = {
      name: "fixture-publish-integrity",
      version: "0.0.0",
      bin: { "install-helper": "./bin/install-helper.js" },
      files: ["dist/", "bin/install-helper.js", "bin/package.json"],
    };
    writeFileSync(join(dir, "package.json"), JSON.stringify(fixturePkg, null, 2));
    mkdirSync(join(dir, "dist"), { recursive: true });
    writeFileSync(join(dir, "dist", "index.js"), "// fixture\n");
    if (withBin) {
      mkdirSync(join(dir, "bin"), { recursive: true });
      writeFileSync(join(dir, "bin", "install-helper.js"), "// fixture\n");
      writeFileSync(join(dir, "bin", "package.json"), '{ "type": "commonjs" }\n');
    }
    // npm pack requires a README when publishing, dry-run does not; keep minimal.
    return dir;
  }

  afterAll(() => {
    for (const d of tmpDirs) rmSync(d, { recursive: true, force: true });
  });

  it("accepts a complete package containing bin/ wrapper files", () => {
    const dir = makeFixture(true);
    expect(verifyMainPackage(dir)).toBe(true);
  });

  it("aborts (exit 1) when bin/ is missing — the 1.1.11/1.1.12 scenario", () => {
    const dir = makeFixture(false);
    // verifyMainPackage calls process.exit(1) on a broken package, which
    // vitest surfaces as a "process.exit unexpectedly" error — and
    // execSync inside would throw first when npm pack exits non-zero.
    // Silence stderr so the intentional failure output stays out of logs.
    const origWrite = process.stderr.write.bind(process.stderr);
    (process.stderr as any).write = () => true;
    let failed = false;
    try {
      verifyMainPackage(dir);
    } catch {
      failed = true;
    } finally {
      (process.stderr as any).write = origWrite;
    }
    expect(failed, "verifyMainPackage must fail on a package missing bin/").toBe(true);
  });
});

// ---------------------------------------------------------------------------
// E. Real tarball integration — strongest guard, closest to the incident.
//    Only meaningful when dist/ has been built; skipped otherwise (CI runs
//    tests without a prior build).
// ---------------------------------------------------------------------------
describe("publish integrity — real npm pack tarball", () => {
  const hasDist = existsSync(join(ROOT, "dist", "index.js"));

  it.skipIf(!hasDist)("tarball contains all required entry files", () => {
    const out = execSync("npm pack --dry-run --json", {
      cwd: ROOT,
      encoding: "utf-8",
    });
    const files = JSON.parse(out)[0].files.map((f: any) => f.path);
    for (const req of [
      "bin/install-helper.js",
      "bin/package.json",
      "dist/index.js",
    ]) {
      expect(files, `tarball missing ${req}`).toContain(req);
    }
  });
});
