// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

const ROOT = path.join(__dirname, "..");
const DIST_PACKAGES = path.join(ROOT, "dist-packages");

const VERSION = JSON.parse(
  fs.readFileSync(path.join(ROOT, "package.json"), "utf-8")
).version;

const TARGETS = [
  { bunTarget: "bun-linux-x64",       pkgName: "install-helper-linux-x64",       binary: "install-helper",       os: ["linux"],  cpu: ["x64"] },
  { bunTarget: "bun-linux-arm64",     pkgName: "install-helper-linux-arm64",     binary: "install-helper",       os: ["linux"],  cpu: ["arm64"] },
  { bunTarget: "bun-darwin-x64",      pkgName: "install-helper-darwin-x64",      binary: "install-helper",       os: ["darwin"], cpu: ["x64"] },
  { bunTarget: "bun-darwin-arm64",    pkgName: "install-helper-darwin-arm64",    binary: "install-helper",       os: ["darwin"], cpu: ["arm64"] },
  { bunTarget: "bun-windows-x64",     pkgName: "install-helper-windows-x64",     binary: "install-helper.exe",   os: ["win32"],  cpu: ["x64"] },
];

function run() {
  console.log(`Building install-helper v${VERSION} npm sub-packages...\n`);

  execSync("node scripts/gen-embedded.cjs", { cwd: ROOT, stdio: "inherit" });
  execSync("npx tsup", { cwd: ROOT, stdio: "inherit" });

  if (fs.existsSync(DIST_PACKAGES)) {
    fs.rmSync(DIST_PACKAGES, { recursive: true });
  }
  fs.mkdirSync(DIST_PACKAGES, { recursive: true });

  for (const target of TARGETS) {
    const pkgDir = path.join(DIST_PACKAGES, target.pkgName);
    fs.mkdirSync(pkgDir, { recursive: true });

    console.log(`  Building ${target.pkgName}...`);
    execSync(
      `bun build src/index.ts --compile --target=${target.bunTarget} --define INSTALL_HELPER_VERSION='"${VERSION}"' --outfile="${path.join(pkgDir, target.binary)}"`,
      { cwd: ROOT, stdio: "pipe" }
    );

    const pkgJson = {
      name: `@cannbot-ai/${target.pkgName}`,
      version: VERSION,
      description: `install-helper binary for ${target.pkgName.replace("install-helper-", "")}`,
      os: target.os,
      cpu: target.cpu,
      files: [target.binary],
    };
    fs.writeFileSync(
      path.join(pkgDir, "package.json"),
      JSON.stringify(pkgJson, null, 2)
    );

    const size = fs.statSync(path.join(pkgDir, target.binary)).size;
    console.log(`  ✓ ${target.pkgName} (${(size / 1024 / 1024).toFixed(0)}MB)`);
  }

  console.log(`\nAll ${TARGETS.length} sub-packages built in dist-packages/`);
  fs.readdirSync(DIST_PACKAGES).forEach((d) => {
    console.log(`  ${d}/`);
  });
}

run();
