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

const SUB_PACKAGES = [
  "install-helper-linux-x64",
  "install-helper-linux-arm64",
  "install-helper-darwin-x64",
  "install-helper-darwin-arm64",
  "install-helper-windows-x64",
];

const REGISTRY = "https://registry.npmjs.org";
const SLEEP_MS = 12000;

function sleep(ms) {
  execSync(`sleep ${ms / 1000}`);
}

function publish(pkgDir, tag = "latest") {
  console.log(`  Publishing ${path.basename(pkgDir)}...`);
  execSync(`npm publish --access public --tag ${tag} --registry ${REGISTRY}`, {
    cwd: pkgDir,
    stdio: "inherit",
  });
}

function run() {
  const pkg = JSON.parse(
    fs.readFileSync(path.join(ROOT, "package.json"), "utf-8")
  );
  const version = pkg.version;
  const tag = version.includes("-") ? "beta" : "latest";
  console.log(`\n==========================================`);
  console.log(`  Publishing install-helper v${version} (tag: ${tag})`);
  console.log(`==========================================\n`);

  console.log("[1/2] Publishing sub-packages...\n");
  for (const subPkg of SUB_PACKAGES) {
    const pkgDir = path.join(DIST_PACKAGES, subPkg);
    if (!fs.existsSync(pkgDir)) {
      console.error(`  ✗ ${subPkg} not found. Run: node scripts/build-npm-packages.cjs`);
      process.exit(1);
    }
    publish(pkgDir, tag);
    console.log(`  ✓ ${subPkg}\n`);
    sleep(SLEEP_MS);
  }

  console.log("[2/2] Publishing main package...\n");
  execSync(`npm publish --access public --tag ${tag} --registry ${REGISTRY}`, {
    cwd: ROOT,
    stdio: "inherit",
  });
  console.log(`\n  ✓ @cannbot-ai/install-helper@${version}\n`);

  console.log(`\n==========================================`);
  console.log(`  Done! Verify:`);
  console.log(`  npm view @cannbot-ai/install-helper-linux-x64@${tag} --registry ${REGISTRY}`);
  console.log(`  npm view @cannbot-ai/install-helper@${tag} --registry ${REGISTRY}`);
  console.log(`==========================================\n`);
}

run();
