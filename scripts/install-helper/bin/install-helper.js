#!/usr/bin/env node
// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------


const { spawn } = require("child_process");
const { existsSync } = require("fs");
const { join, dirname } = require("path");

const PLATFORM_MAP = {
  "linux-x64": "@cannbot-ai/install-helper-linux-x64",
  "linux-arm64": "@cannbot-ai/install-helper-linux-arm64",
  "darwin-x64": "@cannbot-ai/install-helper-darwin-x64",
  "darwin-arm64": "@cannbot-ai/install-helper-darwin-arm64",
  "win32-x64": "@cannbot-ai/install-helper-windows-x64",
};

const key = `${process.platform}-${process.arch}`;
const pkgName = PLATFORM_MAP[key];
const binaryName = process.platform === "win32" ? "install-helper.exe" : "install-helper";
const args = process.argv.slice(2);

if (pkgName) {
  try {
    const pkgPath = require.resolve(`${pkgName}/package.json`);
    const binaryPath = join(dirname(pkgPath), binaryName);
    if (existsSync(binaryPath)) {
      const child = spawn(binaryPath, args, { stdio: "inherit" });
      child.on("exit", (code, signal) => { process.exit(code ?? (signal ? 1 : 0)); });
      child.on("error", () => runJSFallback());
      return;
    }
  } catch {}
}

runJSFallback();

function runJSFallback() {
  const jsPath = join(__dirname, "..", "dist", "index.js");
  if (existsSync(jsPath)) {
    const child = spawn(process.execPath, [jsPath, ...args], { stdio: "inherit" });
    child.on("exit", (code, signal) => { process.exit(code ?? (signal ? 1 : 0)); });
    child.on("error", () => {
      console.error("install-helper: failed to run");
      process.exit(1);
    });
    return;
  }
  console.error(`install-helper binary not found for ${key}`);
  console.error("Try: npm install -g @cannbot-ai/install-helper");
  process.exit(1);
}
