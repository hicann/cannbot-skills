// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { afterEach, describe, expect, it } from "vitest";
import {
  chmodSync,
  cpSync,
  existsSync,
  mkdtempSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const pluginDir = path.resolve(fileURLToPath(new URL("..", import.meta.url)));
const temporaryDirs = new Set<string>();

afterEach(() => {
  for (const directory of temporaryDirs) {
    rmSync(directory, { force: true, recursive: true });
  }
  temporaryDirs.clear();
});

function writeExecutable(filePath: string, contents: string) {
  writeFileSync(filePath, contents, "utf8");
  chmodSync(filePath, 0o755);
}

function runStartScript(command: string, cliArgs: string[] = []) {
  const tempDir = mkdtempSync(path.join(tmpdir(), "cannbot-insight-start-"));
  temporaryDirs.add(tempDir);

  const appDir = path.join(tempDir, "app");
  const mockBinDir = path.join(tempDir, "mock-bin");
  const cliArgsFile = path.join(tempDir, "cli-args.txt");
  const startScript = path.join(appDir, "start.sh");

  mkdirSync(path.join(appDir, "node_modules"), { recursive: true });
  mkdirSync(path.join(appDir, "prisma"), { recursive: true });
  mkdirSync(mockBinDir, { recursive: true });
  cpSync(path.join(pluginDir, "start.sh"), startScript);
  chmodSync(startScript, 0o755);
  writeFileSync(path.join(appDir, "prisma", "dev.db"), "", "utf8");

  writeExecutable(
    path.join(mockBinDir, "node"),
    "#!/bin/sh\nif [ \"$1\" = \"-e\" ]; then\n  printf '20\\n'\nelse\n  printf 'v20.0.0\\n'\nfi\n",
  );
  writeExecutable(path.join(mockBinDir, "curl"), "#!/bin/sh\nexit 0\n");
  writeExecutable(
    path.join(mockBinDir, "npx"),
    "#!/bin/sh\nif [ \"$1\" = \"tsx\" ]; then\n  shift\n  printf '%s\\n' \"$@\" > \"$MOCK_CLI_ARGS_FILE\"\nfi\n",
  );

  const result = spawnSync("bash", [startScript, "-c", command, "--", ...cliArgs], {
    cwd: appDir,
    encoding: "utf8",
    env: {
      ...process.env,
      HOME: tempDir,
      MOCK_CLI_ARGS_FILE: cliArgsFile,
      PATH: `${mockBinDir}:${process.env.PATH ?? ""}`,
    },
  });

  expect(result.status).toBe(0);
  expect(existsSync(cliArgsFile)).toBe(true);
  return readFileSync(cliArgsFile, "utf8").trimEnd().split("\n");
}

describe("start.sh CLI mode", () => {
  it("runs the documented TUI command", () => {
    expect(runStartScript("tui")).toEqual([
      "src/cli/index.ts",
      "tui",
      "--server",
      "http://localhost:21025",
    ]);
  });

  it("keeps special command text as one argument", () => {
    expect(runStartScript("tui; literal-text")).toEqual([
      "src/cli/index.ts",
      "tui; literal-text",
      "--server",
      "http://localhost:21025",
    ]);
  });

  it("forwards explicit CLI arguments individually", () => {
    expect(runStartScript("sessions", ["--limit", "5"])).toEqual([
      "src/cli/index.ts",
      "sessions",
      "--limit",
      "5",
      "--server",
      "http://localhost:21025",
    ]);
  });
});
