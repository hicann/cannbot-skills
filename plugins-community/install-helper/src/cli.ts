// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { Command } from "commander";
import { initCommand } from "./commands/init.js";
import { listCommand } from "./commands/list.js";
import { doctorCommand } from "./commands/doctor.js";
import { statusCommand } from "./commands/status.js";
import { installCommand } from "./commands/install.js";
import { uninstallCommand } from "./commands/uninstall.js";
import { updateCommand } from "./commands/update.js";
import { infoCommand } from "./commands/info.js";
import { langCommand } from "./commands/lang.js";
import { readConfig } from "./utils/config.js";
import { setLanguage } from "./utils/i18n.js";
import { readFileSync, existsSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

declare const INSTALL_HELPER_VERSION: string | undefined;

function readVersion(): string {
  try {
    if (typeof INSTALL_HELPER_VERSION !== "undefined" && INSTALL_HELPER_VERSION) {
      return INSTALL_HELPER_VERSION;
    }
  } catch {
  }
  try {
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = dirname(__filename);
    const candidates = [
      join(__dirname, "..", "package.json"),
      join(__dirname, "package.json"),
      join(process.cwd(), "package.json"),
    ];
    for (const pkgPath of candidates) {
      if (existsSync(pkgPath)) {
        const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));
        if (pkg.version && (pkg.name === "@cannbot-ai/install-helper" || pkg.name?.startsWith("@cannbot-ai/install-helper-"))) {
          return pkg.version;
        }
      }
    }
    return "0.0.0";
  } catch {
    return "0.0.0";
  }
}

export function createCLI(): Command {
  const config = readConfig();
  if (config.language) {
    setLanguage(config.language);
  }

  const program = new Command();

  program
    .name("install-helper")
    .description("CANNBot Install Helper - Interactive installer for CANN operator development skills")
    .version(readVersion());

  program
    .command("init", { isDefault: false })
    .description("Run interactive installation wizard")
    .action(async () => {
      await initCommand();
    });

  program
    .command("list")
    .description("List available plugins")
    .action(async () => {
      await listCommand();
    });

  program
    .command("doctor")
    .description("Run health check")
    .option("--fix", "Auto-fix detected issues")
    .action(async (options: { fix?: boolean }) => {
      await doctorCommand(options);
    });

  program
    .command("status")
    .description("Show installed plugins")
    .action(async () => {
      await statusCommand();
    });

  program
    .command("install [names...]")
    .description("Install plugins or skills (auto-detects type)")
    .option("-t, --tool <tool>", "AI tool (opencode, claude, trae, cursor, copilot)")
    .option("-l, --level <level>", "Install level (project, global)", "project")
    .option("-y, --yes", "Skip confirmation prompts")
    .option("-a, --all", "Install ALL available skills")
    .option("--list", "List all available skills by category")
    .action(async (names: string[], options: { tool?: string; level?: string; yes?: boolean; all?: boolean; list?: boolean }) => {
      await installCommand(names, options);
    });

  program
    .command("uninstall <names...>")
    .description("Uninstall plugins or skills (auto-detects type)")
    .option("-t, --tool <tool>", "AI tool (opencode, claude, trae, cursor, copilot)")
    .option("-l, --level <level>", "Install level (project, global)", "project")
    .action(async (names: string[], options: { tool?: string; level?: string }) => {
      await uninstallCommand(names, options);
    });

  program
    .command("update [plugins...]")
    .description("Update installed plugins (git pull + reinstall)")
    .option("-t, --tool <tool>", "AI tool (opencode, claude, trae, cursor, copilot)")
    .option("-l, --level <level>", "Install level (project, global)", "project")
    .action(async (plugins: string[], options: { tool?: string; level?: string }) => {
      await updateCommand(plugins, options);
    });

  program
    .command("info <plugin>")
    .description("Show plugin details")
    .action(async (plugin: string) => {
      await infoCommand(plugin);
    });

  program
    .command("lang [action] [value]")
    .description("Manage language settings (show/set)")
    .action(async (action?: string, value?: string) => {
      await langCommand(action, value);
    });

  program.action(async () => {
    await initCommand();
  });

  return program;
}
