// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { select } from "@inquirer/prompts";
import chalk from "chalk";
import type { BackupInfo } from "../types/index.js";
import { t } from "../utils/i18n.js";

export async function showOverwriteWarning(
  currentPluginName: string,
  newPluginName: string
): Promise<"overwrite" | "cancel"> {
  console.log();
  console.log(chalk.yellow(`  ⚠️  ${t("backup_detected")}`));
  console.log();
  console.log(`    ${t("backup_current")}: ${chalk.cyan(currentPluginName)}`);
  console.log(`    ${t("backup_installing")}: ${chalk.cyan(newPluginName)}`);
  console.log();
  console.log(`    ${t("backup_overwrite_warning")}`);
  console.log(`    ${t("backup_auto_create")}`);
  console.log();

  const choice = await select({
    message: t("backup_select_action"),
    choices: [
      { name: t("backup_overwrite"), value: "overwrite" },
      { name: t("backup_cancel"), value: "cancel" },
    ],
  });

  return choice;
}

export async function showRestorePrompt(
  backups: BackupInfo[]
): Promise<string | "none"> {
  console.log();
  console.log(chalk.cyan(`  ${t("backup_restore_prompt")}`));
  console.log();

  const choices = backups.map((b) => ({
    name: `${t("backup_restore")} ${b.pluginName} (${b.backupTime})`,
    value: b.filePath,
  }));

  choices.push({ name: t("backup_only_uninstall"), value: "none" });

  const choice = await select({
    message: t("backup_select_action"),
    choices,
  });

  return choice;
}
