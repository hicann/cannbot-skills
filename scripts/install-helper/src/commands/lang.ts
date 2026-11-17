// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import chalk from "chalk";
import { readConfig, updateConfig } from "../utils/config.js";
import { setLanguage, t } from "../utils/i18n.js";

export async function langCommand(
  action?: string,
  value?: string
): Promise<void> {
  const config = readConfig();

  if (!action || action === "show") {
    console.log();
    console.log(chalk.bold(`  ${t("lang_title")}`));
    console.log(chalk.dim("  " + "─".repeat(40)));
    console.log();
    console.log(`  ${chalk.dim(t("lang_current") + ":")} ${config.language}`);
    console.log();
    console.log(chalk.dim(`  ${t("lang_supported")}`));
    console.log(chalk.dim(`  ${t("lang_usage")}`));
    console.log();
    return;
  }

  if (action === "set") {
    if (!value) {
      console.log(chalk.red(`  ${t("lang_specify")}`));
      return;
    }

    if (value !== "zh_CN" && value !== "en_US") {
      console.log(chalk.red(`  ${t("lang_invalid")}`));
      return;
    }

    updateConfig({ language: value });
    setLanguage(value);
    console.log();
    console.log(chalk.green(`  ✓ ${t("lang_set")} ${value}`));
    console.log();
    return;
  }

  console.log(chalk.red(`  ${t("lang_unknown_action")}: ${action}`));
  console.log(chalk.dim(`  ${t("lang_supported_actions")}`));
}
