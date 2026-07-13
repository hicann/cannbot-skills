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
import { scanInstalled } from "../core/manifest.js";
import { t } from "../utils/i18n.js";

export async function statusCommand(): Promise<void> {
  const installed = scanInstalled();
  const installedMap = new Map(installed.map((p) => [p.id, p]));

  console.log();
  console.log(chalk.bold(`  ${t("status_title")}`));
  console.log();

  if (installed.length === 0) {
    console.log(chalk.dim(`  ${t("status_none")}`));
    console.log(chalk.dim(`  ${t("status_hint")}`));
    console.log();
    return;
  }

  for (const inst of installed) {
    console.log(
      chalk.green("  ✓") +
        ` ${inst.displayName}` +
        chalk.dim(` (${inst.tool}, ${inst.level})`)
    );
    console.log(
      chalk.dim(`    ${inst.skillsCount} skills, ${inst.agentsCount} agents`)
    );
    console.log(chalk.dim(`    ${inst.configRoot}`));
    console.log(chalk.dim(`    ${t("status_install_time")}: ${inst.installTime}`));
    console.log();
  }
}
