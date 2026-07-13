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
import ora, { type Ora } from "ora";
import { t } from "./i18n.js";
import { getDisplayWidth } from "../ui/theme.js";

export const logger = {
  success: (msg: string) => console.log(chalk.green("✓") + " " + msg),
  error: (msg: string) => console.log(chalk.red("✗") + " " + msg),
  warn: (msg: string) => console.log(chalk.yellow("⚠") + " " + msg),
  info: (msg: string) => console.log(chalk.cyan("→") + " " + msg),
  step: (msg: string) => console.log(chalk.dim(msg)),
  blank: () => console.log(),
};

export function createSpinner(text: string): Ora {
  return ora({ text, color: "cyan" });
}

export function printBanner(subtitle?: string): void {
  console.log(chalk.cyan(`
   ____    _    _   _ _   _ ____        _
  / ___|  / \\  | \\ | | \\ | | __ )  ___ | |_
 | |     / _ \\ |  \\| |  \\| |  _ \\ / _ \\| __|
 | |___ / ___ \\| |\\  | |\\  | |_) | (_) | |_
  \\____/_/   \\_\\_| \\_|_| \\_|____/ \\___/ \\__|
`));
  if (subtitle) {
    console.log(chalk.bold(`  ${subtitle}`));
    console.log();
  }
}

export function printBoxTitle(title: string, width: number = 65): void {
  const border = "═".repeat(width);
  const displayWidth = getDisplayWidth(title);
  const totalPadding = width - displayWidth;
  const leftPadding = Math.floor(totalPadding / 2);
  const rightPadding = totalPadding - leftPadding;
  const paddedTitle = " ".repeat(Math.max(0, leftPadding)) + title + " ".repeat(Math.max(0, rightPadding));
  console.log();
  console.log(chalk.cyan.bold(`  ╔${border}╗`));
  console.log(chalk.cyan.bold(`  ║`) + chalk.cyan.bold(paddedTitle) + chalk.cyan.bold(`║`));
  console.log(chalk.cyan.bold(`  ╚${border}╝`));
  console.log();
}

export function showOperationHints(isCheckbox: boolean = false): void {
  if (isCheckbox) {
    console.log(`  💡 \x1b[36m↑↓\x1b[0m ${t("hint_select_move")} | \x1b[36m${t("hint_select_space")}\x1b[0m ${t("hint_select_check")} | \x1b[36m⏎\x1b[0m ${t("hint_select_confirm")}（${t("hint_checkbox_suffix")}）\n`);
  } else {
    console.log(`  💡 \x1b[36m↑↓\x1b[0m ${t("hint_select_move")} | \x1b[36m⏎\x1b[0m ${t("hint_select_confirm")}\n`);
  }
}
