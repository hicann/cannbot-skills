// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { createCLI } from "./cli.js";
import { t } from "./utils/i18n.js";

function isExitError(error: any): boolean {
  if (!error) return false;
  return (
    error.name === "ExitPromptError" ||
    error.message?.includes("force closed") ||
    error.message?.includes("SIGINT") ||
    error.constructor?.name === "ExitPromptError"
  );
}

function handleExit(error: any): void {
  if (isExitError(error)) {
    console.log("\n" + t("init_cancelled"));
    process.exit(0);
  }
}

process.on("SIGINT", () => {
  console.log("\n" + t("init_cancelled"));
  process.exit(130);
});

process.on("uncaughtException", (error: any) => {
  handleExit(error);
  throw error;
});

process.on("unhandledRejection", (error: any) => {
  handleExit(error);
  process.exit(1);
});

const program = createCLI();
program.parse();
