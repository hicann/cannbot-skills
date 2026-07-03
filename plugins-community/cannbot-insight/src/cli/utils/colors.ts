// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import chalk from 'chalk';

export const theme = {
  primary: chalk.cyan,
  secondary: chalk.gray,
  success: chalk.green,
  warning: chalk.yellow,
  danger: chalk.red,
  info: chalk.blue,
  muted: chalk.dim,
  bold: chalk.bold,
  highlight: chalk.bgCyan.black,
};

export function formatHeader(text: string): string {
  return theme.bold(theme.primary(text));
}

export function formatLabel(label: string, value: string): string {
  return `${theme.secondary(label)}: ${value}`;
}

export function formatError(message: string): string {
  return theme.danger(message);
}

export function formatSuccess(message: string): string {
  return theme.success(message);
}

export function formatWarning(message: string): string {
  return theme.warning(message);
}

export function formatDivider(width: number = 80): string {
  return theme.muted('─'.repeat(width));
}
