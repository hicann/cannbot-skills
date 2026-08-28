// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

/** DB 侧 proxy 判定的唯一形式：Session.version 带 -proxy 后缀（统一分类器
 * proxy-classify.ts 导入时写入）。UI 徽标 / API 门禁一律用本谓词，不再各写
 * endsWith 字符串比较。 */
export function isProxyVersion(version: string | null | undefined): boolean {
  return version?.endsWith('-proxy') ?? false;
}

// framework（agent 归属：claude-code / opencode / codex）与捕获格式正交：proxy 捕获
// 文件（claude / opencode / codex，version 带 -proxy 后缀，统一分类器
// proxy-classify.ts 写入）都是扩展 claude 格式的 jsonl。
// 凡是"sourcePath 指向 claude 格式 jsonl"的判断（wire 轮次、full-context、
// 增量刷新等）都应走本谓词，而不是单看 framework。
export function isClaudeFormatSession(
  framework: string | null | undefined,
  version: string | null | undefined
): boolean {
  if (framework === 'claude-code') return true;
  return isProxyVersion(version);
}
