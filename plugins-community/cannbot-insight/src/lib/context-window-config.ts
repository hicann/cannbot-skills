// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import fs from "node:fs";
import path from "node:path";

const DEFAULT_CONTEXT_WINDOWS: Record<string, number> = {
  "gpt-4o": 128000,
  "gpt-4o-mini": 128000,
  "gpt-4": 8192,
  "gpt-4-32k": 32768,
  "gpt-4-turbo": 128000,
  "gpt-3.5-turbo": 16385,
  "gpt-4.1": 1047576,
  "gpt-4.1-mini": 1047576,
  "gpt-4.1-nano": 1047576,
  "o3": 200000,
  "o3-mini": 100000,
  "o4-mini": 200000,
  "claude-3-opus": 200000,
  "claude-3-sonnet": 200000,
  "claude-3-haiku": 200000,
  "claude-3.5-sonnet": 200000,
  "claude-3.5-haiku": 200000,
  "claude-sonnet-4": 200000,
  "claude-sonnet-4-5": 200000,
  "claude-sonnet-4-6": 200000,
  "claude-opus-4": 200000,
  "claude-opus-4-7": 200000,
  "claude-opus-4-8": 200000,
  "claude-opus-5": 1048576,
  "claude-haiku-4-5": 200000,
  "glm-5": 200000,
  "glm-5.1": 200000,
  "glm-5.2": 1000000,
  "deepseek-v3": 131072,
  "deepseek-r1": 128000,
  "deepseek-v4": 1000000,
  "deepseek-v4-pro": 1000000,
  "deepseek-v4-flash": 1048576,
  "gemini-2.5-pro": 1048576,
  "gemini-2.5-flash": 1048576,
  "gemini-1.5-pro": 2097152,
  "gemini-1.5-flash": 1048576,
  "qwen3": 131072,
  "qwen3-moe": 32768,
  "qwen3.7-max": 1048576,
  "qwen3-235b": 131072,
  "qwen2.5": 131072,
  "llama-4-maverick": 1048576,
  "llama-3.3-70b": 131072,
  "llama-3.1-405b": 131072,
  "mistral-large": 131072,
  "mistral-medium": 32768,
};

const FALLBACK_DEFAULT = 200000;

interface ContextWindowConfig {
  models?: Record<string, number>;
  defaultContextWindow?: number;
}

let cachedConfig: ContextWindowConfig | null = null;

function loadConfig(): ContextWindowConfig {
  if (cachedConfig !== null) return cachedConfig;

  const configPath = path.join(process.cwd(), "context-windows.json");
  try {
    if (fs.existsSync(configPath)) {
      const raw = fs.readFileSync(configPath, "utf-8");
      cachedConfig = JSON.parse(raw) as ContextWindowConfig;
      return cachedConfig;
    }
  } catch (e) {
    console.warn("Failed to load context-windows.json, using defaults:", e);
  }

  cachedConfig = {};
  return cachedConfig;
}

export function getContextWindowLimit(model: string | null): number {
  const config = loadConfig();
  const defaultWindow = config.defaultContextWindow ?? FALLBACK_DEFAULT;

  if (!model) return defaultWindow;

  if (config.models?.[model]) return config.models[model];

  if (DEFAULT_CONTEXT_WINDOWS[model]) return DEFAULT_CONTEXT_WINDOWS[model];

  const parts = model.split("/");
  if (parts.length >= 2) {
    if (config.models?.[parts[1]]) return config.models[parts[1]];
    if (DEFAULT_CONTEXT_WINDOWS[parts[1]]) return DEFAULT_CONTEXT_WINDOWS[parts[1]];
  }

  for (const key of Object.keys(config.models ?? {})) {
    if (model.includes(key)) return config.models![key];
  }
  for (const key of Object.keys(DEFAULT_CONTEXT_WINDOWS)) {
    if (model.includes(key)) return DEFAULT_CONTEXT_WINDOWS[key];
  }

  return defaultWindow;
}

export function getAllContextWindows(): Record<string, number> {
  const config = loadConfig();
  return { ...DEFAULT_CONTEXT_WINDOWS, ...(config.models ?? {}) };
}

export function getDefaultContextWindow(): number {
  const config = loadConfig();
  return config.defaultContextWindow ?? FALLBACK_DEFAULT;
}

export function resetConfigCache(): void {
  cachedConfig = null;
}
