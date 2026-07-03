// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import type { TokenUsage } from '../shared/types';

export interface ModelPricing {
  inputPricePerToken: number;
  outputPricePerToken: number;
  reasoningPricePerToken: number;
  cacheReadPricePerToken: number;
  cacheWritePricePerToken: number;
}

export const MODEL_PRICING: Record<string, ModelPricing> = {
  'alibaba-cn/glm-5': {
    inputPricePerToken: 0.0000005,
    outputPricePerToken: 0.000002,
    reasoningPricePerToken: 0,
    cacheReadPricePerToken: 0.000000125,
    cacheWritePricePerToken: 0.000000625,
  },
  'glm-5': {
    inputPricePerToken: 0.0000005,
    outputPricePerToken: 0.000002,
    reasoningPricePerToken: 0,
    cacheReadPricePerToken: 0.000000125,
    cacheWritePricePerToken: 0.000000625,
  },
  'anthropic/claude-3.5-sonnet': {
    inputPricePerToken: 0.000003,
    outputPricePerToken: 0.000015,
    reasoningPricePerToken: 0,
    cacheReadPricePerToken: 0.0000003,
    cacheWritePricePerToken: 0.00000375,
  },
  'claude-3.5-sonnet': {
    inputPricePerToken: 0.000003,
    outputPricePerToken: 0.000015,
    reasoningPricePerToken: 0,
    cacheReadPricePerToken: 0.0000003,
    cacheWritePricePerToken: 0.00000375,
  },
  'anthropic/claude-3-opus': {
    inputPricePerToken: 0.000015,
    outputPricePerToken: 0.000075,
    reasoningPricePerToken: 0,
    cacheReadPricePerToken: 0.0000015,
    cacheWritePricePerToken: 0.00001875,
  },
  'openai/gpt-4o': {
    inputPricePerToken: 0.0000025,
    outputPricePerToken: 0.00001,
    reasoningPricePerToken: 0,
    cacheReadPricePerToken: 0.00000125,
    cacheWritePricePerToken: 0,
  },
  'gpt-4o': {
    inputPricePerToken: 0.0000025,
    outputPricePerToken: 0.00001,
    reasoningPricePerToken: 0,
    cacheReadPricePerToken: 0.00000125,
    cacheWritePricePerToken: 0,
  },
};

function resolveModelKey(model: string): string | null {
  if (MODEL_PRICING[model]) return model;
  const parts = model.split('/');
  if (parts.length >= 2 && MODEL_PRICING[parts[1]]) return parts[1];
  for (const key of Object.keys(MODEL_PRICING)) {
    if (model.includes(key)) return key;
  }
  return null;
}

export function calculateCost(model: string, usage: TokenUsage): number {
  if (!usage) return 0;

  const key = resolveModelKey(model);
  if (!key) return 0;

  const pricing = MODEL_PRICING[key];
  const cost =
    usage.input * pricing.inputPricePerToken +
    usage.output * pricing.outputPricePerToken +
    usage.reasoning * pricing.reasoningPricePerToken +
    usage.cacheRead * pricing.cacheReadPricePerToken +
    usage.cacheWrite * pricing.cacheWritePricePerToken;

  return cost;
}

export function calculateCostNullSafe(model: string | null, usage: TokenUsage | null): number {
  if (!model || !usage) return 0;
  return calculateCost(model, usage);
}
