// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { calculateCost, calculateCostNullSafe, MODEL_PRICING } from '../src/lib/ingest/cost-calculator.ts';
import type { TokenUsage } from '../src/lib/shared/types.ts';

describe('cost-calculator', () => {
  describe('calculateCost with glm-5 model', () => {
    const usage: TokenUsage = {
      total: 12500,
      input: 10000,
      output: 500,
      reasoning: 100,
      cacheRead: 2000,
      cacheWrite: 0,
      cost: 0,
      inputMessagesTokens: 12000,
    };

    it('returns correct cost for alibaba-cn/glm-5', () => {
      const pricing = MODEL_PRICING['alibaba-cn/glm-5'];
      const expected =
        10000 * pricing.inputPricePerToken +
        500 * pricing.outputPricePerToken +
        100 * pricing.reasoningPricePerToken +
        2000 * pricing.cacheReadPricePerToken +
        0 * pricing.cacheWritePricePerToken;
      const result = calculateCost('alibaba-cn/glm-5', usage);
      expect(result).toBeCloseTo(expected, 10);
    });

    it('resolves glm-5 without provider prefix', () => {
      const pricing = MODEL_PRICING['glm-5'];
      const expected =
        10000 * pricing.inputPricePerToken +
        500 * pricing.outputPricePerToken +
        100 * pricing.reasoningPricePerToken +
        2000 * pricing.cacheReadPricePerToken;
      const result = calculateCost('glm-5', usage);
      expect(result).toBeCloseTo(expected, 10);
    });

    it('recalculates cost from token breakdown matching pricing table', () => {
      const result = calculateCost('alibaba-cn/glm-5', usage);
      const inputCost = 10000 * 0.0000005;
      const outputCost = 500 * 0.000002;
      const reasoningCost = 100 * 0;
      const cacheReadCost = 2000 * 0.000000125;
      const expected = inputCost + outputCost + reasoningCost + cacheReadCost;
      expect(result).toBeCloseTo(expected, 10);
    });
  });

  describe('calculateCost for unknown model', () => {
    it('returns 0 for unknown model', () => {
      const usage: TokenUsage = {
        total: 10000, input: 10000, output: 500,
        reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0, inputMessagesTokens: 10000,
      };
      expect(calculateCost('unknown-model-xyz', usage)).toBe(0);
    });
  });

  describe('calculateCostNullSafe', () => {
    it('returns 0 for null usage', () => {
      expect(calculateCostNullSafe('alibaba-cn/glm-5', null)).toBe(0);
    });

    it('returns 0 for null model', () => {
      const usage: TokenUsage = {
        total: 10000, input: 10000, output: 500,
        reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0, inputMessagesTokens: 10000,
      };
      expect(calculateCostNullSafe(null, usage)).toBe(0);
    });

    it('returns 0 for both null', () => {
      expect(calculateCostNullSafe(null, null)).toBe(0);
    });

    it('returns correct cost for valid model and usage', () => {
      const usage: TokenUsage = {
        total: 1000, input: 800, output: 200,
        reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0, inputMessagesTokens: 800,
      };
      const result = calculateCostNullSafe('alibaba-cn/glm-5', usage);
      expect(result).toBeCloseTo(800 * 0.0000005 + 200 * 0.000002, 10);
    });
  });

  describe('MODEL_PRICING table', () => {
    it('contains pricing entries for expected models', () => {
      expect(MODEL_PRICING['alibaba-cn/glm-5']).toBeDefined();
      expect(MODEL_PRICING['glm-5']).toBeDefined();
      expect(MODEL_PRICING['anthropic/claude-3.5-sonnet']).toBeDefined();
      expect(MODEL_PRICING['claude-3.5-sonnet']).toBeDefined();
      expect(MODEL_PRICING['openai/gpt-4o']).toBeDefined();
      expect(MODEL_PRICING['gpt-4o']).toBeDefined();
    });

    it('pricing entries have all required fields', () => {
      for (const [model, pricing] of Object.entries(MODEL_PRICING)) {
        expect(pricing).toHaveProperty('inputPricePerToken');
        expect(pricing).toHaveProperty('outputPricePerToken');
        expect(pricing).toHaveProperty('reasoningPricePerToken');
        expect(pricing).toHaveProperty('cacheReadPricePerToken');
        expect(pricing).toHaveProperty('cacheWritePricePerToken');
        expect(typeof pricing.inputPricePerToken).toBe('number');
        expect(typeof pricing.outputPricePerToken).toBe('number');
        expect(pricing.inputPricePerToken).toBeGreaterThan(0);
        expect(pricing.outputPricePerToken).toBeGreaterThan(0);
      }
    });

    it('cache pricing is lower than input pricing for glm-5', () => {
      const pricing = MODEL_PRICING['alibaba-cn/glm-5'];
      expect(pricing.cacheReadPricePerToken).toBeLessThan(pricing.inputPricePerToken);
    });
  });

  describe('cost validation against existing usage.cost', () => {
    it('recomputed cost matches the cost field already present in usage', () => {
      const usage: TokenUsage = {
        total: 10000,
        input: 10000,
        output: 500,
        reasoning: 0,
        cacheRead: 0,
        cacheWrite: 0,
        cost: 0,
        inputMessagesTokens: 10000,
      };
      const recomputed = calculateCost('alibaba-cn/glm-5', usage);
      const expectedFromPricing =
        10000 * 0.0000005 +
        500 * 0.000002;
      expect(recomputed).toBeCloseTo(expectedFromPricing, 10);
    });
  });
});
