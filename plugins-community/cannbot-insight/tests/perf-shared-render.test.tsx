// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// @vitest-environment happy-dom

// Regression guard for the duplicate-key React warning on PerfBenchmarkChart.
// When all values are 0, yTicks = round(yMax*i/3) with yMax clamped to 1 →
// [0,0,1,1]. Keying grid lines by VALUE collides (two 0s, two 1s); React's
// dev build emits "Encountered two children with the same key" during a
// client render. We use createRoot + act (not SSR — SSR doesn't warn) to
// capture the console.error, and assert none mentions duplicate keys.

import { describe, it, expect, vi } from 'vitest';
import { createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react';
import { PerfBenchmarkChart } from '@/components/observe/perf-shared';

describe('PerfBenchmarkChart', () => {
  it('renders without duplicate-key warnings when all values are zero', () => {
    const errors: string[] = [];
    const spy = vi.spyOn(console, 'error').mockImplementation((...a: unknown[]) => {
      errors.push(a.map(String).join(' '));
    });
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    const points = [
      { turnIndex: 0, createdAt: '2026-08-13T11:00:00Z', tpot: 0, tps: 0 },
      { turnIndex: 1, createdAt: '2026-08-13T11:01:00Z', tpot: 0, tps: 0 },
      { turnIndex: 2, createdAt: '2026-08-13T11:02:00Z', tpot: 0, tps: 0 },
    ];
    act(() => { root.render(createElement(PerfBenchmarkChart, { points })); });
    const dupKey = errors.find(e => e.includes('two children with the same key'));
    expect(dupKey ? `duplicate-key warning: ${dupKey}` : undefined).toBeUndefined();
    spy.mockRestore();
    document.body.removeChild(container);
  });
});
