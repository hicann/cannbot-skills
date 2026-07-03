// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import React from 'react';
import { renderTui } from '../../helpers/render-tui';
import { MetricCards } from '@/cli/tui/components/MetricCards';

describe('MetricCards', () => {
  it('renders metric cards with labels and values', () => {
    const cards = [
      { label: 'Tokens', value: '1.2K' },
      { label: 'Cost', value: '$0.05' },
      { label: 'Duration', value: '5.3s' },
    ];
    const { getPlainText } = renderTui(<MetricCards cards={cards} />);
    const output = getPlainText();
    expect(output).toContain('Tokens');
    expect(output).toContain('1.2K');
    expect(output).toContain('Cost');
    expect(output).toContain('$0.05');
    expect(output).toContain('Duration');
    expect(output).toContain('5.3s');
  });

  it('renders with custom colors', () => {
    const cards = [
      { label: 'Test', value: '100', color: 'cyan' },
    ];
    const { getPlainText } = renderTui(<MetricCards cards={cards} />);
    const output = getPlainText();
    expect(output).toContain('Test');
    expect(output).toContain('100');
  });

  it('renders empty cards array gracefully', () => {
    const { getPlainText } = renderTui(<MetricCards cards={[]} />);
    expect(getPlainText()).toBeDefined();
  });
});
