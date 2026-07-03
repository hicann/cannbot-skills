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
import { Spinner } from '@/cli/tui/components/Spinner';

describe('Spinner', () => {
  it('renders with default label', () => {
    const { getPlainText } = renderTui(<Spinner />);
    const output = getPlainText();
    expect(output).toContain('Loading...');
  });

  it('renders with custom label', () => {
    const { getPlainText } = renderTui(<Spinner label="Fetching data..." />);
    const output = getPlainText();
    expect(output).toContain('Fetching data...');
  });

  it('renders a spinner frame character', () => {
    const { getPlainText } = renderTui(<Spinner label="Test" />);
    const output = getPlainText();
    const frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    const hasFrame = frames.some(f => output.includes(f));
    expect(hasFrame).toBe(true);
  });
});
