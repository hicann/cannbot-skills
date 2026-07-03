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
import { KeyBar } from '@/cli/tui/components/KeyBar';

describe('KeyBar', () => {
  it('renders session keys', () => {
    const { getPlainText } = renderTui(<KeyBar screen="sessions" />);
    const output = getPlainText();
    expect(output).toContain('Nav');
    expect(output).toContain('Open');
    expect(output).toContain('Page');
    expect(output).toContain('Quit');
  });

  it('renders session detail keys', () => {
    const { getPlainText } = renderTui(<KeyBar screen="session" />);
    const output = getPlainText();
    expect(output).toContain('Tab');
    expect(output).toContain('Back');
  });

  it('renders help keys', () => {
    const { getPlainText } = renderTui(<KeyBar screen="help" />);
    const output = getPlainText();
    expect(output).toContain('Back');
    expect(output).toContain('Quit');
  });

  it('renders default keys for unknown screen', () => {
    const { getPlainText } = renderTui(<KeyBar screen="unknown" />);
    const output = getPlainText();
    expect(output).toContain('Nav');
  });
});
