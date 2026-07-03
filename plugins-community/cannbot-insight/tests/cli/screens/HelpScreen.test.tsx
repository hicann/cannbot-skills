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
import { HelpScreen } from '@/cli/tui/screens/HelpScreen';
import { BRAND_NAME } from '@/lib/branding';

describe('HelpScreen', () => {
  it('renders help content', () => {
    const { getPlainText } = renderTui(<HelpScreen onBack={() => {}} />);
    const output = getPlainText();
    expect(output).toContain('Navigation');
    expect(output).toContain('↑↓');
    expect(output).toContain('Enter');
    expect(output).toContain('Esc');
    expect(output).toContain('Global');
    expect(output).toContain('Quit');
    expect(output).toContain('Press Esc to go back');
  });

  it('renders version info', () => {
    const { getPlainText } = renderTui(<HelpScreen onBack={() => {}} />);
    const output = getPlainText();
    expect(output).toContain(BRAND_NAME);
  });
});
