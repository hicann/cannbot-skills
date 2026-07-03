// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import React from 'react';
import { renderTui } from '../../helpers/render-tui';
import { Box, Text } from 'ink';
import { useKeyboard } from '@/cli/hooks/useKeyboard';

function TestKeyboardDisplay({ handlers, active }: { handlers: Record<string, () => void>; active: boolean }) {
  useKeyboard(handlers, active);
  return <Text>keyboard:{active ? 'on' : 'off'}</Text>;
}

describe('useKeyboard', () => {
  it('renders component with keyboard active', () => {
    const handlers = { onNavigateUp: vi.fn() };
    const { getPlainText } = renderTui(<TestKeyboardDisplay handlers={handlers} active={true} />);
    const output = getPlainText();
    expect(output).toContain('keyboard:on');
  });

  it('renders component with keyboard inactive', () => {
    const handlers = { onNavigateUp: vi.fn() };
    const { getPlainText } = renderTui(<TestKeyboardDisplay handlers={handlers} active={false} />);
    const output = getPlainText();
    expect(output).toContain('keyboard:off');
  });
});
