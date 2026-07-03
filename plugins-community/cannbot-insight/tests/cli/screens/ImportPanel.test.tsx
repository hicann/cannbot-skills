// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi } from 'vitest';
import React from 'react';
import { renderTui } from '../../helpers/render-tui';
import { ImportPanel } from '@/cli/tui/screens/ImportPanel';
import { InsightClient } from '@/cli/client';

vi.mock('@/cli/hooks/useApi', () => ({
  useApi: () => ({
    data: { sessions: [] },
    loading: false,
    refresh: vi.fn(),
  }),
}));

vi.mock('@/cli/hooks/useKeyboard', () => ({
  useKeyboard: () => {},
}));

describe('ImportPanel', () => {
  it('renders import header', () => {
    const client = new InsightClient('http://localhost:21025');
    const { getPlainText } = renderTui(
      <ImportPanel client={client} onBack={vi.fn()} />
    );
    const output = getPlainText();
    expect(output).toContain('Import Sessions');
  });

  it('renders file path input section', () => {
    const client = new InsightClient('http://localhost:21025');
    const { getPlainText } = renderTui(
      <ImportPanel client={client} onBack={vi.fn()} />
    );
    const output = getPlainText();
    expect(output).toContain('File path');
  });

  it('renders source type display', () => {
    const client = new InsightClient('http://localhost:21025');
    const { getPlainText } = renderTui(
      <ImportPanel client={client} onBack={vi.fn()} />
    );
    const output = getPlainText();
    expect(output).toContain('Source');
    expect(output).toContain('opencode-db');
  });

  it('renders prompt when no file path', () => {
    const client = new InsightClient('http://localhost:21025');
    const { getPlainText } = renderTui(
      <ImportPanel client={client} onBack={vi.fn()} />
    );
    const output = getPlainText();
    expect(output).toContain('Enter a file or directory path');
  });
});
