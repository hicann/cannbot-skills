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
import { AsciiBar } from '@/cli/tui/components/AsciiBar';

describe('AsciiBar', () => {
  it('renders bar with filled and empty segments', () => {
    const { getPlainText } = renderTui(<AsciiBar label="Test" value={50} max={100} width={10} />);
    const output = getPlainText();
    expect(output).toContain('Test');
    expect(output).toContain('█');
    expect(output).toContain('░');
    expect(output).toContain('50.0%');
  });

  it('renders full bar when value equals max', () => {
    const { getPlainText } = renderTui(<AsciiBar label="Full" value={100} max={100} width={10} />);
    const output = getPlainText();
    expect(output).toContain('100.0%');
  });

  it('renders zero bar when value is 0', () => {
    const { getPlainText } = renderTui(<AsciiBar label="Empty" value={0} max={100} width={10} />);
    const output = getPlainText();
    expect(output).toContain('0.0%');
  });

  it('applies warning threshold color', () => {
    const { lastFrame } = renderTui(<AsciiBar label="Warn" value={75} max={100} width={10} warningThreshold={0.7} />);
    const output = lastFrame();
    expect(output).toContain('Warn');
    expect(output).toContain('75.0%');
  });

  it('applies critical threshold color', () => {
    const { lastFrame } = renderTui(<AsciiBar label="Crit" value={95} max={100} width={10} criticalThreshold={0.9} />);
    const output = lastFrame();
    expect(output).toContain('Crit');
    expect(output).toContain('95.0%');
  });

  it('renders with unit', () => {
    const { getPlainText } = renderTui(<AsciiBar label="Memory" value={500} max={1000} width={10} unit="MB" />);
    const output = getPlainText();
    expect(output).toContain('500MB');
    expect(output).toContain('1000MB');
  });
});
