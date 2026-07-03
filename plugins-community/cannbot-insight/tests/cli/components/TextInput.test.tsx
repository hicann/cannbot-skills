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
import { TextInput } from '@/cli/tui/components/TextInput';

describe('TextInput', () => {
  it('renders value and cursor', () => {
    const { getPlainText } = renderTui(<TextInput value="hello" onChange={() => {}} />);
    const output = getPlainText();
    expect(output).toContain('hello');
    expect(output).toContain('█');
  });

  it('renders placeholder when value is empty', () => {
    const { getPlainText } = renderTui(<TextInput value="" onChange={() => {}} placeholder="Enter text..." />);
    const output = getPlainText();
    expect(output).toContain('Enter text...');
  });

  it('does not render placeholder when value has content', () => {
    const { getPlainText } = renderTui(<TextInput value="abc" onChange={() => {}} placeholder="Enter text..." />);
    const output = getPlainText();
    expect(output).not.toContain('Enter text...');
    expect(output).toContain('abc');
  });

  it('calls onChange when typing', () => {
    const onChange = vi.fn();
    const { pressKey } = renderTui(<TextInput value="" onChange={onChange} />);
    pressKey('t');
    expect(onChange).toHaveBeenCalledWith('t');
  });
});
