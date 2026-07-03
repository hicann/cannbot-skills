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
import { TabBar } from '@/cli/tui/components/TabBar';

describe('TabBar', () => {
  const tabs = [
    { key: 'overview', label: 'Overview', icon: '📋' },
    { key: 'turns', label: 'Turns', icon: '🔄' },
    { key: 'workflow', label: 'Workflow', icon: '✦' },
  ];

  it('renders all tab labels', () => {
    const { getPlainText } = renderTui(<TabBar tabs={tabs} activeKey="overview" />);
    const output = getPlainText();
    expect(output).toContain('Overview');
    expect(output).toContain('Turns');
    expect(output).toContain('Workflow');
  });

  it('renders tabs with icons', () => {
    const { getPlainText } = renderTui(<TabBar tabs={tabs} activeKey="overview" />);
    const output = getPlainText();
    expect(output).toContain('📋');
    expect(output).toContain('🔄');
    expect(output).toContain('✦');
  });

  it('renders tabs without icons', () => {
    const plainTabs = [
      { key: 'a', label: 'Alpha' },
      { key: 'b', label: 'Beta' },
    ];
    const { getPlainText } = renderTui(<TabBar tabs={plainTabs} activeKey="a" />);
    const output = getPlainText();
    expect(output).toContain('Alpha');
    expect(output).toContain('Beta');
  });

  it('highlights active tab', () => {
    const { lastFrame } = renderTui(<TabBar tabs={tabs} activeKey="turns" />);
    const output = lastFrame();
    expect(output).toContain('Turns');
  });
});
