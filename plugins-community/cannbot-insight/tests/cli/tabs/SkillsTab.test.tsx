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
import { SkillsTab } from '@/cli/tui/tabs/SkillsTab';
import type { ApiSkillSummary, ApiTurnItem } from '@/cli/types';

const mockSkills: ApiSkillSummary[] = [
  { skillName: 'agent-debug', version: '0.4', invocationCount: 5 },
  { skillName: 'find-skills', version: '1.0', invocationCount: 2 },
];

const mockTurns: ApiTurnItem[] = [];

describe('SkillsTab', () => {
  it('renders skills count header', () => {
    const { getPlainText } = renderTui(<SkillsTab skills={mockSkills} turns={mockTurns} selectedIndex={0} />);
    const output = getPlainText();
    expect(output).toContain('Skills Summary (2)');
  });

  it('renders skill names', () => {
    const { getPlainText } = renderTui(<SkillsTab skills={mockSkills} turns={mockTurns} selectedIndex={0} />);
    const output = getPlainText();
    expect(output).toContain('agent-debug');
    expect(output).toContain('find-skills');
  });

  it('renders empty skills', () => {
    const { getPlainText } = renderTui(<SkillsTab skills={[]} turns={mockTurns} selectedIndex={0} />);
    const output = getPlainText();
    expect(output).toContain('Skills Summary (0)');
  });
});
