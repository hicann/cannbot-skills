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
import { WorkflowTab } from '@/cli/tui/tabs/WorkflowTab';
import type { WorkflowTree } from '@/cli/types';

const mockWorkflow: WorkflowTree = {
  phases: [
    {
      phaseIndex: 0,
      phaseName: 'analysis',
      fullLabel: 'Phase 0: Analysis',
      startTime: null,
      endTime: null,
      durationMs: 5000,
      activeTimeMs: 3000,
      waitTimeMs: 2000,
      totalTokens: 1000,
      totalCost: 0.05,
      toolCallCount: 3,
      subagentCount: 0,
      triggerTurnId: null,
      children: [
        {
          type: 'step',
          stepIndex: 0,
          stepName: 'read',
          stepLabel: 'Read files',
          iterationIndex: null,
          iterationName: null,
          startTime: null,
          endTime: null,
          durationMs: 2000,
          totalTokens: 500,
          totalCost: 0.02,
          toolCallCount: 1,
          bridgeId: null,
          subagentSessionId: null,
          subagentType: null,
          subagentName: null,
          status: 'completed',
          parallelGroupId: null,
          triggerTurnId: null,
        },
      ],
    },
  ],
  summary: {
    totalPhases: 1,
    totalSteps: 1,
    totalCheckpoints: 0,
    totalActiveTimeMs: 3000,
    totalWaitTimeMs: 2000,
    activeTimePct: 60,
    iterations: 0,
  },
};

const mockFlatNodes = [
  { type: 'phase-header' as const, depth: 0, label: '▼ Phase 0: Analysis', detail: '5.0s │ 1.0K │ $0.05 │ 0 sub', color: 'cyan', phaseIndex: 0 },
  { type: 'step' as const, depth: 1, label: '✓ Step 0: Read files ▶', detail: '2.0s │ 0.5K │ 1 tools │ sub', color: 'green', stepKey: 'p0-s0' },
];

describe('WorkflowTab', () => {
  it('renders workflow tree', () => {
    const { getPlainText } = renderTui(
      <WorkflowTab
        workflow={mockWorkflow}
        allTurns={[]}
        sessionModel={null}
        flatNodes={mockFlatNodes}
        cursorIndex={0}
        selectedTurnId={null}
      />
    );
    const output = getPlainText();
    expect(output).toContain('Workflow');
    expect(output).toContain('Phase 0');
    expect(output).toContain('Analysis');
  });

  it('renders summary info', () => {
    const { getPlainText } = renderTui(
      <WorkflowTab
        workflow={mockWorkflow}
        allTurns={[]}
        sessionModel={null}
        flatNodes={mockFlatNodes}
        cursorIndex={0}
        selectedTurnId={null}
      />
    );
    const output = getPlainText();
    expect(output).toContain('1 phases');
    expect(output).toContain('1 steps');
  });
});
