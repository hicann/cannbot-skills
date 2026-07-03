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
import { TreeView } from '@/cli/tui/components/TreeView';
import type { WorkflowPhaseNode } from '@/cli/types';

describe('TreeView', () => {
  it('renders empty phases gracefully', () => {
    const { getPlainText } = renderTui(<TreeView phases={[]} />);
    expect(getPlainText()).toBeDefined();
  });

  it('renders single phase with steps', () => {
    const phases: WorkflowPhaseNode[] = [
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
    ];
    const { getPlainText } = renderTui(<TreeView phases={phases} />);
    const output = getPlainText();
    expect(output).toContain('Phase 0');
    expect(output).toContain('Analysis');
    expect(output).toContain('Read files');
    expect(output).toContain('✓');
  });

  it('renders phase with failed step', () => {
    const phases: WorkflowPhaseNode[] = [
      {
        phaseIndex: 0,
        phaseName: 'test',
        fullLabel: 'Phase 0: Test',
        startTime: null,
        endTime: null,
        durationMs: 3000,
        activeTimeMs: 3000,
        waitTimeMs: 0,
        totalTokens: 200,
        totalCost: 0.01,
        toolCallCount: 1,
        subagentCount: 0,
        triggerTurnId: null,
        children: [
          {
            type: 'step',
            stepIndex: 0,
            stepName: 'run',
            stepLabel: 'Run test',
            iterationIndex: null,
            iterationName: null,
            startTime: null,
            endTime: null,
            durationMs: 1000,
            totalTokens: 100,
            totalCost: 0.01,
            toolCallCount: 0,
            bridgeId: null,
            subagentSessionId: null,
            subagentType: null,
            subagentName: null,
            status: 'failed',
            parallelGroupId: null,
            triggerTurnId: null,
          },
        ],
      },
    ];
    const { getPlainText } = renderTui(<TreeView phases={phases} />);
    const output = getPlainText();
    expect(output).toContain('✗');
  });

  it('renders checkpoint', () => {
    const phases: WorkflowPhaseNode[] = [
      {
        phaseIndex: 0,
        phaseName: 'review',
        fullLabel: 'Phase 0: Review',
        startTime: null,
        endTime: null,
        durationMs: 10000,
        activeTimeMs: 5000,
        waitTimeMs: 5000,
        totalTokens: 1000,
        totalCost: 0.05,
        toolCallCount: 2,
        subagentCount: 0,
        triggerTurnId: null,
        children: [
          {
            type: 'checkpoint',
            checkpointIndex: 0,
            checkpointType: 'block',
            checkpointLabel: 'Confirm changes',
            requestedAt: null,
            approvedAt: null,
            waitTimeMs: 5000,
            triggerTurnId: null,
            responseTurnId: null,
          },
        ],
      },
    ];
    const { getPlainText } = renderTui(<TreeView phases={phases} />);
    const output = getPlainText();
    expect(output).toContain('Checkpoint');
    expect(output).toContain('Confirm changes');
  });
});
