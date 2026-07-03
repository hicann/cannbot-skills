// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { analyzeCommand } from '@/cli/commands/analyze';
import { InsightClient } from '@/cli/client';
import type { WorkflowTree, ApiAnalyzeWorkflowResponse } from '@/cli/types';

const mockAnalyzeWorkflow = vi.fn();
vi.spyOn(InsightClient.prototype, 'analyzeWorkflow').mockImplementation(mockAnalyzeWorkflow);

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

const mockResponse: ApiAnalyzeWorkflowResponse = {
  result: mockWorkflow,
};

describe('analyzeCommand', () => {
  beforeEach(() => {
    mockAnalyzeWorkflow.mockReset();
    mockAnalyzeWorkflow.mockResolvedValue(mockResponse);
  });

  it('registers as a Commander sub-command', () => {
    const cmd = analyzeCommand();
    expect(cmd.name()).toBe('analyze');
  });

  it('has required options', () => {
    const cmd = analyzeCommand();
    const options = cmd.options.map(o => o.long);
    expect(options).toContain('--base-url');
    expect(options).toContain('--api-key');
    expect(options).toContain('--model');
    expect(options).toContain('--json');
  });

  it('has taskId argument', () => {
    const cmd = analyzeCommand();
    expect(cmd._args.length).toBeGreaterThan(0);
  });

  it('formats workflow analysis response', () => {
    const w = mockWorkflow;
    expect(w.phases.length).toBe(1);
    expect(w.summary.totalSteps).toBe(1);
    expect(w.summary.activeTimePct).toBe(60);
    expect(w.phases[0].phaseName).toBe('analysis');
  });

  it('JSON output structure', () => {
    const json = JSON.stringify(mockResponse, null, 2);
    const parsed = JSON.parse(json);
    expect(parsed.result.phases.length).toBe(1);
    expect(parsed.result.summary.totalPhases).toBe(1);
  });
});
