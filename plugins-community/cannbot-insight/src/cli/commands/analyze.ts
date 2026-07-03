// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Command } from 'commander';
import { InsightClient } from '../client';
import { formatHeader, formatDivider, formatLabel } from '../utils/colors';
import { formatTokens, formatCost, formatDuration } from '../utils/format';
import { BRAND_SLUG } from '@/lib/branding';

export function analyzeCommand(): Command {
  const cmd = new Command('analyze');
  cmd
    .description('AI-powered workflow analysis for a session')
    .argument('<taskId>', 'Session task ID')
    .option('--base-url <url>', 'AI provider base URL')
    .option('--api-key <key>', 'AI provider API key')
    .option('--model <model>', 'AI model to use', 'gpt-4o-mini')
    .option('--json', 'Output as JSON')
    .action(async (taskId, opts, command) => {
      const globalOpts = command.parent?.opts() ?? {};
      const client = new InsightClient(globalOpts.server, {
        timeout: +globalOpts.timeout,
      });

      if (!opts.baseUrl || !opts.apiKey) {
        console.error('Error: --base-url and --api-key are required for AI analysis');
        console.error(`Example: ${BRAND_SLUG} analyze <taskId> --base-url https://api.openai.com/v1 --api-key sk-...`);
        process.exit(1);
      }

      const provider = {
        baseUrl: opts.baseUrl,
        apiKey: opts.apiKey,
        model: opts.model,
      };

      console.log(formatHeader(`AI Workflow Analysis: ${taskId}`));
      console.log(formatDivider());
      console.log('');
      console.log(formatLabel('Provider', opts.baseUrl));
      console.log(formatLabel('Model', opts.model));
      console.log('');
      console.log('Analyzing...');

      const response = await client.analyzeWorkflow(taskId, provider);

      if (opts.json) {
        console.log(JSON.stringify(response, null, 2));
        return;
      }

      const workflow = response.result;

      const lines: string[] = [];
      lines.push(formatHeader('Workflow Analysis Result'));
      lines.push(formatDivider());
      lines.push('');
      lines.push(formatLabel('Phases', String(workflow.phases.length)));
      lines.push(formatLabel('Steps', String(workflow.summary.totalSteps)));
      lines.push(formatLabel('Checkpoints', String(workflow.summary.totalCheckpoints)));
      lines.push(formatLabel('Active Time', formatDuration(workflow.summary.totalActiveTimeMs)));
      lines.push(formatLabel('Wait Time', formatDuration(workflow.summary.totalWaitTimeMs)));
      lines.push(formatLabel('Active %', `${workflow.summary.activeTimePct}%`));
      lines.push('');

      for (const phase of workflow.phases) {
        lines.push(formatHeader(`Phase ${phase.phaseIndex}: ${phase.phaseName}`));
        lines.push(formatLabel('Duration', formatDuration(phase.durationMs)));
        lines.push(formatLabel('Tokens', formatTokens(phase.totalTokens)));
        lines.push(formatLabel('Cost', formatCost(phase.totalCost)));
        lines.push(formatLabel('Tool Calls', String(phase.toolCallCount)));
        lines.push('');

        for (const child of phase.children) {
          if (child.type === 'step') {
            lines.push(`  Step ${child.stepIndex}: ${child.stepLabel} (${child.status})`);
            lines.push(`    Duration: ${formatDuration(child.durationMs)} │ Tokens: ${formatTokens(child.totalTokens)}`);
          } else if (child.type === 'checkpoint') {
            lines.push(`  Checkpoint ${child.checkpointIndex}: ${child.checkpointLabel}`);
            lines.push(`    Wait: ${formatDuration(child.waitTimeMs)}`);
          }
        }
        lines.push('');
      }

      console.log(lines.join('\n'));
    });

  return cmd;
}
