// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import { DataTable, DataTableColumn } from '@/cli/tui/components/DataTable';
import type { ApiSkillSummary, ApiTurnItem } from '@/cli/types';
import { formatDuration } from '@/cli/utils/format';

interface SkillsTabProps {
  skills: ApiSkillSummary[];
  turns: ApiTurnItem[];
  selectedIndex: number;
}

const SKILL_COLUMNS: DataTableColumn<ApiSkillSummary>[] = [
  { key: 'skillName', label: 'Skill', width: 30 },
  { key: 'version', label: 'Version', width: 10 },
  { key: 'invocationCount', label: 'Invocations', width: 12 },
];

interface SkillAggregate {
  skillName: string;
  version: string | null;
  invocationCount: number;
  loadCount: number;
  invokeCount: number;
  failCount: number;
  avgDurationMs: number;
}

function aggregateSkillEvents(skills: ApiSkillSummary[], turns: ApiTurnItem[]): SkillAggregate[] {
  const allEvents = turns.flatMap(t => t.skillEvents ?? []);

  return skills.map(ss => {
    const events = allEvents.filter(se => se.skillName === ss.skillName);
    const invokeEvents = events.filter(se => se.eventType === 'invoke' || se.eventType === 'use');
    const loadEvents = events.filter(se => se.eventType === 'load');
    const failCount = events.filter(se => !se.success).length;
    const avgDuration = invokeEvents.length > 0
      ? Math.round(invokeEvents.reduce((sum, e) => sum + 0, 0) / invokeEvents.length)
      : 0;

    return {
      skillName: ss.skillName,
      version: ss.version,
      invocationCount: ss.invocationCount,
      loadCount: loadEvents.length,
      invokeCount: invokeEvents.length,
      failCount,
      avgDurationMs: avgDuration,
    };
  });
}

export function SkillsTab({ skills, turns, selectedIndex }: SkillsTabProps) {
  const aggregates = aggregateSkillEvents(skills, turns);

  return (
    <Box flexDirection="column" gap={1}>
      <Text bold color="cyan">Skills Summary ({skills.length})</Text>
      <DataTable
        columns={SKILL_COLUMNS}
        data={skills}
        selectedIndex={selectedIndex}
      />

      {aggregates.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="cyan">Skill Events</Text>
          {aggregates.map((agg, i) => (
            <Text key={i}>
              {agg.skillName} │ v{agg.version ?? '?'} │ load:{agg.loadCount} │ invoke:{agg.invokeCount}{agg.failCount > 0 ? ` │ ✗${agg.failCount} fail` : ' │ ✓all'}{agg.avgDurationMs > 0 ? ` │ avg ${formatDuration(agg.avgDurationMs)}` : ''}
            </Text>
          ))}
        </Box>
      )}
    </Box>
  );
}

export { SKILL_COLUMNS };
