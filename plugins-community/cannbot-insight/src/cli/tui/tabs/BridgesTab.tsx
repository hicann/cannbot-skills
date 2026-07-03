// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import { DataTable, DataTableColumn } from '@/cli/tui/components/DataTable';
import type { ApiBridgeItem } from '@/cli/types';
import { formatTokens, formatDuration } from '@/cli/utils/format';

interface BridgesTabProps {
  bridges: ApiBridgeItem[];
  selectedIndex: number;
}

const BRIDGE_COLUMNS: DataTableColumn<ApiBridgeItem>[] = [
  { key: 'status', label: 'Status', width: 10 },
  { key: 'subagentName', label: 'Subagent', width: 20 },
  { key: 'subagentType', label: 'Type', width: 12 },
  { key: 'subagentTokens', label: 'Tokens', width: 8, render: (row) => formatTokens(row.subagentTokens) },
  { key: 'subagentLatencyMs', label: 'Latency', width: 10, render: (row) => formatDuration(row.subagentLatencyMs) },
  { key: 'dispatchContent', label: 'Dispatch', width: 30, render: (row) => row.dispatchContent ?? '—' },
];

export function BridgesTab({ bridges, selectedIndex }: BridgesTabProps) {
  return (
    <Box flexDirection="column">
      <Text bold color="cyan">Bridges ({bridges.length})</Text>
      <DataTable
        columns={BRIDGE_COLUMNS}
        data={bridges}
        selectedIndex={selectedIndex}
      />
    </Box>
  );
}

export { BRIDGE_COLUMNS };
