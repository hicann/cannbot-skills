// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';

interface AsciiBarProps {
  label: string;
  value: number;
  max: number;
  width?: number;
  warningThreshold?: number;
  criticalThreshold?: number;
  unit?: string;
}

export function AsciiBar({ label, value, max, width = 20, warningThreshold, criticalThreshold, unit = '' }: AsciiBarProps) {
  const pct = max > 0 ? value / max : 0;
  const filled = Math.round(pct * width);
  const empty = width - filled;

  let color: string = 'green';
  if (criticalThreshold !== undefined && pct >= criticalThreshold) color = 'red';
  else if (warningThreshold !== undefined && pct >= warningThreshold) color = 'yellow';

  return (
    <Box flexDirection="row">
      <Text>{padEnd(label, 12)}</Text>
      <Text color={color}>{'█'.repeat(filled)}</Text>
      <Text color="gray">{'░'.repeat(empty)}</Text>
      <Text> {value}{unit}/{max}{unit} ({(pct * 100).toFixed(1)}%)</Text>
    </Box>
  );
}

function padEnd(str: string, width: number): string {
  if (str.length >= width) return str.slice(0, width);
  return str + ' '.repeat(width - str.length);
}
