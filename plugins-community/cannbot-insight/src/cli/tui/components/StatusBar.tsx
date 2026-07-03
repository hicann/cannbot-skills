// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import { VERSION_DISPLAY } from '@/lib/version';
import type { InsightClient } from '@/cli/client';

interface StatusBarProps {
  client: InsightClient;
  screen: string;
}

export function StatusBar({ client, screen }: StatusBarProps) {
  const baseUrl = client.getConfig().baseUrl;

  return (
    <Box borderStyle="single" borderBottom={true} borderTop={false} borderLeft={false} borderRight={false} paddingX={1}>
      <Text bold color="cyan">{VERSION_DISPLAY}</Text>
      <Text> │ </Text>
      <Text color="yellow">{screen}</Text>
      <Text> │ </Text>
      <Text color="gray">{baseUrl}</Text>
    </Box>
  );
}
