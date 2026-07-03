// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';

interface KeyBarProps {
  screen: string;
}

const SCREEN_KEYS: Record<string, Array<{ key: string; label: string }>> = {
  sessions: [
    { key: '↑↓', label: 'Nav' },
    { key: 'Enter', label: 'Open' },
    { key: 'n/p', label: 'Page' },
    { key: '/', label: 'Search' },
    { key: 'Space', label: 'Mark' },
    { key: 'c', label: 'Cmp2' },
    { key: 'i', label: 'Import' },
    { key: 'u', label: 'Upload' },
    { key: 'd', label: 'Del' },
    { key: 'r', label: 'Refresh' },
    { key: 'q', label: 'Quit' },
  ],
  session: [
    { key: '←→', label: 'Tab' },
    { key: '↑↓', label: 'Navigate' },
    { key: 'Enter', label: 'Select' },
    { key: 'Esc', label: 'Back' },
    { key: 'r', label: 'Refresh' },
    { key: 'q', label: 'Quit' },
  ],
  turn: [
    { key: '↑↓', label: 'Scroll' },
    { key: 'Esc', label: 'Back' },
    { key: 'r', label: 'Refresh' },
    { key: 'q', label: 'Quit' },
  ],
  compare: [
    { key: '↑↓', label: 'Navigate' },
    { key: 'Esc', label: 'Back' },
    { key: 'r', label: 'Refresh' },
    { key: 'q', label: 'Quit' },
  ],
  import: [
    { key: '↑↓', label: 'Navigate' },
    { key: 'Enter', label: 'Select' },
    { key: 'Esc', label: 'Back' },
    { key: 'q', label: 'Quit' },
  ],
  help: [
    { key: 'Esc', label: 'Back' },
    { key: 'q', label: 'Quit' },
  ],
};

export function KeyBar({ screen }: KeyBarProps) {
  const keys = SCREEN_KEYS[screen] ?? SCREEN_KEYS.sessions;

  return (
    <Box borderStyle="single" borderTop={true} borderBottom={false} borderLeft={false} borderRight={false} paddingX={1}>
      {keys.map((k, i) => (
        <Box key={i}>
          {i > 0 ? <Text color="gray"> │ </Text> : null}
          <Text bold color="cyan">{k.key}</Text>
          <Text color="gray">:{k.label}</Text>
        </Box>
      ))}
    </Box>
  );
}
