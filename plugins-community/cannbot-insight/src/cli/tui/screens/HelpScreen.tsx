// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import { useKeyboard } from '@/cli/hooks/useKeyboard';
import { VERSION_DISPLAY } from '@/lib/version';
import { BRAND_NAME } from '@/lib/branding';

interface HelpScreenProps {
  onBack: () => void;
}

export function HelpScreen({ onBack }: HelpScreenProps) {
  useKeyboard({ onEscape: onBack });

  return (
    <Box flexDirection="column" padding={1}>
      <Text bold color="cyan">{VERSION_DISPLAY} — {BRAND_NAME} TUI Help</Text>
      <Text></Text>
      <Text bold>Navigation</Text>
      <Text>  ↑↓      Move selection</Text>
      <Text>  Enter   Select / Open detail</Text>
      <Text>  Esc     Go back</Text>
      <Text>  ←→ / Tab  Switch tabs</Text>
      <Text></Text>
      <Text bold>Session List</Text>
      <Text>  n       Next page</Text>
      <Text>  p       Previous page</Text>
      <Text>  /       Toggle search/filter</Text>
      <Text>  Space   Mark for compare</Text>
      <Text>  c       Compare 2 marked sessions</Text>
      <Text>  i       Open import panel</Text>
      <Text>  d       Delete selected session</Text>
      <Text>  r       Refresh data</Text>
      <Text></Text>
      <Text bold>Global</Text>
      <Text>  q       Quit TUI</Text>
      <Text>  Ctrl+C  Quit TUI</Text>
      <Text>  ?       Help</Text>
      <Text></Text>
      <Text color="gray">Press Esc to go back</Text>
    </Box>
  );
}
