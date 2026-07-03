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
import { BridgesTab } from '@/cli/tui/tabs/BridgesTab';
import type { ApiBridgeItem } from '@/cli/types';

const mockBridges: ApiBridgeItem[] = [
  {
    bridgeId: 'b1', dispatchExecutionId: 'e0', dispatchTurnId: 't1',
    dispatchToolCallId: 'tc1', dispatchContent: 'fix bug', dispatchTimestamp: '2026-06-14T10:00:00Z',
    responseExecutionId: 'e1', responseTurnId: 't2', responseContent: 'done',
    responseTimestamp: '2026-06-14T10:01:00Z', subagentSessionId: 's1',
    subagentType: 'coding', subagentName: 'coder', status: 'completed',
    subagentTokens: 2000, subagentLatencyMs: 60000,
  },
];

describe('BridgesTab', () => {
  it('renders bridges count header', () => {
    const { getPlainText } = renderTui(<BridgesTab bridges={mockBridges} selectedIndex={0} />);
    const output = getPlainText();
    expect(output).toContain('Bridges (1)');
  });

  it('renders bridge data rows', () => {
    const { getPlainText } = renderTui(<BridgesTab bridges={mockBridges} selectedIndex={0} />);
    const output = getPlainText();
    expect(output).toContain('completed');
    expect(output).toContain('coder');
  });

  it('renders empty bridges', () => {
    const { getPlainText } = renderTui(<BridgesTab bridges={[]} selectedIndex={0} />);
    const output = getPlainText();
    expect(output).toContain('Bridges (0)');
  });
});
