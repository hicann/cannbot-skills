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
import { useNavigation } from '@/cli/hooks/useNavigation';
import { Box, Text } from 'ink';

function TestNavComponent() {
  const { nav } = useNavigation();
  return (
    <Box flexDirection="column">
      <Text>Screen: {nav.screen}</Text>
      <Text>Params: {JSON.stringify(nav.params)}</Text>
      <Text>Stack: {nav.stack.length}</Text>
    </Box>
  );
}

describe('useNavigation', () => {
  it('initial state is sessions screen', () => {
    const { getPlainText } = renderTui(<TestNavComponent />);
    const output = getPlainText();
    expect(output).toContain('sessions');
    expect(output).toContain('Stack: 0');
  });

  it('navigates to new screen with params', () => {
    function TestNavNavigate() {
      const { nav, navigate } = useNavigation();
      React.useEffect(() => {
        navigate('session', { taskId: 'test-123' });
      }, []);
      return <Text>Screen: {nav.screen} Params: {nav.params?.taskId ?? 'none'}</Text>;
    }
    const { getPlainText } = renderTui(<TestNavNavigate />);
    const output = getPlainText();
    expect(output).toContain('session');
  });

  it('goBack returns to previous screen', () => {
    function TestNavGoBack() {
      const { nav, navigate, goBack } = useNavigation();
      React.useEffect(() => {
        navigate('session', { taskId: 't1' });
        goBack();
      }, []);
      return <Text>Screen: {nav.screen}</Text>;
    }
    const { getPlainText } = renderTui(<TestNavGoBack />);
    const output = getPlainText();
    expect(output).toContain('sessions');
  });

  it('reset goes back to sessions', () => {
    function TestNavReset() {
      const { nav, navigate, reset } = useNavigation();
      React.useEffect(() => {
        navigate('session');
        navigate('turn');
        reset();
      }, []);
      return <Text>Screen: {nav.screen}</Text>;
    }
    const { getPlainText } = renderTui(<TestNavReset />);
    const output = getPlainText();
    expect(output).toContain('sessions');
  });
});
