// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi } from 'vitest';
import React from 'react';
import { renderTui } from '../../helpers/render-tui';
import { useApi } from '@/cli/hooks/useApi';
import { Box, Text } from 'ink';

describe('useApi', () => {
  it('returns loading state initially then data', async () => {
    const fetcher = vi.fn().mockResolvedValue({ total: 42 });

    function TestApiComponent() {
      const { data, loading, error } = useApi(fetcher, ['test'], { cacheKey: 'test-api-1' });
      return (
        <Box flexDirection="column">
          <Text>loading:{String(loading)}</Text>
          <Text>data:{data ? JSON.stringify(data) : 'null'}</Text>
          <Text>error:{error ? error.message : 'null'}</Text>
        </Box>
      );
    }

    const { getPlainText } = renderTui(<TestApiComponent />);
    await vi.waitFor(() => {
      const output = getPlainText();
      expect(output).toContain('loading:false');
      expect(output).toContain('42');
    });
  });

  it('returns error on failed fetch', async () => {
    const fetcher = vi.fn().mockRejectedValue(new Error('Network error'));

    function TestApiErrorComponent() {
      const { data, loading, error } = useApi(fetcher, ['test'], { cacheKey: 'test-api-error-1' });
      return (
        <Box flexDirection="column">
          <Text>loading:{String(loading)}</Text>
          <Text>error:{error ? error.message : 'null'}</Text>
        </Box>
      );
    }

    const { getPlainText } = renderTui(<TestApiErrorComponent />);
    await vi.waitFor(() => {
      const output = getPlainText();
      expect(output).toContain('loading:false');
      expect(output).toContain('Network error');
    });
  });
});
