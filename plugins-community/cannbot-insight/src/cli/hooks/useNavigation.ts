// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState, useCallback } from 'react';

export type ScreenName = 'sessions' | 'session' | 'turn' | 'compare' | 'import' | 'help';

export interface NavigationState {
  screen: ScreenName;
  stack: Array<{ screen: ScreenName; params?: Record<string, string> }>;
  params?: Record<string, string>;
}

export function useNavigation() {
  const [nav, setNav] = useState<NavigationState>({ screen: 'sessions', stack: [] });

  const navigate = useCallback((screen: ScreenName, params?: Record<string, string>) => {
    setNav(prev => ({
      screen,
      params,
      stack: [...prev.stack, { screen: prev.screen, params: prev.params }],
    }));
  }, []);

  const goBack = useCallback(() => {
    setNav(prev => {
      if (prev.stack.length === 0) return prev;
      const last = prev.stack[prev.stack.length - 1];
      return {
        screen: last.screen,
        params: last.params,
        stack: prev.stack.slice(0, -1),
      };
    });
  }, []);

  const reset = useCallback(() => {
    setNav({ screen: 'sessions', stack: [] });
  }, []);

  return { nav, navigate, goBack, reset };
}
