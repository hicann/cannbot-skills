// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useInput } from 'ink';

interface KeyboardHandlers {
  onNavigateUp?: () => void;
  onNavigateDown?: () => void;
  onEnter?: () => void;
  onEscape?: () => void;
  onTab?: () => void;
  onRefresh?: () => void;
  onQuit?: () => void;
  onSearch?: () => void;
  onPageNext?: () => void;
  onPagePrev?: () => void;
  custom?: Record<string, () => void>;
}

export function useKeyboard(handlers: KeyboardHandlers, active = true) {
  useInput((input, key) => {
    if (!active) return;

    if (key.upArrow) handlers.onNavigateUp?.();
    else if (key.downArrow) handlers.onNavigateDown?.();
    else if (key.return) handlers.onEnter?.();
    else if (key.escape) handlers.onEscape?.();
    else if (key.tab) handlers.onTab?.();
    else if (key.ctrl && input === 'c') handlers.onQuit?.();
    else if (input === 'q') handlers.onQuit?.();
    else if (input === 'r') handlers.onRefresh?.();
    else if (input === '/') handlers.onSearch?.();
    else if (input === 'n') handlers.onPageNext?.();
    else if (input === 'p') handlers.onPagePrev?.();
    else if (handlers.custom?.[input]) handlers.custom[input]();
  }, { isActive: active });
}
