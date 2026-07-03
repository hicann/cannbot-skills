// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import React from 'react';
import { render } from 'ink-testing-library';

export function renderTui(element: React.ReactElement) {
  const instance = render(element);
  return {
    ...instance,
    lastFrame: () => instance.lastFrame(),
    getPlainText: () => instance.lastFrame()?.replace(/\x1b\[[0-9;]*m/g, '') ?? '',
    pressKey: (key: string) => instance.stdin.write(key),
    pressEnter: () => instance.stdin.write('\r'),
    pressEscape: () => instance.stdin.write('\x1b'),
    pressUp: () => instance.stdin.write('\x1b[A'),
    pressDown: () => instance.stdin.write('\x1b[B'),
    pressCtrlC: () => instance.stdin.write('\x03'),
  };
}
