// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';

export interface TabItem {
  key: string;
  label: string;
  icon?: string;
}

interface TabBarProps {
  tabs: TabItem[];
  activeKey: string;
}

export function TabBar({ tabs, activeKey }: TabBarProps) {
  return (
    <Box flexDirection="row" gap={1}>
      {tabs.map(tab => {
        const isActive = tab.key === activeKey;
        const icon = tab.icon ?? '';
        const label = `${icon}${tab.label}`;
        return (
          <Box key={tab.key}>
            {isActive ? (
              <Text bold color="cyan" backgroundColor="gray">{` ${label} `}</Text>
            ) : (
              <Text color="gray">{` ${label} `}</Text>
            )}
          </Box>
        );
      })}
    </Box>
  );
}
