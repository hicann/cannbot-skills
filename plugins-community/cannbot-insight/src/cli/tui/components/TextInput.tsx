// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text, useInput } from 'ink';

interface TextInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit?: (value: string) => void;
  placeholder?: string;
  focus?: boolean;
}

export function TextInput({ value, onChange, onSubmit, placeholder, focus = true }: TextInputProps) {
  useInput((input, key) => {
    if (!focus) return;
    if (key.backspace || key.delete) {
      onChange(value.slice(0, -1));
    } else if (key.return) {
      onSubmit?.(value);
    } else if (!key.ctrl && !key.meta) {
      onChange(value + input);
    }
  }, { isActive: focus });

  return (
    <Box>
      {value.length === 0 && placeholder ? <Text color="gray">{placeholder}</Text> : null}
      <Text>{value}</Text>
      <Text color="gray">█</Text>
    </Box>
  );
}
