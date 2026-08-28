// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import zlib from 'node:zlib';
import { decompress as decompressZstd } from 'fzstd';
import type { Protocol } from './types';

function normalizeContentEncoding(value: string | string[] | undefined): string | null {
  if (Array.isArray(value)) return value.join(',').trim().toLowerCase();
  return value?.trim().toLowerCase() ?? null;
}

export function requestBodyForCapture(
  body: Buffer,
  protocol: Protocol | null,
  contentEncoding: string | string[] | undefined
): Buffer {
  const encoding = normalizeContentEncoding(contentEncoding);
  if (!encoding || encoding === 'identity' || body.length === 0) return body;

  try {
    if (encoding === 'zstd') {
      return protocol === 'responses' ? Buffer.from(decompressZstd(body)) : body;
    }
    return zlib.unzipSync(body);
  } catch {
    return body;
  }
}
