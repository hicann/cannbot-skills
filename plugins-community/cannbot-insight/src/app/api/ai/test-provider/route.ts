// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { baseUrl, apiKey } = body as { baseUrl: string; apiKey: string };

    if (!baseUrl || !apiKey) {
      return NextResponse.json({ error: 'Missing baseUrl or apiKey' }, { status: 400 });
    }

    const url = `${baseUrl.replace(/\/+$/, '')}/models`;
    const res = await fetch(url, {
      headers: { Authorization: `Bearer ${apiKey}` },
      signal: AbortSignal.timeout(15000),
    });

    if (res.ok) {
      const data = await res.json();
      const modelCount = Array.isArray(data.data) ? data.data.length : 0;
      return NextResponse.json({ success: true, message: `✅ 连接成功 (${modelCount} models)` });
    }

    const text = await res.text().catch(() => '');
    return NextResponse.json({ success: false, message: `❌ HTTP ${res.status}: ${text.slice(0, 200)}` });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ success: false, message: `❌ ${message}` }, { status: 500 });
  }
}
