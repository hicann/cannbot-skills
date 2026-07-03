// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { analyzeWorkflow } from '@/lib/ai/analyzer';
import type { AIProviderConfig } from '@/lib/ai/analyzer';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { taskId, provider } = body as {
      taskId: string;
      provider: AIProviderConfig;
    };

    if (!taskId) {
      return NextResponse.json({ error: 'Missing taskId' }, { status: 400 });
    }

    if (!provider?.baseUrl || !provider?.apiKey || !provider?.model) {
      return NextResponse.json({ error: 'Missing provider config (baseUrl, apiKey, model)' }, { status: 400 });
    }

    const result = await analyzeWorkflow(taskId, provider, prisma);

    return NextResponse.json({ result });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
