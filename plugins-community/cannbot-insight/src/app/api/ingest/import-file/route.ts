// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { importSession } from '@/lib/ingest/data-service';
import { BRAND_SOURCE_TYPE } from '@/lib/branding';
import { prisma } from '@/lib/db';

export const maxDuration = 300;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { source, sessionId, filePath } = body;

    if (source !== 'opencode-db' && source !== 'claude-jsonl' && source !== BRAND_SOURCE_TYPE) {
      return NextResponse.json(
        { error: `Unsupported source: "${source}". Supported: opencode-db, claude-jsonl, ${BRAND_SOURCE_TYPE}` },
        { status: 400 }
      );
    }

    if (!sessionId || !filePath) {
      return NextResponse.json(
        { error: 'Missing required fields: sessionId, filePath' },
        { status: 400 }
      );
    }

    const result = await importSession(filePath, sessionId, prisma, filePath, source);

    return NextResponse.json({
      sessionId: result.sessionId,
      imported: result.imported,
      query: result.query ?? null,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
