// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { listCannbay2Sessions, importCannbay2Session, uploadCannbay2Session, GovernanceError } from '@/lib/cannbay2';
import { prisma } from '@/lib/db';

export const maxDuration = 300;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action } = body;

    if (action === 'list') {
      const sessions = listCannbay2Sessions();
      return NextResponse.json({ sessions });
    }

    if (action === 'import') {
      const { sid } = body as { sid: string };
      if (!sid) {
        return NextResponse.json({ error: 'Missing sid' }, { status: 400 });
      }
      try {
        const result = await importCannbay2Session(sid, prisma);
        return NextResponse.json({ result });
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Import failed';
        return NextResponse.json({ error: message }, { status: 500 });
      }
    }

    if (action === 'upload') {
      const { taskId, description } = body as { taskId: string; description?: string };
      if (!taskId) {
        return NextResponse.json({ error: 'Missing taskId' }, { status: 400 });
      }
      try {
        const result = await uploadCannbay2Session(prisma, taskId, description);
        return NextResponse.json(result);
      } catch (err) {
        if (err instanceof GovernanceError) {
          return NextResponse.json({ error: err.message, governance: true }, { status: 422 });
        }
        const message = err instanceof Error ? err.message : 'Upload failed';
        return NextResponse.json({ error: message }, { status: 500 });
      }
    }

    return NextResponse.json({ error: `Unknown action: "${action}". Supported: list, import, upload` }, { status: 400 });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
