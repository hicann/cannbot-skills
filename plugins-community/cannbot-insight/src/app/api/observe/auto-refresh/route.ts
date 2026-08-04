// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { probeAutoRefresh } from '@/lib/ingest/auto-refresh-probe';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const taskId = searchParams.get('taskId');

    if (!taskId) {
      return NextResponse.json({ changed: false }, { status: 400 });
    }

    const session = await prisma.session.findFirst({
      where: { taskId },
      select: { id: true, taskId: true, sourcePath: true, framework: true },
    });

    if (!session || !session.sourcePath) {
      return NextResponse.json({ changed: false });
    }

    const probe = await probeAutoRefresh(
      { id: session.id, taskId: session.taskId, sourcePath: session.sourcePath, framework: session.framework },
      prisma,
    );

    return NextResponse.json({
      countChanged: probe.countChanged,
      sourceMessageCount: probe.sourceMessageCount,
      ourTurnCount: probe.ourTurnCount,
      maxTimeUpdated: probe.maxTimeUpdated,
      streaming: probe.streaming,
      pendingInput: probe.pendingInput,
      settled: probe.settled,
    });
  } catch {
    return NextResponse.json({ changed: false });
  }
}
