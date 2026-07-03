// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { BRAND_SOURCE_TYPE } from '@/lib/branding';
import fs from 'node:fs';
import path from 'node:path';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { dirPath } = body;

    if (!dirPath) {
      return NextResponse.json(
        { error: 'Missing required field: dirPath' },
        { status: 400 },
      );
    }

    const resolved = path.resolve(dirPath);

    if (!fs.existsSync(resolved)) {
      return NextResponse.json(
        { error: `Path does not exist: ${resolved}` },
        { status: 404 },
      );
    }

    const stats = fs.statSync(resolved);
    if (!stats.isDirectory()) {
      return NextResponse.json({
        isDirectory: false,
        entries: [],
        parentPath: path.dirname(resolved),
        currentPath: resolved,
      });
    }

    const names = fs.readdirSync(resolved).sort((a, b) => {
      try {
        const aIsDir = fs.statSync(path.join(resolved, a)).isDirectory();
        const bIsDir = fs.statSync(path.join(resolved, b)).isDirectory();
        if (aIsDir !== bIsDir) return aIsDir ? -1 : 1;
        return a.localeCompare(b);
      } catch {
        return a.localeCompare(b);
      }
    });

    const entries = names.map(name => {
      const full = path.join(resolved, name);
      try {
        const s = fs.statSync(full);
        const ext = path.extname(name).toLowerCase();
        const isDb = ext === '.db' || ext === '.sqlite' || ext === '.sqlite3';
        const isJsonl = ext === '.jsonl';
        const isCannbot = isDb && name.startsWith('cannbot_session_');
        const isDir = s.isDirectory();
        const isImportableFile = !isDir && (isDb || isJsonl);
        return {
          name,
          fullPath: full,
          isDir,
          size: isDir ? 0 : s.size,
          isImportableFile,
          importableType: isCannbot ? BRAND_SOURCE_TYPE : isDb ? 'opencode-db' : isJsonl ? 'claude-jsonl' : null,
        };
      } catch {
        return {
          name,
          fullPath: full,
          isDir: false,
          size: 0,
          isImportable: false,
          isImportableFile: false,
          importableType: null,
        };
      }
    });

    return NextResponse.json({
      isDirectory: true,
      entries,
      parentPath: path.dirname(resolved),
      currentPath: resolved,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
