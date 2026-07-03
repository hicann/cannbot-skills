// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { BRAND_SOURCE_TYPE } from '@/lib/branding';
import type { SessionListItem, RawInteraction } from '../../shared/types';
import { listSessions as opencodeDbListSessions, readSession as opencodeDbReadSession } from './opencode-db';
import { listSessions as claudeJsonlListSessions, readSession as claudeJsonlReadSession } from './claude-jsonl';
import { listSessions as cannbotInsightListSessions, readSession as cannbotInsightReadSession } from './cannbot-insight';

export interface Adapter {
  listSessions(dbPath: string): SessionListItem[];
  readSession(dbPath: string, sessionId: string): RawInteraction[];
}

const opencodeDbAdapter: Adapter = {
  listSessions: opencodeDbListSessions,
  readSession: opencodeDbReadSession,
};

const claudeJsonlAdapter: Adapter = {
  listSessions: claudeJsonlListSessions,
  readSession: claudeJsonlReadSession,
};

const cannbotInsightAdapter: Adapter = {
  listSessions: cannbotInsightListSessions,
  readSession: cannbotInsightReadSession,
};

export function getAdapter(sourceType: string): Adapter | null {
  switch (sourceType) {
    case 'opencode-db':
      return opencodeDbAdapter;
    case 'claude-jsonl':
      return claudeJsonlAdapter;
    case BRAND_SOURCE_TYPE:
      return cannbotInsightAdapter;
    default:
      throw new Error(`Unknown source type: "${sourceType}". Supported types: opencode-db, claude-jsonl, ${BRAND_SOURCE_TYPE}`);
  }
}
