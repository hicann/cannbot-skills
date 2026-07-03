// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import React, { useMemo, useState } from 'react';
import { render, Box, useApp, useInput } from 'ink';
import { StatusBar } from '@/cli/tui/components/StatusBar';
import { KeyBar } from '@/cli/tui/components/KeyBar';
import { SessionList } from '@/cli/tui/screens/SessionList';
import { SessionDetail } from '@/cli/tui/screens/SessionDetail';
import { TurnDetail } from '@/cli/tui/screens/TurnDetail';
import { CompareView } from '@/cli/tui/screens/CompareView';
import { ImportPanel } from '@/cli/tui/screens/ImportPanel';
import { HelpScreen } from '@/cli/tui/screens/HelpScreen';
import { ConfirmDialog } from '@/cli/tui/components/ConfirmDialog';
import { useNavigation } from '@/cli/hooks/useNavigation';
import { InsightClient } from '@/cli/client';
import type { CliConfig } from '@/cli/config';

export interface TuiAppProps {
  config: CliConfig;
}

function App({ config }: TuiAppProps) {
  const client = useMemo(
    () => new InsightClient(config.server, { timeout: config.timeout }),
    [config.server, config.timeout],
  );
  const { exit } = useApp();
  const { nav, navigate, goBack } = useNavigation();

  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [uploadTarget, setUploadTarget] = useState<string | null>(null);

  useInput((input, key) => {
    if (input === 'q' || (key.ctrl && input === 'c')) {
      exit();
    }
    if (input === '?' && nav.screen !== 'help') {
      navigate('help');
    }
  });

  const screenLabels: Record<string, string> = {
    sessions: 'Sessions',
    session: `Session: ${nav.params?.taskId ?? ''}`,
    turn: 'Turn Detail',
    compare: `Compare: ${nav.params?.taskId1 ?? ''} vs ${nav.params?.taskId2 ?? ''}`,
    import: 'Import',
    help: 'Help',
  };

  return (
    <Box flexDirection="column" height="100%">
      <StatusBar client={client} screen={screenLabels[nav.screen] ?? nav.screen} />
      <Box flexGrow={1}>
        {uploadTarget ? (
          <ConfirmDialog
            message={`Upload session "${uploadTarget}" to CANNBay?`}
            onConfirm={async () => {
              await client.uploadSession(uploadTarget, '');
              setUploadTarget(null);
            }}
            onCancel={() => setUploadTarget(null)}
          />
        ) : deleteTarget ? (
          <ConfirmDialog
            message={`Delete session "${deleteTarget}"? This cannot be undone.`}
            onConfirm={async () => {
              await client.deleteSession(deleteTarget);
              setDeleteTarget(null);
            }}
            onCancel={() => setDeleteTarget(null)}
          />
        ) : nav.screen === 'sessions' ? (
          <SessionList
            client={client}
            onSelect={(taskId) => navigate('session', { taskId })}
            onCompare={(taskId1, taskId2) => navigate('compare', { taskId1, taskId2 })}
            onImport={() => navigate('import')}
            onDelete={(taskId) => setDeleteTarget(taskId)}
            onUpload={(taskId) => setUploadTarget(taskId)}
          />
        ) : nav.screen === 'session' && nav.params?.taskId ? (
          <SessionDetail
            client={client}
            taskId={nav.params.taskId}
            onBack={goBack}
            onSelectTurn={(turnId) => navigate('turn', { turnId })}
          />
        ) : nav.screen === 'turn' && nav.params?.turnId ? (
          <TurnDetail client={client} turnId={nav.params.turnId} onBack={goBack} />
        ) : nav.screen === 'compare' && nav.params?.taskId1 && nav.params?.taskId2 ? (
          <CompareView
            client={client}
            taskId1={nav.params.taskId1}
            taskId2={nav.params.taskId2}
            onBack={goBack}
          />
        ) : nav.screen === 'import' ? (
          <ImportPanel client={client} onBack={goBack} />
        ) : nav.screen === 'help' ? (
          <HelpScreen onBack={goBack} />
        ) : (
          <SessionList
            client={client}
            onSelect={(taskId) => navigate('session', { taskId })}
            onCompare={(taskId1, taskId2) => navigate('compare', { taskId1, taskId2 })}
            onImport={() => navigate('import')}
            onDelete={(taskId) => setDeleteTarget(taskId)}
          />
        )}
      </Box>
      <KeyBar screen={nav.screen} />
    </Box>
  );
}

export function runTui(config: CliConfig): Promise<void> {
  if (process.stdin.isTTY) {
    process.stdin.resume();
    process.stdin.setRawMode(true);
  }

  const { waitUntilExit } = render(<App config={config} />, {
    exitOnCtrlC: false,
    patchConsole: false,
  });

  return waitUntilExit().then(() => {
    if (process.stdin.isTTY) {
      process.stdin.setRawMode(false);
      process.stdin.pause();
    }
  });
}
