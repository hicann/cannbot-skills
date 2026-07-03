// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface MetricCardsProps {
  totalSessions: number;
  totalTokens: number;
  totalCost: number;
  avgLatencyMs: number;
  totalErrors?: number;
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function formatCost(n: number): string {
  return `$${n.toFixed(2)}`;
}

function formatLatency(ms: number): string {
  if (ms >= 60_000) return `${(ms / 60_000).toFixed(1)}min`;
  if (ms >= 1_000) return `${(ms / 1_000).toFixed(1)}s`;
  return `${ms}ms`;
}

export function MetricCards({ totalSessions, totalTokens, totalCost, avgLatencyMs, totalErrors }: MetricCardsProps) {
  return (
    <div className={cn("grid gap-4", totalErrors !== undefined ? "grid-cols-5" : "grid-cols-4")}>
      <Card size="sm">
        <CardHeader>
          <CardTitle>Sessions</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-bold">{totalSessions}</p>
        </CardContent>
      </Card>
      <Card size="sm">
        <CardHeader>
          <CardTitle>Total Tokens</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-bold">{formatTokens(totalTokens)}</p>
        </CardContent>
      </Card>
      <Card size="sm">
        <CardHeader>
          <CardTitle>Total Cost</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-bold">{formatCost(totalCost)}</p>
        </CardContent>
      </Card>
      <Card size="sm">
        <CardHeader>
          <CardTitle>Avg Duration</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-bold">{formatLatency(avgLatencyMs)}</p>
        </CardContent>
      </Card>
      {totalErrors !== undefined && (
        <Card size="sm" className={totalErrors > 0 ? "border-red-200 dark:border-red-500/30" : ""}>
          <CardHeader>
            <CardTitle className={totalErrors > 0 ? "text-red-600 dark:text-red-400" : ""}>Errors</CardTitle>
          </CardHeader>
          <CardContent>
            <p className={`text-2xl font-bold ${totalErrors > 0 ? "text-red-600 dark:text-red-400" : "text-muted-foreground"}`}>
              {totalErrors}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
