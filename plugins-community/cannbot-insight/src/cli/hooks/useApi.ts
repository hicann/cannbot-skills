// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';

interface UseApiResult<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
}

const cache = new Map<string, { data: unknown; timestamp: number }>();
const CACHE_TTL = 30_000;

export function useApi<T>(
  fetcher: () => Promise<T>,
  deps: unknown[],
  options?: { cacheKey?: string; ttl?: number },
): UseApiResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const depsRef = useRef(deps);
  depsRef.current = deps;

  const key = useMemo(() => options?.cacheKey ?? JSON.stringify(depsRef.current), [options?.cacheKey]);
  const ttl = options?.ttl ?? CACHE_TTL;

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const cached = cache.get(key);
      if (cached && Date.now() - cached.timestamp < ttl) {
        setData(cached.data as T);
        setLoading(false);
        return;
      }
      const result = await fetcher();
      cache.set(key, { data: result, timestamp: Date.now() });
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [fetcher, key, ttl]);

  useEffect(() => {
    let cancelled = false;
    fetchData().then(() => {
      if (cancelled) return;
    });
    return () => { cancelled = true; };
  }, [fetchData]);

  const refresh = useCallback(async () => {
    cache.delete(key);
    await fetchData();
  }, [key, fetchData]);

  return { data, loading, error, refresh };
}
