// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { InsightError, ApiError, NetworkError, TerminalError, ConfigError } from '@/cli/errors';

describe('InsightError', () => {
  it('sets name and message', () => {
    const err = new InsightError('test message');
    expect(err.name).toBe('InsightError');
    expect(err.message).toBe('test message');
  });

  it('is instanceof Error', () => {
    const err = new InsightError('test');
    expect(err).toBeInstanceOf(Error);
  });

  it('is instanceof InsightError', () => {
    const err = new InsightError('test');
    expect(err).toBeInstanceOf(InsightError);
  });
});

describe('ApiError', () => {
  it('sets name, status, message, retryable', () => {
    const err = new ApiError(404, 'Not found', false);
    expect(err.name).toBe('ApiError');
    expect(err.status).toBe(404);
    expect(err.retryable).toBe(false);
    expect(err.message).toBe('API 404: Not found');
  });

  it('is instanceof InsightError', () => {
    const err = new ApiError(500, 'Server error', true);
    expect(err).toBeInstanceOf(InsightError);
  });

  it('5xx errors are retryable', () => {
    const err = new ApiError(500, 'Server error', true);
    expect(err.retryable).toBe(true);
  });

  it('4xx errors are not retryable', () => {
    const err = new ApiError(400, 'Bad request', false);
    expect(err.retryable).toBe(false);
  });
});

describe('NetworkError', () => {
  it('sets name and message with prefix', () => {
    const err = new NetworkError('ECONNREFUSED');
    expect(err.name).toBe('NetworkError');
    expect(err.message).toBe('Network: ECONNREFUSED');
  });

  it('is instanceof InsightError', () => {
    const err = new NetworkError('test');
    expect(err).toBeInstanceOf(InsightError);
  });
});

describe('TerminalError', () => {
  it('sets name and message with prefix', () => {
    const err = new TerminalError('no tty');
    expect(err.name).toBe('TerminalError');
    expect(err.message).toBe('Terminal: no tty');
  });

  it('is instanceof InsightError', () => {
    const err = new TerminalError('test');
    expect(err).toBeInstanceOf(InsightError);
  });
});

describe('ConfigError', () => {
  it('sets name and message with prefix', () => {
    const err = new ConfigError('invalid config');
    expect(err.name).toBe('ConfigError');
    expect(err.message).toBe('Config: invalid config');
  });

  it('is instanceof InsightError', () => {
    const err = new ConfigError('test');
    expect(err).toBeInstanceOf(InsightError);
  });
});

describe('error hierarchy', () => {
  it('all errors are catchable as InsightError', () => {
    const errors = [
      new ApiError(500, 's', true),
      new NetworkError('s'),
      new TerminalError('s'),
      new ConfigError('s'),
    ];

    for (const err of errors) {
      expect(err).toBeInstanceOf(InsightError);
    }
  });
});
