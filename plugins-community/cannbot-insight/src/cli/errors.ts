// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

export class InsightError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'InsightError';
  }
}

export class ApiError extends InsightError {
  public readonly status: number;
  public readonly retryable: boolean;

  constructor(status: number, message: string, retryable: boolean) {
    super(`API ${status}: ${message}`);
    this.name = 'ApiError';
    this.status = status;
    this.retryable = retryable;
  }
}

export class NetworkError extends InsightError {
  constructor(message: string) {
    super(`Network: ${message}`);
    this.name = 'NetworkError';
  }
}

export class TerminalError extends InsightError {
  constructor(message: string) {
    super(`Terminal: ${message}`);
    this.name = 'TerminalError';
  }
}

export class ConfigError extends InsightError {
  constructor(message: string) {
    super(`Config: ${message}`);
    this.name = 'ConfigError';
  }
}
