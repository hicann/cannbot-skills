// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// AIProviderConfigPanel hydration 契约 IT：渲染路径不得读 localStorage
// （SSR 无 localStorage，客户端 hydration 首渲染读到已存配置会导致
// "✓ 已保存" span 与三个 Input 值的 hydration mismatch）。
// 手法：mock loader 强制返回已存配置（绕过其内部 typeof window 守卫，
// 模拟"客户端有存量的首次渲染"），renderToString 断言输出仍为默认值
// 且 loader 零调用 —— 旧实现（渲染期直接 loadProviderConfig()）此测试必红。
import { describe, it, expect, vi } from 'vitest';
import { createElement } from 'react';
import { renderToString } from 'react-dom/server';
import { loadProviderConfig } from '@/lib/ai-provider-config';

vi.mock('@/lib/ai-provider-config', () => ({
  loadProviderConfig: vi.fn(() => ({ baseUrl: 'https://saved.example/v1', apiKey: 'sk-saved-key', model: 'saved-model' })),
  saveProviderConfig: vi.fn(),
  clearProviderConfig: vi.fn(),
}));

const { AIProviderConfigPanel } = await import('@/components/observe/AIProviderConfigPanel');

describe('AIProviderConfigPanel：渲染路径与 localStorage 解耦（hydration 契约）', () => {
  it('loader 有存量时 SSR 输出仍为默认值、无"已保存"，且渲染期零调用', () => {
    const html = renderToString(createElement(AIProviderConfigPanel, { compact: true }));
    expect(vi.mocked(loadProviderConfig)).not.toHaveBeenCalled();
    expect(html).not.toContain('已保存');
    expect(html).not.toContain('https://saved.example/v1');
    expect(html).not.toContain('saved-model');
    expect(html).toContain('https://dashscope.aliyuncs.com/compatible-mode/v1');
  });
});
