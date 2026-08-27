// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// normalize 层只做布局归一，不做内容 interpret（capture ≠ interpret）：
//   - task-notification → system 摘要等框架行为解释，由 insight 的
//     claude-jsonl adapter 负责（它有 skill-injection / system-reminder
//     同类先例，且对 verbatim 与 norm 两种输入形状都正确）
//   - norm/ = verbatim 行的纯拷贝 + subagents 目录镜像，保证导入路径
//     （norm/<sid>.jsonl → norm/<sid>/subagents/）与捕获目录分离

export interface JsonlLine {
  type?: string;
  message?: { role?: string; content?: unknown } | null;
  timestamp?: string;
  source?: string;
  [key: string]: unknown;
}
