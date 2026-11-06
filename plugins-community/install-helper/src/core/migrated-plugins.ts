// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import type { PluginEntry } from "../types/index.js";

// Plugins migrated to the cannbot repo, with optional frozen local copies.
// Tombstones preserve discovery and aliases, then delegate installation to
// @cannbot-plugin/cannbot. Modern installers use their bundled snapshot;
// these entries do not download source or provide missing plugin assets.
export const MIGRATED_PLUGINS: PluginEntry[] = [
  {
    id: "torch-compile",
    dir: "",
    displayName: "torch.compile 图模式",
    script: "cannbot",
    aliases: ["torch", "compile", "graph"],
    skills: 7,
    agents: 1,
    version: "1.0.4",
    description: "PyTorch npugraph_ex 图模式编排；已迁移至 cannbot 仓，经托管安装器安装（仅 project 级）",
    source: "cannbot",
  },
  {
    id: "ops-perf-optimize",
    dir: "",
    displayName: "算子性能调优",
    script: "cannbot",
    aliases: ["perf-optimize", "perf-optimization", "performance"],
    skills: 0,
    agents: 0,
    description: "AscendC 算子性能调优工作流；已迁移至 cannbot 仓，经托管安装器安装（仅 project 级）",
    source: "cannbot",
  },
  {
    id: "catlass-op-generator",
    dir: "",
    displayName: "Catlass 算子直调",
    script: "cannbot",
    aliases: ["catlass"],
    skills: 6,
    agents: 0,
    version: "0.1.0",
    description: "CATLASS C++ 算子开发工作流；已迁移至 cannbot 仓，经托管安装器安装",
    source: "cannbot",
  },
  {
    id: "model-infer-optimize",
    dir: "",
    displayName: "NPU 推理优化",
    script: "cannbot",
    aliases: ["model-infer", "infer", "inference", "sota", "sota-approach", "model-infer-sota"],
    skills: 14,
    agents: 9,
    version: "1.0.6",
    description: "NPU 模型推理端到端优化；已迁移至 cannbot 仓，经托管安装器安装",
    source: "cannbot",
  },
  {
    id: "ops-direct-invoke",
    dir: "",
    displayName: "AscendC Kernel 直调",
    script: "cannbot",
    aliases: ["ops-direct", "ascendc-direct", "direct", "kernel"],
    skills: 40,
    agents: 3,
    description: "Ascend C 直调算子开发工作流；已迁移至 cannbot 仓，经托管安装器安装（仅 project 级）",
    source: "cannbot",
  },
  {
    id: "ops-direct-invoke-flash",
    dir: "",
    displayName: "AscendC Kernel 直调（快速版）",
    script: "cannbot",
    aliases: ["flash", "kernel-flash"],
    skills: 7,
    agents: 1,
    description: "从数学公式或代码片段快速构建 Ascend C 核函数的工作流；已迁移至 cannbot 仓，经托管安装器安装（仅 project 级）",
    source: "cannbot",
  },
  {
    id: "triton-op-generator",
    dir: "",
    displayName: "Triton 算子开发",
    script: "cannbot",
    aliases: ["triton"],
    skills: 8,
    agents: 0,
    description: "Triton-Ascend 算子 6 阶段自动生成与优化（描述→代码→验证→优化）；已迁移至 cannbot 仓，经托管安装器安装",
    source: "cannbot",
  },
  {
    id: "ops-registry-invoke",
    dir: "",
    displayName: "AscendC 算子注册调用",
    script: "cannbot",
    aliases: ["ops-registry", "ascendc-registry", "registry"],
    skills: 7,
    agents: 0,
    description: "生产级 ACLNN 注册算子全流程：代码→UT/ST→上库；已迁移至 cannbot 仓，经托管安装器安装",
    source: "cannbot",
  },
  {
    id: "collaborative-agent-kernel-evolution",
    dir: "",
    displayName: "CAKE 协同算子进化",
    script: "cannbot",
    aliases: ["cake"],
    skills: 0,
    agents: 0,
    description: "CAKE2 - Collaborative Ascend Kernel Evolution，AI 驱动的 AscendC 算子协同进化；已迁移至 cannbot 仓，经托管安装器安装",
    source: "cannbot",
  },
];

export function isMigratedPlugin(id: string): boolean {
  return MIGRATED_PLUGINS.some((p) => p.id === id);
}
