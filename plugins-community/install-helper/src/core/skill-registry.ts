// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import type { ScannedSkill } from "./scanner.js";

export interface SkillEntry {
  id: string;
  description: string;
  source: string;
}

export interface SkillCategory {
  id: string;
  name: string;
  skills: SkillEntry[];
}

const CATEGORY_DEFS: { id: string; name: string }[] = [
  { id: "knowledge", name: "知识与参考" },
  { id: "env-tools", name: "环境与工具" },
  { id: "debug", name: "调试与诊断" },
  { id: "testing", name: "测试与质量" },
  { id: "ascendc", name: "AscendC 开发" },
  { id: "pypto", name: "PyPTO 开发" },
  { id: "tilelang", name: "TileLang 开发" },
  { id: "triton", name: "Triton 开发" },
  { id: "model", name: "模型推理优化" },
  { id: "graph", name: "图模式" },
  { id: "platform", name: "平台工具" },
  { id: "other", name: "其他 Skills" },
];

let dynamicSkills: SkillEntry[] = [];
let initialized = false;

export function initFromScan(scanned: ScannedSkill[]): void {
  dynamicSkills = scanned.map((s) => ({
    id: s.id,
    description: s.description,
    source: s.source,
  }));
  initialized = true;
}

export function isScanInitialized(): boolean {
  return initialized;
}

const STATIC_SKILL_CATEGORIES: SkillCategory[] = [
  {
    id: "knowledge",
    name: "知识与参考",
    skills: [
      { id: "npu-arch", description: "NPU 架构知识、芯片型号映射", source: "ops" },
      { id: "ascendc-api-best-practices", description: "API 使用最佳实践", source: "ops" },
      { id: "ascendc-tiling-design", description: "Tiling 和 Kernel 设计方法论", source: "ops" },
      { id: "ascendc-docs-search", description: "API 文档索引 + 在线搜索", source: "ops" },
      { id: "ascendc-docs-gen", description: "算子文档写作参考", source: "ops" },
      { id: "ascendc-performance-best-practices", description: "性能优化经验总结", source: "ops" },
      { id: "ascendc-regbase-best-practice", description: "RegBase 算子 API 约束", source: "ops" },
      { id: "ops-precision-standard", description: "算子精度标准（atol/rtol）", source: "ops" },
      { id: "ops-spec-gen", description: "算子 spec.yaml 生成与校验", source: "ops" },
    ],
  },
  {
    id: "env-tools",
    name: "环境与工具",
    skills: [
      { id: "ascendc-env-check", description: "NPU 设备查询、CANN 环境验证", source: "ops" },
      { id: "cann-env-setup", description: "CANN 安装与环境配置指导", source: "ops" },
      { id: "ops-profiling", description: "NPU 性能采集与分析", source: "ops" },
      { id: "ops-simulator", description: "CANN Simulator 精度/性能仿真", source: "ops" },
      { id: "torch-ops-profiler", description: "torch_npu.profiler 性能报告", source: "ops" },
      { id: "aiss-tiling-solver", description: "AISS-TilingSolver 自动求解 Tiling 参数", source: "ops" },
    ],
  },
  {
    id: "debug",
    name: "调试与诊断",
    skills: [
      { id: "ascendc-precision-debug", description: "精度调试、症状-原因速查", source: "ops" },
      { id: "ascendc-runtime-debug", description: "运行时错误码解析", source: "ops" },
      { id: "ascendc-crash-debug", description: "卡死/崩溃调试、Coredump 分析", source: "ops" },
      { id: "ascendc-perf-optimize", description: "性能优化策略制定", source: "ops" },
      { id: "model-infer-precision-debug", description: "NPU 推理精度诊断", source: "model" },
      { id: "model-infer-runtime-debug", description: "NPU 推理运行时错误诊断", source: "model" },
      { id: "torch-npugraph-ex-dfx-triage", description: "npugraph_ex DFX 问题分诊", source: "graph" },
      { id: "torch-npugraph-ex-compile-error-diagnosis", description: "npugraph_ex 编译期报错诊断", source: "graph" },
    ],
  },
  {
    id: "testing",
    name: "测试与质量",
    skills: [
      { id: "ascendc-code-review", description: "代码检视方法论", source: "ops" },
      { id: "ascendc-ut-develop", description: "UT 开发与覆盖率增强", source: "ops" },
      { id: "ascendc-st-design", description: "ST 测试用例设计", source: "ops" },
      { id: "ascendc-whitebox-design", description: "白盒测试用例生成", source: "ops" },
      { id: "ascendc-task-focus", description: "长任务聚焦防迷失", source: "ops" },
      { id: "tilelang-op-test-design", description: "TileLang 测试设计", source: "ops" },
      { id: "tilelang-review", description: "TileLang 代码格式检查", source: "ops" },
      { id: "triton-op-verifier", description: "Triton 算子验证", source: "ops" },
    ],
  },
  {
    id: "ascendc",
    name: "AscendC 开发",
    skills: [
      { id: "ascendc-direct-invoke-template", description: "Kernel 直调工程模板", source: "ops" },
      { id: "ascendc-registry-invoke-template", description: "自定义算子工程模板", source: "ops" },
      { id: "ascendc-direct-invoke-to-registry-invoke", description: "直调转注册调用", source: "ops" },
      { id: "ascendc-registry-invoke-to-direct-invoke", description: "注册调用转直调", source: "ops" },
      { id: "ascendc-blaze-best-practice", description: "Matmul/GEMM Blaze 直调生成", source: "ops" },
      { id: "ascendc-simt-best-practices", description: "SIMT 最佳实践与 API 导航", source: "ops" },
      { id: "ascendc-simt-tiling-design", description: "SIMT 算子切分设计", source: "ops" },
      { id: "torch-ascendc-op-extension", description: "Ascend C 对接 PyTorch", source: "ops" },
      { id: "catlass-op-design", description: "Catlass 算子设计", source: "ops" },
      { id: "catlass-op-develop", description: "Catlass 算子开发", source: "ops" },
      { id: "catlass-op-perf-tune", description: "Catlass 性能调优", source: "ops" },
      { id: "cuda2ascend-simt", description: "CUDA 迁移到 Ascend C SIMT", source: "ops-lab" },
      { id: "ops-direct-invoke-flash", description: "从零构建 Ascend C 核函数", source: "plugins-official/ops-direct-invoke-flash/skills" },
      { id: "ops-registry-invoke-workflow", description: "注册调用工作流", source: "plugins-official/ops-registry-invoke" },
      { id: "ops-easyasc-dsl", description: "EasyASC DSL 算子开发", source: "plugins-community/ops-easyasc-dsl/skill" },
    ],
  },
  {
    id: "pypto",
    name: "PyPTO 开发",
    skills: [
      { id: "pypto-intent-understand", description: "需求意图理解与规格生成", source: "ops" },
      { id: "pypto-api-explore", description: "API 可行性探索与分析", source: "ops" },
      { id: "pypto-op-design", description: "算子方案设计生成", source: "ops" },
      { id: "pypto-golden-generate", description: "Golden 参考实现生成", source: "ops" },
      { id: "pypto-op-develop", description: "算子代码实现与测试", source: "ops" },
      { id: "pypto-precision-debug", description: "精度问题排查", source: "ops" },
      { id: "pypto-precision-compare", description: "精度对比分析", source: "ops" },
      { id: "pypto-op-perf-tune", description: "性能分析与调优", source: "ops" },
    ],
  },
  {
    id: "tilelang",
    name: "TileLang 开发",
    skills: [
      { id: "tilelang-env-check", description: "环境检查与配置验证", source: "ops" },
      { id: "tilelang-submodule-pull", description: "三方库与子模块拉取", source: "ops" },
      { id: "tilelang-api-best-practices", description: "TileLang API 最佳实践", source: "ops" },
      { id: "tilelang-programming-model-guide", description: "Developer/Expert 模式选择", source: "ops" },
      { id: "tilelang-op-design", description: "算子设计文档生成", source: "ops" },
      { id: "tilelang-op-develop", description: "算子代码实现与测试", source: "ops" },
      { id: "tilelang-op-test-design", description: "测试设计与覆盖率分析", source: "ops" },
      { id: "tilelang-perf-optimization", description: "性能调优与劣化检查", source: "ops" },
      { id: "tilelang-review", description: "代码格式检查与修复", source: "ops" },
    ],
  },
  {
    id: "triton",
    name: "Triton 开发",
    skills: [
      { id: "triton-task-extractor", description: "算子任务提取与构建", source: "ops" },
      { id: "triton-op-designer", description: "算法草图设计", source: "ops" },
      { id: "triton-op-coding", description: "Triton 内核代码生成", source: "ops" },
      { id: "triton-op-verifier", description: "算子精度和性能验证", source: "ops" },
      { id: "triton-latency-optimizer", description: "Triton 代码性能优化", source: "ops" },
    ],
  },
  {
    id: "model",
    name: "模型推理优化",
    skills: [
      { id: "model-infer-migrator", description: "框架适配与部署基线", source: "model" },
      { id: "model-infer-parallel-analysis", description: "并行策略分析（TP/EP/DP）", source: "model" },
      { id: "model-infer-parallel-impl", description: "并行切分实施", source: "model" },
      { id: "model-infer-kvcache", description: "KVCache 优化 + FA 替换", source: "model" },
      { id: "model-infer-fusion", description: "torch_npu 融合算子替换", source: "model" },
      { id: "model-infer-graph-mode", description: "torch.compile 图模式适配", source: "model" },
      { id: "model-infer-multi-stream", description: "多流并行优化", source: "model" },
      { id: "model-infer-prefetch", description: "权重预取适配", source: "model" },
      { id: "model-infer-superkernel", description: "SuperKernel 适配", source: "model" },
      { id: "model-infer-precision-debug", description: "NPU 推理精度诊断", source: "model" },
      { id: "model-infer-runtime-debug", description: "NPU 推理运行时错误诊断", source: "model" },
      { id: "model-infer-harmony", description: "端侧鸿蒙 ASR 量化转换与打包", source: "model" },
    ],
  },
  {
    id: "graph",
    name: "图模式",
    skills: [
      { id: "torch-npugraph-ex-knowledge", description: "npugraph_ex 使用指南", source: "graph" },
      { id: "torch-npugraph-ex-template", description: "npugraph_ex MRE 代码模板", source: "graph" },
      { id: "torch-npugraph-ex-dfx-triage", description: "DFX 问题分诊", source: "graph" },
      { id: "torch-npugraph-ex-compile-error-diagnosis", description: "编译期报错诊断", source: "graph" },
      { id: "torch-npugraph-ex-runtime-error-diagnosis", description: "运行时报错诊断", source: "graph" },
      { id: "torch-npugraph-ex-performance-diagnosis", description: "性能诊断", source: "graph" },
      { id: "torch-custom-ops-guide", description: "自定义算子入图指南", source: "graph" },
    ],
  },
  {
    id: "platform",
    name: "平台工具",
    skills: [
      { id: "gitcode-pr-handler", description: "GitCode PR 标题/描述生成", source: "infra" },
      { id: "gitcode-issue-gen", description: "GitCode Issue 生成与关联", source: "infra" },
      { id: "gitcode-issue-handler", description: "GitCode Issue 端到端处置", source: "infra" },
      { id: "gitcode-toolkit", description: "GitCode 协作通用参考", source: "infra" },
      { id: "cannbot-skill-reviewer", description: "Skill 入库质量审查", source: "infra" },
    ],
  },
];

const CATEGORY_MAP: Record<string, string> = {};
for (const cat of STATIC_SKILL_CATEGORIES) {
  for (const skill of cat.skills) {
    CATEGORY_MAP[skill.id] = cat.id;
  }
}

function getActiveSkills(): SkillEntry[] {
  if (initialized) return deduplicate(dynamicSkills);
  return deduplicate(STATIC_SKILL_CATEGORIES.flatMap((c) => c.skills));
}

function deduplicate(skills: SkillEntry[]): SkillEntry[] {
  const seen = new Set<string>();
  const result: SkillEntry[] = [];
  for (const skill of skills) {
    if (!seen.has(skill.id)) {
      result.push(skill);
      seen.add(skill.id);
    }
  }
  return result;
}

export function findSkill(query: string): SkillEntry | undefined {
  const normalized = query.toLowerCase().trim();
  const pool = getActiveSkills();
  return pool.find((s) => s.id === normalized);
}

export function getAllSkills(): SkillEntry[] {
  return getActiveSkills();
}

export function getSkillsByCategory(categoryId: string): SkillEntry[] {
  return getAllCategories().find((c) => c.id === categoryId)?.skills ?? [];
}

export function getAllCategories(): SkillCategory[] {
  const skills = getActiveSkills();
  return CATEGORY_DEFS.map((def) => ({
    id: def.id,
    name: def.name,
    skills: skills.filter((s) => {
      const catId = CATEGORY_MAP[s.id];
      if (catId) return catId === def.id;
      return def.id === "other";
    }),
  })).filter((cat) => cat.skills.length > 0);
}
