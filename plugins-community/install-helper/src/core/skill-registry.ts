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
import { t } from "../utils/i18n.js";

export interface SkillEntry {
  id: string;
  description: string;
  source: string;
  filePath?: string;
}

export interface SkillCategory {
  id: string;
  name: string;
  skills: SkillEntry[];
}

// Categories mirror the repository's scanDirs layout 1:1 — a skill's `source`
// (its directory under the repo root) decides its category, so no per-skill
// mapping is maintained and new skills are categorized automatically. The
// semantic grouping below (SUBCATEGORY_MAP) is kept only as in-category
// ordering hints for the wizard's separator rows.
const CATEGORY_DEFS: { id: string; name: string; nameEn: string }[] = [
  { id: "ops", name: "ops — 算子开发", nameEn: "ops — Operator Development" },
  { id: "model", name: "model — 模型推理", nameEn: "model — Model Inference" },
  { id: "graph", name: "graph — 图模式", nameEn: "graph — Graph Mode" },
  { id: "infra", name: "infra — 基础设施", nameEn: "infra — Infrastructure" },
  { id: "runtime", name: "runtime — Runtime 迁移", nameEn: "runtime — Runtime Migration" },
  { id: "tools", name: "tools — 工具链", nameEn: "tools — Toolchain" },
  { id: "communication", name: "communication — 通信与 HCCL", nameEn: "communication — Communication & HCCL" },
];

// In-category semantic subgroups, used ONLY for ordering/separator hints
// inside a category's skill list. Skills without an entry here sort to the
// end of their category — they are never hidden and never fall into a
// catch-all bucket.
const SUBCATEGORY_MAP: Record<string, string> = {};

let dynamicSkills: SkillEntry[] = [];
let initialized = false;

export function initFromScan(scanned: ScannedSkill[]): void {
  dynamicSkills = scanned.map((s) => ({
    id: s.id,
    description: s.description,
    source: s.source,
    filePath: s.filePath,
  }));
  initialized = true;
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
      { id: "ascendc-api-docs-gen", description: "Ascend C API文档写作参考。当用户需要生成Ascend C API文档或了解Ascend C API文档规范时触发此技能。", source: "ops" },
      { id: "samples", description: "当用户需要查找样例、了解特性用法或获取算子优化建议时使用。", source: "ops" },
      { id: "docs-translation", description: "将本地 CANN 仓库中的中文 Markdown 文档（含 API 文档）翻译为英文，按路径类型生成 _en.md 或 en/ 目录下的同名文件，支持 repoName 排除配置、增量翻译与提交...", source: "infra" },
      { id: "cann-error-code-standard", description: "CANN算子错误码的基准知识源，提供官方文档错误码清单、来源与版本治理、选码和上报规范、机器可读契约。触发：新算子错误码设计、源码静态检查、问题定位或整改Skill需要统一错误码基准时；不直接修...", source: "ops" },
      { id: "ops-design-paradigms", description: "当需要设计或修订 Broadcast、Reduction、Contraction 等算子范式方案，或查询对应算法、资源规划、语言实现约束和故障经验时使用；提供 Ascend C regbase ...", source: "ops" },

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
      { id: "ops-mssanitizer", description: "使用mssanitizer对Ascend C和Triton算子执行内存访问、泄漏、未初始化、竞争与同步检查，自动识别工程类型，运行可复现用例并生成结构化报告。适用于开发完成后的主动内存安全验证和...", source: "ops" },
      { id: "ops-error-code-remediation", description: "为新增或已完成的CANN C/C++算子设计、接入、审计和整改错误码上报路径，覆盖实际log.h解析、失败出口治理、最小编译、负向测试与Markdown报告。触发：用户要求新增算子错误码开发，或...", source: "ops" },

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
      { id: "ascendc-blaze-migration", description: "AscendC Matmul 家族等价迁移到 Blaze", source: "ops" },
      { id: "ascendc-simt-best-practices", description: "SIMT 最佳实践与 API 导航", source: "ops" },
      { id: "ascendc-simt-tiling-design", description: "SIMT 算子切分设计", source: "ops" },
      { id: "torch-ascendc-op-extension", description: "算子对接 PyTorch（kernel 直调 / aclnn 注册双路线）", source: "ops" },
      { id: "catlass-op-design", description: "Catlass 算子设计", source: "ops" },
      { id: "catlass-op-develop", description: "Catlass 算子开发", source: "ops" },
      { id: "catlass-op-perf-tune", description: "Catlass 性能调优", source: "ops" },
      { id: "ascendc-cross-gen-migration", description: "当用户希望将已有 AscendC 算子工程跨架构迁移（当前支持 A2/A3（910b/910_93、DAV_2201）→ A5（950、DAV_3510）与 A2/A3 → 310P（310P3...", source: "ops" },
      { id: "ascendc-mc2-best-practice", description: "Ascend C MC2 通算融合算子（多卡通信+计算融合）开发最佳实践。当用户需要设计、实现、调试、移植或优化通算融合算子，或提及\"MC2\"、\"SHMEM\"、\"APACE\"、\"通算融合\"、\"多...", source: "ops" },
      { id: "ascendc-sync-audit", description: "Ascend C 信号同步问题检验与修正。检测核内流水线同步（SetFlag/WaitFlag、SyncFunc、PipeBarrier、EnQue/DeQue）、核间同步（CrossCoreS...", source: "ops" },
      { id: "ascendc-tensor-api-best-practice", description: "在 Ascend 950 / DAV_3510 上以 Tensor API 为主、C API 为辅开发用户明确指定 Tensor API 的算子时使用。覆盖路线边界、事实调查、DESIGN/PL...", source: "ops" },
      { id: "ops-theoretical-performance-estimation", description: "从算子设计或计算流估算理论性能上限、瓶颈与目标时延，用于设计期方案比较和性能目标制定。当前仅支持 Ascend950 / DAV_3510 的 AscendC RegBase 纯 Vector ...", source: "ops" },
      { id: "ops-kirin-simulator", description: "Kirin 平台 NPU 仿真器技能。提供 CANN Simulator 的使用指导，包括精度仿真、性能仿真。当 Kirin 平台无 NPU 硬件需要仿真验证算子功能，或误用 npusim 报不...", source: "ops" },

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
      { id: "pypto-docs-search", description: "检索普通 PyPTO（Tensor）算子开发资源——API 文档、按错误码排障、教程/安装/工具文档、算子参考实现与 golden。当要查 PyPTO 文档、读算子或 API 用法、按错误码排障...", source: "ops" },
      { id: "pypto-general-debug", description: "PyPTO debugging router for stuck or opaque failures, including tile-shape/L0/L1/alignment/set_cub...", source: "ops" },
      { id: "pypto-memory-template", description: "PyPTO 算子开发中算子专属 memory 文件模板（custom/OP_NAME/MEMORY.md）。定义必备章节、机器可读字段与更新节奏。触发：创建或更新算子的 MEMORY.md 时使用。", source: "ops" },
      { id: "pypto-op-knowledge", description: "PyPTO 算子开发知识库查询技能。串行查询经验表和问题查找表，有结果立即返回，无结果返回\"无匹配\"。触发：查经验表、查问题表、知识库查询、knowledge query。", source: "ops" },
      { id: "pypto-op-perf-panko-manual", description: "Stage 7 PANKO world-model + evaluator contract. Read ONLY by pypto-op-auto-tuner-panko. Defines o...", source: "ops" },
      { id: "pypto-op-plan", description: "PyPTO 算子开发需求规划（requirement planning）——结构相似样例检索与可行性准备。触发：启动新算子开发的需求规划阶段、检索结构相似参考实现时使用。", source: "ops" },
      { id: "pypto-op-review", description: "Op-by-op PyPTO call extraction for debugging custom/OP_NAME/ kernels. Layout / structure validati...", source: "ops" },
      { id: "pypto-op-verify", description: "Validation runner requirements, detailed_tensor_compare usage, success criteria, required deliver...", source: "ops" },
      { id: "pypto-orchestration-manual", description: "pypto-op-orchestrator entry point. Bundles the 3 control documents (principles, team roster, mand...", source: "ops" },

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
      { id: "tilelang-perf-optimization", description: "性能调优与劣化检查", source: "ops" },
      { id: "tilelang-op-profiling", description: "TileLang NPU 性能采集与分析入口。适用于用户需要 Ascend950 算子上板性能采集、msprof 指标分析或性能验收时，读取独立的 Ascend950 技能。", source: "ops" },
      { id: "tilelang-performance-best-practices", description: "TileLang NPU 性能参考查询入口。适用于用户需要性能参考查询时，根据已选 Ascend910 或 Ascend950 平台读取独立技能；两套规则不混用。", source: "ops" },

    ],
  },
  {
    id: "triton",
    name: "Triton 开发",
    skills: [
      { id: "triton-task-extractor", description: "算子任务提取与构建", source: "ops" },
      { id: "triton-op-designer", description: "算法草图设计", source: "ops" },
      { id: "triton-op-coding", description: "Triton 内核代码生成", source: "ops" },
      { id: "triton-latency-optimizer", description: "Triton 代码性能优化", source: "ops" },
      { id: "triton-mfu-analyzer", description: "在昇腾 NPU 上测量单算子的 MFU（Model FLOPs Utilization，官方 operator_mfu 等价口径）， 并按 kernel 拆解瓶颈，回答\"低是低在哪一段、还有没有...", source: "ops" },
      { id: "triton-precision-debug", description: "Triton-Ascend 算子精度对齐调试专家。当算子精度校验（MERE/MARE）不通过时， 按照系统化的五阶隔离法定位 ULP 级差异根因，并提供修复方案。 尤其擅长处理编译器 scala...", source: "ops" },
      { id: "triton-simulator-optimizer", description: "Triton-Ascend 算子的 simulator 流水**采集与诊断**专家。用 msprof op simulator 采集 per-instruction pipe 统计，定位真实瓶颈...", source: "ops" },

    ],
  },
  {
    id: "model",
    name: "模型推理与训练",
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
      { id: "model-infer-harmony", description: "端侧鸿蒙 ASR 量化转换与打包", source: "model" },
      { id: "model-train-precision-numerical-mismatch", description: "PyTorch on NPU 训练正向/反向有限值偏差诊断", source: "model" },
      { id: "model-train-precision-nonfinite", description: "PyTorch on NPU 训练 NaN/Inf/Overflow 诊断", source: "model" },
      { id: "model-train-precision-determinism", description: "PyTorch on NPU 训练非确定性诊断", source: "model" },
      { id: "model-train-log-visualization", description: "NPU 大模型训练日志可视化与对比", source: "model" },
      { id: "model-train-oom-analysis", description: "PyTorch on NPU 训练 OOM 内存诊断与优化", source: "model" },
      { id: "science-model-npu-migration", description: "NPU 代码级迁移（环境门禁/脚本适配/精度性能对比）", source: "plugins-community/science-model-npu-migration", filePath: "plugins-community/science-model-npu-migration/SKILL.md" },
      { id: "model-infer-perf-breakdown", description: "NPU 性能数据拆解技能。把 kernel_details.csv 按用户描述的模型结构切成 component 实例（每层 attn / ffn / moe…），再按用户给的 cluster ...", source: "model" },
      { id: "model-infer-profiling", description: "NPU 性能分析数据采集技能。用于华为昇腾NPU上的 PyTorch 模型性能分析。 触发场景： - 用户需要采集 NPU 性能数据 - 用户提到 profiling、性能分析、tensorbo...", source: "model" },
      { id: "model-infer-quantization", description: "infer 仓模型量化适配改造技能。分析并接入既有 compressed-tensors 量化方案和权重，完成量化产物契约检查、结构参考匹配、量化 runtime 映射、权重加载、post-lo...", source: "model" },
      { id: "model-recommend-analysis", description: "推荐模型昇腾NPU性能分析与优化，分析瓶颈点及图结构中的相似结构，找到融合算子和pass机会；并给出优化建议。触发场景：推荐模型在昇腾NPU上性能不达标或客户推荐业务迁移到昇腾需要调优吞吐和时延...", source: "model" },

    ],
  },
  {
    id: "graph",
    name: "图模式",
    skills: [
      { id: "torch-npugraph-ex-knowledge", description: "npugraph_ex 使用指南", source: "graph" },
      { id: "torch-npugraph-ex-template", description: "npugraph_ex MRE 代码模板", source: "graph" },
      { id: "torch-npugraph-ex-runtime-error-diagnosis", description: "运行时报错诊断", source: "graph" },
      { id: "torch-npugraph-ex-performance-diagnosis", description: "性能诊断", source: "graph" },
      { id: "torch-custom-ops-guide", description: "自定义算子入图指南", source: "graph" },
      { id: "ge-fusion-pass-skill", description: "GE/CANN 自定义融合 pass 开发与诊断", source: "graph" },
      { id: "ge-coredump-diagnose", description: "昇腾 GE 图编译/图执行相关进程崩溃的 coredump 诊断。Use when（满足任一 GE 强信号才触发）： 1) backtrace、日志或用户描述中出现 GE 库名：libge_co...", source: "graph" },
      { id: "ge-dynamic-shape-diagnose", description: "昇腾 GE 动态调度/Unknown Shape 触发场景诊断。触发条件（Use when）：用户遇到图被标记为 unknown graph、触发动态 shape 调度、动态拆分、unknown...", source: "graph" },
      { id: "ge-inductor-autofuse-analysis", description: "融合性能分析。分析Profiling和Dump图，识别可融合范围；支持融合前后对比找出融合劣化片段和可融但未融合片段；支持Inductor+Triton/DVM/AutoFuse(AscendC...", source: "graph" },
      { id: "ge-memory-analysis", description: "GE 内存分析与优化。触发场景：① Device 显存问题——内存占用高、显存占用、HBM 偏大、OOM、内存不释放、显存持续增长、内存归因、内存池分析、devmm、imas；② 复用与寻优——...", source: "graph" },
      { id: "ge-onnx-plugin-development", description: "使用 ge.onnx_plugin 编写 ONNX Python 插件、把自定义算子接入 GE 时使用。触发场景：PyTorch 自定义算子导出 ONNX、编写 onnx_plugin/pars...", source: "graph" },
      { id: "ge-stream-log-analysis", description: "Analyze GE/CANN compile and runtime stream logs with optional GE repository context. Use when a u...", source: "graph" },
      { id: "ge-symbolic-op-development", description: "GE 图编译中的符号算子开发与解读。触发：用户需要新增或修改算子的符号推导 （infer_symbolic_shape/infer）、符号计算（symbolic_kernel），或排查 Symb...", source: "graph" },

    ],
  },
  {
    id: "runtime",
    name: "Runtime 迁移",
    skills: [
      { id: "runtime_migration", description: "CUDA 应用迁移到 CANN 平台指南", source: "runtime" },
      { id: "runtime-llt-generator", description: "为 CANN Runtime 的指定接口、函数、分支、缺陷或当前代码变更设计、编写并验证 LLT，支持通过 coverage_threshold 指定覆盖率门槛，未指定时默认 80%。用户要求“...", source: "runtime" },

    ],
  },
  {
    id: "platform",
    name: "平台工具",
    skills: [
      { id: "gitcode-pr-handler", description: "GitCode PR 标题/描述生成", source: "infra" },
      { id: "gitcode-issue-gen", description: "GitCode Issue 生成与关联", source: "infra" },
      { id: "gitcode-issue-handler", description: "GitCode 单 Issue/批量 Issue 自动处理", source: "infra" },
      { id: "gitcode-toolkit", description: "GitCode 协作通用参考", source: "infra" },
      { id: "cannbot-skill-reviewer", description: "Skill 入库质量审查", source: "infra" },
      { id: "asys-toolkit", description: "CANN 一键式故障信息收集工具 asys 使用指导。覆盖故障信息收集（collect）、业务复跑收集（launch）、软硬件与 Device 状态查询（info）、健康检查（health）、综...", source: "tools" },
      { id: "msaicerr-toolkit", description: "msaicerr 工具使用技能。用于：(1) 分析 AI Core Error 问题的故障信息，输出 info.txt 辅助定位，(2) 将 Dump 文件解析为 .bin/.npy 文件（算子...", source: "tools" },
      { id: "msnpureport-toolkit", description: "msnpureport 工具使用技能。用于：(1) 导出 Device 侧系统类日志和维测信息（单次导出、常驻进程连续导出），(2) 查询 Device 维测配置（日志级别、Coremask、加...", source: "tools" },
      { id: "npu-objdump-toolkit", description: "使用 npu-objdump/msobjdump 分析 Ascend 算子编译产物。触发：查看元数据、提取 device ELF、反汇编 AIC/AIV 指令，或结合报错日志中的异常 PC 定位...", source: "tools" },
      { id: "tool-exception-dump", description: "昇腾 NPU AI Core Error 系统化诊断工作流：环境与输入检查、故障信息收集、场景判定与报错行定位、错误类型识别、单算子复现、证据链汇聚与修复建议，基于 msaicerr/asys ...", source: "tools" },
      { id: "tool-npu-check", description: "npu-check NPU 运行正确性检查工具使用指导，覆盖检查能力、运行配置、命令用法和报告结论边界。触发：用户明确询问 npu-check、memcheck、synccheck，或要求解读 ...", source: "tools" },
      { id: "tool-npu-compute", description: "使用 npu-compute 采集和分析 NPU 性能指标，帮助用户生成或执行采集命令、查询工具支持的指标，并依据带宽、流水线、存储访问和资源冲突等采集结果定位性能问题。支持按指标集合（Set，...", source: "tools" },
      { id: "tools-npu-debug-api", description: "AscendC 调试工具接口使用说明，涵盖 printf、assert/ascendc_assert、__trap、 asc_dump、clock 和 asc_time_stamp 的接口选择、...", source: "tools" },

    ],
  },
  {
    id: "cannbot-dsl",
    name: "CANNBotDSL 开发",
    skills: [
      { id: "cannbot-dsl-best-practice", description: "为 CANNBotDSL 算子提供整体实现、局部写法和生产代码最佳实践。当需要职责划分、Host 约束校验、Channel/同步/布局、注册与适配、编译启动、AICPU 负载均衡 metadat...", source: "ops" },
      { id: "cannbot-dsl-code-reviewer", description: "检视调用方提供的 CANNBotDSL 算子代码，依据真实调用链核查生产执行约束、计算与导出路径，以及命名、魔数和可读性。当用户要求 DSL 代码审查、检视实现或复核 PR 时使用。", source: "ops" },
      { id: "cannbot-dsl-env-check", description: "当需要检查 CANNBotDSL 开发环境，或诊断导入、翻译、编译及 NPU 不可用问题时使用；输出环境诊断结果、可用能力与限制。", source: "ops" },
      { id: "cannbot-dsl-op-design", description: "根据算子规格编写分文件设计文档：数学语义、接口契约、形状与数据类型推导、算法与执行路径、任务调度、资源预算、同步及验证方案。当需要从算子规格形成方案或修订设计文档时使用。", source: "ops" },
      { id: "cannbot-dsl-test-generator", description: "当需要设计、补充或校验 CANNBotDSL 算子测试，或生成独立 CPU Golden 与测试入口时使用；根据调用方提供的规格、设计及实现单元检验义务生成黑盒、白盒和 TDD 用例及校验结果。", source: "ops" },
      { id: "cannbot-dsl-unit-planner", description: "根据 CANNBotDSL 算子的规格和调用方提供的设计资料，规划有依赖关系、可实现并独立验收的纵向实现 Unit，识别关键能力缺口和设计偏离风险，输出 U*.yaml。当用户要求拆分算子实现任...", source: "ops" },
    ],
  },
  {
    id: "catlass",
    name: "Catlass 开发",
    skills: [
      { id: "catlass-cpp-design", description: "CATLASS C++ 通用算子与已登记注意力家族五阶段工作流的 03 方案设计。用于按完整参考完成设计推演，再交付两章 docs/design.md，并应用 R01-R21 与 Stage/同...", source: "ops" },
      { id: "catlass-cpp-develop", description: "CATLASS C++ 通用算子与已登记注意力家族五阶段工作流的 04 算子开发。用于按已评审 Stage 实现 device kernel、host tiling 和直调入口，并进行逐 Sta...", source: "ops" },
      { id: "catlass-cpp-interface", description: "CATLASS C++ 通用算子和已登记注意力家族五阶段工作流的 01 接口确认。触发：开始开发 CATLASS C++ 算子、恢复接口阶段或接口语义变更时，用于初始化工程、确认并冻结 oper...", source: "ops" },
      { id: "catlass-cpp-knowledge", description: "从已安装的 cannbot-knowledge 查询 CATLASS C++ 接口、五阶段指南和注意力算子知识。触发：开发 CATLASS C++ 算子时需要查阅接口、阶段规则，或在设计阶段按 ...", source: "ops" },
      { id: "catlass-cpp-reference", description: "CATLASS C++ 通用算子与已登记注意力家族五阶段工作流的 02 标杆生成。用于维护唯一 PyTorch CPU 标杆、生成 definition、校准精度策略并冻结 golden con...", source: "ops" },
      { id: "catlass-cpp-test", description: "CATLASS C++ 通用算子与已登记注意力家族五阶段工作流的 05 算子测试，也可用于其他类型算子工程的用例泛化与用例设计覆盖评估。默认不询问用户、按工程类型分流：CATLASS C++ 五...", source: "ops" },
    ],
  },
  {
    id: "hccl",
    name: "通信与 HCCL",
    skills: [
      { id: "hccl-aicpu-best-practice", description: "HCCL AICPU 算法实现实践。触发：将已有算法设计或 Dataflow Spec 映射为 template 实现，修改已有 template，或核对 template 与 executor...", source: "communication" },
      { id: "hccl-aicpu-debug", description: "调试 HCCL AICPU 算法的构建、注册选路、Checker 数据错误或线程同步故障。触发：出现 AICPU 构建失败、目标算法未命中、Checker 失败或超时时使用。依据目标源码与原始日...", source: "communication" },
      { id: "hccl-aicpu-design", description: "基于目标 HCCL 版本的源码和算子契约设计 AICPU 算法的层级改动、数据流、资源与选路方案。触发：新增或修改 AICPU 算法，需要形成设计与 Dataflow Spec 时使用。输出分层...", source: "communication" },
      { id: "hccl-aicpu-perf-optimize", description: "为 HCCL AICPU 算法优化设计可复现的基线、候选测量与归因方法。用于明确的性能任务；要求真实目标环境与原始测量证据，不把 HCCL-VM Checker 或未标定成本系数当作性能收益。", source: "communication" },
      { id: "hccl-aicpu-whitebox-design", description: "从 HCCL AICPU 算法的真实源码与 Dataflow Spec 设计可追溯的分支、边界和数据流用例矩阵。触发：设计或审查 AICPU 白盒测试计划时使用；不执行测试、不修改生产源码或宣称...", source: "communication" },
      { id: "hccl-env-check", description: "触发：在 HCCL 算子开发、构建或测试前，需要只读检查 HCCL/HCOMM 仓库、CANN 路径和环境变量是否满足调用方约束时使用；只读输出 READY/NOT_READY、检查证据和缺项。", source: "communication" },
      { id: "hccl-env-setup", description: "触发：本机缺少可用的 HCCL 开发环境，需要 clone 仓库、探测或安装 CANN Toolkit+950 ops、生成环境变量文件时使用。", source: "communication" },
      { id: "hccl-test-tool-hvm", description: "触发：需要用 HCCL-VM 对 AICPU 集合通信算子执行 Checker、生成结构化证据、比较基线与回归、为失败/缺失用例生成定点补测计划，或在功能验证通过后用 estimate 模式估算...", source: "communication" },
    ],
  },
];

for (const cat of STATIC_SKILL_CATEGORIES) {
  for (const skill of cat.skills) {
    SUBCATEGORY_MAP[skill.id] = cat.id;
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

export function getAllCategories(): SkillCategory[] {
  const skills = getActiveSkills();
  const isEn = t("list_name") === "Name";
  return CATEGORY_DEFS.map((def) => ({
    id: def.id,
    name: isEn ? def.nameEn : def.name,
    // Category == repository scanDir (the skill's `source`), decided
    // automatically — no per-skill mapping. Skills with an unknown source
    // (e.g. a plugin-bundled SKILL.md) are still shown, grouped last.
    skills: skills.filter((s) => s.source === def.id),
  })).filter((cat) => cat.skills.length > 0)
    .concat(
      // catch-all for sources outside the known scanDirs (plugin skills etc.)
      (() => {
        const known = new Set(CATEGORY_DEFS.map((d) => d.id));
        const rest = skills.filter((s) => !known.has(s.source));
        return rest.length > 0
          ? [{ id: "other", name: isEn ? "Other Skills" : "其他 Skills", skills: rest }]
          : [];
      })()
    );
}

/** Returns the semantic subgroup of a skill (used for separator rows and
 * in-category ordering); undefined for skills without a known subgroup. */
export function getSkillSubgroup(id: string): string {
  return SUBCATEGORY_MAP[id];
}

/** A named subgroup inside a category (the wizard's second selection level
 * for large domains like ops). `skills` are the category's skills that map
 * to this subgroup. */
export interface SkillSubgroup {
  id: string;
  name: string;
  skills: SkillEntry[];
}

/** Groups a category's skills by their semantic subgroup. Subgroups come
 * from the legacy semantic categories (SUBCATEGORY_MAP) and carry their
 * display names; skills without a subgroup fall into a trailing
 * "ungrouped" bucket instead of being dropped. Sorted by size, descending. */
export function getCategorySubgroups(categoryId: string): SkillSubgroup[] {
  const category = getAllCategories().find((c) => c.id === categoryId);
  if (!category) return [];

  const names = new Map<string, string>();
  for (const cat of STATIC_SKILL_CATEGORIES) {
    if (!names.has(cat.id)) names.set(cat.id, cat.name);
  }

  const buckets = new Map<string, SkillEntry[]>();
  for (const skill of category.skills) {
    const group = SUBCATEGORY_MAP[skill.id];
    const key = group ?? "__ungrouped__";
    const bucket = buckets.get(key) ?? [];
    bucket.push(skill);
    buckets.set(key, bucket);
  }

  return [...buckets.entries()]
    .map(([id, skills]) => ({
      id,
      name: id === "__ungrouped__" ? "未分组" : (names.get(id) ?? id),
      skills,
    }))
    .sort((a, b) => b.skills.length - a.skills.length || a.name.localeCompare(b.name));
}
