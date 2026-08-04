---
name: tilelang2ascend-trace-recorder
description: >
  执行 trace 记录员 Skill。在算子任务完成后，回顾整个执行过程，
  生成结构化的 trace 记录供 meta-agent 优化使用。
  当算子任务完成后需要记录执行过程时，使用此 skill。
argument-hint: >
  输入：output_dir 目录路径、各阶段执行结果信息。
  输出：{output_dir}/trace.md 结构化执行记录。
---

# Trace 记录 Skill

你是一名执行 trace 记录员。你的目标是在算子任务完成后，回顾整个执行过程，生成结构化的 trace 记录。

## 关键限制
- 只允许在 `{output_dir}/` 目录下创建 `trace.md` 文件
- 不要修改 `{output_dir}/` 中的任何其他文件
- 只记录事实，不做改进建议（那是 meta-agent 的工作）

## 信息来源

回顾本次会话中的以下信息：
- 各阶段的执行结果（成功/失败）
- 评测脚本的输出（`@scripts/evaluate_tilelang.sh`、`@scripts/evaluate_ascendc.sh` 的返回）
- ops-profiling（msprof 模式）的性能测试结果
- Agent 的迭代过程（尝试了什么、失败了几次、最终如何解决）
- 遇到的错误信息

补充约定：
- TileLang 功能验证（evaluate_tilelang.sh）为强制步骤：默认必须执行且精度必须通过（精度通过是 Phase 3 Step 4 性能迭代的强制前置）；交付的 correctness gate 仍以 AscendC 验证为准
- 未执行 TileLang 验证属流程违规，必须在"走偏点"中记录；仅当对照实验证明编译器/框架底层不支持时才可合法跳过，应如实记录为"跳过"并写明原因
  （含框架侧问题的对照实验证据，如 fp16 通过 / fp32 失败；若做了 fp16/bf16 代理验证，记录其结论）

## 输出格式

将以下内容写入 `{output_dir}/trace.md`：

```markdown
# Trace: {算子名称}

- 时间: {当前日期时间}
- 算子: {output_dir 对应的算子名}
- 设计路径: {design.md (简单算子) / TileLang (复杂算子)}
- 最终结果: SKIP / PASS / FAIL (tilelang) | PASS / FAIL (ascendc)

## 阶段零: Case 精简

- 结果: 通过 / 失败 / 跳过
- 原始 case 数: {n}
- 精简后 case 数: {n}
- 备注: {如有异常情况}

## 阶段〇.五: 设计文档 (仅简单算子路径)

- 结果: 通过 / 失败 / 跳过
- 说明: 简单算子路径生成 design/design.md；复杂算子路径填写"跳过(走 TileLang 路径)"
- 备注: {如设计文档修正次数}

## 阶段一: TileLang (仅复杂算子路径)

- 结果: 通过 / 失败 / 跳过
- evaluate_tilelang.sh 执行次数: {n}
- 关键错误信息: {评测脚本返回的错误，原文引用}
- 性能设计检查（1e/2e 强制项，依据 design/PERF_DESIGN.md 记录）:
  - 算子类型判定: {纯 Cube / 纯 Vector / CV 融合}
  - 1e block 级初检: {命中的反模式与设计期修正；无命中则写"全部未命中/不适用"}
  - 2e tile 级终检: {命中的反模式与设计期修正；无命中则写"全部未命中/不适用"}
  - 高级策略路由结论: {Cube 型写 cube_advanced_strategies 选用/预留策略；其余写"不适用"}
  - PERF_DESIGN.md 缺失或缺项 → 在"走偏点"中记录为流程违规
- 性能迭代（步骤 4 强制项，依据 perf_tuning/ 记录）:
  - 结果: 达标 / 预算耗尽未达标 / 合法跳过（SKIPPED.md，附原因与对照实验证据）
  - 基线 geomean: {baseline.json 数值} → 最终 geomean: {final_report.md 数值}
  - p_retry 轮数与已实施优化点: {逐项名称 + 各自收益，依据 optimization_log.md 的 [RESULT-#N]}
  - 未达标时: {final_report.md 的上限分析结论（roofline/Amdahl）摘要}
  - perf_tuning/ 缺失必需产物 → 在"走偏点"中记录为流程违规
- Agent 行为记录:
  - 第 1 轮: {agent 做了什么，结果如何}
  - 第 2 轮: {修改了什么，结果如何}
  - ...
- 走偏点: {agent 做了哪些无效/错误/冗余的尝试，以及可能的原因}

## 阶段二: AscendC

- 结果: 通过 / 失败
- 设计路径: design.md (简单算子) / TileLang 转译 (复杂算子)
- 产物: kernel/op_host/<op>.cpp, kernel/op_kernel/<op>.cpp, kernel/ops.h, kernel/register.cpp
- 编译: evaluate_ascendc.sh (cmake + make + whl)
- evaluate_ascendc.sh 执行次数: {n}
- 关键错误信息: {评测脚本返回的错误，原文引用}
- Agent 行为记录:
  - 第 1 轮: {agent 做了什么，结果如何}
  - 第 2 轮: {修改了什么，结果如何}
  - ...
- 走偏点: {agent 做了哪些无效/错误/冗余的尝试，以及可能的原因}


## 阶段三: 性能分析

- 结果: 完成
- ops-profiling（msprof 模式）执行详情:
  - 测试配置: device=npu, warmup=5, repeat=10, seed=0
  - 测试的实现: reference / tilelang / ascendc
  - 总体统计:
    - reference: mean=0.086ms, median=0.070ms, min=0.050ms, max=0.362ms, std=0.046ms
    - tilelang: mean=0.327ms, median=0.202ms, min=0.147ms, max=3.284ms, std=0.503ms
    - ascendc: mean=0.186ms, median=0.090ms, min=0.054ms, max=2.443ms, std=0.366ms
  - 性能结论:
    - 三个实现均执行成功（Status=OK）
    - AscendC 整体快于 TileLang，按 mean 统计约快 1.76x（0.327 / 0.186）
    - AscendC 仍慢于 reference，按 mean 统计约为 reference 的 0.46x；reference 约快 2.16x
    - TileLang 慢于 reference，按 mean 统计约为 reference 的 0.26x；reference 约快 3.80x
  - 典型大 shape case 观察:
    - case[47] shape=(1, 8, 16384, 64), float16, half: reference=0.169ms, tilelang=1.065ms, ascendc=0.626ms
    - case[48] shape=(1, 8, 32768, 64), float16, half: reference=0.261ms, tilelang=1.915ms, ascendc=1.155ms
    - case[49] shape=(1, 8, 32768, 64), bfloat16, interleave: reference=0.292ms, tilelang=3.271ms, ascendc=2.434ms

## 汇总表报告

- 说明: 延迟单位为 ms，按 ops-profiling（msprof 模式）的 mean 统计；加速比 =  PyTorch 参考延迟/生成 AscendC 代码延迟 。加速比>0.6性能0.6x pytorch填是，否则填否。性能0.8x pytorch同理。

| Level | Problem ID | 算子名称 | 算子类型 | 编译通过 | 精度正确 | PyTorch 参考延迟 | 生成AscendC代码延迟 | 加速比 | 最终状态 | 精度正确 | 性能0.6x pytorch | 性能0.8x pytorch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | RotaryMul | vector | ✅ | ✅ | 0.086 | 0.186 | 0.46 | 成功 | 是 | 否 | 否 |

## 评测输出摘要

{粘贴最后一次 evaluate 脚本的关键输出片段，包括 PASS/FAIL 状态和错误详情}
```

### 记录原则

1. **精确引用**: 错误信息、评测输出使用原文，不要改写或总结
2. **行为序列**: 每轮迭代记录 agent 的实际操作（改了什么文件、改了什么逻辑），而非笼统的"修复了 bug"
3. **走偏分析**: 重点记录 agent 做了哪些最终被证明无效的尝试，这是 meta-agent 优化 harness 的核心输入
4. **省略成功**: 如果某阶段一次通过且无异常，简要记录即可，不需要展开
5. **如实跳过**: TileLang 未验证不是异常；如果流程按约定跳过，应明确记录“跳过”及原因，不要误记为失败
