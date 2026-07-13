# 工作流加载规则

## 性能优化任务触发（最高优先级）

当用户请求涉及 AscendC 算子性能优化时（关键词包括但不限于：性能优化、性能调优、perf-optimize、算子优化、加速比、上板性能、cases.csv、msprof、profiling 调优），**必须**按以下顺序执行：

1. **先读取工作流定义**：`.opencode/agents/perf-optimize.md`
2. **读取 subagent prompt 模板**：按 perf-optimize.md 中引用的 `workflows/task-prompts.md`（位于 `perf-optimize.md` 同目录的 `workflows/` 下，需通过其符号链接源目录解析）
3. **按工作流 Step 1→2（→3）顺序调度 subagent**：
   - Step 1：启动 `ascendc-perf-analysis-expert` 采集性能数据并输出《性能调优方案》
   - Step 2：按方案启动 `ascendc-perf-impl-expert` 实施优化，主 agent 统一采集对比
   - Step 3（可选）：接入 CANN-Bench 评测
4. **禁止**主 agent 自己直接参与分析、策略制订或代码实施
5. **禁止**跳过工作流直接开始优化

## 工作流核心约束（摘要）

- 全量 case 覆盖：cases.csv 每个 case 都要分析报告（上限 20）
- 主 agent 只做调度和统一 msprof 采集，不做分析/实施
- 方案融合与并列：不同模板分支互斥时分析阶段融合为 1 方案；同分支冲突才并列对比
- 多轮优化需用户显式 opt-in（max_rounds 或 performance_goal）

## 完整工作流定义

详见 `.opencode/agents/perf-optimize.md`，本文件仅为触发入口。
