---
skill_name: torch-ops-profiler
---

# Case 1: 自定义算子性能 Profiling 与对比报告

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

我需要为一个自定义 Ascend C 算子 layer_norm 进行性能评估，比较自定义算子与 PyTorch 标杆的性能差异。算子目录为 ascend-kernel/csrc/ops/layer_norm/。请按照 torch-ops-profiler 的流程生成性能用例并输出对比报告。

## Expected Output

回复应先读取 csrc/ops/layer_norm/design.md 提取 dtype、参数约束、典型 shape、执行模式，再读取 test/layer_norm-test-cases.md（如存在）提取用例信息。然后设计 JSONL 用例（≥8 条，覆盖所有 dtype 和执行模式，包含小/中/大规模 shape），使用 torch_npu.profiler 进行性能采集（warmup=5、active=5 固定），建立双路径对比（自定义算子 vs 标杆 API 或小算子拼接）。最终产出 <op>_perf_cases.jsonl 用例文件和 <op>_torch_npu_profiler_report.md 性能报告，报告中应包含含 DType 列的统一对比表、全量汇总、按数据类型汇总和 ≥3 条简短分析。

## Expectations
- [contains] torch_npu.profiler
- [contains] warmup=5
- [contains] active=5
- [contains] JSONL
- [contains] design.md

---

# Case 2: 标杆路径决策与使用边界

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

使用 torch-ops-profiler 时，算子没有对应的标杆等价 API，应该怎么办？这个 skill 和 ascendc-precision-debug 有什么区别？

## Expected Output

回复应说明当算子无标杆等价 API 时，必须实现小算子拼接标杆路径，以设计文档中的 PyTorch 参考实现为基础，使用 PyTorch 基础算子组合（如 torch.zeros、切片赋值、.permute()、torch.cat 等）构建标杆路径，且标杆实现必须以张量操作为主（非 Python 标量循环），确保可在 NPU 上执行。报告中须明确标注「无标杆等价接口，标杆路径为小算子拼接」。禁止以「无标杆接口」为由输出单路径报告。torch-ops-profiler 专注于性能 profiling 和对比，而 ascendc-precision-debug 专注于精度调试和问题定位，两者分工不同。应强调用例生成必须先读 design.md，否则可能导致 shape 不符合约束或漏掉关键执行模式。

## Expectations
- [contains] 小算子拼接
- [contains] 双路径对比
- [contains] ascendc-precision-debug
- [contains] design.md
