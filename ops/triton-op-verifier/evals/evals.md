---
skill_name: triton-op-verifier
---

# Case 1: Triton 算子代码验证与性能测试

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

我生成了一个 Triton Ascend softmax 算子的内核代码，需要验证其功能正确性并采集性能数据。生成代码路径为 /workspace/output/iter_0/softmax_triton_ascend_impl.py，任务文件路径为 /workspace/tasks/softmax.py，算子名为 softmax。请按照 triton-op-verifier 的标准验证流程执行。

## Expected Output

回复应按照标准验证流程执行：先使用 scripts/validate_triton_impl.py 对生成代码进行退化预检查（检测三种退化类型：完全无 @triton.jit kernel、有 kernel 但 forward() 未调用、部分计算使用 PyTorch），通过后再创建验证项目文件（xxx_torch.py 和 xxx_triton_ascend_impl.py），然后使用 scripts/verify.py 执行正确性验证，验证通过后使用 scripts/benchmark.py 进行性能测试。验证基于 MERE/MARE 双门限相对误差判定，而非 torch.allclose。验证结果应包含 verify_result.json 中的通过/失败情况和 perf_result.json 中的性能数据。

## Expectations
- [contains] validate_triton_impl.py
- [contains] verify.py
- [contains] benchmark.py
- [contains] MERE
- [contains] MARE

---

# Case 2: 验证前置条件与使用边界

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

使用 triton-op-verifier 需要什么前置条件？在什么情况下应该触发这个 skill？验证失败后应该怎么处理？

## Expected Output

回复应说明 triton-op-verifier 的触发条件是用户需要验证 Triton 算子代码功能正确性或采集其性能数据时。前置条件是需要有生成的内核代码（包含 ModelNew 类的完整 Python 文件）和对应的任务文件（包含 Model 类）。验证失败时应读取 verify_result.json 中的 failures 条目汇总错误信息（包括 error_type 如 CompilationError 和 error_msg），将完整错误信息提交给上游流程进行迭代修复。应强调禁止自己编写 Python 代码来测试算子，必须使用本 skill 自带的 scripts/verify.py 脚本。benchmark.py 内置了 L1 verify 闸门，verify 未全过时禁止运行 benchmark。

## Expectations
- [contains] ModelNew
- [contains] verify.py
- [contains] verify_result.json
- [contains] benchmark
