---
skill_name: pypto-precision-debug
---

# Case 1: 精度问题排查与规避方法

## Config
- Max Tokens: 120000
- Ascend Platform: A2

## Prompt

我的 PyPTO Add 算子精度验证失败，输出结果和 golden 有偏差。请帮我排查精度问题。算子的实现使用了 @pypto.frontend.jit 装饰器，但某些 shape 下（如尾轴为 1 时）精度异常。

## Expected Output

回复应说明精度调试的排查流程：检查前端写法、检查用户代码语法、按优先级尝试快速规避方法、必要时转入 precision-compare 二分定位。

## Expectations
- [contains] pypto.frontend.jit
- [contains] unroll_list
- [contains] submit_before_loop
- [contains] 规避方法

---

# Case 2: 职责范围与使用边界

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

pypto-precision-debug 和 pypto-precision-compare 这两个 skill 在精度问题上如何分工？什么情况下应该先用 precision-debug，什么情况下应该直接用 precision-compare？precision-debug 能定位到框架层面的问题吗？

## Expected Output

回复应说明 pypto-precision-debug 与 pypto-precision-compare 的分工：debug 负责用户代码层面的语法检查和规避方法，不负责底层框架根因排查。当 debug 的规避方法无效时转入 compare 进行精确定位。

## Expectations
- [contains] 用户代码层面
- [contains] 不负责
- [contains] pypto-precision-compare
- [contains] 规避方法
