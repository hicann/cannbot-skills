---
skill_name: ascendc-crash-debug
eval_mode: text
---
# Case 1: Segmentation Fault 崩溃调试

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

我在开发 Ascend C 算子时遇到了 Segmentation Fault 崩溃，程序无法正常运行完毕。请问应该如何系统性地定位和解决这个问题？有哪些调试工具和方法可以使用？不需要执行任何工具调用。

## Expected Output

回复应说明 Segmentation Fault 的系统化调试方法：
- 通过系统配置（如 coredump）来捕获崩溃现场信息
- 利用调试工具分析崩溃现场获取关键定位信息（如调用栈回溯）
- 空指针解引用或内存越界是 Segmentation Fault 的常见原因
- 如堆栈信息不清晰，可使用内存检测工具辅助定位

---

# Case 3: Ascend C 算子卡死问题诊断请求（正向看护）

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 170000
- Ascend Platform: A2
- Distractor skills: ascendc-runtime-debug;ascendc-precision-debug

## Prompt

我的 Ascend C 算子在 NPU 上运行到一半就卡死了，Kernel 没有任何响应，也没有报错信息就直接超时了。请加载 ascendc-crash-debug 技能帮我诊断可能的原因。

## Expected Output

回复应说明 Kernel 卡死的诊断思路：分析 plog 日志辅助定位；检查 EnQue/DeQue 管道同步是否匹配；检查 AllocTensor/FreeTensor 的 Buffer 配对。

---

