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

回复应说明 Segmentation Fault 的系统化调试流程：
- 首先启用 coredump：ulimit -c unlimited，重新运行程序生成 core 文件
- 使用 GDB 分析 coredump：gdb <exe> <core>，执行 bt 和 bt full 命令查看完整调用栈
- 使用 info locals 查看局部变量值辅助定位
- 空指针解引用或内存越界是 Segmentation Fault 的常见原因
- 如果堆栈不清晰，可使用 mssanitizer 工具进行内存错误主动检测

## Expectations

---

# Case 2: Kernel 卡死/超时问题诊断

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

我的 Ascend C 算子在运行时出现卡死/超时问题，Kernel 无响应。请问应该从哪些方面入手排查？有哪些调试方法和工具可以帮助定位问题？不需要执行任何工具调用。

## Expected Output

回复应说明 Kernel 卡死/超时的诊断方法：
- 分析 plog 日志，搜索 ERROR、timeout 等关键词定位卡死位置
- 检查 Watchdog 超时机制，确认是否触发 AICORE_TIMEOUT
- 排查死锁：Pipe 的 EnQue/DeQue 是否匹配，pipe_barrier 位置是否正确
- 使用 printf/Dump 调试、pipeDump、msprof 等工具辅助定位

## Expectations
