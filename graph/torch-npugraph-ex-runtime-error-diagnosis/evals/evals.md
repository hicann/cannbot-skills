---
skill_name: torch-npugraph-ex-runtime-error-diagnosis
eval_mode: text
---

# Case 1: 运行时错误诊断流程

## Config
- Max Tokens: 180000
- Max Tokens (deepseek-v4-flash): 210000
- Max Tokens (glm-5): 195000
- Ascend Platform: A2

## Prompt

我的模型在 npugraph_ex 图捕获成功后运行时出错，报了 aclnn 相关错误。如何诊断这类运行时错误？

## Expected Output

回复应说明运行时错误诊断流程：定位第一个因果错误（ACL/HCCL/aclnn/OOM/stream-event 等），而非下游级联失败。应说明启用 CANN 日志（ASCEND_GLOBAL_LOG_LEVEL、ASCEND_SLOG_PRINT_TO_STDOUT）、从 ~/ascend/log/ 采集 plog 并与崩溃栈对齐时间戳。

## Expectations
- [contains] aclnn
- [contains] ASCEND_GLOBAL_LOG_LEVEL


---

# Case 2: HCCL 通信错误排查

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 165000
- Ascend Platform: A2

## Prompt

npugraph_ex 运行时报了 HCCL 通信错误，可能的原因有哪些？如何定位？

## Expected Output

回复应说明 HCCL 通信错误的常见原因：通信超时、Rank 间同步失败、链路异常等。应说明排查方法包括检查 HCCL 环境变量、查看通信日志、确认集群网络状态。

## Expectations
- [contains] HCCL


---

# Case 3: OOM 与 Stream/Event 同步问题

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 165000
- Ascend Platform: A2

## Prompt

npugraph_ex 运行时设备内存不足（OOM）或者 stream/event 同步出错，应该如何诊断？

## Expected Output

回复应说明 OOM 的排查方法：检查 CANN plog 中内存分配失败上下文，区分 device OOM、workspace 不足、内存碎片等类型，设置环境变量（如 ASCEND_GLOBAL_LOG_LEVEL）辅助诊断。应说明 stream/event 同步问题的诊断：检查跨流依赖、stream id、event 状态，确认同步原语使用是否正确。回复应说明先锚定首个致因报错再逐层排查的诊断流程，并按固定输出格式组织回复（问题归类 → 证据 → 最可能根因 → 下一步最小动作）。

## Expectations
- [contains] OOM
- [contains] stream


---

# Case 4: 正向看护-多 graph skill 环境下正确触发

## Config
- Max Tokens: 180000
- Max Tokens (deepseek-v4-flash): 210000
- Max Tokens (glm-5): 195000
- Distractor skills: torch-npugraph-ex-compile-error-diagnosis;torch-npugraph-ex-dfx-triage;torch-npugraph-ex-knowledge;torch-npugraph-ex-performance-diagnosis
- Ascend Platform: A2

## Prompt

我的模型在 npugraph_ex 图捕获完成后运行时发生 segfault，报了 aclnn 错误码。请帮我诊断这个运行时错误。

## Expected Output

回复应激活 torch-npugraph-ex-runtime-error-diagnosis skill，针对 segfault 和 aclnn 错误给出初步诊断方向，并收集必要的诊断信息（报错日志、复现脚本、环境信息等）。即使在多个 npugraph_ex 诊断 skill 共存的环境下，也应正确激活 runtime-error-diagnosis。

## Expectations
- [contains] aclnn
- [skill_activated] torch-npugraph-ex-runtime-error-diagnosis
