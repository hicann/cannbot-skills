---
name: cannbotdsl-runtime-debug
description: "定位 CANNBotDSL 编译错误和运行时错误（非 NPU 精度问题）时使用。当编译失败需要分类（A 语法/AST 闭包与控制流改写、B IR/lowering 类型不匹配与 SSA dominance、C 设计层 Buffer 超限与 sync 死锁、D bisheng AscendC 编译、F 框架能力缺失），或遇到运行时错误（.so 加载失败、符号缺失、ctypes 参数 marshalling 错误、KernelArgTensor 结构体不匹配）时触发。含错误分类章节：5 种错误类型定义、CODE vs FW 4 步判断决策树、每种类型的处理路由、错误模式库。Triggers: cannbotdsl 编译错误, 运行时错误, 错误分类, A/B/C/D/F 型, .so 加载, ctypes marshalling, DESIGN_ERROR 判定。Developer 在 Stage 3 调用。"
---

# cannbotdsl-runtime-debug

CANNBotDSL 编译错误和运行时错误（非 NPU 精度、非 crash/hang）的分类与定位。核心动作：把错误归到 A/B/C/D/F 五型之一，再路由到对应处理流程。Developer 在 Stage 3 调用。

**真实来源以源码为准**。

## 触发条件

- 编译失败需要分类定位（trace/IR/lowering/translate/bisheng 哪一层）
- 运行时错误：`.so` 加载失败、符号缺失、ctypes marshalling 错误

## 错误分类：5 型（A/B/C/D/F）

| 类型 | 名称 | 出现阶段 | 典型征兆 | 归属 |
| ---- | ---- | -------- | -------- | ---- |
| **A** | 语法/前端 | trace（AST 预处理） | early-exit raise、`range` 参数数、闭包变量缺失 | CODE |
| **B** | IR / lowering | IR build / verify / lowering | 类型不匹配、SSA dominance 违反、VF 区域形成失败 | CODE |
| **C** | 设计层 | verify / runtime | Buffer 超限、sync 死锁、L0 容量 | CODE（回 Stage 2 设计） |
| **D** | bisheng | compile | AscendC C++ 编译失败 | CODE（codegen 或写法） |
| **F** | 框架能力 | 任意 | API 标 Planned/未实现、某特性行为异常且非本地代码问题 | **FW**（转 `cannbotdsl-framework-probe`） |

## CODE vs FW 判断决策树（4 步）

疑似问题时，**不要直接假设是框架 bug**，按序排查：

1. **能定位到自己代码的具体错误吗？**（拼错 API、shape 不符、sync 漏配对）→ CODE（A/B/C/D）。
2. **换最小复现还在吗？** 拆一个 10-30 行最小 kernel（单 tile、单核）复现；若最小 case 正常 → 是自己 kernel 的组合问题，CODE。
3. **最小 case 仍异常，且落在 `local_slice`/`mem_copy`/sync/dynamic control flow/transpose 等工具链路径？** → 可能 F 型，进第 4 步。
4. **查 `skills/probes/` 有无结论？** 有 → 按结论（PASS/FAIL/WORKAROUND/BLOCKED）走；无 → 写 probe 确认（`cannbotdsl-framework-probe`）。只有能定位到前端/IR/lowering/translator/runtime 的具体问题，才归 F 型框架缺陷。

> 铁律：**F 型必须有 probe 或源码 raise 佐证**。查不到证据就是"未确认"，继续按 CODE 排查，不要在框架限制的假设上反复改代码，也不要把自己的 bug 甩给框架。

## 各型处理路由

- **A 型**：读 `../../core-skills/cannbotdsl-programming-model/SKILL.md §3` 控制流改写规则；改 Python 写法（const_expr / 去 early-exit / 显式传参）。
- **B 型**：读 verify 失败时打印的完整 IR module，定位是哪一步 lowering、哪种类型不匹配。
- **C 型**：回 Stage 2 用 `cannbotdsl-op-design` 重算 Buffer 预算 / sync 序列；返回 `DESIGN_ERROR`。
- **D 型**：读 bisheng stderr，确认生成的 AscendC 是否合法 C。
- **F 型**：转 `cannbotdsl-framework-probe`，写 probe、找 workaround。

## 运行时错误定位（编译成功后）

运行时按 IR 生成 → 编译 → ctypes 加载 `.so` → 参数构造 → NPU 执行定位：

- **`.so` 加载失败 / 符号缺失**：查 `LD_LIBRARY_PATH`（torch/torch_npu/install/cann lib）；符号缺失多为链接期库缺失。
- **ctypes 参数 marshalling 错误**：`_RuntimeTensor.to_kernel_arg` 构造 `KernelArgTensor_<dtype>_<ndim>d`；dtype/ndim 与 kernel 签名不符即错位。elemType 映射须与 `TranslateToAscendC.cpp` 的 `elemTypeToCpp` 一致。
- **NPU 执行异常（crash/hang）**：转 `cannbotdsl-crash-debug` + `cannbotdsl-npu-plog-diagnosis`。

### sync_block 跨核 GM 同步原语

当跨核数据通过 GM 中转（非 Channel handoff）时，需用 `sync_block` 原语做跨核同步：

```python
from cannbotdsl import (vec_sync_block_arrive, vec_sync_block_wait,
                     cube_sync_block_arrive, cube_sync_block_wait)
from cannbotdsl.typing.types import PIPE

# Vec 写 GM 后通知 Cube
vec_sync_block_arrive(PIPE.MTE3, flag_id)

# Cube 等 Vec 写完再读 GM
cube_sync_block_wait(PIPE.MTE3, flag_id)

# Cube 写完后通知 Vec（反向同步）
cube_sync_block_arrive(PIPE.MTE2, flag_id + 1)

# Vec 等 Cube 处理完
vec_sync_block_wait(PIPE.MTE2, flag_id + 1)
```

**参数约束**：

| 参数 | 约束 | 违反现象 |
|------|------|---------|
| `pipe` | 必须在 `[0,1,4,5]`（S/V/MTE1/MTE2） | `PIPE.FIXPIPE`(11) 被 bisheng 拒绝：`the ranges of 1st parameter must be [0, 1], [4, 5]` |
| `flag_id` | 必须静态 Python int | 动态 SSA 值不工作（同步不生效，读到陈旧数据或全零） |
| `mode` | 默认 `2`，一般不改 | — |

### double-free 诊断

**症状**：`free(): double free detected in tcache 2`，发生在 IR build 阶段。

**历史结论已失效**：旧文档把多条跨核 Channel 与 factory scratch 的组合归因于独立 arena 冲突。当前 `Buffer` 与 Channel 共享同一个地址 allocator，自动地址不会重叠；应按真实 IR/设备日志继续定位容量、显式 alias 或同步问题。

**排查方向**：
1. 检查 func 内跨核 channel 总数（`Σ depth ≤ 8`）
2. 核对 Buffer/Channel 总物理字节数以及所有显式 `addr=` alias 的生命周期
3. 尝试减少跨核 channel 数量（如将 Vec→Cube handoff 改为 GM 中转 + sync_block）
4. 只有临时量确实需要多级 / 同步语义时才改为 Channel；单块 scratch 保持 Buffer
5. 若 sync 计数配平但仍 hang/crash → 查分发轴顺序致某核负载远超平均（causal 下 m-block 轴放 `idx2crd` 最内层且整除 GRID → 每核工作量恒定不均）。先用 host 侧算术算每核负载，见 `../../core-skills/cannbotdsl-perf-optimize/SKILL.md` 第 0 步

## 门禁

- 报告必须给出错误型别（A/B/C/D/F）+ 定位到的**具体阶段/文件行**，不同层级不混写。
- F 型结论必须有 probe 或源码 raise 佐证；无证据标"未确认"，按 CODE 继续。
- C 型必须回 Stage 2 设计，不在实现层打补丁绕开预算/死锁。

## 参考

- `runtime.py`（marshalling）、`core/compiler/cache.py`（缓存）
- （cannbotdsl-framework-probe 在本仓不可用）（F 型）、`../cannbotdsl-crash-debug/SKILL.md`（crash/hang）
