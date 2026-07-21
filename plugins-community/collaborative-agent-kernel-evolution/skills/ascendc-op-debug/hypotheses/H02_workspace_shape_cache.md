---
id: H02
title: Workspace 跨 Shape 缓存越界
symptom: crash
when: cross_shape_reuse
root_cause: workspace_undersize
evidence: code
escalate_to: null
source: ascendc-debug.md#案例2
---

## triggers
- 单独测试任何一个 shape 均通过
- 同一进程中先运行小 shape（小 D/小 N）再运行大 shape → 崩溃
- 错误信息含 DDR out-of-range / 内存越界 / coredump
- 重启进程后用大 shape 单独测试反而正常

## read_target
- `op_host/{op_name}_custom.cpp` → 查 workspace 大小计算
  - grep: `workspace\|coreWsBytes\|totalWsBytes\|SetWorkspaceSize`
- 检查：workspace 计算公式中是否引用了当前输入的 D、N、S 等变量

## code_pattern
```cpp
// ❌ 按当前 shape 动态计算 workspace
// D=2560 时分配 N*2560*4B，D=5120 时需要 N*5120*4B 但 runtime 仍用缓存的小值
size_t coreWsBytes = (size_t)(N + 1) * D * sizeof(float);  // D 是当前输入！
size_t totalWsBytes = (size_t)usedCoreNum * coreWsBytes;   // usedCoreNum 也可能变
context->SetWorkspaceSize(totalWsBytes);
```

## fix_template
```cpp
// ✅ 按所有支持 shape 的最大值分配
const uint32_t MAX_D = 5120;   // 支持的最大 D
const uint32_t MAX_N = 4;      // 支持的最大 N
// 关键：必须用物理核总数 coreNumAiv，而非当前使用核数 usedCoreNum
// 因为 runtime 按物理核号索引 workspace，usedCoreNum 可能小于实际核数
size_t coreWsBytes  = (size_t)(MAX_N + 1) * MAX_D * sizeof(float);
size_t totalWsBytes = (size_t)aclrtGetCoreNumAiv() * coreWsBytes;
context->SetWorkspaceSize(totalWsBytes);
```

## verify_cmd
- 同一进程中按 D=[2560, 5120] 顺序连续调用（不能分进程），验证大 shape 不崩溃
- 测试序列：`for D in [2560, 5120]: for BS in [1,4,8,16]: test()`

## notes
- CANN runtime 在首次调用时缓存 workspace 地址和大小，后续调用不重新分配
- 必须用 `coreNumAiv`（平台物理 AI Core 总数）而非 `usedCoreNum` 计算 total
- 测试时必须在同一 Python 进程中从小到大遍历所有 shape 组合才能复现
