---
id: H11
title: SetFlag/WaitFlag 不配对或 SetFlag 顺序错误
symptom: multicore_mismatch
when: intermittent
root_cause: sync_missing
evidence: tool_sanitizer
escalate_to: mssanitizer
source: mssanitizer-helper
---

## triggers
- 多核结果偶发性不一致（非确定性）
- synccheck 报告 SetFlag/WaitFlag 不配对
- 流水线 stage 之间数据竞争：某个阶段读到了上一轮的旧数据

## read_target
- `kernel/{op_name}.cpp` → grep `SetFlag\|WaitFlag` 统计数量
- 检查：每个 SetFlag 是否有对应的 WaitFlag，且位于正确的 pipe stage
- 注意：SetFlag 和 WaitFlag 的 event_id 参数是否匹配

## code_pattern
```cpp
// ❌ SetFlag 但没有对应 WaitFlag（或在错误位置）
SetFlag<HardEvent::V_MTE2>(eventId);
// ... 某些代码路径中缺少 WaitFlag ...
// 导致后续计算读到 MTE2 未完成的数据

// ❌ 事件 ID 复用（见 H12）
SetFlag<HardEvent::V_MTE2>(0);   // tile 0
// ... tile 1 中再次使用 event_id=0，覆盖了 tile 0 的状态
```

## fix_template
```cpp
// ✅ 每个 SetFlag 必须有对应的 WaitFlag
SetFlag<HardEvent::MTE2_V>(eventId);
// ... MTE2 操作（DataCopy）...
WaitFlag<HardEvent::MTE2_V>(eventId);
// 此后 Vector 计算才能安全读取数据

// ✅ 不同 tile 使用不同 event_id 或确保前一个 WaitFlag 完成后再复用
```

## verify_cmd
```bash
# 需要特殊编译选项
export ASCEND_LAUNCH_BLOCKING=1
# 在 .cmake 或编译命令中添加调试选项，然后运行 msSanitizer synccheck
# 详见 protocols/run_tools.md 或本地脚本 skills/ascendc-op-debug/scripts/mssanitizer_diagnose.py
```

## notes
- 代码审查（Layer 1）：grep SetFlag/WaitFlag 数量应配对
- 确认不配对后升级到 msSanitizer synccheck（Layer 3）精确定位
- 事件 ID 复用问题见 H12
