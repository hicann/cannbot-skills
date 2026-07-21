---
id: H12
title: 事件 ID 复用导致硬件状态机混乱
symptom: hang
when: multicore_only
root_cause: event_id_reuse
evidence: tool_sanitizer
escalate_to: mssanitizer
source: mssanitizer-helper
---

## triggers
- 507034 超时错误（Vector Core 超时）
- 多 tile 循环时出现挂起，单 tile 正常
- synccheck 报告事件 ID 相关的状态机错误

## read_target
- `kernel/{op_name}.cpp` → grep `SetFlag.*event\|WaitFlag.*event\|EVENT_ID`
- 检查：tile 循环中是否多次使用同一个 event_id（0, 1, 2...）
- 检查：上一个 WaitFlag 是否在下一次 SetFlag 之前完成

## code_pattern
```cpp
// ❌ tile 循环中重复使用 event_id=0，前一个 Wait 未完成就重新 Set
for (uint32_t t = 0; t < dimLoop; ++t) {
    DataCopy(localBuf, gmBuf[t * size], size);
    SetFlag<HardEvent::MTE2_V>(0);   // 每次都用 event_id=0
    // ... 计算 ...
    WaitFlag<HardEvent::MTE2_V>(0);  // 可能与上一次的 Set 状态混乱
}
```

## fix_template
```cpp
// ✅ 方案1：双 buffer 轮换 event_id（ping-pong）
for (uint32_t t = 0; t < dimLoop; ++t) {
    uint8_t eventId = t % 2;   // 0 和 1 交替使用
    DataCopy(localBuf[eventId], gmBuf[t * size], size);
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
    // 处理 localBuf[eventId]
}

// ✅ 方案2：使用 TQue（推荐，框架自动管理事件 ID）
TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
// 框架自动处理 SetFlag/WaitFlag，无需手动管理 event_id
```

## verify_cmd
```bash
# 运行 msSanitizer synccheck 定位具体的事件 ID 冲突
# 详见 protocols/run_tools.md
mssanitizer --tool=synccheck bash run.sh
```

## notes
- 507034 是 Vector Core 超时的错误码，通常由同步问题引发
- TQue 是最安全的方案：框架自动管理事件 ID 和同步
- 详细诊断流程见 protocols/run_tools.md → mhc_post_fusion 超时专项
