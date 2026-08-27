# 同步问题修正模式

> 每类同步问题的修正模板。配合 [sync-checklist.md](sync-checklist.md) 使用。

---

## 修正原则

1. **先确认数据流，再选同步机制**：明确「谁产生数据、谁消费数据、跨几个 PIPE/核」
2. **标准三段流水优先 EnQue/DeQue**：自带同步，无需手动 Flag
3. **跨 PIPE 必须显式 Flag**：方向与数据流匹配
4. **跨核必须 CrossCore 对称**：所有路径都 SetFlag
5. **机制选择优先级（冲突时高优先级胜出）**：case_retriever 修复提示 > 场景规则（一次性初始化路径用 `PipeBarrier<PIPE_ALL>`，简洁且不消耗 eventID；双缓冲/热循环重叠用 SetFlag/WaitFlag 精确方向）> 同族文件风格。**同族风格只决定表面写法**（命名空间前缀、缩进、helper 是否存在），不得以「同族有 Flag 辅助函数」为由推翻前两级的机制选择
6. **修正后重跑 `sync_audit.py --check pair` 复核配对**

---

## 修正 diff API 签名自检（出手前必过）

> **真实失误案例**：修正 diff 把 `PipeBarrier` 写成 `AscendC::PipeBarrier<AscendC::HardEvent::PIPE_ALL>()`——`PIPE_ALL` 不是 `HardEvent` 成员，编译不过。根因：目标文件同族风格是全限定 `AscendC::HardEvent::V_MTE3`，套用全限定风格到 `PipeBarrier` 时错误连带了 `HardEvent::` 前缀。

| API | 模板参数枚举 | 合法示例 | 必错组合 |
|-----|------------|---------|---------|
| `PipeBarrier<>` | `PIPE_*` ISA 顶层 pipe 枚举（PIPE_ALL/PIPE_V/PIPE_M/PIPE_MTE1/PIPE_MTE2/PIPE_MTE3/PIPE_S/PIPE_FIX） | `AscendC::PipeBarrier<PIPE_ALL>();` | `PipeBarrier<HardEvent::PIPE_ALL>`、`PipeBarrier<AscendC::HardEvent::PIPE_ALL>` |
| `SetFlag<>/WaitFlag<>` | `HardEvent` 跨流水方向（31 种合法表见 [sync-mechanisms.md §3](sync-mechanisms.md)，其中**没有任何 `PIPE_*` 成员**） | `AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(id);` | `SetFlag<PIPE_V>`、`SetFlag<HardEvent::PIPE_ALL>` |

对生成的 diff 逐行自检：

1. 出现 `HardEvent::PIPE_` 字样 → **必错**，`PIPE_*` 与 `HardEvent` 不同源
2. `PipeBarrier<>` 尖括号内出现 `HardEvent` 或方向名（如 `V_MTE3`）→ 必错
3. `SetFlag/WaitFlag<>` 尖括号内出现 `PIPE_` → 必错
4. 全限定命名空间只加在函数名上（`AscendC::PipeBarrier<PIPE_ALL>()`），`PIPE_*` 枚举**不加** `AscendC::HardEvent::` 前缀；不确定顶层枚举归属时用 `ascendc-docs-search` 核实，或照抄 ops-nn/ops-tensor 真实代码写法

---

## FIX-SYNC-01/04: Wait 先于 Set / 配对不一致

### 修正步骤
1. 按 EVENT_ID/flag 分组，定位多/少的 Set 或 Wait
2. 调整顺序：SetFlag 先于 WaitFlag（同核内）
3. 补齐缺失的 Set 或删除多余的 Wait（多余 Wait 必删，多余 Set 建议删）

### 修正模板
```cpp
// 修正前：Wait 先于 Set + 个数不一致
WaitFlag<HardEvent::MTE2_V>(E0);
SetFlag<HardEvent::MTE2_V>(E0);
SetFlag<HardEvent::MTE2_V>(E0);   // 多余

// 修正后：Set 先于 Wait，1:1 配对
SetFlag<HardEvent::MTE2_V>(E0);
WaitFlag<HardEvent::MTE2_V>(E0);
```

---

## FIX-SYNC-02: 跨流水依赖缺同步

### 模式 A：标准三段流水 → 改用 EnQue/DeQue
```cpp
// 修正前：AllocTensor 后直接计算
LocalTensor<float> x = inQueue.AllocTensor<float>();
DataCopyPad(x, xGm[off], cp, pp);
Adds<float>(y, x, 1.0f, n);          // ❌ 缺 MTE2→V 同步

// 修正后
LocalTensor<float> x = inQueue.AllocTensor<float>();
DataCopyPad(x, xGm[off], cp, pp);
inQueue.EnQue(x);                     // 标记就绪
LocalTensor<float> xIn = inQueue.DeQue<float>();  // 阻塞等待 MTE2
Adds<float>(y, xIn, 1.0f, n);         // ✅
```

### 模式 B：自定义多 PIPE → 显式 Flag
```cpp
// 修正前：V 算完直接搬出（缺 V→MTE3）
Adds<float>(yLocal, xLocal, 1.0f, n);
DataCopyPad(yGm[off], yLocal, cp);    // ❌ 半成品可能存出

// 修正后
Adds<float>(yLocal, xLocal, 1.0f, n);
SetFlag<HardEvent::V_MTE3>(E0);       // ✅ V→MTE3 同步
WaitFlag<HardEvent::V_MTE3>(E0);
DataCopyPad(yGm[off], yLocal, cp);
```

### 模式 C：双 buffer 轮转，下一轮搬入前等上一轮 MTE3 搬完
```cpp
// 修正前：仅 PipeBarrier<PIPE_V>，跨 PIPE 无效
for (int i = 0; i < nLoop; i++) {
    ComputeV(buf[i % 2]);              // PIPE_V
    PipeBarrier<PIPE_V>();             // ❌ 不跨 PIPE
    DataCopyPad(wsGm[i*tile], buf[i%2], cp);  // PIPE_MTE3 读半成品
}

// 修正后：V→MTE3 同步；循环顶部上提 MTE3→MTE2 守护保证覆盖前已存完
// 首轮无上一轮，需条件保护
for (int i = 0; i < nLoop; i++) {
    if (i > 0) {
        SetFlag<HardEvent::MTE3_MTE2>(E0);
        WaitFlag<HardEvent::MTE3_MTE2>(E0);  // 上一轮 MTE3 已存完
    }
    ComputeV(buf[i % 2]);
    SetFlag<HardEvent::V_MTE3>(E0);   // ✅
    WaitFlag<HardEvent::V_MTE3>(E0);
    DataCopyPad(wsGm[i*tile], buf[i%2], cp);
}
```

### 模式 D：L0C 变化（Cube 侧）
> L0C 由 MMAD 写、Fixpipe 读 L0C→UB。L0A/L0B 重载（MTE1）后下一轮 MMAD 须等 MTE1 完成。
> 具体同步 API 随平台/版本演进，**修正前用 `ascendc-docs-search` 核实当前平台 MTE1_M / Cube 内部保序规则**。
```cpp
// 概念性修正（具体 API 以文档为准）
LoadData2D(...);                 // MTE1: L1→L0
SetFlag<HardEvent::MTE1_M>(E0);  // 等 L0 加载完成
WaitFlag<HardEvent::MTE1_M>(E0);
Mmad(...);                       // Cube: 用 L0A/L0B 算，写 L0C
// L0C→UB Fixpipe 由 Cube 内部保序或对应同步保证
```

---

## FIX-SYNC-03/08: 核间不对称 / 提前 return 跳过 SetFlag

### 模式 A：所有路径都 SetFlag
```cpp
// 修正前：edgeCase 跳过 SetFlag → 死锁
void ComputeMm1(...) {
    if (edgeCase) { return; }
    CrossCoreSetFlag(syncC1V1);
}

// 修正后
void ComputeMm1(...) {
    if (edgeCase) {
        CrossCoreSetFlag(syncC1V1);   // ✅ 提前退出前 Set
        return;
    }
    CrossCoreSetFlag(syncC1V1);
}
```

### 模式 B：单核/合并核 → 删除全部 CrossCore 同步
```cpp
// 修正前：单核 kernel 残留 CrossCore（无人 SetFlag → 死等）
CrossCoreWaitFlag<...>(syncX);   // ❌ 删除

// 修正后：单核场景删除全部 CrossCoreSetFlag/CrossCoreWaitFlag
// 若数据依赖仍存在，改用核内 PipeBarrier<PIPE_*> 或 EnQue/DeQue
```

### 模式 C：改用 Matmul 高阶 API → 删除手写 CrossCore
```cpp
// 修正前：手写 CrossCore + Matmul 高阶 API 混用 → flagId 冲突
CrossCoreSetFlag<...>(syncX);
matmul.IterateAll(...);

// 修正后：删除手写 CrossCore，依赖 Matmul 高阶 API 内部同步
matmul.IterateAll(...);
```

---

## FIX-SYNC-05: HardEvent 方向修正

### 修正步骤
1. 确认数据流方向（谁→谁）
2. 对照 [sync-mechanisms.md](sync-mechanisms.md) §3 选对 HardEvent
3. Set/Wait 同方向

```cpp
// 修正前：V→MTE3 写成 MTE3_V
SetFlag<HardEvent::MTE3_V>(E0);   // ❌ 方向反
WaitFlag<HardEvent::MTE3_V>(E0);
DataCopyPad(yGm[off], yLocal, cp);

// 修正后
SetFlag<HardEvent::V_MTE3>(E0);   // ✅
WaitFlag<HardEvent::V_MTE3>(E0);
DataCopyPad(yGm[off], yLocal, cp);
```

---

## FIX-SYNC-06: 跨 PIPE 误用 PipeBarrier<PIPE_V>

```cpp
// 修正前：V→MTE3 跨 PIPE 仅 PipeBarrier<PIPE_V>
Adds<float>(mid, x, 1.0f, n);
PipeBarrier<PIPE_V>();            // ❌ 不跨 PIPE
DataCopyPad(wsGm[off], mid, cp);

// 修正后：V→MTE3 显式 Flag
Adds<float>(mid, x, 1.0f, n);
SetFlag<HardEvent::V_MTE3>(E0);   // ✅
WaitFlag<HardEvent::V_MTE3>(E0);
DataCopyPad(wsGm[off], mid, cp);
```

---

## FIX-SYNC-07: flagId 冲突/超限

### flagId 超 15 次
```cpp
// 修正：复用 flagId 时确保单核单方向 ≤15 次；超过则改用 PipeBarrier 或重构循环
```

### 与 Matmul 高阶 API 混用
```cpp
// 修正：删除手写 CrossCore，依赖 Matmul 高阶 API 内部 flagId（见 FIX-SYNC-03 模式 C）
```

---

## FIX-SYNC-09: PipeBarrier 粒度收窄

```cpp
// 修正前：PIPE_ALL 过粗
PipeBarrier<PIPE_ALL>();

// 修正后：仅需 V 排序
PipeBarrier<PIPE_V>();
```
> ⚠️ 收窄前确认确实无跨 PIPE 依赖；若跨 PIPE 则不能用 PipeBarrier 替代（见 SYNC-06）。

---

## FIX-SYNC-10: loop 下溢保护

```cpp
// 修正前
uint32_t inIdx = (info.loop - 1) % preLoadNum;  // ❌ loop=0 下溢

// 修正后：首轮单独处理
uint32_t inIdx;
if (info.isFirstIteration) {
    inIdx = 0;  // 默认初始值
} else {
    inIdx = (info.loop - 1) % preLoadNum;  // loop >= 1
}
```

---

## FIX-SYNC-11: 去除冗余同步

```cpp
// 修正前：EnQue/DeQue 已同步，又加 Flag
DataCopyPad(x, xGm[off], cp, pp);
inQueue.EnQue(x);
SetFlag<HardEvent::MTE2_V>(E0);   // ❌ 冗余
WaitFlag<HardEvent::MTE2_V>(E0);
LocalTensor<float> xIn = inQueue.DeQue<float>();

// 修正后：依赖 EnQue/DeQue 隐式同步
DataCopyPad(x, xGm[off], cp, pp);
inQueue.EnQue(x);
LocalTensor<float> xIn = inQueue.DeQue<float>();  // ✅
```

---

## FIX-SYNC-12: SyncAll 分支对称 / 单核去除

```cpp
// 修正前：分支不对称 → 部分核死等
if (blockIdx < half) {
    ComputeA();
    SyncAll();        // ❌ 只有一半核到达
}

// 修正后：SyncAll 移到所有核都到达的位置
if (blockIdx < half) {
    ComputeA();
} else {
    ComputeB();
}
SyncAll();            // ✅ 所有核都到达
```
```cpp
// 单核场景：删除 SyncAll
if (GetBlockNum() == 1) { /* 无需 SyncAll */ }
```

---

## 修正自检清单

修正后用以下命令复核：
```bash
# 1. 配对复核
python3 scripts/sync_audit.py <file> --check pair
# 2. 全量扫描
python3 scripts/sync_audit.py <file>
# 3. 临时加 PipeBarrier<PIPE_ALL> 二分验证（若结果变正确说明仍有缺同步）
```
