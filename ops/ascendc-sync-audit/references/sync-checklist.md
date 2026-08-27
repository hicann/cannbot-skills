# 信号同步检验条例

> 14 条同步检验条例。每条含：严重级别、问题、检测、判定、示例、检视命令。
>
> **用户提出的 4 类问题**对应 SYNC-01/02/03/04；SYNC-05~13 为补充场景。
>
> 严重级别：`[红线]`（正确性/死锁）/ `[高]`（高风险）/ `[性能]`（不影响正确性但损性能）。

---

## 快速索引

| 条例 | 名称 | 级别 | 类别 |
|------|------|------|------|
| SYNC-01 | Wait 先于 Set（先 wait 后 set） | 红线 | 错序 |
| SYNC-02 | 跨流水数据依赖缺同步 | 红线 | 缺失 |
| SYNC-03 | 核间同步对称性 + 残留同步该取消未取消 | 红线 | 核间 |
| SYNC-04 | Set/Wait 个数一致、EVENT_ID/flag 匹配 | 红线 | 配对 |
| SYNC-05 | HardEvent 方向与数据流匹配 | 红线 | 方向 |
| SYNC-06 | 跨 PIPE 依赖误用 PipeBarrier<PIPE_V> 替代 | 红线 | 缺失 |
| SYNC-07 | flagId 复用 ≤15、禁与 Matmul 高阶 API 混用 | 红线 | 冲突 |
| SYNC-08 | 提前 return/break 跳过 SetFlag | 红线 | 死锁 |
| SYNC-09 | PipeBarrier 粒度（PIPE_ALL 过粗 / 连续 >3） | 性能 | 性能 |
| SYNC-10 | 双缓冲 loop 下溢（(loop-1)%N, loop=0） | 高 | 下溢 |
| SYNC-11 | 同步冗余/混用 | 性能 | 冗余 |
| SYNC-12 | SyncAll 全核同步必要性与 `<false>` 语义 | 高 | 全核 |
| SYNC-13 | AtomicAdd 乱序：同流水内原子操作语义顺序需 PipeBarrier 保证 | 红线 | 语义顺序 |

---

## SYNC-01: Wait 先于 Set（先 wait 后 set）`[红线]`

### 问题
同一 EVENT_ID/flag，`WaitFlag` 出现在 `SetFlag` 之前（按执行序），WaitFlag 会死等前序完成，导致 Kernel 挂起。

> 对应用户关注点 (1)。

### 检测
- 脚本：`sync_audit.py --check pair`，同 EVENT_ID/flag 内 Wait 行号 < Set 行号
- 人工：确认二者在同一执行序（非跨函数合理场景，如 AIC 先 Set 后 AIV Wait 属跨核合理序）

### 判定
- 同核内同 EVENT_ID，Wait 先于 Set → **风险（死等）**
- 跨核（CrossCore）：AIC 侧 Set、AIV 侧 Wait，属合理序，**不报**

### 错误示例
```cpp
// ❌ Wait 先于 Set → 死等
WaitFlag<HardEvent::MTE1_M>(EVENT_ID);
SetFlag<HardEvent::MTE1_M>(EVENT_ID);
```

### 正确示例
```cpp
// ✅ SetFlag 先执行
SetFlag<HardEvent::MTE2_S>(EVENT_ID7);
WaitFlag<HardEvent::MTE2_S>(EVENT_ID7);
```

### 检视命令
```bash
grep -n "SetFlag\|WaitFlag" <file>
```

---

## SYNC-02: 跨流水数据依赖缺同步 `[红线]`

### 问题
异步搬运（MTE2/MTE3）、Cube（M）/Vector（V）间数据依赖缺少同步，会读到未完成或半成品数据。**编译期不报错。**

> 对应用户关注点 (2)：L0C buffer 变化、Vector 搬出到 GM 前、搬入到计算之间、下一轮 Vector 前等 MTE3 搬完。

### 检测（按数据流逐点核对）
| 数据依赖 | 必须的同步 | 检测点 |
|---------|-----------|--------|
| MTE2 搬入 → V 计算 | EnQue/DeQue 或 `SetFlag/WaitFlag<MTE2_V>` | 每个 `DataCopy(Data)Pad`(GM→UB) 后、Vector 计算前 |
| V 计算 → MTE3 搬出 | EnQue/DeQue 或 `SetFlag/WaitFlag<V_MTE3>` | 每个 Vector 计算后、`DataCopy(Data)Pad`(UB→GM) 前 |
| 下一轮 V 前，上一轮 MTE3 搬完 | `SetFlag/WaitFlag<MTE3_MTE2>` 或等价 | 循环中复用同 buffer 的下一轮搬入前 |
| L0C 变化（MMAD→Fixpipe） | Cube 内部保序 / `MTE1_M` | MMAD 写 L0C 后 Fixpipe 读 L0C→UB；L0A/L0B 重载后下一轮 MMAD |
| V 内 store→load（RegBase） | `Reg::LocalMemBar<VEC_STORE, VEC_LOAD>` | `Reg::StoreAlign`/`Reg::DataCopy(mem,reg)` 后、`Reg::LoadAlign`/`Reg::DataCopyUnAlign(reg,mem)` 前 |

### 判定
- `AllocTensor` 后 `DataCopy`，未经 `EnQue/DeQue` 或 Flag 直接 Vector 计算 → **风险**
- Vector 计算后未经同步直接 `DataCopy` 出 GM → **风险**
- 双 buffer 轮转循环，下一轮搬入前无 MTE3→MTE2 同步 → **风险**（读到上一轮未存完的半成品）
- RegBase 模式下 `Reg::DataCopy(addr, reg)` 写入后、`Reg::DataCopyUnAlign(reg, addr)` 读取前缺 `Reg::LocalMemBar<VEC_STORE, VEC_LOAD>` → **风险**（V 内 store→load 乱序，参考 cannbot-dsl `vmem_bar("vst_vld")` 范式）

### 错误示例
```cpp
// ❌ MTE2 搬入后直接计算（缺 MTE2→V 同步）
LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
Adds<float>(yLocal, xLocal, 1.0f, count);  // 可能读到未完成搬运的数据

// ❌ V 算完直接搬出（缺 V→MTE3 同步）
Adds<float>(yLocal, xLocal, 1.0f, count);
DataCopyPad(yGm[offset], yLocal, copyParams);  // 半成品可能被存出

// ❌ 双 buffer 轮转，下一轮搬入前无 MTE3→MTE2 同步
// （Pass-1 存 workspace 后 Pass-2 读回，仅 PipeBarrier<PIPE_V> 不够）
```

### 正确示例
```cpp
// ✅ EnQue/DeQue 自动同步
DataCopyPad(xLocal, xGm[gmOffset], copyInParams, padParams);
inQueueX.EnQue(xLocal);
LocalTensor<float> xIn = inQueueX.DeQue<float>();  // 阻塞等待 MTE2
Adds<float>(yLocal, xIn, 1.0f, count);
outQueueY.EnQue(yLocal);
LocalTensor<float> yOut = outQueueY.DeQue<float>();  // 阻塞等待 V
DataCopyPad(yGm[gmOffset], yOut, copyOutParams);

// ✅ 自定义多 PIPE：V→MTE3 显式 Flag
SetFlag<HardEvent::V_MTE3>(EVENT_ID);
WaitFlag<HardEvent::V_MTE3>(EVENT_ID);
DataCopyPad(yGm[offset], yLocal, copyParams);
```

### 检视命令
```bash
grep -n "DataCopy\|DataCopyPad\|EnQue\|DeQue\|SetFlag\|WaitFlag\|PipeBarrier" <file>
```

---

## SYNC-03: 核间同步对称性 + 残留同步该取消未取消 `[红线]`

### 问题
两类核间同步缺陷：
1. **不对称**：`CrossCoreSetFlag`/`CrossCoreWaitFlag` 不一一对应，或某条路径跳过 SetFlag → WaitFlag 永久阻塞死锁。
2. **残留该取消未取消**：迁移/重构后（多核→单核、手写→Matmul 高阶 API、Cube+Vec 合并），残留的 CrossCore 同步无人 SetFlag → 死等。

> 对应用户关注点 (3)：核间同步验证，所有同步是否都取消（指该取消的是否都取消了）。

### 判定
- 每个 `CrossCoreWaitFlag` 必须有且仅有一个对称 `CrossCoreSetFlag`
- 所有能到达 `CrossCoreWaitFlag` 的代码路径（含提前 return/break）都必须触发对应 `CrossCoreSetFlag`
- **单核 / 合并核场景**：所有 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 应已删除；残留即死等源
- **改用 Matmul 高阶 API**：手写 CrossCore 同步应删除（高阶 API 内部已用 flagId，混用冲突，见 SYNC-07）
- Flag 常量应复用参考工程 `cv_sync_constants.h`，禁止自定义 Flag ID

### 错误示例
```cpp
// ❌ 提前 return 跳过 SetFlag → Vec 侧 WaitFlag 永久等待
void ComputeMm1(...) {
    if (edgeCase) {
        return;  // Vec1 的 WaitFlag(C1V1) 死等
    }
    CrossCoreSetFlag(syncC1V1);  // 只有正常路径执行
}

// ❌ 单核场景残留 CrossCore 同步（无人 SetFlag → 死等）
// 单核 kernel 内仍有 CrossCoreWaitFlag(syncX) 但无对应 SetFlag
```

### 正确示例
```cpp
// ✅ 所有路径都 SetFlag
void ComputeMm1(...) {
    if (edgeCase) {
        CrossCoreSetFlag(syncC1V1);  // 提前退出前必须 Set
        return;
    }
    CrossCoreSetFlag(syncC1V1);
}

// ✅ 单核场景：删除全部 CrossCore 同步
```

### 检视命令
```bash
grep -n "CrossCoreSetFlag\|CrossCoreWaitFlag" <file>
# 单核判定：GetBlockNum() == 1 或无 __mix__ 分核时，CrossCore* 应为 0
grep -n "GetBlockNum\|__mix__\|GetTaskRation" <file>
```

---

## SYNC-04: Set/Wait 个数一致、EVENT_ID/flag 匹配 `[红线]`

### 问题
`SetFlag`/`WaitFlag`（含 CrossCore）个数不一致，或成对使用的 EVENT_ID/flag 不匹配，导致同步失效或死等。

> 对应用户关注点 (4)：set 和 wait 的个数是否一致。

### 判定
- 按 EVENT_ID（或 flag 变量名）分组，每组 `SetFlag` 数 == `WaitFlag` 数
- 成对的 Set/Wait 必须用**相同** EVENT_ID/flag
- 多余的 Set（无 Wait 消费）→ 浪费但通常不致命；多余的 Wait（无 Set）→ **死等**

### 错误示例
```cpp
// ❌ 个数不一致：2 个 Set，1 个 Wait
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

// ❌ EVENT_ID 不匹配
SetFlag<HardEvent::MTE2_V>(EVENT_ID7);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID8);  // 不同 ID，同步失效
```

### 正确示例
```cpp
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
```

### 检视命令
```bash
grep -c "SetFlag" <file>; grep -c "WaitFlag" <file>   # 总数应一致
python3 scripts/sync_audit.py <file> --check pair      # 按 EVENT_ID 分组核对
```

---

## SYNC-05: HardEvent 方向与数据流匹配 `[红线]`

### 问题
HardEvent 类型标识同步的流水线对，必须与实际数据流方向匹配。方向反了同步无效。

### 判定
对照 [sync-mechanisms.md](sync-mechanisms.md) §3 HardEvent 方向表（31 种合法方向）：
- MTE2 搬入→V 计算：应用 `MTE2_V`，误用 `V_MTE2` → **风险**
- V 计算→MTE3 搬出：应用 `V_MTE3`，误用 `MTE3_V` → **风险**
- 双 buffer 轮转（存完才能覆盖）：应用 `MTE3_MTE2`，误用 `MTE2_MTE3` → **风险**
- **笔误/未知方向**：不在 31 种合法方向内（如 `MTE2_V` 写成 `MET2_V`）→ 脚本可检出

> ⚠️ 注意：`V_MTE2` 与 `MTE3_V` 等是真实存在的合法方向（非笔误），不能一律当错。

### 错误示例
```cpp
// ❌ 实际是 V 算完→MTE3 存出，方向写反
SetFlag<HardEvent::MTE3_V>(EVENT_ID);  // 应为 V_MTE3（注意 MTE3_V 本身合法，此处是方向选错）
WaitFlag<HardEvent::MTE3_V>(EVENT_ID);
DataCopyPad(yGm[offset], yLocal, copyParams);
```

### 正确示例
```cpp
SetFlag<HardEvent::V_MTE3>(EVENT_ID);
WaitFlag<HardEvent::V_MTE3>(EVENT_ID);
DataCopyPad(yGm[offset], yLocal, copyParams);
```

### 检视
- **脚本侧**：能识别不在 31 种合法方向内的未知方向（疑似笔误），报 SYNC-05「高」级别
- **人工侧**：「该处实际需要哪个方向」依赖数据流，须人工结合 [sync-mechanisms.md](sync-mechanisms.md) §3 判定（脚本无法判定方向是否匹配数据流，只能判笔误）

---

## SYNC-06: 跨 PIPE 依赖误用 PipeBarrier<PIPE_V> 替代 SetFlag/WaitFlag `[红线]`

### 问题
`PipeBarrier<PIPE_V>` 只保证 V-V 排序，**不跨 PIPE**。跨 PIPE 数据依赖（V→MTE3、MTE2→V、MTE3→MTE2）若只用 PipeBarrier<PIPE_V>，会读到半成品中间结果，编译期不报错。

> 来源：`catlass grouped-matmul.md` §2 典型事故模式。

### 判定
- 凡跨 PIPE 依赖（不同 PIPE 间数据流转），必须有对应方向的 `SetFlag/WaitFlag`，不可用 `PipeBarrier<PIPE_V>` 替代
- 典型：V 写中间 UB buffer → MTE3 存 workspace，二者间仅 `PipeBarrier<PIPE_V>` → **风险**

### 错误示例
```cpp
// ❌ V→MTE3 跨 PIPE，仅 PipeBarrier<PIPE_V> 不够
Adds<float>(midLocal, xLocal, 1.0f, count);   // PIPE_V
PipeBarrier<PIPE_V>();                         // ❌ 不跨 PIPE
DataCopyPad(wsGm[offset], midLocal, params);   // PIPE_MTE3 读半成品
```

### 正确示例
```cpp
Adds<float>(midLocal, xLocal, 1.0f, count);
SetFlag<HardEvent::V_MTE3>(EVENT_ID);
WaitFlag<HardEvent::V_MTE3>(EVENT_ID);
DataCopyPad(wsGm[offset], midLocal, params);
```

### 检视
定位 `PipeBarrier<PIPE_V>` 后紧跟跨 PIPE 操作（DataCopy/DataCopyPad 出 GM、MMAD 等）的场景。

---

## SYNC-07: flagId 复用 ≤15、禁与 Matmul 高阶 API 混用 `[红线]`

### 问题
- 同一 flagId 计数器最多设置 15 次，超过行为未定义
- `CrossCoreSetFlag` 内部已被 Matmul 高阶 API 使用，二者混用有 flagId 冲突风险

> 来源：`ascendc-api.md` API-12 官方约束（8.5.0）。

### 判定
- 统计同一 flagId 的 `CrossCoreSetFlag` 调用次数，>15 → **风险**
- 同一 kernel 同时出现 `CrossCoreSetFlag/WaitFlag` 与 `Matmul`/`MatmulSimple` 等高阶 API → **风险**

### 检视命令
```bash
grep -n "CrossCoreSetFlag\|CrossCoreWaitFlag" <file>
grep -n "Matmul\|MatmulSimple\|IterateAll" <file>
```

---

## SYNC-08: 提前 return/break 跳过 SetFlag `[红线]`

### 问题
`CrossCoreSetFlag`/`SetFlag` 后于 `return`/`break`/`continue`，或被条件分支绕过，导致对应 `WaitFlag` 永久阻塞死锁。

### 判定
- 对每个 `CrossCoreSetFlag`/`SetFlag`，向上追踪是否存在 `if/return/break/continue` 绕过它的路径
- 所有能到达对应 `WaitFlag` 的路径，都必须能触发该 SetFlag

### 错误示例
```cpp
// ❌ edgeCase 路径跳过 SetFlag
void ComputeMm1(...) {
    if (edgeCase) { return; }       // 跳过 → WaitFlag 死等
    CrossCoreSetFlag(syncC1V1);
}
```

### 正确示例
```cpp
void ComputeMm1(...) {
    if (edgeCase) {
        CrossCoreSetFlag(syncC1V1); // 提前退出前必须 Set
        return;
    }
    CrossCoreSetFlag(syncC1V1);
}
```

### 检视
脚本按函数级括号追踪，标注 SetFlag 前的 return/break；人工确认分支可达性。

---

## SYNC-09: PipeBarrier 粒度（PIPE_ALL 过粗 / 连续 >3）`[性能]`

### 问题
- `PipeBarrier<PIPE_ALL>` 全流水停顿，性能差；同 PIPE 内本只需 `PipeBarrier<PIPE_V>` 却用 PIPE_ALL → 不必要阻塞
- 同一函数内连续超过 3 个 `PipeBarrier<PIPE_V>` → 应分析是否可合并

> 来源：`ascendc-perf.md` PERF-4、`mc2-specific.md` MC2-02。
> 说明：业务代码倾向冗余 PipeBarrier 以确保安全，属可接受工程实践，不标记为错误。

### 判定
- `PipeBarrier<PIPE_ALL>` 可用 `PipeBarrier<PIPE_V>` 替代 → 性能建议（不标错）
- 连续 >3 个 `PipeBarrier<PIPE_V>` → 性能问题

---

## SYNC-10: 双缓冲 loop 下溢（(loop-1)%N, loop=0）`[高]`

### 问题
双缓冲常用 `(loop - 1) % N` 计算 buffer 索引。`loop` 为 `uint32_t` 且等于 0 时，`loop - 1` 下溢为 `UINT32_MAX`，导致错误 buffer 索引。

> 来源：`ascendc-api.md` API-12 双缓冲 loop 偏移下溢风险。

### 判定
- 出现 `(loop - k) % N` 形式的 buffer 索引，且 `loop` 为无符号类型 → 检查 `loop < k` 时是否有保护分支
- 首次迭代应单独处理，不读上一轮

### 错误示例
```cpp
// ❌ loop=0 时 uint32_t 下溢
uint32_t inIdx = (info.loop - 1) % preLoadNum;
```

### 正确示例
```cpp
if (info.isFirstIteration) {
    // 默认初始值，不读上一轮
} else {
    uint32_t inIdx = (info.loop - 1) % preLoadNum;  // loop >= 1，安全
}
```

---

## SYNC-11: 同步冗余/混用 `[性能]`

### 问题
EnQue/DeQue 已提供隐式同步的位置，又额外加 `SetFlag/WaitFlag`/`PipeBarrier`，造成冗余停顿，损性能。

### 判定
- 标准 TQue 三段流水（Alloc→DataCopy→EnQue→DeQue→计算→EnQue→DeQue→DataCopy）内，无需再手动 Flag
- 同一数据依赖被多种机制重复保护 → 性能建议
- ⚠️ 误删有效同步是红线，判定冗余须确认确有其他机制覆盖

---

## SYNC-12: SyncAll 全核同步必要性与 `<false>` 语义 `[高]`

### 问题
- `SyncAll` 要求所有核都到达，分支不对称会导致部分核死等
- `SyncAll<false>` 同步所有核含 AIC，语义误用导致意外阻塞
- 单核/无多核场景误用 SyncAll 造成不必要开销

> 来源：`mc2-specific.md` MC2-03、术语表。

### 判定
- `SyncAll` 所在分支是否所有核都会到达？（条件分支不对称 → 死等）
- 多核 V 核间数据防踩踏是否需要 SyncAll？
- 单核场景（GetBlockNum()==1）SyncAll 应去除

### 检视命令
```bash
grep -n "SyncAll" <file>
```

---

## SYNC-13: AtomicAdd 乱序 `[红线]`

### 问题

`SetAtomicAdd` 改变了 FIX 流水操作的语义（覆盖→累加）。当 splitK 场景中第一份数据标记非累加（覆盖）、其余数据标记累加时，若 FIX 流水内操作乱序执行，非累加部分可能出现在累加之后，覆盖此前累加结果 → 精度错误。

**关键认知**：同流水内硬件保序**不适用于 atomic 操作语义**。`SetAtomicAdd`/`DisableDmaAtomic` 改变了操作的写语义，即使同流水也需要 `PipeBarrier<PIPE_FIX>` 保证语义顺序。

### 判定

- `SetAtomicAdd` 后紧跟 Fixpipe/DataCopy 写 GM，中间无 `PipeBarrier<PIPE_FIX>` → 红线候选
- 原子模式切换前后要确认所有使用该模式的写操作已完成
- 典型场景：splitK 场景，`isSplitSingleK_ && !isFirstSplitK_` 时 `SetAtomicAdd`，`Fixpipe`/`DataCopy` 累加写 GM

### 错误示例

```cpp
// ❌ SetAtomicAdd 后直接 Fixpipe/DataCopy，atomic 乱序可能覆盖累加结果
if (isSplitSingleK_ && !isFirstSplitK_) {
    AscendC::SetAtomicAdd<float>();
}
AscendC::DataCopy(cGlobal, c1Local, intriParams);  // FIX 写 GM，可能乱序
if (isSplitSingleK_ && isEndSplitK_) {
    AscendC::DisableDmaAtomic();
}
```

### 正确示例

```cpp
// ✅ SetAtomicAdd 后加 PipeBarrier<PIPE_FIX> 保证累加顺序
if (isSplitSingleK_) {
    PipeBarrier<PIPE_FIX>();
    if (!isFirstSplitK_) {
        AscendC::SetAtomicAdd<float>();
    }
}
AscendC::DataCopy(cGlobal, c1Local, intriParams);
if (isSplitSingleK_ && isEndSplitK_) {
    AscendC::DisableDmaAtomic();
}
```

### 检视命令

```bash
grep -n "SetAtomicAdd\|DisableDmaAtomic\|Fixpipe\|DataCopy.*cGlobal" <file>
```

### 核心教训

来自 ops-nn `block_mmad_pingpong_without_que.h` 真实 issue：
- 预期：data0(覆盖) + data1(累加) + data2(累加) + data3(累加) → 精度正确
- 实际：data1(累加) + data2(累加) + data0(覆盖) + data3(累加) → data1/data2 结果丢失
- 修复：每次 Fixpipe/DataCopy 前加 `PipeBarrier<PIPE_FIX>()` 保证语义顺序

---

## SYNC-14: 同步信号与 buffer 索引一致性 `[红线]`

### 问题
双缓冲/多缓冲中，`SetFlag/WaitFlag` 的 flag id 与实际读写的 buffer id 必须来自同一个 ring index。若同步信号使用 `ubComputeLoopIdx_`，但 producer 或 consumer 访问 `ubMte2LoopIdx_` 派生的 output buffer，会出现同步保护的 buffer 与实际写入/读出 buffer 不一致。

### 判定
- `SetFlag/WaitFlag<HardEvent::X_Y>(eventId[loopA & ...])` 保护的 buffer 访问使用 `buffer_[loopB * size]`，且 `loopA != loopB` → 红线候选
- 局部别名必须追根：`uint64_t ubMte2BufferIdx = (ubMte2LoopIdx_ - 1) & ...; buffer_[ubMte2BufferIdx * size]` 的根索引是 `ubMte2LoopIdx_`
- producer 侧也必须检查：`params.biasOutUbAddr = ubBiasOutTotalBuffer_[idx * ...].GetPhyAddr()`、`params.outputPhyAddr`、`params.dstPhyAddr` 等输出地址字段若与同一输出参数对象中的其他输出字段索引不同 → 红线候选
- 修复时优先统一 output buffer id；不要用额外 `SetFlag/WaitFlag` 掩盖 producer 写址错误

### 错误示例
```cpp
params.weightHighBitPhyAddr =
    (__ubuf__ half *)ubHighBitTotalBuffer_[(ubComputeLoopIdx_ & 3) * highSize].GetPhyAddr();
params.biasOutUbAddr =
    (__ubuf__ half *)ubBiasOutTotalBuffer_[ubMte2BufferIdx * biasSize].GetPhyAddr();

DataCopy(weightL1, ubHighBitTotalBuffer_[(ubComputeLoopIdx_ & 3) * highSize], params);
DataCopy(biasL1, ubBiasOutTotalBuffer_[((ubMte2LoopIdx_ - 1) & 1) * biasSize], biasSize);
SetFlag<HardEvent::MTE3_V>(eventId[ubComputeLoopIdx_ & 3]);
```

### 正确示例
```cpp
const uint64_t outBufIdx = ubComputeLoopIdx_ & (outputBufferNum - 1);
params.weightHighBitPhyAddr =
    (__ubuf__ half *)ubHighBitTotalBuffer_[outBufIdx * highSize].GetPhyAddr();
params.biasOutUbAddr =
    (__ubuf__ half *)ubBiasOutTotalBuffer_[outBufIdx * biasSize].GetPhyAddr();

DataCopy(weightL1, ubHighBitTotalBuffer_[outBufIdx * highSize], params);
DataCopy(biasL1, ubBiasOutTotalBuffer_[outBufIdx * biasSize], biasSize);
SetFlag<HardEvent::MTE3_V>(eventId[outBufIdx]);
```

---

## 检视检查清单

- [ ] **SYNC-01**: 是否有 Wait 先于 Set（同核同 EVENT_ID）？
- [ ] **SYNC-02**: 搬入→计算、计算→搬出、下一轮复用、L0C 变化是否都有同步？
- [ ] **SYNC-03**: CrossCore Set/Wait 是否对称？单核/合并核是否残留？提前 return 是否跳过 SetFlag？
- [ ] **SYNC-04**: Set/Wait 个数是否一致？EVENT_ID/flag 是否匹配？
- [ ] **SYNC-05**: HardEvent 方向是否与数据流匹配？
- [ ] **SYNC-06**: 跨 PIPE 依赖是否误用 PipeBarrier<PIPE_V> 替代 Flag？
- [ ] **SYNC-07**: 同一 flagId 是否 ≤15？是否与 Matmul 高阶 API 混用？
- [ ] **SYNC-08**: SetFlag 前是否有 return/break 绕过？
- [ ] **SYNC-09**: PipeBarrier<PIPE_ALL> 是否可收窄？连续 >3 个？
- [ ] **SYNC-10**: (loop-1)%N 是否有 loop=0 保护？
- [ ] **SYNC-11**: EnQue/DeQue 处是否冗余加 Flag？
- [ ] **SYNC-12**: SyncAll 分支是否对称？单核是否残留？
- [ ] **SYNC-13**: AtomicAdd 与 Fixpipe/DataCopy 写回之间是否有必要排序？
- [ ] **SYNC-14**: flag id、producer 输出 buffer id、consumer buffer id 是否同源？

---

## 判定原则

### 描述 WHAT 不 HOW（借鉴 FM-Agent 规约生成思路）

判定同步问题时，描述**该位置的同步保证什么**（WHAT），而非**代码怎么实现**（HOW）：

- **同步保证**：该同步点保证"流水 A 的操作完成后，流水 B 才能操作同一 buffer"
- **触发条件**：描述在何条件下会触发问题（如"当 tileIdx=0 时 return 跳过 SetFlag → 首轮死等"）
- **最小复现场景**：对高置信度问题，给出触发问题的最小条件组合

### 证据驱动

每个 CONFIRMED 判定必须有数据流证据：
- **跨流水依赖**：明确指出"指令 A(PIPE_X) → 指令 B(PIPE_Y)，通过 tensor Z"
- **依赖类型**：WAR（写后读）/ RAW（读后写）/ WAW（写后写）
- **同步覆盖**：明确指出同步点是否在 A 之后、B 之前，方向是否匹配
- 禁止推测未读取的代码行为
