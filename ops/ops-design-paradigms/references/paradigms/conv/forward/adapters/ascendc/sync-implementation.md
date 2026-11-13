> **层级**：AscendC 语言适配层（adapters/ascendc/）。同步原则（谁等谁、为什么等、防死锁）是语言中立的，见 [../../generic/compute-flow.md](../../generic/compute-flow.md) §7；本文件只收录**事件对写法与队列原语的落地细节**。参考实现见 [example/conv2d/](example/conv2d/)。

# 同步落地

## 1 事件法适用面

```
同步层级               是否需要手写事件        原因
┌──────────────────┬─────────────────────┬──────────────────────────────┐
│ L1 级(GM→L1→消费)│  不需要              │ TQue 自动插入（Alloc/EnQue/DeQue/Free）│
│ L0 级(L1→L0→Mmad) │  **需要手写**        │ TBuf 无队列管理，手动 WaitFlag/SetFlag  │
│ L0C 级(Mmad→FIX)  │  不需要显式事件      │ TQue(CO1) 自动串接 + unitFlag 机制     │
│ BT 装载(L1→BT→Mmad)│  不需要              │ TQue(C2) 自动管理                      │
└──────────────────┴─────────────────────┴──────────────────────────────┘
```

## 2 L0 级事件对写法（单缓冲串行正确版）

每步 K 归约内**成对出现**（1 对 Wait + 1 对 Set）：

```cpp
const event_t ev = static_cast<event_t>(0);
// ── 每步 K 归约（kIter）内 ──
WaitFlag<HardEvent::M_MTE1>(ev);   // ① WAR：等上一个 Mmad 读完 L0A
LoadData(A2);  LoadData(B2);      // ② MTE1：装载 L0
SetFlag<HardEvent::MTE1_M>(ev);   // ③ RAW：装载完成
WaitFlag<HardEvent::MTE1_M>(ev);  // ④ RAW：等装载
Mmad(...);                         // ⑤ M：矩阵乘累加
SetFlag<HardEvent::M_MTE1>(ev);   // ⑥ WAR：许可下一次覆盖 L0A
```

时序图（单缓冲）：

```
MTE1:    │ 装L0(0) │    装L0(1) │    装L0(2) │ ...
M   :            │ MMAD(0) │ MMAD(1) │ MMAD(2) │ ...
          └── 每步串行等待（无重叠）──┘
```

> ⚠️ **首轮事件初始态须核实**：上写法假设进入循环前 `M_MTE1` 事件通道已处于"已置位"状态（首个 `WaitFlag` 即刻通过）。若所用环境的事件通道初态为未置位，须在循环外先 `SetFlag<HardEvent::M_MTE1>(ev)` 一次（prime）再进循环，否则首轮即死锁——移植前按所用头文件/硬件手册核实，并在产出中声明该假设。

## 3 双缓冲（乒乓）下的 L0 级事件形态

每份缓冲一套**独立**事件（ping/pong 各自的 RAW 与 WAR），两套交替即流水：

```cpp
const event_t evPing = static_cast<event_t>(0);
const event_t evPong = static_cast<event_t>(1);
uint32_t pingPong = 0;
for (kIter...) {
    event_t ev = pingPong ? evPong : evPing;
    WaitFlag<HardEvent::M_MTE1>(ev);        // 等上一轮同份缓冲的 Mmad 读完（WAR，按份独立）
    LoadData(A2[pingPong * halfSize]);      // 写本份
    LoadData(B2[pingPong * halfSize]);
    SetFlag<HardEvent::MTE1_M>(ev);
    WaitFlag<HardEvent::MTE1_M>(ev);
    Mmad(..., cl0[pingPong * halfCl0]);     // unitFlag：中间步=2，末步=3（翻转）
    SetFlag<HardEvent::M_MTE1>(ev);
    pingPong ^= 1;                           // 下一轮换份
}
```

时序图（双缓冲）：

```
MTE1:  │装L0(0)│ 装L0(1) │ 装L0(2) │ 装L0(3) │ ...    ← 装载连续流动
M   :        │  MMAD(0)  │  MMAD(1)  │  MMAD(2) │ ...   ← 计算连续流动
             └── 装载与计算重叠（流水）──┘
```

双缓冲关键点：

| # | 要点 | 说明 |
|---|------|------|
| 1 | 事件按"份"隔离 | 写 pong 只需等 pong 的 WAR，不等 ping 的 |
| 2 | 地址按份划分 | L0A / L0B / L0C 均按双份划分（TBuf 整块分配后手工切半，或 InitBuffer 份数=2） |
| 3 | TQue 管理的 buffer | 把 InitBuffer 份数设 2 即自动完成上述事件管理，无需手写 |
| 4 | 落地前复核 | 按 generic [compute-flow.md](../../generic/compute-flow.md) §7.3 逐路径复核事件平衡（尾块/早退路径的 Set/Wait 次数必须与正常块一致） |

## 4 Mmad → Fixpipe 的 unitFlag 机制

| 场景 | Mmad unitFlag | Fixpipe unitFlag | 含义 |
|------|--------------|-----------------|------|
| K 归约中间步 | 2 | — | L0C 被 Fixpipe 消费前保持占用 |
| K 归约**末步** | **3** | — | 置位 + 翻转（占用切换到另一份 L0C） |
| Fixpipe | — | 3 | 消费后释放 |

⚠️ **末步误留在 2 的后果未定义**——硬件是否仍正确触发 Fixpipe 消费无文档保证。高危触发面：尾块跳步（内层循环因余量为 0 跳过末尾步），须确保 `isLast` 判定与内层循环界使用同一"逐层取余"口径（见 [compute-flow.md](../../generic/compute-flow.md) §6.6）。

## 5 TQue 模板第二参数与 InitBuffer 份数

以所用头文件定义为准，使用前核实：

| 参数 | 含义 | 取值 |
|------|------|------|
| `TQue<pos, N>` 的 N | 队列深度声明 | ≥ 实际事件并发数（参考实现 CO1 取 2） |
| `InitBuffer(que, N, size)` 的 N | **缓冲份数** | 1 = 单缓冲；2 = 双缓冲 |

参考实现中 CO1 队列声明深度 2、实际份数 1（单缓冲）；开启双缓冲时改 InitBuffer 的份数即可，模板参数语义按头文件定义核实。
