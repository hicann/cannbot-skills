# 信号同步机制全景

> Ascend C 算子同步机制原理速查。本文件是条例判定的依据。
>
> **信息来源**：CANN 官方文档、cannbot-skills 既有可信文档（`ascendc-api-best-practices/references/api-pipeline.md`、`ascendc-code-review/references/ascendc-api.md` API-12、`ascendc-code-review/references/mc2-specific.md` MC2-01/02/03、`catlass-op-develop/references/patterns/grouped-matmul.md`、`ascendc-tiling-design/references/matmul/matmul-fusion-design-template.md`）。涉及具体 API 参数/版本差异时，以 `ascendc-docs-search` 获取的最新官方文档为准。

---

## 1. 为什么需要同步

Ascend C 的数据搬运（MTE2/MTE3）是**异步 DMA**，Cube（M）与 Vector（V）计算也是异步并行流水线。**直接在搬运/计算后的数据上做下一步操作，可能读到未完成或半成品数据，且编译期完全不报错。**

```
GM ──MTE2(异步)──> UB ──Vector(同步)──> UB ──MTE3(异步)──> GM
                                  ↑
                          Cube(M): GM→L1→L0→MMAD→L0C→(Fixpipe)→UB
```

同步的本质：**为跨流水线 / 跨核 的数据依赖插入显式等待点。**

> **例外：同流水内的原子操作语义顺序**。`SetAtomicAdd` 改变了 FIX 流水的写语义（覆盖→累加）。即使同流水内，atomic 操作也是乱序的，需要 `PipeBarrier<PIPE_FIX>` 保证语义顺序。这不是传统跨流水同步，而是**语义顺序保证**——见 SYNC-13。

---

## 2. 六类同步机制

| 机制 | 粒度 | 跨核? | 典型 API | 适用 |
|------|------|-------|---------|------|
| (1) TPipe 队列 | buffer 级 | 否 | `EnQue` / `DeQue` | MTE↔V 三段流水（CopyIn/Compute/CopyOut） |
| (2) 硬件事件 Flag | 流水对级 | 否 | `SetFlag<HardEvent>` / `WaitFlag<HardEvent>` | ISASI 精确同步，自定义多 PIPE 编排 |
| (3) 硬件事件 SyncFunc | 流水对级 | 否 | `SyncFunc<HardEvent>` | 精确到流水对的硬件事件同步 |
| (4) 单流水屏障 | 单 PIPE | 否 | `PipeBarrier<PIPE>` | 同 PIPE 内排序（PIPE_V 内部不自动保序） |
| (5) 跨核 Flag | 核间 | 是 | `CrossCoreSetFlag` / `CrossCoreWaitFlag` | Cube↔Vec 跨核数据依赖 |
| ⑥ 全核/标量屏障 | 全核/域 | 是 | `SyncAll` / `SetScalarBar`/`WaitScalarBar` | 全核栅栏、核类型域内栅栏 |

### (1) EnQue / DeQue（队列隐式同步，推荐）

`TQue` 的 EnQue/DeQue 自带硬件同步点：`EnQue` 标记就绪，`DeQue` 阻塞等待就绪。

```
CopyIn: AllocTensor → DataCopyPad(GM→UB) → EnQue        // 标记就绪
Compute: DeQue(阻塞等待) → Vector 计算 → EnQue            // 标记就绪
CopyOut: DeQue(阻塞等待) → DataCopyPad(UB→GM) → FreeTensor
```

> 来源：`api-pipeline.md`。**EnQue/DeQue 不仅是队列管理，更重要的是 MTE↔V 同步机制。**

### (2) SetFlag / WaitFlag<HardEvent>（ISASI 精确同步）

`SetFlag` 在源流水线上标记"前序指令已发射"，`WaitFlag` 在目标流水线上等待该标志。**必须成对、同 EVENT_ID、SetFlag 先于 WaitFlag**。

> **关键语义（来自 issue DTS2026031726387 教训）**：
> `SetFlag` **不等同于等待前序异步操作完成**。SetFlag 排在源流水线上，但异步搬运（如 MTE3 DataCopy UB→GM）可能在 SetFlag 发出时**尚未真正完成**。如果在异步搬出后、SetFlag 释放 buffer 复用前没有 `PipeBarrier<PIPE_ALL>()` 确保搬出完成，下一轮 buffer 复用可能读到未搬完的数据 → 偶发精度失败。
>
> **正确模式**：异步搬出（DataCopy/CopyOutXxx UB→GM）后、SetFlag 释放 buffer 复用前，必须加 `PipeBarrier<PIPE_ALL>()` 确保搬出完成。**不能用"同流水硬件保序"排除此问题**——SetFlag 是信号指令，不是普通流水线操作，§13 的"同流水保序"不适用于 SetFlag 与前序异步操作的完成关系。

- 自定义多 PIPE 编排（非标准三段流水）时使用
- `PipeBarrier<PIPE_V>` 只保证 V-V 排序，**不跨 PIPE**；跨 PIPE 依赖必须用对应方向的 SetFlag/WaitFlag

> 来源：`mc2-specific.md` 术语表 / MC2-02、`catlass grouped-matmul.md` §2、issue DTS2026031726387。

### (3) SyncFunc<HardEvent>

精确到流水对的硬件事件同步，如 `SyncFunc<HardEvent::S_MTE3>()` 确保计算结果写入通信缓冲区。

> 来源：`mc2-specific.md` MC2-01。

### (4) PipeBarrier<PIPE>

单流水屏障，如 `PipeBarrier<PIPE_V>()` 等待前序 V 操作完成。

- **模板参数是 ISA 顶层 pipe 枚举 `PIPE_*`**（PIPE_ALL/PIPE_V/PIPE_M/PIPE_MTE1/PIPE_MTE2/PIPE_MTE3/PIPE_S/PIPE_FIX），与 SetFlag/WaitFlag 的 `HardEvent` 枚举**不同源**——不存在 `HardEvent::PIPE_ALL`，全限定写法为 `AscendC::PipeBarrier<PIPE_ALL>()`（`PIPE_*` 不加 `AscendC::HardEvent::` 前缀），见 [fix-patterns.md「修正 diff API 签名自检」](fix-patterns.md)
- **PIPE_V 内部不保证自动保序**，依赖同一 tensor 重叠区域的 V 操作间需显式 PipeBarrier<PIPE_V>
- `PipeBarrier<PIPE_ALL>` 粒度过粗，会阻塞所有 PIPE，**热循环内慎用**（SYNC-09）；但以下是合法且推荐场景：①一次性初始化路径（简洁、不消耗 eventID）②异步搬出后、SetFlag 释放 buffer 复用前（见本节 (2) 的 DTS 教训）③多核通信数据区/状态区写入之间（§11）

> 来源：`mc2-specific.md` 术语表、`ascendc-perf.md` PERF-4。

### (5) CrossCoreSetFlag / CrossCoreWaitFlag（核间同步）

多核算子中 Cube 核与 Vec 核通过此机制同步。**必须对称：每个 WaitFlag 有且仅有一个 SetFlag，所有能到达 WaitFlag 的路径都必须触发 SetFlag。**

```
AIC 末尾: CrossCoreSetFlag(AIC_TO_AIV)  →  AIV 开头: CrossCoreWaitFlag(AIC_TO_AIV)
AIV 末尾: CrossCoreSetFlag(AIV_TO_AIC)  →  AIC 开头: CrossCoreWaitFlag(AIV_TO_AIC)
```

> 来源：`ascendc-api.md` API-12、`matmul-fusion-design-template.md` §2.2/2.4。
>
> **官方约束（8.5.0）**：
> - `CrossCoreSetFlag` 内部已被 Matmul 高阶 API 使用，**禁止二者混用**，否则 flagId 冲突
> - **同一 flagId 计数器最多设置 15 次**
> - Flag 常量应复用参考工程 `cv_sync_constants.h`，禁止自定义 Flag ID

### ⑥ SyncAll / SetScalarBar / WaitScalarBar

- `SyncAll`：全核全局同步屏障，所有核执行到同一 SyncAll 点后才能继续。`SyncAll<false>` 同步所有核含 AIC。开销最大。
- `SetScalarBar`/`WaitScalarBar`：标量屏障，核类型域内栅栏（如 AIV 域内跨 lane 合并）。

> 来源：`mc2-specific.md` 术语表、`catlass grouped-matmul.md` §7.4。

---

## 3. HardEvent 方向对照表

HardEvent 标识同步的流水线对，**必须与实际数据流方向匹配**。

> 下方「常用方向」来自 cannbot-skills 文档；「完整方向表（31 种）」来自 CANN 官方文档 + ops-nn/ops-tensor/ops-blas 真实代码全集，随 CANN 版本演进需用 `ascendc-docs-search` 核实补充。

### 常用方向（高频）

| HardEvent | 方向 | 典型场景 |
|-----------|------|---------|
| `MTE2_V` | MTE2 搬入 → Vector | GM→UB 搬完后做矢量运算 |
| `V_MTE3` | Vector → MTE3 存出 | 矢量算完后存进 GM/workspace |
| `S_MTE3` | Scalar → MTE3 | 标量设置后触发 MTE3 搬运 |
| `MTE3_MTE2` | MTE3 → MTE2 | 双 buffer 轮转，存完才能覆盖（下一轮搬入） |
| `M_V` | Cube → Vector | Cube 结果供 Vector 后处理 |
| `V_V` | Vector → Vector | Vector 内部排序 |
| `MTE1_M` | MTE1 搬运完成 → Cube | L1→L0 加载完成后做 MMAD |
| `V_M` | Vector → Cube | Vector 结果供 Cube |
| `V_MTE2` | Vector → MTE2 | V 写 UB 后 MTE2 读（双 buffer 轮转） |
| `MTE3_V` | MTE3 → Vector | MTE3 存出后 Vector 读回 |

### 完整合法方向表（31 种）

> 来源：CANN 官方文档 + ops-nn/ops-tensor/ops-blas 真实代码全集。

```
MTE2_MTE1  MTE1_MTE2  MTE1_M     M_MTE1
MTE2_V     V_MTE2     MTE3_V     V_MTE3
M_V        V_M        V_V
MTE3_MTE1  MTE1_MTE3  MTE1_V
MTE2_M     M_MTE2     V_MTE1
M_FIX      FIX_M
MTE3_MTE2  MTE2_MTE3
S_V        V_S
S_MTE2     MTE2_S
S_MTE3     MTE3_S
MTE2_FIX   FIX_MTE2
FIX_S      M_S        FIX_MTE3
```

> `FIX` = Fixpipe。脚本 `KNOWN_HARDEVENTS` 含此全集，未知方向（如 `MTE2_V` 写成 `MET2_V`）会触发 SYNC-05。

> 来源：`mc2-specific.md` 术语表、`catlass grouped-matmul.md` §2、ops-nn/ops-tensor/ops-blas 真实代码、CANN 官方文档。

### 跨 PIPE 同步事件对照（自定义多 PIPE 编排）

| 数据依赖 | 需要的同步 | 典型场景 |
|---------|-----------|---------|
| V 算完 → MTE3 存出 | `SetFlag/WaitFlag<PIPE_V, PIPE_MTE3>` | SwiGLU/激活算完存进 workspace |
| MTE2 搬入 → V 计算 | `SetFlag/WaitFlag<PIPE_MTE2, PIPE_V>` | GM 搬入后做矢量运算 |
| MTE3 存出 → MTE2 复用同 buffer | `SetFlag/WaitFlag<PIPE_MTE3, PIPE_MTE2>` | 双 buffer 轮转 |
| V-V 之间 | `PipeBarrier<PIPE_V>` | 仅同 PIPE 排序，**不能**替代上面任一条 |

> 来源：`catlass grouped-matmul.md` §2。

---

## 4. 数据流与同步插入点（检验时逐点核对）

### 4.1 标准 Vector 三段流水（EnQue/DeQue 即可）

```
GM --MTE2--> UB --V--> UB --MTE3--> GM
     [EnQue/DeQue 自动同步，无需手动 Flag]
```

### 4.2 自定义多 PIPE（必须手动 Flag）

每个跨 PIPE 数据依赖点都要插入对应方向 Flag：

```
GM --MTE2--> UB:  MTE2→V  SetFlag/WaitFlag
UB --V--> UB:     PipeBarrier<PIPE_V>（同 PIPE 内）
UB --V--> GM:     V→MTE3  SetFlag/WaitFlag
GM(workspace) --MTE2--> UB（下一轮复用）:  MTE3→MTE2  SetFlag/WaitFlag（保证上一轮已存完）
```

### 4.3 Cube 侧 L0C 数据流

```
GM --MTE2/MTE1--> L1 --MTE1--> L0A/L0B --MMAD(M)--> L0C --Fixpipe--> UB
```

- **L0C 变化**：MMAD 累加写 L0C，Fixpipe 读 L0C→UB。L0C buffer 复用/轮转时，须确保上一轮 MMAD 完成、Fixpipe 读出后才可覆盖。
- **L0A/L0B 重载**：下一轮 MMAD 前，MTE1（L1→L0）加载须完成 → `MTE1_M` 同步。
- 多 stage 共享 L1/L0 buffer 轮转时，常量与索引一致性须保证。

> 来源：`matmul-fusion-design-template.md` §2.3/2.4、`api-buffer.md` 多 stage 共享 L1/L0、`api-matmul.md`（L0C 由 fixpipe 写回）。
>
> ⚠️ L0C/Fixpipe 同步细节随平台/版本演进，执行前用 `ascendc-docs-search` 核实当前平台的 MTE1_M / Cube 内部保序规则。

### 4.4 Cube↔Vec 跨核数据流

```
AIC: GM→L1→L0→MMAD→L0C→Fixpipe→UB ──CrossCoreSetFlag(AIC→AIV)──> AIV
AIV: CrossCoreWaitFlag(AIC→AIV) → GM→UB → Eltwise → UB→GM ──CrossCoreSetFlag(AIV→AIC)──> AIC
AIC: CrossCoreWaitFlag(AIV→AIC) → 进入下一 tile
```

> 来源：`matmul-fusion-design-template.md` §2.4 核心计算步骤。

---

## 5. 同步机制选择决策

```
是标准三段流水（CopyIn/Compute/CopyOut）？
  └─ 是 → EnQue/DeQue（机制(1)），无需手动 Flag
  └─ 否，自定义多 PIPE 编排
      └─ 跨 PIPE 数据依赖？
          └─ 是 → SetFlag/WaitFlag<对应方向>（机制(2)）
          └─ 同 PIPE 内 V-V 排序 → PipeBarrier<PIPE_V>（机制(4)）
是跨核（Cube↔Vec）数据依赖？
  └─ 是 → CrossCoreSetFlag/WaitFlag（机制(5)），且不能与 Matmul 高阶 API 混用
  └─ 否
需要全核栅栏？
  └─ 是 → SyncAll（机制⑥），注意 <false> 语义与所有核可达
```

---

## 6. 常见误区

| 误区 | 正确理解 | 来源 |
|------|---------|------|
| AllocTensor 后数据就可用 | AllocTensor 只分配内存，不等待搬运 | api-pipeline.md |
| DataCopy 是同步的 | DataCopy 是异步 DMA，立即返回 | api-pipeline.md |
| 不用 EnQue/DeQue 也能正常工作 | 必须用 EnQue/DeQue 或 PipeBarrier/Flag 同步 | api-pipeline.md |
| PipeBarrier<PIPE_V> 能跨 PIPE | 仅同 PIPE 排序，不跨 PIPE | catlass grouped-matmul.md |
| 同一处加多种同步更安全 | EnQue/DeQue 已同步处再加 Flag 是冗余，损性能 | SYNC-11 |
| 单核也能用 CrossCoreWaitFlag | 单核无人 SetFlag → WaitFlag 永久阻塞 | SYNC-03 |
| 编译通过就同步没问题 | 同步缺失/错序编译期完全不报错 | catlass grouped-matmul.md |

---

## 7. AIC 与 AIV 的流水差异（关键）

| 侧别 | 可用流水 | 同步规则 |
|------|---------|---------|
| **AIC（Cube 核）** | S, M, MTE1, MTE2, MTE3, FIX | **框架自动同步**，算子开发者不需手动插入同步。排查时 AIC 分支不产生缺失告警 |
| **AIV（Vector 核）** | S, V, MTE2, MTE3 | V 内部不保证保序（同 tensor 重叠区域需 PipeBarrier<PIPE_V>）；MTE2/MTE3/V/S 间需开发者插入同步 |

> Kernel 类型判定：`KERNEL_TYPE_AIC_ONLY`（纯 Cube）、`KERNEL_TYPE_AIV_ONLY`（纯 Vector）、`KERNEL_TYPE_MIX_*`（混合）。

---

## 8. API → 流水归属映射

排查时需将每条指令映射到其所属流水，才能判断跨流水依赖。

### 数据搬运

| API | 方向 | 流水 |
|-----|------|------|
| `DataCopy(dst_local, src_gm)` / `DataCopyPad` | GM→UB | MTE2 |
| `DataCopy(dst_gm, src_local)` / `DataCopyPad` | UB→GM | MTE3 |
| `LoadData(...)` | L1→L0A/L0B | MTE1 |
| `Fixpipe(...)` | L0C→GM/L1 | FIX |
| `Te::Copy(CopyGM2L1{})` | GM→L1 | MTE2 |
| `Te::Copy(CopyL12L0A/B{})` | L1→L0 | MTE1 |
| `Te::Copy(CopyL0C2GM{})` | L0C→GM | FIX |

### Vector 计算（PIPE_V）

`Add/Sub/Mul/Div`、`Adds/Muls/Subs/Divs`、`Exp/Sqrt/Ln/Rsqrt`、`Abs/Reciprocal`、`Relu/LeakyRelu`、`Cast`、`Duplicate`、`Compare/Compares`、`Select`、`Gather/GatherMask`、`ReduceMax/Min/Sum`、`WholeReduceMax/Min/Sum`、`Transpose/TransDataTo5HD`、`Brcb`、`Axpy`、`FusedMulAdd`、`AddReluCast`、`CastDequant` 等

### Cube 计算（PIPE_M）

`Mmad()`、`MatMul`（高阶 API）

### Scalar 操作（PIPE_S）

`LocalTensor::SetValue/GetValue`、`GlobalTensor::SetValue/GetValue`、`tensor(idx)` 标量寻址

---

## 9. 跨函数依赖追踪（Stage 3 人工必查）

**函数末尾的异步操作（如 MTE3 DataCopy）在函数返回后可能仍未完成。**

排查规则：
- 追踪每个函数出口处各流水的"最后一条未被同步覆盖的操作"
- 若函数末尾 MTE3/MTE2 后无 barrier/sync，调用者或后续函数中有不同流水操作同一内存 → 缺失同步
- **FreeTensor 不等待异步操作完成**——仅释放 UB 管理权
- **特别注意 Init() 与 Process() 的跨函数依赖**

---

## 10. 循环回边依赖（Stage 3 人工必查）

对每个 for/while 循环体，除单次迭代内依赖外，还必须分析**跨迭代依赖**：
- 循环体末尾最后一条 tensor 读/写 → 循环体开头第一条 tensor 读/写
- 若操作同一 tensor：末尾读→开头写 = WAR；末尾写→开头读 = RAW
- 均需对应方向的同步

---

## 11. 多核通信语义顺序同步

多核 Window/状态通信中，**数据写入（MTE3 写 GM 数据区）和状态信号写入（Scalar/MTE3 写 GM 状态区）之间必须有 `PipeBarrier<PIPE_ALL>()`**，即使操作不同 GM 地址。这是通信协议语义要求：对端核看到状态信号时，数据必须已可见。

排查规则：检测"写 GM 数据区"后紧接"写 GM 状态/标志位"的模式，两者间必须有全流水屏障。

---

## 12. 过度同步判定规则

| 现象 | 判定 | 优化建议 |
|------|------|---------|
| `PipeBarrier<PIPE_ALL>` 但只涉及 2 个流水 | 过度 | 替换为 `TQueSync<src,dst>` 或 `SetFlag/WaitFlag<HardEvent>` |
| 连续两个 PipeBarrier 间无需保护的指令 | 冗余 | 删除前一个 |
| EnQue/DeQue 已覆盖又加 SetFlag/WaitFlag | 冗余 | 删除手动同步 |
| SyncAll 但只有单核 | 无意义 | 删除 |
| 循环每次都 PIPE_ALL 但只需 Double Buffer 交替 | 过度 | 改用 MTE3_MTE2 SetFlag/WaitFlag 交替 |
| SyncAll 但仅两核间有依赖 | 过度 | 替换为 CrossCoreSetFlag/WaitFlag |

### 同步精度优先级（从优到劣）

1. TPipe EnQue/DeQue（Stage 级自动）
2. TQueSync<src,dst>（精确流水对）
3. SetFlag/WaitFlag<HardEvent>（精确流水对）
4. PipeBarrier<specificPipe>（单流水全阻塞）
5. PipeBarrier<PIPE_ALL>（全流水停顿）
6. SyncAll（全核同步）

---

## 13. 同步判定决策树（仅供参考，不用于否决脚本候选）

> **重要**：此决策树仅供参考理解同步原理。脚本输出的候选不得以此决策树为理由否决为误报。实测 LLM 用此决策树推理会得出错误结论（如用"同流水保序"排除 SetFlag 缺 PipeBarrier 问题）。

```
1. A 和 B 操作同一 tensor/内存区域？
   ├─ 否 → 检查是否为多核通信语义依赖（§11：数据区写入 + 状态区写入）
   │        ├─ 是 → 两者间需 PipeBarrier<PIPE_ALL>
   │        └─ 否 → 无依赖
   ├─ 同一 tensor 不同偏移（可静态证明不重叠）→ 无依赖
   └─ 是（或无法证明不重叠）→
       2. A 和 B 属于同一流水？
          ├─ 是 → 同流水内硬件保序
          │        例外：PIPE_V 内同 tensor 重叠区域需 PipeBarrier<PIPE_V>
          │        注意：SetFlag 是信号指令，不适用"同流水保序"——见 §2
          └─ 否 →
              3. 是否在 AIC 分支？
                 ├─ 是 → 框架自动同步
                 └─ 否 →
                     4. A→B 之间是否有有效同步？
                        ├─ 有 → 同步精度是否最优？（§12）
                        └─ 无 → 是否有间接同步？（EnQue/DeQue）
                                   ├─ 是 → 已覆盖
                                   └─ 否 → 缺失同步
```

> **assert 不是终止点**：assert 在 release 模式下编译为空操作，后续代码仍可达。
