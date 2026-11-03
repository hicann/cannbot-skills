# ScatterUpdate 确定路径：三段式段尾覆盖（条件路径）

> Phase2 稳定排序契约 / Phase3 段尾覆盖（last-write-wins）的 (b,k,s) 并行分解 / 确定性保证机制 / golden 精确写序 / 硬件特征与已否决方向。设计文档 §4.2 det 路径的真值源。assign 型 det 为**条件路径**（张量源：重复度高或确定性请求；标量源值同：仅重复度高——性能）——默认路径为 nondet 直写（[type-assign](type-assign.md) §2）。
>
> **Phase1 × Phase3 正交**：Phase1 决定冲突域（key 合成，见 [conflict-domain.md](../../generic/conflict-domain.md)），Phase3 决定并行展开——两者独立组合，Phase3 **不感知** nd / index 覆盖单维 / index 覆盖全部维等 index 形态。

---

## 1 核心范式

覆盖写的默认路径 nondet 直写下，多写者任意赢即覆盖语义正确——排序在本分型**非正确性必需**。三段式在以下条件触发（仅性能或确定性动机）：

```
Phase 1 冲突 key:   任意 index 形态 → 1 维冲突 key（冲突域标识，不含 b/s）
Phase 2 稳定排序:   升序稳定排序 key → sortedKey + perm（perm 保留原始散点坐标）
Phase 3 段尾覆盖:   同 key 连续成段；(b,k,s) 全量任务段尾自判（O(1) 判据：与下一索引
                    不同或全局末位），段尾任务写自己
```

**排序的作用（性能去重 +（张量源）确定性）**：

| 动机 | 说明 |
|------|------|
| **性能** | 去重非正确性必需（多写者任意赢即覆盖语义正确，[type-assign](type-assign.md) §1.1）——高重复时排序段尾单写者规避 **Ascend 同地址并发写慢一个数量级**（§6 硬件特征），把 M 次冲突写降为 unique(key) 次 |
| **确定性（张量源）** | last-write-wins 固定顺序——稳定排序保证段内原始顺序唯一，段尾元素（原始行号最大）为胜者，与线程调度/核数无关 |
| 标量源（值同） | 无确定性问题（任意赢 = 同值 = 天然 bit 确定），仅高重复走排序（纯性能） |

## 2 Phase 2：稳定排序（SortLib 功能契约）

- 契约：升序**稳定**排序 → `sortedKey + perm`，`sortedKey[i] == key[perm[i]]`。**稳定是段内写回顺序确定性（last-write-wins）的来源**。
- 排序对象 = **1 维冲突 key**（排序量 = 冲突单元数 = 索引个数，**不含 b、不含 s**——把 batch/slice 混进排序 key 是已否决的冗余，排序量膨胀 b 倍）。
- Host：`SortLib::SortTilingCompute`（见 [tiling-preprocess](tiling-preprocess.md) §3）。
- Kernel：`SortInvoke` 前后 `pipe.Reset()`，排序后 `SyncAll()`；要求 `SetScheduleMode(1)`。
- 排序仅占 ~2%（74us@185K 直测，§6）——非优化目标，已否决排序优化方向（§7 墓碑）。

## 3 Phase 3：段尾覆盖——三场景并行分解（与 Phase1 正交）

### 3.1 正交声明与统一框架

Phase3 **不感知 nd / index 覆盖单维 / index 覆盖全部维**——它只消费三个数值与两个数组：

- `b`（batch）：index 不覆盖的**前缀维**合轴（独立重复维）
- `k`（排序后索引位置）：Phase2 输出的 sortedKey 下标，同 key 连续成段（冲突段）
- `s`（slice）：index 不覆盖的**后缀维**合轴（key 对应的连续负载块）
- `sortedKey[k]`（冲突 key）+ `perm[k]`（反查原始散点坐标 → source 寻址）

各形态到 (b, k, s) 的映射（Phase1 侧完成，Phase3 透明）：

| 形态 | b | k | s | 走哪个场景 |
|------|---|---|---|-----------|
| index 覆盖全部维 | 1 | N | 1 | (k)（(k,s) 的 s=1 特化） |
| index 覆盖单维，dim 轴中 | prod(shape[:dim]) | M | prod(shape[dim+1:]) | **(b,k,s) 通用** |
| index 覆盖单维，dim 首维 | 1 | M | prod(shape[dim+1:]) | (k,s) |
| index 覆盖单维，dim 末维 | prod(shape[:dim]) | M | 1 | (b,k) |
| nd | 1 | N | prod(shape[K:]) | (k,s) |

### 3.2 三场景并行分解（后两个是第一个某维=1 的特化）

| 场景 | 条件 | 快除次数 | 任务总量 |
|------|------|---------|---------|
| **(b, k, s)** 通用三维 | b>1 ∧ s>1 | 2（÷s、÷k） | b × k × s |
| (k, s) 二维 | b==1 | 1（÷s） | k × s |
| (b, k) 二维 | s==1 | 1（÷k） | b × k |

> **退化交叠**：`b == 1 && s == 1` 时任务空间退化为 `(k)`，任务量 `kNum`，零快除。

```cpp
// 通用 (b,k,s)：task = b·(kNum·sNum) + k·sNum + s，反解两次快除（(k,s)/(b,k) 为省一除特化）
// 段尾判据全局（k==kNum-1 || sortedKey[k+1] != sortedKey[k]）；
// 每个任务 O(1) 自判段尾，段尾任务写自己（无扫描、无跨线程协调）
q = UintDiv(task, sMagic, sShift);        // ÷sNum
s = task - q * sNum;
b = UintDiv(q, kMagic, kShift);           // ÷kNum
k = q - b * kNum;
KeyT key = sortedKey[k];
if (key == SENTINEL_KEY) { continue; }                     // 哨兵段前置跳过（bounds §4）
if (k < kNum - 1 && sortedKey[k+1] == key) { continue; }   // 非段尾（与下一索引同段）
// var 侧 = key + s（key=lin 已是 var 空间元素偏移，slice 连续占 [lin, lin+sNum)；
// = ASC varVec[key/W + sw] 的 W=1 形态）；upd 侧平铺 = pk·sNum + s
varBytes[key + s] = updBytes[perm[k] * sNum + s];   // 段尾任务直写单元素（W=1 兜底），bit-exact
```

**要点**：

- **并行度 = b × k × s**（越大越好）——s 维线程级展开（每输出元素一任务），flat 空间布局使相邻线程拿相邻 s → coalesced 访存。
- **段尾单写者（O(1) 判据，无扫描）**：写条件 = `k == kNum-1 || sortedKey[k+1] != key`（与下一索引不同或全局末位）；每个 (冲突段 × b × s) 恰一个写者——写者是**段尾 k 的任务**，b/s 是独立并行维各任务自带，不参与段判定；段内其余任务判据空转（判据在 s 维冗余执行，O(1) 幂等无害）。对照：归约写范式的段首单写者**串行归约**是归约顺序的语义必需，本分型无此依赖，无需扫描循环。
- **字节宽打包性能路径（元素级 W=1 为通用兜底）**：sliceSize>1 且能被 w 整除时**从大到小择最宽 w**（`ComputePackParam`：从 `8/sizeof(T)` 折半向下试探），受 `W·sizeof(T) ≤ 8 字节`上限——1B dtype 最高 W=8、4B 只能 W=2、**8B dtype 恒元素级**；`sliceVec = sliceSize / W`，var/updates 以 PackT（uint8/16/32/64）视图重解释，每任务一条指令读写 W 元素：`varVec[key/W + sw] = updVec[perm[k]·sliceVec + sw]`（key 为 sliceSize 整数倍、W 为 2 的幂 → key/W 精确）；flat 空间 [N, sliceVec] 相邻线程拿相邻 sw，打包元素天然相邻访存。det/nondet 两路径共用同一 `ComputePackParam`（单一事实源，防 W 计算漂移）。
- **sliceSize==1 快速路径**：s 维消失，任务空间退化为 `b × k`。`b==1`（index 覆盖全部维，或 nd 的 `sliceSize==1`）时进一步退化为 `[N]`（(k) 场景，零快除）；`b>1`（index 覆盖单维·末轴）时走 `(b,k)`，只 `÷k`。段尾判据同款、直写单元素。
- **uint32 坐标计算域降级（det/nondet 统一门控）**：Phase1 反解/线性化与 Phase3 三场景分解的整条坐标计算链统一挂 `IndexT` 模板参数（`useInt32Index` 下发，五项计数门控见 [linearize](linearize.md) §6）；读索引与越界检查保持 IdxT 原值域不参与降级。
- 退化场景是**特化快路径**（省一次快除），不是独立结构：`b==1 && s>1` → `(k,s)` 只 `÷s`；`s==1 && b>1` → `(b,k)` 只 `÷k`；`b==1 && s==1` → `(k)` 零快除（source/目标寻址同步去掉 sNum 乘法）。
- **哨兵段整体跳过**：skip 越界形态下哨兵行沉底成段，`key == SENTINEL_KEY` 前置早退整段不写入、不寻址（[bounds](../../generic/bounds.md) §4；漏判会以 `SENTINEL_KEY + s` 巨值偏移越界写）。
- **纯搬运无算术**：NaN/Inf/±0 原样透传，binary 比对（PackT 字节搬移与读收集范式强制规则同源）。
- 覆盖顺序（最后赢 / 第一个赢）是语义决策点——第一个赢判据反转为段首向（`k==0 || sortedKey[k-1] != key`，与上一索引不同或全局首位），取段首 `perm[k]`，机制同构。
- 稳定排序保证段内原始顺序唯一 → **段尾元素（原始行号最大）= last-write-wins 胜者**，与线程调度/核数无关。

## 4 确定性保证机制（逐一守住）

| 机制 | 作用 |
|------|------|
| 稳定排序 | 同 key 保持原始顺序 → 段尾元素唯一确定（last-write-wins） |
| 段尾判据全局化（k 维） | 与核/线程划分无关 → 分组不漂移 |
| 段尾 O(1) 自判（与下一索引不同或全局末位） | 写者为段内原始行号最大，写回输入序列唯一，与线程调度/核数无关 |
| 纯字节搬运 | 无算术无舍入，bit-exact，binary 比对 |
| 哨兵沉底 | 越界行不写入且不污染有效段（bounds §4） |

## 5 golden 精确写序复现（验收标杆）

- **det golden 必须复现段尾写序**：顺序遍历 + 后写覆盖（last-write-wins，或按声明的前写胜出——第一个赢取段首），纯搬运 binary 比对 (0,0) bit 级。cand 候选集合由 golden 顺序执行派生——det 全等验证候选集合自身的正确性（互证）。
- 越界同步：error 形态 golden raise ⟺ kernel `__trap`；skip 形态 golden 跳过。
- **nondet 路径分层判据**（胜者任意 → **禁止对多样值用例做全等比对**，按竞争结构分层）：

| 层 | 用例类型 | 判据 |
|---|---|---|
| 1 | 无竞争（每 target 唯一贡献） | bit 全等 == golden（唯一写者，值确定） |
| 2 | 同值贡献（同 target 所有 update 同值） | bit 全等（任意赢 = 同值） |
| 3 | 多样值重复 | **元素级候选集合成员判定**：`out[e] ∈ cand[e]`，cand[e] = {所有指向 e 的 update 值} |

- **层 3 实现与纪律**：
  - **成员判定（非全等）**：对每个被重复写的 target 断言 `∃m: out[e] == upd_m[e]`——按 update 行构造 match 掩码、按 target OR 聚合，无需物化候选 set（向量化友好，O(重复度) 次比较零额外内存）。
  - **判定粒度 = 元素级**：nondet 重复 target 的胜者粒度是 **pack**（对齐 store 总线事务，≤8B 完整 bit 模式）——同 slice 的不同 pack 可来自不同 update 行（slice 级"整片同源"不成立）；pack 内 W 元素必然同源 → 元素级成员判定是唯一正确粒度。
  - **bit 级零容差**：纯搬运语义，任何容差都会掩盖撕裂写。
  - **错误模式覆盖**：撕裂写（pack 内新旧 bit 混合 ∉ cand——对齐 store 原子性本应防住，属防御性验证）；写丢失（残值 var[e]——**var[e] ∈ cand 时漏检**，靠值域隔离规避）；写错位置（邻近元素 ∉ cand 或专属断言）。
  - **值域隔离纪律**：var 初始化值域与 update 值域隔离（如 var 用固定模式 0x55 填充、update 取独立值域）→ 写丢失必然可检。
  - **禁止断言（防 flaky）**：层 3 用例**不得**断言跨次一致（`--run` 多轮 bit 恒定）、**不得**断言 nondet == det（跨模式一致）——胜者随线程调度漂移，仅层 1/2 可作此断言。**assign 的非确定性与 dtype 无关，整数不豁免**（对照 reduce 分型"整数可交换 → nondet 跨模式 bit 一致"仅适用于归约路径，不可迁移本范式）。

## 6 硬件特征实测库（950PR，跨算子复用）

| 特征 | 实测值 | 影响 |
|------|--------|------|
| **同地址并发/冲突写（非原子）** | **≈ 慢一个数量级**（相对正常分散写） | assign 高重复走排序的核心动机（§1：nondet 直写低重复最快、高重复被此特征拖慢——排序段尾单写者把 M 次冲突写降为 unique(key) 次） |
| 排序 | 74us@185K | 占总耗时 ~2%，非优化目标 |
| rank 编译期特化 | uint64 域寄存器溢出（rank≥5 退化 87%）；uint32 域净收益 1.04x + 个别稳定退化 | **两域均否决**；解码非瓶颈（内存主导） |

## 7 已否决方向墓碑（勿重启，均实测 + 固化）

| 方向 | 否决证据 | 重启条件 |
|------|---------|---------|
| 段表/段信息预构建（Phase2.5） | 4 次尝试 4 败（含 SIMD GatherMask 单核版 hang） | 无——在线段判据已足够 |
| 编译期 rank 特化（det/nondet） | uint64 域寄存器溢出（rank3/4 退化）；uint32 域净 1.04x 且个别稳定退化 | 解码被证明非瓶颈，无重启动机 |
| 排序优化（提核数/计数排序） | 排序仅占 ~2% | 排序成为瓶颈时 |
| kernel 末尾 V_S join | 无后续 S 消费，task 结束协议排空 | —（已移除，属正确化而非否决） |
