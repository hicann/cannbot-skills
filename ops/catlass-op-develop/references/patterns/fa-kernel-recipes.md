# FA kernel 算子配方库（FFA / s1s2 / S1-sink / kfc 直调）

> **分册说明**：本册是 [fa-kernel-handcraft.md](fa-kernel-handcraft.md) §12 系列的**算子配方部分**（原手册拆分，§ 编号不变）：
> §12.5 FFA 全程实证与终态配方、§12.10 s1s2 单核融合算法剖析、§12.11 fa-sink S1 模板流程、§12.13 kfc 单 TU 直调终态配方。
> 复合特性叠加（sink/varlen/q8/causal/paged）见 [fa-kernel-compound-features.md](fa-kernel-compound-features.md)；骨架见手册 §1-11。

> **简称**：本文档称「配方册」；[fa-kernel-compound-features.md](fa-kernel-compound-features.md) 称「特性册」。

---

## 12.5 FFA 手搓实测（已验证 + 卡点）

> 实测工程（本地自写 FFA kernel，catlass BlockMmadQK + 自写任务/softmax/rescale/host）。**90% 环节已验证正确，卡在跨核整幅同步**。以下是有实据的经验，直接复用。

### 12.5.1 FFA 数学 → 任务映射（已验证正确）

FFA `S=scale·(Q·K1ᵀ+Q·K2ᵀ对角)→softmax→O=W·V1+W·V2对角` 的**两个 GEMM batch 维正交**（a0 的 B 按 n 分组、a1 的 B 按 m 分组），**同一行排布下不可能都高效**。解法（**已验证 S0/S1 max_abs=0.0000**）：

```
a0 = Q·K1ᵀ  任务(n,m0) 行=M  B=K1[n]共享   → S0   （标准 GEMM）
a1 = Q·K2ᵀ  任务(m,n0) 行=N  B=K2[m]共享   → S1   （对角 → 按对角维 batch 拆成标准 GEMM）
o1 = P·V1   任务(n,m0) 行=M  B=V1[n]共享   → O1
o2 = P·V2   任务(m,n0) 行=N  B=V2[m]共享   → O2
```
- **host 预处理**（`scripts/gen_data.py`）：Q0[N,M,D]、Q1[M,N,D]（Q 转置）、K1/V1[N,K,D]、K2/V2[M,K,D]（K2/V2 转置 [K,M,D]→[M,K,D]）、V1ᵀ[N,D,K]/V2ᵀ[M,D,K]。
- **workspace 布局**（全连续，AIV 逐 (n,m) 向量加和）：S0[N,M,K]fp32、S1[M,N,K]fp32、P0[N,M,K]fp16、P1[M,N,K]fp16、O1[N,M,D]fp32、O2[M,N,D]fp32、L[N,M]fp32。
- S0 和 S1 用**不同布局**（S0 按 [N,M,K]、S1 按 [M,N,K]），AIV 逐 (n,m) 读两个 K 向量相加（不同地址，各自连续）。

### 12.5.2 已解决的踩坑（全部实测确认）

| # | 现象 | 根因 | 正确做法 |
|---|---|---|---|
| F1 | Add/Muls/Exp 只处理前 2 个元素 | **4 参版 `Add(dst,src0,src1,count)` 的 count 是元素数**（非 repeat 数） | 传 **K**（元素数），不是 K/64 |
| F2 | ReduceMax 的 dst[0] 不是全局 max | `ReduceMax(dst,src,tmp,count)` 每 64 元素输出 1 个（多 repeat） | **分段**：`ReduceMax(...,64)` 循环 K/64 次，再合并 |
| F3 | O1 错（PV 阶段） | PV 复用 BlockMmadQK（ColumnMajor B），B 物理须为 **V^T=[D,K]**（host 转置），不是原始 V=[K,D] | gen_data 输出 V1ᵀ[N,D,K]；kernel B 偏移 n*D*K |
| F4 | count 语义统一：Exp/Cast/Adds 也是元素数 | 同 F1 | 全传 K/D |

### 12.5.3 ✅ 已解决：跨核同步用"逐块流水 + 全核参与"（实测 PASS）

**现象**：整幅同步（AIC 全算完 S → 通知 AIV 全算 softmax → 通知 AIC 全算 PV）在 `aclnrtSynchronizeStream` 死锁。

**根因**：AscendC `CrossCoreSetFlag/WaitFlag` 是 **1 Set 配 1 Wait 的核间配对**。整幅同步下：
- 只有核 0 Set → 20 个 AIC Wait 只有 1 个通过 → **死锁**。
- 所有核无条件 Set（含没做事的核）→ AIC 提前读半成品 P0 → **精度错**。

**已解决的正确架构（实测 7+ shape 全 PASS）**：
1. **逐块流水（FAI 模式）**：每个 (n0,m0) tile 内 AIC 做 A0/A1 → Set qkReady → AIV softmax → Set softmaxReady → AIC C1/C2 → Set pvReady → AIV rescale。flag 按块 1:1 配对。
2. **全核参与**：每个拥有 tile 的 AIC 核 / AIV 子核对都参与 Set+Wait；**无 tile 的核直接 return，绝不碰 flag**。
3. **AIC 与 AIV 的 taskIdx 必须对齐**（都用 `taskIdx = coreIdx` / `aicIdx`，步进 `coreNum`，处理相同 tile），每对核间 1:1 flag。
4. **多核分发**：`for (taskIdx = coreIdx; taskIdx < totalTiles; taskIdx += coreNum)`，支持 tiles > 核数。

**另一个关键：纯 AIC 方案不可行**（AIC 核无 UB/VECCALC）。AIC 核用 BlockMmad 的 L1/L0（cube 内存）做 GEMM 完全正确，但 **AIC 核上跑 AscendC vector softmax（DataCopy/Add/Exp 到 UB）失败（P0 全 0）**——`resource.ubBuf`（VECCALC）是 AIV 核的内存，AIC 核访问无效。**softmax/rescale 必须在 AIV 核做**（框架 Matmul API 封装路径可绕过此限制，手搓路径不可）。

### 12.5.4 ✅ 已解决：BlockMmad 连续调用需 PipeBarrier（异步残留）

**现象**：GEMM 循环（A0/A1/C1/C2 连续多次 `blockMmadQK`）出现**前几个调用错**（D=32 时 S1 的 m=1,2 错；O2 的 m=1 错），之后稳定。

**根因**：标准 `BlockMmad` 连续调用都复用 `l1ATensor[0]`（isFirst 时载入）。前一次调用的异步 MTE（copyL1ToL0A 读 l1ATensor[0]）未完成时，下一次 `copyGmToL1A` 覆盖 → 前几次"接力"边界错。

**已解决**：**每次 `blockMmadQK` 调用后加 `AscendC::PipeBarrier<PIPE_ALL>()`**（强制本次 GEMM 完全串行）。代价是损失流水重叠（性能下降），但正确。**性能优化时**应改用乒乓 buffer / 减少全屏障。

### 12.5.5 ✅ 已跑通：精度与性能实测

- **精度**：9 shape（D=32/64/128，N/M 跨 16~1024）**matched_ratio=1.0，max_abs<0.016**，全 PASS。
- **性能演进**（N=16 M=128 K=128 D=64，标杆 aclnn 274μs）：
  - v1（逐行 softmax + 全屏障）：1614μs（0.17×）
  - v2（批量 softmax，见 §12.6）：685μs（**0.40×**）
  - 大 shape 最好 case（256×1024×64）：**3561μs vs 标杆 5943μs = 1.67× 反超标杆**
- **几何平均 0.38×**（11 case）。剩余差距主因：tile 数 < 核数时单核串行 + 同步头开销（profiler 显示 Task 611μs 中核最长仅 43μs，**头开销 92.9%**）。

### 12.5.6 性能优化实战经验（v1→v4，全部实测验证，最终几何 1.68× 反超标杆）

**优化演进路径**（16×128×64 case，标杆 272μs）：

| 版本 | 关键改动 | 该 case 用时 | vs 标杆 |
|---|---|---|---|
| v1 | 逐行 softmax + 每 GEMM 后 PIPE_ALL | 1614μs | 0.17× |
| v2 | softmax/rescale 批量化（16 行/批）+ 屏障分层 | 687μs | 0.40× |
| v2' | **MTE1_MTE2 token dance**（替代长循环的 PIPE_ALL） | 454μs | 0.60× |
| v3 | + **动态 tile**（TILE 128→32/64 按形状选择） | **134μs** | **2.04×** |

**关键优化技术（全部实测）**：

| # | 技术 | 收益 | 实现要点 |
|---|---|---|---|
| P1 | softmax/rescale 批量化（16 行/批） | 2.3× | 向量算子 count 传批量元素数；同 n 内 m 连续 → S0 连续 1D 拷贝、S1 用 2D strided load；每 (row,seg) Reduce 用**私有 tmp 槽（≥256 floats）**（槽小读出垃圾→exp 溢出 inf） |
| P2 | 屏障分层 | 含 P1 | **必须 PIPE_ALL**：DataCopy load 后（MTE2→V）、Cast→store 前（V→MTE3，竞态实例：sub0 赢竞态 sub1 读旧零）；纯向量间用 PIPE_V |
| P3 | **MTE1_MTE2 token dance**（W2 的正确解） | 1.5×（687→454） | FAQK 组件每次调用后 Set(MTE1_MTE2, pp)（L1A 读排空信号，内部供 k+2 次调用消费）。调用间插入 `Wait(MTE1_MTE2, ppPrev); Set(MTE1_MTE2, ppPrev)`——**先等读方排空再补回 token**，既消除覆盖竞态又不破坏组件内部事件配对。**短循环（<32 次）用 PIPE_ALL 更稳，长循环（128 次）用 dance**；跨相位边界必须 PIPE_ALL 隔离；循环尾残留 SET 会挂起 profiling 版内核（token 账目要平） |
| P4 | **动态 tile 大小** | 3.4×（454→134） | tile 从编译期常量改为 tiling 字段。实测最优：默认 32；64 当 `tiles₆₄∈[12,32]`（中 shape）。tile 越小核越多但单 GEMM 越小——按形状调优。**注意：TILE≠128 曾在旧代码（全屏障版）触发 AIV 静默写零；token dance 版 32/64 全 PASS——该坑与同步结构相关，非 TILE 本身** |

**最终基线（v4，11 个 Step-4 case，手搓=事件计时 min-of-10，标杆=msprof）**：

| case | 手搓 (μs) | 标杆 (μs) | 加速比 |
|---|---|---|---|
| 16×128×64 (L0) | 133.8 | 272.5 | **2.04×** |
| 32×128×64 | 234.8 | 255.7 | 1.09× |
| 16×128×32 | 124.0 | 372.2 | **3.00×** |
| 16×128×128 | 150.9 | 476.7 | **3.16×** |
| 16×256×64 | 143.6 | 206.3 | 1.44× |
| 64×256×64 | 236.8 | 221.0 | 0.93× |
| 16×512×64 | 135.0 | 226.4 | 1.68× |
| 128×512×64 | 833.0 | 370.2 | 0.44× |
| 256×1024×64 (L3) | 2981.5 | 5852.3 | **1.96×** |
| 16×1024×64 | 239.9 | 1041.9 | **4.34×** |
| 64×256×128 | 262.1 | 441.5 | 1.68× |

**几何平均 1.68×，9/11 反超标杆，10/11 ≥0.6×，唯一弱项 128×512（0.44×）**——该 shape tile 内 GEMM 链太长（每核对 256 次 GEMM 深度 1 串行），需要更深流水（官方用框架 Matmul API 的多级任务环流水）。

**坑（W 系列，实测确认）**：

| # | 现象 | 根因 | 对策 |
|---|---|---|---|
| W1 | 2D strided DataCopy **UB→GM store**（fp16）只写第一个 block | MTE3 2D store 不可靠（MTE2 load 正常） | store 退化为逐行 1D |
| W2 | 无屏障连续 GEMM：前几次调用数据错（L1A 覆盖竞态） | 下一调用的 copyGmToL1A 覆盖 l1A[0] 时，上一调用 MTE1 读未完成；组件内部 Set(MTE1_MTE2,pp) 由 k+2 次调用消费，保护滞后 2 拍 | **P3 token dance** |
| W3 | 全屏障版 TILE≠128 多 tile 时 AIV 静默写零 | 与同步结构相关的旧问题（token dance 版已消失） | 用 v2'+ 同步结构后 TILE 32/64 安全 |
| W4 | 批量寻址漏 tile 偏移 | 全局坐标 = tile 相对坐标 + tile 原点 | 统一 `nG = n0*TILE + r0/mBlk` 换算 |
| W5 | profiler 采集偶发全空（host/device_1 目录空） | 设备多次 core dump 后分析子系统劣化（全设备、官方 exe 也复现） | 降级 **aclrtEvent 流计时**（kernel 前后 event、warmup 3 + 10 次取 min；与 msprof 口径偏差 <5%）；aclnn 每次 launch 有 ~1.4s host 开销（executor 复用无效）不能事件计时标杆侧，官方基线沿用 profiler 健康期的 msprof 数据 |

### 12.5.10 ✅ 纯 skill 生成复合功能变体（sink/varlen/q8，2026-08-17 第二轮验证）

**验证协议**：同 §12.5.8（只读本手册），一次生成 3 个可组合特性 + 全组合，32/32 PASS（20 老 case 回归 + 12 特性组合）。

**特性实现（全部 v4 路径 K=128）**：
| 特性 | 数学 | kernel 落点 |
|---|---|---|
| **FA-Sink**（StreamingLLM 语义） | 虚拟 sink 列（score=sinkScore, V=0）：m'=max(m,sink)，l'+=exp(sink-m')，O 归一化自动正确 | softmax 相位：m 合并处 `if(m<sink)m=sink`；l 存储处标量 exp 加成 |
| **FA-Varlen**（核内前缀 mask，无 mask 张量） | k≥kvLen[b,n] → -3e38 | softmax 相位：`gLen.GetValue` 标量 + K 次 SetValue 构造掩码行 + Add |
| **FA-Q8**（W8A8 式核内反量化） | int8 输入 ×(1/scale) → fp16 staging → 正常 GEMM | **新增 dq 相位**：AIV tile 内反量化本 tile 操作数切片 → Set(dqReady id=4) → AIC Wait(dq) 再 A 相位（第 4 个配对 flag，逐 tile 1:1） |

**开销实测**（16×128×64，基线 157.5μs）：sink +5%、varlen ±0%、q8 +0.3%、四合一 +4.3%——特性近乎免费。

**新坑（W10-W13，全部实锤）**：
| # | 现象 | 根因 | 对策 |
|---|---|---|---|
| W10 | 特性 kernel 输出全零、kernel 正常返回（AIV 静默死亡） | **UB 向量算子 buffer 必须 32B 对齐**（标量 Exp 的 scratch 放 93188=4 偏移 → 核异常；流式路径 93184 对齐所以没事） | 所有 UB 偏移 %32==0；对齐坑的签名=输出全零+正常返回 |
| W11 | 量化输出大 ~750×（=sc²） | gen 的 scale 语义是 int8=x·s（乘），dequant 应 **除**：kernel Muls 需传 **1/s** | scale 契约：gen 写 s（乘法系数），host 求倒数传入 |
| W12 | dav_c220 无 int8→float、无 bf16←half Cast | 该架构 Cast 支持表有限 | int8 链路固定 half 中转；**q8 仅支持 fp16**（bf16+q8 host 拒绝）；另：**量化 golden 必须用 round(x·s)/s（量化-反量化值），不能用 x/s（未量化缩小值）**——这是测试侧最容易犯的语义错 |
| W13 | q8 dequant 输出"看似合法但全错"（反量化值对不上 int8 输入；某 plane 甚至全零），Cast/scale 单测都对 | **AIV DataCopy(GM→UB int8) 与后续 Cast 之间缺 PipeBarrier**——MTE2 加载未完成，V 管 Cast 抢先读 ubI8 陈旧值；fp16 路径同款代码没炸只因碰巧没踩同拍竞态 | DataCopy 加载后、Cast 前必须 `PipeBarrier<PIPE_ALL>`（MTE2→V）；改法：dequantPlane 在 DataCopy 与 Cast 之间补一条屏障即可，其余链（V→V、V→MTE3）原样 |

**特性设计通用法**（可复用到其它变体）：① mask 类特性（varlen/三角/局部窗口）= softmax 相位构造 -3e38 掩码行，与 mask 张量可叠加（双 -3e38 等价 max 语义）；② softmax 状态类特性（sink/logit-bias）= 修改 m/l 递推，不动 GEMM；③ 输入格式类特性（量化/低精度）= 新增 AIV 预处理相位 + 配对 flag，GEMM 主路径零改动。



### 12.5.9 ✅ P1/P2：spec 全覆盖（bf16/多BH/aux输出/K块流式，2026-08-17）

**P1（增量扩展，全部一次或近一次通过）**：
- bf16：kernel 元素类型全走 `ElementQ`，入口克隆 `FAFFABf16`（`bfloat16_t`）；gen/verify 用 torch `view(uint16)` 转 numpy
- B/H：`totalTasks = BH*nBlocks*mBlocks` 展平，`bh = taskIdx/tilesPerBH`；所有张量加 plane 偏移（**V1ᵀ plane 步长 = N·D·K（含 N！），V2ᵀ = M·D·K**）；mask 按 `b = bh/H` 共享
- softmax_max/sum [B,H,N,M,8]：softmax 相位直接 SetValue（host 预清零保证 [1..7]=0）
- 实测坑：P1 store 漏 op1 偏移 → 全 plane 写 plane0 互踩（smax 对但 O 全错——辅助输出对+主输出错=偏移在 store 侧）

**P2（K 块流式 + 在线 softmax，攻 3 个新坑后全通）**：

架构（KB=128，nK=K/128；tileDim 强制 32）：
```
per tile per block j（块级锁步，复用 3 个 flag nK 次）:
  AIC: A0_j(nBlk GEMM→S0t) + A1_j(mBlk→S1t) → Set(qk) → Wait(sm) → C1_j(P0t·V1ᵀcol_j→O1b) + C2_j(→O2b) → Set(pv)
  AIV: Wait(qk) → online softmax_j → Set(sm) → Wait(pv) → Oacc_j
online: mNew=max(mOld,blkMax); α=exp(mOld-mNew)(j=0 置 0); P=exp(S-mNew); l=α·l+ΣP
        Oacc = Oacc·α + O1b + O2b（j=0 直写防 NaN×0）; 末块: O=Oacc/l + 存 smax/ssum
workspace：per-core 槽（20×TILE²×KB，S/P/O 独立槽 + Oacc 专缓冲）——内存与 K/N/M 总量解耦
标量 exp 用向量 API 单元素（`Exp(ubSc,ubSc,1)`），标量 expf 在核内不可用
```

**P2 新坑（全实测）**：

| # | 现象 | 根因 | 对策 |
|---|---|---|---|
| W6 | m/l 跨行混乱（probe 显示单行递推正确、存储值错） | **tile 局部 buffer 的 2D strided LOAD 也混行**（W1 只知 store；全局大 stride 的 2D load 正常、tile 局部小 stride 的会混） | S1t/O2b 全部逐行 1D |
| W7 | C 相位 GEMM 结果乱 | **B 是 [D,K] 平面的列切片：列 stride=K≠KB**；`MakeLayout(KB,D)` 声称 stride=KB 错（v4 中 K==KB 掩盖） | 布局传全宽 `MakeLayout(K,D)`，块范围由 `GemmCoord.k=KB` 表达 |
| W8 | Oacc 阶段 chunk 起始行(r=0)对、r≥1 全错 | O2b 行 stride=nBlk·D 被当连续 1D 拷贝 | 逐行 1D load |
| W9 | 偶发 Bus error（加 printf 即好） | 块级锁步边界缺隔离 | 每块起止 + Wait 后各补 PIPE_ALL（保守但稳，5×压测通过） |

**P2 验证**：K=256/512/1024 × mask/bf16/多BH/D32/64/128/大N/M 共 8+8 组合全 PASS（O+smax+ssum 三输出）；K=128 全量回归 20/20。流式性能：K=1024 1252μs，workspace 仅 ~10MB（非流式方案同 shape 需 1.3GB）——**大 K 只有流式可行**。



### 12.5.8 ✅ skill 闭环验证：纯 skill 生成 FFA-Mask 变体（2026-08-17）

**验证协议**：只读本手册 + FFA 需求说明，不打开任何外部仓 / catlass 组件源码。目标 = 全新功能（atten_mask 掩码，此前从未实现）。

**实现（全部从手册条目推导，一次成型零调试）**：
- 掩码语义：`S += mask×(-3e38)`，mask [N,K] 只依赖 (n,k) → 同一 softmax chunk 的 16 行共享一条 mask 行（chunk 不跨 n，P1 已保证）
- 插入点：scale-Muls 之后、ReduceMax 之前（max/sum 均需排除被掩位置）
- 屏障：mask DataCopy 后 PIPE_ALL（P2 MTE2→V）；Cast→Muls→Add 向量链间 PIPE_V（P2）
- UB：ubM16/ubM32 放 ubO32 之后（67584 起）
- 数据：随机 25% 掩码 + 强制 k=0 打开（防全掩行 nan）

**结果**：
| 验证项 | 标准 | 实测 |
|---|---|---|
| 掩码精度 | ≥6 shape PASS | **8/8 PASS**（matched=1.0，max_abs<0.005，一次通过） |
| 无掩码回归 | 不破坏原路径 | 12/12 PASS（mask=nullptr 分支） |
| 性能 | ≤1.3× 无掩码版 | **1.03~1.08×**（掩码开销 3-8%） |

**结论**：本手册具备独立支撑 FA 变体手搓开发的完备性（skill → 生成 → 测试通过全链路实证）。掩码模式追加仅需：1 个 GM 输入 + chunk 内 1 次 mask 行 load/Cast/Muls + rr 次 Add + 3 个屏障。

---

### 12.5.7 性能对比测试：cannbot ops-profiling 工具化流程（重要——不手写解析）

**口径**：`msprof` 采集 + skill 自带 `msprof_perf_summary.py` 解析，禁止自己写解析脚本。

| 场景 | 工具路径 | 说明 |
|---|---|---|
| 算子目录（model.py vs model_new_ascendc.py） | `msprof_profile_run.sh --quick/--compare --output-dir=<op_dir>` | 标准加速比对比，输出 performance.json/perf_report.md。model 的 `__init__` 做所有 setup（设备任务留在 init 窗口），`forward` 每次 launch 恰好一个目标 kernel（run-split 才干净） |
| 任意 C++ 直调 exe | `msprof_profile_run.sh --warm-up=N --output=<dir> -- <exe> args` + `msprof_perf_summary.py <PROF_GROUP> <ops_dir>` | 标准采集（7 组 aic-metrics），解析输出 "Duration: xxx us" + 瓶颈分析（头开销占比、cube_util、各 pipe 占比）。FA 类 exe 用此路径 |
| 批量 | `--batch --base-dir` | 多算子目录并行 |

**实测陷阱**：
- 解析输出行是 `Duration: 611.312us`（op_summary 聚合），不是 task_time 的 min——对一次 launch 多 task 的 kernel 用 op_summary 口径；
- torch_npu 进程内 ctypes 调 aclnn C API：M≥256 时 GetWorkspaceSize 前崩溃/挂死（与 torch 堆/上下文不兼容），**标杆 aclnn 标杆必须用独立 exe**（子进程 kernel 不被 msprof 采集，所以 exe 要作为 `--quick/--` 的被测应用本身）；
- `msprof_perf_summary.py` 会把 CSV 归档到 `<ops_dir>/docs/perf/round_NNN/`——注意磁盘；
- wrapper 设 `ASCEND_RT_VISIBLE_DEVICES=N` 后进程内设备索引归零——exe 内 `aclrtSetDevice` 要用 0；
- msprof 采集偶发静默失败：PROF 目录 host/device_1 全空、log 报 "Export ... No such file or directory"——被测 app 正常、磁盘可写也会发生（实例：64×256 手搓 kernel 连续 3 次）；同 binary 其它 shape 正常。**遇此先换 case 复测/换 quick 模式，不要怀疑自己的 kernel**；
- 磁盘 100% 时 msprof trace 截断 → 时间失真（见 profiling 教训记忆）；每次采集完即删 PROF 目录。

**FAFFA v4 最终对比**：见 §12.5.6 最终基线表（几何 1.68×，9/11 反超标杆，10/11 ≥0.6×）。

**计时口径**：手搓侧优先 msprof（cannbot standard/quick），profiler 故障期降级 aclrtEvent 流计时（warmup 3 + 10 次取 min，与 msprof 偏差 <5%）；标杆侧 msprof（健康期数据）+ 独立 exe。

**性能报告字段**（skill 解析器直接给出）：Task Duration、核最长耗时、**头开销占比**、aic/aiv 各 pipe（mac/mte/vec/scalar）占比、cube_utilization、Memory/L0/UB 带宽——定位"慢在同步还是计算"一步到位（例：FAFFA v2 头开销 92.9% → 下一步优化目标是并行度而非算力）。

---

### 12.5.11 ✅ FFA 性能优化：FAIQK→FAQK 组件切换 + 紧凑布局批量搬运（2026-08-18 实测，0.52×→~1×）

**背景**：FFA 生成工程（skill 从零生成）v1 用 `MmadAtlasA2FAIQK`，S/O workspace 行 stride 被组件 LayoutC 硬编码 512，AIV 逐行 1D 搬运（msprof AIV mte3 26%），几何 0.52×。参考自写工程（1.68×）切换 `MmadAtlasA2FAQK` 后同 shape 达 ~1×（16×512×64 event 240us vs 标杆 226us）。

**关键经验（全部实测）**：

| # | 要点 | 细节 |
|---|---|---|
| W14 | **FA 融合 GEMM 组件选型** | `FAIQK`（FAI infer）LayoutC 硬编码 512 row-stride + KV 块堆叠协议（nIdx 恒 0 坑，见 N1）；`FAQK`（普通 QK）LayoutC stride 可配 → S/O 可紧凑 [TILE²][KB]/[TILE²][D]，AIV 连续多行一次 DataCopy。**FA 变体要紧凑布局批量搬运 → 用 FAQK**（`Gemm::MmadAtlasA2FAQK`，无模板参数） |
| W15 | **HardEvent 预置必须匹配组件** | FAQK 构造自带预置 6 个（M_MTE1/MTE1_MTE2/FIX_M × 0/1）+ 析构配平。FAIQK 的 18 个手动预置对 FAQK **多余**：预置 token 无人消费 → kernel 出口 WaitFlag 挂死（信号量缺 token）。换组件必须核对构造/析构自管理，删掉多余预置/出口 |
| W16 | **layoutB 行列必须匹配物理排列** | B 是 ColumnMajor 消费，layoutB(rows,cols) stride=rows。V^T 平面物理 [D,K]（d 主序，stride K）→ 必须 `layoutVB(K, D)`（rows=K）。**D==K 时 (D,K) 碰巧对（掩蔽），D<K 时全错**——这是最容易静默错的位置 |
| W17 | **紧凑布局批量搬运** | S0（[i][r][KB] 拍平）、O1b、P0、O 输出 16 行连续 → 单次 DataCopy 大块（消 15/16 次 MTE3 发射）；S1/O2（转置 [r][i]）受 A1 写连续限制只能逐行 |
| W18 | **Duplicate+SetValue 标量散点竞态** | `Duplicate(buf,0,N)` 清零后 `buf.SetValue(i,v)` 散点写，再 `DataCopy` 大块 → **读到全 0（即使中间 PIPE_ALL）**——标量写与向量清空的 UB 同步不可靠。smax/ssum 逐行 `SetValue(0,v)+PIPE_ALL+DataCopy(8)` 是安全形态 |

**FAQK 调用签名**（无 loadQGM、无 blockTable）：
```cpp
blockMmad(gA, gB, gC, layoutA, layoutB, layoutC, GemmCoord{M,N,K}, uint32_t& pingpongFlag, bool isFirst);
// isFirst=true 每次（单块 GEMM，重载 A）；pingpongFlag 调用后翻转；LayoutC 紧凑 {M, cols}
```
**性能路径**：FAIQK(512) 0.52× → FAQK 紧凑 0.56× → +O 批量/smax逐行同 shape ~1×（16×512×64）。剩余瓶颈：AIV scalar（逐行 m/α GetValue）与 S1/O2 逐行（转置固有），大 shape D32/D64 标量主导。


### 12.5.12 ✅ FFA 进一步优化实验 + 泛化性能结论（2026-08-19 实测）

**A. tile 级锁步（参考 s1s2 非流式组织，K=128 PASS / K>128 未收敛）**
- 方案：A0/A1 全 K 列块连发物化 S 槽 [TILE²][K]、两遍式 softmax（pass1 全 K 行 max + pass2 exp/sum/P 写回）、C 相 O1_j 分槽 [nK][TILE²][D] + AIV 累加。锁步 3×nK→3、online-softmax α 链消除。host workspace 同步 full-K。
- 结果：K=128 全 case PASS（D32/64/128、bf16、mask）；**K>128 时 A0 相 j=0 列块写 S0 错**（h1 对 h0 错；循环交换/任务边界 PIPE_ALL/slotP 修正后仍复现，疑 FAQK 组件在 nK>1 连续 GEMM 时 j=0 的 L1B/FIX 时序）。15+ 轮未收敛 → 止损回退 FAQK 流式版。
- 新坑（承接 §12.5.11 追加）：

| # | 现象 | 根因 / 对策 |
|---|---|---|
| W19 | 逐行 ReduceMax/Sum + 大块 vector 混用，只有行 0 对 | **UB 缓冲行布局与 API count 语义必须全局一致**：S0/S1 用 [rr][KSEG] stride 布局，则 Add/Exp/Cast/load 全部按 [rr*KSEG] 逐行，禁止 bulk 紧排 load + per-row 运算混用 |
| W20 | dump 慢跑时对、快跑时错（"竞态假象"） | **workspace 槽步长 kernel/host 两侧必须同一公式**（slotP 用 KB 而 P 布局 full-K → 跨核越界互踩）。任何布局改动同步核对 kernel/host 两处 slot 定义 |

**B. 泛化性能结论（skill Step-4 覆盖矩阵 case，ops-profiling msprof 口径）**
- **大序列（M/K≥1024）是 FA 变体强项**：K 流式在线 softmax 单 kernel，标杆 aclnn 长序列框架调度退化（cube_util 48%/头开销 52%），实测 4.4~7.8× 反超（1024²/2048²/4096×1024/多batch）。
- 基础/中小序列（128~512）0.77~1.14×，小 shape（低 N）1.4~2.6× 反超；Sq≠Sk 0.78~1.00×。
- **大并行度（H×N≥1024）是独立短板**（0.36~0.52×，与序列长无关）：H*N 大 → tile 任务数爆炸（H*N=24576 时每核上千 task），每 task 4 相 GEMM+3 flag 固定开销主导。对症：N 维 pack 进 GEMM M 维（tile 数 ÷32）/ task 批处理，不是序列方向优化。
- 性能测试口径：**必须用 ops-profiling skill**（msprof_profile_run.sh + msprof_perf_summary.py 读 Duration），勿自写计时；标杆 executor 不可复用（每次 launch 重建，N6）。case 选取按 Step-4 覆盖矩阵（B/Sq/Sk 组合、Sq==Sk、Sq<Sk、D 扫描、多 batch），禁止随手枚举。

### 12.5.13 ✅ FFA 参考源码解读 + 自写工程实测 + 性能测试方法论（2026-08-19 补充）

**A. s1s2 非流式组织参考要点（生产版 FFA 实现形态）**
- 生产版是**非流式**：bmm1 一次算完整 S2 全 K → 物化 workspace（mm1Res 双缓冲）→ SoftmaxFlashV2 沿全 K 一次归约 → bmm2 一次。**用 workspace 换时间**，避免流式每块 AIC↔AIV 锁步。
- 生产版用框架 `matmul::Matmul` 高阶 API + `SoftmaxFlashV2`，AIC/AIV 由框架级任务环流水编排（多级任务缓冲 extraInfo[3]），不做逐块 CrossCore 锁步。
- 对照：手搓流式（锁步 3×nK）在大序列（nK 大）时锁步往返 ×nK 放大；非流式只锁 3 次但 workspace 大。**大 K 场景锁步次数是关键**。
**B. 自写工程实测破除"旧工程更快"误判**

- 自写工程声称 1.68× 是 **H=1 小 case（N×M×D 三参数，16×128×64 等）几何平均**；实测其大 shape（1×32×256³×32 = 0.41×、1×16×384³×64 = 0.31×）与手搓新版几乎一样慢。**两代算子大并行度同瓶颈，1.68× 是 case 选取偏向小 shape 的错觉**——对比性能必须按覆盖矩阵选 case，不能只看几何平均。

**C. 性能测试方法论（FA 变体完整版，承接 §12.5.7）**
1. **工具**：必须 `ops-profiling` skill（msprof_profile_run.sh --warm-up=N --output= -- <exe>；msprof_perf_summary.py 读 `Duration:`）。禁自写计时。标杆 executor 不可复用（每次 launch 重建，N6）。
2. **case 选取**：按 design skill Step-4 覆盖矩阵（B/Sq/Sk 组合、Sq==Sk、Sq<Sk、Sq>Sk、D 扫描、多 batch、尾块），**禁止随手枚举无说明 shape**；比例上小/中/大序列都要有（用户口径：<100us 10%/100-500 20%/500-1000 40%/>1000 30% 是指导，非硬性）。
3. **口径澄清**：FA"大 shape"= 序列长大（Sq/Skv），不是总数据量大；B×H×N 是并行度轴。评测按序列长分档。
4. **环境**：采集前查磁盘/设备健康（msprof 异常先怀疑环境，见 profiling-anomaly 记忆）；采完删 PROF 目录。
5. **报告格式**：算子用例 | shape布局(BHNMKD) | shape参数 | catlass用时(μs) | npu标杆用时(μs) | 加速比(npu/catlass)，组平均 + 总平均。

**D. 需求符合性完善 checklist（FA 变体交付前核对）**
- 约束：序列长 %128（tile 整除）、分组 %16（TILE 动态 32/16 支持 %16）、D∈{32,64,128}、B/H 范围
- scale 默认值对齐需求（FA 惯例 1/sqrt(D) vs 需求默认 1.0——显式传参可覆盖，但默认要与需求一致）
- mask 外部格式（需求 [B,1,N,1,K] BOOL/UINT8，host 转内部；1=遮蔽→-3e38）
- 输出 aux [.,8] 槽：有效值在 [0]，其余 0（host 预清零 + kernel 只写槽0）
- 精度双口径：CPU golden（fp32 分块累加）+ npu 标杆互证（smax/ssum 应位级一致）

### 12.5.14 ✅ FFA 完整生成配方（FFA 生成工程 终态架构，2026-08-19 定稿）——"kernel 怎么生成"的可复现路径

> 本节是本轮（catlass-op-design → catlass-op-develop 纯 skill 从零生成、精度双口径全 PASS、泛化几何 2.19×）的**终态架构配方**。照此可直接再生一个同形态 FFA kernel，不必重走弯路。

**生成流程（两段 skill）**：
1. `catlass-op-design`：FA 路由 → 判定"FA 变体（双 KV）"→ 产出 DESIGN.md（组件选型表/UB 预算/坑清单）
2. `catlass-op-develop`：按本手册 §10 变体映射法落 4+3 文件工程（kernel/tiling/common + host + gen/verify/run_cases）

**终态架构（所有公式实测验证）**：
```
任务 = (bh, n0-tile32, m0-tile32)，K 按 128 列块流式（块级锁步，3 flag 复用 nK 次）
AIC（每 task 每 j 块）：
  A0 相: TILE 个 GEMM [TILE,KB,D]  Q0[n0+i,m0:*]×K1[n0+i]→S0[i][r][k]（紧凑 stride=KB）
  A1 相: TILE 个 GEMM              Q2T[m0+r,n0:*]×K2T[m0+r]→S1[r][i][k]
  →Set(qk)→Wait(sm)→
  C1 相: P0[i-tile]×V1T[n0+i] 列片 j→O1b；C2 相: P1[r-tile]×V2T[m0+r] 列片 j→O2b
  →Set(pv)
AIV（taskIdx 对齐，coreIdx=BlockIdx/subBlockNum，coreNum 不除）：
  Wait(qk)→chunk(16行)×[load S0/S1 逐行→Add+Muls(scale)→mask 行加→分段 ReduceMax64×2
  →m/α 递推(紧排 UB 数组)→P=exp(S-mNew) 逐行 Adds→分段 ReduceSum→P cast 写 P0/P1]
  →Set(sm)→Wait(pv)→Oacc=α·old+(O1+O2)→末块 ÷l+cast+O/smax/ssum 写出
```

**组件与布局契约（§12.5.11 成果内嵌）**：
- GEMM 组件：`BlockMmad<MmadAtlasA2FAQK, GemmShape<128,128,128>, GemmShape<128,128,128>, A=RowMajor, B=ColumnMajor, C=RowMajor>`；调用 `blockMmad(gA,gB,gC,layoutA,layoutB,layoutC, GemmCoord{M,N,K}, uint32_t&pp, bool isFirst=true)`；每次 GEMM 后 `PipeBarrier<PIPE_ALL>`（W2）；**勿手动预置 HardEvent**（FAQK 构造自带 6 个，W15）
- host 预处理布局（gen_data 侧转置）：`q0[B,H,N,M,D] / q2t[B,H,M,N,D] / k1[B,H,N,K,D] / k2t[B,H,M,K,D] / v1t[B,H,N,D,K] / v2t[B,H,M,D,K] / maskadd[B,N,K]fp32(0/-3e38)`
- A 相 B 偏移：`gK1[bh,n] + j*KB*D`；C 相 B：`gV1T[bh,n] + j*KB`（V^T 列步长 K，W7）；C 相 layoutVB=(K,D)（W16）
- workspace per-core 紧凑槽：S0/S1=[TILE²][KB]fp32、P0/P1=[TILE²][KB]elem、O1b/O2b 双缓冲[TILE²][D]、Oacc=[TILE²][D]；独立 GM 指针透传（Δ4）
- AIV UB（偏移%32，TILE=32 预算 ~114KB）：S0c/S1c/Sum/Pf 各 [16][KSEG=128..256]、O 系 [16][D]×3、M/L 状态 [TILE²]、分段 Reduce 用 count=64（F2）

**工程 7 文件骨架**：`op_kernel/ffa_kernel.asc`（AIC/AIV 双入口模板分流 g_coreType，extern "C" FAFFAFp16/Bf16）+ `kernel_common.hpp`（常量/Params/Tiling）+ `host/ffa_host.asc`（读 bin→malloc/memcpy→tiling memcpy→hwSync→<<<blockDim=AIC核数>>> launch→dump）+ `gen_data.py`/`verify.py`(标准模板判定+FA mixed-tolerance)/`run_cases.sh` + `CMakeLists.txt`（CATLASS_ARCH=2201 + --npu-arch=dav-2201 + link ascendc_runtime/ascendcl/...，**host 与 kernel 单 TU**：host include kernel 文件）

**验证闭环**：单 case 精度（gen→run→verify）→ 12~15 case 回归（run_cases.sh）→ 标杆互证（FFA_DUMP_RAW=1 gen + bench_official dump off_*.bin 比对，smax/ssum 应位级一致）→ 需求符合性 checklist（§12.5.13-D）→ ops-profiling skill 泛化性能（Step-4 覆盖矩阵选 case）

**已知边界（生成时直接告知用户）**：长序列(M/K≥1024)强项 4.4~7.8×；大并行度(H×N≥1024)短板 0.36~0.52×（对症 N-pack，未实施）；标杆在 ≥4096² 执行超时（M·D≥65536 文档声明退化区）。


---

## 12.10 s1s2 FA kernel 实现原理（可手搓知识，算法级剖析）

> 本节是**实现级剖析**（不靠拷贝、靠理解后能重写）：生产版 FlashAttentionScoreV3 类 kernel 为什么快、结构怎么搭、各相位怎么做。手搓目标：按本节逻辑写出等价 kernel。

### A. 架构总览（为什么单核融合 > 跨核 GM staging）
生产版把 **bmm1(QK) + softmax + bmm2(PV) 放进同一个核**（matmul 在 AIC/Cube，softmax/divide 在 AIV/Vector），关键：
- **S/P 全程不落 GM**：bmm1 的 C 直接写 UB（mm1Res 在 UB），softmax 在 UB 上做，bmm2 的 A（P）从 UB 直接进 cube → **没有 S/P 的 GM 往返**（这是 vs 手搓/FAInfer 跨核 GM staging 快 2-3x 的根因）。
- 代价：AIC/AIV 同核内用 **hardware event（PIPE 间 Set/Wait）** 做数据依赖同步，而非跨核 flag。

### B. 任务切分与 per-core workspace（公式）
```
totalSize = BH * ceil(Sq / 128)          # 每个 task = 一个 (batch, 128 行 query) 块
coreNum   = min(totalSize, aivNum)       # AIV 核数（40）
splitFactor = CeilDiv(totalSize, coreNum)
if (L1Reuse: D∈[64,128]) splitFactor *= 2   # 每核多分一半，配合 L1 驻留复用
blockIdx 映射：AIC = GetBlockIdx()*2；AIV = GetBlockIdx()   # AIV 逻辑核 = AIC 的 2 倍
workspace：每个逻辑核一个独立区，基址 = workspace + blockIdx * totalOffset
  totalOffset = mmNRatioOffset*bmm1AndVec1Ratio + mm2Offset*2*GM_DOUBLE_BUFFER (+ pseAlibi)
  mm1Res[2]（bmm1 结果双缓冲，fp32）、stage1Res[2]、mm2Res[2]（bmm2 结果）、vec2Res[2]
```
- 主循环遍历 `multiCoreInnerOffset..limit`（= blockIdx*splitFactor .. +splitFactor），即**每核处理连续的一段 task**。

### C. 主循环结构（s1×s2 双层 + 三深流水）
```
for multiCoreInnerIdx in [blockIdx*splitFactor, +splitFactor):
    GetS1LoopRange(...)                 # s1 = query 行块（每 task 内再按 S1 块切）
    for bngIdx ...:                      # batch-head 块
        for s1Idx ...:                   # query 块
            taskId++
            extraInfo[taskId%3].SetExtraInfo(...)   # 三深流水：task-2 消费、task-1 计算、task 预取
            if hasNext: IterateBmm1(extraInfo[taskId%3], bmm1)   # QK：入队下一块
            if taskId>0: ProcessVec1(extraInfo[(taskId+2)%3]); SetFlag(MTE3_MTE2)   # softmax：消费 2 块前的 S
            if taskId>1: WaitBmm2Result()                        # PV：消费 1 块前的 O1
            if taskId>0: WaitFlag(MTE3_MTE2); IterateBmm2(extraInfo[(taskId+2)%3])  # PV：等 softmax 后入队
            if taskId>1: ProcessVec2(extraInfo[(taskId+1)%3])    # divide：O=O1/l + aux
```
- **三深流水**：`extraInfo[taskId%3]` 让 QK(task) 与 softmax(task-2) 与 PV(task-1) 重叠——这就是生产版高吞吐的第二个根因（不是每 task 串行 QK→softmax→PV）。
- 事件链：`IterateBmm1 后 MTE1_MTE2`（L1→UB 排空）→ `softmax 后 MTE3_MTE2`（S 写出可见）→ `bmm2 WaitBmm2Result 用 WaitIterateAll`（cube 写 mm2Res 全局可见）。

### D. matmul::Matmul 用法（kfc 机制，直调要照抄）
```
REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), bmm1, &bmm1Tiling, bmm2, &bmm2Tiling)  // 构造时注册
bmm1.SetOrgShape(M,N,K); bmm1.SetTensorA(gA, ...); bmm1.SetTensorB(gB, ...)
bmm1.template IterateAll<false>(mm1Res[taskIdMod2], ...)   // 异步：入队后不等完成
bmm1.WaitIterateAll(); bmm1.End()                           // 等待 + 结束（GetTensorC 前必须 WaitIterateAll）
```
- `IterateAll<false>` = 异步模式（false=不等完成），配 `WaitIterateAll`；**手动 `while(Iterate<true>)` 会跳过 WaitIterateAll 的 cube→GM 可见性保证 → 非确定性错**。
- tiling 结构（`bmm1TilingData/bmm2TilingData`）必须按 `SetBmm1TilingInput/SetBmm2TilingInput` 逐字段填（M/N/K、baseM/baseN、singleCore、fixsplit、shareL1/L0C）。

### E. 各相位细节
- **QK（bmm1）**：A=Q[行=rowNum, K=D]，B=K[列=Skv, K=D]，C=mm1Res[UB fp32]；scale 在 softmax 前 Muls。
- **softmax（ProcessVec1）**：读 mm1Res → 逐行 max → Brcb 广播 → exp → 逐行 sum → P cast fp16 写 stage1Res（UB）→ 供 bmm2 作 A。
- **PV（bmm2）**：A=P[UB, 行=rowNum, K=Skv]，B=V，C=mm2Res[fp32]；**A 直接指向 softmax 的 UB 输出**（不落 GM）。
- **divide（ProcessVec2）**：读 mm2Res + l → Brcb 广播 gl → Div → cast fp16 → 写 O GM；smax/ssum 同步导出。

### F. L1Reuse 原理（D∈[64,128] 必须）
生产版对 `D>=64`（+NO_MASK+fp16）开 `enableL1Reuse`：**K/V 块在 L1 驻留复用**（多个 query 块共享同一 kv 块），此时 splitFactor×2（每核 task 加倍）让 L1 复用收益充分。**模板 enableL1Reuse 与 tiling splitFactor×2 必须配对**——只开模板或只改 tiling 都会错（D=128 按头坏 / varlen B>1 重损）。D<64 或其它条件走非 reuse 模板（L1 小，K/V 不驻留）。

### G. 特性落点（在这个框架内手搓新特性）
- **sink**：softmax 的 m 计算处加 `m=max(m, sink)`、分母加 `exp(sink-m)`；sink 值按头索引 `sinkGm[n2oIdx*gSize+goIdx]`。
- **varlen**：每 (batch,head) 的 kvLen 进 `actualKvseqlen`，tiling 的 s2 循环按长度截断（标杆 no-mask 路径天然支持尾块）。
- **q8**：反量化前置（int8→fp16）或在 bmm 装载时 on-load cast；融合进 bmm 装载可省一次 GM 全量往返（单 batch 长序列收益最大）。
- **mask 变体**：在 softmax 前把 mask 加进 mm1Res（Add 偏移），比 after-softmax 更稳。

### H. 手搓自查
- [ ] S/P 不落 GM（mm1Res/stage1Res 在 UB，bmm2 A 指 UB）
- [ ] 三深流水（taskId%3）——不是每 task 串行
- [ ] `IterateAll<false>`+`WaitIterateAll`（禁手动 Iterate 循环）
- [ ] L1Reuse 模板 + splitFactor×2 配对
- [ ] workspace per-core 基址 = blockIdx*totalOffset
- [ ] 107 例精度（含 196608）+ 对标杆性能同轮复测


---

## 12.11 fa-sink kernel 手搓流程（S1 单遍组织，Skv≤1024 默认路径，2026-08-26）

> **⚠ 2026-08-27 更新**：kfc 单 TU 直调路径（`__mix__(1,2)` + `REGIST_MATMUL_OBJ` + chunk 流式）见 **§12.13**——任意序列长、每 chunk 单 IterateAll、原子累加 O_acc、实测 geomean 0.693 vs 标杆 sink 算子，**优先走 §12.13**。本节保留 S1 模板的算法语义、tiling 公式与长序列基线（0.92x 扫描数据仍有效）；本节的 `SetOrgShape` 5 参形态/tiling 字段手工推导仅在改写官方模板形态时需要。sink/varlen/q8 的相位落点（§12.6/§12.12）在 §12.13 基座上同样适用。
>
> **★2026-09-02 补充**：S1 模板 kernel 的**代码级逐步骤**
> 已单独成文 [fa-sink-handcraft-recipe.md](fa-sink-handcraft-recipe.md)——三深流水精确结构/sink 正确
> SoftmaxFlashV2 isUpdate 播种用法/workspace 精确公式/sparse 跳块；本节是算法语义与 tiling 公式层，
> 生成时两份配合用（先 recipe 后本节查公式）。

> 本节是 **fa-sink（sink/none 变体）kernel 的手搓流程**：从算法到可交付 kernel，**全程手搓、不复制/移植/调用任何外部源码**。算法基础见 §12.1-12.4（FAInfer 骨架）+ §12.10（单核融合 s1s2 算法原理）；本节只补 **Skv≤1024 的"S1 单遍组织"变体差异 + sink 特性落点 + host 驱动 + 验证**，使 skill 能直接据此手搓生成 fa 算子。§12.8/12.9 的"复制外部源码"捷径已弃用。
> fa-sink 性能基线（手搓 S1 单遍）：sink 对标杆 aclnn 0.80x（混合口径）/ 0.922x（H=16 长序列扫至 458752），107/107 精度。剩余差距是 CANN 工具链 codegen（见 E），非源码可修。

### A. 算法变体选择（Skv≤1024 → S1 单遍组织）

FA 单核融合（bmm1 QK + softmax + bmm2 PV，S/P 不落 GM，见 §12.10 A）在 AIV 侧有两种 s2 维组织：

| 变体 | AIV s2 组织 | 适用 | fa-sink 实测对标杆 |
|---|---|---|---|
| s1s2 双层（§12.10）| s1×s2 双层循环 + 多遍 softmax | s2>128 或 D/特性特殊 | 0.58x（Skv≤1024 勿用） |
| **S1 单遍（本节）**| **单遍按 s1 块迭代，s2 一次扫完** | **Skv≤1024 全场景** | **0.80x/0.922x ★** |

- 判据：Skv≤1024（sink 全场景 Skv=128）→ **S1 单遍**（AIV 多遍变单遍，省反复 GM 往返）。
- **S1 vs s1s2 关键差异**：S1 不开 enableL1Reuse、splitFactor 不 ×2（s1s2 对 D∈[64,128] 开 L1R + splitFactor×2）。手搓 S1：host aicRatio=1、`splitFactor=CeilDiv(totalSize,coreNum)` 不翻倍。
- B 变体（alignedS2≤128 且 BNG·S1·S2·2B≤128KB 的极小 case）整 S2 单 softmax pass，仅小 case 备用。

### B. kernel 手搓相位（按此顺序写，不复制源码）

手搓一个 S1 单遍 fa-sink kernel = 写以下相位。s1×s2 双层算法细节引用 §12.10 C/D/E；本节只标 **S1 单遍与 sink 的差异**。

1. **入口（自写 ~30 行）**：`__global__ __mix__(1,2)` → `TPipe`+`SetMaskNorm`+`SfaCopyTiling(GM→结构体)`+`REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, op.bmm2, op.bmm2Nz)`+`if ASCEND_IS_AIC return`（AIC 只 REGIST）+ AIV `op.Init(...)`+`op.Process()`。14 参：`query,key,value,sink,softmaxMax,softmaxSum,attentionOut,__kfc_workspace__,tiling`。kfc 5 项见 §12.8 C / §12.9 F#2。
2. **AIC 侧（cube）**：只 `REGIST_MATMUL_OBJ` 后 return；matmul 实际由 AIV 侧 `op.Process` 内 `IterateAll<false>`+`WaitIterateAll` 驱动（异步，**禁手动 `while(Iterate<true>)`**，否则 cube 写 GM 未全局可见→非确定性错）。
3. **AIV 主循环（S1 单遍）**：按 multiCoreInnerIdx 遍历本核 task 段（每 task=(bh,128 行 query)）。**S1 单遍：s2 一次扫完（不做 s1×s2 双层）**；s1 块内三深流水（`taskId%3`）：QK(task) 与 softmax(task-2) 与 PV(task-1) 重叠（见 §12.10 C）。事件链：`IterateBmm1→MTE1_MTE2`、`softmax→MTE3_MTE2`、`PV 用 WaitIterateAll`。
4. **bmm1 QK（cube）**：A=Q[row=rowNum,K=D]、B=K[col=Skv,K=D]、C=mm1Res[UB fp32]（**S/P 不落 GM**）；scale 在 softmax 前 `Muls`。tiling：M=min(s1Base,S1)、N=s2、K=D；fixsplit M=s1BB(128)、N=s2BB(128)。
5. **online softmax（AIV，含 sink 落点）**：读 mm1Res → 逐行 max → **sink：`m_f=max(m, sinkGm[headIdx])`**（sink=[H] fp32，scale 后尺度）→ exp → 逐行 sum（**`l_f = l·exp(m−m_f) + exp(sink−m_f)`**）→ P cast fp16 写 stage1Res(UB) 供 bmm2 作 A。详见 §12.10 E / §12.6 sink 递推。
6. **bmm2 PV（cube）**：A=P[UB,行=rowNum,K=Skv]（**直接指 softmax 的 UB 输出，不落 GM**）、B=V、C=mm2Res[fp32]；fixsplit M=s1BB、N=min(AlignDown(L0C/(s1BB*4),16),dBB)；shareL1=1MB、shareL0C=256KB。
7. **divide/rescale out（AIV）**：读 mm2Res+l → Brcb 广播 gl → `Div` → cast fp16 → 写 O GM；smax/ssum 同步导出（[BH*Sq,8] fp32 槽 0）。

### C. UB 布局 + 硬件事件同步（手搓要复刻的结构）

- **UB 缓冲（AIV 私有，双缓冲）**：mm1Res[2]（bmm1 结果 fp32 ping/pong）、stage1Res[2]（softmax 后 P fp16，供 bmm2 A）、mm2Res[2]（bmm2 结果 fp32）、vec2Res[2]（divide 中间）、softmaxMax/Sum 缓存。per-core workspace 基址 = workspace + blockIdx·totalOffset，`totalOffset = mmNRatioOffset·bmm1AndVec1Ratio + mm2Offset·2·GM_DOUBLE_BUFFER`。
- **硬件事件（PIPE 间 Set/Wait，非跨核 flag）**：`MTE1_MTE2`（L1→UB 排空）、`MTE3_MTE2`（softmax 写出后 S 可见）、`MTE2_V`（MTE2 装载完 V 可读）、`V_MTE3`（V 计算完可 GM 存）、bmm2 `WaitIterateAll`（cube 写 GM 全局可见）。**每事件 id 每 tile 恰好 1 Set+1 Wait**（§12.5 坑：Set 过剩→aicore trap、Wait 过剩→死锁）。
- 任务切分 / 主循环 / matmul kfc 用法详见 §12.10 B/C/D。
- 14 参顺序（host 一致）：`query, key, value, sink, softmaxMax, softmaxSum, attentionOut, __kfc_workspace__, tiling`。
- sink = [H] fp32 设备指针（scale 后尺度直传，值如 1.5）；无 sink 传合法占位。
- kfc 5 项约定同 §12.8 C（`__kfc_workspace__` 第 8 参 + `GetSysWorkSpacePtr` + 每次 launch 前 `aclrtMemsetAsync(dWs,16MB,0)` + AIC 只 REGIST + 异步 `IterateAll<false>`+`WaitIterateAll`）。

### D. host tiling：SfaBuildTilingS1（手搓推导，S1 单遍）

host 自写 `SfaBuildTilingS1(B,H,S1,S2,D,hasSink,scale,aivNum,actualS2,pack)`，按下方公式推导各字段（tiling 全部自算）：

```
s1BB = min(128, AlignUp(S1,16));  s2BB = min(128, AlignUp(S2,16));  dBB = min(128, AlignUp(D,16))
nRatioMax = 4（默认）；alignedS2<128 → nRatioMax=1
NZND 入口（s2%64!=0 || s2==64 || (s2>=704 && s2%64==0 && s2%128!=0 && B*H*S1>20480)）且 D<=256 → nRatioMax=4
# S1 轴 N:1 grow/shrink（workspaceLimit=131072=8*128*128）
while s1Ratio<nRatioMax && TotalSize(s1Ratio)>(s1Ratio-1)*aiv/s1Ratio
      && s1BB*s1Ratio<alignedS1 && s1Ratio*s1BB*s2BB<=workspaceLimit:  s1Ratio++
while s1Ratio>1 && (TotalSize<=(s1Ratio-1)*aiv/s1Ratio || s1BB*s1Ratio>alignedS1
                    || s1Ratio*s1BB*s2BB>workspaceLimit):  s1Ratio--
s1Base=s1BB*s1Ratio; s1Outer=CeilDiv(CeilDiv(S1,s1BB),s1Ratio); nRatio=1（S1 kernel 不消费）
# coreParams: s1BaseSize=s1Base, s1BaseTailSize=TailSize(S1,s1Base), s1OuterSize=s1Outer,
#   s2BaseSize=s2BB, s2OuterSize=CeilDiv(s2,s2BB), dBaseSize=dBB, bOuter=B, n2Outer=H, gOuter=1...
# multiCoreParams: totalSize=B*H*s1Outer; coreNum=min(totalSize,aiv); splitFactor=CeilDiv(totalSize,coreNum)  （★S1 无 aicRatio×2，与 s1s2 L1R 不同）
# SetBmm1TilingInput: SetShape(min(s1Base,S1), s2, D); SetOrgShape(S1,s2,D,D); SetFixSplit(s1BB,s2BB)
# SetBmm2TilingInput: SetShape(min(s1Base,S1), D, s2); SetOrgShape(S1,D,s2,D);
#   SetFixSplit(s1BB, min(AlignDown(L0C/(s1BB*4),16), dBB))   # L0C=256KB
# bmm1/bmm2TilingData.shareL1Size=L1_SIZE(1MB), shareL0CSize=256KB
```
- **★ 与 s1s2 的关键差异**：S1 **不开 enableL1Reuse、splitFactor 不 ×2**（s1s2 对 D∈[64,128] 开 L1R + splitFactor×2，见 §12.8 B）。S1 路径 `pack.l1Reuse=false`，aicRatio=1。
- **逐字段核对法**：设 `SFA_DEBUG_TILING=1` dump host tiling，按下式自验（s1BB/s2BB/dBB/nRatio/s1Base/s1Outer/coreNum/splitFactor/s1Vec2BaseSize/bmm1·2 FixSplit）。Skv=128 下 s2BB==alignedS2==128，workspace-limit 检查一致；其它 s2 注意应按 `alignedS2`（AlignUp(s2,16)）判、勿误用 `s2BB`（仅 s2==128 时相等）。
- **结论**：S1 tiling 按此式推导即可，无字段偏差可修（排查已穷尽，剩余差距在工具链见 E）。

### E. 优化排查方法论（已穷尽算法/tiling 层；根因=工具链 codegen）

对 fa-sink 做了三层排查，**结论：kernel 算法等价 + tiling 公式正确 → 剩余差距在 CANN 工具链 codegen，非算法/tiling 可修**。复刻此排查可避免在源码上白费功夫：

| 排查项 | 方法 | 结论 |
|---|---|---|
| kernel 算法核对 | 逐相位对照 §12.10 单核融合算法（S1 单遍 vs s1s2 双层） | 算法等价，差距不在算法逻辑 |
| S1 tiling 公式（T4） | `SfaBuildTilingS1` 各字段按 §D 公式自验 | 公式正确，无字段偏差可修 |
| 冗余 `PipeBarrier<PIPE_V>` 移除（AIV 热路径） | 同会话同设备 back-to-back **公平 A/B**（baseline vs 移除） | **delta ±2-7%，无一致方向，落在 6% 测量噪声内**（ref 时间两次紧邻运行漂移 ~6%）→ **无可靠收益，已回退** |

**★ 公平 A/B 规程**（单 case A/B 不可信，协同租户占卡 86-98%）：
1. 同一会话、同一设备、紧邻 back-to-back 跑 baseline .so 与 candidate .so（用两个 harness 目录指向不同 .so）。
2. ref 时间（标杆算子，两次运行本应恒定）的漂移幅度 = 噪声地板；delta < 该漂移 → 判无收益。
3. warmup/repeats 两版必须相同；选样与加速比必须同一轮数据。
- **根因定位**：Cube(MAC) 时间与官方一致，差距全在 AIV/vector 侧；AIV 算法等价 → 差距来自本侧自建 .so（CANN-9.1.0 + `-O2`）vs 标杆 op 包专用编译流水线的 codegen 差异。**要继续缩小需换工具链/编译选项，非改算法/源码**。

### F. 性能对比方法论（单变量序列长扫描，可复现）

**case 生成**（cannbot skill `/aog-input-gen-builder` 的 SCHEMA：`tensor_inputs`/`scalar_inputs`/`shape_derive`/`invariant`）：
- 主扫因子 = seq_len(Sq)；B/H/D/Skv 固定（如 B=1,H=16,D=128,Skv=128），仅 Sq 变。
- Sq 取代表性序列长（如 2048→458752，覆盖短/中/长/超长）。
- **内存 invariant**：`mem = 2*(B·H·Sq·D + B·H·Skv·D)*2 bytes ≤ 预算`。单 batch H=16 时 Sq=458752 仅 3.50GiB（q+out），64GB 卡无 OOM。**B·H 随 Sq 增大而缩小**（超长走 B=1）。

**对比规程**（规避标杆 sink op 同进程随机 161001，handoff pitfall #1）：
1. **逐 case 子进程隔离**：每 case 写一个 1-case model.json，单独跑一次 aog_perf_eval，解析 pr.json。
2. case 目录（`--cases` 的父目录）**只能有 model.json 一个 .json**（见 G 坑#1）；pr.json/pr.html 写到独立目录。
3. 标杆 = 官方 **aclnn**（`torch_npu.npu_fusion_attention(...,sink=[H])` → `aclnnFlashAttentionScoreV3`）。注意 sink 参数只有 aclnn 原生支持，**ATB 的 FA/PagedAttention 不暴露 sink 入参**（语义不可比对，勿用 ATB 做 sink 标杆）。
4. 加速比 = npu标杆/本侧；本组平均 = 各 case 加速比算术平均（与示例同口径）。
5. 设备用 `ASCEND_RT_VISIBLE_DEVICES=<空闲卡>`；AICore 0% 的卡才可信。

### G. 新增坑清单（fa-sink S1 专有，按优先级）

1. **★ case 目录 .json 污染**（最隐蔽）：`aog_perf_eval` 的 `_get_input_groups_from_json` 扫描 `--cases` **父目录**，按**字母序**取第一个 `.json` 当 case 文件读（逐行解析）。若 case 目录里有 `manifest.json`（字母序在 model.json 前）或 `pr.json`，会被当成 case 读 → `JSONDecodeError: line 2 column 1`。**case 文件放独立目录，只放 model.json（+model.py/model_new_ascendc.py）**；manifest/pr.json/pr.html 写到别的目录。
2. **npu 标杆是 aclnn 非 ATB**：fa-sink 的 sink 语义只有 `aclnnFlashAttentionScoreV3`（`npu_fusion_attention` + `sink=`）支持；ATB FA/PagedAttention（BTHKvD 布局）无 sink 入参，只能做"无 sink 纯 FA"性能参照，精度不可比。报告须注明标杆算子名（profiler `operators` 字段）。
3. **算法变体选错（s1s2 vs S1）**：Skv≤1024 用 s1s2 双层 → 0.58x（AIV 多遍）；必须用 S1 单遍 → 0.80x。判据：sink/none + Skv≤1024 → S1 单遍（见 A）。
4. **S1 不开 L1Reuse / splitFactor 不 ×2**：与 s1s2（D∈[64,128] 开 L1R + splitFactor×2）相反。S1 路径 `l1Reuse=false, aicRatio=1`，splitFactor=CeilDiv(totalSize,coreNum) 不翻倍。混用 s1s2 的 L1R 配对规则到 S1 → 错。
5. **barrier 移除微优化无效**：AIV 热路径冗余 `PipeBarrier<PIPE_V>`（被相邻 SetFlag/WaitFlag 覆盖）移除后公平 A/B 无收益（编译器已处理），勿为此改算法（改了无收益反而引入风险）。剩余差距是工具链 codegen。
6. **长序列内存**：B·H·Sq·D·2 随 Sq 线性增长；Sq=458752+H=16 单 batch 3.5GiB OK，但多 batch 多头大 Sq 会爆（B=16·H=64·Sq=200k·D=128 = 9.8GiB q+out，加上标兵翻倍）。超长序列强制 B=1。

### H. fa-sink 终态实测（2026-08-26，device 910B3，profiler V5 device 侧）

单变量扫描（B=1,H=16,D=128,Skv=128，Sq 2048→458752，逐 case 子进程隔离，标杆 aclnnFlashAttentionScoreV3）：

| Sq | 本侧(μs) | npu标杆(μs) | 加速比 |
|---|---|---|---|
| 2048 | 71.9 | 50.7 | 0.705 |
| 8192 | 148.0 | 126.7 | 0.856 |
| 49152 | 660.8 | 627.8 | 0.950 |
| 131072 | 1669.4 | 1618.1 | 0.969 |
| 262144 | 3254.2 | 3173.9 | 0.975 |
| 393216 | 4905.7 | 4888.0 | **0.996** |
| 458752 | 5695.7 | 5535.0 | 0.972 |
| **本组平均(20例)** | 1639.3 | 1586.9 | **0.922** |

- 趋势：序列越长越接近标杆（Sq≥49152 起 0.94-0.996x，Sq=393216 达 0.996x 基本持平）；短序列 Sq=2048 0.705x 最弱（单 batch 小 Sq 并行度不足、launch 开销占比高，固有短板）。
- 混合口径（22 例 S/M/L/XL，B/H/D 多变，Sq 至 200000）均值 0.800x，与交接文档 S1 记忆值 0.795x 一致。
- 精度：30/30 PASS（含 Sq=196608 全部超长序列），S1 默认模板。

### I. 手搓产物自检清单（S1 单遍版）


---


### C. 三算子手搓对照
| 特性 | 叠加位置 | 额外 flag/barrier | 已知坑 |
|---|---|---|---|
| sink | softmax m/l 递推 | 无 | m 读 GM 槽 |
| varlen | softmax 掩码行 | 无 | mask 槽预建 |
| q8 | AIV dequant 前置相位 | dqReady(id=4) 配对 | DataCopy→Cast 必须 PIPE_ALL；scale 用 1/s |
| 叠加 | 三者互不冲突（mask/sink 同 softmax、q8 在 GEMM 前） | dqReady 一次 | 四合一 +4.3% 开销 |

### D. 手搓自检（varlen/q8 版）
- [ ] varlen：mask 槽预建（非逐 chunk SetValue）；batch 偏移与 tiling s2 配套
- [ ] q8：DataCopy→Cast 有 PIPE_ALL；scale 传 1/s；fp16-only
- [ ] 107 例精度（含 Sq=196608、三特性）
- [ ] 对标杆 varlen≥0.9 / q8≥1.0 / sink≥0.6（S1 模板实测 1.12/1.50/0.81）

## 12.13 kfc 单 TU 直调 fa-sink 完整配方（第三轮纯 skill 实证，2026-08-27，BNSD 自注意力任意序列长）

> 本节是 **fa-sink / 标准 FA 在 kfc matmul 单 TU 直调路径上的终态配方**（第三轮纯 skill 生成实证：
> 20 case 序列长分桶套件 Sq=1024→458752，vs `npu_fusion_attention(sink=)` geomean **0.693**（gate≥0.6 PASS），
> 长序列 64k+ 稳定 0.70-0.72×，Sq=1024 反超 1.57×；精度契约 S%1024==0 全 PASS（smax/ssum 与 golden 位级一致 4e-5））。
> 适用于 BNSD 自注意力、D=128、fp16；对 §12.11 的补充与修正（s2 任意长度按 chunk 流式，不再限 Skv≤1024）。
> 环境：910B3 / dav-2201 / CANN 9.1.0，全部实测。

### A. 入口与 launch 管线（支撑设施，缺一不可）

1. **入口形态**：
```cpp
extern "C" __global__ __mix__(1, 2) void FaKernel(
    GM_ADDR query, ..., GM_ADDR attentionOut,
    __kfc_workspace__ __gm__ uint8_t *workspace,   // 第 8 参：属性会给值加 +4GB 设备地址窗口
    GM_ADDR userWsParam,                            // ★user ws 必须独立普通参数（host 传 dWs+16MB）
    GM_ADDR tiling)
{
    TilingData t; CopyTiling(t, tiling);
    Kernel<half> kernel;  AscendC::TPipe pipe;  AscendC::SetMaskNorm();
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), kernel.mm1, &t.bmm1Tiling,
                      kernel.mm2, &t.bmm2Tiling);   // ★双对象注册合法
    if ASCEND_IS_AIC { return; }                    // AIC 只 REGIST 后 return
    kernel.Init(...);  kernel.Process();  kernel.mm1.End();  kernel.mm2.End();
}
```
   - **★+4GB 窗口坑（实测 probe 证实）**：`GetSysWorkSpacePtr() == wsParam − 4GB`。从 workspace 参数做指针加法派生 userWs 会写到 4GB 外（host 读不到、kernel 内自洽、极难排查）。userWs 永远用独立普通 `GM_ADDR` 参数传。
   - workspace 单 buffer：`[0,16MB) kfc/sys + [16MB,+) per-worker user`；host launch 前 `aclrtMemsetAsync` **只需清前 64KB**（kfc 消息队列区）——清全量（几十 MB）= **25ms/launch 的隐形大头**（aclrtMemsetAsync 不走 DMA 快路径）。
   - launch：`Kernel<<<aicCoreNum, nullptr, stream>>>(...)`（中间槽必须 nullptr，传实指针即 507015）。
2. **host tiling 必须用框架计算器** `matmul_tiling::MatmulApiTiling`：
   `SetAType(GM,ND,DT_FLOAT16,false) / SetBType(...,transFlag) / SetCType(GM,ND,DT_FLOAT) / SetShape / SetOrgShape / SetBias(false) / SetBufferSpace() / GetTiling(t)`，
   GetTiling 后补 `shareMode=0; shareL1Size=1MB; shareL0CSize=256KB; depthA1=2; depthB1=2`。**手填 TCubeTiling 字段会挂死**。
3. **B 转置必须运行时 flag**：`SetTensorB(gm, true)`（类型为非转置 `MatmulType<GM,ND,T>`）。**类型级转置（`MatmulType<...,true>`）在该路径产出错值**（C 落盘但数值错）。
4. blockDim=aic 核数；AIV `GetBlockIdx()∈[0,2×)`、**双子核都参与计算**（worker=min(40,totalTasks)，连续段切分，ws 区按 blockIdx 索引）；单子核提前 return 会导致该核 kfc 流程异常。
5. UB 预算 ~100KB/子核；向量行块 ≥8（fp32 向量最小粒度，count<8 的 Duplicate 会 trap）；device lambda 不被 ASC 编译器支持（写私有成员函数）。

### B. GEMM 驱动（每 chunk 各一个 IterateAll 问题——唯一又快又对的形态）

每 chunk（S2_CHUNK=1024 列）：
```cpp
// bmm1: S = Q·Kᵀ（单问题，N-tile 内部迭代）
mm1.SetOrgShape(curM, cols, D);
mm1.SetTensorA(qTensor);
mm1.SetTensorB(kPlane[c0*D], true);
if (尾) mm1.SetTail(curM<128?curM:-1, cols<1024?cols:-1);
mm1.template IterateAll<false>(sDst, false, false, true);   // 异步
... 消费点:  mm1.WaitIterateAll(); mm1.End();

// bmm2: O_acc += P·V（enAtomic=1 原子累加直入 O_acc，省整个 o2 GM 往返）
mm2.SetOrgShape(curM, D, cols, D);      // ★4 参版：orgKb=D 是 B(V) 行 stride，3 参版会把 V 按 K stride 读
mm2.SetTensorA(pWs);
mm2.SetTensorB(vPlane[c0*D], false);
if (curM<128) mm2.SetTail(curM, -1, -1);
mm2.template IterateAll<false>(oAccWs, 1, false, true);     // enAtomic=1
```
**实测否决的替代形态**（全部数值错或慢 4×）：逐 tile `while(Iterate<true>){GetTensorC}`（慢 4×）；`Iterate<false>(k>0)` 异步 enPartialSum（L1 staging 竞态，~5% 元素错）；K>baseK 的单问题（L1 超预算，~5% 元素错）。

**尾块（cols<S2_CHUNK）布局契约**：bmm1 的 C 与 bmm2 的 A 都按 **row-stride=cols packed**（非 tiling N）；**K/N 非 128 倍数的 GEMM 尾数值必坏** → 标准解法：**cols pad 到 128 倍数 + softmax 对 pad 列写 NEG_LARGE 掩码 + host 给 K/V buffer 尾部补 128 行零**（pad 列 exp(−inf)=0，P=0，V pad 无贡献）。

### C. softmax 正确用法（SoftmaxFlashV2 fp32 无 bug——“fp32 per-row max 坏”是用法错误）

```cpp
// 状态：[rows×8] fp32 持久 max/sum（跨 chunk 递推）+ [rows×64] expMax + 32KB tmp
Duplicate(ubMaxSt, sink[h], S1_BLOCK*8);          // sink 播种：m0=sink, l0=1
Duplicate(ubSumSt, hasSink?1.f:0.f, S1_BLOCK*8);
...
DataCopy(ubS, sCur[r*stride], rows*cols);  PipeBarrier<PIPE_ALL>();   // MTE2→V
Muls(ubS, ubS, scale, ROW_BLK*cols);
ubS.SetShapeInfo(ShapeInfo(2, shapeS, 2, shapeS, DataFormat::ND));     // ★2D 形状必设
maxBlk.SetShapeInfo(ShapeInfo(2, shapeMS, DataFormat::ND));            // [rows,8]
SoftMaxTiling smTil = SoftMaxFlashV2TilingFunc(shape, 4, 4, tmpBytes,
                                               /*isUpdate=*/false, /*basic=*/true, ...);  // ★tiling 传 false
SoftmaxFlashV2<float, /*isUpdate=*/true, /*isReuse=*/true, /*basic=*/true, false>(
    ubS, sumBlk, maxBlk, ubS, ubExp, sumBlk, maxBlk, ubTmp, smTil);   // ★fp32 in-place(dst=src)，call 传 true
```
- α 不从 expMax tensor 读（布局不可靠），**自算**：`m_prev/m_new 取 maxBlk 槽 0`（`GetValue(i*8)`）→ `Sub→Exp→GetValue`，α 按 chunk 奇偶双缓冲。
- **16 行/调用**（调用数减半），P 用 [8×cols] 半块两次 Cast→fp16。
- 手写 ReduceMax/ReduceSum 逐行 softmax（fp32 误判 bug 后的替代品）**慢 ~430×，禁用**。

### D. 流水与同步顺序（chunk 级，depth-2 简化版）

```
每 chunk c： prefetch bmm1(c+1)异步 → wait bmm1(c)+End → softmax(c)（写 pWs[c%2]，算 α）
            → wait bmm2(c-1)+End → prescale: O_acc*=α(c) → submit bmm2(c) 原子累加
```
- **★原子累加竞态（非确定性 NaN 的根因）**：bmm2 的 enAtomic C 写是 16 列组 RMW；prescale 的 O_acc 写回若与之并发 → 随机 16 列组 NaN。**prescale 必须严格在上一个 chunk 的 `WaitIterateAll()+End()` 之后、本 chunk 提交之前**。
- S/P 双缓冲（GM per-worker ws，chunk 奇偶两个半区）；ROW_BLK=16、per-worker ws ≈1.2MB。
- finalize：`O=O_acc·(1/l)→cast fp16→out`；smax/ssum 槽 0 按 8 行块批量 SetValue+DataCopy（W18 安全形态）。

### E. 实测基线（910B3 争用卡，标杆=同卡同窗口背靠背回放均摊）

| Sq | 本侧(μs) | 标杆(μs) | 比例 | | Sq | 本侧 | 标杆 | 比例 |
|---|---|---|---|---|---|---|---|---|
| 1024 | 247 | 387 | **1.57** | | 65536 | 452.8m | 315.0m | 0.70 |
| 4096 | 1981 | 1014 | 0.51 | | 131072 | 1.81s | 1.29s | 0.71 |
| 16384 | 28.6m | 15.9m | 0.56 | | 458752 | 22.7s | 16.5s | **0.72** |

geomean(20)=**0.693**；趋势=序列越长越接近标杆（≥64k 稳定 0.70-0.72）。已知边界：S%1024≠0 的尾块在原子路径下仍有竞态 NaN（契约先锁 %1024；根治=o2 双缓冲+显式 merge）；小序列 2k-8k 是 0.51-0.66×（标杆 launch 占优，可做 chunk 自适应）。

### F0. 逐步生成流程（从零到可运行，按此顺序写；每步产出即下一步输入）

1. **写 `kernel_common.hpp`**：常量（S1_BLOCK=128 / S2_CHUNK=1024 / ROW_BLK=16 / AUX_SLOTS=8 / KFC_WS_BYTES=16MB / NEG_LARGE=-3e38）+ `TilingData` POD（B/H/S/D/hasSink/scale/totalTasks/workerNum/softmaxTmpBytes=32KB/stage 调试位 + 两个 `TCubeTiling` 成员）+ LCG 常数（host/verify 共享）+ `CopyTiling`（GM→栈按 int32 逐字拷贝）。
2. **写 host tiling 段**（§A-2 公式逐字段）：两个 `MatmulApiTiling`（bmm1: fp16×fp16→fp32, transB=true, SetShape(128, 1024, D)；bmm2: fp16×fp16→fp32, SetShape(128, D, 1024)）→ `GetTiling` 后补 shareMode/L1/L0C/depth。`FA_SINK_DEBUG_TILING` 打印自验。
3. **写 kernel 类骨架**：成员 = 双 `matmul::Matmul` 对象 + 7 个 `TBuf<VECCALC>`（S 64KB / P16 32KB / tmp 32KB / state ~17KB / O 16KB / aux）+ GM 张量成员 + per-worker ws 区张量（sWs/pWs/o2Ws/oAccWs，按 blockIdx×region 偏移）。
4. **写 entry**（§A-1 的 11 参形态 + REGIST + AIC return + stage 守卫）。
5. **写 `Init`**：GM SetGlobalBuffer 全套 + InitBuffer 全套（尺寸=第 3 步注释）。
6. **写 `Process`/`RunTask` 空转**（任务段切分 + 状态播种 + O_acc 清零），跑 stage -1/-2/0 验证管线（sync done + ws canary）。
7. **写 bmm1 相位**（§B 代码）：先同步 IterateAll 验证 S 数值（dump 对 golden），再加预取双缓冲。