# FA kernel 复合特性叠加与避坑（sink / varlen / q8 / causal / paged / 测量）

> **分册说明**：本册是 [fa-kernel-handcraft.md](fa-kernel-handcraft.md) §12 系列的**特性叠加部分**（原手册拆分，§ 编号不变）：
> §12.5.3 器件级 API 墙、§12.6 复合特性总纲、§12.7 测量与泛化坑、§12.12 varlen/q8 叠加、§12.14 causal/paged 叠加。
> 算子配方见 [fa-kernel-recipes.md](fa-kernel-recipes.md)；骨架见手册 §1-11。

> **简称**：本文档称「特性册」；[fa-kernel-recipes.md](fa-kernel-recipes.md) 称「配方册」。

---

### 12.5.3 器件级 API 墙与 AIV 调用风暴（8 轮优化实测，dav-2201/CANN 9.1，全部设备端二分实证）

> 手搓 kernel 优化到深处必撞**器件级 API 的"文档与实测不符"**。以下结论是 8 轮优化换来的，**文档不可信、以实测为准**：

**① AIV 向量 API 调用风暴（假"标量瓶颈"的真凶）**：
profiler 显示 `aiv_scalar_ratio≈0.66` 时，逐项排查发现大头**不是标量数据搬运，而是逐次向量 API 调用的发射开销**（每 tile 每子核 ~330 次调用 × ~100ns/次 ≈ 25μs/tile）。优化方向=**收敛调用次数**：整行 `ReduceMax(count=128)`（可用）、repeat-params `src1RepStride=0` 行广播（varlen mask 用，可用）、R（行批）加倍、逐行 1 宽 Exp→批量宽。**每 chunk 向量调用次数是第一指标**。

**② 器件级 API 墙（全部二分实证，文档不可信）**：
| API | 实测结论 |
|---|---|
| `Brcb` 行广播 | **输出布局与头文件注释不符**（dav-2201/CANN 9.1）——3 种用法全数值错（广播矩阵成垃圾）。用前必须器件探针 dump UB 验证 |
| `ReduceSum` 整行 count=128 | **静默错误**（vcadd+acc-register 路径单 repeat 元素上限 64）——smax 过但 ssum_rel≈0.75，无 trap。同 count 的 `ReduceMax` 正常（不同指令路径） |
| `ReduceRepeat`(vcmax/vcadd whole-mode) | 公开低阶接口存在但**无公开 dst 布局文档**，盲用 0/36 |
| 可用替代 | 整行 `ReduceMax(count=128)`、repeat-params `src1RepStride=0` 行广播、逐行合并保留 |

**③ HardEvent Set/Wait 严格配对**：每个事件 id **每 tile 恰好 1 Set + 1 Wait**。Set 过剩（在 pending 旗上重复 Set）→ **aicore trap（timeout）**；Wait 过剩 → 死锁。改预取/驻留复用逻辑后必须逐 id 推演账目平衡。

**④ 手搓算法选择**（8 轮优化 + S1 单遍实测）：FAInfer 跨核 GM staging 手搓对标杆 ~0.5x（器件墙）；但 **S1 单遍组织手搓（§12.11，单核融合 S/P 不落 GM）达 0.80x**——换算法变体可突破 0.5x 天花板。FA 变体手搓路径见 §12.11；§12.8/12.9 的"复制官方源码"移植捷径已弃用（非手搓、不满足 skill 直接生成）。


---

## 12.6 复合特性手搓扩展法：在自写 FA kernel 上叠 sink/varlen/q8（总纲）

> 需求"复合功能 FA 算子（不同量化/sink/不同 mask）" = **自写一个基础 FA kernel，三特性作为 kernel 内相位的手搓叠加**（不依赖任何基座/外部库）。基础 FA 骨架见 §12.1-12.4，单遍组织见 §12.11，三特性 kernel 内落点见 §12.5.10 与本节 B，完整手搓流程见 §12.11（sink）/ §12.12（varlen/q8）。实测三特性叠加近乎免费（sink +5%、varlen ±0%、q8 +0.3%，四合一 +4.3%）。

**A. 布局适配（BNSD → batch=B·H 重解释，零拷贝）**
- BNSD `[B,H,S,D]` 在 host 侧重解释为 **batch=B·H, numHeads=1, kvHeads=1**：q/o 的 `[BH,S,D]` 内存序与单头 FA 的 `[batch,S,D]` 逐位相同 → **零拷贝**（这是"复合特性不加布局开销"的关键）。
- k/v 走 `[BH,Skv,D]` 连续布局（blockTable identity 即可，Skv%128==0 时）；v 提供免转置入口（v 直接 [B,H,Skv,D]），避免 per-launch 标量 gather 转置（那是 0.19x 的坑）。
- host per-launch 的 tiling/blockTable/actSeq 等 H2D memcpy 是隐形大头：上下文不变时缓存，重复 launch +20-40%。

**B. 三特性 kernel 内落点（手搓，按此在自写 kernel 相位里加）**
| 特性 | 数学 | kernel 内落点（手搓） | 关键点 |
|---|---|---|---|
| varlen（前缀 mask） | k≥kvLen[b] → -3e38 | softmax 相位：`gLen.GetValue` 标量 + K 次 SetValue 构造掩码行 + Add | mask 构造是标量热点，长序列时预建 per-b mask 槽缓存（§12.12） |
| sink（StreamingLLM 虚拟列） | m'=max(m,sink)；l'+=exp(sink−m') | 见 §12.6-E 完整向量算子链 | sink 值=scale 后尺度直传；O 归一化自动正确 |
| q8（W8A8） | int8×scale → fp16 → 正常 GEMM | 新增 dequant 相位：AIV 前置把本 tile 操作数切片 int8→cast→scale→fp16；配对 flag 通知 AIC | dequant 是纯带宽（元素数×6B）；DataCopy→Cast 间必须 PipeBarrier（W13）；q8 仅 fp16 |

**C. aux 输出（smax/ssum [BH·S,8] fp32）**：gm/gl 经 **Brcb 广播 + slot0 掩码(0x0101…)Add + 连续 DataCopy** 写槽 0；**host 必须做 null 回退**（内部缓冲），否则调用方传 null → kernel 写 0 地址 → 507057 设备错。

**D. AIV 改动纪律**：双子核并发共享 UB——自加 scratch/event flag 必须按 subIdx 离散 + 用 `PipeBarrier<PIPE_ALL>` 替代 flag 舞蹈（否则结果非确定性错）；ASC 编译器不吃 device lambda（写私有成员函数）。

**E. sink 校正的完整向量算子链（RescaleO 的 `isLastStackTile` 分支，已验证形态）**——全部向量 API、`rep = ⌈rows/64⌉`、**每个算子后 `PipeBarrier<PIPE_V>()`**：
```cpp
// 数学: m_f = max(m_run, sink);  l_f = l·exp(m_run−m_f) + exp(sink−m_f)
DataCopy(scSave, gmUbTensor, DataCopyParams(1, rowsRound/8, 0, 0));   // 备份 m_run（★m 用 GM 槽值，UB 槽此刻陈旧）
Maxs(gmUbTensor, gmUbTensor, sinkValue, 0, rep, UnaryRepeatParams(1,1,8,8));       // m_f
Sub(scExp, scSave, gmUbTensor, 0, rep, BinaryRepeatParams(1,1,1,8,8,8)); Exp(scExp, scExp, ...);
Mul(glUbTensor, glUbTensor, scExp, 0, rep, ...);                       // l·exp(m_run−m_f)
Adds(scSave, scSave, -sinkValue, ...); Maxs(scSave, scSave, 0.0f, ...);
Muls(scSave, scSave, -1.0f, ...); Exp(scSave, scSave, ...);           // exp(sink−m_f)
Add(glUbTensor, glUbTensor, scSave, 0, rep, ...);
```
入口相应扩参（14→19）：`..., tiling, smax, ssum, float sinkValue, uint32_t hasSink, uint32_t hasAux`，Params 结构体透传到 RescaleO 构造。实测开销：sink +5%、aux 条件导出近乎免费。

## 12.7 测量与泛化验证的坑（本轮全部踩过，均实测复现）

1. **MIX_AIC 过滤坑**：`npu_fusion_attention` 系 kernel（FlashAttentionScoreV3）与 FAInfer 主 kernel 的 `Accelerator Core=MIX_AIC`。profiler 解析若只滤 `AI_VECTOR_CORE/AI_CORE` 会**整漏融合 kernel**——sink 曾测出 0.01x 假象、q8 曾虚高 2.94x（修复后真实 ~1x）。过滤集必须含 `MIX_AIC`。
2. **aog_perf_eval 的 get_input_groups 优先坑**：driver 优先用 model.py 的 `get_input_groups()`，缺失才解析 model.json——旧 harness 拷贝会**静默测旧 shape**。用 model.json 驱动必须删掉该函数。`--cases` 的父目录会被扫全部 *.json（case 文件放独立目录）。
3. **泛化分桶口径**：用户口径"运行时长 10%/20%/40%/30%"以**被测算子（catlass）耗时**为准分桶选样；标杆耗时分桶是另一口径（勿混）。多算子交付时**每算子独立选 1/2/4/3**（每算子 10 例），合计 30 例自动满足全局 10/20/40/30。
4. **桶窗口会因测量工具修复而移位**：修 bug 前后 asc 时间差 5x → 桶边界 case 换档。补池要补**桶中心锚点**（按修复后数据重标定），边界 case 不可靠。
5. **npu 标杆的 coverage 限制**（npu_fusion_attention，CANN 9.1/torch_npu 2.6）：`sink` 参数 head>16 报 161001（tiling `shapeSize should be 8` 类限制）；varlen 只有 TND 形态（BNSD+长度数组需 host 转 TND，语义等价已验证 matched=1.0）。官方跑不了的 case 标杆回退 torch 拼接并注明。
6. **首次调用/争用异常值**：极小 case 的标杆 ref 偶现 600μs 级尖峰（首调/争用），report 时标注；协同租户争用下 q8（带宽型）轮间可摆 0.9↔4x，**选样判定与加速比必须同一轮数据**。

**12.6 补充（2026-08-21 sink 对标杆差距实测）**：v2.1 微优化（aux 条件导出 + H2D 一次性缓存）后，sink 对标杆 Duration 口径仅 0.440x→0.446x——**差距是结构性的**（FAInfer 示例 epilogue vs 生产版 FlashAttentionScoreV3：我们 aic_mac 12%、AIV scalar 37%，官方有效算力 ~85 TFLOPS vs 我们 ~32）。H2D 缓存只改善墙钟（对拼接 2.18→2.49x）。另：标杆 sink op 在 H=16 上**同进程内随机 161001**（隔离子进程可跑、连续调用偶发 tiling 失败），官方对比需逐 case 子进程隔离。要缩小 Duration 差距必须重写示例 kernel 的 rescale-O epilogue（在线 softmax 多遍 GM 往返是根因）。

**12.7 补（2026-08-25）——aog_perf_eval 不读 tensor value 坑（lens 垃圾值假象）**：
7. driver 生成输入时对 int/float tensor **只按 dtype `randn/randint`**，**不读 model.json 的 `value` 字段**——varlen 的 lens 被 `randint(-10,10)` 生成成垃圾负值（实测 -8）→ 本侧前缀 mask 几乎全 mask、kernel 几乎不计算 → **假快、加速比虚高（曾 6x 假象）**。凡带结构参数（lens/seqlen/cu_seqlens/blockTable）的算子，要么给 model.json 的 `value` 且确认 driver 支持 value 优先（perf 评测脚本 aog_perf_eval.py 已修），要么精度与性能必须用同一套输入交叉验证，否则不可信。MLA 无此问题（exe 回放 + 显式 seqlen 构造，不经通用生成器）。

## 12.8-12.9 [已删除] 曾尝试的"复制外部源码移植"路径

> **2026-08-26 已从本 skill 删除**。历史曾把生产 kernel 源码复制进工程（非手搓、依赖外部仓），违反"skill 自包含、可手搓生成"原则，已废弃。**手搓生成请走**：§12.1-12.4 骨架 + §12.10 算法原理 + §12.11（sink）/§12.12（varlen/q8）流程。


---

- [ ] 选对算法变体：sink/none + Skv≤1024 → S1 单遍组织，非 s1s2 双层
- [ ] 按 §B 7 相位手搓写齐 kernel（入口/AIC REGIST/AIV 主循环/bmm1 QK/online softmax+sink/bmm2 PV/divide-rescale），op 类自写、不复制外部源码
- [ ] S1 路径 **不开 L1Reuse、splitFactor 不 ×2**（aicRatio=1）
- [ ] 入口 14 参 + `__kfc_workspace__` + AIC 只 REGIST + kfc 5 项（见 §12.8 C / §12.9 F#2）
- [ ] host `SfaBuildTilingS1` 按 §D 公式推导（`SFA_DEBUG_TILING=1` dump 自验）
- [ ] UB 双缓冲 + 硬件事件每 id 恰好 1 Set+1 Wait（§C / §12.10 B）
- [ ] case 生成走 aog-input-gen-builder SCHEMA（seq_len 主扫 + 内存 invariant），case 目录只放 model.json
- [ ] 性能对比逐 case 子进程隔离，标杆 = aclnn（`npu_fusion_attention`+`sink=`），注明算子名
- [ ] 精度 ≥30/30 PASS（含超长序列），对标杆 sink ≥0.8x（混合）/ ≥0.9x（H=16 长扫描）

## 12.12 fa-varlen / fa-q8 kernel 手搓流程（在 §12.11 S1 单遍组织上叠加）

> 三特性共用 **§12.13（优先，kfc 单 TU 直调 chunk 流式）或 §12.11（S1 模板形态）** 的 FA 骨架；本节只写 varlen（前缀 mask）与 q8（W8A8 反量化）在自写 kernel 内的**手搓叠加步骤**（不复制任何源码）。sink 完整流程见 §12.11/§12.13。叠加后 107 例精度、三算子对标杆全超 0.6 gate（varlen 1.12x / q8 1.50x / sink 0.81x，Sq 至 196608）。
>
> **★2026-09-02 补充**：varlen kernel 的**代码级逐步骤**
> 见 [fa-varlen-handcraft-recipe.md](fa-varlen-handcraft-recipe.md)——该配方走 s1s2 双层 + TND 直读 +
> 批界 s2End 截断（与本节 A 的"-3e38 前缀掩码行"法是两条等价路径，含对照表）；host 契约注意
> 该配方 seqlen 传**累加和**、kernel 内差分。

### A. fa-varlen 手搓步骤（前缀 mask，Skv≤1024 S1 单遍）
在 §12.11 骨架的 softmax 相位内：
1. **host 传 kvLens[B]**（合法 1~Skv）；按 (b,h) 展开成 per-batch 前缀长度。
2. **softmax 前构造掩码行**：对每行，`k≥kvLen → -3e38`。实现二选一：
   - 简单：`gLen.GetValue` 标量 + K 次 `SetValue` 建掩码行 + `Add`（正确但标量热点）；
   - 优化：**预建 per-b mask 槽**（UB 里 B 份 [K] fp32，入口一次 Duplicate(-3e38)+前缀置 0），循环内 `Add(score, mask[b])`——把 per-chunk 的 K 次 SetValue 摊成一次性（长序列收益大）。
3. **tiling**：varlen 每 batch 一个 tiling（kvLen 不同 → s2 尾块不同）；host 逐 batch launch，指针偏移 `bOff=b*H`。**核对 tiling 的 s2/s2Tail 与 host 偏移配套**（batch 间不串位）。
4. 掩码行加入 softmax 的 exp 前（`-3e38` 与正常 max 不冲突，双 -3e38 等价 max 语义）。

### A2. FA-TND（varlen 接口）host 布局配方（手搓内核基座上）

- **TND Q/O 直透传零拷贝**：TND `[Σsq_b, H, D]` 的连续内存序与内核 BSND 逐 batch 视图等价——Q 传 `q + cu[b]·H·D`、O 传 `o + cu[b]·H·D`（`cu[b]=Σ_{j<b} sq_j`，tiling 逐 batch 累加），**不做全量转置**（全量转置 host 开销可达 0.19× 量级，禁用）。
- **仅 K/V 需转块格式** `[numBlocks,128,H,D]`（尾块 0 填充），per-batch 块数 = ⌈skv_b/128⌉，块表按 batch 段连续排布。
- seqlen 契约：`actualQseqlen[b]=sq_b`、`actualKvseqlen[b]=skv_b`（原始长度非累加）；tiling 逐 batch 累加偏移。
- host 每 launch 的 tiling/blockTable/长度 H2D 在上下文不变时缓存复用（重复 launch 提速 20-40%）。
- npu 标杆 varlen 只有 TND 形态（BNSD+长度数组需 host 转 TND 再喂标杆；语义等价已验证 matched=1.0）。

### B. fa-q8 手搓步骤（W8A8 核内反量化，fp16-only）
在 §12.11 骨架上加 dequant 相位：
1. **host**：`hasQuant` 时把 int8 q/k/v 反量化到 fp16 内部缓冲（或核内）；scale 契约=gen 写乘系数 s、kernel 用 **1/s**（W11）。
2. **kernel 内 dequant 相位（AIV 前置）**：AIV 逐 tile 把本 tile 的 int8 操作数切片 `DataCopy(GM→UB int8) → PipeBarrier<PIPE_ALL>(★MTE2→V) → Cast<half,int8_t> → Muls(scale)` 写回 fp16（staging）。
3. **配对 flag**：dequant 完成后 `Set(dqReady id=4)`，AIC 在 QK 相位前 `Wait(dqReady)`——逐批一次（非逐 tile）。
4. **约束**：dav_c220 无 int8→float 直转，q8 仅支持 fp16 链路；量化 golden 用 `round(x·s)/s`（量化-反量化值），不是 `x/s`（W12）。
5. **性能提示**：dequant 是纯带宽（int8→fp16 全量搬运），单 batch 超长序列（Sq≥32k）时成瓶颈（0.6-0.9x）——优化方向=融合进 bmm 装载（on-load cast 省一次 GM 往返）。

---

## 12.14 FA-Causal / FA-Paged 在 §12.13 基座上的逐步手搓叠加（2026-08-28）

> 两特性都是**只改 bmm1 相位与 chunk 边界**，softmax/bmm2/prescale/finalize 零改动（数学同构：causal=按行限可见列，paged=K/V 指针改查表）。

### A. FA-Causal（下三角）逐步

1. **结构边界（先省一半算力）**：任务 (bh, r0 起的 128 行 Q) 的可见 KV 上限 = `r0 + curM`（k>q 屏蔽）。chunk 循环上界从 `⌈S/1024⌉` 改为 `⌈(r0+curM)/1024⌉`；其中 `c0+cols ≤ r0` 的 chunk 为**满块**（无掩码，走原路径），仅**对角 chunk**（`c0 ≤ r0+curM < c0+cols`）需要掩码。
2. **对角 chunk 掩码（逐行尾部法，构造即正确）**：行 i（全局 q=r0+i）在 chunk 内可见列 `j ≤ i − (c0−r0)`。在 softmax 加载 S 之后、Muls 之前插入：
```cpp
const int32_t vis0 = r0 - c0;                    // ≤0（对角块 c0 ≤ r0）
for (int32_t i = 0; i < ROW_BLK; ++i) {
    const int32_t lim = i - vis0;                // 该行最后可见列（局部）
    for (int32_t j = (lim + 1 < 0 ? 0 : lim + 1); j < cols; ++j)
        ubS.SetValue(i * cols + j, NEG_LARGE);   // 不可见列 → exp=0
}
PipeBarrier<PIPE_V>();
```
   仅对角 chunk 执行（每任务至多一次），代价可忽略；`lim+1 ≤ 0` 的整行全掩（chunk 顶部行在此 chunk 无可见列， SetValue 全行）。
3. **（可选优化，FAInfer 内核的做法）**：固定压缩掩码张量 `[1024,1024] fp16`（`mask[q][k]=1 当 k>q`，host 一次预置常驻），对角块读 GM→UB 后 `mask32 × (−3e38)` 向量加到 S（§7.2 mask 版机制：额外 EVENT_ID2 管 mask 搬运 + ApplyMask）；比逐行 SetValue 快，但需对齐其 LayoutMask 索引约定。两法数学等价，先用法 2 保正确再换优化。
4. **验收**：golden 按行限列计算；`matched=1.0` 且 causal 恰好比 full 省一半时间（上三角整块跳过）。

### B. FA-Paged（block_table 分页）逐步

1. **契约**：`block_table` 为 int32 GM `[B×H 槽 × maxBlocksPerBatch]`，`block_table[bh][t] = 物理块号`（每块 128 token）；K/V 物理布局 `[总块数, 128, H, D]`。
2. **bmm1 从 chunk 级改 128 块级**：原来一个 `[128×1024×D]` 问题变成 8 个 `[128×128×D]` 子问题（与尾块处理同形态），每块的 K 指针按查表解析：
```cpp
for (int32_t t = 0; t < cols / 128; ++t) {                 // chunk 内第 t 个 128 块
    const int32_t phys = blockTableGm.GetValue(btBase + c0/128 + t);   // 标量 GM 读
    mm1.SetOrgShape(curM, 128, D);
    mm1.SetTensorB(kBlocksPlane[uint64_t(phys) * 128 * H * D], true);   // 物理块指针
    ... IterateAll → S 写入 sCur 的第 t×128 列区（行 stride = cols 打包，同尾块布局契约）
}
```
   softmax 状态跨块连续递推（SoftmaxFlashV2 isUpdate 天然支持——每块一次调用、状态数组跨块持久，**这正是 §12.13 softmax 已经在做的**，只是块粒度从 1024 变 128）。
3. **bmm2 同理改块级**（V 指针同查表；P 累加语义不变——enAtomic 逐块累加即可）。
4. **varlen 组合**：每 batch 实际块数 = `⌈kvLen[b]/128⌉`，块循环上界取该值（尾块内 `k ≥ kvLen[b]` 的列按 §12.12-A 掩 NEG_LARGE）。
5. **验收**：golden 用稠密 gather 后的 K/V 计算；恒等 block_table（`table[t]=t`）结果须与非 paged 位级一致。
