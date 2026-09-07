# fa-sink / 标准 FA（S1 模板）手搓完整配方（S1 单遍组织逐步骤）

> **用途**：`O=softmax(scale·Q·Kᵀ[+mask])[+sink]·V` 在 **S1 单遍组织**（Skv≤1024 全场景默认路径，
> 配方册 §12.11 判据）下的 kernel 生成依据——生产级 S1 模板 kernel（~1800 行量级）的
> 核心相位完整拆解 + tiling 公式。
> **与 §12.11/§12.13 的关系**：§12.11 是本 kernel 的算法语义/tiling 公式/性能基线（S1 对标杆 0.80-0.92x）；
> §12.13 是 kfc chunk 流式自研配方（任意序列长，geomean 0.693）。本 recipe 补的是**kernel 类的真实
> 代码结构**（相位顺序/事件配对/sink 正确用法/workspace 公式精确版）——三份合起来闭卷可生成。
>
> **★sink 正确用法（本文件最有价值的一段，勿自创）**：sink 不加虚拟列、不改 GEMM——在 softmax 播种：
> `maxUb←sink[head]`、`sumUb←1.0`，然后 `SoftmaxFlashV2` 走 **isUpdate=true 模板参**（有 sink）vs
> `isUpdate=false`（无 sink，首块自初始化）。m/l 递推由 SoftmaxFlashV2 内部完成（见 §5 SoftMaxCompute）。
>
---

## 0. 架构总览（单核融合 + 三深流水，与配方册 §12.10 一致——此处为代码级精确版）

```
每核 task 段 = [multiCoreInnerOffset, multiCoreInnerLimit)（连续段，非 round-robin）
每 task = (boIdx, n2oIdx, goIdx, s1oIdx)：一锅 128 行 query（S1 单遍：s2 一次扫完）
三深流水（extraInfo[taskId%3]）：QK(task) ‖ softmax(task-2) ‖ PV(task-1)+divide(task-1)
S/P 不落 GM 全程？——否：S 落 per-core GM workspace（mm1Res 双缓冲），P 落 stage1Res（bmm2 的 A）。
  mm1Res/stage1Res/mm2Res 都在 GM，但 ping-pong 双缓冲让搬运与计算重叠。
```

**模板参数**（类 `FlashAttentionScoreS1Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, …>`）：
`layOutType ∈ {LAYOUT_BSH, LAYOUT_SBH, LAYOUT_BNSD}`（三布局偏移公式在 §3/§6）；`enableL1Reuse`
（D∈[64,128] 开：两核共享 task 对、blockIdx/2 切分、奇偶跳过——见 §2）；bmm1Format/bmm2Source
（ND/NZ/CUBE 格式路由，dSize%16≠0 走 NZ 路）。

---

## 1. 成员与 UB 布局（InitBuffer 尺寸）

```cpp
TBuf<> maskTBufPing;   // 11K  atten mask 装载（hasSink 时复用作 softmax exp 输出！见 §5）
TBuf<> maskTBufPong;   // 11K  drop mask 装载
TBuf<> pseTBuf;        // 16K  pse 装载 / Vec2 阶段复用作 softmaxSum 装载
TBuf<> stage1PingBuf;  // 32K  softmax 工作区（fp32 S）+ cast 源
TBuf<> stage1PongBuf;  // 32K  bmm1 ND 格式直读 / Vec2 的 bmm2Res
TBuf<> vecOut;         // 16K  NZ 路输出
TBuf<> softmaxSumBuf;  // 8K   跨 splitN 积攒的 sum（[softmaxCopyOutLimit][vecS1BaseSize][8] fp32）
TBuf<> softmaxMaxBuf;  // 8K   同上 max
TBuf<> softmaxExpBuf;  // 8K   ★仅作 SoftmaxFlashV2 的占位参数（hasSink 时换 maskTBufPing）
TBuf<> commonTBuf;     // 32K  SoftmaxFlashV2 apiTmpBuffer / Vec2 输出 ping-pong
GlobalTensor<T>        mm1Res[2], mm2Res[2];         // GM 双缓冲
GlobalTensor<INPUT_T>  stage1Res[2];                 // P（bmm2 的 A）
```

---

## 2. Process()：三深流水主循环（逐段）

```cpp
int64_t offset = blockIdx * splitFactorSize;                  // 连续 task 段
if constexpr (enableL1Reuse) offset = blockIdx / 2 * splitFactorSize;   // L1R：两核一 task 对
int64_t limit = Min(offset + splitFactorSize, totalSize);
GetS1LoopRange(offset, limit);                                // sparse 负载均衡（sparseStartIdx 表，§7）
SplitS1dExtraInfo extraInfo[3];
int64_t taskId = 0;
event_t evMte3ToMte2 = FetchEventID(HardEvent::MTE3_MTE2);
bool notSecondLast = true, notLast = true;
limit += 2;                                                   // ★多跑 2 个空 task 位给流水 drain
bool unPair = false, lastNotPair = false;
if constexpr (enableL1Reuse) {
    limit += 2;                                               // L1R 再多 2（task 对奇偶）
    if ((limit - offset) % 2) { limit += 1; unPair = true; }
}
for (int64_t idx = offset; idx < limit; ++idx) {
    // 末尾标记（L1R 与非 L1R 的判据差 2——因 task 对）
    if constexpr (enableL1Reuse) {
        if (idx == limit - 4) notSecondLast = false; else if (idx == limit - 2) notLast = false;
        if ((idx - offset) % 2 != blockIdx % 2) continue;     // ★L1R 奇偶跳过：每核只跑自己奇偶的 task
    } else {
        if (idx == limit - 2) notSecondLast = false; else if (idx == limit - 1) notLast = false;
    }
    if (idx + 1 == limit - 4) lastNotPair = unPair;
    bool notLastTwoLoop = notSecondLast && notLast;

    ComputeAxisIdx(idx);                                      // §3
    GetS2LoopRange();                                         // §7 causal/band 跳块
    if (taskId >= 1 && notLast) WaitBmm1Result();             // bmm1.WaitIterateAll()+End()（task-1 的 QK）
    if (notLastTwoLoop) {
        SetExtraInfo(extraInfo[taskId%3], taskId, 0, 0, L1R? idx/2 : idx, lastNotPair);
        IterateBmm1(extraInfo[taskId%3]);                     // 提交 task 的 QK（异步 IterateAll<false>）
    }
    if (taskId > 0 && notLast) {
        ProcessVec1(extraInfo[(taskId+2)%3]);                 // softmax task-2 的 S
        SetFlag<HardEvent::MTE3_MTE2>(evMte3ToMte2);          // P 写 GM 可见
    }
    if (taskId > 1) {                                         // PV task-1 的 P
        (dSizeAlign16 == dSize) ? WaitBmm2Result(bmm2) : WaitBmm2Result(bmm2Nz);
    }
    if (taskId > 0 && notLast) {
        WaitFlag<HardEvent::MTE3_MTE2>(evMte3ToMte2);
        IterateBmm2(extraInfo[(taskId+2)%3], dSizeAlign16==dSize ? bmm2 : bmm2Nz);
    }
    if (taskId > 1) ProcessVec2(extraInfo[(taskId+1)%3]);     // divide task-1 的 O
    ++taskId;
}
```

**三深流水账目**（自查）：每 task 恰好 1×{WaitBmm1, IterateBmm1, ProcessVec1+Set(MTE3_MTE2),
Wait(MTE3_MTE2)+IterateBmm2, WaitBmm2, ProcessVec2}（首尾 task 按 taskId>0/>1 与 notLast 剪掉）。
多算的 2（L1R 4）个空 task 只走 Wait/ProcessVec 消费侧——**不是 bug，是 drain**。

---

## 3. 轴解码：ComputeAxisIdx

```cpp
boIdx  = idx / n2GS1o;                       // batch
n2oIdx = idx % n2GS1o / gS1o;                // kv head（n2=N2 kvHeads）
goIdx  = idx % gS1o / s1OuterSize;           // GQA 组内 head 块
s1oIdx = idx % s1OuterSize;                  // query 128 行块
// n2GS1o = n2Outer*gOuter*s1Outer（tiling coreParams 派生）；GQA：qHead = n2oIdx*gSize + goIdx
```

---

## 4. QK 相位：IterateBmm1 + A/B 装载

```cpp
// SetOrgShape 仅在 s1RealSize 变化或 sparse 时重设（省 API 开销；lastVec1S1RealSize 记忆）
if (s1RealSize != lastVec1S1RealSize || sparseType > 0) {
    bmm1.SetOrgShape(s1RealSize, mm1Kb, mm1Ka, mm1Kb, s2RealSize);   // 5 参：M,N,K,K2,N2
    lastVec1S1RealSize = s1RealSize;
}
Bmm1SetTensorA(extraInfo);      // A=Q：三布局偏移（BNSD: b*n2GS1D + n2o*gS1D + go*s1D + s1o*s1BaseD）
SetBmm1TensorB(extraInfo);      // B=K：SetTensorB(keyGm[kOff], true)（★transB 运行时 flag）+ SetTail(M,N,K)
bmm1.template IterateAll<false>(mm1Res[taskIdMod2], false, false, true);   // 异步；L1R 多一个 lastNotPair 参
```

**布局偏移公式表（BNSD 版；BSH/SBH 按同构规则对称推导）**：
- Q(BNSD)：`bOffset=b·n2S1D…`——实际成员名 n2GS1D/gS1D/s1D/s1BaseD（=H*S1*D 家族，ComputeConstexpr 算好）
- K(BNSD)：`b·n2S2D + n2o·s2D + s2StartIdx·dSize`
- V(BNSD)：同 K 公式（s2StartIdx 起）——causal 时 s2StartIdx>0（跳块）

---

## 5. softmax 相位：ProcessVec1（外层 splitN 循环）+ SoftMaxCompute

### 5.1 ProcessVec1 每 loopIdx（realSplitN = ⌈s2RealSize/列预算⌉ 次）的顺序（★顺序即正确性）

```
① vecS1TailSize 尾块修正（loopIdx==realSplitN-1 → s1RealSize - loopIdx*vecS1BaseSize）
② loopIdx>0: WaitFlag(V_MTE2, evB)                          // 上一轮 P copyout 排空
③ hasPse: PseCopyIn/PseSlopeCopyIn（外mul内add 两种 pseType 分支）
④ GetBmm1Result：DataCopy(mm1Res → UB)（ND 直拷 / NZ 走 NzToNd）+ PipeBarrier<PIPE_V>
   + SetFlag/WaitFlag(MTE2_V)                                // MTE2→V 屏障
⑤ Muls(stage1Ping, S, scaleValue, vecS1TailSize*s2RealSizeAlign64)   // scale
   （PSE_OUTER_ADD_MUL_TYPE 时 scale 延后到 mask 后——保持 add-then-scale 语义）
⑥ CopyInAttenMask(-1)（hasAtten：mask GM→UB）
⑦ ComputeAttenMask（score += mask·negFloat；BAND 二段/PREFIX AND 合并两趟）
⑧ loopIdx<realSplitN-1 且 !hasSink: SetFlag(V_MTE2, evB)     // ★sink 时此 Set 移到 SoftMaxCompute 后
⑨ SoftMaxCompute（5.2）
⑩ loopIdx<realSplitN-1: SetFlag(V_MTE2, evA)                 // bmm1Res UB 复用前放行
⑪ loopIdx>0: WaitFlag(MTE3_V) + PipeBarrier<PIPE_V>
⑫ P 落 GM：T≠INPUT_T（fp16 输出）→ Cast(ROUND) + DataCopy(stage1Res[taskIdMod2][loopIdx*s1BaseTimesS2Align16])
    （2D DataCopyParams：blockCount=vecS1TailSize, blockLen=s2Align16/16, srcStride=(s2Align64-s2Align16)/16）
    NZ 路：Cast → NdToNz（nd2nzParams：dstNzC0Stride=CeilDiv(row,16)*16）
⑬ loopIdx<realSplitN-1: SetFlag(MTE3_V)
```

### 5.2 SoftMaxCompute（sink 正确用法在此）

```cpp
// ① 尾列填 -inf（hasAtten==false 或非压缩 mask 或 s2RealSize<64 时）：
//    Duplicate(srcTensor[s2RealSizeFloorAlign8], negFloatScalar, mask, vecS1TailSize, 1, s2Align64*elems/blockBytes)
// ② srcTensor.SetShapeInfo(ShapeInfo(2, {vecS1TailSize, s2Align64}, 2, origin, ND)) + SetSize
// ③ sumUb/maxUb 取 softmaxSumBuf/MaxBuf[ (loopIdx % softmaxCopyOutLimit)*vecS1BaseSize*fp32BaseSize ]
//    （8K buf 攒 softmaxCopyOutLimit 轮再统一写 GM——S1>256 时循环复用）
// ④ ★sink 播种（hasSink）：
float inMax = sinkGm.GetValue(n2oIdx * gSize + goIdx);      // sink=[H] fp32，按 head 索引（scale 后尺度）
float inSum = 1.0;
SetFlag<HardEvent::S_V>(ev); WaitFlag<HardEvent::S_V>(ev);   // 标量读回屏障
Duplicate(sumUb, (T)inSum, vecS1TailSize * fp32BaseSize);    // l₀=1（★指数域：等价 l₀=exp(sink-m₀)·e^{m₀} 记法）
Duplicate(maxUb, (T)inMax, vecS1TailSize * fp32BaseSize);    // m₀=sink
// ⑤ SoftMaxTiling = SoftMaxFlashV2TilingFuncImpl(vecS1TailSize, s2Align64, sizeof(T), sizeof(T),
//        tmpSize, /*isUpdate=*/false, /*isBasicBlock=*/true)
// ⑥ SoftmaxFlashV2 调用（★第 2 模板参 isUpdate 有无 sink 不同）：
if (hasSink) {
    expUb = maskTBufPing;                                    // ★sink 时 exp 输出换 buffer（避 state 冲突）
    SoftmaxFlashV2<T, /*isUpdate=*/true,  true, true, false, SOFTMAX_DEFAULT_CFG>(
        srcTensor, sumUb, maxUb, srcTensor, expUb, sumUb, maxUb, apiTmpBuffer, newTiling);
} else {
    SoftmaxFlashV2<T, /*isUpdate=*/false, true, true, false, SOFTMAX_DEFAULT_CFG>(
        srcTensor, sumUb, maxUb, srcTensor, expUb, sumUb, maxUb, apiTmpBuffer, newTiling);
}
// in-place（dst=src 首参）；expUb 布局不可依赖（配方册 §12.13-C 同款结论——α 不从 expMax 读）
// ⑦ softmax sum/max 攒批写 GM：loopIdx==realSplitN-1 || (loopIdx+1)%softmaxCopyOutLimit==0 时
//    V_MTE3 Set/Wait → DataCopy(softmaxSumGm/MaxGm[gmOffset], buf[0], calculateSize) → MTE3_V Set
//    gmOffset = b·n2GS1·8 + n2o·gS1·8 + go·s1Size·8 + s1o·s1Base·8 + softmaxOutOffset·8
```

**sink 数学核对**（StreamingLLM）：`m_f=max(m,sink)`、`l_f=l·exp(m−m_f)+exp(sink−m_f)`——本配方用
播种+isUpdate 把这一递推交给 SoftmaxFlashV2 内部；`l₀=1,m₀=sink` 的初态在指数域等价于虚拟 sink 列
（exp(sink−m)=…）。**不要在 kernel 里手写 max/add 链复刻**（特性册 §12.6-E 的手写链是 kfc 路径的替代品，
S1 模板路径用 SoftmaxFlashV2 组件形态）。

---

## 6. PV + divide 相位：IterateBmm2 + ProcessVec2

```cpp
// IterateBmm2：A=stage1Res[taskIdMod2]（softmax 的 P，GM）；B=V（SetTensorB 无 transFlag）
if (s1RealSize != lastVec2S1RealSize || sparseType > 0) {
    bmm2.SetOrgShape(s1RealSize, mm2Kb, s2RealSizeAlign16, mm2Kb, dSize);
    lastVec2S1RealSize = s1RealSize;
}
bmm2.SetTensorA(stage1Res[taskIdMod2]);
bmm2.SetTensorB(valueGm[vCoreOffset]);            // TSCM 路：V 装载驻留 L1（bmm2Source==TSCM）
bmm2.SetTail(s1RealSize, dSize, s2RealSize);      // ★通过 tail 限制本次 K 维=s2RealSize
bmm2.template IterateAll<false>(mm2Res[taskIdMod2], false, false, true);

// ProcessVec2：每 vec2S1Idx（vec2S1BaseSize=tiling coreParams.s1Vec2BaseSize，D 小时可 >128）
// ① DataCopy(mm2Res → bmm2ResUb)；dSize%16≠0：NzToNd（nz2NdInfo 五字段）回 ND
// ② DataCopy(softmaxSumGm[sumGmOffsetLoop] → sumUb)         // ★从 GM 读回累计 l（不是 UB 里留）
// ③ Bmm2ResultDiv：Brcb(expSumUb, sumUb) → Div(bmm2Res, bmm2Res, expSumUb)（行广播除）
//    （AA_INVALID_LINE_HIGH_PRECISION 模式先 AdjustSoftMaxRes 修正无效行）
// ④ Bmm2DataCopyOut：Cast(fp16, ROUND) → DataCopy(attentionOutGm[qCoreOffset + vec2S1 偏移])
//    用 commonTBuf 双 ping-pong（[0]/[8192]）——搬运与上一轮 cast 重叠
// ⑤ 循环外：SetFlag/WaitFlag(MTE3_MTE2) 收尾配平
```

---

## 7. sparse/causal 跳块：GetS1LoopRange + GetS2LoopRange

```cpp
// GetS1LoopRange（sparse>0）：per-core 段改读 multiCoreParams.sparseStartIdx[blockIdx] 表
//   （host 预按三角面积均衡切好的 per-core 起点数组；L1R 时步进 2：sparseStartIdx[b] ~ [b+2]）
// GetS2LoopRange（每 task）：
CAUSAL（下三角）: s2StartIdx=0; s2EndIdx = Min((s1oIdx+1)*s1BaseSize, s2Size);
BAND（对角带）:   s2StartIdx = Max(s1oIdx*s1BaseSize - preTokens, 0)（dropmask bit 模式向下 8 对齐）
                  s2EndIdx   = Min((s1oIdx+1)*s1BaseSize + nextTokens, s2Size);
PREFIX/rightDown 类似 + ComputeOffsetForPrefixRectangle 的压缩 mask 索引
// ★causal 的省算力在 s2EndIdx 截断（整块跳过对角以上）；对角内逐位由 attenMask 压缩矩阵处理
//   （mask [2048,2048] 压缩格式，offset=ComputeOffsetForCausal(delta=Skv-Sq)：delta≤0 → 行偏移
//    Min(-delta, s1Base)；delta>0 → 列偏移 delta·attenMaskS2Size）
```

---

## 8. workspace 精确公式（InitInput——per-core 基址）

```cpp
s2SizeAlign16 = CeilDiv(s2Size, 16)*16;
mmNRatioOffset   = CeilDiv(s1BaseSize*s2SizeAlign16, 256)*256*sizeof(T);        // mm1Res 单份
stage1WsOffset   = mmNRatioOffset;                                               // stage1Res 单份
if (INPUT_T==float) stage1WsOffset = mmNRatioOffset*sizeof(INPUT_T)/2;           // fp32 稀疏不复用
bmm2ResultOffset = CeilDiv(s1BaseSize*dSizeAlign16, 256)*256*sizeof(T);          // mm2Res 单份
pseAlibiOffset   = CeilDiv(pseAlibiBaseS1*pseAlibiBaseS2*2, 512)*512;
totalOffset = blockIdx * (mmNRatioOffset*2 + stage1WsOffset + bmm2ResultOffset*2 + pseAlibiOffset);
mm1Res[0/1]     = ws + totalOffset + {0, mmNRatioOffset}
stage1Res[0/1]  = ws + totalOffset + 2*mmNRatioOffset + {0, stage1WsOffset/2}
mm2Res[0/1]     = ws + totalOffset + 2*mmNRatioOffset + stage1WsOffset + {0, bmm2ResultOffset}
pseAlibiGm      = ws + totalOffset + 2*mmNRatioOffset + stage1WsOffset + 2*bmm2ResultOffset
// 另：dropmask bit 模式时 workspace 头部还有 dropmaskParams.shapeTotalSize（512 对齐）——
//     InitInput 里 dropMaskGm 先占 workspace 头，后续偏移要跳过它
```

---

## 9. entry 形态（手搓直调，按 §12.11-B/§12.13-A）

**手搓直调**：单 TU `__global__ __mix__(1,2)` + `__kfc_workspace__` 第 8 参 + userWs 独立参数 +
`REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, &t.bmm1Tiling, op.bmm2, &t.bmm2Tiling)` +
`if ASCEND_IS_AIC return` + Init/Process（§12.11-B 14 参形态 / §12.13-A 11 参形态二选一）。
matmul tiling 按框架计算器 `matmul_tiling::MatmulApiTiling`（§12.13-A-2），**手填 TCubeTiling 挂死**。

---

## 10. tiling 公式（§12.11-D 全量公式即本 kernel 的 host tiling 提取；源头坐标）

`s1BB/s2BB/dBB、nRatioMax 4 分支（NZND 入口条件）、s1Ratio grow/shrink（workspaceLimit=131072）、
s1Base/s1Outer、coreParams/multiCoreParams、SetBmm1/Bmm2TilingInput（SetShape/SetOrgShape/SetFixsplit）、
shareL1Size=1MB/shareL0CSize=256KB` —— 逐公式见 [fa-kernel-recipes.md](fa-kernel-recipes.md) §12.11-D。
S1 模板不开 enableL1Reuse、splitFactor 不 ×2——与 s1s2 的差异判据同 §12.11-A。
tilingKey 路由（fp16/bf16 × S1/S1S2/B 模板 × layout × ND/NZ 组合）仅 aclnn registry 分发用，
手搓直调不需要。

---

## 11. 与 §12.13（kfc chunk 流式）的选型对照

| 维度 | 本 recipe（S1 模板） | §12.13（kfc chunk） |
|---|---|---|
| 序列长 | Skv≤1024（s2 一次扫完） | 任意（chunk=1024 流式） |
| sink | SoftmaxFlashV2 isUpdate 播种（本文件 §5.2） | Duplicate 播种 m/l 状态（§12.13-C） |
| S/P 存储 | GM per-core 双缓冲（3 段 workspace） | GM per-worker 双缓冲（S/P 两段） |
| bmm2 累加 | 每 task 单次 IterateAll（O=mm2Res 再 ÷l） | enAtomic=1 跨 chunk 原子累加 O_acc |
| 实测 | 对标杆 sink 0.80x/0.922x（§12.11-H） | geomean 0.693（§12.13-E） |
| 手搓复杂度 | 相位多（pse/mask/drop 分支全） | 相位少（无 pse/drop，mask 走 pad+NEG_LARGE） |

**选型**：sink+短中序列→本 recipe；任意序列长/复合特性（varlen/q8 叠加）→§12.13 基座更省事。

## 12. 手搓自检（S1 模板版，在 §12.11-I 基础上补结构项）

- [ ] Process 三深流水的 taskId>0/>1 与 notLast/notSecondLast 剪枝逐项对齐 §2（多 2 空 task 是 drain，勿删）
- [ ] enableL1Reuse 开时：blockIdx/2 切分 + 奇偶 continue + limit 再 +2 三件套成组出现（§12.10-F 配对律）
- [ ] sink：播种 max=sink[head]/sum=1 + SoftmaxFlashV2 isUpdate=true + expUb 换 maskTBufPing（§5.2 三件缺一不可）
- [ ] hasSink 时 V_MTE2 evB 的 Set 在 SoftMaxCompute **之后**（§5.1-⑧ 的位置差异）
- [ ] SetOrgShape 仅 s1RealSize 变化时重设；SetTail 逐次（M,N,K 用 real/tail）
- [ ] workspace 布局按 §8 精确公式（256 对齐 + fp32 稀疏特例 + dropmask 头部特例）
- [ ] softmaxSum/Max 攒批写 GM 的 gmOffset 五段式（§5.2-⑦）与 Vec2 读回偏移一致
- [ ] dSize%16≠0 全链路走 NZ（bmm1Nz/bmm2Nz + NdToNz/NzToNd），dSize%16==0 走 ND——勿混
