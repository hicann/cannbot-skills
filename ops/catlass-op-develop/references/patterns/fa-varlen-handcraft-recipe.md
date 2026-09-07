# fa-varlen（TND 变长）手搓完整配方（s1s2 双层组织逐步骤）

> **用途**：变长 batch（每 batch 实际 qSeqlen/kvSeqlen 不同，TND 布局 `[ΣS_b, N, D]`）的 FA kernel
> 生成依据——生产级 varlen kernel（~1200 行量级）核心差异完整拆解。
> **基座**：本配方 = **s1s2 双层组织**（s2LoopCount 内层）+ S1 kernel 同款三深流水/相位
> （QK/softmax/PV/divide 复用 [fa-sink-handcraft-recipe.md](fa-sink-handcraft-recipe.md) §4-§6，
> 本文件只写**差异**）。**特性册 §12.12-A/A2 的 kfc 叠加路径**（前缀掩码行 + K/V 转块格式）是另一条
> 自研路径，§0 有两路对照。
>
> **★varlen 三个关键事实（与直觉/手册路径不同）**：
> 1. **seqlen 传累加和，kernel 内差分**：`actualQseqlen/actualKvseqlen` 是 npu 口径的**累加数组**
>    （`qListGm[b]`），`GetSeqQlenKvlenByBoidx` 差分出 per-batch 原始长度（b=0 直取）。无需 host 预差分。
> 2. **TND 直读，不转块格式**：Q/K/V 按 `[ΣS, N, D]` 连续内存 + `s1SizeAcc/s2SizeAcc` 前缀偏移直读
>    （§3 偏移公式）——**不做** paged 块化/转置。尾块靠 `SetTail(real)` + softmax 尾列填 -inf。
> 3. **batch 边界不用掩码行**：per-batch `s2EndIdx` 截断（sparse 公式 §4）——每个 batch 的 KV 循环
>    上界就是自己的 kvLen，天然不越界读下一 batch（区别于特性册 §12.12-A 的"-3e38 前缀掩码行"法，
>    那是等长 padded 布局的 kfc 路径做法）。
>
---

## 0. 两条 varlen 路径对照（生成前先选路）

| 维度 | 本 recipe（s1s2 双层直读路径） | 特性册 §12.12-A2（kfc + paged 基座） |
|---|---|---|
| 布局 | TND 直读零转换（Q/O 直透传，K/V 也是 TND） | Q/O 零拷贝直透传；**K/V 转块格式 [numBlocks,128,H,D]**（尾块 0 填充） |
| batch 边界 | s2EndIdx 截断（每 batch 循环上界=kvLen_b） | 前缀掩码行（k≥kvLen_b → -3e38）+ per-batch tiling 逐 launch |
| seqlen 契约 | 累加和数组进 GM，kernel 差分 | per-batch 原始长度进各自 tiling |
| 组织 | s1s2 双层（s2LoopCount 内层，多遍 softmax） | S1 单遍 / chunk 流式（单遍 softmax） |
| 适用 | 对齐标杆语义（npu_fusion_attention TND） | 复合特性叠加（q8 等）、块格式 KV 已有场景 |
| 实测 | 生产级路径口径 | varlen 1.12x vs 标杆（§12.12-D） |

---

## 1. 任务切分：差分解码 + 跨 batch 推进（逐段）

```cpp
// GM 输入：qListGm/kvListGm = int64 累加和数组 [B]（actualSeqLengths/actualSeqLengthsKv）
__aicore__ void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &qLen, int64_t &kvLen) {
    if (boIdx == 0) { qLen = qListGm.GetValue(0);  kvLen = kvListGm.GetValue(0); }
    else            { qLen = qListGm.GetValue(boIdx) - qListGm.GetValue(boIdx - 1);
                      kvLen = kvListGm.GetValue(boIdx) - kvListGm.GetValue(boIdx - 1); }
}

// 任务总数（host tiling 同公式）：totalSize = Σ_b CeilDiv(qLen_b, s1BaseSize) * n2G
//   （★按实际长度切，不是 padded 长度——短 batch 少占核）

// 核内段起点初始化 CalS1OuterSize(offset)：
s1OuterSizeAcc = 0; s1SizeAcc = 0; s2SizeAcc = 0; attenB1SSOffset = 0; boIdx = 0;
for (i = 0; i < bSize; i++) {
    GetSeqQlenKvlenByBoidx(i, s1Len, s2Len);
    acc = Σ_{j≤i} CeilDiv(s1Len_j, s1Base)*n2G;
    if (offset >= acc) {   // 本核段起点还在更后面的 batch → 累计四件套前移
        s1OuterSizeAcc = acc; s1SizeAcc += s1Len; s2SizeAcc += s2Len;
        attenB1SSOffset += s1Len*s2Len; boIdx++;
    } else break;
}

// 每 task 的 ComputeAxisIdx（while 跨 batch，非 if！）：
while (idx >= s1OuterSizeAcc + CeilDiv(s1Len_boIdx, s1Base)*n2G) {
    四件套前移（同上）; boIdx++; 重读 s1Len/s2Len;
}
n2oIdx = (idx - s1OuterSizeAcc) / CeilDiv(s1Len, s1Base) / gSize;
goIdx  = (idx - s1OuterSizeAcc) / CeilDiv(s1Len, s1Base) % gSize;
s1oIdx = (idx - s1OuterSizeAcc) % CeilDiv(s1Len, s1Base);
GetSeqQlenKvlenByBoidx(boIdx, this->s1Size, this->s2Size);   // 本 task 的实际 S1/S2
```

**四件套累计量的用途**：`s1SizeAcc/s2SizeAcc` = Q/K/V 的 batch 前缀偏移（§3）；`attenB1SSOffset` =
attenMask 的 batch 前缀偏移（per-batch S1·S2 累计）；`s1OuterSizeAcc` = 任务前缀。**四个必须同步推进**
（漏一个 → mask/数据错位到别的 batch，典型症状：个别 batch 输出全错）。

---

## 2. Process 主循环差异

```cpp
// 与 S1 版（fa-sink recipe §2）的差异：
// ① 多一层 s2 内层循环（s1s2 双层组织）：
int64_t s2LoopLimit = CeilDiv(s2EndIdx - s2StartIdx, s2BaseNratioSize) - 1;
for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
    if (s2LoopCount == 0) softmaxPingPongCnt++;           // 每 task 首块切 ping-pong
    ... 三深流水体（WaitBmm1 → SetExtraInfo(带 s2LoopCount/s2LoopLimit) → IterateBmm1
        → ProcessVec1[(taskId+2)%3] → WaitBmm2 → IterateBmm2 → ProcessVec2[(taskId+1)%3]）
}
// ② extraInfo 三槽跨 s2 内层复用（taskId 连续递增，不分层）
// ③ IterateBmm1 双形态路由：extraInfo.needNz2Nd==1 ? bmm1Nz : bmm1（fp32/bf16+尾块走 NZ）
// ④ varlen 无 enableL1Reuse 奇偶对（用 needL1Carry 替代，见 §5）
```

---

## 3. TND 偏移公式（CalcQCoreOffset/CalcKCoreOffset/CalcVCoreOffset）

```cpp
// TND [ΣS, N, D]：bOffset 用前缀长度累计（非 b*固定 stride！）
Q: qCoreOffset = s1SizeAcc*n2GD + s1oIdx*s1BaseN2GD + n2oIdx*gD + goIdx*dSize
K: kCoreOffset = s2SizeAcc*n2D  + s2StartIdx*n2D + s2LoopCount*s2BaseNratioN2D + n2oIdx*dSize
V: vCoreOffset = s2SizeAcc*n2D2 + s2StartIdx*n2D2 + ... + n2oIdx*dSize
attenMask:     attenB1SSOffset（= Σ_{j<b} s1Len_j*s2Len_j）+ s1oIdx 块内偏移
// n2GD/n2D = N*D 家族常量（ComputeConstexpr）；★Q 与 K 的 bOffset 用不同累计量（s1SizeAcc vs s2SizeAcc）
```

---

## 4. per-batch sparse 截断（GetS2LoopRange——全按 actualS2Len 截断）

```cpp
GetSeqQlenKvlenByBoidx(boIdx, s1Len, s2Len);              // ★每 task 现读
CAUSAL（下三角）:        s2Start=0; s2End=Min((s1oIdx+1)*s1BaseSize, s2Len);
RIGHT_DOWN_CAUSAL:       s2Start=0; s2End=Min((s1oIdx+1)*s1BaseSize + s2Len - s1Len, s2Len);
                          // ★Sq≠Skv 右对齐 causal 的偏移项 actualS2Len-actualS1Len（等价 sparse_mode=3）
BAND:                    s2Start=Max(s1oIdx*s1Base - s1SparseValidSize, 0);
                         s2End=Min((s1oIdx+1)*s1Base + s2SparseValidSize, s2Len);
                         if (s2End - s2Start <= 0) { s2Start=0; s2End=s2Len; }   // 整块无效→退全扫
BAND_COMPRESS:           带 preTokens/nextTokens 的压缩 mask 偏移
无 sparse（纯 varlen）:  s2Start=0; s2End=s2Len;          // ★batch 边界即此——不越界读下一 batch
```

**尾块两处收口**：① s2RealSize<64 或非压缩 mask 时 softmax 前尾列 `Duplicate(-inf)`（fa-sink recipe
§5.2-① 同款）；② bmm1/bmm2 `SetTail(s1RealSize, s2RealSize, dSize)` 逐次设。**不需要块格式 0 填充**
（与特性册 §12.12-A2 的分块路径不同点）。

---

## 5. needL1Carry（KV 驻留 L1 跨 batch 复用——varlen 专属优化路径）

```cpp
// host PostTiling 判定：
needL1Carry = !isSameAB && dtype != fp32 && !hasRope
              && s1BasicBlock <= S1_BASIC_BLOCK_L1CARRY_MAX
              && dSize <= D_SIZE_L1CARRY_MAX && d2Size <= D2_SIZE_L1CARRY_MAX;
if (needL1Carry) { bmm1TilingData.shareL1Size = 0; bmm2TilingData.shareL1Size = 0; }  // L1 让给驻留
// kernel 侧（entry 宏 INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN）：needL1Carry 时走
//   op.UnpackInit(...) + op.ProcessL1Carry()（PairGetS1LoopRange 成对切分 + KV 整 batch 驻留 L1，
//   跨 task/batch 复用装载），否则常规 REGIST_MATMUL_OBJ + Process()
// 手搓直调：首次生成**不开 L1Carry**（复杂度高、收益依赖小 batch 短序列）；照常规路径生成跑通后再对照
//   ProcessL1Carry 路径评估
```

---

## 6. SameAB（自注意力 Q=K=V，`…_sab.h`）

SameAB 变体：Q/K/V 同一 GM tensor
（TND 自注意力），K/V 装载路径共享（一次装载两用），bmm1 的 B 与 bmm2 的 B 同源。事件常量
`Q_EVENT0=EVENT_ID2 / KV_EVENT0=EVENT_ID4 / P_EVENT0=EVENT_ID6`（类内自管）。
**适用**：训练侧 TND 自注意力 varlen（Sq==Skv 同源）。手搓生成先做非 SameAB 版（三输入独立），
SameAB 作为装载共享优化叠加。

---

## 7. host 契约与标杆

- 输入：q/k/v TND `[ΣS, N, D]`（k/v 可 BSH/TND 按 layOutType）；`actualSeqLengths/actualSeqLengthsKv`
  **累加和** int64 数组 [B]（npu 口径直传，勿预差分）；softmaxMax/Sum `[ΣS, N, 8]` fp32。
- tiling：host 逐 batch 累计 `totalSize = Σ CeilDiv(qLen_b, s1Base)*n2G`；coreParams/multiCoreParams
  同 S1（§12.11-D 公式按实际长度集合算，s1Base 取 max batch 适配）。
- 标杆：`torch_npu.npu_fusion_attention(..., input_layout="TND", actual_seq_qlen=累加, actual_seq_kvlen=累加)`
  （特性册 §12.12-A2：BNSD+长度数组需 host 转 TND 喂标杆，语义等价 matched=1.0 已验证）。
- 手搓直调 entry：fa-sink recipe §9 同款（`__mix__` 单 TU + REGIST_MATMUL_OBJ + userWs 独立参），
  入参追加 qList/kvList 两个 GM。

## 8. 手搓自检（varlen 版）

- [ ] seqlen 是累加和数组、kernel 差分（b=0 特判直取）——host 预差分后传差分 = 全部长度错一拍
- [ ] 四件套累计（s1Outer/s1Size/s2Size/attenB1SS）在 CalS1OuterSize 与 ComputeAxisIdx 两处同步推进
- [ ] TND bOffset 用 s1SizeAcc/s2SizeAcc（Q 与 K 不同累计量），无任何 b*固定 stride
- [ ] 纯 varlen 也有 s2End=s2Len 截断（batch 边界）；sparse 版按 §4 公式（右对齐 causal 带 s2Len-s1Len）
- [ ] s1s2 双层：s2LoopCount 进 SetExtraInfo；extraInfo 三槽按 taskId 全局取模（不分层）
- [ ] 尾块收口两处（-inf 尾列 + SetTail），无块格式 0 填充
- [ ] needL1Carry 首次生成固定 false（shareL1Size 正常 1MB）
- [ ] 标杆对齐 TND 累加口径；精度 ≥ 标杆互证 matched=1.0（B>1 多长度混合 case 必测，探 batch 串位）
