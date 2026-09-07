# FFA / FIA（双 KV TensorList + sharedPrefix）算子配方

> **用途**：FFA / FIA（fused_infer_attention_score 类算子，DeepSeek 推理形态）的**算子语义与 kernel
> 组件级结构**——双 KV（TensorList + 共享前缀）、sink 进 epilogue、FD、LSE 输出、nZ·zN KV、量化
> antiquant 入参族的完整契约。
> **生成路径选型（先读这段）**：
> - **手搓生成 FFA（双 KV 对角形态）→ 手册 [§12.5.14 终态配方](fa-kernel-handcraft.md)（实证几何 2.19×，
>   精度双口径 PASS）**——FAQK 流式配方，已闭卷验证的生成路径。
> - 本文件的价值：该算子族的**组件级结构事实**（sink-in-epilogue / FD 内建 / LSE 输出 / nZ·zN KV /
>   量化 antiquant 入参族）——生成时对齐算子语义、排查对照、或按组件路径复刻时用。
> - 该 kernel 与 [fa-paged-handcraft-recipe.md](fa-paged-handcraft-recipe.md) **同架构**（catlass
>   BlockMmad + CrossCoreFlag `<0x2,PIPE_FIX>` + OnlineSoftmax/RescaleO epilogue + 512 大块堆叠 +
>   PRE_LAUNCH 延迟 PV）——先掌握 paged recipe 再读本文件差异清单即可。

---

## 1. 算子语义（37 参入口契约）

```
O = softmax(scale·(Q[+qRope]·[KSharedPrefix; KTensorList][+kRope]ᵀ + pse + mask)[+sink]·
            [VSharedPrefix; VTensorList]) → attentionOut + softmaxLse(可选)
```

| 入参族 | 参数 | 说明 |
|---|---|---|
| 主体 | query, key(TensorList), value(TensorList) | ★KV 是 **TensorList**（多 tensor 列表）——双 KV 形态 |
| 共享前缀 | keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen | 全 batch 共享的公共前缀 KV（int4 输入时 host 仅做 INT4 视图转换（dtype/view 改写，无数据拷贝）） |
| rope | queryRope, keyRope, keyRopeAntiquantScale | rope 旁路（与 MLA 同思路） |
| sink | learnableSink | ★sink 在 epilogue 内建（`Epilogue::SinkMode`/`ElementSink` 模板参） |
| 量化 | deq/quantScale1/2, quantOffset2, antiquantScale/Offset, key/valueAntiquantScale/Offset | W8A8/反量化入参族（kernel 内 on-load dequant 路径） |
| 变长 | actualSeqLengths(Q)/KV, queryPaddingSize, kvPaddingSize | 累加和口径 + padding 标记 |
| 分页 | blocktable | 跨 tensor 池的物理块号表 |
| 输出 | attentionOut, softmaxLse | LSE 可选（`*_LSEOUT_*` tiling key 位） |

**双 KV 语义**：sharedPrefix（公共前缀，所有 batch 共享一段）+ TensorList local（per-batch 各自池）；
kernel 侧统一按 **paged blockTable 连续寻址**（tiling 的 kvSeqlen = sharedLen + localLen_b 逻辑拼接），
不分两次 launch、不做 host 拼接拷贝。

## 2. kernel 结构（与 fa-paged recipe 的差异清单）

| 维度 | fa-paged（标准 FAInfer 形态） | FIA regular 形态 |
|---|---|---|
| 分流 | `operator()<AIC>/<AIV>` 模板特化 | `#ifdef __DAV_C220_CUBE__/__DAV_C220_VEC__` 同文件并列（MLA 式） |
| 任务型 | `KERNEL_TASK_TYPE` 未显式 | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` |
| 组件 | FAIQK/FAIPV + Tail | 同组件族 + `resetBlockStart(kvStart, pagedBlockSize)`（每 task 重置块内游标） |
| KV 布局 | ND | ND 或 **nZ（K）/zN（V）** 分支（`PAGED_CACHE_FLAG && is_same_v<LayoutK,nZ>`：`kRow=blockStack*strideK` 重排 layout） |
| 大块 | blockStackNum=4（512 列） | `MAX_KV_STACK_LEN` 常量（等价 512）+ `preKVNum=PRE_LAUNCH` 延迟 PV 同款 |
| sink | 无 | `gSink[gmOffsetSink]` 直传 epilogueOnlineSoftmax（8 处调用点全部带 sink 参） |
| mask | 压缩 [1024,1024] | `COMP_TRIU_MASK_DIM_LEN` 压缩上三角（同思路，尺寸常量名不同）+ `LayoutQ layOutFullMask(pseQ,pseKv)` |
| FD | 无 | `IS_FD` 模板位：`kvStart=stS2IdxNow / kvEnd=enS2IdxNow`（flash-decoding 每核只扫自己的 s2 段，MLA FDRescaleO 的组件版对应物） |
| 空任务 | — | `kvSLoopNumTotal<=0 || startIdx>=...` 时 VEC 侧 `epilogueInitOut`（直接初始化输出，防读未写） |
| LSE | 无 | `gLse` + `LayoutLse(totalQTokens, qHeads)` |
| sparse 跳块 | noSkipKvS 公式 | 同公式 + `delStartRow/delEndRow`（band 边界行修正） |
| workspace 槽 | `coreIdx*WS_DB*(PRE_LAUNCH+1)+slot*WS_DB` | 同公式（WORKSPACE_BLOCK_SIZE_DB 同源） |

## 3. 主循环骨架（与 paged recipe Step 3f/4e 同构）

```cpp
blockMmadQK.resetBlockStart(kvStart, pagedBlockSize);
blockMmadPV.resetBlockStart(kvStart, pagedBlockSize);
blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize, qHeads);
for (uint32_t kvSIdx = kvStart; kvSIdx < kvEnd + preKVNum; kvSIdx++) {
    if (kvSIdx < kvEnd) {
        stackSeqTile = (尾块) ? noSkipKvS - kvSIdx*MAX_KV_STACK_LEN : MAX_KV_STACK_LEN;
        isLastStackTile = (kvSIdx + 1) >= kvSLoopNumTotal;
        uint32_t slot = stackSeqCount % (PRE_LAUNCH + 1);
        gmOffsetS = coreIdx*WS_DB*(PRE_LAUNCH+1) + slot*WS_DB;
        GemmCoord shapeQK{rowNum, stackSeqTile, embed};
        LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
        CUBE: blockMmadQK(gQ, gK, gS, gBlockTable[...], layouts, shapeQK,
                          kvSIdx, kvSLoopNumTotal, pagedBlockSize, strideK, keyBnStride);
              CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
        VEC:   (MASK_CAUSAL 分支) epilogueOnlineSoftmax(gP, gS, gSink, gMask, layOutP, layOutS,
                  layOutMask, layOutFullMask, shapeQK, (stackSeqCount==0), qSBlockSize, qNBlockSize, slot, ...);
              CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
    }
    if (kvSIdx >= PRE_LAUNCH) { /* 延迟 PV：blockMmadPV(..., softmaxReady) + Set(pvReady) → VEC rescaleO */ }
    stackSeqCount++;
}
```

**与 paged recipe 的 3 条修复仍适用**（LayoutP stride / isLast / L1 偏移）；mask 版 softmax 调用带
sink/fullMask 扩参（paged recipe Step 8e 的 superset）。

## 4. tilingKey 路由（aclnn registry 分发用）

`QF16_KVF16_OUTF16_{NOLSEOUT,LSEOUT}_TND_{NOCACHE,PAGEDCACHE}[_KVNZ]_{NOMASK,CAUSALMASK}_SPLITFUSE_TILING`
——dtype × LSE × paged × KVNZ × mask 五维组合；decode 形态另有 `FAInferDecoding`（`IS_FD` 段切分版）。
**手搓直调不需要 tilingKey**（仅 registry 分发用）。

## 5. 手搓生成 FFA 的推荐路径（对照表）

| 需求形态 | 路径 |
|---|---|
| 双 KV 对角 FFA（`S=Q·K1ᵀ+Q·K2ᵀ对角`） | **配方册 §12.5.14 终态配方**（FAQK 流式 + K 块流式在线 softmax，几何 2.19×，实证） |
| sharedPrefix + local（DeepSeek 推理形态） | fa-paged recipe 基座 + 本文件 §1 语义（blockTable 跨池编址 + kvSeqlen=shared+local）+ 特性册 §12.14-B paged 查表 |
| sink 叠加 | §12.13-C（kfc 播种）或 fa-sink recipe §5.2（SoftmaxFlashV2 isUpdate） |
| W8A8 量化 KV | antiquant 入参族 + §12.12-B dequant 相位（DataCopy→Cast 间 PIPE_ALL，scale=1/s） |
| FD 长序列 decode | MLA recipe §8（FDRescaleO 组件版）或 §12.13 原子累加版 |

## 6. 标杆对比

- Python 标杆：`torch_npu.npu_fused_infer_attention_score*`（注意 executor 每次 launch 重建，计时口径
  见 [fa-perf-gate-evaluation.md](fa-perf-gate-evaluation.md)：必须 profiler op_summary device 口径）。
- 精度互证：smax/ssum 应与标杆位级一致（配方册 §12.5.14 验证闭环同款）。
