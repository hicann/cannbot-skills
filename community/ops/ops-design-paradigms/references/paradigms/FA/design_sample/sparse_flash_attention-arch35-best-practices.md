# sparse_flash_attention 最佳实践（arch35 / Ascend950PR / DAV_3510）

> MLA + 稀疏：latent 512 + rope 64 = 576 维，token-wise topk 稀疏。源算子：`ops-transformer/attention/sparse_flash_attention`。

## S1 选型

- 基础形态：**MLA**（`attentionMode` host 强制校验为 2；kernel/service 文件名全带 `mla`）。
- 维度：**dSizeNope=512、dSizeRope=64、dSize=576(HAS_ROPE)/512(无)、dSizeV=512**；`HAS_ROPE` 为独立模板参数 + tilingKey 单独占 1 bit，不污染无 rope 路径；queryRope/keyRope 必须同时存在或同时为空。
- trait：**稀疏**，`sparse_indices`(int32 topk) + `sparse_block_size` 控粒度；**arch35 强制 sparse_block_size=1（token-wise）**；sparse_mode 仅 0（全量）或 3（preTokens/nextTokens 窗口，tiling 强制 INT64_MAX，由 kernel 按 actualSeq 自算）。
- V1 vs V2 API：V2 新增 `sinksOptional`([N1], float，仅 950/DAV_3510)；V2 去掉 V1 的 dummy softmaxMax/Sum 自动创建，改严格空指针检查。

## S3 基本块（编译期常量）

来源：[sparse_flash_attention_service_cube_mla_arch35.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_cube_mla_arch35.h)

- **mBaseSize=64、s2BaseSize=128、dBaseSize=576/512、dBaseMatmulSize=128**。
- **qSNumInOneBlock=1：SFA 不切 G 轴，每个基本块只处理一行 query 的 topk**——与 SMLA 的 split-G 路线相反，是本算子的关键差异。
- PA 场景 KV 按 `PA_BSND` 布局，`S2 = maxBlockNumPerBatch × blockSize`；blockSize 须 16 对齐且被 sparseBlockSize 整除；KV 非连续仅允许第 0 维（batch/block 维），`keyStride0 /= (n2Size × qkHeadDim)` 用于跨 block 跳读。
- `returnSoftmaxLse=true` 时不支持 PA_BSND（host 检查）。

## S4 Buffer 清单

| Buffer | 层 | 份数/大小 |
|---|---|---|
| l1Right（BMM1/BMM2 右矩阵 KV） | L1 | **3（`BuffersPolicy3buff`）** |
| v0ResGm（vector 聚合后的稀疏 KV 写回 GM） | GM | **3（`BuffersPolicy3buff`）** |
| bmm1 | UB | 2（`BuffersPolicyDB`） |
| bmm2 | UB | 1（`BuffersPolicySingleBuffer`） |
| stage1OutQue / stage0OutBuf | UB | 各 2 |
| softmaxMax/Sum/Exp | UB | 各 2 |
| L0A / L0B / L0C | L0 | 64KB(2×16KB) / 64KB(2×32KB) / 256KB(2×128KB)，DB |
| l1Q | L1 | 单 buffer 常驻：`s2LoopCount==0` 全载，后续循环复用 |

**稀疏 KV 收集数据面（本算子特有）**：vector 侧 `ProcessSparseKv` 按 `sparseIndicesGm` 读 token 索引，PA 场景再经 `blockTableGm` 找物理 block，**把分散 KV 聚合到 UB 后写回 GM workspace（v0Res 三缓冲）**，供 cube 侧作右矩阵加载——稀疏 gather 在 AIV 完成，cube 拿到的是聚合后的连续数据。

## S5 流水编排

- **3 级流水**：`RunInfo runInfo[3]`，`taskId%3` 轮转——task j 跑 **BMM1(cube)/Vec0(vector)**，task j+2 跑 **BMM2/Vec1**，task j+1 跑 **Vec2**；`PRELOAD_NUM=2` 在最后两个 gS1 循环做 preload 补算。
- `SYNC_MODE=4` 跨核 flag；`CV_RATIO=2`。
- 注意与 SMLA 的阶段命名差异：SFA 有 **Vec0（稀疏 KV gather 聚合）→ BMM1 → Vec1 → BMM2 → Vec2** 五段，Vec0 是稀疏 trait 引入的前置 stage。

## S6 负载均衡

- 任务切分：以 **(B, N2, G×S1)** 为调度单元，统计 `sfaTotalBaseNum` 后按 AIC 核数均分；`bN2Start/gS1Start` → `bN2End/gS1End` 左闭右开（`InitCalcParamsEach`）。
- **IS_SPLIT_G**（G>64 即 N1>64）：相邻两个 cube 核处理同一 s1，核数减半（`coreNum >>= 1`）。
- 稀疏块跳过（[sparse_flash_attention_kvcache.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_kvcache.h)）：`ComputeS2LoopInfo` 按 sparseMode 算 `s2LineStartIdx/s2LineEndIdx`（mode 3 再按窗口与 sparseBlockCount 取 min），`s2LoopEndIdx=ceil(s2LineEndIdx/128)`，无效 s1/s2 组合 `continue` 整块跳过。
- Paged Attention：`isPa` 模板参数由 `kvLayout==PA_BSND` 决定，tilingKey 占 1 bit。

## arch35 特有：Regbase 跨核直传

[util_regbase.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/sparse_flash_attention/op_kernel/arch35/common/util_regbase.h)：
- 定义 `CVSharedParams`：AIC/AIV 经 **`__ssbuf__` 寄存器直传**的共享参数结构（bSize/n2Size/gSize/s1Size/s2Size/dSize/dSizeRope/sparseBlockCount/softmaxScale/blockSize/usedCoreNum/returnSoftmaxLse），**位域压缩**（`dSize:10`、`dSizeVInput:12`、`maskMode:4`）。
- 握手：AIC `CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(15)` 等 vector 写 ssbuf，按 uint32 逐字拷到本地。
- vector 服务内部用 `Reg::` 寄存器级 API（RegTensor/LoadAlign/Cast/CreateMask）做 scale 处理与 softmax SIMD 优化。

## S7 自检结论

- I2：MLA kvHead=1；不切 G 轴（qSNumInOneBlock=1），KV 每行 query 独立 topk 聚合，无跨头重复加载问题。
- I4：无 split-KV 需求时单 task 闭环；FD 未在本算子启用（与 SMLA 差异点）。
- 稀疏不定长：buffer 按 s2BaseSize=128 常量分配，topk 数量运行时只裁剪循环范围。
- K/V 槽池 3 ≥ r+1（3 级流水 r=2）；L0 三端口全 DB 无退化。
- 数值：mode 3 窗口外整块双侧跳过。

## 平台依赖

ssbuf Regbase 跨核直传、SIMD-Regbase 双发（vector 服务内 `Reg::` API）、sinks 仅 DAV_3510、sparse_block_size=1 限制、KERNEL_TYPE_MIX_AIC_1_2——均 950PR 口径。
