---
type: "operator"
title: "block_sparse_attention Arch22 分段流水与向量归约实测事实"
description: "在 DAV_2201 上做 QK→softmax→PV→finalize 融合时，可用的段间同步形态，以及 level-2 归约、WholeReduce、Fixpipe C 布局与在线 softmax 的实测语义与容错要点。"
tags:
- catlass-cpp
- sparse-attention
- block_sparse_attention
- BlockSparseAttention
- 块稀疏注意力
- arch22
status: "stable"
generated: {"by": "process:bsa-arch22-implementation"}
verified: [{"by": "process:device-probe"}]
sources: [{"id": "sparse-attention-device-pipeline-and-reduce-facts-source", "resource": "audit:custody/sparse-attention#device-pipeline-and-reduce-facts", "title": "device-pipeline-and-reduce-facts 固定来源（提取侧独立保管）", "kind": "audit-record"}]
operator_families: ["sparse-attention"]
architectures: ["DAV_2201"]
---

# 接口与概念

融合 QK、online softmax、PV 与 rescale/finalize 的注意力算子需要 Cube 与 Vector 两套执行单元配合，
存在「Cube 产出 S/P 后向量侧未就绪」的数据依赖。本条目给出两类已在设备上实测的结论：
**段间同步形态**（用什么代替跨核握手）与**向量/搬运原语的准确语义**（照文档写会静默出错的地方）。
判据来源见文末「验证方法」。[^sparse-attention-device-pipeline-and-reduce-facts-source]

# 用法

- 融合一次调用时，先按「Cube 段 / Vector 段」把算法切成若干段，段间数据边全部落在 GM，
  段内只用配对的 `HardEvent`（S_V/V_S、MTE2_V/V_MTE2、V_MTE3/MTE3_V）。
- 多元素行的归约用 `WholeReduceSum/WholeReduceMax`，不要用 level-2 的 `ReduceSum/ReduceMax` 直接归约整行。
- 矩阵块调用前，先核对 C 布局的 `shape(1)` 是否等于本次 tile 的 N，以及 `mSize * dstStride`
  是否超过 L0C 的 fp32 容量。
- 在线 softmax 的每行因子（alpha、m、l）要区分「tile 局部下标」与「全局行号」，并按行数分块同步。
- 用 GM 做跨核软握手（轮询标志/标量）时，轮询读必须用不会被优化掉的形式：读前显式无效 cache，
  或改成 MTE 读回后再判。`GlobalTensor::GetValue` 不是 volatile，直接放进 `while` 条件里
  会被编译器提升成「只读一次」；单次读（读一次表项）不受影响。
- 需要矩阵侧与向量侧并行时，最省事的形式是让两者**各自处理互不相干的数据**（例如相邻两批分别落在
  AIC 与 AIV 上）：同一 kernel 内两者即真实并行，**不需要任何跨核 flag**。有依赖的方向必须显式握手
  （AIC 侧跨核等待是已知挂死点）。独立与依赖的边界就是这条规则的适用边界。

# 代码模式

```cpp
// 段间：同一条 stream 上顺序启动四个 kernel，段间不需要任何跨核 flag
for (uint32_t stage = 0; stage < 4; ++stage) { kernel[stage] <<< cores, nullptr, stream >>>; }
```

```cpp
// 行宽 512 的行和（16 行一组，repeat=128 仍在 8 bit 内；块间 srcRepStride = 8 个 32B 单位）
AscendC::WholeReduceSum(part, rowStart, 64, rowsInGroup * 8, 1, 1, 8);   // 每 64 列一个分块和，紧凑排列
AscendC::WholeReduceSum(rowSum, part, 8, rowsInGroup, 1, 1, 1);          // 每行 8 个分块合成行和
// 行最大同理，但 max 变体的 dst 结果间隔为 2，分块结果在标量单元按 2*i 取并按行合并
```

```cpp
// 矩阵块：C 布局的 shape(1) 必须是本次 tile 的 N；行步长只能出现在 stride(0)
LayoutRow layoutS(tileRows, chunkValid, windowStride);   // 对：shape(1)=tile 的列数
GemmCoord shape{tileRows, chunkValid, dim};
blockQK(gQ[qOff + tile * qRowStride], gK[kvOff + src * 128 * kvRowStride],
        gS[sBase + c * 128 + tile * windowStride], layoutQ, layoutK, layoutS, shape);
```

## 算子算法

对每个 (batch, qHead, X 块) 任务：按掩码行把选中的 KV 块按块号升序展开成 gather 序列（chunk 粒度 128），
逐 chunk 算 `S_c = Q K_cᵀ`；在窗口列宽 512 上做屏蔽与在线 softmax（`m`、`l`、`alpha=exp(m_old-m_new)`、
`P=exp(S'-m_new)`）；逐 chunk 算 `OTmp_c = P_c V_c`；最后按窗口用 alpha 累加、除 l，输出 O 与可选
`LSE = m + ln(l)`。归约与 exp 都在 fp32，P 按 elementIn 精度落盘。

## 分核策略与基本块切分

任务切分为「每 task 独占若干窗口（每窗口 4 个 chunk）」，task 在核间按 `taskIdx = coreIdx; taskIdx += coreNum`
轮转；每段（QK/softmax/PV/finalize）都是一次独立启动，因此四段各自轮转同一任务域。
基本块：Cube 用 `GemmShape<128,128,128>` 的 L1/L0 tile，QK 的 M 按 64 行再切（Fixpipe 约束）；
Vector 按 32 行 × 512 列一个 tile，归约按 64 列块 + 16 行一组；尾块行数、尾 chunk 列数、尾窗口列数都按实际值算。

## 数据路径与存储层级

Q/K/V 从 GM 经 L1 进 L0A/L0B，QK 的 S 在 L0C 经 Fixpipe 写 GM workspace，Vector 段把 S 从 GM 搬入 UB 做
softmax，P 由 UB 写回 GM，PV 再把 P/V 搬入 L1 并在 L0C 得 OTmp，OTmp 写回 GM 由 Vector 段累加出 O。
即 `L0C → GM → UB`（Cube→Vector）与 `UB → GM → L1`（Vector→Cube）两条路径，中间结果全部落在 workspace。

## 流水排布、同步关系与数值精度

四段之间**没有**跨核 ready/free 握手：每条数据边的 ready 由 stream 顺序（前一段 kernel 结束）给出，
段内用配对的 HardEvent 保护 UB→UB、UB→GM、L1/L0 的每次交接；空任务（chunkCount=0 或 rows=0）直接跳过，
不产生任何 slot 写入。数值上：S、P、l、O 的累加都在 fp32，P 按 elementIn 精度落盘，
在线 rescale 与 finalize 的除 l 都用运行最大值保护（因子与倒数都在 (0,1] 或 ≥1 的有界区间内）。

# 约束

- 段间数据边必须落在 GM；段内 UB 区域要显式划开，任何两段数据的生命周期不得重叠。
- `WholeReduce*` 的 `mask` 是 8 bit（≤255 元素）；`srcBlkStride/srcRepStride` 以 32 B 块为单位。
- level-2 `ReduceSum/ReduceMax(dst, src, work, count)` 只有 `count == 64` 时等于该 64 元素的归约。
- Fixpipe：`shape(1)` 决定写入列数；`mSize * dstStride ≤ L0C 容量（32768 fp32 元素 = 128 KB）`。
- UB→GM 的 `DataCopyPad` 目的地址与行长按 32 B 对齐；vector op 的 count 与偏移按 8 元素（fp32 32 B）对齐。
- P 落 fp16/bf16 之前必须已被运行最大值归一化到 (0,1]，否则会溢出 65504。

# 失败表现

| 触发点 | 现象 | 观测方式 |
|---|---|---|
| AIC 侧在矩阵块之后 `CrossCoreWaitFlag` | kernel 在 QK 写出 S 之前挂死，S 全 0 | 超时 + 停机 dump |
| C 布局 `shape(1)` 误用行步长 | 单块用例全对、多块用例全错，多出的列重复本 tile 数据 | workspace dump 与期望逐元素比对 |
| `mSize * dstStride` 超 L0C 容量 | AICore 异常 `507015`，日志 `FIXP to read L0C is out of bounds` | 同步返回码 + plog |
| level-2 归约用在 >64 元素的行 | 行和/行最大按 count/64 成倍偏大，P 与 l 不一致 | 中间量 dump 与 NumPy 复算 |
| rescale 方向写成 `exp(m_new - m_old)` | 只有「同一 task 第二个非空窗口」出错 | 逐 task 复算 |
| alpha 写全局行号而算子按 count 读 | 行数超过一个 tile 后，后续 tile 用第一段 tile 的因子 | 逐行 state 比对 |
| UB 区域重叠（cast 输出压到 m/l/row） | O 正确但 LSE 是大数/NaN | LSE 暂存 dump |
| GM 软握手用 `GetValue` 轮询（非 volatile） | 双方都自旋到上界超时：连对方已写入的值也读成「从未写入」 | 两侧各自打印上界自旋计数与首次读值；读法改为 cache 无效/MTE 读回后立即恢复 |

# 验证方法

- 并发性判定：构造「AIC 独占 / AIV 独占 / 两者同时（数据互不相干）」三个独立任务，比对设备任务时长与
  各 pipe 占比 —— 同时跑的时长 ≈ 工作量长边单独跑的时长、且短边 pipe 时间与独占时相同，即判定为真并行；
  不要凭直觉或凭"同时发射"下结论。
- 原语探针：对同一原语构造「同形状、变 repeat / 变步长」的对照实验，逐配置给出同步返回码与 CPU 参考误差；
  覆盖不同归约宽度与分段归约。照文档写会静默出错的语义（归约落位、间隔槽内容、小 repeat 布局）先用它量清再用。
- 端到端判据：冻结用例集 + 双口径判定（比值门禁 + 绝对/相对容差），逐例留证据；
  结论以逐例证据表汇总，不用单一总体判读代替。[^sparse-attention-device-pipeline-and-reduce-facts-source]

[^sparse-attention-device-pipeline-and-reduce-facts-source]: 审计编号 sparse-attention-device-pipeline-and-reduce-facts-source；验证记录与探针输出由提取侧独立保管，不随生成侧资料分发。
