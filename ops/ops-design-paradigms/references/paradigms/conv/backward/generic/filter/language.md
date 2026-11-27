# 卷积反向 filter 分支：分支记号

> **层级**：通用知识层（generic/filter/），语言中立。
>
> **定位**：本文件收录 **filter 分支（权重梯度）专用记号**：核心记号速查、M/N/K 与 GEMM 映射、切分与执行坐标、状态与执行判定。公共术语（张量、轴、卷积属性、数学工具）见 [../language.md](../language.md)，规格绑定见 [preprocessing §8](../preprocessing.md#8-filter-分支的规格归一)。命名与 Host/Device 共享的 TilingData 字段对齐。
>
> 与 input 分支的 M/N/K **同名不同义**（input：M=dedx 空间、N=dedx 通道、K=coutG×核窗口；filter：M=dedy 通道、N=x 通道×核窗口、K=输出平面），两侧文档不得混用记号。

---

## 1 核心记号速查

下表及第 2–3 节采用 MN_StreamK 的窗口和任务组织；Winograd、DLoad 的差异分别在第 1.2、1.3 节定义。

| 记号 | 等价写法 | 值 / 定义 |
|------|---------|----------|
| **M** | `mValue = cout1_g·C0` | 组内 dedy 通道（coutG）对齐 C0=16；**输出块第一维**；`cout1_g=ceilDiv(coutG,C0)` 为组内 dedy 通道分形数 |
| **N** | `nValue = hk·wk·cin1_g·C0` | 组内 x 通道 × 核高宽窗口；**输出块第二维**（与 batch 维 N 区分，由上下文确定）；`cin1_g=ceilDiv(cinG,C0)` |
| **K** | `kValue = ho·wo` | 单层输出平面（dout 内），归约维 |
| **K 之外的独立归约维** | batch×dout | 基本块内分段进 L0C 累加，多段 Fixpipe 原子累加出 GM |
| **A 路 / B 路** | — | **A = dedy**（`[M,K]`，直装不窗口化）；**B = x**（`[K,N]`，Load3D 窗口化） |
| **basicBlockIdx 任务四元组** | `(mIdx, nIdx, dkIdx, gIdx)` | 一个基本块 = M 块 × N 块 × 一个核深度 × 一个组；totalCnt=mCnt·nCnt·dk·G |
| **reduceSegmentsND / reduceSegmentsH** | — | batch×dout / ho 的精度分段长度（累加轴阈值 `SINGLE_BLOCK_ADD_THRESHOLD=65536` 触发） |
| **大核拆分** | `isSplitKernelHW` | kh/kw/dilation/stride/pad 超限或 `kh·wk·C0>65536` 时的核窗口拆分路径 |
| **x_win** | fmap 窗口视图 | x 按正向 pad 补零、stride/dilation 采样形成的窗口矩阵（Load3D 语义） |
| **dedy_flat** | 输出网格视图 | dedy 按输出平面 (ho,wo) 展平后的 `[M,K]` 矩阵（无插零、无反向边界） |

### 1.1 张量与格式（filter 分支）

本分支的张量绑定（在公共 [../language.md](../language.md) §1 的 X/Y/weight/dedy 之外）：

| 张量 | 别名 / 接口字段 | 语义 |
|------|---------|------|
| **x** | fmap | 正向输入（数值输入，与 dedy 配对；只读参与窗口采样） |
| **dedw** | 输出 | FP32 权重梯度；format 独立三选一 NCDHW / NDHWC / DHWCN |
| **filter_size** | 规格属性 | 目标 dedw 形状，按输出格式解读；与 dedw 规格同时给出时应一致 |

格式配对约束：x 与 dedy 同为 FP16、BF16 或 FP32，并同为 NCDHW 或同为 NDHWC；dedw 独立选择三种输出格式（准入见 [scene.md](scene.md) §1）。

### 1.2 Winograd 模板记号（变换域求导）

仅 [winograd 模板](tiling/templates/winograd/tiling.md) 使用，与 §1–§2 的 im2col 记号**互斥**（一个规格只命中一种求解形态）：

| 记号 | 含义 |
|------|------|
| **tile / tileH×tileW** | 输出平面按 2×2 划分的求导 tile；`tileH=ceilDiv(ho,2)`、`tileW=ceilDiv(wo,2)`，每 tile 覆盖 4×4 感受野 |
| **16 个变换坐标** | 正变换后的独立乘加位置 `p∈[0,16)`；每点一次独立 GEMM：M=co、N=ci、K=tile 数 |
| **residentTarget** | 驻留侧张量选择：**较小通道的张量驻留**（ci>co 驻留 dedy，否则驻留 x）；驻留侧变换一次跨块复用，流式侧逐块变换 |
| **watermarkResidentC** | 驻留水位线：当前 kd 内已驻留通道上界，切换 kd 清零；`residentCBound > watermark` 时本块需补驻留变换 |
| **rtTiling.singleShapeCin / singleShapeCout** | Cin 固定 64，Cout 默认 64，按核数条件依次尝试 16/32/48；驻留写入与块迭代**必须同源**（错位后果见适配层） |
| **kGroups / kGroupIdx** | 尾轮切 K 的组数与本组编号；组间经交织写回合并 |
| **cuttableK** | 可切单位数 `dout*batch*ceilDiv(tileH,th)`；一个单位走完全部 W 块，尾轮再按可切范围分组 |

### 1.3 DLoad 模板记号（单批空间驻留）

| 记号 | 含义 |
|---|---|
| baseM / mCompute（coutAlign16） | Cout 块上限 / 当前有效 Cout 向 16 对齐的计算跨度 |
| cinBlock / nCompute（cinAlign16） | `baseN/(hk*wk)`，Cin 块上限 / 当前有效 Cin 向 16 对齐的跨度 |
| baseN / N 排列 | `cinBlock*hk*wk`；矩阵列按 `(tap,Cin1,Cin0)` 排列，`tap=khIdx*wk+kwIdx` |
| baseK / kl0HoWo | 一次消费的输出平面窗口长度上限，按 16 对齐 |
| 任务 | 一个 Cout 块 × Cin 块，拥有全部 dk/hk/wk 输出，并完整归约 batch、dout、ho、wo |
| L0C 核深度段 | 每个 dk 独立保存 `mCompute*nCompute*hk*wk` 个部分和；完成归约后统一写回 |

两路输入仍为 dedy 与 x，采样保持正向核坐标；完整驻留和输出块所有权见 [DLoad Tiling](tiling/templates/dload/tiling.md)。

## 2 M/N/K 与 GEMM 映射

本节采用 MN_StreamK 的数学 16 通道窗口；FP32 的物理 channelSize/k0 为 8，适配层按物理块重新排布，不能直接复用两字节输入的地址公式。

```
dedy (N,Cout,Dout,Ho,Wo)              x (N,Cin,Di,Hi,Wi)                     dedw (Cout,CinG,Dk,Hk,Wk)
    |                                     |                                      ^
    | 展平输出平面（dedy_flat）           | 正向 pad + stride/dilation           | FP32 输出
    v                                     v（x_win，Load3D 窗口化）              |
A 矩阵 [M, K]                     ×   B 矩阵 [K, N]                      =   Y 矩阵 [M, N]
行 = 组内 co                          列 = (ci1,kh,kw,ci0)                   行×列 = 一个 dedw 输出块
列 = 展平的 (ho,wo)                   行 = 同一 (ho,wo)
```

| 记号 | 定义 | 说明 |
|------|------|------|
| **M** | `mValue = cout1_g·C0` | 组内 dedy 通道对齐 16；256 为初始选块目标 |
| **N** | `nValue = hk·wk·cin1_g·C0` | 有效长度 `cinG·hk·wk`；任务 singleCoreN 保持完整通道窗口，L0 baseN 按 C0 对齐，可使用非整窗块 |
| **K** | `kValue = ho·wo` | 每层输出平面；baseK 由 L0A/L0B 容量推出并按 16 对齐，优先对齐 splitWo |
| **核深度 kd** | 任务轴 | 不混入 K/N；每个基本块固定一个 dkIdx（cin 与 dk 分轴） |
| **batch×dout** | 核内归约轴 | 基本块内沿 batch·dout 分段，段内 L0C 累加，段间 Fixpipe 原子累加出 GM |

x 的采样坐标（保持正向几何；`s_a/δ_a/p_a^-` 为 [../language.md](../language.md) §3 的公共属性记号）：

| 方向 | 采样公式 | 说明 |
|------|---------|------|
| H 向 | `hi(kh) = oh·s_h + kh·δ_h − p_h^-` | 越界（pad 区）时 x 取 0 |
| W 向 | `wi(kw) = ow·s_w + kw·δ_w − p_w^-` | 同上 |
| D 向 | `di(kd) = od·s_d + kd·δ_d − p_d^-` | 归约段的全部 od 均无效时该段零贡献、可跳过 |

这条映射是 x_win 窗口化（Load3D 配置）与 golden 生成的共同数学依据；窗口提取见 [compute-flow](compute-flow.md) §4.3，模型推导见 [scene](scene.md) §2.2。

## 3 切分与执行坐标

三个尺度各管一层（与 input 分支的任务/父段/基本块三层同构）：

```
切分对象：
+-----------------------+   +-------------------------+   +---------------------------+
| 任务级（核间分配）    |   | L1 父段级（驻留复用）   |   | L0 基本块级（一次 MMAD）  |
| singleCoreM/N/K       |   | stepKa / stepKb         |   | baseM / baseN / baseK     |
| group × kd × M × N    +-->+ 一份 L1A/L1B 驻留覆盖的 +-->+ 一次乘加的 M×N；每次消费  |
| 一个输出块的完整归约  |   | K 块数；kaIdx/kaStepIdx |   | baseK；尾块按剩余量收缩   |
| 任务四元组解码见      |   | kbIdx/kbStepIdx 定位    |   | baseUseM_/N_/K_ 为本轮    |
| compute-flow §6       |   | 换驻与父段内寻址        |   | 实际计算长度              |
+-----------------------+   +-------------------------+   +---------------------------+
```

| 记号 | 含义 |
|------|------|
| baseM / baseN / baseK | L0 基本块（TilingData 字段 baseM/baseN/baseK）；M/N 初始目标不超过 256，最终受容量及任务调整约束，按 C0 对齐；普通整窗判定为 `(baseN/C0) % (hk*wk) == 0`，其余使用非整窗通路 |
| singleCoreM / singleCoreN / singleCoreK | 一个基本块任务的 M/N/K 范围（`singleCoreM=baseM`、`singleCoreN` 为 cin1_g 的完整核窗口数×C0、StreamK 前 `singleCoreK=kValue`，HWout 分核后保存 H 段长度） |
| singleCoreCout / singleCoreCin | 任务在 co / ci 方向的通道跨度 |
| singleCoreBatchDout / singleCoreHo | batch·dout 基值 / singleCoreK 折算的输出行数 |
| mCnt / nCnt / totalCnt | M 块数 / N 块数 / 基本块总数 `mCnt·nCnt·dk·G` |
| coreNum / usedCoreNum | 共享调度基数 / Host 计算的启动 blockDim；任务按 coreNum 轮转，usedCoreNum 不序列化 |
| calRound / streamkType | type=0：ceilDiv(totalCnt,coreNum)；type≠0：totalCnt/coreNum，余下尾轮按 K 切分 |
| stepKa / stepKb | L1A（dedy）/ L1B（x 窗口父段）驻留覆盖的 K 块数；取彼此整数倍 |
| k | 当前 ComputeLoop 内的全局 K 块号，`kStart=k*baseK`；每次进入该 K 循环从 0 开始 |
| kaIdx / kbIdx | 父段内 K 块号，分别为 `k%stepKa`、`k%stepKb`；用于父段内寻址与换驻判断 |
| kaStepIdx / kbStepIdx | 父段号，分别为 `k/stepKa`、`k/stepKb`，均取整数除法；父段起点为各自父段号乘 step 与 baseK |
| depthA1 / depthB1 | L1 配比深度（求解器内部）；16 起按 2 阶梯衰减，≤2 时双缓冲降为 1（发布为 al1Pbuffer/bl1Pbuffer） |
| splitWo / splitWi / tailWo / tailWi | 输出宽分段长度（wo>512 时按 128 分段）及对应输入宽 / 尾段（发布字段仅 splitWo，其余为派生量） |
| al1Bound / bl1Bound | L1A / L1B 的驻留上界（元素数） |
| isSplitKernelHW | 大核拆分开关（核窗口超过 Load3D/搬运字段上限） |
| iterateOrder | `mValue>nValue` 时为 1，否则 0；控制任务内 M/N 子块遍历与父段复用。任务号仍按 group→kd→N→M（快→慢）解码 |
| singleShapeK / singleShapeBatch_ | 运行时段内 K 长度（`min(ho−hoStart, reduceSegmentsH)·wo`）/ 段内 batch·dout 长度（kernel 运行量，区别于 TilingData 字段） |

## 4 状态与执行判定

| 记号 | 含义 |
|------|------|
| SINGLE_BLOCK_ADD_THRESHOLD | 累加轴阈值 65536；`batch·dout·ho·wo` 超过它则对 batch·dout 分段 |
| initOutput | MN_StreamK 的 AIV 输出预清零阶段（Fill 清零 dedw GM，AIC WaitFlag 等待）；原子累加写回的前置义务 |
| needCompute / invalidDIndex | 段内 dout 全部落在 pad 外时该段无有效贡献，跳过计算（对应"无贡献核位置"的执行侧表达） |
| ReCalDkCinSingleCoreShape | 按 dout 段与 group 尾块收缩当前段 singleCoreM/N 有效范围的运行时重算 |
| l0cPingPongFlag_ | L0C 双缓冲乒乓标志，M/N 子块迭代间翻转 |
| offsetA_ / offsetB_ / offsetC_ | 当前段三路 GM 偏移；每段重算，batch/dout/ho 变化影响 A/B，不影响 C |
