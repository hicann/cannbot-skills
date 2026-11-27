# 卷积反向算子公共语言

> **层级**：通用知识层（generic/），语言中立。
>
> **目的**：为输入梯度、权重梯度与转置卷积的设计、tiling、kernel、验证统一**两分支语义完全相同**的术语与记号。后续所有反向文档均以本表为准，避免同义异名造成设计歧义。
>
> **分工**：**分支专属记号下沉到分支语言**——input 分支（Conv3DBackpropInput / DX、Conv3DTranspose）见 [input/language.md](input/language.md)，filter 分支（Conv3DBackpropFilter / DW）见 [filter/language.md](filter/language.md)。两分支的 M/N/K、mValue/nValue/kValue、baseM/baseN/baseK、singleCore* **同名不同义**，一律以分支文件为准，不得混用。

---

## 0 核心记号速查（公共部分）

| 记号 | 等价写法 | 值 / 定义 |
|------|---------|----------|
| **M0 = m0** | — | 16（M 方向分形基粒度） |
| **N0 = n0** | — | 16（N 方向分形基粒度） |
| **C0 = c0 = K0 = k0** | — | `32B / dtypesize`（元素个数）：FP16/BF16 → 16（与 m0/n0 同值）、FP32 → 8 |
| **stepKa / stepKb** | — | 一份 L1A / L1B 驻留覆盖的 K 计算块数（两分支同义；模板专属细则见分支语言） |
| **五路 pbuffer** | al1/bl1/al0/bl0/cl0Pbuffer | L1A/L1B/L0A/L0B/L0C 的物理槽份数，取 1 或 2 |
| **kernel（实现语境）** | 内核 | 设备侧程序，与 tiling（host 侧）相对 |

> ⚠️ 布局记号 ND/DN/NZ/ZN 里的 "N"、"D"、"Z" 只是快轴/慢轴记号，与 Cube 的 M0/N0/C0 **完全无关**，详见 [input/compute-flow.md](input/compute-flow.md) 与 [filter/compute-flow.md](filter/compute-flow.md) 的格式与轴序。

---

## 1 张量命名（公共数学对象）

| 张量 | 别名 / 接口字段 | 语义 |
|------|---------|------|
| **X / Y** | 正向输入、输出 | 数学定义用（正向卷积的输入/输出），不作为生成入口的给定对象；正向 Y 与接口输出 y 分开 |
| **weight** | filter | 正向权重；数据图中的 wght 也指该对象 |
| **dedy** | out_backprop | 正向输出梯度（两分支共同的数值输入与 A 路来源） |

逻辑形状：X/dedx 为 `(N,Cin,Di,Hi,Wi)`，Y/dedy 为 `(N,Cout,Dout,Ho,Wo)`，weight 为 `(Cout,CinG,Dk,Hk,Wk)`。Cin/Cout 始终按正向卷积命名；代码字段保持 `cin/cout、cinG/coutG、di/hi/wi、dout/ho/wo、dk/hk/wk` 的实际拼写。

各分支的数值输入/输出与接口绑定（input：dedx、input_size、bias 及 Transpose 对象绑定；filter：x、dedw、filter_size）见分支语言。

## 2 维度命名（公共）

| 记号 | 含义 |
|------|------|
| **N** | batch 维 |
| **Cin / Cout** | 正向输入 / 输出总通道数 |
| **G；CinG、CoutG** | 分组数；`CinG=Cin/G`、`CoutG=Cout/G` |
| **Di/Hi/Wi** | dedx 侧深度、高、宽 |
| **Dout/Ho/Wo** | dedy 侧深度、高、宽 |
| **Dk/Hk/Wk** | 核采样点数（dk/hk/wk 为核坐标轴名） |
| **s_a、δ_a** | a 轴的正向 stride、dilation（strideD/H/W、dilationD/H/W） |
| **p_a^-、p_a^+** | 正向前、后两侧 pad（padHead/Tail、padUp/Down、padLeft/Right） |
| **ci / co** | 组内通道坐标；全局通道加 `groupIdx*cinG/coutG` |
| **kd/kh/kw** | 原 weight 的核坐标 |
| **Cout1/Cout0、Cin1/Cin0** | 通道块轴和块内轴；粒度由分支和存储阶段定义，坐标 co1/co0、ci1/ci0 与轴长分开 |

> ci/co 始终是组内坐标，换成全局通道需加组起点 `g×CinG` 或 `g×CoutG`。

### 2.1 空间维统一表达（1D/2D/3D）

| 场景 | 空间维 | 输入格式 | 升维视角 |
|------|--------|---------|---------|
| Conv1D 反向 | {W} | NCW / NWC | H=1 的 2D（即 D=1 的 3D） |
| Conv2D 反向 | {H,W} | NCHW / NHWC | D=1 的 3D |
| Conv3D 反向 | {D,H,W} | NCDHW / NDHWC | 原生 |

> 以三维问题设计和执行；1D/2D 规格按 [前置处理](preprocessing.md) §3 升维并保留轴语义（新增空间轴长度为 1，元素数与地址映射不变）。

## 3 卷积属性词汇（记号与正向范式对齐）

| 属性 | 记号 | 约定 |
|------|------|------|
| 步长 strides | `sD/sH/sW`（下标记号 `s_a`） | 批次和通道分量为 1；按张量轴序给出的先取空间分量（NCHW 系 4 元组取后两位） |
| 填充 pads | `padHead/padTail`（D）、`padTop/padBottom`（H）、`padLeft/padRight`（W）；数学简写 `p_a^-、p_a^+` | **头/上/左在前，尾/下/右在后**；3D 六元组 `[head, tail, top, bottom, left, right]`（2D 原始规格的四元组 `[top, bottom, left, right]` 由前置处理升维归一） |
| 空洞率 dilations | `dD/dH/dW`（符号 `δ_a`） | 卷积核采样点间隔 |
| 分组数 groups | `G` | 正整数；每组通道数为 CinG、CoutG，G 整除 Cin、Cout |
| data_format | NCDHW / NDHWC | 特征张量存储轴序；两分支的配对对象（dedy/dedx、x/dedy）与 weight/dedw 的独立选择见分支语言及场景准入 |

## 4 缓存层级词汇（公共层级与流水）

| 层级 | 公共语义 | SDK 位置 |
|------|---------|---------|
| **L1A** | L1 的 A 路驻留段（两分支 A 路均为 dedy），供 MTE1 多次抽取复用 | `TPosition::A1` |
| **L1B** | L1 的 B 路父段（input=weight 父布局 / filter=x 行带，内容见分支语言） | `TPosition::B1` |
| **L0A** | A 路操作数，M×K，NZ | `TPosition::A2` |
| **L0B** | B 路操作数，K×N，ZN | `TPosition::B2` |
| **L0C** | M×N 部分和，NZ，FP32 | `TPosition::CO1` |
| **父布局** | L1 本次驻留段的实际形状和步幅，子块按父布局寻址 | — |
| **槽 / slot** | 一份物理存储；aSlot/bSlot/l0Slot/cSlot 分别轮转 | — |
| **al1/bl1/al0/bl0/cl0Pbuffer** | 五路物理槽份数，1 或 2 | — |

GM→L1 由 MTE2 完成，L1→L0A/L0B 由 MTE1 完成，矩阵乘加用 M，L0C→GM 用 F（Fixpipe）。SDK 中对应的流水枚举仍是 PIPE_FIX，队列事件仍使用 M_FIX/FIX_M；MTE3 用于适配层明确采用的辅助步骤。

元素偏移用于 Tensor 下标；`*Bytes`、容量和槽基址使用字节；分形步幅另标单位。数值输入按 dtype（FP16/BF16 为 2B、FP32 为 4B），累加与 FP32 输出恒 4B（accBytes）。

`L1Bytes/L0ABytes/L0BBytes/L0CBytes/UBBytes/BTBytes` 表示本次可用资源容量，由适配层根据目标芯片版本、平台信息与 SDK 确定（类型与资源映射见 [buffer-and-types](../adapters/ascendc/buffer-and-types.md)）。Buffer 需求由形状、元素字节数和份数计算；Host 预算、Kernel 槽跨度和地址边界使用同一份容量记录。

各 Buffer 的分支内容与阶段名称（LoadToA1/LoadToA2/MmadLocal…、LoadL12L0a/ExecuteMmad…）见分支语言及分支 compute-flow 的阶段表。

## 5 格式词汇

详细排布图与偏移公式见 [input/compute-flow.md](input/compute-flow.md) 与 [filter/compute-flow.md](filter/compute-flow.md) 的格式与轴序，此处只列术语：

| 格式 | 一句话定义 |
|------|-----------|
| NCDHW / NDHWC | 特征张量的两种存储轴序：通道在外 / 通道在内 |
| DHWCN | weight/dedw 的推导视角（dedy 通道连续），仅用于比较排布 |
| ND | 逻辑矩阵中空间点的通道连续 |
| DN | 通道的空间点连续 |
| NZ | 最内轴按分形块排：块间列优先、块内行优先 |
| ZN | 块间行优先、块内列优先 |

> NZ/ZN 必须说明相对哪一个矩阵，不能仅凭缩写推断通道朝向。

## 6 数学记号

| 记号 | 定义 | 示例 |
|------|------|------|
| `alignUp(x,a)` | 向上对齐到 a 的倍数 | alignUp(35,16)=48 |
| `alignDown(x,a)` | 向下对齐 | alignDown(35,16)=32 |
| `ceilDiv(x,a)` | 向上取整除法 | ceilDiv(35,16)=3 |
| `gcd` | 最大公因数（绑核候选因子生成） | gcd(12,18)=6 |

字段改动时，同时核对其生产者、消费者和单位。

## 7 与外部知识的术语对接

| 外部知识 | 路径 | 对接说明 |
|---------|------|---------|
| 精度标准 | `cannbot-skills/ops/ops-precision-standard` | atol/rtol/ptol 与混合容差的系统性定义 |
