# 卷积反向算子类型与 Buffer

本页提供平台资源获取方式，并保留 input 的数值组合和物理槽说明。分支的几何、任务与 API 知识分别从 [input 适配索引](input/index.md) 和 [filter 适配索引](filter/index.md) 进入。

§1.1–§1.2 的平台资源获取方式由两分支共用。本页其余具体操作数、类型组合、槽尺寸和能力表沿用 input 定义。filter 绑定 A=dedy、B=x、输出 FP32 dedw，使用 [filter 适配入口](filter/index.md) 中的 Bound、搬运和同步契约；不能将 input 的权重父段或输出类型直接代入 filter。

## 1 获取 platformInfo

### 1.1 绑定目标与 SDK

能力记录绑定目标架构、SDK、头文件和编译配置。通过 platformInfo 获取资源，再核对当前 API 组合的支持条件。

| 获取项 | 公共接口 | 本地字段 |
|---|---|---|
| 核数 | PlatformAscendC 的 GetCoreNumAic/GetCoreNumAiv | maxCores、vectorCores |
| 目标标识 | 所用 SDK 的 SocVersion/架构查询 | target、npuArch |
| 片上容量 | GetCoreMemSize | L0ABytes、L0BBytes、L0CBytes、L1Bytes、UBBytes |
| 系统 workspace | 所用 SDK 的平台 workspace 查询 | systemWorkspaceBytes |

部分求解规则将 L0A/L0B 作为相同容量处理；本地记录分别保存两者，使用这类规则时检查其适用关系，不只查询 L0A 就推定任意平台的 L0B。

### 1.2 注册与直调入口

注册侧从 context 取得 platformInfo，以下片段直接使用公共 API；cap 为本工程能力对象：

```cpp
auto *platformInfo = context->GetPlatformInfo();
if (platformInfo == nullptr) return ge::GRAPH_FAILED;
platform_ascendc::PlatformAscendC platform(platformInfo);
platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, cap.L0ABytes);
platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, cap.L0BBytes);
platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, cap.L0CBytes);
platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, cap.L1Bytes);
platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, cap.UBBytes);
```

直调从目标 SDK 的平台管理入口取得相同资源，再调用公共求解器。查询失败或必要资源为零时给出具体错误。核数、API 重载、初始化配比、编译目标和最大地址形成同一份能力记录。

实现中用到的 __NPU_ARCH__、ASC_DEVKIT_VERSION_NUM 与初始化编译分支按目标 SDK 确认。平台查询与本地求解分别实现。

### 1.3 能力判定

platformInfo 提供目标与资源信息；API、布局和事件约定结合所用 SDK 的头文件及编译结果确认。生成工程保存同一份能力记录，不以固定 SoC 名称列表决定准入。

下表为公共 input Kernel 的能力基线。完整 N/K 驻留仍使用相同类型、TQue 和资源检查，区别仅在 L1B 持有范围。

| 能力项 | 确认内容 | 消费位置 |
|---|---|---|
| 资源与数值 | AIC 核数与片上容量；三路同 dtype（FP16/BF16/FP32），FP32 累加；M0/N0=16，C0/K0 随 dtype 取 16/8 | Tiling 起点、预算与槽分配 |
| GM→L1 | 所需 ND2NZ/DN2NZ 重载、目的间隔写入及所选 Fill 通路可用 | L1A 插零、L1B 父布局 |
| L1→L0 | Load3D 窗口及 Load2D 转置/逆序步幅满足所选参数范围 | L0A/L0B；weight 变换按最终 baseN 确定 |
| MMAD→F | FP16/BF16/FP32 输入、FP32 部分和；FP16/BF16 输出做类型转换、FP32 直出（NoQuant），支持所选 ND/DN 写回 | MmadLocal、LoadL0c2Gm |
| FP32 计算模式 | `enable_hf32=false、hf32Flag=0`，Init 使用 `SetHF32Mode(false)` | 普通 FP32 MMAD |
| 可选 bias | BTBytes、FP32 GM→L1→BT 搬运、带 BT 的 Mmad 及 C2 队列；按目标 SDK 绑定 | Transpose 的 bias 预算、分配和首次乘加 |
| 同步与初始化 | TPipe 固定事件初态/收尾及队列交接与本模板一致；初始化方式与启动类型配对 | [同步实现](input/sync-implementation.md#12-初态与收尾的所有权) |
| input C/V 与 Gather | 目标 MIX 配比、scalar 执行形态、VReg/掩码预留、IndexT 及可寻址范围 | [C/V 协同](input/cv-collaboration.md)、[数据准备](input/instruction-mapping.md#34-数据准备三分支与字段) |

原选块规则还要求 L0A/L0B 容量相同、两路输入槽份数相同。几何准入、目标能力、最终计划分别检查；API 窄字段和实际访问范围在选块后复核。能力记录包含所选 API 重载、宏分支、参数上限及编译验证结果，不能仅由容量推定指令可用。

Load3D 的 Repeat 设置与 LoadData 调用按同一 SDK 分支选择，见 [指令映射](input/instruction-mapping.md#42-fmatrix补边与目的分形步幅)。Fixpipe 的参数类型与 ROW_MAJOR/COLUMN_MAJOR 重载配对。平台目标、头文件与设备编译目标保持一致，具体型号只写入运行证据。

`ApplyBalancedMSplit` 的能力标志由适配层单独确认：目标支持当前 BF16/NCDHW 通路，且改变 singleCoreM 后的非整行起点、Load3D 参数及写回范围已经通过检查。该标志为平台/SDK 组合的适配结论，不是 platformInfo 中可直接读取的布尔字段。未确认时保留 standard 已有计划；几何与业务条件仍按 [standard](../../generic/input/tiling/templates/standard/tiling.md#21-一维退化问题的-m-均衡) 检查。

## 2 类型与资源对象

本节及第 3 节的 input 队列和槽公式用于普通三模板及其扩展。N/K 全载时仍按公共 L1B 父布局和 stepKb*baseK 分配一份驻留槽；L1A、L0A/L0B 与 L0C 的份数分别由容量和流水决定。

Kernel 使用 `kernel_operator.h`，以下设备类型与接口位于 `AscendC` 命名空间。FP16 对应 `half`，BF16 对应 `bfloat16_t`，FP32 对应 `float`（SrcT/DstT 同类型，Fixpipe 无 dtype 转换）；累加类型恒为 `float`。设备片段中的几何与分块变量来自所选模板，参数范围先检查、再填入 API 结构。

| 对象 | 类型与布局 | Ascend C 存储位置 |
|---|---|---|
| dedy、weight | FP16/BF16/FP32，按各自 NCDHW 或 NDHWC 规格存储 | `GlobalTensor<T>`，GM |
| L1A | dedy 通道分块驻留片段，供 fmap 窗口提取 | `TPosition::A1`，L1 |
| L1B | 由权重准备分支确定的父布局 | `TPosition::B1`，L1 |
| L0A | fmap 窗口，NZ 的 M×K 片段 | `TPosition::A2`，L0A |
| L0B | `weight_rot`，ZN 的 K×N 片段 | `TPosition::B2`，L0B |
| `L0C` | FP32，NZ 的 M×N 部分和 | `TPosition::CO1`，L0C |
| dedx | 与数值输入同 dtype，NCDHW 或 NDHWC | `GlobalTensor<T>`，GM |
| bias L1 / BT，可选 | FP32，通道连续，长度按 64B 对齐 | `TQue<TPosition::A1,1>` / `TQue<TPosition::C2,1>` |

本映射的类型参数如下；一个 L0 输入分形恒为 512B，L0C 分形为 1024B。

| 输入/输出 dtype | 设备类型 T | dedyBytes / weightBytes / dedxBytes | C0=K0 | c0BitsA/B | M0 / N0 | IndexT |
|---|---|---|---|---|---|---|
| FP16 | half | 2 | 16 | 4 | 16 / 16 | uint16_t |
| BF16 | bfloat16_t | 2 | 16 | 4 | 16 / 16 | uint16_t |
| FP32 | float | 4 | 8 | 3 | 16 / 16 | uint32_t |

`accBytes=4`。IndexT 用于 group enlarge 的 Gather；C04 使用前两种类型。VecTranspose 的 Cin0 固定为 16，按类型设置转置地址表和 32B 数据块。

本页的具体通路使用 `Nd2NzParams/Dn2NzParams`、`LoadData3DParamsV2`、`LoadData2DParamsV2`、`MmadParams`，Fixpipe 片段使用 SDK 类型 `FixpipeParamsArch3510`。这是 API 标识，保留其准确拼写；适配层按第 1.3 节选择匹配的实现与重载。容量、类型、接口重载及布局组合共同构成能力判定。

| 对象 | 实现形式 | 分配依据 |
|---|---|---|
| L1A、L1B | `TQue<TPosition::A1,2>`、`TQue<TPosition::B1,2>` | `al1Pbuffer/bl1Pbuffer` 决定实际分配一份或两份 |
| L0A、L0B | `TBuf<TPosition::A2>`、`TBuf<TPosition::B2>` | 按模板划分固定槽，成对轮换 |
| L0C | 每槽一个 `TQue<TPosition::CO1,1>` | `cl0Pbuffer` 决定槽数，按完整输出块交接 |
| 资源管理 | `TPipe` | 管理 Buffer 与队列所需事件 |

`TPipe::InitBuffer(que,num,len)` 中 `len` 为每份 Buffer 的字节数。需要落实模板的 L1 半区放置时，使用指定地址的重载：`InitBuffer(que, Std::make_tuple(offsetBytes,lenBytes), ...)`。元组数量与实际槽数一致；TQue 的模板深度、物理槽数、每份字节数分别记录。

同一份物理布局同时约束 Host 和 Kernel：槽基址按 32 字节对齐，同时存活的 L1A/L1B 区间互不重叠，最大末地址不超过 `L1Bytes`。L0A/B/C 也分别检查槽跨度、数据占用和最大访问地址。Kernel 分配与 Host 求解共用平台容量；采用编译期参数时，按本次芯片版本和 SDK 能力绑定，并与 platformInfo 核对。

## 3 物理槽划分

### 3.1 L1

单份 L1A 的预留量按当前任务可覆盖的最大高宽和通道段计算，L1B 按 `baseN*stepKb*baseK*weightBytes` 预留。Host 的跨窗口预算还按公共 Tiling 的 copyLines 检查搬运范围。实际父段与预留量分别保留。

令 `aAligned=alignUp(aSlotBytes,32)`、`bAligned=alignUp(bSlotBytes,32)`。无 bias 时 biasAlloc=0；有 bias 时使用 [公共 Tiling](../../generic/input/tiling/tiling.md#51-可选-bias-的候选预算与实际分配) 的最终 biasAlloc：

| 槽 | 字节起点 |
|---|---|
| L1A ping | 0 |
| L1B ping | aAligned |
| bias L1，存在时 | aAligned+bAligned |
| pongBase | max(L1Bytes/2,aAligned+bAligned+biasAlloc) |
| L1A pong，使用时 | pongBase |
| L1B pong，使用时 | pongBase+aAligned |

只为实际份数分配。即使 L1A 单份而 L1B 双份，L1B pong 仍按上述地址处理；检查真实末地址，不只检查数据量总和。

```text
0          aAligned   aAligned+bAligned  aAligned+bAligned+biasAlloc
+----------+----------+------------------+
| L1A ping | L1B ping | bias（可选）     |
+----------+----------+------------------+

……（间隔到 pongBase）

pongBase             pongBase+aAligned
+--------------------+------------------+
| L1A pong（可选）   | L1B pong（可选） |
+--------------------+------------------+   … L1Bytes

仅当对应 al1Pbuffer / bl1Pbuffer 为 2 时才分配 pong 槽。
```

指定地址的 InitBuffer 重载与元组参数必须由目标 SDK 支持；若更换分配方式，保留等价区间和同步责任。

bias 只有一份 L1 槽，BT 使用独立存储，分别检查 biasAlloc。BTBytes 按目标 SDK 的查询或能力定义取得并记录依据；不从 L1 容量推定 BT 容量。无 bias 时不创建两路 bias 队列资源，物理布局退化为原五路方案。具体搬运与尾补齐见 [bias 通路](input/bias-bt-path.md)。

### 3.2 L0

L0A/L0B 用 TBuf 分配各自整块容量，两路双 Buffer 时按半区成对选择；单份时使用起点 0。L0C 每份是独立 CO1 队列，双份各占半区：

```text
L0A/B：slotBase = slotId * (L0A/BBytes / inputSlotCount)
L0C  ：slotBase = cSlotId * (L0CBytes / cl0Pbuffer)
```

InitL0TBuf 的 CO1 队列深度为 1，每队列分配一份；份数由独立队列个数表达。L1A/L1B 队列深度 2 不等于始终分配两份存储。

## 4 校验

五路槽按各自元素字节数、对齐跨度和份数计算容量。Fixpipe 的源通道块为 16 个 float，即 64B。Buffer 计划确定后，继续实现 [同步](input/sync-implementation.md) 与各搬运参数。

带 bias 时追加 L1/BT 的 64B 覆盖检查，并验证 baseN 与 singleCoreCin 不同、GM→L1 只补到 32B 的尾部场景。资源计划与设备实际分配使用同一存在性和尺寸。
