# 卷积反向权重梯度分支：验证方法

本文给出用例、输入、Golden、比对、定位和范式回写，按 filter 分支自身语义验证。计算依据见 [scene](scene.md)、[compute-flow](compute-flow.md) 与 [Tiling](tiling/tiling.md)，API 检查见 [适配层](../../adapters/ascendc/filter/index.md)。

## 1 验证闭环总览

```text
          +-------------------------------------------+
          | case design + declared comparison policy  |
          +---------------------+---------------------+
                                |
                                |
                                v
           +--------------------+--------------------+
           | quantized x / dedy + independent Golden |
           +--------------------+--------------------+
                                |
                                |
                                v
     +--------------------------+--------------------------+
     | Host fields / actual device execution / output dump |
     +--------------------------+--------------------------+
                                |
                                |
                                v
+-------------------------------+-------------------------------+
| shape -> finite values -> elementwise comparison -> coverage  |
+-------------------------------+-------------------------------+
                                |
                                |
                                v
+-------------------------------+-------------------------------+
| locate failure -> update knowledge -> regenerate and regress  |
+---------------------------------------------------------------+
```

先证明数学和布局，再验证 Host 选择、设备实现和整体调用。正确性通过与分支命中分别记录；文档算例通过不代替设备运行，设备精度通过也不替代目标路径覆盖。

## 2 验证范围与用例设计

### 2.1 验证范围声明

| 维度 | 本分支约定 |
|---|---|
| 场景 | Conv1/2/3DBackpropFilter，低维规格升维后统一验证 |
| 数值 | x/dedy 同为 FP16 或 BF16，dedw 为 FP32 |
| 布局 | x/dedy 同为 NCDHW 或 NDHWC；dedw 独立为 NCDHW/NDHWC/DHWCN |
| 分组 | 按实际 CinG/CoutG 独立计算，包括通道尾块 |
| 路径 | MN_StreamK、streamkType=0，普通/拆核与整宽/切宽按最终计划 |
| 执行 | 实际输出预清零，段内 L0C 累加、段间 GM 原子累加 |

### 2.2 用例设计三层次

| 层次 | 目的 | 必要证据 |
|---|---|---|
| L1 属性基础集 | 形状、布局、dtype、卷积连接 | 独立 Golden、有效输出及零贡献位置 |
| L2 Tiling 状态覆盖 | 选块、驻留、切宽、拆核和尾块 | 实际字段、命中分支和正确性 |
| L3 规模边界 | 长归约、资源临界和任务轮转 | 资源、越界、同步、数值分布与重复执行 |

### 2.3 L1 属性组合矩阵

| 因子 | 取值与检查 |
|---|---|
| 核 | 1×1×1、1×2×3、2×2×2、3×3×3、普通字段边界两侧 |
| stride/dilation | 全 1、混合 1/2、各空间轴不同 |
| pads | 0、对称、非对称、存在无贡献核位置 |
| 通道 | 1/15/16/17/31/32，实际通道与补齐域分别检查 |
| groups | 1、多组；给不同组分配不同数值特征 |
| rank | 1D/2D/3D，同数学问题升维前后对应 |
| 输出格式 | 对同一逻辑问题分别输出三种格式，转回逻辑坐标后相同 |
| dtype | 两路同为 FP16 或 BF16；先量化输入再求 Golden |

例如 x=(1,1,1,1,1)、核宽=3、左右 pad=1，合法输出宽为 1；两个边缘核位置的梯度为零。这个合法零贡献例子与“规格超接口核尺寸上限被拒绝”分别验证。

### 2.4 L2 Tiling 状态覆盖

| 状态 | 触发特征 | 验证点 |
|---|---|---|
| 9 点窗口优选 | hk*wk=9 | baseN 起点、回调与实际 N 布局 |
| 非整窗 baseN | `(baseN/16) % (hk*wk) != 0` | Load3D 起点、head_/tail_、分片写回 |
| stepKa 与 stepKb 不同 | 两路父段覆盖长度不同 | 同一 K 配对、换驻与末轮 |
| L1 单/双份切换 | 阶梯搜索触及容量边界 | 最终份数、Bound 与物理地址 |
| 宽片及尾宽 | wo 超阈值、wo%splitWo 非零 | 原行步幅、实际父跨度和尾片 |
| 拆核 | 原窗口参数超过普通字段范围 | 原 kh/kw 采样、单位窗口与 L0C 子片段 |
| 长归约分段 | 跨 65536 选择阈值 | 段内初值、完整归约与原子合并 |
| 无效深度段 | padFront/dilationD 产生全无效或部分无效 id | 跳过合法、首次有效贡献不丢失 |
| 轮转与空任务核 | totalCnt 小于核数、大于核数且不整除 | 唯一归属、初始化通知和尾轮 |
| 组内输入通道尾块 | CinG=17 且 dk>1 | nSize 不多写相邻 ci/kd |

### 2.5 窗口、搬运与写回的联动检查

以下小尺寸用于函数和布局模型验证；真实生成工程由 Host 自然选择分支，保存最终 key、isSplitKernelHW、splitWo、base*、step* 与槽份数。函数参数示例不能代替自然命中记录。

| 子问题 | 输入或状态 | 必须核对的结果 |
|---|---|---|
| 宽片交叠与尾宽 | wi=25、wo=13、wk=3、strideW=2、两侧 pad=1、splitWo=8 | 两片真实 x 列为 `[0,16)`、`[15,25)`；原行步幅不变，尾片 K 状态重算 |
| B 父段裁剪 | hi=7、ho=4、hk=3、strideH=2、padUp=1；W=8、baseK=16、stepKb=1 | 两父段真实行 `[0,4)`、`[3,7)`，顶部 pad 为 1、0；各槽恢复自身元数据 |
| 父段内起点 | K 块从非整行起点开始，且 stepKa!=stepKb | 两路同一 K 对应相同 oh/ow，mStartPt 与 A 行内余数分别正确 |
| 单个有效 co | tailM_=1，实际 K=31 | MTE2 只读一个 co，计算 M 暂扩为 16，CopyOut 恢复 M=1 |
| 拆核采样 | x 平面 5×6、输出 3×3、核 3×3、stride=2、pad=1，固定 hkIdx=wkIdx=0 | 原采样坐标 `[-1,1,3]`；真实 2×2 父网格加采样 pad，等于直接 x 采样 |
| 拆核 L0C 子片段 | baseM=32、三个 N0=16 片段 | float 偏移为 0、512、1024；各片段首次贡献独立，无贡献片段保持零 |
| 拆核 M 尾部跨度 | baseM=144、有效 M=113 | N0 片段仍相隔 144*16 个 float；Fixpipe 按同一物理跨度取源，只写真实 113 个 co |
| 父段跨 M/N 复用 | ORDER_M/ORDER_N；K 块数在 step*Pbuffer 两侧取值；批深度 1/2 | 对照 UpdateMNIdx 的重载、保留和归还标志；重复消费不重复 DeQue |
| 非整窗写回 | hw=4、baseN=48、singleCoreN=64、真实 ci=1 | 核区间依次 `[0,3)`、`[3,4)`；三种格式均只写该真实 ci |
| 写回状态重放 | 同一 N 被多个 M 使用，以及 N 回绕 | nLoopHead_/nLoopCinRemainLen_ 恢复到该 N 的起点，新任务/段重新初始化 |

搬运验证用唯一编号的 x/dedy。先按原张量偏移取得数值，再独立枚举 `(n,od,oh,ow,ci,kh,kw)` 得到数学窗口；比较实际父布局与 MTE1 后的每个 A/B 元素。拆核与普通窗口使用相同数学坐标核对，补零点、有效点和对齐槽分别记录。

写回验证从编号 L0C 出发，按 Fixpipe 的矩阵数、mSize/nSize、步幅和片段源偏移模拟目标地址，再与三种 dedw 的逻辑地址逐项比较。特别检查非整窗的第二块源偏移从新 L0C 起点计算、M 切换时状态重放、dk>1 且 ci 尾部不碰相邻任务。

当前自然 Host 计划发布 singleCoreM=baseM，跨多个 M 子块的父段复用只作为函数级扩展验证，不计入当前设备分支覆盖。

复用验证记录 Alloc/EnQue/DeQue/MTE1/FreeTensor 的槽号和父段身份；每个消费者读取的父段身份应与数学 K 一致。K 尾段、无效行带和无效深度均检查既不等待未生产数据，也不留下无人消费的预取张量。函数模型、Host 命中、SDK 编译和设备结果分别留证。

### 2.6 用例表格式

用例资产至少保存以下字段，实际 CSV/YAML 列名与调用工作流的解析器绑定：

| 类别 | 字段 |
|---|---|
| 身份 | case_id、seed、预期成功或错误类型 |
| 张量 | x/dedy/filter_size、输入 dtype、输入格式与输出格式 |
| 属性 | strides、六侧 pads、dilations、groups |
| 路径标签 | template、isSplitKernelHW、splitWo、baseM/N/K、stepKa/b、五路份数、usedCoreNum |
| 精度 | rtol、atol、ptol 及其依据 |
| 结果 | 编译/运行状态、实际 key/字段、精度、越界/同步状态 |

预期标签与运行后的实际标签分开；只设置 desired tag 而未读取真实字段不算命中。

### 2.6 判定依赖分级与增量接入

数学、shape、分组、布局、dtype 和输出完整性是硬依赖；目标 SDK 编译与实际设备执行是实现验收依赖。性能基线是另外一层证据。

增量接入时先验证能独立复算的小形状，再验证边界与状态覆盖。负向用例先检查 Host 是否明确拒绝非法规格，不把失败运行当作正确拒绝；合法零结果仍属于正向用例。

## 3 输入数据构造

### 3.1 构造规则

两路独立生成非平凡数值，固定 seed。随机基础集可用有界、正负混合分布；布局诊断使用唯一编号；同步/初始化用例在执行前将 dedw 预填非零值，并重复运行。

输入量化发生在保存或送入设备之前，Golden 读取同一批量化后的值。BF16 不能用 FP16 文件代替；FP32 Golden 不再次夹到 FP16 的输出范围。

### 3.2 构造实现模式

```python
import numpy as np
import torch

def make_input(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-0.5, 0.5, shape).astype(np.float32)
    target = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    packed = torch.from_numpy(a).to(target).contiguous()
    decoded = packed.float().numpy()  # Golden 消费量化后的数值
    return packed, decoded

def save_input(path, packed):
    # 两种输入都保存真实的 16 位编码，BF16 不经 np.float16 转换。
    packed.view(torch.uint16).numpy().tofile(path)

x_bits, x_values = make_input((1, 2, 3, 3, 3), "bf16", seed=17)
dy_bits, dy_values = make_input((1, 4, 2, 2, 2), "bf16", seed=29)
```

诊断数据中不同组使用不同幅值或编号，检查组隔离；不同 kd 切片使用不同编号，检查深度映射。全 1 只能作为计数辅助，不作为唯一正确性输入。

## 4 Golden 生成方法

### 4.1 基准实现选择

| 基准 | 用途 |
|---|---|
| CPU torch 的 aten.convolution_backward，output_mask 仅选择 grad_weight | 常规 Golden |
| 按 scene 的直接求和、FP64 累加 | 小形状独立交叉验证，检查布局、分组和非对称 pad |
| 显式 dedy_flat×x_win | 验证窗口化与 N/K 排布；不能独立证明自身重用的错误寻址 |

梯度输出形状来自 filter_size。用于调用框架反向接口的 weight 只提供形状，其数值不应参与 grad_weight。

### 4.2 计算精度：量化输入抬到 FP32

Golden 先读取真实 FP16/BF16 输入，再抬到 FP32 或更高精度计算；不能使用量化前的随机 FP32 值。当前输出固定 FP32，不执行 FP16 输出裁剪、饱和或转换。

长归约可用 FP64 直接求和交叉验证。其结果与设备 FP32 累加存在合理舍入差异，容差依据输入类型、归约长度与既定验收要求记录。

### 4.3 属性归一

strides/dilations 统一为空间三元组，pads 统一为 `(front,back,up,down,left,right)`。框架调用使用完整显式补零后的 x，卷积 padding 设为 0，避免非对称 pad 的方向歧义。

补零顺序为 `F.pad(x,(left,right,up,down,front,back))`。逐轴用原 x 和六侧 pad 验证 dedy 形状，再进入参考计算。

### 4.4 维度与格式归一

低维规格先按共享前置处理升维到 5D。x/dedy 的 NDHWC 转 NCDHW 使用 transpose/permute，不使用 reshape。filter_size 的轴序按输出格式解读，返回 grad_weight 的逻辑轴为 `(Cout,CinG,Dk,Hk,Wk)`。

```text
           +---------------------------------------+
           | actual packed FP16 / BF16 input bytes |
           +-------------------+-------------------+
                               |
                               | same rounded values as device
                               v
        +----------------------+----------------------+
        | decode to FP32 -> permute x / dedy to NCDHW |
        +----------------------+----------------------+
                               |
                               | forward sampling geometry
                               v
      +------------------------+------------------------+
      | explicit six-side x padding -> grad_weight FP32 |
      +------------------------+------------------------+
                               |
                               | preserve group-local ci
                               v
+------------------------------+------------------------------+
| permute to dedw NCDHW / NDHWC / DHWCN -> contiguous output  |
+-------------------------------------------------------------+
```

输出 NDHWC 使用 `(0,2,3,4,1)`，DHWCN 使用 `(2,3,4,1,0)`；分组时 CinG 是组内维度，Cout 保持总输出通道维。

### 4.5 输出 dtype 处理

grad_weight 最终转为连续 FP32 数组，按目标格式落盘。输出元素数恒为 `Cout*CinG*Dk*Hk*Wk`，文件大小为其乘 4。与对齐 mValue/nValue 形成的内部存储量分别核对。

### 4.6 Golden 代码骨架

以下函数消费已升维、已量化再解码为 FP32 的真实输入。返回值已按目标输出格式排列：

```python
import numpy as np
import torch
import torch.nn.functional as F

def feature_to_ncdhw(a, fmt):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 5:
        raise ValueError("normalize spatial rank before golden")
    if fmt == "NDHWC":
        a = a.transpose(0, 4, 1, 2, 3)
    elif fmt != "NCDHW":
        raise ValueError("unsupported feature format")
    return np.ascontiguousarray(a)

def filter_shape_ncdhw(shape, fmt):
    if len(shape) != 5 or any(int(v) <= 0 for v in shape):
        raise ValueError("invalid filter_size")
    if fmt == "NCDHW":
        return tuple(map(int, shape))
    if fmt == "NDHWC":
        co, kd, kh, kw, ci = map(int, shape)
    elif fmt == "DHWCN":
        kd, kh, kw, ci, co = map(int, shape)
    else:
        raise ValueError("unsupported filter format")
    return co, ci, kd, kh, kw

def golden_dw(x, dedy, filter_size, strides, pads, dilations, groups,
              data_format="NCDHW", filter_format="NCDHW"):
    x = feature_to_ncdhw(x, data_format)
    dy = feature_to_ncdhw(dedy, data_format)
    ws = filter_shape_ncdhw(filter_size, filter_format)
    if len(strides) != 3 or len(dilations) != 3 or len(pads) != 6:
        raise ValueError("expected normalized spatial attributes")
    if min(*strides, *dilations) <= 0 or min(pads) < 0 or groups <= 0:
        raise ValueError("invalid convolution attributes")
    if x.shape[0] != dy.shape[0] or x.shape[1] != ws[1]*groups:
        raise ValueError("input batch/group channels mismatch")
    if dy.shape[1] != ws[0] or ws[0] % groups:
        raise ValueError("output group channels mismatch")
    expected = tuple((x.shape[2+a]+pads[2*a]+pads[2*a+1]
                      -(ws[2+a]-1)*dilations[a]-1)//strides[a]+1
                     for a in range(3))
    if min(expected) <= 0 or expected != dy.shape[2:]:
        raise ValueError("forward geometry mismatch")
    pf, pb, pu, pd, pl, pr = pads
    xt = F.pad(torch.from_numpy(x), (pl,pr,pu,pd,pf,pb))
    _, dw, _ = torch.ops.aten.convolution_backward(
        torch.from_numpy(dy), xt, torch.zeros(ws, dtype=torch.float32), None,
        list(strides), [0,0,0], list(dilations), False, [0,0,0],
        groups, [False,True,False])
    a = dw.detach().numpy()
    if filter_format == "NDHWC":
        a = a.transpose(0,2,3,4,1)
    elif filter_format == "DHWCN":
        a = a.transpose(2,3,4,1,0)
    return np.ascontiguousarray(a, dtype=np.float32)
```

最小检查：`x=[1,2,3,4,5]`、dedy=[10,20]、filter_size=(1,1,1,1,3)、strideW=2、pad=0，Golden 为 `[70,100,130]`。三维 2×2×2 核的完整数值见 compute-flow 第 4.2 节。

### 4.7 高频陷阱

| 现象 | 核对 |
|---|---|
| BF16 量化误差整体偏大 | 是否用量化前 FP32 值求 Golden，或把 BF16 编码当作 FP16 |
| 非对称 pad 一侧整体错位 | F.pad 参数顺序与原正向形状公式 |
| 多组通道混合 | x/dedy 用全局通道、weight 梯度用组内 ci |
| 输出格式错 | grad_weight 返回轴与目标排列，未用 reshape 冒充转轴 |
| 大幅值 FP32 输出被截断 | 是否误用了输入梯度分支的 FP16 输出转换 |

## 5 精度比对与通过判定

### 5.1 逐元素比对公式

常规有限值输入下，每点比较 `abs(output-golden) <= atol+rtol*abs(golden)`。输出与 Golden 的 shape、元素数和目标布局首先一致；非预期 NaN/Inf 直接失败。

### 5.2 用例通过判定

记录逐点匹配率、最大绝对/相对误差与首个差异坐标。用例按事先声明的 ptol 判定允许超差比例；要求逐点全通过时 ptol=0。负向用例以规定阶段、规定原因拒绝为准，不进入数值精度统计。

### 5.3 容差取值

rtol/atol/ptol 从调用方规格、所选工作流精度约定或已登记标准取得。FP32 输出不代表计算等同 FP64，长归约与分段相加的次序会影响误差；这些影响需要记录，不自动放宽容差。

### 5.4 比对代码骨架

```python
def compare_dw(output, golden, *, rtol, atol, ptol=0.0):
    out = np.asarray(output, dtype=np.float32)
    ref = np.asarray(golden, dtype=np.float32)
    if out.shape != ref.shape or out.size == 0:
        return False, {"reason": "shape/size mismatch"}
    if not (0 <= ptol <= 1) or rtol < 0 or atol < 0:
        raise ValueError("invalid comparison policy")
    if not np.isfinite(out).all() or not np.isfinite(ref).all():
        return False, {"reason": "unexpected nonfinite"}
    diff = np.abs(out.astype(np.float64)-ref.astype(np.float64))
    ok = diff <= atol + rtol*np.abs(ref.astype(np.float64))
    ratio = float(ok.mean())
    first = None if ok.all() else tuple(np.argwhere(~ok)[0].tolist())
    return (1-ratio <= ptol), {
        "matched_ratio": ratio, "max_abs": float(diff.max()),
        "max_rel": float((diff/np.maximum(np.abs(ref),1e-30)).max()),
        "first_mismatch": first}
```

### 5.5 容差校准方法

先用编号小张量和严格容差排除寻址/漏算，再观察随机及长归约误差。结构性错误先定位；容差修改需要已有标准或数值证据并形成记录，不能以“原子累加”为由接受错布局或相邻核多写。

## 6 参考代码指引

本页第 3～5 节提供独立输入、Golden 和比对骨架；compute-flow 提供小张量排布与窗口结果；mn-streamk 提供字段与预算复算；适配层提供实际参数与 Kernel 阶段代码。

生成工程将这些内容落实为脚本和可重放用例，分别保存直调或注册入口的真实构建/执行记录。Host 字段对照、CPU 数学验证和 NPU 精度是不同证据层。

## 7 失败定位

### 7.1 信息收集

保存规格、seed、真实量化输入、Golden/output、实际 key、字段、编译/运行日志、首次差异坐标及归约段边界。将外部格式坐标转回 `(co,ci,kd,kh,kw)`，再恢复任务与 N 分形位置。

### 7.2 差异分布 → 根因映射

| 差异分布 | 优先检查 |
|---|---|
| 整体全零或量级错误 | 核类型/二进制实例、dedy/x 角色、初值、原子累加状态 |
| 只错某些 kd | x 深度采样、offsetC_ 的 kd 项、跳过段 |
| 沿 kh/kw 周期性错 | N 轴顺序、Load3D 起点、非整窗 head_/tail_ |
| 只错尾 co/ci | 实际通道与对齐跨度、Fixpipe.nSize、组起点 |
| 宽片边界错 | splitWo 尾片、原行间隔、alignedL1UseKa |
| 只有后归约段错 | isFirstMmad、父段身份、offsetA_/B_ 重算、AtomicNone 恢复 |
| 不同输出格式结果不同 | Fixpipe 分发、矩阵数和源目的步幅 |
| 结果多出旧值或重复执行翻倍 | 初始化缺失/范围错误、主计算与清零顺序 |
| 偶发、与核数相关 | 邻区多写、输入槽覆盖、CO1 归还和最终收尾 |

### 7.3 深度定位工具

按最小错误区域依次比较 GM→L1 后的父段、L0A/L0B 的编号窗口、首次及段末 L0C、Fixpipe 后输出。固定任务与段编号，先关闭可重叠预取验证生命周期，再恢复并检查事件；所有临时变更要记录并撤销。

## 8 范式回写

### 8.1 回写时机

API 单位、尾块、布局、分段、同步或初始化错误经人工修复后，将原因和规则补回对应 weight 文档。

### 8.2 回写内容

记录触发规格、真实字段、错误机制、正确公式/代码和最小复现。数学规则、SDK 事实和实测结论分别注明；单个工程通过不能直接扩大范式能力声明。

### 8.3 回写后的收尾义务

从新的生成目录独立重新生成、编译并复验，确认修复来自知识规则；原失败用例与同类邻界规格纳入回归。

## 9 回归、产物管理与验收记录

按用例清单保存输入、Golden、输出、比较策略、路径标签和日志。验收分别报告属性覆盖、实际 Tiling 状态覆盖、正确性、越界/同步与性能基线；失败和未执行项目明确标注。

## 10 检查点

- [ ] Golden 消费真实量化输入，输出 FP32 并按目标格式落盘。
- [ ] 对称/非对称 pad、groups、低维升维和三输出格式交叉验证。
- [ ] 编号矩阵、直接求和、框架权重梯度三者一致。
- [ ] L2 状态以真实字段命中，非整窗与普通/拆核写回均覆盖。
- [ ] 零贡献、非零预填输出、重复执行和长归约存在验证用例。
- [ ] shape/非有限值/数值容差分别判定，无隐式放宽。
- [ ] 人工修复已回写并通过独立重新生成复验。
