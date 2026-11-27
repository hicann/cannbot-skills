# 卷积反向算子验证方法

> **层级**：通用知识层（generic/input/），语言中立。
>
> **定位**：本文件回答"**卷积反向算子怎么验证**"——用例如何设计、输入如何构造、golden 如何生成、精度如何比对、通过判定如何计算、失败如何定位。全部方法论自含于本文件，**不依赖任何执行工具**；外部工程代码仅作为**实现参考**（「参考代码指引」列出位置与借鉴点）。
>
> 术语记号以 [../language.md](../language.md)（公共）与 [language.md](language.md)（本分支）为准；循环与 tiling 语义见 [compute-flow.md](compute-flow.md)、[tiling/tiling.md](tiling/tiling.md)。

---

## 1 验证闭环总览

```
        +-----------------+
        | ① 用例设计      |
        | （用例表+tag）  |
        +--------+--------+
                 |
                 v
         +-------+-------+
         | ② 输入生成    |
         | （随机+约束） |
         +-------+-------+
                 |
                 v
       +---------+---------+
       | ③ golden 生成     |
       | （CPU 参考实现）  |
       +---------+---------+
                 |
                 v
 +---------------+---------------+
 | ④ NPU 执行                    |
 | （kernel 直调或 aclnn 直调）  |
 +---------------+---------------+
                 |
                 v
         +-------+-------+
         | ⑤ 精度比对    |
         | （混合容差）  |
         +---------------+
                 |
                 +--> PASS
                 | FAIL
                 v
+----------------+----------------+
| ⑥ 失败定位                      |
| dump/日志/坐标反解 m=h_i*Wi+w_i |
+----------------+----------------+
                 | 根因在范式知识（非实现笔误）
                 v
⑦ 范式回写 -> 同批用例回归（第 8 节）
```

- ②③ 在 host 侧完成，④ 在 NPU 上执行，⑤ 自动比对；
- ④ 可通过 Kernel 直调（`<<<>>>`）或 ACLNN 注册两段式 API 执行，两种工程复用 [conv3ddx](../../adapters/ascendc/example/conv3ddx/README.md) 的计算实现；含 bias BT 通路与 output_padding 的转置卷积参考 [conv3dtranspose](../../adapters/ascendc/example/conv3dtranspose/README.md)，工程接入由对应工作流完成；
- DSL 的 ④ 使用所选编译实例的 `@host` 入口启动，见 [DSL 入口与验证](../../adapters/dsl/kernel-entrance.md)；Golden、用例和比对规则仍按本页执行。
- 失败用例经 ⑦ 定位后，根因若在范式知识，按「范式回写」关闭——用例是范式的试金石。

**最小验证工程的组成**（职责通用；下表路径为 Ascend C 工程约定，DSL 按调用工程组织）：

| 组件 | 职责 | 落地位置（见「参考代码指引」） |
|------|------|---------|
| 用例表 | 内嵌于数据生成脚本（按「用例设计三层次」设计，含 tiling 状态覆盖与 tag 标注） | 算子工程 tests/ut/op_kernel/*/gen_data.py |
| 数据生成脚本 | 生成 input bin + golden bin | `tests/assets/input.py` / `tests/ut/op_kernel/conv3d_backprop_input_v2_data/gen_data.py` |
| 执行程序 | 读 bin → 设备内存 → 拉起算子 → 结果落盘（含性能采集） | `examples/test_aclnn_conv3d_backprop_input_v2.cpp` |
| 校验脚本 | 混合容差比对，产出结果 | `tests/assets/golden.py` |
| 用例规格 | 逐用例的 shape/属性/精度配置 | `tests/st/arch35/ttk_kernel_*.csv` |

## 2 验证范围与用例设计

### 2.1 验证范围声明

| 维度 | 范围 |
|------|------|
| 场景 | 输入梯度（Conv3DBackpropInput）与转置卷积（Conv3DTranspose） |
| dtype | weight/dedy/dedx 三路一致：FP16、BF16 或 FP32（执行路径见 scene §3.4）；转置卷积 bias 为 FP32 |
| format | dedy/dedx：NCDHW 或 NDHWC（配对）；weight：NCDHW 或 NDHWC（独立） |
| 属性约束 | 正向 pad 为非负；output_padding 满足 `q<stride or q<dilation`；groups ≥ 1 且整除通道 |
| 空间维 | 1D / 2D / 3D（升维后统一为 3D 问题） |

### 2.2 用例设计三层次

用例设计不止覆盖"属性组合"，更要覆盖 **tiling 状态空间**——范式的每一条决策分支都应有用例压到：

| 层次 | 覆盖目标 | 设计方法 |
|------|---------|---------|
| **L1 属性组合** | 语义正确性（stride/pad/dilation/groups/bias/output_padding 交叉） | 属性组合矩阵 + 边界值 |
| **L2 tiling 状态** | 四模板、按 dtype 可达的 weight 变换和数据准备、容量回调、驻留命中/失效、尾组合并；kernel-split 的子核相位与输出重排；FP32 的 C0=8 尾块及模板选择 | 按规格反推 tiling 状态，构造命中各状态的 shape |
| **L3 规模边界** | 性能与资源极限（L1/L0 容量临界、指令限制临界、尾块） | 容量不等式取等边界反解 shape |

新增分支按下表验证。路径标签从真实 Host 结果读取，不能通过改 key 强制命中；CPU 等价验证不能代替设备搬运和同步验证。

| 分支 | 必须覆盖与核对 |
|---|---|
| FP32 | Cin/Cout 跨 8/16 边界；baseK 为 8 的倍数且非 16 的倍数；B2 固定分支 2、单位核直接取块；FP32 文件读写、NoQuant、hf32Flag=0；有/无 bias 及无贡献输出 |
| N/K 全载 | N 方向不切分、完整 K 驻留；核对 stepKb*baseK>=kValue、bl1Pbuffer=1、enableFullLoad 及权重身份；覆盖多个 M 任务、跨行和通道尾块、局部 A1 单/双份、K 多轮、容量回退、首次装载/复用/失效/最终归还；沿用普通 key 和初始化协议 |
| C04 | FP16/BF16；cout=1/2/3/4 与 cout=5 对照；hk*wk 对 4 的四种余数；两路 K 配对、实际尾 K 与 MMAD 对齐；单/双 AIV 协议及重复执行；UB、索引/掩码预留和 gather 索引范围 |
| group enlarge | enlarge=1/>1、G 可整除/不可整除 enlarge、CinG≠CoutG；独立原始组输入验证非对角块为零；发布 cinG/coutG 已放大且 oriGroup 保留原值 |
| VecTranspose | 三种 dtype、常规/子核各自准入；FP32 地址表和 repeat、固定 Cin0=16、cin 尾块及核空间 32B 补齐；多 AIV/无任务 AIV 的完成通知 |
| kernel-split | split-H/split-HW、H/HW 成本选择与回退、cout 全载；pad 相位、3×3 不等长子核、1×1 无贡献相位、奇数 hi/wi；UB 直达与 workspace 中转；dstBytes/accBytes 分别核算 |
| 组合 | kernel-split+VecTranspose；C04/group enlarge 与子核路径的自然互斥；bias 仅在输出归约完成后加一次 |

### 2.3 L1 属性组合矩阵（基础集）

| 属性 | 取值集 | 必测边界 |
|------|--------|---------|
| stride | 1；>1 对称；H/W 不对称 | stride=1 与 stride≥2 各至少一例 |
| pad | 0；对称；**不对称**；反向裁剪（负反向边界） | 不对称 pad 与负边界是反向补边的高危点 |
| dilation | 1；>1 | dilation>1 时有效核尺寸变大，行带随之变化 |
| kernel | 1×1×1；常规（3×3×3）；大核 | 单位核：普通 FP16/BF16 命中分支 1，FP32 命中分支 2；分别核对 B1 朝向与 B2 抽取 |
| groups | 1；>1（CinG 与 CoutG 不同） | 组内非整通道、组隔离 |
| output_padding | 三轴分别非零；同时非零；q=0；q 仅小于 dilation；非法值 | q 只扩大输出范围，不改变 K |
| bias（转置） | 缺省；全零；非零；长度/dtype 错误 | 无 bias 命中 L0C 零初始化 |
| N（batch）/ D | 1；>1 | batch>1 与 D 轴多深度命中外层循环与深度配对 |

### 2.4 L2 tiling 状态覆盖（范式特有，重要）

按 [tiling/tiling.md](tiling/tiling.md) 的决策分支反推规格，shape 量级按其容量公式核算：

| tiling 状态 | 命中方法 | 验证点 |
|------------|--------------------------|--------|
| small-shape | `expectedMaxCnt<thresholdCnt` 且不能被核数整除 | 形状面积上限、M/N 均衡、全载收益/容量回退及最终任务 M |
| l1-resident | `Dk=1、mValue>nValue、Wi<=512` 且容量初筛成立（图大、Co 小） | 完整 K 驻留、跨任务复用、身份失效 |
| standard | 其余常规规格 | 经验选块与任务调整、M 均衡条件 |
| 普通 weight 变换 | FP16/BF16 按核体积、格式和 baseN 选择；FP32 固定仅逆序，单位核直接取块 | L1B 父布局、K 子段、L0B 步幅 |
| 容量回调 | L1 预算超限的 shape（大 Co 或大核） | 减 K/减 M-N/调份数路径、模板不变 |
| 尾块体系 | baseM/baseN/baseK 不整除有效长度 | 有效/对齐双口径（baseUseM_/mCompute）、Fixpipe 尾参数 |
| bias 预算（转置） | baseN 与 singleCoreCin 不同；bias 尾部 32B/64B | isBiasFullLoad 判定、BT 容量临界 |

> 实践技巧：先在本机跑 Host 求解打印，确认构造的规格命中预期模板与 loadB2Condition，再纳入用例表。

### 2.5 用例表格式（CSV）

沿用算子工程的 ttk 用例表（`tests/st/arch35/ttk_kernel_*.csv`）：每行一个用例，核心字段为 shape/属性/dtype/format/精度配置（input_data_ranges、precision_tolerances）；tag 标注预期命中的 tiling 状态与 weight 变换，Host 执行后核对真实命中。命名建议 `<op>_<特征标签>_<序号>`（如 `asym_pad`、`q_only_lt_dil`、`l1_resident`），便于失败归因与回归选择。

### 2.6 判定依赖分级与增量接入

用例对实现的判定分三层，拦截对象与依赖各不相同；前层不拦截的，后层才会暴露：

| 判定层 | 拦截对象 | 依赖 |
|--------|---------|------|
| 精度比对（第 5 节） | 数值错误 | 以原始数学语义为基准；standard 为公共模板的后备路径，仍须通过资源与指令约束。普通三种 weight 变换的归约贡献等价；C04、group enlarge 与 kernel-split 虽改变内部排列或计算几何，也必须保持原始贡献集合和组隔离，不能要求内部 L0B 字节布局相同 |
| tag 核对（第 2.5 节） | 实现与已选策略不一致 | 固定策略版本、目标能力和收益模型参数，再复算模板选择与 loadB2Condition；优化策略变化时重核 tag，不以在线求解器的命中替代正确性判断 |
| 验收记录（第 9 节路径覆盖） | 实现完整性缺口 | 以 Host 最终选择证明四模板、普通三变换、数据准备、初始化及驻留的命中映射 |

由此得到的实现依赖分级：

| 级别 | 内容 | 缺失表现 |
|------|------|---------|
| 硬依赖 | 声明范围内的 key 实例及对应搬运链；容量回调；initOutputFlag 判定与清零执行；转置的 bias 全链；外部格式寻址；所选 C04/group enlarge/kernel-split 协议 | 对应用例可能无法执行或数值错误 |
| 优化依赖 | small-shape、l1-resident、ApplyBalancedMSplit，以及 H/HW 成本比较 | 可用其他已验证且合法的计划保持数值正确；优化收益及分支覆盖需单独核对 |
| 平台绑定 | tag 预期随核数与 L1/L0 容量变化 | 换平台须重核 tag（第 9 节 platformInfo 记录） |

**增量接入顺序**：先验证 standard、普通权重变换、容量回调和初始化；再接入所声明的数据准备与子核分支，最后完善均衡和驻留优化。每步只验收已实现范围，不预设 standard 能覆盖全部扩展场景。复用既有用例表时，先按第 2.1 节过滤范围，再换绑算子名与 golden 插件（第 6 节）。

## 3 输入数据构造

### 3.1 构造规则

| 项 | 约定 | 依据 |
|----|------|------|
| 分布 | 均匀分布 **U(-1, 1)**（用例表 input_data_ranges 默认）；手算复核用小整数/必现值 | 幅值有界，避免 fp16 上溢与累加精度劣化 |
| 种子 | 固定种子（同一种子下生成序列确定） | 失败用例必须可复现 |
| 量化时机 | 生成后立即按输入 dtype（FP16/BF16/FP32）落盘；bias 保持 FP32 | golden 从量化后的输入计算 |
| 格式 | 在逻辑 NCDHW 轴序生成，存储格式按坐标 permutation 落盘，**禁止 reshape** | 同一逻辑数值在两种格式下可直接互验 |
| 升维 | 1D/2D 用例先按 [../preprocessing.md](../preprocessing.md) 升维（插入单维，元素顺序不变） | 与算子侧归一一致 |
| 预填输出 | 测试前用非零值填满 dedx，检查无贡献位置严格为零（或 bias 值）、旧结果消失 | 验证初始化协议与写回覆盖 |

### 3.2 构造实现模式（numpy）

```python
import numpy as np
dtype = storage_dtype(input_dtype)                    # 复用第 4.6 节；输入类型由规格给定
rng = np.random.default_rng(seed)                      # 每用例独立且确定
w = rng.uniform(-1, 1, wShape).astype(dtype)           # weight (Cout,CinG,Dk,Hk,Wk)
dy = rng.uniform(-1, 1, dyShape).astype(dtype)         # dedy (N,Cout,Dout,Ho,Wo)
b = rng.uniform(-1, 1, (Cin,)).astype(np.float32) if has_bias else None
w.tofile(f"{case}_w.bin")
dy.tofile(f"{case}_dy.bin")
if b is not None:
    b.tofile(f"{case}_b.bin")
```

> 参考：opstest 用例表 input_data_ranges 字段语义、算子工程 assets 的 input 脚本（「参考代码指引」）。

## 4 golden 生成方法

### 4.1 基准实现选择

| 基准 | 用途 |
|---|---|
| CPU Torch `conv_transpose3d`，FP64 计算 | 从已量化输入计算高精度参考；weight 的 `(Cout,CinG,Dk,Hk,Wk)` 直接匹配该接口 |
| 独立坐标散射 | 按 `i=o*stride-padBegin+k*dilation` 累加到目标 dedx，复核分组、非对称边界与目标尺寸 |

两个基准都消费逻辑张量及原始卷积属性，不读取被测 Kernel 的 tiling、分形排布或中间结果。小尺寸用例逐元素对拍后，固定一个高精度基准执行验收。

### 4.2 计算精度

输入先按实际 FP16/BF16/FP32 类型落盘，再提升到 FP64 计算，bias 从其 FP32 值提升。高精度 golden 保留 FP64，输出类型转换单独验证。FP32 用例绑定 `enable_hf32=false`。

输出转换单独用于舍入、溢出与特殊值检查；它与高精度数值基准分开，见第 4.5 节。

### 4.3 属性与目标范围

1. strides/dilations 按 format 提取 D/H/W，pads 归一为每轴的 `(begin,end)`。
2. 先以 `padding=0、output_padding=0、bias=None` 计算完整乘积支持区。该区包含所有输入与核采样的贡献。
3. 对每轴从 `padBegin` 取 `targetLength` 个位置；支持区不足时先在末尾补零乘积，再截取。不要先截目标长度再裁非对称 pad。
4. 最后对全部目标位置加一次 bias，包含没有乘积贡献的位置。

target_shape 由前置处理校验后提供：DX 使用 input_size；Transpose 使用属性推导或显式目标形状，已包含 q/rho。它们只改变目标范围，不改变核长。此参考不再把 q/rho 相加后强行传入框架的 output_padding；前后 pad 通过坐标起点和目标长度共同体现。

### 4.4 维度与格式归一

- 1D/2D 按 [前置处理](../preprocessing.md) 升维到 3D。
- dedy/dedx、weight 分别按自己的 format 做轴置换，禁止用 reshape 替代转置。参考计算统一使用逻辑 NCDHW 轴序。
- groups 原样传入，weight 第二轴保持原始 `CinG`；不把 group enlarge 的内部布局送入 golden。
- 其他片上格式仅在对应生成接口声明支持时解包；内部补齐通道不属于外部数值张量。

### 4.5 输出 dtype 与转换检查

FP16、BF16、FP32 分别使用 `np.float16`、`ml_dtypes.bfloat16`、`np.float32` 存储。

`fp16_saturate` 由原用例和目标转换契约指定：true 为 FP16 有限值饱和，false 为常规转换。BF16 直接按其类型舍入，FP32 直接转为 np.float32。

高精度 golden 保存为 FP64；`convert_output_dtype` 另生成输出转换参考。设备输出按声明的 FP16/BF16/FP32 类型读取，再提升后比较。

### 4.6 golden 代码骨架

输入已量化且已归一为逻辑五维；target_shape 为同一轴序。CPU 参考依赖 NumPy、Torch，BF16 文件读写另需 ml_dtypes。

```python
def storage_dtype(name):
    import numpy as np
    if name == "float16":
        return np.dtype(np.float16)
    if name == "float32":
        return np.dtype(np.float32)
    if name == "bfloat16":
        import ml_dtypes
        return np.dtype(ml_dtypes.bfloat16)
    raise ValueError(f"unsupported storage dtype: {name}")


def convert_output_dtype(value, output_dtype, *, fp16_saturate):
    import numpy as np
    value = np.asarray(value, dtype=np.float64)
    if output_dtype == "float16" and fp16_saturate:
        limit = np.finfo(np.float16).max
        value = np.clip(value, -limit, limit)
    with np.errstate(over="ignore", invalid="ignore"):
        return value.astype(storage_dtype(output_dtype)).astype(np.float64)


def golden_dx_or_transpose(dedy, weight, bias_fp32, stride, pad, dil,
                           groups, target_shape):
    import numpy as np
    import torch
    import torch.nn.functional as F

    dy = torch.from_numpy(np.asarray(dedy, dtype=np.float64))
    w = torch.from_numpy(np.asarray(weight, dtype=np.float64))
    target_shape = tuple(int(v) for v in target_shape)
    if len(target_shape) != 5 or any(v <= 0 for v in target_shape):
        raise ValueError("target_shape must be positive logical NCDHW")
    if groups <= 0 or dy.shape[1] % groups or w.shape[0] != dy.shape[1]:
        raise ValueError("invalid groups or weight/dedy channels")
    if target_shape[:2] != (dy.shape[0], w.shape[1]*groups):
        raise ValueError("target batch/channels mismatch")
    if len(pad) != 3 or any(len(p) != 2 or min(p) < 0 for p in pad):
        raise ValueError("pad must contain three nonnegative pairs")

    out = F.conv_transpose3d(dy, w, bias=None, stride=tuple(stride),
                             padding=0, output_padding=0,
                             dilation=tuple(dil), groups=groups).numpy()
    need = [int(p[0])+n for p, n in zip(pad, target_shape[2:])]
    extra = [(0, max(0, n-have)) for n, have in zip(need, out.shape[2:])]
    out = np.pad(out, [(0, 0), (0, 0)]+extra)
    slices = [slice(None), slice(None)]
    slices += [slice(int(p[0]), int(p[0])+n)
               for p, n in zip(pad, target_shape[2:])]
    out = out[tuple(slices)].copy()
    if bias_fp32 is not None:
        bias = np.asarray(bias_fp32, dtype=np.float32)
        if bias.shape != (target_shape[1],):
            raise ValueError("bias length mismatch")
        out += bias.astype(np.float64).reshape(1, -1, 1, 1, 1)
    assert out.shape == target_shape
    return out
```

golden 的计算结果是逻辑 NCDHW。落盘前按 dedx 的外部格式排列；NDHWC 使用 `golden.transpose(0,2,3,4,1)`，NCDHW 保持原轴序，再以 FP64 保存。1D/2D 输出按前置处理的逆映射恢复声明轴序。不能将 NDHWC 设备输出与 NCDHW 的线性 golden 直接比较。

独立散射从真实 dedy、原 weight 核坐标和组内 ci 枚举，向目标范围累加。覆盖非对称 pad、stride 尺寸余量、groups>1、dilation、有/无 bias，并分别检查三种输出类型。

### 4.7 高频陷阱

| 陷阱 | 后果 | 检查 |
|---|---|---|
| 先截目标，再裁 pad | 输出尺寸过短或坐标偏移 | 从完整支持区按 padBegin 和目标长度截取 |
| 用零填充替代所有 q 新位置 | 漏掉仍有乘积贡献的位置 | 先计算完整支持区，再判断范围外的零乘积 |
| bias 先加到支持区、再补零 | 新增无贡献位置缺 bias | 目标范围形成后，对每个输出位置加一次 |
| BF16 经 FP16 cast/clip | 舍入与数值范围错误 | 按实际输出 dtype 转换和读文件 |
| FP32 按 2B 类型读写 | 元素数和数值解释错误 | 输入/输出使用 float32，高精度 golden 使用 float64 |
| weight 再交换通道轴 | 数值或 groups 关系错误 | 保留 `(Cout,CinG,Dk,Hk,Wk)` |

## 5 精度比对与通过判定

### 5.1 逐元素比对

设备输出按实际 dtype 读出后提升到 FP64，与高精度 golden 比较：

`abs(output-golden) <= atol + rtol*abs(golden)`。

同号 Inf 作为相同特殊值；单侧 Inf/NaN 失败。是否允许双 NaN 相等由用例契约指定。形状和元素数先检查，不能把尺寸错误当作精度误差。

### 5.2 用例通过条件

`matched_ratio = 匹配元素数 / 总元素数`。按已选精度契约同时检查匹配率和最大绝对误差；使用 ptol 的调用方令 `required_matched_ratio=1-ptol`。最大误差上限独立传入，不用匹配率掩盖单点大误差。

### 5.3 容差来源

| 项 | 来源与约定 |
|---|---|
| atol / rtol | 优先读取原用例和工作流；缺省按 ops-precision-standard 的实际输出 dtype 选择 |
| ptol / required_matched_ratio | 明确二者的换算，不与 atol/rtol 混用 |
| max_abs_error_limit | 按同一验收契约传入；原协议未设置时传 None，不隐含增加或删除 |
| 特殊值与输出转换 | 声明 NaN/Inf、舍入及 FP16 溢出约定；FP32 按普通精度模式验收 |

生成过程不自动放宽已有阈值。严容差可用于定位；验收仍按原契约执行。BF16 不沿用 FP16 的默认阈值。

### 5.4 比对代码骨架

复用第 4.6 节的 storage_dtype；golden_bin 保存 FP64，线性轴序与设备输出一致。参数均由调用方选定的精度契约传入。

```python
def verify(output_bin, golden_bin, output_dtype, *, rtol, atol, ptol,
           max_abs_error_limit, equal_nan):
    import numpy as np
    out = np.fromfile(output_bin, dtype=storage_dtype(output_dtype)).astype(np.float64)
    gold = np.fromfile(golden_bin, dtype=np.float64)
    if out.size != gold.size or out.size == 0:
        raise ValueError(f"invalid element counts: {out.size}, {gold.size}")
    finite = np.isfinite(out) & np.isfinite(gold)
    same_inf = np.isinf(out) & np.isinf(gold) & (out == gold)
    same_nan = np.isnan(out) & np.isnan(gold) & equal_nan
    error = np.full(out.shape, np.inf)
    error[finite] = np.abs(out[finite]-gold[finite])
    error[same_inf | same_nan] = 0
    ok = same_inf | same_nan
    ok[finite] = error[finite] <= atol+rtol*np.abs(gold[finite])
    matched_ratio = float(np.mean(ok))
    max_abs_error = float(np.max(error))
    max_error_ok = max_abs_error_limit is None or max_abs_error <= max_abs_error_limit
    passed = 1-matched_ratio <= ptol and max_error_ok
    return passed, {"matched_ratio": matched_ratio, "max_abs_error": max_abs_error}
```

### 5.5 差异定位

先检查形状、格式、输入量化与特殊值；再区分舍入/归约顺序差异和结构性错误。仅边界、特定通道或尾块出现错误时，优先核对坐标、组隔离、父段跨度与写回范围。长归约的误差须有分布证据，不能直接按 K 放宽容差。

## 6 参考代码指引

本范式的验证方法论不依赖任何工具执行；以下代码作为**实现参考**（input 构造、golden 生成、精度比对的具体工业实现）：

| 参考代码 | 位置 | 借鉴点 |
|---------|------|--------|
| ttk 精度比对实现 | `gitcode.com/cann/ops-test-kit` → `ttk/core_modules/comparison/is_close.py` | 核对所用版本的 isclose、匹配率及阈值字段；按当前用例契约接入 |
| Conv3DBackpropInputV2 golden 插件 | `ops-nn/conv/conv3d_backprop_input_v2/tests/assets/golden.py` | 接口、dtype/format 解析与输入绑定；裁剪和类型转换按第 4 节独立复核 |
| Conv3DTransposeV2 golden 插件 | `ops-nn/conv/conv3d_transpose_v2/tests/assets/golden.py` | 核对 x/filter/bias 与目标形状的绑定；参考计算按第 4 节处理 |
| opstest 用例表 | `gitcode.com/cann/opstest`（分支 ops-nn）→ `conv/<op>/arch35/*.csv` | 第 2 节的取值来源：既有卷积用例的属性组合与容差登记（rtol/atol/ptol 携带值） |
| 模板声明与 key | Host/Device 共用的模板参数声明 | 当前三实参、合法组合和实际运行字段；示例私有 ABI 按其 README 核对 |
| 范式自带参考实现 | [conv3ddx](../../adapters/ascendc/example/conv3ddx/README.md) | Tiling、Kernel、TilingData 与范式章节的对应关系；直调与 ACLNN 工程共用 |

golden 插件以 `__golden__` 字典注册到 ttk（kernel 模式按算子名路由）；具体算法与参数须按当前算子规格重新核定。

## 7 失败定位

### 7.1 信息收集

- 失败用例的 dump 产物（input/golden/output 三方 bin）、执行日志（tiling key、blockDim、tiling 字段、首个差异点坐标与期望/实际值）、复现命令固定 seed；
- 差异的空间分布（哪些 batch/通道/深度/行列）、差异量级（ULP 级 vs 错数据级 vs 全零）。

### 7.2 差异分布 → 根因映射（范式经验表）

| 差异分布 | 高概率根因 | 核对点 |
|---|---|---|
| 全图大幅错误 / 全 0 | dtype 错配、核类型未声明（AIC 守卫被剔除）、L0C 初始化 | 检查三实参/key、核类型声明、cmatrixInitVal；binary 错配（kernel 名与输入 dtype 核对） |
| 仅输出边缘行/列错 | 反向边界/裁剪错误（backpropPad* 口径、尾部余数 ρ 未按目标形状计算）、golden 的非对称 pad 切片错 | 核对 golden 与 kernel 的 pad 口径、切片两侧各切各的、有符号边界复算 |
| 仅部分通道错（尾部通道） | 通道尾块补零、组内非整通道、L0B 尾部初始化 | 检查 Ci/Co 尾块、三种 weight 变换的尾部补零 |
| 规律性块状错（某 M/N/K 块整体错） | L1 父段切换、K 子块定位（kLocalA/kLocalB）、任务解码 | 按 `m=h_i*Wi+w_i` 反解块坐标；检查 GM 偏移与父段公式 |
| 仅尾块错（任务末 M 块） | 有效/对齐双口径错配（baseUseM_ vs mCompute）、Fixpipe 尾参数 | 检查 MmadParams.m 与 Fixpipe.mSize/srcStride 的取值口径 |
| 行带边界整体错 | dedy_expand 行带交集、负反向边界先裁剪、stride 相位 | 检查驻留行区间与 hSkip 相位、FMatrix 补边 |
| 仅 3D 的 D 边界错 | 深度配对（dTmp 整除/界内检查）或 dilationD 初始化条件 | 核对 kd 采样与 initOutputFlag 条件 |
| 无贡献位置非零（或缺 bias） | 输出预清零缺失、ComputeForBias 未覆盖 | 预填非零 dedx 复跑；核对 initOutputFlag 与无贡献写回 |
| 转 3D 全错但 2D 对、量级正常 | weight GM 寻址未按 kd 换算：`[Cout][Cin][Kd][Kh][Kw]` 中 kd 片跨通道步进是 `Kd*Kh*Kw` | 核对 DN2NZ/ND2NZ 的通道步进与切片基址 |
| 偶发/与核数相关 | 任务解码或多核边界 | 固定 blockDim 复跑；检查任务计数与轴序 |
| 卡住或重复执行失败 | 事件初态/末次归还、held L1B、L0C 交接 | 检查首末轮事件、槽回绕、FreeB1Tensor/End |

> 错值是"有效量级的错误数据"而非 NaN/全 0——说明通路和指令都对，是**寻址/预算口径**错；配合"哪一维、哪些块错"的空间分布即可按表定位。

### 7.3 深度定位工具

- **dump 对拍**：dump 输出/输入/golden 三方 bin → 按 shape 反解差异元素坐标 → 映射回 (batch, ci, di, h_i, w_i) → `m=h_i*Wi+w_i` → 所属核（taskId）、空间段与计算块（衔接 [compute-flow.md](compute-flow.md) 的观测视图）；
- **Tiling/地址与 DAG 核对**：相同规格和资源下，对照求解步骤及最终字段，解释差异；枚举 taskId 恢复有效 dedx 坐标，计数恰为 1；比较分块前后有效 `(co,kd,kh,kw)` 集合，补齐项单独为零；用唯一编号小张量逐元素追踪 GM/L1A/L1B/L0A/L0B/L0C/dedx，检查实际父步幅和槽基址；同时检查 Host 预算、物理最大末地址、指令实际访问（三者不互相替代）；
- **单位核路径**：FP16/BF16 的普通实例使用 `(1,0,0)`，B1 为 `(Cin1,Cout1,Cout0,Cin0)`、B2 转置；FP32 使用 `(2,0,0)`，B1 为 `(Cout1,Cin1,16,8)`、B2 直接取块。两者均核对 stride/pad 采样及实际驻留；
- **交叉验证**：同一用例 kernel 直调 vs aclnn 直调对比，区分"kernel 实现 bug"与"调度/二进制错配"。

运行前用非零值填满 dedx；检查无贡献位置严格为零（或 bias 值）、有贡献位置符合 Golden。改变 weight/dedy 后再次运行，检查旧结果消失。初始化和 L0C 首次乘加分别检查，输入/输出保护区辅助发现越界。有限输入的意外 NaN/Inf 判失败，特殊值测试单列规则。

## 8 范式回写

> 用例是范式的试金石：失败根因落在范式知识（而非实现笔误）时，**必须回写范式文档后才能关闭问题**。

### 8.1 什么时候必须回写

| 根因类别 | 是否回写 | 回写目标 |
|---------|---------|---------|
| 范式知识错误（公式/口径/约束写错） | 必须 | 对应 generic/适配层文档修正 |
| 范式知识缺失（设计未覆盖的分支） | 必须 | 补充对应章节 + 触发用例入回归集 |
| 实现笔误（范式正确、生成代码写错） | 不回写 | 用例入回归集；如属生成流程易错点，可在范式"检查点"处加一句泛化描述 |

### 8.2 回写内容的四条硬性要求

**① 通用性——写"一类场景"，不写"一个规格"**：范式内容必须是通用知识（原理、口径、约束、决策规则），不允许出现针对特定规格的补丁式描述；规格数值仅可作为**示例**出现，且必须标注"示例"，结论本身不得依赖该数值。

**② 通俗性——表格化、逐项列举、图示优先**：能用表格的不用段落；能逐项列举的不用长句；数据流/流程/时序用 ASCII 图；一段只讲一个要点；术语首次出现必须可追溯到 [../language.md](../language.md) 或 [language.md](language.md) 的记号定义。

**③ 完整性自查——补完必须全量重读**：回写完成后全量重读一遍受影响文档；新旧内容矛盾、交叉引用失效、章节编号变化一并处理；不确定、不清晰的描述宁可不写。

**④ 隔离性——用例/验证字眼只允许出现在本文档**：除本文件外，范式内其他文档不允许出现"用例背书"、"已验证"、"上板实证"等表述；知识与证据分离。

### 8.3 回写后的收尾义务

1. 触发回写的用例必须入回归集；
2. 若回写修正了约束/公式，检查是否需要补 L2 用例行（把新约束的触发面压到）；
3. 全量回归通过后，回写条目记入 knowledge-gap-report（缺陷描述 → 修复落点 → 回归证据）。

## 9 回归、产物管理与验收记录

**产物管理**（每次执行验证后保留，复跑与定位的入口）：

```text
run_<timestamp>/
|-- log/          # per-case execution logs
|-- data/         # input / golden / output bins
|-- result/       # validation JSON / CSV
|-- error_cases/  # failed-case subset for regression
+-- report.md     # pass rate / reproduction / timing
```

- 回归集 = L1 基础集 + 历史 error_cases；范式文档每次修订后全量回归；
- 报告须包含：总用例数/PASS 率、失败用例（名称/规格/复现命令）、NPU 耗时排序（性能回归观测）；性能数据注明通路与口径。

生成工程保留以下验收记录，状态使用通过、失败或待验证；第 2～7 节的地址、资源、事件及真实性能检查仍由生成工程完成：

| 记录 | 必须包含 |
|---|---|
| 输入与能力 | 原始规格/用例、dtype/format、SDK 与编译目标、platformInfo 资源及 API/事件适配结论 |
| Host 计划 | 实际 templateId、三实参/key、base*、singleCore*、step*、五路份数、初始化标志和容量检查；转置卷积的 q/rho/形状模式/isBiasFullLoad |
| 路径覆盖 | 用例到四个模板、普通三个 weight 变换、数据准备分支、初始化及驻留命中/失效的映射；以 Host 最终选择证明命中 |
| 设备与调用 | 编译命令、真实设备实例/加载来源、直调或注册入口、初始化与主计算的执行顺序 |
| 数值与再生成 | 独立 Golden、原精度配置、重复执行结果、重新生成目录及完整验收结果 |

按自然可达路径建立覆盖，不要求不可达组合；未覆盖项记录其准入条件与后续用例。正确性通过后，分别记录初始化、主计算和端到端耗时，固定输入、平台、预热与统计方法；l1-resident 另统计首次 L1B 装载和后续复用，数值通过不能证明复用次数或性能收益。

## 10 检查点

- [ ] 用例表同时覆盖 L1 属性组合与 L2 tiling 状态（四模板、普通三 weight 变换、数据准备、尾块、驻留；kernel-split 覆盖两子模式与 cout 全载变体）
- [ ] 验证范围（FP16/BF16/FP32、双格式、groups、output_padding 约束）与当前范式能力声明一致
- [ ] 输入按构造规则生成（固定种子、立即量化落盘、格式按坐标转换）
- [ ] golden 用 torch conv_transpose3d 基准（CANN filter 直接作 weight），属性归一与 kernel 侧口径逐项核对（pad 侧别、非对称切片、q/rho）
- [ ] numpy 手写 golden 已与 torch 基准对拍（对拍用未量化 fp64）
- [ ] 通过判定按"逐元素公式 + 匹配率"两步计算，容差取值有依据（校准记录在案）
- [ ] 无贡献位置预期明确（DX 为零、Transpose 为 bias 值），预填非零输出检查
- [ ] 产物目录完整保留（log/data/result/error_cases/report），失败用例入回归集
- [ ] 性能数据注明通路与口径
- [ ] 失败用例完成根因分类并履行回写义务（错误/缺失必回写，笔误入回归集）
- [ ] 回写内容符合四条要求（通用性/通俗性/完整性/隔离性），非本文档无用例字眼
