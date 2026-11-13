# 卷积反向算子验证方法

> **层级**：通用知识层（generic/input/），语言中立。
>
> **定位**：本文件回答"**卷积反向算子怎么验证**"——用例如何设计、输入如何构造、golden 如何生成、精度如何比对、通过判定如何计算、失败如何定位。全部方法论自含于本文件，**不依赖任何执行工具**；外部工程代码仅作为**实现参考**（「参考代码指引」列出位置与借鉴点）。
>
> 术语记号以 [../language.md](../language.md) 为准；循环与 tiling 语义见 [compute-flow.md](compute-flow.md)、[tiling/tiling.md](tiling/tiling.md)。

---

## 1 验证闭环总览

```text
+---------------------------------------+
| (1) Case design                       |
+-------------------+-------------------+
                    |
                    |
                    v
+-------------------+-------------------+
| (2) Input generation: host            |
+-------------------+-------------------+
                    |
                    |
                    v
+-------------------+-------------------+
| (3) Golden generation: CPU            |
+-------------------+-------------------+
                    |
                    |
                    v
+-------------------+-------------------+
| (4) NPU: kernel / ACLNN               |
+-------------------+-------------------+
                    |
                    |
                    v
+-------------------+-------------------+
| (5) Precision comparison              +---------------pass -> PASS
+-------------------+-------------------+
                    |
                    | fail
                    v
+-------------------+-------------------+
| (7) Failure localization              |
| dump / logs / coordinate mapping      |
+-------------------+-------------------+
                    |
                    |
                    v
+-------------------+-------------------+
| (8) Update paradigm rules             |
| Regenerate and rerun the same cases   |
+---------------------------------------+
```

- ②③ 在 host 侧完成，④ 在 NPU 上执行，⑤ 自动比对；
- ④ 的执行形态两种：kernel 直调（`<<<>>>`）或 aclnn 直调（注册算子+两段式 API）——工程接入参考 [adapters/ascendc/example/conv3ddx_direct/](../../adapters/ascendc/example/conv3ddx_direct/README.md) 与 [conv3ddx_aclnn/](../../adapters/ascendc/example/conv3ddx_aclnn/README.md)；转置卷积变体（含 bias BT 通路与 output_padding 语义的 golden/判定口径）另见 [conv3dtranspose_direct/](../../adapters/ascendc/example/conv3dtranspose_direct/README.md) 与 aclnn 注册形态 [conv3dtranspose_aclnn/](../../adapters/ascendc/example/conv3dtranspose_aclnn/README.md)；
- 失败用例经 ⑦ 定位后，根因若在范式知识，按「范式回写」关闭——用例是范式的试金石。

**最小验证工程的组成**（两种执行形态通用）：

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
| dtype | weight/dedy/dedx 均为 FP16 或 BF16（三路一致）；转置卷积 bias 为 FP32 |
| format | dedy/dedx：NCDHW 或 NDHWC（配对）；weight：NCDHW 或 NDHWC（独立） |
| 属性约束 | 正向 pad 为非负；output_padding 满足 `q<stride or q<dilation`；groups ≥ 1 且整除通道 |
| 空间维 | 1D / 2D / 3D（升维后统一为 3D 问题） |

### 2.2 用例设计三层次

用例设计不止覆盖"属性组合"，更要覆盖 **tiling 状态空间**——范式的每一条决策分支都应有用例压到：

| 层次 | 覆盖目标 | 设计方法 |
|------|---------|---------|
| **L1 属性组合** | 语义正确性（stride/pad/dilation/groups/bias/output_padding 交叉） | 属性组合矩阵 + 边界值 |
| **L2 tiling 状态** | 切分算法每条分支（三模板、三种 weight 变换、容量回调、驻留命中/失效） | 按规格反推 tiling 状态，构造命中各状态的 shape |
| **L3 规模边界** | 性能与资源极限（L1/L0 容量临界、指令限制临界、尾块） | 容量不等式取等边界反解 shape |

### 2.3 L1 属性组合矩阵（基础集）

| 属性 | 取值集 | 必测边界 |
|------|--------|---------|
| stride | 1；>1 对称；H/W 不对称 | stride=1 与 stride≥2 各至少一例 |
| pad | 0；对称；**不对称**；反向裁剪（负反向边界） | 不对称 pad 与负边界是反向补边的高危点 |
| dilation | 1；>1 | dilation>1 时有效核尺寸变大，行带随之变化 |
| kernel | 1×1×1（仅转置退化快路径）；常规（3×3×3）；大核 | 核体积 1 命中 loadB2Condition=3 |
| groups | 1；>1（CinG 与 CoutG 不同） | 组内非整通道、组隔离 |
| output_padding | 三轴分别非零；同时非零；q=0；q 仅小于 dilation；非法值 | q 只扩大输出范围，不改变 K |
| bias（转置） | 缺省；全零；非零；长度/dtype 错误 | 无 bias 命中 L0C 零初始化 |
| N（batch）/ D | 1；>1 | batch>1 与 D 轴多深度命中外层循环与深度配对 |

### 2.4 L2 tiling 状态覆盖（范式特有，重要）

按 [tiling/tiling.md](tiling/tiling.md) 的决策分支反推规格，shape 量级按其容量公式核算：

| tiling 状态 | 命中方法 | 验证点 |
|------------|--------------------------|--------|
| small-shape | `estimatedTasks < maxCores*5//2` 且不能被核数整除（小 shape） | M/N 均衡搜索、候选同代价裁决 |
| l1-resident | `Dk=1、mValue>nValue、Wi<=512` 且容量初筛成立（图大、Co 小） | 完整 K 驻留、跨任务复用、身份失效 |
| standard | 其余常规规格 | 经验选块与任务调整、M 均衡条件 |
| 三种 weight 变换 | 核体积=1（仅转置）；NDHWC 且 `baseN>=hk*wk`（仅逆序）；NCDHW 且 `dk*hk*wk*cin*weightBytes<=64`（仅逆序）；其余（转置加逆序） | L1B 父布局、K 子段、L0B 步幅 |
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
| 精度比对（第 5 节） | 数值错误 | 只依赖数学语义：三个 Tiling 模板是性能与任务划分决策而非正确性决策（standard 为兜底路径，任意合法规格均可求解），三种 weight 变换产生数学等价的 L0B（核体积 1 时"转置加逆序"的逆序为恒等，与"仅转置"一致），差异仅在编译实例与性能 |
| tag 核对（第 2.5 节） | 偏离范式判定链的实现 | 判定链复现：模板选择（[tiling/tiling.md](tiling/tiling.md) 第 2 节）与 loadB2Condition（[tilingkey-templates](../../adapters/ascendc/input/tilingkey-templates.md) 第 2 节）均为确定性规则，按范式实现则同规格必然同命中 |
| 验收记录（第 9 节路径覆盖） | 实现完整性缺口 | 以 Host 最终选择证明三模板、三变换、初始化及驻留的命中映射 |

由此得到的实现依赖分级：

| 级别 | 内容 | 缺失表现 |
|------|------|---------|
| 硬依赖 | 三个 loadB2Condition 编译实例及其搬运链（判定为确定性路由，不可降级到其余变换）；LegalProtection 容量回调；initOutputFlag 判定与清零执行；转置的 bias 全链；NCDHW/NDHWC 双格式寻址 | 对应用例直接失败 |
| 软依赖 | small-shape、l1-resident 求解器；ApplyBalancedMSplit 任务划分优化 | 数值仍绿；tag 核对与第 9 节路径覆盖出缺口、性能退化 |
| 平台绑定 | tag 预期随核数与 L1/L0 容量变化 | 换平台须重核 tag（第 9 节 platformInfo 记录） |

**增量接入顺序**：先实现 standard + 三变换 + 容量回调 + 初始化（全量用例数值即绿，可先行交付正确性）；按 tag 过滤 small-shape / l1-resident 子集补齐对应求解器；最后全量回归并补齐第 9 节路径覆盖证明。**复用既有用例表的两个动作**：按第 2.1 节范围过滤超范围组合（如 FLOAT32 输入）；用例与 golden 插件按新算子名换绑（第 6 节）。

## 3 输入数据构造

### 3.1 构造规则

| 项 | 约定 | 依据 |
|----|------|------|
| 分布 | 均匀分布 **U(-1, 1)**（用例表 input_data_ranges 默认）；手算复核用小整数/必现值 | 幅值有界，避免 fp16 上溢与累加精度劣化 |
| 种子 | 固定种子（同一种子下生成序列确定） | 失败用例必须可复现 |
| 量化时机 | 生成后立即按输入 dtype（FP16/BF16）落盘；bias 保持 FP32 | golden 从量化后的输入计算 |
| 格式 | 在逻辑 NCDHW 轴序生成，存储格式按坐标 permutation 落盘，**禁止 reshape** | 同一逻辑数值在两种格式下可直接互验 |
| 升维 | 1D/2D 用例先按 [../preprocessing.md](../preprocessing.md) 升维（插入单维，元素顺序不变） | 与算子侧归一一致 |
| 预填输出 | 测试前用非零值填满 dedx，检查无贡献位置严格为零（或 bias 值）、旧结果消失 | 验证初始化协议与写回覆盖 |

### 3.2 构造实现模式（numpy）

```python
import numpy as np
rng = np.random.default_rng(seed)                      # 每用例独立且确定
w = rng.uniform(-1, 1, wShape).astype(dtype)           # weight (Cout,CinG,Dk,Hk,Wk)
dy = rng.uniform(-1, 1, dyShape).astype(dtype)         # dedy (N,Cout,Dout,Ho,Wo)
b = rng.uniform(-1, 1, (Cin,)).astype(np.float32)      # bias（转置；无 bias 用例传 None）
w.tofile(f"{case}_w.bin"); dy.tofile(f"{case}_dy.bin"); b.tofile(f"{case}_b.bin")
```

> 参考：opstest 用例表 input_data_ranges 字段语义、算子工程 assets 的 input 脚本（「参考代码指引」）。

## 4 golden 生成方法

### 4.1 基准实现选择

| 优先级 | 基准 | 适用 |
|--------|------|------|
| 1 | **CPU torch（`F.conv_transpose3d`）**，输入抬 **fp32** 计算 | 标准选择；两个场景共用同一基准——输入梯度与转置卷积的数学都等价于转置卷积，且 **CANN filter 布局 `(Cout,CinG,kD,kH,kW)` 可直接作为其 weight，无需转置**（开源 golden 即此做法） |
| 2 | numpy 手写散射/聚集（按 [../language.md](../language.md) 的坐标映射公式） | 无 torch 环境时；实现后须与 torch 对拍一次 |

两场景差异仅在接口侧：输入梯度把 dedy 作参考输入、按目标 dedx 形状裁剪输出；转置卷积把 x 作参考输入、bias 传入参考实现、q 直接作为参考实现的 output_padding。

### 4.2 计算精度：输入抬 fp32

golden 计算一律在 fp32（或更高）进行，**不允许直接用 fp16 数组做参考计算**：

```
x_fp32 = x_fp16.astype(np.float32)   # 同 weight；bias 本为 FP32
y_fp32 = F.conv_transpose3d(x_fp32, w_fp32, ...)   # 累加链无 fp16 舍入
y_fp16 = 溢出截断(y_fp32).astype(np.float16)        # 仅最后一步量化（第 4.5 节）
```

### 4.3 属性归一（与算子实现严格对齐）

golden 必须先做与 kernel 侧相同的属性归一化，再进参考实现：

1. **strides/dilations 长度归一**：原型用五元组（N/C 分量为 1），golden 解析时取空间分量；
2. **pads 六元组拆分**为逐维 `(front, back)`；
3. **非对称 pad 拆分**——参考实现只接受对称 padding：对称部分 `min(front, back)` 传参考实现 padding，**非对称余量从输出切片切除**（前侧余量 `max(0, front-back)` 从输出头部切除，后侧余量从尾部切除）。切片后的坐标与 kernel 侧的 `i = o·s − p⁻ + k·δ` 一致；
4. **output_padding**：q 直接作为参考实现的 output_padding（单侧末尾扩展，`q < stride or q < dilation`）；显式目标形状超出 `base+q` 的余量 rho 同样由 output_padding 表达（`rho < stride`），超出部分切片到目标。

### 4.4 维度与格式归一

- 1D/2D：升维到 3D（与算子侧 [../preprocessing.md](../preprocessing.md) 同一规则），计算后按需降回；
- NDHWC/DHWCN 输入先 transpose 回 NCDHW 再计算（按坐标 permutation，禁止 reshape）；
- 片上格式（FRACTAL_Z_3D 权重、NDC1HWC0 特征）按开源 golden 的转换函数归一回逻辑格式后再计算（转换实现见「参考代码指引」）；
- 分组：参考实现的 groups 直传；filter 的 `(Cout, CinG, ...)` 布局天然匹配。

### 4.5 输出 dtype 处理

fp32 结果转 fp16/bf16 前**先做溢出截断**（clip 到 ±65504）再 cast，避免参考实现自身产生 inf 污染比对：

```python
FP16_MAX = 65504.0
y_fp16 = np.clip(y_fp32, -FP16_MAX, FP16_MAX).astype(np.float16)
# 落盘为 fp32（保留比对精度）：y_fp16.astype(np.float32).tofile(golden_bin)
```

### 4.6 golden 代码骨架（torch 基准，完整可抄）

```python
def golden_dx_or_transpose(x_fp16, w_fp16, bias_fp32, stride, pad, dil, groups,
                           output_padding, target_shape):   # pad 为逐维 (front, back) 显式值
    import torch, torch.nn.functional as F
    import numpy as np
    x = torch.from_numpy(x_fp16.astype(np.float32))       # dedy（DX）或 x（Transpose）
    w = torch.from_numpy(w_fp16.astype(np.float32))       # CANN filter (Cout,CinG,kD,kH,kW)，直接作 weight
    b = torch.from_numpy(bias_fp32) if bias_fp32 is not None else None
    sym = tuple(min(f, t) for f, t in pad)                # 对称部分传 padding（第 4.3 节）
    out = F.conv_transpose3d(x, w, b, stride=stride, padding=sym,
                             output_padding=output_padding,
                             dilation=dil, groups=groups).numpy()
    out = crop_asym_and_target(out, pad, target_shape)    # 非对称余量与目标形状切片（第 4.3 节）
    return np.clip(out, -65504.0, 65504.0).astype(np.float16).astype(np.float32)
```

`crop_asym_and_target` 按第 4.3 节规则实现：先切片到 target_shape，再从头部/尾部切除非对称 pad 余量；完整落地（含 q 反算、hf32 模拟、片上格式转换）直接使用开源 golden（「参考代码指引」第 2/3 行）。

### 4.7 高频陷阱

| 陷阱 | 症状 | 规则 |
|------|------|------|
| **非对称 pad 只传对称部分** | 仅一侧边缘行/列整体错 | 参考实现 padding 只接受对称值；余量必须从输出切片切除，两侧各切各的 |
| **filter 想当然转置** | 全图数值错 | CANN filter `(Cout,CinG,...)` 直接就是 `conv_transpose3d` 的 weight 布局，不要再转置 |
| **q 与 rho 混用** | 转置输出形状错 | q 是属性、直接传 output_padding；rho 是显式目标形状的余量，同由 output_padding 表达后切片；两者都不改动 weight 与 K |
| **手写 golden 未对拍** | 无法区分参考错还是实现错 | numpy 手写基准必须与 torch 对拍一次（第 4.1 节优先级 2 的对拍义务）；对拍用未量化 fp64 结果 |
| **格式转换用 reshape** | NDHWC 用例全错 | 一律按坐标 permutation |

## 5 精度比对与通过判定

### 5.1 逐元素比对公式

比对对象：NPU 输出（fp16 读出后抬 fp32） vs golden（第 4.5 节落盘的 fp32）。逐元素判定采用**混合容差**：

```
单元素通过:  |output - golden| ≤ atol + rtol × |golden|
```

- 语义等价 `np.isclose(output, golden, rtol, atol)`；
- **NaN 规则**：output 与 golden 同为 NaN 判通过（equal_nan）；单侧 NaN 判失败；
- **尺寸不匹配**：output 与 golden 元素数不同 → 直接判失败（形状/写出量错误，不是精度问题）；
- 退化模式：rtol=0 且 atol=0 → 精确相等（位级比对，用于排错定位）。

### 5.2 用例通过判定（怎么算的）

通过判定是**两步计算**：

```
第一步：匹配率 precision = (总元素数 - 超差元素数) / 总元素数
         （超差 = 不满足第 5.1 节单元素条件的元素）
第二步：is_pass = (1 - precision) ≤ ptol    即 超差占比 ≤ ptol
```

即允许 ptol 比例的元素超差，其余必须落在混合容差内。

### 5.3 容差取值

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| rtol | 0.001（宽松档 0.01） | 与 ops-precision-standard FLOAT16 档（2^-9≈1.95e-3）一致 |
| atol | **优先取用例表携带值**（既有用例库常见 0.001 档）；无源值时 1e-8 起步 | atol 吸收 golden≈0 处的绝对误差 |
| ptol | 0.001（宽松档 0.01） | 经验值 |

**容差来源的优先级**：具体 atol/rtol、允许误差点比例、最大误差、特殊值规则优先读取原始用例和工作流；缺省标准读取 ops-precision-standard。不能把 `(rtol,ptol)` 当作 `(atol,rtol)`，也不能以新默认值覆盖原用例阈值。`atol=1e-8` 只用于首轮严排查，不得作为验收判据——大 K（长归约链）用例的 fp16 输出在小数值输出点天然存在 1~2 ULP 的相对误差。**容差只允许向"更严"方向回归收紧，放宽须给出误差分布证据**。

### 5.4 比对代码骨架（numpy，完整可抄）

```python
def verify(output_fp16_bin, golden_fp32_bin, rtol=1e-3, atol=1e-8, ptol=1e-3):
    out = np.fromfile(output_fp16_bin, dtype=np.float16).astype(np.float32)
    gold = np.fromfile(golden_fp32_bin, dtype=np.float32)
    assert out.size == gold.size, f"size mismatch: {out.size} vs {gold.size}"
    ok = np.isclose(out, gold, rtol=rtol, atol=atol, equal_nan=True)
    precision = ok.sum() / out.size
    is_pass = (1 - precision) <= ptol
    return is_pass, {"matched_ratio": precision, "max_abs": np.abs(out - gold).max(),
                     "max_rel": (np.abs(out - gold)[np.abs(gold) > 1e-6] / np.abs(gold)[np.abs(gold) > 1e-6]).max()}
```

### 5.5 容差校准方法

新算子首轮验证建议：先用严容差（rtol=1e-3, atol=1e-8, ptol=0）跑基础集，观察误差分布：

- 误差集中在 1~2 ULP（fp16 约 1e-3 量级）→ 正常舍入，按第 5.3 节定容差；
- 误差随归约长度 K 线性增长 → 累加顺序差异，正常，atol 按 `K×ULP×|x|` 量级放宽；
- 误差出现结构性分布（仅边缘、仅某通道、仅尾块区域）→ **实现 bug，不许靠放容差掩盖**，转第 7 节定位。

## 6 参考代码指引

本范式的验证方法论不依赖任何工具执行；以下代码作为**实现参考**（input 构造、golden 生成、精度比对的具体工业实现）：

| 参考代码 | 位置 | 借鉴点 |
|---------|------|--------|
| ttk 精度比对实现 | `gitcode.com/cann/ops-test-kit` → `ttk/core_modules/comparison/is_close.py` | 第 5 节公式的工业实现：isclose + 匹配率 + ptol 判定 |
| Conv3DBackpropInputV2 golden 插件 | `ops-nn/conv/conv3d_backprop_input_v2/tests/assets/golden.py` | 第 4 节的工业实现：torch conv_transpose3d 基准、非对称 pad 对称+切片、目标形状 output_padding 反算、FRACTAL_Z_3D/NDC1HWC0 格式转换、fp16 溢出截断、hf32 模拟 |
| Conv3DTransposeV2 golden 插件 | `ops-nn/conv/conv3d_transpose_v2/tests/assets/golden.py` | 同基准的转置版本：bias 直传参考实现、q 属性直传 output_padding |
| opstest 用例表 | `gitcode.com/cann/opstest`（分支 ops-nn）→ `conv/<op>/arch35/*.csv` | 第 2 节的取值来源：既有卷积用例的属性组合与容差登记（rtol/atol/ptol 携带值） |
| 五参数 tiling key 真值 | 两工程的 `op_kernel/*_arch35_tiling_key.h` | key 编码与合法组合 |
| 范式自带直调参考实现 | [adapters/ascendc/example/conv3ddx_direct/](../../adapters/ascendc/example/conv3ddx_direct/README.md)（本范式生成的算子代码，20 例验证全绿） | 实现结构、编译配方、测试口径与代码段↔章节映射 |

golden 插件以 `__golden__` 字典注册到 ttk（kernel 模式按算子名路由）；具体算法与参数须按当前算子规格重新核定。

## 7 失败定位

### 7.1 信息收集

- 失败用例的 dump 产物（input/golden/output 三方 bin）、执行日志（tiling key、blockDim、tiling 字段、首个差异点坐标与期望/实际值）、复现命令固定 seed；
- 差异的空间分布（哪些 batch/通道/深度/行列）、差异量级（ULP 级 vs 错数据级 vs 全零）。

### 7.2 差异分布 → 根因映射（范式经验表）

| 差异分布 | 高概率根因 | 核对点 |
|---|---|---|
| 全图大幅错误 / 全 0 | dtype 错配、核类型未声明（AIC 守卫被剔除）、L0C 初始化 | 检查五参数/key、核类型声明、cmatrixInitVal；binary 错配（kernel 名与输入 dtype 核对） |
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
- **仅转置路径**：以单核点（dk=hk=wk=1）场景为基准——Host 判定优先 loadB2Condition=3（五参数 `(3,0,0,true,0)`）、GM→L1B 按 Co×Ci 矩阵读取（父布局 `(Cin1,Cout1,Cout0,Cin0)`）、L1B→L0B Cout1 正序且转置、stride/pad 仍决定采样、驻留按实际 Tiling 模板检查；
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

**② 通俗性——表格化、逐项列举、图示优先**：能用表格的不用段落；能逐项列举的不用长句；数据流/流程/时序用 ASCII 图；一段只讲一个要点；术语首次出现必须可追溯到 [../language.md](../language.md) 的记号定义。

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
| Host 计划 | 实际 templateId、五参数/key、base*、singleCore*、step*、五路份数、初始化标志和容量检查；转置卷积的 q/rho/形状模式/isBiasFullLoad |
| 路径覆盖 | 用例到三个模板、三个 weight 变换、初始化及驻留命中/失效的映射；以 Host 最终选择证明命中 |
| 设备与调用 | 编译命令、真实设备实例/加载来源、直调或注册入口、初始化与主计算的执行顺序 |
| 数值与再生成 | 独立 Golden、原精度配置、重复执行结果、重新生成目录及完整验收结果 |

按自然可达路径建立覆盖，不要求不可达组合；未覆盖项记录其准入条件与后续用例。正确性通过后，分别记录初始化、主计算和端到端耗时，固定输入、平台、预热与统计方法；l1-resident 另统计首次 L1B 装载和后续复用，数值通过不能证明复用次数或性能收益。

## 10 检查点

- [ ] 用例表同时覆盖 L1 属性组合与 L2 tiling 状态（三模板、三 weight 变换、尾块、驻留）
- [ ] 验证范围（FP16/BF16、双格式、groups、output_padding 约束）与当前范式能力声明一致
- [ ] 输入按构造规则生成（固定种子、立即量化落盘、格式按坐标转换）
- [ ] golden 用 torch conv_transpose3d 基准（CANN filter 直接作 weight），属性归一与 kernel 侧口径逐项核对（pad 侧别、非对称切片、q/rho）
- [ ] numpy 手写 golden 已与 torch 基准对拍（对拍用未量化 fp64）
- [ ] 通过判定按"逐元素公式 + 匹配率"两步计算，容差取值有依据（校准记录在案）
- [ ] 无贡献位置预期明确（DX 为零、Transpose 为 bias 值），预填非零输出检查
- [ ] 产物目录完整保留（log/data/result/error_cases/report），失败用例入回归集
- [ ] 性能数据注明通路与口径
- [ ] 失败用例完成根因分类并履行回写义务（错误/缺失必回写，笔误入回归集）
- [ ] 回写内容符合四条要求（通用性/通俗性/完整性/隔离性），非本文档无用例字眼
