# Convolution 范式：卷积正向算子验证方法

> **层级**：通用知识层（generic/），语言中立。
>
> **定位**：本文件回答"**卷积正向算子怎么验证**"——用例如何设计、输入如何构造、golden 如何生成、精度如何比对、通过判定如何计算、失败如何定位。全部方法论自含于本文件。
>
> 术语记号以 [language.md](language.md) 为准；循环与 tiling 语义见 [compute-flow.md](compute-flow.md)、[tiling.md](tiling.md)。

---

## 1 验证闭环总览

```
① 用例设计 ──► ② 输入生成 ──► ③ golden 生成 ──► ④ NPU 执行 ──► ⑤ 精度比对 ──► PASS
   (用例表)      (随机+约束)    (CPU 参考实现)    (拉起算子,        (混合容差)      │
                                                 kernel 直调或                    ▼ FAIL
                                                 aclnn 直调)   ⑥ 失败定位（dump/日志/反汇编）◄─┘
```

- ②③ 在 host 侧完成，④ 在 NPU 上执行，⑤ 自动比对；
- ④ 的执行拉起有两种接入形态：kernel 直调（`<<<>>>`）或 aclnn 直调（注册算子 + 两段式 API）——**拉起接入不在本范式范围**，分别由上层工作流技能保证（本仓 `ops/ascendc-direct-invoke-template`、`ops/ascendc-registry-invoke-template`）；范式侧的算法参考实现（tiling + kernel 主体）见
  [example/conv2d/](../adapters/ascendc/example/conv2d/)；

**最小验证工程的组成**（两种执行形态通用）：

| 组件 | 职责 | 实现参考 |
|------|------|---------|
| 用例表 | 内嵌于数据生成脚本（按 §2 设计，含 tiling 状态覆盖与 tag 标注） | opstest 用例表 / ops-nn 用例 input_data_ranges（§6） |
| 数据生成脚本 | 生成 input bin + golden bin（§3/§4 规则） | ops-nn golden/input 插件（§6） |
| 执行程序 | 读 bin → 设备内存 → 拉起算子 → 结果落盘 | 上层拉起工作流产出的 host 入口（ops/ascendc-direct-invoke-template、ops/ascendc-registry-invoke-template） |
| 校验脚本 | 混合容差比对（§5），产出结果 json | ttk is_close（§6） |
| 一键入口 | 编译→安装→生成→执行→校验 全串联 | 由上层工作流编排 |

## 2 验证范围与用例设计

### 2.1 当前阶段范式验证范围声明

| 维度 | 范围 |
|------|------|
| 空间维 | 1D / 2D / 3D |
| dtype | x/filter/bias/y 均为 **float16** |
| format | NCHW（2D）/ NCDHW（3D）/ NCW（1D） |
| 属性约束 | **pad < kernel**（逐维头尾均小于该维核尺寸）、**groups = 1**、stride ≤ 63（受硬件指令上限约束）、dilation ≤ 255（受硬件指令上限约束） |

### 2.2 用例设计三层次

用例设计不止覆盖"属性组合"，更要覆盖 **tiling 状态空间**——范式的每一条决策分支都应有用例压到：

| 层次 | 覆盖目标 | 设计方法 |
|------|---------|---------|
| **L1 属性组合** | 语义正确性（stride/pad/dilation/bias 交叉） | 属性组合矩阵 + 边界值 |
| **L2 tiling 状态** | 切分算法每条分支（fullLoad 四态、M/HW 模式、M-first/N-first、尾块有/无、单核/多核） | 按规格反推 tiling 状态，构造命中各状态的 shape |
| **L3 规模边界** | 性能与资源极限（L1/L0 容量临界、指令限制临界） | 容量不等式取等边界反解 shape |

### 2.3 L1 属性组合矩阵（基础集）

每个 dim（1D/2D/3D）内按下表交叉，属性至少覆盖：

| 属性 | 取值集 | 必测边界 |
|------|--------|---------|
| stride | 1；>1 对称；H/W 不对称（如 [1,2]） | stride=1 与 stride≥2 各至少一例 |
| pad | 0；对称；**不对称**（如 [1,0,1,0]）；pad=kernel-1（上界） | 不对称 pad 是 pad 拆分逻辑的高危点 |
| dilation | 1；>1（如 2） | dilation>1 时 dkH 变大，行推理随之变化 |
| kernel | 1×1(×1)；常规（3×3、5×5）；大核（如 7×7）；kernel==Hi（感受野全覆盖） | 1×1 命中退化快路径 |
| bias | 有 / 无 | 无 bias 命中 L0C 零初始化 |
| N（batch） | 1；>1 | batch>1 命中 batchIter 层 |

### 2.4 L2 tiling 状态覆盖（范式特有，重要）

按 [tiling.md](tiling.md) 的决策分支反推规格：

| tiling 状态 | 命中方法（fp16 量级参考，按 tiling.md §3.3 占用公式核算） | 验证点 |
|------------|--------------------------|--------|
| ALL 全载 | 小 shape：fmap+weight 全量 ≤ 512KB（如 1×16×8×8, Co=16, 3×3 → 合计约 7KB） | 单缓冲路径、迭代序 |
| BL1 全载（weight 主导） | 图大、Co 小：fmap 全量超 L1 而 weight 可驻留（如 1×64×112×112, Co=64, 3×3 → weight 72KB、fmap 全量 1.6MB） | M-first、A 侧流式重装 |
| AL1 全载（fmap 主导） | 图小、Co 大：fmap 全量可驻留而 weight 超 L1（如 1×256×14×14, Co=1024, 3×3 → fmap 100KB、weight 4.7MB） | N-first、B 侧流式重装 |
| NONE（三向切分） | 图与 Co 都大：双侧全量均超 L1（如 2×128×56×56, Co=512, 3×3 → fmap 约 1.6MB（2×128×56×56×2B）、weight 1.2MB） | K 向重装周期、尾块 |
| M 模式 / HW 模式 | 常规命中 M 模式；Wi>32767 或 M 模式 L1 不可行才落 HW | 模式决策正确性 |
| 通道尾块 | **Ci 非 K0 整数倍**（如 Ci=12、Ci=33）与 **Co 非 N0 整数倍**（如 Co=6、Co=34） | K 尾块补零、N 尾块 |
| M 尾块 | Ho·Wo 非 mL0 整数倍（自然出现）；刻意取 Ho·Wo=质数 | FIXPIPE 尾块写出 |
| 多核切分 | 总规模超过单核处理量，N·Co·M 与核数有/无公因子各一例 | blockDim 反解、边界核负载 |
| **K 分块尾块** | Ci 使 Ci^align/16 非 2 的幂（如 Ci∈[65,80]）**且** 宽输入（Wi ≳ 3000）迫使 kChunk 减半且产生非整除余量 | 嵌套循环逐层取余（compute-flow §6.6 原理）；内层界按尾块余量而非整块口径 |
| **宽 Wo rowWin** | Wo > 2048 且 A 侧行块超预算触发行窗口退化 | M 块大小全路径统一受 L0A 上限约束（tiling §6 校验第 6 条） |
| 1D 退化 | H=1（Hi=Kh=sH=dH=1 且无上下 pad）与 H>1 但升维的 1D 各一例 | 1D 快路径与升维路径一致性 |

> 实践技巧：先在本机跑 tiling 打印，确认构造的规格命中预期状态，再纳入用例表——避免"以为测了 NONE 态，实际落在 BL1 全载"。
> 状态覆盖口径：本表面向按 tiling.md §3.3 **完整四态**实现的生成算子；范式自带的参考实现裁剪为 BL1 驻留/NONE 两态（A 侧驻留未实现，见 example/conv2d README），为其设计用例时 AL1 全载行不适用。


## 3 输入数据构造

### 3.1 构造规则

| 项 | 约定 | 依据 |
|----|------|------|
| 分布 | 均匀分布 **U(-1, 1)**（用例表 input_data_ranges 默认 `((-1,1),)`）；可选截断正态 | 幅值有界，避免 fp16 上溢与累加精度劣化 |
| 种子 | 固定种子（如 42 或按用例序号派生 `seed=base+caseId`）；同一种子下生成序列确定 | 失败用例必须可复现 |
| 必现值 | range 第 3 元起作为必混值（如 `((-1,1,0),)` 保证出现 0） | 验证 0 乘积路径 |
| bias | U(-1,1)，长度 Co；无 bias 用例传全 0 张量（长度仍为 Co） | 全 0 偏置与"无偏置"数值等价，接口统一 |
| dtype | 全部 float16 落盘 | 与算子输入 dtype 一致 |
| 落盘 | 逐张量独立 `.bin`（fp16 原始字节序），文件名 `<case>_<input>.bin` | 执行程序按 shape 读取 |

### 3.2 构造实现模式（numpy）

```python
import numpy as np
rng = np.random.default_rng(seed)                      # 每用例独立且确定
x = rng.uniform(-1, 1, xShape).astype(np.float16)      # fmap
w = rng.uniform(-1, 1, wShape).astype(np.float16)      # weight
b = rng.uniform(-1, 1, (Co,)).astype(np.float16)       # bias（无偏置用例：np.zeros）
x.tofile(f"{case}_x.bin"); w.tofile(f"{case}_w.bin"); b.tofile(f"{case}_b.bin")
```

## 4 golden 生成方法

### 4.1 基准实现选择

| 优先级 | 基准 | 适用 |
|--------|------|------|
| 1 | **CPU torch**（`F.conv1d/conv2d/conv3d`），输入抬 **fp32** 计算 | 标准选择；fp16 输入直接 astype(fp32)，消除参考实现自身的舍入噪声 |
| 2 | numpy 手写滑窗（按 [language.md](language.md) §5 坐标映射） | 无 torch 环境时；实现后须与 torch 对拍一次 |

### 4.2 计算精度：输入抬 fp32

golden 计算一律在 fp32（或更高）进行，**不允许直接用 fp16 数组做参考计算**：

```
x_fp32 = x_fp16.astype(np.float32)   # 同 weight/bias
y_fp32 = F.conv2d(x_fp32, w_fp32, ...)   # 累加链无 fp16 舍入
y_fp16 = 溢出截断(y_fp32).astype(np.float16)   # 仅最后一步量化（§4.5）
```

### 4.3 属性归一（与算子实现严格对齐）

golden 必须先做与 kernel 侧相同的属性归一化，再进参考实现：

1. **pad_mode 展开为显式 pad**：SPECIFIC 直接取 pads；VALID 置 0；SAME/SAME_UPPER/SAME_LOWER 按公式
   `pad = (ceil(in/stride)-1)·stride + dilation·(k-1) - in + 1`（负值截 0；SAME 余数给尾/头）计算；
2. **strides/dilations 长度归一**：原型用 4 元组 `[N,C,H,W]`（空间维取后两位），golden 解析时取相同位；
3. **非对称 pad 拆分**——参考卷积 API 只接受对称 padding，两种等价写法任选其一：
   ```
   写法 A（推荐，ops-nn golden 即此做法）：全部 pad 直接补到输入（F.pad 补 0），参考实现 padding=0
   写法 B：对称部分 min(head,tail) 传参考实现 padding，非对称余量 np.pad 补到输入
   ```

### 4.4 维度与格式归一

- 1D：升维为 H=1 的 2D（输入 (N,C,L)→(N,C,1,L)，filter (Co,Ci,Kw)→(Co,Ci,1,Kw)，stride/dilation 补 1、pad 补 0），计算后降回；
- 3D：直接用 `F.conv3d`；**D 向 pad 同样走 §4.3 拆分**；
- 验证范围内输入已是 NCHW/NCDHW，无格式转换；若扩展 NHWC 等，golden 先 transpose 回 NCHW 再计算，输出再转回（transpose 前后 shape 与算子原型声明一致）。

### 4.5 输出 dtype 处理

fp32 结果转 fp16 前**先做溢出截断**（clip 到 ±65504）再 cast，避免参考实现自身产生 inf 污染比对：

```python
FP16_MAX = 65504.0
y_fp16 = np.clip(y_fp32, -FP16_MAX, FP16_MAX).astype(np.float16)
# 落盘为 fp32（保留比对精度）：y_fp16.astype(np.float32).tofile(golden_bin)
```

### 4.6 golden 代码骨架（torch 基准，完整可抄）

```python
def golden_conv(x_fp16, w_fp16, b_fp16, stride, pad, dil):   # pad 为逐维 (head, tail) 显式值
    import torch, torch.nn.functional as F
    x = torch.from_numpy(x_fp16.astype(np.float32))
    w = torch.from_numpy(w_fp16.astype(np.float32))
    b = torch.from_numpy(b_fp16.astype(np.float32)) if b_fp16 is not None else None
    x = F.pad(x, (padL, padR, padT, padB))        # 非对称 pad 全部补到输入（写法 A）
    y = F.conv2d(x, w, b, stride=stride, padding=0, dilation=dil).numpy()
    return np.clip(y, -65504.0, 65504.0).astype(np.float16).astype(np.float32)
```

### 4.7 手写 golden 的两个高频陷阱

| 陷阱 | 症状 | 规则 |
|------|------|------|
| **填充坐标系混用** | 大范围"精度不符"，但 identity-kernel 探针显示算子取数位置全对 | 给"填充后数组"切片时，起始偏移必须是**填充坐标**（未填充偏移 + pad）；有效输出区间的推导在**未填充坐标**空间完成。两者混用即错位 |
| **广播轴序错位** | 仅 N≠Co 的用例全错，N==Co 的用例恰好全对 | 向量化累加 `y += w[…] * seg[…]` 时显式写全轴序（如 `seg[:, None, …] * w[None, :, …]`），禁止依赖隐式广播对齐 |
| （配套规则） | — | **numpy 手写 golden 必须与 torch 对拍一次**（§4.1 优先级 2 的对拍义务）；对拍用未量化 fp64 结果，不能用已转 fp16 的落盘文件（其量化噪声会淹没对拍精度） |

## 5 精度比对与通过判定

### 5.1 逐元素比对公式

比对对象：NPU 输出（fp16 读出后抬 fp32） vs golden（§4.5 落盘的 fp32）。逐元素判定采用**混合容差**：

```
单元素通过:  |output - golden| ≤ atol + rtol × |golden|
```

- 语义等价 `np.isclose(output, golden, rtol, atol)`；
- **NaN 规则**：output 与 golden 同为 NaN 判通过（equal_nan）；单侧 NaN 判失败；
- **尺寸不匹配**：output 与 golden 元素数不同 → 直接判失败（不是精度问题，是形状/写出量错误）；
- 退化模式：rtol=0 且 atol=0 → 精确相等（位级比对，用于排错定位）。

### 5.2 用例通过判定（怎么算的）

通过判定是**两步计算**：

```
第一步：匹配率 precision = (总元素数 - 超差元素数) / 总元素数
         （超差 = 不满足 §5.1 单元素条件的元素）
第二步：is_pass = (1 - precision) ≤ ptol    即 超差占比 ≤ ptol
```

即允许 ptol 比例的元素超差，其余必须落在混合容差内。空输出用例（总元素数 = 0）：**空对空直接判通过**。

### 5.3 容差取值（fp16 卷积）

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| rtol | 0.001（宽松档 0.01） | 与 ops-precision-standard FLOAT16 档（2^-9≈1.95e-3）一致 |
| atol | **优先取用例表携带值**（既有用例库两级：0.005 / 0.05）；无源值时 1e-8 起步 | atol 吸收 golden≈0 处的绝对误差 |
| ptol | 0.001（宽松档 0.01，与 rtol 同档） | 经验值 |

**容差来源的优先级**（长 K 用例用 1e-8 会误判）：

 `atol=1e-8` **只用于首轮严排查**（把舍入噪声也暴露出来观察分布），不得作为验收判据——大 K（万级累加项）用例的 fp16 输出在小数值输出点天然存在 1~2 ULP 的相对误差，此时 1e-8 会把正常舍入判为失败。判失败的依据是 §5.5 的结构性分布，不是严容差下的零星超差。

> 系统标准：混合容差的完整定义、按 dtype 的阈值表与 `mixed_tolerance_check` 脚本见 `cannbot-skills/ops/ops-precision-standard/references/float_compute.md`；卷积属浮点计算类。**容差只允许向"更严"方向回归收紧，放宽须给出误差分布证据**（直方图 + 最大误差点规格归因）。

### 5.4 比对代码骨架（numpy，完整可抄）

```python
def verify(output_fp16_bin, golden_fp32_bin, rtol=1e-3, atol=1e-8, ptol=1e-3):
    out = np.fromfile(output_fp16_bin, dtype=np.float16).astype(np.float32)
    gold = np.fromfile(golden_fp32_bin, dtype=np.float32)
    assert out.size == gold.size, f"size mismatch: {out.size} vs {gold.size}"
    if out.size == 0:
        return True, {}                                   # 空对空
    ok = np.isclose(out, gold, rtol=rtol, atol=atol, equal_nan=True)
    precision = ok.sum() / out.size
    is_pass = (1 - precision) <= ptol
    return is_pass, {"matched_ratio": precision, "max_abs": np.abs(out - gold).max(),
                     "max_rel": (np.abs(out - gold)[np.abs(gold) > 1e-6] / np.abs(gold)[np.abs(gold) > 1e-6]).max()}
```

### 5.5 容差校准方法

新算子首轮验证建议：先用严容差（rtol=1e-3, atol=1e-8, ptol=0）跑基础集，观察误差分布：

- 误差集中在 1~2 ULP（fp16 约 1e-3 量级）→ 正常舍入，按 §5.3 定容差；
- 误差随 K 线性增长 → 累加顺序差异，正常，atol 按 `K×ULP×|x|` 量级放宽；
- 误差出现结构性分布（仅边缘、仅某通道、仅尾块区域）→ **实现 bug，不许靠放容差掩盖**，转 §7 定位。

## 6 参考代码

本范式的验证方法论不依赖任何工具执行；以下代码作为**实现参考**（input 构造、golden 生成、精度比对的具体工业实现）：

| 参考 | 位置 | 借鉴点 |
|------|------|--------|
| ttk 精度比对实现 | `gitcode.com/cann/ops-test-kit` → `ttk/core_modules/comparison/is_close.py` | §5.1/§5.2 的工业实现：isclose + 匹配率 + ptol 判定（本文件 §5 的公式即由此提炼） |
| ops-nn golden 插件 | `gitcode.com/cann/ops-nn` → `conv/conv2d_v2/tests/assets/golden.py` 等 | §4 的工业实现：pad_mode 展开（get_ori_pad_from_pad_mode）、全 pad 补输入写法、fp16 溢出截断（due_fp16_overflow）、格式转置归一 |
| ops-nn input 约定 | 同上 `input.py` + 用例表 input_data_ranges 字段 | §3 的取值来源：U(-1,1) 基准、必现值语义 |
| opstest 用例表 | `gitcode.com/cann/opstest`（分支 ops-nn）→ `conv/<op>/arch35/*.csv` | §2 的取值来源：既有卷积用例的属性组合与容差登记（rtol/atol/ptol 携带值） |
| 范式参考实现（tiling + kernel） | 本仓 `adapters/ascendc/example/conv2d/` | kernel/tiling 的最小完整落地（含空输出、全 pad 块、K 分块收缩）；拉起与比对工程由上层直调/验证工作流产出 |

## 7 失败定位

### 7.1 信息收集

1. 失败用例的 dump 产物：`*_input_*.bin / *_golden_*.bin / *_output_*.bin`（执行程序落盘）；
2. 执行日志：tiling key、blockDim、tiling tuple、首个差异点（元素坐标 / 期望值 / 实际值 / 差值）；
3. 复现命令固定 seed。

### 7.2 差异分布 → 根因映射（范式经验表）

| 差异分布特征 | 高概率根因 | 定位动作 |
|------------|-----------|---------|
| 仅输出边缘行/列错 | pad 处理错（FMatrix 参数、pad 归一化不一致） | 核对 golden 与 kernel 的 pad 口径；检查 padTopL1/padBottomL1 折算 |
| 仅部分通道错（尾部通道） | Ci/Co 非 N0/K0 整数倍的尾块处理 | 检查 kAL0Tail/nL0Tail 补零与 FIXPIPE 尾块参数 |
| 规律性块状错（某 mL0/nBL1 块整体错） | L1/L0 尾块越界或漏装载 | tiling tuple 反解该块坐标；检查 GM 偏移计算 |
| 全图随机小误差（1~2 ULP） | 正常舍入差异 | 容差校准 |
| 全图大幅错误 / 全 0 | dtype 错配、bias 通路、L0C 初始化 | 检查 cmatrixInitVal/bias BT 装载；binary 错配（kernel 名与输入 dtype 核对） |
| 仅 3D 的 D 边界错 | D 向 pad 软件填充 | 核对 AL1 padHead/Tail 填充 |
| 偶发/与核数相关 | 多核切分边界反解错 | 固定 blockDim 复跑；检查维度反解 |
| **2D 全对、3D 几乎全错（>99%）但数值量级正常** | B 侧 GM 寻址没按 kd 切片换算：weight [Co][Ci][Kd][Kh][Kw] 中 kd 片跨通道步进是 `Kd·Kh·Kw` 而非 `Kh·Kw`，通道块偏移也要乘 Kd | 核对 B 侧 Dn2Nz 的 srcDValue 与切片基址（compute-flow §2.2 寻址公式） |
| **仅各 M 块的尾行错、行对齐起块的块全对、错值量级正常** | AL1 buffer 按行对齐口径分配，但中间列起块的 M 块实际多跨 1 行 → A 侧 Dn2Nz 越界写穿到相邻的 BL1，尾行窗口读到被 B 装载覆盖的脏数据 | 核对 tiling 的行推理是否用最坏情形 `CeilDiv(mAL1+Wo−1, Wo)`（tiling §3.3）；对照 AL1 尺寸与逐块 hiLoad×Wi×cinA×2 的上界 |
| **执行"成功"但输出全 0、core_type 被编成 VectorCore** | Cube kernel 走 registry/aclnn 编译时 TilingKey 未声明核类型，`if ASCEND_IS_AIC` 守卫被剔除 | registry 接入的核类型声明由上层 aclnn 工作流技能保证（`ops/ascendc-registry-invoke-template`）：TilingKey 末位 `ASCENDC_TPL_KERNEL_TYPE_DECL(KT, ASCENDC_TPL_AIC_ONLY)` |

> **错值是"有效量级的错误数据"而非 NaN/全 0**——说明通路和指令都对，是**寻址/预算口径**错；配合"哪一维、哪些块错"的空间分布即可按表定位。

### 7.3 深度定位工具

- **dump 对拍**：dump 输出/输入/golden 三方 bin → 按 shape 反解差异元素坐标 → 映射回 (batch, co, ho, wo) → 观察差异的空间聚集（哪个 tile、哪个通道、块内什么位置）；
- **kernel 反汇编**：用反汇编诊断工具拆解 AIC 指令流，核对 Load3D/MMAD/FIXPIPE 的下发参数与设计值的一致性；
- **交叉验证**：同一用例 kernel 直调 vs aclnn 直调对比，区分"kernel 实现 bug"与"调度/二进制错配"。

## 8 产物管理与回归

```
run_<timestamp>/
├── log/          # 逐用例执行日志
├── data/         # input / golden / output 三方 bin（复跑与定位入口）
├── result/       # 校验结果 json / csv（检查无遗漏）
├── error_cases/  # 失败用例子集（下次复跑入口）
└── report.md     # 总结：PASS率、失败用例规格与复现方式、耗时排序
```

- 回归集 = L1 基础集 + 历史 error_cases；范式文档每次修订后全量回归；
- 报告须包含：总用例数/PASS 率、失败用例（名称/规格/复现命令）、NPU 耗时排序（性能回归观测）；

## 10 检查点

- [ ] 用例表同时覆盖 L1 属性组合与 L2 tiling 状态（fullLoad 四态、尾块、模式）
- [ ] 验证范围（fp16/NCHW-NCDHW/pad<kernel/groups=1）与当前范式能力声明一致
- [ ] 输入按 §3 构造（U(-1,1) 基准、固定种子、必现值、bias 全 0 等价）
- [ ] golden 的属性归一与 kernel 侧口径逐项核对（pad_mode 展开、非对称拆分、4 元组取位）
- [ ] golden 输入抬 fp32 计算，输出经过 fp16 溢出截断
- [ ] numpy 手写 golden 已与 torch 对拍一次（对拍用未量化 fp64）
- [ ] 通过判定按 §5.2 两步计算（匹配率 + ptol），容差取值有依据（§5.3）
- [ ] 空输出用例按"空对空"判定
