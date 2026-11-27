# Reduction 范式归约轴预处理（axes 归一化 + 空 tensor 短路 + 四步合轴）

> **层级**：通用知识层（generic/），语言中立。
>
> 来源：adapters/ascendc/tiling-preprocess.md §2（合轴）、§3（合轴产出校验）。C++ 函数实现（`HandleEmptyTensor()` / `DropSizeOneAxes()` 等四步函数）在适配层。

## 1 axes 归一化

将用户传入的归约轴参数（名称因算子而异：axes/axis/dim/dims 等）解析为合法、规范化的归约轴下标集合。步骤**顺序不可换**：

1. **空 axes 语义判定**：按算子规格和指定框架的具体接口确定显式 `axes=[]` 的含义。接口定义了 `noopWithEmptyAxes` 时按其取值处理。例如对输入 `x=[-3, 4]`，TensorFlow 的 `tf.math.reduce_euclidean_norm(x, axis=[])` 保留两个元素，得到 `[3, 4]`。PyTorch 的 `torch.linalg.vector_norm(x, dim=[])` 对整个向量取范数，得到 `5`。**必须给出本算子的单一结论，不能罗列两种可能**
2. **dtype 解析**：axes 输入可能是不同宽度的整数类型，按实际 dtype 读取数值
3. **越界校验**：对原始值校验 `v < -xRank || v >= xRank` → 报错（须在转正之前——如 `-xRank-1` 转正后变成 `-1`，会漏检）
4. **负数转正**：`v < 0` → `v + xRank`（负数表示从末尾倒数）。归一化后所有轴下标落在 `[0, xRank)`
5. **重复轴处理**：在负轴转正后按算子规格处理重复轴；允许重复轴的去重后继续，明确禁止重复轴的才报错。不能用通用拒绝规则替代算子约定
6. **排序**：去重后升序排序，便于后续合轴按轴顺序处理

归一化完成后，遍历输入 shape，按下标是否在归约集合中标记 A/R 类型，构建初始 A/R 类型表（axisShape + isReduceAxis），作为合轴四步的输入。

> shape 推导端与 tiling 端的归一化逻辑必须完全一致。

## 2 空 tensor 短路（合轴四步之前）

检查 axisShape，**必须在合轴四步之前**：

| 判定 | 条件 | 语义 | 处理 |
|------|------|------|------|
| **EMPTY_A** | 任一 A 轴 size == 0 | output 必为空 tensor | 跳转 Empty 模板，使用核数为 0，所有核早退 |
| **EMPTY_R** | 任一 R 轴 size == 0 且所有 A 轴 size > 0 | 归约结果 = `empty_r_output_value`（算子常量）；最终输出由归约后段决定 | 跳转 Empty 模板，按 aTotal 切分核数；有 PostElewise / post_reduce_input 时继续执行归约后段 |
| 非空 | 所有 axisShape ≥ 1 | 正常归约 | 进入合轴四步 |

**优先级**：EMPTY_A > EMPTY_R（A=0 时 output 必空，无论 R 是否为 0）。

> EMPTY_R 只短路归约段，不短路归约后段：`empty_r_output_value` 只替代空集合归约的结果；若算子含 PostElewise 或 post_reduce_input，仍须按 aTotal 切分核数，读取额外输入、执行 PostElewise / cast_down 后写出。只有 EMPTY_A 短路全部。常量取值见 [scene.md](scene.md) §2 Reducer 语义表。

## 3 四步合轴算法

对非空输入，按固定顺序执行四步（**顺序不可换**）：

### 步骤 1：去 1 轴（DropSizeOneAxes）

无论 A 轴或 R 轴，`size == 1` 的整根轴直接删除。

**不变量（非空保留）**：若整张图所有轴 size 都为 1（∏shape == 1），删除会清空轴列表，此时**保留 1 根占位 `A=1`** 作为结果，不允许列表为空。

### 步骤 2：合轴（FuseAxis）

连续 A 轴合并为 1 根 A 轴，连续 R 轴合并为 1 根 R 轴（size 取乘积）。

### 步骤 3：补 leading A 轴（PadLeadingOneA）

若第 0 轴是 R 轴，前置一根大小为 1 的 A 轴，保证最高轴必为 A。

**stride 计算**：合成 A=1 的 stride = `∏所有轴 size`（整段总元素数），不是 1。

### 步骤 4：补 R 增广（PadRIfPureA，纯 A 退化路径）

若上一步完成后仍无 R 轴（纯 A 场景），按 A 轴 size 分两种：

- **A=1**（全 1 退化，∏shape==1）：**末尾补** `[R=1]`，pattern 由 `A` 变 `AR`（tail-R）。此时 A=R=1 单元素归约，无性能问题；全 1 退化统一走 tail-R 路径，某些算子可因此省去 TailA 模板
- **A>1**（纯 A 但非全 1，如无归约轴的 passthrough）：**前置** `[A=1, R=1]`，pattern 由 `A` 变 `ARA`（tail-A）。⛔ 此场景必须前置 ARA，**绝不能末尾补 R=1 的 AR**——AR 路径下 tail R=1 而 A>1 会导致搬运 burst 长度塌到底，性能崩

## 4 合轴后的 pattern 形态

四步执行完后，pattern 必然形如 `A R A R …`：A 起头、A/R 严格交替、每根轴 size ≥ 2（首部补 1 的 A、紧随其后补 1 的 R 除外）。

| pattern | 轴数 | tail 类型 |
|---------|------|-----------|
| AR | 2 | R |
| ARA | 3 | A |
| ARAR | 4 | R |
| ARARA | 5 | A |
| … | … | … |
| ARARARARA | 9 | A |

**tail 类型判定**：axisNum 偶数 → tail R；axisNum 奇数 → tail A。

**轴类型判定**：合轴后 A 起头 + 严格交替，`axisType[i]` 完全由位置 `i` 的奇偶决定：`i` 偶 → A，`i` 奇 → R。

## 5 MAX_PATTERN_RANK 推导

MAX_PATTERN_RANK = 该算子合轴后 pattern 的**最大可能轴数**，用于 pattern 描述数组分配和合法性校验。

**推导方法**：

1. 按算子自身语义确定归约轴信息——来源因算子而异：可能是显式输入（axes/axis/dim/dims），可能是属性，也可能由格式隐式决定，还可能是算子公式固化的（如固定尾轴归约）。无论来源如何，最终都要得到输入张量各轴的 A/R 标记
2. **在合法输入域内枚举四步预处理的所有可达极端形态**，不能只看初始标记：
   - **非退化形态且 rseg > 0**（无 size==1 轴）：将连续的 R 轴视为 1 段，记段数为 `rseg`；合轴后 A/R 严格交替、A 起头，最大轴数 = `2 × rseg + (标记以 A 结尾 ? 1 : 0)`——以 A 结尾时达 `2 × rseg + 1`，以 R 结尾时为 `2 × rseg`
   - **纯 A（rseg=0，无 R 轴）**：不套用上式，按 `PadRIfPureA` 取：A>1 → `ARA = 3`；∏shape==1 → `AR = 2`
   - **R 轴全退化为 1 且 ∏A > 1**：所有 R 轴被去 1 删除，剩余纯 A 进入 `PadRIfPureA`；A>1 → 前置 `[A=1, R=1]` → `ARA = 3`
   - **所有轴 size 都为 1（∏shape == 1）**：去 1 后保留占位 `A=1` → 末尾补 `R=1` → `AR = 2`
3. 取以上可达形态的最大轴数：固定标记按算子合法输入域判断哪些形态可达；axes 由输入决定时按最坏情形（以 A 结尾）取 `2 × rseg + 1`，rseg 不定则兜底 9
4. 当前框架输入 rank ≤8，pattern 上界为 9；若目标框架支持更高 rank，需同步提高该上界。取 `min(推导值, 9)`

| 归约轴语义 | 合轴后 pattern | MAX_PATTERN_RANK |
|------|------|------|
| 固定标记，以 R 结尾且含 A 轴 | `(AR) × rseg`；去 1 后可达 `ARA` | `max(2 × rseg, 3)` |
| 固定标记，全 R | `AR` | `2` |
| 固定标记，以 A 结尾 | `(AR) × rseg + A` | `2 × rseg + 1` |
| 纯 A（rseg=0） | `ARA`（A 可 >1）或 `AR`（规格固定全 1） | `3` 或 `2` |
| axes 任意（rseg 不定） | 兜底 | 9 |

> 例 1：`RARA` → R0 是 1 段、R2 是 1 段 → `rseg=2` → MAX=5。合轴四步：R A R A → 补 leading A → `A R A R A`（5 轴，A 结尾达上限）。
>
> 例 2（退化反例）：固定尾 R 的 `AR`，输入 shape `(2,1)`、R 轴 size=1 → 去 1 后 `A=2` → `PadRIfPureA` 前置 `[A=1,R=1]` → `ARA=3`。因此该算子 MAX 必须取 3，不能取 `2 × rseg = 2`。

## 6 产出校验样例

**样例 1**：通用 reduce，输入 shape `(2, 3, 4, 5, 6)`，axes=[1, 3]：
- 初始类型表：A0=2, R0=3, A1=4, R1=5, A2=6；四步均无变化
- 结果：pattern = `ARARA`（axisNum=5，tail-A）

**样例 2**：全 1 退化，输入 shape `(1, 1, 1)`，axes=[0, 1, 2]：
- 去 1 轴全删 → 保留占位 `A=1` → 补 R 增广（A=1）→ 末尾补 `[R=1]`
- 结果：pattern = `AR`（axisNum=2，tail-R）

**样例 3**：EMPTY_R 短路，输入 shape `(3, 0, 4)`，axes=[1]：R 轴为 0、A 轴全正 → EMPTY_R，归约结果 = `empty_r_output_value`；无 PostElewise 时直接写出，有 PostElewise 时继续执行后处理

**样例 4**：EMPTY_A 短路，输入 shape `(0, 3, 4)`，axes=[1]：A 轴含 0 → EMPTY_A，output 为空 tensor

**样例 5**：固定尾 R 的退化形态，输入 shape `(2, 1)`，固定 reduce 最后一轴：
- 初始类型表：A0=2, R0=1
- 步骤 1 去 1 轴：R0 删除 → 纯 A=2
- 步骤 2、3 无变化
- 步骤 4 补 R 增广：A>1 → 前置 `[A=1, R=1]`
- 结果：pattern = `ARA`（axisNum=3，tail-A）；该算子 MAX_PATTERN_RANK 必须 ≥ 3
