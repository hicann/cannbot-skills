# 白盒用例补全方法

> 白盒（源码驱动）用例补全的**设计方法**：以算子代码为证据，覆盖黑盒未触及的执行分支。产物为**补充用例表 + gen_data 用例 + 分支覆盖说明**，不产 pytest/ttk。黑盒设计见 [blackbox-design.md](blackbox-design.md)；测试工程与执行见 [test-framework.md](test-framework.md)。
>
> 本文件是基类默认方法（属 virtual），各算子仓可 override；override 时保持逻辑名 `repo-test-develop` 不变。

## 定位

- **输入**：算子代码（`op_kernel/*.asc` kernel、host tiling 计算）+ 已有黑盒用例。
- **产物**：白盒补充用例（**用例表新增行 + gen_data 用例**）+ **分支覆盖说明**（记录每分支 → 命中用例 / 缺口）。**不产** pytest/ttk。
- **一句话**：黑盒据需求设计、看不到内部；白盒据源码补全需求覆盖不到的内部分支。

## 铁律

**无源码依据不成分支**：每条要覆盖的分支/边界都必须有 kernel 或 host tiling 的源码依据（行号/条件原文），读实现、不猜行为。

## 一、枚举执行分支

从 host tiling 计算与 kernel `Process` 控制流枚举**全部活分支**（广度全、不剪枝）。直调算子常见分支：

| 分支类型 | 源码特征 | 说明 |
|----------|----------|------|
| 多核边界 | `blockIdx < blockNum-1` vs 尾核 | 整核 vs 尾核处理不同数据量 |
| 尾块 | 循环内 `count=(i==tileNum-1)? tail : TILE_LENGTH` | 末个 tile 处理尾元素 |
| 非对齐 | 长度非 32B 倍数走 `DataCopyPad` | 对齐 vs 非对齐搬运路径 |
| 单核/多核 | total 元素数相对 `blockNum` 的临界 | 数据量决定是否用满核 |
| dtype 分支 | 按 dtype 走不同计算/搬运 | 各 dtype 独立路径 |
| tilingkey/模板 | `TILING_KEY` / 多模板派发（若有） | 见下 |

**若算子有 tilingkey/模板体系**：枚举 tilingkey——为单一公式（`key = 分量组合`）时路径集 = 各分量取值的笛卡尔积；为逐分支字面量时逐条记录。期望分支集 = 源码枚举出的全部活 key；`orphan = active − declared`（声明缺口）须补齐。

## 二、为每个分支反解触发用例

为每条分支反解出**能触发它的 shape/dtype/参数**：

- **尾块**：取 tile 非整除的 shape（如 `TILE_LENGTH*n + k`，`k≠0`）。
- **非对齐**：末维取非 32B / 非 dtype 对齐倍数（如末维 +1）。
- **阈值**：在切分/降级/模板选择的阈值处取 `阈值 / 阈值-1 / 阈值+1` 的 shape。
- **单核/多核**：取"不足一个核 tile" / "恰好跨核" / "远大于核数" 的 total。
- **最小路径**：单元素 `[1]` 作最小/尾块路径。

## 三、覆盖度（观测 vs 期望）

- **期望**：一、枚举出的全部活分支（有 tilingkey 则为 active_keys 集）。
- **观测**：运行期核对实际命中——直调可借日志/断点确认走了哪条分支；有 tilingkey 时可从运行日志实测 `Tiling Key` 命中。
- **产出**：**分支覆盖说明**——逐分支列出 命中用例 / 缺口，供覆盖达标判定。

## 四、数据档 clean/stress 分离

- **clean**（normal / zero / near_zero / all_ones）：必过，纳入达标必过集。
- **stress**（big / neg_big / denormal）：可能因 ULP 粗或次正规 flush-to-zero **合理失败**，**单独跑、不进硬门**，标注为信息性。

## 五、复用引擎：`ascendc-whitebox-design`（可选，复杂/tilingkey 算子）

源码分析 → 路径枚举 → tilingkey 覆盖 的重活可复用共享引擎：

- **何时用**：kernel 有 tilingkey / 多模板派发、tiling 控制流复杂。**简单算子**（普通 tiling 结构、少量分支）按一–四手工补全更轻，可不用。
- **怎么用（直调适配）**：用其源码分析与 tilingkey 覆盖能力得到路径/覆盖，**把其发射层（pytest 用例脚本）换成本仓用例表 + `gen_data`**（由 `run.sh` 编译执行）——按其路径用例反解 shape/dtype 写成直调用例表；**不产 pytest、不产 ttk CSV**。

## 落盘与下游

- 白盒补充用例 → **test 目录**（随算子工程）；**分支覆盖说明** → test 目录。
- 分支覆盖说明供后续覆盖达标判定核对；expect_error / 精度纪律见 [precision-and-perf.md](precision-and-perf.md)。
