# 阶段 1：A5 依赖盘点与定级

**目标**：把"A5 算子"拆成可执行的迁移清单：哪些必须删、哪些要用户决策、哪些可以原样保留。

## 必读

- `[LOADED] a5-only-inventory`
- `[LOADED] input-contract`
- 按需：`ascendc-regbase-best-practice`（识别 RegBase 形态）、`ascendc-simt-best-practices`（识别 SIMT 形态）

## Step 1.0 先证明“A2 缺口”真的存在

文档、目录名（`arch35`/`arc35`）和静态扫描都可能误判；开始盘点前先拿到运行时证据：

1. 构建目标 soc：`bash build.sh --pkg --soc=ascend910b --ops=<op>`；
2. 看安装包是否含目标 soc kernel：`op_impl/.../kernel/ascend910b/<op>`；
3. 直调 ACLNN（或仓库最小用例）：
   - 报 `SoC version ascend910b verification failed / not configured through AddConfig` → 缺口成立；
   - `getws=0, run=0` 且结果正确 → 该算子 A2 本来可用，**不要按迁移做**，改性能/边界任务或换题。

这一步只花 10–20 分钟，可避免把“已经支持 A2 的算子”当成迁移任务做完整轮次。

## Step 1.1 静态扫描

```bash
python3 scripts/scan_a5_only.py --root "${OP_REPO}" --format md --out migration_scan.md
python3 scripts/scan_a5_only.py --root "${OP_REPO}" --format json --out migration_scan.json
```

产出：按类别分组的命中表 + 等级初判（D1/D2/D3）+ `blocking/decision/verify/advisory` 计数。

## Step 1.2 人工补盲（扫描器覆盖不到的）

1. **宏拼接**：`#define X MicroAPI::Add` 之类，扫描器只看字面量。
2. **`#if` 死分支**：A5 分支里的代码在 arch22 编译时被裁掉，但语义可能仍被其他地方引用。
3. **构建脚本/JSON/ini**：`ascend950`、`arch35` 字符串。
4. **注释里的设计承诺**：例如"这里依赖 L0C→UB 直通，否则性能不够"。
5. **测试代码**：A5 专用 dtype/shape 的用例要改还是删。

## Step 1.3 决策项处理

对每个 `decision` 项（FP8/HiF8/MX、SIMT 替代、eps 去留、dtype 缩减）产出：

| 项 | 原语义 | 候选方案 | 精度影响 | 性能影响 | 用户结论 |
|---|---|---|---|---|---|

- 每项最多 2–3 个候选，给出推荐项与理由。
- **用户未确认前不进入阶段 2**（可在报告里标 `PENDING`，但 GATE-1 要求决策项全部有结论）。
- 用户选择"不支持"时，记录为功能裁剪，并同步修改 `_def.cpp` 的 dtype 列表。

## Step 1.4 定级

按 SKILL.md 的决策树定级：

- 命中 RegBase/MicroAPI 或 SIMT → **D3**
- 命中 A5 专有通路/同步/dtype → **D2**
- 仅工程接线 → **D1**

定级是为了排计划，不是为了"少干活"：D3 也包含 D1 的接线，D2 也可能需要局部重写。

### Step 1.4b 能力下沉最小范围（D2-lite）

若 A5 已有该语义、A2 只是被门禁挡住（非新算法），范围收敛为四处，产出"门禁原条件 → 放开条件"表并作为 GATE-1 附件：

1. host tiling 门禁（`op_host/*_tiling.cpp`）
2. aclnn/metadata 门禁（`*_metadata/op_host/op_api/*.cpp`）
3. tilingKey 组合（`*_tilingkey.h` + 入口 `*.cpp`）
4. arch22 kernel 语义分支（`arch22/*_kernel*.h`）

**边界**：arch35 与公共模板不改；仅当接口签名必须同步时才动一行。不要用"先编译看报错"代替门禁盘点，也不要为四处门禁各跑一轮全量构建。

## Step 1.5 预算初算

```bash
python3 scripts/a2_ub_budget.py --scale-from-a5 --tile-elements <A5_tile> --dtype <T> --align-bytes <n>
```

把 kernel 内 buffer 清单（A5 原值）与 A2 预算差距落表；标记"预计会超限"的项，作为阶段 2 的重点。

## Step 1.6 风险清单

至少覆盖：

- 功能缺失风险（dtype/特性不支持）
- 精度风险（FTZ 回退、Cast 语义、量化语义）
- 性能风险（多一次 GM 往返、tile 缩小、scalar 化）
- 工程风险（构建系统不支持、无 A2 先例、CANN 版本 API 缺失）
- 验证风险（无 A2 设备、无基线、用例覆盖不足）

## 产出：`migration_analysis_<op>.md`

用 `assets/migration-analysis.md.template`，必含：扫描统计、清单表、决策记录、等级、预算对照、风险、计划（谁在什么阶段改什么）。

## 完成判据

- `blocking` 项全部有明确处置方式（改/删/替）。
- `decision` 项全部有用户结论。
- 等级、预算差距、风险、计划已落盘。

## Gate

```
[GATE-1] LEVEL=D1|D2|D3 BLOCKING=<n> DECISIONS=<n>
```

`BLOCKING>0` 是正常的（待改）；`DECISIONS` 必须已全部有结论，否则输出 `BLOCKED`。
