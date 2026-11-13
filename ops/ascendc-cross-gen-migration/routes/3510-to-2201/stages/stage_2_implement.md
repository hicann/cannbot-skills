# 阶段 2：迁移改造

**目标**：让代码在 arch22 上"语义等价地跑起来"，并留下可追溯的变更账本。本阶段不做性能调优（性能在阶段 5，但预算/tiling 必须在这里就正确）。

## 必读

- `[LOADED] api-migration-recipes`
- `[LOADED] resource-budget`
- `[LOADED] build-and-config`

## Step 2.1 冻结语义基线

在改任何代码前写下：公式、dtype、layout、边界行为、精度口径、A5 侧的特殊处理（FTZ/eps/稀疏/int4）。后续每个改动都要能对上基线。

## Step 2.2 先删（按依赖顺序）

1. A5 专有**数据类型**：从 `_def.cpp`、`*_binary.json`、kernel 模板、测试里一起删/替。
2. A5 专有**同步与控制**：`SetCtrlSpr/GetCtrlSpr`、模式 4、BufferID 隐式同步 → 显式事件。
3. A5 专有**数据通路**：`isToUB`/`dualDstCtrl`/`subBlockId`/SSBuffer/NDDMA/CCU。
4. A5 专有**Cube 参数**：`FixpipeParamsArch3510`、`LoadData2DParamsV2`（若 A2 无）、双主配置。
5. A5 专有**精度规避**：`PRECISION_*FTZ_FALSE`、`XxxSpecificMode`、A5 专有 `XxxConfig` 模板参数。
6. A5 专有**工程接线**：`arch35`、`ascend950`、`IsRegbaseSocVersion`、40KB SIMT 预留。

删除时**同步删除其依赖的头文件、宏、tiling 字段**，避免"删了调用但配置还在"。

## Step 2.3 再建（按等级）

### D1 接线级

- 建 `op_kernel/arch22/`（或改现有目录），内容为 A2 可用实现。
- 改架构宏、soc、`AddConfig`、config 目录、CMake、构建命令。
- 删除只服务 A5 的 `opFile`/`_apt` 入口。

### D2 通路级

- **L0C→UB → workspace**：`Fixpipe` 写 workspace，`DataCopy` 回 UB；`GetWorkspaceSizes` 同步改。
- **UB→L1 → UB→GM→L1**（或改数据组织让 L1 从 GM 装）。
- **SSBuffer → GM 参数区**：分配固定偏移，写方 `SetFlag` 后读方 `WaitFlag`。
- **NDDMA → 循环搬运**：先在 GM 连续化再搬，或按维循环 `DataCopy`。
- **模式 4 → 模式 2**：重新核对参与集合，避免死锁。
- **Cube 参数重算**：`FixpipeParamsV220` 字段、NZ→ZZ、`LoadData` stride。
- **dtype 迁移**：按阶段 1 决策实现（反量化/量化/不支持）。

### D3 重写级

- 按 `api-migration-recipes.md` §2 把每个 `__VEC_SCOPE__` 段重写为 MemBase。
- SIMT 按决策树改 SIMD 或 AICPU。
- 重写后**逐段对照基线**（输入输出、边界、中间结果），不要整段"重写后再说"。

## Step 2.4 预算与 tiling

1. 按 `resource-budget.md` 重算 UB/L0C/BT。
2. 删除 A5 的 40KB SIMT 预留；`ubSize` 用 `GetCoreMemSize`。
3. 重算 `tileElems`、`baseM/baseN/baseK`、`usedCoreNum`、`workspace`。
4. 用 `a2_ub_budget.py` 复核，超限则缩 tile（记录缩放比例与尾块占比）。

## Step 2.5 工程接线

按 `build-and-config.md` §1 的 10 项逐条改；改完 `grep` 兜底：

```bash
grep -rn "arch35\|ascend950\|3510\|__CCE_AICORE__ == 310\|SIMT_UB_SIZE_BYTE\|IsRegbaseSocVersion" \
  --include=*.cpp --include=*.h --include=*.hpp --include=*.cmake --include=CMakeLists.txt \
  --include=*.json --include=*.ini --include=*.sh .
```

期望：除"注释里说明历史"外无命中。

## Step 2.6 静态自检

```bash
python3 scripts/scan_a5_only.py --root "${OP_REPO}" --fail-on blocking --format md --out migration_scan_after.md
# 同时检查报告里的 skipped 清单：被跳过/超限文件必须人工确认无 A5 构造
```

要求：`blocking` 清零；`verify` 项已查证；`decision` 项有记录；`advisory` 项留到阶段 5。

## Step 2.7 变更账本

用表格记录：文件 → 改动类型（删/改/新增）→ 原因 → 对应清单项 → 验证方式。交付报告要用。

## 完成判据

- 扫描器 `blocking=0`。
- 预算表通过（或明确记录了超限后的缩放）。
- 工程接线 10 项完成、`grep` 兜底无残留。
- 变更账本完整。

## Gate

```
[GATE-2] MIGRATION=PASS FILES=<修改文件数> RESIDUAL=0
```

`RESIDUAL>0`（仍存在 blocking）时输出 `MIGRATION=INCOMPLETE RESIDUAL=<n>`，不得进入阶段 3。

## TQue 使用边界（A2 常见死锁点）

- `TQue` 的 `AllocTensor` 在 depth 用尽时会**阻塞等待**，不会报错；若在循环内长期持有一个 tensor（例如每核只搬一次的 query），再逐行从同一队列分配，就会在第二/第三次分配处永久卡死，表现为 kernel 不返回、`aclrtSynchronizeStream` 挂住；
- 规则：**常驻数据用独立 `TBuf`**，TQue 只放“申请→使用→立即释放”的逐行/逐块数据；
- 若必须共用 TQue，先确认 `depth ≥ 同时存活的 tensor 数`，并保证循环内不跨迭代持有；
- 排查方法：在 kernel 入口/每个 AllocTensor 前后打点（或先只跑一行）定位卡点。

## DAV_2201 设备侧数学函数（易踩）

- A2 设备代码**没有 libm 标量** `expf/exp/logf/sqrtf/powf`；
- `#include <cmath>` 调 `expf` → `use of undeclared identifier 'expf'`；
- 改用 `__builtin_expf` 也无效，会在链接期报 `ld.lld: error: undefined symbol: expf`；
- 正确做法：把标量搬到 32B 暂存区，用 `AscendC::Exp/Sqrt/Log` 的**单元素向量指令**计算（`AscendC::Exp(dst, src, 1)`），再 `GetValue(0)` 取回。

## A2 标量暂存区（GetValue/SetValue 路径）

设备侧做 `alpha=exp(...)`、`score=dot/rms` 这类标量运算时：

- **每个标量用独立 TBuf**（各自 `Get<float>()`），不要把一个 `LocalTensor<float>` 切成多个小平面（如每 8 元素一段）再 `SetValue/GetValue`；小平面未按 32B 对齐或跨段共享基址时，`SetValue` 写到的位置与 `Exp` 读到的位置可能不一致，表现为 **数据通路（如 p）正确、标量加权通路（如 h）全错**；
- 标量平面建议 ≥64 个 float（256B），保证对齐与 `Duplicate` 余量；
- 顺序必须是：`Duplicate(plane,0)` → `SetValue(0,x)` → `SetFlag<S_V>`/`WaitFlag<S_V>` → `Exp(plane,plane,count)` → `SetFlag<V_S>`/`WaitFlag<V_S>` → `GetValue(0)`；
- 若出现“p 对、h 错”且公式核对无误，优先怀疑标量暂存区的别名/对齐问题，而不是公式。

## 尾块与非对齐 D

- 分块循环里 `count = min(chunkD, dSize-offset)`；对 `LocalTensor` 做 `ReduceSum`/`Duplicate`/`Cast` 前先把 `[count, chunkD)` 清零，否则尾块脏数据会进入归约；
- D 不一定是对齐值的倍数：验证集必须包含 `D=511`、`D=513` 这类非对齐 shape，以及 T 的尾核（`lastTPerCore`）；
- 出现“主用例过、边界用例挂”时，优先检查尾块清零与 `count` 传递，而不是公式。
