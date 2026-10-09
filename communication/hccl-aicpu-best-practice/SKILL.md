---
name: hccl-aicpu-best-practice
description: HCCL AICPU 跨层实现实践与共享技术契约。触发：已有算法设计或需核对 template、executor、selector、TopoMatch 的具体实现、检视和验证时使用。提供资源、同步、构建证据、骨架、静态检查与专项 template 检视；算法语义和 Dataflow Spec 由设计能力负责，不负责 Agent 分工、阶段流转或最终裁决。
---

# HCCL AICPU 实现实践

> Agent 职责、任务状态、权限和完成判定由 `hccl-op-dev` Plugin 负责；跨层约束与验收契约见[共享跨层契约](aicpu-shared-contract.md)，基线与候选构建细节按需读[构建验证参考](references/build-verification.md)。
> template 实现规则见 [规则总纲](references/07-rules.md)；跨层接线见 [executor/selector](references/executor-selector.md) 和 [架构契约](references/architecture.md)。专项 template 检视使用[内置指南](review/GUIDE.md)及 `review/scripts/review_template.py`。脚本是静态筛查，不替代源码检视、编译和双轨验收。

## 0. 这是什么

HCCL 里的 **template ≠ C++ 模板**。它是分层架构最底层的**执行编排单元**：一个具体通信算法在某个执行引擎上的实现，负责"这一轮数据，谁往谁发、走哪条 channel、跑在哪个 thread、什么时候同步"。

```
hccl.h API → <op>.cc → Selector(算法名) → ExecutorRegistry → Executor → ★Template★
                                                                  ↑              ↑
                                                        "分几次搬、每次多大"   "这一次怎么搬"
```

本 Skill 只覆盖 **AICPU 引擎**。下文详细规则针对 `ins_temp_*`（基类 `InsAlgTemplateBase`）；executor、selector 和 TopoMatch 按上方跨层契约及目标源码核对，不把 template 规则外推到其他层。CCU（`ccu_temp_*`）/ AIV（`aiv_temp_*`）不在范围内。

## 0.4 能力调用模式

根据技术语义分为三种模式；**只有模式 C 把 spec 作为实现前置条件**：

| 模式 | 技术范围 | 能力契约 |
|---|---|---|
| **A 只读** | 检视现有 template、解释 API/概念、排查 CMake 接线、查存量 | 不需要 spec；输出源码依据、清单或静态检查结果 |
| **B 局部改动** | 边界检查、日志、CMake 接线、常量、早退分支，且**不改变数据流语义** | 不需要 spec；执行相关静态检查，一旦涉及 slice/偏移/同步则重分类为 C |
| **C 新算法 / 数据流重构** | 新增 template、改 slice 划分、改 scratch 布局、增删同步点、换收发模式 | 无语义 TBD 且检查通过的 Dataflow Spec 是实现输入 |

判据只有一条：**这次改动会不会动"谁的哪段字节搬到谁的哪段字节、在哪个流上、什么时候同步"**。
会动 → C；不会动 → A/B。证据不足时可先标记为 B，一旦发现涉及偏移或同步就重分类为 C。

## 0.5 模式 C 的输入契约：已检查的数据流规格

新增算法或改变 slice、scratch、同步点、收发模式时，本 Skill 接收一份由设计能力形成、语义 TBD 已清零且通过规格与布局检查的 Dataflow Spec。只读分析和不改变数据流的局部改动不强制要求 Spec。设计阶段的算子语义、算法阶段、rank/peer、切片和选择范围以该 Spec 为输入，不在此处另定一套算法策略。

实现时逐项核对目标 executor 的五参数赋值、buffer 类型、资源申请与实际消费，并将代码的 offset、长度、repeat、scratch、thread/notify 及同步点映射回 Spec；发现设计与目标源码冲突时返回具体缺口，不自行变更算法语义。实现细节按[五参数量化说明](references/09-template-data-params.md)和[规则总纲](references/07-rules.md)检查。完整构建与 Checker 不由静态检查替代。

## 1. 能力输入与自定义算法入口

### 1.0 HCCL 仓路径输入契约

**本 Skill 不预设任何仓路径。** 所有脚本和文档里的 `$HCCL_REPO` 都指 HCCL 仓根目录
（含 `src/ops`、`build.sh` 的那一层），只接受以下已确认输入：

1. 调用方显式传入脚本 `--repo <path>`；
2. 调用方已确认 `HCCL_ROOT`，并以 `--repo "$HCCL_ROOT"` 传入或设置 `HCCL_REPO="$HCCL_ROOT"`；
3. 独立调用时已设置 `HCCL_REPO`。

输入缺失时脚本直接返回错误，由调用方补齐；Skill 不猜常见路径、不扫盘、不把示例路径当默认值。多个输入指向不同仓时，必须由调用方传入唯一路径，不静默切换。
使用前还须确认目标是 Git 仓、记录当前分支和 HEAD；若不是 Git 仓或处于 detached HEAD，将该状态显式返回调用方。

```bash
# 两个路径变量，后文一律用它们；SKILL_DIR 是本 skill 所在目录，脚本都在它下面
export HCCL_REPO=<已确认的 HCCL 仓根目录>       # 脚本不传 --repo 时会读它
SKILL_DIR=<本 SKILL.md 所在目录的绝对路径>
ls "$HCCL_REPO/src/ops" "$HCCL_REPO/build.sh" && (cd "$HCCL_REPO" && git status --short --branch)
```

> **脚本在 skill 目录里，不在 HCCL 仓里。** 一旦 `cd` 进 `$HCCL_REPO` 再写
> `python3 scripts/xxx.py` 就会 `can't open file`。**始终用 `"$SKILL_DIR/scripts/xxx.py"` 的绝对路径**，
> 这样在哪个目录下执行都一样。

### 1.1 新算法默认路径（含 Ring / Tree / 用户自定义调度）

1. 消费已确认的算法设计和 Spec，再读[实现入口与来源校验](references/10-implementation-entry.md)及[executor 接线卡](references/11-operator-contracts.md)的目标条目。
2. 用 `scripts/check_sources.py --repo "$HCCL_REPO" --op <算子> [--hcomm-repo <HCOMM仓>]`
   校验卡片来源；匹配时复用，变化或超出范围时只调查对应 executor/接口/分支。
   AllGather Parallel 另读[四阶段契约](references/12-allgather-parallel.md)，校验时加 `--contract all-gather-parallel`。
   AllGather 的注册/选路/初始化查[接入卡](references/allgather-integration.md)，按需加 `--contract all-gather-integration`。
3. 按 [API 动作导航](references/02-api-reference.md#0-按开发动作选择接口) 加载所需接口，将已检查的 Spec 映射为代码。

**无需先找到同类算法、全量扫描 template 或完整阅读 NHR/Mesh。**
算子模板固定输入输出契约和工程接口，通信拓扑、阶段、peer、slice、scratch 与线程调度由本次算法设计。
已有契约不能消解的问题记录为具体缺口，定向读完整函数及必要调用方；核对完成即可继续，不扩展为全仓调研。
需要补充源码调查时，只传已有契约、来源校验结果和具体缺口；角色委派由 Plugin 决定。

修改/继承已有实现时，才按需定位并阅读相关方法：

```bash
python3 "$SKILL_DIR/scripts/list_templates.py" --op all_reduce
python3 "$SKILL_DIR/scripts/list_templates.py" --detail <已有类名>
# --blueprint 是可选的存量算法学习入口，不是新算法前置步骤
```

复用已有分层构件时按 Spec 的所属层级与兄弟 template 核对绑定 executor 及输入输出布局；注册与 executor 的实际接线见[架构说明](references/01-architecture.md#2-template-是怎么被实例化的)。

## 2. 命名与落盘

```
$HCCL_REPO/src/ops/<op>/algorithm/template/aicpu/ins_temp_<snake_name>.h
$HCCL_REPO/src/ops/<op>/algorithm/template/aicpu/ins_temp_<snake_name>.cc
```

> 2026-09 目录重构后 template/executor 统一迁入 `<op>/algorithm/` 下（入口文件也去掉了
> `_op` 后缀）。脚本对重构前的旧布局（`src/ops/<op>/template/aicpu/`）保持兼容，
> 落盘位置跟随仓内实际布局（生成器自动探测）。

- 类名：`InsTemp` + 算子 + 拓扑 + 变体 → `InsTempAllReduceMesh1DOneShot`、`InsTempReduceScatterAicpuReduceNhrPcie`
- 文件名：类名 snake_case，但仓内习惯保留 `1D` 的大写 D（`ins_temp_all_reduce_mesh_1D_one_shot.cc`）——跟同目录邻居保持一致即可
- 头文件卫士：`INS_TEMP_<UPPER_SNAKE>_H`
- namespace：`ops_hccl`
- 头部必须带 CANN Open Software License 版权块（`assets/` 骨架里已内置）

## 3. 接口契约

`InsAlgTemplateBase`（`src/ops/op_common/algorithm/template/alg_v2_template_base.h`）的实现要求分三档：

**必须实现（纯虚，不写编不过）**
- `std::string Describe() const override` — 一行自述，带 `templateRankSize_`
- `void GetNotifyIdxMainToSub(std::vector<u32>&) override`
- `void GetNotifyIdxSubToMain(std::vector<u32>&) override`

**CalcRes / KernelRun 必须实现；scratch 预算须显式声明**
- `HcclResult CalcRes(HcclComm, const OpParam&, const TopoInfoWithNetLayerDetails*, AlgResourceRequest&) override` — 申请 thread / notify / channel
- `HcclResult KernelRun(const OpParam&, const TemplateDataParams&, TemplateResource&) override` — 主编排
- `u64 CalcScratchMultiple(BufferType, BufferType) override` — CCL buffer 放大倍数，**直接决定 executor 切几次 loop**。基类默认返回 0；本 Skill 要求显式覆写以说明预算，确实不用 scratch 时返回 0

**按需实现**
- `HcclResult GetRes(AlgResourceRequest&) const override` — 实际 executor 直接消费时必须实现（如 AllGather Parallel `PrepareResForTemplate`），并与 `CalcRes` 的线程/notify 结果一致；`GetThreadNum()` 按实际调用点及基类默认值判断，见 [资源模型](references/03-resource-model.md)。首次编译前完成此项审查。
- `static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam)` — **不是 virtual**，靠 executor 模板参数静态分发；不写就没有 cost model 参与选路
- `static constexpr TemplateProp props = {.algoType = AlgoType::NHR};` — 参与选路（实现了 `CalcCostCoeff`）的 template 普遍会声明；
  `TemplateProp` **当前只有 `algoType` 一个成员**，枚举值见 `$HCCL_REPO/src/common/alg_parse.h`（MESH / MESH_ONESHOT / MESH_TWOSHOT / NHR / NHR_AICPU_REDUCE / …）

细节见 `references/03-resource-model.md`。

## 4. KernelRun：按用户算法映射

通用工程骨架在 `assets/barebone.{h,cc}.tpl`，只保留类与方法接口；
不预设全连接、归约、线程数、切片或同步顺序。资源和数据流未实现时明确返回错误，不能作为成功算法运行。

从 Spec 实现：输入与资源校验 → 各阶段的切片/传输/计算/同步 → 本阶段输出落位。
循环嵌套与同步位置由实际依赖决定，区分 executor 分块、template repeat 和算法 step。
`count==0`、单 rank、输入输出重叠的行为按算子契约确定；Barrier、变长协议尤其不能套用统一早退。

`threads[0]` 是主线程，其余为从线程；线程数量及动作分配从本算法推导。
使用主从协作时核对 Pre/Post 与 notify 配比，不套用 `threadNum == rankSize` 或“一次前同步+一次后同步”。

写地址前读 [五参数说明](references/09-template-data-params.md)，使用原语前按
[API 动作导航](references/02-api-reference.md#0-按开发动作选择接口) 选择 wrapper/HCOMM 接口。
已有 Mesh/NHR 等写法见 [算法示例](references/04-template-implementation-examples.md)，仅在需要对应实现参考时加载。

## 5. 两处 CMake 登记 —— 最容易漏的一步

AICPU template 同时被编进 **host 侧 `libhccl.so`** 和 **device 侧 `libscatter_aicpu_kernel.so`**，必须两处都加：

1. `$HCCL_REPO/src/ops/{op}/algorithm/template/aicpu/CMakeLists.txt` → `set(src_list ...)`
2. `$HCCL_REPO/src/scatter_aicpu_kernel.cmake` → `add_library(scatter_aicpu_kernel SHARED ...)` 块里对应算子的段落

**只加第 1 处：host 编得过，运行时 AICPU 侧找不到符号。** 这是本仓最高频的接线 bug。

版本守卫（CANN 9.x 才编）、块的确切写法见 [CMake 接线参考](references/05-build-and-verify.md#1-两处-cmake-登记漏一处必踩坑)。

生成器会一次做完「两个源文件 + 两处 CMake」，任一处接不上就整体不落盘并非 0 退出：

```bash
# 先 --dry-run 看落点，确认无误再去掉它真写
python3 "$SKILL_DIR/scripts/new_template.py" --op all_reduce \
        --class InsTempAllReduceRing --pattern barebone --dry-run
python3 "$SKILL_DIR/scripts/check_template.py" \
        src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_ring.cc
```

- `--pattern`：`all-reduce-mesh-oneshot`（**AllReduce 专用**，只能配 `--op all_reduce`）／
  `barebone`（自定义算法通用，不预设拓扑/归约；未实现时返回错误）
- `--algo-type` 指定 `props` 的 `AlgoType`（默认按类名推断，会对着 `alg_parse.h` 校验）；
  `--compat-guard` 放进 CANN 9.x 守卫；`--file` 覆盖由类名推导的文件名（**务必核对与邻居一致**）
- `check_template.py --all` 扫全仓：ERROR 完整打印，其余按类别汇总 WARN（`-v` 看逐条）
- 新文件另跑 `check_template.py --strict-new <文件>`；存量只处理本次改动相关的问题，不为清空 WARN 扩大修改范围。
- `check_template.py --rule <ID>` 查询实现规则；设计 Skill 的 `check_spec.py --rule <ID>` 查询规格规则。跨能力规则登记由仓库测试维护。

## 6. 能力输出与共享验证契约

本 Skill 产出 template 层可复用证据：

- 模式分类，以及模式 C 的 Dataflow Spec、`check_spec.py` 和 `check_layout.py` 结果；
- template 源码/生成预览、两处 CMake 接线状态与 `check_template.py` 结果；
- spec 与代码的 offset、length、repeat/stride、scratch、thread/notify 和同步映射核对结果；
- 未解决的上游输入缺口、外部环境依赖和未验证范围。

完整构建、基线/包归档、安装和 hccl-vm A/B 双轨的客观标准由[共享跨层契约](aicpu-shared-contract.md)和[构建验证参考](references/build-verification.md)定义，执行与阶段裁决由调用 Agent/Plugin 负责。本 Skill 的静态检查、布局样例和离线回归不替代完整构建或 Checker，也不单独给出“任务完成/可提交”结论。

技术接口和证据字段见 `references/05-build-and-verify.md`；共享跨层契约与 template 细则冲突时以跨层契约为准。

## 7. 能力检查清单

**语义**（静态脚本只能覆盖可解析部分）
- [ ] （模式 C）spec 已过 `check_spec.py`，影响语义的 TBD 都已消解；代码与 spec 的 offset/len/流/同步位置逐行对得上
- [ ] `CalcScratchMultiple` 与实际写进 hcclBuff 的最大字节数一致，且**按单次 repeat 报**（没自己乘 `repeatNum`）
- [ ] repeat 覆盖与 executor 输入契约一致；输入/输出分别核对 rank 项和 repeat 项，scratch 单独推导
- [ ] 五参数量化样例通过 `check_layout.py`；期望偏移独立来自布局，公式与 C++ 地址计算一致（`R-FLOW-021`）
- [ ] `CalcRes` 的 `slaveThreadNum` / `notifyNumPerThread` / `notifyNumOnMainThread` 与 `GetNotifyIdx*` 长度自洽

**边界**
- [ ] 对实际存在的除法/取模/索引路径设置 `templateRankSize_` 守卫；`count_ == 0` 按 executor 输入契约判定
- [ ] 64-bit（INT64/UINT64/FP64）与 `HCCL_REDUCE_PROD` 走 AICPU 软归约兜底（`references/04` §5）
- [ ] `channels.at()` 前有 `channels.count()` 校验；返回 `HcclResult` 的调用都用 `CHK_RET` / `CHK_PRT_RET` 包住

**接线与证据接口**
- [ ] spec 的启用方式、支持矩阵、资源申请与代码一致；新增文件已过 `--strict-new`
- [ ] `props` 是 `{.algoType = AlgoType::X}`（**没有 isNhr 这个成员**，写了编不过）
- [ ] 两处 CMake 都加了，`check_template.py` 无 ERROR
- [ ] 已向调用方返回 spec/布局/静态检查结果、源码身份以及未验证范围，供完整构建和双轨验收消费

## 8. 按需加载的参考

| 文件 | 何时读 |
|---|---|
| `references/10-implementation-entry.md` | 已有设计和 Spec 的实现入口：来源校验、骨架及接口缺口 |
| `references/11-operator-contracts.md` | 只读目标算子的语义与 executor 布局，不绑定已有算法 |
| `references/01-architecture.md` | 需要搞清 selector→executor→template 契约、executor 传的 repeat/stride、V1/V2 两代差异 |
| `references/02-api-reference.md` | 查 wrapper 原语签名、DataSlice/TemplateDataParams/ChannelInfo 等结构体字段、HCOMM 直调原语与 dlsym 机制（§9） |
| `references/03-resource-model.md` | 写 CalcRes / GetRes / CalcScratchMultiple、算 thread 和 notify、倍数与 repeatNum 的关系 |
| `references/12-allgather-parallel.md` | 复用 AllGather Parallel：四阶段布局、GetRes、algName 透传与多 loop 预算 |
| `references/04-template-implementation-examples.md` | 可选：对照已有 Mesh/NHR/Gather template 的 C++ 实现写法，不替代算法设计 |
| `references/05-build-and-verify.md` | CMake 接线细节、版本守卫、Template 静态结果和共享证据交接字段 |
| `references/06-template-catalog.md` | 可选：查已有实现、继承关系与绑定 executor；命名约定 |
| [设计能力的数据流规格](../hccl-aicpu-design/references/07-dataflow-spec.md) | 仅在需重新起草或修订 Spec 时加载；已有合格 Spec 可直接作为输入 |
| `references/08-thread-sync-patterns.md` | 多线程算法的**主从分配与同步顺序**：三级同步架构、notify 配比与索引约定、53 模板线程分配全量对照、8 种同步变体与设计规则 |
| `references/07-rules.md` | template 规则 ID、等级、适用范围与架构决策字段；不能覆盖 Plugin 流程与共享验收契约 |
| `references/09-template-data-params.md` | **写地址前必读**：五参数定义、布局量化、边界与 PR 2901 Ring 漏 rank 项案例 |

## 9. 目录内容

```
scripts/check_sources.py    对照 source-baseline.json 校验目标算子及公共接口来源，只读、不扫描全仓
scripts/test_checks.py      离线回归：来源校验、新旧布局生成、严格检查与 CMake 注释（仅写临时目录）
scripts/cmake_utils.py      生成器和检查器共用的 CMake 有效条目解析
scripts/list_templates.py   实时扫描存量 template（--blueprint / --op / --topo / --grep /
                            --detail / --variants / --all-engines / --format md）
scripts/new_template.py     生成 .h/.cc + 两处 CMake 接线（--dry-run / --algo-type /
                            --compat-guard / --file / --force）；任一处接线失败则整体不落盘、退出码非 0
scripts/check_template.py   规范与接线检查（单文件或 --all 扫全仓）；ERROR 退出码 1，WARN 按类别汇总
scripts/compile_probe.py    selftest 内部编译参数重写：移除原对象/依赖输出，源码 include 转向隔离树
scripts/selftest.sh         回归测试：实现脚本 + 生成器故障注入 + 两份骨架真编译（默认隔离 worktree，
                            不碰工作树也不写仓里的 build/；--in-place 就地跑，
                            --with-aicpu-build 加验 device 侧，--strict 让「没验成」也非 0 退出）
assets/all-reduce-mesh-oneshot.{h,cc}.tpl  AllReduce 专用的 Mesh 1D one-shot 完整实现骨架
assets/barebone.{h,cc}.tpl       自定义算法接口骨架，不预设拓扑/归约，未实现时返回错误
../hccl-aicpu-design/dataflow-spec-template.md  数据流规格模板 + 填写说明 + 记法速查（模式 C 的输入）
../hccl-aicpu-design/assets/dataflow-spec.example.md  已核验范例，逆向自 all_reduce mesh one-shot
assets/_license.tpl              CANN Open Software License 版权头
references/                      按需加载的参考（规则总纲与线程同步资料分别保留）
```

生成器 `new_template.py` 会写目标 HCCL 仓；检查和清单脚本只读。是否允许写入由调用 Agent/Plugin 的权限契约决定，本 Skill 不扩大授权。
`selftest.sh` 会往 HCCL 仓生成 probe 文件并编译——默认在 `git worktree` 开的**隔离副本**里做，
不碰原工作树；`--in-place` 会改动工作树，且要求两处 CMake 干净、结束按原始字节还原。调用方必须在执行前具备对应写入授权。
`new_template.py` / `check_template.py` / `list_templates.py` 接受 `--repo`（或环境变量 `HCCL_REPO`），`selftest.sh` 接受位置参数仓路径——**没有内置默认路径**，缺了直接报错；
路径由调用方显式提供，见 §1.0。改完本 Skill 后用 `bash "$SKILL_DIR/scripts/selftest.sh" "$HCCL_REPO"` 自查。

隔离编译要求原仓没有未提交的已跟踪文件修改，且隔离树与原仓 HEAD 一致；否则报告 SKIP。源码 include 路径转向隔离树，原仓 `build/` 中的生成头仍复用现有配置；不支持安全重写的编译器转发/额外输出参数也报告 SKIP。

两份骨架用仓内真实编译参数（`compile_commands.json`，含 `-Werror -std=c++17`，**host 侧**）验证过可编译，
验证过程固化在 `scripts/selftest.sh` 里；**骨架或脚本变更后必须重跑**。
