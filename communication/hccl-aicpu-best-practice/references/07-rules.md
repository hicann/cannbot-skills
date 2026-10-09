# 07 · 规则总纲（Rule Register）

本表是设计规格检查器与 template 实现检查器共用的规则维护登记，每条一个稳定 ID。设计原理见[通信算法设计](../../hccl-aicpu-design/references/08-algorithm-design.md)，Agent 职责、阶段流转与权限由 `hccl-op-dev` Plugin 定义；跨层证据见[共享契约](../aicpu-shared-contract.md)和[构建验证](build-verification.md)。其他文档解释「怎么做」，
**不重复定义规则**——要引用就写 ID（如 `R-CMAKE-002`），脚本报错也带同一个 ID。

## 0. 怎么读这份表

每条规则六个固定字段：

| 字段 | 取值 | 含义 |
|---|---|---|
| **ID** | `R-<域>-<序号>` | 稳定标识，写进脚本报错、评审意见、commit message |
| **等级** | `BLOCKER` / `MUST` / `SHOULD` | 见下 |
| **验证方式** | `自动:<脚本>` / `人工核对` / `外部环境` / `上游输入` | 本能力如何获得结论，见下 |
| **适用** | `全量` / `新增` / `新增+修改` / `模式C` | **存量欠债的出口在这一栏** |
| **规则** | 一句祈使句 | 违反它做了什么 |
| **理由** | 一句话 | 违反的后果，不是重复规则 |

**等级定义**：

| 等级 | 含义 |
|---|---|
| `BLOCKER` | 违反 → 编译失败、运行必错、或结果静默错误 |
| `MUST` | 违反 → 评审必拦；不立刻炸但会埋雷或让后人无法复用 |
| `SHOULD` | 有正当理由可豁免，豁免要在代码或 spec 里写明理由 |

**「验证方式」栏的四个取值**：

| 取值 | 含义 | 能力输出 |
|---|---|---|
| `自动:<脚本>` | 脚本可检查，报错带同一个 ID | 脚本报告及退出码 |
| `人工核对` | 脚本不能完整解析，但可从源码、spec 或公式得到结论 | 结论、证据位置和未覆盖范围 |
| `外部环境` | 需要目标仓、编译或用例环境 | 环境前提、实际结果或未验证项 |
| `上游输入` | 技术证据不能唯一确定的路径、需求语义或范围 | 结构化缺口，由调用 Agent/Plugin 按其契约处理 |

本表只说明能力依赖和证据类型，不为任何 Agent 授权，也不指定谁与用户交互。

**等级 + 适用范围 → 脚本行为**（`check_template.py` 的 `effective_level()` 就是这张表）：

| | `BLOCKER` / `MUST` | `SHOULD` |
|---|---|---|
| **适用=全量** | ERROR（退出码 1） | WARN |
| **适用=新增 / 新增+修改** | 默认 WARN；`--strict-new` 下升 ERROR | WARN |

一句话：**等级决定「被卡时有多硬」，适用范围决定「什么时候卡」**。
所以「BLOCKER + 适用=新增」不矛盾——它对新文件是硬门禁，对存量只是提示。

> `check_spec.py` 没有 `--strict-new`：spec 都是新写的，不存在存量欠债。
> 它每个检查点的 ERROR/WARN 由调用处按具体情形定，同一条规则的「反方向」提示会降一级
> （如 `R-FLOW-009` 的「没用到 SCR 却填了倍数」只报 WARN）。

**「适用」栏的意义（重要）**：仓内存量代码并不满足全部规则。
`check_template.py --all` 在当前仓能跑出上百条 WARN，绝大多数集中在 `适用=新增` 的规则上——
那是**历史欠债，不是规则错了**。因此：

- **读蓝本时**：存量文件违反 `适用=新增` 的规则很正常，**照抄它的数据流，不要照抄它的形制**。
- **写新文件时**：`适用=新增` 的规则一条都不能少。用 `check_template.py --strict-new <file>` 校验，
  它会把这些规则升到 ERROR。骨架生成后仍需 `--strict-new` 检查并处理 TODO。
- **改存量文件时**：`适用=新增+修改` 的规则对本次涉及部分生效；其他 WARN 作为存量欠债记录。

### 哪些规则是机器查的

规则数量以仓库维护测试 `tests/behavior/teams/check_hccl_aicpu_rules.py` 的输出为准；该脚本检查规则登记、等级与适用范围的一致性，不证明检查逻辑完备。
要看当前哪些能自动查：

```bash
python3 "$SKILL_DIR/scripts/check_template.py" --rule    # 列出它能查的全部规则
python3 "$SKILL_DIR/../hccl-aicpu-design/scripts/check_spec.py" --rule
python3 "$SKILL_DIR/scripts/check_template.py" --rule R-SAFE-007   # 查单条
```

静态检查只覆盖可识别的写法；通过不代表语义已验证。源码检视、编译与 Plugin 规定的双轨 Checker 仍需完成。

---

## 1. R-PATH · 路径与落盘

| ID | 等级 | 验证方式 | 适用 | 规则 | 理由 |
|---|---|---|---|---|---|
| R-PATH-001 | BLOCKER | 上游输入 | 全量 | HCCL 仓路径必须由调用方以 `--repo`、父层 `HCCL_ROOT` 或 `HCCL_REPO` 显式提供；缺失或冲突时返回输入错误 | 不猜路径、不扫盘、不静默切换仓 |
| R-PATH-002 | MUST | 人工核对 | 全量 | 目录层级不许写死。落笔前先用 `list_templates.py` 探测当前仓是 `<op>/template/` 还是 `<op>/algorithm/template/` | HCCL 重构过一次，写死的路径两种布局必坏一种 |
| R-PATH-003 | MUST | 自动:check_template | 新增 | template 落 `src/ops/<op>/[algorithm/]template/aicpu/`，`.h` 与 `.cc` 成对同目录 | 不在这个目录下，两处 CMake 的登记形式都对不上 |
| R-PATH-004 | MUST | 人工核对 | 全量 | 存量清单、数量、文件名列表一律实时解析，**不许写死进代码或文档** | 仓在演进，手抄的列表一定过期 |
| R-PATH-005 | MUST | 人工核对 | 全量 | 脚本用 `"$SKILL_DIR/scripts/xxx.py"` 绝对路径调用 | 脚本在 skill 目录，不在 HCCL 仓；`cd` 进仓后相对路径必然 `can't open file` |

## 2. R-NAME · 命名与文件形制

| ID | 等级 | 验证方式 | 适用 | 规则 | 理由 |
|---|---|---|---|---|---|
| R-NAME-001 | SHOULD | 自动:check_template | 新增 | 文件名 `ins_temp_<类名的 snake_case>.{h,cc}` | 仓内 `reduce/` 下有历史遗留不带前缀的，新文件不要跟随 |
| R-NAME-002 | SHOULD | 自动:check_template | 新增 | 类名 `InsTemp` + 算子 + 拓扑 + 变体，如 `InsTempAllReduceMesh1DOneShot` | `list_templates.py` 和 `infer_algo_type()` 都按这个前缀识别 |
| R-NAME-003 | SHOULD | 自动:check_template | 新增 | include guard = `<文件名 UPPER_SNAKE>_H` | 与文件名脱钩后，改名时漏改 guard 会撞车 |
| R-NAME-004 | BLOCKER | 自动:check_template | 全量 | `.h` 必须有 include guard 且有配对 `#endif` | 重复包含直接编译失败 |
| R-NAME-005 | BLOCKER | 自动:check_template | 全量 | `.h` 与 `.cc` 都在 `namespace ops_hccl` 内 | 符号进不了正确 namespace，executor 找不到 |
| R-NAME-006 | BLOCKER | 自动:check_template | 全量 | 文件头必须带 CANN Open Software License 版权块 | pre-commit 的 OAT 合规检查会拦 |
| R-NAME-007 | SHOULD | 人工核对 | 新增 | 文件名保留仓内不规则词元的原大小写（`1D` 不写成 `1_d`），跟同目录邻居一致 | 仓内约定；`new_template.py` 的 `CASE_ATOMS` 负责推导 |

## 3. R-IFACE · 接口契约

> 权威来源是 `$HCCL_REPO` 的 `op_common/**/template/alg_v2_template_base.h` 与
> `common_alg_template_base.h`。**这一节最容易随真仓演进失效，改动前先实读那两个头文件。**

| ID | 等级 | 验证方式 | 适用 | 规则 | 理由 |
|---|---|---|---|---|---|
| R-IFACE-001 | BLOCKER | 自动:check_template | 全量 | 必须继承 `InsAlgTemplateBase`（V2），或继承一个已有的 `InsTemp*` 做变体 | 继承 `AlgTemplateBase` 是 V1 遗留 API，走的是另一套注册与调用路径 |
| R-IFACE-002 | BLOCKER | 自动:check_template | 全量 | 带参构造签名必须是 `(const OpParam&, const u32, const std::vector<std::vector<u32>>&)` | executor 硬编码 `make_shared<T>(param, rankId, subCommRanks)`，签名不符编译失败 |
| R-IFACE-003 | BLOCKER | 自动:check_template | 新增 | 保留默认构造 `XXX() = default;` | 走 FastLaunch 的 executor 用 `make_unique<T>()` 无参构造；缺了编译失败。不走 FastLaunch 的（DPU/Intra/Inter/OmniPipe）可豁免 |
| R-IFACE-004 | BLOCKER | 自动:check_template | 全量 | 实现三个纯虚：`Describe()` / `GetNotifyIdxMainToSub()` / `GetNotifyIdxSubToMain()` | 基类里是 `= 0`，不实现编译不过 |
| R-IFACE-005 | BLOCKER | 自动:check_template | 全量 | 实现三个实质必须：`CalcRes()` / `KernelRun()` / `CalcScratchMultiple()` | CalcRes/KernelRun 基类报错；CalcScratchMultiple 基类返回 0，要求显式覆写以声明预算，不用 scratch 时可返回 0 |
| R-IFACE-006 | BLOCKER | 自动:check_template | 全量 | `props` 只能写成 `static constexpr TemplateProp props = {.algoType = AlgoType::X};`，**禁用已删除的 `isNhr`** | `TemplateProp` 当前只有 `algoType` 一个成员，designated initializer 指向不存在的成员编译失败 |
| R-IFACE-007 | MUST | 自动:check_template | 全量 | `AlgoType::X` 必须在 `src/common/alg_parse.h` 的枚举里（脚本实时解析，不写死） | 枚举值会增删，写死清单必过期 |
| R-IFACE-008 | SHOULD | 自动:check_template | 新增 | 实现了 `CalcCostCoeff` 就声明 `props` | 参与选路的 template 普遍都声明；用于声明算法家族；是否被成本模型消费以当前源码为准，不把缺省属性等同于无法选路 |
| R-IFACE-009 | SHOULD | 自动:check_template | 全量 | `props.algoType` 的拓扑家族（`MESH_*` / `NHR_*`）要与类名一致 | 对不上通常是复制粘贴漏改 |
| R-IFACE-010 | BLOCKER | 自动:check_template | 全量 | 名字带 `Dpu` 的 template 必须在 `.cc` 末尾写 `REGISTER_TEMPLATE_V2("类名", 类名);` | `template/dpu/kernel_launch.cc` 按字符串反查，缺了运行期找不到 |
| R-IFACE-011 | SHOULD | 自动:check_template | 新增 | 非 DPU template **不要**写 `REGISTER_TEMPLATE_V2` | AICPU 侧由 executor 模板参数静态绑定，字符串注册表没有消费者 |
| R-IFACE-012 | BLOCKER | 自动:check_template | 全量 | 禁用 V1 的 `REGISTER_TEMPLATE(枚举, Cls)` | V1/V2 两套注册表，混用会注册到没人查的那张表 |
| R-IFACE-013 | SHOULD | 自动:check_template | 新增 | 必须有某个 `REGISTER_EXEC_V2` / `REGISTER_EXEC_V2_MULTI` 把它当模板参数引用 | 没被引用的 template 永远不会被实例化，等于死代码 |
| R-IFACE-014 | MUST | 自动:check_template + 人工核对 | 新增+修改 | 实际 executor 直接消费 `GetRes()` 时必须实现并与 `CalcRes` 线程/notify 一致；`GetThreadNum()` 按调用点与基类默认值核对 | 脚本检查同文件注册且直接调用 GetRes 的情形；跨文件消费者、继承与数值一致性需人工核对，如 AllGather Parallel PrepareResForTemplate |
| R-IFACE-015 | MUST | 自动:check_template | 全量 | `CalcCostCoeff` 是 `static` 不是 `virtual`，入参以 `src/ops/op_common/selector/cost_model.h` 的 `CalcCostCoeffParam` 为准 | 静态分发；写成 virtual 不会被 executor 调到 |

## 4. R-FLOW · 数据流语义（红线）

> 这一域的错误**编译期一律抓不到**，只能靠 spec 评审 + hccl-vm 双轨验证。
> `check_spec.py` 能在 spec 阶段抓住其中大半——这就是「先 spec 后代码」的全部理由（`R-PROC-002`）。

| ID | 等级 | 验证方式 | 适用 | 规则 | 理由 |
|---|---|---|---|---|---|
| R-FLOW-001 | BLOCKER | 自动:check_spec | 全量 | 写模式原语（`SendRecvBatchWrite*` 等）的**目标必须**带 `@p`，**源不许**带 `@p` | 写模式底层只用 `txSlicesList_`；方向写反 = 填了不生效的参数 |
| R-FLOW-002 | BLOCKER | 自动:check_spec | 全量 | 读模式原语（`SendRecvBatchRead*` 等）的**源必须**带 `@p`，**目标不许**带 `@p` | 读模式底层只用 `rxSlicesList_` |
| R-FLOW-003 | BLOCKER | 自动:check_spec | 全量 | 跨 rank 的一条边**只写一个方向**；本地原语（`LocalCopy`/`LocalReduce`）操作数不许带 `@p` | 写第二行会诱导去填不生效的参数；对端只贡献 channel 做 notify 握手 |
| R-FLOW-004 | BLOCKER | 自动:check_spec | 全量 | 一条边的源长度与目标长度必须相等 | 长度不等 = 越界写或少搬，功能验证才会暴露 |
| R-FLOW-005 | BLOCKER | 自动:check_spec | 全量 | `sync main -> sub` 与 `sync sub -> main` 必须成对 | 缺一半 = 从流未同步就被读，数据竞争 |
| R-FLOW-006 | BLOCKER | 自动:check_spec | 全量 | 用到从流 `T[i]`（i>0）就必须有主从同步 | 同上 |
| R-FLOW-007 | BLOCKER | 人工核对 + 外部环境 | 全量 | `CalcScratchMultiple` 的返回值必须与实际写进 hcclBuff 的最大字节数一致（one-shot=`templateRankSize_`，Mesh two-shot=2，NHR=1，不用 scratch=0） | 报小了越界写，报大了 executor 白白多切 loop |
| R-FLOW-008 | BLOCKER | 自动:check_spec | 全量 | scratch 倍数按**单次 repeat** 报，表达式里**不许出现 `RPT`** | 分层 executor 自己算 `max(另一层倍数, 本层倍数 * repeatNum)`，template 再乘一遍就是双重放大 |
| R-FLOW-009 | MUST | 自动:check_spec | 全量 | 数据流用到了 `SCR`，scratch 倍数就不能填 0 | 倍数 0 表示不受 scratch 约束，与实际使用矛盾 |
| R-FLOW-010 | MUST | 自动:check_spec | 全量 | 规则寻址中 RPT≠1 或分层构件必须覆盖 repeat；特殊消费方式明确记录 | repeat 与通信 step、executor 外层 loop 不同 |
| R-FLOW-011 | MUST | 自动:check_spec | 全量 | 声明了 `RPT` 就要在第 5 章给全 `IRS`/`ORS`/`ISS`/`OSS` 的取值 | 少一个就无法验证地址推导 |
| R-FLOW-012 | SHOULD | 自动:check_spec | 新增 | 默认把 `for (rpt < repeatNum)` 循环写上；确定只服务单层且不打算被复用才可省，且要在 spec 第 5 章写明理由 | `RPT=1` 时零代价退化；省掉的代价是永远不能被复用 |
| R-FLOW-013 | BLOCKER | 自动:check_template | 全量 | `repeatNum` 与四个 stride 由 executor 每轮填入，template **只读不改** | 改了会让上层的地址推导与实际搬运脱节 |
| R-FLOW-014 | MUST | 人工核对 | 全量 | 输入/输出分别按实际布局核对 base + rank*SliceStride + repeat*RepeatStride；scratch 独立推导，特殊消费方式见五参数说明 §7 | 不把本轮长度当完整块步长，不从输入类型猜全部中转布局 |
| R-FLOW-015 | MUST | 人工核对 | 全量 | `threads[0]` 是主流：本地 `LocalCopy`/`LocalReduce` 挂主流，跨 rank 收发挂从流 `threads[1..n-1]` | 挂错流会让同步点失去意义 |
| R-FLOW-016 | MUST | 人工核对 | 全量 | `IsPcieProtocol(channels)` 为真时改用读模式原语（`SendRecvReadReduce` / `SendRecvBatchRead`） | PCIe 链路上写模式性能塌陷 |
| R-FLOW-017 | MUST | 人工核对 | 全量 | 64-bit（INT64/UINT64/FP64）与 `HCCL_REDUCE_PROD` 走 AICPU 软归约兜底 | 硬归约不支持这些组合，结果静默错误 |
| R-FLOW-018 | MUST | 自动:check_spec + 人工核对 | 全量 | `CalcRes` 的 `slaveThreadNum` / `notifyNumPerThread` / `notifyNumOnMainThread` 必须与 `GetNotifyIdx*()` 返回的长度自洽 | notify 索引错位是运行期挂死的高频原因 |
| R-FLOW-019 | MUST | 人工核对 | 全量 | `p` 是通信域 userRank（`channels` 的 key），算法内序号另用 `i`/`r`/`algRank`，**禁止混用** | 本仓高频 bug：拿算法序号去索引 channels |
| R-FLOW-020 | BLOCKER | 人工核对 | 新增+修改 | **按需**加早退：`KernelRun` 里只要存在会除以 / 取模 `templateRankSize_`、或按算法序号索引 `subCommRanks_` 的逻辑，就必须有对应守卫（通常写成 `templateRankSize_ <= 1` 早退） | 单卡场景下除零或越界。**注意存量多数 template 没有 `count_ == 0` 分支**——空输入由 executor 挡在外面，所以这条只要求「用到才守」，不是无条件都写。自动化的那半条是 `R-SAFE-007` |
| R-FLOW-021 | BLOCKER | 自动:check_spec + 人工核对 | 模式C | 第5章给出五参数赋值来源、算法 rank 映射与 layout-check 量化样例；公式对照独立期望偏移通过，Ring 覆盖非零 ISS 与算法 rank | 静态字段存在不能拦截 PR 2901 漏 rank 项；完整语义仍需人工核对及双轨验收 |
| R-FLOW-022 | MUST | 自动:check_template + 人工核对 | 新增 | 常规 AICPU template 的 scratch→out `LocalCopy` 须按绑定 executor 的 `outBuffType` 分流，并在分支旁说明在位或错位依据；DPU/OmniPipe 的 step 布局人工核对 | `HCCL_BUFFER` 输出可能已由上一跳直写，也可能仍需错位搬移；脚本仅对能追到 scratch/output `DataSlice`、同时引用两侧 base offset 却在同一方法中没有读取 `outBuffType` 的文件提示，存量 WARN、新文件 `--strict-new` 为 ERROR |

## 5. R-SPEC · 数据流规格文档

> 只在**模式 C**（新算法 / 重构数据流）生效。模式 A/B 不写 spec。

| ID | 等级 | 验证方式 | 适用 | 规则 | 理由 |
|---|---|---|---|---|---|
| R-SPEC-001 | BLOCKER | 自动:check_spec | 模式C | 10 个章节齐全，标题序号与关键字与 `dataflow-spec-template.md` 一致 | 改标题会让 `check_spec.py` 判为章节缺失 |
| R-SPEC-002 | BLOCKER | 自动:check_spec | 模式C | 开始写代码前，`TBD:` 必须清零 | 带着 TBD 写代码 = 拿猜测当规格 |
| R-SPEC-003 | BLOCKER | 自动:check_spec | 模式C | 模板占位符 `<...>` 必须全部替换 | 同上 |
| R-SPEC-004 | BLOCKER | 自动:check_spec | 模式C | 第 6 章「数据流」必须有代码块，且能解析出至少一条边 | 没有数据流的 spec 不构成输入契约 |
| R-SPEC-005 | BLOCKER | 自动:check_spec | 模式C | 每条数据流边必须有 `->` / `into` / `reduce->` 分隔源和目标（`into` = 累加，`->` = 覆盖） | 分不清累加与覆盖，归约类算法必错 |
| R-SPEC-006 | BLOCKER | 自动:check_spec | 模式C | 第 1 章「元信息」必填字段齐全（算子 / 拓扑 / 变体语义 / 所属层级 / 绑定 executor） | 缺了就不知道落哪个目录、被哪个 executor 实例化 |
| R-SPEC-007 | BLOCKER | 自动:check_spec | 模式C | 第 7 章「边界条件」每一项都要写出行为，不能只勾选 | 空着的边界项 = 未定义行为 |
| R-SPEC-008 | MUST | 人工核对 | 模式C | 一个 spec 文件对应一个 template（一对 `.h`/`.cc`）；分层算法写多份，在「所属层级/兄弟 template」里交代组合关系 | 一份 spec 描述两个 template 会让地址推导互相污染 |
| R-SPEC-009 | MUST | 人工核对 | 模式C | 起草时拿不准一律写 `TBD: <具体问题>`，**不猜** | spec 里一个 offset 写错，代码全对也是错的 |
| R-SPEC-010 | MUST | 上游输入 | 模式C | 先依据父层约束、源码和布局公式解决 scratch/线程/尾块等技术 TBD 并验证；证据无法消解的需求语义或范围取舍作为上游输入缺口 | 避免把可推导的技术问题误归为外部决策 |

## 6. R-CMAKE · 两处接线

| ID | 等级 | 验证方式 | 适用 | 规则 | 理由 |
|---|---|---|---|---|---|
| R-CMAKE-001 | BLOCKER | 自动:check_template | 全量 | 登记进**同目录** `CMakeLists.txt` 的 `set(src_list ...)` | 不登记 host 侧 `libhccl.so` 里没有这个符号 |
| R-CMAKE-002 | BLOCKER | 自动:check_template | 全量 | 登记进 `$HCCL_REPO/src/scatter_aicpu_kernel.cmake` 的 `add_library(scatter_aicpu_kernel SHARED ...)` 块 | ★ 本仓最高频接线 bug：只加第 1 处 → host 编得过，device 侧运行时找不到符号 |
| R-CMAKE-003 | MUST | 人工核对 | 新增 | 只在 CANN 9.x 才编的文件放进 `if(NOT HCCL_CANN_COMPAT_850)` / `list(APPEND src_list ...)` 块 | 放错块会在 850 兼容构建上编译失败 |
| R-CMAKE-004 | MUST | 人工核对 | 新增+修改 | 目录重命名/移动后，同步检查两处 CMake、测试 include 路径、`#include` 相对路径，并清理 build 目录重验 | 见 `$HCCL_REPO/AGENTS.md` §8 |

## 7. R-STYLE · 编码风格

> 权威来源：`$HCCL_REPO/.clang-format`、`$HCCL_REPO/AGENTS.md` §5、
> `$HCCL_REPO/.agents/skills/hccl-review/references/coding-and-security.md`。
> **本节只镜像与 template 编写相关的条目，冲突时以真仓那三份为准。**

| ID | 等级 | 验证方式 | 适用 | 规则 | 理由 |
|---|---|---|---|---|---|
| R-STYLE-001 | MUST | 自动:check_template + clang-format | 全量 | 单行不超过 120 列 | `.clang-format` 强制；pre-commit 会改回去，diff 变脏 |
| R-STYLE-002 | MUST | 自动:check_template + clang-format | 全量 | 4 空格缩进，禁用 tab | 同上 |
| R-STYLE-003 | MUST | 自动:check_template | 新增+修改 | 类名与函数名 PascalCase | `$HCCL_REPO/AGENTS.md` §5 |
| R-STYLE-004 | SHOULD | 自动:check_template | 新增 | **类**的私有/保护成员变量用小驼峰 + 尾下划线（`myRank_`）；POD `struct` 的字段**不**加下划线 | 区分成员与局部变量；仓内 `struct NHRStepInfo` 等 POD 就是不带下划线的 |
| R-STYLE-005 | MUST | 自动:check_template | 新增 | 常量与宏用 `UPPER_SNAKE_CASE` | `$HCCL_REPO/AGENTS.md` §5 |
| R-STYLE-006 | SHOULD | 自动:check_template | 新增 | 单个函数不超过 50 行；超了拆分 | `KernelRun` 常常超标，拆成 `RunIntra`/`RunInter`/`DoLocalReduce` 更好评审 |
| R-STYLE-007 | SHOULD | 自动:check_template | 新增 | 参数超过 6 个就封装成结构体 | 同上 |
| R-STYLE-008 | MUST | 人工核对 | 新增+修改 | 禁魔法数字，用命名常量（含 `2`、`8` 这类 rankSize 阈值） | `if (param.rankSize > 8)` 这类要给常量名和注释 |
| R-STYLE-009 | MUST | 自动:clang-format | 全量 | K&R 大括号、指针右对齐（`T *p`），以 `.clang-format` 为准 | 交给工具，不要手调 |
| R-STYLE-010 | SHOULD | 自动:check_template | 新增 | 日志首参带 `[类名][方法名]` 前缀，如 `HCCL_INFO("[InsTempXxx][CalcRes] ...")` | 仓内绝大多数日志都带前缀；不带前缀的日志在多 template 并发时无法定位 |
| R-STYLE-011 | MUST | 自动:check_template | 新增+修改 | 公共接口要有注释；`Describe()` 一行自述里要带 `templateRankSize_` | `Describe()` 是运行期唯一的自我标识 |

## 8. R-SAFE · 内存与资源安全

> 同上，镜像自 `$HCCL_REPO/.agents/skills/hccl-review/references/coding-and-security.md`，
> 只保留在 template 里真会踩到的条目。命中即 CRITICAL/HIGH。

| ID | 等级 | 验证方式 | 适用 | 规则 | 理由 |
|---|---|---|---|---|---|
| R-SAFE-001 | BLOCKER | 自动:check_template | 新增+修改 | `channels.at(r)` 之前必须有 `channels.count(r)` 校验 | 远端 rank 不在 map 里时 `.at()` 抛异常，AICPU 侧直接崩。存量文件中的裸 `.at()` 属于历史欠债 |
| R-SAFE-002 | BLOCKER | 自动:check_template | 全量 | 禁用 `memcpy`/`strcpy`/`sprintf`/`strcat`，用 `_s` 安全版本并检查返回值 | CANN 编码规范红线；当前全仓 template 里 0 处使用，保持住 |
| R-SAFE-003 | BLOCKER | 自动:check_template | 全量 | 搬运/同步原语的返回值都用 `CHK_RET` / `CHK_PRT_RET` 包住；`return Prim(...)` 与 `ret = Prim(...)` 是正确写法，不算违反 | 吞掉失败会让错误在下一层以更难懂的形式爆出来。脚本认 `SendRecv*` / `Local*` / `PreSync*` / `PostSync*` 这批已知返回 `HcclResult` 的原语 |
| R-SAFE-004 | BLOCKER | 自动:check_template | 新增+修改 | 变量与结构体声明时就初始化（`u32 x = 0;` / `{}` 值初始化）。**当出参传出去也要初始化** | 未初始化的 offset/len 会写到随机地址；出参函数提前返回时那个值会被后面的代码读到。仓内同一文件里 `u32 myAlgRank = 0;` 和 `u32 myAlgRank;` 两种写法都有，前者才是对的 |
| R-SAFE-005 | BLOCKER | 人工核对 | 新增+修改 | 指针解引用前判空（`buffInfo.inputPtr`、`remoteCclMem.addr` 等） | 图模式下部分指针可能为空 |
| R-SAFE-006 | BLOCKER | 人工核对 | 新增+修改 | 数组索引与拷贝长度必须有边界校验 | slice 偏移越界是 功能验证才能抓到的静默错误 |
| R-SAFE-007 | BLOCKER | 自动:check_template | 新增+修改 | 以 `templateRankSize_` / `threadNum_` 作除数或取模前，全文要有非零守卫 | 切片计算里除零会直接崩。满足 `R-FLOW-020` 的早退通常也就满足了这条。查表常量 `DATATYPE_SIZE_TABLE[dataType_]` 不在自动检查范围内（对合法 dtype 恒非零），但仍要自己确认 dtype 合法 |
| R-SAFE-008 | MUST | 人工核对 | 新增+修改 | 窄类型与 `u64`/`size_t` 混算要显式转换；不得回绕 | `count * dataTypeSize` 在大 count 下会溢出 u32 |
| R-SAFE-009 | MUST | 人工核对 | 新增+修改 | 所有路径（含错误路径）资源申请与释放对称 | `CalcRes` 里申请了 channel 但中途 return 会泄漏 |
| R-SAFE-010 | MUST | 自动:check_template | 全量 | 禁止对指针变量 `sizeof` 取数组大小 | 拿到的是指针宽度不是数据长度 |

## 9. R-VERIFY · Template 能力核对

| ID | 等级 | 验证方式 | 适用 | 规则 | 理由 |
|---|---|---|---|---|---|
| R-VERIFY-001 | BLOCKER | 人工核对 | 全量 | `check_template.py` 无 ERROR；新文件另跑 `--strict-new` 无 ERROR | 接线与接口契约的最快闸门，秒级 |
| R-VERIFY-007 | MUST | 人工核对 | 模式C | 代码与 spec 的 offset / len / 流 / 同步位置逐行对得上 | spec 过了但代码没照着写 = spec 白写 |

> 完整构建、pre-commit、基线、安装、hccl-vm 双轨和 ST/UT 权限属于 Plugin 与共享验证契约，不在本规则表重复定义。已删除的 `R-VERIFY-002`—`006`、`008`—`009` 不复用。

## 10. R-PROC · 能力使用约束

| ID | 等级 | 验证方式 | 适用 | 规则 | 理由 |
|---|---|---|---|---|---|
| R-PROC-001 | MUST | 人工核对 | 全量 | 先判模式：**A 只读** / **B 局部改动** / **C 新算法或重构数据流**。判据只有一条——这次改动会不会动「谁的哪段字节搬到谁的哪段字节、在哪个流上、什么时候同步」 | 避免对只读分析和非数据流修改强制引入 spec |
| R-PROC-002 | MUST | 人工核对 | 模式C | Dataflow Spec → 设计 Skill 的 `check_spec`/`check_layout` → 语义 TBD 清零 → 代码映射 → 实现 Skill 的 `check_template`；静态结果交由调用 Agent | 确保 spec 是实现的前置输入，不在能力层重复编排跨层验收 |
| R-PROC-003 | MUST | 人工核对 | 新增+修改 | 实现前核对目标算子/executor 契约及所需接口；来源匹配且范围相同时复用结论。修改/继承已有实现才读相关完整函数和调用关系，不要求新算法存在同类蓝本 | 保持来源可信，避免反复调研无关算法 |
| R-PROC-004 | MUST | 人工核对 | 全量 | 技术结论必须基于完整函数体、调用方和目标版本真实源码；内容未变且适用范围一致时可复用已核对记录，差异定向补查，不得仅根据 diff 或未验证推测 | 保持技术信息可追溯 |
| R-PROC-008 | MUST | 人工核对 | 全量 | 不得 `#include` HCOMM 私有头，不得引入对 `cann/hcomm` 的编译期硬依赖；跨仓调用走 `src/common/hcomm_dlsym/` | `$HCCL_REPO/AGENTS.md` §3 的硬性架构约束 |
| R-PROC-009 | MUST | 人工核对 | 新增 | 官方新算子落 `src/ops/<op>/`，社区试验算子落 `experimental/ops/<op>/`，均按 `selector` + `algorithm/{executor,template}` 组织 | 同上 |

> 修改范围、写入授权、Git 提交/推送和 Issue→PR→CI→检视链路已交由 Agent/Plugin 定义。已删除的 `R-PROC-005`—`007` 不复用。

---

## 11. R-DEC · 架构决策确定性

> **为什么单列一域**：同一句提示词、同一个 skill，三次生成出了三套互不冲突却互不兼容的实现——
> 数据流几乎一致，但在**启用方式、是否改默认选路、支持哪些 runtime 模式、申请全 Mesh 还是邻居链路、
> Cost Model 怎么标定**上各走各的，而且三份都能通过当时的全部检查。
> 前面十个域约束的是「形制」（名字、接口、接线、数据流方向），这一域约束的是「决策」。
> **决策不落定，形制再规范也会漂移。**
>
> 这些字段的载体是 dataflow spec：第 1 章（启用方式 / 默认选路影响）、第 2 章（支持矩阵）、
> 第 3 章（逻辑 peer / 申请链路 / channel 切分）、第 8 章（Cost model 策略 / 阈值证据）。
> `check_spec.py` 会强制它们存在并做交叉校验。

| ID | 等级 | 验证方式 | 适用 | 规则 | 理由 |
|---|---|---|---|---|---|
| R-DEC-001 | BLOCKER | 自动:check_spec | 模式C | spec 第 1 章必须声明**启用方式**：`标准配置显式选中` / `自动选路` / `私有开关` | 不写死这一项，模型会挑「最容易触发」的那条路，三次生成三个答案 |
| R-DEC-002 | BLOCKER | 自动:check_spec | 全量 | 默认选路不变或按批准范围改变，须声明依据、范围及选路验收矩阵 | 性能证据不能豁免存量回归 |
| R-DEC-003 | BLOCKER | 自动:check_template | 全量 | **禁止裸 `getenv` 私有开关**，一律复用仓内既有配置入口（`HCCL_ALGO` 等）。确需新环境变量：先补配置定义、解析、默认值、文档与测试 | 私有开关绕开配置体系，没有文档和默认值，还让两次生成的启用方式互不兼容 |
| R-DEC-004 | BLOCKER | 自动:check_spec | 模式C | 保留既有阈值的依据，禁止凭空生成阈值；性能不纳入本流程验收 | 按父层选路验收契约核对批准范围，不能事后改矩阵掩盖回归 |
| R-DEC-005 | BLOCKER | 人工核对 | 全量 | 按改动核对注册算法名、类名、文件名、props、枚举、DSL、DFX 与 cost model 映射；三处注册/选择名称按父层逐字符一致 | 其余名称语义对应，不要求机械同名或修改无关映射 |
| R-DEC-006 | MUST | 自动:check_spec | 模式C | spec 第 3 章必须分别声明**逻辑 peer**（算法每轮实际访问的对端）与**申请链路**（初始化阶段申请的链路）；不一致要写明过度申请理由。单环这类算法默认只申请前驱/后继 | 「用 ring 算法」不等于「要申请全 Mesh 链路」。不区分这两个概念，就会出现邻居链路与全 Mesh 两种实现 |
| R-DEC-007 | MUST | 自动:check_spec | 模式C | spec 声明目标 runtime 模式及待验证项；验收时只把已验证模式标为支持 | 避免开发前要求已有新算法运行证据的循环依赖 |
| R-DEC-008 | MUST | 自动:check_spec + 人工核对 | 模式C | spec 第 8 章必须声明 Cost model 策略：自动选路或经过成本候选过滤的显式选路均须实现；仅源码证明绕过成本候选的路径可填 `不需要` 并注明依据 | 当前新 selector 在 HCCL_ALGO 过滤前跳过空成本结果；默认不变须显式正向匹配目标算子/executor/算法配置后才返回候选；无配置、不匹配配置必须排除。不能以无条件巨额成本、接口推导系数或少量默认用例通过替代；策略字段由脚本检查，具体链路需人工核对，见 `executor-selector.md` §4 |
| R-DEC-009 | MUST | 自动:check_template | 新增+修改 | `CalcCostCoeff` 里禁止未经测量的硬编码时间常数，一律走 `CostModelManager` 的既有接口（`CalcLatencyParams` / `CalcLaunchParams` / `CalcMeshParam` / `CalcNHRParams`） | `0.000005f`、`1e-6f` 这类常数换台机器就不成立。确属单位换算的，给 `UPPER_SNAKE` 常量名并注明来历 |
| R-DEC-010 | MUST | 自动:check_spec | 模式C | spec 第 2 章必须给全**支持矩阵**：selector（老格式 / 新 DSL / 两者）· 平台 · workflow · inplace · rankSize(1/2/N) · dataSize | 只覆盖一条选择链路，表面上算法注册完成了，实际选不中。逆向存量 spec 里 template 确实不约束的维度，写「template 侧不约束」也算填了 |
| R-DEC-011 | MUST | 自动:check_spec | 模式C | spec 第 3 章必须固定多 channel 切分公式：channel 数来源与上下限、第 `c` 个 channel 的 offset/len、取本 rank 还是前驱/后继的 channel 集合、非整除与空切片策略、各 rank 编号是否一致 | 不固定就会出现任务数、同步开销、cost model 三者对不上 |
| R-DEC-012 | MUST | 人工核对 | 全量 | 优先复用符合父层约束和已确认 spec 的仓内实现；蓝本不符合需求时调整实现 | 不能让存量实现覆盖已确认需求 |
| R-DEC-013 | BLOCKER | 人工核对 | 全量 | 调试期任何影响**通信方向 / buffer / 选路 / 注册属性 / Cost Model / 支持矩阵**的改动，都要**先回写 spec 再改代码**，并重跑 `check_spec.py` | 出现过 spec 声明「不做 Cost Model」而最终代码里加了的情况——spec 不再是真相，下一个人照 spec 读代码必被误导 |
| R-DEC-014 | MUST | 人工核对 | 模式C | 一个任务只能有一个 active spec，spec 要有唯一标识与目标算法注册名；被取代的明确标 `superseded` | 留下多份名称不同、决策不同但都合法的 spec，等于没有规格 |
| R-DEC-015 | MUST | 自动:check_template + 人工核对 | 全量 | 上游约束为只显式启用时使用 `check_template.py --explicit-only`；确定无条件非空成本候选报 ERROR，复杂控制流或函数委派仍须人工核对正向配置守卫 | WARN 不是守卫通过；无配置、不匹配配置均不得启用，定向匹配才允许返回非空候选 |

## 12. 维护这份表

- **加规则**：选好域和等级，序号顺延（**不复用已删规则的号**）。若可自动检查，同时改
  `check_template.py` 或 `check_spec.py` 的 `RULES` 字典并在报错里带上 ID。
- **删规则**：从表里删掉，脚本里同步删。**ID 不回收**，避免旧评审意见指向新规则。
- **改「验证方式」栏**：取值只能是 `自动:<脚本>` / `人工核对` / `外部环境` / `上游输入`
  （可加 ` + clang-format` 后缀）。仓库维护测试会校验这份词表，避免错别字产生未定义类别。
- `上游输入` 只表示能力无法独立闭合的信息缺口；谁负责补齐、是否与用户交互，由 Agent/Plugin 决定。
- **改等级或适用范围**：表和脚本 `RULES` 字典必须一起改——`check_template.py --rule <ID>` 打印的就是脚本里那份，两边不一致会立刻被人发现。
- **每次改完**先跑仓库维护测试，再按目标环境运行 `bash scripts/selftest.sh "$HCCL_REPO"`。
- 本表引用真仓 API 与规范的地方（§3 的基类、§7/§8 的编码规范）**会随真仓演进失效**，
  每次在新仓上工作时顺手核对一遍。
