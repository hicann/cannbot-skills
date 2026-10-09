# 构建与双轨验证技术契约

> **真源归属**：`hccl-aicpu-best-practice`（跨层共享）。事实来源：`${HCCL_ROOT}/build.sh` usage、`${HCCL_ROOT}/AGENTS.md` 第 5 节、`docs/zh/build/build.md` 和目标版本真实源码。
>
> 本文只定义构建、产物、基线和 Checker 证据何时有效。Agent 权限、用户交互、调用顺序、失败回退、任务状态和完成/提交决定由上层 Plugin 与角色契约负责。

## 1. 环境与唯一完整构建入口

```bash
source <CANN路径>/set_env.sh
cd "${HCCL_ROOT}"
bash build.sh --pkg --full -j16
```

首次候选完整构建使用单一 `--pkg --full` 入口，同时覆盖 host 与 device。不要依次运行 `--aicpu`、`--pkg`、`--pkg --full`；当前 `build.sh` 会清理构建/输出目录，重复入口会销毁可复用状态。源码未变且构建树、CANN 来源和配置一致时，可以按真实 CMake target 增量构建 host/device 并重新打包；无法证明打包依赖完整时仍使用单次全量入口。

AICPU 功能验收还要求 §5 的 hccl-vm A/B 双轨通过。HCCL 仓内 ST/UT、单文件编译、静态检查和 Skill 自测均不能替代完整构建或双轨 Checker；若另有范围要求，其结果只能单独记录。

## 2. 基线、源码与产物证据

### 2.1 证据不变量

1. **基线早于候选构建**：首次候选构建前保全可信旧包，并形成与回归同矩阵的结构化基线。缺可信旧包时，可在独立干净工作树重建一次基线；候选包不能回填为旧基线。
2. **源码身份完整**：记录 commit、已跟踪差异、未跟踪源码及其 Unix 权限。未跟踪新增文件也是源码身份的一部分；工具链或构建选项变化时不能沿用旧构建身份。
3. **构建即归档**：把 `.run` 包、参考库、源码快照、完整日志和真实退出码保存到 `build_out` 与构建清理目录之外；baseline/candidate 使用不同目录。记录包 SHA-256、构建命令、CANN 来源和环境身份。
4. **原始状态不可回填**：通过管道保存日志时必须保留构建进程退出码，不能把 `tee` 的成功当成构建成功。不能为补取日志、退出码或时间戳重复构建。
5. **用例证据可比较**：逐用例记录 case-id、算子、dtype、size、通信域、selector/算法覆盖配置、包身份、全部算法名及调用次数、Checker 状态和独立原始日志。只记录 PASS/FAIL 或缺少算法名都不满足比较条件。
6. **切包不等于重编**：缺字段时先恢复原始日志；确需补测则安装已归档且哈希已验证的包。只有可信产物不存在或源码/构建输入变化时才重建。

候选构建输入完整性与候选安装测试就绪性是两个条件：完整但存在未解释 FAIL 的旧基线可以满足构建输入完整性，但不满足安装测试就绪性。安装测试前须按 [定点补测](../../hccl-test-tool-hvm/references/recovery.md) 保留或补齐证据，并由 `repair.py ready` 给出就绪结果；不得改写旧 FAIL。

### 2.2 基线与归档工具

工具入口见 [基线证据与回归比较](../../hccl-test-tool-hvm/references/evidence.md)：

- `scripts/archive_package.py`：归档包、构建日志、退出码、源码快照和环境身份；不触发构建或安装。
- `hccl-test-tool-hvm/run.sh --artifact <artifact.json> --role baseline --phase <标签>`：保存基线逐用例证据；回归使用 `--role regression`。`--phase` 只是标签，不能改变证据角色。
- `hccl-test-tool-hvm/evidence.py check <运行目录> --require-role baseline`：检查基线结构；追加 `--require-pass` 才要求全通过。
- `hccl-test-tool-hvm/evidence.py compare <基线目录> <回归目录>`：在默认选路不变模式下比较用例、选路与调用次数。

首次构建就保存真实退出码：

```bash
if bash build.sh --pkg --full > "$build_log" 2>&1; then
    build_rc=0
else
    build_rc=$?
fi
printf '%s\n' "$build_rc" > "$build_rc_file"
```

### 2.3 预检与候选构建入口

`workflow.py preflight` 记录源码、环境和相关入口指纹；`--previous` 只比较调查输入是否变化，不证明环境完整或算法正确。

```bash
python3 <本skill路径>/scripts/workflow.py preflight \
  --repo <HCCL源码根> --op reducescatter --output <新预检.json> \
  --previous <旧预检.json> --task-dir <任务持久目录> --environment-file <集群配置文件>

python3 <本skill路径>/scripts/workflow.py build \
  --repo <HCCL源码根> --artifact <旧包归档>/artifact.json \
  --baseline <旧selector基线目录> --baseline <新selector无配置基线目录> \
  --require-new-selector --cann <CANN目录或set_env.sh> \
  --output <源码仓外的新候选输出目录> --jobs 16 --check-only
```

检查通过后使用相同参数去掉 `--check-only`。正式执行仍会重新校验输入，并只运行一次 `bash build.sh --pkg --full -j16`。新增显式选路算法默认需要旧 selector 与新 selector 无配置路径两组基线。候选输出必须位于整个 HCCL 源码仓之外；会被构建清理的额外目录用 `--clean-root` 声明。该入口完成构建后，不再手工重复相同全量命令。

`build --check-only` 和正式构建都会解析 CANN 来源，在子环境加载 `set_env.sh` 并检查 host/device 工具。优先显式传入 `--cann`；也支持 `ASCEND_HOME_PATH`、`ASCEND_OPP_PATH` 和当前 `build.sh` 的默认路径。实际构建用 `-p` 固定同一 CANN 来源。

可选单文件冒烟：

```bash
python3 <本skill路径>/scripts/workflow.py build ... \
  --compile-smoke <仓内新cc路径> --compile-database <既有compile_commands.json>
```

缺少唯一编译条目、缓存或 CANN 来源不匹配时，该项记为 SKIP，并保留完整构建要求。独立 `compile_smoke.py --repo ... --target ... --cann ... --database ... --reference <同目录蓝本cc>` 只有在 include 和编译定义已核实时才有意义；缓存来源相同也不能证明新 CMake 接线已进入完整构建。

### 2.4 可恢复源码快照与安装身份

`workflow.py build` 在构建前保存 `source-snapshot/`：基准 commit、已跟踪文件二进制 diff、已变更跟踪文件与未跟踪源码的完整 Unix 权限，以及未跟踪源码内容。成功构建后使用 `archive_package.py --source-snapshot <本次输出/source-snapshot>` 随包归档。手工构建时先执行：

```bash
python3 <本skill路径>/scripts/source_snapshot.py capture \
  --repo <源码仓> --output <新快照目录>

python3 <本skill路径>/scripts/source_snapshot.py restore <归档/source-snapshot> \
  --repository <仍包含基准commit的仓> --output <不存在的恢复目录>
```

恢复只写入新目录，并依赖基准 commit 仍可访问；忽略文件不属于源码快照。旧包没有快照时仍可作为已验证二进制用于基线，但不能据此宣称源码可恢复。

安装参数必须来自当前 `.run` 包的帮助，或同包类型、同版本的已验证记录；不能把某次 `--quiet --full` 等组合推广到所有安装器。安装证据记录包哈希、命令、退出码、安装路径，并在部署后核对 host/device 库哈希。

### 2.5 首次构建前的接入审查

逐个注册 executor 核对实际资源消费者。当前 AllGather Parallel 的 `PrepareResForTemplate` 直接读取两个模板的 `GetRes`，据 `slaveThreadNum + 1` 切分线程，并消费 `notifyNumOnMainThread`；只实现 `CalcRes` 会在运行期产生错误资源切分，编译不能拦截。模板须实现有效 `GetRes` 并与 `CalcRes` 的线程/通知结果一致。`GetThreadNum` 是否必须覆盖取决于 executor/template 调用点及基类默认语义，不能统一以“多 jetty 才需要”判断。细节见 [资源模型](../../hccl-aicpu-best-practice/references/03-resource-model.md)。

在同一份 Spec/交接记录关闭以下接入项，不另外创建重复报告：

| 接入项 | 首次构建前的证据 |
|---|---|
| 布局 | 每个绑定 executor 的调用阶段与 repeat/stride/base offset 覆盖；适用卡片及差异 |
| 资源 | CalcRes/GetRes 与实际消费者一致，含新构造对象和空 channel 分支 |
| 成本候选 | 依赖的 algName/其他字段逐层传入，显式启用守卫使用实际配置来源 |
| 工程接线 | 目标算法枚举与映射、注册全名/DSL、host/device CMake；复用 executor-selector 清单 |
| 已发现的问题 | 修复位置，或不影响已声明路径的证据；不能把“已知待验证”留到完整矩阵 |

按模板实际消费的 `CalcCostCoeffParam` 字段，在每个绑定 executor 检查直接聚合初始化。
旧命令默认只要求 `algName=algName`：

```bash
python3 <本skill路径>/scripts/check_cost_forwarding.py \
  --executor <目标executor.cc> --cost-header <目标cost_model.h> --expected-calls <契约中的调用数>
```

需要读取通信域配置时用 `--require-field algName=algName --require-field comm=comm`；
消费拓扑时再加 `--require-field topoInfo=topoInfo`。显式列表只检查列出的字段，不自动添加 algName。
别名经核对后可写在等号右侧；旧的 `--name-expression` 与 `--require-field` 互斥。
退出码 0/PASS 只证明初始化点使用指定表达式；1/FAIL 表示省略或置空；2/UNVERIFIED 表示语法、
字段或调用数量需要定向核对。该工具不证明实参有效非空、lambda 捕获、控制流可达性或完整选路；
编译与定向命中断言仍保留。AllGather 的 1/4 处调用及各字段缺口见[成本参数矩阵](allgather-integration.md#4-成本参数传递矩阵)。

## 3. 常见编译错误对照

| 错误 | 原因 | 处理 |
|---|---|---|
| `undeclared identifier` | 缺 include | 检查 include 路径；跨仓接口确认 dlsym 层已声明 |
| `undefined reference` | `.cc` 未加入 CMakeLists | 检查对应目录 `CMakeLists.txt` 的 `src_list` |
| selector 选不中新算法（运行时 `NOT_MATCH`） | `selectAlgName` 字符串与 REGISTER name 不同 | 两处逐字符比对 |
| `static_assert base class` | template 类未继承 `InsAlgTemplateBase` | 检查继承与 override 签名 |
| `AICPU_COMPILE` 相关错误 | device 侧编译条件不完整 | 检查 host 专用路径的 `#ifndef AICPU_COMPILE` 守卫 |
| dlsym 接口运行时报 `not supported` | HCOMM 接口不支持 | 调用前用对应 `HcommIsSupportXxx()` 查询，false 走替代路径 |
| 编译告警 | `-Werror` | 修复告警，不规避 |
| OAT 检查失败 | 新文件缺许可头 | 采用同目录 CANN-2.0 许可头 |

## 4. 编码与静态合规

- 风格以目标仓 `.clang-format` 为准：当前为 120 列、4 空格、指针右对齐、K&R 大括号；代码习惯跟随同目录实现。
- C++ 标准以目标 CMake 为准，当前核对为 C++17。
- pre-commit 使用 clang-format v18.1.8 与 OAT 合规检查；新增源文件需要 CANN-2.0 许可头。
- 静态合规结果只证明对应规则已检查，不替代完整构建、独立检视或 Checker。

## 5. hccl-vm 双轨功能验收契约

功能验证使用 `communication/hccl-test-tool-hvm/` 的真实接口；AICPU 引擎为 `AI_CPU`。轨道 A 证明候选算法被命中且功能正确，轨道 B 证明默认路径没有超出既定范围的变化。

| 项 | 技术要求 |
|---|---|
| 基线 | 首次候选构建前，用可信旧包在不设 `HCCL_ALGO` 的默认路径运行与回归相同矩阵，记录 Checker、算法名和调用次数 |
| 环境 | 核对二进制、CANN、设备侧符号、Checker/集群配置和 rootinfo；证据身份变化后重验 |
| 引擎 | `AI_CPU`；需要 QEMU，单用例超时至少 180 秒 |
| 轨道 A | `--env 'HCCL_ALGO=<已核验配置>' --expect-algo <注册全名>`；覆盖冻结矩阵，Checker PASS、失败数 0 且命中断言通过 |
| 轨道 B | 不设置 `HCCL_ALGO`；与基线同矩阵，Checker PASS，并满足 §5.1 的选路契约 |
| 通信域 | 目标 rank 2→112、4→114、8→118、16→128；跨 server 使用 121/122/124/144。112 只作路由冒烟时不能替代目标拓扑 |

轨道 B 的最低回归范围随改动层级扩大：

| 改动层级 | 最低回归范围 |
|---|---|
| 新增算法（template、executor 注册、selector 新分支） | 该算子默认路径各典型档位，与基线同覆盖 |
| 修改 selector 阈值或既有分支 | 该算子全部原有算法档位，以及受影响相邻算子默认路径 |
| 修改 topoMatch 或 `op_common` 公共代码 | 全集合通信算子默认路径冒烟 |

### 5.1 选路验收契约

输入必须明确默认选择是 `不变` 还是 `按上游批准范围改变`：

| 模式 | 判定 |
|---|---|
| 不变 | 默认路径各用例的算法及调用次数与旧基线一致，使用 `hccl-test-tool-hvm/evidence.py compare` |
| 按批准范围改变 | 在观察候选结果前固定变更依据、允许范围和逐用例旧/新算法矩阵；范围内符合新期望，范围外沿用旧基线 |

变更范围包括算子、selector、拓扑、dtype、档位与边界；矩阵包括 case-id、完整测试配置、旧算法/调用次数和新期望算法/调用次数。性能数据不能替代变更依据，也不能豁免 Checker FAIL。批准来源是上游输入；本 Skill 只检查证据字段和技术一致性，不产生授权。

现有 `hccl-test-tool-hvm/evidence.py compare` 只支持“不变”模式。批准变化时：

1. 分别运行 `evidence.py check <基线目录> --require-role baseline --require-pass` 和 `evidence.py check <候选目录> --require-role regression --require-pass`。
2. 确认两批用例集合、构建/测试环境、集群、iterations、warmup、runner、用例环境和测试二进制身份一致；默认路径均无 `HCCL_ALGO` 或通信域算法覆盖。
3. 逐项核对范围内的 `observed_algorithms`、`algorithm_counts` 与新矩阵一致，范围外与基线一致，并覆盖阈值两侧和边界点。
4. 保存人工矩阵核对结果，不修改原始证据，也不把 `compare` 的失败重标为通过。

### 5.2 覆盖与命令示例

多批次默认保存为 [hccl-vm 测试计划](../../hccl-test-tool-hvm/references/test-plan.md)，先 `plan.py check`，
再按 baseline/directed/regression 角色分别执行。候选每个 executor 的首个定向冒烟保持单用例，
将新增或修改的接入链放在前面；所有目标 executor 命中且 Checker 通过后才展开完整覆盖。
批次失败后保留现场并定向诊断，后续未执行矩阵保持 pending；不在共享 VM 上并发运行多个计划。

冻结矩阵需要明确覆盖目的、executor、拓扑、布局参数、配置、期望算法和日志断言。每个目标 executor 至少覆盖最小适用拓扑。多 loop 根据 executor 的真实 `maxCountPerLoop` 公式计算，包含 dtype、scratch 倍数、对齐和分层预算，并以日志证明 `loopTimes > 1`；template 的 `repeatNum > 1` 是另一条路径，需要单独覆盖。已验证的 AllGather Ring `4M + HCCL_BUFFSIZE=1` 只能作为该场景证据，不能推广为其它算法阈值。

使用 `plan.py loop-size` 把已核验阈值换算成刚超过一轮的合法字节数；测试程序 count 换算、
缓冲区合法范围和公式依据一并写入计划断言。避免先试 64M、再试 128M 等逐档探索。
源码/包发生变化后旧候选证据不能自动转给新版本；定点补测按 hccl-vm 的身份约束执行。

```bash
export HWLOC_COMPONENTS=-gl,-opencl

# 轨道 A：强制候选算法并断言命中
bash run.sh -m AI_CPU -o allreduce -d int32 -s 64-4M -t 118 \
  --env 'HCCL_ALGO=<已核验配置>' --expect-algo <注册全名> --timeout 180

# 轨道 B：默认路径，与基线同覆盖
bash run.sh -m AI_CPU -o allreduce -d int32 -s 64-4M -t 118 --timeout 180
```

配置值与注册全名不是同一概念；使用前核对 [HCCL_ALGO 显式选路](executor-selector.md#4-hccl_algo-显式选路保留默认选择) 的目标版本、开关与候选条件。新 selector 定向测试还需要 `--env HCCL_USE_NEW_SELECTOR=1`；回归与安装前基线覆盖新 selector 无配置路径，并清除通信域算法覆盖。

### 5.3 证据判定

- A/B 任一轨 FAIL、算法未命中、用例集合不一致、身份不匹配或存在未解释的证据缺口，功能验收都不能记为 PASS。
- 相同条件的重复运行不增加区分证据；若保留一次偶发性复跑，原失败和复跑结果必须同时保存。
- 基线存在同类 FAIL 只能支持环境归因假设，不能豁免候选失败。环境或测试程序修复后，旧环境证据失效，使用同一候选版本与配置重跑受影响的两轨用例。
- QEMU、安装或 Checker 环境不可用时，技术结果标记为 `UNVERIFIED` 并列出证据与未验证范围，不能改记为 PASS；上层是否进入 `BLOCKED` 不在本文裁决。
- hccl-vm 只验证功能正确性，不证明性能收益。

功能验收的客观结果只有在轨道 A、轨道 B、算法命中断言和选路契约全部通过时才是 PASS。该结果是上层状态判定的一项输入，不等同于任务完成或可提交结论。

`hccl-test-tool-hvm` 的冒烟关卡、参数、`run.sh`、通信域配置与共享内存清理以其 SKILL.md 为真源。执行授权、共享环境 owner、重试/修复流转、sudo 交互、最终状态和提交决策以 `hccl-op-dev` Plugin 与 Agent 契约为准。


## 交付文件与 Spec 身份

源码快照按设计排除 ignored 文件。Spec、评审报告与日志是否存在于磁盘、是否被快照保全、
是否将随代码提交，是不同问题；不能只从 `git status` 推断交付完整。
交接和最终交付时显式列出计划交付文件，Spec 即使被忽略也记录内容哈希。

清单示例（路径相对 HCCL 仓根；`artifact` 也可使用任务目录的绝对路径）：

```json
{"files": [
  {"path": "src/ops/broadcast/algorithm/template/aicpu/ins_temp_broadcast_ring.cc", "role": "source", "delivery": "git"},
  {"path": "docs/zh/architecture/broadcast-ring-dataflow.md", "role": "spec", "delivery": "git"},
  {"path": "/task/archive/review.md", "role": "review", "delivery": "artifact"}
]}
```

按实际任务填写 source/spec/test/report，不机械复制上述路径；待生成报告可在最终交付时加入。

```bash
python3 <本skill路径>/scripts/check_deliverables.py --repo <HCCL仓> \
  --manifest <交付清单.json> --output <新的交付核对.json>
```

退出码：0 表示清单所列文件满足交付方式；1 表示缺失或计划 Git 交付但未跟踪；2 表示输入/执行错误。
工具记录文件存在性、Git 跟踪/忽略状态与 SHA-256，只写新的输出文件，不自动暂存或修改忽略规则。
开发中未跟踪文件会如实提示，可继续开发；最终 Git 交付前由作者显式纳入版本控制。
`artifact` 必须指向实际保存的附件；工具只核对和哈希，不代替复制归档，也不判断清单是否漏列文件。

Review 同时绑定源码身份和 Spec 哈希。仅 Spec 变更时重做契约核对；若变更意味着执行或验证范围改变，
按影响范围重新验证。源码变更继续遵循既有证据一致性规则，不以交付清单替代候选源码快照。
