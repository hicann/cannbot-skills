# Subagent 调用参数详情

源码位置与产物目录遵循 [Ascend950 路径约定](../../../references/source-layout.md)。本阶段沿用并向后续技能/角色传递 `{ops_tilelang_repo}=OPS_TILELANG_DIR` 和 `{tilelang_repo}=TILELANG_DIR`；两者由插件源码准备阶段确定，不由用户另行指定。


本文档是 tilelang-tuning 执行各阶段和调用 Subagent 的**唯一执行手册**。Step 0 的 profiling 入口生成、Step 0.5、2b、2c、2.5 由主 agent 执行；Step 1 和 2a 按注册角色名调度，并将对应消息模板完整传递。角色未注册或调度失败时停止当前阶段并报告，不由主 agent 替代执行。

全流程使用入口选定的 `{backend}` 与 `{tilelang_target}`：`pto → pto`，`ascendc → ascend`。所有编译、pytest 和 profiling 命令必须显式设置 `TILELANG_DEFAULT_TARGET={tilelang_target}`；禁止默认、回退或混用后端。

---

## Step 0 附加步骤：生成独立 profiling 算子文件（必做）

> 目的：生成一次进程即可运行用户提供的全部 case 的独立 profiling 文件。

### 触发条件

`{cases_csv}` 由用户直接提供并通过 1～20 个 case 门禁；`{operator_file}` 是本轮基线中包含目标 kernel 定义的 `.py` 文件。第 1 轮在 Step 0 完成时生成；后续每轮开始前，从该轮最新基线重新生成。

### 生成规则

1. 将 `{operator_file}` 的**完整原始字节内容原样复制**为 `{output_dir}/round{N}/profiling_entry/baseline_<operator_name>_profiling.py` 的前缀，禁止删减、重排、格式化或改写 kernel。
2. 仅在该前缀之后追加 host 侧 profiling 入口：
   - 将 `{cases_csv}` 中用户提供的全部 1～20 个 case 按原顺序写成 `CASES`；case ID、参数名和值必须逐项一致。
   - 输入构造语义复用 `{test_file}` 中的数据生成逻辑，但最终必须显式调用当前生成文件内复制出的本地 kernel，禁止调用会重新路由到原工程或已安装包旧 kernel 的公开 wrapper。
   - host 入口接受 `--device` 及可选的 `--warm-up`、`--launch-count`，在创建 NPU tensor 前选择设备。默认 `warm-up=0`、`launch-count=1`，在同一进程中按顺序为每个 case 调用一次本地目标 kernel；普通 `msprof` fallback 按 `tilelang-op-profiling` Step 2.5 在入口内逐 case 预热并重复正式调用。禁止 `--case-id` 分支、逐 case 子进程和 host 计时。
3. Step 0 只生成文件并做静态检查，不运行 pytest、benchmark 或 kernel。
4. 静态检查 `{operator_file}`、wrapper 和 `{test_file}` 的 target/dispatch 可进入 `{tilelang_target}`；发现硬编码冲突时停止，禁止改写源码或切换后端。
5. 记录后端、`{operator_file}` SHA256、文件字节长度、`{cases_csv}` SHA256、生成文件路径和预期 kernel 名。硬门禁：生成文件前 `source_length` 个字节必须与 `{operator_file}` 字节完全一致；内嵌 case ID、参数值和顺序必须与 `{cases_csv}` 完全一致。

### 最高优先级防陈旧门禁

> ⚠️ **严禁复用旧 profiling 文件测量新源码。** 每轮 Step 1 必须从该轮当前基线源码重新生成；Step 2b 必须分别从本轮 baseline 和每个通过编译及精度门禁的 `optimized_<方案>` 最新目标算子源码重新生成各自独立 profiling 文件。禁止复制、改名、补丁修改或继续使用 Step 0/前一轮/其他方案的 profiling 文件。任一源码前缀 SHA、cases.csv SHA、源码相对路径或运行时 kernel 名不匹配，立即停止采集，重新从对应最新源码生成。

---

## Step 0.5：优化前精度基线

> 目的：在任何分析/优化动作前，把**未优化基线**的精度钉死为回归基准。

### 触发条件
Step 0 已校验用户提供的 `{cases_csv}`；`{code_dir}` 为源码工程根目录，`{test_file}` 为该工程内统一 pytest 正确性测试文件。由主 agent 亲自执行。

```text
请为下述算子建立【优化前精度基线】：

- 基线工程目录：{code_dir}
- 精度测试文件：{test_file}
- 输出目录：{output_dir}
- 后端：{backend}（TILELANG_DEFAULT_TARGET={tilelang_target}）

【任务】
1. 运行 pytest 前完成 NPU 设备绑定预检：
   - 执行 `npu-smi info -l`，按设备块解析完整匹配 `^\s*NPU ID\s*:\s*(\d+)\s*$` 的 ID 与对应 `Product Name`；排除产品名为空或 `NA` 的条目，禁止按非空输出总行数推算设备数。
   - 若父进程已设置 `ASCEND_RT_VISIBLE_DEVICES`，其合法非负整数 ID 列表是候选上限，禁止加入列表外设备；否则以 `npu-smi` 的合法条目为候选。候选跨型号时选择数量最多的同型号组，并列时选择最小 ID 所在组。
   - 要求选中 ID 非空、无重复且均为非负整数。仅启动**一个**独立预检进程，令 `ASCEND_RT_VISIBLE_DEVICES=<selected_ids>`，导入 `torch`/`torch_npu`，要求 `torch.npu.device_count()` 等于选中数量，并在每个逻辑设备上完成 1 元素 NPU tensor 分配与 `torch.npu.synchronize()`。任一设备失败时停止，不逐 ID 试错，也不得启动 pytest。
   - 在报告中记录候选来源、`npu-smi` 的 ID/产品名、排除项、最终选中 ID、批量运行时验证结果和最终 worker 数。后续 collection、pytest 和本算子各优化方案精度回归均使用相同的命令级 `ASCEND_RT_VISIBLE_DEVICES=<selected_ids>`。
2. 使用目标后端执行 `TILELANG_DEFAULT_TARGET={tilelang_target} python -m pytest {test_file} --collect-only -q`，记录完整正确性 nodeid 清单。仅在当前测试实际定义了 benchmark marker 时排除性能用例，并记录排除依据；不得按未确认的 marker 或测试级别过滤用例。
3. 执行清单中的全部正确性用例，最终验收不得使用 `-x`。使用 `TILELANG_DEFAULT_TARGET={tilelang_target} python -m pytest {test_file}`；仅在已确认 xdist 与 worker 设备绑定、隔离机制时增加 `-n <workers>`，并发不超过已验证可用设备数，否则串行。需要排除性能用例时显式传入收集得到的正确性 nodeid。OOM 时降低并发，不减少 case；collection 为空时停止并报告，不能视为通过。
4. 精度标准以 `{test_file}` 中的 `assert_close`、`assert_equal`、`calc_diff` 等实际判断代码为准，禁止修改或放大阈值。
5. 计算基线 `{test_file}` 的 SHA256；后续各阶段须使用字节一致的 pytest 文件，以此保证前后精度判断标准一致。
6. 产出 `{output_dir}/precision-baseline.md`：按通用报告格式记录后端、pytest 文件路径与 SHA256、设备绑定预检、完整 collection 清单、测试配置（使用 TK 生成器时含 `OPS_TILELANG_TEST_LEVEL`）、实际 pytest 命令和精度判断标准，以及逐 case PASS/FAIL、dtype、shape。
7. 门禁：设备预检与 collection 覆盖校验通过，且清单中的全部正确性 case 均 PASS，基线才算通过。任一门禁失败时在报告置 `状态: ❌失败` 并停止。

【返回】
- `{output_dir}/precision-baseline.md` 路径
- 基线是否全绿（是→可进 Step 1；否→列出不达标 case）
```

主 agent 确认 `precision-baseline.md` 全绿后才进 Step 1。基线不达标则停止并向用户报告。

---

## Step 1：性能数据采集与分析

### Subagent 调用约定

主 agent 使用当前宿主提供的原生 subagent 调度能力，请求逻辑名称为 `tilelang-perf-analysis-expert` 的 subagent，并将下述消息完整传递；只替换占位符，不改写正文。

```text
请对以下待调优的算子代码进行性能数据采集与分析，按 **运行算子采集数据 → 性能分析 → 输出调优方案** 三阶段顺序执行：

- 源码工程根目录：{code_dir}（当前轮 baseline 的完整工程根目录）
- 目标算子文件：{operator_file}（本轮基线中包含目标 kernel 定义的源码文件）
- 独立 profiling 算子文件：{profiling_file}（主 agent 从本轮最新 `{operator_file}` 和 `{cases_csv}` 生成，只用于本轮基线采集）
- 测试用例文件：{cases_csv}（由用户直接提供，已校验为 1～20 个完整 case；禁止从 pytest 推导、补充或筛选）
- 输出目录：{output_dir}（算子级隔离目录，性能数据、报告等所有产出物落盘到此目录下。本轮产出物落到 `{output_dir}/round{N}/` 子目录下，N 为当前轮次编号）
- 后端：{backend}（TILELANG_DEFAULT_TARGET={tilelang_target}；所有 `tilelang-op-profiling` 命令使用此值）

进入采集阶段时，先加载名为 `tilelang-op-profiling` 的 Skill，并传入平台 `Ascend950`。进入分析阶段时按下文分别加载所需 Skill；任一 Skill 加载失败即停止并报告。

---

### 阶段一：运行算子并采集性能数据

1. **先验证 profiling 文件与本轮最新源码一致，再编译运行**：
   - 验证 `{profiling_file}` 的源码前缀与 `{operator_file}` 逐字节一致，记录两者 SHA256；验证内嵌 CASES 与 `{cases_csv}` 的 ID、参数和顺序一致
   - 任一校验不一致时立即停止，返回主 agent 从本轮最新 `{operator_file}` 重新生成；禁止自行沿用、修补或复制旧 profiling 文件
   - 按 `tilelang-op-profiling` Step 1 直接运行 `{profiling_file}`，确保全部 case 可在同一进程中执行
   - 编译后**校验 msprof 抓到的 kernel mangled 名与 `{operator_file}` 中的 kernel 函数定义一一对应**，确保采集到的是当前 profiling 文件中的目标 kernel，而非原工程、已安装包或缓存中的旧 kernel
2. 按 `tilelang-op-profiling` Step 2 启动一次 msprof，`--launch-count` 等于 case 数
   - **禁止只选代表性用例**：每个 case 都是用户关注的，必须全部采集
   - 禁止逐 case 重启 Python 或 msprof；数字 launch 目录按顺序与 CASES 一一对应
   - 若 `tilelang-op-profiling` 走 fallback，仍须运行该 profiling 文件中的全部 CASES，禁止替换为其他 probe
   - 所有测试用例均须在 NPU 上运行（禁止 host 侧短路绕过 kernel）
   - **禁止使用 host 侧计时（std::chrono / gettimeofday 等）替代 msprof 采集**
   - **禁止通过第三方评测框架（如 cann_bench eval）间接采集基线性能**：基线性能采集必须由 `tilelang-op-profiling` 直接对 `{profiling_file}` 中从本轮最新 `{operator_file}` 复制出的本地 kernel 采集，不引入第三方框架的 host 侧 wrapper 开销作为未知变量
3. 从 profiling 数据中获取每个 case 的 aic/aiv 耗时，并确定统一 `kernel_time`：AIV-only 使用 aiv_time，AIC-only 使用 aic_time，混合 AIC/AIV 使用 `tilelang-op-profiling` 定义的关键路径 kernel 时间。同一 case 的基线与后续方案必须保持相同口径

【输出：阶段一】
- **全部 case** 的性能数据采集产物（profiling 目录），存放到 `{output_dir}/round{N}/perf_per_case/` 下
- 每个 case 的 kernel_time 汇总表，并记录原始 aic/aiv 字段和计时口径
- **kernel 名校验记录**：列出 msprof 抓到的 kernel mangled 名与本轮最新 `{operator_file}` 中 kernel 函数定义的对应关系
- **后端校验记录**：记录 `{backend}`、`TILELANG_DEFAULT_TARGET={tilelang_target}` 和实际编译 target
- **profiling 来源校验记录**：记录 `{operator_file}`、`{profiling_file}`、两者源码前缀 SHA256、`{cases_csv}` SHA256 和一致性结论
- 采集失败时停止，不进入阶段二

---

### 阶段二：性能分析

1. 先加载名为 `tilelang-perf-optimization` 的 Skill，并传入平台 `Ascend950`。PTO 后端使用该 Skill 分析；AscendC 后端仅复用其中与 `target=ascend` 有直接证据的规则。基于源码、所选后端 lowering 和 profiling 形成策略方向
   - skill 中的优化点和模板是候选来源与示例，不是封闭方案集。须以当前源码、全部 case、资源模型和 profiling 证据为主线，允许迁移、组合、扩展相似原理或提出 skill 未收录的候选；即使已命中模板，也不得忽略与实测瓶颈直接相关的非模板候选。新候选仍须满足 `tilelang-perf-optimization` 的可证伪假设和证据门禁
2. **模板匹配**：先加载名为 `tilelang-performance-best-practices` 的 Skill，并传入平台 `Ascend950`，只采用已验证兼容所选后端的模板
   - **先定算子族再匹配**：根据本轮最新目标算子源码（计算结构、关键 API）和 cases.csv（shape/dtype/attr 特征）判定算子所属的算子族（如 MatMul / Reduction / Elementwise / Broadcast / Conversion / Scalar 等），再到 best-practices 中定位对应族的经验与模板
   - **算子族按 case 级计算模式判定，不按算子名锁单族**：同一算子的不同 case 可能跨多个族（含广播轴→Broadcast 族、纯逐元素→Elementwise 族等），须对每种模式分别映射到对应族、分别检索模板
   - **按路由读取必要资料**：先读取 `references/index.md`、`references/template_status.md` 和匹配算子族的 guide/决策树，再只读取当前 case 命中分支引用的 `.md` 与 `.py` 文件；case 跨多个计算模式时分别走对应算子族路由。禁止递归加载无关算子族或当前分支未引用的全部文件
   - 查到 TileLang `.py` 模板 → 标注完整路径并从 `references/template_status.md` 读取成熟度、已验证范围和性能证据：`PRODUCTION_REFERENCE` 或当前版本 `VERIFIED` 且优化结构已实现的模板可标为“✅可直接拷贝优化模式”；`EXECUTABLE_BASELINE`、`PARTIAL` 或缺少对应优化/性能证据的实现标为“⚠️可执行基线或部分实现”；`DESIGN_ONLY` 或没有当前版本可执行 TileLang kernel 的资料标为“❌仅设计参考”。其他编程模型的代码不作为本工作流的模板或 API 参考
   - 未查到模板代码 → 标注"无货架参考"；若当前仓库或实际 TileLang 源码另有可定位的 API、lowering、精度和适用范围证据，可依据这些直接证据独立判断方案准入，不因模板状态表未登记而自动降为 `DESIGN_ONLY`
3. **API 可用性校验**：按 `tilelang-performance-best-practices` Skill 的事实来源优先级，结合当前仓库已能编译运行的调用方式，定位 `{tilelang_repo}` 中的框架源码并核对实际导入版本，检查相关 API 定义、编译约束、`examples/ascend/` 和所选后端 lowering。不得凭记忆填写 API；无法从 API 定义、lowering、仓内可运行实现或目标版本编译结果建立直接证据的参数，禁止写入 `IMPLEMENTABLE` 方案，仅有设计依据时标记为 `DESIGN_ONLY`
4. **逐 case 分析**：对 `{cases_csv}` 中的**每一个 case** 进行性能瓶颈分析：
   - 给出该 case 的 bound 类型（VEC/MEM/SCALAR BOUND）和具体瓶颈指标
   - 将瓶颈特征相同的 case 归并到同一组，在组内统一说明
   - **每个 case 必须在报告中出现**，不允许遗漏或仅以"同上"替代
5. case 归并分组维度（按优先级）：dtype → bound 类型 → shape 规模（S/M/L）→ 特殊值特征
6. 完整读取 [性能优化搜索覆盖门禁](optimization-search-coverage/optimization_search_coverage_gate.md)，参考开始模板创建 `{output_dir}/round{N}/optimization-search-coverage.json`，记录全部 case、待裁决义务及进入方案的候选 `CREATED` 事件；执行普通校验检查记录结构和引用一致性。普通校验允许 `PENDING`，不得因优化方向尚未实验而阻止输出方案或进入 Step 2

【输出：阶段二】
- 多个性能调优策略方向 + 对应的模板匹配结果（含 B 型族逐 case 选型判定）
- 全部 case 的瓶颈分析结果（按组归并展示，但每个 case 独立列出）
- 分析失败时停止，不进入阶段三

---

### 阶段三：输出《性能调优方案》

> ⚠️ 以阶段二分析结论为输入；**禁止**重做分析或建模。

1. **输出《性能调优方案》报告**：
   - **最多 3 个已准入性能调优方案**，状态只能为 `IMPLEMENTABLE` 或 `EXPERIMENT`，按证据强度和预期效果排序
   - `DESIGN_ONLY` 单列为“未准入候选”，说明缺失的 TileLang API、所选后端 lowering 或验证证据；仅在候选依赖 bundled reference 时说明缺失的可执行模板证据。不占三个方案名额，不传给 impl expert
   - 每个方案包含：优化目标、调优后的 Tiling 参数、调优策略和事实依据。复用 bundled reference 时给出 Skill 名称及其内部参考文件的相对路径、**模板骨架**与**模板分支条件**；未复用模板时给出当前仓库或实际 TileLang 源码路径、API/lowering 直接证据并标注“无货架参考”
   - **逐 case 覆盖表**：列出哪些 case 将被该方案优化，以及每个 case 的预期改进方向和**所属模板分支**
   - 报告最后一节须包含"case 覆盖清单"：以表格列出全部 case，标注每个 case 归属的瓶颈组、适用方案、预期效果
   - **方案融合规则**：
     - 多个优化方向若涉及**不同模板且分支条件互斥**（如 AR 路径 vs ARA 路径，或 R ≤ 阈值 vs R > 阈值），**应融合为一个方案**（一个 kernel 含多个模板分支），而非拆成多个独立方案
     - 若多个方向涉及**同一模板的同一分支**（即对同一组 case 有不同优化策略），则**选最优策略拆为多方案并列**，由 Step 2 实测对比
     - 融合方案的 case 覆盖表须标注每个 case 走哪个模板分支


【输出：阶段三】
- 《性能调优方案》：方案概览 + 具体措施 + Skill 名称及参考文件相对路径 + **逐 case 覆盖清单**
- **报告必须落盘**到 `{output_dir}/round{N}/性能调优方案.md`
- 性能优化搜索覆盖记录必须落盘到 `{output_dir}/round{N}/optimization-search-coverage.json`；普通校验仅要求记录结构和引用一致
- 报告必须包含通用 `状态`、`阶段`、`摘要`、`详细内容` 字段
- 自由发挥处显式标注「知识库暂未收录」

---

【验收标准】
- 三阶段按序完成，不跳步、不倒序
- 《性能调优方案》可含多个可行方案，显式标注覆盖维度
- 各阶段阈值与判据以当次打开的 skill 文件为准
- **全部 case 均已采集性能数据，且在报告中出现**
- 禁止修改算子源码
```

---

## Step 2：方案实施

Step 2 由 **tilelang-tuning（主 agent）** 分三个阶段完成：方案实施、统一性能采集与报告、终选后全量精度回归。

### 阶段 2a：并行方案实施（主 agent → 多个 impl subagent）

主 agent 读取《性能调优方案》，为方案生成唯一安全标识 `sN_<slug>`，然后对每个方案分别请求 subagent `tilelang-perf-impl-expert`，**并行执行**。每个实例只接收一个方案以及下述完整消息：

> ⚠️ **模板优先**：若《性能调优方案》中标注了 TileLang 模板 `.py` 文件，prompt 中须将模板文件完整路径传递给 impl expert。impl expert 遇到 TileLang 编译报错时，须先分析完整报错并按下述顺序排查和修复，禁止只检查列举的常见示例。

```text
请实施以下单个性能调优方案：

- 源码工程根目录：{code_dir}
- 目标算子文件相对路径：{operator_relpath}（相对于 `{code_dir}`；主 agent 后续据此从每个优化方案的最新源码重新生成 profiling 文件）
- 方案描述：<从《性能调优方案》中提取该方案的完整内容，含模板代码路径、关键骨架结构、模板分支条件、逐 case 模板映射>
- 方案标识：<主 agent 生成的唯一安全标识 `sN_<slug>`>
- 精度测试文件：{test_file}（相对于 `{code_dir}` 的 pytest 正确性测试路径，以测试内置参考实现和精度阈值为准）
- 精度基线报告：{output_dir}/precision-baseline.md（当前方案内 pytest 文件 SHA256 必须与其一致）
- 性能用例文件：{cases_csv}（CSV 格式，仅性能采集口径用，**不作精度基准**）
- 输出目录：{output_dir}（优化代码等产出物放到此目录下。本轮产出物落到 `{output_dir}/round{N}/` 子目录下）
- 后端：{backend}（TILELANG_DEFAULT_TARGET={tilelang_target}）
- 硬件探测证据：{hardware_evidence}（同一目标设备的完整 JSON，包含 `full_soc`、`npu_arch` 及来源）
- 算子参考源码根目录：{ops_tilelang_repo}（插件内 repositories/Ascend950/ops-tilelang/）
- 框架源码根目录：{tilelang_repo}（固定为插件内 repositories/Ascend950/tilelang/，与优化副本目录分别记录）

**模板优先指令**：
- 若方案中标注了 TileLang 模板 `.py` 文件路径，将模板中的优化模式拷贝到目标 kernel 的 `.py` 文件中，仅做使其适配目标算子的必要修改
  - 适配范围通常包括 tile shape、buffer shape/dtype、SIMD 参数和 `pass_configs`，但不限于这些项目
  - 必须根据目标 kernel 的接口、数据布局、计算语义和实际编译结果识别其他必要适配，不得只检查上述示例
- 模板内 `T.simd.*` 原生指令的类型和计算结构须保留，禁止未经验证直接降级为 `T.Parallel` 自动展开
  - 应结合目标 buffer 的地址、mask、repeat/stride、dtype 和边界处理检查实际适配项
  - 上述项目是常见检查方向，不构成完整检查清单
- 实施修改前，加载名为 `tilelang-performance-best-practices` 的 Skill，并传入平台 `Ascend950`，对方案涉及的 TileLang API 结合当前仓库中已能编译运行的调用方式核对；定位 `{tilelang_repo}` 中的框架源码，并核对实际导入版本，检查 API 定义、编译约束、`examples/ascend/` 和所选后端 lowering。该 Skill 未安装或无法按名称加载时停止并报告
  - API 检查范围由本方案实际使用的 API 决定，不得只查询本文列举的 API 或示例
  - 对无法确认的 API 用法，须根据实际代码继续查询对应文档或仓库实现
- 遇到 TileLang 编译报错时，必须先读取并分析完整报错信息，确定失败阶段和直接触发位置，再按以下顺序排查并尝试相应修复：
  1. Ascend 硬件约束和算子指令约束
  2. buffer shape、dtype、作用域、生命周期和版本数
  3. TileLang API 参数、调用结构和 lowering 约束
  4. 同步、内存规划及其他相关 `pass_configs`
  5. 模板与目标 kernel 的接口、数据布局和边界处理适配

  常见问题包括 GEMM 缺少 `transpose_B=True`、L0C 使用非 float32 dtype、`T.Kernel` 误传 `threads=`、buffer 版本数不足，以及相关 `pass_configs` 配置不匹配。这些仅为常见示例，不是完整检查清单。

  每轮修复必须基于实际报错继续定位；即使上述示例均未命中，也不得直接简化优化方案。仅当已根据报错、相关 API 文档和现有可运行实现完成排查，仍无法修复时，才简化对应部分，并简要列出实际检查项、尝试过的修复和失败原因。

【任务】
按当前宿主中逻辑名称为 `tilelang-perf-impl-expert` 的 agent 定义执行流程，完成：复制目录 → 实现代码 → 编译 → 精度验证。
目录命名：`{output_dir}/round{N}/optimized_<方案标识>/`（从 `{code_dir}` 复制源码到此目录，在此目录上修改）
复制范围：按角色名 `tilelang-perf-impl-expert` 对应定义中“方案实施”的最小文件集规则执行，复制后必须能够在优化目录内运行和调试 `{test_file}`。
测试文件约束：各方案使用当前方案目录中内容相同且未经修改的 `{test_file}` 副本，不得直接运行原仓库中测试文件的绝对路径，也不得创建方案专用 pytest。若测试覆盖不足，返回主 agent 统一更新后再用于所有方案；运行前确认目标算子模块从当前优化方案目录加载。
精度回归：复用 `precision-baseline.md` 中已确认的设备、用例清单和测试配置。仅在确认设备绑定及隔离时按实际可用设备数并发，否则串行；OOM 时降低并发，不减少 case。修复迭代可使用 `TILELANG_DEFAULT_TARGET={tilelang_target} pytest {test_file} -x`，最终实施验收去掉 `-x` 并覆盖完整清单。
精度失败处理：实现缺陷须修复并重新跑完整目标测试套。只有证据确认某个小优化点受框架问题、API bug、硬件或 API 不支持影响时，才可回退该优化点并保留其余优化；禁止整体回退旧方案。最终仍有 FAIL 时返回失败状态。
完成后按通用报告字段返回：后端与实际 target、优化代码目录、编译状态、完整目标测试套精度明细、pytest SHA256、优化目录内 `{operator_relpath}` 的绝对路径与 SHA256、**模板使用情况**及所有局部回退证据。不采集性能、不生成性能报告，不生成或复用 profiling 文件。
```

主 agent 等待所有 subagent 完成。每个 impl expert 在返回前负责修复本方案的编译或精度问题；最终仍失败的方案须记录失败 case、根因和已尝试修复并排除，不进入阶段 2b。其余编译并通过完整目标测试套精度验收的方案继续独立评选；只有全部方案均失败时，本轮 Step 2 才停止。

### 阶段 2b：统一性能采集与报告（主 agent 执行）

1. **精度确认**：只接收阶段 2a 中编译成功、完整目标测试套验收通过且 pytest SHA256 与基线一致的方案；失败方案不得生成 profiling 文件或进入性能采集。若没有任何成功方案，则 Step 2 失败并停止评选
2. **从每份最新源码分别重新生成 profiling 文件（最高优先级硬门禁）**：
   - 采集前，对本轮 baseline 和阶段 2a 的每个成功 `optimized_<方案>`，分别定位其 `{operator_relpath}`，从该文件的**当前最新字节内容**重新生成独立 profiling 文件，落到 `{output_dir}/round{N}/profiling_entry/<variant>/`。
   - 每个新文件均按 Step 0 规则生成：目标源码完整字节作为原样前缀，host 入口按相同顺序运行 `{cases_csv}` 全部 case。
   - **严禁复用** Step 0、Step 1、前一轮、baseline 或其他优化方案的 profiling 文件；严禁用复制、改名或补丁修改旧文件代替从当前方案最新源码重新生成。即使文件名、kernel 名或方案目录相同，也必须重新生成。
   - 逐 variant 记录并核验：variant 源码绝对路径及 SHA256、profiling 文件路径、源码前缀 SHA256、`{cases_csv}` SHA256、生成时间。源码前缀与该 variant 最新源码不是逐字节一致，或 cases 不一致，立即停止 Step 2b，禁止采集和比较。
3. **统一 msprof 采集**：先加载名为 `tilelang-op-profiling` 的 Skill，并传入平台 `Ascend950`；baseline 和每个成功方案分别按该 Skill Step 2 一次采集全部 case
   - **禁止只选代表性用例**：必须覆盖全部 case
   - `--launch-count` 等于 case 数，禁止逐 case 重启 msprof 或改变 CASES 顺序
   - 若 `tilelang-op-profiling` 走 fallback，必须仍运行该 variant 新生成文件中的本地 kernel 和全部 CASES
   - **禁止使用 host 侧计时（std::chrono / gettimeofday 等）替代 msprof 采集**
   - 编译后**校验 msprof 抓到的 kernel mangled 名与该 profiling 文件对应的 variant 最新源码中的 kernel 函数定义一一对应**；不匹配时视为测到旧/错误 kernel，立即停止并重新生成，禁止沿用结果
   - AIV-only 使用 aiv_time，AIC-only 使用 aic_time，混合 AIC/AIV 使用 `tilelang-op-profiling` 定义的关键路径 kernel 时间；同一 case 的基线与各成功方案保持同一口径
4. **生成《性能调优报告》**并落盘到 `{output_dir}/round{N}/性能调优报告.md`：
   - 记录 `{backend}`、`TILELANG_DEFAULT_TARGET={tilelang_target}`；仅比较同后端结果
   - **逐 case 内核时间对比表**（msprof kernel_time）：行 = 全部 case，列 = 基线 + 各成功方案，并标明各 case 计时口径
   - **逐 case 分析**：对每个 case 明确写出：
     - 当前存在的性能瓶颈是什么（bound 类型 + 关键指标）
     - 采用了什么优化手段（引用具体方案）
     - 加速比提升了多少（基线 kernel_time / 优化后 kernel_time）
   - 瓶颈特征相同的 case **可以归并到同一组统一说明瓶颈和手段**，但每个 case 的加速比数据**必须独立列出**
   - 方案对照表和改进幅度
   - **最佳方案标注**：按全部 case 几何平均加速比选出提升最明显的成功方案及其代码目录。若所有成功方案几何平均加速比均 ≤ 1，则选择本轮基线，并明确记录“无 BEST_DIR/选择基线”
   - 标注采集方式与数据路径
   - 增加“profiling 文件来源门禁”表：逐 variant 列出最新源码路径/SHA256、profiling 文件路径、源码前缀 SHA256、cases.csv SHA256、运行时 kernel 名及全部一致性结论

【验收标准】
- 进入性能比较的方案均编译通过，且阶段 2a 的完整目标用例精度验收通过；失败方案已记录并排除，全部方案失败时未继续评选
- 报告含**每个 case** 的 msprof kernel_time 对比表和逐 case 分析（瓶颈 → 手段 → 加速比）
- 所有 case 均在 NPU 上运行
- baseline 与每个进入性能比较的成功方案均使用从各自最新源码即时重新生成且校验通过的 profiling 文件；不存在跨方案或跨轮次复用
- **方案一致性**：实施参数与《性能调优方案》原始建议一致；若偏离，须标注原因并补充原始建议参数的对照实验
- 报告已落盘

---

### 阶段 2c：终选后全量精度回归（主 agent 执行 · 收尾硬门禁）

阶段 2b 按几何平均加速比选出**最终方案**后，对该方案实现执行一次全量精度回归：

1. 在最终方案或已选择的基线目录下，先确认 `{test_file}` SHA256 与 `precision-baseline.md` 一致，再运行 `TILELANG_DEFAULT_TARGET={tilelang_target} pytest {test_file}`。最终验收不得使用 `-x`，须获得完整逐 case 结果。
2. 使用与 Step 0.5 字节一致的 pytest 精度判断代码判定，要求完整目标用例清单全部 PASS；不另行比较或放大数值阈值。
3. 若出现回归，记录失败 case、完整 pytest 日志、导入源码路径与 SHA256，产出失败报告并停止本轮。主 agent 不修改或回退算子源码；方案实现缺陷的定位、修复和局部回退已经由阶段 2a 的 impl expert 负责，不能在收尾复验中再建立第二套代码实施流程。
4. 产出 `{output_dir}/round{N}/full-precision-regression.md`，包含通用报告字段、后端、实际 target、pytest SHA256 和逐 case PASS/FAIL。

【门禁】相同 pytest 精度判断代码下，`TILELANG_DEFAULT_TARGET={tilelang_target}` 全量回归全部 PASS 才算本轮完成。

---

## Step 2.5：最终代码归档（必做）

### 触发条件

多轮循环终止后、单轮 Step 2 完成后，或 Step 1 判定无需优化后。主 agent 亲自执行。

### 场景识别

| 场景 | 判定 | 校验 |
|------|------|------|
| **A 原始归档** | 代码由 Step 2 impl expert 直接产出，或无可用优化而归档基线 | 受控源码清单校验 |
| **B 恢复归档** | 代码因丢码后重新实施/恢复 | 受控源码清单 + 精度不回归 + 性能不回归 |

> 场景 B 须加跑精度/性能不回归门禁：重新实施的代码即使编译通过、功能精度 bit-identical，也可能在严格精度标准（MARE）或性能口径下回归。

### 执行步骤

**输入**：`BEST_DIR`（最佳方案代码目录，可为空）、`ROUND_BASELINE_DIR`（最佳方案所属轮次的输入基线）、`ORIGINAL_BASELINE_DIR`（原始基线，仅用于总改进记录）、`TEST_FILE`（统一 pytest 文件）、`CASES_CSV`（最终性能 case 文件）、`OPERATOR_RELPATH`（目标算子文件相对工程目录路径）、`BACKEND`/`TILELANG_TARGET`（入口选定值）、`SCENE`（A 或 B）

**1. 选择归档源**：`BEST_DIR` 存在时 `SOURCE_DIR=BEST_DIR`；当 Step 1 判定无需优化，或所有方案都慢于基线而没有 `BEST_DIR` 时，`SOURCE_DIR=ROUND_BASELINE_DIR`，归档状态记为“基线，无可用优化”。

**2. 归档**：使用支持排除规则的复制方式将 `SOURCE_DIR` 内容复制到 `{output_dir}/final_optimized/`，排除 `.git/`、`operators/`、build、dist、`*.egg-info`、缓存、profiling、日志和报告产物；禁止递归复制输出目录。

**3. 受控源码完整性校验**（A/B 均做）：
- 对 `SOURCE_DIR`、`final_optimized/`、`ROUND_BASELINE_DIR` 中受控源码和工程配置生成“相对路径 + SHA256”清单；至少覆盖 `.py/.h/.hpp/.c/.cc/.cpp/.cu/.json/.toml/.yaml/.yml`、`CMakeLists.txt` 和 pytest 配置，并排除第 2 步产物。
- `final_optimized/` 清单必须与 `SOURCE_DIR` 完全一致，否则归档失败。
- `BEST_DIR` 存在时，其清单必须与 `ROUND_BASELINE_DIR` 至少有一项受控源码差异；没有差异则方案实施失败。
- `BEST_DIR` 为空时允许与基线无差异，并须在 manifest 明确写明归档基线及原因。

**3b. 精度不回归门禁**（仅 B）：在 `final_optimized/` 中确认 `TEST_FILE` SHA256 与 `precision-baseline.md` 一致，然后运行 `TILELANG_DEFAULT_TARGET={tilelang_target} pytest {TEST_FILE}`；相同 pytest 精度标准下全部 PASS。回归时正向定位修复，重试 ≤3 轮。

**3c. 性能不回归门禁**（仅 B）：先从 `final_optimized/` 内最新 `OPERATOR_RELPATH` 重新生成场景 B 专用 profiling 文件并完成源码前缀 SHA、`CASES_CSV` SHA、运行时 kernel 名门禁；严禁复用旧文件。随后按 `tilelang-op-profiling` Step 2 一次采集全部性能 case；与丢失前报告按相同 case、相同计时口径对比，几何平均性能 ≥ 丢失前 95%，单 case 退化 ≤ 10%。

**4. 生成 ARCHIVE_MANIFEST.md**：包含通用报告字段、后端、方案轮次、方案标识或“基线”、加速比、受控源码清单摘要、场景 A/B、归档时间、源目录、本轮基线目录、原始基线目录；场景 B 附精度和性能门禁结论。

### 完成判定

- `final_optimized/` 含完整可编译源码 + `ARCHIVE_MANIFEST.md`
- 归档受控源码清单与选定源目录一致；存在 `BEST_DIR` 时还须与该方案所属轮次基线有源码差异
- 场景 B：精度不回归 + 性能不回归门禁通过

### 关键要求

- 场景 B 门禁未通过 → 不归档、如实报告用户

---

## 报告格式通用规范

所有阶段报告都必须包含以下字段，供 tilelang-tuning 解析判断。缺少任一字段视为该阶段未完成：

```markdown
**状态**: ✅完成 / ❌失败 / ⏭️跳过

**阶段**: Step 0.5 精度基线 / Step 1 性能数据采集与分析 / Step 2 方案实施 / Step 2 性能报告 / Step 2 全量精度回归 / Step 2.5 最终归档

**摘要**: 一句话描述本阶段结论

**详细内容**: （各阶段特定内容）
```

### 逐 case 覆盖格式（适用于各阶段报告）

报告中涉及 case 列表时，必须使用以下格式，确保全部 case 覆盖无遗漏：

```markdown
| Case | Shape | dtype | 瓶颈类型 | 适用方案 | 计时口径 | 基线 kernel(us) | 优化后 kernel(us) | 加速比 | 备注 |
|------|-------|-------|---------|---------|----------|-------------------|---------------------|--------|------|
| 1 | xxx | fp16 | VEC BOUND | 方案A | aiv_time | 14.1 | 6.0 | 2.35x | LUT 优化 |
| 2 | xxx | fp32 | VEC BOUND | 方案B | aic_time | 37.6 | 26.7 | 1.41x | Tiling 优化 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
```
