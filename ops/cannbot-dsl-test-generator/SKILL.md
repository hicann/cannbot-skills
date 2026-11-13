---
name: cannbot-dsl-test-generator
description: "当需要设计、补充或校验 CANNBotDSL 算子测试，或生成独立 CPU Golden 与测试入口时使用；根据调用方提供的规格、设计及实现单元检验义务生成黑盒、白盒、TDD 用例及用户性能 case 的标准化及校验结果。"
metadata:
  category: testing
---

# CANNBotDSL 测试生成

## 输入

- 算子规格文件，确定公开接口、数学语义、边界和容差。
- 调用方提供的一份设计 Markdown 文档，供 Golden 和白盒用例引用；文件名与所在目录由调用方决定。
- 调用方提供的 `U*.yaml` 所在目录，供 TDD 用例读取实现边界和检验义务；所需字段见 [TDD 输入与证据契约](references/evidence.md)。
- 可选：已有的测试目录及 `testcase.csv`。各模块在同一 CSV 中维护所属分组。

## 输出

在调用方指定的 `TEST_DIR/` 下维护 `testcase.csv`、`<op>_golden.py`、`test_<op>.py`、固定入口辅助模块和 `testcase_output/{blackbox,whitebox,tdd,performance}/`。算子名来自 `spec.op.name`。脚本校验本 Skill 产出的用例与 Golden，具体用例由测试设计者确定。

各脚本通过 `--spec SPEC`、`--design DESIGN`、`--units-dir UNITS`、`--test-dir TEST_DIR` 接收当前步骤所需的输入和输出位置。 Unit 接收契约预检单独使用 `scripts/build_tdd_cases.py --check-unit-contracts --units-dir UNITS`，只读且不依赖其他测试资产；它与正式 TDD 校验共用契约实现，通过不等于用例或设备结果已通过。

## 精度契约预检

规格冻结前或单独检查测试工具兼容性时，只读取 spec，运行 `scripts/check_tolerance.py --spec SPEC`。此入口不依赖 DESIGN、Unit、Golden 或设备，不生成测试资产；输出含 spec 哈希、逐 dtype 的指标与 backend，所选内置策略。未指定时报告统一的内置默认策略，规格冻结时将实际值及依据写入 per_dtype；已有明确契约时，失败须补齐能力或确认依据，不能用默认值覆盖不受支持的要求。

生成器和示例的初始容差不构成明确契约；`--defaults` 只读输出统一默认提案，不验证或改写当前精度配置。写回待冻结 spec 后须重新普通预检和完整规格校验，不能用提案结果替代。

固定 runner 使用内置 `max_relative`、`bitwise_equal`、`mean_relative` 和 `staged_relative`，按已确认的数值契约比较独立主参考。未指定数值契约时，支持的浮点 dtype 默认 staged_relative 完整两阶段，整数/布尔默认 bitwise_equal；用户明确要求的其他策略优先；资料仅列 MERE/MARE 和阈值不等于要求单阶段，须核对完整判定流程。默认两阶段必须绑定独立同精度参考，缺失或异常明确失败，不自动采用主参考截断。`mean_relative` 保留单阶段平均/最大相对误差，staged_relative 在阶段一失败后执行完整分域兜底，所有正式模式使用同一比较器；`backend` 省略或为 `native`，其他值明确失败，不自动切换策略。主参考可按契约采用同精度或高精度计算，不能由输出 dtype、资料来源或候选结果自动决定。公式、参考职责及非有限值语义见 [Golden 契约](references/golden.md#精度判定契约)。预检只证明工具可消费配置，不证明数值算法正确；来源、参考路径、公式和阈值仍须独立核对。

## 按任务读取与分批写入

只执行调用方要求的模块。黑盒设计仅需 spec、[用例契约](references/cases.md) 和 `build_blackbox_cases.py --inventory`；不预读 DESIGN、Unit、Golden 文档或 runner 源码。正常使用脚本入口，只有实际报错或契约无法解释时，才读相关函数及直接依赖，避免先通读整个 scripts 目录。

完整测试生成按以下产物顺序推进；已有有效产物先检查并复用：

1. 黑盒：读取 inventory 后，在正式 `TEST_DIR/testcase.csv` 创建固定表头并写入已确认的 L0 核心用例，再分批补 L1/L2；已有 CSV 只更新 blackbox 行，使用分组安全更新入口保留其他分组。
2. Golden 与入口：只读 spec 和设计中相关数学/接口章节，保存完整可解析的 Golden，运行 `prepare_golden.py` 校验并安装固定入口，再进行 CPU 数值复核。完成这一步后再展开白盒和全部 Unit。
3. 白盒：按设计索引逐个读取真实执行分支，完成一批即写入 whitebox 行；每个 active 行显式填写 `expected`，判定选择见[用例契约](references/cases.md#白盒)；无需预读 Unit。
4. TDD：按 Unit 依赖顺序逐份读取 YAML，逐批保存相应义务的 tdd 行；最后执行全量义务、依赖与关联检查。

每批写入后先检查 CSV/YAML/Python 能否解析及本批字段、ID/引用是否正确；当前脚本的完整覆盖检查留到对应模块补齐后执行。部分内容缺失时记录待补项，不用 excluded 或放宽校验掩盖未完成。中断恢复先读已有产物和缺项，不重建已确认内容。分批保存不降低最终验收要求，也不增加独立计划文件或中间格式。

## CSV 契约

使用 [用例契约](references/cases.md) 中的固定列。每行 `sheet` 为 `blackbox`、`whitebox`、`tdd` 或 `performance`，`source` 与 `sheet` 相同，`status` 为 `active` 或 `excluded`。可执行行必须提供 `inputs_json`；排除行不执行，必须提供 `exclude_reason`。黑盒的 `design_ref` 为 spec 路径列表，`condition` 为 L0/L1/L2；白盒的 `design_ref` 为设计文件中的实际条目，`condition` 为分支条件；TDD 使用 `unit_id`、`obligation_ids` 和可选 `branch_ids` 追溯原生 Unit。固定测试入口只执行 active 行。

## 黑盒

先运行 `scripts/build_blackbox_cases.py --spec SPEC --test-dir TEST_DIR --inventory`，从标准输出审查 spec 因子和必需路径。测试设计者直接编写 CSV 的 blackbox 行，覆盖 L0 核心、L1 组合/边界/极值以及规格声明的 L2 异常；无法执行的必需路径写 excluded 行及理由。运行 `scripts/build_blackbox_cases.py --spec SPEC --test-dir TEST_DIR` 校验输入配方、错误语义、spec 路径和覆盖；`--coverage` 可重新打印报告，不落盘。生成阶段不执行目标算子。

## 用户性能用例

只标准化调用方提供的 case，不自主选择新 shape、dtype、布局或扩大场景；缺项或与 spec 冲突时返回具体原因。仅需 spec 与用户资料，不依赖 DESIGN、Unit 或 Golden。将确认输入写入 performance 行，按 [用例契约](references/cases.md) 填字段，运行 `scripts/build_performance_cases.py --spec SPEC --test-dir TEST_DIR` 校验，无需黑盒覆盖率、白盒分支或 Unit 义务。无用户 case 时交付表头且该分组为空，报告明确未提供；不补默认 case、不排除用户 case 掩盖缺项。

所有用例写入使用 `scripts/update_case_group.py --test-dir TEST_DIR --sheet <分组>`：标准输入为该分组完整行的 JSON 数组，单元格均为字符串（JSON 配方先编码），空数组用于无 case；锁内重读保留其他组并原子保存。不直接整表覆盖；同一分组保持单写入者。该接口只检查公共格式，保存后执行对应语义校验。

后续测试生成复用同一 CPU Golden 和缓存检查 performance 输入；runner 的全量入口 `--all` 自动包含该组，单组检查可用 `--sheet performance`。两者仅检查正确性，不预热、计时或判定性能达标。空性能组不执行，非空时全部 active case 必须验证；计时、采样和 Profiling 由调用方性能流程负责。

## 独立 Golden 与固定入口

依据 spec 和 DESIGN 直接编写 `TEST_DIR/<op>_golden.py`。其中包含 `DESIGN_REFS`、与 spec 参数一致的 `<op>_golden`、公开接口元数据；CPU 实现不得导入被测算子或设备库。可参考 [Golden 契约](references/golden.md)。人工审查数值路径后运行 `scripts/prepare_golden.py --spec SPEC --design DESIGN --test-dir TEST_DIR` 校验最终 Golden，并安装固定 `test_<op>.py` 与三个结果目录。静态校验不能证明数值等价，须用 CPU 样例和边界复核。

若 spec 声明了 `op.aicpu_kernel`，算子实现模块须提供 `_aicpu_preprocess(inputs)` 钩子：NPU 测试时固定入口自动调用该钩子对输入做 AICPU 前置处理，再执行主算子；CPU Golden 不感知此流程，保持纯 CPU 实现。

## 白盒

依据所提供设计文件中的算法、执行路径和验证矩阵直接编写 CSV 的 whitebox 行。每个目标对应可执行分支或带理由的 excluded 行；分支只使用 spec 公开输入。运行 `scripts/build_whitebox_cases.py --spec SPEC --design DESIGN --test-dir TEST_DIR` 核对 DESIGN 引用、输入配方和分支 ID，不依赖 Unit 或 Golden。

## 实现单元 TDD

依据 `UNITS/U*.yaml` 的 `verification_obligations` 逐项编写 CSV 的 tdd 行，明确公开输入、预期结果、断言和可选白盒分支 ID。数值义务在正式准备和执行时使用 Golden，semantic/device 义务验证公开可观察行为；调用方拆为并行用例草拟时，仅核对合法输入、必要特征、触发、判别依据和义务绑定，不读取或等待并行 Golden/runner，不自行模拟完整算法裁定设计精度。参考路径及同精度舍入由 Golden 按 [Golden 契约](references/golden.md#独立主参考与计算精度) 确定，草拟不提前调用下述依赖完整资产的正式校验，交接后由集成及真实目标测试消费；输入不变性使用测试侧 `input_unchanged`。纯结构承诺由负责验收的 Agent 通过 review_evidence 核验；测试诊断与正式算子分离。遇到输入契约不兼容时，执行 Agent 按证据契约处理并保留验证责任，不向使用者派发内部文件修改任务。实际有效工作量、连续调用与重复验证按 [TDD 输入与证据契约](references/evidence.md) 编写，物化配方确认触发，不能仅靠义务 ID 关联宣布覆盖。完整资产汇合后运行 `scripts/build_tdd_cases.py --spec SPEC --design DESIGN --units-dir UNITS --test-dir TEST_DIR`，校验义务与用例覆盖、TDD 输入配方及实际触发条件、白盒关联及 Golden；直接生成正式资产时不另写中间文件，调用方明确拆分的并行草稿沿其既有暂存与写入权限执行。

`build_tdd_cases.py` 校验成功后，增量准备所有分组全部 active 用例的 CPU 输入和 Golden 缓存，供后续 Red、Green 和回归复用；不导入目标实现或运行 NPU。缓存由 CSV 和 Golden 派生，失效与独立复跑规则见 [脚本生成与执行验收](references/execution.md)。

## 验收

生成运行脚本及验收入口时读取 [脚本生成与执行验收](references/execution.md)，落实退出码、当前结果、可迁移路径与辅助模块交付要求。

以下按本次受托模块适用；仅黑盒设计不要求提前生成 Golden、白盒和 TDD。完整测试生成须满足全部条目。

- CSV 三组 active 用例非空、ID 唯一；所有 excluded 行有明确理由且不会执行。
- 黑盒可追溯 spec，白盒可追溯 DESIGN，TDD 每项义务能回指原生 Unit 和具体用例。
- Golden 的数学、设计阶段与容差与 spec 和设计文件一致；固定入口全量执行使用 `--all --device <设备>`，单组使用 `--sheet <分组> --device <设备>`，设备验证时使用 `--require-accelerator`。参数及结果复用规则见执行验收资料。
- 复核每项 must_distinguish 的输入与判定确实能区分错误实现；核对自然语言 coverage_axes 和结构检查证据来源，脚本通过不代替这些结论。义务无法由当前公开输入及断言表达时，返回具体缺口。
- 不修改冻结的 spec、DESIGN、Unit、目标实现或测试阈值来换取通过。
