---
name: cannbot-dsl-test-generator
description: "当需要设计、补充或校验 CANNBotDSL 算子测试，或生成独立 CPU Golden 与测试入口时使用；根据调用方提供的规格、设计及实现单元检验义务生成黑盒、白盒和 TDD 用例及校验结果。"
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

在调用方指定的 `TEST_DIR/` 下维护 `testcase.csv`、`<op>_golden.py`、`test_<op>.py`、固定入口辅助模块和 `testcase_output/{blackbox,whitebox,tdd}/`。算子名来自 `spec.op.name`。脚本校验本 Skill 产出的用例与 Golden，具体用例由测试设计者确定。

各脚本通过 `--spec SPEC`、`--design DESIGN`、`--units-dir UNITS`、`--test-dir TEST_DIR` 接收当前步骤所需的输入和输出位置。

## 按任务读取与分批写入

只执行调用方要求的模块。黑盒设计仅需 spec、[用例契约](references/cases.md) 和 `build_blackbox_cases.py --inventory`；不预读 DESIGN、Unit、Golden 文档或 runner 源码。正常使用脚本入口，只有实际报错或契约无法解释时，才读相关函数及直接依赖，避免先通读整个 scripts 目录。

完整测试生成按以下产物顺序推进；已有有效产物先检查并复用：

1. 黑盒：读取 inventory 后，在正式 `TEST_DIR/testcase.csv` 创建固定表头并写入已确认的 L0 核心用例，再分批补 L1/L2；已有 CSV 只更新 blackbox 行，保留其他分组。
2. Golden 与入口：只读 spec 和设计中相关数学/接口章节，保存完整可解析的 Golden，运行 `prepare_golden.py` 校验并安装固定入口，再进行 CPU 数值复核。完成这一步后再展开白盒和全部 Unit。
3. 白盒：按设计索引逐个读取真实执行分支，完成一批即写入 whitebox 行；无需预读 Unit。
4. TDD：按 Unit 依赖顺序逐份读取 YAML，逐批保存相应义务的 tdd 行；最后执行全量义务、依赖与关联检查。

每批写入后先检查 CSV/YAML/Python 能否解析及本批字段、ID/引用是否正确；当前脚本的完整覆盖检查留到对应模块补齐后执行。部分内容缺失时记录待补项，不用 excluded 或放宽校验掩盖未完成。中断恢复先读已有产物和缺项，不重建已确认内容。分批保存不降低最终验收要求，也不增加独立计划文件或中间格式。

## CSV 契约

使用 [用例契约](references/cases.md) 中的固定列。每行 `sheet` 为 `blackbox`、`whitebox` 或 `tdd`，`source` 与 `sheet` 相同，`status` 为 `active` 或 `excluded`。可执行行必须提供 `inputs_json`；排除行不执行，必须提供 `exclude_reason`。黑盒的 `design_ref` 为 spec 路径列表，`condition` 为 L0/L1/L2；白盒的 `design_ref` 为设计文件中的实际条目，`condition` 为分支条件；TDD 使用 `unit_id`、`obligation_ids` 和可选 `branch_ids` 追溯原生 Unit。固定测试入口只执行 active 行。

## 黑盒

先运行 `scripts/build_blackbox_cases.py --spec SPEC --test-dir TEST_DIR --inventory`，从标准输出审查 spec 因子和必需路径。测试设计者直接编写 CSV 的 blackbox 行，覆盖 L0 核心、L1 组合/边界/极值以及规格声明的 L2 异常；无法执行的必需路径写 excluded 行及理由。运行 `scripts/build_blackbox_cases.py --spec SPEC --test-dir TEST_DIR` 校验输入配方、错误语义、spec 路径和覆盖；`--coverage` 可重新打印报告，不落盘。生成阶段不执行目标算子。

## 独立 Golden 与固定入口

依据 spec 和 DESIGN 直接编写 `TEST_DIR/<op>_golden.py`。其中包含 `DESIGN_REFS`、与 spec 参数一致的 `<op>_golden`、公开接口元数据；CPU 实现不得导入被测算子或设备库。可参考 [Golden 契约](references/golden.md)。人工审查数值路径后运行 `scripts/prepare_golden.py --spec SPEC --design DESIGN --test-dir TEST_DIR` 校验最终 Golden，并安装固定 `test_<op>.py` 与三个结果目录。静态校验不能证明数值等价，须用 CPU 样例和边界复核。

## 白盒

依据所提供设计文件中的算法、执行路径和验证矩阵直接编写 CSV 的 whitebox 行。每个目标对应可执行分支或带理由的 excluded 行；分支只使用 spec 公开输入。运行 `scripts/build_whitebox_cases.py --spec SPEC --design DESIGN --test-dir TEST_DIR` 核对 DESIGN 引用、输入配方和分支 ID，不依赖 Unit 或 Golden。

## 实现单元 TDD

依据 `UNITS/U*.yaml` 的 `verification_obligations` 逐项编写 CSV 的 tdd 行，明确公开输入、预期结果、断言和可选白盒分支 ID。结构义务需要可执行的 `probe_events` 或 `probe_predicates`；数值义务使用 Golden，并可在同次调用联合结构断言。实际有效工作量、连续调用与重复验证按 [TDD 输入与证据契约](references/evidence.md) 编写，物化配方确认触发，不能仅靠义务 ID 关联宣布覆盖。运行 `scripts/build_tdd_cases.py --spec SPEC --design DESIGN --units-dir UNITS --test-dir TEST_DIR`，校验义务与用例覆盖、TDD 输入配方及实际触发条件、白盒关联及 Golden；不写中间文件。

## 验收

以下按本次受托模块适用；仅黑盒设计不要求提前生成 Golden、白盒和 TDD。完整测试生成须满足全部条目。

- CSV 三组 active 用例非空、ID 唯一；所有 excluded 行有明确理由且不会执行。
- 黑盒可追溯 spec，白盒可追溯 DESIGN，TDD 每项义务能回指原生 Unit 和具体用例。
- Golden 的数学、设计阶段与容差与 spec 和设计文件一致；固定入口按 `--sheet blackbox|whitebox|tdd --device <设备>` 执行，设备验证时使用 `--require-accelerator`。
- 复核每项 must_distinguish 的输入与判定确实能区分错误实现；核对自然语言 coverage_axes 和结构采集来源，脚本通过不代替这些结论。义务无法由当前公开输入及断言表达时，返回具体缺口。
- 不修改冻结的 spec、DESIGN、Unit、目标实现或测试阈值来换取通过。
