---
name: ops-repo-migration
description: >-
  将已有自定义算子工程（包括 op-develop 工作流产物）迁入 ops-math、ops-nn、ops-cv 或
  ops-transformer 开源仓，适配目录、注册、CMake、UT 和 example，出包并回归原有交付件。
  触发：已有算子包迁移、迁移构建问题。保持 kernel、tiling、推导和 ACLNN 实现语义，
  不用于新算子开发或性能调优。不支持 ops-rand。
---

# 算子迁入 ops-* 开源仓

把当前任务的已有算子源包适配为目标仓的 `{op_class}/{OP}/`，出包验证，最终形成一个迁移 commit。
工作流的源包为 `$WORK_DIR/op`；独立调用使用任务提供的源包路径。
共用本流程；只读取选中仓的布局文档。仓级 CMake、同类算子和工具目录都是只读参考。

## 输入与目标契约

- `{OP}`：当前 spec 的 snake_case 算子名，与源工程核对后用于目录和构建参数；注册类型 `{OpType}` 以源工程 `OP_ADD` / `REG_OP` 为准，不能用朴素 PascalCase 猜缩写。
- ACLNN 接口名和签名、输入输出/属性、dtype/format、soc/tiling 目录与 tilingKey、golden、套件及用例数均取当前源工程及已验证产物。安装 vendor 路径从安装记录和实际文件确认，不由注册类型推算。
- 参考算子和测试 fixture 只说明仓库契约，不限定待迁移算子的名字、计算或接口；不得复制它们的名称、双输入/单输出、dtype、芯片或测试规模作为当前任务参数。
- 源包、已验证的 CANN 环境、目标仓 checkout，以及目标 `target_repo` / `op_class`。
- 工作流调用时读取 `$WORK_DIR/.workflow/migrate-decision.json`；独立调用时从用户要求和 spec 确定目标。
- 选仓依据见 [选仓规则](references/target-selection.md)。显式目标优先；泛类别有歧义时记录原因，不能擅自落入某仓。

| 目标 | 标准分类 | 布局与验证差异 |
|---|---|---|
| ops-math | math / conversion / random | [ops-math](references/target-layout-ops-math.md) |
| ops-nn | matmul / conv / activation / foreach / hash / vfusion / index / loss / norm / optim / pooling / quant / rnn / control | [ops-nn](references/target-layout-ops-nn.md) |
| ops-cv | image / objdetect | [ops-cv](references/target-layout-ops-cv.md) |
| ops-transformer | attention / ffn / gmm / mamba / mc2 / mhc / moe / posembedding | [ops-transformer](references/target-layout-ops-transformer.md) |

[repo-profiles.json](references/repo-profiles.json) 是脚本和工作流共用的仓 URL、分类、包名、vendor 后缀及检查配置。
`ops-rand` 不在支持范围：随机采样仍走 ops-math/random。

## 仅作分析时的证据边界

用户只要求方案分析或未提供源包时，可以依据给定信息说明选仓与保留要求，
但须区分**用户给定、实际已核验、尚待核验**，不能把题设参数或参考文档当作源工程证据。
明确列出缺少的源材料及后续核验位置：OpDef/注册、完整 API 签名、dtype/format、
soc/tilingKey、安装 vendor 路径、golden 与套件数量；未读到的内容标为未核验，不猜值。
只作分析的结论须说明尚未完成源契约核验或迁移验证；用户禁止执行时，
不为补证据运行命令、构建或访问硬件。实际迁移仍按下述流程先锁定源契约。

## 工作流程

0. **锁定输入与环境**。从上述事实源记录当前算子契约及来源到迁移报告；缺失或互相矛盾时报告阻碍，不以样例补齐。工作流复用 `$WORK_DIR/ops-test-kit/ttk-verify/env-info.txt`、`$WORK_DIR/op/examples/verify.log` 已证明的 toolkit，TTK 固定 `$WORK_DIR/ops-test-kit`。不得重新选版本或重复询问。独立调用先复用任务给定环境，缺失时才询问。清理混入的 `LD_LIBRARY_PATH` 后 source 这一套 `set_env.sh`；有 GEIR 验证时用 `preflight.sh --require-geir` 检查 `opp/built-in/op_proto/inc/array_ops.h`。构建并行度服从当前机器限制，示例用 `-j8`。
1. **确认目标**。用 `scripts/migration_contract.py profile <target_repo>` 读配置，读该仓布局和当前 checkout 的构建实现；记录 commit。分类必须真实存在且被构建收集；不能用 examples、common、torch_extension 或 experimental 代替分类。版本漂移先核实契约。
2. **隔离迁移**。工作流克隆到 `$WORK_DIR/migration/<target_repo>`，不写 references 下的参考仓。校验已有 checkout 的远端、基线与状态，建 `feat/{OP}`；目标已存在时不得覆盖。仅拷贝源文件，不带构建缓存、安装包或测试输出。
3. **清理模板**。在目标副本移除独立 `build.sh`、构建缓存、工作流专用测试驱动和无关模板残留。`*_proto.cc` 仅在确认是模板空壳后移除，真实注册不能丢失。原工作流的测试和 golden 保持不动。
4. **许可证**。按目标仓 `OAT.xml` 和同类算子补 CANN 许可证注释，保留合法原作者信息。不要靠更改代码绕过门禁。
5. **注册适配**。保持类型、签名、dtype/format、属性、`.AutoContiguous()` 和源包支持的 soc 集。`opFile.value` 与根目录入口及仓库暂存路径匹配（见共同配方）；每个 soc 的注册与编译覆盖都要核对。允许必要的注册胶水适配，不改计算、tiling 或推导。
6. **构建适配**。严格按该仓布局接入。math/nn 可用单个顶层 `add_all_modules_sources`；cv 还要显式接入 UT；transformer 使用 `op_host/CMakeLists.txt` 的 `add_modules_sources_with_soc`。不可把 math 模板套到全部仓。手写 ACLNN 的 `ACLNNTYPE` 用 `aclnn_exclude`。
7. **最小 example**。按源包真实通路写 `examples/test_aclnn_{OP}.cpp`、`test_geir_{OP}.cpp`，只验证调用与输出可读；完整精度验证仍用冻结用例。不保留会被 example 扫描器误收的草稿。
8. **接入测试**。按目标仓 UT glob 和接入链改写 harness；期望值仍来自已验证源包。golden 整文件拷到 `tests/assets/golden.py`，保持计算逻辑不变；许可证注释及目标仓必需的格式化按 [格式化核验](references/formatting-validation.md) 处理。源 golden 和冻结用例保持只读；不得靠抽样替代工作流回归。
9. **验证**。先运行目标文件的 pre-commit 并核验格式化差异，再做结构检查 → 出包 → 安装隔离目录 → 仓内 UT / example → 工作流冻结用例回归 → 源码差异核验。命令和产物对照见 [共同配方](references/migration-recipe.md)。对每个支持的 soc 检查产物；运行条件缺失要报告未验证，不能当通过。UT 必须执行到预期用例，零用例不算通过。
10. **提交与报告**。目标算子树形成一个 commit，不 push、不开 PR；不捏造共同作者。报告记录目标/分类/基线/分支/commit、产物清单、用例数、命令退出码和日志。只提交目标树，构建和证据留在工作目录。

## 完成判据

执行下列命令前，将 `SKILL_DIR` 设置为当前读取的 `SKILL.md` 所在目录的绝对路径，
即本技能的实际安装目录；不要把算子工作目录当作技能目录。

- `bash "$SKILL_DIR/scripts/post_migration_check.sh" <目标算子目录> <soc> <target_repo> <源注册类型>` 无 FAIL；所有源包支持的 soc 都显式检查过，迁移后的 `OP_ADD` / `REG_OP` 与源类型一致。
- 仓顶层 `build.sh --pkg --ops={OP} --vendor_name=<vendor> --soc=<soc> -j8` 成功，唯一匹配的 `.run` 已装入隔离目录。包名与 vendor 后缀从 profile 读取。
- 安装产物与 dev-package 基准的接口、soc、tilingKey 覆盖对应；不要求二进制逐字节一致。
- `-u --ophost --ops={OP}` 的预期 UT 实际执行；适用的 eager / graph example 都成功。没有通路时需用源包证据说明 N/A。
- 工作流调用必须重跑原 kernel / ACLNN / GEIR 冻结套件与已有 torch 冒烟，并证明运行命中了新包；不得改用例、golden、容差或阈值。
- 调用进程须完成资源清理并正常退出；数值 PASS、框架父进程退出 0 均不能掩盖 worker 或原生驱动的析构异常。自定义库的内部符号隔离与公共 ABI 核对见目标布局和共同配方。
- `verify_logic_unchanged.sh` 是**差异清单工具**，工具/预处理失败必须解决；正常退出 0 本身不证明无改动。逐项解释 DIFFERS / ONLY-IN-SRC / ONLY-IN-DST；kernel、tiling、InferShape/InferDataType、ACLNN 和目标 golden 只允许注释与经核验的必要格式化，计算逻辑和行为必须保持。新增/删除不能绕过核验。
- 提交前 pre-commit 必须通过且不再改写受测文件；若门禁又改写文件，先核验差异并重跑受影响的构建与测试，不能提交未受测版本或跳过 hook。

## 工具与参考

- `scripts/preflight.sh [--require-geir] <ascend_env_path> [repo_path] [op_class] [OP] [target_repo]`：环境、分类、占用及 Git 前置检查；仅在源包需要 GEIR 回归时要求其头文件。
- `scripts/migration_contract.py profile <target_repo>`：输出仓配置；`decision <decision_json_path>`：读取该路径的 JSON 文件，校验选仓记录结构与允许的目标/分类；不接收内联 JSON 字符串。
- `scripts/post_migration_check.sh <operator_dir> <soc> [target_repo] [source_optype]`：配置驱动的迁移自检，soc 必填且取自源契约；迁移验收还须传入源注册类型。未给 target_repo 时仅从仓目录名识别，未知名称要求显式传入。
- `scripts/verify_logic_unchanged.sh <src_dir> <operator_dir>`：去注释差异清单。
- `scripts/audit_repo_layout.py ops-math=/path/to/math ops-nn=/path/to/nn ops-cv=/path/to/cv ops-transformer=/path/to/transformer`：只读结构对拍；在临时目录用真实仓库 UT 收集函数执行 CMake glob，不编译、不需 CANN/NPU，输出各仓 commit 和结果。
- [共同迁移配方](references/migration-recipe.md)：环境、许可证、出包与验收；[ops-math 已验证问题](references/pitfalls-ops-math.md) 仅在 math 迁移或同类症状出现时读取，四仓复杂模拟交付件的覆盖范围与限制见 [验证矩阵](references/validation-matrix.md)。

迁移只允许改目录、构建、注册胶水、注释、测试驱动及经核验的目标仓必需格式化。若必须改 kernel 计算、tiling 公式、推导语义、ACLNN 行为或仓级代码才能构建，停止并说明具体阻碍，不把开发/调优混入迁移。
