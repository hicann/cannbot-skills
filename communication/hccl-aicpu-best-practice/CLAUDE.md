# hccl-aicpu-best-practice 维护说明

本目录是 AICPU 实现实践 Skill，也是跨层共享技术契约的承载目录，不是 HCCL 源码仓。Dataflow Spec 的知识与模板位于同级 `hccl-aicpu-design`；本目录提供 AICPU template 技术参考、骨架生成、静态检查和专项检视能力。

本文仅约束 Skill 本身的维护。Agent 职责、用户交互、写入权限、任务状态、跨层构建测试与完成判定由 `plugins-official/hccl-op-dev/` 定义；本 Skill 不扩大授权或安排角色。

## 外部仓输入

需要真实 HCCL 源码的脚本只接受调用方显式提供的 `--repo` 或 `HCCL_REPO`。缺失时返回输入错误；不猜常见路径、不扫盘、不把示例路径当成默认值。

使用外部仓前确认它是 Git 仓，并记录当前分支、HEAD 和工作区状态。非 Git 仓或 detached HEAD 应作为显式状态返回上游。

## 常用维护检查

不依赖 HCCL 构建环境的离线回归：

```bash
python3 -B scripts/test_checks.py
python3 -B ../../tests/behavior/teams/check_hccl_aicpu_rules.py
python3 -B ../hccl-aicpu-design/scripts/test_design.py
```

需要显式 HCCL 仓输入的能力：

```bash
python3 scripts/check_template.py --repo "$HCCL_REPO" --all
python3 scripts/list_templates.py --repo "$HCCL_REPO" --blueprint
python3 scripts/new_template.py --repo "$HCCL_REPO" --op all_reduce \
  --class InsTempAllReduceMesh1DFoo --pattern all-reduce-mesh-oneshot --dry-run
```

`check_template.py` 和 `list_templates.py` 只读；设计 Skill 的 `check_spec.py` 也只读。`new_template.py` 会写入 template `.h/.cc` 和两处 CMake；是否可执行由调用方的权限契约决定，先用 `--dry-run` 检查落点。

Skill 骨架或脚本变更后，在具备外部仓输入时运行：

```bash
bash scripts/selftest.sh "$HCCL_REPO"
```

`selftest.sh` 默认在 `/tmp` 中创建隔离 worktree，编译产物也位于临时目录。`--in-place` 会临时改动原工作树，调用方必须已具备写入授权。`--with-aicpu-build` 用于 Skill 骨架的 device 侧自测，不代替候选源码的完整构建。

## 目录职责

| 路径 | 用途 |
|---|---|
| `SKILL.md` | 能力入口、输入输出契约和按需参考索引 |
| `../hccl-aicpu-design/dataflow-spec-template.md` | 模式 C 的 Dataflow Spec 输入模板 |
| `references/` | API、资源、数据流、同步、规则和 CMake 证据接口 |
| `references/10-implementation-entry.md`、`11-operator-contracts.md` | 已确认算法设计的实现入口与 executor 接线契约，不要求已有算法蓝本 |
| `references/source-baseline.json`、`scripts/check_sources.py` | 已核对来源的内容指纹和只读差异检查；先核对契约再更新基线 |
| `references/12-allgather-parallel.md` | 按需加载的四阶段契约；扩展来源独立标注 commit，用 `--contract all-gather-parallel` 校验 |
| `references/allgather-integration.md` | 实现接入卡：初始化、成本参数、注册和选择器版本；`--contract all-gather-integration` 校验 |
| `references/13-historical-examples.md` | 可选历史定位索引；固定 commit/路径/哈希，不将旧算法设计固化为通用规则 |
| `assets/*.tpl` | `new_template.py` 使用的代码骨架与许可头 |
| `../hccl-aicpu-design/assets/dataflow-spec.example.md` | 从真实 AllReduce Mesh one-shot template 逆向的 spec 范例 |
| `scripts/*.py` | 实现源码/CMake 检查、存量索引和骨架生成；规格/布局检查位于设计 Skill |
| `scripts/selftest.sh` | 离线回归、故障注入和骨架编译 probe |

SKILL.md 仅保留核心契约和路由，详细技术内容下沉到 `references/`。外部仓事实必须来自目标版本真实源码或 CANN/HCCL 可信文档。

## 必须同步的耦合点

- 接口清单：`scripts/check_template.py` 的 `REQUIRED` ↔ `SKILL.md` §3 ↔ `references/03-resource-model.md`。
- 真仓 API 镜像：骨架的 `CalcCostCoeff`/`props` ↔ `references/03-resource-model.md` ↔ 目标仓 `cost_model.h`、`common_alg_template_base.h`、`alg_parse.h`。
- Dataflow 原语：设计 Skill `scripts/check_spec.py` 的 `WRITE_PRIMS`/`READ_PRIMS`/`LOCAL_PRIMS` ↔ `../hccl-aicpu-design/references/07-dataflow-spec.md`。
- Spec 章节：设计 Skill `scripts/check_spec.py` 的 `SECTIONS` ↔ `../hccl-aicpu-design/dataflow-spec-template.md` 的 `## 1..10`；表格关键字改名时同步 `table_value()` 调用。
- repeat/stride：Dataflow Spec 第 5 章 ↔ `references/09-template-data-params.md` ↔ 设计 Skill 的 `scripts/check_layout.py` 和量化回归。
- CMake 锚点：`new_template.py`/`cmake_utils.py` ↔ `references/05-build-and-verify.md` §1 ↔ 目标仓实际 CMake。
- 骨架占位符：`assets/*.tpl` ↔ `new_template.py` 的 `render()`/`subs`。
- 自定义算法路由：设计/实现 Skill ↔ `R-PROC-003/004` ↔ Architect；不得恢复成新增算法必读完整存量蓝本。
- 算子卡与 API 动作导航：相关来源变化时更新对应结论与 `source-baseline.json`，不通过只刷新哈希宣称核验。
- Parallel 接入卡：布局、资源与成本传递变化分别核对；`check_cost_forwarding.py` 仅覆盖直接聚合初始化，不把脚本 PASS 当作完整接入证明。
- 成本检查按消费者选择 `--require-field`；默认仅 algName，兼容旧命令。来源检查保留总体状态并输出逐组状态，不能用未变组掩盖实际依赖的变化组。
- 调研交接：自定义入口 ↔ Architect/prompt；传递实际加载路径、已有依据和未解问题，不新增默认子代理或绕过独立检视。
- 规则登记：`references/07-rules.md` ↔ 实现 Skill 的 `check_template.py` / 设计 Skill 的 `check_spec.py` 的 `RULES` 字典 ↔ 仓库测试 `tests/behavior/teams/check_hccl_aicpu_rules.py` 词表。

## 能力级不变式

- 只有新增算法或修改 slice、scratch、同步点、收发模式时，Dataflow Spec 才是实现前置条件。
- 技术 TBD 用真实源码、公式和接口契约消解；需求歧义作为上游输入缺口返回。
- AICPU template 必须同时登记进局部 `CMakeLists.txt` 和 `src/scatter_aicpu_kernel.cmake`。
- 跨 rank 边只描述一个方向；写模式仅消费 tx slices，读模式仅消费 rx slices。
- `CalcScratchMultiple` 与单次 repeat 的最大 scratch 写入范围一致；repeat/stride 由 executor 提供，template 只读。
- 静态检查和 Skill 自测不替代完整构建或 hccl-vm Checker。跨层技术判据见 `aicpu-shared-contract.md` 和 `references/build-verification.md`。
