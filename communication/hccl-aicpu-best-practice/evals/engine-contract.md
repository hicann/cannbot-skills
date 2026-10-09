---
skill_name: hccl-aicpu-best-practice
---

# AICPU 共享契约评测历史

评测分为能力契约问答、离线工具行为和真实源码验证。问答通过不代表 HCCL 算法已经编译或通过 Checker。

## 能力契约问答

原共享入口的 16 项文本用例已并入本目录的 `evals.json`（编号 3–18），覆盖分层路由、上游输入缺口、Skill/Agent/Plugin 边界、构建前基线、双轨验收、同步知识，以及已经授权的 selector 变更、自定义算法、来源漂移、成本参数、验证顺序与调研交接。由评测运行器向独立会话加载 `hccl-aicpu-best-practice`、[共享跨层契约](../aicpu-shared-contract.md)与必要参考后给出 prompt，以 expected_output 人工判定；禁止工具的用例只评估契约理解。

关键判据：继承已有信息与授权；不编造 API 或验证成功；默认不追加 HCCL ST/UT；仅显式启用时默认选路不变；已批准的默认选路变更按预先固定的矩阵验收。当前 `hccl-test-tool-hvm/evidence.py compare` 不支持放宽选路比较，须明确区分自动比较与人工矩阵验收。

## 离线工具行为

在本仓根目录运行：

```bash
python3 -B -m unittest discover -s communication/hccl-aicpu-best-practice/tests -v
python3 -B communication/hccl-aicpu-best-practice/scripts/test_checks.py
python3 -B tests/behavior/teams/check_hccl_aicpu_rules.py
bash tests/behavior/teams/test-hccl-op-dev-workflow.sh
```

用例均在临时目录写入，覆盖：

| 行为 | 失败判据 |
|---|---|
| 源码快照与构建前置检查 | 不能恢复源码、篡改/缺失证据未被拒绝、仓内输出目录被接受、重复构建 |
| 新旧布局骨架与 CMake 接线 | 把注释内条目认作有效登记、兼容分支错误、接线失败仍落盘 |
| 生成路径边界 | 绝对路径、父目录穿越、符号链接可导致目标外写入，包括 `--force` |
| 自定义骨架与来源复用 | 未实现骨架返回成功、Ring/Tree 被误标 Mesh、无关算子变化导致匹配失效、缺失来源仍报 MATCH |
| 成本字段与分组来源 | 仅补 algName 就放过消费者需要的 comm/topoInfo、漏检查四个调用之一、总体漂移掩盖各组真实状态 |
| spec 选路契约 | 把“不改变”误判为改变、批准变更缺范围/依据/矩阵仍接受、显式启用同时改变默认路径 |
| 编译命令重写 | 原对象/依赖文件被改写、源码 include 混用、原仓脏或 HEAD 不一致仍混用配置 |
| 地址与语义筛查 | 地址错误、非法操作数、无条件成本候选等既有反例未被拦截 |

`test_compile_probe.py` 用本机 g++ 真编译最小 C++ fixture，借助隔离树中不同的头文件常量验证 include 实际来源；缺少 g++/git 时会明确跳过。其结果不是 HCCL 骨架编译证据。

Plugin 行为测试会在临时 Git 仓中构造任务目录和源码 diff，通过公开的路由、状态流转、恢复链和产物门禁 CLI dry-run 正常链、Review/Test 修复链与环境阻塞。构建与测试证据是合成的，不会调用真实 HCCL/CANN 构建或 Agent 运行器。

## 真实源码验证

向子 skill 的 `selftest.sh` 传入明确的 HCCL 仓路径，以 `--strict` 防止缺环境被当成成功。默认隔离 worktree；可先克隆到临时目录，使原仓保持只读。编译数据库优先取 `build/compile_commands.json`，缺失才取仓根文件；复制到独立克隆时应只重定位路径，保留原编译参数及 CANN 配置，记录源 HEAD 和数据库来源。

该测试的默认编译仅覆盖两份骨架的 host 单文件编译；device 构建需要额外参数和环境，真实算法的双轨 Checker 仍按父 skill 流程执行。不得由离线回归或单文件编译推断完整构建/Checker PASS。

## OpenCode 调研复盘后的先验补充（2026-09-28）

- 新增按需 AllGather 接入卡和 16 个文件的独立来源组，覆盖注册、初始化、成本参数及选择器版本；
  历史 Ring 索引固定 commit/路径/哈希，5 份文件与本地 Git 对象匹配，不声称其算法验收通过。
- 调研交接传递已有契约、实际加载路径/基线哈希与未解问题，缺口解决即停止补查；同步 Architect 和调用模板。
- 真实 `978d01b6` 仓：接入卡与 Parallel 卡分组 MATCH，旧公共/单层组 CHANGED 按原样保留，HCOMM 接口组 MATCH。
  成本检查默认仍只查 algName；新选项检出 Sole 的 comm 空值及 Parallel 四处省略的 algName/comm/topoInfo。
  真实源码的内存副本验证：只补 algName 仍被多字段检查拒绝，补齐所需字段后通过；未修改原 HCCL/HCOMM。
- 离线回归：父 Skill 47 项、Template 29 项、Plugin 行为 64 项通过；父/子 quick_validate 及 107 条规则登记检查通过。
- Template 在既有 `/tmp` 隔离克隆中自测 13 PASS / 0 FAIL / 0 SKIP，两份骨架使用真实 CANN 参数通过 host 编译。
- 新增文本评测 14–16，涵盖委派范围、成本参数/选择器版本及历史报告纠偏；尚未运行独立会话评测。
- 未重新运行 OpenCode 完整开发任务或算法全量构建/Checker，尚无端到端耗时对照。

## AllGather Ring 执行复盘后的优化验证（2026-09-24）

- 新增 AllGather Parallel 四阶段契约和按需来源组，基准为 HCCL `978d01b68c477572962b5be67c4ea9929ca9e2d7`。
  旧的公共/单层来源基线保留；当前工作区变化按 CHANGED 报告，不通过刷新哈希掩盖差异。
- 新增成本参数传递检查：真实基准 executor 检出 4 处 algName 省略；用户修复后的工作区 4 处通过。
  只读源码，未修改 HCCL/HCOMM；未知初始化写法返回 UNVERIFIED，不能等价为接入通过。
- 离线回归：父 Skill 42 项、Template 29 项、Review 10 项、VM 33 项通过，共 114 项。
  覆盖省略/空值/字段顺序变化/未知语法、按需来源漂移、S03 正反例、批次失败停止和统计排除 dry-run/pending。
- Template 隔离克隆自测 13 PASS / 0 FAIL / 0 SKIP，两份骨架以真实 CANN 参数完成 host 编译；
  Review 只读真仓自测 52 PASS / 0 FAIL。检视自测中的两份旧 Spec fixture 补上数据流围栏，保留原缺陷断言。
- 使用用户本次真实 Ring Spec 的临时副本加回“没有 sync main -> sub / sync sub -> main”的说明，
  S00/S03 均无误报。四个 Skill 的 quick_validate、相对文档链接和 diff 空白检查通过；Plugin 行为回归 64 项通过。
- 新增文本评测 11–13（构建前闭环、多 loop 与统计、S03/未知成本写法）；评测集共 13 项，新增场景尚未运行独立会话评测。
- 未执行算法完整构建、device 验证、Checker 或端到端耗时对照，不据工具测试承诺节省分钟数。

## 自定义算法入口验证（2026-09-24）

- 新增 Ring、无归约/Barrier、来源漂移三个文本用例（8–10）；已写入评测集，尚未运行独立会话评测。
- 子 Skill 离线回归 28 项通过，含来源未提交修改/缺失/未提供/无关算子变化、Ring/Tree 属性推断，
  以及用最小 C++ fixture 编译运行未实现骨架，确认其失败且不改资源。107 条规则登记一致。
- HCCL `88a2a65cdd46a6b16091282fdeac81a391dcfd22`、HCOMM `ce7297c6f90013e2ba9b2f62d54151722819ab0d`：
  10 个算子的来源清单均 MATCH。单次 AllReduce 来源检查在本机耗时约 0.036 秒，仅代表文件校验耗时。
- 原 HCCL 仓只读；在 `/tmp` 独立克隆重定位现有编译数据库，保留原编译参数/CANN 配置，
  执行 `selftest.sh --strict`：13 PASS、0 FAIL、0 SKIP。Mesh one-shot 和新版 barebone 均通过 host 编译及静态检查。
- Plugin 行为回归 64 项通过；父/子 Skill 的 quick_validate、参考文件链接与 diff 空白检查通过。
- 未执行 device 全量构建、真实算法 Checker 或前后耗时对比；不能由这些结果宣称完整算法验收通过或调研已降至某个分钟数。

## 修复验证记录（2026-09-14，09-15 按新规则重验）

- 离线回归：子层 23 项通过；剔除 10 条 Agent/Plugin 流程与跨层验收重复规则后，106 条 template 能力规则登记一致。父层当前 33 项测试通过，见下节。
- 独立行为验证：运行一项“已授权 AllReduce 默认 selector 阈值变更，仅输出方案”的只读场景；正确继承授权并使用人工矩阵验收，未伪称执行成功。首次结果暴露 selector 回归范围在入口/参考中不一致，已同步为受影响范围，独立复测也按该范围制定计划。此记录不代表 `evals.json` 7 项已全部执行。
- 真实源码验证：以本地 HCCL HEAD `170ddeec539b4d693028ce6e0cf5c58933e4d46d` 的独立克隆运行 `selftest.sh --strict`，13 PASS、0 FAIL、0 SKIP；两份骨架均以现有 CANN 9.2 配置、C++17 和 `-Werror` 完成 host 编译。
- 父/子结构门禁无 error/warn，quick_validate 均通过。未运行 HCCL 正式全构建、device 构建或 hccl-vm Checker。

## 源码快照权限修复记录（2026-09-15）

- 新快照在 `tracked_modes` 中记录 Git 可见的已变更跟踪文件完整 Unix 权限，恢复后显式 `chmod`；旧 schema v1 快照无该可选字段时继续兼容。
- 父层原有 31 项测试全部通过；新增旧快照兼容和非法权限拒绝测试后共 33 项通过。
- 子层 23 项测试通过，106 条 template 能力规则登记一致。该结果只验证离线工具，不代表 HCCL 正式构建或 hccl-vm Checker 已执行。

## Plugin 端到端 dry-run 记录（2026-09-17）

- 在临时 Git 仓中验证正常链、Review 修复链、源码类 Test 修复链和环境阻塞，4/4 通过。
- 33 项路由/流转、17 项产物门禁和 4 项 E2E 核心工作流测试共 54/54 通过；加入 2 项旧 Skill 兼容测试后，`test-hccl-op-dev-workflow.sh` 总计 56/56 通过。
- dry-run 使用真实 Git 分支、HEAD 和 diff 哈希，但构建/测试证据为合成 fixture；未调用 HCCL/CANN 正式构建、Checker 或真实 Agent 运行器。

## 旧 Skill 路径清理记录（2026-09-24）

- 2026-09-24 曾删除旧版 `communication/hccl-aicpu-formal/` 兼容目录，AICPU 当时统一使用过渡名 `hccl-aicpu-formal-new`；2026-09-29 重构版曾迁回 `hccl-aicpu-formal`，此后五类能力迁为 `communication/` 下的平级 Skill，共享契约收敛于 `hccl-aicpu-best-practice/aicpu-shared-contract.md`。以上旧名称仅记录历史。
- 隔离 OpenCode 项目安装验证新父 Skill、template 子 Skill 和 template review Skill 的链接均指向预期源码；安装清单版本为 1.2.4，共登记 13 个技术 Skill。
- 该测试只验证安装和路由契约，不代表执行了真实 HCCL/CANN 构建、Checker 或 Agent 调度。

## 共享环境与 HCCL-VM Skill 边界记录（2026-09-17）

- `hccl-vm` 入口由 390 行全流程收敛为 124 行技术能力路由；环境就绪细节移入按需读取的 `references/environment.md`，不再下载安装、推进阶段或判断整个任务完成。
- `hccl-env-check` 只输出 `READY`/`NOT_READY` 和检查证据，不写任务状态或控制后续阶段；环境准备统一由 `hccl-env-setup` 提供并明确外部写入与授权边界。
- 新增 4 项边界行为测试；与原 56 项合并后，`test-hccl-op-dev-workflow.sh` 总计 60/60 通过。该结果不执行真实环境安装或 Checker。

## HCOMM 环境 Skill 清理记录（2026-09-24）

- `hccl-env-setup` 已覆盖 HCCL/HCOMM 仓准备、CANN Toolkit+950 ops 和统一环境文件生成；独立 `hcomm-env-setup` 没有运行时调用或配套脚本，因此删除。
- marketplace、Plugin skills、安装白名单和 quickstart 同步清理；隔离安装清单版本为 1.2.5，共登记 12 个技术 Skill。
