> 平台固定为 `Ascend950`。本工作流定义该平台的算子设计、生成、测试、性能调优与代码检视流程，使用对应分支的技能、角色和资源。

# TileLang 算子开发工作流

源码位置与产物目录遵循 [Ascend950 路径约定](../../references/source-layout.md)。本阶段沿用并向后续技能/角色传递 `{ops_tilelang_repo}=OPS_TILELANG_DIR` 和 `{tilelang_repo}=TILELANG_DIR`；两者由插件源码准备阶段确定，不由用户另行指定。


本工作流只针对 TileLang 算子在 Ascend950 NPU 上进行算子设计、生成、ST 测试、性能调优与最终代码 review 的完整工作流，不涉及 CUDA、Metal 等其他后端的算子开发。

默认顺序为：方案设计 → 算子生成 → 指定用例精度通过 → ST 测试 → 询问是否调优 → 用户确认后进入性能调优 → 调优完成后询问是否进行代码 review → 用户确认后 review 最终交付代码。

跨 Skill 调用统一按名称加载，例如“先加载名为 `tilelang-op-design` 的 Skill，并传入平台 `Ascend950`”。由已安装的 Skill 入口选择本平台正文；加载失败时停止当前阶段并报告。子代理按注册角色名调度，缺失时停止并报告。工作流、参考和步骤文件按文档中的相对路径读取；脚本以所属工作流或 Skill 的真实目录定位，产物以用户项目目录或明确指定的输出目录定位。

设计完成后直接继续生成。必要信息不足时暂停并引导补充。泛化用例扩展不在设计和生成阶段执行，由 ST 测试阶段完成。

## 后端选择约束

本工作流各阶段创建或修改的算子、测试、profiling 及辅助 Python 代码均不得通过 `target="..."`、`target='...'` 或其他等价方式固定 TileLang 后端；JIT 装饰器、builder 和编译 API 均省略 `target` 参数，Python 代码不得设置后端环境变量。后端统一由启动命令通过 `TILELANG_DEFAULT_TARGET` 选择：PTO 使用 `TILELANG_DEFAULT_TARGET=pto`，AscendC 使用 `TILELANG_DEFAULT_TARGET=ascend`。切换后端时不得修改源码。

## 源码准备

进入本分支、开始方案设计前，运行 [源码准备脚本](scripts/prepare_framework.sh)。两个仓库固定放在插件目录下，不接受路径参数或环境变量覆盖：

| 仓库 | 源码地址 | 插件内目录 |
|---|---|---|
| ops-tilelang | `https://gitcode.com/cann/ops-tilelang.git` | `repositories/Ascend950/ops-tilelang/` |
| tilelang | `https://github.com/tile-ai/tilelang.git` | `repositories/Ascend950/tilelang/` |

先解析本 workflow.md 的真实目录为 `WORKFLOW_DIR`，调用：

```bash
bash "$WORKFLOW_DIR/scripts/prepare_framework.sh"
```

脚本依次以 `--depth=1 --no-recurse-submodules` 浅克隆上述两个主仓库，用于查阅算子、API、lowering 和框架实现，不默认下载子模块。每个仓库独立检查，有效仓库已存在则跳过，不自动 pull 或重置修改，也不删除已下载的子模块。克隆失败停止本次准备并报告错误；不将目录存在视为克隆成功。两个主仓库均就绪后输出 `SOURCE_READY`；该标记仅表示参考源码就绪，不表示子模块、编译依赖或运行环境完整。

后续查证确实需要某个子模块内部实现时，先核对该仓库的 `.gitmodules`，再按父仓库记录的提交按需获取对应路径。例如，需要 TVM 内部源码时执行：

```bash
git -C "$TILELANG_DIR" submodule update --init --depth=1 -- 3rdparty/tvm
```

不使用 `--recursive` 批量下载嵌套依赖，也不使用 `--remote` 改变参考版本；需要嵌套子模块时，在其父仓库中按同样方式单独获取。获取失败时报告具体缺口，不把缺失源码当作已查证依据。

### 首次准备进度

执行前说明正在准备 Ascend950 源码。脚本显示 Git 真实进度，每 15 秒报告当前阶段和耗时，启动时打印 `.preparation/Ascend950/` 下的日志位置。使用持续运行会话，约每 15～30 秒读取输出并向用户反馈；宿主隐藏输出时读取日志文件。不要在下载期间重复启动脚本或编造百分比。

记录输出的 `OPS_TILELANG_DIR`（算子源码）和 `TILELANG_DIR`（TileLang 框架源码）并传给后续技能；核对实际目录结构后定位实现与测试。脚本仅下载源码。实际运行使用的 TileLang 编译器路径仍按 Python 导入结果确定，不能将下载的源码视为已安装的编译器。

用户已有运行环境时按本分支技能核实后继续。缺少编译环境时明确报告缺口；可以进行不依赖运行环境的设计与静态分析，依赖编译运行的步骤保持未验证，不声称已完成环境安装。

## 方案设计

先加载名为 `tilelang-op-design` 的 Skill，并传入平台 `Ascend950`，按其模板生成 `design.md` 并完成自检。已有报告复用时同样自检。必要输入不足时先按该 Skill 完成澄清，不生成待补充版报告。

设计完成后，工作流从当前上下文或 `design.md` 取得“目标后端”；两者均有记录时核对一致，不一致则返回设计阶段澄清。后续调用算子生成、ST 测试和性能调优时传入该目标后端，PTO 对应 `TILELANG_DEFAULT_TARGET=pto`，AscendC 对应 `TILELANG_DEFAULT_TARGET=ascend`。

相对资源路径以本文件真实目录解析；产物目录以用户启动会话的项目目录为基准。

## 算子生成

先加载名为 `tilelang-op-develop` 的 Skill，并传入平台 `Ascend950` 和目标后端，实现并调试指定用例，仅交付 `<op_name>.py` 与 `test_<op_name>.py`。设计和生成默认产物目录为用户启动会话的项目目录下的 `operators/<op_name>/`；用户指定目录或已有算子目录优先。

生成阶段以全部指定用例实际精度通过为完成条件。完成后给出两个文件路径和精度结果，随后直接进入 ST 测试。

## ST 测试

先加载名为 `tilelang-op-test-design` 的 Skill，并传入平台 `Ascend950` 和目标后端，审查算子契约和已有 `test_<op_name>.py`，补齐适用的功能、精度、边界、梯度、状态修改、异常拒绝、布局与接口、后端与分支、随机性覆盖，并保留可复现的执行证据。ST 发现算子缺陷时返回生成阶段修复，随后重跑相关测试。

ST 完成后给出 pytest 测试路径、可信覆盖结论和实际执行结果，再询问：“算子已生成且 ST 验证完成，是否继续进行性能调优？”随后暂停，等待明确答复；此前对完整流程的授权不替代本次确认。用户确认后，携带源码目录、算子内核、测试文件、后端、用户原始用例及 ST 结果进入性能调优；用户拒绝则结束，未答复则保持暂停，不加载调优工作流或执行性能采集。

## 性能调优

以下入口规则仅在进入调优阶段后生效。进入本阶段即自动作为 `tilelang-tuning` 主 agent 工作；不得要求用户再次显式调用 custom agent、指定 agent 名称或提供工作流 Markdown 路径。

在回复用户或执行任何调优动作前，完整读取 [tilelang-tuning 工作流正文](agents/tilelang-tuning.md)。各宿主的角色配置只用于发现和调度，业务正文以本分支文件为准。继续按正文中的相对路径读取所引用的 workflow 文件，不得压缩、改写或跳过工作流规则。

工作流中的“用户当前工作目录”和 `{cwd}` 均指用户启动会话的项目目录。所有调优产物因此统一写入该目录下的 `operators/`，不写入本 workflow.md 所在目录。

如果正文或角色配置缺失，停止进入调优步骤，报告安装不完整，并提示重新运行插件 `init.sh` 安装当前宿主工具。

性能调优全部完成并确定最终代码版本后，询问：“性能调优已完成，是否对最终交付代码进行 code review？”随后暂停，等待用户明确答复；此前对完整流程或性能调优的授权不替代本次确认。用户拒绝则结束，未答复则保持暂停，不执行 review。

## 代码 Review

最终交付代码为最终选定的 `<op_name>.py`、`test_<op_name>.py`，以及明确列为交付物的其他测试或运行必需辅助代码；调优后以 `final_optimized/` 中的版本为准。profiling、性能对比、调试脚本和废弃候选不是交付文件，review 只检查最终交付代码。

仅在用户明确同意后执行，且这是本工作流的最后一步。先加载名为 `tilelang-review` 的 Skill，并传入平台 `Ascend950`，完整遵循该 Skill 规定的环境检测、格式与条例检视、报告生成、修复确认和结束检查流程，不得以自行设计的 review 流程替代。该 Skill 要求的“明确输入”由本工作流汇总出的 review 文件清单提供，无需再次要求用户提供或粘贴代码。开始调用 Skill 前先向用户列出实际 review 文件清单，并将同一清单作为检查脚本的显式输入。

## Codex NPU 设备命令

在 Codex 中，任何会访问 Ascend NPU 或检查 NPU 设备节点的命令都必须直接以以下前缀启动：

```bash
env -u ASCEND_RT_VISIBLE_DEVICES
```

这包括但不限于 `pytest`、精度验证脚本、预运行、`msprof`、`npu-smi` 和 `asys`。不得先在普通沙箱中试运行再重试。`env -u ASCEND_RT_VISIBLE_DEVICES` 本身只负责清除逻辑设备映射，不会授予沙箱外权限；Codex 调用工具时还必须直接设置 `sandbox_permissions=require_escalated`。若当前机器尚无匹配的持久批准规则，首次执行应直接发起权限提升请求，获批后运行；存在匹配规则时直接复用。需要指定设备时，由命令自身使用物理 device id（例如 `--device 0`）。

正确性测试默认按实际可用 NPU 数量设置并发；只确认一个物理设备时使用单 worker，避免多个进程同时初始化同一设备。不得通过提前导入 TileLang 的 `sitecustomize.py` 修改 pytest 环境，以免污染 xdist 的启动握手。
