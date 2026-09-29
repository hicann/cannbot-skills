---
type: "workflow"
title: "开发与验证"
description: "环境、构建、精度定位、性能采集和迭代。"
tags: ["catlass-cpp", "workflow", "develop", "test", "pr1069"]
status: "stable"
generated: {"by": "process:catlass-cpp-pr1069-extraction", "at": "2026-09-18T00:00:00Z"}
verified: [{"by": "process:architecture-separation-source-check", "at": "2026-09-20T00:00:00Z"}]
sources: [{"id": "pr1069-development-and-validation", "resource": "git:cann/cannbot-skills@4950cbd7c45e0d44cb8ec52598edf53cdc7f16ca:ops/catlass-linear-attention-workflow/references/development-and-validation.md", "title": "PR1069 fixed source for Linear Attention 开发与验证", "kind": "repository"}, {"id": "catlass-arch-separation", "resource": "git:cann/catlass@0d78a73c192fc04741a4fd57c94080209424775c", "title": "CATLASS fixed source for AtlasA2 and Ascend950 architecture separation", "kind": "repository"}]
consumers: ["catlass-cpp-develop", "catlass-cpp-test"]
---
# 接口与概念

本 concept 是 PR1069 固定来源的结构化入口；规范性技术正文完整保留在“PR1069 原始正文”。

## 算子算法

算法定义、公式、计算顺序和边界语义以原始正文及当前任务冻结的 operator/golden contract 为准。

## 分核策略与基本块切分

分核、任务组、基本块和尾块规则以原始正文为准，并需针对当前 shape、SoC 和资源重新推导。

## 数据路径与存储层级

GM、workspace、L1、UB、L0 的地址、生命周期和复用要求以原始正文为设计输入。

## 流水排布、同步关系与数值精度

Stage、slot、CrossCore/HardEvent、累加和转换规则以原始正文为准，实际结论必须通过验证。

# 用法

先通过 compact query 定位本 concept，再完整读取；不得把知识内容当作当前工程已验收的证据。

# 代码模式

原始正文中的伪代码和命令保持原义。具体 API 必须核对目标版本 CATLASS/CANN 源码。

# 约束

不得删减或放宽原始规则、阈值、失败条件、状态恢复或验收要求。

# 失败表现

若设计、代码或测试与原始约束不一致，按五阶段 workflow 的 issue_type/resume_from 矩阵恢复。

# 验证方法

对照 frontmatter `sources` 中的固定 PR SHA 和原文件逐段校验；阶段通过仍需实际构建、精度和性能证据。

# PR1069 原始正文

# 开发、精度定位与性能验证

本参考将逐 Stage 开发、首个差异定位和性能迭代方法落到 CANNBot 直调工程。
阶段推进及恢复状态按本 Skill 的 `SKILL.md`，代码写法参考
`kernel-stage-sync-patterns.md`，精度入口和判定标准按 `precision-policy.md`。
以下目录占位符沿用 `SKILL.md` 的定义；命令在实际开发用的 Linux 环境执行。

## 环境与平台核验

读取目标环境实际安装的 CANN 环境脚本，在当前 shell 加载后核验工具：

```bash
source "{cann_set_env}"
command -v bisheng cmake python npu-smi
cmake --version
python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
npu-smi info
```

`{cann_set_env}` 填写本次使用的 CANN 环境脚本完整路径。根据安装包版本信息、头文件、库和
目标平台示例确认 CANN、编译器及 CATLASS 版本配套；确认设备健康、实际 SoC 和本次使用的设备。
性能采集前另行核验 `msprof --help` 及所需参数。

从该版本 CANN 的平台查询接口及对应 SoC 资料核对 AIC/AIV 核数、L1/L0/UB 可用容量和目标
指令的 dtype 支持。需要查询实际设备能力时，参考安装包中的平台接口头文件和示例编写最小
host 查询程序；CATLASS 示例中的 `GetCoreNumAic()` 等用法需与当前安装版本匹配。
将查得的版本、设备能力、依据和检查结论写入 `docs/precheck.md`，供任务切分、容量计算和编译配置使用。
缺少依赖或能力信息时补齐后复查。

架构预检必须显式记录并锁定以下分支，不能仅写“A2/A3/A5 均支持”：[^catlass-arch-separation]

| 检查项 | Atlas A2/A3 | A5/Ascend950 |
|---|---|---|
| CATLASS 编译标识 | `CATLASS_ARCH=2201`、`Arch::AtlasA2` | `CATLASS_ARCH=3510`、`Arch::Ascend950` |
| Vector 模型 | MemBase；核对 `TPipe`/`TQue`/`TBuf`、pass 和 EventID | RegBase/VF 为可选路径；核对 VF、寄存器压力、mask 和编译结果 |
| Cube→Vector | 预期 `L0C -> GM -> UB` | 组件支持时可 `L0C -> AIV UB`；否则使用 GM 基线 |
| Vector→Cube | 预期 `UB -> GM workspace -> L1` | 组件支持时可评估 `AIV UB -> AIC L1`；保留 GM 基线 |
| 资源基线 | UB 192 KiB、L0C 128 KiB、FixBuf 7 KiB | UB 248 KiB、L0C 256 KiB、FixBuf 16 KiB |

容量值只是 CATLASS `Arch` 上限的版本化输入；`docs/precheck.md` 仍需扣除组件、编译器和当前 kernel
实际预留。若目标版本缺少对应 specialization，设计必须回退到该架构可用的数据路径，不能借用另一
架构的 API、类型或资源预算。

## 构建并运行直调工程

初始化脚本提供目录和工作流模板。开发阶段在当前算子工程中生成 `CMakeLists.txt`、`run.sh`、
`op_host/`、`op_kernel/` 及 `test/` 中的测试入口。直接查阅工作区 `catlass/README.md`、
`catlass/include/`、`catlass/examples/` 和当前 CANN 安装包的示例，确认所用 API 的签名、
编译语言、架构选项、头文件和链接库；把实际使用的构建配置写入当前工程。
源码查询用于确认当前设计需要的具体用法，函数组织、调度和同步由当前 `docs/design.md` 决定。

host `main()` 按接口读取输入和用例参数，完成 ACL 初始化、设备与 stream 创建、tiling、
device 内存及 workspace 分配、输入拷贝和 `kernel<<<blockDim, nullptr, stream>>>(...)` 启动。
kernel 使用传入的 device 指针及详设中的地址偏移；host 等待 stream 完成后拷回全部输出，
按约定的 shape、原始 dtype 和字节数交给测试程序，并释放本次调用的资源。逐项检查 ACL 返回值，
使运行失败能够被测试入口识别。

确定当前 `CMakeLists.txt` 的编译目标及架构配置后执行：

```bash
cmake -S "{operator_dir}" -B "{operator_dir}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "{operator_dir}/build" --target "{build_target}" --parallel "{jobs}"
bash "{operator_dir}/run.sh" {case_arguments}
```

`{build_target}` 是本工程定义的可执行目标，`{jobs}` 是本次构建并行数，`{case_arguments}`
按本工程 `run.sh` 已实现的参数填写，包含本轮用例和设备选择。CANN/CATLASS 的额外 CMake
选项按本工程及实际安装版本补充。将可复现的构建、运行命令和参数说明写入算子 README；
每轮重新启动测试进程，核对执行文件和代码版本对应本轮构建结果。

每个 Stage 按“实现、构建、核对设计、运行、比较、记录”的顺序完成，再接入后续 Stage。
后一个 chunk 依赖前一个 chunk 的状态时，按设计保留状态传播顺序，验证初始状态、多 chunk
连续处理及尾 chunk；各 chunk 输入在启动前已经完备时，按设计独立分配任务并验证任务间数据隔离。
混合场景沿状态依赖组织整体调度，其中独立计算按设计并行执行。
采用本地示例中的同步写法时，将生产者完成写入、消费者读取和物理槽位释放对应到当前数据依赖；
同一核连续处理多个 task、head 或 chunk 时，覆盖首次使用、切换、回绕和尾任务。
需要确认实际执行顺序时，测试版本可使用独立且按核隔离的记录区，记录任务、slot 和事件步骤；
正式精度及性能验证使用关闭该记录的版本。

## 精度定位

固定失败用例的输入、随机种子、代码版本、有效区域和工程精度策略，以 CPU PyTorch 标杆
生成对应预期结果。逐 Stage 及整算子输出使用 `precision-policy.md` 中的 `compare_case()`
或其命令行入口；比较配置缺失时先修正测试信息并重跑。

整算子失败时，沿设计的数据依赖比较已有 Stage 中间结果，找到首次出现差异的 Stage，再缩小到
该 Stage 的 head、chunk、tail 或边界分区。结合失败位置、数值分布和报告中的指标，依次核对
公式与累加顺序、dtype 转换、地址与布局、搬运有效区、同步通知、slot 复用及输出写回。
需要更多中间结果时，在测试版本加入对应检查点，并保持原有生产消费关系。

将复现用例、首个差异位置、根因和代码位置写入 `docs/validation.md`。实现错误按
`precision_debug` 分支修复；发现设计、标杆或接口问题时，按 `SKILL.md` 的恢复矩阵回到
相应阶段。定向精度修复执行失败用例、受影响 Stage 和最小边界精度用例；通过后进入全量验收。

## 性能采集与迭代

使用已通过精度的构建版本，在约定的模型用例上固定输入、SoC、设备、预热、采样次数和统计方法。
测试程序实现预热与重复调用，并使输出记录能区分预热和正式样本；基线与候选方案使用相同条件。
确认当前 `msprof --help` 支持相应参数后，可直接采集已构建的 host 程序：

```bash
msprof --output="{profile_dir}" --task-time=on "{executable}" {run_arguments}
```

`{profile_dir}` 是本轮新建的采集目录，`{executable}` 是本轮 host 可执行文件，
`{run_arguments}` 使用其已实现的用例、设备、预热和采样参数。读取本轮生成的 `op_summary`，
按实际 kernel 名称和调用记录筛出正式样本，以 `Task Duration(us)` 对照设计中的性能目标。
涉及多个 kernel 时按设计约定的计算范围逐项统计。保留原始数据和样本选择依据，记录波动与统计值。

需要定位瓶颈时，按工具版本增加 AIC/AIV、MTE、带宽和流水采集：Scalar 看热路径逐元素访问，
MTE 看搬运粒度与重复搬运，VEC/CUBE 看计算分工、tile 和有效计算占比，AIC/AIV 等待看生产
消费进度、写回完成和缓冲区复用。同步位置由实际依赖决定；逐 head 通知、成组执行和流水距离
分别结合当前数据到达与消费情况分析。调度策略、tile、驻留和双缓冲以目标平台实测选择。

每轮只调整一个主要变量，从最早受影响 Stage 复验到整算子，精度通过后比较性能。
将改动、瓶颈证据及优化前后结果写入 `docs/validation.md`；方案变化按恢复矩阵更新设计后继续。
最终覆盖范围、性能达标结论和归档要求统一按 `SKILL.md` 的全量验收规则执行。

[^catlass-arch-separation]: git:cann/catlass@0d78a73c192fc04741a4fd57c94080209424775c; paths: README.md, include/catlass/arch/arch.hpp, include/catlass/gemm/tile/copy_l0c_to_gm.hpp, include/catlass/gemm/tile/copy_l0c_to_ub.hpp, include/catlass/epilogue/tile/copy_ub_to_l1_tla.hpp

[^pr1069-development-and-validation]: git:cann/cannbot-skills@4950cbd7c45e0d44cb8ec52598edf53cdc7f16ca:ops/catlass-linear-attention-workflow/references/development-and-validation.md
