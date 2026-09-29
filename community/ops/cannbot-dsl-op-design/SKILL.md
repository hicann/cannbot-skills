---
name: cannbot-dsl-op-design
description: "根据算子 spec.yaml 为 CANNBotDSL 算子编写分文件设计文档：数学语义、公开 Python 接口、Shape/DType、执行路径、Launch 配置、Host 调度、DSL Kernel、API 与内存以及测试方案。当用户要求设计 DSL 算子、从 spec 形成方案或修订这些设计文件时使用。"
---

# CANNBotDSL 算子设计

## 输入

- 必需：算子的 `spec.yaml`。数学、接口、Shape/DType/Layout、边界、精度与确定性以它为依据。
- 按需：调用方提供的导出接口与生命周期约束，用于明确导出入口、参数、产物和资源有效期。
- 按需：用户需求、目标设备和 CANNBotDSL 能力资料、已有算子代码、性能数据与测试。缺少依据时在相关设计位置写明待确认事项，不编造 API 或资源数字。
- DSL 入口使用普通 Python 调用 `cannbotdsl.compile(@host 入口, specs...)` 获得程序，Kernel launch 直接写在 `@host` 中；`@jit` 用于 DSL 内部辅助计算，VF 使用 `with vf(...)`。

## 输出

在调用方指定的设计输出目录直接生成以下主题文件。主题文件使用 `templates/design/` 下的对应模板；另按 `templates/INDEX.md.templ` 生成独立的 `INDEX.md`，用章节和相对链接组织实际产物：

| 内容 | 文件 |
|---|---|
| 语义、接口、推导、端到端测试方案、开发视图 | `Overview.md`、`Interface.md`、`InferShapeDtype.md`、`Validation.md`、`DevView.md` |
| 静态配置和执行路径 | `TemplatePartition.md` |
| Launch 参数、metadata、资源计划 | `LaunchData.md` |
| Host 配置、任务划分、Launch 顺序 | `HostLaunch.md` |
| DSL Kernel 阶段、计算与同步 | `Kernel.md` |
| DSL API 与内存管理 | `API.md` |
| 每条真实执行路径的详细设计 | `branches/DESIGN-BRANCH-<id>.md` |

分支文件从 `templates/DESIGN-BRANCH.md.templ` 生成。即使只有一条执行路径，也生成对应的分支文件。分支 ID 只是文档中的引用名称，不要求实现中存在同名变量或硬件键。

## 编写方法

1. 在 Overview、Interface、InferShapeDtype 中承接 spec 的公式、对象、组合和边界，不重新定义另一套语义。
2. 从公式选择算法和设备计算阶段。在 TemplatePartition 中区分静态配置、编译期特化、运行时参数与真正不同的执行路径；仅因参数值不同不必拆分路径。
3. 在 LaunchData 中说明配置、参数、必要的 metadata 和资源。在 HostLaunch 中写输入检查、输出与 Workspace 分配、任务划分、布局视图和启动顺序。Host 可处理元数据、分配和已确认的零拷贝视图；输入相关的设备值计算由 DSL Kernel 承担。
4. 在 Kernel 中说明角色、阶段、坐标、搬运、计算、写回与同步。是否需要 Cube/Vector 分工、多阶段或双缓冲由本算子决定。分支文件写各路径的具体切分、资源、计算链和边界处理。
5. 在 API 中记录实际选用的 DSL 调用、来源和适用前提，在 Validation 中写独立 Golden、用例覆盖、精度及有依据的性能目标。

6. 正文完成后编排 `INDEX.md`：按语义与接口、路径与分支、执行与资源、验证与开发的阅读顺序分组，分支链接紧跟路径划分；目录树反映实际文件布局，链接标题取自对应文档。索引只写文件职责和导航，上游资料需要收录时单列参考资料，不重复设计结论。
7. 各主题和分支文件保留一个一级标题，内部小节逐级展开。调用方要求汇总 `DESIGN.md` 时，以索引中的设计章节和链接顺序组织正文：总标题为一级、章节为二级、来源文档标题为三级，内部小节相应降级，代码块原样保留。`INDEX.md` 独立存放，不作为正文来源嵌入 `DESIGN.md`。

模板中的 DSL 伪代码用于展示对象职责、数据所有权和填写位置。选择适用的骨架后，用本算子的公式、合法 API、参数和容量证据逐项替换；未使用的路径直接删去。伪代码本身不是可编译性或设备行为的证据。

各文件通过内容和链接互相衔接。分支判定与场景覆盖写入 `TemplatePartition.md`。

## 完成标准

- 每个受支持的 dtype 组合、输出和执行路径都有可追溯的公式、参数、资源依据与验证用例；不适用项说明原因。
- Host 的配置选择、Launch 参数和 Kernel 签名一致；tile/任务范围、分支覆盖、内存峰值与同步条件在各文件间一致。
- 实际采用的 DSL API 有来源和适用边界。缺少编译或设备证据的调用明确标为待验证，并写出补证动作。
- `INDEX.md` 的目录树、章节顺序和链接与实际产物一致，分支文件与 `TemplatePartition.md` 一一对应；汇总正文遵循同一层级且不包含索引正文。
