# 复杂模拟交付件验证矩阵

记录日期：2026-10-22。以下算子名称、dtype、用例数和芯片仅标识测试证据，
不是迁移输入的默认值或支持列表。实际迁移仍从当前源交付件提取完整契约。

## 已执行范围

环境为 x86_64、CANN 9.3.0 weekly.20260923.01、真实 Ascend950PR_9579。
源包与目标包分别安装在独立 CANN_Master / CANN_Revise，使用不同 vendor 和注册名称，
防止命中内置参考算子。TTK 固定为 `80ec38d7622582331d409049dec22706ad72aef6`，每次显式单进程；构建并行度不超过 8。

| 目标仓 / 参考算子 | 迁移前 kernel / ACLNN | 迁移后 kernel / ACLNN | 目标 UT | 状态 |
| --- | --- | --- | --- | --- |
| ops-nn / RmsNorm | 65/65、65/65 | 65/65、65/65 | host 27/27 | 前轮完整通过，本轮复用证据 |
| ops-transformer / RotaryPositionEmbedding | 282/282、294/294 | 282/282、294/294 | host 24/24、API 8/8 | 完整通过 |
| ops-cv / GridSampler2D | 72/72、72/72 | 72/72、72/72 | host 16/16 | 完整通过，中断批次已完整复跑 |
| ops-math / Cumsum | 96/96、120/120 | 96/96、120/120 | host 3/3、API 24/24 | 完整通过 |

四仓均实际构建、正常安装；原生 ACLNN / GEIR 在迁移前后分别执行并完整清理，
退出码均为 0。合计 2132 次数值用例执行、102 个目标 UT、16 次原生程序执行通过。
2132 是多通路、迁移前后的执行次数，不是独立用例数。
暂停后已按用户恢复指示完整重跑 CV 最后一批，并补齐 math 的源包、目标包验收；
原中断及源准备失败记录保留，不计入通过数。

参考仓基点：

- ops-math：`16700ac8d829f8e66180c612f45ed4384b51e2e0`
- ops-nn：`82889267c9eee08660fc28a5e9bb98196ea0248f`
- ops-cv：`47d74481f41824a56dacda49368c03b915bdcae1`
- ops-transformer：`d0a83f16fbdc70e821ceaeec328819104b196f9d`

## 覆盖差异

- nn：归约、双输出、epsilon、五种输入 dtype 配对、自动生成 ACLNN、两个实际 tiling 分支；保留全部 45 条参考 ST 并补充 20 条。
- transformer：手写 ACLNN V1/V2、可选 Tensor 的 NULL 通路、三种 dtype、四种旋转模式、广播与三维输入；保留全部 282 条参考 kernel ST，并补充 12 条 V2 API 测例。实际命中五个 tiling key，不宣称所有分支都覆盖。
- cv：NCHW 输入和 ND grid、三种插值、三种 padding、align_corners 两值、两种 dtype 和两组形状；验证格式索引、SIMT kernel 与 GEIR 推导注册。8 条原推导 UT 保留，另外增加 8 条 tiling 正反例。
- math：张量常量 axis、正负轴、exclusive/reverse、多维扫描、四种测试 dtype、两种 axis dtype、手写 V1/V2 API；源/目标实际命中四个相同 tiling key，16 个编译 dtype 条目保持一致。V1 的 24 条与 V2 的 96 条数值回归均通过。

源交付件由参考源码组装，源基线准备与迁移分开记录。源包缺少独立构建依赖、
推导注册或测试参数映射时，先修正并重测源基线，再冻结；不能把这类调整说成迁移过程未改语义的证据。
另一次 TopKV2 源基线试验失败并归档，没有计入迁移成功。

## 通用检查与边界

已核对完成样例的冻结输入、golden、实现、API 签名、二进制配置、实际 vendor 和共享库依赖。
transformer 的 API UT 只移动目录并调整 include；cv 增加测试，保留原断言。
源包与目标包不要求二进制哈希相同；分别核对安装内容与各自打包 payload 相同。
math 的 46 个实现/golden 文件与冻结源逐字节相同；自定义 API 库保留全部 4 个公开
ACLNN 符号，内部强全局对象导出为 0，原生程序完整清理后正常退出。

本轮证据存于任务工作区 `validation/complex-four-repo/`，nn 前轮证据在
`validation/complex-migration/`。产品 PR 不包含这些测试算子的实现。
2026-10-22 的产品通用工具回归为 37 项，包含 64 个名称 / 仓库 / 接口模式 / soc 组合及 CPU 动态库退出正反例。

2026-10-29 评审修复后，完整工具回归为 **56/56 通过、0 跳过**：补齐 Git 状态失败阻断、
显式 UT 父子目录接入、CMake 块注释与未闭合注释、许可证扩展名、nn 310P UT 目录和
math example 递归发现的正反例。四仓真实 CMake 收源的 8 项 A2/A5 探针与额外 1 项
nn 310P 探针均通过。该轮为 CPU 工具及规则验证，未重跑上述 NPU/TTK 历史用例。

这些是模拟交付件的工程迁移回放，没有运行 Cannbot 实际全链调度。
未覆盖其他芯片、全部 dtype/tiling 分支、完整 GEIR CSV、torch wrapper、
完整内存 sanitizer 或性能验收。不能据此保证任意未知算子无条件成功。
