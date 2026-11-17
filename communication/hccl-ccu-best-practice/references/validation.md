# 验证、诊断与交付证据

## 0. 在冻结矩阵前核对运行条件

AllGather先按 [测试输入预检](test-input-preflight.md) 运行 `preview_cases.py --op all_gather`，计算实际rank/count/bytes及等效输入；再结合下方 [工具与内存约束](#工具与内存的已知约束)，把预检结果写进既有 TEST_PLAN/环境记录，不为每项生成新报告：

| 检查 | 执行方法与结论边界 |
|---|---|
| 实际库身份 | 解析将加载的 libhccl/libhcomm、VM/proxy 与兼容库路径，与可信包/构建清单比较哈希/构建元数据。源码 HEAD、库 mtime、符号存在不能单独证明 ABI 匹配；没有对应构建记录则标身份未确认 |
| 目标模式 | Verifier 在已有授权和可信旧包条件下，优先一个最小 CCU_SCHED/CCU_MS 冒烟再展开矩阵。旧包不可信时先按原流程建立可信基线；不要为了“预检”先构建候选或越过角色权限 |
| 工具可表达域 | 查看当前工具帮助与 hccl_test 参数解析/对齐源码，确认 size=0、非对齐、迭代/重放语义；已知不支持时一次取证并转不可达/受限项，不重复换参数试绕过 |
| 宿主容量 | 冻结 S 的含义、rank、放大系数、常驻开销/并发、现场可用内存与保留余量；算估计峰值再确定最大用例。AllGather 模型见接入卡，其他算子按其实际 buffer 模型 |
| 断言可观测性 | 每条 --expect-algo/命中日志/pattern明确产物、产生它的源码位置及本用例能到达该行的依据；API短路、selector旁路、缓存重放或工具判据都可能使产物缺失。不可观测则首版采用有依据的替代断言或明确UNVERIFIED；替代的API功能断言不能冒充目标kernel命中 |
| 分支可达 | 区分 count=0 的 API 提前返回、executor loopTimes=0 和 kernel 防御分支；`-n` 重复调用不自动等于“换地址 FastLaunch”；记录实际命中证据 |

六算子实际输入、root/reduceType与工具分配按 [按算子输入预检](test-input-preflight.md) 核对；不要把AllGather的两次ceil用于AllReduce/Broadcast/Reduce。非零root/非sum目前需要另核对定向runner及证据，不能声称当前HVM计划已覆盖。

2026-10-19 任务曾出现旧代际 libhcomm 在 CCU TP 建链崩溃、Notify index=5 超实际配额、GiB 级多 loop 用例压垮宿主、工具拒绝零数据。它们是优先排查入口，不是对当前所有环境的预设诊断。库不匹配时保存差异并按已有授权修复环境，不猜修算法；不要固定某个历史哈希为所有用户的正确版本。

既有可信基线与新候选的矩阵、工具/库身份、拓扑、dtype、size、默认选路契约都未变时，引用原始 run 和一致性检查结果；只补变化项，不重跑未受影响矩阵或重写整份历史。重设计改变语义/配置的部分仍需重新冻结与验证。完整构建和双轨门禁继续执行；已有合法编译缓存可复用，不在本 Skill 默认删 `--full` 或凭 device 哈希相同跳过构建。

### 工具与内存的已知约束

工具知识依据：2026-10-20 核对 `CANN_HOME/tools/hccl_test/common/src/hccl_test_common.cc`，SHA-256 `eb815fb31039c93669dce5fcb776be60de6ec661bcaf87f182e1e1b92c2834cb`，定位 `check_data_count`（本次源码325–379行）。该工具不在HCCL/HCOMM知识卡自动校验范围；现场先比较源码哈希，并用可信构建清单/已冻结二进制身份确认对应关系，不能仅以卡MATCH推定工具也相同。不同版本只补查解析/对齐函数。

完整输入变换、四文件来源和可执行预检见 [测试输入预检](test-input-preflight.md)；AllGather的count计算含两次向上取整，不能普遍用CLI整数除法代替。已核对的 `check_data_count` 拒绝 `-b/-e<=0`；对齐分支会调整 min/max bytes。`-s 1` 也不保证得到 count=0；“传了非对齐字节”不等于实际执行了对应 tail。冻结工具二进制/源码身份，记录实际 count/bytes。工具版本变化时只复核解析/对齐函数，不能为旧缺口反复试启动参数。

当前测试约定的 AllGather `-s` 是全 rank 总字节 S 时：每 rank 输入 S/N、输出 S；N 进程合计基本输入输出约 `(N+1)*S`，再叠加 Checker 复制/期望值、VM/QEMU 常驻与并发开销。按现场可用内存和保守放大系数预算，预留余量，并串行共享环境。**64 MiB 是历史任务的现场上限，不是所有机器的固定阈值**。普通 Sole 多 loop 需每 rank 跨越实际上界，不能为覆盖该分支直接启动数 GiB 的全局用例；不足时明确 UNVERIFIED，静态分析不冒充运行。

## 1. 静态验证

通常用 `check_contracts.py --launch <launch.json> --sync <sync.json> --hcomm-root <HCOMM根>` 一次执行三个原检查器；保留JSON中各项退出码、契约哈希与原始输出，报告引用汇总中的对应项。独立检查器可用于定位单项失败；不因一项失败跳过其他契约，也不把汇总成功视作功能或完整评审PASS。

- 运行来源定位脚本并保存目标源码身份；关注文件哈希能识别同一 HEAD 下的未提交修改。
- 运行启动参数、同步资源和生成器算术检查器，随后对照实际template/kernel/FastLaunch源码；带缓存头时声明cache_launch_offset，物理缓存更新索引和启动LoadArg索引分列。它不读取 C++，`consistent` 只表示 JSON 声明内部一致。
- 人工检查字节布局、归约贡献、资源所有权、channel→peer/die、跨端同步依赖和 CMake/selector 可达性。
- 明确支持与不支持范围；未改变默认选择也要检查新注册属性是否意外改变既有候选。

## 2. 构建

目标 HCCL 的 `AGENTS.md`、`docs/zh/build/build.md` 和 `build.sh --help` 是构建入口。当前仓提供 `bash build.sh --pkg`；在调用方已经授权的目标构建环境执行，保留完整日志、退出码、构建输入身份与实际产物路径。

CCU template/kernel 在已核验样本中编入 host `hccl` target。编译成功仍不证明实际动态加载了本次产物，也不证明 HCOMM 符号、资源及硬件功能匹配；记录 CANN/HCOMM 版本与实际加载库路径。修改 HCCL 算法不自动要求重编 HCOMM。

## 3. 功能与路由

测试计划包含目标拓扑和通信域、rank、dtype/reduce、数据量、尾块、跨 executor 分块、in-place（如支持）、零数据/单 rank（按算子语义）和 FastLaunch 重放。

优先做最小定向用例，确认**实际注册算法名与调用次数**，再扩大矩阵：

- 定向测试证明修改后的算法实际执行，不能用另一条 fallback 的正确结果代替。
- 保留可信旧版本基线，候选默认路径使用相同输入矩阵，检查选路与输出回归。
- FastLaunch 用更换 input/output buffer 的重复调用检查 stale address/派生 alias 字段；cache key 内变化应证明重新注册/失效。
- 超出支持矩阵的输入应走已声明的拒绝/回退行为，不把回退运行算新算法覆盖。

可用时使用 `HCOMM/test/hccl_vm` 的工具；当前公共 `hccl-test-tool-hvm` 支持 `CCU_SCHED` 和 `CCU_MS`。只有实际已安装该 Skill 时才按其 `run.sh --help` 使用参数和证据约定。它是可选工具，不是本 Skill 必须存在的依赖。未安装时依据 HCOMM 的测试文档/当前工具帮助选择入口，不猜测脚本路径，也不照搬 AICPU 的配置 DSL。

VM/Checker 的通过只覆盖其模型能力。真机多 rank 集成、版本兼容和性能结论必须来自相应环境。没有设备或工具时，交付静态/构建结果并具体列出未执行测试。

### 单 rank 定向断言

当前 AllGather 的 `all_gather.cc::AllGatherOutPlaceCommon` 在 userRankSize==1 时执行SingleRankProc并在Selector之前返回（本次核对213–217行）；正常首次N=1请求到不了 `selector_engine.cc` 的 “the selected algo type is” 产物。不能给此行统一套 --expect-algo 后把结构性缺日志判成算法SOURCE失败。

首版计划逐行注明可达性：N=1可有独立的API短路/功能证据，但它不证明新CCU kernel的单rank分支；确实需要该覆盖时另选可达入口，否则明确受限/UNVERIFIED及源码检视依据。零数据、缓存路径和fallback同样核对断言产生点。避免把一个全局批次句误套到所有特殊用例。

## 4. 性能估算与实测

调用方要求 estimate（如 hccl-op-generator 候选验证）时，在 A/B 功能验证全部通过后复用同一候选包、HCCL/HCOMM 身份、冻结 CCU_SCHED/CCU_MS 模式与选路配置，按已安装 `hccl-test-tool-hvm/references/estimate.md` 串行启动独立会话。该参考是命令、固定参数、环境和证据格式的真源；不复制执行脚本，不通过功能 run.sh/plan.py 或 Checker 汇总器执行/验收 estimate。没有该 Skill 或当前工具不支持时返回具体缺项，由调用方处理完成状态。

从本次日志重新核对目标算法命中。共享 estimate.json 额外记录 `engine=CCU`、`ccu_mode` 和 `hcomm_source_identity`；`source_identity` 保持 HCCL 单仓身份。保留真实命令/退出码、候选包、集群与日志/profiling/trace 哈希，在调用方指定报告中呈现组网、dtype/数据量、makespan、观察与限制。失败、未执行、未命中或缺证据时保留独立功能结果并报告阻塞原因，不用 fallback 结果或示例耗时补成功，也不因估算值较高自动修改源码。

estimate 是所记录条件下的仿真估算，不证明真机延迟/带宽或性能收益。若任务要求性能改善，再使用下述同条件实测与可信基线作比较。

固定源版本、动态库、硬件、拓扑、rank、dtype/count、warmup 与重复次数，记录候选和基线的同口径延迟/带宽及离散情况。优化一次改变一个有依据的瓶颈因素，例如流水深度、channel 并发、分片或等待位置；不得用 VM 逻辑正确或理论分析宣布性能收益。

## 5. 按证据诊断

| 症状 | 优先检查 |
|---|---|
| 注册失败/资源不足 | channel die 一致性、resGroup 并发假设、XN/CKE/片上资源、版本支持 |
| 首次运行正确、重放错误 | cachedArgs 索引、地址基址偏移、派生字段更新、cache key 与 token 生命周期 |
| 挂起 | 实际选路、等待位是否有 producer、条件分支、零片早退、跨轮信号复用 |
| 特定尾片/rank 错 | userRank/局部索引/channel 映射、tail/repeat 和 executor 分块 |
| 归约值错误 | 输出初值、每片贡献次数、dtype/op、并发写和同步 |
| 新算法未命中 | 注册名、属性、selector/环境解析、编译守卫和实际加载库 |
| 编译过但运行缺符号 | HCCL dlsym 符号、实际 HCOMM 动态库、版本守卫 |

测试失败时保留原始证据，区分源码缺陷与环境不匹配。环境不变、证据不变时不盲目重试；下一次实验应有明确变化和区分标准。不要扩大为安装/覆盖共享 CANN/HCOMM，除非任务已有相应授权。
