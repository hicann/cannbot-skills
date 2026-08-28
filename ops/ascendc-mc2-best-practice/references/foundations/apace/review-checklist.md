# apace 路线代码审查验收条件

> Reviewer 在 Step 4 逐项检查。违反任意适用项 = FAIL。
> 本文件是**全局红线**的唯一规范源（规则全文、理由与"详见"链接）；**场景约束**的完整定义以对应场景文档为规范源，本表只做索引与一句话判定。编号 R1-R20 保持稳定，跨文档引用不断链。

## 全局红线定义（所有 apace 算子必须满足）

| # | 约束 | 说明 | 详见 |
|---|------|------|------|
| R1 | 禁止 `__schedmode__(1)` 和 `[[bisheng::core_ratio(1,1)]]` | 会导致 AIC/AIV 串行调度→死锁（`aclError:507015`）；核配比唯一由 `KERNEL_TYPE_MIX_AIC_1_1` 保证为 1:1 | [`architecture.md`](fundamentals/architecture.md) §10 ① |
| R2 | 有 `KERNEL_TYPE_MIX_AIC_1_1` | 每个入口函数都含核配比声明 | [`architecture.md`](fundamentals/architecture.md) §10 ① |
| R3 | 入口变体覆盖 dtype 合同全部组合 + host 运行期 dispatch | 变体数由 dtype 合同决定（FP8 E4M3/E5M2 双组合 = 4 变体；单一组合可单入口）；硬编码单入口 → 异 dtype 字节流被错误模板解释，精度系统性失败 | [`operator-anatomy.md`](operator-design/operator-anatomy.md) §5.3/§7.2 |
| R4 | `block/` `tiling/` 未修改 | 与官网仓原始文件完全一致 | [`architecture.md`](fundamentals/architecture.md) §10 ③ |
| R5 | CrossCore flag idx 配对 | AIV `WaitFlag` idx == AIC `SetFlag` idx（含计数式配对） | [`fusion.md`](fundamentals/fusion.md) §3 |
| R6 | CommContext 与引擎匹配 | UDMA 模式有 `__gm__ CommContext*`；HCCL windows 无 | [`communication.md`](fundamentals/communication.md) |
| R7 | 禁止 `AscendC::Matmul` 高阶 API | 高阶 API 不支持 AIV+URMA 直调场景（与 blaze-shmem 共享此约束） | [`compute.md`](fundamentals/compute.md) |
| R8 | 禁止 HCCL 高阶 API（`Hccl::*`） | HCCL 集合通信库依赖框架注入上下文，AIV+URMA 直调场景拿不到（与 blaze-shmem 共享此约束） | [`comm_shmem.md`](../blaze-shmem/comm_shmem.md) §5 |
| R11 | host 前置校验在 fork/建链前拒绝非法输入 | 整除/对齐/核数下限/Win 容量等校验，报错可操作，main.cpp 与 gen_data.py 双侧；compute-first 场景的完整 9 项清单（含 flag 峰值/512KB/R×T/归约核数）见场景文档 | [`development-guide.md`](operator-design/development-guide.md) §3.5 |
| R12 | UB 静态通信区隔离 | commBuf/barrierBuf 与 TPipe 管理 buffer 物理隔离（静态偏移或 guard TBuf），混用重叠 → 通信数据被踩踏 → 死锁 | [`communication.md`](fundamentals/communication.md) 陷阱 #9 |
| R13 | 通信默认多核并行 | 通信对象 `totalJobs=rankSize`（每核负责 1 个 target 并行 PUT），TeamBarrier `totalJobs=1`——两者正交。"多核写同一 UBMEM flag 竞态 → 退化为 totalJobs=1"是已证伪的臆造约束：退化为单核串行 PUT 通信时间放大 R 倍 | [`communication.md`](fundamentals/communication.md) §2.2；[`optimization-playbook.md`](troubleshooting/optimization-playbook.md) §3 |
| R14 | Win 区数据/元数据分离 + 单轮 PUT ≤ 512KB | PUT/GET 数据不得覆盖 Win 区内元数据/barrier 区（偏移按 host 建链布局确定，host/kernel 同源）：官网布局 barrier flag 在独立 BARRIER_BUF、Win 数据区从 0 可用；共享布局须按约定偏移跳过头部（布局细则与已验证实现见 [`communication.md`](fundamentals/communication.md) 陷阱 #12；缺失 → 覆盖 flag → "假通过"）。perRoundChunkBytes ≤ 512KB（生产实测经验值，官方代码无显式约束，超出间歇 FAIL） | [`communication.md`](fundamentals/communication.md) 陷阱 #12/#13；[`fusion.md`](fundamentals/fusion.md) §6.2.4 |
| R15 | 投产级性能验证门槛 | 精度 PASS 不算投产：性能验证必须覆盖真实大 shape 矩阵（非 toy shape）× R=2/4 双档，并与参考路径（mc2 融合算子 / hccl 分步）对标归档；仅基线采集、无对标 = 未达投产门槛 | [`host-and-testing.md`](operator-design/host-and-testing.md)；[`optimization-playbook.md`](troubleshooting/optimization-playbook.md) §1 |
| R20 | perf 模式 L2 flush 实接线 | 性能采集每轮主 kernel 前必须实际调用 L2 flush kernel（msprof 记录数 == 轮数）；只分配 cacheFlush buffer 不调 kernel = 死代码 = MTE2 带宽虚高 | [`host-and-testing.md`](operator-design/host-and-testing.md) §4 |

## 场景约束（按算子语义特征自动适用，完整定义以场景文档为规范源）

> **适用判据（不依赖场景注册表命中）**：代码含 compute-first 编排（先算后通信、staging 即通信源）即适用 R9/R10/R16；含归约模块（reduceSum）即适用 R17-R19。Reviewer 只要看到对应代码形态就必须检查，不得以"场景未正式命中"为由跳过。

| # | 适用场景 | 约束（一句话判定） | 规范源 |
|---|---------|---------------------|--------|
| R9 | compute-first | host 侧强制校验 flag 计数峰值 ≤ 15（T>1 逐轮配对时峰值 = T），超限硬件异常中断 | [`fusion.md`](fundamentals/fusion.md) §6.2.3 |
| R10 | compute-first | 默认 T 派生强制 `T \| mSeg` 无尾块（PUT 钩子 src 偏移限制）；无法保证时走策略 A（tail padding 32 对齐 + realFragmentSize 限读 + 多套 tiling）——两条路径均合法，禁止为凑无尾块而牺牲流水粒度 | [`fusion.md`](fundamentals/fusion.md) §6.2.7 |
| R16 | compute-first | A 各 rank 段 GM 连续时 mm 默认 FragmentTensor 消 R 循环；vendor R×T 子调用路径须论证 SCALAR 占比可接受（R×T 小 + 大 shape） | [`fusion.md`](fundamentals/fusion.md) §6.2.2 |
| R21 | compute-first | **localLast 编排禁止移除**：frag kernel 必须含 fragment 重排 `[remote..., local]` + 边界提前 SetFlag(flagA)；移除后每轮 Set 双 flag → 峰值 2T（T≤7）+ 丧失通信提前启动。`cFragAddrs_` 顺序写错才是 A/C 错位根因，非 localLast 本身 | [`fusion.md`](fundamentals/fusion.md) §6.2.2 |
| R17 | compute-first（含归约模块） | 归约搬入/输出必须 2D DataCopyPad 且 blockCount=本批行数（多行批量 + 手动 UB）；blockCount=1 逐行归约 = 性能 FAIL；TQue 单 buffer 模型不适用归约批量累加；strided 场景隐式上限防御见下层 API 文档。**归约必须多核分治**：归约核集合内 `SplitToCore(tileM, rsCoreNum, GetBlockIdx(), ...)` 连续行块均分，禁止单核归约（如 `GetBlockIdx()==0` 独立承担全量行 = 写竞争/归约独占 = 性能 FAIL） | [`fusion.md`](fundamentals/fusion.md) §6.2.6；`ascendc-api-best-practices` skill `references/api-pipeline.md` / `api-datacopy.md` |
| R18 | compute-first（含 BF16→FP32 归约） | 禁止 in-place BF16→FP32 Cast（FP32 占 2× 空间覆盖未读 BF16 → 精度系统性错误），必须独立 srcFP32 双缓冲 | [`fusion.md`](fundamentals/fusion.md) §6.2.6 纪律 2；`ascendc-api-best-practices` skill `references/api-precision.md` |
| R19 | compute-first（含归约模块） | MTE2_V/V_MTE2/V_MTE3/MTE3_V 四类 HardEvent 同迭代 Set/Wait 配对（含循环结束消费残留事件）；Set 无配对 Wait = 挂死（507014） | 事件模板：[`scenarios/compute-first-reduce-scatter/development.md`](scenarios/compute-first-reduce-scatter/development.md) §5.3；纪律：[`fusion.md`](fundamentals/fusion.md) §6.2.6 |

## 红线项（操作化检查方法）

> 场景约束（R9/R10/R16/R21/R17-R19）仅在对应场景命中时检查；其余为全局检查项。

| # | 检查方法 |
|---|---------|
| R1 | `grep -rn "schedmode\|core_ratio"` 应为空 |
| R2 | 每个 `__global__` 入口函数含 `KERNEL_TYPE_MIX_AIC_1_1` |
| R3 | 入口变体数 == dtype 合同组合数（FP8 双组合 = 4 变体）；main.cpp 有运行期 dtype dispatch，无硬编码单一入口 |
| R4 | `block/` `tiling/` 与官网仓原始文件 diff 一致；算子目录无共享层副本 |
| R5 | AIV `WaitFlag` idx == AIC `SetFlag` idx（含计数式配对） |
| R6 | UDMA 模式入口有 `__gm__ CommContext*`；HCCL windows 无 |
| R7 | `grep -rn "AscendC::Matmul"` 应为空 |
| R8 | `grep -rn "Hccl::"` 应为空 |
| R9 | main.cpp 有 flag 计数峰值 ≤ 15 强制校验（T>1 时峰值 = T）；**若 flag 峰值 = 2T（每轮 Set 双 flag）而非 T（localLast 双 flag）= 性能 FAIL**（[`fusion.md`](fundamentals/fusion.md) §6.2.2/§6.2.3） |
| R10 | T 派生默认 `T \| mSeg` 无尾块；有尾块时走策略 A（padding 32 对齐 + realFragmentSize + 多套 tiling） |
| R11 | host 前置校验在 fork/建链前执行，main.cpp 与 gen_data.py 双侧；非法输入报错可操作；compute-first 场景 9 项齐全（场景 development.md §3.1） |
| R12 | commBuf/barrierBuf 与 TPipe 管理 buffer 物理隔离（静态偏移或 guard TBuf）；**`TPipe` 与 `MakeMemPtr` 必须二选一，禁止混用**——`grep -n "TPipe\|InitBuffer" kernel/` 与 `grep -n "MakeMemPtr" kernel/` 若同时出现于同一 kernel 目录且未隔离 = FAIL（[`operator-anatomy.md`](operator-design/operator-anatomy.md) §4.3） |
| R13 | 通信对象 `totalJobs=rankSize`、TeamBarrier `totalJobs=1`；出现"totalJobs=1 避免 UBMEM flag 竞态"类设计 = FAIL |
| R14 | Win 数据/元数据偏移 host/kernel 同源；perRoundChunkBytes ≤ 512KB 有 host 校验 |
| R15 | profiling/ 含真实大 shape × R=2/4 双档 × 三路径对标归档；仅 toy shape 基线 = FAIL |
| R16 | mm 内核为 FragmentTensor 自研；vendor 路径 DESIGN.md 有 SCALAR 占比论证；"vendor kernel + FragmentTensor C 输出"设计 = 阻塞级错误 |
| R21 | frag kernel 含 localLast fragment 重排（`[remote..., local]` + 边界提前 SetFlag(flagA)）；`grep -n "localLast\|remote.*local\|flagA" kernel/` 无 localLast 相关代码 = FAIL；每轮 Set 双 flag（flagA+flagB）替代 localLast 双 flag = 性能 FAIL（峰值 2T） |
| R17 | 归约搬入/输出 2D DataCopyPad 且 blockCount=本批行数；strided 场景（N>单次列宽）有 redUbM≤32 或 1D 退化（例外条件见场景约束 R17） |
| R18 | 归约有独立 srcFP32 双缓冲；`grep -n "Cast<float" kernel/` 确认无 in-place 加宽 Cast |
| R19 | MTE2_V/V_MTE2/V_MTE3/MTE3_V 同迭代 Set/Wait 配对；循环结束有残留事件消费（含次数守卫） |
| R20 | perf 循环每轮调用 L2 flush kernel；msprof 结果中 flush kernel 记录数 == 轮数 |

## 常见 FAIL 原因

| 现象 | 根因 | 修复方向 |
|:---|:---|:---|
| 代码中含 `__schedmode__(1)` | 误加调度属性 | 删除，核配比由 `KERNEL_TYPE_MIX_AIC_1_1` 保证 |
| 代码中含 `Hccl::AllReduce` | 误用 HCCL 高阶 API | 改用 `CollectiveComm` 四段式 API |
| 代码中含 `AscendC::Matmul` | 误用 asc-devkit 接口 | 替换为 `Blaze::Gemm::Block::BlockMmad` |
| `block/` 或 `tiling/` 有改动 | 误改共享层 | 恢复共享层文件，只在 `kernel/<op>/` 下改 |
| 精度对不上但无报错 | flag idx 不配对 / splitKNum 配置错 | 核对 flag 编排和 splitKNum 规则 |
| 死锁（aclError:507015） | schedmode 或 flag 不配对；**或通信 UB 静态区被 TPipe 分配覆盖** | 检查无 schedmode；检查 CrossCore flag idx 配对；检查 guard TBuf/静态偏移隔离（[`communication.md`](fundamentals/communication.md) 陷阱 #9） |
| golden 全错且误差不收敛 | golden 切分轴/每卡语义写错 | 回设计阶段核对 golden 语义小节（[`workflow_integration.md`](workflow_integration.md) Step 2 §golden 语义），先修 gen_data 再怀疑 kernel |
| T=1 全 PASS、T>1 精度错 | 多 tile 布局或流水路径 bug（T=1 退化路径会掩盖） | 按 tile/轮次定位超差分布，核对多 tile 下的输出布局与 flag 计数配对 |
| 性能差：通信时间 ≈ R × 单 target 时间 | 通信被退化为单核 totalJobs=1 串行 PUT（臆造"UBMEM flag 竞态"约束） | 改 totalJobs=rankSize 多核并行（R13），TeamBarrier 保持 totalJobs=1 |
| 性能差：AIC CUBE/MTE2 流水占比接近饱和但 cube_utilization 极低 | R×T 子 mm 调用每轮重建 Params，SCALAR 主 bound（特征信号） | 改 FragmentTensor 一次调用消 R 循环（R16）/ 减少 T / Params 增量更新（[`fusion.md`](fundamentals/fusion.md) §6.2.2） |
| 性能差：归约模块耗时高、归约 VEC 占比低但 flag/同步频繁 | 逐行归约（blockCount=1），flag 次数 = 行数×N段数×R | 手动 UB + 多行批量归约 + 2D DataCopyPad blockCount=多行（R17） |
| 精度"假通过"但大 shape/多轮紊乱 | PUT/GET 数据覆盖了 Win 区内元数据/barrier 区 | 数据区与元数据区分离，host/kernel 偏移同源（R14）——精度验证发现不了，靠红线拦截 |
| compute-first 归约读 staging 得 0/旧值 | 多核归约写竞争（多核写同一 yGm）、flag 配对缺失，或归约地址与 mm 写入不同源——**非 cache 问题** | 按序排查：flag idx 配对 → SplitToCore 多核分治（R17）→ staging 写/读地址同源；**禁止直接对 staging 加 dcci**（staging 可见性由 CrossCore flag 配对保证，参考实现不依赖 dcci，加 dcci 属误诊掩盖真根因，见 [`communication.md`](fundamentals/communication.md) 陷阱 #14） |
| 死锁（aclError:507014，950 归约路径） | 归约 SetFlag\<V_MTE2\> 无配对 WaitFlag（跨迭代 Set-Set 无中间 Wait） | 补齐 pingpong slot 事件配对 + 循环结束消费残留事件（R19，事件模板见 [`scenarios/compute-first-reduce-scatter/development.md`](scenarios/compute-first-reduce-scatter/development.md) §5.3） |
| 精度 FAIL：E5M2 变体全元素不通过且误差量级稳定（不收敛） | main.cpp kernel dispatch 硬编码为 E4M3E4M3 入口，E5M2 字节流被 E4M3 模板错误解释 | 实现全部 dtype 变体入口 + host 运行期 dtype dispatch 宏（R3，[`development-guide.md`](operator-design/development-guide.md) §3.5） |
| 精度 FAIL：N>redUbN 时部分输出为零，呈周期性分布（period=redUbM） | 2D DataCopyPad srcStride>0 时 blockCount 超 DAV_3510 隐式上限（约 29-32 行），超出行静默丢零 | host 限制 redUbM ≤ 32（方案 A）+ strided 场景 1D 逐行（方案 B），（R17 例外，[`fusion.md`](fundamentals/fusion.md) §6.2.6 纪律 3） |
| 精度 FAIL：归约结果系统性错误（误差随来源序累积） | in-place BF16→FP32 Cast，FP32 输出覆盖同 buffer 未读 BF16 数据 | 独立 srcFP32 双缓冲（R18） |
| perf 模式 MTE2 带宽虚高 / 性能数据不可复现 | L2 flush 未接线（cacheFlush buffer 分配但未调用 kernel，死代码） | 接入 heavy_add_kernel 每轮 flush + 同步（R20，[`host-and-testing.md`](operator-design/host-and-testing.md) §4 模板） |
| 大 shape 无法运行 / 运行期间歇 FAIL | perRoundChunkBytes 超 512KB（UDMA 可靠性阈值）或 R×T 超 32（FragmentTensor 上限），host 未校验 | host 前置校验补 #7/#8（R11，T 派生联合 maxTileM 约束） |
| DESIGN.md 与代码不一致 | localMatmul 等参数变更后未同步文档 | 同步 DESIGN.md |
