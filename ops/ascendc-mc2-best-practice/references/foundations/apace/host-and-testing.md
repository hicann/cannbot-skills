# apace 算子解剖（host 与测试）

> 本文档覆盖 apace 算子的 host 侧与测试侧：CommContext 传递、host 初始化序列、kernel launch、ST 测试工程与性能采集。

## 目录

1. [CommContext 传递模式](#1-commcontext-传递模式)
2. [Host 侧初始化序列（UDMA 模式）](#2-host-侧初始化序列udma-模式)
3. [Kernel Launch](#3-kernel-launch)
4. [perf 模式](#4-perf-模式)
5. [性能采集工程模板（AG ST）](#5-性能采集工程模板ag-st)

---

## 1. CommContext 传递模式

### 模式对比

| 模式 | CommContext | 上下文获取方式 | 支持直调 |
|:---|:---|:---|:---|
| **UDMA** | 第一参数，`__gm__` 指针传递 | Impl::Init 从指针提取 `udmaCtx` 和 `ubmemCtx` | ✅ |
| **HCCL windows** | 不传 | kernel 内部 `GetHcclContext`（依赖框架注入） | ❌ |

> **重要限制**：HCCL windows 模式**无 `__global__` 入口，不支持 CANNBot Kernel 直调工作流**。官网 apace 现有两个算子均为 UDMA 模式。详见 [`architecture.md`](architecture.md) §10 ④。

### 验收条件

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | UDMA 模式 | 入口签名含 `__gm__ CommContext*` 参数 |
| 2 | Host 构造 | Host 侧构造 CommContext → 写入 GM → 传指针 |
| 3 | CommContext 定义位置 | `CommContext` 聚合体（udmaCtx+ubmemCtx）定义在**本算子 tiling_data.h** 中（PUT 在全局命名空间；AG 在 `Apace::AivComm` 命名空间） |

---

## 2. Host 侧初始化序列（UDMA 模式）

完整顺序（与官网两份 ST `main.cpp` 一致，`runAllToAllMatmul` / `RunAllGatherQuantMatmul`）：

```
fork rankNum 子进程（每进程一个 rank）
  └── RunKernel(rankId)
        0. tiling 推导（GetTilingData + CommTilingData 填充，见 operator-anatomy.md §3）
        1. aclInit + aclrtSetDevice + aclrtCreateStream
        2. TCP 交换 HcclRootInfo（RootInfoExchanger::Exchange，tests/st/utils/root_info_exchanger.h）
        3. HcclCommConfigInit + HcclCommInitRootInfoConfig 创建 HcclComm
        4. CommChannelBuilder::CreateDeviceContext(&hostCtx, sizeof(CommContext), ctxTag,
                                                   &hostCtx.udmaCtx, &hostCtx.ubmemCtx)
           → 返回 device GM 中的 CommContext*
        5. 跨 rank host barrier（RootInfoExchanger::Barrier()）  ← 强制，不可省略
        6. ReadFile 输入 + aclrtMalloc + aclrtMemcpy（H2D）
        7. <<<>>> launch kernel（devContext 作第一参数）
```

> CreateDeviceContext 内部建链机制（URMA/UBMEM 通道填充、memset 归属、BARRIER_BUF_SIZE）详见 [`communication.md`](communication.md) §6。

**不可省略的前置/后置条件**：

| 条件 | 原因 |
|:---|:---|
| **CreateDeviceContext 后跨 rank host barrier** | builder 头文件注释明确要求：确保所有 channel 握手完成再 launch，否则 TeamBarrier 轮询远端未就绪 flag 会无限挂死 |
| engine 一致性 | `HcclChannelAcquire` 与 `HcclEngineCtxGet/Create/Copy` 使用同一 engine（`BUILDER_COMM_ENGINE_AIV = 4`，comm_channel_builder.h），否则 ctxTag 复用失效 |

**资源生命周期**：builder 无清理接口，context 按 ctxTag 复用；释放路径与两份 ST 处置差异详见 [`communication.md`](communication.md) §6。

> HCCL Host API 签名详见 `ascendc-api-best-practices` skill `references/api-hccl-host.md`；字段布局与验收条件详见 [`communication.md`](communication.md) §5/§6。

---

## 3. Kernel Launch

`__global__` 入口由 Host 侧 `main.cpp` 通过 `<<<>>>` 调用。

### `<<<>>>` 三参数

| 参数 | 含义 | 来源 |
|:---|:---|:---|
| `usedCoreNum` | block 数（AIC 核数） | PUT：`tilingData.tileQbmmTilingData.usedCoreNum`；AG：`tilingData.mmTile.usedCoreNum`（均由 GetTilingData 推导） |
| `nullptr` | shared mem（不显式指定） | — |
| `stream` | ACL stream | `aclrtCreateStream(&stream)` |

### main.cpp 整体结构（两份官网 ST 一致）

```
main()
  ├── fork rankNum 个子进程（每进程一个 rank）
  │   └── RunKernel(rankNum, rankId, ...)
  │       ├── 填充 tiling（GetTilingData + CommTilingData，详见 operator-anatomy.md §3）
  │       ├── aclInit + aclrtSetDevice + aclrtCreateStream
  │       ├── TCP 交换 HcclRootInfo（RootInfoExchanger）
  │       ├── HcclCommInitRootInfoConfig 创建 HCCL comm
  │       ├── CommChannelBuilder::CreateDeviceContext 填充 CommContext
  │       ├── 跨 rank host barrier（RootInfoExchanger::Barrier()，强制，见 §2）
  │       ├── ReadFile 加载 input 数据
  │       ├── aclrtMalloc + aclrtMemcpy（H2D）
  │       ├── launch（precision 单次 / perf 循环）
  │       ├── aclrtMemcpy（D2H）+ WriteFile("npu_out.bin")
  │       └── 资源释放（aclrtFree 输入输出 → HcclCommDestroy → aclrtDestroyStream → aclrtResetDevice → aclFinalize）
  └── waitpid 收集所有子进程状态
```

### 验收条件

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | 计时方式 | PUT perf 模式用 `std::chrono::steady_clock`（PERF_LOOP_COUNT=10，统计去掉首轮）；AG perf 模式用 `aclrtEvent`（BENCHMARK_ITERATIONS=20，`aclrtEventElapsedTime` 取平均） |
| 2 | tiling 传递 | `tilingData` 按值传递给 kernel（编译器拷贝到 GM） |
| 3 | CommContext 传递 | `devContext` 按指针传递（`CommContext*` 类型 device 指针，由 `CommChannelBuilder::CreateDeviceContext` 返回，详见 [`communication.md`](communication.md) §6） |
| 4 | 输出写入 | 两份 ST 在 precision 与 perf 模式下均写 `npu_out.bin` |
| 5 | 模式区分 | 两份 ST 均支持 `precision`（默认）/ `perf` 命令行模式 |

---

## 4. perf 模式

PUT ST（`runAllToAllMatmul` perf 分支）实测：

| # | 项 | 官网现状 |
|:---|:---|:---|
| 1 | 计时 | `std::chrono::steady_clock`，PERF_LOOP_COUNT=10 轮，统计时去掉首轮（avg/min/max） |
| 2 | 同步 | 循环结束后 `aclrtSynchronizeStream` |
| 3 | cacheFlush buffer | 分配 128M×uint16（256MB）并填充 0x0000，**未被任何 kernel 消费（死代码）**——官网尚未实现真正的 L2 flush |
| 4 | boostIn/boostOut | 同样分配并填充（0x3F00）但未被消费（死代码） |

**skill 侧 L2 flush 扩展**（消除前一轮 L2 cache 热度污染，方法论见 [`../shared/profiling_mc2.md`](../../shared/profiling_mc2.md)）：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | L2 flush kernel | 每轮主 kernel 前调用 L2 flush kernel |
| 2 | flush buffer | 大小大于目标架构 L2 容量，填充非零数据 |
| 3 | 同步 | flush 后 `aclrtSynchronizeStream` 确保完成 |
| 4 | 资源释放 | flush buffer 在主循环后释放 |

> ⚠️ L2 flush 为 **skill 侧方法论**，官网仓无 flush kernel 实体；官网 perf 分支的 cacheFlush buffer 是未接线死代码，不能当作已实现的 L2 flush 引用。

---

## 5. 性能采集工程模板（AG ST）

`all_gather_quant_matmul` ST 工程提供了更完善的性能采集模板，推荐新算子参考（均为仓内实测文件）：

| # | 文件 | 用途 |
|:---|:---|:---|
| 1 | `apace/tests/st/all_gather_quant_matmul/run.sh` | ST 驱动脚本，支持 cases.csv 行选择、`--perf` msprof 采集、`KERNEL_TIMEOUT` 超时覆盖 |
| 2 | `apace/tests/st/all_gather_quant_matmul/scripts/parse_prof.py` | 解析 msprof `op_summary_*.csv`：latency 为 warmup-skip（WARMUP_SKIP=3）+ outlier 过滤（>1.2×min）后的每卡 average；cube_utilization 取 median |
| 3 | `parse_prof.py --check-latest-threshold` | CI 门禁模式，末行输出 latency 数字（供 grep 提取），exit 2 表示超阈值 |
| 4 | `KERNEL_SUBSTR` 适配 | KERNEL_SUBSTR 需替换为新算子入口名（parse_prof.py 硬编码 AllGatherQuantMatmulKernel） |

---

## 后续阅读

| 文档 | 何时读 |
|:---|:---|
| [`operator-anatomy.md`](operator-anatomy.md) | kernel 侧骨架（tiling/Impl/入口） |
| [`communication.md`](communication.md) | 建链机制（builder 内部步骤、CommContext 字段） |
| [`development-guide.md`](development-guide.md) | 开发流程与验收清单 |
| [`../shared/profiling_mc2.md`](../../shared/profiling_mc2.md) | msprof 采集与多卡后处理方法论 |
