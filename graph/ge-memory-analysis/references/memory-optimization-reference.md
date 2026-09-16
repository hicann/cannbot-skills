# 内存优化手段参考

## 1. 配置入口（**仅当用户要求给出配置方法时**才按下表动态获取，不主动输出）

| 入口 | 覆盖范围 | 权威源（用时读取） |
|------|---------|-------------------|
| session option（C++） | 全部 options 参数（全局/session/graph 级，最全入口） | GE 仓 [ge_api.h](https://gitcode.com/cann/ge/blob/master/inc/external/ge/ge_api.h)（Session 构造/AddGraph 签名）；各参数配置示例见 `https://gitcode.com/cann/ge/blob/master/docs/zh/api/graph_engine_api/cpp/ge/options_params/*.md` 对应小节"配置示例" |
| session option（Python） | 同 C++（dict 传入） | `api/python/ge/ge/session/session.py`（Session.__init__ / add_graph） |
| session option（ACL） | ACLmdlConfig 系列 | `inc/external/acl/acl_mdl.h` + `https://gitcode.com/cann/ge/blob/master/docs/zh/api/` ACL 章节 |
| atc 命令行 | 仅部分参数有专用参数名 | `https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/atc_tools/CLI_options/`（94 个参数文档） |
| atc `--raw_ge_options` | 透传任意 GE options（JSON） | `https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/atc_tools/CLI_options/--raw_ge_options.md`（含 JSON 格式、合并与优先级规则） |
| 环境变量 | 仅个别参数 | `https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/env_vars/` |

优先级：ATC 显式参数 > `--raw_ge_options` JSON > ATC 默认值；graph > session > 全局级。

## 2. 动态获取方法（用户索要配置方法时执行，按用户实际入口选一种）

> 前置：①~④ 的 grep/ls 命令需在 **GE 仓本地克隆**中执行；无本地克隆时可点击网页链接检索（目录页可逐级浏览）。

```bash
# 一次性准备：本地克隆 GE 仓（已有跳过）
git clone https://gitcode.com/cann/ge.git && cd ge

# ① 查某参数的取值/约束/配置示例（以 atomicCleanPolicy 为例，在 GE 仓根目录执行）
grep -rn -A20 'ge.exec.atomicCleanPolicy' docs/zh/api/graph_engine_api/cpp/ge/options_params/
#    网页检索: https://gitcode.com/cann/ge/tree/master/docs/zh/api/graph_engine_api/cpp/ge/options_params/

# ② 查 atc 是否有对应命令行参数
ls docs/zh/user_guides/atc_tools/CLI_options/ | grep -i <关键词>   # 如 atomic、topo
#    网页检索: https://gitcode.com/cann/ge/tree/master/docs/zh/user_guides/atc_tools/CLI_options/

# ③ 查参数常量定义/默认值（代码侧）
grep -rn '<参数名>' inc/graph_metadef/external/ge_common/ge_common_api_types.h

# ④ Python/ACL 入口签名
grep -n 'def __init__\|def add_graph' api/python/ge/ge/session/session.py
```

**规则**：给出的配置代码必须来自上述源的当前版本（不凭记忆写）；文档与代码不一致时以代码为准并注明。用户未要求配置方法时，只给推荐值和参考文档路径即可。

## 3. 优化手段速查表

参数完整定义以对外文档为准；本文只给内存优化场景的**触发指标与推荐值**：

| 手段 | 适用场景（触发指标） | 配置项及推荐值 | 参考文档（仓库内，与 hiascend.com 同源） |
|------|---------------------|--------|------------------------------------------|
| 外置权重 | 权重/常量占大头 | — | https://gitcode.com/cann/ge/blob/master/docs/zh/design/features/external_weight.md |
| 零拷贝 | 用户输入输出占比高（io 大） | — | https://gitcode.com/cann/ge/blob/master/docs/zh/design/features/zero_copy.md |
| 减少流数 | streams 多导致复用差 | — | https://gitcode.com/cann/ge/blob/master/docs/zh/design/features/stream_allocator.md |
| 档位合并 | 多档位取 max 导致偏大 | — | https://gitcode.com/cann/ge/blob/master/docs/zh/design/features/dynamic_gear.md |
| 消除冗余拷贝 | TensorMove 多 | — | https://gitcode.com/cann/ge/blob/master/docs/zh/design/features/tensormove_delete.md |
| 内存冲突消解 | 共享内存读写/布局冲突 | — | https://gitcode.com/cann/ge/blob/master/docs/zh/design/features/memory_conflict.md |
| 清零策略改单独清理 | summary 中 `atomic` 大 | `ge.exec.atomicCleanPolicy="1"`（atc `--atomic_clean_policy=1`）；memset 算子内存过大时用 | https://gitcode.com/cann/ge/blob/master/docs/zh/api/graph_engine_api/cpp/ge/options_params/memory_management.md |
| 换拓扑排序算法 | `topo` 不优且 reuse rate 低 | `ge.topoSortingMode`（atc `--topo_sorting_mode`）；BFS 复用差时试 DFS(1)/RDFS(2)/StableRDFS(3)，逐个对比 reuse rate | https://gitcode.com/cann/ge/blob/master/docs/zh/api/graph_engine_api/cpp/ge/options_params/operator_and_graph_compilation.md |
| 动静态图内存扩展/复用 | 多图共存内存偏大（smp） | `ge.exec.staticMemoryPolicy="4"`（动静都扩展；仅动态 shape 用 "3"）；多图并发执行不可用 2/4 | https://gitcode.com/cann/ge/blob/master/docs/zh/api/graph_engine_api/cpp/ge/options_params/memory_management.md；https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/env_vars/GE_USE_STATIC_MEMORY.md |
| 内存优先策略 | reuse rate 低、图优化空间小（mop） | `ge.exec.memoryOptimizationPolicy="MemoryPriority"`；内存紧张时开启，性能敏感场景慎用 | https://gitcode.com/cann/ge/blob/master/docs/zh/design/modules/graph_metadef/ascend-ir.md（内存优先排序） |

## 4. 运行时占用归因与常见原因（显存持续增长/不释放）

**归因步骤**：
1. `npu-smi info` 周期采样，确认是单调增长还是稳定后不归还
2. 检查 API 配对：`aclmdlLoad ↔ aclmdlUnload`、`CreateSession ↔ DestroySession`、`GEInitialize ↔ GEFinalize`
3. `imas total` 对比 GE 实际申请量与预期模型大小，区分 GE 申请 vs 驱动/RM 预留

**Device 显存常见原因：**

| 原因 | 典型现象 |
|------|---------|
| Session/模型未销毁 | 显存不归还，每次迭代涨一份 |
| VarManager 变量残留 | 跨 session 显存累积（见 https://gitcode.com/cann/ge/blob/master/docs/zh/design/features/variable_manager.md） |
| 内存池扩容不收缩 | **设计行为，非缺陷**（池化复用，进程退出归还） |
| 驱动/RM 预留 | 非 GE 申请，imas total 对比可排除 |

## 5. 源码修改前置约束（面向 GE 开发者）

如需修改 GE 编译器内存分配源码，请先阅读 GE 仓 [memory-constraints.md](https://gitcode.com/cann/ge/blob/master/docs/zh/design/constraints/memory-constraints.md)（静态复用禁改图、对齐策略、悬挂块等约束）。
如需修改 GE 运行时内存源码，请先阅读 GE 仓 [rt2_runtime.md](https://gitcode.com/cann/ge/blob/master/docs/zh/design/constraints/rt2_runtime.md)（RT2 动态 shape）或 [known_shape_runtime.md](https://gitcode.com/cann/ge/blob/master/docs/zh/design/constraints/known_shape_runtime.md)（V1 静态）。
