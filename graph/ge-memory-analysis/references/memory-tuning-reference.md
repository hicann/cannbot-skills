# 内存寻优参考（dump 图离线重放）

> **前置条件**：本节依赖 GE 开发环境（GE 仓源码 + CANN toolkit），完整搭建步骤见下；**寻优脚本与 UT 编译须在 GE 仓根目录执行**（脚本内 `scripts/build_fwk.sh`、`build_ut/` 均为 GE 仓相对路径；dump 图路径同理）。
>
> **适用范围**：仅**静态 shape 场景**（离线重放的是编译期静态内存分配流程）。**动态 shape 场景（RT2.0 内存池，imas_mem_pool.csv 有数据 / imas total 显示 GE-MemoryPool）不支持**——其内存为运行时池分配，无编译期布局可重放；勿向该场景用户索要 dump 图寻优，优化走 memory-optimization-reference.md（如 smp 3/4 缓解池碎片）。
>
> 非 GE 开发者（仅做日志诊断）可跳过本节，SKILL.md 其余场景不受影响。

**完整搭建与执行步骤**：

```bash
# 0. 准备 GE 开发环境（一次性，非 GE 开发者也需完成本步才能使用寻优）
#    a. 下载 GE 代码（已有 ge 仓或个人 fork 的跳过）
git clone https://gitcode.com/cann/ge.git && cd ge
#    b. 安装 CANN toolkit（已装跳过，从 hiascend 获取），并设置路径
export ASCEND_INSTALL_PATH=${HOME}/Ascend/ascend-toolkit/latest/
# 1. 安装 imas 工具（对比表指标解析依赖；已装可跳过，安装命令见 imas-report-reference.md 第 1 节）
# 2. 编译 UT（-d = dump graph 模式，已含寻优用例；需排除 metadef/parser，复用已安装版本）
#    在 GE 仓根目录执行；已存在（build_ut/tests/ge/ut/ge/ge_manual_test）则直接复用跳过编译，
#    不存在时寻优脚本也会自动编译
BUILD_METADEF=OFF BUILD_PARSER=OFF bash scripts/build_fwk.sh -d -j16

# 3. 一键寻优（仍在 GE 仓根目录执行，脚本用绝对/仓外路径指向本 skill；遍历 topo × mop 组合输出对比表）
bash <cannbot-skills仓路径>/graph/ge-memory-analysis/scripts/mem_tuning.sh <静态shape+引擎分配后的dump图.txt>
```

## 1. 寻优脚本 mem_tuning.sh

**前提：imas 工具已安装**（对比表指标由 imas 解析各组合日志得出，未装时脚本会报错并提示安装命令）：

```bash
git clone https://gitee.com/stevenaw/imas.git imas && cd imas && bash make_imas
```

```bash
bash graph/ge-memory-analysis/scripts/mem_tuning.sh <dump图.txt> [ut可执行文件]
# 默认 UT 路径: build_ut/tests/ge/ut/ge/ge_manual_test（已存在则复用，不存在时脚本自动编译）
```

**脚本内置**：
- `unset LD_LIBRARY_PATH / ASCEND_OPP_PATH`（UT 依赖同目录 rpath）
- `ASCEND_GLOBAL_LOG_LEVEL=1`（info 级，保证 `[IMAS]AfterAssignMemory` 输出）
- 用例失败检测（仅匹配 gtest `FAILED` 标记，避免误匹配日志 WARNING）
- 对比表指标由 imas 工具解析各组合日志（`imas total` 汇总行，需 imas 目录存在，可用 `GE_MT_IMAS_DIR` 指定）

**默认寻优维度**：topo=[0 1 2 3] × mop=[default MemoryPriority]

**可选环境变量**：

| 变量 | 作用 |
|------|------|
| `GE_MT_MODES` | topo 模式子集（如 "0 2"） |
| `GE_MT_MOPS` | 内存策略子集（如 "default MemoryPriority"） |
| `GE_MT_LOG_DIR` | 日志目录（默认 mem_tuning_logs） |

**输出**：组合对比表 + 最优组合（含推荐配置）+ 各组合完整日志 `topo<m>_mop<p>.log`；最优组合日志可生成 imas 报告深入分析：`cd imas && ./imas report <LOG_DIR>/topo<m>_mop<p>.log`

## 2. dump 图获取

编译期 dump 图 txt 由环境变量控制：

| 环境变量 | 作用 |
|---------|------|
| `DUMP_GE_GRAPH` | 开关（设 1 开启） |
| `DUMP_GRAPH_LEVEL` | dump 阶段（如含 "Build" 串则输出 Build 阶段图） |

**阶段要求**：寻优必须用**静态 shape + 引擎分配后**阶段的图（如 `ge_proto_xxx_AfterAssignedLogicStreams.txt`、`ge_proto_xxx_graph_0_Build.txt`）：
- 动态 shape 图（含 `origin_symbol_shape`）算不出 size（全 0）——动态场景不适用本节寻优（判定与替代手段见前述"适用范围"），改走 memory-optimization-reference.md 配置优化
- 引擎分配前的图（如 BeforeCanFuse）缺引擎属性，分区失败

## 3. UT 用例

GE 仓 [memory_assigner_manual_test.cc](https://gitcode.com/cann/ge/blob/master/tests/ge/ut/ge/graph/build/memory_assigner_manual_test.cc)（用例 `memory_assign_with_dump_graph`），环境变量驱动：

| 环境变量 | 作用 |
|---------|------|
| `GE_MT_DUMP_FILE` | dump 图 txt 路径（必须含 .txt 才执行） |
| `GE_MT_BUILD_FOR_EVALUATE` | 设 1 走 GraphBuilder::BuildForEvaluate 评估路径（含 topo 选项） |
| `GE_MT_TOPOSORTING_MODE` | topo 模式（0/1/2/3，寻优核心变量） |
| `GE_MT_IO_REUSE` | io 复合复用（如 "1:1" 开输入+输出复用） |
| `GE_MT_IO_MEM_ALLOC_MODE` / `GE_MT_MEMORY_OPTIMIZATION_POLICY` / `GE_MT_ENABLE_SINGLE_STREAM` | io 分配模式 / 内存策略 / 单流，可组合寻优 |

默认 UT 可执行文件：`build_ut/tests/ge/ut/ge/ge_manual_test`

## 4. 寻优结果展示格式（agent 展示寻优结果时遵循）

**markdown 环境用 markdown 表格**（中文列名，渲染对齐由 markdown 保证）；脚本终端输出为 ASCII 列名表格 + 中文图例行（中文字符宽度在不同终端/字体下渲染不一致，终端版保证任何等宽环境对齐）。

```markdown
==== 内存寻优结果: <dump图名> ====
维度: topo=[...] x 内存策略=[...]

| # | 总内存大小 | 复用后大小 | 理论最小值 | 集中清零大小 | 流数 | topo | 内存策略 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 27869696 (26.58M) | 6704640 (6.39M) | 5359616 (5.11M) | 0 (0B) | 1 | 0(BFS) | MemoryPriority |
| 2 | 27869696 (26.58M) | 6743040 (6.43M) | 5359616 (5.11M) | 2097152 (2.00M) | 4 | 1(DFS) | default |
| ... | | | | | | | |

最优组合: topo=0(BFS), 内存策略=MemoryPriority, 复用后大小=6704640 (6.39M)
推荐配置: ge.topoSortingMode="0" + ge.exec.memoryOptimizationPolicy="MemoryPriority"
```

展示规则：
- **列序**：序号(#)、**总内存大小、复用后大小、理论最小值**（三个内存指标在前，体现"总量→复用后→极限"的收敛关系）、**集中清零大小、流数**（两个辅助指标紧随，源自 imas 汇总行 atomic+continuous 之和与 streams）、**topo、内存策略**（参数标识在最后两列）
- **指标来源**：各组合日志由 imas 工具（`imas total`，与 `imas report` 汇总行同源）解析，多图/多memtype 时内存指标求和、流数取最大值
- **集中清零大小**：imas 汇总行 `atomic`（集中清零）与 `continuous`（连续内存）之和，两者均为不可复用内存，值越大说明结构性的浪费越大
- **流数**：imas 汇总行 `streams`，流越多并发越高但复用通常越差
- **topo 列**：值+实际生效名，如 `0(BFS)`（从日志 `topo_mode[...]` 确认）
- **大小格式**：`字节数 (可读值)`，如 `6743040 (6.43M)`；可读值自动换算 K/M/G
- **排序**：按复用后大小**从小到大**（最优在第 1 行，紧邻结论）；大小相同时**内存策略 default 排前**（同效果优先推荐默认策略，免额外配置）
- **结论**：最优组合行（含可读值）+ 推荐配置（可直接使用的 option 串）+ imas 深入分析命令

## 5. 注意

寻优结论需重新编译真实验证（离线重放的 topo 与在线可能因 pass 差异略有出入）。
