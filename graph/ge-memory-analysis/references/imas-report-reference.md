# imas 工具与报告字段参考

## 1. imas 工具使用

### 工具准备

```bash
# 本仓库已构建：/home/ge_develop/imas/（含 imas、imas_tool 可执行文件）
# 首次获取（方式一：git clone）：
git clone https://gitee.com/stevenaw/imas.git && cd imas && bash make_imas
# 方式二（clone 不可用时，下载 zip 解压，已验证可用）：
curl -L -o imas.zip https://gitee.com/stevenaw/imas/repository/archive/master.zip \
  && unzip -q imas.zip && mv imas-master imas && cd imas && bash make_imas
```

两种方式统一落到 `imas/` 目录（skill 内路径均按 `imas/` 引用）。

### 命令行（支持单文件和 glob 多文件）

| 命令 | 用途 |
|------|------|
| `./imas report <log>` | 生成内存报告（静态图 + RT2.0 动态图内存分配） |
| `./imas total <log>` | 查看内存总大小（静态图、内存池、从 rts 实际申请） |
| `./imas offset <log>` | 内存 offset 明细 |
| `./imas addr <log>` | 内存地址映射 |
| `./imas var <log>` | 变量/session 信息 |
| `./imas base <log>` | 基址信息 |
| `./imas check <log>` | 地址合法性校验 |

### 日志要求

模型生成和加载阶段日志，Info 级。
多日志文件（plog 目录）：`./imas report /root/ascend/log/debug/plog/plog*`

日志开关：`export ASCEND_GLOBAL_LOG_LEVEL=1`（atc 加 `--log=info`），
`export ASCEND_HOST_LOG_FILE_NUM=1000` 修改日志最大文件数

**内容校验（拿到日志先做，缺失则开上述开关重新获取）**：

- run plog：`grep -c 'svm_mem_stats_show' <run plog>` 须 > 0（devmm 组件统计，**进程正常退出时打印**；文件截断 / 日志级别不足 / 进程未正常退出时缺失）
- debug plog：`grep -c 'MallocMemory' <debug plog>` 须 > 0（imas 解析来源，如 `MallocMemory:MemoryAllocator::MallocMemory device_id = 0, size= xxx`；文件截断 / 日志级别不足时为 0）
- run 日志备用汇总：`grep -c 'AfterAssignMemory' <run plog>`（无 debug 日志时的简单复用汇总，见下节）
- 校验为 0 时**不得硬分析**：明确告知用户缺失的标志行及原因（截断 / 级别不足 / 阶段不符），提示开日志开关（`ASCEND_GLOBAL_LOG_LEVEL=1`，atc 加 `--log=info`）重新运行获取新日志

### 输出文件

`./imas report` 生成于 `mem_report/`，可用 Excel 打开：

| 文件 | 内容 |
|------|------|
| `imas_mem_graph.csv` | 静态图内存复用报告（每算子分配明细 + 末行 summary） |
| `imas_mem_addr.csv` | 内存地址报告 |
| `imas_mem_variable.csv` | 变量内存复用报告 |
| `imas_mem_pool.csv` | 动态图（RT2.0）内存复用报告 |

### AfterAssignMemory 日志（缺少 debug 日志时的备用路径）

run 日志里 grep `AfterAssignMemory` 可获取简单内存复用汇总（每图一行，GE 内存分配完成时打印）：
```bash
grep -i 'AfterAssignMemory' <run日志>/plog/plog-*.log
```
```
[IMAS]AfterAssignMemory : allreduce_graph_0 memoffset[268470272], memtype[2],
theory_min[268470272], zero_copy[134218752], total_size[268470272],
no_reuse[134218752], streams[3], topo_mode[BFS], mop[], io_reuse[0:0], smp[0]
```
字段与 imas_mem_graph.csv 末行 summary 对应：memoffset=memory_size（实际分配）、theory_min/zero_copy(=io)/no_reuse/streams/topo_mode 同名。**无 reuse rate 和 reach_theory_rate**，需自行换算：`reuse rate ≈ (total_size - memoffset) / total_size`。适用于快速判断；详细分析仍需 debug 日志生成完整报告。

注意：必须用 `./imas` 脚本而非直接调 `imas_tool`（脚本会先 grep 预处理日志）。

### devmm 组件统计（run plog 归因第一步的字段细节）

统计行格式（每类内存 × 每个模块一行，另有 Cached_size 汇总行）：
```
[devmm][_svm_mem_stats_show]MEM_DEV_HUGE_HBM dev0 Mem stats (Bytes).
(module_name=GE; module_id=45; current_alloced_size=0;
allocated_peak_size=274726912; alloc_cnt=3; free_cnt=3)
```

内存类型：

| 类型 | 含义 |
|------|------|
| `HOST_MEM` | Host 侧内存 |
| `MEM_DEV_SMALL_HBM` / `MEM_DEV_HUGE_HBM` | Device HBM 小页/大页（显存主力） |
| `MEM_DEV_SMALL_DDR` / `MEM_DEV_HUGE_DDR` | Device DDR 小页/大页 |

关键字段：

| 字段 | 含义 |
|------|------|
| `module_name` | 组件：GE（图内存/权重）、HCCL（集合通信 buffer）、RUNTIME（rts）、APP（用户 aclrtMalloc） |
| `current_alloced_size` | 当前未释放字节数（退出时=0 说明无泄漏） |
| `allocated_peak_size` | 峰值占用，**归因主要看这个** |
| `alloc_cnt` / `free_cnt` | 分配/释放次数（不等且 current>0 → 疑似泄漏） |
| `Cached_size` | 该类内存驱动缓存，不还 OS（非泄漏） |

## 2. 报告主体判定：静态 shape 图 vs 动态 shape 图

生成报告后**先判定图类型**，两者 csv 主体与分析路径完全不同：

| 判定项 | 静态 shape 图 | 动态 shape 图（RT2.0 执行） |
|--------|--------------|---------------------------|
| imas total GE 分类 | GE-StaticGraph | GE-MemoryPool（"page caching"） |
| imas_mem_graph.csv | 有数据（算子块复用明细 + summary） | **空（0 行）** |
| imas_mem_pool.csv | 空 | 有数据（池块分配明细） |
| 内存模型 | 编译期块级复用：512 块对齐、offset、生命周期 | 运行时按需分配：2M 物理页（cached）内切 64K 最小块，地址复用 |
| 核心指标 | reuse rate / reach_theory_rate / theory_min | 池分配量 vs cached_size（页粒度利用率）、地址复用强度 |

### 动态 shape 图行格式（imas_mem_pool.csv，键值对列）

```
Process_ID,allocator_id,ALLOC,name,NAME,size,SIZE,realsize,REALSIZE,
address,ADDR,life time begin,LB,life time end,LE,cached_size,CS,
theory_size,TS,kernel_free,KF,pool_free,PF
```

| 字段 | 含义 |
|------|------|
| allocator_id | 池实例：`ge_N`（业务张量池）/ `rts_N`（rts 内部，name 常为空） |
| name | 算子节点名；空 = rts 内部分配 |
| size | 池内分配块大小（最小粒度 64K，小张量也占一块） |
| realsize | 算子实际所需大小（小张量常见 544B 级，size/realsize = 块粒度损耗） |
| address | 物理块地址，**同地址多条 = 池复用**（按地址聚合计数看真实物理占用） |
| cached_size | 所属物理页大小（2M page caching，页不收缩属设计行为） |
| kernel_free / pool_free | kernel 释放 / 池释放标记 |

### 动态 shape 图分析要点

- **Top 展示按 address 聚合**：单条 size 常并列 64K（最小块），按 size 降序无区分度；按 address 聚合计数才是物理占用大头（如 147 条分配落 6 个物理块 = 复用正常）
- 小张量块粒度损耗（64K 块装 544B）与池页不收缩均为池化设计行为，非泄漏
- `after_reuse > total` 为块对齐膨胀（同静态图）
- 变量区照常看 imas_mem_variable.csv（不受图类型影响）
- **after_reuse 与峰值活跃差距的分解方法**（"碎片没完全消失"时）：① 按 lb/le 错峰模拟峰值活跃量（imas_mem_pool.csv 算子行可直接算）；② 若池物理申请 = ceil(峰值活跃/申请单元)×申请单元，差距是**最小申请粒度的取整台阶而非碎片**——申请单元从日志 `MallocPhysicalPage:MallocMemory` 记录直接验证（如 ExpandableActiveMemoryAllocator 按 8M 粒度申请，`7 次 × 8388608 = 56M`）；③ 峰值活跃与 theory_min 的差才是执行时序/粒度尾差（如 64K 块对齐）

### imas total 输出头字段（Malloc memory from device 段）

```
DeviceId: 0
StaticMemoryPolicy: 0
Device total         65452113920(60.96G)
Used total           54853632(52.31M)
|-GE                 ...
```

| 字段 | 含义 |
|------|------|
| DeviceId | 设备号 |
| **StaticMemoryPolicy** | 动/静态图内存复用策略（非 0 时 GE-StaticGraph 与 GE-MemoryPool 可能共享扩展区，影响归因；动态 shape 内存池碎片问题可通过取值 3/4 缓解）。**具体取值说明以正式发布文档为准（防取值变化，不在此展开）**：`https://gitcode.com/cann/ge/blob/master/docs/zh/api/graph_engine_api/cpp/ge/options_params/memory_management.md` 的 `ge.exec.staticMemoryPolicy` 小节（配置入口）+ `https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/env_vars/GE_USE_STATIC_MEMORY.md`（环境变量，两者不可同时用），取值代码定义见 GE 仓 [graph_var_manager.h](https://gitcode.com/cann/ge/blob/master/base/graph/manager/graph_var_manager.h) |
| Device total | 整机内存容量（不计入占比） |
| Used total | 本进程 device 实际占用（下级分类按此算占比） |

**分类展示示例**（原始输出为缩进列表 `|-GE-StaticGraph  268470272(256.03M)`，建议整理成此格式展示：按大小降序、GE 子项缩进、附占 Used 比，归因一目了然）：

| 组件/分类 | 大小 | 占 Used 比 | 说明 |
|-----------|------|-----------|------|
| Used total | 394.61M | 100% | 本进程 device 实际占用 |
| **GE** | **260.03M** | **65.9%** | GE 合计 |
| &nbsp;&nbsp;├ StaticGraph | 256.03M | 65.0% | 静态图内存（算子输出、feature map） |
| &nbsp;&nbsp;├ Variable | 2.00M | 0.5% | 变量/常量/权重 |
| &nbsp;&nbsp;└ davinci_model_load | 2.00M | 0.5% | 模型加载（task/结构） |
| APP | 128.00M | 32.4% | 用户 aclrtMalloc |
| HCCL | 6.57M | 1.7% | 通信 buffer |

## 3. 静态图 Summary 行指标（imas_mem_graph.csv 末行）

```
Process_X,g_Y,memory_type,Z,memory_size,N,reuse rate:R%
detail[reach_theory_rate:RT total:T after_reuse:AR theory_min:TM
io:IO no_reuse:NR atomic:A continuous:C streams:S topo:TOPO mop:MOP smp:SMP]
```

| 指标 | 含义 | 优化层级 |
|------|------|---------|
| **total** | 零复用时总内存 | 基线（图结构） |
| **after_reuse** | 复用后实际内存 | 两层均有 |
| **theory_min** | 算法估算的理论最小值 | 拓扑 + 图结构 |
| **reuse rate** | `(total - after_reuse) / total`，取决于拓扑+图结构 | 拓扑 + 图结构 |
| **reach_theory_rate** | `theory_min / after_reuse`（上限 100%），衡量复用算法质量 | 复用算法 |
| **io** | 零拷贝内存（用用户 I/O 地址） | — |
| **no_reuse** | 不可复用内存（算子设定） | — |
| **streams** | 流数，越多并发越高但复用越差 | 流分配 |
| **topo** | 拓扑排序算法（如 DFS/BFS） | 拓扑排序 |
| **atomic** | 需集中清零的内存，不可复用 | — |
| **continuous** | 连续内存大小，影响复用 | — |
| **mop** | 内存优先配置策略 | 配置 |
| **smp** | 静态内存配置策略 | 配置 |

**注意**：`total` 由 `realsize`（32 字节对齐）计算，`after_reuse` 由 `size`（512 字节块对齐）计算。大量小张量时 512 字节块对齐可导致 `after_reuse > total`，是膨胀效应，不是复用失败。

### reach_theory_rate < 100% 常见根因与排查（SKILL.md 1.3 决策树的深入层）

after_reuse 与 theory_min 的差距来源：

| 根因 | 机制 | 报告特征 |
|------|------|---------|
| **连续内存整块生命周期长**（最常见） | Concat/Conv2DTranspose 等连续输入必须物理相邻，整块生命周期被**最晚消费的成员**拖长，且不可拆分错峰复用。**理论值计算未建模此约束**：`AddMemoryStat`/`ReleaseMemorys`（GE 仓 [block_mem_assigner.cc](https://gitcode.com/cann/ge/blob/master/compiler/graph/build/memory/block_mem_assigner.cc)）按块各自生命周期错峰算峰值水位，连续块也按 symbol 级生命周期扣减（`reuse_mem_` 不看 continuous），理论偏乐观 | reassign/nopadding 行多；连续段大块 lt 跨度长（如 26→93），与后段块交叠 |
| **整块 best-fit 失配**（无连续内存时的主因） | 理论模型假设空闲空间可任意切割拼接，实际分配要求**整块容纳**：空洞比请求小（哪怕差 512 字节对齐尾差）即失配，只能向高位新推。典型如多个大块同窗存活时（如 3×12.5M），底座被前一空洞卡住 | 大块 offset 远超理论活跃量；峰值 = 某 io/大块顶端，而非复用区收敛点 |
| **io 块隔离放置** | Data 输入等 zero_copy/no_reuse 块**不参与错峰复用**，直接追加在复用区顶端，与下方空间留缝隙；理论模型中它与其它块错峰共存 | 报告中 `le=4294967294`（永久生命周期）的 Data 块位于高 offset；io 项非 0 且 after_reuse - 复用区顶 ≈ io 大小 |
| 512 字节块对齐膨胀 | 大量小张量，块对齐损耗 | size 与 realsize 差值累积，after_reuse 可能 > total |
| 多流并行 | 跨流块生命周期按并行窗口估算 | streams > 1 时 theory_min 偏乐观 |

**排查方法**：算子级明细中按 offset 找出连续段（reassign 行指向的主块），核对其生命周期与 theory_min 假设的差异；连续段跨度 + 交叠块的额外占用 ≈ after_reuse - theory_min。**无连续内存（continuous=0）但仍有差距时**：模拟理论水位（按 lb/le 错峰累加 size，imas_mem_graph.csv 算子行可直接算），对偏差时刻的活跃块核对实际 offset——失配空洞（整块装不下）与 io 块隔离追加是两大来源；reach_theory_rate 在 90%~100% 属分配器模型精度的正常水平，非结构性浪费。

## 4. 算子行字段

### 行格式

```
Process_ID,Graph_ID,optype,OP_TYPE,name_NAME_offset to ,OFFSET,streamid,STREAM_ID,memory_type,MEM_TYPE,size,SIZE,realsize,REALSIZE,noalignsize,NOALIGN,life time begin,LT_BEGIN,life time end,LT_END,child:r:c:z:s,CHILD:R:C:Z:S,isref,ISREF,batch,BATCH
```

### 字段详解

| 字段 | 含义 |
|------|------|
| `name` | csv 中 name 列原始值，需解析出真实算子名（见下方 name 解析规则） |
| `optype` | 算子类型 |
| `offset to` | 内存偏移，base_addr + offset = 实际地址 |
| `streamid` | 算子所在流 ID。多流可并行执行但降低复用效果 |
| `memory_type` | 内存类型码（如 2 = HBM device 内存） |
| `size` | 实际分配大小（512 块对齐，isref 取 max） |
| `realsize` | 算子所需大小（32 字节对齐） |
| `noalignsize` | 原始大小（shape x dtype） |
| `life time begin/end` | topo 序 ID；end 为消费者 topo ID。4294967294=永久不可复用；A.end=N、B.begin=N 边界不重叠可共享；多流场景 ID 为流内 topo 序，跨流重叠需考虑流间并行 |
| `child:r:c:z:s` | 5 位复合字段 = 子块层级:复用:连续:零拷贝:同流（child=0 无子块，子块与父块共享内存） |
| `isref` | 1=与前一算子共享内存（同一 offset），size 取两者 max |
| `batch` | 动态多 batch 分支名 |
| `reassign output/input` | 连续内存布局下 offset 重分配（Concat 输入、Split 输出） |
| `nopadding` | 连续块无 512 对齐填充，数据紧密相邻 |

**lt_end 格式**：可能含调试信息如 `3--4294967295`，实际结束值取 `--` 之后（即 4294967295）。

### name 解析规则

csv name 列原始值为 `name_<节点全名>_offset to`（张量索引是节点全名的一部分，完整保留）：
- 去掉 `name_` 前缀和 `_offset to` 后缀 → **节点全名**（展示用，同图内唯一）
- 节点全名构成：`<原算子名>_<插入pass标记>_<optype>`（插入节点）或 `<原算子名>`（原图节点）
  - `data_x_HcclContinuousMemcpyPass_Identity_output_0` → 节点 `data_x_HcclContinuousMemcpyPass_Identity`（HcclContinuousMemcpyPass 在 data_x 后插入的 Identity），张量 `output_0`
  - `data_x_output_0` → 节点 `data_x`（原图 Data），张量 `output_0`
- `_output_0` / `_workspace_0`：第 0 个输出 / workspace，是张量索引

### reassign 行格式

```
Process_ID,Graph_ID,optype,OP_TYPE,name_NAME_offset to ,OFFSET,streamid,STREAM_ID,memory_type,MEM_TYPE,size,0,realsize,REALSIZE,reassign output,nopadding,1
```

- `size=0`：未分配额外内存（offset 在既有连续块内重新分配）

### 算子级展示规则

- **按 graph 分组展示**：多图时**禁止合并**，每个 graph（csv 第 2 列，如 `allreduce_graph_0`）单独一张表 + 各自 summary；分析时明确说明图名/图数量。多进程（不同 plog 文件）同理按 Process 分组
- **条目多时**：按 `size` 降序先展示 **Top 10** 内存占用大的算子（通常覆盖绝大部分内存），其余条目汇总说明；条目少（≤10）时全量展示
- **筛选规则**（Top 大算子场景）：只展示 `child=0` 的主块（子块与父块共享内存，不重复计）；`offset` 相同的条目表示复用在一起，**只展示第一个**；但 **`isref=1` 表示该算子与前一算子共用同一块内存**（同一 offset，size 取两者 max），展示时需两行一起出现以体现共用关系，isref 列标注（如 `→data_x` 表示与 data_x 共用）
- **展示列按信息优先级排序**：**算子名、内存类别（output_0/workspace_0）、offset、大小（原始字节数）、生命周期、r:c:z:s** 在前，optype、备注在后；**不设"来源"列**（节点全名已含 pass 溯源，冗余）
- **生命周期**取 `life time begin` 最小值 → `life time end` 最大值（`3--4294967295` 取 `--` 后的值）；特殊值替代：`4294967295` 和 `4294967294` 统一展示为 `∞`（模型生命周期，不可复用）。**单元格内不换行**，用紧凑格式 `2→∞`（去掉箭头两侧空格）；表头列名用 `lt` 等短名，避免表头换行
- **标题栏标注**：表头列名保持简短（`r:c:z:s`），含义**不在表头展开**，统一放表后脚注单独解释
- **表格格式**：算子明细表统一用 **markdown 管道表格**（`|` 分隔列、`---` 分隔表头），列 = 算子名、类别、offset、大小、lt、r:c:z:s（+可选 optype/备注）；不在表格中使用 ASCII 网格或代码块包裹。渲染效果由所在环境决定，源格式保持 markdown 便于富文本环境（GitCode/Web）完整展示
- **算子名**：从 csv name 列解析出**节点全名**（解析规则见上），展示用节点全名（含 pass 溯源，同图内唯一，无需"来源"列）。**同名前缀出现多条是正常的**：插入节点继承原算子名（如 `data_x_HcclContinuousMemcpyPass_Identity` 与 `data_x`），全名不同，非重复条目
- **算子名列宽度控制**：名字单行展示**不换行**（保持节点全名完整、可复制），表格整体过宽由渲染器横向滚动，不用 `<br>` 拆行

## 5. 内存踩踏（stomping）检测与常见踩内存问题分析

### 5.1 触发场景与根因分类

典型触发：**内存复用变化后出现算子精度问题**——网络结构变化（改图/加减控制边/换拓扑）改变复用关系，或手动配置不复用后精度恢复，均说明内存布局变化影响了数据正确性。按"GE 复用是否错误 × 算子访问是否越界"分四类根因：

| # | 根因 | 机制 | 分析手段 |
|---|------|------|---------|
| 1 | **复用错误踩踏**（GE 分配器问题） | 生命周期重叠的块被分到同一 offset，后写者覆写先写者的输出 | 内存报告即可分析（5.2） |
| 2 | **算子多写踩踏**（复用正确，算子越界写） | 算子写超自身 size，覆写地址相邻的其它块 | 看执行算子顺序和内存相邻性（5.3） |
| 3 | **算子多读脏数据** | 算子读了不该读的地址（读越界/读错块），从被复用过的内存读到脏数据 | 实址对账 + 数据比对（5.3） |
| 4 | **算子少写脏数据** | 算子内存 1024 实际只写 512，未写部分来自复用块则残留上一生命周期脏数据 | 实址对账 + 数据比对（5.3） |
| 5 | **VA/PA 映射冲突踩踏**（仅动态 shape + `staticMemoryPolicy` 3/4） | Va1 与 Va2 虚拟地址区间不重叠，但绑定到相同物理页 Pa，实际同一块物理内存互相覆写 | 运行时 VA/PA 校验日志（5.6） |

判别顺序：先走 5.2 排除/确认根因 1（纯报告分析、成本低）；复用确认正确后走 5.3 数据级验证定位 2/3/4。动态 shape 图且配置 `staticMemoryPolicy` 3/4 时，报告均基于 VA 维度（imas_mem_graph.csv / imas_mem_pool.csv 看不到虚拟化层），须先走 5.6 VA/PA 校验排除根因 5。实测定位嫌疑算子用 5.5 dump watch 模式（一次运行，复用错误和算子踩踏均可发现，**仅静态图**）或 5.4 不复用配置二分（多轮运行，仅静态图复用）。

### 5.2 根因 1：复用错误踩踏检测（内存报告分析）

模拟执行顺序检测内存覆盖：A 的输出喂给 C，若执行在 A、C 之间的 B 与 A 的 offset 重叠，B 会覆写 A 的输出，C 读到脏数据。

1. 按 `life time begin` 排序（模拟单流执行顺序）
2. 追踪数据流，找生产者-消费者对（A→C）
3. 对每对 A→C，扫描满足 `lt_begin(A) < lt_begin(B) < lt_begin(C)` 的算子 B：
   - 若 `offset(B)` 与 `[offset(A), offset(A)+size(A))` 重叠 → **检测到内存踩踏**
4. 多流场景：不同流的算子可能并发执行，需考虑跨流并行

```
按 life time begin 排序：
  Op_A   offset=0    size=512    lt_begin=1   → 喂给 Op_C
  Op_B   offset=0    size=256    lt_begin=2   → 与 A 的范围重叠！
  Op_C   offset=512  size=128    lt_begin=3   ← 读 A 的输出（已被 B 踩踏）
```

### 5.3 根因 2/3/4：算子多写/多读/少写（执行顺序与内存相邻性）

复用正确时，精度问题来自算子对内存的错误访问，三个分析要素：

- **执行算子顺序**：实际执行序（非编译 topo 序），多流按流并行窗口推演
- **内存相邻性**：嫌疑算子与受害块的地址相邻关系——越界写只波及相邻块（报告 offset 相邻 ≈ 物理地址相邻，精确地址用 imas addr）
- **实址核查**：算子实际下发的 input/output/workspace 地址（三路径路由见 addr-reference.md），对账报告 offset 与实地址，确认嫌疑算子与受害块的真实地址关系

数据级验证（定位多写/多读/少写的关键）：dump 嫌疑算子的输入/输出数据，比对"应写范围 vs 实写内容"：

| 根因 | dump 比对特征 |
|------|--------------|
| 多写 | 受害块在嫌疑算子执行后出现非预期数据（写入内容可溯源到嫌疑算子） |
| 少写 | 算子输出块"半新半旧"：前段为本算子输出、后段为上一生命周期块的残留值 |
| 多读 | 算子输入地址超出其输入张量范围，或输入值来自复用块的旧数据 |

### 5.4 不复用配置（二分定位手段）

精度问题疑似复用相关时，用不复用配置做对照实验和二分定位。生效判定日志（GEEVENT）：`Reuse memory close, ...`：

| 配置 | 粒度 | 用法 | 正式发布资料 |
|------|------|------|--------------|
| 环境变量 `OP_NO_REUSE_MEM` | 指定算子 | 逗号分隔多个，按**算子名或类型**匹配（可混合配置）；类型匹配兼容 V1~V4 后缀；有总长度与条目数上限（超限报错失效） | [OP_NO_REUSE_MEM.md](https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/env_vars/OP_NO_REUSE_MEM.md) |
| `ge.exec.disableReuseMemory`（session option） | 整网 | 取值仅 `0`/`1`（非法值报 PARAM_INVALID） | [memory_management.md](https://gitcode.com/cann/ge/blob/master/docs/zh/api/graph_engine_api/cpp/ge/options_params/memory_management.md) |
| atc `--disable_reuse_memory` | 整网 | int32，默认 `0`，取值 `0`/`1` | [--disable_reuse_memory.md](https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/atc_tools/CLI_options/--disable_reuse_memory.md) |

定位流程：整网不复用跑一次（精度恢复 → 确认复用相关）→ `OP_NO_REUSE_MEM` 按层/类型二分缩小 → 收敛到嫌疑算子 → 走 5.2/5.3 定根因。多轮二分成本高时，优先用 5.5 watch 模式一次运行定位。

### 5.5 dump watch 模式定位踩内存算子（实测通用手段）

**原理**：实际运行时开启 watcher 模式，每个算子执行后都 dump 一份被观察算子（watcher node）的输出内存，dump 文件命名 `<执行算子名>_To_<watcher算子名>`。怀疑哪个算子的输出内存被踩，就把该算子配成 watcher——**不管踩内存来自复用错误（根因 1）还是算子越界写（根因 2），哪个算子执行后 watcher 内存发生非预期变化，哪个算子就是嫌疑**。一次运行即可定位嫌疑算子，优于 5.4 的多轮二分。

**配置关键字**（dump 配置，具体配置方法参考对应发布资料）：
- `watcher_nodes`：被观察算子列表（精确匹配，可多个），**非空即开启 watcher 模式**
- `layer`：触发算子列表，每个 layer 算子执行后触发一次 watcher 内存 dump；留空 = 全部算子触发
- 正式发布资料：[datadump.md](https://gitcode.com/cann/ge/blob/master/docs/zh/design/features/datadump.md)（RT2.0 不支持 watcher 模式等约束见该文档"约束与限制"）

**判定**：对比 `<opX_To_<watcher>>` 系列 dump 文件——watcher 内存只应在与其有数据依赖的算子执行前后变化；无依赖算子 opX 执行后内容变化 → opX 为嫌疑，再区分根因：
- opX 与 watcher 的 offset 重叠（按 5.2 核对生命周期）→ 根因 1（复用错误）
- offset 不重叠但地址相邻（imas addr 看实际地址）→ 根因 2（算子越界写），转 5.3 数据级验证

**限制**：
- RT2.0 动态 shape 算子不支持（告警 `Dynamic shape op %s does not support watcher mode` 后跳过）
- overflow dump（adump）与 watcher 模式互斥
- 子图模型上配置 watcher 可能不生效（有告警日志）

### 5.6 根因 5：VA/PA 映射冲突踩踏（动态 shape + staticMemoryPolicy 3/4）

**场景与机制**：前四类根因基于内存报告（offset / address / 生命周期）分析，**报告地址均为虚拟地址维度，看不到虚拟化层**。动态 shape 图（RT2.0）配置 `ge.exec.staticMemoryPolicy` 为 `3`（仅动态 shape 内存动态扩展）/`4`（静+动同时扩展）时，内存走虚拟地址扩展机制：预留大段虚拟地址空间，物理页按需映射（MapMem）到虚拟页。**Va1 与 Va2 虚拟地址和大小均不重叠，但若绑定到相同物理页 Pa，实际仍是同一块物理内存**——虚拟地址隔离失效，写 Va1 即踩 Va2；且 5.4 不复用配置、5.5 watch 模式对动态内存池均不适用，静态报告分析（5.2/5.3）也无从发现。

**检测手段**：GE 内置 VA/PA 映射一致性校验（物理页级 VA 记录表，登记/删除每个物理页上映射的 VA 区间），**仅在日志级别 INFO 及以下时开启，默认 WARNING 级别不校验**。复现前设置 `ASCEND_GLOBAL_LOG_LEVEL=1`，运行后检索 plog：

```bash
grep -E 'virtual and physical page mapping check failed|is using. using va info|va intersection' <plog>
```

| 报错关键字 | 含义 |
|-----------|------|
| `virtual and physical page mapping check failed, ...` | VA/PA 映射登记校验失败（附 base/page_size/index 定位信息） |
| `pa_index[N] is using. using va info: {...}, new va info: {...}` | **同一物理页被不同 VA 映射**（Va 不重叠绑同 Pa 的直接证据，两段 va info 分别为已占用 VA 与新申请 VA） |
| `va intersection, pa_index[N]` | 同一物理页上的 VA 区间相交 |

命中即确认 VA/PA 映射冲突。校验失败后分配器置快速失败标志，后续分配直接报错，可防踩踏扩大。

**staticMemoryPolicy 取值**：`0`（默认，动态分配不扩展）/`2`（静态 shape 内存扩展）/`3`（仅动态 shape 扩展，缓解池碎片）/`4`（静+动同时扩展），正式发布资料见 [memory_management.md](https://gitcode.com/cann/ge/blob/master/docs/zh/api/graph_engine_api/cpp/ge/options_params/memory_management.md)；虚拟内存兼容性设计（rtReserveMemAddress 预留及 fallback）见 GE 仓 [memory-constraints.md](https://gitcode.com/cann/ge/blob/master/docs/zh/design/constraints/memory-constraints.md) 约束 3。

## 6. 实测示例（allreduce 图）

```
summary: memory_size 268470272(256.03M), reuse rate:0%
detail[reach_theory_rate:100.00 total:256.03M after_reuse:256.03M
theory_min:256.03M io:128.00M no_reuse:128.00M atomic:0 continuous:0 streams:3 topo:BFS]
```

算子行（按 size 排序，生命周期取 begin 最小 → end 最大，markdown 管道表格）：

| 算子名（节点全名） | 类别 | offset | 大小(字节) | lt | r:c:z:s |
|---------------------|----------|--------|------------|----|---------|
| data_x_HcclContinuousMemcpyPass_Identity | output_0 | 0 | 67109376 | 2→∞ | 1:0:0:1 |
| allreduce_op | output_0 | 67109376 | 67109376 | 3→∞ | 1:0:0:1 |
| identity_0_Node_Output_MemLayoutConflict_0 | output_0 | 134251520 | 67109376 | 1→∞ | 0:0:1:1 |
| data_x | output_0 | 201360896 | 67109376 | 0→∞ | 0:0:1:1 |
| allreduce_op | workspace_0 | 134218752 | 32768 | 3→3 | 1:0:0:1 |

脚注：
- `data_x_HcclContinuousMemcpyPass_Identity` = HcclContinuousMemcpyPass 在 data_x 后插入的 Identity（optype=Identity）
- `identity_0_Node_Output_MemLayoutConflict_0` = MemLayoutConflict 插入（optype=Identity，零拷贝）
- `allreduce_op` = 原图 HcomAllReduce（output_0 输出 / workspace_0 工作区）；`data_x` = 原图 Data 输入节点（零拷贝）
- lt = 生命周期；∞ = 模型生命周期（4294967295/4294967294）；r:c:z:s = 复用:连续:零拷贝:同流

分析：
- **reuse rate=0%** 但 **reach_theory_rate=100%**：复用算法已最优，增长空间在图结构
- **io=128M=no_reuse**：两个 64M 零拷贝块（z:1，Data 和经 MemLayoutConflict 插入的 Identity），占总量一半，不可复用（设计如此）
- **lt_end=4294967294（永久）**：零拷贝块生命周期为模型级，无法参与复用
- **实际需优化的是剩余 128M**：两个 64M 块（Identity/HcomAllReduce 各一）生命周期均持续到模型结束，互相重叠无法共享 offset

结论：当前结构下已最优；进一步压缩需改图结构（如缩短 HcomAllReduce 输出的消费时机）或评估零拷贝占比是否合理。
