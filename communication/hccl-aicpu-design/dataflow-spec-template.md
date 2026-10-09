# HCCL AICPU Template 数据流规格 · 模板与填写说明

写一个 HCCL template，本质是先说清楚**数据怎么流动**，再机械翻译成 C++。
这份文件就是描述数据流动的表单。模式 C 中，它是代码实现的前置输入。

---

# 一、怎么用

**三种输入情形，选一种：**

| 场景 | 做法 |
|---|---|
| 算法数据流已明确 | 复制下方模板正文并填写，然后运行校验。 |
| 只有概要需求 | 先根据已知信息起草，无证据的地方标成 `TBD: <具体问题>`；技术项查源码消解，需求取舍作为上游输入缺口。 |
| 基于已有实现改动 | 先逆向现有 template 的 spec，再在其上修改。参考 `assets/dataflow-spec.example.md`（逆向自 `ins_temp_all_reduce_mesh_1D_one_shot.cc`）。 |

**流程：**

```
上游需求 ──> 起草 spec ──> check_spec/check_layout ──> 按证据消解 TBD ──> 作为实现输入
                                        ↑
                             语义 TBD 未消解不进入对应实现
```

```bash
cp dataflow-spec-template.md  <你的类名>-spec.md      # 复制下方「模板正文」部分即可
python3 "<skill 目录>/scripts/check_spec.py" <你的类名>-spec.md   # 自查，有 ERROR 先修
```

**一个 spec 文件对应一个 template（一对 `.h`/`.cc`）。** 分层算法（Broadcast = Scatter_intra +
AllGather_inter）写多份 spec，各自在第 1 章交代组合关系。

**为什么值得先写 spec：** spec 里一个 offset 写错，代码写得再对也是错的；
而 spec 是人五分钟能 review 完的，几百行 C++ 不是。

---

# 二、记法速查

填第 6 章「数据流」要用到的全部记法都在这里，**填表前花五分钟读完这一节就够了**。

## 2.1 地址句柄

一条数据流边 = `源 -> 目标`。源和目标都用下面五个句柄之一：

```
IN [off, len]     本 rank 的 userIn        输入 buffer
OUT[off, len]     本 rank 的 userOut       输出 buffer
SCR[off, len]     本 rank 的 hcclBuff      跨 rank 可见的 scratch
SCR@p[off, len]   rank p 的 hcclBuff       对端的 scratch（这是跨 rank 传输的落点）
OUT@p / IN@p      rank p 的 user buffer    仅图模式下可直接访问对端时才用
```

`off` 是相对本轮 loop 基址的字节偏移，`len` 是字节数。两者都可以是表达式。

## 2.2 预定义符号

```
S    本轮 loop 的字节数            C    本轮 loop 的元素个数
N    本 template 视角的 rank 总数   R    我的 rank
p    对端 rank（通信域 userRank）   NCH  每个远端的 channel 数（多 jetty 时 >1）
DT   单个元素的字节数

RPT  本次调用内的逻辑组数            IRS / ORS  相邻 repeat 起点的间距（in / out）
                                    ISS / OSS  一块之内相邻 rank 的间距（in / out）
```

后五个由 executor 填写，template 只读；取值由实际布局决定，单层没有固定组合。
四个 stride 是字节步长，不是空隙或本轮切片长度；repeat 不是 Ring step 数。
填写前必读 `references/09-template-data-params.md`，分别推导输入、输出和 scratch。

自定义符号在第 5 章「切片」里先定义，之后直接引用，例如：

```
SS = (C / N) * DT        # 单片大小，向下取整
TS = C * DT - SS*(N-1)   # 尾片大小
```

> ⚠️ `p` 是**通信域 userRank**，不是算法内序号。算法内序号（`0..N-1`）另用 `i` / `r` 表示。
> 两者混用是本仓最高频的 bug。

## 2.3 动作

**跨 rank 传输**（下表左列直接写进第 6 章）：

```
SendRecvBatchWrite         源 -> SCR@p[...]        双向交换，我推给 p
SendRecvBatchWriteReduce   源 reduce-> SCR@p[...]  推过去并在对端归约
SendRecvBatchRead          SCR@p[...] -> 目标      双向交换，我从 p 拉
SendRecvBatchReadReduce    SCR@p[...] reduce-> 目标 拉过来并在本地归约
SendBatchWrite             源 -> SCR@p[...]        单向发
RecvBatchRead              SCR@p[...] -> 目标      单向收
```

**本地操作：**

```
LocalCopy     A -> B          纯覆盖，B 被 A 覆写
LocalReduce   A into B        累加，语义是 B = reduce(B, A)
```

`into` 和 `->` 的区别是刻意的：`into` 表示目标是累加位置（读-改-写）。

**同步：**

```
sync main -> sub        主流通知从流开始（必须与下面成对）
sync sub -> main        从流通知主流已完成
barrier(all threads)    等所有流的任务真正落地；只在 64-bit/PROD 软归约前用
```

**流：**

```
T0      主流。示例把本地 copy / reduce 挂它；自定义算法按数据依赖分配
T[i]    第 i 个从流（i 从 1 起）。Mesh 下"第 i 个从流对第 i 个远端"
T[ch]   按 channel 下标取流（多 jetty，ch ∈ 0..NCH-1）
all     所有流，只用于 barrier
```

## 2.4 控制流

```
repeat for rpt in 0..RPT-1:         # 最外层。RPT=1 时自动退化，写了不亏
for i in 1..N-1:                    # 闭区间，含两端
for r in 0..N-1 where r != R:       # 带过滤条件
batch for k in 0..M-1:              # 攒成一次批量下发（一个 SendRecvBatch* 调用）
if 谓词: ...  / else: ...
return                              # 提前返回成功
# 井号开头是注释
```

**可用的谓词**（不要自造，避免翻译时二义）：

```
C == 0            空数据
N == 1            单 rank
needAicpuReduce   数据类型是 INT64/UINT64/FP64，或 reduceOp 是 PROD
isPcie            链路是 PCIe（通常要切成读模式）
graphMode         图模式 OFFLOAD
symMem            对称内存
inFromScratch     输入已经在 CCL buffer 里（仍须核对生产者布局，不据此猜 scratch 跨度）
```

## 2.5 四条最容易错的

**① 跨 rank 边只写一个方向。**
写模式底层只用 tx 那侧的 slice 做搬运，读模式只用 rx 那侧；另一侧只贡献一个 channel 做握手。
所以写模式只写"我推给谁"，读模式只写"我从谁拉"，对称的另一半由框架保证。写第二行是错的。

**② 写模式的目标必须带 `@p`，读模式的源必须带 `@p`。**
写模式 `IN[0,S] -> SCR[...]`（目标没有 `@p`）等于没跨 rank，是常见笔误。

**③ 源和目标的 `len` 必须相等。**
传输长度以源为准，两侧写成不同长度说明 spec 本身有歧义。

**④ 别把 `RPT` 乘进 scratch 倍数。**
第 4 章的倍数按**单次 repeat** 报，被复用时 executor 会自己乘上 `RPT`。
自己先乘一遍等于双重放大，loop 次数翻倍、带宽白掉一半，而且编译和 Checker 都不报错。
反过来，第 6 章写地址时**要**带上 `rpt*IRS`——那是地址，不是预算。

---

# 三、填写顺序建议

不要从第 1 章顺着填，按这个顺序想更省力：

1. **先画第 4 章的 Buffer 布局图** —— 数据落在哪，决定了一切。scratch 倍数从图上直接读出来。
2. **再填第 5 章切片** —— 每一份数据多大、放哪个偏移、尾块归谁。
3. **然后写第 6 章数据流** —— 有了布局和切片，每一步的 `off/len` 就是照着填。
4. **回填第 3 章资源** —— 数据流里用到几个流，就要几个 thread。
5. **补第 2、7、8、9 章** —— 适用条件、边界、cost model、验证。
6. **最后填第 1 章元信息** —— 命名这时候反而最清楚。

拿不准的地方**一律写 `TBD` 加上具体问题，不要猜**，然后汇总到第 10 章。以下实现问题优先用源码、公式和接口契约查证推导；无法确定的需求语义或范围作为上游输入缺口：

- scratch 倍数到底几倍？
- 数据切不整时余数归谁？向上还是向下取整？
- 对端遍历要不要错峰？
- one-shot 还是 two-shot？
- 多 jetty 场景数据怎么二次切分？
- 64-bit / PROD 要不要走软归约兜底？
- 图模式下落点是对端 scratch 还是对端 output？

---

---

# ↓↓↓ 以下是模板正文，复制到你的 `<类名>-spec.md` ↓↓↓

---

# 数据流规格：<类名>

## 1. 元信息

> 判定值写在本行第一个反引号中，补充说明放在外面。架构决策规则见 `references/07-rules.md` §11。

> 填写提示：类名 = `InsTemp` + 算子 + 拓扑 + 变体。文件名是类名的 snake_case，
> 但仓内习惯保留 `1D` 的大写 D（`ins_temp_all_reduce_mesh_1D_one_shot`），跟同目录邻居保持一致。
> 不知道该叫什么就先 `TBD`，`scripts/list_templates.py --op <算子>` 可以看现有命名。

| 字段 | 值 | 说明 |
|---|---|---|
| spec 类型 | `<新建 / 逆向存量>` | 逆向存量 = 照着已有实现反写，如实反映源码即可 |
| 算子 | `<all_reduce / all_gather / reduce_scatter / broadcast / ...>` | 决定落盘目录 |
| 类名 | `<InsTempXxx>` | |
| 文件名 | `<ins_temp_xxx>` | 不带扩展名 |
| 引擎 | `aicpu` | 本 skill 只覆盖 AICPU |
| 拓扑 | `<Mesh1D / NHR / Ring / Tree / 用户定义>` | 写明逻辑通信关系与物理拓扑约束 |
| 变体 | `<one-shot / two-shot / OmniPipe / ...>` | 决定 scratch 倍数量级 |
| 父类 | `<InsAlgTemplateBase>` | 若是已有 template 的变体，填那个类名，只覆写差异方法 |
| 所属层级 | `<单层 / level0-intra / level1-inter>` | |
| 兄弟 template | `<无 / 与 XXX 组合>` | 分层算法填组合关系 |
| 算法名 | `<selector 吐出的字符串>` | executor 注册时用 |
| 绑定 executor | `<InsV2XxxExecutor + TopoMatch1D>` | 谁把本 template 当模板参数吃进去 |
| 契约依据 | `<卡片/方法/分支、check_sources 输出及差异核对记录>` | 按绑定 executor 和调用阶段复用，来源变化只核对差异 |
| 接入风险闭环 | `<无 / 问题、修复或不影响的依据、验证方式>` | 首次构建前关闭影响已声明路径的缺口 |
| 启用方式 | `<标准配置显式选中 / 自动选路>` | 默认标准配置显式选中；配置格式查目标仓实现 |
| 默认选路影响 | `不变` | 或填 `按批准范围改变`；取值写在首个反引号内，变更范围和验收矩阵须预先固定 |
| 选路变更依据 | 无 | 改变默认选择时引用上游需求或已有批准；不能用 benchmark 替代 |
| 允许变更范围 | 无 | 改变时填写算子、selector、拓扑、dtype、档位及边界 |
| 选路验收矩阵 | 无 | 改变时引用预先固定的逐用例旧/新期望算法矩阵；范围外沿用基线 |

一句话说明这个算法在做什么（会成为代码里的 `Describe()`）：

> <一句话>

## 2. 适用条件

> 填写提示：这里决定 `CalcCostCoeff` 开头要不要早退。不适用的规格直接返回空，
> 表示"我不参与这一档的算法选择"。

| 维度 | 约束 | 理由 |
|---|---|---|
| rankSize | `<如 2..8>` | 超出范围 cost model 返回空 |
| 拓扑 | `<如 level0 全连接>` | |
| 数据类型 | `<全部 / 排除 X>` | |
| reduceOp | `<全部 / 排除 PROD>` | |
| 数据量 | `<小数据量占优 / 大数据量占优>` | |
| 模式 | `<单算子 OPBASE / 图模式 OFFLOAD / 都支持>` | |
| selector | `<目标选择链路及配置解析入口>` | 以目标仓实现为准 |
| 平台 | `<目标平台>` | 在验收时提供支持证据 |
| inplace | `<支持 / 拒绝 / 回退及原因>` | 与验证矩阵一致 |

## 3. 资源

> 填写提示：`T0` 是主流，其余是从流。数据流里用到几个流，这里就要填几个。
> 若 thread 数来自 `NCH`（多 jetty）而不是 `N`，翻译时必须覆写 `GetThreadNum()` 并用它截断，
> 否则会用到没同步过的流。

| 项 | 表达式 | 说明 |
|---|---|---|
| thread 总数 | `<N / NCH / ...>` | 主流 + 从流 |
| slaveThreadNum | `<thread 总数 - 1>` | |
| notifyNumPerThread | `<每个从流 1 个>` | |
| notifyNumOnMainThread | `<= slaveThreadNum>` | 每个从流在主流上占一个 |
| notifyIdx main→sub | `<全 0>` | 从流各自只有 1 个 notify |
| notifyIdx sub→main | `<0,1,2,...>` | 第 i 个从流占主流第 i 个 |
| channel 申请 | `<CalcChannelRequestMesh1D / CalcChannelRequestNhr / ...>` | |
| 每远端 channel 数 NCH | `<1 / CalcChannelsPerRank(...)>` | >1 时数据要二次切分 |
| 逻辑 peer | `<算法每轮实际访问的对端集合>` | |
| 申请链路 | `<初始化申请的链路集合>` | |
| 过度申请理由 | `<无 / 与逻辑 peer 不一致的理由>` | |
| 多 channel 切分 | `<offset/len 公式、channel 来源、非整除和空切片策略>` | NCH=1 填不切分 |

## 4. Buffer 布局

> 填写提示：**先画这张图，再填别的。** 画出一轮 loop 内 scratch 被划成几段、每段给谁。
> scratch 倍数就是段数，它是 executor 反推"每轮能搬多少"的唯一依据——
> **报小了会越界踩内存，报大了会白白多切 loop 掉性能。**

```
SCR (hcclBuff)  一轮 loop 占用 = <份数> × <单份大小>
┌──────────┬──────────┬─────┬──────────┐
│ <段 0>   │ <段 1>   │ ... │ <段 n>   │      每段 <大小>
└──────────┴──────────┴─────┴──────────┘
 0          <off1>     <off2>     <offn>

IN  ： <本 rank 输入的形态与大小>
OUT ： <本 rank 输出的形态与大小>
```

| 项 | 值 |
|---|---|
| **scratch 倍数** | `<N / 2 / 1 / 0>` ← 必须等于上图里 scratch 被划成的份数 |
| 理由 | `<为什么是这个数>` |

> 参考取值：one-shot 全量交换 = `N`；Mesh two-shot = `2`（理论 1 份够，非均衡切分留余量）；
> NHR 原地跑 = `1`；完全不过 scratch（如图模式直连对端 output）= `0`。
>
> ⚠️ **按单次 repeat 报，倍数表达式里不许出现 `RPT`。** 被 2/3 级 executor 复用时，
> 它会自己算 `max(另一层倍数, 本层倍数 * RPT)`。自己先乘一遍 = 双重放大。`check_spec.py` 会查这条。

## 5. 切片

> 填写提示：先定义符号，后面数据流直接引用。重点交代**尾块归谁**——
> 数据量除不尽 rank 数时，是最后一片吃掉余数，还是向上取整后最后一片变短。

自定义符号：

```
<例：SS = S              one-shot 不切片，每段就是整份>
<例：SS = (C / N) * DT   向下取整；尾片 TS = C*DT - SS*(N-1)>
```

切法与尾块归属：

> <文字描述>

> ⚠️ 如果切片下标用的是 `R`（通信域 userRank），这个 template 只在
> `userRank == 算法内序号` 的单层场景成立；分层复用时必须先换算成 algRank。

**repeat 与 stride**（按实际 executor 函数与分支填写，不猜默认值）：

| 符号 | 值 | 说明 |
|---|---|---|
| `RPT` | `<逻辑组数>` | 本次调用覆盖的 repeat 数，不是通信 step 数 |
| `IRS` | `<输入 repeat 字节步长>` | `RPT=1` 时该值不参与实际偏移 |
| `ORS` | `<输出 repeat 字节步长>` | 不要求等于 IRS |
| `ISS` | `<输入 slice 字节步长>` | 可以为 0；不得把 sliceSize 当作通用值 |
| `OSS` | `<输出 slice 字节步长>` | 可以为 0；与算子及实际布局一致 |
| scratch 的 repeat 跨度 | `<独立推导的字节步长及依据>` | 核对 buffer 别名、生产者布局；S*N 不是通用公式 |
| 参数赋值来源 | `<executor 文件、函数、分支、版本>` | 检查 inputPtr 是否来自前一阶段 OUTPUT |
| 算法 rank 映射 | `<userRank 到 algRank 的映射>` | Ring step 的 sendIdx/recvIdx 另列，不能当 repeat |

> 填写提示：**`RPT` 填 1 也建议把 `repeat for rpt in 0..RPT-1:` 写进第 6 章**——
> `RPT=1, IRS=ORS=0` 时循环自动退化，零代价；漏写的话，这个 template 一旦被分层 executor
> 当 intra/inter 构件复用，就只会处理第 0 块数据。
> 确定只用于单层、也不打算被复用的可以省，在本章写明理由（`check_spec.py` 会给一条 WARN 提醒）。
> 取值与特殊消费方式见 `references/09-template-data-params.md`。输入和输出分别核对 rank 项与 repeat 项。

**量化样例（必填）**：按 `references/09-template-data-params.md` §9 的格式添加一个或多个
`layout-check` JSON 代码块。每条 case 的 expected 从布局独立推导，formula 对应拟实现/实际 C++ 地址表达式。
覆盖 `L<D`、非零 base、非零算法 rank、非零 stride 与多 repeat 的适用组合；Ring 至少有 `ISS>0 && i>0`。
不支持的组合写明依据。运行 `python3 "$SKILL_DIR/scripts/check_layout.py" <spec.md>`。

## 6. 数据流

> 填写提示：记法见本文件第二节。**跨 rank 边只写一个方向**（写模式写"我推给谁"，
> 读模式写"我从谁拉"）。同步必须成对。

```
# 早退
if C == 0: return
if N == 1:
    T0: LocalCopy  IN[0,S] -> OUT[0,S]
    return

repeat for rpt in 0..RPT-1:      # RPT=1 时退化；块内地址统一带 rpt*IRS / rpt*ORS

    # 阶段一：<名字>
    T0: <动作>

    sync main -> sub

    # 阶段二：<名字>
    for i in 1..N-1:
        p = <对端选取规则>
        T[i]: <原语>  <源> -> <目标>

    sync sub -> main

    # 阶段三：<名字>
    if needAicpuReduce: barrier(all threads)
    for r in <范围>:
        T0: <原语>
```

## 7. 边界条件

> 填写提示：逐项勾选，**不适用也要写明理由**，不要留空。留空的项翻译时会被漏掉。

- [ ] `C == 0`（空数据）：<行为>
- [ ] `N == 1`（单 rank）：<行为，通常只做 LocalCopy>
- [ ] 尾块 / 非均衡切分：<行为>
- [ ] `needAicpuReduce`（INT64 / UINT64 / FP64 / PROD）：<走软归约兜底，还是本算法不涉及归约>
- [ ] `isPcie`（PCIe 链路）：<切读模式，还是不支持>
- [ ] `graphMode`（图模式）：<落点换成对端 output？scratch 倍数是否变 0>
- [ ] `symMem`（对称内存）：<支持与否>
- [ ] 多 jetty（`NCH > 1`）：<数据二次切分规则，还是不支持>
- [ ] `RPT > 1`（被分层 executor 当构件复用）：<支持；地址一律带 rpt*IRS / 还是本 template 只用于单层>
- [ ] inplace / 输入输出重叠：<行为>

涉及新增支持性限制或模式分支时，填写支持性契约：

| 条件/模式 | 字段赋值位置 | 选路消费时是否有效 | 执行消费/阻断位置 | 验证或不可达依据 |
|---|---|---|---|---|
| <按目标版本列出相关条件及完整枚举> | <文件:符号> | <值来源及后续变化> | <守卫/资源分支> | <用例/源码证据/未覆盖> |

检查每个可达模式的 scratch 预算和实际访问一致；“不支持”须有有效阻断证据。
全域协议判断说明同质拓扑约束，零数据/单 rank 的早退与守卫顺序说明语义。
同步点在第 6 章注明 repeat/step/分支作用域，静态数量相同不代表时序一致。
支持范围、阻断条件与消费顺序以目标版本的 selector、executor 和 template 源码为准；复制规格正文时保留实际来源记录。

## 8. Cost model

自动选路或经过成本候选过滤的显式选路均需实现；只有目标源码证明绕过成本候选时才可填“不需要”。按目标版本的配置解析和候选消费链核对 `HCCL_ALGO` 等显式选路条件，候选启用与性能标定分开说明。

> 填写提示：需要进入成本候选时写明启用条件；确实绕过成本候选时才可写「不实现，继承基类空实现」，并保留策略字段与源码依据。

| 项 | 值 |
|---|---|
| 不适用时 | `<如 rankSize > 8 直接返回空>` |
| 策略 | `<必须实现 / 不需要；注明选择链路依据>` |
| 参数传递链 | `<绑定 executor 的成本调用点、algName 等依赖字段及来源>` |
| 自动阈值证据 | `<无阈值 / 既有依据及来源>`；不新增性能验收，不豁免默认路由回归 |
| netType | `<MESH / CLOS>` |
| taskNum | `<本算法下发的 task 量级，影响延迟项>` |
| 传输项 A | `<CalcMeshParam / CalcNHRParams>` |
| n 的取法 | `<executor 统一传 two-shot 的单片大小；one-shot 需乘回 rankSize>` |
| 本地计算项 B | `<localCopy + (N-1) × localReduce / 无>` |

## 9. 验证

| 项 | 值 |
|---|---|
| hccl-vm 参数（轨道 A 定向） | `AI_CPU` 引擎 + `--env 'HCCL_ALGO=<已核验配置>' --expect-algo <注册全名>` + 本算子/目标数据类型/数据量 + 按 rank 数选主验证通信域；最终客观判据由验证能力提供 |
| hccl-vm 参数（轨道 B 回归） | 不设 `HCCL_ALGO` 默认路径 + 与基线同覆盖（首次候选构建前保全旧包并检查完整基线）；按设计时固定的矩阵核对选路 |
| 覆盖规格 | `<如 单 Server 2/4/8 卡；2 Server × 8 卡>` |
| 重点场景 | `<如 count 边界、非对齐 count、64-bit 类型、PROD、大 count 触发多轮 loop>` |
| 测试计划 | `<plan.json 路径；按 executor 单用例冒烟后展开覆盖>` |
| 多 loop / repeat 依据 | `<executor 阈值、测试程序 bytes/count 换算、对齐与独立日志断言>` |

## 10. 待确认

> 确认完请逐条删除。**本节非空时不要开始写代码。**

- [ ] `TBD: <问题 1>`
- [ ] `TBD: <问题 2>`

---

---

# 四、填完之后

```bash
python3 "<skill 目录>/scripts/check_spec.py" <你的>-spec.md
```

它会查：10 个章节是否齐全、有没有残留 `TBD` 或没替换的占位符、
数据流里每条边的方向和长度是否自洽、同步是否成对、
scratch 倍数与数据流对 SCR 的使用是否矛盾、倍数里有没有误乘 `RPT`、
第 5 章的 repeat 声明与第 6 章的 `repeat for` 是否自洽、
thread 数与用到的从流是否矛盾、边界条件有没有留空。

全绿之后，这份 spec 可作为 template 实现输入：
实现能力依据 spec 生成骨架并接好两处 CMake，再映射 `KernelRun` 并检查规范与接线。完整编译和双轨验收由调用 Agent/Plugin 按相应能力契约执行；本设计 Skill 不宣布功能通过。

更细的东西按需查 `references/`：

| 想知道 | 看 |
|---|---|
| 记法的完整定义、spec 各章对应生成什么代码 | `references/07-dataflow-spec.md` |
| 算子语义、形态选择与自定义调度检查 | `references/08-algorithm-design.md` |
| wrapper 原语和资源字段的确切接口 | 目标版本 HCCL/HCOMM 源码；在 Spec 中记录文件、函数与源码身份 |
