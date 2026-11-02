# 基础范式开发规则

> 适用：`references/paradigms/` 下的 VECTOR + SIMD 的基础范式，SIMT/CUBE 范式不在此范围内。  
> adapter DSL 样例待补充。  
> 新增范式、新增/修改范式文档，均以本规范为准。  
> 版本：v1.0  

---

## 1 目标

本规范约束基础范式的知识组织与文档写法，达到三个目的：

1. **可组合（融合友好）**：多个范式的知识能按"同名模块、同构章节"直接对齐、对比、组合使用（如 reduction + broadcast 的融合设计），不需要逐份重新理解结构。
2. **可最小加载**：一个功能模块 = 一份可单独打开的叶子文档，职责单一；使用者按索引即可定位"某功能模块的那一份文档、那一节"，不需要通读整棵范式树。
3. **可高质量开发**：新增范式或文档时有明确形状与自检清单，文档结构一致、职责单一、可机检。

---

## 2 总则（五条原则）

### 2.1 分层放置

| 层 | 目录 | 承接内容 | 判定 |
|----|------|---------|------|
| 通用知识层 | `generic/` | 语言无关的算法、模型、方法论、坐标系、切分与同步模型 | 换一门语言重写时仍成立 |
| 语言适配层 | `adapters/<lang>/` | 强耦合具体语言的实现：API 实体、代码模板、编译器约束、TilingKey/TilingData 具体形态、参考实现 | 只在具体语言下成立 |
| 跨范式公共 | `common/<lang>/` | 跨范式、语言绑定的规则（cast、搬运、同步、值依赖、VF 编程…） | 不隶属于任何单个范式；按语言分目录，语言互斥 |

- `generic/` 中的具体 API 名只能出现在“示例”标注中；删除 API 名后，算法、公式与结论必须仍成立。能力相关量用 `capability-contract.md` 的谓词表达。
- generic 只定义语言中立模型，adapter 只写该语言落地。
- 算法只在 generic 定义；adapter 不重复定义、只引用，可摘要但须标注依据；偏离模型须写明差异与原因，避免双源漂移。
- `common/` 只承载**语言绑定的跨范式规则**，按 `common/<lang>/` 组织；语言中立的跨范式规则暂不单设层，若出现再评审新增。

### 2.2 能力谓词契约

- `generic/` 文档对语言/硬件能力的依赖一律用 `generic/capability-contract.md` 的**谓词**表达（对齐粒度、缓存容量、指令能力、融合链长……）。
- 每个 `adapters/<lang>/capability-mapping.md` 为谓词提供真值与 API 实体对照；"待验证"项必须以假设 + 验证义务的形式输出。
- 新增通用算法引用新能力时，先补谓词定义，再补各语言适配层真值。

### 2.3 单一职责：一份文档 = 一个功能模块

- 文档按功能模块拆分（场景/模板划分、预处理、切分、存活、同步、入口……），禁止在一份文档里混写多个无关模块。
- **同功能模块跨范式同名**（见 §4 模块表）；范式独有的算法/工具另立**专有文档**（§4.3），同样一份文档只讲一件事。

### 2.4 默认骨架

- 标准文档**默认采用 §5.2 骨架**：保持跨范式文档同构、可逐节对照。
- 本范式不适用的标准章节：保留标题、内容写「不涉及」。
- 可选节规则（含 `## 0 速查`）见 §5.4。
- 扩展章节编号顺延；§0 属于模板词汇，不参与顺延。

### 2.5 索引自描述

- 每份**叶子**文档都必须被其所在层的 `index.md` 收录；索引条目写清 `purpose / read_when`（见 §6.2）。
- 修饰范式根 `index.md` 使用 §7.3 的 step 表；其 `generic/` 与 `adapters/<lang>/` 的层 `index.md` 仍按本条。
- 文档与索引同步变更：新增文档当回合补索引；删除/改名文档同时清理索引与链接。

---

## 3 目录结构与命名

### 3.1 目录形状

```
common/                               # 跨范式公共知识（与范式根同级）
├── index.md                          # 路由：指向各语言子层 index
└── <lang>/                           # 按语言分
    ├── index.md                      # 检索表：文档 | purpose | read_when
    └── <rules>.md

{P}/                                  # 范式根；目录名小写，与 op.paradigms 的路由表映射对应
├── patterns.md                       # 唯一入口：场景判定 + 语言路由 + 消费规则
├── generic/                          # 语言中立知识层
│   ├── index.md                      # 检索表：文档 | purpose | read_when
│   └── <module>.md                   # 标准模块（§4）+ 专有模块（§4.3）
└── adapters/<lang>/                  # 语言适配层（lang 小写，如 ascendc / dsl）
    ├── index.md                      # 定位 + 能力映射入口 + 阅读路径 + 公共依赖 + 消费规则
    ├── capability-mapping.md         # 谓词真值表
    ├── design-spec.md                # 设计约束：输入输出 / 按主题读取 / 一致性检查
    ├── overview.md                   # TilingKey / TilingData 具体编码；分类摘要
    ├── tiling-preprocess.md          # 算子信息获取 / 合轴 / 校验
    ├── kernel-entrance.md            # 可选：入口注册 / 模板分发
    ├── value-dependency.md           # 可选：本范式的值依赖规则
    ├── templates/{T}/                # 每模板三件套；T = 模板名（base/group/empty/standard…）
    │   ├── dag-buffers.md
    │   ├── tiling.md
    │   └── kernel.md
    ├── example/                      # 参考实现（登记 example-routes.yaml）
    └── <extra>.md                    # 专有文档（survival-node / vf-fusion-rules / …）

{M}/                                  # 修饰范式（参考 §7）
├── patterns.md                       # 入口：类型/不接管声明 + 链接根 index.md
├── index.md                          # 修饰索引：step -> knowledge（根级）
├── generic/                          # 语言中立知识层
│   ├── index.md                      # 检索表：文档 | purpose | read_when
│   └── <knowledge>.md
└── adapters/<lang>/                  # 可选：语言适配层
    ├── index.md
    ├── <knowledge>.md
    └── example/                      # 可选
```

> 本图只表达放置位置，不表示文档必须存在；基础范式必选文档见 §5.3、§6，修饰范式见 §7；其余按需创建。

### 3.2 命名规则

| 对象 | 规则 | 示例 |
|------|------|------|
| 范式目录 | 小写；与 `op.paradigms` 的映射以路由表为准 | `reduction`、`broadcast` |
| 层目录 | `generic/`；`adapters/<lang>/`（lang 小写） | `adapters/ascendc/` |
| 标准模块文档 | 功能模块名，kebab-case；**不带范式前缀**（目录已表达范式） | `tiling.md`、`kernel-entrance.md` |
| 模板子目录 | `templates/{T}/`；`T` 用模板名 | `templates/binary-base/`、`templates/standard/` |
| 专有文档 | 语义名，kebab-case；必须在层 `index.md` 登记 | `binary-tree.md`、`vf-fusion-rules.md` |
| 示例 | `example/`；路径登记在 `example-routes.yaml` | `example/xdivy_custom_package/` |

禁止事项：目录内再套无信息量的 `references/` 层级；同一功能模块出现两份同层文档；文档名与索引中的名字不一致。

---

## 4 标准模块与文档清单

> 修饰范式不强制标准模块表；知识文档按 §4.3 专有文档组织；根 `index.md` 见 §7。

### 4.1 标准模块表（跨范式同名）

**adapter 元文档**（不解决具体功能问题，进入适配层/设计阶段先读）：

| 文档 | 作用 |
|------|------|
| `index.md` | 适配层入口与检索 |
| `design-spec.md` | 设计任务输入输出 + 跨文档一致性检查 |

**功能模块表**：

| 模块 | generic 文档 | adapters/&lt;lang&gt; 文档 | 模块定位 |
|------|-------------|--------------------------|----------|
| 场景模型（含模板划分） | `scene.md` | `overview.md` | 场景语义与模板划分 |
| 能力契约（可选） | `capability-contract.md` | `capability-mapping.md` | 语言/硬件能力谓词 |
| 预处理（可选） | `shape-preprocessing.md` | `tiling-preprocess.md` | shape 归一化 |
| 值依赖（可选） | — | `value-dependency.md` | 取值型输入处理 |
| 存活与融合（可选） | `liveness-fusion.md` | `templates/{T}/dag-buffers.md` | DAG 与 Buffer 划分 |
| 切分（Tiling） | `tiling.md` | `templates/{T}/tiling.md` | 切分算法与落地 |
| Kernel（可选） | — | `templates/{T}/kernel.md` | kernel 模板实现 |
| 入口（可选） | — | `kernel-entrance.md` | 入口注册与模板分发 |
| 同步（可选） | `sync-model.md` | （可并入 `kernel.md` §7） | 依赖识别与同步点推导 |

> §5.1 是已创建 generic 文档的内容要求；§5.3 是 adapter 必选文档及其知识项。未列出的文档按需创建，创建后按 §5.2 骨架写；本表只描述模块定位与文档对应，不重复列举。
>
> 同一行的 generic 文档是模型/算法源，adapter 文档是其本语言本地化扩展；adapter 只写本地化差异与实体，不重新定义 generic 结论。

- 模块表是全库对照的锚：同模块文档在不同范式下**同名**，便于跨范式对齐与融合。
- 使用者按模块加载最小集：要"切分知识"就取各范式的 `generic/tiling.md`（+ 需要落地细节时取 `adapters/<lang>/templates/{T}/tiling.md`）。

### 4.2 模板实现的层归属

模板分类、选择规则与各模板数据流骨架固定在 `generic/scene.md`；每模板的 `dag-buffers` / `tiling` / `kernel` 默认放 `adapters/<lang>/templates/{T}/`。

### 4.3 专有文档（范式独有）

范式独特的算法与工具知识单列文档，例如：

- reduction：`tree-reduction`、`tuple-reduce` 等
- broadcast：`compute-flow`、`verification`、`kernel-helpers`、`pitfalls` 等

要求：

1. 命名使用语义名，不以模块表行号或章节号命名（如 `module-3.md`、`doc-4.3.md`、`extra-1.md`）；
2. 必须登记所在层 `index.md`（`purpose/read_when` 写清）；
3. 与标准模块内容重叠时，标明主从关系与互链，不复制两份。

### 4.4 可选模块

- 可选模块只在范式确有该需求时创建。
- 以值依赖为例：`adapters/<lang>/value-dependency.md`，承载本范式"哪些输入需要取值、为什么需要、三处声明（op_def / INFERSHAPE / OPTILING）的范式增量"。

---

## 5 章节骨架与知识项

### 5.1 通用层文档内容要求（generic/，章节自由）

> 以下文档按需创建；未创建的不适用。
>
> generic 不固定骨架，章节自由；不要求与 adapter 同名章节。

| 已创建文档 | 内容 |
|------|---------|
| `scene.md` | 坐标系/术语、模板分类（含选择规则）、各模板数据流骨架、编译期特化（模板选择键）、能力边界；能力依赖在正文使用点就地引用谓词名；不出现语言 API/注册宏/struct |
| `capability-contract.md` | 全量谓词定义表（含“影响的通用层文档”）、使用示例 |
| `shape-preprocessing.md` | 归一化算法步骤 |
| `tiling.md` | 语言中立的 Tiling 算法：片上存储切分、多核切分 |
| `liveness-fusion.md` | 存活节点、融合决策方法论等 |
| `sync-model.md` | 同步点推导方法、核内同步/核间同步 |

### 5.2 适配层标准文档骨架（adapters/&lt;lang&gt;/，本地化扩展）

> adapter 文档是 generic 内容的本地化扩展；文档按需创建；一旦创建，按以下骨架的章节顺序/编号写。某章节不需要时，标题保留，内容写「不涉及」。
>
> 增删与扩展规则见 §5.4。
>
> 修饰范式可复用本节骨架；根 `index.md` 见 §7。

**`overview.md`**
```
## 1 范式术语          （与语言相关的术语）
## 2 模板分类摘要      （依据 generic/scene.md）
## 3 TilingKey         （### 3.1 TilingKey 定义；### 3.2 TilingKey 实际分发表）
## 4 TilingData        （### 4.1 TilingData 定义；### 4.2 TilingData 实际对应关系）
```

> `overview.md` 的 §1 只列与语言相关的术语（generic 已定义的不重复定义）；§2 是模板分类摘要，定义源在 `generic/scene.md`；§3/§4 是本语言的 TilingKey/TilingData 具体定义与实际分发表。
>
> TilingKey/TilingData 是 AscendC 术语；其他语言用本语言的对应机制替换，没有则不写。

**`tiling-preprocess.md`**
```
## 1 算子信息获取      （### 1.1 平台信息获取；### 1.2 算子输入输出获取及校验；可引用common公共知识）
## 2 合轴
```

**`kernel-entrance.md`**
```
## 1 kernel 入口
```

**`templates/{T}/dag-buffers.md`**
```
## 1 DAG 分析和 Buffer 划分方案
```

**`templates/{T}/tiling.md`**
```
## 1 片上存储切分
## 2 多核切分
## 3 切分策略微调
## 4 整体切分伪码
## 5 切分后处理
        （### 5.1 片上存储大小计算；### 5.2 Workspace 分配；### 5.3 TilingKey 设置；
          ### 5.4 TilingData 设置；### 5.5 ScheduleMode 设置）
```

**`templates/{T}/kernel.md`**
```
## 1 kernel 概述
## 2 Kernel 骨架代码
## 3 地址偏移计算
## 4 CopyIn 逻辑
## 5 Compute 逻辑
## 6 CopyOut 逻辑
## 7 同步
        （### 7.1 流水同步（核内）；### 7.2 核间同步）
```

### 5.3 适配层必选文档与必需知识项（adapters/&lt;lang&gt;/）

> 不论是否采用默认骨架，以下必选文档必须创建，且其知识项必须在对应文档中齐全且可定位；允许用「不涉及」或指向同层文档的方式显式声明。
>
> `patterns.md` 与各层 `index.md` 也必须在；其要求见 §6。
>
> 可定位 = 由 H2/H3 标题或表格表头承载，标题/表头包含知识项的名称（可追加说明）。

| 必选文档 | 必须覆盖的知识项 |
|------|------------------|
| `overview.md` | TilingKey/TilingData 的具体定义、实际分发表（key 值 → kernel 实例）、TilingData 实际对应关系；模板分类见 `generic/scene.md` |
| `templates/{T}/tiling.md` | 片上存储切分（Vector: UB 切分，含切分轴选择与轴类别约束）、多核切分、切分策略微调、整体切分伪码、切分后处理（片上存储大小 / Workspace / TilingKey / TilingData / ScheduleMode） |

### 5.4 「不涉及」与可选节

- 标准骨架中不适用于本范式的章节：**保留标题，内容写「不涉及」**。
- 可选节：`## 0 速查` 及骨架中显式标注为可选的其它章节，无需声明。
- `## 0 速查` 的约束：标题固定；内容为本文档关键结论/公式/判定规则的摘要，不引入新结论，与正文同批修改；薄文档可省略。
- H3 章节及以下自由增补；扩展 H2 编号顺延，不得插在标准序号中间。

### 5.5 设计契约（强制）

> 本节仅适用于 Vector + SIMD；CUBE 使用静态 Tensor 编程，不依赖 TPipe/TBuf，另行适配。
>
> 片上存储：Vector 指 UB；CUBE 指 L0（L0A/L0B/L0C）/ L1

#### 5.5.1 切分策略

切分策略多种多样，融合场景下 agent 很难做到正确的切分。这里我们统一成先做 UB 切分、再做多核切分、最后按微调策略调整切分。

#### 5.5.2 片上存储切分命名

> 下表 `ub*` 是 Vector 适配层命名；其他语言用本语言存储层级名。

| 概念 | 单切分 | 多切分（以 A/R 为例） |
|------|--------|----------------------|
| 切分轴 | `ubSplitIdx` | `aUbSplitIdx` / `rUbSplitIdx` |
| 切分因子 | `ubFactor` | `aUbFactor` / `rUbFactor` |

- `ubSplitIdx` 是切分轴索引；`ubFactor` 是该轴上切分因子。

#### 5.5.3 多核切分

> 多核切分须 fuse UB 切分轴及外轴；采用大小核均分

- 只能在 **UB 外轴** 上切：每个切分轴的外层块 + `ubSplitIdx` 左侧所有外轴；`ubFactor` 及其右侧内轴不得跨核拆。
- 将所有 UB 外轴 fuse 成扁平 tile 空间：`totalTiles = ∏(所有 UB 外轴 size，含每个切分轴的外层块数)`。
- `totalTiles == 0` 时仍按 1 核发起，kernel 侧短路；`usedCoreNum = min(maxCores, totalTiles)`。
- 均分：`baseTiles = totalTiles / usedCoreNum`，`bigCoreCnt = totalTiles % usedCoreNum`；前 `bigCoreCnt` 核各多 1 个 tile，其余核 `baseTiles` 个。

#### 5.5.4 TilingKey（AscendC 默认适配层）

- 严禁使用 `TILING_KEY_IS` 定义 TilingKey，必须使用 `ASCENDC_TPL_ARGS_DECL` 模板编程方式
- 严禁 TilingKey 包含 dtype / format；dtype 由 `DTYPE_<INPUT_NAME>` 注入，format 由运行时决定
- 必须有完整的注释，明确 TilingKey 各字段的含义

#### 5.5.5 Kernel（AscendC 默认适配层）

- Vector：统一 `TBuf` 单缓冲，不使用 `TQue`；CUBE：静态 Tensor/L1/L0，不依赖 TPipe/TBuf
- 核内流水同步使用 `Mutex Lock/Unlock`，不使用 `SetFlag` / `WaitFlag`

#### 5.5.6 平台与 API（AscendC 默认适配层）

- 平台参数（coreNum/ubSize/blockSize/cacheLineSize）必须从接口获取，禁止写死

---

## 6 索引与路由

> `patterns.md` 由 `routes.yaml` 按 `op.paradigms` 加载；未注册范式走 `general-methodology(-dsl).md`。示例由 `example-routes.yaml` 按 `op.paradigms × op.language` 登记。

### 6.1 `patterns.md`（范式唯一入口）

```markdown
## 1 场景判定         # 判定规则（形式条件）+ 支持/不支持场景 + 典型例子（不支持场景回退到 `general-methodology.md`）
## 2 语言路由         # 按 op.language 选适配层；缺失/未知 → 报告输入缺口，不默认选语言；互斥规则
## 3 通用知识层索引    # generic/ 文件与覆盖一览（指向 generic/index.md）
## 4 消费规则         # 按需加载、≤5 叶子限额、能力真值、示例检视、设计依据、范式优先级
```

### 6.2 层 `index.md`（`generic/` 与 `adapters/<lang>/`）

检索表列固定为：

| 文档 | purpose | read_when |
|------|---------|-----------|

- `purpose`：一句话说清"本文档是什么知识"；
- `read_when`：什么任务/问题下该读它；

索引面向"按任务挑选文档"撰写：一个条目应足以判断"要不要打开这份文档"，避免导览式散文。优先读层索引的阅读路径表（任务阶段 → 入口文档），再按需展开叶子。

`adapters/<lang>/index.md` 另需包含：定位、能力映射入口、按任务阶段的**阅读路径**表、公共知识依赖（指向 `common/<lang>/` 的相对链接）、消费规则、参考实现入口。

消费约定：单个子问题最多展开 **5** 个叶子文档；仍未闭环则拆分为更小的子问题。

路由判断（是否属于本范式）只在 `patterns.md` §1；层索引中 `scene.md` 的 `read_when` 写“建立场景模型”，不承担路由判断。

### 6.3 `common/index.md` 与语言子层

- `common/index.md` 是路由：列出已支持语言及其 `common/<lang>/index.md`。
- `common/<lang>/index.md` 建检索表（列同 6.2），每条写清适用场景；一次任务只读命中语言的子层（语言互斥）。

---

## 7 修饰范式

### 7.1 定义与边界

修饰范式是挂在主范式某一步上的增量知识，复用基础范式的组织架构，但不是独立范式：

- 复用 `generic/` + 可选 `adapters/<lang>/` 的分层；语言中立知识放 `generic/`，AscendC 等举例/API 约束放对应 adapter。
- 不自带模板体系、数据流、buffer 布局、TilingKey/TilingData 结构；这些由主范式决定。
- 只声明自己修改/补充/约束/校验哪一步；不重写主范式。
- 当前修饰范式：NumericalStable（数值稳定化）、DynamicShape（值依赖契约）、VariableOutput（shape 定形链路）、Stateful（in-place 顺序约束）、Recurrence（递推依赖）。
- 补充范式 MaskPredicate 不套用本章的 `{M}/` 结构；Quantization 是路由，LayoutTransform/FusedComposite 是独立/复合范式。

### 7.2 目录与文档

```text
{M}/                      # 修饰范式根
├── patterns.md           # 入口：类型/不接管声明 + 链接根 index.md
├── index.md              # 修饰索引：step -> knowledge（根级，唯一挂载表）
├── generic/              # 语言中立知识层
│   ├── index.md          # 检索表：文档 | purpose | read_when
│   └── <knowledge>.md    # 修饰知识（按 §4.3 专有文档组织）
└── adapters/<lang>/      # 可选：语言适配层（举例/API 约束）
    ├── index.md
    ├── <knowledge>.md
    └── example/          # 可选
```

- `patterns.md`、`generic/`、`adapters/<lang>/` 沿用 §3、§4.3、§6 的分层与索引规则；只有根 `index.md` 是修饰范式新增的。
- `generic/` 必需（至少一份知识文档）；`adapters/<lang>/` 可选。
- 不强制模板三件套；修饰知识按 §4.3 专有文档组织。
- `patterns.md` 可简化：类型/不接管声明 + 根 `index.md` 链接；其余章节自由（有 adapter 时需语言路由）。

### 7.3 根 `index.md`

根 `index.md` 只回答：哪段知识挂在哪一步。

| 列 | 含义 |
|------|------|
| `开发步骤` | 固定为 `formula` / `infershape` / `buffer` / `tiling` / `kernel` |
| `知识` | `generic/...` 或 `adapters/<lang>/...` 的文档/章节链接 |
| `说明` | 一句话说明这段知识对主范式的增量 |

步骤定义：

- `formula`：公式/计算语义、Pass 结构、数值稳定化、递推与谓词语义。
- `infershape`：原型中的 shape/值依赖声明，以及 `InferShape` / `InferShapeRange` / 未知维回退。
- `buffer`：DAG/存活分析、UB buffer 清单与 UB 预算。
- `tiling`：切分算法、多核切分、workspace、TilingKey/TilingData。
- `kernel`：入口、地址偏移、CopyIn/Compute/CopyOut、同步。

- 一个修饰范式影响多个步骤时写多行；每行只对应一段知识。
- 同一范式可在同一步骤写多行（如 `infershape` 的原型声明与 shape 推导分开）。
- 根 `index.md` 必须覆盖所有会影响主范式设计的知识。
- 无法归入上述步骤的知识，可通过 issue 反馈。

示例（VariableOutput）：

| 开发步骤 | 知识 | 说明 |
|----------|------|------|
| `infershape` | `generic/outshape-contract.md#op-def` | 输出声明 `OutputShapeDependOnCompute` / `ParamType`（op_def 声明并入 infershape） |
| `infershape` | `generic/outshape-contract.md#range` | `InferShapeRange` 的 min/max |
| `tiling` | `generic/outshape-contract.md#workspace` | 计数 workspace 布局 |
| `kernel` | `generic/outshape-contract.md#yshape` | `yshape` 写回 + 计数骨架 |

典型挂载：NumericalStable → `formula` / `buffer` / `kernel`；DynamicShape → `infershape` / `tiling`；VariableOutput → `infershape` / `tiling` / `kernel`；Stateful → `kernel`；Recurrence → `formula` / `buffer` / `tiling` / `kernel`。

---

## 附录 A 文档自检清单（作者提交前逐条核对）

- [ ] 放置层正确（generic 无语言实体；adapter 不重造通用算法）
- [ ] 命名符合 §3.2；同功能模块跨范式同名
- [ ] 值依赖（如适用）：独立 `value-dependency.md`，已在层索引登记；无需求则不建
- [ ] 专有文档：语义命名 + 层索引登记 + 与标准模块的主从/互链明确
- [ ] 层 `index.md` 已收录本文件；`purpose / read_when` 齐全且准确
- [ ] 跨范式公共规则（如有）：放在 `common/<lang>/`，并登记 `common/index.md` 与该子层索引
- [ ] 相对链接全部可达；引用的谓词在 `capability-contract.md` 有定义、在本语言 `capability-mapping.md` 有真值
