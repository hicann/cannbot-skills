# ScatterUpdate 范式 CANNBotDSL 设计约束（design-spec）

> **层级**：DSL 语言适配层。语言中立场景判定见 [../../patterns.md](../../patterns.md)；语言规则与能力谓词见 [language-rules.md](language-rules.md)。

## 输入与输出

**输入**：算子公式（spec 语义）、公开 Python 参数、shape/dtype/layout 范围、写语义（本范式固定 assign 覆盖写，reduce 归约写划界至 AtomicUpdate 范式）、越界语义声明（[../../generic/bounds.md](../../generic/bounds.md) 优先级决策结果）、确定性要求、目标平台事实、当前设计问题。

**输出**：场景判定结论（写语义划界 + index 形态 × 越界 × 确定性）、结构键设计（det·nondet 双程序族）、结构化 8 维契约 / workspace 布局表、det 三段式段尾覆盖编排、nondet 直写路径、golden 设计、零重编译验证计划；文档文件名与章节组织由调用方指定。

## 按主题读取

| 主题 | 文档 |
|------|------|
| 覆盖写语义与变体 | [../../generic/semantics.md](../../generic/semantics.md) |
| 越界三形态与哨兵契约 | [../../generic/bounds.md](../../generic/bounds.md) |
| 冲突域与排序量 | [../../generic/conflict-domain.md](../../generic/conflict-domain.md) |
| 语言规则与能力谓词 | [language-rules.md](language-rules.md) |
| 动态 shape 程序结构 | [dynamic-shape.md](dynamic-shape.md) |
| det 段尾覆盖 | [deterministic.md](deterministic.md) |
| nondet 直写（语言中立骨架） | [../ascendc/type-assign.md](../ascendc/type-assign.md) §4.1 |
| 快除数学（AscendC 侧） | [../../../common/uintdiv-rules.md](../../../common/uintdiv-rules.md) |
| 参照实现 | [example/](example/README.md) |

## 一致性检查

- 场景判定覆盖 spec 全部维度；**确定性分治**（默认 nondet 直写任意赢；det 段尾覆盖为条件路径——张量源高重复或确定性请求，标量源值同仅高重复）
- **结构键含零 shape 数值**；det·nondet 双程序族独立懒编译
- **主循环 = 三参 grid-stride**；段内扫描（数据依赖退出）用前置条件 while，禁 `while True + break`
- **shape 通道 = 8 维结构化 metadata 直读 + `@host` 派生标量**（零 params；assign 标量源以 1 元素张量承载）
- **派发 = compiled-program**（禁 dim3 JIT 直调于生产）
- **wrapper 同步语义 = enqueue-and-return**：全 scatter 算子越界统一 skip（哨兵沉底/直跳），无 err flag → 禁任何内部 synchronize（graph-capture 友好，写入可见性由流序保证）
- **excl/全局计数原子发布在 tile 循环外**（每核每轮一次）——循环内发布在多波（每核多 tile）时重复累加 → 位置越界写
- **跨 vf 域共享子式 kernel 级预计算**（坐标基址/步进类）——防 CSE 跨域合并编译崩溃；预计算链同时构成 canonical 源
- **纯搬运字节视角**：全 dtype 按字节宽 bit-exact 搬运（禁原生标量赋值承载非常规 dtype）
- **wrapper 视图 pack（条件增强——大 slice 用例带宽收益数倍）**：大 slice W 递降 pack（`8//elem_bytes` 起步；slice/inner 整除 ∧ 首地址 pack 对齐双判据——SIMT 元素对齐硬约束仅需查 base，访问地址 = base + k·W 恒 W 对齐；f8/bool→uint8 位级别名）；kernel 零改动（packed 域 lin 由视图 strides 预除）；inplace 张量不可拷贝对齐 → 降档至 W=1；pack/非 pack 双路径必测。reduce 型禁 pack（姊妹范式 AtomicUpdate）
- **volatile 逐点对表** ASC 源码修饰清单（stcg = 跨核直写/键写；普通 gm[k] = barrier 稳定读；var 终态写 = tracked 普通写）；自旋内 dcci 反外提锚 + 上限安全网
- 越界检查在 index **原值域**、窄化截断最后；**越界 = skip**：det 哨兵沉底、nondet 直跳，双路径同语义；无 err flag（enqueue-and-return）
- KeyT 位宽按冲突域值域自适应（哨兵占用键值域最大值时须保哨兵恒空闲）
- **多波规模（每核多 tile）为必测回归项**（excl 发布位置错误的失效域）
- 空输入（N == 0 或 var 为空）宿主早退先于排序 tiling 预计算
- golden：det 路径复现段尾写序（last-write-wins，binary 比对）；nondet 路径用无竞争/同值两层用例（任意赢与 last-write-wins 数学等价）
- 零重编译断言在验证计划中
- 能力依赖以 [language-rules.md](language-rules.md) §0 谓词表为真值；设计引用的谓词须在表内
