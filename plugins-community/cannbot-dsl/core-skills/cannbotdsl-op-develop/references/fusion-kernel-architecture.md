# 融合算子架构与参数处理规范

## 适用场景

- 新写或重构 CANNBotDSL 融合算子、长流水 kernel、存在多个执行边界的 kernel。
- 用户要求参考指定示例或相近算子的组织方式、buffer 规划、shape/参数处理方式。
- 代码已经能跑，但入口 layout、类职责、host wrapper、测试参数或注释风格混乱。

## 典型症状

- 对外接口使用一种 layout，kernel 内又整体 reshape/transpose 成另一套大 tensor。
- `run()` 中包含大量 host 侧 layout 转换、临时 tensor 构造或逐元素 Python 循环。
- Cube、Vector、调度、golden、shape 适配逻辑混在一起，后续难以改流水和同步。
- 测试参数使用内部 chunk 维度，和真实用户接口或同类示例不一致。
- buffer 申请分散、命名随意，无法看出 UB/L1/L0/GM 的所有权和生命周期。
- 为了先跑通而把短生命周期中间量大量落 GM，后续代码出现大量 scratch、sync id 和重复 GM 往返。
- 参考指定示例时，只复用了局部 API，没有复用它的职责边界、buffer 生命周期和参数处理方式。

## 写代码前的硬性准备

开始写融合算子或重构前，先在实现说明、todo 或文件顶部注释中写清楚以下 5 张表。表没想清楚时不要先写“临时可跑”的 GM scratch 版本。

1. **公开调用契约表**
   - 公开调用契约是调用方能看见并必须遵守的接口：`run()` 的真实输入、输出、标量属性、dtype、layout、shape 约束和错误条件。
   - 内部 scratch、预计算辅助数据、workspace、tiling、调试输出不进入公开调用契约，除非它们本来就是用户可见输入或输出。

2. **逻辑轴角色表**
   - 对每个输入、输出和中间量标注逻辑轴角色，例如批量轴、分组轴、序列/空间轴、特征轴、归约轴、状态轴、打包轴、广播轴。
   - 后续变量名、buffer 名、`tile_view` 形状和注释必须跟这张表一致。
   - 禁止因为两个轴当前 tile 大小相同，就把不同语义轴混用、混命名或复用错误布局。实现 tile 可以重合，逻辑轴角色不能重合。

3. **buffer 生命周期表**
   - 每个中间量标注生产者、消费者、生命周期、推荐层级和是否需要跨执行边界。
   - 默认层级选择：重复读或阶段内驻留数据优先 L1/UB，matmul 输入输出走 L1/L0/FIXPIPE，短生命周期 vector 临时量走 UB。
   - GM 只能用于公开输入输出、跨执行边界/跨 kernel 数据、L1/UB 容量放不下的数据、或已确认的 DSL/toolchain 限制。
   - 表里必须单独标出 dtype 生命周期：输入 dtype、on-chip 计算 dtype、workspace/GM dtype、最终输出 dtype，以及 L0C2GM add/atomic 的 dtype。
   - L0C 是 FP32 不等于最终 GM 累加也是 FP32。BF16/FP16 GM 输出要确认 fixpipe cast 和 atomic setter 是否按 GM 目标 dtype 生效。

4. **GM 例外表**
   - 每一个新增 GM scratch 都必须写明原因：跨执行边界、容量不足、跨 AIV/AIC 可见性、layout/stride 限制或工具链缺口。
   - 如果原因只是“实现简单”，不能放 GM，必须重新设计成 UB/L1/L0 通路。
   - 因工具链限制临时放 GM 时，在代码注释和 todo 中同时标注后续 on-chip 化路径。

5. **同步和生命周期表**
   - 标注每个 V2C/C2V handoff 的生产 pipe、消费 pipe、保护数据、sync id、生命周期。
   - 存在多个执行边界时要明确边界前后的保留数据；边界内 buffer 可以覆盖复用，跨边界只保留必要数据。
   - 同步语义以当前实现的真实 producer/consumer 链为准；不能用相近实现的时序假设替代当前 kernel 的消费证据。
   - 区分 ready token 和 free/release token：消费者必须等待真实 producer 发出的 ready，不能用初始化 free token 或本阶段自 Set/Wait 伪装数据可见。
   - 执行边界如果只回收 allocator 游标，必须同时说明后续是否还会读取同一物理 buffer；只有确认没有跨 pipe 或跨边界消费者仍在读时，才能 rewind 后覆盖。
   - 使用 per-subblock token offset 的 handoff，发布和等待都要按 `id + subblock*16` 建表，不能只让多个 subblock 等同一个 base id。
   - L0C 经 FIXPIPE 写回后要分别考虑两个消费者：后续 M 复用 L0C 缓冲区需要 `FIXPIPE -> M`，后续 MTE2 读 GM 需要 `FIXPIPE -> MTE2`。二者不能互相替代。

6. **VF 和 dtype 表**
   - 每个 `with vf()` 对应一个逻辑向量公式，列出输入 UB、输出 UB、中间 FP32/BF16 buffer、producer、consumer。
   - MTE2/MTE3、sync、跨 PIPE 同步原语在 VF 外；VF 内只保留支持融合的 vector op 和 UB<->UB copy。
   - 如果使用两个 BF16 UB 避免输入/输出别名，必须在表里写清原始输入和最终输出的生命周期。

## 设计原则

1. **公共接口优先**
   - 先确定用户可见的公开调用接口：shape、dtype、layout、可选参数、输出 shape。
   - 参考指定融合算子风格时，公开输入输出 layout 必须保持用户输入布局；内部计算需要的重排只能作为 kernel 内 tile/chunk 视图映射或 workspace 布局出现。
   - 测试参数必须使用公共接口维度，不用内部 chunk 维度伪装成用户输入。
   - 内部 chunk、tile、padding、workspace 只在 kernel 或 helper 内出现。
   - 从自定义算子迁移到 CANNBotDSL 示例时，`workspaceGM`、`tilingGM` 是算子注册/运行时尾参，不是 Python host 侧需要传入的真实 tensor。CANNBotDSL `run()` 公开参数应只包含用户真实输入、真实输出和标量属性。
   - 原算子 host 内部申请的 scratch、辅助表、workspace tensor 不能照搬成 `run()` 参数；这些要么变成 kernel 内 UB/L1/L0 buffer，要么变成测试内部临时对象。

2. **kernel 内完成 tile 级 layout 映射**
   - 优先保持输入输出的公共 layout，在 kernel 内通过 `BlockInfo` 计算 tile/chunk 坐标，再用 `tile_view`、strided `mem_copy` 或 copy engine 取当前块。
   - 不要为了实现方便，在 host 侧先整体转成内部大 layout；除非这本来就是算子对用户承诺的真实输入输出格式，或性能/工具链限制已经明确。
   - 需要内部 workspace 时，把 workspace 设计成计算友好的 layout，并用当前 tile 的读写路径填充。
   - shape 先按逻辑轴角色分类，再决定实现 tile。不要为了套用某个固定模板，把不同语义的数据强行改成同一种 tile 形状。
   - 某个轴是否切分是 buffer/硬件实现决策，不是算法默认要求。先区分该轴在当前公式里是输出轴、归约轴、广播轴还是状态轴，再决定是否引入 tile 循环。

3. **调度和计算分层**
   - 顶层 kernel 负责：读取 shape、分核、tile id 到逻辑坐标映射、流水调度、跨模块同步顺序。
   - Cube 类负责：AIC/matmul/L1/L0/FIXPIPE 相关搬运和计算。
   - Vector 类负责：AIV/UB/vector/vf/规约、变换、结构化规则和 elementwise 相关逻辑。
   - BlockInfo/Tiling helper 负责：tile 范围、稀疏/尾块/变长边界，不做实际计算。
   - Cube/Vector 构造函数优先接收明确的计算 tile、逻辑轴长度、子块编号和必要接口；不要用 `Cube(ctx)` / `Vector(ctx)` 让类任意访问整个 kernel。
   - copy engine 属于实际使用者：GM->L1、L1->L0、FIXPIPE 相关 engine 放 Cube；UB->L1、格式转换、结构化辅助数据相关 engine 放 Vector。不要在顶层 kernel 创建后到处透传。
   - 类名优先使用 `Cube`、`Vector`、`BlockInfo` 这类通用角色名；复杂辅助函数放回对应角色类，避免在文件外堆大量 `_xxx` helper 形成第二套隐式架构。
   - 参考指定示例时，优先复用它的职责边界和数据流，不要只复制局部 API 调用。

4. **buffer 所有权清晰**
   - buffer 在最接近使用者的类里申请，避免顶层 kernel 成为所有 buffer 的堆放点。
   - 单块临时存储使用 `Buffer(MemLoc.*, shape, dtype, ...)`；double/depth-N staging 使用 `Channel(..., depth=N)`。
   - 只有在明确需要别名复用或特殊生命周期时才使用底层 buffer 包装；同时用中文注释说明复用关系。
   - 目标实现默认走 on-chip 通路：GM 输入搬到 L1/UB，Cube 在 L1/L0/FIXPIPE 间推进，Vector 在 UB 内处理，再通过明确 handoff 写到 L1/UB 或最终 GM。
   - 阶段内复用或多次消费的数据应优先常驻 L1/UB；不要每次写 GM 再读回。
   - 算法内部临时量、局部累加、结构化辅助张量、广播/变换临时区等短生命周期数据不应默认落 GM。只有容量或工具链限制证明不可行时，才可作为临时 GM scratch。
   - 如果使用 `Buffer(..., addr=...)`、Channel 显式地址或 GM 区域编址，必须说明别名关系和生命周期；Buffer/Channel 都不会替你提供 alias 排序保护。
   - 手动复用物理地址时同时核对视图 shape 和字节区间；不能把较大视图挂到较小 buffer 的地址上，即使 dtype 或 tile 维度看起来相近。
   - L0A/L0B/L0C double buffer 需要先算清单级容量和 arena 上限；如果一个 tile 已经占满对应 L0 空间，advance 到第二级会越界，必须退回单级或缩小 tile。
   - 多个 GM 输出、workspace 或 scratch 共存时，store 目标和 offset 要单独校验；不要只凭变量名判断写到了正确输出。

5. **参数命名稳定**
   - 公共维度命名保持一致，例如 `B/S/N/D` 或 `T/N/D`，不要在同一文件里混用 `H/N/head` 指同一维。
   - 内部维度命名要带语义，例如 `chunk_num`、`chunk_pos`、`tile_m_idx`、`head_idx`。
   - 常量不要使用裸缩写；使用带语义的名称，并在定义处说明含义、单位和是否为公开约束。
   - 中间变量名要反映真实逻辑轴角色和数据语义。不要因为两个 tile 形状相同，就使用同一个命名前缀或把一个语义的数据伪装成另一个语义。
   - 只有真实发生某轴分块时才引入对应的 `*_tile_size`、`*_tile_idx`、`*_tile_num`，并说明这是实现分块，不是算法公式。
   - 通用代码注释不要写只在个人上下文里成立的内部代称；需要区分版本或阶段时写清楚算法阶段、buffer 区域或数据语义。
   - 文件内保留一处 layout 映射说明，后续代码只引用这套命名。
   - buffer 和 tile 名字优先表达领域语义。不要因为当前 API 参数是矩阵乘的左右操作数，就把真实公式里的数据改名成 `lhs/rhs`；使用能追踪公式的数据名，例如 `<tensor_role>_l1`、`<state_name>_ub`、`<output_name>_tile`。

6. **方法名表达真实动作**
   - 普通类方法不要命名成 `_emit_xxx` 这类难以判断行为的形式；`emit` 不等于计算，也不说明数据来源和去向。
   - 计算过程用 `calculate_*` 或 `compute_*`，搬运过程用 `load_*`、`store_*`、`copy_*`。
   - 多个 matmul 或多阶段结果相加时用 `accumulate_*`。

7. **复杂数学过程必须可读**
   - 分块求逆、迭代近似、组合公式、递推更新等复杂过程，要用中文注释解释数据区域、近似公式、组合顺序和输出含义。
   - 注释重点是算法不变量和 buffer 语义，例如“缓冲 0/1 保存上一轮近似与本轮修正项”“先组合对角块，再写回尾块”，不要逐行翻译赋值语句。

## 推荐结构

```python
class Cube:
    """AIC 侧 matmul、L1/L0、FIXPIPE 搬运。"""

    def __init__(self, compute_tile, ...):
        # 申请 Cube 自己使用的 L1/L0 buffer 和 copy engine。
        ...


class Vector:
    """AIV 侧 UB、结构化规则、elementwise、归一化或后处理。"""

    def __init__(self, vector_tile, subblock_idx, ...):
        # 申请 Vector 自己使用的 UB buffer 和 UB/L1 搬运 engine。
        ...


class BlockInfo:
    """tile 范围、边界、稀疏或变长策略。"""


class Operator:
    @kernel
    def kernel(...):
        # 读取公共 shape
        # 线性 tile id -> 逻辑坐标
        # tile_view/strided copy 取当前 tile
        # 调用 Cube/Vector 推进流水

    @jit
    def run(...):
        # 薄封装：校验、scratch、launch
```

## 从已有 AscendC 算子迁移到 CANNBotDSL

迁移目标是“语义和公开调用接口对齐”，不是把原工程的 host scratch 和注册尾参暴露给 DSL 调用方。

推荐步骤：

1. **冻结公开调用接口**
   - 列出真实输入、真实输出和标量属性。
   - 标明 dtype 变化、layout 变化、shape 约束和错误条件。
   - `workspaceGM`、`tilingGM`、scratch、结构化辅助张量、临时中间结果不进入 `run()` 参数。

2. **按参考结构组织文件**
   - `SHAPE_LIST` 使用用户可见 shape。
   - `run()` 是薄封装，只 launch kernel，不构造大批 host 临时 tensor。
   - 类按硬件角色、数据流角色和调度角色拆分，避免把计算、搬运、shape 适配、golden 混在同一个对象里。
   - 如果用户要求参考某个实现风格，必须先对照参考文件列出可迁移的结构模式：类职责、构造参数、buffer 申请方式、copy engine 归属、执行边界、shape 参数处理。不要只复制局部 API 调用。

3. **kernel 内按 base block 实现**
   - 内部算法需要 base block、chunk、tile、padding 或特殊布局时，在 kernel 内用 UB/L1/L0 buffer 和 `tile_view`/`local_slice` 表达。
   - 结构化辅助张量、边界标记、稀疏/三角/窗口等规则尽量在 Vector 内部生成或用局部 buffer 表达。
   - 不要把内部高维 scratch 当成外部 tensor 传入。
   - 对来自参考实现的 L1 resident、TSCM、L0 double buffer、RHS 预取路径，单块映射为 Buffer，多级/同步路径映射为 Channel + copy engine。无法由 Channel 表达的旧手动多级方案标为不支持，不要直接退化成 GM workspace 或伪造 API。

4. **先调度后性能**
   - 先做一版可解释、可 translate、可 smoke 的实现。
   - 混合 AIC/AIV 的第一段 AIC 工作必须能独立开始，不能等待 AIV 后面才会生产的 L1。
   - 若原算法第一步需要 Vector 预处理后再 Cube matmul，优先重写等价公式，让第一拍 Cube 从 GM 直接启动，再由 Vector 对 Cube 结果做缩放、边界或结构化规则处理。
   - “先跑通”不能作为破坏架构的理由。若为了工具链限制临时使用 GM，必须保持 shape 命名正确、职责边界正确，并在同一提交中留下 on-chip 化 todo。

5. **逐步验证**
   - `py_compile`。
   - 编译通过（AOT compile），检查 public entry 和 wait/set flag 的源码结构。
   - zero smoke，期望零输入输出可解释。
   - weak random correctness，再放大 shape 或优化 Cube 路径。

## 混合 AIC/AIV 调度经验

- AIC 和 AIV 是同一 kernel 内两个分支，源码顺序不等于“先跑完 AIV 再跑 AIC”。如果 AIC 分支开头 wait 一个 AIV 后面才 set 的 token，可能直接死锁。
- 通用流水原则：生产者先处理自己能从 GM/L1 独立取得的数据，消费者只 wait 已经被前序逻辑保证会生产的 token。有自然延迟或 drain 时，才适合跨 pipe handoff。
- C2V 的 L0C→UB handoff 要初始化 Vector 侧可读写 token，AIC FIXPIPE wait 后写 UB，再 set ready，AIV wait 后消费并 release。
- V2C 的 UB→L1 handoff 要初始化 Cube 侧 free token，AIV MTE3 wait free 后写 L1，再 set ready，AIC MTE1 wait ready 后读 L1，再 release free。
- 同一个 sync id 不要跨多个不相关 buffer 复用；`id+16` 只用于明确的第二 subblock 或同一 handoff 的成对 token。
- GM handoff 也按同样规则处理：producer 写完 GM scratch 后发 ready，consumer wait ready 后才能读；release/free token 只表达复用许可，不表达数据已经生产。
- 如果 Cube 等两个 AIV subblock 的 GM ready，Vector 生产侧必须分别发布 base 和 `base+16`；如果两个 Vector subblock 消费 Cube/FIXPIPE 结果，也必须分别等待 base 和 `base+16`。
- 如果同一阶段先 `SetFlag` 再立即 `WaitFlag`，要先判断它是否消费了初始化 token，而不是等待上游真实 producer。
- barrier、notify/wait 和 release 链只在确认没有真实消费者后才能删除；保护 scratch 或 GM 可见性的同步点不能因为当前小 shape 不 hang 就视为多余。
- `tile_view`/`local_slice` view 可能没有稳定同步标识。同步锁 owning buffer，view 只作为 `mem_copy`/vector op 的 operand。

## 参数和 layout 操作模式

- 对 `B,S,N,D` 输入，优先在 kernel 内映射：

  ```text
  tile_id -> batch_idx, head_idx, seq_tile_idx
  gm tile -> input[batch_idx, :, head_idx, :] 的当前 seq tile
  ```

- 对变长或扁平输入，优先使用：

  ```text
  [T,N,D] + actual_seq_lengths
  seq_start/seq_end/chunk_group 在 kernel 内推进
  ```

- 对需要内部 chunk workspace 的融合算子，推荐：

  ```text
  外部输入: public layout
  当前 tile 读取: public layout + stride/tile_view
  内部 workspace: compute-friendly layout
  输出写回: public layout
  ```

- 不推荐：

  ```text
  run() 里把完整输入整体 reshape/transpose/copy 成内部 layout
  kernel 只认识内部 layout
  测试只覆盖内部 layout
  ```

## 注释和测试

- 注释使用中文，解释 layout 映射、同步不变量、buffer 复用、特殊 API 限制；不要逐行解释显而易见的赋值。
- 分支注释说明真实生产/消费链，例如某个分支生产 GM/UB 基底，另一个分支等待 ready 后做矩阵更新、写回或累加。不要用“参考某实现”“对齐某实现”替代数据流说明。
- VF 注释要写公式和 dtype 变化，例如 `output_low = cast_low(compute_high(input_low, scale_high))`，具体变量名按当前算子公式填写。
- `SHAPE_LIST` 使用公共接口 shape，至少包含小 smoke shape 和目标代表 shape。
- golden 可以用 PyTorch reshape/transpose 表达数学语义，但 kernel 入参和输出必须保持公共接口。
- 精度和性能验证要分开：先最小 correctness，再目标 shape，再 profiling。

## 修复模式

- 如果已有代码用内部 layout 作为公共入口，先补一层真正的公共接口测试，再把 kernel 内 tile mapping 改到公共 layout。
- 如果类职责混乱，先拆 `Cube`/`Vector`/`BlockInfo`，保持核心计算不变，再移动 buffer 申请到对应类。
- 如果 host wrapper 很重，逐步把整体 layout 转换替换为 kernel 内 tile 级读取和写回。
- 如果必须保留内部入口用于调试，命名为 `run_internal_*`，公共 `run()` 不应暴露内部 chunk 参数。

## 代码自检

提交或交付前做一次面向架构的自检：

- 搜索 `Cube(ctx)`、`Vector(ctx)`、`self.ctx`：除非是临时迁移文件，否则说明职责边界没有拆干净。
- 搜索 `gm_*scratch`：每个 GM scratch 都必须能在 GM 例外表中找到理由；否则改成 UB/L1/L0。
- 搜索形如 `axis_a // axis_b`、`*_tile_num` 的切分常量：确认这是实现分块，不是算法公式；变量名必须带真实逻辑轴语义。
- 搜索带形状暗示的变量名和固定 tile shape：确认命名和真实逻辑轴角色一致，不是因为尺寸相同而复用错误语义。
- 搜索 `UBBuffer(`、手动地址和区域常量：确认只有别名复用或特殊生命周期才使用，并有中文注释。
- 检查 copy engine 是否归属 Cube/Vector，而不是由顶层 kernel 创建后随处透传。
- 检查短生命周期算法临时量、迭代临时量、局部累加和结构化辅助张量是否常驻 L1/UB/L0；若落 GM，必须有容量或工具链限制说明。
- 搜索 `lhs` / `rhs` / `tmp` / `problem` / `参考` / `对齐`：确认命名和注释不是来源说明或代数占位，而是当前公式、输入集、buffer 生命周期的真实语义。通用库可使用代数名，具体算子文件不应滥用。
- 搜索 `make_copy_engine(dtype=dtypes.float32, eltwise_op="add")`：确认 GM 目标 dtype、atomic setter 和最终输出 dtype 已由单测覆盖；不要把 engine dtype 误写成最终 GM dtype。
