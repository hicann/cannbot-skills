---
name: cannbotdsl-kernel-structure
description: "把一个已定架构的 CANNBotDSL 算子落成具体的 kernel 代码骨架时使用——不是决定'算法怎么算'（那是 op-design/cv-fusion），而是决定'代码怎么摆'：如何按计算流程把 kernel 拆成 cube 模块类 + vec 模块类 + 顶层编排类三层，每个模块暴露哪些方法（load_/compute_/store_ 命名范式、@jit 何时加），每个 Channel 声明在哪一层（核内 staging channel 归模块类、跨核 handoff channel 归顶层类、vec 内部状态用裸 scratch 还是 Channel），以及多 stage 软件流水的派发循环怎么写（stage-gated `if global_idx>=k` 主循环 + drain 尾循环 + DelayLineGroup 延迟索引 + 主循环/drain stage body 必须逐字一致）。当需要：从算子计算流程反推 cube/vec 模块边界、给模块方法定命名与粒度、决定某个 Channel 该声明在模块类还是 kernel 类、把 N 段 stage 摆成 warmup/steady/drain 的派发骨架、或 review 已有 kernel 的结构分层是否合理时触发。含三层拆分总则、模块划分判据、Channel 归属规则表、多 stage 派发骨架模板、结构陷阱清单。Triggers: kernel 代码结构, kernel 骨架, 代码分层, cube 模块划分, vector 模块划分, 模块边界, Matmul 类, Vector 类, 模块方法命名, channel 声明位置, channel 归属, 核内 channel, 跨核 channel 放哪, stage 划分, 多 stage 流水代码, stage-gated loop, global_idx, drain 尾循环, DelayLineGroup 派发, 结构 review。注意：算法归类/tiling/字节预算归 cannbotdsl-op-design；stage-graph 抽象架构决策/跨核 sync 预算归 cannbotdsl-cv-fusion；Channel 4 相协议机制归 cannbotdsl-channel；核内 PIPE 重叠归 cube-pipeline/vec-pipeline；preload_num≥3 的深流水 storage 归 cannbotdsl-channel 与 cannbotdsl-perf-optimize；@jit/@kernel 语言语义与最小骨架归 cannbotdsl-programming-model；完整 FA 变体归 cannbotdsl-flash-attention。Developer 在 Stage 3 开写代码前调用，把设计翻译成类/方法/派发骨架。"
---

# cannbotdsl-kernel-structure

CANNBotDSL 算子的**代码结构范式层**。输入是 `../cannbotdsl-cv-fusion/SKILL.md` §1 已画好的 stage-graph（哪几段在哪个核、handoff 拓扑、depth 预算），输出是**具体的 Python 类/方法骨架**：三层怎么分、方法怎么命名、每个 Channel 声明在哪一层、多 stage 派发循环怎么摆。

**本 skill 只管"代码怎么摆"，不管"算法怎么算"。** 算法归类 / tiling / 字节预算 → `../cannbotdsl-op-design/SKILL.md`；stage-graph 与跨核 sync 预算 → `../cannbotdsl-cv-fusion/SKILL.md`；Channel 原语语义 → `../cannbotdsl-channel/SKILL.md`。Developer 在 Stage 3 开写前调用，把设计翻译成骨架。

## 0. 使用前提

- 已有 stage-graph（cv-fusion §1 的链式速记 + 阶段节点表 + 边表）。纯 Vec / 纯 Cube 单核算子结构简单，直接看 `../cannbotdsl-programming-model/SKILL.md` 的最小骨架即可，不必进本 skill。
- 本 skill 的范例是**通用 composite CV 算子**：`O = f(A@B) @ C` 形态——一次 cube 矩阵乘（产出 S）→ 一段 vec 后处理（对 S 做归约/逐元素，产出 P）→ 第二次 cube 矩阵乘（P@C 产出 T）→ 一段 vec 收尾（产出 O）。即 stage-graph 的 `CVCV` 拓扑。单跳 `CV` 是它的退化。

## 1. 三层拆分总则

一个 CV 算子的代码**恒定分三层**，职责互不重叠：

| 层 | 是什么 | 装什么 | 不装什么 |
|----|--------|--------|----------|
| **Cube 模块类** | 一个普通 class（如 `CubeMM`） | 所有矩阵乘 + L0C→UB 的 fixpipe 搬出；核内 staging channel（L1/L0A/L0B/L0C）声明在此 | 跨核 handoff channel、vec 计算、派发流程 |
| **Vec 模块类** | 一个普通 class（如 `VecPost`） | 所有逐元素/归约/mask/cast；vec 内部状态与 scratch 声明在此；vec 产出的同核 channel | 跨核 handoff channel、matmul、派发流程 |
| **编排层（kernel 类）** | `@kernel` class | 跨核 handoff channel 声明在此；实例化两个模块；`__call__` 里写多 stage 派发循环；多核 dispatch / 循环坐标推导 | 具体算子计算（下沉到模块方法） |

**核心原则**：模块类是"能力提供者"（无状态编排、只暴露单步方法），编排层是"流程组织者"（决定这些单步按什么顺序、什么延迟发射）。**计算下沉到模块方法、流程上提到编排层**——这样同一对模块类可被不同派发策略（无流水 / DB / 深流水）复用，改流水深度不动模块代码。

## 2. 从计算流程划模块

### 2.1 划分判据（逐段问一句）

沿 stage-graph 的链，每个计算段问："**这一步的主导操作落在哪个核？**"

- 有 `matmul`（L0A×L0B→L0C 累加）或 fixpipe 搬运 → **进 cube 模块**。
- 逐元素 / 归约 / mask / select / cast → **进 vec 模块**。
- 灰色地带（`cast`/`muls`/`scale` 这类轻算子）：能在 fixpipe drain 时顺带做的别单开 vec 段——这是 cv-fusion §2 的分工决策，**结构上体现为"把它并进相邻模块的某个方法，而不是新开一个方法"**。

### 2.2 方法命名范式（一个方法 = 一个 PIPE 相干单步）

每个模块方法只做**一个 PIPE 相干的动作**，粒度对齐 stage-graph 的一个子步。命名用动词前缀表意图：

| 前缀 | 语义 | 典型体 | 加 `@jit`？ |
|------|------|--------|:-----------:|
| `load_*` | GM→L1 搬入（MTE2，channel-first produce） | `mem_copy(self.x_l1, gm, engine=nd2nz)` | 否 |
| `compute_*` | L1→L0 + matmul | `mem_copy(l0a, x_l1); matmul(l0c, l0a, l0b)` | 否 |
| `store_*` | L0C→UB fixpipe（跨核 produce） | `mem_copy(handoff_ub, self.l0c, engine=fixpipe)` | 否 |
| `<algo>_*`（vec） | 一段 vec 算法（softmax/norm/…） | 含 `with vf(mode=...)` 的逐元素/归约 | **是**（体内有 raw-vf `for`） |
| `init_*` / `update_*`（vec） | 跨迭代累加器的首次 / 后续 | `res = init` / `res = res*k + new` | 视体内有无 vf `for` |
| `finalize_*`（vec） | 收尾归一化 + cast + 写出 GM | vf 除法 → `cast` → `mem_copy(gm, ub)` | 拆：vf 部分单独 `@jit`，channel/GM 写在外层 |

**`@jit` 铁律**：只要方法体内有 **raw-vf 的 `for` 循环**，该方法必须直接是 `@jit` 函数体——raw-vf 的 `for` 不能藏进普通 helper，否则被 AST 全展开、VF 折叠严重退化。

> **真实 NPU 实测数据（910B, shape 1×16×4096×4096×128, int8-QK FA kernel）**：含 raw-vf `for` 的 Vector 方法不加 `@jit` → **aiv_vec_time = 2,935 us**；加上 `@jit` → **420 us**。差距 **~7×**，原因是缺少 `@jit` 时 VF 编译器只能在庞大的 `@kernel` 上下文中做有限折叠，而 `@jit` 让每个方法成为独立编译单元、VF 深度折叠。**排查 vec 性能问题时，第一步就检查所有含 `with vf(mode="raw"):` 的方法是否都有 `@jit`。**

所以 `finalize_*` 常拆成"内层 `@jit` vf 段 + 外层普通包装写 channel/GM"两块。纯 channel-first 的 `mem_copy`/`matmul` 步（`load_`/`compute_`/`store_`）不加 `@jit`。

### 2.3 模块骨架模板

```python
class CubeMM:                          # cube 能力提供者
    def __init__(self, tile_m, tile_n, tile_k):
        self.nd2nz  = make_copy_engine(format_transform="nd2nz", dtype=dtypes.float16)
        self.fixpipe = make_copy_engine(dtype=dtypes.float32, dual_dst_ctl=1)
        # ── 核内 staging channel 声明在模块类（§3 归属规则）──
        self.a_l1 = Channel(MemLoc.L1, shape=(tile_m, tile_k), dtype=dtypes.float16, depth=2)
        self.l0a  = Channel(MemLoc.L0A, shape=(tile_m, tile_k), dtype=dtypes.float16, depth=2)
        self.l0b  = Channel(MemLoc.L0B, shape=(tile_k, tile_n), dtype=dtypes.float16, depth=2)
        self.l0c  = Channel(MemLoc.L0C, shape=(tile_m, tile_n), dtype=dtypes.float32, depth=2)

    def load_a(self, a_gm):                              # GM→L1
        mem_copy(self.a_l1, a_gm, engine=self.nd2nz)
    def compute_s(self, b_l1):                           # A@B→L0C
        mem_copy(self.l0a, self.a_l1); mem_copy(self.l0b, b_l1)
        matmul(self.l0c, self.l0a, self.l0b, init=True)
    def store_s(self, handoff_ub):                       # L0C→UB（跨核 produce，编排层传入 channel）
        mem_copy(handoff_ub, self.l0c, engine=self.fixpipe)

class VecPost:                         # vec 能力提供者
    def __init__(self, tile_vm, tile_n, subblock_idx):
        # ── vec 内部状态：跨迭代累加器用裸 scratch（§3）──
        self.res = Buffer(MemLoc.UB, (tile_vm, tile_n), dtypes.float32)
        # ── vec 产出的同核 channel 声明在模块类 ──
        self.o_ub  = Channel(MemLoc.UB, shape=(tile_vm, tile_n), dtype=dtypes.float16, depth=1)
        self.subblock_idx = subblock_idx

    @jit
    def post(self, s_ub, ...):                           # 一段 vec 算法（含 raw-vf for）
        with vf(mode="raw"):
            for row in range(...): ...
    def finalize(self, o_gm):                            # 拆：vf 段 @jit，写出在外层
        self._finalize_vf()                              # 内层 @jit
        cast(self.o_ub, self.res)
        mem_copy(tile_view(o_gm, ..., (self.subblock_idx, 0)), self.o_ub)
```

## 3. Channel 声明位置（归属规则）

**每个 Channel 声明在哪一层，由它连接谁决定**——这是本 skill 最易错、最该记的一张表：

| Channel 用途 | 声明在 | 判据 | 例 |
|--------------|:------:|------|-----|
| **核内 staging**（GM→L1→L0→L0C 的中转） | **对应模块类** `__init__` | 只被本模块的方法读写，外界看不见 | `CubeMM.a_l1 / l0a / l0b / l0c` |
| **vec 产出的同核 channel**（vec 写 → 同核 GM/后续消费） | **vec 模块类** `__init__` | 生产消费都在 vec 核内 | `VecPost.o_ub` |
| **vec 内部跨迭代状态 / scratch** | **vec 模块类**，用 `Buffer(MemLoc.UB, ...)` | 无 ring / 无 sync 需求的驻留累加器、lane 向量 | 累加器 `res`、softmax 的 max/sum/exp 表 |
| **跨核 handoff**（cube↔vec 交接，`kind=CrossCore`） | **顶层 kernel 类** `__init__`，作参数传进模块方法 | 是两模块共享的边，谁都不该独占；且 CrossCore sync 预算须全局核算 | `qk_ub`（C→V）、`v2c_l1`（V→C）、`pv_ub`（C→V） |

**两条推论**：

1. **跨核 channel 绝不声明在模块类里。** 模块方法通过**参数**接收它（`store_s(self, handoff_ub)` / `compute_t(self, v2c_l1)`），编排层持有并传入。理由：① 它连接两个模块，归属任一模块都破坏"模块只管自己核内"的封装；② `Σ(CrossCore depth) ≤ 8` 是**全局 per-kernel 预算**（`../cannbotdsl-cv-fusion/SKILL.md` §4.1），必须在能看到所有 handoff 的编排层集中声明才数得清。

2. **Buffer vs Channel 的选择**：跨迭代要 ring/sync 语义（生产者-消费者错开）→ Channel；只是驻留的累加器/临时 lane 向量、由同一段 vf 读改写 → Buffer（无 buf_id、无 sync 开销）。选择细则见 `../cannbotdsl-channel/SKILL.md` §5。

> Channel 一律用 typed 声明（给完整整块 tile 的 `shape`/`dtype` + `depth`）——这是**布局层**的写法，正交于本 skill 的"声明在哪一层"。尾块由框架在使用点自动插 `local_slice` 视图，一份编译产物仍可跑多组动态 (M,K,N)。规则见 `../cannbotdsl-channel/SKILL.md` §1。

> 核内 staging Channel 的 buf_id/sync 由框架自动分配、只服务本模块，故归模块类；跨核 Channel 归顶层统一核算。Buffer 只用于无同步的单块 scratch，按实际使用者归属。旧手动 NBuffer/L1 buf_id 复用模型已不受支持。

## 4. 多 stage 流水的派发骨架

编排层 `__call__` 把 N 个 stage 摆成**软件流水**：稳态时 cube 算 tile n+1 的 Stage 0，vec 同时算 tile n 的 Stage 1……用一个单调计数器 + 延迟线实现。**这是"结构范式"；`preload_num≥3` 的 storage 必须用 `Channel(..., depth=N)`，用法见 `../cannbotdsl-channel/SKILL.md`。需要 `%N` 手动随机槽且无法改写为 Channel FIFO 的旧方案当前不支持。**

### 4.1 三个结构件

1. **`global_idx`** 单调计数器：贯穿整个迭代空间，`stage k` 只在 `global_idx >= k` 时才发射（= warmup 自动填充：前 k 次迭代该 stage 空转）。
2. **`DelayLineGroup`**：只存**推导每个 stage 所需的最小循环坐标**（如 `tile_idx`、`n_idx`），不存中间数据。`stage k` 用 `dl.<field>.tap(k)` 取"k 拍之前"的坐标，再 `idx2crd` 还原出 batch/head/m 等派生索引。
3. **drain 尾循环**：主循环结束后，流水里还有 `preload_num` 个 tile 没排空，用 `for remain_idx in range(preload_num)` 补发 stage 1..N-1（`if remain_idx < k and global_idx >= k`）。

### 4.2 派发骨架模板

```python
@kernel
class my_kernel:
    def __init__(self, ...):
        self.preload_num = 3
        # ── 跨核 handoff channel 声明在编排层（§3 归属规则）──
        self.c2v_ub = Channel(MemLoc.UB, ..., depth=2, kind=ChannelKind.CrossCore)
        self.pv_ub  = Channel(MemLoc.UB, ..., depth=2, kind=ChannelKind.CrossCore)
        self.v2c_l1 = Channel(MemLoc.L1, ..., depth=3, kind=ChannelKind.CrossCore)
        self.cube = CubeMM(...)
        self.vec  = VecPost(..., self.subblock_idx)

    def __call__(self, out, a, b, c, ...):
        dl = DelayLineGroup(self.preload_num + 1, 'tile', 'n')
        global_idx = 0
        for tile_idx in range(start, start + tiles_per_block):
            # 每个 m-tile 只做一次的搬入（load-once，channel-first read-many）
            self.cube.load_a(tile_view(a, ..., (m_idx, 0)))
            for n_idx in range(n_end):
                # ── Stage 0: 第一次 matmul ──
                if global_idx >= 0:
                    dl.push(tile=tile_idx, n=n_idx)
                    self.cube.load_b(tile_view(b, ..., (n_idx, 0)))
                    self.cube.compute_s()
                    self.cube.store_s(self.c2v_ub)              # C→V handoff 传入
                # ── Stage 1: vec 后处理 ──
                if global_idx >= 1:
                    t, n = dl.tile.tap(1), dl.n.tap(1)          # 取延迟坐标
                    ... = idx2crd(t, [...])                     # 还原派生索引
                    self.vec.post(self.c2v_ub, ...)
                    self.vec.store_p(self.v2c_l1)               # V→C handoff
                # ── Stage 2: 第二次 matmul ──
                if global_idx >= 2:
                    t, n = dl.tile.tap(2), dl.n.tap(2)
                    self.cube.load_c(...); self.cube.compute_t(self.v2c_l1)
                    self.cube.store_t(self.pv_ub)
                # ── Stage 3: vec 收尾 ──
                if global_idx >= 3:
                    t, n = dl.tile.tap(3), dl.n.tap(3)
                    ...  # init_o / update_o；末 tile 再 finalize_o
                dl.advance(); global_idx += 1
        # ── drain: 排空剩余流水 ──
        for remain_idx in range(self.preload_num):
            if remain_idx < 1 and global_idx >= 1: <Stage 1 逐字重复>
            if remain_idx < 2 and global_idx >= 2: <Stage 2 逐字重复>
            if remain_idx < 3 and global_idx >= 3: <Stage 3 逐字重复>
            dl.advance(); global_idx += 1
```

### 4.3 主循环 / drain 的 stage body 必须逐字一致

drain 里的 Stage 1/2/3 是主循环对应 stage 的**逐字复制**（同样的 `tap(k)` + `idx2crd` + 模块方法调用序列）。**这是此范式已知的结构冗余**：两处不同步会造成排空阶段行为偏差、且不报错。缓解——把每个 stage body 抽成编排层的一个私有方法（`self._stage1(global_idx)`），主循环和 drain 都调它，消除逐字复制。若坚持内联（如深流水里 stage body 依赖大量局部量），则务必**成对修改**并在 review 时逐 stage 对照（结构 review 见 §6）。

## 5. 组装顺序（从 stage-graph 到骨架）

1. **抄 stage 节点表** → 每个 cube 节点在 `CubeMM` 加一个 `compute_*`/`store_*`，每个 vec 节点在 `VecPost` 加一个 `<algo>_*`。
2. **抄边表** → 每条**前向跨核边**在编排层 `__init__` 声明一个 `CrossCore` Channel；核内边的 staging channel 落进对应模块类。
3. **定 `preload_num`** → = 链上前向跨核跳数（`CVCV` = 3 拍延迟 → `preload_num=3`）；`DelayLineGroup(preload_num+1, ...)`。
4. **摆派发** → 按 §4.2 模板，stage k 用 `if global_idx >= k` 门控 + `tap(k)` 取延迟坐标 + 传入对应 handoff channel。
5. **写 drain** → 复制 stage 1..N-1（或调用抽出的 `_stage_k`）。
6. **多核 dispatch / tail block** → `block_idx` 切迭代空间、`get_subblock_id()` 定 split-M 半区（`../cannbotdsl-programming-model/SKILL.md`）。

## 6. 结构陷阱（review 清单）

1. **跨核 channel 声明进了模块类** → sync 预算数不清 + 破坏封装。跨核 handoff 一律上提编排层，模块方法用参数收（§3 推论 1）。
2. **一个方法塞多个 PIPE 动作**（load+compute+store 揉一起）→ 派发层无法在 stage 间错位、流水失效。一方法一 PIPE 相干单步（§2.2）。
3. **raw-vf `for` 藏进普通 helper**（没 `@jit`）→ AST 全展开、VF 折叠严重退化——实测 int8-QK FA 场景 **~7× vec 性能损失**（2,935 us → 420 us, §2.2 铁律）。这是 VF 性能第一优先级检查项：排查 vec 慢的 kernel 先确认每个含 `with vf(mode="raw"):` 的方法是否标注 `@jit`。
4. **主循环与 drain 的 stage body 不一致** → 排空阶段静默错。抽私有方法或成对改（§4.3）。
5. **该用裸 scratch 的累加器声明成了 Channel**（或反之）→ 多余 buf_id/sync 开销，或缺 ring 语义致数值错。按 §3 表选（细则 channel §6）。
6. **计算逻辑写进编排层 `__call__`** → 模块无法复用、流水策略与算法耦死。计算下沉模块方法，`__call__` 只组织流程（§1）。
7. **`preload_num` 与链上跨核跳数不符** → 延迟线 tap 深度错位，读到未就绪 / 已覆盖的 handoff slot。`preload_num` = 前向跨核跳数（§5.3）。

## 注意（反触发 / 下沉边界）

- 算法归类 / tiling 推导 / 字节 · buf_id 预算 → `../cannbotdsl-op-design/SKILL.md`。
- stage-graph（哪段在哪核、handoff 拓扑、跨核 sync 预算、融/拆决策）→ `../cannbotdsl-cv-fusion/SKILL.md`（本 skill 的**上游输入**）。
- Channel 4 相协议 / channel-first vs 四原语 / typed 声明与尾块自动推断 / SameCore·CrossCore → `../cannbotdsl-channel/SKILL.md`。
- 核内 PIPE 重叠（cube MTE2/MTE1/M/FIXPIPE；vec MTE2/V/MTE3 + vf 折叠）→ `../cannbotdsl-cube-pipeline/SKILL.md` / `../cannbotdsl-vec-pipeline/SKILL.md`。
- `preload_num≥3` 深流水的 Channel depth / warmup·steady·drain 实现 → `../cannbotdsl-channel/SKILL.md` + `../cannbotdsl-perf-optimize/SKILL.md`（本 skill 只给派发骨架的**形状**）。
- `@jit`/`@kernel` 语义、trace-time、Buffer/Channel、`DelayLineGroup`/`idx2crd`/`tile_view` 原语、最小 kernel 骨架 → `../cannbotdsl-programming-model/SKILL.md`。
- 完整 Flash Attention 变体移植 → `../cannbotdsl-flash-attention/SKILL.md`。

## 参考

- 上游 / 下沉：`../cannbotdsl-cv-fusion/SKILL.md`（stage-graph）、`../cannbotdsl-channel/SKILL.md`（Channel 与深流水 storage）、`../cannbotdsl-perf-optimize/SKILL.md`（流水实现与验证）、`../cannbotdsl-programming-model/SKILL.md`（语言语义与原语）
- `references/kernel-coding-style.md`（三层职责分离、L1 归属、VF 封装 owning UB、buf_id 复用、host 精简的编码风格清单）
