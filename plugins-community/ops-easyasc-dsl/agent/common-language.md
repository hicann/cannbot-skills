# Common Language

This file is the fixed terminology baseline loaded once after `agent/ROUTER.md`
in every new conversation. It owns definitions and Chinese/English aliases, not
device limits, implementation behavior, patterns, or workflow rules. Follow the
linked owner when a definition is not enough.

Keep this baseline at no more than 250 lines and 15 KiB. Move detailed facts or
recipes to their focused owner instead of expanding this glossary.

## 1. Device aliases

| Alias | Meaning | Chinese / repository mapping |
| --- | --- | --- |
| `A2`, `b3`, `b*` | Ascend A2 target family | 昇腾 A2; public facade `easyasc/a2.py`; C220 codegen family |
| `A3`, `910C`, `9362` | Ascend 910C `Ascend910_9362` target | 昇腾 910C; public facade `easyasc/a3.py`; C220 codegen with compile unit `ascend910_93` |
| `A5`, `950` | Ascend 950 target | 昇腾 950; public facade `easyasc/a5.py`; C310 codegen family |
| `A5PR`, `950pr` | Ascend 950 PR target | 昇腾 950PR; public facade `easyasc/a5pr.py`; C310 codegen family |
| `C220` | A2/A3 device-code family tag | A2/A3 cube macro family |
| `C310` | A5/A5PR device-code family tag | 950 cube macro family |

Capacities, core counts, and target differences belong to
`agent/references/facts-device-runtime.md`.

## 2. Memory hierarchy

| Term | Meaning | Chinese |
| --- | --- | --- |
| `GM` | off-chip global memory and public tensor storage | 全局内存 |
| `L1` | cube-side local staging memory | 一级本地内存 |
| `L0A`, `L0B` | left/right cube operand buffers | 矩阵操作数缓冲 |
| `L0C` | cube accumulator buffer | 矩阵累加器缓冲 |
| `L0AMX`, `L0BMX` | A5 MX scale buffers paired with L0 operands | MX 专用 L0 缓冲 |
| `UB` | vector-side unified buffer | 统一缓冲区 |
| `BT`, `C2` | matmul bias-table memory | 偏置表缓冲 |
| `workspace` | kernel-private GM scratch, not a public output | 工作空间 |

Resource sizes belong to `agent/references/facts-device-runtime.md`; allocation
and layout rules belong to `agent/references/facts-authoring.md`.

## 3. Pipeline sides and pipes

| Term | Meaning | Chinese |
| --- | --- | --- |
| cube side | matrix/load/fixpipe execution side | 矩阵侧 |
| vec side | vector or VF execution side | 向量侧 |
| `MTE2` | load pipe | 加载引擎 |
| `MTE1` | L1-to-L0/BT pipe | L1 到 L0/BT 搬运引擎 |
| `M` | matrix-compute pipe | 矩阵计算管道 |
| `FIX` | cube writeback/fixpipe | 写回管道 |
| `MTE3` | vector-side store pipe | 存储引擎 |
| `V` | vector-compute pipe | 向量计算管道 |
| `S` | scheduling/control stream | 调度流 |
| lookahead | producer issues future tiles before the matching consumer | 前瞻调度 |
| pipeline bubble | idle interval within an active pipeline | 流水线空泡 |
| pipe occupancy | modeled active cycles divided by a stated time span | 管道占用率 |

Exact op-to-pipe mapping belongs to `agent/references/facts-device-runtime.md`.
Scheduling patterns belong to `agent/references/pattern-index.md` and
`agent/references/optimization/levers.md`.

## 4. Buffer and scalar types

| Term | Meaning | Chinese |
| --- | --- | --- |
| `Tensor` | one local tensor allocation | 本地张量 |
| `DBuff` | two-slot rotating local buffer | 双缓冲 |
| `TBuff` | three-slot rotating local buffer | 三缓冲 |
| `QBuff` | four-slot rotating local buffer | 四缓冲 |
| `GMTensor` | one GM tensor and kernel ABI value | 全局张量 |
| `GMTensorList` | dynamic same-dtype, same-rank list of GM tensors | 动态全局张量列表 |
| `Var` / `Expr` | runtime symbolic scalar/expression | 运行时变量/表达式 |
| `Reg`, `RegList` | A5 VF register value/list | 寄存器/寄存器列表 |
| `MaskReg` | A5 VF lane-selection register | 掩码寄存器 |

Public object semantics belong to `doc/api/tensor_buffer.md`,
`doc/api/register.md`, and `doc/api/op_exec.md`.

## 5. Tiling terms

| Term | Meaning | Chinese |
| --- | --- | --- |
| `TILE_M`, `TILE_N`, `TILE_K` | local tile sizes along matrix axes | M/N/K 维分块大小 |
| `valid_m`, `valid_n`, `valid_k` | live elements in a boundary tile | 尾块有效元素数 |
| tail / tail tile | final tile that is not full-sized | 尾块 |
| `split_m`, `split_n`, `mix` | surrounding kernel's cross-core ownership topology | M/N/二维核间切分 |
| `splitk` | shortcut tiling of one core's K loop | 单核 K 维分块 |
| `splitn` | shortcut tiling of one core's N loop | 单核 N 维分块 |
| `nosplit` | one shortcut matmul tile for the supplied region | 不切分 |
| `is_init` | first K tile overwrites/initializes L0C | 初始化分块 |
| accumulate tile | later K tile adds into existing L0C | 累加分块 |

`splitk` and `splitn` do not create cross-core work or a merge. Hard legality
rules belong to `agent/references/facts-authoring.md`; tail rules belong to
`agent/references/constraints/tail.md`.

## 6. Layout and transfer terms

| Term | Meaning | Chinese |
| --- | --- | --- |
| `ND` | dense row-major layout | 稠密行主序 |
| `NZ` | cube fractal layout | NZ 分形布局 |
| `ZZ`, `ZN`, `NN` | hardware fractal variants/decodes | ZZ/ZN/NN 分形变体 |
| `nd2nz`, `nz2nd` | dense-to-fractal / fractal-to-dense conversion | 稠密与分形转换 |
| `reinterpret` | zero-copy dtype view | 零拷贝类型重解释 |
| burst | one contiguous datamove unit | 突发搬运单元 |
| `n_burst` | number of datamove bursts | burst 次数 |
| `burst_len` | payload per burst | burst 长度 |
| stride / step | gap between consecutive burst starts | burst 步长 |
| carrier | storage dtype whose bits encode another logical dtype | 载体类型 |
| `pack4`, `unpack4` | compact/expand four 8-bit VF lanes per 32-bit word | 四合一打包/一分四解包 |

Layout mechanics belong to `doc/api/tensor_buffer.md` and cube/datamove API
pages. Device-specific restrictions belong to the focused constraint pages.

## 7. Synchronization terms

| Term | Meaning | Chinese |
| --- | --- | --- |
| signal / token | one published synchronization credit | 信号/令牌 |
| ready | consumer released capacity for reuse | 可复用/容量信号 |
| valid | producer published completed data | 数据有效信号 |
| `SEvent`, `DEvent`, `TEvent`, `QEvent` | one/two/three/four-credit event families | 单/双/三/四槽事件 |
| `CvMutex` | reusable cube-to-vec ownership handoff | Cube 到 Vec 互斥量 |
| `VcMutex` | reusable vec-to-cube ownership handoff | Vec 到 Cube 互斥量 |
| `auto_sync` | automatic same-side dependency insertion | 自动同步 |
| barrier / `bar_*` | wait for selected pipe work to retire | 管道屏障 |
| preset | initial event credit available before the first producer | 预置信号 |

Protocols and legal pipe pairs belong to `agent/references/constraints/sync.md`
and `doc/api/synchronize.md`.

## 8. Device-specific authoring terms

| Term | Meaning | Chinese |
| --- | --- | --- |
| `@vf()` | A5 vector-function decorator | A5 向量函数 |
| micro | A5 register-level VF operation family | 寄存器级计算 |
| `@simt` | A5 SIMT escape hatch for irregular work | SIMT 线程并行模块 |
| MX / MXFP8 / MXFP4 | A5 block-scaled low-precision format families | 微缩放低精度格式 |
| e8m0 | unsigned exponent-only MX scale encoding | 8 位指数缩放因子 |
| sub-block | one of two A2 vector lanes per cube core | A2 向量子块 |
| cube-to-vec bridge | handoff from L0C output to vec-side UB | Cube 到 Vec 数据桥 |
| vec-to-cube bridge | handoff from UB output to cube-side L1 | Vec 到 Cube 数据桥 |

A5 operation details belong to `agent/references/constraints/a5.md` and the A5
micro API pages. A2 bridges belong to
`agent/references/patterns/a2-mixed-pipeline.md`.

## 9. Compiler and runtime terms

| Term | Meaning | Chinese |
| --- | --- | --- |
| IR | instruction intermediate representation | 中间表示 |
| lowering | conversion from IR to generated C++ | 下降/代码生成 |
| pruning | dead or side-inapplicable instruction removal | 剪枝 |
| codegen | emission of C++ and runtime scaffolding | 代码生成 |
| `OpExec` | generation/simulator/runtime wrapper | 算子执行器 |
| `gen_only` | emit sources without building/running hardware | 仅生成 |
| CANN | Ascend software/toolchain stack | 昇腾软件栈 |
| CANNSIM | CANN device simulator | CANN 模拟器 |
| `@func()` | decorator that keeps DSL AST naming active in a helper | DSL 辅助函数装饰器 |

Implementation lookup belongs to `agent/references/code-paths.md`; public
runtime use belongs to `doc/api/op_exec.md`.

## 10. Precision terms

| Term | Meaning | Chinese |
| --- | --- | --- |
| precision path / boundary | exact dtype, cast, accumulation, and rounding route | 精度路径/精度边界 |
| lossy boundary | a cast/layout step that may discard numerical information | 有损边界 |
| `plan_tolerance` | end-to-end budget versus the original oracle | 计划总误差预算 |
| `implementation_tolerance` | DSL budget versus planned references | 实现误差预算 |

Precision rules belong to `agent/references/constraints/precision.md`.

## 11. Common Chinese and English pairs

| Chinese | English in this repository |
| --- | --- |
| 算子 | kernel / operator |
| 流水线 | pipeline |
| 分块 / 切分 | tile / split |
| 尾块 | tail tile |
| 双缓冲 / 乒乓 | double-buffer / ping-pong |
| 同步 / 屏障 | synchronization / barrier |
| 数据搬运 | datamove |
| 全局内存 | GM / global memory |
| 矩阵侧 / 向量侧 | cube side / vec side |
| 互斥量 / 事件 | mutex / event |
| 写回 / 累加器 | writeback / accumulator |
| 掩码 | mask |
| 暂存 | staging |

In kernel code, *staging* means moving data to a memory level for later use;
a *pipeline stage* is a compute/dataflow phase.

## 12. Matmul bias terms

| Term | Meaning |
| --- | --- |
| BT / C2 bias | fp32 or int32 bias row consumed by MMAD on the init K tile |
| bias staging | contiguous GM-to-L1 followed by L1-to-BT movement |
| BT slot | one half of the automatic depth-2 BT buffer |
| shortcut bias tile | full N for nosplit/split-K, or declared N tile for split-N |

Capacity and device limits belong to `agent/references/facts-device-runtime.md`.
Public staging/dtype contracts belong to `doc/api/cube.md` and
`doc/api/shortcuts.md`; implementation lookup belongs to
`agent/references/code-paths.md`.

## 13. Workflow and delivery terms

| Term | Meaning |
| --- | --- |
| delivery granularity | user-visible number of runtime agent/example/kernels/launches |
| single-kernel delivery | one final `@kernel` invoked by one production `OpExec` |
| logical kernel stage | internal load/compute/handoff/store phase, not a launch |
| runtime kernel / launch | one delivered `@kernel` invocation |
| probe kernel | temporary task-local kernel that isolates one uncertain boundary |
| semantic coverage | mapping from every contract operation to an owning stage |
| host semantics | formula computation performed outside the delivered kernel |

Workflow selection belongs only to `agent/ROUTER.md`.
