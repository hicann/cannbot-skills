---
name: catlass-python-design
description: Use when specifying a new CATLASS Python DSL operator and producing its Torch oracle, definition, workloads, executable reference evidence, and frozen Markdown preliminary design
---

# CATLASS Python Design（门禁引导）

> **入口**：本阶段开始前先加载 `catlass-python-workflow` 并输出阶段卡。
> **硬前置**：`.catlass-python/references/catlass/examples/python/tla_dsl` 必须已成功下载且能
> 列出 `.py` 样例；缺失时先按 workflow §6 下载，下载失败即停止，不得用知识库或已安装包
> 源码替代。
> 默认直通：产物齐备并自检通过后直接进入 develop，不等待用户批准；用户要求先评审时才暂停。

> **自动执行**：遇到问题只要有可行方案就逐步尝试，一个方案失败就换下一个，不停下来找
> 用户确认。等待远端任务、编译、benchmark、设备或长跑用例时直接等待并轮询，不结束回合。
> 仅当缺少只有用户能提供的信息、或可行路径已穷尽时才提问。

> **超时纪律**：编译、算子运行、benchmark、profiling 与普通 shell 命令必须拆成独立命令块；
> 编译单独成块、不套 60s，但必须带明确上限（默认 `timeout 600s`）；单例运行一条命令只跑
> 一个 case/shape，加 `timeout -k 5 60s`；批量按 2~5 片拆分，分片命令用分片级超时（用例数
> × 60s + 余量），片内逐例 60s 硬超时，禁止整批套一次性 60s 或裸跑。
> 远端/容器调试时，`timeout` 必须写在最终执行环境的命令内，外层只负责连接。
> 超时或卡住立即终止，不得死等或加大超时。终止后先做卡健康判定：同一张卡的小探针或
> 官方算子能过 → 判本算子死锁并转 debug；探针也失败 → 判设备问题并换卡重试一次。

> **先生成、再推进**：本阶段先产出 `DESIGN.md`、`reference.py`、`definition.json`、
> `workload.jsonl`、`workload_dev.jsonl` 与校验脚本，再考虑执行。纯本地生成模式下本地
> 只写文件：reference 运行、静态检查、模拟/打桩检查全部留到真实环境（远端服务器或本机
> NPU 机器）；一开始就拿到服务器信息或本来就在服务器上，同样是先生成再推进。

本 Skill 负责算子规格、初步设计和 Torch 精度标杆，输出 `DESIGN.md`、`reference.py`、
`definition.json`、`workload.jsonl`、`workload_dev.jsonl` 与 reference 执行记录
（纯本地生成模式下是 `blocked` / `not_run` 记录加精确重跑命令）。
不实现 CATLASS Python solution，不做性能测试或优化。

## 执行模式（开始即确定）

进入设计前先根据用户说明确定执行模式，并写入 `DESIGN.md` 的 `执行模式：` 行：

| 模式 | 判定依据 | 推进上限 |
|---|---|---|
| `remote-detached` | 本地生成，当前拿不到远端连接，也没有本机 NPU | 只能推进到 develop 的代码生成完成；focused test 记 `deferred` 并附精确重跑命令，之后必须先连上远端 |
| `remote-connected` | 本地生成，远端调试环境可连接（或已给出连接方式） | 全流程真实执行，不允许 `deferred` |
| `local` | 本地生成并在本机调试（本机有 NPU 与 CANN） | 全流程真实执行，不允许 `deferred` |

判定只依据用户说明与工作区已有声明，禁止在本地跑 `npu-smi`、导入 `torch_npu` 或做任何
环境探测。用户未说明且工作区没有声明时，先确认一次，不得自行猜测。`design` 阶段本身
不允许推迟；full test #1 及之后的所有阶段都必须在真实环境执行，门禁不会接受推迟。

## 性能要求（开始即确定）

同时确定是否采集性能，并写入 `DESIGN.md` 的 `性能要求：` 行：

| 取值 | 判定依据 | 后续流程 |
|---|---|---|
| `requested` | 用户明确提出性能目标、时延、加速比、吞吐或 profiling 要求 | optimize 后验收全量测试（kernel 变化时跑 full test #3，无融合记 #2），再执行 benchmark 与 profiling |
| `none` | 用户没有明确提性能，只要求功能正确或实现算子 | optimize 后验收全量测试（kernel 变化时跑 full test #3，无融合记 #2），直接进入 finish，不执行 benchmark 与 profiling |

默认取 `none`。只有用户原话里出现性能相关诉求才写 `requested`，不得以“顺便测一下”为由
自行升级；用户后续补充性能要求时，在 `ITERATIONS.md` 记录一条设计修订并更新该行。
两种情况都必须完成 optimize（模式 B 2~3 轮 / 模式 A 一轮；baseline 之后重算的
`skeleton_ratio ≥ 0.80` 时按 `early_stop` 记 0 轮）；优化
采纳改动了 kernel 时补跑 `full test #3`（无融合阶段时记 `#2`），全部未采纳、kernel 未变化时
可复用最近一次全量测试（有融合时为 `#2`，无融合时为 `#1`）。

## 语言与实现边界

CATLASS Python 是以 Python 语法编写的算子 DSL，不是 C++ 算子库。设计必须面向
`python/tla_dsl` 的 Python 接口和执行模型：

- 算子入口采用 `@tla.kernel`、Python 函数、`tla.Tensor` 和 `tla.*` API；
- 实现文件应为 `.py`，方案必须给出 Python DSL，而不是 C++、CUDA C++、Ascend C 或
  伪装成 DSL 的 C++ 模板代码；
- 不得设计 C++ class/template、头文件、namespace、宏、指针算术、RAII 或
  `<<<grid, block>>>` launch；
- GM/L1/L0/UB、tiling、layout、copy、MMAD、vector、flag 和 mutex 等概念必须映射到
  CATLASS Python DSL 的现有接口；
- C++、MLIR 或生成代码只能作为后端资料，不能成为算子公开接口或初步实现方案；
- 默认 `fusion.mode=performance`：kernel 划分由你按算子语义与整体性能自行决定，允许多个
  `@tla.kernel`、多个 executor、GM 中间张量及 shape/dtype/layout 分派；
- 用户明确要求单个融合 kernel 时设 `fusion.mode=single_kernel`。**设计仍按自然划分出方案**：
  先给出 `fusion.structural_verdict: single | multi`，并把它作为 design 阶段唯一的实现方案，
  不为"好合并"预留、缩小或改动结构：
  - 自然划分就是单 kernel：按普通单 kernel 设计继续，写 `merge_required: false`；
  - 自然划分是多 kernel：写 `merge_required: true`。**design 不把多 kernel 方案改写为单
    kernel 方案**，只把它登记为实现链路的第二阶段任务：develop 按自然划分落地多 kernel 版，
    通过 `full test #1` 后进入独立的 `merge` 阶段完成融合，融合版通过 `full test #2`。
    DESIGN 的「融合与合并」写融合计划：段清单、段间边界清单（依赖方向、前后段 tile/任务
    映射是否一致、覆盖核范围）、容量可行性（两段是否要求同时驻留 L0C/L1），以及 merge
    阶段入口条件（`full test #1` passed，多 kernel 基线源码、SHA 与逐例结果已保留）。
  - **同步原语与 pipe 不在 design 选型**：段间怎么接、用哪个 flag、走哪条 pipe 统一在 merge
    阶段按
    `knowledge/ops/catlass-dsl/optimizations/multi_kernel_to_single_kernel.md`
    一次决定；design 不得为此调研 `SyncAll` / `cross_flag` / atomic，也不做合并原型。这是
    用户约束驱动的合并，不是结构判断出错，记录里要把"自然多 kernel"和"交付单 kernel"两件事
    都写清楚；
- `structural_verdict`（自然划分）与用户诉求正交，四种组合都必须支持，`merge_required`
  只由「用户要求单 kernel」且「自然划分是多 kernel」同时成立时为 `true`：

  | 用户诉求 | 自然划分 | `merge_required` | 交付路径 |
  |---|---|---|---|
  | `single_kernel` | 单 kernel | `false` | develop 直接交付单 kernel |
  | `single_kernel` | 多 kernel | `true` | develop 多 kernel → FT#1 → merge → FT#2 |
  | `performance` | 单 kernel | `false` | develop 直接交付单 kernel |
  | `performance` | 多 kernel | `false` | develop 交付多 kernel，merge 记 `not_applicable` |

  不得因为 `structural_verdict: multi` 就默认融合，也不得因为用户没提单 kernel 就强行拆成
  多 kernel；`merge_required` 无论取值都要登记，`performance` 模式下融合计划表留空即可。
  **`structural_verdict` 与 `merge_required` 必须各占独立一行**（gate 按行首匹配，只写在
  说明句里不算登记，design 门禁会直接判失败）；
- 合并不可行时（如两段各需一份顶满 L0C 的输出 tile），在 DESIGN 写明不可行的具体容量账与
  两个候选方案（缩 tile / 落 GM），不得为了满足"单 kernel"而悄悄降低精度或改数学语义；
- 每个 `@tla.kernel` 必须自包含：除形式参数、函数内局部值、Python 内建符号和受信任的
  `tla` API 命名空间外，不得读取模块级 tile、shape、dtype、params、开关或 closure 配置；
- 固定 tile/dtype/params 在 kernel 函数内定义；运行时 shape 通过显式 scalar/tensor
  参数、tiling data 或 tensor metadata 取得。

若目标仓库同时包含 C++ CATLASS 与 CATLASS Python，必须先定位 Python DSL 样例和
`catlass.tla` 导入方式。没有真实 Python DSL API 支持的设计项标记为待验证假设。

## Torch 精度标杆

设计阶段必须在当前 workspace 生成项目专用 `reference.py`，只使用公开 `torch` API 表达
算子数学语义，不得 import candidate、复用 candidate 输出或调用待实现算子。reference 与
candidate 在同一个目标 device 上执行；只有完全没有 NPU 时才退回 CPU，且只允许小 shape。

### 执行分层

| 层 | 内容 | 执行要求 |
|---|---|---|
| L1 静态 | Python 语法、`run` 入口、摘要与 solution 一致性 | 真实环境执行；纯本地生成模式推迟到真实环境 |
| L2 reference | Torch 精度 oracle（与 candidate 同一 `npu:<id>`） | 真实环境执行；纯本地生成模式推迟并记 `blocked` / `not_run` |
| L3 NPU | 编译、上板、精度、benchmark、profiling | 真实环境执行；缺环境记 `blocked` / `not_run` |

执行位置：有真实环境（远端服务器或本机 NPU 机器）就在该环境执行 L1/L2；纯本地生成模式
下本地一行都不跑，只生成文件并写好精确命令，等真实环境就绪后直接执行。本地不做静态
检查、模拟检查、打桩检查或任何形式的代跑。

L2 执行口径：

- 调用方（`validate_reference.py` 或 bench 脚本）必须提供 `--device`，把输入落到该
  `npu:<id>` 后再调 `reference.run`；`reference.run` 的签名与 bench 契约一致，不额外接
  device 参数。
- 冻结 workload 的全 shape 校验必须在 NPU 上执行；CPU 只允许在没有 NPU 时用于小 shape
  冒烟（M≤8、V≤256、H≤64）。不得用 CPU 跑全 shape，也不得以 CPU 上的完成时间或是否
  跑完作为性能、可行性或精度证据。
- 小 shape 冒烟只证明 `reference.py` 可执行，不能替代全 shape 校验。
- 完全没有 NPU 时，把冻结 workload 的全 shape L2 校验记录为 `blocked`，写明原因和精确
  重跑 argv（必须带 `--device`），并在 `DESIGN.md` 与 `ITERATIONS.md` 同步标注；有设备
  后补跑全 shape 校验，仍走 device，不得改用 CPU。
- 允许带 L2 `blocked`（纯本地生成模式记 `blocked` / `not_run`）进入 develop，但 `finish` 前必须
  补跑全部非 performance required cases；未补跑不得标记完成。
- 禁止为判断“能不能跑”做 CPU / NPU 算力或环境探测；环境缺失直接记录。

`reference.py` 必须导出与 `catlass-python-bench` 完全相同的 `run`：

```python
def run(*inputs):
    """Return one Tensor or a flat tuple/list ordered like definition.outputs."""
```

用例不在 `reference.py` 中声明。按 `catlass-python-bench` 的接口生成：

- `definition.json`：输入输出规格及完整 Torch `reference` 源码，`axes` 固定为空对象，
  `reference` 必须与同批生成的 `reference.py` 逐字一致；
- `workload.jsonl`：每行一个 required case，`uuid` 等于 `case_id`；可选 `origin`
  标记来源——用户明确提供的用例写 `user`，agent 自行补充的写 `agent_supplemented`，
  缺省按 `agent_supplemented` 处理；
- `solution.json` 由 develop 在实现完成后生成，不属于 design 输出。

case 规则：

- 不使用 axis binding：`definition.json` 的 `axes` 必须为 `{}`，workload 的 `axes`
  必须为 `null`；禁止用 axis 名称、`const`、`var` 或 `expr` 间接表达 shape；
- 每个 tensor input 在 workload 中直接写出数值 `shape` 和 `dtype`，标量写 `value`，
  可选空输入写 `null`；
- 每个 case 的 shape、dtype、layout selector、序列边界、布尔开关、标量参数和初始 state
  必须直接保存在该行；`uuid` 只标识用例，不得承担配置查表；
- 允许 `custom_inputs_entrypoint` 与 `type: custom` 生成特殊分布输入，但不得恢复 axis
  binding，也不得在 `reference.py` 维护 `_CASE_SPECS` 等 case table；
- 边界、尾块和广播尺寸分别用独立 workload 表达；随机输入固定 seed；
- tolerance 的字段语义、默认 `required_matched_ratio` 与 NaN/Inf 规则以
  [`catlass-python-bench`](../catlass-python-bench/SKILL.md) 为准；design 负责按累加精度与
  舍入分析给出具体 `max_atol` / `max_rtol` / `max_error_cap`，不能为通过用例放宽；
- definition/workload 不保存 solution 输出或伪造 expected。

### Dev Workload 子集

除全量 `workload.jsonl` 外，design 还必须冻结 `workload_dev.jsonl` 供 optimize 迭代：

- 条数：`workload_dev.jsonl` 总数不超过 40 条。用户用例（`workload.jsonl` 中
  `origin=user` 的行）不少于 30 条时冻结 **30~40** 条；不足 30 条时全部选入即可，
  不要求补齐；没有用户用例时按覆盖需要自定条数，不以下限 30 为目标；文件内 `uuid`
  顺序即用例顺序；
- 每行必须是对象，只允许 `uuid`、`inputs`（必填）和 `axes`、`tolerance`、`origin`、
  `source`（可选）；`uuid` 唯一且符合安全 ID 规则；
- 每行标注 `origin`：
  - `origin=user`：`uuid` 必须同时是 `workload.jsonl` 中的 required case，`source` 等于 `uuid`；
  - `origin=agent_supplemented`：不得声明 `source`，该行必须能真实执行；
- 用户用例不超过 40 条时子集必须全覆盖；超过 40 条时至少包含 30 条用户用例，
  其余用补充用例补齐；
- 子集必须覆盖代表性用例与用户点名/文档反复举例的用例。全部 optimize 轮次共用这份子集，
  不得每轮换样本。

### Reference 校验

参考校验（真实环境执行；纯本地生成模式只写脚本与命令，不运行）：

1. 在 `DESIGN.md` 的 `## Reference Validation` 小节写入 `reference.py`、
   `definition.json`、`workload.jsonl` 的相对路径与 SHA-256；
2. 自行编写一次性校验脚本（例如 `tools/validate_reference.py`），导入 `reference.py`，
   按 workload 生成输入、执行 `run`、按 tolerance 比对，输出逐 case 结果 JSON；
3. 在真实环境执行该脚本，把命令、退出码、通过数和证据路径追加到 `ITERATIONS.md`；
   纯本地生成模式记 `blocked` / `not_run` 并附精确 argv，不在本地运行；
4. 任一输入文件变化后重新运行全部 case，更新摘要与 revision，不得只改摘要。

三个输入文件的摘要和用例集一旦冻结即不再变更；新增或修改 `DESIGN.md` 的实现方案
不需要重新冻结 reference（见下方“设计是工作草稿”）。

## 过程

1. 只读探索目标仓库，确认 `python/tla_dsl` 来源、导入方式、已有 `.py` 算子模式、
   构建/测试入口和参考实现。用户指定标杆文件并禁止参考兄弟目录时，只读标杆文件、
   用例文件与 catlass 参考源码。
2. 明确算子名称、family、用途和可观察行为。按 `catlass-python-knowledge` 检索相关条目，
   把查询关键词、采用的条目路径与适用约束写入 `DESIGN.md`。
3. 固定每个输入输出的 symbolic shape、dtype、layout 和语义。
4. 写清数学计算、广播/尾块/空输入等边界行为，并给出「数值契约」：每个阶段/中间量的
   存储 dtype、累加 dtype、窄化或提升发生在哪一步、舍入/饱和/近似；reference 自己的
   精度路径（f16/bf16 是否先 `.float()` 再算、结果按输出 dtype 还是 f32 转回）也要写明。
   单 kernel 内部的 C→V→C、epilogue、fixpipe 窄化同样逐行列出，不只看跨 kernel 边。
   契约必须由「标杆步骤映射」得来：逐行拆 `reference.py` 的每个计算步骤（含 cast、scale、
   归约、scatter、广播），标出对应 kernel 阶段与对齐结论，不得跳步、合并或只列自己想到的
   中间量；漏一步视为契约未完成。
   累加与归约顺序（K 分块与 `BK` 循环、是否 splitK、atomic add、分段归约、online softmax）、
   padding 与被 mask lane 的归约语义、NaN/Inf 传播一并写清。舍入模式的支持范围以目标
   `ascend-catlass-python` 版本的包实际 API 能力为准，版本内不同子包也可能不一致，不支持时
   写替代方案，不得按“应该支持”假设（例如 RINT 这类 round mode 在不同子包的支持情况
   并不一致，2.2.0 仍在演进，必须按目标版本实测或查 API 确认）。容差按输出 dtype 写入
   definition（atol/rtol/matched_ratio/max_error_cap），debug 期不得放宽。
   **数学语义必须一致，实现精度可升档**：
   **数学语义（必须一致）**：算子定义、输入输出关系、mask 与索引、归并/散写（如 scatter）
   的合并关系（哪些元素合并、重复索引是否累加、越界如何处理），以及 softmax 等非线性步骤
   的数学定义（如 max 减法、归一化形式），必须与标杆表达同一个数学过程；改变这些按精度
   bug 处理。softmax 的 online / 两遍式等实现顺序属于精度实现，按升档规则处理。
   **精度不降即可自由实现**：存储/累加 dtype、归约与累加顺序、分块方式、舍入位置与模式
   属于精度实现。实现精度严格不低于标杆：dtype 与累加 dtype 逐项不低于且至少一项更高；
   舍入不按模式名排序，以同一输入下该步的误差界不劣于标杆为准（给出推导或对照实验）。
   满足时不再要求逐项复现标杆；顺序、分块、舍入位置都可以按实现需要选择，只需保证数学
   过程不变，并在 deviation 项记录标杆原实现、实现新选择、推导依据、预期偏差量级与影响用例
   （实测值在 develop 补齐）。
   **同精度与降精度**：与标杆同精度时，顺序重排与分块方式变化只产生浮点尾差，允许；但不得
   增加舍入次数或提前舍入。精度低于标杆、增加或提前舍入、引入额外低精度落盘，按精度 bug 处理。
   因 DSL 限制既无法按标杆精度表达、也无法升档时，单列 deviation：哪一项、原因、量级估算或
   对照实验、影响用例范围、high risk 标记和补偿计划，不得只写“容差内”。
   **容差只用于判定**：只吸收数学语义一致后的浮点尾差，不得用于规避语义一致性，也不得在
   调试阶段放宽；判定不过时按误差分析重推容差并记录，重推仍不过按精度 bug 处理。本条不反向
   成立：精度更高不等于可以改变数学语义。
   **被切开的 matmul 按整体处理**：标杆把同一个 matmul 的 K 维（或 batch 维）按外部
   预算切成多段、逐段累加到同一个输出时（典型形态是 `for seg: C.addmm_(A[seg].T,
   B[seg])`，段长由临时缓冲预算而非算法语义决定），design 阶段必须先做「分段判定」
   并写入 `DESIGN.md`：段间除对 C 的累加外没有其他读写、没有跨段归约或非线性依赖、
   切分不改变数学结果。判定成立时按**一个完整 matmul** 设计：K 方向走单链，中间累加
   统一用目标设备可用的最高精度（默认 fp32 L0C），只在写回输出 dtype 时舍入一次；
   不得逐段复现低精度累加器的中间舍入，也不得为复现段边界在 GM 上做逐段读改写。
   该选择属于上文「精度不降即可自由实现」，必须在 deviation 中记录：段数、
   标杆的舍入次数、预期偏差量级、对照实验和影响用例。判定不成立（段间存在归约/
   非线性/其他数据依赖，或用户明确要求逐段 bit 级复现）时按原分段语义实现，并写明
   原因。不得因「分段是标杆的显存优化」就直接认定它可以合并，也不得因「合并后精度
   更高」就跳过判定和 deviation。
   **「完整 matmul」的实现判据**：判定成立后，实现必须同时满足三条（本条优先于上文
   精度升档规则，精度更高也不例外）——① 单个输出 tile 的全部 K 贡献在同一次 kernel
   launch 内累加完成；② 累加器只存在于片上（默认 fp32 L0C），不得跨 launch 落在 GM，
   `atomic add`、GM 读改写、fp32 中间缓冲回读都按跨 launch 累加器处理；③ 只在写回
   输出 dtype 时舍入一次。按输出 tile / block 划分任务不改变累加器归属；把同一个输出
   tile 的 K 贡献分散到多个 launch 再合并，一律按分段实现处理，不能因为「精度仍是 fp32」
   就认为合规。
   **中间量预算必须以设备可用显存为上限反推**：确定 chunk / tile 前先写显存账（可用
   HBM − 输入输出 − 工作区余量），预算取设备上限与算法需要中的较小值；不得直接沿用
   保守默认值（如 2 GiB）而不核对设备上限。预算取小导致的分段属于实现自造的约束，
   不能作为「完整 matmul 不可行」的依据。
   **中间量放不下时的降级顺序**：① 把中间量预算提到设备可用上限；② 改布局或复用缓冲；
   ③ 只有 ①② 都实测不可行（写明实测显存与失败现象）才允许 split-K 或跨 launch 累加，
   并在 deviation 中单列「完整 matmul 不可行」的依据、替代方案对照与影响用例。
   `DESIGN.md` 同时出现「判定成立」与 atomic / 跨 chunk 累加缓冲时，必须补足上述实测
   依据，否则判定无效。
5. 生成 `reference.py` 与 `definition.json`，把每个非 performance required case 写成
   `workload.jsonl` 一行，并按规则生成 `workload_dev.jsonl`。
6. 按 L2 口径在真实环境运行 reference cases 并记录证据；全 shape 不可完成时先冒烟
   再记 `blocked`。纯本地生成模式不在本地运行，只写命令并记 `blocked` / `not_run`。
7. 使用 CATLASS Python DSL API 给出融合范围、各 kernel 职责与调用顺序、中间张量、
   tiling/layout、内存空间、pipeline 和同步方案，解释性能选择依据。tiling 块数按核数
   取整，行块对齐 cube fractal（16 行），依据见 `dsl/pattern-gemm-optimization-ladder`。
   包含 MMAD 的算子（纯 matmul 与 CV 融合都算）必须完整读取该条目，并在 `DESIGN.md` 的
   「GEMM 骨架与优化阶梯」一节声明目标级别（第 2/3 级）、主 matmul 的骨架验证 shape
   （M/N/K/dtype/batch）与来源 dev 用例 uuid、全部 matmul 阶段清单（名称 / 所在 kernel /
   预计占比 / 各自最大 dev shape）、首版骨架必须包含的双缓冲结构、明确排除的反模式，
   以及 skeleton 验收判据（含豁免）；每个 matmul 阶段都要在 develop 各自跑一次骨架评测，
   不得只测主 GEMM。只写条目名或“参考”不算完成。是否评测只看“算子是否包含 MMAD”，
   不按整体时间占比判断：CV 融合算子即使 vector 段占比更高，也要逐个 matmul 声明 shape
   并评测，因为瓶颈通常在 matmul 段；纯 vector 算子不适用。骨架是测量工件，不改变最终
   kernel 划分与 `fusion.mode`；不得因为要做骨架评测就把 CV 融合算子拆成独立 Cube / Vector
   kernel。默认目标为第 3 级，选第 2 级必须说明第 3 级为何不可行。skeleton 性能判据、
   豁免与带宽受限规则以
   `catlass-python-develop/SKILL.md` 的「skeleton 性能 DoD」为准。
   先判定算子整体结构（阶段数、Cube/Vector 交替方式、数据流、任务映射、同步方式），
   再按整体结构选一个**主参考样例**：Cube-only（纯 MMAD）参考
   `tla/kernel/matmul/basic_mmad/basic_matmul.py`；Cube-Vector（MMAD 后接 vector 后处理）参考
   `experimental/kernel/matmul/matmul_exp_scale_ub/matmul_exp_scale_ub.py`；
   Cube-Vector-Cube（两段 MMAD 中间夹 vector）参考
   `experimental/kernel/matmul/matmul_add_chain/matmul_add_chain.py`；纯 vector 参考
   `tla/kernel/vec/basic_vadd/basic_vadd.py`、`dsl_tutorials/vector_simd_ops/`、
   `dsl_tutorials/vector_simt_ops/`。
   **整体结构优先于分步参考**：主参考是 kernel 划分、缓冲层次和同步骨架的基线，不是
   约束；算法或硬件要求不同时，可以在 DESIGN 里调整并写清偏离点与理由。主参考缺少某个
   局部机制（如 gather SIMT、atomic add、特定 epilogue）时，用局部样例补齐。没有足够
   契合的主参考时，允许组合多个分步样例，但必须在 DESIGN 里显式设计组合后的整体结构
   （阶段顺序、接口、同步），并照常跑 skeleton DoD；不得无说明地把样例片段拼在一起。
   参考样例必须在 `.catlass-python/references/catlass/` 的 `examples/python/tla_dsl/` 或
   `python/tla_dsl/tests/end_to_end/` 下实际存在；
   在 DESIGN 的「结构类型与参考样例」一节分别记录主参考（整体结构匹配点、偏离点）和
   局部参考（补齐的机制或组合方案）。目标版本没有同名样例时，选最接近的整体结构样例
   并记录差异与改造点，不得凭记忆引用。
   登记样例时必须同时登记它在当前环境的可运行性（DESIGN 的「参考样例可运行性」字段）：
   主参考必须实跑；局部参考只在直接复用其机制时实跑，其余可记 `not_run` 并写明未复用原因。
   `remote-connected` / `local` 下用样例自带的入口（`run()` 或等价入口）原样实跑一次
   （不改源码，取小 shape，覆盖样例声明的关键机制——如 scatter、SPLIT_M 等，必须让对应
   分支真正执行），编译按 `timeout 600s`、单次运行按 `timeout -k 5 60s`；命令可直接内联，
   也可落在 `tools/sample_probe.py`，日志与退出码放 `evidence/reference/`，并记录 shape。
   结果记 `passed` / `failed`；`remote-detached` 下记 `not_run` 并写出精确重跑命令，作为
   远端第一个执行动作。主参考记 `failed` 时，必须写明失败机制（编译 / 运行 / 超时）、
   是否影响拟用结构与降级方案，不得按未验证的结构直接冻结设计；局部参考记 `failed` 时
   说明改用哪个等价样例。
   容差/偏差测量与样例 probe 同口径：`remote-connected` / `local` 下只允许在 NPU 上跑
   1~2 个最大用例，单例 `timeout -k 5 60s`；**禁止 CPU 全尺寸复现标杆分块**（例如按
   chunk 循环 `addmm_` 复现 96 段 bf16 累加的探针）。design 只写容差公式与解析上界，
   实测放到 develop 首跑执行；`remote-detached` 下记 `not_run` 并写出精确重跑命令。
   设计期先完整读取 `knowledge/ops/catlass-dsl/runbooks/runtime/multi_kernel_precision_localization.md` 的「通用规则」
   （中间量 dtype、layout、舍入位置、归约顺序、独立 oracle），不要等到测试失败才看。
   多 kernel 或跨 kernel 中间张量的算子，还必须为每条跨 kernel 边定义精度契约：数学语义、
   shape/stride/有效区、输入输出与累加 dtype、舍入位置、重复 ID/归约/chunk 顺序、消费
   分支和对应 oracle；并在 DESIGN 的「跨 kernel 阶段 oracle」一节给出阶段检查计划（局部
   检查、真实链路、双向替换）。没有中间量 oracle 时，明确写出替代验证方式。
   同时给出搬运量核算（Σ 重读次数 × 体积）与按有效 MTE2 带宽折算的搬运下界，tiling 参数
   偏离参考样例时逐项说明差异及对搬运量的影响。
   写明 `fusion.mode`、单融合是否由用户明确提出，以及是否登记「自然多 kernel → 交付单
   kernel」的第二阶段融合（`structural_verdict` / `merge_required`、边界清单、merge 阶段
   入口条件 `full test #1` 是否齐备）；没有实测支持的性能判断标记为待验证假设。
8. 明确正确性 oracle、required cases、`focused_cases`、focused/full test 命令、风险、
   允许路径、benchmark/profiling 命令和可选性能门槛。要求执行器输出实际 argv、退出码
   和逐 case 结果。
9. 从 [`templates/DESIGN.md`](templates/DESIGN.md) 创建面向用户的设计文档，写入目标 DSL
   版本与执行模式，并在 `ITERATIONS.md` 记录 design 里程碑。文档创建后只允许用 `Edit`
   定点修改，禁止整文件重写；同一内容不在对话与文档中各写一遍（对话只留结论摘要）；
   按小节增量构建，禁止“先写粗略全文再整体推翻重写”。DESIGN 大小由算子复杂度决定，
   不设上限；超过 30 KB 时检查是否有重复描述（同一 deviation 多节重复、样例代码整段抄入）。
10. 向用户展示 `DESIGN.md`、Resolved Plan 与 reference case 结果（纯本地生成模式
    展示 `blocked` / `not_run` 状态与精确命令）；默认直通时直接进入 develop，用户要求
    先评审时才暂停。

初步设计不是已验证实现结论，不得虚构 benchmark 或 profiling 数据。涉及 DSL 执行语义、
同步、内存、layout、精度 oracle、公共接口或构建配置时，`risk_level` 必须为 `high` 并
填写 targeted review focus。

**设计是工作草稿，不是合同。** 评测口径（allowed paths、required cases、benchmark
命令、融合模式）冻结后不再修改；但 `DESIGN.md` 中的实现方案（tiling、
数据流、buffer 规划、同步策略、kernel 划分）允许在 develop 阶段随实测修改：

- 修改后在 `ITERATIONS.md` 记录一条“设计修订”（改了什么、为什么改、依据哪次实测）；
- 只有最终交付时才把 `DESIGN.md` 固化为只读快照。

设计应当在“写→跑→看数”的循环中收敛，而不是在动笔前一次性消灭所有不确定性。

冻结前再次检查：算子主体和 reference 都是 Python 文件；算法引用的 `tla.*` 接口可以从
本地源码或知识条目定位；「数值契约」表已覆盖每个阶段/中间量（存储 dtype、累加 dtype、
窄化或舍入位置），没有留空或只写“同 torch”；标准 definition/workload 覆盖全部非
performance required cases（`blocked` 项附原因与精确 argv）；正文包含三个文件的实际
SHA-256；没有把 C++ CATLASS API 当作 DSL API；每个 decorated kernel 自包含；融合边界
有明确性能依据。

## 输出

```text
DESIGN.md
reference.py
definition.json
workload.jsonl
workload_dev.jsonl
evidence/reference-cases.json      # 自行生成的一次性校验结果
```

`DESIGN.md` 必须冻结 benchmark 命令、metric JSON path，并设置 profiling 要求。
声明性能门槛时还必须写明方向、阈值、迭代上限、停滞阈值和最小提升比例。门槛只能来自
用户提示或用户确认；由知识条目、基线实现或 roofline 推算出的数值只是参考目标，不得
写成本设计的门槛。没有性能门槛不表示跳过 optimize：无门槛时进入模式 B（explore），
执行 2~3 轮优化（上限 3 轮，命中 `early_stop` 时 0 轮）；也不表示跳过 benchmark 或 profiling。
