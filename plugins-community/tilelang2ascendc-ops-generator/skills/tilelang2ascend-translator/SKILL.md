---
name: tilelang2ascend-translator
description: >
  AscendC kernel 转译与实现专家 Skill。将 TileLang 设计转译为 AscendC kernel，
  并生成 model_new_ascendc.py 调用 AscendC kernel。
  当 TileLang 设计完成需要转译为 AscendC kernel 时，使用此 skill。
argument-hint: >
  输入：output_dir 目录路径（包含 tile_level/ 和 model_new_tilelang.py）。
  输出：kernel/ 下的 AscendC 实现、model_new_ascendc.py。
---

# AscendC Kernel 转译 Skill

你是一名 AscendC kernel 转译与实现专家。你的目标是将 TileLang 设计转译为 AscendC kernel，并生成 `{output_dir}/model_new_ascendc.py` 调用 AscendC kernel，最终通过 AscendC 验证。TileLang 在这里是设计输入，不是 correctness gate。

## 前置条件
本阶段开始前，以下产物必须已经存在：
- `{output_dir}/design/tile_level/` — TileLang tile-level 设计，作为转译输入
- `{output_dir}/model_new_tilelang.py` — TileLang 绑定层/设计表达，可参考但不作为正确性依据

## 关键限制
- 必须将核心计算融合成单个算子实现，不要拆分成多个独立算子。
- `model_new_ascendc.py` 中禁止使用 torch 算子；只允许进行张量创建，张量变换以及调用你实现的自定义算子。
- 在 AscendC 实现中应尽可能避免标量逐元素写法，优先使用块级或向量化操作；只有在确实无法避免时才使用标量逻辑。
- 只允许修改或新增 `{output_dir}/` 目录中的文件，不要改动其他目录中的文件。
- 只允许读取当前工作区目录结构内的文件与子目录；禁止读取当前工作区之外的任何路径，包括父目录、兄弟目录、用户目录、绝对路径以及系统其他目录。
- 禁止读取 `.claude/skills/tilelang2ascend-translator/references/TileLangAscendProgrammingGuide.md`；该文档是 TileLang 编程指南，仅供 TileLang 阶段使用，与本阶段无关。
- 严格按照算子描述生成kernel，ascend c kernel的功能应该和标杆完全一致，不能出现部分功能使用ascend c，部分使用torch算子的情况
- 即使测试用例中不包含某个功能或者分支对应的case，也要生成对应的ascend c kernel代码
- **🛑 参考实现 ≠ 可复制代码**：`workflows/templates/archive_tasks/` 用于理解结构范式（目录组织、API 用法、EXEC_KERNEL_CMD 传参模式、缓冲规划），**禁止整体照抄其代码**。复用任何参考代码必须：① 按当前算子的 shape/dtype/归约路径/广播形态**逐行适配**；② 重新推导 tiling 与 UB 预算（不沿用 archive 的硬编码参数）；③ 全量验证通过。archive 中存在的缺陷不得被复制进新算子。
- **🛑 同步机制强制门禁**: 涉及 MIX_AIC / CrossCore / WorkspaceQueue / 死锁 / 全零输出 时，必须先完成 **步骤 0-C** 的同步 checklist。详见下方步骤 0-C 章节。
- **🛑 性能计时口径**：性能数据只接受 device 侧 kernel 时间（msprof Task_Duration，经 ops-profiling 采集）；禁止用 torch_npu Event / 墙钟计时作为性能结论——其读数包含 host 下发间隙与共享设备干扰，µs 级算子会出现数量级假数据（实测发生过参考侧虚高 ~100x 的事故）。
- **🛑 长时间性能测试防卡死**：执行耗时较长的性能测试 / 批量 benchmark（msprof 采集、逐 case 长跑等）时，必须对测试进程做**轮询 + 超时保护**——周期性检查其是否仍在推进（输出增长 / 进程存活 / 心跳），超时立即终止并上报，防止 kernel 挂死（hang / aicore timeout）导致无限等待、吞掉整个生成流程。

### 算子设计准则（必须遵守）

以下三条准则在 AscendC kernel 设计与转译全程中必须遵守：

**准则 1：UB 空间复用与扩满**

设计计算块时，尽可能实现 buffer 复用，减少临时 buffer 的申请；扩大 UB 使用量，实现尽可能用满所有可用 UB 空间。

- **复用优先**：当多个计算步骤的 buffer 生命周期不重叠时，应复用同一 TBuf 而非申请新 buffer。例如 softmax 的 max buffer 和 sum buffer 在不同 pass 中使用，可复用同一 TBuf
- **扩满 UB**：在不超过 `GetCoreMemSize(UB)` 上限的前提下，增大 tile size 使 UB 利用率尽可能接近 100%。通过 `platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(UB)` 获取 UB 总量，所有 InitBuffer 分配之和应接近该值
- **减少临时 buffer**：优先使用 `ReinterpretCast` 复用已有 buffer 的内存空间，而非申请新 TBuf

**准则 2：避免不必要的 Cast**

在不影响精度的条件下，不考虑额外的数据类型转换（即 Cast 操作）。

- **默认不 Cast**：若输入为 fp16/bf16 且该 dtype 下 AscendC API 支持直接计算，则不做 fp16→fp32→fp16 的升降精度往返。例如 `SoftMax<half>` 可直接处理 fp16 输入，无需先 Cast 到 fp32
- **精度优先例外**：当某步计算在当前 dtype 下会导致精度丢失（如 reduce 在 fp16 下精度不足），才允许 Cast 到更高精度
- **注意**：本准则受准则 3 约束——若内置 API 要求特定 dtype，允许为满足 API 调用条件而做 Cast

**准则 3：优先使用内置库算子或 API 接口**

设计算子代码时，在保证精度及可运行的条件下，优先选择可调用的内置库算子或 API 接口的途径，可为了满足内置库算子或 API 的使用条件而进行数据类型转换。若该方法设计出的算子在最终性能测试过程中加速比低于 0.8x，则考虑将内置库算子或 API 替换为自定义步骤计算的途径。若自定义步骤计算性能更低，则仍采用内置库算子或 API 的途径。若某种途径会导致精度丢失或运行错误，则不予考虑该途径。

- **优先级**：内置库算子/API（如 `at::softmax`、`SoftMax` 高阶 API、`aclnnXxx`）> 自定义步骤计算（手动 Exp/Reduce/Div）
- **允许为 API 做 Cast**：若内置 API 要求特定 dtype（如 `SoftMax<half>` 要求 fp16 输入），允许 Cast 以满足调用条件（本条优先于准则 2）
- **op_host 层面也适用**：当内置库算子（如 `at::softmax` dispatch 到 `aclnnSoftmax`）能在 op_host 中直接调用且精度完全匹配时，优先采用该途径而非自定义 kernel
- **性能回退阈值 0.8x**：内置 API 路径在性能测试中加速比 < 0.8x 时，尝试自定义实现；若自定义更慢则回退到内置 API
- **精度/运行错误排除**：任一路径导致精度丢失或运行错误，立即排除该路径，不考虑采用

## 目标任务目录结构
```text
.
├── {output_dir}/         # 当前活跃任务目录
│   ├── model.py          # 参考 PyTorch 模型，禁止修改
│   ├── <op_name>.json    # 原始测试用例文件（备份保留）
│   ├── <op_name>.json.bak# 原始 .json 备份
│   ├── design/           # TileLang DSL 用于表达 kernel 设计
│   │   ├── design.md     # 设计文档（简单算子路径）或 不存在（复杂算子路径）
│   │   ├── block_level/  # TileLang block-level 设计（已由上一阶段完成）
│   │   └── tile_level/   # TileLang tile-level 设计（已由上一阶段完成，作为转译输入）
│   ├── kernel/           # AscendC kernel（op_host/ + op_kernel/ 分层）
│   │   ├── CMakeLists.txt
│   │   ├── setup.py      # whl 打包配置
│   │   ├── ops.h         # 算子声明
│   │   ├── register.cpp  # torch.ops.npu.* 注册（仅注册）
│   │   ├── op_host/
│   │   │   └── <op_name>.cpp  # Host 端: tiling + EXEC_KERNEL_CMD
│   │   ├── op_kernel/
│   │   │   └── <op_name>.cpp
│   │   └── utils/        # 固定工具（从模板复制，不生成）
│   │       └── torch_kernel_helper.h
│   ├── test/             # 测试目录
│   │   ├── <op_name>-test-cases.md
│   │   └── test_<op_name>.py
│   ├── model_new_tilelang.py # 上一阶段产物，可参考但不要修改
│   └── model_new_ascendc.py  # AscendC wrapper → 内部调用 torch.ops.npu.<op>()
└── <other_tasks>/        # 其他历史任务，可作为参考实现
```

## Skill 参考资料
本 skill 提供以下参考资料：
- `.claude/skills/tilelang2ascend-translator/references/dsl2Ascendc.md` — TileLang 转 AscendC 指南
- `.claude/skills/tilelang2ascend-translator/references/TileLang-AscendC-API-Mapping.md` — TileLang 与 AscendC API 映射表
- `.claude/skills/tilelang2ascend-translator/references/AscendCVerification.md` — AscendC 验证指南
- `.claude/skills/tilelang2ascend-translator/references/ascendc_reduce_patterns.md` — 归约族算子实现指南（(O,R,I) 路由、补零/行距铁律、应避免的结构、精度与验证约定）
- `.claude/skills/tilelang2ascend-translator/references/ascendc_shuffle_patterns.md` — 重排/搬运类算子实现指南（固定开销约束、硬件 pattern 指令、广播消费结构、核数分档、已知低效结构）
- `.claude/skills/tilelang2ascend-translator/references/attention-patterns/AttentionPatternIndex.md` — Attention / FlashAttention 类算子的模式路由索引（TND、paged KV cache、mask/causal、GQA/MQA、MLA、topk sparse KV、sink attention）
- `.claude/skills/tilelang2ascend-translator/scripts/evaluate_ascendc.sh` — AscendC 评测脚本
- `workflows/templates/archive_tasks/` — 历史成功任务，host/kernel 完整参考实现（**编译/运行时错误时优先查阅**）
- 共享演进知识库（`$CANNBOT_KNOWLEDGE_ROOT` 的 `runbooks/`）— 历史走偏点与成功模式，**经 knowledge-query skill 检索（步骤 0-K 必读，命中即规避）**

### 🛑 官方文档目录（asc-devkit，强制查阅）

以下所有路径相对于 `asc-devkit/` 根目录。**编写/修改任何 kernel 代码前，必须先查阅对应的官方文档。禁止凭记忆或猜测 API 签名、参数、dtype 支持矩阵。**

| 查阅入口 | 内容 | 何时查阅 |
|----------|------|---------|
| `asc-devkit/docs/api/Ascend-C-API列表.md` | API 分类总览与快速索引 | 每次代码生成前 |
| `asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/类型转换/Cast.md` | Cast API 签名与 dtype 支持 | 使用 Cast 时 |
| `asc-devkit/docs/api/SIMD-API/基础数据结构/GlobalTensor/GlobalTensor简介.md` | GlobalTensor 完整 API | 使用 GlobalTensor 时 |
| `asc-devkit/docs/api/SIMD-API/基础数据结构/LocalTensor/LocalTensor简介.md` | LocalTensor 完整 API | 使用 LocalTensor 时 |
| `asc-devkit/docs/guide/算子实践参考/SIMD算子实现/矢量编程/基础矢量算子.md` | CopyIn→Compute→CopyOut 标准范式 | 每次代码生成前 |
| `asc-devkit/docs/guide/算子实践参考/SIMD算子实现/矢量编程/TBuf的使用.md` | UB 临时缓冲区管理 | 分配 TBuf 时 |
| `asc-devkit/docs/guide/算子实践参考/SIMD算子实现/融合算子编程/` | 多步计算融合模式 | 融合算子时 |
| `asc-devkit/examples/01_simd_cpp_api/` | 官方 SIMD C++ 示例 | API 用法不确定时 |
| `workflows/templates/archive_tasks/rms_norm/` | EXEC_KERNEL_CMD 正确传参模式 | 编写 op_host 时 |

除非用户明确指定其他目录，否则默认使用传入的 `output_dir` 作为当前任务目录。
其他任务目录可以作为参考实现。

---

### 🛑 步骤 0-A: Attention 算子模式路由（Attention / FlashAttention 类算子强制执行）

🛑 **无论算子类型如何，第一步必须读取 model.py**：
   Read `{output_dir}/model.py` 的 forward() 方法，逐行检查计算逻辑。
   转译阶段的输入固然是 tile_level 设计文件，但判断算子是否属于 Attention 类
   必须回到 model.py 的原始计算逻辑。**禁止**凭 tile_level 文件名或记忆跳过此步。

**触发条件**：读取 model.py 后，检查 forward() 是否包含以下任一特征：
- `softmax(Q @ K^T / sqrt(d)) @ V` 或等价 attention 计算模式（如 `F.softmax(matmul(Q, K^T) / sqrt(dk)) @ V`）
- `scaled_dot_product_attention` / `F.scaled_dot_product_attention`
- 类名包含 `Attention` / `SDPA` / `Flash`
- tile-level 设计中包含 Q/K/V 三输入 attention 结构

如果触发条件满足，必须逐个完成以下 checklist：

```
0-A.1 🛑 读取 AttentionPatternIndex.md（必须，不可跳过）:
    Read .claude/skills/tilelang2ascend-translator/references/attention-patterns/AttentionPatternIndex.md

0-A.2 🛑 逐条回答"生成前问题"中的 7 个诊断问题，记录命中的模式:
    1. 输入是标准 [B,H,S,D] 还是 (T,H,D) 拼接布局？
    2. K/V 是连续 tensor 还是 paged cache？
    3. Hq 和 Hkv 是否相等？
    4. Dqk 和 Dv 是否相等？
    5. 是否有 sink_k/sink_v？
    6. 是否有 indices/topk？
    7. 是否有 causal、padding、显式 mask？
    
    如果 7 项全否定 → 命中"标准 Attention" → 下一步 0-A.3 读 archive 模板
    如果任一命中 → 下一步 0-A.3 读对应的 pattern 文档（可组合）

0-A.3 🛑 只读取命中的文档（渐进式披露，只读需要的）:
    - 命中模式 → Read 对应文档顶部的"先读这个"

0-A.4 🛑 在思考中确认:
    - 已读的 pattern 文档列表及其关键语义边界
    - 组合顺序（多模式命中时按 TND → Head Sharing → MLA → Sink → Sparse → Paged → Mask 顺序理解）
    - 本算子的 AscendC 转译策略应与命中的 pattern 对齐
```

**门禁规则**：
- 如果触发条件满足但 0-A.1-0-A.4 未完成 → **禁止**进入步骤 1，**禁止**编写任何 kernel/ 代码
- 如果触发条件不满足 → 跳过步骤 0-A，直接进入步骤 0-B
- 禁止凭记忆或经验跳过模式文档直接转译

### 🛑 步骤 0-A2: 归约 / 重排类算子实现指南路由（命中特征时强制执行）

**触发条件**（步骤 0-A 读取 model.py forward() 后一并检查）：
- 归约族特征：`torch.sum / mean / max / min / prod` 等沿维（或全部）归约计算；
  以及均值/方差统计量型算子（`layer_norm` / `LayerNorm` / `batch_norm` / `rms_norm` /
  `var` / `std` 等——其 forward 必然内含归约）
- 重排/搬运类特征：奇偶交织 / stride 切片重组（含 `chunk`/`split`/`cat`/`stack`
  半区拆分重组）/ gather / scatter / 广播消费（如 RoPE 交织、RotaryMul 旋转乘、
  permute 类变体）

如果命中，必须完成以下 checklist：

```
0-A2.1 🛑 只读取命中族的实现指南（渐进式披露，只读需要的）:
    - 归约族 → Read .claude/skills/tilelang2ascend-translator/references/ascendc_reduce_patterns.md
    - 重排/搬运类 → Read .claude/skills/tilelang2ascend-translator/references/ascendc_shuffle_patterns.md
    （两族同命中 → 都读）

0-A2.2 🛑 在思考中确认:
    - 归约族：本算子落入 (O,R,I) 哪条路径（A 跨行 RA / B 多行批归约 / C 分块两级树），
      以及补零/行距、精度约定等铁律的落点
    - 重排/搬运类：本算子的重排结构走哪条硬件 pattern 路线，
      固定开销结构（launch 建表 / 广播物化 / strided 拼写回）是否全部规避
    - 本算子的 AscendC 转译策略应与命中指南的结构规则对齐
```

**门禁规则**：
- 命中但 0-A2.1/0-A2.2 未完成 → **禁止**进入步骤 1，**禁止**编写任何 kernel/ 代码
- 未命中 → 跳过 0-A2，直接进入步骤 0-B
- 禁止凭记忆或经验跳过指南直接转译

---

## 🛑 步骤 0-K: 演进知识检索（每次代码生成/修改前强制执行）

**触发条件**：任何 kernel 代码生成 / 修复迭代开始前，均须执行。

```
0-K.1 🛑 检索共享演进知识库（必须，不可跳过）:
    用 cannbot-knowledge 插件的 knowledge-query skill
    （scripts/knowledge_query.py，root 由 knowledge.env 解析，当前 /home/asc-gen-knowledge）:
    python3 knowledge_query.py preflight --task "<本算子类型 + 关键结构特征>" --brief
    先读 route/read_first/relevance，再按需 get 整卡。

0-K.2 🛑 按维度标签补充检索命中卡片:
    - 本算子涉及多核/混合核/搬移/跨核通信 → search --query "同步 竞态" --scope runbooks/
      （或 --tags sync 等价语义），读 sync 维度卡
    - 计划使用 Gather/SetVectorMask/归约等 API → 检索 api 维度卡
    - 涉及 tiling/缓冲/核数/多 dtype → 检索 tiling 维度卡
    - 涉及 FP16/BF16 数学函数精度 → 检索 precision 维度卡
    - 命中卡片"触发条件" → 全文精读其"正确做法"，进入 0-K.3

0-K.3 🛑 在思考中确认:
    - 命中的已知坑清单及其规避策略（如: 不用 TBuf 做 DMA 主搬移、
      核数上限按 UB 池数、count-mode Gather 不可用 → isSetMask=true）
    - 规避策略将如何体现在本算子的 op_host/op_kernel 代码中
```

**门禁规则**：
- 触发条件满足但 0-K.1-0-K.3 未完成 → **禁止**进入步骤 1，**禁止**编写任何 kernel/ 代码
- 知识库条目与官方文档矛盾时以官方文档为准，并将差异在 trace 走偏点中记录
  （供演进循环更新条目）
- 此门禁在**每次修复迭代**中都需重新检查（不仅限于首次）

---

## 🛑 步骤 0-B: 查阅官方文档（每次代码生成/修改前强制执行）

**在 Edit/Write 任何 `kernel/` 下的代码文件之前，必须完成以下查阅步骤。此步骤不可跳过。**

```
0.1 阅读标准范式:
    asc-devkit/docs/guide/算子实践参考/SIMD算子实现/矢量编程/基础矢量算子.md
    → 确认 CopyIn→Compute→CopyOut 的完整流水线模式

0.2 阅读 EXEC_KERNEL_CMD 正确模式:
    workflows/templates/archive_tasks/rms_norm/kernel/op_host/rms_norm.cpp
    → 确认: 所有 tiling 参数必须是独立标量左值，禁止传 struct 指针
    → 确认: blockDim = usedCoreNum（多核统一分发），禁止 host 侧逐核循环
    → 确认: 核数来源于平台 API 动态获取（GetCoreNumAic/Aiv），非硬编码常量

0.3 逐个查阅要使用的 API 文档:
    根据算子计算逻辑，列出所有将使用的 AscendC API，然后**逐个**查阅以下精确路径的文档。
    ⚠️ 每个 API 必须确认: ① 模板参数（类型/非类型）② 函数参数（个数/类型）③ dtype 支持矩阵 ④ work buffer 需求。
    **禁止凭记忆或猜测 API 签名**。

    ── 数据搬运 ──
    - DataCopyPad → asc-devkit/docs/api/SIMD-API/基础API/Memory数据搬运/DataCopyPad(ISASI).md
      ⚠️ 签名两态: GM→UB 4参(dst,src,cp,pp), UB→GM 3参(dst,src,cp)
    - DataCopy → asc-devkit/docs/api/SIMD-API/基础API/Memory数据搬运/DataCopy/DataCopy.md

    ── 类型转换 ──
    - Cast → asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/类型转换/Cast.md
      ⚠️ 确认 bfloat16→float32 和 float32→bfloat16 的 RoundMode 参数

    ── 矢量计算 (Memory) ──
    - Mul → asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/基础算术/Mul.md
    - Add → asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/基础算术/Add.md
    - Sub → asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/基础算术/Sub.md
    - Rsqrt → asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/基础算术/Rsqrt.md

    ── 标量计算 (Reg) ──
    - Muls → asc-devkit/docs/api/SIMD-API/基础API/Reg矢量计算/基础算术/Muls-27.md
    - Adds → asc-devkit/docs/api/SIMD-API/基础API/Reg矢量计算/基础算术/Adds-28.md
    - Rsqrt (scalar) → 与矢量 Rsqrt 同族，查阅基础算术目录

    ── 高阶 API ──
    - ReduceSum → asc-devkit/docs/api/SIMD-API/高阶API/归约操作/ReduceSum接口/ReduceSum-90.md
      ⚠️ 模板: <T, pattern, isReuseSource>, 参数: (dst,src,workBuf,srcShape[],srcInnerPad)
      ⚠️ GetReduceSumMaxMinTmpSize → 同目录 GetReduceSumMaxMinTmpSize.md
    - Broadcast → asc-devkit/docs/api/SIMD-API/高阶API/张量变换/Broadcast.md
      ⚠️ 模板: <T, dim, axis, isReuseSource>, dim∈{1,2}, axis∈{0,1}
    - Cos → asc-devkit/docs/api/SIMD-API/高阶API/数学计算/Cos接口/Cos.md
      ⚠️ GetCosMaxMinTmpSize → 同目录 GetCosMaxMinTmpSize.md
    - Sin → asc-devkit/docs/api/SIMD-API/高阶API/数学计算/Sin接口/Sin.md
      ⚠️ GetSinMaxMinTmpSize → 同目录 GetSinMaxMinTmpSize.md

    ── 同步控制 ──
    - PipeBarrier → asc-devkit/docs/api/SIMD-API/基础API/同步控制/核内同步/PipeBarrier(ISASI).md
      ⚠️ 确认 PIPE_MTE2/PIPE_MTE3/PIPE_V/PIPE_ALL 各 barrier 的放置位置规则
    - CrossCoreSetFlag/WaitFlag → .claude/skills/tilelang2ascend-translator/references/ascendc-sync-guide.md
      ⚠️ 确认 mode2 下 Set/Wait 两侧 PIPE 参数完整且配对
      ⚠️ AIC 侧: Set<0x2, PIPE_FIX> + Wait<0x2, PIPE_FIX>
      ⚠️ AIV 侧: Set<0x2, PIPE_MTE2> + Wait<0x2, PIPE_MTE2>（MTE3 写用 PIPE_MTE3）
      ⚠️ 封装泛型工具类（如 WorkspaceQueue）时 ProducerAcquire/ConsumerAcquire 必须将 PIPE 模板化传入

    📋 **查阅完成后，在思考中列出 API 签名清单**:
    对每个 API 记录:
    - 完整模板参数 (如 ReduceSum<float, AscendC::Pattern::Reduce::AR, false>)
    - 完整函数参数名和类型
    - work buffer 需求 (需要/不需要, 如需要则列出 GetXxxMaxMinTmpSize 的查阅结果)
    - dtype 约束

0.4 查阅 TBuf 用法:
    asc-devkit/docs/guide/算子实践参考/SIMD算子实现/矢量编程/TBuf的使用.md
    → 确认 UB 临时缓冲区的正确分配模式

0.5 🛑 验证所有 work buffer 尺寸（运行时正确性铁律）:
    对于每个使用 TBuf<uint8_t> 作为 work buffer 传入的 API，**必须在 host 端通过对应的
    GetXxxMaxMinTmpSize 计算正确尺寸，禁止在 kernel 中硬编码 work buffer 大小**。

    │ Work Buffer 使用者 │ 尺寸获取 API (host 端调用) │ API 文档 │
    │-------------------│---------------------------│---------│
    │ ReduceSum         │ GetReduceSumMaxMinTmpSize │ asc-devkit/docs/api/SIMD-API/高阶API/归约操作/ReduceSum接口/GetReduceSumMaxMinTmpSize.md │
    │ ReduceMax         │ GetReduceMaxMaxMinTmpSize │ asc-devkit/docs/api/SIMD-API/高阶API/归约操作/ReduceMax接口/GetReduceMaxMaxMinTmpSize.md │
    │ ReduceMin         │ GetReduceMinMaxMinTmpSize │ asc-devkit/docs/api/SIMD-API/高阶API/归约操作/ReduceMin接口/GetReduceMinMaxMinTmpSize.md │
    │ Cos               │ GetCosMaxMinTmpSize       │ asc-devkit/docs/api/SIMD-API/高阶API/数学计算/Cos接口/GetCosMaxMinTmpSize.md │
    │ Sin               │ GetSinMaxMinTmpSize       │ asc-devkit/docs/api/SIMD-API/高阶API/数学计算/Sin接口/GetSinMaxMinTmpSize.md │
    │ SinCos            │ GetSinCosMaxMinTmpSize    │ asc-devkit/docs/api/SIMD-API/高阶API/数学计算/SinCos接口/GetSinCosMaxMinTmpSize.md │
    │ Broadcast         │ GetBroadCastMaxMinTmpSize │ asc-devkit/docs/api/SIMD-API/高阶API/张量变换/GetBroadCastMaxMinTmpSize.md │

    **验证步骤 (每次编写 kernel 前强制执行)**:
    a. 列出本算子所有使用 work buffer 的 API
    b. 逐一查阅上表中对应的 GetXxxMaxMinTmpSize 文档
    c. 确认每个 API 的 work buffer 最小/最大尺寸计算方法
    d. 在 host 端 tiling 函数中调用 GetXxxMaxMinTmpSize，将结果作为 tiling 参数传入 kernel
    e. kernel 中 InitBuffer 的 work buffer 尺寸必须来自 tiling 参数，**禁止**硬编码为固定值
    f. 在思考中记录: 每个 work buffer 的计算结果和对应的 API 名称

    ⚠️ 本步骤为**运行时正确性硬性要求**。硬编码 work buffer 尺寸 < 实际所需最小值
       会导致 vector core timeout (507034) / UB 内存违例等运行时错误。

⚠️ 未完成以上 0.1-0.5 全部步骤前，禁止进入步骤 1 编写代码。Attention 类算子还必须完成步骤 0-A。
   查阅完成后，在思考中明确列出已查阅的文档路径及其关键约束。
```

---

### 🛑 步骤 0-C: 同步机制门禁（涉及跨核同步 / MIX_AIC / 输出异常时强制执行）

**触发条件**（任一满足即触发）：
- kernel 使用了 `KERNEL_TYPE_MIX_AIC` 混合核模式
- 代码中出现 `CrossCoreSetFlag` / `CrossCoreWaitFlag` / `WorkspaceQueue`
- 运行时出现**全零输出**、死锁、hang、vector core timeout (507034)
- 编译后功能验证 FAIL 但无编译错误

如果触发条件满足，必须逐个完成以下 checklist：

```
0-C.1 🛑 读取 ascendc-sync-guide.md（必须，不可跳过）:
    Read .claude/skills/tilelang2ascend-translator/references/ascendc-sync-guide.md 全文

0-C.2 🛑 逐条在思考中确认以下 checkpoint:
    ① PIPE 配对:
       - AIC 侧所有 CrossCore Set/Wait → PIPE_FIX
       - AIV 侧所有 CrossCore Set/Wait → PIPE_MTE2（MTE3 写操作用 PIPE_MTE3）
       - WaitFlag 是否漏写 PIPE 模板参数
    ② CV1:2 模式信号计数:
       - AIC Set 1 次 → 两个 AIV 各 Wait 1 次
       - 两个 AIV 各 Set 1 次 → AIC Wait 2 次（等两个 AIV 都完成）
    ③ 封装泛型工具类:
       - WorkspaceQueue ProducerAcquire/ConsumerAcquire 是否通过模板参数将 PIPE 传入两侧
    ④ InitFreeSlots:
       - 是否仅 Consumer 侧调用一次（禁止 Producer/Consumer 两侧重复调用）
    ⑤ 条件分支:
       - 是否可能跳过 Set/Wait 导致对方死等（如提前 return）
    ⑥ TQue BUFFER_NUM:
       - 是否 ≥ 循环中同时持有的 queue tensor 数量 + 1

0-C.3 🛑 如果存在 AIC↔AIV 交叉依赖（如 AIC 等 AIV 的 SIG_P_READY，AIV 同时等 AIC 的 SIG_O_READY）:
    - 画出信号时序图，确认不存在循环等待（A 等 B 设 X，B 同时等 A 设 Y）
    - 确认 PRELAUNCH 延迟是否足够打破循环依赖
```

**门禁规则**：
- 触发条件满足但 0-C.1-0-C.3 未完成 → **禁止** Edit/Write 任何涉及同步的代码
- 禁止凭经验修改 CrossCore 参数而不查阅 sync-guide
- 此门禁在**每次修复迭代**中都需重新检查（不仅限于首次）

---

### 🛑 步骤 0-D: 性能设计门禁（每次代码生成前强制执行）

#### 核心原理：搬运单元容量最大化

NPU 上每一次数据搬运（DataCopyPad DMA）、每一次流水线迭代（tile 循环）都携带**固定开销**（DMA 建链、地址计算、同步信号、流水线起停）。实际性能取决于：

```
有效性能 = 有用数据量 / (有用数据量 + 固定开销)
```

当每次搬运/迭代处理的数据量很小时，固定开销 dominate，性能急剧下降。**因此，一切性能优化的根因都指向一个目标：让每个流水线单元（DMA 调用、tile 迭代、计算步骤）处理尽可能多的数据，把固定开销摊到最小。**

#### 因果链与检查清单

以下 5 项检查按因果链组织，前项是后项的使能条件：

```
① 消除冗余搬运（wrapper 零拷贝）
   │  原理: 多做一次全量拷贝 = 多一份 100% 固定开销, 零有效数据
   │  检查: forward() 中是否有 permute/contiguous/reshape/to(dtype)？
   │  要求: 0 次全量拷贝。所有布局/精度变换移入 kernel 内部
   │  度量: wrapper_copy_bytes = 0
   │
   ② 单次 DMA 传满（禁止逐行循环）
   │  原理: 逐行调用 32 次 DMA = 32 倍固定开销; 2D strided 1 次 = 1 倍
   │  检查: CopyIn/CopyOut 中是否有 for 循环逐行调用 DataCopyPad？
   │  要求: 跨行数据用 2D DataCopyPad (blockCount>1, srcStride) 一次加载
   │  度量: bytes_per_dma_call 应接近 UB tile 大小（如 4KB+），而非 128B
   │
   ③ 每个 tile 算满（Tile 利用率 ≥ 80%）
   │  原理: tile 被截断为 64 元素时, 每步向量指令只处理 64 元素 → 固定开销占比极高
   │  检查: 对所有测试用例, actual_tileSize 是否接近 TILE_SIZE？
   │  要求: rowSize < TILE_SIZE 时启用 multi-row tile, 打满 TILE_SIZE
   │  度量: tile_utilization = actual_tileSize / TILE_SIZE ≥ 80%
   │
   ④ 流水线不空转（双缓冲）
   │  原理: BUFFER_NUM=1 时, tile N 的 CopyIn 等 tile N-1 的 CopyOut 完成 → 流水线空泡
   │  检查: TQue 的 BUFFER_NUM 是否 ≥ 2？
   │  要求: VECIN/VECOUT 队列 BUFFER_NUM=2, 实现 CopyIn/Compute/CopyOut 重叠
   │  度量: pipeline_overlap = 1 (BUFFER_NUM≥2)
   │
   ⑤ 多核不空闲（核利用率 ≥ 90%）
   │  原理: 空闲核不做有效计算但仍承担初始化开销
   │  检查: usedCoreNum 是否接近物理核数（大张量时）？
   │  要求: usedCoreNum = min(物理核数, max(1, totalOutput)), 动态获取, 禁止硬编码
   │  度量: core_utilization = usedCoreNum / physicalCoreNum ≥ 90%
```

**三维分解（outerSize, dimSize, innerSize）是 ①②③ 的共同使能条件**：
- 它定义 `rowSize`、`inputRowSize`、`stride`，使 kernel 能直接在原始内存布局上操作任意 dim
- 没有它，kernel 只能处理 dim=-1，其余 dim 必须 wrapper 做全量拷贝（违反 ①）
- 没有它，无法计算 `rowSize`，无法检测 multi-row，无法计算 2D stride（违反 ②③）

**门禁规则**：
- ①②③ 未确认前，禁止进入步骤 1 编写代码——违反将导致 10~30 倍性能劣化
- ④⑤ 未确认前，禁止进入步骤 1——违反将导致 2~5 倍性能劣化
- 同步机制（TQue+EnQue/DeQue 替代 TBuf+PipeBarrier）是 ④ 的前置——错误同步会导致 V 核读到未就绪数据（精度失败）

**经验数据**（来自 SwiGLU 算子 A/B 对比，验证因果链）：
```
             ①wrapper   ②DMA       ③tile    ④buffer  Avg Speedup
未优化:       permute拷贝  逐行循环    3.1%     BUFFER=1   1.16x
优化后:       零拷贝      2D strided  100%     BUFFER=2   1.72x
```
仅修复 ②③（multi-row + 2D DMA），即使 ① 仍有缺陷，Avg Speedup 仍从 1.16x 提升到 1.72x。


---

### 🛑 步骤 0-E: 加速比驱动二级闭环优化（Phase 5 性能分析后、Phase 6 全量验证前强制执行）

#### 执行时机
- **前置条件**：Phase 5 (ops-profiling 性能分析) 已完成，加速比数据已采集
- **执行窗口**：Phase 5 完成后、Phase 6 (全量验证) 启动前
- **触发依据**：Phase 5 性能分析产出的加速比数据（bench_time / custom_time）
- **适用范围**：数学计算类算子强制执行（Sort/TopK/Reduce/MatMul/Norm/Elementwise 等）；纯数据搬运类算子（Copy/Gather 等无计算）可跳过

#### 二级闭环结构
本步骤采用二级串联闭环：**第一级（算法层）先触发，通过后第二级（类型层）再触发**。两级相互独立，各有 3 次迭代上限。所有优化方案使用通用描述，不绑定具体算子名。

---

#### 0-E.1 第一级 — 整体加速比算法对比

**触发条件**：整体加速比 ≤ 0.8x（所有 dtype 加速比的 geomean）

**执行流程**：
```
Step 1: 识别自定义算子当前采用的算法
        ├─ 检查 kernel 代码中实际调用的 AscendC 高阶 API / 自定义算法路径
        └  记录算法名（如 MERGE_SORT / RADIX_SELECT / SEQUENTIAL_REDUCE / TREE_REDUCE）

Step 2: 调研 CANN 内置实现的算法
        ├─ 查阅 asc-devkit/docs/api/ 或 aclnn 文档，确定 CANN 标杆算子使用的算法
        └  记录标杆算法名

Step 3: 对比算法代差
        ├─ 查阅下方"算法对比参考表"
        └  判断自定义算法 vs 标杆算法是否存在代差

Step 4: 决策
        ├─ 算法不同且标杆更先进 → 采用更先进算法重新生成 kernel:
        │    4a. 回到 Phase 3 重新做 TileLang 设计（更换算法，如 MERGE_SORT → RADIX_SELECT）
        │    4b. 重新走 Phase 4 转译 + 编译 + 精度验证（evaluate_ascendc.sh）
        │        ├─ PASS → 重新采集加速比（ops-profiling --quick），回到 Step 1 重新评估
        │        └─ FAIL → 走 Phase 4 对应修复流程（A 类走 4.5A，D 类走 4.5D）
        │    4c. 迭代上限：3 次（每次包含完整的 Phase 3→Phase 4 流程）
        │    4d. 若 3 次后算法仍无法对齐或加速比仍不达标 → 进入 Step 5 微优化排查
        ├─ 算法相同且整体加速比 > 0.8x → 第一级 PASS，进入第二级（0-E.2）
        └─ 算法相同但整体加速比仍 ≤ 0.8x → 进入 Step 5 微优化排查

Step 5: 算法已最优但不达标 — 微优化排查
        ├─ 检查 Step 0-D 性能因果链 5 项是否已全部应用：
        │  ① wrapper 零拷贝 ② 单次 DMA 传满 ③ tile 利用率≥80%
        │  ④ 双缓冲 BUFFER_NUM≥2 ⑤ 核利用率≥90%
        ├─ 若有未应用项 → 应用后重新评估加速比（计入第一级迭代次数）
        └─ 全部已应用仍不达标 → 判定为"算法/硬件限制"
           记录上限分析（Roofline + Amdahl），第一级退出进入第二级
```

**算法对比参考表**：

| 算子族 | 落后算法 | 先进算法 | 典型代差 |
|--------|---------|---------|---------|
| Sort | BUBBLE_SORT / 逐行冒泡 | RADIX_SORT（按 bit 分桶） | 3-10x |
| TopK | MERGE_SORT (O(n log k)) | RADIX_SELECT (O(n)) | 2-5x |
| ReduceSum | SEQUENTIAL_REDUCE（顺序累加） | TREE_REDUCE（树形归约） | 2-4x |
| MatMul | 逐块 GEMM（无 L0 cache 利用） | Blaze/tensor_api 分块（L0A/L0B/L0C 缓存） | 3-8x |
| Norm | 逐元素 Exp/Reduce/Div 三遍扫描 | 融合单遍扫描（Online Softmax / RMSNorm 单 pass） | 2-3x |

**门禁规则**：
- 整体加速比 ≤ 0.8x 但未完成 0-E.1 算法对比 + Step 5 微优化排查 → **禁止**进入 Phase 6
- 算法代差未消除（仍使用落后算法）→ 强制重新生成 kernel，直到算法对齐或达 3 次迭代上限
- 算法已最优 + 微优化已全部应用仍不达标 → 记录上限分析后可退出第一级（不阻塞）

---

#### 0-E.2 第二级 — BF16 加速比专项优化

**触发条件**：整体加速比 > 0.8x（第一级已通过），但分 dtype 统计中 **BF16 加速比 ≤ 0.8x**

**执行流程**：
```
Step 1: 分 dtype 统计加速比
        ├─ 按 fp32 / fp16 / bf16 分别统计 (bench_time / custom_time)
        └  确认仅 BF16 不达标（fp32/fp16 均达标）

Step 2: 诊断当前 BF16 处理策略
        ├─ 策略 A: BF16 原生计算（kernel 直接在 BF16 上操作）
        ├─ 策略 B: Host 端 Cast（model_new_ascendc.py 中 x.to(fp32) → kernel fp32 → .to(bf16)）
        └─ 策略 C: Kernel-internal Cast（kernel 内 Cast<fp32, bf16> 升精度计算后 Cast 回）

Step 3: 按算子类型 + 当前策略决定是否调整 Cast 策略
        ├─ 归约/累加类算子 + 当前非 kernel-internal Cast → 调整为 kernel-internal Cast，进入 Step 4
        ├─ 数据重排类算子（memory-bound）+ 当前为 Host Cast → 调整为 kernel-internal Cast，进入 Step 4
        ├─ 数据重排类算子（memory-bound）+ 当前为 kernel-internal Cast → 已是最优策略，检查 RoundMode 和填充值
        ├─ 数据重排类算子（compute-bound）+ 当前为 Host Cast → 评估 UB 容量后决定
        ├─ 逐元素/纯 Vector 算子 + 当前为 Host Cast → 调整为 BF16 原生（移除 host Cast）
        ├─ 逐元素/纯 Vector 算子 + 当前为 BF16 原生 → 已是最优，第二级无优化空间，退出
        └─ 所有"已是最优策略"情况 → 检查 RoundMode/填充值是否符合硬性约束，不符则修复

Step 4: 如需调整 Cast 策略 → 执行完整迭代（迭代上限 3 次）:
  4a. 修改 kernel 代码（kernel/op_kernel/<op>.cpp 增加/调整 Cast 逻辑）
  4b. 修改 wrapper（model_new_ascendc.py 移除/添加 host 端 Cast，保持 wrapper 零搬运原则）
  4c. 重新编译 + 精度验证（evaluate_ascendc.sh）
      ├─ PASS → 继续 4d
      └─ FAIL（D 类精度不匹配）→ 走 4.5D 精度修复流程（消耗 d_retry，不计入 0-E.2 的 3 次上限）
  4d. 重新采集加速比（ops-profiling --quick）
  4e. 评估:
      ├─ BF16 加速比 > 0.8x → 第二级 PASS，进入 Phase 6
      └─ BF16 加速比仍 ≤ 0.8x 且迭代 < 3 次 → 回到 4a
```

**RoundMode 硬性约束速查表**：

| Cast 方向 | RoundMode | 原因 |
|-----------|-----------|------|
| BF16 → FP32 | `CAST_NONE` | mantissa 7bit → 23bit 无损扩展，无需舍入 |
| FP32 → BF16 | `CAST_RINT` | mantissa 23bit → 7bit 有损舍入，round-to-nearest-even |
| FP32 → FP16 | `CAST_RINT` | mantissa 23bit → 10bit 有损舍入 |
| FP16 → FP32 | `CAST_NONE` | mantissa 10bit → 23bit 无损扩展 |

**⚠️ 禁止误用**：
- 误用 `CAST_RINT` 做 BF16→FP32 会产生无效值（如 1.1939e-39），因 CAST_RINT 在扩展时尝试舍入，破坏 mantissa
- 填充值必须使用目标 dtype 可表示范围：BF16 用 ±3.38953139e38f（BF16 max finite），**禁止**用 FP32 max (3.4028235e38f)，Cast 回 BF16 会溢出为 Inf

**Kernel-internal Cast 适用范围表**：

| 算子类型 | Kernel-internal Cast | 原因 |
|---------|---------------------|------|
| 归约/累加（Reduce/Mean/Sum） | ✅ 强制使用 | FP16/BF16 下累加精度不足 |
| 逐元素/纯 Vector（Mul/Add/Sub） | ❌ 禁止 | BF16 原生精度足够，Cast 增加 GM 流量 |
| 数据重排 memory-bound（Sort/TopK/Gather） | ✅ 推荐 | host 端 Cast 导致 GM 流量 5 倍放大（640MB vs 128MB），kernel 内 Cast 减少 80% GM 流量；UB 内数据量翻倍的开销远小于 GM 流量节省（TopKV2 实测验证：BF16 加速比从 0.617x 提升至达标） |
| 数据重排 compute-bound | ⚠️ 需评估 UB 容量 | Cast 使 UB 数据量翻倍可能溢出，需确认 UB 容量充足后再使用 |
| 矩阵乘（MatMul/Linear） | ✅ 允许 | API 要求特定 dtype，或精度需求 |
| 激活函数（Softmax/LayerNorm） | ✅ 允许 | 中间归约步骤需高精度 |

**GM 流量对比**（kernel-internal Cast vs host 端 Cast，以 128MB BF16 输入为例）：
- kernel 内 Cast：输入 BF16 (128MB) → kernel 内升 FP32 计算 → 输出 BF16 (128MB) = **256MB GM 流量**
- host 端 Cast：输入 BF16 (128MB) → host 升 FP32 (256MB) → kernel FP32 (256MB) → host 降 BF16 (128MB) = **640MB GM 流量**
- kernel 内 Cast 减少 GM 流量 ~60%，对 memory-bound 算子有显著收益

**门禁规则**：
- 整体达标但 BF16 不达标且未完成 0-E.2 诊断 → **禁止**进入 Phase 6
- Cast 策略调整后必须重新验证精度（回到步骤 3-步骤 4 精度验证流程）

---

#### 0-E.3 闭环退出条件

```
第一级（算法层）:
  ├─ 3 次内算法对齐且整体加速比 > 0.8x → 第一级 PASS，进入第二级
  └─ 3 次后仍未达标 → 记录失败原因，跳过第二级直接进入 Phase 6（全量验证时标注）

第二级（类型层）:
  ├─ 3 次内 BF16 加速比 > 0.8x → 第二级 PASS，进入 Phase 6
  └─ 3 次后仍未达标 → 记录失败原因，进入 Phase 6（全量验证时标注）

整体退出:
  ├─ 两级均 PASS → 正常进入 Phase 6
  ├─ 任一级达 3 次上限未达标 → 仍进入 Phase 6，但在 trace.md 中记录未达标项
  └─ 总迭代上限：6 次（第一级 3 + 第二级 3）
```

**门禁规则**：
- 0-E.3 退出后必须更新 trace.md，记录两级迭代次数和最终加速比
- 禁止以"0-E 未达标"为由阻塞 Phase 6——0-E 是优化闭环，不是阻塞门禁

---

## 流程
执行以下各步骤前，必须先完成 **步骤 0-A（如触发）、步骤 0-B、步骤 0-C（如触发）、步骤 0-D 的全部查阅**，再开始实现、验证与迭代。

**步骤 0-E 执行说明**：0-E 在 Phase 5 性能分析完成后、Phase 6 全量验证前执行。0-E 是加速比驱动的二级闭环优化，包含算法对比（第一级 0-E.1）和 BF16 专项优化（第二级 0-E.2），两级各有 3 次迭代上限，总迭代上限 6 次。0-E 是优化闭环而非阻塞门禁——即使未达标也不阻塞 Phase 6，但必须在 trace.md 中记录未达标项与迭代次数。0-E 优化后重新生成 kernel 时，重新测量加速比用 ops-profiling --quick（不重走完整 Phase 5 流程）。

### 步骤 1: TileLang 转译成 AscendC

将 `{output_dir}/design/tile_level/` 下的 TileLang 设计转译为对应的 AscendC 实现。生成以下文件：
- `{output_dir}/kernel/op_host/<op_name>.cpp` — Host 端 (tiling 计算 + kernel launch)
- `{output_dir}/kernel/op_kernel/<op_name>.cpp` — Device 端 (CopyIn → Compute → CopyOut)
- `{output_dir}/kernel/ops.h` — 算子函数声明
- `{output_dir}/kernel/register.cpp` — torch.ops.npu.* 注册
- `{output_dir}/kernel/setup.py` — whl 打包配置
- `{output_dir}/kernel/CMakeLists.txt` — CMake 编译配置
- `{output_dir}/kernel/utils/kernel_common.h` — CopyTiling 等公共工具
参考文档：`.claude/skills/tilelang2ascend-translator/references/dsl2Ascendc.md`
**🛑 实施转译前必须先完成步骤 0-A（如触发）、步骤 0-B 和步骤 0-C（如触发）的全部查阅，然后阅读 `.claude/skills/tilelang2ascend-translator/references/TileLang-AscendC-API-Mapping.md` 逐一确认 API 映射。禁止跳过 Mapping 直接编写 AscendC 代码。**

**op_host/<op_name>.cpp** 模式：
- include `torch_kernel_helper.h` + `tiling/platform/platform_ascendc.h`
 - 🛑 **核数获取（禁止硬编码）**：根据算子计算特征选择正确的 API，  **禁止 `constexpr int32_t = 20` 或 `min(20, ...)` 等硬编码**：
     ```cpp
     auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();

     // 纯 Vector 计算（Norm/激活函数/逐元素等）→ GetCoreNumAiv()
     int32_t totalCoreNum = platform->GetCoreNumAiv();

     // 纯 Cube 计算（MatMul/矩阵乘）→ GetCoreNumAic()
     int32_t totalCoreNum = platform->GetCoreNumAic();

     // Cube+Vector 融合（Attention/CV融合等）→ CalcTschBlockDim(), sliceNum为数据切分的份数
     int32_t totalCoreNum = platform->CalcTschBlockDim(
         sliceNum, platform->GetCoreNumAic(), platform->GetCoreNumAiv());
     ```
- 使用平台 API 获取 `GetCoreMemSize(UB)`
- Block 级 tiling: Cache Line 512B 对齐，formerNum/formerLength/tailNum/tailLength
- UB 级 tiling: bufferCoefficient 推导，32B 对齐 tileLength
- **🛑 EXEC_KERNEL_CMD 传参铁律**: 所有 tiling 参数必须是**独立标量左值**，**禁止传 struct 指针**。参照 `workflows/templates/archive_tasks/rms_norm/kernel/op_host/rms_norm.cpp` 的正确模式
- blockDim = usedCoreNum（多核统一分发），kernel 内部通过 `GetBlockIdx()` 计算工作范围
- `EXEC_KERNEL_CMD` 所有参数必须为**左值**（具名变量），禁止传入临时变量/右值/字面量。`double` 先转 `float` 局部变量，`bool` 用 `int64_t` 替代，表达式先赋给局部变量再传入

**op_kernel/<op_name>.cpp** 模式：
- template class `Kernel<OpName>` 含 Init/Process/CopyIn/Compute/CopyOut
- BUFFER_NUM = 2 (double buffer)；如算子需要在循环中同时持有多个 queue tensor，需相应增大 BUFFER_NUM
- DataCopyPad 用于 GM↔UB 搬运
- dtype 处理遵循准则 2：默认不做额外 Cast；仅当 API 不支持当前 dtype 或精度不足时才升精度到 FP32 计算（如 reduce 类操作在 fp16 下精度不足）
- UB buffer 分配遵循准则 1：优先复用 TBuf，扩满 UB 空间
- 整核/尾核偏移和尾块对齐处理

**🛑 搬运单元容量最大化：三维分解 + Multi-row Tile + 2D DataCopyPad（必须检查）**：

当算子需要沿任意 dim 操作（非仅末尾维度），或输出 shape 与输入不同（如 chunk/split 后逐半处理：SwiGLU/GeGLU/ReGLU 等），必须通过三维分解使 kernel 直接在原始内存布局上操作，避免 wrapper 做全量拷贝。

**根因**：DMA 每次调用有固定开销（建链、地址计算、同步）。三维分解提供 `rowSize`/`inputRowSize`/`stride` 参数，使 kernel 能：(1) 免 wrapper 拷贝直接访问任意 dim 数据；(2) 检测 multi-row 打满 TILE_SIZE；(3) 用 2D strided 一次 DMA 加载多行。三者共同将固定开销摊到最小。

**第一步：三维分解（对应指标 1: Wrapper 零搬运）**

Host 侧（op_host）必须将任意 shape 分解为三维逻辑结构：
```cpp
// dim 之前所有维度乘积
int64_t outerSize = 1;
for (int32_t i = 0; i < normDim; ++i) outerSize *= x.size(i);
// dim 维大小
int64_t dimSize = x.size(normDim);
// dim 之后所有维度乘积
int64_t innerSize = 1;
for (int32_t i = normDim + 1; i < ndim; ++i) innerSize *= x.size(i);

int64_t halfDim = dimSize / 2;
int64_t rowSize = halfDim * innerSize;       // 输出每"行"元素数
int64_t inputRowSize = dimSize * innerSize;  // 输入每"行"元素数 = 2 * rowSize
int64_t totalOutput = outerSize * rowSize;
```

**关键恒等式**：`inputRowSize = 2 * rowSize`（恒成立，因为 dimSize = 2 * halfDim）。

这意味着 `a` 和 `b` 在同一行内相邻（a 在前半，b 在后半），跨行间隔为 `inputRowSize`。kernel 通过 stride 直接访问，**无需 wrapper 做 permute+contiguous**。

**第二步：Multi-row Tile 检测（对应指标 2: Tile 利用率）**

在 `Init()` 中检测：
```cpp
if (rowSize > 0U && rowSize < static_cast<uint32_t>(TILE_SIZE) &&
    (rowSize * sizeof(dataType)) % 32U == 0U) {
    multiRowMode = true;
    elementsPerCore = CeilDivU32(elementsPerCore, rowSize) * rowSize;  // 核间行对齐
}
```

**第三步：2D DataCopyPad 跨行加载（对应指标 3: DMA 效率）**

`CopyIn()` 中 multi-row 模式使用 2D strided DataCopyPad：
```cpp
// 一次 DMA 加载 numRows 行的 a 数据（跳过 b 块）
DataCopyExtParams copyParams{
    static_cast<uint16_t>(numRows),  // blockCount = 行数
    rowSize * sizeof(T),              // blockLen = 每行字节数
    rowSize * sizeof(T),             // srcStride = 行间隔（跳过 b 块，单位字节）
    0,                                // dstStride = UB 中连续排列
    0
};
DataCopyPad(aLocal, xGm[aBaseOffset], copyParams, padParams);
DataCopyPad(bLocal, xGm[bBaseOffset], copyParams, padParams);  // bBaseOffset = aBaseOffset + rowSize
```

**stride 语义**（来源：`ascendc-api-best-practices/references/api-datacopy.md`）：
- `srcStride`：GM 侧，单位**字节**，含义为前一块尾部到后一块头部的距离
- `dstStride`：UB 侧，单位 **32 字节块**，0 表示连续

**禁止的替代方案**：逐行循环调用 1D DataCopyPad（违反指标 3，tile 数量膨胀 32 倍）。

详见：
- `ascendc-api-best-practices/references/api-datacopy.md` 的「2D strided 跨行加载」场景
- `ascendc-tiling-design/references/elewise/tiling.md` 第七章 Multi-row Tile 优化

**经验教训**：未启用 multi-row 时，`rowSize=64` 的 case tile 被截断为 64（vs 理想 2048），tile 数量膨胀 32 倍，固定开销 dominate，性能可劣化 10~30 倍。同一算子仅 `dim` 参数不同即产生 20+ 倍延迟差异。

   **ops.h** 模式：
   ```cpp
   namespace ascend_kernel {
   at::Tensor <op_name>(<参数列表>);
   }
   ```

   **register.cpp** 模式：
   ```cpp
   #include "ops.h"
   #include <torch/library.h>

   TORCH_LIBRARY_FRAGMENT(npu, m) {
       m.def("<op_name>(<schema>) -> Tensor");
   }
   TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
       m.impl("<op_name>", TORCH_FN(ascend_kernel::<op_name>));
   }
   ```

### 步骤 2: 编写 model_new_ascendc.py + 编译验证

编写 `{output_dir}/model_new_ascendc.py`。

**🛑 Wrapper 最小化原则（对应指标 1: Wrapper 零搬运）**：

`model_new_ascendc.py` 的 forward() **禁止**包含以下操作：
- `x.permute(...)` + `.contiguous()` — 全量数据拷贝，性能杀手
- `x.reshape(...)` 强制改变张量布局 — 应在 kernel 内部处理
- `x.to(dtype)` 全量精度转换 — 应在 kernel 内部通过 Cast 处理
- 任何 `torch.*` / `F.*` 计算算子

**允许**的操作：
- `torch.ops.npu.<op_name>(x, ...)` 直接调用
- 必要时 `x.contiguous()` 仅当输入确实非连续时（kernel 要求连续输入）

**理想模式**（Wrapper 极简，kernel 原生处理所有维度/精度）：
```python
import torch
import torch_npu

def run(x, dim=-1):
    return torch.ops.npu.swiglu(x, dim)
```

**兼容模式**（当 kernel 仅支持特定布局时使用，但必须在 PERF_DESIGN.md 中记录原因）：
```python
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, x, dim=-1):
        if not x.is_contiguous():
            x = x.contiguous()
        return torch.ops.npu.swiglu(x, dim)

_model = None
def run(x, dim=-1):
    global _model
    if _model is None:
        _model = ModelNew()
    return _model(x, dim)
```

**⚠️ npu-kernelbench 兼容性**：当 solution.json 用于 npu-kernelbench 评测时：
- 禁止 `torch.ops.load_library()`（被 anti-hack 检测拦截）
- entry_point 应为模块级函数 `run()`，不是 class 方法
- runner 会自动加载编译产物 `.so`，wrapper 无需自行加载

然后调用 `.claude/skills/tilelang2ascend-translator/scripts/evaluate_ascendc.sh {output_dir}` 编译并验证（内部 cmake + make + whl 安装）。

---

### 步骤 3: 错误修复迭代

迭代上限为 **5 次**（与 ascend-kernel-developer Phase 4.5A 对齐）。每次修复前必须执行以下步骤：

#### 🛑 3.0 修复前查阅（每次修复强制执行，不可跳过）

**根据错误类型，查阅对应的 asc-devkit 文档或历史案例：**

🛑 **跨核同步专项门禁（涉及 CrossCore / WorkspaceQueue / MIX_AIC / 全零输出时强制执行）**:
- **必须先完成步骤 0-C 的全部 checklist**（含读取 ascendc-sync-guide.md 全文 + 逐条确认 6 项 checkpoint）
- 未完成步骤 0-C → **禁止** Edit/Write 任何 CrossCore/WorkspaceQueue 相关代码

| 错误类型 | 必须查阅 |
|---------|---------|
| **编译错误: API 签名不匹配** | `asc-devkit/docs/api/Ascend-C-API列表.md` → 定位 API → 查阅该 API 的独立 .md 文档确认正确签名 |
| **编译错误: 类型不匹配** | `asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/类型转换/Cast.md` 确认 dtype 支持矩阵 |
| **编译错误: GlobalTensor/LocalTensor** | `asc-devkit/docs/api/SIMD-API/基础数据结构/` 下对应简介.md |
| **运行时 vector core exception / UB 违例 / all-zero output** | ① 🛑 **优先执行步骤 0-C** 完成 sync checklist<br>② `asc-devkit/docs/guide/算子实践参考/.../TBuf的使用.md` 检查 buffer 大小<br>③ `workflows/templates/archive_tasks/rms_norm/` 对比 EXEC_KERNEL_CMD 传参模式<br>④ 检查是否有 struct 指针被传给 `EXEC_KERNEL_CMD`（常见根因） |
| **运行时 hang/死锁 / 跨核数据不流通** | 🛑 **必须先执行步骤 0-C**（含读取 ascendc-sync-guide.md 全文 + 6 项 checkpoint），再逐项排查 |
| **多核非确定性（单核正确/多核错，失败行随时序漂移）** | ① 用 `usedCoreNum=1` 单核强制复跑二分：单核对/多核错 ⇒ launch 模型正确、问题在 compute 侧数据竞争<br>② 查 Gather 源是否 alias TQue 队列 tensor（跨迭代 slot 复用，见 `references/ascendc_shuffle_patterns.md` §1.7）<br>③ 查输入/输出是否误用 TBuf（见步骤 0-C） |
| **运行时 vector core timeout (507034)** | 🛑 这是硬件级别的 core 挂起错误。按顺序排查:<br>① **work buffer 尺寸**: 检查所有 API 的 work buffer (ReduceSum/Cos/Sin/Broadcast) 是否通过 GetXxxMaxMinTmpSize 正确计算 — 硬编码不足是最常见根因<br>② **Buffer 总溢出**: 计算所有 InitBuffer 分配的总 UB 字节数，确认不超过 GetCoreMemSize(UB)<br>③ **PipeBarrier 配对**: 每个 GM→UB (MTE2) 后必须有 PIPE_MTE2 barrier; 每个 V 计算块结束后必须有 PIPE_V barrier; 每个 UB→GM (MTE3) 前必须有 PIPE_V barrier<br>④ **循环边界**: 检查所有循环的边界类型一致性 (int32_t vs int64_t)，确认不会因类型不匹配导致死循环<br>⑤ **隔离法**: 将 kernel 逐步简化为 identity copy，每次恢复一个操作，定位触发 timeout 的具体 API<br>⑥ **参考历史**: 查阅 workflows/templates/archive_tasks/ 中相似规模的融合算子，对比 work buffer 计算方式 |
| **精度不匹配 (MERE/MARE 超标)** | 调用 `ascendc-precision-debug` skill（见步骤 4） |

**⚠️ 在查阅完成并在思考中列出根因分析之前，禁止 Edit/Write 任何 kernel 代码。**

#### 3.1 分析错误输出，结合查阅结论确定根因
#### 3.2 修改 kernel/ 下的代码
#### 3.3 运行 evaluate_ascendc.sh
#### 3.4 如果 PASS → 完成，退出
#### 3.5 如果 FAIL 且迭代次数 < 5 → 回到 3.0
#### 3.6 如果 FAIL 且达到 5 次 → 进入步骤 4 (精度 skill 深度诊断)

---

### 步骤 4: 精度 Skill 深度诊断（步骤 3 耗尽后）

当步骤 3 的 5 次迭代无法解决时，按以下顺序调用精度 skill：

```
4.1 🛑 调用 Skill "ascendc-precision-debug"，传入 output_dir + 错误输出
    等待返回诊断结论和修复建议。此步骤不可跳过。

4.2 根据建议修改 kernel/ 代码，运行 evaluate_ascendc.sh

4.3 如果仍 FAIL 且连续失败 < 7 次 → 回到 4.1

4.4 如果 7 次后仍 FAIL → 
    🛑 调用 Skill "ascendc-precision-tuning"，传入 output_dir + 错误输出
    等待返回取证→审计→修复分析。此步骤不可跳过。

4.5 根据建议修改 kernel/ 代码，运行 evaluate_ascendc.sh

4.6 如果仍 FAIL 且连续失败 < 5 次 → 回到 4.4

4.7 如果所有步骤耗尽仍 FAIL → 报告当前状态，记录 trace
```

## 精度验证标准

**五类决策矩阵**（由 `verification_ascendc.py` 自动判定）：

| 类别 | 触发条件 | 判定标准 |
|---|---|---|
| 非计算类 | `ASCENDC_NON_COMPUTE=1` | view-as-int 二进制完全一致（含 NaN bit pattern） |
| bool 输出 | 输出 dtype 为 bool | `torch.equal` 严格相等 |
| 整数计算类 | 输入最高精度为 int 且输出为 int | `|actual − golden| == 0` |
| 量化计算类 | 输入最高精度为 float 且输出为 int | `|actual − golden| <= 1` |
| 浮点计算类 | 输出为 float 类型 | 三项 AND 判定（max_error_cap + matched_ratio ≥ 0.9 + MERE < rel_threshold） |

- **输入类型自动推断**：从实际输入 tensor 中取最高精度 dtype 分类为 float / int / no_tensor。
- **浮点三项判定**：① 100% 元素满足 `|diff| <= atol + rtol * |golden|`；② 分桶 matched_ratio ≥ 0.9（小值域绝对误差 / 正常域相对误差）；③ MERE（均值相对误差）< 阈值。三项全部通过才算通过。
