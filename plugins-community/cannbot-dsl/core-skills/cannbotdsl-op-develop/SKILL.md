---
name: cannbotdsl-op-develop
description: "使用 CANNBotDSL 开发、调试或评审昇腾 NPU 算子时使用。适用于 Python DSL kernel 编写、CANNIR/AscVec/AscCube lowering、AscendC translate、runtime/NPU 精度验证、端到端示例调试、补充 pytest 或设计文档。Triggers: CANNBotDSL, cannbotdsl, CANNIR, AscVec, AscCube, lowering, TranslateToAscendC, local_slice, layout algebra, mixed AIV/AIC, DSL 算子开发, runtime 精度调试。"
---

# CANNBotDSL 算子开发技能

本 skill 用于在 cannbot-arena 仓（当前工作区根目录）内开发、调试和评审 CANNBotDSL 算子。优先基于源码核对，而不是凭记忆假设 API。

## 快速路由

| 任务 | 首先读取 |
|------|----------|
| 写新 DSL 算子或示例 | `../cannbotdsl-api-reference/SKILL.md` |
| **开发 Flash Attention 类算子** | `../../op-skills/cannbotdsl-flash-attention/SKILL.md`（blueprint、buffer budget、已知陷阱） |
| **开发 MLA 类算子** | `../../op-skills/cannbotdsl-mla/SKILL.md`（归约轴 concat 分解、两向 chunk、buffer 预算、head folding） |
| **静默算错：编译通过但数值不对** | `../cannbotdsl-probe-debug/SKILL.md`（最小探针、参数扫描找边界、误差形态定位） |
| 写或重构融合算子的整体架构、shape/layout 入口、硬件角色分层 | `references/fusion-kernel-architecture.md` |
| 把 split launch / host loop / 多阶段算子收敛成单 kernel | `../cannbotdsl-op-design/references/single-kernel-fusion-lessons.md`、`references/fusion-kernel-architecture.md` |
| 调试 buffer/layout/tail | `../cannbotdsl-tiling-design/SKILL.md` |
| 调试 vector fusion/VF、BF16/FP16 GM 累加或 AIV 生命周期 | `../cannbotdsl-vf-fusion/references/vf-lifecycle-dtype.md` |
| 调试 runtime/NPU 精度 | `../../debug-skills/cannbotdsl-precision-debug/references/runtime-precision-checklist.md` |
| 整理 kernel 代码结构、三层职责分离、L1 归属、VF 封装、host 精简 | `../cannbotdsl-kernel-structure/references/kernel-coding-style.md` |
| 调试混合 AIV/AIC 或 CANNBotDSL 工具链问题 | `../../debug-skills/cannbotdsl-precision-debug/references/mixed-kernel-debug-lessons.md` |

## 工作流

1. **确认目标**
   - 算子语义、shape/dtype/layout、数据范围、目标后端。
   - 明确是新开发、修 bug、补测试、补文档还是评审。
   - 先冻结公开调用契约：公开入口函数（如 `sigmoid(x, *, out=None)`）的真实输入、输出、标量属性、dtype、shape 约束和错误条件。内部 scratch、workspace、tiling 不应作为公开参数。

2. **定位代码路径**
   - 用 `rg` 查 API、测试。
   - 优先打开最小相关文件，不批量读取无关文档。

3. **设计实现**
   - 选择 DSL API、buffer 层级、layout、tiling/tail、同步策略。
   - **落地设计阶段定的常规 double buffer**：DESIGN.md（`../cannbotdsl-op-design/SKILL.md §7`）标为 `depth≥2` 的 storage 实现即用 `Channel(..., depth=N)`，不要默认单缓冲；单块临时量才用 Buffer。实测不可行时回退设计（`DESIGN_ERROR`）而非静默 `depth=1`。
   - 设计验证命令，先最小再目标场景。
   - 写融合算子或重构前必须先按 `references/fusion-kernel-architecture.md` 建立公开调用契约、逻辑轴角色、buffer 生命周期、GM 例外和同步/生命周期表。
   - 写或重构 `with vf()`、混合 AIC/AIV handoff、BF16/FP16 GM add 前，必须先读 `../cannbotdsl-vf-fusion/references/vf-lifecycle-dtype.md`。
   - 默认实现走 UB/L1/L0/FIXPIPE 通路；GM scratch 只能用于公开输入输出、跨执行边界/跨 kernel、容量限制或已确认工具链限制，不能因为"临时可跑"而落 GM。

4. **编码与验证**
   - 代码改动贴近现有风格。
   - 对应补测试或更新已有测试（遵循三件套结构）。
   - 运行最小验证命令，记录真实结果。
   - 验证链路按 `py_compile` → `pytest --collect-only` → 最小 NPU case → 精度验证推进；任一阶段不能运行时记录阻塞原因。

5. **调试分流（按症状进对应 skill，不在本 skill 展开流程）**
   - 编译/运行时错误分类（A/B/C/D/F 型）→ `../../debug-skills/cannbotdsl-runtime-debug/SKILL.md`。
   - NPU 精度/随机输出/分层定位 → `../../debug-skills/cannbotdsl-precision-debug/SKILL.md`。
   - NPU crash / hang / sync 死锁 → `../../debug-skills/cannbotdsl-crash-debug/SKILL.md`（设备 plog 读到根因交 `../../debug-skills/cannbotdsl-npu-plog-diagnosis/SKILL.md`）。
   - 通用原则：先最小复现确认问题层级，只有能定位到前端/IR/lowering/translator/runtime 的问题才归类为 CANNBotDSL 缺陷。

## 环境模板

NPU runtime 验证的环境模板权威定义在 `../cannbotdsl-env-setup/SKILL.md`；实际路径、CANN 安装位置按当前机器调整。

bench 脚本常用环境变量：
```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0   # 避免 torch 自动加载 NPU 后端
```

## 常用命令

只运行与改动相关的最小命令。若命令依赖 NPU/CANN 环境而当前不可用，明确标记 blocked。

## 代码路径速查

| 区域 | 路径 |
|------|------|
| DSL API 参考 | `../cannbotdsl-api-reference/SKILL.md` |

## 质量门禁

交付前检查：

- 代码能解释每个 buffer、slice、sync 和 layout 选择。
- 回答和阶段记录必须说明本轮使用过哪些 skill/reference；如果用户要求遵从某个 skill 仓或规范，未读取就是门禁失败。
- 注释只描述当前文件职责、对象生命周期、buffer ownership 和同步语义；不要用"像某参考实现一样"替代当前代码自己的语义说明。
- 融合算子优先保持公共接口 layout，kernel 内做 tile 级映射；不要无理由在 host 侧整体转成内部大 layout。
- 硬件角色、数据流角色、边界/tiling helper、顶层调度职责清晰，复杂 helper 归入对应角色类，测试参数使用公共接口维度。
- copy engine、UB/L1/L0 storage 在最接近使用者的类里申请；单块临时量用 `Buffer(MemLoc.*, ...)`，depth-N/同步 storage 用 `Channel(..., depth=N)`。
- buffer ownership 要和 API 迁移同步完成：角色类应按职责内部申请并持有自己的 buffer/copy engine，顶层 kernel 只负责 tile 调度、共享 handoff 的对象绑定和调度顺序。
- `SHAPE_LIST`、`run()` 和 pytest 参数使用公共接口维度；base-block、chunk、结构化辅助数据、scratch 是 kernel 内部实现细节。
- 常量、方法名和注释表达真实语义：不用 `C/D` 裸缩写、`_emit_xxx` 泛名或 `sum/pack` 混淆动作。
- 目标路径至少有一个验证命令。
- 新增测试不依赖个人绝对路径或临时中间产物文件。
- 不引入未说明的 dtype/shape 限制。
- 结论必须说明问题归因层级，不能把不同层级混写。

## 常见风险（编码期 API 陷阱）

- `local_slice` 静态 offset 在 vector lowering 或 UB->UB `mem_copy` 中可能丢失，且不支持动态 SSA offset。
- `mem_copy(..., transpose=True)` 对 unsupported copy 方向缺少 verifier，可能静默忽略语义。
- `@jit` 依赖 file-backed source，`python -c` / REPL / notebook 需要额外处理。
- NPU packaged operator 缺失导致校验侧 torch_npu 报错；优先 CPU 创建 tensor 后 `.npu()`。
- `tile_view`/`local_slice` view 不一定有稳定同步标识；跨 PIPE 同步应锁原始 owning buffer，view 只作为算子操作数。
- 高层 `cast(int8_dst, fp16_src)` 会产生 stride-2 反交织静默错误（见 `../cannbotdsl-api-reference/SKILL.md` §5 #20）；退到 raw vf 用 `vstore_pack(mode=B32_TO_B8)`。
- `const_expr(cond)` 守卫在 cond 变负时静默失效（如 `NPAD = VH - BMV` 变负时 `if const_expr(NPAD > 0):` 被跳过，越界写读照常发生）。给守卫加下界断言（`assert VH >= BMV`），或让参数在 host 侧推导而非硬编码。详见 `../cannbotdsl-vf-fusion/SKILL.md` 陷阱 11。
- 把循环拆成"便宜 n-1 轮 + 尾部一次贵的调用"会让常驻 channel 变成一写 + 两个读作用域，`cannir-resolve-channel-operands` 编译期拒绝。在循环体内用运行时 `scf.if` 分支选择路径是合法的。详见 `../cannbotdsl-kernel-structure/SKILL.md` 陷阱 8。

## 输出要求

回答或阶段结果中必须说明：

- 修改了哪些文件。
- 运行了哪些验证命令。
- 哪些验证没跑及原因。
- 残留风险或后续建议。
