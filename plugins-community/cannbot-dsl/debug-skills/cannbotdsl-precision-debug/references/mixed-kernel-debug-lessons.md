# CANNBotDSL 工具链问题调试经验

## 适用边界

本文件只沉淀 CANNBotDSL 前端、CANNIR、lowering、TranslateToAscendC、runtime/debug tooling 的问题定位经验。

本文件按可复现的工具链行为归因，不用具体算子实现细节替代 CANNBotDSL 前端、IR、lowering、translator 或 runtime/debug tooling 证据。

## 问题归因标准

归类为 CANNBotDSL 问题，需要满足至少一项：

- 同一 DSL 语义在 lowered IR/AscendC 中被丢失或改变。
- CANNBotDSL API 接受了 unsupported 语义但没有报错或 warning。
- CANNBotDSL 前端能力边界不清晰，报错不能指导用户迁移。
- CANNBotDSL verifier 缺少静态检查，明显 runtime 风险只能靠人工 review。
- CANNBotDSL runtime/debug tooling 无法定位阶段或同步事件。

不满足以上条件时，不写入 CANNBotDSL 改进建议。

## 优先排查项（快速索引）

1. `local_slice` 静态 offset 是否进入 vector op、DMA 和 UB→UB copy。
2. UB→UB `mem_copy` 是否同时使用 source 与 destination offset。
3. `local_slice(offset=SSA)` 是否误用当前不支持的动态 offset。
4. 动态 `if/for/while` 是否位于 `@jit/@kernel` AST 预处理区域。
5. `mem_copy(..., transpose=True)` 是否用于已支持 copy 方向。
6. sync id 是否需要静态 registry/audit。
7. 现有手段是否足以判断问题出在编译期还是 runtime 阶段。

## 高价值最小复现

### `local_slice` 静态 offset

现象：非零 `offset` view 在 vector op 或 copy 中像 base 0。

定位：

- 用最小 UB buffer 和 `local_slice(offset!=0)` 构造 `muls/add/reduce/mem_copy`。
- 检查 AscVec IR 中 `loadalign/storealign/copy_*` 是否包含非零 element offset。
- 如果 IR offset 正确但 NPU 错，继续查 translator 或 AscendC CAPI 参数。

### UB→UB `mem_copy` 双 offset

现象：目标写入位置正确但内容来自源 base 0，或源正确但覆盖目标 base 0。

定位：

- 构造 source view 和 destination view 都有非零 offset 的 UB→UB copy。
- 检查 `loadalign` 使用 source offset，`storealign` 使用 destination offset。
- plain copy 和 format convert 应分别覆盖。

### 动态 offset 能力边界

现象：`local_slice(offset=动态 i64 SSA)` 抛 `TypeError`。

定位：

- 当前这是 CANNBotDSL API 能力边界，不是算子 bug。
- 如果需求必须动态 offset，需要扩展 CANNIR op/type、verifier 和 lowering。
- 临时方案只能是静态展开、`tile_view(coord=SSA)` 或专门动态寻址 API。

### 动态控制流 scope

现象：普通 helper 中对动态 i64 SSA/dynamic bool 做 Python `if` 报 truthiness 错误。

定位：

- 动态控制流必须位于 `@jit/@kernel` AST 预处理区域。
- 普通 helper 不会被改写，不能承载动态 `if/for/while`。
- 可选迁移：inline 到 `@jit/@kernel`，或使用显式算术 select。

### `transpose=True` verifier 缺口

现象：`mem_copy(..., transpose=True)` 用在 unsupported copy 方向时，transform 通过但语义被忽略。

定位：

- L1→L0B 是已知支持路径，应能看到 transpose 属性或对应 CAPI。
- L1→L0A 等 unsupported 路径应由 verifier 报错。
- 静默丢弃用户显式语义属于 CANNBotDSL 工具链问题。

### `@jit` source 可用性

现象：`python -c`、stdin、REPL、notebook cell 中定义的 `@jit` 函数报 `Failed to get source`。

定位：

- CANNBotDSL AST preprocess 依赖 `inspect.getsourcelines`。
- 当前 workaround 是把函数保存到 `.py` 文件中运行。
- 若要改善，需要 source registry 或交互式专用入口。

### sync id 静态审计

现象：同一 sync id 被多个无关 handoff 复用，CANNBotDSL transform 不报警。

定位：

- 当前 verifier 主要检查 pipe 合法性，不检查 sync id 生命周期。
- 建议用临时 registry 记录 id、data、producer、consumer、pipe。
- 内建 debug mode 应输出 sync event 表。

### JIT/debug 可观测性

现象：长 kernel 无输出时，不清楚卡在编译、launch 还是 sync。

定位：

- 用阶段打印（`print_tensor`/`print_scalar`）在 kernel 各阶段插锚点，缩小卡点范围。
- 当前缺少统一阶段耗时和 `kernel-debug.json`。
- 建议 JIT runner 输出阶段耗时、copy/sync/matmul/buffer 结构化摘要。

## 复现要求

每个 CANNBotDSL 问题单独一个 case：

- 独立脚本，不依赖大算子。
- 中文注释写清问题、失败方式、成功方式、修改建议。
- 对照输出只用于说明 DSL 可观察行为，不能替代工具链归因证据。
- 优先验证能否编译通过，其次再跑 NPU runtime。

## 常用验证

```bash
python -m py_compile （problems 样例不在本仓） （problems 样例不在本仓）
pytest （problems 样例不在本仓） -q -s
pytest <case> -q -s
```

## 典型结论模板

- **非工具链问题**：最小复现无法定位到 CANNBotDSL 前端、IR、lowering、translator、runtime 或 debug tooling。
- **CANNBotDSL API 边界**：API 当前明确不支持，例如动态 `local_slice.offset`。
- **CANNBotDSL verifier 缺口**：unsupported 语义被接受但未报错。
- **CANNBotDSL lowering bug**：DSL 语义在 CANNIR→AscVec/AscCube/AscendC 中丢失。
- **CANNBotDSL tooling 缺口**：缺少阶段耗时、sync event 表等定位能力。
