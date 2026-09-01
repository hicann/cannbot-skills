# 编码红线与解决方案

> 本仓算子编码中明确禁止的红线。每条红线均为「违反 = 检视不通过」的硬约束，触发即为阻塞项。逐条核对；命中任一条，按「解决方案」列修正后方可通过。
>
> 红线分两类：**A 类 = 提交反作弊**（通用算子质量底线，评测集提交时违反 = 整算子 0 分，最高优先级）；**B 类 = 通用 Ascend C 编码**（代码质量阻塞项）。
> A 类权威原文在评测集的提交规则文档（如 cann-bench `docs/guide/submission_rules.md`），本文为摘要——检视时以原文为准。

---

## A 类：提交反作弊红线（评测集提交时违反 = 整算子 0 分）

A 类红线是算子实现的通用质量底线——要求「提交者实现真实 NPU kernel」。以下行为无论是否有评测集均为阻塞级问题；**评测集提交时（模式 A）命中即判无效提交 = 整算子 0 分**。

### A1. 调用 PyTorch / torch_npu 内置计算 API 代算（无效提交）

候选算子执行路径中，**不应直接调用** PyTorch / torch_npu 内置计算 API 完成目标算子，即使只把部分计算交给现成 API 也属无效。

| 红线 | 检测 | 解决方案 |
|------|------|---------|
| 包装层调 `F.softmax` / `x @ y` / `torch.mm` / `F.conv2d` 等 | 审查 plugin 层调用 | 核心计算由提交 kernel 内 Ascend C 原语完成 |
| 改用 `torch.ops.aten.matmul` / `x.matmul(y)` / `torch.nn.functional.*` 等同类绕过 | 审查 aten/torch_npu op 调用 | 同上——计算主体在 kernel |

> 允许在 **kernel 内**使用 Ascend C 原生 API/intrinsic（`AscendC::Add`/`Mul`/`Exp`/`Mmad` 等）——它们编译期成为提交 kernel 的一部分，不等同于包装层调现成算子。

### A2. 用 PyTorch / torch_npu 处理输入输出 tensor（无效提交）

输入预处理、输出后处理、中间 tensor 变换也是目标算子实现的一部分。不能用 PyTorch / torch_npu tensor API 先完成 transpose/permute/contiguous/reshape-copy/cast/slice/gather/scatter 等实质性数据搬运，再交给 kernel。

| 红线 | 检测 | 解决方案 |
|------|------|---------|
| `x.transpose(0,1).contiguous()` 后再 launch kernel | 审查 plugin 层 tensor 操作 | transpose/permute 在 kernel 内用 Ascend C 数据搬运实现 |
| `y.permute(0,2,1).contiguous()` 返回输出 | 审查返回前 tensor 操作 | 输出重排在 kernel 内完成 |
| `x.cpu()` 处理后搬回（见 A4） | 审查 device 迁移 | 全程在 NPU kernel 完成 |

> 此类 I/O 搬运属**人工审查**判定（框架不自动拦截 Gather/Transpose 等以其为核心语义的算子）；matmul/conv/softmax 等计算类绕过由 `TorchOpGuard`/`DeviceResidencyGuard` **自动拦截**（默认 block）。

### A3. 路由到 CANN 内置同名算子（无效提交）

提交工程不应只是把任务转发给评测环境已有的 CANN 内置同名算子。

| 红线 | 检测 | 解决方案 |
|------|------|---------|
| 包装层调 `aclnnSoftmaxGetWorkspaceSize` + `aclnnSoftmax` | 审查 aclnn 调用 | 提交自己的 kernel，kernel 内用 Ascend C 原语组合 |
| `ADD_TO_LAUNCHER_LIST_AICORE(Softmax)` 注册/转发现成 AiCore 实现 | 审查 launcher 注册 | 删除转发，实现自有 kernel |

### A4. CPU fallback 或未执行提交 NPU kernel（无效提交）

候选算子不能把输入搬回 CPU 计算再把结果搬回 NPU。即使结果数值正确，只要核心计算不是由提交 NPU kernel 完成，也属无效。

| 红线 | 检测 | 解决方案 |
|------|------|---------|
| `torch.exp(x.cpu()).to(x.device)` | 审查 `.cpu()` 调用 | 全程 NPU kernel 计算 |
| `(x.cpu() + y.cpu()).npu()` | 审查 host 侧计算 | 同上 |

### A5. 缓存输出 / 固定输出 / 按输入地址命中（无效提交）

实现不能缓存某次执行结果在后续 repeat 中直接返回；也不应依赖固定输入地址/shape/dtype/attrs/公开 case 特征返回预设结果。

| 红线 | 检测 | 解决方案 |
|------|------|---------|
| `_cache = {}; key=x.data_ptr(); if key in _cache: return _cache[key]` | 审查缓存逻辑 | 每次真实计算，不缓存 |
| `_last_output` 固定返回 | 审查全局状态 | 删除固定输出 |
| 按公开 case 的固定 shape/dtype 返回常量 | 审查 shape/dtype 分支条件 | 实现须泛化到所有合法输入 |

> 性能测量阶段框架**轮换输入地址**（每个 repeat 喂独立 clone，`data_ptr()` 不同）——按地址命中的缓存会 cache miss 并在精度复检中暴露。该轮换仅作用于性能阶段；精度阶段不轮换。

### A6. 篡改 profiler 或 timing API（无效提交）

提交代码不应 monkey-patch / 替换 / 删除评测环境的计时、同步、性能采集或运行时 API。

| 红线 | 检测 | 解决方案 |
|------|------|---------|
| `torch.npu.synchronize = lambda *a, **k: None` | 审查 monkey-patch | 删除篡改 |
| `torch_npu.profiler.profile = fake_profile` | 审查 profiler 替换 | 删除篡改 |

### A7. 返回 FakeTensor / 懒求值包装器 / 伪 Tensor（无效提交）

候选算子必须返回真实 `torch.Tensor`。用对象包装真实计算、延迟求值、返回 Tensor 子类伪装结果，都破坏评测边界。

| 红线 | 检测 | 解决方案 |
|------|------|---------|
| `class LazyTensor: ...` 包装返回 | 审查返回类型 | 返回真实 `torch.Tensor` |
| 返回 Tensor 子类伪装结果 | 审查返回类型 | 同上 |

---

## B 类：通用 Ascend C 编码红线

以下为 kernel 实现的代码质量阻塞项，违反 = CP5 检视不通过（回退 3.1 修复）。

### B1. 硬件参数硬编码（阻塞）

所有硬件参数必须运行时动态获取，禁止写死。

| 红线 | 检测 | 解决方案 |
|------|------|---------|
| 写死核数 `blockDim = 8;` | `grep -n "blockDim\s*=\s*[0-9]" *.cpp` | `platform_ascendc::PlatformAscendCManager` 动态获取核数 |
| 写死核索引 `blockIdx = 0;` | `grep -n "blockIdx\s*=\s*[0-9]" *.cpp` | `AscendC::GetBlockIdx()` 获取 |
| 写死分块大小 `constexpr TILE = 4096;` | 目视 + grep | 根据 UB 容量动态计算或验证固定值在所有目标 SoC UB 内 |
| 任何硬编码 UB 大小 | 目视 | 运行时 `GetCoreMemSize(UB, ubSize)` 获取 |

### B2. 数据搬运 API 误用（阻塞）

| 红线 | 原因 | 解决方案 |
|------|------|---------|
| `DataCopy(GM, UB)` / `DataCopy(UB, GM)` 处理非对齐 | 不支持非对齐，易致隐蔽 bug | 统一改用 `DataCopyPad` + `DataCopyExtParams`/`DataCopyPadExtParams` |
| `GlobalTensor::SetValue/GetValue`（非调试） | 逐元素访问效率极低 | `DataCopyPad` 批量搬运 |

```cpp
// 正确：DataCopyPad 处理非对齐
AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
```

### B3. 流水线与数据流（阻塞）

| 红线 | 解决方案 |
|------|---------|
| DataCopy 后未用 EnQue/DeQue 同步 | DataCopy 后补 EnQue/DeQue 配对同步 |
| `AllocTensor` 无对应 `FreeTensor`（内存泄漏） | 每个 Alloc 配对 Free；`grep -c AllocTensor`/`grep -c FreeTensor` 核对数量 |
| EnQue/DeQue 未配对 | 配对；`grep -c EnQue`/`grep -c DeQue` 核对 |
| 冗余 PipeBarrier（同 pipe 连续操作间加 barrier） | 只在跨 pipe 数据依赖点保留 barrier |
| **热路径（主循环内）无差别使用 `PipeBarrier<PIPE_ALL>`** | 依赖只涉及单一队列时用定向 barrier，见下方「定向 barrier 优先原则」 |

> Pipe 归属：PIPE_MTE2（GM→UB）、PIPE_V（矢量/归约/Cast）、PIPE_MTE3（UB→GM）、Scalar。跨 pipe 且存在 RAW/WAW 依赖才需 barrier；同 pipe 硬件保序。
>
> **定向 barrier 优先原则**：`PipeBarrier` 按队列定向同步——依赖只涉及单一队列时用定向 barrier；`PIPE_ALL` 会排空全部队列（MTE2/MTE3/V/S 全等到调用点），每次都是整段流水线气泡，主循环内逐操作全同步会把 double buffer 重叠度打回串行。依赖映射：等 GM→UB 完成 → `PIPE_MTE2`；等 Vector 完成 → `PIPE_V`；等 UB→GM 完成 → `PIPE_MTE3`；等 Cube 完成 → `PIPE_AIC`；标量 `GetValue/SetValue` 回读/回写 UB 属 V 依赖 → `PIPE_V`。`PIPE_ALL` 仅限多队列汇聚（VF 读 UB 前、跨核 flag 收发前）与调试定位；调试定位后必须替换为定向 barrier 或 EnQue/DeQue 再交付。

### B4. Kernel 内禁止事项（阻塞）

| 红线 | 解决方案 |
|------|---------|
| Kernel 内使用 `std::` 命名空间函数 | 改用 Ascend C API |
| 动态内存分配（`new` / `malloc`） | 用 `TPipe` + `TQue`/`TBuf` 管理内存 |
| 递归调用 | 改写为迭代 |
| 使用未初始化变量 | 使用前初始化 |

### B5. API 用法未经验证（阻塞）

| 红线 | 解决方案 |
|------|---------|
| 凭记忆/猜测写入 API，未查文档 | 用任何 API 前通过 `ascendc-docs-search` 查阅 `{API_NAME}` 官方文档 |
| 检视意见推荐 API 未附文档来源 | 推荐 API 的修复建议须附官方文档来源，禁止凭记忆 |
