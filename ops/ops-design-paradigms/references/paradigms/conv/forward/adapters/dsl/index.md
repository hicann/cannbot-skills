# Convolution 范式 CANNBotDSL 语言适配层

> **层级**：CANNBotDSL 语言适配层（adapters/dsl/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。

## 1 定位

本目录承载 Convolution 范式中**强耦合 CANNBotDSL（下文简称 DSL）**的知识：DSL 实体映射、能力谓词真值、Host/Kernel 代码组织、尾块与同步写法。开发语言为 `dsl` 时，与通用知识层组合消费：

```
通用算法/模型（generic/）× DSL 实现细节（本目录）→ CANNBotDSL 设计与代码产出
```

与 AscendC 适配层**互斥**：命中 `dsl` 后不加载 [../ascendc/index.md](../ascendc/index.md)，也不把其 `TilingKey/TilingData/TPosition/FMatrix` 当作本语言的设计依据。两边共享的只有通用层语义。

## 2 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| 方案设计 | [../../generic/index.md](../../generic/index.md) → [capability-mapping.md](capability-mapping.md) | 通用层的场景/数据流/切分/验证模型，叠加 DSL 的平台真值、Buffer 形态与约束归属 |
| Host 与切分落地 | [implementation-guide.md](implementation-guide.md) §1–§3 | 三层职责、指令/容量硬约束、解析代价模型、有界搜索、多核任务划分、空输出与缓存 |
| Kernel 编码 | [implementation-guide.md](implementation-guide.md) §4–§6 | 连续任务区间、n 分组/输出行组/k 分块、整行带与 B 驻留、M 超块与行展平、`load3d` 窗口与尾块、Buffer 与同步 |
| 与 AscendC 的差异核对 | [implementation-guide.md](implementation-guide.md) §7 | TilingKey/TilingData、FMatrix/NZ、unitFlag、fullLoad 的对应关系 |
| 性能调优 | [tuning.md](tuning.md) | 测量口径、有效手段与收益量级、无效尝试、结构上限、代价模型校准与交接清单 |
| 编码前核对 | [example/conv/](example/conv/README.md) | 已跑通的 1D/2D/3D 单文件实现（底层接口：`load3d` + 手写 NZ 打包）+ 设备验收用例 |
| 验证与缺口 | [capability-mapping.md](capability-mapping.md) §5–§6 | 数值合同、待验证项与复现方式 |

## 3 文档索引

| 文档 | purpose | read_when |
|------|---------|-----------|
| [capability-mapping.md](capability-mapping.md) | 平台/版本真值、类型与 Buffer 绑定、指令↔DSL API 对照、约束归属、待验证缺口、数值合同 | 需要确认某个能力在 DSL 上是否存在/是否已验证、核算片上预算、或声明假设时 |
| [implementation-guide.md](implementation-guide.md) | Host/Kernel 分层、tiling 搜索与代价模型、容量公式、多核映射、kernel 骨架、尾块/同步、与 AscendC 的差异 | 开始写 DSL 卷积实现、或把设计落到具体 API/循环结构时 |
| [tuning.md](tuning.md) | 性能调优经验：测量口径、有效结构手段与收益量级、无效/有害尝试、结构上限判断、代价模型校准方法、交接清单 | 需要调优、复现性能结论、或判断是否已到结构上限时 |
| [example/conv/](example/conv/README.md) | Conv1D/2D/3D 参考实现（`conv_dsl.py`，底层接口）+ 设备验收（`tests/test_device.py`） | 编码前核对写法、或需要复现设备验收时 |

## 4 能力真值来源

1. **运行时平台参数**：容量/核数/指令上限的获取方式见 [capability-mapping.md](capability-mapping.md) §1。
2. **DSL 公开 API 文档**（CANNBotDSL 仓 `docs/zh/api/`）：`data-movement/convolution/load3d.md`、`data-structures/convolution/load3d-*.md`、`data-movement/mem-copy.md`、`data-structures/storage/buffer/`。`data-movement/convolution/conv2d-*` 与 `data-structures/convolution/` 的专用接口文档可读，但其 N 尾块缺陷见 [capability-mapping.md](capability-mapping.md) §3.1，不得据此实现尾块路径。
3. **头文件/包源码**：接口存在性与参数约束以实际安装的 `cannbotdsl` 包为准；版本差异按 [capability-mapping.md](capability-mapping.md) §1 记录。

平台参数（容量、核数）必须从 `get_mem_size` / `get_platform_info` 读取，禁止写死。

## 5 消费规则

| # | 规则 |
|---|------|
| 1 | 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载 AscendC 适配层、不参考其 API 与代码模板 |
| 2 | 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题 |
| 3 | [capability-mapping.md](capability-mapping.md) §5 中标注**待验证**的能力不得当作已验证真值；据此得出的设计结论必须声明假设并给出验证义务 |
| 4 | 编码前必须检视 [example/conv/](example/conv/README.md)，但不得把示例的 tile/阈值/用例当作通用默认值——按当前算子规格重新核算 |
| 5 | 数值验收以独立 golden + 逐元素容差为准；编译通过、IR 校验通过都不等于数值正确 |
| 6 | `cannbotdsl.ops.conv` 专用接口在 N 方向尾块缺陷修复前，**不得用于 `BN > n_actual` 的路径**；参考实现统一走底层接口 |
| 7 | 性能结论必须标注计时口径（min/p25/median、重复次数、平台与 DSL 版本）；共享设备上的单次 median 不作为结论；原始数据不入库 |
| 8 | **拉起接入不是本范式的内容**：aclnn 注册、`<<<>>>` 直调、AOT/Native 发布由上层工作流负责，本层只交付计算主体 |

## 6 参考实现入口

[example/conv/](example/conv/README.md)：

- `conv_dsl.py`：单文件，1D/2D/3D 共用；Host 指令预检 + 有界 tiling 搜索（MT/NT≤224、CT≤128、B 驻留/流式、L1/L0 双缓冲、M 超块、多行输出展平）+ 多核流式 Cube kernel；走底层接口 `load3d` / `from_tensor(physical_layout=...)` / `mem_copy` / `matmul`。
- `tests/test_device.py`：设备验收（快速集 3 类 rank + 程序缓存复用 + 空输出零编译；31 规格泛化矩阵及其空输出）。**需要 NPU，不在仓库普通 CI 执行**。
- 验证方法以 [../../generic/verification.md](../../generic/verification.md) 为准；设备测试需要 NPU、不在仓库普通 CI 执行。性能数据、采集/调参脚本与环境快照留在库外验证报告，调参经验见 [tuning.md](tuning.md)。

该示例登记在范式路由表 `example-routes.yaml` 的 `SlidingWindow.dsl`，由 `resolve_example.py` 按 `op.paradigms × op.language` 注入。
