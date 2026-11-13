# Convolution 范式 AscendC 语言适配层

> **层级**：AscendC 语言适配层（adapters/ascendc/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。

## 1 定位

本目录承载 Convolution 范式中**强耦合 AscendC 语言/工具链**的知识：API 实体与参数结构体、类型与 buffer 基线、同步写法、编译配方。开发语言为 AscendC（由调用方明确提供）时，与通用知识层组合消费：

```
通用算法/模型（generic/）× AscendC 实现细节（本目录）→ AscendC 设计与代码产出
```

指令的**语义、参数限制与数据流**（LOAD3D 怎么滑窗、容量不等式等）属语言中立知识，在 [../../generic/compute-flow.md](../../generic/compute-flow.md) §4 与 [../../generic/tiling.md](../../generic/tiling.md)，本目录不重复。

## 2 阅读路径

```
任务流程                        入口文档                         辅助参考实现
──────────────────────────    ───────────────────────────    ──────────────────────────
骨架搭建                  ──►  buffer-and-types.md         │
  │                            (类型/buffer/TQue-TBuf 选型)    │
  ▼                                                           │
逐条指令编码              ──►  instruction-mapping.md       ├──  example/conv2d/
  │                            (指令↔API 对照/位域填充)        │    (tiling + kernel
  ▼                                                           │     参考实现)
同步插入                  ──►  sync-implementation.md       │
  │                            (事件对/unitFlag/双缓冲)        │
  ▼                                                           │
bias 功能                 ──►  bias-bt-path.md              │
  │                            (BT 装载序列/Mmad 初值)         │
  ▼                                                           │
多路径模板组织            ──►  tilingkey-templates.md       │
                              (key 打包/模板枚举)             │
```

## 3 文档索引

| 文档 | 覆盖 | 典型消费时机 |
|------|------|-------------|
| [buffer-and-types.md](buffer-and-types.md) | L0C 累加类型基线、TPosition↔物理 buffer 对照、TQue/TBuf 选型、平台真值查询 | 骨架搭建 |
| [instruction-mapping.md](instruction-mapping.md) | 硬件指令 ↔ AscendC API 对照、参数结构体、位域填充 | 逐条指令编码 |
| [sync-implementation.md](sync-implementation.md) | L0 级事件对写法（单/双缓冲）、TQue 自动事件、unitFlag 机制 | 同步插入 |
| [bias-bt-path.md](bias-bt-path.md) | BT 装载序列（6 步表）、Mmad 初值参数、越界适配、BT 容量约束 | bias 功能 |
| [tilingkey-templates.md](tilingkey-templates.md) | key 生成三步、模板组织三件套、维度集最小化 | 多路径模板组织 |
| [example/conv2d/](example/conv2d/) | conv2d 推理算子参考实现（tiling 分核/单核切分 + kernel 七层迭代） | 编码前核对 |

## 4 公共知识依赖

以下跨范式 AscendC 公共知识位于范式外，按需展开：

| 主题 | 位置 |
|------|------|
| 同步与一致性 | [../../../../common/sync-and-consistency.md](../../../../common/sync-and-consistency.md) |
| DataCopyPad 规则 | [../../../../common/datacopypad-rules.md](../../../../common/datacopypad-rules.md) |
| API 白名单（存在性唯一判据） | [../../../../../knowledge/api/regbase_api_whitelist.md](../../../../../knowledge/api/regbase_api_whitelist.md) |

## 5 消费规则

| # | 规则 |
|---|------|
| 1 | 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载其他语言适配层 |
| 2 | 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题 |
| 3 | 编码前必须检视参考实现 [example/conv2d/](example/conv2d/)（tiling + kernel 主体），并按当前算子规格重新核定算法与参数 |
| 4 | **拉起接入不是本范式的内容**：`<<<>>>` 内核直调与 aclnn 注册两段式是拉起算子的两种接入形态，分别由上层工作流技能保证——本仓 `ops/ascendc-direct-invoke-template`（直调）、`ops/ascendc-registry-invoke-template`（注册，含 Cube kernel 核类型声明 `ASCENDC_TPL_KERNEL_TYPE_DECL(KT, ASCENDC_TPL_AIC_ONLY)`；漏声明会按 VectorCore 编译、`if ASCEND_IS_AIC` 守卫被剔除、算子空跑输出全 0，故障特征见 [verification.md](../../generic/verification.md) §7.2） |
| 5 | 通用层引用的能力真值（容量、指令限制）以所用头文件树实际定义为准；出现版本差异按 [buffer-and-types.md](buffer-and-types.md) §1.2 处理并声明假设 |

## 6 正式算子工程封装

本范式覆盖卷积算法与验证知识；工程骨架细节（目录规范、构建脚本、安装部署、拉起接入）见仓内技能：

- aclnn 注册形态：`ops/ascendc-registry-invoke-template`
- `<<<>>>` 内核直调形态：`ops/ascendc-direct-invoke-template`
