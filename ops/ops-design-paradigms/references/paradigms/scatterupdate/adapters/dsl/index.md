# ScatterUpdate 范式 CANNBotDSL 语言适配层

> **层级**：DSL 语言适配层（adapters/dsl/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)；DSL 语言规则与能力谓词表见 [language-rules.md](language-rules.md)；芯片级硬件事实见 [common/simt-hardware-model.md](../../../common/simt-hardware-model.md)（公共层语言中立）。

## 1 定位

本目录承载 ScatterUpdate 范式中**强耦合 CANNBotDSL** 的知识：det 三段式段尾覆盖落地（嵌入排序/lookback/excl 发布）、nondet 直写路径、双门控派发、volatile 语义映射。开发语言为 DSL 时与通用知识层组合消费：

```
通用算法/模型（generic/）× DSL 实现细节（本目录）→ DSL 设计与代码产出
```

**参照实现**：tensor_scatter_nd_update_dsl（assign × nd 坐标元组，det/nondet 双路径），组织形态见 [example/](example/README.md)。

## 2 强制规则（生成前必读）

1. **先定型再设计**：按 patterns.md 场景判定（写语义划界 + index 形态 × 越界 × 确定性）；src 表示为型内形态决策
2. **确定性分治**：默认 nondet 直写——多写者任意赢即覆盖语义正确，全 dtype 无白名单（单条 store 天然防撕裂）；det 排序段尾覆盖为条件路径（张量源：高重复或确定性请求；标量源值同：仅高重复）
3. **确定性开关传导**：DSL 无框架级确定性查询——wrapper 检测 `torch.are_deterministic_algorithms_enabled()` 复用 torch 机制（与框架传导同源）；异常值语义须严格判定
4. **SIMT 主循环恒三参 grid-stride**；段扫描（数据依赖退出）用前置条件 while，禁 `while True + break`
5. **volatile 语义逐处对表 ASC**（[deterministic.md](deterministic.md) §4 / language-rules.md §5）：l2cache_load 仅轮询点、普通 gm[k] 为 barrier 稳定读、stcg 为跨核直写与复合 kernel 键写——**读写两侧分别对表**，勿泛化
6. **excl/全局计数原子发布必须在 tile 循环外**（每核每轮一次）——循环内发布在多波（每核多 tile）时重复累加 → 位置越界写（失效模式详见 [deterministic.md](deterministic.md) §2.3）
7. **跨 vf 域共享子式一律 kernel 级预计算**——同名子式被 CSE 跨域合并触发编译崩溃（上游缺陷：FormVfRegions 阶段，无诊断）；预计算参数同时构成 canonical 源（严格守卫 API 被除数）
8. **越界 = skip**：det 哨兵沉底、nondet 直跳；无 err flag，wrapper enqueue-and-return（双路径语义一致）
9. **空输入早退**先于排序 tiling 预计算
10. **KeyT 位宽自适应**（generic/conflict-domain.md §3）进结构键；哨兵 key（skip 形态）取键值域最大值
11. **固定除数统一整除**：固定除数统一使用 `//`（不设快除域分支）
12. **纯搬运字节视角**：全 dtype 按字节宽（W∈{1,2,4,8}）bit-exact 搬运，禁原生标量赋值承载非常规 dtype

## 3 阅读路径

| 任务 | 入口 | 内容 |
|------|------|------|
| 设计契约 | [design-spec.md](design-spec.md) | 输入/输出 / 按主题读取 / 一致性检查清单 |
| 能力谓词映射 | [capability-mapping.md](capability-mapping.md) | 通用契约 23 项 DSL 真值；DSL 专属谓词见 language-rules.md §0 |
| 程序结构 | [dynamic-shape.md](dynamic-shape.md) | 结构键 / det·nondet 双程序族 / 8 维 metadata 直读 |
| det 段尾覆盖 | [deterministic.md](deterministic.md) | 嵌入排序 / lookback / excl 发布 / (b,k,s) 分解 / 段尾覆盖 |
| nondet 直写 | [type-assign](../ascendc/type-assign.md) §4.1（语言中立骨架）+ language-rules.md | 直写覆盖 / 双门控 / 越界 skip 直跳 |
| 参照实现 | [example/](example/README.md) | assign × nd 基例 tensor_scatter_nd_update_dsl 文件组织与知识映射 |

## 4 范式内语言知识

- [language-rules.md](language-rules.md) — 能力谓词表 / import 纪律 / 动态 shape / grid-stride / canonical / volatile 映射 / DSL 前端行为模型（必读）
- [capability-mapping.md](capability-mapping.md) — 通用 23 项能力谓词的 DSL 真值映射（必读）
- [common/simt-hardware-model.md](../../../common/simt-hardware-model.md) — 芯片级硬件模型（存储层级/VF/DCache/volatile/超发核，语言中立）
- [../../generic/thread-layout.md](../../generic/thread-layout.md) — SIMT 线程排布模型（det 三阶段共用核 / (b,k,s) 一维化 / 超发核 guard，语言中立）
- [../../../common/uintdiv-rules.md](../../../common/uintdiv-rules.md) — 快除数学（AscendC 侧）
- [../../../common/sync-and-consistency.md](../../../common/sync-and-consistency.md) — 同步纪律
