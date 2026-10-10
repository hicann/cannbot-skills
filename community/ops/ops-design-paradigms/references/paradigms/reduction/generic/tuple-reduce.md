# Reduction 范式 Tuple Reduce 场景模型

> **层级**：通用知识层（generic/），语言中立。
>
> 来源：adapters/ascendc/tuple-reduce.md（原 reduction-tuple-reduce.md）。AscendC 落地（kernel 入口 for 循环代码、`SyncAll` 调用）在适配层。

## 1 场景定义

**Tuple Reduce**（本范式术语）= 对同一输入（或同一组输入）沿**相同归约轴**做 N 次独立归约的算子形态（典型：多输出归约算子）。

> `N_REDUCES` = 独立归约过程数。一次归约过程内部可产出多个分叉输出，分叉输出不增加 `N_REDUCES`——过程数与输出张量数解耦。`N_REDUCES` 为算子级常量（由算子公式固定），不进模板选择键、不进 TilingData。

## 2 核心约束

1. **共享归约轴**：所有归约过程的归约轴完全相同（共享切分结果的前提）
2. **跨过程同步按全局判定**：全局首轮 = 第 0 个过程（跳过等待），全局末轮 = 第 N−1 个过程（跳过置位），中间轮次正常置位/等待——见 [sync-model.md](sync-model.md) §3.2

## 3 总体策略（与标准范式的差异）

- **模板选择键不变**，不新增模板参数
- 差异仅在归约编排内部：归约过程按 `processIdx` 分发，搬出写到不同输出指针
- **合轴、切分、多核切分、二分缓存树算法等全部不变**

### 3.1 循环复用模型

N 次归约**复用同一份**片上 buffer 与 workspace：

- ⛔ **禁止在 workspace 公式或片上预算中乘以 N**——所有归约过程串行执行，生命周期不重叠，复用合法
- CopyIn 执行 N 次（每次循环各做一次），带宽开销为标准范式的 N 倍（性能特征，非正确性问题）

### 3.2 存活节点一条流规则

存活节点分析**分阶段**（归约前/归约/归约后段，Group 为 Phase 1/Phase 2）各自画**一条流**，禁止按 for 循环展开成多轮：

1. 公共步骤（所有过程都执行）只出现一次
2. 过程独有步骤写成按 `processIdx` 分发的并列分支，分割粒度 = 归约过程
3. 分支互斥——峰值取各分支 **max**，禁止相加、不按分支单独记账（详见 [liveness-fusion.md](liveness-fusion.md) §4）

### 3.3 归约前分发的同步语义

`processIdx` 是运行时变量——过程独有的 elementwise 分发用运行时 if/else（不可用编译期分支）。

### 3.4 Group 轮末栅栏

Group 分支每轮归约过程末尾需**跨核栅栏**：下一轮 Phase 1 写 workspace/片上前，本轮 Phase 2 必须读完——栅栏同时兜住 workspace 的跨过程 WAR 与片上侧跨过程 WAR（见 [sync-model.md](sync-model.md) §5）。Base/Empty 入口循环不插入额外跨核同步，跨过程 WAR 由核内流水同步按全局首末轮规则覆盖。

## 4 设计产出要求

1. **场景判定结论**：本算子是否 Tuple Reduce、`N_REDUCES` 取值（由算子公式决定）
2. **分发结构**：过程独有的归约前/后 elementwise 按 `processIdx` 的分支表；搬出到各输出指针的映射
3. **复用声明**：片上 buffer 与 workspace 与标准范式一致、N 次复用、禁止乘 N（显式写出）
4. **同步全局首末轮**：跨过程 WAR 的等待/置位跳过规则按全局 processIdx 判定（显式写出）
5. **Group 轮末栅栏**：若使用 Group 模板，每轮末尾栅栏及其兜住的 WAR 范围（显式写出）
