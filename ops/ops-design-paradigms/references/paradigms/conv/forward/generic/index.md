# Convolution 范式通用知识层索引

> **层级**：通用知识层（generic/），语言中立。与语言适配层（`../adapters/<lang>/`）组合消费：先经 [../patterns.md](../patterns.md) 的语言路由确定适配层，再按下表按需展开。

## 检索表

| 文档 | purpose | read_when | keywords |
|------|---------|-----------|----------|
| [language.md](language.md) | 统一术语与记号：张量/维度/属性词汇表、M/N/K 三元组、切分记号体系（mAL1/nBL1/Tail/fullLoad/*Dim）、数学记号、核心记号速查表 | 建立全局心智模型；查阅任一文档前对齐记号 | M/N/K、M0/N0/K0、C0/C1、pad 约定、mAL1、nBL1、Tail、fullLoad、db/pingpong |
| [compute-flow.md](compute-flow.md) | 计算流程：卷积公式与滑窗原理、四侧数据流（fmap/weight/bias/output）、五种数据格式（ND/DN/NZ/FRACTAL_Z/FRACTAL_Z_3D）、全部指令（语义/限制/单位/取值规则）、多核分核、单核迭代伪代码、四级流水同步 | kernel 实现设计；理解数据怎么流、指令怎么用、循环怎么写、同步怎么插 | im2col、行推理、LOAD3D、尾核、迭代伪代码、四级流水、防死锁 |
| [tiling.md](tiling.md) | 切分算法：M/HW/basic block 三模式（含模式选型决策）、基本块核间/核内切分原理、核间绑核三步法（绑满→不超核数→代价模型）、核内 L0 先 L1 后决策、fullLoad 四态、pingpong 开关、K 轴优先全载、指令带宽差异、TilingKey 位打包 | tiling 设计；模式选型；绑核策略推导；块大小决策 | M 模式、HW 模式、basic block、模式选型、绑核、代价模型、fullLoad、pBufferFlag、TilingKey |
| [verification.md](verification.md) | 验证方法：三层用例设计（属性组合/tiling 状态/规模边界）、输入构造、golden 生成（属性归一/pad 拆分/溢出截断）、精度比对与通过判定计算（混合容差+匹配率+ptol）、参考代码指引（ttk/ops-nn/opstest）、差异分布→根因映射 | 用例设计；golden 脚本编写；精度比对脚本编写；精度失败定位；| 用例表 CSV、golden、混合容差、rtol/atol/ptol、匹配率、isclose、dump 定位、参考代码 |
| [scene.md](scene.md) | 场景扩展：1D/3D 维度扩展、量化卷积（offset_x/反量化/定点移位）、NHWC、groups 对角阵化、pad≥kernel、C04 小通道 | 规格超出标准路径（fp16、NCW/NCHW/NCDHW、groups=1、pad<kernel）时 | 3D D 轴、量化、offset_x、NHWC、OPT_GROUP、对角阵、allPad、C04 |

## 阅读路径建议

```
                      ┌─ 入门 ─────────────────────────────────────────┐
                      │  language.md（记号速查）                        │
                      │    ↓                                             │
                      │  compute-flow.md §1-§3（公式/数据流/格式）      │
                      └────────────────────────────────────────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                              ▼
              ┌─ 设计 ──────────────────┐   ┌─ 验证 ─────────────────┐
              │ tiling.md（先切分）       │   │ verification.md          │
              │   ↓                      │   │ （用例/golden/比对/     │
              │ compute-flow.md §4-§7    │   │  定位）             │
              │ （后实现）                │   └─────────────────────────┘
              └─────────────────────────┘
                     │
                     ▼
              ┌─ 扩展 ─────────────────────────────────────────┐
              │ 规格超出标准路径时 → scene.md 检索表进入对应章节 │
              └────────────────────────────────────────────────┘
```

## 使用规则

1. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分为更小的子问题
2. 通用层不含具体语言 API 的用法细节；文中出现的指令名以 [compute-flow.md](compute-flow.md) §4.1 的"指令名 ↔ AscendC API 对照表"为准，其他语言按各自适配层映射
3. 标准路径（fp16、NCW/NCHW/NCDHW、groups=1、pad<kernel）只读主干四件套；scene.md 仅在规格超出时读取
4. 具体语言的实现（代码模板、参考源码、生成规范）在适配层（`../adapters/<lang>/`），不在本文档族内
