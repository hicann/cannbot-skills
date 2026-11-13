# 卷积反向 input 分支知识索引

> **层级**：input 分支通用知识（generic/input/），语言中立。与对应语言的 input 适配层组合消费：先经 [范式入口](../../patterns.md) 的语言路由确定适配层，再按下表按需展开。

## 1 检索表

| 文档 | purpose | read_when | keywords |
|------|---------|-----------|----------|
| [language.md](../language.md) | 统一术语与记号：张量/维度/属性词汇、M/N/K 三元组、坐标映射、切分记号体系（baseUse*/mCompute/stepK）、模板与状态字段、bias 词汇、核心记号速查表 | 建立全局心智模型；查阅任一文档前对齐记号 | M/N/K、m0/n0/C0、dedy_expand、fmap、weight_rot、baseM/baseN/baseK、singleCore*、stepKa/stepKb、loadB2Condition、五路 pbuffer |
| [scene.md](scene.md) | 场景与计算模型：场景准入、前放大（dedy 插零）、rot180 与通道交换、反向边界与尾部余数、乘加归约、1×1×1 退化、转置卷积场景（对象绑定/output_padding/bias）、能力边界、数据流 backbone | 建立数学模型；确认规格属于本范式；理解"反向卷积怎么算" | 前放大、插零、rot180、反向补边、尾部余数、目标形状、转置卷积、output_padding、bias |
| [preprocessing.md](../preprocessing.md) | 规格归一：两种进入场景、属性统一、1D/2D 升维、形状校验与派生量（有效核尺寸/ρ/dedy_expand/反向补边）、输出形状模式、执行前置约定与输出预清零 | 拿到原始规格后整理为统一 3D 问题 | 升维、input_size、形状校验、派生量、输出预清零、initOutputFlag |
| [compute-flow.md](compute-flow.md) | 计算流程：数据流总览（阶段与指令职责/每侧通路图/两路总图）、格式与轴序（NCDHW/NDHWC/DHWCN、ND/DN/NZ/ZN、转换全景）、指令与单位总表（地址公式/窗口网格/补齐职责）、img2col 与窗口归约、数据搬运与写回、多核任务、单核迭代、DAG 与四级流水 | kernel 实现设计；理解数据怎么流、任务怎么分、循环怎么写、同步怎么插 | img2col、M/N/K 遍历序、L1A/L1B 父段、四级流水、任务解码、L0C 交接 |
| [tiling/tiling.md](tiling/tiling.md) | 切分算法（tiling/ 文件夹入口）：模板选择（优先序/决策流程）、L0 基本块与 K 窗口配对、L1 驻留容量、CalStepK 候选、LegalProtection 容量回调、SetSingleCoreInfo 任务恢复、Host 传参字段表；三个求解模板（standard 经验选块与任务调整 / small-shape 小任务量 M/N 均衡搜索 / l1-resident 完整 K 权重驻留，各自的选择条件、起点、Host 求解顺序与数值示例）由其索引展开 | tiling 设计与求解实现；模板选型；块大小与驻留推导 | standard/small-shape/l1-resident、IsCapable、InitBaseMNK、AdjustBaseMNK、CalStepK、stepKa/stepKb、容量回调、窗口配对、enableFullLoad |
| [verification.md](verification.md) | 验证方法：验证闭环总览、三层用例设计（属性组合/tiling 状态/规模边界）、判定依赖分级与增量接入、输入构造、golden 生成（torch conv_transpose3d 基准与属性归一）、精度比对与通过判定、参考代码指引、失败定位（差异分布→根因映射）、范式回写 | 用例设计；golden 脚本编写；精度比对；精度失败定位；范式回写；增量开发用例接入 | torch 基准、非对称 pad 切片、混合容差、rtol/atol/ptol、输出预清零检查、根因映射、回写、硬依赖、软依赖、增量接入 |

## 2 阅读路径建议

```text
+------------------------+------------------------+
| Entry: scene / language / preprocessing         |
+------------------------+------------------------+
                         |
                         v
+------------------------+------------------------+
| Design: tiling -> templates -> compute-flow     |
+------------------------+------------------------+
                         |
                         v
+------------------------+------------------------+
| Implementation: input adapter / API / sync      |
+------------------------+------------------------+
                         |
                         v
+------------------------+------------------------+
| Verification: golden / compare / feedback       |
+------------------------+------------------------+
```

## 3 使用规则

1. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分为更小的子问题。
2. 通用层不含具体语言 API 的用法细节；文中出现的指令名与单位以 [compute-flow.md](compute-flow.md) 的阶段表为准，API 落地在语言适配层。
3. 转置卷积场景先读本层 scene 与 preprocessing 的场景绑定与形状模式，再进入公共计算层；bias 的资源影响计入公共容量检查。
4. 具体语言的实现（代码模板、参考源码、生成规范）在适配层（`../../adapters/<lang>/input/`），不在本文档族内。
