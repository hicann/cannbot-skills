# 卷积反向范式通用知识索引

本层按求导对象分为 input 与 filter。公共知识统一轴义、属性和规格描述；计算模型、数据流、Tiling 与验证从对应分支索引进入。范式准入与语言选择见 [范式入口](../patterns.md)。

## 1 公共知识

| 文档 | purpose | read_when | keywords |
|---|---|---|---|
| [language.md](language.md) | 张量、轴、属性与单位；按分支绑定 M/N/K 和执行字段 | 统一术语；读取当前分支记号 | Cin/Cout、groups、stride/dilation/pad、元素/字节、input/filter |
| [preprocessing.md](preprocessing.md) | 属性展开、低维升维、正向形状关系；按分支解释目标形状和初始化前置 | 将原始规格整理为统一三维问题 | input_size/filter_size、轴映射、dtype/format、形状校验 |

## 2 分支知识

| 分支索引 | purpose | read_when | keywords |
|---|---|---|---|
| [input/index.md](input/index.md) | 输入梯度与转置卷积的场景、计算流、切分及验证 | 求 dedx，或生成 Conv3DTranspose | dedy/weight/dedx、反向边界、rot180、bias |
| [filter/index.md](filter/index.md) | 权重梯度的场景、计算流、切分及验证 | 求 dedw，生成 Conv3DBackpropFilter / Conv3DDW | x/dedy/dedw、正向窗口、MN_StreamK、分段归约 |

## 3 阅读路径

```text
+------------------------------+------------------------------+
| language.md / preprocessing.md                              |
+------------------------------+------------------------------+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
+-------------+-------------+     +-------------+-------------+
| input/index.md            |     | filter/index.md           |
+-------------+-------------+     +-------------+-------------+
              |                                 |
              v                                 v
+---------------------------+     +---------------------------+
| language adapter / input  |     | language adapter / filter |
+---------------------------+     +---------------------------+
```

## 4 消费规则

1. 先确定求导对象，再加载公共知识和一个分支索引；分支记号、操作数和初始化协议配套使用。
2. 单个子问题最多展开 5 个叶子文档，按场景、切分、计算、同步和验证继续拆分。
3. 语言 API、平台能力与工程接入从 [语言适配索引](../adapters/ascendc/index.md) 进入。
