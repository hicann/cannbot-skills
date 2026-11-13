# 卷积反向范式 AscendC 适配索引

本目录将公共规格和分支计算模型映射到 Ascend C。先经 [范式入口](../../patterns.md) 确定语言与求导对象，再读取公共能力和对应的分支索引。

## 1 公共知识

| 文档 | purpose | read_when | keywords |
|---|---|---|---|
| [buffer-and-types.md](buffer-and-types.md) §1.1–§1.2 | 获取 platformInfo、核数、L1/L0 容量及目标 SDK 信息 | 建立资源与能力依据 | platformInfo、AIC/AIV、元素/字节、SDK |
| [通用知识索引](../../generic/index.md) | 统一规格、轴义、属性与分支选择 | 进入实现前对齐问题 | language、preprocessing、input/filter |

具体操作数类型、槽尺寸、布局转换和同步协议按分支消费。buffer-and-types 的后续具体示例沿用 input 协议，filter 使用其分支中的类型与 Bound 定义。

## 2 分支知识

| 分支索引 | purpose | read_when | keywords |
|---|---|---|---|
| [input/index.md](input/index.md) | 输入梯度与转置卷积的指令、Kernel、同步、字段及示例 | 实现 dedx 或转置卷积 | weight 变换、Load3D、BT、input_size |
| [filter/index.md](filter/index.md) | 权重梯度的指令、Kernel、同步、字段及示例 | 实现 dedw | x 窗口、MIX 清零、FP32 原子写回、filter_size |

## 3 阅读路径

```text
+------------------------------+------------------------------+
| platformInfo / resources / SDK / tensor roles               |
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
| input examples            |     | filter examples           |
+---------------------------+     +---------------------------+
```

## 4 消费规则

1. 一次任务只加载命中语言及求导分支的实现资料。每个子问题最多展开 5 个叶子文档。
2. 核类型、实际 blockDim、初始化参与单元、调度模式和事件协议形成同一启动契约。
3. 按分支索引选取直调或 ACLNN 示例；重新核对当前规格、平台、SDK 与最终字段。
4. 工程骨架分别参考仓内 ascendc-direct-invoke-template、ascendc-registry-invoke-template；本页只索引算法适配知识。
