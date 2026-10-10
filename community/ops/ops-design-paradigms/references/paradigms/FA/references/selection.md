# Attention 变种选型

> 选型分两步：先定**基础形态**（GQA / MHA / MLA，沿 head / hidden-dim 变化的三选一），再按需**正交叠加** trait（量化 / 稀疏，各自独立可选）。

## 基础形态（三选一，必选）

| 形态 | 数学差异 | 选型触发条件 |
|---|---|---|
| **GQA**（默认） | `Hq = G × Hkv` | 默认路径；`Hq % Hkv == 0` |
| **MHA**（GQA 退化） | `G = 1`（`Hq = Hkv`） | GQA 的退化特例 |
| **MLA**（latent 变体） | KV 经 latent 压缩，`kvHeadNum = 1` | 数学公式含 latent absorption |

## 正交叠加 trait（各自独立可选，叠在基础形态之上）

| trait | 触发条件 |
|---|---|
| **量化** | Q / KV dtype ∉ {fp16 / bf16} |
| **稀疏** | KV 稀疏 pattern 由外部索引 / 掩码数据给定（规则掩码如 causal / 滑窗 / Sink 属 Feature，不触发） |

> 量化 trait 改变核心计算路径，稀疏 trait 仅影响块跳过（各自契约见 trait 文件）。

## Feature（可叠加在任意组合之上）

`causal` / `RoPE` / `PSE` / `Sink` / `滑动窗口` / 其他位置或掩码特性。

## 选型规则

1. 数学公式与标准 attention 一致 + `Hq % Hkv == 0` → **GQA 形态**（`G=1` 即 MHA 退化）。
2. 数学公式含 latent absorption → **MLA 形态**（kvHeadNum=1 + latent 维）。
3. **量化 / 稀疏是正交 trait，不是子族**：定完基础形态后按需叠加。

> **说明**：FlashDecoding（split-KV reduce）是**并行实现技术**，不是子族 / trait，可叠加在任意组合之上。
