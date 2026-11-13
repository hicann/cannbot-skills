# TilingKey 与多模板组织

> **层级**：AscendC 语言适配层（adapters/ascendc/）。维度集的选择依据（哪些决策进 key、哪些不进）是语言中立的，见 [../../generic/tiling.md](../../generic/tiling.md) §4；本文件给 AscendC 的机制与组织方式。
>
> **状态标注**：参考实现为单模板（未拆 key），本文件为机制说明，落地时按所用头文件的宏定义核实。

## 1 机制：为什么需要 TilingKey

**解决的问题**：块大小是连续值，kernel 模板无法为每个值实例化；把影响代码路径的**离散决策**打包成整型 key，相同 key 共享一个编译实例。

```
host 侧                                    kernel 侧
┌──────────────┐    uint64 key    ┌─────────────────────────────┐
│ tiling 决策   │ ──────────────► │ 按 key 值分发到对应模板实例    │
│ → 模板参数元组 │   （随 tiling    │ → 每个 key = 一份编译好的      │
│ → 按位打包    │    数据下发）     │   kernel binary             │
└──────────────┘                  └─────────────────────────────┘
```

## 2 host 侧：key 的生成（三步）

| 步骤 | 操作 | 规则 |
|------|------|------|
| ① 取枚举值 | 逐维按 tiling 决策结果取值 | 维度集见 [tiling.md](../../generic/tiling.md) §4（含裁剪建议） |
| ② 互斥修正 | 不合法组合归一到合法值 | 如"仅 N 全载"仅在 M 模式成立；不修正则 kernel 侧无实例可分发 |
| ③ 按位打包 | 每维分配位宽（2 值 1bit、4 值 2bit…），拼接为 uint64 | **位分配一经发布不可变更**（key 值即兼容性契约） |

## 3 kernel 侧：多模板组织（三件套）

```cpp
// ① 声明模板参数组（与 host 维度集一一对应，顺序即位序）
ASCENDC_TPL_ARGS_DECL(Conv, FmapTiling, WeightTiling, L1PingPong, L0PingPong,
                      OutputOrder, IterOrder);

// ② 按合法组合枚举实例（每个组合一个显式实例化分支）
ASCENDC_TPL_SEL(Conv, /*key0*/) { ConvKernel<T0, T1, ...><<<...>>>(...); }
ASCENDC_TPL_SEL(Conv, /*key1*/) { ConvKernel<...><<<...>>>(...); }

// ③ host 侧 launch 前按同样的位分配打包
uint64_t key = GET_TPL_TILING_KEY(...);
context->SetTilingKey(key);
```

## 4 组织建议

| # | 原则 | 说明 |
|---|------|------|
| 1 | **维度集最小化** | 只把"改变代码路径"的决策进 key（循环结构、乒乓、通路形态）；块大小数值走 TilingData 运行时参数——否则实例数爆炸 |
| 2 | 实例数预算 | 实例数 = 各维取值数之积（合法组合裁剪后）；超过两位数时应先裁维度集 |
| 3 | 新维度追加 | 新维度放在 key 的**最高位之前**预留空位；已有 key 值保持不变 |
| 4 | 与拉起接入的边界 | registry/aclnn 编译特有的适配（如 Cube kernel 核类型声明 `ASCENDC_TPL_KERNEL_TYPE_DECL(KT, ASCENDC_TPL_AIC_ONLY)`）属拉起接入而非 TilingKey 机制本身，由上层 aclnn 工作流技能保证（本仓 `ops/ascendc-registry-invoke-template`；漏声明的故障特征见 [../../generic/verification.md](../../generic/verification.md) §7.2） |
