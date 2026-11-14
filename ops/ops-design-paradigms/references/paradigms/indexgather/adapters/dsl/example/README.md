# IndexGather DSL example

IndexGather 范式 DSL 侧的**代码组织形态参照**（单基例：A 型 @host shape 派生）。

## 定位声明

本 example 展示 DSL 读收集算子的**代码形态**——动态 shape 结构键、wrapper 归一/pack、grid-stride kernel、@host 编排。**设计知识（分型判定、越界形态决策、合轴算法）一律以范式 `generic/` + `adapters/dsl/` 文档为真值源**，两者有出入时以文档为准。

## 基例：custom_index_select（A 型 + 18 dtype 位级别名 + @host shape 派生）

| 文件 | 形态要点 | 知识真值源 |
|------|---------|-----------|
| `custom_index_select/custom_index_select.py`（单文件） | **@host shape 派生**（wrapper 结构化 reshape (outer, gds, inner)/(outer, m, inner)；@host 从 Tensor shape 派生 inner/m/gds/total——cannir.extract_shape 运行期值 → 运行期 blocks → 显式维度 view 展平；**帧契约 3 槽零标量**）/ **use64 域**（派生值随 use64 走 i64 / 窄化 i32 → use64=False 全链 int32 真快速路径；动态 shape 下 view(-1) 禁用，须显式维度）/ signed 归一（字节宽→int8/16/32/64 位保真，绕开 bisheng 无符号常量缺陷）/ **视图 pack**（inner 维 W 递降 8//elem_bytes，整除 ∧ 首地址对齐双判据）/ 越界零写 dt(0) 编译期常量 / THREAD_NUM=2048 | dsl/dynamic-shape.md §1/§3、dsl/kernel-impl.md §2 |

单文件组织：@kernel 类 + @host 编排 + 动态 launcher（结构键缓存 dict）+ wrapper（校验/归一/pack/分配/发射）。18 dtype = 11 直通（DSL 原生 spec）+ 7 位级别名（torch 层 view 进出：bool→u8、f64/complex64→i64、complex32→i32、fp8×3→u8）。

## 骨架

```
index_select（@host shape 派生型——零 params 通道、零标量帧槽）:
  wrapper: 校验 → signed 归一 → 视图 pack → 结构化 reshape → 输出分配
           → program(x_r, index, out_r)
  @host:   shape 派生（extract_shape 运行期值）→ use64 窄化 → blocks 计算
           → 显式维度 view 展平 → kernel[blocks](...)
  kernel:  标量参数直用 → grid-stride range → 三因子坐标分解
           → j 原值域检查 → 零写/搬移
launcher: 结构键懒编译（同键 shape 零重编译）；SIMT 主循环恒三参 range（B-7）
```
