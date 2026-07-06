# Task A 产出验证

校验 `S2P1_path_list.json` 中对源码的声明是否真实。辅助文件 `S2P0_source_scope.md` 仅用于路径映射。仅校验 `reachability == "reachable"` 的 path。

---

## A1.1 source 行号

解析每条 reachable path 的 `source`（格式 `{tiling_short}:{L_t} → {kernel_short}:{L_k}`，L_k 可为 range），Read 对应行号 ±N 行源码验证位置真实性。

**tiling 侧**（L_t ±3）：该位置须包含分支逻辑（`if`/`else if`/`switch`/`case`/`IsMatchXxx()`/`set_tilingKey`），空行/注释/无关代码 → **fail**。

**kernel 侧**（L_k ±5）：该位置须在某个 `TILING_KEY_IS(...)` 块内（花括号配对确定块边界）。range 格式下 range 内至少一行在块内即 pass。所有块之外 → **fail**。

---

## A1.2 conditions

对每条 reachable path，Read tiling 源码 `L_t ±15` 行（扩展至当前函数边界），逐条验证 `conditions[]`。
当 conditions 中出现命名常量（如 `FLT_EPSILON`、`MIN_C_SIZE` 等）但在 ±15 行窗口内找不到定义时，须 Grep 该常量名在整个 tiling 源码文件中的 `static const` / `constexpr` / `#define` 定义，解析其实际值后再比对。

| 类型 | 规则 |
|------|------|
| `{"var",op,value/ref}` 纯运算符 | var（+ref）变量名在源码段出现；op 方向一致；value 数值匹配 |
| `{"var",op,...}` 复合 op | op 为描述性文本（如 `* dtypeSizeX_ >= 128`），降级匹配：op 中所有变量名 + 常量在源码段出现即可 |
| `{"expr",op,value}` | expr 中所有变量名在源码段出现；op 方向一致；value 匹配 |
| 运算符取反 | 源码中路径条件常以否定形式表达（如 `if (cond) return;` 表示路径条件为 `!cond`），验证时需考虑运算符取反（`>=` ↔ `<`、`==` ↔ `!=`），取反后一致即 pass |
| `{"boundary_check":"..."}` | 自由文本，同时查 tiling 源码段 **和** kernel apt.cpp 对应 `TILING_KEY_IS` 块内的分支判定 |

普通 condition 不匹配 → **fail**。boundary_check 不可解析或源码段与 kernel 块均不存在对应判定 → **warn**（不阻塞）。

---

## A1.3 key_instructions

定位 apt.cpp 对应 `TILING_KEY_IS` 块，在块内查找模板实例化语句。比对签名 `ClassName<T1,...,Tn>`：

- 模板参数数量必须一致
- 固定参数位（`bool`/`int`/`uint32_t` 等）须完全匹配
- `*` 通配位接受任何非空值
- 模板参数名接受占位符标识（如 `{dtype_placeholder}`）或具体数据类型枚举值（如 `ge::DT_FLOAT16`）

**块内存在性扫描**：至少一个实例化完全匹配即 pass。类名不匹配 / 参数数量不一致 / 无实例化匹配 → **fail**。

---

## A1.4 source_constraints

验证顶层 `source_constraints[]` 中每条约束：

| 字段 | 规则 | 判定 |
|------|------|:----:|
| `source_location` | 文件路径映射成功（见下方映射）；行号 ±3 处确有代码 | **fail** |
| `source_expr` | 字面匹配源码中该位置表达式 | **fail** |
| `variables` | 每项出现在 `source_expr` 文本中，或是 source_expr 中子表达式的语义等价写法 | **warn**（不自洽） |

**fail 报告要求**：当 source_expr 判定为 fail 时，报告中必须引用源码该位置的实际行内容（原文复制，非转述），格式为：
- S2 声称：`{source_expr}`
- 源码实际（L{line}）：`{actual_source_line}`
- 差异：{具体差异描述}

---

## A1 整体判定

任一子项 fail → **fail**；仅 boundary_check / variables warn → **pass_with_warnings**；全 pass 无 warn → **pass**。

---

## 文件名映射

| 使用场景 | tiling 文件 | kernel 文件 |
|---------|------------|------------|
| `path.source`（短名如 `tiling.cpp`/`kernel.cpp`） | → `manifest.tiling.file_list` 首条（P0 优先） | → `manifest.kernel.file_list[0]` |
| `source_constraints.source_location`（完整 basename） | 子串匹配 `manifest.tiling.file_list[].path` | 子串匹配 `manifest.kernel.file_list[].path` |

映射失败 → **fail**。

## 字面匹配容忍度

接受：空格/换行/注释差异、`this->x` vs `x`、单双引号。不接受：constexpr 名替换（`128`→`MIN_C_SIZE`）、运算符反转（`>`↔`>=`）、表达式重组。

## 输出格式

```
| ID | 状态 | verified/total | 备注 |
|----|------|---------------|------|
| A1.1 | pass/fail | v/t | tiling: vt/tt; kernel: vk/tk |
| A1.2 | pass/fail/warn | v/t | boundary_check warn: N |
| A1.3 | pass/fail | v/t | |
| A1.4 | pass/fail/warn | v/t | variables warn: N |
```
