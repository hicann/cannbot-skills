---
name: ascendc-registry-invoke-to-direct-invoke
description: 当用户想把自定义算子工程中的 kernel 模板改造成 `<<<>>>` kernel 直调形式，或从自定义算子工程中抽取某个 kernel 模板并转换成 `<<<>>>` 直调方式时使用。触发：用户提到"自定义算子转直调"、"从算子工程抽 kernel"、"kernel 模板改 `<<<>>>`"等。不适用于从零开发新算子
---

# AscendC 自定义算子转 `<<<>>>` kernel 直调

## 一句话原则

**这是算子代码迁移，不是代码重写。**

`<<<>>>` 直调模式下，原算子工程的"注册框架"不存在了。必须改的只有这套框架胶水——其余一切零修改。

如果你发现自己在修改 kernel 类的成员函数、"优化"tiling 公式、合并分支写法、"顺手"重命名字段——停下。你越界了。

---

## 核心三原则

### 原则 1：kernel 实现代码零修改

**"kernel 实现代码"范围**：
- kernel 类定义（类体、成员变量、所有成员函数）
- kernel 类调用的算法函数（Compute / DataCopy / Reduce / Cast 等）
- 这些代码内部的控制流、循环、分支、计算顺序

**唯一允许的修改只有 3 类**：

1. **include 指令替换**：把跨目录相对 `#include "../..."` 换成本地 include（到 `local_deps.h` 或 SDK 稳定头）
2. **命名空间包装（可选）**：把原文件内容整体套进统一命名空间，避免 `using namespace AscendC` 污染包含方翻译单元
3. **TilingData 类型来源替换**：原来通过 `BEGIN_TILING_DATA_DEF` 宏生成的 TilingData 类，改为来自本地 POD struct（字段名、字段类型、字段顺序与原宏 `TILING_DATA_FIELD_DEF` 一致，因此 kernel 中 `tl_->field` 的访问一行都不动）

**不允许的修改**：
- 合并或拆开成员函数（不要把 `CopyIn / CopyOut / Load / Compute` 这类方法合成一个）
- 重命名字段、参数、局部变量
- "顺手"优化循环结构、计算顺序、分支写法
- 把类成员改成自由函数、或反过来
- 删除看起来"没用"的成员变量或方法

**判断标准**：从原算子工程同步新版本 kernel 时，应当能直接覆盖 kernel 实现文件，而不需要同步修改别处。如果做不到，说明你把框架胶水和算法代码混在同一个文件里了。

### 原则 2：tiling 计算逻辑零修改

**"tiling 计算逻辑"范围**：
- 如何根据输入 shape 和平台参数计算 blockDim、baseRowLen、baseColLen、splitFactor 等切块参数
- 所有数学公式、对齐运算、上/下取整
- 分支判断（如 `is32BAligned == 1` vs `== 0`、fullload vs splitD 等）

**唯一允许的修改只有 3 类**（均为"接口替换"，不是"逻辑改动"）：

1. **平台接口替换**：`gert::TilingContext*` 获取平台参数 → `platform_ascendc::PlatformAscendCManager::GetInstance()`（功能等价）
2. **框架胶水删除**：去掉 `IMPL_OP_OPTILING` / `REGISTER_TILING_DATA_CLASS` 注册宏；`OP_LOGE / OP_LOGI / OP_LOGD` 换成 `throw std::runtime_error` 或 `printf`
3. **输入参数形态**：`gert::Shape` → 直接传 `uint64_t rowLen, colLen`；`ge::DataType` → 本地自定义 enum

**不允许的修改**：
- 合并对齐 / 非对齐分支（即使公式"看起来差不多"）
- 把某分支特有的 `AlignUp(x, y)` 套到其他分支上
- "统一"上/下取整、"统一"公式写法以追求代码简洁
- 删除看似多余的边界判断、clamp、溢出检查
- 调整乘除法的计算顺序

**为什么这条特别重要**：改 tiling 数学极少立即编译错，但会悄悄改掉切块语义，让 correctness 或性能偏离上游实现。tiling 公式逐行搬运，分支独立保留，分母/分子按原样。

### 原则 3：只改"注册框架胶水"

真正必须改的只有 3 处——因为 `<<<>>>` 直调模式下这 3 处对应的宏 / 接口 / 分发机制根本不存在：

1. **Kernel 入口**：原 `extern "C" __global__` + `GET_TILING_DATA` + `TILING_KEY_IS(N)` 运行时分发 → 按 tilingKey 划分维度拆成独立 `__global__` 函数、tiling struct by-value 入参
2. **TilingData 定义**：`BEGIN_TILING_DATA_DEF` / `TILING_DATA_FIELD_DEF` / `END_TILING_DATA_DEF` 宏 → 普通 C++ POD struct
3. **Host tiling 接口**：`gert::TilingContext*` → `PlatformAscendCManager::GetInstance()`；去掉注册宏和 OP_LOG

除此以外，默认一切不动。

---

## 全量迁移优先

**默认把原算子源码文件整体搬到目标目录，不做选择性裁剪。** 只有用户明确要求时才裁剪。

### 全量搬运的对象（当前算子自己的源文件）

- **kernel 实现头**：整文件搬，不挑成员函数、不删"没用的"方法
- **TilingData struct**：保留 `BEGIN_TILING_DATA_DEF` 里定义的**所有字段**，不因"kernel 没访问"就删
- **Host tiling 逻辑**：保留**所有 variant 路径**（grad / quant / fullload / splitD 等子分支），不选取单一路径
- **入口文件**：整文件搬，不裁剪 tilingKey 分支

### 什么时候允许裁剪

只有用户明确提出，比如：
- "只要 fullload 版本，grad 不要"
- "tiling 里 quant 相关字段都可以删"
- "fp32 路径不需要"

裁剪时只删用户明确指定的内容，其它保持全量。

### 为什么默认全量

1. **保真性**：不用判断"这个字段/路径是否真用不到"。字段可能通过宏字符串化、SFINAE、模板特化间接使用，grep 找不到，一旦误删编译看不出，运行时才炸。
2. **可同步性**：原算子更新新版本，可以直接覆盖目标文件而不需要重新做差分。这和原则 1 的"判断标准"完全一致。
3. **和原则 1/2 形成闭环**：不仅不改代码，也不选择性删代码——二者一起才能做到"整文件可覆盖同步"。

### 例外：外部依赖仍按最小闭包

"全量迁移"只适用于**当前算子自己的源文件**。跨目录的外部依赖（上游 `*_base.h` / `*_common.h` / `norm_common/*` 等）仍然按**最小闭包**搬——这些是别人的代码，整包拖会把无关实现带进来、污染目标目录、扩大维护面。

一句话区分：
- **当前算子自己的 `.h` / `.cpp`** → **整文件搬**
- **跨目录 include 过来的头** → **只搬 kernel 真正使用的符号**

---

## 信息来源约束

本 skill 区分两类"源"——算法源和约定源，读取规则不同。

### 算法源：只来自原始算子源码

**原始算子源码**（用户提供的 kernel 源 + tiling 源）是所有算法逻辑和数学公式的**唯一**来源。

以下内容不能从目标仓库的其他 sample / story 复制或参考：
- Kernel 计算逻辑、成员函数拆分
- Tiling 数学公式、切块分支
- 数据搬运 helper、规约实现
- 算子特有的 traits、cast 辅助

原因：目标仓里的其他 sample 实现的是别的算子，抄它们会污染当前算子的保真性。

### 约定源：应该参考目标仓库对齐

目标仓库里"怎么组织一个算子工程"的**工程约定**应当参考并保持一致——新算子不要发明和邻居不一致的新规矩。

**应参考并对齐的工程约定**：
- **目录布局**：如 `include/` vs `src/` 分层、kernel/tiling/host 放哪里、`cmake/` 子目录位置
- **文件命名**：后缀用 `.h` / `.hpp`；前缀/后缀规则（`_kernel` / `_entry` / `_tilingdata` / `_tiling` 等命名惯例）；kernel 和 tiling 文件名怎么关联
- **CMakeLists.txt 写法**：target_include_directories 组织、编译选项管理、依赖链接模式
- **include 路径风格**：项目内部 include 用相对路径还是 `target_include_directories` 下的逻辑路径
- **命名空间选择**：统一顶层命名空间（如 `project_name::op_name`）
- **README / 测试目录的组织方式**

**判断方法**：`Glob` 和 `Read` 目标仓库里 2–3 个同类 story / sample 的目录树、文件命名、CMakeLists.txt 骨架（只看组织结构，不看算法实现），归纳共同约定后采用。如果同类 story 之间约定就不一致，选影响面最大或最新维护的那个作为参考。

### Host 驱动的算法无关样板

Host 驱动里 ACL init / 内存管理 / 类型转换 / 结果比对 / `main` 骨架这些**算法无关**的部分，按优先级选择来源：

1. 目标仓库有可参考的同类 story / sample → **对齐邻居写法**（细则见下文"Host 驱动来源决策"）
2. 目标仓里没有参考对象 → 读 `references/host-driver-template.md` 作为起点
3. 不涉及 host 驱动 → 跳过

CMakeLists.txt 属于"工程约定"，一律按目标仓既有 cmake 体系对齐。

---

## 默认工作流

### Step 1: 确认交付边界，并对齐目标仓约定

先和用户确认 3 件事：
- **源目录**：kernel 源 + tiling 源（tiling 源可能在 `op_host` 或类似位置，不同工程布局不同）
- **目标目录**：放在当前路径根 / 子目录 / 保留源层级中的哪种——由用户决定，不预设
- **交付边界**：只做 kernel 依赖解耦，还是要做到 kernel + tiling + host driver 可独立编译运行

**同时调研目标仓约定**：在目标仓库里 `Glob` + `Read` 2–3 个同类 story / sample（只看组织结构，不看算法），归纳：
- 目录布局（include/src/cmake 的分层方式）
- 文件命名规则（`.h` / `.hpp` 后缀、`_kernel` / `_entry` / `_tilingdata` 等后缀约定）
- CMakeLists.txt 骨架
- include 路径风格
- 顶层命名空间

本算子的目录结构和文件命名**跟这些邻居约定对齐**，不要发明新规矩。如果同类 story 之间就不一致，选影响面最大或最新维护的那个参考。

不同算子工程的目录布局差异很大，不要硬套模板结构；用户给什么目录就基于什么目录推进，结合目标仓邻居约定决定文件命名和放置。

### Step 2: 源文件盘点

用 `Glob` / `Read` / `Grep` 弄清：
- kernel 源目录下有哪些 `.h` / `.cpp`
- 哪个是入口文件（命名可能是 `*_apt.cpp` / `*_invocation.cpp` / 其他）
- kernel 实现是 header-only 还是有独立 `.cpp`
- 哪些头文件已经是"从上游模板改造过的半本地化版本"（优先复用，不重建）
- tiling 源文件位置（typically `op_host/xxx_tiling.cpp` 或类似）

### Step 3: 追踪相对 include，建立符号级依赖图

找出 kernel 文件中所有：
- `#include "../..."`
- `#include "../../..."`

对每个相对 include：
1. 记录当前 kernel 真正使用了其中哪些符号（函数、常量、traits、类型、宏）
2. 记录这些符号的**真实**定义位置

不要因为某个头被 include 了就整包复制。特别警惕"表面归属 ≠ 真实定义"的转手依赖——`using A::foo`，但 `foo` 真实定义在命名空间 `B` 中，`A` 只是通过 include 间接暴露。迁移时要顺手修正 `using` 归属。

### Step 4: 依赖解耦——整文件搬当前算子源码，外部依赖做最小闭包

本步骤执行"全量迁移优先"原则：

- **当前算子源码文件**（kernel 实现头 / tiling 源 / 入口文件）→ **整文件搬**，结构、内容、分层一律不动（原则 1）。不挑成员函数、不删"没用"的方法、不裁剪 tilingKey 分支。
- **跨目录的外部依赖** → **最小闭包搬到 `<op>_local_deps.h`**。

`<op>_local_deps.h` 的职责：
- SDK include（`kernel_operator.h` 必须；`simt_api/asc_bf16.h` 等按需）
- 从上游搬来的最小外部依赖闭包（仅限 kernel 实际使用的符号）

即使 kernel 文件没有 `..` 相对 include，也创建 `local_deps.h` 来统一收口 SDK include——好处是 kernel 实现头只 include 一个入口，后续增减 SDK 依赖只改一处。

**外部依赖最小闭包分类**（仅针对跨目录搬来的符号，不含当前算子自己的源码）：

| 类别 | 典型内容 | 注意点 |
|------|---------|--------|
| 平台常量 | `GetUbBlockSize` / `GetVRegSize` 等 | 若只是返回固定常量，可改成 `constexpr` 小包装 |
| 基础 traits | `is_same` / `bfloat16_t` 兼容定义 / 对齐工具 | 只搬需要的一小段，不要整份 `*_base.h` 拖过来 |
| common 层工具 | `CeilDiv` / `CeilAlign` / cast 辅助 / 多级规约 | 最容易命名空间错位，核实真实定义源 |
| 跨算子 helper | 从别的算子目录 include 过来的 `DataCopyImpl` / `ComputeXxx` 等 | 只搬 kernel 真正调用的函数，并补齐它们的直接闭包 |

核心要求：**外部依赖只搬 kernel 真正调用到的**，不要整包拖上游 `*_common.h`；**当前算子源码整文件搬**，不要擅自裁剪自己的文件。

### Step 5: 入口改造（原则 3 第 1 项）

原工程入口典型形态：
```cpp
extern "C" __global__ __aicore__ void op_name(GM_ADDR x, ..., GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    if (TILING_KEY_IS(1)) { KernelA<fp16>(...); }
    else if (TILING_KEY_IS(0)) { KernelA<fp32>(...); }
}
```

`<<<>>>` 直调模式改为：
- **按 tilingKey 的实际划分维度拆独立入口函数**。tilingKey 划分依据因算子而异（dtype、计算模式、shape 分支等），按原算子实际用什么维度分就怎么拆
- 每个入口只实例化自己对应的模板路径
- 去掉 `extern "C"`
- 去掉 `GET_TILING_DATA` / `TILING_KEY_IS`
- `GM_ADDR tiling` → `const XxxTilingData tilingData`（by value）
- 如果原始代码有 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` 等宏则保留；没有不要主动添加

**入口函数的约束**（放在独立的 `<op>_entry.h` 等文件中，不要混进 kernel 实现头）：
- 只做 3 件事：`AscendC::TPipe pipe;` → 构造 kernel 类对象 → `op.Init(...); op.Process();`
- 不出现 `DTYPE_*` 宏、`GET_TILING_DATA`、`TILING_KEY_IS`——这些都属于原注册框架
- 入口头 include `kernel_operator.h` + `<op>_tilingdata.h` + kernel 实现头
- 避免在入口头里 `using namespace`，优先用 `::AscendC::...` 这类显式限定

把入口单独放进独立头文件的原因：不污染原 kernel 文件的零修改属性（原则 1）。主机集成文件只 include 入口头，不直接 include kernel 实现头。

**关于模板 `__global__` 函数的已知问题**：`-xasc` 编译器在处理"模板 `__global__` + `<<<>>>` 调用"且模板参数为 `bfloat16_t` 时，`GM_ADDR`（`__gm__ uint8_t*`）参数会被错误 mangle 成 `__bf16`，链接时报 undefined symbol。遇到此问题改用显式命名的非模板入口（如 `op_fp32` / `op_fp16` / `op_bf16`）。

**Host launch ABI 一致性**：如果入口参数是 `GM_ADDR`，host 侧 device 指针也从 `aclrtMalloc` 开始就保持 `GM_ADDR`：
```cpp
GM_ADDR inputDevice = nullptr;
aclrtMalloc(reinterpret_cast<void**>(&inputDevice), size, ACL_MEM_MALLOC_HUGE_FIRST);
op_fp32<<<tl.usedCoreNum, nullptr, stream>>>(inputDevice, ..., tilingData);
aclrtFree(inputDevice);
```
不要先用 `void*` 持有，launch 时再 `reinterpret_cast<GM_ADDR>(voidPtr)`——AscendC 编译器会直接拒绝这种转换。

### Step 6: TilingData 定义改造（原则 3 第 2 项）

把 `BEGIN_TILING_DATA_DEF` / `TILING_DATA_FIELD_DEF` / `END_TILING_DATA_DEF` 机械替换成普通 C++ struct：
- 不需要加 `#pragma pack`
- 字段名、字段类型、字段顺序**与宏定义一致**
- **默认保留所有字段**（按"全量迁移优先"原则）。即使 kernel 没访问某些字段，也不要主动删——可能通过宏字符串化、SFINAE 间接使用。只有用户明确要求裁剪（如"quant 字段都可以删"）时才删指定字段
- 放在 `<op>_tilingdata.h`（仅依赖 `<cstdint>`，kernel 侧和 host 侧共用）

### Step 7: Host tiling 接口替换（原则 3 第 3 项）

如果用户要求 host tiling 可独立编译：

**接口替换**（功能等价，不改数学）：
```cpp
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
const NpuArch npuArch = ascendcPlatform->GetCurNpuArch();
uint64_t ubSize = 0;
ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
const uint32_t totalCore = ascendcPlatform->GetCoreNumAiv();
```
需要 include `"platform/platform_ascendc.h"`。

**其他替换**：
- `OP_LOGE` / `OP_LOGI` / `OP_LOGD` → `throw std::runtime_error(...)` 或 `printf`
- `ge::DataType` → 自定义 enum
- `gert::Shape` → 传 `uint64_t` 维度
- `IMPL_OP_OPTILING` / `REGISTER_TILING_DATA_CLASS` → 直接删

**数学逻辑严格保真**（再次强调原则 2）：
- 对齐/非对齐分支独立保留
- 分子 / 分母 / 上下取整按原样
- **默认保留所有 variant 路径**（grad / quant / fullload / splitD 等）。按"全量迁移优先"原则，不要擅自删减路径；只有用户明确要求（如"只要 fullload"）才裁剪指定路径
- 所有保留的路径内部逻辑一行不动

### Step 8: `-xasc` host/device 双 pass 编译兼容（有条件触发）

**判断是否需要本节**：`Grep` kernel 实现头中是否出现 `MicroAPI::` / `Reg::` / `__VEC_SCOPE__` / `RegTensor` / `MaskReg`。没有就跳过本节。

`-xasc` 编译器对同一个 `.cpp` 做两次编译：host pass 定义 `__NPU_HOST__`，device pass 定义 `__NPU_ARCH__`。上述 MicroAPI/Reg 相关类型仅 device pass 可用。

**需要的改动**（仅针对使用了这些类型的实现头）：

1. 所有使用 MicroAPI/Reg 的 kernel 实现头，在 include 之后、namespace 开始前加 `#if !defined(__NPU_HOST__)`，文件末尾 namespace 关闭后加对应 `#endif`
2. **不要**给 `local_deps.h` / `tilingdata.h` 等纯类型定义头加 guard——它们 host/device 两侧都需要可见

**注意**：加 `__NPU_HOST__` guard 本质是对 kernel 实现头的 include 和宏保护的"包裹"，不是对 kernel 代码内容的修改。kernel 类体、算法函数本身仍然是零修改。

**`__global__` 入口的兼容模式**：函数签名放在 guard 外（host pass 需要它生成 launch stub），函数体内部用 `#if !defined(__NPU_HOST__)` 保护实现：
```cpp
__global__ __aicore__ void my_kernel(GM_ADDR x, ..., const TilingData td) {
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    MyKernelClass<...> op(&pipe);
    op.Init(...); op.Process();
#endif
}
```

错误做法：
- 把整个 `__global__` 函数放在 `#if !defined(__NPU_HOST__)` 里 → host pass 看不到函数声明，报 `use of undeclared identifier`
- 在 guard 外放 forward declaration、guard 内放定义 → device pass 同时看到声明和定义，报 `call is ambiguous`

### Step 9: Host 驱动（可选）

如果用户要求独立可运行，按"Host 驱动来源决策"章节选择样板来源：目标仓有邻居 story 就对齐邻居；没有才读 `references/host-driver-template.md` 作为起点。

---

## 输入校验策略（可选）

如果需要在主机侧做输入校验，拆成独立函数 `check_inputs()`，不要混进 Meta 函数或 shape inference（shape 推断和合法性校验是两件事）。

**做**：
- Attr 自身合法性（`bs > 0`）
- 每个张量的 rank 和 dtype
- 与 attr 或张量形状直接可推导的跨张量关系

**不做**：
- 依赖张量数值才能判定的关系（会触发 D2H 同步，违背 NPU 异步调度）
- 调用方惯例差异导致的跨张量等量关系（校验过死会误伤合法用法）

原算子 `op_tiling` 里通常也没有数据依赖校验，保持对齐即可。

---

## 输出格式要求

完成任务时至少给出：

1. **目标文件集合**：最终落地了哪些 `.h` / `.cpp`
2. **被删除的相对 include**：精确到文件和 include 语句
3. **本地化的依赖清单**：按原来源头文件分组列出搬了哪些符号
4. **`local_deps.h` 职责**：装了哪类依赖
5. **剩余外部假设**：哪些宏 / tiling / dtype 仍依赖外部环境
6. **静态验证结论**：`..` include / 错误命名空间 `using` / 本地闭包是否通过

若任务是 `<<<>>>` 直调 / standalone：明确说明当前版本是 "kernel-only 自包含" 还是 "kernel + tiling + direct-launch glue 闭环"。

---

## 验证清单

详见 `references/custom-op-to-kernel-launch-checklist.md`。核心静态检查：

1. **相对 include 清零**：目标目录 grep `#include "../` 应无匹配
2. **命名空间归属**：所有 `using XXX::symbol` 的 symbol 真实定义必须在 XXX
3. **本地闭包完整**：kernel 引用的所有非 SDK 符号能在目标目录或稳定 SDK 头解析
4. **死引用清理**：无未使用 `using` / include
5. **入口文件自包含**：若同时适配了入口，明确说明 `DTYPE_*` / `GET_TILING_DATA*` / `TILING_KEY_IS` / tiling struct 的来源
6. **Host launch ABI 一致**：host 变量 / launch 调用 / kernel 入口三侧 ABI 一致（全用 `GM_ADDR`）
7. **Tiling 公式逐分支等价**：对齐/非对齐分支仍独立；无"统一化"合并

---

## 常见坑

1. **误把整份 `*_base.h` / `*_common.h` 原样复制**——拖一堆无关实现进来，只搬最小闭包
2. **不核实真实定义源**——`using A::foo` 但 `foo` 实际定义在命名空间 `B`，迁移时必须追到真实定义位置
3. **只删 include 不补 helper**——符号解析断掉
4. **纯 kernel 依赖解耦和 host 集成混做**——先确认交付边界再推进
5. **忽略入口文件仍依赖原工程宏**——include 改完 `*_apt.cpp` 看起来"在当前目录里了"，但仍不能独立编译。必须显式说明
6. **host 先用 `void*` 再 `reinterpret_cast<GM_ADDR>`**——AscendC 编译器会直接拒绝这种转换
7. **抽 tiling 时合并对齐/非对齐分支**——最危险的改动，不会立即编译错但会悄悄改变切块语义
8. **不区分 host/device pass 直接搬 MicroAPI/Reg kernel 头**——host pass 报大量类型不存在错误，需加 `__NPU_HOST__` guard
9. **模板 `__global__` 函数触发 bf16 mangling 错误**——改用显式命名非模板入口
10. **host 自定义 struct 直接作 kernel 模板参数**——host 辅助类型（如 `SampleBFloat16`）只用于 host 侧数据生成和 golden，launch 时应映射为 `bfloat16_t`

---

## 模式识别：快速诊断

开始前先诊断算子的迁移工作量重心：

| 诊断维度 | 重外部依赖型 | 轻外部依赖型 |
|---------|-------------|-------------|
| kernel 中 `..` include 数量 | 多（3+） | 少或无 |
| 主要工作量 | 依赖闭包提取 | 入口改造 + tiling 提取 |
| `local_deps.h` 内容 | 装外部搬运的函数/常量 | 仅集中 SDK include |

多数算子混合两种特征。统计 `..` include 数量 + 看 tilingKey 分发逻辑，就能判断主要工作量在哪。但不管属于哪一类，三条核心原则都适用。

---

## 工具偏好

- `Glob` 找文件
- `Grep` 找 include 和符号引用
- `Read` 读源文件和定义文件
- `Edit` 改现有文件
- `Write` 只用于新增 `local_deps.h` / `entry.h` / `tilingdata.h` 或新增目标副本

不要用 shell `grep/cat/find` 代替这些专用工具。

---

## Host 驱动来源决策

如果交付边界要求写 standalone host 驱动（independent `main` + golden 校验），按以下优先级决定样板来源：

1. **目标仓库有可参考的同类 story / sample**（有现成的 ACL init、main 骨架、golden 比对模式）
   → **优先对齐目标仓邻居**。读 1–2 个邻居 story 的 main.cpp 归纳模式，本算子跟那个体系走。不要引入"目标仓里没人这么写"的样板。

2. **目标仓里没有 standalone host 驱动惯例**（新建的独立 story，没有参考对象）
   → 读 `references/host-driver-template.md`，里面有 ACL init / BFloat16 辅助 / 类型转换 trait / 结果比对 / `main` 骨架，算子无关，替换 `/* OPERATOR-SPECIFIC */` 标注部分即可上手。

3. **任务不需要 host 驱动**（只做 kernel 依赖解耦、或集成到 PyTorch extension / aclnn wrapper / CI 框架）
   → 不要碰这部分，也不要读 template。

无论哪种来源，都要贯彻一条硬约束：**`GM_ADDR` 贯穿 host / launch / kernel 三侧**，不要用 `void*` 中转再 `reinterpret_cast<GM_ADDR>`（AscendC 编译器会直接拒绝这种转换）。

CMakeLists.txt 属于"工程约定"，按目标仓邻居的 cmake 体系对齐（见上文"信息来源约束"）。

---

## 一句话原则（再次强调）

**kernel 代码零修改。tiling 数学零修改。只改注册框架胶水。默认全量迁移，不做选择性裁剪。**

**算法源只来自原算子源码；工程约定（目录/命名/CMake）向目标仓邻居 story 对齐；host 驱动样板优先对齐目标仓，目标仓无参考时读 `references/host-driver-template.md`。**
