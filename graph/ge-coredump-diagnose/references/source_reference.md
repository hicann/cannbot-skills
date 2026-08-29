# 源码引用与函数调用栈格式

诊断报告中引用源码位置或展示源码调用链时，统一使用本格式。源码位置必须能让读者直接回到对应代码仓定位文件和行号。

本格式用于报告里的**源码调用链 / 逻辑调用链**，不是 coredump 的原始 gdb backtrace。coredump 报告应先原样保留原始 backtrace，再按需补充源码调用链；没有真实运行时堆栈时，正文优先称“源码调用链”或“场景调用路径”。

---

## 1. 源码引用格式

### 1.1 源码别名

源码别名在诊断流程 Step 2 与用户确认：每个源码仓根路径取一个短别名，默认取仓目录名（如 `ge`、`metadef`）。别名一经确认全程不变，并在报告“源码版本”一节登记别名与根路径的对应关系。

### 1.2 函数定义位置

```text
ClassName::FuncName [别名/仓内相对路径/file.cc:line]
```

示例：

```text
GraphManager::BuildGraph [ge/compiler/graph/manager/graph_manager.cc:812]
OpDesc::GetIrAttrNames [metadef/inc/external/graph/operator.h:210]
```

### 1.3 调用点位置

需要证明父子调用关系时，在被调用函数后补充 Caller 函数体内的实际调用点：

```text
└─ ClassName::Callee [别名/path/callee.cc:80]
   ← call@ [别名/path/caller.cc:120]
```

定义位置回答“函数在哪里实现”，调用点回答“这条调用边在哪里发生”，两者不可互相替代。

### 1.4 通用规则

1. **函数名**：优先 `ClassName::FuncName`，不加参数列表；命名空间函数用 `namespace::FuncName`；无类和命名空间的 C 函数用 `FuncName`。
2. **源码位置**：统一 `[别名/仓内相对路径/file.cc:line]`；`path` 是对应源码仓根目录下的相对路径，不写本机绝对路径。
3. **版本一致性**：引用哪个版本就必须实际搜索该版本的物理目录；禁止用其他版本的源码或行号替代。
4. **行号**：优先单行号；确需短范围时用 `[别名/path/file.cc:start-end]`。
5. **日志位置**：日志行号写成 `<日志文件或目录>:<line>`，不混入源码引用格式。
6. **源码片段**：正文优先引用位置并解释行为；必须展示时只保留与结论直接相关的短片段。

---

## 2. 生成源码调用链的基本方法

1. 用 `grep -rn` 在目标仓全目录定位函数定义和调用点，不限定单一文件后缀；不使用 `rg`/ripgrep。
2. 阅读 Caller 的完整函数体，确认调用表达式、执行顺序、分支条件和关键参数。
3. 只展开与崩溃场景有关的路径；纯日志、无关 helper 和未命中分支可以省略，但要说明省略范围。
4. 先确认调用关系，再绘制 ASCII；禁止根据业务叙述、变量来源或函数名相似性直接推断缩进。
5. 没有 Caller 函数体内的调用点证据时，不得建立确定的父子调用关系；写入缺失信息或标为间接调用候选。

---

## 3. 默认轻量调用链格式

简单调用链不强制输出证据表，也不强制固定深度。使用固定树形前缀，在容易误解的边上标注调用点：

```text
ClassName::Root                                [别名/path/root.cc:40]
├─ ClassName::Prepare                          [别名/path/root.cc:55]
│  ← call@ [别名/path/root.cc:80]
└─ ClassName::Execute                          [别名/path/execute.cc:90]
   ← call@ [别名/path/root.cc:95]
   └─ namespace::Launch                        [别名/path/launch.cc:30]
      ← call@ [别名/path/execute.cc:120]
```

树形字符规则：

- 中间分支：`├─`
- 最后分支：`└─`
- 后续还有兄弟节点：`│  `
- 后续没有兄弟节点：三个空格
- 每深入一层增加三个字符宽度，不为视觉对齐破坏树形前缀

---

## 4. 缩进语义硬约束

### 4.1 允许增加缩进的关系

只有 `direct_call` 可以形成父子缩进：子函数的调用表达式必须直接出现在父函数体内。对 ASCII 树中的每条父子边，都应满足“最近父函数 == 边的 Caller，当前函数 == 边的 Callee，关系 == direct_call”。

### 4.2 必须保持同级的关系

同一个 Caller 函数体内的顺序调用必须保持同级，即使后一个调用使用了前一个调用的返回值。

源码语义：

```cpp
const auto &model = GetModel(model_id);
return model->NnExecute(...);
```

正确表达：

```text
ModelManager::ExecuteModel
├─ ModelManager::GetModel                    ← 返回 model
└─ DavinciModel::NnExecute                   ← model 作为 receiver
```

禁止表达为：

```text
ModelManager::GetModel
└─ DavinciModel::NnExecute
```

### 4.3 不属于父子调用的关系

- `data_dependency`：值或对象被后续调用消费，用 `←` / `→` 文字说明，不增加缩进。
- `return_value`：函数返回结果，用 `← 返回 ...` 标注，不增加缩进。
- `indirect_candidate`：函数指针、回调、虚函数或动态注册目标无法唯一确认时，使用 `⇢` 并标注“候选/待运行时确认”，不得伪装成确定调用。
- 仅有相同函数名、日志先后顺序或相同对象类型，不构成调用关系证据。

---

## 5. 分支呈现

互斥分支禁止混成一条调用路径。先用简短列表说明分支条件，再分别展示场景路径；默认只详细展开用户关注的路径，其他路径保留一行摘要。

```text
分支点：ModelManager::ExecuteModel

- P1 Hybrid：hybrid_model != nullptr
- P2 Static：hybrid_model == nullptr && static_model != nullptr
- P3 Error：两类模型均不存在
```

P1：

```text
ModelManager::ExecuteModel
├─ ModelManager::GetHybridModel              ← 返回 hybrid_model
└─ HybridDavinciModel::Execute               ← hybrid_model != nullptr
```

常见控制流：

- `if/else`、三目、`switch`：互斥结果分配不同 Path ID；
- 提前返回：路径末尾标注 `× 返回 <状态>`，不再向下展开；
- 循环：函数节点只画一次，用 `↻ 每次迭代` 标注，禁止复制多份相同子树；
- 错误处理：主成功路径和错误路径分开；用户未关注错误路径时可只列触发条件和返回值；
- 外层收口：调用返回上层后发生的同步、清理或结果转换，放回真实 Caller 下，不挂在下层被调用函数下面。

---

## 6. 复杂场景的调用边证据表

出现下列任一情况时，先建立调用边证据表：

- 多层或嵌套分支；
- 跨多个仓库或动态库（coredump 常见：GE 仓 + 外部 `.so`）；
- 函数指针、回调、虚函数或动态注册；
- 调用关系存在争议；
- 用户明确要求严格复核。

表格格式：

| Edge ID | Caller | Callsite | Callee | Definition | Relation | Condition |
|---|---|---|---|---|---|---|
| E01 | `ClassA::Run` | `[ge/path/a.cc:120]` | `ClassB::Prepare` | `[ge/path/b.cc:50]` | `direct_call` | always |
| E02 | `ClassA::Run` | `[ge/path/a.cc:135]` | `Callback` | `unknown` | `indirect_candidate` | callback registered |

ASCII 节点可附 `[E01]` 便于逐边反查。只有 `Relation=direct_call` 的边可以进入确定调用树。

---

## 7. 标注符号

标注写在源码位置或函数说明之后：

- `★`：关键决策、异常点、崩溃函数或目标函数
- `←`：返回值、数据来源、回指或关注点（如 `← 崩溃函数`、`← 非法参数来源`）
- `→`：数据去向或后续消费关系
- `⇢`：间接调用候选，目标未静态确认
- `↻`：循环执行
- `×`：提前返回或路径终止

这些符号只补充关系含义，不能代替真实的父子缩进证据。

---

## 8. 输出前自检

1. 每个子节点的调用表达式是否真的位于最近父函数体内？
2. 同一 Caller 的顺序调用是否保持同级？
3. Getter、返回对象和参数传递是否被误画成调用嵌套？
4. 互斥分支是否拆成不同场景路径？
5. 外层同步、清理和错误转换是否放回真实 Caller？
6. 间接调用是否明确标注不确定性？
7. 源码位置是否包含实际别名和行号？
8. 是否只展开用户关注路径？

任一项无法确认时，不得用确定语气输出对应调用边；应补读源码或写入缺失信息。
