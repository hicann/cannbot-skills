# Blaze 编译问题排查

## 适用范围

用于 Step 4 中实际发生的 CMake configure、ASC/C++ 编译、链接、头文件解析和项目构建边界问题。不处理运行时精度、layout、同步或定制场景专属问题。

## 触发信号

- CMake configure/generate 失败；
- ASC 源文件未按设备语言编译，或设备编译选项缺失；
- Blaze/tensor_api 头缺失，或同名符号、宏、模板、constexpr 语义冲突；
- `undefined reference`、入口符号或模板实例无法解析；
- 实际命令引用项目外历史目录、错误源码副本或旧构建路径；
- 构建结果与当前 action/checkpoint 的预期不一致。

## 先做什么

先保留完整错误输出，并查看失败 target 的实际 configure、compile 或 link 命令。根据错误只检查相关信息：源文件语言、include 来源、头文件解析路径、声明/定义、链接输入或缓存路径。不要在没有实际命令和错误证据时猜测库名、include 顺序或设备选项。

排障和修复以 `<project-root>/operators/<operator_name>/` 为项目边界，不受 PLAN manifest 或当前 action 文件清单限制。修改后更新 PLAN 第 2、4--8 章，重新执行受影响 action/checkpoint，并把错误、实际修改和结果追加到第 11 章 `execution_record`。

## ASC 语言和编译选项

当 ASC 源文件被按普通 C++ 处理，或设备侧选项没有进入命令时：

1. 查看实际 target 的源文件列表和语言属性；
2. 确认设备侧源文件使用当前工具链要求的 ASC 构建入口；
3. 确认设备架构和 ASC 选项只作用于相应 target/source；
4. 修改已授权的项目构建文件，重新生成并构建。

不要为快速通过而把所有 host 源文件都标为 ASC，也不要把设备选项设置成全局编译选项。

## 头文件来源和同名冲突

当报错涉及 Blaze/tensor_api 头、同名声明、模板或 constexpr 语义时：

1. 从实际编译命令区分项目内 include、工具链内置 include 和其他外部 include；
2. 必要时用同一 compiler、语言和参数查看预处理/include trace，确认真正解析的头文件；
3. 对照项目内 Blaze/tensor_api 副本的 include 链，判断是否被 CANN 或历史副本中的同名头抢占；
4. 在受影响 target 上做最小 include 调整并重新核验解析结果。

一个已知类型是：项目内 tensor_api 与 CANN 工具链同时提供同名头，但其中函数的 constexpr、设备侧可用性或模板声明不同。此时不能通过包装同名函数、宏重定义或修改工具链头掩盖冲突。若源码使用引号形式 include，且 trace 证明工具链头抢占了项目内同源头，可以只对受影响的 ASC target 增加项目内 `-iquote` 路径；这是一种证据驱动的修复，不是默认配置。

禁止修改编译器、CANN 安装目录、Blaze 源码、tensor_api 源码或 Skill Asset 原文件，也不要用强制预包含或伪声明绕过真实 API。

## 链接和项目边界

遇到未解析符号时，依次确认：

1. 定义所在源文件是否进入当前 target；
2. 声明、定义、Wrapper 和入口 ABI 是否一致；
3. 所需项目库或既有工具链/运行时库是否进入实际 link command；
4. 路径是否错误指向其他项目、历史构建目录或不同版本副本。

可以在项目根内修正 source、target 级链接配置和既有依赖路径，也可以补充项目内 helper 或构建文件，但不要照搬其他工程的库清单或全局链接目录。若修复需要新的外部依赖，返回 Step 3。

## Blaze 源码工作区问题

如果实际命令表明项目内副本、`ops-tensor/`、submodule 或缓存来源不一致：

- 只读核对 Step 1 建立的 Blaze 源码工作区和递归 submodule；
- 确认项目内 `blaze/`、`tensor_api/` 副本来自当前 Blaze 源码；
- 确认构建没有读取其他项目或历史目录；
- 源码位置、submodule 或项目副本不完整时停止修改并返回 Step 1；
- 当前 Blaze 源码的类型、入口、Params 或 API 事实不足时返回 Step 2。

不要用旧二进制、历史副本或项目外源码临时替代当前项目输入。

## 修复边界和验证

Step 4 可以在项目根内调整源码、include 顺序、target 编译选项和当前项目/既有工具链依赖的链接配置，以实现冻结设计。修复后重新执行失败的 checkpoint 及受影响的下游 checkpoint；构建通过后继续执行 PLAN 中的功能、精度和回归 checkpoint，不能把“编译成功”当作算子验证完成。实现问题应继续留在 Step 4 诊断和修复，除非证据要求改变冻结设计或补充上游源码事实。

以下情况停止修复并返回 Step 3：

- 需要改变实现路线、Blaze 组装方案或定制场景；
- 需要改变算子接口、Kernel ABI 或 Tiling/Params 语义；
- 需要改变数据语义、地址/ABI、支持范围或验证范围；
- 需要新的 Blaze 源码事实或新的外部依赖。
