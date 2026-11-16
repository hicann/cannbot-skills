# kernel 产物与加载契约

需要准备或加载 kernel 时读取。本文件维护产物和 ABI；不负责生成任意语言 kernel 本体。

## 目录

- [产物与加载方式](#产物与加载方式)
- [Ascend C 二进制提取](#ascend-c-二进制提取)
- [加载属性与入口](#加载属性与入口)
- [参数与边界](#参数与边界)
- [缓存和分发](#缓存和分发)

## 产物与加载方式

| 产物 | 处理 | 官方示例 |
|---|---|---|
| Ascend C `.asc` 同库编译 | ASC CMake 工具链托管 binary，直接调用 launch 包装符号 | ascendc_add_custom |
| 算子工程 `.o + .json` | 核对 .o 是否为 device ELF，按 json 读取 metadata/ABI | 产物格式按实际 OPC/工具链确认 |
| Triton `.npubin` | 读取 bytes，BinaryLoadFromData→BinaryGetFunction→LaunchKernelWithHostArgs | triton_add_custom |
| TileLang `.so` | dlopen/dlsym 或链接；生命周期、依赖与分发需明确 | tilelang_add_custom 系列 |
| PyPTO JIT | 目标版本是否导出独立 binary 先核对；基线示例预热后在 Python 回调执行 | pypto_add_custom |
| 无 device kernel | HostCpuExecuteOp 或 Python Host execute | host_cpu_add_custom |

同一类产物可以用 C++ 或 Python 交付件接入；不要仅凭开发语言决定 GE 能力。
Ascend C 源码同库编译使用 `CMakeLists_ascendc.txt`；预编译产物使用普通 CXX 模板。

## Ascend C 二进制提取

不能把所有 `.o` 都视为 device ELF：直接编译源码的输出可能是含 device 段的 host 对象。
先用 `scripts/inspect_kernel.py` 区分，或用 readelf 核对 Machine/Type 和段表。

GE `python_compilable_add_custom` 的目标工具链示例：

```bash
bisheng -c my_kernel.asc -o host.o --npu-arch=dav-<NpuArch> -I<Ascend-C-include>
llvm-objcopy -O binary --only-section=.aicore_binary host.o device.o
readelf -h device.o
readelf -sW device.o
```

提取的是 `.aicore_binary`，不是 host `.text`；`-O binary` 避免再次包成 host 对象。
编译参数和 include 布局随安装版本变化，先核对编译器与官方同版本样例。
若需 `-x cce` 或另一套 arch 参数，按该工具链的帮助和最小试编确认；不把单次实测命令通用于所有版本。

输出必须是可加载 device 格式，非空文件只是必要条件。对于官方 device ELF 路线，确认 ELF64、可执行类型、达芬奇 machine（示例 0x1029）及真实全局入口。
脚本拒绝 host ELF，不把该 ELF 检查套用于所有第三方容器或共享库。

## 加载属性与入口

优先从实际 `.json`/device metadata 核对 kernelName、magic/coreType、blockDim、workspace 和参数布局。
blockDim/workspace 的 -1 表示待决定，不能直接转无符号参数下发。
源码 KERNEL_TYPE 标注不能代替产物检查；出现 `_mix_aic/_mix_aiv` 等符号时需按工具链确认入口。

GE 的 magic 字符串与 ACL 的 binary load 数值选项是两套接口，不直接互换。
命令式加载参数按目标 `acl_rt.h` 核对，检查每一步返回值并保持资源生命周期。
声明式 launch info 无 magic 字段，但 GE 构建侧会读取节点已有 TVM_ATTR_NAME_MAGIC，缺失才使用默认值；不能据此断言 MIX 完全不可表达。
非默认加载路线的验证要求见 [场景选型](scenarios.md#加载属性兼容性)。

## 参数与边界

- 参数顺序、标量宽度、packed/对齐规则都取自真实 kernel ABI，不能统一假定所有语言使用 Triton 布局。
- Triton 某些产物包含 grid 参数或 FFTS 地址，Ascend C 某些入口只需 I/O 地址；逐项核对。
- Annotated 的追加顺序就是 kernel 参数顺序；检查 AppendArg 和 AddLaunch 返回结果。
- ceil(n/block_size) 不会自动为 kernel 添加尾块保护。只有 kernel 已做边界处理时才能接受尾块。
- 空张量需定义为跳过 launch 或专门路径，不能一律强制 block_dim=1 后访问空内存。
- 本 Skill Python 骨架按固定 1024 float 元素块约束，只接受正数且整除的输入；若扩展范围，必须同时修改 kernel ABI、Host 校验和验收用例。

## 缓存和分发

只缓存自有 bytes、metadata 或有清晰释放策略的资源，不缓存借用 Tensor/ctx。
缓存键覆盖特化输入输出、attrs、平台；持久缓存还覆盖源码、依赖、编译参数和工具链。
无法完整确定依赖时关闭编译缓存。写产物使用独立工作目录，避免多个工程/进程互相覆盖。

编译时机与交付形态是两个独立选择：

| 场景 | 默认方案 | 备选与适用条件 |
|---|---|---|
| 在线命令式，已有或构建期预编译的 device binary | 将 binary 作为独立文件随 `.so` 分发，运行时读取并加载 | 有单 `.so` 或无法分发独立文件等需求时，可评估把预编译字节内嵌 `.so` |
| Ascend C 源码 | 可评估 ASC 工具链的源码同库编译，也可预编译后分发独立 binary | 按现有工程、目标工具链和部署约束选择；源码同库编译与把预编译字节内嵌是不同机制 |
| 离线 OM | 确认 binary 和额外状态的携带渠道；命令式独立状态默认评估 `PortableOp` | 声明式收集或同库方案可能适用，见 [离线状态携带](scenarios.md#离线状态携带) |

推荐预编译不意味着必须内嵌：不要将「读文件」和「内嵌字节」合并为一个推荐选项，也不要从用户选择「预编译」推断其要求单 `.so`。在线默认方案应明确 binary 的构建输出、安装位置和运行时读取路径，并验证从其他工作目录启动仍可加载；不依赖临时目录。
内嵌可减少独立文件，但更换 kernel 需要重新链接、增加工具链及链接符号约束，应说明取舍。
用户明确给出交付约束时遵循其选择；已有工程保留原有方案，除非用户要求改造。

与交付件同目录不等于自动进入 OM；离线部署在移走构建目录和 OPP 后，以新进程验证真实依赖。
离线交付件还须链接 `custom_op_registry_static`，使 `.so` 导出 pull creator 符号 `GetRegisteredCustomOpCreator*`；缺失会让 OM 加载期符号解析失败，并导致整个模型加载失败。
链接与验收见 [OPP 与构建](build-verify.md#opp-与构建)。
