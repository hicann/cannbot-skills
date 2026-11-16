# 构建、部署与验收

生成工程或部署时读取。故障排查见 [troubleshooting.md](troubleshooting.md)，接口和源码依据见 [sources.md](sources.md)。

## 目录

- [模板选择](#模板选择)
- [OPP 与构建](#opp-与构建)
- [Python 构图到 OM](#python-构图到-om)
- [离线 OM 执行契约](#离线-om-执行契约)
- [分阶段验收](#分阶段验收)
- [完整示例](#完整示例)

## 模板选择

| 需求 | 取用文件 | 还需完成 |
|---|---|---|
| C++ 命令式 | my_eager_execute_op.cpp | 真实 binary 加载、launch、资源管理 |
| C++ Compile + Portable | my_compilable_portable_op.cpp | 编译、launch、完整缓存 key 与格式校验 |
| C++ 声明式 | my_annotated_args_op.cpp | 真实产物、加载属性、参数和推导 |
| C++ shape/dtype 推导 | my_shape_infer_op.cpp | 合并到同一个实现类，不单独注册第二个实现 |
| C++ 原型与普通构建 | proto_reg.h、proto_reg.cpp、CMakeLists.txt | 统一 op type、设置支持的 dtype/attrs |
| Ascend C 源码同库 | CMakeLists_ascendc.txt | ASC 工具链、launch 包装和 kernel 源码 |
| Python 接入骨架 | python_custom_op.py | 同版本 kernel、ES 原型/构图、bridge、部署 |

不要把全部 *_op.cpp 一次复制进 src；每个算子只注册最终选定的实现类。
proto_reg.cpp 包含原型头，保证 REG_OP 实际进入交付件；只复制/安装头文件不等于执行注册。
自动映射时统一类名和原型；保留类名时使用显式 op type 注册。
骨架包含未实现步骤时会明确失败，编译通过不是完整工程验收。

## OPP 与构建

```text
project/
├── CMakeLists.txt
├── src/
│   ├── custom_op.cpp
│   ├── proto_reg.cpp
│   └── proto_reg.h
└── build/opp/
    └── op_graph/
        ├── include/proto_reg.h
        └── lib/linux/<x86_64|aarch64>/libcust_opapi.so
```

GE 有两条加载路径：OPP 根的固定名 libcust_opapi.so，以及 op_graph/lib/<os>/<arch>/ 下任意名 .so。优先用子目录，避免根目录重名。
其他构图入口所需的 TF/ONNX parser 或 PyTorch converter 按任务额外提供。
Python-only 实现无需强造 C++ 实现库，但 ES 代码生成可能需要单独的原型库。

```bash
cmake -S . -B build -DASCEND_HOME_PATH="$ASCEND_HOME_PATH"
cmake --build build --parallel
```

按实际模板确定 OPP_OUTPUT_PATH，检查产物存在和导出/未解析符号，再设置 ASCEND_CUSTOM_OPP_PATH。
保留旧环境条目前先确认路径有效；Python loader 对不存在路径会硬失败。
Linux 上检查动态依赖和 ABI，不能把 `--no-as-needed` 视为替代正确链接与注册的通用修复。
Ascend C 同库编译按 ASC 工具链组织并链接 ascendc_runtime，不把外部共享库误当 ASC 源码编译。

离线（OM）交付件必须链接**静态库** `libcustom_op_registry_static.a`，与 kernel 语言无关。
CMake 里写库名 `custom_op_registry_static`，它位于安装目录 `lib64/`；该 target 声明为 STATIC、只安装 `.a`，**没有同名 `.so`**，按 `.so` 查找会误判为库不存在而漏链。
它提供 pull creator 符号 `GetRegisteredCustomOpCreatorAbiVersion`、`…Num`、`…s`。

成因在注册宏：交付件加载时构造 `CustomOpCreatorRegister`。
在线会经 `CustomOpFactory::RegisterCustomOpCreator` **主动注册 creator**，故官方 online 样例未链接也能运行（`eager_attached_stream_add` 是仍链接的例外）。
离线加载 OM 内算子 `.so` 时 `IsOfflineCustomOpSoLoading()` 为真（由 `ScopedOfflineCustomOpSoLoadGuard` 置位），工厂注册被抑制，GE 改为对 `.so` 句柄 `dlsym` 这三个符号拉取 creator，交付件必须自带符号。
该解析位于 `ResolvePullCreatorSymbols`（`base/common/helper/custom_op_registry_builder.cc:66`），入口如 `aclmdlLoadFromFile`。
缺失即日志 `[CUSTOM OP] pull creator ABI symbols are incomplete.`，并导致**整个模型加载失败**，不只是该算子不可用。

pull registry 只收 `kTop` + `kCustom` 的实现（`NeedRegisterLocalCreator`）：只注册 `kBottom` 的实现即使链接了该库，离线也 pull 不到。
交付件可能被 SO in OM 收集或复用离线加载时，一律按离线要求链接；两套模板 CMake 已含该库，不要照搬在线样例删除。

## Python 构图到 OM

以官方 `annotated_args_refresh_add_custom/offline/python` 为对照：

1. CMake 使用 GenerateEsPackage 的 add_es_library_and_whl，构建原型库与 ES wheel。
2. 将 wheel 安装到工程私有目录；设置 PYTHONPATH、ES 库路径和 Python 插件目录。
3. 用 GraphBuilder 构图并 save_to_air。
4. 使用目标 atc 编译 AIR，再独立进程加载 OM 并对拍。

```python
from ge.es.graph_builder import GraphBuilder
from ge.es.custom import AnnotatedAddCustom  # 来自生成的 ES 包
from ge.graph.types import DataType, Format

builder = GraphBuilder("my_graph")
x = builder.create_input(index=0, name="x", data_type=DataType.DT_FLOAT,
                         format=Format.FORMAT_ND, shape=[8192])
y = builder.create_input(index=1, name="y", data_type=DataType.DT_FLOAT,
                         format=Format.FORMAT_ND, shape=[8192])
builder.set_graph_output(AnnotatedAddCustom(x, y), 0)
builder.build_and_reset().save_to_air("build/my.air")
```

```bash
atc --model=build/my.air --framework=1 --output=build/my --soc_version=<目标SoC>
```

交叉编译的 host_env_os/host_env_cpu 等参数按目标工具版本设置。
`ge.es.custom` 是生成产物，不是手写占位模块；模块与 target 对应关系参考官方 CMake。
典型环境条目为 build/whl_package、build/es_output/lib64，以及 OPP 根和 src/ge。
离线依赖完整性按 [场景选型](scenarios.md#离线状态携带)检查。

## 离线 OM 执行契约

先区分两种「离线动态」，ATC 参数、OM 体积与执行能力都不同：

| 形态 | ATC 参数 | 执行能力 | 适用 |
|---|---|---|---|
| 多档位 | `--input_shape="x:-1;y:-1"` + `--dynamic_dims="1024,1024;4096,4096"` | 每档仍是静态图，声明式可用 | shape 可枚举 |
| unknown 维 | 只给 `--input_shape="x:-1;y:-1"` | 运行期按当轮 shape 执行，需命令式 | shape 不可枚举 |

`--dynamic_dims` 用分号分隔**档位**、逗号分隔同档内各 `-1` 维，OM 内按 `gears`/`gear_idx` 组织；`--help` 写的 "Separate multiple nodes with semicolons" 容易读反。
`dynamic_batch_size`、`dynamic_image_size`、`dynamic_dims` 只能设一个；`insert_op_conf` 与 `dynamic_dims` 不能同时给；模型含 `-1` 维而未给 `--input_shape` 报 E10001。
用户说「动态 shape」时先确认是可枚举档位还是任意 shape，再定接口与参数。

执行序列按 OM 形态选择，两者不通用：

```text
静态/多档位 OM：aclmdlLoadFromFile → aclmdlExecute
unknown 维 OM ：desc buffer size 返回 0，须按本轮 n 分配
              → aclCreateTensorDesc + aclmdlSetDatasetTensorDesc（输入、输出都要设）
              → aclmdlCreateExecConfigHandle（不得为 nullptr）
              → aclmdlExecuteV2(modelId, in, out, stream, handle)
```

`aclmdlExecConfigAttr` 只有超时、workspace、QoS、优先级与性能模式，**没有任何 shape/dims 项**：动态维度只能经 dataset 的 TensorDesc 传，不要往 config handle 里塞 shape。
`aclmdlExecuteV2` 内部按 `IsRuntimeV2Enable` 与 `GetExecutor(modelId)` 分派到 `RuntimeV2ModelExecute`；`aclmdlSetInputDynamicDims` 没有该分派，直接调 V1 的 `GeExecutor::SetDynamicDims`，对 RT2 model id 报 145003。

| 错误码 | 日志线索 | 根因 |
|---|---|---|
| 145003 | `Invalid model id` | 对 RT2 模型用 `aclmdlSetInputDynamicDims` 设动态维度 |
| 500002 | `[Calc][TensorSizeByShape] shape_size[-1]` | 用 `aclmdlExecute` 跑 unknown 维 OM，无 shape 通道 |
| 100000 | `[Check][handle] param must not be null.` | `aclmdlExecuteV2` 的 config handle 传了 nullptr |

E10003/E10004 属 ATC 参数族报错，实测遇到时先记录完整命令行再定位，不按固定映射假定。
OM 体积随档位数与模型规模变化（多档位与 unknown 维可差两个数量级），不作为固定倍数承诺。

## 分阶段验收

| 阶段 | 必须有的证据 | 不能替代的后续检查 |
|---|---|---|
| 静态 | 语法、原型/注册一致、产物 ABI 与未实现项检查；离线交付件的链接集合含静态库 `custom_op_registry_static` | 编译、加载、数值 |
| 构建 | 目标头文件/库下完整编译链接日志；离线交付件用 `nm -D --defined-only` 或 `readelf --dyn-syms` 确认导出 `GetRegisteredCustomOpCreator*` | dlopen 和回调被选中 |
| 加载 | 实际加载交付件、正确注册、进入预期执行分支 | 输出数值 |
| 数值 | 与参考计算比较，全部输出和约定误差 | 重放、跨进程部署 |
| 重放 | 多轮不同数据与换址、多个 shape/dtype/attrs、节点/模型隔离 | OM 自包含恢复 |
| 部署 | 新进程、移走构建产物或原 OPP、恢复自定义状态并数值对拍 | 其他硬件/版本兼容性 |

拒绝空张量/尾块的示例需要验证拒绝；声称支持的工程必须验证正确结果。
模板 TODO 只能报告为未完成，不能以成功状态跳过计算。
最终报告记录版本、命令、已通过阶段、未执行项及原因；不使用固定措辞代替证据。

## 完整示例

[assets/host_cpu_add](../assets/host_cpu_add/) 提供官方 Python Host 常量折叠工程的固定基线副本：原型、CMake、插件、GraphBuilder/Session、数值断言和运行脚本齐全。
在匹配 GE/CANN 环境中，从该目录运行 `bash run.sh`。
脚本仅向本工程 build/ 下安装生成 wheel，不更改系统 Python 包。
它验证常量折叠/Host 计算，不证明 device launch、动态输入或 OM 部署可用。
来源和适用范围见示例内 [说明](../assets/host_cpu_add/EXAMPLE.md)。

本机无匹配 CANN/NPU 时，执行 `scripts/check_env.py` 并记录缺口；可运行本 Skill 的 unittest 回归，但不能将模拟编译器检查当作真实 kernel 编译。
