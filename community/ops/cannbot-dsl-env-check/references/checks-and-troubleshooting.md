# 检查定义与故障处理

本页说明各检查阶段的执行内容、通过条件、失败影响和处理方法。执行模式与编译门禁以 `SKILL.md` 为准。

## 状态语义

| 状态 | 含义 |
|---|---|
| `PASS` | 已执行且满足通过条件 |
| `FAIL` | 已执行但未满足通过条件 |
| `SKIP` | 因前置检查失败而未执行 |

## CANNBotDSL 导入

检查使用当前选定解释器导入 `cannbotdsl`，并记录实际模块文件和版本。

通过条件：导入成功，而且模块信息可以解析。

失败影响：DSL 翻译和 DSL 编译标记为 `SKIP`，编译门禁失败。

处理方法：

- 使用同一解释器执行 `python -m pip show cannbotdsl`；
- 使用同一解释器执行 `python -m pip check`；
- 检查 wheel 是否安装到当前解释器；
- 检查 wheel 的 Python ABI、系统架构和依赖是否匹配；
- 不要用 `PYTHONPATH` 指向源码树来掩盖安装问题。

## DSL 翻译

检查设置 `CANNBOTDSL_PIPE_STAGE=translate`，通过 `cannbotdsl.compile(probe_entry, *specs)` 编译固定的 `@host → @kernel` 探针，验证 DSL 前端和 AscendC 翻译。固定探针位于真实 `.py` 文件中，以满足前端获取 AST 源码的要求。空 Kernel 保留一个 Tensor 参数以满足编译入口契约，仅验证入口编译；不依赖 UB 分配、搬运或核信息 API。

通过条件：翻译完成，ProviderCallable 的 `so_path` 为空，并且 `bisheng` 没有运行。

失败影响：DSL 编译标记为 `SKIP`，编译门禁失败。

处理方法：

- 确认实际导入的 CANNBotDSL 模块路径和版本；
- 核对 wheel 与 CANN Toolkit 的版本关系；
- 检查 AscendC 翻译依赖是否正确加载；
- 保留固定探针的原始错误，不要修改业务 kernel 来规避环境问题。

## DSL 编译

检查设置 `CANNBOTDSL_PIPE_STAGE=compile`，编译与翻译阶段相同的固定探针。

通过条件：ProviderCallable 的 `so_path` 非空，并且对应 `.so` 文件真实存在且非空。该阶段不执行生成的程序；检查后调用 `close()` 释放编译返回对象。

失败影响：编译门禁失败。

处理方法：

- 检查当前进程中的 `ASCEND_HOME_PATH`、`PATH` 和 `LD_LIBRARY_PATH`；
- 检查 `bisheng` 及编译后端是否可执行；
- 核对 CANNBotDSL wheel 与 CANN Toolkit 的版本关系；
- 检查编译缓存和输出目录权限。

## 探针失败定位

`PROBE_INTERFACE_MISMATCH` 表示探针依赖的公开接口、规格构造或返回对象接口不满足当前调用要求。按已安装版本更新探针，并保留门禁失败；不回退旧接口或自动改写探针以获得 PASS。

`DSL_TRANSLATE_FAILED` / `DSL_COMPILE_FAILED` 记录编译调用失败的阶段，不能单凭错误码断定根因是 Toolkit。接口签名或语义变化也可能在编译期间暴露。直接运行 `probe_compile_stage.py translate` 或 `compile`，结合 stderr 原始堆栈及 environment.yaml 的版本、模块路径定位。

`PROBE_ARTIFACT_INVALID` 表示编译结果未满足该阶段的产物条件；`PROBE_CLEANUP_FAILED` 表示程序对象释放失败。以上结果均使编译门禁失败。

## NPU 运行环境

检查在同一解释器中依次验证：

- `torch` 可以导入；
- `torch_npu` 可以导入；
- `torch.npu.is_available()` 为真；
- `torch.npu.device_count()` 返回可用设备；
- 可以读取当前进程可见设备的逻辑编号和名称。

该检查独立于编译门禁，只判断当前进程是否具备继续真机验证的条件。检查通过不代表 DSL kernel 已经运行，也不代表数值精度或性能已经验证。

失败影响：编译能力链通过时，执行模式为 `compile_only`；允许继续 DSL 生成和编译，但真机精度与性能必须标记为未执行。

| 问题代码 | 含义 | 重点检查 |
|---|---|---|
| `TORCH_IMPORT_FAILED` | 无法导入 `torch` | 当前解释器中的 torch 安装 |
| `TORCH_BACKEND_AUTOLOAD_FAILED` | `torch` 自动加载 `torch_npu` 后端失败 | torch、torch_npu、CANN 的版本配套和后端扩展 |
| `TORCH_NPU_IMPORT_FAILED` | 无法导入 `torch_npu` | torch、torch_npu、CANN 的版本关系和动态库 |
| `NPU_QUERY_FAILED` | 查询 NPU 状态异常 | 驱动、运行时初始化和设备权限 |
| `NPU_UNAVAILABLE` | 后端可导入但没有可用设备 | 设备映射、驱动状态、容器挂载和 CANN 环境 |

## CANN 补充环境检查

CANN Toolkit、OPP、vendor、Simulator 和工具检查用于补充定位，不替代固定探针。

- 真实 DSL 编译通过时，以编译结果为准；
- 补充检查存在警告时，仍在 `environment.yaml` 中保留事实和原因；
- 无法可靠获取的硬件参数保持为 `null`，不根据设备名称猜测；
- 当前进程可见设备与服务器物理设备分别记录，不混为同一作用域。
