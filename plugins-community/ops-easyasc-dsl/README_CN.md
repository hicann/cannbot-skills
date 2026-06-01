# ops-easyasc-dsl

[English README](README.md)

***有天我一拍脑袋想看看 AI 究竟能做成啥样，所以就有了这个仓。代码基本是纯添加零天然。我只是个傻读代码和指手画脚的人类罢了。***

`ops-easyasc-dsl` 把 easyasc DSL → AscendC 的工作流封装成一个 skill。其内部仍然提供用 Python 描述 Ascend 风格混合 kernel 的 DSL 框架，它可以：

- 从 Python 代码生成指令 IR
- 把 IR 下沉成拆分后的 cube/vec 代码路径
- 在内置模拟器中运行 kernel
- 为非模拟器场景生成 custom-op 源码产物

这个仓库主要围绕三件事展开：

1. 用 `@kernel`、`Tensor`、`GMTensor`、`Var` 以及可选的 `@vf` 微函数辅助，在 Python 里编写 kernel。
2. 由框架负责构建指令 IR、补上各执行侧需要的同步，并把程序拆成 cube 和 vec 两条路径。
3. 再通过模拟器或生成出来的运行时产物验证结果。

## Skill 入口

面向用户的 skill 入口是 [`skill/SKILL.md`](skill/SKILL.md)。可复用的工作流位于 [`agent/`](agent/) 下。

在阅读已归档的运行时/文档内容或运行示例之前，需要先按需还原：

```bash
bash agent/scripts/init.sh
```

脚本是幂等的，只会还原缺失的目录。

## 为什么会有这个仓库

这个代码库主要服务于 kernel 的开发、实验和调试，尤其适合下面这些场景：

- 快速验证新的纯 cube 或混合 cube/vec 流水线原型
- 在模拟环境里验证尾块处理、tiling 和精度边界
- 查已有 kernel，看看哪些 DSL 写法和实现套路是可行的

## 安装

不用安装。执行 `bash agent/scripts/init.sh` 还原 `easyasc/` 后，想办法把 `easyasc.a5` 或者 `easyasc.a2` 给 import 进来即可。

## 快速开始

示例环境（仅作参考，并非必须）：

```bash
# 仅作参考——请根据本地情况调整
conda activate torch210npu
```

然后运行一个最小可执行的 kernel 示例（需先执行 `bash agent/scripts/init.sh`）：

```bash
python agent/example/kernels/a5/matmul_float_mmad.py
```

这个示例展示了最小的端到端流程：

1. 用 `@kernel` 定义一个 kernel
2. 通过 `OpExec(..., simulator=True)` 启动它
3. 把模拟输出和 PyTorch 参考结果做对比

## `OpExec` 真机构建 + CANNSIM 所需环境变量

使用默认的 `simulator=False` 跑 `OpExec(kernel)` 时，框架会在**当前工作目录**生成并执行 `b.sh`、`r.sh`。脚本里已写入通用基线；若本机布局不同，请在启动 Python 前自行导出下表变量。

| 变量 | 何时需要设置 | 示例值 |
|------|--------------|--------|
| `ASCEND_HOME_PATH` | 必须；指向你的 CANN 安装根 | `<你的 CANN 安装路径>`（其下应有 `bin/setenv.bash`） |
| `ASCEND_CUSTOM_OPP_PATH` | 需要拼接额外自定义 OPP 根 | 可为空；导出后避免 `set -u` 与 `set_env.bash` 冲突 |
| `EASYASC_PYTHON_BIN` | CANN `opbuild` 调用的 `python3` 需要 **NumPy** | 含有该解释器的目录（如某 conda 环境的 `bin`），会加入 `PATH` |
| `PYTHONPATH` | 需从仓库根 import `easyasc` | 仓库根目录 |

`b.sh` / `r.sh` 用 **`EASYASC_ROOT`**（脚本所在目录）拼接相对路径；请在**仓库根目录**跑代码生成，使脚本落在根目录。脚本会尝试 `source "${ASCEND_HOME_PATH}/bin/setenv.bash"`，配置厂商库 `LD_LIBRARY_PATH`，并用 `cannsim` 跑 aclnn 测试二进制。

若 `op_build` 缺 `.so`，检查 `ASCEND_HOME_PATH` 与 `setenv.bash`；若报缺少 `numpy`，设置 `EASYASC_PYTHON_BIN` 指向带 NumPy 的 Python 所在目录。

## 核心概念

- `easyasc.a5`：面向 A5 风格架构与指令序列的一套公开 DSL 接口。当前仓库里大多数 kernel 和测试都在使用它，包含 cube、vec、micro、寄存器、cast 和调试辅助功能。
- `easyasc.a2`：面向 A2 风格架构与指令序列的另一套公开 DSL 接口。它不是 `a5` 的兼容层，而是针对另一类指令族的并列架构接口。
- `GMTensor`：全局内存张量，对应 kernel 的输入和输出。
- `Tensor`：位于 `L1`、`L0A`、`L0B`、`L0C` 或 `UB` 中的片上张量。
- `DBuff` / `TBuff`：用来描述基于 slot 复用的缓冲张量辅助类型。
- `Var`：用于表示循环边界、维度和符号形状的标量值。
- `OpExec`：运行时入口，可用于模拟器执行或代码生成。

## 典型开发流程

1. 先把精确的 PyTorch 公式写清楚。
2. 选择流水线拓扑：
   - 仅 cube
   - cube -> vec
   - vec -> cube
   - vec -> cube -> vec
   - cube -> vec -> cube
3. 用 Python 把 kernel 实现出来。
4. 使用 `OpExec(..., simulator=True)` 验证。
5. 如果重复出现的标量维度让形状推导变得不明确，就显式加上 `shape_bindings`。
6. 只有在模拟器结果和参考实现对齐之后，再继续看生成产物和硬件相关执行。

## 仓库结构

- `skill/` — skill 入口（`skill/SKILL.md`）
- `agent/` — 可复用的 easyasc DSL → AscendC 工作流
  - `agent/ROUTER.md` — 渐进披露的 router
  - `agent/scripts/` — 维护脚本（包含 `init.sh`）
  - `agent/assets/` — 归档的 runtime/docs（`ops-easyasc-dsl-runtime.tar.gz`）与 example（`ops-easyasc-dsl-example.tar.gz`）
  - `agent/example/` — 精选 kernel 示例与手动 demo（按需还原）
  - `agent/references/` / `agent/playbooks/` / `agent/index/` — 参考、playbook 与 JSON 索引
- 由 `agent/scripts/init.sh` 按需还原：
  - `easyasc/` — DSL 运行时与 codegen
  - `doc/` — 英文文档
  - `doc_cn/` — 中文文档镜像
  - `agent/example/kernels/` — 精选示例 kernel
  - `agent/example/demo/` — 按设备家族整理的手动运行示例

注意：`testcases/` 已从交付的 skill 包中移除。

## 文档索引

`doc_cn/` 目录由 `agent/scripts/init.sh` 还原：

- [快速开始](doc_cn/01_quickstart.md)
- [编程模型](doc_cn/02_programming_model.md)
- [编写第一个 Kernel](doc_cn/03_write_your_first_kernel.md)
- [混合流水线与同步](doc_cn/04_mixed_pipeline_and_sync.md)
- [模拟器与 Trace](doc_cn/05_simulator_and_trace.md)
- [代码生成与运行时](doc_cn/06_codegen_and_runtime.md)
- [Kernel 模式与范式](doc_cn/07_kernel_patterns.md)
- [测试与验证](doc_cn/08_testing_and_validation.md)
- [API 参考](doc_cn/09_api_reference.md)
- [故障排查](doc_cn/10_troubleshooting.md)
- [面向贡献者的架构说明](doc_cn/11_architecture_for_contributors.md)
- [Stub 与 Codegen 名字对照表](doc_cn/12_stub_to_codegen_name_map.md)
