# ops-easyasc-dsl

[English README](README.md)

***人工，然后智能。***

`ops-easyasc-dsl` 将 easyasc 的 DSL 到 AscendC 工作流打包为一个 skill。
easyasc 是一个用于编写 Ascend 风格 kernel 的 Python DSL：经过装饰的
Python 函数会转换成指令 IR，框架可以进一步将其拆分成 cube/vec 两条路径，
在内置模拟器中执行，或下沉为 custom-op 源码产物。

## Skill 入口

面向用户的 skill 入口是插件根目录的 `SKILL.md`，可复用的工作流位于 `agent/`。
在阅读归档的运行时/文档内容或运行示例之前，先按需恢复它们：

```bash
bash agent/scripts/init.sh
```

该脚本幂等，只恢复缺失的目录树（`easyasc/`、`doc/`、`doc_cn/`、
`agent/scripts/` 维护工具与 `agent/example/`）。

本插件采用“直接使用源码仓库”的交付方式，不是一个可安装的 Python package。
建议先在模拟器中验证，并确保插件根目录位于 `PYTHONPATH`；只有 kernel 与参考
实现对齐后，再进入依赖 CANN 的执行路径。

## 可以用它做什么

- 通过同一套 Python 表达方式编写纯 cube 或混合 cube/vec kernel
- 在内置模拟器中验证 tiling、尾块、同步和精度边界
- 生成可用于 CANNSIM 或 Ascend 真机的源码与运行时产物
- 从可运行的参考 kernel 中学习受支持的 DSL 模式

## 公开目标接口

| Import | 目标 profile | Worker 数量 | 目标特有能力 |
|---|---|---:|---|
| `easyasc.a2` | Ascend A2，默认 B3 | 20 cube / 40 vec | A2 vector API 与 A2 int4 契约 |
| `easyasc.a3` | Ascend 910C，`Ascend910_9362` | 20 cube / 40 vec | 面向 `ascend910_93` 编译的 A2/C220 API |
| `easyasc.a5` | Ascend 950 | 32 cube / 64 vec | `@vf`、`@simt`、寄存器 micro API 与 MX 格式 |
| `easyasc.a5pr` | Ascend 950PR | 28 cube / 56 vec | 使用 950PR profile 的 A5/C310 API |

`a2` 与 `a3` 共享 C220 编写 API，但选择不同的设备与构建 profile；A5 系列则是
并列的另一套架构接口。

> **目标隔离规则：**一个 Python 进程只能导入 `easyasc.a2`、`easyasc.a3`、
> `easyasc.a5`、`easyasc.a5pr` 中的一个。导入 facade 会选择进程级全局设备状态；受 Python
> 模块缓存影响，重新导入先前 facade 不会恢复旧 target。切换 target 时应启动
> 新进程。

## 快速开始

### 1. 还原归档 payload

```bash
bash agent/scripts/init.sh
```

运行时（`easyasc/`）、文档（`doc/`、`doc_cn/`）、维护工具与示例都封装在
`agent/assets/` 下的归档中，执行本步骤后才会出现在磁盘上。

### 2. 准备 Python 环境

如果仓库所在机器提供了 `torch210npu` conda 环境，优先使用它：

```bash
conda activate torch210npu
```

如果只需要新建一个模拟器环境，可以安装仓库依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果已经在使用 CANN / `torch-npu` 环境，应保持 `torch` 与 `torch-npu`
版本匹配，只补装缺少的依赖。`requirements.txt` 不负责安装 CANN 本身。

### 3. 把源码仓库加入 `PYTHONPATH`

在仓库根目录运行示例前，先执行：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

直接运行嵌套目录中的 Python 脚本时，Python 自动加入 import 路径的是脚本所在
目录，而不是仓库根目录，因此这里需要显式设置。

### 4. 运行最小模拟器示例

```bash
python agent/example/kernels/a5/matmul/matmul_float_mmad.py
```

成功运行后，输出末尾会出现类似结果：

```text
max_abs_diff=0.000000e+00
```

这个示例会定义一个 kernel，通过 `OpExec(..., simulator=True)` 执行，并将
输出与 `x @ y.t()` 比较。

## 最小 kernel 结构

完整可运行源码位于 `agent/example/kernels/a5/matmul/matmul_float_mmad.py`
（`init.sh` 还原后可见）：

```python
from easyasc.a5 import *


@kernel()
def matmul_float_mmad_kernel(x: GMTensor, y: GMTensor, z: GMTensor, M: Var, N: Var, K: Var):
    l1x = Tensor(DT.float, [M, K], Position.L1)
    l1y = Tensor(DT.float, [N, K], Position.L1)
    l0c = Tensor(DT.float, [M, N], Position.L0C)
    with auto_sync():
        l1x <<= x[:, :]
        l1y <<= y[:, :]
        matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
        z[:, :] <<= l0c
    return z
```

`GMTensor` 构成公开的 GM 输入输出契约，本地 `Tensor` 描述片上存储，`<<=`
则根据每一组源和目标发射相应的数据搬运或写回指令。

## 执行模式

| 目标 | `OpExec` 配置 | 额外要求 |
|---|---|---|
| 开发与调试 | `simulator=True` | 只需要 Python 依赖 |
| 查看生成产物 | `simulator=False, gen_only=True` | 所选生成路径需要的依赖 |
| 使用 CANNSIM | `simulator=False, cannsim=True` | 兼容的 CANN 安装 |
| 构建并在真机运行 | `simulator=False` | 兼容的 CANN 安装与 Ascend 设备 |

`OpExec` 默认使用 `simulator=False`，因此在编写阶段应显式传入
`simulator=True`。生成目录布局、环境变量、CANNSIM chipset、构建/运行脚本
以及日志位置统一参见 `doc_cn/06_codegen_and_runtime.md`。

## 推荐开发流程

1. 写出精确的 PyTorch 参考公式，包括 cast 顺序。
2. 选择目标 facade 和流水线拓扑。
3. 实现 kernel，并使用 `simulator=True` 验证。
4. 覆盖尾块 shape；标量推导有歧义时显式提供 `shape_bindings`。
5. 检查生成产物，再进入 CANNSIM 或真机执行。

## 文档与示例入口

下表中的 `doc_cn/` 文档与示例目录均封装在归档 payload 内，阅读前请先执行
`bash agent/scripts/init.sh` 解包还原；未还原的全新 checkout 中这些路径不存在。

| 需求 | 建议入口 |
|---|---|
| 完成第一次运行 | `doc_cn/01_quickstart.md` |
| 理解概念和 kernel 语法 | `doc_cn/02_programming_model.md` 与 `doc_cn/03_write_your_first_kernel.md` |
| 查看完整文档地图 | `doc_cn/index.md` |
| 查询公开 API | `doc_cn/api/index.md` |
| 查看 feature 与 dtype 契约 | `doc_cn/topics/index.md` |
| 选择可运行 kernel | `agent/example/kernels/README.md` |
| 理解模拟器与生成行为 | `doc_cn/05_simulator_and_trace.md` 及 `doc_cn/06_codegen_and_runtime.md` |
| 排查常见问题 | `doc_cn/10_troubleshooting.md` |

## 仓库结构

- `easyasc/`：公开 facade、parser/codegen、模拟器与运行时
- `agent/example/kernels/`：按目标组织的精选单 kernel 参考实现
- `agent/example/projects/`：组合多个 kernel 的多文件系统工程
- `agent/example/demo/`：不属于 kernel catalog 的端到端框架示例
- `agent/example/testcases/`：parser、模拟器、codegen 与工具回归测试
- `doc/`：canonical 英文文档
- `doc_cn/`：中文文档
- `agent/`：面向 AI / agent 贡献者的 router-first 指引

其中 `easyasc/`、`doc/`、`doc_cn/`、`agent/example/` 与 `agent/scripts/*.py`
工具都通过 `agent/assets/` 下的两个归档交付，`init.sh` 还原后才存在；
`agent/` 指引文档为仓内明文。

修改框架的贡献者还应阅读 `doc_cn/11_architecture_for_contributors.md`
与 `agent/example/testcases/README.md`。
