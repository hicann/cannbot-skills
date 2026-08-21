---
type: CATLASS DSL Debugging Guide
title: AICError 异常定位与 msDebug Core dump 解析
description: 采集 AICError 现场、加载异常算子 Core dump，并从异常核、调用栈、寄存器和内存逐层定位到 CATLASS DSL 代码。
tags: [catlass-dsl, debug, aicerror, msdebug, core-dump]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-07-28T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-07-28T00:00:00Z'}
sources:
  - id: msdebug
    resource: https://gitcode.com/Ascend/msdebug/blob/77f50d2388c58b3b73279da604fc953ebb21676b/docs/zh/user_guide/msdebug_user_guide.md
    title: MindStudio Debugger 使用指南
operator_families: [elementwise, matmul]
arch: [c310]
---

# 接口与概念

AICError 是 AI Core 执行期异常。Host 侧 launch 通常是异步的，因此报错的
`synchronize` 或后续 API 不一定是根因所在；先把同步边界收窄到单次 kernel
调用，再采集异常现场。msDebug 能加载异常算子 Core dump，查看异常核、tensor
和参数内存、寄存器及调用栈。[^msdebug]

定位链固定为：

```text
Host 首个失败同步点
  -> 异常 .core 与同次构建的 kernel.o/fatbin
  -> ascend info summary 选择异常 AIC/AIV 核
  -> bt / PC 映射到 kernel 代码
  -> ARGS、TILING_DATA、INPUT/OUTPUT/WORKSPACE 地址与长度
  -> CATLASS DSL 的 tile、layout、extent、同步或搬运参数
```

# 用法

## 1. 收窄复现边界

在每个候选 kernel 后立即同步，记录首个失败的 kernel、shape、dtype、layout、
block_dim、device、CATLASS/CANN 版本和编译产物路径：

```python
executor(tla_x, tla_z, block_dim=block_count)
try:
    torch.npu.synchronize()
except RuntimeError as error:
    raise RuntimeError(
        "AICError candidate: kernel=vadd "
        f"shape={tuple(dev_x.shape)} dtype={dev_x.dtype} "
        f"block_dim={block_count}"
    ) from error
```

这一步只定位“哪个 launch 首先在同步点失败”，不能单凭 Host 栈把错误归因到
某条 DSL 语句。

## 2. 开启异常 Core dump

对支持环境变量配置的运行方式，在启动复现进程前设置：

```bash
mkdir -p ./artifacts/aicerror
export ASCEND_DUMP_SCENE=aic_err_detail_dump
export ASCEND_DUMP_PATH=./artifacts/aicerror
```

`aic_err_detail_dump` 会导出 AI Core 的内存、寄存器和调用栈现场；
`ASCEND_DUMP_PATH` 可使用绝对路径或相对路径。[^msdebug]

若运行入口通过 `aclInit` 加载配置，则在实际被加载的 `acl.json` 中配置：

```json
{
  "dump": {
    "dump_scene": "aic_err_detail_dump",
    "dump_path": "./artifacts/aicerror"
  }
}
```

单算子 API 场景应确认 `aclInit` 收到该文件；PyTorch 场景应确认修改的是当前
`torch_npu` 安装实际使用的 `acl.json`。环境变量和 `acl.json` 选择一种经过当前
CANN 版本验证的入口，避免把“没有生成 Core”误判为算子没有异常。[^msdebug]

## 3. 保存配套产物并加载 Core

保留同一次构建的 `.core`、`kernel.o` 或 fatbin、Host 日志和运行参数。若要查看
可靠的源码调用栈，使用工具链支持的 `-O2`/`-O3` 与 `-g` 生成含调试信息的
kernel 对象或 fatbin，然后加载：

```bash
msdebug --core ./artifacts/aicerror/operator.core ./build/operator.fatbin
```

也可先进入 msDebug 再加载：

```text
(msdebug) target create "./build/operator.fatbin" --core "./artifacts/aicerror/operator.core"
```

只分析现场摘要时可省略调试对象；需要源码栈时必须使用与 Core 对应的原始构建
产物，不能用修复后或重新编译的文件替换。[^msdebug]

## 4. 从异常核定位到现场

```text
(msdebug) ascend info summary
(msdebug) ascend aiv 34
(msdebug) bt
(msdebug) register read $PC
(msdebug) register read -a
(msdebug) x -m GM -f uint8_t[] 0x12c041200000 -s 16 -c 4
(msdebug) x -m DCACHE -f uint8_t[] 0x12c140110000 -s 16 -c 4
```

`ascend info summary` 先给出 CoreId/CoreType/PC，以及
`DEVICE_KERNEL_OBJECT`、`STACK`、`ARGS`、`TILING_DATA`、输入、输出和 workspace
的地址与大小；星号表示当前聚焦核。根据 CoreType 使用 `ascend aic <id>` 或
`ascend aiv <id>` 切换到异常核，再查看调用栈、寄存器和摘要中列出的有效内存。
地址、核号和读取长度必须替换为当前 Core 的实际值。[^msdebug]

# 代码模式

## 地址范围核对

从 `ascend info summary` 复制 tensor 基址和 Size，并用实际 dtype、shape、
stride 计算访问范围：

```python
def byte_range(base, coord, strides, itemsize, vector_width=1):
    element_offset = sum(index * stride for index, stride in zip(coord, strides))
    first = base + element_offset * itemsize
    last = first + vector_width * itemsize
    return first, last  # 半开区间 [first, last)


first, last = byte_range(
    base=0x12C041200000,
    coord=(block_m * tile_m, block_n * tile_n),
    strides=(logical_n, 1),
    itemsize=2,
    vector_width=32,
)
assert allocation_base <= first < last <= allocation_base + allocation_size
```

边界 tile 必须使用有效 extent，而不是无条件沿用完整 tile 的搬运长度。若 PC
落在 copy/load/store 附近，依次核对 GM 基址、layout/stride、block 坐标、
尾块长度、recast 后元素大小和 workspace 容量。

## 现场到 DSL 的归因顺序

```text
MTE/copy 类 stop reason
  -> 地址区间、对齐、搬运长度、尾块
Vector/Cube 计算类 stop reason
  -> dtype、mask、shape、输入是否初始化
同步或后续核失败
  -> set/wait 配对、pipe barrier、跨核 flag 生命周期
ARGS/TILING_DATA 与预期不符
  -> Host extent、block_dim、动态 shape 参数打包
```

一次只修改一个候选根因；修复后先跑原始最小复现，再跑单 tile、非对称 shape、
边界 tile 和完整 workload。

# 约束

- Core、kernel 调试对象、Host 日志和运行参数必须来自同一次复现与同一次构建。
- 硬件在指令触发异常后可能继续执行若干指令再上报；Core 中部分内存和寄存器
  可能已变化。msDebug 通常会修正 PC，但仍应结合 stop reason、调用栈和地址范围
  交叉验证。[^msdebug]
- 查看调用栈时优先使用 `-O2`/`-O3` + `-g`。`-O0` 强制 no-inline 时栈内存可能
  不可靠，通常只能信任第 0 帧。[^msdebug]
- 只读取 `ascend info summary` 标记为有效的地址；`(invalid)` 地址不能作为
  正常 tensor 内容解读。
- Core 可能包含输入、输出、workspace、参数和代码信息，应限制文件权限和传播，
  调试结束后按项目数据策略处置。
- `aic_err_detail_dump` 用于异常现场分析，不替代 oracle、msSanitizer 或同步
  竞争检查；修复必须通过原始 correctness 测试验证。

# 失败表现

| 表现 | 首查 |
| --- | --- |
| AICError 出现在后续 API | 在每次候选 launch 后立即 `torch.npu.synchronize()` |
| 没有生成 `.core` | dump 配置是否被当前进程加载、路径权限、CANN/驱动支持 |
| `target create` 无法加载 | Core 与 kernel 对象格式、路径、架构和构建是否匹配 |
| `bt` 只有地址或无源码行 | 是否提供同构建的 `kernel.o`/fatbin，是否含 `-g` |
| `bt` 在 `-O0` 下出现异常栈帧 | 改用 `-O2`/`-O3` + `-g` 复现，只信任第 0 帧 |
| 多个核都有现场 | 用 summary 的 PC/stop reason 逐核切换，不只看默认聚焦核 |
| tensor 地址显示 `(invalid)` | 不读取该地址，转查有效副本、参数和 PC |
| PC 位于 copy/load/store | 核对 allocation 半开区间、尾块、stride、dtype 字节数 |
| 修复后不再崩溃但结果错误 | 恢复 oracle，覆盖边界 tile、非对称 shape 和完整 workload |

# 验证方法

静态验证要求：

```bash
python3 skills/catlass-dsl-knowledge/scripts/record_knowledge.py validate \
  --project-root .
python3 skills/catlass-dsl-knowledge/scripts/record_knowledge.py query \
  --project-root . --text aicerror
```

设备验证要求保存：首个失败同步点、生成的 Core、匹配的 kernel 调试对象、
`ascend info summary`、异常核的 `bt`/PC、实际检查过的地址范围，以及修复前后
同一最小复现的结果。本文核对了固定提交的 msDebug 流程，未在 NPU 上生成或解析
Core。

[^msdebug]: 固定提交的 msDebug 用户指南中“解析异常算子 dump 文件功能介绍”及相关命令说明。
