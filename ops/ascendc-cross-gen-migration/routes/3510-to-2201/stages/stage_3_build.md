# 阶段 3：编译与安装

**目标**：在目标 CANN 上产出可安装、可被调用的算子包，并用最小用例证明"能跑"。

## 必读

- `[LOADED] build-and-config`
- `[LOADED] pitfalls`


## Step 3.0 构建前检查磁盘/配额

```bash
quota -s 2>/dev/null | tail -3   # 有配额限制时先看是否接近 hard limit
df -h /home /tmp | tail -2
```

- 用户配额满时编译会报 `Disk quota exceeded` / `No space left on device`，**不是代码错误**；
- 清理已完成任务的 `work/build`、旧 `eval_*` 安装目录，或把 `TMPDIR` 指到本地盘（`/tmp`）；
- 不要删其他人的工作区；只清自己创建的任务目录。

## Step 3.1 清理

```bash
source "${CANN_SET_ENV}"
cd "${OP_REPO}"
rm -rf ./build        # 仅在首次构建、切换 soc/arch 或怀疑缓存污染时执行
```

首次构建或切换 `--soc`/arch 时必须清 `build/`（arch35 的缓存与生成脚本会污染 arch22 构建）。

**增量构建**：同一 soc 下的第二轮及以后不要删 `build/`。大算子（注意力反向类）全量构建可达 1–2 小时，增量通常几分钟；只改文档/README 时不要触发重编，改 kernel/host 时让构建工具自行增量。

**构建预算**：S 类任务（单点注册/单内核）构建 ≤3 次，M 类任务 ≤6 次。超过预算仍未通过时，先停下来读日志定位，不要用 `rm -rf build` 全量重建来掩盖问题；只有切换 soc/arch 或已证明缓存损坏时才清理。**批量改完再构建**：一次改完一组相关文件后构建一次，不要“改一行、重建一次”。

## Step 3.2 单算子编译

```bash
bash build.sh --pkg --ops="${OP_NAME}" --soc="${SOC_BUILD}" 2>&1 | tee "build_${SOC_BUILD}.log"
echo "exit=$?"
```

- 先只编一个算子，错误信息聚焦。
- 失败先查 `references/pitfalls.md` §1，再查 `ascendc-runtime-debug`。
- 记录：编译耗时、警告（尤其 dtype/宏相关）、产物路径。


## Step 3.2b 避免连带重编另一 soc

- 改**共享文件**（如 `scatter_pa_kv_cache` 的 tiling/def、公共头文件）时，构建系统可能把另一 soc（ascend950/arch35）的整套 kernel 模板也重编一遍，耗时可达 10–30 分钟；
- 优先把改动限制在目标 soc 的专属文件（`op_kernel/arch22/`、A2 专用 tiling/def），避免触碰共享 tiling；
- 构建开始后先看日志里在编哪个 soc：若出现大量 `--soc_version=Ascend950...` 的 `opc.py` 任务，说明触发了 950 重编，应评估是否值得等待，必要时改回最小改动面；
- 时间盒内如果被 950 重编拖住，保留已生成的 A2 包用于评测，不要在会话里反复等待全量 950 编译。



## Step 3.0 工具链自检（构建前 10 秒）

先编译一个空文件确认宿主工具链可用，再跑 `build.sh`：

```bash
echo 'int main(){return 0;}' > /tmp/cc_check.c && cc /tmp/cc_check.c -o /tmp/cc_check && echo CC_OK
echo 'int main(){return 0;}' > /tmp/cxx_check.cpp && c++ /tmp/cxx_check.cpp -o /tmp/cxx_check && echo CXX_OK
```

- 失败（如 `cannot find Scrt1.o`、`cstdint not found`）说明缺 libc-dev/头文件或 `CC/CXX` 未指到可用编译器；
- **先修工具链/环境变量，不要反复重跑 build.sh**：这类错误每次都会复现，重跑只会浪费时间。

## Step 3.2a 固定构建命令与环境契约

每次构建都用同一套命令，避免手工在 `gen/` 目录调 `opc.py`/`bisheng`：

```bash
source /usr/local/Ascend/cann-<ver>/set_env.sh
cd "${OP_REPO}"                      # 必须在仓库根执行
export CPLUS_INCLUDE_PATH="/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11"
export C_INCLUDE_PATH="/usr/include:/usr/include/aarch64-linux-gnu"
bash build.sh --pkg --soc=ascend910b --ops=<only-target-op> -j8 > /tmp/build.log 2>&1
```

- 只编目标算子：不要顺手把共享算子（如 `scatter_pa_kv_cache`）一起编，产物与配额都会翻倍；
- **并行度按 CPU/内存压力自适应（-j8 ~ -j32）**：构建前先看 `nproc`、`free -g`、`uptime`；
  - 内存充足（free ≥ 32 G）、load 低、无其他构建：可用 `-j16` ~ `-j32`；
  - 内存一般（free 8–32 G）或有其他会话：`-j8`；
  - 内存紧张（free < 8 G）、同机已有重构建：`-j4` ~ `-j8`；
  - 一旦出现 `Error 137`（被 kill），立即降一档（如 32→16→8）再重试，不要原参数重跑；
- 同一主机不要同时跑两个重构建会话；
- `Error 137` = 进程被 kill（多为 OOM），先看内存/并发，不要 clean 重试；
- `cstdint not found` / `libplatform.so: cannot open` = 没 source set_env 或没带 include/lib 路径，不是代码错误。


## Step 3.2c A2 注册三件套（缺一不可）

用 `--soc=ascend910b` 能编译出 A2 二进制，前提是仓库里存在 A2 的注册文件：

1. `op_host/op_tiling/arch22/<op>_arch22_tiling.cpp`：`REGISTER_TILING_TEMPLATE_WITH_ARCH(<Op>, <TilingClass>, NpuArch::DAV_2201, <priority>)`；
2. `op_kernel/arch22/`（kernel 实现）+ 独立入口（若入口按 soc 分支）；
3. `op_host/<op>_def.cpp` 中 `AICore().AddConfig("ascend910b", ...)`。

**打包纪律**：最后一次构建必须是 `--soc=ascend910b`；如果随后又跑了一次 `--soc=ascend950`，`build_out/` 会被 950 包覆盖，安装后表现为“只有 ascend950、没有 ascend910b”，运行时报 `JSON configuration file ... cannot be found`。

**验收**：安装后检查 `op_impl/ai_core/tbe/kernel/ascend910b/<op>` 存在，再直调 `getws/run`。

## Step 3.3 安装

按仓约定二选一：

```bash
bash install.sh
# 或
./custom_opp_<platform>.run
```

安装后确认：

```bash
ls "${ASCEND_OPP_PATH}/vendors/" 2>/dev/null
echo "${ASCEND_CUSTOM_OPP_PATH:-}"       # 自定义算子路径
```

## Step 3.4 注册与接口验证

```bash
# aclnn 接口存在性（按仓产物）
find "${ASCEND_OPP_PATH}" -name "libcust_opapi.so" 2>/dev/null | head
# torch 绑定（如有）
python3 - <<'PY'
import torch, torch_npu
print(torch.ops.npu)
PY
```

确认 `torch.ops.npu.<op>`（或 aclnn 接口）可解析，不报 `AttributeError`。

## Step 3.5 最小用例冒烟

用一个最小 dtype + 最小 shape 跑通（不追求精度全过，但不许崩）：

- 无 `UB out-of-range` / L0C 溢出 / 非法地址
- 无超时/卡死
- 输出 buffer 被真实写入（非全 0/未初始化）

失败 → 回阶段 2 对应项，不要进入全量精度。

## Step 3.6 记录

- 构建日志路径、安装包路径、安装命令、注册验证输出。
- 最小冒烟用例（shape/dtype/命令/结果）。

## 完成判据

- `--soc="${SOC_BUILD}"` 编译成功，产物存在。
- 安装成功，接口可调用。
- 最小用例无崩溃、无越界。

## Gate

```
**前置条件（硬性）**：只有在 A2 真机（`NpuArch=2201` / `ascend910b` / 910B/910_93）上完成编译与安装，才允许输出 `BUILD=PASS INSTALL=PASS`。若当前机器是 A5（DAV_3510）或其他架构，只能记录：

```
[GATE-3] BUILD=BLOCKED REASON=NO_A2_DEVICE
```

**A5 机器不能替代 A2 验证**：A5 上编译通过、安装成功、跑通用例，都不构成 A2 的 `GATE-3`/`GATE-4`/`GATE-5` 证据。

[GATE-3] BUILD=PASS INSTALL=PASS
```

任一步失败：`BUILD=FAIL` / `INSTALL=FAIL`，附日志关键行，停止。

## 环境变量不跨 shell

`build.sh` 每次可能由新的非交互 shell 执行：`CPLUS_INCLUDE_PATH`、`CC/CXX`、`LD_LIBRARY_PATH` 不会自动继承。

- 把环境写进同一条命令：`source set_env.sh && export CPLUS_INCLUDE_PATH=... && bash build.sh ...`；
- 出现 `bits/c++config.h`、`cstdint` 找不到，先确认当前 shell 的 include 环境，而不是改代码。
