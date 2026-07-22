# msprof op simulator 采集与解析

> 通用说明：本文档描述 `msprof op simulator` 的采集流程与两表（`instr_exe.csv` + `code_exe.csv`）联合解析方法，适用于任意 Triton-Ascend 算子。不绑定具体算子/shape。

## 环境准备

> ⚠️ **CANN 路径不要硬编码版本号**（不一定是 `cann-8.5.0`）。优先用环境变量 `ASCEND_HOME_PATH`，否则自动探测 `/usr/local/Ascend/cann-*` 下唯一/最新目录。

```bash
# SOC_VERSION 按目标芯片替换，如 Ascend910B1 / Ascend910B2 / Ascend910B3 / Ascend910B4 等
export SOC_VERSION=Ascend910B1

# CANN 安装根：优先 $ASCEND_HOME_PATH，否则自动探测 /usr/local/Ascend/cann-*
CANN_HOME="${ASCEND_HOME_PATH:-$(ls -d /usr/local/Ascend/cann-* 2>/dev/null | sort -V | tail -1)}"
[ -z "$CANN_HOME" ] && { echo "未找到 CANN 安装"; return 1; }
source "$CANN_HOME/set_env.sh"
export LD_LIBRARY_PATH="$CANN_HOME/aarch64-linux/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH"

# Triton 侧：simulator 采集必须开启，确保 kernel 强制重新编译且携带行号信息便于指令级定位
export TRITON_DEBUG=1
export TRITON_ALLWAYS_COMPILE=1
export TRITON_DISABLE_LINE_INFO=0
```

要点：
- `CANN_HOME` 决定实际版本路径（如 `cann-8.5.0` / `cann-9.1.0` 等），不要写死。
- simulator lib 在 `$CANN_HOME/aarch64-linux/simulator/${SOC_VERSION}/lib`（库文件名为 `libruntime_camodel.so` / `libpem_davinci.so` 等，"simulator" 在**目录路径**而非文件名，故 `find -name '*simulator*.so'` 找不到——须按目录定位）。
- `${SOC_VERSION}` 须与 `--soc-version` 保持一致。

## 采集命令

```bash
msprof op simulator \
  --application="python prof_small.py" \
  --output=prof_out \
  --kernel-name=<kernel 函数名> \
  --launch-count=1 \
  --soc-version=${SOC_VERSION} \
  --timeout=60
```

关键参数：

- `--kernel-name=<func>`：**只采集指定 kernel 的指令级统计**（过滤辅助 kernel 如 `ZerosLike`/`Empty`/`OnesLike`，否则 `launch-count=1` 会采到先启动的辅助 kernel）。编译后 kernel 名可能带后缀（如 `_mix_aic`），`--kernel-name` 用源码函数名做**子串匹配**即可。
- `--launch-count=N`：采集 N 个目标 kernel 实例。诊断用 1 即可。
- `--soc-version=${SOC_VERSION}`：指定仿真芯片，需与 `LD_LIBRARY_PATH` 中的 `${SOC_VERSION}` 一致（如 `Ascend910B1`、`Ascend910B2` 等，按目标 arch 填写）。
- `--aic-metrics=...`：**可选，非必填**。不传时默认即产出 `instr_exe.csv`（per-instruction pipe/cycles）与 `code_exe.csv`（per-source-line cycles）两张核心表，足以完成"瓶颈类型 + 位置"联合定位；仅在需要更细的 pipe-utilization 汇总时按需添加（如 `PipeUtilization`、`MemoryUB`）。**禁止当成必填前提**——加与不加不影响两张核心表的有无，徒增参数面。

⚠️ **simulator 不支持 `--launch-skip-before-match`**（那是 board 模式参数）。要跳过辅助 kernel，只能用 `--kernel-name`。

### ⚠️ 关键行为认知（避免误判，必读）

1. **整应用都被仿真，不止 `--kernel-name` 指定的 kernel**：`--kernel-name` 只过滤**采集**哪个 kernel 的统计，**不限制执行**——脚本里 `forward()` 触发的**所有** kernel 都在 simulator 里逐指令执行。因此总仿真时间取决于**整个应用**（所有 kernel × 各自 shape），而非仅目标 kernel。即使只采集其中一个 kernel，若脚本含多个重 kernel 或 shape 偏大，仍会很慢/超时。对策：脚本只跑**单次 forward**（无 warmup 循环）、shape 取**最小可复现瓶颈**（每维 ~64 起，最多 128），**不要**因为"只采一个 kernel"就放心用大 shape。

2. **多 kernel 算子须逐个采集（硬门禁）**：完整门禁流程（AST 枚举全部 `@triton.jit` kernel → 逐个 `--kernel-name` 采集 → 产出覆盖表）见 SKILL.md「全 kernel 采集覆盖门禁」，此处不重复。**关键原因**：不同 kernel 瓶颈类型可能不同（一个 Cube 空等、另一个访存 bound、第三个标量降级），只采第一个就下全局结论会漏。

3. **子进程退出码非 0 不一定失败**：目标 kernel 跑完（脚本打印 `DONE`）后，profiler 拆解阶段偶发崩溃（典型如退出码 134 / `std::bad_weak_ptr`），日志出现 `Running task failed, data parsing start`——但 `OPPROF_*/.../*.csv` **仍已落盘**。**以两张 CSV 是否存在且非空为成功判据**，不要因子进程非 0 退出就重跑或放弃。

## 用例脚本要求

```python
import importlib.util, torch
spec = importlib.util.spec_from_file_location("impl", "<代码路径>")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import torch_npu  # noqa
dev = torch.device("npu")
# 小 shape! simulator 对大 shape + atomic_add 会卡死/极慢
# 按算子实际入参构造一组"最小可复现瓶颈"的 shape（每维 ~64 起步，最多 128 量级）
torch.manual_seed(0)
inputs = [ ... ]   # 用上述小 shape 构造 forward 所需的全部张量/标量入参
mdl = m.ModelNew(<init>).to(dev)
out = mdl(*inputs)          # 单次调用（无 warmup 循环），减少仿真工作量
torch.npu.synchronize()
print("DONE")
```

要点：
- **shape 要小**（每维 ~64，最多 128 量级）。simulator 逐指令仿真，shape 大 + 含 `atomic_add`（跨核竞争）会卡死（实测 reduce 维 128 + atomic_add 20min+ 无产出，64 几秒完成）。
- **单次调用**（无 warmup 循环），减少仿真工作量。注意"单次 forward"仍会触发算子的**所有** kernel（见上文"整应用都被仿真"）。
- 若 kernel 用 `torch.zeros_like`（atomic 累积的输出需零初始化），会产生 `ZerosLike` kernel launch，必须用 `--kernel-name` 跳过。

## 产出结构

```
prof_out/OPPROF_<timestamp>_<id>/
└── simulator/
    ├── core0.cubecore0/        # Cube 核（每核一个子目录）
    │   ├── core0.cubecore0_instr_exe.csv   ← per-instruction 统计（定位"哪种指令热"）
    │   ├── core0.cubecore0_code_exe.csv    ← per-source-line 统计（定位"哪一行热"）
    │   └── trace.json
    ├── core0.veccore0/         # Vector 核 0
    ├── core0.veccore1/         # Vector 核 1
    └── ... core1..coreN
```

> **路径定位用 `find`，不要硬编码层级**：不同 msprof 版本/参数下，CSV 可能在 `OPPROF_*/simulator/coreN.*/` 下，也可能多一层（如 `device0/` 或 `<kernel_name>/0/`）。统一用：
> ```bash
> find prof_out -name "*_instr_exe.csv" -o -name "*_code_exe.csv"
> ```
> 拿到实际路径后再解析，避免路径假设错误。

910B 系列每个物理核含 1 cubecore + 2 veccore（Cube:Vec=1:2，不同代际核结构可能不同，以实际芯片为准）。

> **⚠️ 必须看 4 张表，不止 Cube 的 2 张**：每个核目录下都有 `*_instr_exe.csv` + `*_code_exe.csv` 两张表。诊断时**Cube 核（`core0.cubecore0/`）和 Vector 核（`core0.veccore0/`）各自的两张表都要解析**，共 4 张 CSV：
> - **Cube 两表**：看 dot（MMAD）是否真热、Cube 是否在 `WAIT_FLAG` 空等 Vector（Cube↔Vector 串行铁证）。
> - **Vector 两表**：看 Vector 在忙什么（MTE2 load / MTE3 store / VECTOR 逐元素 / SCALAR 标量降级），以及这些 cycles 落在 `.py` 哪一行（定位是哪条 `tl.load`/`tl.store`/`tl.exp` 热）。
>
> 两核证据**互补**：Cube 显示"在等"，Vector 显示"在忙什么"——缺了 Vector 两表就不知道 Cube 等的到底是什么、该优化哪条 Vector 侧指令。禁止只看 Cube 两表就下结论。
>
> 实操：对 `core0.cubecore0` 和 `core0.veccore0` 各跑一遍下文"表 1 + 表 2"的解析（路径换成对应核目录下的 CSV）。

## CSV 解析：两表联合定位（类型 + 位置）

> 核心方法论：**`instr_exe.csv` 回答"哪种指令/pipe 在烧 cycles"（瓶颈类型）；`code_exe.csv` 回答"这段 cycles 花在 kernel 哪一行"（瓶颈位置）。两者配合 = 类型 + 位置 = 精准定位。**

### 表 1：`instr_exe.csv`（瓶颈类型）

字段：`instr, addr, pipe, call_count, cycles, running_time(us), detail`

```python
import csv, os
# path 由 find 定位，按核区分：Cube 用 .../core0.cubecore0/core0.cubecore0_instr_exe.csv，
# Vector 用 .../core0.veccore0/core0.veccore0_instr_exe.csv（两个核各解析一遍）
rows = list(csv.DictReader(open(path)))
rows.sort(key=lambda r: int(r['cycles']), reverse=True)
total = sum(int(r['cycles']) for r in rows)
# 按 pipe 汇总占比
pipe = {}
for r in rows:
    pipe[r['pipe']] = pipe.get(r['pipe'], 0) + int(r['cycles'])
for p, c in sorted(pipe.items(), key=lambda x: -x[1]):
    print(f"{p}: {c/total*100:.1f}%")
# top 指令
for r in rows[:10]:
    print(r['instr'], r['pipe'], r['call_count'], r['cycles'])
```

关键指标：
- `WAIT_FLAG_DEVI`（FLOWCTRL pipe）：核空等（等另一个 pipe / 核的 flag）
- `MMAD`（CUBE pipe）：Cube 矩阵乘加（真正的 dot 算力）
- `CMPN`/`ADD`/`LD`/`ST`（SCALAR pipe）：标量降级的比较/算术（calls = 元素数 → 标量降级）
- `VEXP`/`VCMAX`/`VCADD`（VECTOR pipe）：向量化计算
- `MTE2`/`MTE3`：GM↔UB 搬运（load/store）
- `BAR`：barrier 同步

### 表 2：`code_exe.csv`（瓶颈位置）

字段：`code, call_count, cycles, running_time(us)`

- `code`：`<源文件绝对路径>:<行号>`，指向 binary 调试信息引用的源文件。开启 `TRITON_DISABLE_LINE_INFO=0` 时，`code` 通常**直接指向 `.py` 源码行**（line info 透传），少数情况指向编译产物 `kernel_meta/<kernel>_kernel.cpp`。
- `call_count`：该源码行的硬件执行次数（循环内行 = 循环展开后命中次数）。**读法**：若多行 `call_count` 几乎相等，说明它们是同一循环体内的相邻语句，循环体跑了该次数——判断"热点是循环内某条语句"的直接证据。
- `cycles`：该行所有指令的 cycles 总和（按行聚合）。

```python
import csv, re, os
def parse_code_exe(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r['call_count'] = int(r['call_count'])
        r['cycles'] = int(r['cycles'])
    rows.sort(key=lambda r: r['cycles'], reverse=True)
    total = sum(r['cycles'] for r in rows)
    return rows, total

rows, total = parse_code_exe(path)  # .../core0.cubecore0_code_exe.csv
print(f"total = {total} cycles, top source lines:")
for r in rows[:10]:
    m = re.match(r'^(.*):(\d+)$', r['code'])
    ln = m.group(2) if m else '?'
    fn = os.path.basename(m.group(1)) if m else r['code']
    print(f"  L{ln}  {r['cycles']} cyc ({r['cycles']/total*100:.1f}%)  calls={r['call_count']}  {fn}")
```

### 从 code 行 → 瓶颈类型 → 修复技术

拿到 top 行号后，**打开 `code` 列指向的源文件该行及上下文 ±5 行**，读出该行对应的硬件语义。下表覆盖 910B Triton-Ascend 编译产物中的高频硬件构造：

| 源文件该行出现的构造 | 所在 pipe | code_exe 热点含义 | 命中诊断规则 | 修复方向（latency-optimizer 优化点） |
|---|---|---|---|---|
| `tds_2dsync` / `td::load` / `__global_to_local` | MTE2 | GM→L1/UB 搬运热 | 规则 3（访存 bound） | 7（Pass 合并）/ 21（物化解耦）/ 10（循环不变外提）/ 增大 tile |
| `td::store` / `__local_to_global` | MTE3 | UB→GM 写回热 | 规则 3 | 7 / 21 / 减少中间写回 |
| `mmad` / `MMAD` intrinsic | CUBE(MMAD) | 真矩阵乘算力热 | 规则 4（真计算 bound） | 无现成优化点（硬件极限判据）→ 增大 tile / bf16 化（精度允许时），均不可行回 4.6 |
| `fixp_l0c2ub` / L0C→UB move | CUBE→Vector | dot 结果搬给 Vector 侧逐元素处理 | 规则 1（Cube↔Vector 串行） | 19（Cube/MTE3 解耦）/ 21（物化解耦） |
| `set_flag` / `wait_flag` / `WAIT_FLAG_DEVI` | FLOWCTRL | 核/pipe 间空等 | 规则 1 | 19（消除依赖链）/ 减少循环内跨 pipe 依赖 |
| `vadds`/`vmuls`/`vexp`/`vsub` calls ≈ tile 元素数 | VECTOR | 向量化逐元素（正常） | — | 若占比过高考虑 10（循环不变量外提） |
| `cmpn`/`add`(i32) calls ≈ tile 元素数 | SCALAR | i32 比较标量降级 | 规则 2 | 6（避免标量降级）/ 5（Scalar→Vector）/ 17（冗余边界） |
| `atomic_add` / `atomic_xchg` | — | 跨核原子写热 | 多为优化点 19 副作用 | 评估写冲突密度，冲突大则退回分 pass |
| `bar` / `syncall` | BAR | 跨核/跨 pipe barrier | 规则 5 | 19（减少循环内依赖链） |

**关键判据**：`fixp_l0c2ub` + `wait_flag` 同时出现在 top 行 = Cube 把 dot 结果搬到 UB 等 Vector 处理 = 规则 1 的铁证。比单看 `instr_exe.csv` 的 `WAIT_FLAG_DEVI` 占比更精确——你能直接看到**哪一行 dot 的结果在等 Vector**，从而知道该拆哪个 pass。

**另一种典型 pattern**：top 行是某个**循环头**（`for ... in range(...)`）且 `instr_exe` 显示 `WAIT_FLAG_DEVI` 高占比 → Cube 在循环每次迭代的边界空等（通常是等上一迭代 Vector 侧 load/store/exp 完成）。此时瓶颈不在循环头本身，而在循环体内 per-iteration 的跨 pipe 依赖链；拆解/物化该依赖链即可。

### code_exe ↔ instr_exe 联合定位流程

> 对 **Cube 核（`core0.cubecore0`）和 Vector 核（`core0.veccore0`）各做一遍**（共 4 张表），再两核交叉验证。

1. 解析 `instr_exe.csv`（Cube+Vector 各一份）→ pipe 占比 → 规则 N（瓶颈类型）。
2. 解析 `code_exe.csv`（Cube+Vector 各一份）→ top-N 热源码行 → 打开 ±5 行读构造。
3. 用上表把"构造"映射到"规则/技术"，与 step 1 交叉验证：
   - 一致 → 高置信，直接应用技术。
   - 不一致 → 以 `code_exe` 位置为准重新判断（`instr_exe` 占比可能被某条超长 cycles 指令拉偏），回 bottleneck-diagnosis.md 复核。
   - **两核交叉**：Cube `WAIT_FLAG` 高 + Vector 某 pipe（MTE2/MTE3/VECTOR）高 → Cube 等的就是 Vector 那条 busy 链，优化 Vector 侧该指令即解 Cube 空等。
4. 应用技术 → 重采 → 看 top 热行是否转移 / cycles 是否下降（两核都看）。

### 回溯到可编辑源码

`code` 列指向的源文件能否直接编辑，取决于其形态：

**情形 A：`code` 指向 Triton 编译产物（`kernel_meta/*_kernel.cpp`）**
- ⚠️ **禁止直接改 `kernel_meta/*_kernel.cpp`**——它是编译产物，每次重编覆盖，改了也白改。
- 回溯路径：
  1. 在该热点行附近读出变量名 / 数据流（如 `buf_q` 指向的 tile、`fixp_l0c2ub` 搬的是哪个 dot 的累加器）。
  2. 对照 Triton 源 `.py`：该 dot / load / where 对应 `@triton.jit` kernel 里的哪条 `tl.dot` / `tl.load` / `tl.where`。
  3. 修改对应 Triton 源码行，重新编译触发 `kernel_meta` 重新生成，再采集。

**情形 B：`code` 直接指向 `.py`（line info 透传，常见）**
- 直接改对应 `tl.*` 语句即可，无需经过 `kernel_meta`。

判定方法：看 `code` 列路径后缀（`.cpp` vs `.py`）与所在目录（`kernel_meta/` 下即编译产物），据此选 A/B 分支。

若 `code` 列指向的文件不存在或路径残缺（simulator 偶发 line info 丢失），回退到 `instr_exe.csv` 的 `addr` 列做指令级定位。

## 仿真卡死/超时排查

- **症状**：`msprof.log` 停在 "Extract N relations from kernel" 不动，python 子进程 CPU 飙高（1h+），只产出 `dump/` 无 `simulator/`。
- **原因 1**：kernel 含 `atomic_add`（跨核写竞争），simulator 串行化仿真极慢。
- **原因 2**：shape 过大，或脚本含多个重 kernel（整应用都被仿真，见上文）。
- **处理**：杀进程；shape 减半（如 S=128→64）；确认脚本只跑单次 forward、只构造最小 shape。若仍卡死，改用 board profiler（`msprof op`，真实 NPU，快但只给 kernel 级时间无指令级）。

## board 模式（备选，快但粗）

```bash
msprof op --application="python script.py" --output=out --aic-metrics=PipeUtilization,MemoryUB
```

board 模式跑真实 NPU，秒级完成，给 kernel 总时间 + pipe 概览，但**无 per-instruction CSV**。用于端到端确认；指令级诊断仍需 simulator。
