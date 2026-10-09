# 06 · 存量 Template：实时索引与命名约定

本页是查已有实现的可选工具。自定义算法先读 [算子契约](11-operator-contracts.md)，不要求存在同类蓝本。

存量完整清单由脚本生成；已核对的算子契约和接口摘要允许固化，并用来源指纹发现变化。
用 `scripts/list_templates.py` 实时扫描 `$HCCL_REPO/src/ops/*/algorithm/template/**`，从源码解析出每个 template 的特征。

> 两个路径变量见 `SKILL.md` §1.0：`$HCCL_REPO` 是调用方显式提供的 HCCL 仓根目录（脚本无内置默认值，
> 缺失时返回输入错误），`$SKILL_DIR` 是本 Skill 所在目录。脚本不在 HCCL 仓里，**一律用绝对路径调用**。

## 1. 挑蓝本：`--blueprint`

```bash
python3 "$SKILL_DIR/scripts/list_templates.py" --blueprint
```

按"意图 → 蓝本文件"输出，并实时校验首选文件是否还在仓内；不在会提示自行挑选。覆盖的意图：

最小可运行骨架（只同步不搬数据）· Mesh one-shot · Mesh two-shot · NHR 多 step ·
Gather 类（repeat/stride/对称内存/图模式）· Scatter 类 · ReduceScatter 类 ·
变长算子 `*_v` · AlltoAll 类 · DPU 展开 · 继承已有 template 的变体

决定复用已有实现时，阅读被复用的完整函数和必要调用方；仅作为可选学习材料时按问题取用。64-bit/PROD 完整处理参考
`ins_temp_all_reduce_aicpu_reduce_nhr`，多级参考 `ins_temp_all_reduce_nhr`。

> template 写完要接 selector 分支时：selector 侧阈值常量（如 `AR_AICPU_1D_64DATATYPE_DATA_SIZE`）
> 定义于 `src/ops/<op>/selector/<op>_auto_selector.cc`，新增分支前先读该文件已有分支。

## 2. 全量索引与筛选

```bash
python3 "$SKILL_DIR/scripts/list_templates.py"                  # 全部 AICPU template
python3 "$SKILL_DIR/scripts/list_templates.py" --op all_reduce  # 只看某个算子
python3 "$SKILL_DIR/scripts/list_templates.py" --topo nhr       # 只看 NHR 类（mesh 同理）
python3 "$SKILL_DIR/scripts/list_templates.py" --grep OneShot   # 类名/文件名模糊匹配
python3 "$SKILL_DIR/scripts/list_templates.py" --all-engines    # 连 CCU / AIV / V1 一起列
python3 "$SKILL_DIR/scripts/list_templates.py" --format md      # Markdown 表格
```

每行给出的、**都是从源码现场解析**的事实：

| 列 | 含义 | 解析自 |
|---|---|---|
| `LOC` | .cc 行数，粗略代表复杂度 | 文件 |
| `SCRATCH` | `CalcScratchMultiple()` 的返回表达式 | 函数体（回溯局部变量赋值） |
| `THREADS` | 线程总数表达式 | `GetThreadNum()`，否则 `CalcRes/GetRes` 里的 `slaveThreadNum` |
| `CHANNEL` | `CalcRes` 里用的 channel 申请函数 | `CalcChannelRequest*` 调用 |
| `CMAKE` | `H`=host `libhccl.so`，`D`=device `libscatter_aicpu_kernel.so` | 两处 CMake |
| 尾部标记 | `[变体←父类]` `[<algoType>]` `[strReg]` `[未被executor引用]` | .h / .cc / executor 扫描 |

**`CMAKE` 列不是 `HD` 就说明接线漏了**，参见 `05-build-and-verify.md` §1。

## 3. 单个 template 详情：`--detail`

```bash
python3 "$SKILL_DIR/scripts/list_templates.py" --detail InsTempAllReduceNHR
```

额外给出父类、`props.algoType`、是否字符串注册，以及**被哪些算法名 / 哪些 executor 引用**——
这是把 template 和 selector 吐出的算法名对上号的最快路径。脚本识别全部 5 种绑定宏：

`REGISTER_EXEC_V2` · `REGISTER_EXEC_V2_MULTI` · `REGISTER_EXECUTOR_BY_TWO_TEMPS` ·
`REGISTER_EXECUTOR_BY_FOUR_TEMPS` · `REGISTER_EXECUTOR_IMPL_NO_TOPOMATCH`

（`REGISTER_EXECUTOR_BY_TOPO` / `REGISTER_EXECUTOR_IMPL` 不带 template 参数，不参与绑定。）

## 4. 变体继承关系：`--variants`

```bash
python3 "$SKILL_DIR/scripts/list_templates.py" --variants
```

新算法若只是在已有 template 上改一部分，**继承那个 template 类，而不是继承 `InsAlgTemplateBase`**：

```cpp
class InsTempAllGatherOmniPipeMesh1D    : public InsTempAllGatherMesh1D  { ... };
class InsTempAllGatherMesh1D1DZAxisDetour : public InsTempAllGatherMesh1D { ... };
class InsTempReduceScatterOmniPipeNHR   : public InsTempReduceScatterNHR { ... };
```

只覆写真正变化的方法（通常是 `KernelRun` 或某个 `RunXxx`），`CalcRes` / `CalcScratchMultiple` 直接继承。
`--variants` 会列出当前所有这类关系，帮你判断该挂在哪个父类下。

---

## 5. 命名约定（这部分是稳定的，不随仓库变）

### 5.1 前缀

| 引擎 | 文件前缀 | 类前缀 | 基类 |
|---|---|---|---|
| AICPU / DPU / host_nic | `ins_temp_*` | `InsTemp*` | `InsAlgTemplateBase` |
| CCU | `ccu_temp_*` | `CcuTemp*` | `CcuAlgTemplateBase` |
| AIV | `aiv_temp_*` | `AivTemp*` | `AivAlgTemplateBase` |

> `src/ops/reduce/algorithm/template/aicpu/` 下有几个历史遗留文件不带 `ins_temp_` 前缀、
> 类名也没有 `InsTemp`（`ReduceMesh1D`、`ReduceNHR` 等）。**新文件不要跟这个风格。**

### 5.2 类名构词

`InsTemp` + 算子 + 拓扑 + 变体，例如
`InsTempAllReduceMesh1DOneShot`、`InsTempReduceScatterAicpuReduceNHRPcie`。

| 片段 | 含义 |
|---|---|
| `Mesh1D` / `mesh_1D` | level0 全连接拓扑 |
| `NHR` / `nhr` | Non-power-of-two Halving-Doubling，多 step 对数算法 |
| `OneShot` / `TwoShot` | 一步全交换 vs ReduceScatter+AllGather 两阶段 |
| `MeshChunk` | Mesh 内再按 chunk 二次切分 |
| `AicpuReduce` | 归约在 AICPU 上软件完成（64-bit / PROD） |
| `OmniPipe` | 多 step 流水化切分，见 `op_common/omnipipe_template_utils.h` |
| `Intra` / `Inter` | 多层算法的层内 / 层间构件，由 `*_BY_TWO_TEMPS` / `*_BY_FOUR_TEMPS` 组合 |
| `Dpu` | 任务下沉到 DPU 展开（**必须**写 `REGISTER_TEMPLATE_V2` 字符串注册） |
| `Pcie` | 走 PCIe 链路的读模式变体 |
| `MultiJetty` / `MultiLink` | 一个远端多条 channel |
| `ZAxisDetour` | Z 轴绕行拓扑变体 |
| `OrderPreserved` | 保序归约（batch-invariant） |
| `2Die` | 双 die 场景 |
| `Mem2Mem` | 内存到内存直搬（主要出现在 CCU 侧） |

### 5.3 文件名

类名转 snake_case，但 `1D` 保留大写 D（`ins_temp_all_reduce_mesh_1D_one_shot.cc`），
`NHR`/`DPU`/`UBX` 等缩写转小写。`scripts/new_template.py` 会自动推导，
**推导结果务必和同目录邻居对一眼**，不一致就用 `--file` 显式指定。

### 5.4 多层算法的组合方式

Broadcast = Scatter(intra) + AllGather(inter)，AllReduce 三层 = ReduceScatter(intra) +
ReduceScatter(inter) + AllGather(inter) + AllGather(intra)。
这些不是一个大 template，而是几个 `*Intra` / `*Inter` 小 template 由
`REGISTER_EXECUTOR_BY_TWO_TEMPS` / `REGISTER_EXECUTOR_BY_FOUR_TEMPS` 组装给同一个 sequence executor。
写分层算法时优先考虑**复用已有的 intra/inter 构件**，而不是从零写一个大而全的 template。
