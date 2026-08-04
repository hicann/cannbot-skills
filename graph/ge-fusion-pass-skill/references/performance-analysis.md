# 性能分析（两档，带开关）

融合 pass 的性能分析分两档，**默认只做 L0 图层核对**；只有用户**显式要求**详细性能分析时才升到 L1 profiling。由统一 skill 阶段三按开关选择。

> 档位开关（统一 skill 阶段三读取）：
> - 默认 / 未指定 → **L0**（轻量、无需跑 profiling、无需 NPU 也能部分做）。
> - 用户说"详细性能分析 / 结合 profiling / 看优化前后性能变化 / msprof" → **L1**。

---

## L0（默认）：图层收益核对，对照 pass 意图

**不是"算子越少越好"——要对照需求分析文档 §4 的预期图变换，按这个 pass 声明的优化意图判定。** 融合类减少算子；decompose 类**增加**算子（如 grouped Conv → Split+多 Conv2D+Concat），意图是换成更易被已有 kernel/后续优化处理的形态，节点数上升是预期的。

对比优化前 dump（`ge_onnx_*_PreRunBegin.pbtxt`）与 pass 后 dump，核对：

| 指标 | 怎么看 | 对哪类 pass 有意义 |
|---|---|---|
| 算子数变化 | 前后节点总数 | 融合类应下降；decompose 类会上升（预期） |
| 目标算子是否被替换 | MatMul+Add→GEMM、嵌套 AddCustom 合并等（对照 `tips/dump-log-diff-checklist.md`） | 全部 |
| 中间 tensor 数 | 被消除的中间输出 | 融合类 |
| 冗余拷贝/转换是否减少 | dump 里 `TransData`/`Transpose`/`Cast`/`Reshape`/memcpy 类节点前后数量 | data_format 修改、move/消除类 |
| 关键路径长度 | 目标子图的算子级数（深度） | 拓扑重排类 |

L0 产出：一张"前后图层对照表"+ 一句"是否达成文档 §4 声明的优化意图"。**这是默认交付项，不需要 profiling，也不假设有 NPU。**

---

## L1（用户要求时）：结合 profiling 分析优化前后性能变化

用 profiling 实测优化前后**设备侧真实耗时**变化。L1 需要可用 NPU 与 CANN profiling 工具；不可用则如实标注"未运行"（门禁 G3），退回只给 L0。

### 采集：两次 profiling（baseline vs optimized）

对**同一输入、同一 soc**各跑一次，只差在 pass 是否生效：

- **baseline**：pass 未加载（C++ 不 build pass / Python 不设 `ASCEND_GE_PY_PASS_PATH`），或对原始模型。
- **optimized**：pass 生效后重新编译/加载的模型。

用 `msprof` 采集（应用命令 = 该模型的推理/前向命令，如 ATC 产物的 benchmark、`data/` 下在线脚本）：

```bash
# baseline
msprof --application="<baseline 推理命令>" --output=./prof_base --ai-core=on --aic-metrics=PipeUtilization
# optimized
msprof --application="<optimized 推理命令>" --output=./prof_opt  --ai-core=on --aic-metrics=PipeUtilization
# 解析导出（生成 CSV/timeline）
msprof --export=on --output=./prof_base
msprof --export=on --output=./prof_opt
```

> 具体 flag（`--aic-metrics`、`--export`、是否用 `msprof op` / `msprof analyze`）随 CANN 版本变化，落定前按 `references/knowledge-base.md` 查本环境 CANN 开发工具文档确认，不硬记。无 NPU / 无 msprof → L1 标注"未运行"，只交付 L0。

### 解读：对比 baseline 与 optimized 的产物

profiling 导出目录 `.../mindstudio_profiler_output/` 下：

| 文件 | 看什么 | 对 pass 的意义 |
|---|---|---|
| `op_summary_*.csv` | 每个算子实例的 `Task Duration(us)`、aicore time、op type | 被融合/替换的算子是否消失、replacement 新算子耗时；总设备耗时（各行 Task Duration 求和） |
| `op_statistic_*.csv` | 每种 op type 的 Count / Total Time / Avg / Ratio | op-type 计数与总耗时变化（MatMul+Add 是否变 GEMM；`TransData`/`Transpose` 是否减少） |
| timeline json（流水线图） | AI Core / AI CPU / 通信各流的时间线、气泡 | 关键路径是否缩短、拷贝/转换占比是否下降 |
| `api_statistic_*.csv` | runtime/acl API 耗时 | 辅助定位非算子开销 |

### 优化前后对比要点（L1 报告核心）

1. **总设备耗时**：optimized 相对 baseline 的 `op_summary` Task Duration 总和降了多少（绝对值 + 百分比）。
2. **op-type 结构变化**：`op_statistic` 前后 diff——目标算子被替换、拷贝/转换类（`TransData`/`Transpose`/`Cast`/memcpy）计数与总耗时是否下降。
3. **replacement 是否引入退化**：新建算子（如 GEMM、拆分后的多个 Conv2D）总耗时是否低于被替换的原算子；decompose 类要特别看"节点变多但总耗时是否仍下降/持平"。
4. **结论对照意图**：实测变化是否印证文档 §4 声明的优化意图；若节点减少但设备耗时未降（甚至升），如实指出——图层收益 ≠ 真实性能收益。

L1 产出：baseline/optimized 两份 profiling 摘要 + 上述 4 点对比 + 明确结论（性能提升/持平/退化，附数据）。**数据缺失的项一律标"未运行"，不外推、不编造。**
