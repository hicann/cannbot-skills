# msprof 工具使用指南（AscendC 性能分析）

**来源**：整合自 triton-ascend-dev-main/guides/msprof-op.md
**适用场景**：AscendC 算子性能数据采集
**目标**：生成 op_summary.csv 和流水图用于性能分析

---

## 概览：两种采集模式

| 模式 | 使用场景 | 输出数据 | 成本 |
|------|----------|----------|------|
| **msprof op**（真机采集） | 实际运行环境的性能分析 | CSV + 计算-访存热力图 + Roofline + Cache 热力图 + 流水图 + 代码热点图 | 中等（需要真机） |
| **msprof op simulator**（仿真） | 开发调试阶段的详细仿真调优 | 指令流水图 + 代码热点图 + 内存路径吞吐波形 | 高（需要设置仿真环境） |

---

## 使用场景选择

### 何时使用 msprof op（真机采集）

**适用场景**：
- 获取真实硬件执行数据
- 快速定位宏观瓶颈（计算 vs 访存）
- 生成 op_summary.csv 用于 Phase 1 分析

**示例命令**：
```bash
msprof op --kernel-name={算子名称} {测试命令}
```

**具体示例**：
```bash
# 示例：采集 my_add 算子的性能数据
msprof op --kernel-name=my_add pytest test_my_add.py
```

### 何时使用 msprof op simulator（仿真）

**适用场景**：
- Phase 1 建议无效，需要更细粒度的流水图
- 需要查看指令级流水线
- 需要查看代码热点和调用栈

**示例命令**：
```bash
msprof op simulator --kernel-name={算子名称} --soc-version={芯片型号} {测试命令}
```

**具体示例**：
```bash
# 示例：仿真 my_add 算子，目标芯片 Ascend910B1
msprof op simulator --kernel-name=my_add --soc-version=Ascend910B1 pytest test_my_add.py
```

**前置条件**：
- 设置仿真芯片类型（通过环境变量或命令行参数）
- 如需代码热点图和调用栈，需要设置 `-g` 编译选项：
  ```bash
  export TRITON_DISABLE_LINE_INFO=0  # 对于 Triton
  # 对于 AscendC，在编译时添加 -g 选项
  ```

---

## 输出数据说明

### msprof op 输出目录结构

```
<RESULT_MSPROF_OP_DIR>/
└── {算子名称}_{日期时间}/
    ├── OPPROF_{日期时间}_{ID}/
    │   ├── OpBasicInfo.csv              # 算子基本信息
    │   ├── ArithmeticUtilization.csv    # 计算利用率
    │   ├── PipeUtilization.csv          # 流水线分解
    │   ├── ResourceConflictRatio.csv    # 资源冲突/等待
    │   ├── Memory.csv                   # GM/L1/UB 带宽与数据量
    │   ├── MemoryL0.csv                 # L0 缓存带宽
    │   ├── MemoryUB.csv                 # UB 缓存带宽
    │   └── L2Cache.csv                  # L2 缓存命中率
    └── insight_view/                    # Insight 可视化（需工具打开）
```

### msprof op simulator 输出目录结构

```
<SIMULATOR_OUTPUT_DIR>/
└── {算子名称}_{日期时间}/
    ├── pipeline/                        # 流水图（SVG/PNG）
    │   ├── block_{id}_pipeline.svg
    │   └── ...
    └── hotspot/                         # 代码热点图
        ├── hotspot_analysis.html
        └── ...
```

---

## 采集工作流

### 标准工作流（渐进式）

```
Step 1: 快速采集（真机）
  ├─ 使用 msprof op 采集基础性能数据
  ├─ 获得 op_summary.csv 和各类 CSV
  └─ 用于 Phase 1 分析（轻量级）

Step 2: 深度分析（仅在需要时）
  ├─ Phase 1 建议无效？
  ├─ 使用 msprof op（带流水图选项）或 simulator
  ├─ 获得流水图和代码热点
  └─ 用于 Phase 2 分析（深度）
```

### Phase 1 采集命令（快速）

**目标**：获取 CSV 数据，快速定位宏观瓶颈

```bash
# 最小化采集（仅 CSV）
msprof op --kernel-name={算子名称} {测试命令}

# 示例
msprof op --kernel-name=my_gemm pytest test_gemm.py
```

**数据用途**：
- 输入到 Phase 1 分析框架（`csv_systematic_analysis_framework.md`）
- 使用 `deep_research` subskill 生成优化建议

### Phase 2 采集命令（深度）

**目标**：获取流水图，进行指令级分析

**方案 A：真机流水图**（如果支持）
```bash
msprof op --kernel-name={算子名称} --export-timeline {测试命令}
```

**方案 B：仿真流水图**（更详细）
```bash
# 前置：启用调试信息
export TRITON_DISABLE_LINE_INFO=0  # 或在编译选项中加 -g

# 采集
msprof op simulator --kernel-name={算子名称} --soc-version={芯片型号} {测试命令}
```

**数据用途**：
- 结合 CSV + 流水图进行综合分析
- 使用 `deep_research` subskill（Phase 2）

---

## 常见问题与注意事项

### 1. kernel-name 如何确定？

**方法1**：查看代码中的函数名
```cpp
// AscendC 代码
extern "C" __global__ __aicore__ void my_add_kernel(...) {
    // kernel-name 就是 "my_add_kernel"
}
```

**方法2**：运行一次不带 msprof，从日志中查看

### 2. 采集失败怎么办？

**检查清单**：
- [ ] 算子功能是否正常？（先确保不带 msprof 能运行）
- [ ] kernel-name 是否正确？
- [ ] 是否有多个同名 kernel？（需要区分或使用通配符）
- [ ] 环境变量是否正确设置？

### 3. 流水图为什么是空的？

**可能原因**：
- 仿真模式需要设置 `--soc-version`
- 需要编译时加 `-g` 选项（通过环境变量或编译参数）
- 算子执行时间过短（<1us），流水图可能无法捕获

### 4. CSV 数据不完整？

**可能原因**：
- `Task Type` 不同导致某些字段为空（例如纯 AI_CORE 模式没有 `aiv_*` 字段）
- 检查 `OpBasicInfo.csv` 中的 `Task Type` 字段

---

## 数据目录规范（与 skill 集成）

### 标准目录结构

```
workspace/InputMessages/raw/{op_name}/
├── code/                                # 算子源码
│   └── {op_name}.cpp
├── profiling_data/
│   ├── profiling_csv/                   # Phase 1 数据
│   │   ├── op_summary.csv               # 主要分析对象
│   │   ├── OpBasicInfo.csv
│   │   ├── ArithmeticUtilization.csv
│   │   └── ...（其他 CSV）
│   └── profiling_flowcharts/            # Phase 2 数据（按需）
│       ├── block_0_pipeline.svg
│       ├── block_1_pipeline.svg
│       └── hotspot_analysis.html
└── roofline/
    └── goal.md                          # 性能目标
```

### 数据导入流程

**Step 1**：采集数据
```bash
msprof op --kernel-name={op_name} pytest test_{op_name}.py
```

**Step 2**：整理到标准目录
```bash
# 假设 msprof 输出在当前目录
cp OPPROF_*/OpBasicInfo.csv workspace/InputMessages/raw/{op_name}/profiling_data/profiling_csv/
cp OPPROF_*/*.csv workspace/InputMessages/raw/{op_name}/profiling_data/profiling_csv/
# 生成 op_summary.csv（如果 msprof 没有自动生成）
# 可能需要从多个 CSV 合并关键字段
```

**Step 3**（如有流水图）：
```bash
cp simulator_output/pipeline/* workspace/InputMessages/raw/{op_name}/profiling_data/profiling_flowcharts/
```

---

## 与 skill 工作流的集成

### Phase 0: Fast Triage（快速分流）

**数据需求**：`op_summary.csv`（或 `OpBasicInfo.csv`）

**采集命令**：
```bash
msprof op --kernel-name={op_name} {test_command}
```

**数据路径**：
```
workspace/InputMessages/raw/{op}/profiling_data/profiling_csv/op_summary.csv
```

---

### Phase 1: Lightweight Analysis（轻量级分析）

**数据需求**：完整的 CSV 集合（8 个维度）

**采集命令**：同 Phase 0

**数据路径**：
```
workspace/InputMessages/raw/{op}/profiling_data/profiling_csv/*.csv
```

**分析工具**：`deep_research` subskill

---

### Phase 2: Medium-Depth Analysis（中度深度分析）

**数据需求**：CSV + 流水图

**采集命令**：
```bash
# 方案 A：真机带流水图
msprof op --kernel-name={op_name} --export-timeline {test_command}

# 方案 B：仿真（更详细）
msprof op simulator --kernel-name={op_name} --soc-version=Ascend910B1 {test_command}
```

**数据路径**：
```
workspace/InputMessages/raw/{op}/profiling_data/profiling_flowcharts/*.svg
```

**分析工具**：`deep_research` subskill

---

## 最佳实践

### 1. 迭代式采集

```
First Run: 仅采集 CSV（快速）
  ├─ 用于 Phase 1 快速诊断
  └─ 生成初步优化建议

Second Run（如需）: 采集流水图
  ├─ Phase 1 建议无效
  ├─ 需要指令级分析
  └─ 用于 Phase 2 深度诊断
```

### 2. 版本对比

```bash
# 优化前
msprof op --kernel-name=my_op --output-dir=./before pytest test.py

# 应用优化

# 优化后
msprof op --kernel-name=my_op --output-dir=./after pytest test.py

# 对比 Task Duration(us)
```

### 3. 多形状测试

```bash
# 测试不同输入形状
for shape in "1024,1024" "2048,2048" "4096,4096"; do
    msprof op --kernel-name=my_op --output-dir=./shape_${shape} pytest test.py --shape=${shape}
done
```

---

## 参考链接

- 原始文档：`triton-ascend-dev-main/guides/msprof-op.md`
- 官方文档：[Ascend CANN 算子调优工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/devaids/optool/atlasopdev_16_0082.html)
- 配套文档：
  - `csv_systematic_analysis_framework.md`（Phase 1 分析）
  - `deep_research` 子技能（Phase 2 分析）
  - `op_summary_header_guide.md`（字段定义）

---

**最后更新**：2026-02-24
**维护者**：code-performance-advisor skill
