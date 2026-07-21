# 🚀 Code Performance Advisor - 快速上手指南

> **v2.0 架构升级版** - Session隔离 + 动态适配 + 配置驱动

---

## 📦 前置条件

```bash
# 1. 确保在skill根目录
cd skills/code-performance-advisor

# 2. 确保规则索引已初始化
bash bootstrap.sh

# 3. 确认路径配置（可选）
cat assets/configs/paths.yaml
```

---

## ⚡ 30秒快速开始

```bash
# 运行优化workflow
python scripts/analysis_engine/workflow.py run --op fastgelu --mode interactive

# 输出示例:
# ✅ Session created: 20260225_143022_fastgelu_interactive_a7f3
# [PHASE: INIT] Profiling extracted:
#    Task Type: AI_CORE
#    Duration: 7260.54 us
#    Bottleneck Hint: Scalar-heavy + Memory-bound
# [PHASE: SUGGEST] Ready for suggestion generation
#    📚 Recommended subskill: suggest
#    💡 Claude Code can freely explore...
```

---

## 📋 核心命令速查

## 🧊 Baseline 在哪？我改完代码怎么找回最初输入？

本 skill 约定：`workspace/inputs/<op>/` 是从 CAKE2 `output/<op>/` 拷贝来的“输入基线”，用于 tagging/scoring；真正的修改应在 **session 工作区**进行。

- **输入基线（不要改）**：`workspace/inputs/<op>/code/`
- **每次运行生成的工作副本（在这里改）**：`workspace/sessions/<session_id>/working_code/`
- **不可变快照（永远保留最初版本）**：`workspace/sessions/<session_id>/baseline_snapshot/code/`

如果你“改完找不到最初输入代码”，优先去对应 session 的 `baseline_snapshot/code/`（新 session）或 `working_code/`（老 session）找。

### 运行优化
```bash
# Interactive模式（推荐）
python scripts/analysis_engine/workflow.py run --op <operator> --mode interactive

# Auto模式（自动化）
python scripts/analysis_engine/workflow.py run --op <operator> --mode auto

# 指定max-iter（默认5）
python scripts/analysis_engine/workflow.py run --op <operator> --max-iter 10
```

### 恢复Session
```bash
# 自动恢复算子的最新session
python scripts/analysis_engine/workflow.py resume --op fastgelu

# 恢复特定session
python scripts/analysis_engine/workflow.py resume --session-id 20260225_143022_fastgelu_interactive_a7f3

# 查看workflow状态
python scripts/analysis_engine/workflow.py status --session-id <session_id>
```

### 管理Sessions
```bash
# 列出所有sessions
python scripts/analysis_engine/session_manager.py list

# 列出特定算子
python scripts/analysis_engine/session_manager.py list --op fastgelu

# 查看详情
python scripts/analysis_engine/session_manager.py info <session_id>

# 清理过期sessions
python scripts/analysis_engine/session_manager.py cleanup --dry-run  # 预览
python scripts/analysis_engine/session_manager.py cleanup           # 实际清理
```

### 性能对比
```bash
# 对比两个sessions
python scripts/utils/compare_sessions.py <session_id_1> <session_id_2>

# 对比算子的所有sessions（排名）
python scripts/utils/compare_sessions.py --op fastgelu --top 10
```

---

## 🎯 典型使用场景

### 场景1: 首次优化算子

```bash
# Step 1: 运行workflow
python scripts/analysis_engine/workflow.py run --op matmul --mode interactive

# Step 2: Workflow进入SUGGEST阶段后，Claude Code会：
#   - 读取 subskills/suggest.md 获取推理框架
#   - 探索 assets/rules/ 查找相关规则
#   - 生成优化建议

# Step 3: 应用建议后，workflow自动进入下一阶段
#   - BUILD: 编译代码
#   - EVALUATE: 运行性能测试
#   - REPORT: 生成性能报告

# Step 4: 查看结果
python scripts/analysis_engine/session_manager.py info <session_id>
```

### 场景2: 并发优化多个算子

```bash
# Terminal 1
python scripts/analysis_engine/workflow.py run --op fastgelu --mode auto &

# Terminal 2
python scripts/analysis_engine/workflow.py run --op matmul --mode auto &

# Terminal 3
python scripts/analysis_engine/workflow.py run --op layernorm --mode auto &

# 查看所有sessions（完全隔离，无冲突）
python scripts/analysis_engine/session_manager.py list

# 输出:
# 🔄 20260225_150000_fastgelu_auto_a1b2    (running)
# 🔄 20260225_150005_matmul_auto_c3d4      (running)
# 🔄 20260225_150010_layernorm_auto_e5f6   (running)
```

### 场景3: 多次迭代优化

```bash
# 第1次尝试
python scripts/analysis_engine/workflow.py run --op fastgelu --mode auto
# → Session 1: 20260225_100000_fastgelu_auto_a1b2 (获得30%提升)

# 第2次尝试（不同策略）
python scripts/analysis_engine/workflow.py run --op fastgelu --mode auto
# → Session 2: 20260225_110000_fastgelu_auto_c3d4 (获得45%提升)

# 第3次尝试（激进优化）
python scripts/analysis_engine/workflow.py run --op fastgelu --mode auto
# → Session 3: 20260225_120000_fastgelu_auto_e5f6 (获得55%提升)

# 对比所有尝试
python scripts/utils/compare_sessions.py --op fastgelu --top 10

# 输出:
# #  Session ID                          Status      Duration    Improvement  Target
# 1  20260225_120000_fastgelu_auto_e5f6  ✅ completed 3240.12 us     55.3%    ✅
# 2  20260225_110000_fastgelu_auto_c3d4  ✅ completed 3950.45 us     45.6%    ✅
# 3  20260225_100000_fastgelu_auto_a1b2  ✅ completed 5040.78 us     30.2%    ✅
```

### 场景4: 迁移旧数据

```bash
# 检查旧数据
ls workspace/sessions/iterations/
# 输出: fastgelu/ matmul/ layernorm/

# 预览迁移
python scripts/utils/migrate_to_sessions.py --workspace workspace --dry-run

# 实际迁移
python scripts/utils/migrate_to_sessions.py --workspace workspace

# 验证迁移结果
python scripts/analysis_engine/session_manager.py list
```

---

## 🔧 配置自定义

### 修改CAKE2路径
编辑 `assets/configs/paths.yaml`:
```yaml
external:
  cake2_root: /custom/path/to/CAKE2  # 或相对路径: ../../../
  build_output: output/{op_name}/{OpName}Custom
```

### 调整Session保留策略
编辑 `assets/configs/session_config.yaml`:
```yaml
retention:
  keep_completed_days: 60    # 保留成功sessions 60天
  keep_failed_days: 14       # 保留失败sessions 14天
  keep_interrupted_days: 7   # 保留中断sessions 7天

limits:
  max_sessions_per_operator: 100  # 每个算子最多100个sessions
```

---

## 📊 性能指标说明

Workflow会自动提取以下指标：

### Cube算子 (AI_CORE)
- `aic_mac_ratio`: MAC利用率
- `aic_scalar_ratio`: Scalar指令占比
- `aic_vec_ratio`: Vector指令占比
- `aic_mte1_bandwidth`: Memory带宽利用率

### Vector算子 (AI_VECTOR_CORE)
- `aiv_vec_ratio`: Vector指令占比
- `aiv_scalar_ratio`: Scalar指令占比
- `aiv_bandwidth`: 带宽利用率

### MIX算子
- 同时包含上述两类指标

**瓶颈自动检测**:
- Scalar-heavy: scalar_ratio > 30%
- Memory-bound: bandwidth < 50%
- Compute-bound: mac_ratio < 60%

---

## 🐛 常见问题

### Q1: Session创建失败
```bash
# 检查workspace/inputs/<op>/是否存在
ls workspace/inputs/fastgelu/
# 应包含: code/ 和 profiling/
```

### Q2: Profiling提取失败
```bash
# 检查CSV是否存在
ls workspace/inputs/fastgelu/profiling/op_summary.csv

# 检查CSV格式（必须包含Task Type行）
head -n 20 workspace/inputs/fastgelu/profiling/op_summary.csv
```

### Q3: Session磁盘占用过大
```bash
# 查看sessions大小
du -sh workspace/sessions/*

# 清理过期sessions
python scripts/analysis_engine/session_manager.py cleanup

# 调整保留策略（见上文"配置自定义"）
```

### Q4: 如何查看某个算子的历史最佳性能
```bash
python scripts/utils/compare_sessions.py --op fastgelu --top 1
# 第1名即为历史最佳
```

---

## 🆘 获取帮助

```bash
# 查看workflow帮助
python scripts/analysis_engine/workflow.py --help

# 查看session_manager帮助
python scripts/analysis_engine/session_manager.py --help

# 查看迁移工具帮助
python scripts/utils/migrate_to_sessions.py --help
```

---

## ✨ 核心优势

1. **Session隔离** - 多算子并行，无数据冲突
2. **动态适配** - 自动识别算子类型（Vector/Cube/MIX）
3. **配置驱动** - 路径可定制，便于移植
4. **历史追溯** - 完整保留每次优化的代码快照和性能数据
5. **LLM友好** - 简化提示，让Claude Code自由探索

---

**版本**: v2.0
**更新日期**: 2026-02-25
**状态**: ✅ 生产就绪

**立即开始**: `python scripts/analysis_engine/workflow.py run --op <your_operator> --mode interactive`
