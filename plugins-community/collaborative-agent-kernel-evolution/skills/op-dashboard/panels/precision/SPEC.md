# Panel: 精度分析（Tab 3）— 规范 v2

---

## 数据来源

| 文件 | 提取内容 |
|------|---------|
| `evaluation_result*.txt` | 每 case 的 match_rate/max_diff/mean_diff/passed |
| `precision_results.json` | v2 schema：ratios(max_re/mean_re/rmse) + ans_vs_golden |
| `panels/precision/data.json` | 脚本提取的机器可读精度数据（Claude 读此文件做分析） |

### data.json 结构（脚本生成，Claude 读取）

```json
{
  "n_pass": 10,
  "n_total": 10,
  "cases": [
    {
      "id": 0,
      "name": "small_basic",
      "shape": "{'M': 32, 'K': 64, 'N': 48}",
      "passed": true,
      "precision": {
        "match_rate": 100.0,
        "max_diff": 0.0,
        "mean_diff": 0.0,
        "max_re": null,
        "mean_re": null
      }
    },
    {
      "id": 6,
      "name": "large_basic",
      "passed": true,
      "precision": {
        "match_rate": 100.0,
        "max_diff": 0.125,
        "mean_diff": 0.00108
      }
    }
  ]
}
```

### 精度阈值参考（Ascend 910B2）

| 指标 | 绿色（好） | 橙色（注意） | 红色（超标） |
|------|-----------|-------------|-------------|
| max_re ratio | ≤ 0.5 | 0.5–10.0 | > 10.0 |
| match_rate | 100% | 99%–100% | < 99% |
| max_diff (fp16) | ≤ 1e-3 | 1e-3–1e-2 | > 1e-2 |
| max_diff (bf16) | ≤ 0.125 | 0.125–0.5 | > 0.5 |

---

## 分析协议

Claude 读取 `data.json` 后，**必须回答以下五个问题**：

1. **总体结论**：全部 case PASS 还是有 FAIL？最大误差量级是多少？与当前数据类型（fp16/bf16/int8）的理论精度上界相比如何？

2. **误差类型解读**：本算子主要用哪种精度指标（match_rate/max_diff/mean_diff/max_re）？这个指标的物理含义是什么？

3. **误差分布规律**：哪些 case 误差最大？是否有规律（大 shape 误差更大？特定 dtype？含 bias/offset 的 case？）

4. **误差来源推断**：从算子数学结构推断误差来自哪里？（int8→fp32 量化误差？fp32→fp16 截断？bias 累积？Matmul 累加误差？）

5. **风险评估**：当前精度是否满足典型 AI 推理/训练精度要求？有无需要关注的边界 case？

---

## 输出格式（analysis.md）

Claude 写入 `output/{op_name}/panels/precision/analysis.md`，**必须严格遵守以下结构**：

### 精度全通过时

```markdown
## 精度分析

### 总体结论
[PASS] — 全 N 个 case 通过，最大误差 X（类型：fp16/bf16，理论上界 Y）。

### 误差类型解读
（本算子使用的精度指标含义；为何选用这些指标。）

### 误差分布规律
（列出最大误差的 case 名称和数值；是否有 shape/dtype 相关规律。）

### 误差来源推断
（从算子数学结构分析误差来源；各误差项的量级合理性。）

### 风险评估
（精度是否满足需求；有无需关注的边界情况；建议。）
```

### 精度未通过时（含 FAIL Case 诊断）

```markdown
## 精度分析

### 总体结论
[FAIL] — N/M 个 case 通过（FAIL：case1, case2, ...）。
最大误差 X（类型：fp16/bf16）。

### FAIL Case 诊断
逐 case 列出失败详情（必须引用 data.json 中的具体数值）：

| Case | Shape | 失败指标 | 数值 | 疑似原因 |
|------|-------|---------|------|---------|
| case_name | {M:32...} | max_re | 1234.5 | 输出全零 / 数值溢出 / tiling 越界 |

常见失败模式对照：
- **输出全零**：max_re 极大（>1000），match_rate=0% → 地址映射错误 / workspace 未初始化 / kernel 未执行
- **数值溢出**：max_diff > 1e3，特定大 shape case 才触发 → tiling 越界写入 / 累加溢出
- **精度偏差**：max_re 2-50，部分 case 失败 → 类型转换截断 / 算法数值不稳定
- **偶发不一致**：多次运行结果不同 → 多核竞态 / 未初始化内存

### 误差类型解读
（同精度通过）

### 误差分布规律
（分别描述 PASS case 和 FAIL case 的误差数值，分析规律：是否特定 shape/dtype 才失败？）

### 误差来源推断
（针对 FAIL case 的具体失败原因；结合算子结构推断根因。）

### 风险评估
（当前状态不可发布；推荐优先修复的 FAIL case；预期修复方向。）
```

---

## 验收规则

- [ ] `analysis.md` 包含全部 5 个 `###` 小节
- [ ] 总体结论中含 `[PASS]` 或 `[FAIL]` 关键词
- [ ] 误差分布规律小节中提到了至少 1 个 case 名称和具体数值
- [ ] 误差来源推断与算子类型匹配（量化算子提到 int8 误差；向量算子提到 fp16 截断等）
- [ ] 不包含性能数字（μs/ms/speedup）和 tiling 参数
- [ ] **（精度未通过时额外）** 包含 `### FAIL Case 诊断` 小节，逐 case 列出失败指标数值和疑似原因
