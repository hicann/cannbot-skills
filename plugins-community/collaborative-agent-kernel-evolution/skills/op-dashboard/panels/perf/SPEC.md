# Panel: 性能报告（Tab 4）— 规范 v2

---

## 数据来源

优先级（从高到低）：

| 优先级 | 文件 | 内容 |
|--------|------|------|
| 1 | `op_summary_custom_fn_*.csv` + `op_summary_reference_fn_*.csv` | 平铺 msprof CSV，每 case 独立文件 |
| 2 | `profiling/multi_case_report.csv` | 多 case 汇总 msprof 计时 |
| 3 | `profiling/report.txt` | 单 case msprof 计时（msprof_fair_bench.py 输出） |
| 4 | `evaluation_result*.txt` | 内嵌 speedup（快速评估，精度低） |
| 5 | `panels/perf/data.json` | 脚本提取的机器可读性能数据（Claude 读此文件做分析） |

### data.json 结构（脚本生成，Claude 读取）

```json
{
  "avg_speedup": 0.45,
  "cases": [
    {
      "id": 0,
      "name": "small_basic",
      "shape": "{'M': 32, 'K': 64, 'N': 48}",
      "performance": {
        "ref_time_us": 6.8,
        "custom_time_us": 11.7,
        "speedup": 0.58,
        "source": "msprof"
      }
    },
    {
      "id": 6,
      "name": "large_basic",
      "performance": {
        "ref_time_us": 8.5,
        "custom_time_us": 24.8,
        "speedup": 0.34,
        "source": "msprof"
      }
    }
  ]
}
```

### 重要说明：参考基准含义

| 参考实现 | 含义 | 何时使用 |
|---------|------|---------|
| `MemSet` kernel | 最小计算基准（内存清零）| AscendC 默认，反映 launch overhead |
| `PyTorch reference` | 等价功能的 PyTorch 实现 | msprof_fair_bench.py 方案 |
| 若 speedup < 1x | 算子比 MemSet 还慢 | 说明 launch overhead 占主导或算子太小 |

---

## 分析协议

Claude 读取 `data.json` 后，**必须回答以下四个问题**：

1. **基准说明**：参考实现是什么（`source` 字段）？`MemSet` 基准的含义是什么？speedup < 1x 是否意味着性能差？

2. **性能规律**：小/中/大 shape 下性能如何变化？是否存在 overhead 主导的拐点（即 shape 小时 speedup 低，shape 大时 speedup 高）？

3. **瓶颈推断**：根据 speedup 数值和 shape 规律，当前瓶颈是什么？（Launch overhead？Global Memory 带宽？Compute bound？AIC/AIV 流水线利用率？）

4. **优化建议**：给出 1-3 条具体的、可操作的优化方向（需要结合 tiling 设计和算子特性）。

---

## 输出格式（analysis.md）

Claude 写入 `output/{op_name}/panels/perf/analysis.md`，**必须严格遵守以下结构**：

```markdown
## 性能分析

### 基准说明
参考实现为 [MemSet kernel / PyTorch reference]。含义：[解释]。
当前平均 speedup = X（若 < 1x，说明：[解释]）。

### 性能规律
（按 small/medium/large shape 分别描述 speedup 趋势；
 指出拐点（若有）：从哪个 shape 开始 compute 开始主导 overhead。）

### 瓶颈推断
（根据 speedup 数值和规律，分析主要瓶颈；
 给出量化依据，如"large_basic(512×1024×2048) speedup=0.34x，
 custom耗时24.8us，推测 Global Memory 搬运 XX GB 占主导"。）

### 优化建议
1. [具体建议，如：增大 tile 尺寸以提高 GM 带宽利用率]
2. [具体建议，如：使用 double-buffer 流水掩盖 GM 访问延迟]
3. [可选第三条]
```

---

## 验收规则

- [ ] `analysis.md` 包含全部 4 个 `###` 小节
- [ ] 基准说明小节明确指出 source 类型（msprof/npu_event）
- [ ] 性能规律小节区分了 small/medium/large shape（若测试 case 覆盖多个规模）
- [ ] 瓶颈推断小节提到了 data.json 中的具体数值（case 名称 + speedup 值）
- [ ] 优化建议小节至少 1 条
- [ ] 不包含精度指标（max_diff/match_rate 等）和 tiling 常量
- [ ] **（精度未通过时额外）** 基准说明小节首行注明 `⚠ 精度未通过，性能数据仅供调试参考`
