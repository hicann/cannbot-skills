# `output/<project>/` 项目目录与报告统一约定

## 1. 目录结构

```text
output/<project>/
├── src/
│   └── kernels/<op>/
├── tests/
│   ├── cpu/
│   └── npu/
├── docs/
│   ├── REPORT.md
│   └── verification.json
└── manifests/
```

arch22→arch35 移植在 `src/kernels/<op>/` 中保留源代与目标代的清晰边界；正向→反向生成在同一算子目录内区分 forward 与 backward 产物。报告只引用仓库内相对路径。

## 2. `verification.json`

```json
{
  "schema_version": "1.0",
  "project": "<project>",
  "environment": {
    "soc": "<soc>",
    "cann": "<version>",
    "compiler": "<version>"
  },
  "precision": {
    "truth": "cpu_fp64_or_audited_cpu_pytorch",
    "cases_total": 0,
    "cases_passed": 0,
    "max_abs_error": 0.0,
    "max_rel_error": 0.0
  },
  "performance": {
    "baseline_npu_ms": 0.0,
    "candidate_npu_ms": 0.0,
    "speedup": 0.0,
    "warmup": 3,
    "iterations": 10
  },
  "determinism": {
    "runs": 5,
    "passed": false
  }
}
```

性能字段只记录相同目标 NPU 环境内的基线/候选 A/B；精度字段使用 CPU 真值。不得将不同硬件上的时延混入同一个 speedup。

## 3. `REPORT.md`

报告依次包含：

1. 环境指纹与复现命令。
2. arch22→arch35 移植差异，或 forward→backward 的梯度契约。
3. CPU 真值精度结果与失败用例。
4. 相同目标 NPU 上的性能 A/B 和 profiling 结论。
5. 确定性、已知限制和交付文件列表。

## 4. 更新流程

修改实现后同步更新 `verification.json` 与 `REPORT.md`，检查所有链接、用例计数和产物路径；只有精度、确定性和结构检查都通过后，才能标记交付完成。
