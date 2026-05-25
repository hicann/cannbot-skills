# Catlass 最终轮审查附加检查

> 本文件由 Reviewer 在最终轮审查（总分 >= 70 且无必须修复项）时读取执行。

---

## 交付件检查清单（D1–D9）

**适用时机**：当审查预计通过（PASS / PASS WITH NOTES）时，在最终轮审查中执行。所有必选项必须满足才能判定 PASS。

| # | 交付件 | 路径 | 检查标准 |
|---|--------|------|---------|
| D1 | op_kernel 源码 | `operators/{operator_name}/op_kernel/{operator_name}.asc` | 独立编译通过，无警告；含 catlass 拼装 + `Kernel{}(params)` |
| D2 | op_host 源码 | `operators/{operator_name}/op_host/{operator_name}.asc` + `{operator_name}_tiling.h` | 含 ACL 框架 + Tiling 计算 |
| D3 | 构建文件 | `operators/{operator_name}/CMakeLists.txt` | 通过 `verify_cmake_config.py` 校验，含 `-I<catlass>/include` + `-DCATLASS_ARCH=<arch>` |
| D4 | Golden 数据生成 | `operators/{operator_name}/scripts/gen_data.py` + `golden.py` | 支持所有要求的 dtype / 转置组合 |
| D5 | 验证脚本 | `operators/{operator_name}/scripts/verify_result.py` | 支持 atol/rtol 阈值（FP32/FP16/BF16 各异） |
| D6 | 运行脚本 | `operators/{operator_name}/run.sh` | 编译 → gen_data → 跑可执行 → verify 全流程跑通 |
| D7 | 算子文档 | `operators/{operator_name}/README.md` | 含算子概述、catlass 选型摘要、API 映射、编译运行指南、测试结果、已知限制 |
| D8 | 设计文档 | `operators/{operator_name}/docs/DESIGN.md` | §1.2 catlass 选型 / §1.3 参考 example / §1.4 Kernel 适配 / §2.1 TilingKey 分支 / §2.2 Workspace 完备 |
| D9 | 计划与审查 | `docs/PLAN.md` + `docs/REVIEW.md` | PLAN 进度全部完成；REVIEW 当前轮已写入 |

---

## 代码清洁检查（最终轮专用）

| # | 检查项 | Grep 命令 | 要求 |
|---|--------|----------|------|
| C1 | printf/cout 残留 | `grep -n "printf\|cout" operators/{operator_name}/op_*/*.asc` | 无残留（或仅保留必要错误提示） |
| C2 | TODO/FIXME 残留 | `grep -n "TODO\|FIXME\|HACK\|XXX" operators/{operator_name}/op_*/*.asc` | 无残留 |
| C3 | 注释掉的代码块 | 目视检查 | 无大段注释代码 |
| C4 | 调试用硬编码 | 目视 + `grep -n "blockDim\s*=\s*[0-9]\|blockIdx\s*=\s*[0-9]"` | 无硬编码 |

---

## catlass C1–C11 终检

最终轮必须**重新**输出 catlass C1–C11 状态汇总（每项标 通过 / 失败）。任一 C1–C7 失败 → 直接 FAIL。

---

## 精度全覆盖验证（最终轮专用）

**独立运行所有 dtype × TilingKey 分支组合**，记录完整结果：

```
对每个 dtype 组合 in DESIGN.md §2.1 列出的合法组合:
  对每个 shape in 测试用例列表 (Level 0 + Level 1 + Level 2):
    1. python3 scripts/gen_data.py [shape] [dtype]
    2. ./build/{operator_name} [shape] [dtype]
    3. python3 scripts/verify_result.py [shape] [dtype]
    -> 记录: max_abs_err, max_rel_err, mismatch_count, PASS/FAIL
```

**结果汇总表**（必须在 REVIEW.md 中呈现）：

| TilingKey 分支 | dtype 组合 | Shape (M/N/K) | Max Abs Err | Max Rel Err | Mismatch | 状态 |
|---------------|----------|---------------|-------------|-------------|----------|------|
| 0 | fp16/fp16/fp32 | 128/256/256 | ... | ... | 0 | PASS |
| ... | ... | ... | ... | ... | ... | ... |

---

## 性能终检

| 项目 | 要求 |
|------|------|
| Task Duration | 与 catlass 同形态 example 基线差距 <30% |
| 调优证据（如调优） | PRE/POST 各一份 msprof 数据 + 单变量变更日志 |
| 归档目录 | `operators/{operator_name}/docs/perf/round_NNN/` |

---

## 最终轮审查流程

```
1. 执行评分检查表（维度 1-7），计算总分
   │
2. 判断是否进入最终轮附加检查：
   ├── 总分 < 70 或存在必须修复项（C1–C7 失败）→ 直接 FAIL，跳过附加检查
   └── 总分 >= 70 且无必须修复项 → 继续步骤 3
   │
3. 执行交付件检查清单（D1–D9）
4. 执行代码清洁检查（C1–C4）
5. catlass C1–C11 终检
6. 执行精度全覆盖验证，记录完整结果表
7. 性能终检
   │
8. 汇总判定：
   ├── 总分 >= 80 + 无必须修复 + D1–D9 齐全 + 代码清洁 + catlass C1–C7 全通过 → PASS
   ├── 总分 70-79 + 无必须修复 + D1–D9 齐全 + catlass C1–C7 全通过 → PASS WITH NOTES
   └── 其他 → FAIL（列出未满足项）
   │
9. 写入 REVIEW.md
```
