# Broadcast 范式设计验证方法论（golden/opt 交叉验证）

> **层级**：通用知识层（generic/），语言中立（harness 为 Python，与开发语言无关）。
>
> 来源：adapters/ascendc/tiling.md §6/§7（原 broadcast-standard-tiling.md）。NDDMA 参数块等语言相关验证项在适配层。

## 1 方法论

用两份**独立实现**跑随机 case 对比，验证计算流设计正确性：

- **golden**：spec 公式的逐行翻译（原始计算语义）
- **opt**：计算流伪码的逐行翻译（融合 + 重排后的物理执行序）

两者必须独立实现（禁止互相参考）。随机 case 至少 180 组，容差默认 `rtol=1e-5, atol=1e-5`。

**用途**：验证 P trace 选定的三要素组合（API 能力 + 融合策略 + buffer 分配）能否兑现为正确计算流。

## 2 使用流程

1. 写 `golden.py` — 导出 `golden(*args, **kwargs)` 和 `generate_test_cases()`
2. 写 `opt.py` — 导出 `opt(*args, **kwargs)`，签名与 golden 一致，独立实现
3. 把下方 harness 存为 `verify.py`，取消 `from golden import ...` / `from opt import ...` 两行注释
4. `python3 verify.py`，exit 必须为 0
5. `cat verify_results.json` 确认 `total >= 180` 且 `failed == 0`

`generate_test_cases()` 返回 `list[dict]`，每项至少含 `"args"`（位置参数 list）、`"kwargs"`（关键字参数，至少含 `"dtype"`）、可选 `"tolerance"`（`{"rtol": float, "atol": float}`）。

## 3 验证 harness（通用模板）

```python
#!/usr/bin/env python3
"""算子物理计算流验证脚本（通用模板）。

约定：
    golden.py  — 导出 golden(*args, **kwargs) 和 generate_test_cases()
    opt.py     — 导出 opt(*args, **kwargs)，签名与 golden 一致，独立实现
"""

import json
import sys
import numpy as np

# ── 填入你的 golden / opt 模块 ──
# from golden import golden, generate_test_cases
# from opt import opt


def main():
    cases = generate_test_cases()
    passed = 0
    failed = 0
    failures = []

    for i, case in enumerate(cases):
        case_id = case.get("case_id", f"case_{i:04d}")
        args = case["args"]
        kwargs = {k: v for k, v in case.items() if k not in ("args", "tolerance", "case_id")}
        tol = case.get("tolerance", {"rtol": 1e-5, "atol": 1e-5})

        try:
            g = golden(*args, **kwargs)
            o = opt(*args, **kwargs)

            if isinstance(g, np.ndarray):
                ok = bool(np.allclose(g, o, rtol=tol["rtol"], atol=tol["atol"], equal_nan=True))
            elif isinstance(g, (tuple, list)):
                ok = all(
                    np.allclose(gi, oi, rtol=tol["rtol"], atol=tol["atol"])
                    for gi, oi in zip(g, o)
                )
            else:
                ok = bool(np.isclose(g, o, rtol=tol["rtol"], atol=tol["atol"]))

            if ok:
                passed += 1
            else:
                failed += 1
                failures.append({"case_id": case_id, "error": "tolerance exceeded"})

        except Exception as e:
            failed += 1
            failures.append({"case_id": case_id, "error": f"{type(e).__name__}: {e}"})

    total = len(cases)

    with open("verify_results.json", "w") as f:
        json.dump(
            {"total": total, "passed": passed, "failed": failed, "failures": failures[:20]},
            f, indent=2, default=str,
        )

    with open("verify_results.txt", "w") as f:
        f.write(f"Total: {total}  Passed: {passed}  Failed: {failed}\n")
        for r in failures[:10]:
            f.write(f"  FAIL {r['case_id']}: {r['error']}\n")
        f.write("ALL PASSED\n" if failed == 0 and total >= 180 else "INCOMPLETE OR FAILED\n")

    print(f"Total: {total}  Passed: {passed}  Failed: {failed}")
    if total < 180:
        print(f"ERROR: total={total} < 180 — 试验次数不足!")
        sys.exit(2)
    if failed > 0:
        print("FAILED")
        sys.exit(1)
    print("ALL PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

## 4 切分逻辑交叉验证块

切分与搬运逻辑按纯逻辑分块，各出 Python 对比，随机 case 100 次：

| # | 验什么（通用块） | 关键检查点 |
|---|--------|----------|
| 1 | 计算流序列 | golden(原始公式) vs opt(融合+重排) vs kernel 计算序 — 三者一致 |
| 2 | 循环 count | `count = ubBlockLength × Π maxBroShape[d>ubSplitIdx]`；split 轴 = 最内维时 inner=1 |
| 3 | 地址偏移 | B 轴 stride=0 → offset 不贡献；padding 轴 stride=0 → offset 不贡献；GM 偏移 = Σ coord×stride |

第 4 块（多维搬运参数，如 AscendC 的 NDDMA 五字段）与语言强耦合，在对应适配层的 tiling/kernel 文档中定义。

> **注意**：交叉验证的输入数据要和 kernel 看到的完全一致——验证脚本用 `maxBro`（实际 rank 元素），kernel 用 `maxBroShape[RANK]`（特化档位维数，含 padding）。差一个 padding 值就是 bug 盲区。

## 5 何时执行

agent 自行判断：
- 计算逻辑复杂（链长 >5、含分支/并行路径、多输出）、不确定正确性时执行
- 简单算子（≤3 步链、单输出、无 精度转换+融合 混合）可跳过
