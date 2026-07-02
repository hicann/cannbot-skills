# 约束

## 铁律

```
NO PYTEST FILE WITHOUT VERIFYING IT CAN BE COLLECTED AND EXECUTED

NO INLINE PARAMS — PARAMS 必须通过 json.load 从 S5_mapped_cases_low.json 运行时读取，
禁止将参数组合硬编码/内嵌到 pytest 文件中。修改 S5_mapped_cases_low.json 后不应需要重新生成 pytest 文件。

NO MODIFIED TOLERANCE — 精度标准（rtol/atol）禁止修改，这是质量保证的最后一道门槛。
精度不达标的 case 通过 XFAIL 记录偏差信息，保留可追溯性，不阻塞门禁交付。
```

## 验证流程

生成 `conftest.py` + `S6_test_{op_name}.py` 后，按以下顺序验证：

1. `python -m py_compile conftest.py S6_test_{op_name}.py` — 语法检查。失败 = 文件有语法错误，必须修复。
2. `pytest --cases-file=S5_mapped_cases_low.json --collect-only S6_test_{op_name}.py` — 收集检查（`--cases-file` 可识别说明 conftest.py 生效）。失败时区分原因：
   - **SyntaxError / NameError** → 文件有问题，修复后重试
   - **ModuleNotFoundError（torch_npu 等）** → 环境缺依赖，不是文件问题，可以继续
   - **其他** → 具体分析
3. `ASCEND_GLOBAL_LOG_LEVEL=1 pytest S6_test_{op_name}.py --cases-file=S5_mapped_cases_low.json -q --tb=line` — **执行检查 + 日志采集**。目标：所有用例都能合法运行。判定规则：
   - **合法**：PASSED / XFAIL / SKIPPED
   - **非法 — 必须修复**：
     - **FAILED**（任何类型的失败，包括 assertion）
     - **ERROR**（import 失败、API 不存在、kernel crash 等）
     - **RuntimeError**（API 签名错误、shape 不匹配、Tiling 失败等）
     - **AttributeError**（API 名称错误、模块缺少属性等）
     - **假 PASS**（NPU 不可用时静默通过，未做任何实质验证）
   - 精度不达标的 AssertionError **必须**用 `try/except` 包裹 `assert_close`，捕获后调用 `pytest.xfail(reason=...)`，使结果为 XFAIL 而非 FAILED
   - 修复后必须重新运行确认，直到 0 个 FAILED / ERROR / RuntimeError / AttributeError
4. 复制 plog 并生成 tilingkey 覆盖率报告：
   ```bash
   mkdir -p tests/whitebox/tilingkey_logs/
   PLOG=$(ls -t ~/ascend/log/debug/plog/plog-*.log | head -1)
   cp "$PLOG" tests/whitebox/tilingkey_logs/{op_name}_full.log
   python {skill_scripts}/compute_tilingkey_coverage.py \
     --log-path tests/whitebox/tilingkey_logs/{op_name}_full.log \
     --param-def tests/whitebox/S2P2_param_def.json \
     --output-dir tests/whitebox/
   ```
    输出：`S6_tilingkey_coverage.json`（含全局与 per_group 覆盖率）。覆盖率报告为信息性产出，不阻塞门禁交付。
    
    **注意**：部分算子存在 tiling 侧 `SetTilingKey` 的值与 kernel 侧 `TILING_KEY_IS` 的值不一致的情况（如 bitmask 映射：源码 key=0 在 plog 中打印为 `tilingKey: 1`）。覆盖率分析时需注意两侧 key 的对应关系，必要时手动核对源码确认映射规则。

## 严格禁止

1. 禁止手动编写参数组合——必须使用 S5_mapped_cases_low.json 中的全部内容
2. 禁止将 JSON 内容内嵌为 PARAMS 列表——必须使用 json.load 从 S5_mapped_cases_low.json 运行时读取
3. 禁止手动推导 tensor shape/dtype——必须直接从 `case["tensors"]` 读取
4. 禁止假设 NPU 环境一定可用——使用 `pytest.importorskip("torch_npu")` 守护
5. 禁止在测试函数中硬编码具体参数值
6. 禁止声称"文件已生成"但没跑 `pytest --collect-only` 验证
7. 禁止声称"文件已通过验证"但没跑 `pytest S6_test_{op_name}.py` 实际执行验证
8. 禁止因输出模式不同而跳过用例——全量解包返回值，按模式验证对应输出
9. 禁止在 prompt 或代码中硬编码算子特定的 tensor 名称、属性名称或输出名称
