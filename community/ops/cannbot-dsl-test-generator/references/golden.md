# 直接编写最终 Golden

在 `TEST_DIR/<op>_golden.py` 编写纯 CPU PyTorch 参考实现。文件应包含：

- `DESIGN_REFS`：列出所提供设计文件中真实的算法或阶段条目。
- `def <op>_golden(<spec 输入>, <spec 属性>):`：参数顺序与 spec 相同；可选输入提供默认值。
- `OP_NAME`、`INPUT_NAMES`、`OPTIONAL_INPUT_NAMES`、`OUTPUT_NAMES`、`ATTR_DEFAULTS`、`REQUIRED_ATTR_NAMES`、`TOLERANCE`：与当前 spec 一致的字面量元数据。
- `GOLDEN_FUNCTION = <op>_golden`、`make_inputs`、`simulate`：固定测试入口需要的公共接口。

多输出按 spec.outputs 顺序返回。Golden 不导入目标算子、CANNBotDSL 或设备库，不复制设备调度。重要数值阶段在注释中回指 spec 字段与 DESIGN ID。运行 `prepare_golden.py --spec SPEC --design DESIGN --test-dir TEST_DIR` 检查函数、元数据、CPU 独立性和容差指标，并安装固定测试入口；脚本不重写 Golden。静态检查后仍须用可运行的 CPU 示例、边界及 DESIGN Stage 人工复核数学等价性。
