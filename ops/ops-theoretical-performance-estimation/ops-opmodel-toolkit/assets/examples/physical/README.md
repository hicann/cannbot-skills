# 最小物理执行示例

本例演示 Cube 到双 AIV 的交接、尾块、输出覆盖和声明式检查。M=160、N=64、K=128，
按 M 分为 64、64、32 三块，分到两个 AIC group，每块对应一个 Cube task 和两个 AIV task。
向量周期为合成成本，`calibrated: false`；这不是设备性能或执行数值精度的验收。

在用户项目目录，激活已安装 OpModel 的环境后运行：

```bash
cp -R "$OPMODEL_SKILL_DIR/assets/examples/physical" work/my_physical
opmodel work/my_physical --case work/my_physical/case.yaml \
  --hardware work/my_physical/hardware.yaml --output work/physical \
  --backend both --trace --gates capacity,assignment_coverage,region_coverage,quantity,runtime
```

先按[安装说明](../../../references/install.md)创建 `work/`；目标目录应尚不存在。
`result.json` 给出周期、后端比较和各项校验状态，`assignment.json` 应有 9 个任务，
trace 和统计文件展示执行。检查状态应为 `passed`，不能用进程成功代替检查结果。
`both` 只证明本例支持域内的 SimPy/Replay 一致性，不能代替数值或设备验收。
