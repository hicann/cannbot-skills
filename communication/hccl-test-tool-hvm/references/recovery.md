# 定点补测与切包前检查

目的：在旧包仍安装时发现不可比较的基线；保留已通过证据，只补失败或缺失用例，减少整批重跑和往返切包。不按 `opIter=1`、`addr=0x0`、期望值 1000 等日志片段自动认定环境问题。

## 1. 测试前拒绝错误参数

`run.sh` 在启动 VM/mpirun 前调用 `parameters.py` 校验 dtype，`plan.py check` 也会预检。优先读取指定测试程序目录相邻的 `common/src/hccl_test_common.cc` 中 `test_typenames`；没有源码时使用已核对的 CANN 9.2.0 拼写集。此版本是 `bfp16`，不是 `bf16`，不会自动改名。合法拼写不代表目标算法支持该类型；版本增加类型时先核对对应源码并指定 `test_bin_dir`。

## 2. 只生成失败或缺失用例

下面工具不自动安装、不执行测试。路径均使用实际 skill 位置；补测沿用原来的包、环境、角色、迭代次数、通信域和算法命中断言。

```bash
python3 <hccl-test-tool-hvm>/repair.py plan --run <原run目录> --output <新补测计划.json>
python3 <hccl-test-tool-hvm>/plan.py check <新补测计划.json>
python3 <hccl-test-tool-hvm>/plan.py run <新补测计划.json> --role baseline --output <新补测批次目录>
python3 <hccl-test-tool-hvm>/repair.py merge --base <原run目录> \
  --rerun <补测中的实际run目录> --output <新合并证据目录>
```

`--role` 与原记录一致；多个实际 run 用重复 `--rerun` 合并。补测仍按一次失败即暂停的计划规则执行，不自动无限循环。旧记录缺少 replay 路径时显式提供 `--cann`、`--install-dir`、`--test-bin-dir`、`--artifact`；不猜安装目录。原计划中的多 loop/repeat 等自定义日志断言仍需按原计划验证，补测计划不从日志推断覆盖要求。

合并要求包身份及运行配置一致，拒绝替换已通过用例；原矩阵不会缩小。补齐此前未执行的用例可完成证据矩阵。FAIL 后 PASS 保留原日志及 `PASS_AFTER_RETRY`，严格验收仍失败；它只是定位信息。已合并的失败历史禁止再次自动生成诊断尝试。原始 run 与合并目录都应持久保留，哈希变化会使证据失效。

同条件诊断最多一次。若仍不稳定，停止重复执行，定位环境或代码原因；修复后按新配置重新采集受影响的基线/回归及定向证据，不能删除失败历史来刷通过。

## 3. 切离旧包前收口

```bash
python3 <hccl-test-tool-hvm>/repair.py ready \
  --baseline <旧selector基线或合并目录> \
  --baseline <新selector基线或合并目录> --require-new-selector
```

该命令要求完整、全 PASS、同一旧包，且 selector 组齐全；不负责推导矩阵是否覆盖 spec，覆盖范围仍按已审查测试计划核对。未通过表示当前证据不能作为可比较基线；是否继续编码、构建或安装由上层工作流决定。通过也只证明切包前基线技术条件满足，不等同于候选验收或任务完成。

`run.sh` 和 `plan.py` 始终为具体测试设置 `HWLOC_COMPONENTS=-gl,-opencl`；基线/回归不设 `HCCL_ALGO`，新算法定向测试仍用明确的 `HCCL_ALGO` 和注册名命中断言。
