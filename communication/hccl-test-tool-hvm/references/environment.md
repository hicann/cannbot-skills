# HCCL-VM 环境就绪检查

本参考只判定 HCCL-VM 测试所需的技术条件，不下载安装软件、不修改 rootinfo、不推进上层工作流状态。

## 输入

优先使用调用方显式传入的路径；否则才读取已加载环境中的变量：

- `HCCL_ROOT`
- `CANN_HOME` 与 `${CANN_HOME}/set_env.sh`
- `ASCEND_HOME_PATH`
- HCCL-VM `install_dir`
- 可选的 `test_bin_dir`

不得从用户名、家目录或旧日志推测路径。

## 复用已确认的环境记录

路径首次经调用方确认并核验后，保存到任务目录的环境记录，后续直接继承，避免每批重新探查。
可复用现有 plan、manifest 与 artifact 的字段，不另复制整份日志或导出全部环境变量：

| 类别 | 保存及启动时核对 |
|---|---|
| 源码/工具 | 已确认的 HCCL/HCOMM 根、`${HCOMM_CODE_HOME}/test/hccl_vm`、当前 skill 工具路径 |
| 运行环境 | set_env.sh、hccl-vm、目标测试二进制、cluster、通信域/rootinfo 及其引用文件的真实路径和哈希 |
| 加载库 | CANN host/device 与 VM aarch64 目录中实际使用库的路径、软链接落点、哈希 |
| 证据 | 旧包 artifact、baseline run 目录、候选包及部署记录；保留各自来源身份 |

启动时一次批量只读核对已记录文件，不扫盘猜路径、不 source 来历未核验的脚本。
路径、落点或内容变化时只重新调查相关项；环境记录未变只允许复用发现结果，不能代替当前库身份检查。
旧冒烟或基线能否复用仍按完整证据契约决定，不能由一个“READY”标记或环境文件自身哈希推出。
构建/部署会改变库身份，切包后重新核对实际加载位置；源码身份也不能仅由包名推断。

## 静态检查

记录每项的解析后绝对路径和存在性：

1. `<install_dir>/bin/hccl-vm` 可执行；
2. `<CANN_HOME>/set_env.sh` 可读；
3. `<test_bin_dir>` 或 `${ASCEND_HOME_PATH}/tools/hccl_test/bin/` 中存在目标算子测试程序；
4. AI_CPU 模式下，`<install_dir>/lib/aarch64/` 包含当前版本要求的设备侧库；
5. `/etc/hccl_rootinfo.json` 可读，`topo_file_path` 指向存在的 topo 文件；
6. 选择的 cluster 和通信域配置能在当前安装目录解析。

缺少任一必需项时输出 `NOT_READY`、失败项、实际路径和检查命令；不要在本检查中自动运行安装器、下载包或写 `/etc`。

## 可执行冒烟

只有调用方明确授权执行共享测试环境操作时才运行。按目标引擎选择最小用例；同时覆盖 CCU 和 AICPU 仅适用于需要证明两条链路都可用的环境验收。

```bash
# CCU_SCHED
bash <hccl-test-tool-hvm>/run.sh \
  --install-dir <install_dir> --cann <CANN_HOME>/set_env.sh \
  -m CCU_SCHED -o allreduce -d int32 -s 64 -t 112 --timeout 120 \
  --log-dir <evidence-parent>

# AI_CPU
bash <hccl-test-tool-hvm>/run.sh \
  --install-dir <install_dir> --cann <CANN_HOME>/set_env.sh \
  -m AI_CPU -o allreduce -d int32 -s 64 -t 112 --timeout 180 \
  --log-dir <evidence-parent>
```

不要用永久标记文件代替当前环境证据。CANN、HCCL-VM、测试程序、rootinfo、设备库或集群配置变化后，旧冒烟结果失效。

## 输出

输出 `READY` 或 `NOT_READY`，并记录：

- 检查时间和主机；
- 所有输入路径及版本/哈希；
- 静态检查逐项结果；
- 若执行冒烟，保存 run 目录、命令、实际算法、退出码和 Checker 汇总；
- 未执行项及原因。

`READY` 只表示记录身份下的 HCCL-VM 环境满足本次测试前置条件，不代表算法、构建或开发任务通过。
