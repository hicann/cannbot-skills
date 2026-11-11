# Kirin 平台开发指南

Kirin 系列芯片（以 Kirin 开头的平台）是端侧 AI 处理器，开发验证为「麒麟设备部署推理（hdc/hap）+ CANN Simulator（camodel 仿真）」双链路。本文件为 Kirin 环境要点汇总（前提条件与安装方法，源出 cann-recipes-harmony-infer 官方文档）。

## 前提条件

| 依赖 | 版本要求 |
|---|---|
| python | >= 3.7.0 |
| gcc | >= 7.3.0 |
| cmake | >= 3.16.0 |
| pigz | 可选（提升打包速度），建议 >= 2.4 |
| dos2unix / Gawk | 需要 |
| glibc | >= 2.34 |

> 检查逻辑见 [scripts/check_kirin_env.sh](../scripts/check_kirin_env.sh) 段 1（`command -v` + `version_ge`，只检查不安装）。批量安装可用 upstream 仓根目录的 `install_deps.sh`（env-check 不代跑）。

## CANN 包安装

Kirin 开发需要包内同时具备 ATC 所需的完整 `opp` 与目标平台 `simulator`；常见发布形态为 mobile-station：

```bash
# 包从官方发布渠道获取，文件名形如：
# Ascend-cann-toolkit_${cann_version}_linux-${arch}-mobile-station.run
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}-mobile-station.run
./Ascend-cann-toolkit_${cann_version}_linux-${arch}-mobile-station.run --install --force --install-path=${install_path}
# 配置环境变量（安装路径由 --install-path 决定）
source ${install_path}/cann-${cann_version}/set_env.sh
```

- `${arch}`：CPU 架构，用 `uname -m` 获取（如 aarch64 / x86_64）

## Kirin 算子编译依赖（kirin-bisheng）

部分 Kirin 算子编译需额外安装 `kirin-bisheng` 子包；未安装时直接编译会报错。

```bash
# 从同版本 CANN 包中解压后安装 bisheng 编译器
./Ascend-cann-toolkit_${cann_version}_linux-${arch}-mobile-station.run --tar xvf -C ./path
cd ./path/run_package/
./cann-bisheng-compiler_${cann_version}_linux-${arch}.run --full --quiet --install-path=${install_path}
```

> 算子工程编译命令（`build.sh --pkg --soc=... --ops=...`）与产物安装见 `ops-kirin-device-verify` 步骤 1-2。

<!--
  来源（cann-recipes-harmony-infer 仓）:
    - docs/quick_install.md：前提条件 + mobile-station 包安装
    - docs/ascend_ops_guide.md：kirin-bisheng 编译器安装（§2.1）
-->

## GLIBCXX 运行时

host 可执行程序直接链接 CANN 动态库时，若库所需的 GLIBCXX 符号高于系统 libstdc++ 提供值，加载会失败。

判别方法：

```bash
readelf -V <目标 .so> | grep GLIBCXX                                   # 该库需要的最高符号版本
strings "$(g++ -print-file-name=libstdc++.so.6)" | grep GLIBCXX       # 系统提供的版本
ldd <可执行> | grep -E "libstdc\+\+|ascendcl"                          # 确认实际解析到的库
```

报错形态：`version 'GLIBCXX_x.y.z' not found (required by <某 .so>)`。

解决：`LD_PRELOAD=<提供该符号的 libstdc++.so.6>`（可取自执行 harness 的 `ext_lib/`，或 CANN 包自带的高版本 libstdc++），或升级系统 libstdc++。

自动探测：`bash scripts/check_kirin_env.sh` 的「GLIBCXX 运行时」段。

## 参考资料

- [cann-recipes-harmony-infer](https://gitcode.com/cann/cann-recipes-harmony-infer) - 鸿蒙端侧推理实践（含完整工程样例）
- [quick_install.md](https://gitcode.com/cann/cann-recipes-harmony-infer/blob/master/docs/quick_install.md) - 环境安装指南
- [ascendc_develop_guide.md](https://gitcode.com/cann/cann-recipes-harmony-infer/blob/master/docs/ascendc_develop_guide.md) - Ascend C 开发指南
