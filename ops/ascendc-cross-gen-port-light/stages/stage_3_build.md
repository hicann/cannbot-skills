# 阶段 3：编译与安装

**前置条件**：`[GATE-2] FILES_MODIFIED=[...]` 已输出。

## LOADED 检查点

| 序号 | MUST READ 文件 | LOADED Token |
|------|---------------|-------------|
| 1 | `references/impl/l2-guide.md`（编译错误速查见「改动 4」末节） | `[LOADED] l2-guide` |

**LOADED Token 输出后，方可继续。**

## Step 3.1：准备构建安装脚本

**MUST 使用模板化脚本**，禁止自行拼编译命令。

### 3.1.1 填充 build_and_install.sh

读取 `references/build_system/build_and_install.sh.template`，将阶段 0 的环境变量填入：

| 模板变量 | 填入值 | 来源 |
|---------|--------|------|
| `{{CANN_SET_ENV}}` | CANN set_env.sh 完整路径 | 阶段 0 CANN 环境 |
| `{{CONDA_ENV}}` | 环境名 | 阶段 0 Python 环境 |
| `{{SOC}}` | `ascend950` | 阶段 1 目标 SoC |
| `{{OP_NAME}}` | 算子名 | 阶段 0 算子信息 |
| `{{OP_DIR}}` | 算子工程根目录 | 阶段 0 算子信息 |

将填充变量以表格形式在对话中展示（供用户审阅留痕）：

```
构建配置：
| 变量 | 值 |
|------|-----|
| CANN 环境 | /path/to/set_env.sh |
| Conda 环境 | env_name |
| 目标 SoC | ascend950 |
| 算子名 | op_name |
| 工程目录 | /path/to/op_dir |
```

展示后写入脚本并直接执行；发现配置错误时修正后重新执行。

### 3.1.2 执行构建安装

```bash
bash build_and_install.sh
```

脚本会自动完成：编译 → 安装 → 验证安装。

## Step 3.2：全量仓编译参考（按需）

如果编译错误涉及 CMake 配置或编译选项，可查阅全量仓：

- **CMake 配置**：`$DEVKIT_PATH/cmake/` 目录下查找相关 CMake 模块
- **编译脚本**：`$DEVKIT_PATH/build.sh` 查看官方编译流程

读取全量仓文件后输出：`[LOADED] $DEVKIT_PATH/<相对路径>`

## Step 3.3：处理编译错误

如果编译失败：

1. 在日志中搜索错误：`grep -n "[Ee]rror" build.log`
2. 查阅已加载的 `l2-guide.md`（「改动 4」末节编译错误速查）定位修复
3. 修复后重新执行脚本
4. 最多 3 次排错循环，3 次后仍失败则停止并向用户报告详细错误信息

> **安装/构建类故障**（安装步路径拼接错误、runtime 回退内置 kernel、stub 库 `libopapi_math.so` 缺失、nm 符号不可见、依赖下载失败）见 `references/build_system/build-troubleshooting.md`；下表覆盖**编译期**常见错误。

### 常见编译错误速查

| 错误特征 | 根因 | 修复 |
|----------|------|------|
| `Exec format error: bisheng` | bisheng 脚本缺少 shebang | `sed -i '1i#!/bin/bash' build/gen_bisheng_dir/bisheng` |
| `Error 137` | OOM Kill | 限制并行度 `-j4` |
| BF16 编译错误 | arch35/ 保留了 V220 guards | 移除 `__CCE_AICORE__ == 220` |
| `RegTensor` 未声明 | 缺少 MicroAPI 头文件/命名空间 | `using AscendC::MicroAPI::RegTensor` |
| `CastTrait` 模板错误 | CastTrait 参数不匹配 | 检查四要素 |
| `ToFloat<>` static_assert 失败 | A5 上 ToFloat 仅支持 BF16/FP8/HiFloat8 | 先 ReinterpretCast 再 ToFloat |
| `--ops= xxx`（ops 名为空） | `--ops=` 后有多余空格 | 修正为 `--ops=xxx`（无空格） |
| cube 类签名错误成片爆发（`no matching function/constructor` 60-180 处、`sizeof(UbLayout) <= 248*1024` 断言） | 低阶直跑 API 的 const 限定符/构造签名不匹配、辅助函数定义遗漏、UB 超 3510 编译期上限 | 查 `cube-migration-guide.md`「低阶直跑改造常见编译错误速查」（FA-score 实测四类）；按错误类型去重后再修，禁止按报错数量评估 |

## Step 3.4：验证安装

安装脚本会自动执行此步。手动验证：

```bash
source ${CANN_SET_ENV}
# 检查算子 .so 是否已安装
find ${CANN_PATH}/opp/vendors/ -name "libcust_opapi.so" -exec nm -D {} \; | grep -i "op_name"
```

## Gate 输出条件

- [ ] build_and_install.sh 执行成功
- [ ] 编译产物已生成
- [ ] 算子已安装到 CANN 运行环境
- [ ] libcust_opapi.so 中包含算子符号

**全部通过 → 输出 `[GATE-3] BUILD=PASS INSTALL=PASS` → 进入阶段 4**
