# arch22 工程接线与构建

> 阶段 2 末尾接工程、阶段 3 编译安装。**先照抄仓内现有 arch22 算子的接线，再改本算子**——不同算子仓（ops-nn / ops-math / ops-transformer / 自定义仓）的构建宏与目录约定不同，本文给通用形态，落地以目标仓先例为准。

## 1. 接线清单（逐项改完再编译）

| # | 位置 | A5 形态 | A2 形态 | 备注 |
|---|---|---|---|---|
| 1 | kernel 目录 | `op_kernel/arch35/` | `op_kernel/arch22/` | 内容必须只含 A2 可用实现；不要把 arch35 文件复制过来改名 |
| 2 | kernel 入口 | `xxx_apt.cpp` 等 A5 入口 | 复用/新建 arch22 入口（按仓约定） | 入口文件里的 include 与宏分发要一起改 |
| 3 | 架构宏 | `__NPU_ARCH__ == 3510`、`__CCE_AICORE__ == 310` | `__NPU_ARCH__ == 2201`、`__CCE_AICORE__ == 220` | 死分支删除；保留的跨代分支必须两端都真能编 |
| 4 | 原型定义 | `AddConfig("ascend950", cfg)` + `opFile.value` | `AddConfig("ascend910b", cfg)`（910_93 按仓约定另配） | 删除 A5 的独立 `OpAICoreConfig` 与 `opFile` 切换 |
| 5 | dtype 列表 | 含 FP8/HiFloat8 等 | 去掉 A2 不支持项，或按 §dtype 决策保留 | 列表改了 kernel 模板也必须跟着改 |
| 6 | config 目录 | `op_host/config/ascend950/` | `op_host/config/ascend910b/`（`*_binary.json`、`*_simplified_key.ini`） | 从现有 A2 算子复制后改字段，不要从 A5 目录改名 |
| 7 | tiling | `*_tiling_arch35.cpp`、`SUPPORT_TILING_DIR "arch35"` | `*_tiling_arch22.cpp`（或公共实现）、`"arch22"` | A5 的 40KB SIMT 预留必须删 |
| 8 | CMake | `SUPPORT_COMPUTE_UNIT "ascend950"` | `"ascend910b"` | 检查是否有 `add_ops_compile_options` 的 arch 分支 |
| 9 | 构建命令 | `--soc=ascend950` | `--soc=ascend910b` | 双横线；910_93 用 `--soc=ascend910_93`（按仓先例） |
| 10 | 测试/脚本 | 写死 `ascend950`、`arch35` 的脚本 | 改 `ascend910b`、`arch22` | `grep -rn "ascend950\|arch35"` 兜底 |

### 1.1 先找"模板算子"

```bash
# 在目标仓里找一个已支持 A2 的算子，照它的目录与构建接线
ls op_kernel/arch22/ 2>/dev/null | head
grep -rn "AddConfig(\"ascend910b\"" --include=*_def.cpp . | head
grep -rn "SUPPORT_TILING_DIR" --include=CMakeLists.txt . | head
```

没有 arch22 先例的仓（纯 A5 新仓）说明"该仓尚未支持 A2"，需要先确认构建系统版本是否支持 `ascend910b`，否则阶段 0 就要上报。

## 2. 构建与安装（通用形态）

```bash
# 0) 环境（阶段 0 记录的变量）
source "${CANN_SET_ENV}"

# 1) 清构建目录（避免 arch35 缓存残留）
rm -rf ./build

# 2) 只编目标算子，先拿编译信号
bash build.sh --pkg --ops=<op_name> --soc=ascend910b 2>&1 | tee build.log

# 3) 安装自定义算子包（两种形态按仓选择）
bash install.sh                       # 仓内脚本形态
# 或： ./custom_opp_<platform>.run      # run 包形态

# 4) 验证注册与产物
ls "${ASCEND_OPP_PATH}/vendors/"      # 自定义算子 vendor 目录
python3 -c "import torch, torch_npu; print(torch.ops.npu.<op_name>)"  # 绑定存在性（如有 torch 绑定）
```

- 构建参数以 `bash build.sh --help` 的实际输出为准；**拼写错误（`-soc` 单横线、`ascend910B` 大写）会得到误导性报错**。
- 编译通过但运行时报"算子未注册"，多半是 `_def.cpp` 的 `AddConfig` 字符串或 config 目录与 `--soc` 不一致。

## 3. 常见构建错误 → 根因

| 现象 | 根因 | 处置 |
|---|---|---|
| `__VEC_SCOPE__` / `RegTensor` / `MicroAPI` 未定义 | arch22 编译链路没有 RegBase 头/宏 | 说明阶段 2 未删干净：按 `api-migration-recipes.md` §2 重写 |
| `'FixpipeParamsArch3510' was not declared` | A5 专有结构体 | 改 `FixpipeParamsV220`（或目标版本对应结构），字段重算 |
| `SetCtrlSpr` / `GetCtrlSpr` 未定义 | A5 SPR 接口 | 删除调用（A2 无此能力） |
| `CrossCoreSetFlag<4, ...>` 模板实例化失败 | A2 不支持模式 4 | 改模式 2 或普通事件同步 |
| `DT_FLOAT8_*` / `hifloat8` 未定义 | A2 的 ge 枚举/头文件无该 dtype | 按 dtype 决策去掉或替换 |
| `undefined reference to ...arch35...` | CMake/tiling/include 仍指向 arch35 | 全仓 `grep arch35`，改目录与 `SUPPORT_TILING_DIR` |
| 算子编译成功但注册表里没有 | `AddConfig` 字符串不是 `ascend910b`，或 config 目录缺失 | 对齐 `--soc` / `AddConfig` / `config/<soc>` 三处 |
| `Unsupported NPU architecture or soc` | bisheng/构建系统不认该 soc 拼写或版本不支持 | 查 `build.sh --help` 与仓内先例；必要时确认 CANN 版本 |
| 编译 OOM / 进程被 kill | 并行度过高、模板实例爆炸（A5 dtype 组合残留） | 降并行、先删 A5 dtype 组合再编 |
| 运行报 `UB out-of-range`（507035 类） | tiling 仍按 A5 预算 | 按 `resource-budget.md` 重算，用 `a2_ub_budget.py` 复核 |

## 4. 分步验证（不要一步到全量）

1. **单算子编译**：`--ops=<op>` 只编一个，失败信息聚焦。
2. **空跑/最小 shape**：先用最小可运行 shape 验证 kernel 不崩、无 UB 越界。
3. **dtype 逐个开**：一次只开一个 dtype，定位模板分支问题。
4. **再全量**：全部 dtype/shape 打开，进入阶段 4。

## 5. 可复现命令清单（交付物）

阶段 3 结束时，把以下命令原样写入报告（路径用变量，不写死个人目录）：

```bash
source "${CANN_SET_ENV}"
cd "${OP_REPO}"
rm -rf ./build
bash build.sh --pkg --ops="${OP_NAME}" --soc="${SOC_BUILD}" 2>&1 | tee build_${SOC_BUILD}.log
bash install.sh
```

> `SOC_BUILD` 取值：`ascend910b`（910B 系列）或 `ascend910_93`（A3）。两者 `NpuArch` 同为 2201，但构建/配置字符串不同，**不要混用**。
