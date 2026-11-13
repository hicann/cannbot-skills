# 没有现成测试时的验证通道

> 适用：仓库没有可用的 pytest/UT，或公共 torch API 的分发/回退无法证明 custom 包真的被执行。
> 目标：用**直调 ACLNN** 建立可信的最小验证，而不是依赖“跑了但不知道跑了谁”。

## 1. 先做依赖与分发自检

| 检查 | 命令/现象 | 处置 |
|---|---|---|
| pytest 可用 | `python -m pytest --version` | 缺 pytest/pandas/openpyxl 时，离线环境从同版本环境拷贝 site-packages，或改用直调 |
| 测试数据依赖 | `import tensorflow` / `import ml_dtypes` | 仅 dtype 映射用途时，用 numpy stub（`bfloat16 = np.float16`） |
| custom 包是否被调用 | 日志里出现 vendor 的报错/门禁文案；否则可能回退到内置算子 | 改用 ACLNN 直调 |
| 公共 API 回退特征 | `torch.ops` 报 “not supported on the NPU backend”，或开关特性后行为完全不变 | 不要用它判定 custom 行为，改直调 |
| vendor 库路径 | `<install>/vendors/<vendor>/op_api/lib/libcust_opapi.so` | 直调时显式 `LD_LIBRARY_PATH` 指向它 |
| custom 包是否真的含目标 soc | 安装产物必须有 `op_impl/ai_core/tbe/kernel/<soc>/` 和 `config/<soc>/binary_info_config.json` | 二者缺一却仍能跑通，说明走的是内置算子/回退，**不算迁移成功** |

## 2. ACLNN 直调骨架（ctypes）

库的职责要分清：`libascendcl.so` 提供 Runtime（`aclrtMalloc/Memcpy/Stream`、`aclrtCtxSetSysParamOpt`），**`libnnopbase.so` 提供 `aclCreateTensor` / `aclCreateTensorList` / `aclCreateIntArray`**，vendor 的 `libcust_opapi.so` 提供 `aclnn<Op>GetWorkspaceSize` / `aclnn<Op>`。

```python
import ctypes as C
acl = C.CDLL("libascendcl.so", mode=C.RTLD_GLOBAL)
meta = C.CDLL("libnnopbase.so", mode=C.RTLD_GLOBAL)
op = C.CDLL(f"{VENDOR_LIB}/libcust_opapi.so", mode=C.RTLD_GLOBAL)

acl.aclInit(None); acl.aclrtSetDevice(0)
stream = C.c_void_p(); acl.aclrtCreateStream(C.byref(stream))
# 输入：aclrtMalloc + H2D；aclCreateTensor(dims, ndim, dtype, None(连续), 0, ACL_FORMAT_ND, storage, ndim, dev_ptr)
# 调用：GetWorkspaceSize(..., &ws, &executor) -> aclrtMalloc(ws) -> aclnn<Op>(ws, size, executor, stream)
# 输出：aclrtSynchronizeStream + D2H 读回
```

**dtype 枚举容易错**（`acl_base_rt.h`）：`ACL_FLOAT=0`、`ACL_FLOAT16=1`、`ACL_INT8=2`、`ACL_INT32=3`、`ACL_UINT8=4`、`ACL_INT16=6`、`ACL_UINT16=7`、`ACL_UINT32=8`、**`ACL_INT64=9`**、**`ACL_UINT64=10`**、`ACL_DOUBLE=11`、`ACL_BOOL=12`、**`ACL_BF16=27`**；`ACL_FORMAT_ND=2`。

- 张量列表类算子（如 GroupedMatmul V4）用 `aclCreateTensorList(ptr_array, n)`，签名里是 `aclTensorList*`，不能按单张量传；
- 失败时读 `aclGetRecentErrMsg()`，它包含 tiling/kernel 的真实报错，比 torch 的 `RuntimeError` 更接近根因。

## 3. 先证明“缺口”真的存在

不要只凭文档、目录名或静态扫描就认定 A2 缺失能力（本仓曾出现“文档说 A2 不支持、实际能跑”的情况）。最小证据链：

1. 构建目标 soc：`build.sh --pkg --soc=ascend910b --ops=<op>`；
2. 检查安装包内是否有目标 soc 的 kernel：`op_impl/.../kernel/ascend910b/<op>`；
3. 直调 ACLNN：若报 `SoC version ascend910b verification failed / not configured through AddConfig` → 缺口成立；
4. 若能 `getws=0, run=0` 且结果正确 → 该算子 A2 本来可用，应改做性能/边界任务或换题。

## 4. 常见坑

> 先跑 `python3 scripts/preflight_env.py`：一次列出 CANN/依赖/构建工具/vendor 库缺项，避免边试边发现缺包。

| 坑 | 表现 | 处置 |
|---|---|---|
| mask/索引张量 dtype 错 | 不报错但结果错乱 | 按文档确认（如 BSA `blockSparseMask` 是 **INT8**，不是 INT32） |
| 输出按布局取错 | corr 接近 0 或形状对不上 | BNSD 取 `out[0,0]`，BSND 取 `out[0,:,0,:]`；TND 按 `T` 维 |
| 尺寸约束不满足 | GetWorkspaceSize 成功、run 失败 | 读 tiling 报错（如 `qBlockNum >= 128`），先构造最小合法 shape |
| 用 `torch.ops.*` 调 custom | 报 “not supported on the NPU backend” 回退 CPU | 改用 ACLNN 直调 |
| 依赖 torch_npu 公共 API | 开关确定性/特性后行为不变 | 同样改用 ACLNN 直调，避免内置算子兜底 |
| 必填可选项缺失 | `xxx is required, but it is empty`（FAG 的 attentionIn/softmax、SFMLA 的 cu_seqlens） | 按签名补齐；反向类算子先用 forward ACLNN 或 CPU 生成 softmax 统计量 |
| 枚举值/参数个数错 | `wrong type`、`incorrect dtype DT_INT32. It should be DT_UINT64` | 对照签名逐个核对 dtype 与参数个数 |
| 反向类 golden 累加写错 | gradWeight 与参考差一个 batch 维（`dot` 形状不匹配或漏 batch） | 按 `gw[i] += (gy[t] * x[src]).sum(axis=0)` 累加 batch 维 |
| mask 形状/dtype 猜错 | `mask must be [B,S]`、`mask must be BOOL` | 以文档/报错为准；先不带 mask 跑通再加 |

## 5. 最小验收

1. 一个 **已支持路径**（如 BNSD）与 CPU 参考对齐：`corr=1.0`，`max_abs` 在 fp16 容差内；
2. 一个 **目标路径**（改动前失败/报门禁）在改动后 run_status=0 且精度对齐；
3. 记录：dtype/形状/布局、参考实现、改动前后状态、原始报错。

## 6. flag/枚举参数与 routing 类 golden

- **bool/枚举参数会改变输出语义**：如 `expertTokensNumFlag=false` 时 count 输出可能不写入（读回全 0）、`expertTokensNumType=0` 输出 cumsum 而非每专家计数、`rowIdxType` 决定 gather/scatter 索引；写 golden 前逐项确认。
- **routing/排序类**：按 (expert, 原始行顺序) 稳定排序，扁平索引 `i*K+j`；gather 索引是 `out[sortedRowIdx[i]] = i`，scatter 是 `out[i] = sortedRowIdx[i]`；容量/丢弃模式会改变输出行数。
- **“Optional”不等于可传 nullptr**：部分输出即使非量化也要求非空（如 `expandedScaleOut`），以报错为准。
- **动态 shape 算子**：`msprof` 可能只产出 `api_statistic`、没有 `op_summary`；此时退回 ACL event 计时并在结论里注明口径，不要伪造 kernel 时间。

## 反向算子验证底线

`getws=0` + `run=0` 只证明接口与 tiling 可用，不证明梯度正确。反向类算子必须至少满足其一：

1. 仓库自带 pytest（`tests/` 下 `*_grad` 用例）全通过；
2. 自建 golden：用 float64 实现同一语义，逐输出（含 dq/dk/dv/d_sinks）比较 `matched_ratio` 与 `max_abs`。
