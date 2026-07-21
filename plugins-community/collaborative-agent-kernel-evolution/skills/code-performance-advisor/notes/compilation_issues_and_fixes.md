# AscendC 算子编译问题与修复指南

> **目的**：记录在 `engram_gate_fusion` 算子编译/评估过程中遇到的真实问题，供 CC 自主诊断和修复。
> **适用范围**：所有通过 CAKE2 生成的 AscendC 算子，路径格式 `output/{op_name}/{op_name}Custom/`

---

## 问题一：`cmake/util/` 下所有脚本均为空文件（最高频根因）

### 症状
```
mv: cannot stat '.../build_out/op_kernel/binary/ascend910b/*.json': No such file or directory
```
`cmake --build --target binary` 显示 `[100%] Built target EngramGateFusionCustomBf16_ascend910b`，
但 `binary/ascend910b/` 目录下没有任何 `.json` 或 `.o` 文件。

### 根因
CAKE2 生成算子时，`{op_name}Custom/cmake/util/` 目录下的所有脚本（`.py`、`.sh`）都是
**空占位文件（0 字节）**，不包含任何实际内容。cmake 流程依赖这些脚本：
- `ascendc_get_op_name.py`：读取 `.ini` 获取 op 名，cmake 用它创建内核编译 target
- `ascendc_compile_kernel.py`：实际驱动 `ccec` 编译内核，空文件时静默退出（exit 0）不产生输出
- `gen_ops_filter.sh`、`gen_version_info.sh` 等：构建其他 target 时被调用

### 诊断方法
```bash
ls -la output/{op_name}/{op_name}Custom/cmake/util/ | grep " 0 "
# 如果大量文件显示 0 字节，即为此问题
```

### 修复方法（自主可执行）
从任意已成功构建的参考算子复制完整的 util 目录：
```bash
# 参考算子：fastgelu（已验证可用）
cp -r output/fastgelu/FastgeluCustom/cmake/util/. \
      output/{op_name}/{op_name}Custom/cmake/util/

# 赋予 shell 脚本执行权限（复制后可能丢失）
chmod +x output/{op_name}/{op_name}Custom/cmake/util/*.sh
```

### 修复后验证
```bash
# 验证关键脚本有内容
wc -l output/{op_name}/{op_name}Custom/cmake/util/ascendc_compile_kernel.py
# 应输出 200+ 行，不是 0 或 1

# 重新 cmake configure（必须！否则 target 不会重新生成）
cmake -S output/{op_name}/{op_name}Custom \
      -B output/{op_name}/{op_name}Custom/build_out \
      -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest \
      -DASCEND_COMPUTE_UNIT=ascend910b \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_BINARY_PACKAGE=True \
      -Dvendor_name=customize \
      -DCMAKE_INSTALL_PREFIX=output/{op_name}/{op_name}Custom/build_out \
      -DASCEND_PYTHON_EXECUTABLE=python3 \
      -DASCEND_PACK_SHARED_LIBRARY=False

# Makefile 中应出现 {OpName}_ascend910b target
grep "ascend910b" output/{op_name}/{op_name}Custom/build_out/op_kernel/Makefile
```

---

## 问题二：Shell 脚本权限不足

### 症状
```
/bin/sh: 1: .../cmake/util/gen_ops_filter.sh: Permission denied
gmake[2]: *** [...] Error 126
```
`cmake --build --target package` 失败，`binary` target 成功但 `package` target 失败。

### 根因
从其他路径复制的 `.sh` 文件可能丢失可执行权限（`-rw-r--r--` 而非 `-rwxr-x--`）。

### 修复方法
```bash
chmod +x output/{op_name}/{op_name}Custom/cmake/util/*.sh
# 然后重新执行 cmake --build --target package
```

---

## 问题三：内核二进制未重新编译（cmake 缓存问题）

### 症状
修改了 `.cpp` 内核代码后，`cmake --build --target package` 输出中**没有**
`[ascend910b] Generating {OpName}...` 行，binary 文件时间戳未更新。

### 根因
cmake 的自定义 target（AscendC 内核编译）使用时间戳判断是否需要重编译。
如果 binary 输出文件比源文件更新，则跳过。

### 修复方法
```bash
# 删除内核二进制输出目录，强制重新编译
rm -rf output/{op_name}/{op_name}Custom/build_out/op_kernel/binary/ascend910b/
rm -rf output/{op_name}/{op_name}Custom/build_out/op_kernel/CMakeFiles/{OpName}_ascend910b

# 然后重建
cmake --build output/{op_name}/{op_name}Custom/build_out --target package
```

---

## 问题四：`evaluate.py` 报 `UnboundLocalError: std_aclnn_symbols`

### 症状
```
UnboundLocalError: local variable 'std_aclnn_symbols' referenced before assignment
```
在 `skills/ascendc_evalution/scripts/evaluate.py` 中执行时报错。

### 根因
`std_aclnn_symbols` 变量在 `if custom_aclnn_apis:` 块内部初始化，但在块外部被引用。
当 `custom_aclnn_apis` 为空/False 时触发。

### 修复方法（已修复，记录备查）
文件：`skills/ascendc_evalution/scripts/evaluate.py`

```python
# 错误写法（变量作用域在 if 块内）：
if custom_aclnn_apis:
    std_aclnn_symbols: set = set()  # ← 作用域错误
    ...
if cust_lib.exists() and std_aclnn_symbols:  # ← 此处 UnboundLocalError

# 正确写法（变量在 if 块外初始化）：
std_aclnn_symbols: set = set()  # ← 移到 if 块之前
if custom_aclnn_apis:
    ...
if cust_lib.exists() and std_aclnn_symbols:  # ← 正常
```

---

## 问题五：`generate_pybind.py` 覆盖了正常工作的 `custom_ops_lib.so`

### 症状
```
ImportError: dynamic module does not define module export function (PyInit_custom_ops_lib)
```
evaluate.py 在 import custom_ops_lib 时报错，之前能正常运行。

### 根因
`generate_pybind.py` 期望 `output/{op_name}/{op_name}.cpp` 文件存在。
如果脚本实际使用了空的 `aclnn_{op_name}.cpp`（0 字节），会编译出一个损坏的 `.whl`，
安装后覆盖系统级 `/usr/local/lib/python3.10/dist-packages/custom_ops_lib*.so`。

### 诊断方法
```bash
# 检查 aclnn cpp 是否为空
ls -la output/{op_name}/{op_name}Custom/op_host/aclnn_{op_impl_name}.cpp
# 如果是 0 字节，不要运行 generate_pybind.py

# 检查系统 so 是否损坏
python3 -c "import custom_ops_lib; print(dir(custom_ops_lib))"
# 报 ImportError 说明 so 已损坏
```

### 修复方法
```bash
# 从 output/{op_name}/ 备份恢复原始 so（CAKE2 会在此处保存副本）
cp output/{op_name}/custom_ops_lib.cpython-310-aarch64-linux-gnu.so \
   /usr/local/lib/python3.10/dist-packages/

# 验证恢复
python3 -c "import custom_ops_lib; print('OK')"
```

### 预防措施
运行 `generate_pybind.py` 前，先确认：
```bash
# aclnn cpp 必须非空
[ -s output/{op_name}/{op_name}Custom/op_host/aclnn_*.cpp ] && echo "OK" || echo "EMPTY - 跳过 pybind"
```
若为空，**跳过** `generate_pybind.py`，直接用已有的 `custom_ops_lib.so`。

---

## 问题六：AscendC 内核中标量读写 GlobalTensor 的正确 API

### 症状
```
error: no member named 'GetValue' in namespace 'AscendC'
```
或
```
error: cast between floating and unsigned integer variable is not allowed in aicore function
```

### 根因
在 AscendC 内核中，对 GlobalTensor/LocalTensor 的标量访问有严格限制：
- `AscendC::GetValue(tensor, idx)` — **不存在**，会编译报错
- `wt[idx] = value` — LocalTensor 的 `[]` 不支持直接 float 赋值

### 正确 API

```cpp
// 从 GlobalTensor 读标量（如 convWeightGm）
float val = convWeightGm.GetValue(index);   // ✅ 对象方法，无 AscendC:: 前缀

// 向 LocalTensor 写标量（如 wt）
wt.SetValue(index, val);                    // ✅

// 从 LocalTensor 读标量
float v = localTensor.GetValue(index);      // ✅

// 错误写法
AscendC::GetValue(tensor, idx);             // ❌ 不存在
tensor[idx] = float_val;                    // ❌ 编译器拒绝 float 赋值
```

### 示例：正确的 GlobalTensor 标量转置循环

```cpp
AscendC::LocalTensor<float> wt = weightBuf.Get<float>();
for (uint32_t k = 0; k < kernelSize; k++) {
    for (uint32_t d = 0; d < tileD; d++) {
        float val = convWeightGm.GetValue((myHC * tileD + d) * kernelSize + k);
        wt.SetValue(k * tileD + d, val);
    }
}
AscendC::PipeBarrier<PIPE_ALL>();
```

---

## 问题七：DataCopy 替换标量循环后正确性失败（已知未解决根因）

### 症状
使用 `DataCopy` 替换 GlobalTensor 标量读取循环后，correctness 测试失败：
```
svec: 2278.00 (limit: 2.0) ← EXCEEDED
mean_re: 7.61 (limit: 2.0) ← EXCEEDED
```

### 背景
尝试用以下 DataCopy 替换 `convWeightGm.GetValue()` 标量循环：
```cpp
// Python 端预转置 conv_weight: [HC*D, 1, K] → reshape(HC,D,K) → permute(0,2,1) → [HC,K,D]
// 内核端：
AscendC::DataCopy(wt, convWeightGm[myHC * kernelSize * D], kernelSize * tileD);
```
地址计算经过分析在逻辑上是正确的，编译也无报错，但运行时精度失败。

### 已排除的原因
- ✗ SiLU 的 `Reciprocal` vs `Div` 精度差异（改回 Div 后结果完全相同）
- ✗ Python 端预转置逻辑错误（地址推导已验证正确）
- ✗ DataCopy 对齐问题（`4 * 1024 = 4096` floats，对齐 32 bytes ✓）
- ✗ GlobalBuffer 大小不匹配（`HC*D*K = 16384` 与预转置张量元素数相同 ✓）

### 当前建议（保守策略）
**不要**用 DataCopy 替换对 GlobalTensor 的标量转置读取，直到根因明确。
保持使用 `.GetValue()` + `.SetValue()` 的标量循环，虽然较慢但结果正确。

### 若未来要复现调试
可加 `--check-only` 模式（不做性能测试）验证正确性，并对比：
1. 只改 PipeBarrier（不改 DataCopy）→ 是否通过？
2. 只改 DataCopy（不改 PipeBarrier）→ 是否失败？
3. 加回单独 PipeBarrier（每个 DataCopy 后一个）→ 是否通过？

---

## 标准构建流程（完整可复用版本）

适用于所有 CAKE2 生成的 AscendC 算子，每次从头编译时执行：

```bash
OP_NAME=engram_gate_fusion           # 修改为目标算子名
OP_CLASS=EngramGateFusionCustomBf16  # 修改为目标类名（CamelCase）
OP_DIR=output/${OP_NAME}/${OP_NAME}Custom/${OP_CLASS}

# Step 1: 确保 cmake/util 脚本完整
if [ $(find ${OP_DIR}/cmake/util -name "*.py" -size 0 | wc -l) -gt 2 ]; then
    echo "[FIX] Copying cmake/util from fastgelu reference..."
    cp -r output/fastgelu/FastgeluCustom/cmake/util/. ${OP_DIR}/cmake/util/
fi
chmod +x ${OP_DIR}/cmake/util/*.sh

# Step 2: cmake configure
cmake -S ${OP_DIR} -B ${OP_DIR}/build_out \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest \
  -DASCEND_COMPUTE_UNIT=ascend910b \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_BINARY_PACKAGE=True \
  -Dvendor_name=customize \
  -DCMAKE_INSTALL_PREFIX=${OP_DIR}/build_out \
  -DASCEND_PYTHON_EXECUTABLE=python3 \
  -DASCEND_PACK_SHARED_LIBRARY=False

# Step 3: 强制重编译内核（删除旧 binary）
rm -rf ${OP_DIR}/build_out/op_kernel/binary/ascend910b/
rm -rf ${OP_DIR}/build_out/op_kernel/CMakeFiles/${OP_CLASS}_ascend910b

# Step 4: 构建完整 package
cmake --build ${OP_DIR}/build_out --target package

# Step 5: 安装
bash ${OP_DIR}/build_out/custom_opp_ubuntu_aarch64.run \
     --install-path=output/${OP_NAME}

# Step 6: 评估
python3 ${CLAUDE_PLUGIN_ROOT}/skills/ascendc-evaluation/scripts/evaluate.py ${OP_NAME} \
    --advanced-perf --task-type vector
```

---

## 快速诊断检查表

遇到编译问题时，按顺序检查：

```
□ 1. cmake/util/ 下是否有空文件？
      find {OP_DIR}/cmake/util -size 0
      → 有空文件：cp -r from fastgelu reference

□ 2. shell 脚本是否有执行权限？
      ls -la {OP_DIR}/cmake/util/*.sh | grep -v "x"
      → 无执行权：chmod +x *.sh

□ 3. cmake 是否已重新 configure？
      grep "ascend910b" {OP_DIR}/build_out/op_kernel/Makefile
      → 无 target：重新运行 cmake -S -B

□ 4. 内核 binary 是否需要强制重编？
      ls {OP_DIR}/build_out/op_kernel/binary/ascend910b/
      → 文件存在但代码已改：rm -rf binary/ascend910b/ 后重建

□ 5. custom_ops_lib.so 是否正常？
      python3 -c "import custom_ops_lib; print('OK')"
      → ImportError：从 output/{op_name}/ 恢复备份 so

□ 6. evaluate.py 是否有 UnboundLocalError？
      → 检查 std_aclnn_symbols 初始化位置是否在 if 块外
```

---

## 问题八：`binary/dynamic` 是文件不是目录导致 CPack 失败

### 症状
```
CPack Error: file COPY ".../op_kernel/binary/dynamic" to ".../customize_impl/dynamic/": Is a directory.
```
`cmake --build --target package` 失败，但 `binary` target 成功。

### 根因
`binary/` 下存在名为 `dynamic` 的**普通文件**（13KB），但 CPack cmake_install.cmake 期望它是目录。
该文件是前一次不完整构建的残留物。

### 修复方法
```bash
OP_BUILDOUT="output/{op_name}/{op_name}Custom/build_out"
# 删除文件，重建为目录
rm -f "${OP_BUILDOUT}/op_kernel/binary/dynamic"
mkdir -p "${OP_BUILDOUT}/op_kernel/binary/dynamic"
# 同时清理 CPack 缓存
rm -rf "${OP_BUILDOUT}/_CPack_Packages"
rm -f "${OP_BUILDOUT}/custom_opp_ubuntu_aarch64.run"
# 重新 package
cmake --build ${OP_BUILDOUT} --target package
```

---

## 问题九：内核倒数 API 名称错误——`AscendC::Rec` 不存在，正确是 `AscendC::Reciprocal`

### 症状
```
error: no member named 'Rec' in namespace 'AscendC'
```
尝试用 `AscendC::Rec(dst, src, 1)` 计算向量倒数时编译失败。

### 根因
**API 名称拼错**。AscendC 内核侧的倒数计算函数名为 `Reciprocal`，不是 `Rec`。

来源：
- `ascendc_api_validation_reference.md` 第 129 行明确列出：
  `| Reciprocal(dst, src, n) | 倒数 | AscendC::Reciprocal(dst, src, 1024); |`
- `op_api_list.md` 第 180 行的 `aclnnReciprocal` 是 **host 层（aclnn）API**，
  对应的 **内核层（AscendC）API 是 `AscendC::Reciprocal`**，两者不是同一层。

### 正确 API
```cpp
// ❌ 错误（API 不存在）：
AscendC::Rec(rms, rms, 1);

// ✅ 正确（内核侧倒数）：
AscendC::Reciprocal(rms, rms, 1);   // count=1 时计算单元素倒数
float rmsInv = rms.GetValue(0);     // 再读出用作标量乘数

// 完整向量化 RMSNorm（不含标量除法）：
AscendC::Muls(rms, rms, invD, 1);
AscendC::Adds(rms, rms, normEps, 1);
AscendC::Sqrt(rms, rms, 1);
AscendC::Reciprocal(rms, rms, 1);  // ← 代替 1.0f / GetValue(0)
float rmsInv = rms.GetValue(0);
AscendC::Muls(dst, src, rmsInv, tileD);
```

### TODO（待验证）
`AscendC::Reciprocal(dst, src, 1)` 是否在 CANN 8.3.RC2 / ascend910b 下编译通过，
尚未在本机实测。下次实现 VecRMSNorm 时应优先尝试此 API，确认后可消除最后一次标量除法。

---

## 问题十：`custom_ops_lib.so` 导入需要 `torch` 先导入

### 症状
```
ImportError: libc10.so: cannot open shared object file: No such file or directory
```
直接 `import custom_ops_lib` 失败，但 `import torch; import custom_ops_lib` 成功。

### 根因
`custom_ops_lib.so` 依赖 `libc10.so`、`libtorch_cpu.so`、`libtorch_npu.so` 等 torch 库，
但这些库不在 `ldconfig` 缓存或系统 `LD_LIBRARY_PATH` 中。
`torch` 自身用 RPATH 机制加载，一旦 torch 导入完成这些 so 就在内存中，
后续 `custom_ops_lib` 的动态链接即可复用。

### 诊断
```bash
# 检查 .so 依赖
readelf -d output/{op_name}/custom_ops_lib.cpython-310-aarch64-linux-gnu.so | grep NEEDED
# 验证
python3 -c "import torch; import custom_ops_lib; print('OK')"
```

### 注意
`evaluate.py` 在模块顶层 `import torch`（line 77），因此正常运行时不会触发此问题。
只有手动单独测试 `custom_ops_lib` 时需要先 `import torch`。

---

## 性能优化经验：VecRMSNorm 效果分析（R3P4）

### 优化内容
将 `RmsNormScalar` 中的标量链替换为向量化操作：
- 旧：`ReduceSum → GetValue → scalar_mul → scalar_add → Duplicate → Sqrt → GetValue → scalar_div`
- 新：`ReduceSum → Muls(1) → Adds(1) → Sqrt(1) → GetValue → scalar_div`
- 消除了每个 RMSNorm 的首次 GetValue + Duplicate（共 3 个调用点）

### 结果
| 版本 | custom_time | aiv_scalar_ratio |
|------|-------------|-----------------|
| R3P3 基线 | ~935 μs | 42.5% |
| R3P4 VecRMSNorm | ~942 μs | 41.6% |

scalar_ratio 仅降低 0.9%，性能几乎无变化。

### 根因分析
**瓶颈不在 RMSNorm，在 Gate 计算**。每个循环迭代 Gate 有：
- 4 次 `GetValue`（2次 ReduceSum 后读、1次 Sqrt 后读、1次 Exp 后读）
- 6 次标量运算（`* 0.03125`、`abs`、`safeAbs`、`sign`、`gateInput`、sigmoid）

VecRMSNorm 仅节省了 3 次 GetValue（每次调用 1 次 vs 之前 2 次），
但 Gate 的 7 次 GetValue + 6 次标量运算主导了 `aiv_scalar_ratio`。

### 结论
- `aiv_scalar_ratio ≈ 41%` 在当前算法下接近极限
- Gate 的标量依赖链（ReduceSum → scalar_branch → Duplicate → Sqrt → ...）是真正瓶颈
- Gate 有条件分支（sign 计算），难以完全向量化
- 下一步优化方向：重新设计 Gate 算法（e.g. 近似 sigmoid、消除 sqrt(abs) 路径）

---

*记录时间：2026-03-02（更新）*
*来源：engram_gate_fusion 算子优化会话（R3P3 → R3P4）*
*适用 CANN 版本：8.x，ascend-toolkit/latest，910B2*
