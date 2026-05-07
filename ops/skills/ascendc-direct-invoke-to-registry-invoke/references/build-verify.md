# 构建验证闭环(Build & Verify)

> 本文件描述 **skill 最后一道工序**:在生成完所有源文件之后,如何用 msopgen 创建验证工程、编译、自动修复并交付给用户。对应 [SKILL.md](../SKILL.md) 步骤 9(构建验证)。
>
> 规则引用全部走 [R-xxx](rules.md) 编号,不在此处复述;失败现象对应的案例走 [Case N](cases.md)。

## 总流程

```
[生成完成] → (A) 环境探测 → 可用 ──→ (B) 自动闭环编译/安装/一致性验证 ──→ (C) 成功/失败报告
                            ↓
                            不可用 ──→ (D) 降级:输出人工验证手册
```

涉及规则:[R-047](rules.md#r-047优先探测-msopgen不可用则降级人工手册), [R-048](rules.md#r-048验证工程目录与用户输出同级), [R-049](rules.md#r-049编译循环前先跑-opdef-静态预检), [R-050](rules.md#r-050编译循环最多-5-轮--错误映射表), [R-059](rules.md#r-059阶段-6-必须直接执行-custom_opp-run-安装算子包), [R-060](rules.md#r-060阶段-7-必须生成-example-并用原直调输入做-aclnn-二进制一致性验证)。

---

## (A) 环境探测

按优先级依次尝试定位 `msopgen` 可执行文件。一旦命中即停止,把命中路径记为 `MSOPGEN_BIN`、其对应的 CANN 根目录记为 `CANN_PACKAGE_PATH`:

```bash
# 1) 用户已 source 过 set_env.sh,PATH 里有 msopgen
command -v msopgen 2>/dev/null

# 2) 常见环境变量
[ -x "${ASCEND_HOME_PATH}/python/site-packages/bin/msopgen" ] && \
    echo "${ASCEND_HOME_PATH}/python/site-packages/bin/msopgen"
[ -x "${ASCEND_TOOLKIT_HOME}/python/site-packages/bin/msopgen" ] && \
    echo "${ASCEND_TOOLKIT_HOME}/python/site-packages/bin/msopgen"
[ -x "${INSTALL_DIR}/python/site-packages/bin/msopgen" ] && \
    echo "${INSTALL_DIR}/python/site-packages/bin/msopgen"

# 3) 默认安装位置
for p in \
    /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/msopgen \
    "${HOME}/Ascend/ascend-toolkit/latest/python/site-packages/bin/msopgen"; do
    [ -x "$p" ] && echo "$p" && break
done
```

`CANN_PACKAGE_PATH` 取法:从 `MSOPGEN_BIN` 向上三级(`python/site-packages/bin/msopgen` → 根);或直接用 `${ASCEND_HOME_PATH}` / `${ASCEND_TOOLKIT_HOME}` 实际值。

**判定**:

- **成功**(拿到可执行的 `MSOPGEN_BIN`,且 `${CANN_PACKAGE_PATH}` 下能看到 `compiler/` / `toolkit/`)→ 走 (B)
- **失败**(全部路径不存在,或 Windows / 纯 dev 容器等没装 CANN 的环境)→ 走 (D)

Windows 平台(`OS Version: win32`)一般无 CANN 工具链,默认直接判失败。

---

## (B) 自动闭环编译

### B.1 准备变量

```bash
USER_DIR="<UserOutputDir>"                      # 绝对路径
PARENT_DIR="$(dirname "${USER_DIR}")"
OP_NAME="<OpCamel>"                             # 大驼峰算子名
VERIFY_DIR="${PARENT_DIR}/${OP_NAME,,}_verify_project_$(date +%Y%m%d_%H%M%S)"

# SOC 必须是 CANN 芯片系列名(family),遵循 R-034:
#   npu-smi 的 Ascend910B3 / Ascend910B4 → ascend910b
#   npu-smi 的 Ascend310P3 / Ascend310P1 → ascend310p
#   npu-smi 的 Ascend910A / Ascend910     → ascend910
# 合法系列名可从 ${ASCEND_HOME_PATH}/compiler/data/platform_config/*.ini 文件名得到
SOC="ascend910b"
```

**约束**(引用 [R-048](rules.md#r-048验证工程目录与用户输出同级) / [R-058](rules.md#r-058useroutputdir-与-verifyprojectdir-必须独立共存禁止清理验证目录)):

- `<VerifyProjectDir>` **必须**与 `<UserOutputDir>` 同级(`${PARENT_DIR}/xxx`)
- `<VerifyProjectDir>` **绝对路径不得等于** `<UserOutputDir>`,**也不得互为父子** — [R-058]
- **禁止**放在 `<UserOutputDir>` 内部;**禁止**用 `/tmp` / `/var/tmp`
- 目录名加时间戳防覆盖
- 每次重试编译**不**重新 `msopgen gen`,只重覆盖源文件
- **两者都是最终交付件**:验证目录在任何阶段(包括失败)**都不得 `rm -rf`**,必须原封保留作为编译证据链;需要重跑 msopgen 时用 `mv` 改名备份,不用 `rm -rf` — [R-058]

**独立性硬校验**(跟在上面变量声明后,**不通过不得进 B.2**):

```bash
USER_ABS="$(readlink -f "${USER_DIR}" 2>/dev/null || realpath "${USER_DIR}")"
VERIFY_ABS="$(readlink -f "${VERIFY_DIR}" 2>/dev/null || realpath "${VERIFY_DIR}" 2>/dev/null || echo "${VERIFY_DIR}")"

[ "${USER_ABS}" = "${VERIFY_ABS}" ] && { echo "[FATAL] VERIFY_DIR == USER_DIR,违反 [R-058]"; exit 1; }
case "${VERIFY_ABS}/" in "${USER_ABS}/"*) echo "[FATAL] VERIFY_DIR 位于 USER_DIR 内部,违反 [R-058]"; exit 1 ;; esac
case "${USER_ABS}/" in "${VERIFY_ABS}/"*) echo "[FATAL] USER_DIR 位于 VERIFY_DIR 内部,违反 [R-058]"; exit 1 ;; esac
case "${VERIFY_ABS}" in /tmp/*|/var/tmp/*) echo "[FATAL] VERIFY_DIR 落在临时目录,违反 [R-058]"; exit 1 ;; esac
echo "[OK] USER=${USER_ABS}  VERIFY=${VERIFY_ABS} 路径独立性通过"
```

### B.2 首次生成骨架 + 替换源文件

```bash
"${MSOPGEN_BIN}" gen \
    -i "${USER_DIR}/${OP_NAME,,}.json" \
    -c "ai_core-${SOC}" \
    -lan cpp \
    -out "${VERIFY_DIR}"

# ---- 结构校验门([R-053]) ----
# msopgen 的 -lan 默认是 py (TBE DSL 旧结构),必须校验真生成了 Ascend C 新结构
# 正向:op_host/ + op_kernel/ 必须同时存在
# 负向:tbe/ / impl/ / op_info_cfg/ 任一出现即视为落回 TBE DSL
if [ ! -d "${VERIFY_DIR}/op_host" ] || [ ! -d "${VERIFY_DIR}/op_kernel" ] \
   || [ -d "${VERIFY_DIR}/tbe" ] || [ -d "${VERIFY_DIR}/impl" ] || [ -d "${VERIFY_DIR}/op_info_cfg" ]; then
    echo "[FATAL] msopgen gen 未生成 Ascend C 标准结构 (op_host/ + op_kernel/)。"
    echo "        当前落成目录: $(ls ${VERIFY_DIR})"
    echo "        诊断:极大概率是 -lan cpp 未生效(msopgen 默认 -lan py 走 TBE DSL)。"
    # [R-058] 禁止 rm -rf;重命名备份后换新时间戳目录重跑
    BACKUP_DIR="${VERIFY_DIR}.badstructure_$(date +%s)"
    mv "${VERIFY_DIR}" "${BACKUP_DIR}"
    echo "        已将坏结构重命名为 ${BACKUP_DIR} (未删除,供事后核查)"
    echo "        修复:重跑上一条 msopgen gen 并确保 -lan cpp 显式出现,VERIFY_DIR 用新时间戳。"
    exit 1
fi

# 删除 msopgen 默认合并文件([R-056] 硬要求:绝不把合并 cpp 带进交付)
#   msopgen gen 默认会生成 op_host/<op>.cpp,里面把 OpDef + InferShape + Tiling 揉一起
#   本 skill 交付拆分 4 件套(_def/_tiling/_infershape + tiling.h),必须先删掉合并版
rm -f "${VERIFY_DIR}/op_host/${OP_NAME,,}.cpp"

# 覆盖 op_host 全部(拆分后的 _def / _infershape / _tiling + tiling.h + config/)
cp -rf "${USER_DIR}/op_host/." "${VERIFY_DIR}/op_host/"

# 覆盖 op_kernel 全部(cpp + h)
cp -rf "${USER_DIR}/op_kernel/." "${VERIFY_DIR}/op_kernel/"

# op_graph 若存在一并覆盖
if [ -d "${USER_DIR}/op_graph" ]; then
    mkdir -p "${VERIFY_DIR}/op_graph"
    cp -rf "${USER_DIR}/op_graph/." "${VERIFY_DIR}/op_graph/"
fi

# CMakePresets.json 的 ASCEND_CANN_PACKAGE_PATH 改成 ${CANN_PACKAGE_PATH}
# 用 sed 修改
```

若 msopgen 默认生成的 `op_host/CMakeLists.txt` 硬编码 `<op>.cpp`(没用 glob),同步改成引用拆分后的三个 cpp。先 grep 确认:

```bash
grep -n "${OP_NAME,,}.cpp" "${VERIFY_DIR}/op_host/CMakeLists.txt"
```

### B.2.1 Host 文件拆分校验门([R-056](rules.md#r-056host-必须严格拆成-4-个文件禁止合并单-cpp))

**目的**:msopgen 默认生成合并的 `op_host/<op>.cpp`(OpDef + InferShape + Tiling 揉一起)。本 skill 的交付形态是**拆分 4 件套**——必须确保 `<UserOutputDir>` 与 `<VerifyProjectDir>` 两处都符合拆分形态,**绝不**让合并文件混进交付结果。

```bash
# 对 <UserOutputDir> 做严格校验(最重要;交付产物)
HOST_DIR="${USER_DIR}/op_host"
for F in "${OP_NAME,,}_tiling.h" "${OP_NAME,,}_tiling.cpp" "${OP_NAME,,}_def.cpp" "${OP_NAME,,}_infershape.cpp"; do
    [ -f "${HOST_DIR}/${F}" ] || { echo "[FATAL] <UserOutputDir>/op_host/ 缺少拆分文件 ${F}"; exit 1; }
done
[ -f "${HOST_DIR}/${OP_NAME,,}.cpp" ] && { echo "[FATAL] <UserOutputDir>/op_host/${OP_NAME,,}.cpp 不应存在(必须拆成 _def/_tiling/_infershape 三个 cpp)"; exit 1; }

# 内容错位自检:OpDef / Tiling / InferShape 必须各就各位
rg -q "class\s+${OP_NAME}\s*:\s*public\s+\w*OpDef\b" "${HOST_DIR}/${OP_NAME,,}_def.cpp" \
    || { echo "[FATAL] OpDef 定义未出现在 ${OP_NAME,,}_def.cpp"; exit 1; }
rg -q "IMPL_OP_OPTILING\s*\(\s*${OP_NAME}\s*\)" "${HOST_DIR}/${OP_NAME,,}_tiling.cpp" \
    || { echo "[FATAL] IMPL_OP_OPTILING 未出现在 ${OP_NAME,,}_tiling.cpp"; exit 1; }
rg -q "IMPL_OP_INFERSHAPE\s*\(\s*${OP_NAME}\s*\)" "${HOST_DIR}/${OP_NAME,,}_infershape.cpp" \
    || { echo "[FATAL] IMPL_OP_INFERSHAPE 未出现在 ${OP_NAME,,}_infershape.cpp"; exit 1; }
rg -q "class\s+${OP_NAME}\s*:\s*public\s+\w*OpDef\b" "${HOST_DIR}/${OP_NAME,,}_tiling.cpp" \
    && { echo "[FATAL] OpDef 误写在 ${OP_NAME,,}_tiling.cpp,需搬回 _def.cpp"; exit 1; }
rg -q "IMPL_OP_OPTILING" "${HOST_DIR}/${OP_NAME,,}_def.cpp" \
    && { echo "[FATAL] IMPL_OP_OPTILING 误写在 ${OP_NAME,,}_def.cpp,需搬到 _tiling.cpp"; exit 1; }

# 同时核验 <VerifyProjectDir>:msopgen 的合并 <op>.cpp 必须已在 B.2 被 rm 掉
VERIFY_HOST="${VERIFY_DIR}/op_host"
[ -f "${VERIFY_HOST}/${OP_NAME,,}.cpp" ] && { echo "[FATAL] <VerifyProjectDir>/op_host/${OP_NAME,,}.cpp 未被删除;B.2 步骤漏了 rm -f"; exit 1; }
for F in "${OP_NAME,,}_tiling.h" "${OP_NAME,,}_tiling.cpp" "${OP_NAME,,}_def.cpp" "${OP_NAME,,}_infershape.cpp"; do
    [ -f "${VERIFY_HOST}/${F}" ] || { echo "[FATAL] <VerifyProjectDir>/op_host/ 缺少拆分文件 ${F};B.2 cp -rf 未成功"; exit 1; }
done

echo "[OK] host 文件拆分校验通过"
```

**失败自愈流程**(任一点未通过):

1. **不要**把错写位置的代码"留在原地补丁" — 直接把整段搬回正确文件
2. 若生成阶段就写成合并 `<op>.cpp`:**不修改该文件**,而是:
   - 新建 `<op>_def.cpp`:只包含 `class <Op> : public OpDef {...}` + `OP_ADD(<Op>)`
   - 新建 `<op>_tiling.cpp`:只包含 `TilingFunc` + `IMPL_OP_OPTILING`
   - 新建 `<op>_infershape.cpp`:只包含 `InferShape` / `InferDataType` + `IMPL_OP_INFERSHAPE`
   - 删除原 `<op>.cpp`
3. 回 B.2.1 再跑;通过才进 B.3.0

### B.3.0 编译前 Kernel 语义完整性预检([R-054](rules.md#r-054kernel-计算逻辑原样迁移禁止改写最高优先级凌驾于任何风格偏好) / [R-055](rules.md#r-055kernel-语义完整性静态校验编译前执行))

**目的**:拦截"agent 自作主张把 kernel 重写/简化/合并"这类**编译能过、数值会错**的隐形缺陷。核心原则:`.asc` 的 kernel 计算逻辑是**黑盒**,本 skill **只做搬运不做改写**(详见 [R-054])。

```bash
ASC_FILE="<UserOutputDir>/../rms_norm.asc"    # 或用户指定的原始 .asc 路径
KERNEL_DIR="${USER_DIR}/op_kernel"

# (1) 类名存在性:原 .asc 里每个 `class <ClassName>` 都必须在 op_kernel 里找到
MISSING_CLASS=0
for C in $(rg -oN "class\s+(\w+)" "${ASC_FILE}" -r '$1' | sort -u); do
    rg -q "\b${C}\b" "${KERNEL_DIR}" || { echo "[FATAL] kernel class ${C} 丢失"; MISSING_CLASS=$((MISSING_CLASS+1)); }
done

# (2) __aicore__ inline 成员函数名集合必须完整保留
MISSING_FN=0
for F in $(rg -oN "__aicore__\s+inline\s+\w[\w:<>\s,\*&]*?\s+(\w+)\s*\(" "${ASC_FILE}" -r '$1' | sort -u); do
    rg -q "\b${F}\s*\(" "${KERNEL_DIR}" || { echo "[FATAL] kernel 成员函数 ${F} 丢失"; MISSING_FN=$((MISSING_FN+1)); }
done

# (3) AscendC 计算 API 调用次数单调不减(允许为 schMode 分支复制粘贴,不允许删减)
API_SHRINK=0
for API in ReduceSum Mul Div Add Sub Sqrt Rsqrt Cast Duplicate Brcb DataCopyPad DataCopy; do
    N_ASC=$(rg -c "\b${API}\s*\(" "${ASC_FILE}" 2>/dev/null || echo 0)
    N_NEW=$(rg -c "\b${API}\s*\(" "${KERNEL_DIR}" 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
    if [ "${N_NEW}" -lt "${N_ASC}" ]; then
        echo "[FATAL] ${API} 在新 kernel 中出现 ${N_NEW} 次,原 .asc ${N_ASC} 次,有计算被删/合并"
        API_SHRINK=$((API_SHRINK+1))
    fi
done

# (4) 禁止把 main / ReadConfig / KernelCall 搬进 op_kernel/
HOST_RESIDUE=$(rg -c "\b(main|ReadConfig|KernelCall)\s*\(" "${KERNEL_DIR}" 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')

if [ ${MISSING_CLASS} -gt 0 ] || [ ${MISSING_FN} -gt 0 ] || [ ${API_SHRINK} -gt 0 ] || [ ${HOST_RESIDUE} -gt 0 ]; then
    echo "[FATAL] kernel 语义完整性校验未通过,进入失败自愈流程"
    exit 1
fi
```

**失败自愈流程**:

1. **不要**去改这个校验脚本(改脚本适配错误的迁移 == 欺骗自己)
2. 打开原 `.asc` 的 kernel 段(`class <ClassName>` 到 `KernelCall` 之前),**整段复制**到 `op_kernel/<op>.h` / `op_kernel/<op>.cpp`
3. 只做 [R-054] 允许的 7 类"形式适配"(include / 入口签名 / `GET_TILING_DATA` / `TILING_KEY_IS` / `AscendC::` 前缀 / `bfloat16_t` / STL 脱敏),**其它一律不动**
4. 回到 B.3.0 再跑;通过才进 B.3(OpDef 预检);连续两次仍失败 → 停止,告诉用户 `.asc` 结构需要先分段,手工协助
5. 尤其要避免的"过度迁移"模式:
   - ❌ 把 `1.0f / sqrt(x)` 改成 `rsqrt(x)`——**数学等价 ≠ 舍入等价**
   - ❌ 把多个 `Mul + Div` 合并成一个表达式——**改了累加顺序**
   - ❌ 把 `template <uint32_t schMode>` 模板实例化去掉、换成运行时 if——**改了编译期分发**
   - ❌ 自己加辅助 lambda 或 `constexpr` helper——**凭空增量**
   - ❌ 把 `.asc` 里原有的注释删掉"显得干净"——**注释里可能藏有 tiling 契约**

### B.3 编译前 OpDef 静态预检([R-049](rules.md#r-049编译循环前先跑-opdef-静态预检))

专门拦截"OpDef 空壳"这类编译期不报、运行期才炸的缺陷(对应 [Case 6](cases.md#case-6op_hostop_defcpp-空壳最隐蔽最常见的偷懒型失败))。

```bash
DEF_FILE="${USER_DIR}/op_host/${OP_NAME,,}_def.cpp"

IN_CNT=$(rg -c "\.Input\(" "${DEF_FILE}" || echo 0)
OUT_CNT=$(rg -c "\.Output\(" "${DEF_FILE}" || echo 0)
EMPTY_DT=$(rg -c "\.DataType\(\s*\{\s*\}\s*\)" "${DEF_FILE}" || echo 0)
EMPTY_FMT=$(rg -c "\.Format\(\s*\{\s*\}\s*\)" "${DEF_FILE}" || echo 0)
PLACEHOLDER=$(rg -c "<DType[0-9]+>|<Format[0-9]+>|<soc>|<Op>|<op>|TODO:" "${DEF_FILE}" || echo 0)
AIC_OK=$(rg -c 'AICore\(\)\.AddConfig\("ascend[0-9a-z_]+"' "${DEF_FILE}" || echo 0)
ADD_OK=$(rg -c "OP_ADD\(" "${DEF_FILE}" || echo 0)
```

**判定**(任一不通过 → **整文件重写**,**不进**编译循环):

| 判定 | 通过标准 | 规则 |
| ---- | ---- | ---- |
| `IN_CNT >= 1` | `.Input(` 出现 ≥ 1 | [R-028](rules.md#r-028opdef-完工九点自检) |
| `OUT_CNT >= 1` | `.Output(` 出现 ≥ 1 | [R-028](rules.md#r-028opdef-完工九点自检) |
| `EMPTY_DT == 0` | 无空 `.DataType({})` | [R-028](rules.md#r-028opdef-完工九点自检) |
| `EMPTY_FMT == 0` | 无空 `.Format({})` | [R-028](rules.md#r-028opdef-完工九点自检) |
| `PLACEHOLDER == 0` | 无占位符残留 | [R-029](rules.md#r-029禁止任何占位符残留) |
| `AIC_OK >= 1` | `AICore().AddConfig("ascend<family>", ...)` | [R-034](rules.md#r-034soc-用系列名三处一致), [R-035](rules.md#r-035opaicoreconfig-六-flag--extendcfginfo) |
| `ADD_OK == 1` | 末尾 `OP_ADD(<Op>);` | [R-036](rules.md#r-036op_addop-末尾一行) |

**失败自愈流程**:

1. 立即打开 `<UserOutputDir>/op_host/<op>_def.cpp` 读全文
2. 回放 [R-027 契约表](rules.md#r-027开工前必填-opdef-契约表);契约表没填完先补
3. 按契约表**整文件重写** `<op>_def.cpp`(不做局部打补丁)
4. 回到 B.3 再跑;通过才进 B.4;**禁止**连续失败 3 次仍进编译循环

> 说明:B.3.0(kernel 语义完整性)和 B.3(OpDef 静态预检)是两道独立的硬门,**必须**按 B.3.0 → B.3 顺序通过才进 B.4。任一门失败都**不得**直接跳到编译循环碰运气。

### B.4 编译循环(最多 5 轮)

```bash
cd "${VERIFY_DIR}"
for i in $(seq 1 5); do
    echo "=== build attempt #$i ==="
    bash build.sh 2>&1 | tee "build_attempt_$i.log"
    if [ -f build_out/custom_opp_*.run ]; then
        echo "BUILD OK"
        break
    fi
done
```

**每次失败后**:

1. **读 `build_attempt_$i.log` 的最后 200 行**
2. 对照下面"错误关键字 → 修复动作 → 规则引用"表定位
3. **优先修 `<UserOutputDir>` 下的原文件**(用户最终成果),再 `cp -rf` 同步到 `<VerifyProjectDir>`
4. 重跑 `bash build.sh`

### B.5 错误关键字 → 修复动作 → 规则引用

| 错误关键字 | 修复动作(定位到 `<UserOutputDir>` 下的文件) | 规则 / 案例 |
| ---- | ---- | ---- |
| `adv_api/... No such file or directory` / `lib/... No such file or directory` | `op_host/<op>_tiling.cpp` / `op_host/<op>_tiling.h` 中 `#include "adv_api/..."` 全部替换为 `#include "tiling/tiling_api.h"` | [R-015](rules.md#r-015host-侧-ascendc-tiling-api-只能来自-tilingtiling_apih), [Case 1](cases.md#case-1host-侧盲目照搬-asc-的-adv_api-等-sdk-私有-include) |
| `free(): double free detected in tcache 2` / `malloc_consolidate()` | `op_host/<op>_tiling.cpp` 删除 `XxxTiling local = {};` 与后续 `tiling.fieldName = local;` / `tiling.set_fieldName(local);`,把 `tiling.fieldName` 作 OUT 引用直接传给 `AscendC::GetXxxTilingInfo(..., tiling.fieldName, ...)` | [R-009](rules.md#r-009sdk-官方-tiling-结构用-tiling_data_field_def_struct-直接嵌入), [Case 2](cases.md#case-2tiling_data_field_def_struct-嵌套-sdk-tilinglocal--拷贝-触发-double-free) |
| `Var: SCH_MODE_X in TILING_KEY_IS(...) can not be processed as numeric variables in the precompilation phase` | `op_kernel/<op>.cpp` 把 `TILING_KEY_IS(SCH_MODE_X)` 改成数字字面量 `TILING_KEY_IS(N)` | [R-020](rules.md#r-020tiling_key_is-用数字字面量), [Case 3](cases.md#case-3tiling_key_is-用数字字面量不用-constexpr) |
| `'const struct gert::Shape' has no member named 'GetStorageShape'` / `'gert::InferShapeContext' has no member named 'SetOutputShape'` | `op_host/<op>_infershape.cpp` 按 [Case 4](cases.md#case-4infershape-与-tiling-混用-shape-apigetstorageshape--setoutputshape-不存在) 模板改写 | [R-038](rules.md#r-038gertinfershapecontext-专用-api), [Case 4](cases.md#case-4infershape-与-tiling-混用-shape-apigetstorageshape--setoutputshape-不存在) |
| `'ND' is not a member of 'ge::Format'` / `cannot convert ... to std::vector<ge::Format>` | `op_host/<op>_def.cpp` 把 `ge::Format::ND` / `"ND"` 改为 `ge::FORMAT_ND` | [R-031](rules.md#r-031format-用-geformat_nd) |
| `The soc version of op <Op> is not configured` / `cannot find chip config for ai_core-...` / `ASCEND_COMPUTE_UNIT` 未匹配 / `device target is empty` | 四处统一改成**系列名**(全小写+无尾部数字,如 `ascend910b` / `ascend310p`):`op_host/<op>_def.cpp` 的 `AICore().AddConfig`、`msopgen gen -c ai_core-<SOC>`、`<VerifyProjectDir>/CMakePresets.json` 的 `ASCEND_COMPUTE_UNIT`、`op_host/config/<soc>/` 目录名。**禁用** `ascend910b4` / `Ascend910B3` 等具体型号 | [R-034](rules.md#r-034soc-用系列名三处一致) |
| `no member named 'min' in namespace 'std'` / `'std::max' has not been declared` | `op_kernel/<op>.*` 把 `std::min(a,b)` 改三元 `(a<b)?a:b`;删 `#include <algorithm>` 等 STL 头 | [R-021](rules.md#r-021device-端禁用-stl) |
| `use of undeclared identifier 'GetBlockIdx'` / `'GetBlockNum'` | `op_kernel/<op>.*` 加 `AscendC::` 前缀 | [R-022](rules.md#r-022device-端内建函数加-ascendc-前缀) |
| `unknown type name 'bfloat16'; did you mean 'bfloat16_t'?` | `op_kernel/<op>.*` 所有 C++ 类型位置裸 `bfloat16` 改为 `bfloat16_t`(**不改** host 端 `op_host/` 下字符串字面量 `"bfloat16"` 与注释) | [R-023](rules.md#r-023device-端-c-数据类型用实现名), [Case 5](cases.md#case-5kernel-侧-bf16-类型名统一用-bfloat16_t) |
| `'OpDef' has not been declared` / `class '<Op>' does not have any field named 'OpDef'` | `op_host/<op>_def.cpp`:把 `class <Op> : public OpDef {};` 包进 `namespace ops { ... }`,或改为 `public ops::OpDef` | [R-030](rules.md#r-030opdef-在-ops-命名空间作用域内可解析) |
| `undefined macro GET_TILING_DATA_WITH_STRUCT` / `REGISTER_TILING_DEFAULT` | 误用非 msOpGen 标准宏。`op_kernel/<op>.cpp` 改回 `GET_TILING_DATA(tilingData, tiling)`;`op_host/<op>_tiling.h` 用 `BEGIN_TILING_DATA_DEF` + `REGISTER_TILING_DATA_CLASS` | [R-001](rules.md#r-001唯一目标框架--cann-标准-msopgen), [R-005](rules.md#r-005tilingdata-唯一定义于-hostkernel-侧不得重复声明), [R-010](rules.md#r-010register_tiling_data_class-注册) |
| msopgen 输出目录下**没有** `op_host/` + `op_kernel/`,出现 `impl/` / `tbe/` / `op_info_cfg/` / `.py` 文件 | msopgen 落回 TBE DSL 旧模板(-lan 默认 `py`)。**禁止** `rm -rf "${VERIFY_DIR}"`([R-058]),改为 `mv "${VERIFY_DIR}" "${VERIFY_DIR}.badstructure_$(date +%s)"` 重命名备份,然后用**新时间戳**的 VERIFY_DIR 重跑 `msopgen gen` 并**显式**写 `-lan cpp`;确认命令里没被 shell 拆行/引号问题吞掉 | [R-053](rules.md#r-053msopgen-gen-必须显式带--lan-cpp-并校验生成结构) / [R-058](rules.md#r-058useroutputdir-与-verifyprojectdir-必须独立共存禁止清理验证目录) |
| **编译成功但 aclnn 运行数值与原 `.asc` 直调结果不匹配**(精度误差 / NaN / 量级偏差) | 典型信号:agent 把 kernel 做了"语义等价重写"。用 B.3.0 的 (1)(2)(3) 三条 grep 对 `.asc` 和 `op_kernel/` 逐项比对;**找到差异段 → 从 `.asc` 原文整段重贴,仅做 [R-054] 允许的 7 类形式适配**。**禁止**通过"调 eps / 改累加顺序 / 换 `rsqrt`"来掩盖 | [R-054](rules.md#r-054kernel-计算逻辑原样迁移禁止改写最高优先级凌驾于任何风格偏好), [R-055](rules.md#r-055kernel-语义完整性静态校验编译前执行) |
| `TILING_DATA_FIELD_DEF ... is not POD` | `op_host/<op>_tiling.h` 把嵌入的用户 struct 字段扁平化,或改用 `TILING_DATA_FIELD_DEF_STRUCT` 嵌入 SDK 结构 | [R-008](rules.md#r-008字段类型约束), [R-009](rules.md#r-009sdk-官方-tiling-结构用-tiling_data_field_def_struct-直接嵌入) |
| `undefined reference to TilingFunc` / `<Op>::<Op>` | 算子名不对齐;检查 `IMPL_OP_OPTILING(<Op>).Tiling(TilingFunc);` / `OP_ADD(<Op>);` / `REGISTER_TILING_DATA_CLASS(<Op>, <Op>TilingData)` 三处严格一致 | [R-003](rules.md#r-003算子名一处大驼峰全局对齐), [R-010](rules.md#r-010register_tiling_data_class-注册), [R-013](rules.md#r-013tilingfunc-注册), [R-036](rules.md#r-036op_addop-末尾一行) |
| `The input/output/attr of op <Op> is not configured` / `Failed to get op definition of <Op>` / `op proto of <Op> is empty` / (aclnn 运行时) `<Op>NotRegistered` / `Op<Op>:GetOpInfoFailed` | OpDef 空壳;`<UserOutputDir>/op_host/<op>_def.cpp` 按 Case 6 的九点自检**整文件重写**。**禁止**保留 `<DType1>` / `<soc>` / `// TODO` | [R-027](rules.md#r-027开工前必填-opdef-契约表) ~ [R-036](rules.md#r-036op_addop-末尾一行), [Case 6](cases.md#case-6op_hostop_defcpp-空壳最隐蔽最常见的偷懒型失败) |
| 其他未列出 | 查看完整日志,grep `error:` / `fatal error:` / `ld:`,对照 [rules.md](rules.md) + [cases.md](cases.md);无法定位 → 跳出循环,把日志尾部返给用户 | —— |

**循环终止**(对应 [R-050](rules.md#r-050编译循环最多-5-轮--错误映射表)):

- ✅ `build_out/` 下出现 `custom_opp_*.run` → 进 (C)
- ❌ 连续 5 轮仍失败,或同错连续 2 轮未解决 → 停止,告知用户

---

### B.6 阶段 6:安装算子包([R-059](rules.md#r-059阶段-6-必须直接执行-custom_opp-run-安装算子包))

编译成功后必须直接执行 `.run` 安装包。阶段 6 失败不得进入阶段 7。

```bash
cd "${VERIFY_DIR}"
RUN_PKG="$(ls build_out/custom_opp_*.run | head -n 1)"
[ -n "${RUN_PKG}" ] || { echo "[FATAL] 未找到 custom_opp_*.run,阶段 5 未完成"; exit 1; }

"./${RUN_PKG}" 2>&1 | tee install_custom_opp.log
INSTALL_RC=${PIPESTATUS[0]}
[ "${INSTALL_RC}" -eq 0 ] || { echo "[FATAL] 算子包安装失败,见 ${VERIFY_DIR}/install_custom_opp.log"; exit 1; }
```

**判定**:

- 成功:安装命令返回 0,`install_custom_opp.log` 保留,进入 B.7
- 失败:优先检查 CANN 环境变量、安装权限、旧自定义包冲突、`ASCEND_CUSTOM_OPP_PATH` / `ASCEND_OPP_PATH`;修复后重跑 B.6

### B.7 阶段 7:二进制一致性验证([R-060](rules.md#r-060阶段-7-必须生成-example-并用原直调输入做-aclnn-二进制一致性验证))

阶段 7 在最终 msopgen 工程下生成 `example/`,编译运行 `opapi_test`,并把 aclnn 输出与原 kernel 直调输出做 byte-level compare。

```bash
DIRECT_DIR="<DirectCallProjectDir>"                  # 例如 rms_norm
ORIGIN_INPUT_DIR="${DIRECT_DIR}/input"
ORIGIN_OUTPUT_DIR="${DIRECT_DIR}/output"
EXAMPLE_DIR="${VERIFY_DIR}/example"
ACLNN_OUTPUT_DIR="${EXAMPLE_DIR}/aclnn_output"

mkdir -p "${EXAMPLE_DIR}" "${ACLNN_OUTPUT_DIR}"
cp -rf "${ORIGIN_INPUT_DIR}" "${EXAMPLE_DIR}/input"
cp -rf "${ORIGIN_OUTPUT_DIR}" "${EXAMPLE_DIR}/output"
```

1. **生成或复用 aclnn harness**  
   在 `${EXAMPLE_DIR}` 生成 `main.cpp` 和 `CMakeLists.txt`。`main.cpp` 的关键内容从 `rms_norm_verify_project_20260425_181328/example/main.cpp` 提取:文件读写、`GetShapeSize`、`Init`、`CreateAclTensor`、workspace 查询、aclnn 执行、stream 同步、device-to-host 拷贝、资源释放。

   `main.cpp` 必须按算子名拼接 aclnn 接口:

   ```cpp
   // <Op> = RmsNorm 时:
   #include "aclnn_rms_norm.h"
   ret = aclnnRmsNormGetWorkspaceSize(/* inputs */, /* attrs */, /* outputs */, &workspaceSize, &executor);
   ret = aclnnRmsNorm(workspaceAddr, workspaceSize, executor, stream);
   ```

   实际输入/输出 tensor、attr 顺序必须来自阶段 1 的 OpDef 契约表和原 `.asc` / 直调 `main`。不得照搬 RmsNorm 的参数到其它算子。

2. **抽取维度 / dtype / attr**  
   测试输入输出维度优先从原直调数据脚本抽取。例如 RmsNorm 读取 `rms_norm/scripts/gen_data.py`:

   - `rows` → total rows / A1
   - `cols` → reduction length / R
   - `epsilon` → aclnn attr
   - `dtype` → aclTensor dtype 与输入文件 element size
   - `input/input_x.bin`、`input/input_gamma.bin` → aclnn 输入
   - `output/golden_y.bin`、`output/golden_rstd.bin` → 二进制比较基准
   - `r_length_align = ceil(cols / align_unit) * align_unit` → 文件读取/输出 shape 需要匹配直调对齐

   若原工程另有 `params.txt`、`run.sh` 或 case 配置,优先使用这些实际生成数据时落盘的参数,不得猜 shape、dtype 或 attr。

3. **生成 `example/CMakeLists.txt`**  
   CMake 模板见 [reference.md](reference.md#examplecmakeliststxt)。必须生成可执行文件 `opapi_test`,通过 `ASCEND_CUSTOM_PATH` 或 `/usr/local/Ascend/cann` 定位 CANN,并显式链接 `libascendcl.so`、`libnnopbase.so`、`libopapi_math.so`、`libopapi_nn.so`、`libcust_opapi.so`。

4. **编译并运行 `opapi_test`**  

   ```bash
   cd "${VERIFY_DIR}/example"
   mkdir -p build
   cd build
   cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE 2>&1 | tee cmake_example.log
   make 2>&1 | tee make_example.log
   cd bin
   ./opapi_test 2>&1 | tee ../../opapi_test.log
   ```

   如果 `opapi_test` 需要 case 参数,参数也必须来自原直调工程的 `run.sh` / 配置 / `gen_data.py`,不要临时发明。

5. **逐文件二进制比较**  

   ```bash
   cd "${EXAMPLE_DIR}"
   : > binary_compare.log
   FAIL=0
   for REF in output/*.bin; do
       NAME="$(basename "${REF}")"
       NEW="${ACLNN_OUTPUT_DIR}/${NAME}"
       if [ ! -f "${NEW}" ]; then
           echo "[FATAL] 缺少 aclnn 输出 ${NEW}" | tee -a binary_compare.log
           FAIL=$((FAIL+1))
           continue
       fi
       if cmp -s "${REF}" "${NEW}"; then
           sha256sum "${REF}" "${NEW}" | tee -a binary_compare.log
           echo "[OK] ${NAME} binary identical" | tee -a binary_compare.log
       else
           echo "[FATAL] ${NAME} binary differs" | tee -a binary_compare.log
           cmp -l "${REF}" "${NEW}" | head -n 20 | tee -a binary_compare.log
           FAIL=$((FAIL+1))
       fi
   done
   [ "${FAIL}" -eq 0 ] || exit 1
   ```

**不一致处理**:

1. 先确认 harness 的输入 dtype、输出 dtype、shape、attr 与原直调完全一致
2. 确认 `example/input/` 和 `example/output/` 确实来自原 kernel 直调,`opapi_test` 输出写到了 `example/aclnn_output/` 而不是覆盖基准
3. 再跑 B.3.0 kernel 语义完整性预检,按 [R-054](rules.md#r-054kernel-计算逻辑原样迁移禁止改写最高优先级凌驾于任何风格偏好) 回原 `.asc` 整段重搬差异段
4. 禁止用 rtol/atol 容差、文件大小相同或前几个数相同来放行;byte-level 不等即阶段 7 失败

---

## (C) 成功/失败报告

### C.1 构建成功

agent 最终回复**必须**包含以下**双目录交付清单**([R-058]):

- **`<UserOutputDir>`(交付件 1 / 纯净源码)**:绝对路径 + 内部产物清单(`op_host/<op>_{tiling.h,tiling.cpp,def.cpp,infershape.cpp}` + `op_kernel/<op>.{h,cpp}` + `op_host/config/<soc>/*` + `<op>.json`)
- **`<VerifyProjectDir>`(交付件 2 / 编译证据链)**:绝对路径 + `build_out/custom_opp_*.run` 的完整路径;**明确告知用户此目录已原封保留**,可随时 `cd <VerifyProjectDir> && bash build.sh` 复现编译;**禁止**建议用户清理此目录([R-058])
- **阶段 6 安装结果**:`install_custom_opp.log` 路径 + 安装命令返回码
- **阶段 7 一致性结果**:`example/` 路径 + `example/build/bin/opapi_test` 路径 + `example/aclnn_output/` 路径 + `example/binary_compare.log` 路径 + 是否所有 `.bin` byte-level identical
- 本轮自动修复**实际触发**的规则 / 案例清单(如"触发了 [Case 3](cases.md#case-3):把 `TILING_KEY_IS(SCH_MODE_X)` 改成字面量"),便于用户复盘
- 若修复过程中产生过 `${VERIFY_DIR}.badstructure_*` / `${VERIFY_DIR}.bak_*` 等备份目录,一并列出(也属交付证据的一部分)
- 阶段 7 harness 仅用于二进制一致性验证([R-051](rules.md#r-051skill-只生成阶段-7-二进制一致性验证-harness)),不扩展为泛化精度测试框架

### C.2 达到重试上限仍失败

agent **必须**:

- 明确说明"已尝试 N 轮自动修复仍失败"
- 给出最后一次 `build_attempt_N.log` 的**尾部 100 行**(重点保留 `error:` / `fatal error:` / 关键 grep 上下文)
- 列出每轮**尝试过的**修复动作与结果
- **同时列出** `<UserOutputDir>` 和 `<VerifyProjectDir>` 的绝对路径 — 两者**都**作为本次交付件保留供用户人工接手分析([R-058])
- **禁止** `rm -rf "${VERIFY_DIR}"`;编译失败时 `<VerifyProjectDir>/build_out/` 下的 `build_attempt_*.log` 是用户人工定位问题的第一手证据,必须原封保留([R-058])
- 若失败发生在阶段 6 / 7,给出 `install_custom_opp.log` / `binary_compare.log` 的关键尾部,并明确说明未完成安装或未通过二进制一致性验证

---

## (D) 降级:环境不可用时的人工验证手册

若 (A) 探测**失败**,agent **不得**尝试 `msopgen` / `bash build.sh`(失败且无意义)。直接把以下内容作为最终回复的一部分输出给用户。

**严禁**在示例路径里使用 `/tmp/xxx`,始终用 `<OutputProjectDir>` 占位,提示用户替换成自定义持久化目录。

---

> ⚠️ **当前环境未检测到 `msopgen` 可执行文件**(已尝试 `command -v msopgen`、`${ASCEND_HOME_PATH}` / `${ASCEND_TOOLKIT_HOME}` / `${INSTALL_DIR}` 以及 `/usr/local/Ascend/ascend-toolkit/latest/...` 等常见路径)。本次**无法**自动完成 "msopgen 工程生成 → 文件替换 → 编译验证" 闭环。
>
> 本 skill 已在 `<UserOutputDir>` 下生成全部源文件。请你**在一台已安装 CANN 工具包的机器上**(Linux + Ascend NPU 或 CANN 开发镜像)手动完成以下验证。本次由于环境不可用,未完成编译、安装和二进制一致性验证。
>
> ### 1. 使用 msopgen 生成标准算子工程骨架
>
> 先**自行选择一个持久化目录**作为算子工程根目录(例如 `~/projects/<Op>Proj`、`$HOME/work/<Op>Proj`;不要用 `/tmp` 这类会被清理的临时路径),下文用 `<OutputProjectDir>` 代指:
>
> ```bash
> ${INSTALL_DIR}/python/site-packages/bin/msopgen gen \
>     -i <UserOutputDir>/<op>.json \
>     -c ai_core-<soc_family> \
>     -lan cpp \
>     -out <OutputProjectDir>
> ```
>
> 其中:
> - `${INSTALL_DIR}`:CANN 安装路径(如 `/usr/local/Ascend/ascend-toolkit/latest`)
> - `-i`:本 skill 生成的 `<UserOutputDir>/<op>.json`
> - `-c ai_core-<soc_family>`:**CANN 芯片系列名**(family),**不是**具体型号。示例 `ai_core-ascend910b` / `ai_core-ascend310p`——全小写,**不带**尾部数字(不是 `ascend910b4` / `Ascend910B3`)。`npu-smi info` 的 Name(如 `Ascend910B3` / `Ascend910B4`)全小写并去尾部数字即得 `ascend910b`。合法系列名列表见 `${INSTALL_DIR}/compiler/data/platform_config/*.ini` 文件名
> - **`-lan cpp` 不能省略**([R-053](rules.md#r-053msopgen-gen-必须显式带--lan-cpp-并校验生成结构)):msopgen `-lan` **默认是 `py`**(走旧版 TBE Python DSL,产物为 `tbe/impl/*.py` + `op_info_cfg/`,与本 skill 生成的 `op_host/*.cpp` + `op_kernel/*.cpp` 完全不兼容)。命令跑完**立刻**检查 `<OutputProjectDir>/` 下是否同时存在 `op_host/` 与 `op_kernel/`,若看到 `tbe/` / `impl/` / `op_info_cfg/`,说明 `-lan cpp` 没生效,删掉目录重跑
> - `-out <OutputProjectDir>`:**用户自定义**的算子工程目录路径
>
> ### 2. 用本 skill 生成的源文件替换 msopgen 默认模板
>
> | 来源(本 skill 生成) | 目标(msopgen 工程) | 说明 |
> | ---- | ---- | ---- |
> | `<UserOutputDir>/op_host/<op>_tiling.h` | `op_host/<op>_tiling.h` | 覆盖 |
> | `<UserOutputDir>/op_host/<op>_tiling.cpp` | `op_host/<op>_tiling.cpp` | 新增(msopgen 默认把 tiling 合并在 `<op>.cpp`,需拆分) |
> | `<UserOutputDir>/op_host/<op>_def.cpp` | `op_host/<op>_def.cpp` | 新增(同上) |
> | `<UserOutputDir>/op_host/<op>_infershape.cpp` | `op_host/<op>_infershape.cpp` | 新增 |
> | `<UserOutputDir>/op_host/config/<soc>/` | `op_host/config/<soc>/` | 新增(`_binary.json` + `_simplified_key.ini`) |
> | `<UserOutputDir>/op_kernel/<op>.cpp` | `op_kernel/<op>.cpp` | 覆盖 |
> | `<UserOutputDir>/op_kernel/<op>.h` | `op_kernel/<op>.h` | 新增 |
> | `<UserOutputDir>/op_graph/` | `op_graph/`(如需图模式) | 新增(msopgen 默认不生成;仅图模式需要) |
>
> ### 3. 处理 msopgen 默认 `op_host/<op>.cpp`
>
> msopgen 默认在 `op_host/` 下生成合并文件 `<op>.cpp`(含 OpDef + InferShape + Tiling)。本 skill 已将其**拆分**为 `<op>_def.cpp` + `<op>_infershape.cpp` + `<op>_tiling.cpp`:
>
> - **直接删除** `<OutputProjectDir>/op_host/<op>.cpp`(原合并文件)
> - 把拆分后三个 cpp + `<op>_tiling.h` 全部放入 `<OutputProjectDir>/op_host/`
> - 若 msopgen 生成的 `op_host/CMakeLists.txt` 是 glob 收集 `*.cpp`(`aux_source_directory` / `file(GLOB ...)`),无需改;否则改成引用拆分后三个 cpp
>
> ### 4. 修改 `CMakePresets.json`
>
> 打开 `<OutputProjectDir>/CMakePresets.json`:
> - `ASCEND_CANN_PACKAGE_PATH`:改成本地 CANN 实际安装路径
> - `ASCEND_COMPUTE_UNIT.value`:**系列名**(全小写,无尾部数字),与 `msopgen gen -c ai_core-<soc_family>` 的 `<soc_family>`、以及 `op_host/<op>_def.cpp` 中 `AICore().AddConfig("<soc_family>", cfg)` **严格一致**。误写具体型号会让 cmake 找不到该芯片 ai_core 支持
>
> ### 5. 编译 / 打包 / 安装
>
> ```bash
> cd <OutputProjectDir>
> bash build.sh
> ./build_out/custom_opp_*.run     # 直接运行 .run,不要加 --quiet / --install
> ```
>
> ### 6. 二进制一致性验证
>
> 复用原 kernel 直调工程的 `input/` 作为 aclnn 输入,参考 `<OutputProjectDir>/example/main.cpp` 编写/调整 aclnn 调用 harness。harness 输出写到独立目录(如 `<OutputProjectDir>/aclnn_output/`),然后与原直调 `output/` 逐 `.bin` 比较:
>
> ```bash
> for ref in <DirectCallProjectDir>/output/*.bin; do
>     name="$(basename "$ref")"
>     cmp -s "$ref" "<OutputProjectDir>/aclnn_output/$name" || {
>         echo "[FATAL] $name binary differs"
>         exit 1
>     }
> done
> echo "[OK] all outputs are binary identical"
> ```
>
> ### 7. 若编译 / 安装 / 一致性验证失败
>
> 对照 [rules.md](rules.md)、[cases.md](cases.md)、[verification-checklist.md](verification-checklist.md) 的红灯表逐项排查(这些是 agent 在自动闭环模式下会自动修复的所有已知问题)。
>
> ### 参考
>
> - [CANN 创建算子工程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_0060.html)
> - [aclnn 接口文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/aolapi/context/ops-nn/)
