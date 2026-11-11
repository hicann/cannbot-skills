#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ==========================================================================================================
# Kirin（鸿蒙端侧）平台开发环境检查脚本
# 用途：检查 kirin 端侧算子开发三条链路的前置条件——开发环境 / CANN 环境与能力（环境就绪 + atc→omc、camodel sim 直调）/ 设备连接与识别
# 平台：Linux 宿主机（x86_64/aarch64）；设备检查段经 hdc 直连执行（hdc 环境视用户部署而定）
# 纪律：本脚本只做检查与报告，不做任何安装；缺失项给出「缺了什么 → 导致哪个功能不可用 → 如何补齐」
# 参考依据：
#   - 开发环境项：官方前提条件 + install_deps.sh 检查逻辑（version_ge/command -v）
#   - CANN 组件项：包形态对照经验（环境就绪判据 + 两条能力线判据）；atc 用法参考见 tool-atc skill
#   - 设备识别项：hdc shell "hidumper -c base" 取设备型号（芯片型号以 hdc 查询为准，未查到由用户指定）
# 用法：
#   bash check_kirin_env.sh                # 全量检查（含设备段）
#   bash check_kirin_env.sh --skip-device  # 跳过设备段（仅宿主机检查）
# 输出：能力矩阵 + 缺失清单；
# ==========================================================================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'; NC='\033[0m'
WARNINGS=0
FAILURES=0
SKIP_DEVICE=0
[ "$1" = "--skip-device" ] && SKIP_DEVICE=1

echo "Kirin 平台开发环境检查"
echo "=================================="

# 与官方 install_deps.sh 同逻辑的版本比较（来源：cann-recipes-harmony-infer）
version_ge() {
    IFS='.' read -r -a curr_arr <<< "$1"
    IFS='.' read -r -a req_arr <<< "$2"
    for ((i=0; i<${#req_arr[@]}; i++)); do
        local curr=${curr_arr[i]:-0}
        local req=${req_arr[i]}
        if (( curr > req )); then return 0
        elif (( curr < req )); then return 1
        fi
    done
    return 0
}

check_version() {  # check_version <名称> <命令> <取版本参数...> <要求版本>（要求版本不传入命令）
    local name="$1"; shift
    local req="${!#}"
    local cmd=("${@:1:$#-1}")
    local ver
    ver=$("${cmd[@]}" 2>/dev/null | head -1)
    ver=$(echo "$ver" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
    if [ -z "$ver" ]; then
        echo -e "${RED}✗ $name 未安装${NC}"
        echo -e "  补齐：见 references/kirin_platform_guide.md（前提条件/安装方法）"
        FAILURES=$((FAILURES + 1)); return 1
    fi
    if version_ge "$ver" "$req"; then
        echo -e "${GREEN}✓ $name = $ver（要求 >= $req）${NC}"
        return 0
    fi
    echo -e "${RED}✗ $name = $ver，低于要求 $req${NC}"
    echo -e "  补齐：见 references/kirin_platform_guide.md（前提条件/安装方法）"
    FAILURES=$((FAILURES + 1)); return 1
}

# ---------------------------------------------------------------
# 段 1：开发环境检查（官方前提条件）
# ---------------------------------------------------------------
echo ""
echo -e "${YELLOW}[1/3] 开发环境检查（依据官方前提条件）${NC}"

check_version "python3" python3 --version 3.7.0
check_version "gcc" gcc --version 7.3.0
check_version "cmake" cmake --version 3.16.0

if command -v pigz &> /dev/null; then
    check_version "pigz（可选，加速打包）" pigz --version 2.4
else
    echo -e "${YELLOW}⚠ pigz 未安装（可选，仅影响打包速度）${NC}"
fi

for tool in dos2unix gawk; do
    if command -v $tool &> /dev/null; then
        echo -e "${GREEN}✓ $tool 可用${NC}"
    else
        echo -e "${RED}✗ $tool 未安装（官方前提条件项）${NC}"
        echo -e "  补齐：见 references/kirin_platform_guide.md（前提条件/安装方法）"
        FAILURES=$((FAILURES + 1))
    fi
done

GLIBC_VER=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
if [ -n "$GLIBC_VER" ] && version_ge "$GLIBC_VER" 2.34; then
    echo -e "${GREEN}✓ glibc = $GLIBC_VER（要求 >= 2.34）${NC}"
else
    echo -e "${RED}✗ glibc = ${GLIBC_VER:-未知}，低于 2.34（官方前提条件）${NC}"
    FAILURES=$((FAILURES + 1))
fi

# Python 包（数据生成/onnx 构造/精度比对）
echo "--- Python 包（数据生成/模型构造/精度比对）---"
for pkg in numpy onnx; do
    if python3 -c "import $pkg" 2>/dev/null; then
        echo -e "${GREEN}✓ python 包 $pkg 可导入${NC}"
    else
        echo -e "${YELLOW}⚠ python 包 $pkg 不可导入（用例数据生成/onnx 构造/精度比对需要）${NC}"
        echo -e "  补齐：pip3 install $pkg"
        WARNINGS=$((WARNINGS + 1))
    fi
done
if python3 -c "import paramiko" 2>/dev/null; then
    echo -e "${GREEN}✓ python 包 paramiko 可导入（跨机 hdc 自动化可选依赖）${NC}"
else
    echo -e "${YELLOW}⚠ python 包 paramiko 不可导入（仅在跨机 hdc 自动化场景需要）${NC}"
fi

# ---------------------------------------------------------------
# 段 2：CANN 环境与能力检查（先环境就绪，再两条能力线独立判定）
# ---------------------------------------------------------------
echo ""
echo -e "${YELLOW}[2/3] CANN 环境与能力检查（环境就绪 → atc→omc / camodel sim 直调）${NC}"

if [ -z "$ASCEND_HOME_PATH" ]; then
    echo -e "${RED}✗ ASCEND_HOME_PATH 未设置（未 source set_env.sh）${NC}"
    FOUND_ENVS=$(find /usr/local/Ascend "$HOME/Ascend" -maxdepth 3 -name set_env.sh 2>/dev/null | head -5)
    if [ -n "$FOUND_ENVS" ]; then
        echo -e "  检测到候选 set_env.sh（任选其一 source 后重跑本脚本更准确）："
        while IFS= read -r env_line; do
            echo "    $env_line"
        done <<< "$FOUND_ENVS"
        PKG_ROOTS=""
        while IFS= read -r f; do PKG_ROOTS="$PKG_ROOTS $(dirname "$f")"; done <<< "$FOUND_ENVS"
        echo -e "  注：以下检查按候选包根探测（假定路径不含空格），结果可能不完整"
    else
        echo -e "  补齐：安装含完整 opp 与目标平台 simulator 的 CANN 包后 source <安装目录>/set_env.sh（见 references/kirin_platform_guide.md）"
        PKG_ROOTS="/usr/local/Ascend/ascend-toolkit/latest $HOME/Ascend/ascend-toolkit/latest"
    fi
    FAILURES=$((FAILURES + 1))
else
    PKG_ROOTS="$ASCEND_HOME_PATH"
fi

pick_arch_dir() {  # pick_arch_dir <CANN 包根>：优先返回 <root>/<arch>-linux，否则回退 <root>
    local root="$1" d
    for d in "$root/$(uname -m)-linux" "$root/x86_64-linux" "$root/aarch64-linux"; do
        [ -d "$d" ] && { echo "$d"; return; }
    done
    echo "$root"
}

echo "--- CANN 环境就绪 ---"
for root in $PKG_ROOTS; do
    [ -d "$root" ] || continue
    ARCH_DIR=$(pick_arch_dir "$root")
    MISS=""
    INFO=$(find "$root" -maxdepth 2 \( -name "ascend_toolkit_install.info" -o -name "ascend_all_cann_install.info" \) 2>/dev/null | head -1)
    [ -n "$INFO" ] || MISS="$MISS 安装信息文件(*install.info)"
    { [ -d "$ARCH_DIR/bin" ] || [ -d "$root/bin" ]; } || MISS="$MISS bin/"
    { [ -d "$ARCH_DIR/lib64" ] || [ -d "$root/lib64" ]; } || MISS="$MISS lib64/"
    { [ -d "$root/ccec_compiler" ] || [ -d "$ARCH_DIR/ccec_compiler" ]; } || MISS="$MISS ccec_compiler/"
    if [ -z "$MISS" ]; then
        echo -e "${GREEN}✓ CANN 环境就绪: $root（安装信息 + bin/lib64/ccec_compiler 齐备）${NC}"
    else
        echo -e "${RED}✗ CANN 环境不完整: $root（缺:$MISS）→ 组件定位/编译入口将失败${NC}"
        FAILURES=$((FAILURES+1))
    fi
done

ATC_OK=0; SIM_OK=0
declare -A ATC_DETAIL SIM_DETAIL

for root in $PKG_ROOTS; do
    ARCH_DIR=$(pick_arch_dir "$root")

    # ---- 能力线 1：atc→omc 转换 ----
    ATC_BIN=""
    for c in "$ARCH_DIR/bin/atc" "$root/bin/atc"; do
        [ -x "$c" ] && ATC_BIN="$c" && break
    done
    OPP_DIR="$ARCH_DIR/opp"
    [ -d "$OPP_DIR" ] || OPP_DIR="$root/opp"
    # 官方包形态：内置算子组件位于 opp/built-in/；旧形态可能在 opp/ 顶层，故 built-in 优先、顶层回退。
    # 图/原型组件在不同 CANN 版本有 op_graph / op_proto 两种命名，二者其一即视为齐备。
    OPP_BASE="$OPP_DIR"
    [ -d "$OPP_DIR/built-in" ] && OPP_BASE="$OPP_DIR/built-in"
    MISSING_OPP=""
    if ! { [ -d "$OPP_BASE/op_graph" ] || [ -d "$OPP_BASE/op_proto" ] \
        || [ -d "$OPP_DIR/op_graph" ] || [ -d "$OPP_DIR/op_proto" ]; }; then
        MISSING_OPP="$MISSING_OPP op_graph/op_proto"
    fi
    for d in op_impl framework; do
        { [ -d "$OPP_BASE/$d" ] || [ -d "$OPP_DIR/$d" ]; } || MISSING_OPP="$MISSING_OPP $d"
    done
    if [ -n "$ATC_BIN" ] && [ -z "$MISSING_OPP" ]; then
        ATC_OK=1
        ATC_DETAIL[$root]="${GREEN}atc ✓（$ATC_BIN）| opp 完整（$OPP_BASE 下 op_graph（旧版 op_proto）、op_impl、framework 齐备）${NC}"
    elif [ -n "$ATC_BIN" ] && [ -n "$MISSING_OPP" ]; then
        ATC_DETAIL[$root]="${RED}atc 可执行但 opp 不完整（$OPP_BASE 缺:$MISSING_OPP）→ atc 转 omc 将报 E20002（FE 初始化失败）${NC}"
    else
        ATC_DETAIL[$root]="${RED}atc 不可用${NC}"
    fi

    # ---- 能力线 2：camodel sim 直调 ----
    SIM_LIBS=0; SIM_SHELL=0; SIM_NAMES=""
    for platdir in "$ARCH_DIR"/simulator/*/; do
        [ -d "$platdir" ] || continue
        SIMLIB="${platdir%/}/lib"
        [ -f "$SIMLIB/libruntime_camodel.so" ] || continue
        SIM_LIBS=$((SIM_LIBS+1))
        SIM_NAMES="$SIM_NAMES $(basename "${platdir%/}")"
        [ -f "$SIMLIB/libascendcl.so" ] && SIM_SHELL=$((SIM_SHELL+1))
    done
    if [ "$SIM_LIBS" -gt 0 ] && [ "$SIM_SHELL" -ge "$SIM_LIBS" ]; then
        SIM_OK=1
        SIM_DETAIL[$root]="${GREEN}camodel sim 直调 ✓（$SIM_LIBS 个 simulator：$SIM_NAMES，含 camodel 版 libascendcl.so 壳）${NC}"
    elif [ "$SIM_LIBS" -gt 0 ]; then
        SIM_DETAIL[$root]="${RED}simulator 目录在但缺 camodel 版 libascendcl.so 壳（$SIM_SHELL/$SIM_LIBS）→ 直调将报 aclInit 500000${NC}"
    else
        SIM_DETAIL[$root]="${YELLOW}无 Kirin simulator 目录${NC}"
    fi
done

echo "--- 能力矩阵 ---"
for root in $PKG_ROOTS; do
    echo -e "包: $root"
    echo -e "  [atc→omc]        ${ATC_DETAIL[$root]}"
    echo -e "  [camodel sim直调] ${SIM_DETAIL[$root]}"
done
[ "$ATC_OK" = 0 ] && { echo -e "  缺什么：完整 opp（opp/built-in 下 op_graph（旧版 op_proto）、op_impl、framework）→ 哪个功能不可用：atc 转 omc → 如何补齐：安装 opp 完整的 CANN 包（references/kirin_platform_guide.md；判据见 tool-atc 报错排查 E20002）"; FAILURES=$((FAILURES+1)); }
[ "$SIM_OK" = 0 ] && { echo -e "  缺什么：含 camodel ACL 壳的 simulator → 哪个功能不可用：sim 直调（ST/UT 仿真执行）→ 如何补齐：更换含目标平台 simulator 的 CANN 包"; WARNINGS=$((WARNINGS+1)); }

# GLIBCXX 运行时：按目标库实际所需符号判定，不假设固定版本
echo "--- GLIBCXX 运行时 ---"
GXX_BIN=$(command -v g++)
if [ -n "$GXX_BIN" ] && [ -x "$GXX_BIN" ]; then
    SYS_GLIBCXX=$(strings "$(g++ -print-file-name=libstdc++.so.6)" 2>/dev/null </dev/null | grep -oE 'GLIBCXX_3\.4\.[0-9]+' | sort -uV | tail -1)
    if [ -n "$SYS_GLIBCXX" ]; then
        REQ_GLIBCXX=""
        if command -v readelf >/dev/null 2>&1; then
            for lib in "$ARCH_DIR"/simulator/*/lib/libruntime_camodel.so; do
                [ -f "$lib" ] || continue
                v=$(readelf -V "$lib" 2>/dev/null | grep -oE 'GLIBCXX_3\.4\.[0-9]+' | sort -uV | tail -1)
                [ -n "$v" ] && REQ_GLIBCXX=$(printf '%s\n%s\n' "$REQ_GLIBCXX" "$v" | sort -uV | tail -1)
            done
        fi
        if [ -n "$REQ_GLIBCXX" ] && version_ge "${SYS_GLIBCXX#GLIBCXX_}" "${REQ_GLIBCXX#GLIBCXX_}"; then
            echo -e "${GREEN}✓ 系统 libstdc++ ${SYS_GLIBCXX} >= simulator 所需 ${REQ_GLIBCXX}${NC}"
        elif [ -n "$REQ_GLIBCXX" ]; then
            echo -e "${YELLOW}⚠ 系统 libstdc++ ${SYS_GLIBCXX} < simulator 所需 ${REQ_GLIBCXX}：camodel 直调需 LD_PRELOAD 满足该符号的 libstdc++，或使用自带 ext_lib 的 harness${NC}"
            WARNINGS=$((WARNINGS + 1))
        else
            echo -e "${GREEN}✓ 系统 libstdc++ 最高 ${SYS_GLIBCXX}（未从 simulator 库解析到更高需求）${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ 无法探测系统 libstdc++（g++ 缺失或库不可读）；camodel 直调若报 GLIBCXX not found，用 LD_PRELOAD 提供所需符号${NC}"
        WARNINGS=$((WARNINGS + 1))
    fi
fi
# ---------------------------------------------------------------
# 段 3：设备检查（复用 tool-hdc skill；--skip-device 可跳过）
# ---------------------------------------------------------------
echo ""
echo -e "${YELLOW}[3/3] 设备检查（hdc）${NC}"
if [ "$SKIP_DEVICE" = 1 ]; then
    echo "（--skip-device，跳过）"
else
    HDC_BIN=""
    if command -v hdc &> /dev/null; then
        HDC_BIN="hdc"
    elif [ -n "$HDC_PATH" ] && [ -x "$HDC_PATH/hdc" ]; then  # HDC_PATH 为本脚本自定义覆盖项（指定 hdc 所在目录），非 CANN/hdc 官方环境变量
        HDC_BIN="$HDC_PATH/hdc"
    fi
    if [ -z "$HDC_BIN" ]; then
        echo -e "${YELLOW}⚠ 未找到 hdc，设备检查项标记为待确认；请确认 hdc 安装位置与环境（定位与自检见 tool-hdc skill: scripts/find_hdc.sh）${NC}"
        echo -e "  hdc 定位与自检脚本见 tool-hdc skill: scripts/find_hdc.sh"
        WARNINGS=$((WARNINGS + 1))
    else
        TARGETS=$($HDC_BIN list targets 2>/dev/null | grep -vE "^\[|Empty|total" | grep -E "[0-9A-Za-z]" | head -5)
        if [ -z "$TARGETS" ]; then
            echo -e "${YELLOW}⚠ hdc 可用但无设备在位（hdc list targets 为空）——请检查设备连接，或确认设备未被占用${NC}"
            WARNINGS=$((WARNINGS + 1))
        else
            echo -e "${GREEN}✓ 设备在位: $(echo "$TARGETS" | tr '\n' ' ')${NC}"
            SN=$(echo "$TARGETS" | head -1)
            # hidumper -c base 为主（MarketName/ProductModel/HardwareModel 等）；过滤在宿主机侧完成，
            # 勿把管道/重定向跟在 hdc shell 后（会被整体传入设备侧 shell）
            HID=$($HDC_BIN -t "$SN" shell "hidumper -c base" 2>/dev/null | tr -d '\r')
            PNAME=$(echo "$HID" | grep -m1 "MarketName" | awk -F': ' '{print $2}')
            PMODEL=$(echo "$HID" | grep -m1 "ProductModel" | awk -F': ' '{print $2}')
            [ -z "$PNAME" ] && PNAME=$($HDC_BIN -t "$SN" shell param get const.product.name 2>/dev/null | tr -d '\r\n')
            [ -z "$PMODEL" ] && PMODEL=$($HDC_BIN -t "$SN" shell param get const.product.model 2>/dev/null | tr -d '\r\n')
            echo "  设备型号: ${PNAME:-未知} / ${PMODEL:-未知}（SN=$SN）"
            echo -e "  ${BLUE}芯片型号：以 hdc 查询结果为准（hidumper 输出字段）；未查到则由用户指定${NC}"
        fi
    fi
fi

# ---------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------
echo ""
echo "=================================="
echo "检查结果"
echo "  能力矩阵: atc→omc=$([ $ATC_OK = 1 ] && echo ✓ || echo ✗) | camodel sim直调=$([ $SIM_OK = 1 ] && echo ✓ || echo ✗)"
echo "  失败 $FAILURES 项 / 提醒 $WARNINGS 项"
if [ "$FAILURES" -gt 0 ]; then
    echo -e "${RED}✗ 存在缺失项，按上述「缺什么→不可用功能→补齐方式」处理${NC}"
elif [ "$WARNINGS" -gt 0 ]; then
    echo -e "${YELLOW}⚠ 主链路可用，存在提醒项${NC}"
else
    echo -e "${GREEN}✓ Kirin 开发环境检查通过${NC}"
fi
[ "$FAILURES" -gt 0 ] && exit 1
exit 0
