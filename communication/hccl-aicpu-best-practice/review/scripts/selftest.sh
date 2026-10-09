#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# review_template.py 的回归测试。**改了脚本或 references 里的规则口径就跑它。**
#
#   bash scripts/selftest.sh <HCCL 仓根目录>      # 或先 export HCCL_REPO=...
#
# 全程**只读**用户的 HCCL 仓：故障注入都在 mktemp 出来的一个最小仓骨架里做
# （只复制一份 template 源文件 + src/common/alg_parse.h），不碰工作树、不写 build/。
set -u

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REVIEW="$SKILL_DIR/scripts/review_template.py"
FIX="$SKILL_DIR/assets/fixtures"
# 离线规则回归与第 6 节不依赖 HCCL 仓；其余章节核对真实 API 和蓝本。
PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); printf '  ✓ %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); printf '  ✗ %s\n' "$1"; }
hdr()  { printf '\n== %s ==\n' "$1"; }

hdr "离线规则回归（Ring 适用性与 Markdown 围栏）"
if python3 -B "$SKILL_DIR/scripts/test_review.py"; then
    ok "离线规则回归"
else
    bad "离线规则回归"
fi

check_detect_topo() {
hdr "6. 算法族识别（只看命名，不许靠变量名兜底）"
# 回归 detect_topo：紧凑形式会抹掉下划线，让两字母的 HD 永远被字母夹住而漏认；
# 夹具/第 2、3 节都是 --topo 强制指定的，测不到这条路径，所以这里直接测函数。
python3 - "$SKILL_DIR" > "$OUT" 2>&1 <<'PYEOF'
import sys, os, re
sys.path.insert(0, os.path.join(sys.argv[1], "scripts"))
import review_template as rt
cases = [
    ("InsTempAllReduceHd",              "ins_temp_all_reduce_hd",               "hd"),
    ("InsTempAllReduceHD",              "ins_temp_all_reduce_hd",               "hd"),
    ("InsTempFixtureHdGood",            "ins_temp_fixture_hd_good",             "hd"),
    ("InsTempAllReduceHalvingDoubling", "ins_temp_all_reduce_halving_doubling", "hd"),
    ("InsTempRecursiveHd",              "ins_temp_recursive_hd",                "hd"),
    ("InsTempAllGatherRing",            "ins_temp_all_gather_ring",             "ring"),
    ("InsTempAllReduceMesh1DOneShot",   "ins_temp_all_reduce_mesh_1D_one_shot", "mesh"),
    ("InsTempAllGatherNhr",             "ins_temp_all_gather_nhr",              "nhr"),
    ("InsTempAllReduceThreshold",       "ins_temp_all_reduce_threshold",        "unknown"),
]
bad = [(c, w, rt.detect_topo(c, f, "", "")) for c, f, w in cases
       if rt.detect_topo(c, f, "", "") != w]
for c, w, g in bad:
    print("MISMATCH %s 期望 %s 实得 %s" % (c, w, g))
helper_bad = []
if rt.has_repeat_loop('HCCL_INFO("repeatNum=%u", tempAlgParams.repeatNum);'):
    helper_bad.append("日志里的 repeatNum 被误认成循环")
if not rt.has_repeat_loop('for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {}'):
    helper_bad.append("真实 repeat 循环没有识别")
section = rt.markdown_section('## 4. Buffer 布局\n### 4.1 细节\n| **scratch 倍数** | `N` |\n## 5. 切片\n', 4, r'Buffer\s*布局')
if section is None or "scratch 倍数" not in section:
    helper_bad.append("章节内子标题导致 spec 正文被截断")
for msg in helper_bad:
    print("MISMATCH", msg)
print("OK" if not bad and not helper_bad else "FAILED")
PYEOF
if grep -q '^OK$' "$OUT"; then ok "9 个命名归族与 repeat 循环识别正确"; else bad "纯函数检查失败："; cat "$OUT"; fi
}

REPO="${1:-${HCCL_REPO:-}}"
if [ -z "$REPO" ]; then
    # 其余各节都要读真仓：X01 依赖 src/common/alg_parse.h 的 AlgoType，注入依赖真仓蓝本。
    # 这里不拿假的 alg_parse.h 顶替（真仓 API 不许在 skill 里写死镜像），只如实跳过。
    echo "未指定 HCCL 仓，只跑离线规则回归与算法族识别；其余需要："
    echo "  bash scripts/selftest.sh <HCCL 仓根目录>   # 或 export HCCL_REPO"
    WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
    OUT="$WORK/out.txt"
    check_detect_topo
    printf '\n通过 %d，失败 %d（跳过其余各节：需要 HCCL 仓）\n' "$PASS" "$FAIL"
    [ "$FAIL" -eq 0 ] || exit 1
    exit 0
fi
REPO="$(cd "$REPO" && pwd)"
[ -d "$REPO/src/ops" ] || { echo "不像是 HCCL 仓（缺 src/ops）: $REPO"; exit 2; }


# expect_rules <说明> <输出文件> <必须出现的规则号...>
expect_rules() {
    local what="$1" out="$2"; shift 2
    local missing=""
    for r in "$@"; do grep -qE "(BLOCKER|MAJOR|MINOR|INFO) +$r " "$out" || missing="$missing $r"; done
    if [ -z "$missing" ]; then ok "$what：$* 全部报出"; else bad "$what：漏报$missing"; sed -n '1,40p' "$out"; fi
}
# expect_only <说明> <输出文件> <允许出现的规则号...>（BLOCKER/MAJOR 层面）
expect_only() {
    local what="$1" out="$2"; shift 2
    local allow=" $* " extra=""
    while read -r rule; do
        case "$allow" in *" $rule "*) ;; *) extra="$extra $rule";; esac
    done < <(grep -E '^(✗ BLOCKER|▲ MAJOR)' "$out" | awk '{print $3}' | sort -u)
    if [ -z "$extra" ]; then ok "$what：没有预期外的 BLOCKER/MAJOR"; else bad "$what：多报$extra"; sed -n '1,40p' "$out"; fi
}

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
OUT="$WORK/out.txt"

hdr "0. 规则目录可打印"
if python3 "$REVIEW" --rules > "$OUT" 2>&1 && grep -q "C01" "$OUT" && grep -q "S00" "$OUT"; then
    ok "--rules（含 S00）"
else
    bad "--rules"; cat "$OUT"
fi

# X01 必须逐项判断。这里模拟“枚举和两张映射表已有 Ring，只有 ALGO_TYPES 未登记”的半完成状态。
X01R="$WORK/x01-partial"; mkdir -p "$X01R/src/common"
cat > "$X01R/src/common/alg_parse.cc" <<'XEOF'
static const std::map<std::string, std::string> ALGO_TYPES = {{"mesh", "Mesh"}};
const std::map<AlgoType, std::string>& GetAlgoTypeToNameMap()
{
    static const std::map<AlgoType, std::string> map = {{AlgoType::RING, "Ring"}};
}
const std::map<std::string, AlgoType>& GetAlgoNameToTypeMap()
{
    static const std::map<std::string, AlgoType> map = {{"Ring", AlgoType::RING}};
}
XEOF
python3 - "$SKILL_DIR" "$X01R" > "$OUT" 2>&1 <<'PYEOF'
import os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "scripts"))
import review_template as rt
class F: repo = sys.argv[2]
rv = rt.Review("x", "ring", "")
rt.algo_gap(F(), rv, {"RING"}, "RING", ("ring",))
hint = "\n".join(f.hint or "" for f in rv.findings)
print(hint)
ok = (len(rv.findings) == 1 and rv.findings[0].sev == rt.BLOCKER
      and "`ALGO_TYPES`" in hint
      and "GetAlgoTypeToNameMap" not in hint and "GetAlgoNameToTypeMap" not in hint)
print("OK" if ok else "FAILED")
PYEOF
if grep -q '^OK$' "$OUT"; then ok "X01 半完成状态只报告真实缺口"; else bad "X01 仍会虚构缺口"; cat "$OUT"; fi

python3 "$REVIEW" --repo "$REPO" --all --topo ring > "$OUT" 2>&1
if [ $? -eq 2 ] && grep -q -- "--topo 只能用于单个 template" "$OUT"; then
    ok "拒绝把 --topo 强制作用于 --all"
else
    bad "--all --topo 未被拒绝"; sed -n '1,12p' "$OUT"
fi

hdr "1. 存量基线：全仓不应有 BLOCKER / MAJOR"
# ★ 先确认扫到了东西。以前这里不看退出码也不看数量，仓布局一变就会打印
#   "✓ 0 个存量 template 基线干净" —— 一条什么都没检查的假通过。
NTMPL=$(ls "$REPO"/src/ops/*/algorithm/template/aicpu/*.cc 2>/dev/null | grep -vc op_common || true)
python3 "$REVIEW" --repo "$REPO" --all --quiet > "$OUT" 2>&1
RC=$?
tail -1 "$OUT"
UNEXPECTED=$(grep -E '^(✗ BLOCKER|▲ MAJOR)' "$OUT" | awk '{print $3}' | grep -v '^X01$' | sort -u)
if [ "$NTMPL" -eq 0 ]; then
    bad "全仓一个 template 都没扫到（布局变了？）——基线这一节无从谈起"
    sed -n '1,15p' "$OUT"
elif [ "$RC" -gt 1 ]; then
    bad "--all 执行失败（退出码 $RC）"
    sed -n '1,15p' "$OUT"
elif [ -z "$UNEXPECTED" ]; then
    ok "$NTMPL 个存量 template 基线干净（X01 除外）"
    grep -cE '^✗ BLOCKER X01' "$OUT" | grep -qv '^0$' && \
        printf '    注：仓里有 ring/HD template 报 X01（选路侧未登记），属真实待办\n'
else
    bad "存量里出现了预期外的 BLOCKER/MAJOR：$UNEXPECTED —— 真仓变了，或规则口径过严（references/01 §4）"
    grep -E '^(✗ BLOCKER|▲ MAJOR)' "$OUT" | sort | uniq -c
fi

hdr "2. Ring 夹具（不依赖仓内实现是否存在，靠夹具锁口径）"
python3 "$REVIEW" --repo "$REPO" --topo ring "$FIX/ins_temp_all_reduce_ring_bad.cc" > "$OUT" 2>&1
expect_rules "ring 反例" "$OUT" C01 R01 R02 R03 R05 R06
python3 "$REVIEW" --repo "$REPO" --topo ring "$FIX/ins_temp_all_reduce_ring_good.cc" > "$OUT" 2>&1
expect_only "ring 正例" "$OUT" X01

hdr "3. HD 夹具"
python3 "$REVIEW" --repo "$REPO" --topo hd "$FIX/ins_temp_all_reduce_hd_bad.cc" > "$OUT" 2>&1
expect_rules "hd 反例" "$OUT" H01 H02 H03 H06
python3 "$REVIEW" --repo "$REPO" --topo hd "$FIX/ins_temp_all_reduce_hd_good.cc" > "$OUT" 2>&1
expect_only "hd 正例" "$OUT" X01

hdr "4. 故障注入（在临时仓骨架里改真仓蓝本的副本，不碰 $REPO）"
SRC="$REPO/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot"
if [ ! -f "$SRC.cc" ]; then
    bad "找不到蓝本 $SRC.cc —— 真仓改名了，注入用例需要跟着换（见 references/01 §5）"
else
    MINI="$WORK/mini"; DST="$MINI/src/ops/all_reduce/algorithm/template/aicpu"
    mkdir -p "$DST" "$MINI/src/common"
    cp "$SRC.cc" "$SRC.h" "$DST/"
    cp "$REPO/src/common/alg_parse.h" "$MINI/src/common/" 2>/dev/null
    # 机械层 W01 要查两处 CMake，所以最小骨架里也得有它们（原样复制，只读）
    cp "$REPO/src/ops/all_reduce/algorithm/template/aicpu/CMakeLists.txt" "$DST/" 2>/dev/null
    cp "$REPO/src/scatter_aicpu_kernel.cmake" "$MINI/src/" 2>/dev/null
    # executor 目录留空即可：W06 只报 MINOR
    mkdir -p "$MINI/src/ops/all_reduce/algorithm/executor"
    T="$DST/ins_temp_all_reduce_mesh_1D_one_shot.cc"
    cp "$T" "$WORK/orig.cc"

    python3 "$REVIEW" --repo "$MINI" "$T" > "$OUT" 2>&1
    expect_only "未注入的副本" "$OUT"

    # ① scratch 倍数改成 1 —— one-shot 必须是 templateRankSize_
    sed -i 's/u64 scratchMultiple = templateRankSize_;/u64 scratchMultiple = 1;/' "$T"
    python3 "$REVIEW" --repo "$MINI" "$T" > "$OUT" 2>&1
    expect_rules "注入①scratch 倍数写错" "$OUT" M01
    cp "$WORK/orig.cc" "$T"

    # ② scratch 倍数乘 repeatNum —— 双重放大
    sed -i 's/u64 scratchMultiple = templateRankSize_;/u64 scratchMultiple = templateRankSize_ * tempAlgParams_.repeatNum;/' "$T"
    python3 "$REVIEW" --repo "$MINI" "$T" > "$OUT" 2>&1
    expect_rules "注入②倍数里乘了 repeatNum" "$OUT" C01
    cp "$WORK/orig.cc" "$T"

    # ③ 删掉后置同步 —— 主流会在从流写完前去归约
    sed -i '/PostSyncInterThreads/d' "$T"
    python3 "$REVIEW" --repo "$MINI" "$T" > "$OUT" 2>&1
    expect_rules "注入③删掉 PostSyncInterThreads" "$OUT" C07
    cp "$WORK/orig.cc" "$T"

    # ④ 从 scatter_aicpu_kernel.cmake 里摘掉这一行 —— 本仓最高频的接线 bug
    sed -i '\|ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc|d' "$MINI/src/scatter_aicpu_kernel.cmake"
    python3 "$REVIEW" --repo "$MINI" "$T" > "$OUT" 2>&1
    expect_rules "注入④漏登记 scatter_aicpu_kernel.cmake" "$OUT" W01
    cp "$REPO/src/scatter_aicpu_kernel.cmake" "$MINI/src/"

    # ⑤ 删掉 CalcScratchMultiple 的实现 —— 基类默认实现直接返回失败
    sed -i 's/^u64 InsTempAllReduceMesh1DOneShot::CalcScratchMultiple/u64 InsTempAllReduceMesh1DOneShot::CalcScratchMultipleXX/' "$T"
    python3 "$REVIEW" --repo "$MINI" "$T" > "$OUT" 2>&1
    expect_rules "注入⑤删掉 CalcScratchMultiple 实现" "$OUT" W02
    cp "$WORK/orig.cc" "$T"

    # ⑥ props 里写一个 alg_parse.h 枚举里没有的 AlgoType —— 选路永远匹配不到
    cp "$DST/ins_temp_all_reduce_mesh_1D_one_shot.h" "$WORK/orig.h"
    sed -i 's/AlgoType::MESH_ONESHOT/AlgoType::MESH_NOT_A_REAL_TYPE/' "$DST/ins_temp_all_reduce_mesh_1D_one_shot.h"
    python3 "$REVIEW" --repo "$MINI" "$T" > "$OUT" 2>&1
    expect_rules "注入⑥props 写了不存在的 AlgoType" "$OUT" W03
    cp "$WORK/orig.h" "$DST/ins_temp_all_reduce_mesh_1D_one_shot.h"

    python3 "$REVIEW" --repo "$MINI" "$T" > "$OUT" 2>&1
    expect_only "还原后" "$OUT"

    # ⑦ --semantic-only 应当跳过机械层
    sed -i '\|ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc|d' "$MINI/src/scatter_aicpu_kernel.cmake"
    python3 "$REVIEW" --repo "$MINI" --semantic-only "$T" > "$OUT" 2>&1
    if grep -qE "(BLOCKER|MAJOR) +W01 " "$OUT"; then bad "--semantic-only 没跳过机械层"; else ok "--semantic-only 跳过了 W01"; fi
    cp "$REPO/src/scatter_aicpu_kernel.cmake" "$MINI/src/"
fi

hdr "5. 代码 ↔ spec 对照"
python3 "$REVIEW" --repo "$REPO" "$SRC.cc" --spec "$FIX/spec-mismatch.md" > "$OUT" 2>&1
expect_rules "故意写歪的 spec" "$OUT" S01 S02 S03

printf '# 这不是 dataflow spec\n' > "$WORK/invalid-spec.md"
python3 "$REVIEW" --repo "$REPO" "$SRC.cc" --spec "$WORK/invalid-spec.md" > "$OUT" 2>&1
expect_rules "格式不完整的 spec" "$OUT" S00

cat > "$WORK/post-sync-mismatch.md" <<'SPEOF'
## 4. Buffer 布局
| 项 | 值 |
|---|---|
| **scratch 倍数** | `N` |
## 5. 切片
| 符号 | 值 |
|---|---|
| `RPT` | `1` |
## 6. 数据流
```text
sync main -> sub
```
SPEOF
python3 "$REVIEW" --repo "$REPO" "$SRC.cc" --spec "$WORK/post-sync-mismatch.md" > "$OUT" 2>&1
expect_rules "只缺 sub→main 的 spec" "$OUT" S03

check_detect_topo

hdr "7. 更多规则的故障注入（每条各建一个骨架）"
# 这些规则在真仓和夹具上都不触发（存量本来就没犯），只能靠注入证明它们真的抓得住。
# 每个蓝本缺失都会 bad 出来，不静默跳过——真仓改名时要跟着换（references/01 §5）。
MINI_N=0
MINI_DIR=""; MINI_CC=""
mini_for() {   # <op> <filebase>：建最小仓骨架，设置 MINI_DIR / MINI_CC
    local op="$1" fb="$2" src="$REPO/src/ops/$1/algorithm/template/aicpu/$2"
    [ -f "$src.cc" ] && [ -f "$src.h" ] || return 1
    MINI_N=$((MINI_N+1))
    MINI_DIR="$WORK/mini_$MINI_N"
    local d="$MINI_DIR/src/ops/$op/algorithm/template/aicpu"
    mkdir -p "$d" "$MINI_DIR/src/common" "$MINI_DIR/src/ops/$op/algorithm/executor"
    cp "$src.cc" "$src.h" "$d/"
    cp "$REPO/src/common/alg_parse.h" "$MINI_DIR/src/common/" 2>/dev/null
    cp "$REPO/src/ops/$op/algorithm/template/aicpu/CMakeLists.txt" "$d/" 2>/dev/null
    cp "$REPO/src/scatter_aicpu_kernel.cmake" "$MINI_DIR/src/" 2>/dev/null
    MINI_CC="$d/$fb.cc"
}
inject() {     # <说明> <op> <filebase> <sed脚本> <期望规则...>
    local what="$1" op="$2" fb="$3" expr="$4"; shift 4
    if ! mini_for "$op" "$fb"; then
        bad "$what：找不到蓝本 src/ops/$op/algorithm/template/aicpu/$fb.{cc,h}（真仓改名了？）"
        return
    fi
    sed -i "$expr" "$MINI_CC" "${MINI_CC%.cc}.h"
    python3 "$REVIEW" --repo "$MINI_DIR" "$MINI_CC" > "$OUT" 2>&1
    expect_rules "$what" "$OUT" "$@"
}

inject "注入⑧one-shot 归约去掉收尾 LocalReduce" \
    all_reduce ins_temp_all_reduce_mesh_1D_one_shot \
    's/\bLocalReduce(/LocalReduceOff(/g' M05
inject "注入⑨two-shot 倍数改成 rankSize" \
    all_reduce ins_temp_all_reduce_mesh_1D_two_shot \
    's/u64 multiple = 2;/u64 multiple = templateRankSize_;/' M02
inject "注入⑩NHR 倍数改成 3" \
    all_gather ins_temp_all_gather_nhr \
    's/u64 scratchMultiple = templateRankSize_;/u64 scratchMultiple = 3;/' N01
inject "注入⑪NHR step 数写死" \
    all_gather ins_temp_all_gather_nhr \
    's/step < nSteps/step < 4/' N03
inject "注入⑫用了 channelsPerRank_ 却没有 GetThreadNum" \
    all_gather ins_temp_all_gather_nhr \
    's/GetThreadNum/GetThreadNumX/g' N04
inject "注入⑬软归约三步顺序调反" \
    all_reduce ins_temp_reduce_scatter_mesh_1D_intra \
    '0,/HcommBatchModeEnd/s//HcommTMPX/; 0,/HcommBatchModeStart/s//HcommBatchModeEnd/; 0,/HcommTMPX/s//HcommBatchModeStart/' \
    C10

# H05 只能在夹具上造：有 2 的幂判断、但没有 part1 / blockSize 的淘汰与回传
HDW="$WORK/h05"; mkdir -p "$HDW"
cp "$FIX/ins_temp_all_reduce_hd_good.cc" "$FIX/ins_temp_all_reduce_hd_good.h" "$HDW/"
sed -i 's/blockSize_/pow2Cnt_/g; s/part1Size_/extraCnt_/g' "$HDW"/ins_temp_all_reduce_hd_good.*
sed -i 's|while ((pow2Cnt_ << 1) <= templateRankSize_) { pow2Cnt_ = pow2Cnt_ << 1; }|pow2Cnt_ = IsPowerOfTwo(templateRankSize_) ? templateRankSize_ : 1;|' \
    "$HDW/ins_temp_all_reduce_hd_good.cc"
python3 "$REVIEW" --repo "$REPO" --topo hd "$HDW/ins_temp_all_reduce_hd_good.cc" > "$OUT" 2>&1
expect_rules "注入⑭有 2 的幂判断但没有 part1 淘汰" "$OUT" H05

# R04 / C08：ring 正例去掉所有线程间同步与 notify —— step 之间没有同步，
# 资源三字段也跟着缺，两条都该报出来。
R04W="$WORK/r04"; mkdir -p "$R04W"
cp "$FIX/ins_temp_all_reduce_ring_good.cc" "$FIX/ins_temp_all_reduce_ring_good.h" "$R04W/"
sed -i -E 's/.*(PreSyncInterThreads|PostSyncInterThreads|[Nn]otify).*//g' "$R04W/ins_temp_all_reduce_ring_good.cc"
python3 "$REVIEW" --repo "$REPO" --topo ring "$R04W/ins_temp_all_reduce_ring_good.cc" > "$OUT" 2>&1
expect_rules "注入⑮ring 去掉 step 间同步与 notify" "$OUT" R04 C08

hdr "8. --base：与 base 快照比对，纯删除/改 .h/改 CMake 都不能漏"
# 归属靠「在 base 快照上再评一遍、比对 finding 身份」，不看行号。
# 下面三条是按行号归属时确定会漏的高危改动，必须都抓到并让 --strict 返回 1。
BW="$WORK/basetest"
if [ ! -f "$SRC.cc" ]; then
    bad "--base 用例：找不到蓝本 $SRC.cc"
else
    mkdir -p "$BW/src/ops/all_reduce/algorithm/template/aicpu" "$BW/src/common"
    cp "$SRC.cc" "$SRC.h" "$BW/src/ops/all_reduce/algorithm/template/aicpu/"
    cp "$REPO/src/ops/all_reduce/algorithm/template/aicpu/CMakeLists.txt" \
       "$BW/src/ops/all_reduce/algorithm/template/aicpu/" 2>/dev/null
    cp "$REPO/src/common/alg_parse.h" "$REPO/src/common/alg_parse.cc" "$BW/src/common/" 2>/dev/null
    cp "$REPO/src/scatter_aicpu_kernel.cmake" "$BW/src/" 2>/dev/null
    mkdir -p "$BW/src/ops/all_reduce/algorithm/executor"
    cat > "$BW/src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_exec.cc" <<'EXEOF'
#include "ins_temp_all_reduce_mesh_1D_one_shot.h"
namespace ops_hccl {
REGISTER_EXEC_V2(AllReduceMesh1DOneShot, InsTempAllReduceMesh1DOneShot);
}
EXEOF
    ( cd "$BW" && git init -q && git add -A \
      && git -c user.name=t -c user.email=t@t commit -qm base ) >/dev/null 2>&1
    BT="$BW/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot"

    # base_case <说明> <期望新增的规则> —— 跑 --base --strict，要求该规则出现在「本次引入」段且退出码为 1
    base_case() {
        local what="$1" rule="$2"
        python3 "$REVIEW" --repo "$BW" --base HEAD --strict > "$OUT" 2>&1
        local rc=$?
        local newsec; newsec=$(sed -n '/★ 本次改动引入/,/^-- ·/p' "$OUT")
        if printf '%s' "$newsec" | grep -qE " $rule " && [ "$rc" -eq 1 ]; then
            ok "$what：$rule 归「本次引入」且 --strict 返回 1"
        else
            bad "$what：$rule 没进「本次引入」或退出码不是 1（实得 $rc）"
            sed -n '1,25p' "$OUT"
        fi
        ( cd "$BW" && git checkout -- . ) 2>/dev/null
        rm -f "$BW/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_ring."[ch][ch]
    }

    python3 "$REVIEW" --repo "$BW" --base HEAD > "$OUT" 2>&1
    if grep -q "没有任何影响 AICPU template 的改动" "$OUT"; then ok "干净工作树：--base 认定无改动"
    else bad "干净工作树却报出了改动"; sed -n '1,10p' "$OUT"; fi

    # ① 纯删除：没有任何「新增行」，按行号归属必漏
    sed -i '/PostSyncInterThreads/d' "$BT.cc"
    base_case "纯删除 PostSyncInterThreads" C07

    # ② 只改 .h：.cc 一行没动
    sed -i 's/AlgoType::MESH_ONESHOT/AlgoType::MESH_NOT_REAL/' "$BT.h"
    base_case "只改 .h 的 AlgoType" W03

    # ③ 只改全局 CMake：连 template 文件都没动，自动选目标也必须覆盖到
    sed -i '\|ins_temp_all_reduce_mesh_1D_one_shot.cc|d' "$BW/src/scatter_aicpu_kernel.cmake"
    base_case "只删 scatter_aicpu_kernel.cmake 的接线" W01

    # ④ 改 .cc 里的语句（原有能力不能退化）
    sed -i 's/u64 scratchMultiple = templateRankSize_;/u64 scratchMultiple = 1;/' "$BT.cc"
    base_case "改 scratch 倍数" M01

    # ⑤ 存量欠账不能算到作者头上，且默认不展开造成噪声
    sed -i '/PostSyncInterThreads/d' "$BT.cc"
    python3 "$REVIEW" --repo "$BW" --base HEAD > "$OUT" 2>&1
    if grep -q '^-- · 存量欠账' "$OUT" || sed -n '/★ 本次改动引入/,/^-- ·/p' "$OUT" | grep -qE ' (C03|C05|C11|W04|W06) '; then
        bad "存量 MINOR 被算成了本次引入"; sed -n '1,25p' "$OUT"
    else
        ok "存量 MINOR 只汇总、不展开也不算成本次引入"
    fi
    python3 "$REVIEW" --repo "$BW" --base HEAD --show-baseline > "$OUT" 2>&1
    if grep -q '^-- · 存量欠账' "$OUT"; then ok "--show-baseline 可按需展开存量"
    else bad "--show-baseline 没有展开存量"; sed -n '1,25p' "$OUT"; fi
    ( cd "$BW" && git checkout -- . ) 2>/dev/null

    # 删除类改动：以前 affected_templates 对状态 D 一律 continue，整类漏掉
    cp "$REPO/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot.cc" \
       "$REPO/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot.h" \
       "$BW/src/ops/all_reduce/algorithm/template/aicpu/" 2>/dev/null
    ( cd "$BW" && git add -A && git -c user.name=t -c user.email=t@t commit -qm two ) >/dev/null 2>&1

    rm -f "$BT.h"
    base_case "只删掉配套 .h" C02

    rm -f "$BT.cc" "$BT.h"
    base_case "整份删掉但 CMake 仍留引用" W01

    # 删干净 = template + 两处 CMake + executor 的 include 与注册都清掉
    rm -f "$BT.cc" "$BT.h"
    sed -i '\|ins_temp_all_reduce_mesh_1D_one_shot|d' "$BW/src/scatter_aicpu_kernel.cmake" \
        "$BW/src/ops/all_reduce/algorithm/template/aicpu/CMakeLists.txt" 2>/dev/null
    rm -f "$BW/src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_exec.cc"
    python3 "$REVIEW" --repo "$BW" --base HEAD --strict > "$OUT" 2>&1
    if [ $? -eq 0 ] && grep -q "没有残留引用" "$OUT"; then
        ok "整份删掉且接线清干净：只报 INFO，退出 0"
    else
        bad "删干净的场景不该阻塞"; sed -n '1,15p' "$OUT"
    fi
    ( cd "$BW" && git checkout -- . ) 2>/dev/null

    # spec 归属：只改 spec、代码不动，S01 必须算本次引入（base 侧要用 base 版本的 spec）
    printf '## 4. Buffer 布局\n\n| 项 | 值 |\n|---|---|\n| **scratch 倍数** | `N` |\n\n## 5. 切片\n\n| 符号 | 值 |\n|---|---|\n| `RPT` | `1` |\n\n## 6. 数据流\n\n```text\nsync main -> sub\nsync sub -> main\n```\n' > "$BW/spec.md"
    ( cd "$BW" && git add -A && git -c user.name=t -c user.email=t@t commit -qm spec ) >/dev/null 2>&1
    sed -i 's/`N`/`2`/' "$BW/spec.md"
    python3 "$REVIEW" --repo "$BW" --base HEAD --strict --spec "$BW/spec.md" "$BT.cc" > "$OUT" 2>&1
    local_rc=$?
    if sed -n '/★ 本次改动引入/,/^-- ·/p' "$OUT" | grep -q S01 && [ "$local_rc" -eq 1 ]; then
        ok "只改 spec：S01 归「本次引入」且 --strict 返回 1"
    else
        bad "改了 spec 却没算本次引入（base 侧仍在用当前 spec？）"; sed -n '1,15p' "$OUT"
    fi
    ( cd "$BW" && git checkout -- . ) 2>/dev/null

    # 只删 .cc、保留 .h —— 同样是配对缺失，不能被当成「整份删掉」只报 INFO
    rm -f "$BT.cc"
    sed -i '\|ins_temp_all_reduce_mesh_1D_one_shot|d' "$BW/src/scatter_aicpu_kernel.cmake" \
        "$BW/src/ops/all_reduce/algorithm/template/aicpu/CMakeLists.txt" 2>/dev/null
    base_case "只删 .cc 保留 .h" C02

    # 删干净了 template 与两处 CMake，但 executor 还留着 include 与注册 —— 确定的编译失败。
    # 类名必须从 base 快照的 .h 解析：按文件名拼会拼成 AllReduceMesh1dOneShot，永远匹配不上。
    rm -f "$BT.cc" "$BT.h"
    sed -i '\|ins_temp_all_reduce_mesh_1D_one_shot|d' "$BW/src/scatter_aicpu_kernel.cmake" \
        "$BW/src/ops/all_reduce/algorithm/template/aicpu/CMakeLists.txt" 2>/dev/null
    python3 "$REVIEW" --repo "$BW" --base HEAD --strict > "$OUT" 2>&1
    rc=$?
    if [ "$rc" -eq 1 ] && grep -q "REGISTER_EXEC_V2(InsTempAllReduceMesh1DOneShot)" "$OUT"; then
        ok "executor 残留 include/注册：W01 报出真实类名且退出 1"
    else
        bad "executor 悬空引用没抓到（类名拼错了？）"; sed -n '1,15p' "$OUT"
    fi
    ( cd "$BW" && git checkout -- . ) 2>/dev/null

    # 已暂存的 rename：旧路径的接线不能因为「只记新名」而被漏掉
    ( cd "$BW" && git mv "src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc" \
        "src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_renamed.cc" \
      && git mv "src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.h" \
        "src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_renamed.h" ) >/dev/null 2>&1
    python3 "$REVIEW" --repo "$BW" --base HEAD --strict > "$OUT" 2>&1
    rc=$?
    if [ "$rc" -eq 1 ] && grep -q "已删除" "$OUT"; then
        ok "已暂存的 rename：旧路径的悬空接线报出且退出 1"
    else
        bad "rename 的旧路径被漏掉了"; sed -n '1,15p' "$OUT"
    fi
    ( cd "$BW" && git reset -q --hard HEAD ) >/dev/null 2>&1

    # ⑥ 整份新增：X01 这类定位不到行的 finding 也要算本次引入
    # 真仓可能已登记 Ring；在临时副本显式注入 DSL 登记缺口，不依赖历史缺失状态。
    sed -i 's/"ring"/"missing_for_test"/g' "$BW/src/common/alg_parse.cc"
    NEW="$BW/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_ring"
    sed -e 's/InsTempAllReduceMesh1DOneShot/InsTempAllReduceRing/g' "$SRC.cc" > "$NEW.cc"
    sed -e 's/InsTempAllReduceMesh1DOneShot/InsTempAllReduceRing/g' "$SRC.h" > "$NEW.h"
    python3 "$REVIEW" --repo "$BW" --base HEAD "$NEW.cc" > "$OUT" 2>&1
    if sed -n '/★ 本次改动引入/,/^-- ·/p' "$OUT" | grep -q X01; then
        ok "整份新增的 template：X01 也算「本次引入」"
    else
        bad "整份新增的 template 里 X01 没算本次引入"; sed -n '1,20p' "$OUT"
    fi
    rm -f "$NEW.cc" "$NEW.h"
fi

hdr "9. ring / HD 的严重度口径（按判据来源分档，别悄悄改回去）"
# 有文献背书的判据维持 MAJOR；纯本仓映射（scratch 倍数、线程模型）只报 INFO；
# H01 因检测靠命名会误报而降为 MAJOR；X01 查的是仓内可核验事实，维持 BLOCKER。
sev_is() {   # <说明> <输出文件> <规则> <期望级别>
    local what="$1" out="$2" rule="$3" want="$4"
    local got; got=$(grep -oE "(BLOCKER|MAJOR|MINOR|INFO) +$rule " "$out" | head -1 | awk '{print $1}')
    if [ "$got" = "$want" ]; then ok "$what：$rule = $want"
    else bad "$what：$rule 期望 $want，实得 ${got:-未报出}"; fi
}
python3 "$REVIEW" --repo "$REPO" --topo ring "$FIX/ins_temp_all_reduce_ring_bad.cc" > "$OUT" 2>&1
sev_is "ring 反例" "$OUT" R02 MAJOR
sev_is "ring 反例" "$OUT" R05 INFO
sev_is "ring 反例" "$OUT" R06 INFO
python3 "$REVIEW" --repo "$REPO" --topo hd "$FIX/ins_temp_all_reduce_hd_bad.cc" > "$OUT" 2>&1
sev_is "hd 反例" "$OUT" H01 MAJOR
sev_is "hd 反例" "$OUT" H03 MAJOR
sev_is "hd 反例" "$OUT" H06 INFO

hdr "10. 文档与脚本的严重度、章节索引一致"
# references/01 §3 的判据表里点名的规则，级别必须和脚本实际报的一致。
# 之前 C01 降 MAJOR、R06 降 INFO 时这张表都漏改了。
python3 - "$SKILL_DIR" > "$OUT" 2>&1 <<'PYEOF'
import sys, os, re
root = sys.argv[1]
sys.path.insert(0, os.path.join(root, "scripts"))
import review_template as rt
src = open(os.path.join(root, "scripts", "review_template.py"), encoding="utf-8").read()
# 脚本里每条规则实际用到的severity（同一规则可能多处，取全集）
actual = {}
for rule, sev in re.findall(r'rv\.add\(\s*"(\w+)"\s*,\s*([A-Z]+)', src):
    actual.setdefault(rule, set()).add(sev)
for r in rt.UNVERIFIED_MAPPING:
    actual.setdefault(r, set()).add("INFO")
doc = open(os.path.join(root, "references", "01-review-method.md"), encoding="utf-8").read()
bad = []
for line in doc.splitlines():
    m = re.match(r"\|\s*(BLOCKER|MAJOR|MINOR|INFO)\s*\|", line)
    if not m:
        continue
    level = m.group(1)
    for rule in re.findall(r"\b([WCMNRHXS]\d\d)\b", line):
        if rule in actual and level not in actual[rule]:
            bad.append("%s 文档标 %s，脚本实际 %s" % (rule, level, "/".join(sorted(actual[rule]))))

# RULES 指向 references/02 的章节号必须真能找到该规则，防止章节增删后整体错位。
common = open(os.path.join(root, "references", "02-common-checks.md"), encoding="utf-8").read()
for rule, _, ref in rt.RULES:
    m = re.fullmatch(r"references/02 §(\d+)", ref)
    if not m:
        continue
    rule_heading = re.search(r"^#{2,6}.*\b%s\b.*$" % re.escape(rule), common, re.M)
    sections = list(re.finditer(r"^##\s+(\d+)\.", common[:rule_heading.end()] if rule_heading else "", re.M))
    actual_section = sections[-1].group(1) if sections else None
    if not rule_heading or actual_section != m.group(1):
        bad.append("%s 规则目录指向 §%s，实际标题为 %s" %
                   (rule, m.group(1), actual_section if actual_section is not None else "未找到"))
for b in bad:
    print("MISMATCH", b)
print("OK" if not bad else "FAILED")
PYEOF
if grep -q '^OK$' "$OUT"; then ok "严重度与 references/02 章节索引一致"; else bad "文档口径不一致："; cat "$OUT"; fi

hdr "结果"
printf '通过 %d，失败 %d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
