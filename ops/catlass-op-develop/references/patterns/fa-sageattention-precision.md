# 精度/挂死取证方法论（已固化验证的完整流程）

> 核心思想：**numpy 精确仿真作真值 + kernel 中间量 GM dump + 假设拟合定位**。
> 该方法在本次开发中定位了全部 20 个 bug，无一例外。

## 1. 三层真值体系

```
golden  = FP32 全精度 attention (numpy, 从 q.bin/k.bin/v.bin 直接算)
emu     = numpy 精确仿真"本算子设计语义" (含量化/补偿/online-softmax/split-K)
kernel  = 设备实际输出 o.bin

判定: emu vs golden ≈ 0.9999 → 设计正确
      kernel vs emu 偏差 → kernel 实现错误 (而非算法问题)
```

## 2. numpy 精确仿真（emu.py 核心，直接可用）

```python
# 每 step j:
S = (qhat @ khat[j].T) * (sm/(sQ*sK_j)) + comp_j[:,None]
#   comp_j = (qhat @ mu[j]) * (sm/sQ)      ← sK 代数消去!
nm = S.max(-1, keepdims=True)
if j > 0: nm = maximum(nm, prev_nm)         # online merge
P = exp(S - nm); es = P.sum(-1)
if j == 0: O = P@V_j; l = es
else:      a = exp(prev_nm - nm); l = a*l + es; O = a*O + P@V_j
prev_nm = nm
result = O / l
```
量化输入 qhat/khat/mu/sK 从 GM dump 的 bin 文件读（kernel A 的产物），
逐位复现 kernel 语义。**先修好 emu 再修 kernel** —— emu 对 golden 0.9999
证明数学设计正确，之后 kernel 与 emu 的差异全是实现 bug。

## 3. GM dump 布局（SAGE_DUMP=1 → qdbg.bin 512KB）

| 偏移 | 内容 | 写入时机 |
|------|------|---------|
| [0,192K) / [192K,384K) | sub0/sub1 的 S_int（SM 入口 dump sUb[m2]）| 每 SM 步（后步覆盖）|
| +128K | P 取证（fp16，两 sub 竞写）| 每 SM 步 |
| +144K (+336K) | O 取证（RS 入口 dump oUb[m2]）| 每 RS 步 |
| [400K,416K) | 每 step 状态快照（12 槽×64 fp32，step×4K）| 每 SM 步 |
| [416K,424K) | 每 step comp 快照（step×2K）| 每 SM 步 |
| +250K | AIC pr[0..3] 进度（标量写，AIC 可靠）| 每 AIC 相位 |
| [452K,460K) | AIV SageMark 槽（alive/smDone/rsWait/rsDone）| 各相位 |
| [464K,496K) | merge 内部（载入后 m/l、算完后 w）| merge |

host 侧落盘：`qhat/khat/sk_scale/mu/o/qdbg/pml/po → <dir>/*.bin`
（mu 是 ×8 冗余布局；pml/po 仅 splitK>1 时）。

## 4. 逐层判定流程（哪层开始错 → 直接指向根因）

```
emu 对不对 (vs golden)
  └→ 状态链对不对 (stchk: lm/sum/alpha 逐 step 逐槽 vs numpy)
       └→ comp 对不对 (compchk: 逐行 vs numpy)
            └→ S_int 对不对 (vs qhat@khat.T 精确整数比较)
                 └→ O 输出对不对 (ochk: vs P_j@V_j)
```

行/列模式特征（直接指认根因类）：
- **cols 0-63 坏、64-127 好** → 第二次调用操作数覆盖在飞 mmad（L0A 槽溢出）
- **rows 32-63 坏** → AIV 子核半槽竞态 / MM2 surplus 提前覆盖 / merge sub 映射错
- **特定行散布错**（max 未更新）→ 状态链槽位
- **误差随步数 ~3×/步增长** → per-block scale 错（×8 索引）/ comp 错
- **整个 work 输出 = 线性组合但系数错** → merge 权重（拟合系数 vs 期望）

## 5. 假设拟合（当直接比较不够时）

**dScale 组合拟合**（定位 scale 用错哪块）：
```python
for ds_i in range(nb):
    S1 = E1 * (sm/(sQ*skv[ds_i*8])) + comp1
    nm = maximum(S1.max(-1), m0)
    print(f'ds_{ds_i}: lm match={allclose(lmk, nm)}')  # 全部用块 i 的 scale?
```

**逐行系数拟合**（定位 merge/softmax 权重错）：
```python
A = stack([po[k][r] for k in range(SK)], 1)   # [D, SK]
c = lstsq(A, o[r])                             # 实际用的系数
expected = w_[:,r] / lstar[r]                  # 期望系数
# 系数交换 → 输入序错; 比率错 → m* 错; 全零 → 该行未写
```

## 6. 挂死取证

1. **分相位标记**（AIC 用标量 GM 写 pr[]，AIV 用 SageMark=Duplicate+DataCopyPad）：
   alive/smDone/rsWait/rsDone + it 号 → 看最后停在哪相位哪迭代
2. **AIC 与 AIV 分别看**：AIC 到 it=P+2 而 AIV 卡中段 → flag 等待分析
3. **flag 计数平衡表**：按 I6 语义逐对核算 set/wait（本会话靠此排除了一整类嫌疑）
4. **单发 vs 双发**（NOWARMUP）：单发对双发错 → 跨 launch flag 残留
   （尾部 set 无人消费时 surplus 留到下一 launch）
5. **dump 探针改变时序**——dump 模式与 no-dump 的挂点可能不同，结论必须交叉验证

## 7. 安全运行器（srun.sh，防 stale 欺骗）

```bash
#!/bin/bash
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
DEV=${SAGE_DEV:?必须显式 SAGE_DEV=N 指定卡}
DIR=$1; shift || true
export ASCEND_RT_VISIBLE_DEVICES=$DEV
rm -f "$DIR/o.bin"
START=$(date +%s)
timeout 30 $ROOT/build/sage_fa "$DIR" --device 0 "$@" > /tmp/srun.log 2>&1
RC=$?; END=$(date +%s)
[ $RC -ne 0 ] && { echo "RUN-HANG(rc=$RC, $((END-START))s) — 结果无效"; exit 2; }
[ ! -f "$DIR/o.bin" ] && { echo "RUN-NOOUT — 结果无效"; exit 3; }
AGE=$(( $(stat -c %Y "$DIR/o.bin") - START ))
[ $AGE -lt 0 ] || [ $AGE -gt $((END-START+1)) ] && { echo "RUN-STALE — 结果无效"; exit 4; }
echo "RUN-OK($((END-START))s)"
```

## 8. 精度回归套件结构（22 case 全维度）

```bash
run_case tag b h hkv sq sk d causal dtype   # gen_data → srun → verify
# 维度: 基础形状/ q·kv 尾块 / causal×3 / GQA(8:1,4:1,2:1) / D=64×3
#       / B>1×2 / bf16×2 / 长序列 32K→200K (--max-planes 采样校验防 golden OOM)
```
验收：kernel vs FP32 golden cosine ≥ 0.99（实测全部 0.99989~1.0）。
