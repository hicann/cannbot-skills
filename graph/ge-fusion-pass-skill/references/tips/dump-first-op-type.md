# tip: 按 op type 匹配前先 dump 取真实名（最常见的静默失败原因）

> 📎 导航落点：`references/pass-development-paradigm.md` §1（输入识别与证据优先级）、`references/fusion-troubleshooting.md` §3（是否产生候选与匹配）。本文件仍是该硬性做法的**唯一权威定义源**，正文未改动。迁移映射见 `tips/MIGRATION.md`。

**读者**：② 开发 / ①。

## 症状

按前端框架里的算子名（`MatMul`、`Conv2D` 等）写 pattern 或 `GNode` type 判断 → **静默不命中** → fusion 跳过 → 首次 ATC 看似成功却没融合。这是最常见的静默失败原因，因为不报错。

## 根因

ONNX/TF 的算子名经 GE 导入后会被**改名**成内部 op type，且**因模型而异**：同一个 MatMul，导入后可能是 `MatMulV2`、`BatchMatMulV2` 或仍叫 `MatMul`（取决于 rank/shape/导出方式，**无法静态推断**）。按框架名写匹配就会漏。

## 硬性做法（写 pattern / 函数式 Graph/GNode 匹配前先做）

先拿导入后的真实 op type 再写匹配：

```bash
cd data && python torch_gen_onnx.py && cd ..          # 先生成 model.onnx（若 case 用脚本生成）
source $ASCEND_HOME_PATH/set_env.sh && export DUMP_GE_GRAPH=1
atc --model=./data/model.onnx --framework=5 --soc_version=<soc> --output=./data/model 2>&1 | tee dump.log
grep -hoE 'op_type: "ge:[^"]+"' ge_onnx_*_PreRunBegin.pbtxt | sort -u   # ← 模型实际 op type
```

用 `PreRunBegin` 里列出的**实际**名字写 pattern 或 `GNode` type 判断（如匹配 `MatMulV2`/`BatchMatMulV2` 而非前端脚本里的 `MatMul`）。

- `<soc>` 用 `scripts/detect_soc_version.sh` 取真实完整值，不手写不猜。
- **函数式 graph pass 同样适用**：凡源码里要按 `GNode` type/name 匹配现有节点，都必须先从 `PreRunBegin` dump 取真实 GE op type。

## 自查

- 匹配用的 op type，是不是逐个对照过 `PreRunBegin` dump 里的实际名？
- 是否因导入改名而用了 `MatMulV2`/`BatchMatMulV2` 等，而非前端脚本里的框架名？
- dump 里能不能看到目标结构被命中（对照 `dump-log-diff-checklist.md`）？
