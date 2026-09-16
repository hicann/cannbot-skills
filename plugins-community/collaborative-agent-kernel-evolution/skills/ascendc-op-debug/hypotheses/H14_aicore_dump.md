---
id: H14
title: AI Core Dump 分析（msaicerr 工具链）
symptom: crash
when: always
root_cause: aicore_dump_analysis
evidence: tool_msaicerr
escalate_to: msaicerr
source: msaicerr-toolkit
---

## triggers
- 运行时收到 AI Core error，有 dump 文件生成
- 错误码 161xxx / 561xxx 系列
- aclGetRecentErrMsg 返回非空错误信息
- plog 中出现 AI Core exception 关键字

## read_target
- 检查 dump 目录：`ls ~/ascend/log/dump/` 或工程配置的 dump 路径
- grep plog 中的错误码：`grep -E "AICore|ERROR|DDR" ~/ascend/log/debug/plog/*.log | tail -50`

## code_pattern
```
# 这类 bug 通常无法从源码直接识别，需要 dump 分析
# 典型触发场景：
# - 访问越界的 GM 地址
# - 非法指令
# - 硬件异常（ECC 错误等）
```

## fix_template
```bash
# 步骤1：解析 AI Core 错误报告 —— 用仓内 tools/msaicerr-toolkit 技能，
# 它封装 CANN Toolkit 自带的 msaicerr.py（参数为 -p / -out）
python3 msaicerr.py -p ~/ascend/log/dump/ -out error_report/

# 步骤2：解析 tiling 数据（如有）—— 仓内无此工具，按
# op_host/<op>_custom_tiling.h 的字段顺序与类型直接 struct.unpack tiling.bin
# （详见 protocols/run_tools.md 的同名步骤）

# 步骤3：对照 error_report 中的地址，查 kernel 源码对应位置
```

## verify_cmd
- 修复后重跑，确认无新 AI Core error 生成
- 对照 protocols/parse_log.md 中的错误码表确认修复方向

## notes
- 此 hypothesis 是 Layer 3 兜底，代码审查（H01-H13）均未命中时使用
- 详细工具使用见 `protocols/run_tools.md` → msaicerr 章节
- 错误码对照见 `protocols/parse_log.md`
