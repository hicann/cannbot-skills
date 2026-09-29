# environment.yaml 结构

本 skill 只维护一份当前生效的结构，不写结构版本号。

同目录下的 `env_selected_soc.yaml` 记录默认选中卡的探测信息，字段说明见 [env-selected-soc-schema.md](env-selected-soc-schema.md)。

实际产物的每个非空数据行都会追加中文注释。普通行统一对齐到适中的注释列；路径、诊断等长行只在值后保留两个空格，不参与全局对齐。

顶层字段严格按照阅读优先级排列：

| 字段 | 含义 |
|---|---|
| final_status | 最终结论、DSL 编译门禁、执行范围以及允许/阻止动作 |
| stage_status | CANNBotDSL 导入、DSL 翻译、DSL 编译和 NPU 运行环境的阶段结果 |
| software_environment | Python、CANNBotDSL、CANN Toolkit、OPP/vendor、Simulator 和工具 |
| server_information | 当前进程可见设备与服务器物理设备诊断 |
| checked_at | UTC 检查时间 |

核心结构示例：

    final_status:
      status: WARNING
      summary: DSL 生成与编译可用，但当前不能进行 NPU 真机验证
      compile_gate: PASS
      execution_mode: compile_only
      reason: 编译能力已确认，但 NPU 不可用；仅跳过真机验证
      allowed_actions:
        - dsl_generate
        - dsl_compile
      blocked_actions:
        - npu_correctness
        - npu_performance
    stage_status:
      cannbotdsl_import:
        status: PASS
        code: OK
        summary: 当前解释器可以导入 CANNBotDSL
        detail: ''
        possible_causes: []
        actions: []
      dsl_translation:
        status: PASS
        code: OK
        summary: CANNBotDSL 前端与 AscendC 翻译通过
        detail: so_path 为空
        possible_causes: []
        actions: []
      dsl_compilation:
        status: PASS
        code: OK
        summary: CANNBotDSL 编译后端通过并生成 .so
        detail: so=<runtime-path> size=<bytes>
        possible_causes: []
        actions: []
      npu_runtime:
        status: FAIL
        code: NPU_UNAVAILABLE
        summary: PyTorch NPU 后端未识别到可用设备
        detail: ''
        possible_causes: []
        actions: []
    software_environment:
      runtime:
        python_executable: <selected-python>
        python_version: <version>
        cannbotdsl_module_file: <runtime-path>
        cannbotdsl_version: <version-or-null>
      cann:
        status: PASS
        ascend_home_path: <runtime-path>
        cann_version: <version-or-null>
        set_env_script_exists: true
        ascend_opp_path: <runtime-path-or-null>
        opp_path_exists: true
        vendors: []
        simulator: []
        tools:
          msprof: <runtime-path-or-null>
          cannsim: <runtime-path-or-null>
        warnings: []
    server_information:
      npu:
        runtime_visible:
          scope: current_process_visible_devices
          available: false
          device_count: 0
          devices: []
        physical_inventory:
          source: npu-smi
          detected_count: 8
          devices:
            - physical_device_id: 0
              health: OK
          architecture:
            full_soc: <asys-chip-info-or-null>
            npu_arch: <asys-arch-info-or-null>
            source: asys
          warnings: []
          note: 物理设备清单用于环境诊断；设备名称不从 npu-smi Chip Name 推导，核数未知时不猜测
    checked_at: <UTC-time>

`final_status.status` 的含义：

| 值 | 含义 |
|---|---|
| PASS | 编译能力和 NPU 运行环境均通过，可执行完整验证 |
| WARNING | 编译能力通过，但 NPU 运行环境失败；仅能生成和编译 |
| FAIL | 编译能力链失败，当前只应进行环境诊断 |

当前进程可见设备数不等于服务器物理卡数；容器映射和设备可见性设置可能缩小该范围。主机清单优先使用 `npu-smi info -m`，健康状态使用结构化 health 查询，架构信息使用 asys 硬件报告。无法可靠获取的芯片核数不猜测。

产物先在内存中生成并用 `yaml.safe_load` 回读校验，校验通过后在目标目录内原子替换。
