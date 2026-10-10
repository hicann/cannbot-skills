#!/bin/bash
# HCCL-VM 组合测试脚本 (适配新 checker: hcomm-master/test/hccl_vm)
#
# 与旧版差异:
#   - 安装目录自动探测为 <脚本目录>/hccl_vm_install，可用 --install-dir 覆盖
#   - CANN 环境用实际路径 (默认 ${CANN_HOME}/set_env.sh)，可用 --cann 覆盖
#   - 不再 source hccl_config.sh (其内置 CANN 路径在本环境不存在)，改为脚本内直接设置全部环境变量
#   - mpirun 增加 --oversubscribe (新 README 用法)
#   - 修正算子二进制名: alltoall->alltoall_test, alltoallv->alltoallv_test; 移除不存在的 gather
#   - 模式切换用 export 替代 sed 改 hccl_config.sh
#
# 每个用例的执行流程:
#   1. ./hccl-vm start <cluster> < case_script.sh   (进入 hvm 会话)
#   2.   hccl-vm plugin install @runner              (可选)
#   3.   hccl-vm mock-comm <id>
#   4.   mpirun --oversubscribe ... -c 1
#   5.   hccl-vm plugin run @checker
#   6.   exit                                         (退出 hvm 会话)
#   7. 回到本脚本，解析结果，继续下一个用例
#
# 用法: bash run_test.sh [OPTIONS]
#   --cann PATH              CANN set_env.sh 路径 (默认: ${CANN_HOME}/set_env.sh)
#   --install-dir PATH       hccl_vm_install 安装目录 (默认: <脚本目录>/hccl_vm_install)
#   -m, --mode MODE          运行模式，逗号分隔: CCU_SCHED,CCU_MS,AI_CPU
#   -c, --cluster CLUSTER    集群拓扑文件 (默认: ascend950_cluster_4_server_normal.yaml)
#   -o, --operator OP        算子，逗号分隔 (默认: reduce_scatter)
#   -d, --dtype TYPE         数据类型，逗号分隔 (默认: int32)
#   -s, --size RANGE         数据量范围，如 64-1048576 或 64 (默认: 64)
#   -t, --comm-domain ID     通信域编号，逗号分隔 (默认: 112)
#   --env K=V                透传环境变量到测试进程 (可多次, 如 --env HCCL_ALGO=xxx)
#   --expect-algo NAME       断言日志中选中的算法为 NAME, 不命中记 FAIL (防假 PASS)
#   -n, --iter NUM           迭代次数 (默认: 1)
#   -w, --warmup NUM         预热次数 (默认: 0)
#   --timeout SEC            单用例超时秒数 (默认: 180，即3分钟)
#   --runner                 启用 Runner 插件
#   --no-checker             不执行 Checker 校验
#   --dry-run                仅打印命令，不执行
#   -h, --help               帮助
#
# 示例:
#   bash run_test.sh -m CCU_MS,AI_CPU -o allreduce,reduce_scatter -d int32,fp16 -s 64-8192 -t 112,118 --runner
#   bash run_test.sh -m CCU_SCHED,CCU_MS,AI_CPU -o allgather -d int16 -s 64-1M -t 112,121 --runner
#   bash run_test.sh --dry-run -o allreduce -d int32 -t 112
#   # 定向验证新算法 (轨道A): 强制选到新算法并断言命中
#   bash run_test.sh -m AI_CPU -o allreduce -d int32 -s 64 -t 118 --env HCCL_ALGO='<已核验DSL配置>' --expect-algo <新算法名>
#   # 存量回归 (轨道B): 不设 HCCL_ALGO, 默认选择路径
#   bash run_test.sh -m AI_CPU -o allreduce -d int32 -s 64 -t 118

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${SCRIPT_DIR}/hccl_vm_install"
CANN_SET_ENV="${CANN_HOME:-}/set_env.sh"
LOG_DIR="${SCRIPT_DIR}/test_logs"
RESULT_FILE=""
PHASE="test"
ROLE="ordinary"
SOURCE_ID=""
ARTIFACT=""
TEST_BIN_DIR=""
EVIDENCE="${SCRIPT_DIR}/evidence.py"

# ==================== 默认配置 ====================
MODES="CCU_SCHED"
CLUSTER="ascend950_cluster_4_server_normal.yaml"
OPERATORS="reduce_scatter"
DTYPES="int32"
SIZE_RANGE="64"
COMM_DOMAINS="112"
ITER_NUM=1
WARMUP_NUM=0
ENABLE_RUNNER=0
ENABLE_CHECKER=1
DRY_RUN=0
TIMEOUT=180
PLUGIN_WAIT=8    # exit 前等待秒数，给 checker/runner 插件收尾，避免 5s 内未退出留孤儿
EXPECT_ALGO=""   # 期望选中的算法名 (非空时断言日志命中, 不命中记 FAIL)
ENV_VARS=()      # 透传环境变量 (K=V 列表, export 后子进程继承)

# ==================== 映射表 ====================
# 算子名 -> hccl_test 二进制名 (已按实际编译产物校正)
declare -A OP_BIN=(
    [allgather]="all_gather_test"
    [allreduce]="all_reduce_test"
    [alltoall]="alltoall_test"
    [alltoallv]="alltoallv_test"
    [reduce]="reduce_test"
    [reduce_scatter]="reduce_scatter_test"
    [scatter]="scatter_test"
    [broadcast]="broadcast_test"
)

# 算子额外的命令行参数 (reduce 类需要 -o sum)
declare -A OP_EXTRA=(
    [allgather]=""
    [allreduce]="-o sum"
    [alltoall]=""
    [alltoallv]=""
    [reduce]="-o sum"
    [reduce_scatter]="-o sum"
    [scatter]=""
    [broadcast]=""
)

declare -A COMM_RANKS=(
    [111]=1  [112]=2  [114]=4  [118]=8
    [121]=2  [122]=4  [124]=8
    [128]=16 [12_2_4]=6 [141]=8 [142]=12 [144]=16
)

# ==================== 统计 ====================
TOTAL=0
PASSED=0
FAILED=0
FAILED_LIST=()
PASS_WITH_ERR=0
PASS_ERR_LIST=()

# ==================== 参数解析 ====================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --log-dir)        LOG_DIR="$2"; shift 2 ;;
            --phase)          PHASE="$2"; shift 2 ;;
            --role)           ROLE="$2"; shift 2 ;;
            --source-id)      SOURCE_ID="$2"; shift 2 ;;
            --artifact)       ARTIFACT="$2"; shift 2 ;;
            --test-bin-dir)   TEST_BIN_DIR="$2"; shift 2 ;;
            --cann)           CANN_SET_ENV="$2"; shift 2 ;;
            --install-dir)    INSTALL_DIR="$2"; shift 2 ;;
            -m|--mode)        MODES="$2"; shift 2 ;;
            -c|--cluster)     CLUSTER="$2"; shift 2 ;;
            -o|--operator)    OPERATORS="$2"; shift 2 ;;
            -d|--dtype)       DTYPES="$2"; shift 2 ;;
            -s|--size)        SIZE_RANGE="$2"; shift 2 ;;
            -t|--comm-domain) COMM_DOMAINS="$2"; shift 2 ;;
            --env)            ENV_VARS+=("$2"); shift 2 ;;
            --expect-algo)    EXPECT_ALGO="$2"; shift 2 ;;
            -n|--iter)        ITER_NUM="$2"; shift 2 ;;
            -w|--warmup)      WARMUP_NUM="$2"; shift 2 ;;
            --runner)         ENABLE_RUNNER=1; shift ;;
            --no-checker)     ENABLE_CHECKER=0; shift ;;
            --timeout)        TIMEOUT="$2"; shift 2 ;;
            --plugin-wait)    PLUGIN_WAIT="$2"; shift 2 ;;
            --dry-run)        DRY_RUN=1; shift ;;
            -h|--help)        usage; exit 0 ;;
            *) echo -e "${RED}未知参数: $1${NC}"; usage; exit 1 ;;
        esac
    done
}

usage() {
    cat << 'EOF'
HCCL-VM 组合测试脚本 (适配新 checker)

用法: bash run_test.sh [OPTIONS]

选项:
  --log-dir PATH        日志父目录；每轮自动建立唯一 run-id 子目录
  --role ROLE           ordinary / baseline / directed / regression（默认 ordinary）
  --phase NAME          自定义标签，不决定测试角色或绕过身份检查
  --source-id ID        commit + 工作区快照标识（可由 --artifact 提供）
  --artifact PATH       archive_package.py 生成的 artifact.json（基线检查需要）
  --test-bin-dir PATH   测试程序目录（默认 CANN tools/hccl_test/bin）
  --cann PATH            CANN set_env.sh 路径 (默认: ${CANN_HOME}/set_env.sh)
  --install-dir PATH     hccl_vm_install 安装目录 (默认: <脚本目录>/hccl_vm_install)
  -m, --mode MODE        运行模式，逗号分隔 (默认: CCU_SCHED)
                         支持: CCU_SCHED,CCU_MS,AI_CPU
                         示例: -m CCU_SCHED,CCU_MS,AI_CPU
  -c, --cluster FILE     集群拓扑文件 (默认: ascend950_cluster_4_server_normal.yaml)
  -o, --operator OP      算子，逗号分隔 (默认: reduce_scatter)
                         支持: allgather,allreduce,alltoall,alltoallv,
                               reduce,reduce_scatter,scatter,broadcast
  -d, --dtype TYPE       数据类型，逗号分隔 (默认: int32)
                         支持: int8,int16,int32,fp16,fp32,uint8,uint16,
                               uint32,bfp16,hif8,fp8e4m3,fp8e5m2,fp8e8m0
  -s, --size RANGE       数据量范围 (默认: 64)
                         格式: 64          单个值
                               64-8192     从64按2倍递增到8192
                               64-1M       支持K/M后缀
                               64,128,256  逗号枚举
  -t, --comm-domain ID   通信域编号，逗号分隔 (默认: 112)
                         可选: 111(1r),112(2r),114(4r),118(8r),121(2r跨server),
                               122(4r跨server),124(8r跨server),
                               128(16r),12_2_4(6r非对称),141(8r),142(12r),144(16r)
  --env K=V              透传环境变量到测试进程 (可多次)
                         示例: --env HCCL_ALGO='<已核验DSL配置>'  (格式以 HCCL 仓内实现为准)
  --expect-algo NAME     断言日志中选中的算法为 NAME (取自 "the selected algo type is")
                         不命中记 FAIL; 用例名追加 _algo-NAME 便于区分定向/回归两轨
  -n, --iter NUM         迭代次数 (默认: 1)
  -w, --warmup NUM       预热次数 (默认: 0)
    --runner               启用 Runner 插件 (模拟执行，-c 1 校验才有意义)
    --no-checker           不执行 Checker 校验
    --timeout SEC          单用例超时秒数 (默认: 180，即3分钟)
  --dry-run              仅打印命令，不执行
  -h, --help             帮助

示例:
  bash run_test.sh -m CCU_MS -o allreduce,reduce_scatter -d int32,fp16 -s 64-8192 -t 112,118 --runner
  bash run_test.sh -m AI_CPU -o allgather -d int16 -s 64-1M -t 112,121 --runner
  bash run_test.sh --dry-run -o allreduce -d int32 -t 112
EOF
}

# ==================== 工具函数 ====================
log_info()  { echo -e "${BLUE}[INFO]  $1${NC}"; }
log_pass()  { echo -e "${GREEN}[PASS]  $1${NC}"; }
log_fail()  { echo -e "${RED}[FAIL]  $1${NC}"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $1${NC}"; }
log_case()  { echo -e "${CYAN}[CASE]  $1${NC}"; }

parse_csv() {
    echo "$1" | tr ',' '\n' | sed '/^$/d'
}

parse_size() {
    local val="$1"
    val="${val^^}"
    if [[ "$val" == *K ]]; then
        echo $((${val%K} * 1024))
    elif [[ "$val" == *M ]]; then
        echo $((${val%M} * 1048576))
    elif [[ "$val" == *G ]]; then
        echo $((${val%G} * 1073741824))
    else
        echo "$val"
    fi
}

expand_sizes() {
    local range_str="$1"
    local result=""
    IFS=',' read -ra parts <<< "${range_str}"
    for part in "${parts[@]}"; do
        part=$(echo "$part" | xargs)
        if [[ "$part" == *"-"* ]]; then
            local begin_str end_str
            begin_str=$(echo "$part" | cut -d'-' -f1)
            end_str=$(echo "$part" | cut -d'-' -f2)
            local begin=$(parse_size "$begin_str")
            local end=$(parse_size "$end_str")
            if [[ "$begin" -eq "$end" ]]; then
                result="${result} ${begin}"
            elif [[ "$begin" -lt "$end" ]]; then
                local cur="$begin"
                while [[ "$cur" -le "$end" ]]; do
                    result="${result} ${cur}"
                    cur=$((cur * 2))
                done
                if [[ "$cur" -ne "$end" ]]; then
                    result="${result} ${end}"
                fi
            else
                result="${result} ${begin}"
            fi
        else
            result="${result} $(parse_size "$part")"
        fi
    done
    echo "$result" | tr ' ' '\n' | sort -n -u | sed '/^$/d'
}

get_rank_num() {
    local comm_id="$1"
    if [[ -n "${COMM_RANKS[$comm_id]}" ]]; then
        echo "${COMM_RANKS[$comm_id]}"
    else
        local yaml_file="${INSTALL_DIR}/config/topo_meta/${comm_id}.yaml"
        if [[ -f "$yaml_file" ]]; then
            grep -oP 'rankNum:\s*\K\d+' "$yaml_file" 2>/dev/null || echo "2"
        else
            echo "2"
        fi
    fi
}

format_size() {
    local s="$1"
    if [[ "$s" -ge 1073741824 ]] && [[ $((s % 1073741824)) -eq 0 ]]; then
        echo "$((s / 1073741824))G"
    elif [[ "$s" -ge 1048576 ]] && [[ $((s % 1048576)) -eq 0 ]]; then
        echo "$((s / 1048576))M"
    elif [[ "$s" -ge 1024 ]] && [[ $((s % 1024)) -eq 0 ]]; then
        echo "$((s / 1024))K"
    else
        echo "$s"
    fi
}

# ==================== 环境准备 ====================
setup_env() {
    # 1. 加载 CANN 环境 (设置 ASCEND_HOME_PATH 等)
    if [[ -n "${CANN_SET_ENV}" ]]; then
        if [[ -f "${CANN_SET_ENV}" ]]; then
            source "${CANN_SET_ENV}"
            log_info "已加载 CANN: ${CANN_SET_ENV}"
        else
            log_fail "CANN set_env.sh 不存在: ${CANN_SET_ENV} (用 --cann 指定，或确保 \$CANN_HOME 已设置并 source hccl-env.sh)"
            exit 1
        fi
    else
        log_fail "CANN_HOME 未设置，请先 source hccl-env.sh 或用 --cann 指定 set_env.sh 路径"
        exit 1
    fi

    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        log_fail "ASCEND_HOME_PATH 未设置，请检查 CANN 环境 (--cann)"
        exit 1
    fi

    # 2. 设置 HCCL-VM 运行所需的全部环境变量 (等价于 source hccl_config.sh，
    #    但用本机实际 CANN 路径，并补全新版缺失的 LD_LIBRARY_PATH)
    export HCCL_VM_INSTALL_DIR="${INSTALL_DIR}"
    export RANK_TABLE_FILE="${INSTALL_DIR}/data/ranktable.json"
    export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/devlib:${LD_LIBRARY_PATH}"
    export ASCEND_GLOBAL_LOG_LEVEL=1
    export ASCEND_SLOG_PRINT_TO_STDOUT=1
    export HCCL_INDEPENDENT_OP=1
    export QEMU_LD_PREFIX=/usr/aarch64-linux-gnu
    export LANG=zh_CN.UTF-8
    export LC_ALL=zh_CN.UTF-8

    # 3. 透传用户指定的环境变量 (如 HCCL_ALGO, 子进程自动继承)
    for kv in "${ENV_VARS[@]}"; do
        if [[ "$kv" != *=* ]]; then
            log_fail "--env 参数格式错误: '${kv}' (应为 K=V)"
            exit 1
        fi
        case "${kv%%=*}" in
            HCCL_VM_INSTALL_DIR|HCCL_OP_EXPANSION_MODE|HCCL_ENABLE_OPEN_CCU)
                log_fail "安装路径与引擎变量由 --install-dir 和 --mode 管理，不能用 --env 覆盖"
                exit 1 ;;
        esac
        export "$kv"
        log_info "透传环境变量: ${kv}"
    done
    if [[ -n "${EXPECT_ALGO}" ]]; then
        log_info "期望算法: ${EXPECT_ALGO} (未命中将记 FAIL)"
    fi
}

# ==================== 执行单个用例 ====================
run_one_case() {
    local mode="$1"
    local op="$2"
    local dtype="$3"
    local size="$4"
    local comm_id="$5"
    local rank_num="$6"
    local case_idx="$7"

    local bin="${OP_BIN[$op]}"
    if [[ -z "$bin" ]]; then
        log_fail "未知算子: $op"
        return 1
    fi
    local extra="${OP_EXTRA[$op]}"
    local test_bin="${TEST_BIN_DIR:-${ASCEND_HOME_PATH}/tools/hccl_test/bin}/${bin}"
    local case_name="${mode}_${op}_${dtype}_$(format_size ${size})_comm${comm_id}"
    [[ -n "${EXPECT_ALGO}" ]] && case_name="${case_name}_algo-${EXPECT_ALGO}"
    local case_log="${LOG_DIR}/${case_idx}_${case_name}.log"
    local tmp_script="${LOG_DIR}/case_${case_idx}.sh"

    log_case "[${case_idx}/${TOTAL}] ${case_name} (${rank_num}ranks)"

    export HWLOC_COMPONENTS=-gl,-opencl
    # 生成注入 hvm 会话的脚本
    # 每条命令前 echo 打印，命令本身同步阻塞执行，下一条等上一条结束后才执行
    {
        echo "#!/bin/bash"
        echo ""
        if [[ ${ENABLE_RUNNER} -eq 1 ]]; then
            echo "echo '>>> hccl-vm plugin install @runner'"
            echo "hccl-vm plugin install @runner"
            echo "echo \"[CMD DONE] hccl-vm plugin install @runner\""
        fi
        echo "echo '>>> hccl-vm mock-comm ${comm_id}'"
        echo "hccl-vm mock-comm ${comm_id}"
        echo "echo \"[CMD DONE] hccl-vm mock-comm ${comm_id}\""
        local mpi_check_flag=1
        local quoted_test_bin
        printf -v quoted_test_bin '%q' "$test_bin"
        local mpi_cmd="mpirun --allow-run-as-root --oversubscribe -np ${rank_num} ${quoted_test_bin} -b ${size} -e ${size} -d ${dtype} ${extra} -w ${WARMUP_NUM} -n ${ITER_NUM} -c ${mpi_check_flag} < /dev/null"
        echo "echo '>>> ${mpi_cmd}'"
        echo "${mpi_cmd}"
        echo "echo \"[CMD DONE] mpirun (exit=\$?)\""
        if [[ ${ENABLE_CHECKER} -eq 1 ]]; then
            echo "echo '>>> hccl-vm plugin run @checker'"
            echo "hccl-vm plugin run @checker"
            echo "echo \"[CMD DONE] hccl-vm plugin run @checker\""
        fi
        echo "echo '>>> sleep ${PLUGIN_WAIT}s (给 checker/runner 插件收尾时间)'"
        echo "sleep ${PLUGIN_WAIT}"
        echo "echo '>>> exit'"
        echo "exit"
    } > "${tmp_script}"
    chmod +x "${tmp_script}"

    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo -e "${YELLOW}  [DRY-RUN] cd ${INSTALL_DIR}/bin && ./hccl-vm start ${CLUSTER} < ${tmp_script}${NC}"
        echo -e "${YELLOW}  [DRY-RUN] 脚本内容:${NC}"
        sed 's/^/    /' "${tmp_script}"
        rm -f "${tmp_script}"
        return 0
    fi

    # 执行: 启动 hvm, 注入命令, exit 退出
    # 环境变量已在父进程 setup_env() 设置并 export，子进程自动继承
    # 用 setsid 起独立进程组，超时后杀整棵进程树，fail 后继续下一个用例
    local MY_PID=$$
    local start_time=$(date +%s)
    setsid bash -c "cd '${INSTALL_DIR}/data' && rm -f sqe_info_rank_* mc_instr_info_rank_* all_rank_input_output.txt 2>/dev/null; rm -f /dev/shm/HcclCommPool /dev/shm/HcclAicpuData /dev/shm/DEV1_* /dev/shm/DEV2_* /dev/shm/ra_sock_* 2>/dev/null; rm -rf /dev/shm/hccl_sync 2>/dev/null; cd '${INSTALL_DIR}/bin' && ./hccl-vm start '${CLUSTER}' < '${tmp_script}' > '${case_log}' 2>&1" &
    local hvm_pid=$!

    # 轮询等待进程结束或超时
    local timed_out=0
    while kill -0 ${hvm_pid} 2>/dev/null; do
        local now=$(date +%s)
        if [[ $((now - start_time)) -ge ${TIMEOUT} ]]; then
            timed_out=1
            break
        fi
        sleep 2
    done

    if [[ ${timed_out} -eq 1 ]]; then
        log_warn "用例超时 (${TIMEOUT}s)，杀进程树..."
        # 杀整个进程组（setsid 保证所有子进程在同一组）
        local pgid=$(ps -o pgid= -p ${hvm_pid} 2>/dev/null | tr -d ' ')
        if [[ -n "${pgid}" ]]; then
            kill -9 -${pgid} 2>/dev/null || true
        fi
        kill -9 ${hvm_pid} 2>/dev/null || true
        wait ${hvm_pid} 2>/dev/null || true
        sleep 2
        # 清理超时残留的共享内存
        rm -f /dev/shm/HcclCommPool /dev/shm/HcclAicpuData /dev/shm/DEV1_* /dev/shm/DEV2_* /dev/shm/ra_sock_* 2>/dev/null
        rm -rf /dev/shm/hccl_sync 2>/dev/null
    else
        wait ${hvm_pid} 2>/dev/null || true
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    rm -f "${tmp_script}"

    log_info "日志: ${case_log}"

    # 提取并打屏算法选择（首个 + 末个）
    local algo_first=$(grep -oP 'the selected algo type is \K[A-Za-z][A-Za-z0-9]*' "${case_log}" 2>/dev/null | head -1)
    local algo_last=$(grep -oP 'the selected algo type is \K[A-Za-z][A-Za-z0-9]*' "${case_log}" 2>/dev/null | tail -1)
    if [[ -n "$algo_first" ]]; then
        if [[ "$algo_first" == "$algo_last" ]]; then
            log_info "算法: ${algo_first}"
        else
            log_info "算法(首个): ${algo_first} | 算法(末个): ${algo_last}"
        fi
    fi

    # 先持久化完整证据，再决定用例状态；解析失败不得沿用先前 PASS。
    local verdict="FAIL" detail="证据写入失败" evidence_result
    local timeout_args=()
    [[ ${timed_out} -eq 1 ]] && timeout_args+=(--timed-out)
    if evidence_result=$(python3 "$EVIDENCE" record --run-dir "$LOG_DIR" \
        --mode "$mode" --op "$op" --dtype "$dtype" --size "$size" --comm "$comm_id" \
        --log "$case_log" --test-binary "$test_bin" --duration "$duration" \
        --expected "$EXPECT_ALGO" "${timeout_args[@]}"); then
        IFS=$'\t' read -r verdict detail <<< "$evidence_result"
    fi
    detail="${detail} | algo first=${algo_first:-none}, last=${algo_last:-none}"

    # 输出结果
    if [[ "$verdict" == "PASS" ]]; then
        PASSED=$((PASSED + 1))
        # 检查 PASS 但日志含 error 的情况 (排除已知噪声)
        local err_line=$(grep -iE "\[error\]" "${case_log}" 2>/dev/null \
                         | grep -ivE "ErrorDecorator|InitApiError|ApiError|task_fail_callback|TaskFailCallBack|task_abort|exception dump" \
                         | head -1)
        if [[ -n "$err_line" ]]; then
            PASS_WITH_ERR=$((PASS_WITH_ERR + 1))
            local err_brief=$(echo "$err_line" | sed -E 's/^\[error\]\[[^]]*\]\[[^]]*\] //' | head -c 110)
            PASS_ERR_LIST+=("${case_name} | ${err_brief}")
            log_pass "${case_name} | ${detail} | ${duration}s"
            log_warn "  [PASS含error] ${err_brief}"
            echo "${case_name}|PASS(WARN)|${detail}|${err_brief}|${duration}s" >> "${RESULT_FILE}"
        else
            log_pass "${case_name} | ${detail} | ${duration}s"
            echo "${case_name}|PASS|${detail}|${duration}s" >> "${RESULT_FILE}"
        fi
    else
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("${case_name} | ${detail} | ${case_log}")
        log_fail "${case_name} | ${detail} | ${duration}s"
        echo "${case_name}|FAIL|${detail}|${duration}s" >> "${RESULT_FILE}"
    fi

    echo ""

    if [[ "$verdict" == "PASS" ]]; then
        return 0
    else
        return 1
    fi
}

# ==================== 主流程 ====================
main() {
    parse_args "$@"

    local mode_list=($(parse_csv "${MODES}"))
    local op_list=($(parse_csv "${OPERATORS}"))
    local dtype_list=($(parse_csv "${DTYPES}"))
    local comm_list=($(parse_csv "${COMM_DOMAINS}"))
    local size_list=($(expand_sizes "${SIZE_RANGE}"))

    for op in "${op_list[@]}"; do
        [[ -n "${OP_BIN[$op]:-}" ]] || { log_fail "未知算子: $op"; exit 1; }
    done
    for mode in "${mode_list[@]}"; do
        case "$mode" in AI_CPU|CCU_SCHED|CCU_MS) ;; *) log_fail "未知模式: $mode"; exit 1 ;; esac
    done

    TOTAL=$((${#mode_list[@]} * ${#op_list[@]} * ${#dtype_list[@]} * ${#size_list[@]} * ${#comm_list[@]}))

    if [[ ${TOTAL} -eq 0 ]]; then
        log_fail "没有生成任何用例，请检查参数"
        exit 1
    fi

    mkdir -p "$LOG_DIR" || exit 1
    LOG_DIR=$(mktemp -d "${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S)_XXXXXX") || exit 1
    LOG_DIR=$(cd "$LOG_DIR" && pwd)
    RESULT_FILE="${LOG_DIR}/result.txt"
    echo "# HCCL-VM 测试结果 - $(date '+%Y-%m-%d %H:%M:%S')" > "${RESULT_FILE}"
    echo "# 用例|结果|详情|耗时" >> "${RESULT_FILE}"

    echo "=========================================="
    echo "  HCCL-VM 组合测试 (新 checker)"
    echo "=========================================="
    echo ""
    echo "  安装目录:   ${INSTALL_DIR}"
    echo "  CANN:       ${CANN_SET_ENV}"
    echo "  运行模式:   ${mode_list[*]}"
    echo "  集群拓扑:   ${CLUSTER}"
    echo "  算子:       ${op_list[*]}"
    echo "  数据类型:   ${dtype_list[*]}"
    echo "  数据量:     $(echo ${size_list[*]} | sed 's/ /\,/g')"
    echo "  通信域:     ${comm_list[*]}"
    echo "  透传变量:   $([ ${#ENV_VARS[@]} -gt 0 ] && echo "${ENV_VARS[*]}" || echo '(无)')"
    echo "  期望算法:   ${EXPECT_ALGO:-'(不断言)'}"
    echo "  Runner:     $([ ${ENABLE_RUNNER} -eq 1 ] && echo ON || echo OFF)"
    echo "  Checker:    $([ ${ENABLE_CHECKER} -eq 1 ] && echo ON || echo OFF)"
    echo "  总用例数:   ${TOTAL}"
    echo ""
    echo "=========================================="
    echo ""

    if [[ ! -d "${INSTALL_DIR}" ]]; then
        log_fail "安装目录不存在: ${INSTALL_DIR}"
        log_warn "请先构建 checker:"
        log_warn "  cd ${SCRIPT_DIR}"
        log_warn "  source ${CANN_SET_ENV}"
        log_warn "  export HCOMM_CODE_HOME=<hcomm源码路径>"
        log_warn "  export HCCL_CODE_HOME=<hccl源码路径>"
        log_warn "  bash ./build.sh --full"
        log_warn "或用 --install-dir 指定已构建的安装目录"
        exit 1
    fi

    if [[ ! -d "${INSTALL_DIR}/bin" ]]; then
        log_fail "安装目录缺少 bin/ 子目录: ${INSTALL_DIR}/bin"
        exit 1
    fi

    # 加载 CANN 环境并设置 HCCL-VM 运行环境变量
    setup_env
    export HWLOC_COMPONENTS=-gl,-opencl
    python3 "${SCRIPT_DIR}/parameters.py" --dtypes "$DTYPES" \
        --test-bin-dir "${TEST_BIN_DIR:-${ASCEND_HOME_PATH}/tools/hccl_test/bin}" || exit 1
    # 全部用例清单在测试前落盘；中断后 check 能识别缺项。
    python3 "$EVIDENCE" init --run-dir "$LOG_DIR" --install-dir "$INSTALL_DIR" \
        --cann-home "$ASCEND_HOME_PATH" --modes "$MODES" --ops "$OPERATORS" --dtypes "$DTYPES" \
        --sizes "$(IFS=,; echo "${size_list[*]}")" --comms "$COMM_DOMAINS" --cluster "$CLUSTER" \
        --role "$ROLE" --expected "$EXPECT_ALGO" --phase "$PHASE" --source-id "$SOURCE_ID" --artifact "$ARTIFACT" \
        --cann-script "$CANN_SET_ENV" --test-bin-dir "${TEST_BIN_DIR:-${ASCEND_HOME_PATH}/tools/hccl_test/bin}" \
        --iterations "$ITER_NUM" --warmup "$WARMUP_NUM" --runner "$ENABLE_RUNNER" --dry-run "$DRY_RUN" || exit 1
    if [[ "$DRY_RUN" -eq 0 ]]; then
        # 同一用户的脚本实例共用 /dev/shm，不能并行清理或采集。
        exec 9>"${TMPDIR:-/tmp}/hccl-vm-${UID}.lock"
        flock -n 9 || { log_fail "另一个 hccl-vm run.sh 正在使用共享环境"; exit 1; }
    fi

    local case_idx=0
    for mode in "${mode_list[@]}"; do
        # 切换模式: 直接 export (替代旧版 sed 改 hccl_config.sh)
        export HCCL_OP_EXPANSION_MODE="${mode}"
        case "${mode}" in
            CCU_*) export HCCL_ENABLE_OPEN_CCU=1 ;;
            *)     unset HCCL_ENABLE_OPEN_CCU ;;
        esac
        log_info "切换模式: ${mode} (HCCL_OP_EXPANSION_MODE=${mode})"

        for op in "${op_list[@]}"; do
            if [[ -z "${OP_BIN[$op]}" ]]; then
                log_warn "跳过未知算子: ${op}"
                continue
            fi
            for dtype in "${dtype_list[@]}"; do
                for size in "${size_list[@]}"; do
                    for comm_id in "${comm_list[@]}"; do
                        case_idx=$((case_idx + 1))
                        local rank_num=$(get_rank_num "${comm_id}")
                        run_one_case "${mode}" "${op}" "${dtype}" "${size}" "${comm_id}" "${rank_num}" "${case_idx}" || true
                    done
                done
            done
        done
    done

    local evidence_status=0
    if [[ "$DRY_RUN" -eq 0 && ( "$ROLE" == baseline || "$ROLE" == regression ) ]]; then
        python3 "$EVIDENCE" check "$LOG_DIR" || evidence_status=1
    fi
    # ==================== 最终报告 ====================
    echo ""
    echo "=========================================="
    echo "  测试报告"
    echo "=========================================="
    echo ""
    echo "  总用例:  ${TOTAL}"
    echo -e "  通过:    ${GREEN}${PASSED}${NC}"
    echo -e "  失败:    ${RED}${FAILED}${NC}"
    echo -e "  PASS但含error: ${YELLOW}${PASS_WITH_ERR}${NC}"
    echo "  结果文件: ${RESULT_FILE}"
    echo "  结构化证据: ${LOG_DIR}/results.jsonl"
    echo ""

    if [[ ${#PASS_ERR_LIST[@]} -gt 0 ]]; then
        echo -e "  ${YELLOW}PASS 但日志含 error 的用例:${NC}"
        for e in "${PASS_ERR_LIST[@]}"; do
            echo -e "    ${YELLOW}- ${e}${NC}"
        done
        echo ""
    fi

    if [[ ${#FAILED_LIST[@]} -gt 0 ]]; then
        echo "  失败用例:"
        for f in "${FAILED_LIST[@]}"; do
            echo -e "    ${RED}- ${f}${NC}"
        done
        echo ""
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log_info "dry-run 完成，仅生成命令，未执行验证"
        exit 0
    fi
    if [[ ${FAILED} -eq 0 && ${evidence_status} -eq 0 && ${PASSED} -eq ${TOTAL} ]]; then
        log_pass "所有 ${TOTAL} 个用例通过!"
        exit 0
    else
        log_fail "${FAILED}/${TOTAL} 个用例失败"
        exit 1
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
