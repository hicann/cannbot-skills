{{LICENSE}}

// 自定义算法工程骨架：不预设拓扑、peer、线程数、切片、归约或同步顺序。
// TODO: 按通过检查的 Dataflow Spec 实现资源、scratch、数据流与选路支持条件。
// 未完成时 CalcRes/KernelRun 明确失败，CalcCostCoeff 不提供候选；不能作为可运行算法。

#include "{{HEADER}}"

namespace ops_hccl {

{{CLASS}}::{{CLASS}}(
    const OpParam& param, const u32 rankId,
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

{{CLASS}}::~{{CLASS}}() {}

std::vector<CostModelParam> {{CLASS}}::CalcCostCoeff(CalcCostCoeffParam param)
{
    (void)param;
    // TODO: 按实际支持矩阵和启用方式提供候选，成本来自本算法的操作量。
    // 空返回表示尚未接入选路；不得复制 Mesh/NHR 成本或填假权重使检查通过。
    return {};
}

HcclResult {{CLASS}}::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)resourceRequest;
    // TODO: 从 spec 的逻辑 peer、thread/channel 映射和同步边推导申请量。
    // 使用目标版本 channel.h 中已核验的接口，不假定存在 CalcChannelRequestRing。
    // 若绑定 executor 直接消费 GetRes/GetThreadNum，还须实现对应方法并保持一致。
    HCCL_ERROR("[{{CLASS}}][CalcRes] Custom algorithm resource plan is not implemented.");
    return HcclResult::HCCL_E_INTERNAL;
}

u64 {{CLASS}}::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // TODO: 按 executor 的预算单位和单次 repeat 的最大占用推导，核对各 buffer 分支。
    // 0 仅为未实现占位值；CalcRes/KernelRun 的失败不能在预算完成前解除。
    return 0;
}

void {{CLASS}}::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    // TODO: 按实际从线程及 notify 配比填写；不默认使用 rankSize-1 个从线程。
    notifyIdxMainToSub.clear();
}

void {{CLASS}}::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    // TODO: 与资源申请、前置同步及各阶段后置同步的索引保持一致。
    notifyIdxSubToMain.clear();
}

HcclResult {{CLASS}}::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    (void)param;
    (void)tempAlgParams;
    (void)templateResource;
    // TODO: 按 spec 映射下列内容；具体阶段、循环嵌套和线程分配由算法决定。
    //   1. 验证资源、rank 映射、算子边界及当前 executor 布局契约。
    //   2. 区分 executor 分块、repeat 和算法 step，分别推导输入/输出/scratch 地址。
    //   3. 实现每步传输/归约与数据依赖；仅在实际使用从线程时安排主从同步。
    //   4. 证明接收数据可见、scratch 可复用以及结果落入本阶段的目标槽位。
    // 空数据、单 rank、无归约算子的行为也按契约实现，不套用 AllReduce 早退。
    HCCL_ERROR("[{{CLASS}}][KernelRun] Custom algorithm dataflow is not implemented.");
    return HcclResult::HCCL_E_INTERNAL;
}

} // namespace ops_hccl
