{{LICENSE}}

#ifndef {{GUARD}}
#define {{GUARD}}

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

// Mesh 1D one-shot 骨架：
//   ① 主流 LocalCopy(input → output)
//   ② 每个从流把整份 input 写进一个对端的 scratch 分段，同时收下对端的那一份
//   ③ 主流把 scratch 里其余 rankSize-1 份 LocalReduce 进 output
// scratch 占用 = templateRankSize_ × 单份数据量
class {{CLASS}} : public InsAlgTemplateBase {
public:
    static constexpr TemplateProp props = {.algoType = AlgoType::{{ALGOTYPE}}};
    {{CLASS}}() = default;   // FastLaunch 路径需要，勿删
    explicit {{CLASS}}(
        const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>>& subCommRanks);
    ~{{CLASS}}() override;

    std::string Describe() const override
    {
        std::string info = "Template of {{DESC}} with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

private:
    HcclResult CalcSlice(const u64 dataSize, RankSliceInfo& sliceInfoVec) const;
    HcclResult RunExchange(
        const OpParam& param, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const std::vector<ThreadHandle>& threads, const TemplateDataParams& tempAlgParams,
        const RankSliceInfo& sliceInfoVec);
    HcclResult PostLocalReduce(
        const OpParam& param, const std::vector<ThreadHandle>& threads, const TemplateDataParams& tempAlgParams,
        const RankSliceInfo& sliceInfoVec);

    bool needAicpuReduce_{false};
    u64 processSize_{0};
    u64 count_{0};
};

} // namespace ops_hccl

#endif // {{GUARD}}
