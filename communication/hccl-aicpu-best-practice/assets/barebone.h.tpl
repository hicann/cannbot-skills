{{LICENSE}}

#ifndef {{GUARD}}
#define {{GUARD}}

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class {{CLASS}} : public InsAlgTemplateBase {
public:
    // 算法属性。TemplateProp 当前只有 algoType 一个成员，枚举见 src/common/alg_parse.h
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

    // 参与 cost model 打分；static，由 executor 通过模板参数静态调用
    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

    // executor 直接消费 GetRes 时必须实现；GetThreadNum 按实际调用点判断：
    // HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    // u64 GetThreadNum() const override;

    // 按 spec 添加切片、阶段和资源状态；无归约算子不引入归约成员。
};

} // namespace ops_hccl

#endif // {{GUARD}}
