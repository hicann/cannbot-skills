// 共享声明：sync14_ 最小可编译上下文（示例间共用，避免重复代码）
#ifndef SYNC14_COMMON_H_
#define SYNC14_COMMON_H_

struct CopyParams {};

struct Buffer {
    void *GetPhyAddr();
};

Buffer ubBiasOutTotalBuffer_[4096];
Buffer outputGm_[4096];
int eventId[4];
CopyParams copyParams;
constexpr int TILE_SIZE = 128;

#endif  // SYNC14_COMMON_H_