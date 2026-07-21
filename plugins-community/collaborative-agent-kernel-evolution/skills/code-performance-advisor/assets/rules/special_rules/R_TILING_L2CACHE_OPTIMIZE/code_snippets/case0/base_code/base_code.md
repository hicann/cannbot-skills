# Base Code

```cpp
// 传统遍历：按行优先顺序
for (uint64_t mTileIndex = 0; mTileIndex < mTileCntL2; mTileIndex++) {
    for (uint64_t nTileIndex = 0; nTileIndex < nTileCntL2; nTileIndex++) {
        // 每次换行时，左矩阵数据被全部置换
        // 右矩阵数据也无法复用
        block_.UpdateBlockCnt(mTileIndex, nTileIndex);
        block_.InitBlockIndex(index);
        
        for (uint64_t j = 0; j < block_.params_.realRound; j++) {
            mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], isTransposeA);
            mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], isTransposeB);
            mm_.Iterate();
            mm_.GetTensorC(cGlobal_[block_.offset_.offsetC], enAtomic);
        }
    }
}
```