# P2P 迁移模板

## cudaMemcpyPeerAsync 跨 Device 验证模板

当 CUDA 用例包含跨 Device `cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, ...)`，迁移到 CANN 兼容层验证时，按下面模板生成 CANN 侧验证代码。该模板用于验证 Runtime 兼容层，不改变业务源码里的 API 名称。

```cpp
int canAccessPeer = 0;
if (cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1) != cudaSuccess || canAccessPeer == 0) {
    // SKIP: 当前拓扑不支持 0->1 P2P。
    return;
}

cudaSetDevice(0);
cudaDeviceEnablePeerAccess(1, 0);

cudaSetDevice(1);
cudaDeviceEnablePeerAccess(0, 0);

// 运行脚本中同时设置：
// export CUDA_COMPAT_DEVICE_MALLOC_POLICY=p2p

void *srcOn0 = nullptr;
void *dstOn1 = nullptr;
cudaStream_t streamOn1 = nullptr;

cudaSetDevice(0);
cudaMalloc(&srcOn0, bytes);
cudaMemcpy(srcOn0, hostIn, bytes, cudaMemcpyHostToDevice);

cudaSetDevice(1);
cudaMalloc(&dstOn1, bytes);
cudaStreamCreate(&streamOn1);
cudaMemcpyPeerAsync(dstOn1, 1, srcOn0, 0, bytes, streamOn1);
cudaStreamSynchronize(streamOn1);
cudaMemcpy(hostOut, dstOn1, bytes, cudaMemcpyDeviceToHost);

cudaStreamDestroy(streamOn1);
cudaFree(dstOn1);
cudaSetDevice(0);
cudaFree(srcOn0);
cudaDeviceDisablePeerAccess(1);
cudaSetDevice(1);
cudaDeviceDisablePeerAccess(0);
cudaSetDevice(0);
```

## 必须保留的顺序

- peer access 必须在 P2P device buffer 分配前开启。
- 必须开启双向 peer access：`0->1` 和 `1->0`。
- 目的 Device 的内存、stream 和同步必须在目的 Device 上处理。
- 验证脚本必须设置 `CUDA_COMPAT_DEVICE_MALLOC_POLICY=p2p`。

如果遗漏上述任一条件，CANN 可能在 `cudaStreamSynchronize` 返回 `cudaErrorSystemDriverMismatch`；随后出现的数据 mismatch 是连带结果。
