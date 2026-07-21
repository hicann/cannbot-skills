### Example Input Module
```
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)

M = 1024
K = 1024
N = 1024

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []
```
### Example input dsl
```
import tile.language as tl
import math
import torch

@ascend_kernel
def matmul_kernel(A_ptr, B_ptr, C_ptr,
                  M, N, K,
                  block_M, block_N, k_tile, n_blocks):
    """
    Kernel computes one (block_M x block_N) tile of C per program/core.
    Assumes no tail handling (all tiles are full).
    """

    pid = tl.program_id(0)

    # tile coordinates for this program
    pid_m = pid // n_blocks
    pid_n = pid % n_blocks

    # global tile starts
    m_start = pid_m * block_M
    n_start = pid_n * block_N
    k_blocks = K // k_tile

    # -----------------------
    # Allocate L1 and L0 buffers (all declared here)
    # -----------------------
    A_L1 = tl.alloc_l1((block_M, k_tile), dtype=tl.float32)
    B_L1 = tl.alloc_l1((k_tile, block_N), dtype=tl.float32)
    C_L0 = tl.alloc_l0c((block_M, block_N), dtype=tl.float32)

    # Main K loop: load A_L1, B_L1, compute into C_L0
    for kb in range(k_blocks):

        k_start = kb * k_tile

        with tl.copyin():
            tl.load(A_ptr + (m_start * K + k_start), A_L1)
            tl.load(B_ptr + (k_start * N + n_start), B_L1)

        with tl.compute():
            tl.gemm_v0(A_L1, B_L1, C_L0, init=(kb == 0))

    with tl.copyout():
        tl.store(C_ptr + (m_start * N + n_start), C_L0)


# -----------------------
# Host: global planning + launch
# -----------------------
def matmul_host(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
    M, K = A.shape
    K, N = B.shape

    # Core Partitioning
    # No inter-core sync required: each core writes distinct C tile
    block_M = 256
    block_N = 256

    m_blocks = M // block_M
    n_blocks = N // block_N

    n_cores = m_blocks * n_blocks

    # Tiling Strategy (Host)
    # Choose block sizes to fit your hardware L1 with safety headroom.
    k_tile  = 128

    matmul_kernel[n_cores](
        A, B, C,
        M, N, K,
        block_M, block_N, k_tile, n_blocks
    )

```
### Example Output TileLang
import tilelang
import tilelang.language as T
import torch.nn as nn
import torch

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, dtype="float32", accum_dtype="float"):
    block_M = 256
    block_N = 256
    m_num = M // block_M
    n_num = N // block_N
    K_L1 = 128

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)

            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D tensors"
        assert a.shape[1] == b.shape[0], "Matrix dimension mismatch for matmul"
        M, K = a.shape
        K2, N = b.shape
        assert K == K2

        func = matmul(M, N, K)
        return func(a, b)
