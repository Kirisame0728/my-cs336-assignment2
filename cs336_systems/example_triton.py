import triton
import triton.language as tl
import torch
from triton.language import dtype

# import os
# os.environ["TRITON_INTERPRET"] = "1"

autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
]



@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_a_M, stride_a_K,
    stride_b_K, stride_b_N,
    stride_c_M, stride_c_N,
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # group size refers to # clos of tiles for a group
    GROUP_SIZE: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.

    # Program ID
    PID = tl.program_id(0)
    # Number of program ids along the M axis
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M)
    # Number of programs ids along the N axis
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N)
    # Number of programs in group
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    # Id of the group this program is in
    group_id = PID // num_PID_in_group
    # Row-id of the first program in the group
    first_PID_in_group_along_M = group_id * GROUP_SIZE
    # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
    grout_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE)
    # *Within groups*, programs are ordered in a column-major order
    # Row-id of the program in the *launch grid*
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % grout_size_adj)
    # Col-id of the program in the *launch grid*
    PID_N = (PID % num_PID_in_group) // grout_size_adj

    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_M)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)

    a_offsets = offsets_M[:, None] * stride_a_M + offsets_K[None, :] * stride_a_K
    b_offsets = offsets_K[:, None] * stride_b_K + offsets_N[None, :] * stride_b_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = offsets_K < k * BLOCK_SIZE_K
        a = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=mask[:, None], other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)

        a_offsets += BLOCK_SIZE_K * stride_a_K
        b_offsets += BLOCK_SIZE_K * stride_b_K
    accumulator = accumulator.to(tl.float16)
    c_offsets = offsets_M[:, None] * stride_c_M + offsets_N[None, :] * stride_c_N
    c_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N)
    tl.store(c_ptr + c_offsets, accumulator.to(tl.float16), mask=c_mask)




def matmul(a, b):
    (M, K), (_, N) = a.shape, b.shape
    c = torch.empty((M, N), device=a.device)
    grid = lambda meta: triton.cdiv(M ,meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N'])
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
