import triton
import triton.language as tl
import torch

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    q_mask = q_offsets < N_QUERIES

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    q = tl.load(
        Q_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for start_k in range(0, N_KEYS, K_TILE_SIZE):
        k = tl.load(
            K_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        v = tl.load(
            V_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        k_offsets = start_k + tl.arange(0, K_TILE_SIZE)
        k_mask = k_offsets < N_KEYS
        s_ij = tl.dot(q, tl.trans(k)) * scale

        s_ij = tl.where(k_mask[None, :], s_ij, -float("inf"))
        s_ij = tl.where(q_mask[:, None], s_ij, -float("inf"))

        row_max = tl.max(s_ij, axis=1)
        m_ij = tl.maximum(m_i, row_max)

        p_tilde = tl.exp(s_ij - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)

        l_i = alpha * l_i + tl.sum(p_tilde, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p_tilde, v)

        m_i = m_ij

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    o = acc / l_i[:, None]
    L_val = m_i + tl.log(l_i)
    tl.store(
        O_block_ptr,
        o,
        boundary_check=(0, 1),
    )

    tl.store(
        L_ptr + batch_index * stride_lb + q_offsets * stride_lq,
        L_val,
        mask=q_mask,
    )

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        if is_causal:
            raise NotImplementedError("Current Triton flash_fwd_kernel does not implement causal masking")
        B_q = 16
        B_k = 16
        B, N_q, d = Q.shape
        N_k = K.shape[1]
        device = Q.device

        O = torch.empty((B, N_q, d), device=device, dtype=Q.dtype)
        L = torch.empty((B, N_q), device=device, dtype=torch.float32)

        scale = d ** -0.5

        grid = (triton.cdiv(N_q, B_q), B)

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k,
            scale,
            D=d,
            Q_TILE_SIZE=B_q,
            K_TILE_SIZE=B_k,
            num_warps=4,
            num_stages=2,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError