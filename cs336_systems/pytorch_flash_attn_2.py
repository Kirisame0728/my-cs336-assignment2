from math import ceil
import torch


class FlashAttentionTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B_q = 16
        B_k = 16
        B, N_q, d = Q.shape
        N_k = K.shape[1]
        device = Q.device
        T_q = ceil(N_q / B_q)
        T_k = ceil(N_k / B_k)
        Q_tiled = Q.reshape(B, T_q, B_q, d)
        K_tiled = K.reshape(B, T_k, B_k, d)
        V_tiled = V.reshape(B, T_k, B_k, d)
        scaling = d ** 0.5
        O = torch.empty((B, N_q, d), device=device)
        L = torch.empty((B, N_q), device=device)
        for b in range(B):
            for i in range(T_q):
                Q_i = Q_tiled[b, i]
                O_i = torch.zeros(B_q, d, device=device)
                l_i = torch.zeros((B_q,), device=device)
                m_i = torch.full((B_q,), float("-inf"), device=device)
                for j in range(T_k):
                    K_j, V_j = K_tiled[b, j], V_tiled[b, j]
                    tiled_S = (Q_i @ K_j.transpose(-2, -1)) / scaling
                    m_ij = torch.max(tiled_S, dim=-1).values
                    m_i_new = torch.maximum(m_ij, m_i) # m_i is referred to as m_i,j-1
                    P_ij = torch.exp(tiled_S - m_i_new[:, None])
                    l_i_new = torch.sum(P_ij, dim=-1) + l_i * torch.exp(m_i - m_i_new)
                    O_i = torch.exp(m_i - m_i_new)[:, None] * O_i + P_ij @ V_j
                    m_i = m_i_new
                    l_i = l_i_new
                O[b, i * B_q:(i + 1) * B_q, :] = (1 / l_i[:, None]) * O_i
                L[b, i*B_q:(i+1)*B_q] = m_i + torch.log(l_i)
        O = O.to(Q.dtype)
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O


    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError