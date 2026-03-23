from math import ceil
import torch
import math


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
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors

        B_q = 16
        B_k = 16

        B, N_q, d = Q.shape
        N_k = K.shape[1]
        device = Q.device

        T_q = ceil(N_q / B_q)
        T_k = ceil(N_k / B_k)

        D = torch.sum(O * dO, dim=-1)  # (B, N_q)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        for b in range(B):
            for j in range(T_k):
                k_start, k_end = j * B_k, (j + 1) * B_k
                K_j = K[b, k_start:k_end, :]  # (B_k, d)
                V_j = V[b, k_start:k_end, :]  # (B_k, d)

                dK_j = torch.zeros((B_k, d), device=device, dtype=Q.dtype)
                dV_j = torch.zeros((B_k, d), device=device, dtype=Q.dtype)

                for i in range(T_q):
                    q_start, q_end = i * B_q, (i + 1) * B_q
                    Q_i = Q[b, q_start:q_end, :]  # (B_q, d)
                    dO_i = dO[b, q_start:q_end, :]  # (B_q, d)
                    L_i = L[b, q_start:q_end]  # (B_q,)
                    D_i = D[b, q_start:q_end]  # (B_q,)

                    S_ij = (Q_i @ K_j.transpose(-2, -1)) / math.sqrt(d)  # (B_q, B_k)
                    P_ij = torch.exp(S_ij - L_i[:, None])  # (B_q, B_k)

                    dV_j += P_ij.transpose(-2, -1) @ dO_i  # (B_k, d)
                    dP_ij = dO_i @ V_j.transpose(-2, -1)  # (B_q, B_k)
                    dS_ij = P_ij * (dP_ij - D_i[:, None]) / math.sqrt(d)  # (B_q, B_k)

                    dQ[b, q_start:q_end, :] += dS_ij @ K_j  # (B_q, d)
                    dK_j += dS_ij.transpose(-2, -1) @ Q_i  # (B_k, d)

                dK[b, k_start:k_end, :] = dK_j
                dV[b, k_start:k_end, :] = dV_j

        return dQ, dK, dV, None