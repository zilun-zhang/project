from __future__ import annotations
from typing import List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config5 import (DEVICE, EDGE_TIME_MIN)
from utils5 import distribute_nodes_across_layers, build_order_from_widths, mask_allowed_pairs


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, act=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), act(),
            nn.Linear(hidden, hidden), act(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)



class StructureToGraphDecoder5(nn.Module):

    def __init__(self, pair_hidden=64, time_hidden=64):
        super().__init__()
        # 关键：挂在 self 上，才会被注册为可训练参数
        # 输入特征是 10 维：[li, lj, di, pi_norm, pj_norm] + s(5)
        self.edge_mlp = nn.Sequential(
            nn.Linear(10, pair_hidden), nn.ReLU(),
            nn.Linear(pair_hidden, pair_hidden), nn.ReLU(),
            nn.Linear(pair_hidden, 1)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(10, time_hidden), nn.ReLU(),
            nn.Linear(time_hidden, time_hidden), nn.ReLU(),
            nn.Linear(time_hidden, 1)
        )

    @staticmethod
    def _unnorm_s(s: torch.Tensor):
        from config5 import NORM_N, NORM_E, NORM_L, NORM_W, NORM_T
        N = (s[0] * NORM_N).clamp(min=1.0)
        E = (s[1] * NORM_E).clamp(min=0.0)
        L = (s[2] * NORM_L).clamp(min=1.0)
        W = (s[3] * NORM_W).clamp(min=1.0)
        T = (s[4] * NORM_T).clamp(min=0.0)
        # Round where needed
        N_int = int(torch.round(N).item())
        L_int = int(torch.round(L).item())
        W_int = int(torch.round(W).item())
        return N_int, E.item(), L_int, W_int, T.item()

    def forward(self, s: torch.Tensor, widths: Optional[List[int]] = None):
        """s: shape [5] on DEVICE. widths: python list or None.
        Returns: logits[N,N], time_mat[N,N], widths(list[int])
        """
        assert s.dim() == 1 and s.shape[0] == 5
        N_int, _E, L_int, W_int, T_total = self._unnorm_s(s)
        if widths is None:
            widths = distribute_nodes_across_layers(N_int, L_int, W_int)
        # Sanity
        if sum(widths) != N_int:
            # fix if rounding error
            diff = N_int - sum(widths)
            if widths:
                widths[-1] += diff
            else:
                widths = [N_int]
        order = build_order_from_widths(widths)
        N = len(order)
        assert N == N_int

        # Build mask for allowed pairs
        allow = torch.from_numpy(mask_allowed_pairs(widths)).to(s.device)

        # Build pair features
        # For efficiency, create tensors of li, lj, di, pi_norm, pj_norm
        li = torch.tensor([li for (li, _pi) in order], device=s.device).view(N, 1).repeat(1, N)
        lj = torch.tensor([lj for (lj, _pj) in order], device=s.device).view(1, N).repeat(N, 1)
        di = lj - li  # >= 0 ideally; we'll mask later
        pi = torch.tensor([pi for (_li, pi) in order], device=s.device).view(N, 1).repeat(1, N)
        pj = torch.tensor([pj for (_lj, pj) in order], device=s.device).view(1, N).repeat(N, 1)
        # Normalize positions within (approx) W_int to [0,1]
        pi_n = pi / max(1, W_int - 1)
        pj_n = pj / max(1, W_int - 1)

        # Broadcast s to pair-shape
        s_b = s.view(1, 1, -1).repeat(N, N, 1)

        pair_feats = torch.stack([li, lj, di, pi_n, pj_n], dim=2)  # [N,N,5]
        pair_feats = torch.cat([pair_feats, s_b], dim=2)           # [N,N,10]

        # Push through MLPs
        logits = self.edge_mlp(pair_feats).squeeze(-1)             # [N,N]
        raw_time = self.time_mlp(pair_feats).squeeze(-1)           # [N,N]
        time_mat = F.softplus(raw_time) + EDGE_TIME_MIN            # ensure > 0

        # Mask out illegal pairs by setting logits to -inf and times to 0
        neg_inf = torch.finfo(logits.dtype).min
        logits = torch.where(allow > 0.5, logits, torch.full_like(logits, neg_inf))
        time_mat = torch.where(allow > 0.5, time_mat, torch.zeros_like(time_mat))

        return logits, time_mat, widths