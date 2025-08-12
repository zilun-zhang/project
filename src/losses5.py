# losses5.py
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from config5 import (
    MAX_LONGEST_PATH_TIME, NORM_T,
    W_BCE, W_TIME, W_TOTALT, W_LONGEST, W_DAG, W_DEG_COV, W_SRC_SINK_SOFT,
    SRC_SINK_TAU, SRC_SINK_K, W_TIME_NODE
)
from utils5 import longest_path_time_from_mats
from config5 import W_NODE_TIME_UNI

class LossPack5(nn.Module):
    """边BCE + （可选）时间项 + 结构项（度覆盖、软单源/单汇、最长路、DAG光滑）"""
    def __init__(self):
        super().__init__()
        self.w_bce = W_BCE
        self.w_time = W_TIME
        self.w_totalT = W_TOTALT
        self.w_longest = W_LONGEST
        self.w_dag = W_DAG
        self.w_deg_cov = W_DEG_COV
        self.w_src_sink_soft = W_SRC_SINK_SOFT
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.w_time_node = W_TIME_NODE

    @staticmethod
    def node_out_time_uniformity(time_mat: torch.Tensor, A_mask: torch.Tensor) -> torch.Tensor:
        mask = A_mask.float()
        if mask.sum() < 1:
            return torch.tensor(0.0, device=time_mat.device)
        row_sum = (time_mat * mask).sum(dim=1)
        row_cnt = mask.sum(dim=1).clamp_min(1.0)
        row_mean = (row_sum / row_cnt).unsqueeze(1)
        dev = (time_mat - row_mean).abs() * mask
        return dev.sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _row_mean_with_mask(M: torch.Tensor, mask: torch.Tensor):
        m = mask.float()
        row_sum = (M * m).sum(dim=1)
        row_cnt = m.sum(dim=1)
        row_mean = row_sum / row_cnt.clamp_min(1.0)
        return row_mean, row_cnt

    def node_time_loss(self, time_mat: torch.Tensor, T_target: torch.Tensor, A_target: torch.Tensor):
        mask = (A_target > 0.5)
        r_pred, c_pred = self._row_mean_with_mask(time_mat, mask)
        r_tgt, c_tgt = self._row_mean_with_mask(T_target, mask)
        valid = (c_tgt > 0)
        if valid.any():
            return F.l1_loss(r_pred[valid], r_tgt[valid])
        return torch.tensor(0.0, device=time_mat.device)


    @staticmethod
    def _prob_and_mask(A_logits: torch.Tensor):
        valid = torch.isfinite(A_logits)
        prob = torch.zeros_like(A_logits)
        if valid.any():
            prob[valid] = torch.sigmoid(A_logits[valid])
        return prob, valid

    @staticmethod
    def dag_penalty(sigA: torch.Tensor) -> torch.Tensor:
        A = sigA * sigA
        expm = torch.matrix_exp(A)
        return torch.trace(expm) - expm.shape[0]

    @staticmethod
    def degree_coverage_penalty(sigA: torch.Tensor, widths: List[int]) -> torch.Tensor:
        """非首层入度≥1，非末层出度≥1 的软约束（可回传）"""
        # 构造每个节点的layer索引
        idx = []
        for li, w in enumerate(widths):
            idx += [li] * w
        idx = torch.tensor(idx, device=sigA.device)
        first_mask = (idx == idx.min())
        last_mask  = (idx == idx.max())

        in_deg  = sigA.sum(dim=0)  # soft in-degree
        out_deg = sigA.sum(dim=1)  # soft out-degree

        need_in  = (~first_mask)
        need_out = (~last_mask)

        # 想让 >=1：用 ReLU(1 - deg) 作为不足惩罚
        pen_in  = torch.relu(1.0 - in_deg[need_in]).mean()  if need_in.any()  else torch.tensor(0.0, device=sigA.device)
        pen_out = torch.relu(1.0 - out_deg[need_out]).mean() if need_out.any() else torch.tensor(0.0, device=sigA.device)
        return pen_in + pen_out

    @staticmethod
    def source_sink_penalty_soft(sigA: torch.Tensor) -> torch.Tensor:
        """软版本的单源/单汇：用平滑指示函数统计“近零入/出度”的节点个数，拉向1"""
        in_deg  = sigA.sum(dim=0)
        out_deg = sigA.sum(dim=1)
        # 近似 1[in_deg==0] ≈ σ(k*(τ - in_deg))
        src_soft  = torch.sigmoid(SRC_SINK_K * (SRC_SINK_TAU - in_deg))
        sink_soft = torch.sigmoid(SRC_SINK_K * (SRC_SINK_TAU - out_deg))
        num_src_soft  = src_soft.sum()
        num_sink_soft = sink_soft.sum()
        return (num_src_soft - 1.0) ** 2 + (num_sink_soft - 1.0) ** 2

    @staticmethod
    def total_time_penalty(time_mat: torch.Tensor, target_total_T: float) -> torch.Tensor:
        total_pred = time_mat.sum()
        return (total_pred - target_total_T) ** 2 / (NORM_T ** 2)

    @staticmethod
    def longest_path_penalty(A_logits: torch.Tensor, time_mat: torch.Tensor, widths: List[int]) -> torch.Tensor:
        with torch.no_grad():
            prob, valid = LossPack5._prob_and_mask(A_logits)
            A_bin = (prob > 0.5).float().cpu().numpy()
            T = time_mat.detach().cpu().numpy()
            lpt = longest_path_time_from_mats(A_bin, T, widths)
        if lpt <= MAX_LONGEST_PATH_TIME:
            return torch.tensor(0.0, device=A_logits.device)
        return torch.tensor((lpt - MAX_LONGEST_PATH_TIME) ** 2 / (MAX_LONGEST_PATH_TIME ** 2),
                            device=A_logits.device)




    def forward(self, A_logits, A_target, time_mat, T_target, target_total_T, widths, valid_mask=None):
        prob, valid = self._prob_and_mask(A_logits)

        # BCE（合法位置），空掩码保护
        if valid.any():
            bce = self.bce(A_logits[valid], A_target[valid])
        else:
            bce = torch.tensor(0.0, device=A_logits.device)

        # 时间项（你目前权重=0）
        pos = (A_target > 0.5) & valid
        time_loss = F.l1_loss(time_mat[pos], T_target[pos]) if pos.any() else torch.tensor(0.0, device=A_logits.device)

        time_node = self.node_time_loss(time_mat, T_target, A_target)
        # 总时长（预测 vs 目标）
        totalT_pred = time_mat[pos].sum() if pos.any() else torch.tensor(0.0, device=A_logits.device)
        #totalT = F.l1_loss(totalT_pred, target_total_T)

        # 最长路（若你已有实现，可调用；这里给个占位 0）
        longest = torch.tensor(0.0, device=A_logits.device)

        # NEW: 节点出边一致性
        node_uni = self.node_out_time_uniformity(time_mat, pos)

        # 结构项
        deg_cov  = self.degree_coverage_penalty(prob, widths)
        srcsink  = self.source_sink_penalty_soft(prob)
        totalT   = self.total_time_penalty(time_mat, target_total_T)
        longest  = self.longest_path_penalty(A_logits, time_mat, widths)
        dag_h    = torch.tensor(0.0, device=A_logits.device)  # 如需显示可改为 self.dag_penalty(prob)

        loss = (self.w_bce * bce
                + self.w_time * time_loss
                + self.w_totalT * totalT
                + self.w_deg_cov * deg_cov
                + self.w_src_sink_soft * srcsink
                + self.w_longest * longest
                + self.w_dag * dag_h
                + W_NODE_TIME_UNI * node_uni
                + self.w_time_node * time_node  # ★ 新增：节点时间损失
                )

        terms = {
            "bce": float(bce.detach().item()),
            "time": float(time_loss.detach().item()),
            "totalT": float(totalT.detach().item()),
            "deg_cov": float(deg_cov.detach().item()),
            "src_sink": float(srcsink.detach().item()),
            "longest": float(longest.detach().item()),
            "dag_h": float(dag_h.detach().item()),
            "node_uni": float(node_uni.detach().item()),
            "loss": float(loss.detach().item()),
            "time_node": float(time_node.detach().item()),  # ★ 节点时间 L1
        }
        return loss, terms
