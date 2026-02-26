from typing import Dict, Tuple

import torch
import torch.nn as nn


class CausalConsistencyLoss(nn.Module):
    """
    Causal consistency regularizer combining:
      Scheme 1 — L_corr  = -Pearson(m_g, c_d)
      Scheme 3 — L_wreg  = Σ c_i * (m_g_i - c_i)^2
      Total     = alpha * L_corr + beta * L_wreg
    """

    def __init__(
        self,
        gene_drug_causal: torch.Tensor,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        super().__init__()
        self.register_buffer("gene_drug_causal", gene_drug_causal.clone())
        self.alpha = alpha
        self.beta = beta

    def forward(
        self, gene_mask: torch.Tensor, target_drug_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mask = gene_mask.squeeze()
        c = self.gene_drug_causal[:, target_drug_idx].to(mask.device)

        L_corr = _neg_pearson(mask, c)
        L_wreg = (c * (mask - c) ** 2).sum()

        loss = self.alpha * L_corr + self.beta * L_wreg
        return loss, {"corr": L_corr.detach(), "wreg": L_wreg.detach()}


def _neg_pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.std() > 1e-8 and b.std() > 1e-8:
        return -torch.corrcoef(torch.stack([a, b]))[0, 1]
    return torch.tensor(0.0, device=a.device)
