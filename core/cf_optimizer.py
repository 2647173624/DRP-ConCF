from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Adam

from .adapter import BaseModelAdapter


def _fwd(adapter: BaseModelAdapter, gene: torch.Tensor, drug_data: Dict) -> torch.Tensor:
    fn = getattr(adapter, "predict_with_grad", None)
    return fn(gene, drug_data) if fn is not None else adapter.predict(gene, drug_data)


class CausalCFOptimizer:
    """
    Causal counterfactual optimizer (Phase 2 of DRP-ConCF).

    Jointly optimizes dual masks (m_g, m_d) in logit space:
        gene_cf = gene * (1 - m_g)
        drug_cf = drug * (1 - m_d)

    Loss (5 terms):
        L = λ_flip·L_flip + λ_gene·L_gene + λ_drug·L_drug
            + λ_coh·L_coherence + λ_causal·L_causal

    L_causal = α·(-Pearson(m_g, c_d)) + β·Σ c_i(m_g_i - c_i)^2
    """

    def __init__(
        self,
        adapter: BaseModelAdapter,
        gene_drug_causal: torch.Tensor,
        drug_drug_causal: Optional[torch.Tensor] = None,
        device: str = "cpu",
        lambda_flip: float = 1.0,
        lambda_gene: float = 0.01,
        lambda_drug: float = 0.05,
        lambda_coherence: float = 0.05,
        lambda_causal: float = 0.3,
        causal_corr_weight: float = 0.7,
        causal_reg_weight: float = 0.3,
        top_k_genes: int = 100,
        init_strategy: str = "weighted",
    ):
        self.adapter = adapter
        self.device = device
        self.gene_drug_causal = gene_drug_causal.to(device)
        self.drug_drug_causal = drug_drug_causal.to(device) if drug_drug_causal is not None else None

        self.lambda_flip = lambda_flip
        self.lambda_gene = lambda_gene
        self.lambda_drug = lambda_drug
        self.lambda_coherence = lambda_coherence
        self.lambda_causal = lambda_causal
        self.alpha = causal_corr_weight
        self.beta = causal_reg_weight
        self.top_k = top_k_genes
        self.init_strategy = init_strategy

        n_drugs = gene_drug_causal.shape[1]
        self._top_genes = {
            d: torch.argsort(gene_drug_causal[:, d], descending=True)[:top_k_genes]
            for d in range(n_drugs)
        }

    # ── Mask initialization ─────────────────────────────────────────────────

    def _init_gene_logit(self, n_genes: int, drug_idx: int) -> torch.Tensor:
        if self.init_strategy == "random":
            return (torch.randn(n_genes, device=self.device) * 0.1).requires_grad_(True)
        scores = self.gene_drug_causal[:, drug_idx]
        if self.init_strategy == "top_causal":
            init = torch.full((n_genes,), -2.2, device=self.device)
            init[self._top_genes[drug_idx]] = 0.0
        else:  # weighted
            lo, hi = scores.min(), scores.max()
            p = 0.1 + 0.8 * (scores - lo) / (hi - lo) if hi > lo else torch.full_like(scores, 0.5)
            p = p.clamp(0.01, 0.99)
            init = torch.log(p / (1 - p))
        return init.detach().clone().requires_grad_(True)

    # ── Main optimization ───────────────────────────────────────────────────

    def generate_counterfactual(
        self,
        gene_expr: torch.Tensor,
        drug_data: Dict,
        target_drug_idx: int,
        target_flip: str = "resistant",
        n_steps: int = 300,
        lr: float = 0.03,
        mask_threshold: float = 0.3,
        verbose: bool = True,
        log_interval: int = 100,
    ) -> Dict:
        if gene_expr.dim() == 1:
            gene_expr = gene_expr.unsqueeze(0)
        gene_expr = gene_expr.to(self.device)
        n_genes = gene_expr.shape[1]

        with torch.no_grad():
            orig = self.adapter.predict(gene_expr, drug_data)
            if orig.dim() == 1:
                orig = orig.unsqueeze(0)
        orig_label = (orig[0, target_drug_idx] > 0.5).item()

        gene_logit = self._init_gene_logit(n_genes, target_drug_idx)
        opt = Adam([gene_logit], lr=lr)

        best_mask, best_pred, best_loss = None, orig.clone(), float("inf")

        for step in range(1, n_steps + 1):
            opt.zero_grad()
            m_g = torch.sigmoid(gene_logit)
            cf_gene = gene_expr * (1 - m_g)
            cf_pred = _fwd(self.adapter, cf_gene, drug_data)
            if cf_pred.dim() == 1:
                cf_pred = cf_pred.unsqueeze(0)

            loss, ld = self._loss(m_g, cf_pred, target_drug_idx, target_flip)
            loss.backward()
            opt.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_mask = m_g.detach().clone()
                best_pred = cf_pred.detach().clone()

            if verbose and step % log_interval == 0:
                flip_now = (best_pred[0, target_drug_idx] > 0.5).item() != orig_label
                print(
                    f"  Step {step:>3}/{n_steps} | "
                    f"L_flip={ld['flip'].item():.4f}  "
                    f"L_causal={ld['causal'].item():.4f}  "
                    f"flip={'YES' if flip_now else 'no'}"
                )

        final = best_mask.squeeze()
        cf_label = (best_pred[0, target_drug_idx] > 0.5).item()

        return {
            "gene_mask": final.cpu(),
            "cf_gene_expr": (gene_expr * (1 - best_mask)).cpu(),
            "original_pred": orig.cpu(),
            "cf_pred": best_pred.cpu(),
            "flip_success": orig_label != cf_label,
            "n_modified_genes": int((final > mask_threshold).sum()),
            "top_modified_genes": torch.argsort(final, descending=True)[:20].cpu().numpy(),
            "target_drug_idx": target_drug_idx,
            "best_loss": best_loss,
        }

    # ── Loss ────────────────────────────────────────────────────────────────

    def _loss(
        self,
        m_g: torch.Tensor,
        cf_pred: torch.Tensor,
        drug_idx: int,
        target_flip: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        d: Dict[str, torch.Tensor] = {}
        mask = m_g.squeeze()
        p = cf_pred[0, drug_idx].clamp(1e-7, 1 - 1e-7)

        # L_flip
        d["flip"] = (-torch.log(1 - p) if target_flip == "resistant" else -torch.log(p))

        # L_gene  (sparsity)
        d["gene"] = mask.abs().sum()

        # L_drug  (placeholder — set to 0 when no drug mask)
        d["drug"] = torch.zeros(1, device=mask.device).squeeze()

        # L_coherence — cosine(m_g, inter-drug variance)
        var_d = self.gene_drug_causal.var(dim=1)
        d["coh"] = -(mask @ var_d) / (mask.norm() * var_d.norm() + 1e-8)

        # L_causal — Scheme1 + Scheme3
        c = self.gene_drug_causal[:, drug_idx]
        if mask.std() > 1e-8 and c.std() > 1e-8:
            L_corr = -torch.corrcoef(torch.stack([mask, c]))[0, 1]
        else:
            L_corr = torch.tensor(0.0, device=mask.device)
        L_wreg = (c * (mask - c) ** 2).sum()
        d["causal"] = self.alpha * L_corr + self.beta * L_wreg

        total = (
            self.lambda_flip * d["flip"]
            + self.lambda_gene * d["gene"]
            + self.lambda_drug * d["drug"]
            + self.lambda_coherence * d["coh"]
            + self.lambda_causal * d["causal"]
        )
        return total, d
