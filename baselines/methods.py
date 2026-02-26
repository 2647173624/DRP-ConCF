from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.adapter import BaseModelAdapter


class BaseExplainer:
    name: str = "base"

    def explain(self, gene_expr: torch.Tensor, drug_data: Dict, target_drug_idx: int) -> Dict:
        raise NotImplementedError


# ── 1. SHAP (KernelSHAP with gene grouping) ────────────────────────────────

class SHAPExplainer(BaseExplainer):
    """
    KernelSHAP — Shapley values via weighted coalitions.
    Genes are partitioned into n_groups clusters; SHAP computed on groups,
    then distributed to individual genes.
    """
    name = "SHAP"

    def __init__(self, adapter: BaseModelAdapter, n_samples: int = 256, n_groups: int = 50):
        self.adapter = adapter
        self.n_samples = n_samples
        self.n_groups = n_groups

    def explain(self, gene_expr: torch.Tensor, drug_data: Dict, target_drug_idx: int) -> Dict:
        if gene_expr.dim() == 1: gene_expr = gene_expr.unsqueeze(0)
        gene_expr = gene_expr.detach()
        device, n_genes = gene_expr.device, gene_expr.shape[-1]
        n_groups = min(self.n_groups, n_genes)
        group_ids = np.array_split(np.arange(n_genes), n_groups)

        Z = np.random.binomial(1, 0.5, (self.n_samples, n_groups)).astype(np.float32)
        Z = np.vstack([np.zeros(n_groups), np.ones(n_groups), Z]).astype(np.float32)
        full = np.zeros((len(Z), n_genes), dtype=np.float32)
        for gi, gidx in enumerate(group_ids):
            full[:, gidx] = Z[:, gi:gi+1]

        preds = []
        with torch.no_grad():
            for i in range(0, len(full), 16):
                bm = torch.from_numpy(full[i:i+16]).to(device)
                p = self.adapter.predict(gene_expr * bm, drug_data)
                if p.dim() == 1: p = p.unsqueeze(0)
                preds.append(p[:, target_drug_idx].cpu())
        y = torch.cat(preds).numpy()

        from scipy.special import comb
        sz = Z.sum(1)
        w = np.ones(len(Z))
        for i in range(len(Z)):
            s = int(sz[i])
            if 0 < s < n_groups:
                w[i] = (n_groups - 1) / (comb(n_groups, s) * s * (n_groups - s) + 1e-20)
            else:
                w[i] = 1e6

        W = np.diag(np.sqrt(w))
        ZW = W @ np.column_stack([Z, np.ones(len(Z))])
        coef, _, _, _ = np.linalg.lstsq(W @ y, ZW, rcond=None)
        # correct direction
        coef, _, _, _ = np.linalg.lstsq(ZW, W @ y, rcond=None)
        group_shap = coef[:n_groups]

        gene_shap = np.zeros(n_genes, np.float32)
        for gi, gidx in enumerate(group_ids):
            gene_shap[gidx] = group_shap[gi] / len(gidx)
        gene_shap = np.abs(gene_shap)
        if gene_shap.max() > 0: gene_shap /= gene_shap.max()

        return {"gene_mask": torch.from_numpy(gene_shap).float(), "method": self.name}


# ── 2. LIME ────────────────────────────────────────────────────────────────

class LIMEExplainer(BaseExplainer):
    """Local linear surrogate with cosine-kernel proximity weights."""
    name = "LIME"

    def __init__(self, adapter: BaseModelAdapter, n_samples: int = 200):
        self.adapter = adapter
        self.n_samples = n_samples

    def explain(self, gene_expr: torch.Tensor, drug_data: Dict, target_drug_idx: int) -> Dict:
        if gene_expr.dim() == 1: gene_expr = gene_expr.unsqueeze(0)
        gene_expr = gene_expr.detach()
        device, n_genes = gene_expr.device, gene_expr.shape[-1]

        masks = torch.bernoulli(torch.ones(self.n_samples, n_genes) * 0.5)
        preds = []
        with torch.no_grad():
            for i in range(0, self.n_samples, 16):
                bm = masks[i:i+16].to(device)
                p = self.adapter.predict(gene_expr * bm, drug_data)
                if p.dim() == 1: p = p.unsqueeze(0)
                preds.append(p[:, target_drug_idx].cpu())
        y = torch.cat(preds).numpy()

        dist = (masks - 1).abs().sum(1) / n_genes
        prox = torch.exp(-dist**2 / (2 * 0.25**2)).numpy()

        W = np.diag(np.sqrt(prox))
        Xw = W @ np.column_stack([masks.numpy(), np.ones(self.n_samples)])
        coef, _, _, _ = np.linalg.lstsq(Xw, W @ y, rcond=None)
        imp = np.abs(coef[:n_genes])
        if imp.max() > 0: imp /= imp.max()
        return {"gene_mask": torch.from_numpy(imp).float(), "method": self.name}


# ── 3. CADS ────────────────────────────────────────────────────────────────

class CADSExplainer(BaseExplainer):
    """
    CADS dual-mask: learns M_c (causal) and M_t=1-M_c (trivial).
    Causal branch preserves prediction; trivial branch diverges.
    """
    name = "CADS"

    def __init__(self, adapter: BaseModelAdapter, n_steps: int = 150, lr: float = 0.01,
                 phi: float = 0.5, lam_entropy: float = 0.01):
        self.adapter = adapter
        self.n_steps = n_steps
        self.lr = lr
        self.phi = phi
        self.lam_entropy = lam_entropy

    def explain(self, gene_expr: torch.Tensor, drug_data: Dict, target_drug_idx: int) -> Dict:
        if gene_expr.dim() == 1: gene_expr = gene_expr.unsqueeze(0)
        gene_expr = gene_expr.detach()
        device, n_genes = gene_expr.device, gene_expr.shape[-1]

        with torch.no_grad():
            op = self.adapter.predict(gene_expr, drug_data)
            if op.dim() == 1: op = op.unsqueeze(0)
            y_orig = op[0, target_drug_idx].item()

        mask_net = nn.Sequential(nn.Linear(n_genes, 256), nn.ReLU(), nn.Linear(256, n_genes)).to(device)
        opt = torch.optim.Adam(mask_net.parameters(), lr=self.lr)
        pfn = getattr(self.adapter, "predict_with_grad", self.adapter.predict)

        for _ in range(self.n_steps):
            opt.zero_grad()
            alpha_c = torch.sigmoid(mask_net(gene_expr))
            alpha_t = 1 - alpha_c
            pc = pfn(gene_expr * alpha_c, drug_data)
            if pc.dim() == 1: pc = pc.unsqueeze(0)
            pt = pfn(gene_expr * alpha_t, drug_data)
            if pt.dim() == 1: pt = pt.unsqueeze(0)
            yt = torch.tensor(y_orig, device=device)
            L_c = F.mse_loss(pc[0, target_drug_idx], yt)
            L_t = -F.mse_loss(pt[0, target_drug_idx], yt)
            H = -(alpha_c * torch.log(alpha_c + 1e-8) + alpha_t * torch.log(alpha_t + 1e-8)).mean()
            (L_c + self.phi * L_t + self.lam_entropy * H).backward()
            opt.step()

        with torch.no_grad():
            m = torch.sigmoid(mask_net(gene_expr)).squeeze().cpu()
        if m.max() > m.min(): m = (m - m.min()) / (m.max() - m.min())
        return {"gene_mask": m, "method": self.name}


# ── 4. DRExplainer ─────────────────────────────────────────────────────────

class DRExplainerExplainer(BaseExplainer):
    """
    DRExplainer-style: GNNExplainer edge-mask objective applied to gene features.
    Loss = MI(pred, orig) + λ_sparse*|M|_1 + λ_entropy*H(M) + size_constraint.
    """
    name = "DRExplainer"

    def __init__(self, adapter: BaseModelAdapter, n_steps: int = 150, lr: float = 0.05,
                 lam_sparse: float = 0.005, lam_entropy: float = 0.01, target_coverage: float = 0.10):
        self.adapter = adapter
        self.n_steps = n_steps
        self.lr = lr
        self.lam_sparse = lam_sparse
        self.lam_entropy = lam_entropy
        self.target_coverage = target_coverage

    def explain(self, gene_expr: torch.Tensor, drug_data: Dict, target_drug_idx: int) -> Dict:
        if gene_expr.dim() == 1: gene_expr = gene_expr.unsqueeze(0)
        gene_expr = gene_expr.detach()
        device, n_genes = gene_expr.device, gene_expr.shape[-1]

        with torch.no_grad():
            op = self.adapter.predict(gene_expr, drug_data)
            if op.dim() == 1: op = op.unsqueeze(0)
            y_orig = op[0, target_drug_idx]

        logits = (torch.randn(n_genes, device=device) * 0.01).requires_grad_(True)
        opt = torch.optim.Adam([logits], lr=self.lr)
        pfn = getattr(self.adapter, "predict_with_grad", self.adapter.predict)

        for _ in range(self.n_steps):
            opt.zero_grad()
            m = torch.sigmoid(logits)
            p = pfn(gene_expr * m.unsqueeze(0), drug_data)
            if p.dim() == 1: p = p.unsqueeze(0)
            L_pred = F.mse_loss(p[0, target_drug_idx], y_orig)
            L_sparse = m.sum() / n_genes
            H = -(m * torch.log(m + 1e-8) + (1-m) * torch.log(1-m + 1e-8)).mean()
            L_size = (m.mean() - self.target_coverage) ** 2
            (L_pred + self.lam_sparse * L_sparse + self.lam_entropy * H + 0.5 * L_size).backward()
            opt.step()

        with torch.no_grad():
            fm = torch.sigmoid(logits).cpu()
        if fm.max() > fm.min(): fm = (fm - fm.min()) / (fm.max() - fm.min())
        return {"gene_mask": fm, "method": self.name}


# ── 5. IDDGCN ──────────────────────────────────────────────────────────────

class IDDGCNExplainer(BaseExplainer):
    """
    IDDGCN: ExplaiNE gradient attribution + biological structure loss.
    Groups genes by expression quartile; enforces target proportions T=[0.4,0.4,0.1,0.1].
    """
    name = "IDDGCN"

    def __init__(self, adapter: BaseModelAdapter, n_steps: int = 150, lr: float = 0.05,
                 target_proportions: Optional[List[float]] = None,
                 lam_bio: float = 0.10, lam_sparse: float = 0.005):
        self.adapter = adapter
        self.n_steps = n_steps
        self.lr = lr
        self.target_props = target_proportions or [0.4, 0.4, 0.1, 0.1]
        self.lam_bio = lam_bio
        self.lam_sparse = lam_sparse

    def explain(self, gene_expr: torch.Tensor, drug_data: Dict, target_drug_idx: int) -> Dict:
        if gene_expr.dim() == 1: gene_expr = gene_expr.unsqueeze(0)
        device, n_genes = gene_expr.device, gene_expr.shape[-1]

        # Phase 1: ExplaiNE gradient
        x = gene_expr.clone().detach().requires_grad_(True)
        pfn = getattr(self.adapter, "predict_with_grad", self.adapter.predict)
        p = pfn(x, drug_data)
        if p.dim() == 1: p = p.unsqueeze(0)
        p[0, target_drug_idx].backward()
        grad = x.grad[0].abs().detach()

        # Phase 2: group by expression quartile
        expr = gene_expr.squeeze().detach()
        q = torch.quantile(expr.float().cpu(), torch.tensor([0.25, 0.5, 0.75])).to(device)
        gid = torch.zeros(n_genes, dtype=torch.long, device=device)
        gid[expr > q[2]] = 0
        gid[(expr > q[1]) & (expr <= q[2])] = 1
        gid[(expr > q[0]) & (expr <= q[1])] = 2
        gid[expr <= q[0]] = 3

        with torch.no_grad():
            op = self.adapter.predict(gene_expr, drug_data)
            if op.dim() == 1: op = op.unsqueeze(0)
            y_orig = op[0, target_drug_idx]

        init = (grad / (grad.max() + 1e-8) * 2 - 1).clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([init], lr=self.lr)
        target_t = torch.tensor(self.target_props, device=device, dtype=torch.float32)

        for _ in range(self.n_steps):
            opt.zero_grad()
            m = torch.sigmoid(init)
            pm = pfn(gene_expr.detach() * m.unsqueeze(0), drug_data)
            if pm.dim() == 1: pm = pm.unsqueeze(0)
            P_loss = F.mse_loss(pm[0, target_drug_idx], y_orig)
            ms = m.sum() + 1e-8
            C = torch.zeros(4, device=device)
            for g in range(4):
                gm = m[gid == g]
                C[g] = gm.sum() / ms if gm.numel() > 0 else 0.0
            B_loss = ((target_t - C) ** 2).mean()
            (P_loss + self.lam_bio * B_loss + self.lam_sparse * m.sum() / n_genes).backward()
            opt.step()

        with torch.no_grad():
            om = torch.sigmoid(init).cpu()
            gn = grad.cpu() / (grad.cpu().max() + 1e-8)
            combined = 0.4 * gn + 0.6 * om
            if combined.max() > combined.min():
                combined = (combined - combined.min()) / (combined.max() - combined.min())
        return {"gene_mask": combined, "method": self.name}


# ── Factory ────────────────────────────────────────────────────────────────

EXPLAINER_REGISTRY = {
    "shap": SHAPExplainer,
    "lime": LIMEExplainer,
    "cads": CADSExplainer,
    "drexplainer": DRExplainerExplainer,
    "iddgcn": IDDGCNExplainer,
}


def create_explainer(method: str, adapter: BaseModelAdapter, **kwargs) -> BaseExplainer:
    cls = EXPLAINER_REGISTRY.get(method.lower())
    if cls is None:
        raise ValueError(f"Unknown method '{method}'. Available: {list(EXPLAINER_REGISTRY)}")
    return cls(adapter, **kwargs)
