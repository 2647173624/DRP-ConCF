from typing import Callable, Dict, List, Optional

import numpy as np
import torch


# Literature-curated drug→biomarker gene sets (GDSC/CCLE)
GDSC_BIOMARKERS: Dict[str, set] = {
    "5-Fluorouracil":   {"TYMS", "RRM2", "TYMP", "TK1", "FOLR1", "CDKN1A", "CCND1", "ERBB2"},
    "Cisplatin":        {"ABCC3", "CCND1", "CDKN2A", "CDKN1A"},
    "Cyclophosphamide": {"ALDH1A1", "GSTP1", "GSTM1", "NQO1", "CBR1", "CDKN1A", "SOD2"},
    "Docetaxel":        {"TUBB2A", "TUBB2B", "TUBA1A", "TUBA1B", "STMN1", "CYP1B1", "CDKN2A", "CDKN1A", "SRC", "EGFR", "ERBB2"},
    "Doxorubicin":      {"TOP2A", "CDKN1A", "CDKN2A", "NQO1", "CBR1", "AKR1C3", "PTGR1", "MYC", "CCND1", "HIF1A", "VEGFA", "CXCR4", "ERBB2"},
    "Etoposide":        {"TOP2A", "CDKN1A", "MYC", "CCND1", "CDKN2A"},
    "Gemcitabine":      {"RRM2", "TYMS", "FOLR1", "CDKN2A", "MUC1", "EGFR", "CCND1", "CDKN1A"},
    "Paclitaxel":       {"TUBB2A", "TUBA1A", "STMN1", "CDKN2A", "CCND1", "ERBB2", "EGFR"},
    "Temozolomide":     {"IDH1", "IDH2", "CDKN2A", "CDKN1A", "GSTP1", "EGFR"},
}

DRUG_NAMES_DEFAULT: List[str] = [
    "5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel",
    "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide",
]


# ── Fidelity+ ──────────────────────────────────────────────────────────────

def fidelity_plus(
    predict_fn: Callable,
    gene_expr: torch.Tensor,
    drug_data: Dict,
    importance_mask: torch.Tensor,
    target_drug_idx: int,
    top_k_ratio: float = 0.1,
) -> float:
    """Keep top-k important genes, zero others. Return 1 - |orig - masked|."""
    n_genes = gene_expr.shape[-1]
    k = max(1, int(n_genes * top_k_ratio))
    top_idx = torch.argsort(importance_mask, descending=True)[:k]

    with torch.no_grad():
        orig = predict_fn(gene_expr, drug_data)
        if orig.dim() == 1: orig = orig.unsqueeze(0)
        orig_p = orig[0, target_drug_idx].item()

        masked = torch.zeros_like(gene_expr)
        masked[0, top_idx] = gene_expr[0, top_idx]
        mp = predict_fn(masked, drug_data)
        if mp.dim() == 1: mp = mp.unsqueeze(0)
        masked_p = mp[0, target_drug_idx].item()

    return 1.0 - abs(orig_p - masked_p)


# ── Fidelity- ──────────────────────────────────────────────────────────────

def fidelity_minus(
    predict_fn: Callable,
    gene_expr: torch.Tensor,
    drug_data: Dict,
    importance_mask: torch.Tensor,
    target_drug_idx: int,
    top_k_ratio: float = 0.1,
) -> float:
    """Remove top-k important genes. Return |orig - removed|."""
    n_genes = gene_expr.shape[-1]
    k = max(1, int(n_genes * top_k_ratio))
    top_idx = torch.argsort(importance_mask, descending=True)[:k]

    with torch.no_grad():
        orig = predict_fn(gene_expr, drug_data)
        if orig.dim() == 1: orig = orig.unsqueeze(0)
        orig_p = orig[0, target_drug_idx].item()

        removed = gene_expr.clone()
        removed[0, top_idx] = 0.0
        rp = predict_fn(removed, drug_data)
        if rp.dim() == 1: rp = rp.unsqueeze(0)
        removed_p = rp[0, target_drug_idx].item()

    return abs(orig_p - removed_p)


# ── Sparsity ───────────────────────────────────────────────────────────────

def sparsity(importance_mask: torch.Tensor, threshold: float = 0.1) -> float:
    n = importance_mask.shape[0]
    return 1.0 - (importance_mask > threshold).sum().item() / n


# ── Stability ──────────────────────────────────────────────────────────────

def stability(
    explain_fn: Callable,
    gene_expr: torch.Tensor,
    drug_data: Dict,
    target_drug_idx: int,
    n_perturbations: int = 5,
    noise_scale: float = 0.01,
) -> float:
    """Return 1 - avg normalised distance between base and perturbed masks."""
    n_genes = gene_expr.shape[-1]
    orig_res = explain_fn(gene_expr, drug_data, target_drug_idx)
    orig_mask = _to_mask(orig_res, n_genes)
    orig_norm = orig_mask.norm() + 1e-8
    diffs = []
    for _ in range(n_perturbations):
        noise = torch.randn_like(gene_expr) * noise_scale * (gene_expr.std() + 1e-8)
        res = explain_fn(gene_expr + noise, drug_data, target_drug_idx)
        pm = _to_mask(res, n_genes)
        diffs.append(((orig_mask - pm).norm() / orig_norm).item())
    return max(0.0, 1.0 - float(np.mean(diffs)))


# ── Biomarker Overlap Rate (BOR) ───────────────────────────────────────────

def biomarker_overlap_rate(
    importance_mask: torch.Tensor,
    gene_names: List[str],
    target_drug_idx: int,
    drug_names: Optional[List[str]] = None,
    top_k_ratio: float = 0.1,
) -> float:
    if drug_names is None:
        drug_names = DRUG_NAMES_DEFAULT
    drug_name = drug_names[target_drug_idx]
    if drug_name not in GDSC_BIOMARKERS:
        return 0.0
    n_genes = importance_mask.shape[0]
    k = max(1, int(n_genes * top_k_ratio))
    top_genes = {gene_names[i] for i in torch.argsort(importance_mask, descending=True)[:k].tolist()}
    return len(top_genes & GDSC_BIOMARKERS[drug_name]) / k


# ── Drug Specificity Index (DSI) ───────────────────────────────────────────

def drug_specificity_index(
    importance_mask: torch.Tensor,
    gene_drug_causal: torch.Tensor,
    target_drug_idx: int,
    top_k_ratio: float = 0.1,
) -> float:
    n_genes = importance_mask.shape[0]
    n_drugs = gene_drug_causal.shape[1]
    k = max(1, int(n_genes * top_k_ratio))
    top_idx = torch.argsort(importance_mask, descending=True)[:k]
    c_t = gene_drug_causal[top_idx, target_drug_idx]
    if n_drugs > 1:
        omask = torch.ones(n_drugs, dtype=torch.bool)
        omask[target_drug_idx] = False
        c_o = gene_drug_causal[top_idx][:, omask].mean(1)
    else:
        c_o = torch.zeros_like(c_t)
    return float(((c_t - c_o) / (c_o.abs() + 1e-8)).mean().item())


# ── Helpers ─────────────────────────────────────────────────────────────────

def _to_mask(result: Dict, n_genes: int) -> torch.Tensor:
    mask = result.get("gene_mask", torch.zeros(n_genes))
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    return mask.float().squeeze()
