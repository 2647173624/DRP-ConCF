from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .adapter import BaseModelAdapter
from .utils import drug_contrastive_normalize, normalize_columns


# Known drug-gene biomarker associations for validation
KNOWN_BIOMARKERS: Dict[str, List[int]] = {
    "EGFR":  [0, 3, 7],
    "TP53":  [0, 1, 4, 8],
    "BRCA1": [1, 4],
    "BRCA2": [1, 4],
    "ERBB2": [3, 7],
    "MGMT":  [8],
    "TYMS":  [0],
    "TOP2A": [4, 5],
    "TUBB3": [3, 7],
    "RRM1":  [6],
}


class CausalDiscovery:
    """
    Learn gene→drug and drug→drug causal graphs via intervention experiments.

    Phase 1 of DRP-ConCF: for each gene i and drug d, estimate the causal
    effect Δ_{id} = E[|P(Y_d | do(X_i=v)) − P(Y_d | X)|], then apply
    drug-contrastive normalization to obtain C_gd ∈ R^{n_genes × n_drugs}.
    """

    def __init__(self, adapter: BaseModelAdapter, device: str = "cpu"):
        self.adapter = adapter
        self.device = device
        self.n_drugs = adapter.n_drugs
        self.gene_drug_causal: Optional[torch.Tensor] = None
        self.drug_drug_causal: Optional[torch.Tensor] = None
        self._by_intervention: Dict[str, torch.Tensor] = {}

    # ── Core: single intervention ───────────────────────────────────────────

    @torch.no_grad()
    def learn_gene_drug_causal_graph(
        self,
        gene_data: torch.Tensor,
        drug_data: Dict,
        n_samples: int = 100,
        intervention_type: str = "knockout",
        batch_size: int = 64,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Estimate C_gd using a single intervention type.
        Returns normalized [n_genes, n_drugs] tensor.
        """
        gene_data = gene_data.to(self.device)
        N, n_genes = gene_data.shape

        if N > n_samples:
            idx = torch.randperm(N)[:n_samples]
            gene_data, N = gene_data[idx], n_samples

        baseline = self._batch_predict(gene_data, drug_data, batch_size)
        gene_mean = gene_data.mean(0)
        gene_p99  = torch.quantile(gene_data, 0.99, dim=0)

        raw = torch.zeros(n_genes, self.n_drugs, device=self.device)
        it = tqdm(range(n_genes), desc=f"[Causal] {intervention_type}") if verbose else range(n_genes)

        for i in it:
            g = gene_data.clone()
            if   intervention_type == "knockout": g[:, i] = 0.0
            elif intervention_type == "mean":     g[:, i] = gene_mean[i]
            elif intervention_type == "shuffle":  g[:, i] = gene_data[torch.randperm(N), i]
            elif intervention_type == "extreme":  g[:, i] = gene_p99[i]
            else: raise ValueError(f"Unknown intervention: {intervention_type}")

            pred = self._batch_predict(g, drug_data, batch_size)
            raw[i] = (pred - baseline).abs().mean(0)

        C = drug_contrastive_normalize(raw).cpu()
        self._by_intervention[intervention_type] = C
        self.gene_drug_causal = C
        return C

    # ── Multi-intervention fusion ───────────────────────────────────────────

    @torch.no_grad()
    def learn_gene_drug_fused(
        self,
        gene_data: torch.Tensor,
        drug_data: Dict,
        n_samples: int = 100,
        interventions: Optional[List[str]] = None,
        fusion: str = "average",
        verbose: bool = True,
    ) -> torch.Tensor:
        if interventions is None:
            interventions = ["knockout", "mean", "shuffle", "extreme"]

        graphs = []
        for k, iv in enumerate(interventions):
            if verbose:
                print(f"\n  [{k+1}/{len(interventions)}] Running '{iv}' intervention ...")
            graphs.append(
                self.learn_gene_drug_causal_graph(
                    gene_data, drug_data, n_samples, iv, verbose=verbose
                )
            )

        stacked = torch.stack(graphs)
        if fusion == "average":
            C = stacked.mean(0)
        elif fusion == "max":
            C = stacked.max(0)[0]
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        C = normalize_columns(C)
        self.gene_drug_causal = C
        return C

    # ── Drug→Drug graph ─────────────────────────────────────────────────────

    @torch.no_grad()
    def learn_drug_drug_causal_graph(
        self,
        gene_data: torch.Tensor,
        drug_data: Dict,
        n_samples: int = 100,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> torch.Tensor:
        gene_data = gene_data.to(self.device)
        N = gene_data.shape[0]
        if N > n_samples:
            idx = torch.randperm(N)[:n_samples]
            gene_data, N = gene_data[idx], n_samples

        baseline = self._batch_predict(gene_data, drug_data, batch_size)
        C_dd = torch.zeros(self.n_drugs, self.n_drugs, device=self.device)

        it = tqdm(range(self.n_drugs), desc="[Causal] drug-drug") if verbose else range(self.n_drugs)
        for di in it:
            md = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in drug_data.items()}
            if "node_x" in md:
                md["node_x"] = md["node_x"].clone()
                md["node_x"][di] = 0.0
            pred = self._batch_predict(gene_data, md, batch_size)
            C_dd[di] = (pred - baseline).abs().mean(0)

        mx = C_dd.max()
        if mx > 0:
            C_dd /= mx
        self.drug_drug_causal = C_dd.cpu()
        return C_dd.cpu()

    # ── Top-gene query ──────────────────────────────────────────────────────

    def get_top_causal_genes(
        self,
        drug_idx: int,
        top_k: int = 20,
        gene_names: Optional[List[str]] = None,
    ) -> Dict:
        if self.gene_drug_causal is None:
            raise RuntimeError("Run learn_*() first.")
        scores = self.gene_drug_causal[:, drug_idx]
        top_idx = torch.argsort(scores, descending=True)[:top_k]
        result = {"indices": top_idx.numpy(), "scores": scores[top_idx].numpy()}
        if gene_names is not None:
            result["names"] = [gene_names[i] for i in top_idx]
        return result

    # ── Causal-guided mask initialization ──────────────────────────────────

    def get_causal_init_mask(
        self, drug_idx: int, n_genes: int, top_k: int = 100, strategy: str = "weighted"
    ) -> torch.Tensor:
        if strategy == "random":
            return torch.ones(n_genes) * 0.1
        if self.gene_drug_causal is None:
            raise RuntimeError("Run learn_*() first.")
        scores = self.gene_drug_causal[:, drug_idx]
        if strategy == "top_causal":
            mask = torch.full((n_genes,), 0.1)
            mask[torch.argsort(scores, descending=True)[:top_k]] = 0.5
        elif strategy == "weighted":
            lo, hi = scores.min(), scores.max()
            mask = 0.1 + 0.8 * (scores - lo) / (hi - lo) if hi > lo else torch.full((n_genes,), 0.5)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return mask

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, save_dir: str) -> None:
        import os; os.makedirs(save_dir, exist_ok=True)
        if self.gene_drug_causal is not None:
            torch.save(self.gene_drug_causal, f"{save_dir}/gene_drug_causal.pt")
        if self.drug_drug_causal is not None:
            torch.save(self.drug_drug_causal, f"{save_dir}/drug_drug_causal.pt")

    def load(self, save_dir: str) -> None:
        import os
        p = f"{save_dir}/gene_drug_causal.pt"
        if os.path.exists(p):
            self.gene_drug_causal = torch.load(p, map_location="cpu", weights_only=True)
        p = f"{save_dir}/drug_drug_causal.pt"
        if os.path.exists(p):
            self.drug_drug_causal = torch.load(p, map_location="cpu", weights_only=True)

    # ── Internal ────────────────────────────────────────────────────────────

    def _batch_predict(self, gene_data: torch.Tensor, drug_data: Dict, batch_size: int) -> torch.Tensor:
        parts = []
        N = gene_data.shape[0]
        for s in range(0, N, batch_size):
            parts.append(self.adapter.predict(gene_data[s:s+batch_size], drug_data))
        return torch.cat(parts, 0)
