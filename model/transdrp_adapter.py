"""
TransDRP adapter — wraps AdversarialNetwork into the BaseModelAdapter interface.

TransDRP architecture:
  Encoder  : FeatMLP  (gene_expr → 64-dim)
  Classifier: GraphMLP (GAT, 9-drug outputs)
  Discriminator: domain-adversarial head

Usage:
  adapter = TransDRPAdapter(adversarial_network, device="cuda")
  probs = adapter.predict(gene_expr, {"node_x": ..., "edge_index": ...})
"""

import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

# Add TransDRP source directory to path so that model imports resolve.
# Adjust this path if your TransDRP-main is in a different location.
_TRANSDRP_CANDIDATES = [
    Path("G:/TransDRP_test/TransDRP-main"),
    Path("G:/TransDRP-main"),
]
for _p in _TRANSDRP_CANDIDATES:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
        break

from core.adapter import BaseModelAdapter


class TransDRPAdapter(BaseModelAdapter):
    """
    Adapter for the TransDRP (AdversarialNetwork) model.

    Parameters
    ----------
    model : nn.Module
        A trained AdversarialNetwork instance.
    device : str
        Computation device.
    n_drug_outputs : int
        Number of drug outputs (default 9).
    """

    def __init__(self, model: nn.Module, device: str = "cpu", n_drug_outputs: int = 9):
        super().__init__(model, device)
        self._n_drugs = n_drug_outputs

    @property
    def n_drugs(self) -> int:
        return self._n_drugs

    def predict(self, gene_expr: torch.Tensor, drug_data: Dict) -> torch.Tensor:
        """No-grad forward pass. Returns sigmoid probabilities [batch, n_drugs]."""
        gene_expr = gene_expr.to(self.device)
        node_x    = drug_data["node_x"].to(self.device)
        edge_idx  = drug_data["edge_index"].to(self.device)
        with torch.no_grad():
            _, out, _ = self.model(gene_expr, alpha=0.0, node_x=node_x, edge_index=edge_idx)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return torch.sigmoid(out)

    def predict_with_grad(self, gene_expr: torch.Tensor, drug_data: Dict) -> torch.Tensor:
        """With-grad forward pass (used during CF optimization)."""
        gene_expr = gene_expr.to(self.device)
        node_x    = drug_data["node_x"].to(self.device)
        edge_idx  = drug_data["edge_index"].to(self.device)
        _, out, _ = self.model(gene_expr, alpha=0.0, node_x=node_x, edge_index=edge_idx)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return torch.sigmoid(out)
