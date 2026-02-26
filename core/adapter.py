from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class BaseModelAdapter(ABC):
    """Abstract adapter interface that any DRP backbone must implement."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @abstractmethod
    def predict(self, gene_expr: torch.Tensor, drug_data: Dict) -> torch.Tensor:
        """Return per-drug response probabilities [batch, n_drugs]."""
        ...

    @abstractmethod
    def predict_with_grad(self, gene_expr: torch.Tensor, drug_data: Dict) -> torch.Tensor:
        """Same as predict but keeps the computational graph for backprop."""
        ...

    @property
    @abstractmethod
    def n_drugs(self) -> int:
        """Number of drug outputs."""
        ...

    def forward_with_mask(
        self,
        gene_expr: torch.Tensor,
        gene_mask: torch.Tensor,
        drug_data: Dict,
    ) -> torch.Tensor:
        masked = gene_expr * (1.0 - gene_mask.unsqueeze(0))
        return self.predict_with_grad(masked, drug_data)
