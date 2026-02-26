from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


# ── Config ─────────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None) -> Dict:
    with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f)
        if user:
            config = _deep_merge(config, user)
    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


# ── Tensor helpers ──────────────────────────────────────────────────────────

def normalize_columns(tensor: torch.Tensor) -> torch.Tensor:
    result = tensor.clone()
    for j in range(result.shape[1]):
        col_max = result[:, j].max()
        if col_max > 0:
            result[:, j] /= col_max
    return result


def drug_contrastive_normalize(raw_effects: torch.Tensor) -> torch.Tensor:
    """
    S_{id} = ReLU( Δ_{id} − mean_{d'≠d}(Δ_{id'}) )
    Removes shared non-specific effects, isolating drug-specific contributions.
    Output is column-normalized to [0, 1].
    """
    n_genes, n_drugs = raw_effects.shape
    specific = torch.zeros_like(raw_effects)
    for d in range(n_drugs):
        others = torch.cat(
            [raw_effects[:, :d], raw_effects[:, d + 1:]], dim=1
        ).mean(dim=1)
        specific[:, d] = torch.relu(raw_effects[:, d] - others)
    return normalize_columns(specific)
