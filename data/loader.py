"""
Data loading utilities for DRP-ConCF.

Handles loading of:
  - CCLE gene expression data (source domain, training)
  - TCGA gene expression data (target domain, test/evaluation)
  - Drug graph features and topology (shared between domains)

All data is sourced from the TransDRP project directory.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Locate TransDRP source (adjust path if needed)
_TRANSDRP_CANDIDATES = [
    Path("G:/TransDRP_test/TransDRP-main"),
    Path("G:/TransDRP-main"),
]
_TRANSDRP_ROOT: Optional[Path] = None
for _p in _TRANSDRP_CANDIDATES:
    if _p.exists():
        _TRANSDRP_ROOT = _p
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break


def load_transdrp_data(
    drug_names: Optional[List[str]] = None,
    device: str = "cpu",
    model_variant: str = "pretrain_num_epochs_100_uda_num_epochs_300",
) -> Tuple[torch.Tensor, Dict, List[str], object]:
    """
    Load TCGA test data, drug graph, and TransDRP model.

    Parameters
    ----------
    drug_names : list of str, optional
        Drug subset to load. Defaults to the standard 9-drug panel.
    device : str
        Computation device.
    model_variant : str
        Subdirectory name under TransDRP model_save/.

    Returns
    -------
    gene_data : torch.Tensor
        TCGA gene expression [N, n_genes].
    drug_data : dict
        {"node_x": [n_drugs, feat_dim], "edge_index": [2, n_edges]}.
    gene_names : list of str
        Gene symbols (length = n_genes).
    model : nn.Module
        Loaded AdversarialNetwork (already eval() mode).
    """
    if _TRANSDRP_ROOT is None:
        raise RuntimeError(
            "TransDRP source not found. "
            "Set _TRANSDRP_ROOT in data/loader.py to your TransDRP-main path."
        )

    if drug_names is None:
        drug_names = [
            "5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel",
            "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide",
        ]

    import pandas as pd
    import config
    from dataload import get_ccle_multi_labeled_dataloader, get_tcga_multi_labeled_dataloaders
    from utility import edge_extract
    from models import AdversarialNetwork, FeatMLP, GraphMLP

    # ── Gene features ────────────────────────────────────────────────────
    gex_df = pd.read_csv(config.gex_feature_file, index_col=0)
    gene_names = list(gex_df.columns.drop("Tissue"))
    input_dim = len(gene_names)

    # ── Drug graph ───────────────────────────────────────────────────────
    # Side effect: populates config.drug_feat and config.label_graph
    _ = list(get_ccle_multi_labeled_dataloader(
        gex_features_df=gex_df, batch_size=1, drug=drug_names,
        seed=2024, measurement="AUC", threshold_gdsc=0.5, threshold_label=0.01,
    ))
    drug_node_x    = torch.tensor(config.drug_feat, dtype=torch.float32).to(device)
    drug_edge_index = torch.tensor(edge_extract(config.label_graph), dtype=torch.long).to(device)
    drug_data = {"node_x": drug_node_x, "edge_index": drug_edge_index}

    # ── TCGA test data ───────────────────────────────────────────────────
    tcga_loader = get_tcga_multi_labeled_dataloaders(
        gex_features_df=gex_df, drug=drug_names, batch_size=64,
    )
    parts = [batch[0] for batch in tcga_loader]
    gene_data = torch.cat(parts, dim=0).to(device)

    # ── Model ────────────────────────────────────────────────────────────
    encoder    = FeatMLP(input_dim=input_dim, output_dim=64,
                         hidden_dims=[512, 256, 64], drop=0.2).to(device)
    classifier = GraphMLP(
        input_dim=64 + drug_node_x.size(1), output_dim=1,
        hidden_dims=[64, 32, 16], drug_num=len(drug_names), drop=0.2,
    ).to(device)
    model = AdversarialNetwork(encoder, classifier, fix_source=False).to(device)

    ckpt = _TRANSDRP_ROOT / "model_save" / model_variant / "AdversarialNetwork.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    return gene_data, drug_data, gene_names, model
