"""
DRP-ConCF  — Drug Response Prediction via Causal Counterfactual Explanation
===========================================================================
Entry point: two-phase execution
  Phase 1  Causal graph discovery  (intervention experiments → C_gd matrix)
  Phase 2  Counterfactual explanation  (CF mask optimization → gene ranking)

Usage
-----
  python main.py [--samples N] [--drug DRUG_NAME] [--steps N] [--device cpu|cuda]

Example
-------
  python main.py --drug Cisplatin --samples 20
"""

import sys
import os
import time
import argparse
from pathlib import Path

import numpy as np
import torch

# ── Path setup ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

# ── Imports ─────────────────────────────────────────────────────────────────
from data.loader import load_transdrp_data
from model.transdrp_adapter import TransDRPAdapter
from core.causal_discovery import CausalDiscovery
from core.cf_optimizer import CausalCFOptimizer
from core.evaluation import (
    fidelity_plus, fidelity_minus, sparsity, stability,
    biomarker_overlap_rate, DRUG_NAMES_DEFAULT,
)

# ── Drug list ───────────────────────────────────────────────────────────────
DRUG_NAMES = DRUG_NAMES_DEFAULT   # 9 standard drugs

# ── Helpers ─────────────────────────────────────────────────────────────────

def _sep(char="─", width=70):
    print(char * width)

def _hdr(title: str):
    _sep("═")
    print(f"  {title}")
    _sep("═")

def _step(tag: str, msg: str):
    print(f"  [{tag}] {msg}")

def _ok(msg: str):
    print(f"        >> {msg}")

def parse_args():
    p = argparse.ArgumentParser(description="DRP-ConCF — causal CF explanation for DRP models")
    p.add_argument("--samples",   type=int,   default=20,         help="Samples for causal discovery")
    p.add_argument("--drug",      type=str,   default="Cisplatin", help="Target drug name")
    p.add_argument("--steps",     type=int,   default=300,         help="CF optimization steps")
    p.add_argument("--lr",        type=float, default=0.03,        help="CF learning rate")
    p.add_argument("--top_k",     type=int,   default=10,          help="Top-K genes to display")
    p.add_argument("--device",    type=str,   default="auto",      help="cpu | cuda | auto")
    p.add_argument("--cache_dir", type=str,   default=None,        help="Cache directory for causal graph")
    p.add_argument("--save_dir",  type=str,   default=None,        help="Directory to save causal graph")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Causal Graph Discovery
# ═══════════════════════════════════════════════════════════════════════════

def phase1_causal_discovery(adapter, gene_data, drug_data, args):
    _hdr("Phase 1 / Causal Graph Discovery")

    cd = CausalDiscovery(adapter, device=args.device)

    # Try to load from cache first
    if args.cache_dir and Path(args.cache_dir).exists():
        cache_gd = Path(args.cache_dir) / "gene_drug_causal.pt"
        if cache_gd.exists():
            _step("2/4", "Loading causal graph from cache ...")
            cd.load(args.cache_dir)
            _ok(f"Loaded  C_gd  shape: {list(cd.gene_drug_causal.shape)}")
            _ok(f"Cache path: {args.cache_dir}")
            return cd

    _step("1/4", f"Samples: {args.samples}  |  Intervention: knockout")
    _step("2/4", "Running knockout interventions for each gene ...")
    print()

    t0 = time.time()
    C_gd = cd.learn_gene_drug_causal_graph(
        gene_data=gene_data,
        drug_data=drug_data,
        n_samples=args.samples,
        intervention_type="knockout",
        verbose=True,
    )
    elapsed = time.time() - t0

    print()
    _step("3/4", "Applying drug-contrastive normalization ...")
    _ok("S_id = ReLU( Δ_id − mean_{d'≠d}(Δ_id') )  →  column-normalized")

    _step("4/4", "Causal graph ready.")
    _ok(f"C_gd  shape : {list(C_gd.shape)}  [n_genes × n_drugs]")
    _ok(f"Non-zero (>0.1): {(C_gd > 0.1).sum().item()} / {C_gd.numel()}")
    _ok(f"Time elapsed : {elapsed:.1f}s")

    if args.save_dir:
        cd.save(args.save_dir)
        _ok(f"Saved to: {args.save_dir}")

    return cd


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Counterfactual Explanation
# ═══════════════════════════════════════════════════════════════════════════

def phase2_cf_explanation(adapter, cd, gene_data, drug_data, gene_names, args):
    _hdr("Phase 2 / Counterfactual Explanation")

    if args.drug not in DRUG_NAMES:
        print(f"  [!] Unknown drug '{args.drug}'. Available: {DRUG_NAMES}")
        sys.exit(1)
    drug_idx = DRUG_NAMES.index(args.drug)

    # Select a sensitive sample (original prediction > 0.5 for this drug)
    _step("1/3", f"Selecting sample for drug: {args.drug}  (index {drug_idx})")
    sample_idx = None
    with torch.no_grad():
        preds = adapter.predict(gene_data[:100], drug_data)   # check first 100
    for i in range(min(100, gene_data.shape[0])):
        if preds[i, drug_idx].item() > 0.5:
            sample_idx = i
            break
    if sample_idx is None:
        sample_idx = 0
        print("        >> No sensitive sample found in first 100; using index 0")
    else:
        _ok(f"Sample index: {sample_idx}  |  Predicted prob: {preds[sample_idx, drug_idx]:.4f}")

    gene_expr = gene_data[sample_idx:sample_idx+1]

    # Build optimizer
    _step("2/3", f"Initializing CF optimizer  (init: weighted causal prior)")
    opt = CausalCFOptimizer(
        adapter=adapter,
        gene_drug_causal=cd.gene_drug_causal,
        device=args.device,
        lambda_flip=1.0,
        lambda_gene=0.01,
        lambda_drug=0.05,
        lambda_coherence=0.05,
        lambda_causal=0.3,
        causal_corr_weight=0.7,
        causal_reg_weight=0.3,
        top_k_genes=100,
        init_strategy="weighted",
    )

    print()
    _ok(f"Steps: {args.steps}  |  LR: {args.lr}  |  Target: resistant (prob → 0)")
    print()

    t0 = time.time()
    result = opt.generate_counterfactual(
        gene_expr=gene_expr,
        drug_data=drug_data,
        target_drug_idx=drug_idx,
        target_flip="resistant",
        n_steps=args.steps,
        lr=args.lr,
        verbose=True,
        log_interval=max(1, args.steps // 3),
    )
    elapsed = time.time() - t0
    print()

    # Results
    _step("3/3", "Explanation results")
    orig_p = result["original_pred"][0, drug_idx].item()
    cf_p   = result["cf_pred"][0, drug_idx].item()
    flip   = result["flip_success"]

    _ok(f"Drug          : {args.drug}")
    _ok(f"Original prob : {orig_p:.4f}  ({'sensitive' if orig_p > 0.5 else 'resistant'})")
    _ok(f"CF prob       : {cf_p:.4f}  ({'sensitive' if cf_p > 0.5 else 'resistant'})")
    _ok(f"Flip success  : {'YES' if flip else 'NO'}")
    _ok(f"Modified genes: {result['n_modified_genes']}  (threshold > 0.3)")
    _ok(f"Time elapsed  : {elapsed:.1f}s")

    # Top causal genes
    print()
    _sep("─")
    print(f"  Top-{args.top_k} most important genes for {args.drug}:")
    _sep("─")
    mask = result["gene_mask"]
    top_idx = torch.argsort(mask, descending=True)[:args.top_k]
    causal_scores = cd.gene_drug_causal[:, drug_idx]

    header = f"  {'Rank':>4}  {'Gene':<14}  {'Mask':>6}  {'Causal':>7}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for rank, gi in enumerate(top_idx, 1):
        gi = gi.item()
        gname = gene_names[gi] if gi < len(gene_names) else f"Gene{gi}"
        mval  = mask[gi].item()
        cval  = causal_scores[gi].item()
        print(f"  {rank:>4}  {gname:<14}  {mval:>6.4f}  {cval:>7.4f}")

    return result, mask


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════

def phase3_evaluation(adapter, cd, result, mask, gene_data, drug_data, gene_names, args):
    _hdr("Phase 3 / Evaluation Metrics")
    drug_idx = DRUG_NAMES.index(args.drug)
    sample_idx_range = min(30, gene_data.shape[0])

    _step("1/4", "Fidelity+  (keep top-10% genes, measure prediction consistency)")
    gene_expr = gene_data[0:1]
    fp = fidelity_plus(adapter.predict, gene_expr, drug_data, mask, drug_idx, top_k_ratio=0.1)
    _ok(f"Fidelity+  = {fp:.4f}  (higher → keeping selected genes preserves prediction)")

    _step("2/4", "Fidelity-  (remove top-10% genes, measure prediction change)")
    fm = fidelity_minus(adapter.predict, gene_expr, drug_data, mask, drug_idx, top_k_ratio=0.1)
    _ok(f"Fidelity-  = {fm:.4f}  (higher → selected genes are truly important)")

    _step("3/4", "Sparsity   (fraction of genes NOT selected)")
    sp = sparsity(mask, threshold=0.3)
    _ok(f"Sparsity   = {sp:.4f}  (higher → more concise explanation)")

    _step("4/4", "Biomarker Overlap Rate  (top-10% vs literature biomarkers)")
    bor = biomarker_overlap_rate(mask, gene_names, drug_idx, DRUG_NAMES, top_k_ratio=0.1)
    _ok(f"BOR        = {bor:.4f}  (fraction of top genes that are known {args.drug} biomarkers)")

    _sep("─")
    print("  Summary")
    _sep("─")
    print(f"  {'Metric':<22} {'Score':>8}")
    print(f"  {'─'*22} {'─'*8}")
    print(f"  {'Fidelity+':<22} {fp:>8.4f}")
    print(f"  {'Fidelity-':<22} {fm:>8.4f}")
    print(f"  {'Sparsity':<22} {sp:>8.4f}")
    print(f"  {'Biomarker Overlap (BOR)':<22} {bor:>8.4f}")
    _sep("─")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    _hdr("DRP-ConCF  — Causal Counterfactual Explanation for Drug Response")
    print(f"  Device: {args.device}  |  Drug: {args.drug}  |  Samples: {args.samples}")
    _sep()
    print()

    # ── Load data & model ────────────────────────────────────────────────
    _hdr("Setup / Loading Data and Model")
    _step("1/2", "Loading TransDRP model and TCGA test data ...")
    t0 = time.time()
    gene_data, drug_data, gene_names, model = load_transdrp_data(
        drug_names=DRUG_NAMES, device=args.device,
    )
    adapter = TransDRPAdapter(model, device=args.device, n_drug_outputs=len(DRUG_NAMES))
    elapsed = time.time() - t0
    _ok(f"Gene data : {gene_data.shape[0]} samples × {gene_data.shape[1]} genes")
    _ok(f"Drug graph: {drug_data['node_x'].shape[0]} drugs")
    _ok(f"Loaded in : {elapsed:.1f}s")

    _step("2/2", "Sanity check ...")
    with torch.no_grad():
        sp = adapter.predict(gene_data[:1], drug_data)
    _ok(f"Prediction shape: {list(sp.shape)}  |  Values: {[f'{v:.3f}' for v in sp[0].tolist()]}")
    print()

    # ── Phase 1 ──────────────────────────────────────────────────────────
    print()
    cd = phase1_causal_discovery(adapter, gene_data, drug_data, args)
    print()

    # ── Phase 2 ──────────────────────────────────────────────────────────
    print()
    result, mask = phase2_cf_explanation(adapter, cd, gene_data, drug_data, gene_names, args)
    print()

    # ── Phase 3 (optional metrics) ────────────────────────────────────────
    print()
    phase3_evaluation(adapter, cd, result, mask, gene_data, drug_data, gene_names, args)
    print()

    _sep("═")
    print("  Done.")
    _sep("═")
    print()


if __name__ == "__main__":
    main()
