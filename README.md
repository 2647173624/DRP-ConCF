# DRP-ConCF

**Drug Response Prediction via Causal Counterfactual Explanation**

A two-phase framework that learns drug-specific causal gene signatures from
intervention experiments, then generates counterfactual explanations by
optimizing a gene mask guided by the learned causal structure.

---

## Project Structure

```
DRP-ConCF/
├── main.py               one-click entry: Phase 1 + Phase 2
├── requirements.txt
├── configs/
│   └── default.yaml      hyperparameters
├── core/                 framework core
│   ├── adapter.py        BaseModelAdapter ABC
│   ├── causal_discovery.py   Phase 1: C_gd learning via interventions
│   ├── cf_optimizer.py       Phase 2: dual-mask CF optimization
│   ├── causal_loss.py        L_causal (Scheme1 + Scheme3)
│   ├── evaluation.py         Fidelity+/-, Sparsity, Stability, BOR, DSI
│   └── utils.py              drug_contrastive_normalize, load_config
├── model/
│   └── transdrp_adapter.py   TransDRP (AdversarialNetwork) adapter
├── baselines/
│   └── methods.py        SHAP, LIME, CADS, DRExplainer, IDDGCN
├── data/
│   └── loader.py         CCLE/TCGA data loading (requires TransDRP source)
└── assets/
    └── framework.png
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with defaults (Cisplatin, 20 samples, 300 CF steps)
python main.py

# Specify drug, sample count, and CF steps
python main.py --drug Doxorubicin --samples 30 --steps 500

# Use GPU
python main.py --device cuda --drug Paclitaxel
```

### Command-line options

| Option | Default | Description |
|---|---|---|
| `--drug` | `Cisplatin` | Target drug name |
| `--samples` | `20` | Samples for causal discovery |
| `--steps` | `300` | CF optimization steps |
| `--lr` | `0.03` | CF learning rate |
| `--top_k` | `10` | Top-K genes to display |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |
| `--cache_dir` | `None` | Load pre-computed causal graph |
| `--save_dir` | `None` | Save causal graph after Phase 1 |

---

## Two-Phase Pipeline

### Phase 1 — Causal Graph Discovery

For each gene $i$ and drug $d$, estimate the causal effect via knockout
intervention:

$$\Delta_{id} = \frac{1}{N}\sum_n \bigl|P(Y_d \mid \text{do}(X_i=0)) - P(Y_d \mid X)\bigr|$$

Apply drug-contrastive normalization to isolate drug-specific effects:

$$S_{id} = \text{ReLU}\!\left(\Delta_{id} - \frac{1}{D-1}\sum_{d'\neq d}\Delta_{id'}\right)$$

Output: $C_{gd} \in \mathbb{R}^{n_\text{genes} \times n_\text{drugs}}$

### Phase 2 — Counterfactual Optimization

Jointly optimize gene mask $m_g \in [0,1]^{n_\text{genes}}$ in logit space:

$$\mathcal{L} = \lambda_\text{flip}\mathcal{L}_\text{flip} + \lambda_g\mathcal{L}_\text{gene} + \lambda_d\mathcal{L}_\text{drug} + \lambda_\text{coh}\mathcal{L}_\text{coherence} + \lambda_\text{causal}\mathcal{L}_\text{causal}$$

where $\mathcal{L}_\text{causal} = \alpha \cdot (-\text{Pearson}(m_g, c_d)) + \beta \cdot \sum_i c_i(m_{g,i} - c_i)^2$

Mask initialization uses causal priors: $p_i = 0.1 + 0.8\cdot\frac{c_i - \min c}{\max c - \min c}$

---

## Supported Drugs

5-Fluorouracil, Cisplatin, Cyclophosphamide, Docetaxel, Doxorubicin,
Etoposide, Gemcitabine, Paclitaxel, Temozolomide

---

## Dependencies

Requires the [TransDRP](https://github.com/...) source tree for data loading
and model architecture. Set `_TRANSDRP_CANDIDATES` in `data/loader.py` and
`model/transdrp_adapter.py` to point to your local TransDRP-main directory.

---

## Evaluation Metrics

| Metric | Definition |
|---|---|
| Fidelity+ | Prediction consistency when keeping only top-k genes |
| Fidelity- | Prediction change when removing top-k genes |
| Sparsity | Fraction of genes NOT selected |
| BOR | Overlap of top-k genes with GDSC literature biomarkers |
| DSI | Drug-specificity index vs. other drugs' causal scores |
