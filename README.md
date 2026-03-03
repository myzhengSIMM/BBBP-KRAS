# GalaxyAir

Structure-constrained molecular generation for optimizing KRAS inhibitors with improved blood-brain barrier permeability.

> Journal of Pharmaceutical Analysis, 2025, 15(8): 101337
> DOI: [10.1016/j.jpha.2025.101337](https://doi.org/10.1016/j.jpha.2025.101337)

---

## Overview

GalaxyAir is a structure-constrained molecular generation pipeline that optimizes KRAS inhibitors for blood-brain barrier (BBB) permeability. It combines:

| Component | Method | Details |
|-----------|--------|---------|
| Molecular generator | SMILES VAE (bidirectional GRU encoder + GRU decoder) | Contrastive pretraining on ChEMBL KRAS triplets |
| RL fine-tuning | Policy gradient | Reward = 0.4 × affinity + 0.4 × BBBp + 0.2 × QED |
| BBB predictor | AttentiveFP GNN + active learning (margin sampling) | MCC = 0.8215, 918 labeled molecules, 42 rounds |
| KRAS affinity predictor | PBCNet (physics-informed graph attention) | Fine-tuned on 267 ChEMBL molecules, pChEMBL 3.79–10.02 |
| logD filter | RTlogD (Random Forest) | Post-generation filter, optimal window 1–3 |
| QED | `rdkit.Chem.QED.qed()` | Used in reward only |

---

## Repository Structure

```
GalaxyAir/
├── config/
│   ├── generator.yaml          # VAE hyperparameters
│   ├── bbb_predictor.yaml      # AttentiveFP + active learning config
│   └── logd_predictor.yaml     # RTlogD Random Forest config
├── data/
│   └── sample/
│       ├── train_triplets.txt  # (Lead, Opt, Control) SMILES triplets
│       ├── valid.txt           # Validation SMILES
│       ├── bbb_data.csv        # BBB labelled data (SMILES, BBB)
│       └── logd_data.csv       # logD labelled data (SMILES, logD)
├── galaxyair/
│   ├── affinity/
│   │   ├── pbcnet.py           # PBCNetPredictor (full_3d + surrogate modes)
│   │   └── scorer.py           # AffinityScorer callable wrapper
│   ├── bbb/
│   │   ├── model.py            # BBBClassifier (AttentiveFP)
│   │   ├── active_learning.py  # ActiveLearner, EntropySampling, MarginSampling
│   │   ├── scorer.py           # BBBScorer callable wrapper
│   │   └── train.py            # train_with_active_learning()
│   ├── generator/
│   │   ├── autoencoder.py      # SmilesAutoencoder (pretrain + finetune)
│   │   ├── dataset.py          # SmilesVocabulary, TripletSmilesDataset
│   │   ├── decoder.py          # SmilesDecoder (GRU)
│   │   ├── encoder.py          # SmilesEncoder (bidirectional GRU)
│   │   └── reward.py           # RewardFunction, ReplayBuffer
│   ├── logd/
│   │   ├── model.py            # RTLogDPredictor (Random Forest)
│   │   └── scorer.py           # LogDScorer + filter_smiles()
│   ├── metrics/
│   │   └── evaluation.py       # compute_kirs(), compute_cds(), evaluate_generation()
│   └── utils/
│       ├── molecular.py        # QED, Tanimoto, compute_kras_bbbp_score
│       └── sa_scorer.py        # SA score (Ertl & Schuffenhauer 2009)
├── scripts/
│   ├── train_bbb.py            # BBB active learning training pipeline
│   ├── pretrain_generator.py   # VAE contrastive pretraining
│   ├── finetune_generator.py   # RL fine-tuning with reward function
│   └── generate_molecules.py  # Generation + logD filter + metrics
├── environment.yml
└── README.md
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate galaxyair
```

### PBCNet (full 3D mode)

The full 3D inference mode requires the PBCNet repository and pretrained weights:

```bash
git clone https://github.com/myzhengSIMM/PBCNet path/to/pbcnet
# Download PBCNet.pth from the release page and set weights_path in config
```

The surrogate mode (ECFP-2048 + Ridge regression) is used by default during RL fine-tuning and requires only SMILES inputs.

---

## Usage

### 1. Train BBB Predictor

```bash
python scripts/train_bbb.py --config config/bbb_predictor.yaml
```

Runs AttentiveFP GNN training with margin sampling active learning.
Target: MCC ≥ 0.82 after ~42 query rounds.

### 2. Pretrain the SMILES VAE

```bash
python scripts/pretrain_generator.py --config config/generator.yaml
```

Contrastive pretraining on (Lead, Opt, Control) triplets.

### 3. Fine-tune with RL

```bash
python scripts/finetune_generator.py --config config/generator.yaml
```

Policy gradient fine-tuning.
Reward = **0.4 × affinity + 0.4 × BBBp + 0.2 × QED**
Tanimoto similarity threshold = 0.4

### 4. Generate Molecules

```bash
python scripts/generate_molecules.py \
    --config config/generator.yaml \
    --seed_smiles data/sample/valid.txt \
    --output results/generated.csv \
    --n_per_seed 20
```

Generates molecules, applies logD filter (optimal window 1–3), and reports KIRS and CDS metrics.

---

## Metrics

| Metric | Definition | Paper Result |
|--------|-----------|-------------|
| KIRS_20 | `N_success / N_total` (top-20 generated per seed, 34 pairs) | 61.76 % |
| CDS | `0.5 × D_intra + 0.5 × D_inter` | — |
| BBB MCC | Matthews correlation coefficient | 0.8215 |

A generated molecule is counted as a success if:
- Tanimoto similarity to seed ≥ 0.4
- Predicted affinity score ≥ 0.5
- Predicted BBB score ≥ 0.5

---

## Citation

```bibtex
@article{galaxyair2025,
  title   = {Optimizing blood-brain barrier permeability in KRAS inhibitors:
             A structure-constrained molecular generation approach},
  journal = {Journal of Pharmaceutical Analysis},
  year    = {2025},
  volume  = {15},
  number  = {8},
  pages   = {101337},
  doi     = {10.1016/j.jpha.2025.101337}
}
```

### Dependencies

- **PBCNet**: Yu J, Sheng X, et al. Physics-Informed Graph Attention for Binding Affinity Prediction. [myzhengSIMM/PBCNet](https://github.com/myzhengSIMM/PBCNet)
- **AttentiveFP**: Xiong Z, et al. Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism. *J Med Chem* 2020. DOI: 10.1021/acs.jmedchem.9b00959
- **RTlogD**: [WangYitian123/RTlogD](https://github.com/WangYitian123/RTlogD)
