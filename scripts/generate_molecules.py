"""Generate novel KRAS inhibitor candidates and apply logD filtering.

Usage
-----
    python scripts/generate_molecules.py \\
        --config config/generator.yaml \\
        --seed_smiles data/sample/valid.txt \\
        --output results/generated.csv \\
        --n_per_seed 20

Pipeline
--------
1. Load fine-tuned SMILES VAE.
2. For each seed molecule, sample *n_per_seed* SMILES strings.
3. Validate SMILES with RDKit; deduplicate.
4. Apply post-generation logD filter (RTlogD, optimal window 1–3).
5. Score all accepted molecules (affinity, BBBp, QED, logD).
6. Compute KIRS and CDS evaluation metrics.
7. Write results to CSV.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd
import torch
import yaml
from rdkit import Chem

from galaxyair.affinity.scorer import AffinityScorer
from galaxyair.bbb.scorer import BBBScorer
from galaxyair.generator.autoencoder import SmilesAutoencoder
from galaxyair.generator.dataset import ValidationSmilesDataset
from galaxyair.logd.scorer import LogDScorer
from galaxyair.metrics.evaluation import evaluate_generation
from galaxyair.utils.molecular import compute_qed, compute_tanimoto_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate KRAS inhibitor candidates with logD filtering."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/generator.yaml",
        help="Path to generator.yaml",
    )
    parser.add_argument(
        "--logd_config",
        type=str,
        default="config/logd_predictor.yaml",
        help="Path to logd_predictor.yaml",
    )
    parser.add_argument(
        "--seed_smiles",
        type=str,
        default=None,
        help="Path to seed SMILES file (one per line). "
             "Defaults to paths.finetune_data in the config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/generated.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--n_per_seed",
        type=int,
        default=20,
        help="Number of molecules to generate per seed (paper uses 20 for KIRS_20).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    logger = logging.getLogger(__name__)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.logd_config) as f:
        logd_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---- Load model ----
    weights_dir = Path(cfg["paths"]["weights_dir"])
    model = SmilesAutoencoder.from_config(cfg["paths"]["config_save"], device=device)
    model.load_weights(weights_dir / "finetuned_weights.pt")
    logger.info("Fine-tuned model loaded.")

    # ---- Seed molecules ----
    seed_path = args.seed_smiles or cfg["paths"]["finetune_data"]
    seed_dataset = ValidationSmilesDataset(
        seed_path, cfg["paths"]["vocab"], device
    )
    seed_smiles: List[str] = [seed_dataset[i]["smiles_s"] for i in range(len(seed_dataset))]
    # Strip SOS/EOS tokens to get raw SMILES
    seed_smiles = [
        s.lstrip(seed_dataset.vocab.SOS_CHAR).rstrip(seed_dataset.vocab.EOS_CHAR)
        for s in seed_smiles
    ]
    logger.info("Loaded %d seed molecules.", len(seed_smiles))

    # ---- Scorers ----
    affinity_scorer = AffinityScorer(
        weights_path=cfg["paths"].get("surrogate_affinity_weights"),
        device=device,
    )
    bbb_scorer = BBBScorer(
        weights_path=cfg["paths"]["bbb_weights"],
        device=device,
    )
    logd_scorer = LogDScorer(
        rtlogd_repo_path=logd_cfg["paths"]["rtlogd_repo"],
        weights_path=logd_cfg["paths"]["weights"],
        logd_min=logd_cfg["filter"]["logd_min"],
        logd_max=logd_cfg["filter"]["logd_max"],
        device=device,
    )

    # ---- Generate ----
    all_records = []
    generated_per_seed: List[List[str]] = []

    for seed in seed_smiles:
        generated = model.generate(
            seed_smiles=seed,
            n_samples=args.n_per_seed,
            max_len=cfg["finetuning"]["max_seq_len"],
            temperature=args.temperature,
        )

        # Validate & deduplicate
        valid_gen: List[str] = []
        seen = set()
        for smi in generated:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            canon = Chem.MolToSmiles(mol)
            if canon not in seen:
                seen.add(canon)
                valid_gen.append(canon)

        generated_per_seed.append(valid_gen)

        # Apply logD filter
        accepted, rejected = logd_scorer.filter_smiles(valid_gen)
        logger.debug(
            "Seed %s: %d valid, %d accepted by logD filter, %d rejected.",
            seed[:20], len(valid_gen), len(accepted), len(rejected),
        )

        for smi in valid_gen:
            aff = float(affinity_scorer(smi))
            bbb = float(bbb_scorer(smi))
            qed = float(compute_qed(smi))
            logd_val = float(logd_scorer.predict(smi))
            sim = float(compute_tanimoto_similarity(seed, smi))
            all_records.append({
                "seed_smiles": seed,
                "generated_smiles": smi,
                "tanimoto_to_seed": sim,
                "affinity_score": aff,
                "bbb_score": bbb,
                "qed_score": qed,
                "logd": logd_val,
                "logd_accepted": smi in accepted,
                "composite_score": 0.4 * aff + 0.4 * bbb + 0.2 * qed,
            })

    # ---- Metrics ----
    logger.info("Computing KIRS and CDS metrics …")
    metrics = evaluate_generation(
        seed_smiles=seed_smiles,
        generated_smiles=generated_per_seed,
        affinity_fn=lambda s: float(affinity_scorer(s)),
        bbb_fn=lambda s: float(bbb_scorer(s)),
        top_k=args.n_per_seed,
    )
    logger.info(
        "KIRS_%d = %.4f  (success=%d / total=%d)",
        args.n_per_seed,
        metrics["kirs"],
        metrics["n_success"],
        metrics["n_total"],
    )
    logger.info(
        "CDS = %.4f  (D_intra=%.4f, D_inter=%.4f)",
        metrics["cds"],
        metrics["d_intra"],
        metrics["d_inter"],
    )

    # ---- Save results ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_records)
    df.to_csv(output_path, index=False)
    logger.info("Results saved to %s  (%d rows)", output_path, len(df))

    # Append metric summary
    metrics_path = output_path.with_suffix(".metrics.yaml")
    import yaml as _yaml
    with open(metrics_path, "w") as f:
        _yaml.dump(
            {
                f"kirs_{args.n_per_seed}": metrics["kirs"],
                "n_success": metrics["n_success"],
                "n_total": metrics["n_total"],
                "cds": metrics["cds"],
                "d_intra": metrics["d_intra"],
                "d_inter": metrics["d_inter"],
            },
            f,
        )
    logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
