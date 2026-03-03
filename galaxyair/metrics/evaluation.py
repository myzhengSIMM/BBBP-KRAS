"""
Evaluation metrics for the GalaxyAir molecular generation pipeline.

Metrics (Section 2.6 of paper)
--------------------------------
KIRS (Knowledge-Integrated Reproduction Score)
    N_success / N_total

    A generated molecule is counted as a *success* if it simultaneously
    satisfies ALL three criteria relative to its seed (input) molecule:
      1. Tanimoto similarity ≥ threshold_similarity (default 0.4)
      2. Predicted affinity score ≥ threshold_affinity  (default 0.5)
      3. Predicted BBB score     ≥ threshold_bbb        (default 0.5)

    KIRS_20 is reported in the paper at 61.76 % (34 seed→refined pairs,
    9 unique seeds, top-20 generated molecules evaluated per seed).

CDS (Composite Diversity Score)
    DS = 0.5 × D_intra + 0.5 × D_inter

    D_intra: mean pairwise Tanimoto *distance* (1 − similarity) within
             the generated set.
    D_inter: mean Tanimoto *distance* between each generated molecule and
             the nearest known training molecule.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _morgan_fps(smiles_list: List[str], radius: int = 2, n_bits: int = 2048):
    """Return a list of Morgan fingerprints (None for invalid SMILES)."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fps.append(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            if mol is not None
            else None
        )
    return fps


def _tanimoto_matrix(fps_a, fps_b) -> np.ndarray:
    """Compute a len(fps_a) × len(fps_b) Tanimoto similarity matrix.

    Entries for None fingerprints are set to 0.0.
    """
    n, m = len(fps_a), len(fps_b)
    mat = np.zeros((n, m), dtype=np.float32)
    for i, fp_a in enumerate(fps_a):
        if fp_a is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp_a, fps_b)
        for j, s in enumerate(sims):
            mat[i, j] = s if fps_b[j] is not None else 0.0
    return mat


# ---------------------------------------------------------------------------
# KIRS
# ---------------------------------------------------------------------------

def compute_kirs(
    seed_smiles: List[str],
    generated_smiles: List[List[str]],
    affinity_fn: Callable[[str], float],
    bbb_fn: Callable[[str], float],
    threshold_similarity: float = 0.4,
    threshold_affinity: float = 0.5,
    threshold_bbb: float = 0.5,
    top_k: Optional[int] = 20,
) -> Dict[str, float]:
    """Compute the Knowledge-Integrated Reproduction Score (KIRS).

    Parameters
    ----------
    seed_smiles:
        List of N seed (input) SMILES strings.
    generated_smiles:
        List of N lists, each containing the generated SMILES for the
        corresponding seed molecule.
    affinity_fn:
        Callable mapping a SMILES string to a normalized affinity score
        in [0, 1].
    bbb_fn:
        Callable mapping a SMILES string to a BBB probability in [0, 1].
    threshold_similarity:
        Minimum Tanimoto similarity to the seed (default 0.4).
    threshold_affinity:
        Minimum predicted affinity score (default 0.5).
    threshold_bbb:
        Minimum predicted BBB score (default 0.5).
    top_k:
        If not None, only the first *top_k* generated molecules per seed
        are evaluated (paper uses top_k=20 for KIRS_20).

    Returns
    -------
    dict with keys:
        "kirs"           – overall KIRS value in [0, 1]
        "n_success"      – total number of successful molecules
        "n_total"        – total number of evaluated molecules
        "per_seed_kirs"  – list of per-seed KIRS values
    """
    if len(seed_smiles) != len(generated_smiles):
        raise ValueError(
            "seed_smiles and generated_smiles must have the same length."
        )

    seed_fps = _morgan_fps(seed_smiles)
    n_success_total = 0
    n_total = 0
    per_seed_kirs: List[float] = []

    for seed_fp, gen_list in zip(seed_fps, generated_smiles):
        candidates = gen_list[:top_k] if top_k is not None else gen_list
        n_success = 0
        n_eval = 0
        for gen_smi in candidates:
            mol = Chem.MolFromSmiles(gen_smi)
            if mol is None:
                n_eval += 1
                continue

            # Similarity check
            if seed_fp is not None:
                gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                sim = DataStructs.TanimotoSimilarity(seed_fp, gen_fp)
            else:
                sim = 0.0

            if sim < threshold_similarity:
                n_eval += 1
                continue

            # Property checks
            aff = affinity_fn(gen_smi)
            bbb = bbb_fn(gen_smi)
            if aff >= threshold_affinity and bbb >= threshold_bbb:
                n_success += 1
            n_eval += 1

        per_seed_kirs.append(n_success / n_eval if n_eval > 0 else 0.0)
        n_success_total += n_success
        n_total += n_eval

    kirs = n_success_total / n_total if n_total > 0 else 0.0
    return {
        "kirs": kirs,
        "n_success": n_success_total,
        "n_total": n_total,
        "per_seed_kirs": per_seed_kirs,
    }


# ---------------------------------------------------------------------------
# CDS
# ---------------------------------------------------------------------------

def compute_cds(
    generated_smiles: List[str],
    training_smiles: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute the Composite Diversity Score (CDS).

    DS = 0.5 × D_intra + 0.5 × D_inter

    Parameters
    ----------
    generated_smiles:
        Molecules produced by the model.
    training_smiles:
        Known training molecules used to compute D_inter.  If None,
        D_inter is set to 0.0 and CDS reduces to D_intra.

    Returns
    -------
    dict with keys:
        "cds"     – composite diversity score in [0, 1]
        "d_intra" – mean intra-set Tanimoto distance
        "d_inter" – mean inter-set Tanimoto distance (0 if training_smiles is None)
    """
    valid_gen = [s for s in generated_smiles if Chem.MolFromSmiles(s) is not None]
    if len(valid_gen) < 2:
        return {"cds": 0.0, "d_intra": 0.0, "d_inter": 0.0}

    gen_fps = _morgan_fps(valid_gen)
    valid_fps = [fp for fp in gen_fps if fp is not None]

    # D_intra: mean pairwise distance within generated set
    sim_matrix = _tanimoto_matrix(valid_fps, valid_fps)
    n = len(valid_fps)
    upper_tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pairwise_sims = sim_matrix[upper_tri_mask]
    d_intra = float(np.mean(1.0 - pairwise_sims)) if len(pairwise_sims) > 0 else 0.0

    # D_inter: mean distance from generated to nearest training molecule
    d_inter = 0.0
    if training_smiles is not None and len(training_smiles) > 0:
        train_fps = [fp for fp in _morgan_fps(training_smiles) if fp is not None]
        if train_fps:
            cross_sim = _tanimoto_matrix(valid_fps, train_fps)
            # For each generated mol, take max similarity to training set,
            # then convert to distance
            nearest_sim = cross_sim.max(axis=1)
            d_inter = float(np.mean(1.0 - nearest_sim))

    cds = 0.5 * d_intra + 0.5 * d_inter
    return {"cds": cds, "d_intra": d_intra, "d_inter": d_inter}


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def evaluate_generation(
    seed_smiles: List[str],
    generated_smiles: List[List[str]],
    affinity_fn: Callable[[str], float],
    bbb_fn: Callable[[str], float],
    training_smiles: Optional[List[str]] = None,
    threshold_similarity: float = 0.4,
    threshold_affinity: float = 0.5,
    threshold_bbb: float = 0.5,
    top_k: Optional[int] = 20,
) -> Dict[str, float]:
    """Run KIRS and CDS evaluation and return a combined report.

    Parameters
    ----------
    seed_smiles, generated_smiles, affinity_fn, bbb_fn:
        Forwarded to compute_kirs.
    training_smiles:
        Forwarded to compute_cds for D_inter calculation.
    threshold_similarity, threshold_affinity, threshold_bbb, top_k:
        Forwarded to compute_kirs.

    Returns
    -------
    Merged dict with keys from both compute_kirs and compute_cds.
    """
    flat_generated = [smi for gen_list in generated_smiles for smi in gen_list]

    kirs_result = compute_kirs(
        seed_smiles=seed_smiles,
        generated_smiles=generated_smiles,
        affinity_fn=affinity_fn,
        bbb_fn=bbb_fn,
        threshold_similarity=threshold_similarity,
        threshold_affinity=threshold_affinity,
        threshold_bbb=threshold_bbb,
        top_k=top_k,
    )
    cds_result = compute_cds(flat_generated, training_smiles)
    return {**kirs_result, **cds_result}
