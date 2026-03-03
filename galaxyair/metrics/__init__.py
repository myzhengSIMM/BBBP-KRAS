"""Evaluation metrics for the GalaxyAir molecular generation pipeline.

Metrics (Section 2.6 of the paper)
------------------------------------
KIRS  – Knowledge-Integrated Reproduction Score
CDS   – Composite Diversity Score
"""

from galaxyair.metrics.evaluation import (
    compute_kirs,
    compute_cds,
    evaluate_generation,
)

__all__ = [
    "compute_kirs",
    "compute_cds",
    "evaluate_generation",
]
