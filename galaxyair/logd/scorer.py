"""logD scoring interface used as a post-generation filter.

Per Section 2.3 of the paper:
  "Given the non-linear relationship between logD and optimal membrane
   permeability (typically between 1 and 3), logD is not included in the
   reward function but is used as a filtering condition."
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

from galaxyair.logd.model import RTLogDPredictor


class LogDScorer:
    """Thin wrapper around RTLogDPredictor for single-molecule scoring and filtering.

    Parameters
    ----------
    rtlogd_repo_path:
        Path to the cloned RTlogD repository (WangYitian123/RTlogD).
    weights_path:
        Path to the RTlogD pre-trained weights
        (final_model/RTlogD/model_pretrain_76.pth).
    logd_min, logd_max:
        Acceptable logD window for CNS penetration.
        Paper guidance: optimal membrane permeability at logD 1–3.
    device:
        Computation device.
    """

    def __init__(
        self,
        rtlogd_repo_path: Union[str, Path],
        weights_path: Union[str, Path],
        logd_min: float = 1.0,
        logd_max: float = 3.0,
        device=None,
    ) -> None:
        self._predictor = RTLogDPredictor(
            rtlogd_repo_path=rtlogd_repo_path,
            weights_path=weights_path,
            device=device,
        )
        self._logd_min = logd_min
        self._logd_max = logd_max

    def __call__(self, smiles: str) -> float:
        """Predict logD for a single molecule.

        Returns ``float('nan')`` for invalid SMILES.
        """
        return self._predictor.predict_single(smiles)

    def predict(self, smiles: Union[str, List[str]]) -> Union[float, "np.ndarray"]:
        """Predict logD for one or more SMILES strings.

        Parameters
        ----------
        smiles:
            A single SMILES string or a list of SMILES strings.

        Returns
        -------
        float (single) or np.ndarray of shape (n,) (batch).
        """
        if isinstance(smiles, str):
            return self._predictor.predict_single(smiles)
        return self._predictor.predict(smiles)

    def is_acceptable(self, smiles: str) -> bool:
        """Return True if the molecule's logD falls within the filter window."""
        logd = self._predictor.predict_single(smiles)
        if math.isnan(logd):
            return False
        return self._logd_min <= logd <= self._logd_max

    def filter_smiles(
        self, smiles_list: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Split a list of SMILES into (accepted, rejected) by logD.

        Returns
        -------
        accepted:
            SMILES with logD inside [logd_min, logd_max].
        rejected:
            SMILES outside the window or invalid.
        """
        preds = self._predictor.predict(smiles_list)
        accepted, rejected = [], []
        for smi, logd in zip(smiles_list, preds):
            if not math.isnan(float(logd)) and self._logd_min <= float(logd) <= self._logd_max:
                accepted.append(smi)
            else:
                rejected.append(smi)
        return accepted, rejected
