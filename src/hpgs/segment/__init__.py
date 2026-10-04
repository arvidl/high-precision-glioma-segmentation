"""Tumor segmentation: fastMONAI / nnU-Net v2 wrappers and the BraTS label scheme."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# BraTS label scheme (verified per-dataset before use).
LABEL_BG = 0
LABEL_NCR = 1  # necrotic / non-enhancing tumor core
LABEL_ED = 2  # peritumoral edema
LABEL_ET = 4  # enhancing tumor


@dataclass(frozen=True)
class TumorMasks:
    """Container for derived tumor sub-region binary masks."""

    wt: np.ndarray  # whole tumor    = {NCR, ED, ET}
    tc: np.ndarray  # tumor core     = {NCR, ET}
    et: np.ndarray  # enhancing      = {ET}


def derive_subregions(label_map: np.ndarray) -> TumorMasks:
    """Derive WT/TC/ET binary masks from a multi-class BraTS-style label map."""
    wt = np.isin(label_map, [LABEL_NCR, LABEL_ED, LABEL_ET]).astype(np.uint8)
    tc = np.isin(label_map, [LABEL_NCR, LABEL_ET]).astype(np.uint8)
    et = (label_map == LABEL_ET).astype(np.uint8)
    return TumorMasks(wt=wt, tc=tc, et=et)


def predict_tumor(image_paths: dict[str, str], engine: str = "nnunet_brats") -> np.ndarray:
    """Run a tumor-segmentation engine on a 4-channel multiparametric MRI input.

    Parameters
    ----------
    image_paths
        Mapping of channel name to file path. Expected keys:
        ``"T1"``, ``"T1c"``, ``"T2"``, ``"FLAIR"`` (all bias-corrected, native space).
    engine
        ``"nnunet_brats"`` | ``"raidionics"``. Cursor TODO: implement both backends.
    """
    raise NotImplementedError("predict_tumor TODO — engine dispatch + inference")
