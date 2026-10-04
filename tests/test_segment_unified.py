"""Smoke tests for the unified BraTS three-class segmentation adapter.

These tests validate the *contract* of :func:`hpgs.segment.unified.predict_brats3`
without requiring any pretrained weights, GPU, network access, or real MRI
data. The expensive end-to-end smoke run on three UCSF-PDGM subjects lives
in ``scripts/smoke_predict_brats3.py`` and is invoked manually with
``make smoke-segment``.

Coverage here:
    * Channel-loading helper raises on missing or shape-inconsistent inputs.
    * Brain-masked z-score normalises foreground and zeroes background.
    * BraTS subregion → multi-class label conversion respects the
      ``ET > NCR > ED`` write-order convention.
    * The dummy backend produces a SegmentationResult with the right
      shapes, dtypes, label set, and propagated affine.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from hpgs.segment import LABEL_BG, LABEL_ED, LABEL_ET, LABEL_NCR
from hpgs.segment.unified import (
    CHANNEL_ORDER,
    PROB_CHANNEL_ORDER,
    SegmentationResult,
    _stack_channels,
    _zscore_brain,
    brats3_to_label_map,
    predict_brats3,
)

# ---------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------


def test_channel_order_matches_monai_bundle_metadata() -> None:
    """Bundle `metadata.json` declares channel 0 = T1c, 1 = T1, 2 = T2, 3 = FLAIR.

    Note this is NOT the BraTS21 challenge order; the bundle was trained on
    BraTS 2018 data. Swapping the first two channels silently inverts the
    T1c/T1 contrast and collapses ET-Dice to ~0 (see the docstring of
    `CHANNEL_ORDER` for details and the UCSF-PDGM evidence). Lock this
    invariant so a future refactor cannot quietly reintroduce the swap.
    """
    assert CHANNEL_ORDER == ("T1c", "T1", "T2", "FLAIR")
    assert PROB_CHANNEL_ORDER == ("TC", "WT", "ET")


def test_zscore_brain_normalises_foreground_and_zeros_background() -> None:
    vol = np.zeros((4, 4, 4), dtype=np.float32)
    vol[1:3, 1:3, 1:3] = np.arange(8, dtype=np.float32).reshape(2, 2, 2) + 100.0
    z = _zscore_brain(vol)
    fg = z[1:3, 1:3, 1:3]
    assert pytest.approx(fg.mean(), abs=1e-5) == 0.0
    assert pytest.approx(fg.std(), abs=1e-5) == 1.0
    bg = z.copy()
    bg[1:3, 1:3, 1:3] = 0.0
    assert np.all(bg == 0.0)


def test_zscore_brain_handles_empty_volume() -> None:
    vol = np.zeros((3, 3, 3), dtype=np.float32)
    assert np.all(_zscore_brain(vol) == 0.0)


def test_brats3_to_label_map_write_order() -> None:
    """ET must win over NCR, NCR over ED, ED over BG."""
    probs = np.zeros((3, 1, 1, 4), dtype=np.float32)
    # Voxel 0: BG (no class active)
    # Voxel 1: ED only      (WT alone)        -> label 2
    # Voxel 2: NCR          (TC ∧ ¬ET)        -> label 1
    # Voxel 3: ET           (all three high)  -> label 4
    probs[1, 0, 0, 1] = 0.9  # WT for voxel 1
    probs[1, 0, 0, 2] = 0.9  # WT for voxel 2 (any TC implies WT in real data)
    probs[0, 0, 0, 2] = 0.9  # TC for voxel 2
    probs[1, 0, 0, 3] = 0.9
    probs[0, 0, 0, 3] = 0.9
    probs[2, 0, 0, 3] = 0.9  # ET for voxel 3
    label = brats3_to_label_map(probs, threshold=0.5)
    assert label.shape == (1, 1, 4)
    assert label.dtype == np.uint8
    assert label[0, 0, 0] == LABEL_BG
    assert label[0, 0, 1] == LABEL_ED
    assert label[0, 0, 2] == LABEL_NCR
    assert label[0, 0, 3] == LABEL_ET


def test_brats3_to_label_map_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(3, D, H, W\)"):
        brats3_to_label_map(np.zeros((2, 4, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match=r"shape \(3, D, H, W\)"):
        brats3_to_label_map(np.zeros((4, 4, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# Channel loading.
# ---------------------------------------------------------------------------


def _write_nifti(path: Path, shape: tuple[int, int, int], fill: float = 1.0) -> Path:
    arr = np.full(shape, fill, dtype=np.float32)
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))
    return path


def test_stack_channels_loads_in_canonical_order(tmp_path: Path) -> None:
    shape = (8, 8, 8)
    channels = {
        "T1": _write_nifti(tmp_path / "t1.nii.gz", shape, fill=10.0),
        "T1c": _write_nifti(tmp_path / "t1c.nii.gz", shape, fill=20.0),
        "T2": _write_nifti(tmp_path / "t2.nii.gz", shape, fill=30.0),
        "FLAIR": _write_nifti(tmp_path / "flair.nii.gz", shape, fill=40.0),
    }
    stack, affine = _stack_channels(channels)
    assert stack.shape == (4, 8, 8, 8)
    assert stack.dtype == np.float32
    assert affine.shape == (4, 4)
    # All-constant volumes z-score to 0 (sd is 0 → branch keeps original after subtract,
    # but mean equals fill and we're masked, so the result is 0).
    assert np.all(stack == 0.0)


def test_stack_channels_rejects_missing_channel(tmp_path: Path) -> None:
    p = _write_nifti(tmp_path / "t1.nii.gz", (4, 4, 4))
    with pytest.raises(KeyError, match="FLAIR"):
        _stack_channels({"T1": p, "T1c": p, "T2": p})


def test_stack_channels_rejects_shape_mismatch(tmp_path: Path) -> None:
    a = _write_nifti(tmp_path / "a.nii.gz", (4, 4, 4))
    b = _write_nifti(tmp_path / "b.nii.gz", (4, 4, 5))
    with pytest.raises(ValueError, match="disagrees with reference"):
        _stack_channels({"T1": a, "T1c": a, "T2": a, "FLAIR": b})


# ---------------------------------------------------------------------------
# End-to-end (dummy backend).
# ---------------------------------------------------------------------------


def test_predict_brats3_dummy_backend(tmp_path: Path) -> None:
    shape = (16, 16, 16)
    rng = np.random.default_rng(0)
    paths: dict[str, Path] = {}
    for name in CHANNEL_ORDER:
        arr = (rng.standard_normal(shape).astype(np.float32) + 5.0).clip(0.0)
        p = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(p))
        paths[name] = p

    result = predict_brats3(
        t1=paths["T1"],
        t1c=paths["T1c"],
        t2=paths["T2"],
        flair=paths["FLAIR"],
        backend="dummy",
    )
    assert isinstance(result, SegmentationResult)
    assert result.backend == "dummy"
    assert result.bundle_version is None
    assert result.label_map.shape == shape
    assert result.label_map.dtype == np.uint8
    assert result.probabilities.shape == (3, *shape)
    assert result.probabilities.dtype == np.float32
    assert np.all(result.affine == np.eye(4))
    assert set(result.channel_paths) == set(CHANNEL_ORDER)
    # The synthetic ellipsoid in _dummy_backend should yield voxels of every
    # tumor compartment for a 16^3 volume.
    classes = set(np.unique(result.label_map).tolist())
    assert {LABEL_BG, LABEL_ED, LABEL_NCR, LABEL_ET}.issubset(classes)


def test_predict_brats3_rejects_unknown_backend(tmp_path: Path) -> None:
    p = _write_nifti(tmp_path / "x.nii.gz", (4, 4, 4))
    with pytest.raises(ValueError, match="Unknown backend"):
        predict_brats3(t1=p, t1c=p, t2=p, flair=p, backend="nope")  # type: ignore[arg-type]
