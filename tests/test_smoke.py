"""Smoke tests for HPGS — fast, no real MRI required."""

from __future__ import annotations

import numpy as np
import pytest

from hpgs import SEED, __version__
from hpgs.metrics import (
    absolute_volumetric_error,
    dice,
    sensitivity,
    specificity,
    volumetric_error,
)
from hpgs.segment import LABEL_ED, LABEL_ET, LABEL_NCR, derive_subregions


def test_version_and_seed() -> None:
    assert isinstance(__version__, str) and len(__version__) > 0
    assert SEED == 20260415


def test_dice_perfect_and_disjoint() -> None:
    a = np.zeros((4, 4, 4), dtype=np.uint8)
    a[1:3, 1:3, 1:3] = 1
    assert dice(a, a) == pytest.approx(1.0)
    b = np.zeros_like(a)
    b[0, 0, 0] = 1
    assert dice(a, b) == 0.0


def test_sensitivity_specificity() -> None:
    gt = np.zeros((3, 3, 3), dtype=np.uint8)
    gt[0, 0, 0] = 1
    pred = np.zeros_like(gt)
    pred[0, 0, 0] = 1
    pred[2, 2, 2] = 1  # one false positive
    assert sensitivity(pred, gt) == pytest.approx(1.0)
    assert specificity(pred, gt) < 1.0


def test_volumetric_error_signs() -> None:
    gt = np.zeros((2, 2, 2), dtype=np.uint8)
    gt[0, 0, 0] = 1  # 1 voxel
    pred = np.ones_like(gt)  # 8 voxels
    assert volumetric_error(pred, gt, voxel_volume_mm3=1.0) == 7.0
    assert absolute_volumetric_error(pred, gt) == 7.0


def test_derive_subregions() -> None:
    lm = np.array([[[LABEL_NCR, LABEL_ED], [LABEL_ET, 0]]], dtype=np.uint8)
    sub = derive_subregions(lm)
    assert sub.wt.sum() == 3
    assert sub.tc.sum() == 2  # NCR + ET
    assert sub.et.sum() == 1


def test_hitplot_burden() -> None:
    # Lazy import: keeps the matplotlib/seaborn/plotly stack off the
    # pytest-collection path so unrelated tests still collect on a
    # bare-bones interpreter.
    from hpgs.hitplot import parcel_tumor_burden  # noqa: PLC0415

    parc = np.array([[[1, 1, 2], [2, 2, 3]]], dtype=np.int32)
    tum = np.array([[[1, 0, 1], [0, 0, 0]]], dtype=np.uint8)
    df = parcel_tumor_burden(parc, tum, {1: "A", 2: "B", 3: "C"})
    # parcel 1 has 2 vox / 1 tum → 50%; parcel 2 has 3 vox / 1 tum → 33.33%; parcel 3 0%
    assert set(df["name"]) == {"A", "B", "C"}
    a = df.set_index("name").loc["A"]
    assert a["pct_burden"] == pytest.approx(50.0)
