"""Unit tests for ``hpgs.metrics.hausdorff95``.

HD95 is one of the three headline per-compartment metrics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hpgs.metrics import hausdorff95


def _filled_block(
    shape: tuple[int, int, int], lo: tuple[int, int, int], hi: tuple[int, int, int]
) -> np.ndarray:
    """Return a boolean mask with a solid axis-aligned block set to True."""
    a = np.zeros(shape, dtype=bool)
    a[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True
    return a


def test_hd95_identical_masks_is_zero() -> None:
    a = _filled_block((20, 20, 20), (5, 5, 5), (15, 15, 15))
    assert hausdorff95(a, a, voxel_spacing=(1.0, 1.0, 1.0)) == 0.0


def test_hd95_both_empty_is_nan() -> None:
    z = np.zeros((10, 10, 10), dtype=bool)
    assert math.isnan(hausdorff95(z, z, voxel_spacing=(1.0, 1.0, 1.0)))


def test_hd95_one_empty_is_inf() -> None:
    z = np.zeros((10, 10, 10), dtype=bool)
    a = _filled_block((10, 10, 10), (3, 3, 3), (7, 7, 7))
    assert math.isinf(hausdorff95(a, z, voxel_spacing=(1.0, 1.0, 1.0)))
    assert math.isinf(hausdorff95(z, a, voxel_spacing=(1.0, 1.0, 1.0)))


def test_hd95_translated_block_isotropic_spacing() -> None:
    # Two identical 10x10x10 blocks; the second is shifted by 3 voxels
    # along the x-axis. Surfaces are 10x10 sheets that are also shifted
    # by 3 voxels, so the directed surface-to-surface distance from
    # every point on one sheet to the nearest point on the other sheet
    # is exactly 3 mm under isotropic 1 mm spacing. The 95th percentile
    # of a constant sequence equals that constant.
    shape = (40, 40, 40)
    a = _filled_block(shape, (5, 5, 5), (15, 15, 15))
    b = _filled_block(shape, (8, 5, 5), (18, 15, 15))
    assert hausdorff95(a, b, voxel_spacing=(1.0, 1.0, 1.0)) == pytest.approx(3.0, abs=1e-9)


def test_hd95_anisotropic_spacing_scales_distance() -> None:
    # Same translation as above (3 voxels along x), but x-spacing is
    # 0.5 mm now: the answer should be 3 * 0.5 = 1.5 mm.
    shape = (40, 40, 40)
    a = _filled_block(shape, (5, 5, 5), (15, 15, 15))
    b = _filled_block(shape, (8, 5, 5), (18, 15, 15))
    assert hausdorff95(a, b, voxel_spacing=(0.5, 1.0, 1.0)) == pytest.approx(1.5, abs=1e-9)


def test_hd95_robust_to_outlier_voxel_at_p95() -> None:
    # Two identical blocks plus a single far-away "spurious" voxel in
    # one of them. The 95th percentile should NOT be dominated by that
    # one outlier (that's the whole point of HD95 vs. classic HD), so
    # the value should be small relative to the actual outlier distance.
    shape = (60, 30, 30)
    a = _filled_block(shape, (5, 5, 5), (15, 15, 15))
    b = a.copy()
    b[55, 28, 28] = True  # far-away spurious voxel
    hd95 = hausdorff95(a, b, voxel_spacing=(1.0, 1.0, 1.0))
    # The exact distance to the outlier is sqrt((55-14)^2 + (28-14)^2 + (28-14)^2)
    # ~ sqrt(1681+196+196) ~ 45.5 mm; HD95 must be << that because the
    # outlier accounts for a vanishing fraction of the surface voxels.
    outlier_distance = math.sqrt((55 - 14) ** 2 + (28 - 14) ** 2 + (28 - 14) ** 2)
    assert hd95 < 0.1 * outlier_distance


def test_hd95_classical_recovered_at_percentile_100() -> None:
    # With percentile=100 the metric reduces to the classical symmetric
    # Hausdorff distance, dominated by the worst outlier.
    shape = (60, 30, 30)
    a = _filled_block(shape, (5, 5, 5), (15, 15, 15))
    b = a.copy()
    b[55, 14, 14] = True
    hd_classical = hausdorff95(a, b, voxel_spacing=(1.0, 1.0, 1.0), percentile=100.0)
    # The single outlier sits at (55, 14, 14); nearest surface voxel of
    # `a` along x is x=14 (block runs from 5 to 14 inclusive), so the
    # exact distance is 55 - 14 = 41 mm.
    assert hd_classical == pytest.approx(41.0, abs=1e-9)


def test_hd95_rejects_shape_mismatch() -> None:
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros((10, 10, 11), dtype=bool)
    with pytest.raises(ValueError, match="shape mismatch"):
        hausdorff95(a, b, voxel_spacing=(1.0, 1.0, 1.0))


def test_hd95_rejects_voxel_spacing_arity_mismatch() -> None:
    a = np.zeros((10, 10, 10), dtype=bool)
    with pytest.raises(ValueError, match="voxel_spacing"):
        hausdorff95(a, a, voxel_spacing=(1.0, 1.0))  # type: ignore[arg-type]
