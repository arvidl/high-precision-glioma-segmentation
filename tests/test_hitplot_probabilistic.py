"""Unit tests for ``hpgs.hitplot.probabilistic_hitplot``.

segmentation variability). The probabilistic Hit-Plot is the second
member of the Hit-Plot family committed to in
``docs/scope_pr7.md`` § 1; it propagates per-class sigmoid
probabilities from the unified segmenter through the same
(region x compartment) overlap matrix that ``compartment_region_matrix``
produces, but in closed form (Poisson-binomial mean and variance) so
the numbers are bitwise-deterministic for a fixed input.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from hpgs.hitplot import compartment_region_matrix, probabilistic_hitplot
from hpgs.segment import LABEL_ED, LABEL_ET, LABEL_NCR


def _make_parc_and_lut(
    shape: tuple[int, int, int] = (10, 10, 10),
) -> tuple[np.ndarray, dict[int, str]]:
    parc = np.zeros(shape, dtype=np.int16)
    parc[:5, :, :] = 1
    parc[5:, :, :] = 2
    return parc, {1: "left", 2: "right"}


def _binary_to_brats3_probs(label_map: np.ndarray) -> np.ndarray:
    """Build a (3, X, Y, Z) hard-probability stack consistent with WT/TC/ET.

    Channel 0 = WT, 1 = TC, 2 = ET (matches the unified segmenter
    output convention).
    """
    wt = np.isin(label_map, (LABEL_NCR, LABEL_ED, LABEL_ET)).astype(np.float64)
    tc = np.isin(label_map, (LABEL_NCR, LABEL_ET)).astype(np.float64)
    et = (label_map == LABEL_ET).astype(np.float64)
    return np.stack([wt, tc, et], axis=0)


def test_probabilistic_schema_and_shape() -> None:
    parc, lut = _make_parc_and_lut()
    probs = np.full((3, *parc.shape), 0.3, dtype=np.float64)
    df = probabilistic_hitplot(probs, parc, lut)
    expected = {
        "label",
        "name",
        "compartment",
        "parcel_voxels",
        "overlap_voxels_mean",
        "overlap_voxels_std",
        "overlap_volume_mm3_mean",
        "overlap_volume_mm3_std",
        "pct_of_parcel_mean",
        "pct_of_parcel_std",
    }
    assert set(df.columns) == expected
    assert len(df) == 6  # 2 regions x 3 compartments
    assert set(df["compartment"].unique()) == {"WT", "TC", "ET"}


def test_probabilistic_uniform_p_matches_poisson_binomial_formula() -> None:
    parc, lut = _make_parc_and_lut()
    p = 0.3
    probs = np.full((3, *parc.shape), p, dtype=np.float64)
    df = probabilistic_hitplot(probs, parc, lut, voxel_volume_mm3=1.0).set_index(
        ["label", "compartment"]
    )
    # Each region has 5*10*10 = 500 voxels.
    n = 500
    expected_mean = n * p
    expected_var = n * p * (1.0 - p)
    expected_std = np.sqrt(expected_var)
    for lab in (1, 2):
        for comp in ("WT", "TC", "ET"):
            assert df.loc[(lab, comp), "overlap_voxels_mean"] == pytest.approx(expected_mean)
            assert df.loc[(lab, comp), "overlap_voxels_std"] == pytest.approx(expected_std)
            assert df.loc[(lab, comp), "pct_of_parcel_mean"] == pytest.approx(100.0 * p)
            assert df.loc[(lab, comp), "pct_of_parcel_std"] == pytest.approx(
                100.0 * expected_std / n
            )


def test_probabilistic_mean_equals_deterministic_for_hard_probs() -> None:
    """When p in {0, 1}, probabilistic-mean must equal deterministic overlap.

    This is the cross-check between ``probabilistic_hitplot`` and
    ``compartment_region_matrix``: the agreement panel relies on it.
    """
    shape = (10, 10, 10)
    parc, lut = _make_parc_and_lut(shape)
    label_map = np.zeros(shape, dtype=np.int16)
    label_map[:5, :5, :5] = LABEL_ET
    label_map[5:, :5, :5] = LABEL_NCR
    label_map[5:, 5:, :5] = LABEL_ED

    det = compartment_region_matrix(label_map, parc, lut).set_index(["label", "compartment"])
    probs = _binary_to_brats3_probs(label_map)
    prob_df = probabilistic_hitplot(probs, parc, lut).set_index(["label", "compartment"])

    for key in det.index:
        assert prob_df.loc[key, "overlap_voxels_mean"] == pytest.approx(
            det.loc[key, "overlap_voxels"]
        )
        # All probabilities are 0 or 1 -> Var = 0 -> std = 0.
        assert prob_df.loc[key, "overlap_voxels_std"] == pytest.approx(0.0)
        assert prob_df.loc[key, "pct_of_parcel_mean"] == pytest.approx(
            det.loc[key, "pct_of_parcel"]
        )


def test_probabilistic_p_half_maximises_variance() -> None:
    parc, lut = _make_parc_and_lut()
    n = 500  # voxels per region
    df_half = probabilistic_hitplot(np.full((3, *parc.shape), 0.5), parc, lut).set_index(
        ["label", "compartment"]
    )
    df_quarter = probabilistic_hitplot(np.full((3, *parc.shape), 0.25), parc, lut).set_index(
        ["label", "compartment"]
    )
    df_one = probabilistic_hitplot(np.ones((3, *parc.shape)), parc, lut).set_index(
        ["label", "compartment"]
    )
    # Variance of sum of indep Bernoullis is n*p*(1-p), maximised at p=0.5.
    assert df_half.loc[(1, "WT"), "overlap_voxels_std"] == pytest.approx(np.sqrt(n * 0.25))
    assert (
        df_quarter.loc[(1, "WT"), "overlap_voxels_std"]
        < df_half.loc[(1, "WT"), "overlap_voxels_std"]
    )
    # p == 1 is deterministic -> std = 0; mean = n.
    assert df_one.loc[(1, "WT"), "overlap_voxels_mean"] == pytest.approx(n)
    assert df_one.loc[(1, "WT"), "overlap_voxels_std"] == pytest.approx(0.0)


def test_probabilistic_compartment_axis_kwarg() -> None:
    """Accept either (C, X, Y, Z) or (X, Y, Z, C) layout."""
    parc, lut = _make_parc_and_lut()
    probs_cxyz = np.full((3, *parc.shape), 0.4)
    probs_xyzc = np.moveaxis(probs_cxyz, 0, -1)
    df_a = probabilistic_hitplot(probs_cxyz, parc, lut, compartment_axis=0)
    df_b = probabilistic_hitplot(probs_xyzc, parc, lut, compartment_axis=-1)
    pd.testing.assert_frame_equal(df_a, df_b)


def test_probabilistic_rejects_out_of_range_probabilities() -> None:
    parc, lut = _make_parc_and_lut()
    bad = np.full((3, *parc.shape), 0.5)
    bad[0, 0, 0, 0] = 1.5
    with pytest.raises(ValueError, match=r"probabilities must lie in \[0, 1\]"):
        probabilistic_hitplot(bad, parc, lut)


def test_probabilistic_rejects_channel_count_mismatch() -> None:
    parc, lut = _make_parc_and_lut()
    probs_two_channel = np.full((2, *parc.shape), 0.5)
    with pytest.raises(ValueError, match="2 channels"):
        probabilistic_hitplot(probs_two_channel, parc, lut, compartments=("WT", "TC", "ET"))


def test_probabilistic_rejects_spatial_shape_mismatch() -> None:
    parc, lut = _make_parc_and_lut()
    bad_probs = np.full((3, 10, 10, 11), 0.5)
    with pytest.raises(ValueError, match="spatial shape"):
        probabilistic_hitplot(bad_probs, parc, lut)


def test_probabilistic_rejects_unknown_compartment() -> None:
    parc, lut = _make_parc_and_lut()
    probs = np.full((1, *parc.shape), 0.5)
    with pytest.raises(ValueError, match="unknown compartment"):
        probabilistic_hitplot(probs, parc, lut, compartments=("FOO",))


def test_probabilistic_rejects_dimension_mismatch() -> None:
    parc, lut = _make_parc_and_lut()
    probs_3d = np.full(parc.shape, 0.5)  # missing compartment axis
    with pytest.raises(ValueError, match=r"probabilities.ndim"):
        probabilistic_hitplot(probs_3d, parc, lut)


def test_probabilistic_skips_lut_labels_absent_from_parcellation() -> None:
    parc, lut = _make_parc_and_lut()
    probs = np.full((3, *parc.shape), 0.5)
    lut_with_ghost = {**lut, 99: "ghost"}
    df = probabilistic_hitplot(probs, parc, lut_with_ghost)
    assert 99 not in set(df["label"])


def test_probabilistic_csv_round_trip_preserves_values() -> None:
    parc, lut = _make_parc_and_lut()
    probs = np.full((3, *parc.shape), 0.42)
    df = probabilistic_hitplot(probs, parc, lut)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    df2 = pd.read_csv(buf)
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True),
        df2.reset_index(drop=True),
        check_dtype=False,
    )


def test_probabilistic_voxel_volume_scales_volumes_only() -> None:
    parc, lut = _make_parc_and_lut()
    probs = np.full((3, *parc.shape), 0.3)
    a = probabilistic_hitplot(probs, parc, lut, voxel_volume_mm3=1.0)
    b = probabilistic_hitplot(probs, parc, lut, voxel_volume_mm3=2.5)
    # voxel counts and percentages should be identical;
    # volumes should scale by 2.5.
    pd.testing.assert_series_equal(a["overlap_voxels_mean"], b["overlap_voxels_mean"])
    pd.testing.assert_series_equal(a["pct_of_parcel_mean"], b["pct_of_parcel_mean"])
    np.testing.assert_allclose(b["overlap_volume_mm3_mean"], 2.5 * a["overlap_volume_mm3_mean"])
    np.testing.assert_allclose(b["overlap_volume_mm3_std"], 2.5 * a["overlap_volume_mm3_std"])
