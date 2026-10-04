"""Unit tests for ``hpgs.hitplot.compartment_region_matrix``.

Hit-Plot v1 (matrix form) is the canonical PR-7 producer that writes
``data/derivatives_cohort50/sub-XXXX/hitplot/sub-XXXX_hitplot_*.csv``;
the agreement panel reads these CSVs and pivots them on
``(label, compartment)`` -> ``pct_of_parcel``.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
import pytest

from hpgs.hitplot import compartment_region_matrix
from hpgs.segment import LABEL_ED, LABEL_ET, LABEL_NCR


def _make_synthetic_pair() -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """Build a tiny synthetic (label_map, parcellation, lut) triple.

    Layout (10x10x10 volume), x runs along the first axis:
        parcellation:  region 1 fills x in [0, 5),
                       region 2 fills x in [5, 10);
                       so each region has 500 voxels.
        label_map:     ET (label 4) fills the (5x5x5) cube
                       x in [0, 5), y in [0, 5), z in [0, 5);
                       NCR (label 1) fills the (5x5x5) cube
                       x in [5, 10), y in [0, 5), z in [0, 5);
                       ED (label 2) fills the (5x5x5) cube
                       x in [5, 10), y in [5, 10), z in [0, 5).

    Therefore, by construction:
        WT in region 1 = ET cube (125 voxels);
        TC in region 1 = ET cube (125 voxels);
        ET in region 1 = ET cube (125 voxels);
        WT in region 2 = NCR cube + ED cube (250 voxels);
        TC in region 2 = NCR cube (125 voxels);
        ET in region 2 = 0 voxels.
        Compartment totals: WT=375, TC=250, ET=125.
    """
    shape = (10, 10, 10)
    parcellation = np.zeros(shape, dtype=np.int16)
    parcellation[:5, :, :] = 1
    parcellation[5:, :, :] = 2

    label_map = np.zeros(shape, dtype=np.int16)
    label_map[:5, :5, :5] = LABEL_ET
    label_map[5:, :5, :5] = LABEL_NCR
    label_map[5:, 5:, :5] = LABEL_ED

    lut = {1: "left", 2: "right"}
    return label_map, parcellation, lut


def test_matrix_long_format_schema_and_shape() -> None:
    label_map, parc, lut = _make_synthetic_pair()
    df = compartment_region_matrix(label_map, parc, lut)

    expected_cols = {
        "label",
        "name",
        "compartment",
        "parcel_voxels",
        "compartment_voxels",
        "overlap_voxels",
        "overlap_volume_mm3",
        "pct_of_parcel",
        "pct_of_compartment",
    }
    assert set(df.columns) == expected_cols
    # 2 regions x 3 compartments = 6 rows
    assert len(df) == 6
    assert set(df["compartment"].unique()) == {"WT", "TC", "ET"}


def test_matrix_known_overlap_counts() -> None:
    label_map, parc, lut = _make_synthetic_pair()
    df = compartment_region_matrix(label_map, parc, lut, voxel_volume_mm3=2.0)

    # Index by (label, compartment) for direct lookup.
    df_idx = df.set_index(["label", "compartment"])

    # Region totals (each axis-x slab has 5*10*10 = 500 voxels).
    for lab in (1, 2):
        for comp in ("WT", "TC", "ET"):
            assert int(df_idx.loc[(lab, comp), "parcel_voxels"]) == 500

    # Overlaps from the docstring construction.
    expected = {
        (1, "WT"): 125,
        (1, "TC"): 125,
        (1, "ET"): 125,
        (2, "WT"): 250,
        (2, "TC"): 125,
        (2, "ET"): 0,
    }
    for key, n in expected.items():
        assert int(df_idx.loc[key, "overlap_voxels"]) == n
        assert df_idx.loc[key, "overlap_volume_mm3"] == pytest.approx(n * 2.0)
        assert df_idx.loc[key, "pct_of_parcel"] == pytest.approx(100.0 * n / 500)

    # Compartment totals: WT=375, TC=250, ET=125.
    for comp, total in (("WT", 375), ("TC", 250), ("ET", 125)):
        for lab in (1, 2):
            assert int(df_idx.loc[(lab, comp), "compartment_voxels"]) == total


def test_matrix_pct_of_compartment_handles_empty_compartment() -> None:
    # Build a label map with no ET at all.
    label_map, parc, lut = _make_synthetic_pair()
    label_map[label_map == LABEL_ET] = LABEL_ED  # no ET voxels left
    df = compartment_region_matrix(label_map, parc, lut)
    et_rows = df[df["compartment"] == "ET"]
    assert len(et_rows) == 2
    # No ET voxels anywhere -> compartment_voxels == 0 -> pct_of_compartment NaN.
    assert et_rows["compartment_voxels"].eq(0).all()
    assert et_rows["pct_of_compartment"].isna().all()
    # But pct_of_parcel must be a finite 0.0, not NaN.
    assert (et_rows["pct_of_parcel"] == 0.0).all()


def test_matrix_skips_lut_labels_absent_from_parcellation() -> None:
    label_map, parc, lut = _make_synthetic_pair()
    lut_with_ghost = {**lut, 99: "ghost"}  # label 99 has zero voxels
    df = compartment_region_matrix(label_map, parc, lut_with_ghost)
    assert 99 not in set(df["label"])


def test_matrix_compartments_kwarg_subsets() -> None:
    label_map, parc, lut = _make_synthetic_pair()
    df = compartment_region_matrix(label_map, parc, lut, compartments=("WT",))
    assert set(df["compartment"].unique()) == {"WT"}
    assert len(df) == 2  # 2 regions x 1 compartment


def test_matrix_rejects_unknown_compartment() -> None:
    label_map, parc, lut = _make_synthetic_pair()
    with pytest.raises(ValueError, match="unknown compartment"):
        compartment_region_matrix(label_map, parc, lut, compartments=("WT", "FOO"))


def test_matrix_rejects_shape_mismatch() -> None:
    label_map, _, lut = _make_synthetic_pair()
    bad_parc = np.zeros((10, 10, 11), dtype=np.int16)
    with pytest.raises(ValueError, match="!= parcellation shape"):
        compartment_region_matrix(label_map, bad_parc, lut)


def test_matrix_rejects_empty_lut() -> None:
    label_map, parc, _ = _make_synthetic_pair()
    with pytest.raises(ValueError, match="empty label_lut"):
        compartment_region_matrix(label_map, parc, {})


def test_matrix_sorting_is_per_compartment_descending() -> None:
    label_map, parc, lut = _make_synthetic_pair()
    df = compartment_region_matrix(label_map, parc, lut)
    # Within each compartment, pct_of_parcel must be non-increasing.
    for _comp, sub in df.groupby("compartment"):
        pcts = sub["pct_of_parcel"].tolist()
        assert pcts == sorted(pcts, reverse=True)


def test_matrix_csv_round_trip_preserves_values() -> None:
    label_map, parc, lut = _make_synthetic_pair()
    df = compartment_region_matrix(label_map, parc, lut)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    df2 = pd.read_csv(buf)
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True),
        df2.reset_index(drop=True),
        check_dtype=False,
    )


def test_matrix_returns_empty_when_parcellation_is_all_background() -> None:
    label_map, _, lut = _make_synthetic_pair()
    parc = np.zeros_like(label_map)
    df = compartment_region_matrix(label_map, parc, lut)
    assert df.empty
    # An empty DataFrame with no rows still must not blow up downstream code.
    assert isinstance(df, pd.DataFrame)
    assert math.isnan(float("nan"))  # marker; pytest sees no failure
